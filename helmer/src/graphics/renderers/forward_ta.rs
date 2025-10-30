use crate::{
    graphics::renderer_common::{
        atmosphere::AtmosphereRenderer,
        common::{
            Aabb, CASCADE_SPLITS, CameraUniforms, CascadeUniform, EguiRenderData, FRAMES_IN_FLIGHT,
            LightData, Material, MaterialShaderData, Mesh, MeshLod, ModelPushConstant,
            NUM_CASCADES, PbrConstants, RenderData, RenderMessage, RenderTrait, ShaderConstants,
            ShadowPipeline, ShadowUniforms, SkyUniforms, Vertex,
        },
        error::RendererError,
    },
    provided::components::{Camera, LightType},
    runtime::asset_server::{AssetKind, MaterialGpuData},
};
use egui_wgpu::Renderer as EguiRenderer;
use glam::{Mat4, Quat, Vec3, Vec4Swizzles};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{info, warn};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

// --- CONSTANTS ---
const MAX_LIGHTS: usize = 248;
const MAX_TEXTURES_PER_TYPE: u32 = 256; // Reduced for lower-end devices
const DEFAULT_TEXTURE_RESOLUTION: u32 = 512; // Standard resolution for texture arrays
const MAX_MATERIALS: usize = 2048;
const SHADOW_MAP_RESOLUTION: u32 = 512;

/// Forward renderer optimized for lower-end devices
pub struct ForwardRendererTA {
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    window_size: PhysicalSize<u32>,

    // Maps an asset Handle ID to texture indices
    handle_id_to_texture_indices: HashMap<usize, TextureIndices>,

    // Materials waiting for textures
    pending_materials: Vec<MaterialGpuData>,

    // Core Pipelines
    forward_pipeline: Option<wgpu::RenderPipeline>,
    shadow_pipeline: Option<ShadowPipeline>,

    // Bind Group Layouts
    scene_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    material_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Bind Groups
    scene_data_bind_groups: Vec<wgpu::BindGroup>,
    material_bind_groups: HashMap<usize, wgpu::BindGroup>,

    // Depth
    depth_texture: Option<wgpu::Texture>,
    depth_texture_view: Option<wgpu::TextureView>,

    // Buffers
    camera_buffers: Vec<wgpu::Buffer>,
    lights_buffers: Vec<wgpu::Buffer>,
    material_buffers: HashMap<usize, wgpu::Buffer>,
    shadow_light_vp_buffer: Option<wgpu::Buffer>,

    // Asset Storage
    meshes: HashMap<usize, Mesh>,
    materials: HashMap<usize, Material>,

    // Texture Arrays (non-bindless for compatibility)
    albedo_texture_array: Option<wgpu::Texture>,
    normal_texture_array: Option<wgpu::Texture>,
    mr_texture_array: Option<wgpu::Texture>,

    albedo_array_view: Option<wgpu::TextureView>,
    normal_array_view: Option<wgpu::TextureView>,
    mr_array_view: Option<wgpu::TextureView>,

    next_texture_indices: TextureIndices,

    // Samplers
    texture_sampler: Option<wgpu::Sampler>,

    // Shadow Resources
    shadow_map_texture: Option<wgpu::Texture>,
    shadow_map_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    shadow_depth_texture: Option<wgpu::Texture>,
    shadow_depth_view: Option<wgpu::TextureView>,
    shadow_uniforms_buffer: Option<wgpu::Buffer>,
    cascade_views: Option<Vec<wgpu::TextureView>>,

    // atmosphere
    sky_uniforms_buffers: Vec<wgpu::Buffer>,
    atmosphere: Option<AtmosphereRenderer>,

    // render constants
    render_constants_bind_group_layout: Option<wgpu::BindGroupLayout>,
    render_constants_bind_group: Option<wgpu::BindGroup>,
    render_constants_buffer: Option<wgpu::Buffer>,

    // EGUI
    egui_renderer: EguiRenderer,
    current_egui_data: Option<EguiRenderData>,

    // State
    frame_index: usize,
    current_render_data: Option<RenderData>,
    logic_frame_duration: Duration,
    last_timestamp: Option<Instant>,
    prev_sky_uniforms: SkyUniforms,
    needs_atmosphere_precompute: bool,

    // Sky Resources
    sky_pipeline: Option<wgpu::RenderPipeline>,
    sky_bind_group_layout: Option<wgpu::BindGroupLayout>,
    sky_bind_groups: Vec<wgpu::BindGroup>,
    scene_sampler: Option<wgpu::Sampler>,
}

#[derive(Clone, Copy, Debug)]
struct TextureIndices {
    albedo: u32,
    normal: u32,
    metallic_roughness: u32,
}

impl Default for TextureIndices {
    fn default() -> Self {
        Self {
            albedo: 0,
            normal: 0,
            metallic_roughness: 0,
        }
    }
}

impl ForwardRendererTA {
    pub async fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        adapter: wgpu::Adapter,
        size: PhysicalSize<u32>,
        target_tickrate: f32,
    ) -> Result<Self, RendererError> {
        // --- Device, Queue ---
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: std::mem::size_of::<PbrConstants>() as u32,
                    max_texture_array_layers: MAX_TEXTURES_PER_TYPE,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .map_err(|e| {
                RendererError::ResourceCreation(format!("Failed to create device: {}", e))
            })?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let egui_renderer = EguiRenderer::new(
            &device,
            surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        let mut renderer = Self {
            adapter,
            instance,
            device: Arc::new(device),
            queue,
            surface,
            surface_config,
            window_size: size,
            handle_id_to_texture_indices: HashMap::new(),
            pending_materials: Vec::new(),
            forward_pipeline: None,
            shadow_pipeline: None,
            scene_data_bind_group_layout: None,
            material_bind_group_layout: None,
            scene_data_bind_groups: Vec::new(),
            material_bind_groups: HashMap::new(),
            depth_texture: None,
            depth_texture_view: None,
            camera_buffers: Vec::new(),
            lights_buffers: Vec::new(),
            material_buffers: HashMap::new(),
            shadow_light_vp_buffer: None,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            albedo_texture_array: None,
            normal_texture_array: None,
            mr_texture_array: None,
            albedo_array_view: None,
            normal_array_view: None,
            mr_array_view: None,
            next_texture_indices: TextureIndices::default(),
            texture_sampler: None,
            shadow_map_texture: None,
            shadow_map_view: None,
            shadow_sampler: None,
            shadow_depth_texture: None,
            shadow_depth_view: None,
            shadow_uniforms_buffer: None,
            cascade_views: None,
            sky_uniforms_buffers: Vec::new(),
            atmosphere: None,
            render_constants_bind_group_layout: None,
            render_constants_bind_group: None,
            render_constants_buffer: None,
            egui_renderer,
            current_egui_data: None,
            frame_index: 0,
            current_render_data: None,
            logic_frame_duration: Duration::from_secs_f32(1.0 / target_tickrate),
            last_timestamp: None,
            prev_sky_uniforms: SkyUniforms::default(),
            needs_atmosphere_precompute: true,
            sky_pipeline: None,
            sky_bind_group_layout: None,
            sky_bind_groups: Vec::new(),
            scene_sampler: None,
        };

        renderer.initialize_resources()?;
        Ok(renderer)
    }

    fn initialize_resources(&mut self) -> Result<(), RendererError> {
        let device = &self.device;

        // --- Create Buffers ---
        self.camera_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("camera-uniforms-{}", i)),
                    size: std::mem::size_of::<CameraUniforms>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        self.lights_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("lights-buffer-{}", i)),
                    size: (std::mem::size_of::<LightData>() * MAX_LIGHTS) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        self.render_constants_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Constants Uniform Buffer"),
            size: std::mem::size_of::<ShaderConstants>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.queue.write_buffer(
            self.render_constants_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&ShaderConstants::default()),
        );

        // --- Create Samplers ---
        self.texture_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.scene_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Scene Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        // --- Render constants resources ---
        self.render_constants_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Constants Bind Group Layout"),
                entries: &[buffer_binding(
                    0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Uniform,
                )],
            },
        ));

        // --- Sky/Atmosphere resource creation ---
        self.sky_uniforms_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("sky-uniforms-{}", i)),
                    size: std::mem::size_of::<SkyUniforms>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        self.atmosphere = Some(AtmosphereRenderer::new(&self.device));

        self.create_texture_arrays();
        self.upload_default_textures();
        self.create_shadow_resources();
        self.create_pipelines_and_layouts();

        self.resize(self.window_size);

        info!("initialized forward (TA) renderer");
        Ok(())
    }

    fn create_texture_arrays(&mut self) {
        let device = &self.device;
        let size = wgpu::Extent3d {
            width: DEFAULT_TEXTURE_RESOLUTION,
            height: DEFAULT_TEXTURE_RESOLUTION,
            depth_or_array_layers: MAX_TEXTURES_PER_TYPE,
        };

        // Albedo array
        self.albedo_texture_array = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Albedo Texture Array"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        self.albedo_array_view = Some(self.albedo_texture_array.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            },
        ));

        // Normal array
        self.normal_texture_array = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normal Texture Array"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        self.normal_array_view = Some(self.normal_texture_array.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            },
        ));

        // Metallic-Roughness array
        self.mr_texture_array = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MR Texture Array"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        self.mr_array_view = Some(self.mr_texture_array.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            },
        ));
    }

    fn upload_default_textures(&mut self) {
        let default_size = (DEFAULT_TEXTURE_RESOLUTION * DEFAULT_TEXTURE_RESOLUTION) as usize;
        let default_width = DEFAULT_TEXTURE_RESOLUTION;
        let default_height = DEFAULT_TEXTURE_RESOLUTION;

        // White albedo (RGBA8)
        let white_pixel = [255u8, 255, 255, 255];
        let white_data: Vec<u8> = white_pixel
            .iter()
            .cycle()
            .take(default_size * 4) // 4 bytes per pixel
            .copied()
            .collect();

        self.upload_texture_to_array(
            &white_data,
            self.albedo_texture_array.as_ref().unwrap(),
            0, // layer
            default_width,
            default_height,
        );

        // Flat normal (RGBA8)
        // Corresponds to a tangent-space vector of (0, 0, 1)
        let flat_normal_pixel = [128u8, 128, 255, 255];
        let flat_normal_data: Vec<u8> = flat_normal_pixel
            .iter()
            .cycle()
            .take(default_size * 4)
            .copied()
            .collect();

        self.upload_texture_to_array(
            &flat_normal_data,
            self.normal_texture_array.as_ref().unwrap(),
            0, // layer
            default_width,
            default_height,
        );

        // Default Metallic/Roughness
        // We'll use AO = 1.0, Roughness = 1.0, Metallic = 0.0
        // This maps to R=255, G=255, B=0 in the texture.
        let default_mr_pixel = [255u8, 255, 0, 255];
        let default_mr_data: Vec<u8> = default_mr_pixel
            .iter()
            .cycle()
            .take(default_size * 4)
            .copied()
            .collect();

        self.upload_texture_to_array(
            &default_mr_data,
            self.mr_texture_array.as_ref().unwrap(),
            0, // layer
            default_width,
            default_height,
        );

        // Start indices for new textures at 1, since 0 is now the default.
        self.next_texture_indices = TextureIndices {
            albedo: 1,
            normal: 1,
            metallic_roughness: 1,
        };
    }

    fn upload_texture_to_array(
        &self,
        data: &[u8],
        target_array: &wgpu::Texture,
        layer: u32,
        width: u32,
        height: u32,
    ) {
        // Silencing a likely erroneous/version-specific warning from the toolchain.
        #[allow(deprecated)]
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target_array,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn create_shadow_resources(&mut self) {
        let device = &self.device;

        let shadow_texture_desc = wgpu::TextureDescriptor {
            label: Some("VSM Shadow Map Texture Array"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_RESOLUTION,
                height: SHADOW_MAP_RESOLUTION,
                depth_or_array_layers: NUM_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        self.shadow_map_texture = Some(device.create_texture(&shadow_texture_desc));
        self.shadow_map_view = Some(self.shadow_map_texture.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("VSM Shadow Map View"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            },
        ));

        self.shadow_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        let shadow_depth_desc = wgpu::TextureDescriptor {
            label: Some("Shadow Pass Depth Texture"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_RESOLUTION,
                height: SHADOW_MAP_RESOLUTION,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        self.shadow_depth_texture = Some(device.create_texture(&shadow_depth_desc));
        self.shadow_depth_view = Some(
            self.shadow_depth_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.shadow_uniforms_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Uniforms Buffer"),
            size: std::mem::size_of::<ShadowUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let mut cascade_views = Vec::with_capacity(NUM_CASCADES);
        for i in 0..NUM_CASCADES {
            let cascade_view = self.shadow_map_texture.as_ref().unwrap().create_view(
                &wgpu::TextureViewDescriptor {
                    label: Some(&format!("Cascade View {}", i)),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                },
            );
            cascade_views.push(cascade_view);
        }
        self.cascade_views = Some(cascade_views);

        self.create_shadow_pipeline();
    }

    fn create_shadow_pipeline(&mut self) {
        let device = &self.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shadow.wgsl"));

        let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
        let aligned_mat4_size = wgpu::util::align_to(mat4_size, alignment);

        self.shadow_light_vp_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Light VP (Dynamic Uniform)"),
            size: NUM_CASCADES as wgpu::BufferAddress * aligned_mat4_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(mat4_size),
                },
                count: None,
            }],
        });

        let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Light VP Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.shadow_light_vp_buffer.as_ref().unwrap(),
                    offset: 0,
                    size: wgpu::BufferSize::new(mat4_size as u64),
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layout,
                self.render_constants_bind_group_layout.as_ref().unwrap(),
            ],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..std::mem::size_of::<ModelPushConstant>() as u32,
            }],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.shadow_pipeline = Some(ShadowPipeline {
            pipeline,
            bind_group,
        });
    }

    fn create_pipelines_and_layouts(&mut self) {
        let device = &self.device;

        // Load shader
        let forward_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/forward_ta.wgsl"));

        // Scene data layout
        self.scene_data_bind_group_layout = Some(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Scene Data Bind Group Layout"),
                entries: &[
                    // Camera uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Lights buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Shadow map
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Shadow sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Shadow uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Material layout
        self.material_bind_group_layout = Some(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Material Bind Group Layout"),
                entries: &[
                    // Material data
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Albedo texture array
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Normal texture array
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // MR texture array
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            },
        ));

        self.sky_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Sky Scene BGL"),
                entries: &[
                    buffer_binding(
                        0,
                        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                    buffer_binding(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward Pipeline Layout"),
            bind_group_layouts: &[
                self.scene_data_bind_group_layout.as_ref().unwrap(),
                self.material_bind_group_layout.as_ref().unwrap(),
                &self.atmosphere.as_ref().unwrap().sampling_bind_group_layout,
            ],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..std::mem::size_of::<ModelPushConstant>() as u32,
            }],
        });

        self.forward_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Forward Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &forward_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &forward_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));

        self.create_sky_pipeline();
        self.create_scene_bind_groups();
    }

    fn create_sky_pipeline(&mut self) {
        let shader = self
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/sky_sampled.wgsl"));
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sky Pipeline Layout"),
                bind_group_layouts: &[
                    self.sky_bind_group_layout.as_ref().unwrap(),
                    &self.atmosphere.as_ref().unwrap().sampling_bind_group_layout,
                    self.render_constants_bind_group_layout.as_ref().unwrap(),
                ],
                push_constant_ranges: &[],
            });
        self.sky_pipeline = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Sky Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));
    }

    fn create_sky_bind_groups(&mut self) {
        let depth_view = self.depth_texture_view.as_ref().unwrap();
        self.sky_bind_groups = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Sky Scene BG {}", i)),
                    layout: self.sky_bind_group_layout.as_ref().unwrap(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.camera_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.sky_uniforms_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                self.scene_sampler.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                    ],
                })
            })
            .collect();
    }

    fn create_scene_bind_groups(&mut self) {
        let device = &self.device;

        self.scene_data_bind_groups = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.scene_data_bind_group_layout.as_ref().unwrap(),
                    label: Some(&format!("Scene Data Bind Group {}", i)),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.camera_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.lights_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                self.shadow_map_view.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(
                                self.shadow_sampler.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self
                                .shadow_uniforms_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.sky_uniforms_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self
                                .render_constants_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        self.render_constants_bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Constants Bind Group"),
                layout: self.render_constants_bind_group_layout.as_ref().unwrap(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .render_constants_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                }],
            }),
        );
    }

    fn create_depth_texture(&mut self) {
        let device = &self.device;
        let size = wgpu::Extent3d {
            width: self.surface_config.width,
            height: self.surface_config.height,
            depth_or_array_layers: 1,
        };

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.depth_texture_view = Some(depth_texture.create_view(&Default::default()));
        self.depth_texture = Some(depth_texture);
    }

    pub fn add_texture(
        &mut self,
        data: &[u8],
        kind: AssetKind,
        _format: wgpu::TextureFormat, // This is unused for now but kept for API stability
        width: u32,
        height: u32,
    ) -> Result<u32, RendererError> {
        // This version fixes the borrow error without adding new dependencies.
        // It assumes the `data` slice contains raw, uncompressed RGBA8 pixel data.

        // Step 1: Get the target array and the next available index. This mutable
        // borrow is contained entirely within this match block.
        let (target_array, index) = match kind {
            AssetKind::Albedo => {
                let current_index = self.next_texture_indices.albedo;
                if current_index >= MAX_TEXTURES_PER_TYPE {
                    return Err(RendererError::ResourceCreation(
                        "Albedo texture array full".into(),
                    ));
                }
                self.next_texture_indices.albedo += 1;
                (self.albedo_texture_array.as_ref().unwrap(), current_index)
            }
            AssetKind::Normal => {
                let current_index = self.next_texture_indices.normal;
                if current_index >= MAX_TEXTURES_PER_TYPE {
                    return Err(RendererError::ResourceCreation(
                        "Normal texture array full".into(),
                    ));
                }
                self.next_texture_indices.normal += 1;
                (self.normal_texture_array.as_ref().unwrap(), current_index)
            }
            AssetKind::MetallicRoughness => {
                let current_index = self.next_texture_indices.metallic_roughness;
                if current_index >= MAX_TEXTURES_PER_TYPE {
                    return Err(RendererError::ResourceCreation(
                        "Metallic-Roughness texture array full".into(),
                    ));
                }
                self.next_texture_indices.metallic_roughness += 1;
                (self.mr_texture_array.as_ref().unwrap(), current_index)
            }
            _ => {
                return Err(RendererError::ResourceCreation(
                    "Unsupported texture type".into(),
                ));
            }
        };

        // Step 2: Upload the raw texture data. This is an immutable borrow of `self`.
        // Because the mutable borrow from Step 1 is finished, there is no conflict.
        // Note: We upload the texture with its original width and height into a
        // texture array layer that has a fixed size (DEFAULT_TEXTURE_RESOLUTION).
        // The GPU will handle how this is sampled (usually by treating the texture
        // coordinates as normalized from 0.0 to 1.0 across the uploaded portion).
        self.upload_texture_to_array(data, target_array, index, width, height);

        Ok(index)
    }

    pub fn add_mesh(
        &mut self,
        id: usize,
        vertices: &[Vertex],
        lod_indices: &[Vec<u32>],
        bounds: Aabb,
    ) -> Result<(), RendererError> {
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh-vbo-{}", id)),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let mut gpu_lods = Vec::new();
        for (lod_level, indices) in lod_indices.iter().enumerate() {
            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-ibo-{}-lod{}", id, lod_level)),
                    contents: bytemuck::cast_slice(indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            gpu_lods.push(MeshLod {
                index_buffer,
                index_count: indices.len() as u32,
            });
        }

        self.meshes.insert(
            id,
            Mesh {
                vertex_buffer,
                lods: gpu_lods,
                bounds,
            },
        );
        Ok(())
    }

    pub fn add_material(&mut self, id: usize, material: Material) -> Result<(), RendererError> {
        if id >= MAX_MATERIALS {
            return Err(RendererError::ResourceCreation(format!(
                "Material ID {} exceeds maximum of {}",
                id, MAX_MATERIALS
            )));
        }

        // Create material uniform buffer
        let shader_data = MaterialShaderData::from(&material);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Material Buffer {}", id)),
                contents: bytemuck::bytes_of(&shader_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        self.material_buffers.insert(id, buffer);

        // Create bind group for this material
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: self.material_bind_group_layout.as_ref().unwrap(),
            label: Some(&format!("Material Bind Group {}", id)),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.material_buffers[&id].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        self.albedo_array_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.normal_array_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        self.mr_array_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(
                        self.texture_sampler.as_ref().unwrap(),
                    ),
                },
            ],
        });

        self.material_bind_groups.insert(id, bind_group);
        self.materials.insert(id, material);
        Ok(())
    }

    fn update_uniforms(&self, render_data: &RenderData, alpha: f32) {
        let camera = &render_data.camera_component;
        let eye = render_data
            .previous_camera_transform
            .position
            .lerp(render_data.current_camera_transform.position, alpha);
        let rotation = Quat::from(render_data.previous_camera_transform.rotation)
            .slerp(render_data.current_camera_transform.rotation, alpha);
        let forward = rotation * Vec3::Z;
        let up = rotation * Vec3::Y;
        let view_matrix = Mat4::look_at_rh(eye, eye + forward, up);
        let projection_matrix = Mat4::perspective_infinite_reverse_rh(
            camera.fov_y_rad,
            camera.aspect_ratio,
            camera.near_plane,
        );

        let inv_proj = projection_matrix.inverse();
        let inv_view_proj = (projection_matrix * view_matrix).inverse();

        let camera_uniforms = CameraUniforms {
            view_matrix: view_matrix.to_cols_array_2d(),
            projection_matrix: projection_matrix.to_cols_array_2d(),
            inverse_projection_matrix: inv_proj.to_cols_array_2d(),
            inverse_view_projection_matrix: inv_view_proj.to_cols_array_2d(),
            view_position: eye.to_array(),
            light_count: render_data.lights.len() as u32,
            prev_view_proj: [[0.0; 4]; 4], // PLACEHOLDER
            frame_index: 0,                // PLACEHOLDER,
            _padding: [0; 3],
        };
        self.queue.write_buffer(
            &self.camera_buffers[self.frame_index],
            0,
            bytemuck::bytes_of(&camera_uniforms),
        );

        let light_data: Vec<LightData> = render_data
            .lights
            .iter()
            .take(MAX_LIGHTS)
            .map(|light| {
                let position = light
                    .previous_transform
                    .position
                    .lerp(light.current_transform.position, alpha);
                let rotation = Quat::from(light.previous_transform.rotation)
                    .slerp(light.current_transform.rotation, alpha);
                let direction = (rotation * -Vec3::Z).normalize_or_zero();
                LightData {
                    position: position.into(),
                    light_type: match light.light_type {
                        LightType::Directional => 0,
                        LightType::Point => 1,
                        LightType::Spot { .. } => 2,
                    },
                    color: light.color,
                    intensity: light.intensity,
                    direction: direction.into(),
                    _padding: 0.0,
                }
            })
            .collect();
        self.queue.write_buffer(
            &self.lights_buffers[self.frame_index],
            0,
            bytemuck::cast_slice(&light_data),
        );
    }

    fn run_shadow_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_data: &RenderData,
        static_camera_view: &Mat4,
        alpha: f32,
    ) {
        const FAR_CASCADE_DISTANCE: f32 = 500.0;
        const MAX_SHADOW_CASTING_DISTANCE: f32 = FAR_CASCADE_DISTANCE * 1.5;
        const MAX_SHADOW_CASTING_DISTANCE_SQ: f32 =
            MAX_SHADOW_CASTING_DISTANCE * MAX_SHADOW_CASTING_DISTANCE;

        let shadow_light = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional));

        if let (Some(light), Some(shadow_pipeline)) = (shadow_light, self.shadow_pipeline.as_ref())
        {
            let camera_pos = render_data.current_camera_transform.position;

            let mut culled_objects = HashMap::new();
            for object in &render_data.objects {
                if object.casts_shadow {
                    let distance_sq = object
                        .current_transform
                        .position
                        .distance_squared(camera_pos);
                    let is_culled = distance_sq > MAX_SHADOW_CASTING_DISTANCE_SQ;
                    culled_objects.insert(object.id, is_culled);
                }
            }

            let mut scene_bounds_min = Vec3::splat(f32::MAX);
            let mut scene_bounds_max = Vec3::splat(f32::MIN);
            for object in &render_data.objects {
                if object.casts_shadow {
                    if *culled_objects.get(&object.id).unwrap_or(&true) {
                        continue;
                    }

                    if let Some(mesh) = self.meshes.get(&object.mesh_id) {
                        let model_matrix = Mat4::from_scale_rotation_translation(
                            object.current_transform.scale,
                            object.current_transform.rotation,
                            object.current_transform.position,
                        );
                        for &corner in &mesh.bounds.get_corners() {
                            let world_corner = (model_matrix * corner.extend(1.0)).xyz();
                            scene_bounds_min = scene_bounds_min.min(world_corner);
                            scene_bounds_max = scene_bounds_max.max(world_corner);
                        }
                    }
                }
            }

            let dynamic_scene_bounds = Aabb {
                min: scene_bounds_min,
                max: scene_bounds_max,
            };

            let camera = &render_data.camera_component;
            let shadow_uniforms = self.calculate_cascades(
                camera,
                static_camera_view,
                light.current_transform.rotation,
                &dynamic_scene_bounds,
            );

            self.queue.write_buffer(
                self.shadow_uniforms_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&shadow_uniforms),
            );

            let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress;
            let alignment =
                self.device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
            let aligned_mat4_size = wgpu::util::align_to(mat4_size, alignment);

            for i in 0..NUM_CASCADES {
                self.queue.write_buffer(
                    self.shadow_light_vp_buffer.as_ref().unwrap(),
                    (i as wgpu::BufferAddress) * aligned_mat4_size,
                    bytemuck::bytes_of(&shadow_uniforms.cascades[i].light_view_proj),
                );
            }

            for i in 0..NUM_CASCADES {
                let offset = (i as u32) * (aligned_mat4_size as u32);
                let cascade_view = &self.cascade_views.as_ref().unwrap()[i];

                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("Shadow Pass {}", i)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &cascade_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 1.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.shadow_depth_view.as_ref().unwrap(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                shadow_pass.set_pipeline(&shadow_pipeline.pipeline);
                shadow_pass.set_bind_group(0, &shadow_pipeline.bind_group, &[offset]);
                shadow_pass.set_bind_group(
                    1,
                    self.render_constants_bind_group.as_ref().unwrap(),
                    &[],
                );

                for object in &render_data.objects {
                    if object.casts_shadow {
                        if *culled_objects.get(&object.id).unwrap_or(&true) {
                            continue;
                        }

                        if let Some(mesh) = self.meshes.get(&object.mesh_id) {
                            if mesh.lods.is_empty() {
                                continue;
                            }

                            // Ensure the lod_index is valid to prevent a panic
                            let lod_index = object.lod_index.min(mesh.lods.len() - 1);
                            let lod = &mesh.lods[lod_index];

                            let position = object
                                .previous_transform
                                .position
                                .lerp(object.current_transform.position, alpha);
                            let rotation = Quat::from(object.previous_transform.rotation)
                                .slerp(object.current_transform.rotation, alpha);
                            let scale = object
                                .previous_transform
                                .scale
                                .lerp(object.current_transform.scale, alpha);

                            let model_matrix =
                                Mat4::from_scale_rotation_translation(scale, rotation, position);
                            let push_constants = ModelPushConstant {
                                model_matrix: model_matrix.to_cols_array_2d(),
                            };
                            shadow_pass.set_push_constants(
                                wgpu::ShaderStages::VERTEX,
                                0,
                                bytemuck::bytes_of(&push_constants),
                            );
                            shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            shadow_pass.set_index_buffer(
                                lod.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..lod.index_count, 0, 0..1);
                        }
                    }
                }
            }
        }
    }

    fn calculate_cascades(
        &self,
        camera: &Camera,
        camera_view: &Mat4,
        light_rotation: Quat,
        scene_bounds: &Aabb,
    ) -> ShadowUniforms {
        let light_dir = (light_rotation * -Vec3::Z).normalize();
        let inv_camera_view = camera_view.inverse();
        let tan_half_fovy = (camera.fov_y_rad / 2.0).tan();
        let scene_corners = scene_bounds.get_corners();

        let mut uniforms = ShadowUniforms {
            cascades: [CascadeUniform::default(); NUM_CASCADES],
        };

        for i in 0..NUM_CASCADES {
            let z_near = CASCADE_SPLITS[i];
            let z_far = CASCADE_SPLITS[i + 1];

            let h_near = 2.0 * tan_half_fovy * z_near;
            let w_near = h_near * camera.aspect_ratio;
            let h_far = 2.0 * tan_half_fovy * z_far;
            let w_far = h_far * camera.aspect_ratio;

            let corners_view = [
                Vec3::new(w_near / 2.0, h_near / 2.0, -z_near),
                Vec3::new(-w_near / 2.0, h_near / 2.0, -z_near),
                Vec3::new(w_near / 2.0, -h_near / 2.0, -z_near),
                Vec3::new(-w_near / 2.0, -h_near / 2.0, -z_near),
                Vec3::new(w_far / 2.0, h_far / 2.0, -z_far),
                Vec3::new(-w_far / 2.0, h_far / 2.0, -z_far),
                Vec3::new(w_far / 2.0, -h_far / 2.0, -z_far),
                Vec3::new(-w_far / 2.0, -h_far / 2.0, -z_far),
            ];

            let frustum_corners_world: [Vec3; 8] =
                std::array::from_fn(|i| (inv_camera_view * corners_view[i].extend(1.0)).xyz());

            let world_center = frustum_corners_world.iter().sum::<Vec3>() / 8.0;
            let light_view = Mat4::look_at_rh(world_center - light_dir, world_center, Vec3::Y);

            let mut cascade_min = Vec3::splat(f32::MAX);
            let mut cascade_max = Vec3::splat(f32::MIN);
            for corner in frustum_corners_world {
                let trf = light_view * corner.extend(1.0);
                cascade_min = cascade_min.min(trf.xyz());
                cascade_max = cascade_max.max(trf.xyz());
            }

            let mut scene_min_z = f32::MAX;
            let mut scene_max_z = f32::MIN;
            for corner in &scene_corners {
                let trf = light_view * corner.extend(1.0);
                scene_min_z = scene_min_z.min(trf.z);
                scene_max_z = scene_max_z.max(trf.z);
            }

            let light_proj = Mat4::orthographic_rh(
                cascade_min.x,
                cascade_max.x,
                cascade_min.y,
                cascade_max.y,
                -scene_max_z,
                -scene_min_z,
            );

            let final_light_vp = light_proj * light_view;
            uniforms.cascades[i] = CascadeUniform {
                light_view_proj: final_light_vp.to_cols_array_2d(),
                split_depth: [-z_far, 0.0, 0.0, 0.0],
            };
        }

        uniforms
    }
}

impl RenderTrait for ForwardRendererTA {
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);

            self.create_depth_texture();
            self.create_sky_bind_groups();
        }
    }

    fn render(&mut self) -> Result<(), RendererError> {
        let Some(ref render_data) = self.current_render_data.as_ref() else {
            return Ok(());
        };

        let now = Instant::now();
        let actual_logic_duration =
            self.last_timestamp
                .map_or(self.logic_frame_duration, |last_ts| {
                    let duration = render_data.timestamp.saturating_duration_since(last_ts);
                    if duration > Duration::from_millis(200) {
                        self.logic_frame_duration
                    } else {
                        duration
                    }
                });
        let time_since_current = now.saturating_duration_since(render_data.timestamp);
        let alpha = (time_since_current.as_secs_f32() / actual_logic_duration.as_secs_f32())
            .clamp(0.0, 1.0);
        self.last_timestamp = Some(render_data.timestamp);

        let output_frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost) => {
                self.resize(self.window_size);
                return Ok(());
            }
            Err(e) => {
                return Err(RendererError::ResourceCreation(format!(
                    "Surface error: {}",
                    e
                )));
            }
        };

        let output_view = output_frame.texture.create_view(&Default::default());

        self.update_uniforms(render_data, alpha);

        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;

        let directional_light = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional));

        let (sun_dir, sun_color, sun_intensity) = if let Some(light) = directional_light {
            (
                (light.current_transform.rotation * Vec3::Z).normalize_or_zero(),
                light.color,
                light.intensity,
            )
        } else {
            (
                // Default sun
                Vec3::new(0.2, 0.8, 0.1).normalize(),
                Vec3::ONE.to_array(),
                100.0,
            )
        };

        let sky_uniforms = SkyUniforms {
            sun_direction: sun_dir.to_array(),
            _padding: 0.0,
            sun_color: sun_color,
            sun_intensity,
            ground_albedo: [0.3, 0.25, 0.2], // Brownish ground
            ground_brightness: 1.0,
        };

        self.queue.write_buffer(
            &self.sky_uniforms_buffers[buffer_index],
            0,
            bytemuck::bytes_of(&sky_uniforms),
        );

        if sky_uniforms != self.prev_sky_uniforms {
            self.prev_sky_uniforms = sky_uniforms;

            self.needs_atmosphere_precompute = true;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Command Encoder"),
            });

        // --- ATMOSPHERE PRECOMPUTATION PASS ---
        if self.needs_atmosphere_precompute {
            self.needs_atmosphere_precompute = false;

            if let (Some(atmo), Some(data)) = (self.atmosphere.as_ref(), &self.current_render_data)
            {
                atmo.precompute(
                    &mut encoder,
                    &self.queue,
                    &sky_uniforms,
                    &data.render_config.shader_constants,
                );
            }
        }

        let camera_transform = &render_data.current_camera_transform;
        let eye = camera_transform.position;
        let forward = camera_transform.forward();
        let up = camera_transform.up();

        if render_data.render_config.shadow_pass {
            // Shadow pass
            let static_camera_view = Mat4::look_at_rh(eye, eye + forward, up);

            self.run_shadow_pass(&mut encoder, render_data, &static_camera_view, alpha);
        }

        // Forward rendering pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Forward Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.depth_texture_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        render_pass.set_pipeline(self.forward_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, &self.scene_data_bind_groups[self.frame_index], &[]);
        render_pass.set_bind_group(
            2,
            &self.atmosphere.as_ref().unwrap().sampling_bind_group,
            &[],
        );

        // Sort objects by material for better batching
        let mut sorted_objects = render_data.objects.clone();
        sorted_objects.sort_by_key(|obj| obj.material_id);

        let mut current_material_id = None;

        for object in &sorted_objects {
            if let (Some(mesh), true) = (
                self.meshes.get(&object.mesh_id),
                self.material_bind_groups.contains_key(&object.material_id),
            ) {
                if mesh.lods.is_empty() {
                    continue;
                }

                // Ensure the lod_index is valid to prevent a panic
                let lod_index = object.lod_index.min(mesh.lods.len() - 1);
                let lod = &mesh.lods[lod_index];

                // Set material bind group if changed
                if current_material_id != Some(object.material_id) {
                    render_pass.set_bind_group(
                        1,
                        &self.material_bind_groups[&object.material_id],
                        &[],
                    );
                    current_material_id = Some(object.material_id);
                }

                let position = object
                    .previous_transform
                    .position
                    .lerp(object.current_transform.position, alpha);
                let rotation = Quat::from(object.previous_transform.rotation)
                    .slerp(object.current_transform.rotation, alpha);
                let scale = object
                    .previous_transform
                    .scale
                    .lerp(object.current_transform.scale, alpha);
                let model_matrix = Mat4::from_scale_rotation_translation(scale, rotation, position);
                let push_constants = ModelPushConstant {
                    model_matrix: model_matrix.to_cols_array_2d(),
                };
                render_pass.set_push_constants(
                    wgpu::ShaderStages::VERTEX,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(lod.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..lod.index_count, 0, 0..1);
            }
        }

        drop(render_pass);

        if render_data.render_config.sky_pass {
            let mut sky_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            sky_pass.set_pipeline(self.sky_pipeline.as_ref().unwrap());
            sky_pass.set_bind_group(0, &self.sky_bind_groups[buffer_index], &[]);
            sky_pass.set_bind_group(
                1,
                &self.atmosphere.as_ref().unwrap().sampling_bind_group,
                &[],
            );
            sky_pass.set_bind_group(2, self.render_constants_bind_group.as_ref().unwrap(), &[]);
            sky_pass.draw(0..3, 0..1);
        }

        if render_data.render_config.egui_pass {
            if let Some(egui_data) = &mut self.current_egui_data {
                let screen_descriptor = &mut egui_data.screen_descriptor;
                screen_descriptor.size_in_pixels =
                    [self.window_size.width, self.window_size.height];

                self.egui_renderer.update_buffers(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &egui_data.primitives,
                    &screen_descriptor,
                );

                {
                    let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Egui Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &output_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    self.egui_renderer.render(
                        &mut rpass.forget_lifetime(),
                        &egui_data.primitives,
                        &screen_descriptor,
                    );
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output_frame.present();

        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;

        Ok(())
    }

    fn process_message(&mut self, message: RenderMessage) {
        match message {
            RenderMessage::CreateMesh {
                id,
                vertices,
                lod_indices,
                bounds,
            } => {
                self.add_mesh(id, &vertices, &lod_indices, bounds).unwrap();
            }
            RenderMessage::CreateTexture {
                id,
                name: _,
                kind,
                data,
                format,
                width,
                height,
            } => {
                if let Ok(gpu_index) = self.add_texture(&data, kind, format, width, height) {
                    // This logic is imperfect for textures that aren't 1:1 with a material,
                    // but it works for the current model.
                    let indices = match kind {
                        AssetKind::Albedo => TextureIndices {
                            albedo: gpu_index,
                            ..Default::default()
                        },
                        AssetKind::Normal => TextureIndices {
                            normal: gpu_index,
                            ..Default::default()
                        },
                        AssetKind::MetallicRoughness => TextureIndices {
                            metallic_roughness: gpu_index,
                            ..Default::default()
                        },
                        _ => return,
                    };
                    self.handle_id_to_texture_indices.insert(id, indices);
                }
            }
            RenderMessage::CreateMaterial(mat_data) => {
                self.pending_materials.push(mat_data);
            }
            RenderMessage::RenderData(data) => self.update_render_data(data),
            RenderMessage::EguiData(data) => {
                // Upload textures immediately when message arrives
                for (id, delta) in &data.textures_delta.set {
                    self.egui_renderer
                        .update_texture(&self.device, &self.queue, *id, delta);
                }

                // Free old textures
                for id in &data.textures_delta.free {
                    self.egui_renderer.free_texture(id);
                }

                self.current_egui_data = Some(data);
            }
            RenderMessage::Resize(size) => self.resize(size),
            RenderMessage::Shutdown => {}
            _ => {}
        }
    }

    fn update_render_data(&mut self, render_data: RenderData) {
        if let Some(current_data) = &self.current_render_data {
            if current_data.render_config != render_data.render_config {
                self.needs_atmosphere_precompute = true;

                if current_data.render_config.shader_constants
                    != render_data.render_config.shader_constants
                {
                    self.queue.write_buffer(
                        self.render_constants_buffer.as_ref().unwrap(),
                        0,
                        bytemuck::bytes_of(&render_data.render_config.shader_constants),
                    );
                } else {
                    let _ = self.initialize_resources();
                }
            }
        }
        self.current_render_data = Some(render_data);
    }

    fn resolve_pending_materials(&mut self) {
        if self.pending_materials.is_empty() {
            return;
        }

        let mut newly_completed = Vec::new();

        self.pending_materials.retain(|mat_data| {
            let mut all_ready = true;
            let mut indices = TextureIndices::default();

            // Check albedo
            if let Some(handle_id) = mat_data.albedo_texture_id {
                if let Some(tex_indices) = self.handle_id_to_texture_indices.get(&handle_id) {
                    indices.albedo = tex_indices.albedo;
                } else {
                    all_ready = false;
                }
            }

            // Check normal
            if let Some(handle_id) = mat_data.normal_texture_id {
                if let Some(tex_indices) = self.handle_id_to_texture_indices.get(&handle_id) {
                    indices.normal = tex_indices.normal;
                } else {
                    all_ready = false;
                }
            }

            // Check MR
            if let Some(handle_id) = mat_data.metallic_roughness_texture_id {
                if let Some(tex_indices) = self.handle_id_to_texture_indices.get(&handle_id) {
                    indices.metallic_roughness = tex_indices.metallic_roughness;
                } else {
                    all_ready = false;
                }
            }

            if all_ready {
                let final_material = Material {
                    albedo: mat_data.albedo,
                    metallic: mat_data.metallic,
                    roughness: mat_data.roughness,
                    ao: mat_data.ao,
                    emission_strength: mat_data.emission_strength,
                    emission_color: mat_data.emission_color,
                    albedo_texture_index: indices.albedo as i32,
                    normal_texture_index: indices.normal as i32,
                    metallic_roughness_texture_index: indices.metallic_roughness as i32,
                    emission_texture_index: 0, // Simplified: no emission textures
                };

                newly_completed.push((mat_data.id, final_material));
                false // Remove from pending
            } else {
                true // Keep pending
            }
        });

        for (id, material) in newly_completed {
            if let Err(e) = self.add_material(id, material) {
                warn!("Failed to add material {}: {}", id, e);
            }
        }
    }
}

fn buffer_binding(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
