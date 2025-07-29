//! src/graphics/renderer/deferred_renderer.rs

use crate::{
    graphics::renderer::{
        error::RendererError,
        renderer::{
            Aabb, CASCADE_SPLITS, CameraUniforms, CascadeUniform, FRAMES_IN_FLIGHT, LightData,
            Material, MaterialShaderData, Mesh, ModelPushConstant, NUM_CASCADES, PbrConstants,
            RenderData, RenderTrait, SHADOW_MAP_RESOLUTION, ShadowPipeline, ShadowUniforms,
            TextureManager, Vertex, WGPU_CLIP_SPACE_CORRECTION,
        },
    },
    provided::components::{Camera, LightType},
    runtime::{
        asset_server::{AssetKind, MaterialGpuData},
        runtime::RenderMessage,
    },
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3, Vec4, Vec4Swizzles};
use std::{
    collections::HashMap,
    num::NonZeroU32,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{info, warn};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

// --- CONSTANTS ---
pub const MAX_LIGHTS: usize = 2048;
const MAX_TOTAL_TEXTURES: u32 = 4096;
const DEFAULT_TEXTURE_RESOLUTION: u32 = 1024;

/// The high-end renderer using a deferred pipeline and bindless textures.
pub struct DeferredRenderer {
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    window_size: PhysicalSize<u32>,

    // Maps an asset Handle ID (from AssetServer) to a final GPU texture array index
    handle_id_to_texture_index: HashMap<usize, usize>,

    // A queue for materials waiting for their textures to be loaded on the GPU
    pending_materials: Vec<MaterialGpuData>,

    // Core Pipelines
    geometry_pipeline: Option<wgpu::RenderPipeline>,
    lighting_pipeline: Option<wgpu::RenderPipeline>,
    shadow_pipeline: Option<ShadowPipeline>,
    ssr_pipeline: Option<wgpu::RenderPipeline>,
    ssgi_pipeline: Option<wgpu::RenderPipeline>,

    // Bind Group Layouts
    scene_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    object_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssr_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssgi_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssr_camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    lighting_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ibl_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Bind Groups
    scene_data_bind_groups: Vec<wgpu::BindGroup>,
    object_data_bind_group: Option<wgpu::BindGroup>,
    ssr_inputs_bind_group: Option<wgpu::BindGroup>,
    ssgi_inputs_bind_group: Option<wgpu::BindGroup>,
    ssr_camera_bind_groups: Vec<wgpu::BindGroup>,
    lighting_inputs_bind_group: Option<wgpu::BindGroup>,
    ibl_bind_group: Option<wgpu::BindGroup>,

    // Main Render Targets & Depth
    main_depth_texture: Option<wgpu::Texture>,
    main_depth_texture_view: Option<wgpu::TextureView>,

    // G-Buffer Textures
    gbuf_normal_texture: Option<wgpu::Texture>,
    gbuf_albedo_texture: Option<wgpu::Texture>,
    gbuf_mra_texture: Option<wgpu::Texture>,
    gbuf_emission_texture: Option<wgpu::Texture>,
    gbuf_normal_texture_view: Option<wgpu::TextureView>,
    gbuf_albedo_texture_view: Option<wgpu::TextureView>,
    gbuf_mra_texture_view: Option<wgpu::TextureView>,
    gbuf_emission_texture_view: Option<wgpu::TextureView>,

    // Reflection & Lighting Textures
    ssr_texture: Option<wgpu::Texture>,
    ssr_texture_view: Option<wgpu::TextureView>,
    ssgi_texture: Option<wgpu::Texture>,
    ssgi_texture_view: Option<wgpu::TextureView>,
    history_texture: Option<wgpu::Texture>,
    history_texture_view: Option<wgpu::TextureView>,

    // IBL Textures
    brdf_lut_texture: Option<wgpu::Texture>,
    brdf_lut_view: Option<wgpu::TextureView>,
    irradiance_map_texture: Option<wgpu::Texture>,
    irradiance_map_view: Option<wgpu::TextureView>,
    prefiltered_env_map_texture: Option<wgpu::Texture>,
    prefiltered_env_map_view: Option<wgpu::TextureView>,

    // Buffers
    camera_buffers: Vec<wgpu::Buffer>,
    lights_buffers: Vec<wgpu::Buffer>,
    material_uniform_buffer: Option<wgpu::Buffer>,
    shadow_light_vp_buffer: Option<wgpu::Buffer>,

    // Asset Storage
    meshes: HashMap<usize, Mesh>,
    materials: HashMap<usize, Material>,
    texture_manager: TextureManager,

    // Texture Collections
    albedo_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    normal_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    mr_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    emission_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,

    // Samplers
    pbr_sampler: Option<wgpu::Sampler>,
    gbuffer_sampler: Option<wgpu::Sampler>,
    scene_sampler: Option<wgpu::Sampler>,
    ibl_sampler: Option<wgpu::Sampler>,
    brdf_lut_sampler: Option<wgpu::Sampler>,

    // Shadow Resources
    shadow_map_texture: Option<wgpu::Texture>,
    shadow_map_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    shadow_depth_texture: Option<wgpu::Texture>,
    shadow_depth_view: Option<wgpu::TextureView>,
    shadow_uniforms_buffer: Option<wgpu::Buffer>,
    cascade_views: Option<Vec<wgpu::TextureView>>,

    // State
    frame_index: usize,
    current_render_data: Option<RenderData>,
    logic_frame_duration: Duration,
    last_timestamp: Option<Instant>,
}

impl DeferredRenderer {
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
                required_features: wgpu::Features::PUSH_CONSTANTS
                    | wgpu::Features::FLOAT32_FILTERABLE
                    | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                    | wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                required_limits: wgpu::Limits {
                    // Add push constant limit
                    max_push_constant_size: std::mem::size_of::<PbrConstants>() as u32,
                    max_binding_array_elements_per_shader_stage: MAX_TOTAL_TEXTURES,
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
            .find(|f| !f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mut renderer = Self {
            adapter,
            instance,
            device: Arc::new(device),
            queue,
            surface,
            surface_config,
            window_size: size,
            handle_id_to_texture_index: HashMap::new(),
            pending_materials: Vec::new(),
            geometry_pipeline: None,
            lighting_pipeline: None,
            shadow_pipeline: None,
            ssr_pipeline: None,
            ssgi_pipeline: None,
            scene_data_bind_group_layout: None,
            object_data_bind_group_layout: None,
            ssr_inputs_bind_group_layout: None,
            ssgi_inputs_bind_group_layout: None,
            ssr_camera_bind_group_layout: None,
            lighting_inputs_bind_group_layout: None,
            ibl_bind_group_layout: None,
            scene_data_bind_groups: Vec::new(),
            object_data_bind_group: None,
            ssr_inputs_bind_group: None,
            ssgi_inputs_bind_group: None,
            ssr_camera_bind_groups: Vec::new(),
            lighting_inputs_bind_group: None,
            ibl_bind_group: None,
            main_depth_texture: None,
            main_depth_texture_view: None,
            gbuf_normal_texture: None,
            gbuf_albedo_texture: None,
            gbuf_mra_texture: None,
            gbuf_emission_texture: None,
            gbuf_normal_texture_view: None,
            gbuf_albedo_texture_view: None,
            gbuf_mra_texture_view: None,
            gbuf_emission_texture_view: None,
            ssr_texture: None,
            ssr_texture_view: None,
            ssgi_texture: None,
            ssgi_texture_view: None,
            history_texture: None,
            history_texture_view: None,
            brdf_lut_texture: None,
            brdf_lut_view: None,
            irradiance_map_texture: None,
            irradiance_map_view: None,
            prefiltered_env_map_texture: None,
            prefiltered_env_map_view: None,
            camera_buffers: Vec::new(),
            lights_buffers: Vec::new(),
            material_uniform_buffer: None,
            shadow_light_vp_buffer: None,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            texture_manager: TextureManager::default(),
            albedo_textures: Vec::new(),
            normal_textures: Vec::new(),
            mr_textures: Vec::new(),
            emission_textures: Vec::new(),
            pbr_sampler: None,
            gbuffer_sampler: None,
            scene_sampler: None,
            ibl_sampler: None,
            brdf_lut_sampler: None,
            shadow_map_texture: None,
            shadow_map_view: None,
            shadow_sampler: None,
            shadow_depth_texture: None,
            shadow_depth_view: None,
            shadow_uniforms_buffer: None,
            cascade_views: None,
            frame_index: 0,
            current_render_data: None,
            logic_frame_duration: Duration::from_secs_f32(1.0 / target_tickrate),
            last_timestamp: None,
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

        self.material_uniform_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("materials-buffer"),
            size: (std::mem::size_of::<MaterialShaderData>() * 256) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // --- Create Samplers ---
        self.pbr_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PBR Filtering Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.gbuffer_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("G-Buffer Non-Filtering Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        self.scene_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Scene Filtering Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.ibl_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.brdf_lut_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BRDF LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest, // No mipmapping
            ..Default::default()
        }));

        self.upload_default_textures();
        self.create_render_target_textures();
        self.create_shadow_resources();
        self.create_pipelines_and_bind_groups();
        self.create_object_data_bind_group();

        self.resize(self.window_size);

        info!("initialized deferred (bindless) renderer");
        Ok(())
    }

    fn upload_default_textures(&mut self) {
        // --- Albedo (White) ---
        let white_pixel = [255u8, 255, 255, 255];
        let (tex, view) = self.create_and_upload_texture(
            "Default Albedo",
            &white_pixel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1,
            1, // 1x1 dimension
        );
        self.albedo_textures.push((tex, view));

        // --- Normal (Flat) ---
        // Represents a normal of (0, 0, 1). Packed as (0.5, 0.5, 1.0) -> [128, 128, 255]
        let flat_normal_pixel = [128u8, 128, 255, 255];
        let (tex, view) = self.create_and_upload_texture(
            "Default Normal",
            &flat_normal_pixel,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        self.normal_textures.push((tex, view));

        // --- Metallic/Roughness/AO (Default PBR values) ---
        // AO=1.0 (red=255), Roughness=0.8 (green=204), Metallic=0.0 (blue=0)
        let default_mr_pixel = [255u8, 204, 0, 255];
        let (tex, view) = self.create_and_upload_texture(
            "Default MR",
            &default_mr_pixel,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        self.mr_textures.push((tex, view));

        let (tex, view) = self.create_and_upload_texture(
            "Default Emission",
            &white_pixel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1,
            1,
        );
        self.emission_textures.push((tex, view));
    }

    fn create_and_upload_texture(
        &self,
        label: &str,
        data: &[u8],
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // For a 1x1 texture, bytes_per_row is just the size of the pixel data (4 bytes for RGBA8)
        self.queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            texture_size,
        );

        let view = texture.create_view(&Default::default());
        (texture, view)
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
            format: wgpu::TextureFormat::Rg32Float,
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
            format: wgpu::TextureFormat::Depth32Float,
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

        // --- NEW: Calculate aligned buffer size for multiple matrices ---
        let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
        let aligned_mat4_size = wgpu::util::align_to(mat4_size, alignment);

        self.shadow_light_vp_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Light VP (Dynamic Uniform)"),
            size: NUM_CASCADES as wgpu::BufferAddress * aligned_mat4_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // --- NEW: The layout now specifies a DYNAMIC buffer ---
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true, // <-- The key change
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
                    // Explicitly tell the binding its size is one matrix.
                    // This is the key to making dynamic offsets work.
                    size: wgpu::BufferSize::new(mat4_size as u64),
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
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
                    format: wgpu::TextureFormat::Rg32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back), // Use front-face culling for Peter-Panning
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // Use Less for standard 0-1 depth
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

    fn upload_texture_slice(&self, data: &[u8], target_array: &wgpu::Texture, layer_index: u32) {
        let bytes_per_pixel = 4u32;
        let bytes_per_row = bytes_per_pixel * DEFAULT_TEXTURE_RESOLUTION;

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: target_array,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_index,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(DEFAULT_TEXTURE_RESOLUTION),
            },
            wgpu::Extent3d {
                width: DEFAULT_TEXTURE_RESOLUTION,
                height: DEFAULT_TEXTURE_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_uncompressed_texture_slice(
        &self,
        data: &[u8],
        target_array: &wgpu::Texture,
        layer_index: u32,
        bytes_per_pixel: u32,
    ) {
        // This helper now takes bytes_per_pixel to be more generic.
        let bytes_per_row = bytes_per_pixel * DEFAULT_TEXTURE_RESOLUTION;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target_array,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_index,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(DEFAULT_TEXTURE_RESOLUTION),
            },
            wgpu::Extent3d {
                width: DEFAULT_TEXTURE_RESOLUTION,
                height: DEFAULT_TEXTURE_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_compressed_texture_slice(
        &self,
        data: &[u8],
        target_array: &wgpu::Texture,
        layer_index: u32,
        format: wgpu::TextureFormat,
    ) {
        let block_dimensions = format.block_dimensions();
        let block_size_bytes = format.block_size(None).unwrap();
        let width_in_blocks = DEFAULT_TEXTURE_RESOLUTION / block_dimensions.0;
        let bytes_per_row = width_in_blocks * block_size_bytes as u32;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target_array,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_index,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(DEFAULT_TEXTURE_RESOLUTION / block_dimensions.1),
            },
            wgpu::Extent3d {
                width: DEFAULT_TEXTURE_RESOLUTION,
                height: DEFAULT_TEXTURE_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn add_texture(
        &mut self,
        data: &[u8],
        kind: AssetKind,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<usize, RendererError> {
        // Determine which texture list to add to
        let target_list = match kind {
            AssetKind::Albedo => &mut self.albedo_textures,
            AssetKind::Normal => &mut self.normal_textures,
            AssetKind::MetallicRoughness => &mut self.mr_textures,
            AssetKind::Emission => &mut self.emission_textures,
        };

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("Bindless Texture"),
            view_formats: &[],
        });

        let block_dimensions = format.block_dimensions();
        let block_size_bytes = format.block_size(None).unwrap_or(4); // Default to 4 for uncompressed
        let width_in_blocks = width / block_dimensions.0;
        let height_in_blocks = height / block_dimensions.1;
        let bytes_per_row = width_in_blocks * block_size_bytes;

        self.queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height_in_blocks),
            },
            texture_size,
        );

        let view = texture.create_view(&Default::default());
        target_list.push((texture, view));

        // The new index is simply the last position in the list.
        let new_index = target_list.len() - 1;

        // CRITICAL: We need to rebuild the object data bind group now that
        // a new texture has been added to the list.
        self.create_object_data_bind_group();

        Ok(new_index)
    }

    pub fn add_mesh(
        &mut self,
        id: usize,
        vertices: &[Vertex],
        indices: &[u32],
        bounds: Aabb,
    ) -> Result<(), RendererError> {
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh-vbo-{}", id)),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh-ibo-{}", id)),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        self.meshes.insert(
            id,
            Mesh {
                vertex_buffer,
                index_buffer,
                index_count: indices.len() as u32,
                bounds,
            },
        );
        Ok(())
    }

    pub fn add_material(&mut self, id: usize, material: Material) -> Result<(), RendererError> {
        if id >= 256 {
            return Err(RendererError::ResourceCreation(format!(
                "Material ID {} is out of bounds. Maximum is 255.",
                id
            )));
        }

        let shader_data = MaterialShaderData::from(&material);

        let material_offset =
            (id * std::mem::size_of::<MaterialShaderData>()) as wgpu::BufferAddress;
        self.queue.write_buffer(
            self.material_uniform_buffer.as_ref().unwrap(),
            material_offset,
            bytemuck::bytes_of(&shader_data),
        );

        self.materials.insert(id, material);
        Ok(())
    }

    fn create_render_target_textures(&mut self) {
        let device = &self.device;
        let size = wgpu::Extent3d {
            width: self.surface_config.width,
            height: self.surface_config.height,
            depth_or_array_layers: 1,
        };

        // Main Depth Texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Main Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.main_depth_texture_view = Some(depth_texture.create_view(&Default::default()));
        self.main_depth_texture = Some(depth_texture);

        // G-Buffer Textures
        let gbuffer_usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        let gbuf_normal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("G-Buffer Normal"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: gbuffer_usage,
            view_formats: &[],
        });
        self.gbuf_normal_texture_view = Some(gbuf_normal_texture.create_view(&Default::default()));
        self.gbuf_normal_texture = Some(gbuf_normal_texture);

        let gbuf_albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("G-Buffer Albedo"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: gbuffer_usage,
            view_formats: &[],
        });
        self.gbuf_albedo_texture_view = Some(gbuf_albedo_texture.create_view(&Default::default()));
        self.gbuf_albedo_texture = Some(gbuf_albedo_texture);

        let gbuf_mra_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("G-Buffer MRA"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: gbuffer_usage,
            view_formats: &[],
        });
        self.gbuf_mra_texture_view = Some(gbuf_mra_texture.create_view(&Default::default()));
        self.gbuf_mra_texture = Some(gbuf_mra_texture);

        let gbuf_emission_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("G-Buffer Emission"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: gbuffer_usage,
            view_formats: &[],
        });
        self.gbuf_emission_texture_view =
            Some(gbuf_emission_texture.create_view(&Default::default()));
        self.gbuf_emission_texture = Some(gbuf_emission_texture);

        // --- NEW Reflection/Lighting Textures ---
        let hdr_texture_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;

        // This texture will hold the result of the SSR pass
        let ssr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR Result Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: hdr_texture_usage,
            view_formats: &[],
        });
        self.ssr_texture_view = Some(ssr_texture.create_view(&Default::default()));
        self.ssr_texture = Some(ssr_texture);

        // This texture will hold the result of the SSGI pass
        let ssgi_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSGI Result Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float, // HDR for lighting
            usage: hdr_texture_usage,                 // Same usage as SSR
            view_formats: &[],
        });
        self.ssgi_texture_view = Some(ssgi_texture.create_view(&Default::default()));
        self.ssgi_texture = Some(ssgi_texture);

        let surface_caps = self.surface.get_capabilities(&self.adapter);
        let surface_format = surface_caps.formats[0];

        let history_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("History Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.history_texture_view = Some(history_texture.create_view(&Default::default()));
        self.history_texture = Some(history_texture);

        // --- NEW IBL Textures (Dummy Placeholders) ---
        let brdf_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BRDF LUT"),
            size: wgpu::Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.brdf_lut_view = Some(brdf_lut.create_view(&Default::default()));
        self.brdf_lut_texture = Some(brdf_lut);

        let irradiance_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Irradiance Map"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.irradiance_map_view = Some(irradiance_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        }));
        self.irradiance_map_texture = Some(irradiance_map);

        let prefiltered_env_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Prefiltered Environment Map"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 6,
            },
            mip_level_count: 5, // For roughness levels
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.prefiltered_env_map_view = Some(prefiltered_env_map.create_view(
            &wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            },
        ));
        self.prefiltered_env_map_texture = Some(prefiltered_env_map);
    }

    fn create_pipelines_and_bind_groups(&mut self) {
        let device = &self.device;

        // --- Load Shaders ---
        let g_buffer_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/g_buffer.wgsl"));
        let ssr_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/ssr.wgsl"));
        let ssgi_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/ssgi.wgsl"));
        let lighting_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/lighting.wgsl"));

        // --- Create Layouts ---
        let (
            scene_data_bind_group_layout,
            object_data_bind_group_layout,
            ssr_inputs_bind_group_layout,
            ssgi_inputs_bind_group_layout,
            ssr_camera_bind_group_layout,
            lighting_inputs_bind_group_layout,
            ibl_bind_group_layout,
        ) = self.create_bind_group_layouts();

        // --- Create Pipeline Layouts ---
        let geometry_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Geometry Pipeline Layout"),
                bind_group_layouts: &[
                    &scene_data_bind_group_layout,
                    &object_data_bind_group_layout,
                ],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    range: 0..std::mem::size_of::<PbrConstants>() as u32,
                }],
            });

        let ssr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Pipeline Layout"),
            bind_group_layouts: &[&ssr_inputs_bind_group_layout, &ssr_camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let ssgi_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSGI Pipeline Layout"),
            // same inputs as SSR (G-buffer + Camera)
            bind_group_layouts: &[
                &ssgi_inputs_bind_group_layout,
                &ssr_camera_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let lighting_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Lighting Pipeline Layout"),
                bind_group_layouts: &[
                    &lighting_inputs_bind_group_layout,
                    &scene_data_bind_group_layout,
                    &ibl_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // --- Create Pipelines ---
        self.geometry_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Geometry Pipeline"),
                layout: Some(&geometry_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &g_buffer_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &g_buffer_shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(wgpu::TextureFormat::Rgba16Float.into()), // Normal
                        Some(wgpu::TextureFormat::Rgba8UnormSrgb.into()), // Albedo
                        Some(wgpu::TextureFormat::Rgba8Unorm.into()),  // MRA
                        Some(wgpu::TextureFormat::Rgba16Float.into()), // Emission
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
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

        self.ssr_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("SSR Pipeline"),
                layout: Some(&ssr_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ssr_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssr_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float, // SSR result
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            }),
        );

        self.ssgi_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("SSGI Pipeline"),
                layout: Some(&ssgi_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ssgi_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssgi_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float, // SSGI result
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            }),
        );

        self.lighting_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Lighting Pipeline"),
                layout: Some(&lighting_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &lighting_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &lighting_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format, // Final output
                        blend: None,
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

        // --- Store Layouts on Self ---
        self.scene_data_bind_group_layout = Some(scene_data_bind_group_layout);
        self.object_data_bind_group_layout = Some(object_data_bind_group_layout);
        self.ssr_inputs_bind_group_layout = Some(ssr_inputs_bind_group_layout);
        self.ssgi_inputs_bind_group_layout = Some(ssgi_inputs_bind_group_layout);
        self.ssr_camera_bind_group_layout = Some(ssr_camera_bind_group_layout);
        self.lighting_inputs_bind_group_layout = Some(lighting_inputs_bind_group_layout);
        self.ibl_bind_group_layout = Some(ibl_bind_group_layout);

        // --- Create Bind Groups ---
        self.create_bind_groups();
    }

    fn create_bind_group_layouts(
        &self,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        let device = &self.device;

        let scene_data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scene Data Bind Group Layout"),
                entries: &[
                    buffer_binding(
                        0,
                        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                    buffer_binding(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Storage { read_only: true },
                    ),
                    texture_binding(2, true, wgpu::TextureViewDimension::D2Array, false),
                    sampler_binding(3, true, wgpu::SamplerBindingType::Filtering),
                    buffer_binding(
                        4,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                ],
            });

        let object_data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Object Data Bind Group Layout"),
                entries: &[
                    // Material buffer
                    buffer_binding(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Storage { read_only: true },
                    ),
                    // Sampler
                    sampler_binding(2, true, wgpu::SamplerBindingType::Filtering),
                    // A SINGLE entry for ALL textures
                    wgpu::BindGroupLayoutEntry {
                        binding: 1, // The master texture array
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: Some(std::num::NonZeroU32::new(MAX_TOTAL_TEXTURES).unwrap()),
                    },
                ],
            });

        let ssr_inputs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSR Inputs Bind Group Layout"),
                entries: &[
                    // MODIFIED: Now takes gbuf_albedo instead of a separate scene_texture
                    texture_binding(0, false, wgpu::TextureViewDimension::D2, false), // gbuf_normal
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, false), // gbuf_mra
                    texture_binding(2, false, wgpu::TextureViewDimension::D2, true),  // depth
                    texture_binding(3, true, wgpu::TextureViewDimension::D2, false),  // history
                    sampler_binding(4, false, wgpu::SamplerBindingType::NonFiltering), // gbuf_sampler
                    sampler_binding(5, true, wgpu::SamplerBindingType::Filtering), // scene_sampler
                ],
            });

        let ssgi_inputs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSGI Inputs Bind Group Layout"),
                entries: &[
                    texture_binding(0, false, wgpu::TextureViewDimension::D2, false), // gbuf_normal
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, false), // gbuf_albedo
                    texture_binding(2, false, wgpu::TextureViewDimension::D2, true),  // depth
                    texture_binding(3, true, wgpu::TextureViewDimension::D2, false),  // history
                    sampler_binding(4, false, wgpu::SamplerBindingType::NonFiltering), // gbuf_sampler
                    sampler_binding(5, true, wgpu::SamplerBindingType::Filtering), // scene_sampler
                ],
            });

        let ssr_camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSR Camera Bind Group Layout"),
                entries: &[buffer_binding(
                    0,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Uniform,
                )],
            });

        let lighting_inputs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lighting Inputs Bind Group Layout"),
                entries: &[
                    texture_binding(0, false, wgpu::TextureViewDimension::D2, true), // Depth
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, false), // Normal
                    texture_binding(2, false, wgpu::TextureViewDimension::D2, false), // Albedo
                    texture_binding(3, false, wgpu::TextureViewDimension::D2, false), // MRA
                    texture_binding(4, false, wgpu::TextureViewDimension::D2, false), // Emission
                    sampler_binding(5, false, wgpu::SamplerBindingType::NonFiltering), // Sampler
                    texture_binding(6, false, wgpu::TextureViewDimension::D2, false), // SSR Result
                    texture_binding(7, false, wgpu::TextureViewDimension::D2, false), // SSGI Result
                ],
            });

        let ibl_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("IBL Bind Group Layout"),
                entries: &[
                    texture_binding(0, true, wgpu::TextureViewDimension::D2, false),
                    texture_binding(1, true, wgpu::TextureViewDimension::Cube, false),
                    texture_binding(2, true, wgpu::TextureViewDimension::Cube, false),
                    sampler_binding(3, true, wgpu::SamplerBindingType::Filtering),
                    sampler_binding(4, true, wgpu::SamplerBindingType::Filtering),
                ],
            });

        (
            scene_data_bind_group_layout,
            object_data_bind_group_layout,
            ssr_inputs_bind_group_layout,
            ssgi_inputs_bind_group_layout,
            ssr_camera_bind_group_layout,
            lighting_inputs_bind_group_layout,
            ibl_bind_group_layout,
        )
    }

    fn create_object_data_bind_group(&mut self) {
        // 1. Get a reference to a default texture view to use for padding.
        //    The default albedo (a 1x1 white pixel) at index 0 is perfect for this.
        let default_view = &self.albedo_textures[0].1;

        // 2. Flatten all the *currently loaded* texture views into one list, just like before.
        let mut all_views: Vec<&wgpu::TextureView> = Vec::new();
        all_views.extend(self.albedo_textures.iter().map(|(_, view)| view));
        all_views.extend(self.normal_textures.iter().map(|(_, view)| view));
        all_views.extend(self.mr_textures.iter().map(|(_, view)| view));
        all_views.extend(self.emission_textures.iter().map(|(_, view)| view));

        // 3. Pad the list to the required size.
        //    `MAX_TOTAL_TEXTURES` should be the same constant used in your layout (e.g., 4096).
        let current_len = all_views.len();
        if current_len < MAX_TOTAL_TEXTURES as usize {
            // Add the default view to the list until it reaches the required length.
            all_views.resize(MAX_TOTAL_TEXTURES as usize, default_view);
        }

        // 4. Create the bind group. `all_views` now has the exact length required by the layout.
        self.object_data_bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: self.object_data_bind_group_layout.as_ref().unwrap(),
                label: Some("Object Data Bind Group (Bindless)"),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self
                            .material_uniform_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1, // The single texture array binding
                        resource: wgpu::BindingResource::TextureViewArray(&all_views),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            self.pbr_sampler.as_ref().unwrap(),
                        ),
                    },
                ],
            }),
        );
    }

    fn create_bind_groups(&mut self) {
        let device = &self.device;

        // Bind groups for scene data (one per frame in flight)
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
                    ],
                })
            })
            .collect();

        // Bind groups for just the camera (for SSR pass)
        self.ssr_camera_bind_groups = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.ssr_camera_bind_group_layout.as_ref().unwrap(),
                    label: Some(&format!("SSR Camera Bind Group {}", i)),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.camera_buffers[i].as_entire_binding(),
                    }],
                })
            })
            .collect();

        // Bind group for SSR inputs
        self.ssr_inputs_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Inputs Bind Group"),
            layout: self.ssr_inputs_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.gbuf_normal_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        self.gbuf_mra_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.main_depth_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        self.history_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(
                        self.gbuffer_sampler.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(self.scene_sampler.as_ref().unwrap()),
                },
            ],
        }));

        // Bind group for SSGI inputs
        self.ssgi_inputs_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSGI Inputs Bind Group"),
            layout: self.ssgi_inputs_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.gbuf_normal_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        self.gbuf_albedo_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.main_depth_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        self.history_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(
                        self.gbuffer_sampler.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(self.scene_sampler.as_ref().unwrap()),
                },
            ],
        }));

        // Bind group for final lighting inputs
        self.lighting_inputs_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Lighting Inputs Bind Group"),
                layout: self.lighting_inputs_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            self.main_depth_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_normal_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_albedo_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_mra_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_emission_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(
                            self.gbuffer_sampler.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(
                            self.ssr_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            self.ssgi_texture_view.as_ref().unwrap(),
                        ),
                    },
                ],
            }));

        // Bind group for IBL textures
        self.ibl_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL Bind Group"),
            layout: self.ibl_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.brdf_lut_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        self.irradiance_map_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.prefiltered_env_map_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(self.ibl_sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(
                        self.brdf_lut_sampler.as_ref().unwrap(),
                    ),
                },
            ],
        }));
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

        // Apply the correction here
        let corrected_proj = WGPU_CLIP_SPACE_CORRECTION * projection_matrix;

        let inv_proj = corrected_proj.inverse();
        let inv_view_proj = (corrected_proj * view_matrix).inverse();

        let camera_uniforms = CameraUniforms {
            view_matrix: view_matrix.to_cols_array_2d(),
            projection_matrix: corrected_proj.to_cols_array_2d(),
            inverse_projection_matrix: inv_proj.to_cols_array_2d(),
            inverse_view_projection_matrix: inv_view_proj.to_cols_array_2d(),
            view_position: eye.to_array(),
            light_count: render_data.lights.len() as u32,
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
        // This should be greater than the farthest shadow cascade to allow objects
        // just outside the view to cast shadows into it.
        const FAR_CASCADE_DISTANCE: f32 = 500.0; // The distance of the farthest shadow cascade.
        const MAX_SHADOW_CASTING_DISTANCE: f32 = FAR_CASCADE_DISTANCE * 1.5;
        const MAX_SHADOW_CASTING_DISTANCE_SQ: f32 =
            MAX_SHADOW_CASTING_DISTANCE * MAX_SHADOW_CASTING_DISTANCE;

        let shadow_light = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional));

        if let (Some(light), Some(shadow_pipeline)) = (shadow_light, self.shadow_pipeline.as_ref())
        {
            assert!(
                light.current_transform.rotation.is_normalized(),
                "Directional light rotation is not normalized!"
            );

            let camera_pos = render_data.current_camera_transform.position;

            // --- Step 1: Pre-calculate culling status for all objects ---
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

            // --- Step 2: Calculate scene bounds using the pre-culled results ---
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

            // Calculate alignment for dynamic uniform buffer offsets.
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
                            load: wgpu::LoadOp::Clear(1.0), // Clear depth to max value.
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                shadow_pass.set_pipeline(&shadow_pipeline.pipeline);
                shadow_pass.set_bind_group(0, &shadow_pipeline.bind_group, &[offset]);

                for object in &render_data.objects {
                    if object.casts_shadow {
                        if *culled_objects.get(&object.id).unwrap_or(&true) {
                            continue;
                        }

                        if let Some(mesh) = self.meshes.get(&object.mesh_id) {
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
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
                }
            }
        }
    }

    fn run_geometry_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_data: &RenderData,
        alpha: f32,
    ) {
        let gbuffer_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: self.gbuf_normal_texture_view.as_ref().unwrap(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: self.gbuf_albedo_texture_view.as_ref().unwrap(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: self.gbuf_mra_texture_view.as_ref().unwrap(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: self.gbuf_emission_texture_view.as_ref().unwrap(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];
        let mut geometry_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Geometry Pass"),
            color_attachments: &gbuffer_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.main_depth_texture_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        geometry_pass.set_pipeline(self.geometry_pipeline.as_ref().unwrap());
        geometry_pass.set_bind_group(0, &self.scene_data_bind_groups[self.frame_index], &[]);
        geometry_pass.set_bind_group(1, self.object_data_bind_group.as_ref().unwrap(), &[]);
        for object in &render_data.objects {
            if let (Some(mesh), true) = (
                self.meshes.get(&object.mesh_id),
                self.materials.contains_key(&object.material_id),
            ) {
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
                let push_constants = PbrConstants {
                    model_matrix: model_matrix.to_cols_array_2d(),
                    material_id: object.material_id as u32,
                    _p: [0; 3],
                };
                geometry_pass.set_push_constants(
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
                geometry_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                geometry_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                geometry_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }

    fn run_ssgi_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.ssgi_texture_view.as_ref().unwrap(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(self.ssgi_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.ssgi_inputs_bind_group.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, &self.ssr_camera_bind_groups[self.frame_index], &[]);
        pass.draw(0..3, 0..1);
    }

    fn run_ssr_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.ssr_texture_view.as_ref().unwrap(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(self.ssr_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.ssr_inputs_bind_group.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, &self.ssr_camera_bind_groups[self.frame_index], &[]);
        pass.draw(0..3, 0..1);
    }

    fn run_lighting_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Lighting Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
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
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(self.lighting_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.lighting_inputs_bind_group.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, &self.scene_data_bind_groups[self.frame_index], &[]);
        pass.set_bind_group(2, self.ibl_bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
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
            // 1. Get the world-space corners of this cascade's frustum slice
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

            // 2. Center the light on the cascade's frustum slice
            let world_center = frustum_corners_world.iter().sum::<Vec3>() / 8.0;

            let light_view = Mat4::look_at_rh(world_center - light_dir, world_center, Vec3::Y);

            // 3. Calculate the tightest possible orthographic projection around the cascade
            let mut cascade_min = Vec3::splat(f32::MAX);
            let mut cascade_max = Vec3::splat(f32::MIN);
            for corner in frustum_corners_world {
                let trf = light_view * corner.extend(1.0);
                cascade_min = cascade_min.min(trf.xyz());
                cascade_max = cascade_max.max(trf.xyz());
            }

            // 4. Expand the tight projection to include the entire scene's depth
            let mut scene_min_z = f32::MAX;
            let mut scene_max_z = f32::MIN;
            for corner in &scene_corners {
                let trf = light_view * corner.extend(1.0);
                scene_min_z = scene_min_z.min(trf.z);
                scene_max_z = scene_max_z.max(trf.z);
            }

            // 5. Create the final projection matrix
            let light_proj = Mat4::orthographic_rh(
                cascade_min.x,
                cascade_max.x,
                cascade_min.y,
                cascade_max.y,
                // THE FIX: Convert negative Z coordinates into positive distances
                -scene_max_z,
                -scene_min_z,
            );

            // 6. Finalize and store
            let final_light_vp = light_proj * light_view;
            uniforms.cascades[i] = CascadeUniform {
                light_view_proj: final_light_vp.to_cols_array_2d(),
                // Store the depth value in the first component of the array.
                split_depth: [-z_far, 0.0, 0.0, 0.0],
            };
        }

        uniforms
    }
}

impl RenderTrait for DeferredRenderer {
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);

            // Recreate all screen-size dependent textures and their bind groups
            self.create_render_target_textures();
            self.create_bind_groups();
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

        // --- Update Uniforms ---
        self.update_uniforms(render_data, alpha);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Command Encoder"),
            });

        let camera_transform = &render_data.current_camera_transform;
        let eye = camera_transform.position;
        let forward = camera_transform.forward();
        let up = camera_transform.up();
        let static_camera_view = Mat4::look_at_rh(eye, eye + forward, up);

        // --- 1. Shadow Pass ---
        self.run_shadow_pass(&mut encoder, render_data, &static_camera_view, alpha);

        // --- 2. Geometry Pass ---
        self.run_geometry_pass(&mut encoder, render_data, alpha);

        // --- 3. SSGI Pass ---
        self.run_ssgi_pass(&mut encoder);

        // --- 4. SSR Pass ---
        self.run_ssr_pass(&mut encoder);

        // --- 5. Lighting/Composite Pass ---
        self.run_lighting_pass(&mut encoder, &output_view);

        // --- 6. Copy to History Buffer ---
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &output_frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: self.history_texture.as_ref().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
        );

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
                indices,
                bounds,
            } => {
                self.add_mesh(id, &vertices, &indices, bounds).unwrap();
            }
            RenderMessage::CreateTexture {
                id,
                name,
                kind,
                data,
                format,
                width,
                height,
            } => {
                if let Ok(gpu_index) = self.add_texture(&data, kind, format, width, height) {
                    self.handle_id_to_texture_index.insert(id, gpu_index);
                }
            }
            RenderMessage::CreateMaterial(mat_data) => {
                self.pending_materials.push(mat_data);
            }
            RenderMessage::RenderData(data) => self.update_render_data(data),
            RenderMessage::Resize(size) => self.resize(size),
            RenderMessage::Shutdown => { /* Handle shutdown */ }
        }
    }

    fn update_render_data(&mut self, render_data: RenderData) {
        self.current_render_data = Some(render_data);
    }

    fn resolve_pending_materials(&mut self) {
        if self.pending_materials.is_empty() {
            return;
        }

        let mut newly_completed = Vec::new();

        self.pending_materials.retain(|mat_data| {
            // Helper to resolve a handle ID to a final GPU index
            let resolve = |id: Option<usize>| -> (bool, Option<i32>) {
                match id {
                    None => (true, Some(0)), // No texture needed, so it's "ready"
                    Some(handle_id) => {
                        if let Some(&gpu_index) = self.handle_id_to_texture_index.get(&handle_id) {
                            (true, Some(gpu_index as i32)) // Found it! It's ready.
                        } else {
                            (false, None) // Not found yet. Not ready.
                        }
                    }
                }
            };

            let (albedo_ready, albedo_idx) = resolve(mat_data.albedo_texture_id);
            let (normal_ready, normal_idx) = resolve(mat_data.normal_texture_id);
            let (mr_ready, mr_idx) = resolve(mat_data.metallic_roughness_texture_id);
            let (emission_ready, emission_idx) = resolve(mat_data.emission_texture_id);

            // If all textures this material needs are ready...
            if albedo_ready && normal_ready && mr_ready && emission_ready {
                // Get the local indices first (e.g., albedo_idx might be 5, meaning the 5th albedo texture)
                let albedo_local_idx = albedo_idx.unwrap();
                let normal_local_idx = normal_idx.unwrap();
                let mr_local_idx = mr_idx.unwrap();
                let emission_local_idx = emission_idx.unwrap();

                // --- Calculate Final, Global Indices ---
                // The albedo textures are first, so their offset is 0.
                let albedo_final_idx = albedo_local_idx;

                // The normal textures come after all albedo textures.
                let normal_final_idx = self.albedo_textures.len() as i32 + normal_local_idx;

                // The MR textures come after all albedo and normal textures.
                let mr_final_idx =
                    (self.albedo_textures.len() + self.normal_textures.len()) as i32 + mr_local_idx;

                // The emission textures are last.
                let emission_final_idx = (self.albedo_textures.len()
                    + self.normal_textures.len()
                    + self.mr_textures.len()) as i32
                    + emission_local_idx;

                // Create the final material struct using these new indices
                let final_material = Material {
                    // Use the full path
                    albedo: mat_data.albedo,
                    metallic: mat_data.metallic,
                    roughness: mat_data.roughness,
                    ao: mat_data.ao,
                    emission_strength: mat_data.emission_strength,
                    emission_color: mat_data.emission_color,
                    albedo_texture_index: albedo_final_idx,
                    normal_texture_index: normal_final_idx,
                    metallic_roughness_texture_index: mr_final_idx,
                    emission_texture_index: emission_final_idx,
                };

                newly_completed.push((mat_data.id, final_material));
                return false; // Remove from pending list
            }

            true // Keep in pending list
        });

        // Add all newly completed materials to the renderer's main storage
        for (id, material) in newly_completed {
            if let Err(e) = self.add_material(id, material) {
                warn!("Failed to add material {}: {}", id, e);
            }
        }
    }
}

// --- BIND GROUP HELPER FUNCTIONS ---

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

fn texture_binding(
    binding: u32,
    filterable: bool,
    view_dimension: wgpu::TextureViewDimension,
    is_depth: bool,
) -> wgpu::BindGroupLayoutEntry {
    let sample_type = if is_depth {
        wgpu::TextureSampleType::Depth
    } else {
        wgpu::TextureSampleType::Float { filterable }
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_binding(
    binding: u32,
    filtering: bool,
    _ty: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    let ty = if filtering {
        wgpu::SamplerBindingType::Filtering
    } else {
        wgpu::SamplerBindingType::NonFiltering
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}
