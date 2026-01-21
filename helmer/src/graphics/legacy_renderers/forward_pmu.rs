use egui_wgpu::Renderer as EguiRenderer;
use glam::{Mat4, Quat, Vec3, Vec4Swizzles};
use std::{
    cell::RefCell,
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{info, warn};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::{
    graphics::common::{
        atmosphere::AtmospherePrecomputer,
        constants::{CASCADE_SPLITS, FRAMES_IN_FLIGHT, NUM_CASCADES, SHADOW_MAP_RESOLUTION},
        error::RendererError,
        mipmap::MipmapGenerator,
        renderer::{
            Aabb, CameraUniforms, CascadeUniform, EguiRenderData, InstanceRaw, LightData, Mesh,
            MeshLod, MeshLodPayload, ModelPushConstant, PbrConstants, RenderData, RenderMessage,
            RenderObject, RenderTrait, ShaderConstants, ShadowPipeline, ShadowUniforms,
            SkyUniforms, Vertex, build_mip_uploads, calc_mip_level_count,
        },
    },
    provided::components::{Camera, LightType},
    runtime::asset_server::MaterialGpuData,
};

// --- CONSTANTS ---
pub const MAX_LIGHTS: usize = 248;

// --- Structs for Shader Data ---

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniforms {
    albedo: [f32; 4],
    emission_color: [f32; 3],
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,
    _padding: f32,
}

// --- Low-End Path Specific Structs ---
pub struct MaterialLowEnd {
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
}

/// The low-end renderer using a single forward pass and per-material bind groups.
pub struct ForwardRendererPMU {
    // WGPU Core
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    mipmap_generator: MipmapGenerator,

    // Shared Resources
    meshes: HashMap<usize, Mesh>,
    camera_buffers: Vec<wgpu::Buffer>,
    lights_buffers: Vec<wgpu::Buffer>,
    pbr_sampler: Option<wgpu::Sampler>,
    main_depth_texture: Option<wgpu::Texture>,
    main_depth_texture_view: Option<wgpu::TextureView>,

    // Shadow Resources
    shadow_pipeline: Option<ShadowPipeline>,
    shadow_uniforms_buffer: Option<wgpu::Buffer>,
    shadow_light_vp_buffer: Option<wgpu::Buffer>,
    shadow_map_texture: Option<wgpu::Texture>,
    shadow_map_view: Option<wgpu::TextureView>,
    shadow_depth_texture: Option<wgpu::Texture>,
    shadow_depth_view: Option<wgpu::TextureView>,
    cascade_views: Option<Vec<wgpu::TextureView>>,
    shadow_sampler: Option<wgpu::Sampler>,

    // atmosphere
    sky_uniforms_buffers: Vec<wgpu::Buffer>,
    atmosphere: Option<AtmospherePrecomputer>,

    // Forward-Specific Resources
    forward_pipeline: Option<wgpu::RenderPipeline>,
    scene_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    material_bind_group_layout: Option<wgpu::BindGroupLayout>,
    scene_data_bind_groups: Vec<wgpu::BindGroup>,
    materials: HashMap<usize, MaterialLowEnd>,

    // instancing
    shadow_instance_buffer: RefCell<Option<wgpu::Buffer>>,
    shadow_instance_capacity: RefCell<usize>,
    forward_instance_buffer: RefCell<Option<wgpu::Buffer>>,
    forward_instance_capacity: RefCell<usize>,

    // IBL Resources
    ibl_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ibl_bind_group: Option<wgpu::BindGroup>,
    ibl_sampler: Option<wgpu::Sampler>,
    brdf_lut_sampler: Option<wgpu::Sampler>,
    brdf_lut_view: Option<wgpu::TextureView>,
    irradiance_map_view: Option<wgpu::TextureView>,
    prefiltered_env_map_view: Option<wgpu::TextureView>,

    // render constants
    render_constants_bind_group_layout: Option<wgpu::BindGroupLayout>,
    render_constants_bind_group: Option<wgpu::BindGroup>,
    render_constants_buffer: Option<wgpu::Buffer>,

    // Texture Management
    loaded_texture_views: HashMap<usize, Arc<wgpu::TextureView>>,
    default_albedo_view: Option<Arc<wgpu::TextureView>>,
    default_normal_view: Option<Arc<wgpu::TextureView>>,
    default_mr_view: Option<Arc<wgpu::TextureView>>,
    default_emission_view: Option<Arc<wgpu::TextureView>>,

    // EGUI
    egui_renderer: EguiRenderer,
    current_egui_data: Option<EguiRenderData>,

    // State
    window_size: PhysicalSize<u32>,
    frame_index: usize,
    pending_materials: Vec<MaterialGpuData>,
    current_render_data: Option<Arc<RenderData>>,
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

impl ForwardRendererPMU {
    pub async fn new(
        _instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        size: PhysicalSize<u32>,
        target_tickrate: f32,
    ) -> Result<Self, RendererError> {
        let required_features = wgpu::Features::empty();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Forward Renderer Device"),
                required_features,
                required_limits: wgpu::Limits {
                    max_immediate_size: 0,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .map_err(|e| {
                RendererError::ResourceCreation(format!("Failed to create device: {}", e))
            })?;

        let device = Arc::new(device);
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

        let mipmap_generator = MipmapGenerator::new(&device);

        let egui_renderer = EguiRenderer::new(
            &device,
            surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        let mut renderer = Self {
            device,
            queue,
            surface,
            surface_config,
            window_size: size,
            mipmap_generator,
            meshes: HashMap::new(),
            camera_buffers: Vec::new(),
            lights_buffers: Vec::new(),
            pbr_sampler: None,
            main_depth_texture: None,
            main_depth_texture_view: None,
            shadow_pipeline: None,
            shadow_uniforms_buffer: None,
            shadow_light_vp_buffer: None,
            shadow_map_texture: None,
            shadow_map_view: None,
            shadow_depth_texture: None,
            shadow_depth_view: None,
            cascade_views: None,
            shadow_sampler: None,
            sky_uniforms_buffers: Vec::new(),
            atmosphere: None,
            materials: HashMap::new(),
            shadow_instance_buffer: RefCell::new(None),
            shadow_instance_capacity: RefCell::new(0),
            forward_instance_buffer: RefCell::new(None),
            forward_instance_capacity: RefCell::new(0),
            forward_pipeline: None,
            scene_data_bind_group_layout: None,
            material_bind_group_layout: None,
            scene_data_bind_groups: Vec::new(),
            ibl_bind_group_layout: None,
            ibl_bind_group: None,
            ibl_sampler: None,
            brdf_lut_sampler: None,
            brdf_lut_view: None,
            irradiance_map_view: None,
            prefiltered_env_map_view: None,
            render_constants_bind_group_layout: None,
            render_constants_bind_group: None,
            render_constants_buffer: None,
            loaded_texture_views: HashMap::new(),
            default_albedo_view: None,
            default_normal_view: None,
            default_mr_view: None,
            default_emission_view: None,
            egui_renderer,
            current_egui_data: None,
            frame_index: 0,
            pending_materials: Vec::new(),
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

        info!("initialized forward (PMU) renderer");
        Ok(renderer)
    }

    fn initialize_resources(&mut self) -> Result<(), RendererError> {
        self.create_samplers();
        self.atmosphere = Some(AtmospherePrecomputer::new(&self.device));
        self.create_default_textures();
        self.create_ibl_resources();
        self.create_layouts();
        self.create_pipelines();
        self.create_shared_buffers();
        self.create_shadow_resources();
        self.create_bind_groups();
        self.resize(self.window_size);
        Ok(())
    }

    fn create_samplers(&mut self) {
        self.pbr_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PBR Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }));
        self.ibl_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }));
        self.brdf_lut_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BRDF LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        self.scene_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Scene Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }));
    }

    fn create_default_textures(&mut self) {
        let create_default = |label, data, format| {
            let size = wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            };
            let texture = self.device.create_texture_with_data(
                &self.queue,
                &wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                data,
            );
            Arc::new(texture.create_view(&Default::default()))
        };
        self.default_albedo_view = Some(create_default(
            "Default Albedo",
            &[255, 255, 255, 255],
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ));
        self.default_normal_view = Some(create_default(
            "Default Normal",
            &[128, 128, 255, 255],
            wgpu::TextureFormat::Rgba8Unorm,
        ));
        self.default_mr_view = Some(create_default(
            "Default MR",
            &[255, 204, 0, 255],
            wgpu::TextureFormat::Rgba8Unorm,
        ));

        // --- FIX: Default emission texture must be WHITE, not black ---
        self.default_emission_view = Some(create_default(
            "Default Emission",
            &[255, 255, 255, 255],
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ));
    }

    fn create_ibl_resources(&mut self) {
        let brdf_lut = self.device.create_texture(&wgpu::TextureDescriptor {
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

        let irradiance_map = self.device.create_texture(&wgpu::TextureDescriptor {
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

        let prefiltered_env_map = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Prefiltered Env Map"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 6,
            },
            mip_level_count: 5,
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
    }

    fn create_layouts(&mut self) {
        self.scene_data_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Forward Scene BGL"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
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
        self.material_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Forward Material BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
                ],
            },
        ));
        self.ibl_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("IBL Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            },
        ));
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
    }

    fn create_pipelines(&mut self) {
        let shader = self
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/forward_pmu.wgsl"));
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Forward Pipeline Layout"),
                bind_group_layouts: &[
                    self.scene_data_bind_group_layout.as_ref().unwrap(),
                    self.material_bind_group_layout.as_ref().unwrap(),
                    &self.atmosphere.as_ref().unwrap().sampling_bind_group_layout,
                    self.ibl_bind_group_layout.as_ref().unwrap(),
                ],
                immediate_size: 0,
            });
        self.forward_pipeline = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Forward Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc(), InstanceRaw::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(self.surface_config.format.into())],
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
                multiview_mask: None,
                cache: None,
            },
        ));
        self.create_shadow_pipeline();
        self.create_sky_pipeline();
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
                immediate_size: 0,
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
                multiview_mask: None,
                cache: None,
            },
        ));
    }

    fn create_bind_groups(&mut self) {
        self.create_scene_bind_groups();
        self.ibl_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    fn create_shared_buffers(&mut self) {
        self.camera_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("camera-uniforms-{}", i)),
                    size: std::mem::size_of::<CameraUniforms>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        self.lights_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("lights-buffer-{}", i)),
                    size: (std::mem::size_of::<LightData>() * MAX_LIGHTS) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

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
    }

    fn create_scene_bind_groups(&mut self) {
        self.scene_data_bind_groups = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Forward Scene BG {}", i)),
                    layout: self.scene_data_bind_group_layout.as_ref().unwrap(),
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
    }

    fn create_sky_bind_groups(&mut self) {
        let depth_view = self.main_depth_texture_view.as_ref().unwrap();
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
        let shadow_map_texture = device.create_texture(&shadow_texture_desc);
        self.shadow_map_view = Some(
            shadow_map_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("VSM Shadow Map View"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
        );
        self.shadow_map_texture = Some(shadow_map_texture);

        self.shadow_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
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
        let shadow_depth_texture = device.create_texture(&shadow_depth_desc);
        self.shadow_depth_view = Some(shadow_depth_texture.create_view(&Default::default()));
        self.shadow_depth_texture = Some(shadow_depth_texture);

        self.shadow_uniforms_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Uniforms Buffer"),
            size: std::mem::size_of::<ShadowUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.cascade_views = Some(
            (0..NUM_CASCADES)
                .map(|i| {
                    self.shadow_map_texture.as_ref().unwrap().create_view(
                        &wgpu::TextureViewDescriptor {
                            label: Some(&format!("Cascade View {}", i)),
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_array_layer: i as u32,
                            array_layer_count: Some(1),
                            ..Default::default()
                        },
                    )
                })
                .collect(),
        );
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
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
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
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
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
            multiview_mask: None,
            cache: None,
        });

        self.shadow_pipeline = Some(ShadowPipeline {
            pipeline,
            bind_group,
        });
    }

    pub fn add_mesh(
        &mut self,
        id: usize,
        lods: &[MeshLodPayload],
        bounds: Aabb,
    ) -> Result<(), RendererError> {
        let mut gpu_lods = Vec::new();
        for lod in lods {
            let vertex_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-vbo-{}-lod{}", id, lod.lod_index)),
                    contents: bytemuck::cast_slice(lod.vertices.as_ref()),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-ibo-{}-lod{}", id, lod.lod_index)),
                    contents: bytemuck::cast_slice(lod.indices.as_ref()),
                    usage: wgpu::BufferUsages::INDEX,
                });
            gpu_lods.push(MeshLod {
                vertex_buffer,
                index_buffer,
                index_count: lod.indices.len() as u32,
            });
        }

        self.meshes.insert(
            id,
            Mesh {
                lods: gpu_lods,
                bounds,
            },
        );
        Ok(())
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
            // --- 1. Culling and Scene Bounds Calculation ---
            let camera_pos = render_data.current_camera_transform.position;
            let mut culled_objects = HashMap::new();
            let mut scene_bounds_min = Vec3::splat(f32::MAX);
            let mut scene_bounds_max = Vec3::splat(f32::MIN);

            let mut shadow_casters = Vec::new();

            for object in &render_data.objects {
                if object.casts_shadow {
                    let distance_sq = object
                        .current_transform
                        .position
                        .distance_squared(camera_pos);
                    let is_culled = distance_sq > MAX_SHADOW_CASTING_DISTANCE_SQ;
                    culled_objects.insert(object.id, is_culled);

                    if !is_culled {
                        shadow_casters.push(object); // Add to a list for sorting
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
            }
            let dynamic_scene_bounds = Aabb {
                min: scene_bounds_min,
                max: scene_bounds_max,
            };

            // --- 2. Cascade Calculation and Uniform Uploads ---
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

            // --- 3. Build CPU-side instance data and batch info ---
            // (Key is (mesh_id, lod_index))
            // (Value is (instance_offset, instance_count))
            let mut batch_info: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
            let mut all_instances: Vec<InstanceRaw> = Vec::new();

            // Sort objects by mesh/lod to create contiguous batches
            shadow_casters.sort_by_key(|obj| (obj.mesh_id, obj.lod_index));

            for object in shadow_casters {
                if let Some(mesh) = self.meshes.get(&object.mesh_id) {
                    if mesh.lods.is_empty() {
                        continue;
                    }
                    let lod_index = object.lod_index.min(mesh.lods.len() - 1);
                    let key = (object.mesh_id, lod_index);

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

                    let instance_data = InstanceRaw {
                        model_matrix: model_matrix.to_cols_array_2d(),
                    };

                    let current_offset = all_instances.len() as u32;
                    all_instances.push(instance_data);

                    let entry = batch_info.entry(key).or_insert((current_offset, 0));
                    entry.1 += 1;
                }
            }

            let total_instances = all_instances.len();
            if total_instances == 0 {
                return; // No shadows to cast
            }

            // --- 4. Check and resize the GPU buffer if needed ---
            let mut capacity = self.shadow_instance_capacity.borrow_mut();
            let mut buffer = self.shadow_instance_buffer.borrow_mut();

            if total_instances > *capacity || buffer.is_none() {
                if let Some(old_buffer) = buffer.take() {
                    old_buffer.destroy();
                }
                let new_capacity = (total_instances as f32 * 1.5).ceil() as usize;

                *buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Shadow Instance Buffer"),
                    size: (new_capacity * std::mem::size_of::<InstanceRaw>())
                        as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                *capacity = new_capacity;
            }

            // --- 5. Upload all instance data in one go ---
            self.queue.write_buffer(
                buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&all_instances),
            );

            // --- 6. Run Render Pass ---
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
                    multiview_mask: None,
                });
                shadow_pass.set_pipeline(&shadow_pipeline.pipeline);
                shadow_pass.set_bind_group(0, &shadow_pipeline.bind_group, &[offset]);
                shadow_pass.set_bind_group(
                    1,
                    self.render_constants_bind_group.as_ref().unwrap(),
                    &[],
                );

                // Bind the one persistent buffer
                shadow_pass.set_vertex_buffer(1, buffer.as_ref().unwrap().slice(..));

                // Draw all batches
                for ((mesh_id, lod_index), (instance_offset, instance_count)) in &batch_info {
                    if let Some(mesh) = self.meshes.get(mesh_id) {
                        let lod = &mesh.lods[*lod_index];

                        shadow_pass.set_vertex_buffer(0, lod.vertex_buffer.slice(..));
                        shadow_pass.set_index_buffer(
                            lod.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );

                        let instance_range = *instance_offset..(*instance_offset + *instance_count);
                        shadow_pass.draw_indexed(0..lod.index_count, 0, instance_range);
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
            cascade_count: NUM_CASCADES as u32,
            _pad0: [0; 3],
            _pad1: [0; 4],
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
            _pad_light: [0; 4],
            prev_view_proj: [[0.0; 4]; 4], // PLACEHOLDER
            frame_index: 0,                // PLACEHOLDER,
            _pad_after_frame: [0; 3],
            _padding: [0; 3],
            _pad_after_padding: 0,
            _pad_end: [0; 4],
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
}

impl RenderTrait for ForwardRendererPMU {
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            let (tex, view) = Self::create_depth_texture(&self.device, &self.surface_config);
            self.main_depth_texture = Some(tex);
            self.main_depth_texture_view = Some(view);
            self.create_sky_bind_groups();
        }
    }

    fn render(&mut self) -> Result<(), RendererError> {
        let Some(ref render_data) = self.current_render_data.as_ref() else {
            return Ok(());
        };

        let output_frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost) => {
                self.resize(self.window_size);
                return Ok(());
            }
            Err(e) => return Err(RendererError::ResourceCreation(e.to_string())),
        };
        let output_view = output_frame.texture.create_view(&Default::default());

        // --- Alpha Calculation for Interpolation ---
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

        // --- Update Shared Buffers (Camera, Lights) ---
        self.update_uniforms(render_data, alpha);

        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;

        // --- Sky and Atmosphere ---
        let directional_light = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional));

        let sky_cfg = &render_data.render_config.shader_constants;
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
            ground_albedo: sky_cfg.sky_ground_albedo,
            ground_brightness: sky_cfg.sky_ground_brightness,
            night_ambient_color: sky_cfg.night_ambient_color,
            sun_angular_radius_cos: sky_cfg.sun_angular_radius_cos,
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
                label: Some("Forward Command Encoder"),
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

        // --- SHADOW PASS ---
        let camera_transform = &render_data.current_camera_transform;
        let eye = camera_transform.position;
        let forward = camera_transform.forward();
        let up = camera_transform.up();

        if render_data.render_config.shadow_pass {
            let static_camera_view = Mat4::look_at_rh(eye, eye + forward, up);
            self.run_shadow_pass(&mut encoder, render_data, &static_camera_view, alpha);
        }

        // --- 1. Build CPU-side instance data and batch info ---
        // (Key is (mesh_id, lod_index, material_id))
        // (Value is (instance_offset, instance_count))
        let mut batch_info: HashMap<(usize, usize, usize), (u32, u32)> = HashMap::new();
        let mut all_instances: Vec<InstanceRaw> = Vec::new();

        let mut sorted_objects = render_data.objects.iter().collect::<Vec<_>>();
        sorted_objects.sort_by_key(|obj| (obj.mesh_id, obj.lod_index, obj.material_id));

        for object in sorted_objects {
            if let (Some(mesh), true) = (
                self.meshes.get(&object.mesh_id),
                self.materials.contains_key(&object.material_id),
            ) {
                if mesh.lods.is_empty() {
                    continue;
                }
                let lod_index = object.lod_index.min(mesh.lods.len() - 1);
                let key = (object.mesh_id, lod_index, object.material_id);

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

                let instance_data = InstanceRaw {
                    model_matrix: model_matrix.to_cols_array_2d(),
                };

                let current_offset = all_instances.len() as u32;
                all_instances.push(instance_data);

                let entry = batch_info.entry(key).or_insert((current_offset, 0));
                entry.1 += 1;
            }
        }

        let total_instances = all_instances.len();

        if total_instances > 0 {
            // --- 2. Check and resize the GPU buffer if needed ---
            let mut capacity = self.forward_instance_capacity.borrow_mut();
            let mut buffer = self.forward_instance_buffer.borrow_mut();

            if total_instances > *capacity || buffer.is_none() {
                if let Some(old_buffer) = buffer.take() {
                    old_buffer.destroy();
                }
                let new_capacity = (total_instances as f32 * 1.5).ceil() as usize;

                *buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Forward Instance Buffer"),
                    size: (new_capacity * std::mem::size_of::<InstanceRaw>())
                        as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                *capacity = new_capacity;
            }

            // --- 3. Upload all instance data in one go ---
            self.queue.write_buffer(
                buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&all_instances),
            );
        }

        // --- 4. MAIN FORWARD PASS ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Forward Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.main_depth_texture_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if total_instances > 0 {
                pass.set_pipeline(self.forward_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, &self.scene_data_bind_groups[self.frame_index], &[]);
                pass.set_bind_group(
                    2,
                    &self.atmosphere.as_ref().unwrap().sampling_bind_group,
                    &[],
                );
                pass.set_bind_group(3, self.ibl_bind_group.as_ref().unwrap(), &[]);

                // Bind the one persistent buffer
                pass.set_vertex_buffer(
                    1,
                    self.forward_instance_buffer
                        .borrow()
                        .as_ref()
                        .unwrap()
                        .slice(..),
                );

                // --- 5. Draw all batches ---
                for ((mesh_id, lod_index, material_id), (instance_offset, instance_count)) in
                    batch_info
                {
                    if let (Some(mesh), Some(material)) =
                        (self.meshes.get(&mesh_id), self.materials.get(&material_id))
                    {
                        let lod = &mesh.lods[lod_index];

                        pass.set_bind_group(1, &material.bind_group, &[]);
                        pass.set_vertex_buffer(0, lod.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            lod.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );

                        let instance_range = instance_offset..(instance_offset + instance_count);
                        pass.draw_indexed(0..lod.index_count, 0, instance_range);
                    }
                }
            }
        }

        // --- SKY PASS ---
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
                multiview_mask: None,
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

        // --- EGUI PASS ---
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
                        multiview_mask: None,
                    });

                    self.egui_renderer.render(
                        &mut rpass.forget_lifetime(),
                        &egui_data.primitives,
                        &screen_descriptor,
                    );
                }
            }
        }

        // --- Submit and Present ---
        self.queue.submit(std::iter::once(encoder.finish()));
        output_frame.present();
        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }

    fn process_message(&mut self, message: RenderMessage) {
        match message {
            RenderMessage::CreateTexture {
                id,
                data,
                format,
                width,
                height,
                ..
            } => {
                let is_compressed = format.is_compressed();
                let full_mip_levels = calc_mip_level_count(width, height);
                let (uploads, used_bytes) =
                    build_mip_uploads(format, width, height, data.as_ref().len(), full_mip_levels);
                if uploads.is_empty() {
                    warn!("Forward PMU texture upload missing mip data.");
                    return;
                }
                if used_bytes != data.as_ref().len() {
                    warn!(
                        "Forward PMU texture upload size mismatch (used {}, provided {}).",
                        used_bytes,
                        data.as_ref().len()
                    );
                }
                let provided_levels = uploads.len() as u32;
                let mut mip_level_count = full_mip_levels;
                if is_compressed && provided_levels < full_mip_levels {
                    warn!(
                        "Compressed forward PMU texture missing mip data ({} of {}).",
                        provided_levels, full_mip_levels
                    );
                    mip_level_count = provided_levels.max(1);
                }

                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    format,
                    usage: {
                        let mut usage =
                            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
                        if !is_compressed {
                            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
                        }
                        usage
                    },
                    label: None,
                    mip_level_count,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    view_formats: &[],
                });

                for upload in &uploads {
                    let end = upload.offset + upload.size;
                    self.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: upload.level,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &data.as_ref()[upload.offset..end],
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(upload.bytes_per_row),
                            rows_per_image: Some(upload.rows_per_image),
                        },
                        wgpu::Extent3d {
                            width: upload.width,
                            height: upload.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }

                if !is_compressed && mip_level_count > 1 && provided_levels < mip_level_count {
                    let start_level = provided_levels.saturating_sub(1);
                    let mut mip_encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Mipmap Generation Encoder"),
                            });

                    self.mipmap_generator.generate_mips_from(
                        &mut mip_encoder,
                        &self.device,
                        &texture,
                        start_level,
                        mip_level_count,
                    );

                    // Submit the mip generation commands immediately
                    self.queue.submit(std::iter::once(mip_encoder.finish()));
                }

                self.loaded_texture_views
                    .insert(id, Arc::new(texture.create_view(&Default::default())));
            }
            RenderMessage::CreateMaterial(mat_data) => self.pending_materials.push(mat_data),
            RenderMessage::CreateMesh {
                id,
                total_lods: _,
                lods,
                bounds,
            } => {
                self.add_mesh(id, &lods, bounds).unwrap();
            }
            RenderMessage::RenderData(data) => self.update_render_data(data),
            RenderMessage::RenderDelta(_) => {}
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
            RenderMessage::Control(_) => {}
            RenderMessage::Resize(size) => self.resize(size),
            _ => {}
        }
    }

    fn update_render_data(&mut self, render_data: Arc<RenderData>) {
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
                } else if current_data.render_config.shadow_pass
                    != render_data.render_config.shadow_pass
                {
                    let _ = self.initialize_resources();
                    //self.resize(self.window_size);
                } else if current_data.render_config.direct_lighting_pass
                    != render_data.render_config.direct_lighting_pass
                {
                    //self.resize(self.window_size);
                } else if current_data.render_config.sky_pass != render_data.render_config.sky_pass
                {
                    //self.resize(self.window_size);
                } else if current_data.render_config.ssr_pass != render_data.render_config.ssr_pass
                {
                    //self.resize(self.window_size);
                } else if current_data.render_config.ssgi_pass
                    != render_data.render_config.ssgi_pass
                {
                    //self.resize(self.window_size);
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
        let mut i = 0;
        while i < self.pending_materials.len() {
            let mat_data = &self.pending_materials[i];

            let all_textures_loaded = mat_data
                .albedo_texture_id
                .map_or(true, |id| self.loaded_texture_views.contains_key(&id))
                && mat_data
                    .normal_texture_id
                    .map_or(true, |id| self.loaded_texture_views.contains_key(&id))
                && mat_data
                    .metallic_roughness_texture_id
                    .map_or(true, |id| self.loaded_texture_views.contains_key(&id))
                && mat_data
                    .emission_texture_id
                    .map_or(true, |id| self.loaded_texture_views.contains_key(&id));

            if all_textures_loaded {
                newly_completed.push(self.pending_materials.swap_remove(i));
            } else {
                i += 1;
            }
        }

        for mat_data in newly_completed {
            let get_view =
                |id: Option<usize>, default: &Arc<wgpu::TextureView>| -> Arc<wgpu::TextureView> {
                    id.and_then(|i| self.loaded_texture_views.get(&i))
                        .cloned()
                        .unwrap_or_else(|| default.clone())
                };
            let albedo_view = get_view(
                mat_data.albedo_texture_id,
                self.default_albedo_view.as_ref().unwrap(),
            );
            let normal_view = get_view(
                mat_data.normal_texture_id,
                self.default_normal_view.as_ref().unwrap(),
            );
            let mr_view = get_view(
                mat_data.metallic_roughness_texture_id,
                self.default_mr_view.as_ref().unwrap(),
            );
            let emission_view = get_view(
                mat_data.emission_texture_id,
                self.default_emission_view.as_ref().unwrap(),
            );

            let material_uniforms = MaterialUniforms {
                albedo: mat_data.albedo,
                emission_color: mat_data.emission_color,
                metallic: mat_data.metallic,
                roughness: mat_data.roughness,
                ao: mat_data.ao,
                emission_strength: mat_data.emission_strength,
                _padding: 0.0,
            };

            let uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Material Uniform Buffer"),
                        contents: bytemuck::bytes_of(&material_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.material_bind_group_layout.as_ref().unwrap(),
                label: Some("Forward Material Bind Group"),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(
                            self.pbr_sampler.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&mr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&emission_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });
            self.materials.insert(
                mat_data.id,
                MaterialLowEnd {
                    bind_group,
                    uniform_buffer,
                },
            );
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
