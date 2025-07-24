use crate::graphics::renderer::error::RendererError;
use crate::graphics::renderer::renderer::{
    Aabb, CASCADE_SPLITS, CameraUniforms, CascadeUniform, FRAMES_IN_FLIGHT, LightData, MAX_LIGHTS,
    Mesh, ModelPushConstant, NUM_CASCADES, PbrConstants, RenderData, RenderObject, RenderTrait,
    SHADOW_MAP_RESOLUTION, ShadowPipeline, ShadowUniforms, Vertex,
};
use crate::provided::components::{Camera, LightType};
use crate::runtime::asset_server::MaterialGpuData;
use crate::runtime::runtime::RenderMessage;
use glam::{Mat4, Quat, Vec3, Vec4, Vec4Swizzles};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::info;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

const WGPU_CLIP_SPACE_CORRECTION: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.5, 0.0),
    Vec4::new(0.0, 0.0, 0.5, 1.0),
);

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

    // Forward-Specific Resources
    forward_pipeline: Option<wgpu::RenderPipeline>,
    scene_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    material_bind_group_layout: Option<wgpu::BindGroupLayout>,
    scene_data_bind_groups: Vec<wgpu::BindGroup>,
    materials: HashMap<usize, MaterialLowEnd>,

    // IBL Resources
    ibl_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ibl_bind_group: Option<wgpu::BindGroup>,
    ibl_sampler: Option<wgpu::Sampler>,
    brdf_lut_sampler: Option<wgpu::Sampler>,
    brdf_lut_view: Option<wgpu::TextureView>,
    irradiance_map_view: Option<wgpu::TextureView>,
    prefiltered_env_map_view: Option<wgpu::TextureView>,

    // Texture Management
    loaded_texture_views: HashMap<usize, Arc<wgpu::TextureView>>,
    default_albedo_view: Option<Arc<wgpu::TextureView>>,
    default_normal_view: Option<Arc<wgpu::TextureView>>,
    default_mr_view: Option<Arc<wgpu::TextureView>>,
    default_emission_view: Option<Arc<wgpu::TextureView>>,

    // State
    window_size: PhysicalSize<u32>,
    frame_index: usize,
    pending_materials: Vec<MaterialGpuData>,
    current_render_data: Option<RenderData>,
    logic_frame_duration: Duration,
    last_timestamp: Option<Instant>,
}

impl ForwardRendererPMU {
    pub async fn new(
        _instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        size: PhysicalSize<u32>,
        target_tickrate: f32,
    ) -> Result<Self, RendererError> {
        let required_features = wgpu::Features::PUSH_CONSTANTS | wgpu::Features::FLOAT32_FILTERABLE;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Forward Renderer Device"),
                required_features,
                required_limits: wgpu::Limits {
                    max_push_constant_size: std::mem::size_of::<PbrConstants>() as u32,
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

        let mut renderer = Self {
            device,
            queue,
            surface,
            surface_config,
            window_size: size,
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
            materials: HashMap::new(),
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
            loaded_texture_views: HashMap::new(),
            default_albedo_view: None,
            default_normal_view: None,
            default_mr_view: None,
            default_emission_view: None,
            frame_index: 0,
            pending_materials: Vec::new(),
            current_render_data: None,
            logic_frame_duration: Duration::from_secs_f32(1.0 / target_tickrate),
            last_timestamp: None,
        };

        renderer.initialize_resources()?;

        info!("initialized forward (PMU) renderer");
        Ok(renderer)
    }

    fn initialize_resources(&mut self) -> Result<(), RendererError> {
        self.create_samplers();
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
            ..Default::default()
        }));
        self.ibl_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
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
                    self.ibl_bind_group_layout.as_ref().unwrap(),
                ],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX,
                    range: 0..std::mem::size_of::<PbrConstants>() as u32,
                }],
            });
        self.forward_pipeline = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Forward Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
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
                multiview: None,
                cache: None,
            },
        ));
        self.create_shadow_pipeline();
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
            format: wgpu::TextureFormat::Rg16Float,
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
            multiview: None,
            cache: None,
        });

        self.shadow_pipeline = Some(ShadowPipeline {
            pipeline,
            bind_group,
        });
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
                    culled_objects.insert(object.id, distance_sq > MAX_SHADOW_CASTING_DISTANCE_SQ);
                }
            }

            let mut scene_bounds_min = Vec3::splat(f32::MAX);
            let mut scene_bounds_max = Vec3::splat(f32::MIN);
            for object in &render_data.objects {
                if object.casts_shadow && !*culled_objects.get(&object.id).unwrap_or(&true) {
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
                for object in &render_data.objects {
                    if object.casts_shadow && !*culled_objects.get(&object.id).unwrap_or(&true) {
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

        self.update_uniforms(render_data, alpha);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Forward Command Encoder"),
            });

        let camera_transform = &render_data.current_camera_transform;
        let eye = camera_transform.position;
        let forward = camera_transform.forward();
        let up = camera_transform.up();
        let static_camera_view = Mat4::look_at_rh(eye, eye + forward, up);
        self.run_shadow_pass(&mut encoder, render_data, &static_camera_view, alpha);

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Forward Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
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
            });

            pass.set_pipeline(self.forward_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, &self.scene_data_bind_groups[self.frame_index], &[]);
            pass.set_bind_group(2, self.ibl_bind_group.as_ref().unwrap(), &[]);

            let mut batched_objects: HashMap<usize, Vec<&RenderObject>> = HashMap::new();
            for object in &render_data.objects {
                batched_objects
                    .entry(object.material_id)
                    .or_default()
                    .push(object);
            }

            for (material_id, objects) in &batched_objects {
                if let Some(material) = self.materials.get(material_id) {
                    pass.set_bind_group(1, &material.bind_group, &[]);
                    for object in objects {
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
                            let push_constants = PbrConstants {
                                model_matrix: model_matrix.to_cols_array_2d(),
                                material_id: *material_id as u32,
                                _p: [0; 3],
                            };
                            pass.set_push_constants(
                                wgpu::ShaderStages::VERTEX,
                                0,
                                bytemuck::bytes_of(&push_constants),
                            );
                            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
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
            RenderMessage::CreateTexture {
                id,
                data,
                format,
                width,
                height,
                ..
            } => {
                let texture = self.device.create_texture_with_data(
                    &self.queue,
                    &wgpu::TextureDescriptor {
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        label: None,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &data,
                );
                self.loaded_texture_views
                    .insert(id, Arc::new(texture.create_view(&Default::default())));
            }
            RenderMessage::CreateMaterial(mat_data) => self.pending_materials.push(mat_data),
            RenderMessage::CreateMesh {
                id,
                vertices,
                indices,
                bounds,
            } => {
                let vbo = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                let ibo = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                self.meshes.insert(
                    id,
                    Mesh {
                        vertex_buffer: vbo,
                        index_buffer: ibo,
                        index_count: indices.len() as u32,
                        bounds,
                    },
                );
            }
            RenderMessage::RenderData(data) => self.update_render_data(data),
            RenderMessage::Resize(size) => self.resize(size),
            _ => {}
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
