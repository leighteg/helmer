use crate::{
    graphics::renderer::error::RendererError,
    provided::components::{LightType, Transform},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use std::sync::Arc;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tracing::info;
use wgpu::{Adapter, util::DeviceExt}; // Import for create_buffer_init
use winit::window::Window;

// --- CONSTANTS (Unchanged) ---
const MAX_TEXTURE_COUNT: u32 = 256;
const TEXTURE_RESOLUTION: u32 = 1024;
const MAX_LIGHTS: usize = 32;
const FRAMES_IN_FLIGHT: usize = 3;

// --- RESOURCE STRUCTS (wgpu types) ---

/// Represents a loaded mesh with its vertex and index buffers.
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

pub struct TextureManager {
    pub textures: HashMap<String, usize>,
    pub next_texture_index: usize,
}

impl Default for TextureManager {
    fn default() -> Self {
        Self {
            textures: HashMap::new(),
            next_texture_index: 1, // Reserve index 0 for default textures
        }
    }
}

// Material struct remains the same logically.
#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub albedo_texture_index: i32,
    pub normal_texture_index: i32,
    pub metallic_roughness_texture_index: i32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            ao: 1.0,
            albedo_texture_index: 0,
            normal_texture_index: 0,
            metallic_roughness_texture_index: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderObject {
    pub previous_transform: Transform, // last logic tick
    pub current_transform: Transform,  // current logic tick
    pub mesh_id: usize,
    pub material_id: usize,
}

#[derive(Debug, Clone)]
pub struct RenderLight {
    pub previous_transform: Transform,
    pub current_transform: Transform,
    pub color: [f32; 3],
    pub intensity: f32,
    pub light_type: LightType,
}

pub struct RenderData {
    pub objects: Vec<RenderObject>,
    pub lights: Vec<RenderLight>,
    pub previous_camera_transform: Transform,
    pub current_camera_transform: Transform,
}

// --- SHADER DATA STRUCTS (bytemuck, no mev) ---

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CameraUniforms {
    view_matrix: [[f32; 4]; 4],
    projection_matrix: [[f32; 4]; 4],
    view_position: [f32; 3],
    light_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LightData {
    position: [f32; 3],
    light_type: u32,
    color: [f32; 3],
    intensity: f32,
    direction: [f32; 3],
    _padding: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PbrConstants {
    model_matrix: [[f32; 4]; 4],
    material_id: u32,
    _p: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MaterialShaderData {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    ao: f32,
    _p1: f32,
    albedo_texture_index: i32,
    normal_texture_index: i32,
    metallic_roughness_texture_index: i32,
    _p2: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
    pub tangent: [f32; 3],
}

impl Vertex {
    // CHANGE: Add a descriptor function for wgpu pipeline creation.
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // normal
        2 => Float32x2, // tex_coord
        3 => Float32x3, // tangent
    ];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

// --- MAIN RENDERER STRUCT (wgpu) ---

pub struct Renderer {
    // CHANGE: wgpu core components
    instance: wgpu::Instance,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>, // Use 'static lifetime with Arc<Window>
    surface_config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,

    // CHANGE: wgpu resources
    pipeline: Option<wgpu::RenderPipeline>,
    pbr_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pbr_bind_groups: Vec<wgpu::BindGroup>,
    depth_texture: Option<wgpu::Texture>,
    depth_texture_view: Option<wgpu::TextureView>,

    camera_buffers: Vec<wgpu::Buffer>,
    lights_buffers: Vec<wgpu::Buffer>,
    material_uniform_buffer: Option<wgpu::Buffer>,

    meshes: HashMap<usize, Mesh>,
    materials: HashMap<usize, Material>,
    texture_manager: TextureManager,

    albedo_texture_array: Option<wgpu::Texture>,
    normal_texture_array: Option<wgpu::Texture>,
    metallic_roughness_texture_array: Option<wgpu::Texture>,
    sampler: Option<wgpu::Sampler>,

    frame_index: usize,
    current_render_data: Option<RenderData>,

    /// The fixed duration of a single logic update (e.g., 1.0 / 60.0).
    logic_frame_duration: Duration,
    /// The wall-clock time when the last `RenderData` packet was received.
    last_logic_update_time: Instant,
}

impl Renderer {
    // CHANGE: `new` is now async due to wgpu's adapter/device request flow.
    pub async fn new(window: Arc<Window>, logic_fps: f32) -> Result<Self, RendererError> {
        let size = window.inner_size();

        // --- Instance, Adapter, Device, Queue ---
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(target_os = "windows")]
            backends: wgpu::Backends::DX12,
            #[cfg(not(target_os = "windows"))]
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // The surface needs to live as long as the window that created it.
        // `Arc<Window>` ensures safety.
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter: Adapter;
        match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(new_adapter) => adapter = new_adapter,
            Err(err) => {
                return Err(RendererError::ResourceCreation(format!(
                    "Failed to find a suitable GPU adapter: {}",
                    err
                )));
            }
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    // Add push constant limit
                    max_push_constant_size: std::mem::size_of::<PbrConstants>() as u32,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mut instance = Self {
            instance,
            device: Arc::new(device),
            queue,
            surface,
            surface_config,
            window,
            pipeline: None,
            pbr_bind_group_layout: None,
            pbr_bind_groups: Vec::new(),
            depth_texture: None,
            depth_texture_view: None,
            camera_buffers: Vec::new(),
            lights_buffers: Vec::new(),
            material_uniform_buffer: None,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            texture_manager: TextureManager::default(),
            albedo_texture_array: None,
            normal_texture_array: None,
            metallic_roughness_texture_array: None,
            sampler: None,
            frame_index: 0,
            current_render_data: None,
            logic_frame_duration: Duration::from_secs_f32(1.0 / logic_fps),
            last_logic_update_time: Instant::now(),
        };

        instance.initialize_resources().unwrap();
        info!("initialized renderer");

        return Ok(instance);
    }

    // --- Resource Initialization ---
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

        // --- Create Textures and Sampler ---
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Texture Array"),
            size: wgpu::Extent3d {
                width: TEXTURE_RESOLUTION,
                height: TEXTURE_RESOLUTION,
                depth_or_array_layers: MAX_TEXTURE_COUNT,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Usage must include COPY_DST to be a destination for copy operations.
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        };

        self.albedo_texture_array = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("albedo-texture-array"),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            ..texture_desc
        }));

        self.normal_texture_array = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normal-texture-array"),
            format: wgpu::TextureFormat::Rgba8Unorm,
            ..texture_desc
        }));

        self.metallic_roughness_texture_array =
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("metallic-roughness-texture-array"),
                format: wgpu::TextureFormat::Rgba8Unorm,
                ..texture_desc
            }));

        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // --- Upload Default Texture Data ---
        self.upload_default_textures();
        self.create_depth_texture();
        self.create_pbr_bind_groups();
        self.create_pipeline();

        Ok(())
    }

    // Helper for uploading default textures
    fn upload_default_textures(&self) {
        let resolution_area = (TEXTURE_RESOLUTION * TEXTURE_RESOLUTION) as usize;

        // Albedo (White)
        let white_pixel = [255u8, 255, 255, 255];
        let default_albedo_data: Vec<u8> = white_pixel
            .iter()
            .cycle()
            .take(resolution_area * 4)
            .copied()
            .collect();
        self.upload_texture_slice(
            &default_albedo_data,
            self.albedo_texture_array.as_ref().unwrap(),
            0,
        );

        // Normal (Flat)
        let flat_normal_pixel = [128u8, 128, 255, 255]; // Represents (0, 0, 1)
        let default_normal_data: Vec<u8> = flat_normal_pixel
            .iter()
            .cycle()
            .take(resolution_area * 4)
            .copied()
            .collect();
        self.upload_texture_slice(
            &default_normal_data,
            self.normal_texture_array.as_ref().unwrap(),
            0,
        );

        // Metallic/Roughness (Black/Grey)
        let default_mr_pixel = [255u8, 204, 0, 255]; // R=AO, G=Roughness, B=Metallic
        let default_mr_data: Vec<u8> = default_mr_pixel
            .iter()
            .cycle()
            .take(resolution_area * 4)
            .copied()
            .collect();
        self.upload_texture_slice(
            &default_mr_data,
            self.metallic_roughness_texture_array.as_ref().unwrap(),
            0,
        );
    }

    // Helper to upload a single texture slice
    fn upload_texture_slice(&self, data: &[u8], target_array: &wgpu::Texture, layer_index: u32) {
        let bytes_per_pixel = 4u32;
        let bytes_per_row = bytes_per_pixel * TEXTURE_RESOLUTION;

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
                rows_per_image: Some(TEXTURE_RESOLUTION),
            },
            wgpu::Extent3d {
                width: TEXTURE_RESOLUTION,
                height: TEXTURE_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );
    }

    // --- Public Resource Management ---

    pub fn add_texture(
        &mut self,
        name: &str,
        data: &[u8],
        target_array: &wgpu::Texture, // Pass the specific array to add to
    ) -> Result<usize, RendererError> {
        if let Some(index) = self.texture_manager.textures.get(name) {
            return Ok(*index);
        }
        if self.texture_manager.next_texture_index >= MAX_TEXTURE_COUNT as usize {
            return Err(RendererError::ResourceCreation(
                "Texture array is full".to_string(),
            ));
        }

        let index = self.texture_manager.next_texture_index;
        self.texture_manager.next_texture_index += 1;

        self.upload_texture_slice(data, target_array, index as u32);

        self.texture_manager
            .textures
            .insert(name.to_string(), index);
        Ok(index)
    }

    pub fn add_mesh(
        &mut self,
        id: usize,
        vertices: &[Vertex],
        indices: &[u32],
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

        let shader_data = MaterialShaderData {
            albedo: material.albedo,
            metallic: material.metallic,
            roughness: material.roughness,
            ao: material.ao,
            _p1: 0.0,
            albedo_texture_index: material.albedo_texture_index,
            normal_texture_index: material.normal_texture_index,
            metallic_roughness_texture_index: material.metallic_roughness_texture_index,
            _p2: 0,
        };

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

    // --- Pipeline and Bind Group Creation ---

    fn create_pbr_bind_groups(&mut self) {
        let device = &self.device;
        let albedo_view = self
            .albedo_texture_array
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let normal_view = self
            .normal_texture_array
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mr_view = self
            .metallic_roughness_texture_array
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PBR Bind Group Layout"),
            entries: &[
                // Camera Uniforms
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
                // Lights Buffer
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
                // Materials Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Albedo Texture Array
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
                // Normal Texture Array
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Metallic/Roughness Texture Array
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        self.pbr_bind_groups = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("PBR Bind Group {}", i)),
                    layout: &bind_group_layout,
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
                            resource: self
                                .material_uniform_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&albedo_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&mr_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(
                                self.sampler.as_ref().unwrap(),
                            ),
                        },
                    ],
                })
            })
            .collect();

        self.pbr_bind_group_layout = Some(bind_group_layout);
    }

    fn create_depth_texture(&mut self) {
        let depth_texture_desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = self.device.create_texture(&depth_texture_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_texture = Some(texture);
        self.depth_texture_view = Some(view);
    }

    fn create_pipeline(&mut self) {
        let device = &self.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/pbr.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[self.pbr_bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..std::mem::size_of::<PbrConstants>() as u32,
            }],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR Pipeline"),
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
                    format: self.surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.pipeline = Some(pipeline);
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            // Re-create depth texture with the new size
            self.create_depth_texture();
        }
    }

    pub fn update_render_data(&mut self, render_data: RenderData) {
        self.current_render_data = Some(render_data);

        self.last_logic_update_time = Instant::now();
    }

    // --- RENDER FUNCTION ---
    pub fn render(&mut self) -> Result<(), RendererError> {
        if self.pipeline.is_none() {
            self.initialize_resources()?;
        }

        let Some(ref render_data) = self.current_render_data else {
            return Ok(());
        };

        let time_since_update = self.last_logic_update_time.elapsed();
        let alpha = (time_since_update.as_secs_f32() / self.logic_frame_duration.as_secs_f32())
            .clamp(0.0, 1.0);

        // --- Get Frame and Encoder ---
        let output_frame: wgpu::SurfaceTexture;
        match self.surface.get_current_texture() {
            Ok(new_frame) => output_frame = new_frame,
            Err(e) => {
                // Handle surface errors (e.g., window resized, timeout)
                match e {
                    wgpu::SurfaceError::Lost => {
                        self.resize(self.window.inner_size()); // Reconfigure surface
                    }
                    wgpu::SurfaceError::OutOfMemory => {
                        // Catastrophic error, should probably panic.
                        return Err(RendererError::ResourceCreation(
                            "WGPU Surface: Out of Memory".to_string(),
                        ));
                    }
                    _ => (), // Other errors like Outdated/Timeout can be ignored for now.
                }
                return Err(RendererError::ResourceCreation(format!(
                    "Failed to acquire next swap chain texture: {}",
                    e
                )));
            }
        };

        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // --- Update Uniforms ---
        // --- CAMERA INTERPOLATION ---
        let prev_cam = &render_data.previous_camera_transform;
        let curr_cam = &render_data.current_camera_transform;
        let eye = prev_cam.position.lerp(curr_cam.position, alpha);
        let rotation = Quat::from(prev_cam.rotation).slerp(curr_cam.rotation, alpha);
        let forward = rotation * Vec3::Z;
        let up = rotation * Vec3::Y;
        let view_matrix = Mat4::look_at_rh(eye, eye + forward, up);
        let projection_matrix = Mat4::perspective_rh(
            (45.0_f32).to_radians(),
            self.surface_config.width as f32 / self.surface_config.height as f32,
            0.1,
            1000.0,
        );
        let camera_uniforms = CameraUniforms {
            view_matrix: view_matrix.to_cols_array_2d(),
            projection_matrix: projection_matrix.to_cols_array_2d(),
            view_position: eye.to_array(),
            light_count: render_data.lights.len() as u32,
        };
        self.queue.write_buffer(
            &self.camera_buffers[self.frame_index],
            0,
            bytemuck::bytes_of(&camera_uniforms),
        );

        // --- LIGHT INTERPOLATION ---
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

        // --- Command Encoding ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Command Encoder"),
            });

        {
            // Scoped to release borrow of encoder
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_texture_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            render_pass.set_pipeline(self.pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, &self.pbr_bind_groups[self.frame_index], &[]);

            for object in &render_data.objects {
                let Some(mesh) = self.meshes.get(&object.mesh_id) else {
                    continue;
                };
                if !self.materials.contains_key(&object.material_id) {
                    continue;
                };

                // Interpolate position, rotation, and scale for this specific object.
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

                let constants = PbrConstants {
                    model_matrix: model_matrix.to_cols_array_2d(),
                    material_id: object.material_id as u32,
                    _p: [0; 3],
                };

                render_pass.set_push_constants(
                    wgpu::ShaderStages::VERTEX,
                    0,
                    bytemuck::bytes_of(&constants),
                );

                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        // --- Submit and Present ---
        self.queue.submit(std::iter::once(encoder.finish()));
        output_frame.present();

        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;

        Ok(())
    }
}
