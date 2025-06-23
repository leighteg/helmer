use crate::{
    graphics::renderer::error::RendererError,
    provided::components::{Camera, LightType, Transform},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3, Vec4, Vec4Swizzles, vec4};
use std::sync::Arc;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tracing::info;
use wgpu::{Adapter, util::DeviceExt}; // Import for create_buffer_init
use winit::window::Window;

#[derive(Copy, Clone)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn get_corners(&self) -> [Vec3; 8] {
        [
            self.min,
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            self.max,
        ]
    }
}

// --- CONSTANTS ---
const MAX_TEXTURE_COUNT: u32 = 256;
const TEXTURE_RESOLUTION: u32 = 1024;
const MAX_LIGHTS: usize = 32;
const SHADOW_MAP_RESOLUTION: u32 = 2048;
const NUM_CASCADES: usize = 4;
const FRAMES_IN_FLIGHT: usize = 3;
const SCENE_BOUNDS: Aabb = Aabb {
    min: Vec3::new(-50.0, -10.0, -50.0),
    max: Vec3::new(50.0, 50.0, 50.0),
};

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

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub emission_strength: f32,
    pub emission_color: [f32; 3],
    pub albedo_texture_index: i32,
    pub normal_texture_index: i32,
    pub metallic_roughness_texture_index: i32,
    pub emission_texture_index: i32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            ao: 1.0,
            emission_strength: 0.0,
            emission_color: [0.0, 0.0, 0.0],
            albedo_texture_index: 0,
            normal_texture_index: 0,
            metallic_roughness_texture_index: 0,
            emission_texture_index: -1, // -1 indicates no emission texture
        }
    }
}

impl Material {
    /// Creates a new emissive material with the specified emission properties
    pub fn with_emission(
        mut self,
        emission_color: [f32; 3],
        emission_strength: f32,
        emission_texture_index: Option<i32>,
    ) -> Self {
        self.emission_color = emission_color;
        self.emission_strength = emission_strength;
        self.emission_texture_index = emission_texture_index.unwrap_or(-1);
        self
    }

    /// Sets the emission properties of the material
    pub fn set_emission(
        &mut self,
        emission_color: [f32; 3],
        emission_strength: f32,
        emission_texture_index: Option<i32>,
    ) {
        self.emission_color = emission_color;
        self.emission_strength = emission_strength;
        self.emission_texture_index = emission_texture_index.unwrap_or(-1);
    }

    /// Returns true if this material has any emission
    pub fn is_emissive(&self) -> bool {
        self.emission_strength > 0.0
            && (self.emission_color[0] > 0.0
                || self.emission_color[1] > 0.0
                || self.emission_color[2] > 0.0
                || self.emission_texture_index >= 0)
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
    pub camera_component: Camera,
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
    emission_strength: f32,
    albedo_texture_index: i32,
    normal_texture_index: i32,
    metallic_roughness_texture_index: i32,
    emission_texture_index: i32,
    emission_color: [f32; 3],
    _padding: f32,
}

impl From<&Material> for MaterialShaderData {
    fn from(material: &Material) -> Self {
        Self {
            albedo: material.albedo,
            metallic: material.metallic,
            roughness: material.roughness,
            ao: material.ao,
            emission_strength: material.emission_strength,
            albedo_texture_index: material.albedo_texture_index,
            normal_texture_index: material.normal_texture_index,
            metallic_roughness_texture_index: material.metallic_roughness_texture_index,
            emission_texture_index: material.emission_texture_index,
            emission_color: material.emission_color,
            _padding: 0.0,
        }
    }
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CascadeUniform {
    light_view_proj: [[f32; 4]; 4],
    split_depth: f32,
    _padding: [f32; 3], // Ensure alignment
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShadowUniforms {
    cascades: [CascadeUniform; NUM_CASCADES],
}

// Helper struct to group shadow resources
struct ShadowPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    light_vp_buffer: wgpu::Buffer,
    light_vp_bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ModelPushConstant {
    model_matrix: [[f32; 4]; 4],
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
    emission_texture_array: Option<wgpu::Texture>,
    sampler: Option<wgpu::Sampler>,

    shadow_pipeline: Option<ShadowPipeline>,
    shadow_map_texture: Option<wgpu::Texture>,
    shadow_map_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    shadow_depth_texture: Option<wgpu::Texture>, // For depth testing during shadow pass
    shadow_depth_view: Option<wgpu::TextureView>,
    shadow_uniforms_buffer: Option<wgpu::Buffer>,

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
                required_features: wgpu::Features::PUSH_CONSTANTS
                    | wgpu::Features::FLOAT32_FILTERABLE,
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
            present_mode: wgpu::PresentMode::Fifo,
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
            emission_texture_array: None,
            sampler: None,
            shadow_pipeline: None,
            shadow_map_texture: None,
            shadow_map_view: None,
            shadow_sampler: None,
            shadow_depth_texture: None,
            shadow_depth_view: None,
            shadow_uniforms_buffer: None,
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

        self.emission_texture_array = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("emission-texture-array"),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
        self.create_shadow_resources();
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

        let default_emission_data: Vec<u8> = white_pixel
            .iter()
            .cycle()
            .take(resolution_area * 4)
            .copied()
            .collect();
        self.upload_texture_slice(
            &default_emission_data,
            self.emission_texture_array.as_ref().unwrap(),
            0,
        );
    }

    fn create_shadow_resources(&mut self) {
        let device = &self.device;

        // 1. Shadow Map Texture (VSM stores depth and depth^2 in a color texture)
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
            format: wgpu::TextureFormat::Rg32Float, // Two 32-bit floats for depth and depth^2
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

        // 2. Sampler for the shadow map
        self.shadow_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // 3. Depth texture for the shadow pass itself
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

        // 4. Uniform buffer for cascade data
        self.shadow_uniforms_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Uniforms Buffer"),
            size: std::mem::size_of::<ShadowUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // 5. Create the shadow pipeline
        self.create_shadow_pipeline();
    }

    fn create_shadow_pipeline(&mut self) {
        let device = &self.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shadow.wgsl"));

        let light_vp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Light VP Buffer"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let light_vp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Light VP Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_vp_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..std::mem::size_of::<ModelPushConstant>() as u32, // Only model matrix
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
                cull_mode: Some(wgpu::Face::Front), // "Peter Panning" fix
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    // Another "shadow acne" fix
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
            bind_group_layout,
            light_vp_buffer,
            light_vp_bind_group,
        });
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
        let emission_view = self
            .emission_texture_array
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_map_view = self.shadow_map_view.as_ref().unwrap();
        let shadow_sampler = self.shadow_sampler.as_ref().unwrap();

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
                // Emission Texture Array
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // shadow map view
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // shadow sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // shadow uniforms buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, // Vertex for view-space depth
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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
                            resource: wgpu::BindingResource::TextureView(&emission_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Sampler(
                                self.sampler.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(shadow_map_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::Sampler(shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
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
        let camera = &render_data.camera_component;

        let prev_cam = &render_data.previous_camera_transform;
        let curr_cam = &render_data.current_camera_transform;
        let eye = prev_cam.position.lerp(curr_cam.position, alpha);
        let rotation = Quat::from(prev_cam.rotation).slerp(curr_cam.rotation, alpha);
        let forward = rotation * Vec3::Z;
        let up = rotation * Vec3::Y;
        let view_matrix = Mat4::look_at_rh(eye, eye + forward, up);
        let projection_matrix = Mat4::perspective_rh(
            camera.fov_y_rad,
            camera.aspect_ratio,
            camera.near_plane,
            camera.far_plane,
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

        // Find the primary directional light to cast shadows from
        let shadow_light = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional));

        if let Some(light) = shadow_light {
            let shadow_pipeline = self.shadow_pipeline.as_ref().unwrap();

            // 1. Calculate Cascade View-Projection Matrices
            let shadow_uniforms = self.calculate_cascades(
                &projection_matrix,
                &view_matrix,
                light.current_transform.rotation,
            );
            self.queue.write_buffer(
                self.shadow_uniforms_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&shadow_uniforms),
            );

            // 2. Perform Shadow Render Pass for each cascade
            let shadow_depth_view = self.shadow_depth_view.as_ref().unwrap();

            for i in 0..NUM_CASCADES {
                // Update the light's VP matrix for this specific cascade
                self.queue.write_buffer(
                    &shadow_pipeline.light_vp_buffer,
                    0,
                    bytemuck::bytes_of(&shadow_uniforms.cascades[i].light_view_proj),
                );

                // Create a view for the specific cascade layer
                let cascade_view = self.shadow_map_texture.as_ref().unwrap().create_view(
                    &wgpu::TextureViewDescriptor {
                        label: Some(&format!("Cascade View {}", i)),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: i as u32,
                        array_layer_count: Some(1),
                        ..Default::default()
                    },
                );

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
                            }), // Clear to max depth
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: shadow_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                shadow_pass.set_pipeline(&shadow_pipeline.pipeline);
                shadow_pass.set_bind_group(0, &shadow_pipeline.light_vp_bind_group, &[]);

                // Render all objects into the cascade
                for object in &render_data.objects {
                    if let Some(mesh) = self.meshes.get(&object.mesh_id) {
                        let model_matrix = Mat4::from_scale_rotation_translation(
                            object.current_transform.scale,
                            object.current_transform.rotation.into(),
                            object.current_transform.position,
                        );
                        shadow_pass.set_push_constants(
                            wgpu::ShaderStages::VERTEX,
                            0,
                            bytemuck::bytes_of(&model_matrix.to_cols_array_2d()),
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

                let push_constants = PbrConstants {
                    model_matrix: model_matrix.to_cols_array_2d(),
                    material_id: object.material_id as u32,
                    _p: [0; 3],
                };

                render_pass.set_push_constants(
                    wgpu::ShaderStages::VERTEX,
                    0,
                    bytemuck::bytes_of(&push_constants),
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

    fn calculate_cascades(&self, proj: &Mat4, view: &Mat4, light_rotation: Quat) -> ShadowUniforms {
        let light_dir = (light_rotation * -Vec3::Z).normalize();
        let inv_camera_view_proj = (*proj * *view).inverse();
        let camera = &self.current_render_data.as_ref().unwrap().camera_component;

        let cascade_splits_percentages = [0.0, 0.07, 0.2, 0.5, 1.0];

        let mut uniforms = ShadowUniforms {
            cascades: [CascadeUniform {
                light_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                split_depth: 0.0,
                _padding: [0.0; 3],
            }; NUM_CASCADES],
        };

        for i in 0..NUM_CASCADES {
            let p_near_percent = cascade_splits_percentages[i];
            let p_far_percent = cascade_splits_percentages[i + 1];

            let frustum_corners_world: [Vec3; 8] = (0..8)
                .map(|n| {
                    let p_ndc = Vec3::new(
                        (n % 2) as f32 * 2.0 - 1.0,
                        (n / 2 % 2) as f32 * 2.0 - 1.0,
                        (n / 4) as f32,
                    );
                    let z = p_near_percent + (p_ndc.z * (p_far_percent - p_near_percent));
                    let p_world_h = inv_camera_view_proj * vec4(p_ndc.x, p_ndc.y, z, 1.0);
                    p_world_h.xyz() / p_world_h.w
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            let frustum_center = frustum_corners_world.iter().sum::<Vec3>() / 8.0;

            let up = if light_dir.y.abs() > 0.999 {
                Vec3::X
            } else {
                Vec3::Y
            };
            let light_view_mat = Mat4::look_at_rh(frustum_center - light_dir, frustum_center, up);

            let mut frustum_min_ls = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
            let mut frustum_max_ls = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
            for &corner in &frustum_corners_world {
                let trf = light_view_mat * corner.extend(1.0);
                frustum_min_ls = frustum_min_ls.min(trf.xyz());
                frustum_max_ls = frustum_max_ls.max(trf.xyz());
            }

            // Re-introduce the minimum cascade size check to prevent the projection from collapsing.
            // This is especially important for cascade 0 when the camera is close to objects.
            let min_cascade_size = 1.0;
            if frustum_max_ls.x - frustum_min_ls.x < min_cascade_size {
                let mid = (frustum_max_ls.x + frustum_min_ls.x) / 2.0;
                frustum_max_ls.x = mid + min_cascade_size / 2.0;
                frustum_min_ls.x = mid - min_cascade_size / 2.0;
            }
            if frustum_max_ls.y - frustum_min_ls.y < min_cascade_size {
                let mid = (frustum_max_ls.y + frustum_min_ls.y) / 2.0;
                frustum_max_ls.y = mid + min_cascade_size / 2.0;
                frustum_min_ls.y = mid - min_cascade_size / 2.0;
            }

            let scene_corners_world = SCENE_BOUNDS.get_corners();
            let mut scene_min_ls = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
            for &corner in &scene_corners_world {
                let trf = light_view_mat * corner.extend(1.0);
                scene_min_ls = scene_min_ls.min(trf.xyz());
            }

            let final_min_ls = frustum_min_ls.min(scene_min_ls);
            let final_max_ls = frustum_max_ls; // Use the expanded frustum max

            let light_proj_mat = Mat4::orthographic_rh(
                final_min_ls.x,
                final_max_ls.x,
                final_min_ls.y,
                final_max_ls.y,
                final_min_ls.z, // Use combined near plane for casters
                final_max_ls.z, // Use frustum far plane for precision
            );

            // The texel snapping / stability fixes should still be applied here if you want them.

            let split_depth =
                camera.near_plane + (camera.far_plane - camera.near_plane) * p_far_percent;
            uniforms.cascades[i] = CascadeUniform {
                light_view_proj: (light_proj_mat * light_view_mat).to_cols_array_2d(),
                split_depth: -split_depth,
                _padding: [0.0; 3],
            };
        }
        uniforms
    }
}
