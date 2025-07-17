use crate::{
    graphics::renderer::error::RendererError,
    provided::components::{Camera, LightType, Transform},
    runtime::{
        asset_server::{AssetKind, MaterialGpuData},
        runtime::RenderMessage,
    },
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3, Vec4, Vec4Swizzles, vec4};
use std::sync::Arc;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tracing::{info, warn};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

const WGPU_CLIP_SPACE_CORRECTION: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.5, 0.0),
    Vec4::new(0.0, 0.0, 0.5, 1.0),
);

#[derive(Copy, Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn calculate(vertices: &Vec<Vertex>) -> Self {
        let mut min_bounds = Vec3::splat(f32::MAX);
        let mut max_bounds = Vec3::splat(f32::MIN);

        for vertex in vertices {
            min_bounds = min_bounds.min(Vec3::from(vertex.position));
            max_bounds = max_bounds.max(Vec3::from(vertex.position));
        }

        Self {
            min: min_bounds,
            max: max_bounds,
        }
    }

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
const FRAMES_IN_FLIGHT: usize = 3;
const MAX_LIGHTS: usize = 256;
const MAX_TEXTURE_COUNT: u32 = 256;
const TEXTURE_RESOLUTION: u32 = 1024;
const SHADOW_MAP_RESOLUTION: u32 = 2048;
const NUM_CASCADES: usize = 4;
const CASCADE_SPLITS: [f32; 5] = [0.1, 15.0, 40.0, 100.0, 300.0];

// --- RESOURCE STRUCTS ---

/// Represents a loaded mesh with its vertex and index buffers.
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub bounds: Aabb,
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
    pub casts_shadow: bool,
}

#[derive(Debug, Clone)]
pub struct RenderLight {
    pub previous_transform: Transform,
    pub current_transform: Transform,
    pub color: [f32; 3],
    pub intensity: f32,
    pub light_type: LightType,
}

#[derive(Debug, Clone)]
pub struct RenderData {
    pub objects: Vec<RenderObject>,
    pub lights: Vec<RenderLight>,
    pub previous_camera_transform: Transform,
    pub current_camera_transform: Transform,
    pub camera_component: Camera,

    pub timestamp: Instant,
}

// --- SHADER DATA STRUCTS (bytemuck, no mev) ---

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CameraUniforms {
    view_matrix: [[f32; 4]; 4],
    projection_matrix: [[f32; 4]; 4],
    inverse_projection_matrix: [[f32; 4]; 4], // NEW
    inverse_view_projection_matrix: [[f32; 4]; 4],
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
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
    pub tangent: [f32; 4],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // normal
        2 => Float32x2, // tex_coord
        3 => Float32x4, // tangent
    ];

    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        tex_coord: [f32; 2],
        tangent: [f32; 4],
    ) -> Self {
        Self {
            position,
            normal,
            tex_coord,
            tangent,
        }
    }

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
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
    _padding: [f32; 3],
}

impl Default for CascadeUniform {
    fn default() -> Self {
        Self {
            light_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            split_depth: 0.0,
            _padding: [0.0; 3],
        }
    }
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

// --- MAIN RENDERER STRUCT ---

pub struct Renderer {
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

    // Bind Group Layouts
    scene_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    object_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssr_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssr_camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    lighting_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ibl_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Bind Groups
    scene_data_bind_groups: Vec<wgpu::BindGroup>,
    object_data_bind_group: Option<wgpu::BindGroup>,
    ssr_inputs_bind_group: Option<wgpu::BindGroup>,
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

    // Asset Storage
    meshes: HashMap<usize, Mesh>,
    materials: HashMap<usize, Material>,
    texture_manager: TextureManager,

    // Texture Arrays
    albedo_texture_array: Option<wgpu::Texture>,
    normal_texture_array: Option<wgpu::Texture>,
    metallic_roughness_texture_array: Option<wgpu::Texture>,
    emission_texture_array: Option<wgpu::Texture>,

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

impl Renderer {
    pub async fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        size: PhysicalSize<u32>,
        target_tickrate: f32,
    ) -> Result<Self, RendererError> {
        // --- Adapter, Device, Queue ---
        let adapter: wgpu::Adapter;
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
                    | wgpu::Features::FLOAT32_FILTERABLE
                    | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
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
            scene_data_bind_group_layout: None,
            object_data_bind_group_layout: None,
            ssr_inputs_bind_group_layout: None,
            ssr_camera_bind_group_layout: None,
            lighting_inputs_bind_group_layout: None,
            ibl_bind_group_layout: None,
            scene_data_bind_groups: Vec::new(),
            object_data_bind_group: None,
            ssr_inputs_bind_group: None,
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
            meshes: HashMap::new(),
            materials: HashMap::new(),
            texture_manager: TextureManager::default(),
            albedo_texture_array: None,
            normal_texture_array: None,
            metallic_roughness_texture_array: None,
            emission_texture_array: None,
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

        self.resize(self.window_size);

        info!("initialized renderer");
        Ok(())
    }

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
                cull_mode: Some(wgpu::Face::Front), // Use front-face culling for Peter-Panning
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
            bind_group_layout,
            light_vp_buffer,
            light_vp_bind_group,
        });
    }

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

    fn upload_uncompressed_texture_slice(
        &self,
        data: &[u8],
        target_array: &wgpu::Texture,
        layer_index: u32,
        bytes_per_pixel: u32,
    ) {
        // This helper now takes bytes_per_pixel to be more generic.
        let bytes_per_row = bytes_per_pixel * TEXTURE_RESOLUTION;
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
                rows_per_image: Some(TEXTURE_RESOLUTION),
            },
            wgpu::Extent3d {
                width: TEXTURE_RESOLUTION,
                height: TEXTURE_RESOLUTION,
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
        let width_in_blocks = TEXTURE_RESOLUTION / block_dimensions.0;
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
                rows_per_image: Some(TEXTURE_RESOLUTION / block_dimensions.1),
            },
            wgpu::Extent3d {
                width: TEXTURE_RESOLUTION,
                height: TEXTURE_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn add_texture(
        &mut self,
        name: &str,
        data: &[u8],
        kind: AssetKind,
        format: wgpu::TextureFormat,
    ) -> Result<usize, RendererError> {
        // 1. Check for duplicates first.
        if let Some(index) = self.texture_manager.textures.get(name) {
            return Ok(*index);
        }

        // 2. Check if we have space for a new texture.
        if self.texture_manager.next_texture_index >= MAX_TEXTURE_COUNT as usize {
            return Err(RendererError::ResourceCreation(
                "Texture array is full".to_string(),
            ));
        }

        // 4. Get the target array, which is now guaranteed to exist.
        let target_array = match kind {
            AssetKind::Albedo => self.albedo_texture_array.as_ref().unwrap(),
            AssetKind::Normal => self.normal_texture_array.as_ref().unwrap(),
            AssetKind::MetallicRoughness => self.metallic_roughness_texture_array.as_ref().unwrap(),
            AssetKind::Emission => self.emission_texture_array.as_ref().unwrap(),
        };

        // 5. Get the next available index and increment the manager.
        let index = self.texture_manager.next_texture_index;
        self.texture_manager.next_texture_index += 1;

        // 6. Upload the new texture data to its assigned slice in the array.
        let block_size = format.block_size(None).unwrap_or(1);
        if block_size > 1 {
            self.upload_compressed_texture_slice(data, target_array, index as u32, format);
        } else {
            let bytes_per_pixel = format.block_size(None).unwrap_or(4);
            self.upload_uncompressed_texture_slice(
                data,
                target_array,
                index as u32,
                bytes_per_pixel,
            );
        }

        // 7. Store the name-to-index mapping for future deduplication.
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
        let lighting_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/lighting.wgsl"));

        // --- Create Layouts ---
        let (
            scene_data_bind_group_layout,
            object_data_bind_group_layout,
            ssr_inputs_bind_group_layout,
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
                    stages: wgpu::ShaderStages::VERTEX,
                    range: 0..std::mem::size_of::<PbrConstants>() as u32,
                }],
            });

        let ssr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Pipeline Layout"),
            bind_group_layouts: &[&ssr_inputs_bind_group_layout, &ssr_camera_bind_group_layout],
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
                    buffer_binding(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Storage { read_only: true },
                    ),
                    texture_binding(1, true, wgpu::TextureViewDimension::D2Array, false),
                    texture_binding(2, true, wgpu::TextureViewDimension::D2Array, false),
                    texture_binding(3, true, wgpu::TextureViewDimension::D2Array, false),
                    texture_binding(4, true, wgpu::TextureViewDimension::D2Array, false),
                    sampler_binding(5, true, wgpu::SamplerBindingType::Filtering),
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
                    texture_binding(3, true, wgpu::TextureViewDimension::D2, false),  // gbuf_albedo
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
            ssr_camera_bind_group_layout,
            lighting_inputs_bind_group_layout,
            ibl_bind_group_layout,
        )
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

        // Bind group for object data
        self.object_data_bind_group = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: self.object_data_bind_group_layout.as_ref().unwrap(),
                label: Some("Object Data Bind Group"),
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
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self
                                .albedo_texture_array
                                .as_ref()
                                .unwrap()
                                .create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self
                                .normal_texture_array
                                .as_ref()
                                .unwrap()
                                .create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &self
                                .metallic_roughness_texture_array
                                .as_ref()
                                .unwrap()
                                .create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &self
                                .emission_texture_array
                                .as_ref()
                                .unwrap()
                                .create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(
                            self.pbr_sampler.as_ref().unwrap(),
                        ),
                    },
                ],
            }),
        );

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

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
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

    pub fn update_render_data(&mut self, render_data: RenderData) {
        self.current_render_data = Some(render_data);
    }

    pub fn process_message(&mut self, message: RenderMessage) {
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
            } => {
                if let Ok(gpu_index) = self.add_texture(&name, &data, kind, format) {
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

    /// Call this every frame before your `render()` call.
    pub fn resolve_pending_materials(&mut self) {
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
                // ...build the final material struct for the renderer...
                let final_material = crate::graphics::renderer::renderer::Material {
                    // Use the full path
                    albedo: mat_data.albedo,
                    metallic: mat_data.metallic,
                    roughness: mat_data.roughness,
                    ao: mat_data.ao,
                    emission_strength: mat_data.emission_strength,
                    emission_color: mat_data.emission_color,
                    albedo_texture_index: albedo_idx.unwrap(),
                    normal_texture_index: normal_idx.unwrap(),
                    metallic_roughness_texture_index: mr_idx.unwrap(),
                    emission_texture_index: emission_idx.unwrap(),
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

    pub fn render(&mut self) -> Result<(), RendererError> {
        let Some(ref render_data) = self.current_render_data.clone() else {
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
        self.run_shadow_pass(&mut encoder, render_data, &static_camera_view);

        // --- 2. Geometry Pass ---
        self.run_geometry_pass(&mut encoder, render_data, alpha);

        // --- 3. SSR Pass ---
        self.run_ssr_pass(&mut encoder);

        // --- 4. Lighting/Composite Pass ---
        self.run_lighting_pass(&mut encoder, &output_view);

        // --- 5. Copy to History Buffer ---
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
            output_frame.texture.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        output_frame.present();

        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;

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
    ) {
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

            let mut scene_bounds_min = Vec3::splat(f32::MAX);
            let mut scene_bounds_max = Vec3::splat(f32::MIN);
            for object in &render_data.objects {
                if object.casts_shadow {
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
            let camera_view = Mat4::look_at_rh(
                render_data.current_camera_transform.position,
                render_data.current_camera_transform.position
                    + render_data.current_camera_transform.forward(),
                render_data.current_camera_transform.up(),
            );
            let camera_proj = Mat4::perspective_infinite_reverse_rh(
                camera.fov_y_rad,
                camera.aspect_ratio,
                camera.near_plane,
            );

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

            for i in 0..NUM_CASCADES {
                self.queue.write_buffer(
                    &shadow_pipeline.light_vp_buffer,
                    0,
                    bytemuck::bytes_of(&shadow_uniforms.cascades[i].light_view_proj),
                );

                let cascade_view = &self.cascade_views.as_ref().unwrap()[i];

                let depth_ops = wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0), // Always clear to the furthest value
                    store: wgpu::StoreOp::Store, // Store depth for the current pass to work correctly
                };

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
                        depth_ops: Some(depth_ops), // Use the new, corrected operations
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                shadow_pass.set_pipeline(&shadow_pipeline.pipeline);
                shadow_pass.set_bind_group(0, &shadow_pipeline.light_vp_bind_group, &[]);

                for object in &render_data.objects {
                    if object.casts_shadow {
                        if let Some(mesh) = self.meshes.get(&object.mesh_id) {
                            let model_matrix = Mat4::from_scale_rotation_translation(
                                object.current_transform.scale,
                                object.current_transform.rotation,
                                object.current_transform.position,
                            );
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
                    wgpu::ShaderStages::VERTEX,
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

        //  println!("Light direction: {:?}", light_dir);

        let mut uniforms = ShadowUniforms {
            cascades: [CascadeUniform::default(); NUM_CASCADES],
        };

        for i in 0..NUM_CASCADES {
            //  println!("\n=== CASCADE {} ===", i);

            // --- 1. Get the world-space corners of the cascade's frustum slice ---
            let z_near = CASCADE_SPLITS[i];
            let z_far = CASCADE_SPLITS[i + 1];

            // println!("z_near: {}, z_far: {}", z_near, z_far);

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

            // --- 2. Find the bounds of this frustum slice in world space ---
            let mut world_min = Vec3::splat(f32::MAX);
            let mut world_max = Vec3::splat(f32::MIN);

            for corner in frustum_corners_world {
                world_min = world_min.min(corner);
                world_max = world_max.max(corner);
            }

            let world_center = (world_min + world_max) * 0.5;
            let world_size = world_max - world_min;

            //println!("World bounds: {:?} to {:?}", world_min, world_max);
            // println!("World center: {:?}, size: {:?}", world_center, world_size);

            // --- 3. Create light view matrix ---
            // Position light far enough back to see the entire frustum
            let light_distance = world_size.length() * 2.0;
            let light_position = world_center - light_dir * light_distance;

            //println!("Light position: {:?} (distance: {})", light_position, light_distance);

            let light_view = Mat4::look_at_rh(light_position, world_center, Vec3::Y);

            // --- 4. Find the frustum bounds in light space ---
            let mut light_space_min = Vec3::splat(f32::MAX);
            let mut light_space_max = Vec3::splat(f32::MIN);

            for corner in frustum_corners_world {
                let ls_point = (light_view * corner.extend(1.0)).xyz();
                light_space_min = light_space_min.min(ls_point);
                light_space_max = light_space_max.max(ls_point);
            }

            //println!("Light space bounds: {:?} to {:?}", light_space_min, light_space_max);

            // --- 5. Create orthographic projection ---
            // CRITICAL: For RH coordinate system, near > far in Z
            // The Z values are negative, so we need to swap and negate them
            let z_near = -light_space_max.z - 5.0; // Add padding
            let z_far = -light_space_min.z + 5.0; // Add padding

            //println!("Orthographic params: x({} to {}), y({} to {}), z({} to {})",
            //   light_space_min.x - 5.0, light_space_max.x + 5.0,
            //  light_space_min.y - 5.0, light_space_max.y + 5.0,
            //   z_near, z_far);

            let light_proj = Mat4::orthographic_rh(
                light_space_min.x - 5.0,
                light_space_max.x + 5.0,
                light_space_min.y - 5.0,
                light_space_max.y + 5.0,
                z_near,
                z_far,
            );

            // --- 3. Use scene bounds but clamp to reasonable size ---
            let scene_size = scene_bounds.max - scene_bounds.min;
            let scene_radius = scene_size.length() * 0.5;

            // CRITICAL: Clamp scene radius to avoid precision issues
            let clamped_radius = scene_radius.min(200.0); // Max 200 units radius

            // Scale the orthographic projection based on CLAMPED scene size
            let ortho_size = (clamped_radius * 1.2).max(20.0); // At least 20 units, with 20% padding
            let ortho_depth = (clamped_radius * 2.5).max(50.0); // At least 50 units depth

            // Scale the light distance based on CLAMPED scene size
            let light_distance = (clamped_radius * 1.5).max(30.0); // At least 30 units back

            let light_position = -light_dir * light_distance;
            let target = Vec3::ZERO;

            //println!("Scene radius: {} -> clamped: {}, ortho_size: {}, ortho_depth: {}, light_distance: {}",
            //   scene_radius, clamped_radius, ortho_size, ortho_depth, light_distance);
            //println!("Light position: {:?}, Target: {:?}", light_position, target);

            let light_view = Mat4::look_at_rh(light_position, target, Vec3::Y);
            let light_proj = Mat4::orthographic_rh(
                -ortho_size,
                ortho_size,
                -ortho_size,
                ortho_size,
                0.1,
                ortho_depth,
            );

            let final_light_vp = WGPU_CLIP_SPACE_CORRECTION * light_proj * light_view;

            //println!("Final light VP matrix: {:?}", final_light_vp);

            // Test that the world center projects to roughly (0,0,0) in NDC
            let projected = final_light_vp * world_center.extend(1.0);
            let ndc = projected.xyz() / projected.w;
            //println!("World center NDC: {:?}", ndc);

            uniforms.cascades[i] = CascadeUniform {
                light_view_proj: final_light_vp.to_cols_array_2d(),
                split_depth: -z_far,
                _padding: [0.0; 3],
            };
        }

        uniforms
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
