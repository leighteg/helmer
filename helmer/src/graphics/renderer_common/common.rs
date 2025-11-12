use crate::{
    graphics::{
        config::RenderConfig,
        renderer_common::error::RendererError,
        renderers::{
            deferred::DeferredRenderer, forward_pmu::ForwardRendererPMU,
            forward_ta::ForwardRendererTA,
        },
    },
    provided::components::{Camera, LightType, Transform},
    runtime::asset_server::{AssetKind, MaterialGpuData},
};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::{collections::HashMap, env, time::Instant};
use tracing::info;
use winit::dpi::PhysicalSize;

// --- CONSTANTS (Shared across renderers) ---
pub const FRAMES_IN_FLIGHT: usize = 3;
pub const SHADOW_MAP_RESOLUTION: u32 = 2048;
pub const NUM_CASCADES: usize = 4;
pub const CASCADE_SPLITS: [f32; 5] = [0.1, 15.0, 40.0, 100.0, 300.0];

// --- The Public Rendering Trait ---
/// Defines the common interface for any rendering backend in the engine.
pub trait RenderTrait {
    /// Resizes the surface and any screen-dependent resources.
    fn resize(&mut self, new_size: PhysicalSize<u32>);
    /// Renders a single frame.
    fn render(&mut self) -> Result<(), RendererError>;
    /// Processes an incoming message from the main application thread.
    fn process_message(&mut self, message: RenderMessage);
    /// Updates the render data for the next frame.
    fn update_render_data(&mut self, render_data: RenderData);
    /// Resolves pending materials whose textures have finished loading.
    fn resolve_pending_materials(&mut self);
}

/// Factory function that detects hardware capabilities and initializes the best renderer.
pub async fn initialize_renderer(
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    size: PhysicalSize<u32>,
    target_tickrate: f32,
) -> Result<Box<dyn RenderTrait>, RendererError> {
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

    let supported_features = adapter.features();

    // Check for the features required for the high-end "bindless" deferred renderer.
    let supports_high_end = if supported_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
        && supported_features
            .contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
    {
        info!("Adapter supports bindless texturing.");
        true
    } else {
        info!("Adapter does not support bindless texturing.");
        false
    };

    let path_str = env::var("HELMER_PATH").unwrap_or_else(|_| "auto".to_string());

    let prefers_high_end = !path_str.as_str().starts_with("forward");

    let renderer: Box<dyn RenderTrait> = if supports_high_end && prefers_high_end {
        info!("Initializing High-End Deferred Renderer.");

        let renderer =
            DeferredRenderer::new(instance, surface, adapter, size, target_tickrate).await?;
        Box::new(renderer)
    } else {
        info!("Initializing Low-End Forward Renderer.");

        if path_str.as_str().ends_with("TA") {
            Box::new(
                ForwardRendererTA::new(instance, surface, adapter, size, target_tickrate).await?,
            )
        } else {
            Box::new(
                ForwardRendererPMU::new(instance, surface, &adapter, size, target_tickrate).await?,
            )
        }
    };

    Ok(renderer)
}

// --- SHARED DATA STRUCTURES ---

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
            emission_texture_index: -1,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialShaderData {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub emission_strength: f32,
    pub albedo_idx: i32,
    pub normal_idx: i32,
    pub metallic_roughness_idx: i32,
    pub emission_idx: i32,
    pub emission_color: [f32; 3],
    pub _padding: f32,
}

impl From<&Material> for MaterialShaderData {
    fn from(material: &Material) -> Self {
        Self {
            albedo: material.albedo,
            metallic: material.metallic,
            roughness: material.roughness,
            ao: material.ao,
            emission_strength: material.emission_strength,
            albedo_idx: material.albedo_texture_index,
            normal_idx: material.normal_texture_index,
            metallic_roughness_idx: material.metallic_roughness_texture_index,
            emission_idx: material.emission_texture_index,
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

/// Holds the per-instance data (model matrix) to be sent to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub model_matrix: [[f32; 4]; 4],
}

impl InstanceRaw {
    /// Describes the layout of the instance buffer for the wgpu pipeline.
    /// Shader locations 5-8 are used (0-4 are taken by Vertex).
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // This buffer steps forward once per *instance*, not per vertex.
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // model_matrix (takes 4 slots)
                // A mat4f is four vec4f
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5, // 0-4 are for Vertex
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub struct MeshLod {
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer, // A single vertex buffer is shared across all LODs
    pub lods: Vec<MeshLod>,
    pub bounds: Aabb,
}

#[derive(Debug, Clone)]
pub struct RenderObject {
    pub id: usize,
    pub previous_transform: Transform,
    pub current_transform: Transform,
    pub mesh_id: usize,
    pub material_id: usize,
    pub casts_shadow: bool,
    pub lod_index: usize,
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
    pub render_config: RenderConfig,
}

// --- SHARED SHADER DATA STRUCTS ---

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view_matrix: [[f32; 4]; 4],
    pub projection_matrix: [[f32; 4]; 4],
    pub inverse_projection_matrix: [[f32; 4]; 4],
    pub inverse_view_projection_matrix: [[f32; 4]; 4],
    pub view_position: [f32; 3],
    pub light_count: u32,
    pub prev_view_proj: [[f32; 4]; 4],
    pub frame_index: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LightData {
    pub position: [f32; 3],
    pub light_type: u32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub direction: [f32; 3],
    pub _padding: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PbrConstants {
    pub model_matrix: [[f32; 4]; 4],
    pub material_id: u32,
    pub _p: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CascadeUniform {
    pub light_view_proj: [[f32; 4]; 4],
    pub split_depth: [f32; 4],
}

impl Default for CascadeUniform {
    fn default() -> Self {
        Self {
            light_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            split_depth: [0.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowUniforms {
    pub cascades: [CascadeUniform; NUM_CASCADES],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ModelPushConstant {
    pub model_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Default)]
pub struct SkyUniforms {
    pub sun_direction: [f32; 3],
    pub _padding: f32,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub ground_albedo: [f32; 3],
    pub ground_brightness: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct ShaderConstants {
    pub mip_bias: f32,

    pub shade_mode: u32,
    pub light_model: u32,
    pub skylight_contribution: u32,

    pub planet_radius: f32,
    pub atmosphere_radius: f32,
    pub sky_light_samples: u32,
    _pad0: u32,

    pub ssr_coarse_steps: u32,
    pub ssr_binary_search_steps: u32,
    pub ssr_linear_step_size: f32,
    pub ssr_thickness: f32,

    pub ssr_max_distance: f32,
    pub ssr_roughness_fade_start: f32,
    pub ssr_roughness_fade_end: f32,
    _pad1: u32,

    pub ssgi_num_rays: u32,
    pub ssgi_num_steps: u32,
    pub ssgi_ray_step_size: f32,
    pub ssgi_thickness: f32,

    pub ssgi_blend_factor: f32,
    _pad2: f32, _pad3: f32, _pad4: f32,

    pub evsm_c: f32,
    pub pcf_radius: u32,
    pub pcf_min_scale: f32,
    pub pcf_max_scale: f32,

    pub pcf_max_distance: f32,
    pub ssgi_intensity: f32,

    // Final padding only
    _final_padding: [f32; 2],
}

impl Default for ShaderConstants {
    fn default() -> Self {
        Self {
            // general
            mip_bias: 0.0,

            // lighting
            shade_mode: 0,
            light_model: 0,
            skylight_contribution: 1,

            // Sky
            planet_radius: 6371e3,
            atmosphere_radius: 6471e3,
            sky_light_samples: 6,
            _pad0: 0,

            // SSR Defaults
            ssr_coarse_steps: 160,
            ssr_binary_search_steps: 6,
            ssr_linear_step_size: 0.07,
            ssr_thickness: 0.1,
            ssr_max_distance: 250.0,
            ssr_roughness_fade_start: 0.1,
            ssr_roughness_fade_end: 0.5,
            _pad1: 0,

            // SSGI Defaults
            ssgi_num_rays: 6,
            ssgi_num_steps: 12,
            ssgi_ray_step_size: 0.6,
            ssgi_thickness: 0.4,
            ssgi_blend_factor: 0.15,

            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,

            // shadows Default
            evsm_c: 20.0,
            pcf_radius: 2,
            pcf_min_scale: 1.0,
            pcf_max_scale: 3.5,
            pcf_max_distance: 80.0,

            // Composite Default
            ssgi_intensity: 30.0,

            _final_padding: [0.0; 2],
        }
    }
}

// --- SHARED HELPER STRUCTS ---

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

    pub fn get_corners(&self) -> [Vec3; 8] {
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

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) / 2.0
    }

    pub fn extents(&self) -> Vec3 {
        (self.max - self.min) / 2.0
    }
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

pub struct ShadowPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
}

pub enum RenderMessage {
    RenderData(RenderData),
    Resize(PhysicalSize<u32>),
    Shutdown,

    // --- Asset Pipeline Messages ---
    CreateMesh {
        id: usize,
        vertices: Vec<Vertex>,
        lod_indices: Vec<Vec<u32>>,
        bounds: Aabb,
    },
    CreateTexture {
        id: usize,
        name: String, // The file path for deduplication
        kind: AssetKind,
        data: Vec<u8>,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    },
    CreateMaterial(MaterialGpuData),

    EguiData(EguiRenderData),
}

pub struct EguiRenderData {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub screen_descriptor: egui_wgpu::ScreenDescriptor,
}
