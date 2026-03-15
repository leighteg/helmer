use crate::graphics::{
    backend::binding_backend::BindingBackendChoice,
    common::{
        config::RenderConfig,
        constants::{
            MAX_SHADOW_CASCADES, MESHLET_MAX_PRIMS, MESHLET_MAX_VERTS, MESHLET_WORKGROUP_SIZE,
        },
        error::RendererError,
        graph::RenderGraphSpec,
    },
    graph::logic::{gpu_resource_pool::GpuResourcePoolConfig, transient_heap::TransientHeapConfig},
    legacy_renderers::{
        deferred::DeferredRenderer, forward_pmu::ForwardRendererPMU, forward_ta::ForwardRendererTA,
    },
    render_graphs::builtin_graph_template_catalog,
    renderer::GraphRenderer,
};
use bytemuck::{Pod, Zeroable};
use crossbeam_channel::Sender;
use egui::{Color32, ColorImage, Vec2 as EguiVec2};
use glam::{Mat4, Quat, Vec2, Vec3};
use hashbrown::{HashMap, HashSet};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    env,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    },
};
use tracing::info;
use web_time::Instant;
use winit::{dpi::PhysicalSize, window::Window};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuBackend {
    Auto,
    Vulkan,
    Dx12,
    Metal,
    Gl,
}

impl WgpuBackend {
    pub fn from_env(value: Option<String>) -> Self {
        match value.as_deref().unwrap_or_default().to_lowercase().as_str() {
            "vulkan" => WgpuBackend::Vulkan,
            "dx12" => WgpuBackend::Dx12,
            "metal" => WgpuBackend::Metal,
            "gl" => WgpuBackend::Gl,
            _ => WgpuBackend::Auto,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            WgpuBackend::Auto => "Auto",
            WgpuBackend::Vulkan => "Vulkan",
            WgpuBackend::Dx12 => "DX12",
            WgpuBackend::Metal => "Metal",
            WgpuBackend::Gl => "OpenGL",
        }
    }

    pub fn to_backends(self) -> wgpu::Backends {
        match self {
            WgpuBackend::Auto => {
                #[cfg(target_arch = "wasm32")]
                {
                    wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    wgpu::Backends::all()
                }
            }
            WgpuBackend::Vulkan => wgpu::Backends::VULKAN,
            WgpuBackend::Dx12 => wgpu::Backends::DX12,
            WgpuBackend::Metal => wgpu::Backends::METAL,
            WgpuBackend::Gl => wgpu::Backends::GL,
        }
    }

    pub fn to_backend(self) -> Option<wgpu::Backend> {
        match self {
            WgpuBackend::Auto => None,
            WgpuBackend::Vulkan => Some(wgpu::Backend::Vulkan),
            WgpuBackend::Dx12 => Some(wgpu::Backend::Dx12),
            WgpuBackend::Metal => Some(wgpu::Backend::Metal),
            WgpuBackend::Gl => Some(wgpu::Backend::Gl),
        }
    }
}

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
    fn update_render_data(&mut self, render_data: Arc<RenderData>);
    /// Resolves pending materials whose textures have finished loading.
    fn resolve_pending_materials(&mut self);
}

/// Factory function that detects hardware capabilities and initializes the best render path.
pub async fn initialize_renderer(
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    size: PhysicalSize<u32>,
    target_tickrate: f32,
    asset_stream_sender: Sender<AssetStreamingRequest>,
    shared_stats: Arc<RendererStats>,
    allow_experimental_features: bool,
    backend_choice: WgpuBackend,
    binding_backend_choice: BindingBackendChoice,
) -> Result<GraphRenderer, RendererError> {
    let adapters = instance
        .enumerate_adapters(backend_choice.to_backends())
        .await;
    let mut candidates: Vec<wgpu::Adapter> = adapters
        .into_iter()
        .filter(|adapter| adapter.is_surface_supported(&surface))
        .collect();
    if let Some(backend) = backend_choice.to_backend() {
        candidates.retain(|adapter| adapter.get_info().backend == backend);
    }

    let adapter = if !candidates.is_empty() {
        candidates.sort_by_key(|adapter| match adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Cpu => 3,
            wgpu::DeviceType::Other => 4,
        });
        candidates.remove(0)
    } else {
        match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(new_adapter) => new_adapter,
            Err(err) => {
                return Err(RendererError::ResourceCreation(format!(
                    "Failed to find a suitable GPU adapter: {}",
                    err
                )));
            }
        }
    };

    let supported_features = adapter.features();

    // Check for the features required for the high-end "bindless" deferred renderer.
    let supports_high_end = if supported_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
        && supported_features
            .contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
    {
        info!("adapter supports bindless texturing");
        true
    } else {
        info!("adapter does not support bindless texturing");
        false
    };

    let path_str = env::var("HELMER_PATH").unwrap_or_else(|_| "auto".to_string());

    let prefers_high_end = !path_str.as_str().starts_with("forward");

    Ok(GraphRenderer::new(
        instance,
        surface,
        &adapter,
        size,
        target_tickrate,
        asset_stream_sender,
        shared_stats,
        allow_experimental_features,
        binding_backend_choice,
        builtin_graph_template_catalog(),
    )
    .await?)
}

// --- SHARED DATA STRUCTURES ---

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    pub fn from_matrix(matrix: Mat4) -> Self {
        let (scale, rotation, position) = matrix.to_scale_rotation_translation();
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    pub fn from_position(position: [f32; 3]) -> Self {
        Self {
            position: Vec3::from_array(position),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot { angle: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub light_type: LightType,
    pub color: Vec3,
    pub intensity: f32,
}

impl Light {
    pub fn directional(color: Vec3, intensity: f32) -> Self {
        Self {
            light_type: LightType::Directional,
            color,
            intensity,
        }
    }

    pub fn point(color: Vec3, intensity: f32) -> Self {
        Self {
            light_type: LightType::Point,
            color,
            intensity,
        }
    }

    pub fn spot(color: Vec3, intensity: f32, angle: f32) -> Self {
        Self {
            light_type: LightType::Spot { angle },
            color,
            intensity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
    pub near_plane: f32,
    pub far_plane: f32,
}

impl Camera {
    pub fn new(fov_y_rad: f32, aspect_ratio: f32, near_plane: f32, far_plane: f32) -> Self {
        Self {
            fov_y_rad,
            aspect_ratio,
            near_plane,
            far_plane,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(std::f32::consts::FRAC_PI_4, 1.7, 0.1, 100.0)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ActiveCamera;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlphaMode {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
    Premultiplied = 3,
    Additive = 4,
}

impl Default for AlphaMode {
    fn default() -> Self {
        AlphaMode::Opaque
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetKind {
    Albedo,
    Normal,
    MetallicRoughness,
    Emission,
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
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: Option<f32>,
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
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaterialGpuData {
    pub id: usize,
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub emission_strength: f32,
    pub emission_color: [f32; 3],
    pub albedo_texture_id: Option<usize>,
    pub normal_texture_id: Option<usize>,
    pub metallic_roughness_texture_id: Option<usize>,
    pub emission_texture_id: Option<usize>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: Option<f32>,
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
    pub alpha_mode: u32,
    pub alpha_cutoff: f32,
    pub _pad_alpha: [u32; 2],
}

impl From<&Material> for MaterialShaderData {
    fn from(material: &Material) -> Self {
        let alpha_cutoff = material.alpha_cutoff.unwrap_or(0.0);
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
            alpha_mode: material.alpha_mode as u32,
            alpha_cutoff,
            _pad_alpha: [0; 2],
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
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // normal
        2 => Float32x2, // tex_coord
        3 => Float32x4, // tangent
        4 => Uint32x4,  // joints
        5 => Float32x4, // weights
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
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn with_skinning(
        position: [f32; 3],
        normal: [f32; 3],
        tex_coord: [f32; 2],
        tangent: [f32; 4],
        joints: [u32; 4],
        weights: [f32; 4],
    ) -> Self {
        Self {
            position,
            normal,
            tex_coord,
            tangent,
            joints,
            weights,
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
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct MeshletDesc {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
}

#[derive(Clone, Debug, Default)]
pub struct MeshletLodData {
    pub descs: Arc<[MeshletDesc]>,
    pub vertices: Arc<[u32]>,
    pub indices: Arc<[u32]>,
}

impl MeshletLodData {
    pub fn meshlet_count(&self) -> u32 {
        self.descs.len().min(u32::MAX as usize) as u32
    }

    pub fn is_empty(&self) -> bool {
        self.descs.is_empty()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshDrawParams {
    pub instance_base: u32,
    pub instance_count: u32,
    pub meshlet_base: u32,
    pub meshlet_count: u32,
    pub flags: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub depth_bias: f32,
    pub rect_pad: f32,
    pub _pad3: [f32; 2],
}

/// Holds the per-instance data (model matrix) to be sent to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub model_matrix: [[f32; 4]; 4],
    pub skin_offset: u32,
    pub skin_count: u32,
    pub _pad0: [u32; 2],
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
                    shader_location: 6, // 0-5 are for Vertex
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress
                        + mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// Per-instance payload consumed by G-buffer and forward passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GBufferInstanceRaw {
    pub model_matrix: [[f32; 4]; 4],
    pub material_id: u32,
    pub visibility: u32,
    pub skin_offset: u32,
    pub skin_count: u32,
    pub bounds_center: [f32; 4],
    pub bounds_extents: [f32; 4],
}

impl GBufferInstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress
                        + mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress
                        + mem::size_of::<[u32; 2]>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress
                        + mem::size_of::<[u32; 3]>() as wgpu::BufferAddress,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// Per-instance payload consumed by the shadow map pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowInstanceRaw {
    pub model_matrix: [[f32; 4]; 4],
    pub material_id: u32,
    pub skin_offset: u32,
    pub skin_count: u32,
    pub _pad0: u32,
}

impl ShadowInstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ShadowInstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress
                        + mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress
                        + 2 * mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

pub struct MeshLod {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

pub struct Mesh {
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
    pub skin_offset: u32,
    pub skin_count: u32,
}

#[derive(Debug, Clone)]
pub struct RenderLight {
    pub id: usize,
    pub previous_transform: Transform,
    pub current_transform: Transform,
    pub color: [f32; 3],
    pub intensity: f32,
    pub light_type: LightType,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpriteSpace {
    Screen = 0,
    World = 1,
}

impl Default for SpriteSpace {
    fn default() -> Self {
        SpriteSpace::World
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpriteBlendMode {
    Alpha = 0,
    Premultiplied = 1,
    Additive = 2,
}

impl Default for SpriteBlendMode {
    fn default() -> Self {
        SpriteBlendMode::Alpha
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpriteAnimationPlayback {
    Loop = 0,
    Once = 1,
    PingPong = 2,
}

impl Default for SpriteAnimationPlayback {
    fn default() -> Self {
        Self::Loop
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteSheetAnimation {
    pub enabled: bool,
    pub columns: u32,
    pub rows: u32,
    pub start_frame: u32,
    pub frame_count: u32,
    pub fps: f32,
    pub playback: SpriteAnimationPlayback,
    pub phase: f32,
    pub paused: bool,
    pub paused_frame: u32,
    pub flip_x: bool,
    pub flip_y: bool,
    pub frame_uv_inset: [f32; 2],
}

impl Default for SpriteSheetAnimation {
    fn default() -> Self {
        Self {
            enabled: false,
            columns: 1,
            rows: 1,
            start_frame: 0,
            frame_count: 0,
            fps: 12.0,
            playback: SpriteAnimationPlayback::Loop,
            phase: 0.0,
            paused: false,
            paused_frame: 0,
            flip_x: false,
            flip_y: false,
            frame_uv_inset: [0.0, 0.0],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSpriteImageSequence {
    pub enabled: bool,
    pub texture_ids: Arc<Vec<usize>>,
    pub start_frame: u32,
    pub frame_count: u32,
    pub fps: f32,
    pub playback: SpriteAnimationPlayback,
    pub phase: f32,
    pub paused: bool,
    pub paused_frame: u32,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl Default for RenderSpriteImageSequence {
    fn default() -> Self {
        Self {
            enabled: false,
            texture_ids: Arc::new(Vec::new()),
            start_frame: 0,
            frame_count: 0,
            fps: 12.0,
            playback: SpriteAnimationPlayback::Loop,
            phase: 0.0,
            paused: false,
            paused_frame: 0,
            flip_x: false,
            flip_y: false,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlignH {
    Left = 0,
    Center = 1,
    Right = 2,
}

impl Default for TextAlignH {
    fn default() -> Self {
        TextAlignH::Left
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlignV {
    Top = 0,
    Center = 1,
    Bottom = 2,
    Baseline = 3,
}

impl Default for TextAlignV {
    fn default() -> Self {
        TextAlignV::Baseline
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextFontStyle {
    Normal = 0,
    Italic = 1,
    Oblique = 2,
}

impl Default for TextFontStyle {
    fn default() -> Self {
        TextFontStyle::Normal
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSprite {
    pub id: u64,
    pub position: Vec3,
    pub rotation: Quat,
    pub size: Vec2,
    pub color: [f32; 4],
    pub texture_id: Option<usize>,
    pub image_sequence: Option<RenderSpriteImageSequence>,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub sheet_animation: SpriteSheetAnimation,
    pub pivot: [f32; 2],
    pub clip_rect: Option<[f32; 4]>,
    pub layer: f32,
    pub space: SpriteSpace,
    pub blend_mode: SpriteBlendMode,
    pub billboard: bool,
    pub pick_id: u32,
}

impl Default for RenderSprite {
    fn default() -> Self {
        Self {
            id: 0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            size: Vec2::splat(1.0),
            color: [1.0; 4],
            texture_id: None,
            image_sequence: None,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            sheet_animation: SpriteSheetAnimation::default(),
            pivot: [0.5, 0.5],
            clip_rect: None,
            layer: 0.0,
            space: SpriteSpace::World,
            blend_mode: SpriteBlendMode::Alpha,
            billboard: false,
            pick_id: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderText2d {
    pub id: u64,
    pub text: String,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec2,
    pub color: [f32; 4],
    pub font_path: Option<String>,
    pub font_family: Option<String>,
    pub font_size: f32,
    pub font_weight: f32,
    pub font_width: f32,
    pub font_style: TextFontStyle,
    pub line_height_scale: f32,
    pub letter_spacing: f32,
    pub word_spacing: f32,
    pub underline: bool,
    pub strikethrough: bool,
    pub max_width: Option<f32>,
    pub align_h: TextAlignH,
    pub align_v: TextAlignV,
    pub space: SpriteSpace,
    pub billboard: bool,
    pub blend_mode: SpriteBlendMode,
    pub layer: f32,
    pub clip_rect: Option<[f32; 4]>,
    pub pick_id: u32,
}

impl Default for RenderText2d {
    fn default() -> Self {
        Self {
            id: 0,
            text: String::new(),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec2::ONE,
            color: [1.0; 4],
            font_path: None,
            font_family: None,
            font_size: 16.0,
            font_weight: 400.0,
            font_width: 1.0,
            font_style: TextFontStyle::Normal,
            line_height_scale: 1.0,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            underline: false,
            strikethrough: false,
            max_width: None,
            align_h: TextAlignH::Left,
            align_v: TextAlignV::Baseline,
            space: SpriteSpace::World,
            billboard: false,
            blend_mode: SpriteBlendMode::Alpha,
            layer: 0.0,
            clip_rect: None,
            pick_id: 0,
        }
    }
}

/// Hint describing how urgently an asset should be (re)streamed to the GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetStreamKind {
    Mesh,
    Material,
    Texture,
}

#[derive(Debug, Clone)]
pub struct AssetStreamingRequest {
    pub id: usize,
    pub kind: AssetStreamKind,
    /// Higher numbers mean higher priority
    pub priority: f32,
    /// Optional maximum LOD to upload for meshes (coarser == larger number)
    pub max_lod: Option<usize>,
    /// Request a low-res mip chain when under heavy VRAM pressure
    pub force_low_res: bool,
}

#[derive(Debug, Clone)]
pub struct MeshLodPayload {
    pub lod_index: usize,
    pub vertices: Arc<[Vertex]>,
    pub indices: Arc<[u32]>,
    pub meshlets: MeshletLodData,
}

#[derive(Debug, Clone)]
pub struct RenderData {
    pub objects: Vec<RenderObject>,
    pub lights: Vec<RenderLight>,
    pub sprites: Vec<RenderSprite>,
    pub text_2d: Vec<RenderText2d>,
    pub ui: UiRenderData,
    pub previous_camera_transform: Transform,
    pub current_camera_transform: Transform,
    pub camera_component: Camera,
    pub render_main_scene_to_swapchain: bool,
    pub viewports: Vec<RenderViewportRequest>,
    pub timestamp: Instant,
    pub render_config: RenderConfig,
    pub render_graph: RenderGraphSpec,
    pub gizmo: GizmoData,
    pub skin_palette: Vec<Mat4>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UiRenderRect {
    pub id: u64,
    pub rect: [f32; 4],
    pub color: [f32; 4],
    pub clip_rect: Option<[f32; 4]>,
    pub layer: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UiRenderImage {
    pub id: u64,
    pub rect: [f32; 4],
    pub texture_id: Option<usize>,
    pub tint: [f32; 4],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub clip_rect: Option<[f32; 4]>,
    pub layer: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UiRenderText {
    pub id: u64,
    pub rect: [f32; 4],
    pub text: String,
    pub color: [f32; 4],
    pub font_size: f32,
    pub align_h: TextAlignH,
    pub align_v: TextAlignV,
    pub wrap: bool,
    pub cursor: Option<usize>,
    pub show_caret: bool,
    pub caret_color: Option<[f32; 4]>,
    pub selection: Option<[usize; 2]>,
    pub selection_color: Option<[f32; 4]>,
    pub clip_rect: Option<[f32; 4]>,
    pub layer: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UiRenderCommand {
    Rect(UiRenderRect),
    Image(UiRenderImage),
    Text(UiRenderText),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct UiRenderData {
    pub commands: Vec<UiRenderCommand>,
}

#[derive(Debug, Clone)]
pub struct RenderViewportRequest {
    pub id: u64,
    pub camera_transform: Transform,
    pub camera_component: Camera,
    pub texture_handle: u64,
    pub texture_is_managed: bool,
    pub target_size: [u32; 2],
    pub temporal_history: bool,
    pub immediate_resize: bool,
    pub graph_template: Option<String>,
    pub gizmo_options: RenderViewportGizmoOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderViewportGizmoOptions {
    pub show_gizmos: bool,
    pub show_camera_gizmos: bool,
    pub show_directional_light_gizmos: bool,
    pub show_point_light_gizmos: bool,
    pub show_spot_light_gizmos: bool,
}

impl Default for RenderViewportGizmoOptions {
    fn default() -> Self {
        Self {
            show_gizmos: true,
            show_camera_gizmos: true,
            show_directional_light_gizmos: true,
            show_point_light_gizmos: true,
            show_spot_light_gizmos: true,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoMode {
    None = 0,
    Translate = 1,
    Rotate = 2,
    Scale = 3,
}

impl Default for GizmoMode {
    fn default() -> Self {
        GizmoMode::None
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis {
    None = 0,
    X = 1,
    Y = 2,
    Z = 3,
    Center = 4,
    XY = 5,
    XZ = 6,
    YZ = 7,
}

impl Default for GizmoAxis {
    fn default() -> Self {
        GizmoAxis::None
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoIconKind {
    Camera = 0,
    LightDirectional = 1,
    LightPoint = 2,
    LightSpot = 3,
    AudioEmitter = 4,
}

impl Default for GizmoIconKind {
    fn default() -> Self {
        GizmoIconKind::Camera
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoIcon {
    pub position: Vec3,
    pub rotation: Quat,
    pub size: f32,
    pub color: Vec3,
    pub alpha: f32,
    pub kind: GizmoIconKind,
    pub params: [f32; 4],
}

impl Default for GizmoIcon {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            size: 0.0,
            color: Vec3::ZERO,
            alpha: 1.0,
            kind: GizmoIconKind::Camera,
            params: [0.0; 4],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoLine {
    pub start: Vec3,
    pub end: Vec3,
}

impl Default for GizmoLine {
    fn default() -> Self {
        Self {
            start: Vec3::ZERO,
            end: Vec3::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoStyle {
    pub ring_segments: u32,
    pub translate_thickness_scale: f32,
    pub translate_thickness_min: f32,
    pub translate_head_length_scale: f32,
    pub translate_head_width_scale: f32,
    pub scale_thickness_scale: f32,
    pub scale_thickness_min: f32,
    pub scale_head_length_scale: f32,
    pub scale_box_scale: f32,
    pub rotate_radius_scale: f32,
    pub rotate_thickness_scale: f32,
    pub rotate_thickness_min: f32,
    pub plane_offset_scale: f32,
    pub plane_size_scale: f32,
    pub plane_alpha: f32,
    pub origin_size_scale: f32,
    pub origin_size_min: f32,
    pub axis_color_x: [f32; 3],
    pub axis_color_y: [f32; 3],
    pub axis_color_z: [f32; 3],
    pub origin_color: [f32; 3],
    pub axis_alpha: f32,
    pub origin_alpha: f32,
    pub selection_thickness_scale: f32,
    pub selection_thickness_min: f32,
    pub selection_color: [f32; 3],
    pub selection_alpha: f32,
    pub outline_thickness_scale: f32,
    pub outline_thickness_min: f32,
    pub outline_color: [f32; 3],
    pub outline_alpha: f32,
    pub icon_thickness_scale: f32,
    pub icon_thickness_min: f32,
    pub hover_mix: f32,
    pub active_mix: f32,
}

impl Default for GizmoStyle {
    fn default() -> Self {
        Self {
            ring_segments: 32,
            translate_thickness_scale: 0.05,
            translate_thickness_min: 0.015,
            translate_head_length_scale: 0.22,
            translate_head_width_scale: 2.6,
            scale_thickness_scale: 0.05,
            scale_thickness_min: 0.015,
            scale_head_length_scale: 0.18,
            scale_box_scale: 2.0,
            rotate_radius_scale: 0.85,
            rotate_thickness_scale: 0.03,
            rotate_thickness_min: 0.01,
            plane_offset_scale: 0.22,
            plane_size_scale: 0.2,
            plane_alpha: 0.35,
            origin_size_scale: 0.06,
            origin_size_min: 0.03,
            axis_color_x: [1.0, 0.2, 0.2],
            axis_color_y: [0.2, 1.0, 0.2],
            axis_color_z: [0.2, 0.4, 1.0],
            origin_color: [0.9, 0.9, 0.9],
            axis_alpha: 1.0,
            origin_alpha: 1.0,
            selection_thickness_scale: 0.03,
            selection_thickness_min: 0.01,
            selection_color: [1.0, 0.85, 0.2],
            selection_alpha: 1.0,
            outline_thickness_scale: 0.02,
            outline_thickness_min: 0.006,
            outline_color: [0.35, 0.85, 1.0],
            outline_alpha: 1.0,
            icon_thickness_scale: 0.025,
            icon_thickness_min: 0.008,
            hover_mix: 0.3,
            active_mix: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GizmoData {
    pub mode: GizmoMode,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub size: f32,
    pub hover_axis: GizmoAxis,
    pub active_axis: GizmoAxis,
    pub selection_enabled: bool,
    pub selection_min: Vec3,
    pub selection_max: Vec3,
    pub outline_lines: std::sync::Arc<[GizmoLine]>,
    pub outline_revision: u64,
    pub icons: Vec<GizmoIcon>,
    pub icons_revision: u64,
    pub style: GizmoStyle,
}

impl Default for GizmoData {
    fn default() -> Self {
        Self {
            mode: GizmoMode::None,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            size: 0.0,
            hover_axis: GizmoAxis::None,
            active_axis: GizmoAxis::None,
            selection_enabled: false,
            selection_min: Vec3::ZERO,
            selection_max: Vec3::ZERO,
            outline_lines: std::sync::Arc::from(Vec::new()),
            outline_revision: 0,
            icons: Vec::new(),
            icons_revision: 0,
            style: GizmoStyle::default(),
        }
    }
}

impl PartialEq for GizmoData {
    fn eq(&self, other: &Self) -> bool {
        let outline_equal = if self.outline_lines.is_empty() && other.outline_lines.is_empty() {
            true
        } else if self.outline_revision != 0 || other.outline_revision != 0 {
            self.outline_revision == other.outline_revision
                && self.outline_lines.len() == other.outline_lines.len()
        } else {
            std::sync::Arc::ptr_eq(&self.outline_lines, &other.outline_lines)
        };
        let icons_equal = if self.icons.is_empty() && other.icons.is_empty() {
            true
        } else if self.icons_revision != 0 || other.icons_revision != 0 {
            self.icons_revision == other.icons_revision && self.icons.len() == other.icons.len()
        } else {
            self.icons == other.icons
        };
        self.mode == other.mode
            && self.position == other.position
            && self.rotation == other.rotation
            && self.scale == other.scale
            && self.size == other.size
            && self.hover_axis == other.hover_axis
            && self.active_axis == other.active_axis
            && self.selection_enabled == other.selection_enabled
            && self.selection_min == other.selection_min
            && self.selection_max == other.selection_max
            && outline_equal
            && icons_equal
            && self.style == other.style
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RenderObjectDelta {
    pub id: usize,
    pub transform: Transform,
    pub mesh_id: usize,
    pub material_id: usize,
    pub casts_shadow: bool,
    pub lod_index: usize,
    pub skin_offset: u32,
    pub skin_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderLightDelta {
    pub id: usize,
    pub transform: Transform,
    pub color: [f32; 3],
    pub intensity: f32,
    pub light_type: LightType,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderCameraDelta {
    pub transform: Transform,
    pub camera: Camera,
}

#[derive(Debug, Clone, Default)]
pub struct RenderDelta {
    pub full: bool,
    pub objects_upsert: Vec<RenderObjectDelta>,
    pub objects_remove: Vec<usize>,
    pub lights_upsert: Vec<RenderLightDelta>,
    pub lights_remove: Vec<usize>,
    pub sprites: Option<Vec<RenderSprite>>,
    pub text_2d: Option<Vec<RenderText2d>>,
    pub ui: Option<UiRenderData>,
    pub camera: Option<RenderCameraDelta>,
    pub render_main_scene_to_swapchain: Option<bool>,
    pub viewports: Option<Vec<RenderViewportRequest>>,
    pub render_config: Option<RenderConfig>,
    pub render_graph: Option<RenderGraphSpec>,
    pub gizmo: Option<GizmoData>,
    pub skin_palette: Option<Vec<Mat4>>,
    pub streaming_requests: Option<Vec<AssetStreamingRequest>>,
}

impl RenderDelta {
    pub fn is_empty(&self) -> bool {
        !self.full
            && self.objects_upsert.is_empty()
            && self.objects_remove.is_empty()
            && self.lights_upsert.is_empty()
            && self.lights_remove.is_empty()
            && self.sprites.is_none()
            && self.text_2d.is_none()
            && self.ui.is_none()
            && self.camera.is_none()
            && self.render_main_scene_to_swapchain.is_none()
            && self.viewports.is_none()
            && self.render_config.is_none()
            && self.render_graph.is_none()
            && self.gizmo.is_none()
            && self.skin_palette.is_none()
            && self.streaming_requests.is_none()
    }

    pub fn merge_from(&mut self, other: RenderDelta) {
        if other.full {
            *self = other;
            return;
        }

        if self.full {
            self.apply_delta_to_full(&other);
            return;
        }

        let mut object_updates: HashMap<usize, RenderObjectDelta> =
            self.objects_upsert.drain(..).map(|o| (o.id, o)).collect();
        let mut object_removals: HashSet<usize> = self.objects_remove.drain(..).collect();
        for obj in other.objects_upsert {
            object_removals.remove(&obj.id);
            object_updates.insert(obj.id, obj);
        }
        for id in other.objects_remove {
            object_updates.remove(&id);
            object_removals.insert(id);
        }
        self.objects_upsert = object_updates.into_values().collect();
        self.objects_remove = object_removals.into_iter().collect();

        let mut light_updates: HashMap<usize, RenderLightDelta> =
            self.lights_upsert.drain(..).map(|l| (l.id, l)).collect();
        let mut light_removals: HashSet<usize> = self.lights_remove.drain(..).collect();
        for light in other.lights_upsert {
            light_removals.remove(&light.id);
            light_updates.insert(light.id, light);
        }
        for id in other.lights_remove {
            light_updates.remove(&id);
            light_removals.insert(id);
        }
        self.lights_upsert = light_updates.into_values().collect();
        self.lights_remove = light_removals.into_iter().collect();

        if other.camera.is_some() {
            self.camera = other.camera;
        }
        if other.sprites.is_some() {
            self.sprites = other.sprites;
        }
        if other.text_2d.is_some() {
            self.text_2d = other.text_2d;
        }
        if other.ui.is_some() {
            self.ui = other.ui;
        }
        if other.render_main_scene_to_swapchain.is_some() {
            self.render_main_scene_to_swapchain = other.render_main_scene_to_swapchain;
        }
        if other.viewports.is_some() {
            self.viewports = other.viewports;
        }
        if other.render_config.is_some() {
            self.render_config = other.render_config;
        }
        if other.render_graph.is_some() {
            self.render_graph = other.render_graph;
        }
        if other.gizmo.is_some() {
            self.gizmo = other.gizmo;
        }
        if other.skin_palette.is_some() {
            self.skin_palette = other.skin_palette;
        }
        if other.streaming_requests.is_some() {
            self.streaming_requests = other.streaming_requests;
        }
    }

    fn apply_delta_to_full(&mut self, delta: &RenderDelta) {
        let mut object_updates: HashMap<usize, RenderObjectDelta> =
            self.objects_upsert.drain(..).map(|o| (o.id, o)).collect();
        for obj in &delta.objects_upsert {
            object_updates.insert(obj.id, *obj);
        }
        for id in &delta.objects_remove {
            object_updates.remove(id);
        }
        self.objects_upsert = object_updates.into_values().collect();
        self.objects_remove.clear();

        let mut light_updates: HashMap<usize, RenderLightDelta> =
            self.lights_upsert.drain(..).map(|l| (l.id, l)).collect();
        for light in &delta.lights_upsert {
            light_updates.insert(light.id, *light);
        }
        for id in &delta.lights_remove {
            light_updates.remove(id);
        }
        self.lights_upsert = light_updates.into_values().collect();
        self.lights_remove.clear();

        if delta.camera.is_some() {
            self.camera = delta.camera;
        }
        if delta.sprites.is_some() {
            self.sprites = delta.sprites.clone();
        }
        if delta.text_2d.is_some() {
            self.text_2d = delta.text_2d.clone();
        }
        if delta.ui.is_some() {
            self.ui = delta.ui.clone();
        }
        if delta.render_main_scene_to_swapchain.is_some() {
            self.render_main_scene_to_swapchain = delta.render_main_scene_to_swapchain;
        }
        if delta.viewports.is_some() {
            self.viewports = delta.viewports.clone();
        }
        if delta.render_config.is_some() {
            self.render_config = delta.render_config;
        }
        if delta.render_graph.is_some() {
            self.render_graph = delta.render_graph.clone();
        }
        if delta.gizmo.is_some() {
            self.gizmo = delta.gizmo.clone();
        }
        if delta.skin_palette.is_some() {
            self.skin_palette = delta.skin_palette.clone();
        }
        if delta.streaming_requests.is_some() {
            self.streaming_requests = delta.streaming_requests.clone();
        }
    }
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
    pub _pad_light: [u32; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub frame_index: u32,
    pub _pad_after_frame: [u32; 3],
    pub _padding: [u32; 3],
    pub _pad_after_padding: u32,
    /// Extra padding to satisfy 16-byte struct alignment for WGSL std140 layout
    pub _pad_end: [u32; 4],
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
    pub cascade_count: u32,
    pub _pad0: [u32; 3],
    pub _pad1: [u32; 4],
    pub cascades: [CascadeUniform; MAX_SHADOW_CASCADES],
}

impl Default for ShadowUniforms {
    fn default() -> Self {
        Self {
            cascade_count: MAX_SHADOW_CASCADES as u32,
            _pad0: [0; 3],
            _pad1: [0; 4],
            cascades: [CascadeUniform::default(); MAX_SHADOW_CASCADES],
        }
    }
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
    pub night_ambient_color: [f32; 3],
    pub sun_angular_radius_cos: f32,
}

#[repr(C)]
#[derive(
    Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, PartialEq, Serialize, Deserialize,
)]
pub struct ShaderConstants {
    pub mip_bias: f32,

    pub shade_mode: u32,
    pub shade_smooth: u32,
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
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,

    pub evsm_c: f32,
    pub pcf_radius: u32,
    pub pcf_min_scale: f32,
    pub pcf_max_scale: f32,

    pub pcf_max_distance: f32,
    pub ssgi_intensity: f32,
    _final_padding: [f32; 2],

    pub ssr_jitter_strength: f32,
    pub _pad_after_ssr_jitter: [f32; 3],

    pub rayleigh_scattering_coeff: [f32; 3],
    pub rayleigh_scale_height: f32,
    pub mie_scattering_coeff: f32,
    pub mie_absorption_coeff: f32,
    pub mie_scale_height: f32,
    pub mie_preferred_scattering_dir: f32,
    pub ozone_absorption_coeff: [f32; 3],
    pub ozone_center_height: f32,
    pub ozone_falloff: f32,
    pub sun_angular_radius_cos: f32,
    pub _pad_atmo0: [f32; 2],
    pub night_ambient_color: [f32; 3],
    pub _pad_atmo1: f32,
    pub sky_ground_albedo: [f32; 3],
    pub sky_ground_brightness: f32,
    pub _pad_end: [f32; 3],
}

impl Default for ShaderConstants {
    fn default() -> Self {
        Self {
            // general
            mip_bias: 0.0,

            // lighting
            shade_mode: 0,
            shade_smooth: 1,
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

            ssr_jitter_strength: 0.2,
            _pad_after_ssr_jitter: [0.0; 3],

            rayleigh_scattering_coeff: [5.8e-6, 13.5e-6, 33.1e-6],
            rayleigh_scale_height: 8_000.0,
            mie_scattering_coeff: 3.0e-6,
            mie_absorption_coeff: 0.3e-6,
            mie_scale_height: 1_200.0,
            mie_preferred_scattering_dir: 0.76,
            ozone_absorption_coeff: [0.65e-6, 1.881e-6, 0.085e-6],
            ozone_center_height: 25_000.0,
            ozone_falloff: 15_000.0,
            sun_angular_radius_cos: 0.999_956,
            _pad_atmo0: [0.0; 2],
            night_ambient_color: [0.0002, 0.0004, 0.0008],
            _pad_atmo1: 0.0,
            sky_ground_albedo: [0.3, 0.25, 0.2],
            sky_ground_brightness: 1.0,
            _pad_end: [0.0; 3],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq, Default)]
pub struct DdgiGridConstants {
    pub origin: [f32; 3],
    pub spacing: f32,
    pub counts: [u32; 3],
    pub probe_resolution: u32,
    pub max_distance: f32,
    pub normal_bias: f32,
    pub hysteresis: f32,
    pub update_stride: u32,
    pub frame_index: u32,
    pub reset: u32,
    pub total_probes: u32,
    pub _pad0: u32,
}

// --- SHARED HELPER STRUCTS ---

#[derive(Copy, Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn calculate(vertices: &[Vertex]) -> Self {
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

#[cfg(not(target_arch = "wasm32"))]
pub struct NativeRenderInit {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface<'static>,
}

pub enum RenderMessage {
    RenderData(Arc<RenderData>),
    RenderDelta(RenderDelta),
    Resize(PhysicalSize<u32>),
    WindowRecreated {
        window: Arc<Window>,
        size: PhysicalSize<u32>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    WindowRecreatedWithInit {
        size: PhysicalSize<u32>,
        render_init: NativeRenderInit,
    },
    Shutdown,
    Control(RenderControl),

    // --- Asset Pipeline Messages ---
    CreateMesh {
        id: usize,
        total_lods: usize,
        lods: Vec<MeshLodPayload>,
        bounds: Aabb,
    },
    CreateTexture {
        id: usize,
        name: String, // The file path for deduplication
        kind: AssetKind,
        data: Arc<[u8]>,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    },
    CreateMaterial(MaterialGpuData),

    EguiData(EguiRenderData),
}

pub fn render_message_payload_bytes(message: &RenderMessage) -> usize {
    match message {
        RenderMessage::CreateMesh { lods, .. } => {
            let mut bytes = 0usize;
            for lod in lods {
                bytes = bytes.saturating_add(
                    lod.vertices
                        .len()
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                );
                bytes = bytes
                    .saturating_add(lod.indices.len().saturating_mul(std::mem::size_of::<u32>()));
                bytes = bytes.saturating_add(
                    crate::graphics::common::meshlets::meshlet_lod_size_bytes(&lod.meshlets),
                );
            }
            bytes
        }
        RenderMessage::CreateTexture { data, .. } => data.len(),
        RenderMessage::CreateMaterial(mat) => std::mem::size_of_val(mat),
        _ => 0,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MipUploadLayout {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub offset: usize,
    pub size: usize,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
}

pub fn calc_mip_level_count(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32 + 1
}

pub fn mip_level_data_size(format: wgpu::TextureFormat, width: u32, height: u32) -> usize {
    let block_size = format.block_copy_size(None).unwrap_or(4) as usize;
    let (block_w, block_h) = format.block_dimensions();
    let blocks_w = ((width.max(1) + block_w - 1) / block_w).max(1);
    let blocks_h = ((height.max(1) + block_h - 1) / block_h).max(1);
    block_size * blocks_w as usize * blocks_h as usize
}

pub fn build_mip_uploads(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    data_len: usize,
    max_levels: u32,
) -> (Vec<MipUploadLayout>, usize) {
    let mut layouts = Vec::new();
    let mut offset = 0usize;
    let mut mip_width = width.max(1);
    let mut mip_height = height.max(1);
    for level in 0..max_levels {
        let block_size = format.block_copy_size(None).unwrap_or(4);
        let (block_w, block_h) = format.block_dimensions();
        let blocks_w = ((mip_width + block_w - 1) / block_w).max(1);
        let blocks_h = ((mip_height + block_h - 1) / block_h).max(1);
        let bytes_per_row = block_size * blocks_w;
        let rows_per_image = blocks_h;
        let size = bytes_per_row as usize * rows_per_image as usize;
        if offset + size > data_len {
            break;
        }
        layouts.push(MipUploadLayout {
            level,
            width: mip_width,
            height: mip_height,
            offset,
            size,
            bytes_per_row,
            rows_per_image,
        });
        offset += size;
        mip_width = (mip_width / 2).max(1);
        mip_height = (mip_height / 2).max(1);
    }
    (layouts, offset)
}

pub struct EguiRenderData {
    pub version: u64,
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub screen_descriptor: egui_wgpu::ScreenDescriptor,
}

#[derive(Clone)]
pub struct EguiNativeTextureBinding {
    pub texture_id: egui::TextureId,
    pub texture_view: wgpu::TextureView,
    pub texture_filter: wgpu::FilterMode,
}

#[derive(Clone, Default)]
pub struct EguiNativeTextures {
    pub bindings: Vec<EguiNativeTextureBinding>,
}

#[derive(Clone)]
pub struct EguiCachedTexture {
    pub image: egui::ImageData,
    pub options: egui::epaint::textures::TextureOptions,
}

/// CPU-side mirror of the egui texture atlas so new renderers can do full uploads (e.g. after a resize or GPU eviction) without relying on previous GPU state.
#[derive(Clone, Default)]
pub struct EguiTextureCache {
    pub atlas: HashMap<egui::TextureId, EguiCachedTexture>,
    pub last_delta: Option<egui::TexturesDelta>,
    /// Bumped whenever GPU state is considered lost (resize/evict) so passes can force a re-upload
    pub epoch: u64,
}

fn color_image_size(img: &ColorImage) -> [usize; 2] {
    img.size
}

fn resize_color_image(img: &mut ColorImage, new_size: [usize; 2]) {
    if img.size == new_size {
        return;
    }
    let old = std::mem::take(img);
    let mut new_img = ColorImage::filled(new_size, Color32::TRANSPARENT);
    let copy_w = old.size[0].min(new_size[0]);
    let copy_h = old.size[1].min(new_size[1]);
    for y in 0..copy_h {
        let src = y * old.size[0];
        let dst = y * new_size[0];
        new_img.pixels[dst..dst + copy_w].copy_from_slice(&old.pixels[src..src + copy_w]);
    }
    new_img.source_size = EguiVec2::new(new_size[0] as f32, new_size[1] as f32);
    *img = new_img;
}

fn apply_patch_to_color_image(base: &mut ColorImage, pos: [usize; 2], patch: &ColorImage) {
    if patch.size[0] == 0 || patch.size[1] == 0 {
        return;
    }
    let needed = [pos[0] + patch.size[0], pos[1] + patch.size[1]];
    if color_image_size(base) != needed
        && (color_image_size(base)[0] < needed[0] || color_image_size(base)[1] < needed[1])
    {
        resize_color_image(
            base,
            [needed[0].max(base.size[0]), needed[1].max(base.size[1])],
        );
    }

    for y in 0..patch.size[1] {
        let dst_offset = (pos[1] + y) * base.size[0] + pos[0];
        let src_offset = y * patch.size[0];
        base.pixels[dst_offset..dst_offset + patch.size[0]]
            .copy_from_slice(&patch.pixels[src_offset..src_offset + patch.size[0]]);
    }
}

pub fn apply_egui_delta(cache: &mut EguiTextureCache, delta: &egui::TexturesDelta) {
    for id in &delta.free {
        cache.atlas.remove(id);
    }

    for (id, image_delta) in &delta.set {
        let image_size = image_delta.image.size();
        if image_size[0] == 0 || image_size[1] == 0 {
            continue;
        }

        let mut base_image = match cache.atlas.remove(id) {
            Some(existing) => match existing.image {
                egui::ImageData::Color(color) => (*color).clone(),
            },
            None => ColorImage::filled(image_size, Color32::TRANSPARENT),
        };

        match image_delta.pos {
            Some(pos) => {
                let patch = match &image_delta.image {
                    egui::ImageData::Color(c) => (**c).clone(),
                };
                apply_patch_to_color_image(&mut base_image, pos, &patch);
            }
            None => {
                base_image = match &image_delta.image {
                    egui::ImageData::Color(c) => (**c).clone(),
                };
            }
        }

        cache.atlas.insert(
            *id,
            EguiCachedTexture {
                image: egui::ImageData::Color(Arc::new(base_image)),
                options: image_delta.options,
            },
        );
    }

    if !delta.set.is_empty() || !delta.free.is_empty() {
        cache.last_delta = Some(delta.clone());
    }
}

pub fn build_full_egui_delta(cache: &EguiTextureCache) -> Option<egui::TexturesDelta> {
    if cache.atlas.is_empty() {
        return None;
    }
    let mut full = egui::TexturesDelta::default();
    for (id, tex) in &cache.atlas {
        full.set.push((
            *id,
            egui::epaint::ImageDelta::full(tex.image.clone(), tex.options),
        ));
    }
    Some(full)
}

pub struct RendererStats {
    pub vram_used_bytes: AtomicU64,
    pub vram_soft_limit_bytes: AtomicU64,
    pub vram_hard_limit_bytes: AtomicU64,
    pub vram_soft_limit_per_kind: [AtomicU64; 9],
    pub vram_hard_limit_per_kind: [AtomicU64; 9],
    pub resident_resources: AtomicU32,
    pub idle_frames: AtomicU32,
    pub occlusion_status: AtomicU32,
    pub occlusion_last_frame: AtomicU32,
    pub occlusion_instance_count: AtomicU32,
    pub occlusion_camera_stable: AtomicU32,
    pub gpu_draw_count: AtomicU32,
    pub gpu_mesh_count: AtomicU32,
    pub gpu_instance_capacity: AtomicU32,
    pub gpu_visible_capacity: AtomicU32,
    pub gpu_shadow_capacity: AtomicU32,
    pub gpu_total_capacity: AtomicU64,
    pub gpu_fallbacks: AtomicU32,
    pub profiling_enabled: AtomicBool,
    pub render_prepare_globals_us: AtomicU64,
    pub render_streaming_plan_us: AtomicU64,
    pub render_occlusion_us: AtomicU64,
    pub render_graph_us: AtomicU64,
    pub render_graph_pass_us: AtomicU64,
    pub render_graph_encoder_create_us: AtomicU64,
    pub render_graph_encoder_finish_us: AtomicU64,
    pub render_graph_overhead_us: AtomicU64,
    pub render_resource_mgmt_us: AtomicU64,
    pub render_ui_build_us: AtomicU64,
    pub render_ui_rebuilt: AtomicU32,
    pub render_ui_command_count: AtomicU32,
    pub render_ui_instance_count: AtomicU32,
    pub render_ui_batch_count: AtomicU32,
    pub render_ui_texture_count: AtomicU32,
    pub render_acquire_us: AtomicU64,
    pub render_submit_us: AtomicU64,
    pub render_present_us: AtomicU64,
    pub pass_timings: RwLock<Vec<RenderPassTiming>>,
}

#[derive(Clone, Debug)]
pub struct RenderDeviceCaps {
    pub adapter_info: wgpu::AdapterInfo,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
    pub downlevel_caps: wgpu::DownlevelCapabilities,
}

impl RenderDeviceCaps {
    pub fn supports_mesh_shaders(&self) -> bool {
        self.features
            .contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
    }

    pub fn supports_mesh_shader_multiview(&self) -> bool {
        self.features
            .contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW)
    }

    pub fn supports_mesh_shader_points(&self) -> bool {
        self.features
            .contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER_POINTS)
    }

    fn mesh_limits_ok(&self) -> bool {
        let limits = &self.limits;
        limits.max_task_mesh_workgroup_total_count > 0
            && limits.max_task_mesh_workgroups_per_dimension > 0
            && limits.max_mesh_invocations_per_workgroup >= MESHLET_WORKGROUP_SIZE
            && limits.max_mesh_invocations_per_dimension > 0
            && limits.max_mesh_output_vertices >= MESHLET_MAX_VERTS as u32
            && limits.max_mesh_output_primitives >= MESHLET_MAX_PRIMS as u32
    }

    pub fn supports_mesh_pipeline(&self) -> bool {
        if self.adapter_info.backend != wgpu::Backend::Vulkan {
            return false;
        }
        self.features
            .contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
            && self.mesh_limits_ok()
    }

    pub fn supports_multi_draw_indirect(&self) -> bool {
        self.downlevel_caps
            .flags
            .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
    }

    pub fn supports_multi_draw_indirect_count(&self) -> bool {
        self.features
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
    }

    pub fn supports_immediates(&self) -> bool {
        self.features.contains(wgpu::Features::IMMEDIATES)
    }

    pub fn supports_compute(&self) -> bool {
        self.downlevel_caps
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    }

    pub fn supports_vertex_storage(&self) -> bool {
        self.downlevel_caps
            .flags
            .contains(wgpu::DownlevelFlags::VERTEX_STORAGE)
    }

    pub fn supports_fragment_storage_buffers(&self) -> bool {
        self.downlevel_caps
            .flags
            .contains(wgpu::DownlevelFlags::FRAGMENT_STORAGE)
            && self.limits.max_storage_buffers_per_shader_stage > 0
    }

    pub fn supports_compute_storage_textures(&self) -> bool {
        self.supports_compute() && self.limits.max_storage_textures_per_shader_stage > 0
    }

    pub fn supports_independent_blend(&self) -> bool {
        self.downlevel_caps
            .flags
            .contains(wgpu::DownlevelFlags::INDEPENDENT_BLEND)
    }

    pub fn transient_textures_save_memory(&self) -> bool {
        self.adapter_info.transient_saves_memory
    }
}

pub fn mesh_shader_visibility(device: &wgpu::Device) -> wgpu::ShaderStages {
    if device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
    {
        wgpu::ShaderStages::MESH
    } else {
        wgpu::ShaderStages::empty()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeshTaskTiling {
    pub tile_meshlets: u32,
    pub tile_instances: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub task_count: u32,
}

pub fn mesh_task_tiling(
    limits: &wgpu::Limits,
    meshlet_count: u32,
    instance_count: u32,
) -> MeshTaskTiling {
    if meshlet_count == 0 || instance_count == 0 {
        return MeshTaskTiling {
            tile_meshlets: 0,
            tile_instances: 0,
            tiles_x: 0,
            tiles_y: 0,
            task_count: 0,
        };
    }

    let max_dim = limits.max_task_mesh_workgroups_per_dimension.max(1);
    let max_total = limits.max_task_mesh_workgroup_total_count.max(1);

    let mut tile_meshlets = meshlet_count.min(max_dim);
    let mut tile_instances = instance_count.min(max_dim);

    let max_total_u64 = max_total as u64;
    let mut total = (tile_meshlets as u64) * (tile_instances as u64);
    if total > max_total_u64 {
        tile_instances = ((max_total_u64 / tile_meshlets as u64).max(1)) as u32;
        tile_instances = tile_instances.min(max_dim);
        total = (tile_meshlets as u64) * (tile_instances as u64);
        if total > max_total_u64 {
            tile_meshlets = ((max_total_u64 / tile_instances as u64).max(1)) as u32;
            tile_meshlets = tile_meshlets.min(max_dim);
            total = (tile_meshlets as u64) * (tile_instances as u64);
        }
        if total > max_total_u64 {
            tile_meshlets = max_total.min(max_dim);
            tile_instances = 1;
        }
    }

    let tiles_x = (meshlet_count + tile_meshlets - 1) / tile_meshlets;
    let tiles_y = (instance_count + tile_instances - 1) / tile_instances;
    let task_count = tiles_x.saturating_mul(tiles_y);

    MeshTaskTiling {
        tile_meshlets,
        tile_instances,
        tiles_x,
        tiles_y,
        task_count,
    }
}

pub fn color_load_op(clear: wgpu::Color, dont_care: bool) -> wgpu::LoadOp<wgpu::Color> {
    if dont_care {
        // Safety: callers should only opt in when they fully overwrite the attachment
        wgpu::LoadOp::DontCare(unsafe { wgpu::LoadOpDontCare::enabled() })
    } else {
        wgpu::LoadOp::Clear(clear)
    }
}

pub fn transient_usage(usage: wgpu::TextureUsages, enable_transient: bool) -> wgpu::TextureUsages {
    if !enable_transient {
        return usage;
    }
    let extra = usage - wgpu::TextureUsages::RENDER_ATTACHMENT;
    if extra.is_empty() {
        usage | wgpu::TextureUsages::TRANSIENT
    } else {
        usage
    }
}

impl Default for RendererStats {
    fn default() -> Self {
        Self {
            vram_used_bytes: AtomicU64::new(0),
            vram_soft_limit_bytes: AtomicU64::new(0),
            vram_hard_limit_bytes: AtomicU64::new(0),
            vram_soft_limit_per_kind: std::array::from_fn(|_| AtomicU64::new(0)),
            vram_hard_limit_per_kind: std::array::from_fn(|_| AtomicU64::new(0)),
            resident_resources: AtomicU32::new(0),
            idle_frames: AtomicU32::new(0),
            occlusion_status: AtomicU32::new(OCCLUSION_STATUS_DISABLED),
            occlusion_last_frame: AtomicU32::new(u32::MAX),
            occlusion_instance_count: AtomicU32::new(0),
            occlusion_camera_stable: AtomicU32::new(0),
            gpu_draw_count: AtomicU32::new(0),
            gpu_mesh_count: AtomicU32::new(0),
            gpu_instance_capacity: AtomicU32::new(0),
            gpu_visible_capacity: AtomicU32::new(0),
            gpu_shadow_capacity: AtomicU32::new(0),
            gpu_total_capacity: AtomicU64::new(0),
            gpu_fallbacks: AtomicU32::new(0),
            profiling_enabled: AtomicBool::new(false),
            render_prepare_globals_us: AtomicU64::new(0),
            render_streaming_plan_us: AtomicU64::new(0),
            render_occlusion_us: AtomicU64::new(0),
            render_graph_us: AtomicU64::new(0),
            render_graph_pass_us: AtomicU64::new(0),
            render_graph_encoder_create_us: AtomicU64::new(0),
            render_graph_encoder_finish_us: AtomicU64::new(0),
            render_graph_overhead_us: AtomicU64::new(0),
            render_resource_mgmt_us: AtomicU64::new(0),
            render_ui_build_us: AtomicU64::new(0),
            render_ui_rebuilt: AtomicU32::new(0),
            render_ui_command_count: AtomicU32::new(0),
            render_ui_instance_count: AtomicU32::new(0),
            render_ui_batch_count: AtomicU32::new(0),
            render_ui_texture_count: AtomicU32::new(0),
            render_acquire_us: AtomicU64::new(0),
            render_submit_us: AtomicU64::new(0),
            render_present_us: AtomicU64::new(0),
            pass_timings: RwLock::new(Vec::new()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderPassTiming {
    pub name: String,
    pub order: usize,
    pub enabled: bool,
    pub duration_us: u64,
}

pub const OCCLUSION_STATUS_DISABLED: u32 = 0;
pub const OCCLUSION_STATUS_NO_GBUFFER: u32 = 1;
pub const OCCLUSION_STATUS_NO_INSTANCES: u32 = 2;
pub const OCCLUSION_STATUS_NO_HIZ: u32 = 3;
pub const OCCLUSION_STATUS_RAN: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StreamingCaps {
    pub global: usize,
    pub mesh: usize,
    pub material: usize,
    pub texture: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StreamingByteCaps {
    pub global: u64,
    pub mesh: u64,
    pub material: u64,
    pub texture: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StreamingTuning {
    pub caps_none: StreamingCaps,
    pub caps_soft: StreamingCaps,
    pub caps_hard: StreamingCaps,
    pub caps_bytes_none: StreamingByteCaps,
    pub caps_bytes_soft: StreamingByteCaps,
    pub caps_bytes_hard: StreamingByteCaps,
    pub priority_floor_none: f32,
    pub priority_floor_soft: f32,
    pub priority_floor_hard: f32,
    pub priority_size_bias: f32,
    pub priority_distance_bias: f32,
    pub sprite_screen_priority_multiplier: f32,
    pub sprite_screen_distance_bias_min: f32,
    pub sprite_sequence_prefetch_frames: u32,
    pub sprite_sequence_prefetch_priority_scale: f32,
    pub sprite_sequence_pingpong_prefetch_priority_scale: f32,
    pub sprite_texture_priority_scale: f32,
    pub priority_lod_bias: f32,
    pub lod_bias_soft: usize,
    pub lod_bias_hard: usize,
    pub force_lowest_lod_hard: bool,
    pub force_low_res_hard: bool,
    pub pressure_release_frames: u32,
    pub soft_upgrade_delay_frames: u32,
    pub upgrade_cooldown_frames: u32,
    pub prediction_frames: f32,
    pub prediction_motion_epsilon: f32,
    pub prediction_distance_threshold: f32,
    pub resident_priority_boost: f32,
    pub inflight_priority_boost: f32,
    pub shadow_priority_boost: f32,
    pub recent_evict_penalty: f32,
    pub recent_evict_frames: u32,
    pub evict_retry_frames: u32,
    pub evict_retry_priority: f32,
    pub upgrade_priority_soft: f32,
    pub low_res_priority_soft: f32,
    pub low_res_priority_hard: f32,
    pub priority_near: f32,
    pub priority_critical: f32,
    pub inflight_cooldown_frames: u32,
    pub priority_bump_factor: f32,
    pub fallback_mesh_bytes: u64,
    pub fallback_material_bytes: u64,
    pub fallback_texture_bytes: u64,
    pub evict_soft_grace_frames: u32,
    pub evict_hard_grace_frames: u32,
    pub evict_soft_protect_priority: f32,
    pub evict_unplanned_idle_frames: u32,
    pub hard_idle_frames_before_evict: u32,
    pub pool_idle_frames_before_evict: u32,
    pub pool_streaming_min_residency_frames: u32,
    pub pool_max_evictions_per_tick: usize,
    pub pool_eviction_scan_budget: usize,
    pub pool_eviction_purge_budget: usize,
    pub asset_map_initial_capacity: usize,
    pub asset_map_max_load_factor: f32,
    pub transient_heap_initial_capacity: usize,
    pub transient_heap_max_load_factor: f32,
    pub transient_heap_max_free_per_desc: usize,
    pub transient_heap_max_total_free: usize,
    pub graph_encoder_batch_size: usize,
}

impl Default for StreamingTuning {
    fn default() -> Self {
        Self {
            caps_none: StreamingCaps {
                global: 32_768,
                mesh: 32_768,
                material: 32_768,
                texture: 32_768,
            },
            caps_soft: StreamingCaps {
                global: 16_384,
                mesh: 4_096,
                material: 4_096,
                texture: 8_192,
            },
            caps_hard: StreamingCaps {
                global: 8_192,
                mesh: 2_048,
                material: 2_048,
                texture: 4_096,
            },
            caps_bytes_none: StreamingByteCaps {
                global: 2 * 1024 * 1024 * 1024,
                mesh: 1024 * 1024 * 1024,
                material: 64 * 1024 * 1024,
                texture: 1024 * 1024 * 1024,
            },
            caps_bytes_soft: StreamingByteCaps {
                global: 1024 * 1024 * 1024,
                mesh: 512 * 1024 * 1024,
                material: 32 * 1024 * 1024,
                texture: 512 * 1024 * 1024,
            },
            caps_bytes_hard: StreamingByteCaps {
                global: 512 * 1024 * 1024,
                mesh: 256 * 1024 * 1024,
                material: 16 * 1024 * 1024,
                texture: 256 * 1024 * 1024,
            },
            priority_floor_none: 0.0,
            priority_floor_soft: 0.005,
            priority_floor_hard: 0.02,
            priority_size_bias: 1.0,
            priority_distance_bias: 1.0,
            sprite_screen_priority_multiplier: 2.0,
            sprite_screen_distance_bias_min: 0.25,
            sprite_sequence_prefetch_frames: 2,
            sprite_sequence_prefetch_priority_scale: 0.85,
            sprite_sequence_pingpong_prefetch_priority_scale: 0.8,
            sprite_texture_priority_scale: 0.75,
            priority_lod_bias: 1.0,
            lod_bias_soft: 1,
            lod_bias_hard: 2,
            force_lowest_lod_hard: true,
            force_low_res_hard: true,
            pressure_release_frames: 24,
            soft_upgrade_delay_frames: 12,
            upgrade_cooldown_frames: 48,
            prediction_frames: 6.0,
            prediction_motion_epsilon: 1.0e-6,
            prediction_distance_threshold: 0.1,
            resident_priority_boost: 1.15,
            inflight_priority_boost: 1.08,
            shadow_priority_boost: 1.15,
            recent_evict_penalty: 0.6,
            recent_evict_frames: 80,
            evict_retry_frames: 80,
            evict_retry_priority: 0.08,
            upgrade_priority_soft: 0.16,
            low_res_priority_soft: 0.08,
            low_res_priority_hard: 0.2,
            priority_near: 0.12,
            priority_critical: 0.28,
            inflight_cooldown_frames: 6,
            priority_bump_factor: 1.05,
            fallback_mesh_bytes: 4 * 1024 * 1024,
            fallback_material_bytes: std::mem::size_of::<MaterialGpuData>() as u64,
            fallback_texture_bytes: 4 * 1024 * 1024,
            evict_soft_grace_frames: 8,
            evict_hard_grace_frames: 2,
            evict_soft_protect_priority: 0.05,
            evict_unplanned_idle_frames: 120,
            hard_idle_frames_before_evict: 2,
            pool_idle_frames_before_evict: 120,
            pool_streaming_min_residency_frames: 90,
            pool_max_evictions_per_tick: 12,
            pool_eviction_scan_budget: 2048,
            pool_eviction_purge_budget: 64,
            asset_map_initial_capacity: 1024,
            asset_map_max_load_factor: 0.75,
            transient_heap_initial_capacity: 64,
            transient_heap_max_load_factor: 0.75,
            transient_heap_max_free_per_desc: 32,
            transient_heap_max_total_free: 10_000,
            graph_encoder_batch_size: 0,
        }
    }
}

impl StreamingTuning {
    pub fn pool_config(&self) -> GpuResourcePoolConfig {
        GpuResourcePoolConfig {
            asset_map_initial_capacity: self.asset_map_initial_capacity,
            asset_map_max_load_factor: self.asset_map_max_load_factor,
            transient_heap: TransientHeapConfig {
                initial_capacity: self.transient_heap_initial_capacity,
                max_load_factor: self.transient_heap_max_load_factor,
                max_free_per_desc: self.transient_heap_max_free_per_desc,
                max_total_free: self.transient_heap_max_total_free,
            },
            idle_frames_before_evict: self.pool_idle_frames_before_evict,
            streaming_min_residency_frames: self.pool_streaming_min_residency_frames,
            max_evictions_per_tick: self.pool_max_evictions_per_tick,
            eviction_scan_budget: self.pool_eviction_scan_budget,
            eviction_purge_budget: self.pool_eviction_purge_budget,
        }
    }
}

pub enum RenderControl {
    SetGpuBudget {
        soft_limit_bytes: u64,
        hard_limit_bytes: u64,
        idle_frames: Option<u32>,
        per_kind_soft: Option<Vec<u64>>,
        per_kind_hard: Option<Vec<u64>>,
    },
    SetStreamingTuning(StreamingTuning),
    EvictAll {
        restream_assets: bool,
    },
    SetPassEnabled {
        pass: String,
        enabled: bool,
    },
    RecreateDevice {
        backend: WgpuBackend,
        binding_backend: BindingBackendChoice,
        allow_experimental_features: bool,
    },
}
