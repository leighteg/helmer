pub mod composite;
pub mod ddgi_probe_update;
pub mod ddgi_resample;
pub mod debug_composite;
pub mod depth_copy;
pub mod downsample;
pub mod egui;
pub mod forward;
pub mod gbuffer;
pub mod hiz;
pub mod lighting;
pub mod raytraced;
pub mod reflection_combine;
pub mod rt_reflections;
pub mod rt_reflections_denoise;
pub mod shadow;
pub mod sky;
pub mod ssgi;
pub mod ssgi_denoise;
pub mod ssgi_upsample;
pub mod ssr;

use std::{ops::Range, sync::Arc};

use glam::Mat4;
use winit::dpi::PhysicalSize;

use crate::graphics::{
    backend::binding_backend::BindingBackendKind,
    common::{
        config::RenderConfig,
        renderer::{LightData, RenderDeviceCaps, ShaderConstants, SkyUniforms},
    },
    graph::definition::resource_id::ResourceId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleMode {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GBufferBundleKey {
    pub mode: BundleMode,
    pub draw_version: u64,
    pub material_bindings_version: u64,
    pub texture_array_size: u32,
    pub binding_backend: BindingBackendKind,
    pub resource_epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowBundleKey {
    pub mode: BundleMode,
    pub draw_version: u64,
    pub matrices_version: u64,
    pub resource_epoch: u64,
}

/// Per-frame GPU data made available to passes via "FrameInputHub"
#[derive(Clone)]
pub struct FrameGlobals {
    pub frame_index: u32,
    pub device_caps: Arc<RenderDeviceCaps>,
    pub binding_backend: BindingBackendKind,
    pub camera_buffer: wgpu::Buffer,
    pub lights_buffer: Option<wgpu::Buffer>,
    pub lights_len: u32,
    pub render_constants_buffer: wgpu::Buffer,
    pub ddgi_grid_buffer: wgpu::Buffer,
    pub shadow_uniforms_buffer: Option<wgpu::Buffer>,
    pub shadow_matrices_buffer: Option<wgpu::Buffer>,
    pub sky_buffer: wgpu::Buffer,
    pub material_buffer: Option<wgpu::Buffer>,
    pub material_textures: Option<Arc<Vec<MaterialTextureSet>>>,
    pub material_bindings_version: u64,
    pub texture_views: Vec<wgpu::TextureView>,
    pub texture_array_size: u32,
    pub rt_texture_arrays: Option<RayTracingTextureArrays>,
    pub pbr_sampler: wgpu::Sampler,
    pub shadow_sampler: wgpu::Sampler,
    pub scene_sampler: wgpu::Sampler,
    pub point_sampler: wgpu::Sampler,
    pub blue_noise_view: wgpu::TextureView,
    pub blue_noise_sampler: wgpu::Sampler,
    pub fallback_view: wgpu::TextureView,
    pub fallback_volume_view: wgpu::TextureView,
    pub hiz_view: Option<wgpu::TextureView>,
    pub ibl_brdf_view: wgpu::TextureView,
    pub ibl_irradiance_view: wgpu::TextureView,
    pub ibl_prefiltered_view: wgpu::TextureView,
    pub ibl_sampler: wgpu::Sampler,
    pub brdf_lut_sampler: wgpu::Sampler,
    pub atmosphere_bind_group: wgpu::BindGroup,
    pub debug_params_buffer: wgpu::Buffer,
    pub gbuffer_instances: Option<InstanceBuffer>,
    pub gbuffer_batches: Arc<Vec<DrawBatch>>,
    pub gbuffer_indirect: Option<wgpu::Buffer>,
    pub shadow_instances: Option<InstanceBuffer>,
    pub shadow_batches: Arc<Vec<DrawBatch>>,
    pub shadow_indirect: Option<wgpu::Buffer>,
    pub gbuffer_mesh_tasks: Option<wgpu::Buffer>,
    pub shadow_mesh_tasks: Option<wgpu::Buffer>,
    pub gpu_draws: Arc<Vec<IndirectDrawBatch>>,
    pub gpu_instance_count: u32,
    pub forward_instances: Option<InstanceBuffer>,
    pub forward_batches: Arc<Vec<DrawBatch>>,
    pub alpha: f32,
    pub camera_view_proj: Mat4,
    pub prev_view_proj: Mat4,
    pub lights: Vec<LightData>,
    pub shader_constants: ShaderConstants,
    pub sky_uniforms: SkyUniforms,
    pub surface_size: PhysicalSize<u32>,
    pub render_config: RenderConfig,
    pub occlusion_camera_stable: bool,
    pub gbuffer_bundle_key: GBufferBundleKey,
    pub shadow_bundle_key: ShadowBundleKey,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugCompositeParams {
    pub flags: u32,
    pub _pad0: [u32; 3],
    pub _pad1: [u32; 4],
    pub _pad2: [u32; 4],
}

#[derive(Clone)]
pub struct InstanceBuffer {
    pub buffer: wgpu::Buffer,
    pub count: u32,
    pub stride: wgpu::BufferAddress,
}

#[derive(Clone)]
pub struct MaterialTextureSet {
    pub albedo: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub metallic_roughness: wgpu::TextureView,
    pub emission: wgpu::TextureView,
}

#[derive(Clone)]
pub struct RayTracingTextureArrays {
    pub albedo: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub metallic_roughness: wgpu::TextureView,
    pub emission: wgpu::TextureView,
    pub layers: u32,
}

#[derive(Clone)]
pub struct DrawBatch {
    pub mesh_id: usize,
    pub lod: usize,
    pub index_count: u32,
    pub instance_range: Range<u32>,
    pub material_id: u32,
    pub vertex: ResourceId,
    pub index: ResourceId,
    pub meshlet_descs: ResourceId,
    pub meshlet_vertices: ResourceId,
    pub meshlet_indices: ResourceId,
    pub meshlet_count: u32,
}

#[derive(Clone)]
pub struct IndirectDrawBatch {
    pub mesh_id: usize,
    pub lod: usize,
    pub material_id: u32,
    pub vertex: ResourceId,
    pub index: ResourceId,
    pub meshlet_descs: ResourceId,
    pub meshlet_vertices: ResourceId,
    pub meshlet_indices: ResourceId,
    pub meshlet_count: u32,
    pub instance_base: u32,
    pub instance_capacity: u32,
    pub indirect_offset: u64,
    pub mesh_task_offset: u64,
    pub mesh_task_count: u32,
    pub mesh_task_tile_meshlets: u32,
    pub mesh_task_tile_instances: u32,
}

/// Swapchain output for the frame
#[derive(Clone)]
pub struct SwapchainFrameInput {
    pub view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
}

#[derive(Clone)]
pub struct RayTracingFrameInput {
    pub rt_extent: PhysicalSize<u32>,
    pub tlas_node_count: u32,
    pub blas_nodes: wgpu::Buffer,
    pub blas_indices: wgpu::Buffer,
    pub blas_triangles: wgpu::Buffer,
    pub blas_descs: wgpu::Buffer,
    pub tlas_nodes: wgpu::Buffer,
    pub tlas_indices: wgpu::Buffer,
    pub instances: wgpu::Buffer,
    pub constants: wgpu::Buffer,
}
