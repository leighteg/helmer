use crossbeam_channel::{Sender, TrySendError};
use hashbrown::{HashMap, HashSet};
use std::{env, sync::Arc, time::Instant};
use tracing::{info, warn};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::graphics::passes::gbuffer::GBufferPass;
use crate::graphics::passes::shadow::ShadowPass;
use crate::graphics::{
    backend::{
        bind_group::BindGroupBackend,
        binding_backend::{
            BindingBackend, BindingBackendChoice, BindingBackendKind, BindlessConfig,
        },
        bindless_fallback::BindlessFallbackBackend,
        bindless_modern::BindlessModernBackend,
    },
    config::RenderConfig,
    graph::{
        definition::{
            resource_desc::ResourceDesc,
            resource_flags::{ResourceFlags, ResourceUsageHints},
            resource_id::{ResourceId, ResourceKind},
        },
        logic::{
            frame_inputs::FrameInputHub,
            gpu_resource_pool::GpuResourcePool,
            graph_executor::{RenderGraphExecutionStats, RenderGraphExecutor},
            render_graph::{RenderGraph, RenderGraphCompilation},
            residency::Residency,
        },
    },
    passes::{
        BundleMode, FrameGlobals, GBufferBundleKey, IndirectDrawBatch, MaterialTextureSet,
        ShadowBundleKey, SwapchainFrameInput,
    },
    render_graphs::default_graph_spec,
    renderer_common::{
        atmosphere::AtmospherePrecomputer,
        common::{
            Aabb, AssetStreamKind, AssetStreamingRequest, CameraUniforms, CascadeUniform,
            EguiTextureCache, InstanceRaw, LightData, MaterialShaderData, MeshletDesc,
            MeshletLodData, OCCLUSION_STATUS_DISABLED, OCCLUSION_STATUS_NO_GBUFFER,
            OCCLUSION_STATUS_NO_HIZ, OCCLUSION_STATUS_NO_INSTANCES, OCCLUSION_STATUS_RAN,
            RenderControl, RenderData, RenderDelta, RenderDeviceCaps, RenderLight,
            RenderLightDelta, RenderMessage, RenderObject, RenderObjectDelta, RenderPassTiming,
            RendererStats, ShaderConstants, ShadowUniforms, SkyUniforms, StreamingTuning, Vertex,
            apply_egui_delta, mesh_task_tiling,
        },
        error::RendererError,
        graph::{RenderGraphBuildParams, RenderGraphConfigSignature, RenderGraphSpec},
        meshlets::build_meshlet_lod,
        mipmap::MipmapGenerator,
    },
};
use crate::provided::components::{Camera, LightType, Transform};
use crate::runtime::asset_server::MaterialGpuData;
use glam::{Mat3, Mat4, Quat, Vec3, Vec4Swizzles};

const FALLBACK_MESH_KEY: usize = usize::MAX;
const GPU_FALLBACK_MESH_INDEX: u32 = 0;

struct MeshLodResource {
    buffer: ResourceId,
    index_count: u32,
    meshlets: MeshletGpu,
}

struct MeshletGpu {
    descs: ResourceId,
    vertices: ResourceId,
    indices: ResourceId,
    count: u32,
}

struct MeshGpu {
    vertex: ResourceId,
    lods: Vec<MeshLodResource>,
    bounds: Aabb,
}

struct ResolvedDrawMesh {
    mesh_id: usize,
    lod_index: usize,
    bounds_center: Vec3,
    bounds_extents: Vec3,
}

struct MaterialEntry {
    buffer: ResourceId,
    meta: MaterialGpuData,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstanceInput {
    prev_model: [[f32; 4]; 4],
    curr_model: [[f32; 4]; 4],
    material_id: u32,
    mesh_id: u32,
    casts_shadow: u32,
    _pad0: u32,
    bounds_center: [f32; 4],
    bounds_extents: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMeshMeta {
    lod_count: u32,
    base_draw: u32,
    instance_capacity: u32,
    base_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCullParams {
    instance_count: u32,
    draw_count: u32,
    mesh_count: u32,
    flags: u32,
    lod0_dist_sq: f32,
    lod1_dist_sq: f32,
    lod2_dist_sq: f32,
    alpha: f32,
    occlusion_depth_bias: f32,
    occlusion_rect_pad: f32,
    output_capacity: u32,
    _pad1: u32,
}

#[derive(Clone, Copy, PartialEq)]
struct GpuCullSignature {
    frustum_culling: bool,
    occlusion_culling: bool,
    lod: bool,
    lod0: f32,
    lod1: f32,
    lod2: f32,
    depth_bias: f32,
    rect_pad: f32,
    mesh_tasks_enabled: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawMeshTasksIndirectArgs {
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshTaskMeta {
    meshlet_count: u32,
    instance_capacity: u32,
    tile_meshlets: u32,
    tile_instances: u32,
    task_offset: u32,
    _pad0: [u32; 3],
}

struct ReusableBuffer {
    // One slot per in-flight frame to avoid clobbering buffers still in use by the GPU.
    slots: Vec<(Option<wgpu::Buffer>, u64, wgpu::BufferUsages)>,
    label: &'static str,
}

impl ReusableBuffer {
    fn new(label: &'static str, frames_in_flight: usize) -> Self {
        let slots = frames_in_flight.max(1);
        Self {
            slots: (0..slots)
                .map(|_| (None, 0, wgpu::BufferUsages::empty()))
                .collect(),
            label,
        }
    }

    fn resize_slots(&mut self, frames_in_flight: usize) {
        let slots = frames_in_flight.max(1);
        if self.slots.len() == slots {
            return;
        }
        self.slots = (0..slots)
            .map(|_| (None, 0, wgpu::BufferUsages::empty()))
            .collect();
    }

    fn ensure(
        &mut self,
        device: &wgpu::Device,
        needed: u64,
        usage: wgpu::BufferUsages,
        frame_index: u32,
    ) -> &wgpu::Buffer {
        self.ensure_with_status(device, needed, usage, frame_index)
            .0
    }

    fn ensure_with_status(
        &mut self,
        device: &wgpu::Device,
        needed: u64,
        usage: wgpu::BufferUsages,
        frame_index: u32,
    ) -> (&wgpu::Buffer, bool) {
        let slot_idx = (frame_index as usize) % self.slots.len();
        let (buffer, size, prev_usage) = &mut self.slots[slot_idx];

        // Allocate with some headroom to reduce re-allocations.
        let target_size = needed.max(256).next_power_of_two();
        let needs_realloc = buffer
            .as_ref()
            .map_or(true, |_| *size < target_size || *prev_usage != usage);

        if needs_realloc {
            *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: target_size,
                usage,
                mapped_at_creation: false,
            }));
            *size = target_size;
            *prev_usage = usage;
        }

        (buffer.as_ref().expect("buffer must exist"), needs_realloc)
    }
}

struct BufferCache {
    camera: ReusableBuffer,
    lights: ReusableBuffer,
    render_constants: ReusableBuffer,
    sky: ReusableBuffer,
    shadow_uniforms: ReusableBuffer,
    shadow_matrices: ReusableBuffer,
    materials: ReusableBuffer,
    debug_params: ReusableBuffer,
    occlusion_params: ReusableBuffer,
    gpu_cull_params: ReusableBuffer,
    gbuffer_instances: ReusableBuffer,
    shadow_instances: ReusableBuffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OcclusionParams {
    instance_count: u32,
    _pad0: [u32; 3],
    depth_bias: f32,
    rect_pad: f32,
    _pad1: [f32; 2],
}

impl BufferCache {
    fn new(frames_in_flight: usize) -> Self {
        Self {
            camera: ReusableBuffer::new("GraphRenderer/Camera", frames_in_flight),
            lights: ReusableBuffer::new("GraphRenderer/Lights", frames_in_flight),
            render_constants: ReusableBuffer::new(
                "GraphRenderer/RenderConstants",
                frames_in_flight,
            ),
            sky: ReusableBuffer::new("GraphRenderer/SkyUniforms", frames_in_flight),
            shadow_uniforms: ReusableBuffer::new("GraphRenderer/ShadowUniforms", frames_in_flight),
            shadow_matrices: ReusableBuffer::new("GraphRenderer/ShadowMatrices", frames_in_flight),
            materials: ReusableBuffer::new("GraphRenderer/Materials", frames_in_flight),
            debug_params: ReusableBuffer::new("GraphRenderer/DebugParams", frames_in_flight),
            occlusion_params: ReusableBuffer::new(
                "GraphRenderer/OcclusionParams",
                frames_in_flight,
            ),
            gpu_cull_params: ReusableBuffer::new("GraphRenderer/GpuCullParams", frames_in_flight),
            gbuffer_instances: ReusableBuffer::new(
                "GraphRenderer/GBufferInstances",
                frames_in_flight,
            ),
            shadow_instances: ReusableBuffer::new(
                "GraphRenderer/ShadowInstances",
                frames_in_flight,
            ),
        }
    }

    fn resize(&mut self, frames_in_flight: usize) {
        self.camera.resize_slots(frames_in_flight);
        self.lights.resize_slots(frames_in_flight);
        self.render_constants.resize_slots(frames_in_flight);
        self.sky.resize_slots(frames_in_flight);
        self.shadow_uniforms.resize_slots(frames_in_flight);
        self.shadow_matrices.resize_slots(frames_in_flight);
        self.materials.resize_slots(frames_in_flight);
        self.debug_params.resize_slots(frames_in_flight);
        self.occlusion_params.resize_slots(frames_in_flight);
        self.gpu_cull_params.resize_slots(frames_in_flight);
        self.gbuffer_instances.resize_slots(frames_in_flight);
        self.shadow_instances.resize_slots(frames_in_flight);
    }
}

/// Dense slot storage keyed by asset id. Avoids hash lookups in hot paths.
struct SlotVec<T> {
    entries: Vec<Option<T>>,
}

impl<T> SlotVec<T> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn insert(&mut self, id: usize, value: T) {
        if id >= self.entries.len() {
            self.entries.resize_with(id + 1, || None);
        }
        self.entries[id] = Some(value);
    }

    fn get(&self, id: usize) -> Option<&T> {
        self.entries.get(id).and_then(|v| v.as_ref())
    }

    fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        self.entries.get_mut(id).and_then(|v| v.as_mut())
    }

    fn remove(&mut self, id: usize) -> Option<T> {
        if id >= self.entries.len() {
            None
        } else {
            self.entries[id].take()
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Clone, Copy)]
struct MaterialIndexEntry {
    version: u64,
    index: u32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct MeshLodKey {
    mesh_id: usize,
    lod: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct MeshLodMaterialKey {
    mesh_id: usize,
    lod: usize,
    material_id: u32,
}

struct InstanceBatch<K, T> {
    key: K,
    instances: Vec<T>,
}

struct InstanceBatcher<K, T> {
    map: HashMap<K, usize>,
    batches: Vec<InstanceBatch<K, T>>,
    active: usize,
}

impl<K, T> InstanceBatcher<K, T>
where
    K: Copy + Eq + std::hash::Hash,
{
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            batches: Vec::new(),
            active: 0,
        }
    }

    fn reset(&mut self) {
        self.map.clear();
        self.active = 0;
    }

    fn push(&mut self, key: K, instance: T) {
        let idx = if let Some(existing) = self.map.get(&key) {
            *existing
        } else {
            let idx = self.active;
            if idx == self.batches.len() {
                self.batches.push(InstanceBatch {
                    key,
                    instances: Vec::new(),
                });
            } else {
                self.batches[idx].key = key;
                self.batches[idx].instances.clear();
            }
            self.map.insert(key, idx);
            self.active += 1;
            idx
        };
        self.batches[idx].instances.push(instance);
    }

    fn active_batches(&self) -> &[InstanceBatch<K, T>] {
        &self.batches[..self.active]
    }
}

#[derive(Clone, Copy)]
struct StreamingState {
    priority: f32,
    last_frame: u32,
    requested_lod: Option<usize>,
    force_low_res: bool,
}

struct ActiveGraph {
    spec_version: u64,
    surface_size: PhysicalSize<u32>,
    surface_format: wgpu::TextureFormat,
    swapchain_id: ResourceId,
    resource_ids: Vec<ResourceId>,
    hiz_id: Option<ResourceId>,
    graph: RenderGraph,
    compiled: RenderGraphCompilation,
    config_signature: RenderGraphConfigSignature,
}

struct RenderSceneState {
    data: RenderData,
    object_indices: HashMap<usize, usize>,
    light_indices: HashMap<usize, usize>,
    material_use_counts: Vec<u32>,
    material_active: Vec<bool>,
    active_materials: Vec<usize>,
    material_listed: Vec<bool>,
    material_usage_dirty: bool,
}

enum ObjectUpdateResult {
    Inserted {
        index: usize,
        material_id: usize,
    },
    Updated {
        index: usize,
        prev_mesh_id: usize,
        prev_material_id: usize,
        prev_lod_index: usize,
        prev_casts_shadow: bool,
        transform_changed: bool,
    },
    Unchanged,
}

struct GpuDrivenFrame {
    gbuffer_instances: Option<crate::graphics::passes::InstanceBuffer>,
    shadow_instances: Option<crate::graphics::passes::InstanceBuffer>,
    gbuffer_indirect: Option<wgpu::Buffer>,
    shadow_indirect: Option<wgpu::Buffer>,
    gbuffer_mesh_tasks: Option<wgpu::Buffer>,
    shadow_mesh_tasks: Option<wgpu::Buffer>,
    draws: Arc<Vec<IndirectDrawBatch>>,
}

impl RenderSceneState {
    fn transform_eq(a: &Transform, b: &Transform) -> bool {
        a.position == b.position && a.rotation == b.rotation && a.scale == b.scale
    }

    fn new(data: RenderData) -> Self {
        let mut object_indices = HashMap::new();
        object_indices.reserve(data.objects.len());
        for (idx, obj) in data.objects.iter().enumerate() {
            object_indices.insert(obj.id, idx);
        }
        let mut light_indices = HashMap::new();
        light_indices.reserve(data.lights.len());
        for (idx, light) in data.lights.iter().enumerate() {
            light_indices.insert(light.id, idx);
        }
        let mut state = Self {
            data,
            object_indices,
            light_indices,
            material_use_counts: Vec::new(),
            material_active: Vec::new(),
            active_materials: Vec::new(),
            material_listed: Vec::new(),
            material_usage_dirty: false,
        };
        state.rebuild_material_usage();
        state
    }

    fn begin_frame(&mut self) {
        self.data.previous_camera_transform = self.data.current_camera_transform;
    }

    fn clear_objects(&mut self) {
        self.data.objects.clear();
        self.object_indices.clear();
        self.material_use_counts.clear();
        self.material_active.clear();
        self.active_materials.clear();
        self.material_listed.clear();
        self.material_usage_dirty = true;
    }

    fn clear_lights(&mut self) {
        self.data.lights.clear();
        self.light_indices.clear();
    }

    fn upsert_object(&mut self, obj: RenderObjectDelta) -> ObjectUpdateResult {
        if let Some(&idx) = self.object_indices.get(&obj.id) {
            let entry = &mut self.data.objects[idx];
            let prev_mesh_id = entry.mesh_id;
            let prev_material_id = entry.material_id;
            let prev_lod_index = entry.lod_index;
            let prev_casts_shadow = entry.casts_shadow;
            let transform_changed = !Self::transform_eq(&entry.current_transform, &obj.transform);
            let mesh_changed = prev_mesh_id != obj.mesh_id;
            let material_changed = prev_material_id != obj.material_id;
            let lod_changed = prev_lod_index != obj.lod_index;
            let casts_shadow_changed = prev_casts_shadow != obj.casts_shadow;

            entry.previous_transform = entry.current_transform;
            entry.current_transform = obj.transform;
            if !(transform_changed
                || mesh_changed
                || material_changed
                || lod_changed
                || casts_shadow_changed)
            {
                return ObjectUpdateResult::Unchanged;
            }
            entry.mesh_id = obj.mesh_id;
            entry.material_id = obj.material_id;
            entry.casts_shadow = obj.casts_shadow;
            entry.lod_index = obj.lod_index;
            if material_changed {
                self.drop_material_usage(prev_material_id);
                self.bump_material_usage(obj.material_id);
            }
            ObjectUpdateResult::Updated {
                index: idx,
                prev_mesh_id,
                prev_material_id,
                prev_lod_index,
                prev_casts_shadow,
                transform_changed,
            }
        } else {
            let idx = self.data.objects.len();
            self.data.objects.push(RenderObject {
                id: obj.id,
                previous_transform: obj.transform,
                current_transform: obj.transform,
                mesh_id: obj.mesh_id,
                material_id: obj.material_id,
                casts_shadow: obj.casts_shadow,
                lod_index: obj.lod_index,
            });
            self.object_indices.insert(obj.id, idx);
            self.bump_material_usage(obj.material_id);
            ObjectUpdateResult::Inserted {
                index: idx,
                material_id: obj.material_id,
            }
        }
    }

    fn remove_object(&mut self, id: usize) -> Option<(RenderObject, Option<usize>)> {
        let idx = self.object_indices.remove(&id)?;
        let last_idx = self.data.objects.len().saturating_sub(1);
        let removed = self.data.objects.swap_remove(idx);
        let moved_idx = if idx < last_idx {
            let moved_id = self.data.objects[idx].id;
            self.object_indices.insert(moved_id, idx);
            Some(idx)
        } else {
            None
        };
        self.drop_material_usage(removed.material_id);
        Some((removed, moved_idx))
    }

    fn rebuild_material_usage(&mut self) {
        self.material_use_counts.clear();
        self.material_active.clear();
        self.active_materials.clear();
        self.material_listed.clear();
        let material_ids: Vec<usize> = self
            .data
            .objects
            .iter()
            .map(|obj| obj.material_id)
            .collect();
        for material_id in material_ids {
            self.bump_material_usage(material_id);
        }
        self.material_usage_dirty = true;
    }

    fn active_material_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.active_materials
            .iter()
            .copied()
            .filter(|id| self.material_active.get(*id).copied().unwrap_or(false))
    }

    fn take_material_usage_dirty(&mut self) -> bool {
        let dirty = self.material_usage_dirty;
        self.material_usage_dirty = false;
        dirty
    }

    fn ensure_material_capacity(&mut self, material_id: usize) {
        if material_id >= self.material_use_counts.len() {
            let new_len = material_id.saturating_add(1);
            self.material_use_counts.resize(new_len, 0);
            self.material_active.resize(new_len, false);
            self.material_listed.resize(new_len, false);
        }
    }

    fn bump_material_usage(&mut self, material_id: usize) {
        self.ensure_material_capacity(material_id);
        let count = &mut self.material_use_counts[material_id];
        if *count == 0 {
            self.material_active[material_id] = true;
            if !self.material_listed[material_id] {
                self.active_materials.push(material_id);
                self.material_listed[material_id] = true;
            }
            self.material_usage_dirty = true;
        }
        *count = count.saturating_add(1);
    }

    fn drop_material_usage(&mut self, material_id: usize) {
        if material_id >= self.material_use_counts.len() {
            return;
        }
        let count = &mut self.material_use_counts[material_id];
        if *count == 0 {
            return;
        }
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.material_active[material_id] = false;
            self.material_usage_dirty = true;
        }
    }

    fn upsert_light(&mut self, light: RenderLightDelta) -> bool {
        if let Some(&idx) = self.light_indices.get(&light.id) {
            let entry = &mut self.data.lights[idx];
            let transform_changed = !Self::transform_eq(&entry.current_transform, &light.transform);
            let color_changed = entry.color != light.color;
            let intensity_changed = entry.intensity != light.intensity;
            let type_changed = entry.light_type != light.light_type;
            let changed = transform_changed || color_changed || intensity_changed || type_changed;
            entry.previous_transform = entry.current_transform;
            entry.current_transform = light.transform;
            entry.color = light.color;
            entry.intensity = light.intensity;
            entry.light_type = light.light_type;
            changed
        } else {
            let idx = self.data.lights.len();
            self.data.lights.push(RenderLight {
                id: light.id,
                previous_transform: light.transform,
                current_transform: light.transform,
                color: light.color,
                intensity: light.intensity,
                light_type: light.light_type,
            });
            self.light_indices.insert(light.id, idx);
            true
        }
    }

    fn remove_light(&mut self, id: usize) {
        let Some(idx) = self.light_indices.remove(&id) else {
            return;
        };
        let last_idx = self.data.lights.len().saturating_sub(1);
        self.data.lights.swap_remove(idx);
        if idx < last_idx {
            let moved_id = self.data.lights[idx].id;
            self.light_indices.insert(moved_id, idx);
        }
    }
}

/// The render graph renderer
pub struct GraphRenderer {
    // WGPU Core
    instance: wgpu::Instance,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device_caps: Arc<RenderDeviceCaps>,

    backend: Box<dyn BindingBackend>,
    binding_backend_kind: BindingBackendKind,
    bindless_requires_uniform_indexing: bool,
    force_bindgroups_backend: bool,
    pool: GpuResourcePool,
    active_graph: Option<ActiveGraph>,
    shared_stats: Arc<RendererStats>,
    pass_overrides: HashMap<String, bool>,
    pass_timings: Vec<RenderPassTiming>,

    frame_inputs: FrameInputHub,
    mipmap_generator: MipmapGenerator,

    meshes: SlotVec<MeshGpu>,
    fallback_mesh: Option<MeshGpu>,
    materials: SlotVec<MaterialEntry>,
    material_index_map: SlotVec<MaterialIndexEntry>,
    material_index_order: Vec<usize>,
    mesh_lod_state: SlotVec<usize>,
    texture_low_res_state: SlotVec<bool>,
    // Materials that can't upload yet (usually waiting on textures); keeps dependency metadata alive for streaming.
    pending_materials: HashMap<usize, MaterialGpuData>,
    current_render_data: Option<RenderSceneState>,

    surface_size: PhysicalSize<u32>,
    prev_view_proj: Mat4,
    fallback_view: wgpu::TextureView,
    fallback_volume_view: wgpu::TextureView,
    default_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    scene_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,
    blue_noise_texture: wgpu::Texture,
    blue_noise_view: wgpu::TextureView,
    blue_noise_sampler: wgpu::Sampler,
    ibl_sampler: wgpu::Sampler,
    brdf_lut_sampler: wgpu::Sampler,
    brdf_lut_texture: wgpu::Texture,
    brdf_lut_view: wgpu::TextureView,
    irradiance_map_texture: wgpu::Texture,
    irradiance_map_view: wgpu::TextureView,
    prefiltered_env_map_texture: wgpu::Texture,
    prefiltered_env_map_view: wgpu::TextureView,
    atmosphere_precomputer: AtmospherePrecomputer,
    needs_atmosphere_precompute: bool,
    prev_sky_uniforms: SkyUniforms,
    prev_shader_constants: ShaderConstants,
    occlusion_bgl: wgpu::BindGroupLayout,
    occlusion_pipeline: wgpu::ComputePipeline,
    occlusion_stable_frames: u32,
    gpu_cull_bgl: wgpu::BindGroupLayout,
    gpu_cull_clear_pipeline: wgpu::ComputePipeline,
    gpu_cull_pipeline: wgpu::ComputePipeline,
    gpu_mesh_tasks_pipeline: wgpu::ComputePipeline,
    shadow_format: wgpu::TextureFormat,
    frames_in_flight: usize,
    buffer_cache: BufferCache,
    gpu_instance_buffer: Option<wgpu::Buffer>,
    gpu_instance_capacity: usize,
    gpu_visible_buffer: Option<wgpu::Buffer>,
    gpu_visible_capacity: usize,
    gpu_total_capacity: usize,
    gpu_shadow_buffer: Option<wgpu::Buffer>,
    gpu_shadow_capacity: usize,
    gpu_indirect_gbuffer: Option<wgpu::Buffer>,
    gpu_indirect_shadow: Option<wgpu::Buffer>,
    gpu_mesh_tasks_gbuffer: Option<wgpu::Buffer>,
    gpu_mesh_tasks_shadow: Option<wgpu::Buffer>,
    gpu_mesh_tasks_capacity: usize,
    gpu_mesh_task_meta_buffer: Option<wgpu::Buffer>,
    gpu_mesh_task_meta_capacity: usize,
    gpu_mesh_meta_buffer: Option<wgpu::Buffer>,
    gpu_mesh_meta_len: usize,
    gpu_mesh_meta_capacity: usize,
    gpu_mesh_indices: Vec<u32>,
    gpu_mesh_material_indices: HashMap<(usize, u32), u32>,
    gpu_indirect_capacity: usize,
    gpu_draws: Arc<Vec<IndirectDrawBatch>>,
    gpu_draw_count: u32,
    gpu_instances_dirty: bool,
    gpu_draws_dirty: bool,
    gpu_instance_updates: Vec<usize>,
    gpu_driven_active: bool,
    supports_indirect_first_instance: bool,
    warned_indirect_first_instance_missing: bool,
    warned_multi_draw_missing: bool,
    warned_mesh_shaders_missing: bool,
    gpu_cull_dirty: bool,
    gpu_cull_last_frame: u32,
    last_gpu_cull_signature: Option<GpuCullSignature>,

    frame_index: u32,
    asset_stream_sender: Sender<AssetStreamingRequest>,
    streaming_inflight: SlotVec<StreamingState>,
    recently_evicted: SlotVec<u32>,
    gbuffer_batcher:
        InstanceBatcher<MeshLodMaterialKey, crate::graphics::passes::gbuffer::GBufferInstanceRaw>,
    shadow_batcher: InstanceBatcher<MeshLodKey, InstanceRaw>,
    streaming_tuning: StreamingTuning,
    streaming_pressure: MemoryPressure,
    streaming_pressure_frame: u32,
    streaming_requests: Option<Vec<AssetStreamingRequest>>,
    streaming_dirty: bool,
    streaming_last_frame: u32,
    streaming_mesh_scratch: StreamRequestScratch,
    streaming_material_scratch: StreamRequestScratch,
    streaming_texture_scratch: StreamRequestScratch,
    streaming_mesh_requests: Vec<StreamRequest>,
    streaming_material_requests: Vec<StreamRequest>,
    streaming_texture_requests: Vec<StreamRequest>,
    streaming_request_cursor: usize,
    streaming_scan_cursor: usize,
    egui_texture_cache: EguiTextureCache,
    prev_idle_frames_before_evict: Option<u32>,
    pending_render_delta: Option<RenderDelta>,
    material_version: u64,
    material_bindings_version: u64,
    materials_dirty: bool,
    instances_dirty: bool,
    shadow_instances_dirty: bool,
    shadow_uniforms_dirty: bool,
    shadow_bounds_dirty: bool,
    shadow_bounds: Option<Aabb>,
    gbuffer_draws_dirty: bool,
    shadow_draws_dirty: bool,
    gbuffer_draws_version: u64,
    shadow_draws_version: u64,
    gpu_bundle_version: u64,
    bundle_resource_epoch: u64,
    bundle_resource_change_frame: u32,
    shadow_matrices_version: u64,
    bundle_invalidate_pending: bool,
    cached_material_buffer: Option<wgpu::Buffer>,
    cached_texture_views: Vec<wgpu::TextureView>,
    cached_texture_array_size: u32,
    cached_material_signature: u64,
    cached_texture_overflow: bool,
    cached_material_textures: Option<Arc<Vec<MaterialTextureSet>>>,
    cached_material_textures_signature: u64,
    cached_gbuffer_instances: Option<crate::graphics::passes::InstanceBuffer>,
    cached_gbuffer_batches: Arc<Vec<crate::graphics::passes::DrawBatch>>,
    cached_shadow_instances: Option<crate::graphics::passes::InstanceBuffer>,
    cached_shadow_batches: Arc<Vec<crate::graphics::passes::DrawBatch>>,
    cached_shadow_uniforms_buffer: Option<wgpu::Buffer>,
    cached_shadow_matrices_buffer: Option<wgpu::Buffer>,
    last_instance_pressure: MemoryPressure,
}

pub(crate) struct RendererSnapshot {
    pool: GpuResourcePool,
    current_render_data: Option<RenderSceneState>,
    pending_render_delta: Option<RenderDelta>,
    meshes: SlotVec<MeshGpu>,
    materials: SlotVec<MaterialEntry>,
    material_index_map: SlotVec<MaterialIndexEntry>,
    material_index_order: Vec<usize>,
    material_version: u64,
    material_bindings_version: u64,
    pending_materials: HashMap<usize, MaterialGpuData>,
    mesh_lod_state: SlotVec<usize>,
    texture_low_res_state: SlotVec<bool>,
    pass_overrides: HashMap<String, bool>,
    egui_texture_cache: EguiTextureCache,
    streaming_tuning: StreamingTuning,
    streaming_requests: Option<Vec<AssetStreamingRequest>>,
}

struct MaterialBuildResult {
    buffer: Option<wgpu::Buffer>,
    views: Vec<wgpu::TextureView>,
    size: u32,
    signature: u64,
    changed: bool,
    texture_overflow: bool,
}

struct MaterialTextureBuildResult {
    textures: Arc<Vec<MaterialTextureSet>>,
    signature: u64,
    changed: bool,
}

impl GraphRenderer {
    pub async fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        size: PhysicalSize<u32>,
        _target_tickrate: f32,
        asset_stream_sender: Sender<AssetStreamingRequest>,
        shared_stats: Arc<RendererStats>,
        allow_experimental_features: bool,
        binding_backend_choice: BindingBackendChoice,
    ) -> Result<Self, RendererError> {
        let supported_features = adapter.features();
        let adapter_info = adapter.get_info();
        let modern_bindless_features = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::BUFFER_BINDING_ARRAY
            | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY;
        let mut binding_backend_choice = match binding_backend_choice {
            BindingBackendChoice::BindlessModern => {
                if supported_features.contains(modern_bindless_features) {
                    BindingBackendChoice::BindlessModern
                } else if supported_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY) {
                    BindingBackendChoice::BindlessFallback
                } else {
                    BindingBackendChoice::BindGroups
                }
            }
            BindingBackendChoice::BindlessFallback => {
                if supported_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY) {
                    BindingBackendChoice::BindlessFallback
                } else {
                    BindingBackendChoice::BindGroups
                }
            }
            BindingBackendChoice::BindGroups => BindingBackendChoice::BindGroups,
            BindingBackendChoice::Auto => {
                if supported_features.contains(modern_bindless_features) {
                    BindingBackendChoice::BindlessModern
                } else if supported_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY) {
                    BindingBackendChoice::BindlessFallback
                } else {
                    BindingBackendChoice::BindGroups
                }
            }
        };

        let needs_texture_arrays = matches!(
            binding_backend_choice,
            BindingBackendChoice::BindlessModern | BindingBackendChoice::BindlessFallback
        );
        if needs_texture_arrays
            && !supported_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
        {
            return Err(RendererError::ResourceCreation(
                "Device does not support texture arrays (required)".to_string(),
            ));
        }
        if needs_texture_arrays
            && !supported_features.contains(
                wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            )
        {
            tracing::warn!(
                "Device missing non-uniform indexing support; bindless backend will force per-material batching"
            );
        }
        let mut required_features = wgpu::Features::empty();
        if needs_texture_arrays {
            required_features |= wgpu::Features::TEXTURE_BINDING_ARRAY;
            if supported_features.contains(
                wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            ) {
                required_features |=
                    wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
            }
        }
        if supported_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
            required_features |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
        }
        if supported_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            required_features |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        }
        if supported_features.contains(wgpu::Features::FLOAT32_FILTERABLE) {
            required_features |= wgpu::Features::FLOAT32_FILTERABLE;
        }
        if matches!(binding_backend_choice, BindingBackendChoice::BindlessModern)
            && supported_features.contains(modern_bindless_features)
        {
            required_features |= modern_bindless_features;
        }

        let mesh_shaders_allowed =
            allow_experimental_features && adapter_info.backend == wgpu::Backend::Vulkan;
        let experimental_required = if mesh_shaders_allowed {
            let mut requested = wgpu::Features::empty();
            let mesh_supported =
                supported_features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER);
            if mesh_supported {
                requested |= wgpu::Features::EXPERIMENTAL_MESH_SHADER;
            }
            requested
        } else {
            wgpu::Features::empty()
        };
        required_features |= experimental_required;
        let experimental_features = if !experimental_required.is_empty() {
            // Safety: enabling experimental features is an explicit user opt-in.
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Render Device"),
                required_features,
                required_limits: adapter.limits(),
                experimental_features,
                ..Default::default()
            })
            .await
            .map_err(|e| {
                RendererError::ResourceCreation(format!("Failed to create device: {}", e))
            })?;

        let device = Arc::new(device);
        let device_caps = Arc::new(RenderDeviceCaps {
            adapter_info,
            features: device.features(),
            limits: device.limits(),
            downlevel_caps: adapter.get_downlevel_capabilities(),
        });
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

        let device_features = device.features();
        let supports_indirect_first_instance =
            device_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        let bindless_config =
            BindlessConfig::default().clamp_to_device(device_features, &device.limits());

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/DefaultSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/ShadowSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: None,
            ..Default::default()
        });

        let scene_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/SceneSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/PointSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let ibl_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/IBLSampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let brdf_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/BRDFLUTSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let make_solid_tex = |color: [u8; 4], device: &wgpu::Device, queue: &wgpu::Queue| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Renderer/Fallback2D"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &color,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            texture.create_view(&Default::default())
        };

        let fallback_view = make_solid_tex([0, 0, 0, 0], &device, &queue);
        let fallback_volume_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/Fallback3D"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_volume_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0, 0, 0, 0],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fallback_volume_view = fallback_volume_texture.create_view(&Default::default());

        // Deterministic blue noise texture for stochastic passes (SSGI, etc.).
        let blue_noise_extent = wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 1,
        };
        let mut blue_noise_data =
            vec![0u8; (blue_noise_extent.width * blue_noise_extent.height * 4) as usize];
        let mut seed = 1u32;
        for chunk in blue_noise_data.chunks_exact_mut(4) {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            chunk[0] = (seed >> 24) as u8;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            chunk[1] = (seed >> 16) as u8;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            chunk[2] = (seed >> 8) as u8;
            chunk[3] = 255;
        }
        let blue_noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/BlueNoise"),
            size: blue_noise_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &blue_noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &blue_noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * blue_noise_extent.width),
                rows_per_image: Some(blue_noise_extent.height),
            },
            blue_noise_extent,
        );
        let blue_noise_view = blue_noise_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });
        let blue_noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/BlueNoiseSampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let brdf_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/BRDFLUT"),
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
        let brdf_lut_view = brdf_lut_texture.create_view(&Default::default());

        let irradiance_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/IrradianceMap"),
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
        let irradiance_map_view =
            irradiance_map_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });

        let prefiltered_env_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/PrefilteredEnvMap"),
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
        let prefiltered_env_map_view =
            prefiltered_env_map_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });

        let atmosphere_precomputer = AtmospherePrecomputer::new(&device);

        let occlusion_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Occlusion/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let occlusion_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Occlusion/PipelineLayout"),
            bind_group_layouts: &[&occlusion_bgl],
            immediate_size: 0,
        });

        let occlusion_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/hiz_cull.wgsl"));
        let occlusion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Occlusion/Pipeline"),
            layout: Some(&occlusion_layout),
            module: &occlusion_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gpu_cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuCull/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<CameraUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GpuCullParams>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let gpu_cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GpuCull/PipelineLayout"),
            bind_group_layouts: &[&gpu_cull_bgl],
            immediate_size: 0,
        });
        let gpu_cull_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/gpu_cull.wgsl"));
        let gpu_cull_clear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuCull/ClearPipeline"),
                layout: Some(&gpu_cull_layout),
                module: &gpu_cull_shader,
                entry_point: Some("clear_draws"),
                compilation_options: Default::default(),
                cache: None,
            });
        let gpu_cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GpuCull/Pipeline"),
            layout: Some(&gpu_cull_layout),
            module: &gpu_cull_shader,
            entry_point: Some("cull_instances"),
            compilation_options: Default::default(),
            cache: None,
        });
        let gpu_mesh_tasks_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuCull/MeshTasksPipeline"),
                layout: Some(&gpu_cull_layout),
                module: &gpu_cull_shader,
                entry_point: Some("build_mesh_tasks"),
                compilation_options: Default::default(),
                cache: None,
            });

        let supports_float32_filterable = device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        let shadow_format = if supports_float32_filterable {
            wgpu::TextureFormat::Rg32Float
        } else {
            wgpu::TextureFormat::Rg16Float
        };

        let binding_backend_choice = match binding_backend_choice {
            BindingBackendChoice::BindlessModern
                if !device_features.contains(modern_bindless_features) =>
            {
                warn!(
                    "Bindless modern backend requested but not supported; falling back to bindless fallback"
                );
                BindingBackendChoice::BindlessFallback
            }
            BindingBackendChoice::BindlessFallback
                if !device_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY) =>
            {
                warn!(
                    "Bindless fallback backend requested but texture arrays unsupported; falling back to bind groups"
                );
                BindingBackendChoice::BindGroups
            }
            other => other,
        };

        let (backend, binding_backend_kind, backend_name): (
            Box<dyn BindingBackend>,
            BindingBackendKind,
            &str,
        ) = match binding_backend_choice {
            BindingBackendChoice::BindGroups => (
                Box::new(BindGroupBackend::new(bindless_config.clone())),
                BindingBackendKind::BindGroups,
                "bind-groups",
            ),
            BindingBackendChoice::BindlessFallback => (
                Box::new(BindlessFallbackBackend::new(bindless_config.clone())),
                BindingBackendKind::BindlessFallback,
                "bindless-fallback",
            ),
            BindingBackendChoice::BindlessModern => (
                Box::new(BindlessModernBackend::new(bindless_config.clone())),
                BindingBackendKind::BindlessModern,
                "bindless-modern",
            ),
            BindingBackendChoice::Auto => {
                if device_features.contains(modern_bindless_features) {
                    (
                        Box::new(BindlessModernBackend::new(bindless_config.clone())),
                        BindingBackendKind::BindlessModern,
                        "bindless-modern",
                    )
                } else if device_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY) {
                    (
                        Box::new(BindlessFallbackBackend::new(bindless_config.clone())),
                        BindingBackendKind::BindlessFallback,
                        "bindless-fallback",
                    )
                } else {
                    (
                        Box::new(BindGroupBackend::new(bindless_config.clone())),
                        BindingBackendKind::BindGroups,
                        "bind-groups",
                    )
                }
            }
        };
        let bindless_requires_uniform_indexing = matches!(
            binding_backend_kind,
            BindingBackendKind::BindlessModern | BindingBackendKind::BindlessFallback
        ) && !device_features.contains(
            wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        );
        info!("Selected binding backend: {}", backend_name);

        let mipmap_generator = MipmapGenerator::new(&device);

        // Budget defaults: tuned for low-memory GPUs
        let mb = 1024 * 1024;
        let soft_mb = env::var("HELMER_GPU_BUDGET_SOFT_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(512)
            .max(64);
        let hard_mb = env::var("HELMER_GPU_BUDGET_HARD_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(896)
            .max(soft_mb);

        let streaming_tuning = StreamingTuning::default();
        let frames_in_flight = RenderConfig::default().frames_in_flight.max(1) as usize;
        let pool_config = streaming_tuning.pool_config();
        let pool = GpuResourcePool::new(soft_mb * mb, hard_mb * mb, pool_config);
        let frame_inputs = FrameInputHub::new();
        let mut renderer = Self {
            instance,
            device,
            queue,
            surface,
            surface_config,
            device_caps,

            backend,
            binding_backend_kind,
            bindless_requires_uniform_indexing,
            force_bindgroups_backend: false,
            pool,
            active_graph: None,
            shared_stats,
            pass_overrides: HashMap::new(),
            pass_timings: Vec::new(),

            frame_inputs,
            mipmap_generator,

            meshes: SlotVec::new(),
            fallback_mesh: None,
            materials: SlotVec::new(),
            material_index_map: SlotVec::new(),
            material_index_order: Vec::new(),
            mesh_lod_state: SlotVec::new(),
            texture_low_res_state: SlotVec::new(),
            pending_materials: HashMap::new(),
            current_render_data: None,

            surface_size: size,
            prev_view_proj: Mat4::IDENTITY,
            fallback_view,
            fallback_volume_view,
            default_sampler,
            shadow_sampler,
            scene_sampler,
            point_sampler,
            blue_noise_texture,
            blue_noise_view,
            blue_noise_sampler,
            ibl_sampler,
            brdf_lut_sampler,
            brdf_lut_texture,
            brdf_lut_view,
            irradiance_map_texture,
            irradiance_map_view,
            prefiltered_env_map_texture,
            prefiltered_env_map_view,
            atmosphere_precomputer,
            needs_atmosphere_precompute: true,
            prev_sky_uniforms: SkyUniforms::default(),
            prev_shader_constants: ShaderConstants::default(),
            occlusion_bgl,
            occlusion_pipeline,
            occlusion_stable_frames: 0,
            gpu_cull_bgl,
            gpu_cull_clear_pipeline,
            gpu_cull_pipeline,
            gpu_mesh_tasks_pipeline,
            shadow_format,
            frames_in_flight,
            buffer_cache: BufferCache::new(frames_in_flight),
            gpu_instance_buffer: None,
            gpu_instance_capacity: 0,
            gpu_visible_buffer: None,
            gpu_visible_capacity: 0,
            gpu_total_capacity: 0,
            gpu_shadow_buffer: None,
            gpu_shadow_capacity: 0,
            gpu_indirect_gbuffer: None,
            gpu_indirect_shadow: None,
            gpu_mesh_tasks_gbuffer: None,
            gpu_mesh_tasks_shadow: None,
            gpu_mesh_tasks_capacity: 0,
            gpu_mesh_task_meta_buffer: None,
            gpu_mesh_task_meta_capacity: 0,
            gpu_mesh_meta_buffer: None,
            gpu_mesh_meta_len: 0,
            gpu_mesh_meta_capacity: 0,
            gpu_mesh_indices: Vec::new(),
            gpu_mesh_material_indices: HashMap::new(),
            gpu_indirect_capacity: 0,
            gpu_draws: Arc::new(Vec::new()),
            gpu_draw_count: 0,
            gpu_instances_dirty: true,
            gpu_draws_dirty: true,
            gpu_instance_updates: Vec::new(),
            gpu_driven_active: false,
            supports_indirect_first_instance,
            warned_indirect_first_instance_missing: false,
            warned_multi_draw_missing: false,
            warned_mesh_shaders_missing: false,
            gpu_cull_dirty: true,
            gpu_cull_last_frame: u32::MAX,
            last_gpu_cull_signature: None,

            frame_index: 0,
            asset_stream_sender,
            streaming_inflight: SlotVec::new(),
            recently_evicted: SlotVec::new(),
            gbuffer_batcher: InstanceBatcher::new(),
            shadow_batcher: InstanceBatcher::new(),
            streaming_tuning,
            streaming_pressure: MemoryPressure::None,
            streaming_pressure_frame: 0,
            streaming_requests: None,
            streaming_dirty: false,
            streaming_last_frame: 0,
            streaming_mesh_scratch: StreamRequestScratch::new(),
            streaming_material_scratch: StreamRequestScratch::new(),
            streaming_texture_scratch: StreamRequestScratch::new(),
            streaming_mesh_requests: Vec::new(),
            streaming_material_requests: Vec::new(),
            streaming_texture_requests: Vec::new(),
            streaming_request_cursor: 0,
            streaming_scan_cursor: 0,
            egui_texture_cache: EguiTextureCache::default(),
            prev_idle_frames_before_evict: None,
            pending_render_delta: None,
            material_version: 0,
            material_bindings_version: 0,
            materials_dirty: true,
            instances_dirty: true,
            shadow_instances_dirty: true,
            shadow_uniforms_dirty: true,
            shadow_bounds_dirty: true,
            shadow_bounds: None,
            gbuffer_draws_dirty: true,
            shadow_draws_dirty: true,
            gbuffer_draws_version: 0,
            shadow_draws_version: 0,
            gpu_bundle_version: 0,
            bundle_resource_epoch: 0,
            bundle_resource_change_frame: 0,
            shadow_matrices_version: 0,
            bundle_invalidate_pending: false,
            cached_material_buffer: None,
            cached_texture_views: Vec::new(),
            cached_texture_array_size: 0,
            cached_material_signature: 0,
            cached_texture_overflow: false,
            cached_material_textures: None,
            cached_material_textures_signature: 0,
            cached_gbuffer_instances: None,
            cached_gbuffer_batches: Arc::new(Vec::new()),
            cached_shadow_instances: None,
            cached_shadow_batches: Arc::new(Vec::new()),
            cached_shadow_uniforms_buffer: None,
            cached_shadow_matrices_buffer: None,
            last_instance_pressure: MemoryPressure::None,
        };

        renderer.init_fallback_mesh();
        renderer.update_stats();
        info!("initialized renderer");
        Ok(renderer)
    }

    pub fn render(&mut self) -> Result<(), RendererError> {
        let profiling_enabled = self
            .shared_stats
            .profiling_enabled
            .load(std::sync::atomic::Ordering::Relaxed);
        let acquire_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let output_frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost) => {
                //self.resize(self.window_size);
                return Ok(());
            }
            Err(e) => return Err(RendererError::ResourceCreation(e.to_string())),
        };
        if let Some(start) = acquire_start {
            self.shared_stats.render_acquire_us.store(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let output_view = output_frame.texture.create_view(&Default::default());

        self.begin_frame();
        self.apply_pending_render_delta();
        self.drain_pool_evictions();
        self.drain_pool_binding_changes();

        let mut scene = if let Some(scene) = self.current_render_data.take() {
            scene
        } else {
            self.clear_active_graph();
            self.clear_and_present(&output_view, output_frame);
            self.frame_index = self.frame_index.wrapping_add(1);
            return Ok(());
        };

        let graph_spec = scene.data.render_graph.clone();
        let graph_sig = RenderGraphConfigSignature::from_render_config(&scene.data.render_config);

        if let Err(err) = self.ensure_graph_ready(&graph_spec, graph_sig) {
            self.current_render_data = Some(scene);
            return Err(err);
        }

        let swapchain_id = match self.active_graph.as_ref() {
            Some(active) => active.swapchain_id,
            None => {
                warn!("Render graph missing; skipping frame");
                self.current_render_data = Some(scene);
                self.clear_and_present(&output_view, output_frame);
                self.frame_index = self.frame_index.wrapping_add(1);
                return Ok(());
            }
        };

        self.update_swapchain_entry(swapchain_id, output_view.clone());
        let streaming_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        self.refresh_streaming_plan(&scene.data);
        if let Some(start) = streaming_start {
            self.shared_stats.render_streaming_plan_us.store(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let globals_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let globals = if let Some(globals) = self.prepare_frame_globals(&scene) {
            globals
        } else {
            self.current_render_data = Some(scene);
            self.clear_and_present(&output_view, output_frame);
            self.frame_index = self.frame_index.wrapping_add(1);
            return Ok(());
        };
        if let Some(start) = globals_start {
            self.shared_stats.render_prepare_globals_us.store(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let occlusion_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        if globals.render_config.gpu_driven {
            self.run_gpu_culling(&globals);
        } else {
            self.run_occlusion_culling(&globals);
        }
        if let Some(start) = occlusion_start {
            self.shared_stats.render_occlusion_us.store(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        self.run_atmosphere_precompute(&globals);
        self.frame_inputs.set(globals);

        self.frame_inputs.set(SwapchainFrameInput {
            view: output_view,
            format: self.surface_config.format,
        });

        self.backend
            .begin_frame(&self.device, &self.queue, &self.pool, self.frame_index);

        let pressure = self.update_streaming_pressure();
        let (graph, compiled_graph) = match self.active_graph.as_ref() {
            Some(active) => (&active.graph, &active.compiled),
            None => {
                warn!("Render graph missing compilation; skipping frame");
                self.frame_index = self.frame_index.wrapping_add(1);
                return Ok(());
            }
        };

        if let Some(prev_idle_frames_before_evict) = self.prev_idle_frames_before_evict.take() {
            self.pool.idle_frames_before_evict = prev_idle_frames_before_evict;
        }
        if pressure == MemoryPressure::Hard
            && self.pool.idle_frames_before_evict
                > self.streaming_tuning.hard_idle_frames_before_evict
        {
            self.prev_idle_frames_before_evict = Some(self.pool.idle_frames_before_evict);
            self.pool.idle_frames_before_evict =
                self.streaming_tuning.hard_idle_frames_before_evict;
        }

        let mut graph_exec_stats = RenderGraphExecutionStats::default();
        let pass_timings = if profiling_enabled {
            Some(&mut self.pass_timings)
        } else {
            None
        };
        let graph_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let cmds = RenderGraphExecutor::execute(
            graph,
            compiled_graph,
            &self.device,
            &self.queue,
            &mut self.pool,
            self.backend.as_ref(),
            self.frame_index,
            &self.frame_inputs,
            self.streaming_tuning.graph_encoder_batch_size,
            Some(&self.pass_overrides),
            if profiling_enabled {
                Some(&mut graph_exec_stats)
            } else {
                None
            },
            pass_timings,
        );
        if let Some(start) = graph_start {
            let graph_total_us = start.elapsed().as_micros() as u64;
            self.shared_stats
                .render_graph_us
                .store(graph_total_us, std::sync::atomic::Ordering::Relaxed);
            let accounted_us = graph_exec_stats
                .pass_execute_us
                .saturating_add(graph_exec_stats.encoder_create_us)
                .saturating_add(graph_exec_stats.encoder_finish_us);
            self.shared_stats.render_graph_pass_us.store(
                graph_exec_stats.pass_execute_us,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.shared_stats.render_graph_encoder_create_us.store(
                graph_exec_stats.encoder_create_us,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.shared_stats.render_graph_encoder_finish_us.store(
                graph_exec_stats.encoder_finish_us,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.shared_stats.render_graph_overhead_us.store(
                graph_total_us.saturating_sub(accounted_us),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        if profiling_enabled {
            *self.shared_stats.pass_timings.write() = self.pass_timings.clone();
        }

        let resource_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let evicted = self.pool.tick_eviction(self.frame_index);
        if !evicted.is_empty() {
            self.bundle_invalidate_pending = true;
        }
        if matches!(pressure, MemoryPressure::Hard) {
            self.pool.compact_transients(true);
        } else if matches!(pressure, MemoryPressure::Soft) {
            self.pool.compact_transients(false);
        }
        self.track_evicted_resources(evicted);
        self.update_stats();
        if let Some(start) = resource_start {
            self.shared_stats.render_resource_mgmt_us.store(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        // --- Submit and Present ---
        let submit_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        self.queue.submit(cmds);
        if let Some(start) = submit_start {
            self.shared_stats.render_submit_us.store(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let present_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        output_frame.present();
        if let Some(start) = present_start {
            self.shared_stats.render_present_us.store(
                start.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        self.frame_index = self.frame_index.wrapping_add(1);
        self.current_render_data = Some(scene);
        Ok(())
    }

    fn update_stats(&self) {
        let vram = self.pool.vram_budget();
        self.shared_stats.vram_used_bytes.store(
            vram.global.current_bytes,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared_stats.vram_soft_limit_bytes.store(
            vram.global.soft_limit_bytes,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared_stats.vram_hard_limit_bytes.store(
            vram.global.hard_limit_bytes,
            std::sync::atomic::Ordering::Relaxed,
        );
        for (idx, class) in vram.per_kind.iter().enumerate() {
            self.shared_stats.vram_soft_limit_per_kind[idx]
                .store(class.soft_limit_bytes, std::sync::atomic::Ordering::Relaxed);
            self.shared_stats.vram_hard_limit_per_kind[idx]
                .store(class.hard_limit_bytes, std::sync::atomic::Ordering::Relaxed);
        }
        let resident = self.pool.resident_count();
        self.shared_stats
            .resident_resources
            .store(resident, std::sync::atomic::Ordering::Relaxed);
        self.shared_stats.idle_frames.store(
            self.pool.idle_frames_before_evict,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    fn sync_pass_overrides(&mut self) {
        let Some(active) = self.active_graph.as_ref() else {
            return;
        };

        let mut valid = HashSet::new();
        self.pass_timings.clear();
        for (order, node_id) in active.compiled.sorted_nodes.iter().enumerate() {
            let name = active
                .compiled
                .pass_names
                .get(*node_id)
                .copied()
                .unwrap_or("unknown");
            valid.insert(name.to_string());
            let enabled = self.pass_overrides.get(name).copied().unwrap_or(true);
            self.pass_timings.push(RenderPassTiming {
                name: name.to_string(),
                order,
                enabled,
                duration_us: 0,
            });
        }
        self.pass_overrides.retain(|name, _| valid.contains(name));
        *self.shared_stats.pass_timings.write() = self.pass_timings.clone();
    }

    fn apply_control(&mut self, control: RenderControl) {
        match control {
            RenderControl::SetGpuBudget {
                soft_limit_bytes,
                hard_limit_bytes,
                idle_frames,
                per_kind_soft,
                per_kind_hard,
            } => {
                self.pool
                    .set_budget(soft_limit_bytes, hard_limit_bytes.max(soft_limit_bytes));
                if let (Some(soft), Some(hard)) =
                    (per_kind_soft.as_deref(), per_kind_hard.as_deref())
                {
                    self.pool.set_per_kind_budgets(soft, hard);
                }
                if let Some(idle) = idle_frames {
                    self.pool.idle_frames_before_evict = idle;
                }
                let evicted = self.pool.tick_eviction(self.frame_index);
                if !evicted.is_empty() {
                    self.bundle_invalidate_pending = true;
                }
                self.track_evicted_resources(evicted);
                self.update_stats();
            }
            RenderControl::SetStreamingTuning(tuning) => {
                self.streaming_tuning = tuning;
                self.pool.apply_config(self.streaming_tuning.pool_config());
                self.streaming_dirty = true;
                self.streaming_last_frame = 0;
                self.instances_dirty = true;
                self.shadow_instances_dirty = true;
                self.shadow_uniforms_dirty = true;
                self.shadow_bounds_dirty = true;
                self.gbuffer_draws_dirty = true;
                self.shadow_draws_dirty = true;
            }
            RenderControl::EvictAll { restream_assets } => {
                if let Some(active) = self.active_graph.take() {
                    self.evict_graph_resources(&active);
                    if !restream_assets {
                        self.active_graph = Some(active);
                    }
                }
                self.pool.evict_all();
                self.bump_egui_epoch();
                self.streaming_inflight = SlotVec::new();
                self.recently_evicted = SlotVec::new();
                self.streaming_pressure = MemoryPressure::None;
                self.streaming_pressure_frame = self.frame_index;
                self.materials_dirty = true;
                self.instances_dirty = true;
                self.shadow_instances_dirty = true;
                self.shadow_uniforms_dirty = true;
                self.shadow_bounds_dirty = true;
                self.gbuffer_draws_dirty = true;
                self.shadow_draws_dirty = true;
                self.note_bundle_resource_change();
                if restream_assets {
                    if let Some(state) = self.current_render_data.take() {
                        self.streaming_dirty = true;
                        self.streaming_last_frame = 0;
                        self.refresh_streaming_plan(&state.data);
                        self.current_render_data = Some(state);
                    }
                }
                self.update_stats();
            }
            RenderControl::SetPassEnabled { pass, enabled } => {
                self.pass_overrides.insert(pass, enabled);
                self.sync_pass_overrides();
            }
            RenderControl::RecreateDevice { .. } => {
                // Handled by the render thread loop before dispatching here.
            }
        }
    }

    fn clear_and_present(
        &mut self,
        output_view: &wgpu::TextureView,
        output_frame: wgpu::SurfaceTexture,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GraphRenderer/ClearOnly"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GraphRenderer/ClearSwapchain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output_frame.present();
    }

    pub fn process_message(&mut self, message: RenderMessage) {
        match message {
            RenderMessage::CreateMesh {
                id,
                vertices,
                lod_indices,
                meshlets,
                bounds,
            } => {
                if let Err(err) =
                    self.upload_mesh(id, vertices.as_ref(), &lod_indices, &meshlets, bounds)
                {
                    warn!("Failed to upload mesh {id}: {err:?}");
                }
            }
            RenderMessage::CreateTexture {
                id,
                name: _,
                kind: _,
                data,
                format,
                width,
                height,
            } => {
                if let Err(err) = self.upload_texture(id, data.as_ref(), format, width, height) {
                    warn!("Failed to upload texture {id}: {err:?}");
                }
                self.resolve_pending_materials();
            }
            RenderMessage::CreateMaterial(mat) => {
                if self.try_upload_material(mat.clone()) {
                    self.pending_materials.remove(&mat.id);
                } else {
                    self.pending_materials.insert(mat.id, mat);
                }
            }
            RenderMessage::RenderData(data) => {
                self.ingest_render_data(data);
            }
            RenderMessage::RenderDelta(delta) => {
                self.queue_render_delta(delta);
            }
            RenderMessage::EguiData(data) => {
                // Keep a CPU-side mirror of the atlas so we can rebuild GPU textures after resizes/evictions.
                apply_egui_delta(&mut self.egui_texture_cache, &data.textures_delta);
                self.frame_inputs.set(self.egui_texture_cache.clone());
                self.frame_inputs.set(data);
            }
            RenderMessage::Control(ctrl) => {
                self.apply_control(ctrl);
            }
            RenderMessage::Resize(size) => self.handle_resize(size),
            RenderMessage::WindowRecreated { .. } => {}
            RenderMessage::Shutdown => {}
        }
        self.drain_pool_evictions();
        self.drain_pool_binding_changes();
    }

    pub fn poll_device(&self, poll_type: wgpu::PollType) {
        let _ = self.device.poll(poll_type);
    }
}

impl GraphRenderer {
    fn begin_frame(&mut self) {
        if let Some(state) = self.current_render_data.as_mut() {
            state.begin_frame();
        }
    }

    fn queue_render_delta(&mut self, delta: RenderDelta) {
        if delta.is_empty() {
            return;
        }
        if let Some(pending) = self.pending_render_delta.as_mut() {
            pending.merge_from(delta);
        } else {
            self.pending_render_delta = Some(delta);
        }
    }

    fn apply_pending_render_delta(&mut self) {
        let Some(delta) = self.pending_render_delta.take() else {
            return;
        };
        self.apply_render_delta(delta);
    }

    fn apply_render_delta(&mut self, delta: RenderDelta) {
        let RenderDelta {
            full,
            objects_upsert,
            objects_remove,
            lights_upsert,
            lights_remove,
            camera,
            render_config,
            render_graph,
            streaming_requests,
        } = delta;
        let mut objects_changed = full;
        let mut lights_changed = full;
        let camera_changed = camera.is_some();
        let config_changed = render_config.is_some();
        let mut materials_needed = false;
        let material_version = self.material_version;

        if self.current_render_data.is_none() {
            let config = render_config.unwrap_or_else(RenderConfig::default);
            let graph = render_graph.clone().unwrap_or_else(default_graph_spec);
            let (camera_component, camera_transform) = match camera {
                Some(cam) => (cam.camera, cam.transform),
                None => (Camera::default(), Transform::default()),
            };
            let data = RenderData {
                objects: Vec::new(),
                lights: Vec::new(),
                previous_camera_transform: camera_transform,
                current_camera_transform: camera_transform,
                camera_component,
                timestamp: Instant::now(),
                render_config: config,
                render_graph: graph,
            };
            self.current_render_data = Some(RenderSceneState::new(data));
        }

        let use_material_batches = self.use_material_batches();
        let Some(state) = self.current_render_data.as_mut() else {
            return;
        };

        let prev_config = state.data.render_config;

        if full {
            state.clear_objects();
            state.clear_lights();
            self.gpu_instance_updates.clear();
            self.gpu_instances_dirty = true;
            self.gpu_draws_dirty = true;
            self.gbuffer_draws_dirty = true;
            self.shadow_draws_dirty = true;
        }

        if let Some(cam) = camera {
            if full {
                state.data.previous_camera_transform = cam.transform;
            }
            if !full {
                state.data.previous_camera_transform = state.data.current_camera_transform;
            }
            state.data.current_camera_transform = cam.transform;
            state.data.camera_component = cam.camera;
        } else if full {
            state.data.previous_camera_transform = state.data.current_camera_transform;
        }

        if let Some(config) = render_config {
            state.data.render_config = config;
        }
        if let Some(graph) = render_graph {
            state.data.render_graph = graph;
        }
        let gpu_driven =
            state.data.render_config.gpu_driven && self.supports_indirect_first_instance;

        for id in objects_remove {
            if let Some((removed, moved_idx)) = state.remove_object(id) {
                objects_changed = true;
                self.gpu_draws_dirty = true;
                self.gbuffer_draws_dirty = true;
                if removed.casts_shadow {
                    self.shadow_draws_dirty = true;
                }
                if let Some(idx) = moved_idx {
                    self.gpu_instance_updates.push(idx);
                }
            }
        }
        let material_index_map = &self.material_index_map;
        for obj in objects_upsert {
            let mesh_id = obj.mesh_id;
            let material_id = obj.material_id;
            let casts_shadow = obj.casts_shadow;
            let lod_index = obj.lod_index;
            match state.upsert_object(obj) {
                ObjectUpdateResult::Inserted { index, material_id } => {
                    objects_changed = true;
                    self.gpu_instance_updates.push(index);
                    self.gpu_draws_dirty = true;
                    self.gbuffer_draws_dirty = true;
                    if casts_shadow {
                        self.shadow_draws_dirty = true;
                    }
                    if material_index_map
                        .get(material_id)
                        .filter(|entry| entry.version == material_version)
                        .is_none()
                    {
                        materials_needed = true;
                    }
                }
                ObjectUpdateResult::Updated {
                    index,
                    prev_mesh_id,
                    prev_material_id,
                    prev_lod_index,
                    prev_casts_shadow,
                    transform_changed,
                } => {
                    let mesh_changed = prev_mesh_id != mesh_id;
                    let material_changed = prev_material_id != material_id;
                    let lod_changed = prev_lod_index != lod_index;
                    let casts_shadow_changed = prev_casts_shadow != casts_shadow;
                    let any_change = transform_changed
                        || mesh_changed
                        || material_changed
                        || lod_changed
                        || casts_shadow_changed;
                    if any_change {
                        objects_changed = true;
                    }
                    if mesh_changed {
                        self.gpu_draws_dirty = true;
                    }
                    if material_changed && use_material_batches {
                        self.gpu_draws_dirty = true;
                        self.gbuffer_draws_dirty = true;
                    }
                    if mesh_changed || (lod_changed && !gpu_driven) {
                        self.gbuffer_draws_dirty = true;
                        if prev_casts_shadow || casts_shadow {
                            self.shadow_draws_dirty = true;
                        }
                    }
                    if casts_shadow_changed {
                        self.shadow_draws_dirty = true;
                    }
                    if material_changed
                        && material_index_map
                            .get(material_id)
                            .filter(|entry| entry.version == material_version)
                            .is_none()
                    {
                        materials_needed = true;
                    }
                    if transform_changed || mesh_changed || material_changed || casts_shadow_changed
                    {
                        self.gpu_instance_updates.push(index);
                    }
                }
                ObjectUpdateResult::Unchanged => {}
            }
        }
        for id in lights_remove {
            state.remove_light(id);
            lights_changed = true;
        }
        for light in lights_upsert {
            if state.upsert_light(light) {
                lights_changed = true;
            }
        }

        state.data.timestamp = Instant::now();
        if objects_changed {
            self.instances_dirty = true;
            self.shadow_instances_dirty = true;
            self.shadow_uniforms_dirty = true;
            self.shadow_bounds_dirty = true;
        }
        if lights_changed {
            self.shadow_uniforms_dirty = true;
        }
        if materials_needed {
            self.materials_dirty = true;
        }
        if state.take_material_usage_dirty() {
            self.materials_dirty = true;
        }
        if camera_changed {
            self.shadow_uniforms_dirty = true;
        }
        if config_changed {
            let next = state.data.render_config;
            if next.gbuffer_pass != prev_config.gbuffer_pass {
                self.instances_dirty = true;
            }
            if next.shadow_pass != prev_config.shadow_pass {
                self.shadow_instances_dirty = true;
                self.shadow_uniforms_dirty = true;
                self.shadow_bounds_dirty = true;
            }
        }
        if let Some(requests) = streaming_requests {
            self.streaming_requests = Some(requests);
            self.streaming_dirty = true;
        } else if full {
            self.streaming_requests = Some(Vec::new());
            self.streaming_dirty = true;
        }
    }

    fn refresh_streaming_plan(&mut self, data: &RenderData) {
        self.resolve_pending_materials();
        if self.current_pressure() != self.streaming_pressure {
            self.streaming_dirty = true;
        }
        let interval = data.render_config.streaming_interval_frames;
        let interval_elapsed =
            interval > 0 && self.frame_index.saturating_sub(self.streaming_last_frame) >= interval;
        if !self.streaming_dirty && !interval_elapsed {
            let evicted = self.pool.tick_eviction(self.frame_index);
            self.track_evicted_resources(evicted);
            return;
        }
        self.streaming_last_frame = self.frame_index;
        self.streaming_dirty = false;

        let extra_requests = self.streaming_requests.take();
        let allow_full_scan = data.render_config.streaming_allow_full_scan;
        let streaming_plan =
            self.build_streaming_plan(data, extra_requests.as_deref(), allow_full_scan);
        self.streaming_requests = extra_requests;
        let request_budget = data.render_config.streaming_request_budget as usize;
        self.mark_streaming_plan_usage(&streaming_plan);
        self.process_streaming_plan(&streaming_plan, request_budget);
        let evicted = self.pool.tick_eviction(self.frame_index);
        self.track_evicted_resources(evicted);
        self.frame_inputs.set(streaming_plan);
    }

    fn bump_egui_epoch(&mut self) {
        self.egui_texture_cache.epoch = self.egui_texture_cache.epoch.wrapping_add(1);
        self.frame_inputs.set(self.egui_texture_cache.clone());
    }

    fn pressure_rank(pressure: MemoryPressure) -> u8 {
        match pressure {
            MemoryPressure::None => 0,
            MemoryPressure::Soft => 1,
            MemoryPressure::Hard => 2,
        }
    }

    fn streaming_pressure_age(&self) -> u32 {
        self.frame_index
            .saturating_sub(self.streaming_pressure_frame)
    }

    fn update_streaming_pressure(&mut self) -> MemoryPressure {
        let immediate = self.current_pressure();
        if immediate != self.streaming_pressure {
            let immediate_rank = Self::pressure_rank(immediate);
            let current_rank = Self::pressure_rank(self.streaming_pressure);
            if immediate_rank > current_rank {
                self.streaming_pressure = immediate;
                self.streaming_pressure_frame = self.frame_index;
            } else if self.streaming_pressure_age() >= self.streaming_tuning.pressure_release_frames
            {
                self.streaming_pressure = immediate;
                self.streaming_pressure_frame = self.frame_index;
            }
        }
        self.streaming_pressure
    }

    fn streaming_caps(&self, pressure: MemoryPressure) -> (usize, usize, usize, usize) {
        match pressure {
            MemoryPressure::Hard => {
                let caps = self.streaming_tuning.caps_hard;
                (caps.global, caps.mesh, caps.material, caps.texture)
            }
            MemoryPressure::Soft => {
                let caps = self.streaming_tuning.caps_soft;
                (caps.global, caps.mesh, caps.material, caps.texture)
            }
            MemoryPressure::None => {
                let caps = self.streaming_tuning.caps_none;
                (caps.global, caps.mesh, caps.material, caps.texture)
            }
        }
    }

    pub fn into_parts(self) -> (wgpu::Instance, wgpu::Surface<'static>) {
        (self.instance, self.surface)
    }

    pub(crate) fn adapter_backend(&self) -> wgpu::Backend {
        self.device_caps.adapter_info.backend
    }

    pub(crate) fn take_snapshot(&mut self) -> RendererSnapshot {
        let vram = self.pool.vram_budget().global.clone();
        let pool_config = self.streaming_tuning.pool_config();
        let replacement_pool = GpuResourcePool::new(
            vram.soft_limit_bytes,
            vram.hard_limit_bytes.max(vram.soft_limit_bytes),
            pool_config,
        );

        RendererSnapshot {
            pool: std::mem::replace(&mut self.pool, replacement_pool),
            current_render_data: self.current_render_data.take(),
            pending_render_delta: self.pending_render_delta.take(),
            meshes: std::mem::replace(&mut self.meshes, SlotVec::new()),
            materials: std::mem::replace(&mut self.materials, SlotVec::new()),
            material_index_map: std::mem::replace(&mut self.material_index_map, SlotVec::new()),
            material_index_order: std::mem::take(&mut self.material_index_order),
            material_version: self.material_version,
            material_bindings_version: self.material_bindings_version,
            pending_materials: std::mem::take(&mut self.pending_materials),
            mesh_lod_state: std::mem::replace(&mut self.mesh_lod_state, SlotVec::new()),
            texture_low_res_state: std::mem::replace(
                &mut self.texture_low_res_state,
                SlotVec::new(),
            ),
            pass_overrides: std::mem::take(&mut self.pass_overrides),
            egui_texture_cache: std::mem::take(&mut self.egui_texture_cache),
            streaming_tuning: self.streaming_tuning,
            streaming_requests: self.streaming_requests.take(),
        }
    }

    pub(crate) fn restore_snapshot(&mut self, snapshot: RendererSnapshot) {
        let RendererSnapshot {
            pool,
            current_render_data,
            pending_render_delta,
            meshes,
            materials,
            material_index_map,
            material_index_order,
            material_version,
            material_bindings_version,
            pending_materials,
            mesh_lod_state,
            texture_low_res_state,
            pass_overrides,
            egui_texture_cache,
            streaming_tuning,
            streaming_requests,
        } = snapshot;

        self.pool = pool;
        self.pool.apply_config(streaming_tuning.pool_config());
        self.pool.clear_transient_aliases();

        self.current_render_data = current_render_data;
        self.pending_render_delta = pending_render_delta;
        self.meshes = meshes;
        self.materials = materials;
        self.material_index_map = material_index_map;
        self.material_index_order = material_index_order;
        self.material_version = material_version;
        self.material_bindings_version = material_bindings_version;
        self.pending_materials = pending_materials;
        self.mesh_lod_state = mesh_lod_state;
        self.texture_low_res_state = texture_low_res_state;
        self.pass_overrides = pass_overrides;
        self.egui_texture_cache = egui_texture_cache;
        self.streaming_tuning = streaming_tuning;
        self.streaming_requests = streaming_requests;

        self.streaming_inflight = SlotVec::new();
        self.recently_evicted = SlotVec::new();
        self.streaming_pressure = MemoryPressure::None;
        self.streaming_pressure_frame = self.frame_index;
        self.streaming_dirty = true;
        self.streaming_last_frame = 0;
        self.streaming_request_cursor = 0;
        self.streaming_scan_cursor = 0;
        self.force_bindgroups_backend = false;

        self.materials_dirty = true;
        self.instances_dirty = true;
        self.shadow_instances_dirty = true;
        self.shadow_uniforms_dirty = true;
        self.shadow_bounds_dirty = true;
        self.shadow_bounds = None;
        self.gbuffer_draws_dirty = true;
        self.shadow_draws_dirty = true;
        self.gpu_instances_dirty = true;
        self.gpu_draws_dirty = true;
        self.gpu_instance_updates.clear();
        self.gpu_cull_dirty = true;

        self.cached_material_buffer = None;
        self.cached_texture_views.clear();
        self.cached_texture_array_size = 0;
        self.cached_material_signature = 0;
        self.cached_texture_overflow = false;
        self.cached_material_textures = None;
        self.cached_material_textures_signature = 0;
        self.cached_gbuffer_instances = None;
        self.cached_gbuffer_batches = Arc::new(Vec::new());
        self.cached_shadow_instances = None;
        self.cached_shadow_batches = Arc::new(Vec::new());
        self.cached_shadow_uniforms_buffer = None;
        self.cached_shadow_matrices_buffer = None;

        self.gpu_instance_buffer = None;
        self.gpu_instance_capacity = 0;
        self.gpu_visible_buffer = None;
        self.gpu_visible_capacity = 0;
        self.gpu_total_capacity = 0;
        self.gpu_shadow_buffer = None;
        self.gpu_shadow_capacity = 0;
        self.gpu_indirect_gbuffer = None;
        self.gpu_indirect_shadow = None;
        self.gpu_mesh_tasks_gbuffer = None;
        self.gpu_mesh_tasks_shadow = None;
        self.gpu_mesh_tasks_capacity = 0;
        self.gpu_mesh_task_meta_buffer = None;
        self.gpu_mesh_task_meta_capacity = 0;
        self.gpu_mesh_meta_buffer = None;
        self.gpu_mesh_meta_len = 0;
        self.gpu_mesh_meta_capacity = 0;
        self.gpu_mesh_indices.clear();
        self.gpu_mesh_material_indices.clear();
        self.gpu_indirect_capacity = 0;
        self.gpu_draws = Arc::new(Vec::new());
        self.gpu_draw_count = 0;

        self.bundle_resource_epoch = 0;
        self.bundle_resource_change_frame = 0;
        self.bundle_invalidate_pending = true;

        self.needs_atmosphere_precompute = true;
        self.fallback_mesh = None;
        self.init_fallback_mesh();
        self.bump_egui_epoch();
    }

    fn streaming_upgrade_allowed(&self, pressure: MemoryPressure, priority: f32) -> bool {
        let age = self.streaming_pressure_age();
        match pressure {
            MemoryPressure::Hard => false,
            MemoryPressure::Soft => {
                age >= self.streaming_tuning.soft_upgrade_delay_frames
                    && priority >= self.streaming_tuning.upgrade_priority_soft
            }
            MemoryPressure::None => age >= self.streaming_tuning.upgrade_cooldown_frames,
        }
    }

    fn adjust_request_priority(
        &self,
        resource: ResourceId,
        asset_id: usize,
        base_priority: f32,
    ) -> f32 {
        let mut priority = base_priority;
        if let Some(entry) = self.pool.entry(resource) {
            if entry.residency == Residency::Resident {
                priority *= self.streaming_tuning.resident_priority_boost;
            }
        }
        if self.streaming_inflight.get(asset_id).is_some() {
            priority *= self.streaming_tuning.inflight_priority_boost;
        }

        if let Some(last_evicted) = self.recently_evicted.get(asset_id) {
            if self.frame_index.saturating_sub(*last_evicted)
                < self.streaming_tuning.recent_evict_frames
            {
                priority *= self.streaming_tuning.recent_evict_penalty;
            }
        }

        priority
    }

    fn current_pressure(&self) -> MemoryPressure {
        let budget = self.pool.vram_budget().global.clone();
        if budget.current_bytes >= budget.hard_limit_bytes {
            MemoryPressure::Hard
        } else if budget.current_bytes >= budget.soft_limit_bytes {
            MemoryPressure::Soft
        } else {
            MemoryPressure::None
        }
    }

    fn drain_pool_evictions(&mut self) {
        let evicted = self.pool.drain_evictions();
        if !evicted.is_empty() {
            self.track_evicted_resources(evicted);
        }
    }

    fn drain_pool_binding_changes(&mut self) {
        let changes = self.pool.drain_binding_changes();
        if changes.is_empty() {
            return;
        }
        self.invalidate_bundles_for_resources(&changes);

        let mut texture_changed = false;
        let mut mesh_buffer_changed = false;
        let mut material_buffer_changed = false;
        for id in &changes {
            match id.kind() {
                ResourceKind::Texture => texture_changed = true,
                ResourceKind::Buffer => {
                    if let Some(entry) = self.pool.entry(*id) {
                        if let Some(asset_id) = entry.asset_stream_id.map(|v| v as usize) {
                            if self.meshes.get(asset_id).is_some() {
                                mesh_buffer_changed = true;
                            } else if self.materials.get(asset_id).is_some()
                                || self.pending_materials.contains_key(&asset_id)
                            {
                                material_buffer_changed = true;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if texture_changed {
            self.cached_texture_views.clear();
            self.cached_texture_array_size = 0;
            self.cached_material_textures = None;
            self.cached_material_textures_signature = 0;
        }
        if texture_changed || material_buffer_changed {
            self.materials_dirty = true;
        }
        if mesh_buffer_changed {
            self.instances_dirty = true;
            self.shadow_instances_dirty = true;
            self.shadow_uniforms_dirty = true;
            self.shadow_bounds_dirty = true;
            self.gbuffer_draws_dirty = true;
            self.shadow_draws_dirty = true;
            self.gpu_draws_dirty = true;
        }
    }

    fn note_bundle_resource_change(&mut self) {
        self.bundle_resource_epoch = self.bundle_resource_epoch.wrapping_add(1);
        self.bundle_resource_change_frame = self.frame_index;
        self.bundle_invalidate_pending = true;
        self.invalidate_render_bundles();
    }

    fn invalidate_render_bundles(&mut self) {
        let Some(active) = self.active_graph.as_ref() else {
            return;
        };
        for node in active.graph.nodes() {
            if let Some(data) = node.pass.user_data() {
                if let Some(pass) = data.downcast_ref::<GBufferPass>() {
                    pass.clear_bundle_cache();
                } else if let Some(pass) = data.downcast_ref::<ShadowPass>() {
                    pass.clear_bundle_cache();
                }
            }
        }
    }

    fn invalidate_bundles_for_resources(&mut self, evicted: &[ResourceId]) {
        let Some(active) = self.active_graph.as_ref() else {
            return;
        };
        for node in active.graph.nodes() {
            if let Some(data) = node.pass.user_data() {
                if let Some(pass) = data.downcast_ref::<GBufferPass>() {
                    pass.invalidate_bundles_for_resources(evicted);
                } else if let Some(pass) = data.downcast_ref::<ShadowPass>() {
                    pass.invalidate_bundles_for_resources(evicted);
                }
            }
        }
    }

    fn track_evicted_resources(&mut self, evicted: Vec<ResourceId>) {
        if evicted.is_empty() {
            return;
        }

        self.invalidate_bundles_for_resources(&evicted);

        let mut texture_evicted = false;
        let mut mesh_buffer_evicted = false;
        let mut material_buffer_evicted = false;
        for id in evicted {
            match id.kind() {
                ResourceKind::Texture => texture_evicted = true,
                ResourceKind::Buffer => {
                    if let Some(entry) = self.pool.entry(id) {
                        if let Some(asset_id) = entry.asset_stream_id.map(|v| v as usize) {
                            if self.meshes.get(asset_id).is_some() {
                                mesh_buffer_evicted = true;
                            } else if self.materials.get(asset_id).is_some()
                                || self.pending_materials.contains_key(&asset_id)
                            {
                                material_buffer_evicted = true;
                            }
                        }
                    }
                }
                _ => {}
            }
            if let Some(asset_id) = self
                .pool
                .entry(id)
                .and_then(|e| e.asset_stream_id)
                .map(|v| v as usize)
            {
                self.streaming_inflight.remove(asset_id);
                self.recently_evicted.insert(asset_id, self.frame_index);
            }
        }

        if texture_evicted {
            self.cached_texture_views.clear();
            self.cached_texture_array_size = 0;
            self.cached_material_textures = None;
            self.cached_material_textures_signature = 0;
        }
        if texture_evicted || material_buffer_evicted {
            self.materials_dirty = true;
        }
        if mesh_buffer_evicted {
            self.instances_dirty = true;
            self.shadow_instances_dirty = true;
            self.shadow_uniforms_dirty = true;
            self.shadow_bounds_dirty = true;
            self.gbuffer_draws_dirty = true;
            self.shadow_draws_dirty = true;
            self.gpu_draws_dirty = true;
        }
    }

    fn pre_evict_for_upload(&mut self, additional_bytes: u64) {
        if additional_bytes == 0 {
            return;
        }
        let budget = self.pool.vram_budget().global.clone();
        if budget.hard_limit_bytes == 0 {
            return;
        }
        let projected = budget.current_bytes.saturating_add(additional_bytes);
        let target = if projected > budget.hard_limit_bytes {
            projected - budget.hard_limit_bytes
        } else if projected > budget.soft_limit_bytes {
            projected - budget.soft_limit_bytes
        } else {
            0
        };
        if target > 0 {
            let evicted = self.pool.evict_budget_based(self.frame_index, target, None);
            self.track_evicted_resources(evicted);
        }
    }

    pub(crate) fn prepare_for_recreate(&mut self) {
        // Release swapchain views and GPU handles before the surface/device is dropped.
        self.clear_active_graph();
        self.frame_inputs.remove::<SwapchainFrameInput>();
        self.bump_egui_epoch();
        self.pool.evict_all();
        self.update_stats();
    }

    fn handle_resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        // Release swapchain views before reconfiguring to avoid DX12 resize failures.
        self.clear_active_graph();
        self.bump_egui_epoch();
        self.frame_inputs.remove::<SwapchainFrameInput>();

        self.surface_size = new_size;
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn clear_active_graph(&mut self) {
        if let Some(active) = self.active_graph.take() {
            self.evict_graph_resources(&active);
        }
        self.pool.clear_transient_aliases();
    }

    fn ensure_graph_ready(
        &mut self,
        spec: &RenderGraphSpec,
        config_sig: RenderGraphConfigSignature,
    ) -> Result<(), RendererError> {
        let needs_rebuild = match &self.active_graph {
            None => true,
            Some(active) => {
                active.spec_version != spec.version
                    || active.surface_size != self.surface_size
                    || active.surface_format != self.surface_config.format
                    || active.config_signature != config_sig
            }
        };

        if needs_rebuild {
            self.clear_active_graph();

            let params = RenderGraphBuildParams {
                surface_size: self.surface_size,
                surface_format: self.surface_config.format,
                shadow_format: self.shadow_format,
                device_caps: Arc::clone(&self.device_caps),
                blue_noise_view: Arc::new(self.blue_noise_view.clone()),
                blue_noise_sampler: Arc::new(self.blue_noise_sampler.clone()),
                config: config_sig,
                shadow_map_resolution: config_sig.shadow_map_resolution,
                shadow_cascade_count: config_sig.shadow_cascade_count,
            };

            let build = spec.build(&params, &mut self.pool);
            let graph = build.graph;
            let compiled = graph.compile(&self.pool).map_err(|e| {
                RendererError::ResourceCreation(format!("Graph compile failed: {e:?}"))
            })?;

            let alias_pairs: Vec<(ResourceId, ResourceId)> = compiled
                .transient_aliases
                .iter()
                .map(|alias| (alias.alias, alias.root))
                .collect();
            if config_sig.use_transient_aliasing {
                self.pool.apply_transient_aliases(&alias_pairs);
            } else {
                self.pool.clear_transient_aliases();
            }

            self.active_graph = Some(ActiveGraph {
                spec_version: spec.version,
                surface_size: self.surface_size,
                surface_format: self.surface_config.format,
                swapchain_id: build.swapchain_id,
                resource_ids: build.resource_ids,
                hiz_id: build.hiz_id,
                graph,
                compiled,
                config_signature: config_sig,
            });
            self.sync_pass_overrides();
        }

        Ok(())
    }

    fn evict_graph_resources(&mut self, active: &ActiveGraph) {
        for id in &active.resource_ids {
            self.pool.evict(*id);
        }
        self.pool.evict(active.swapchain_id);
    }

    fn update_swapchain_entry(&mut self, id: ResourceId, view: wgpu::TextureView) {
        if self.pool.entry(id).is_none() {
            self.pool.ensure_logical(
                id,
                ResourceDesc::External,
                ResourceUsageHints::default(),
                self.frame_index,
            );
        }
        if let Some(entry) = self.pool.entry_mut(id) {
            entry.texture_view = Some(view);
        }
        self.pool.mark_resident(id, self.frame_index);
    }

    fn init_fallback_mesh(&mut self) {
        if self.fallback_mesh.is_some() {
            return;
        }

        let p = 0.5f32;
        let v = |position, normal, tex_coord, tangent| {
            Vertex::new(position, normal, tex_coord, tangent)
        };
        let vertices = vec![
            // +X
            v(
                [p, -p, -p],
                [1.0, 0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ),
            v(
                [p, p, -p],
                [1.0, 0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ),
            v([p, p, p], [1.0, 0.0, 0.0], [1.0, 1.0], [0.0, 1.0, 0.0, 1.0]),
            v(
                [p, -p, p],
                [1.0, 0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ),
            // -X
            v(
                [-p, -p, p],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ),
            v(
                [-p, p, p],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ),
            v(
                [-p, p, -p],
                [-1.0, 0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ),
            v(
                [-p, -p, -p],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ),
            // +Y
            v(
                [-p, p, -p],
                [0.0, 1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ),
            v(
                [-p, p, p],
                [0.0, 1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ),
            v([p, p, p], [0.0, 1.0, 0.0], [1.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
            v(
                [p, p, -p],
                [0.0, 1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ),
            // -Y
            v(
                [-p, -p, p],
                [0.0, -1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ),
            v(
                [-p, -p, -p],
                [0.0, -1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ),
            v(
                [p, -p, -p],
                [0.0, -1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0, -1.0, 1.0],
            ),
            v(
                [p, -p, p],
                [0.0, -1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0, -1.0, 1.0],
            ),
            // +Z
            v(
                [-p, -p, p],
                [0.0, 0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ),
            v(
                [p, -p, p],
                [0.0, 0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ),
            v([p, p, p], [0.0, 0.0, 1.0], [1.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
            v(
                [-p, p, p],
                [0.0, 0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ),
            // -Z
            v(
                [p, -p, -p],
                [0.0, 0.0, -1.0],
                [0.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0],
            ),
            v(
                [-p, -p, -p],
                [0.0, 0.0, -1.0],
                [1.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0],
            ),
            v(
                [-p, p, -p],
                [0.0, 0.0, -1.0],
                [1.0, 1.0],
                [-1.0, 0.0, 0.0, 1.0],
            ),
            v(
                [p, p, -p],
                [0.0, 0.0, -1.0],
                [0.0, 1.0],
                [-1.0, 0.0, 0.0, 1.0],
            ),
        ];
        let indices: [u32; 36] = [
            0, 1, 2, 0, 2, 3, // +X
            4, 5, 6, 4, 6, 7, // -X
            8, 9, 10, 8, 10, 11, // +Y
            12, 13, 14, 12, 14, 15, // -Y
            16, 17, 18, 16, 18, 19, // +Z
            20, 21, 22, 20, 22, 23, // -Z
        ];

        let vertex_usage =
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE;
        let index_usage =
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE;
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FallbackMesh/Vertices"),
                contents: bytemuck::cast_slice(vertices.as_slice()),
                usage: vertex_usage,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FallbackMesh/Indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: index_usage,
            });

        let vertex_desc = ResourceDesc::Buffer {
            size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
            usage: vertex_usage,
        };
        let index_desc = ResourceDesc::Buffer {
            size: (indices.len() * std::mem::size_of::<u32>()) as u64,
            usage: index_usage,
        };
        let vertex_hints = ResourceUsageHints {
            flags: ResourceFlags::PINNED
                | ResourceFlags::PREFER_RESIDENT
                | ResourceFlags::STABLE_ID,
            estimated_size_bytes: vertex_desc.estimate_size_bytes(),
        };
        let index_hints = ResourceUsageHints {
            flags: ResourceFlags::PINNED
                | ResourceFlags::PREFER_RESIDENT
                | ResourceFlags::STABLE_ID,
            estimated_size_bytes: index_desc.estimate_size_bytes(),
        };
        let vertex_id = self.pool.create_logical(
            vertex_desc.clone(),
            Some(vertex_hints),
            self.frame_index,
            None,
        );
        let index_id = self.pool.create_logical(
            index_desc.clone(),
            Some(index_hints),
            self.frame_index,
            None,
        );
        self.insert_buffer_entry(vertex_id, vertex_desc, vertex_buffer, vertex_hints, None);
        self.insert_buffer_entry(index_id, index_desc, index_buffer, index_hints, None);

        let meshlet_data = build_meshlet_lod(vertices.as_slice(), &indices);
        let meshlet_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let empty_desc = MeshletDesc {
            vertex_offset: 0,
            vertex_count: 0,
            index_offset: 0,
            index_count: 0,
            bounds_center: [0.0; 3],
            bounds_radius: 0.0,
        };
        let descs_bytes = if meshlet_data.descs.is_empty() {
            bytemuck::bytes_of(&empty_desc)
        } else {
            bytemuck::cast_slice(meshlet_data.descs.as_ref())
        };
        let verts_bytes = if meshlet_data.vertices.is_empty() {
            bytemuck::cast_slice(&[0u32])
        } else {
            bytemuck::cast_slice(meshlet_data.vertices.as_ref())
        };
        let inds_bytes = if meshlet_data.indices.is_empty() {
            bytemuck::cast_slice(&[0u32])
        } else {
            bytemuck::cast_slice(meshlet_data.indices.as_ref())
        };

        let descs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FallbackMesh/MeshletDescs"),
                contents: descs_bytes,
                usage: meshlet_usage,
            });
        let descs_desc = ResourceDesc::Buffer {
            size: descs_bytes.len() as u64,
            usage: meshlet_usage,
        };
        let descs_hints = ResourceUsageHints {
            flags: ResourceFlags::PREFER_RESIDENT | ResourceFlags::STREAMING,
            estimated_size_bytes: descs_desc.estimate_size_bytes(),
        };
        let descs_id = self.pool.create_logical(
            descs_desc.clone(),
            Some(descs_hints),
            self.frame_index,
            None,
        );
        self.insert_buffer_entry(descs_id, descs_desc, descs_buffer, descs_hints, None);

        let verts_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FallbackMesh/MeshletVerts"),
                contents: verts_bytes,
                usage: meshlet_usage,
            });
        let verts_desc = ResourceDesc::Buffer {
            size: verts_bytes.len() as u64,
            usage: meshlet_usage,
        };
        let verts_hints = ResourceUsageHints {
            flags: ResourceFlags::PREFER_RESIDENT | ResourceFlags::STREAMING,
            estimated_size_bytes: verts_desc.estimate_size_bytes(),
        };
        let verts_id = self.pool.create_logical(
            verts_desc.clone(),
            Some(verts_hints),
            self.frame_index,
            None,
        );
        self.insert_buffer_entry(verts_id, verts_desc, verts_buffer, verts_hints, None);

        let inds_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FallbackMesh/MeshletIndices"),
                contents: inds_bytes,
                usage: meshlet_usage,
            });
        let inds_desc = ResourceDesc::Buffer {
            size: inds_bytes.len() as u64,
            usage: meshlet_usage,
        };
        let inds_hints = ResourceUsageHints {
            flags: ResourceFlags::PREFER_RESIDENT | ResourceFlags::STREAMING,
            estimated_size_bytes: inds_desc.estimate_size_bytes(),
        };
        let inds_id =
            self.pool
                .create_logical(inds_desc.clone(), Some(inds_hints), self.frame_index, None);
        self.insert_buffer_entry(inds_id, inds_desc, inds_buffer, inds_hints, None);

        self.fallback_mesh = Some(MeshGpu {
            vertex: vertex_id,
            lods: vec![MeshLodResource {
                buffer: index_id,
                index_count: indices.len() as u32,
                meshlets: MeshletGpu {
                    descs: descs_id,
                    vertices: verts_id,
                    indices: inds_id,
                    count: meshlet_data.meshlet_count(),
                },
            }],
            bounds: Aabb {
                min: Vec3::splat(-p),
                max: Vec3::splat(p),
            },
        });
    }

    fn upload_mesh(
        &mut self,
        id: usize,
        vertices: &[Vertex],
        lod_indices: &[Arc<[u32]>],
        meshlets: &[MeshletLodData],
        bounds: Aabb,
    ) -> Result<(), RendererError> {
        let vertex_bytes = (vertices.len() * std::mem::size_of::<Vertex>()) as u64;
        let lod_bytes: u64 = lod_indices
            .iter()
            .map(|indices| (indices.len() * std::mem::size_of::<u32>()) as u64)
            .sum();
        let meshlet_bytes: u64 = meshlets
            .iter()
            .map(|lod| {
                let descs = lod.descs.len() as u64 * std::mem::size_of::<MeshletDesc>() as u64;
                let verts = lod.vertices.len() as u64 * std::mem::size_of::<u32>() as u64;
                let inds = lod.indices.len() as u64 * std::mem::size_of::<u32>() as u64;
                descs + verts + inds
            })
            .sum();
        let new_bytes = vertex_bytes + lod_bytes + meshlet_bytes;
        let mut reclaimed_bytes = 0u64;
        if let Some(existing) = self.meshes.get(id) {
            if let Some(entry) = self.pool.entry(existing.vertex) {
                if entry.residency == Residency::Resident {
                    reclaimed_bytes = reclaimed_bytes.saturating_add(entry.desc_size_bytes);
                }
            }
            for lod in &existing.lods {
                if let Some(entry) = self.pool.entry(lod.buffer) {
                    if entry.residency == Residency::Resident {
                        reclaimed_bytes = reclaimed_bytes.saturating_add(entry.desc_size_bytes);
                    }
                }
                for meshlet_id in [
                    lod.meshlets.descs,
                    lod.meshlets.vertices,
                    lod.meshlets.indices,
                ] {
                    if let Some(entry) = self.pool.entry(meshlet_id) {
                        if entry.residency == Residency::Resident {
                            reclaimed_bytes = reclaimed_bytes.saturating_add(entry.desc_size_bytes);
                        }
                    }
                }
            }
        }
        let net_bytes = new_bytes.saturating_sub(reclaimed_bytes);
        self.pre_evict_for_upload(net_bytes);

        let existing_lod_ids: Vec<ResourceId> = self
            .meshes
            .get(id)
            .map(|mesh| mesh.lods.iter().map(|lod| lod.buffer).collect())
            .unwrap_or_default();

        if let Some(existing) = self.meshes.get(id) {
            for lod in &existing.lods {
                self.pool.evict(lod.buffer);
                self.pool.evict(lod.meshlets.descs);
                self.pool.evict(lod.meshlets.vertices);
                self.pool.evict(lod.meshlets.indices);
            }
        }

        let vertex_usage =
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE;
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh-vbo-{}", id)),
                contents: bytemuck::cast_slice(vertices),
                usage: vertex_usage,
            });

        let vertex_desc = ResourceDesc::Buffer {
            size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
            usage: vertex_usage,
        };
        let mut vertex_hints = ResourceUsageHints {
            flags: ResourceFlags::PREFER_RESIDENT,
            estimated_size_bytes: vertex_desc.estimate_size_bytes(),
        };
        vertex_hints.flags |= ResourceFlags::FREQUENT_UPDATE;
        vertex_hints.flags |= ResourceFlags::STREAMING;

        let vertex_id = self
            .pool
            .asset_id_to_resource(ResourceKind::Buffer, id as u32);
        self.insert_buffer_entry(
            vertex_id,
            vertex_desc,
            vertex_buffer,
            vertex_hints,
            Some(id),
        );

        let index_usage =
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE;
        let meshlet_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let empty_meshlet = MeshletLodData::default();
        let mut lods = Vec::new();
        for (lod, indices) in lod_indices.iter().enumerate() {
            let meshlet_data = meshlets.get(lod).unwrap_or(&empty_meshlet);
            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-ibo-{}-lod{}", id, lod)),
                    contents: bytemuck::cast_slice(indices.as_ref()),
                    usage: index_usage,
                });

            let desc = ResourceDesc::Buffer {
                size: (indices.len() * std::mem::size_of::<u32>()) as u64,
                usage: index_usage,
            };
            let mut hints = ResourceUsageHints {
                flags: ResourceFlags::PREFER_RESIDENT | ResourceFlags::STREAMING,
                estimated_size_bytes: desc.estimate_size_bytes(),
            };
            hints.flags |= ResourceFlags::STABLE_ID;
            let mut index_id = None;
            if let Some(&candidate) = existing_lod_ids.get(lod) {
                if self.pool.entry(candidate).is_some() {
                    index_id = Some(candidate);
                }
            }
            let index_id = index_id.unwrap_or_else(|| {
                self.pool
                    .create_logical(desc.clone(), Some(hints), self.frame_index, None)
            });
            let mut entry = crate::graphics::graph::logic::residency::GpuResourceEntry::new(
                index_id,
                desc.kind(),
                hints.estimated_size_bytes,
                hints,
                self.frame_index,
                desc,
            );
            entry.buffer = Some(index_buffer);
            entry.asset_stream_id = Some(id as u32);
            self.pool.insert_entry(entry);
            self.pool.mark_used(index_id, self.frame_index);

            let meshlet_count = meshlet_data.meshlet_count();
            let empty_desc = MeshletDesc {
                vertex_offset: 0,
                vertex_count: 0,
                index_offset: 0,
                index_count: 0,
                bounds_center: [0.0; 3],
                bounds_radius: 0.0,
            };
            let descs_bytes = if meshlet_data.descs.is_empty() {
                bytemuck::bytes_of(&empty_desc)
            } else {
                bytemuck::cast_slice(meshlet_data.descs.as_ref())
            };
            let verts_bytes = if meshlet_data.vertices.is_empty() {
                bytemuck::cast_slice(&[0u32])
            } else {
                bytemuck::cast_slice(meshlet_data.vertices.as_ref())
            };
            let inds_bytes = if meshlet_data.indices.is_empty() {
                bytemuck::cast_slice(&[0u32])
            } else {
                bytemuck::cast_slice(meshlet_data.indices.as_ref())
            };

            let descs_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-meshlets-{}-lod{}-descs", id, lod)),
                    contents: descs_bytes,
                    usage: meshlet_usage,
                });
            let descs_desc = ResourceDesc::Buffer {
                size: descs_bytes.len() as u64,
                usage: meshlet_usage,
            };
            let descs_hints = ResourceUsageHints {
                flags: ResourceFlags::PREFER_RESIDENT | ResourceFlags::STREAMING,
                estimated_size_bytes: descs_desc.estimate_size_bytes(),
            };
            let descs_id = self.pool.create_logical(
                descs_desc.clone(),
                Some(descs_hints),
                self.frame_index,
                None,
            );
            self.insert_buffer_entry(descs_id, descs_desc, descs_buffer, descs_hints, Some(id));

            let verts_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-meshlets-{}-lod{}-verts", id, lod)),
                    contents: verts_bytes,
                    usage: meshlet_usage,
                });
            let verts_desc = ResourceDesc::Buffer {
                size: verts_bytes.len() as u64,
                usage: meshlet_usage,
            };
            let verts_hints = ResourceUsageHints {
                flags: ResourceFlags::PREFER_RESIDENT | ResourceFlags::STREAMING,
                estimated_size_bytes: verts_desc.estimate_size_bytes(),
            };
            let verts_id = self.pool.create_logical(
                verts_desc.clone(),
                Some(verts_hints),
                self.frame_index,
                None,
            );
            self.insert_buffer_entry(verts_id, verts_desc, verts_buffer, verts_hints, Some(id));

            let inds_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-meshlets-{}-lod{}-indices", id, lod)),
                    contents: inds_bytes,
                    usage: meshlet_usage,
                });
            let inds_desc = ResourceDesc::Buffer {
                size: inds_bytes.len() as u64,
                usage: meshlet_usage,
            };
            let inds_hints = ResourceUsageHints {
                flags: ResourceFlags::PREFER_RESIDENT | ResourceFlags::STREAMING,
                estimated_size_bytes: inds_desc.estimate_size_bytes(),
            };
            let inds_id = self.pool.create_logical(
                inds_desc.clone(),
                Some(inds_hints),
                self.frame_index,
                None,
            );
            self.insert_buffer_entry(inds_id, inds_desc, inds_buffer, inds_hints, Some(id));

            lods.push(MeshLodResource {
                buffer: index_id,
                index_count: indices.len() as u32,
                meshlets: MeshletGpu {
                    descs: descs_id,
                    vertices: verts_id,
                    indices: inds_id,
                    count: meshlet_count,
                },
            });
        }

        self.meshes.insert(
            id,
            MeshGpu {
                vertex: vertex_id,
                lods,
                bounds,
            },
        );
        let requested_lod = self
            .streaming_inflight
            .get(id)
            .and_then(|state| state.requested_lod)
            .unwrap_or(0);
        self.mesh_lod_state.insert(id, requested_lod);
        self.streaming_inflight.remove(id);
        self.instances_dirty = true;
        self.shadow_instances_dirty = true;
        self.shadow_uniforms_dirty = true;
        self.shadow_bounds_dirty = true;
        self.gbuffer_draws_dirty = true;
        self.shadow_draws_dirty = true;
        self.gpu_instances_dirty = true;
        self.gpu_draws_dirty = true;

        Ok(())
    }

    fn upload_texture(
        &mut self,
        id: usize,
        data: &[u8],
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<(), RendererError> {
        let mip_level_count = (width.max(height) as f32).log2().floor() as u32 + 1;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let desc = ResourceDesc::Texture2D {
            width,
            height,
            mip_levels: mip_level_count,
            layers: 1,
            format,
            usage,
        };
        let rid = self
            .pool
            .asset_id_to_resource(ResourceKind::Texture, id as u32);
        let new_bytes = desc.estimate_size_bytes();
        let reclaimed_bytes = self
            .pool
            .entry(rid)
            .and_then(|entry| {
                if entry.residency == Residency::Resident {
                    Some(entry.desc_size_bytes)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let net_bytes = new_bytes.saturating_sub(reclaimed_bytes);
        self.pre_evict_for_upload(net_bytes);

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            label: Some(&format!("texture-{}", id)),
            view_formats: &[],
        });

        let block_size = format.block_size(None).unwrap_or(4);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(block_size * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        if mip_level_count > 1 {
            let mut mip_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("texture-mip-encoder"),
                    });
            self.mipmap_generator.generate_mips(
                &mut mip_encoder,
                &self.device,
                &texture,
                mip_level_count,
            );
            self.queue.submit(std::iter::once(mip_encoder.finish()));
        }

        let view = texture.create_view(&Default::default());
        let hints = ResourceUsageHints {
            flags: ResourceFlags::PREFER_RESIDENT
                | ResourceFlags::FREQUENT_UPDATE
                | ResourceFlags::STREAMING,
            estimated_size_bytes: desc.estimate_size_bytes(),
        };

        let mut entry = crate::graphics::graph::logic::residency::GpuResourceEntry::new(
            rid,
            desc.kind(),
            hints.estimated_size_bytes,
            hints,
            self.frame_index,
            desc,
        );
        entry.texture = Some(texture);
        entry.texture_view = Some(view);
        entry.asset_stream_id = Some(id as u32);
        self.pool.insert_entry(entry);
        self.pool.mark_used(rid, self.frame_index);
        let low_res = self
            .streaming_inflight
            .get(id)
            .map(|state| state.force_low_res)
            .unwrap_or(false);
        self.texture_low_res_state.insert(id, low_res);
        self.streaming_inflight.remove(id);
        self.materials_dirty = true;
        Ok(())
    }

    fn try_upload_material(&mut self, mat_data: MaterialGpuData) -> bool {
        let mat_id = mat_data.id;
        let mut texture_index =
            |pool: &mut GpuResourcePool, opt_id: Option<usize>| -> Option<i32> {
                match opt_id {
                    None => Some(-1),
                    Some(tex_id) => {
                        let rid = pool.asset_id_to_resource(ResourceKind::Texture, tex_id as u32);
                        let ready = pool
                            .entry(rid)
                            .map(|e| e.residency == Residency::Resident && e.texture_view.is_some())
                            .unwrap_or(false);
                        if ready {
                            Some(rid.index() as i32)
                        } else {
                            None
                        }
                    }
                }
            };

        let albedo_idx = match texture_index(&mut self.pool, mat_data.albedo_texture_id) {
            Some(idx) => idx,
            None => return false,
        };
        let normal_idx = match texture_index(&mut self.pool, mat_data.normal_texture_id) {
            Some(idx) => idx,
            None => return false,
        };
        let mra_idx = match texture_index(&mut self.pool, mat_data.metallic_roughness_texture_id) {
            Some(idx) => idx,
            None => return false,
        };
        let emission_idx =
            texture_index(&mut self.pool, mat_data.emission_texture_id).unwrap_or(-1);

        let shader_data = MaterialShaderData {
            albedo: mat_data.albedo,
            metallic: mat_data.metallic,
            roughness: mat_data.roughness,
            ao: mat_data.ao,
            emission_strength: mat_data.emission_strength,
            albedo_idx,
            normal_idx,
            metallic_roughness_idx: mra_idx,
            emission_idx,
            emission_color: mat_data.emission_color,
            _padding: 0.0,
        };

        let desc = ResourceDesc::Buffer {
            size: std::mem::size_of::<MaterialShaderData>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        };
        let rid = self
            .pool
            .asset_id_to_resource(ResourceKind::Buffer, mat_id as u32);
        let new_bytes = desc.estimate_size_bytes();
        let reclaimed_bytes = self
            .pool
            .entry(rid)
            .and_then(|entry| {
                if entry.residency == Residency::Resident {
                    Some(entry.desc_size_bytes)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let net_bytes = new_bytes.saturating_sub(reclaimed_bytes);
        self.pre_evict_for_upload(net_bytes);

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("material-buffer-{}", mat_data.id)),
                contents: bytemuck::bytes_of(&shader_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let hints = ResourceUsageHints {
            flags: ResourceFlags::PREFER_RESIDENT | ResourceFlags::STREAMING,
            estimated_size_bytes: desc.estimate_size_bytes(),
        };
        self.insert_buffer_entry(rid, desc, buffer, hints, Some(mat_id));

        self.materials.insert(
            mat_id,
            MaterialEntry {
                buffer: rid,
                meta: mat_data,
            },
        );
        self.materials_dirty = true;
        self.streaming_inflight.remove(mat_id);
        true
    }

    fn resolve_pending_materials(&mut self) {
        let pending_ids: Vec<usize> = self.pending_materials.keys().copied().collect();
        for id in pending_ids {
            if let Some(mat) = self.pending_materials.get(&id).cloned() {
                if self.try_upload_material(mat) {
                    self.pending_materials.remove(&id);
                }
            }
        }
    }

    fn material_meta(&self, id: usize) -> Option<&MaterialGpuData> {
        self.materials
            .get(id)
            .map(|m| &m.meta)
            .or_else(|| self.pending_materials.get(&id))
    }

    fn material_index(&self, id: usize) -> Option<u32> {
        self.material_index_map
            .get(id)
            .filter(|entry| entry.version == self.material_version)
            .map(|entry| entry.index)
    }

    fn insert_buffer_entry(
        &mut self,
        id: ResourceId,
        desc: ResourceDesc,
        buffer: wgpu::Buffer,
        hints: ResourceUsageHints,
        asset_stream_id: Option<usize>,
    ) {
        let mut entry = crate::graphics::graph::logic::residency::GpuResourceEntry::new(
            id,
            desc.kind(),
            hints.estimated_size_bytes,
            hints,
            self.frame_index,
            desc,
        );
        entry.buffer = Some(buffer);
        entry.asset_stream_id = asset_stream_id.map(|v| v as u32);
        self.pool.insert_entry(entry);
        self.pool.mark_used(id, self.frame_index);
    }

    fn ingest_render_data(&mut self, data: Arc<RenderData>) {
        self.pending_render_delta = None;
        self.streaming_requests = None;
        self.streaming_dirty = true;
        self.streaming_scan_cursor = 0;
        self.materials_dirty = true;
        self.instances_dirty = true;
        self.shadow_instances_dirty = true;
        self.shadow_uniforms_dirty = true;
        self.shadow_bounds_dirty = true;
        self.gbuffer_draws_dirty = true;
        self.shadow_draws_dirty = true;
        self.gpu_instances_dirty = true;
        self.gpu_draws_dirty = true;
        let data = (*data).clone();
        self.refresh_streaming_plan(&data);
        self.current_render_data = Some(RenderSceneState::new(data));
    }

    fn mark_streaming_plan_usage(&mut self, plan: &StreamingPlan) {
        for req in &plan.requests {
            match req.kind {
                AssetStreamKind::Mesh => {
                    let Some(mesh) = self.meshes.get(req.asset_id) else {
                        continue;
                    };
                    if mesh.lods.is_empty() {
                        continue;
                    }
                    self.pool.mark_used(mesh.vertex, self.frame_index);
                    let desired = req.max_lod.unwrap_or(0);
                    let require_meshlets = self.device_caps.supports_mesh_pipeline();
                    if let Some(lod_index) =
                        self.select_resident_lod(mesh, desired, require_meshlets)
                    {
                        if let Some(lod) = mesh.lods.get(lod_index) {
                            self.pool.mark_used(lod.buffer, self.frame_index);
                            self.pool.mark_used(lod.meshlets.descs, self.frame_index);
                            self.pool.mark_used(lod.meshlets.vertices, self.frame_index);
                            self.pool.mark_used(lod.meshlets.indices, self.frame_index);
                        }
                    }
                }
                AssetStreamKind::Material => {
                    if self.materials.get(req.asset_id).is_none() {
                        continue;
                    }
                    let rid = self
                        .pool
                        .asset_id_to_resource(ResourceKind::Buffer, req.asset_id as u32);
                    self.pool.mark_used(rid, self.frame_index);
                }
                AssetStreamKind::Texture => {
                    let rid = self
                        .pool
                        .asset_id_to_resource(ResourceKind::Texture, req.asset_id as u32);
                    self.pool.mark_used(rid, self.frame_index);
                }
            }
        }
    }

    fn accumulate_streaming_for_object(
        &mut self,
        obj: &RenderObject,
        camera_pos: Vec3,
        predicted_cam: Vec3,
        pressure: MemoryPressure,
        priority_floor: f32,
        base_lod_bias: usize,
        critical_threshold: f32,
    ) {
        let world_center = obj.current_transform.position;
        let dist_now = (world_center - camera_pos).length();
        let dist_pred = (world_center - predicted_cam).length();
        let distance = dist_now.min(dist_pred);

        let mesh_radius = self
            .meshes
            .get(obj.mesh_id)
            .map(|m| m.bounds.extents().length())
            .unwrap_or(1.0);
        let size_bias = self.streaming_tuning.priority_size_bias.max(0.0);
        let distance_bias = self.streaming_tuning.priority_distance_bias.max(0.0);
        let lod_bias = self.streaming_tuning.priority_lod_bias.max(f32::EPSILON);
        let denom = (distance + distance_bias).max(f32::EPSILON);
        let base_priority = (mesh_radius + size_bias) / denom;
        let lod_penalty = 1.0 / (obj.lod_index as f32 + lod_bias);
        let mut priority = base_priority * lod_penalty;
        if obj.casts_shadow {
            priority *= self.streaming_tuning.shadow_priority_boost;
        }
        let critical = priority >= critical_threshold;
        if !critical && priority_floor > 0.0 && priority < priority_floor {
            return;
        }

        let mut desired_lod = obj.lod_index;
        let prediction_margin = self.streaming_tuning.prediction_distance_threshold.max(0.0);
        if dist_pred + prediction_margin < dist_now {
            desired_lod = desired_lod.saturating_sub(1);
        }

        let mut lod_bias = base_lod_bias;
        if priority > self.streaming_tuning.priority_near {
            lod_bias = lod_bias.saturating_sub(1);
        }
        if critical {
            lod_bias = lod_bias.saturating_sub(1);
        }
        desired_lod = desired_lod.saturating_add(lod_bias);

        if let Some(mesh_gpu) = self.meshes.get(obj.mesh_id) {
            let max_idx = mesh_gpu.lods.len().saturating_sub(1);
            if matches!(pressure, MemoryPressure::Hard)
                && self.streaming_tuning.force_lowest_lod_hard
            {
                desired_lod = max_idx;
            } else {
                desired_lod = desired_lod.min(max_idx);
            }
        }

        let force_low_res = match pressure {
            MemoryPressure::Hard => {
                self.streaming_tuning.force_low_res_hard
                    || priority < self.streaming_tuning.low_res_priority_hard
            }
            MemoryPressure::Soft => priority < self.streaming_tuning.low_res_priority_soft,
            MemoryPressure::None => false,
        };

        let mesh_resource = self
            .pool
            .asset_id_to_resource(ResourceKind::Buffer, obj.mesh_id as u32);
        let mesh_priority = self.adjust_request_priority(mesh_resource, obj.mesh_id, priority);
        self.streaming_mesh_scratch.upsert(
            obj.mesh_id,
            mesh_resource,
            AssetStreamKind::Mesh,
            Some(desired_lod),
            mesh_priority,
            false,
            critical,
        );

        let material_resource = self
            .pool
            .asset_id_to_resource(ResourceKind::Buffer, obj.material_id as u32);
        let material_priority =
            self.adjust_request_priority(material_resource, obj.material_id, priority);
        self.streaming_material_scratch.upsert(
            obj.material_id,
            material_resource,
            AssetStreamKind::Material,
            None,
            material_priority,
            false,
            critical,
        );

        if let Some(mat) = self.material_meta(obj.material_id) {
            let tex_ids = [
                mat.albedo_texture_id,
                mat.normal_texture_id,
                mat.metallic_roughness_texture_id,
                mat.emission_texture_id,
            ];
            for tex in tex_ids.iter().flatten() {
                let texture_resource = self
                    .pool
                    .asset_id_to_resource(ResourceKind::Texture, *tex as u32);
                let texture_priority =
                    self.adjust_request_priority(texture_resource, *tex, priority);
                self.streaming_texture_scratch.upsert(
                    *tex,
                    texture_resource,
                    AssetStreamKind::Texture,
                    None,
                    texture_priority,
                    force_low_res,
                    critical,
                );
            }
        }
    }

    fn build_streaming_plan(
        &mut self,
        data: &RenderData,
        extra_requests: Option<&[AssetStreamingRequest]>,
        allow_full_scan: bool,
    ) -> StreamingPlan {
        let pressure = self.update_streaming_pressure();
        let (global_cap, mesh_cap, material_cap, texture_cap) = self.streaming_caps(pressure);
        let priority_floor = match pressure {
            MemoryPressure::Hard => self.streaming_tuning.priority_floor_hard,
            MemoryPressure::Soft => self.streaming_tuning.priority_floor_soft,
            MemoryPressure::None => self.streaming_tuning.priority_floor_none,
        };
        let base_lod_bias: usize = match pressure {
            MemoryPressure::Hard => self.streaming_tuning.lod_bias_hard,
            MemoryPressure::Soft => self.streaming_tuning.lod_bias_soft,
            MemoryPressure::None => 0,
        };

        let camera_pos = data.current_camera_transform.position;
        let camera_delta =
            data.current_camera_transform.position - data.previous_camera_transform.position;
        let motion_epsilon = self.streaming_tuning.prediction_motion_epsilon.max(0.0);
        let prediction_frames = self.streaming_tuning.prediction_frames.max(0.0);
        let predicted_cam = if camera_delta.length_squared() > motion_epsilon {
            camera_pos + camera_delta * prediction_frames
        } else {
            camera_pos
        };

        self.streaming_mesh_scratch.clear();
        self.streaming_material_scratch.clear();
        self.streaming_texture_scratch.clear();

        let mut upsert_req = |scratch: &mut StreamRequestScratch,
                              resource: ResourceId,
                              kind: AssetStreamKind,
                              max_lod: Option<usize>,
                              adjusted_prio: f32,
                              asset_id: usize,
                              force_low_res: bool,
                              critical: bool| {
            scratch.upsert(
                asset_id,
                resource,
                kind,
                max_lod,
                adjusted_prio,
                force_low_res,
                critical,
            );
        };

        let critical_threshold = self.streaming_tuning.priority_critical;

        let do_full_scan = allow_full_scan;
        let scan_budget = data.render_config.streaming_scan_budget as usize;
        let object_count = data.objects.len();
        if do_full_scan || scan_budget == 0 || object_count <= scan_budget {
            for obj in &data.objects {
                self.accumulate_streaming_for_object(
                    obj,
                    camera_pos,
                    predicted_cam,
                    pressure,
                    priority_floor,
                    base_lod_bias,
                    critical_threshold,
                );
            }
            if object_count > 0 {
                self.streaming_scan_cursor = 0;
            }
        } else if object_count > 0 {
            let start = self.streaming_scan_cursor.min(object_count);
            let mut scanned = 0usize;
            while scanned < scan_budget {
                let idx = (start + scanned) % object_count;
                let obj = &data.objects[idx];
                self.accumulate_streaming_for_object(
                    obj,
                    camera_pos,
                    predicted_cam,
                    pressure,
                    priority_floor,
                    base_lod_bias,
                    critical_threshold,
                );
                scanned += 1;
            }
            self.streaming_scan_cursor = (start + scanned) % object_count;
        }

        if let Some(requests) = extra_requests {
            for req in requests {
                let mut priority = req.priority;
                let critical = priority >= critical_threshold;
                if !critical && priority_floor > 0.0 && priority < priority_floor {
                    continue;
                }

                let force_low_res = match pressure {
                    MemoryPressure::Hard => {
                        self.streaming_tuning.force_low_res_hard
                            || priority < self.streaming_tuning.low_res_priority_hard
                    }
                    MemoryPressure::Soft => priority < self.streaming_tuning.low_res_priority_soft,
                    MemoryPressure::None => false,
                };

                match req.kind {
                    AssetStreamKind::Mesh => {
                        let mut desired_lod = req.max_lod.unwrap_or(0);
                        let mut lod_bias = base_lod_bias;
                        if priority > self.streaming_tuning.priority_near {
                            lod_bias = lod_bias.saturating_sub(1);
                        }
                        if critical {
                            lod_bias = lod_bias.saturating_sub(1);
                        }
                        desired_lod = desired_lod.saturating_add(lod_bias);

                        if let Some(mesh_gpu) = self.meshes.get(req.id) {
                            let max_idx = mesh_gpu.lods.len().saturating_sub(1);
                            if matches!(pressure, MemoryPressure::Hard)
                                && self.streaming_tuning.force_lowest_lod_hard
                            {
                                desired_lod = max_idx;
                            } else {
                                desired_lod = desired_lod.min(max_idx);
                            }
                        }

                        let mesh_resource = self
                            .pool
                            .asset_id_to_resource(ResourceKind::Buffer, req.id as u32);
                        priority = self.adjust_request_priority(mesh_resource, req.id, priority);
                        upsert_req(
                            &mut self.streaming_mesh_scratch,
                            mesh_resource,
                            AssetStreamKind::Mesh,
                            Some(desired_lod),
                            priority,
                            req.id,
                            false,
                            critical,
                        );
                    }
                    AssetStreamKind::Material => {
                        let material_resource = self
                            .pool
                            .asset_id_to_resource(ResourceKind::Buffer, req.id as u32);
                        priority =
                            self.adjust_request_priority(material_resource, req.id, priority);
                        upsert_req(
                            &mut self.streaming_material_scratch,
                            material_resource,
                            AssetStreamKind::Material,
                            None,
                            priority,
                            req.id,
                            false,
                            critical,
                        );

                        if let Some(mat) = self.material_meta(req.id) {
                            let tex_ids = [
                                mat.albedo_texture_id,
                                mat.normal_texture_id,
                                mat.metallic_roughness_texture_id,
                                mat.emission_texture_id,
                            ];
                            for tex in tex_ids.iter().flatten() {
                                let texture_resource = self
                                    .pool
                                    .asset_id_to_resource(ResourceKind::Texture, *tex as u32);
                                let texture_priority =
                                    self.adjust_request_priority(texture_resource, *tex, priority);
                                upsert_req(
                                    &mut self.streaming_texture_scratch,
                                    texture_resource,
                                    AssetStreamKind::Texture,
                                    None,
                                    texture_priority,
                                    *tex,
                                    force_low_res,
                                    critical,
                                );
                            }
                        }
                    }
                    AssetStreamKind::Texture => {
                        let texture_resource = self
                            .pool
                            .asset_id_to_resource(ResourceKind::Texture, req.id as u32);
                        priority = self.adjust_request_priority(texture_resource, req.id, priority);
                        upsert_req(
                            &mut self.streaming_texture_scratch,
                            texture_resource,
                            AssetStreamKind::Texture,
                            None,
                            priority,
                            req.id,
                            force_low_res,
                            critical,
                        );
                    }
                }
            }
        }

        let cmp_priority = |a: &StreamRequest, b: &StreamRequest| {
            b.priority
                .total_cmp(&a.priority)
                .then_with(|| a.resource.raw().cmp(&b.resource.raw()))
        };
        let mut cap_and_sort = |requests: &mut Vec<StreamRequest>, cap: usize| {
            if cap == 0 {
                requests.clear();
                return;
            }
            if requests.len() > cap {
                requests.select_nth_unstable_by(cap, cmp_priority);
                requests.truncate(cap);
            }
            requests.sort_by(cmp_priority);
        };
        let mut split_requests = |requests: &mut Vec<StreamRequest>| {
            let mut critical = Vec::new();
            let mut optional = Vec::new();
            for req in requests.drain(..) {
                if req.critical {
                    critical.push(req);
                } else {
                    optional.push(req);
                }
            }
            (critical, optional)
        };

        self.streaming_mesh_scratch
            .drain_into(&mut self.streaming_mesh_requests);
        self.streaming_material_scratch
            .drain_into(&mut self.streaming_material_requests);
        self.streaming_texture_scratch
            .drain_into(&mut self.streaming_texture_requests);

        let (mut mesh_critical, mut mesh_optional) =
            split_requests(&mut self.streaming_mesh_requests);
        let (mut material_critical, mut material_optional) =
            split_requests(&mut self.streaming_material_requests);
        let (mut texture_critical, mut texture_optional) =
            split_requests(&mut self.streaming_texture_requests);

        cap_and_sort(&mut mesh_optional, mesh_cap);
        cap_and_sort(&mut material_optional, material_cap);
        cap_and_sort(&mut texture_optional, texture_cap);

        let mut critical_requests = Vec::new();
        critical_requests.append(&mut mesh_critical);
        critical_requests.append(&mut material_critical);
        critical_requests.append(&mut texture_critical);

        let mut optional_requests = Vec::new();
        optional_requests.append(&mut mesh_optional);
        optional_requests.append(&mut material_optional);
        optional_requests.append(&mut texture_optional);

        let critical_len = critical_requests.len();
        cap_and_sort(&mut critical_requests, critical_len);
        let remaining = global_cap.saturating_sub(critical_requests.len());
        cap_and_sort(&mut optional_requests, remaining);

        let mut requests = Vec::with_capacity(critical_requests.len() + optional_requests.len());
        requests.append(&mut critical_requests);
        requests.append(&mut optional_requests);

        let mut priority_lookup = HashMap::with_capacity(requests.len());
        for req in &requests {
            priority_lookup
                .entry(req.resource)
                .and_modify(|p: &mut f32| *p = (*p).max(req.priority))
                .or_insert(req.priority);
        }

        StreamingPlan {
            requests,
            priority_lookup,
            pressure,
        }
    }

    fn evict_streaming(&mut self, plan: &StreamingPlan) {
        let global_budget = self.pool.vram_budget().global.clone();
        if plan.pressure == MemoryPressure::None
            || global_budget.current_bytes <= global_budget.soft_limit_bytes
        {
            return;
        }
        let mut need = global_budget.current_bytes - global_budget.soft_limit_bytes;
        if self.streaming_tuning.pool_eviction_scan_budget == 0 {
            return;
        }
        let scan_budget = self.streaming_tuning.pool_eviction_scan_budget;
        let grace_frames = match plan.pressure {
            MemoryPressure::Hard => self.streaming_tuning.evict_hard_grace_frames,
            MemoryPressure::Soft => self.streaming_tuning.evict_soft_grace_frames,
            MemoryPressure::None => 0,
        };
        let mut scanned = 0usize;
        let mut to_evict = Vec::new();
        for idx in self.pool.lru_tail_indices() {
            if scanned >= scan_budget || need == 0 {
                break;
            }
            scanned += 1;
            let entry = match self.pool.entry_by_index(idx) {
                Some(entry) => entry,
                None => continue,
            };
            if entry.residency != Residency::Resident {
                continue;
            }
            if entry.is_pinned() {
                continue;
            }
            if entry.flags().contains(ResourceFlags::TRANSIENT) {
                continue;
            }
            if entry.last_used_frame == self.frame_index {
                continue;
            }
            if entry.last_used_frame + grace_frames > self.frame_index {
                continue;
            }
            let priority = plan.priority_lookup.get(&entry.id).copied().unwrap_or(0.0);
            if plan.pressure == MemoryPressure::Soft
                && priority > self.streaming_tuning.evict_soft_protect_priority
                && entry.idle_frames(self.frame_index) < self.pool.idle_frames_before_evict
            {
                continue;
            }
            let size = entry.desc_size_bytes;
            let id = entry.id;
            to_evict.push(id);
            need = need.saturating_sub(size);
        }
        if to_evict.is_empty() {
            return;
        }
        for id in &to_evict {
            self.pool.evict(*id);
        }
        self.track_evicted_resources(to_evict);
    }

    fn evict_unplanned_idle(&mut self, plan: &StreamingPlan) {
        let idle_frames = self.streaming_tuning.evict_unplanned_idle_frames;
        if idle_frames == 0 {
            return;
        }
        let scan_budget = self.streaming_tuning.pool_eviction_scan_budget;
        let max_evictions = self.streaming_tuning.pool_max_evictions_per_tick;
        if scan_budget == 0 || max_evictions == 0 {
            return;
        }

        let mut scanned = 0usize;
        let mut evicted = 0usize;
        let mut to_evict = Vec::new();

        for idx in self.pool.lru_tail_indices() {
            if scanned >= scan_budget || evicted >= max_evictions {
                break;
            }
            scanned += 1;
            let entry = match self.pool.entry_by_index(idx) {
                Some(entry) => entry,
                None => continue,
            };
            if entry.residency != Residency::Resident {
                continue;
            }
            if entry.is_pinned() {
                continue;
            }
            if entry.flags().contains(ResourceFlags::TRANSIENT) {
                continue;
            }
            if entry.asset_stream_id.is_none() {
                continue;
            }
            if plan.priority_lookup.contains_key(&entry.id) {
                continue;
            }
            if entry.last_used_frame.saturating_add(idle_frames) > self.frame_index {
                continue;
            }

            to_evict.push(entry.id);
            evicted += 1;
        }

        if to_evict.is_empty() {
            return;
        }
        for id in &to_evict {
            self.pool.evict(*id);
        }
        self.track_evicted_resources(to_evict);
    }

    fn process_streaming_plan(&mut self, plan: &StreamingPlan, request_budget: usize) {
        self.evict_streaming(plan);
        self.evict_unplanned_idle(plan);

        let total = plan.requests.len();
        if total == 0 {
            self.streaming_request_cursor = 0;
            return;
        }
        let max_scan = if request_budget == 0 {
            total
        } else {
            request_budget.min(total)
        };
        let mut index = self.streaming_request_cursor % total;
        let mut processed = 0usize;
        let mut advance = |index: &mut usize, processed: &mut usize| {
            *processed += 1;
            *index = (*index + 1) % total;
        };

        while processed < max_scan {
            let req = &plan.requests[index];
            if plan.pressure == MemoryPressure::Hard
                && req.priority < self.streaming_tuning.priority_floor_hard
            {
                advance(&mut index, &mut processed);
                continue;
            }
            let mesh_primary = req.kind == AssetStreamKind::Mesh
                && req.resource
                    == self
                        .pool
                        .asset_id_to_resource(ResourceKind::Buffer, req.asset_id as u32);
            let current_mesh_lod = if mesh_primary {
                self.mesh_lod_state.get(req.asset_id).copied()
            } else {
                None
            };
            let mesh_lod_mismatch = if mesh_primary {
                req.max_lod.map_or(false, |desired| {
                    current_mesh_lod.map_or(true, |current| current != desired)
                })
            } else {
                false
            };
            let mesh_upgrade = if mesh_primary {
                match (req.max_lod, current_mesh_lod) {
                    (Some(desired), Some(current)) => desired < current,
                    _ => false,
                }
            } else {
                false
            };
            let texture_current_low = if req.kind == AssetStreamKind::Texture {
                self.texture_low_res_state.get(req.asset_id).copied()
            } else {
                None
            };
            let texture_low_res_mismatch = if req.kind == AssetStreamKind::Texture {
                texture_current_low
                    .map_or(req.force_low_res, |current| current != req.force_low_res)
            } else {
                false
            };
            let texture_upgrade = if req.kind == AssetStreamKind::Texture {
                matches!(texture_current_low, Some(true)) && !req.force_low_res
            } else {
                false
            };
            let upgrade = mesh_upgrade || texture_upgrade;
            let entry_opt = self.pool.entry(req.resource);
            let asset_id = if let Some(entry) = entry_opt {
                if entry.residency == Residency::Resident
                    && !mesh_lod_mismatch
                    && !texture_low_res_mismatch
                {
                    advance(&mut index, &mut processed);
                    continue;
                }
                entry
                    .asset_stream_id
                    .map(|v| v as usize)
                    .or_else(|| {
                        self.pool
                            .asset_id_from_resource(req.resource)
                            .map(|v| v as usize)
                    })
                    .or(Some(req.asset_id))
            } else {
                self.pool
                    .asset_id_from_resource(req.resource)
                    .map(|v| v as usize)
                    .or(Some(req.asset_id))
            };

            if let Some(asset_id) = asset_id {
                if upgrade && !self.streaming_upgrade_allowed(plan.pressure, req.priority) {
                    advance(&mut index, &mut processed);
                    continue;
                }
                if let Some(last_frame) = self.recently_evicted.get(asset_id) {
                    let age = self.frame_index.saturating_sub(*last_frame);
                    if age < self.streaming_tuning.evict_retry_frames
                        && req.priority < self.streaming_tuning.evict_retry_priority
                    {
                        advance(&mut index, &mut processed);
                        continue;
                    }
                    if upgrade && age < self.streaming_tuning.recent_evict_frames {
                        advance(&mut index, &mut processed);
                        continue;
                    }
                    if age < self.streaming_tuning.recent_evict_frames
                        && !mesh_lod_mismatch
                        && !texture_low_res_mismatch
                    {
                        advance(&mut index, &mut processed);
                        continue;
                    }
                }
                let should_send = match self.streaming_inflight.get(asset_id) {
                    Some(existing) => {
                        let beyond_cooldown = self.frame_index.saturating_sub(existing.last_frame)
                            > self.streaming_tuning.inflight_cooldown_frames;
                        let priority_bump = req.priority
                            > existing.priority * self.streaming_tuning.priority_bump_factor;
                        let lod_change = req.max_lod != existing.requested_lod;
                        let low_res_change = req.force_low_res != existing.force_low_res;
                        beyond_cooldown && (priority_bump || lod_change || low_res_change)
                    }
                    None => true,
                };
                if should_send {
                    let request = AssetStreamingRequest {
                        id: asset_id,
                        kind: req.kind,
                        priority: req.priority,
                        max_lod: req.max_lod,
                        force_low_res: req.force_low_res,
                    };
                    match self.asset_stream_sender.try_send(request) {
                        Ok(_) => {
                            self.streaming_inflight.insert(
                                asset_id,
                                StreamingState {
                                    priority: req.priority,
                                    last_frame: self.frame_index,
                                    requested_lod: req.max_lod,
                                    force_low_res: req.force_low_res,
                                },
                            );
                        }
                        Err(TrySendError::Full(_)) => {
                            self.streaming_dirty = true;
                            self.streaming_request_cursor = index % total;
                            return;
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            warn!("Asset stream channel disconnected");
                            self.streaming_request_cursor = 0;
                            return;
                        }
                    }
                }
            }

            advance(&mut index, &mut processed);
        }

        if max_scan < total {
            self.streaming_request_cursor = index % total;
        } else {
            self.streaming_request_cursor = 0;
        }
    }

    fn prepare_frame_globals(&mut self, scene: &RenderSceneState) -> Option<FrameGlobals> {
        let render_data = &scene.data;
        let alpha = 1.0;
        let prev_view_proj = self.prev_view_proj;
        let mut frame_render_config = render_data.render_config;
        let frames_in_flight = frame_render_config.frames_in_flight.max(1) as usize;
        if frames_in_flight != self.frames_in_flight {
            self.frames_in_flight = frames_in_flight;
            self.buffer_cache.resize(frames_in_flight);
        }
        if frame_render_config.gpu_driven && !self.supports_indirect_first_instance {
            if !self.warned_indirect_first_instance_missing {
                self.warned_indirect_first_instance_missing = true;
                tracing::warn!(
                    "GPU-driven indirect rendering disabled: INDIRECT_FIRST_INSTANCE not supported"
                );
            }
            frame_render_config.gpu_driven = false;
            self.shared_stats
                .gpu_fallbacks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if frame_render_config.gpu_multi_draw_indirect
            && !self.device_caps.supports_multi_draw_indirect()
        {
            if !self.warned_multi_draw_missing {
                self.warned_multi_draw_missing = true;
                tracing::warn!("Multi-draw indirect disabled: MULTI_DRAW_INDIRECT not supported");
            }
            frame_render_config.gpu_multi_draw_indirect = false;
            self.shared_stats
                .gpu_fallbacks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if frame_render_config.use_mesh_shaders && !self.device_caps.supports_mesh_pipeline() {
            if !self.warned_mesh_shaders_missing {
                self.warned_mesh_shaders_missing = true;
                tracing::warn!(
                    "Mesh shader path disabled: experimental mesh shader support unavailable"
                );
            }
            frame_render_config.use_mesh_shaders = false;
            self.shared_stats
                .gpu_fallbacks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let require_meshlets =
            frame_render_config.use_mesh_shaders && self.device_caps.supports_mesh_pipeline();
        let gpu_driven = frame_render_config.gpu_driven;
        if gpu_driven != self.gpu_driven_active {
            self.gpu_driven_active = gpu_driven;
            self.gpu_instances_dirty = true;
            self.gpu_draws_dirty = true;
            self.gpu_instance_updates.clear();
            self.gpu_cull_dirty = true;
        }
        if self.bundle_invalidate_pending {
            self.bundle_invalidate_pending = false;
            self.gbuffer_draws_version = self.gbuffer_draws_version.wrapping_add(1);
            self.shadow_draws_version = self.shadow_draws_version.wrapping_add(1);
            self.gpu_bundle_version = self.gpu_bundle_version.wrapping_add(1);
        }

        let (camera_uniforms, lights, view_proj, view_matrix) =
            self.calculate_uniforms(render_data, alpha);
        let camera_buffer = {
            let (buf, reallocated) = self.buffer_cache.camera.ensure_with_status(
                &self.device,
                std::mem::size_of::<CameraUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&camera_uniforms));
            let buffer = buf.clone();
            if reallocated {
                self.note_bundle_resource_change();
            }
            buffer
        };
        let lights_buffer = if lights.is_empty() {
            None
        } else {
            let bytes = bytemuck::cast_slice(&lights);
            let buf = self.buffer_cache.lights.ensure(
                &self.device,
                bytes.len() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue.write_buffer(buf, 0, bytes);
            Some(buf.clone())
        };

        let shader_constants = render_data.render_config.shader_constants;
        let render_constants_buffer = {
            let (buf, reallocated) = self.buffer_cache.render_constants.ensure_with_status(
                &self.device,
                std::mem::size_of_val(&shader_constants) as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&shader_constants));
            let buffer = buf.clone();
            if reallocated {
                self.note_bundle_resource_change();
            }
            buffer
        };

        let sky_uniforms = self.build_sky_uniforms(render_data);
        let sky_buffer = {
            let buf = self.buffer_cache.sky.ensure(
                &self.device,
                std::mem::size_of::<SkyUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&sky_uniforms));
            buf.clone()
        };

        if self.streaming_pressure != self.last_instance_pressure {
            self.instances_dirty = true;
            self.shadow_instances_dirty = true;
            self.shadow_uniforms_dirty = true;
            self.shadow_bounds_dirty = true;
            self.gbuffer_draws_dirty = true;
            self.shadow_draws_dirty = true;
        }

        let materials_dirty = self.materials_dirty;
        let material_needs_rebuild = materials_dirty || self.cached_texture_array_size == 0;
        let mut texture_overflow = self.cached_texture_overflow;
        let (material_buffer, texture_views, texture_array_size) = if material_needs_rebuild {
            let build = self.build_material_data(scene, materials_dirty);
            self.materials_dirty = false;
            texture_overflow = build.texture_overflow;
            self.cached_texture_overflow = build.texture_overflow;
            if build.changed {
                self.cached_material_buffer = build.buffer.clone();
                self.cached_texture_views = build.views.clone();
                self.cached_texture_array_size = build.size.max(1);
                self.cached_material_signature = build.signature;
                (build.buffer, build.views, self.cached_texture_array_size)
            } else {
                (
                    self.cached_material_buffer.clone(),
                    self.cached_texture_views.clone(),
                    self.cached_texture_array_size.max(1),
                )
            }
        } else {
            (
                self.cached_material_buffer.clone(),
                self.cached_texture_views.clone(),
                self.cached_texture_array_size.max(1),
            )
        };

        if !self.force_bindgroups_backend
            && matches!(
                self.binding_backend_kind,
                BindingBackendKind::BindlessModern | BindingBackendKind::BindlessFallback
            )
            && texture_overflow
        {
            self.force_bindgroups_backend = true;
            self.instances_dirty = true;
            self.gbuffer_draws_dirty = true;
            self.gpu_draws_dirty = true;
            self.gpu_instances_dirty = true;
            self.gpu_instance_updates.clear();
            self.gpu_cull_dirty = true;
            tracing::warn!(
                max_sampled_textures_per_shader_stage = self
                    .device_caps
                    .limits
                    .max_sampled_textures_per_shader_stage,
                "Bindless backend disabled: texture array limit exceeded; switching to bind groups."
            );
        }

        let frame_binding_backend = self.effective_binding_backend();
        let (texture_views, texture_array_size) =
            if frame_binding_backend == BindingBackendKind::BindGroups {
                let mut views = texture_views;
                if views.is_empty() {
                    views.push(self.fallback_view.clone());
                }
                (views, 1u32)
            } else {
                (texture_views, texture_array_size)
            };

        let material_textures = if frame_binding_backend == BindingBackendKind::BindGroups {
            let build = self.build_material_textures(materials_dirty);
            if build.changed {
                self.cached_material_textures = Some(build.textures.clone());
                self.cached_material_textures_signature = build.signature;
                self.material_bindings_version = self.material_bindings_version.wrapping_add(1);
            }
            self.cached_material_textures.clone()
        } else {
            None
        };

        let gbuffer_pass = render_data.render_config.gbuffer_pass;
        let shadow_pass = render_data.render_config.shadow_pass;
        let mut gbuffer_indirect = None;
        let mut shadow_indirect = None;
        let mut gbuffer_mesh_tasks = None;
        let mut shadow_mesh_tasks = None;
        let mut gpu_draws = Arc::new(Vec::new());
        let gpu_instance_count = render_data.objects.len() as u32;

        let mut gpu_frame = None;
        if gpu_driven && (gbuffer_pass || shadow_pass) {
            gpu_frame = self.prepare_gpu_driven_frame(render_data, alpha);
        }
        let use_gpu_driven = gpu_frame.is_some();
        if gpu_driven && !use_gpu_driven {
            frame_render_config.gpu_driven = false;
            self.shared_stats
                .gpu_fallbacks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let (
            gbuffer_instances,
            gbuffer_batches,
            shadow_instances,
            shadow_batches,
            shadow_uniforms_buffer,
            shadow_matrices_buffer,
        ) = if let Some(frame) = gpu_frame {
            gbuffer_indirect = frame.gbuffer_indirect.clone();
            shadow_indirect = frame.shadow_indirect.clone();
            gbuffer_mesh_tasks = frame.gbuffer_mesh_tasks.clone();
            shadow_mesh_tasks = frame.shadow_mesh_tasks.clone();
            gpu_draws = frame.draws.clone();

            let (shadow_uniforms_buffer, shadow_matrices_buffer) = if shadow_pass {
                let needs_rebuild =
                    self.shadow_uniforms_dirty || self.cached_shadow_uniforms_buffer.is_none();
                if needs_rebuild {
                    let (uniforms, matrices) =
                        self.build_shadow_uniform_buffers(render_data, view_matrix, alpha);
                    self.cached_shadow_uniforms_buffer = uniforms.clone();
                    self.cached_shadow_matrices_buffer = matrices.clone();
                    self.shadow_uniforms_dirty = false;
                    (uniforms, matrices)
                } else {
                    (
                        self.cached_shadow_uniforms_buffer.clone(),
                        self.cached_shadow_matrices_buffer.clone(),
                    )
                }
            } else {
                (None, None)
            };

            (
                frame.gbuffer_instances,
                Arc::new(Vec::new()),
                frame.shadow_instances,
                Arc::new(Vec::new()),
                shadow_uniforms_buffer,
                shadow_matrices_buffer,
            )
        } else {
            let (gbuffer_instances, gbuffer_batches) = if gbuffer_pass {
                if self.instances_dirty || self.cached_gbuffer_instances.is_none() {
                    let (instances, batches) =
                        self.build_gbuffer_instances(render_data, alpha, require_meshlets);
                    self.cached_gbuffer_instances = instances.clone();
                    self.cached_gbuffer_batches = batches.clone();
                    if self.gbuffer_draws_dirty {
                        self.gbuffer_draws_version = self.gbuffer_draws_version.wrapping_add(1);
                    }
                    self.gbuffer_draws_dirty = false;
                    self.instances_dirty = false;
                    self.last_instance_pressure = self.streaming_pressure;
                    (instances, batches)
                } else {
                    (
                        self.cached_gbuffer_instances.clone(),
                        self.cached_gbuffer_batches.clone(),
                    )
                }
            } else {
                (None, Arc::new(Vec::new()))
            };

            let (shadow_instances, shadow_batches, shadow_uniforms_buffer, shadow_matrices_buffer) =
                if shadow_pass {
                    let needs_instances = self.shadow_instances_dirty
                        || self.cached_shadow_instances.is_none()
                        || self.cached_shadow_uniforms_buffer.is_none();
                    if needs_instances {
                        let (instances, batches, uniforms, matrices) = self.build_shadow_data(
                            render_data,
                            alpha,
                            view_matrix,
                            require_meshlets,
                        );
                        self.cached_shadow_instances = instances.clone();
                        self.cached_shadow_batches = batches.clone();
                        self.cached_shadow_uniforms_buffer = uniforms.clone();
                        self.cached_shadow_matrices_buffer = matrices.clone();
                        if self.shadow_draws_dirty {
                            self.shadow_draws_version = self.shadow_draws_version.wrapping_add(1);
                        }
                        self.shadow_draws_dirty = false;
                        self.shadow_instances_dirty = false;
                        self.shadow_uniforms_dirty = false;
                        self.last_instance_pressure = self.streaming_pressure;
                        (instances, batches, uniforms, matrices)
                    } else if self.shadow_uniforms_dirty {
                        let (uniforms, matrices) =
                            self.build_shadow_uniform_buffers(render_data, view_matrix, alpha);
                        self.cached_shadow_uniforms_buffer = uniforms.clone();
                        self.cached_shadow_matrices_buffer = matrices.clone();
                        self.shadow_uniforms_dirty = false;
                        (
                            self.cached_shadow_instances.clone(),
                            self.cached_shadow_batches.clone(),
                            uniforms,
                            matrices,
                        )
                    } else {
                        (
                            self.cached_shadow_instances.clone(),
                            self.cached_shadow_batches.clone(),
                            self.cached_shadow_uniforms_buffer.clone(),
                            self.cached_shadow_matrices_buffer.clone(),
                        )
                    }
                } else {
                    (None, Arc::new(Vec::new()), None, None)
                };

            (
                gbuffer_instances,
                gbuffer_batches,
                shadow_instances,
                shadow_batches,
                shadow_uniforms_buffer,
                shadow_matrices_buffer,
            )
        };

        let forward_instances = gbuffer_instances.clone();
        let forward_batches = gbuffer_batches.clone();
        let shadow_uniforms_buffer = match shadow_uniforms_buffer {
            Some(buf) => Some(buf),
            None => {
                let mut uniforms = ShadowUniforms::default();
                uniforms.cascade_count = render_data.render_config.shadow_cascade_count.clamp(
                    1,
                    crate::graphics::renderer_common::common::MAX_SHADOW_CASCADES as u32,
                );
                let buf = self.buffer_cache.shadow_uniforms.ensure(
                    &self.device,
                    std::mem::size_of::<ShadowUniforms>() as u64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    self.frame_index,
                );
                self.queue
                    .write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
                Some(buf.clone())
            }
        };

        let bundle_mode = if use_gpu_driven {
            BundleMode::Gpu
        } else {
            BundleMode::Cpu
        };
        let bundle_resource_epoch = if frame_render_config.render_bundles {
            self.bundle_resource_epoch
        } else {
            0
        };
        let gbuffer_bundle_key = GBufferBundleKey {
            mode: bundle_mode,
            draw_version: if use_gpu_driven {
                self.gpu_bundle_version
            } else {
                self.gbuffer_draws_version
            },
            material_bindings_version: self.material_bindings_version,
            texture_array_size,
            binding_backend: frame_binding_backend,
            resource_epoch: bundle_resource_epoch,
        };
        let shadow_bundle_key = ShadowBundleKey {
            mode: bundle_mode,
            draw_version: if use_gpu_driven {
                self.gpu_bundle_version
            } else {
                self.shadow_draws_version
            },
            matrices_version: self.shadow_matrices_version,
            resource_epoch: bundle_resource_epoch,
        };

        self.prev_view_proj = view_proj;

        let atmos_changed = sky_uniforms != self.prev_sky_uniforms
            || shader_constants.planet_radius != self.prev_shader_constants.planet_radius
            || shader_constants.atmosphere_radius != self.prev_shader_constants.atmosphere_radius
            || shader_constants.sky_light_samples != self.prev_shader_constants.sky_light_samples;
        if atmos_changed {
            self.prev_sky_uniforms = sky_uniforms;
            self.prev_shader_constants = shader_constants;
            self.needs_atmosphere_precompute = true;
        }

        let debug_params = crate::graphics::passes::DebugCompositeParams {
            flags: render_data.render_config.debug_flags,
            _pad0: [0; 3],
            _pad1: [0; 4],
            _pad2: [0; 4],
        };
        let debug_params_buffer = {
            let buf = self.buffer_cache.debug_params.ensure(
                &self.device,
                std::mem::size_of::<crate::graphics::passes::DebugCompositeParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&debug_params));
            buf.clone()
        };

        let occlusion_camera_stable = {
            let prev_pos = render_data.previous_camera_transform.position;
            let cur_pos = render_data.current_camera_transform.position;
            let pos_delta = prev_pos.distance(cur_pos);
            let prev_rot = Quat::from(render_data.previous_camera_transform.rotation);
            let cur_rot = Quat::from(render_data.current_camera_transform.rotation);
            let rot_delta = 1.0 - prev_rot.dot(cur_rot).abs();
            let pos_epsilon = render_data
                .render_config
                .occlusion_stable_pos_epsilon
                .max(0.0);
            let rot_epsilon = render_data
                .render_config
                .occlusion_stable_rot_epsilon
                .max(0.0);
            pos_delta < pos_epsilon && rot_delta < rot_epsilon
        };
        if occlusion_camera_stable {
            self.occlusion_stable_frames = self.occlusion_stable_frames.saturating_add(1);
        } else {
            self.occlusion_stable_frames = 0;
        }

        let hiz_view = self
            .active_graph
            .as_ref()
            .and_then(|graph| graph.hiz_id)
            .and_then(|id| self.pool.texture_view(id).cloned());

        Some(FrameGlobals {
            frame_index: self.frame_index,
            device_caps: Arc::clone(&self.device_caps),
            binding_backend: frame_binding_backend,
            camera_buffer,
            lights_buffer,
            lights_len: lights.len() as u32,
            render_constants_buffer,
            shadow_uniforms_buffer,
            shadow_matrices_buffer,
            sky_buffer,
            material_buffer,
            material_textures,
            material_bindings_version: self.material_bindings_version,
            texture_views,
            texture_array_size,
            pbr_sampler: self.default_sampler.clone(),
            shadow_sampler: self.shadow_sampler.clone(),
            scene_sampler: self.scene_sampler.clone(),
            point_sampler: self.point_sampler.clone(),
            blue_noise_view: self.blue_noise_view.clone(),
            blue_noise_sampler: self.blue_noise_sampler.clone(),
            fallback_view: self.fallback_view.clone(),
            fallback_volume_view: self.fallback_volume_view.clone(),
            hiz_view,
            ibl_brdf_view: self.brdf_lut_view.clone(),
            ibl_irradiance_view: self.irradiance_map_view.clone(),
            ibl_prefiltered_view: self.prefiltered_env_map_view.clone(),
            ibl_sampler: self.ibl_sampler.clone(),
            brdf_lut_sampler: self.brdf_lut_sampler.clone(),
            atmosphere_bind_group: self.atmosphere_precomputer.sampling_bind_group.clone(),
            debug_params_buffer,
            gbuffer_instances,
            gbuffer_batches,
            gbuffer_indirect,
            shadow_instances,
            shadow_batches,
            shadow_indirect,
            gbuffer_mesh_tasks,
            shadow_mesh_tasks,
            gpu_draws,
            gpu_instance_count,
            forward_instances,
            forward_batches,
            alpha,
            camera_view_proj: view_proj,
            prev_view_proj,
            lights,
            shader_constants,
            sky_uniforms,
            surface_size: self.surface_size,
            render_config: frame_render_config,
            occlusion_camera_stable,
            gbuffer_bundle_key,
            shadow_bundle_key,
        })
    }

    fn run_occlusion_culling(&mut self, globals: &FrameGlobals) {
        let stats = &self.shared_stats;
        stats.occlusion_camera_stable.store(
            globals.occlusion_camera_stable as u32,
            std::sync::atomic::Ordering::Relaxed,
        );

        if !globals.render_config.occlusion_culling {
            stats.occlusion_status.store(
                OCCLUSION_STATUS_DISABLED,
                std::sync::atomic::Ordering::Relaxed,
            );
            stats
                .occlusion_instance_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        if !globals.render_config.gbuffer_pass {
            stats.occlusion_status.store(
                OCCLUSION_STATUS_NO_GBUFFER,
                std::sync::atomic::Ordering::Relaxed,
            );
            stats
                .occlusion_instance_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        let instances = match globals.gbuffer_instances.as_ref() {
            Some(buf) if buf.count > 0 => buf,
            _ => {
                stats.occlusion_status.store(
                    OCCLUSION_STATUS_NO_INSTANCES,
                    std::sync::atomic::Ordering::Relaxed,
                );
                stats
                    .occlusion_instance_count
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        };
        stats
            .occlusion_instance_count
            .store(instances.count, std::sync::atomic::Ordering::Relaxed);

        let hiz_id = match self.active_graph.as_ref().and_then(|graph| graph.hiz_id) {
            Some(id) => id,
            None => {
                stats.occlusion_status.store(
                    OCCLUSION_STATUS_NO_HIZ,
                    std::sync::atomic::Ordering::Relaxed,
                );
                return;
            }
        };

        let hiz_view = match self.pool.texture_view(hiz_id) {
            Some(view) => view.clone(),
            None => {
                stats.occlusion_status.store(
                    OCCLUSION_STATUS_NO_HIZ,
                    std::sync::atomic::Ordering::Relaxed,
                );
                return;
            }
        };

        let params = OcclusionParams {
            instance_count: instances.count,
            _pad0: [0; 3],
            depth_bias: globals.render_config.gpu_cull_depth_bias.max(0.0),
            rect_pad: globals.render_config.gpu_cull_rect_pad.max(0.0),
            _pad1: [0.0; 2],
        };
        let params_buffer = {
            let buf = self.buffer_cache.occlusion_params.ensure(
                &self.device,
                std::mem::size_of::<OcclusionParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&params));
            buf.clone()
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Occlusion/BG"),
            layout: &self.occlusion_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: globals.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Occlusion/Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Occlusion/Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.occlusion_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (instances.count + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        stats
            .occlusion_status
            .store(OCCLUSION_STATUS_RAN, std::sync::atomic::Ordering::Relaxed);
        stats
            .occlusion_last_frame
            .store(self.frame_index, std::sync::atomic::Ordering::Relaxed);
    }

    fn run_gpu_culling(&mut self, globals: &FrameGlobals) {
        let stats = &self.shared_stats;
        stats.occlusion_camera_stable.store(
            globals.occlusion_camera_stable as u32,
            std::sync::atomic::Ordering::Relaxed,
        );

        if !globals.render_config.gbuffer_pass && !globals.render_config.shadow_pass {
            stats.occlusion_status.store(
                OCCLUSION_STATUS_DISABLED,
                std::sync::atomic::Ordering::Relaxed,
            );
            stats
                .occlusion_instance_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        let instance_count = globals.gpu_instance_count;
        if instance_count == 0 || self.gpu_draw_count == 0 {
            stats.occlusion_status.store(
                OCCLUSION_STATUS_NO_INSTANCES,
                std::sync::atomic::Ordering::Relaxed,
            );
            stats
                .occlusion_instance_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        let (
            input,
            gbuffer_out,
            shadow_out,
            mesh_meta,
            gbuffer_indirect,
            shadow_indirect,
            mesh_task_meta,
            gbuffer_mesh_tasks,
            shadow_mesh_tasks,
        ) = match (
            self.gpu_instance_buffer.as_ref(),
            self.gpu_visible_buffer.as_ref(),
            self.gpu_shadow_buffer.as_ref(),
            self.gpu_mesh_meta_buffer.as_ref(),
            self.gpu_indirect_gbuffer.as_ref(),
            self.gpu_indirect_shadow.as_ref(),
            self.gpu_mesh_task_meta_buffer.as_ref(),
            self.gpu_mesh_tasks_gbuffer.as_ref(),
            self.gpu_mesh_tasks_shadow.as_ref(),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f), Some(g), Some(h), Some(i)) => {
                (a, b, c, d, e, f, g, h, i)
            }
            _ => {
                stats.occlusion_status.store(
                    OCCLUSION_STATUS_NO_INSTANCES,
                    std::sync::atomic::Ordering::Relaxed,
                );
                stats
                    .occlusion_instance_count
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        };

        let mut occlusion_enabled = false;
        let mut occlusion_status = OCCLUSION_STATUS_DISABLED;
        let mut hiz_view = self.fallback_view.clone();
        if globals.render_config.occlusion_culling {
            if !globals.render_config.gbuffer_pass {
                occlusion_status = OCCLUSION_STATUS_NO_GBUFFER;
            } else if let Some(hiz_id) = self.active_graph.as_ref().and_then(|g| g.hiz_id) {
                if let Some(view) = self.pool.texture_view(hiz_id) {
                    hiz_view = view.clone();
                    occlusion_enabled = true;
                    occlusion_status = OCCLUSION_STATUS_RAN;
                } else {
                    occlusion_status = OCCLUSION_STATUS_NO_HIZ;
                }
            } else {
                occlusion_status = OCCLUSION_STATUS_NO_HIZ;
            }
        }

        let lod0 = globals.render_config.gpu_lod0_distance.max(0.0);
        let lod1 = globals.render_config.gpu_lod1_distance.max(lod0);
        let lod2 = globals.render_config.gpu_lod2_distance.max(lod1);
        let mesh_tasks_enabled =
            globals.render_config.use_mesh_shaders && globals.device_caps.supports_mesh_pipeline();
        let current_signature = GpuCullSignature {
            frustum_culling: globals.render_config.frustum_culling,
            occlusion_culling: globals.render_config.occlusion_culling,
            lod: globals.render_config.lod,
            lod0,
            lod1,
            lod2,
            depth_bias: globals.render_config.gpu_cull_depth_bias.max(0.0),
            rect_pad: globals.render_config.gpu_cull_rect_pad.max(0.0),
            mesh_tasks_enabled,
        };
        let config_changed = self
            .last_gpu_cull_signature
            .map_or(true, |sig| sig != current_signature);
        let cull_interval = globals.render_config.cull_interval_frames;
        let interval_ok = cull_interval == 0
            || self.gpu_cull_last_frame == u32::MAX
            || self.frame_index.saturating_sub(self.gpu_cull_last_frame) >= cull_interval;
        let needs_cull = self.gpu_cull_dirty
            || config_changed
            || (!globals.occlusion_camera_stable && interval_ok);
        if !needs_cull {
            stats
                .occlusion_instance_count
                .store(instance_count, std::sync::atomic::Ordering::Relaxed);
            stats
                .occlusion_status
                .store(occlusion_status, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        let flags = (globals.render_config.frustum_culling as u32)
            | ((occlusion_enabled as u32) << 1)
            | ((globals.render_config.lod as u32) << 2);
        let mesh_count = (self.gpu_mesh_meta_len).min(u32::MAX as usize) as u32;
        let output_capacity = (self.gpu_total_capacity).min(u32::MAX as usize) as u32;
        let params = GpuCullParams {
            instance_count,
            draw_count: self.gpu_draw_count,
            mesh_count,
            flags,
            lod0_dist_sq: lod0 * lod0,
            lod1_dist_sq: lod1 * lod1,
            lod2_dist_sq: lod2 * lod2,
            alpha: globals.alpha,
            occlusion_depth_bias: globals.render_config.gpu_cull_depth_bias.max(0.0),
            occlusion_rect_pad: globals.render_config.gpu_cull_rect_pad.max(0.0),
            output_capacity,
            _pad1: 0,
        };
        let params_buffer = {
            let buf = self.buffer_cache.gpu_cull_params.ensure(
                &self.device,
                std::mem::size_of::<GpuCullParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&params));
            buf.clone()
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuCull/BG"),
            layout: &self.gpu_cull_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gbuffer_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shadow_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mesh_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gbuffer_indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: globals.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: mesh_task_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: gbuffer_mesh_tasks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: shadow_mesh_tasks.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuCull/Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuCull/Clear"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gpu_cull_clear_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (self.gpu_draw_count + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuCull/Instances"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gpu_cull_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (instance_count + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        if mesh_tasks_enabled {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuCull/MeshTasks"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gpu_mesh_tasks_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (self.gpu_draw_count + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.gpu_cull_dirty = false;
        self.gpu_cull_last_frame = self.frame_index;
        self.last_gpu_cull_signature = Some(current_signature);

        stats
            .occlusion_instance_count
            .store(instance_count, std::sync::atomic::Ordering::Relaxed);
        stats
            .occlusion_last_frame
            .store(self.frame_index, std::sync::atomic::Ordering::Relaxed);
        stats
            .occlusion_status
            .store(occlusion_status, std::sync::atomic::Ordering::Relaxed);
    }

    fn run_atmosphere_precompute(&mut self, globals: &FrameGlobals) {
        if !self.needs_atmosphere_precompute {
            return;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Atmosphere/Encoder"),
            });
        self.atmosphere_precomputer.precompute(
            &mut encoder,
            &self.queue,
            &globals.sky_uniforms,
            &globals.shader_constants,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        self.needs_atmosphere_precompute = false;
    }

    fn calculate_uniforms(
        &self,
        render_data: &RenderData,
        alpha: f32,
    ) -> (CameraUniforms, Vec<LightData>, Mat4, Mat4) {
        let camera = &render_data.camera_component;
        let prev_camera = render_data.previous_camera_transform;
        let cur_camera = render_data.current_camera_transform;
        let interp = alpha > 0.0 && alpha < 1.0;
        let eye = if interp {
            prev_camera.position.lerp(cur_camera.position, alpha)
        } else if alpha <= 0.0 {
            prev_camera.position
        } else {
            cur_camera.position
        };
        let rotation = if interp {
            Quat::from(prev_camera.rotation).slerp(cur_camera.rotation, alpha)
        } else if alpha <= 0.0 {
            Quat::from(prev_camera.rotation)
        } else {
            Quat::from(cur_camera.rotation)
        };
        let forward = rotation * Vec3::Z;
        let up = rotation * Vec3::Y;
        let view_matrix = Mat4::look_at_rh(eye, eye + forward, up);
        let projection_matrix = Mat4::perspective_infinite_reverse_rh(
            camera.fov_y_rad,
            camera.aspect_ratio,
            camera.near_plane,
        );

        let view_proj = projection_matrix * view_matrix;
        let inv_proj = projection_matrix.inverse();
        let inv_view_proj = view_proj.inverse();

        let camera_uniforms = CameraUniforms {
            view_matrix: view_matrix.to_cols_array_2d(),
            projection_matrix: projection_matrix.to_cols_array_2d(),
            inverse_projection_matrix: inv_proj.to_cols_array_2d(),
            inverse_view_projection_matrix: inv_view_proj.to_cols_array_2d(),
            view_position: eye.to_array(),
            light_count: render_data.lights.len() as u32,
            _pad_light: [0; 4],
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            frame_index: self.frame_index as u32,
            _pad_after_frame: [0; 3],
            _padding: [0; 3],
            _pad_after_padding: 0,
            _pad_end: [0; 4],
        };

        let light_data: Vec<LightData> = render_data
            .lights
            .iter()
            .map(|light| {
                let position = if interp {
                    light
                        .previous_transform
                        .position
                        .lerp(light.current_transform.position, alpha)
                } else if alpha <= 0.0 {
                    light.previous_transform.position
                } else {
                    light.current_transform.position
                };
                let rotation = if interp {
                    Quat::from(light.previous_transform.rotation)
                        .slerp(light.current_transform.rotation, alpha)
                } else if alpha <= 0.0 {
                    Quat::from(light.previous_transform.rotation)
                } else {
                    Quat::from(light.current_transform.rotation)
                };
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

        (camera_uniforms, light_data, view_proj, view_matrix)
    }

    fn build_sky_uniforms(&self, render_data: &RenderData) -> SkyUniforms {
        let directional_light = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional));
        let sky_cfg = &render_data.render_config.shader_constants;

        let (sun_dir, sun_color, sun_intensity) = if let Some(light) = directional_light {
            let rotation = Quat::from(light.current_transform.rotation);
            (
                (rotation * Vec3::Z).normalize_or_zero(),
                light.color,
                light.intensity,
            )
        } else {
            (Vec3::new(0.2, 0.8, 0.1).normalize(), [1.0, 1.0, 1.0], 100.0)
        };

        SkyUniforms {
            sun_direction: sun_dir.to_array(),
            _padding: 0.0,
            sun_color,
            sun_intensity,
            ground_albedo: sky_cfg.sky_ground_albedo,
            ground_brightness: sky_cfg.sky_ground_brightness,
            night_ambient_color: sky_cfg.night_ambient_color,
            sun_angular_radius_cos: sky_cfg.sun_angular_radius_cos,
        }
    }

    fn build_material_data(
        &mut self,
        scene: &RenderSceneState,
        force_rebuild: bool,
    ) -> MaterialBuildResult {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut textures: Vec<wgpu::TextureView> = Vec::new();
        textures.push(self.fallback_view.clone());
        let fallback_idx = 0i32;

        let limit = self
            .device
            .limits()
            .max_sampled_textures_per_shader_stage
            .max(1) as usize;

        // Keep texture selection deterministic and prefer the ones most referenced this frame
        // only when we risk exceeding the binding limit.
        let mut texture_use_counts: HashMap<usize, u32> = HashMap::new();
        let mut needed_materials: Vec<usize> = scene.active_material_ids().collect();
        needed_materials.sort_unstable();

        let mut mapping_changed = false;
        for mat_id in &needed_materials {
            match self.material_index_map.get(*mat_id) {
                Some(entry) if entry.version == self.material_version => {}
                _ => {
                    let index = (self.material_index_order.len() + 1) as u32;
                    self.material_index_order.push(*mat_id);
                    self.material_index_map.insert(
                        *mat_id,
                        MaterialIndexEntry {
                            version: self.material_version,
                            index,
                        },
                    );
                    mapping_changed = true;
                }
            }
        }

        for mat_id in &needed_materials {
            if let Some(entry) = self.materials.get(*mat_id) {
                let meta = &entry.meta;
                for tex in [
                    meta.albedo_texture_id,
                    meta.normal_texture_id,
                    meta.metallic_roughness_texture_id,
                    meta.emission_texture_id,
                ]
                .into_iter()
                .flatten()
                {
                    *texture_use_counts.entry(tex).or_insert(0) += 1;
                }
            }
        }

        let mut texture_ids: Vec<usize> = texture_use_counts.keys().copied().collect();
        let mut texture_indices: HashMap<usize, i32> = HashMap::new();

        if texture_ids.len() <= limit {
            // Simple deterministic ordering when we fit: ascending id.
            texture_ids.sort_unstable();
        } else {
            // Prioritize by usage when we would overflow the binding budget.
            texture_ids.sort_by(|a, b| {
                let ca = texture_use_counts.get(a).copied().unwrap_or(0);
                let cb = texture_use_counts.get(b).copied().unwrap_or(0);
                cb.cmp(&ca).then_with(|| a.cmp(b))
            });
        }
        let texture_overflow = texture_ids.len() > limit;

        let mut sig_hasher = DefaultHasher::new();
        limit.hash(&mut sig_hasher);
        texture_ids.len().hash(&mut sig_hasher);

        for tex_id in texture_ids {
            if textures.len() >= limit {
                break;
            }
            let rid = self
                .pool
                .asset_id_to_resource(ResourceKind::Texture, tex_id as u32);
            let view_opt = self.pool.texture_view(rid);
            let has_view = view_opt.is_some();
            tex_id.hash(&mut sig_hasher);
            has_view.hash(&mut sig_hasher);
            let view = view_opt
                .cloned()
                .unwrap_or_else(|| self.fallback_view.clone());
            self.pool.mark_used(rid, self.frame_index);
            let idx = textures.len() as i32;
            texture_indices.insert(tex_id, idx);
            textures.push(view);
        }

        let mut material_sig_hasher = DefaultHasher::new();
        sig_hasher.finish().hash(&mut material_sig_hasher);
        self.material_index_order
            .len()
            .hash(&mut material_sig_hasher);
        needed_materials.len().hash(&mut material_sig_hasher);
        for mat_id in &needed_materials {
            mat_id.hash(&mut material_sig_hasher);
            if let Some(entry) = self.materials.get(*mat_id) {
                let meta = &entry.meta;
                for value in meta.albedo {
                    value.to_bits().hash(&mut material_sig_hasher);
                }
                meta.metallic.to_bits().hash(&mut material_sig_hasher);
                meta.roughness.to_bits().hash(&mut material_sig_hasher);
                meta.ao.to_bits().hash(&mut material_sig_hasher);
                meta.emission_strength
                    .to_bits()
                    .hash(&mut material_sig_hasher);
                for value in meta.emission_color {
                    value.to_bits().hash(&mut material_sig_hasher);
                }
                meta.albedo_texture_id.hash(&mut material_sig_hasher);
                meta.normal_texture_id.hash(&mut material_sig_hasher);
                meta.metallic_roughness_texture_id
                    .hash(&mut material_sig_hasher);
                meta.emission_texture_id.hash(&mut material_sig_hasher);
            } else {
                0u8.hash(&mut material_sig_hasher);
            }
        }
        let signature = material_sig_hasher.finish();
        let signature_changed = signature != self.cached_material_signature
            || self.cached_texture_array_size == 0
            || mapping_changed;
        if !force_rebuild && !signature_changed {
            return MaterialBuildResult {
                buffer: self.cached_material_buffer.clone(),
                views: self.cached_texture_views.clone(),
                size: self.cached_texture_array_size.max(1),
                signature,
                changed: false,
                texture_overflow,
            };
        }

        let texture_array_size = textures.len().max(1) as u32;
        if force_rebuild && !signature_changed {
            self.material_bindings_version = self.material_bindings_version.wrapping_add(1);
            return MaterialBuildResult {
                buffer: self.cached_material_buffer.clone(),
                views: textures,
                size: texture_array_size,
                signature,
                changed: true,
                texture_overflow,
            };
        }

        self.material_bindings_version = self.material_bindings_version.wrapping_add(1);
        let fallback_material = MaterialShaderData {
            albedo: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ao: 1.0,
            emission_strength: 0.0,
            albedo_idx: -1,
            normal_idx: -1,
            metallic_roughness_idx: -1,
            emission_idx: -1,
            emission_color: [0.0, 0.0, 0.0],
            _padding: 0.0,
        };
        let material_capacity = self.material_index_order.len() + 1;
        let mut material_gpu = vec![fallback_material; material_capacity];

        for mat_id in &needed_materials {
            let meta = match self.materials.get(*mat_id) {
                Some(entry) => &entry.meta,
                None => continue,
            };

            let tex_idx = |opt_id: Option<usize>| -> i32 {
                match opt_id {
                    None => -1,
                    Some(id) => *texture_indices.get(&id).unwrap_or(&fallback_idx),
                }
            };

            let albedo_idx = tex_idx(meta.albedo_texture_id);
            let normal_idx = tex_idx(meta.normal_texture_id);
            let mra_idx = tex_idx(meta.metallic_roughness_texture_id);
            let emission_idx = tex_idx(meta.emission_texture_id);

            let shader_data = MaterialShaderData {
                albedo: meta.albedo,
                metallic: meta.metallic,
                roughness: meta.roughness,
                ao: meta.ao,
                emission_strength: meta.emission_strength,
                albedo_idx,
                normal_idx,
                metallic_roughness_idx: mra_idx,
                emission_idx,
                emission_color: meta.emission_color,
                _padding: 0.0,
            };

            if let Some(entry) = self.material_index_map.get(*mat_id) {
                let idx = entry.index as usize;
                if idx < material_gpu.len() {
                    material_gpu[idx] = shader_data;
                }
            }
        }

        let material_buffer = if material_gpu.is_empty() {
            None
        } else {
            let bytes = bytemuck::cast_slice(&material_gpu);
            let buf = self.buffer_cache.materials.ensure(
                &self.device,
                bytes.len() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue.write_buffer(buf, 0, bytes);
            Some(buf.clone())
        };

        MaterialBuildResult {
            buffer: material_buffer,
            views: textures,
            size: texture_array_size,
            signature,
            changed: true,
            texture_overflow,
        }
    }

    fn build_material_textures(&mut self, force_rebuild: bool) -> MaterialTextureBuildResult {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut sig_hasher = DefaultHasher::new();
        self.material_version.hash(&mut sig_hasher);
        self.material_index_order.len().hash(&mut sig_hasher);
        let material_ids: Vec<usize> = self.material_index_order.clone();

        for mat_id in &material_ids {
            mat_id.hash(&mut sig_hasher);
            let Some(entry) = self.materials.get(*mat_id) else {
                0u8.hash(&mut sig_hasher);
                continue;
            };
            let (
                albedo,
                metallic,
                roughness,
                ao,
                emission_strength,
                emission_color,
                albedo_id,
                normal_id,
                mra_id,
                emission_id,
            ) = {
                let meta = &entry.meta;
                (
                    meta.albedo,
                    meta.metallic,
                    meta.roughness,
                    meta.ao,
                    meta.emission_strength,
                    meta.emission_color,
                    meta.albedo_texture_id,
                    meta.normal_texture_id,
                    meta.metallic_roughness_texture_id,
                    meta.emission_texture_id,
                )
            };

            for value in albedo {
                value.to_bits().hash(&mut sig_hasher);
            }
            metallic.to_bits().hash(&mut sig_hasher);
            roughness.to_bits().hash(&mut sig_hasher);
            ao.to_bits().hash(&mut sig_hasher);
            emission_strength.to_bits().hash(&mut sig_hasher);
            for value in emission_color {
                value.to_bits().hash(&mut sig_hasher);
            }

            let texture_sig =
                |opt_id: Option<usize>, hasher: &mut DefaultHasher, pool: &mut GpuResourcePool| {
                    match opt_id {
                        Some(id) => {
                            id.hash(hasher);
                            let rid = pool.asset_id_to_resource(ResourceKind::Texture, id as u32);
                            pool.binding_version(rid).hash(hasher);
                        }
                        None => {
                            0u8.hash(hasher);
                        }
                    }
                };

            texture_sig(albedo_id, &mut sig_hasher, &mut self.pool);
            texture_sig(normal_id, &mut sig_hasher, &mut self.pool);
            texture_sig(mra_id, &mut sig_hasher, &mut self.pool);
            texture_sig(emission_id, &mut sig_hasher, &mut self.pool);
        }

        let signature = sig_hasher.finish();
        let signature_changed = signature != self.cached_material_textures_signature
            || self.cached_material_textures.is_none();
        if !force_rebuild && !signature_changed {
            return MaterialTextureBuildResult {
                textures: self
                    .cached_material_textures
                    .clone()
                    .unwrap_or_else(|| Arc::new(Vec::new())),
                signature,
                changed: false,
            };
        }

        let mut textures = Vec::with_capacity(self.material_index_order.len().saturating_add(1));
        let fallback = MaterialTextureSet {
            albedo: self.fallback_view.clone(),
            normal: self.fallback_view.clone(),
            metallic_roughness: self.fallback_view.clone(),
            emission: self.fallback_view.clone(),
        };
        textures.push(fallback);

        let mut resolve_tex = |opt_id: Option<usize>| -> wgpu::TextureView {
            let Some(id) = opt_id else {
                return self.fallback_view.clone();
            };
            let rid = self
                .pool
                .asset_id_to_resource(ResourceKind::Texture, id as u32);
            let view = self
                .pool
                .texture_view(rid)
                .cloned()
                .unwrap_or_else(|| self.fallback_view.clone());
            self.pool.mark_used(rid, self.frame_index);
            view
        };

        for mat_id in &material_ids {
            let Some(entry) = self.materials.get(*mat_id) else {
                textures.push(MaterialTextureSet {
                    albedo: self.fallback_view.clone(),
                    normal: self.fallback_view.clone(),
                    metallic_roughness: self.fallback_view.clone(),
                    emission: self.fallback_view.clone(),
                });
                continue;
            };
            let (albedo_id, normal_id, mra_id, emission_id) = {
                let meta = &entry.meta;
                (
                    meta.albedo_texture_id,
                    meta.normal_texture_id,
                    meta.metallic_roughness_texture_id,
                    meta.emission_texture_id,
                )
            };
            textures.push(MaterialTextureSet {
                albedo: resolve_tex(albedo_id),
                normal: resolve_tex(normal_id),
                metallic_roughness: resolve_tex(mra_id),
                emission: resolve_tex(emission_id),
            });
        }

        MaterialTextureBuildResult {
            textures: Arc::new(textures),
            signature,
            changed: true,
        }
    }

    fn build_gbuffer_instances(
        &mut self,
        render_data: &RenderData,
        alpha: f32,
        require_meshlets: bool,
    ) -> (
        Option<crate::graphics::passes::InstanceBuffer>,
        Arc<Vec<crate::graphics::passes::DrawBatch>>,
    ) {
        let pressure = self.streaming_pressure;
        let lod_bias = match pressure {
            MemoryPressure::Hard => self.streaming_tuning.lod_bias_hard,
            MemoryPressure::Soft => self.streaming_tuning.lod_bias_soft,
            MemoryPressure::None => 0,
        };
        let force_lowest =
            matches!(pressure, MemoryPressure::Hard) && self.streaming_tuning.force_lowest_lod_hard;
        let meshes = &self.meshes;
        let material_index_map = &self.material_index_map;
        let material_version = self.material_version;

        self.gbuffer_batcher.reset();
        let interp = alpha > 0.0 && alpha < 1.0;

        for object in &render_data.objects {
            let mat_idx = match material_index_map.get(object.material_id) {
                Some(entry) if entry.version == material_version => entry.index,
                _ => continue,
            };

            let mut desired_lod = object.lod_index.saturating_add(lod_bias);
            if force_lowest {
                desired_lod = usize::MAX;
            } else if let Some(mesh) = meshes.get(object.mesh_id) {
                let max_idx = mesh.lods.len().saturating_sub(1);
                desired_lod = desired_lod.min(max_idx);
            }
            let resolved =
                match self.resolve_draw_mesh(object.mesh_id, desired_lod, require_meshlets) {
                    Some(resolved) => resolved,
                    None => continue,
                };
            let batch_material_id = if self.use_material_batches() {
                mat_idx
            } else {
                0
            };
            let key = MeshLodMaterialKey {
                mesh_id: resolved.mesh_id,
                lod: resolved.lod_index,
                material_id: batch_material_id,
            };

            let position = if interp {
                object
                    .previous_transform
                    .position
                    .lerp(object.current_transform.position, alpha)
            } else if alpha <= 0.0 {
                object.previous_transform.position
            } else {
                object.current_transform.position
            };
            let rotation = if interp {
                Quat::from(object.previous_transform.rotation)
                    .slerp(object.current_transform.rotation, alpha)
            } else if alpha <= 0.0 {
                Quat::from(object.previous_transform.rotation)
            } else {
                Quat::from(object.current_transform.rotation)
            };
            let scale = if interp {
                object
                    .previous_transform
                    .scale
                    .lerp(object.current_transform.scale, alpha)
            } else if alpha <= 0.0 {
                object.previous_transform.scale
            } else {
                object.current_transform.scale
            };
            let model_matrix = Mat4::from_scale_rotation_translation(scale, rotation, position);
            let bounds_center = resolved.bounds_center;
            let bounds_extents = resolved.bounds_extents;

            let instance = crate::graphics::passes::gbuffer::GBufferInstanceRaw {
                model_matrix: model_matrix.to_cols_array_2d(),
                material_id: mat_idx,
                visibility: 1,
                _pad0: [0; 2],
                bounds_center: [bounds_center.x, bounds_center.y, bounds_center.z, 1.0],
                bounds_extents: [bounds_extents.x, bounds_extents.y, bounds_extents.z, 0.0],
            };

            self.gbuffer_batcher.push(key, instance);
        }

        let batches = self.gbuffer_batcher.active_batches();
        if batches.is_empty() {
            return (None, Arc::new(Vec::new()));
        }
        let total_instances: usize = batches.iter().map(|b| b.instances.len()).sum();
        if total_instances == 0 {
            return (None, Arc::new(Vec::new()));
        }

        let mut instance_data: Vec<crate::graphics::passes::gbuffer::GBufferInstanceRaw> =
            Vec::with_capacity(total_instances);
        let mut draw_batches = Vec::with_capacity(batches.len());

        for batch in batches {
            let offset = instance_data.len() as u32;
            instance_data.extend_from_slice(&batch.instances);
            let count = batch.instances.len() as u32;
            if count == 0 {
                continue;
            }
            let mesh = if batch.key.mesh_id == FALLBACK_MESH_KEY {
                self.fallback_mesh.as_ref()
            } else {
                meshes.get(batch.key.mesh_id)
            };
            let Some(mesh) = mesh else {
                continue;
            };
            if let Some(lod) = mesh.lods.get(batch.key.lod) {
                draw_batches.push(crate::graphics::passes::DrawBatch {
                    mesh_id: batch.key.mesh_id,
                    lod: batch.key.lod,
                    index_count: lod.index_count,
                    instance_range: offset..(offset + count),
                    material_id: batch.key.material_id,
                    vertex: mesh.vertex,
                    index: lod.buffer,
                    meshlet_descs: lod.meshlets.descs,
                    meshlet_vertices: lod.meshlets.vertices,
                    meshlet_indices: lod.meshlets.indices,
                    meshlet_count: lod.meshlets.count,
                });
            }
        }

        if instance_data.is_empty() {
            return (None, Arc::new(Vec::new()));
        }

        let instance_bytes = bytemuck::cast_slice(&instance_data);
        let instance_buffer = {
            let (buf, reallocated) = self.buffer_cache.gbuffer_instances.ensure_with_status(
                &self.device,
                instance_bytes.len() as u64,
                wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
                self.frame_index,
            );
            self.queue.write_buffer(buf, 0, instance_bytes);
            let buffer = buf.clone();
            if reallocated {
                self.note_bundle_resource_change();
            }
            buffer
        };

        (
            Some(crate::graphics::passes::InstanceBuffer {
                buffer: instance_buffer,
                count: instance_data.len() as u32,
                stride: std::mem::size_of::<crate::graphics::passes::gbuffer::GBufferInstanceRaw>()
                    as u64,
            }),
            Arc::new(draw_batches),
        )
    }

    fn build_shadow_data(
        &mut self,
        render_data: &RenderData,
        alpha: f32,
        view_matrix: Mat4,
        require_meshlets: bool,
    ) -> (
        Option<crate::graphics::passes::InstanceBuffer>,
        Arc<Vec<crate::graphics::passes::DrawBatch>>,
        Option<wgpu::Buffer>,
        Option<wgpu::Buffer>,
    ) {
        let pressure = self.streaming_pressure;
        let lod_bias = match pressure {
            MemoryPressure::Hard => self.streaming_tuning.lod_bias_hard,
            MemoryPressure::Soft => self.streaming_tuning.lod_bias_soft,
            MemoryPressure::None => 0,
        };
        let force_lowest =
            matches!(pressure, MemoryPressure::Hard) && self.streaming_tuning.force_lowest_lod_hard;
        let meshes = &self.meshes;

        self.shadow_batcher.reset();
        let interp = alpha > 0.0 && alpha < 1.0;

        let mut scene_bounds_min = Vec3::splat(f32::MAX);
        let mut scene_bounds_max = Vec3::splat(f32::MIN);
        let mut has_bounds = false;

        for object in &render_data.objects {
            if !object.casts_shadow {
                continue;
            }
            let mut desired_lod = object.lod_index.saturating_add(lod_bias);
            if force_lowest {
                desired_lod = usize::MAX;
            } else if let Some(mesh) = meshes.get(object.mesh_id) {
                let max_idx = mesh.lods.len().saturating_sub(1);
                desired_lod = desired_lod.min(max_idx);
            }
            let resolved =
                match self.resolve_draw_mesh(object.mesh_id, desired_lod, require_meshlets) {
                    Some(resolved) => resolved,
                    None => continue,
                };

            let position = if interp {
                object
                    .previous_transform
                    .position
                    .lerp(object.current_transform.position, alpha)
            } else if alpha <= 0.0 {
                object.previous_transform.position
            } else {
                object.current_transform.position
            };
            let rotation = if interp {
                Quat::from(object.previous_transform.rotation)
                    .slerp(object.current_transform.rotation, alpha)
            } else if alpha <= 0.0 {
                Quat::from(object.previous_transform.rotation)
            } else {
                Quat::from(object.current_transform.rotation)
            };
            let scale = if interp {
                object
                    .previous_transform
                    .scale
                    .lerp(object.current_transform.scale, alpha)
            } else if alpha <= 0.0 {
                object.previous_transform.scale
            } else {
                object.current_transform.scale
            };
            let model_matrix = Mat4::from_scale_rotation_translation(scale, rotation, position);

            let bounds_center = resolved.bounds_center;
            let bounds_extents = resolved.bounds_extents;
            let scale_abs = scale.abs();
            let world_center = position + rotation * (bounds_center * scale);
            let rot = Mat3::from_quat(rotation);
            let abs_rot = Mat3::from_cols(rot.x_axis.abs(), rot.y_axis.abs(), rot.z_axis.abs());
            let world_extents = abs_rot * (bounds_extents * scale_abs);
            let world_min = world_center - world_extents;
            let world_max = world_center + world_extents;
            scene_bounds_min = scene_bounds_min.min(world_min);
            scene_bounds_max = scene_bounds_max.max(world_max);
            has_bounds = true;

            let instance = InstanceRaw {
                model_matrix: model_matrix.to_cols_array_2d(),
            };

            let key = MeshLodKey {
                mesh_id: resolved.mesh_id,
                lod: resolved.lod_index,
            };
            self.shadow_batcher.push(key, instance);
        }

        let batches = self.shadow_batcher.active_batches();
        let total_instances: usize = batches.iter().map(|b| b.instances.len()).sum();

        let mut instance_data: Vec<InstanceRaw> = Vec::with_capacity(total_instances);
        let mut draw_batches = Vec::with_capacity(batches.len());

        for batch in batches {
            let offset = instance_data.len() as u32;
            instance_data.extend_from_slice(&batch.instances);
            let count = batch.instances.len() as u32;
            if count == 0 {
                continue;
            }
            let mesh = if batch.key.mesh_id == FALLBACK_MESH_KEY {
                self.fallback_mesh.as_ref()
            } else {
                meshes.get(batch.key.mesh_id)
            };
            let Some(mesh) = mesh else {
                continue;
            };
            if let Some(lod) = mesh.lods.get(batch.key.lod) {
                draw_batches.push(crate::graphics::passes::DrawBatch {
                    mesh_id: batch.key.mesh_id,
                    lod: batch.key.lod,
                    index_count: lod.index_count,
                    instance_range: offset..(offset + count),
                    material_id: 0,
                    vertex: mesh.vertex,
                    index: lod.buffer,
                    meshlet_descs: lod.meshlets.descs,
                    meshlet_vertices: lod.meshlets.vertices,
                    meshlet_indices: lod.meshlets.indices,
                    meshlet_count: lod.meshlets.count,
                });
            }
        }

        let instance_buffer = if instance_data.is_empty() {
            None
        } else {
            let bytes = bytemuck::cast_slice(&instance_data);
            let (buf, reallocated) = self.buffer_cache.shadow_instances.ensure_with_status(
                &self.device,
                bytes.len() as u64,
                wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
                self.frame_index,
            );
            self.queue.write_buffer(buf, 0, bytes);
            let buffer = buf.clone();
            if reallocated {
                self.note_bundle_resource_change();
            }
            Some(buffer)
        };

        let bounds = if !has_bounds {
            Aabb {
                min: Vec3::ZERO,
                max: Vec3::ONE,
            }
        } else {
            Aabb {
                min: scene_bounds_min,
                max: scene_bounds_max,
            }
        };
        self.shadow_bounds = Some(bounds);
        self.shadow_bounds_dirty = false;

        let (shadow_uniforms, matrices) =
            self.calculate_cascades(render_data, &view_matrix, &bounds);

        let shadow_uniforms_buffer = {
            let buf = self.buffer_cache.shadow_uniforms.ensure(
                &self.device,
                std::mem::size_of::<ShadowUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&shadow_uniforms));
            buf.clone()
        };

        let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>() as u64;
        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let aligned = wgpu::util::align_to(mat4_size, alignment);
        let total_size = aligned * matrices.len() as u64;
        let shadow_matrices_buffer = {
            let (buf, reallocated) = self.buffer_cache.shadow_matrices.ensure_with_status(
                &self.device,
                total_size,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            if reallocated {
                self.shadow_matrices_version = self.shadow_matrices_version.wrapping_add(1);
            }
            for (i, m) in matrices.iter().enumerate() {
                self.queue
                    .write_buffer(buf, i as u64 * aligned, bytemuck::bytes_of(m));
            }
            let buffer = buf.clone();
            if reallocated {
                self.note_bundle_resource_change();
            }
            buffer
        };

        (
            instance_buffer.map(|buffer| crate::graphics::passes::InstanceBuffer {
                buffer,
                count: instance_data.len() as u32,
                stride: std::mem::size_of::<InstanceRaw>() as u64,
            }),
            Arc::new(draw_batches),
            Some(shadow_uniforms_buffer),
            Some(shadow_matrices_buffer),
        )
    }

    fn shadow_bounds_for(&mut self, render_data: &RenderData, alpha: f32) -> Aabb {
        if self.shadow_bounds_dirty || self.shadow_bounds.is_none() {
            let bounds = self
                .compute_shadow_bounds(render_data, alpha)
                .unwrap_or(Aabb {
                    min: Vec3::ZERO,
                    max: Vec3::ONE,
                });
            self.shadow_bounds = Some(bounds);
            self.shadow_bounds_dirty = false;
            bounds
        } else {
            self.shadow_bounds.unwrap()
        }
    }

    fn compute_shadow_bounds(&self, render_data: &RenderData, alpha: f32) -> Option<Aabb> {
        let interp = alpha > 0.0 && alpha < 1.0;
        let mut scene_bounds_min = Vec3::splat(f32::MAX);
        let mut scene_bounds_max = Vec3::splat(f32::MIN);
        let mut has_bounds = false;

        for object in &render_data.objects {
            if !object.casts_shadow {
                continue;
            }
            let mesh = match self.meshes.get(object.mesh_id) {
                Some(mesh) => mesh,
                None => match self.fallback_mesh.as_ref() {
                    Some(mesh) => mesh,
                    None => continue,
                },
            };
            if mesh.lods.is_empty() {
                continue;
            }

            let position = if interp {
                object
                    .previous_transform
                    .position
                    .lerp(object.current_transform.position, alpha)
            } else if alpha <= 0.0 {
                object.previous_transform.position
            } else {
                object.current_transform.position
            };
            let rotation = if interp {
                Quat::from(object.previous_transform.rotation)
                    .slerp(object.current_transform.rotation, alpha)
            } else if alpha <= 0.0 {
                Quat::from(object.previous_transform.rotation)
            } else {
                Quat::from(object.current_transform.rotation)
            };
            let scale = if interp {
                object
                    .previous_transform
                    .scale
                    .lerp(object.current_transform.scale, alpha)
            } else if alpha <= 0.0 {
                object.previous_transform.scale
            } else {
                object.current_transform.scale
            };

            let bounds_center = mesh.bounds.center();
            let bounds_extents = mesh.bounds.extents();
            let scale_abs = scale.abs();
            let world_center = position + rotation * (bounds_center * scale);
            let rot = Mat3::from_quat(rotation);
            let abs_rot = Mat3::from_cols(rot.x_axis.abs(), rot.y_axis.abs(), rot.z_axis.abs());
            let world_extents = abs_rot * (bounds_extents * scale_abs);
            let world_min = world_center - world_extents;
            let world_max = world_center + world_extents;
            scene_bounds_min = scene_bounds_min.min(world_min);
            scene_bounds_max = scene_bounds_max.max(world_max);
            has_bounds = true;
        }

        if has_bounds {
            Some(Aabb {
                min: scene_bounds_min,
                max: scene_bounds_max,
            })
        } else {
            None
        }
    }

    fn build_shadow_uniform_buffers(
        &mut self,
        render_data: &RenderData,
        view_matrix: Mat4,
        alpha: f32,
    ) -> (Option<wgpu::Buffer>, Option<wgpu::Buffer>) {
        let bounds = self.shadow_bounds_for(render_data, alpha);
        let (shadow_uniforms, matrices) =
            self.calculate_cascades(render_data, &view_matrix, &bounds);

        let shadow_uniforms_buffer = {
            let buf = self.buffer_cache.shadow_uniforms.ensure(
                &self.device,
                std::mem::size_of::<ShadowUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&shadow_uniforms));
            Some(buf.clone())
        };

        let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>() as u64;
        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let aligned = wgpu::util::align_to(mat4_size, alignment);
        let total_size = aligned * matrices.len() as u64;
        let shadow_matrices_buffer = {
            let (buf, reallocated) = self.buffer_cache.shadow_matrices.ensure_with_status(
                &self.device,
                total_size,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                self.frame_index,
            );
            if reallocated {
                self.shadow_matrices_version = self.shadow_matrices_version.wrapping_add(1);
            }
            for (i, m) in matrices.iter().enumerate() {
                self.queue
                    .write_buffer(buf, i as u64 * aligned, bytemuck::bytes_of(m));
            }
            let buffer = buf.clone();
            if reallocated {
                self.note_bundle_resource_change();
            }
            Some(buffer)
        };

        (shadow_uniforms_buffer, shadow_matrices_buffer)
    }

    fn buffer_resident(&self, id: ResourceId) -> bool {
        self.pool
            .entry(id)
            .is_some_and(|entry| entry.residency == Residency::Resident && entry.buffer.is_some())
    }

    fn meshlet_resident(&self, lod: &MeshLodResource) -> bool {
        self.buffer_resident(lod.meshlets.descs)
            && self.buffer_resident(lod.meshlets.vertices)
            && self.buffer_resident(lod.meshlets.indices)
    }

    fn collect_resident_lods(&self, mesh: &MeshGpu, require_meshlets: bool) -> Vec<usize> {
        if mesh.lods.is_empty() || !self.buffer_resident(mesh.vertex) {
            return Vec::new();
        }
        let mut lods = Vec::with_capacity(mesh.lods.len());
        for (idx, lod) in mesh.lods.iter().enumerate() {
            if self.buffer_resident(lod.buffer) && (!require_meshlets || self.meshlet_resident(lod))
            {
                lods.push(idx);
            }
        }
        lods
    }

    fn select_resident_lod(
        &self,
        mesh: &MeshGpu,
        desired: usize,
        require_meshlets: bool,
    ) -> Option<usize> {
        if mesh.lods.is_empty() || !self.buffer_resident(mesh.vertex) {
            return None;
        }
        let max_idx = mesh.lods.len().saturating_sub(1);
        let mut idx = desired.min(max_idx);
        for candidate in idx..mesh.lods.len() {
            if self.buffer_resident(mesh.lods[candidate].buffer)
                && (!require_meshlets || self.meshlet_resident(&mesh.lods[candidate]))
            {
                return Some(candidate);
            }
        }
        while idx > 0 {
            idx -= 1;
            if self.buffer_resident(mesh.lods[idx].buffer)
                && (!require_meshlets || self.meshlet_resident(&mesh.lods[idx]))
            {
                return Some(idx);
            }
        }
        None
    }

    fn resolve_draw_mesh(
        &self,
        mesh_id: usize,
        desired_lod: usize,
        require_meshlets: bool,
    ) -> Option<ResolvedDrawMesh> {
        if let Some(mesh) = self.meshes.get(mesh_id) {
            if let Some(lod) = self.select_resident_lod(mesh, desired_lod, require_meshlets) {
                return Some(ResolvedDrawMesh {
                    mesh_id,
                    lod_index: lod,
                    bounds_center: mesh.bounds.center(),
                    bounds_extents: mesh.bounds.extents(),
                });
            }
        }
        if let Some(mesh) = self.fallback_mesh.as_ref() {
            if let Some(lod) = self.select_resident_lod(mesh, 0, require_meshlets) {
                return Some(ResolvedDrawMesh {
                    mesh_id: FALLBACK_MESH_KEY,
                    lod_index: lod,
                    bounds_center: mesh.bounds.center(),
                    bounds_extents: mesh.bounds.extents(),
                });
            }
        }
        None
    }

    fn material_index_for(&self, material_id: usize, material_version: u64) -> u32 {
        self.material_index_map
            .get(material_id)
            .filter(|entry| entry.version == material_version)
            .map(|entry| entry.index)
            .unwrap_or(0)
    }

    fn effective_binding_backend(&self) -> BindingBackendKind {
        if self.force_bindgroups_backend {
            BindingBackendKind::BindGroups
        } else {
            self.binding_backend_kind
        }
    }

    fn use_material_batches(&self) -> bool {
        self.effective_binding_backend() == BindingBackendKind::BindGroups
            || self.bindless_requires_uniform_indexing
    }

    fn gpu_mesh_index(&self, mesh_id: usize) -> u32 {
        self.gpu_mesh_indices
            .get(mesh_id)
            .copied()
            .unwrap_or(GPU_FALLBACK_MESH_INDEX)
    }

    fn gpu_mesh_material_index(&self, mesh_id: usize, material_id: u32) -> u32 {
        let key = (mesh_id, material_id);
        if let Some(index) = self.gpu_mesh_material_indices.get(&key) {
            return *index;
        }
        let fallback_key = (FALLBACK_MESH_KEY, material_id);
        if let Some(index) = self.gpu_mesh_material_indices.get(&fallback_key) {
            return *index;
        }
        u32::MAX
    }

    fn build_gpu_instance_input(
        &self,
        obj: &RenderObject,
        material_version: u64,
        use_prev_transform: bool,
    ) -> GpuInstanceInput {
        let curr = Mat4::from_scale_rotation_translation(
            obj.current_transform.scale,
            obj.current_transform.rotation,
            obj.current_transform.position,
        );
        let prev = if use_prev_transform {
            Mat4::from_scale_rotation_translation(
                obj.previous_transform.scale,
                obj.previous_transform.rotation,
                obj.previous_transform.position,
            )
        } else {
            curr
        };
        let material_id = self.material_index_for(obj.material_id, material_version);
        let (bounds_center, bounds_extents) = match self.meshes.get(obj.mesh_id) {
            Some(mesh) => (mesh.bounds.center(), mesh.bounds.extents()),
            None => match self.fallback_mesh.as_ref() {
                Some(mesh) => (mesh.bounds.center(), mesh.bounds.extents()),
                None => (Vec3::ZERO, Vec3::ZERO),
            },
        };
        let mesh_id = if self.use_material_batches() {
            self.gpu_mesh_material_index(obj.mesh_id, material_id)
        } else {
            self.gpu_mesh_index(obj.mesh_id)
        };

        GpuInstanceInput {
            prev_model: prev.to_cols_array_2d(),
            curr_model: curr.to_cols_array_2d(),
            material_id,
            mesh_id,
            casts_shadow: obj.casts_shadow as u32,
            _pad0: 0,
            bounds_center: [bounds_center.x, bounds_center.y, bounds_center.z, 1.0],
            bounds_extents: [bounds_extents.x, bounds_extents.y, bounds_extents.z, 0.0],
        }
    }

    fn write_gpu_instance_run(
        &self,
        buffer: &wgpu::Buffer,
        render_data: &RenderData,
        material_version: u64,
        use_prev_transform: bool,
        stride: u64,
        inputs: &mut Vec<GpuInstanceInput>,
        run_start: usize,
        run_end: usize,
    ) {
        if run_start > run_end || run_start >= render_data.objects.len() {
            return;
        }
        let end = run_end.min(render_data.objects.len().saturating_sub(1));
        let len = end - run_start + 1;
        inputs.clear();
        inputs.reserve(len);
        for obj in &render_data.objects[run_start..=end] {
            inputs.push(self.build_gpu_instance_input(obj, material_version, use_prev_transform));
        }
        let offset = run_start as u64 * stride;
        self.queue
            .write_buffer(buffer, offset, bytemuck::cast_slice(inputs));
    }

    fn rebuild_gpu_draw_data(
        &mut self,
        render_data: &RenderData,
    ) -> (
        Vec<GpuMeshMeta>,
        Vec<IndirectDrawBatch>,
        Vec<DrawIndexedIndirectArgs>,
        usize,
        Vec<MeshTaskMeta>,
        u32,
        Vec<u32>,
    ) {
        let require_meshlets =
            render_data.render_config.use_mesh_shaders && self.device_caps.supports_mesh_pipeline();
        let mesh_count = self.meshes.len().saturating_add(1);
        let mut mesh_counts = vec![0u32; mesh_count];
        let mut mesh_lods: Vec<Option<Vec<usize>>> = vec![None; self.meshes.len()];
        let mut mesh_indices = vec![GPU_FALLBACK_MESH_INDEX; self.meshes.len()];
        let fallback_lods = self
            .fallback_mesh
            .as_ref()
            .map(|mesh| self.collect_resident_lods(mesh, require_meshlets))
            .unwrap_or_default();
        let fallback_available = !fallback_lods.is_empty();

        for mesh_id in 0..self.meshes.len() {
            let Some(mesh) = self.meshes.get(mesh_id) else {
                continue;
            };
            let lods = self.collect_resident_lods(mesh, require_meshlets);
            let has_lods = !lods.is_empty();
            mesh_lods[mesh_id] = Some(lods);
            if has_lods && mesh_id < u32::MAX as usize {
                mesh_indices[mesh_id] = (mesh_id as u32).saturating_add(1);
            }
        }

        for obj in &render_data.objects {
            let mesh_index = mesh_indices
                .get(obj.mesh_id)
                .copied()
                .unwrap_or(GPU_FALLBACK_MESH_INDEX);
            if mesh_index == GPU_FALLBACK_MESH_INDEX && !fallback_available {
                continue;
            }
            if let Some(slot) = mesh_counts.get_mut(mesh_index as usize) {
                *slot = slot.saturating_add(1);
            }
        }

        let mut meta = vec![
            GpuMeshMeta {
                lod_count: 0,
                base_draw: 0,
                instance_capacity: 0,
                base_instance: 0,
            };
            mesh_count
        ];
        let mut draws = Vec::new();
        let mut indirect = Vec::new();
        let mut mesh_task_meta = Vec::new();
        let mut mesh_task_offset: u32 = 0;
        let mut draw_index: u32 = 0;
        let mut instance_base: u32 = 0;

        for mesh_index in 0..mesh_count {
            let count = mesh_counts[mesh_index];
            if count == 0 {
                continue;
            }
            let (mesh, lod_indices) = if mesh_index == GPU_FALLBACK_MESH_INDEX as usize {
                let Some(mesh) = self.fallback_mesh.as_ref() else {
                    continue;
                };
                if fallback_lods.is_empty() {
                    continue;
                }
                (mesh, fallback_lods.as_slice())
            } else {
                let mesh_id = mesh_index.saturating_sub(1);
                let Some(mesh) = self.meshes.get(mesh_id) else {
                    continue;
                };
                let Some(lods) = mesh_lods.get(mesh_id).and_then(|entry| entry.as_ref()) else {
                    continue;
                };
                if lods.is_empty() {
                    continue;
                }
                (mesh, lods.as_slice())
            };

            let lod_count = lod_indices.len() as u32;
            meta[mesh_index] = GpuMeshMeta {
                lod_count,
                base_draw: draw_index,
                instance_capacity: count,
                base_instance: instance_base,
            };
            for (draw_lod_idx, lod_idx) in lod_indices.iter().enumerate() {
                let lod = &mesh.lods[*lod_idx];
                let base_instance = instance_base + (draw_lod_idx as u32) * count;
                let meshlet_count = lod.meshlets.count;
                let tiling = mesh_task_tiling(&self.device_caps.limits, meshlet_count, count);
                let task_offset = mesh_task_offset;
                mesh_task_offset = mesh_task_offset.saturating_add(tiling.task_count);
                indirect.push(DrawIndexedIndirectArgs {
                    index_count: lod.index_count,
                    instance_count: 0,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: base_instance,
                });
                draws.push(IndirectDrawBatch {
                    mesh_id: mesh_index,
                    lod: draw_lod_idx,
                    material_id: 0,
                    vertex: mesh.vertex,
                    index: lod.buffer,
                    meshlet_descs: lod.meshlets.descs,
                    meshlet_vertices: lod.meshlets.vertices,
                    meshlet_indices: lod.meshlets.indices,
                    meshlet_count,
                    instance_base: base_instance,
                    instance_capacity: count,
                    indirect_offset: (draw_index as u64)
                        * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
                    mesh_task_offset: (task_offset as u64)
                        * std::mem::size_of::<DrawMeshTasksIndirectArgs>() as u64,
                    mesh_task_count: tiling.task_count,
                    mesh_task_tile_meshlets: tiling.tile_meshlets,
                    mesh_task_tile_instances: tiling.tile_instances,
                });
                mesh_task_meta.push(MeshTaskMeta {
                    meshlet_count,
                    instance_capacity: count,
                    tile_meshlets: tiling.tile_meshlets,
                    tile_instances: tiling.tile_instances,
                    task_offset,
                    _pad0: [0; 3],
                });
                draw_index = draw_index.wrapping_add(1);
            }
            instance_base = instance_base.saturating_add(count.saturating_mul(lod_count));
        }

        (
            meta,
            draws,
            indirect,
            instance_base as usize,
            mesh_task_meta,
            mesh_task_offset,
            mesh_indices,
        )
    }

    fn rebuild_gpu_draw_data_per_material(
        &mut self,
        render_data: &RenderData,
    ) -> (
        Vec<GpuMeshMeta>,
        Vec<IndirectDrawBatch>,
        Vec<DrawIndexedIndirectArgs>,
        usize,
        Vec<MeshTaskMeta>,
        u32,
        HashMap<(usize, u32), u32>,
    ) {
        let require_meshlets =
            render_data.render_config.use_mesh_shaders && self.device_caps.supports_mesh_pipeline();
        let mut mesh_lods: Vec<Option<Vec<usize>>> = vec![None; self.meshes.len()];
        let fallback_lods = self
            .fallback_mesh
            .as_ref()
            .map(|mesh| self.collect_resident_lods(mesh, require_meshlets))
            .unwrap_or_default();
        let fallback_available = !fallback_lods.is_empty();

        for mesh_id in 0..self.meshes.len() {
            let Some(mesh) = self.meshes.get(mesh_id) else {
                continue;
            };
            let lods = self.collect_resident_lods(mesh, require_meshlets);
            mesh_lods[mesh_id] = Some(lods);
        }

        let material_version = self.material_version;
        let mut mesh_material_counts: HashMap<(usize, u32), u32> = HashMap::new();
        for obj in &render_data.objects {
            let mesh_key = match mesh_lods.get(obj.mesh_id).and_then(|entry| entry.as_ref()) {
                Some(lods) if !lods.is_empty() => obj.mesh_id,
                _ if fallback_available => FALLBACK_MESH_KEY,
                _ => continue,
            };
            let material_id = self.material_index_for(obj.material_id, material_version);
            let count = mesh_material_counts
                .entry((mesh_key, material_id))
                .or_insert(0);
            *count = count.saturating_add(1);
        }

        let mut mesh_material_keys: Vec<(usize, u32)> =
            mesh_material_counts.keys().copied().collect();
        mesh_material_keys.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut meta = Vec::new();
        let mut draws = Vec::new();
        let mut indirect = Vec::new();
        let mut mesh_task_meta = Vec::new();
        let mut mesh_task_offset: u32 = 0;
        let mut draw_index: u32 = 0;
        let mut instance_base: u32 = 0;
        let mut mesh_material_indices: HashMap<(usize, u32), u32> =
            HashMap::with_capacity(mesh_material_keys.len());

        for (mesh_key, material_id) in mesh_material_keys {
            let count = match mesh_material_counts.get(&(mesh_key, material_id)) {
                Some(count) if *count > 0 => *count,
                _ => continue,
            };

            let (mesh, lod_indices) = if mesh_key == FALLBACK_MESH_KEY {
                let Some(mesh) = self.fallback_mesh.as_ref() else {
                    continue;
                };
                if fallback_lods.is_empty() {
                    continue;
                }
                (mesh, fallback_lods.as_slice())
            } else {
                let Some(mesh) = self.meshes.get(mesh_key) else {
                    continue;
                };
                let Some(lods) = mesh_lods.get(mesh_key).and_then(|entry| entry.as_ref()) else {
                    continue;
                };
                if lods.is_empty() {
                    continue;
                }
                (mesh, lods.as_slice())
            };

            let lod_count = lod_indices.len() as u32;
            let meta_index = meta.len() as u32;
            mesh_material_indices.insert((mesh_key, material_id), meta_index);
            meta.push(GpuMeshMeta {
                lod_count,
                base_draw: draw_index,
                instance_capacity: count,
                base_instance: instance_base,
            });

            for (draw_lod_idx, lod_idx) in lod_indices.iter().enumerate() {
                let lod = &mesh.lods[*lod_idx];
                let base_instance = instance_base + (draw_lod_idx as u32) * count;
                let meshlet_count = lod.meshlets.count;
                let tiling = mesh_task_tiling(&self.device_caps.limits, meshlet_count, count);
                let task_offset = mesh_task_offset;
                mesh_task_offset = mesh_task_offset.saturating_add(tiling.task_count);
                indirect.push(DrawIndexedIndirectArgs {
                    index_count: lod.index_count,
                    instance_count: 0,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: base_instance,
                });
                draws.push(IndirectDrawBatch {
                    mesh_id: mesh_key,
                    lod: draw_lod_idx,
                    material_id,
                    vertex: mesh.vertex,
                    index: lod.buffer,
                    meshlet_descs: lod.meshlets.descs,
                    meshlet_vertices: lod.meshlets.vertices,
                    meshlet_indices: lod.meshlets.indices,
                    meshlet_count,
                    instance_base: base_instance,
                    instance_capacity: count,
                    indirect_offset: (draw_index as u64)
                        * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
                    mesh_task_offset: (task_offset as u64)
                        * std::mem::size_of::<DrawMeshTasksIndirectArgs>() as u64,
                    mesh_task_count: tiling.task_count,
                    mesh_task_tile_meshlets: tiling.tile_meshlets,
                    mesh_task_tile_instances: tiling.tile_instances,
                });
                mesh_task_meta.push(MeshTaskMeta {
                    meshlet_count,
                    instance_capacity: count,
                    tile_meshlets: tiling.tile_meshlets,
                    tile_instances: tiling.tile_instances,
                    task_offset,
                    _pad0: [0; 3],
                });
                draw_index = draw_index.wrapping_add(1);
            }
            instance_base = instance_base.saturating_add(count.saturating_mul(lod_count));
        }

        (
            meta,
            draws,
            indirect,
            instance_base as usize,
            mesh_task_meta,
            mesh_task_offset,
            mesh_material_indices,
        )
    }

    fn rebuild_gpu_draw_buffers(&mut self, render_data: &RenderData) {
        let use_material_batches = self.use_material_batches();
        let (
            meta,
            draws,
            indirect,
            total_capacity,
            mesh_task_meta,
            mesh_task_total,
            mapping_changed,
        ) = if use_material_batches {
            let (
                meta,
                draws,
                indirect,
                total_capacity,
                mesh_task_meta,
                mesh_task_total,
                mesh_material_indices,
            ) = self.rebuild_gpu_draw_data_per_material(render_data);
            let mapping_changed = mesh_material_indices != self.gpu_mesh_material_indices;
            self.gpu_mesh_material_indices = mesh_material_indices;
            self.gpu_mesh_indices.clear();
            (
                meta,
                draws,
                indirect,
                total_capacity,
                mesh_task_meta,
                mesh_task_total,
                mapping_changed,
            )
        } else {
            let (
                meta,
                draws,
                indirect,
                total_capacity,
                mesh_task_meta,
                mesh_task_total,
                mesh_indices,
            ) = self.rebuild_gpu_draw_data(render_data);
            let mapping_changed = mesh_indices != self.gpu_mesh_indices;
            self.gpu_mesh_indices = mesh_indices;
            self.gpu_mesh_material_indices.clear();
            (
                meta,
                draws,
                indirect,
                total_capacity,
                mesh_task_meta,
                mesh_task_total,
                mapping_changed,
            )
        };
        self.gpu_draws = Arc::new(draws);
        self.gpu_draw_count = indirect.len() as u32;
        self.gpu_mesh_meta_len = meta.len();
        self.gpu_total_capacity = total_capacity;
        self.gpu_draws_dirty = false;
        self.gpu_bundle_version = self.gpu_bundle_version.wrapping_add(1);
        if mapping_changed {
            self.gpu_instances_dirty = true;
            self.gpu_instance_updates.clear();
        }
        self.gpu_cull_dirty = true;

        if !meta.is_empty() {
            let needed = meta.len();
            if self.gpu_mesh_meta_capacity < needed || self.gpu_mesh_meta_buffer.is_none() {
                let capacity = needed.max(1).next_power_of_two();
                let size = capacity * std::mem::size_of::<GpuMeshMeta>();
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuDriven/MeshMeta"),
                    size: size as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.gpu_mesh_meta_capacity = capacity;
                self.gpu_mesh_meta_buffer = Some(buffer);
            }
            if let Some(buffer) = self.gpu_mesh_meta_buffer.as_ref() {
                self.queue
                    .write_buffer(buffer, 0, bytemuck::cast_slice(&meta));
            }
        } else {
            self.gpu_mesh_meta_buffer = None;
            self.gpu_mesh_meta_capacity = 0;
        }

        if self.gpu_draw_count > 0 {
            let draw_needed = self.gpu_draw_count as usize;
            let task_needed = mesh_task_total as usize;
            if self.gpu_indirect_capacity < draw_needed
                || self.gpu_indirect_gbuffer.is_none()
                || self.gpu_indirect_shadow.is_none()
            {
                let capacity = draw_needed.max(1).next_power_of_two();
                let indirect_size = capacity * std::mem::size_of::<DrawIndexedIndirectArgs>();
                let gbuffer_indirect = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuDriven/GBufferIndirect"),
                    size: indirect_size as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let shadow_indirect = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuDriven/ShadowIndirect"),
                    size: indirect_size as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.gpu_indirect_capacity = capacity;
                self.gpu_indirect_gbuffer = Some(gbuffer_indirect);
                self.gpu_indirect_shadow = Some(shadow_indirect);
            }
            if let (Some(gbuffer_indirect), Some(shadow_indirect)) = (
                self.gpu_indirect_gbuffer.as_ref(),
                self.gpu_indirect_shadow.as_ref(),
            ) {
                self.queue
                    .write_buffer(gbuffer_indirect, 0, bytemuck::cast_slice(&indirect));
                self.queue
                    .write_buffer(shadow_indirect, 0, bytemuck::cast_slice(&indirect));
            }

            if self.gpu_mesh_tasks_capacity < task_needed
                || self.gpu_mesh_tasks_gbuffer.is_none()
                || self.gpu_mesh_tasks_shadow.is_none()
            {
                let capacity = task_needed.max(1).next_power_of_two();
                let task_size = capacity * std::mem::size_of::<DrawMeshTasksIndirectArgs>();
                let gbuffer_tasks = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuDriven/GBufferMeshTasks"),
                    size: task_size as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let shadow_tasks = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuDriven/ShadowMeshTasks"),
                    size: task_size as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.gpu_mesh_tasks_capacity = capacity;
                self.gpu_mesh_tasks_gbuffer = Some(gbuffer_tasks);
                self.gpu_mesh_tasks_shadow = Some(shadow_tasks);
            }

            if self.gpu_mesh_task_meta_capacity < draw_needed
                || self.gpu_mesh_task_meta_buffer.is_none()
            {
                let capacity = draw_needed.max(1).next_power_of_two();
                let size = capacity * std::mem::size_of::<MeshTaskMeta>();
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuDriven/MeshTaskMeta"),
                    size: size as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.gpu_mesh_task_meta_capacity = capacity;
                self.gpu_mesh_task_meta_buffer = Some(buffer);
            }
            if let Some(buffer) = self.gpu_mesh_task_meta_buffer.as_ref() {
                self.queue
                    .write_buffer(buffer, 0, bytemuck::cast_slice(&mesh_task_meta));
            }
        } else {
            self.gpu_indirect_gbuffer = None;
            self.gpu_indirect_shadow = None;
            self.gpu_indirect_capacity = 0;
            self.gpu_mesh_tasks_gbuffer = None;
            self.gpu_mesh_tasks_shadow = None;
            self.gpu_mesh_tasks_capacity = 0;
            self.gpu_mesh_task_meta_buffer = None;
            self.gpu_mesh_task_meta_capacity = 0;
        }
    }

    fn set_gpu_driven_stats(
        &self,
        draw_count: u32,
        mesh_count: usize,
        instance_capacity: usize,
        visible_capacity: usize,
        shadow_capacity: usize,
        total_capacity: usize,
    ) {
        let stats = &self.shared_stats;
        stats
            .gpu_draw_count
            .store(draw_count, std::sync::atomic::Ordering::Relaxed);
        stats.gpu_mesh_count.store(
            mesh_count.min(u32::MAX as usize) as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        stats.gpu_instance_capacity.store(
            instance_capacity.min(u32::MAX as usize) as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        stats.gpu_visible_capacity.store(
            visible_capacity.min(u32::MAX as usize) as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        stats.gpu_shadow_capacity.store(
            shadow_capacity.min(u32::MAX as usize) as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        stats
            .gpu_total_capacity
            .store(total_capacity as u64, std::sync::atomic::Ordering::Relaxed);
    }

    fn prepare_gpu_driven_frame(
        &mut self,
        render_data: &RenderData,
        _alpha: f32,
    ) -> Option<GpuDrivenFrame> {
        if render_data.objects.is_empty() {
            self.gpu_instance_updates.clear();
            self.set_gpu_driven_stats(0, 0, 0, 0, 0, 0);
            return None;
        }

        let instance_count = render_data.objects.len();
        if self.gpu_draws_dirty {
            self.rebuild_gpu_draw_buffers(render_data);
        }

        let material_version = self.material_version;
        let stride = std::mem::size_of::<GpuInstanceInput>() as u64;
        let use_prev_transform = _alpha > 0.0 && _alpha < 1.0;

        if self.gpu_instance_updates.len() > 1 {
            self.gpu_instance_updates.sort_unstable();
            self.gpu_instance_updates.dedup();
        }
        let update_count = self.gpu_instance_updates.len();

        let rebuild_instances = self.gpu_instances_dirty
            || self.gpu_instance_capacity < instance_count
            || update_count >= instance_count;
        let mut instance_updated = false;
        if rebuild_instances {
            let capacity = instance_count.max(1).next_power_of_two();
            self.gpu_instance_capacity = capacity;
            let buffer_size = (capacity as u64) * stride;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuDriven/Instances"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut inputs = Vec::with_capacity(instance_count);
            for obj in &render_data.objects {
                inputs.push(self.build_gpu_instance_input(
                    obj,
                    material_version,
                    use_prev_transform,
                ));
            }
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&inputs));
            self.gpu_instance_buffer = Some(buffer);
            self.gpu_instances_dirty = false;
            self.gpu_instance_updates.clear();
            instance_updated = true;
        } else if update_count > 0 {
            let buffer = self
                .gpu_instance_buffer
                .as_ref()
                .expect("gpu instance buffer missing");
            let mut run_start: Option<usize> = None;
            let mut run_end: usize = 0;
            let mut inputs: Vec<GpuInstanceInput> = Vec::new();

            for &idx in &self.gpu_instance_updates {
                if idx >= instance_count {
                    continue;
                }
                match run_start {
                    None => {
                        run_start = Some(idx);
                        run_end = idx;
                    }
                    Some(start) => {
                        if idx == run_end + 1 {
                            run_end = idx;
                        } else {
                            self.write_gpu_instance_run(
                                buffer,
                                render_data,
                                material_version,
                                use_prev_transform,
                                stride,
                                &mut inputs,
                                start,
                                run_end,
                            );
                            run_start = Some(idx);
                            run_end = idx;
                        }
                    }
                }
            }
            if let Some(start) = run_start {
                self.write_gpu_instance_run(
                    buffer,
                    render_data,
                    material_version,
                    use_prev_transform,
                    stride,
                    &mut inputs,
                    start,
                    run_end,
                );
            }
            self.gpu_instance_updates.clear();
            instance_updated = true;
        }
        if instance_updated {
            self.gpu_cull_dirty = true;
        }

        if self.gpu_draw_count == 0 && !render_data.objects.is_empty() {
            self.rebuild_gpu_draw_buffers(render_data);
        }

        if self.gpu_total_capacity == 0 || self.gpu_draw_count == 0 {
            self.set_gpu_driven_stats(
                self.gpu_draw_count,
                self.gpu_mesh_meta_len,
                self.gpu_instance_capacity,
                self.gpu_visible_capacity,
                self.gpu_shadow_capacity,
                self.gpu_total_capacity,
            );
            return None;
        }

        let mut bundle_buffers_reallocated = false;
        if self.gpu_visible_capacity < self.gpu_total_capacity {
            let capacity = self.gpu_total_capacity.max(1).next_power_of_two();
            self.gpu_visible_capacity = capacity;
            let size = (capacity
                * std::mem::size_of::<crate::graphics::passes::gbuffer::GBufferInstanceRaw>())
                as u64;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuDriven/GBufferInstances"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.gpu_visible_buffer = Some(buffer);
            bundle_buffers_reallocated = true;
        }
        if self.gpu_shadow_capacity < self.gpu_total_capacity {
            let capacity = self.gpu_total_capacity.max(1).next_power_of_two();
            self.gpu_shadow_capacity = capacity;
            let size = (capacity * std::mem::size_of::<InstanceRaw>()) as u64;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuDriven/ShadowInstances"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.gpu_shadow_buffer = Some(buffer);
            bundle_buffers_reallocated = true;
        }
        if bundle_buffers_reallocated {
            self.gpu_bundle_version = self.gpu_bundle_version.wrapping_add(1);
        }

        let gbuffer_instances = self.gpu_visible_buffer.as_ref().map(|buffer| {
            crate::graphics::passes::InstanceBuffer {
                buffer: buffer.clone(),
                count: instance_count as u32,
                stride: std::mem::size_of::<crate::graphics::passes::gbuffer::GBufferInstanceRaw>()
                    as u64,
            }
        });
        let shadow_instances =
            self.gpu_shadow_buffer
                .as_ref()
                .map(|buffer| crate::graphics::passes::InstanceBuffer {
                    buffer: buffer.clone(),
                    count: instance_count as u32,
                    stride: std::mem::size_of::<InstanceRaw>() as u64,
                });

        self.set_gpu_driven_stats(
            self.gpu_draw_count,
            self.gpu_mesh_meta_len,
            self.gpu_instance_capacity,
            self.gpu_visible_capacity,
            self.gpu_shadow_capacity,
            self.gpu_total_capacity,
        );

        Some(GpuDrivenFrame {
            gbuffer_instances,
            shadow_instances,
            gbuffer_indirect: self.gpu_indirect_gbuffer.clone(),
            shadow_indirect: self.gpu_indirect_shadow.clone(),
            gbuffer_mesh_tasks: self.gpu_mesh_tasks_gbuffer.clone(),
            shadow_mesh_tasks: self.gpu_mesh_tasks_shadow.clone(),
            draws: self.gpu_draws.clone(),
        })
    }

    fn calculate_cascades(
        &self,
        render_data: &RenderData,
        camera_view: &Mat4,
        scene_bounds: &Aabb,
    ) -> (ShadowUniforms, Vec<[[f32; 4]; 4]>) {
        let light_dir = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional))
            .map(|l| {
                let rotation = Quat::from(l.current_transform.rotation);
                (rotation * -Vec3::Z).normalize_or_zero()
            })
            .unwrap_or_else(|| Vec3::new(0.0, -1.0, 0.0));
        let inv_camera_view = camera_view.inverse();
        let tan_half_fovy = (render_data.camera_component.fov_y_rad / 2.0).tan();
        let scene_corners = scene_bounds.get_corners();

        const MAX_CASCADES: usize = crate::graphics::renderer_common::common::MAX_SHADOW_CASCADES;
        let cascade_count = render_data
            .render_config
            .shadow_cascade_count
            .clamp(1, MAX_CASCADES as u32) as usize;
        let mut splits = render_data.render_config.shadow_cascade_splits;
        splits[0] = splits[0].max(render_data.camera_component.near_plane.max(0.0));
        for idx in 1..=cascade_count {
            if splits[idx] < splits[idx - 1] + f32::EPSILON {
                splits[idx] = splits[idx - 1] + f32::EPSILON;
            }
        }
        let shadow_resolution = render_data.render_config.shadow_map_resolution.max(1) as f32;

        let mut uniforms = ShadowUniforms {
            cascade_count: cascade_count as u32,
            _pad0: [0; 3],
            _pad1: [0; 4],
            cascades: [CascadeUniform::default(); MAX_CASCADES],
        };
        let mut matrices = Vec::with_capacity(cascade_count);

        for i in 0..cascade_count {
            let z_near = splits[i];
            let z_far = splits[i + 1];

            let h_near = 2.0 * tan_half_fovy * z_near;
            let w_near = h_near * render_data.camera_component.aspect_ratio;
            let h_far = 2.0 * tan_half_fovy * z_far;
            let w_far = h_far * render_data.camera_component.aspect_ratio;

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
                std::array::from_fn(|idx| (inv_camera_view * corners_view[idx].extend(1.0)).xyz());

            let world_center = frustum_corners_world.iter().sum::<Vec3>() / 8.0;
            let light_view = Mat4::look_at_rh(world_center - light_dir, world_center, Vec3::Y);

            // Match monolithic behavior: XY from frustum slice, Z from scene bounds.
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

            // Texel-align without shrinking the extents to avoid flicker and clipping.
            let extent = cascade_max - cascade_min;
            let texel_x = (extent.x / shadow_resolution).max(f32::EPSILON);
            let texel_y = (extent.y / shadow_resolution).max(f32::EPSILON);
            cascade_min.x = (cascade_min.x / texel_x).floor() * texel_x;
            cascade_min.y = (cascade_min.y / texel_y).floor() * texel_y;
            cascade_max.x = (cascade_max.x / texel_x).ceil() * texel_x;
            cascade_max.y = (cascade_max.y / texel_y).ceil() * texel_y;

            // Depth padding to cover dynamic bounds; mirrors monolithic behavior with safety margin.
            let depth_padding = (scene_max_z - scene_min_z).abs().max(4.0);

            let light_proj = Mat4::orthographic_rh(
                cascade_min.x,
                cascade_max.x,
                cascade_min.y,
                cascade_max.y,
                -(scene_max_z + depth_padding),
                -(scene_min_z - depth_padding),
            );

            let final_light_vp = light_proj * light_view;
            uniforms.cascades[i] = CascadeUniform {
                light_view_proj: final_light_vp.to_cols_array_2d(),
                split_depth: [-z_far, 0.0, 0.0, 0.0],
            };
            matrices.push(final_light_vp.to_cols_array_2d());
        }

        (uniforms, matrices)
    }
}

#[derive(Clone)]
struct StreamRequest {
    resource: ResourceId,
    priority: f32,
    kind: AssetStreamKind,
    max_lod: Option<usize>,
    asset_id: usize,
    force_low_res: bool,
    critical: bool,
}

struct StreamRequestScratch {
    entries: Vec<Option<StreamRequest>>,
    touched: Vec<usize>,
}

impl StreamRequestScratch {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            touched: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for id in self.touched.drain(..) {
            if let Some(slot) = self.entries.get_mut(id) {
                *slot = None;
            }
        }
    }

    fn upsert(
        &mut self,
        asset_id: usize,
        resource: ResourceId,
        kind: AssetStreamKind,
        max_lod: Option<usize>,
        priority: f32,
        force_low_res: bool,
        critical: bool,
    ) {
        if asset_id >= self.entries.len() {
            self.entries.resize_with(asset_id + 1, || None);
        }
        match self.entries[asset_id].as_mut() {
            Some(existing) => {
                existing.priority = existing.priority.max(priority);
                if let Some(lod) = max_lod {
                    existing.max_lod = existing
                        .max_lod
                        .map(|current| current.min(lod))
                        .or(Some(lod));
                }
                existing.force_low_res |= force_low_res;
                existing.critical |= critical;
            }
            None => {
                self.entries[asset_id] = Some(StreamRequest {
                    resource,
                    priority,
                    kind,
                    max_lod,
                    asset_id,
                    force_low_res,
                    critical,
                });
                self.touched.push(asset_id);
            }
        }
    }

    fn drain_into(&mut self, out: &mut Vec<StreamRequest>) {
        out.clear();
        out.reserve(self.touched.len());
        for id in self.touched.drain(..) {
            if let Some(slot) = self.entries.get_mut(id) {
                if let Some(req) = slot.take() {
                    out.push(req);
                }
            }
        }
    }
}

#[derive(Clone)]
struct StreamingPlan {
    requests: Vec<StreamRequest>,
    priority_lookup: HashMap<ResourceId, f32>,
    pressure: MemoryPressure,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MemoryPressure {
    None,
    Soft,
    Hard,
}
