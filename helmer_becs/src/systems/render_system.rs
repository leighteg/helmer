use bevy_ecs::{
    prelude::{
        Added, Changed, DetectChanges, Entity, Or, Query, Ref, RemovedComponents, Res, ResMut,
        Resource, With,
    },
    system::{Local, ParamSet, SystemParam},
};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use glam::{Mat4, Quat, Vec2, Vec3};
use hashbrown::{HashMap, HashSet, hash_map::Entry};
use helmer::{
    graphics::{
        common::{
            config::RenderConfig,
            graph::RenderGraphSpec,
            renderer::{
                Aabb, AssetStreamKind, AssetStreamingRequest, GizmoData, RenderCameraDelta,
                RenderDelta, RenderLightDelta, RenderObjectDelta, StreamingTuning,
            },
        },
        render_graphs::default_graph_spec,
    },
    provided::components::{Camera, Light, MeshRenderer, Transform},
    runtime::{asset_server::MeshAabbMap, runtime::RuntimeProfiling},
};
use parking_lot::RwLock;
use std::{cmp::Ordering, collections::VecDeque, sync::Arc, thread, time::Instant};
use tracing::warn;

use crate::{
    BevyActiveCamera, BevyAssetServer, BevyCamera, BevyLight, BevyLodTuning, BevyMeshRenderer,
    BevyRenderWorkerTuning, BevyRuntimeConfig, BevyRuntimeProfiling, BevyStreamingTuning,
    BevyTransform,
};

//================================================================================
// Bevy Wrapper & Type Aliases
//================================================================================

// Resource Wrappers
#[derive(Resource, Default)]
pub struct RenderPacket(pub Option<RenderDelta>);

#[derive(Resource, Default)]
pub struct RenderObjectCount(pub usize);

#[derive(Resource, Clone)]
pub struct RenderGraphResource(pub RenderGraphSpec);

impl Default for RenderGraphResource {
    fn default() -> Self {
        Self(default_graph_spec())
    }
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct RenderGizmoState(pub GizmoData);

impl Default for RenderGizmoState {
    fn default() -> Self {
        Self(GizmoData::default())
    }
}

//================================================================================
// Frustum Culling Logic
//================================================================================

/// A geometric frustum defined by 6 planes, used for culling.
pub struct Frustum {
    view_proj: Mat4,
}

impl Frustum {
    /// Creates a frustum from the camera transform and projection parameters.
    pub fn from_camera(
        transform: &Transform,
        fov_y: f32,
        aspect: f32,
        near_plane: f32,
        _far_plane: f32,
    ) -> Self {
        let eye = transform.position;
        let forward = transform.forward().normalize_or_zero();
        let up = transform.up().normalize_or_zero();
        let view = Mat4::look_at_rh(eye, eye + forward, up);
        let projection = Mat4::perspective_infinite_reverse_rh(
            fov_y.max(f32::EPSILON),
            aspect.max(f32::EPSILON),
            near_plane.max(f32::EPSILON),
        );
        let view_proj = projection * view;

        Self { view_proj }
    }

    /// Checks if a transformed AABB intersects the frustum.
    pub fn intersects_aabb(&self, aabb: &Aabb, transform: &Transform) -> bool {
        let center_world =
            transform.position + transform.rotation * (aabb.center() * transform.scale);
        let extents = aabb.extents();
        let scale_abs = transform.scale.abs();
        let axis_x = transform.rotation * (Vec3::X * (extents.x * scale_abs.x));
        let axis_y = transform.rotation * (Vec3::Y * (extents.y * scale_abs.y));
        let axis_z = transform.rotation * (Vec3::Z * (extents.z * scale_abs.z));
        self.intersects_obb(center_world, axis_x, axis_y, axis_z)
    }

    /// Checks if an oriented bounding box intersects the frustum.
    pub fn intersects_obb(
        &self,
        center_world: Vec3,
        axis_x: Vec3,
        axis_y: Vec3,
        axis_z: Vec3,
    ) -> bool {
        let mut min_ndc = Vec2::splat(1.0);
        let mut max_ndc = Vec2::splat(-1.0);
        let mut valid = false;
        let signs = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];

        for sign in signs {
            let corner = center_world + axis_x * sign.x + axis_y * sign.y + axis_z * sign.z;
            let clip = self.view_proj * corner.extend(1.0);
            if clip.w > 0.0 {
                let ndc = clip.truncate() / clip.w;
                let ndc_xy = Vec2::new(ndc.x, ndc.y);
                min_ndc = min_ndc.min(ndc_xy);
                max_ndc = max_ndc.max(ndc_xy);
                valid = true;
            }
        }

        if !valid {
            return false;
        }
        if max_ndc.x < -1.0 || min_ndc.x > 1.0 {
            return false;
        }
        if max_ndc.y < -1.0 || min_ndc.y > 1.0 {
            return false;
        }
        true
    }
}

//================================================================================
// Helper Structs
//================================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodTuning {
    pub lod0_distance: f32,
    pub lod1_distance: f32,
    pub lod2_distance: f32,
    pub hysteresis: f32,
    pub smoothing: f32,
    pub min_change_frames: u32,
}

impl Default for LodTuning {
    fn default() -> Self {
        Self {
            lod0_distance: 25.0,
            lod1_distance: 60.0,
            lod2_distance: 120.0,
            hysteresis: 0.1,
            smoothing: 0.25,
            min_change_frames: 6,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct LodState {
    smoothed_distance_sq: f32,
    lod_index: usize,
    last_change_frame: u32,
    last_seen_frame: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderWorkerTuning {
    pub change_queue_capacity: usize,
    pub cell_size: f32,
    pub full_radius: f32,
    pub hlod_radius: f32,
    pub max_full_cells: usize,
    pub max_hlod_cells: usize,
    pub hlod_enabled: bool,
    pub hlod_min_cell_objects: usize,
    pub hlod_proxy_mesh_id: usize,
    pub hlod_proxy_material_id: usize,
    pub hlod_proxy_scale: f32,
    pub hlod_proxy_id_base: usize,
    pub hlod_hysteresis: f32,
    pub cull_camera_move_threshold: f32,
    pub cull_camera_rot_threshold: f32,
    pub streaming_camera_move_threshold: f32,
    pub streaming_camera_rot_threshold: f32,
    pub streaming_motion_speed_threshold: f32,
    pub streaming_motion_budget_scale: f32,
    pub streaming_object_budget: usize,
    pub streaming_request_budget: usize,
    pub per_object_culling: bool,
    pub per_object_lod: bool,
}

impl Default for RenderWorkerTuning {
    fn default() -> Self {
        Self {
            change_queue_capacity: 16,
            cell_size: 25.0,
            full_radius: 240.0,
            hlod_radius: 800.0,
            max_full_cells: 1024,
            max_hlod_cells: 2048,
            hlod_enabled: true,
            hlod_min_cell_objects: 8,
            hlod_proxy_mesh_id: 0,
            hlod_proxy_material_id: 0,
            hlod_proxy_scale: 1.0,
            hlod_proxy_id_base: 1_000_000_000,
            hlod_hysteresis: 0.1,
            cull_camera_move_threshold: 0.05,
            cull_camera_rot_threshold: 0.002,
            streaming_camera_move_threshold: 0.15,
            streaming_camera_rot_threshold: 0.01,
            streaming_motion_speed_threshold: 0.6,
            streaming_motion_budget_scale: 0.35,
            streaming_object_budget: 2048,
            streaming_request_budget: 1024,
            per_object_culling: true,
            per_object_lod: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CellKey {
    x: i32,
    y: i32,
    z: i32,
}

impl CellKey {
    fn from_position(pos: Vec3, cell_size: f32) -> Self {
        let inv = 1.0 / cell_size;
        Self {
            x: (pos.x * inv).floor() as i32,
            y: (pos.y * inv).floor() as i32,
            z: (pos.z * inv).floor() as i32,
        }
    }

    fn center(self, cell_size: f32) -> Vec3 {
        Vec3::new(
            (self.x as f32 + 0.5) * cell_size,
            (self.y as f32 + 0.5) * cell_size,
            (self.z as f32 + 0.5) * cell_size,
        )
    }
}

struct CellData {
    objects: Vec<Entity>,
    full_active: bool,
    proxy_active: bool,
    proxy_sent: bool,
    proxy_id: usize,
    last_proxy_transform: Transform,
}

impl CellData {
    fn new(proxy_id: usize) -> Self {
        Self {
            objects: Vec::new(),
            full_active: false,
            proxy_active: false,
            proxy_sent: false,
            proxy_id,
            last_proxy_transform: Transform::default(),
        }
    }
}

#[inline]
fn transform_approx_eq(a: &Transform, b: &Transform, epsilon: f32, rotation_epsilon: f32) -> bool {
    a.position.abs_diff_eq(b.position, epsilon)
        && a.scale.abs_diff_eq(b.scale, epsilon)
        && a.rotation.dot(b.rotation).abs() >= 1.0 - rotation_epsilon
}

fn select_lod_hysteresis(
    distance_sq: f32,
    current_lod: usize,
    thresholds: [f32; 3],
    hysteresis: f32,
) -> usize {
    let hysteresis = hysteresis.clamp(0.0, 0.9);
    let expand = 1.0 + hysteresis;
    let shrink = 1.0 - hysteresis;

    match current_lod {
        0 => {
            if distance_sq > thresholds[0] * expand {
                1
            } else {
                0
            }
        }
        1 => {
            if distance_sq > thresholds[1] * expand {
                2
            } else if distance_sq < thresholds[0] * shrink {
                0
            } else {
                1
            }
        }
        2 => {
            if distance_sq > thresholds[2] * expand {
                3
            } else if distance_sq < thresholds[1] * shrink {
                1
            } else {
                2
            }
        }
        _ => {
            if distance_sq < thresholds[2] * shrink {
                2
            } else {
                3
            }
        }
    }
}

fn compute_lod_index(
    state: &mut LodState,
    distance_sq: f32,
    frame_index: u32,
    lod_smoothing: f32,
    lod_thresholds: [f32; 3],
    lod_hysteresis: f32,
    min_change_frames: u32,
    is_new: bool,
) -> usize {
    state.last_seen_frame = frame_index;
    if lod_smoothing <= 0.0 {
        state.smoothed_distance_sq = distance_sq;
    } else {
        state.smoothed_distance_sq += (distance_sq - state.smoothed_distance_sq) * lod_smoothing;
    }

    let desired_lod = select_lod_hysteresis(
        state.smoothed_distance_sq,
        state.lod_index,
        lod_thresholds,
        lod_hysteresis,
    );

    if is_new {
        state.lod_index = desired_lod;
        state.last_change_frame = frame_index;
    } else if desired_lod != state.lod_index
        && frame_index.saturating_sub(state.last_change_frame) >= min_change_frames
    {
        state.lod_index = desired_lod;
        state.last_change_frame = frame_index;
    }

    state.lod_index
}

//================================================================================
// Render Extraction Worker
//================================================================================

#[derive(Clone, Copy)]
struct RenderObjectUpdate {
    entity: Entity,
    transform: Transform,
    mesh_renderer: MeshRenderer,
}

#[derive(Clone, Copy)]
struct RenderLightUpdate {
    entity: Entity,
    transform: Transform,
    light: Light,
}

#[derive(Default)]
pub struct RenderWorkerState {
    worker: Option<RenderWorker>,
    frame_index: u32,
    camera_sent: bool,
    config_sent: bool,
    needs_full_sync: bool,
    last_render_config: Option<RenderConfig>,
    last_render_graph_version: Option<u64>,
    last_streaming_tuning: Option<StreamingTuning>,
    last_lod_tuning: Option<LodTuning>,
    last_worker_tuning: Option<RenderWorkerTuning>,
    last_gizmo: Option<GizmoData>,
}

#[derive(SystemParam)]
pub struct RenderSystemResources<'w> {
    asset_server: Option<Res<'w, BevyAssetServer>>,
    runtime_config: Option<Res<'w, BevyRuntimeConfig>>,
    render_graph: Option<Res<'w, RenderGraphResource>>,
    gizmo_state: Option<Res<'w, RenderGizmoState>>,
    streaming_tuning: Option<Res<'w, BevyStreamingTuning>>,
    lod_tuning: Option<Res<'w, BevyLodTuning>>,
    worker_tuning: Option<Res<'w, BevyRenderWorkerTuning>>,
    runtime_profiling: Option<Res<'w, BevyRuntimeProfiling>>,
}

struct ProfilingScope {
    profiling: Option<Arc<RuntimeProfiling>>,
    start: Option<Instant>,
}

impl ProfilingScope {
    fn new(profiling: Option<Arc<RuntimeProfiling>>) -> Self {
        let start = profiling.as_ref().and_then(|profiling| {
            if profiling.enabled.load(std::sync::atomic::Ordering::Relaxed) {
                Some(Instant::now())
            } else {
                None
            }
        });
        Self { profiling, start }
    }
}

impl Drop for ProfilingScope {
    fn drop(&mut self) {
        let Some(profiling) = self.profiling.as_ref() else {
            return;
        };
        let Some(start) = self.start.take() else {
            return;
        };
        profiling.ecs_render_data_us.store(
            start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

struct RenderWorker {
    change_tx: Sender<RenderChange>,
    result_rx: Receiver<RenderResult>,
    in_flight: bool,
}

enum RenderChange {
    UpsertObjects {
        objects: Vec<RenderObjectUpdate>,
    },
    RemoveObjects {
        entities: Vec<Entity>,
    },
    UpsertLights {
        lights: Vec<RenderLightUpdate>,
    },
    RemoveLights {
        entities: Vec<Entity>,
    },
    UpdateCamera {
        camera: Camera,
        transform: Transform,
    },
    UpdateConfig {
        render_config: RenderConfig,
        render_graph: RenderGraphSpec,
        streaming_tuning: StreamingTuning,
        lod_tuning: LodTuning,
        worker_tuning: RenderWorkerTuning,
    },
    RequestFrame {
        frame_index: u32,
    },
}

struct RenderResult {
    render_delta: RenderDelta,
    visible_object_count: usize,
}

struct RenderObjectEntry {
    transform: Transform,
    mesh_id: usize,
    material_id: usize,
    casts_shadow: bool,
    visible: bool,
    lod_state: LodState,
    last_cull_transform: Transform,
    last_bounds_transform: Transform,
    bounds_center: Vec3,
    axis_x: Vec3,
    axis_y: Vec3,
    axis_z: Vec3,
    mesh_bounds: Aabb,
    mesh_bounds_valid: bool,
    bounds_dirty: bool,
    cell_key: CellKey,
    cell_index: usize,
    last_visible: bool,
    cull_dirty: bool,
    last_sent_transform: Transform,
    last_sent_mesh_id: usize,
    last_sent_material_id: usize,
    last_sent_casts_shadow: bool,
    last_sent_lod_index: usize,
    last_sent_visible: bool,
}

impl RenderObjectEntry {
    fn new(
        transform: Transform,
        mesh_renderer: MeshRenderer,
        cell_key: CellKey,
        cell_index: usize,
    ) -> Self {
        Self {
            transform,
            mesh_id: mesh_renderer.mesh_id,
            material_id: mesh_renderer.material_id,
            casts_shadow: mesh_renderer.casts_shadow,
            visible: mesh_renderer.visible,
            lod_state: LodState {
                last_seen_frame: u32::MAX,
                ..Default::default()
            },
            last_cull_transform: transform,
            last_bounds_transform: transform,
            bounds_center: Vec3::ZERO,
            axis_x: Vec3::ZERO,
            axis_y: Vec3::ZERO,
            axis_z: Vec3::ZERO,
            mesh_bounds: Aabb {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
            },
            mesh_bounds_valid: false,
            bounds_dirty: true,
            cell_key,
            cell_index,
            last_visible: false,
            cull_dirty: true,
            last_sent_transform: transform,
            last_sent_mesh_id: mesh_renderer.mesh_id,
            last_sent_material_id: mesh_renderer.material_id,
            last_sent_casts_shadow: mesh_renderer.casts_shadow,
            last_sent_lod_index: 0,
            last_sent_visible: false,
        }
    }

    fn update(&mut self, transform: Transform, mesh_renderer: MeshRenderer) {
        self.transform = transform;
        self.bounds_dirty = true;

        if self.mesh_id != mesh_renderer.mesh_id {
            self.mesh_id = mesh_renderer.mesh_id;
            self.lod_state = LodState {
                last_seen_frame: u32::MAX,
                ..Default::default()
            };
            self.cull_dirty = true;
            self.mesh_bounds_valid = false;
        }

        if self.material_id != mesh_renderer.material_id {
            self.material_id = mesh_renderer.material_id;
        }

        if self.casts_shadow != mesh_renderer.casts_shadow {
            self.casts_shadow = mesh_renderer.casts_shadow;
        }

        if self.visible != mesh_renderer.visible {
            self.visible = mesh_renderer.visible;
            self.cull_dirty = true;
        }
    }
}

struct RenderLightEntry {
    transform: Transform,
    light: Light,
    last_sent_transform: Transform,
    last_sent_light: Light,
    last_sent_present: bool,
}

impl RenderLightEntry {
    fn new(transform: Transform, light: Light) -> Self {
        Self {
            transform,
            light,
            last_sent_transform: transform,
            last_sent_light: light,
            last_sent_present: false,
        }
    }

    fn update(&mut self, transform: Transform, light: Light) {
        self.transform = transform;
        self.light = light;
    }
}

fn spawn_render_worker(
    mesh_aabb_map: Arc<RwLock<MeshAabbMap>>,
    change_capacity: usize,
) -> RenderWorker {
    let capacity = change_capacity.max(1);
    let (change_tx, change_rx) = bounded(capacity);
    let (result_tx, result_rx) = bounded(1);

    thread::Builder::new()
        .name("render-data-worker".to_string())
        .spawn(move || render_worker_loop(change_rx, result_tx, mesh_aabb_map))
        .expect("render data worker thread");

    RenderWorker {
        change_tx,
        result_rx,
        in_flight: false,
    }
}

fn try_send_change(
    worker_state: &mut RenderWorkerState,
    worker: &RenderWorker,
    change: RenderChange,
    label: &str,
) -> bool {
    match worker.change_tx.try_send(change) {
        Ok(_) => true,
        Err(TrySendError::Full(_)) => {
            warn!("render worker change queue full; deferring (drop {label})");
            worker_state.camera_sent = false;
            worker_state.config_sent = false;
            worker_state.needs_full_sync = true;
            false
        }
        Err(TrySendError::Disconnected(_)) => {
            warn!("render worker disconnected");
            worker_state.camera_sent = false;
            worker_state.config_sent = false;
            worker_state.needs_full_sync = true;
            false
        }
    }
}

fn render_worker_loop(
    change_rx: Receiver<RenderChange>,
    result_tx: Sender<RenderResult>,
    mesh_aabb_map: Arc<RwLock<MeshAabbMap>>,
) {
    let mut objects: HashMap<Entity, RenderObjectEntry> = HashMap::new();
    let mut lights: HashMap<Entity, RenderLightEntry> = HashMap::new();
    let mut cells: HashMap<CellKey, CellData> = HashMap::new();
    let mut camera_component = Camera::default();
    let mut logical_camera_transform = Transform::default();
    let mut render_camera_transform = Transform::default();
    let mut render_config = RenderConfig::default();
    let mut render_graph = default_graph_spec();
    let mut streaming_tuning = StreamingTuning::default();
    let mut lod_tuning = LodTuning::default();
    let mut worker_tuning = RenderWorkerTuning::default();
    let mut next_proxy_id = worker_tuning.hlod_proxy_id_base;
    let mut has_camera = false;
    let mut last_camera_component = Camera::default();
    let mut last_render_camera_transform = Transform::default();
    let mut has_previous_camera = false;
    let mut last_stream_camera_transform = Transform::default();
    let mut has_stream_camera = false;
    let mut last_stream_motion_transform = Transform::default();
    let mut has_stream_motion = false;
    let mut last_cull_camera_transform = Transform::default();
    let mut has_cull_camera = false;
    let mut config_dirty = true;
    let mut culling_dirty = true;
    let mut lod_dirty = true;
    let mut streaming_dirty = true;
    let mut cells_dirty = true;
    let mut rebuild_cells = false;
    let mut force_full = false;
    let mut last_cull_frame = 0u32;
    let mut last_lod_frame = 0u32;
    let mut last_stream_frame = 0u32;
    let mut pending_frame: Option<u32> = None;
    let mut pending_removed_objects: Vec<usize> = Vec::new();
    let mut pending_removed_lights: Vec<usize> = Vec::new();
    let mut pending_removed_proxies: Vec<usize> = Vec::new();
    let mut has_sent_any = false;
    let mut dirty_objects: Vec<Entity> = Vec::new();
    let mut visible_object_count: usize = 0;
    let mut streaming_queue: VecDeque<Entity> = VecDeque::new();

    let apply_change = |change: RenderChange,
                        objects: &mut HashMap<Entity, RenderObjectEntry>,
                        lights: &mut HashMap<Entity, RenderLightEntry>,
                        cells: &mut HashMap<CellKey, CellData>,
                        next_proxy_id: &mut usize,
                        camera_component: &mut Camera,
                        logical_camera_transform: &mut Transform,
                        render_camera_transform: &mut Transform,
                        render_config: &mut RenderConfig,
                        render_graph: &mut RenderGraphSpec,
                        streaming_tuning: &mut StreamingTuning,
                        lod_tuning: &mut LodTuning,
                        worker_tuning: &mut RenderWorkerTuning,
                        has_camera: &mut bool,
                        config_dirty: &mut bool,
                        culling_dirty: &mut bool,
                        lod_dirty: &mut bool,
                        streaming_dirty: &mut bool,
                        cells_dirty: &mut bool,
                        rebuild_cells: &mut bool,
                        force_full: &mut bool,
                        pending_frame: &mut Option<u32>,
                        pending_removed_objects: &mut Vec<usize>,
                        pending_removed_lights: &mut Vec<usize>,
                        pending_removed_proxies: &mut Vec<usize>,
                        dirty_objects: &mut Vec<Entity>,
                        visible_object_count: &mut usize| {
        match change {
            RenderChange::UpsertObjects { objects: updates } => {
                for update in updates {
                    let cell_size = worker_tuning.cell_size.max(f32::EPSILON);
                    let cell_key = CellKey::from_position(update.transform.position, cell_size);
                    match objects.entry(update.entity) {
                        Entry::Occupied(mut entry) => {
                            let (old_key, old_index) = {
                                let entry_ref = entry.get();
                                (entry_ref.cell_key, entry_ref.cell_index)
                            };
                            let mut moved_entity: Option<(Entity, usize)> = None;
                            if old_key != cell_key {
                                if let Some(cell) = cells.get_mut(&old_key) {
                                    let idx = old_index;
                                    let last_idx = cell.objects.len().saturating_sub(1);
                                    cell.objects.swap_remove(idx);
                                    if idx < last_idx {
                                        let moved = cell.objects[idx];
                                        moved_entity = Some((moved, idx));
                                    }
                                    if cell.objects.is_empty() {
                                        if cell.proxy_sent {
                                            pending_removed_proxies.push(cell.proxy_id);
                                            *visible_object_count =
                                                (*visible_object_count).saturating_sub(1);
                                        }
                                        cells.remove(&old_key);
                                    }
                                }

                                let cell = cells.entry(cell_key).or_insert_with(|| {
                                    let id = *next_proxy_id;
                                    *next_proxy_id = next_proxy_id.wrapping_add(1);
                                    CellData::new(id)
                                });
                                let new_index = cell.objects.len();
                                cell.objects.push(update.entity);
                                *cells_dirty = true;
                                let entry_mut = entry.get_mut();
                                entry_mut.cell_key = cell_key;
                                entry_mut.cell_index = new_index;
                                entry_mut.update(update.transform, update.mesh_renderer);
                                if !entry_mut.cull_dirty {
                                    entry_mut.cull_dirty = true;
                                    dirty_objects.push(update.entity);
                                }
                            } else {
                                let entry_mut = entry.get_mut();
                                entry_mut.update(update.transform, update.mesh_renderer);
                                if !entry_mut.cull_dirty {
                                    entry_mut.cull_dirty = true;
                                    dirty_objects.push(update.entity);
                                }
                            }
                            if let Some((moved, idx)) = moved_entity {
                                if let Some(moved_entry) = objects.get_mut(&moved) {
                                    moved_entry.cell_index = idx;
                                }
                            }
                        }
                        Entry::Vacant(entry) => {
                            let cell = cells.entry(cell_key).or_insert_with(|| {
                                let id = *next_proxy_id;
                                *next_proxy_id = next_proxy_id.wrapping_add(1);
                                CellData::new(id)
                            });
                            let cell_index = cell.objects.len();
                            cell.objects.push(update.entity);
                            entry.insert(RenderObjectEntry::new(
                                update.transform,
                                update.mesh_renderer,
                                cell_key,
                                cell_index,
                            ));
                            dirty_objects.push(update.entity);
                            *cells_dirty = true;
                        }
                    }
                }
                *streaming_dirty = true;
            }
            RenderChange::RemoveObjects { entities } => {
                for entity in entities {
                    if let Some(entry) = objects.remove(&entity) {
                        if let Some(cell) = cells.get_mut(&entry.cell_key) {
                            let idx = entry.cell_index;
                            let last_idx = cell.objects.len().saturating_sub(1);
                            cell.objects.swap_remove(idx);
                            if idx < last_idx {
                                let moved = cell.objects[idx];
                                if let Some(moved_entry) = objects.get_mut(&moved) {
                                    moved_entry.cell_index = idx;
                                }
                            }
                            if cell.objects.is_empty() {
                                if cell.proxy_sent {
                                    pending_removed_proxies.push(cell.proxy_id);
                                    *visible_object_count = visible_object_count.saturating_sub(1);
                                }
                                cells.remove(&entry.cell_key);
                            }
                        }
                        if entry.last_visible {
                            *visible_object_count = visible_object_count.saturating_sub(1);
                        }
                        if entry.last_sent_visible {
                            pending_removed_objects.push(entity.index() as usize);
                        }
                    }
                }
                *streaming_dirty = true;
                *cells_dirty = true;
            }
            RenderChange::UpsertLights { lights: updates } => {
                for update in updates {
                    lights
                        .entry(update.entity)
                        .and_modify(|entry| entry.update(update.transform, update.light))
                        .or_insert_with(|| RenderLightEntry::new(update.transform, update.light));
                }
            }
            RenderChange::RemoveLights { entities } => {
                for entity in entities {
                    if let Some(entry) = lights.remove(&entity) {
                        if entry.last_sent_present {
                            pending_removed_lights.push(entity.index() as usize);
                        }
                    }
                }
            }
            RenderChange::UpdateCamera { camera, transform } => {
                let first_camera = !*has_camera;
                *camera_component = camera;
                *logical_camera_transform = transform;
                *has_camera = true;
                if first_camera || !render_config.freeze_render_camera {
                    *render_camera_transform = transform;
                }
            }
            RenderChange::UpdateConfig {
                render_config: updated_config,
                render_graph: updated_graph,
                streaming_tuning: updated_streaming,
                lod_tuning: updated_lod,
                worker_tuning: updated_worker,
            } => {
                let prev_config = *render_config;
                let prev_streaming = *streaming_tuning;
                let prev_lod = *lod_tuning;
                let prev_worker = *worker_tuning;
                *render_config = updated_config;
                *render_graph = updated_graph;
                *streaming_tuning = updated_streaming;
                *lod_tuning = updated_lod;
                *worker_tuning = updated_worker;
                *config_dirty = true;
                if prev_config.freeze_render_camera != updated_config.freeze_render_camera
                    && *has_camera
                {
                    *render_camera_transform = *logical_camera_transform;
                }
                if prev_config.frustum_culling != updated_config.frustum_culling
                    || prev_config.transform_epsilon != updated_config.transform_epsilon
                    || prev_config.rotation_epsilon != updated_config.rotation_epsilon
                    || prev_config.cull_interval_frames != updated_config.cull_interval_frames
                    || prev_config.gpu_driven != updated_config.gpu_driven
                {
                    *culling_dirty = true;
                }
                if prev_config.lod != updated_config.lod
                    || prev_config.transform_epsilon != updated_config.transform_epsilon
                    || prev_config.rotation_epsilon != updated_config.rotation_epsilon
                    || prev_config.lod_interval_frames != updated_config.lod_interval_frames
                    || prev_config.gpu_driven != updated_config.gpu_driven
                    || prev_lod != updated_lod
                {
                    *lod_dirty = true;
                }
                if prev_config.streaming_interval_frames != updated_config.streaming_interval_frames
                    || prev_streaming != updated_streaming
                {
                    *streaming_dirty = true;
                }
                if prev_worker != updated_worker {
                    *cells_dirty = true;
                    if prev_worker.cell_size != updated_worker.cell_size {
                        *rebuild_cells = true;
                        *force_full = true;
                    }
                    if prev_worker.hlod_proxy_mesh_id != updated_worker.hlod_proxy_mesh_id
                        || prev_worker.hlod_proxy_material_id
                            != updated_worker.hlod_proxy_material_id
                        || prev_worker.hlod_proxy_scale != updated_worker.hlod_proxy_scale
                        || prev_worker.hlod_enabled != updated_worker.hlod_enabled
                        || prev_worker.hlod_radius != updated_worker.hlod_radius
                        || prev_worker.full_radius != updated_worker.full_radius
                    {
                        *force_full = true;
                    }
                }
            }
            RenderChange::RequestFrame { frame_index } => {
                *pending_frame = Some(frame_index);
            }
        }
    };

    loop {
        let change = match change_rx.recv() {
            Ok(change) => change,
            Err(_) => break,
        };

        apply_change(
            change,
            &mut objects,
            &mut lights,
            &mut cells,
            &mut next_proxy_id,
            &mut camera_component,
            &mut logical_camera_transform,
            &mut render_camera_transform,
            &mut render_config,
            &mut render_graph,
            &mut streaming_tuning,
            &mut lod_tuning,
            &mut worker_tuning,
            &mut has_camera,
            &mut config_dirty,
            &mut culling_dirty,
            &mut lod_dirty,
            &mut streaming_dirty,
            &mut cells_dirty,
            &mut rebuild_cells,
            &mut force_full,
            &mut pending_frame,
            &mut pending_removed_objects,
            &mut pending_removed_lights,
            &mut pending_removed_proxies,
            &mut dirty_objects,
            &mut visible_object_count,
        );

        while let Ok(change) = change_rx.try_recv() {
            apply_change(
                change,
                &mut objects,
                &mut lights,
                &mut cells,
                &mut next_proxy_id,
                &mut camera_component,
                &mut logical_camera_transform,
                &mut render_camera_transform,
                &mut render_config,
                &mut render_graph,
                &mut streaming_tuning,
                &mut lod_tuning,
                &mut worker_tuning,
                &mut has_camera,
                &mut config_dirty,
                &mut culling_dirty,
                &mut lod_dirty,
                &mut streaming_dirty,
                &mut cells_dirty,
                &mut rebuild_cells,
                &mut force_full,
                &mut pending_frame,
                &mut pending_removed_objects,
                &mut pending_removed_lights,
                &mut pending_removed_proxies,
                &mut dirty_objects,
                &mut visible_object_count,
            );
        }

        let Some(frame_index) = pending_frame.take() else {
            continue;
        };
        if !has_camera {
            continue;
        }

        if rebuild_cells {
            cells.clear();
            next_proxy_id = worker_tuning.hlod_proxy_id_base;
            let cell_size = worker_tuning.cell_size.max(f32::EPSILON);
            for (&entity, entry) in objects.iter_mut() {
                let cell_key = CellKey::from_position(entry.transform.position, cell_size);
                let cell = cells.entry(cell_key).or_insert_with(|| {
                    let id = next_proxy_id;
                    next_proxy_id = next_proxy_id.wrapping_add(1);
                    CellData::new(id)
                });
                entry.cell_key = cell_key;
                entry.cell_index = cell.objects.len();
                cell.objects.push(entity);
            }
            rebuild_cells = false;
            cells_dirty = true;
        }

        let transform_epsilon = render_config.transform_epsilon.max(0.0);
        let rotation_epsilon = render_config.rotation_epsilon.clamp(0.0, 1.0);
        let cull_move_threshold = worker_tuning.cull_camera_move_threshold.max(0.0);
        let cull_rot_threshold = worker_tuning.cull_camera_rot_threshold.clamp(0.0, 1.0);
        let stream_move_threshold = worker_tuning.streaming_camera_move_threshold.max(0.0);
        let stream_rot_threshold = worker_tuning.streaming_camera_rot_threshold.clamp(0.0, 1.0);

        let stream_camera_changed = !has_stream_camera
            || !transform_approx_eq(
                &logical_camera_transform,
                &last_stream_camera_transform,
                stream_move_threshold,
                stream_rot_threshold,
            );
        if stream_camera_changed {
            last_stream_camera_transform = logical_camera_transform;
            has_stream_camera = true;
            streaming_dirty = true;
        }

        let camera_component_changed = camera_component != last_camera_component;
        let camera_transform_changed = !transform_approx_eq(
            &render_camera_transform,
            &last_render_camera_transform,
            transform_epsilon,
            rotation_epsilon,
        );
        let camera_changed =
            camera_component_changed || !has_previous_camera || camera_transform_changed;

        let cull_camera_changed = !has_cull_camera
            || !transform_approx_eq(
                &render_camera_transform,
                &last_cull_camera_transform,
                cull_move_threshold,
                cull_rot_threshold,
            );

        let frustum = if render_config.frustum_culling {
            Some(Frustum::from_camera(
                &render_camera_transform,
                camera_component.fov_y_rad,
                camera_component.aspect_ratio,
                camera_component.near_plane,
                camera_component.far_plane,
            ))
        } else {
            None
        };

        let frustum_culling = render_config.frustum_culling;
        let gpu_driven = render_config.gpu_driven;
        let per_object_culling = worker_tuning.per_object_culling && frustum_culling && !gpu_driven;
        let lod_enabled = render_config.lod;
        let lod_smoothing = lod_tuning.smoothing.clamp(0.0, 1.0);
        let lod_hysteresis = lod_tuning.hysteresis.clamp(0.0, 0.9);
        let lod0 = lod_tuning.lod0_distance.max(0.0);
        let lod1 = lod_tuning.lod1_distance.max(lod0);
        let lod2 = lod_tuning.lod2_distance.max(lod1);
        let lod_thresholds = [lod0 * lod0, lod1 * lod1, lod2 * lod2];

        let render_camera_pos = render_camera_transform.position;
        let stream_camera_pos = logical_camera_transform.position;
        let cull_interval = render_config.cull_interval_frames;
        let cull_interval_ok =
            cull_interval == 0 || frame_index.saturating_sub(last_cull_frame) >= cull_interval;
        let do_full_cull = culling_dirty || (cull_camera_changed && cull_interval_ok);
        let lod_interval = render_config.lod_interval_frames;
        let lod_interval_ok =
            lod_interval == 0 || frame_index.saturating_sub(last_lod_frame) >= lod_interval;
        let do_full_lod = lod_dirty || (cull_camera_changed && lod_interval_ok);
        let full_scan = do_full_cull || do_full_lod || cells_dirty || force_full;

        if do_full_cull {
            last_cull_frame = frame_index;
            last_cull_camera_transform = render_camera_transform;
            has_cull_camera = true;
        }
        if do_full_lod {
            last_lod_frame = frame_index;
        }

        let motion_delta = if has_stream_motion {
            (logical_camera_transform.position - last_stream_motion_transform.position).length()
        } else {
            0.0
        };
        last_stream_motion_transform = logical_camera_transform;
        has_stream_motion = true;
        let motion_threshold = worker_tuning.streaming_motion_speed_threshold.max(0.0);
        let high_motion = motion_delta > motion_threshold;
        let motion_scale = if high_motion {
            worker_tuning.streaming_motion_budget_scale.clamp(0.0, 1.0)
        } else {
            1.0
        };

        let streaming_interval = render_config.streaming_interval_frames;
        let streaming_interval_ok = streaming_interval == 0
            || frame_index.saturating_sub(last_stream_frame) >= streaming_interval;
        let send_streaming = streaming_dirty
            || streaming_interval_ok
            || !has_sent_any
            || !streaming_queue.is_empty();
        let per_object_lod =
            worker_tuning.per_object_lod && lod_enabled && (!gpu_driven || send_streaming);
        let mut streaming_hints: Option<HashMap<(AssetStreamKind, usize), AssetStreamingRequest>> =
            send_streaming.then(HashMap::new);
        if let Some(hints) = streaming_hints.as_mut() {
            let reserve = worker_tuning.streaming_object_budget.saturating_mul(2);
            if reserve > 0 {
                hints.reserve(reserve);
            }
        }

        let mesh_aabb_map = mesh_aabb_map.read();

        let mut objects_upsert = Vec::new();
        let mut objects_remove = std::mem::take(&mut pending_removed_objects);
        let proxy_valid = worker_tuning.hlod_enabled
            && worker_tuning.hlod_proxy_mesh_id != 0
            && worker_tuning.hlod_proxy_material_id != 0;

        let mut push_streaming_hints =
            |entry: &RenderObjectEntry, aabb: &Aabb, lod_index: usize| {
                let Some(hints) = streaming_hints.as_mut() else {
                    return;
                };
                let hint_priority = {
                    let size = aabb.extents().length();
                    let distance = (stream_camera_pos - entry.transform.position).length();
                    let size_bias = streaming_tuning.priority_size_bias.max(0.0);
                    let distance_bias = streaming_tuning.priority_distance_bias.max(0.0);
                    let lod_bias = streaming_tuning.priority_lod_bias.max(f32::EPSILON);
                    let lod_penalty = 1.0 / (lod_index as f32 + lod_bias);
                    let shadow_boost = if entry.casts_shadow {
                        streaming_tuning.shadow_priority_boost
                    } else {
                        1.0
                    };
                    let denom = (distance + distance_bias).max(f32::EPSILON);
                    (size + size_bias) * shadow_boost * lod_penalty / denom
                };
                hints
                    .entry((AssetStreamKind::Mesh, entry.mesh_id))
                    .and_modify(|req| {
                        req.priority = req.priority.max(hint_priority);
                        req.max_lod = req.max_lod.map(|l| l.min(lod_index)).or(Some(lod_index));
                    })
                    .or_insert(AssetStreamingRequest {
                        id: entry.mesh_id,
                        kind: AssetStreamKind::Mesh,
                        priority: hint_priority,
                        max_lod: Some(lod_index),
                        force_low_res: false,
                    });
                hints
                    .entry((AssetStreamKind::Material, entry.material_id))
                    .and_modify(|req| req.priority = req.priority.max(hint_priority))
                    .or_insert(AssetStreamingRequest {
                        id: entry.material_id,
                        kind: AssetStreamKind::Material,
                        priority: hint_priority,
                        max_lod: None,
                        force_low_res: false,
                    });
            };

        let mut process_object =
            |entity: Entity,
             entry: &mut RenderObjectEntry,
             cell_full_active: bool,
             objects_remove: &mut Vec<usize>,
             visible_object_count: &mut usize| {
                let current_transform = entry.transform;

                if !cell_full_active || !entry.visible {
                    if entry.last_visible {
                        entry.last_visible = false;
                        *visible_object_count = (*visible_object_count).saturating_sub(1);
                    }
                    if entry.last_sent_visible {
                        objects_remove.push(entity.index() as usize);
                        entry.last_sent_visible = false;
                    }
                    entry.cull_dirty = false;
                    return;
                }

                if !entry.mesh_bounds_valid {
                    if let Some(aabb) = mesh_aabb_map.0.get(&entry.mesh_id) {
                        entry.mesh_bounds = *aabb;
                        entry.mesh_bounds_valid = true;
                        entry.bounds_dirty = true;
                    } else {
                        if entry.last_visible {
                            entry.last_visible = false;
                            *visible_object_count = (*visible_object_count).saturating_sub(1);
                        }
                        if entry.last_sent_visible {
                            objects_remove.push(entity.index() as usize);
                            entry.last_sent_visible = false;
                        }
                        entry.cull_dirty = false;
                        return;
                    }
                }

                let aabb = entry.mesh_bounds;
                let transform_changed = !transform_approx_eq(
                    &current_transform,
                    &entry.last_cull_transform,
                    transform_epsilon,
                    rotation_epsilon,
                );
                let bounds_changed = entry.bounds_dirty
                    || !transform_approx_eq(
                        &current_transform,
                        &entry.last_bounds_transform,
                        transform_epsilon,
                        rotation_epsilon,
                    );
                if bounds_changed {
                    let center_world = current_transform.position
                        + current_transform.rotation * (aabb.center() * current_transform.scale);
                    let extents = aabb.extents();
                    let scale_abs = current_transform.scale.abs();
                    entry.axis_x =
                        current_transform.rotation * (Vec3::X * (extents.x * scale_abs.x));
                    entry.axis_y =
                        current_transform.rotation * (Vec3::Y * (extents.y * scale_abs.y));
                    entry.axis_z =
                        current_transform.rotation * (Vec3::Z * (extents.z * scale_abs.z));
                    entry.bounds_center = center_world;
                    entry.last_bounds_transform = current_transform;
                    entry.bounds_dirty = false;
                }

                let was_visible = entry.last_visible;
                let recompute_visibility = do_full_cull || transform_changed || entry.cull_dirty;
                let mut visible = was_visible;

                if recompute_visibility {
                    if per_object_culling {
                        visible = frustum
                            .as_ref()
                            .map(|frustum| {
                                frustum.intersects_obb(
                                    entry.bounds_center,
                                    entry.axis_x,
                                    entry.axis_y,
                                    entry.axis_z,
                                )
                            })
                            .unwrap_or(true);
                    } else {
                        visible = true;
                    }

                    entry.last_cull_transform = current_transform;
                    entry.last_visible = visible;
                    entry.cull_dirty = false;

                    if visible != was_visible {
                        if visible {
                            *visible_object_count = (*visible_object_count).saturating_add(1);
                        } else {
                            *visible_object_count = (*visible_object_count).saturating_sub(1);
                        }
                    }
                }

                if !visible {
                    if !gpu_driven {
                        if entry.last_sent_visible {
                            objects_remove.push(entity.index() as usize);
                            entry.last_sent_visible = false;
                        }
                        return;
                    }

                    let needs_update = !entry.last_sent_visible
                        || !transform_approx_eq(
                            &current_transform,
                            &entry.last_sent_transform,
                            transform_epsilon,
                            rotation_epsilon,
                        )
                        || entry.mesh_id != entry.last_sent_mesh_id
                        || entry.material_id != entry.last_sent_material_id
                        || entry.casts_shadow != entry.last_sent_casts_shadow;
                    if needs_update {
                        objects_upsert.push(RenderObjectDelta {
                            id: entity.index() as usize,
                            transform: current_transform,
                            mesh_id: entry.mesh_id,
                            material_id: entry.material_id,
                            casts_shadow: entry.casts_shadow,
                            lod_index: entry.last_sent_lod_index,
                        });
                        entry.last_sent_visible = true;
                        entry.last_sent_transform = current_transform;
                        entry.last_sent_mesh_id = entry.mesh_id;
                        entry.last_sent_material_id = entry.material_id;
                        entry.last_sent_casts_shadow = entry.casts_shadow;
                    }
                    return;
                }

                let became_visible = visible && !was_visible;
                let is_new = entry.lod_state.last_seen_frame == u32::MAX || became_visible;
                let lod_index = if per_object_lod {
                    if !do_full_lod && !transform_changed && !is_new {
                        entry.lod_state.last_seen_frame = frame_index;
                        entry.lod_state.lod_index
                    } else {
                        let camera_local = current_transform.rotation.conjugate()
                            * (render_camera_pos - current_transform.position);
                        let camera_local = camera_local / current_transform.scale;
                        let closest_point_local = camera_local.clamp(aabb.min, aabb.max);
                        let distance_sq = camera_local.distance_squared(closest_point_local);
                        compute_lod_index(
                            &mut entry.lod_state,
                            distance_sq,
                            frame_index,
                            lod_smoothing,
                            lod_thresholds,
                            lod_hysteresis,
                            lod_tuning.min_change_frames,
                            is_new,
                        )
                    }
                } else if lod_enabled {
                    entry.last_sent_lod_index
                } else {
                    0
                };

                let lod_changed = lod_index != entry.last_sent_lod_index;
                let should_send_lod = !gpu_driven || send_streaming;
                let needs_update = !entry.last_sent_visible
                    || !transform_approx_eq(
                        &current_transform,
                        &entry.last_sent_transform,
                        transform_epsilon,
                        rotation_epsilon,
                    )
                    || entry.mesh_id != entry.last_sent_mesh_id
                    || entry.material_id != entry.last_sent_material_id
                    || entry.casts_shadow != entry.last_sent_casts_shadow
                    || (lod_changed && should_send_lod);

                push_streaming_hints(entry, &aabb, lod_index);

                if needs_update {
                    objects_upsert.push(RenderObjectDelta {
                        id: entity.index() as usize,
                        transform: current_transform,
                        mesh_id: entry.mesh_id,
                        material_id: entry.material_id,
                        casts_shadow: entry.casts_shadow,
                        lod_index,
                    });
                    entry.last_sent_visible = true;
                    entry.last_sent_transform = current_transform;
                    entry.last_sent_mesh_id = entry.mesh_id;
                    entry.last_sent_material_id = entry.material_id;
                    entry.last_sent_casts_shadow = entry.casts_shadow;
                    entry.last_sent_lod_index = lod_index;
                }
            };

        if full_scan {
            let cell_size = worker_tuning.cell_size.max(f32::EPSILON);
            let full_radius = worker_tuning.full_radius.max(0.0);
            let hlod_radius = worker_tuning.hlod_radius.max(full_radius);
            let full_radius_sq = full_radius * full_radius;
            let hlod_radius_sq = hlod_radius * hlod_radius;
            let cell_extents = Vec3::splat(cell_size * 0.5);
            let axis_x = Vec3::X * cell_extents.x;
            let axis_y = Vec3::Y * cell_extents.y;
            let axis_z = Vec3::Z * cell_extents.z;

            let mut full_candidates: Vec<(f32, CellKey)> = Vec::new();
            let mut hlod_candidates: Vec<(f32, CellKey)> = Vec::new();
            let mut consider_cell = |key: CellKey, cell: &CellData| {
                if cell.objects.is_empty() {
                    return;
                }
                let center = key.center(cell_size);
                let distance_sq = (center - render_camera_pos).length_squared();
                let visible = if frustum_culling {
                    frustum
                        .as_ref()
                        .map(|frustum| frustum.intersects_obb(center, axis_x, axis_y, axis_z))
                        .unwrap_or(true)
                } else {
                    true
                };
                if !visible {
                    return;
                }
                if distance_sq <= full_radius_sq {
                    full_candidates.push((distance_sq, key));
                } else if worker_tuning.hlod_enabled && distance_sq <= hlod_radius_sq {
                    hlod_candidates.push((distance_sq, key));
                }
            };

            let range_min =
                CellKey::from_position(render_camera_pos - Vec3::splat(hlod_radius), cell_size);
            let range_max =
                CellKey::from_position(render_camera_pos + Vec3::splat(hlod_radius), cell_size);
            let range_x = (range_max.x - range_min.x + 1).max(0) as i64;
            let range_y = (range_max.y - range_min.y + 1).max(0) as i64;
            let range_z = (range_max.z - range_min.z + 1).max(0) as i64;
            let range_cells = range_x.saturating_mul(range_y).saturating_mul(range_z);

            if range_cells > 0 && range_cells <= (cells.len() as i64).saturating_mul(2) {
                for x in range_min.x..=range_max.x {
                    for y in range_min.y..=range_max.y {
                        for z in range_min.z..=range_max.z {
                            let key = CellKey { x, y, z };
                            if let Some(cell) = cells.get(&key) {
                                consider_cell(key, cell);
                            }
                        }
                    }
                }
            } else {
                for (key, cell) in cells.iter() {
                    consider_cell(*key, cell);
                }
            }

            let max_full = worker_tuning.max_full_cells;
            if max_full == 0 {
                full_candidates.clear();
            } else if full_candidates.len() > max_full {
                let pivot = max_full.saturating_sub(1);
                full_candidates.select_nth_unstable_by(pivot, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
                });
                full_candidates.truncate(max_full);
            }
            let max_hlod = worker_tuning.max_hlod_cells;
            if max_hlod == 0 {
                hlod_candidates.clear();
            } else if hlod_candidates.len() > max_hlod {
                let pivot = max_hlod.saturating_sub(1);
                hlod_candidates.select_nth_unstable_by(pivot, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
                });
                hlod_candidates.truncate(max_hlod);
            }

            let full_set: HashSet<CellKey> = full_candidates.iter().map(|(_, key)| *key).collect();
            let hlod_set: HashSet<CellKey> = hlod_candidates.iter().map(|(_, key)| *key).collect();

            for (key, cell) in cells.iter_mut() {
                let new_full = full_set.contains(key);
                let mut new_proxy = !new_full && hlod_set.contains(key);
                if new_proxy && cell.objects.len() < worker_tuning.hlod_min_cell_objects {
                    new_proxy = false;
                }
                if new_proxy && !proxy_valid {
                    new_proxy = false;
                }

                if cell.full_active && !new_full {
                    for entity in cell.objects.iter() {
                        if let Some(entry) = objects.get_mut(entity) {
                            if entry.last_visible {
                                entry.last_visible = false;
                                visible_object_count = visible_object_count.saturating_sub(1);
                            }
                            if entry.last_sent_visible {
                                objects_remove.push(entity.index() as usize);
                                entry.last_sent_visible = false;
                            }
                        }
                    }
                }
                cell.full_active = new_full;

                if cell.proxy_active && !new_proxy {
                    if cell.proxy_sent {
                        objects_remove.push(cell.proxy_id);
                        visible_object_count = visible_object_count.saturating_sub(1);
                    }
                    cell.proxy_active = false;
                    cell.proxy_sent = false;
                }
                if !cell.proxy_active && new_proxy {
                    cell.proxy_active = true;
                    cell.proxy_sent = false;
                }
                if new_full && cell.proxy_active {
                    if cell.proxy_sent {
                        objects_remove.push(cell.proxy_id);
                        visible_object_count = visible_object_count.saturating_sub(1);
                    }
                    cell.proxy_active = false;
                    cell.proxy_sent = false;
                }
            }

            for key in full_set.iter() {
                if let Some(cell) = cells.get_mut(key) {
                    for &entity in cell.objects.iter() {
                        if let Some(entry) = objects.get_mut(&entity) {
                            process_object(
                                entity,
                                entry,
                                true,
                                &mut objects_remove,
                                &mut visible_object_count,
                            );
                        }
                    }
                }
            }

            if proxy_valid {
                let proxy_scale = worker_tuning.hlod_proxy_scale.max(0.0);
                let proxy_mesh_id = worker_tuning.hlod_proxy_mesh_id;
                let proxy_material_id = worker_tuning.hlod_proxy_material_id;
                for key in hlod_set.iter() {
                    let Some(cell) = cells.get_mut(key) else {
                        continue;
                    };
                    if !cell.proxy_active {
                        continue;
                    }
                    let center = key.center(cell_size);
                    let proxy_transform = Transform {
                        position: center,
                        rotation: Quat::IDENTITY,
                        scale: Vec3::splat(cell_size * proxy_scale),
                    };
                    let needs_update = !cell.proxy_sent
                        || !transform_approx_eq(
                            &proxy_transform,
                            &cell.last_proxy_transform,
                            transform_epsilon,
                            rotation_epsilon,
                        );
                    if needs_update {
                        objects_upsert.push(RenderObjectDelta {
                            id: cell.proxy_id,
                            transform: proxy_transform,
                            mesh_id: proxy_mesh_id,
                            material_id: proxy_material_id,
                            casts_shadow: false,
                            lod_index: 0,
                        });
                        if !cell.proxy_sent {
                            visible_object_count = visible_object_count.saturating_add(1);
                        }
                        cell.proxy_sent = true;
                        cell.last_proxy_transform = proxy_transform;
                    }
                }
            }

            dirty_objects.clear();
            if do_full_cull {
                culling_dirty = false;
            }
            if do_full_lod {
                lod_dirty = false;
            }
            cells_dirty = false;
            force_full = false;
        } else {
            for entity in dirty_objects.drain(..) {
                if let Some(entry) = objects.get_mut(&entity) {
                    let cell_full_active = cells
                        .get(&entry.cell_key)
                        .map(|cell| cell.full_active)
                        .unwrap_or(false);
                    process_object(
                        entity,
                        entry,
                        cell_full_active,
                        &mut objects_remove,
                        &mut visible_object_count,
                    );
                }
            }
        }

        let request_budget = worker_tuning.streaming_request_budget;
        if send_streaming {
            if request_budget == 0 {
                streaming_queue.clear();
                last_stream_frame = frame_index;
                streaming_dirty = false;
            } else {
                if streaming_queue.is_empty() || streaming_dirty {
                    streaming_queue.clear();
                    let cell_size = worker_tuning.cell_size.max(f32::EPSILON);
                    let mut active_cells: Vec<(f32, CellKey)> = Vec::new();
                    for (key, cell) in cells.iter() {
                        if !cell.full_active {
                            continue;
                        }
                        let center = key.center(cell_size);
                        let distance_sq = (center - stream_camera_pos).length_squared();
                        active_cells.push((distance_sq, *key));
                    }
                    active_cells.sort_by(|a, b| {
                        a.0.total_cmp(&b.0)
                            .then_with(|| a.1.x.cmp(&b.1.x))
                            .then_with(|| a.1.y.cmp(&b.1.y))
                            .then_with(|| a.1.z.cmp(&b.1.z))
                    });
                    for (_, key) in active_cells {
                        if let Some(cell) = cells.get(&key) {
                            streaming_queue.extend(cell.objects.iter().copied());
                        }
                    }
                    streaming_dirty = false;
                }

                let budget =
                    ((worker_tuning.streaming_object_budget as f32) * motion_scale).ceil() as usize;
                let mut processed = 0usize;
                while processed < budget {
                    let Some(entity) = streaming_queue.pop_front() else {
                        break;
                    };
                    let Some(entry) = objects.get_mut(&entity) else {
                        continue;
                    };
                    if !entry.visible {
                        continue;
                    }
                    if !entry.mesh_bounds_valid {
                        if let Some(aabb) = mesh_aabb_map.0.get(&entry.mesh_id) {
                            entry.mesh_bounds = *aabb;
                            entry.mesh_bounds_valid = true;
                        } else {
                            continue;
                        }
                    }
                    let lod_index = if lod_enabled {
                        entry.lod_state.lod_index
                    } else {
                        0
                    };
                    push_streaming_hints(entry, &entry.mesh_bounds, lod_index);
                    processed = processed.saturating_add(1);
                }

                if let Some(hints) = streaming_hints.as_mut() {
                    if proxy_valid {
                        let cell_size = worker_tuning.cell_size.max(f32::EPSILON);
                        let cell_size_vec = Vec3::splat(cell_size * 0.5);
                        for (key, cell) in cells.iter() {
                            if !cell.proxy_active {
                                continue;
                            }
                            let center = key.center(cell_size);
                            let size = cell_size_vec.length();
                            let distance = (stream_camera_pos - center).length();
                            let size_bias = streaming_tuning.priority_size_bias.max(0.0);
                            let distance_bias = streaming_tuning.priority_distance_bias.max(0.0);
                            let lod_bias = streaming_tuning.priority_lod_bias.max(f32::EPSILON);
                            let lod_penalty = 1.0 / lod_bias;
                            let denom = (distance + distance_bias).max(f32::EPSILON);
                            let hint_priority = (size + size_bias) * lod_penalty / denom;
                            hints
                                .entry((AssetStreamKind::Mesh, worker_tuning.hlod_proxy_mesh_id))
                                .and_modify(|req| {
                                    req.priority = req.priority.max(hint_priority);
                                    req.max_lod = req.max_lod.map(|l| l.min(0)).or(Some(0));
                                })
                                .or_insert(AssetStreamingRequest {
                                    id: worker_tuning.hlod_proxy_mesh_id,
                                    kind: AssetStreamKind::Mesh,
                                    priority: hint_priority,
                                    max_lod: Some(0),
                                    force_low_res: false,
                                });
                            hints
                                .entry((
                                    AssetStreamKind::Material,
                                    worker_tuning.hlod_proxy_material_id,
                                ))
                                .and_modify(|req| req.priority = req.priority.max(hint_priority))
                                .or_insert(AssetStreamingRequest {
                                    id: worker_tuning.hlod_proxy_material_id,
                                    kind: AssetStreamKind::Material,
                                    priority: hint_priority,
                                    max_lod: None,
                                    force_low_res: false,
                                });
                        }
                    }
                }

                if streaming_queue.is_empty() {
                    last_stream_frame = frame_index;
                }
            }
        }

        drop(mesh_aabb_map);

        if !pending_removed_proxies.is_empty() {
            objects_remove.extend(pending_removed_proxies.drain(..));
        }

        let mut lights_upsert = Vec::new();
        let mut lights_remove = std::mem::take(&mut pending_removed_lights);
        for (&entity, entry) in lights.iter_mut() {
            let current_transform = entry.transform;
            let needs_update = !entry.last_sent_present
                || !transform_approx_eq(
                    &current_transform,
                    &entry.last_sent_transform,
                    transform_epsilon,
                    rotation_epsilon,
                )
                || entry.light != entry.last_sent_light;
            if needs_update {
                lights_upsert.push(RenderLightDelta {
                    id: entity.index() as usize,
                    transform: current_transform,
                    color: entry.light.color.into(),
                    intensity: entry.light.intensity,
                    light_type: entry.light.light_type,
                });
                entry.last_sent_present = true;
                entry.last_sent_transform = current_transform;
                entry.last_sent_light = entry.light;
            }
        }

        let send_camera = camera_changed || !has_sent_any;
        let send_config = config_dirty || !has_sent_any;
        last_render_camera_transform = render_camera_transform;
        last_camera_component = camera_component;
        has_previous_camera = true;

        objects_remove.sort_unstable();
        objects_remove.dedup();
        lights_remove.sort_unstable();
        lights_remove.dedup();

        let mut render_delta = RenderDelta {
            full: !has_sent_any || force_full,
            objects_upsert,
            objects_remove,
            lights_upsert,
            lights_remove,
            camera: None,
            render_config: None,
            render_graph: None,
            gizmo: None,
            streaming_requests: None,
        };

        if send_camera {
            render_delta.camera = Some(RenderCameraDelta {
                transform: render_camera_transform,
                camera: camera_component,
            });
        }
        if send_config {
            render_delta.render_config = Some(render_config);
            render_delta.render_graph = Some(render_graph.clone());
        }
        config_dirty = false;

        if send_streaming {
            if request_budget == 0 {
                streaming_dirty = false;
            } else {
                let mut streaming_plan = streaming_hints
                    .unwrap_or_default()
                    .into_values()
                    .collect::<Vec<_>>();
                if request_budget > 0 && streaming_plan.len() > request_budget {
                    streaming_plan.sort_by(|a, b| {
                        b.priority
                            .partial_cmp(&a.priority)
                            .unwrap_or(Ordering::Equal)
                    });
                    streaming_plan.truncate(request_budget);
                }
                if streaming_plan.is_empty() && (!cells.is_empty() || !streaming_queue.is_empty()) {
                    streaming_dirty = true;
                } else {
                    render_delta.streaming_requests = Some(streaming_plan);
                }
                if streaming_queue.is_empty() {
                    last_stream_frame = frame_index;
                    if render_delta.streaming_requests.is_some() {
                        streaming_dirty = false;
                    }
                }
            }
        }

        if result_tx
            .send(RenderResult {
                render_delta,
                visible_object_count,
            })
            .is_err()
        {
            break;
        }
        has_sent_any = true;
        force_full = false;
    }
}

//================================================================================
// The Bevy System
//================================================================================

/// Collects all data required for rendering, performs culling and LOD selection.
#[allow(clippy::too_many_arguments)]
pub fn render_data_system(
    mut worker_state: Local<RenderWorkerState>,
    resources: RenderSystemResources,
    mut render_packet: ResMut<RenderPacket>,
    mut render_object_count: ResMut<RenderObjectCount>,
    camera_query: Query<(Ref<BevyCamera>, Ref<BevyTransform>), With<BevyActiveCamera>>,
    mut object_queries: ParamSet<(
        Query<(Entity, &BevyTransform, &BevyMeshRenderer)>,
        Query<
            (Entity, &BevyTransform, &BevyMeshRenderer),
            Or<(
                Added<BevyTransform>,
                Added<BevyMeshRenderer>,
                Changed<BevyTransform>,
                Changed<BevyMeshRenderer>,
            )>,
        >,
    )>,
    mut removed_mesh_renderers: RemovedComponents<BevyMeshRenderer>,
    mut removed_transforms: RemovedComponents<BevyTransform>,
    mut light_queries: ParamSet<(
        Query<(Entity, &BevyTransform, &BevyLight)>,
        Query<
            (Entity, &BevyTransform, &BevyLight),
            Or<(
                Added<BevyTransform>,
                Added<BevyLight>,
                Changed<BevyTransform>,
                Changed<BevyLight>,
            )>,
        >,
    )>,
    mut removed_lights: RemovedComponents<BevyLight>,
) {
    let (runtime_config, asset_server) = match (
        resources.runtime_config.as_ref(),
        resources.asset_server.as_ref(),
    ) {
        (Some(config), Some(server)) => (config.0, server),
        _ => {
            warn!(
                "Required resources (RuntimeConfig, AssetServer) not found. Skipping render data extraction."
            );
            return;
        }
    };

    let _profiling_scope =
        ProfilingScope::new(resources.runtime_profiling.as_ref().map(|p| p.0.clone()));

    let worker_tuning = resources
        .worker_tuning
        .as_ref()
        .map(|t| t.0)
        .unwrap_or_default();
    let change_capacity = worker_tuning.change_queue_capacity.max(1);
    let capacity_changed = worker_state
        .last_worker_tuning
        .map(|t| t.change_queue_capacity != worker_tuning.change_queue_capacity)
        .unwrap_or(false);

    if worker_state.worker.is_none() || capacity_changed {
        let mesh_aabb_map = asset_server.0.lock().mesh_aabb_map.clone();
        worker_state.worker = Some(spawn_render_worker(mesh_aabb_map, change_capacity));
        worker_state.camera_sent = false;
        worker_state.config_sent = false;
        worker_state.frame_index = 0;
        worker_state.needs_full_sync = true;
        worker_state.last_render_config = None;
        worker_state.last_render_graph_version = None;
        worker_state.last_streaming_tuning = None;
        worker_state.last_lod_tuning = None;
        worker_state.last_worker_tuning = None;
    }

    let mut worker = match worker_state.worker.take() {
        Some(worker) => worker,
        None => return,
    };

    let render_graph = resources
        .render_graph
        .as_ref()
        .map(|r| r.0.clone())
        .unwrap_or_else(default_graph_spec);
    let streaming_tuning = resources
        .streaming_tuning
        .as_ref()
        .map(|t| t.0)
        .unwrap_or_default();
    let lod_tuning = resources
        .lod_tuning
        .as_ref()
        .map(|t| t.0)
        .unwrap_or_default();
    let mut render_config = runtime_config.render_config;
    render_config.gpu_lod0_distance = lod_tuning.lod0_distance;
    render_config.gpu_lod1_distance = lod_tuning.lod1_distance;
    render_config.gpu_lod2_distance = lod_tuning.lod2_distance;

    let mut config_changed = !worker_state.config_sent;
    if worker_state
        .last_render_config
        .map_or(true, |cfg| cfg != render_config)
    {
        config_changed = true;
    }
    if worker_state
        .last_render_graph_version
        .map_or(true, |ver| ver != render_graph.version)
    {
        config_changed = true;
    }
    if worker_state
        .last_streaming_tuning
        .map_or(true, |tuning| tuning != streaming_tuning)
    {
        config_changed = true;
    }
    if worker_state
        .last_lod_tuning
        .map_or(true, |tuning| tuning != lod_tuning)
    {
        config_changed = true;
    }
    if worker_state
        .last_worker_tuning
        .map_or(true, |tuning| tuning != worker_tuning)
    {
        config_changed = true;
    }

    let mut direct_delta = RenderDelta::default();
    let mut send_failed = worker.change_tx.is_full();
    if send_failed {
        worker_state.needs_full_sync = true;
        worker_state.camera_sent = false;
        worker_state.config_sent = false;
    }

    if config_changed {
        if !send_failed
            && !try_send_change(
                &mut *worker_state,
                &worker,
                RenderChange::UpdateConfig {
                    render_config,
                    render_graph: render_graph.clone(),
                    streaming_tuning,
                    lod_tuning,
                    worker_tuning,
                },
                "config update",
            )
        {
            send_failed = true;
        }
        direct_delta.render_config = Some(render_config);
        direct_delta.render_graph = Some(render_graph.clone());
        if !send_failed {
            worker_state.config_sent = true;
            worker_state.last_render_config = Some(render_config);
            worker_state.last_render_graph_version = Some(render_graph.version);
            worker_state.last_streaming_tuning = Some(streaming_tuning);
            worker_state.last_lod_tuning = Some(lod_tuning);
            worker_state.last_worker_tuning = Some(worker_tuning);
        } else {
            worker_state.config_sent = false;
        }
    }

    let gizmo_data = resources
        .gizmo_state
        .as_ref()
        .map(|state| state.0)
        .unwrap_or_default();
    if worker_state
        .last_gizmo
        .map_or(true, |last| last != gizmo_data)
    {
        direct_delta.gizmo = Some(gizmo_data);
        worker_state.last_gizmo = Some(gizmo_data);
    }

    let mut camera_available = false;
    if let Ok((camera, transform)) = camera_query.single() {
        camera_available = true;
        let camera_changed =
            camera.is_changed() || transform.is_changed() || !worker_state.camera_sent;
        if camera_changed {
            if !send_failed
                && !try_send_change(
                    &mut *worker_state,
                    &worker,
                    RenderChange::UpdateCamera {
                        camera: (*camera).0,
                        transform: (*transform).0,
                    },
                    "camera update",
                )
            {
                send_failed = true;
            }
            direct_delta.camera = Some(RenderCameraDelta {
                transform: (*transform).0,
                camera: (*camera).0,
            });
            if !send_failed {
                worker_state.camera_sent = true;
            } else {
                worker_state.camera_sent = false;
            }
        }
    } else {
        warn!("No active camera found in scene for rendering.");
        worker_state.camera_sent = false;
    }

    let full_sync = worker_state.needs_full_sync;

    if !send_failed {
        let removed_mesh_count = removed_mesh_renderers.len();
        let removed_light_count = removed_lights.len();
        let removed_transform_count = removed_transforms.len();

        let mut removed_objects =
            Vec::with_capacity(removed_mesh_count.saturating_add(removed_transform_count));
        let mut removed_light_entities =
            Vec::with_capacity(removed_light_count.saturating_add(removed_transform_count));

        for entity in removed_mesh_renderers.read() {
            removed_objects.push(entity);
        }
        for entity in removed_lights.read() {
            removed_light_entities.push(entity);
        }
        for entity in removed_transforms.read() {
            removed_objects.push(entity);
            removed_light_entities.push(entity);
        }

        if !full_sync
            && !removed_objects.is_empty()
            && !try_send_change(
                &mut *worker_state,
                &worker,
                RenderChange::RemoveObjects {
                    entities: removed_objects,
                },
                "remove objects",
            )
        {
            send_failed = true;
        }

        if !full_sync
            && !removed_light_entities.is_empty()
            && !try_send_change(
                &mut *worker_state,
                &worker,
                RenderChange::RemoveLights {
                    entities: removed_light_entities,
                },
                "remove lights",
            )
        {
            send_failed = true;
        }

        if !send_failed && worker.change_tx.is_full() {
            worker_state.needs_full_sync = true;
            worker_state.camera_sent = false;
            worker_state.config_sent = false;
            send_failed = true;
        }

        if !send_failed {
            let object_updates = if full_sync {
                let objects_query = object_queries.p0();
                let objects_iter = objects_query.iter();
                let (object_count, _) = objects_iter.size_hint();
                let mut updates = Vec::with_capacity(object_count);
                for (entity, transform, mesh_renderer) in objects_iter {
                    updates.push(RenderObjectUpdate {
                        entity,
                        transform: transform.0,
                        mesh_renderer: mesh_renderer.0,
                    });
                }
                updates
            } else {
                let objects_query = object_queries.p1();
                let objects_iter = objects_query.iter();
                let (object_count, _) = objects_iter.size_hint();
                let mut updates = Vec::with_capacity(object_count);
                for (entity, transform, mesh_renderer) in objects_iter {
                    updates.push(RenderObjectUpdate {
                        entity,
                        transform: transform.0,
                        mesh_renderer: mesh_renderer.0,
                    });
                }
                updates
            };

            if !object_updates.is_empty()
                && !try_send_change(
                    &mut *worker_state,
                    &worker,
                    RenderChange::UpsertObjects {
                        objects: object_updates,
                    },
                    "upsert objects",
                )
            {
                send_failed = true;
            }
        }

        if !send_failed && worker.change_tx.is_full() {
            worker_state.needs_full_sync = true;
            worker_state.camera_sent = false;
            worker_state.config_sent = false;
            send_failed = true;
        }

        if !send_failed {
            let light_updates = if full_sync {
                let lights_query = light_queries.p0();
                let lights_iter = lights_query.iter();
                let (light_count, _) = lights_iter.size_hint();
                let mut updates = Vec::with_capacity(light_count);
                for (entity, transform, light) in lights_iter {
                    updates.push(RenderLightUpdate {
                        entity,
                        transform: transform.0,
                        light: light.0,
                    });
                }
                updates
            } else {
                let lights_query = light_queries.p1();
                let lights_iter = lights_query.iter();
                let (light_count, _) = lights_iter.size_hint();
                let mut updates = Vec::with_capacity(light_count);
                for (entity, transform, light) in lights_iter {
                    updates.push(RenderLightUpdate {
                        entity,
                        transform: transform.0,
                        light: light.0,
                    });
                }
                updates
            };

            if !light_updates.is_empty()
                && !try_send_change(
                    &mut *worker_state,
                    &worker,
                    RenderChange::UpsertLights {
                        lights: light_updates,
                    },
                    "upsert lights",
                )
            {
                send_failed = true;
            }
        }

        if full_sync && !send_failed {
            worker_state.needs_full_sync = false;
        }
    }

    let mut latest_result = None;
    while let Ok(result) = worker.result_rx.try_recv() {
        latest_result = Some(result);
    }

    let direct_delta = if direct_delta.is_empty() {
        None
    } else {
        Some(direct_delta)
    };

    if let Some(result) = latest_result {
        worker.in_flight = false;
        let mut render_delta = result.render_delta;
        render_delta.camera = None;
        render_delta.render_config = None;
        render_delta.render_graph = None;
        if let Some(requests) = render_delta.streaming_requests.as_deref() {
            asset_server.0.lock().publish_streaming_plan(requests);
        }
        if let Some(direct) = direct_delta {
            render_delta.merge_from(direct);
        }
        render_packet.0 = Some(render_delta);
        render_object_count.0 = result.visible_object_count;
    } else if let Some(direct) = direct_delta {
        render_packet.0 = Some(direct);
    }

    if camera_available && !worker.in_flight && !send_failed {
        let frame_index = worker_state.frame_index;
        worker_state.frame_index = worker_state.frame_index.wrapping_add(1);

        if !try_send_change(
            &mut *worker_state,
            &worker,
            RenderChange::RequestFrame { frame_index },
            "frame request",
        ) {
            send_failed = true;
        }
        if !send_failed {
            worker.in_flight = true;
        }
    }

    worker_state.worker = Some(worker);
}
