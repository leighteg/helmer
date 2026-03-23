use crate::components::{
    ActiveCamera, Camera, Light, MeshRenderer, SkinnedMeshRenderer, Transform,
};
use crate::ecs::{
    ecs_core::{ECSCore, Entity},
    system::System,
};
use glam::{Mat3, Mat4, Vec3, Vec4, Vec4Swizzles};
use hashbrown::{HashMap, HashSet};
use helmer_animation::{Animator, Pose, write_skin_palette};
use helmer_asset::runtime::asset_server::AssetServer;
use helmer_render::graphics::{
    common::{
        config::{RenderConfig, SkinningMode},
        graph::RenderGraphSpec,
        renderer::{
            Aabb, AssetStreamKind, AssetStreamingRequest, RenderCameraDelta, RenderDelta,
            RenderLightDelta, RenderObjectDelta, Vertex,
        },
    },
    render_graphs::default_graph_spec,
};
use helmer_render::runtime::RuntimeConfig;
use helmer_window::runtime::input_manager::InputManager;
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::warn;

/// A geometric frustum defined by 6 planes, used for culling.
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Creates a frustum from a combined view-projection matrix using a more robust
    /// row-based extraction method compatible with glam's column-major matrices.
    pub fn from_matrix(mat: Mat4) -> Self {
        let mut planes = [Vec4::ZERO; 6];

        // Extract planes from rows (equivalent to columns of the transpose)
        let row0 = mat.row(0);
        let row1 = mat.row(1);
        let row2 = mat.row(2);
        let row3 = mat.row(3);

        planes[0] = row3 + row0; // Left
        planes[1] = row3 - row0; // Right
        planes[2] = row3 + row1; // Bottom
        planes[3] = row3 - row1; // Top
        planes[4] = row3 + row2; // Near
        planes[5] = row3 - row2; // Far

        // Normalize the plane equations
        for plane in &mut planes {
            *plane /= plane.xyz().length();
        }
        Self { planes }
    }

    /// Checks if a transformed AABB intersects the frustum using a robust and
    /// more accurate Separating Axis Theorem (SAT) based approach.
    pub fn intersects_aabb(&self, aabb: &Aabb, transform: &Transform) -> bool {
        let model_matrix = Mat4::from_scale_rotation_translation(
            transform.scale,
            transform.rotation,
            transform.position,
        );

        let center_world = model_matrix.transform_point3(aabb.center());
        let extents_local = aabb.extents();

        // Calculate the world-space extents (the projection of the AABB's half-diagonals)
        let axis_x = model_matrix.col(0).xyz() * extents_local.x;
        let axis_y = model_matrix.col(1).xyz() * extents_local.y;
        let axis_z = model_matrix.col(2).xyz() * extents_local.z;

        for plane in &self.planes {
            let normal = plane.xyz();

            // Project the AABB's radius onto the plane normal.
            let r = axis_x.dot(normal).abs() + axis_y.dot(normal).abs() + axis_z.dot(normal).abs();

            // Calculate the signed distance from the AABB's center to the plane.
            let s = plane.dot(center_world.extend(1.0));

            // If the center is farther from the plane than its projected radius (on the negative side),
            // then the entire box is outside the frustum and can be culled.
            if s < -r {
                return false;
            }
        }

        true // The AABB intersects all planes, so it's visible.
    }
}

/// An intermediate struct holding one frame's state. Now includes the selected LOD index.
#[derive(Clone)]
pub struct ExtractedState {
    pub objects: HashMap<Entity, (Transform, usize, usize, bool, usize)>, // Transform, mesh_id, material_id, casts_shadow, lod_index
    pub lights: HashMap<Entity, (Transform, Light)>,
    pub camera_transform: Transform,
    pub camera_component: Camera,
    pub render_config: RenderConfig,
    pub render_graph: RenderGraphSpec,
}

impl Default for ExtractedState {
    fn default() -> Self {
        Self {
            objects: HashMap::new(),
            lights: HashMap::new(),
            camera_transform: Transform::default(),
            camera_component: Camera::default(),
            render_config: RenderConfig::default(),
            render_graph: default_graph_spec(),
        }
    }
}

#[derive(Default)]
pub struct RenderPacket(pub Option<RenderDelta>);

#[derive(Clone)]
struct SkinningEntry {
    offset: usize,
    count: usize,
    pose: Pose,
    globals: Vec<Mat4>,
}

#[derive(Clone, Copy)]
struct CpuSkinnedMesh {
    mesh_id: usize,
    base_mesh_id: usize,
}

#[derive(Default)]
struct SkinningState {
    palette: Vec<Mat4>,
    entries: HashMap<Entity, SkinningEntry>,
    free_ranges: Vec<std::ops::Range<usize>>,
    cpu_meshes: HashMap<Entity, CpuSkinnedMesh>,
    mode: SkinningMode,
    full_sync_requested: bool,
    has_gpu_skinning: bool,
    palette_dirty: bool,
}

impl SkinningState {
    fn begin_frame(&mut self, mode: SkinningMode) {
        if self.mode != mode {
            self.mode = mode;
            self.full_sync_requested = true;
        }
        self.has_gpu_skinning = !matches!(mode, SkinningMode::Cpu);
        self.palette_dirty = false;
    }

    fn take_full_sync(&mut self) -> bool {
        let flag = self.full_sync_requested;
        self.full_sync_requested = false;
        flag
    }

    fn ensure_entry(
        &mut self,
        entity: Entity,
        joint_count: usize,
        pose: Pose,
    ) -> &mut SkinningEntry {
        if !self.entries.contains_key(&entity) {
            let offset = self.allocate_range(joint_count);
            self.entries.insert(
                entity,
                SkinningEntry {
                    offset,
                    count: joint_count,
                    pose,
                    globals: vec![Mat4::IDENTITY; joint_count],
                },
            );
            return self.entries.get_mut(&entity).unwrap();
        }

        let needs_resize = self.entries.get(&entity).unwrap().count != joint_count;
        if needs_resize {
            let (old_offset, old_count) = {
                let e = self.entries.get(&entity).unwrap();
                (e.offset, e.count)
            };
            self.free_range(old_offset, old_count);
            let new_offset = self.allocate_range(joint_count);
            let entry = self.entries.get_mut(&entity).unwrap();
            entry.offset = new_offset;
            entry.count = joint_count;
            entry.pose = pose;
            entry.globals.resize(joint_count, Mat4::IDENTITY);
            self.full_sync_requested = true;
        }

        self.entries.get_mut(&entity).unwrap()
    }

    fn ensure_palette_capacity(&mut self, required: usize, growth: f32) {
        if required <= self.palette.len() {
            return;
        }
        let mut new_len = if self.palette.is_empty() {
            0
        } else {
            self.palette.len()
        };
        while new_len < required {
            let grow = (new_len as f32 * growth.max(1.0)).ceil() as usize;
            new_len = grow.max(required).max(new_len + 1);
        }
        self.palette.resize(new_len, Mat4::IDENTITY);
        self.palette_dirty = true;
    }

    fn allocate_range(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        if let Some((idx, range)) = self
            .free_ranges
            .iter()
            .enumerate()
            .find(|(_, range)| range.len() >= count)
            .map(|(idx, range)| (idx, range.clone()))
        {
            self.free_ranges.swap_remove(idx);
            let start = range.start;
            let remaining = range.len().saturating_sub(count);
            if remaining > 0 {
                self.free_ranges.push((start + count)..range.end);
            }
            return start;
        }
        let offset = self.palette.len();
        self.palette.resize(offset + count, Mat4::IDENTITY);
        offset
    }

    fn free_range(&mut self, offset: usize, count: usize) {
        if count == 0 {
            return;
        }
        self.free_ranges.push(offset..(offset + count));
        self.merge_free_ranges();
    }

    fn merge_free_ranges(&mut self) {
        if self.free_ranges.len() < 2 {
            return;
        }
        self.free_ranges.sort_by_key(|range| range.start);
        let mut merged = Vec::with_capacity(self.free_ranges.len());
        let mut current = self.free_ranges[0].clone();
        for range in self.free_ranges.iter().skip(1) {
            if range.start <= current.end {
                current.end = current.end.max(range.end);
            } else {
                merged.push(current);
                current = range.clone();
            }
        }
        merged.push(current);
        self.free_ranges = merged;
    }

    fn cleanup_missing(&mut self, seen: &HashSet<Entity>) {
        let mut removed = Vec::new();
        for id in self.entries.keys() {
            if !seen.contains(id) {
                removed.push(*id);
            }
        }
        if removed.is_empty() {
            return;
        }
        for id in removed {
            if let Some(entry) = self.entries.remove(&id) {
                self.free_range(entry.offset, entry.count);
                self.full_sync_requested = true;
            }
            self.cpu_meshes.remove(&id);
        }
    }

    fn skin_params_for(&self, entity: Entity) -> (u32, u32) {
        if matches!(self.mode, SkinningMode::Cpu) {
            return (0, 0);
        }
        if let Some(entry) = self.entries.get(&entity) {
            return (
                entry.offset.min(u32::MAX as usize) as u32,
                entry.count.min(u32::MAX as usize) as u32,
            );
        }
        (0, 0)
    }
}

fn skin_vertices_cpu(vertices: &[Vertex], palette: &[Mat4]) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(vertices.len());
    if palette.is_empty() {
        return vertices.to_vec();
    }
    for v in vertices {
        let pos = Vec3::from(v.position);
        let norm = Vec3::from(v.normal);
        let tan = Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]);
        let mut skinned_pos = Vec3::ZERO;
        let mut skinned_norm = Vec3::ZERO;
        let mut skinned_tan = Vec3::ZERO;

        for i in 0..4 {
            let weight = v.weights[i];
            if weight <= 0.0 {
                continue;
            }
            let mut joint = v.joints[i] as usize;
            if joint >= palette.len() {
                joint = palette.len().saturating_sub(1);
            }
            let m = palette[joint];
            let m3 = Mat3::from_cols(m.x_axis.xyz(), m.y_axis.xyz(), m.z_axis.xyz());
            let p = (m * Vec4::new(pos.x, pos.y, pos.z, 1.0)).xyz();
            skinned_pos += weight * p;
            skinned_norm += weight * (m3 * norm);
            skinned_tan += weight * (m3 * tan);
        }

        let skinned_norm = skinned_norm.normalize_or_zero();
        let skinned_tan = skinned_tan.normalize_or_zero();
        out.push(Vertex::with_skinning(
            [skinned_pos.x, skinned_pos.y, skinned_pos.z],
            [skinned_norm.x, skinned_norm.y, skinned_norm.z],
            v.tex_coord,
            [skinned_tan.x, skinned_tan.y, skinned_tan.z, v.tangent[3]],
            v.joints,
            v.weights,
        ));
    }
    out
}

#[derive(Clone)]
pub struct RenderGraphResource(pub RenderGraphSpec);

impl Default for RenderGraphResource {
    fn default() -> Self {
        Self(default_graph_spec())
    }
}

/// Collects all data required for rendering, performs culling and LOD selection.
pub struct RenderDataSystem {
    previous_state: Option<ExtractedState>,
    skinning: SkinningState,
}

impl RenderDataSystem {
    pub fn new() -> Self {
        Self {
            previous_state: None,
            skinning: SkinningState::default(),
        }
    }
}

impl Default for RenderDataSystem {
    fn default() -> Self {
        Self::new()
    }
}

// Squared distances for LOD transitions.
const LOD_THRESHOLDS: [f32; 3] = [25.0 * 25.0, 60.0 * 60.0, 120.0 * 120.0];

fn transform_changed(a: &Transform, b: &Transform, epsilon: f32, rotation_epsilon: f32) -> bool {
    !a.position.abs_diff_eq(b.position, epsilon)
        || !a.scale.abs_diff_eq(b.scale, epsilon)
        || a.rotation.dot(b.rotation).abs() < 1.0 - rotation_epsilon
}

fn camera_component_changed(a: &Camera, b: &Camera) -> bool {
    a.fov_y_rad != b.fov_y_rad
        || a.aspect_ratio != b.aspect_ratio
        || a.near_plane != b.near_plane
        || a.far_plane != b.far_plane
}

impl System for RenderDataSystem {
    fn name(&self) -> &str {
        "RenderDataSystem"
    }

    fn run(&mut self, dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let streaming_hints: Vec<AssetStreamingRequest>;
        let runtime_config = match ecs.get_resource::<RuntimeConfig>() {
            Some(config) => config.clone(),
            None => {
                warn!("RuntimeConfig resource not found");
                return;
            }
        };
        self.skinning
            .begin_frame(runtime_config.render_config.skinning_mode);
        let growth = runtime_config.render_config.skin_palette_growth.max(1.0);
        let mut seen = HashSet::new();
        let mut vertex_budget = runtime_config.render_config.cpu_skinning_vertex_budget as usize;
        let unlimited_budget = vertex_budget == 0;
        let asset_server = ecs.get_resource::<Arc<Mutex<AssetServer>>>().cloned();

        ecs.component_pool
            .query_mut_for_each::<(Animator, SkinnedMeshRenderer), _>(|entity, (anim, skinned)| {
                seen.insert(entity);
                let skeleton = &skinned.skin.skeleton;
                let joint_count = skeleton.joint_count();
                if joint_count == 0 {
                    return;
                }

                let (offset, count) = {
                    let entry = self.skinning.ensure_entry(
                        entity,
                        joint_count,
                        Pose::from_skeleton(skeleton),
                    );
                    (entry.offset, entry.count)
                };

                self.skinning
                    .ensure_palette_capacity(offset + count, growth);

                {
                    let SkinningState {
                        entries,
                        palette,
                        palette_dirty,
                        ..
                    } = &mut self.skinning;
                    let entry = entries.get_mut(&entity).unwrap();
                    anim.evaluate(skeleton, dt, &mut entry.pose);
                    let palette_slice = &mut palette[offset..(offset + count)];
                    write_skin_palette(
                        skeleton,
                        &entry.pose.locals,
                        &mut entry.globals,
                        palette_slice,
                    );
                    *palette_dirty = true;
                }

                if matches!(self.skinning.mode, SkinningMode::Cpu) {
                    let Some(asset_server) = asset_server.as_ref() else {
                        return;
                    };
                    let base_mesh_id = skinned.mesh_id;

                    if !self.skinning.cpu_meshes.contains_key(&entity) {
                        self.skinning.full_sync_requested = true;
                        let palette_slice = &self.skinning.palette[offset..(offset + count)];
                        let mesh = asset_server.lock().get_mesh(base_mesh_id);
                        let new_entry = if let Some(mesh) = mesh {
                            if let Some(lod) = mesh.lods.read().first() {
                                let skinned_vertices =
                                    skin_vertices_cpu(&lod.vertices, palette_slice);
                                let handle = asset_server
                                    .lock()
                                    .add_mesh(skinned_vertices, lod.indices.to_vec());
                                CpuSkinnedMesh {
                                    mesh_id: handle.id,
                                    base_mesh_id,
                                }
                            } else {
                                CpuSkinnedMesh {
                                    mesh_id: base_mesh_id,
                                    base_mesh_id,
                                }
                            }
                        } else {
                            CpuSkinnedMesh {
                                mesh_id: base_mesh_id,
                                base_mesh_id,
                            }
                        };
                        self.skinning.cpu_meshes.insert(entity, new_entry);
                    }

                    let (current_cpu_mesh_id, base_changed) = {
                        let cpu_entry = self.skinning.cpu_meshes.get_mut(&entity).unwrap();
                        let changed = cpu_entry.base_mesh_id != base_mesh_id;
                        if changed {
                            cpu_entry.base_mesh_id = base_mesh_id;
                        }
                        (cpu_entry.mesh_id, changed)
                    };
                    if base_changed {
                        self.skinning.full_sync_requested = true;
                    }

                    if current_cpu_mesh_id == base_mesh_id {
                        return;
                    }
                    let mesh = asset_server.lock().get_mesh(base_mesh_id);
                    let Some(mesh) = mesh else {
                        return;
                    };
                    let lods = mesh.lods.read();
                    let Some(lod) = lods.first() else {
                        return;
                    };
                    let vertex_count = lod.vertices.len();
                    if !unlimited_budget {
                        if vertex_budget < vertex_count {
                            return;
                        }
                        vertex_budget = vertex_budget.saturating_sub(vertex_count);
                    }
                    let palette_slice = &self.skinning.palette[offset..(offset + count)];
                    let skinned_vertices = skin_vertices_cpu(&lod.vertices, palette_slice);
                    asset_server.lock().update_mesh(
                        current_cpu_mesh_id,
                        skinned_vertices,
                        lod.indices.to_vec(),
                    );
                }
            });

        ecs.component_pool
            .query_for_each::<(SkinnedMeshRenderer, Transform), _>(|entity, (skinned, _)| {
                if seen.contains(&entity) {
                    return;
                }
                seen.insert(entity);
                let skeleton = &skinned.skin.skeleton;
                let joint_count = skeleton.joint_count();
                if joint_count == 0 {
                    return;
                }

                let (offset, count) = {
                    let entry = self.skinning.ensure_entry(
                        entity,
                        joint_count,
                        Pose::from_skeleton(skeleton),
                    );
                    (entry.offset, entry.count)
                };

                self.skinning
                    .ensure_palette_capacity(offset + count, growth);

                {
                    let SkinningState {
                        entries,
                        palette,
                        palette_dirty,
                        ..
                    } = &mut self.skinning;
                    let entry = entries.get_mut(&entity).unwrap();
                    entry.pose.reset_to_bind(skeleton);
                    let palette_slice = &mut palette[offset..(offset + count)];
                    write_skin_palette(
                        skeleton,
                        &entry.pose.locals,
                        &mut entry.globals,
                        palette_slice,
                    );
                    *palette_dirty = true;
                }

                if matches!(self.skinning.mode, SkinningMode::Cpu) {
                    let Some(asset_server) = asset_server.as_ref() else {
                        return;
                    };
                    let base_mesh_id = skinned.mesh_id;

                    if !self.skinning.cpu_meshes.contains_key(&entity) {
                        self.skinning.full_sync_requested = true;
                        let palette_slice = &self.skinning.palette[offset..(offset + count)];
                        let mesh = asset_server.lock().get_mesh(base_mesh_id);
                        let new_entry = if let Some(mesh) = mesh {
                            if let Some(lod) = mesh.lods.read().first() {
                                let skinned_vertices =
                                    skin_vertices_cpu(&lod.vertices, palette_slice);
                                let handle = asset_server
                                    .lock()
                                    .add_mesh(skinned_vertices, lod.indices.to_vec());
                                CpuSkinnedMesh {
                                    mesh_id: handle.id,
                                    base_mesh_id,
                                }
                            } else {
                                CpuSkinnedMesh {
                                    mesh_id: base_mesh_id,
                                    base_mesh_id,
                                }
                            }
                        } else {
                            CpuSkinnedMesh {
                                mesh_id: base_mesh_id,
                                base_mesh_id,
                            }
                        };
                        self.skinning.cpu_meshes.insert(entity, new_entry);
                    }

                    let (current_cpu_mesh_id, base_changed) = {
                        let cpu_entry = self.skinning.cpu_meshes.get_mut(&entity).unwrap();
                        let changed = cpu_entry.base_mesh_id != base_mesh_id;
                        if changed {
                            cpu_entry.base_mesh_id = base_mesh_id;
                        }
                        (cpu_entry.mesh_id, changed)
                    };
                    if base_changed {
                        self.skinning.full_sync_requested = true;
                    }

                    if current_cpu_mesh_id == base_mesh_id {
                        return;
                    }
                    let mesh = asset_server.lock().get_mesh(base_mesh_id);
                    let Some(mesh) = mesh else {
                        return;
                    };
                    let lods = mesh.lods.read();
                    let Some(lod) = lods.first() else {
                        return;
                    };
                    let vertex_count = lod.vertices.len();
                    if !unlimited_budget {
                        if vertex_budget < vertex_count {
                            return;
                        }
                        vertex_budget = vertex_budget.saturating_sub(vertex_count);
                    }
                    let palette_slice = &self.skinning.palette[offset..(offset + count)];
                    let skinned_vertices = skin_vertices_cpu(&lod.vertices, palette_slice);
                    asset_server.lock().update_mesh(
                        current_cpu_mesh_id,
                        skinned_vertices,
                        lod.indices.to_vec(),
                    );
                }
            });

        self.skinning.cleanup_missing(&seen);
        if self.skinning.take_full_sync() {
            self.previous_state = None;
        }
        let transform_epsilon = runtime_config.render_config.transform_epsilon.max(0.0);
        let rotation_epsilon = runtime_config
            .render_config
            .rotation_epsilon
            .clamp(0.0, 1.0);
        let gpu_driven = runtime_config.render_config.gpu_driven;

        let current_state = {
            let mesh_aabb_map = match ecs.get_resource::<Arc<Mutex<AssetServer>>>() {
                Some(map) => map.lock().mesh_aabb_map.read().0.clone(),
                None => {
                    warn!("MeshAabbMap resource not found, culling will be skipped.");
                    return;
                }
            };

            let (camera_component, camera_transform) = if let Some((cam, trans, _)) = ecs
                .component_pool
                .query_exact::<(Camera, Transform, ActiveCamera)>()
                .next()
            {
                (*cam, *trans)
            } else {
                warn!("No active camera found in scene for rendering!");
                return;
            };

            let view_matrix = Mat4::look_at_rh(
                camera_transform.position,
                camera_transform.position + camera_transform.forward(),
                camera_transform.up(),
            );
            let projection_matrix = Mat4::perspective_infinite_reverse_rh(
                camera_component.fov_y_rad,
                camera_component.aspect_ratio,
                camera_component.near_plane,
            );
            let frustum = Frustum::from_matrix(projection_matrix * view_matrix);

            let render_graph_override = ecs.get_resource::<RenderGraphResource>().cloned();

            let mut state = ExtractedState::default();
            state.camera_component = camera_component;
            state.camera_transform = camera_transform;
            state.render_config = runtime_config.render_config;
            state.render_graph = render_graph_override
                .map(|r| r.0)
                .or_else(|| self.previous_state.as_ref().map(|s| s.render_graph.clone()))
                .unwrap_or_else(default_graph_spec);
            let mut streaming_map: HashMap<(AssetStreamKind, usize), AssetStreamingRequest> =
                HashMap::new();

            ecs.component_pool
                .query_for_each::<(Transform, MeshRenderer), _>(
                    |entity, (transform, mesh_renderer)| {
                        if !mesh_renderer.visible {
                            return;
                        }

                        if let Some(aabb) = mesh_aabb_map.get(&mesh_renderer.mesh_id) {
                            let frustum_visible = if runtime_config.render_config.frustum_culling {
                                frustum.intersects_aabb(aabb, transform)
                            } else {
                                true
                            };
                            if !frustum_visible && !runtime_config.render_config.gpu_driven {
                                return;
                            }

                            // --- LOD SELECTION ---
                            let lod_index: usize = if runtime_config.render_config.lod {
                                let model_matrix = Mat4::from_scale_rotation_translation(
                                    transform.scale,
                                    transform.rotation,
                                    transform.position,
                                );
                                let inverse_model = model_matrix.inverse();
                                let camera_pos_local =
                                    inverse_model.transform_point3(camera_transform.position);
                                let closest_point_local =
                                    camera_pos_local.clamp(aabb.min, aabb.max);
                                let distance_sq =
                                    camera_pos_local.distance_squared(closest_point_local);

                                LOD_THRESHOLDS
                                    .iter()
                                    .filter(|&&threshold| distance_sq > threshold)
                                    .count()
                            } else {
                                0
                            };

                            if frustum_visible {
                                let hint_priority = {
                                    let size = aabb.extents().length();
                                    let distance =
                                        (camera_transform.position - transform.position).length();
                                    let lod_penalty = 1.0 / (lod_index as f32 + 1.0);
                                    let shadow_boost = if mesh_renderer.casts_shadow {
                                        1.15
                                    } else {
                                        1.0
                                    };
                                    (size + 1.0) * shadow_boost * lod_penalty / (distance + 1.0)
                                };
                                streaming_map
                                    .entry((AssetStreamKind::Mesh, mesh_renderer.mesh_id))
                                    .and_modify(|req| {
                                        req.priority = req.priority.max(hint_priority);
                                        req.max_lod = req
                                            .max_lod
                                            .map(|l| l.min(lod_index))
                                            .or(Some(lod_index));
                                    })
                                    .or_insert(AssetStreamingRequest {
                                        id: mesh_renderer.mesh_id,
                                        kind: AssetStreamKind::Mesh,
                                        priority: hint_priority,
                                        max_lod: Some(lod_index),
                                        force_low_res: false,
                                    });
                                streaming_map
                                    .entry((AssetStreamKind::Material, mesh_renderer.material_id))
                                    .and_modify(|req| {
                                        req.priority = req.priority.max(hint_priority)
                                    })
                                    .or_insert(AssetStreamingRequest {
                                        id: mesh_renderer.material_id,
                                        kind: AssetStreamKind::Material,
                                        priority: hint_priority,
                                        max_lod: None,
                                        force_low_res: false,
                                    });
                            }

                            state.objects.insert(
                                entity,
                                (
                                    *transform,
                                    mesh_renderer.mesh_id,
                                    mesh_renderer.material_id,
                                    mesh_renderer.casts_shadow,
                                    lod_index,
                                ),
                            );
                        }
                    },
                );

            ecs.component_pool
                .query_for_each::<(Transform, SkinnedMeshRenderer), _>(
                    |entity, (transform, skinned)| {
                        if !skinned.visible {
                            return;
                        }
                        if state.objects.contains_key(&entity) {
                            return;
                        }

                        let base_mesh_id = skinned.mesh_id;
                        if let Some(aabb) = mesh_aabb_map.get(&base_mesh_id) {
                            let frustum_visible = if runtime_config.render_config.frustum_culling {
                                frustum.intersects_aabb(aabb, transform)
                            } else {
                                true
                            };
                            if !frustum_visible && !runtime_config.render_config.gpu_driven {
                                return;
                            }

                            let lod_index: usize = if runtime_config.render_config.lod
                                && !matches!(self.skinning.mode, SkinningMode::Cpu)
                            {
                                let model_matrix = Mat4::from_scale_rotation_translation(
                                    transform.scale,
                                    transform.rotation,
                                    transform.position,
                                );
                                let inverse_model = model_matrix.inverse();
                                let camera_pos_local =
                                    inverse_model.transform_point3(camera_transform.position);
                                let closest_point_local =
                                    camera_pos_local.clamp(aabb.min, aabb.max);
                                let distance_sq =
                                    camera_pos_local.distance_squared(closest_point_local);

                                LOD_THRESHOLDS
                                    .iter()
                                    .filter(|&&threshold| distance_sq > threshold)
                                    .count()
                            } else {
                                0
                            };

                            if frustum_visible {
                                let hint_priority = {
                                    let size = aabb.extents().length();
                                    let distance =
                                        (camera_transform.position - transform.position).length();
                                    let lod_penalty = 1.0 / (lod_index as f32 + 1.0);
                                    let shadow_boost =
                                        if skinned.casts_shadow { 1.15 } else { 1.0 };
                                    (size + 1.0) * shadow_boost * lod_penalty / (distance + 1.0)
                                };
                                streaming_map
                                    .entry((AssetStreamKind::Mesh, base_mesh_id))
                                    .and_modify(|req| {
                                        req.priority = req.priority.max(hint_priority);
                                        req.max_lod = req
                                            .max_lod
                                            .map(|l| l.min(lod_index))
                                            .or(Some(lod_index));
                                    })
                                    .or_insert(AssetStreamingRequest {
                                        id: base_mesh_id,
                                        kind: AssetStreamKind::Mesh,
                                        priority: hint_priority,
                                        max_lod: Some(lod_index),
                                        force_low_res: false,
                                    });
                                streaming_map
                                    .entry((AssetStreamKind::Material, skinned.material_id))
                                    .and_modify(|req| {
                                        req.priority = req.priority.max(hint_priority)
                                    })
                                    .or_insert(AssetStreamingRequest {
                                        id: skinned.material_id,
                                        kind: AssetStreamKind::Material,
                                        priority: hint_priority,
                                        max_lod: None,
                                        force_low_res: false,
                                    });
                            }

                            let render_mesh_id = if matches!(self.skinning.mode, SkinningMode::Cpu)
                            {
                                self.skinning
                                    .cpu_meshes
                                    .get(&entity)
                                    .map(|entry| entry.mesh_id)
                                    .unwrap_or(base_mesh_id)
                            } else {
                                base_mesh_id
                            };

                            state.objects.insert(
                                entity,
                                (
                                    *transform,
                                    render_mesh_id,
                                    skinned.material_id,
                                    skinned.casts_shadow,
                                    lod_index,
                                ),
                            );
                        }
                    },
                );

            ecs.component_pool.query_for_each::<(Transform, Light), _>(
                |entity, (transform, light)| {
                    state.lights.insert(entity, (*transform, *light));
                },
            );

            streaming_hints = streaming_map.into_values().collect();

            state
        };

        let full_snapshot = self.previous_state.is_none();
        let prev_state = self.previous_state.as_ref().unwrap_or(&current_state);

        let mut objects_upsert = Vec::new();
        if full_snapshot {
            for (&entity, &(transform, mesh_id, material_id, casts_shadow, lod_index)) in
                current_state.objects.iter()
            {
                let (skin_offset, skin_count) = self.skinning.skin_params_for(entity);
                objects_upsert.push(RenderObjectDelta {
                    id: entity,
                    transform,
                    mesh_id,
                    material_id,
                    casts_shadow,
                    lod_index,
                    skin_offset,
                    skin_count,
                });
            }
        } else {
            for (&entity, &(transform, mesh_id, material_id, casts_shadow, lod_index)) in
                current_state.objects.iter()
            {
                let changed = match prev_state.objects.get(&entity) {
                    None => true,
                    Some((prev_transform, prev_mesh, prev_material, prev_shadow, prev_lod)) => {
                        let lod_changed = lod_index != *prev_lod;
                        transform_changed(
                            &transform,
                            prev_transform,
                            transform_epsilon,
                            rotation_epsilon,
                        ) || mesh_id != *prev_mesh
                            || material_id != *prev_material
                            || casts_shadow != *prev_shadow
                            || (!gpu_driven && lod_changed)
                    }
                };
                if changed {
                    let (skin_offset, skin_count) = self.skinning.skin_params_for(entity);
                    objects_upsert.push(RenderObjectDelta {
                        id: entity,
                        transform,
                        mesh_id,
                        material_id,
                        casts_shadow,
                        lod_index,
                        skin_offset,
                        skin_count,
                    });
                }
            }
        }

        let mut objects_remove = Vec::new();
        if !full_snapshot {
            for entity in prev_state.objects.keys() {
                if !current_state.objects.contains_key(entity) {
                    objects_remove.push(*entity);
                }
            }
        }

        let mut lights_upsert = Vec::new();
        if full_snapshot {
            for (&entity, &(transform, light_component)) in current_state.lights.iter() {
                lights_upsert.push(RenderLightDelta {
                    id: entity,
                    transform,
                    color: light_component.color.into(),
                    intensity: light_component.intensity,
                    light_type: light_component.light_type,
                });
            }
        } else {
            for (&entity, &(transform, light_component)) in current_state.lights.iter() {
                let changed = match prev_state.lights.get(&entity) {
                    None => true,
                    Some((prev_transform, prev_light)) => {
                        transform_changed(
                            &transform,
                            prev_transform,
                            transform_epsilon,
                            rotation_epsilon,
                        ) || light_component != *prev_light
                    }
                };
                if changed {
                    lights_upsert.push(RenderLightDelta {
                        id: entity,
                        transform,
                        color: light_component.color.into(),
                        intensity: light_component.intensity,
                        light_type: light_component.light_type,
                    });
                }
            }
        }

        let mut lights_remove = Vec::new();
        if !full_snapshot {
            for entity in prev_state.lights.keys() {
                if !current_state.lights.contains_key(entity) {
                    lights_remove.push(*entity);
                }
            }
        }

        let mut render_delta = RenderDelta {
            full: full_snapshot,
            objects_upsert,
            objects_remove,
            lights_upsert,
            lights_remove,
            static_sprites: full_snapshot.then(|| Arc::new(Vec::new())),
            sprites: full_snapshot.then(Vec::new),
            text_2d: full_snapshot.then(Vec::new),
            ..RenderDelta::default()
        };

        let camera_changed = full_snapshot
            || transform_changed(
                &current_state.camera_transform,
                &prev_state.camera_transform,
                transform_epsilon,
                rotation_epsilon,
            )
            || camera_component_changed(
                &current_state.camera_component,
                &prev_state.camera_component,
            );
        if camera_changed {
            render_delta.camera = Some(RenderCameraDelta {
                transform: current_state.camera_transform,
                camera: current_state.camera_component,
            });
        }

        let config_changed =
            full_snapshot || current_state.render_config != prev_state.render_config;
        if config_changed {
            render_delta.render_config = Some(current_state.render_config);
        }

        let graph_changed =
            full_snapshot || current_state.render_graph.version != prev_state.render_graph.version;
        if graph_changed {
            render_delta.render_graph = Some(current_state.render_graph.clone());
        }

        if self.skinning.has_gpu_skinning
            && !self.skinning.palette.is_empty()
            && (full_snapshot || self.skinning.palette_dirty)
        {
            render_delta.skin_palette = Some(self.skinning.palette.clone());
        }

        if let Some(server) = ecs.get_resource::<Arc<Mutex<AssetServer>>>() {
            server.lock().publish_streaming_plan(&streaming_hints);
        }
        render_delta.streaming_requests = Some(streaming_hints);

        if let Some(render_packet) = ecs.get_resource_mut::<RenderPacket>() {
            render_packet.0 = Some(render_delta);
        } else {
            eprintln!(
                "[ERROR] RenderPacket resource not found in ECS. Please register it on startup."
            );
        }

        self.previous_state = Some(current_state);
    }
}
