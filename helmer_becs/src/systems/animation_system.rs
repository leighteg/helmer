use bevy_ecs::prelude::{Entity, Query, Res, ResMut, Resource};
use glam::{Mat3, Mat4, Vec3, Vec4};
use hashbrown::{HashMap, HashSet};
use helmer::{
    animation::{Pose, write_skin_palette},
    graphics::common::config::SkinningMode,
    graphics::common::renderer::Vertex,
};

use crate::{
    BevyAnimator, BevyAssetServerParam, BevyPoseOverride, BevyRuntimeConfig,
    BevySkinnedMeshRenderer, DeltaTime,
};

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

#[derive(Resource, Default)]
pub struct SkinningResource {
    palette: Vec<Mat4>,
    entries: HashMap<usize, SkinningEntry>,
    free_ranges: Vec<std::ops::Range<usize>>,
    cpu_meshes: HashMap<usize, CpuSkinnedMesh>,
    mode: SkinningMode,
    full_sync_requested: bool,
    has_gpu_skinning: bool,
}

impl SkinningResource {
    pub fn take_full_sync(&mut self) -> bool {
        let flag = self.full_sync_requested;
        self.full_sync_requested = false;
        flag
    }

    pub fn skin_params_for(&self, entity_id: usize) -> Option<(u32, u32)> {
        if matches!(self.mode, SkinningMode::Cpu) {
            return Some((0, 0));
        }
        self.entries.get(&entity_id).map(|entry| {
            (
                entry.offset.min(u32::MAX as usize) as u32,
                entry.count.min(u32::MAX as usize) as u32,
            )
        })
    }

    pub fn should_send_palette(&self) -> bool {
        self.has_gpu_skinning && !self.palette.is_empty()
    }

    pub fn palette(&self) -> &[Mat4] {
        &self.palette
    }

    pub fn cpu_mesh_id_for(&self, entity_id: usize) -> Option<usize> {
        self.cpu_meshes.get(&entity_id).map(|entry| entry.mesh_id)
    }

    fn begin_frame(&mut self, mode: SkinningMode) {
        if self.mode != mode {
            self.mode = mode;
            self.full_sync_requested = true;
        }
        self.has_gpu_skinning = !matches!(mode, SkinningMode::Cpu);
    }

    fn ensure_entry(
        &mut self,
        entity_id: usize,
        joint_count: usize,
        skeleton_pose: &Pose,
    ) -> &mut SkinningEntry {
        if !self.entries.contains_key(&entity_id) {
            let offset = self.allocate_range(joint_count);
            self.entries.insert(
                entity_id,
                SkinningEntry {
                    offset,
                    count: joint_count,
                    pose: skeleton_pose.clone(),
                    globals: vec![Mat4::IDENTITY; joint_count],
                },
            );
            // new skinned mesh entries need a render sync so skin offsets are propagated
            self.full_sync_requested = true;
        }

        let entry = self
            .entries
            .get_mut(&entity_id)
            .expect("skinning entry must exist");

        let mut resize = None;
        if entry.count != joint_count {
            resize = Some((entry.offset, entry.count));
        }
        drop(entry);

        if let Some((old_offset, old_count)) = resize {
            self.free_range(old_offset, old_count);
            let new_offset = self.allocate_range(joint_count);
            let entry = self
                .entries
                .get_mut(&entity_id)
                .expect("skinning entry must exist");
            entry.offset = new_offset;
            entry.count = joint_count;
            entry.pose = skeleton_pose.clone();
            entry.globals.resize(joint_count, Mat4::IDENTITY);
            self.full_sync_requested = true;
        }

        self.entries
            .get_mut(&entity_id)
            .expect("skinning entry must exist")
    }

    fn ensure_palette_capacity(&mut self, required: usize, growth: f32) {
        if required <= self.palette.len() {
            return;
        }
        let mut new_len = if self.palette.is_empty() {
            0usize
        } else {
            self.palette.len()
        };
        while new_len < required {
            let grow = (new_len as f32 * growth.max(1.0)).ceil() as usize;
            new_len = grow.max(required).max(new_len + 1);
        }
        self.palette.resize(new_len, Mat4::IDENTITY);
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

    fn cleanup_missing(&mut self, seen: &HashSet<usize>) {
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
        let weights = v.weights;
        let joints = v.joints;

        let mut skinned_pos = Vec3::ZERO;
        let mut skinned_norm = Vec3::ZERO;
        let mut skinned_tan = Vec3::ZERO;

        for i in 0..4 {
            let weight = weights[i];
            if weight <= 0.0 {
                continue;
            }
            let mut joint = joints[i] as usize;
            if joint >= palette.len() {
                joint = palette.len().saturating_sub(1);
            }
            let m = palette[joint];
            let m3 = Mat3::from_cols(
                m.x_axis.truncate(),
                m.y_axis.truncate(),
                m.z_axis.truncate(),
            );
            let p = (m * Vec4::new(pos.x, pos.y, pos.z, 1.0)).truncate();
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

pub fn skinning_system(
    mut skinning: ResMut<SkinningResource>,
    time: Res<DeltaTime>,
    runtime_config: Res<BevyRuntimeConfig>,
    asset_server: Option<BevyAssetServerParam<'_>>,
    mut query: Query<(
        Entity,
        &BevySkinnedMeshRenderer,
        Option<&mut BevyAnimator>,
        Option<&BevyPoseOverride>,
    )>,
) {
    let render_config = runtime_config.0.render_config;
    let mode = render_config.skinning_mode;
    skinning.begin_frame(mode);

    let mut seen = HashSet::new();
    let mut vertex_budget = render_config.cpu_skinning_vertex_budget as usize;
    let unlimited_budget = vertex_budget == 0;
    let growth = render_config.skin_palette_growth.max(1.0);

    for (entity, skinned, animator, pose_override) in query.iter_mut() {
        let entity_id = entity.to_bits() as usize;
        seen.insert(entity_id);

        let skin = &skinned.0.skin;
        let skeleton = &skin.skeleton;
        let joint_count = skeleton.joint_count();
        if joint_count == 0 {
            continue;
        }

        let base_pose = Pose::from_skeleton(skeleton);
        let (offset, count) = {
            let entry = skinning.ensure_entry(entity_id, joint_count, &base_pose);
            (entry.offset, entry.count)
        };
        skinning.ensure_palette_capacity(offset + count, growth);

        {
            let SkinningResource {
                palette,
                entries,
                cpu_meshes,
                full_sync_requested,
                ..
            } = &mut *skinning;

            {
                let entry = entries
                    .get_mut(&entity_id)
                    .expect("skinning entry must exist");
                if let Some(override_pose) = pose_override.filter(|pose| pose.0.enabled) {
                    let override_locals = &override_pose.0.pose.locals;
                    if entry.pose.locals.len() != override_locals.len() {
                        entry.pose.locals = override_locals.clone();
                    } else {
                        entry.pose.locals.clone_from_slice(override_locals);
                    }
                } else if let Some(mut animator) = animator {
                    animator.0.evaluate(skeleton, time.0, &mut entry.pose);
                } else {
                    entry.pose.reset_to_bind(skeleton);
                }

                let palette_slice = &mut palette[offset..(offset + count)];
                write_skin_palette(
                    skeleton,
                    &entry.pose.locals,
                    &mut entry.globals,
                    palette_slice,
                );
            }

            if matches!(mode, SkinningMode::Cpu) {
                let asset_server = match asset_server.as_ref() {
                    Some(server) => server,
                    None => continue,
                };
                let base_mesh_id = skinned.0.mesh_id;
                let palette_slice = &palette[offset..(offset + count)];
                let cpu_entry = cpu_meshes.entry(entity_id).or_insert_with(|| {
                    *full_sync_requested = true;
                    let mesh = asset_server.0.lock().get_mesh(base_mesh_id);
                    if let Some(mesh) = mesh {
                        if let Some(lod) = mesh.lods.read().first() {
                            let skinned_vertices = skin_vertices_cpu(&lod.vertices, palette_slice);
                            let handle = asset_server
                                .0
                                .lock()
                                .add_mesh(skinned_vertices, lod.indices.to_vec());
                            return CpuSkinnedMesh {
                                mesh_id: handle.id,
                                base_mesh_id,
                            };
                        }
                    }
                    CpuSkinnedMesh {
                        mesh_id: base_mesh_id,
                        base_mesh_id,
                    }
                });

                if cpu_entry.base_mesh_id != base_mesh_id {
                    cpu_entry.base_mesh_id = base_mesh_id;
                    *full_sync_requested = true;
                }

                if cpu_entry.mesh_id == base_mesh_id {
                    continue;
                }

                let mesh = asset_server.0.lock().get_mesh(base_mesh_id);
                let Some(mesh) = mesh else {
                    continue;
                };
                let lods = mesh.lods.read();
                let Some(lod) = lods.first() else {
                    continue;
                };
                let vertex_count = lod.vertices.len();
                if !unlimited_budget {
                    if vertex_budget < vertex_count {
                        continue;
                    }
                    vertex_budget = vertex_budget.saturating_sub(vertex_count);
                }
                let skinned_vertices = skin_vertices_cpu(&lod.vertices, palette_slice);
                asset_server.0.lock().update_mesh(
                    cpu_entry.mesh_id,
                    skinned_vertices,
                    lod.indices.to_vec(),
                );
            }
        }
    }

    skinning.cleanup_missing(&seen);
}
