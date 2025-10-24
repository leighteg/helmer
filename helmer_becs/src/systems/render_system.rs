use bevy_ecs::{
    prelude::{Entity, Query, Res, ResMut, Resource, With},
    system::Local,
};
use glam::{Mat4, Vec4, Vec4Swizzles};
use hashbrown::HashMap;
use helmer::{
    graphics::{
        config::RenderConfig,
        renderer_common::common::{Aabb, RenderData, RenderLight, RenderObject},
    },
    provided::components::{Camera, Light, Transform},
};
use std::time::Instant;
use tracing::warn;

use crate::{
    BevyActiveCamera, BevyAssetServer, BevyCamera, BevyLight, BevyMeshRenderer, BevyRuntimeConfig,
    BevyTransform,
};

//================================================================================
// Bevy Wrapper & Type Aliases
//================================================================================

// Resource Wrappers
#[derive(Resource, Default)]
pub struct RenderPacket(pub Option<RenderData>);

//================================================================================
// Frustum Culling Logic
//================================================================================

/// A geometric frustum defined by 6 planes, used for culling.
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Creates a frustum from a combined view-projection matrix.
    pub fn from_matrix(mat: Mat4) -> Self {
        let mut planes = [Vec4::ZERO; 6];
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

        for plane in &mut planes {
            *plane /= plane.xyz().length();
        }
        Self { planes }
    }

    /// Checks if a transformed AABB intersects the frustum.
    pub fn intersects_aabb(&self, aabb: &Aabb, transform: &Transform) -> bool {
        let model_matrix = Mat4::from_scale_rotation_translation(
            transform.scale,
            transform.rotation,
            transform.position,
        );
        let center_world = model_matrix.transform_point3(aabb.center());
        let extents_local = aabb.extents();
        let axis_x = model_matrix.col(0).xyz() * extents_local.x;
        let axis_y = model_matrix.col(1).xyz() * extents_local.y;
        let axis_z = model_matrix.col(2).xyz() * extents_local.z;

        for plane in &self.planes {
            let normal = plane.xyz();
            let r = axis_x.dot(normal).abs() + axis_y.dot(normal).abs() + axis_z.dot(normal).abs();
            let s = plane.dot(center_world.extend(1.0));
            if s < -r {
                return false;
            }
        }
        true
    }
}

//================================================================================
// Helper Structs
//================================================================================

/// An intermediate struct holding one frame's state.
#[derive(Clone, Default)]
pub struct ExtractedState {
    pub objects: HashMap<Entity, (Transform, usize, usize, bool, usize)>, // Transform, mesh_id, material_id, casts_shadow, lod_index
    pub lights: HashMap<Entity, (Transform, Light)>,
    pub camera_transform: Transform,
    pub camera_component: Camera,
    pub render_config: RenderConfig,
}

// Squared distances for LOD transitions.
const LOD_THRESHOLDS: [f32; 3] = [25.0 * 25.0, 60.0 * 60.0, 120.0 * 120.0];

//================================================================================
// The Bevy System
//================================================================================

/// Collects all data required for rendering, performs culling and LOD selection.
#[allow(clippy::too_many_arguments)]
pub fn render_data_system(
    // Local state, persists across system runs
    mut previous_state: Local<Option<ExtractedState>>,

    // Resources
    asset_server_res: Option<Res<BevyAssetServer>>,
    runtime_config_res: Option<Res<BevyRuntimeConfig>>,
    mut render_packet: ResMut<RenderPacket>,

    // Queries
    camera_query: Query<(&BevyCamera, &BevyTransform), With<BevyActiveCamera>>,
    objects_query: Query<(Entity, &BevyTransform, &BevyMeshRenderer)>,
    lights_query: Query<(Entity, &BevyTransform, &BevyLight)>,
) {
    // --- 1. Extract Data and Prepare ---
    let (runtime_config, asset_server) = match (runtime_config_res, asset_server_res) {
        (Some(config), Some(server)) => (config.0, server),
        _ => {
            warn!(
                "Required resources (RuntimeConfig, AssetServer) not found. Skipping render data extraction."
            );
            return;
        }
    };

    let mesh_aabb_map = asset_server.0.lock().mesh_aabb_map.read().0.clone();

    let (camera_component, camera_transform) = match camera_query.single() {
        Ok((cam, trans)) => (cam.0, trans.0),
        Err(_) => {
            warn!("No active camera found in scene for rendering.");
            return;
        }
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

    let mut current_state = ExtractedState {
        camera_component,
        camera_transform,
        render_config: runtime_config.render_config,
        ..Default::default()
    };

    // --- 2. Process Renderable Objects (Culling & LOD) ---
    for (entity, transform, mesh_renderer) in objects_query.iter() {
        let transform = transform.0; // unwrap
        let mesh_renderer = mesh_renderer.0; // unwrap

        if !mesh_renderer.visible {
            continue;
        }

        if let Some(aabb) = mesh_aabb_map.get(&mesh_renderer.mesh_id) {
            // Frustum Culling
            if runtime_config.render_config.frustum_culling
                && !frustum.intersects_aabb(aabb, &transform)
            {
                continue;
            }

            // LOD Selection
            let lod_index = if runtime_config.render_config.lod {
                let model_matrix = Mat4::from_scale_rotation_translation(
                    transform.scale,
                    transform.rotation,
                    transform.position,
                );
                let camera_pos_local = model_matrix
                    .inverse()
                    .transform_point3(camera_transform.position);
                let closest_point_local = camera_pos_local.clamp(aabb.min, aabb.max);
                let distance_sq = camera_pos_local.distance_squared(closest_point_local);
                LOD_THRESHOLDS
                    .iter()
                    .filter(|&&threshold| distance_sq > threshold)
                    .count()
            } else {
                0
            };

            current_state.objects.insert(
                entity,
                (
                    transform,
                    mesh_renderer.mesh_id,
                    mesh_renderer.material_id,
                    mesh_renderer.casts_shadow,
                    lod_index,
                ),
            );
        }
    }

    // --- 3. Process Lights ---
    for (entity, transform, light) in lights_query.iter() {
        current_state.lights.insert(entity, (transform.0, light.0));
    }

    // --- 4. Assemble Final RenderData Packet ---
    let prev_state_for_lerp = previous_state.as_ref().unwrap_or(&current_state);

    let objects = current_state
        .objects
        .iter()
        .map(
            |(&entity, &(current_transform, mesh_id, material_id, casts_shadow, lod_index))| {
                let previous_transform = prev_state_for_lerp
                    .objects
                    .get(&entity)
                    .map_or(current_transform, |(pt, _, _, _, _)| *pt);
                RenderObject {
                    id: entity.index() as usize,
                    mesh_id,
                    material_id,
                    current_transform,
                    previous_transform,
                    casts_shadow,
                    lod_index,
                }
            },
        )
        .collect();

    let lights = current_state
        .lights
        .iter()
        .map(|(&entity, &(current_transform, light_component))| {
            let previous_transform = prev_state_for_lerp
                .lights
                .get(&entity)
                .map_or(current_transform, |(pt, _)| *pt);
            RenderLight {
                color: light_component.color.into(),
                intensity: light_component.intensity,
                current_transform,
                previous_transform,
                light_type: light_component.light_type,
            }
        })
        .collect();

    let render_data = RenderData {
        objects,
        lights,
        camera_component: current_state.camera_component,
        current_camera_transform: current_state.camera_transform,
        previous_camera_transform: prev_state_for_lerp.camera_transform,
        timestamp: Instant::now(),
        render_config: current_state.render_config,
    };

    render_packet.0 = Some(render_data);

    // --- 5. Store Current State for Next Frame ---
    *previous_state = Some(current_state);
}
