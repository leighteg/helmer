use crate::{
    ecs::{
        ecs_core::{ECSCore, Entity},
        system::System,
    },
    graphics::{
        config::RenderConfig,
        renderer::renderer::{Aabb, RenderData, RenderLight, RenderObject},
    },
    provided::components::{ActiveCamera, Camera, Light, MeshRenderer, Transform},
    runtime::{config::RuntimeConfig, input_manager::InputManager},
};
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use hashbrown::HashMap;
use std::time::Instant;
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

/// A resource mapping Mesh Handles to their AABBs, populated by the AssetServer.
#[derive(Default)]
pub struct MeshAabbMap(pub HashMap<usize, Aabb>);

/// An intermediate struct holding one frame's state. Now includes the selected LOD index.
#[derive(Clone, Default)]
pub struct ExtractedState {
    pub objects: HashMap<Entity, (Transform, usize, usize, bool, usize)>, // Transform, mesh_id, material_id, casts_shadow, lod_index
    pub lights: HashMap<Entity, (Transform, Light)>,
    pub camera_transform: Transform,
    pub camera_component: Camera,
    pub render_config: RenderConfig,
}

#[derive(Default)]
pub struct RenderPacket(pub Option<RenderData>);

/// Collects all data required for rendering, performs culling and LOD selection.
pub struct RenderDataSystem {
    previous_state: Option<ExtractedState>,
}

impl RenderDataSystem {
    pub fn new() -> Self {
        Self {
            previous_state: None,
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

impl System for RenderDataSystem {
    fn name(&self) -> &str {
        "RenderDataSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let current_state = {
            let mesh_aabb_map = match ecs.get_resource::<MeshAabbMap>() {
                Some(map) => map.0.clone(),
                None => {
                    warn!("MeshAabbMap resource not found, culling will be skipped.");
                    return;
                }
            };
            let runtime_config = match ecs.get_resource::<RuntimeConfig>() {
                Some(config) => config.clone(),
                None => {
                    warn!("RuntimeConfig resource not found");
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
                warn!("No active camera found in scene for rendering.");
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

            let mut state = ExtractedState::default();
            state.camera_component = camera_component;
            state.camera_transform = camera_transform;
            state.render_config = runtime_config.render_config;

            ecs.component_pool
                .query_for_each::<(Transform, MeshRenderer), _>(
                    |entity, (transform, mesh_renderer)| {
                        if !mesh_renderer.visible {
                            return;
                        }

                        if let Some(aabb) = mesh_aabb_map.get(&mesh_renderer.mesh_id) {
                            // --- Frustum Culling (Correct) ---
                            if runtime_config.render_config.frustum_culling {
                                if !frustum.intersects_aabb(aabb, transform) {
                                    return;
                                }
                            }

                            // --- LOD SELECTION ---
                            // This method calculates the distance to the closest point on the object's bounding box.
                            let lod_index: usize = if runtime_config.render_config.lod {
                                // 1. Get the matrix to transform from world space to the object's local space.
                                let model_matrix = Mat4::from_scale_rotation_translation(
                                    transform.scale,
                                    transform.rotation,
                                    transform.position,
                                );
                                let inverse_model = model_matrix.inverse();

                                // 2. Move the camera's position into the object's local space.
                                let camera_pos_local =
                                    inverse_model.transform_point3(camera_transform.position);

                                // 3. Find the closest point on the axis-aligned bounding box to the camera's local position.
                                //    The `clamp` function is perfect for this. If the camera is inside the box, this will be the camera's own position.
                                let closest_point_local =
                                    camera_pos_local.clamp(aabb.min, aabb.max);

                                // 4. The squared distance is now between the camera (in local space) and the closest point on the box (in local space).
                                //    If the camera is inside the box, this distance is 0, correctly selecting LOD 0.
                                let distance_sq =
                                    camera_pos_local.distance_squared(closest_point_local);

                                LOD_THRESHOLDS
                                    .iter()
                                    .filter(|&&threshold| distance_sq > threshold)
                                    .count()
                            } else {
                                0
                            };

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

            ecs.component_pool.query_for_each::<(Transform, Light), _>(
                |entity, (transform, light)| {
                    state.lights.insert(entity, (*transform, *light));
                },
            );

            state
        };

        let prev_state = self.previous_state.as_ref().unwrap_or(&current_state);

        let objects = current_state
            .objects
            .iter()
            .map(
                |(&entity, &(current_transform, mesh_id, material_id, casts_shadow, lod_index))| {
                    let previous_transform = prev_state
                        .objects
                        .get(&entity)
                        .map_or(current_transform, |(pt, _, _, _, _)| *pt);
                    RenderObject {
                        id: entity,
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
                let previous_transform = prev_state
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
            previous_camera_transform: prev_state.camera_transform,
            timestamp: Instant::now(),
            render_config: current_state.render_config,
        };

        if let Some(render_packet) = ecs.get_resource_mut::<RenderPacket>() {
            render_packet.0 = Some(render_data);
        } else {
            eprintln!(
                "[ERROR] RenderPacket resource not found in ECS. Please register it on startup."
            );
        }

        self.previous_state = Some(current_state);
    }
}
