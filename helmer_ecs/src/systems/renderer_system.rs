use crate::ecs::{
    ecs_core::{ECSCore, Entity},
    system::System,
};
use glam::{Mat4, Vec4, Vec4Swizzles};
use hashbrown::HashMap;
use helmer::{
    graphics::{
        config::RenderConfig,
        render_graphs::default_graph_spec,
        renderer_common::common::{
            Aabb, AssetStreamKind, AssetStreamingRequest, RenderCameraDelta, RenderDelta,
            RenderLightDelta, RenderObjectDelta,
        },
        renderer_common::graph::RenderGraphSpec,
    },
    provided::components::{ActiveCamera, Camera, Light, MeshRenderer, Transform},
    runtime::{asset_server::AssetServer, config::RuntimeConfig, input_manager::InputManager},
};
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
pub struct RenderGraphResource(pub RenderGraphSpec);

impl Default for RenderGraphResource {
    fn default() -> Self {
        Self(default_graph_spec())
    }
}

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

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let mut streaming_hints: Vec<AssetStreamingRequest> = Vec::new();
        let runtime_config = match ecs.get_resource::<RuntimeConfig>() {
            Some(config) => config.clone(),
            None => {
                warn!("RuntimeConfig resource not found");
                return;
            }
        };
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
                objects_upsert.push(RenderObjectDelta {
                    id: entity,
                    transform,
                    mesh_id,
                    material_id,
                    casts_shadow,
                    lod_index,
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
                    objects_upsert.push(RenderObjectDelta {
                        id: entity,
                        transform,
                        mesh_id,
                        material_id,
                        casts_shadow,
                        lod_index,
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
            camera: None,
            render_config: None,
            render_graph: None,
            streaming_requests: None,
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
