use bevy_ecs::{
    prelude::{Entity, Query, Res, ResMut, Resource, With, Without},
    system::Local,
};
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use helmer_becs::{
    BevyActiveCamera, BevyCamera, BevyInputManager, BevyTransform,
    physics::{
        components::{FixedCollider, PhysicsHandle},
        physics_resource::PhysicsResource,
    },
};
use rapier3d::{
    na::Vector3,
    parry::query::ShapeCastOptions,
    prelude::{Isometry, QueryFilter, RigidBodyType, Vector},
};
use std::f32::consts::PI;
use winit::event::MouseButton;

// --- Ray Struct ---
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

// --- System Local State ---

/// Internal state for the DragSystem.
#[derive(Default)]
pub struct DragState {
    dragged_entity: Option<(Entity, RigidBodyType)>,
    drag_distance: f32,
    was_mouse_button_active_last_frame: bool,
}

// --- Helper Functions ---

/// Creates a ray from the camera through the cursor's position on the screen.
fn screen_point_to_ray(
    camera_query: &Query<(&BevyCamera, &BevyTransform), With<BevyActiveCamera>>,
    input: &BevyInputManager,
) -> Option<Ray> {
    let input = input.0.read();

    // Get the active camera's components
    let (camera, transform) = camera_query.single().ok()?;
    let camera = camera.0; // unwrap
    let transform = transform.0; // unwrap

    // Convert cursor position to NDC
    let x = (2.0 * input.cursor_position.x as f32) / input.window_size.x as f32 - 1.0;
    let y = 1.0 - (2.0 * input.cursor_position.y as f32) / input.window_size.y as f32;

    // Use the same projection matrix as the original system
    // NOTE: The original system used perspective_rh, not infinite_reverse_rh
    let projection_matrix = Mat4::perspective_rh(
        camera.fov_y_rad,
        camera.aspect_ratio,
        camera.near_plane,
        camera.far_plane,
    );

    let view_matrix = Mat4::look_at_rh(
        transform.position,
        transform.position + transform.forward(),
        transform.up(),
    );

    // Calculate inverse view-projection
    let view_proj_inv = (projection_matrix * view_matrix).inverse();
    let near_point = view_proj_inv * Vec4::new(x, y, -1.0, 1.0); // Near plane in NDC
    let far_point = view_proj_inv * Vec4::new(x, y, 1.0, 1.0); // Far plane in NDC

    // De-homogenize
    let near_world = near_point.xyz() / near_point.w;
    let far_world = far_point.xyz() / far_point.w;

    // Create the ray
    // The origin is the camera's position (or near_world, both work for a ray)
    let origin = near_world;
    let direction = (far_world - near_world).normalize();

    Some(Ray { origin, direction })
}

/// Checks for the intersection of a ray with a sphere.
fn ray_sphere_intersection(ray: &Ray, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ray.origin - center;
    let a = ray.direction.length_squared();
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.length_squared() - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        None
    } else {
        let t = (-b - discriminant.sqrt()) / (2.0 * a);
        if t > 0.0 { Some(t) } else { None }
    }
}

//================================================================================
// The System
//================================================================================

/// Handles clicking and dragging physics-enabled entities with the mouse.
pub fn drag_system(
    // Local state, persists across system runs
    mut state: Local<DragState>,

    // Resources
    input_res: Option<Res<BevyInputManager>>,
    physics_res: Option<ResMut<PhysicsResource>>,

    // Queries
    camera_query: Query<(&BevyCamera, &BevyTransform), With<BevyActiveCamera>>,
    // Query for all entities that can be dragged:
    // - Must have a Transform (to move)
    // - Must have a PhysicsHandle (to interact with physics)
    // - Must NOT have a FixedCollider (so we don't drag the ground)
    mut draggable_query: Query<
        (Entity, &mut BevyTransform, &PhysicsHandle),
        Without<BevyActiveCamera>,
    >,
) {
    // --- 1. Get Resources and Input State ---
    let (Some(input), Some(mut physics)) = (input_res, physics_res) else {
        // If resources aren't available, do nothing.
        return;
    };

    let is_active = input.0.read().is_mouse_button_active(MouseButton::Left);
    let is_pressed = is_active && !state.was_mouse_button_active_last_frame;
    let is_released = !is_active && state.was_mouse_button_active_last_frame;

    // --- 2. DRAG START ---
    if is_pressed {
        if let Some(ray) = screen_point_to_ray(&camera_query, &input) {
            let mut closest_hit: Option<(Entity, f32)> = None;

            // Iterate over all entities that are draggable
            for (entity, transform, _handle) in draggable_query.iter() {
                // Use a simple sphere intersection for picking
                let radius = 0.5 * transform.0.scale.max_element();
                if let Some(distance) = ray_sphere_intersection(&ray, transform.0.position, radius)
                {
                    if closest_hit.is_none() || distance < closest_hit.unwrap().1 {
                        closest_hit = Some((entity, distance));
                    }
                }
            }

            // If we hit an entity, start dragging it
            if let Some((hit_id, distance)) = closest_hit {
                // Set the physics body to Kinematic
                // We can safely query here because draggable_query ensures the handle exists.
                if let Ok((_, _, handle)) = draggable_query.get(hit_id) {
                    if let Some(rb) = physics.rigid_body_set.get_mut(handle.rigid_body) {
                        state.dragged_entity = Some((hit_id, rb.body_type()));
                        state.drag_distance = distance;
                        rb.set_body_type(RigidBodyType::KinematicPositionBased, true);
                    }
                }
            }
        }
    }

    // --- 3. DRAGGING ---
    if is_active {
        if let Some((dragged_id, _)) = state.dragged_entity {
            if let Some(ray) = screen_point_to_ray(&camera_query, &input) {
                let new_pos = ray.origin + ray.direction * state.drag_distance;

                // Get the mutable components for the dragged entity
                if let Ok((_, mut transform, handle)) = draggable_query.get_mut(dragged_id) {
                    // Update engine transform
                    transform.0.position = new_pos;

                    // Update physics rigid body
                    if let Some(rb) = physics.rigid_body_set.get_mut(handle.rigid_body) {
                        if rb.is_kinematic() {
                            let mut new_isometry = *rb.position();
                            new_isometry.translation.vector = new_pos.to_array().into();
                            rb.set_next_kinematic_position(new_isometry);
                        }
                    }
                }
            }
        }
    }

    // --- 4. DRAG END ---
    if is_released {
        if let Some((dragged_id, dragged_body_type)) = state.dragged_entity {
            let mut final_pos: Option<Vec3> = None;

            // We use get_mut because we will write to transform at the end.
            if let Ok((_, mut transform, physics_handle)) = draggable_query.get_mut(dragged_id) {
                final_pos = Some(transform.0.position); // Default final pos is current pos

                // --- Optional: Shape Cast Logic to snap to ground ---
                // (This logic is complex and depends heavily on the collider setup)
                if let Some(collider) = physics.collider_set.get(physics_handle.collider) {
                    let shape = collider.shape();
                    let shape_isometry = Isometry::new(
                        transform.0.position.to_array().into(),
                        Vector3::from_vec(transform.0.rotation.to_array().to_vec()),
                    );

                    let shape_vel = rapier3d::na::Vector3::new(0.0, -1.0, 0.0); // Cast downwards
                    let filter = QueryFilter::new().exclude_collider(physics_handle.collider);
                    let max_time_of_impact = 100.0; // Max cast distance

                    if let Some((_hit_handle, hit)) = physics.query_pipeline.cast_shape(
                        &physics.rigid_body_set,
                        &physics.collider_set,
                        &shape_isometry,
                        &shape_vel,
                        shape,
                        ShapeCastOptions {
                            max_time_of_impact,
                            stop_at_penetration: true,
                            ..Default::default()
                        },
                        filter,
                    ) {
                        // Snap the position to the hit point
                        let snapped_y = transform.0.position.y - hit.time_of_impact;
                        final_pos = Some(Vec3::new(
                            transform.0.position.x,
                            snapped_y,
                            transform.0.position.z,
                        ));
                    }
                }

                // --- Apply Final Position ---
                if let Some(pos) = final_pos {
                    transform.0.position = pos;
                }

                // --- Update Physics Body ---
                if let Some(rb) = physics.rigid_body_set.get_mut(physics_handle.rigid_body) {
                    if let Some(pos) = final_pos {
                        let mut new_isometry = *rb.position();
                        new_isometry.translation.vector = pos.to_array().into();
                        rb.set_position(new_isometry, true);
                    }
                    rb.set_body_type(dragged_body_type, true);
                    rb.wake_up(true);
                }
            }

            // Clear the state
            state.dragged_entity = None;
        }
    }

    // --- 5. Update State for Next Frame ---
    state.was_mouse_button_active_last_frame = is_active;
}
