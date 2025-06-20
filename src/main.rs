use std::{any::TypeId, collections::HashSet, f32::consts::FRAC_PI_2};

use glam::{DVec2, Mat4, Quat, Vec3, Vec4, Vec4Swizzles};
use helmer_rs::{
    ecs::{ecs_core::{ECSCore, Entity}, system::System},
    provided::components::{ActiveCamera, Camera, Light, LightType, MeshAsset, MeshRenderer, Transform},
    runtime::{
        input_manager::{self, InputManager},
        runtime::Runtime,
    },
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use winit::{event::MouseButton, keyboard::KeyCode};

fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .unwrap();

    tracing::info!("2025 leighton");

    let mut runtime = Runtime::new(|app| {
        let mut ecs_guard = app.ecs.write().unwrap();

        ecs_guard.system_scheduler.register_system(
            SpinnerSystem {},
            10,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );

        ecs_guard.system_scheduler.register_system(
            FreecamSystem::new(1.0, 0.5),
            10,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );

        ecs_guard.system_scheduler.register_system(
            DragSystem::new(),
            10,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );

        let camera_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            camera_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, -3.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(camera_entity, Camera::default());
        ecs_guard.add_component(camera_entity, ActiveCamera {});

        let yaw = 30.0f32.to_radians();
        let pitch = 30.0f32.to_radians();
        let roll = 0.0f32.to_radians();

        let rotation = glam::Quat::from_axis_angle(glam::Vec3::Y, yaw) * // Yaw
               glam::Quat::from_axis_angle(glam::Vec3::X, pitch) * // Pitch
               glam::Quat::from_axis_angle(glam::Vec3::Z, roll); // Roll

        // Create some demo entities
        let cube_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            cube_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, 0.0),
                rotation,
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(cube_entity, MeshRenderer::new(0, 0, true));

        let sphere_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            sphere_entity,
            Transform {
                position: glam::Vec3::new(1.0, 0.0, 0.0),
                rotation,
                scale: glam::Vec3::from_array([0.5, 0.5, 0.5]),
            },
        );
        ecs_guard.add_component(sphere_entity, MeshRenderer::new(1, 0, true));

        let light_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity,
            Transform {
                position: glam::Vec3::new(0.0, 1.5, 2.0),
                rotation: glam::Quat::from_array([0.0, 0.0, 0.0, 1.0]),
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(light_entity, Light::point(glam::vec3(0.0, 0.0, 1.0), 10.0));
        ecs_guard.add_component(light_entity, MeshRenderer::new(0, 0, true));

        let light_entity_2 = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity_2,
            Transform {
                position: glam::Vec3::new(-1.5, 0.0, 0.0),
                rotation: glam::Quat::from_array([0.0, 0.0, 0.0, 1.0]),
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(
            light_entity_2,
            Light::point(glam::vec3(1.0, 0.0, 0.0), 10.0),
        );
        ecs_guard.add_component(light_entity_2, MeshRenderer::new(0, 0, true));
    });
    runtime.init();
}

struct SpinnerSystem {}
impl System for SpinnerSystem {
    fn name(&self) -> &str {
        "SpinnerSystem"
    }

    fn run(
        &mut self,
        dt: f32,
        ecs: &mut helmer_rs::ecs::ecs_core::ECSCore,
        input_manager: &InputManager,
    ) {
        let rotation_speed = 0.50 * dt;
        let delta_x_rotation = Quat::from_axis_angle(glam::Vec3::X, rotation_speed);
        let delta_y_rotation = Quat::from_axis_angle(glam::Vec3::Y, rotation_speed);
        let delta_z_rotation = Quat::from_axis_angle(glam::Vec3::Z, rotation_speed * 2.0);

        ecs.component_pool
            .query_exact_mut_for_each::<(Transform, MeshRenderer), _>(
                |(transform, mesh_renderer)| {
                    transform.rotation *= delta_x_rotation * delta_y_rotation * delta_z_rotation;
                    // Re-normalize the quaternion to prevent floating-point drift over time.
                    transform.rotation = transform.rotation.normalize();
                },
            );
    }
}

struct FreecamSystem {
    speed: f32,
    sensitivity: f32,

    yaw: f32,
    pitch: f32,

    // State for handling activation
    last_cursor_position: DVec2,
    is_looking: bool, // Our new state flag
}

impl FreecamSystem {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            yaw: 0.0,
            pitch: 0.0,
            last_cursor_position: DVec2::ZERO,
            is_looking: false, // Start in an inactive state
        }
    }
}

impl System for FreecamSystem {
    fn name(&self) -> &str {
        "FreecamSystem"
    }

    fn run(
        &mut self,
        dt: f32,
        ecs: &mut helmer_rs::ecs::ecs_core::ECSCore,
        input_manager: &InputManager,
    ) {
        if !input_manager.is_mouse_button_active(&MouseButton::Right) {
            self.is_looking = false;
            return;
        }

        if self.speed >= 0.5 || input_manager.mouse_wheel.y.is_sign_positive() {
            self.speed += input_manager.mouse_wheel.y / 2.0;
        }
        let mut speed = self.speed;

        // --- 1. Handle Rotation from Mouse Input ---
        if !self.is_looking {
            // This is the first frame the button is pressed.
            // Reset last_cursor_position to the current position to avoid the "spaz".
            self.last_cursor_position = input_manager.cursor_position;
            self.is_looking = true;
        } else {
            // We are actively looking, so calculate delta and apply rotation.
            let cursor_delta = input_manager.cursor_position - self.last_cursor_position;
            self.last_cursor_position = input_manager.cursor_position;

            if cursor_delta.length_squared() > 0.0 {
                self.yaw -= cursor_delta.x as f32 * self.sensitivity / 100.0;
                self.pitch += cursor_delta.y as f32 * self.sensitivity / 100.0;

                // Clamp pitch
                let pitch_limit = FRAC_PI_2 - 0.01;
                self.pitch = self.pitch.clamp(-pitch_limit, pitch_limit);
            }
        }

        // --- 2. Calculate Final Orientation ---
        let yaw_rotation = Quat::from_axis_angle(Vec3::Y, self.yaw);
        let pitch_rotation = Quat::from_axis_angle(Vec3::X, self.pitch);
        let orientation = yaw_rotation * pitch_rotation;

        // --- 3. Handle Movement from Keyboard Input ---
        let mut velocity = Vec3::ZERO;
        let forward = orientation * Vec3::Z; // Use -Z for forward in a right-handed system
        let right = orientation * Vec3::X;

        for key in input_manager.active_keys.iter() {
            match key {
                KeyCode::KeyW => velocity += forward,
                KeyCode::KeyS => velocity -= forward,
                KeyCode::KeyA => velocity += right,
                KeyCode::KeyD => velocity -= right,
                KeyCode::Space => velocity += Vec3::Y,
                KeyCode::KeyC => velocity -= Vec3::Y,
                KeyCode::ShiftLeft => speed *= 2.5,
                _ => {}
            }
        }

        // --- 4. Apply Updates to ECS Components ---
        ecs.component_pool
            .query_exact_mut_for_each::<(Transform, Camera, ActiveCamera), _>(|(transform, _, _)| {
                transform.rotation = orientation;
                if velocity.length_squared() > 0.0 {
                    transform.position += velocity.normalize() * speed * dt;
                }
            });
    }
}

// --- Debug Flag ---
const ENABLE_DRAG_SYSTEM_LOGGING: bool = false;

// --- Ray Struct ---
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// The DragSystem is responsible for handling the logic of clicking and dragging entities.
pub struct DragSystem {
    dragged_entity: Option<Entity>,
    drag_offset: Vec3,
    drag_plane_y: f32,
    was_mouse_button_active_last_frame: bool,
}

impl DragSystem {
    pub fn new() -> Self {
        Self {
            dragged_entity: None,
            drag_offset: Vec3::ZERO,
            drag_plane_y: 0.0,
            was_mouse_button_active_last_frame: false,
        }
    }

    /// Creates a ray from the camera through the cursor's position on the screen.
    fn screen_point_to_ray(
        &self,
        ecs: &ECSCore,
        input_manager: &InputManager,
    ) -> Option<Ray> {
        let (camera, camera_transform) = ecs
            .component_pool
            .query::<(Camera, Transform)>()
            .next()
            .expect("DragSystem requires one active Camera in the scene.");

        // Convert cursor position to Normalized Device Coordinates (NDC)
        let x = (2.0 * input_manager.cursor_position.x as f32) / input_manager.window_size.x as f32 - 1.0;
        let y = 1.0 - (2.0 * input_manager.cursor_position.y as f32) / input_manager.window_size.y as f32;

        // Define the view and projection matrices
        let view_matrix = Mat4::look_at_rh(
            camera_transform.position,
            camera_transform.position + camera_transform.forward(),
            camera_transform.up(),
        );
        let projection_matrix = Mat4::perspective_rh(
            camera.fov_y_rad,
            camera.aspect_ratio,
            camera.near_plane,
            camera.far_plane,
        );

        // Unproject the screen point back into world space.
        let view_proj_inv = (projection_matrix * view_matrix).inverse();
        let near_point = view_proj_inv * Vec4::new(x, y, -1.0, 1.0);
        let far_point = view_proj_inv * Vec4::new(x, y, 1.0, 1.0);

        // Calculate the ray from these points.
        let origin = camera_transform.position;
        let direction = (far_point / far_point.w - near_point / near_point.w).normalize().truncate();

        if ENABLE_DRAG_SYSTEM_LOGGING {
            println!("[DEBUG] Raycast initiated: origin={:?}, direction={:?}", origin, direction);
        }

        Some(Ray { origin, direction })
    }

    /// Performs a simple intersection test between a ray and a sphere.
    fn ray_sphere_intersection(&self, ray: &Ray, center: Vec3, radius: f32) -> Option<f32> {
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
}

impl Default for DragSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for DragSystem {
    fn name(&self) -> &str {
        "DragSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, input_manager: &InputManager) {
        // --- 1. DETERMINE MOUSE STATE CHANGES ---
        let is_active = input_manager.is_mouse_button_active(&MouseButton::Left);
        let is_pressed = is_active && !self.was_mouse_button_active_last_frame;

        if is_active {
            // --- 2. HANDLE DRAG START (Single-frame event on mouse press) ---
            if is_pressed {
                if ENABLE_DRAG_SYSTEM_LOGGING { println!("[DEBUG] Mouse Press Detected at {:?}", input_manager.cursor_position); }

                if let Some(ray) = self.screen_point_to_ray(ecs, input_manager) {
                    let mut closest_hit: Option<(Entity, f32)> = None;

                    let draggable_entities = ecs.component_pool.get_entities_with_all(&[TypeId::of::<Transform>()]);

                    if ENABLE_DRAG_SYSTEM_LOGGING { println!("[DEBUG] Found {} draggable entities", draggable_entities.len()); }

                    for &id in &draggable_entities {
                        if let Some(transform) = ecs.component_pool.get::<Transform>(id) {
                            let radius = 0.5 * transform.scale.max_element();
                            if ENABLE_DRAG_SYSTEM_LOGGING { println!("[DEBUG] Testing entity {:?}, pos: {:?}, radius: {}", id, transform.position, radius); }
                            if let Some(distance) = self.ray_sphere_intersection(&ray, transform.position, radius) {
                                if ENABLE_DRAG_SYSTEM_LOGGING { println!("[DEBUG] Intersection found! Entity {:?} at distance {}", id, distance); }
                                if closest_hit.is_none() || distance < closest_hit.unwrap().1 {
                                    closest_hit = Some((id, distance));
                                }
                            }
                        }
                    }

                    if let Some((hit_id, distance)) = closest_hit {
                        let hit_point = ray.origin + ray.direction * distance;
                        if let Some(transform) = ecs.component_pool.get_mut::<Transform>(hit_id) {
                            if ENABLE_DRAG_SYSTEM_LOGGING { println!("[SUCCESS] Drag initiated for entity {:?}", hit_id); }
                            self.dragged_entity = Some(hit_id);
                            self.drag_offset = hit_point - transform.position;
                            self.drag_plane_y = transform.position.y;
                        }
                    }
                }
            }
            // --- 3. HANDLE DRAGGING (When a drag is already in progress) ---
            else if let Some(dragged_id) = self.dragged_entity {
                if let Some(ray) = self.screen_point_to_ray(ecs, input_manager) {
                    let plane_normal = Vec3::Y;
                    let plane_origin = Vec3::new(0.0, self.drag_plane_y, 0.0);

                    let denom = ray.direction.dot(plane_normal);
                    if denom.abs() > 1e-6 {
                        let t = (plane_origin - ray.origin).dot(plane_normal) / denom;
                        if t >= 0.0 {
                            let hit_point = ray.origin + ray.direction * t;
                            if let Some(transform) = ecs.component_pool.get_mut::<Transform>(dragged_id) {
                                let new_pos = hit_point - self.drag_offset;
                                if ENABLE_DRAG_SYSTEM_LOGGING { println!("[DEBUG] Dragging entity {:?}, new_pos: {:?}", dragged_id, new_pos); }
                                transform.position = new_pos;
                            }
                        }
                    }
                }
            }
        } else {
            // --- 4. HANDLE DRAG END (When mouse button is released) ---
            if self.dragged_entity.is_some() {
                if ENABLE_DRAG_SYSTEM_LOGGING { println!("[DEBUG] Drag ended."); }
                self.dragged_entity = None;
            }
        }

        // --- 5. UPDATE STATE FOR NEXT FRAME ---
        self.was_mouse_button_active_last_frame = is_active;
    }
}