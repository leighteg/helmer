use std::{any::TypeId, collections::HashSet, env, f32::consts::FRAC_PI_2};

use glam::{DVec2, Mat4, Quat, Vec3, Vec4, Vec4Swizzles};
use helmer_engine::{
    ecs::{
        ecs_core::{ECSCore, Entity},
        system::System,
    },
    graphics::{
        renderer::renderer::{Aabb, Material},
        renderer_system::{RenderDataSystem, RenderPacket},
    },
    physics::{
        components::{ColliderShape, DynamicRigidBody, FixedCollider, PhysicsHandle},
        physics_resource::PhysicsResource,
        systems::{
            CleanupPhysicsSystem, PhysicsStepSystem, SyncEntitiesToPhysicsSystem,
            SyncPhysicsToEntitiesSystem,
        },
    },
    provided::components::{
        ActiveCamera, Camera, Light, LightType, MeshAsset, MeshRenderer, Transform,
    },
    runtime::{
        config::RuntimeConfig, input_manager::{self, InputManager}, runtime::{RenderMessage, Runtime}, scene_system::SceneRoot
    },
};
use rand::Rng;
use rapier3d::{
    math::Isometry,
    na::Vector3,
    parry::query::ShapeCastOptions,
    prelude::{QueryFilter, RigidBodyType},
};
use winit::{event::MouseButton, keyboard::KeyCode};

fn main() {
    let current_path = env::current_dir().expect("Failed to find executable path");
    if current_path.ends_with("helmer-rs") {
        env::set_current_dir(current_path.join("test_game"))
            .expect("Failed to change working directory");
    }

    let mut runtime = Runtime::new(|app| {
        let mut ecs_guard = app.ecs.write();

        ecs_guard.system_scheduler.register_system(
            ConfigToggleSystem {},
            30,
            vec![],
            HashSet::from([]),
            HashSet::from([]),
        );

        // Priority 30: Input and high-level camera control. Runs first.
        ecs_guard.system_scheduler.register_system(
            FreecamSystem::new(1.0, 0.5),
            30,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );

        // Priority 30: Input-based object interaction. Can run in parallel with Freecam.
        ecs_guard.system_scheduler.register_system(
            DragSystem::new(),
            30,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );

        /*// Priority 25: General game logic that modifies transforms.
        ecs_guard.system_scheduler.register_system(
            SpinnerSystem { time_elapsed: 0.0 },
            25,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );*/

        /*ecs_guard.system_scheduler.register_system(
            SpawnSystem::new(),
            25,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );*/

        let asset_server = app.asset_server.as_ref().unwrap().lock();

        // Load meshes from .glb files
        let box_handle = asset_server.load_mesh("assets/models/box.glb");

        let city_handle = asset_server.load_scene("assets/models/city.glb");

        let sponza_handle = asset_server.load_scene("assets/models/sponza.glb");

        let raptor_handle = asset_server.load_scene("assets/models/ford_raptor.glb");

        //let duck_handle = asset_server.load_mesh("assets/models/duck.glb");

        // Load materials from .ron files
        let basic_material_handle = asset_server.load_material("assets/materials/basic.ron");
        let metal_material_handle = asset_server.load_material("assets/materials/shiny_metal.ron");
        let red_light_material_handle =
            asset_server.load_material("assets/materials/red_light.ron");
        let blue_light_material_handle =
            asset_server.load_material("assets/materials/blue_light.ron");
        //let duck_material_handle = asset_server.load_material("assets/materials/duck.ron");

        // Note: You don't need to explicitly load "assets/pattern.ktx2".
        // The AssetServer will automatically find that path inside `blue_light_material.ron`
        // and load it as a dependency.

        // --- 2. Handle Procedurally Generated Meshes ---
        // These are not loaded from files, so we create them and send them
        // directly to the renderer via a RenderMessage.

        let uv_sphere_mesh = MeshAsset::uv_sphere("uv sphere".into(), 32, 32);
        let uv_sphere_mesh_handle =
            asset_server.add_mesh(uv_sphere_mesh.vertices.unwrap(), uv_sphere_mesh.indices);

        let plane_mesh = MeshAsset::plane("plane".into());
        let plane_mesh_handle =
            asset_server.add_mesh(plane_mesh.vertices.unwrap(), plane_mesh.indices);

        let camera_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            camera_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, -3.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(
            camera_entity,
            Camera {
                far_plane: 300.0,
                ..Default::default()
            },
        );
        ecs_guard.add_component(camera_entity, ActiveCamera {});

        let yaw = 30.0f32.to_radians();
        let pitch = 30.0f32.to_radians();
        let roll = 0.0f32.to_radians();

        let rotation = glam::Quat::from_axis_angle(glam::Vec3::Y, yaw) * // Yaw
               glam::Quat::from_axis_angle(glam::Vec3::X, pitch) * // Pitch
               glam::Quat::from_axis_angle(glam::Vec3::Z, roll); // Roll

        // Create some demo entities
        let ground_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            ground_entity,
            Transform {
                position: glam::Vec3::new(0.0, -5.0, 0.0),
                rotation: glam::Quat::default(),
                scale: glam::Vec3::from([500.0, 0.001, 500.0]),
            },
        );
        ecs_guard.add_component(
            ground_entity,
            MeshRenderer::new(plane_mesh_handle.id, metal_material_handle.id, false, false),
        );
        ecs_guard.add_component(ground_entity, ColliderShape::Cuboid);
        ecs_guard.add_component(ground_entity, FixedCollider {});

        let city_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            city_entity,
            Transform {
                position: glam::Vec3::new(0.0, -5.0, 0.0),
                rotation: glam::Quat::default(),
                scale: glam::Vec3::from_array([3.0; 3]),
            },
        );
        ecs_guard.add_component(city_entity, SceneRoot(city_handle));

        let sponza_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            sponza_entity,
            Transform {
                position: glam::Vec3::new(25.0, -4.0, 0.0),
                rotation: glam::Quat::default(),
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(sponza_entity, SceneRoot(sponza_handle));

        let raptor_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            raptor_entity,
            Transform {
                position: glam::Vec3::new(0.0, -5.0, 0.0),
                rotation: glam::Quat::from_rotation_y(90.0),
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(raptor_entity, SceneRoot(raptor_handle));

        let cube_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            cube_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, 0.0),
                rotation,
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(
            cube_entity,
            MeshRenderer::new(box_handle.id, metal_material_handle.id, true, true),
        );
        ecs_guard.add_component(cube_entity, ColliderShape::Cuboid);
        ecs_guard.add_component(cube_entity, DynamicRigidBody { mass: 1.0 });

        let sphere_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            sphere_entity,
            Transform {
                position: glam::Vec3::new(1.0, 0.0, 0.0),
                rotation,
                scale: glam::Vec3::from_array([0.5; 3]),
            },
        );
        ecs_guard.add_component(
            sphere_entity,
            MeshRenderer::new(
                uv_sphere_mesh_handle.id,
                metal_material_handle.id,
                true,
                true,
            ),
        );
        ecs_guard.add_component(sphere_entity, ColliderShape::Sphere);
        ecs_guard.add_component(sphere_entity, DynamicRigidBody { mass: 0.5 });

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
        ecs_guard.add_component(light_entity, ColliderShape::Cuboid);
        ecs_guard.add_component(
            light_entity,
            MeshRenderer::new(box_handle.id, blue_light_material_handle.id, true, true),
        );
        ecs_guard.add_component(light_entity, DynamicRigidBody { mass: 5.0 });

        let light_entity_2 = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity_2,
            Transform {
                position: glam::Vec3::new(-1.5, 0.0, -4.0),
                rotation: glam::Quat::from_array([0.0, 0.0, 0.0, 1.0]),
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(
            light_entity_2,
            Light::point(glam::vec3(1.0, 0.0, 0.0), 10.0),
        );
        ecs_guard.add_component(
            light_entity_2,
            MeshRenderer::new(box_handle.id, red_light_material_handle.id, true, true),
        );
        ecs_guard.add_component(light_entity_2, ColliderShape::Cuboid);
        ecs_guard.add_component(light_entity_2, DynamicRigidBody { mass: 10.0 });

        let sun_rotation = Quat::from_euler(
            glam::EulerRot::YXZ,
            20.0f32.to_radians(),  // Y rotation - very slight side angle
            -50.0f32.to_radians(), // X rotation - steeper downward angle
            20.0f32.to_radians(),  // Z rotation - no roll
        );

        let sun_entity: usize = ecs_guard.create_entity();
        ecs_guard.add_component(
            sun_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, 0.0),
                rotation: sun_rotation,
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(
            sun_entity,
            Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0),
        );
    });
    runtime.init();
}

struct ConfigToggleSystem {}

impl System for ConfigToggleSystem {
    fn name(&self) -> &str {
        "ConfigToggleSystem"
    }

    fn run(
        &mut self,
        dt: f32,
        ecs: &mut helmer_engine::ecs::ecs_core::ECSCore,
        input_manager: &InputManager,
    ) {
        let runtime_config = ecs.get_resource_mut::<RuntimeConfig>().unwrap();

        for key in input_manager.just_pressed.iter() {
            match key {
                KeyCode::KeyZ => runtime_config.render_config.shadow_pass = !runtime_config.render_config.shadow_pass,
                KeyCode::KeyG => runtime_config.render_config.ssgi_pass = !runtime_config.render_config.ssgi_pass,
                KeyCode::Digit0 => runtime_config.render_config.direct_lighting_pass = !runtime_config.render_config.direct_lighting_pass,
                KeyCode::KeyH => runtime_config.render_config.sky_pass = !runtime_config.render_config.sky_pass,
                KeyCode::KeyR => runtime_config.render_config.ssr_pass = !runtime_config.render_config.ssr_pass,
                KeyCode::KeyF => runtime_config.render_config.frustum_culling = !runtime_config.render_config.frustum_culling,
                KeyCode::KeyL => runtime_config.render_config.lod = !runtime_config.render_config.lod,
                KeyCode::KeyU => runtime_config.render_config.egui_pass = !runtime_config.render_config.egui_pass,
                _ => {}
            }
        }
    }
}

struct SpinnerSystem {
    time_elapsed: f32,
}
impl System for SpinnerSystem {
    fn name(&self) -> &str {
        "SpinnerSystem"
    }

    fn run(
        &mut self,
        dt: f32,
        ecs: &mut helmer_engine::ecs::ecs_core::ECSCore,
        input_manager: &InputManager,
    ) {
        self.time_elapsed += dt;

        // Rotation speeds
        let y_rotation_speed = 0.05 * dt; // around world
        let z_rotation_speed = 0.1 * dt; // optional twist
        let oscillation_speed = 0.1; // how fast it bobs up/down

        // Oscillating X (up/down)
        let base_angle = -20.0f32.to_radians(); // horizon
        let deviation = 28.0f32.to_radians();
        let x_angle = base_angle + deviation * (self.time_elapsed * oscillation_speed).sin();
        let rotation_x = Quat::from_axis_angle(glam::Vec3::X, x_angle);

        // Incremental Y/Z rotation (around the world)
        let delta_y = Quat::from_axis_angle(glam::Vec3::Y, y_rotation_speed);
        let delta_z = Quat::from_axis_angle(glam::Vec3::Z, z_rotation_speed);

        ecs.component_pool
            .query_exact_mut_for_each::<(Transform, Light), _>(|(transform, _)| {
                // Apply Y/Z world rotation first
                transform.rotation = delta_y * delta_z * transform.rotation;

                // Extract current Y/Z angles to preserve world spin
                let (y, _, z) = transform.rotation.to_euler(glam::EulerRot::YXZ);

                // Compose final rotation: oscillating X + world Y/Z
                transform.rotation = Quat::from_euler(glam::EulerRot::YXZ, y, x_angle, z);

                // Normalize to avoid drift
                transform.rotation = transform.rotation.normalize();
            });
    }
}

struct FreecamSystem {
    speed: f32,
    sensitivity: f32,

    yaw: f32,
    pitch: f32,

    last_cursor_position: DVec2,
    is_looking: bool,

    base_fov: Option<f32>,
    target_fov: f32,
    fov_lerp_speed: f32,
}

impl FreecamSystem {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            yaw: 0.0,
            pitch: 0.0,
            last_cursor_position: DVec2::ZERO,
            is_looking: false,

            base_fov: None,
            target_fov: 0.0,
            fov_lerp_speed: 8.0,
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
        ecs: &mut helmer_engine::ecs::ecs_core::ECSCore,
        input_manager: &InputManager,
    ) {
        // --- 1. Handle Rotation (Mouse and Controller Right Stick) ---
        if input_manager.is_mouse_button_active(&MouseButton::Right) {
            if !self.is_looking {
                self.last_cursor_position = input_manager.cursor_position;
                self.is_looking = true;
            } else {
                let cursor_delta = input_manager.cursor_position - self.last_cursor_position;
                self.last_cursor_position = input_manager.cursor_position;
                if cursor_delta.length_squared() > 0.0 {
                    self.yaw -= cursor_delta.x as f32 * self.sensitivity / 100.0;
                    self.pitch += cursor_delta.y as f32 * self.sensitivity / 100.0;
                }
            }
        } else {
            self.is_looking = false;
        }

        let maybe_gamepad_id = input_manager.first_gamepad_id();
        if let Some(gamepad_id) = maybe_gamepad_id {
            const CONTROLLER_SENSITIVITY: f32 = 2.0;
            let right_stick_x =
                input_manager.get_controller_axis(gamepad_id, gilrs::Axis::RightStickX);
            let right_stick_y =
                input_manager.get_controller_axis(gamepad_id, gilrs::Axis::RightStickY);
            self.yaw -= right_stick_x * CONTROLLER_SENSITIVITY * dt;
            self.pitch -= right_stick_y * CONTROLLER_SENSITIVITY * dt;
        }

        let pitch_limit = FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-pitch_limit, pitch_limit);

        // --- 2. Handle Speed Adjustment (Mouse Wheel and D-Pad) ---
        self.speed += input_manager.mouse_wheel.y * 2.0;
        if let Some(gamepad_id) = maybe_gamepad_id {
            if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::RightTrigger) {
                self.speed += 10.0 * dt;
            }
            if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::LeftTrigger) {
                self.speed -= 10.0 * dt;
            }
        }
        self.speed = self.speed.max(0.5);
        let mut speed = self.speed;

        // --- 3. Calculate Final Orientation & Direction Vectors ---
        let yaw_rotation = Quat::from_axis_angle(Vec3::Y, self.yaw);
        let pitch_rotation = Quat::from_axis_angle(Vec3::X, self.pitch);
        let orientation = yaw_rotation * pitch_rotation;
        let forward = orientation * Vec3::Z;
        let right = orientation * -Vec3::X;

        // --- 4. Handle Movement Input ---
        let mut velocity = Vec3::ZERO;

        // A. From Keyboard
        for key in input_manager.active_keys.iter() {
            match key {
                KeyCode::KeyW => velocity += forward,
                KeyCode::KeyS => velocity -= forward,
                KeyCode::KeyA => velocity -= right,
                KeyCode::KeyD => velocity += right,
                KeyCode::Space => velocity += Vec3::Y,
                KeyCode::KeyC => velocity -= Vec3::Y,
                _ => {}
            }
        }

        // B. From Controller
        if let Some(gamepad_id) = maybe_gamepad_id {
            let left_stick_y =
                input_manager.get_controller_axis(gamepad_id, gilrs::Axis::LeftStickY);
            let left_stick_x =
                input_manager.get_controller_axis(gamepad_id, gilrs::Axis::LeftStickX);
            velocity += forward * left_stick_y;
            velocity += right * left_stick_x;

            let move_up_amount = input_manager.get_right_trigger_value(gamepad_id);
            let move_down_amount = input_manager.get_left_trigger_value(gamepad_id);
            velocity += Vec3::Y * move_up_amount;
            velocity -= Vec3::Y * move_down_amount;

            if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::LeftThumb) {
                speed *= 2.5;
            }
        }

        if input_manager.is_key_active(&KeyCode::ShiftLeft) {
            speed *= 2.5;
        }

        // --- 5. Apply Updates to ECS Components ---
        ecs.component_pool
            .query_exact_mut_for_each::<(Transform, Camera, ActiveCamera), _>(
                |(transform, camera, _)| {
                    transform.rotation = orientation;
                    if velocity.length_squared() > 0.0 {
                        transform.position += velocity.normalize() * speed * dt;
                    }

                    // --- FOV handling ---
                    if self.base_fov.is_none() {
                        self.base_fov = Some(camera.fov_y_rad);
                        self.target_fov = camera.fov_y_rad;
                    }

                    let base_fov = self.base_fov.unwrap();
                    let boost_active = input_manager.is_key_active(&KeyCode::ShiftLeft)
                        || maybe_gamepad_id.map_or(false, |id| {
                            input_manager.is_controller_button_active(id, gilrs::Button::LeftThumb)
                        });

                    // Pick target fov
                    self.target_fov = if boost_active {
                        base_fov * 1.15
                    } else {
                        base_fov
                    };

                    // Smoothly interpolate camera.fov_y_rad -> target_fov
                    let t = 1.0 - (-self.fov_lerp_speed * dt).exp();
                    camera.fov_y_rad = camera.fov_y_rad + (self.target_fov - camera.fov_y_rad) * t;
                },
            );
    }
}

// --- Ray Struct ---
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// The DragSystem is responsible for handling the logic of clicking and dragging entities.
/// It now interacts directly with the physics engine to move dynamic bodies.
pub struct DragSystem {
    dragged_entity: Option<Entity>,
    drag_distance: f32,
    was_mouse_button_active_last_frame: bool,
}

impl DragSystem {
    pub fn new() -> Self {
        Self {
            dragged_entity: None,
            drag_distance: 0.0,
            was_mouse_button_active_last_frame: false,
        }
    }

    /// Creates a ray from the camera through the cursor's position on the screen.
    fn screen_point_to_ray(&self, ecs: &ECSCore, input_manager: &InputManager) -> Option<Ray> {
        let (camera_transform, camera, _) = ecs
            .component_pool
            .query::<(Transform, Camera, ActiveCamera)>()
            .next()
            .expect("DragSystem requires one entity with Transform, Camera, and ActiveCamera components.");

        let x = (2.0 * input_manager.cursor_position.x as f32) / input_manager.window_size.x as f32
            - 1.0;
        let y = 1.0
            - (2.0 * input_manager.cursor_position.y as f32) / input_manager.window_size.y as f32;

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

        let view_proj_inv = (projection_matrix * view_matrix).inverse();
        let near_point = view_proj_inv * Vec4::new(x, y, -1.0, 1.0);
        let far_point = view_proj_inv * Vec4::new(x, y, 1.0, 1.0);

        let origin = camera_transform.position;
        let direction = (far_point / far_point.w - near_point / near_point.w)
            .normalize()
            .truncate();

        Some(Ray { origin, direction })
    }

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
        let is_active = input_manager.is_mouse_button_active(&MouseButton::Left);
        let is_pressed = is_active && !self.was_mouse_button_active_last_frame;
        let is_released = !is_active && self.was_mouse_button_active_last_frame;

        // --- DRAG START ---
        if is_pressed {
            if let Some(ray) = self.screen_point_to_ray(ecs, input_manager) {
                let mut closest_hit: Option<(Entity, f32)> = None;
                let draggable_entities = ecs
                    .component_pool
                    .get_entities_with_all(&[TypeId::of::<Transform>()]);

                for &id in &draggable_entities {
                    if ecs.component_pool.get::<FixedCollider>(id).is_some() {
                        continue;
                    }
                    if let Some(transform) = ecs.component_pool.get::<Transform>(id) {
                        let radius = 0.5 * transform.scale.max_element();
                        if let Some(distance) =
                            self.ray_sphere_intersection(&ray, transform.position, radius)
                        {
                            if closest_hit.is_none() || distance < closest_hit.unwrap().1 {
                                closest_hit = Some((id, distance));
                            }
                        }
                    }
                }

                if let Some((hit_id, distance)) = closest_hit {
                    self.dragged_entity = Some(hit_id);
                    self.drag_distance = distance;

                    if let Some(handle) = ecs.component_pool.get::<PhysicsHandle>(hit_id).copied() {
                        if let Some(physics) = ecs.get_resource_mut::<PhysicsResource>() {
                            if let Some(rb) = physics.rigid_body_set.get_mut(handle.rigid_body) {
                                rb.set_body_type(RigidBodyType::KinematicPositionBased, true);
                            }
                        }
                    }
                }
            }
        }

        // --- DRAGGING ---
        if is_active {
            if let Some(dragged_id) = self.dragged_entity {
                if let Some(ray) = self.screen_point_to_ray(ecs, input_manager) {
                    let new_pos = ray.origin + ray.direction * self.drag_distance;

                    if let Some(transform) = ecs.component_pool.get_mut::<Transform>(dragged_id) {
                        transform.position = new_pos;
                    }

                    if let Some(handle) =
                        ecs.component_pool.get::<PhysicsHandle>(dragged_id).copied()
                    {
                        if let Some(physics) = ecs.get_resource_mut::<PhysicsResource>() {
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
        }

        // --- DRAG END ---
        if is_released {
            if let Some(dragged_id) = self.dragged_entity {
                let mut final_pos: Option<Vec3> = None;

                let transform_option = ecs.component_pool.get::<Transform>(dragged_id).copied();
                let physics_handle_option =
                    ecs.component_pool.get::<PhysicsHandle>(dragged_id).copied();

                if let Some(transform) = transform_option {
                    final_pos = Some(transform.position);

                    if let (Some(physics), Some(handle)) =
                        (ecs.get_resource::<PhysicsResource>(), physics_handle_option)
                    {
                        if let Some(collider) = physics.collider_set.get(handle.collider) {
                            let shape = collider.shape();
                            let shape_isometry = Isometry::new(
                                transform.position.to_array().into(),
                                Vector3::from_vec(transform.rotation.to_array().to_vec()),
                            );
                            let shape_vel = rapier3d::na::Vector3::new(0.0, -1.0, 0.0);
                            let filter = QueryFilter::new().exclude_collider(handle.collider);
                            let max_time_of_impact = 100.0;

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
                                let snapped_y = transform.position.y - hit.time_of_impact;
                                final_pos = Some(Vec3::new(
                                    transform.position.x,
                                    snapped_y,
                                    transform.position.z,
                                ));
                            }
                        }
                    }
                }

                if let Some(pos) = final_pos {
                    if let Some(transform) = ecs.component_pool.get_mut::<Transform>(dragged_id) {
                        transform.position = pos;
                    }
                }

                if let Some(handle) = physics_handle_option {
                    if let Some(physics) = ecs.get_resource_mut::<PhysicsResource>() {
                        if let Some(rb) = physics.rigid_body_set.get_mut(handle.rigid_body) {
                            if let Some(pos) = final_pos {
                                let mut new_isometry = *rb.position();
                                new_isometry.translation.vector = pos.to_array().into();
                                rb.set_position(new_isometry, true);
                            }
                            rb.set_body_type(RigidBodyType::Dynamic, true);
                            rb.wake_up(true);
                        }
                    }
                }
            }
            self.dragged_entity = None;
        }

        self.was_mouse_button_active_last_frame = is_active;
    }
}

struct SpawnSystem {
    spawned_entities: Vec<Entity>,
}

impl SpawnSystem {
    pub fn new() -> Self {
        Self {
            spawned_entities: Vec::new(),
        }
    }
}

impl System for SpawnSystem {
    fn name(&self) -> &str {
        "SpawnSystem"
    }

    fn run(
        &mut self,
        dt: f32,
        ecs: &mut helmer_engine::ecs::ecs_core::ECSCore,
        input_manager: &InputManager,
    ) {
        // CLEANUP
        const SPAWN_ITERATIONS: usize = 4;
        const MAX_CONCURRENT_ENTITIES: usize = 3000;

        if self.spawned_entities.len() >= MAX_CONCURRENT_ENTITIES {
            ecs.destroy_entity(self.spawned_entities.remove(0));
        }

        let mut dead_entities: Vec<Entity> = Vec::new();

        for entity in self.spawned_entities.iter() {
            match ecs.get_component::<Transform>(*entity) {
                Some(transform) => {
                    if transform.position.y < -6.0 {
                        ecs.destroy_entity(*entity);

                        dead_entities.insert(dead_entities.len(), *entity);
                    }
                }
                _ => {}
            }
        }

        for entity in dead_entities.drain(0..dead_entities.len()) {
            match self.spawned_entities.iter().position(|&r| r == entity) {
                Some(index) => {
                    self.spawned_entities.remove(index);
                }
                _ => {}
            }
        }

        // -----

        let mut rng = rand::rng();

        for _ in 0..SPAWN_ITERATIONS {
            let new_entity = ecs.create_entity();

            self.spawned_entities
                .insert(self.spawned_entities.len(), new_entity);

            let mut random_x: f32 = rng.random_range(-13.0..13.0);
            let mut random_y: f32 = rng.random_range(0.0..50.0);
            let mut random_z: f32 = rng.random_range(-13.0..13.0);

            let position = Vec3::new(random_x, random_y, random_z);

            random_x = rng.random_range(-10.0..10.0);
            random_y = rng.random_range(-10.0..10.0);
            random_z = rng.random_range(-10.0..10.0);

            let rotation = Quat::from_xyzw(random_x, random_y, random_z, 1.0);

            let scale: f32 = rng.random_range(0.1..5.0);

            ecs.add_component(
                new_entity,
                Transform {
                    position,
                    rotation,
                    scale: Vec3::from_array([scale; 3]),
                },
            );

            let mesh_id: usize = rng.random_range(0..1);
            let material_id: usize = rng.random_range(3..6);

            match material_id {
                4 => {
                    ecs.add_component(
                        new_entity,
                        Light::point(glam::vec3(1.0, 0.0, 0.0), 10.0 * scale),
                    );
                }
                5 => {
                    ecs.add_component(
                        new_entity,
                        Light::point(glam::vec3(0.0, 0.0, 1.0), 10.0 * scale),
                    );
                }
                _ => {}
            }

            ecs.add_component(
                new_entity,
                MeshRenderer::new(mesh_id, material_id, true, true),
            );

            match mesh_id {
                0 => {
                    ecs.add_component(new_entity, ColliderShape::Cuboid);
                }
                1 => {
                    ecs.add_component(new_entity, ColliderShape::Sphere);
                }
                _ => {}
            }

            ecs.add_component(new_entity, DynamicRigidBody { mass: scale });

            let _ = rng.reseed();
        }
    }
}
