use std::{any::TypeId, collections::HashSet, f32::consts::FRAC_PI_2};

use glam::{DVec2, Mat4, Quat, Vec3, Vec4Swizzles};
use helmer_rs::{
    ecs::system::System,
    provided::components::{Camera, Light, LightType, MeshAsset, MeshRenderer, Transform},
    runtime::{input_manager::{self, InputManager}, runtime::Runtime}
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
            FreecamSystem::new(1.0, 1.0),
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
        ecs_guard.add_component(camera_entity, Camera::new(0.0, 1.0, 1.0, 1.0));

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
        ecs_guard.add_component(cube_entity, MeshRenderer::new(0, 0));

        let light_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity,
            Transform {
                position: glam::Vec3::new(0.0, 1.5, 5.0),
                rotation: glam::Quat::from_array([0.0, 0.0, 0.0, 1.0]),
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(
            light_entity,
            Light::spot(glam::vec3(0.0, 0.0, 1.0), 10.0, 60.0),
        );
        ecs_guard.add_component(light_entity, MeshRenderer::new(0, 0));

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
        ecs_guard.add_component(light_entity_2, MeshRenderer::new(0, 0));
    });
    runtime.init();
}

struct SpinnerSystem {}
impl System for SpinnerSystem {
    fn name(&self) -> &str {
        "SpinnerSystem"
    }

    fn run(&mut self, dt: f32, ecs: &mut helmer_rs::ecs::ecs_core::ECSCore, input_manager: &InputManager) {
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

    fn run(&mut self, dt: f32, ecs: &mut helmer_rs::ecs::ecs_core::ECSCore, input_manager: &InputManager) {
        if self.speed >= 0.1 || input_manager.mouse_wheel.y.is_sign_positive() {
            self.speed += input_manager.mouse_wheel.y / 12.0;
        }
        let mut speed = self.speed;
        
        // --- 1. Handle Rotation from Mouse Input ---

        // Check if the condition to look is met (e.g., left mouse button is down)
        if input_manager.is_mouse_button_active(&MouseButton::Left) {
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
                    self.yaw   -= cursor_delta.x as f32 * self.sensitivity * dt;
                    self.pitch += cursor_delta.y as f32 * self.sensitivity * dt;

                    // Clamp pitch
                    let pitch_limit = FRAC_PI_2 - 0.01;
                    self.pitch = self.pitch.clamp(-pitch_limit, pitch_limit);
                }
            }
        } else {
            // Mouse button is not pressed, so we are not looking.
            self.is_looking = false;
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
                _ => {},
            }
        }

        // --- 4. Apply Updates to ECS Components ---
        ecs.component_pool.query_exact_mut_for_each::<(Transform, Camera), _>(
            |(transform, _)| {
                transform.rotation = orientation;
                if velocity.length_squared() > 0.0 {
                    transform.position += velocity.normalize() * speed * dt;
                }
            },
        );
    }
}