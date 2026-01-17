use std::f32::consts::FRAC_PI_2;

use glam::{DVec2, Quat, Vec3};
use helmer::{
    provided::components::{ActiveCamera, Camera, Transform},
    runtime::input_manager::InputManager,
};
use helmer_ecs::ecs::system::System;
use winit::{event::MouseButton, keyboard::KeyCode};

pub struct FreecamSystem {
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
        ecs: &mut helmer_ecs::ecs::ecs_core::ECSCore,
        input_manager: &InputManager,
    ) {
        // --- 1. Handle Rotation (Mouse and Controller Right Stick) ---
        if input_manager.is_mouse_button_active(MouseButton::Right) {
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

        if input_manager.is_key_active(KeyCode::ShiftLeft) {
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
                    let boost_active = input_manager.is_key_active(KeyCode::ShiftLeft)
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
