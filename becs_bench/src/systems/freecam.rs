use std::f32::consts::FRAC_PI_2;

use bevy_ecs::{
    prelude::{Query, Res, With},
    system::Local,
};
use glam::{DVec2, Quat, Vec3};

use helmer::{
    provided::components::{Camera, Transform},
    runtime::input_manager::InputManager,
};
use helmer_becs::BevyInputManager;
use winit::{event::MouseButton, keyboard::KeyCode};

use crate::{BevyActiveCamera, BevyCamera, BevyTransform, DeltaTime};

//================================================================================
// Helper Functions
//================================================================================

/// Extracts yaw and pitch from a quaternion assuming Y-up, Z-forward.
fn extract_yaw_pitch(rot: Quat) -> (f32, f32) {
    let forward = rot * Vec3::Z;

    let yaw = forward.x.atan2(forward.z);
    let pitch = (-forward.y).asin();

    (yaw, pitch)
}

//================================================================================
// Local State
//================================================================================

#[derive(Default)]
pub struct FreecamState {
    speed: f32,
    sensitivity: f32,
    fov_lerp_speed: f32,
    is_looking: bool,
    last_cursor_position: DVec2,
    current_fov_multiplier: f32,
}

//================================================================================
// System
//================================================================================

pub fn freecam_system(
    mut state: Local<FreecamState>,
    input: Res<BevyInputManager>,
    time: Res<DeltaTime>,
    mut query: Query<(&mut BevyTransform, &mut BevyCamera), With<BevyActiveCamera>>,
) {
    // Initialize state if not set
    if state.speed == 0.0 {
        state.speed = 1.0; // Default speed
        state.sensitivity = 0.3; // Default sensitivity
        state.fov_lerp_speed = 8.0;
        state.current_fov_multiplier = 1.0;
    }

    let dt = time.0;
    let input_manager = &input.0.read();

    const PITCH_LIMIT: f32 = FRAC_PI_2 - 0.01;
    const BOOST_AMOUNT: f32 = 1.15;
    const CONTROLLER_SENSITIVITY: f32 = 2.0;

    let maybe_gamepad_id = input_manager.first_gamepad_id();

    // Rotation input deltas
    let mut yaw_delta = 0.0;
    let mut pitch_delta = 0.0;

    if input_manager.is_mouse_button_active(MouseButton::Right) {
        if !state.is_looking {
            state.last_cursor_position = input_manager.cursor_position;
            state.is_looking = true;
        } else {
            let cursor_delta = input_manager.cursor_position - state.last_cursor_position;
            state.last_cursor_position = input_manager.cursor_position;

            yaw_delta -= cursor_delta.x as f32 * state.sensitivity / 100.0;
            pitch_delta += cursor_delta.y as f32 * state.sensitivity / 100.0;
        }
    } else {
        state.is_looking = false;
    }

    if let Some(gamepad_id) = maybe_gamepad_id {
        yaw_delta -= input_manager.get_controller_axis(gamepad_id, gilrs::Axis::RightStickX)
            * CONTROLLER_SENSITIVITY
            * dt;
        pitch_delta -= input_manager.get_controller_axis(gamepad_id, gilrs::Axis::RightStickY)
            * CONTROLLER_SENSITIVITY
            * dt;
    }

    // Adjust movement speed
    state.speed += input_manager.mouse_wheel.y * 2.0;

    if let Some(gamepad_id) = maybe_gamepad_id {
        if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::RightTrigger) {
            state.speed += 10.0 * dt;
        }
        if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::LeftTrigger) {
            state.speed -= 10.0 * dt;
        }
    }

    state.speed = state.speed.max(0.5);
    let mut speed = state.speed;

    // Sprint boost check
    let mut boost_active = input_manager.is_key_active(KeyCode::ShiftLeft);

    if let Some(gamepad_id) = maybe_gamepad_id {
        if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::LeftThumb) {
            boost_active = true;
            speed *= 2.5;
        }
    }

    if boost_active {
        speed *= 2.5;
    }

    for (mut transform, mut camera) in query.iter_mut() {
        let transform = &mut transform.0;
        let camera = &mut camera.0;

        // Extract current yaw & pitch from rotation
        let (mut yaw, mut pitch) = extract_yaw_pitch(transform.rotation);

        // Apply rotation deltas
        yaw += yaw_delta;
        pitch += pitch_delta;

        pitch = pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);

        // Rebuild rotation
        let yaw_rot = Quat::from_axis_angle(Vec3::Y, yaw);
        let pitch_rot = Quat::from_axis_angle(Vec3::X, pitch);
        let orientation = yaw_rot * pitch_rot;

        transform.rotation = orientation;

        // Movement vectors
        let forward = orientation * Vec3::Z;
        let right = orientation * -Vec3::X;

        let mut velocity = Vec3::ZERO;

        for key in &input_manager.active_keys {
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

        if let Some(gamepad_id) = maybe_gamepad_id {
            let lx = input_manager.get_controller_axis(gamepad_id, gilrs::Axis::LeftStickX);
            let ly = input_manager.get_controller_axis(gamepad_id, gilrs::Axis::LeftStickY);
            velocity += right * lx;
            velocity += forward * ly;

            let up = input_manager.get_right_trigger_value(gamepad_id);
            let down = input_manager.get_left_trigger_value(gamepad_id);
            velocity += Vec3::Y * up;
            velocity -= Vec3::Y * down;
        }

        if let Some(norm_velocity) = velocity.try_normalize() {
            transform.position += norm_velocity * speed * dt;
        }

        // FOV boost handling
        let target_multiplier = if boost_active { BOOST_AMOUNT } else { 1.0 };
        let safe_multiplier = state.current_fov_multiplier.clamp(0.01, BOOST_AMOUNT);
        let base_fov = camera.fov_y_rad / safe_multiplier;
        let t = 1.0 - (-state.fov_lerp_speed * dt).exp();
        state.current_fov_multiplier += (target_multiplier - state.current_fov_multiplier) * t;
        camera.fov_y_rad = base_fov * state.current_fov_multiplier;
    }
}
