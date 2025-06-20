use glam::{DVec2, UVec2, Vec2};
use hashbrown::HashSet;
use tracing::info;
use winit::{event::MouseButton, keyboard::KeyCode};

pub struct InputManager {
    pub active_keys: HashSet<KeyCode>,
    pub active_mouse_buttons: HashSet<MouseButton>,
    pub mouse_wheel: Vec2,
    mouse_wheel_accumulator: Vec2,
    pub cursor_position: DVec2,
    pub window_size: UVec2,
}

impl InputManager {
    pub fn new() -> Self {
        let instance = Self {
            active_keys: HashSet::new(),
            active_mouse_buttons: HashSet::new(),
            mouse_wheel: Vec2::ZERO,
            mouse_wheel_accumulator: Vec2::ZERO,
            cursor_position: DVec2::ZERO,
            window_size: UVec2::ZERO,
        };

        info!("initialized InputManager");

        return instance;
    }

    pub fn is_key_active(&self, keycode: &KeyCode) -> bool {
        self.active_keys.contains(keycode)
    }

    pub fn is_mouse_button_active(&self, button: &MouseButton) -> bool {
        self.active_mouse_buttons.contains(button)
    }

    pub fn add_scroll(&mut self, delta: glam::Vec2) {
        self.mouse_wheel_accumulator += delta;
    }

    pub fn prepare_for_next_frame(&mut self) {
        self.mouse_wheel = self.mouse_wheel_accumulator;
        self.mouse_wheel_accumulator = Vec2::ZERO;
    }
}