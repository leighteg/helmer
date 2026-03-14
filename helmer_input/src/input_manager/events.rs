use crate::input_manager::{InputEvent, InputManager};
use glam::{DVec2, Vec2};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

impl InputManager {
    pub(crate) fn process_window_events(&mut self) {
        let events = {
            let mut queue = self.event_queue.lock();
            std::mem::take(&mut *queue)
        };

        for event in events {
            match event {
                InputEvent::Keyboard { key, pressed } => {
                    if self.wants_keyboard_input() {
                        continue;
                    }

                    if pressed {
                        if !self.active_keys.contains(&key) {
                            self.just_pressed.insert(key);
                        }
                        self.active_keys.insert(key);
                    } else {
                        self.active_keys.remove(&key);
                    }
                }
                InputEvent::CursorMoved(pos) => {
                    self.cursor_position = pos;
                }
                InputEvent::MouseMotion(delta) => {
                    self.mouse_motion += delta;
                }
                InputEvent::MouseButton { button, pressed } => {
                    if pressed {
                        self.active_mouse_buttons.insert(button);
                    } else {
                        self.active_mouse_buttons.remove(&button);
                    }
                }
                InputEvent::MouseWheel(delta) => {
                    self.add_scroll_unfiltered(delta);
                    if self.wants_pointer_input() {
                        continue;
                    }

                    self.add_scroll(delta);
                }
            }
        }
    }

    pub fn is_key_active(&self, keycode: KeyCode) -> bool {
        self.active_keys.contains(&keycode)
    }

    pub fn was_just_pressed(&self, key: KeyCode) -> bool {
        self.just_pressed.contains(&key)
    }

    pub fn is_mouse_button_active(&self, button: MouseButton) -> bool {
        !self.wants_pointer_input() && self.active_mouse_buttons.contains(&button)
    }

    pub fn wants_pointer_input(&self) -> bool {
        self.egui_wants_pointer || self.ui_wants_pointer
    }

    pub fn wants_keyboard_input(&self) -> bool {
        self.egui_wants_key
    }

    pub fn add_scroll(&mut self, delta: Vec2) {
        self.mouse_wheel_accumulator += delta;
    }

    pub fn add_scroll_unfiltered(&mut self, delta: Vec2) {
        self.mouse_wheel_unfiltered_accumulator += delta;
    }

    pub fn prepare_for_next_frame(&mut self) {
        self.mouse_wheel = self.mouse_wheel_accumulator;
        self.mouse_wheel_accumulator = Vec2::ZERO;
        self.mouse_wheel_unfiltered = self.mouse_wheel_unfiltered_accumulator;
        self.mouse_wheel_unfiltered_accumulator = Vec2::ZERO;
    }

    pub fn clear_just_pressed(&mut self) {
        self.just_pressed.clear();
    }

    pub fn clear_queues(&mut self) {
        self.active_keys.clear();
        self.active_mouse_buttons.clear();
        self.just_pressed.clear();
        self.mouse_wheel = Vec2::ZERO;
        self.mouse_wheel_accumulator = Vec2::ZERO;
        self.mouse_wheel_unfiltered = Vec2::ZERO;
        self.mouse_wheel_unfiltered_accumulator = Vec2::ZERO;
        self.mouse_motion = DVec2::ZERO;
    }
}
