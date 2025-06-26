use std::sync::Mutex;

use glam::{DVec2, UVec2, Vec2};
use hashbrown::HashSet;
use tracing::info;
use winit::{event::MouseButton, keyboard::KeyCode};

#[derive(Debug, Clone)]
pub enum InputEvent {
    Keyboard { key: KeyCode, pressed: bool },
    MouseButton { button: MouseButton, pressed: bool },
    CursorMoved(DVec2),
    MouseWheel(Vec2),
}

pub struct InputManager {
    event_queue: Mutex<Vec<InputEvent>>,

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
            event_queue: Mutex::new(Vec::new()),
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

    // Called by the main thread. Very fast, low contention.
    pub fn push_event(&self, event: InputEvent) {
        self.event_queue.lock().unwrap().push(event);
    }

    // Called by the logic thread once per tick.
    pub fn process_events(&mut self) {
        // Take ownership of all events queued since the last frame
        let mut queue = self.event_queue.lock().unwrap();
        let events = std::mem::take(&mut *queue);
        drop(queue); // Release the lock immediately

        for event in events {
            match event {
                InputEvent::Keyboard { key, pressed } => {
                    if pressed { self.active_keys.insert(key); }
                    else { self.active_keys.remove(&key); }
                }
                InputEvent::CursorMoved(pos) => {
                    self.cursor_position = pos;
                }
                InputEvent::MouseButton { button, pressed } => {
                    if pressed { self.active_mouse_buttons.insert(button); }
                    else { self.active_mouse_buttons.remove(&button); }
                }
                InputEvent::MouseWheel(delta) => {
                    self.add_scroll(delta);
                }
            }
        }
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