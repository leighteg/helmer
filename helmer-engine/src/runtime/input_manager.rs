use std::sync::Mutex;

use gilrs::{Axis, Button, GamepadId, Gilrs};
use glam::{DVec2, UVec2, Vec2};
use hashbrown::{HashMap, HashSet};
use tracing::{info, warn};
use winit::{event::MouseButton, keyboard::KeyCode};

#[derive(Debug, Clone)]
pub enum InputEvent {
    Keyboard { key: KeyCode, pressed: bool },
    MouseButton { button: MouseButton, pressed: bool },
    CursorMoved(DVec2),
    MouseWheel(Vec2),
}

/// Stores the current state of a connected controller.
#[derive(Default, Debug)]
pub struct ControllerState {
    pub active_buttons: HashSet<Button>,
    pub axis_values: HashMap<Axis, f32>,
    /// The analog value of the right trigger (0.0 to 1.0).
    pub right_trigger_value: f32,
    /// The analog value of the left trigger (0.0 to 1.0).
    pub left_trigger_value: f32,
}

pub struct InputManager {
    event_queue: Mutex<Vec<InputEvent>>,
    gilrs: Mutex<Gilrs>,

    // Keyboard and Mouse state
    pub active_keys: HashSet<KeyCode>,
    pub active_mouse_buttons: HashSet<MouseButton>,
    pub mouse_wheel: Vec2,
    mouse_wheel_accumulator: Vec2,
    pub cursor_position: DVec2,
    pub window_size: UVec2,

    // Controller state
    pub controller_states: HashMap<GamepadId, ControllerState>,
}

impl InputManager {
    pub fn new() -> Self {
        let gilrs_instance = match Gilrs::new() {
            Ok(g) => {
                info!("Gilrs initialized for controller support.");
                g
            }
            Err(e) => {
                warn!("Failed to initialize gilrs, controller input unavailable: {}", e);
                Gilrs::new().unwrap()
            }
        };

        Self {
            event_queue: Mutex::new(Vec::new()),
            gilrs: Mutex::new(gilrs_instance),
            active_keys: HashSet::new(),
            active_mouse_buttons: HashSet::new(),
            mouse_wheel: Vec2::ZERO,
            mouse_wheel_accumulator: Vec2::ZERO,
            cursor_position: DVec2::ZERO,
            window_size: UVec2::ZERO,
            controller_states: HashMap::new(),
        }
    }

    /// Called by the main/window thread to queue keyboard and mouse events.
    pub fn push_event(&self, event: InputEvent) {
        self.event_queue.lock().unwrap().push(event);
    }

    /// Called by the logic thread once per tick to process all pending input.
    pub fn process_events(&mut self) {
        // --- 1. Poll gilrs for controller events and update state directly ---
        {
            let mut gilrs = self.gilrs.lock().unwrap();
            while let Some(gilrs::Event { id, event, .. }) = gilrs.next_event() {
                match event {
                    // Handle analog triggers reported as `ButtonChanged` events.
                    gilrs::EventType::ButtonChanged(button, value, _) => {
                        let deadzone = 0.05;
                        let final_value = if value < deadzone { 0.0 } else { value };
                        let state = self.controller_states.entry(id).or_default();

                        match button {
                            gilrs::Button::RightTrigger2 => state.right_trigger_value = final_value,
                            gilrs::Button::LeftTrigger2 => state.left_trigger_value = final_value,
                            _ => {} // Other buttons are handled by ButtonPressed/Released
                        }
                    }

                    gilrs::EventType::ButtonPressed(button, _) => {
                        // Ensure triggers aren't added to the standard button set.
                        if button != gilrs::Button::RightTrigger2 && button != gilrs::Button::LeftTrigger2 {
                            self.controller_states.entry(id).or_default().active_buttons.insert(button);
                        }
                    }
                    gilrs::EventType::ButtonReleased(button, _) => {
                        if button != gilrs::Button::RightTrigger2 && button != gilrs::Button::LeftTrigger2 {
                            self.controller_states.entry(id).or_default().active_buttons.remove(&button);
                        }
                    }

                    gilrs::EventType::AxisChanged(axis, value, _) => {
                        const DEADZONE: f32 = 0.15;
                        let final_value = if value.abs() < DEADZONE { 0.0 } else { value };
                        self.controller_states.entry(id).or_default().axis_values.insert(axis, final_value);
                    }
                    gilrs::EventType::Connected => {
                        info!("Gamepad {} connected", id);
                        self.controller_states.entry(id).or_default();
                    }
                    gilrs::EventType::Disconnected => {
                        info!("Gamepad {} disconnected", id);
                        self.controller_states.remove(&id);
                    }
                    _ => {}
                }
            }
        }

        // --- 2. Process winit events (keyboard/mouse) from the queue ---
        let events = {
            let mut queue = self.event_queue.lock().unwrap();
            std::mem::take(&mut *queue)
        };

        for event in events {
            match event {
                InputEvent::Keyboard { key, pressed } => {
                    if pressed { self.active_keys.insert(key); } else { self.active_keys.remove(&key); }
                }
                InputEvent::CursorMoved(pos) => { self.cursor_position = pos; }
                InputEvent::MouseButton { button, pressed } => {
                    if pressed { self.active_mouse_buttons.insert(button); } else { self.active_mouse_buttons.remove(&button); }
                }
                InputEvent::MouseWheel(delta) => { self.add_scroll(delta); }
            }
        }
    }
    
    // --- Query Methods ---

    pub fn is_key_active(&self, keycode: &KeyCode) -> bool { self.active_keys.contains(keycode) }
    pub fn is_mouse_button_active(&self, button: &MouseButton) -> bool { self.active_mouse_buttons.contains(button) }
    pub fn is_controller_button_active(&self, gamepad_id: GamepadId, button: Button) -> bool {
        self.controller_states.get(&gamepad_id).map_or(false, |state| state.active_buttons.contains(&button))
    }
    pub fn get_controller_axis(&self, gamepad_id: GamepadId, axis: Axis) -> f32 {
        self.controller_states.get(&gamepad_id).and_then(|state| state.axis_values.get(&axis)).copied().unwrap_or(0.0)
    }
    pub fn first_gamepad_id(&self) -> Option<GamepadId> { self.controller_states.keys().next().copied() }

    // --- Direct Trigger Query Functions ---

    /// Gets the analog value of the right trigger.
    pub fn get_right_trigger_value(&self, gamepad_id: GamepadId) -> f32 {
        self.controller_states.get(&gamepad_id).map_or(0.0, |state| state.right_trigger_value)
    }

    /// Gets the analog value of the left trigger.
    pub fn get_left_trigger_value(&self, gamepad_id: GamepadId) -> f32 {
        self.controller_states.get(&gamepad_id).map_or(0.0, |state| state.left_trigger_value)
    }

    // --- Internal State Management ---

    pub fn add_scroll(&mut self, delta: glam::Vec2) { self.mouse_wheel_accumulator += delta; }
    pub fn prepare_for_next_frame(&mut self) {
        self.mouse_wheel = self.mouse_wheel_accumulator;
        self.mouse_wheel_accumulator = Vec2::ZERO;
    }
}