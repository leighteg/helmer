use std::sync::Mutex;

use egui::Pos2;
use gilrs::{Axis, Button, GamepadId, Gilrs};
use glam::{DVec2, UVec2, Vec2};
use hashbrown::{HashMap, HashSet};
use tracing::{info, warn};
use winit::{
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{Key, KeyCode, ModifiersState, NamedKey},
};

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
    pub just_pressed: HashSet<KeyCode>,
    pub active_mouse_buttons: HashSet<MouseButton>,
    pub mouse_wheel: Vec2,
    mouse_wheel_accumulator: Vec2,
    pub cursor_position: DVec2,
    pub window_size: UVec2,
    pub scale_factor: f64,

    // Controller state
    pub controller_states: HashMap<GamepadId, ControllerState>,

    // egui
    pub egui_events: Mutex<Vec<egui::Event>>,
    pub egui_modifiers: Mutex<egui::Modifiers>,
    pub egui_pointer_pos: Mutex<Option<egui::Pos2>>,
    pub egui_pointer_down: Mutex<bool>,
    pub egui_last_pointer_pos: Mutex<Option<egui::Pos2>>,
    pub egui_last_pointer_down: Mutex<bool>,
    pub egui_wants_pointer: bool,
    pub egui_wants_key: bool,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            event_queue: Mutex::new(Vec::new()),
            gilrs: Mutex::new(Gilrs::new().unwrap()),
            active_keys: HashSet::new(),
            just_pressed: HashSet::new(),
            active_mouse_buttons: HashSet::new(),
            mouse_wheel: Vec2::ZERO,
            mouse_wheel_accumulator: Vec2::ZERO,
            cursor_position: DVec2::ZERO,
            window_size: UVec2::ZERO,
            scale_factor: 1.0,
            controller_states: HashMap::new(),
            egui_events: Mutex::new(Vec::new()),
            egui_modifiers: Mutex::new(egui::Modifiers::default()),
            egui_pointer_pos: Mutex::new(None),
            egui_pointer_down: Mutex::new(false),
            egui_last_pointer_pos: Mutex::new(None),
            egui_last_pointer_down: Mutex::new(false),
            egui_wants_pointer: false,
            egui_wants_key: false,
        }
    }

    /// Called by the main/window thread to queue keyboard and mouse events.
    pub fn push_event(&self, event: InputEvent) {
        self.event_queue.lock().unwrap().push(event);
    }

    /// Called by the logic thread once per tick to process all pending input.
    pub fn process_events(&mut self) {
        self.just_pressed.clear();

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
                        if button != gilrs::Button::RightTrigger2
                            && button != gilrs::Button::LeftTrigger2
                        {
                            self.controller_states
                                .entry(id)
                                .or_default()
                                .active_buttons
                                .insert(button);
                        }
                    }
                    gilrs::EventType::ButtonReleased(button, _) => {
                        if button != gilrs::Button::RightTrigger2
                            && button != gilrs::Button::LeftTrigger2
                        {
                            self.controller_states
                                .entry(id)
                                .or_default()
                                .active_buttons
                                .remove(&button);
                        }
                    }

                    gilrs::EventType::AxisChanged(axis, value, _) => {
                        const DEADZONE: f32 = 0.15;
                        let final_value = if value.abs() < DEADZONE { 0.0 } else { value };
                        self.controller_states
                            .entry(id)
                            .or_default()
                            .axis_values
                            .insert(axis, final_value);
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
                    if self.egui_wants_key {
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
                    if !self.egui_wants_pointer {
                        self.cursor_position = pos;
                    }
                }
                InputEvent::MouseButton { button, pressed } => {
                    if self.egui_wants_pointer {
                        continue;
                    }

                    if pressed {
                        self.active_mouse_buttons.insert(button);
                    } else {
                        self.active_mouse_buttons.remove(&button);
                    }
                }
                InputEvent::MouseWheel(delta) => {
                    if self.egui_wants_pointer {
                        continue;
                    }

                    self.add_scroll(delta);
                }
            }
        }
    }

    // --- Query Methods ---

    pub fn is_key_active(&self, keycode: KeyCode) -> bool {
        self.active_keys.contains(&keycode)
    }
    pub fn was_just_pressed(&self, key: KeyCode) -> bool {
        self.just_pressed.contains(&key)
    }
    pub fn is_mouse_button_active(&self, button: MouseButton) -> bool {
        self.active_mouse_buttons.contains(&button)
    }
    pub fn is_controller_button_active(&self, gamepad_id: GamepadId, button: Button) -> bool {
        self.controller_states
            .get(&gamepad_id)
            .map_or(false, |state| state.active_buttons.contains(&button))
    }
    pub fn get_controller_axis(&self, gamepad_id: GamepadId, axis: Axis) -> f32 {
        self.controller_states
            .get(&gamepad_id)
            .and_then(|state| state.axis_values.get(&axis))
            .copied()
            .unwrap_or(0.0)
    }
    pub fn first_gamepad_id(&self) -> Option<GamepadId> {
        self.controller_states.keys().next().copied()
    }

    // --- Direct Trigger Query Functions ---

    /// Gets the analog value of the right trigger.
    pub fn get_right_trigger_value(&self, gamepad_id: GamepadId) -> f32 {
        self.controller_states
            .get(&gamepad_id)
            .map_or(0.0, |state| state.right_trigger_value)
    }

    /// Gets the analog value of the left trigger.
    pub fn get_left_trigger_value(&self, gamepad_id: GamepadId) -> f32 {
        self.controller_states
            .get(&gamepad_id)
            .map_or(0.0, |state| state.left_trigger_value)
    }

    // --- Internal State Management ---

    pub fn add_scroll(&mut self, delta: glam::Vec2) {
        self.mouse_wheel_accumulator += delta;
    }
    pub fn prepare_for_next_frame(&mut self) {
        self.mouse_wheel = self.mouse_wheel_accumulator;
        self.mouse_wheel_accumulator = Vec2::ZERO;
    }

    // --- EGUI ---

    pub fn update_egui_state_from_winit(&self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let pos = egui::pos2(
                    (position.x / self.scale_factor) as f32,
                    (position.y / self.scale_factor) as f32,
                );
                *self.egui_pointer_pos.lock().unwrap() = Some(pos);
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                *self.egui_pointer_down.lock().unwrap() = state.is_pressed();
            }

            WindowEvent::ModifiersChanged(new_state) => {
                let mut mods = self.egui_modifiers.lock().unwrap();
                mods.shift = new_state.state().contains(ModifiersState::SHIFT);
                mods.ctrl = new_state.state().contains(ModifiersState::CONTROL);
                mods.alt = new_state.state().contains(ModifiersState::ALT);
                mods.command = new_state.state().contains(ModifiersState::SUPER);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                let pressed = state == &ElementState::Pressed;

                // Only handle physical characters and known named keys
                if let Key::Named(named) = logical_key {
                    if let Some(egui_key) = winit_key_to_egui(*named) {
                        let mut events = self.egui_events.lock().unwrap();
                        events.push(egui::Event::Key {
                            key: egui_key,
                            physical_key: None,
                            pressed,
                            repeat: *repeat,
                            modifiers: *self.egui_modifiers.lock().unwrap(),
                        });
                    }
                } else if let Key::Character(c) = logical_key {
                    if pressed {
                        let mut events = self.egui_events.lock().unwrap();
                        events.push(egui::Event::Text(c.to_string()));
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;

                let mut events = self.egui_events.lock().unwrap();
                let modifiers = *self.egui_modifiers.lock().unwrap();

                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        events.push(egui::Event::MouseWheel {
                            unit: egui::MouseWheelUnit::Line,
                            delta: egui::vec2(*x, *y),
                            modifiers,
                        });
                    }
                    MouseScrollDelta::PixelDelta(p) => {
                        events.push(egui::Event::MouseWheel {
                            unit: egui::MouseWheelUnit::Point,
                            delta: egui::vec2(p.x as f32, p.y as f32),
                            modifiers,
                        });
                    }
                }
            }

            _ => {}
        }
    }

    pub fn build_egui_raw_input(&self, screen_size: UVec2) -> egui::RawInput {
        let pointer_pos = *self.egui_pointer_pos.lock().unwrap();
        let pointer_down = *self.egui_pointer_down.lock().unwrap();
        let last_pointer_pos = *self.egui_last_pointer_pos.lock().unwrap();
        let last_pointer_down = *self.egui_last_pointer_down.lock().unwrap();
        let modifiers = *self.egui_modifiers.lock().unwrap();

        let mut events = self
            .egui_events
            .lock()
            .unwrap()
            .drain(..)
            .collect::<Vec<_>>();

        if pointer_pos != last_pointer_pos {
            if let Some(pos) = pointer_pos {
                events.push(egui::Event::PointerMoved(pos));
            }
        }

        if pointer_down != last_pointer_down {
            if let Some(pos) = pointer_pos {
                events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: pointer_down,
                    modifiers,
                });
            }
        }

        *self.egui_last_pointer_pos.lock().unwrap() = pointer_pos;
        *self.egui_last_pointer_down.lock().unwrap() = pointer_down;

        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(screen_size.x as f32, screen_size.y as f32),
            )),
            modifiers,
            events,
            ..Default::default()
        }
    }
}

fn winit_key_to_egui(key: winit::keyboard::NamedKey) -> Option<egui::Key> {
    use egui::Key as EguiKey;
    use winit::keyboard::NamedKey as WinitKey;

    Some(match key {
        WinitKey::ArrowUp => EguiKey::ArrowUp,
        WinitKey::ArrowDown => EguiKey::ArrowDown,
        WinitKey::ArrowLeft => EguiKey::ArrowLeft,
        WinitKey::ArrowRight => EguiKey::ArrowRight,
        WinitKey::Escape => EguiKey::Escape,
        WinitKey::Tab => EguiKey::Tab,
        WinitKey::Backspace => EguiKey::Backspace,
        WinitKey::Enter => EguiKey::Enter,
        WinitKey::Space => EguiKey::Space,
        WinitKey::Delete => EguiKey::Delete,
        WinitKey::Home => EguiKey::Home,
        WinitKey::End => EguiKey::End,
        WinitKey::PageUp => EguiKey::PageUp,
        WinitKey::PageDown => EguiKey::PageDown,
        WinitKey::Insert => EguiKey::Insert,
        WinitKey::F1 => EguiKey::F1,
        WinitKey::F2 => EguiKey::F2,
        WinitKey::F3 => EguiKey::F3,
        WinitKey::F4 => EguiKey::F4,
        WinitKey::F5 => EguiKey::F5,
        WinitKey::F6 => EguiKey::F6,
        WinitKey::F7 => EguiKey::F7,
        WinitKey::F8 => EguiKey::F8,
        WinitKey::F9 => EguiKey::F9,
        WinitKey::F10 => EguiKey::F10,
        WinitKey::F11 => EguiKey::F11,
        WinitKey::F12 => EguiKey::F12,
        _ => return None,
    })
}
