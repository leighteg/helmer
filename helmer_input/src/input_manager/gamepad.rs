use crate::input_manager::{Axis, Button, GamepadId, InputManager};
#[cfg(target_arch = "wasm32")]
use hashbrown::HashSet;
use tracing::info;

impl InputManager {
    pub(crate) fn poll_controller_state(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_native_gamepads();

        #[cfg(target_arch = "wasm32")]
        self.poll_web_gamepads();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_native_gamepads(&mut self) {
        let mut gilrs = self.gilrs.lock();
        while let Some(gilrs::Event { id, event, .. }) = gilrs.next_event() {
            match event {
                gilrs::EventType::ButtonChanged(button, value, _) => {
                    let deadzone = 0.05;
                    let final_value = if value < deadzone { 0.0 } else { value };
                    let state = self.controller_states.entry(id).or_default();

                    match button {
                        gilrs::Button::RightTrigger2 => state.right_trigger_value = final_value,
                        gilrs::Button::LeftTrigger2 => state.left_trigger_value = final_value,
                        _ => {}
                    }
                }
                gilrs::EventType::ButtonPressed(button, _) => {
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

    pub fn is_controller_button_active(&self, gamepad_id: GamepadId, button: Button) -> bool {
        self.controller_states
            .get(&gamepad_id)
            .is_some_and(|state| state.active_buttons.contains(&button))
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

    pub fn get_right_trigger_value(&self, gamepad_id: GamepadId) -> f32 {
        self.controller_states
            .get(&gamepad_id)
            .map_or(0.0, |state| state.right_trigger_value)
    }

    pub fn get_left_trigger_value(&self, gamepad_id: GamepadId) -> f32 {
        self.controller_states
            .get(&gamepad_id)
            .map_or(0.0, |state| state.left_trigger_value)
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_web_gamepads(&mut self) {
        use wasm_bindgen::JsCast;
        use web_sys::Gamepad;

        let Some(window) = web_sys::window() else {
            return;
        };
        let navigator = window.navigator();
        let gamepads = match navigator.get_gamepads() {
            Ok(list) => list,
            Err(_) => return,
        };

        let previous_ids: HashSet<GamepadId> = self.controller_states.keys().copied().collect();
        let mut connected_ids: HashSet<GamepadId> = HashSet::new();

        for (index, pad_value) in gamepads.iter().enumerate() {
            if pad_value.is_null() || pad_value.is_undefined() {
                continue;
            }
            let gamepad: Gamepad = pad_value.unchecked_into();
            let id = index as GamepadId;
            connected_ids.insert(id);

            let state = self.controller_states.entry(id).or_default();
            state.active_buttons.clear();
            state.axis_values.clear();
            state.left_trigger_value = 0.0;
            state.right_trigger_value = 0.0;

            let buttons = gamepad.buttons();
            let get_button = |idx: u32| -> Option<web_sys::GamepadButton> {
                let value = buttons.get(idx);
                if value.is_null() || value.is_undefined() {
                    None
                } else {
                    Some(value.unchecked_into())
                }
            };

            const TRIGGER_DEADZONE: f32 = 0.05;
            const TRIGGER_THRESHOLD: f32 = 0.5;

            if let Some(button) = get_button(6) {
                let value = button.value() as f32;
                state.left_trigger_value = if value < TRIGGER_DEADZONE { 0.0 } else { value };
                if value > TRIGGER_THRESHOLD {
                    state.active_buttons.insert(Button::LeftTrigger);
                }
            }
            if let Some(button) = get_button(7) {
                let value = button.value() as f32;
                state.right_trigger_value = if value < TRIGGER_DEADZONE { 0.0 } else { value };
                if value > TRIGGER_THRESHOLD {
                    state.active_buttons.insert(Button::RightTrigger);
                }
            }

            const BUTTON_MAP: &[(u32, Button)] = &[
                (0, Button::South),
                (1, Button::East),
                (2, Button::West),
                (3, Button::North),
                (4, Button::LeftShoulder),
                (5, Button::RightShoulder),
                (8, Button::Select),
                (9, Button::Start),
                (10, Button::LeftThumb),
                (11, Button::RightThumb),
                (12, Button::DPadUp),
                (13, Button::DPadDown),
                (14, Button::DPadLeft),
                (15, Button::DPadRight),
                (16, Button::Mode),
            ];

            for (idx, button) in BUTTON_MAP.iter().copied() {
                if let Some(pad_button) = get_button(idx) {
                    if pad_button.pressed() {
                        state.active_buttons.insert(button);
                    }
                }
            }

            let axes = gamepad.axes();
            let mut axis_values: Vec<f32> = Vec::new();
            for idx in 0..axes.length() {
                let value = axes.get(idx).as_f64().unwrap_or(0.0).clamp(-1.0, 1.0) as f32;
                axis_values.push(value);
            }

            const DEADZONE: f32 = 0.15;
            let apply_deadzone =
                |value: f32| -> f32 { if value.abs() < DEADZONE { 0.0 } else { value } };

            if !axis_values.is_empty() {
                state
                    .axis_values
                    .insert(Axis::LeftStickX, apply_deadzone(axis_values[0]));
            }
            if axis_values.len() > 1 {
                state
                    .axis_values
                    .insert(Axis::LeftStickY, apply_deadzone(axis_values[1]));
            }
            if axis_values.len() > 2 {
                state
                    .axis_values
                    .insert(Axis::RightStickX, apply_deadzone(axis_values[2]));
            }
            if axis_values.len() > 3 {
                state
                    .axis_values
                    .insert(Axis::RightStickY, apply_deadzone(axis_values[3]));
            }

            if state.left_trigger_value > 0.0 {
                state
                    .axis_values
                    .insert(Axis::LeftZ, state.left_trigger_value);
            }
            if state.right_trigger_value > 0.0 {
                state
                    .axis_values
                    .insert(Axis::RightZ, state.right_trigger_value);
            }
        }

        for id in previous_ids.difference(&connected_ids) {
            info!("Gamepad {} disconnected", id);
            self.controller_states.remove(id);
        }
        for id in connected_ids.difference(&previous_ids) {
            info!("Gamepad {} connected", id);
            self.controller_states.entry(*id).or_default();
        }
    }
}
