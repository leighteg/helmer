#[cfg(not(target_arch = "wasm32"))]
use gilrs::Gilrs;
#[cfg(not(target_arch = "wasm32"))]
pub use gilrs::{Axis, Button, GamepadId};
use glam::{DVec2, UVec2, Vec2};
use hashbrown::{HashMap, HashSet};
use parking_lot::Mutex;
#[cfg(target_arch = "wasm32")]
use parking_lot::RwLock;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use winit::{event::MouseButton, keyboard::KeyCode};

mod egui_bridge;
mod events;
mod gamepad;

#[cfg(target_arch = "wasm32")]
mod web_gamepad {
    pub type GamepadId = usize;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Axis {
        LeftStickX,
        LeftStickY,
        RightStickX,
        RightStickY,
        LeftZ,
        RightZ,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Button {
        South,
        East,
        West,
        North,
        LeftShoulder,
        RightShoulder,
        LeftTrigger,
        RightTrigger,
        LeftTrigger2,
        RightTrigger2,
        LeftThumb,
        RightThumb,
        Select,
        Start,
        Mode,
        DPadUp,
        DPadDown,
        DPadLeft,
        DPadRight,
    }
}

#[cfg(target_arch = "wasm32")]
pub use web_gamepad::{Axis, Button, GamepadId};

#[derive(Debug, Clone)]
pub enum InputEvent {
    Keyboard { key: KeyCode, pressed: bool },
    MouseButton { button: MouseButton, pressed: bool },
    CursorMoved(DVec2),
    MouseMotion(DVec2),
    MouseWheel(Vec2),
}

#[derive(Default, Debug)]
pub struct ControllerState {
    pub active_buttons: HashSet<Button>,
    pub axis_values: HashMap<Axis, f32>,
    pub right_trigger_value: f32,
    pub left_trigger_value: f32,
}

pub struct InputManager {
    event_queue: Mutex<Vec<InputEvent>>,
    #[cfg(not(target_arch = "wasm32"))]
    gilrs: Mutex<Gilrs>,

    pub active_keys: HashSet<KeyCode>,
    pub just_pressed: HashSet<KeyCode>,
    pub active_mouse_buttons: HashSet<MouseButton>,
    pub mouse_wheel: Vec2,
    pub mouse_wheel_accumulator: Vec2,
    pub mouse_wheel_unfiltered: Vec2,
    pub mouse_wheel_unfiltered_accumulator: Vec2,
    pub mouse_motion: DVec2,
    pub cursor_position: DVec2,
    pub window_size: UVec2,
    pub scale_factor: f64,

    pub controller_states: HashMap<GamepadId, ControllerState>,

    pub egui_events: Mutex<Vec<egui::Event>>,
    pub egui_modifiers: Mutex<egui::Modifiers>,
    pub egui_pointer_pos: Mutex<Option<egui::Pos2>>,
    pub egui_pointer_down: Mutex<bool>,
    pub egui_pointer_down_secondary: Mutex<bool>,
    pub egui_pointer_down_middle: Mutex<bool>,
    pub egui_last_pointer_pos: Mutex<Option<egui::Pos2>>,
    pub egui_last_pointer_down: Mutex<bool>,
    pub egui_last_pointer_down_secondary: Mutex<bool>,
    pub egui_last_pointer_down_middle: Mutex<bool>,
    pub egui_wants_pointer: bool,
    pub egui_wants_key: bool,

    pub ui_wants_pointer: bool,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_INPUT_MANAGER: std::cell::RefCell<Option<std::sync::Arc<RwLock<InputManager>>>> =
        std::cell::RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
pub fn register_web_input_manager(manager: std::sync::Arc<RwLock<InputManager>>) {
    WEB_INPUT_MANAGER.with(|slot| {
        *slot.borrow_mut() = Some(manager);
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn helmer_clear_input_state() {
    WEB_INPUT_MANAGER.with(|slot| {
        let binding = slot.borrow();
        let Some(manager) = binding.as_ref() else {
            return;
        };
        let mut guard = manager.write();
        guard.clear_egui_state();
        guard.clear_queues();
    });
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            event_queue: Mutex::new(Vec::new()),
            #[cfg(not(target_arch = "wasm32"))]
            gilrs: Mutex::new(Gilrs::new().unwrap()),
            active_keys: HashSet::new(),
            just_pressed: HashSet::new(),
            active_mouse_buttons: HashSet::new(),
            mouse_wheel: Vec2::ZERO,
            mouse_wheel_accumulator: Vec2::ZERO,
            mouse_wheel_unfiltered: Vec2::ZERO,
            mouse_wheel_unfiltered_accumulator: Vec2::ZERO,
            mouse_motion: DVec2::ZERO,
            cursor_position: DVec2::ZERO,
            window_size: UVec2::ZERO,
            scale_factor: 1.0,
            controller_states: HashMap::new(),
            egui_events: Mutex::new(Vec::new()),
            egui_modifiers: Mutex::new(egui::Modifiers::default()),
            egui_pointer_pos: Mutex::new(None),
            egui_pointer_down: Mutex::new(false),
            egui_pointer_down_secondary: Mutex::new(false),
            egui_pointer_down_middle: Mutex::new(false),
            egui_last_pointer_pos: Mutex::new(None),
            egui_last_pointer_down: Mutex::new(false),
            egui_last_pointer_down_secondary: Mutex::new(false),
            egui_last_pointer_down_middle: Mutex::new(false),
            egui_wants_pointer: false,
            egui_wants_key: false,
            ui_wants_pointer: false,
        }
    }

    pub fn push_event(&self, event: InputEvent) {
        self.event_queue.lock().push(event);
    }

    pub fn process_events(&mut self) {
        self.just_pressed.clear();
        self.mouse_motion = DVec2::ZERO;
        self.poll_controller_state();
        self.process_window_events();
    }
}
