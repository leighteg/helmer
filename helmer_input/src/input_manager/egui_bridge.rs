use crate::input_manager::InputManager;
use glam::UVec2;
use winit::{
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{Key, ModifiersState},
};

impl InputManager {
    pub fn update_egui_state_from_winit(&self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let pos = egui::pos2(
                    (position.x / self.scale_factor) as f32,
                    (position.y / self.scale_factor) as f32,
                );
                *self.egui_pointer_pos.lock() = Some(pos);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                *self.egui_pointer_down.lock() = state.is_pressed();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                *self.egui_pointer_down_secondary.lock() = state.is_pressed();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Middle,
                ..
            } => {
                *self.egui_pointer_down_middle.lock() = state.is_pressed();
            }
            WindowEvent::ModifiersChanged(new_state) => {
                let state = new_state.state();
                let mut mods = self.egui_modifiers.lock();
                mods.shift = state.contains(ModifiersState::SHIFT);
                mods.ctrl = state.contains(ModifiersState::CONTROL);
                mods.alt = state.contains(ModifiersState::ALT);
                #[cfg(target_os = "macos")]
                {
                    mods.mac_cmd = state.contains(ModifiersState::SUPER);
                    mods.command = mods.mac_cmd;
                }
                #[cfg(not(target_os = "macos"))]
                {
                    mods.mac_cmd = false;
                    mods.command = mods.ctrl;
                }
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

                if let Key::Named(named) = logical_key {
                    if let Some(egui_key) = winit_key_to_egui(*named) {
                        let mut events = self.egui_events.lock();
                        events.push(egui::Event::Key {
                            key: egui_key,
                            physical_key: None,
                            pressed,
                            repeat: *repeat,
                            modifiers: *self.egui_modifiers.lock(),
                        });
                        if pressed && matches!(named, winit::keyboard::NamedKey::Space) {
                            events.push(egui::Event::Text(" ".to_string()));
                        }
                    }
                } else if let Key::Character(c) = logical_key {
                    let mut events = self.egui_events.lock();
                    let modifiers = *self.egui_modifiers.lock();

                    if let Some(egui_key) = winit_char_to_egui_key(c.as_ref()) {
                        events.push(egui::Event::Key {
                            key: egui_key,
                            physical_key: None,
                            pressed,
                            repeat: *repeat,
                            modifiers,
                        });
                    }

                    if pressed && !is_command_shortcut(modifiers) {
                        events.push(egui::Event::Text(c.to_string()));
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;

                let mut events = self.egui_events.lock();
                let modifiers = *self.egui_modifiers.lock();

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
        let pointer_pos = *self.egui_pointer_pos.lock();
        let pointer_down = *self.egui_pointer_down.lock();
        let pointer_down_secondary = *self.egui_pointer_down_secondary.lock();
        let pointer_down_middle = *self.egui_pointer_down_middle.lock();
        let last_pointer_pos = *self.egui_last_pointer_pos.lock();
        let last_pointer_down = *self.egui_last_pointer_down.lock();
        let last_pointer_down_secondary = *self.egui_last_pointer_down_secondary.lock();
        let last_pointer_down_middle = *self.egui_last_pointer_down_middle.lock();
        let modifiers = *self.egui_modifiers.lock();

        let mut events = self.egui_events.lock().drain(..).collect::<Vec<_>>();

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

        if pointer_down_secondary != last_pointer_down_secondary {
            if let Some(pos) = pointer_pos {
                events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Secondary,
                    pressed: pointer_down_secondary,
                    modifiers,
                });
            }
        }

        if pointer_down_middle != last_pointer_down_middle {
            if let Some(pos) = pointer_pos {
                events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Middle,
                    pressed: pointer_down_middle,
                    modifiers,
                });
            }
        }

        *self.egui_last_pointer_pos.lock() = pointer_pos;
        *self.egui_last_pointer_down.lock() = pointer_down;
        *self.egui_last_pointer_down_secondary.lock() = pointer_down_secondary;
        *self.egui_last_pointer_down_middle.lock() = pointer_down_middle;

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

    pub fn clear_egui_state(&mut self) {
        self.egui_events.lock().clear();
        self.egui_wants_key = false;
        self.egui_wants_pointer = false;
        *self.egui_pointer_down_secondary.lock() = false;
        *self.egui_pointer_down_middle.lock() = false;
        *self.egui_last_pointer_down.lock() = false;
        *self.egui_last_pointer_down_secondary.lock() = false;
        *self.egui_last_pointer_down_middle.lock() = false;
        *self.egui_last_pointer_pos.lock() = None;
        *self.egui_modifiers.lock() = egui::Modifiers::NONE;
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

fn winit_char_to_egui_key(character: &str) -> Option<egui::Key> {
    let mut chars = character.chars();
    let first = chars.next()?;
    if chars.next().is_some() {
        return None;
    }

    Some(match first.to_ascii_uppercase() {
        '0' => egui::Key::Num0,
        '1' => egui::Key::Num1,
        '2' => egui::Key::Num2,
        '3' => egui::Key::Num3,
        '4' => egui::Key::Num4,
        '5' => egui::Key::Num5,
        '6' => egui::Key::Num6,
        '7' => egui::Key::Num7,
        '8' => egui::Key::Num8,
        '9' => egui::Key::Num9,
        'A' => egui::Key::A,
        'B' => egui::Key::B,
        'C' => egui::Key::C,
        'D' => egui::Key::D,
        'E' => egui::Key::E,
        'F' => egui::Key::F,
        'G' => egui::Key::G,
        'H' => egui::Key::H,
        'I' => egui::Key::I,
        'J' => egui::Key::J,
        'K' => egui::Key::K,
        'L' => egui::Key::L,
        'M' => egui::Key::M,
        'N' => egui::Key::N,
        'O' => egui::Key::O,
        'P' => egui::Key::P,
        'Q' => egui::Key::Q,
        'R' => egui::Key::R,
        'S' => egui::Key::S,
        'T' => egui::Key::T,
        'U' => egui::Key::U,
        'V' => egui::Key::V,
        'W' => egui::Key::W,
        'X' => egui::Key::X,
        'Y' => egui::Key::Y,
        'Z' => egui::Key::Z,
        _ => return None,
    })
}

fn is_command_shortcut(modifiers: egui::Modifiers) -> bool {
    (modifiers.command || modifiers.ctrl) && !modifiers.alt
}
