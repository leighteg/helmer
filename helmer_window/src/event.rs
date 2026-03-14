use glam::{DVec2, Vec2};
use std::path::PathBuf;
use std::sync::Arc;
use winit::window::Window;
use winit::{event::MouseButton, keyboard::KeyCode};

#[derive(Debug, Clone, Copy)]
pub struct WindowState {
    pub width: u32,
    pub height: u32,
    pub scale_factor: f64,
}

#[derive(Debug, Clone)]
pub enum WindowRuntimeEventKind {
    Started {
        window: Arc<Window>,
        state: WindowState,
    },
    Resized(WindowState),
    Keyboard {
        key: KeyCode,
        pressed: bool,
    },
    MouseButton {
        button: MouseButton,
        pressed: bool,
    },
    MouseMoved(DVec2),
    MouseMotion(DVec2),
    MouseWheel(Vec2),
    DroppedFile(PathBuf),
    CloseRequested,
    Tick {
        dt: f32,
    },
}

#[derive(Debug, Clone)]
pub struct WindowRuntimeEvent {
    pub kind: WindowRuntimeEventKind,
}
