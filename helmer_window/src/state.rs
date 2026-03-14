use std::sync::{
    RwLock,
    atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeCursorGrabMode {
    #[default]
    None,
    Confined,
    Locked,
}

impl RuntimeCursorGrabMode {
    fn encode(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Confined => 1,
            Self::Locked => 2,
        }
    }

    fn decode(value: u8) -> Self {
        match value {
            1 => Self::Confined,
            2 => Self::Locked,
            _ => Self::None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Confined => "confined",
            Self::Locked => "locked",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeCursorStateSnapshot {
    pub visible: bool,
    pub grab_mode: RuntimeCursorGrabMode,
}

impl Default for RuntimeCursorStateSnapshot {
    fn default() -> Self {
        Self {
            visible: true,
            grab_mode: RuntimeCursorGrabMode::None,
        }
    }
}

#[derive(Debug)]
pub struct RuntimeCursorState {
    visible: AtomicBool,
    grab_mode: AtomicU8,
    warp_pending: AtomicBool,
    warp_x_bits: AtomicU64,
    warp_y_bits: AtomicU64,
    dirty: AtomicBool,
}

impl Default for RuntimeCursorState {
    fn default() -> Self {
        Self {
            visible: AtomicBool::new(true),
            grab_mode: AtomicU8::new(RuntimeCursorGrabMode::None.encode()),
            warp_pending: AtomicBool::new(false),
            warp_x_bits: AtomicU64::new(0),
            warp_y_bits: AtomicU64::new(0),
            dirty: AtomicBool::new(true),
        }
    }
}

impl RuntimeCursorState {
    pub fn snapshot(&self) -> RuntimeCursorStateSnapshot {
        RuntimeCursorStateSnapshot {
            visible: self.visible.load(Ordering::Relaxed),
            grab_mode: RuntimeCursorGrabMode::decode(self.grab_mode.load(Ordering::Relaxed)),
        }
    }

    pub fn set_visible(&self, visible: bool) -> bool {
        let changed = self.visible.swap(visible, Ordering::AcqRel) != visible;
        if changed {
            self.dirty.store(true, Ordering::Release);
        }
        changed
    }

    pub fn set_grab_mode(&self, grab_mode: RuntimeCursorGrabMode) -> bool {
        let mode = grab_mode.encode();
        let changed = self.grab_mode.swap(mode, Ordering::AcqRel) != mode;
        if changed {
            self.dirty.store(true, Ordering::Release);
        }
        changed
    }

    pub fn set(&self, snapshot: RuntimeCursorStateSnapshot) -> bool {
        let mut changed = false;
        changed |= self.set_visible(snapshot.visible);
        changed |= self.set_grab_mode(snapshot.grab_mode);
        changed
    }

    pub fn reset(&self) -> bool {
        self.set(RuntimeCursorStateSnapshot::default())
    }

    pub fn request_warp(&self, x: f64, y: f64) {
        self.warp_x_bits.store(x.to_bits(), Ordering::Release);
        self.warp_y_bits.store(y.to_bits(), Ordering::Release);
        self.warp_pending.store(true, Ordering::Release);
    }

    pub fn take_warp_request(&self) -> Option<(f64, f64)> {
        if !self.warp_pending.swap(false, Ordering::AcqRel) {
            return None;
        }
        let x = f64::from_bits(self.warp_x_bits.load(Ordering::Acquire));
        let y = f64::from_bits(self.warp_y_bits.load(Ordering::Acquire));
        Some((x, y))
    }

    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::AcqRel)
    }

    pub fn has_pending_update(&self) -> bool {
        self.dirty.load(Ordering::Acquire) || self.warp_pending.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeWindowTitleMode {
    #[default]
    Stats,
    Custom,
    CustomWithStats,
}

impl RuntimeWindowTitleMode {
    fn encode(self) -> u8 {
        match self {
            Self::Stats => 0,
            Self::Custom => 1,
            Self::CustomWithStats => 2,
        }
    }

    fn decode(value: u8) -> Self {
        match value {
            1 => Self::Custom,
            2 => Self::CustomWithStats,
            _ => Self::Stats,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stats => "stats",
            Self::Custom => "custom",
            Self::CustomWithStats => "custom_with_stats",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeWindowControlSnapshot {
    pub title_mode: RuntimeWindowTitleMode,
    pub custom_title: String,
    pub title_update_ms: u32,
    pub target_tickrate: f32,
    pub target_fps: Option<f32>,
    pub fullscreen: bool,
    pub resizable: bool,
    pub decorations: bool,
    pub maximized: bool,
    pub minimized: bool,
    pub visible: bool,
}

#[derive(Debug)]
pub struct RuntimeWindowControl {
    title_mode: AtomicU8,
    custom_title: RwLock<String>,
    title_update_ms: AtomicU32,
    target_tickrate_bits: AtomicU32,
    target_fps_bits: AtomicU32,
    fullscreen: AtomicBool,
    resizable: AtomicBool,
    decorations: AtomicBool,
    maximized: AtomicBool,
    minimized: AtomicBool,
    visible: AtomicBool,
}

impl Default for RuntimeWindowControl {
    fn default() -> Self {
        Self {
            title_mode: AtomicU8::new(RuntimeWindowTitleMode::Stats.encode()),
            custom_title: RwLock::new("helmer engine".to_string()),
            title_update_ms: AtomicU32::new(200),
            target_tickrate_bits: AtomicU32::new(120.0f32.to_bits()),
            target_fps_bits: AtomicU32::new(0.0f32.to_bits()),
            fullscreen: AtomicBool::new(false),
            resizable: AtomicBool::new(true),
            decorations: AtomicBool::new(true),
            maximized: AtomicBool::new(false),
            minimized: AtomicBool::new(false),
            visible: AtomicBool::new(true),
        }
    }
}

impl RuntimeWindowControl {
    pub fn snapshot(&self) -> RuntimeWindowControlSnapshot {
        RuntimeWindowControlSnapshot {
            title_mode: self.title_mode(),
            custom_title: self.custom_title(),
            title_update_ms: self.title_update_ms(),
            target_tickrate: self.target_tickrate(),
            target_fps: self.target_fps(),
            fullscreen: self.fullscreen(),
            resizable: self.resizable(),
            decorations: self.decorations(),
            maximized: self.maximized(),
            minimized: self.minimized(),
            visible: self.visible(),
        }
    }

    pub fn title_mode(&self) -> RuntimeWindowTitleMode {
        RuntimeWindowTitleMode::decode(self.title_mode.load(Ordering::Relaxed))
    }

    pub fn set_title_mode(&self, mode: RuntimeWindowTitleMode) {
        self.title_mode.store(mode.encode(), Ordering::Relaxed);
    }

    pub fn custom_title(&self) -> String {
        self.custom_title
            .read()
            .map(|title| title.clone())
            .unwrap_or_else(|_| "helmer engine".to_string())
    }

    pub fn set_custom_title(&self, title: impl Into<String>) {
        if let Ok(mut current) = self.custom_title.write() {
            *current = title.into();
        }
    }

    pub fn title_update_ms(&self) -> u32 {
        self.title_update_ms.load(Ordering::Relaxed).max(1)
    }

    pub fn set_title_update_ms(&self, value: u32) {
        self.title_update_ms.store(value.max(1), Ordering::Relaxed);
    }

    pub fn target_tickrate(&self) -> f32 {
        let value = f32::from_bits(self.target_tickrate_bits.load(Ordering::Relaxed));
        if value.is_finite() && value > 0.0 {
            value
        } else {
            120.0
        }
    }

    pub fn set_target_tickrate(&self, value: f32) {
        let clamped = if value.is_finite() {
            value.max(1.0)
        } else {
            120.0
        };
        self.target_tickrate_bits
            .store(clamped.to_bits(), Ordering::Relaxed);
    }

    pub fn target_fps(&self) -> Option<f32> {
        let value = f32::from_bits(self.target_fps_bits.load(Ordering::Relaxed));
        if value.is_finite() && value > 0.0 {
            Some(value)
        } else {
            None
        }
    }

    pub fn set_target_fps(&self, value: Option<f32>) {
        let next = match value {
            Some(value) if value.is_finite() && value > 0.0 => value,
            _ => 0.0,
        };
        self.target_fps_bits
            .store(next.to_bits(), Ordering::Relaxed);
    }

    pub fn fullscreen(&self) -> bool {
        self.fullscreen.load(Ordering::Relaxed)
    }

    pub fn set_fullscreen(&self, value: bool) {
        self.fullscreen.store(value, Ordering::Relaxed);
    }

    pub fn resizable(&self) -> bool {
        self.resizable.load(Ordering::Relaxed)
    }

    pub fn set_resizable(&self, value: bool) {
        self.resizable.store(value, Ordering::Relaxed);
    }

    pub fn decorations(&self) -> bool {
        self.decorations.load(Ordering::Relaxed)
    }

    pub fn set_decorations(&self, value: bool) {
        self.decorations.store(value, Ordering::Relaxed);
    }

    pub fn maximized(&self) -> bool {
        self.maximized.load(Ordering::Relaxed)
    }

    pub fn set_maximized(&self, value: bool) {
        self.maximized.store(value, Ordering::Relaxed);
    }

    pub fn minimized(&self) -> bool {
        self.minimized.load(Ordering::Relaxed)
    }

    pub fn set_minimized(&self, value: bool) {
        self.minimized.store(value, Ordering::Relaxed);
    }

    pub fn visible(&self) -> bool {
        self.visible.load(Ordering::Relaxed)
    }

    pub fn set_visible(&self, value: bool) {
        self.visible.store(value, Ordering::Relaxed);
    }
}
