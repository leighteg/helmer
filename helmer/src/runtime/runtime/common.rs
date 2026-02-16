use std::sync::{
    RwLock,
    atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering},
};
use std::time::Duration;
use web_time::Instant;

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
            // force one initial application when a window becomes available
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
        let title = title.into();
        if let Ok(mut current) = self.custom_title.write() {
            *current = title;
        }
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

#[derive(Default)]
pub struct PerformanceMetrics {
    pub fps: AtomicU32,
    pub tps: AtomicU32,
}

#[derive(Debug)]
pub struct RuntimeTuning {
    pub render_message_capacity: AtomicUsize,
    pub asset_stream_queue_capacity: AtomicUsize,
    pub asset_worker_queue_capacity: AtomicUsize,
    pub max_pending_asset_uploads: AtomicUsize,
    pub max_pending_asset_bytes: AtomicUsize,
    pub asset_uploads_per_frame: AtomicUsize,
    pub wgpu_poll_interval_frames: AtomicU32,
    /// 0 = off, 1 = Poll, 2 = Wait
    pub wgpu_poll_mode: AtomicU32,
    pub pixels_per_line: AtomicU32,
    pub title_update_ms: AtomicU32,
    pub resize_debounce_ms: AtomicU32,
    pub max_logic_steps_per_frame: AtomicU32,
    pub target_tickrate_bits: AtomicU32,
    pub target_fps_bits: AtomicU32,
    pub pending_asset_uploads: AtomicUsize,
    pub pending_asset_bytes: AtomicUsize,
}

impl Default for RuntimeTuning {
    fn default() -> Self {
        let is_wasm = cfg!(target_arch = "wasm32");
        let message_capacity = 96;
        Self {
            render_message_capacity: AtomicUsize::new(message_capacity),
            asset_stream_queue_capacity: AtomicUsize::new(message_capacity),
            asset_worker_queue_capacity: AtomicUsize::new(message_capacity),
            max_pending_asset_uploads: AtomicUsize::new(48),
            max_pending_asset_bytes: AtomicUsize::new(512 * 1024 * 1024),
            asset_uploads_per_frame: AtomicUsize::new(if is_wasm { 2 } else { 8 }),
            wgpu_poll_interval_frames: AtomicU32::new(1),
            wgpu_poll_mode: AtomicU32::new(1),
            pixels_per_line: AtomicU32::new(38),
            title_update_ms: AtomicU32::new(200),
            resize_debounce_ms: AtomicU32::new(500),
            max_logic_steps_per_frame: AtomicU32::new(4),
            target_tickrate_bits: AtomicU32::new(120.0f32.to_bits()),
            target_fps_bits: AtomicU32::new(0.0f32.to_bits()),
            pending_asset_uploads: AtomicUsize::new(0),
            pending_asset_bytes: AtomicUsize::new(0),
        }
    }
}

impl RuntimeTuning {
    pub fn load_target_tickrate(&self) -> f32 {
        let value = f32::from_bits(self.target_tickrate_bits.load(Ordering::Relaxed));
        if value.is_finite() && value >= 1.0 {
            value
        } else {
            120.0
        }
    }

    pub fn store_target_tickrate(&self, tickrate: f32) {
        let value = if tickrate.is_finite() {
            tickrate.max(1.0)
        } else {
            120.0
        };
        self.target_tickrate_bits
            .store(value.to_bits(), Ordering::Relaxed);
    }

    pub fn load_target_fps(&self) -> Option<f32> {
        let value = f32::from_bits(self.target_fps_bits.load(Ordering::Relaxed));
        if value.is_finite() && value > 0.0 {
            Some(value)
        } else {
            None
        }
    }

    pub fn store_target_fps(&self, fps: Option<f32>) {
        let value = match fps {
            Some(value) if value.is_finite() && value > 0.0 => value,
            _ => 0.0,
        };
        self.target_fps_bits
            .store(value.to_bits(), Ordering::Relaxed);
    }

    pub fn try_reserve_asset_upload(&self, bytes: usize) -> bool {
        let max_uploads = self.max_pending_asset_uploads.load(Ordering::Relaxed);
        let max_bytes = self.max_pending_asset_bytes.load(Ordering::Relaxed);

        let uploads = self.pending_asset_uploads.fetch_add(1, Ordering::Relaxed) + 1;
        let bytes_total = self.pending_asset_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;

        let over_uploads = if max_uploads == 0 {
            uploads > 0
        } else {
            uploads > max_uploads
        };
        let over_bytes = if max_bytes == 0 {
            bytes_total > 0
        } else {
            bytes_total > max_bytes
        };

        if over_uploads || over_bytes {
            self.pending_asset_uploads.fetch_sub(1, Ordering::Relaxed);
            self.pending_asset_bytes.fetch_sub(bytes, Ordering::Relaxed);
            return false;
        }

        true
    }

    pub fn release_asset_upload(&self, bytes: usize) {
        let _ =
            self.pending_asset_uploads
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(1))
                });
        let _ = self
            .pending_asset_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(bytes))
            });
    }
}

#[derive(Debug)]
pub struct RuntimeProfiling {
    pub enabled: AtomicBool,
    pub history_samples: AtomicU32,
    pub render_pass_plot_limit: AtomicU32,
    pub main_event_us: AtomicU64,
    pub main_update_us: AtomicU64,
    pub logic_frame_us: AtomicU64,
    pub logic_asset_us: AtomicU64,
    pub logic_input_us: AtomicU64,
    pub logic_tick_us: AtomicU64,
    pub logic_schedule_us: AtomicU64,
    pub logic_render_send_us: AtomicU64,
    pub ecs_render_data_us: AtomicU64,
    pub ecs_scene_spawn_us: AtomicU64,
    pub ecs_scene_update_us: AtomicU64,
    pub render_thread_frame_us: AtomicU64,
    pub render_thread_messages_us: AtomicU64,
    pub render_thread_upload_us: AtomicU64,
    pub render_thread_render_us: AtomicU64,
}

impl Default for RuntimeProfiling {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            history_samples: AtomicU32::new(240),
            render_pass_plot_limit: AtomicU32::new(6),
            main_event_us: AtomicU64::new(0),
            main_update_us: AtomicU64::new(0),
            logic_frame_us: AtomicU64::new(0),
            logic_asset_us: AtomicU64::new(0),
            logic_input_us: AtomicU64::new(0),
            logic_tick_us: AtomicU64::new(0),
            logic_schedule_us: AtomicU64::new(0),
            logic_render_send_us: AtomicU64::new(0),
            ecs_render_data_us: AtomicU64::new(0),
            ecs_scene_spawn_us: AtomicU64::new(0),
            ecs_scene_update_us: AtomicU64::new(0),
            render_thread_frame_us: AtomicU64::new(0),
            render_thread_messages_us: AtomicU64::new(0),
            render_thread_upload_us: AtomicU64::new(0),
            render_thread_render_us: AtomicU64::new(0),
        }
    }
}

pub struct LogicFrame {
    pub steps: u32,
    pub dt: f32,
}

pub struct LogicClock {
    last_time: Instant,
    accumulator: Duration,
    fixed_dt: Duration,
    deterministic: bool,
    target_tickrate: f32,
}

impl LogicClock {
    pub fn new(target_tickrate: f32, deterministic: bool) -> Self {
        let tickrate = target_tickrate.max(1.0);
        let fixed_dt = Duration::from_secs_f32(1.0 / tickrate);
        Self {
            last_time: Instant::now(),
            accumulator: Duration::ZERO,
            fixed_dt,
            deterministic,
            target_tickrate: tickrate,
        }
    }

    pub fn set_tickrate(&mut self, target_tickrate: f32, now: Instant) {
        let tickrate = target_tickrate.max(1.0);
        if (tickrate - self.target_tickrate).abs() <= f32::EPSILON {
            return;
        }
        self.target_tickrate = tickrate;
        self.fixed_dt = Duration::from_secs_f32(1.0 / tickrate);
        self.reset(now);
    }

    pub fn advance(&mut self, now: Instant, max_steps_per_frame: u32) -> LogicFrame {
        let elapsed = now.saturating_duration_since(self.last_time);
        self.last_time = now;

        if !self.deterministic {
            return LogicFrame {
                steps: 1,
                dt: elapsed.as_secs_f32(),
            };
        }

        let fixed_secs = self.fixed_dt.as_secs_f64();
        let max_steps = max_steps_per_frame.max(1);
        let max_accum = Duration::from_secs_f64(fixed_secs * max_steps as f64);

        self.accumulator = self.accumulator.saturating_add(elapsed);
        if self.accumulator > max_accum {
            self.accumulator = max_accum;
        }

        let mut steps = (self.accumulator.as_secs_f64() / fixed_secs).floor() as u32;
        if steps > max_steps {
            steps = max_steps;
        }

        if steps > 0 {
            let consumed = Duration::from_secs_f64(fixed_secs * steps as f64);
            self.accumulator = self.accumulator.saturating_sub(consumed);
        }

        LogicFrame {
            steps,
            dt: self.fixed_dt.as_secs_f32(),
        }
    }

    pub fn reset(&mut self, now: Instant) {
        self.last_time = now;
        self.accumulator = Duration::ZERO;
    }
}
