use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;
use web_time::Instant;

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
