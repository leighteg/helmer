use crate::graphics::{
    backend::binding_backend::BindingBackendChoice,
    common::{config::RenderConfig, renderer::WgpuBackend},
};
use std::env;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

fn parse_bool_env(value: &str) -> Option<bool> {
    let normalized = value.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "1" | "true" | "yes" | "on") {
        return Some(true);
    }
    if matches!(normalized.as_str(), "0" | "false" | "no" | "off") {
        return Some(false);
    }
    None
}

fn parse_positive_f32_env(key: &str) -> Option<f32> {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn runtime_single_thread_override() -> bool {
    for key in [
        "HELMER_RUNTIME_SINGLE_THREADED",
        "HELMER_FORCE_SINGLE_THREADED",
    ] {
        if let Ok(value) = env::var(key)
            && let Some(parsed) = parse_bool_env(&value)
        {
            return parsed;
        }
    }
    false
}

fn device_parallelism() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .max(1)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub egui: bool,
    pub wgpu_experimental_features: bool,
    pub wgpu_backend: WgpuBackend,
    pub binding_backend: BindingBackendChoice,
    pub render_config: RenderConfig,
    pub fixed_timestep: bool,
    pub target_tickrate: f32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let wgpu_experimental_features = env::var("HELMER_WGPU_EXPERIMENTAL")
            .ok()
            .map(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let wgpu_backend = match env::var("HELMER_BACKEND").ok() {
            Some(value) => WgpuBackend::from_env(Some(value)),
            None if cfg!(target_arch = "wasm32") => WgpuBackend::Gl,
            None => WgpuBackend::Auto,
        };
        let binding_backend =
            BindingBackendChoice::from_env(env::var("HELMER_BINDING_BACKEND").ok());
        let fixed_timestep = env::var("HELMER_FIXED_TIMESTEP")
            .ok()
            .map(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(cfg!(target_arch = "wasm32"));
        let target_tickrate = 120.0;
        Self {
            egui: true,
            wgpu_experimental_features,
            wgpu_backend,
            binding_backend,
            render_config: RenderConfig::default(),
            fixed_timestep,
            target_tickrate,
        }
    }
}

#[derive(Debug)]
pub struct RuntimeTuning {
    pub task_worker_count: AtomicUsize,
    pub render_message_capacity: AtomicUsize,
    pub asset_message_capacity: AtomicUsize,
    pub asset_stream_queue_capacity: AtomicUsize,
    pub asset_worker_queue_capacity: AtomicUsize,
    pub max_pending_asset_uploads: AtomicUsize,
    pub max_pending_asset_bytes: AtomicUsize,
    pub asset_uploads_per_frame: AtomicUsize,
    pub wgpu_poll_interval_frames: AtomicU32,
    /// 0 = off, 1 = poll, 2 = wait
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
        let parallelism = device_parallelism().max(1);
        let single_threaded_runtime =
            is_wasm || parallelism <= 1 || runtime_single_thread_override();
        let task_worker_count = if single_threaded_runtime {
            0
        } else {
            parallelism.saturating_sub(1).max(1)
        };
        let render_message_capacity = if single_threaded_runtime { 8 } else { 16 };
        let asset_message_capacity = if single_threaded_runtime {
            64
        } else {
            (parallelism * 64).clamp(64, 1024)
        };
        let worker_queue_capacity = if single_threaded_runtime {
            32
        } else {
            (parallelism * 48).clamp(48, 768)
        };
        let max_pending_asset_uploads = if single_threaded_runtime {
            24
        } else {
            (parallelism * 24).clamp(24, 512)
        };
        let max_pending_asset_bytes = if single_threaded_runtime {
            256 * 1024 * 1024
        } else {
            (parallelism * 128 * 1024 * 1024).clamp(256 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
        };
        let asset_uploads_per_frame = if is_wasm {
            2
        } else if single_threaded_runtime {
            2
        } else {
            parallelism.clamp(2, 32)
        };
        Self {
            task_worker_count: AtomicUsize::new(task_worker_count),
            render_message_capacity: AtomicUsize::new(render_message_capacity),
            asset_message_capacity: AtomicUsize::new(asset_message_capacity),
            asset_stream_queue_capacity: AtomicUsize::new(asset_message_capacity),
            asset_worker_queue_capacity: AtomicUsize::new(worker_queue_capacity),
            max_pending_asset_uploads: AtomicUsize::new(max_pending_asset_uploads),
            max_pending_asset_bytes: AtomicUsize::new(max_pending_asset_bytes),
            asset_uploads_per_frame: AtomicUsize::new(asset_uploads_per_frame),
            wgpu_poll_interval_frames: AtomicU32::new(1),
            wgpu_poll_mode: AtomicU32::new(1),
            pixels_per_line: AtomicU32::new(38),
            title_update_ms: AtomicU32::new(200),
            resize_debounce_ms: AtomicU32::new(500),
            max_logic_steps_per_frame: AtomicU32::new(4),
            target_tickrate_bits: AtomicU32::new(120.0f32.to_bits()),
            target_fps_bits: AtomicU32::new(
                parse_positive_f32_env("HELMER_TARGET_FPS")
                    .unwrap_or(0.0)
                    .to_bits(),
            ),
            pending_asset_uploads: AtomicUsize::new(0),
            pending_asset_bytes: AtomicUsize::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::RuntimeTuning;

    #[test]
    fn render_message_queue_stays_latency_bounded_relative_to_asset_queue() {
        let tuning = RuntimeTuning::default();
        let render_capacity = tuning.render_message_capacity.load(Ordering::Relaxed);
        let asset_capacity = tuning.asset_message_capacity.load(Ordering::Relaxed);

        assert!(render_capacity <= 16);
        assert!(asset_capacity >= 64);
        assert!(asset_capacity > render_capacity);
    }
}

impl RuntimeTuning {
    pub fn load_task_worker_count(&self) -> usize {
        self.task_worker_count.load(Ordering::Relaxed)
    }

    pub fn store_task_worker_count(&self, worker_count: usize) {
        let value = if cfg!(target_arch = "wasm32") {
            0
        } else {
            let max_workers = device_parallelism().saturating_sub(1).max(1);
            worker_count.min(max_workers)
        };
        self.task_worker_count.store(value, Ordering::Relaxed);
    }

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
