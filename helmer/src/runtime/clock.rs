use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};
use std::time::Duration;
use web_time::Instant;

#[derive(Default)]
pub struct PerformanceMetrics {
    pub fps: AtomicU32,
    pub tps: AtomicU32,
}

#[derive(Clone)]
pub struct RuntimePerformanceMetricsResource(pub Arc<PerformanceMetrics>);

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

    pub fn tickrate(&self) -> f32 {
        self.target_tickrate
    }
}

impl PerformanceMetrics {
    pub fn set_fps(&self, fps: u32) {
        self.fps.store(fps, Ordering::Relaxed);
    }

    pub fn set_tps(&self, tps: u32) {
        self.tps.store(tps, Ordering::Relaxed);
    }
}
