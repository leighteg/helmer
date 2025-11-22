use generational_arena::Index;
use std::time::{Duration, Instant};

use crate::graphics::graph::logic::resource_pool::evictable_pool::EvictablePool;

pub struct TimerWheel {
    buckets: Vec<Vec<Index>>,
    resolution: Duration,
    timeout: Duration,
    current: usize,
}

impl TimerWheel {
    pub fn new(timeout: Duration, resolution: Duration) -> Self {
        let slots = (timeout.as_millis() / resolution.as_millis()) as usize;

        Self {
            buckets: vec![Vec::new(); slots],
            resolution,
            timeout,
            current: 0,
        }
    }

    pub fn schedule(&mut self, idx: Index, last_used: Instant, wheel_index: &mut Option<usize>) {
        // Remove from previous bucket if present
        if let Some(old) = wheel_index.take() {
            self.buckets[old].retain(|&i| i != idx);
        }

        let deadline = last_used + self.timeout;
        let delta_ms = deadline.elapsed().as_millis();

        let slot = ((delta_ms / self.resolution.as_millis() as u128) % self.buckets.len() as u128)
            as usize;

        self.buckets[slot].push(idx);
        *wheel_index = Some(slot);
    }

    pub fn tick<T>(&mut self, pool: &mut EvictablePool<T>, now: Instant) {
        let bucket = &mut self.buckets[self.current];

        for idx in bucket.drain(..) {
            if let Some(res) = pool.arena.get(idx) {
                if now.duration_since(res.last_used) >= self.timeout {
                    pool.arena.remove(idx);
                }
            }

            pool.wheel_index[idx.into_raw_parts().0] = None;
        }

        self.current = (self.current + 1) % self.buckets.len();
    }
}
