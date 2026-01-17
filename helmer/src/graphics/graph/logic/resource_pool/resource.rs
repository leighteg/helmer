use std::time::{Duration, Instant};

pub struct Resource<T> {
    pub inner: T,

    pub first_used: Instant,
    pub last_used: Instant,
}

impl<T> Resource<T> {
    pub fn new(inner: T) -> Self {
        let now = Instant::now();
        Self {
            inner,

            first_used: now,
            last_used: now,
        }
    }

    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.first_used)
    }

    pub fn idle_time(&self) -> Duration {
        Instant::now().duration_since(self.last_used)
    }
}
