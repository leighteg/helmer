use std::time::Instant;

pub struct Resource<T> {
    inner: T,

    first_used: Instant,
    last_used: Instant,
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
}
