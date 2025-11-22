use generational_arena::{Arena, Index};

use crate::graphics::graph::logic::resource_pool::resource::Resource;

pub struct EvictablePool<T> {
    pub arena: Arena<Resource<T>>,
    pub wheel_index: Vec<Option<usize>>, // wheel bucket for each index
}

impl<T> EvictablePool<T> {
    pub fn new() -> Self {
        Self {
            arena: Arena::new(),
            wheel_index: Vec::new(),
        }
    }
}

impl<T> EvictablePool<T> {
    pub fn insert(&mut self, resource: Resource<T>) -> Index {
        let idx = self.arena.insert(resource);
        let raw = idx.into_raw_parts().0;

        // ensure vector is large enough
        if raw >= self.wheel_index.len() {
            self.wheel_index.resize(raw + 1, None);
        }

        idx
    }

    pub fn remove(&mut self, idx: Index) -> Option<Resource<T>> {
        let result = self.arena.remove(idx);
        if result.is_some() {
            let raw = idx.into_raw_parts().0;
            if raw < self.wheel_index.len() {
                self.wheel_index[raw] = None;
            }
        }
        result
    }
}
