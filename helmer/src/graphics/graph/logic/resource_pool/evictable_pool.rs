use generational_arena::Arena;

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
