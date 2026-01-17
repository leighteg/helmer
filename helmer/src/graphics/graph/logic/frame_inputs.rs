#![allow(dead_code)]

use hashbrown::HashMap;
use std::{
    any::{Any, TypeId},
    sync::Arc,
};

/// Type-erased per-frame data payloads that passes can downcast.
/// This lets the logic thread push arbitrary types (RenderData or otherwise)
/// and have render passes fetch only what they care about.
#[derive(Default)]
pub struct FrameInputHub {
    inputs: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl FrameInputHub {
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
        }
    }

    /// Insert or replace a value for the given type.
    pub fn set<T: Any + Send + Sync>(&mut self, value: T) {
        self.inputs.insert(
            TypeId::of::<T>(),
            Arc::new(value) as Arc<dyn Any + Send + Sync>,
        );
    }

    /// Insert an Arc without cloning the inner value.
    pub fn set_arc<T: Any + Send + Sync>(&mut self, value: Arc<T>) {
        self.inputs
            .insert(TypeId::of::<T>(), value as Arc<dyn Any + Send + Sync>);
    }

    /// Fetch a cloned Arc for the requested type.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        self.inputs
            .get(&TypeId::of::<T>())
            .and_then(|v| v.clone().downcast::<T>().ok())
    }

    pub fn contains<T: Any + Send + Sync>(&self) -> bool {
        self.inputs.contains_key(&TypeId::of::<T>())
    }

    pub fn remove<T: Any + Send + Sync>(&mut self) {
        self.inputs.remove(&TypeId::of::<T>());
    }
}
