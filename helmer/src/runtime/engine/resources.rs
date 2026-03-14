use hashbrown::HashMap;
use std::any::{Any, TypeId};
use std::sync::{Arc, RwLock};

#[derive(Default)]
pub struct RuntimeResources {
    inner: RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

impl RuntimeResources {
    pub fn insert<T>(&self, value: T)
    where
        T: Send + Sync + 'static,
    {
        if let Ok(mut guard) = self.inner.write() {
            guard.insert(TypeId::of::<T>(), Arc::new(value));
        }
    }

    pub fn get<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        let guard = self.inner.read().ok()?;
        let value = guard.get(&TypeId::of::<T>())?;
        value.clone().downcast::<T>().ok()
    }

    pub fn remove<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        let mut guard = self.inner.write().ok()?;
        let value = guard.remove(&TypeId::of::<T>())?;
        value.downcast::<T>().ok()
    }
}
