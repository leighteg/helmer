use std::any::{Any, TypeId};
use hashbrown::HashMap;
use super::resource::Resource;

pub struct ResourcePool {
    resources: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl ResourcePool {
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
        }
    }
    
    pub fn add<T: Resource>(&mut self, resource: T) {
        let type_id = TypeId::of::<T>();
        self.resources.insert(type_id, Box::new(resource));
    }

    pub fn get<T: Resource>(&self) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.resources.get(&type_id)
            .and_then(|boxed_resource| boxed_resource.downcast_ref::<T>())
    }

    pub fn get_mut<T: Resource>(&mut self) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.resources.get_mut(&type_id)
            .and_then(|boxed_resource| boxed_resource.downcast_mut::<T>())
    }
    
    pub fn remove<T: Resource>(&mut self) {
        let type_id = TypeId::of::<T>();
        self.resources.remove(&type_id);
    }
}