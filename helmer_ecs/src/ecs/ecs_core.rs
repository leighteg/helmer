use std::{any::TypeId, collections::HashSet};

use tracing::info;

use crate::ecs::{resource::Resource, resource_pool::ResourcePool};

use super::{
    component::Component, component_pool::ComponentPool,
};

pub type Entity = usize;

pub struct ECSCore {
    next_entity_id: Entity,
    entities: HashSet<Entity>,
    pub component_pool: ComponentPool,
    pub resource_pool: ResourcePool,
}

impl ECSCore {
    pub fn new() -> Self {
        let instance = Self {
            next_entity_id: 0,
            entities: HashSet::new(),
            component_pool: ComponentPool::new(),
            resource_pool: ResourcePool::new(),
        };

        info!("initialized ECSCore");

        return instance;
    }

    pub fn create_entity(&mut self) -> Entity {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        self.entities.insert(id);
        id
    }

    pub fn destroy_entity(&mut self, entity: Entity) {
        self.entities.remove(&entity);
        self.component_pool.remove_entity(entity);
    }

    pub fn entity_exists(&self, entity: Entity) -> bool {
        self.entities.contains(&entity)
    }

    pub fn get_all_entities(&self) -> Vec<Entity> {
        self.entities.clone().into_iter().collect()
    }

    pub fn add_component<T: Component + 'static>(&mut self, entity: Entity, component: T) {
        self.component_pool.insert(entity, component);
    }

    pub fn get_component<T: Component + 'static>(&self, entity: Entity) -> Option<&T> {
        self.component_pool.get::<T>(entity)
    }

    pub fn get_component_mut<T: Component + 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        self.component_pool.get_mut::<T>(entity)
    }

    pub fn get_components(&self, entity: Entity) -> Vec<&dyn Component> {
        self.component_pool.get_all_for_entity(entity)
    }

    pub fn get_components_mut(&mut self, entity: Entity) -> Vec<&mut dyn Component> {
        self.component_pool.get_all_for_entity_mut(entity)
    }

    pub fn get_all_components_of_type<T: Component + 'static>(&self) -> Vec<&T> {
        self.component_pool.get_all::<T>()
    }

    pub fn get_all_components_of_type_mut<T: Component + 'static>(&mut self) -> Vec<&mut T> {
        self.component_pool.get_all_mut::<T>()
    }

    pub fn get_all_entities_with_component<T: Component + 'static>(&self) -> Vec<(Entity, &T)> {
        self.component_pool.get_all_with_entities::<T>()
    }

    pub fn add_resource<T: Resource>(&mut self, resource: T) {
        self.resource_pool.add(resource);
    }

    pub fn get_resource<T: Resource>(&self) -> Option<&T> {
        self.resource_pool.get::<T>()
    }

    pub fn get_resource_mut<T: Resource>(&mut self) -> Option<&mut T> {
        self.resource_pool.get_mut::<T>()
    }

    pub fn resource_scope<T: Resource, R>(
        &mut self,
        f: impl FnOnce(&mut Self, &mut T) -> R,
    ) -> Option<R> {
        let type_id = TypeId::of::<T>();

        // temporarily take a raw pointer to the resource
        let resource_ptr = self
            .resource_pool
            .resources
            .get_mut(&type_id)?
            .downcast_mut::<T>()? as *mut T;

        // SAFETY:
        // - `resource_ptr` points to a unique entry inside `self.resources`.
        // - we won't remove/replace that entry while the pointer is used.
        // - we ensure `T` is only borrowed once at a time by user convention.
        unsafe { Some(f(self, &mut *resource_ptr)) }
    }

    pub fn remove_resource<T: Resource>(&mut self) {
        self.resource_pool.remove::<T>();
    }
}
