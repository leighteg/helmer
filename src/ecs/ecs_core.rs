use std::collections::HashSet;

use super::{component::Component, component_pool::ComponentPool, system_manager::SystemManager};

pub type Entity = usize;

pub struct ECSCore {
    next_entity_id: Entity,
    entities: HashSet<Entity>,
    component_pool: ComponentPool,
    pub system_manager: SystemManager,
}

impl ECSCore {
    pub fn new() -> Self {
        Self {
            next_entity_id: 0,
            entities: HashSet::new(),
            component_pool: ComponentPool::new(),
            system_manager: SystemManager::new(),
        }
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
}