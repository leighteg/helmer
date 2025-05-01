use std::collections::HashMap;

use super::{component::Component, component_pool::ComponentPool};

pub type Entity = usize;

pub struct ECSCore {
    entities: HashMap<Entity, Vec<usize>>,
    next_entity_id: Entity,
    component_pool: ComponentPool,
}

impl ECSCore {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            next_entity_id: 0,
            component_pool: ComponentPool::new(),
        }
    }

    pub fn create_entity(&mut self) -> Entity {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        self.entities.insert(id, Vec::new());
        id
    }

    pub fn destroy_entity(&mut self, entity: Entity) {
        if let Some(components) = self.entities.remove(&entity) {
            for id in components {
                self.component_pool.remove(id);
            }
        }
    }

    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        if let Some(components) = self.entities.get_mut(&entity) {
            let id = self.component_pool.insert(component);
            components.push(id);
        }
    }

    pub fn get_components(&self, entity: Entity) -> Vec<&dyn Component> {
        self.entities.get(&entity)
            .map(|ids| {
                ids.iter()
                    .filter_map(|&id| self.component_pool.get(id).map(|c| c.as_ref()))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_components_of_type<T: Component>(&self, entity: Entity) -> Vec<&T> {
        self.entities.get(&entity)
            .map(|ids| {
                ids.iter()
                    .filter_map(|&id| {
                        self.component_pool
                            .get(id)
                            .and_then(|comp| comp.as_any().downcast_ref::<T>())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_components_by_type<T: Component>(&self) -> Vec<&T> {
        self.component_pool
            .iter()
            .filter_map(|comp| comp.as_any().downcast_ref::<T>())
            .collect()
    }

    pub fn get_components_by_type_mut<T: Component>(&mut self) -> Vec<&mut T> {
        self.component_pool
            .iter_mut()
            .filter_map(|comp| comp.as_any_mut().downcast_mut::<T>())
            .collect()
    }
}