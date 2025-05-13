use std::any::{Any, TypeId};
use rayon::prelude::*;
use hashbrown::HashMap;

use super::{component::Component, ecs_core::Entity};

pub struct ComponentPool {
    components: HashMap<TypeId, HashMap<Entity, Box<dyn Component>>>,
}

impl ComponentPool {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
        }
    }

    pub fn insert<T: Component + 'static>(&mut self, entity: Entity, component: T) {
        let type_id = TypeId::of::<T>();
        let entry = self.components
            .entry(type_id)
            .or_insert_with(HashMap::new);
        entry.insert(entity, Box::new(component));
    }

    pub fn get<T: Component + 'static>(&self, entity: Entity) -> Option<&T> {
        self.components.get(&TypeId::of::<T>())
            .and_then(|map| map.get(&entity))
            .and_then(|boxed| boxed.as_any().downcast_ref::<T>())
    }

    pub fn get_mut<T: Component + 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        self.components.get_mut(&TypeId::of::<T>())
            .and_then(|map| map.get_mut(&entity))
            .and_then(|boxed| boxed.as_any_mut().downcast_mut::<T>())
    }

    pub fn get_all_for_entity(&self, entity: Entity) -> Vec<&dyn Component> {
        self.components
            .par_values()
            .filter_map(|map| map.get(&entity).map(|boxed| boxed.as_ref() as &dyn Component))
            .collect()
    }
    
    pub fn get_all_for_entity_mut(&mut self, entity: Entity) -> Vec<&mut dyn Component> {
        self.components
            .par_values_mut()
            .filter_map(|map| map.get_mut(&entity).map(|boxed| boxed.as_mut() as &mut dyn Component))
            .collect()
    }

    pub fn remove<T: Component + 'static>(&mut self, entity: Entity) {
        if let Some(map) = self.components.get_mut(&TypeId::of::<T>()) {
            map.remove(&entity);
        }
    }

    pub fn get_all<T: Component + 'static>(&self) -> Vec<&T> {
        self.components.get(&TypeId::of::<T>())
            .map(|map| {
                map.par_values()
                    .filter_map(|boxed| boxed.as_any().downcast_ref::<T>())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_all_mut<T: Component + 'static>(&mut self) -> Vec<&mut T> {
        self.components.get_mut(&TypeId::of::<T>())
            .map(|map| {
                map.par_values_mut()
                    .filter_map(|boxed| boxed.as_any_mut().downcast_mut::<T>())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_all_with_entities<T: Component + 'static>(&self) -> Vec<(Entity, &T)> {
        self.components.get(&TypeId::of::<T>())
            .map(|map| {
                map.par_iter()
                    .filter_map(|(entity, boxed)| {
                        boxed.as_any().downcast_ref::<T>()
                            .map(|component| (*entity, component))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_all_with_entities_mut<T: Component + 'static>(&mut self) -> Vec<(Entity, &mut T)> {
        self.components.get_mut(&TypeId::of::<T>())
            .map(|map| {
                map.par_iter_mut()
                    .filter_map(|(entity, boxed)| {
                        boxed.as_any_mut().downcast_mut::<T>()
                            .map(|component| (*entity, component))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn remove_entity(&mut self, entity: Entity) {
        self.components.par_values_mut().for_each(|map| {
            map.remove(&entity);
        })
    }
}