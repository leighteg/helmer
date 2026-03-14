use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::any::TypeId;

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
        let entry = self.components.entry(type_id).or_insert_with(HashMap::new);
        entry.insert(entity, Box::new(component));
    }

    pub fn get<T: Component + 'static>(&self, entity: Entity) -> Option<&T> {
        self.components
            .get(&TypeId::of::<T>())
            .and_then(|map| map.get(&entity))
            .and_then(|boxed| boxed.as_any().downcast_ref::<T>())
    }

    pub fn get_mut<T: Component + 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        self.components
            .get_mut(&TypeId::of::<T>())
            .and_then(|map| map.get_mut(&entity))
            .and_then(|boxed| boxed.as_any_mut().downcast_mut::<T>())
    }

    /// Queries for two mutable components for a single entity using `get_many_mut`.
    ///
    /// Returns `Some((&mut T1, &mut T2))` if the entity has both components, otherwise `None`.
    pub fn query_mut<'a, T1, T2>(&'a mut self, entity: Entity) -> Option<(&'a mut T1, &'a mut T2)>
    where
        T1: Component + 'static,
        T2: Component + 'static,
    {
        let type_id1 = TypeId::of::<T1>();
        let type_id2 = TypeId::of::<T2>();

        // Ensure we're not trying to mutably borrow the same component type twice.
        if type_id1 == type_id2 {
            return None;
        }

        let [map1, map2] = self.components.get_many_mut([&type_id1, &type_id2]);

        let comp1 = map1
            .unwrap()
            .get_mut(&entity)?
            .as_any_mut()
            .downcast_mut::<T1>()?;
        let comp2 = map2
            .unwrap()
            .get_mut(&entity)?
            .as_any_mut()
            .downcast_mut::<T2>()?;

        Some((comp1, comp2))
    }

    /// Queries for three mutable components for a single entity using `get_many_mut`.
    ///
    /// Returns `Some((&mut T1, &mut T2, &mut T3))` if the entity has all components, otherwise `None`.
    pub fn query_mut_three<'a, T1, T2, T3>(
        &'a mut self,
        entity: Entity,
    ) -> Option<(&'a mut T1, &'a mut T2, &'a mut T3)>
    where
        T1: Component + 'static,
        T2: Component + 'static,
        T3: Component + 'static,
    {
        let type_id1 = TypeId::of::<T1>();
        let type_id2 = TypeId::of::<T2>();
        let type_id3 = TypeId::of::<T3>();

        // Ensure distinct types
        if type_id1 == type_id2 || type_id1 == type_id3 || type_id2 == type_id3 {
            return None;
        }

        let [map1, map2, map3] = self
            .components
            .get_many_mut([&type_id1, &type_id2, &type_id3]);

        let comp1 = map1
            .unwrap()
            .get_mut(&entity)?
            .as_any_mut()
            .downcast_mut::<T1>()?;
        let comp2 = map2
            .unwrap()
            .get_mut(&entity)?
            .as_any_mut()
            .downcast_mut::<T2>()?;
        let comp3 = map3
            .unwrap()
            .get_mut(&entity)?
            .as_any_mut()
            .downcast_mut::<T3>()?;

        Some((comp1, comp2, comp3))
    }

    pub fn get_all_for_entity(&self, entity: Entity) -> Vec<&dyn Component> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.components
                .par_values()
                .filter_map(|map| {
                    map.get(&entity)
                        .map(|boxed| boxed.as_ref() as &dyn Component)
                })
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.components
                .values()
                .filter_map(|map| {
                    map.get(&entity)
                        .map(|boxed| boxed.as_ref() as &dyn Component)
                })
                .collect()
        }
    }

    pub fn get_all_for_entity_mut(&mut self, entity: Entity) -> Vec<&mut dyn Component> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.components
                .par_values_mut()
                .filter_map(|map| {
                    map.get_mut(&entity)
                        .map(|boxed| boxed.as_mut() as &mut dyn Component)
                })
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.components
                .values_mut()
                .filter_map(|map| {
                    map.get_mut(&entity)
                        .map(|boxed| boxed.as_mut() as &mut dyn Component)
                })
                .collect()
        }
    }

    pub fn remove<T: Component + 'static>(&mut self, entity: Entity) {
        if let Some(map) = self.components.get_mut(&TypeId::of::<T>()) {
            map.remove(&entity);
        }
    }

    pub fn get_all<T: Component + 'static>(&self) -> Vec<&T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.components
                .get(&TypeId::of::<T>())
                .map(|map| {
                    map.par_values()
                        .filter_map(|boxed| boxed.as_any().downcast_ref::<T>())
                        .collect()
                })
                .unwrap_or_default()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.components
                .get(&TypeId::of::<T>())
                .map(|map| {
                    map.values()
                        .filter_map(|boxed| boxed.as_any().downcast_ref::<T>())
                        .collect()
                })
                .unwrap_or_default()
        }
    }

    pub fn get_all_mut<T: Component + 'static>(&mut self) -> Vec<&mut T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.components
                .get_mut(&TypeId::of::<T>())
                .map(|map| {
                    map.par_values_mut()
                        .filter_map(|boxed| boxed.as_any_mut().downcast_mut::<T>())
                        .collect()
                })
                .unwrap_or_default()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.components
                .get_mut(&TypeId::of::<T>())
                .map(|map| {
                    map.values_mut()
                        .filter_map(|boxed| boxed.as_any_mut().downcast_mut::<T>())
                        .collect()
                })
                .unwrap_or_default()
        }
    }

    pub fn get_all_with_entities<T: Component + 'static>(&self) -> Vec<(Entity, &T)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.components
                .get(&TypeId::of::<T>())
                .map(|map| {
                    map.par_iter()
                        .filter_map(|(entity, boxed)| {
                            boxed
                                .as_any()
                                .downcast_ref::<T>()
                                .map(|component| (*entity, component))
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.components
                .get(&TypeId::of::<T>())
                .map(|map| {
                    map.iter()
                        .filter_map(|(entity, boxed)| {
                            boxed
                                .as_any()
                                .downcast_ref::<T>()
                                .map(|component| (*entity, component))
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
    }

    pub fn get_all_with_entities_mut<T: Component + 'static>(&mut self) -> Vec<(Entity, &mut T)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.components
                .get_mut(&TypeId::of::<T>())
                .map(|map| {
                    map.par_iter_mut()
                        .filter_map(|(entity, boxed)| {
                            boxed
                                .as_any_mut()
                                .downcast_mut::<T>()
                                .map(|component| (*entity, component))
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.components
                .get_mut(&TypeId::of::<T>())
                .map(|map| {
                    map.iter_mut()
                        .filter_map(|(entity, boxed)| {
                            boxed
                                .as_any_mut()
                                .downcast_mut::<T>()
                                .map(|component| (*entity, component))
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
    }

    pub fn remove_entity(&mut self, entity: Entity) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.components.par_values_mut().for_each(|map| {
                map.remove(&entity);
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.components.values_mut().for_each(|map| {
                map.remove(&entity);
            });
        }
    }
}

// Helper trait for component queries
pub trait ComponentQuery {
    type Item<'a>;
    type ItemMut<'a>;

    fn type_ids() -> Vec<TypeId>;
    fn fetch<'a>(pool: &'a ComponentPool, entity: Entity) -> Option<Self::Item<'a>>;
    fn fetch_mut<'a>(pool: &'a mut ComponentPool, entity: Entity) -> Option<Self::ItemMut<'a>>;
}

// Implement ComponentQuery for single components
impl<T: Component + 'static> ComponentQuery for T {
    type Item<'a> = &'a T;
    type ItemMut<'a> = &'a mut T;

    fn type_ids() -> Vec<TypeId> {
        vec![TypeId::of::<T>()]
    }

    fn fetch<'a>(pool: &'a ComponentPool, entity: Entity) -> Option<Self::Item<'a>> {
        pool.get::<T>(entity)
    }

    fn fetch_mut<'a>(pool: &'a mut ComponentPool, entity: Entity) -> Option<Self::ItemMut<'a>> {
        pool.get_mut::<T>(entity)
    }
}

// Implement for 2-component tuples
impl<T1, T2> ComponentQuery for (T1, T2)
where
    T1: Component + 'static,
    T2: Component + 'static,
{
    type Item<'a> = (&'a T1, &'a T2);
    type ItemMut<'a> = (&'a mut T1, &'a mut T2);

    fn type_ids() -> Vec<TypeId> {
        vec![TypeId::of::<T1>(), TypeId::of::<T2>()]
    }

    fn fetch<'a>(pool: &'a ComponentPool, entity: Entity) -> Option<Self::Item<'a>> {
        Some((pool.get::<T1>(entity)?, pool.get::<T2>(entity)?))
    }

    fn fetch_mut<'a>(pool: &'a mut ComponentPool, entity: Entity) -> Option<Self::ItemMut<'a>> {
        if TypeId::of::<T1>() == TypeId::of::<T2>() {
            return None; // Can't mutably borrow same component type twice
        }
        pool.query_mut(entity)
    }
}

// Implement for 3-component tuples
impl<T1, T2, T3> ComponentQuery for (T1, T2, T3)
where
    T1: Component + 'static,
    T2: Component + 'static,
    T3: Component + 'static,
{
    type Item<'a> = (&'a T1, &'a T2, &'a T3);
    type ItemMut<'a> = (&'a mut T1, &'a mut T2, &'a mut T3);

    fn type_ids() -> Vec<TypeId> {
        vec![TypeId::of::<T1>(), TypeId::of::<T2>(), TypeId::of::<T3>()]
    }

    fn fetch<'a>(pool: &'a ComponentPool, entity: Entity) -> Option<Self::Item<'a>> {
        Some((
            pool.get::<T1>(entity)?,
            pool.get::<T2>(entity)?,
            pool.get::<T3>(entity)?,
        ))
    }

    fn fetch_mut<'a>(pool: &'a mut ComponentPool, entity: Entity) -> Option<Self::ItemMut<'a>> {
        let type_ids = vec![TypeId::of::<T1>(), TypeId::of::<T2>(), TypeId::of::<T3>()];
        let unique_types: HashSet<_> = type_ids.iter().collect();
        if unique_types.len() != type_ids.len() {
            return None; // Can't mutably borrow same component type twice
        }
        pool.query_mut_three(entity)
    }
}

// Generate more tuple implementations using a macro
macro_rules! impl_component_query_tuple {
    ($($t:ident : $idx:tt),+) => {
        impl<$($t: Component + 'static),+> ComponentQuery for ($($t,)+) {
            type Item<'a> = ($(&'a $t,)+);
            type ItemMut<'a> = ($(&'a mut $t,)+);

            fn type_ids() -> Vec<TypeId> {
                vec![$(TypeId::of::<$t>()),+]
            }

            fn fetch<'a>(pool: &'a ComponentPool, entity: Entity) -> Option<Self::Item<'a>> {
                Some(($(pool.get::<$t>(entity)?,)+))
            }

            fn fetch_mut<'a>(pool: &'a mut ComponentPool, entity: Entity) -> Option<Self::ItemMut<'a>> {
                // Check for duplicate component types
                let type_ids = Self::type_ids();
                let unique_types: HashSet<_> = type_ids.iter().collect();
                if unique_types.len() != type_ids.len() {
                    return None;
                }

                // For larger tuples, we'll need to implement a more general solution
                // For now, return None to indicate unsupported
                None
            }
        }
    };
}

// Generate implementations for 4-10 component tuples (mutable queries not fully implemented)
impl_component_query_tuple!(T1:0, T2:1, T3:2, T4:3);
impl_component_query_tuple!(T1:0, T2:1, T3:2, T4:3, T5:4);
impl_component_query_tuple!(T1:0, T2:1, T3:2, T4:3, T5:4, T6:5);
impl_component_query_tuple!(T1:0, T2:1, T3:2, T4:3, T5:4, T6:5, T7:6);
impl_component_query_tuple!(T1:0, T2:1, T3:2, T4:3, T5:4, T6:5, T7:6, T8:7);
impl_component_query_tuple!(T1:0, T2:1, T3:2, T4:3, T5:4, T6:5, T7:6, T8:7, T9:8);
impl_component_query_tuple!(T1:0, T2:1, T3:2, T4:3, T5:4, T6:5, T7:6, T8:7, T9:8, T10:9);

// Query iterator for immutable references
pub struct QueryIter<'a, Q: ComponentQuery> {
    pool: &'a ComponentPool,
    entities: Vec<Entity>,
    current_index: usize,
    _phantom: std::marker::PhantomData<Q>,
}

impl<'a, Q: ComponentQuery> Iterator for QueryIter<'a, Q> {
    type Item = Q::Item<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_index < self.entities.len() {
            let entity = self.entities[self.current_index];
            self.current_index += 1;

            if let Some(components) = Q::fetch(self.pool, entity) {
                return Some(components);
            }
        }
        None
    }
}

// For mutable queries, we'll use a different approach that collects results first
pub struct QueryMutResults<Q: ComponentQuery> {
    results: Vec<Q::ItemMut<'static>>,
    current_index: usize,
}

impl<Q: ComponentQuery> Iterator for QueryMutResults<Q> {
    type Item = Q::ItemMut<'static>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.results.len() {
            let result = unsafe {
                // SAFETY: We're transferring ownership of the result
                std::ptr::read(&self.results[self.current_index])
            };
            self.current_index += 1;
            Some(result)
        } else {
            None
        }
    }
}

impl ComponentPool {
    /// Query entities that have ALL specified components (exact match)
    pub fn query_exact<Q: ComponentQuery>(&self) -> QueryIter<Q> {
        let required_types = Q::type_ids();
        let entities = self.get_entities_with_exactly(&required_types);

        QueryIter {
            pool: self,
            entities,
            current_index: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Query entities that have AT LEAST the specified components (may have others)
    pub fn query<Q: ComponentQuery>(&self) -> QueryIter<Q> {
        let required_types = Q::type_ids();
        let entities = self.get_entities_with_all(&required_types);

        QueryIter {
            pool: self,
            entities,
            current_index: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Execute a closure for each entity matching the query
    pub fn query_for_each<Q: ComponentQuery, F>(&mut self, mut f: F)
    where
        F: FnMut(Entity, Q::Item<'_>),
    {
        let required_types = Q::type_ids();
        let entities = self.get_entities_with_all(&required_types);

        for entity in entities {
            if let Some(components) = Q::fetch(self, entity) {
                f(entity, components);
            }
        }
    }

    /// Execute a closure for each entity matching the query (mutable access)
    pub fn query_mut_for_each<Q: ComponentQuery, F>(&mut self, mut f: F)
    where
        F: FnMut(Entity, Q::ItemMut<'_>),
    {
        let required_types = Q::type_ids();
        let entities = self.get_entities_with_all(&required_types);

        for entity in entities {
            if let Some(components) = Q::fetch_mut(self, entity) {
                f(entity, components);
            }
        }
    }

    /// Execute a closure for each entity matching the exact query (mutable access)
    pub fn query_exact_for_each<Q: ComponentQuery, F>(&self, mut f: F)
    where
        F: FnMut(Entity, Q::Item<'_>),
    {
        let required_types = Q::type_ids();
        let entities = self.get_entities_with_exactly(&required_types);

        for entity in entities {
            if let Some(components) = Q::fetch(self, entity) {
                f(entity, components);
            }
        }
    }

    /// Execute a closure for each entity matching the exact query (mutable access)
    pub fn query_exact_mut_for_each<Q: ComponentQuery, F>(&mut self, mut f: F)
    where
        F: FnMut(Q::ItemMut<'_>),
    {
        let required_types = Q::type_ids();
        let entities = self.get_entities_with_exactly(&required_types);

        for entity in entities {
            if let Some(components) = Q::fetch_mut(self, entity) {
                f(components);
            }
        }
    }

    /// Get entities that have ALL of the specified component types (may have others)
    pub fn get_entities_with_all(&self, type_ids: &[TypeId]) -> Vec<Entity> {
        if type_ids.is_empty() {
            return Vec::new();
        }

        let mut result_entities: Option<HashSet<Entity>> = None;

        for &type_id in type_ids {
            if let Some(component_map) = self.components.get(&type_id) {
                let entities: HashSet<Entity> = component_map.keys().cloned().collect();

                result_entities = match result_entities {
                    None => Some(entities),
                    Some(existing) => Some(existing.intersection(&entities).cloned().collect()),
                };
            } else {
                return Vec::new(); // Component type doesn't exist
            }
        }

        result_entities.unwrap_or_default().into_iter().collect()
    }

    /// Get entities that have EXACTLY the specified component types (no more, no less)
    pub fn get_entities_with_exactly(&self, type_ids: &[TypeId]) -> Vec<Entity> {
        let required_set: HashSet<TypeId> = type_ids.iter().cloned().collect();
        let mut result = Vec::new();

        // Get all entities that have at least the required components
        let candidates = self.get_entities_with_all(type_ids);

        for entity in candidates {
            // Count how many component types this entity has
            let entity_types: HashSet<TypeId> = self
                .components
                .iter()
                .filter_map(|(type_id, map)| {
                    if map.contains_key(&entity) {
                        Some(*type_id)
                    } else {
                        None
                    }
                })
                .collect();

            // Only include if entity has exactly the required components
            if entity_types == required_set {
                result.push(entity);
            }
        }

        result
    }

    /// Get count of entities matching a query
    pub fn count<Q: ComponentQuery>(&self) -> usize {
        self.get_entities_with_all(&Q::type_ids()).len()
    }

    /// Get count of entities with exactly matching components
    pub fn count_exact<Q: ComponentQuery>(&self) -> usize {
        self.get_entities_with_exactly(&Q::type_ids()).len()
    }
}

// Convenience macros for easier querying
#[macro_export]
macro_rules! query {
    ($pool:expr, $($component:ty),+) => {
        $pool.query::<($($component,)+)>()
    };
}

#[macro_export]
macro_rules! query_exact {
    ($pool:expr, $($component:ty),+) => {
        $pool.query_exact::<($($component,)+)>()
    };
}
