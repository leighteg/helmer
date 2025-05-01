use std::collections::HashMap;

use super::{ecs_core::ECSCore, system::System};

pub struct SystemEntry {
    pub name: String,
    pub enabled: bool,
    pub priority: i32,
    pub system: Box<dyn System>,
}

pub struct SystemManager {
    systems: Vec<SystemEntry>,
    name_to_index: HashMap<String, usize>,
}

impl SystemManager {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            name_to_index: HashMap::new(),
        }
    }

    pub fn register_with_priority<S: System + 'static>(&mut self, system: S, priority: i32) {
        let name = system.name().to_string();
        let entry = SystemEntry {
            name: name.clone(),
            enabled: true,
            priority,
            system: Box::new(system),
        };
        self.name_to_index.insert(name, self.systems.len());
        self.systems.push(entry);
        self.sort_systems_by_priority();
    }

    fn sort_systems_by_priority(&mut self) {
        self.systems.sort_by_key(|entry| entry.priority);
        // Rebuild index after sort
        self.name_to_index = self.systems
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.clone(), i))
            .collect();
    }

    pub fn register<S: System + 'static>(&mut self, system: S) {
        self.register_with_priority(system, 0);
    }

    pub fn run_all(&mut self, ecs: &mut ECSCore) {
        for entry in &mut self.systems {
            if entry.enabled {
                entry.system.run(ecs);
            }
        }
    }

    pub fn run_named(&mut self, ecs: &mut ECSCore, name: &str) {
        if let Some(&idx) = self.name_to_index.get(name) {
            let entry = &mut self.systems[idx];
            if entry.enabled {
                entry.system.run(ecs);
            }
        }
    }

    pub fn enable(&mut self, name: &str) {
        if let Some(&idx) = self.name_to_index.get(name) {
            self.systems[idx].enabled = true;
        }
    }

    pub fn disable(&mut self, name: &str) {
        if let Some(&idx) = self.name_to_index.get(name) {
            self.systems[idx].enabled = false;
        }
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        self.name_to_index
            .get(name)
            .and_then(|&idx| Some(self.systems[idx].enabled))
            .unwrap_or(false)
    }

    pub fn set_priority(&mut self, name: &str, new_priority: i32) {
        if let Some(&idx) = self.name_to_index.get(name) {
            self.systems[idx].priority = new_priority;
            self.sort_systems_by_priority();
        }
    }
}