use std::collections::HashMap;

use super::{ecs_core::ECSCore, system::System};

pub struct SystemManager {
    systems: Vec<Box<dyn System>>,
    name_to_index: HashMap<String, usize>,
}

impl SystemManager {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            name_to_index: HashMap::new(),
        }
    }

    pub fn register<S: System + 'static>(&mut self, system: S) {
        let name = system.name().to_string();
        self.name_to_index.insert(name.clone(), self.systems.len());
        self.systems.push(Box::new(system));
    }

    pub fn run_all(&mut self, ecs: &mut ECSCore) {
        for system in self.systems.iter_mut() {
            system.run(ecs);
        }
    }

    pub fn run_named(&mut self, ecs: &mut ECSCore, name: &str) {
        if let Some(&idx) = self.name_to_index.get(name) {
            self.systems[idx].run(ecs);
        }
    }

    pub fn disable(&mut self, name: &str) {
        if let Some(idx) = self.name_to_index.remove(name) {
            self.systems.remove(idx);
            // Rebuild index
            self.name_to_index = self.systems
                .iter()
                .enumerate()
                .map(|(i, sys)| (sys.name().to_string(), i))
                .collect();
        }
    }
}