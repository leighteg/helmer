use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use rayon::prelude::*;

use super::{ecs_core::ECSCore, system::System};

pub struct SystemEntry {
    pub name: String,
    pub enabled: bool, 
    pub priority: i32,
    pub system: Box<dyn System + Send + Sync>,
    pub depends_on: Vec<String>,
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

    pub fn register_with_priority<S: System + 'static + Send + Sync>(
        &mut self,
        system: S,
        priority: i32,
        depends_on: Vec<String>,
    ) {
        let name = system.name().to_string();
        let entry = SystemEntry {
            name: name.clone(),
            enabled: true,
            priority,
            system: Box::new(system),
            depends_on,
        };
        self.name_to_index.insert(name, self.systems.len());
        self.systems.push(entry);
    }

    fn build_dependency_graph(&self) -> HashMap<String, Vec<String>> {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        for system in &self.systems {
            graph.entry(system.name.clone()).or_default();
            for dep in &system.depends_on {
                graph.entry(dep.clone()).or_default().push(system.name.clone());
            }
        }

        graph
    }

    fn topological_groups(&self) -> Vec<Vec<String>> {
        let graph = self.build_dependency_graph();
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        for (node, neighbors) in &graph {
            in_degree.entry(node.clone()).or_insert(0);
            for neighbor in neighbors {
                *in_degree.entry(neighbor.clone()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter_map(|(name, &deg)| if deg == 0 { Some(name.clone()) } else { None })
            .collect();

        let mut result = vec![];

        while !queue.is_empty() {
            let mut group = vec![];

            for _ in 0..queue.len() {
                if let Some(node) = queue.pop_front() {
                    group.push(node.clone());

                    if let Some(neighbors) = graph.get(&node) {
                        for neighbor in neighbors {
                            if let Some(deg) = in_degree.get_mut(neighbor) {
                                *deg -= 1;
                                if *deg == 0 {
                                    queue.push_back(neighbor.clone());
                                }
                            }
                        }
                    }
                }
            }

            result.push(group);
        }

        result
    }

    pub fn run_all(&mut self, ecs_core: &Arc<RwLock<ECSCore>>) {
        let groups = self.topological_groups();
        
        // Extract all the systems we need to run into a separate Vec
        // along with their enabled status
        let system_data: Vec<(String, bool)> = self.systems
            .iter()
            .map(|entry| (entry.name.clone(), entry.enabled))
            .collect();

        // Process each group in sequence (while allowing systems within a group to run in parallel)
        for group in groups {
            // Create thread-safe wrappers for each system
            let systems_to_run: Vec<(usize, bool)> = group
                .iter()
                .filter_map(|name| {
                    self.name_to_index.get(name).map(|&idx| (idx, system_data[idx].1))
                })
                .collect();
            
            // Run systems in parallel if they're enabled
            systems_to_run.par_iter().for_each(|&(idx, enabled)| {
                if enabled {
                    // Clone the Arc to share the ECSCore
                    let ecs = Arc::clone(ecs_core);
                    
                    // Access system in a thread-safe way
                    {
                        // Lock the ECSCore to modify it
                        if let Ok(mut ecs_lock) = ecs.write() {
                            // Run the system
                            self.systems[idx].system.run(&mut ecs_lock);
                        }
                    }
                }
            });
        }
    }
    
    // Helper method to enable/disable systems by name
    pub fn set_system_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(&idx) = self.name_to_index.get(name) {
            self.systems[idx].enabled = enabled;
            true
        } else {
            false
        }
    }
}