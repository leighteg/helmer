use rayon::prelude::*;
use tracing::{info, trace, error};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::{ecs_core::ECSCore, system::System};

// Extended system entry with scheduling metadata
pub struct SystemEntry {
    pub name: String,
    pub enabled: bool,
    pub priority: i32,
    pub system: Arc<Mutex<dyn System + Send + Sync>>,
    pub depends_on: Vec<String>,

    // Scheduling metadata
    pub component_reads: HashSet<std::any::TypeId>,
    pub component_writes: HashSet<std::any::TypeId>,
    pub avg_execution_time: Duration,
    pub last_execution_time: Option<Instant>,
}

pub struct SystemScheduler {
    systems: Vec<SystemEntry>,
    name_to_index: HashMap<String, usize>,
    execution_groups: Vec<Vec<usize>>, // Pre-computed execution groups
    dirty: bool,                       // Whether groups need recomputation
}

impl SystemScheduler {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            name_to_index: HashMap::new(),
            execution_groups: Vec::new(),
            dirty: true,
        }
    }

    pub fn register_system<S: System + 'static + Send + Sync>(
        &mut self,
        system: S,
        priority: i32,
        depends_on: Vec<String>,
        reads: HashSet<std::any::TypeId>,
        writes: HashSet<std::any::TypeId>,
    ) {
        let name = system.name().to_string();
        let entry = SystemEntry {
            name: name.clone(),
            enabled: true,
            priority,
            system: Arc::new(Mutex::new(system)),
            depends_on,
            component_reads: reads,
            component_writes: writes,
            avg_execution_time: Duration::from_micros(100), // Default estimate
            last_execution_time: None,
        };

        self.name_to_index.insert(name, self.systems.len());
        self.systems.push(entry);
        self.dirty = true;
    }

    // Check if two systems can run in parallel
    fn can_run_parallel(&self, a: usize, b: usize) -> bool {
        let sys_a = &self.systems[a];
        let sys_b = &self.systems[b];

        // Check explicit dependencies
        if sys_a.depends_on.contains(&sys_b.name) || sys_b.depends_on.contains(&sys_a.name) {
            return false;
        }

        // Check component access conflicts
        // Two systems conflict if one writes to a component the other reads or writes
        for write_type in &sys_a.component_writes {
            if sys_b.component_writes.contains(write_type)
                || sys_b.component_reads.contains(write_type)
            {
                return false;
            }
        }

        for write_type in &sys_b.component_writes {
            if sys_a.component_writes.contains(write_type)
                || sys_a.component_reads.contains(write_type)
            {
                return false;
            }
        }

        true
    }

    // Build execution groups with parallel safety
    fn rebuild_execution_groups(&mut self) {
        if !self.dirty {
            trace!("Groups not dirty, skipping rebuild");
            return;
        }

        info!(
            "Rebuilding execution groups for {} systems",
            self.systems.len()
        );
        self.execution_groups.clear();
        let mut remaining: Vec<usize> = (0..self.systems.len()).collect();

        // Sort by priority (higher priority first)
        remaining.sort_by(|&a, &b| self.systems[b].priority.cmp(&self.systems[a].priority));
        trace!(
            "Systems sorted by priority: {:?}",
            remaining
                .iter()
                .map(|&i| (&self.systems[i].name, self.systems[i].priority))
                .collect::<Vec<_>>()
        );

        let mut group_count = 0;
        while !remaining.is_empty() {
            let mut current_group = Vec::new();
            let mut i = 0;

            while i < remaining.len() {
                let system_idx = remaining[i];
                let mut can_add = true;

                // Check if this system can run with systems already in the group
                for &existing_idx in &current_group {
                    if !self.can_run_parallel(system_idx, existing_idx) {
                        can_add = false;
                        break;
                    }
                }

                if can_add {
                    current_group.push(system_idx);
                    remaining.remove(i);
                    trace!(
                        "Added system '{}' to group {}",
                        self.systems[system_idx].name, group_count
                    );
                } else {
                    i += 1;
                }
            }

            if !current_group.is_empty() {
                trace!(
                    "Created group {} with {} systems",
                    group_count,
                    current_group.len()
                );
                self.execution_groups.push(current_group);
                group_count += 1;
            } else {
                error!("ERROR: Empty group created, breaking to avoid infinite loop");
                break;
            }
        }

        trace!("Total execution groups: {}", self.execution_groups.len());
        self.dirty = false;
    }

    pub fn run_all(&mut self, ecs_core: &mut ECSCore) {
        trace!("SystemScheduler::run_all() called");
        self.rebuild_execution_groups();

        trace!(
            "Processing {} execution groups",
            self.execution_groups.len()
        );

        for (group_idx, group) in self.execution_groups.iter().enumerate() {
            trace!(
                "Processing group {} with {} systems",
                group_idx,
                group.len()
            );

            let systems_to_run: Vec<_> = group
                .iter()
                .filter(|&&idx| self.systems[idx].enabled)
                .map(|&idx| (Arc::clone(&self.systems[idx].system), idx))
                .collect();

            // Run systems in parallel
            let execution_times: Vec<_> = systems_to_run
                .iter()
                .map(|(system, idx)| {
                    let start = Instant::now();
                    let mut execution_time = Duration::from_nanos(0);

                    if let Ok(mut sys) = system.lock() {
                        sys.run(ecs_core);
                        execution_time = start.elapsed();
                    } else {
                        tracing::error!("failed to lock system!")
                    }

                    (*idx, execution_time)
                })
                .collect();

            // Update execution time statistics (this part is correct)
            for (idx, duration) in execution_times {
                if duration > Duration::from_nanos(0) {
                    let system = &mut self.systems[idx];
                    system.avg_execution_time = Duration::from_nanos(
                        (system.avg_execution_time.as_nanos() as f64 * 0.9
                            + duration.as_nanos() as f64 * 0.1) as u64,
                    );
                    system.last_execution_time = Some(Instant::now());
                }
            }
        }
    }

    pub fn get_system_stats(&self) -> Vec<(String, Duration, bool)> {
        self.systems
            .iter()
            .map(|sys| (sys.name.clone(), sys.avg_execution_time, sys.enabled))
            .collect()
    }
}
