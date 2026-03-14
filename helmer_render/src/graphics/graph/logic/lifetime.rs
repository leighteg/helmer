#![allow(dead_code)]

use crate::graphics::graph::definition::resource_id::ResourceId;

/// Simple lifetime analyzer for the render graph.
/// For each resource, records first and last pass index.
#[derive(Copy, Clone, Debug)]
pub struct ResourceLifetime {
    pub first_pass: u32,
    pub last_pass: u32,
}

pub struct LifetimeAnalyzer {
    resources: Vec<ResourceId>,
    lifetimes: Vec<Option<ResourceLifetime>>,
}

impl LifetimeAnalyzer {
    pub fn with_resources(mut resources: Vec<ResourceId>) -> Self {
        resources.sort_unstable_by_key(|r| r.raw());
        resources.dedup_by_key(|r| r.raw());
        let lifetimes = vec![None; resources.len()];
        Self {
            resources,
            lifetimes,
        }
    }

    /// Called by graph builder / compiler when a resource is used in a pass.
    pub fn note_use(&mut self, res: ResourceId, pass_index: u32) {
        let Ok(idx) = self.resources.binary_search_by_key(&res.raw(), |r| r.raw()) else {
            return;
        };

        match &mut self.lifetimes[idx] {
            Some(entry) => {
                entry.first_pass = entry.first_pass.min(pass_index);
                entry.last_pass = entry.last_pass.max(pass_index);
            }
            None => {
                self.lifetimes[idx] = Some(ResourceLifetime {
                    first_pass: pass_index,
                    last_pass: pass_index,
                });
            }
        }
    }

    pub fn lifetime(&self, res: ResourceId) -> Option<ResourceLifetime> {
        let idx = self
            .resources
            .binary_search_by_key(&res.raw(), |r| r.raw())
            .ok()?;
        self.lifetimes[idx]
    }

    pub fn entries(&self) -> impl Iterator<Item = (ResourceId, ResourceLifetime)> + '_ {
        self.resources
            .iter()
            .copied()
            .zip(self.lifetimes.iter().copied())
            .filter_map(|(res, lifetime)| lifetime.map(|life| (res, life)))
    }
}
