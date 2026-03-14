#![allow(dead_code)]

use parking_lot::RwLock;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use crate::graphics::graph::definition::{
    resource_desc::ResourceDesc, resource_flags::ResourceUsageHints, resource_id::ResourceId,
};

/// Logical resource registry that lives with the graph definition on the logic thread.
/// It only hands out ResourceIds and remembers the declared descriptors/hints.
/// The GPU pool on the render thread will realize these resources as needed.
#[derive(Clone, Default)]
pub struct RenderGraphResources {
    base_index: u32,
    counter: Arc<AtomicU32>,
    records: Arc<RwLock<Vec<Option<ResourceRecordEntry>>>>,
}

#[derive(Clone)]
struct ResourceRecordEntry {
    id: ResourceId,
    record: ResourceRecord,
}

#[derive(Clone)]
pub struct ResourceRecord {
    pub desc: ResourceDesc,
    pub hints: ResourceUsageHints,
}

impl RenderGraphResources {
    /// "base_index" keeps IDs from colliding with asset IDs or other systems.
    pub fn new(base_index: u32) -> Self {
        Self {
            base_index,
            counter: Arc::new(AtomicU32::new(0)),
            records: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn allocate(&self, desc: ResourceDesc, hints: Option<ResourceUsageHints>) -> ResourceId {
        let id = self.next_id(&desc);
        let hints = hints.unwrap_or_else(|| ResourceUsageHints {
            estimated_size_bytes: desc.estimate_size_bytes(),
            ..Default::default()
        });

        let local_idx = id.index().saturating_sub(self.base_index as usize);
        let mut records = self.records.write();
        if local_idx >= records.len() {
            records.resize_with(local_idx + 1, || None);
        }
        records[local_idx] = Some(ResourceRecordEntry {
            id,
            record: ResourceRecord { desc, hints },
        });

        id
    }

    fn next_id(&self, desc: &ResourceDesc) -> ResourceId {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed);
        ResourceId::new(desc.kind(), self.base_index + idx, 0)
    }

    pub fn record(&self, id: ResourceId) -> Option<ResourceRecord> {
        let local_idx = id.index().checked_sub(self.base_index as usize)?;
        self.records
            .read()
            .get(local_idx)
            .and_then(|entry| entry.as_ref().map(|entry| entry.record.clone()))
    }

    pub fn iter(&self) -> Vec<(ResourceId, ResourceRecord)> {
        self.records
            .read()
            .iter()
            .filter_map(|entry| entry.as_ref())
            .map(|entry| (entry.id, entry.record.clone()))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.counter.load(Ordering::Relaxed) as usize
    }
}
