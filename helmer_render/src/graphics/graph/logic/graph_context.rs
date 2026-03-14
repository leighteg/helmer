#![allow(dead_code)]

use crate::graphics::graph::definition::resource_id::ResourceId;

/// During graph building, each pass receives this context.
/// It declares what resources are read/written.
pub struct RenderGraphContext {
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
}

impl RenderGraphContext {
    pub fn new() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Mark resource as read.
    pub fn read(&mut self, id: ResourceId) {
        self.reads.push(id);
    }

    /// Mark resource as written.
    pub fn write(&mut self, id: ResourceId) {
        self.writes.push(id);
    }

    /// Convenience: read/write same resource.
    pub fn rw(&mut self, id: ResourceId) {
        self.reads.push(id);
        self.writes.push(id);
    }
}
