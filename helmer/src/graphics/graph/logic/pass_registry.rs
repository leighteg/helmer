#![allow(dead_code)]

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::Arc,
};

use crate::graphics::graph::definition::{render_pass::RenderPass, resource_id::ResourceId};
use crate::graphics::graph::logic::{
    gpu_resource_pool::GpuResourcePool, render_graph::RenderGraph,
};

/// Trait implemented by pass output structs to expose the resources they touch.
pub trait PassResourceOutput {
    fn resource_ids(&self) -> Vec<ResourceId>;
}

#[derive(Clone)]
pub struct PassHandle<O> {
    pub node: usize,
    pub outputs: Arc<O>,
}

#[derive(Clone)]
struct PassEntry {
    node: usize,
    outputs: Arc<dyn Any + Send + Sync>,
    resources: Vec<ResourceId>,
}

/// Type-indexed store for pass outputs and node metadata.
#[derive(Default, Clone)]
pub struct PassStore {
    entries: HashMap<TypeId, PassEntry>,
}

impl PassStore {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn insert<O>(&mut self, node: usize, outputs: O) -> PassHandle<O>
    where
        O: PassResourceOutput + Any + Send + Sync,
    {
        let resources = outputs.resource_ids();
        let outputs = Arc::new(outputs);

        self.entries.insert(
            TypeId::of::<O>(),
            PassEntry {
                node,
                outputs: outputs.clone(),
                resources,
            },
        );

        PassHandle { node, outputs }
    }

    pub fn handle<O: Any + Send + Sync>(&self) -> Option<PassHandle<O>> {
        self.entries.get(&TypeId::of::<O>()).and_then(|entry| {
            entry
                .outputs
                .clone()
                .downcast::<O>()
                .ok()
                .map(|outputs| PassHandle {
                    node: entry.node,
                    outputs,
                })
        })
    }

    pub fn outputs<O: Any + Send + Sync>(&self) -> Option<Arc<O>> {
        self.handle::<O>().map(|h| h.outputs)
    }

    pub fn node<O: Any + Send + Sync>(&self) -> Option<usize> {
        self.entries.get(&TypeId::of::<O>()).map(|entry| entry.node)
    }

    pub fn remove<O: Any + Send + Sync>(&mut self) -> Option<PassHandle<O>> {
        self.entries.remove(&TypeId::of::<O>()).and_then(|entry| {
            entry
                .outputs
                .downcast::<O>()
                .ok()
                .map(|outputs| PassHandle {
                    node: entry.node,
                    outputs,
                })
        })
    }

    pub fn resource_ids(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.entries
            .values()
            .flat_map(|entry| entry.resources.iter().copied())
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Helper to build a graph and registry together.
pub struct PassGraphBuilder<'a> {
    graph: RenderGraph,
    pool: &'a mut GpuResourcePool,
    store: PassStore,
}

impl<'a> PassGraphBuilder<'a> {
    pub fn new(pool: &'a mut GpuResourcePool) -> Self {
        Self {
            graph: RenderGraph::new(),
            pool,
            store: PassStore::new(),
        }
    }

    pub fn add<P, O, F>(&mut self, build: F) -> PassHandle<O>
    where
        P: RenderPass,
        O: PassResourceOutput + Any + Send + Sync,
        F: FnOnce(&mut GpuResourcePool, &PassStore) -> (P, O),
    {
        let (pass, outputs) = build(self.pool, &self.store);
        let node = self.graph.add_pass(pass);
        self.store.insert(node, outputs)
    }

    pub fn into_parts(self) -> (RenderGraph, PassStore) {
        (self.graph, self.store)
    }
}
