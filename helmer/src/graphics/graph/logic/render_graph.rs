#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graphics::graph::definition::render_pass::RenderPass;
use crate::graphics::graph::definition::resource_flags::ResourceFlags;
use crate::graphics::graph::definition::resource_id::ResourceId;
use crate::graphics::graph::logic::gpu_resource_pool::GpuResourcePool;
use crate::graphics::graph::logic::graph_context::RenderGraphContext;
use crate::graphics::graph::logic::lifetime::{LifetimeAnalyzer, ResourceLifetime};

/// A node in the graph DAG
pub struct RenderGraphNode {
    pub id: usize,
    pub pass: Box<dyn RenderPass>,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
}

/// Transient alias mapping produced by compilation
#[derive(Clone, Copy, Debug)]
pub struct TransientAlias {
    pub alias: ResourceId,
    pub root: ResourceId,
}

/// Final compiled graph ordering
pub struct RenderGraphCompilation {
    /// Order of execution (sorted DAG)
    pub sorted_nodes: Vec<usize>,
    /// Topological levels for parallel recording
    pub levels: Vec<Vec<usize>>,
    /// Pass order per node (node id -> order)
    pub pass_order: Vec<u32>,
    /// Stable pass names per node (node id -> name)
    pub pass_names: Vec<&'static str>,
    /// Transient alias pairs (alias -> root)
    pub transient_aliases: Vec<TransientAlias>,
}

/// Errors constructing or sorting the graph
#[derive(Debug)]
pub enum GraphError {
    CycleDetected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AccessKind {
    Read,
    Write,
    ReadWrite,
}

impl AccessKind {
    fn order(self) -> u8 {
        match self {
            AccessKind::Read => 0,
            AccessKind::Write => 1,
            AccessKind::ReadWrite => 2,
        }
    }

    fn merge(self, other: AccessKind) -> AccessKind {
        match (self, other) {
            (AccessKind::Read, AccessKind::Write) | (AccessKind::Write, AccessKind::Read) => {
                AccessKind::ReadWrite
            }
            (AccessKind::ReadWrite, _) | (_, AccessKind::ReadWrite) => AccessKind::ReadWrite,
            (a, _) => a,
        }
    }

    fn reads(self) -> bool {
        matches!(self, AccessKind::Read | AccessKind::ReadWrite)
    }

    fn writes(self) -> bool {
        matches!(self, AccessKind::Write | AccessKind::ReadWrite)
    }
}

#[derive(Clone, Copy, Debug)]
struct ResourceUse {
    resource: ResourceId,
    pass: usize,
    access: AccessKind,
}

#[derive(Clone, Copy, Debug)]
struct AccessEntry {
    pass: usize,
    access: AccessKind,
}

struct ResourceAccess {
    resource: ResourceId,
    readers: Vec<usize>,
    writers: Vec<usize>,
    accesses: Vec<AccessEntry>,
}

#[derive(Clone)]
struct TransientCandidate {
    id: ResourceId,
    desc: crate::graphics::graph::definition::resource_desc::ResourceDesc,
    lifetime: ResourceLifetime,
    hash: u64,
}

/// The graph builder
pub struct RenderGraph {
    nodes: Vec<RenderGraphNode>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn nodes(&self) -> &[RenderGraphNode] {
        &self.nodes
    }

    /// Add a pass to the graph
    pub fn add_pass<P: RenderPass>(&mut self, pass: P) -> usize {
        let mut ctx = RenderGraphContext::new();
        pass.setup(&mut ctx);

        let id = self.nodes.len();

        let mut reads = ctx.reads;
        reads.sort_unstable_by_key(|r| r.raw());
        reads.dedup_by_key(|r| r.raw());

        let mut writes = ctx.writes;
        writes.sort_unstable_by_key(|r| r.raw());
        writes.dedup_by_key(|r| r.raw());

        self.nodes.push(RenderGraphNode {
            id,
            pass: Box::new(pass),
            reads,
            writes,
        });

        id
    }

    /// Compile graph: build DAG + sort + alias transients
    pub fn compile(&self, pool: &GpuResourcePool) -> Result<RenderGraphCompilation, GraphError> {
        let node_count = self.nodes.len();
        if node_count == 0 {
            return Ok(RenderGraphCompilation {
                sorted_nodes: Vec::new(),
                levels: Vec::new(),
                pass_order: Vec::new(),
                pass_names: Vec::new(),
                transient_aliases: Vec::new(),
            });
        }

        let mut uses = Vec::new();
        for node in &self.nodes {
            for &res in &node.reads {
                uses.push(ResourceUse {
                    resource: res,
                    pass: node.id,
                    access: AccessKind::Read,
                });
            }
            for &res in &node.writes {
                uses.push(ResourceUse {
                    resource: res,
                    pass: node.id,
                    access: AccessKind::Write,
                });
            }
        }

        uses.sort_unstable_by(|a, b| {
            a.resource
                .raw()
                .cmp(&b.resource.raw())
                .then_with(|| a.pass.cmp(&b.pass))
                .then_with(|| a.access.order().cmp(&b.access.order()))
        });

        let mut resource_accesses = Vec::new();
        let mut idx = 0usize;
        while idx < uses.len() {
            let resource = uses[idx].resource;
            let mut readers = Vec::new();
            let mut writers = Vec::new();
            let mut accesses = Vec::new();

            while idx < uses.len() && uses[idx].resource.raw() == resource.raw() {
                let pass = uses[idx].pass;
                let mut access = uses[idx].access;
                idx += 1;

                while idx < uses.len()
                    && uses[idx].resource.raw() == resource.raw()
                    && uses[idx].pass == pass
                {
                    access = access.merge(uses[idx].access);
                    idx += 1;
                }

                if access.reads() {
                    readers.push(pass);
                }
                if access.writes() {
                    writers.push(pass);
                }

                accesses.push(AccessEntry { pass, access });
            }

            resource_accesses.push(ResourceAccess {
                resource,
                readers,
                writers,
                accesses,
            });
        }

        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut reads_since_write: Vec<usize> = Vec::new();
        for access in &resource_accesses {
            let mut last_write: Option<usize> = None;
            reads_since_write.clear();

            for entry in &access.accesses {
                match entry.access {
                    AccessKind::Read => {
                        if let Some(writer) = last_write {
                            edges.push((writer, entry.pass));
                        }
                        reads_since_write.push(entry.pass);
                    }
                    AccessKind::Write => {
                        if let Some(writer) = last_write {
                            edges.push((writer, entry.pass));
                        }
                        for reader in &reads_since_write {
                            edges.push((*reader, entry.pass));
                        }
                        reads_since_write.clear();
                        last_write = Some(entry.pass);
                    }
                    AccessKind::ReadWrite => {
                        if let Some(writer) = last_write {
                            edges.push((writer, entry.pass));
                        }
                        for reader in &reads_since_write {
                            edges.push((*reader, entry.pass));
                        }
                        reads_since_write.clear();
                        last_write = Some(entry.pass);
                    }
                }
            }
        }

        edges.sort_unstable();
        edges.dedup();

        let mut adjacency = vec![Vec::new(); node_count];
        let mut indegree = vec![0usize; node_count];
        for (src, dst) in edges {
            if src == dst {
                continue;
            }
            adjacency[src].push(dst);
            indegree[dst] += 1;
        }

        let mut ready: Vec<usize> = indegree
            .iter()
            .enumerate()
            .filter_map(|(idx, &count)| if count == 0 { Some(idx) } else { None })
            .collect();
        let mut sorted_nodes = Vec::with_capacity(node_count);
        let mut levels = Vec::new();
        let mut indegree_work = indegree;

        while !ready.is_empty() {
            ready.sort_unstable();
            let level = ready;
            let mut next = Vec::new();

            for &node in &level {
                sorted_nodes.push(node);
                for &dst in &adjacency[node] {
                    let count = indegree_work.get_mut(dst).unwrap();
                    *count -= 1;
                    if *count == 0 {
                        next.push(dst);
                    }
                }
            }

            levels.push(level);
            ready = next;
        }

        if sorted_nodes.len() != node_count {
            return Err(GraphError::CycleDetected);
        }

        let mut pass_order = vec![0u32; node_count];
        for (order, node) in sorted_nodes.iter().enumerate() {
            pass_order[*node] = order as u32;
        }
        let mut pass_names = vec![""; node_count];
        for node in &self.nodes {
            pass_names[node.id] = node.pass.name();
        }

        let mut lifetimes = LifetimeAnalyzer::with_resources(
            resource_accesses
                .iter()
                .map(|access| access.resource)
                .collect(),
        );
        for access in &resource_accesses {
            let order = access.accesses.iter().map(|entry| pass_order[entry.pass]);
            for pass_index in order {
                lifetimes.note_use(access.resource, pass_index);
            }
        }

        let transient_aliases = build_transient_aliases(&lifetimes, pool);

        Ok(RenderGraphCompilation {
            sorted_nodes,
            levels,
            pass_order,
            pass_names,
            transient_aliases,
        })
    }
}

fn build_transient_aliases(
    lifetimes: &LifetimeAnalyzer,
    pool: &GpuResourcePool,
) -> Vec<TransientAlias> {
    let mut candidates = Vec::new();

    for (id, lifetime) in lifetimes.entries() {
        let Some(entry) = pool.entry(id) else {
            continue;
        };
        if !entry.hints.flags.contains(ResourceFlags::TRANSIENT) {
            continue;
        }

        candidates.push(TransientCandidate {
            id,
            desc: entry.desc.clone(),
            lifetime,
            hash: entry.desc.fast_hash(),
        });
    }

    if candidates.is_empty() {
        return Vec::new();
    }

    candidates.sort_unstable_by(|a, b| a.hash.cmp(&b.hash));

    let mut aliases = Vec::new();
    let mut idx = 0usize;
    while idx < candidates.len() {
        let hash = candidates[idx].hash;
        let mut end = idx + 1;
        while end < candidates.len() && candidates[end].hash == hash {
            end += 1;
        }

        let mut group_used = vec![false; end - idx];
        for local_start in 0..(end - idx) {
            if group_used[local_start] {
                continue;
            }
            let base = &candidates[idx + local_start];
            let mut group = Vec::new();
            for local in local_start..(end - idx) {
                if group_used[local] {
                    continue;
                }
                let candidate = &candidates[idx + local];
                if candidate.desc == base.desc {
                    group_used[local] = true;
                    group.push(candidate);
                }
            }

            if group.len() <= 1 {
                continue;
            }

            let mut ordered: Vec<_> = group.into_iter().cloned().collect();
            ordered.sort_unstable_by(|a, b| {
                a.lifetime
                    .first_pass
                    .cmp(&b.lifetime.first_pass)
                    .then(a.lifetime.last_pass.cmp(&b.lifetime.last_pass))
            });

            let mut heap: BinaryHeap<Reverse<(u32, ResourceId)>> = BinaryHeap::new();
            for candidate in ordered {
                if let Some(Reverse((last, root))) = heap.peek().copied() {
                    if last < candidate.lifetime.first_pass {
                        heap.pop();
                        aliases.push(TransientAlias {
                            alias: candidate.id,
                            root,
                        });
                        heap.push(Reverse((candidate.lifetime.last_pass, root)));
                        continue;
                    }
                }

                heap.push(Reverse((candidate.lifetime.last_pass, candidate.id)));
            }
        }

        idx = end;
    }

    aliases
}
