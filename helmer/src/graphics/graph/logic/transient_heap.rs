#![allow(dead_code)]
use super::{residency::GpuResourceEntry, vram_budget::VramBudget};
use crate::graphics::graph::definition::{
    resource_desc::ResourceDesc,
    resource_flags::{ResourceFlags, ResourceUsageHints},
    resource_id::{ResourceId, ResourceKind},
};

/// A single pool of free resources for a specific descriptor
struct FreeList {
    resources: Vec<ResourceId>,
}

/// Hash bucket for open addressing
struct Bucket {
    hash: u64,
    desc: ResourceDesc,
    free_list: FreeList,
}

#[derive(Clone, Copy, Debug)]
pub struct TransientHeapConfig {
    pub initial_capacity: usize,
    pub max_load_factor: f32,
    pub max_free_per_desc: usize,
    pub max_total_free: usize,
}

impl TransientHeapConfig {
    fn normalized(self) -> Self {
        let capacity = self.initial_capacity.max(1).next_power_of_two();
        let max_load = self.max_load_factor.max(f32::EPSILON);
        Self {
            initial_capacity: capacity,
            max_load_factor: max_load,
            max_free_per_desc: self.max_free_per_desc,
            max_total_free: self.max_total_free,
        }
    }
}

/// High-performance transient resource allocator using open-addressed hash table
pub struct TransientHeap {
    /// Open-addressed hash table with linear probing.
    /// We use a custom implementation to avoid HashMap's Eq/Hash requirements.
    buckets: Vec<Option<Bucket>>,
    /// Number of occupied buckets
    count: usize,
    /// Mask for quick modulo (capacity is always power of 2)
    mask: usize,
    /// Total free resources tracked across all buckets
    total_free: usize,
    config: TransientHeapConfig,
}

impl TransientHeap {
    pub fn new(config: TransientHeapConfig) -> Self {
        let config = config.normalized();
        Self::with_capacity(config.initial_capacity, config)
    }

    pub fn set_config(&mut self, config: TransientHeapConfig) {
        self.config = config.normalized();
        if self.buckets.len() < self.config.initial_capacity {
            self.resize_to(self.config.initial_capacity);
        }
    }

    pub fn config(&self) -> TransientHeapConfig {
        self.config
    }

    fn with_capacity(capacity: usize, config: TransientHeapConfig) -> Self {
        let capacity = capacity.max(1).next_power_of_two();
        Self {
            buckets: (0..capacity).map(|_| None).collect(),
            count: 0,
            mask: capacity - 1,
            total_free: 0,
            config,
        }
    }

    /// Find the bucket index for a descriptor, or the first empty slot.
    /// Returns (index, found_existing)
    #[inline]
    fn find_slot(&self, desc: &ResourceDesc, hash: u64) -> (usize, bool) {
        let mut idx = (hash as usize) & self.mask;
        let mut probe_count = 0;

        loop {
            match &self.buckets[idx] {
                None => return (idx, false),
                Some(bucket) => {
                    // Check hash first (fast), then full equality
                    if bucket.hash == hash && bucket.desc == *desc {
                        return (idx, true);
                    }
                }
            }

            // Linear probing
            probe_count += 1;
            idx = (idx + 1) & self.mask;

            // Safety: prevent infinite loop (should never happen with load factor < 1)
            debug_assert!(probe_count < self.buckets.len(), "Hash table full!");
            if probe_count >= self.buckets.len() {
                return (idx, false);
            }
        }
    }

    fn resize_to(&mut self, new_capacity: usize) {
        let new_capacity = new_capacity.max(1).next_power_of_two();
        let old_buckets =
            std::mem::replace(&mut self.buckets, (0..new_capacity).map(|_| None).collect());

        self.mask = new_capacity - 1;
        self.count = 0;

        // Rehash all entries
        for bucket in old_buckets.into_iter().flatten() {
            let (idx, _) = self.find_slot(&bucket.desc, bucket.hash);
            self.buckets[idx] = Some(bucket);
            self.count += 1;
        }
    }

    /// Resize the hash table when load factor exceeds threshold
    fn resize(&mut self) {
        let new_capacity = self.buckets.len() * 2;
        self.resize_to(new_capacity);
    }

    /// Check if we need to resize
    #[inline]
    fn should_resize(&self) -> bool {
        (self.count as f32) / (self.buckets.len() as f32) > self.config.max_load_factor
    }

    /// Mark a transient resource available for reuse in future frames.
    ///
    /// Time complexity: O(1) average case, O(k) worst case where k = probe length
    pub fn release(&mut self, desc: &ResourceDesc, id: ResourceId) {
        if self.config.max_free_per_desc == 0 || self.config.max_total_free == 0 {
            return;
        }
        if self.total_free >= self.config.max_total_free {
            return;
        }

        let hash = desc.fast_hash();
        let (idx, found) = self.find_slot(desc, hash);

        if found {
            if let Some(bucket) = &mut self.buckets[idx] {
                if bucket.free_list.resources.len() >= self.config.max_free_per_desc {
                    return;
                }
                bucket.free_list.resources.push(id);
                self.total_free += 1;
            }
        } else {
            if self.should_resize() {
                self.resize();
                let (idx, _) = self.find_slot(desc, hash);
                self.buckets[idx] = Some(Bucket {
                    hash,
                    desc: desc.clone(),
                    free_list: FreeList {
                        resources: vec![id],
                    },
                });
            } else {
                self.buckets[idx] = Some(Bucket {
                    hash,
                    desc: desc.clone(),
                    free_list: FreeList {
                        resources: vec![id],
                    },
                });
            }
            self.count += 1;
            self.total_free += 1;
        }
    }

    /// Try to grab a previously allocated transient resource compatible with desc.
    ///
    /// Time complexity: O(1) average case, O(k) worst case where k = probe length
    #[inline]
    pub fn acquire(&mut self, desc: &ResourceDesc) -> Option<ResourceId> {
        let hash = desc.fast_hash();
        let (idx, found) = self.find_slot(desc, hash);

        if found {
            if let Some(bucket) = &mut self.buckets[idx] {
                let id = bucket.free_list.resources.pop();
                if id.is_some() {
                    self.total_free = self.total_free.saturating_sub(1);
                }
                return id;
            }
        }
        None
    }

    /// Compact the table by removing empty buckets and optionally shrinking.
    /// Call this periodically (e.g., every few frames) to reclaim memory.
    pub fn compact(&mut self, shrink: bool) {
        // Remove empty buckets by rehashing
        let old_buckets: Vec<_> = self
            .buckets
            .iter_mut()
            .filter_map(|b| b.take())
            .filter(|b| !b.free_list.resources.is_empty())
            .collect();

        self.count = old_buckets.len();
        self.total_free = old_buckets
            .iter()
            .map(|b| b.free_list.resources.len())
            .sum();

        // Optionally shrink to fit
        if shrink && self.count < self.buckets.len() / 4 {
            let new_capacity = (self.count * 2)
                .max(self.config.initial_capacity)
                .next_power_of_two();
            self.buckets = (0..new_capacity).map(|_| None).collect();
            self.mask = new_capacity - 1;
        } else {
            self.buckets.iter_mut().for_each(|b| *b = None);
        }

        // Reinsert non-empty buckets
        for bucket in old_buckets {
            let (idx, _) = self.find_slot(&bucket.desc, bucket.hash);
            self.buckets[idx] = Some(bucket);
        }
    }

    /// Clear all free resources without deallocating storage
    pub fn clear(&mut self) {
        for bucket in self.buckets.iter_mut().flatten() {
            bucket.free_list.resources.clear();
        }
        self.total_free = 0;
    }

    /// Get statistics for debugging/profiling
    pub fn stats(&self) -> TransientHeapStats {
        let total_free = self.total_free;

        let memory = std::mem::size_of::<Option<Bucket>>() * self.buckets.capacity()
            + self
                .buckets
                .iter()
                .flatten()
                .map(|b| {
                    std::mem::size_of::<ResourceDesc>()
                        + b.free_list.resources.capacity() * std::mem::size_of::<ResourceId>()
                })
                .sum::<usize>();

        // Calculate average probe length
        let mut total_probes = 0;
        let mut samples = 0;
        for bucket in self.buckets.iter().flatten() {
            let (mut idx, mut probes) = ((bucket.hash as usize) & self.mask, 0);
            while self.buckets[idx].is_some() {
                if let Some(b) = &self.buckets[idx] {
                    if b.hash == bucket.hash {
                        break;
                    }
                }
                probes += 1;
                idx = (idx + 1) & self.mask;
            }
            total_probes += probes;
            samples += 1;
        }

        TransientHeapStats {
            unique_descriptors: self.count,
            total_free_resources: total_free,
            memory_bytes: memory,
            capacity: self.buckets.len(),
            load_factor: (self.count as f32) / (self.buckets.len() as f32),
            avg_probe_length: if samples > 0 {
                total_probes as f32 / samples as f32
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TransientHeapStats {
    pub unique_descriptors: usize,
    pub total_free_resources: usize,
    pub memory_bytes: usize,
    pub capacity: usize,
    pub load_factor: f32,
    pub avg_probe_length: f32,
}

/// Allocate (or reuse) a transient resource entry in the pool.
/// This is called by GpuResourcePool.
pub fn allocate_transient_entry(
    heap: &mut TransientHeap,
    vram: &mut VramBudget,
    pool_entries: &mut Vec<Option<GpuResourceEntry>>,
    desc: ResourceDesc,
    frame_index: u32,
) -> GpuResourceEntry {
    let kind = desc.kind();
    let est_size = desc.estimate_size_bytes();
    let hints = ResourceUsageHints {
        flags: ResourceFlags::TRANSIENT,
        estimated_size_bytes: est_size,
    };

    // try reuse first
    if let Some(reuse_id) = heap.acquire(&desc) {
        let idx = reuse_id.index();
        if let Some(Some(entry)) = pool_entries.get_mut(idx).map(|e| e.as_mut()) {
            entry.last_used_frame = frame_index;
            entry.residency = super::residency::Residency::Resident;
            return entry.clone();
        }
    }

    // otherwise allocate a new slot in the pool
    let mut idx = None;
    for (i, slot) in pool_entries.iter().enumerate() {
        if slot.is_none() {
            idx = Some(i);
            break;
        }
    }
    let index = idx.unwrap_or_else(|| {
        pool_entries.push(None);
        pool_entries.len() - 1
    }) as u32;

    let id = ResourceId::new(ResourceKind::Transient, index, 0);
    let mut entry = GpuResourceEntry::new(id, kind, est_size, hints, frame_index, desc);
    entry.residency = super::residency::Residency::Resident;
    vram.add(kind, est_size);
    pool_entries[index as usize] = Some(entry.clone());
    entry
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TransientHeapConfig {
        TransientHeapConfig {
            initial_capacity: 64,
            max_load_factor: 0.75,
            max_free_per_desc: 32,
            max_total_free: 10_000,
        }
    }

    #[test]
    fn test_basic_acquire_release() {
        let mut heap = TransientHeap::new(test_config());
        let desc = ResourceDesc::Buffer {
            size: 1024,
            usage: wgpu::BufferUsages::VERTEX,
        };
        let id = ResourceId::new(ResourceKind::Transient, 0, 0);

        assert!(heap.acquire(&desc).is_none());
        heap.release(&desc, id);
        assert_eq!(heap.acquire(&desc), Some(id));
        assert!(heap.acquire(&desc).is_none());
    }

    #[test]
    fn test_many_descriptors() {
        let mut heap = TransientHeap::new(test_config());

        // Insert 10,000 unique descriptors
        for i in 0..10_000 {
            let desc = ResourceDesc::Buffer {
                size: 1024 + i as u64,
                usage: wgpu::BufferUsages::VERTEX,
            };
            let id = ResourceId::new(ResourceKind::Transient, i, 0);
            heap.release(&desc, id);
        }

        let stats = heap.stats();
        assert_eq!(stats.unique_descriptors, 10_000);
        assert!(stats.load_factor < 0.75);
        assert!(
            stats.avg_probe_length < 5.0,
            "Probe length too high: {}",
            stats.avg_probe_length
        );
    }

    #[test]
    fn test_collision_handling() {
        let mut heap = TransientHeap::new(test_config());

        // Create descriptors that might collide
        for i in 0..1000 {
            let desc = ResourceDesc::Buffer {
                size: (i % 10) as u64 * 1024, // Only 10 unique sizes
                usage: wgpu::BufferUsages::VERTEX,
            };
            let id = ResourceId::new(ResourceKind::Transient, i, 0);
            heap.release(&desc, id);
        }

        // All should be retrievable
        for i in 0..1000 {
            let desc = ResourceDesc::Buffer {
                size: (i % 10) as u64 * 1024,
                usage: wgpu::BufferUsages::VERTEX,
            };
            assert!(heap.acquire(&desc).is_some());
        }
    }

    #[test]
    fn test_compact() {
        let mut heap = TransientHeap::new(test_config());

        for i in 0..100 {
            let desc = ResourceDesc::Buffer {
                size: 1024 + i as u64,
                usage: wgpu::BufferUsages::VERTEX,
            };
            let id = ResourceId::new(ResourceKind::Transient, i, 0);
            heap.release(&desc, id);
        }

        // Drain half
        for i in 0..50 {
            let desc = ResourceDesc::Buffer {
                size: 1024 + i as u64,
                usage: wgpu::BufferUsages::VERTEX,
            };
            heap.acquire(&desc);
        }

        let before = heap.stats();
        heap.compact(false);
        let after = heap.stats();

        assert_eq!(after.unique_descriptors, 50);
        assert!(after.memory_bytes <= before.memory_bytes);
    }
}
