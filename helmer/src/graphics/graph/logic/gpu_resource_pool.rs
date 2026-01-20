#![allow(dead_code)]

use crate::graphics::graph::definition::{
    resource_desc::ResourceDesc,
    resource_flags::{ResourceFlags, ResourceUsageHints},
    resource_id::{ResourceId, ResourceKind},
};
use crate::graphics::graph::logic::graph_resources::RenderGraphResources;

use super::{
    residency::{GpuResourceEntry, Residency},
    transient_heap::{TransientHeap, TransientHeapConfig},
    vram_budget::VramBudget,
};

const GENERATION_MASK: u32 = 0x00FF_FFFF;

fn resource_kind_count() -> usize {
    ResourceKind::Transient as usize + 1
}

#[derive(Clone, Copy, Debug)]
pub struct GpuResourcePoolConfig {
    pub asset_map_initial_capacity: usize,
    pub asset_map_max_load_factor: f32,
    pub transient_heap: TransientHeapConfig,
    pub idle_frames_before_evict: u32,
    pub streaming_min_residency_frames: u32,
    pub max_evictions_per_tick: usize,
    pub eviction_scan_budget: usize,
    pub eviction_purge_budget: usize,
}

struct PoolSlot {
    generation: u32,
    entry: Option<GpuResourceEntry>,
    asset_id: Option<u32>,
    alias_of: Option<ResourceId>,
    binding_version: u64,
}

impl PoolSlot {
    fn new() -> Self {
        Self {
            generation: 0,
            entry: None,
            asset_id: None,
            alias_of: None,
            binding_version: 0,
        }
    }

    fn is_free(&self) -> bool {
        self.entry.is_none() && self.asset_id.is_none() && self.alias_of.is_none()
    }
}

#[derive(Clone, Copy)]
struct LruLinks {
    prev: Option<usize>,
    next: Option<usize>,
    in_list: bool,
}

impl Default for LruLinks {
    fn default() -> Self {
        Self {
            prev: None,
            next: None,
            in_list: false,
        }
    }
}

struct LruList {
    head: Option<usize>,
    tail: Option<usize>,
    links: Vec<LruLinks>,
}

impl LruList {
    fn new() -> Self {
        Self {
            head: None,
            tail: None,
            links: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, len: usize) {
        if self.links.len() < len {
            self.links.resize_with(len, LruLinks::default);
        }
    }

    fn remove(&mut self, idx: usize) {
        if idx >= self.links.len() || !self.links[idx].in_list {
            return;
        }

        let prev = self.links[idx].prev;
        let next = self.links[idx].next;

        if let Some(prev_idx) = prev {
            self.links[prev_idx].next = next;
        } else {
            self.head = next;
        }

        if let Some(next_idx) = next {
            self.links[next_idx].prev = prev;
        } else {
            self.tail = prev;
        }

        self.links[idx].prev = None;
        self.links[idx].next = None;
        self.links[idx].in_list = false;
    }

    fn push_front(&mut self, idx: usize) {
        if idx >= self.links.len() {
            return;
        }

        let old_head = self.head;
        self.links[idx].prev = None;
        self.links[idx].next = old_head;
        self.links[idx].in_list = true;

        if let Some(head_idx) = old_head {
            self.links[head_idx].prev = Some(idx);
        } else {
            self.tail = Some(idx);
        }

        self.head = Some(idx);
    }

    fn touch(&mut self, idx: usize) {
        if idx >= self.links.len() {
            return;
        }
        if self.links[idx].in_list {
            self.remove(idx);
        }
        self.push_front(idx);
    }

    fn iter_tail(&self) -> LruIter<'_> {
        LruIter {
            current: self.tail,
            links: &self.links,
        }
    }
}

struct LruIter<'a> {
    current: Option<usize>,
    links: &'a [LruLinks],
}

impl<'a> Iterator for LruIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current?;
        self.current = self.links.get(idx).and_then(|link| link.prev);
        Some(idx)
    }
}

struct AssetSlotBucket {
    asset_id: u32,
    slot: usize,
}

struct AssetSlotMap {
    buckets: Vec<Option<AssetSlotBucket>>,
    count: usize,
    mask: usize,
    max_load_factor: f32,
}

impl AssetSlotMap {
    fn new(initial_capacity: usize, max_load_factor: f32) -> Self {
        let capacity = initial_capacity.max(1).next_power_of_two();
        Self {
            buckets: (0..capacity).map(|_| None).collect(),
            count: 0,
            mask: capacity - 1,
            max_load_factor: max_load_factor.max(f32::EPSILON),
        }
    }

    fn set_config(&mut self, initial_capacity: usize, max_load_factor: f32) {
        self.max_load_factor = max_load_factor.max(f32::EPSILON);
        if self.buckets.len() < initial_capacity {
            self.resize_to(initial_capacity);
        }
    }

    fn get(&self, asset_id: u32) -> Option<usize> {
        let (idx, found) = self.find_slot(asset_id);
        if found {
            self.buckets[idx].as_ref().map(|b| b.slot)
        } else {
            None
        }
    }

    fn insert(&mut self, asset_id: u32, slot: usize) {
        if self.should_resize() {
            self.resize();
        }

        let (idx, found) = self.find_slot(asset_id);
        if found {
            if let Some(bucket) = &mut self.buckets[idx] {
                bucket.slot = slot;
            }
            return;
        }

        self.buckets[idx] = Some(AssetSlotBucket { asset_id, slot });
        self.count += 1;
    }

    fn find_slot(&self, asset_id: u32) -> (usize, bool) {
        let mut idx = (asset_id as usize) & self.mask;
        let mut probe_count = 0usize;

        loop {
            match &self.buckets[idx] {
                None => return (idx, false),
                Some(bucket) => {
                    if bucket.asset_id == asset_id {
                        return (idx, true);
                    }
                }
            }

            probe_count += 1;
            idx = (idx + 1) & self.mask;
            debug_assert!(probe_count < self.buckets.len(), "Asset map full!");
            if probe_count >= self.buckets.len() {
                return (idx, false);
            }
        }
    }

    fn should_resize(&self) -> bool {
        (self.count as f32) / (self.buckets.len() as f32) > self.max_load_factor
    }

    fn resize(&mut self) {
        let new_capacity = self.buckets.len() * 2;
        self.resize_to(new_capacity);
    }

    fn resize_to(&mut self, new_capacity: usize) {
        let new_capacity = new_capacity.max(1).next_power_of_two();
        let old = std::mem::replace(&mut self.buckets, (0..new_capacity).map(|_| None).collect());
        self.mask = new_capacity - 1;
        self.count = 0;

        for bucket in old.into_iter().flatten() {
            let (idx, _) = self.find_slot(bucket.asset_id);
            self.buckets[idx] = Some(bucket);
            self.count += 1;
        }
    }
}

/// Global GPU resource pool for the render graph.
pub struct GpuResourcePool {
    slots: Vec<PoolSlot>,
    free_list: Vec<usize>,
    lru: LruList,
    asset_maps: Vec<AssetSlotMap>,
    vram: VramBudget,
    transient_heap: TransientHeap,
    resident_count: u32,
    eviction_scan_budget: usize,
    eviction_purge_budget: usize,
    purge_cursor: usize,
    binding_epoch: u64,
    pending_evictions: Vec<ResourceId>,
    binding_changes: Vec<ResourceId>,

    /// Idle frame threshold before an evictable resource becomes a candidate.
    pub idle_frames_before_evict: u32,
    streaming_min_residency_frames: u32,
    max_evictions_per_tick: usize,
}

impl GpuResourcePool {
    pub fn new(global_soft: u64, global_hard: u64, config: GpuResourcePoolConfig) -> Self {
        let mut asset_maps = Vec::with_capacity(resource_kind_count());
        for _ in 0..resource_kind_count() {
            asset_maps.push(AssetSlotMap::new(
                config.asset_map_initial_capacity,
                config.asset_map_max_load_factor,
            ));
        }

        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            lru: LruList::new(),
            asset_maps,
            vram: VramBudget::new(global_soft, global_hard),
            transient_heap: TransientHeap::new(config.transient_heap),
            resident_count: 0,
            eviction_scan_budget: config.eviction_scan_budget,
            eviction_purge_budget: config.eviction_purge_budget,
            purge_cursor: 0,
            binding_epoch: 0,
            pending_evictions: Vec::new(),
            binding_changes: Vec::new(),
            idle_frames_before_evict: config.idle_frames_before_evict,
            streaming_min_residency_frames: config.streaming_min_residency_frames,
            max_evictions_per_tick: config.max_evictions_per_tick,
        }
    }

    pub fn apply_config(&mut self, config: GpuResourcePoolConfig) {
        for map in &mut self.asset_maps {
            map.set_config(
                config.asset_map_initial_capacity,
                config.asset_map_max_load_factor,
            );
        }
        self.transient_heap.set_config(config.transient_heap);
        self.idle_frames_before_evict = config.idle_frames_before_evict;
        self.streaming_min_residency_frames = config.streaming_min_residency_frames;
        self.max_evictions_per_tick = config.max_evictions_per_tick;
        self.eviction_scan_budget = config
            .eviction_scan_budget
            .max(config.max_evictions_per_tick);
        self.eviction_purge_budget = config.eviction_purge_budget;
    }

    fn next_generation(generation: u32) -> u32 {
        generation.wrapping_add(1) & GENERATION_MASK
    }

    fn ensure_slot_capacity(&mut self, idx: usize) {
        if idx >= self.slots.len() {
            self.slots.resize_with(idx + 1, PoolSlot::new);
            self.lru.ensure_capacity(self.slots.len());
        }
    }

    fn bump_binding_version(&mut self, idx: usize) {
        if let Some(slot) = self.slots.get_mut(idx) {
            slot.binding_version = slot.binding_version.wrapping_add(1);
        }
    }

    fn alloc_slot(&mut self) -> usize {
        if let Some(idx) = self.free_list.pop() {
            return idx;
        }
        let idx = self.slots.len();
        self.slots.push(PoolSlot::new());
        self.lru.ensure_capacity(self.slots.len());
        idx
    }

    fn slot_index(&self, id: ResourceId) -> Option<usize> {
        let idx = id.index();
        let slot = self.slots.get(idx)?;
        if slot.generation != id.generation() {
            return None;
        }
        Some(idx)
    }

    fn resolve_slot_index(&self, id: ResourceId) -> Option<usize> {
        let mut current = id;
        let mut hops = 0usize;
        loop {
            let idx = self.slot_index(current)?;
            let alias = self.slots[idx].alias_of;
            match alias {
                Some(next) => {
                    current = next;
                    hops += 1;
                    if hops > self.slots.len() {
                        return None;
                    }
                }
                None => return Some(idx),
            }
        }
    }

    fn resolve_id(&self, id: ResourceId) -> Option<ResourceId> {
        let mut current = id;
        let mut hops = 0usize;
        loop {
            let idx = self.slot_index(current)?;
            let alias = self.slots[idx].alias_of;
            match alias {
                Some(next) => {
                    current = next;
                    hops += 1;
                    if hops > self.slots.len() {
                        return None;
                    }
                }
                None => return Some(current),
            }
        }
    }

    fn ensure_slot_for_id(&mut self, id: ResourceId) -> Option<usize> {
        let idx = id.index();
        self.ensure_slot_capacity(idx);
        let slot = &mut self.slots[idx];

        if slot.is_free() {
            slot.generation = id.generation();
            slot.alias_of = None;
            return Some(idx);
        }

        if slot.generation != id.generation() {
            return None;
        }

        Some(idx)
    }

    fn remove_entry(&mut self, idx: usize) {
        if idx >= self.slots.len() {
            return;
        }
        if let Some(old) = self.slots[idx].entry.take() {
            if old.residency == Residency::Resident {
                self.vram.sub(old.kind, old.desc_size_bytes);
                self.resident_count = self.resident_count.saturating_sub(1);
                self.lru.remove(idx);
            }
            if old.residency == Residency::Resident && Self::should_record_binding_change(&old) {
                self.bump_binding_version(idx);
                self.binding_changes.push(old.id);
            }
        }
    }

    fn make_resident(&mut self, idx: usize, entry: &mut GpuResourceEntry) {
        let size = entry.desc_size_bytes;
        let kind = entry.kind;
        self.ensure_budget(kind, size, entry.last_used_frame);

        if self.vram.can_allocate(kind, size) {
            self.vram.add(kind, size);
            if entry.residency != Residency::Resident {
                self.resident_count = self.resident_count.saturating_add(1);
            }
            entry.residency = Residency::Resident;
            self.lru.touch(idx);
        } else {
            entry.residency = Residency::Evicted;
            entry.buffer = None;
            entry.texture = None;
            entry.texture_view = None;
            entry.sampler = None;
            self.lru.remove(idx);
        }
    }

    fn ensure_budget(&mut self, kind: ResourceKind, size: u64, frame_index: u32) {
        let global_hard = self.vram.global.hard_limit_bytes;
        let kind_hard = self.vram.per_kind[kind as usize].hard_limit_bytes;
        if global_hard == 0 || kind_hard == 0 {
            return;
        }

        let global_over = self
            .vram
            .global
            .current_bytes
            .saturating_add(size)
            .saturating_sub(global_hard);
        let kind_over = self.vram.per_kind[kind as usize]
            .current_bytes
            .saturating_add(size)
            .saturating_sub(kind_hard);

        let target = global_over.max(kind_over);
        if target == 0 {
            return;
        }

        let _ = self.evict_budget_based_internal(frame_index, target, Some(kind), true);
    }

    fn evict_index(&mut self, idx: usize, record_eviction: bool) {
        if idx >= self.slots.len() {
            return;
        }
        let mut bump_epoch = false;
        let mut record_id = None;
        let mut record_binding_change = None;
        if let Some(entry) = self.slots[idx].entry.as_mut() {
            let was_resident = entry.residency == Residency::Resident;
            if was_resident && Self::should_record_binding_change(entry) {
                record_binding_change = Some(entry.id);
            }
            if was_resident {
                self.vram.sub(entry.kind, entry.desc_size_bytes);
                self.resident_count = self.resident_count.saturating_sub(1);
                self.lru.remove(idx);
            }
            entry.residency = Residency::Evicted;
            entry.buffer = None;
            entry.texture = None;
            entry.texture_view = None;
            entry.sampler = None;
            if was_resident {
                bump_epoch = true;
                if record_eviction {
                    record_id = Some(entry.id);
                }
            }
        }
        if bump_epoch {
            self.bump_binding_epoch();
            if let Some(id) = record_id {
                self.pending_evictions.push(id);
            }
        }
        if let Some(id) = record_binding_change {
            self.bump_binding_version(idx);
            self.binding_changes.push(id);
        }
    }

    fn purge_evicted(&mut self, budget: usize) -> usize {
        if budget == 0 || self.slots.is_empty() {
            return 0;
        }

        let mut removed = 0usize;
        let slot_count = self.slots.len();
        let mut scanned = 0usize;

        while scanned < budget {
            let idx = self.purge_cursor % slot_count;
            self.purge_cursor = if idx + 1 >= slot_count { 0 } else { idx + 1 };
            scanned += 1;

            let slot = &mut self.slots[idx];
            if slot.asset_id.is_some() || slot.alias_of.is_some() {
                continue;
            }
            let remove = match slot.entry.as_ref() {
                Some(entry) => {
                    entry.residency == Residency::Evicted
                        && !entry.hints.flags.contains(ResourceFlags::STABLE_ID)
                }
                None => false,
            };
            if !remove {
                continue;
            }

            slot.entry = None;
            slot.generation = Self::next_generation(slot.generation);
            self.free_list.push(idx);
            removed += 1;
        }

        removed
    }

    pub fn asset_id_to_resource(&mut self, kind: ResourceKind, asset_id: u32) -> ResourceId {
        let kind_idx = kind as usize;
        if let Some(slot) = self.asset_maps[kind_idx].get(asset_id) {
            self.ensure_slot_capacity(slot);
            let generation = self.slots[slot].generation;
            return ResourceId::new(kind, slot as u32, generation);
        }

        let slot = self.alloc_slot();
        self.slots[slot].asset_id = Some(asset_id);
        self.slots[slot].alias_of = None;
        let generation = self.slots[slot].generation;
        self.asset_maps[kind_idx].insert(asset_id, slot);
        ResourceId::new(kind, slot as u32, generation)
    }

    pub fn asset_id_from_resource(&self, id: ResourceId) -> Option<u32> {
        let idx = self.resolve_slot_index(id)?;
        self.slots.get(idx)?.asset_id
    }

    /// Insert a fully constructed GPU resource entry.
    /// Used when the renderer actually creates a wgpu::Buffer/Texture/etc.
    pub fn insert_entry(&mut self, mut entry: GpuResourceEntry) {
        let Some(idx) = self.ensure_slot_for_id(entry.id) else {
            return;
        };

        let entry_id = entry.id;
        self.remove_entry(idx);
        self.make_resident(idx, &mut entry);
        let is_resident = entry.residency == Residency::Resident;
        let record_binding_change = is_resident && Self::should_record_binding_change(&entry);
        self.slots[idx].entry = Some(entry);
        if is_resident {
            self.bump_binding_epoch();
        }
        if record_binding_change {
            self.binding_changes.push(entry_id);
        }
    }

    /// Create a logical resource (no actual wgpu allocation yet).
    /// You can call this from the graph when declaring a resource.
    pub fn create_logical(
        &mut self,
        desc: ResourceDesc,
        hints: Option<ResourceUsageHints>,
        frame_index: u32,
        asset_id: Option<u32>,
    ) -> ResourceId {
        let kind = desc.kind();
        let est_size = desc.estimate_size_bytes();
        let mut hints = hints.unwrap_or(ResourceUsageHints {
            flags: ResourceFlags::empty(),
            estimated_size_bytes: est_size,
        });
        hints.flags |= ResourceFlags::STABLE_ID;

        let id = if let Some(asset) = asset_id {
            self.asset_id_to_resource(kind, asset)
        } else {
            let slot = self.alloc_slot();
            let generation = self.slots[slot].generation;
            ResourceId::new(kind, slot as u32, generation)
        };

        let Some(idx) = self.ensure_slot_for_id(id) else {
            return id;
        };

        self.remove_entry(idx);
        let entry = GpuResourceEntry::new(id, kind, est_size, hints, frame_index, desc);
        self.slots[idx].entry = Some(entry);

        id
    }

    /// Mark a resource as used this frame (for idle eviction & LRU).
    pub fn mark_used(&mut self, id: ResourceId, frame_index: u32) {
        let Some(idx) = self.resolve_slot_index(id) else {
            return;
        };
        if let Some(entry) = self.slots[idx].entry.as_mut() {
            entry.last_used_frame = frame_index;
            if entry.residency == Residency::Resident {
                self.lru.touch(idx);
            }
        }
    }

    /// Get immutable view to an entry.
    pub fn entry(&self, id: ResourceId) -> Option<&GpuResourceEntry> {
        let idx = self.resolve_slot_index(id)?;
        self.slots.get(idx)?.entry.as_ref()
    }

    /// Get mutable access to an entry.
    pub fn entry_mut(&mut self, id: ResourceId) -> Option<&mut GpuResourceEntry> {
        let idx = self.resolve_slot_index(id)?;
        self.slots.get_mut(idx)?.entry.as_mut()
    }

    /// Evict a resource immediately.
    pub fn evict(&mut self, id: ResourceId) {
        if let Some(idx) = self.resolve_slot_index(id) {
            self.evict_index(idx, false);
        }
    }

    /// Allocate or reuse a transient resource via the transient heap.
    pub fn allocate_transient(&mut self, desc: ResourceDesc, frame_index: u32) -> ResourceId {
        let kind = desc.kind();
        let est_size = desc.estimate_size_bytes();

        while let Some(reuse_id) = self.transient_heap.acquire(&desc) {
            if let Some(idx) = self.slot_index(reuse_id) {
                if let Some(mut entry) = self.slots[idx].entry.take() {
                    entry.last_used_frame = frame_index;
                    if entry.residency != Residency::Resident {
                        self.make_resident(idx, &mut entry);
                    } else {
                        self.lru.touch(idx);
                    }
                    self.slots[idx].entry = Some(entry);
                    return reuse_id;
                }
            }
        }

        let slot = self.alloc_slot();
        let generation = self.slots[slot].generation;
        let id = ResourceId::new(ResourceKind::Transient, slot as u32, generation);
        let hints = ResourceUsageHints {
            flags: ResourceFlags::TRANSIENT,
            estimated_size_bytes: est_size,
        };
        let mut entry = GpuResourceEntry::new(id, kind, est_size, hints, frame_index, desc);
        self.make_resident(slot, &mut entry);
        self.slots[slot].entry = Some(entry);
        id
    }

    /// Called when a transient resource is not needed anymore *this frame*.
    /// This will make it eligible for reuse in future frames.
    pub fn release_transient(&mut self, desc: &ResourceDesc, id: ResourceId) {
        if self.slot_index(id).is_some() {
            self.transient_heap.release(desc, id);
        }
    }

    pub fn compact_transients(&mut self, shrink: bool) {
        self.transient_heap.compact(shrink);
    }

    pub fn apply_transient_aliases(&mut self, aliases: &[(ResourceId, ResourceId)]) {
        self.clear_transient_aliases();
        for (alias, root) in aliases {
            self.set_alias(*alias, *root);
        }
    }

    pub fn clear_transient_aliases(&mut self) {
        for slot in &mut self.slots {
            slot.alias_of = None;
        }
    }

    fn set_alias(&mut self, alias: ResourceId, root: ResourceId) {
        if alias.raw() == root.raw() {
            return;
        }
        let alias_idx = alias.index();
        let root_idx = root.index();
        let max_idx = alias_idx.max(root_idx);
        self.ensure_slot_capacity(max_idx);

        if self.slots[alias_idx].asset_id.is_some() {
            return;
        }
        if self.slots[alias_idx].is_free() {
            self.slots[alias_idx].generation = alias.generation();
        }

        let should_evict = self.slots[alias_idx]
            .entry
            .as_ref()
            .is_some_and(|entry| entry.residency == Residency::Resident);
        if should_evict {
            self.evict_index(alias_idx, false);
        }
        self.slots[alias_idx].entry = None;
        self.slots[alias_idx].alias_of = Some(root);

        let root_slot = &mut self.slots[root_idx];
        if root_slot.is_free() {
            root_slot.generation = root.generation();
        }
    }

    /// Global VRAM budget object (for tuning).
    pub fn vram_budget(&self) -> &VramBudget {
        &self.vram
    }

    pub fn vram_budget_mut(&mut self) -> &mut VramBudget {
        &mut self.vram
    }

    pub fn binding_epoch(&self) -> u64 {
        self.binding_epoch
    }

    pub fn binding_version(&self, id: ResourceId) -> u64 {
        self.resolve_slot_index(id)
            .and_then(|idx| self.slots.get(idx))
            .map(|slot| slot.binding_version)
            .unwrap_or(0)
    }

    pub fn drain_evictions(&mut self) -> Vec<ResourceId> {
        std::mem::take(&mut self.pending_evictions)
    }

    pub fn drain_binding_changes(&mut self) -> Vec<ResourceId> {
        let mut changes = std::mem::take(&mut self.binding_changes);
        if changes.len() > 1 {
            changes.sort_unstable();
            changes.dedup();
        }
        changes
    }

    fn bump_binding_epoch(&mut self) {
        self.binding_epoch = self.binding_epoch.wrapping_add(1);
    }

    pub fn set_budget(&mut self, soft: u64, hard: u64) {
        self.vram.set_global_limits(soft, hard);
        for class in self.vram.per_kind.iter_mut() {
            class.soft_limit_bytes = soft;
            class.hard_limit_bytes = hard.max(soft);
        }
    }

    pub fn set_per_kind_budgets(&mut self, soft: &[u64], hard: &[u64]) {
        let count = self.vram.per_kind.len();
        let len = soft.len().min(hard.len()).min(count);
        for idx in 0..len {
            self.vram.per_kind[idx].soft_limit_bytes = soft[idx];
            self.vram.per_kind[idx].hard_limit_bytes = hard[idx].max(soft[idx]);
        }
    }

    pub fn set_streaming_eviction_limits(
        &mut self,
        min_residency_frames: u32,
        max_evictions_per_tick: usize,
        eviction_scan_budget: usize,
    ) {
        self.streaming_min_residency_frames = min_residency_frames;
        self.max_evictions_per_tick = max_evictions_per_tick;
        self.eviction_scan_budget = eviction_scan_budget.max(max_evictions_per_tick);
    }

    pub fn evict_all(&mut self) {
        let had_resident = self.resident_count > 0;
        let soft = self.vram.global.soft_limit_bytes;
        let hard = self.vram.global.hard_limit_bytes;
        for slot in &mut self.slots {
            if let Some(entry) = slot.entry.as_mut() {
                if entry.residency == Residency::Resident
                    && Self::should_record_binding_change(entry)
                {
                    slot.binding_version = slot.binding_version.wrapping_add(1);
                    self.binding_changes.push(entry.id);
                }
                entry.residency = Residency::Evicted;
                entry.buffer = None;
                entry.texture = None;
                entry.texture_view = None;
                entry.sampler = None;
            }
        }
        for idx in 0..self.slots.len() {
            self.lru.remove(idx);
        }
        self.clear_transient_aliases();
        let config = self.transient_heap.config();
        self.transient_heap = TransientHeap::new(config);
        self.vram = VramBudget::new(soft, hard);
        self.resident_count = 0;
        self.purge_cursor = 0;
        if had_resident {
            self.bump_binding_epoch();
        }
    }

    /// Budget + idle-based eviction.
    /// "frame_index" is current frame; "target_bytes" is how much to free.
    pub fn evict_budget_based(
        &mut self,
        frame_index: u32,
        target_bytes: u64,
        kind_filter: Option<ResourceKind>,
    ) -> Vec<ResourceId> {
        self.evict_budget_based_internal(frame_index, target_bytes, kind_filter, false)
    }

    fn evict_budget_based_internal(
        &mut self,
        frame_index: u32,
        mut target_bytes: u64,
        kind_filter: Option<ResourceKind>,
        record_evictions: bool,
    ) -> Vec<ResourceId> {
        if target_bytes == 0 {
            return Vec::new();
        }
        if self.max_evictions_per_tick == 0 {
            return Vec::new();
        }
        if self.eviction_scan_budget == 0 {
            return Vec::new();
        }

        let mut evicted = Vec::new();
        let mut evicted_count = 0usize;
        let indices: Vec<usize> = self
            .lru
            .iter_tail()
            .take(self.eviction_scan_budget)
            .collect();

        for idx in indices {
            let Some(entry) = self.slots[idx].entry.as_ref() else {
                continue;
            };
            if entry.residency != Residency::Resident {
                continue;
            }
            if entry.is_pinned() {
                continue;
            }
            if entry.is_streaming()
                && frame_index.saturating_sub(entry.created_frame)
                    < self.streaming_min_residency_frames
            {
                continue;
            }
            if let Some(kind) = kind_filter {
                if entry.kind != kind {
                    continue;
                }
            }
            if entry.last_used_frame == frame_index {
                continue;
            }
            if !entry.is_streaming()
                && entry.idle_frames(frame_index) < self.idle_frames_before_evict
            {
                continue;
            }

            let id = entry.id;
            let size = entry.desc_size_bytes;
            self.evict_index(idx, record_evictions);
            evicted.push(id);
            target_bytes = target_bytes.saturating_sub(size);
            evicted_count += 1;
            if target_bytes == 0 || evicted_count >= self.max_evictions_per_tick {
                break;
            }
        }

        if self.eviction_purge_budget > 0 {
            self.purge_evicted(self.eviction_purge_budget);
        }

        evicted
    }

    /// Run periodic eviction pass based on soft budget and idle time.
    pub fn tick_eviction(&mut self, frame_index: u32) -> Vec<ResourceId> {
        let mut evicted = Vec::new();
        let global_over = self
            .vram
            .global
            .current_bytes
            .saturating_sub(self.vram.global.soft_limit_bytes);
        if global_over > 0 {
            evicted.extend(self.evict_budget_based(frame_index, global_over, None));
        }

        for idx in 1..self.vram.per_kind.len() {
            let class = &self.vram.per_kind[idx];
            if class.current_bytes > class.soft_limit_bytes {
                let kind = unsafe { std::mem::transmute::<u8, ResourceKind>(idx as u8) };
                let over = class.current_bytes - class.soft_limit_bytes;
                evicted.extend(self.evict_budget_based(frame_index, over, Some(kind)));
            }
        }

        evicted
    }

    pub fn iter_entries(&self) -> impl Iterator<Item = &GpuResourceEntry> {
        self.slots.iter().filter_map(|slot| slot.entry.as_ref())
    }

    pub fn entries_len(&self) -> usize {
        self.slots.len()
    }

    pub fn entry_by_index(&self, idx: usize) -> Option<&GpuResourceEntry> {
        self.slots.get(idx)?.entry.as_ref()
    }

    pub fn lru_tail_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.lru.iter_tail()
    }

    pub fn resident_count(&self) -> u32 {
        self.resident_count
    }

    pub fn mark_resident(&mut self, id: ResourceId, frame_index: u32) {
        let Some(idx) = self.resolve_slot_index(id) else {
            return;
        };
        if let Some(mut entry) = self.slots[idx].entry.take() {
            entry.last_used_frame = frame_index;
            if entry.residency != Residency::Resident {
                self.make_resident(idx, &mut entry);
            } else {
                self.lru.touch(idx);
            }
            self.slots[idx].entry = Some(entry);
        }
    }

    /// Ensure the logical entries declared by the render-graph resources exist in the pool.
    /// This does not create GPU objects; it only seeds metadata so passes can realize them.
    pub fn sync_logicals(&mut self, resources: &RenderGraphResources, frame_index: u32) {
        for (id, record) in resources.iter() {
            self.ensure_logical(id, record.desc.clone(), record.hints, frame_index);
        }
    }

    pub fn ensure_logical(
        &mut self,
        id: ResourceId,
        desc: ResourceDesc,
        mut hints: ResourceUsageHints,
        frame_index: u32,
    ) {
        let Some(idx) = self.ensure_slot_for_id(id) else {
            return;
        };
        if self.slots[idx].entry.is_none() {
            hints.flags |= ResourceFlags::STABLE_ID;
            let entry = GpuResourceEntry::new(
                id,
                desc.kind(),
                desc.estimate_size_bytes(),
                hints,
                frame_index,
                desc,
            );
            self.slots[idx].entry = Some(entry);
        }
    }

    /// Realize a logical texture entry with an actual GPU texture and view.
    /// This keeps VRAM accounting in sync when render-graph targets are created or resized.
    pub fn realize_texture(
        &mut self,
        id: ResourceId,
        desc: ResourceDesc,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        frame_index: u32,
    ) {
        let resolved_id = self.resolve_id(id).unwrap_or(id);
        let Some(idx) = self.ensure_slot_for_id(resolved_id) else {
            return;
        };

        let mut hints = ResourceUsageHints::default();
        let mut asset_stream_id = None;
        let mut created_frame = frame_index;
        if let Some(existing) = self.slots[idx].entry.as_ref() {
            hints = existing.hints;
            asset_stream_id = existing.asset_stream_id;
            created_frame = existing.created_frame;
        }

        hints.estimated_size_bytes = desc.estimate_size_bytes();
        let mut entry = GpuResourceEntry::new(
            resolved_id,
            desc.kind(),
            hints.estimated_size_bytes,
            hints,
            frame_index,
            desc,
        );
        entry.texture = Some(texture);
        entry.texture_view = Some(view);
        entry.asset_stream_id = asset_stream_id;
        entry.created_frame = created_frame;
        entry.last_used_frame = frame_index;

        self.insert_entry(entry);
        self.mark_used(resolved_id, frame_index);
    }

    fn should_record_binding_change(entry: &GpuResourceEntry) -> bool {
        if entry.flags().contains(ResourceFlags::TRANSIENT) {
            return false;
        }
        matches!(
            entry.kind,
            ResourceKind::Buffer | ResourceKind::Texture | ResourceKind::Sampler
        )
    }
}

impl GpuResourcePool {
    #[inline]
    pub fn texture_view(&self, id: ResourceId) -> Option<&wgpu::TextureView> {
        self.entry(id)?.texture_view.as_ref()
    }

    #[inline]
    pub fn texture(&self, id: ResourceId) -> Option<&wgpu::Texture> {
        self.entry(id)?.texture.as_ref()
    }

    #[inline]
    pub fn buffer(&self, id: ResourceId) -> Option<&wgpu::Buffer> {
        self.entry(id)?.buffer.as_ref()
    }

    #[inline]
    pub fn sampler(&self, id: ResourceId) -> Option<&wgpu::Sampler> {
        self.entry(id)?.sampler.as_ref()
    }
}
