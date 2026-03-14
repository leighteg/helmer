#![allow(dead_code)]

use crate::graphics::graph::definition::resource_id::ResourceKind;

#[derive(Clone, Debug)]
pub struct VramClassBudget {
    pub soft_limit_bytes: u64,
    pub hard_limit_bytes: u64,
    pub current_bytes: u64,
}

impl VramClassBudget {
    pub fn new(soft: u64, hard: u64) -> Self {
        Self {
            soft_limit_bytes: soft,
            hard_limit_bytes: hard,
            current_bytes: 0,
        }
    }

    pub fn can_allocate(&self, size: u64) -> bool {
        self.current_bytes + size <= self.hard_limit_bytes
    }

    pub fn will_exceed_soft(&self, size: u64) -> bool {
        self.current_bytes + size > self.soft_limit_bytes
    }

    pub fn add(&mut self, size: u64) {
        self.current_bytes += size;
    }

    pub fn sub(&mut self, size: u64) {
        self.current_bytes = self.current_bytes.saturating_sub(size);
    }
}

#[derive(Clone, Debug)]
pub struct VramBudget {
    pub global: VramClassBudget,
    pub per_kind: [VramClassBudget; 9], // up to kind = 8
}

impl VramBudget {
    pub fn new(global_soft: u64, global_hard: u64) -> Self {
        let default_class = VramClassBudget::new(global_soft, global_hard);
        Self {
            global: default_class.clone(),
            per_kind: [
                default_class.clone(), // 0 (unused)
                default_class.clone(), // 1: Mesh
                default_class.clone(), // 2: Material
                default_class.clone(), // 3: Texture
                default_class.clone(), // 4: TextureView
                default_class.clone(), // 5: Sampler
                default_class.clone(), // 6: Buffer
                default_class.clone(), // 7: External
                default_class.clone(), // 8: Transient
            ],
        }
    }

    #[inline]
    fn idx(kind: ResourceKind) -> usize {
        kind as u8 as usize
    }

    pub fn can_allocate(&self, kind: ResourceKind, size: u64) -> bool {
        self.global.can_allocate(size) && self.per_kind[Self::idx(kind)].can_allocate(size)
    }

    pub fn will_exceed_soft(&self, kind: ResourceKind, size: u64) -> bool {
        self.global.will_exceed_soft(size) || self.per_kind[Self::idx(kind)].will_exceed_soft(size)
    }

    pub fn add(&mut self, kind: ResourceKind, size: u64) {
        self.global.add(size);
        self.per_kind[Self::idx(kind)].add(size);
    }

    pub fn sub(&mut self, kind: ResourceKind, size: u64) {
        self.global.sub(size);
        self.per_kind[Self::idx(kind)].sub(size);
    }

    pub fn set_global_limits(&mut self, soft: u64, hard: u64) {
        let soft = soft.max(0);
        let hard = hard.max(soft);
        self.global.soft_limit_bytes = soft;
        self.global.hard_limit_bytes = hard;
    }

    pub fn set_per_kind_limits(&mut self, kind: ResourceKind, soft: u64, hard: u64) {
        let soft = soft.max(0);
        let hard = hard.max(soft);
        let class = &mut self.per_kind[Self::idx(kind)];
        class.soft_limit_bytes = soft;
        class.hard_limit_bytes = hard;
    }
}
