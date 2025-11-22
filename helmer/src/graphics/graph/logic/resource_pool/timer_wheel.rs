use generational_arena::Index;
use std::time::{Duration, Instant};

use crate::graphics::{
    graph::logic::resource_pool::evictable_pool::EvictablePool, renderer_common::common::Mesh,
    renderers::forward_pmu::MaterialLowEnd,
};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum PoolId {
    Meshes,
    Materials,
    Textures,
    TextureViews,
    Samplers,
    BindGroupLayouts,
    BindGroups,
    Buffers,
}

pub struct TimerWheel {
    buckets: Vec<Vec<(PoolId, Index)>>,
    pub(crate) resolution: Duration,
    timeout: Duration,
    current: usize,
}

impl TimerWheel {
    pub fn new(timeout: Duration, resolution: Duration) -> Self {
        let slots = (timeout.as_millis() / resolution.as_millis()) as usize;

        Self {
            buckets: vec![Vec::new(); slots],
            resolution,
            timeout,
            current: 0,
        }
    }

    pub fn schedule(
        &mut self,
        pool_id: PoolId,
        idx: Index,
        last_used: Instant,
        now: Instant,
        wheel_index: &mut Option<usize>,
    ) {
        // Remove from previous bucket if present
        if let Some(old) = wheel_index.take() {
            self.buckets[old].retain(|&(p, i)| p != pool_id || i != idx);
        }

        let elapsed = now.duration_since(last_used);
        if elapsed >= self.timeout {
            // Already expired, do not schedule
            return;
        }

        let remaining = self.timeout - elapsed;
        let num_ticks =
            ((remaining.as_millis() as u128) / (self.resolution.as_millis() as u128)) as usize;

        let slot = (self.current + num_ticks) % self.buckets.len();

        self.buckets[slot].push((pool_id, idx));
        *wheel_index = Some(slot);
    }

    pub fn tick(
        &mut self,
        now: Instant,
        meshes: &mut EvictablePool<Mesh>,
        materials: &mut EvictablePool<MaterialLowEnd>,
        textures: &mut EvictablePool<wgpu::Texture>,
        texture_views: &mut EvictablePool<wgpu::TextureView>,
        samplers: &mut EvictablePool<wgpu::Sampler>,
        bind_group_layouts: &mut EvictablePool<wgpu::BindGroupLayout>,
        bind_groups: &mut EvictablePool<wgpu::BindGroup>,
        buffers: &mut EvictablePool<wgpu::Buffer>,
    ) {
        let bucket = &mut self.buckets[self.current];

        let mut to_reschedule: Vec<(PoolId, Index, Instant)> = Vec::new();

        for (pool_id, idx) in bucket.drain(..) {
            let raw = idx.into_raw_parts().0;
            match pool_id {
                PoolId::Meshes => {
                    if let Some(res) = meshes.arena.get(idx) {
                        if now.duration_since(res.last_used) >= self.timeout {
                            meshes.remove(idx);
                        } else {
                            to_reschedule.push((pool_id, idx, res.last_used));
                        }
                    }
                    meshes.wheel_index[raw] = None;
                }
                PoolId::Materials => {
                    if let Some(res) = materials.arena.get(idx) {
                        if now.duration_since(res.last_used) >= self.timeout {
                            materials.remove(idx);
                        } else {
                            to_reschedule.push((pool_id, idx, res.last_used));
                        }
                    }
                    materials.wheel_index[raw] = None;
                }
                PoolId::Textures => {
                    if let Some(res) = textures.arena.get(idx) {
                        if now.duration_since(res.last_used) >= self.timeout {
                            textures.remove(idx);
                        } else {
                            to_reschedule.push((pool_id, idx, res.last_used));
                        }
                    }
                    textures.wheel_index[raw] = None;
                }
                PoolId::TextureViews => {
                    if let Some(res) = texture_views.arena.get(idx) {
                        if now.duration_since(res.last_used) >= self.timeout {
                            texture_views.remove(idx);
                        } else {
                            to_reschedule.push((pool_id, idx, res.last_used));
                        }
                    }
                    texture_views.wheel_index[raw] = None;
                }
                PoolId::Samplers => {
                    if let Some(res) = samplers.arena.get(idx) {
                        if now.duration_since(res.last_used) >= self.timeout {
                            samplers.remove(idx);
                        } else {
                            to_reschedule.push((pool_id, idx, res.last_used));
                        }
                    }
                    samplers.wheel_index[raw] = None;
                }
                PoolId::BindGroupLayouts => {
                    if let Some(res) = bind_group_layouts.arena.get(idx) {
                        if now.duration_since(res.last_used) >= self.timeout {
                            bind_group_layouts.remove(idx);
                        } else {
                            to_reschedule.push((pool_id, idx, res.last_used));
                        }
                    }
                    bind_group_layouts.wheel_index[raw] = None;
                }
                PoolId::BindGroups => {
                    if let Some(res) = bind_groups.arena.get(idx) {
                        if now.duration_since(res.last_used) >= self.timeout {
                            bind_groups.remove(idx);
                        } else {
                            to_reschedule.push((pool_id, idx, res.last_used));
                        }
                    }
                    bind_groups.wheel_index[raw] = None;
                }
                PoolId::Buffers => {
                    if let Some(res) = buffers.arena.get(idx) {
                        if now.duration_since(res.last_used) >= self.timeout {
                            buffers.remove(idx);
                        } else {
                            to_reschedule.push((pool_id, idx, res.last_used));
                        }
                    }
                    buffers.wheel_index[raw] = None;
                }
            }
        }

        self.current = (self.current + 1) % self.buckets.len();

        for (pool_id, idx, last_used) in to_reschedule {
            let raw = idx.into_raw_parts().0;
            let wheel_index = match pool_id {
                PoolId::Meshes => &mut meshes.wheel_index[raw],
                PoolId::Materials => &mut materials.wheel_index[raw],
                PoolId::Textures => &mut textures.wheel_index[raw],
                PoolId::TextureViews => &mut texture_views.wheel_index[raw],
                PoolId::Samplers => &mut samplers.wheel_index[raw],
                PoolId::BindGroupLayouts => &mut bind_group_layouts.wheel_index[raw],
                PoolId::BindGroups => &mut bind_groups.wheel_index[raw],
                PoolId::Buffers => &mut buffers.wheel_index[raw],
            };
            self.schedule(pool_id, idx, last_used, now, wheel_index);
        }
    }
}
