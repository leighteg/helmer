#![allow(dead_code)]

use super::resource_flags::ResourceUsageHints;
use crate::graphics::graph::definition::resource_id::ResourceKind;

#[derive(Clone, Debug, PartialEq)]
pub enum ResourceDesc {
    Buffer {
        size: u64,
        usage: wgpu::BufferUsages,
    },
    Texture2D {
        width: u32,
        height: u32,
        mip_levels: u32,
        layers: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    },
    Sampler {
        desc: wgpu::SamplerDescriptor<'static>,
    },
    External, // swapchain, etc.
}

impl ResourceDesc {
    pub fn kind(&self) -> ResourceKind {
        match self {
            ResourceDesc::Buffer { .. } => ResourceKind::Buffer,
            ResourceDesc::Texture2D { .. } => ResourceKind::Texture,
            ResourceDesc::Sampler { .. } => ResourceKind::Sampler,
            ResourceDesc::External => ResourceKind::External,
        }
    }

    pub fn estimate_size_bytes(&self) -> u64 {
        match self {
            ResourceDesc::Buffer { size, .. } => *size,
            ResourceDesc::Texture2D {
                width,
                height,
                mip_levels,
                layers,
                format,
                ..
            } => {
                let (block_w, block_h) = format.block_dimensions();
                let block_size = format.block_copy_size(None).unwrap_or(4) as u64;
                let layer_count = (*layers).max(1) as u64;
                let mut total = 0u64;
                let mut w = *width as u64;
                let mut h = *height as u64;
                for _ in 0..*mip_levels {
                    let blocks_w = (w.max(1) + block_w as u64 - 1) / block_w as u64;
                    let blocks_h = (h.max(1) + block_h as u64 - 1) / block_h as u64;
                    total += blocks_w * blocks_h * block_size * layer_count;
                    w >>= 1;
                    h >>= 1;
                }
                total
            }
            ResourceDesc::Sampler { .. } => 0,
            ResourceDesc::External => 0,
        }
    }

    pub fn with_hints(self) -> (Self, ResourceUsageHints) {
        let est = self.estimate_size_bytes();
        let hints = ResourceUsageHints {
            flags: Default::default(),
            estimated_size_bytes: est,
        };
        (self, hints)
    }

    pub fn fast_hash(&self) -> u64 {
        use hashbrown::DefaultHashBuilder;
        use std::hash::{BuildHasher, Hash, Hasher};

        fn hash_enum<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHashBuilder::default().build_hasher();
            value.hash(&mut hasher);
            hasher.finish()
        }

        let mut hash = 0u64;
        match self {
            ResourceDesc::Buffer { size, usage } => {
                hash = hash.wrapping_add(*size);
                hash = hash.rotate_left(13) ^ usage.bits() as u64;
            }
            ResourceDesc::Texture2D {
                width,
                height,
                mip_levels,
                layers,
                format,
                usage,
            } => {
                hash = hash
                    .wrapping_add(*width as u64)
                    .wrapping_add((*height as u64) << 16)
                    .wrapping_add((*mip_levels as u64) << 32)
                    .wrapping_add((*layers as u64) << 40);
                hash ^= hash_enum(format).rotate_left(9);
                hash ^= (usage.bits() as u64).rotate_left(17);
            }
            ResourceDesc::Sampler { desc: sampler } => {
                hash ^= hash_enum(&sampler.address_mode_u).rotate_left(5);
                hash ^= hash_enum(&sampler.address_mode_v).rotate_left(11);
                hash ^= hash_enum(&sampler.address_mode_w).rotate_left(17);
                hash ^= hash_enum(&sampler.mag_filter).rotate_left(23);
                hash ^= hash_enum(&sampler.min_filter).rotate_left(29);
                hash ^= hash_enum(&sampler.mipmap_filter).rotate_left(37);
                hash ^= sampler.lod_min_clamp.to_bits() as u64;
                hash ^= sampler.lod_max_clamp.to_bits().rotate_left(19) as u64;
                hash ^= sampler
                    .compare
                    .map(|v| hash_enum(&v))
                    .unwrap_or(0)
                    .rotate_left(41);
                hash ^= (sampler.anisotropy_clamp as u64).rotate_left(7);
                hash ^= sampler
                    .border_color
                    .map(|c| hash_enum(&c))
                    .unwrap_or(0)
                    .rotate_left(3);
            }
            ResourceDesc::External => {
                hash = hash.rotate_left(1) ^ 0x1;
            }
        }

        hash ^ (hash >> 29) ^ (hash << 17)
    }
}
