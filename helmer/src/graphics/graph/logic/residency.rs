#![allow(dead_code)]

use crate::graphics::graph::definition::{
    resource_desc::ResourceDesc,
    resource_flags::{ResourceFlags, ResourceUsageHints},
    resource_id::{ResourceId, ResourceKind},
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Residency {
    Resident,
    Evicted,
    PendingUpload,
}

#[derive(Debug, Clone)]
pub struct GpuResourceEntry {
    pub id: ResourceId,
    pub kind: ResourceKind,
    pub desc: ResourceDesc,
    pub desc_size_bytes: u64,
    pub hints: ResourceUsageHints,
    pub residency: Residency,
    pub last_used_frame: u32,
    pub created_frame: u32,
    /// Optional - set when we evict but want a quick re-stream
    pub asset_stream_id: Option<u32>,

    // Raw GPU handles - only one of these will be Some for a given entry
    pub buffer: Option<wgpu::Buffer>,
    pub texture: Option<wgpu::Texture>,
    pub texture_view: Option<wgpu::TextureView>,
    pub sampler: Option<wgpu::Sampler>,
}

impl GpuResourceEntry {
    pub fn new(
        id: ResourceId,
        kind: ResourceKind,
        estimated_size_bytes: u64,
        hints: ResourceUsageHints,
        frame_index: u32,
        desc: ResourceDesc,
    ) -> Self {
        Self {
            id,
            kind,
            desc,
            desc_size_bytes: estimated_size_bytes,
            hints,
            residency: Residency::PendingUpload,
            last_used_frame: frame_index,
            created_frame: frame_index,
            asset_stream_id: None,
            buffer: None,
            texture: None,
            texture_view: None,
            sampler: None,
        }
    }

    #[inline]
    pub fn flags(&self) -> ResourceFlags {
        self.hints.flags
    }

    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.flags().contains(ResourceFlags::PINNED)
    }

    #[inline]
    pub fn is_streaming(&self) -> bool {
        self.flags().contains(ResourceFlags::STREAMING)
    }

    pub fn idle_frames(&self, cur_frame: u32) -> u32 {
        cur_frame.saturating_sub(self.last_used_frame)
    }

    #[inline]
    pub fn texture_format(&self) -> Option<wgpu::TextureFormat> {
        match self.desc {
            ResourceDesc::Texture2D { format, .. } => Some(format),
            _ => None,
        }
    }

    #[inline]
    pub fn is_depth_texture(&self) -> bool {
        matches!(
            self.desc,
            ResourceDesc::Texture2D { format, .. } if format.is_depth_stencil_format()
        )
    }
}
