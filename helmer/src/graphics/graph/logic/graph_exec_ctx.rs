#![allow(dead_code)]

use wgpu::CommandEncoder;

use crate::graphics::backend::binding_backend::RenderPassCtx;

/// Context passed to render pass execute().
/// Wraps backend + resource pool + device + encoder.
pub struct RenderGraphExecCtx<'a> {
    pub rpctx: RenderPassCtx<'a>,
    pub encoder: &'a mut CommandEncoder,
}

impl<'a> RenderGraphExecCtx<'a> {
    pub fn device(&self) -> &wgpu::Device {
        self.rpctx.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.rpctx.queue
    }
}
