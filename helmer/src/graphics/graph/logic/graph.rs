use crate::graphics::graph::logic::{pass::CompiledPassNode, resource_pool::pool::ResourcePool};

#[derive(Default)]
pub struct CompiledRenderGraph {
    pub passes: Vec<Box<dyn CompiledPassNode>>,
}

impl CompiledRenderGraph {
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_size: winit::dpi::PhysicalSize<u32>,
        resources: &mut ResourcePool,
    ) {
        for pass in self.passes.iter_mut() {
            pass.resize(device, queue, new_size, resources);
        }
    }

    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut ResourcePool,
    ) {
        for pass in self.passes.iter_mut() {
            pass.run(device, queue, encoder, resources);
        }
    }
}
