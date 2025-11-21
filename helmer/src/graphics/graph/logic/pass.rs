use crate::graphics::graph::logic::resource_pool::pool::ResourcePool;

pub trait CompiledPassNode {
    /// Called once when the graph is built
    fn initialize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &mut ResourcePool,
    );

    /// Called when the swapchain is resized
    fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_size: winit::dpi::PhysicalSize<u32>,
        resources: &mut ResourcePool,
    );

    /// Called every frame
    fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut ResourcePool,
    );
}