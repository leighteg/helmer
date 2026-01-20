use crate::graphics::graph::{
    definition::pass::{MaterialHandle, MeshHandle, TargetId},
    logic::{pass::CompiledPassNode, resource_pool::pool::ResourcePool},
};

pub struct ForwardPass {
    mesh_handles: Vec<MeshHandle>,
    material_handles: Vec<MaterialHandle>,
    output_target: TargetId,

    // GPU-side cached state
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl ForwardPass {
    pub fn new(
        mesh_handles: Vec<MeshHandle>,
        material_handles: Vec<MaterialHandle>,
        output_target: TargetId,
    ) -> Self {
        Self {
            mesh_handles,
            material_handles,
            output_target,
            pipeline: None,
            bind_group_layout: None,
        }
    }
}

impl CompiledPassNode for ForwardPass {
    fn initialize(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        resources: &mut ResourcePool,
    ) {
        // Create pipeline & layout
        let shader = device.create_shader_module(wgpu::include_wgsl!("forward.wgsl"));

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward bg layout"),
            entries: &[
                // ...
            ],
        });

        self.bind_group_layout = Some(layout);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward pipeline layout"),
            bind_group_layouts: &[self.bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { /* ... */ },
            fragment: Some(wgpu::FragmentState { /* ... */ }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });

        self.pipeline = Some(pipeline);
    }

    fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_size: winit::dpi::PhysicalSize<u32>,
        resources: &mut ResourcePool,
    ) {
        // If we need to recreate target textures
        // The ResourcePool can recreate or resize attachments
    }

    fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut ResourcePool,
    ) {
        let pipeline = self.pipeline.as_ref().unwrap();

        // Acquire target view (color attachment)
        let target_view = resources.get_texture_view(self.output_target).unwrap();

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Forward Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        rp.set_pipeline(pipeline);

        // Draw meshes
        for mesh_handle in &self.mesh_handles {
            let mesh = resources.get_mesh(*mesh_handle).unwrap();
            let Some(lod) = mesh.lods.first() else {
                continue;
            };

            rp.set_vertex_buffer(0, lod.vertex_buffer.slice(..));
            rp.set_index_buffer(lod.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Set per-material bind groups if needed
            // rp.set_bind_group(0, ...);

            rp.draw_indexed(0..lod.index_count, 0, 0..1);
        }
    }
}
