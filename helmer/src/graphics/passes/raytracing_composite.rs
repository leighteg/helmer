use crate::graphics::{
    graph::{
        definition::{
            render_pass::RenderPass, resource_desc::ResourceDesc, resource_id::ResourceId,
        },
        logic::{
            gpu_resource_pool::GpuResourcePool, graph_context::RenderGraphContext,
            graph_exec_ctx::RenderGraphExecCtx,
        },
    },
    passes::{FrameGlobals, SwapchainFrameInput},
    renderer_common::common::color_load_op,
};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct RayTracingCompositeInputs {
    pub accumulation: ResourceId,
    pub swapchain: ResourceId,
}

#[derive(Clone)]
pub struct RayTracingCompositePass {
    inputs: RayTracingCompositeInputs,
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    bgl0: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    format: Arc<RwLock<wgpu::TextureFormat>>,
}

impl RayTracingCompositePass {
    pub fn new(
        pool: &mut GpuResourcePool,
        accumulation: ResourceId,
        swapchain_format: wgpu::TextureFormat,
    ) -> Self {
        let swapchain = pool.create_logical(ResourceDesc::External, None, 0, None);
        Self {
            inputs: RayTracingCompositeInputs {
                accumulation,
                swapchain,
            },
            pipeline: Arc::new(RwLock::new(None)),
            bgl0: Arc::new(RwLock::new(None)),
            format: Arc::new(RwLock::new(swapchain_format)),
        }
    }

    pub fn inputs(&self) -> RayTracingCompositeInputs {
        self.inputs
    }

    fn ensure_pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let current = *self.format.read();
        if self.pipeline.read().is_some() && current == format {
            return;
        }
        *self.format.write() = format;

        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RayTracingComposite/BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RayTracingComposite/Layout"),
            bind_group_layouts: &[&bgl0],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("raytracing_composite.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RayTracingComposite/Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.bgl0.write() = Some(bgl0);
    }
}

impl RenderPass for RayTracingCompositePass {
    fn name(&self) -> &'static str {
        "RayTracingCompositePass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.inputs.accumulation);
        ctx.write(self.inputs.swapchain);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        let swapchain = match ctx.rpctx.frame_inputs.get::<SwapchainFrameInput>() {
            Some(s) => s,
            None => return,
        };
        let accumulation = match ctx.rpctx.pool.texture_view(self.inputs.accumulation) {
            Some(v) => v.clone(),
            None => return,
        };

        self.ensure_pipeline(ctx.device(), swapchain.format);
        let pipeline = match self.pipeline.read().as_ref() {
            Some(p) => p.clone(),
            None => return,
        };
        let bgl0 = match self.bgl0.read().as_ref() {
            Some(l) => l.clone(),
            None => return,
        };

        let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RayTracingComposite/BG0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accumulation),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&frame.scene_sampler),
                },
            ],
        });

        let dont_care = frame.render_config.use_dont_care_load_ops;
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RayTracingComposite/Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &swapchain.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: color_load_op(wgpu::Color::BLACK, dont_care),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
