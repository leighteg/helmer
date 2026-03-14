use parking_lot::RwLock;
use std::sync::Arc;

use crate::graphics::{
    graph::{
        definition::{render_pass::RenderPass, resource_id::ResourceId},
        logic::{
            gpu_resource_pool::GpuResourcePool, graph_context::RenderGraphContext,
            graph_exec_ctx::RenderGraphExecCtx,
        },
    },
    passes::{FrameGlobals, SwapchainFrameInput},
};

#[derive(Clone, Copy, Debug)]
pub struct GizmoOutputs {
    pub swapchain: ResourceId,
}

#[derive(Clone)]
pub struct GizmoPass {
    outputs: GizmoOutputs,
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    camera_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    gizmo_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    format: Arc<RwLock<wgpu::TextureFormat>>,
}

impl GizmoPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        swapchain: ResourceId,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let _ = pool.entry(swapchain);
        Self {
            outputs: GizmoOutputs { swapchain },
            pipeline: Arc::new(RwLock::new(None)),
            camera_bgl: Arc::new(RwLock::new(None)),
            gizmo_bgl: Arc::new(RwLock::new(None)),
            format: Arc::new(RwLock::new(surface_format)),
        }
    }

    pub fn outputs(&self) -> GizmoOutputs {
        self.outputs
    }

    fn ensure_pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let current_format = *self.format.read();
        if self.pipeline.read().is_some() && current_format == format {
            return;
        }

        *self.format.write() = format;

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GizmoPass/CameraBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let gizmo_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GizmoPass/GizmoBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GizmoPass/PipelineLayout"),
            bind_group_layouts: &[&camera_bgl, &gizmo_bgl],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/gizmo.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GizmoPass/Pipeline"),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.camera_bgl.write() = Some(camera_bgl);
        *self.gizmo_bgl.write() = Some(gizmo_bgl);
    }
}

impl RenderPass for GizmoPass {
    fn name(&self) -> &'static str {
        "GizmoPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.rw(self.outputs.swapchain);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        if !frame.render_config.gizmo_pass {
            return;
        }
        if frame.gizmo_vertex_count == 0 {
            return;
        }
        let gizmo_params = match frame.gizmo_params_buffer.as_ref() {
            Some(buf) => buf,
            None => return,
        };
        let gizmo_icons = match frame.gizmo_icon_buffer.as_ref() {
            Some(buf) => buf,
            None => return,
        };
        let gizmo_outline = match frame.gizmo_outline_buffer.as_ref() {
            Some(buf) => buf,
            None => return,
        };
        let swapchain = match ctx.rpctx.frame_inputs.get::<SwapchainFrameInput>() {
            Some(v) => v,
            None => return,
        };

        self.ensure_pipeline(ctx.device(), swapchain.format);

        let camera_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GizmoPass/CameraBG"),
            layout: self.camera_bgl.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.camera_buffer.as_entire_binding(),
            }],
        });

        let gizmo_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GizmoPass/GizmoBG"),
            layout: self.gizmo_bgl.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gizmo_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gizmo_icons.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gizmo_outline.as_entire_binding(),
                },
            ],
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/Gizmo"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &swapchain.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
        pass.set_bind_group(0, &camera_bg, &[]);
        pass.set_bind_group(1, &gizmo_bg, &[]);
        pass.draw(0..frame.gizmo_vertex_count, 0..1);
    }
}
