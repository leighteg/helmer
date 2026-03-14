use parking_lot::RwLock;
use std::sync::Arc;

use crate::graphics::common::renderer::{color_load_op, transient_usage};
use crate::graphics::graph::{
    definition::{render_pass::RenderPass, resource_desc::ResourceDesc, resource_id::ResourceId},
    logic::{graph_context::RenderGraphContext, graph_exec_ctx::RenderGraphExecCtx},
};
use crate::graphics::passes::FrameGlobals;

#[derive(Clone, Copy, Debug)]
pub struct DepthCopyOutputs {
    pub depth_copy: ResourceId,
}

#[derive(Clone)]
pub struct DepthCopyPass {
    depth: ResourceId,
    depth_copy: ResourceId,
    extent: (u32, u32),
    format: wgpu::TextureFormat,
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
}

fn depth_copy_output_is_single_channel(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::R32Float | wgpu::TextureFormat::R16Float
    )
}

impl DepthCopyPass {
    pub fn new(
        depth: ResourceId,
        depth_copy: ResourceId,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            depth,
            depth_copy,
            extent: (width, height),
            format,
            pipeline: Arc::new(RwLock::new(None)),
            bgl: Arc::new(RwLock::new(None)),
        }
    }

    pub fn outputs(&self) -> DepthCopyOutputs {
        DepthCopyOutputs {
            depth_copy: self.depth_copy,
        }
    }

    fn ensure_target(
        &self,
        ctx: &mut RenderGraphExecCtx,
        use_transient_textures: bool,
    ) -> Option<wgpu::TextureView> {
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            use_transient_textures,
        );
        let resource_desc = ResourceDesc::Texture2D {
            width: self.extent.0,
            height: self.extent.1,
            mip_levels: 1,
            layers: 1,
            format: self.format,
            usage,
        };
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("DepthCopy/Texture"),
            size: wgpu::Extent3d {
                width: self.extent.0,
                height: self.extent.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage,
            view_formats: &[],
        };

        let needs_create = match ctx.rpctx.pool.entry(self.depth_copy) {
            Some(entry) => {
                let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                    let size = t.size();
                    size.width == self.extent.0 && size.height == self.extent.1
                });
                !tex_ok
            }
            None => true,
        };

        if needs_create {
            let texture = ctx.device().create_texture(&texture_desc);
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            ctx.rpctx.pool.realize_texture(
                self.depth_copy,
                resource_desc.clone(),
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            if let Some(entry) = ctx.rpctx.pool.entry_mut(self.depth_copy) {
                entry.texture_view = Some(view.clone());
            }
            ctx.rpctx
                .pool
                .mark_resident(self.depth_copy, ctx.rpctx.frame_index);
            return Some(view);
        }

        let view = {
            let texture = ctx
                .rpctx
                .pool
                .entry(self.depth_copy)
                .and_then(|e| e.texture.as_ref())?;
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            })
        };
        if let Some(entry) = ctx.rpctx.pool.entry_mut(self.depth_copy) {
            entry.texture_view = Some(view.clone());
        }
        ctx.rpctx
            .pool
            .mark_resident(self.depth_copy, ctx.rpctx.frame_index);

        Some(view)
    }

    fn ensure_pipeline(&self, device: &wgpu::Device) {
        if self.pipeline.read().is_some() {
            return;
        }

        let output_is_single = depth_copy_output_is_single_channel(self.format);
        let entry_point = if output_is_single {
            "fs_main"
        } else {
            "fs_main_rgba"
        };
        let write_mask = if output_is_single {
            wgpu::ColorWrites::RED
        } else {
            wgpu::ColorWrites::ALL
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DepthCopy/BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DepthCopy/PipelineLayout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/depth_copy.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DepthCopy/Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(entry_point),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
                    blend: None,
                    write_mask,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.bgl.write() = Some(bgl);
    }
}

impl RenderPass for DepthCopyPass {
    fn name(&self) -> &'static str {
        "DepthCopyPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.depth);
        ctx.write(self.depth_copy);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        let dont_care = frame.render_config.use_dont_care_load_ops;
        self.ensure_pipeline(ctx.device());

        let depth_view = match ctx.rpctx.pool.texture_view(self.depth) {
            Some(v) => v.clone(),
            None => return,
        };
        let depth_copy_view =
            match self.ensure_target(ctx, frame.render_config.use_transient_textures) {
                Some(v) => v,
                None => return,
            };

        let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DepthCopy/BG"),
            layout: self.bgl.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            }],
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/DepthCopy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &depth_copy_view,
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

        let pipeline = self.pipeline.read();
        let pipeline = match pipeline.as_ref() {
            Some(pipeline) => pipeline,
            None => return,
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
