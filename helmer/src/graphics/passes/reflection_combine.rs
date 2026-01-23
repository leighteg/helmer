use parking_lot::RwLock;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::graphics::common::renderer::{color_load_op, transient_usage};
use crate::graphics::graph::{
    definition::{
        render_pass::RenderPass, resource_desc::ResourceDesc, resource_flags::ResourceFlags,
        resource_id::ResourceId,
    },
    logic::{
        gpu_resource_pool::GpuResourcePool, graph_context::RenderGraphContext,
        graph_exec_ctx::RenderGraphExecCtx,
    },
};
use crate::graphics::passes::FrameGlobals;

#[derive(Clone, Copy, Debug)]
pub struct ReflectionCombineOutputs {
    pub reflection: ResourceId,
}

#[derive(Clone)]
pub struct ReflectionCombinePass {
    primary: ResourceId,
    fallback: ResourceId,
    outputs: ReflectionCombineOutputs,
    extent: (u32, u32),
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    use_transient_textures: bool,
}

impl ReflectionCombinePass {
    pub fn new(
        pool: &mut GpuResourcePool,
        primary: ResourceId,
        fallback: ResourceId,
        width: u32,
        height: u32,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
    ) -> Self {
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            use_transient_textures,
        );
        let (desc, mut hints) = ResourceDesc::Texture2D {
            width,
            height,
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
        }
        .with_hints();
        if use_transient_aliasing {
            hints.flags |= ResourceFlags::TRANSIENT;
        }
        let reflection = pool.create_logical(desc, Some(hints), 0, None);

        Self {
            primary,
            fallback,
            outputs: ReflectionCombineOutputs { reflection },
            extent: (width, height),
            pipeline: Arc::new(RwLock::new(None)),
            bgl: Arc::new(RwLock::new(None)),
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> ReflectionCombineOutputs {
        self.outputs
    }

    fn ensure_target(&self, ctx: &mut RenderGraphExecCtx) -> Option<wgpu::TextureView> {
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            self.use_transient_textures,
        );
        let desc = ResourceDesc::Texture2D {
            width: self.extent.0,
            height: self.extent.1,
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
        };
        let needs_create = ctx
            .rpctx
            .pool
            .entry(self.outputs.reflection)
            .map(|e| e.texture.is_none())
            .unwrap_or(true);
        let view = if needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("Reflections/Combined"),
                size: wgpu::Extent3d {
                    width: self.extent.0,
                    height: self.extent.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            ctx.rpctx.pool.realize_texture(
                self.outputs.reflection,
                desc,
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            Some(view)
        } else {
            ctx.rpctx
                .pool
                .entry(self.outputs.reflection)
                .and_then(|e| e.texture.as_ref())
                .map(|tex| tex.create_view(&Default::default()))
        };

        if let Some(entry) = ctx.rpctx.pool.entry_mut(self.outputs.reflection) {
            if let Some(ref v) = view {
                entry.texture_view = Some(v.clone());
            }
        }
        ctx.rpctx
            .pool
            .mark_resident(self.outputs.reflection, ctx.rpctx.frame_index);
        view
    }

    fn ensure_pipeline(&self, device: &wgpu::Device) {
        if self.pipeline.read().is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reflections/CombineBGL"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Reflections/CombineLayout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/reflection_combine.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Reflections/CombinePipeline"),
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
                targets: &[Some(wgpu::TextureFormat::Rgba16Float.into())],
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

impl RenderPass for ReflectionCombinePass {
    fn name(&self) -> &'static str {
        "ReflectionCombinePass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.primary);
        ctx.read(self.fallback);
        ctx.write(self.outputs.reflection);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };

        let output = match self.ensure_target(ctx) {
            Some(v) => v,
            None => return,
        };
        let primary = match ctx.rpctx.pool.texture_view(self.primary) {
            Some(v) => v.clone(),
            None => return,
        };
        let fallback = match ctx.rpctx.pool.texture_view(self.fallback) {
            Some(v) => v.clone(),
            None => return,
        };

        self.ensure_pipeline(ctx.device());
        let pipeline = match self.pipeline.read().as_ref() {
            Some(p) => p.clone(),
            None => return,
        };
        let bgl = match self.bgl.read().as_ref() {
            Some(l) => l.clone(),
            None => return,
        };

        let params = [
            frame.render_config.ddgi_reflection_fallback_mix,
            0.0,
            0.0,
            0.0,
        ];
        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reflections/CombineParams"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reflections/CombineBG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&primary),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&fallback),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&frame.scene_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let dont_care = frame.render_config.use_dont_care_load_ops;
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/ReflectionCombine"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output,
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
