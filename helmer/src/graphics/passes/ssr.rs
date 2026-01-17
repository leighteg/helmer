use parking_lot::RwLock;
use std::sync::Arc;

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
use crate::graphics::passes::gbuffer::GBufferOutputs;
use crate::graphics::renderer_common::common::{color_load_op, transient_usage};

#[derive(Clone, Copy, Debug)]
pub struct SsrOutputs {
    pub reflection: ResourceId,
}

#[derive(Clone)]
pub struct SsrPass {
    gbuffer: GBufferOutputs,
    lighting: ResourceId,
    outputs: SsrOutputs,
    extent: (u32, u32),
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    inputs_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    camera_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    constants_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    use_transient_textures: bool,
}

impl SsrPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        gbuffer: GBufferOutputs,
        lighting: ResourceId,
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
            gbuffer,
            lighting,
            outputs: SsrOutputs { reflection },
            extent: (width, height),
            pipeline: Arc::new(RwLock::new(None)),
            mesh_pipeline: Arc::new(RwLock::new(None)),
            inputs_bgl: Arc::new(RwLock::new(None)),
            camera_bgl: Arc::new(RwLock::new(None)),
            constants_bgl: Arc::new(RwLock::new(None)),
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> SsrOutputs {
        self.outputs
    }

    fn ensure_target(&self, ctx: &mut RenderGraphExecCtx) -> Option<wgpu::TextureView> {
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            self.use_transient_textures,
        );
        let resource_desc = ResourceDesc::Texture2D {
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
                label: Some("SSR/Reflection"),
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
                resource_desc,
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

        let inputs_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR/Inputs"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR/Camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let constants_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR/Constants"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR/Layout"),
            bind_group_layouts: &[&inputs_bgl, &camera_bgl, &constants_bgl],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/ssr.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSR/Pipeline"),
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
        *self.inputs_bgl.write() = Some(inputs_bgl);
        *self.camera_bgl.write() = Some(camera_bgl);
        *self.constants_bgl.write() = Some(constants_bgl);
    }

    fn ensure_mesh_pipeline(&self, device: &wgpu::Device) {
        if self.mesh_pipeline.read().is_some() {
            return;
        }

        let inputs_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR/Inputs"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR/Camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let constants_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR/Constants"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR/MeshLayout"),
            bind_group_layouts: &[&inputs_bgl, &camera_bgl, &constants_bgl],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/ssr_mesh.wgsl"));
        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("SSR/MeshPipeline"),
            layout: Some(&layout),
            task: None,
            mesh: wgpu::MeshState {
                module: &shader,
                entry_point: Some("ms_main"),
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
            multiview: None,
            cache: None,
        });

        *self.mesh_pipeline.write() = Some(pipeline);
        *self.inputs_bgl.write() = Some(inputs_bgl);
        *self.camera_bgl.write() = Some(camera_bgl);
        *self.constants_bgl.write() = Some(constants_bgl);
    }
}

impl RenderPass for SsrPass {
    fn name(&self) -> &'static str {
        "SsrPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.gbuffer.normal);
        ctx.read(self.gbuffer.mra);
        ctx.read(self.gbuffer.depth_copy);
        ctx.read(self.lighting);
        ctx.write(self.outputs.reflection);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        let dont_care = frame.render_config.use_dont_care_load_ops;
        let use_mesh =
            frame.render_config.use_mesh_shaders && frame.device_caps.supports_mesh_pipeline();
        if use_mesh {
            self.ensure_mesh_pipeline(ctx.device());
        } else {
            self.ensure_pipeline(ctx.device());
        }

        let Some(normal_view) = ctx.rpctx.pool.texture_view(self.gbuffer.normal).cloned() else {
            return;
        };
        let Some(mra_view) = ctx.rpctx.pool.texture_view(self.gbuffer.mra).cloned() else {
            return;
        };
        let Some(depth_view) = ctx
            .rpctx
            .pool
            .texture_view(self.gbuffer.depth_copy)
            .cloned()
        else {
            return;
        };
        let Some(lighting_view) = ctx.rpctx.pool.texture_view(self.lighting).cloned() else {
            return;
        };
        let Some(output_view) = self.ensure_target(ctx) else {
            return;
        };

        let inputs_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR/InputsBG"),
            layout: self.inputs_bgl.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&mra_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lighting_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&frame.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&frame.scene_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&frame.blue_noise_view),
                },
            ],
        });

        let camera_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR/CameraBG"),
            layout: self.camera_bgl.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.camera_buffer.as_entire_binding(),
            }],
        });

        let constants_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR/ConstantsBG"),
            layout: self.constants_bgl.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.render_constants_buffer.as_entire_binding(),
            }],
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/SSR"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
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

        if use_mesh {
            pass.set_pipeline(self.mesh_pipeline.read().as_ref().unwrap());
        } else {
            pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
        }
        pass.set_bind_group(0, &inputs_bg, &[]);
        pass.set_bind_group(1, &camera_bg, &[]);
        pass.set_bind_group(2, &constants_bg, &[]);
        if use_mesh {
            pass.draw_mesh_tasks(1, 1, 1);
        } else {
            pass.draw(0..3, 0..1);
        }
    }
}
