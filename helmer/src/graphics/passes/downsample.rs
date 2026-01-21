use parking_lot::RwLock;
use std::sync::Arc;

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
use crate::graphics::passes::{FrameGlobals, gbuffer::GBufferOutputs};

#[derive(Clone, Copy, Debug)]
pub struct DownsampleOutputs {
    pub depth: ResourceId,
    pub normal: ResourceId,
    pub albedo: ResourceId,
    pub lighting_diffuse: ResourceId,
}

#[derive(Clone)]
pub struct DownsamplePass {
    gbuffer: GBufferOutputs,
    lighting_diffuse: ResourceId,
    outputs: DownsampleOutputs,
    extent: (u32, u32),
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    use_transient_textures: bool,
}

impl DownsamplePass {
    pub fn new(
        pool: &mut GpuResourcePool,
        gbuffer: GBufferOutputs,
        lighting_diffuse: ResourceId,
        width: u32,
        height: u32,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
    ) -> Self {
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);

        let mut make_target = |format: wgpu::TextureFormat| {
            let usage = transient_usage(
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                use_transient_textures,
            );
            let (desc, mut hints) = ResourceDesc::Texture2D {
                width: half_w,
                height: half_h,
                mip_levels: 1,
                layers: 1,
                format,
                usage,
            }
            .with_hints();
            if use_transient_aliasing {
                hints.flags |= ResourceFlags::TRANSIENT;
            }
            pool.create_logical(desc, Some(hints), 0, None)
        };

        let outputs = DownsampleOutputs {
            depth: make_target(wgpu::TextureFormat::R32Float),
            normal: make_target(wgpu::TextureFormat::Rgba16Float),
            albedo: make_target(wgpu::TextureFormat::Rgba16Float),
            lighting_diffuse: make_target(wgpu::TextureFormat::Rgba16Float),
        };

        Self {
            gbuffer,
            lighting_diffuse,
            outputs,
            extent: (half_w, half_h),
            pipeline: Arc::new(RwLock::new(None)),
            mesh_pipeline: Arc::new(RwLock::new(None)),
            bgl: Arc::new(RwLock::new(None)),
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> DownsampleOutputs {
        self.outputs
    }

    fn ensure_targets(&self, ctx: &mut RenderGraphExecCtx) -> Option<DownsampleOutputs> {
        let mut make_view =
            |id: ResourceId, format: wgpu::TextureFormat| -> Option<wgpu::TextureView> {
                let usage = transient_usage(
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    self.use_transient_textures,
                );
                let resource_desc = ResourceDesc::Texture2D {
                    width: self.extent.0,
                    height: self.extent.1,
                    mip_levels: 1,
                    layers: 1,
                    format,
                    usage,
                };
                let needs_create = ctx
                    .rpctx
                    .pool
                    .entry(id)
                    .map(|e| {
                        e.texture.as_ref().map_or(true, |tex| {
                            let size = tex.size();
                            size.width != self.extent.0 || size.height != self.extent.1
                        })
                    })
                    .unwrap_or(true);

                let view = if needs_create {
                    let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                        label: Some("Downsample/Target"),
                        size: wgpu::Extent3d {
                            width: self.extent.0,
                            height: self.extent.1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    });
                    let view = texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        ..Default::default()
                    });
                    ctx.rpctx.pool.realize_texture(
                        id,
                        resource_desc,
                        texture,
                        view.clone(),
                        ctx.rpctx.frame_index,
                    );
                    Some(view)
                } else {
                    ctx.rpctx
                        .pool
                        .entry(id)
                        .and_then(|e| e.texture.as_ref())
                        .map(|tex| {
                            tex.create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(wgpu::TextureViewDimension::D2),
                                base_mip_level: 0,
                                mip_level_count: Some(1),
                                ..Default::default()
                            })
                        })
                };
                if let Some(v) = view.as_ref() {
                    if let Some(entry) = ctx.rpctx.pool.entry_mut(id) {
                        entry.texture_view = Some(v.clone());
                    }
                    ctx.rpctx.pool.mark_resident(id, ctx.rpctx.frame_index);
                }
                view
            };

        let depth = make_view(self.outputs.depth, wgpu::TextureFormat::R32Float)?;
        let normal = make_view(self.outputs.normal, wgpu::TextureFormat::Rgba16Float)?;
        let albedo = make_view(self.outputs.albedo, wgpu::TextureFormat::Rgba16Float)?;
        let lighting = make_view(
            self.outputs.lighting_diffuse,
            wgpu::TextureFormat::Rgba16Float,
        )?;

        Some(DownsampleOutputs {
            depth: self.outputs.depth,
            normal: self.outputs.normal,
            albedo: self.outputs.albedo,
            lighting_diffuse: self.outputs.lighting_diffuse,
        })
        .map(|_| DownsampleOutputs {
            depth: self.outputs.depth,
            normal: self.outputs.normal,
            albedo: self.outputs.albedo,
            lighting_diffuse: self.outputs.lighting_diffuse,
        })
    }

    fn ensure_pipeline(&self, device: &wgpu::Device) {
        if self.pipeline.read().is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/downsample.wgsl"));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Downsample/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Downsample/PipelineLayout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Downsample/Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.bgl.write() = Some(bgl);
        *self.pipeline.write() = Some(pipeline);
    }

    fn ensure_mesh_pipeline(&self, device: &wgpu::Device) {
        if self.mesh_pipeline.read().is_some() {
            return;
        }

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/downsample_mesh.wgsl"));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Downsample/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Downsample/MeshPipelineLayout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("Downsample/MeshPipeline"),
            layout: Some(&pipeline_layout),
            task: None,
            mesh: wgpu::MeshState {
                module: &shader,
                entry_point: Some("ms_main"),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        *self.bgl.write() = Some(bgl);
        *self.mesh_pipeline.write() = Some(pipeline);
    }
}

impl RenderPass for DownsamplePass {
    fn name(&self) -> &'static str {
        "DownsamplePass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.gbuffer.depth_copy);
        ctx.read(self.gbuffer.normal);
        ctx.read(self.gbuffer.albedo);
        ctx.read(self.lighting_diffuse);
        ctx.write(self.outputs.depth);
        ctx.write(self.outputs.normal);
        ctx.write(self.outputs.albedo);
        ctx.write(self.outputs.lighting_diffuse);
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

        let depth_view = match ctx.rpctx.pool.texture_view(self.gbuffer.depth_copy) {
            Some(v) => v.clone(),
            None => return,
        };
        let normal_view = match ctx.rpctx.pool.texture_view(self.gbuffer.normal) {
            Some(v) => v.clone(),
            None => return,
        };
        let albedo_view = match ctx.rpctx.pool.texture_view(self.gbuffer.albedo) {
            Some(v) => v.clone(),
            None => return,
        };
        let lighting_view = match ctx.rpctx.pool.texture_view(self.lighting_diffuse) {
            Some(v) => v.clone(),
            None => return,
        };

        // ensure outputs exist for current extent
        if self.ensure_targets(ctx).is_none() {
            return;
        }

        let depth_half_view = match ctx.rpctx.pool.texture_view(self.outputs.depth) {
            Some(v) => v.clone(),
            None => return,
        };
        let normal_half_view = match ctx.rpctx.pool.texture_view(self.outputs.normal) {
            Some(v) => v.clone(),
            None => return,
        };
        let albedo_half_view = match ctx.rpctx.pool.texture_view(self.outputs.albedo) {
            Some(v) => v.clone(),
            None => return,
        };
        let lighting_half_view = match ctx.rpctx.pool.texture_view(self.outputs.lighting_diffuse) {
            Some(v) => v.clone(),
            None => return,
        };

        let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Downsample/BG"),
            layout: self.bgl.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lighting_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&frame.point_sampler),
                },
            ],
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/Downsample"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &depth_half_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load_op(wgpu::Color::BLACK, dont_care),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &normal_half_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load_op(wgpu::Color::BLACK, dont_care),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &albedo_half_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load_op(wgpu::Color::BLACK, dont_care),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &lighting_half_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load_op(wgpu::Color::BLACK, dont_care),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
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
        pass.set_bind_group(0, &bg, &[]);
        if use_mesh {
            pass.draw_mesh_tasks(1, 1, 1);
        } else {
            pass.draw(0..3, 0..1);
        }
    }
}
