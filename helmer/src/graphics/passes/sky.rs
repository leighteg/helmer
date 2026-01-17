use crate::graphics::{
    graph::{
        definition::{
            render_pass::RenderPass, resource_desc::ResourceDesc, resource_flags::ResourceFlags,
            resource_id::ResourceId,
        },
        logic::{
            gpu_resource_pool::GpuResourcePool, graph_context::RenderGraphContext,
            graph_exec_ctx::RenderGraphExecCtx,
        },
    },
    passes::FrameGlobals,
    renderer_common::common::{ShaderConstants, color_load_op, transient_usage},
};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct SkyOutputs {
    pub sky: ResourceId,
}

#[derive(Clone)]
pub struct SkyPass {
    depth: ResourceId,
    outputs: SkyOutputs,
    extent: (u32, u32),
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    scene_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    atmosphere_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    constants_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    use_transient_textures: bool,
}

impl SkyPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        depth: ResourceId,
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
        let sky = pool.create_logical(desc, Some(hints), 0, None);

        Self {
            depth,
            outputs: SkyOutputs { sky },
            extent: (width, height),
            pipeline: Arc::new(RwLock::new(None)),
            mesh_pipeline: Arc::new(RwLock::new(None)),
            scene_bgl: Arc::new(RwLock::new(None)),
            atmosphere_bgl: Arc::new(RwLock::new(None)),
            constants_bgl: Arc::new(RwLock::new(None)),
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> SkyOutputs {
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
        let needs_create = match ctx.rpctx.pool.entry(self.outputs.sky) {
            Some(entry) => {
                let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                    let size = t.size();
                    size.width == self.extent.0 && size.height == self.extent.1
                });
                !tex_ok
            }
            None => true,
        };

        let view = if needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("Sky/Target"),
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
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            ctx.rpctx.pool.realize_texture(
                self.outputs.sky,
                resource_desc.clone(),
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            Some(view)
        } else {
            ctx.rpctx
                .pool
                .entry(self.outputs.sky)
                .and_then(|e| e.texture.as_ref())
                .map(|tex| {
                    tex.create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        ..Default::default()
                    })
                })
        };

        if let Some(entry) = ctx.rpctx.pool.entry_mut(self.outputs.sky) {
            if let Some(ref v) = view {
                entry.texture_view = Some(v.clone());
            }
        }
        ctx.rpctx
            .pool
            .mark_resident(self.outputs.sky, ctx.rpctx.frame_index);
        view
    }

    fn ensure_pipeline(&self, device: &wgpu::Device) {
        if self.pipeline.read().is_some() {
            return;
        }

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky/SceneBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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
            ],
        });

        let atmosphere_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky/AtmosphereBGL"),
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
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let constants_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky/ConstantsBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ShaderConstants>() as u64
                    ),
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky/PipelineLayout"),
            bind_group_layouts: &[&scene_bgl, &atmosphere_bgl, &constants_bgl],
            immediate_size: 0,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/sky_sampled.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky/Pipeline"),
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
        *self.scene_bgl.write() = Some(scene_bgl);
        *self.atmosphere_bgl.write() = Some(atmosphere_bgl);
        *self.constants_bgl.write() = Some(constants_bgl);
    }

    fn ensure_mesh_pipeline(&self, device: &wgpu::Device) {
        if self.mesh_pipeline.read().is_some() {
            return;
        }

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky/SceneBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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
            ],
        });

        let atmosphere_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky/AtmosphereBGL"),
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
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let constants_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky/ConstantsBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ShaderConstants>() as u64
                    ),
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky/MeshPipelineLayout"),
            bind_group_layouts: &[&scene_bgl, &atmosphere_bgl, &constants_bgl],
            immediate_size: 0,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/sky_sampled_mesh.wgsl"));

        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("Sky/MeshPipeline"),
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
        *self.scene_bgl.write() = Some(scene_bgl);
        *self.atmosphere_bgl.write() = Some(atmosphere_bgl);
        *self.constants_bgl.write() = Some(constants_bgl);
    }
}

impl RenderPass for SkyPass {
    fn name(&self) -> &'static str {
        "SkyPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.depth);
        ctx.write(self.outputs.sky);
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
        let sky_view = match self.ensure_target(ctx) {
            Some(v) => v,
            None => return,
        };

        let depth_view = match ctx.rpctx.pool.texture_view(self.depth) {
            Some(v) => v.clone(),
            None => return,
        };

        let scene_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky/SceneBG"),
            layout: self.scene_bgl.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.sky_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&frame.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
            ],
        });

        let constants_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky/ConstantsBG"),
            layout: self.constants_bgl.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.render_constants_buffer.as_entire_binding(),
            }],
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/Sky"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &sky_view,
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

        if frame.render_config.sky_pass {
            if use_mesh {
                pass.set_pipeline(self.mesh_pipeline.read().as_ref().unwrap());
            } else {
                pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
            }
            pass.set_bind_group(0, &scene_bg, &[]);
            pass.set_bind_group(1, &frame.atmosphere_bind_group, &[]);
            pass.set_bind_group(2, &constants_bg, &[]);
            if use_mesh {
                pass.draw_mesh_tasks(1, 1, 1);
            } else {
                pass.draw(0..3, 0..1);
            }
        }
    }
}
