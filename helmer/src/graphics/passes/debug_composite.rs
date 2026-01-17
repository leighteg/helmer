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
    passes::{DebugCompositeParams, FrameGlobals, SwapchainFrameInput, gbuffer::GBufferOutputs},
    renderer_common::common::{ShaderConstants, color_load_op},
};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct DebugCompositeInputs {
    pub direct_lighting: ResourceId,
    pub ssgi: Option<ResourceId>,
    pub ssr: Option<ResourceId>,
    pub albedo: ResourceId,
    pub emission: ResourceId,
    pub normal: ResourceId,
    pub mra: ResourceId,
    pub depth: ResourceId,
    pub sky: ResourceId,
    pub swapchain: ResourceId,
}

#[derive(Clone)]
pub struct DebugCompositePass {
    inputs: DebugCompositeInputs,
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    bgl0: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    bgl1: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    bgl2: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    bgl3: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    bgl4: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    format: Arc<RwLock<wgpu::TextureFormat>>,
}

impl DebugCompositePass {
    pub fn new(
        pool: &mut GpuResourcePool,
        lighting: ResourceId,
        ssgi: Option<ResourceId>,
        ssr: Option<ResourceId>,
        gbuffer: GBufferOutputs,
        sky: ResourceId,
        swapchain_format: wgpu::TextureFormat,
    ) -> Self {
        let swapchain = pool.create_logical(ResourceDesc::External, None, 0, None);
        Self {
            inputs: DebugCompositeInputs {
                direct_lighting: lighting,
                ssgi,
                ssr,
                albedo: gbuffer.albedo,
                emission: gbuffer.emission,
                normal: gbuffer.normal,
                mra: gbuffer.mra,
                depth: gbuffer.depth_copy,
                sky,
                swapchain,
            },
            pipeline: Arc::new(RwLock::new(None)),
            mesh_pipeline: Arc::new(RwLock::new(None)),
            bgl0: Arc::new(RwLock::new(None)),
            bgl1: Arc::new(RwLock::new(None)),
            bgl2: Arc::new(RwLock::new(None)),
            bgl3: Arc::new(RwLock::new(None)),
            bgl4: Arc::new(RwLock::new(None)),
            format: Arc::new(RwLock::new(swapchain_format)),
        }
    }

    pub fn swapchain_id(&self) -> ResourceId {
        self.inputs.swapchain
    }

    pub fn inputs(&self) -> DebugCompositeInputs {
        self.inputs
    }

    fn ensure_pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let current_format = *self.format.read();
        if self.pipeline.read().is_some() && current_format == format {
            return;
        }

        *self.format.write() = format;

        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DebugComposite/InputsBGL"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
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

        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DebugComposite/IBLBGL"),
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
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DebugComposite/CameraBGL"),
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

        let bgl3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DebugComposite/ConstantsBGL"),
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

        let bgl4 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DebugComposite/DebugBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<DebugCompositeParams>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DebugComposite/PipelineLayout"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3, &bgl4],
            immediate_size: 0,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/composite_debug.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DebugComposite/Pipeline"),
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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.bgl0.write() = Some(bgl0);
        *self.bgl1.write() = Some(bgl1);
        *self.bgl2.write() = Some(bgl2);
        *self.bgl3.write() = Some(bgl3);
        *self.bgl4.write() = Some(bgl4);
    }

    fn ensure_mesh_pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let current_format = *self.format.read();
        if self.mesh_pipeline.read().is_some() && current_format == format {
            return;
        }

        self.ensure_pipeline(device, format);

        let (bgl0, bgl1, bgl2, bgl3, bgl4) = (
            self.bgl0.read(),
            self.bgl1.read(),
            self.bgl2.read(),
            self.bgl3.read(),
            self.bgl4.read(),
        );
        let (bgl0, bgl1, bgl2, bgl3, bgl4) = (
            bgl0.as_ref().unwrap(),
            bgl1.as_ref().unwrap(),
            bgl2.as_ref().unwrap(),
            bgl3.as_ref().unwrap(),
            bgl4.as_ref().unwrap(),
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DebugComposite/MeshPipelineLayout"),
            bind_group_layouts: &[bgl0, bgl1, bgl2, bgl3, bgl4],
            immediate_size: 0,
        });

        let shader = device
            .create_shader_module(wgpu::include_wgsl!("../shaders/composite_debug_mesh.wgsl"));

        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("DebugComposite/MeshPipeline"),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        *self.mesh_pipeline.write() = Some(pipeline);
    }
}

impl RenderPass for DebugCompositePass {
    fn name(&self) -> &'static str {
        "DebugCompositePass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.inputs.direct_lighting);
        if let Some(id) = self.inputs.ssgi {
            ctx.read(id);
        }
        if let Some(id) = self.inputs.ssr {
            ctx.read(id);
        }
        ctx.read(self.inputs.albedo);
        ctx.read(self.inputs.emission);
        ctx.read(self.inputs.normal);
        ctx.read(self.inputs.mra);
        ctx.read(self.inputs.depth);
        ctx.read(self.inputs.sky);
        ctx.write(self.inputs.swapchain);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        let dont_care = frame.render_config.use_dont_care_load_ops;
        let swapchain = match ctx.rpctx.frame_inputs.get::<SwapchainFrameInput>() {
            Some(v) => v,
            None => return,
        };

        let use_mesh =
            frame.render_config.use_mesh_shaders && frame.device_caps.supports_mesh_pipeline();
        if use_mesh {
            self.ensure_mesh_pipeline(ctx.device(), swapchain.format);
        } else {
            self.ensure_pipeline(ctx.device(), swapchain.format);
        }

        let fallback = frame.fallback_view.clone();

        let direct_view = ctx
            .rpctx
            .pool
            .texture_view(self.inputs.direct_lighting)
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        let ssgi_view = self
            .inputs
            .ssgi
            .and_then(|id| ctx.rpctx.pool.texture_view(id).cloned())
            .unwrap_or_else(|| fallback.clone());
        let ssr_view = self
            .inputs
            .ssr
            .and_then(|id| ctx.rpctx.pool.texture_view(id).cloned())
            .unwrap_or_else(|| fallback.clone());
        let albedo_view = ctx
            .rpctx
            .pool
            .texture_view(self.inputs.albedo)
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        let emission_view = ctx
            .rpctx
            .pool
            .texture_view(self.inputs.emission)
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        let normal_view = ctx
            .rpctx
            .pool
            .texture_view(self.inputs.normal)
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        let mra_view = ctx
            .rpctx
            .pool
            .texture_view(self.inputs.mra)
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        let depth_view = ctx
            .rpctx
            .pool
            .texture_view(self.inputs.depth)
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        let sky_view = ctx
            .rpctx
            .pool
            .texture_view(self.inputs.sky)
            .cloned()
            .unwrap_or_else(|| fallback.clone());

        let bg0 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DebugComposite/InputsBG"),
            layout: self.bgl0.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&direct_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ssgi_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ssr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&emission_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&frame.scene_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&mra_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&sky_view),
                },
            ],
        });

        let bg1 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DebugComposite/IBLBG"),
            layout: self.bgl1.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&frame.ibl_brdf_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&frame.ibl_irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&frame.ibl_prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&frame.ibl_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&frame.brdf_lut_sampler),
                },
            ],
        });

        let bg2 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DebugComposite/CameraBG"),
            layout: self.bgl2.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.camera_buffer.as_entire_binding(),
            }],
        });

        let bg3 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DebugComposite/ConstantsBG"),
            layout: self.bgl3.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.render_constants_buffer.as_entire_binding(),
            }],
        });

        let bg4 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DebugComposite/DebugBG"),
            layout: self.bgl4.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.debug_params_buffer.as_entire_binding(),
            }],
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/DebugComposite"),
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

        if use_mesh {
            pass.set_pipeline(self.mesh_pipeline.read().as_ref().unwrap());
        } else {
            pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
        }
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.set_bind_group(2, &bg2, &[]);
        pass.set_bind_group(3, &bg3, &[]);
        pass.set_bind_group(4, &bg4, &[]);
        if use_mesh {
            pass.draw_mesh_tasks(1, 1, 1);
        } else {
            pass.draw(0..3, 0..1);
        }
    }
}
