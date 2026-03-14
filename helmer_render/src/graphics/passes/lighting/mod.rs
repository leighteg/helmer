use crate::graphics::{
    common::renderer::{
        CameraUniforms, LightData, ShaderConstants, ShadowUniforms, SkyUniforms, color_load_op,
        transient_usage,
    },
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
};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct LightingOutputs {
    pub lighting: ResourceId,
    pub lighting_diffuse: ResourceId,
}

#[derive(Clone)]
pub struct LightingPass {
    gbuffer: crate::graphics::passes::gbuffer::GBufferOutputs,
    shadow: crate::graphics::passes::shadow::ShadowOutputs,
    outputs: LightingOutputs,
    extent: (u32, u32),
    format: wgpu::TextureFormat,
    use_uniform_lights: bool,
    max_uniform_lights: usize,
    use_array_scattering: bool,
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    gbuf_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    scene_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    atmosphere_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    constants_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    use_transient_textures: bool,
}

impl LightingPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        gbuffer: crate::graphics::passes::gbuffer::GBufferOutputs,
        shadow: crate::graphics::passes::shadow::ShadowOutputs,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
        supports_fragment_storage_buffers: bool,
        use_array_scattering: bool,
    ) -> Self {
        let use_uniform_lights = !supports_fragment_storage_buffers;
        let max_uniform_lights = crate::graphics::common::constants::WEBGL_MAX_LIGHTS;
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            use_transient_textures,
        );
        let (desc, mut hints) = ResourceDesc::Texture2D {
            width,
            height,
            mip_levels: 1,
            layers: 1,
            format,
            usage,
        }
        .with_hints();
        if use_transient_aliasing {
            hints.flags |= ResourceFlags::TRANSIENT;
        }
        let lighting = pool.create_logical(desc.clone(), Some(hints.clone()), 0, None);

        Self {
            gbuffer,
            shadow,
            outputs: LightingOutputs {
                lighting,
                lighting_diffuse: pool.create_logical(desc, Some(hints), 0, None),
            },
            extent: (width, height),
            format,
            use_uniform_lights,
            max_uniform_lights,
            use_array_scattering,
            pipeline: Arc::new(RwLock::new(None)),
            mesh_pipeline: Arc::new(RwLock::new(None)),
            gbuf_bgl: Arc::new(RwLock::new(None)),
            scene_bgl: Arc::new(RwLock::new(None)),
            atmosphere_bgl: Arc::new(RwLock::new(None)),
            constants_bgl: Arc::new(RwLock::new(None)),
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> LightingOutputs {
        self.outputs
    }

    fn ensure_target(
        &self,
        ctx: &mut RenderGraphExecCtx,
        id: ResourceId,
        extent: (u32, u32),
    ) -> Option<wgpu::TextureView> {
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            self.use_transient_textures,
        );
        let resource_desc = ResourceDesc::Texture2D {
            width: extent.0,
            height: extent.1,
            mip_levels: 1,
            layers: 1,
            format: self.format,
            usage,
        };
        let needs_create = match ctx.rpctx.pool.entry(id) {
            Some(entry) => {
                let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                    let size = t.size();
                    size.width == extent.0 && size.height == extent.1
                });
                !tex_ok
            }
            None => true,
        };
        let view = if needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("Lighting"),
                size: wgpu::Extent3d {
                    width: extent.0,
                    height: extent.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
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
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        ..Default::default()
                    })
                })
        };

        if let Some(entry) = ctx.rpctx.pool.entry_mut(id) {
            if let Some(ref v) = view {
                entry.texture_view = Some(v.clone());
            }
        }
        ctx.rpctx.pool.mark_resident(id, ctx.rpctx.frame_index);
        view
    }

    fn ensure_layouts(
        &self,
        device: &wgpu::Device,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        if let (Some(gbuf), Some(scene), Some(atmosphere), Some(constants)) = (
            self.gbuf_bgl.read().clone(),
            self.scene_bgl.read().clone(),
            self.atmosphere_bgl.read().clone(),
            self.constants_bgl.read().clone(),
        ) {
            return (gbuf, scene, atmosphere, constants);
        }

        let gbuf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting/GBufferBGL"),
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
            ],
        });

        let lights_binding = if self.use_uniform_lights {
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        (std::mem::size_of::<LightData>() * self.max_uniform_lights) as u64,
                    ),
                },
                count: None,
            }
        } else {
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let scene_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<CameraUniforms>() as u64
                    ),
                },
                count: None,
            },
            lights_binding,
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
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
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ShadowUniforms>() as u64
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<SkyUniforms>() as u64
                    ),
                },
                count: None,
            },
        ];
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting/SceneBGL"),
            entries: &scene_entries,
        });

        let scattering_view_dimension = if self.use_array_scattering {
            wgpu::TextureViewDimension::D2Array
        } else {
            wgpu::TextureViewDimension::D3
        };

        let atmosphere_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting/AtmosphereBGL"),
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
                        view_dimension: scattering_view_dimension,
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
            label: Some("Lighting/ConstantsBGL"),
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

        *self.gbuf_bgl.write() = Some(gbuf_bgl.clone());
        *self.scene_bgl.write() = Some(scene_bgl.clone());
        *self.atmosphere_bgl.write() = Some(atmosphere_bgl.clone());
        *self.constants_bgl.write() = Some(constants_bgl.clone());

        (gbuf_bgl, scene_bgl, atmosphere_bgl, constants_bgl)
    }

    fn ensure_pipeline(&self, device: &wgpu::Device) {
        if self.pipeline.read().is_some() {
            return;
        }
        let (gbuf_bgl, scene_bgl, atmosphere_bgl, constants_bgl) = self.ensure_layouts(device);

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lighting/PipelineLayout"),
            bind_group_layouts: &[&gbuf_bgl, &scene_bgl, &atmosphere_bgl, &constants_bgl],
            immediate_size: 0,
        });

        let shader = if self.use_uniform_lights {
            if self.use_array_scattering {
                device.create_shader_module(wgpu::include_wgsl!("lighting_webgl_array.wgsl"))
            } else {
                device.create_shader_module(wgpu::include_wgsl!("lighting_webgl.wgsl"))
            }
        } else if self.use_array_scattering {
            device.create_shader_module(wgpu::include_wgsl!("lighting_array.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("lighting.wgsl"))
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lighting/Pipeline"),
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
                targets: &[Some(self.format.into()), Some(self.format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
    }

    fn ensure_mesh_pipeline(&self, device: &wgpu::Device) {
        if self.mesh_pipeline.read().is_some() {
            return;
        }
        let (gbuf_bgl, scene_bgl, atmosphere_bgl, constants_bgl) = self.ensure_layouts(device);

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lighting/MeshPipelineLayout"),
            bind_group_layouts: &[&gbuf_bgl, &scene_bgl, &atmosphere_bgl, &constants_bgl],
            immediate_size: 0,
        });

        let shader = if self.use_uniform_lights {
            if self.use_array_scattering {
                device.create_shader_module(wgpu::include_wgsl!("lighting_mesh_webgl_array.wgsl"))
            } else {
                device.create_shader_module(wgpu::include_wgsl!("lighting_mesh_webgl.wgsl"))
            }
        } else if self.use_array_scattering {
            device.create_shader_module(wgpu::include_wgsl!("lighting_mesh_array.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("lighting_mesh.wgsl"))
        };

        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("Lighting/MeshPipeline"),
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
                targets: &[Some(self.format.into()), Some(self.format.into())],
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

impl RenderPass for LightingPass {
    fn name(&self) -> &'static str {
        "LightingPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.gbuffer.normal);
        ctx.read(self.gbuffer.albedo);
        ctx.read(self.gbuffer.mra);
        ctx.read(self.gbuffer.emission);
        ctx.read(self.gbuffer.depth_copy);
        ctx.read(self.shadow.map);
        ctx.write(self.outputs.lighting);
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

        let depth_tex_size = ctx
            .rpctx
            .pool
            .entry(self.gbuffer.depth_copy)
            .and_then(|e| e.texture.as_ref())
            .map(|t| t.size());
        let desired_extent = depth_tex_size
            .map(|s| (s.width, s.height))
            .unwrap_or(self.extent);

        let lighting_view = match self.ensure_target(ctx, self.outputs.lighting, desired_extent) {
            Some(v) => v,
            None => return,
        };
        let lighting_diffuse_view =
            match self.ensure_target(ctx, self.outputs.lighting_diffuse, desired_extent) {
                Some(v) => v,
                None => return,
            };
        let disable_lighting = !frame.render_config.direct_lighting_pass;

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
        let mra_view = match ctx.rpctx.pool.texture_view(self.gbuffer.mra) {
            Some(v) => v.clone(),
            None => return,
        };
        let emission_view = match ctx.rpctx.pool.texture_view(self.gbuffer.emission) {
            Some(v) => v.clone(),
            None => return,
        };
        let shadow_view = match ctx.rpctx.pool.texture_view(self.shadow.map) {
            Some(v) => v.clone(),
            None => return,
        };
        let shadow_uniforms = match frame.shadow_uniforms_buffer.as_ref() {
            Some(buf) => buf,
            None => return,
        };

        let lights_buffer = frame.lights_buffer.clone().unwrap_or_else(|| {
            let size = if self.use_uniform_lights {
                (std::mem::size_of::<crate::graphics::common::renderer::LightData>()
                    * self.max_uniform_lights) as u64
            } else {
                std::mem::size_of::<crate::graphics::common::renderer::LightData>() as u64
            };
            let usage = if self.use_uniform_lights {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE
            };
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Lighting/EmptyLights"),
                size,
                usage,
                mapped_at_creation: false,
            })
        });

        let gbuf_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting/GBufferBG"),
            layout: self.gbuf_bgl.read().as_ref().unwrap(),
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
                    resource: wgpu::BindingResource::TextureView(&mra_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&emission_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                },
            ],
        });

        let scene_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting/SceneBG"),
            layout: self.scene_bgl.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&frame.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shadow_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: frame.sky_buffer.as_entire_binding(),
                },
            ],
        });

        let constants_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting/ConstantsBG"),
            layout: self.constants_bgl.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.render_constants_buffer.as_entire_binding(),
            }],
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/Lighting"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &lighting_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load_op(wgpu::Color::BLACK, dont_care),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &lighting_diffuse_view,
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

        if disable_lighting {
            return;
        }

        if use_mesh {
            pass.set_pipeline(self.mesh_pipeline.read().as_ref().unwrap());
        } else {
            pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
        }
        pass.set_bind_group(0, &gbuf_bg, &[]);
        pass.set_bind_group(1, &scene_bg, &[]);
        pass.set_bind_group(2, &frame.atmosphere_bind_group, &[]);
        pass.set_bind_group(3, &constants_bg, &[]);
        if use_mesh {
            pass.draw_mesh_tasks(1, 1, 1);
        } else {
            pass.draw(0..3, 0..1);
        }
    }
}
