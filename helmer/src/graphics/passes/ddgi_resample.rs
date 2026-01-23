use parking_lot::RwLock;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::graphics::common::renderer::transient_usage;
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
use crate::graphics::passes::{
    FrameGlobals, ddgi_probe_update::DdgiProbeOutputs, gbuffer::GBufferOutputs,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DdgiResampleConstants {
    indirect_scale: f32,
    specular_scale: f32,
    specular_confidence: f32,
    reflection_roughness_start: f32,
    reflection_roughness_end: f32,
    temporal_weight: f32,
    spatial_weight: f32,
    reservoir_mix: f32,
    diffuse_samples: u32,
    specular_samples: u32,
    spatial_samples: u32,
    spatial_radius: u32,
    history_depth_threshold: f32,
    min_candidate_weight: f32,
    specular_cone_angle: f32,
    visibility_normal_bias: f32,
    visibility_spacing_bias: f32,
    visibility_max_bias: f32,
    visibility_receiver_bias: f32,
    visibility_variance_scale: f32,
    visibility_bleed_min: f32,
    visibility_bleed_max: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DdgiReservoir {
    data0: [f32; 4],
    data1: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct DdgiResampleOutputs {
    pub diffuse: ResourceId,
    pub specular: ResourceId,
}

#[derive(Clone)]
pub struct DdgiResamplePass {
    probes: DdgiProbeOutputs,
    gbuffer: GBufferOutputs,
    outputs: DdgiResampleOutputs,
    extent: (u32, u32),
    pipeline: Arc<RwLock<Option<wgpu::ComputePipeline>>>,
    bgl0: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    bgl1: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    bgl2: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    bgl3: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    constants: Arc<RwLock<Option<wgpu::Buffer>>>,
    reservoir_a: Arc<RwLock<Option<wgpu::Buffer>>>,
    reservoir_b: Arc<RwLock<Option<wgpu::Buffer>>>,
    reservoir_bytes: Arc<RwLock<u64>>,
    use_transient_textures: bool,
}

impl DdgiResamplePass {
    pub fn new(
        pool: &mut GpuResourcePool,
        probes: DdgiProbeOutputs,
        gbuffer: GBufferOutputs,
        width: u32,
        height: u32,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
    ) -> Self {
        let usage = transient_usage(
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            use_transient_textures,
        );
        let (desc, mut hints) = ResourceDesc::Texture2D {
            width: width.max(1),
            height: height.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
        }
        .with_hints();
        if use_transient_aliasing {
            hints.flags |= ResourceFlags::TRANSIENT;
        }
        let diffuse = pool.create_logical(desc.clone(), Some(hints), 0, None);
        let specular = pool.create_logical(desc, Some(hints), 0, None);

        Self {
            probes,
            gbuffer,
            outputs: DdgiResampleOutputs { diffuse, specular },
            extent: (width, height),
            pipeline: Arc::new(RwLock::new(None)),
            bgl0: Arc::new(RwLock::new(None)),
            bgl1: Arc::new(RwLock::new(None)),
            bgl2: Arc::new(RwLock::new(None)),
            bgl3: Arc::new(RwLock::new(None)),
            constants: Arc::new(RwLock::new(None)),
            reservoir_a: Arc::new(RwLock::new(None)),
            reservoir_b: Arc::new(RwLock::new(None)),
            reservoir_bytes: Arc::new(RwLock::new(0)),
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> DdgiResampleOutputs {
        self.outputs
    }

    fn ensure_targets(
        &self,
        ctx: &mut RenderGraphExecCtx,
    ) -> Option<(wgpu::TextureView, wgpu::TextureView)> {
        let usage = transient_usage(
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            self.use_transient_textures,
        );
        let desc = ResourceDesc::Texture2D {
            width: self.extent.0.max(1),
            height: self.extent.1.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
        };
        let mut ensure_texture = |id: ResourceId, label: &str| -> Option<wgpu::TextureView> {
            let needs_create = ctx
                .rpctx
                .pool
                .entry(id)
                .map(|e| e.texture.is_none())
                .unwrap_or(true);
            let view = if needs_create {
                let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: self.extent.0.max(1),
                        height: self.extent.1.max(1),
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
                    id,
                    desc.clone(),
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
                    .map(|tex| tex.create_view(&Default::default()))
            };
            if let Some(entry) = ctx.rpctx.pool.entry_mut(id) {
                if let Some(ref v) = view {
                    entry.texture_view = Some(v.clone());
                }
            }
            ctx.rpctx.pool.mark_resident(id, ctx.rpctx.frame_index);
            view
        };

        let diffuse = ensure_texture(self.outputs.diffuse, "DDGI/Diffuse")?;
        let specular = ensure_texture(self.outputs.specular, "DDGI/Specular")?;
        Some((diffuse, specular))
    }

    fn ensure_constants(&self, device: &wgpu::Device) -> wgpu::Buffer {
        if let Some(buf) = self.constants.read().as_ref() {
            return buf.clone();
        }
        let mut guard = self.constants.write();
        if let Some(buf) = guard.as_ref() {
            return buf.clone();
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DDGI/ResampleConstants"),
            size: std::mem::size_of::<DdgiResampleConstants>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *guard = Some(buffer.clone());
        buffer
    }

    fn ensure_reservoirs(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
        let bytes = (self.extent.0.max(1) as u64)
            * (self.extent.1.max(1) as u64)
            * (std::mem::size_of::<DdgiReservoir>() as u64);
        let needs_realloc = {
            let current = *self.reservoir_bytes.read();
            current != bytes
        };
        if needs_realloc {
            let mut bytes_guard = self.reservoir_bytes.write();
            *bytes_guard = bytes;
            let mut a_guard = self.reservoir_a.write();
            let mut b_guard = self.reservoir_b.write();
            *a_guard = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DDGI/ReservoirA"),
                size: bytes.max(1),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            *b_guard = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DDGI/ReservoirB"),
                size: bytes.max(1),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }
        let a = self
            .reservoir_a
            .read()
            .as_ref()
            .expect("reservoir A missing")
            .clone();
        let b = self
            .reservoir_b
            .read()
            .as_ref()
            .expect("reservoir B missing")
            .clone();
        (a, b)
    }

    fn ensure_pipeline(&self, device: &wgpu::Device) {
        if self.pipeline.read().is_some() {
            return;
        }

        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DDGI/ResampleInputsBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DDGI/ResampleUniformsBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            DdgiResampleConstants,
                        >() as u64),
                    },
                    count: None,
                },
            ],
        });

        let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DDGI/ResampleReservoirBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bgl3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DDGI/ResampleOutputsBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DDGI/ResampleLayout"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3],
            immediate_size: 0,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/ddgi_resample.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DDGI/ResamplePipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("ddgi_resample"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.bgl0.write() = Some(bgl0);
        *self.bgl1.write() = Some(bgl1);
        *self.bgl2.write() = Some(bgl2);
        *self.bgl3.write() = Some(bgl3);
    }
}

impl RenderPass for DdgiResamplePass {
    fn name(&self) -> &'static str {
        "DdgiResamplePass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.probes.irradiance_a);
        ctx.read(self.probes.irradiance_b);
        ctx.read(self.probes.distance_a);
        ctx.read(self.probes.distance_b);
        ctx.read(self.gbuffer.normal);
        ctx.read(self.gbuffer.mra);
        ctx.read(self.gbuffer.depth_copy);
        ctx.write(self.outputs.diffuse);
        ctx.write(self.outputs.specular);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };

        let (diffuse_view, specular_view) = match self.ensure_targets(ctx) {
            Some(v) => v,
            None => return,
        };

        let normal = match ctx.rpctx.pool.texture_view(self.gbuffer.normal) {
            Some(v) => v.clone(),
            None => return,
        };
        let mra = match ctx.rpctx.pool.texture_view(self.gbuffer.mra) {
            Some(v) => v.clone(),
            None => return,
        };
        let depth = match ctx.rpctx.pool.texture_view(self.gbuffer.depth_copy) {
            Some(v) => v.clone(),
            None => return,
        };
        let selection = self.probes.select(frame.frame_index);
        let irradiance = match ctx.rpctx.pool.texture_view(selection.irradiance_write) {
            Some(v) => v.clone(),
            None => return,
        };
        let distance = match ctx.rpctx.pool.texture_view(selection.distance_write) {
            Some(v) => v.clone(),
            None => return,
        };

        self.ensure_pipeline(ctx.device());
        let pipeline = match self.pipeline.read().as_ref() {
            Some(p) => p.clone(),
            None => return,
        };
        let bgl0 = match self.bgl0.read().as_ref() {
            Some(l) => l.clone(),
            None => return,
        };
        let bgl1 = match self.bgl1.read().as_ref() {
            Some(l) => l.clone(),
            None => return,
        };
        let bgl2 = match self.bgl2.read().as_ref() {
            Some(l) => l.clone(),
            None => return,
        };
        let bgl3 = match self.bgl3.read().as_ref() {
            Some(l) => l.clone(),
            None => return,
        };

        let ssgi_intensity = frame.render_config.shader_constants.ssgi_intensity;
        let indirect_scale = if ssgi_intensity.abs() > f32::EPSILON {
            frame.render_config.ddgi_intensity / ssgi_intensity
        } else {
            0.0
        };
        let constants = DdgiResampleConstants {
            indirect_scale,
            specular_scale: frame.render_config.ddgi_specular_scale
                * frame.render_config.ddgi_reflection_strength,
            specular_confidence: frame.render_config.ddgi_specular_confidence
                * frame.render_config.ddgi_reflection_strength,
            reflection_roughness_start: frame.render_config.ddgi_reflection_roughness_start,
            reflection_roughness_end: frame.render_config.ddgi_reflection_roughness_end,
            temporal_weight: frame.render_config.ddgi_reservoir_temporal_weight,
            spatial_weight: frame.render_config.ddgi_reservoir_spatial_weight,
            reservoir_mix: frame.render_config.ddgi_resample_reservoir_mix,
            diffuse_samples: frame.render_config.ddgi_resample_diffuse_samples.max(1),
            specular_samples: frame.render_config.ddgi_resample_specular_samples.max(1),
            spatial_samples: frame.render_config.ddgi_resample_spatial_samples,
            spatial_radius: frame.render_config.ddgi_resample_spatial_radius,
            history_depth_threshold: frame.render_config.ddgi_resample_history_depth_threshold,
            min_candidate_weight: frame.render_config.ddgi_resample_min_candidate_weight,
            specular_cone_angle: frame.render_config.ddgi_resample_specular_cone_angle,
            visibility_normal_bias: frame.render_config.ddgi_visibility_normal_bias,
            visibility_spacing_bias: frame.render_config.ddgi_visibility_spacing_bias,
            visibility_max_bias: frame.render_config.ddgi_visibility_max_bias,
            visibility_receiver_bias: frame.render_config.ddgi_visibility_receiver_bias,
            visibility_variance_scale: frame.render_config.ddgi_visibility_variance_scale,
            visibility_bleed_min: frame.render_config.ddgi_visibility_bleed_min,
            visibility_bleed_max: frame.render_config.ddgi_visibility_bleed_max,
            _pad0: 0.0,
        };
        let constants_buffer = self.ensure_constants(ctx.device());
        ctx.queue()
            .write_buffer(&constants_buffer, 0, bytemuck::bytes_of(&constants));

        let (reservoir_a, reservoir_b) = self.ensure_reservoirs(ctx.device());
        let use_a_as_read = (frame.frame_index & 1) == 0;
        let (reservoir_read, reservoir_write) = if use_a_as_read {
            (reservoir_a, reservoir_b)
        } else {
            (reservoir_b, reservoir_a)
        };

        let bg0 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DDGI/ResampleInputsBG"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&normal),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&mra),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&irradiance),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&distance),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&frame.scene_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&frame.point_sampler),
                },
            ],
        });

        let bg1 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DDGI/ResampleUniformsBG"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.ddgi_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: constants_buffer.as_entire_binding(),
                },
            ],
        });

        let bg2 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DDGI/ResampleReservoirBG"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reservoir_read.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: reservoir_write.as_entire_binding(),
                },
            ],
        });

        let bg3 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DDGI/ResampleOutputsBG"),
            layout: &bgl3,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&specular_view),
                },
            ],
        });

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RenderGraph/DDGI_Resample"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.set_bind_group(2, &bg2, &[]);
        pass.set_bind_group(3, &bg3, &[]);

        let group_size = 8;
        let groups_x = (self.extent.0.max(1) + group_size - 1) / group_size;
        let groups_y = (self.extent.1.max(1) + group_size - 1) / group_size;
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
}
