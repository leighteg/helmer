use parking_lot::RwLock;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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
use crate::graphics::passes::{FrameGlobals, gbuffer::GBufferOutputs};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RtReflectionsDenoiseParams {
    radius: u32,
    depth_sigma: f32,
    normal_sigma: f32,
    color_sigma: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RtReflectionsDenoiseOutputs {
    pub reflection: ResourceId,
}

#[derive(Clone)]
pub struct RtReflectionsDenoisePass {
    input: ResourceId,
    gbuffer: GBufferOutputs,
    outputs: RtReflectionsDenoiseOutputs,
    pipeline: Arc<RwLock<Option<wgpu::ComputePipeline>>>,
    bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    use_transient_textures: bool,
}

impl RtReflectionsDenoisePass {
    pub fn new(
        pool: &mut GpuResourcePool,
        input: ResourceId,
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
        let reflection = pool.create_logical(desc, Some(hints), 0, None);

        Self {
            input,
            gbuffer,
            outputs: RtReflectionsDenoiseOutputs { reflection },
            pipeline: Arc::new(RwLock::new(None)),
            bgl: Arc::new(RwLock::new(None)),
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> RtReflectionsDenoiseOutputs {
        self.outputs
    }

    fn ensure_target(
        &self,
        ctx: &mut RenderGraphExecCtx,
        extent: (u32, u32),
    ) -> Option<wgpu::TextureView> {
        let usage = transient_usage(
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            self.use_transient_textures,
        );
        let desc = ResourceDesc::Texture2D {
            width: extent.0.max(1),
            height: extent.1.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
        };
        let needs_create = match ctx.rpctx.pool.entry(self.outputs.reflection) {
            Some(entry) => entry.texture.as_ref().map_or(true, |tex| {
                let size = tex.size();
                size.width != extent.0 || size.height != extent.1
            }),
            None => true,
        };
        let view = if needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("RTReflections/Denoise"),
                size: wgpu::Extent3d {
                    width: extent.0.max(1),
                    height: extent.1.max(1),
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
                .and_then(|entry| entry.texture.as_ref())
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
            label: Some("RTReflections/DenoiseBGL"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            RtReflectionsDenoiseParams,
                        >() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("RTReflections/DenoiseLayout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/rt_reflections_denoise.wgsl"
        ));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RTReflections/DenoisePipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("rt_reflections_denoise"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.bgl.write() = Some(bgl);
    }
}

impl RenderPass for RtReflectionsDenoisePass {
    fn name(&self) -> &'static str {
        "RtReflectionsDenoisePass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.input);
        ctx.read(self.gbuffer.normal);
        ctx.read(self.gbuffer.depth_copy);
        ctx.write(self.outputs.reflection);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        let input_view = match ctx.rpctx.pool.texture_view(self.input) {
            Some(v) => v.clone(),
            None => return,
        };
        let input_tex = match ctx.rpctx.pool.texture(self.input) {
            Some(t) => t,
            None => return,
        };
        let extent = {
            let size = input_tex.size();
            (size.width.max(1), size.height.max(1))
        };
        let output = match self.ensure_target(ctx, extent) {
            Some(v) => v,
            None => return,
        };
        let normal = match ctx.rpctx.pool.texture_view(self.gbuffer.normal) {
            Some(v) => v.clone(),
            None => return,
        };
        let depth = match ctx.rpctx.pool.texture_view(self.gbuffer.depth_copy) {
            Some(v) => v.clone(),
            None => return,
        };

        let mut depth_sigma = frame.render_config.rt_reflection_denoise_depth_sigma;
        let mut normal_sigma = frame.render_config.rt_reflection_denoise_normal_sigma;
        let mut color_sigma = frame.render_config.rt_reflection_denoise_color_sigma;
        if !depth_sigma.is_finite() {
            depth_sigma = 0.0;
        }
        if !normal_sigma.is_finite() {
            normal_sigma = 0.0;
        }
        if !color_sigma.is_finite() {
            color_sigma = 0.0;
        }

        let params = RtReflectionsDenoiseParams {
            radius: frame.render_config.rt_reflection_denoise_radius,
            depth_sigma: depth_sigma.max(0.0),
            normal_sigma: normal_sigma.max(0.0),
            color_sigma: color_sigma.max(0.0),
        };
        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RTReflections/DenoiseParams"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        self.ensure_pipeline(ctx.device());
        let pipeline = match self.pipeline.read().as_ref() {
            Some(p) => p.clone(),
            None => return,
        };
        let bgl = match self.bgl.read().as_ref() {
            Some(l) => l.clone(),
            None => return,
        };

        let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RTReflections/DenoiseBG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normal),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&output),
                },
            ],
        });

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RenderGraph/RT_ReflectionsDenoise"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let group_size = 8;
        let groups_x = (extent.0 + group_size - 1) / group_size;
        let groups_y = (extent.1 + group_size - 1) / group_size;
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
}
