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
};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct HiZOutputs {
    pub hiz: ResourceId,
}

#[derive(Clone)]
pub struct HiZPass {
    depth: ResourceId,
    outputs: HiZOutputs,
    init_pipeline: Arc<RwLock<Option<wgpu::ComputePipeline>>>,
    downsample_pipeline: Arc<RwLock<Option<wgpu::ComputePipeline>>>,
    init_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    downsample_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
}

impl HiZPass {
    pub fn new(pool: &mut GpuResourcePool, depth: ResourceId, width: u32, height: u32) -> Self {
        let mip_levels = calc_mip_levels(width, height);
        let (desc, mut hints) = ResourceDesc::Texture2D {
            width,
            height,
            mip_levels,
            layers: 1,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        }
        .with_hints();
        hints.flags |= ResourceFlags::PREFER_RESIDENT;
        let hiz = pool.create_logical(desc, Some(hints), 0, None);

        Self {
            depth,
            outputs: HiZOutputs { hiz },
            init_pipeline: Arc::new(RwLock::new(None)),
            downsample_pipeline: Arc::new(RwLock::new(None)),
            init_bgl: Arc::new(RwLock::new(None)),
            downsample_bgl: Arc::new(RwLock::new(None)),
        }
    }

    pub fn outputs(&self) -> HiZOutputs {
        self.outputs
    }

    fn ensure_texture(&self, ctx: &mut RenderGraphExecCtx, extent: (u32, u32)) -> Option<u32> {
        let mip_levels = calc_mip_levels(extent.0, extent.1);
        let desc = ResourceDesc::Texture2D {
            width: extent.0,
            height: extent.1,
            mip_levels,
            layers: 1,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        };
        let needs_create = match ctx.rpctx.pool.entry(self.outputs.hiz) {
            Some(entry) => {
                let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                    let size = t.size();
                    size.width == extent.0 && size.height == extent.1
                });
                !tex_ok || entry.desc.mip_levels() != mip_levels
            }
            None => true,
        };

        if needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("HiZ/Texture"),
                size: wgpu::Extent3d {
                    width: extent.0,
                    height: extent.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(mip_levels),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            ctx.rpctx.pool.realize_texture(
                self.outputs.hiz,
                desc.clone(),
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            if let Some(entry) = ctx.rpctx.pool.entry_mut(self.outputs.hiz) {
                entry.texture_view = Some(view);
            }
            ctx.rpctx
                .pool
                .mark_resident(self.outputs.hiz, ctx.rpctx.frame_index);
            return Some(mip_levels);
        }

        ctx.rpctx
            .pool
            .mark_resident(self.outputs.hiz, ctx.rpctx.frame_index);
        Some(mip_levels)
    }

    fn ensure_pipelines(&self, device: &wgpu::Device) {
        if self.init_pipeline.read().is_some() && self.downsample_pipeline.read().is_some() {
            return;
        }

        let init_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HiZ/InitBGL"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HiZ/DownsampleBGL"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let init_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HiZ/InitLayout"),
            bind_group_layouts: &[&init_bgl],
            immediate_size: 0,
        });

        let downsample_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HiZ/DownsampleLayout"),
            bind_group_layouts: &[&downsample_bgl],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("hiz.wgsl"));

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HiZ/InitPipeline"),
            layout: Some(&init_layout),
            module: &shader,
            entry_point: Some("init"),
            compilation_options: Default::default(),
            cache: None,
        });

        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("HiZ/DownsamplePipeline"),
                layout: Some(&downsample_layout),
                module: &shader,
                entry_point: Some("downsample"),
                compilation_options: Default::default(),
                cache: None,
            });

        *self.init_pipeline.write() = Some(init_pipeline);
        *self.downsample_pipeline.write() = Some(downsample_pipeline);
        *self.init_bgl.write() = Some(init_bgl);
        *self.downsample_bgl.write() = Some(downsample_bgl);
    }
}

impl RenderPass for HiZPass {
    fn name(&self) -> &'static str {
        "HiZPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.depth);
        ctx.write(self.outputs.hiz);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        if !frame.device_caps.supports_compute_storage_textures() {
            return;
        }

        let depth_tex = match ctx.rpctx.pool.texture_view(self.depth) {
            Some(v) => v.clone(),
            None => return,
        };

        let extent = (
            frame.surface_size.width.max(1),
            frame.surface_size.height.max(1),
        );
        let mip_levels = match self.ensure_texture(ctx, extent) {
            Some(levels) => levels,
            None => return,
        };

        self.ensure_pipelines(ctx.device());

        let hiz_texture = match ctx.rpctx.pool.entry(self.outputs.hiz) {
            Some(entry) => entry.texture.as_ref(),
            None => None,
        };
        let hiz_texture = match hiz_texture {
            Some(t) => t,
            None => return,
        };

        let init_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: 0,
            mip_level_count: Some(1),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let init_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HiZ/InitBG"),
            layout: self.init_bgl.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&init_view),
                },
            ],
        });

        let mut cpass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RenderGraph/HiZInit"),
                timestamp_writes: None,
            });
        cpass.set_pipeline(self.init_pipeline.read().as_ref().unwrap());
        cpass.set_bind_group(0, &init_bg, &[]);
        let dispatch_x = (extent.0 + 7) / 8;
        let dispatch_y = (extent.1 + 7) / 8;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(cpass);

        let mut mip_width = extent.0;
        let mut mip_height = extent.1;
        for mip in 1..mip_levels {
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);

            let src_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            let dst_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });

            let downsample_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HiZ/DownsampleBG"),
                layout: self.downsample_bgl.read().as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dst_view),
                    },
                ],
            });

            let mut down_pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RenderGraph/HiZDownsample"),
                    timestamp_writes: None,
                });
            down_pass.set_pipeline(self.downsample_pipeline.read().as_ref().unwrap());
            down_pass.set_bind_group(0, &downsample_bg, &[]);
            down_pass.dispatch_workgroups((mip_width + 7) / 8, (mip_height + 7) / 8, 1);
        }
    }
}

fn calc_mip_levels(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height).max(1);
    32 - max_dim.leading_zeros()
}

trait ResourceDescMipLevels {
    fn mip_levels(&self) -> u32;
}

impl ResourceDescMipLevels for ResourceDesc {
    fn mip_levels(&self) -> u32 {
        match self {
            ResourceDesc::Texture2D { mip_levels, .. } => *mip_levels,
            _ => 1,
        }
    }
}
