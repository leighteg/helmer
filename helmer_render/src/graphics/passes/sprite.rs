use hashbrown::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::graphics::{
    common::{
        config::SpriteLightingMode,
        renderer::{LightData, SkyUniforms, SpriteBlendMode},
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
    passes::{FrameGlobals, SpriteInstanceRaw, SwapchainFrameInput},
};

const MAX_SPRITE_LIGHTS: usize = 32;
const SPRITE_FLAG_WORLD_SPACE: u32 = 1u32 << 2;
const SPRITE_FLAG_DEPTH_WRITE: u32 = 1u32 << 3;

#[derive(Clone, Copy, Debug)]
pub struct SpriteOutputs {
    pub swapchain: ResourceId,
    pub pick_map: ResourceId,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpritePassParamsGpu {
    viewport_size: [f32; 2],
    viewport_inv_size: [f32; 2],
    depth_flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpriteLightsGpu {
    lights: [LightData; MAX_SPRITE_LIGHTS],
}

#[derive(Clone)]
struct FallbackDepthCopy {
    width: u32,
    height: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Clone)]
struct WorldDepthSurface {
    width: u32,
    height: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Clone)]
pub struct SpritePass {
    depth_copy: Option<ResourceId>,
    outputs: SpriteOutputs,
    alpha_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    premultiplied_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    additive_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    alpha_depth_write_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    alpha_depth_read_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    premultiplied_depth_read_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    additive_depth_read_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    camera_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    texture_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    params_buffer: Arc<RwLock<Option<wgpu::Buffer>>>,
    lights_buffer: Arc<RwLock<Option<wgpu::Buffer>>>,
    format: Arc<RwLock<wgpu::TextureFormat>>,
    fallback_depth: Arc<RwLock<Option<FallbackDepthCopy>>>,
    world_depth: Arc<RwLock<Option<WorldDepthSurface>>>,
}

impl SpritePass {
    pub fn new(
        pool: &mut GpuResourcePool,
        swapchain: ResourceId,
        depth_copy: Option<ResourceId>,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
    ) -> Self {
        let _ = pool.entry(swapchain);
        if let Some(depth_copy) = depth_copy {
            let _ = pool.entry(depth_copy);
        }

        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let (pick_desc, mut pick_hints) = ResourceDesc::Texture2D {
            width: width.max(1),
            height: height.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::R32Uint,
            usage,
        }
        .with_hints();
        if use_transient_textures {
            pick_hints.flags |= ResourceFlags::TRANSIENT;
        }
        if use_transient_aliasing {
            // ResourceFlags does not expose a dedicated alias bit yet; keep the resource transient.
            pick_hints.flags |= ResourceFlags::TRANSIENT;
        }
        let pick_map = pool.create_logical(pick_desc, Some(pick_hints), 0, None);

        Self {
            depth_copy,
            outputs: SpriteOutputs {
                swapchain,
                pick_map,
            },
            alpha_pipeline: Arc::new(RwLock::new(None)),
            premultiplied_pipeline: Arc::new(RwLock::new(None)),
            additive_pipeline: Arc::new(RwLock::new(None)),
            alpha_depth_write_pipeline: Arc::new(RwLock::new(None)),
            alpha_depth_read_pipeline: Arc::new(RwLock::new(None)),
            premultiplied_depth_read_pipeline: Arc::new(RwLock::new(None)),
            additive_depth_read_pipeline: Arc::new(RwLock::new(None)),
            camera_bgl: Arc::new(RwLock::new(None)),
            texture_bgl: Arc::new(RwLock::new(None)),
            params_buffer: Arc::new(RwLock::new(None)),
            lights_buffer: Arc::new(RwLock::new(None)),
            format: Arc::new(RwLock::new(surface_format)),
            fallback_depth: Arc::new(RwLock::new(None)),
            world_depth: Arc::new(RwLock::new(None)),
        }
    }

    pub fn outputs(&self) -> SpriteOutputs {
        self.outputs
    }

    fn ensure_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let mut guard = self.params_buffer.write();
        if guard.is_none() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SpritePass/Params"),
                size: std::mem::size_of::<SpritePassParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *guard = Some(buffer);
        }
        guard.as_ref().unwrap().clone()
    }

    fn ensure_lights_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let mut guard = self.lights_buffer.write();
        if guard.is_none() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SpritePass/Lights"),
                size: std::mem::size_of::<SpriteLightsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *guard = Some(buffer);
        }
        guard.as_ref().unwrap().clone()
    }

    fn ensure_pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        supports_independent_blend: bool,
    ) {
        if self.alpha_pipeline.read().is_some()
            && self.premultiplied_pipeline.read().is_some()
            && self.additive_pipeline.read().is_some()
            && self.alpha_depth_write_pipeline.read().is_some()
            && self.alpha_depth_read_pipeline.read().is_some()
            && self.premultiplied_depth_read_pipeline.read().is_some()
            && self.additive_depth_read_pipeline.read().is_some()
            && *self.format.read() == format
        {
            return;
        }

        *self.format.write() = format;

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpritePass/CameraBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            SpritePassParamsGpu,
                        >() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<SpriteLightsGpu>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<SkyUniforms>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpritePass/TextureBGL"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SpritePass/PipelineLayout"),
            bind_group_layouts: &[&camera_bgl, &texture_bgl],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/sprite.wgsl"));

        const INSTANCE_ATTRS: [wgpu::VertexAttribute; 8] = wgpu::vertex_attr_array![
            0 => Float32x4,
            1 => Float32x4,
            2 => Float32x4,
            3 => Float32x4,
            4 => Float32x4,
            5 => Float32x4,
            6 => Float32x4,
            7 => Uint32x4,
        ];

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SpriteInstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRS,
        };

        let make_pipeline =
            |label: &str,
             blend: Option<wgpu::BlendState>,
             depth_write_enabled: bool,
             depth_compare: Option<wgpu::CompareFunction>| {
                let color_blend = if supports_independent_blend {
                    blend
                } else {
                    None
                };
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[instance_layout.clone()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format,
                                blend: color_blend,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::R32Uint,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: depth_compare.map(|depth_compare| wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled,
                        depth_compare,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };

        let alpha = make_pipeline(
            "SpritePass/PipelineAlpha",
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
            None,
        );
        let alpha_depth_write = make_pipeline(
            "SpritePass/PipelineAlphaDepthWrite",
            Some(wgpu::BlendState::ALPHA_BLENDING),
            true,
            Some(wgpu::CompareFunction::GreaterEqual),
        );
        let alpha_depth_read = make_pipeline(
            "SpritePass/PipelineAlphaDepthRead",
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
            Some(wgpu::CompareFunction::GreaterEqual),
        );
        let premultiplied = make_pipeline(
            "SpritePass/PipelinePremultiplied",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            false,
            None,
        );
        let premultiplied_depth_read = make_pipeline(
            "SpritePass/PipelinePremultipliedDepthRead",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            false,
            Some(wgpu::CompareFunction::GreaterEqual),
        );
        let additive = make_pipeline(
            "SpritePass/PipelineAdditive",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            false,
            None,
        );
        let additive_depth_read = make_pipeline(
            "SpritePass/PipelineAdditiveDepthRead",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            false,
            Some(wgpu::CompareFunction::GreaterEqual),
        );

        *self.alpha_pipeline.write() = Some(alpha);
        *self.premultiplied_pipeline.write() = Some(premultiplied);
        *self.additive_pipeline.write() = Some(additive);
        *self.alpha_depth_write_pipeline.write() = Some(alpha_depth_write);
        *self.alpha_depth_read_pipeline.write() = Some(alpha_depth_read);
        *self.premultiplied_depth_read_pipeline.write() = Some(premultiplied_depth_read);
        *self.additive_depth_read_pipeline.write() = Some(additive_depth_read);
        *self.camera_bgl.write() = Some(camera_bgl);
        *self.texture_bgl.write() = Some(texture_bgl);
    }

    fn ensure_pick_target(
        &self,
        ctx: &mut RenderGraphExecCtx,
        width: u32,
        height: u32,
    ) -> Option<wgpu::TextureView> {
        let width = width.max(1);
        let height = height.max(1);
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let desc = ResourceDesc::Texture2D {
            width,
            height,
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::R32Uint,
            usage,
        };
        let target_width = width;
        let target_height = height;

        let needs_recreate = ctx
            .rpctx
            .pool
            .entry(self.outputs.pick_map)
            .map(|entry| {
                if entry.texture_view.is_none() {
                    return true;
                }
                match &entry.desc {
                    ResourceDesc::Texture2D {
                        width,
                        height,
                        mip_levels,
                        layers,
                        format,
                        usage: existing_usage,
                    } => {
                        *width != target_width
                            || *height != target_height
                            || *mip_levels != 1
                            || *layers != 1
                            || *format != wgpu::TextureFormat::R32Uint
                            || *existing_usage != usage
                    }
                    _ => true,
                }
            })
            .unwrap_or(true);

        if needs_recreate {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("SpritePass/PickMap"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            ctx.rpctx.pool.realize_texture(
                self.outputs.pick_map,
                desc,
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            return Some(view);
        }

        ctx.rpctx.pool.texture_view(self.outputs.pick_map).cloned()
    }

    fn ensure_fallback_depth_view(
        &self,
        ctx: &mut RenderGraphExecCtx,
        width: u32,
        height: u32,
    ) -> Option<wgpu::TextureView> {
        let width = width.max(1);
        let height = height.max(1);

        {
            let mut guard = self.fallback_depth.write();
            let needs_recreate = guard
                .as_ref()
                .is_none_or(|depth| depth.width != width || depth.height != height);
            if needs_recreate {
                let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("SpritePass/FallbackDepth"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                *guard = Some(FallbackDepthCopy {
                    width,
                    height,
                    _texture: texture,
                    view,
                });
            }
        }

        let view = self
            .fallback_depth
            .read()
            .as_ref()
            .map(|depth| depth.view.clone())?;

        {
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RenderGraph/SpriteFallbackDepthClear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        Some(view)
    }

    fn ensure_world_depth_view(
        &self,
        ctx: &mut RenderGraphExecCtx,
        width: u32,
        height: u32,
    ) -> Option<wgpu::TextureView> {
        let width = width.max(1);
        let height = height.max(1);

        {
            let mut guard = self.world_depth.write();
            let needs_recreate = guard
                .as_ref()
                .is_none_or(|depth| depth.width != width || depth.height != height);
            if needs_recreate {
                let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("SpritePass/WorldDepth"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                *guard = Some(WorldDepthSurface {
                    width,
                    height,
                    _texture: texture,
                    view,
                });
            }
        }

        self.world_depth
            .read()
            .as_ref()
            .map(|depth| depth.view.clone())
    }

    fn scene_depth_view(
        &self,
        ctx: &mut RenderGraphExecCtx,
        width: u32,
        height: u32,
    ) -> Option<(wgpu::TextureView, bool)> {
        if let Some(depth_copy) = self.depth_copy
            && let Some(view) = ctx.rpctx.pool.texture_view(depth_copy).cloned()
        {
            return Some((view, true));
        }

        self.ensure_fallback_depth_view(ctx, width, height)
            .map(|view| (view, false))
    }
}

impl RenderPass for SpritePass {
    fn name(&self) -> &'static str {
        "SpritePass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.rw(self.outputs.swapchain);
        ctx.write(self.outputs.pick_map);
        if let Some(depth_copy) = self.depth_copy {
            ctx.read(depth_copy);
        }
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(frame) => frame,
            None => return,
        };
        if !frame.render_config.sprite_pass {
            return;
        }

        let swapchain = match ctx.rpctx.frame_inputs.get::<SwapchainFrameInput>() {
            Some(s) => s,
            None => return,
        };

        self.ensure_pipeline(
            ctx.device(),
            swapchain.format,
            frame.device_caps.supports_independent_blend(),
        );

        let pick_view = match self.ensure_pick_target(
            ctx,
            swapchain.size_in_pixels[0],
            swapchain.size_in_pixels[1],
        ) {
            Some(view) => view.clone(),
            None => return,
        };
        let (scene_depth_view, has_scene_depth) = match self.scene_depth_view(
            ctx,
            swapchain.size_in_pixels[0],
            swapchain.size_in_pixels[1],
        ) {
            Some(depth) => depth,
            None => return,
        };

        let viewport = [
            swapchain.size_in_pixels[0].max(1) as f32,
            swapchain.size_in_pixels[1].max(1) as f32,
        ];
        let has_world_lit_batches = frame.sprite_batches.iter().any(|batch| {
            batch.instance_range.start < batch.instance_range.end
                && (batch.flags & SPRITE_FLAG_WORLD_SPACE) != 0
                && !matches!(batch.blend_mode, SpriteBlendMode::Additive)
        });
        let light_count = if has_world_lit_batches {
            frame.lights.len().min(MAX_SPRITE_LIGHTS)
        } else {
            0
        };
        let params = SpritePassParamsGpu {
            viewport_size: viewport,
            viewport_inv_size: [1.0 / viewport[0], 1.0 / viewport[1]],
            depth_flags: [
                u32::from(has_scene_depth),
                light_count as u32,
                match frame.render_config.sprite_lighting_mode {
                    SpriteLightingMode::Fragment => 0,
                    SpriteLightingMode::Vertex => 1,
                },
                0,
            ],
        };
        let params_buffer = self.ensure_params_buffer(ctx.device());
        ctx.queue()
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        let lights_buffer = self.ensure_lights_buffer(ctx.device());
        let mut lights = SpriteLightsGpu {
            lights: [LightData {
                position: [0.0; 3],
                light_type: 0,
                color: [0.0; 3],
                intensity: 0.0,
                direction: [0.0; 3],
                _padding: 0.0,
            }; MAX_SPRITE_LIGHTS],
        };
        for (dst, src) in lights
            .lights
            .iter_mut()
            .zip(frame.lights.iter().take(light_count))
        {
            *dst = *src;
        }
        ctx.queue()
            .write_buffer(&lights_buffer, 0, bytemuck::bytes_of(&lights));

        let camera_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpritePass/CameraBG"),
            layout: self.camera_bgl.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: frame.sky_buffer.as_entire_binding(),
                },
            ],
        });

        let texture_layout = self.texture_bgl.read().as_ref().unwrap().clone();
        let mut texture_bind_groups: HashMap<u32, wgpu::BindGroup> = HashMap::new();
        for batch in frame.sprite_batches.iter() {
            if batch.instance_range.start >= batch.instance_range.end {
                continue;
            }
            if texture_bind_groups.contains_key(&batch.texture_slot) {
                continue;
            }
            let texture_slot = batch.texture_slot as usize;
            let view = frame
                .sprite_textures
                .get(texture_slot)
                .unwrap_or(&frame.fallback_view);
            let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SpritePass/TextureBG"),
                layout: &texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&frame.point_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&scene_depth_view),
                    },
                ],
            });
            texture_bind_groups.insert(batch.texture_slot, bind_group);
        }

        let instance_buffer = match frame.sprite_instances.as_ref() {
            Some(buffer) if buffer.count > 0 => buffer,
            _ => {
                ctx.rpctx.mark_used(self.outputs.swapchain);
                ctx.rpctx.mark_used(self.outputs.pick_map);
                return;
            }
        };
        let has_world_opaque_batches = frame.sprite_batches.iter().any(|batch| {
            batch.instance_range.start < batch.instance_range.end
                && (batch.flags & SPRITE_FLAG_WORLD_SPACE) != 0
                && (batch.flags & SPRITE_FLAG_DEPTH_WRITE) != 0
        });
        let has_world_translucent_batches = frame.sprite_batches.iter().any(|batch| {
            batch.instance_range.start < batch.instance_range.end
                && (batch.flags & SPRITE_FLAG_WORLD_SPACE) != 0
                && (batch.flags & SPRITE_FLAG_DEPTH_WRITE) == 0
        });
        let has_screen_batches = frame.sprite_batches.iter().any(|batch| {
            batch.instance_range.start < batch.instance_range.end
                && (batch.flags & SPRITE_FLAG_WORLD_SPACE) == 0
        });
        let world_depth_view = if has_world_opaque_batches || has_world_translucent_batches {
            match self.ensure_world_depth_view(
                ctx,
                swapchain.size_in_pixels[0],
                swapchain.size_in_pixels[1],
            ) {
                Some(view) => Some(view),
                None => return,
            }
        } else {
            None
        };
        let mut pick_initialized = false;
        let mut world_depth_initialized = false;

        if let Some(world_depth_view) = world_depth_view.as_ref() {
            if has_world_opaque_batches {
                let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("RenderGraph/SpriteWorldOpaque"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &swapchain.view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &pick_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: if pick_initialized {
                                    wgpu::LoadOp::Load
                                } else {
                                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                                },
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: world_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: if world_depth_initialized {
                                wgpu::LoadOp::Load
                            } else {
                                wgpu::LoadOp::Clear(0.0)
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_vertex_buffer(0, instance_buffer.buffer.slice(..));
                pass.set_bind_group(0, &camera_bg, &[]);
                for batch in frame.sprite_batches.iter() {
                    if batch.instance_range.start >= batch.instance_range.end
                        || (batch.flags & SPRITE_FLAG_WORLD_SPACE) == 0
                        || (batch.flags & SPRITE_FLAG_DEPTH_WRITE) == 0
                    {
                        continue;
                    }
                    let Some(bind_group) = texture_bind_groups.get(&batch.texture_slot) else {
                        continue;
                    };
                    pass.set_pipeline(self.alpha_depth_write_pipeline.read().as_ref().unwrap());
                    pass.set_bind_group(1, bind_group, &[]);
                    pass.draw(0..6, batch.instance_range.clone());
                }
                pick_initialized = true;
                world_depth_initialized = true;
            }

            if has_world_translucent_batches {
                let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("RenderGraph/SpriteWorldBlend"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &swapchain.view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &pick_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: if pick_initialized {
                                    wgpu::LoadOp::Load
                                } else {
                                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                                },
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: world_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: if world_depth_initialized {
                                wgpu::LoadOp::Load
                            } else {
                                wgpu::LoadOp::Clear(0.0)
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_vertex_buffer(0, instance_buffer.buffer.slice(..));
                pass.set_bind_group(0, &camera_bg, &[]);
                for batch in frame.sprite_batches.iter() {
                    if batch.instance_range.start >= batch.instance_range.end
                        || (batch.flags & SPRITE_FLAG_WORLD_SPACE) == 0
                        || (batch.flags & SPRITE_FLAG_DEPTH_WRITE) != 0
                    {
                        continue;
                    }
                    let Some(bind_group) = texture_bind_groups.get(&batch.texture_slot) else {
                        continue;
                    };
                    let pipeline = match batch.blend_mode {
                        SpriteBlendMode::Alpha => self.alpha_depth_read_pipeline.read(),
                        SpriteBlendMode::Premultiplied => {
                            self.premultiplied_depth_read_pipeline.read()
                        }
                        SpriteBlendMode::Additive => self.additive_depth_read_pipeline.read(),
                    };
                    pass.set_pipeline(pipeline.as_ref().unwrap());
                    pass.set_bind_group(1, bind_group, &[]);
                    pass.draw(0..6, batch.instance_range.clone());
                }
                pick_initialized = true;
            }
        }

        if has_screen_batches {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RenderGraph/SpriteScreen"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &swapchain.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pick_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if pick_initialized {
                                wgpu::LoadOp::Load
                            } else {
                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_vertex_buffer(0, instance_buffer.buffer.slice(..));
            pass.set_bind_group(0, &camera_bg, &[]);
            for batch in frame.sprite_batches.iter() {
                if batch.instance_range.start >= batch.instance_range.end
                    || (batch.flags & SPRITE_FLAG_WORLD_SPACE) != 0
                {
                    continue;
                }
                let Some(bind_group) = texture_bind_groups.get(&batch.texture_slot) else {
                    continue;
                };
                let pipeline = match batch.blend_mode {
                    SpriteBlendMode::Alpha => self.alpha_pipeline.read(),
                    SpriteBlendMode::Premultiplied => self.premultiplied_pipeline.read(),
                    SpriteBlendMode::Additive => self.additive_pipeline.read(),
                };
                pass.set_pipeline(pipeline.as_ref().unwrap());
                pass.set_bind_group(1, bind_group, &[]);
                pass.draw(0..6, batch.instance_range.clone());
            }
        }

        ctx.rpctx.mark_used(self.outputs.swapchain);
        ctx.rpctx.mark_used(self.outputs.pick_map);
    }
}
