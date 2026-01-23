use parking_lot::RwLock;
use std::{num::NonZeroU32, sync::Arc};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::graphics::backend::binding_backend::BindingBackendKind;
use crate::graphics::common::raytracing::{
    RT_FLAG_DIRECT_LIGHTING, RT_FLAG_SHADOWS, RT_FLAG_USE_TEXTURES,
};
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
use crate::graphics::passes::{FrameGlobals, RayTracingFrameInput, gbuffer::GBufferOutputs};

#[derive(Clone, Copy, Debug)]
pub struct RtReflectionsOutputs {
    pub reflection: ResourceId,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RtReflectionParams {
    params0: [f32; 4],
    params1: [u32; 4],
    params2: [u32; 4],
}

struct RtReflectionsPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl0: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    bgl2: wgpu::BindGroupLayout,
    bgl3: Option<wgpu::BindGroupLayout>,
    texture_array_size: u32,
}

#[derive(Clone)]
pub struct RtReflectionsPass {
    gbuffer: GBufferOutputs,
    outputs: RtReflectionsOutputs,
    history: ResourceId,
    textured: Arc<RwLock<Option<RtReflectionsPipeline>>>,
    textured_arrays: Arc<RwLock<Option<RtReflectionsPipeline>>>,
    untextured: Arc<RwLock<Option<RtReflectionsPipeline>>>,
    fallback_storage: Arc<RwLock<Option<wgpu::Buffer>>>,
    use_transient_textures: bool,
}

impl RtReflectionsPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        gbuffer: GBufferOutputs,
        width: u32,
        height: u32,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
    ) -> Self {
        let usage = transient_usage(
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
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

        let (history_desc, mut history_hints) = ResourceDesc::Texture2D {
            width: width.max(1),
            height: height.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        }
        .with_hints();
        history_hints.flags |= ResourceFlags::PREFER_RESIDENT;
        let history = pool.create_logical(history_desc, Some(history_hints), 0, None);

        Self {
            gbuffer,
            outputs: RtReflectionsOutputs { reflection },
            history,
            textured: Arc::new(RwLock::new(None)),
            textured_arrays: Arc::new(RwLock::new(None)),
            untextured: Arc::new(RwLock::new(None)),
            fallback_storage: Arc::new(RwLock::new(None)),
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> RtReflectionsOutputs {
        self.outputs
    }

    fn ensure_target(
        &self,
        ctx: &mut RenderGraphExecCtx,
        extent: (u32, u32),
    ) -> Option<wgpu::TextureView> {
        let usage = transient_usage(
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
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
                label: Some("RTReflections/Output"),
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
                desc.clone(),
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
            if let Some(ref view) = view {
                entry.texture_view = Some(view.clone());
            }
        }
        ctx.rpctx
            .pool
            .mark_resident(self.outputs.reflection, ctx.rpctx.frame_index);
        view
    }

    fn ensure_history(
        &self,
        ctx: &mut RenderGraphExecCtx,
        extent: (u32, u32),
    ) -> Option<wgpu::TextureView> {
        let desc = ResourceDesc::Texture2D {
            width: extent.0.max(1),
            height: extent.1.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        };
        let needs_create = match ctx.rpctx.pool.entry(self.history) {
            Some(entry) => entry.texture.as_ref().map_or(true, |tex| {
                let size = tex.size();
                size.width != extent.0 || size.height != extent.1
            }),
            None => true,
        };
        let view = if needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("RTReflections/History"),
                size: wgpu::Extent3d {
                    width: extent.0.max(1),
                    height: extent.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            ctx.rpctx.pool.realize_texture(
                self.history,
                desc.clone(),
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            Some(view)
        } else {
            ctx.rpctx
                .pool
                .entry(self.history)
                .and_then(|entry| entry.texture.as_ref())
                .map(|tex| tex.create_view(&Default::default()))
        };

        if let Some(entry) = ctx.rpctx.pool.entry_mut(self.history) {
            if let Some(ref view) = view {
                entry.texture_view = Some(view.clone());
            }
        }
        ctx.rpctx
            .pool
            .mark_resident(self.history, ctx.rpctx.frame_index);
        view
    }

    fn compute_reflection_extent(
        &self,
        surface_size: winit::dpi::PhysicalSize<u32>,
        tlas_nodes: u32,
        cfg: &crate::graphics::common::config::RenderConfig,
        samples_per_frame: u32,
        direct_lighting: bool,
        direct_light_samples: u32,
        max_dim: u32,
    ) -> (u32, u32) {
        let surface_pixels = (surface_size.width as u64)
            .saturating_mul(surface_size.height as u64)
            .max(1);
        let mut rays_per_pixel = samples_per_frame.max(1) as u64;
        if direct_lighting {
            rays_per_pixel = rays_per_pixel.saturating_mul(direct_light_samples.max(1) as u64);
        }

        let mut ray_budget = cfg.rt_reflection_ray_budget.max(1);
        let complexity = tlas_nodes.max(1) as u64;
        let complexity_base = cfg.rt_reflection_complexity_base.max(1);
        let complexity_exponent = cfg.rt_reflection_complexity_exponent.max(0.0);
        if complexity > complexity_base && complexity_exponent > 0.0 {
            let scale =
                (complexity as f64 / complexity_base as f64).powf(complexity_exponent as f64);
            if scale.is_finite() && scale > 0.0 {
                ray_budget = (ray_budget as f64 / scale).round().max(1.0) as u64;
            }
        }

        let min_pixels = cfg
            .rt_reflection_min_pixel_budget
            .max(1)
            .min(surface_pixels);
        let pixel_budget = (ray_budget / rays_per_pixel.max(1)).clamp(min_pixels, surface_pixels);
        let mut scale = (pixel_budget as f64 / surface_pixels as f64).sqrt();
        let resolution_scale = if cfg.rt_reflection_resolution_scale.is_finite() {
            cfg.rt_reflection_resolution_scale.max(0.0)
        } else {
            1.0
        };
        scale *= resolution_scale as f64;

        let width = ((surface_size.width as f64) * scale).round().max(1.0) as u32;
        let height = ((surface_size.height as f64) * scale).round().max(1.0) as u32;
        (width.min(max_dim), height.min(max_dim))
    }

    fn create_common_bgls(
        device: &wgpu::Device,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RTReflections/BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            crate::graphics::common::renderer::CameraUniforms,
                        >() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            crate::graphics::common::raytracing::RtConstants,
                        >() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            crate::graphics::common::renderer::ShaderConstants,
                        >() as u64),
                    },
                    count: None,
                },
            ],
        });

        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RTReflections/GBufferBGL"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            RtReflectionParams,
                        >() as u64),
                    },
                    count: None,
                },
            ],
        });

        let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RTReflections/OutputBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        (bgl0, bgl1, bgl2)
    }

    fn create_texture_bgl_arrays(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RTReflections/TextureBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_texture_bgl_bindless(
        device: &wgpu::Device,
        array_size: u32,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RTReflections/BindlessBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(NonZeroU32::new(array_size.max(1)).expect("array size")),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn ensure_pipeline_untextured(&self, device: &wgpu::Device) {
        if self.untextured.read().is_some() {
            return;
        }

        let (bgl0, bgl1, bgl2) = Self::create_common_bgls(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RTReflections/Layout"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
            immediate_size: 0,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/rt_reflections.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RTReflections/Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("rt_reflections"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.untextured.write() = Some(RtReflectionsPipeline {
            pipeline,
            bgl0,
            bgl1,
            bgl2,
            bgl3: None,
            texture_array_size: 0,
        });
    }

    fn ensure_pipeline_textured_arrays(&self, device: &wgpu::Device) {
        if self.textured_arrays.read().is_some() {
            return;
        }

        let (bgl0, bgl1, bgl2) = Self::create_common_bgls(device);
        let bgl3 = Self::create_texture_bgl_arrays(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RTReflections/LayoutArrays"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3],
            immediate_size: 0,
        });

        let shader = device
            .create_shader_module(wgpu::include_wgsl!("../shaders/rt_reflections_arrays.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RTReflections/PipelineArrays"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("rt_reflections"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.textured_arrays.write() = Some(RtReflectionsPipeline {
            pipeline,
            bgl0,
            bgl1,
            bgl2,
            bgl3: Some(bgl3),
            texture_array_size: 0,
        });
    }

    fn ensure_pipeline_textured(&self, device: &wgpu::Device, array_size: u32) {
        if let Some(existing) = self.textured.read().as_ref() {
            if existing.texture_array_size == array_size {
                return;
            }
        }

        let (bgl0, bgl1, bgl2) = Self::create_common_bgls(device);
        let bgl3 = Self::create_texture_bgl_bindless(device, array_size);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RTReflections/LayoutBindless"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/rt_reflections_bindless.wgsl"
        ));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RTReflections/PipelineBindless"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("rt_reflections"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.textured.write() = Some(RtReflectionsPipeline {
            pipeline,
            bgl0,
            bgl1,
            bgl2,
            bgl3: Some(bgl3),
            texture_array_size: array_size,
        });
    }

    fn fallback_storage(&self, device: &wgpu::Device) -> wgpu::Buffer {
        if let Some(buf) = self.fallback_storage.read().as_ref() {
            return buf.clone();
        }
        let mut guard = self.fallback_storage.write();
        if let Some(buf) = guard.as_ref() {
            return buf.clone();
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RTReflections/Fallback"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        *guard = Some(buffer.clone());
        buffer
    }
}

impl RenderPass for RtReflectionsPass {
    fn name(&self) -> &'static str {
        "RtReflectionsPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.gbuffer.normal);
        ctx.read(self.gbuffer.mra);
        ctx.read(self.gbuffer.depth_copy);
        ctx.write(self.outputs.reflection);
        ctx.rw(self.history);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        let inputs = match ctx.rpctx.frame_inputs.get::<RayTracingFrameInput>() {
            Some(i) => i,
            None => return,
        };

        let samples_per_frame = frame.render_config.rt_reflection_samples_per_frame.max(1);
        let direct_light_samples = frame
            .render_config
            .rt_reflection_direct_light_samples
            .max(1);
        let direct_lighting = frame.render_config.rt_reflection_direct_lighting;
        let extent = self.compute_reflection_extent(
            frame.surface_size,
            inputs.tlas_node_count,
            &frame.render_config,
            samples_per_frame,
            direct_lighting,
            direct_light_samples,
            frame.device_caps.limits.max_texture_dimension_2d,
        );
        let output = match self.ensure_target(ctx, extent) {
            Some(v) => v,
            None => return,
        };
        let history = match self.ensure_history(ctx, extent) {
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
        let mut flags = 0u32;
        if frame.render_config.rt_reflection_direct_lighting {
            flags |= RT_FLAG_DIRECT_LIGHTING;
        }
        if frame.render_config.rt_reflection_shadows {
            flags |= RT_FLAG_SHADOWS;
        }
        if frame.render_config.rt_use_textures {
            flags |= RT_FLAG_USE_TEXTURES;
        }

        let mut history_weight = frame.render_config.rt_reflection_history_weight;
        if !history_weight.is_finite() {
            history_weight = 0.0;
        }
        if !frame.render_config.rt_reflection_accumulation {
            history_weight = 0.0;
        }
        history_weight = history_weight.clamp(0.0, 1.0);

        let mut history_depth_threshold = frame.render_config.rt_reflection_history_depth_threshold;
        if !history_depth_threshold.is_finite() {
            history_depth_threshold = 0.0;
        }
        history_depth_threshold = history_depth_threshold.max(0.0);

        let interleave = frame.render_config.rt_reflection_interleave.max(1);
        let max_accumulation_frames = frame.render_config.rt_reflection_max_accumulation_frames;

        let params = RtReflectionParams {
            params0: [
                history_weight,
                history_depth_threshold,
                extent.0.max(1) as f32,
                extent.1.max(1) as f32,
            ],
            params1: [
                interleave,
                samples_per_frame,
                direct_light_samples,
                max_accumulation_frames,
            ],
            params2: [flags, 0, 0, 0],
        };
        let params_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RTReflections/Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let supports_bindless = frame
            .device_caps
            .features
            .contains(wgpu::Features::TEXTURE_BINDING_ARRAY);
        let want_textures = frame.render_config.rt_use_textures;
        let use_array_textures = want_textures && frame.rt_texture_arrays.is_some();
        let use_bindless_textures = want_textures
            && !use_array_textures
            && supports_bindless
            && frame.binding_backend != BindingBackendKind::BindGroups;

        let (pipeline, bgl0, bgl1, bgl2, bgl3, texture_array_size) = if use_array_textures {
            self.ensure_pipeline_textured_arrays(ctx.device());
            let guard = self.textured_arrays.read();
            let state = match guard.as_ref() {
                Some(state) => state,
                None => return,
            };
            (
                state.pipeline.clone(),
                state.bgl0.clone(),
                state.bgl1.clone(),
                state.bgl2.clone(),
                state.bgl3.clone(),
                state.texture_array_size,
            )
        } else if use_bindless_textures {
            let texture_view_count = frame.texture_views.len().max(1) as u32;
            let device_limit = frame
                .device_caps
                .limits
                .max_sampled_textures_per_shader_stage
                .max(1);
            let array_size = texture_view_count.min(device_limit).max(1);
            self.ensure_pipeline_textured(ctx.device(), array_size);
            let guard = self.textured.read();
            let state = match guard.as_ref() {
                Some(state) => state,
                None => return,
            };
            (
                state.pipeline.clone(),
                state.bgl0.clone(),
                state.bgl1.clone(),
                state.bgl2.clone(),
                state.bgl3.clone(),
                state.texture_array_size,
            )
        } else {
            self.ensure_pipeline_untextured(ctx.device());
            let guard = self.untextured.read();
            let state = match guard.as_ref() {
                Some(state) => state,
                None => return,
            };
            (
                state.pipeline.clone(),
                state.bgl0.clone(),
                state.bgl1.clone(),
                state.bgl2.clone(),
                state.bgl3.clone(),
                state.texture_array_size,
            )
        };

        let fallback = self.fallback_storage(ctx.device());
        let lights = frame.lights_buffer.as_ref().unwrap_or(&fallback).clone();
        let materials = frame.material_buffer.as_ref().unwrap_or(&fallback).clone();

        let bg0 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RTReflections/BG0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inputs.constants.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inputs.tlas_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: inputs.tlas_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: inputs.instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: inputs.blas_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: inputs.blas_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: inputs.blas_triangles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: inputs.blas_descs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: materials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: frame.sky_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: frame.render_constants_buffer.as_entire_binding(),
                },
            ],
        });

        let bg1 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RTReflections/GBufferBG"),
            layout: &bgl1,
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
                    resource: wgpu::BindingResource::TextureView(&history),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let bg2 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RTReflections/OutputBG"),
            layout: &bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&output),
            }],
        });

        let bg3 = if use_array_textures {
            let layout = match bgl3.as_ref() {
                Some(layout) => layout,
                None => return,
            };
            let arrays = match frame.rt_texture_arrays.as_ref() {
                Some(arrays) => arrays,
                None => return,
            };
            Some(ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RTReflections/TextureBG"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&arrays.albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&arrays.normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&arrays.metallic_roughness),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&arrays.emission),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                    },
                ],
            }))
        } else if use_bindless_textures {
            let layout = match bgl3.as_ref() {
                Some(layout) => layout,
                None => return,
            };
            let array_size = texture_array_size.max(1) as usize;
            let mut texture_views: Vec<&wgpu::TextureView> = Vec::with_capacity(array_size);
            for view in frame.texture_views.iter().take(array_size) {
                texture_views.push(view);
            }
            while texture_views.len() < array_size {
                texture_views.push(&frame.fallback_view);
            }
            Some(ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RTReflections/BindlessBG"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureViewArray(&texture_views),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                    },
                ],
            }))
        } else {
            None
        };

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RenderGraph/RT_Reflections"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.set_bind_group(2, &bg2, &[]);
        if let Some(bg3) = bg3.as_ref() {
            pass.set_bind_group(3, bg3, &[]);
        }

        let group_size = 8;
        let groups_x = (extent.0.max(1) + group_size - 1) / group_size;
        let groups_y = (extent.1.max(1) + group_size - 1) / group_size;
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        drop(pass);

        let accum_tex = ctx.rpctx.pool.texture(self.outputs.reflection);
        let history_tex = ctx.rpctx.pool.texture(self.history);
        if let (Some(accum_tex), Some(history_tex)) = (accum_tex, history_tex) {
            ctx.encoder.copy_texture_to_texture(
                accum_tex.as_image_copy(),
                history_tex.as_image_copy(),
                wgpu::Extent3d {
                    width: extent.0.max(1),
                    height: extent.1.max(1),
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}
