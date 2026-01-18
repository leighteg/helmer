use crate::graphics::backend::binding_backend::BindingBackendKind;
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
    passes::{FrameGlobals, RayTracingFrameInput},
    raytracing::RtConstants,
    renderer_common::common::{CameraUniforms, ShaderConstants},
};
use parking_lot::RwLock;
use std::{num::NonZeroU32, sync::Arc};

struct RayTracingPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl0: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    bgl2: Option<wgpu::BindGroupLayout>,
    texture_array_size: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct RayTracingOutputs {
    pub accumulation: ResourceId,
}

#[derive(Clone)]
pub struct RayTracingPass {
    outputs: RayTracingOutputs,
    history: ResourceId,
    textured: Arc<RwLock<Option<RayTracingPipeline>>>,
    textured_arrays: Arc<RwLock<Option<RayTracingPipeline>>>,
    untextured: Arc<RwLock<Option<RayTracingPipeline>>>,
    fallback_storage: Arc<RwLock<Option<wgpu::Buffer>>>,
}

impl RayTracingPass {
    pub fn new(pool: &mut GpuResourcePool, width: u32, height: u32) -> Self {
        let (desc, mut hints) = ResourceDesc::Texture2D {
            width: width.max(1),
            height: height.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        }
        .with_hints();
        hints.flags |= ResourceFlags::PREFER_RESIDENT;
        let accumulation = pool.create_logical(desc, Some(hints), 0, None);
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
            outputs: RayTracingOutputs { accumulation },
            history,
            textured: Arc::new(RwLock::new(None)),
            textured_arrays: Arc::new(RwLock::new(None)),
            untextured: Arc::new(RwLock::new(None)),
            fallback_storage: Arc::new(RwLock::new(None)),
        }
    }

    pub fn outputs(&self) -> RayTracingOutputs {
        self.outputs
    }

    fn ensure_textures(&self, ctx: &mut RenderGraphExecCtx, extent: (u32, u32)) {
        let desc = ResourceDesc::Texture2D {
            width: extent.0.max(1),
            height: extent.1.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        };
        let needs_create = match ctx.rpctx.pool.entry(self.outputs.accumulation) {
            Some(entry) => {
                let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                    let size = t.size();
                    size.width == extent.0 && size.height == extent.1
                });
                !tex_ok
            }
            None => true,
        };

        if needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("RayTracing/Accumulation"),
                size: wgpu::Extent3d {
                    width: extent.0.max(1),
                    height: extent.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            ctx.rpctx.pool.realize_texture(
                self.outputs.accumulation,
                desc.clone(),
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            if let Some(entry) = ctx.rpctx.pool.entry_mut(self.outputs.accumulation) {
                entry.texture_view = Some(view);
            }
        }

        let history_desc = ResourceDesc::Texture2D {
            width: extent.0.max(1),
            height: extent.1.max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        };
        let history_needs_create = match ctx.rpctx.pool.entry(self.history) {
            Some(entry) => {
                let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                    let size = t.size();
                    size.width == extent.0 && size.height == extent.1
                });
                !tex_ok
            }
            None => true,
        };
        if history_needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("RayTracing/History"),
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
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            ctx.rpctx.pool.realize_texture(
                self.history,
                history_desc.clone(),
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            if let Some(entry) = ctx.rpctx.pool.entry_mut(self.history) {
                entry.texture_view = Some(view);
            }
        }

        ctx.rpctx
            .pool
            .mark_resident(self.outputs.accumulation, ctx.rpctx.frame_index);
        ctx.rpctx
            .pool
            .mark_resident(self.history, ctx.rpctx.frame_index);
    }

    fn create_common_bgls(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RayTracing/BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<CameraUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<RtConstants>() as u64
                        ),
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
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ShaderConstants>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RayTracing/BGL1"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            ],
        });

        (bgl0, bgl1)
    }

    fn ensure_pipeline_untextured(&self, device: &wgpu::Device) {
        if self.untextured.read().is_some() {
            return;
        }

        let (bgl0, bgl1) = Self::create_common_bgls(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RayTracing/PipelineLayout"),
            bind_group_layouts: &[&bgl0, &bgl1],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("raytracing_untextured.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RayTracing/Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.untextured.write() = Some(RayTracingPipeline {
            pipeline,
            bgl0,
            bgl1,
            bgl2: None,
            texture_array_size: 0,
        });
    }

    fn ensure_pipeline_textured(&self, device: &wgpu::Device, array_size: u32) {
        let array_size = array_size.max(1);
        if let Some(existing) = self.textured.read().as_ref() {
            if existing.texture_array_size == array_size {
                return;
            }
        }

        let (bgl0, bgl1) = Self::create_common_bgls(device);
        let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RayTracing/BGL2"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(NonZeroU32::new(array_size).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RayTracing/PipelineLayout"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("raytracing.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RayTracing/Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.textured.write() = Some(RayTracingPipeline {
            pipeline,
            bgl0,
            bgl1,
            bgl2: Some(bgl2),
            texture_array_size: array_size,
        });
    }

    fn ensure_pipeline_textured_arrays(&self, device: &wgpu::Device) {
        if self.textured_arrays.read().is_some() {
            return;
        }

        let (bgl0, bgl1) = Self::create_common_bgls(device);
        let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RayTracing/BGL2Arrays"),
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
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RayTracing/PipelineLayoutArrays"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("raytracing_arrays.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RayTracing/PipelineArrays"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.textured_arrays.write() = Some(RayTracingPipeline {
            pipeline,
            bgl0,
            bgl1,
            bgl2: Some(bgl2),
            texture_array_size: 0,
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
            label: Some("RayTracing/FallbackStorage"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        *guard = Some(buffer.clone());
        buffer
    }
}

impl RenderPass for RayTracingPass {
    fn name(&self) -> &'static str {
        "RayTracingPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.rw(self.outputs.accumulation);
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

        let size = inputs.rt_extent;
        self.ensure_textures(ctx, (size.width, size.height));
        let accumulation = match ctx.rpctx.pool.texture_view(self.outputs.accumulation) {
            Some(view) => view.clone(),
            None => return,
        };
        let history = match ctx.rpctx.pool.texture_view(self.history) {
            Some(view) => view.clone(),
            None => return,
        };

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

        let (pipeline, bgl0, bgl1, bgl2) = if use_array_textures {
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
            )
        };

        let fallback = self.fallback_storage(ctx.device());
        let lights = frame.lights_buffer.as_ref().unwrap_or(&fallback).clone();
        let materials = frame.material_buffer.as_ref().unwrap_or(&fallback).clone();

        let bgl0 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RayTracing/BG0"),
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

        let bgl1 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RayTracing/BG1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&frame.blue_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&frame.blue_noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&accumulation),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&history),
                },
            ],
        });

        let bgl2 = if use_array_textures {
            let bgl2 = match bgl2 {
                Some(layout) => layout,
                None => return,
            };
            let arrays = match frame.rt_texture_arrays.as_ref() {
                Some(arrays) => arrays,
                None => return,
            };
            Some(ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RayTracing/BG2Arrays"),
                layout: &bgl2,
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
            let bgl2 = match bgl2 {
                Some(layout) => layout,
                None => return,
            };
            let texture_view_count = frame.texture_views.len().max(1) as u32;
            let device_limit = frame
                .device_caps
                .limits
                .max_sampled_textures_per_shader_stage
                .max(1);
            let array_size = texture_view_count.min(device_limit).max(1) as usize;
            let mut texture_views: Vec<&wgpu::TextureView> = Vec::with_capacity(array_size);
            for view in frame.texture_views.iter().take(array_size) {
                texture_views.push(view);
            }
            while texture_views.len() < array_size {
                texture_views.push(&frame.fallback_view);
            }

            Some(ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RayTracing/BG2"),
                layout: &bgl2,
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

        let workgroup = 8u32;
        let dispatch_x = (size.width + workgroup - 1) / workgroup;
        let dispatch_y = (size.height + workgroup - 1) / workgroup;

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RayTracing/Trace"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bgl0, &[]);
        pass.set_bind_group(1, &bgl1, &[]);
        if let Some(bgl2) = bgl2.as_ref() {
            pass.set_bind_group(2, bgl2, &[]);
        }
        pass.dispatch_workgroups(dispatch_x.max(1), dispatch_y.max(1), 1);
        drop(pass);

        let accum_tex = ctx.rpctx.pool.texture(self.outputs.accumulation);
        let history_tex = ctx.rpctx.pool.texture(self.history);
        if let (Some(accum_tex), Some(history_tex)) = (accum_tex, history_tex) {
            ctx.encoder.copy_texture_to_texture(
                accum_tex.as_image_copy(),
                history_tex.as_image_copy(),
                wgpu::Extent3d {
                    width: size.width.max(1),
                    height: size.height.max(1),
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}
