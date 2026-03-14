use parking_lot::RwLock;
use std::{num::NonZeroU32, sync::Arc};

use crate::graphics::backend::binding_backend::BindingBackendKind;
use crate::graphics::common::renderer::DdgiGridConstants;
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
use crate::graphics::passes::{FrameGlobals, RayTracingFrameInput};

#[derive(Clone, Copy, Debug)]
pub struct DdgiProbeOutputs {
    pub irradiance_a: ResourceId,
    pub irradiance_b: ResourceId,
    pub distance_a: ResourceId,
    pub distance_b: ResourceId,
}

#[derive(Clone, Copy, Debug)]
pub struct DdgiProbeSelection {
    pub irradiance_read: ResourceId,
    pub irradiance_write: ResourceId,
    pub distance_read: ResourceId,
    pub distance_write: ResourceId,
}

impl DdgiProbeOutputs {
    pub fn select(&self, frame_index: u32) -> DdgiProbeSelection {
        if (frame_index & 1) == 0 {
            DdgiProbeSelection {
                irradiance_read: self.irradiance_a,
                irradiance_write: self.irradiance_b,
                distance_read: self.distance_a,
                distance_write: self.distance_b,
            }
        } else {
            DdgiProbeSelection {
                irradiance_read: self.irradiance_b,
                irradiance_write: self.irradiance_a,
                distance_read: self.distance_b,
                distance_write: self.distance_a,
            }
        }
    }
}

struct DdgiProbeUpdatePipeline {
    pipeline: wgpu::ComputePipeline,
    bgl0: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    bgl2: wgpu::BindGroupLayout,
    bgl3: Option<wgpu::BindGroupLayout>,
    texture_array_size: u32,
}

#[derive(Clone)]
pub struct DdgiProbeUpdatePass {
    outputs: DdgiProbeOutputs,
    textured: Arc<RwLock<Option<DdgiProbeUpdatePipeline>>>,
    textured_arrays: Arc<RwLock<Option<DdgiProbeUpdatePipeline>>>,
    untextured: Arc<RwLock<Option<DdgiProbeUpdatePipeline>>>,
    fallback_storage: Arc<RwLock<Option<wgpu::Buffer>>>,
}

impl DdgiProbeUpdatePass {
    pub fn new(pool: &mut GpuResourcePool, probe_resolution: u32, probe_count: u32) -> Self {
        let (irr_desc, mut irr_hints) = ResourceDesc::Texture2D {
            width: probe_resolution.max(1),
            height: probe_resolution.max(1),
            mip_levels: 1,
            layers: probe_count.max(1),
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        }
        .with_hints();
        irr_hints.flags |= ResourceFlags::PREFER_RESIDENT;
        let irradiance_a = pool.create_logical(irr_desc.clone(), Some(irr_hints), 0, None);
        let irradiance_b = pool.create_logical(irr_desc, Some(irr_hints), 0, None);

        let (dist_desc, mut dist_hints) = ResourceDesc::Texture2D {
            width: probe_resolution.max(1),
            height: probe_resolution.max(1),
            mip_levels: 1,
            layers: probe_count.max(1),
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        }
        .with_hints();
        dist_hints.flags |= ResourceFlags::PREFER_RESIDENT;
        let distance_a = pool.create_logical(dist_desc.clone(), Some(dist_hints), 0, None);
        let distance_b = pool.create_logical(dist_desc, Some(dist_hints), 0, None);

        Self {
            outputs: DdgiProbeOutputs {
                irradiance_a,
                irradiance_b,
                distance_a,
                distance_b,
            },
            textured: Arc::new(RwLock::new(None)),
            textured_arrays: Arc::new(RwLock::new(None)),
            untextured: Arc::new(RwLock::new(None)),
            fallback_storage: Arc::new(RwLock::new(None)),
        }
    }

    pub fn outputs(&self) -> DdgiProbeOutputs {
        self.outputs
    }

    fn ensure_textures(
        &self,
        ctx: &mut RenderGraphExecCtx,
        probe_resolution: u32,
        probe_count: u32,
    ) -> Option<(
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::TextureView,
    )> {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let desc = ResourceDesc::Texture2D {
            width: probe_resolution.max(1),
            height: probe_resolution.max(1),
            mip_levels: 1,
            layers: probe_count.max(1),
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
        };

        let mut ensure_texture = |id: ResourceId, label: &str| -> Option<wgpu::TextureView> {
            let needs_create = ctx
                .rpctx
                .pool
                .entry(id)
                .map(|entry| {
                    let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                        let size = t.size();
                        size.width == probe_resolution.max(1)
                            && size.height == probe_resolution.max(1)
                            && size.depth_or_array_layers == probe_count.max(1)
                    });
                    !tex_ok
                })
                .unwrap_or(true);
            let view = if needs_create {
                let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: probe_resolution.max(1),
                        height: probe_resolution.max(1),
                        depth_or_array_layers: probe_count.max(1),
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(probe_count.max(1)),
                    ..Default::default()
                });
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
                    .and_then(|entry| entry.texture.as_ref())
                    .map(|tex| {
                        tex.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2Array),
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            base_array_layer: 0,
                            array_layer_count: Some(probe_count.max(1)),
                            ..Default::default()
                        })
                    })
            };
            if let Some(entry) = ctx.rpctx.pool.entry_mut(id) {
                if let Some(ref view) = view {
                    entry.texture_view = Some(view.clone());
                }
            }
            ctx.rpctx.pool.mark_resident(id, ctx.rpctx.frame_index);
            view
        };

        let irradiance_a = ensure_texture(self.outputs.irradiance_a, "DDGI/IrradianceA")?;
        let irradiance_b = ensure_texture(self.outputs.irradiance_b, "DDGI/IrradianceB")?;
        let distance_a = ensure_texture(self.outputs.distance_a, "DDGI/DistanceA")?;
        let distance_b = ensure_texture(self.outputs.distance_b, "DDGI/DistanceB")?;
        Some((irradiance_a, irradiance_b, distance_a, distance_b))
    }

    fn create_common_bgls(
        device: &wgpu::Device,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DDGI/ProbeUpdateBGL0"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DDGI/ProbeUpdateGridBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<DdgiGridConstants>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DDGI/ProbeUpdateTargetsBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        (bgl0, bgl1, bgl2)
    }

    fn create_texture_bgl_arrays(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DDGI/ProbeUpdateTextureBGL"),
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
            label: Some("DDGI/ProbeUpdateBindlessBGL"),
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
            label: Some("DDGI/ProbeUpdateLayout"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
            immediate_size: 0,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/ddgi_probe_update.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DDGI/ProbeUpdatePipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("probe_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.untextured.write() = Some(DdgiProbeUpdatePipeline {
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
            label: Some("DDGI/ProbeUpdateLayoutArrays"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/ddgi_probe_update_arrays.wgsl"
        ));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DDGI/ProbeUpdatePipelineArrays"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("probe_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.textured_arrays.write() = Some(DdgiProbeUpdatePipeline {
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
            label: Some("DDGI/ProbeUpdateLayoutBindless"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/ddgi_probe_update_bindless.wgsl"
        ));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DDGI/ProbeUpdatePipelineBindless"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("probe_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        *self.textured.write() = Some(DdgiProbeUpdatePipeline {
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
            label: Some("DDGI/ProbeUpdateFallback"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        *guard = Some(buffer.clone());
        buffer
    }
}

impl RenderPass for DdgiProbeUpdatePass {
    fn name(&self) -> &'static str {
        "DdgiProbeUpdatePass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.rw(self.outputs.irradiance_a);
        ctx.rw(self.outputs.irradiance_b);
        ctx.rw(self.outputs.distance_a);
        ctx.rw(self.outputs.distance_b);
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

        let mut counts = [
            frame.render_config.ddgi_probe_count_x.max(1),
            frame.render_config.ddgi_probe_count_y.max(1),
            frame.render_config.ddgi_probe_count_z.max(1),
        ];
        let max_tex_dim = frame.device_caps.limits.max_texture_dimension_2d.max(1);
        let mut probe_resolution = frame
            .render_config
            .ddgi_probe_resolution
            .max(1)
            .min(max_tex_dim);
        if probe_resolution == 0 {
            probe_resolution = 1;
        }
        let max_layers = frame.device_caps.limits.max_texture_array_layers.max(1);
        let mut total = counts[0] as u64 * counts[1] as u64 * counts[2] as u64;
        if total > max_layers as u64 {
            let scale = (max_layers as f64 / total as f64).cbrt();
            if scale.is_finite() && scale > 0.0 {
                counts[0] = ((counts[0] as f64 * scale).floor() as u32).max(1);
                counts[1] = ((counts[1] as f64 * scale).floor() as u32).max(1);
                counts[2] = ((counts[2] as f64 * scale).floor() as u32).max(1);
            }
            total = counts[0] as u64 * counts[1] as u64 * counts[2] as u64;
            while total > max_layers as u64 {
                let mut largest = 0usize;
                if counts[1] > counts[largest] {
                    largest = 1;
                }
                if counts[2] > counts[largest] {
                    largest = 2;
                }
                if counts[largest] <= 1 {
                    break;
                }
                counts[largest] -= 1;
                total = counts[0] as u64 * counts[1] as u64 * counts[2] as u64;
            }
        }
        let probe_count = total.max(1) as u32;

        let (irradiance_a, irradiance_b, distance_a, distance_b) =
            match self.ensure_textures(ctx, probe_resolution, probe_count) {
                Some(v) => v,
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
            label: Some("DDGI/ProbeUpdateBG0"),
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
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: frame.skin_palette_buffer.as_entire_binding(),
                },
            ],
        });

        let bg1 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DDGI/ProbeUpdateGridBG"),
            layout: &bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.ddgi_grid_buffer.as_entire_binding(),
            }],
        });

        let selection = self.outputs.select(frame.frame_index);
        let read_a = selection.irradiance_read == self.outputs.irradiance_a;
        let (irradiance_read, irradiance_write) = if read_a {
            (&irradiance_a, &irradiance_b)
        } else {
            (&irradiance_b, &irradiance_a)
        };
        let (distance_read, distance_write) = if read_a {
            (&distance_a, &distance_b)
        } else {
            (&distance_b, &distance_a)
        };

        let bg2 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DDGI/ProbeUpdateTargetsBG"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(irradiance_read),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(irradiance_write),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(distance_read),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(distance_write),
                },
            ],
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
                label: Some("DDGI/ProbeUpdateTextureBG"),
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
                label: Some("DDGI/ProbeUpdateBindlessBG"),
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
                label: Some("RenderGraph/DDGI_ProbeUpdate"),
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
        let groups_x = (probe_resolution + group_size - 1) / group_size;
        let groups_y = (probe_resolution + group_size - 1) / group_size;
        pass.dispatch_workgroups(groups_x, groups_y, probe_count.max(1));
    }
}
