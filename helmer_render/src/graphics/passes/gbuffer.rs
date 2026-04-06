use parking_lot::RwLock;
use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

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

use crate::graphics::backend::binding_backend::BindingBackendKind;
use crate::graphics::common::renderer::{
    CameraUniforms, GBufferInstanceRaw, MaterialShaderData, MeshDrawParams, MeshTaskTiling,
    ShaderConstants, Vertex, mesh_shader_visibility, mesh_task_tiling, transient_usage,
};
use crate::graphics::passes::{FrameGlobals, GBufferBundleKey};

/// Resource IDs for the G-buffer attachments
#[derive(Clone, Copy, Debug)]
pub struct GBufferOutputs {
    pub normal: ResourceId,
    pub albedo: ResourceId,
    pub mra: ResourceId,
    pub emission: ResourceId,
    pub depth: ResourceId,
    pub depth_copy: ResourceId,
}

#[derive(Clone, Copy, Debug)]
pub struct GBufferFormats {
    pub normal: wgpu::TextureFormat,
    pub albedo: wgpu::TextureFormat,
    pub emission: wgpu::TextureFormat,
    pub mra: wgpu::TextureFormat,
    pub depth: wgpu::TextureFormat,
    pub depth_copy: wgpu::TextureFormat,
    pub output_depth_copy: bool,
}

impl GBufferFormats {
    pub fn select(max_color_attachment_bytes_per_sample: u32) -> Self {
        let force_split = max_color_attachment_bytes_per_sample <= 32;
        let full = Self::full();
        let full_bytes = full.bytes_per_sample_with_depth_copy();
        if !force_split && full_bytes <= max_color_attachment_bytes_per_sample {
            return full;
        }

        let compact = Self::compact();
        let compact_bytes = compact.bytes_per_sample_with_depth_copy();
        if !force_split && compact_bytes <= max_color_attachment_bytes_per_sample {
            tracing::warn!(
                max_color_attachment_bytes_per_sample,
                full_bytes_per_sample = full_bytes,
                compact_bytes_per_sample = compact_bytes,
                "G-buffer formats downgraded to compact to fit device color attachment limits."
            );
            return compact;
        }

        let ultra = Self::ultra();
        let ultra_bytes = ultra.bytes_per_sample_with_depth_copy();
        if !force_split && ultra_bytes <= max_color_attachment_bytes_per_sample {
            tracing::warn!(
                max_color_attachment_bytes_per_sample,
                full_bytes_per_sample = full_bytes,
                compact_bytes_per_sample = compact_bytes,
                ultra_bytes_per_sample = ultra_bytes,
                "G-buffer formats downgraded to ultra to fit device color attachment limits."
            );
            return ultra;
        }

        let mut split_full = Self::full();
        split_full.output_depth_copy = false;
        let split_full_bytes = split_full.bytes_per_sample_without_depth_copy();
        if split_full_bytes <= max_color_attachment_bytes_per_sample {
            let split_reason = if force_split {
                "G-buffer inline depth copy disabled by device limit; using split depth copy pass."
            } else {
                "G-buffer inline depth copy exceeds device limit; using split depth copy pass."
            };
            tracing::warn!(
                max_color_attachment_bytes_per_sample,
                full_bytes_per_sample = full_bytes,
                compact_bytes_per_sample = compact_bytes,
                ultra_bytes_per_sample = ultra_bytes,
                split_bytes_per_sample = split_full_bytes,
                "{split_reason}"
            );
            return split_full;
        }

        let mut split_compact = Self::compact();
        split_compact.output_depth_copy = false;
        let split_compact_bytes = split_compact.bytes_per_sample_without_depth_copy();
        if split_compact_bytes <= max_color_attachment_bytes_per_sample {
            tracing::warn!(
                max_color_attachment_bytes_per_sample,
                full_bytes_per_sample = full_bytes,
                compact_bytes_per_sample = compact_bytes,
                ultra_bytes_per_sample = ultra_bytes,
                split_bytes_per_sample = split_compact_bytes,
                "G-buffer inline depth copy exceeds device limit; using compact split layout."
            );
            return split_compact;
        }

        let mut split_ultra = Self::ultra();
        split_ultra.output_depth_copy = false;
        let split_ultra_bytes = split_ultra.bytes_per_sample_without_depth_copy();

        tracing::warn!(
            max_color_attachment_bytes_per_sample,
            full_bytes_per_sample = full_bytes,
            compact_bytes_per_sample = compact_bytes,
            ultra_bytes_per_sample = ultra_bytes,
            split_bytes_per_sample = split_ultra_bytes,
            "G-buffer formats exceed device color attachment limits; using ultra split layout."
        );
        split_ultra
    }

    fn full() -> Self {
        Self {
            normal: wgpu::TextureFormat::Rgba16Float,
            albedo: wgpu::TextureFormat::Rgba16Float,
            emission: wgpu::TextureFormat::Rgba16Float,
            mra: wgpu::TextureFormat::Rgba8Unorm,
            depth: wgpu::TextureFormat::Depth32Float,
            depth_copy: wgpu::TextureFormat::R32Float,
            output_depth_copy: true,
        }
    }

    fn compact() -> Self {
        Self {
            normal: wgpu::TextureFormat::Rgba8Unorm,
            albedo: wgpu::TextureFormat::Rgba8Unorm,
            emission: wgpu::TextureFormat::Rgba16Float,
            mra: wgpu::TextureFormat::Rgba8Unorm,
            depth: wgpu::TextureFormat::Depth32Float,
            depth_copy: wgpu::TextureFormat::R32Float,
            output_depth_copy: true,
        }
    }

    pub fn ultra() -> Self {
        Self {
            normal: wgpu::TextureFormat::Rgba8Unorm,
            albedo: wgpu::TextureFormat::Rgba8Unorm,
            emission: wgpu::TextureFormat::Rgba8Unorm,
            mra: wgpu::TextureFormat::Rgba8Unorm,
            depth: wgpu::TextureFormat::Depth32Float,
            depth_copy: wgpu::TextureFormat::R32Float,
            output_depth_copy: true,
        }
    }

    fn color_formats_with_depth_copy(&self) -> [wgpu::TextureFormat; 5] {
        // Order larger alignments first to minimize padding in the attachment byte limit
        [
            self.depth_copy,
            self.normal,
            self.albedo,
            self.emission,
            self.mra,
        ]
    }

    fn color_formats_without_depth_copy(&self) -> [wgpu::TextureFormat; 4] {
        [self.normal, self.albedo, self.emission, self.mra]
    }

    fn color_targets(&self) -> Vec<Option<wgpu::ColorTargetState>> {
        if self.output_depth_copy {
            vec![
                Some(self.depth_copy.into()),
                Some(self.normal.into()),
                Some(self.albedo.into()),
                Some(self.emission.into()),
                Some(self.mra.into()),
            ]
        } else {
            vec![
                Some(self.normal.into()),
                Some(self.albedo.into()),
                Some(self.emission.into()),
                Some(self.mra.into()),
            ]
        }
    }

    fn color_format_options(&self) -> Vec<Option<wgpu::TextureFormat>> {
        if self.output_depth_copy {
            vec![
                Some(self.depth_copy),
                Some(self.normal),
                Some(self.albedo),
                Some(self.emission),
                Some(self.mra),
            ]
        } else {
            vec![
                Some(self.normal),
                Some(self.albedo),
                Some(self.emission),
                Some(self.mra),
            ]
        }
    }

    fn bytes_per_sample_with_depth_copy(&self) -> u32 {
        bytes_per_sample(self.color_formats_with_depth_copy().iter().copied())
    }

    fn bytes_per_sample_without_depth_copy(&self) -> u32 {
        bytes_per_sample(self.color_formats_without_depth_copy().iter().copied())
    }
}

fn bytes_per_sample(formats: impl IntoIterator<Item = wgpu::TextureFormat>) -> u32 {
    let mut total = 0u32;
    for format in formats {
        let byte_cost = format
            .target_pixel_byte_cost()
            .expect("color attachment has no byte cost");
        let alignment = format
            .target_component_alignment()
            .expect("color attachment has no component alignment");
        total = align_up(total, alignment);
        total = total.saturating_add(byte_cost);
    }
    total
}

fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment <= 1 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

#[derive(Clone)]
pub struct GBufferPass {
    extent: (u32, u32),
    outputs: GBufferOutputs,
    formats: GBufferFormats,
    use_uniform_materials: bool,
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    camera_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    material_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    constants_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    mesh_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    mesh_params: Arc<RwLock<Option<wgpu::Buffer>>>,
    mesh_params_capacity: Arc<AtomicU64>,
    texture_array_size: Arc<AtomicU32>,
    backend_kind: Arc<RwLock<Option<BindingBackendKind>>>,
    material_bind_groups: Arc<RwLock<Option<MaterialBindGroupCache>>>,
    bundle_cache: Arc<RwLock<Vec<GBufferBundleSlot>>>,
}

#[derive(Default)]
struct GBufferBundleSlot {
    key: Option<GBufferBundleKey>,
    bundle: Option<wgpu::RenderBundle>,
    resources: Vec<ResourceId>,
}

struct MaterialBindGroupCache {
    version: u64,
    groups: Arc<Vec<wgpu::BindGroup>>,
}

fn indirect_draw_run_len(
    draws: &[crate::graphics::passes::IndirectDrawBatch],
    start: usize,
    include_material: bool,
) -> u32 {
    let Some(draw) = draws.get(start) else {
        return 0;
    };
    let stride = std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;
    let mut count = 1u32;
    let mut next_offset = draw.indirect_offset + stride;
    let mut next_index = start + 1;
    while let Some(next) = draws.get(next_index) {
        if next.vertex != draw.vertex
            || next.index != draw.index
            || next.indirect_offset != next_offset
            || (include_material && next.material_id != draw.material_id)
        {
            break;
        }
        count += 1;
        next_offset += stride;
        next_index += 1;
    }
    count
}

impl GBufferPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        width: u32,
        height: u32,
        formats: GBufferFormats,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
        use_uniform_materials: bool,
    ) -> Self {
        let mut make_rt = |format: wgpu::TextureFormat, label: &str| {
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
            let id = pool.create_logical(desc, Some(hints), 0, None);
            /*tracing::debug!(
                "Allocated logical G-buffer target '{label}' with id {:?}",
                id
            );*/
            id
        };

        let outputs = GBufferOutputs {
            normal: make_rt(formats.normal, "normal"),
            albedo: make_rt(formats.albedo, "albedo"),
            mra: make_rt(formats.mra, "mra"),
            emission: make_rt(formats.emission, "emission"),
            depth: make_rt(formats.depth, "depth"),
            depth_copy: make_rt(formats.depth_copy, "depth_copy"),
        };

        Self {
            extent: (width, height),
            outputs,
            formats,
            use_uniform_materials,
            pipeline: Arc::new(RwLock::new(None)),
            mesh_pipeline: Arc::new(RwLock::new(None)),
            camera_bgl: Arc::new(RwLock::new(None)),
            material_bgl: Arc::new(RwLock::new(None)),
            constants_bgl: Arc::new(RwLock::new(None)),
            mesh_bgl: Arc::new(RwLock::new(None)),
            mesh_params: Arc::new(RwLock::new(None)),
            mesh_params_capacity: Arc::new(AtomicU64::new(0)),
            texture_array_size: Arc::new(AtomicU32::new(1)),
            backend_kind: Arc::new(RwLock::new(None)),
            material_bind_groups: Arc::new(RwLock::new(None)),
            bundle_cache: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn outputs(&self) -> GBufferOutputs {
        self.outputs
    }

    pub fn clear_bundle_cache(&self) {
        let mut cache = self.bundle_cache.write();
        for slot in cache.iter_mut() {
            slot.key = None;
            slot.bundle = None;
            slot.resources.clear();
        }
    }

    pub fn invalidate_bundles_for_resources(&self, evicted: &[ResourceId]) -> bool {
        if evicted.is_empty() {
            return false;
        }

        let mut cache = self.bundle_cache.write();
        let mut invalidated = false;
        for slot in cache.iter_mut() {
            if slot.bundle.is_none() || slot.resources.is_empty() {
                continue;
            }
            let should_clear = evicted
                .iter()
                .any(|id| slot.resources.binary_search(id).is_ok());
            if should_clear {
                slot.key = None;
                slot.bundle = None;
                slot.resources.clear();
                invalidated = true;
            }
        }
        invalidated
    }

    fn gather_bundle_resources(frame: &FrameGlobals, use_indirect: bool) -> Vec<ResourceId> {
        let mut resources = Vec::new();
        if use_indirect {
            for draw in frame.gpu_draws.iter() {
                resources.push(draw.vertex);
                resources.push(draw.index);
            }
        } else {
            for batch in frame.gbuffer_batches.iter() {
                resources.push(batch.vertex);
                resources.push(batch.index);
            }
        }
        resources.sort_unstable();
        resources.dedup();
        resources
    }

    fn ensure_target(
        &self,
        ctx: &mut RenderGraphExecCtx,
        id: ResourceId,
        format: wgpu::TextureFormat,
        label: &str,
        use_transient_textures: bool,
    ) -> wgpu::TextureView {
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            use_transient_textures,
        );
        let resource_desc = ResourceDesc::Texture2D {
            width: self.extent.0,
            height: self.extent.1,
            mip_levels: 1,
            layers: 1,
            format,
            usage,
        };
        let texture_desc = wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: self.extent.0,
                height: self.extent.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        };

        // Check if we need to create or recreate the target (size mismatch or missing)
        let needs_create = match ctx.rpctx.pool.entry(id) {
            Some(entry) => {
                let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                    let size = t.size();
                    size.width == self.extent.0 && size.height == self.extent.1
                });
                !tex_ok
            }
            None => true,
        };

        if needs_create {
            let texture = ctx.device().create_texture(&texture_desc);
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            ctx.rpctx.pool.realize_texture(
                id,
                resource_desc.clone(),
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            if let Some(entry) = ctx.rpctx.pool.entry_mut(id) {
                entry.texture_view = Some(view.clone());
            }
            ctx.rpctx.pool.mark_resident(id, ctx.rpctx.frame_index);
            return view;
        }

        // Always refresh the view to a single mip-level renderable view in case the entry was reused
        let view = {
            let texture = ctx
                .rpctx
                .pool
                .entry(id)
                .and_then(|e| e.texture.as_ref())
                .expect("G-buffer texture missing");
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            })
        };
        if let Some(entry) = ctx.rpctx.pool.entry_mut(id) {
            entry.texture_view = Some(view.clone());
        }
        ctx.rpctx.pool.mark_resident(id, ctx.rpctx.frame_index);

        view
    }

    fn ensure_backend(&self, binding_backend: BindingBackendKind) {
        let mut current = self.backend_kind.write();
        if current.map_or(true, |kind| kind != binding_backend) {
            *self.pipeline.write() = None;
            *self.mesh_pipeline.write() = None;
            *self.camera_bgl.write() = None;
            *self.material_bgl.write() = None;
            *self.constants_bgl.write() = None;
            *self.mesh_bgl.write() = None;
            *self.material_bind_groups.write() = None;
            self.texture_array_size.store(1, Ordering::Relaxed);
            *current = Some(binding_backend);
        }
    }

    fn ensure_pipeline(
        &self,
        device: &wgpu::Device,
        binding_backend: BindingBackendKind,
        texture_array_size: u32,
    ) {
        self.ensure_backend(binding_backend);
        if binding_backend == BindingBackendKind::BindGroups {
            if self.pipeline.read().is_some() {
                return;
            }
        } else {
            let current_size = self.texture_array_size.load(Ordering::Relaxed);
            if self.pipeline.read().is_some() && current_size == texture_array_size {
                return;
            }
        }

        let array_size = texture_array_size.max(1);
        if binding_backend != BindingBackendKind::BindGroups {
            self.texture_array_size.store(array_size, Ordering::Relaxed);
        } else {
            self.texture_array_size.store(1, Ordering::Relaxed);
        }

        let mesh_stage = mesh_shader_visibility(device);
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GBuffer/CameraBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | mesh_stage,
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
                    visibility: wgpu::ShaderStages::VERTEX | mesh_stage,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let material_entries = if binding_backend == BindingBackendKind::BindGroups {
            let material_buffer_entry = wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: if self.use_uniform_materials {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: if self.use_uniform_materials {
                        wgpu::BufferSize::new(std::mem::size_of::<MaterialShaderData>() as u64)
                    } else {
                        None
                    },
                },
                count: None,
            };
            vec![
                material_buffer_entry,
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
            ]
        } else {
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
                    count: Some(std::num::NonZeroU32::new(array_size).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ]
        };
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GBuffer/MaterialBGL"),
            entries: &material_entries,
        });

        let constants_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GBuffer/ConstantsBGL"),
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
            label: Some("GBuffer/PipelineLayout"),
            bind_group_layouts: &[&camera_bgl, &material_bgl, &constants_bgl],
            immediate_size: 0,
        });

        let shader = if binding_backend == BindingBackendKind::BindGroups {
            if self.use_uniform_materials {
                if self.formats.output_depth_copy {
                    device.create_shader_module(wgpu::include_wgsl!(
                        "../shaders/g_buffer_bindgroups_webgl.wgsl"
                    ))
                } else {
                    device.create_shader_module(wgpu::include_wgsl!(
                        "../shaders/g_buffer_bindgroups_no_depth_webgl.wgsl"
                    ))
                }
            } else if self.formats.output_depth_copy {
                device.create_shader_module(wgpu::include_wgsl!(
                    "../shaders/g_buffer_bindgroups.wgsl"
                ))
            } else {
                device.create_shader_module(wgpu::include_wgsl!(
                    "../shaders/g_buffer_bindgroups_no_depth.wgsl"
                ))
            }
        } else if self.formats.output_depth_copy {
            device.create_shader_module(wgpu::include_wgsl!("../shaders/g_buffer.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("../shaders/g_buffer_no_depth.wgsl"))
        };

        let targets = self.formats.color_targets();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GBuffer/Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), GBufferInstanceRaw::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: self.formats.depth,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.camera_bgl.write() = Some(camera_bgl);
        *self.material_bgl.write() = Some(material_bgl);
        *self.constants_bgl.write() = Some(constants_bgl);
    }

    fn ensure_mesh_pipeline(
        &self,
        device: &wgpu::Device,
        binding_backend: BindingBackendKind,
        texture_array_size: u32,
    ) {
        if mesh_shader_visibility(device).is_empty() {
            return;
        }

        if binding_backend == BindingBackendKind::BindGroups {
            if self.mesh_pipeline.read().is_some() {
                return;
            }
        } else {
            let current_size = self.texture_array_size.load(Ordering::Relaxed);
            if self.mesh_pipeline.read().is_some() && current_size == texture_array_size {
                return;
            }
        }

        self.ensure_pipeline(device, binding_backend, texture_array_size);

        let (camera_bgl, material_bgl, constants_bgl) = (
            self.camera_bgl.read(),
            self.material_bgl.read(),
            self.constants_bgl.read(),
        );
        let (camera_bgl, material_bgl, constants_bgl) = (
            camera_bgl.as_ref().unwrap(),
            material_bgl.as_ref().unwrap(),
            constants_bgl.as_ref().unwrap(),
        );

        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GBuffer/MeshBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<MeshDrawParams>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::MESH,
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
            label: Some("GBuffer/MeshPipelineLayout"),
            bind_group_layouts: &[camera_bgl, material_bgl, constants_bgl, &mesh_bgl],
            immediate_size: 0,
        });

        let shader = if binding_backend == BindingBackendKind::BindGroups {
            if self.formats.output_depth_copy {
                device.create_shader_module(wgpu::include_wgsl!(
                    "../shaders/g_buffer_mesh_bindgroups.wgsl"
                ))
            } else {
                device.create_shader_module(wgpu::include_wgsl!(
                    "../shaders/g_buffer_mesh_bindgroups_no_depth.wgsl"
                ))
            }
        } else if self.formats.output_depth_copy {
            device.create_shader_module(wgpu::include_wgsl!("../shaders/g_buffer_mesh.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!(
                "../shaders/g_buffer_mesh_no_depth.wgsl"
            ))
        };

        let targets = self.formats.color_targets();
        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("GBuffer/MeshPipeline"),
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
                targets: &targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: self.formats.depth,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        *self.mesh_pipeline.write() = Some(pipeline);
        *self.mesh_bgl.write() = Some(mesh_bgl);
    }

    fn ensure_bundle_cache(&self, frames_in_flight: usize) {
        let slots = frames_in_flight.max(1);
        let mut cache = self.bundle_cache.write();
        if cache.len() != slots {
            cache.clear();
            cache.resize_with(slots, GBufferBundleSlot::default);
        }
    }

    fn create_camera_bind_group(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<wgpu::BindGroup> {
        let camera_layout = self.camera_bgl.read();
        let camera_layout = camera_layout.as_ref()?;
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GBuffer/CameraBG"),
            layout: camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.skin_palette_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    fn create_constants_bind_group(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<wgpu::BindGroup> {
        let constants_layout = self.constants_bgl.read();
        let constants_layout = constants_layout.as_ref()?;
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GBuffer/ConstantsBG"),
            layout: constants_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.render_constants_buffer.as_entire_binding(),
            }],
        }))
    }

    fn create_bind_groups_bindless(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<(wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup)> {
        let camera_bg = self.create_camera_bind_group(device, frame)?;
        let material_bg = self.create_material_bind_group_bindless(device, frame)?;
        let constants_bg = self.create_constants_bind_group(device, frame)?;
        Some((camera_bg, material_bg, constants_bg))
    }

    fn create_material_bind_group_bindless(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<wgpu::BindGroup> {
        let material_layout = self.material_bgl.read();
        let material_layout = material_layout.as_ref()?;

        let array_size = self.texture_array_size.load(Ordering::Relaxed);
        let mut texture_views: Vec<&wgpu::TextureView> = Vec::with_capacity(array_size as usize);
        for view in frame.texture_views.iter() {
            texture_views.push(view);
        }
        while texture_views.len() < array_size as usize {
            texture_views.push(&frame.fallback_view);
        }

        let material_buffer = frame.material_buffer.as_ref()?;
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GBuffer/MaterialBG"),
            layout: material_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureViewArray(&texture_views),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                },
            ],
        }))
    }

    fn ensure_material_bind_groups(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<Arc<Vec<wgpu::BindGroup>>> {
        let material_textures = frame.material_textures.as_ref()?;
        let material_layout = self.material_bgl.read();
        let material_layout = material_layout.as_ref()?;
        let version = frame.material_bindings_version;
        if let Some(cache) = self.material_bind_groups.read().as_ref() {
            if cache.version == version && cache.groups.len() == material_textures.len() {
                return Some(cache.groups.clone());
            }
        }

        let mut groups = Vec::with_capacity(material_textures.len());
        if self.use_uniform_materials {
            let material_buffer = frame.material_uniform_buffer.as_ref()?;
            let stride = frame.material_uniform_stride;
            let element_size = std::mem::size_of::<MaterialShaderData>() as u64;
            if stride < element_size {
                return None;
            }
            for (index, set) in material_textures.iter().enumerate() {
                let offset = stride.saturating_mul(index as u64);
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GBuffer/MaterialBG"),
                    layout: material_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: material_buffer,
                                offset,
                                size: wgpu::BufferSize::new(element_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&set.albedo),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&set.normal),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&set.metallic_roughness),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&set.emission),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                        },
                    ],
                });
                groups.push(group);
            }
        } else {
            let material_buffer = frame.material_buffer.as_ref()?;
            for set in material_textures.iter() {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GBuffer/MaterialBG"),
                    layout: material_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: material_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&set.albedo),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&set.normal),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&set.metallic_roughness),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&set.emission),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                        },
                    ],
                });
                groups.push(group);
            }
        }
        let groups = Arc::new(groups);
        *self.material_bind_groups.write() = Some(MaterialBindGroupCache {
            version,
            groups: groups.clone(),
        });
        Some(groups)
    }

    fn build_bundle(
        &self,
        ctx: &RenderGraphExecCtx,
        frame: &FrameGlobals,
        use_indirect: bool,
    ) -> Option<wgpu::RenderBundle> {
        let pipeline = self.pipeline.read();
        let pipeline = pipeline.as_ref()?;
        let binding_backend = frame.binding_backend;

        let camera_bg = self.create_camera_bind_group(ctx.device(), frame)?;
        let constants_bg = self.create_constants_bind_group(ctx.device(), frame)?;
        let material_bg = if binding_backend == BindingBackendKind::BindGroups {
            None
        } else {
            Some(self.create_material_bind_group_bindless(ctx.device(), frame)?)
        };
        let material_groups = if binding_backend == BindingBackendKind::BindGroups {
            Some(self.ensure_material_bind_groups(ctx.device(), frame)?)
        } else {
            None
        };
        let instances = frame.gbuffer_instances.as_ref()?;

        let color_formats = self.formats.color_format_options();
        let mut encoder =
            ctx.device()
                .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: Some("GBuffer/Bundle"),
                    color_formats: &color_formats,
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: self.formats.depth,
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: 1,
                    multiview: None,
                });

        encoder.set_pipeline(pipeline);
        encoder.set_bind_group(0, &camera_bg, &[]);
        if let Some(material_bg) = material_bg.as_ref() {
            encoder.set_bind_group(1, material_bg, &[]);
        }
        encoder.set_bind_group(2, &constants_bg, &[]);
        encoder.set_vertex_buffer(1, instances.buffer.slice(..));

        if use_indirect {
            let indirect = frame.gbuffer_indirect.as_ref()?;
            let mut last_material_idx = None;
            let mut last_vertex = None;
            let mut last_index = None;
            for draw in frame.gpu_draws.iter() {
                if let Some(groups) = material_groups.as_ref() {
                    let material_idx = draw.material_id as usize;
                    if last_material_idx != Some(material_idx) {
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        encoder.set_bind_group(1, material_bg, &[]);
                        last_material_idx = Some(material_idx);
                    }
                }
                let vertex = ctx
                    .rpctx
                    .pool
                    .entry(draw.vertex)
                    .and_then(|e| e.buffer.as_ref());
                let index = ctx
                    .rpctx
                    .pool
                    .entry(draw.index)
                    .and_then(|e| e.buffer.as_ref());
                let (Some(vertex), Some(index)) = (vertex, index) else {
                    continue;
                };
                if last_vertex != Some(draw.vertex) {
                    encoder.set_vertex_buffer(0, vertex.slice(..));
                    last_vertex = Some(draw.vertex);
                }
                if last_index != Some(draw.index) {
                    encoder.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                    last_index = Some(draw.index);
                }
                encoder.draw_indexed_indirect(indirect, draw.indirect_offset);
            }
        } else {
            let mut last_material_idx = None;
            let mut last_vertex = None;
            let mut last_index = None;
            for batch in frame.gbuffer_batches.iter() {
                let vertex = ctx
                    .rpctx
                    .pool
                    .entry(batch.vertex)
                    .and_then(|e| e.buffer.as_ref());
                let index = ctx
                    .rpctx
                    .pool
                    .entry(batch.index)
                    .and_then(|e| e.buffer.as_ref());
                let (Some(vertex), Some(index)) = (vertex, index) else {
                    continue;
                };
                if let Some(groups) = material_groups.as_ref() {
                    let material_idx = batch.material_id as usize;
                    if last_material_idx != Some(material_idx) {
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        encoder.set_bind_group(1, material_bg, &[]);
                        last_material_idx = Some(material_idx);
                    }
                }
                if last_vertex != Some(batch.vertex) {
                    encoder.set_vertex_buffer(0, vertex.slice(..));
                    last_vertex = Some(batch.vertex);
                }
                if last_index != Some(batch.index) {
                    encoder.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                    last_index = Some(batch.index);
                }
                encoder.draw_indexed(0..batch.index_count, 0, batch.instance_range.clone());
            }
        }

        Some(encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("GBuffer/Bundle"),
        }))
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &'static str {
        "GBuffer"
    }

    fn clear_cached_bundles(&self) {
        self.clear_bundle_cache();
    }

    fn invalidate_cached_bundles_for_resources(&self, resources: &[ResourceId]) -> bool {
        self.invalidate_bundles_for_resources(resources)
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.write(self.outputs.normal);
        ctx.write(self.outputs.albedo);
        ctx.write(self.outputs.mra);
        ctx.write(self.outputs.emission);
        ctx.write(self.outputs.depth);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        let binding_backend = frame.binding_backend;
        let mut use_mesh =
            frame.render_config.use_mesh_shaders && frame.device_caps.supports_mesh_pipeline();
        let use_indirect = frame.render_config.gpu_driven
            && frame.gbuffer_indirect.is_some()
            && !frame.gpu_draws.is_empty();
        let use_multi_draw = use_indirect
            && frame.render_config.gpu_multi_draw_indirect
            && frame.device_caps.supports_multi_draw_indirect();
        let has_materials = frame.material_buffer.is_some()
            && (binding_backend != BindingBackendKind::BindGroups
                || frame.material_textures.is_some());
        let draw_enabled = frame.render_config.gbuffer_pass
            && frame.device_caps.supports_vertex_storage()
            && has_materials
            && frame.gbuffer_instances.is_some()
            && (use_indirect || !frame.gbuffer_batches.is_empty());
        if use_mesh && draw_enabled {
            let max_storage = frame.device_caps.limits.max_storage_buffer_binding_size as u64;
            let instances_ok = frame
                .gbuffer_instances
                .as_ref()
                .is_some_and(|instances| instances.buffer.size() <= max_storage);
            if instances_ok {
                let mut buffers_ok = true;
                let buffer_ok = |pool: &GpuResourcePool, id: ResourceId| -> bool {
                    pool.entry(id)
                        .is_some_and(|entry| entry.desc_size_bytes <= max_storage)
                };
                if use_indirect {
                    for draw in frame.gpu_draws.iter() {
                        if draw.mesh_task_count == 0 {
                            continue;
                        }
                        if !buffer_ok(ctx.rpctx.pool, draw.vertex)
                            || !buffer_ok(ctx.rpctx.pool, draw.meshlet_descs)
                            || !buffer_ok(ctx.rpctx.pool, draw.meshlet_vertices)
                            || !buffer_ok(ctx.rpctx.pool, draw.meshlet_indices)
                        {
                            buffers_ok = false;
                            break;
                        }
                    }
                } else {
                    for batch in frame.gbuffer_batches.iter() {
                        if !buffer_ok(ctx.rpctx.pool, batch.vertex)
                            || !buffer_ok(ctx.rpctx.pool, batch.meshlet_descs)
                            || !buffer_ok(ctx.rpctx.pool, batch.meshlet_vertices)
                            || !buffer_ok(ctx.rpctx.pool, batch.meshlet_indices)
                        {
                            buffers_ok = false;
                            break;
                        }
                    }
                }
                if !buffers_ok {
                    use_mesh = false;
                }
            } else {
                use_mesh = false;
            }
        }
        let bundles_active = frame.gbuffer_render_bundles && draw_enabled && !use_mesh;
        if !bundles_active {
            let has_cached = self
                .bundle_cache
                .read()
                .iter()
                .any(|slot| slot.bundle.is_some() || slot.key.is_some());
            if has_cached {
                let mut cache = self.bundle_cache.write();
                for slot in cache.iter_mut() {
                    slot.key = None;
                    slot.bundle = None;
                    slot.resources.clear();
                }
            }
        }
        if draw_enabled {
            if use_mesh {
                self.ensure_mesh_pipeline(ctx.device(), binding_backend, frame.texture_array_size);
            } else {
                self.ensure_pipeline(ctx.device(), binding_backend, frame.texture_array_size);
            }
        }
        let cached_bundle = if draw_enabled && bundles_active {
            let frames_in_flight = frame.render_config.frames_in_flight.max(1) as usize;
            self.ensure_bundle_cache(frames_in_flight);
            let slot_index = (frame.frame_index as usize) % frames_in_flight;
            let mut cache = self.bundle_cache.write();
            let slot = &mut cache[slot_index];
            if slot.key != Some(frame.gbuffer_bundle_key) || slot.bundle.is_none() {
                slot.key = Some(frame.gbuffer_bundle_key);
                slot.bundle = self.build_bundle(ctx, frame.as_ref(), use_indirect);
                slot.resources = Self::gather_bundle_resources(frame.as_ref(), use_indirect);
            }
            slot.bundle.clone()
        } else {
            None
        };
        let (camera_bg, constants_bg, material_bg, material_groups) = if draw_enabled
            && cached_bundle.is_none()
        {
            if binding_backend == BindingBackendKind::BindGroups {
                let camera_bg = match self.create_camera_bind_group(ctx.device(), frame.as_ref()) {
                    Some(bg) => bg,
                    None => return,
                };
                let constants_bg =
                    match self.create_constants_bind_group(ctx.device(), frame.as_ref()) {
                        Some(bg) => bg,
                        None => return,
                    };
                let material_groups =
                    match self.ensure_material_bind_groups(ctx.device(), frame.as_ref()) {
                        Some(groups) => groups,
                        None => return,
                    };
                (
                    Some(camera_bg),
                    Some(constants_bg),
                    None,
                    Some(material_groups),
                )
            } else {
                let (camera_bg, material_bg, constants_bg) =
                    match self.create_bind_groups_bindless(ctx.device(), frame.as_ref()) {
                        Some(groups) => groups,
                        None => return,
                    };
                (Some(camera_bg), Some(constants_bg), Some(material_bg), None)
            }
        } else {
            (None, None, None, None)
        };

        let normal_view = self.ensure_target(
            ctx,
            self.outputs.normal,
            self.formats.normal,
            "gbuffer-normal",
            frame.render_config.use_transient_textures,
        );
        let albedo_view = self.ensure_target(
            ctx,
            self.outputs.albedo,
            self.formats.albedo,
            "gbuffer-albedo",
            frame.render_config.use_transient_textures,
        );
        let emission_view = self.ensure_target(
            ctx,
            self.outputs.emission,
            self.formats.emission,
            "gbuffer-emission",
            frame.render_config.use_transient_textures,
        );
        let mra_view = self.ensure_target(
            ctx,
            self.outputs.mra,
            self.formats.mra,
            "gbuffer-mra",
            frame.render_config.use_transient_textures,
        );
        let depth_view = self.ensure_target(
            ctx,
            self.outputs.depth,
            self.formats.depth,
            "gbuffer-depth",
            frame.render_config.use_transient_textures,
        );
        let depth_copy_view = if self.formats.output_depth_copy {
            Some(self.ensure_target(
                ctx,
                self.outputs.depth_copy,
                self.formats.depth_copy,
                "gbuffer-depth-copy",
                frame.render_config.use_transient_textures,
            ))
        } else {
            None
        };

        let mut color_attachments =
            Vec::with_capacity(if self.formats.output_depth_copy { 5 } else { 4 });
        if let Some(depth_copy_view) = depth_copy_view.as_ref() {
            color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                view: depth_copy_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }));
        }
        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
            view: &normal_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        }));
        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
            view: &albedo_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        }));
        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
            view: &emission_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        }));
        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
            view: &mra_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        }));

        let depth_attachment = Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });

        let device = ctx.device().clone();
        let queue = ctx.queue().clone();

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/GBuffer"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if !draw_enabled {
            return;
        }

        if let Some(bundle) = cached_bundle {
            pass.execute_bundles(std::iter::once(&bundle));
            return;
        }

        let camera_bg = match camera_bg.as_ref() {
            Some(bg) => bg,
            None => return,
        };
        let constants_bg = match constants_bg.as_ref() {
            Some(bg) => bg,
            None => return,
        };

        if use_mesh {
            let instances = match frame.gbuffer_instances.as_ref() {
                Some(buf) => buf,
                None => return,
            };
            let mesh_pipeline = self.mesh_pipeline.read();
            let mesh_pipeline = match mesh_pipeline.as_ref() {
                Some(pipeline) => pipeline,
                None => return,
            };
            let mesh_bgl = self.mesh_bgl.read();
            let mesh_bgl = match mesh_bgl.as_ref() {
                Some(layout) => layout,
                None => return,
            };

            let params_size = std::mem::size_of::<MeshDrawParams>() as u64;
            let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
            let params_stride = wgpu::util::align_to(params_size, alignment);

            let mut direct_tilings: Vec<MeshTaskTiling> = Vec::new();
            let total_tiles = if use_indirect {
                frame
                    .gpu_draws
                    .iter()
                    .map(|draw| draw.mesh_task_count)
                    .sum()
            } else {
                let limits = &frame.device_caps.limits;
                direct_tilings.reserve(frame.gbuffer_batches.len());
                let mut total = 0u32;
                for batch in frame.gbuffer_batches.iter() {
                    let instance_count = batch
                        .instance_range
                        .end
                        .saturating_sub(batch.instance_range.start);
                    let tiling = mesh_task_tiling(limits, batch.meshlet_count, instance_count);
                    total = total.saturating_add(tiling.task_count);
                    direct_tilings.push(tiling);
                }
                total
            };

            if total_tiles == 0 {
                return;
            }

            let needed = params_stride * total_tiles as u64;
            let params_buffer = {
                let mut buffer_guard = self.mesh_params.write();
                let capacity = self.mesh_params_capacity.load(Ordering::Relaxed);
                if buffer_guard.is_none() || capacity < needed {
                    let new_capacity = needed.max(params_stride).next_power_of_two();
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("GBuffer/MeshDrawParams"),
                        size: new_capacity,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    *buffer_guard = Some(buffer);
                    self.mesh_params_capacity
                        .store(new_capacity, Ordering::Relaxed);
                }
                buffer_guard.as_ref().unwrap().clone()
            };

            let occlusion_enabled = frame.render_config.occlusion_culling
                && frame.occlusion_camera_stable
                && frame.hiz_view.is_some();
            let flags =
                (frame.render_config.frustum_culling as u32) | ((occlusion_enabled as u32) << 1);
            let depth_bias = frame.render_config.gpu_cull_depth_bias.max(0.0);
            let rect_pad = frame.render_config.gpu_cull_rect_pad.max(0.0);

            let mut params_bytes = vec![0u8; needed as usize];
            let mut tile_cursor = 0u32;
            if use_indirect {
                for draw in frame.gpu_draws.iter() {
                    if draw.mesh_task_count == 0 {
                        continue;
                    }
                    let tile_meshlets = draw.mesh_task_tile_meshlets;
                    let tile_instances = draw.mesh_task_tile_instances;
                    if tile_meshlets == 0 || tile_instances == 0 {
                        continue;
                    }
                    let tiles_x = (draw.meshlet_count + tile_meshlets - 1) / tile_meshlets;
                    let tiles_y = (draw.instance_capacity + tile_instances - 1) / tile_instances;

                    for tile_y in 0..tiles_y {
                        let instance_base = draw.instance_base + tile_y * tile_instances;
                        let instance_count = draw
                            .instance_capacity
                            .saturating_sub(tile_y * tile_instances)
                            .min(tile_instances);
                        if instance_count == 0 {
                            continue;
                        }
                        for tile_x in 0..tiles_x {
                            let meshlet_base = tile_x * tile_meshlets;
                            let meshlet_count = draw
                                .meshlet_count
                                .saturating_sub(meshlet_base)
                                .min(tile_meshlets);
                            if meshlet_count == 0 {
                                continue;
                            }
                            let params = MeshDrawParams {
                                instance_base,
                                instance_count,
                                meshlet_base,
                                meshlet_count,
                                flags,
                                _pad0: 0,
                                _pad1: 0,
                                _pad2: 0,
                                depth_bias,
                                rect_pad,
                                _pad3: [0.0; 2],
                            };
                            let offset = (tile_cursor as u64) * params_stride;
                            let start = offset as usize;
                            params_bytes[start..start + params_size as usize]
                                .copy_from_slice(bytemuck::bytes_of(&params));
                            tile_cursor = tile_cursor.saturating_add(1);
                        }
                    }
                }
            } else {
                for (idx, batch) in frame.gbuffer_batches.iter().enumerate() {
                    let tiling = direct_tilings[idx];
                    if tiling.task_count == 0 {
                        continue;
                    }
                    for tile_y in 0..tiling.tiles_y {
                        let instance_base =
                            batch.instance_range.start + tile_y * tiling.tile_instances;
                        let instance_count = batch
                            .instance_range
                            .end
                            .saturating_sub(
                                batch.instance_range.start + tile_y * tiling.tile_instances,
                            )
                            .min(tiling.tile_instances);
                        if instance_count == 0 {
                            continue;
                        }
                        for tile_x in 0..tiling.tiles_x {
                            let meshlet_base = tile_x * tiling.tile_meshlets;
                            let meshlet_count = batch
                                .meshlet_count
                                .saturating_sub(meshlet_base)
                                .min(tiling.tile_meshlets);
                            if meshlet_count == 0 {
                                continue;
                            }
                            let params = MeshDrawParams {
                                instance_base,
                                instance_count,
                                meshlet_base,
                                meshlet_count,
                                flags,
                                _pad0: 0,
                                _pad1: 0,
                                _pad2: 0,
                                depth_bias,
                                rect_pad,
                                _pad3: [0.0; 2],
                            };
                            let offset = (tile_cursor as u64) * params_stride;
                            let start = offset as usize;
                            params_bytes[start..start + params_size as usize]
                                .copy_from_slice(bytemuck::bytes_of(&params));
                            tile_cursor = tile_cursor.saturating_add(1);
                        }
                    }
                }
            }
            queue.write_buffer(&params_buffer, 0, &params_bytes);

            pass.set_pipeline(mesh_pipeline);
            pass.set_bind_group(0, camera_bg, &[]);
            if binding_backend != BindingBackendKind::BindGroups {
                let material_bg = match material_bg.as_ref() {
                    Some(bg) => bg,
                    None => return,
                };
                pass.set_bind_group(1, material_bg, &[]);
            }
            pass.set_bind_group(2, constants_bg, &[]);

            let hiz_view = frame.hiz_view.as_ref().unwrap_or(&frame.fallback_view);
            let task_stride = std::mem::size_of::<wgpu::util::DispatchIndirectArgs>() as u64;
            let mut params_tile_index = 0u32;
            if use_indirect {
                let mesh_tasks = match frame.gbuffer_mesh_tasks.as_ref() {
                    Some(buf) => buf,
                    None => return,
                };
                let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                    match material_groups.as_ref() {
                        Some(groups) => Some(groups),
                        None => return,
                    }
                } else {
                    None
                };
                for draw in frame.gpu_draws.iter() {
                    let draw_tiles = draw.mesh_task_count;
                    if draw_tiles == 0 {
                        continue;
                    }
                    let tile_meshlets = draw.mesh_task_tile_meshlets;
                    let tile_instances = draw.mesh_task_tile_instances;
                    if tile_meshlets == 0 || tile_instances == 0 {
                        params_tile_index = params_tile_index.saturating_add(draw_tiles);
                        continue;
                    }
                    let tiles_x = (draw.meshlet_count + tile_meshlets - 1) / tile_meshlets;
                    let tiles_y = (draw.instance_capacity + tile_instances - 1) / tile_instances;
                    if tiles_x == 0 || tiles_y == 0 {
                        params_tile_index = params_tile_index.saturating_add(draw_tiles);
                        continue;
                    }
                    let meshlet_descs = match ctx
                        .rpctx
                        .pool
                        .entry(draw.meshlet_descs)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                    };
                    let meshlet_vertices = match ctx
                        .rpctx
                        .pool
                        .entry(draw.meshlet_vertices)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                    };
                    let meshlet_indices = match ctx
                        .rpctx
                        .pool
                        .entry(draw.meshlet_indices)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                    };
                    let vertex_data = match ctx
                        .rpctx
                        .pool
                        .entry(draw.vertex)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                    };

                    let mesh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GBuffer/MeshBG"),
                        layout: mesh_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: instances.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: meshlet_descs.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meshlet_vertices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: meshlet_indices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: vertex_data.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &params_buffer,
                                    offset: 0,
                                    size: wgpu::BufferSize::new(params_size),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(hiz_view),
                            },
                        ],
                    });

                    if let Some(groups) = material_groups.as_ref() {
                        let material_idx = draw.material_id as usize;
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        pass.set_bind_group(1, material_bg, &[]);
                    }
                    let mut draw_tile_index = 0u32;
                    for _tile_y in 0..tiles_y {
                        for _tile_x in 0..tiles_x {
                            let params_offset = (params_tile_index as u64) * params_stride;
                            if params_offset > u32::MAX as u64 {
                                return;
                            }
                            let task_offset =
                                draw.mesh_task_offset + (draw_tile_index as u64) * task_stride;
                            pass.set_bind_group(3, &mesh_bg, &[params_offset as u32]);
                            pass.draw_mesh_tasks_indirect(mesh_tasks, task_offset);
                            params_tile_index = params_tile_index.saturating_add(1);
                            draw_tile_index = draw_tile_index.saturating_add(1);
                        }
                    }
                }
            } else {
                let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                    match material_groups.as_ref() {
                        Some(groups) => Some(groups),
                        None => return,
                    }
                } else {
                    None
                };
                for (idx, batch) in frame.gbuffer_batches.iter().enumerate() {
                    let tiling = direct_tilings[idx];
                    if tiling.task_count == 0 {
                        continue;
                    }
                    if tiling.tiles_x == 0 || tiling.tiles_y == 0 {
                        params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                        continue;
                    }
                    let meshlet_descs = match ctx
                        .rpctx
                        .pool
                        .entry(batch.meshlet_descs)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                    };
                    let meshlet_vertices = match ctx
                        .rpctx
                        .pool
                        .entry(batch.meshlet_vertices)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                    };
                    let meshlet_indices = match ctx
                        .rpctx
                        .pool
                        .entry(batch.meshlet_indices)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                    };
                    let vertex_data = match ctx
                        .rpctx
                        .pool
                        .entry(batch.vertex)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                    };

                    let mesh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GBuffer/MeshBG"),
                        layout: mesh_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: instances.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: meshlet_descs.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meshlet_vertices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: meshlet_indices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: vertex_data.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &params_buffer,
                                    offset: 0,
                                    size: wgpu::BufferSize::new(params_size),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(hiz_view),
                            },
                        ],
                    });

                    if let Some(groups) = material_groups {
                        let material_idx = batch.material_id as usize;
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        pass.set_bind_group(1, material_bg, &[]);
                    }
                    for tile_y in 0..tiling.tiles_y {
                        for tile_x in 0..tiling.tiles_x {
                            let params_offset = (params_tile_index as u64) * params_stride;
                            if params_offset > u32::MAX as u64 {
                                return;
                            }
                            pass.set_bind_group(3, &mesh_bg, &[params_offset as u32]);
                            let meshlet_base = tile_x * tiling.tile_meshlets;
                            let meshlet_count = batch
                                .meshlet_count
                                .saturating_sub(meshlet_base)
                                .min(tiling.tile_meshlets);
                            let instance_base =
                                batch.instance_range.start + tile_y * tiling.tile_instances;
                            let instance_count = batch
                                .instance_range
                                .end
                                .saturating_sub(instance_base)
                                .min(tiling.tile_instances);
                            pass.draw_mesh_tasks(meshlet_count, instance_count, 1);
                            params_tile_index = params_tile_index.saturating_add(1);
                        }
                    }
                }
            }
        } else {
            pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
            pass.set_bind_group(0, camera_bg, &[]);
            if binding_backend != BindingBackendKind::BindGroups {
                let material_bg = match material_bg.as_ref() {
                    Some(bg) => bg,
                    None => return,
                };
                pass.set_bind_group(1, material_bg, &[]);
            }
            pass.set_bind_group(2, constants_bg, &[]);

            pass.set_vertex_buffer(
                1,
                frame.gbuffer_instances.as_ref().unwrap().buffer.slice(..),
            );

            if use_indirect {
                let indirect = frame.gbuffer_indirect.as_ref().unwrap();
                if use_multi_draw {
                    let draws = frame.gpu_draws.as_ref();
                    let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                        match material_groups.as_ref() {
                            Some(groups) => Some(groups),
                            None => return,
                        }
                    } else {
                        None
                    };
                    let include_material = material_groups.is_some();
                    let mut i = 0usize;
                    while i < draws.len() {
                        let draw = &draws[i];
                        let vertex = match ctx
                            .rpctx
                            .pool
                            .entry(draw.vertex)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => {
                                i += 1;
                                continue;
                            }
                        };
                        let index = match ctx
                            .rpctx
                            .pool
                            .entry(draw.index)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => {
                                i += 1;
                                continue;
                            }
                        };
                        if let Some(groups) = material_groups.as_ref() {
                            let material_idx = draw.material_id as usize;
                            let material_bg = groups
                                .get(material_idx)
                                .or_else(|| groups.first())
                                .expect("material bind groups empty");
                            pass.set_bind_group(1, material_bg, &[]);
                        }
                        pass.set_vertex_buffer(0, vertex.slice(..));
                        pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                        let count = indirect_draw_run_len(draws, i, include_material);
                        let j = i + count as usize;
                        pass.multi_draw_indexed_indirect(indirect, draw.indirect_offset, count);
                        i = j;
                    }
                } else {
                    let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                        match material_groups.as_ref() {
                            Some(groups) => Some(groups),
                            None => return,
                        }
                    } else {
                        None
                    };
                    let mut last_material_idx = None;
                    let mut last_vertex = None;
                    let mut last_index = None;
                    for draw in frame.gpu_draws.iter() {
                        let vertex = match ctx
                            .rpctx
                            .pool
                            .entry(draw.vertex)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => continue,
                        };
                        let index = match ctx
                            .rpctx
                            .pool
                            .entry(draw.index)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => continue,
                        };
                        if let Some(groups) = material_groups.as_ref() {
                            let material_idx = draw.material_id as usize;
                            if last_material_idx != Some(material_idx) {
                                let material_bg = groups
                                    .get(material_idx)
                                    .or_else(|| groups.first())
                                    .expect("material bind groups empty");
                                pass.set_bind_group(1, material_bg, &[]);
                                last_material_idx = Some(material_idx);
                            }
                        }
                        if last_vertex != Some(draw.vertex) {
                            pass.set_vertex_buffer(0, vertex.slice(..));
                            last_vertex = Some(draw.vertex);
                        }
                        if last_index != Some(draw.index) {
                            pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                            last_index = Some(draw.index);
                        }
                        pass.draw_indexed_indirect(indirect, draw.indirect_offset);
                    }
                }
            } else {
                let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                    match material_groups.as_ref() {
                        Some(groups) => Some(groups),
                        None => return,
                    }
                } else {
                    None
                };
                let mut last_material_idx = None;
                let mut last_vertex = None;
                let mut last_index = None;
                for batch in frame.gbuffer_batches.iter() {
                    let vertex = match ctx
                        .rpctx
                        .pool
                        .entry(batch.vertex)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => continue,
                    };
                    let index = match ctx
                        .rpctx
                        .pool
                        .entry(batch.index)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => continue,
                    };

                    if let Some(groups) = material_groups {
                        let material_idx = batch.material_id as usize;
                        if last_material_idx != Some(material_idx) {
                            let material_bg = groups
                                .get(material_idx)
                                .or_else(|| groups.first())
                                .expect("material bind groups empty");
                            pass.set_bind_group(1, material_bg, &[]);
                            last_material_idx = Some(material_idx);
                        }
                    }
                    if last_vertex != Some(batch.vertex) {
                        pass.set_vertex_buffer(0, vertex.slice(..));
                        last_vertex = Some(batch.vertex);
                    }
                    if last_index != Some(batch.index) {
                        pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                        last_index = Some(batch.index);
                    }
                    pass.draw_indexed(0..batch.index_count, 0, batch.instance_range.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::indirect_draw_run_len;
    use crate::graphics::graph::definition::resource_id::{ResourceId, ResourceKind};
    use crate::graphics::passes::IndirectDrawBatch;

    fn draw(vertex: u32, index: u32, material_id: u32, indirect_offset: u64) -> IndirectDrawBatch {
        IndirectDrawBatch {
            mesh_id: 0,
            lod: 0,
            material_id,
            vertex: ResourceId::new(ResourceKind::Buffer, vertex, 0),
            index: ResourceId::new(ResourceKind::Buffer, index, 0),
            meshlet_descs: ResourceId::new(ResourceKind::Buffer, 100, 0),
            meshlet_vertices: ResourceId::new(ResourceKind::Buffer, 101, 0),
            meshlet_indices: ResourceId::new(ResourceKind::Buffer, 102, 0),
            meshlet_count: 0,
            instance_base: 0,
            instance_capacity: 0,
            indirect_offset,
            mesh_task_offset: 0,
            mesh_task_count: 0,
            mesh_task_tile_meshlets: 0,
            mesh_task_tile_instances: 0,
        }
    }

    #[test]
    fn indirect_draw_run_len_groups_contiguous_mesh_runs() {
        let stride = std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;
        let draws = vec![
            draw(1, 2, 0, 0),
            draw(1, 2, 0, stride),
            draw(1, 2, 0, stride * 2),
            draw(1, 3, 0, stride * 3),
        ];

        assert_eq!(indirect_draw_run_len(&draws, 0, false), 3);
        assert_eq!(indirect_draw_run_len(&draws, 3, false), 1);
    }

    #[test]
    fn indirect_draw_run_len_respects_material_boundaries() {
        let stride = std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;
        let draws = vec![
            draw(1, 2, 0, 0),
            draw(1, 2, 1, stride),
            draw(1, 2, 1, stride * 2),
        ];

        assert_eq!(indirect_draw_run_len(&draws, 0, true), 1);
        assert_eq!(indirect_draw_run_len(&draws, 1, true), 2);
        assert_eq!(indirect_draw_run_len(&draws, 0, false), 3);
    }
}
