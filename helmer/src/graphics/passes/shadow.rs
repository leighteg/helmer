use parking_lot::RwLock;
use std::{
    any::Any,
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
};

use crate::graphics::{
    backend::binding_backend::BindingBackendKind,
    common::{
        constants::MAX_SHADOW_CASCADES,
        renderer::{
            MeshDrawParams, MeshTaskTiling, mesh_shader_visibility, mesh_task_tiling,
            transient_usage,
        },
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
    passes::{FrameGlobals, ShadowBundleKey},
};

#[derive(Clone, Copy, Debug)]
pub struct ShadowOutputs {
    pub map: ResourceId,
    pub depth: ResourceId,
}

struct MaterialBindGroupCache {
    version: u64,
    groups: Arc<Vec<wgpu::BindGroup>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowInstanceRaw {
    pub model_matrix: [[f32; 4]; 4],
    pub material_id: u32,
    pub skin_offset: u32,
    pub skin_count: u32,
    pub _pad0: u32,
}

impl ShadowInstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ShadowInstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress
                        + mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress
                        + 2 * mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

#[derive(Clone)]
pub struct ShadowPass {
    outputs: ShadowOutputs,
    format: wgpu::TextureFormat,
    shadow_map_resolution: u32,
    cascade_count: u32,
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    vp_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    material_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    rc_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    mesh_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    mesh_params: Arc<RwLock<Option<wgpu::Buffer>>>,
    mesh_params_capacity: Arc<AtomicU64>,
    aligned_mat4_size: Arc<AtomicU64>,
    texture_array_size: Arc<AtomicU32>,
    backend_kind: Arc<RwLock<Option<BindingBackendKind>>>,
    material_bind_groups: Arc<RwLock<Option<MaterialBindGroupCache>>>,
    bundle_cache: Arc<RwLock<Vec<ShadowBundleSlot>>>,
}

#[derive(Default)]
struct ShadowBundleSlot {
    key: Option<ShadowBundleKey>,
    bundles: Option<Vec<wgpu::RenderBundle>>,
    resources: Vec<ResourceId>,
    draw_signature: u64,
}

fn shadow_format_is_rg(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rg32Float | wgpu::TextureFormat::Rg16Float
    )
}

impl ShadowPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        format: wgpu::TextureFormat,
        shadow_map_resolution: u32,
        shadow_cascade_count: u32,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
    ) -> Self {
        let shadow_map_resolution = shadow_map_resolution.max(1);
        let cascade_count = shadow_cascade_count.clamp(1, MAX_SHADOW_CASCADES as u32);
        let shadow_usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            use_transient_textures,
        );
        let (shadow_desc, mut shadow_hints) = ResourceDesc::Texture2D {
            width: shadow_map_resolution,
            height: shadow_map_resolution,
            mip_levels: 1,
            layers: cascade_count,
            format,
            usage: shadow_usage,
        }
        .with_hints();
        if use_transient_aliasing {
            shadow_hints.flags |= ResourceFlags::TRANSIENT;
        }
        let map = pool.create_logical(shadow_desc, Some(shadow_hints), 0, None);

        let depth_usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            use_transient_textures,
        );
        let (depth_desc, mut depth_hints) = ResourceDesc::Texture2D {
            width: shadow_map_resolution,
            height: shadow_map_resolution,
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Depth32Float,
            usage: depth_usage,
        }
        .with_hints();
        if use_transient_aliasing {
            depth_hints.flags |= ResourceFlags::TRANSIENT;
        }
        let depth = pool.create_logical(depth_desc, Some(depth_hints), 0, None);

        Self {
            outputs: ShadowOutputs { map, depth },
            format,
            shadow_map_resolution,
            cascade_count,
            pipeline: Arc::new(RwLock::new(None)),
            mesh_pipeline: Arc::new(RwLock::new(None)),
            vp_bgl: Arc::new(RwLock::new(None)),
            material_bgl: Arc::new(RwLock::new(None)),
            rc_bgl: Arc::new(RwLock::new(None)),
            mesh_bgl: Arc::new(RwLock::new(None)),
            mesh_params: Arc::new(RwLock::new(None)),
            mesh_params_capacity: Arc::new(AtomicU64::new(0)),
            aligned_mat4_size: Arc::new(AtomicU64::new(0)),
            texture_array_size: Arc::new(AtomicU32::new(1)),
            backend_kind: Arc::new(RwLock::new(None)),
            material_bind_groups: Arc::new(RwLock::new(None)),
            bundle_cache: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn outputs(&self) -> ShadowOutputs {
        self.outputs
    }

    pub fn clear_bundle_cache(&self) {
        let mut cache = self.bundle_cache.write();
        for slot in cache.iter_mut() {
            slot.key = None;
            slot.bundles = None;
            slot.resources.clear();
            slot.draw_signature = 0;
        }
    }

    pub fn invalidate_bundles_for_resources(&self, evicted: &[ResourceId]) {
        if evicted.is_empty() {
            return;
        }

        let mut cache = self.bundle_cache.write();
        for slot in cache.iter_mut() {
            if slot.bundles.is_none() || slot.resources.is_empty() {
                continue;
            }
            let should_clear = evicted
                .iter()
                .any(|id| slot.resources.binary_search(id).is_ok());
            if should_clear {
                slot.key = None;
                slot.bundles = None;
                slot.resources.clear();
                slot.draw_signature = 0;
            }
        }
    }

    fn gather_bundle_resources(frame: &FrameGlobals, use_indirect: bool) -> Vec<ResourceId> {
        let mut resources = Vec::new();
        if use_indirect {
            for draw in frame.gpu_draws.iter() {
                resources.push(draw.vertex);
                resources.push(draw.index);
            }
        } else {
            for batch in frame.shadow_batches.iter() {
                resources.push(batch.vertex);
                resources.push(batch.index);
            }
        }
        resources.sort_unstable();
        resources.dedup();
        resources
    }

    fn bundle_signature(pool: &GpuResourcePool, frame: &FrameGlobals, use_indirect: bool) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        let mut mix = |value: u64| {
            hash ^= value;
            hash = hash.wrapping_mul(0x100000001b3);
        };

        mix(use_indirect as u64);
        if use_indirect {
            for draw in frame.gpu_draws.iter() {
                mix(draw.vertex.raw());
                mix(pool.binding_version(draw.vertex));
                mix(draw.index.raw());
                mix(pool.binding_version(draw.index));
                mix(draw.indirect_offset);
            }
        } else {
            for batch in frame.shadow_batches.iter() {
                mix(batch.vertex.raw());
                mix(pool.binding_version(batch.vertex));
                mix(batch.index.raw());
                mix(pool.binding_version(batch.index));
                mix(batch.index_count as u64);
                mix(batch.instance_range.start as u64);
                mix(batch.instance_range.end as u64);
            }
        }

        hash
    }

    fn ensure_targets(
        &self,
        ctx: &mut RenderGraphExecCtx,
    ) -> (wgpu::TextureView, wgpu::TextureView) {
        // Shadow map (array)
        let map_view = {
            let resource_desc = ResourceDesc::Texture2D {
                width: self.shadow_map_resolution,
                height: self.shadow_map_resolution,
                mip_levels: 1,
                layers: self.cascade_count,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            };
            let needs_create = ctx
                .rpctx
                .pool
                .entry(self.outputs.map)
                .map(|e| e.texture.is_none())
                .unwrap_or(true);

            let view = if needs_create {
                let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("ShadowMap"),
                    size: wgpu::Extent3d {
                        width: self.shadow_map_resolution,
                        height: self.shadow_map_resolution,
                        depth_or_array_layers: self.cascade_count,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("ShadowMapView"),
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                });
                ctx.rpctx.pool.realize_texture(
                    self.outputs.map,
                    resource_desc,
                    texture,
                    view.clone(),
                    ctx.rpctx.frame_index,
                );
                Some(view)
            } else {
                ctx.rpctx
                    .pool
                    .entry(self.outputs.map)
                    .and_then(|e| e.texture.as_ref())
                    .map(|tex| {
                        tex.create_view(&wgpu::TextureViewDescriptor {
                            label: Some("ShadowMapView"),
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            dimension: Some(wgpu::TextureViewDimension::D2Array),
                            ..Default::default()
                        })
                    })
            }
            .unwrap();
            if let Some(entry) = ctx.rpctx.pool.entry_mut(self.outputs.map) {
                entry.texture_view = Some(view.clone());
            }
            ctx.rpctx
                .pool
                .mark_resident(self.outputs.map, ctx.rpctx.frame_index);
            view
        };

        let depth_view = {
            let resource_desc = ResourceDesc::Texture2D {
                width: self.shadow_map_resolution,
                height: self.shadow_map_resolution,
                mip_levels: 1,
                layers: 1,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            };
            let needs_create = ctx
                .rpctx
                .pool
                .entry(self.outputs.depth)
                .map(|e| e.texture.is_none())
                .unwrap_or(true);

            let view = if needs_create {
                let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("ShadowDepth"),
                    size: wgpu::Extent3d {
                        width: self.shadow_map_resolution,
                        height: self.shadow_map_resolution,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    ..Default::default()
                });
                ctx.rpctx.pool.realize_texture(
                    self.outputs.depth,
                    resource_desc,
                    texture,
                    view.clone(),
                    ctx.rpctx.frame_index,
                );
                Some(view)
            } else {
                ctx.rpctx
                    .pool
                    .entry(self.outputs.depth)
                    .and_then(|e| e.texture.as_ref())
                    .map(|tex| {
                        tex.create_view(&wgpu::TextureViewDescriptor {
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            ..Default::default()
                        })
                    })
            }
            .unwrap();
            if let Some(entry) = ctx.rpctx.pool.entry_mut(self.outputs.depth) {
                entry.texture_view = Some(view.clone());
            }
            ctx.rpctx
                .pool
                .mark_resident(self.outputs.depth, ctx.rpctx.frame_index);
            view
        };

        (map_view, depth_view)
    }

    fn ensure_backend(&self, binding_backend: BindingBackendKind) {
        let mut current = self.backend_kind.write();
        if current.map_or(true, |kind| kind != binding_backend) {
            *self.pipeline.write() = None;
            *self.mesh_pipeline.write() = None;
            *self.vp_bgl.write() = None;
            *self.material_bgl.write() = None;
            *self.rc_bgl.write() = None;
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

        let output_is_rg = shadow_format_is_rg(self.format);
        let fragment_entry = if output_is_rg {
            "fs_main"
        } else {
            "fs_main_rgba"
        };
        let write_mask = if output_is_rg {
            wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN
        } else {
            wgpu::ColorWrites::ALL
        };
        let mesh_stage = mesh_shader_visibility(device);
        let shader = if binding_backend == BindingBackendKind::BindGroups {
            device.create_shader_module(wgpu::include_wgsl!("../shaders/shadow_bindgroups.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("../shaders/shadow.wgsl"))
        };

        let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        self.aligned_mat4_size.store(
            wgpu::util::align_to(mat4_size, alignment),
            Ordering::Relaxed,
        );

        let vp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow/VPBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | mesh_stage,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(mat4_size),
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
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
            label: Some("Shadow/MaterialBGL"),
            entries: &material_entries,
        });

        let rc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow/RCBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                        crate::graphics::common::renderer::ShaderConstants,
                    >() as u64),
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow/PipelineLayout"),
            bind_group_layouts: &[&vp_bgl, &material_bgl, &rc_bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow/Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    crate::graphics::common::renderer::Vertex::desc(),
                    ShadowInstanceRaw::desc(),
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
                    blend: None,
                    write_mask,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.vp_bgl.write() = Some(vp_bgl);
        *self.material_bgl.write() = Some(material_bgl);
        *self.rc_bgl.write() = Some(rc_bgl);
        *self.pipeline.write() = Some(pipeline);
    }

    fn ensure_mesh_pipeline(
        &self,
        device: &wgpu::Device,
        binding_backend: BindingBackendKind,
        texture_array_size: u32,
    ) {
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

        if mesh_shader_visibility(device).is_empty() {
            return;
        }

        self.ensure_pipeline(device, binding_backend, texture_array_size);

        let (vp_bgl, material_bgl, rc_bgl) = (
            self.vp_bgl.read(),
            self.material_bgl.read(),
            self.rc_bgl.read(),
        );
        let (vp_bgl, material_bgl, rc_bgl) = (
            vp_bgl.as_ref().unwrap(),
            material_bgl.as_ref().unwrap(),
            rc_bgl.as_ref().unwrap(),
        );

        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow/MeshBGL"),
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
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow/MeshPipelineLayout"),
            bind_group_layouts: &[vp_bgl, material_bgl, rc_bgl, &mesh_bgl],
            immediate_size: 0,
        });

        let shader = if binding_backend == BindingBackendKind::BindGroups {
            device.create_shader_module(wgpu::include_wgsl!(
                "../shaders/shadow_mesh_bindgroups.wgsl"
            ))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("../shaders/shadow_mesh.wgsl"))
        };
        let write_mask = if shadow_format_is_rg(self.format) {
            wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN
        } else {
            wgpu::ColorWrites::ALL
        };

        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("Shadow/MeshPipeline"),
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
                    format: self.format,
                    blend: None,
                    write_mask,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        *self.mesh_bgl.write() = Some(mesh_bgl);
        *self.mesh_pipeline.write() = Some(pipeline);
    }

    fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let vp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow/VPBG"),
            layout: self.vp_bgl.read().as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: frame.shadow_matrices_buffer.as_ref().unwrap(),
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<[[f32; 4]; 4]>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.skin_palette_buffer.as_entire_binding(),
                },
            ],
        });

        let rc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow/RCBG"),
            layout: self.rc_bgl.read().as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.render_constants_buffer.as_entire_binding(),
            }],
        });

        (vp_bg, rc_bg)
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

        Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow/MaterialBG"),
                layout: material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame
                            .material_buffer
                            .as_ref()
                            .expect("material buffer missing")
                            .as_entire_binding(),
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
            }),
        )
    }

    fn ensure_material_bind_groups(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<Arc<Vec<wgpu::BindGroup>>> {
        let material_buffer = frame.material_buffer.as_ref()?;
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
        for set in material_textures.iter() {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow/MaterialBG"),
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
                        resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                    },
                ],
            });
            groups.push(group);
        }
        let groups = Arc::new(groups);
        *self.material_bind_groups.write() = Some(MaterialBindGroupCache {
            version,
            groups: groups.clone(),
        });
        Some(groups)
    }

    fn ensure_bundle_cache(&self, frames_in_flight: usize) {
        let slots = frames_in_flight.max(1);
        let mut cache = self.bundle_cache.write();
        if cache.len() != slots {
            cache.clear();
            cache.resize_with(slots, ShadowBundleSlot::default);
        }
    }

    fn build_bundles(
        &self,
        ctx: &RenderGraphExecCtx,
        frame: &FrameGlobals,
        binding_backend: BindingBackendKind,
        use_indirect: bool,
    ) -> Option<Vec<wgpu::RenderBundle>> {
        let pipeline = self.pipeline.read();
        let pipeline = pipeline.as_ref()?;
        let (vp_bg, rc_bg) = self.create_bind_groups(ctx.device(), frame);
        let material_bg = if binding_backend == BindingBackendKind::BindGroups {
            None
        } else {
            self.create_material_bind_group_bindless(ctx.device(), frame)
        };
        let material_groups = if binding_backend == BindingBackendKind::BindGroups {
            self.ensure_material_bind_groups(ctx.device(), frame)
        } else {
            None
        };
        let instances = frame.shadow_instances.as_ref()?;

        let aligned_size = self.aligned_mat4_size.load(Ordering::Relaxed) as u32;
        let color_formats = [Some(self.format)];
        let mut bundles = Vec::with_capacity(self.cascade_count as usize);

        for cascade in 0..self.cascade_count {
            let mut encoder =
                ctx.device()
                    .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                        label: Some("Shadow/Bundle"),
                        color_formats: &color_formats,
                        depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                            format: wgpu::TextureFormat::Depth32Float,
                            depth_read_only: false,
                            stencil_read_only: true,
                        }),
                        sample_count: 1,
                        multiview: None,
                    });

            encoder.set_pipeline(pipeline);
            encoder.set_bind_group(0, &vp_bg, &[aligned_size.saturating_mul(cascade as u32)]);
            if binding_backend != BindingBackendKind::BindGroups {
                let material_bg = material_bg.as_ref()?;
                encoder.set_bind_group(1, material_bg, &[]);
            }
            encoder.set_bind_group(2, &rc_bg, &[]);
            encoder.set_vertex_buffer(1, instances.buffer.slice(..));

            if use_indirect {
                let indirect = frame.shadow_indirect.as_ref()?;
                let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                    material_groups.as_ref()
                } else {
                    None
                };
                for draw in frame.gpu_draws.iter() {
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
                    if let Some(groups) = material_groups {
                        let material_idx = draw.material_id as usize;
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        encoder.set_bind_group(1, material_bg, &[]);
                    }
                    encoder.set_vertex_buffer(0, vertex.slice(..));
                    encoder.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                    encoder.draw_indexed_indirect(indirect, draw.indirect_offset);
                }
            } else {
                let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                    material_groups.as_ref()
                } else {
                    None
                };
                for batch in frame.shadow_batches.iter() {
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
                    if let Some(groups) = material_groups {
                        let material_idx = batch.material_id as usize;
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        encoder.set_bind_group(1, material_bg, &[]);
                    }
                    encoder.set_vertex_buffer(0, vertex.slice(..));
                    encoder.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                    encoder.draw_indexed(0..batch.index_count, 0, batch.instance_range.clone());
                }
            }

            bundles.push(encoder.finish(&wgpu::RenderBundleDescriptor {
                label: Some("Shadow/Bundle"),
            }));
        }

        Some(bundles)
    }
}

impl RenderPass for ShadowPass {
    fn name(&self) -> &'static str {
        "ShadowPass"
    }

    fn user_data(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.write(self.outputs.map);
        ctx.write(self.outputs.depth);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };

        let device = ctx.device().clone();
        let queue = ctx.queue().clone();
        let binding_backend = frame.binding_backend;
        let texture_array_size = frame.texture_array_size;
        let has_materials = frame.material_buffer.is_some()
            && (binding_backend != BindingBackendKind::BindGroups
                || frame.material_textures.is_some());

        let mut use_mesh =
            frame.render_config.use_mesh_shaders && frame.device_caps.supports_mesh_pipeline();
        let use_indirect = frame.render_config.gpu_driven
            && frame.shadow_indirect.is_some()
            && !frame.gpu_draws.is_empty();
        let use_mesh_indirect = frame.render_config.gpu_driven
            && frame.shadow_mesh_tasks.is_some()
            && !frame.gpu_draws.is_empty();
        let use_multi_draw = use_indirect
            && frame.render_config.gpu_multi_draw_indirect
            && frame.device_caps.supports_multi_draw_indirect()
            && binding_backend != BindingBackendKind::BindGroups;
        let (_map_view, depth_view) = self.ensure_targets(ctx);

        let shadow_texture = ctx
            .rpctx
            .pool
            .entry(self.outputs.map)
            .and_then(|e| e.texture.as_ref());

        let shadow_texture = match shadow_texture {
            Some(t) => t,
            None => return,
        };

        let mesh_draw_enabled = frame.render_config.shadow_pass
            && has_materials
            && frame.shadow_instances.is_some()
            && frame.shadow_matrices_buffer.is_some()
            && (use_mesh_indirect || !frame.shadow_batches.is_empty());
        if use_mesh && mesh_draw_enabled {
            let max_storage = frame.device_caps.limits.max_storage_buffer_binding_size as u64;
            let instances_ok = frame
                .shadow_instances
                .as_ref()
                .is_some_and(|instances| instances.buffer.size() <= max_storage);
            if instances_ok {
                let mut buffers_ok = true;
                let buffer_ok = |pool: &GpuResourcePool, id: ResourceId| -> bool {
                    pool.entry(id)
                        .is_some_and(|entry| entry.desc_size_bytes <= max_storage)
                };
                if use_mesh_indirect {
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
                    for batch in frame.shadow_batches.iter() {
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

        if use_mesh {
            let draw_enabled = mesh_draw_enabled;
            let aligned_size = self.aligned_mat4_size.load(Ordering::Relaxed) as u32;

            if !draw_enabled {
                for cascade in 0..self.cascade_count {
                    let cascade_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("ShadowCascadeView"),
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: cascade as u32,
                        array_layer_count: Some(1),
                        ..Default::default()
                    });

                    let _ = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("RenderGraph/Shadow"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &cascade_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
                return;
            }

            self.ensure_mesh_pipeline(&device, binding_backend, texture_array_size);
            let (vp_bg, rc_bg) = self.create_bind_groups(&device, frame.as_ref());
            let material_bg = if binding_backend == BindingBackendKind::BindGroups {
                None
            } else {
                self.create_material_bind_group_bindless(&device, frame.as_ref())
            };
            let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                self.ensure_material_bind_groups(&device, frame.as_ref())
            } else {
                None
            };
            if binding_backend == BindingBackendKind::BindGroups && material_groups.is_none() {
                return;
            }
            if binding_backend != BindingBackendKind::BindGroups && material_bg.is_none() {
                return;
            }
            let instances = match frame.shadow_instances.as_ref() {
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
            let total_tiles = if use_mesh_indirect {
                frame
                    .gpu_draws
                    .iter()
                    .map(|draw| draw.mesh_task_count)
                    .sum()
            } else {
                let limits = &frame.device_caps.limits;
                direct_tilings.reserve(frame.shadow_batches.len());
                let mut total = 0u32;
                for batch in frame.shadow_batches.iter() {
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
                for cascade in 0..self.cascade_count {
                    let cascade_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("ShadowCascadeView"),
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: cascade as u32,
                        array_layer_count: Some(1),
                        ..Default::default()
                    });

                    let _ = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("RenderGraph/Shadow"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &cascade_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
                return;
            }

            let needed = params_stride * total_tiles as u64;
            let params_buffer = {
                let mut buffer_guard = self.mesh_params.write();
                let capacity = self.mesh_params_capacity.load(Ordering::Relaxed);
                if buffer_guard.is_none() || capacity < needed {
                    let new_capacity = needed.max(params_stride).next_power_of_two();
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Shadow/MeshDrawParams"),
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

            let flags = frame.render_config.frustum_culling as u32;

            let mut params_bytes = vec![0u8; needed as usize];
            let mut tile_cursor = 0u32;
            if use_mesh_indirect {
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
                        for tile_x in 0..tiles_x {
                            let meshlet_base = tile_x * tile_meshlets;
                            let meshlet_count = draw
                                .meshlet_count
                                .saturating_sub(meshlet_base)
                                .min(tile_meshlets);
                            let params = MeshDrawParams {
                                instance_base,
                                instance_count,
                                meshlet_base,
                                meshlet_count,
                                flags,
                                _pad0: 0,
                                _pad1: 0,
                                _pad2: 0,
                                depth_bias: 0.0,
                                rect_pad: 0.0,
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
                for (idx, batch) in frame.shadow_batches.iter().enumerate() {
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
                        for tile_x in 0..tiling.tiles_x {
                            let meshlet_base = tile_x * tiling.tile_meshlets;
                            let meshlet_count = batch
                                .meshlet_count
                                .saturating_sub(meshlet_base)
                                .min(tiling.tile_meshlets);
                            let params = MeshDrawParams {
                                instance_base,
                                instance_count,
                                meshlet_base,
                                meshlet_count,
                                flags,
                                _pad0: 0,
                                _pad1: 0,
                                _pad2: 0,
                                depth_bias: 0.0,
                                rect_pad: 0.0,
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

            let mut mesh_bgs: Vec<Option<wgpu::BindGroup>> = Vec::new();
            if use_mesh_indirect {
                mesh_bgs.reserve(frame.gpu_draws.len());
                for draw in frame.gpu_draws.iter() {
                    if draw.mesh_task_count == 0 {
                        mesh_bgs.push(None);
                        continue;
                    }
                    let tile_meshlets = draw.mesh_task_tile_meshlets;
                    let tile_instances = draw.mesh_task_tile_instances;
                    if tile_meshlets == 0 || tile_instances == 0 {
                        mesh_bgs.push(None);
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
                            mesh_bgs.push(None);
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
                            mesh_bgs.push(None);
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
                            mesh_bgs.push(None);
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
                            mesh_bgs.push(None);
                            continue;
                        }
                    };

                    let mesh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Shadow/MeshBG"),
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
                        ],
                    });
                    mesh_bgs.push(Some(mesh_bg));
                }
            } else {
                mesh_bgs.reserve(frame.shadow_batches.len());
                for (idx, batch) in frame.shadow_batches.iter().enumerate() {
                    let tiling = direct_tilings[idx];
                    if tiling.task_count == 0 {
                        mesh_bgs.push(None);
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
                            mesh_bgs.push(None);
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
                            mesh_bgs.push(None);
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
                            mesh_bgs.push(None);
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
                            mesh_bgs.push(None);
                            continue;
                        }
                    };

                    let mesh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Shadow/MeshBG"),
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
                        ],
                    });
                    mesh_bgs.push(Some(mesh_bg));
                }
            }

            let task_stride = std::mem::size_of::<wgpu::util::DispatchIndirectArgs>() as u64;
            for cascade in 0..self.cascade_count {
                let cascade_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("ShadowCascadeView"),
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: cascade as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                });

                let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("RenderGraph/Shadow"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &cascade_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 1.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                pass.set_pipeline(mesh_pipeline);
                pass.set_bind_group(0, &vp_bg, &[aligned_size.saturating_mul(cascade as u32)]);
                if binding_backend != BindingBackendKind::BindGroups {
                    let material_bg = match material_bg.as_ref() {
                        Some(bg) => bg,
                        None => return,
                    };
                    pass.set_bind_group(1, material_bg, &[]);
                }
                pass.set_bind_group(2, &rc_bg, &[]);

                let mut params_tile_index = 0u32;
                if use_mesh_indirect {
                    let mesh_tasks = match frame.shadow_mesh_tasks.as_ref() {
                        Some(buf) => buf,
                        None => return,
                    };
                    for (draw_idx, draw) in frame.gpu_draws.iter().enumerate() {
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
                        let tiles_y =
                            (draw.instance_capacity + tile_instances - 1) / tile_instances;
                        if tiles_x == 0 || tiles_y == 0 {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                        let Some(mesh_bg) = mesh_bgs[draw_idx].as_ref() else {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        };
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
                                pass.set_bind_group(3, mesh_bg, &[params_offset as u32]);
                                pass.draw_mesh_tasks_indirect(mesh_tasks, task_offset);
                                params_tile_index = params_tile_index.saturating_add(1);
                                draw_tile_index = draw_tile_index.saturating_add(1);
                            }
                        }
                    }
                } else {
                    for (batch_idx, batch) in frame.shadow_batches.iter().enumerate() {
                        let tiling = direct_tilings[batch_idx];
                        if tiling.task_count == 0 {
                            continue;
                        }
                        if tiling.tiles_x == 0 || tiling.tiles_y == 0 {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                        let Some(mesh_bg) = mesh_bgs[batch_idx].as_ref() else {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        };
                        if let Some(groups) = material_groups.as_ref() {
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
                                pass.set_bind_group(3, mesh_bg, &[params_offset as u32]);
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
            }
            return;
        }

        self.ensure_pipeline(&device, binding_backend, texture_array_size);
        let clear_only = !frame.render_config.shadow_pass
            || !has_materials
            || frame.shadow_instances.is_none()
            || frame.shadow_matrices_buffer.is_none()
            || (!use_indirect && frame.shadow_batches.is_empty());
        let bundles_active = frame.render_config.render_bundles && !clear_only;
        if !bundles_active {
            let has_cached = self
                .bundle_cache
                .read()
                .iter()
                .any(|slot| slot.bundles.is_some() || slot.key.is_some());
            if has_cached {
                let mut cache = self.bundle_cache.write();
                for slot in cache.iter_mut() {
                    slot.key = None;
                    slot.bundles = None;
                    slot.resources.clear();
                    slot.draw_signature = 0;
                }
            }
        }

        let cached_bundles = if bundles_active {
            let draw_signature =
                Self::bundle_signature(ctx.rpctx.pool, frame.as_ref(), use_indirect);
            let frames_in_flight = frame.render_config.frames_in_flight.max(1) as usize;
            self.ensure_bundle_cache(frames_in_flight);
            let slot_index = (frame.frame_index as usize) % frames_in_flight;
            let mut cache = self.bundle_cache.write();
            let slot = &mut cache[slot_index];
            if slot.key != Some(frame.shadow_bundle_key)
                || slot.bundles.is_none()
                || slot.draw_signature != draw_signature
            {
                slot.key = Some(frame.shadow_bundle_key);
                slot.bundles =
                    self.build_bundles(ctx, frame.as_ref(), binding_backend, use_indirect);
                slot.resources = Self::gather_bundle_resources(frame.as_ref(), use_indirect);
                slot.draw_signature = draw_signature;
            }
            slot.bundles.clone()
        } else {
            None
        };
        let (vp_bg, rc_bg, material_bg, material_groups) =
            if cached_bundles.is_none() && !clear_only {
                let (vp_bg, rc_bg) = self.create_bind_groups(&device, frame.as_ref());
                if binding_backend == BindingBackendKind::BindGroups {
                    let material_groups =
                        match self.ensure_material_bind_groups(&device, frame.as_ref()) {
                            Some(groups) => groups,
                            None => return,
                        };
                    (Some(vp_bg), Some(rc_bg), None, Some(material_groups))
                } else {
                    let material_bg =
                        match self.create_material_bind_group_bindless(&device, frame.as_ref()) {
                            Some(bg) => bg,
                            None => return,
                        };
                    (Some(vp_bg), Some(rc_bg), Some(material_bg), None)
                }
            } else {
                (None, None, None, None)
            };

        // Render each cascade separately
        for cascade in 0..self.cascade_count {
            let cascade_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("ShadowCascadeView"),
                base_mip_level: 0,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: cascade as u32,
                array_layer_count: Some(1),
                ..Default::default()
            });

            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RenderGraph/Shadow"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &cascade_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if let Some(ref bundles) = cached_bundles {
                if let Some(bundle) = bundles.get(cascade as usize) {
                    pass.execute_bundles(std::iter::once(bundle));
                    continue;
                }
            }

            if let (Some(vp_bg), Some(rc_bg)) = (vp_bg.as_ref(), rc_bg.as_ref()) {
                pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
                pass.set_bind_group(
                    0,
                    vp_bg,
                    &[self.aligned_mat4_size.load(Ordering::Relaxed) as u32 * cascade as u32],
                );
                if binding_backend != BindingBackendKind::BindGroups {
                    let material_bg = material_bg.as_ref().expect("material bind group missing");
                    pass.set_bind_group(1, material_bg, &[]);
                }
                pass.set_bind_group(2, rc_bg, &[]);

                pass.set_vertex_buffer(
                    1,
                    frame.shadow_instances.as_ref().unwrap().buffer.slice(..),
                );

                if use_indirect {
                    let indirect = frame.shadow_indirect.as_ref().unwrap();
                    if use_multi_draw {
                        let stride =
                            std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;
                        let draws = frame.gpu_draws.as_ref();
                        let mut i = 0usize;
                        while i < draws.len() {
                            let draw = &draws[i];
                            let vertex_buffer = match ctx
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
                            pass.set_vertex_buffer(0, vertex_buffer.slice(..));

                            let index_buffer = match ctx
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
                            pass.set_index_buffer(
                                index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );

                            let mut count = 1u32;
                            let mut next_offset = draw.indirect_offset + stride;
                            let mut j = i + 1;
                            while j < draws.len() {
                                let next = &draws[j];
                                if next.vertex != draw.vertex
                                    || next.index != draw.index
                                    || next.indirect_offset != next_offset
                                {
                                    break;
                                }
                                count += 1;
                                next_offset += stride;
                                j += 1;
                            }
                            pass.multi_draw_indexed_indirect(indirect, draw.indirect_offset, count);
                            i = j;
                        }
                    } else {
                        for draw in frame.gpu_draws.iter() {
                            let vertex_buffer = match ctx
                                .rpctx
                                .pool
                                .entry(draw.vertex)
                                .and_then(|e| e.buffer.as_ref())
                            {
                                Some(buf) => buf,
                                None => continue,
                            };
                            pass.set_vertex_buffer(0, vertex_buffer.slice(..));

                            let index_buffer = match ctx
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
                                let material_bg = groups
                                    .get(material_idx)
                                    .or_else(|| groups.first())
                                    .expect("material bind groups empty");
                                pass.set_bind_group(1, material_bg, &[]);
                            }
                            pass.set_index_buffer(
                                index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed_indirect(indirect, draw.indirect_offset);
                        }
                    }
                } else {
                    for batch in frame.shadow_batches.iter() {
                        let vertex_buffer = match ctx
                            .rpctx
                            .pool
                            .entry(batch.vertex)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => continue,
                        };
                        pass.set_vertex_buffer(0, vertex_buffer.slice(..));

                        let index_buffer = match ctx
                            .rpctx
                            .pool
                            .entry(batch.index)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => continue,
                        };
                        if let Some(groups) = material_groups.as_ref() {
                            let material_idx = batch.material_id as usize;
                            let material_bg = groups
                                .get(material_idx)
                                .or_else(|| groups.first())
                                .expect("material bind groups empty");
                            pass.set_bind_group(1, material_bg, &[]);
                        }
                        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..batch.index_count, 0, batch.instance_range.clone());
                    }
                }
            }
        }
    }
}
