#![allow(dead_code)]

use std::{num::NonZeroU32, sync::Arc};

use parking_lot::RwLock;

use crate::graphics::backend::binding_backend::{
    BindingBackend, BindingBackendKind, BindlessConfig, PassBindingPolicy, PipelineCache,
    ShaderDefineSet, ShaderKey,
};
use crate::graphics::graph::{
    definition::resource_desc::ResourceDesc,
    definition::resource_id::{ResourceId, ResourceKind},
    logic::gpu_resource_pool::GpuResourcePool,
    logic::residency::Residency,
};

use crate::graphics::renderers::forward_pmu::MaterialLowEnd as MaterialGPU;

/// Modern bindless backend: texture_binding_array + storage buffer arrays.
/// Fully unified texture array (texture_2d<f32>) and buffer array.
pub struct BindlessModernBackend {
    config: BindlessConfig,
    pipeline_cache: RwLock<PipelineCache>,
    pipeline_layout: RwLock<Option<Arc<wgpu::PipelineLayout>>>,

    // Global bindless layouts/groups
    layout: RwLock<Option<Arc<wgpu::BindGroupLayout>>>,
    bind_group: RwLock<Option<Arc<wgpu::BindGroup>>>,

    // CPU-side views, used to build bind group
    textures: RwLock<Vec<Option<Arc<wgpu::TextureView>>>>,
    samplers: RwLock<Vec<Option<Arc<wgpu::Sampler>>>>,
    buffers: RwLock<Vec<Option<Arc<wgpu::Buffer>>>>,

    default_texture: RwLock<Option<Arc<wgpu::TextureView>>>,
    default_sampler: RwLock<Option<Arc<wgpu::Sampler>>>,
    default_buffer: RwLock<Option<Arc<wgpu::Buffer>>>,
    last_epoch: RwLock<Option<u64>>,
}

impl BindlessModernBackend {
    pub fn new(config: BindlessConfig) -> Self {
        let tex_cap = config.max_textures as usize;
        let samp_cap = config.max_samplers as usize;
        let buf_cap = config.max_buffers as usize;

        Self {
            config,
            pipeline_cache: RwLock::new(PipelineCache::new()),
            pipeline_layout: RwLock::new(None),
            layout: RwLock::new(None),
            bind_group: RwLock::new(None),
            textures: RwLock::new(vec![None; tex_cap]),
            samplers: RwLock::new(vec![None; samp_cap]),
            buffers: RwLock::new(vec![None; buf_cap]),
            default_texture: RwLock::new(None),
            default_sampler: RwLock::new(None),
            default_buffer: RwLock::new(None),
            last_epoch: RwLock::new(None),
        }
    }

    fn ensure_layout(&self, device: &wgpu::Device) -> Arc<wgpu::BindGroupLayout> {
        if let Some(l) = self.layout.read().as_ref() {
            return l.clone();
        }

        let mut write_guard = self.layout.write();
        if let Some(l) = write_guard.as_ref() {
            return l.clone();
        }

        debug_assert!(
            device.features().contains(
                wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::BUFFER_BINDING_ARRAY
                    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
            ),
            "BindlessModernBackend requires full bindless feature set"
        );

        // Layout:
        // binding 0: array<texture_2d<f32>>
        // binding 1: array<sampler>
        // binding 2: array<storage, read> buffers (raw)
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BindlessModern/Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(NonZeroU32::new(self.config.max_textures).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: Some(NonZeroU32::new(self.config.max_samplers).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: Some(NonZeroU32::new(self.config.max_buffers).unwrap()),
                },
            ],
        });

        let arc = Arc::new(layout);
        *write_guard = Some(arc.clone());
        arc
    }

    fn ensure_defaults(
        &self,
        device: &wgpu::Device,
    ) -> (
        Arc<wgpu::TextureView>,
        Arc<wgpu::Sampler>,
        Arc<wgpu::Buffer>,
    ) {
        let tex = {
            let mut guard = self.default_texture.write();
            guard
                .get_or_insert_with(|| {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("BindlessModern/DefaultTexture"),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    Arc::new(texture.create_view(&Default::default()))
                })
                .clone()
        };

        let sampler = {
            let mut guard = self.default_sampler.write();
            guard
                .get_or_insert_with(|| {
                    Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("BindlessModern/DefaultSampler"),
                        ..Default::default()
                    }))
                })
                .clone()
        };

        let buffer = {
            let mut guard = self.default_buffer.write();
            guard
                .get_or_insert_with(|| {
                    Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("BindlessModern/DefaultBuffer"),
                        size: 256,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }))
                })
                .clone()
        };

        (tex, sampler, buffer)
    }

    fn rebuild_bind_group(&self, device: &wgpu::Device) {
        let layout = self.ensure_layout(device);

        let tex_guard = self.textures.read();
        let samp_guard = self.samplers.read();
        let buf_guard = self.buffers.read();
        let (fallback_tex, fallback_sampler, fallback_buffer) = self.ensure_defaults(device);

        // Fill in with defaults (if any slot None, we substitute a dummy resource).
        // for simplicity, renderer is required to have default view / sampler / buffer - can add fields for those defaults on this struct if needed.
        let tex_views: Vec<&wgpu::TextureView> = tex_guard
            .iter()
            .map(|tv| tv.as_deref().unwrap_or(fallback_tex.as_ref()))
            .collect();
        let sams: Vec<&wgpu::Sampler> = samp_guard
            .iter()
            .map(|s| s.as_deref().unwrap_or(fallback_sampler.as_ref()))
            .collect();
        let bufs: Vec<wgpu::BufferBinding> = buf_guard
            .iter()
            .map(|b| {
                b.as_ref()
                    .map(|buffer| buffer.as_entire_buffer_binding())
                    .unwrap_or_else(|| fallback_buffer.as_entire_buffer_binding())
            })
            .collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BindlessModern/BindGroup"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&tex_views),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::SamplerArray(&sams),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::BufferArray(&bufs),
                },
            ],
        });

        *self.bind_group.write() = Some(Arc::new(bind_group));
    }

    fn map_texture_from_pool(&self, pool: &GpuResourcePool) {
        let mut textures = self.textures.write();
        for slot in textures.iter_mut() {
            *slot = None;
        }
        let cap = textures.len();
        for entry in pool.iter_entries() {
            if entry.kind != ResourceKind::Texture {
                continue;
            }
            if entry.residency != Residency::Resident {
                continue;
            }
            if !entry.is_streaming() {
                continue;
            }
            let (layers, usage) = match &entry.desc {
                ResourceDesc::Texture2D { layers, usage, .. } => (*layers, *usage),
                _ => (1, wgpu::TextureUsages::empty()),
            };
            if layers != 1 {
                continue;
            }
            if !usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
                continue;
            }
            let idx = entry.id.index();
            if idx >= cap {
                continue;
            }
            if entry.is_depth_texture() {
                textures[idx] = None;
                continue;
            }
            if let Some(ref tex) = entry.texture {
                let size = tex.size();
                // bindless texture array only supports plain 2d, single-layer textures.
                if tex.dimension() != wgpu::TextureDimension::D2 || size.depth_or_array_layers != 1
                {
                    textures[idx] = None;
                    continue;
                }

                textures[idx] = Some(Arc::new(tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    ..Default::default()
                })));
            }
        }
    }

    fn map_buffers_from_pool(&self, pool: &GpuResourcePool) {
        let mut buffers = self.buffers.write();
        for slot in buffers.iter_mut() {
            *slot = None;
        }
        let cap = buffers.len();
        for entry in pool.iter_entries() {
            if entry.kind != ResourceKind::Buffer {
                continue;
            }
            if entry.residency != Residency::Resident {
                continue;
            }
            let idx = entry.id.index();
            if idx >= cap {
                continue;
            }
            if let Some(ref buf) = entry.buffer {
                buffers[idx] = Some(Arc::new(buf.clone()));
            }
        }
    }

    fn map_samplers_from_pool(&self, pool: &GpuResourcePool) {
        let mut samplers = self.samplers.write();
        for slot in samplers.iter_mut() {
            *slot = None;
        }
        let cap = samplers.len();
        for entry in pool.iter_entries() {
            if entry.kind != ResourceKind::Sampler {
                continue;
            }
            if entry.residency != Residency::Resident {
                continue;
            }
            let idx = entry.id.index();
            if idx >= cap {
                continue;
            }
            if let Some(ref sam) = entry.sampler {
                samplers[idx] = Some(Arc::new(sam.clone()));
            }
        }
    }

    fn bindless_bg(&self) -> Option<Arc<wgpu::BindGroup>> {
        self.bind_group.read().clone()
    }

    pub fn backend_kind(&self) -> BindingBackendKind {
        BindingBackendKind::BindlessModern
    }

    pub fn pipeline_cache(&self) -> &RwLock<PipelineCache> {
        &self.pipeline_cache
    }
}

impl BindingBackend for BindlessModernBackend {
    fn begin_frame(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        pool: &GpuResourcePool,
        _frame_index: u32,
    ) {
        let epoch = pool.binding_epoch();
        let needs_rebuild = {
            let last = self.last_epoch.read();
            let epoch_matches = last.map_or(false, |prev| prev == epoch);
            !epoch_matches || self.bind_group.read().is_none()
        };
        if !needs_rebuild {
            return;
        }
        *self.last_epoch.write() = Some(epoch);

        // Scan pool and update arrays; rebuild bindgroup.
        self.map_texture_from_pool(pool);
        self.map_samplers_from_pool(pool);
        self.map_buffers_from_pool(pool);
        self.rebuild_bind_group(device);
    }

    fn begin_pass(&self, rp: &mut wgpu::RenderPass, policy: PassBindingPolicy) {
        if policy == PassBindingPolicy::Minimal {
            return;
        }
        if let Some(bg) = self.bindless_bg() {
            rp.set_bind_group(0, bg.as_ref(), &[]);
        }
    }

    fn bind_texture(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {
        // Shaders index by handle (ResourceId.index()).
    }

    fn bind_buffer(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {
        // Shaders index by handle (ResourceId.index()).
    }

    fn bind_sampler(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {
        // Shaders index by handle (ResourceId.index()).
    }

    fn bind_material(&self, _rp: &mut wgpu::RenderPass, _mat: &MaterialGPU) {
        // The material just contains texture indices, which the shader reads via uniform/SSBO; there's no per-material bind here.
    }

    fn pipeline_layout(
        &self,
        device: &wgpu::Device,
        _shader: ShaderKey,
    ) -> Arc<wgpu::PipelineLayout> {
        if let Some(layout) = self.pipeline_layout.read().as_ref() {
            return layout.clone();
        }
        let bindless_layout = self.ensure_layout(device);
        let layout = Arc::new(
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BindlessModern/PipelineLayout"),
                bind_group_layouts: &[&bindless_layout],
                immediate_size: 0,
            }),
        );
        *self.pipeline_layout.write() = Some(layout.clone());
        layout
    }

    fn pipeline_bgl(
        &self,
        device: &wgpu::Device,
        _shader: ShaderKey,
    ) -> Vec<wgpu::BindGroupLayoutEntry> {
        let _ = device; // unused
        Vec::new()
    }

    fn shader_defines(&self) -> ShaderDefineSet {
        ShaderDefineSet::default()
            .with("HEL_BINDLESS", "1")
            .with("HEL_BINDLESS_MODERN", "1")
    }

    fn invalidate_pipelines(&self) {
        self.pipeline_cache
            .write()
            .invalidate(Some(self.backend_kind()));
        *self.pipeline_layout.write() = None;
    }
}
