#![allow(dead_code)]

use std::{collections::HashMap, num::NonZeroU32, sync::Arc};

use parking_lot::RwLock;

use crate::graphics::backend::binding_backend::{
    BindingBackend, BindingBackendKind, BindlessConfig, PassBindingPolicy, PipelineCache,
    ShaderDefineSet, ShaderKey,
};
use crate::graphics::graph::{
    definition::resource_desc::ResourceDesc,
    definition::resource_id::{ResourceId, ResourceKind},
    logic::gpu_resource_pool::GpuResourcePool,
    logic::residency::{GpuResourceEntry, Residency},
};
use crate::graphics::legacy_renderers::forward_pmu::MaterialLowEnd as MaterialGPU;

/// Fallback backend: emulates bindless via fixed-size arrays and indirection tables.
/// Shaders still use indices, but those indices go through a small table.
pub struct BindlessFallbackBackend {
    config: BindlessConfig,
    pipeline_cache: RwLock<PipelineCache>,
    pipeline_layout: RwLock<Option<Arc<wgpu::PipelineLayout>>>,

    layout: RwLock<Option<Arc<wgpu::BindGroupLayout>>>,
    bind_group: RwLock<Option<Arc<wgpu::BindGroup>>>,

    // Compact tables: logical handle (ResourceId.index()) -> slot in fixed array
    texture_slots: RwLock<HashMap<usize, u32>>,
    buffer_slots: RwLock<HashMap<usize, u32>>,
    sampler_slots: RwLock<HashMap<usize, u32>>,

    // GPU arrays
    texture_views: RwLock<Vec<Option<Arc<wgpu::TextureView>>>>,
    samplers: RwLock<Vec<Option<Arc<wgpu::Sampler>>>>,
    buffers: RwLock<Vec<Option<Arc<wgpu::Buffer>>>>,

    default_texture: RwLock<Option<Arc<wgpu::TextureView>>>,
    default_sampler: RwLock<Option<Arc<wgpu::Sampler>>>,
    default_buffer: RwLock<Option<Arc<wgpu::Buffer>>>,
    last_epoch: RwLock<Option<u64>>,
}

impl BindlessFallbackBackend {
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
            texture_slots: RwLock::new(HashMap::new()),
            buffer_slots: RwLock::new(HashMap::new()),
            sampler_slots: RwLock::new(HashMap::new()),
            texture_views: RwLock::new(vec![None; tex_cap]),
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

        let features = device.features();
        let use_texture_array = features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY);
        let use_sampler_array = use_texture_array;
        let use_buffer_array = features.contains(wgpu::Features::BUFFER_BINDING_ARRAY)
            && features.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY);

        // layout: same shape as modern, but wgpu will emulate arrays if HW doesnt support full bindless
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BindlessFallback/Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: use_texture_array
                        .then(|| NonZeroU32::new(self.config.max_textures).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: use_sampler_array
                        .then(|| NonZeroU32::new(self.config.max_samplers).unwrap()),
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
                    count: use_buffer_array
                        .then(|| NonZeroU32::new(self.config.max_buffers).unwrap()),
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
                        label: Some("BindlessFallback/DefaultTexture"),
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
                    Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("BindlessFallback/DefaultTextureView"),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        ..Default::default()
                    }))
                })
                .clone()
        };

        let sampler = {
            let mut guard = self.default_sampler.write();
            guard
                .get_or_insert_with(|| {
                    Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("BindlessFallback/DefaultSampler"),
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
                        label: Some("BindlessFallback/DefaultBuffer"),
                        size: 256,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }))
                })
                .clone()
        };

        (tex, sampler, buffer)
    }

    fn alloc_slot(map: &mut HashMap<usize, u32>, max: u32, handle: usize) -> Option<u32> {
        if handle >= max as usize {
            return None;
        }
        let slot = handle as u32;
        map.insert(handle, slot);
        Some(slot)
    }

    fn update_from_pool(&self, device: &wgpu::Device, pool: &GpuResourcePool) {
        let mut tex_slots = self.texture_slots.write();
        let mut buf_slots = self.buffer_slots.write();
        let mut sam_slots = self.sampler_slots.write();

        tex_slots.clear();
        buf_slots.clear();
        sam_slots.clear();

        let mut textures = self.texture_views.write();
        let mut buffers = self.buffers.write();
        let mut samplers = self.samplers.write();

        for slot in textures.iter_mut() {
            *slot = None;
        }
        for slot in buffers.iter_mut() {
            *slot = None;
        }
        for slot in samplers.iter_mut() {
            *slot = None;
        }

        for entry in pool.iter_entries() {
            if entry.residency != Residency::Resident {
                continue;
            }
            let h = entry.id.index();

            match entry.kind {
                ResourceKind::Texture => {
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
                    if let Some(slot) =
                        Self::alloc_slot(&mut tex_slots, self.config.max_textures, h)
                    {
                        let is_depth = entry.is_depth_texture();
                        if let Some(slot_view) = textures.get_mut(slot as usize) {
                            if is_depth {
                                *slot_view = None;
                            } else if let Some(ref tex) = entry.texture {
                                let size = tex.size();
                                if tex.dimension() == wgpu::TextureDimension::D2
                                    && size.depth_or_array_layers == 1
                                {
                                    *slot_view = Some(Arc::new(tex.create_view(
                                        &wgpu::TextureViewDescriptor {
                                            label: Some("BindlessFallback/TexView"),
                                            dimension: Some(wgpu::TextureViewDimension::D2),
                                            ..Default::default()
                                        },
                                    )));
                                }
                            }
                        }
                    }
                }
                ResourceKind::Buffer => {
                    if let Some(ref buf) = entry.buffer {
                        if let Some(slot) =
                            Self::alloc_slot(&mut buf_slots, self.config.max_buffers, h)
                        {
                            if let Some(slot_buf) = buffers.get_mut(slot as usize) {
                                *slot_buf = Some(Arc::new(buf.clone()));
                            }
                        }
                    }
                }
                ResourceKind::Sampler => {
                    if let Some(ref sam) = entry.sampler {
                        if let Some(slot) =
                            Self::alloc_slot(&mut sam_slots, self.config.max_samplers, h)
                        {
                            if let Some(slot_sam) = samplers.get_mut(slot as usize) {
                                *slot_sam = Some(Arc::new(sam.clone()));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        self.rebuild_bind_group(device);
    }

    fn rebuild_bind_group(&self, device: &wgpu::Device) {
        let layout = self.ensure_layout(device);
        let (fallback_tex, fallback_sampler, fallback_buffer) = self.ensure_defaults(device);
        let features = device.features();
        let use_texture_array = features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY);
        let use_sampler_array = use_texture_array;
        let use_buffer_array = features.contains(wgpu::Features::BUFFER_BINDING_ARRAY)
            && features.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY);

        let textures = self.texture_views.read();
        let samplers = self.samplers.read();
        let buffers = self.buffers.read();

        let tex_array: Vec<&wgpu::TextureView> = if use_texture_array {
            textures
                .iter()
                .map(|tv| tv.as_deref().unwrap_or(fallback_tex.as_ref()))
                .collect()
        } else {
            Vec::new()
        };
        let sam_array: Vec<&wgpu::Sampler> = if use_sampler_array {
            samplers
                .iter()
                .map(|sam| sam.as_deref().unwrap_or(fallback_sampler.as_ref()))
                .collect()
        } else {
            Vec::new()
        };
        let buf_array: Vec<wgpu::BufferBinding> = if use_buffer_array {
            buffers
                .iter()
                .map(|buf| {
                    buf.as_ref()
                        .map(|b| b.as_entire_buffer_binding())
                        .unwrap_or_else(|| fallback_buffer.as_entire_buffer_binding())
                })
                .collect()
        } else {
            Vec::new()
        };

        let single_tex = textures
            .iter()
            .find_map(|tv| tv.as_deref())
            .unwrap_or(fallback_tex.as_ref());
        let single_sam = samplers
            .iter()
            .find_map(|sam| sam.as_deref())
            .unwrap_or(fallback_sampler.as_ref());
        let single_buf = buffers
            .iter()
            .find_map(|buf| buf.as_ref())
            .map(|b| b.as_entire_buffer_binding())
            .unwrap_or_else(|| fallback_buffer.as_entire_buffer_binding());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BindlessFallback/BindGroup"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: if use_texture_array {
                        wgpu::BindingResource::TextureViewArray(&tex_array)
                    } else {
                        wgpu::BindingResource::TextureView(single_tex)
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: if use_sampler_array {
                        wgpu::BindingResource::SamplerArray(&sam_array)
                    } else {
                        wgpu::BindingResource::Sampler(single_sam)
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: if use_buffer_array {
                        wgpu::BindingResource::BufferArray(&buf_array)
                    } else {
                        wgpu::BindingResource::Buffer(single_buf)
                    },
                },
            ],
        });

        *self.bind_group.write() = Some(Arc::new(bg));
    }

    fn update_texture_binding(&self, id: ResourceId, entry: Option<&GpuResourceEntry>) -> bool {
        if id.kind() != ResourceKind::Texture {
            return false;
        }
        let handle = id.index();
        if handle >= self.config.max_textures as usize {
            return false;
        }

        let mut tex_slots = self.texture_slots.write();
        let mut textures = self.texture_views.write();
        tex_slots.remove(&handle);
        let Some(slot_view) = textures.get_mut(handle) else {
            return false;
        };
        *slot_view = None;

        let Some(entry) = entry else {
            return true;
        };
        if entry.kind != ResourceKind::Texture
            || entry.residency != Residency::Resident
            || !entry.is_streaming()
        {
            return true;
        }
        let (layers, usage) = match &entry.desc {
            ResourceDesc::Texture2D { layers, usage, .. } => (*layers, *usage),
            _ => (1, wgpu::TextureUsages::empty()),
        };
        if layers != 1
            || !usage.contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || entry.is_depth_texture()
        {
            return true;
        }
        if let Some(ref tex) = entry.texture {
            let size = tex.size();
            if tex.dimension() == wgpu::TextureDimension::D2 && size.depth_or_array_layers == 1 {
                tex_slots.insert(handle, handle as u32);
                *slot_view = Some(Arc::new(tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("BindlessFallback/TexView"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    ..Default::default()
                })));
            }
        }
        true
    }

    fn update_buffer_binding(&self, id: ResourceId, entry: Option<&GpuResourceEntry>) -> bool {
        if id.kind() != ResourceKind::Buffer {
            return false;
        }
        let handle = id.index();
        if handle >= self.config.max_buffers as usize {
            return false;
        }

        let mut buf_slots = self.buffer_slots.write();
        let mut buffers = self.buffers.write();
        buf_slots.remove(&handle);
        let Some(slot_buf) = buffers.get_mut(handle) else {
            return false;
        };
        *slot_buf = None;
        if let Some(buffer) = entry
            .filter(|entry| {
                entry.kind == ResourceKind::Buffer && entry.residency == Residency::Resident
            })
            .and_then(|entry| entry.buffer.as_ref().map(|buffer| Arc::new(buffer.clone())))
        {
            buf_slots.insert(handle, handle as u32);
            *slot_buf = Some(buffer);
        }
        true
    }

    fn update_sampler_binding(&self, id: ResourceId, entry: Option<&GpuResourceEntry>) -> bool {
        if id.kind() != ResourceKind::Sampler {
            return false;
        }
        let handle = id.index();
        if handle >= self.config.max_samplers as usize {
            return false;
        }

        let mut sam_slots = self.sampler_slots.write();
        let mut samplers = self.samplers.write();
        sam_slots.remove(&handle);
        let Some(slot_sampler) = samplers.get_mut(handle) else {
            return false;
        };
        *slot_sampler = None;
        if let Some(sampler) = entry
            .filter(|entry| {
                entry.kind == ResourceKind::Sampler && entry.residency == Residency::Resident
            })
            .and_then(|entry| {
                entry
                    .sampler
                    .as_ref()
                    .map(|sampler| Arc::new(sampler.clone()))
            })
        {
            sam_slots.insert(handle, handle as u32);
            *slot_sampler = Some(sampler);
        }
        true
    }

    fn apply_binding_change(&self, pool: &GpuResourcePool, id: ResourceId) -> bool {
        let entry = pool.entry(id);
        self.update_texture_binding(id, entry)
            || self.update_buffer_binding(id, entry)
            || self.update_sampler_binding(id, entry)
    }

    fn bg(&self) -> Option<Arc<wgpu::BindGroup>> {
        self.bind_group.read().clone()
    }

    pub fn backend_kind(&self) -> BindingBackendKind {
        BindingBackendKind::BindlessFallback
    }

    pub fn pipeline_cache(&self) -> &RwLock<PipelineCache> {
        &self.pipeline_cache
    }
}

impl BindingBackend for BindlessFallbackBackend {
    fn begin_frame(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        pool: &GpuResourcePool,
        _frame_index: u32,
        binding_changes: &[ResourceId],
    ) {
        let epoch = pool.binding_epoch();
        let (needs_rebuild, bind_group_missing, epoch_matches) = {
            let last = self.last_epoch.read();
            let epoch_matches = last.map_or(false, |prev| prev == epoch);
            let bind_group_missing = self.bind_group.read().is_none();
            (
                !epoch_matches || bind_group_missing,
                bind_group_missing,
                epoch_matches,
            )
        };
        if !needs_rebuild {
            return;
        }
        let full_rebuild = bind_group_missing || !epoch_matches && binding_changes.is_empty();
        let any_change = if full_rebuild {
            self.update_from_pool(device, pool);
            true
        } else {
            binding_changes
                .iter()
                .copied()
                .any(|id| self.apply_binding_change(pool, id))
        };
        *self.last_epoch.write() = Some(epoch);
        if !any_change && !bind_group_missing {
            return;
        }
        self.rebuild_bind_group(device);
    }

    fn begin_pass(&self, rp: &mut wgpu::RenderPass, policy: PassBindingPolicy) {
        if policy == PassBindingPolicy::Minimal {
            return;
        }
        if let Some(bg) = self.bg() {
            rp.set_bind_group(0, bg.as_ref(), &[]);
        }
    }

    fn bind_texture(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {
        // Shader will do indirection table lookup (logical handle -> slot).
    }

    fn bind_buffer(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {
        // Shader will do indirection table lookup (logical handle -> slot).
    }

    fn bind_sampler(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {
        // Shader will do indirection table lookup (logical handle -> slot).
    }

    fn bind_material(&self, _rp: &mut wgpu::RenderPass, _mat: &MaterialGPU) {
        // As with modern, materials use handles/indices in uniforms/SSBOs.
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
                label: Some("BindlessFallback/PipelineLayout"),
                bind_group_layouts: &[&bindless_layout],
                immediate_size: 0,
            }),
        );
        *self.pipeline_layout.write() = Some(layout.clone());
        layout
    }

    fn pipeline_bgl(
        &self,
        _device: &wgpu::Device,
        _shader: ShaderKey,
    ) -> Vec<wgpu::BindGroupLayoutEntry> {
        Vec::new()
    }

    fn shader_defines(&self) -> ShaderDefineSet {
        ShaderDefineSet::default()
            .with("HEL_BINDLESS", "1")
            .with("HEL_BINDLESS_FALLBACK", "1")
    }

    fn invalidate_pipelines(&self) {
        self.pipeline_cache
            .write()
            .invalidate(Some(self.backend_kind()));
        *self.pipeline_layout.write() = None;
    }
}
