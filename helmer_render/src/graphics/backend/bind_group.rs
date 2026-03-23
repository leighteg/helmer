use std::sync::Arc;

use parking_lot::RwLock;

use crate::graphics::backend::binding_backend::{
    BindingBackend, BindlessConfig, PassBindingPolicy, PipelineCache, ShaderDefineSet, ShaderKey,
};
use crate::graphics::graph::{
    definition::resource_id::ResourceId, logic::gpu_resource_pool::GpuResourcePool,
};
use crate::graphics::legacy_renderers::forward_pmu::MaterialLowEnd as MaterialGPU;

/// Conventional bind-group backend (non-bindless).
pub struct BindGroupBackend {
    pipeline_cache: RwLock<PipelineCache>,
    pipeline_layout: RwLock<Option<Arc<wgpu::PipelineLayout>>>,
    globals_bgl: RwLock<Option<Arc<wgpu::BindGroupLayout>>>,
}

impl BindGroupBackend {
    pub fn new(_config: BindlessConfig) -> Self {
        Self {
            pipeline_cache: RwLock::new(PipelineCache::new()),
            pipeline_layout: RwLock::new(None),
            globals_bgl: RwLock::new(None),
        }
    }

    fn ensure_globals_bgl(&self, device: &wgpu::Device) -> Arc<wgpu::BindGroupLayout> {
        if let Some(bgl) = self.globals_bgl.read().as_ref() {
            return bgl.clone();
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BindGroups/GlobalsBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
            ],
        });

        let arc = Arc::new(bgl);
        *self.globals_bgl.write() = Some(arc.clone());
        arc
    }
}

impl BindingBackend for BindGroupBackend {
    fn begin_frame(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _pool: &GpuResourcePool,
        _frame_index: u32,
        _binding_changes: &[ResourceId],
    ) {
    }

    fn begin_pass(&self, _rp: &mut wgpu::RenderPass, _policy: PassBindingPolicy) {}

    fn bind_texture(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {}

    fn bind_buffer(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {}

    fn bind_sampler(&self, _rp: &mut wgpu::RenderPass, _id: ResourceId) {}

    fn bind_material(&self, _rp: &mut wgpu::RenderPass, _mat: &MaterialGPU) {}

    fn pipeline_layout(
        &self,
        device: &wgpu::Device,
        _shader: ShaderKey,
    ) -> Arc<wgpu::PipelineLayout> {
        if let Some(layout) = self.pipeline_layout.read().as_ref() {
            return layout.clone();
        }
        let globals = self.ensure_globals_bgl(device);
        let layout = Arc::new(
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BindGroups/PipelineLayout"),
                bind_group_layouts: &[&globals],
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
        let _ = self.ensure_globals_bgl(device);
        vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
    }

    fn shader_defines(&self) -> ShaderDefineSet {
        ShaderDefineSet::default()
            .with("HEL_BINDGROUP_PATH", "1")
            .with("HEL_BINDGROUPS", "1")
    }

    fn invalidate_pipelines(&self) {
        self.pipeline_cache.write().invalidate(None);
        *self.pipeline_layout.write() = None;
        *self.globals_bgl.write() = None;
    }
}
