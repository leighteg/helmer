use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use winit::dpi::PhysicalSize;

use crate::graphics::{
    common::{config::RenderConfig, renderer::RenderDeviceCaps},
    graph::{
        definition::resource_id::ResourceId,
        logic::{gpu_resource_pool::GpuResourcePool, render_graph::RenderGraph},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderGraphConfigSignature {
    pub ssgi_pass: bool,
    pub ssgi_denoise_pass: bool,
    pub ssr_pass: bool,
    pub egui_pass: bool,
    pub occlusion_culling: bool,
    pub shadow_map_resolution: u32,
    pub shadow_cascade_count: u32,
    pub use_transient_textures: bool,
    pub use_transient_aliasing: bool,
}

impl RenderGraphConfigSignature {
    pub fn from_render_config(cfg: &RenderConfig) -> Self {
        Self {
            ssgi_pass: cfg.ssgi_pass,
            ssgi_denoise_pass: cfg.ssgi_denoise_pass,
            ssr_pass: cfg.ssr_pass,
            egui_pass: cfg.egui_pass,
            occlusion_culling: cfg.occlusion_culling,
            shadow_map_resolution: cfg.shadow_map_resolution,
            shadow_cascade_count: cfg.shadow_cascade_count,
            use_transient_textures: cfg.use_transient_textures,
            use_transient_aliasing: cfg.use_transient_aliasing,
        }
    }
}

/// Parameters the render thread provides when building a render graph
pub struct RenderGraphBuildParams {
    pub surface_size: PhysicalSize<u32>,
    pub surface_format: wgpu::TextureFormat,
    pub shadow_format: wgpu::TextureFormat,
    pub device_caps: Arc<RenderDeviceCaps>,
    pub blue_noise_view: Arc<wgpu::TextureView>,
    pub blue_noise_sampler: Arc<wgpu::Sampler>,
    pub config: RenderGraphConfigSignature,
    pub shadow_map_resolution: u32,
    pub shadow_cascade_count: u32,
}

/// Output of a render-graph build step
pub struct RenderGraphBuildOutput {
    pub graph: RenderGraph,
    pub swapchain_id: ResourceId,
    pub resource_ids: Vec<ResourceId>,
    pub hiz_id: Option<ResourceId>,
}

type GraphBuilderFn = dyn for<'a> Fn(&RenderGraphBuildParams, &'a mut GpuResourcePool) -> RenderGraphBuildOutput
    + Send
    + Sync;

static GRAPH_SPEC_VERSION: AtomicU64 = AtomicU64::new(1);

/// Logic-thread owned specification for the render graph
#[derive(Clone)]
pub struct RenderGraphSpec {
    pub name: &'static str,
    pub version: u64,
    pub builder: Arc<GraphBuilderFn>,
}

impl std::fmt::Debug for RenderGraphSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderGraphSpec")
            .field("name", &self.name)
            .field("version", &self.version)
            .finish()
    }
}

impl RenderGraphSpec {
    pub fn new<F>(name: &'static str, version: u64, builder: F) -> Self
    where
        F: for<'a> Fn(&RenderGraphBuildParams, &'a mut GpuResourcePool) -> RenderGraphBuildOutput
            + Send
            + Sync
            + 'static,
    {
        Self {
            name,
            version,
            builder: Arc::new(builder),
        }
    }

    pub fn build(
        &self,
        params: &RenderGraphBuildParams,
        pool: &mut GpuResourcePool,
    ) -> RenderGraphBuildOutput {
        (self.builder)(params, pool)
    }

    pub fn unique<F>(name: &'static str, builder: F) -> Self
    where
        F: for<'a> Fn(&RenderGraphBuildParams, &'a mut GpuResourcePool) -> RenderGraphBuildOutput
            + Send
            + Sync
            + 'static,
    {
        let version = GRAPH_SPEC_VERSION.fetch_add(1, Ordering::Relaxed);
        Self::new(name, version, builder)
    }
}
