#![allow(dead_code)]

use std::{any::Any, collections::HashMap, sync::Arc};

use crate::graphics::graph::{
    definition::resource_id::ResourceId,
    logic::{frame_inputs::FrameInputHub, gpu_resource_pool::GpuResourcePool},
};
use crate::graphics::legacy_renderers::forward_pmu::MaterialLowEnd as MaterialGPU;
use wgpu::{Features, Limits};

/// Which binding backend is active
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum BindingBackendKind {
    BindlessModern,
    BindlessFallback,
    BindGroups,
}

/// User/runtime selection for binding backend (Auto defers to feature detection)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum BindingBackendChoice {
    Auto,
    BindlessModern,
    BindlessFallback,
    BindGroups,
}

impl BindingBackendChoice {
    pub fn from_env(value: Option<String>) -> Self {
        match value.as_deref().unwrap_or_default().to_lowercase().as_str() {
            "bindgroups" | "bindgroup" | "bind_group" => BindingBackendChoice::BindGroups,
            "bindless_fallback" | "fallback" => BindingBackendChoice::BindlessFallback,
            "bindless_modern" | "modern" => BindingBackendChoice::BindlessModern,
            _ => BindingBackendChoice::Auto,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BindingBackendChoice::Auto => "Auto",
            BindingBackendChoice::BindlessModern => "Bindless (Modern)",
            BindingBackendChoice::BindlessFallback => "Bindless (Fallback)",
            BindingBackendChoice::BindGroups => "Bind Groups",
        }
    }
}

/// Per-pass binding policy (how much the backend does automatically)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PassBindingPolicy {
    Full,
    OnlyGlobals,
    OnlyResources,
    Minimal,
}

/// Small shader define set to inject into WGSL / shader compilation
#[derive(Clone, Default)]
pub struct ShaderDefineSet {
    pub defines: Vec<(&'static str, &'static str)>,
}

impl ShaderDefineSet {
    pub fn with(mut self, k: &'static str, v: &'static str) -> Self {
        self.defines.push((k, v));
        self
    }
}

/// Key for pipelines (!extend with permutation bits!)
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ShaderKey {
    pub id: u64,
}

/// Cross-backend pipeline cache
pub struct PipelineCache {
    per_backend: HashMap<BindingBackendKind, HashMap<ShaderKey, Arc<wgpu::RenderPipeline>>>,
    pub enable_caching: bool,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            per_backend: HashMap::new(),
            enable_caching: true,
        }
    }

    pub fn get(
        &self,
        kind: BindingBackendKind,
        key: ShaderKey,
    ) -> Option<Arc<wgpu::RenderPipeline>> {
        if !self.enable_caching {
            return None;
        }
        self.per_backend.get(&kind)?.get(&key).cloned()
    }

    pub fn insert(
        &mut self,
        kind: BindingBackendKind,
        key: ShaderKey,
        pipeline: Arc<wgpu::RenderPipeline>,
    ) {
        if !self.enable_caching {
            return;
        }
        self.per_backend
            .entry(kind)
            .or_default()
            .insert(key, pipeline);
    }

    pub fn invalidate(&mut self, kind: Option<BindingBackendKind>) {
        match kind {
            Some(k) => {
                self.per_backend.remove(&k);
            }
            None => {
                self.per_backend.clear();
            }
        }
    }
}

/// Context the graph passes give to the backend when binding
pub struct RenderPassCtx<'a> {
    pub backend: &'a dyn BindingBackend,
    pub pool: &'a mut GpuResourcePool,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub frame_index: u32,
    pub frame_inputs: &'a FrameInputHub,
    pub user_data: Option<&'a dyn Any>,
}

impl<'a> RenderPassCtx<'a> {
    /// Inform the pool that a resource was accessed this frame
    pub fn mark_used(&mut self, id: ResourceId) {
        self.pool.mark_used(id, self.frame_index);
    }

    /// Start a render pass with a chosen policy (backend sets bind groups)
    pub fn begin_pass(&self, rp: &mut wgpu::RenderPass<'_>, policy: PassBindingPolicy) {
        self.backend.begin_pass(rp, policy);
    }
}

/// Bindless/backing array configuration
#[derive(Clone, Debug)]
pub struct BindlessConfig {
    pub max_textures: u32,
    pub max_samplers: u32,
    pub max_buffers: u32,
}

impl Default for BindlessConfig {
    fn default() -> Self {
        Self {
            max_textures: 4096,
            max_samplers: 256,
            max_buffers: 4096,
        }
    }
}

impl BindlessConfig {
    pub fn clamp_to_device(self, features: Features, limits: &Limits) -> Self {
        fn clamp(desired: u32, limit: u32) -> u32 {
            desired.max(1).min(limit.max(1))
        }

        let mut cfg = self;

        cfg.max_textures = clamp(
            cfg.max_textures,
            limits.max_sampled_textures_per_shader_stage,
        );
        cfg.max_samplers = clamp(cfg.max_samplers, limits.max_samplers_per_shader_stage);
        cfg.max_buffers = clamp(cfg.max_buffers, limits.max_storage_buffers_per_shader_stage);

        if !features.contains(Features::TEXTURE_BINDING_ARRAY) {
            cfg.max_textures = 1;
        }
        if !features.contains(Features::TEXTURE_BINDING_ARRAY) {
            cfg.max_samplers = 1;
        }
        if !(features.contains(Features::BUFFER_BINDING_ARRAY)
            && features.contains(Features::STORAGE_RESOURCE_BINDING_ARRAY))
        {
            cfg.max_buffers = 1;
        }

        cfg
    }
}

pub trait BindingBackend: Send + Sync {
    /// Called once per frame by the renderer.
    /// Backend can scan the pool and update its tables.
    fn begin_frame(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pool: &GpuResourcePool,
        frame_index: u32,
    );

    /// Called at pass begin (sets bindgroups / arrays)
    fn begin_pass(&self, rp: &mut wgpu::RenderPass, policy: PassBindingPolicy);

    /// Bind a single texture by ResourceId (index bits are the handle)
    fn bind_texture(&self, rp: &mut wgpu::RenderPass, id: ResourceId);

    /// Bind a single buffer by ResourceId
    fn bind_buffer(&self, rp: &mut wgpu::RenderPass, id: ResourceId);

    /// Bind sampler by ResourceId
    fn bind_sampler(&self, rp: &mut wgpu::RenderPass, id: ResourceId);

    /// Bind whole material
    fn bind_material(&self, rp: &mut wgpu::RenderPass, mat: &MaterialGPU);

    /// Return a backend-specific pipeline layout based on shader key
    fn pipeline_layout(
        &self,
        device: &wgpu::Device,
        shader: ShaderKey,
    ) -> Arc<wgpu::PipelineLayout>;

    /// Return bind group layout entries for this backend's globals
    fn pipeline_bgl(
        &self,
        device: &wgpu::Device,
        shader: ShaderKey,
    ) -> Vec<wgpu::BindGroupLayoutEntry>;

    /// Shader defines to inject into WGSL
    fn shader_defines(&self) -> ShaderDefineSet;

    /// Notify backend to invalidate its internal pipeline cache
    fn invalidate_pipelines(&self);
}
