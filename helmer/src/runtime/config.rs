use crate::graphics::{
    backend::binding_backend::BindingBackendChoice, config::RenderConfig,
    renderer_common::common::WgpuBackend,
};
use std::env;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub egui: bool,
    pub wgpu_experimental_features: bool,
    pub wgpu_backend: WgpuBackend,
    pub binding_backend: BindingBackendChoice,

    pub render_config: RenderConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let wgpu_experimental_features = env::var("HELMER_WGPU_EXPERIMENTAL")
            .ok()
            .map(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let wgpu_backend = WgpuBackend::from_env(env::var("HELMER_BACKEND").ok());
        let binding_backend =
            BindingBackendChoice::from_env(env::var("HELMER_BINDING_BACKEND").ok());
        RuntimeConfig {
            egui: true,
            wgpu_experimental_features,
            wgpu_backend,
            binding_backend,

            render_config: RenderConfig::default(),
        }
    }
}
