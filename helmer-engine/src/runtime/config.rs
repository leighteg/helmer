use crate::graphics::config::RenderConfig;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub render_config: RenderConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            render_config: RenderConfig::default(),
        }
    }
}