use crate::graphics::config::RenderConfig;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub egui: bool,
    
    pub render_config: RenderConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            egui: false,
            
            render_config: RenderConfig::default(),
        }
    }
}