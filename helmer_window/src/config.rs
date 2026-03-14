use std::env;

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub fixed_timestep: bool,
    pub target_tickrate: f32,
    pub target_fps: Option<f32>,
    pub title_update_ms: u32,
    pub title: String,
    pub maximized: bool,
    pub visible: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let fixed_timestep = env::var("HELMER_FIXED_TIMESTEP")
            .ok()
            .map(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(cfg!(target_arch = "wasm32"));

        let target_tickrate = 120.0;
        let target_fps = env::var("HELMER_TARGET_FPS")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0);
        let title_update_ms = env::var("HELMER_TITLE_UPDATE_MS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(200);

        Self {
            fixed_timestep,
            target_tickrate,
            target_fps,
            title_update_ms,
            title: "helmer".to_string(),
            maximized: false,
            visible: true,
        }
    }
}
