use crate::graphics::renderer::renderer::ShaderConstants;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    pub shadow_pass: bool,
    pub direct_lighting_pass: bool,
    pub sky_pass: bool,
    pub ssgi_pass: bool,
    pub ssgi_denoise_pass: bool,
    pub ssr_pass: bool,
    pub egui_pass: bool,

    pub max_lights_forward: usize,
    pub max_lights_deferred: usize,

    pub frustum_culling: bool,
    pub lod: bool,

    pub shader_constants: ShaderConstants,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            shadow_pass: true,
            direct_lighting_pass: true,
            sky_pass: true,
            ssgi_pass: true,
            ssgi_denoise_pass: true,
            ssr_pass: true,
            egui_pass: false,

            max_lights_forward: 256,
            max_lights_deferred: 2048,

            frustum_culling: true,
            lod: true,

            shader_constants: ShaderConstants::default(),
        }
    }
}