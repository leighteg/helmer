#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    pub shadow_pass: bool,
    pub direct_lighting_pass: bool,
    pub sky_pass: bool,
    pub ssgi_pass: bool,
    pub ssgi_denoise_pass: bool,
    pub ssr_pass: bool,

    pub max_lights_forward: usize,
    pub max_lights_deferred: usize,

    pub frustum_culling: bool,
    pub lod: bool,
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

            max_lights_forward: 248,
            max_lights_deferred: 2048,

            frustum_culling: true,
            lod: true,
        }
    }
}