use crate::graphics::renderer_common::common::{MAX_SHADOW_CASCADES, ShaderConstants};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    pub gbuffer_pass: bool,
    pub shadow_pass: bool,
    pub direct_lighting_pass: bool,
    pub sky_pass: bool,
    pub ssgi_pass: bool,
    pub ssgi_denoise_pass: bool,
    pub ssr_pass: bool,
    pub egui_pass: bool,

    pub freeze_render_camera: bool,

    pub max_lights_forward: usize,
    pub max_lights_deferred: usize,

    pub frames_in_flight: u32,
    pub shadow_map_resolution: u32,
    pub shadow_cascade_count: u32,
    pub shadow_cascade_splits: [f32; MAX_SHADOW_CASCADES + 1],

    pub frustum_culling: bool,
    pub occlusion_culling: bool,
    pub lod: bool,
    pub gpu_driven: bool,
    pub gpu_multi_draw_indirect: bool,
    pub render_bundles: bool,
    pub bundle_invalidate_on_resource_changes: bool,
    pub bundle_cache_stable_frames: u32,
    pub deterministic_rendering: bool,
    pub use_dont_care_load_ops: bool,
    pub use_mesh_shaders: bool,
    pub use_transient_textures: bool,
    pub use_transient_aliasing: bool,
    pub gpu_cull_depth_bias: f32,
    pub gpu_cull_rect_pad: f32,
    pub gpu_lod0_distance: f32,
    pub gpu_lod1_distance: f32,
    pub gpu_lod2_distance: f32,
    pub transform_epsilon: f32,
    pub rotation_epsilon: f32,
    pub cull_interval_frames: u32,
    pub lod_interval_frames: u32,
    pub streaming_interval_frames: u32,
    pub streaming_scan_budget: u32,
    pub streaming_request_budget: u32,
    pub streaming_allow_full_scan: bool,
    pub occlusion_stable_pos_epsilon: f32,
    pub occlusion_stable_rot_epsilon: f32,

    pub debug_flags: u32,

    pub shader_constants: ShaderConstants,
}

pub const DEBUG_FLAG_UNLIT: u32 = 1 << 0;
pub const DEBUG_FLAG_DIRECT: u32 = 1 << 1;
pub const DEBUG_FLAG_INDIRECT: u32 = 1 << 2;
pub const DEBUG_FLAG_REFLECTIONS: u32 = 1 << 3;
pub const DEBUG_FLAG_EMISSION: u32 = 1 << 4;
pub const DEBUG_FLAG_SKY: u32 = 1 << 5;

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            gbuffer_pass: true,
            shadow_pass: true,
            direct_lighting_pass: true,
            sky_pass: true,
            ssgi_pass: true,
            ssgi_denoise_pass: true,
            ssr_pass: true,
            egui_pass: true,

            freeze_render_camera: false,

            max_lights_forward: 256,
            max_lights_deferred: 2048,

            frames_in_flight: 3,
            shadow_map_resolution: 2048,
            shadow_cascade_count: 4,
            shadow_cascade_splits: [0.1, 15.0, 40.0, 100.0, 300.0],

            frustum_culling: true,
            occlusion_culling: true,
            lod: true,
            gpu_driven: true,
            gpu_multi_draw_indirect: true,
            render_bundles: true,
            bundle_invalidate_on_resource_changes: true,
            bundle_cache_stable_frames: 3,
            deterministic_rendering: true,
            use_dont_care_load_ops: false,
            use_mesh_shaders: false,
            use_transient_textures: true,
            use_transient_aliasing: true,
            gpu_cull_depth_bias: 0.01,
            gpu_cull_rect_pad: 1.0,
            gpu_lod0_distance: 25.0,
            gpu_lod1_distance: 60.0,
            gpu_lod2_distance: 120.0,
            transform_epsilon: 0.0001,
            rotation_epsilon: 0.0001,
            cull_interval_frames: 1,
            lod_interval_frames: 1,
            streaming_interval_frames: 1,
            streaming_scan_budget: 2048,
            streaming_request_budget: 512,
            streaming_allow_full_scan: false,
            occlusion_stable_pos_epsilon: 0.002,
            occlusion_stable_rot_epsilon: 1.0e-5,

            debug_flags: DEBUG_FLAG_DIRECT
                | DEBUG_FLAG_INDIRECT
                | DEBUG_FLAG_REFLECTIONS
                | DEBUG_FLAG_EMISSION
                | DEBUG_FLAG_SKY,

            shader_constants: ShaderConstants::default(),
        }
    }
}
