struct Constants {
    // lighting
    shade_mode: u32,
    light_model: u32,
    skylight_contribution: u32,

    // sky
    planet_radius: f32,
    atmosphere_radius: f32,
    sky_light_samples: u32,

    // SSR
    ssr_coarse_steps: u32,
    ssr_binary_search_steps: u32,
    ssr_linear_step_size: f32,
    ssr_thickness: f32,
    ssr_max_distance: f32,
    ssr_roughness_fade_start: f32,
    ssr_roughness_fade_end: f32,

    // SSGI
    ssgi_num_rays: u32,
    ssgi_num_steps: u32,
    ssgi_ray_step_size: f32,
    ssgi_thickness: f32,
    ssgi_blend_factor: f32,

    // shadows
    evsm_c: f32,
    pcf_radius: u32,
    pcf_min_scale: f32,
    pcf_max_scale: f32,
    pcf_max_distance: f32,

    // composite
    ssgi_intensity: f32,

    _padding: vec4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
}

struct LightVP {
    matrix: mat4x4<f32>,
}

struct ModelPushConstant {
    model_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> light_vp: LightVP;
@vertex var<push_constant> model: ModelPushConstant;

@group(1) @binding(0)
var<uniform> constants: Constants;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model_matrix * vec4<f32>(in.position, 1.0);
    let clip_pos = light_vp.matrix * world_pos;
    out.clip_position = clip_pos;
    // pass depth from the vertex shader to get linear depth, which is better for VSM.
    // clip_pos.z is linear for orthographic projections.
    out.depth = clip_pos.z;
    return out;
}

struct FragmentOutput {
    @location(0) moments: vec2<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    let depth = in.depth; // depth is in [0, 1] range

    // Warp the depth exponentially
    let warped_depth = exp(constants.evsm_c * (depth - 1.0));

    let moment1 = warped_depth;
    let moment2 = warped_depth * warped_depth;

    out.moments = vec2<f32>(moment1, moment2);
    return out;
}