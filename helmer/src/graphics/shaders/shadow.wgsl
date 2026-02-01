//=============== STRUCTS ===============//

struct VertexInput {
    @location(0) position: vec3<f32>,
    // These are part of the Vertex::desc() but not used here
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

// Per-instance data read from vertex buffer 1
struct InstanceInput {
    @location(5) model_matrix_col_0: vec4<f32>,
    @location(6) model_matrix_col_1: vec4<f32>,
    @location(7) model_matrix_col_2: vec4<f32>,
    @location(8) model_matrix_col_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32, // Normalized 0-1 depth
}

// Per-cascade uniform
struct LightVP {
    view_proj: mat4x4<f32>,
}

struct Constants {
    // general
    mip_bias: f32,

    // lighting
    shade_mode: u32,
    shade_smooth: u32,
    light_model: u32,
    skylight_contribution: u32,

    // sky
    planet_radius: f32,
    atmosphere_radius: f32,
    sky_light_samples: u32,
    _pad0: u32,

    // SSR
    ssr_coarse_steps: u32,
    ssr_binary_search_steps: u32,
    ssr_linear_step_size: f32,
    ssr_thickness: f32,
    ssr_max_distance: f32,
    ssr_roughness_fade_start: f32,
    ssr_roughness_fade_end: f32,
    _pad1: u32,

    // SSGI
    ssgi_num_rays: u32,
    ssgi_num_steps: u32,
    ssgi_ray_step_size: f32,
    ssgi_thickness: f32,
    ssgi_blend_factor: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,

    // shadows
    evsm_c: f32,
    pcf_radius: u32,
    pcf_min_scale: f32,
    pcf_max_scale: f32,
    pcf_max_distance: f32,
    ssgi_intensity: f32,
    _final_padding: vec4<f32>,
};

//=============== BINDINGS ===============//

@group(0) @binding(0) var<uniform> light_vp: LightVP; 
@group(1) @binding(0) var<uniform> render_constants: Constants;


//=============== VERTEX SHADER ===============//
@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    // Reconstruct model_matrix from instance input
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_col_0,
        instance.model_matrix_col_1,
        instance.model_matrix_col_2,
        instance.model_matrix_col_3
    );

    var out: VertexOutput;
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);
    let clip_pos = light_vp.view_proj * world_pos;

    // NDC z is already 0..1 for WebGPU.
    let ndc_z = clip_pos.z / clip_pos.w;
    out.clip_position = clip_pos;
    out.depth = ndc_z;

    return out;
}

fn evsm_moments(depth: f32) -> vec2<f32> {
    let warped_depth = exp(render_constants.evsm_c * (depth - 1.0));
    return vec2<f32>(warped_depth, warped_depth * warped_depth);
}

//=============== FRAGMENT SHADER ===============//
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<f32> {
    // Variance Shadow Mapping (VSM)
    // We store the warped depth (moment 1) and warped depth squared (moment 2)
    let depth = clamp(in.depth, 0.0, 1.0);
    return evsm_moments(depth);
}

@fragment
fn fs_main_rgba(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = clamp(in.depth, 0.0, 1.0);
    let moments = evsm_moments(depth);
    return vec4<f32>(moments, 0.0, 1.0);
}