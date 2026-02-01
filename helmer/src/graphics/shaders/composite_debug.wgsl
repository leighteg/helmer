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

const PI: f32 = 3.14159265359;
const MAX_REFLECTION_LOD: f32 = 4.0;
const EMISSIVE_THRESHOLD: f32 = 0.01;

const FLAG_UNLIT: u32 = 1u;
const FLAG_DIRECT: u32 = 1u << 1u;
const FLAG_INDIRECT: u32 = 1u << 2u;
const FLAG_REFLECTIONS: u32 = 1u << 3u;
const FLAG_EMISSION: u32 = 1u << 4u;
const FLAG_SKY: u32 = 1u << 5u;

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
    _pad_light: vec4<u32>,
    prev_view_proj: mat4x4<f32>,
    frame_index: u32,
    _padding: vec3<u32>,
    _pad_end: vec4<u32>,
}

struct DebugCompositeParams {
    flags: u32,
    _pad0: vec3<u32>,
    _pad1: vec4<u32>,
}

@group(0) @binding(0) var direct_lighting_tex: texture_2d<f32>;
@group(0) @binding(1) var ssgi_tex: texture_2d<f32>;
@group(0) @binding(2) var ssr_tex: texture_2d<f32>;
@group(0) @binding(3) var albedo_tex: texture_2d<f32>;
@group(0) @binding(4) var emission_tex: texture_2d<f32>;
@group(0) @binding(5) var scene_sampler: sampler;
@group(0) @binding(6) var normal_tex: texture_2d<f32>;
@group(0) @binding(7) var mra_tex: texture_2d<f32>;
@group(0) @binding(8) var depth_tex: texture_2d<f32>;
@group(0) @binding(9) var sky_tex: texture_2d<f32>;

@group(1) @binding(0) var brdf_lut: texture_2d<f32>;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var prefiltered_env_map: texture_cube<f32>;
@group(1) @binding(3) var ibl_sampler: sampler;
@group(1) @binding(4) var brdf_lut_sampler: sampler;

@group(2) @binding(0) var<uniform> camera: CameraUniforms;
@group(3) @binding(0) var<uniform> constants: Constants;
@group(4) @binding(0) var<uniform> debug: DebugCompositeParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

fn fresnel_schlick_roughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0)
        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn sample_depth(tex: texture_2d<f32>, uv: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(tex, 0));
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let max_coord = max(size - vec2<i32>(1), vec2<i32>(0));
    let coord = clamp(vec2<i32>(clamped_uv * vec2<f32>(size)), vec2<i32>(0), max_coord);
    return textureLoad(tex, coord, 0).x;
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = vec2<f32>(
        f32((in_vertex_index << 1u) & 2u),
        f32(in_vertex_index & 2u)
    );
    out.clip_position = vec4<f32>(
        out.tex_coords.x * 2.0 - 1.0,
        1.0 - out.tex_coords.y * 2.0,
        0.0,
        1.0
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = sample_depth(depth_tex, in.tex_coords);
    if (depth <= 0.0) {
        if ((debug.flags & FLAG_SKY) != 0u) {
            let sky_color = textureSampleLevel(sky_tex, scene_sampler, in.tex_coords, 0.0).rgb;
            return vec4<f32>(sky_color, 1.0);
        }
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let albedo_sample = textureSampleLevel(albedo_tex, scene_sampler, in.tex_coords, 0.0);
    let albedo = albedo_sample.rgb;
    let alpha = albedo_sample.a;
    let emission = textureSampleLevel(emission_tex, scene_sampler, in.tex_coords, 0.0).rgb;
    let emission_strength = length(emission);

    let direct_light = textureSampleLevel(direct_lighting_tex, scene_sampler, in.tex_coords, 0.0).rgb;
    let indirect_diffuse_ssgi = textureSampleLevel(ssgi_tex, scene_sampler, in.tex_coords, 0.0).rgb;
    let ssr_sample = textureSampleLevel(ssr_tex, scene_sampler, in.tex_coords, 0.0);
    let indirect_specular_ssr = ssr_sample.rgb;
    let ssr_confidence = ssr_sample.a;

    let mra = textureSampleLevel(mra_tex, scene_sampler, in.tex_coords, 0.0);
    let metallic = mra.r;
    let roughness = mra.g;
    let raw_ao = mra.b;
    let ao = mix(0.25, 1.0, raw_ao);

    let packed_normal = textureSampleLevel(normal_tex, scene_sampler, in.tex_coords, 0.0).xyz;
    let N = normalize(packed_normal * 2.0 - 1.0);

    let ndc = vec4<f32>(in.tex_coords.x * 2.0 - 1.0, (1.0 - in.tex_coords.y) * 2.0 - 1.0, depth, 1.0);
    let world_pos_h = camera.inverse_view_projection_matrix * ndc;
    let world_position = world_pos_h.xyz / world_pos_h.w;

    let V = normalize(camera.view_position - world_position);
    let R = reflect(-V, N);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    let F_ibl = fresnel_schlick_roughness(max(dot(N, V), 0.0), F0, roughness);
    let kS_ibl = F_ibl;
    var kD_ibl = vec3(1.0) - kS_ibl;
    kD_ibl *= (1.0 - metallic);

    let irradiance = textureSampleLevel(irradiance_map, ibl_sampler, N, 0.0).rgb;
    let diffuse_ibl = irradiance * albedo;

    let prefiltered_color = textureSampleLevel(prefiltered_env_map, ibl_sampler, R, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSampleLevel(brdf_lut, brdf_lut_sampler, vec2<f32>(max(dot(N, V), 0.0), roughness), 0.0).rg;
    let specular_ibl = prefiltered_color * (F_ibl * brdf.x + brdf.y);

    let total_incoming_indirect_diffuse = (indirect_diffuse_ssgi * constants.ssgi_intensity) + irradiance;
    let indirect_diffuse = select(total_incoming_indirect_diffuse, total_incoming_indirect_diffuse * albedo, constants.shade_mode != 2u) * kD_ibl * ao;
    let indirect_specular = mix(specular_ibl * ao, indirect_specular_ssr, ssr_confidence);

    var color = vec3<f32>(0.0);
    if ((debug.flags & FLAG_UNLIT) != 0u) {
        color += albedo;
    }
    if ((debug.flags & FLAG_DIRECT) != 0u) {
        color += direct_light;
    }
    if ((debug.flags & FLAG_INDIRECT) != 0u) {
        color += indirect_diffuse;
    }
    if ((debug.flags & FLAG_REFLECTIONS) != 0u) {
        color += indirect_specular;
    }
    if ((debug.flags & FLAG_EMISSION) != 0u) {
        color += emission;
    }

    if (emission_strength > EMISSIVE_THRESHOLD && (debug.flags & FLAG_EMISSION) == 0u) {
        // Avoid black emissive-only surfaces when emission toggle is off.
        color = select(color, vec3(0.0), emission_strength > EMISSIVE_THRESHOLD);
    }

    let tonemapped = color / (color + vec3<f32>(1.0));
    let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma_corrected, alpha);
}
