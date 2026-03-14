const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const MAX_SHADOW_CASCADES: u32 = 4u;
const MAX_WEBGL_LIGHTS: u32 = 256u;
const EPSILON: f32 = 0.00001;
const EMISSIVE_THRESHOLD: f32 = 0.01;

const ALPHA_MODE_OPAQUE: u32 = 0u;
const ALPHA_MODE_MASK: u32 = 1u;
const ALPHA_MODE_BLEND: u32 = 2u;
const ALPHA_MODE_PREMULTIPLIED: u32 = 3u;
const ALPHA_MODE_ADDITIVE: u32 = 4u;

const POISSON_DISK_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2(-0.94201624, -0.39906216),
    vec2(0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845),
    vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554),
    vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507),
    vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790)
);

struct Constants {
    mip_bias: f32,
    shade_mode: u32,
    shade_smooth: u32,
    light_model: u32,
    skylight_contribution: u32,
    planet_radius: f32,
    atmosphere_radius: f32,
    sky_light_samples: u32,
    _pad0: u32,
    ssr_coarse_steps: u32,
    ssr_binary_search_steps: u32,
    ssr_linear_step_size: f32,
    ssr_thickness: f32,
    ssr_max_distance: f32,
    ssr_roughness_fade_start: f32,
    ssr_roughness_fade_end: f32,
    _pad1: u32,
    ssgi_num_rays: u32,
    ssgi_num_steps: u32,
    ssgi_ray_step_size: f32,
    ssgi_thickness: f32,
    ssgi_blend_factor: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    evsm_c: f32,
    pcf_radius: u32,
    pcf_min_scale: f32,
    pcf_max_scale: f32,
    pcf_max_distance: f32,
    ssgi_intensity: f32,
    _final_padding: vec4<f32>,
};

struct MaterialData {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,
    albedo_idx: i32,
    normal_idx: i32,
    metallic_roughness_idx: i32,
    emission_idx: i32,
    emission_color: vec3<f32>,
    _padding: f32,
    alpha_mode: u32,
    alpha_cutoff: f32,
    _pad_alpha0: u32,
    _pad_alpha1: u32,
}

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

struct SkyUniforms {
    sun_direction: vec3<f32>,
    _padding: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    ground_albedo: vec3<f32>,
    ground_brightness: f32,
    night_ambient_color: vec3<f32>,
    sun_angular_radius_cos: f32,
};

struct LightData {
    position: vec3<f32>,
    light_type: u32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    _padding: f32,
}

struct CascadeData {
    light_view_proj: mat4x4<f32>,
    split_depth: vec4<f32>,
}

struct ShadowUniforms {
    cascade_count: u32,
    _pad0: vec3<u32>,
    cascades: array<CascadeData, MAX_SHADOW_CASCADES>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
}

struct InstanceInput {
    @location(6) model_matrix_col_0: vec4<f32>,
    @location(7) model_matrix_col_1: vec4<f32>,
    @location(8) model_matrix_col_2: vec4<f32>,
    @location(9) model_matrix_col_3: vec4<f32>,
    @location(10) material_id: u32,
    @location(11) visibility: u32,
    @location(12) skin_offset: u32,
    @location(13) skin_count: u32,
}

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
    @location(5) @interpolate(flat) material_id: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> skin_matrices: array<mat4x4<f32>>;

@group(1) @binding(0) var<uniform> lights_buffer: array<LightData, MAX_WEBGL_LIGHTS>;
@group(1) @binding(1) var shadow_map: texture_2d_array<f32>;
@group(1) @binding(2) var shadow_sampler: sampler;
@group(1) @binding(3) var<uniform> shadow_uniforms: ShadowUniforms;
@group(1) @binding(4) var<uniform> sky: SkyUniforms;
@group(1) @binding(5) var<uniform> constants: Constants;
@group(1) @binding(6) var brdf_lut: texture_2d<f32>;
@group(1) @binding(7) var irradiance_map: texture_cube<f32>;
@group(1) @binding(8) var prefiltered_env_map: texture_cube<f32>;
@group(1) @binding(9) var ibl_sampler: sampler;
@group(1) @binding(10) var brdf_lut_sampler: sampler;

@group(2) @binding(0) var<storage, read> materials: array<MaterialData>;
@group(2) @binding(1) var textures: binding_array<texture_2d<f32>>;
@group(2) @binding(2) var pbr_sampler: sampler;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len < EPSILON { return vec3<f32>(0.0, 0.0, 1.0); }
    return v / len;
}

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if abs(det) < EPSILON {
        return mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    }

    let inv_det = 1.0 / det;
    var res: mat3x3<f32>;
    res[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det;
    res[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    res[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    res[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    res[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    res[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det;
    res[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det;
    res[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det;
    res[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det;
    return res;
}

struct SkinnedVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    tangent: vec3<f32>,
}

fn apply_skinning(vertex: VertexInput, skin_offset: u32, skin_count: u32) -> SkinnedVertex {
    var out: SkinnedVertex;
    out.position = vertex.position;
    out.normal = vertex.normal;
    out.tangent = vertex.tangent.xyz;

    if (skin_count == 0u) {
        return out;
    }

    let weights = vertex.weights;
    let joint0 = min(vertex.joints.x, skin_count - 1u);
    let joint1 = min(vertex.joints.y, skin_count - 1u);
    let joint2 = min(vertex.joints.z, skin_count - 1u);
    let joint3 = min(vertex.joints.w, skin_count - 1u);

    var skinned_pos = vec4<f32>(0.0);
    var skinned_norm = vec3<f32>(0.0);
    var skinned_tan = vec3<f32>(0.0);

    if (weights.x > 0.0) {
        let m = skin_matrices[skin_offset + joint0];
        skinned_pos += (m * vec4<f32>(vertex.position, 1.0)) * weights.x;
        skinned_norm += (m * vec4<f32>(vertex.normal, 0.0)).xyz * weights.x;
        skinned_tan += (m * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz * weights.x;
    }
    if (weights.y > 0.0) {
        let m = skin_matrices[skin_offset + joint1];
        skinned_pos += (m * vec4<f32>(vertex.position, 1.0)) * weights.y;
        skinned_norm += (m * vec4<f32>(vertex.normal, 0.0)).xyz * weights.y;
        skinned_tan += (m * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz * weights.y;
    }
    if (weights.z > 0.0) {
        let m = skin_matrices[skin_offset + joint2];
        skinned_pos += (m * vec4<f32>(vertex.position, 1.0)) * weights.z;
        skinned_norm += (m * vec4<f32>(vertex.normal, 0.0)).xyz * weights.z;
        skinned_tan += (m * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz * weights.z;
    }
    if (weights.w > 0.0) {
        let m = skin_matrices[skin_offset + joint3];
        skinned_pos += (m * vec4<f32>(vertex.position, 1.0)) * weights.w;
        skinned_norm += (m * vec4<f32>(vertex.normal, 0.0)).xyz * weights.w;
        skinned_tan += (m * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz * weights.w;
    }

    out.position = skinned_pos.xyz;
    out.normal = skinned_norm;
    out.tangent = skinned_tan;
    return out;
}

fn distribution_ggx(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH2 = NdotH * NdotH;
    let denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0)
        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn chebyshev_inequality(depth: f32, moments: vec2<f32>) -> f32 {
    var current_depth = depth;
    current_depth = exp(constants.evsm_c * (current_depth - 1.0));
    if current_depth <= moments.x {
        return 1.0;
    }

    var variance = moments.y - (moments.x * moments.x);
    variance = max(variance, 0.0);

    let d = current_depth - moments.x;
    let p_max = variance / (variance + d * d);

    return smoothstep(0.2, 1.0, p_max);
}

fn calculate_shadow_factor(
    world_pos: vec3<f32>,
    view_z: f32,
    N: vec3<f32>,
    L: vec3<f32>
) -> f32 {
    let cascade_count = max(1u, min(shadow_uniforms.cascade_count, MAX_SHADOW_CASCADES));
    var cascade_index = i32(cascade_count - 1u);
    for (var i = 0u; i < cascade_count; i = i + 1u) {
        if view_z > shadow_uniforms.cascades[i].split_depth.x {
            cascade_index = i32(i);
            break;
        }
    }
    let cascade = shadow_uniforms.cascades[cascade_index];

    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);
    if shadow_pos_clip.w < EPSILON {
        return 1.0;
    }

    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);

    if any(shadow_uv < vec2(0.0)) || any(shadow_uv > vec2(1.0)) || shadow_coord.z < 0.0 || shadow_coord.z > 1.0 {
        return 1.0;
    }

    let base_radius = f32(constants.pcf_radius);
    let dist_factor = clamp(view_z / constants.pcf_max_distance, 0.0, 1.0);
    let scale = mix(constants.pcf_min_scale, constants.pcf_max_scale, dist_factor);
    let filter_radius = base_radius * scale;

    var sample_count = 16;
    if cascade_index > 1 || dist_factor > 0.7 {
        sample_count = 8;
    }

    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map, 0u));
    let texel_size = filter_radius / shadow_map_size;

    var total_moments = vec2<f32>(0.0);
    if sample_count == 16 {
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[0] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[1] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[2] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[3] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[4] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[5] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[6] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[7] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[8] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[9] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[10] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[11] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[12] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[13] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[14] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[15] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments *= (1.0 / 16.0);
    } else {
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[0] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[2] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[4] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[6] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[8] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[10] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[12] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments += textureSampleLevel(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[14] * texel_size, u32(cascade_index), 0.0).rg;
        total_moments *= (1.0 / 8.0);
    }

    return chebyshev_inequality(shadow_coord.z, total_moments);
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VSOut {
    var out: VSOut;
    if (instance.visibility == 0u) {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.world_position = vec3<f32>(0.0);
        out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
        out.world_tangent = vec3<f32>(1.0, 0.0, 0.0);
        out.world_bitangent = vec3<f32>(0.0, 1.0, 0.0);
        out.tex_coord = vec2<f32>(0.0);
        out.material_id = 0u;
        return out;
    }
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_col_0,
        instance.model_matrix_col_1,
        instance.model_matrix_col_2,
        instance.model_matrix_col_3
    );

    let skinned = apply_skinning(vertex, instance.skin_offset, instance.skin_count);
    let world_pos = model_matrix * vec4<f32>(skinned.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.projection_matrix * camera.view_matrix * world_pos;

    let model_mat3 = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    );
    let normal_matrix = transpose(mat3_inverse(model_mat3));
    let N = safe_normalize(normal_matrix * skinned.normal);
    let T = safe_normalize(normal_matrix * skinned.tangent);
    let B = cross(N, T) * vertex.tangent.w;

    out.world_normal = N;
    out.world_tangent = T;
    out.world_bitangent = B;
    out.tex_coord = vertex.tex_coord;
    out.material_id = instance.material_id;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let material = materials[in.material_id];
    let mip_bias = constants.mip_bias;

    let has_albedo = material.albedo_idx >= 0i;
    let albedo_idx = select(0i, material.albedo_idx, has_albedo);
    let albedo_sample = textureSampleBias(textures[albedo_idx], pbr_sampler, in.tex_coord, mip_bias);
    var albedo = material.albedo.rgb * select(vec3<f32>(1.0), albedo_sample.rgb, has_albedo);
    var alpha = material.albedo.a * select(1.0, albedo_sample.a, has_albedo);

    let has_mra = material.metallic_roughness_idx >= 0i;
    let mra_idx = select(0i, material.metallic_roughness_idx, has_mra);
    let mra_sample = textureSampleBias(textures[mra_idx], pbr_sampler, in.tex_coord, mip_bias);
    let mra_factor = select(vec3<f32>(1.0), mra_sample.rgb, has_mra);
    var metallic = material.metallic * mra_factor.b;
    var roughness = material.roughness * mra_factor.g;
    let ao_raw = material.ao * mra_factor.r;

    let has_emission = material.emission_idx >= 0i;
    let emission_idx = select(0i, material.emission_idx, has_emission);
    let emission_sample = textureSampleBias(textures[emission_idx], pbr_sampler, in.tex_coord, mip_bias).rgb;
    let emission = material.emission_color * material.emission_strength
        * select(vec3<f32>(1.0), emission_sample, has_emission);

    let smooth_normal = safe_normalize(in.world_normal);
    var geom_normal = smooth_normal;
    if constants.shade_smooth == 0u {
        var flat_normal = safe_normalize(cross(dpdx(in.world_position), dpdy(in.world_position)));
        if dot(flat_normal, smooth_normal) < 0.0 {
            flat_normal = -flat_normal;
        }
        geom_normal = flat_normal;
    }

    let has_normal = material.normal_idx >= 0i;
    let normal_idx = select(0i, material.normal_idx, has_normal);
    let tangent_space_normal =
        textureSampleBias(textures[normal_idx], pbr_sampler, in.tex_coord, mip_bias).xyz * 2.0 - 1.0;
    let T = safe_normalize(in.world_tangent - geom_normal * dot(geom_normal, in.world_tangent));
    var B = safe_normalize(cross(geom_normal, T));
    if dot(B, in.world_bitangent) < 0.0 {
        B = -B;
    }
    let tbn = mat3x3<f32>(T, B, geom_normal);
    let mapped_normal = safe_normalize(tbn * tangent_space_normal);
    let N = select(geom_normal, mapped_normal, has_normal);

    let alpha_mode = material.alpha_mode;
    if alpha_mode == ALPHA_MODE_MASK && alpha < material.alpha_cutoff {
        discard;
    }
    if alpha_mode == ALPHA_MODE_OPAQUE {
        alpha = 1.0;
    }

    roughness = max(roughness, MIN_ROUGHNESS);

    var ao_direct = ao_raw;
    ao_direct = mix(ao_direct, 1.0, 1.0 - smoothstep(0.0, 0.1, ao_direct));
    let ao_indirect = mix(0.25, 1.0, ao_raw);

    let V = safe_normalize(camera.view_position - in.world_position);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    var direct_lighting = vec3<f32>(0.0);
    if constants.shade_mode != 1u {
        let view_pos = camera.view_matrix * vec4<f32>(in.world_position, 1.0);
        let sun_height_factor = max(sky.sun_direction.y, 0.0);
        let sun_fade = pow(sun_height_factor, 1.5);
        let is_lit = constants.shade_mode == 0u;
        let light_count = min(camera.light_count, MAX_WEBGL_LIGHTS);
        for (var i = 0u; i < light_count; i = i + 1u) {
            let light = lights_buffer[i];
            var L: vec3<f32>;
            var radiance: vec3<f32>;
            var shadow_multiplier = 1.0;

            if light.light_type == 0u {
                L = safe_normalize(-light.direction);
                radiance = light.color * light.intensity * sun_fade;
                let NdotL = max(dot(N, L), 0.0);
                let bias_amount = 0.001 + 0.005 * (1.0 - NdotL);
                let biased_world_pos = in.world_position + N * bias_amount;
                shadow_multiplier = calculate_shadow_factor(biased_world_pos, view_pos.z, N, L);
            } else {
                let to_light_vector = light.position - in.world_position;
                let dist_sq = dot(to_light_vector, to_light_vector);
                if dist_sq < EPSILON { continue; }
                L = to_light_vector / sqrt(dist_sq);
                let attenuation = 1.0 / (dist_sq + 1.0);
                radiance = light.color * light.intensity * attenuation;
            }

            let NdotL = max(dot(N, L), 0.0);
            if NdotL > 0.0 {
                let H = safe_normalize(V + L);
                let NdotH = max(dot(N, H), 0.0);
                let NdotV = max(dot(N, V), 0.0);

                let NDF = distribution_ggx(NdotH, roughness);
                let G = geometry_smith(N, V, L, roughness);
                let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

                let specular_numerator = NDF * G * F;
                let specular_denominator = 4.0 * NdotV * NdotL + EPSILON;
                let specular_brdf = specular_numerator / specular_denominator;

                let diffuse_brdf = albedo / PI;

                let kS = F;
                let kD = vec3<f32>(1.0) - kS;

                let is_stylized = constants.light_model == 1u;
                let final_radiance = select(radiance * NdotL * shadow_multiplier, radiance * shadow_multiplier, is_stylized);

                let current_pbr = select((kD * (1.0 - metallic) * PI + specular_brdf), (kD * (1.0 - metallic) * diffuse_brdf + specular_brdf), is_lit) * final_radiance;

                direct_lighting += current_pbr;
            }
        }
    }

    direct_lighting *= ao_direct;

    var indirect_diffuse = vec3<f32>(0.0);
    var indirect_specular = vec3<f32>(0.0);
    if constants.shade_mode != 1u {
        let R = reflect(-V, N);
        let F_ibl = fresnel_schlick_roughness(max(dot(N, V), 0.0), F0, roughness);
        let kS_ibl = F_ibl;
        var kD_ibl = vec3(1.0) - kS_ibl;
        kD_ibl *= (1.0 - metallic);

        let irradiance = textureSampleLevel(irradiance_map, ibl_sampler, N, 0.0).rgb;
        let max_lod = max(f32(textureNumLevels(prefiltered_env_map)) - 1.0, 0.0);
        let prefiltered_color = textureSampleLevel(prefiltered_env_map, ibl_sampler, R, roughness * max_lod).rgb;
        let brdf = textureSampleLevel(brdf_lut, brdf_lut_sampler, vec2<f32>(max(dot(N, V), 0.0), roughness), 0.0).rg;
        let specular_ibl = prefiltered_color * (F_ibl * brdf.x + brdf.y);

        let diffuse_base = select(irradiance, irradiance * albedo, constants.shade_mode != 2u);
        indirect_diffuse = diffuse_base * kD_ibl * ao_indirect;
        indirect_specular = specular_ibl * ao_indirect;
    }

    let emission_strength = length(emission);
    let lit_color = direct_lighting + indirect_diffuse + indirect_specular;
    var final_hdr = select(lit_color, vec3(0.0), emission_strength > EMISSIVE_THRESHOLD) + emission;
    if constants.shade_mode == 1u {
        final_hdr = albedo + emission;
    }

    let tonemapped = final_hdr / (final_hdr + vec3<f32>(1.0));
    var color = pow(tonemapped, vec3<f32>(1.0 / 2.2));

    if alpha_mode == ALPHA_MODE_PREMULTIPLIED || alpha_mode == ALPHA_MODE_ADDITIVE {
        color *= alpha;
    }

    return vec4<f32>(color, alpha);
}
