enable wgpu_mesh_shader;

const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const MAX_SHADOW_CASCADES: u32 = 4u;
const MAX_VERTS: u32 = 64u;
const MAX_PRIMS: u32 = 124u;
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

struct GBufferInstance {
    model_matrix: mat4x4<f32>,
    material_id: u32,
    visibility: u32,
    skin_offset: u32,
    skin_count: u32,
    bounds_center: vec4<f32>,
    bounds_extents: vec4<f32>,
}

struct MeshletDesc {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
    bounds_center: vec3<f32>,
    bounds_radius: f32,
}

struct MeshDrawParams {
    instance_base: u32,
    instance_count: u32,
    meshlet_base: u32,
    meshlet_count: u32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    depth_bias: f32,
    rect_pad: f32,
    _pad3: vec2<f32>,
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

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
}

struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
    @builtin(vertices) vertices: array<VSOut, MAX_VERTS>,
    @builtin(primitives) primitives: array<PrimitiveOutput, MAX_PRIMS>,
}

var<workgroup> mesh_output: MeshOutput;
var<workgroup> mesh_visible: u32;
var<workgroup> mesh_model: mat4x4<f32>;
var<workgroup> mesh_material_id: u32;
var<workgroup> mesh_skin_offset: u32;
var<workgroup> mesh_skin_count: u32;
var<workgroup> mesh_normal_matrix: mat3x3<f32>;
var<workgroup> mesh_vert_count: u32;
var<workgroup> mesh_prim_count: u32;
var<workgroup> meshlet_vertex_offset: u32;
var<workgroup> meshlet_index_offset: u32;
var<workgroup> meshlet_center: vec3<f32>;
var<workgroup> meshlet_radius: f32;

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> skin_matrices: array<mat4x4<f32>>;

@group(1) @binding(0) var<storage, read> lights_buffer: array<LightData>;
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
@group(2) @binding(1) var albedo_tex: texture_2d<f32>;
@group(2) @binding(2) var normal_tex: texture_2d<f32>;
@group(2) @binding(3) var metallic_roughness_tex: texture_2d<f32>;
@group(2) @binding(4) var emission_tex: texture_2d<f32>;
@group(2) @binding(5) var pbr_sampler: sampler;

@group(3) @binding(0) var<storage, read> instances: array<GBufferInstance>;
@group(3) @binding(1) var<storage, read> meshlet_descs: array<MeshletDesc>;
@group(3) @binding(2) var<storage, read> meshlet_vertices: array<u32>;
@group(3) @binding(3) var<storage, read> meshlet_indices: array<u32>;
@group(3) @binding(4) var<storage, read> vertex_data: array<u32>;
@group(3) @binding(5) var<uniform> draw_params: MeshDrawParams;
@group(3) @binding(6) var hiz_tex: texture_2d<f32>;

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

fn load_f32(index: u32) -> f32 {
    return bitcast<f32>(vertex_data[index]);
}

fn load_u32(index: u32) -> u32 {
    return vertex_data[index];
}

fn load_vec2(base: u32) -> vec2<f32> {
    return vec2<f32>(load_f32(base), load_f32(base + 1u));
}

fn load_vec3(base: u32) -> vec3<f32> {
    return vec3<f32>(load_f32(base), load_f32(base + 1u), load_f32(base + 2u));
}

fn load_vec4(base: u32) -> vec4<f32> {
    return vec4<f32>(
        load_f32(base),
        load_f32(base + 1u),
        load_f32(base + 2u),
        load_f32(base + 3u)
    );
}

fn load_uvec4(base: u32) -> vec4<u32> {
    return vec4<u32>(
        load_u32(base),
        load_u32(base + 1u),
        load_u32(base + 2u),
        load_u32(base + 3u)
    );
}

fn max_scale(model: mat4x4<f32>) -> f32 {
    let x = length(model[0].xyz);
    let y = length(model[1].xyz);
    let z = length(model[2].xyz);
    return max(x, max(y, z));
}

struct SkinnedVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    tangent: vec4<f32>,
}

fn apply_skinning(
    position: vec3<f32>,
    normal: vec3<f32>,
    tangent: vec4<f32>,
    joints: vec4<u32>,
    weights: vec4<f32>,
    skin_offset: u32,
    skin_count: u32
) -> SkinnedVertex {
    if (skin_count == 0u) {
        return SkinnedVertex(position, normal, tangent);
    }

    let joint0 = min(joints.x, skin_count - 1u);
    let joint1 = min(joints.y, skin_count - 1u);
    let joint2 = min(joints.z, skin_count - 1u);
    let joint3 = min(joints.w, skin_count - 1u);

    var skinned_pos = vec3<f32>(0.0);
    var skinned_norm = vec3<f32>(0.0);
    var skinned_tan = vec3<f32>(0.0);

    if (weights.x > 0.0) {
        let m = skin_matrices[skin_offset + joint0];
        let n = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
        skinned_pos += weights.x * (m * vec4<f32>(position, 1.0)).xyz;
        skinned_norm += weights.x * (n * normal);
        skinned_tan += weights.x * (n * tangent.xyz);
    }
    if (weights.y > 0.0) {
        let m = skin_matrices[skin_offset + joint1];
        let n = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
        skinned_pos += weights.y * (m * vec4<f32>(position, 1.0)).xyz;
        skinned_norm += weights.y * (n * normal);
        skinned_tan += weights.y * (n * tangent.xyz);
    }
    if (weights.z > 0.0) {
        let m = skin_matrices[skin_offset + joint2];
        let n = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
        skinned_pos += weights.z * (m * vec4<f32>(position, 1.0)).xyz;
        skinned_norm += weights.z * (n * normal);
        skinned_tan += weights.z * (n * tangent.xyz);
    }
    if (weights.w > 0.0) {
        let m = skin_matrices[skin_offset + joint3];
        let n = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
        skinned_pos += weights.w * (m * vec4<f32>(position, 1.0)).xyz;
        skinned_norm += weights.w * (n * normal);
        skinned_tan += weights.w * (n * tangent.xyz);
    }

    return SkinnedVertex(skinned_pos, skinned_norm, vec4<f32>(skinned_tan, tangent.w));
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

fn sphere_in_frustum(view_proj: mat4x4<f32>, center: vec3<f32>, radius: f32) -> bool {
    let row0 = vec4<f32>(view_proj[0][0], view_proj[1][0], view_proj[2][0], view_proj[3][0]);
    let row1 = vec4<f32>(view_proj[0][1], view_proj[1][1], view_proj[2][1], view_proj[3][1]);
    let row2 = vec4<f32>(view_proj[0][2], view_proj[1][2], view_proj[2][2], view_proj[3][2]);
    let row3 = vec4<f32>(view_proj[0][3], view_proj[1][3], view_proj[2][3], view_proj[3][3]);

    let planes = array<vec4<f32>, 6>(
        row3 + row0,
        row3 - row0,
        row3 + row1,
        row3 - row1,
        row3 + row2,
        row3 - row2
    );

    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = planes[i];
        let dist = dot(plane.xyz, center) + plane.w;
        let len = length(plane.xyz);
        if (dist < -radius * len) {
            return false;
        }
    }

    return true;
}

@mesh(mesh_output)
@workgroup_size(64)
fn ms_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    if (local_id.x == 0u) {
        mesh_visible = 1u;
        mesh_vert_count = 0u;
        mesh_prim_count = 0u;
        if (workgroup_id.y >= draw_params.instance_count) {
            mesh_visible = 0u;
        } else {
            let instance_index = draw_params.instance_base + workgroup_id.y;
            let inst = instances[instance_index];
            if (inst.visibility == 0u) {
                mesh_visible = 0u;
            }

            if (workgroup_id.x >= draw_params.meshlet_count) {
                mesh_visible = 0u;
            } else {
                let meshlet_index = draw_params.meshlet_base + workgroup_id.x;
                let meshlet = meshlet_descs[meshlet_index];
                let vert_count = meshlet.vertex_count;
                let prim_count = meshlet.index_count / 3u;
                if (vert_count > MAX_VERTS || prim_count > MAX_PRIMS) {
                    mesh_visible = 0u;
                } else {
                    mesh_vert_count = vert_count;
                    mesh_prim_count = prim_count;
                    meshlet_vertex_offset = meshlet.vertex_offset;
                    meshlet_index_offset = meshlet.index_offset;
                    meshlet_center = meshlet.bounds_center;
                    meshlet_radius = meshlet.bounds_radius;

                    mesh_model = inst.model_matrix;
                    mesh_material_id = inst.material_id;
                    mesh_skin_offset = inst.skin_offset;
                    mesh_skin_count = inst.skin_count;
                    let model_mat3 = mat3x3<f32>(
                        mesh_model[0].xyz,
                        mesh_model[1].xyz,
                        mesh_model[2].xyz
                    );
                    mesh_normal_matrix = transpose(mat3_inverse(model_mat3));

                    let flags = draw_params.flags;
                    let frustum_enabled = (flags & 1u) != 0u;
                    let occlusion_enabled = (flags & 2u) != 0u;

                    if (mesh_visible != 0u) {
                        let world_center =
                            (mesh_model * vec4<f32>(meshlet_center, 1.0)).xyz;
                        let world_radius = meshlet_radius * max_scale(mesh_model);
                        let view_proj = camera.projection_matrix * camera.view_matrix;

                        if (frustum_enabled) {
                            if (!sphere_in_frustum(view_proj, world_center, world_radius)) {
                                mesh_visible = 0u;
                            }
                        }

                        if (mesh_visible != 0u && occlusion_enabled) {
                            let prev_view_proj = camera.prev_view_proj;
                            let extents = vec3<f32>(world_radius);
                            let signs = array<vec3<f32>, 8>(
                                vec3<f32>(-1.0, -1.0, -1.0),
                                vec3<f32>(1.0, -1.0, -1.0),
                                vec3<f32>(-1.0, 1.0, -1.0),
                                vec3<f32>(1.0, 1.0, -1.0),
                                vec3<f32>(-1.0, -1.0, 1.0),
                                vec3<f32>(1.0, -1.0, 1.0),
                                vec3<f32>(-1.0, 1.0, 1.0),
                                vec3<f32>(1.0, 1.0, 1.0)
                            );

                            var min_ndc = vec2<f32>(1.0, 1.0);
                            var max_ndc = vec2<f32>(-1.0, -1.0);
                            var max_depth = 0.0;
                            var valid_prev = false;
                            var valid_any = false;

                            for (var i = 0u; i < 8u; i = i + 1u) {
                                let world = world_center + extents * signs[i];

                                let clip_prev = prev_view_proj * vec4<f32>(world, 1.0);
                                if (clip_prev.w > 0.0) {
                                    let ndc_prev = clip_prev.xyz / clip_prev.w;
                                    min_ndc = min(min_ndc, ndc_prev.xy);
                                    max_ndc = max(max_ndc, ndc_prev.xy);
                                    max_depth = max(max_depth, ndc_prev.z);
                                    valid_prev = true;
                                    valid_any = true;
                                }

                                let clip_cur = view_proj * vec4<f32>(world, 1.0);
                                if (clip_cur.w > 0.0) {
                                    valid_any = true;
                                }
                            }

                            if (valid_prev && valid_any) {
                                let uv_min = clamp(min_ndc * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
                                let uv_max = clamp(max_ndc * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
                                if (uv_min.x < uv_max.x && uv_min.y < uv_max.y) {
                                    let base_dims = vec2<f32>(textureDimensions(hiz_tex, 0));
                                    let rect = (uv_max - uv_min) * base_dims;
                                    let max_dim = max(rect.x, rect.y);

                                    let levels = max(textureNumLevels(hiz_tex), 1u);
                                    var mip = 0u;
                                    if (max_dim > 1.0 && levels > 1u) {
                                        let in = u32(floor(log2(max_dim)));
                                        mip = min(in + 1u, levels - 1u);
                                    }

                                    let mip_level = i32(mip);
                                    let mip_dims = vec2<i32>(textureDimensions(hiz_tex, mip_level));
                                    let mip_dims_f = vec2<f32>(mip_dims);
                                    let pad = vec2<f32>(draw_params.rect_pad);
                                    let min_f = clamp(uv_min * mip_dims_f - pad, vec2<f32>(0.0), mip_dims_f - vec2<f32>(1.0));
                                    let max_f = clamp(uv_max * mip_dims_f + pad, vec2<f32>(0.0), mip_dims_f - vec2<f32>(1.0));
                                    let min_tex = vec2<i32>(min_f);
                                    let max_tex = vec2<i32>(max_f);

                                    let d0 = textureLoad(hiz_tex, min_tex, mip_level).x;
                                    let d1 = textureLoad(hiz_tex, vec2<i32>(max_tex.x, min_tex.y), mip_level).x;
                                    let d2 = textureLoad(hiz_tex, vec2<i32>(min_tex.x, max_tex.y), mip_level).x;
                                    let d3 = textureLoad(hiz_tex, max_tex, mip_level).x;
                                    let hiz_depth = min(min(d0, d1), min(d2, d3));

                                    let depth = clamp(max_depth, 0.0, 1.0);
                                    if (depth + draw_params.depth_bias < hiz_depth) {
                                        mesh_visible = 0u;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (mesh_visible == 0u || mesh_vert_count == 0u || mesh_prim_count == 0u) {
            mesh_output.vertex_count = 0u;
            mesh_output.primitive_count = 0u;
        } else {
            mesh_output.vertex_count = mesh_vert_count;
            mesh_output.primitive_count = mesh_prim_count;
        }
    }

    workgroupBarrier();

    if (mesh_visible == 0u) {
        return;
    }

    if (local_id.x < mesh_vert_count) {
        let global_index = meshlet_vertices[meshlet_vertex_offset + local_id.x];
        let base = global_index * 20u;
        let position = load_vec3(base);
        let normal = load_vec3(base + 3u);
        let tex_coord = load_vec2(base + 6u);
        let tangent = load_vec4(base + 8u);
        let joints = load_uvec4(base + 12u);
        let weights = load_vec4(base + 16u);
        let skinned = apply_skinning(
            position,
            normal,
            tangent,
            joints,
            weights,
            mesh_skin_offset,
            mesh_skin_count
        );

        let world_pos = mesh_model * vec4<f32>(skinned.position, 1.0);
        let clip_position = camera.projection_matrix * camera.view_matrix * world_pos;

        let N = safe_normalize(mesh_normal_matrix * skinned.normal);
        let T = safe_normalize(mesh_normal_matrix * skinned.tangent.xyz);
        let B = cross(N, T) * skinned.tangent.w;

        mesh_output.vertices[local_id.x].clip_position = clip_position;
        mesh_output.vertices[local_id.x].world_position = world_pos.xyz;
        mesh_output.vertices[local_id.x].world_normal = N;
        mesh_output.vertices[local_id.x].tex_coord = tex_coord;
        mesh_output.vertices[local_id.x].world_tangent = T;
        mesh_output.vertices[local_id.x].world_bitangent = B;
        mesh_output.vertices[local_id.x].material_id = mesh_material_id;
    }

    for (var prim = local_id.x; prim < mesh_prim_count; prim = prim + 64u) {
        let base = meshlet_index_offset + prim * 3u;
        let i0 = meshlet_indices[base];
        let i1 = meshlet_indices[base + 1u];
        let i2 = meshlet_indices[base + 2u];
        mesh_output.primitives[prim].indices = vec3<u32>(i0, i1, i2);
    }
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let material = materials[in.material_id];
    let mip_bias = constants.mip_bias;

    let has_albedo = material.albedo_idx >= 0i;
    let albedo_sample = textureSampleBias(albedo_tex, pbr_sampler, in.tex_coord, mip_bias);
    var albedo = material.albedo.rgb * select(vec3<f32>(1.0), albedo_sample.rgb, has_albedo);
    var alpha = material.albedo.a * select(1.0, albedo_sample.a, has_albedo);

    let has_mra = material.metallic_roughness_idx >= 0i;
    let mra_sample = textureSampleBias(metallic_roughness_tex, pbr_sampler, in.tex_coord, mip_bias);
    let mra_factor = select(vec3<f32>(1.0), mra_sample.rgb, has_mra);
    var metallic = material.metallic * mra_factor.b;
    var roughness = material.roughness * mra_factor.g;
    let ao_raw = material.ao * mra_factor.r;

    let has_emission = material.emission_idx >= 0i;
    let emission_sample = textureSampleBias(emission_tex, pbr_sampler, in.tex_coord, mip_bias).rgb;
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
    let tangent_space_normal =
        textureSampleBias(normal_tex, pbr_sampler, in.tex_coord, mip_bias).xyz * 2.0 - 1.0;
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
        for (var i = 0u; i < camera.light_count; i = i + 1u) {
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
