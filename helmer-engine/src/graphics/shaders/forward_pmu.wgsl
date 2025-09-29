//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const NUM_CASCADES: u32 = 4u;
const EVSM_C = 20.0;
const EPSILON: f32 = 0.0001;
const MAX_REFLECTION_LOD: f32 = 4.0;

//=============== STRUCTS ===============//

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
    @location(5) view_space_depth: f32,
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
}

struct MaterialUniforms {
    albedo: vec4<f32>,
    emission_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,
    // No padding needed here, WGSL handles it
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

struct PbrConstants {
    model_matrix: mat4x4<f32>,
    material_id: u32,
    _p: vec3<u32>,
}

//=============== BINDINGS ===============//

// GROUP 0: Scene-wide data
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights_buffer: array<LightData>;
@group(0) @binding(2) var shadow_map: texture_2d_array<f32>;
@group(0) @binding(3) var shadow_sampler: sampler;
@group(0) @binding(4) var<uniform> shadow_uniforms: array<CascadeData, NUM_CASCADES>;

// GROUP 1: Per-Material data
@group(1) @binding(0) var pbr_sampler: sampler;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_texture: texture_2d<f32>;
@group(1) @binding(3) var mr_texture: texture_2d<f32>;
@group(1) @binding(4) var emission_texture: texture_2d<f32>;
@group(1) @binding(5) var<uniform> material: MaterialUniforms;

// GROUP 2 for Image-Based Lighting
@group(2) @binding(0) var brdf_lut: texture_2d<f32>;
@group(2) @binding(1) var irradiance_map: texture_cube<f32>;
@group(2) @binding(2) var prefiltered_env_map: texture_cube<f32>;
@group(2) @binding(3) var env_map_sampler: sampler; 
@group(2) @binding(4) var brdf_lut_sampler: sampler;

@vertex
var<push_constant> constants: PbrConstants;

//=============== UTILITY FUNCTIONS ===============//

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = determinant(m);
    if abs(det) < EPSILON {
        return mat3x3<f32>(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );
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
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn chebyshev_inequality(depth: f32, moments: vec2<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    var current_depth = depth;

    // Warp the depth value
    current_depth = exp(EVSM_C * (current_depth - 1.0));

    // Chebyshev test
    if current_depth <= moments.x {
        return 1.0;
    }

    var variance = moments.y - (moments.x * moments.x);
    variance = max(variance, 0.0);

    let d = current_depth - moments.x;
    let p_max = variance / (variance + d * d);

    return smoothstep(0.2, 1.0, p_max);
}

fn calculate_shadow_factor(world_pos: vec3<f32>, view_z: f32, N: vec3<f32>, L: vec3<f32>) -> f32 {
    var cascade_index = i32(NUM_CASCADES - 1);
    for (var i = 0i; i < i32(NUM_CASCADES); i = i + 1i) {
        if view_z > shadow_uniforms[i].split_depth.x {
            cascade_index = i;
            break;
        }
    }
    let cascade = shadow_uniforms[cascade_index];
    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);
    if shadow_pos_clip.w < EPSILON { return 1.0; }
    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);
    if any(shadow_uv < vec2(0.0)) || any(shadow_uv > vec2(1.0)) || shadow_coord.z < 0.0 || shadow_coord.z > 1.0 {
        return 1.0;
    }
    let moments = textureSample(shadow_map, shadow_sampler, shadow_uv, u32(cascade_index)).rg;
    return chebyshev_inequality(shadow_coord.z, moments, N, L);
}

//=============== VERTEX SHADER ===============//
@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_position_vec4 = constants.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;
    let view_pos = camera.view_matrix * world_position_vec4;
    out.clip_position = camera.projection_matrix * view_pos;
    out.view_space_depth = view_pos.z;
    let model_mat3 = mat3x3<f32>(constants.model_matrix[0].xyz, constants.model_matrix[1].xyz, constants.model_matrix[2].xyz);
    let normal_matrix = transpose(mat3_inverse(model_mat3));
    out.world_normal = normalize(normal_matrix * vertex.normal);
    let tangent_world = normalize(model_mat3 * vertex.tangent.xyz);
    out.world_tangent = normalize(tangent_world - dot(tangent_world, out.world_normal) * out.world_normal);
    out.world_bitangent = cross(out.world_normal, out.world_tangent) * vertex.tangent.w;
    out.tex_coord = vertex.tex_coord;
    return out;
}

//=============== FRAGMENT SHADER ===============//
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo_sample = textureSample(albedo_texture, pbr_sampler, in.tex_coord);
    let albedo = albedo_sample.rgb * material.albedo.rgb;
    let alpha = albedo_sample.a * material.albedo.a;

    let mr_sample = textureSample(mr_texture, pbr_sampler, in.tex_coord);
    let metallic = mr_sample.b * material.metallic;
    let roughness = max(mr_sample.g * material.roughness, MIN_ROUGHNESS);
    let ao = mr_sample.r * material.ao;

    var emission = material.emission_color * material.emission_strength;
    emission *= textureSample(emission_texture, pbr_sampler, in.tex_coord).rgb;

    let tangent_space_normal = textureSample(normal_texture, pbr_sampler, in.tex_coord).xyz * 2.0 - 1.0;
    let N_geom = normalize(in.world_normal);
    let T = normalize(in.world_tangent);
    let B = normalize(in.world_bitangent);
    let tbn = mat3x3<f32>(T, B, N_geom);
    let N = normalize(tbn * tangent_space_normal);

    let V = normalize(camera.view_position - in.world_position);
    let R = reflect(-V, N);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    
    // --- 1. DIRECT LIGHTING ---
    var Lo = vec3<f32>(0.0);

    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;
        var shadow_multiplier = 1.0;

        if light.light_type == 0u { // Directional
            L = normalize(-light.direction);
            radiance = light.color * light.intensity;

            let NdotL = max(dot(N, L), 0.0);
            let bias_amount = 0.001 + 0.005 * (1.0 - NdotL);
            let biased_world_position = in.world_position + N * bias_amount;
            shadow_multiplier = calculate_shadow_factor(biased_world_position, in.view_space_depth, N, L);
        } else { // Point
            let to_light = light.position - in.world_position;
            let dist_sq = dot(to_light, to_light);
            if dist_sq < EPSILON { continue; }
            L = to_light / sqrt(dist_sq);
            radiance = light.color * light.intensity / (dist_sq + 1.0);
        }

        let H = normalize(V + L);
        let NdotL = max(dot(N, L), 0.0);
        if NdotL > 0.0 {
            let NDF = distribution_ggx(max(dot(N, H), 0.0), roughness);
            let G = geometry_smith(N, V, L, roughness);
            let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + EPSILON;
            let specular = numerator / denominator;
            let kS = F;
            var kD = vec3<f32>(1.0) - kS;
            kD *= (1.0 - metallic);
            Lo += (kD * albedo / PI + specular) * radiance * NdotL * shadow_multiplier;
        }
    }
    
    // --- 2. INDIRECT (IMAGE-BASED) LIGHTING ---
    let F_ibl = fresnel_schlick_roughness(max(dot(N, V), 0.0), F0, roughness);
    let kS_ibl = F_ibl;
    var kD_ibl = vec3(1.0) - kS_ibl;
    kD_ibl *= (1.0 - metallic);

    let irradiance = textureSample(irradiance_map, env_map_sampler, N).rgb;
    let diffuse_indirect = irradiance * albedo;

    let prefiltered_color = textureSampleLevel(prefiltered_env_map, env_map_sampler, R, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSample(brdf_lut, brdf_lut_sampler, vec2<f32>(max(dot(N, V), 0.0), roughness)).rg;
    let specular_indirect = prefiltered_color * (F_ibl * brdf.x + brdf.y);

    let ambient = kD_ibl * diffuse_indirect * ao;

    let specular_indirect_occluded = specular_indirect * ao;

    // --- 3. FINAL COMPOSITION ---
    var final_hdr_color = ambient + Lo + specular_indirect_occluded + emission;

    let tonemapped = final_hdr_color / (final_hdr_color + vec3<f32>(1.0));
    let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, alpha);
}
