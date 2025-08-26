//! src/graphics/shaders/forward.wgsl

//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const NUM_CASCADES: u32 = 4u;
const VSM_MIN_VARIANCE: f32 = 0.00002;
const SHADOW_BLEEDING_REDUCTION: f32 = 0.4;
const EPSILON: f32 = 0.00001;

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
    @location(5) view_z: f32,
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
}

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
}

struct ModelPushConstant {
    model_matrix: mat4x4<f32>,
}

//=============== BINDINGS ===============//
// Scene data
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights_buffer: array<LightData>;
@group(0) @binding(2) var shadow_map: texture_2d_array<f32>;
@group(0) @binding(3) var shadow_sampler: sampler;
@group(0) @binding(4) var<uniform> shadow_uniforms: array<CascadeData, NUM_CASCADES>;

// Material data
@group(1) @binding(0) var<uniform> material: MaterialData;
@group(1) @binding(1) var albedo_textures: texture_2d_array<f32>;
@group(1) @binding(2) var normal_textures: texture_2d_array<f32>;
@group(1) @binding(3) var mr_textures: texture_2d_array<f32>;
@group(1) @binding(4) var texture_sampler: sampler;

@vertex var<push_constant> model: ModelPushConstant;

//=============== UTILITY FUNCTIONS ===============//
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len < EPSILON) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return v / len;
}

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (abs(det) < EPSILON) {
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

//=============== PBR FUNCTIONS ===============//
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

fn chebyshev_inequality(depth: f32, moments: vec2<f32>) -> f32 {
    let bias: f32 = 0.005;
    let current_depth = depth - bias;
    if current_depth <= moments.x {
        return 1.0;
    }
    var variance = moments.y - (moments.x * moments.x);
    variance = max(variance, VSM_MIN_VARIANCE);
    let d = current_depth - moments.x;
    let p_max = variance / (variance + d * d);
    return smoothstep(SHADOW_BLEEDING_REDUCTION, 1.0, p_max);
}

//=============== SHADOW FUNCTIONS ===============//
fn calculate_shadow_factor(world_pos: vec3<f32>, view_z: f32) -> f32 {
    var cascade_index = i32(NUM_CASCADES - 1);
    for (var i = 0i; i < i32(NUM_CASCADES); i = i + 1i) {
        if view_z > shadow_uniforms[i].split_depth.x {
            cascade_index = i;
            break;
        }
    }

    let cascade = shadow_uniforms[cascade_index];
    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);

    if shadow_pos_clip.w < EPSILON {
        return 1.0;
    }

    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);

    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 || 
       shadow_coord.z < 0.0 || shadow_coord.z > 1.0 {
        return 1.0;
    }

    let moments = textureSample(shadow_map, shadow_sampler, shadow_uv, u32(cascade_index)).rg;
    return chebyshev_inequality(shadow_coord.z, moments);
}

//=============== VERTEX SHADER ===============//
@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_position_vec4 = model.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;
    out.clip_position = camera.projection_matrix * camera.view_matrix * world_position_vec4;
    
    let model_mat3 = mat3x3<f32>(
        model.model_matrix[0].xyz,
        model.model_matrix[1].xyz,
        model.model_matrix[2].xyz
    );
    let normal_matrix = transpose(mat3_inverse(model_mat3));

    let N = safe_normalize(normal_matrix * vertex.normal);
    let T = safe_normalize(normal_matrix * vertex.tangent.xyz);
    let B = cross(N, T) * vertex.tangent.w;

    out.world_normal = N;
    out.world_tangent = T;
    out.world_bitangent = B;
    out.tex_coord = vertex.tex_coord;
    
    // Calculate view space z for cascade selection
    let view_pos = camera.view_matrix * world_position_vec4;
    out.view_z = view_pos.z;
    
    return out;
}

//=============== FRAGMENT SHADER ===============//
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // --- Albedo Calculation (FIXED) ---
    var albedo_color: vec3<f32>;
    var alpha: f32;
    if (material.albedo_idx >= 0i) {
        let albedo_sample = textureSample(albedo_textures, texture_sampler, in.tex_coord, u32(material.albedo_idx));
        albedo_color = albedo_sample.rgb * material.albedo.rgb;
        alpha = albedo_sample.a * material.albedo.a;
    } else {
        albedo_color = material.albedo.rgb;
        alpha = material.albedo.a;
    }

    // --- Metallic/Roughness/AO Calculation (FIXED) ---
    var ao: f32;
    var metallic: f32;
    var roughness: f32;
    if (material.metallic_roughness_idx >= 0i) {
        let mr_sample = textureSample(mr_textures, texture_sampler, in.tex_coord, u32(material.metallic_roughness_idx));
        ao = mr_sample.r * material.ao;
        metallic = mr_sample.b * material.metallic;
        roughness = max(mr_sample.g * material.roughness, MIN_ROUGHNESS);
    } else {
        ao = material.ao;
        metallic = material.metallic;
        roughness = max(material.roughness, MIN_ROUGHNESS);
    }

    // --- Normal Calculation (Correct as-is) ---
    var N: vec3<f32>;
    if (material.normal_idx >= 0i) {
        let tangent_space_normal = textureSample(normal_textures, texture_sampler, in.tex_coord, u32(material.normal_idx)).xyz * 2.0 - 1.0;
        let T = safe_normalize(in.world_tangent);
        let B = safe_normalize(in.world_bitangent);
        let N_geom = safe_normalize(in.world_normal);
        let tbn = mat3x3<f32>(T, B, N_geom);
        N = safe_normalize(tbn * tangent_space_normal);
    } else {
        N = safe_normalize(in.world_normal);
    }

    // Basic material properties
    let V = safe_normalize(camera.view_position - in.world_position);
    let F0 = mix(vec3<f32>(0.04), albedo_color, metallic);

    // Calculate lighting
    var Lo = vec3<f32>(0.0);
    let shadow_factor = calculate_shadow_factor(in.world_position, in.view_z);

    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;
        var shadow_multiplier = 1.0;

        if light.light_type == 0u { // Directional
            L = safe_normalize(-light.direction);
            radiance = light.color * light.intensity;
            shadow_multiplier = shadow_factor;
        } else { // Point
            let to_light_vector = light.position - in.world_position;
            let dist_sq = dot(to_light_vector, to_light_vector);
            if dist_sq < EPSILON { continue; }
            L = to_light_vector / sqrt(dist_sq);
            let attenuation = 1.0 / (dist_sq + 1.0);
            radiance = light.color * light.intensity * attenuation;
        }

        let NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            let H = safe_normalize(V + L);
            let NdotH = max(dot(N, H), 0.0);
            let HdotV = max(dot(H, V), 0.0);

            // Cook-Torrance BRDF
            let NDF = distribution_ggx(NdotH, roughness);
            let G = geometry_smith(N, V, L, roughness);
            let F = fresnel_schlick(HdotV, F0);

            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + EPSILON;
            let specular = numerator / denominator;

            // Diffuse
            let kS = F;
            let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

            Lo += (kD * albedo_color / PI + specular) * radiance * NdotL * shadow_multiplier;
        }
    }

    // Simple ambient
    //let ambient = vec3<f32>(0.03) * albedo_color * ao;
    
    // Add emission
    let emission = material.emission_color * material.emission_strength;

    // Combine
    var color = Lo + emission;

    // Tone mapping and gamma correction
    //color = color / (color + vec3<f32>(1.0));
    //color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, alpha);
}