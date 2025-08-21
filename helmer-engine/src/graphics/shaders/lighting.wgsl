//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const NUM_CASCADES: u32 = 4u;
const VSM_MIN_VARIANCE: f32 = 0.00002;
const SHADOW_BLEEDING_REDUCTION: f32 = 0.4;
const EPSILON: f32 = 0.00001;

//=============== STRUCTS ===============//
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

//=============== BINDINGS ===============//
@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuf_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuf_mra: texture_2d<f32>;
@group(0) @binding(4) var gbuf_emission: texture_2d<f32>;
@group(0) @binding(5) var gbuf_sampler: sampler;

@group(1) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(1) var<storage, read> lights_buffer: array<LightData>;
@group(1) @binding(2) var shadow_map: texture_2d_array<f32>;
@group(1) @binding(3) var shadow_sampler: sampler;
@group(1) @binding(4) var<uniform> shadow_uniforms: array<CascadeData, NUM_CASCADES>;

//=============== UTILITY & PBR FUNCTIONS ===============//
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len < EPSILON) { return vec3<f32>(0.0); }
    return v / len;
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
fn chebyshev_inequality(depth: f32, moments: vec2<f32>) -> f32 {
    let bias: f32 = 0.005;
    let current_depth = depth - bias;
    if (current_depth <= moments.x) {
        return 1.0;
    }
    var variance = moments.y - (moments.x * moments.x);
    variance = max(variance, VSM_MIN_VARIANCE);
    let d = current_depth - moments.x;
    let p_max = variance / (variance + d * d);
    return smoothstep(SHADOW_BLEEDING_REDUCTION, 1.0, p_max);
}

fn calculate_shadow_factor(world_pos: vec3<f32>, view_z: f32) -> f32 {
    var cascade_index = i32(NUM_CASCADES - 1);
    for (var i = 0i; i < i32(NUM_CASCADES); i = i + 1i) {
        if (view_z > shadow_uniforms[i].split_depth.x) {
            cascade_index = i;
            break;
        }
    }
    let cascade = shadow_uniforms[cascade_index];
    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);
    if (shadow_pos_clip.w < EPSILON) { return 1.0; }
    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);
    if (any(shadow_uv < vec2(0.0)) || any(shadow_uv > vec2(1.0)) || shadow_coord.z < 0.0 || shadow_coord.z > 1.0) {
        return 1.0;
    }
    let moments = textureSample(shadow_map, shadow_sampler, shadow_uv, u32(cascade_index)).rg;
    return chebyshev_inequality(shadow_coord.z, moments);
}

//=============== SHADERS ===============//
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let screen_uv = frag_coord.xy / vec2<f32>(textureDimensions(gbuf_normal));
    let depth = textureSample(depth_texture, gbuf_sampler, screen_uv);
    if (depth >= 1.0) { discard; }

    let ndc = vec4<f32>(screen_uv.x * 2.0 - 1.0, (1.0 - screen_uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_pos_h = camera.inverse_view_projection_matrix * ndc;
    let world_position = world_pos_h.xyz / world_pos_h.w;

    // **FIX**: Correctly unpack the normal from [0, 1] to [-1, 1] range.
    let packed_normal = textureSample(gbuf_normal, gbuf_sampler, screen_uv).xyz;
    let N = safe_normalize(packed_normal * 2.0 - 1.0);

    let albedo = textureSample(gbuf_albedo, gbuf_sampler, screen_uv).rgb;
    let mra_sample = textureSample(gbuf_mra, gbuf_sampler, screen_uv);
    let metallic = mra_sample.r;
    let roughness = max(mra_sample.g, MIN_ROUGHNESS);

    let V = safe_normalize(camera.view_position - world_position);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    var direct_lighting = vec3<f32>(0.0);

    let view_pos = camera.view_matrix * vec4<f32>(world_position, 1.0);
    let shadow_factor = calculate_shadow_factor(world_position, view_pos.z);
    
    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;
        var shadow_multiplier = 1.0;
        
        if (light.light_type == 0u) { // Directional
            L = safe_normalize(-light.direction);
            radiance = light.color * light.intensity;
            shadow_multiplier = shadow_factor;
        } else { // Point
            let to_light_vector = light.position - world_position;
            let dist_sq = dot(to_light_vector, to_light_vector);
            if (dist_sq < EPSILON) { continue; }
            L = to_light_vector / sqrt(dist_sq);
            let attenuation = 1.0 / (dist_sq + 1.0);
            radiance = light.color * light.intensity * attenuation;
        }
        
        let NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            let H = safe_normalize(V + L);
            let NdotH = max(dot(N, H), 0.0);
            
            let NDF = distribution_ggx(NdotH, roughness);
            let G = geometry_smith(N, V, L, roughness);
            let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
            
            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + EPSILON;
            let specular = numerator / denominator;
            
            let kS = F;
            let kD = vec3<f32>(1.0) - kS;
            let diffuse_color_contribution = kD * (1.0 - metallic);
            
            direct_lighting += (diffuse_color_contribution * albedo / PI + specular) * radiance * NdotL * shadow_multiplier;
        }
    }

    return vec4<f32>(direct_lighting, 1.0);
}