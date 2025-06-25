//=============== CONSTANTS ===============//

const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const NUM_CASCADES: u32 = 4u;
const VSM_MIN_VARIANCE: f32 = 0.00002;
const SHADOW_BLEEDING_REDUCTION: f32 = 0.4;

//=============== STRUCTS ===============//

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
    @location(5) @interpolate(flat) material_id: u32,
    @location(6) view_space_depth: f32,
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
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

struct MaterialData {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,  // Added emission strength
    albedo_texture_index: i32,
    normal_texture_index: i32,
    metallic_roughness_texture_index: i32,
    emission_texture_index: i32,  // Added emission texture index
    emission_color: vec3<f32>,    // Added emission color
    _padding: f32,
}

struct CascadeData {
    light_view_proj: mat4x4<f32>,
    split_depth: f32,
    // padding would be here if needed for alignment
}

struct PbrConstants {
    model_matrix: mat4x4<f32>,
    material_id: u32,
}

// --- BINDINGS ---
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


// --- PBR & SHADOW UTILITY FUNCTIONS (Full code included) ---
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
    let ggx2 = geometry_schlick_ggx(NdotV, roughness);
    let ggx1 = geometry_schlick_ggx(NdotL, roughness);
    return ggx1 * ggx2;
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn chebyshev_inequality(depth: f32, moments: vec2<f32>) -> f32 {
    let bias = 0.005;
    if (depth - bias <= moments.x) {
        return 1.0;
    }
    var variance = moments.y - (moments.x * moments.x);
    variance = max(variance, VSM_MIN_VARIANCE);
    let d = depth - moments.x;
    let p_max = variance / (variance + d * d);
    return smoothstep(SHADOW_BLEEDING_REDUCTION, 1.0, p_max);
}

fn calculate_shadow_factor(world_pos: vec3<f32>, view_z: f32) -> f32 {
    var cascade_index = -1;
    for (var i = 0i; i < i32(NUM_CASCADES); i = i + 1i) {
        if (view_z > shadow_uniforms[i].split_depth) {
            cascade_index = i;
            break;
        }
    }

    if (cascade_index < 0) {
        return 1.0;
    }

    let cascade = shadow_uniforms[cascade_index];
    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);
    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);

    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 || shadow_coord.z > 1.0) {
        return 1.0;
    }

    let moments = textureSample(shadow_map, shadow_sampler, shadow_uv, u32(cascade_index)).rg;
    return chebyshev_inequality(shadow_coord.z, moments);
}

// --- SHADERS ---
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let screen_uv = frag_coord.xy / vec2<f32>(textureDimensions(gbuf_normal));

    // --- Reconstruct World Position from Depth ---
    let depth = textureSample(depth_texture, gbuf_sampler, screen_uv);

    // Discard skybox/background fragments
    if (depth >= 1.0) {
        discard;
    }
    
    // Create Normalized Device Coordinate (NDC)
    // Y is flipped because WGPU's NDC Y-coordinate is upward
    let ndc = vec4<f32>(screen_uv.x * 2.0 - 1.0, (1.0 - screen_uv.y) * 2.0 - 1.0, depth, 1.0);
    
    // Un-project from NDC to world space
    let world_pos_h = camera.inverse_view_projection_matrix * ndc;
    let world_position = world_pos_h.xyz / world_pos_h.w;

    let packed_normal = textureSample(gbuf_normal, gbuf_sampler, screen_uv).xyz;
    let N = normalize(packed_normal * 2.0 - 1.0);

    let albedo_sample = textureSample(gbuf_albedo, gbuf_sampler, screen_uv);
    let albedo = albedo_sample.rgb;
    let alpha = albedo_sample.a;

    let mra_sample = textureSample(gbuf_mra, gbuf_sampler, screen_uv);
    let metallic = mra_sample.r;
    let roughness = max(mra_sample.g, MIN_ROUGHNESS);
    let ao = mra_sample.b;

    // FIXED: Sample the full RGB emission color from its dedicated texture
    let emission = textureSample(gbuf_emission, gbuf_sampler, screen_uv).rgb;

    let V = normalize(camera.view_position - world_position);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    let view_pos = camera.view_matrix * vec4<f32>(world_position, 1.0);
    let shadow_factor = calculate_shadow_factor(world_position, view_pos.z);

    var Lo = vec3<f32>(0.0);
    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;
        var shadow_multiplier = 1.0;

        if (light.light_type == 0u) { // Directional
            L = normalize(-light.direction);
            radiance = light.color * light.intensity;
            shadow_multiplier = shadow_factor;
        } else { // Point
            let to_light_vector = light.position - world_position;
            let dist_sq = dot(to_light_vector, to_light_vector);
            if (dist_sq < 0.0001) { continue; }
            L = normalize(to_light_vector);
            let attenuation = 1.0 / dist_sq;
            radiance = light.color * light.intensity * attenuation;
        }

        let H = normalize(V + L);
        let NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            let NdotV = max(dot(N, V), 0.0);
            let NdotH = max(dot(N, H), 0.0);
            let HdotV = max(dot(H, V), 0.0);
            let NDF = distribution_ggx(NdotH, roughness);
            let G = geometry_smith(N, V, L, roughness);
            let F = fresnel_schlick(HdotV, F0);
            let numerator = NDF * G * F;
            let denominator = 4.0 * NdotV * NdotL + 0.001;
            let specular = numerator / denominator;
            let kS = F;
            var kD = vec3<f32>(1.0) - kS;
            kD *= (1.0 - metallic);
            Lo += (kD * albedo / PI + specular) * radiance * NdotL * shadow_multiplier;
        }
    }

    let ambient = vec3<f32>(0.03) * albedo * ao;
    // FIXED: Add the full, correct emission color
    var final_color = ambient + Lo + emission;

    final_color = final_color / (final_color + vec3<f32>(1.0));
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(final_color, alpha);
}