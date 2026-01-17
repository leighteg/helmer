//=============== CONSTANTS ===============//

const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const MAX_SHADOW_CASCADES: u32 = 4u;
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
    split_depth: vec4<f32>,
}

struct ShadowUniforms {
    cascade_count: u32,
    _pad0: vec3<u32>,
    cascades: array<CascadeData, MAX_SHADOW_CASCADES>,
}

struct PbrConstants {
    model_matrix: mat4x4<f32>,
    material_id: u32,
}

//=============== BINDINGS ===============//

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights_buffer: array<LightData>;
@group(0) @binding(2) var<storage, read> materials_buffer: array<MaterialData>;
@group(0) @binding(3) var albedo_texture_array: texture_2d_array<f32>;
@group(0) @binding(4) var normal_texture_array: texture_2d_array<f32>;
@group(0) @binding(5) var metallic_roughness_texture_array: texture_2d_array<f32>;
@group(0) @binding(6) var emission_texture_array: texture_2d_array<f32>;
@group(0) @binding(7) var pbr_sampler: sampler;
@group(0) @binding(8) var shadow_map: texture_2d_array<f32>;
@group(0) @binding(9) var shadow_sampler: sampler;
@group(0) @binding(10) var<uniform> shadow_uniforms: ShadowUniforms;

var<immediate> constants: PbrConstants;

//=============== MATRIX UTILITY FUNCTIONS ===============//

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

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

//=============== VERTEX SHADER ===============//

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_position_vec4 = constants.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;
    let view_pos = camera.view_matrix * world_position_vec4;
    out.clip_position = camera.projection_matrix * view_pos;
    out.view_space_depth = view_pos.z;

    let model_mat3 = mat3x3<f32>(
        constants.model_matrix[0].xyz,
        constants.model_matrix[1].xyz,
        constants.model_matrix[2].xyz,
    );

    let normal_matrix = transpose(mat3_inverse(model_mat3));
    
    out.world_normal = normalize(normal_matrix * vertex.normal);
    out.world_tangent = normalize((constants.model_matrix * vec4<f32>(vertex.tangent, 0.0)).xyz);
    out.world_bitangent = normalize(cross(out.world_normal, out.world_tangent));

    out.tex_coord = vertex.tex_coord;
    out.material_id = constants.material_id;

    return out;
}

//=============== PBR UTILITY FUNCTIONS ===============//

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
        return 1.0; // Not in shadow
    }

    var variance = moments.y - (moments.x * moments.x);
    variance = max(variance, VSM_MIN_VARIANCE);

    let d = depth - moments.x;
    let p_max = variance / (variance + d * d);

    // One-tailed inequality using p_max
    return smoothstep(SHADOW_BLEEDING_REDUCTION, 1.0, p_max);
}

fn calculate_shadow_factor(world_pos: vec3<f32>, view_z: f32) -> f32 {
    // 1. Select the correct cascade for this fragment
    let cascade_count = max(
        1u,
        min(shadow_uniforms.cascade_count, MAX_SHADOW_CASCADES)
    );
    var cascade_index = i32(cascade_count - 1u);
    for (var i = 0u; i < cascade_count; i = i + 1u) {
        if (view_z > shadow_uniforms.cascades[i].split_depth.x) {
            cascade_index = i32(i);
            break;
        }
    }

    // 2. Project to light space and perform lookup
    let cascade = shadow_uniforms.cascades[cascade_index];
    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);
    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);

    let fragment_light_depth = shadow_pos_clip.z;

    // Boundary Check
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 || fragment_light_depth > 1.0) {
        return 1.0;
    }

    // VSM Lookup and Test
    let moments = textureSample(shadow_map, shadow_sampler, shadow_uv, u32(cascade_index)).rg;
    return chebyshev_inequality(fragment_light_depth, moments);
}

//=============== FRAGMENT SHADER ===============//

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // --- 1. Material & PBR Property Setup ---
    let material = materials_buffer[in.material_id];

    let albedo_sample = textureSample(
        albedo_texture_array,
        pbr_sampler,
        in.tex_coord,
        material.albedo_texture_index
    );
    let albedo = albedo_sample.rgb * material.albedo.rgb;
    let alpha = albedo_sample.a * material.albedo.a;

    let mr_sample = textureSample(
        metallic_roughness_texture_array,
        pbr_sampler,
        in.tex_coord,
        material.metallic_roughness_texture_index
    );

    let ao = mr_sample.r * material.ao;
    let metallic = mr_sample.b * material.metallic;
    let roughness = max(mr_sample.g * material.roughness, MIN_ROUGHNESS);

    // --- 2. Emission Setup ---
    var emission = material.emission_color * material.emission_strength;
    if material.emission_texture_index >= 0i {
        let emission_sample = textureSample(
            emission_texture_array,
            pbr_sampler,
            in.tex_coord,
            material.emission_texture_index
        );
        emission *= emission_sample.rgb;
    }

    // --- 3. Normal Mapping ---
    var N: vec3<f32>;
    if material.normal_texture_index >= 0i {
        let tangent_space_normal = textureSample(
            normal_texture_array,
            pbr_sampler,
            in.tex_coord,
            material.normal_texture_index
        ).xyz * 2.0 - 1.0;

        let T = normalize(in.world_tangent);
        let B = normalize(in.world_bitangent);
        let N_geom = normalize(in.world_normal);
        let tbn = mat3x3<f32>(T, B, N_geom);
        N = normalize(tbn * tangent_space_normal);
    } else {
        N = normalize(in.world_normal);
    }

    // --- 4. View & Fresnel Setup ---
    let V = normalize(camera.view_position - in.world_position);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    // --- 4.5. Shadow Calculation ---
    let shadow_factor = calculate_shadow_factor(in.world_position, in.view_space_depth);
    
    // --- 5. Lighting Calculation ---
    var Lo = vec3<f32>(0.0);
    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;

        var shadow_multiplier = 1.0;
        if light.light_type == 0u { // Directional Light
            L = normalize(-light.direction);
            radiance = light.color * light.intensity;
            shadow_multiplier = shadow_factor;
        } else { // Point Light
            let to_light_vector = light.position - in.world_position;
            let dist_sq = dot(to_light_vector, to_light_vector);
            if dist_sq < 0.0001 { continue; }
            L = normalize(to_light_vector);
            let attenuation = 1.0 / dist_sq;
            radiance = light.color * light.intensity * attenuation;
        }

        let H = normalize(V + L);
        let NdotL = max(dot(N, L), 0.0);
        if NdotL > 0.0 {
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
    
    // --- 6. Final Color Composition with Emission ---
    let ambient = vec3<f32>(0.03) * albedo * ao;
    var final_color = ambient + Lo + emission;  // Add emission here

    // Tone mapping and gamma correction
    final_color = final_color / (final_color + vec3<f32>(1.0));
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(final_color, alpha);
}
