//=============== CONSTANTS ===============//

const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;

//=============== STRUCTS ===============//

// (Structs are unchanged from your original code)
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
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
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
    _p1: f32,
    albedo_texture_index: i32,
    normal_texture_index: i32,
    metallic_roughness_texture_index: i32,
    _p2: i32,
}

struct PbrConstants {
    model_matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    material_id: u32,
    _p: vec3<u32>,
}

//=============== BINDINGS ===============//

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights_buffer: array<LightData>; // Renamed for clarity
@group(0) @binding(2) var<storage, read> materials_buffer: array<MaterialData>; // Renamed for clarity

// ✅ CHANGE #1: Use the correct texture_2d_array type.
@group(0) @binding(3) var albedo_texture_array: texture_2d_array<f32>;
@group(0) @binding(4) var normal_texture_array: texture_2d_array<f32>;
@group(0) @binding(5) var metallic_roughness_texture_array: texture_2d_array<f32>;

@group(0) @binding(6) var pbr_sampler: sampler;

var<push_constant> constants: PbrConstants;

//=============== VERTEX SHADER ===============//

// (Vertex shader is unchanged)
@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_position_vec4 = constants.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;

    out.clip_position = camera.projection_matrix * camera.view_matrix * world_position_vec4;

    out.world_normal = normalize((constants.normal_matrix * vec4<f32>(vertex.normal, 0.0)).xyz);
    out.world_tangent = normalize((constants.model_matrix * vec4<f32>(vertex.tangent, 0.0)).xyz);
    out.world_bitangent = normalize(cross(out.world_normal, out.world_tangent));

    out.tex_coord = vertex.tex_coord;

    return out;
}

//=============== PBR UTILITY FUNCTIONS ===============//

// (PBR utility functions are unchanged)
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

//=============== FRAGMENT SHADER ===============//

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // --- 1. Material & PBR Property Setup ---

    let material = materials_buffer[constants.material_id];

    // ✅ FIX: Use the 4-argument version of textureSample.
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

    // --- 2. Normal Mapping ---

    var N: vec3<f32>;
    if material.normal_texture_index > 0i {
        // ✅ FIX: Use the 4-argument version of textureSample here as well.
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

    // --- 3. View & Fresnel Setup ---
    let V = normalize(camera.view_position - in.world_position);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    
    // --- 4. Lighting Calculation ---
    var Lo = vec3<f32>(0.0);
    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;

        if light.light_type == 0u { // Directional Light
            if length(light.direction) < 0.0001 { continue; }
            L = normalize(-light.direction);
            radiance = light.color * light.intensity;
        } else if light.light_type == 2u { // ✅ NEW: Spot Light
            let to_light_vector = light.position - in.world_position;
            let light_dir = normalize(light.direction);
            let spot_cos_angle = dot(normalize(to_light_vector), -light_dir);

            // 60 degrees was your spot light angle from main.rs. cos(30) is half the angle.
            let spot_cos_cutoff = 0.866; // cos(30 degrees)

            if spot_cos_angle > spot_cos_cutoff {
                let dist_sq = dot(to_light_vector, to_light_vector);
                if dist_sq < 0.0001 { continue; }
                L = normalize(to_light_vector);
                let attenuation = 1.0 / dist_sq;
                let spot_effect = smoothstep(spot_cos_cutoff, spot_cos_cutoff + 0.05, spot_cos_angle);
                radiance = light.color * light.intensity * attenuation * spot_effect;
            } else {
                // Outside the cone, radiance is zero.
                radiance = vec3f(0.0);
            }
        } else { // Point Light (light_type == 1u)
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

            Lo += (kD * albedo / PI + specular) * radiance * NdotL;
        }
    }
    
    // --- 5. Final Color Composition ---
    let ambient = vec3<f32>(0.03) * albedo * ao;
    var final_color = ambient + Lo;

    final_color = final_color / (final_color + vec3<f32>(1.0));
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(final_color, alpha);
}