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
    _padding: f32,
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
    _padding: f32,
}

struct PbrConstants {
    model_matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    material_id: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights: array<LightData>;
@group(0) @binding(2) var<storage, read> materials: array<MaterialData>;
@group(0) @binding(3) var albedo_texture: texture_2d<f32>;
@group(0) @binding(4) var normal_texture: texture_2d<f32>;
@group(0) @binding(5) var metallic_roughness_texture: texture_2d<f32>;
@group(0) @binding(6) var texture_sampler: sampler;

var<push_constant> constants: PbrConstants;

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let world_position = constants.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position.xyz;
    
    let view_position = camera.view_matrix * world_position;
    out.clip_position = camera.projection_matrix * view_position;
    
    // Safe normal transformation
    let world_normal = (constants.normal_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;
    let normal_length = length(world_normal);
    out.world_normal = select(vec3<f32>(0.0, 1.0, 0.0), world_normal / normal_length, normal_length > 0.001);
    
    let world_tangent = (constants.model_matrix * vec4<f32>(vertex.tangent, 0.0)).xyz;
    let tangent_length = length(world_tangent);
    out.world_tangent = select(vec3<f32>(1.0, 0.0, 0.0), world_tangent / tangent_length, tangent_length > 0.001);
    
    out.world_bitangent = normalize(cross(out.world_normal, out.world_tangent));
    
    out.tex_coord = vertex.tex_coord;
    
    return out;
}

// Safe utility functions with bounds checking
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = clamp(roughness, 0.01, 1.0);
    let a2 = a * a;
    let n_dot_h_clamped = clamp(n_dot_h, 0.0, 1.0);
    let denom = n_dot_h_clamped * n_dot_h_clamped * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * max(denom * denom, 0.0001));
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = clamp(roughness + 1.0, 1.0, 2.0);
    let k = (r * r) / 8.0;
    let n_dot_v_clamped = clamp(n_dot_v, 0.0, 1.0);
    let denom = n_dot_v_clamped * (1.0 - k) + k;
    return n_dot_v_clamped / max(denom, 0.0001);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let n_dot_l = clamp(dot(n, l), 0.0, 1.0);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx1 * ggx2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let cos_theta_clamped = clamp(cos_theta, 0.0, 1.0);
    let f0_clamped = clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0));
    return f0_clamped + (1.0 - f0_clamped) * pow(1.0 - cos_theta_clamped, 5.0);
}

fn sample_normal_map(tex_coord: vec2<f32>, world_normal: vec3<f32>, world_tangent: vec3<f32>, world_bitangent: vec3<f32>) -> vec3<f32> {
    // Clamp texture coordinates to prevent out-of-bounds access
    let safe_tex_coord = clamp(tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
    let normal_sample = textureSample(normal_texture, texture_sampler, safe_tex_coord).rgb;
    let normal_tangent = normalize(normal_sample * 2.0 - 1.0);
    
    // Ensure all vectors are normalized and valid
    let t = normalize(world_tangent);
    let b = normalize(world_bitangent);
    let n = normalize(world_normal);
    
    let tbn = mat3x3<f32>(t, b, n);
    return normalize(tbn * normal_tangent);
}

fn calculate_light_contribution(
    light: LightData,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32
) -> vec3<f32> {
    var light_dir: vec3<f32>;
    var attenuation: f32 = 1.0;
    
    // Safe light direction calculation
    if (light.light_type == 0u) {
        // Directional light
        let dir_length = length(light.direction);
        light_dir = select(vec3<f32>(0.0, -1.0, 0.0), -light.direction / dir_length, dir_length > 0.001);
    } else {
        // Point/spot light
        let light_vec = light.position - world_pos;
        let distance = length(light_vec);
        light_dir = select(vec3<f32>(0.0, 1.0, 0.0), light_vec / distance, distance > 0.001);
        attenuation = 1.0 / max(distance * distance + 1.0, 1.0);
    }
    
    let half_dir = normalize(view_dir + light_dir);
    
    let n_dot_l = clamp(dot(normal, light_dir), 0.0, 1.0);
    let n_dot_v = clamp(dot(normal, view_dir), 0.0, 1.0);
    let n_dot_h = clamp(dot(normal, half_dir), 0.0, 1.0);
    let v_dot_h = clamp(dot(view_dir, half_dir), 0.0, 1.0);
    
    // Early exit if no light contribution
    if (n_dot_l < 0.001) {
        return vec3<f32>(0.0);
    }
    
    let f0 = mix(vec3<f32>(0.04), clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(metallic, 0.0, 1.0));
    
    let ndf = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(normal, view_dir, light_dir, roughness);
    let f = fresnel_schlick(v_dot_h, f0);
    
    let numerator = ndf * g * f;
    let denominator = max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let specular = numerator / denominator;
    
    let ks = f;
    var kd = clamp(vec3<f32>(1.0) - ks, vec3<f32>(0.0), vec3<f32>(1.0));
    kd = kd * (1.0 - clamp(metallic, 0.0, 1.0));
    
    let safe_color = clamp(light.color, vec3<f32>(0.0), vec3<f32>(10.0));
    let safe_intensity = clamp(light.intensity, 0.0, 100.0);
    let radiance = safe_color * safe_intensity * clamp(attenuation, 0.0, 1.0);
    
    return (kd * albedo / 3.14159265 + specular) * radiance * n_dot_l;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // CRITICAL: Early bounds checking to prevent crashes
    let material_count = arrayLength(&materials);
    if (material_count == 0u) {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta error color
    }
    
    // Safe texture coordinate clamping
    let safe_tex_coord = clamp(in.tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
    
    // Safe texture sampling
    let albedo_sample = textureSample(albedo_texture, texture_sampler, safe_tex_coord);
    let metallic_roughness_sample = textureSample(metallic_roughness_texture, texture_sampler, safe_tex_coord);
    
    // Safe material access with bounds checking
    let safe_material_id = min(constants.material_id, material_count - 1u);
    let material = materials[safe_material_id];
    
    // Safe material properties
    let albedo = clamp(albedo_sample.rgb * material.albedo.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let metallic = clamp(metallic_roughness_sample.b * material.metallic, 0.0, 1.0);
    let roughness = clamp(metallic_roughness_sample.g * material.roughness, 0.04, 1.0);
    let ao = clamp(material.ao, 0.0, 1.0);
    
    // Safe normal calculation
    let normal = sample_normal_map(safe_tex_coord, in.world_normal, in.world_tangent, in.world_bitangent);
    
    // Safe view direction
    let view_vec = camera.view_position - in.world_position;
    let view_distance = length(view_vec);
    let view_dir = select(vec3<f32>(0.0, 0.0, 1.0), view_vec / view_distance, view_distance > 0.001);
    
    // Safe ambient calculation
    var final_color = clamp(vec3<f32>(0.03) * albedo * ao, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Safe light loop with hard limit
    let light_count = min(arrayLength(&lights), 32u); // Hard limit to prevent infinite loops
    for (var i = 0u; i < light_count; i = i + 1u) {
        let light_contribution = calculate_light_contribution(
            lights[i],
            in.world_position,
            normal,
            view_dir,
            albedo,
            metallic,
            roughness
        );
        final_color += clamp(light_contribution, vec3<f32>(0.0), vec3<f32>(10.0));
    }
    
    // Safe tone mapping
    final_color = clamp(final_color, vec3<f32>(0.0), vec3<f32>(100.0));
    final_color = final_color / (final_color + vec3<f32>(1.0));
    final_color = pow(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    
    let final_alpha = clamp(albedo_sample.a * material.albedo.a, 0.0, 1.0);
    return vec4<f32>(final_color, final_alpha);
}