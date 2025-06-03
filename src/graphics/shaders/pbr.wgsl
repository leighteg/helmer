// PBR WGSL Shader for MEV Renderer

// Vertex input structure
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
}

// Camera uniforms
struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
}

// Light data
struct LightData {
    position: vec3<f32>,
    light_type: u32, // 0: directional, 1: point, 2: spot
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
}

// Material data
struct MaterialData {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
}

// Push constants
struct PbrConstants {
    model_matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    material_id: u32,
}

// Uniform bindings
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights: array<LightData>;
@group(0) @binding(2) var<storage, read> materials: array<MaterialData>;
@group(0) @binding(3) var albedo_texture: texture_2d<f32>;
@group(0) @binding(4) var normal_texture: texture_2d<f32>;
@group(0) @binding(5) var metallic_roughness_texture: texture_2d<f32>;
@group(0) @binding(6) var texture_sampler: sampler;

var<push_constant> constants: PbrConstants;

// Vertex shader
@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform position to world space
    let world_position = constants.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position.xyz;
    
    // Transform to clip space
    let view_position = camera.view_matrix * world_position;
    out.clip_position = camera.projection_matrix * view_position;
    
    // Transform normal to world space
    out.world_normal = normalize((constants.normal_matrix * vec4<f32>(vertex.normal, 0.0)).xyz);
    
    // Transform tangent to world space
    out.world_tangent = normalize((constants.model_matrix * vec4<f32>(vertex.tangent, 0.0)).xyz);
    
    // Calculate bitangent
    out.world_bitangent = cross(out.world_normal, out.world_tangent);
    
    // Pass through texture coordinates
    out.tex_coord = vertex.tex_coord;
    
    return out;
}

// PBR utility functions
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    let denom = n_dot_v * (1.0 - k) + k;
    return n_dot_v / denom;
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx1 * ggx2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Sample normal from normal map
fn sample_normal_map(tex_coord: vec2<f32>, world_normal: vec3<f32>, world_tangent: vec3<f32>, world_bitangent: vec3<f32>) -> vec3<f32> {
    let normal_sample = textureSample(normal_texture, texture_sampler, tex_coord).rgb;
    let normal_tangent = normalize(normal_sample * 2.0 - 1.0);
    
    let tbn = mat3x3<f32>(
        normalize(world_tangent),
        normalize(world_bitangent),
        normalize(world_normal)
    );
    
    return normalize(tbn * normal_tangent);
}

// Calculate lighting contribution from a single light
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
    
    // Calculate light direction and attenuation based on light type
    if (light.light_type == 0u) { // Directional light
        light_dir = normalize(-light.direction);
    } else if (light.light_type == 1u) { // Point light
        light_dir = normalize(light.position - world_pos);
        let distance = length(light.position - world_pos);
        attenuation = 1.0 / (distance * distance);
    } else { // Spot light (simplified)
        light_dir = normalize(light.position - world_pos);
        let distance = length(light.position - world_pos);
        attenuation = 1.0 / (distance * distance);
        // Add spot light cone calculation here if needed
    }
    
    let half_dir = normalize(view_dir + light_dir);
    
    // Calculate dot products
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);
    
    // Calculate F0 (surface reflection at zero incidence)
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    
    // Cook-Torrance BRDF
    let ndf = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(normal, view_dir, light_dir, roughness);
    let f = fresnel_schlick(v_dot_h, f0);
    
    let numerator = ndf * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001; // Add small value to prevent division by zero
    let specular = numerator / denominator;
    
    // For energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    let ks = f; // The energy of light that gets reflected
    var kd = vec3<f32>(1.0) - ks; // Remaining energy, goes to diffuse
    
    // Multiply kD by the inverse metalness such that only non-metals 
    // have diffuse lighting, or a linear blend if partly metal
    kd = kd * (1.0 - metallic);
    
    let radiance = light.color * light.intensity * attenuation;
    
    return (kd * albedo / 3.14159265 + specular) * radiance * n_dot_l;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample textures
    let albedo_sample = textureSample(albedo_texture, texture_sampler, in.tex_coord);
    let metallic_roughness_sample = textureSample(metallic_roughness_texture, texture_sampler, in.tex_coord);
    
    // Get material properties
    let material = materials[constants.material_id];
    let albedo = albedo_sample.rgb * material.albedo.rgb;
    let metallic = metallic_roughness_sample.b * material.metallic;
    let roughness = metallic_roughness_sample.g * material.roughness;
    let ao = material.ao;
    
    // Sample normal map and calculate world normal
    let normal = sample_normal_map(in.tex_coord, in.world_normal, in.world_tangent, in.world_bitangent);
    
    // Calculate view direction
    let view_dir = normalize(camera.view_position - in.world_position);
    
    // Initialize final color with ambient lighting
    var final_color = vec3<f32>(0.03) * albedo * ao;
    
    // Calculate lighting from all lights
    let light_count = arrayLength(&lights);
    for (var i = 0u; i < light_count; i = i + 1u) {
        final_color += calculate_light_contribution(
            lights[i],
            in.world_position,
            normal,
            view_dir,
            albedo,
            metallic,
            roughness
        );
    }
    
    // HDR tonemapping (simple Reinhard)
    final_color = final_color / (final_color + vec3<f32>(1.0));
    
    // Gamma correction
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(final_color, albedo_sample.a * material.albedo.a);
}