//=============== CONSTANTS ===============//

const EPSILON: f32 = 0.00001;

//=============== STRUCTS ===============//

struct GBufferInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
    @location(5) @interpolate(flat) material_id: u32,
}

struct GBufferOutput {
    @location(0) normal: vec4<f32>,
    @location(1) albedo: vec4<f32>,
    @location(2) mra: vec4<f32>, // Metallic, Roughness, AO
    @location(3) emission: vec4<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
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

struct PbrConstants {
    model_matrix: mat4x4<f32>,
    material_id: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<storage, read> materials_buffer: array<MaterialData>;
@group(1) @binding(1) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(2) var pbr_sampler: sampler;

@vertex
var<push_constant> constants: PbrConstants;

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

    // If the determinant is zero or very close to it, the matrix is not invertible.
    // Return the identity matrix as a safe fallback to prevent division by zero.
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

@vertex
fn vs_main(vertex: VertexInput) -> GBufferInput {
    var out: GBufferInput;

    let world_position_vec4 = constants.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;
    out.clip_position = camera.projection_matrix * camera.view_matrix * world_position_vec4;
    
    let model_mat3 = mat3x3<f32>(
        constants.model_matrix[0].xyz,
        constants.model_matrix[1].xyz,
        constants.model_matrix[2].xyz
    );
    let normal_matrix = transpose(mat3_inverse(model_mat3));

    let N = safe_normalize(normal_matrix * vertex.normal);
    let T = safe_normalize(normal_matrix * vertex.tangent.xyz);
    let B = cross(N, T) * vertex.tangent.w;

    out.world_normal = N;
    out.world_tangent = T;
    out.world_bitangent = B; 
    
    out.tex_coord = vertex.tex_coord;
    out.material_id = constants.material_id;
    
    return out;
}

@fragment
fn fs_main(in: GBufferInput) -> GBufferOutput {
    var out: GBufferOutput;
    let material = materials_buffer[in.material_id];

    // --- Albedo Calculation (with fallback) ---
    var albedo_sample = material.albedo;
    if (material.albedo_idx >= 0i) {
        albedo_sample = textureSample(textures[material.albedo_idx], pbr_sampler, in.tex_coord);
    }
    let albedo_color = albedo_sample.rgb * material.albedo.rgb;
    let alpha = albedo_sample.a * material.albedo.a;

    // --- MRA Calculation (with fallback) ---
    var metallic = material.metallic;
    var roughness = material.roughness;
    var ao = material.ao;
    if (material.metallic_roughness_idx >= 0i) {
        // Standard GLTF ORM (Occlusion, Roughness, Metallic) texture packing
        let mr_sample = textureSample(textures[material.metallic_roughness_idx], pbr_sampler, in.tex_coord);
        ao *= mr_sample.r;
        roughness *= mr_sample.g;
        metallic *= mr_sample.b;
    }

    // --- Emission Calculation ---
    var emission_color = material.emission_color * material.emission_strength;
    if (material.emission_idx >= 0i) {
        emission_color *= textureSample(textures[material.emission_idx], pbr_sampler, in.tex_coord).rgb;
    }

    // --- Normal Mapping ---
    var N: vec3<f32>;
    if (material.normal_idx >= 0i) {
        let tangent_space_normal = textureSample(textures[material.normal_idx], pbr_sampler, in.tex_coord).xyz * 2.0 - 1.0;
        let T = safe_normalize(in.world_tangent);
        let B = safe_normalize(in.world_bitangent);
        let N_geom = safe_normalize(in.world_normal);
        let tbn = mat3x3<f32>(T, B, N_geom);
        N = safe_normalize(tbn * tangent_space_normal);
    } else {
        N = safe_normalize(in.world_normal);
    }

    // --- Pack and Output ---
    // Ensure the normal written to the G-Buffer is also valid
    out.normal = vec4<f32>(N * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(albedo_color, alpha);
    out.mra = vec4<f32>(metallic, roughness, ao, 1.0);
    out.emission = vec4<f32>(emission_color, 1.0);

    return out;
}