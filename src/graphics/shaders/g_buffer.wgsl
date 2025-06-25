// shaders/g_buffer.wgsl

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
    @location(3) tangent: vec3<f32>,
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
    emission_strength: f32,
    albedo_texture_index: i32,
    normal_texture_index: i32,
    metallic_roughness_texture_index: i32,
    emission_texture_index: i32,
    emission_color: vec3<f32>,
    _padding: f32,
}

struct PbrConstants {
    model_matrix: mat4x4<f32>,
    material_id: u32,
}

//=============== BINDINGS ===============//
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights_buffer: array<LightData>;

@group(1) @binding(0) var<storage, read> materials_buffer: array<MaterialData>;
@group(1) @binding(1) var albedo_texture_array: texture_2d_array<f32>;
@group(1) @binding(2) var normal_texture_array: texture_2d_array<f32>;
@group(1) @binding(3) var metallic_roughness_texture_array: texture_2d_array<f32>;
@group(1) @binding(4) var emission_texture_array: texture_2d_array<f32>;
@group(1) @binding(5) var pbr_sampler: sampler;

@vertex
var<push_constant> constants: PbrConstants;

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

@vertex
fn vs_main(vertex: VertexInput) -> GBufferInput {
    var out: GBufferInput;
    let world_position_vec4 = constants.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;
    let view_pos = camera.view_matrix * world_position_vec4;
    out.clip_position = camera.projection_matrix * view_pos;
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

@fragment
fn fs_main(in: GBufferInput) -> GBufferOutput {
    var out: GBufferOutput;
    let material = materials_buffer[in.material_id];

    let albedo_sample = textureSample(albedo_texture_array, pbr_sampler, in.tex_coord, material.albedo_texture_index);
    let albedo_color = albedo_sample.rgb * material.albedo.rgb;
    let alpha = albedo_sample.a * material.albedo.a;

    let mr_sample = textureSample(metallic_roughness_texture_array, pbr_sampler, in.tex_coord, material.metallic_roughness_texture_index);
    let ao = mr_sample.r * material.ao;
    let metallic = mr_sample.b * material.metallic;
    let roughness = mr_sample.g * material.roughness;

    // FIXED: Calculate the full emission color
    var emission_color = material.emission_color * material.emission_strength;
    if (material.emission_texture_index >= 0i) {
        emission_color *= textureSample(emission_texture_array, pbr_sampler, in.tex_coord, material.emission_texture_index).rgb;
    }

    var N: vec3<f32>;
    if (material.normal_texture_index >= 0i) {
        let tangent_space_normal = textureSample(normal_texture_array, pbr_sampler, in.tex_coord, material.normal_texture_index).xyz * 2.0 - 1.0;
        let T = normalize(in.world_tangent);
        let B = normalize(in.world_bitangent);
        let N_geom = normalize(in.world_normal);
        let tbn = mat3x3<f32>(T, B, N_geom);
        N = normalize(tbn * tangent_space_normal);
    } else {
        N = normalize(in.world_normal);
    }

    // --- Pack and Output ---
    out.normal = vec4<f32>(N * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(albedo_color, alpha);
    out.mra = vec4<f32>(metallic, roughness, ao, 1.0);
    out.emission = vec4<f32>(emission_color, 1.0);

    return out;
}