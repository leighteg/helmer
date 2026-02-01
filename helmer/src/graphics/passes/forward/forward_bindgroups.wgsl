const EPSILON: f32 = 0.0001;

struct Constants {
    mip_bias: f32,
    shade_mode: u32,
    shade_smooth: u32,
    light_model: u32,
    skylight_contribution: u32,
    planet_radius: f32,
    atmosphere_radius: f32,
    sky_light_samples: u32,
    _pad0: u32,
    ssr_coarse_steps: u32,
    ssr_binary_search_steps: u32,
    ssr_linear_step_size: f32,
    ssr_thickness: f32,
    ssr_max_distance: f32,
    ssr_roughness_fade_start: f32,
    ssr_roughness_fade_end: f32,
    _pad1: u32,
    ssgi_num_rays: u32,
    ssgi_num_steps: u32,
    ssgi_ray_step_size: f32,
    ssgi_thickness: f32,
    ssgi_blend_factor: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    evsm_c: f32,
    pcf_radius: u32,
    pcf_min_scale: f32,
    pcf_max_scale: f32,
    pcf_max_distance: f32,
    ssgi_intensity: f32,
    _final_padding: vec4<f32>,
};

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

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct InstanceInput {
    @location(5) model_matrix_col_0: vec4<f32>,
    @location(6) model_matrix_col_1: vec4<f32>,
    @location(7) model_matrix_col_2: vec4<f32>,
    @location(8) model_matrix_col_3: vec4<f32>,
    @location(9) material_id: u32,
    @location(10) visibility: u32,
}

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
    @location(5) @interpolate(flat) material_id: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> materials: array<MaterialData>;
@group(1) @binding(1) var albedo_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var mra_tex: texture_2d<f32>;
@group(1) @binding(4) var emission_tex: texture_2d<f32>;
@group(1) @binding(5) var pbr_sampler: sampler;
@group(2) @binding(0) var<uniform> constants: Constants;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len < EPSILON { return vec3<f32>(0.0, 0.0, 1.0); }
    return v / len;
}

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if abs(det) < EPSILON {
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
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VSOut {
    var out: VSOut;
    if (instance.visibility == 0u) {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.world_position = vec3<f32>(0.0);
        out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
        out.world_tangent = vec3<f32>(1.0, 0.0, 0.0);
        out.world_bitangent = vec3<f32>(0.0, 1.0, 0.0);
        out.tex_coord = vec2<f32>(0.0);
        out.material_id = 0u;
        return out;
    }
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_col_0,
        instance.model_matrix_col_1,
        instance.model_matrix_col_2,
        instance.model_matrix_col_3
    );

    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.projection_matrix * camera.view_matrix * world_pos;

    let model_mat3 = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    );
    let normal_matrix = transpose(mat3_inverse(model_mat3));
    let N = safe_normalize(normal_matrix * vertex.normal);
    let T = safe_normalize(normal_matrix * vertex.tangent.xyz);
    let B = cross(N, T) * vertex.tangent.w;

    out.world_normal = N;
    out.world_tangent = T;
    out.world_bitangent = B;
    out.tex_coord = vertex.tex_coord;
    out.material_id = instance.material_id;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let material = materials[in.material_id];

    var albedo = material.albedo.rgb;
    var alpha = material.albedo.a;
    if material.albedo_idx >= 0i {
        let sample = textureSampleBias(albedo_tex, pbr_sampler, in.tex_coord, constants.mip_bias);
        albedo *= sample.rgb;
        alpha *= sample.a;
    }

    var emission = material.emission_color * material.emission_strength;
    if material.emission_idx >= 0i {
        emission *= textureSampleBias(emission_tex, pbr_sampler, in.tex_coord, constants.mip_bias).rgb;
    }

    return vec4<f32>(albedo + emission, alpha);
}
