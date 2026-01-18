enable wgpu_mesh_shader;

const EPSILON: f32 = 0.0001;
const MAX_VERTS: u32 = 64u;
const MAX_PRIMS: u32 = 124u;

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
    _final_padding: vec2<f32>,
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

struct GBufferInstance {
    model_matrix: mat4x4<f32>,
    material_id: u32,
    visibility: u32,
    _pad0: vec2<u32>,
    bounds_center: vec4<f32>,
    bounds_extents: vec4<f32>,
}

struct MeshletDesc {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
    bounds_center: vec3<f32>,
    bounds_radius: f32,
}

struct MeshDrawParams {
    instance_base: u32,
    instance_count: u32,
    meshlet_base: u32,
    meshlet_count: u32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    depth_bias: f32,
    rect_pad: f32,
    _pad3: vec2<f32>,
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

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
}

struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
    @builtin(vertices) vertices: array<VSOut, MAX_VERTS>,
    @builtin(primitives) primitives: array<PrimitiveOutput, MAX_PRIMS>,
}

var<workgroup> mesh_output: MeshOutput;
var<workgroup> mesh_visible: u32;
var<workgroup> mesh_model: mat4x4<f32>;
var<workgroup> mesh_material_id: u32;
var<workgroup> mesh_normal_matrix: mat3x3<f32>;
var<workgroup> mesh_vert_count: u32;
var<workgroup> mesh_prim_count: u32;
var<workgroup> meshlet_vertex_offset: u32;
var<workgroup> meshlet_index_offset: u32;
var<workgroup> meshlet_center: vec3<f32>;
var<workgroup> meshlet_radius: f32;

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> materials: array<MaterialData>;
@group(1) @binding(1) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(2) var pbr_sampler: sampler;
@group(2) @binding(0) var<uniform> constants: Constants;

@group(3) @binding(0) var<storage, read> instances: array<GBufferInstance>;
@group(3) @binding(1) var<storage, read> meshlet_descs: array<MeshletDesc>;
@group(3) @binding(2) var<storage, read> meshlet_vertices: array<u32>;
@group(3) @binding(3) var<storage, read> meshlet_indices: array<u32>;
@group(3) @binding(4) var<storage, read> vertex_data: array<u32>;
@group(3) @binding(5) var<uniform> draw_params: MeshDrawParams;
@group(3) @binding(6) var hiz_tex: texture_2d<f32>;

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

fn load_f32(index: u32) -> f32 {
    return bitcast<f32>(vertex_data[index]);
}

fn load_vec2(base: u32) -> vec2<f32> {
    return vec2<f32>(load_f32(base), load_f32(base + 1u));
}

fn load_vec3(base: u32) -> vec3<f32> {
    return vec3<f32>(load_f32(base), load_f32(base + 1u), load_f32(base + 2u));
}

fn load_vec4(base: u32) -> vec4<f32> {
    return vec4<f32>(
        load_f32(base),
        load_f32(base + 1u),
        load_f32(base + 2u),
        load_f32(base + 3u)
    );
}

fn max_scale(model: mat4x4<f32>) -> f32 {
    let x = length(model[0].xyz);
    let y = length(model[1].xyz);
    let z = length(model[2].xyz);
    return max(x, max(y, z));
}

fn sphere_in_frustum(view_proj: mat4x4<f32>, center: vec3<f32>, radius: f32) -> bool {
    let row0 = vec4<f32>(view_proj[0][0], view_proj[1][0], view_proj[2][0], view_proj[3][0]);
    let row1 = vec4<f32>(view_proj[0][1], view_proj[1][1], view_proj[2][1], view_proj[3][1]);
    let row2 = vec4<f32>(view_proj[0][2], view_proj[1][2], view_proj[2][2], view_proj[3][2]);
    let row3 = vec4<f32>(view_proj[0][3], view_proj[1][3], view_proj[2][3], view_proj[3][3]);

    let planes = array<vec4<f32>, 6>(
        row3 + row0,
        row3 - row0,
        row3 + row1,
        row3 - row1,
        row3 + row2,
        row3 - row2
    );

    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = planes[i];
        let dist = dot(plane.xyz, center) + plane.w;
        let len = length(plane.xyz);
        if (dist < -radius * len) {
            return false;
        }
    }

    return true;
}

@mesh(mesh_output)
@workgroup_size(64)
fn ms_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    if (local_id.x == 0u) {
        mesh_visible = 1u;
        mesh_vert_count = 0u;
        mesh_prim_count = 0u;
        if (workgroup_id.y >= draw_params.instance_count) {
            mesh_visible = 0u;
        } else {
            let instance_index = draw_params.instance_base + workgroup_id.y;
            let inst = instances[instance_index];
            if (inst.visibility == 0u) {
                mesh_visible = 0u;
            }

            if (workgroup_id.x >= draw_params.meshlet_count) {
                mesh_visible = 0u;
            } else {
                let meshlet_index = draw_params.meshlet_base + workgroup_id.x;
                let meshlet = meshlet_descs[meshlet_index];
                let vert_count = meshlet.vertex_count;
                let prim_count = meshlet.index_count / 3u;
                if (vert_count > MAX_VERTS || prim_count > MAX_PRIMS) {
                    mesh_visible = 0u;
                } else {
                    mesh_vert_count = vert_count;
                    mesh_prim_count = prim_count;
                    meshlet_vertex_offset = meshlet.vertex_offset;
                    meshlet_index_offset = meshlet.index_offset;
                    meshlet_center = meshlet.bounds_center;
                    meshlet_radius = meshlet.bounds_radius;

                    mesh_model = inst.model_matrix;
                    mesh_material_id = inst.material_id;
                    let model_mat3 = mat3x3<f32>(
                        mesh_model[0].xyz,
                        mesh_model[1].xyz,
                        mesh_model[2].xyz
                    );
                    mesh_normal_matrix = transpose(mat3_inverse(model_mat3));

                    let flags = draw_params.flags;
                    let frustum_enabled = (flags & 1u) != 0u;
                    let occlusion_enabled = (flags & 2u) != 0u;

                    if (mesh_visible != 0u) {
                        let world_center =
                            (mesh_model * vec4<f32>(meshlet_center, 1.0)).xyz;
                        let world_radius = meshlet_radius * max_scale(mesh_model);
                        let view_proj = camera.projection_matrix * camera.view_matrix;

                        if (frustum_enabled) {
                            if (!sphere_in_frustum(view_proj, world_center, world_radius)) {
                                mesh_visible = 0u;
                            }
                        }

                        if (mesh_visible != 0u && occlusion_enabled) {
                            let prev_view_proj = camera.prev_view_proj;
                            let extents = vec3<f32>(world_radius);
                            let signs = array<vec3<f32>, 8>(
                                vec3<f32>(-1.0, -1.0, -1.0),
                                vec3<f32>(1.0, -1.0, -1.0),
                                vec3<f32>(-1.0, 1.0, -1.0),
                                vec3<f32>(1.0, 1.0, -1.0),
                                vec3<f32>(-1.0, -1.0, 1.0),
                                vec3<f32>(1.0, -1.0, 1.0),
                                vec3<f32>(-1.0, 1.0, 1.0),
                                vec3<f32>(1.0, 1.0, 1.0)
                            );

                            var min_ndc = vec2<f32>(1.0, 1.0);
                            var max_ndc = vec2<f32>(-1.0, -1.0);
                            var max_depth = 0.0;
                            var valid_prev = false;
                            var valid_any = false;

                            for (var i = 0u; i < 8u; i = i + 1u) {
                                let world = world_center + extents * signs[i];

                                let clip_prev = prev_view_proj * vec4<f32>(world, 1.0);
                                if (clip_prev.w > 0.0) {
                                    let ndc_prev = clip_prev.xyz / clip_prev.w;
                                    min_ndc = min(min_ndc, ndc_prev.xy);
                                    max_ndc = max(max_ndc, ndc_prev.xy);
                                    max_depth = max(max_depth, ndc_prev.z);
                                    valid_prev = true;
                                    valid_any = true;
                                }

                                let clip_cur = view_proj * vec4<f32>(world, 1.0);
                                if (clip_cur.w > 0.0) {
                                    valid_any = true;
                                }
                            }

                            if (valid_prev && valid_any) {
                                let uv_min = clamp(min_ndc * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
                                let uv_max = clamp(max_ndc * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
                                if (uv_min.x < uv_max.x && uv_min.y < uv_max.y) {
                                    let base_dims = vec2<f32>(textureDimensions(hiz_tex, 0));
                                    let rect = (uv_max - uv_min) * base_dims;
                                    let max_dim = max(rect.x, rect.y);

                                    let levels = max(textureNumLevels(hiz_tex), 1u);
                                    var mip = 0u;
                                    if (max_dim > 1.0 && levels > 1u) {
                                        let in = u32(floor(log2(max_dim)));
                                        mip = min(in + 1u, levels - 1u);
                                    }

                                    let mip_level = i32(mip);
                                    let mip_dims = vec2<i32>(textureDimensions(hiz_tex, mip_level));
                                    let mip_dims_f = vec2<f32>(mip_dims);
                                    let pad = vec2<f32>(draw_params.rect_pad);
                                    let min_f = clamp(uv_min * mip_dims_f - pad, vec2<f32>(0.0), mip_dims_f - vec2<f32>(1.0));
                                    let max_f = clamp(uv_max * mip_dims_f + pad, vec2<f32>(0.0), mip_dims_f - vec2<f32>(1.0));
                                    let min_tex = vec2<i32>(min_f);
                                    let max_tex = vec2<i32>(max_f);

                                    let d0 = textureLoad(hiz_tex, min_tex, mip_level).x;
                                    let d1 = textureLoad(hiz_tex, vec2<i32>(max_tex.x, min_tex.y), mip_level).x;
                                    let d2 = textureLoad(hiz_tex, vec2<i32>(min_tex.x, max_tex.y), mip_level).x;
                                    let d3 = textureLoad(hiz_tex, max_tex, mip_level).x;
                                    let hiz_depth = min(min(d0, d1), min(d2, d3));

                                    let depth = clamp(max_depth, 0.0, 1.0);
                                    if (depth + draw_params.depth_bias < hiz_depth) {
                                        mesh_visible = 0u;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (mesh_visible == 0u || mesh_vert_count == 0u || mesh_prim_count == 0u) {
            mesh_output.vertex_count = 0u;
            mesh_output.primitive_count = 0u;
        } else {
            mesh_output.vertex_count = mesh_vert_count;
            mesh_output.primitive_count = mesh_prim_count;
        }
    }

    workgroupBarrier();

    if (mesh_visible == 0u) {
        return;
    }

    if (local_id.x < mesh_vert_count) {
        let global_index = meshlet_vertices[meshlet_vertex_offset + local_id.x];
        let base = global_index * 12u;
        let position = load_vec3(base);
        let normal = load_vec3(base + 3u);
        let tex_coord = load_vec2(base + 6u);
        let tangent = load_vec4(base + 8u);

        let world_pos = mesh_model * vec4<f32>(position, 1.0);
        let clip_position = camera.projection_matrix * camera.view_matrix * world_pos;

        let N = safe_normalize(mesh_normal_matrix * normal);
        let T = safe_normalize(mesh_normal_matrix * tangent.xyz);
        let B = cross(N, T) * tangent.w;

        mesh_output.vertices[local_id.x].clip_position = clip_position;
        mesh_output.vertices[local_id.x].world_position = world_pos.xyz;
        mesh_output.vertices[local_id.x].world_normal = N;
        mesh_output.vertices[local_id.x].tex_coord = tex_coord;
        mesh_output.vertices[local_id.x].world_tangent = T;
        mesh_output.vertices[local_id.x].world_bitangent = B;
        mesh_output.vertices[local_id.x].material_id = mesh_material_id;
    }

    for (var prim = local_id.x; prim < mesh_prim_count; prim = prim + 64u) {
        let base = meshlet_index_offset + prim * 3u;
        let i0 = meshlet_indices[base];
        let i1 = meshlet_indices[base + 1u];
        let i2 = meshlet_indices[base + 2u];
        mesh_output.primitives[prim].indices = vec3<u32>(i0, i1, i2);
    }
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let material = materials[in.material_id];

    var albedo = material.albedo.rgb;
    var alpha = material.albedo.a;
    if material.albedo_idx >= 0i {
        let sample = textureSampleBias(textures[material.albedo_idx], pbr_sampler, in.tex_coord, constants.mip_bias);
        albedo *= sample.rgb;
        alpha *= sample.a;
    }

    var emission = material.emission_color * material.emission_strength;
    if material.emission_idx >= 0i {
        emission *= textureSampleBias(textures[material.emission_idx], pbr_sampler, in.tex_coord, constants.mip_bias).rgb;
    }

    return vec4<f32>(albedo + emission, alpha);
}
