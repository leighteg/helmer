enable wgpu_mesh_shader;

const MAX_VERTS: u32 = 64u;
const MAX_PRIMS: u32 = 124u;

struct LightVP {
    view_proj: mat4x4<f32>,
}

struct Constants {
    // general
    mip_bias: f32,

    // lighting
    shade_mode: u32,
    light_model: u32,
    skylight_contribution: u32,

    // sky
    planet_radius: f32,
    atmosphere_radius: f32,
    sky_light_samples: u32,
    _pad0: u32,

    // SSR
    ssr_coarse_steps: u32,
    ssr_binary_search_steps: u32,
    ssr_linear_step_size: f32,
    ssr_thickness: f32,
    ssr_max_distance: f32,
    ssr_roughness_fade_start: f32,
    ssr_roughness_fade_end: f32,
    _pad1: u32,

    // SSGI
    ssgi_num_rays: u32,
    ssgi_num_steps: u32,
    ssgi_ray_step_size: f32,
    ssgi_thickness: f32,
    ssgi_blend_factor: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,

    // shadows
    evsm_c: f32,
    pcf_radius: u32,
    pcf_min_scale: f32,
    pcf_max_scale: f32,
    pcf_max_distance: f32,
    ssgi_intensity: f32,
    _final_padding: vec2<f32>,
};

struct ShadowInstance {
    model_matrix: mat4x4<f32>,
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
}

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
}

struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
    @builtin(vertices) vertices: array<VertexOutput, MAX_VERTS>,
    @builtin(primitives) primitives: array<PrimitiveOutput, MAX_PRIMS>,
}

var<workgroup> mesh_output: MeshOutput;
var<workgroup> mesh_visible: u32;
var<workgroup> mesh_model: mat4x4<f32>;
var<workgroup> mesh_vert_count: u32;
var<workgroup> mesh_prim_count: u32;
var<workgroup> mesh_max_vertex: u32;
var<workgroup> meshlet_vertex_offset: u32;
var<workgroup> meshlet_index_offset: u32;
var<workgroup> meshlet_center: vec3<f32>;
var<workgroup> meshlet_radius: f32;

@group(0) @binding(0) var<uniform> light_vp: LightVP;
@group(1) @binding(0) var<uniform> render_constants: Constants;
@group(2) @binding(0) var<storage, read> instances: array<ShadowInstance>;
@group(2) @binding(1) var<storage, read> meshlet_descs: array<MeshletDesc>;
@group(2) @binding(2) var<storage, read> meshlet_vertices: array<u32>;
@group(2) @binding(3) var<storage, read> meshlet_indices: array<u32>;
@group(2) @binding(4) var<storage, read> vertex_data: array<u32>;
@group(2) @binding(5) var<uniform> draw_params: MeshDrawParams;

fn load_f32(index: u32) -> f32 {
    return bitcast<f32>(vertex_data[index]);
}

fn load_vec3(base: u32) -> vec3<f32> {
    return vec3<f32>(load_f32(base), load_f32(base + 1u), load_f32(base + 2u));
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

        let instance_len = arrayLength(&instances);
        let meshlet_desc_len = arrayLength(&meshlet_descs);
        let meshlet_vert_len = arrayLength(&meshlet_vertices);
        let meshlet_index_len = arrayLength(&meshlet_indices);
        mesh_max_vertex = arrayLength(&vertex_data) / 12u;

        if (workgroup_id.y >= draw_params.instance_count) {
            mesh_visible = 0u;
        } else {
            let instance_index = draw_params.instance_base + workgroup_id.y;
            if (instance_index >= instance_len) {
                mesh_visible = 0u;
            } else {
                let inst = instances[instance_index];

                if (workgroup_id.x >= draw_params.meshlet_count) {
                    mesh_visible = 0u;
                } else {
                    let meshlet_index = draw_params.meshlet_base + workgroup_id.x;
                    if (meshlet_index >= meshlet_desc_len) {
                        mesh_visible = 0u;
                    } else {
                        let meshlet = meshlet_descs[meshlet_index];
                        let vert_offset = meshlet.vertex_offset;
                        let vert_count = meshlet.vertex_count;
                        let index_offset = meshlet.index_offset;
                        let index_count = meshlet.index_count;
                        if (vert_offset >= meshlet_vert_len || vert_count > (meshlet_vert_len - vert_offset)) {
                            mesh_visible = 0u;
                        } else if (index_offset >= meshlet_index_len || index_count > (meshlet_index_len - index_offset)) {
                            mesh_visible = 0u;
                        } else {
                            let prim_count = index_count / 3u;

                            if (vert_count == 0u || index_count == 0u) {
                                mesh_visible = 0u;
                            } else if (index_count % 3u != 0u) {
                                mesh_visible = 0u;
                            } else if (vert_count > MAX_VERTS || prim_count > MAX_PRIMS) {
                                mesh_visible = 0u;
                            } else {
                                mesh_vert_count = vert_count;
                                mesh_prim_count = prim_count;
                                meshlet_vertex_offset = vert_offset;
                                meshlet_index_offset = index_offset;
                                meshlet_center = meshlet.bounds_center;
                                meshlet_radius = meshlet.bounds_radius;

                                mesh_model = inst.model_matrix;

                                let flags = draw_params.flags;
                                let frustum_enabled = (flags & 1u) != 0u;

                                if (frustum_enabled) {
                                    let world_center =
                                        (mesh_model * vec4<f32>(meshlet_center, 1.0)).xyz;
                                    let world_radius = meshlet_radius * max_scale(mesh_model);

                                    if (!sphere_in_frustum(light_vp.view_proj, world_center, world_radius)) {
                                        mesh_visible = 0u;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (mesh_visible == 0u || mesh_vert_count == 0u || mesh_prim_count == 0u || mesh_max_vertex == 0u) {
            mesh_visible = 0u;
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
        let safe_index = min(global_index, mesh_max_vertex - 1u);
        let base = safe_index * 12u;
        let position = load_vec3(base);

        let world_position = mesh_model * vec4<f32>(position, 1.0);
        let clip_position = light_vp.view_proj * world_position;

        mesh_output.vertices[local_id.x].clip_position = clip_position;
        mesh_output.vertices[local_id.x].depth = clip_position.z / clip_position.w;
    }

    for (var prim = local_id.x; prim < mesh_prim_count; prim = prim + 64u) {
        let base = meshlet_index_offset + prim * 3u;
        let max_index = mesh_vert_count - 1u;
        let i0 = min(meshlet_indices[base], max_index);
        let i1 = min(meshlet_indices[base + 1u], max_index);
        let i2 = min(meshlet_indices[base + 2u], max_index);
        mesh_output.primitives[prim].indices = vec3<u32>(i0, i1, i2);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<f32> {
    let depth = clamp(in.depth, 0.0, 1.0);
    let warped_depth = exp(render_constants.evsm_c * (depth - 1.0));
    return vec2<f32>(warped_depth, warped_depth * warped_depth);
}
