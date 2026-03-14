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

struct InstanceInput {
    prev_model: mat4x4<f32>,
    curr_model: mat4x4<f32>,
    bounds_center: vec4<f32>,
    bounds_extents: vec4<f32>,
    material_id: u32,
    mesh_id: u32,
    casts_shadow: u32,
    skin_offset: u32,
    skin_count: u32,
    alpha_mode: u32,
    _pad0: array<u32, 2>,
}

struct GBufferInstance {
    model_matrix: mat4x4<f32>,
    material_id: u32,
    visibility: u32,
    skin_offset: u32,
    skin_count: u32,
    bounds_center: vec4<f32>,
    bounds_extents: vec4<f32>,
}

struct ShadowInstance {
    model_matrix: mat4x4<f32>,
    material_id: u32,
    skin_offset: u32,
    skin_count: u32,
    _pad0: u32,
}

const ALPHA_MODE_OPAQUE: u32 = 0u;
const ALPHA_MODE_MASK: u32 = 1u;
const ALPHA_MODE_BLEND: u32 = 2u;
const ALPHA_MODE_PREMULTIPLIED: u32 = 3u;
const ALPHA_MODE_ADDITIVE: u32 = 4u;
const FLAG_TRANSPARENT_SHADOWS: u32 = 8u;

struct MeshMeta {
    lod_count: u32,
    base_draw: u32,
    instance_capacity: u32,
    base_instance: u32,
}

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct DrawMeshTasksIndirect {
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
}

struct MeshTaskMeta {
    meshlet_count: u32,
    instance_capacity: u32,
    tile_meshlets: u32,
    tile_instances: u32,
    task_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct CullParams {
    instance_count: u32,
    draw_count: u32,
    mesh_count: u32,
    flags: u32,
    lod0_dist_sq: f32,
    lod1_dist_sq: f32,
    lod2_dist_sq: f32,
    alpha: f32,
    occlusion_depth_bias: f32,
    occlusion_rect_pad: f32,
    output_capacity: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> instances_in: array<InstanceInput>;
@group(0) @binding(1) var<storage, read_write> gbuffer_out: array<GBufferInstance>;
@group(0) @binding(2) var<storage, read_write> shadow_out: array<ShadowInstance>;
@group(0) @binding(3) var<storage, read> mesh_meta: array<MeshMeta>;
@group(0) @binding(4) var<storage, read_write> gbuffer_draws: array<DrawIndexedIndirect>;
@group(0) @binding(5) var<storage, read_write> shadow_draws: array<DrawIndexedIndirect>;
@group(0) @binding(6) var hiz_tex: texture_2d<f32>;
@group(0) @binding(7) var<uniform> camera: CameraUniforms;
@group(0) @binding(8) var<uniform> params: CullParams;
@group(0) @binding(9) var<storage, read> mesh_task_meta: array<MeshTaskMeta>;
@group(0) @binding(10) var<storage, read_write> gbuffer_mesh_tasks: array<DrawMeshTasksIndirect>;
@group(0) @binding(11) var<storage, read_write> shadow_mesh_tasks: array<DrawMeshTasksIndirect>;

fn select_model(prev_model: mat4x4<f32>, curr_model: mat4x4<f32>, alpha: f32) -> mat4x4<f32> {
    if (alpha <= 0.0) {
        return prev_model;
    }
    if (alpha >= 1.0) {
        return curr_model;
    }
    return mat4x4<f32>(
        mix(prev_model[0], curr_model[0], alpha),
        mix(prev_model[1], curr_model[1], alpha),
        mix(prev_model[2], curr_model[2], alpha),
        mix(prev_model[3], curr_model[3], alpha),
    );
}

fn frustum_visible(min_ndc: vec2<f32>, max_ndc: vec2<f32>) -> bool {
    if (max_ndc.x < -1.0 || min_ndc.x > 1.0) {
        return false;
    }
    if (max_ndc.y < -1.0 || min_ndc.y > 1.0) {
        return false;
    }
    return true;
}

@compute @workgroup_size(64, 1, 1)
fn clear_draws(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.draw_count) {
        return;
    }
    atomicStore(&gbuffer_draws[idx].instance_count, 0u);
    atomicStore(&shadow_draws[idx].instance_count, 0u);
}

@compute @workgroup_size(64, 1, 1)
fn cull_instances(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.instance_count) {
        return;
    }

    let inst = instances_in[idx];
    if (inst.mesh_id >= params.mesh_count) {
        return;
    }
    let mesh = mesh_meta[inst.mesh_id];
    if (mesh.lod_count == 0u || mesh.instance_capacity == 0u) {
        return;
    }

    let flags = params.flags;
    let frustum_enabled = (flags & 1u) != 0u;
    let occlusion_enabled = (flags & 2u) != 0u;
    let lod_enabled = (flags & 4u) != 0u;
    let transparent_shadows = (flags & FLAG_TRANSPARENT_SHADOWS) != 0u;

    let model = select_model(inst.prev_model, inst.curr_model, params.alpha);
    let view_proj = camera.projection_matrix * camera.view_matrix;
    let prev_view_proj = camera.prev_view_proj;

    let center = inst.bounds_center.xyz;
    let extents = inst.bounds_extents.xyz;

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
    var min_prev = vec2<f32>(1.0, 1.0);
    var max_prev = vec2<f32>(-1.0, -1.0);
    var max_depth = 0.0;
    var valid_cur = false;
    var valid_prev = false;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let local = center + extents * signs[i];
        let world = model * vec4<f32>(local, 1.0);

        let clip_cur = view_proj * world;
        if (clip_cur.w > 0.0) {
            let ndc_cur = clip_cur.xyz / clip_cur.w;
            min_ndc = min(min_ndc, ndc_cur.xy);
            max_ndc = max(max_ndc, ndc_cur.xy);
            valid_cur = true;
        }

        let clip_prev = prev_view_proj * world;
        if (clip_prev.w > 0.0) {
            let ndc_prev = clip_prev.xyz / clip_prev.w;
            min_prev = min(min_prev, ndc_prev.xy);
            max_prev = max(max_prev, ndc_prev.xy);
            max_depth = max(max_depth, ndc_prev.z);
            valid_prev = true;
        }
    }

    if (frustum_enabled) {
        if (!valid_cur || !frustum_visible(min_ndc, max_ndc)) {
            return;
        }
    }

    if (occlusion_enabled && valid_prev) {
        let uv_min = clamp(min_prev * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
        let uv_max = clamp(max_prev * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
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
            let pad = vec2<f32>(params.occlusion_rect_pad);
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
            if (depth + params.occlusion_depth_bias < hiz_depth) {
                return;
            }
        }
    }

    var lod_index: u32 = 0u;
    if (lod_enabled) {
        let world_center = (model * vec4<f32>(center, 1.0)).xyz;
        let diff = world_center - camera.view_position;
        let distance_sq = dot(diff, diff);
        if (distance_sq > params.lod0_dist_sq) {
            lod_index = 1u;
        }
        if (distance_sq > params.lod1_dist_sq) {
            lod_index = 2u;
        }
        if (distance_sq > params.lod2_dist_sq) {
            lod_index = 3u;
        }
    }

    if (lod_index >= mesh.lod_count) {
        lod_index = mesh.lod_count - 1u;
    }

    let draw_index = mesh.base_draw + lod_index;
    if (draw_index >= params.draw_count) {
        return;
    }
    let capacity = mesh.instance_capacity;
    let base = mesh.base_instance + lod_index * capacity;
    if (base >= params.output_capacity) {
        return;
    }
    let remaining = params.output_capacity - base;
    if (capacity > remaining) {
        return;
    }
    let transparent_mode = inst.alpha_mode == ALPHA_MODE_BLEND
        || inst.alpha_mode == ALPHA_MODE_PREMULTIPLIED
        || inst.alpha_mode == ALPHA_MODE_ADDITIVE;
    if (!transparent_mode) {
        let write_idx = atomicAdd(&gbuffer_draws[draw_index].instance_count, 1u);
        if (write_idx < capacity) {
            let out_idx = base + write_idx;
            gbuffer_out[out_idx].model_matrix = model;
            gbuffer_out[out_idx].material_id = inst.material_id;
            gbuffer_out[out_idx].visibility = 1u;
            gbuffer_out[out_idx].skin_offset = inst.skin_offset;
            gbuffer_out[out_idx].skin_count = inst.skin_count;
            gbuffer_out[out_idx].bounds_center = inst.bounds_center;
            gbuffer_out[out_idx].bounds_extents = inst.bounds_extents;
        }
    }

    if (inst.casts_shadow != 0u && (!transparent_mode || transparent_shadows)) {
        let shadow_idx = atomicAdd(&shadow_draws[draw_index].instance_count, 1u);
        if (shadow_idx < capacity) {
            let out_idx = base + shadow_idx;
            shadow_out[out_idx].model_matrix = model;
            shadow_out[out_idx].material_id = inst.material_id;
            shadow_out[out_idx].skin_offset = inst.skin_offset;
            shadow_out[out_idx].skin_count = inst.skin_count;
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn build_mesh_tasks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.draw_count) {
        return;
    }

    let _meta = mesh_task_meta[idx];
    if (_meta.meshlet_count == 0u || _meta.instance_capacity == 0u) {
        return;
    }
    if (_meta.tile_meshlets == 0u || _meta.tile_instances == 0u) {
        return;
    }

    let tiles_x = (_meta.meshlet_count + _meta.tile_meshlets - 1u) / _meta.tile_meshlets;
    let tiles_y = (_meta.instance_capacity + _meta.tile_instances - 1u) / _meta.tile_instances;
    let total_tiles = tiles_x * tiles_y;
    let gbuffer_instances = atomicLoad(&gbuffer_draws[idx].instance_count);
    let shadow_instances = atomicLoad(&shadow_draws[idx].instance_count);

    for (var tile = 0u; tile < total_tiles; tile = tile + 1u) {
        let tile_x = tile % tiles_x;
        let tile_y = tile / tiles_x;

        let meshlet_base = tile_x * _meta.tile_meshlets;
        let meshlet_count = min(
            _meta.tile_meshlets,
            _meta.meshlet_count - meshlet_base
        );

        let instance_base = tile_y * _meta.tile_instances;
        let instance_count = min(
            _meta.tile_instances,
            _meta.instance_capacity - instance_base
        );

        var gbuffer_visible = 0u;
        if (gbuffer_instances > instance_base) {
            gbuffer_visible = min(gbuffer_instances - instance_base, instance_count);
        }

        var shadow_visible = 0u;
        if (shadow_instances > instance_base) {
            shadow_visible = min(shadow_instances - instance_base, instance_count);
        }

        let task_index = _meta.task_offset + tile;
        gbuffer_mesh_tasks[task_index].group_count_x = meshlet_count;
        gbuffer_mesh_tasks[task_index].group_count_y = gbuffer_visible;
        gbuffer_mesh_tasks[task_index].group_count_z = 1u;
        shadow_mesh_tasks[task_index].group_count_x = meshlet_count;
        shadow_mesh_tasks[task_index].group_count_y = shadow_visible;
        shadow_mesh_tasks[task_index].group_count_z = 1u;
    }
}
