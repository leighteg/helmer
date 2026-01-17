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

struct InstanceData {
    model_matrix: mat4x4<f32>,
    material_id: u32,
    visibility: u32,
    _pad0: vec2<u32>,
    bounds_center: vec4<f32>,
    bounds_extents: vec4<f32>,
}

struct CullParams {
    instance_count: u32,
    _pad0: vec3<u32>,
    depth_bias: f32,
    rect_pad: f32,
    _pad1: vec2<f32>,
}

@group(0) @binding(0) var<storage, read_write> instances: array<InstanceData>;
@group(0) @binding(1) var hiz_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;
@group(0) @binding(3) var<uniform> params: CullParams;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.instance_count) {
        return;
    }

    let inst = instances[idx];
    let center = inst.bounds_center.xyz;
    let extents = inst.bounds_extents.xyz;
    let view_proj = camera.projection_matrix * camera.view_matrix;
    let prev_view_proj = camera.prev_view_proj;

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
        let local = center + extents * signs[i];
        let world = inst.model_matrix * vec4<f32>(local, 1.0);

        let clip_prev = prev_view_proj * world;
        if (clip_prev.w > 0.0) {
            let ndc_prev = clip_prev.xyz / clip_prev.w;
            min_ndc = min(min_ndc, ndc_prev.xy);
            max_ndc = max(max_ndc, ndc_prev.xy);
            max_depth = max(max_depth, ndc_prev.z);
            valid_prev = true;
            valid_any = true;
        }

        let clip_cur = view_proj * world;
        if (clip_cur.w > 0.0) {
            let ndc_cur = clip_cur.xyz / clip_cur.w;
            min_ndc = min(min_ndc, ndc_cur.xy);
            max_ndc = max(max_ndc, ndc_cur.xy);
            valid_any = true;
        }
    }

    if (!valid_prev || !valid_any) {
        instances[idx].visibility = 1u;
        return;
    }

    let uv_min = clamp(min_ndc * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_max = clamp(max_ndc * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    if (uv_min.x >= uv_max.x || uv_min.y >= uv_max.y) {
        instances[idx].visibility = 1u;
        return;
    }

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
    let pad = vec2<f32>(params.rect_pad);
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
    let occluded = depth + params.depth_bias < hiz_depth;
    instances[idx].visibility = select(1u, 0u, occluded);
}
