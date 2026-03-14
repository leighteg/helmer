@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(depth_tex, 0);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let depth = textureLoad(depth_tex, vec2<i32>(gid.xy), 0).x;
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(depth, 0.0, 0.0, 1.0));
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(dst_tex);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) {
        return;
    }

    let src_coord = vec2<i32>(gid.xy) * 2;
    let src_dims = textureDimensions(src_tex, 0);

    let x0 = clamp(src_coord.x, 0, i32(src_dims.x) - 1);
    let y0 = clamp(src_coord.y, 0, i32(src_dims.y) - 1);
    let x1 = clamp(src_coord.x + 1, 0, i32(src_dims.x) - 1);
    let y1 = clamp(src_coord.y + 1, 0, i32(src_dims.y) - 1);

    let d0 = textureLoad(src_tex, vec2<i32>(x0, y0), 0).x;
    let d1 = textureLoad(src_tex, vec2<i32>(x1, y0), 0).x;
    let d2 = textureLoad(src_tex, vec2<i32>(x0, y1), 0).x;
    let d3 = textureLoad(src_tex, vec2<i32>(x1, y1), 0).x;

    let max_depth = max(max(d0, d1), max(d2, d3));
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(max_depth, 0.0, 0.0, 1.0));
}
