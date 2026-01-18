@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / vec2<f32>(dims);
    let color = textureSampleLevel(src_tex, src_sampler, uv, 0.0);
    textureStore(dst_tex, vec2<i32>(gid.xy), color);
}
