@group(0) @binding(0) var depth_tex: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

fn load_depth(pos: vec4<f32>) -> f32 {
    let coords = vec2<i32>(pos.xy);
    return textureLoad(depth_tex, coords, 0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    return load_depth(pos);
}

@fragment
fn fs_main_rgba(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let depth = load_depth(pos);
    return vec4<f32>(depth, 0.0, 0.0, 1.0);
}
