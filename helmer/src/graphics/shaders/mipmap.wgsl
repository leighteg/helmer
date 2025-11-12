@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

struct Varying {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> Varying {
    var out: Varying;
    let x = f32(in_vertex_index & 1u) * 2.0;
    let y = f32(in_vertex_index & 2u) * 2.0;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: Varying) -> @location(0) vec4<f32> {
    // Sample the source texture (the previous, larger mip level)
    // The linear sampler will average the 4 pixels for us.
    return textureSample(t_src, s_src, in.uv);
}