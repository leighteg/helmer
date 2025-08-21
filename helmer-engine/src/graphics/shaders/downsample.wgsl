@group(0) @binding(0) var full_res_depth: texture_depth_2d;
@group(0) @binding(1) var full_res_normal: texture_2d<f32>;
@group(0) @binding(2) var full_res_lighting: texture_2d<f32>;
@group(0) @binding(3) var s_point: sampler; 

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

struct FragmentOutput {
    @location(0) half_res_depth: f32,
    @location(1) half_res_normal: vec4<f32>,
    @location(2) half_res_lighting: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(full_res_depth));
    
    // Sample a 2x2 block in the full-res textures
    let uv00 = in.uv;
    let uv10 = in.uv + vec2(texel_size.x, 0.0);
    let uv01 = in.uv + vec2(0.0, texel_size.y);
    let uv11 = in.uv + texel_size;

    // Use minimum depth (closest to camera) to preserve geometry edges
    let d0 = textureSample(full_res_depth, s_point, uv00);
    let d1 = textureSample(full_res_depth, s_point, uv10);
    let d2 = textureSample(full_res_depth, s_point, uv01);
    let d3 = textureSample(full_res_depth, s_point, uv11);
    let min_depth = min(min(d0, d1), min(d2, d3));

    // For normal and lighting, a simple average of the 2x2 block is effective
    let n0 = textureSample(full_res_normal, s_point, uv00);
    let n1 = textureSample(full_res_normal, s_point, uv10);
    let n2 = textureSample(full_res_normal, s_point, uv01);
    let n3 = textureSample(full_res_normal, s_point, uv11);
    let avg_normal = (n0 + n1 + n2 + n3) * 0.25;

    let l0 = textureSample(full_res_lighting, s_point, uv00);
    let l1 = textureSample(full_res_lighting, s_point, uv10);
    let l2 = textureSample(full_res_lighting, s_point, uv01);
    let l3 = textureSample(full_res_lighting, s_point, uv11);
    let avg_lighting = (l0 + l1 + l2 + l3) * 0.25;

    var out: FragmentOutput;
    out.half_res_depth = min_depth;
    out.half_res_normal = avg_normal;
    out.half_res_lighting = avg_lighting;
    return out;
}