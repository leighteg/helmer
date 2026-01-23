@group(0) @binding(0) var primary_tex: texture_2d<f32>;
@group(0) @binding(1) var fallback_tex: texture_2d<f32>;
@group(0) @binding(2) var scene_sampler: sampler;
@group(0) @binding(3) var<uniform> params: vec4<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = vec2<f32>(
        f32((in_vertex_index << 1u) & 2u),
        f32(in_vertex_index & 2u)
    );
    out.clip_position = vec4<f32>(
        out.tex_coords.x * 2.0 - 1.0,
        1.0 - out.tex_coords.y * 2.0,
        0.0,
        1.0
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let primary = textureSample(primary_tex, scene_sampler, in.tex_coords);
    let fallback = textureSample(fallback_tex, scene_sampler, in.tex_coords);
    let mix_val = clamp(params.x, 0.0, 1.0);
    let fallback_weight = fallback.a * mix(1.0 - primary.a, 1.0, mix_val);
    let color = primary.rgb * primary.a + fallback.rgb * fallback_weight;
    let alpha = clamp(primary.a + fallback_weight, 0.0, 1.0);
    return vec4<f32>(color, alpha);
}
