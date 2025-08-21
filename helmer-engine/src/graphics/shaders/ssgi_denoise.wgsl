// High-Performance Separable Filter

const EPSILON: f32 = 0.001;

@group(0) @binding(0) var t_noisy_input: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_normal: texture_2d<f32>;
@group(0) @binding(3) var s_input: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let radius = 3;
    let depth_tolerance = 0.1;
    let normal_tolerance = 0.9;
    
    let texel_size = 1.0 / vec2<f32>(textureDimensions(t_noisy_input));
    
    let center_color = textureSample(t_noisy_input, s_input, in.uv).rgb;
    let center_depth = textureSample(t_depth, s_input, in.uv);
    let center_normal = normalize(textureSample(t_normal, s_input, in.uv).xyz * 2.0 - 1.0);
    
    var final_color = center_color;
    var total_weight = 1.0;

    // --- Separable Cross-Blur ---
    // Horizontal Pass
    for (var i = -radius; i <= radius; i = i + 1) {
        if (i == 0) { continue; }
        let offset = vec2<f32>(f32(i) * texel_size.x, 0.0);
        let sample_uv = in.uv + offset;

        let sample_depth = textureSample(t_depth, s_input, sample_uv);
        let sample_normal = normalize(textureSample(t_normal, s_input, sample_uv).xyz * 2.0 - 1.0);
        
        let depth_diff = abs(sample_depth - center_depth);
        let normal_diff = dot(sample_normal, center_normal);

        let weight = select(0.0, 1.0, depth_diff < depth_tolerance && normal_diff > normal_tolerance);
        
        final_color += textureSample(t_noisy_input, s_input, sample_uv).rgb * weight;
        total_weight += weight;
    }

    // Vertical Pass
    for (var i = -radius; i <= radius; i = i + 1) {
        if (i == 0) { continue; }
        let offset = vec2<f32>(0.0, f32(i) * texel_size.y);
        let sample_uv = in.uv + offset;

        let sample_depth = textureSample(t_depth, s_input, sample_uv);
        let sample_normal = normalize(textureSample(t_normal, s_input, sample_uv).xyz * 2.0 - 1.0);
        
        let depth_diff = abs(sample_depth - center_depth);
        let normal_diff = dot(sample_normal, center_normal);

        let weight = select(0.0, 1.0, depth_diff < depth_tolerance && normal_diff > normal_tolerance);
        
        final_color += textureSample(t_noisy_input, s_input, sample_uv).rgb * weight;
        total_weight += weight;
    }
    
    return vec4<f32>(final_color / total_weight, 1.0);
}