// Spatiotemporal Variance-Guided Filter (SVGF-style) Denoiser

struct Camera {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    prev_view_proj: mat4x4<f32>, 
};

@group(0) @binding(0) var t_noisy_input: texture_2d<f32>;      // Raw SSGI (Half-Res)
@group(0) @binding(1) var t_depth: texture_2d<f32>;             // Depth (Half-Res)
@group(0) @binding(2) var t_normal: texture_2d<f32>;            // Normals (Half-Res)
@group(0) @binding(3) var t_history: texture_2d<f32>;           // Denoised from last frame (Half-Res)
@group(0) @binding(4) var s_linear: sampler;
@group(0) @binding(5) var s_point: sampler;
@group(1) @binding(0) var<uniform> camera: Camera;

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

fn world_from_depth(uv: vec2<f32>, depth: f32, inv_vp: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world = inv_vp * ndc;
    return world.xyz / world.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(t_noisy_input, 0));
    let center_uv = in.uv;
    
    let center_depth = textureSampleLevel(t_depth, s_point, center_uv, 0.0).r;
    if (center_depth >= 1.0) {
        return vec4(0.0);
    }
    
    let center_normal = normalize(textureSampleLevel(t_normal, s_point, center_uv, 0.0).xyz * 2.0 - 1.0);
    let center_color = textureSampleLevel(t_noisy_input, s_linear, center_uv, 0.0).rgb;

    // --- 1. Temporal Reprojection with History Rejection ---
    let world_pos = world_from_depth(center_uv, center_depth, camera.inverse_view_projection_matrix);
    var prev_clip = camera.prev_view_proj * vec4(world_pos, 1.0);
    prev_clip /= prev_clip.w;
    let prev_uv = saturate(prev_clip.xy * vec2(0.5, -0.5) + 0.5);
    
    var alpha = 0.1;

    // Check if previous frame's data is valid
    let prev_depth = textureSampleLevel(t_depth, s_point, prev_uv, 0.0).r;
    let prev_world_pos = world_from_depth(prev_uv, prev_depth, camera.inverse_view_projection_matrix);
    let prev_normal = normalize(textureSampleLevel(t_normal, s_point, prev_uv, 0.0).xyz * 2.0 - 1.0);

    let world_dist = distance(world_pos, prev_world_pos);
    let normal_dot = dot(center_normal, prev_normal);

    // If history is too different, reject it by increasing alpha
    if (world_dist > 0.5 || normal_dot < 0.95) {
        alpha = 0.5;
    }
    
    let history_color = textureSampleLevel(t_history, s_linear, prev_uv, 0.0).rgb;
    var accumulated_color = mix(history_color, center_color, alpha);

    // --- 2. Variance Estimation ---
    var moment1 = accumulated_color;
    var moment2 = accumulated_color * accumulated_color;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) { continue; }
            let neighbor_uv = center_uv + vec2(f32(x), f32(y)) * texel_size;
            let neighbor_color = textureSampleLevel(t_noisy_input, s_linear, neighbor_uv, 0.0).rgb;
            moment1 += neighbor_color;
            moment2 += neighbor_color * neighbor_color;
        }
    }
    moment1 /= 9.0;
    moment2 /= 9.0;
    let variance = max(vec3(0.0), moment2 - (moment1 * moment1));
    let filter_strength = 1.0 - smoothstep(0.0, 0.05, length(variance));

    // --- 3. Atrous Spatial Filtering ---
    var filtered_color = accumulated_color;
    let radius = 2;
    let depth_tolerance = 0.1;
    let normal_tolerance = 0.9;

    for (var i = 0; i < 4; i++) { // 4 iterations of the filter
        let step_width = pow(2.0, f32(i));
        var sum = filtered_color;
        var total_weight = 1.0;

        for (var y = -radius; y <= radius; y++) {
            for (var x = -radius; x <= radius; x++) {
                if (x == 0 && y == 0) { continue; }

                let offset = vec2(f32(x), f32(y)) * texel_size * step_width;
                let sample_uv = center_uv + offset;

                let sample_depth = textureSampleLevel(t_depth, s_point, sample_uv, 0.0).r;
                let sample_normal = normalize(textureSampleLevel(t_normal, s_point, sample_uv, 0.0).xyz * 2.0 - 1.0);
                
                let depth_diff = abs(sample_depth - center_depth);
                let normal_dot_spatial = dot(sample_normal, center_normal);

                var weight = select(0.0, 1.0, depth_diff < depth_tolerance && normal_dot_spatial > normal_tolerance);
                weight *= filter_strength;

                sum += textureSampleLevel(t_noisy_input, s_linear, sample_uv, 0.0).rgb * weight;
                total_weight += weight;
            }
        }
        filtered_color = sum / total_weight;
    }

    return vec4(filtered_color, 1.0);
}
