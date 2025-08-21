// ssgi_denoise.wgsl (True Separable Bilateral Filter)

const EPSILON: f32 = 0.001;

// Add this uniform to your bindings
struct DenoiseUniforms {
    // 0 for Horizontal, 1 for Vertical
    direction: u32, 
};
@group(1) @binding(0) var<uniform> denoise_uniforms: DenoiseUniforms;

// (Keep your existing bindings and vertex shader)

// --- Helper Functions ---
// Convert non-linear depth buffer value to linear view-space Z
fn linearize_depth(depth: f32, projection_matrix: mat4x4<f32>) -> f32 {
    // This depends on your projection matrix. For a standard perspective matrix:
    // P[2][2] = (far + near) / (near - far)
    // P[3][2] = (2 * far * near) / (near - far)
    // Invert to get: view_z = P[3][2] / (depth * 2.0 - 1.0 - P[2][2])
    // You should pass these projection terms in your camera uniform.
    // As a simpler (but less robust) alternative, we can just use the depth diff.
    return depth; // Using raw depth for simplicity, but linearizing is better.
}

// Gaussian weight function
fn gaussian(x: f32, sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    return exp(-(x * x) / (2.0 * sigma_sq));
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let radius: i32 = 4;
    let sigma: f32 = 2.5;
    let depth_tolerance: f32 = 0.05;  // More sensitive now
    let normal_tolerance: f32 = 0.9;
    
    let texel_size = 1.0 / vec2<f32>(textureDimensions(t_noisy_input));
    
    let center_color = textureSample(t_noisy_input, s_input, in.uv).rgb;
    let center_depth = linearize_depth(textureLoad(t_depth, vec2<u32>(in.uv * texel_size_inverse), 0), ...); // Ideally use linear depth
    let center_normal = normalize(textureLoad(t_normal, vec2<u32>(in.uv * texel_size_inverse), 0).xyz * 2.0 - 1.0);
    
    var final_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i: i32 = -radius; i <= radius; i = i + 1) {
        let offset_dir = select(vec2<f32>(f32(i), 0.0), vec2<f32>(0.0, f32(i)), denoise_uniforms.direction == 1u);
        let sample_uv = in.uv + offset_dir * texel_size;

        if (any(sample_uv < vec2(0.0)) || any(sample_uv > vec2(1.0))) {
            continue;
        }

        let sample_color = textureSample(t_noisy_input, s_input, sample_uv).rgb;
        let sample_depth = linearize_depth(textureSample(t_depth, s_input, sample_uv), ...);
        let sample_normal = normalize(textureSample(t_normal, s_input, sample_uv).xyz * 2.0 - 1.0);
        
        let depth_diff = abs(sample_depth - center_depth);
        let normal_sim = dot(sample_normal, center_normal);

        // Bilateral weights
        let geo_weight = select(0.0, 1.0, depth_diff < depth_tolerance && normal_sim > normal_tolerance);
        let spatial_weight = gaussian(f32(i), sigma);
        let weight = spatial_weight * geo_weight;
        
        final_color += sample_color * weight;
        total_weight += weight;
    }
    
    if (total_weight < EPSILON) {
        return vec4<f32>(center_color, 1.0);
    }

    return vec4<f32>(final_color / total_weight, 1.0);
}