enable wgpu_mesh_shader;

@group(0) @binding(0) var low_res_ssgi: texture_2d<f32>;
@group(0) @binding(1) var full_res_depth: texture_2d<f32>;
@group(0) @binding(2) var full_res_normal: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;
@group(0) @binding(4) var s_point: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
};

var<workgroup> mesh_output: MeshOutput;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

fn write_vertex(index: u32) {
    let x = f32(index / 2u) * 4.0 - 1.0;
    let y = f32(index % 2u) * 4.0 - 1.0;
    mesh_output.vertices[index].clip_position = vec4<f32>(x, y, 1.0, 1.0);
    mesh_output.vertices[index].uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
}

@mesh(mesh_output)
@workgroup_size(1)
fn ms_main() {
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    write_vertex(0u);
    write_vertex(1u);
    write_vertex(2u);
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-0.5 * (x * x) / (sigma * sigma));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_depth = textureSample(full_res_depth, s_point, in.uv).x;
    if (center_depth >= 1.0) {
        return vec4(0.0);
    }
    let center_normal = normalize(textureSample(full_res_normal, s_point, in.uv).xyz * 2.0 - 1.0);

    let low_res_texel = 1.0 / vec2<f32>(textureDimensions(low_res_ssgi, 0));
    
    var final_color = vec3(0.0);
    var total_weight = 0.0;
    let radius = 2;
    let depth_sigma = 0.01;
    let normal_sigma = 0.1;

    // Joint Bilateral Upsampling using a 5x5 kernel
    for (var y = -radius; y <= radius; y = y + 1) {
        for (var x = -radius; x <= radius; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * low_res_texel;
            let sample_uv = in.uv + offset;

            let sample_depth = textureSample(full_res_depth, s_point, sample_uv).x;
            let sample_normal = normalize(textureSample(full_res_normal, s_point, sample_uv).xyz * 2.0 - 1.0);

            let depth_diff = abs(sample_depth - center_depth);
            // Use 1.0 - dot product for normal difference
            let normal_diff = acos(saturate(dot(center_normal, sample_normal)));

            let depth_weight = gaussian(depth_diff, depth_sigma);
            let normal_weight = gaussian(normal_diff, normal_sigma);
            let weight = depth_weight * normal_weight;
            
            final_color += textureSample(low_res_ssgi, s_linear, sample_uv).rgb * weight;
            total_weight += weight;
        }
    }
    
    if (total_weight < 0.001) {
        // Fallback to simple linear sampling if no valid neighbors are found
        return textureSample(low_res_ssgi, s_linear, in.uv);
    }
    
    return vec4(final_color / total_weight, 1.0);
}
