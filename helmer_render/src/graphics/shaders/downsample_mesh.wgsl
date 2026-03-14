enable wgpu_mesh_shader;

@group(0) @binding(0) var full_res_depth: texture_2d<f32>;
@group(0) @binding(1) var full_res_normal: texture_2d<f32>;
@group(0) @binding(2) var full_res_albedo: texture_2d<f32>;
@group(0) @binding(3) var full_res_lighting_diffuse: texture_2d<f32>;
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

struct FragmentOutput {
    @location(0) half_res_depth: f32,
    @location(1) half_res_normal: vec4<f32>,
    @location(2) half_res_albedo: vec4<f32>,
    @location(3) half_res_lighting_diffuse: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(full_res_depth, 0));
    
    let uv00 = in.uv;
    let uv10 = in.uv + vec2(texel_size.x, 0.0);
    let uv01 = in.uv + vec2(0.0, texel_size.y);
    let uv11 = in.uv + texel_size;

    // Use minimum depth (closest to camera) to preserve geometry edges
    let d0 = textureSample(full_res_depth, s_point, uv00).x;
    let d1 = textureSample(full_res_depth, s_point, uv10).x;
    let d2 = textureSample(full_res_depth, s_point, uv01).x;
    let d3 = textureSample(full_res_depth, s_point, uv11).x;
    let min_depth = min(min(d0, d1), min(d2, d3));

    let n0 = textureSample(full_res_normal, s_point, uv00);
    let n1 = textureSample(full_res_normal, s_point, uv10);
    let n2 = textureSample(full_res_normal, s_point, uv01);
    let n3 = textureSample(full_res_normal, s_point, uv11);
    let avg_normal = (n0 + n1 + n2 + n3) * 0.25;

    let a0 = textureSample(full_res_albedo, s_point, uv00);
    let a1 = textureSample(full_res_albedo, s_point, uv10);
    let a2 = textureSample(full_res_albedo, s_point, uv01);
    let a3 = textureSample(full_res_albedo, s_point, uv11);
    let avg_albedo = (a0 + a1 + a2 + a3) * 0.25;

    let l0 = textureSample(full_res_lighting_diffuse, s_point, uv00);
    let l1 = textureSample(full_res_lighting_diffuse, s_point, uv10);
    let l2 = textureSample(full_res_lighting_diffuse, s_point, uv01);
    let l3 = textureSample(full_res_lighting_diffuse, s_point, uv11);
    let avg_lighting_diffuse = (l0 + l1 + l2 + l3) * 0.25;

    var out: FragmentOutput;
    out.half_res_depth = min_depth;
    out.half_res_normal = avg_normal;
    out.half_res_albedo = avg_albedo;
    out.half_res_lighting_diffuse = avg_lighting_diffuse;
    return out;
}
