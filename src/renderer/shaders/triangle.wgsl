// Triangle.wgsl - Basic shader for rendering a triangle
// Put this in your project's "shaders" directory

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct Constants {
    time: f32,
    width: u32,
    height: u32,
};

@group(0) @binding(0) var<uniform> constants: Constants;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5)
    );
    
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0)
    );
    
    // Rotate the triangle based on time
    let angle = constants.time;
    let pos = positions[vertex_index];
    let rotated_x = pos.x * cos(angle) - pos.y * sin(angle);
    let rotated_y = pos.x * sin(angle) + pos.y * cos(angle);
    let rotated_pos = vec2<f32>(rotated_x, rotated_y);
    
    var output: VertexOutput;
    output.position = vec4<f32>(rotated_pos, 0.0, 1.0);
    output.color = vec4<f32>(colors[vertex_index], 1.0);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}