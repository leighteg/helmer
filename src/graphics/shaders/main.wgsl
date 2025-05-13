// Basic Color Shader (wgsl)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct Constants {
    time: f32,
    x: f32,
    y: f32,
    width: u32,
    height: u32,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> constants: Constants;

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5),
        vec2<f32>(0.0, 0.5),
    );
    
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.5, 1.0),
    );

    let pos = positions[vertex_idx];
    let aspect = f32(constants.width) / f32(constants.height);
    let transformed_pos = vec2<f32>(pos.x / aspect + constants.x, pos.y + constants.y);
    
    var output: VertexOutput;
    output.position = vec4<f32>(transformed_pos, 0.0, 1.0);
    output.color = constants.color;
    output.uv = uvs[vertex_idx];
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// ----------------------------
// Textured Shader (wgsl)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct Constants {
    model_view_proj: mat4x4<f32>,
    time: f32,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var texture: texture_2d<f32>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>
) -> VertexOutput {
    var output: VertexOutput;
    output.position = constants.model_view_proj * vec4<f32>(position, 1.0);
    output.color = color * constants.color;
    output.uv = uv;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(texture, texture_sampler, input.uv);
    return tex_color * input.color;
}

// ----------------------------
// Animated Sprite Shader (wgsl)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct Constants {
    model_view_proj: mat4x4<f32>,
    time: f32,
    color: vec4<f32>,
    frame_count: u32,
    current_frame: u32,
    rows: u32,
    columns: u32,
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var spritesheet: texture_2d<f32>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>
) -> VertexOutput {
    var output: VertexOutput;
    output.position = constants.model_view_proj * vec4<f32>(position, 1.0);
    output.color = color * constants.color;
    
    // Calculate sprite sheet coordinates
    let frame_width = 1.0 / f32(constants.columns);
    let frame_height = 1.0 / f32(constants.rows);
    
    let current_column = constants.current_frame % constants.columns;
    let current_row = constants.current_frame / constants.columns;
    
    let frame_x = f32(current_column) * frame_width;
    let frame_y = f32(current_row) * frame_height;
    
    // Adjust UVs to point to the current frame in the sprite sheet
    output.uv = vec2<f32>(
        frame_x + uv.x * frame_width,
        frame_y + uv.y * frame_height
    );
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(spritesheet, texture_sampler, input.uv);
    return tex_color * input.color;
}

// ----------------------------
// Basic Lighting Shader (wgsl)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) world_pos: vec3<f32>,
};

struct Constants {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    color: vec4<f32>,
    light_position: vec3<f32>,
    light_color: vec3<f32>,
    ambient_strength: f32,
    specular_strength: f32,
    view_position: vec3<f32>,
};

@group(0) @binding(0) var<uniform> constants: Constants;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec4<f32>
) -> VertexOutput {
    var output: VertexOutput;
    
    // Transform to clip space
    let model_position = constants.model * vec4<f32>(position, 1.0);
    output.position = constants.projection * constants.view * model_position;
    
    // Transform normal
    let normal_matrix = mat3x3<f32>(
        constants.model[0].xyz,
        constants.model[1].xyz,
        constants.model[2].xyz
    );
    output.normal = normalize(normal_matrix * normal);
    
    output.world_pos = model_position.xyz;
    output.color = color * constants.color;
    output.uv = uv;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Ambient component
    let ambient = constants.ambient_strength * constants.light_color;
    
    // Diffuse component
    let normal = normalize(input.normal);
    let light_dir = normalize(constants.light_position - input.world_pos);
    let diff = max(dot(normal, light_dir), 0.0);
    let diffuse = diff * constants.light_color;
    
    // Specular component
    let view_dir = normalize(constants.view_position - input.world_pos);
    let reflect_dir = reflect(-light_dir, normal);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    let specular = constants.specular_strength * spec * constants.light_color;
    
    // Combine components
    let result = (ambient + diffuse + specular) * input.color.rgb;
    
    return vec4<f32>(result, input.color.a);
}

// ----------------------------
// Post-processing Shader Example (wgsl)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Constants {
    time: f32,
    effect_strength: f32,
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var screen_texture: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_idx], 0.0, 1.0);
    output.uv = uvs[vertex_idx];
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Wave distortion effect
    let wave_strength = constants.effect_strength * 0.01;
    let distort_x = sin(input.uv.y * 40.0 + constants.time) * wave_strength;
    let distort_uv = vec2<f32>(input.uv.x + distort_x, input.uv.y);
    
    // Sample the screen texture with distorted UVs
    let color = textureSample(screen_texture, texture_sampler, distort_uv);
    
    // Apply a vignette effect
    let vignette = 1.0 - length(input.uv - 0.5) * 1.5;
    let vignette_color = color.rgb * clamp(vignette, 0.0, 1.0);
    
    return vec4<f32>(vignette_color, color.a);
}