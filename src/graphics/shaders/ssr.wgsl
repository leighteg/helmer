//=============== STRUCTS ===============//
struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
};

//=============== BINDINGS ===============//
@group(0) @binding(0) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(1) var gbuf_mra: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
// IMPORTANT: SSR should sample the lit scene from the previous frame, not just albedo.
// bind the output of lighting pass from frame N-1 here for frame N.
@group(0) @binding(3) var lit_scene_texture: texture_2d<f32>;
@group(0) @binding(4) var gbuf_sampler: sampler;
@group(0) @binding(5) var scene_sampler: sampler; // Should be a linear sampler

@group(1) @binding(0) var<uniform> camera: CameraUniforms;

//=============== CONSTANTS ===============//
const COARSE_STEPS = 32u;
const BINARY_SEARCH_STEPS = 6u; // Refinement steps
const STEP_SIZE = 0.15;         // A constant, linear step size
const MAX_DISTANCE = 250.0;     // Max reflection distance
const THICKNESS = 0.1;          // Surface thickness for intersection
const STRIDE = 2.0;             // Multiplier for the base step size
const ROUGHNESS_FADE_START = 0.1;
const ROUGHNESS_FADE_END = 0.5;
const EPSILON = 0.0001;
const SELF_INTERSECTION_TOLERANCE = 0.05; // A small tolerance to prevent a reflection ray from immediately hitting the surface it came from.

//=============== UTILITY FUNCTIONS ===============//

// A simple hash function to generate a pseudo-random value from screen coordinates.
fn rand(uv: vec2<f32>) -> f32 {
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
}

// Reconstructs View Space position from depth and screen UVs.
// Returns a vec4 homogeneous coordinate; the caller MUST check .w before dividing.
fn unproject_to_view_h(uv: vec2<f32>, depth: f32) -> vec4<f32> {
    // Restore the Y-flip, as the "pasted on" artifact proved it was necessary.
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
    let clip = vec4<f32>(ndc, 1.0);
    return camera.inverse_projection_matrix * clip;
}

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (abs(det) < 0.00001) {
        // Return identity if matrix is not invertible
        return mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    }

    let inv_det = 1.0 / det;
    var res: mat3x3<f32>;
    res[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det;
    res[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    res[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    res[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    res[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    res[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det;
    res[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det;
    res[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det;
    res[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det;
    return res;
}


//=============== SHADERS ===============//
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let screen_size = vec2<f32>(textureDimensions(gbuf_normal));
    let screen_uv = frag_coord.xy / screen_size;

    let mra = textureSample(gbuf_mra, gbuf_sampler, screen_uv);
    let roughness = mra.g;

    if (roughness > ROUGHNESS_FADE_END) {
        discard;
    }

    let origin_depth = textureSample(depth_texture, gbuf_sampler, screen_uv);
    if (origin_depth >= 1.0) {
        discard;
    }

    let origin_vs_h = unproject_to_view_h(screen_uv, origin_depth);
    if (abs(origin_vs_h.w) < EPSILON) { discard; }
    let origin_vs = origin_vs_h.xyz / origin_vs_h.w;

    let packed_normal = textureSample(gbuf_normal, gbuf_sampler, screen_uv).xyz;
    let N_world = normalize(packed_normal * 2.0 - 1.0);

    let normal_matrix = transpose(mat3_inverse(mat3x3<f32>(
        camera.view_matrix[0].xyz,
        camera.view_matrix[1].xyz,
        camera.view_matrix[2].xyz,
    )));
    let N_vs = normalize(normal_matrix * N_world);
    
    let V_vs = normalize(-origin_vs);
    let R_vs = reflect(-V_vs, N_vs);

    // --- Hierarchical Ray Marching with Dithering ---
    var step = 0.05 + rand(screen_uv) * 0.05;
    var ray_pos = origin_vs + R_vs * step;
    
    var hit_found = false;
    var last_pos = origin_vs;

    for (var i = 0u; i < COARSE_STEPS; i = i + 1u) {
        last_pos = ray_pos;
        ray_pos += R_vs * step;
        step *= 1.2;
        
        let projected_pos_cs = camera.projection_matrix * vec4(ray_pos, 1.0);
        if (projected_pos_cs.w <= EPSILON) { break; }
        let projected_uv = (projected_pos_cs.xy / projected_pos_cs.w) * vec2(0.5, -0.5) + 0.5;
        if (any(projected_uv < vec2(0.0)) || any(projected_uv > vec2(1.0)) || length(origin_vs - ray_pos) > MAX_DISTANCE) { break; }

        let scene_depth = textureSample(depth_texture, gbuf_sampler, projected_uv);
        let scene_vs_h = unproject_to_view_h(projected_uv, scene_depth);
        if (abs(scene_vs_h.w) < EPSILON) { continue; }
        let scene_vs = scene_vs_h.xyz / scene_vs_h.w;

        if (ray_pos.z < scene_vs.z && abs(scene_vs.z - origin_vs.z) > SELF_INTERSECTION_TOLERANCE) {
            hit_found = true;
            break;
        }
    }

    if (!hit_found) { discard; }

    // --- Binary Search Refinement ---
    var start_pos = last_pos;
    var end_pos = ray_pos;

    for (var j = 0u; j < BINARY_SEARCH_STEPS; j = j + 1u) {
        let mid_pos = (start_pos + end_pos) * 0.5;
        let mid_cs = camera.projection_matrix * vec4(mid_pos, 1.0);
        if (mid_cs.w <= EPSILON) { break; }
        let mid_uv = (mid_cs.xy / mid_cs.w) * vec2(0.5, -0.5) + 0.5;
        
        let scene_depth_at_mid = textureSample(depth_texture, gbuf_sampler, mid_uv);
        let scene_vs_at_mid_h = unproject_to_view_h(mid_uv, scene_depth_at_mid);
        if (abs(scene_vs_at_mid_h.w) <= EPSILON) { break; }
        let scene_z_at_mid = scene_vs_at_mid_h.z / scene_vs_at_mid_h.w;

        if (mid_pos.z < scene_z_at_mid) {
            end_pos = mid_pos;
        } else {
            start_pos = mid_pos;
        }
    }

    // --- Final Color and Confidence ---
    let final_cs = camera.projection_matrix * vec4(end_pos, 1.0);
    if (final_cs.w <= EPSILON) { discard; }
    let refined_uv = (final_cs.xy / final_cs.w) * vec2(0.5, -0.5) + 0.5;

    let reflection_color = textureSample(lit_scene_texture, scene_sampler, refined_uv);
    
    let edge_fade = smoothstep(0.0, 0.1, min(min(refined_uv.x, 1.0 - refined_uv.x), min(refined_uv.y, 1.0 - refined_uv.y)));
    let roughness_fade = 1.0 - smoothstep(ROUGHNESS_FADE_START, ROUGHNESS_FADE_END, roughness);
    let confidence = edge_fade * roughness_fade;

    return vec4(reflection_color.rgb, confidence);
}