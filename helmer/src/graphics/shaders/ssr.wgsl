struct Constants {
    // lighting
    shade_mode: u32,
    light_model: u32,
    skylight_contribution: u32,

    // sky
    planet_radius: f32,
    atmosphere_radius: f32,
    sky_light_samples: u32,

    // SSR
    ssr_coarse_steps: u32,
    ssr_binary_search_steps: u32,
    ssr_linear_step_size: f32,
    ssr_thickness: f32,
    ssr_max_distance: f32,
    ssr_roughness_fade_start: f32,
    ssr_roughness_fade_end: f32,

    // SSGI
    ssgi_num_rays: u32,
    ssgi_num_steps: u32,
    ssgi_ray_step_size: f32,
    ssgi_thickness: f32,
    ssgi_blend_factor: f32,

    // shadows
    evsm_c: f32,
    pcf_radius: u32,
    pcf_min_scale: f32,
    pcf_max_scale: f32,
    pcf_max_distance: f32,

    // composite
    ssgi_intensity: f32,

    _padding: vec4<f32>,
};

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
@group(0) @binding(3) var lit_scene_texture: texture_2d<f32>;
@group(0) @binding(4) var gbuf_sampler: sampler;
@group(0) @binding(5) var scene_sampler: sampler; // Should be a linear sampler

@group(1) @binding(0) var<uniform> camera: CameraUniforms;

@group(2) @binding(0) var<uniform> constants: Constants;

//=============== CONSTANTS ===============//
// Using a stricter thickness check to improve reflection accuracy on top surfaces.
const COARSE_STEPS = 160u;      // Keep high for precision.
const LINEAR_STEP_SIZE = 0.07;  // A constant, linear step size
const THICKNESS = 0.1;          // Surface thickness for intersection

const BINARY_SEARCH_STEPS = 6u; // Refinement steps
const MAX_DISTANCE = 250.0; // Max reflection distance
const ROUGHNESS_FADE_START = 0.1;
const ROUGHNESS_FADE_END = 0.5;
const EPSILON = 0.0001;
const SELF_INTERSECTION_TOLERANCE = 0.05; // A small tolerance to prevent a reflection ray from immediately hitting the surface it came from.

//=============== UTILITY FUNCTIONS ===============//

fn rand(uv: vec2<f32>) -> f32 {
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
}

fn unproject_to_view_h(uv: vec2<f32>, depth: f32) -> vec4<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
    let clip = vec4<f32>(ndc, 1.0);
    return camera.inverse_projection_matrix * clip;
}

fn get_view_pos(uv: vec2<f32>) -> vec3<f32> {
    let depth = textureSample(depth_texture, gbuf_sampler, uv);
    let view_h = unproject_to_view_h(uv, depth);
    return view_h.xyz / (view_h.w + EPSILON);
}

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (abs(det) < 0.00001) {
        return mat3x3<f32>(); // Return identity
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

    // --- 1. G-Buffer Unpack ---
    let mra = textureSample(gbuf_mra, gbuf_sampler, screen_uv);
    let roughness = mra.g;
    if (roughness > constants.ssr_roughness_fade_end) {
        discard;
    }

    let origin_depth = textureSample(depth_texture, gbuf_sampler, screen_uv);
    if (origin_depth >= 1.0) { discard; } // Skybox
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

    // --- 2. Linear Ray Marching with Thickness Check ---
    var ray_pos_vs = origin_vs + N_vs * SELF_INTERSECTION_TOLERANCE;
    
    let dither_amount = rand(screen_uv);
    let step_size = constants.ssr_linear_step_size * (1.0 + dither_amount * 0.2);
    
    var hit_found = false;
    var last_ray_pos_vs = ray_pos_vs;

    for (var i = 0u; i < constants.ssr_coarse_steps; i = i + 1u) {
        last_ray_pos_vs = ray_pos_vs;
        ray_pos_vs += R_vs * step_size;

        let ray_pos_cs = camera.projection_matrix * vec4(ray_pos_vs, 1.0);
        if (ray_pos_cs.w <= EPSILON) { break; }
        
        let hit_uv = (ray_pos_cs.xy / ray_pos_cs.w) * vec2(0.5, -0.5) + 0.5;

        if (any(hit_uv < vec2(0.0)) || any(hit_uv > vec2(1.0)) || length(origin_vs - ray_pos_vs) > constants.ssr_max_distance) {
            break;
        }

        let scene_vs_at_hit = get_view_pos(hit_uv);

        // --- Intersection Test with Thickness ---
        let ray_depth = ray_pos_vs.z;
        let scene_depth = scene_vs_at_hit.z;
        let depth_difference = scene_depth - ray_depth;

        if (depth_difference > 0.0 && depth_difference < constants.ssr_thickness) {
            hit_found = true;
            break;
        }
    }

    if (!hit_found) {
        discard;
    }

    // --- 3. Binary Search Refinement ---
    var start_pos = last_ray_pos_vs;
    var end_pos = ray_pos_vs;

    for (var j = 0u; j < constants.ssr_binary_search_steps; j = j + 1u) {
        let mid_pos = (start_pos + end_pos) * 0.5;
        let mid_cs = camera.projection_matrix * vec4(mid_pos, 1.0);
        if (mid_cs.w <= EPSILON) { break; }
        let mid_uv = (mid_cs.xy / mid_cs.w) * vec2(0.5, -0.5) + 0.5;
        
        let scene_vs_at_mid = get_view_pos(mid_uv);
        let depth_difference = scene_vs_at_mid.z - mid_pos.z;

        // Using the same thickness check as the coarse march for consistent, accurate refinement.
        if (depth_difference > 0.0 && depth_difference < constants.ssr_thickness) {
            end_pos = mid_pos;
        } else {
            start_pos = mid_pos;
        }
    }

    // --- 4. Final Color and Confidence ---
    let final_cs = camera.projection_matrix * vec4(end_pos, 1.0);
    if (final_cs.w <= EPSILON) { discard; }
    let refined_uv = (final_cs.xy / final_cs.w) * vec2(0.5, -0.5) + 0.5;

    let reflection_color = textureSample(lit_scene_texture, scene_sampler, refined_uv);
    
    let edge_fade = smoothstep(0.0, 0.1, min(min(refined_uv.x, 1.0 - refined_uv.x), min(refined_uv.y, 1.0 - refined_uv.y)));
    let roughness_fade = 1.0 - smoothstep(constants.ssr_roughness_fade_start, constants.ssr_roughness_fade_end, roughness);
    let confidence = edge_fade * roughness_fade;

    return vec4(reflection_color.rgb, confidence);
}
