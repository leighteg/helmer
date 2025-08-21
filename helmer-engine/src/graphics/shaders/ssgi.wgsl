const PI: f32 = 3.14159265359;

// --- TUNING PARAMETERS ---
const NUM_RAYS: i32 = 6;
const NUM_STEPS: i32 = 12;
const RAY_STEP_SIZE: f32 = 0.6;
const THICKNESS: f32 = 0.4; 
const BLEND_FACTOR: f32 = 0.15;
const EPSILON: f32 = 0.001;

// --- STRUCTS & BINDINGS ---
struct Camera {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
    prev_view_proj: mat4x4<f32>,
    frame_index: u32,
};
@group(0) @binding(0) var t_normal: texture_2d<f32>;
@group(0) @binding(1) var t_albedo: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(0) @binding(3) var t_history: texture_2d<f32>;
@group(0) @binding(4) var t_direct_lighting : texture_2d<f32>;
@group(0) @binding(5) var s_gbuffer: sampler;
@group(0) @binding(6) var s_history: sampler;
@group(1) @binding(0) var<uniform> camera: Camera;
@group(2) @binding(0) var t_blue_noise: texture_2d<f32>;
@group(2) @binding(1) var s_blue_noise: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// --- HELPER FUNCTIONS ---
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

fn world_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world = camera.inverse_view_projection_matrix * ndc;
    return world.xyz / world.w;
}

fn view_pos_from_uv_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view_h = camera.inverse_projection_matrix * clip;
    return view_h.xyz / (view_h.w + EPSILON);
}

fn create_onb(normal: vec3<f32>) -> mat3x3<f32> {
    let up_vec = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.x) > 0.999);
    let tangent = normalize(cross(up_vec, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

fn cosine_sample_hemisphere(r: vec2<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * r.x;
    let r_sqrt = sqrt(r.y);
    let x = r_sqrt * cos(phi);
    let y = r_sqrt * sin(phi);
    let z = sqrt(max(0.0, 1.0 - r.y));
    return vec3<f32>(x, y, z);
}

// --- MAIN SHADER ---
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let frag_coord = vec2<u32>(floor(in.uv * vec2<f32>(textureDimensions(t_depth))));
    let depth = textureLoad(t_depth, frag_coord, 0);
    if depth >= 1.0 { discard; }

    let origin_vs = view_pos_from_uv_depth(in.uv, depth);

    let world_normal = normalize(textureLoad(t_normal, frag_coord, 0).xyz * 2.0 - 1.0);
    let normal_matrix = mat3x3<f32>(camera.view_matrix[0].xyz, camera.view_matrix[1].xyz, camera.view_matrix[2].xyz);
    let normal_vs = normalize(normal_matrix * world_normal);

    let tangent_to_view = create_onb(normal_vs);
    var indirect_light = vec3<f32>(0.0);

    let blue_noise_dims = vec2<f32>(textureDimensions(t_blue_noise));
    let blue_noise_uv = in.uv * vec2<f32>(textureDimensions(t_direct_lighting)) / blue_noise_dims;
    let blue_noise = textureSample(t_blue_noise, s_blue_noise, blue_noise_uv).xy;

    for (var i = 0; i < NUM_RAYS; i += 1) {
        let r = fract(blue_noise + vec2<f32>(f32(i) * 0.137, f32(camera.frame_index) * 0.379));
        let ray_dir_tangent = cosine_sample_hemisphere(r);
        let ray_dir_vs = tangent_to_view * ray_dir_tangent;

        for (var j = 1; j <= NUM_STEPS; j += 1) {
            let sample_pos_vs = origin_vs + ray_dir_vs * RAY_STEP_SIZE * f32(j);

            var sample_pos_clip = camera.projection_matrix * vec4(sample_pos_vs, 1.0);
            if sample_pos_clip.w <= EPSILON { break; }
            sample_pos_clip /= sample_pos_clip.w;
            let sample_uv = sample_pos_clip.xy * vec2(0.5, -0.5) + 0.5;

            if any(sample_uv < vec2(0.0)) || any(sample_uv > vec2(1.0)) { break; }

            let non_linear_depth_at_sample = textureSample(t_depth, s_history, sample_uv);
            let scene_vs_at_sample = view_pos_from_uv_depth(sample_uv, non_linear_depth_at_sample);

            let ray_depth = -sample_pos_vs.z;
            let scene_depth = -scene_vs_at_sample.z;

            if ray_depth > scene_depth && ray_depth < scene_depth + THICKNESS {
                let hit_light = textureSample(t_direct_lighting, s_history, sample_uv).rgb;
                let hit_albedo = textureSample(t_albedo, s_history, sample_uv).rgb;
                indirect_light += hit_light * hit_albedo;
                break;
            }
        }
    }

    let ssgi_result = indirect_light / f32(NUM_RAYS);
    
    // --- Temporal Blending with History Rejection ---
    let world_pos = world_from_depth(in.uv, depth);
    var prev_clip = camera.prev_view_proj * vec4(world_pos, 1.0);
    prev_clip /= (prev_clip.w + EPSILON);
    let prev_uv = saturate(prev_clip.xy * vec2(0.5, -0.5) + 0.5);

    let prev_result = textureSample(t_history, s_history, prev_uv).rgb;

// --- History Rejection based on velocity and geometry ---
    let prev_depth = textureSample(t_depth, s_history, prev_uv);
    let prev_world_pos = world_from_depth(prev_uv, prev_depth);
    let world_dist = distance(world_pos, prev_world_pos);

// A smaller blend factor stabilizes the image faster, but causes more ghosting.
// We increase the blend factor (i.e., use more of the current frame's noisy result) 
// if the reprojected sample is likely invalid.
    let history_validity = 1.0 - saturate(world_dist / 0.75); // As distance increases, validity decreases
    let adaptive_blend_factor = mix(1.0, BLEND_FACTOR, history_validity);

// Clamp the history sample to the neighborhood of the current pixel to prevent ghosting
// from disoccluded bright spots. This is a form of color box filtering.
    var aabb_max = vec3<f32>(-1.0);
    var aabb_min = vec3<f32>(1.0);
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset_uv = in.uv + vec2<f32>(f32(x), f32(y)) / vec2<f32>(textureDimensions(t_direct_lighting));
            let neighbor_color = textureSample(t_direct_lighting, s_history, offset_uv).rgb;
            aabb_max = max(aabb_max, neighbor_color);
            aabb_min = min(aabb_min, neighbor_color);
        }
    }
    let clamped_prev_result = clamp(prev_result, aabb_min, aabb_max);

    let blended_light = mix(clamped_prev_result, ssgi_result, adaptive_blend_factor);

    return vec4<f32>(blended_light, 1.0);
}