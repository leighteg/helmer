const PI: f32 = 3.14159265359;

// RAW QUALITY
// These control the core workload. Lower values are faster but produce more noise.
const NUM_RAYS: i32 = 4;
const NUM_STEPS: i32 = 8;

// EFFECT RADIUS & PRECISION
const RAY_STEP_SIZE: f32 = 0.3;
const THICKNESS: f32 = 0.15; // How "thick" a surface is when checking for ray intersections.

// GHOSTING vs. NOISE
// A HIGHER blend factor reduces ghosting but shows more noise. (e.g., 0.25)
// A LOWER blend factor smooths noise better but causes more ghosting. (e.g., 0.05)
const BLEND_FACTOR: f32 = 0.2;

// --- STRUCTS & BINDINGS ---
struct Camera {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
};
@group(0) @binding(0) var t_normal: texture_2d<f32>;
@group(0) @binding(1) var t_albedo: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(0) @binding(3) var t_history: texture_2d<f32>;
@group(0) @binding(4) var s_gbuffer: sampler;
@group(0) @binding(5) var s_history: sampler;
@group(1) @binding(0) var<uniform> camera: Camera;

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

fn noise(p: vec2<f32>) -> vec2<f32> {
    return fract(sin(vec2(dot(p, vec2(12.9898, 78.233)), dot(p, vec2(54.33, 31.22)))) * 43758.5453);
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
    let screen_dims = vec2<f32>(textureDimensions(t_depth));
    let frag_coord = vec2<u32>(floor(in.uv * screen_dims));

    let world_normal = textureLoad(t_normal, frag_coord, 0).xyz;
    let depth = textureLoad(t_depth, frag_coord, 0);

    if (depth >= 1.0) {
        discard;
    }
    
    let view_proj = camera.projection_matrix * camera.view_matrix;
    let world_pos = world_from_depth(in.uv, depth);
    var indirect_light = vec3<f32>(0.0);
    let tangent_to_world = create_onb(world_normal);
    let temporal_noise_offset = noise(world_pos.xy + world_pos.z).x;

    // Ray marching loop
    for (var i = 0; i < NUM_RAYS; i += 1) {
        let seed = vec2<f32>(frag_coord) + temporal_noise_offset;
        let r = noise(seed + f32(i));
        let ray_dir_tangent = cosine_sample_hemisphere(r);
        let ray_dir_world = tangent_to_world * ray_dir_tangent;
        
        for (var j = 1; j <= NUM_STEPS; j += 1) {
            let step = ray_dir_world * RAY_STEP_SIZE * f32(j);
            let sample_pos_world = world_pos + step;
            var sample_pos_clip = view_proj * vec4(sample_pos_world, 1.0);
            sample_pos_clip /= sample_pos_clip.w;
            let sample_uv = sample_pos_clip.xy * vec2(0.5, -0.5) + 0.5;

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                break;
            }

            let depth_at_sample = textureSample(t_depth, s_history, sample_uv);
            let surface_pos_at_sample = world_from_depth(sample_uv, depth_at_sample);
            let dist_to_ray = distance(camera.view_position, sample_pos_world);
            let dist_to_surface = distance(camera.view_position, surface_pos_at_sample);

            if (dist_to_ray > dist_to_surface && dist_to_ray < dist_to_surface + THICKNESS) {
                let hit_light = textureSample(t_history, s_history, sample_uv).rgb;
                indirect_light += hit_light;
                break;
            }
        }
    }

    let final_light = indirect_light / f32(NUM_RAYS);
    
    // Temporal blending
    var prev_clip = view_proj * vec4(world_pos, 1.0);
    prev_clip /= prev_clip.w;
    let prev_uv = saturate(prev_clip.xy * vec2(0.5, -0.5) + 0.5);
    let prev_result = textureSample(t_history, s_history, prev_uv).rgb;
    let blended_light = mix(prev_result, final_light, BLEND_FACTOR); 
    
    return vec4<f32>(blended_light, 1.0);
}