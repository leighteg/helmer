//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const NUM_CASCADES: u32 = 4u;
const EVSM_C = 40.0;
const EPSILON: f32 = 0.00001;

//=============== STRUCTS ===============//
struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
}
struct SkyUniforms {
    sun_direction: vec3<f32>,
    _padding: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
};
struct LightData {
    position: vec3<f32>,
    light_type: u32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    _padding: f32,
}
struct CascadeData {
    light_view_proj: mat4x4<f32>,
    split_depth: vec4<f32>,
}

//=============== BINDINGS ===============//
@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuf_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuf_mra: texture_2d<f32>;
@group(0) @binding(4) var gbuf_emission: texture_2d<f32>;
@group(0) @binding(5) var gbuf_sampler: sampler;

@group(1) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(1) var<storage, read> lights_buffer: array<LightData>;
@group(1) @binding(2) var shadow_map: texture_2d_array<f32>;
@group(1) @binding(3) var shadow_sampler: sampler;
@group(1) @binding(4) var<uniform> shadow_uniforms: array<CascadeData, NUM_CASCADES>;
@group(1) @binding(5) var<uniform> sky: SkyUniforms; // ADDED sky uniforms

//=============== UTILITY & PBR FUNCTIONS ===============//
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len < EPSILON { return vec3<f32>(0.0); }
    return v / len;
}
fn distribution_ggx(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH2 = NdotH * NdotH;
    let denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}
fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}
fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
fn chebyshev_inequality(depth: f32, moments: vec2<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    // bias to push the shadow onto the surface to fix acne and detachment.
    let min_bias = 0.0005; // smaller values now that EVSM is handling the heavy lifting
    let max_bias = 0.005;
    let NdotL = max(dot(N, L), 0.0);
    let bias = mix(max_bias, min_bias, NdotL);

    var current_depth = depth - bias;

    // Warp the depth value
    current_depth = exp(EVSM_C * (current_depth - 1.0));

    // Chebyshev test
    if current_depth <= moments.x {
        return 1.0;
    }

    var variance = moments.y - (moments.x * moments.x);
    variance = max(variance, 0.0);

    let d = current_depth - moments.x;
    let p_max = variance / (variance + d * d);

    return smoothstep(0.2, 1.0, p_max);
}

fn calculate_shadow_factor(world_pos: vec3<f32>, view_z: f32, N: vec3<f32>, L: vec3<f32>) -> f32 {
    var cascade_index = i32(NUM_CASCADES - 1);
    for (var i = 0i; i < i32(NUM_CASCADES); i = i + 1i) {
        if view_z > shadow_uniforms[i].split_depth.x {
            cascade_index = i;
            break;
        }
    }
    let cascade = shadow_uniforms[cascade_index];
    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);
    if shadow_pos_clip.w < EPSILON { return 1.0; }
    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);
    if any(shadow_uv < vec2(0.0)) || any(shadow_uv > vec2(1.0)) || shadow_coord.z < 0.0 || shadow_coord.z > 1.0 {
        return 1.0;
    }
    let moments = textureSample(shadow_map, shadow_sampler, shadow_uv, u32(cascade_index)).rg;
    return chebyshev_inequality(shadow_coord.z, moments, N, L);
}

//=============== SKY CALCULATION FUNCTIONS ===============//
const planet_radius = 6371e3;
const atmosphere_radius = 6471e3;
const rayleigh_scattering_coeff = vec3(5.5e-6, 13.0e-6, 22.4e-6);
const rayleigh_scale_height = 8e3;
const mie_scattering_coeff = 21e-6;
const mie_scale_height = 1.2e3;
const mie_preferred_scattering_dir = 0.76;

fn ray_sphere_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_radius: f32) -> vec2<f32> {
    let b = dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;
    var delta = b * b - c;
    if delta < 0.0 { return vec2(-1.0); }
    delta = sqrt(delta);
    return vec2(-b - delta, -b + delta);
}

fn get_transmittance_to_sun(sample_pos: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let dist_to_atmosphere = ray_sphere_intersect(sample_pos, sun_dir, atmosphere_radius).y;
    let num_light_samples = 8;
    let light_step_size = dist_to_atmosphere / f32(num_light_samples);
    var optical_depth = vec3(0.0);
    for (var j = 0; j < num_light_samples; j = j + 1) {
        let light_pos = sample_pos + sun_dir * (f32(j) + 0.5) * light_step_size;
        let height = length(light_pos) - planet_radius;
        if height < 0.0 { return vec3(0.0); }
        let rayleigh_density = exp(-height / rayleigh_scale_height);
        let mie_density = exp(-height / mie_scale_height);
        optical_depth += (rayleigh_scattering_coeff * rayleigh_density + mie_scattering_coeff * mie_density) * light_step_size;
    }
    return exp(-optical_depth);
}

fn get_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let camera_pos = vec3(0.0, planet_radius + 1.0, 0.0);
    let dist_to_atmosphere = ray_sphere_intersect(camera_pos, view_dir, atmosphere_radius).y;
    let num_samples = 8; // Fewer samples for performance in lighting pass
    let step_size = dist_to_atmosphere / f32(num_samples);
    var transmittance_to_camera = vec3(1.0);
    var scattered_light = vec3(0.0);
    for (var i = 0; i < num_samples; i = i + 1) {
        let sample_pos = camera_pos + view_dir * (f32(i) + 0.5) * step_size;
        let height = length(sample_pos) - planet_radius;
        if height < 0.0 { break; }
        let rayleigh_density = exp(-height / rayleigh_scale_height);
        let mie_density = exp(-height / mie_scale_height);
        let optical_depth_step = (rayleigh_scattering_coeff * rayleigh_density + mie_scattering_coeff * mie_density) * step_size;
        transmittance_to_camera *= exp(-optical_depth_step);
        let transmittance_to_sun = get_transmittance_to_sun(sample_pos, sun_dir);
        let cos_theta = dot(view_dir, sun_dir);
        let rayleigh_phase = 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
        let g = mie_preferred_scattering_dir;
        let mie_phase = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + cos_theta * cos_theta)) / ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
        let in_scattered_r = rayleigh_scattering_coeff * rayleigh_density * rayleigh_phase;
        let in_scattered_m = mie_scattering_coeff * mie_density * mie_phase;
        scattered_light += (in_scattered_r + in_scattered_m) * transmittance_to_sun * transmittance_to_camera * step_size;
    }
    return scattered_light * sky.sun_color * sky.sun_intensity;
}

//=============== SHADERS ===============//
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct LightingOutput {
    @location(0) full_pbr: vec4<f32>,
    @location(1) diffuse_only: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> LightingOutput {
    let screen_uv = frag_coord.xy / vec2<f32>(textureDimensions(gbuf_normal));
    let depth = textureSample(depth_texture, gbuf_sampler, screen_uv);

    if depth <= 0.0 { 
        discard;
    }

    let ndc = vec4<f32>(screen_uv.x * 2.0 - 1.0, (1.0 - screen_uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_pos_h = camera.inverse_view_projection_matrix * ndc;
    let world_position = world_pos_h.xyz / world_pos_h.w;

    let packed_normal = textureSample(gbuf_normal, gbuf_sampler, screen_uv).xyz;
    let N = safe_normalize(packed_normal * 2.0 - 1.0);

    let albedo = textureSample(gbuf_albedo, gbuf_sampler, screen_uv).rgb;
    let mra_sample = textureSample(gbuf_mra, gbuf_sampler, screen_uv);
    let metallic = mra_sample.r;
    let roughness = max(mra_sample.g, MIN_ROUGHNESS);
    var ao = mra_sample.b;

    ao = mix(ao, 1.0, 1.0 - smoothstep(0.0, 0.1, ao));

    let sun_height_factor = max(sky.sun_direction.y, 0.0); 
    let sun_fade = pow(sun_height_factor, 1.5); // adjust exponent for faster/slower fade

    let V = safe_normalize(camera.view_position - world_position);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    var direct_lighting = vec3<f32>(0.0);
    var diffuse_lighting = vec3<f32>(0.0);

    let view_pos = camera.view_matrix * vec4<f32>(world_position, 1.0);

    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;
        var shadow_multiplier = 1.0;

        if light.light_type == 0u { // Directional
            L = safe_normalize(-light.direction);
            radiance = light.color * light.intensity * sun_fade;
            shadow_multiplier = calculate_shadow_factor(world_position, view_pos.z, N, L);
        } else { // Point
            let to_light_vector = light.position - world_position;
            let dist_sq = dot(to_light_vector, to_light_vector);
            if dist_sq < EPSILON { continue; }
            L = to_light_vector / sqrt(dist_sq);
            let attenuation = 1.0 / (dist_sq + 1.0);
            radiance = light.color * light.intensity * attenuation;
        }

        let NdotL = max(dot(N, L), 0.0);
        if NdotL > 0.0 {
            let H = safe_normalize(V + L);
            let NdotH = max(dot(N, H), 0.0);
            let NdotV = max(dot(N, V), 0.0);

            let NDF = distribution_ggx(NdotH, roughness);
            let G = geometry_smith(N, V, L, roughness);
            let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

            let specular_numerator = NDF * G * F;
            let specular_denominator = 4.0 * NdotV * NdotL + EPSILON;
            let specular_brdf = specular_numerator / specular_denominator;

            let diffuse_brdf = albedo / PI;

            let kS = F;
            let kD = vec3<f32>(1.0) - kS;

            let final_radiance = radiance * NdotL * shadow_multiplier;

            let current_pbr = (kD * (1.0 - metallic) * diffuse_brdf + specular_brdf) * final_radiance;
            let current_diffuse_only = (kD * (1.0 - metallic) * diffuse_brdf) * final_radiance;

            direct_lighting += current_pbr;
            diffuse_lighting += current_diffuse_only;
        }
    }

    // --- SKY AMBIENT LIGHTING ---
    // Calculate sky color based on the surface normal to get ambient diffuse light.
    let sky_ambient_color = get_sky_color(N, normalize(sky.sun_direction));
    let F_ambient = fresnel_schlick(max(dot(N, V), 0.0), F0);
    let kS_ambient = F_ambient;
    let kD_ambient = (vec3<f32>(1.0) - kS_ambient) * (1.0 - metallic);
    let ambient_contribution = kD_ambient * albedo * sky_ambient_color;
    direct_lighting += ambient_contribution;
    diffuse_lighting += ambient_contribution;

    var out: LightingOutput;
    out.full_pbr = vec4<f32>(direct_lighting * ao, 1.0);
    out.diffuse_only = vec4<f32>(diffuse_lighting * ao, 1.0);
    return out;
}