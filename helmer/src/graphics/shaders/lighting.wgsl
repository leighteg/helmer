struct Constants {
    // general
    mip_bias: f32,

    // lighting
    shade_mode: u32,
    light_model: u32,
    skylight_contribution: u32,

    // sky
    planet_radius: f32,
    atmosphere_radius: f32,
    sky_light_samples: u32,
    _pad0: u32,

    // SSR
    ssr_coarse_steps: u32,
    ssr_binary_search_steps: u32,
    ssr_linear_step_size: f32,
    ssr_thickness: f32,
    ssr_max_distance: f32,
    ssr_roughness_fade_start: f32,
    ssr_roughness_fade_end: f32,
    _pad1: u32,

    // SSGI
    ssgi_num_rays: u32,
    ssgi_num_steps: u32,
    ssgi_ray_step_size: f32,
    ssgi_thickness: f32,
    ssgi_blend_factor: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,

    // shadows
    evsm_c: f32,
    pcf_radius: u32,
    pcf_min_scale: f32,
    pcf_max_scale: f32,
    pcf_max_distance: f32,
    ssgi_intensity: f32,
    _final_padding: vec2<f32>,
};

//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const MAX_SHADOW_CASCADES: u32 = 4u;
const EPSILON: f32 = 0.00001;

const POISSON_DISK_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2(-0.94201624, -0.39906216),
    vec2(0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845),
    vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554),
    vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507),
    vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790)
);

//=============== STRUCTS ===============//
struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
    _pad_light: vec4<u32>,
    prev_view_proj: mat4x4<f32>,
    frame_index: u32,
    _padding: vec3<u32>,
    _pad_end: vec4<u32>,
}
struct SkyUniforms {
    sun_direction: vec3<f32>,
    _padding: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    ground_albedo: vec3<f32>,
    ground_brightness: f32,
    night_ambient_color: vec3<f32>,
    sun_angular_radius_cos: f32,
};
struct AtmosphereParams {
    planet_radius: f32,
    atmosphere_radius: f32,
    sun_intensity: f32,
    _padding: f32,
    sun_direction: vec3<f32>,
    _padding2: f32,
    rayleigh_scattering_coeff: vec3<f32>,
    rayleigh_scale_height: f32,
    mie_scattering_coeff: f32,
    mie_absorption_coeff: f32,
    mie_scale_height: f32,
    mie_preferred_scattering_dir: f32,
    ozone_absorption_coeff: vec3<f32>,
    ozone_center_height: f32,
    ozone_falloff: f32,
    _pad_atmo0: vec3<f32>,
    ground_albedo: vec3<f32>,
    ground_brightness: f32,
    night_ambient_color: vec3<f32>,
    _pad_atmo1: f32,
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
struct ShadowUniforms {
    cascade_count: u32,
    _pad0: vec3<u32>,
    cascades: array<CascadeData, MAX_SHADOW_CASCADES>,
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
@group(1) @binding(4) var<uniform> shadow_uniforms: ShadowUniforms;
@group(1) @binding(5) var<uniform> sky: SkyUniforms;

@group(2) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(1) var scattering_lut: texture_3d<f32>;
@group(2) @binding(2) var irradiance_lut: texture_2d<f32>;
@group(2) @binding(3) var atmosphere_sampler: sampler;
@group(2) @binding(4) var<uniform> atmosphere: AtmosphereParams;

@group(3) @binding(0) var<uniform> constants: Constants;

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
    var current_depth = depth;

    // Warp the depth value
    current_depth = exp(constants.evsm_c * (current_depth - 1.0));

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

fn calculate_shadow_factor(
    world_pos: vec3<f32>,
    view_z: f32,
    N: vec3<f32>,
    L: vec3<f32>
) -> f32 {
    // 1. Determine cascade
    let cascade_count = max(
        1u,
        min(shadow_uniforms.cascade_count, MAX_SHADOW_CASCADES)
    );
    var cascade_index = i32(cascade_count - 1u);
    for (var i = 0u; i < cascade_count; i = i + 1u) {
        if view_z > shadow_uniforms.cascades[i].split_depth.x {
            cascade_index = i32(i);
            break;
        }
    }
    let cascade = shadow_uniforms.cascades[cascade_index];

    // 2. Project to shadow space
    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);
    if shadow_pos_clip.w < EPSILON {
        return 1.0;
    }

    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);

    // 3. Bounds check
    if any(shadow_uv < vec2(0.0)) || any(shadow_uv > vec2(1.0)) || shadow_coord.z < 0.0 || shadow_coord.z > 1.0 {
        return 1.0;
    }

    // 4. Calculate dynamic filter size
    let base_radius = f32(constants.pcf_radius);
    let dist_factor = clamp(view_z / constants.pcf_max_distance, 0.0, 1.0);
    let scale = mix(constants.pcf_min_scale, constants.pcf_max_scale, dist_factor);
    let filter_radius = base_radius * scale;

    // 5. Choose sample count based on quality settings
    // Use fewer samples for far cascades or distant geometry
    var sample_count = 16;
    if cascade_index > 1 || dist_factor > 0.7 {
        sample_count = 8; // Reduce samples for far/distant shadows
    }
    
    // 6. Sample setup - compute once
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map, 0u));
    let texel_size = filter_radius / shadow_map_size;

    var total_moments = vec2<f32>(0.0);
    
    // 7. Poisson disk sampling
    if sample_count == 16 {
        // 16-tap sampling
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[0] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[1] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[2] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[3] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[4] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[5] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[6] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[7] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[8] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[9] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[10] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[11] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[12] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[13] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[14] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[15] * texel_size, u32(cascade_index)).rg;
        total_moments *= (1.0 / 16.0);
    } else {
        // 8-tap sampling - every other sample
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[0] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[2] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[4] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[6] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[8] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[10] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[12] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[14] * texel_size, u32(cascade_index)).rg;
        total_moments *= (1.0 / 8.0);
    }

    // 8. EVSM calculation
    return chebyshev_inequality(shadow_coord.z, total_moments, N, L);
}

// --- SKY SAMPLING FUNCTIONS ---
fn ray_sphere_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_radius: f32) -> vec2<f32> {
    let b = dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;
    var delta = b * b - c;
    if delta < 0.0 { return vec2<f32>(-1.0); }
    delta = sqrt(delta);
    return vec2<f32>(-b - delta, -b + delta);
}

fn altitude_mu_to_uv(altitude: f32, mu: f32, radius: f32, atmos_radius: f32) -> vec2<f32> {
    let alt_range = atmos_radius - radius;
    let u = (altitude - radius) / alt_range;
    let v = (mu + 1.0) * 0.5;
    return saturate(vec2<f32>(u, v));
}

fn scattering_lut_coords(altitude: f32, mu_s: f32, mu_v_s: f32, radius: f32, atmos_radius: f32) -> vec3<f32> {
    let alt_range = atmos_radius - radius;
    let u = (altitude - radius) / alt_range;
    let v = (mu_s + 1.0) * 0.5;
    let w = (mu_v_s + 1.0) * 0.5;
    return saturate(vec3<f32>(u, v, w));
}

fn get_transmittance(world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let altitude = atmosphere.planet_radius; // Flat approximation, ground level
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let mu = dot(view_dir, up);
    let uv = altitude_mu_to_uv(altitude, mu, atmosphere.planet_radius, atmosphere.atmosphere_radius);
    return textureSample(transmittance_lut, atmosphere_sampler, uv).rgb;
}

fn get_irradiance(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let altitude = atmosphere.planet_radius; // Flat approximation, ground level
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let mu_s = dot(atmosphere.sun_direction, up);
    let uv = altitude_mu_to_uv(altitude, mu_s, atmosphere.planet_radius, atmosphere.atmosphere_radius);
    let sky_irradiance = textureSample(irradiance_lut, atmosphere_sampler, uv).rgb;
    // Removed dot(normal, up) to apply sky light uniformly and avoid pitch black areas
    return sky_irradiance;
}

fn get_scattering_color(world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let altitude = atmosphere.planet_radius; // Flat approximation, ground level
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let mu_s = dot(atmosphere.sun_direction, up);
    let mu_v = dot(view_dir, up);
    let mu_v_s = dot(view_dir, atmosphere.sun_direction);

    let coords = scattering_lut_coords(altitude, mu_s, mu_v_s, atmosphere.planet_radius, atmosphere.atmosphere_radius);
    let scatter = textureSample(scattering_lut, atmosphere_sampler, coords).rgb;

    return scatter;
}
//=============== END SKY SAMPLING ===============//

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
    let screen_uv = frag_coord.xy / vec2<f32>(textureDimensions(gbuf_normal, 0));
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
    let emission = textureSample(gbuf_emission, gbuf_sampler, screen_uv).rgb;
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

    let is_lit = constants.shade_mode == 0u;

    if constants.shade_mode != 1u {
        for (var i = 0u; i < camera.light_count; i = i + 1u) {
            let light = lights_buffer[i];
            var L: vec3<f32>;
            var radiance: vec3<f32>;
            var shadow_multiplier = 1.0;

            if light.light_type == 0u { // Directional
                L = safe_normalize(-light.direction);
                radiance = light.color * light.intensity * sun_fade;

                let NdotL = max(dot(N, L), 0.0);
                let bias_amount = 0.001 + 0.005 * (1.0 - NdotL);
                let biased_world_position = world_position + N * bias_amount;
                shadow_multiplier = calculate_shadow_factor(biased_world_position, view_pos.z, N, L);
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

                let is_stylized = constants.light_model == 1u;
                let final_radiance = select(radiance * NdotL * shadow_multiplier, radiance * shadow_multiplier, is_stylized);

                let current_pbr = select((kD * (1.0 - metallic) * PI + specular_brdf), (kD * (1.0 - metallic) * diffuse_brdf + specular_brdf), is_lit) * final_radiance;
                let current_diffuse_only = (kD * (1.0 - metallic) * diffuse_brdf) * final_radiance;

                direct_lighting += current_pbr;
                diffuse_lighting += current_diffuse_only;
            }
        }

        // --- SKY AMBIENT LIGHTING ---
        if constants.skylight_contribution == 1u { // FULL (PBR)
            let sky_visibility = ao;
            let up = vec3<f32>(0.0, 1.0, 0.0);

            // Sample precomputed irradiance (incident light)
            let diffuse_sky_color = get_irradiance(world_position, N);
    
            // Sample precomputed scattering for reflections
            let R = reflect(-V, N);
            let reflection_sky_color = get_irradiance(world_position, R) * 0.5 + get_transmittance(world_position, R) * 0.5;

            let F_ambient = fresnel_schlick(max(dot(N, V), 0.0), F0);
            let kS_ambient = F_ambient * (1.0 - roughness * 0.7);
            let kD_ambient = (vec3<f32>(1.0) - kS_ambient) * (1.0 - metallic);

            // Energy-conserving ambient (added / PI for correct diffuse energy, fixes grey/washed-out look)
            let diffuse_contribution = kD_ambient * albedo * diffuse_sky_color / PI;
            let specular_contribution = kS_ambient * reflection_sky_color;

            let total_contribution = diffuse_contribution + specular_contribution;

            let direct_diffuse_contribution = select(
                kD_ambient * diffuse_sky_color / PI,
                diffuse_contribution,
                is_lit
            );

            direct_lighting += (direct_diffuse_contribution + specular_contribution) * sky_visibility;
            diffuse_lighting += diffuse_contribution * sky_visibility;

        } else if constants.skylight_contribution == 2u { // STYLIZED FULL
            var sky_visibility = ao;
            sky_visibility = mix(1.0, sky_visibility, 0.5);

            let up = vec3<f32>(0.0, 1.0, 0.0);

            // Slightly bias normals upward for a dreamy, painterly look
            let biased_normal = normalize(mix(N, up, 0.5));
            let biased_reflection = reflect(-V, biased_normal);

            // --- Blend between sun scattering and general sky irradiance ---
            let sun_scatter = get_scattering_color(world_position, atmosphere.sun_direction);
            let sky_reflection = get_irradiance(world_position, biased_reflection);

            // Blend — 0.5 gives balanced, stylized but not sun-locked look
            let sky_specular = mix(sun_scatter, sky_reflection, 0.5);
            let sky_diffuse = get_irradiance(world_position, biased_normal) / PI;

            // Fresnel & energy conservation
            var F_ambient = fresnel_schlick(max(dot(N, V), 0.0), F0);
            F_ambient = mix(F_ambient, vec3<f32>(0.04), 0.5);

            let kS = F_ambient * (1.0 - roughness * 0.7);
            let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

            let diffuse_contribution = kD * albedo * sky_diffuse;
            let specular_contribution = kS * sky_specular;

            let total_contribution = diffuse_contribution + specular_contribution;

            let direct_diffuse_contribution = select(
                kD * sky_diffuse,
                diffuse_contribution,
                is_lit
            );

            direct_lighting += (direct_diffuse_contribution + specular_contribution) * sky_visibility;
            diffuse_lighting += diffuse_contribution * sky_visibility;
        } else if constants.skylight_contribution == 3u { // SIMPLE
            var sky_visibility = mix(1.0, ao, 0.5);
            let up = vec3<f32>(0.0, 1.0, 0.0);

            let flat_sky_color = get_irradiance(world_position, up) / PI; // Sample irradiance straight up

            let flat_kD = vec3(1.0) - metallic;

            let diffuse_contribution = flat_kD * albedo * flat_sky_color;

            let total_contribution = diffuse_contribution * sky_visibility;

            let direct_diffuse_contribution = select(
                flat_kD * flat_sky_color,
                diffuse_contribution,
                is_lit
            );

            direct_lighting += direct_diffuse_contribution * sky_visibility;
            diffuse_lighting += diffuse_contribution * sky_visibility;
        }
    }

    var out: LightingOutput;

    if constants.shade_mode == 1u { // UNLIT
        out.full_pbr = vec4<f32>(albedo, 1.0);
        out.diffuse_only = vec4(albedo + emission, 1.0);
    } else { // LIT
        out.full_pbr = vec4<f32>(direct_lighting * ao, 1.0);
        out.diffuse_only = vec4(diffuse_lighting * ao + emission, 1.0);
    }

    return out;
}
