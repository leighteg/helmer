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

struct SkyUniforms {
    sun_direction: vec3<f32>,
    _padding: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
};

struct AtmosphereParams {
    planet_radius: f32,
    atmosphere_radius: f32,
    sun_intensity: f32,
    _padding: f32,
    sun_direction: vec3<f32>,
    _padding2: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> sky: SkyUniforms;
@group(0) @binding(2) var scene_sampler: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;

@group(1) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(1) var scattering_lut: texture_3d<f32>;
@group(1) @binding(2) var irradiance_lut: texture_2d<f32>;
@group(1) @binding(3) var atmosphere_sampler: sampler;
@group(1) @binding(4) var<uniform> atmosphere: AtmosphereParams;

@group(2) @binding(0) var<uniform> constants: Constants;

const PI = 3.14159265;

// --- Atmospheric Scattering ---
const rayleigh_scattering_coeff: vec3<f32> = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const rayleigh_scale_height: f32 = 8e3;

const mie_scattering_coeff: f32 = 21e-6;
const mie_absorption_coeff: f32 = 4.4e-6;
const mie_extinction_coeff: f32 = mie_scattering_coeff + mie_absorption_coeff;
const mie_scale_height: f32 = 1.2e3;
const mie_preferred_scattering_dir: f32 = 0.758;

// --- Ozone ---
const ozone_absorption_coeff: vec3<f32> = vec3(0.65e-6, 1.881e-6, 0.085e-6);
const ozone_center_height: f32 = 25e3;
const ozone_falloff: f32 = 15e3;

// --- Ground ---
const ground_albedo: vec3<f32> = vec3(0.05);

// --- Sun Disk ---
const SUN_ANGULAR_RADIUS_COS: f32 = 0.9998;

// --- Night Sky ---
const night_ambient_color: vec3<f32> = vec3(0.0002, 0.0004, 0.0008);

// =============== HELPER FUNCTIONS ===============
fn ray_sphere_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_radius: f32) -> vec2<f32> {
    let b = dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;
    var delta = b * b - c;
    if delta < 0.0 { return vec2<f32>(-1.0); }
    delta = sqrt(delta);
    return vec2<f32>(-b - delta, -b + delta);
}

fn altitude_mu_to_uv(altitude: f32, mu: f32, planet_radius: f32, atmosphere_radius: f32) -> vec2<f32> {
    let alt_range = atmosphere_radius - planet_radius;
    let u = (altitude - planet_radius) / alt_range;
    let v = (mu + 1.0) * 0.5;
    return saturate(vec2<f32>(u, v));
}

fn scattering_lut_coords(altitude: f32, mu_s: f32, mu_v: f32, planet_radius: f32, atmosphere_radius: f32) -> vec3<f32> {
    let alt_range = atmosphere_radius - planet_radius;
    let u = (altitude - planet_radius) / alt_range;
    let v = (mu_s + 1.0) * 0.5;
    let w = (mu_v + 1.0) * 0.5;
    return saturate(vec3<f32>(u, v, w));
}

fn ozone_density(height: f32) -> f32 {
    return max(0.0, 1.0 - abs(height - ozone_center_height) / ozone_falloff);
}

fn rayleigh_phase_function(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn mie_phase_function(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let k = 3.0 / (8.0 * PI);
    let num = (1.0 - g2) * (1.0 + cos_theta * cos_theta);
    let denom = (2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return k * num / denom;
}

// =============== TRANSMITTANCE TO SUN (Raymarched) ===============
fn get_transmittance_to_sun(sample_pos: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let intersect = ray_sphere_intersect(sample_pos, sun_dir, atmosphere.atmosphere_radius);
    if intersect.y <= 0.0 { return vec3<f32>(0.0); }

    let ray_len = intersect.y;
    let samples = 16;
    let step = ray_len / f32(samples);
    var od = vec3<f32>(0.0);

    for (var i = 0; i < samples; i = i + 1) {
        let p = sample_pos + sun_dir * (f32(i) + 0.5) * step;
        let h = length(p) - atmosphere.planet_radius;
        if h < 0.0 { return vec3<f32>(0.0); }

        let rd = exp(-h / rayleigh_scale_height);
        let md = exp(-h / mie_scale_height);
        let odens = ozone_density(h);

        od += (rayleigh_scattering_coeff * rd + vec3<f32>(mie_extinction_coeff) * md + ozone_absorption_coeff * odens) * step;
    }
    return exp(-od);
}

// =============== LUT SAMPLING ===============
fn get_transmittance(world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let altitude = length(world_pos);
    let up = world_pos / altitude;
    let mu = dot(view_dir, up);
    let uv = altitude_mu_to_uv(altitude, mu, atmosphere.planet_radius, atmosphere.atmosphere_radius);
    return textureSample(transmittance_lut, atmosphere_sampler, uv).rgb;
}

fn get_scattering_color(world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let altitude = length(world_pos);
    let up = world_pos / altitude;
    let mu_s = dot(atmosphere.sun_direction, up);
    let mu_v = dot(view_dir, up);

    let coords = scattering_lut_coords(altitude, mu_s, mu_v, atmosphere.planet_radius, atmosphere.atmosphere_radius);

    var scatter = textureSample(scattering_lut, atmosphere_sampler, coords).rgb;

    return scatter;
}

// =============== RAYMARCH FOR GROUND (creates atmospheric bands) ===============
struct RaymarchResult {
    scattered_light: vec3<f32>,
    transmittance: vec3<f32>,
};

fn raymarch_to_ground(camera_pos: vec3<f32>, view_dir: vec3<f32>, ray_len: f32, sun_dir: vec3<f32>) -> RaymarchResult {
    let num_samples = i32(constants.sky_light_samples);
    let step_size = ray_len / f32(num_samples);

    var transmittance_to_camera = vec3<f32>(1.0);
    var scattered_light = vec3<f32>(0.0);

    let cos_theta = dot(view_dir, sun_dir);
    let rayleigh_phase = rayleigh_phase_function(cos_theta);
    let mie_phase = mie_phase_function(cos_theta, mie_preferred_scattering_dir);

    for (var i = 0; i < num_samples; i = i + 1) {
        let sample_pos = camera_pos + view_dir * (f32(i) + 0.5) * step_size;
        let height = length(sample_pos) - atmosphere.planet_radius;

        if height < 0.0 { break; }

        let rayleigh_density = exp(-height / rayleigh_scale_height);
        let mie_density = exp(-height / mie_scale_height);
        let ozone_dens = ozone_density(height);

        let optical_depth_step = (rayleigh_scattering_coeff * rayleigh_density + vec3<f32>(mie_extinction_coeff) * mie_density + ozone_absorption_coeff * ozone_dens) * step_size;
        transmittance_to_camera *= exp(-optical_depth_step);

        let transmittance_to_sun = get_transmittance_to_sun(sample_pos, sun_dir);

        let in_scattered_r = rayleigh_scattering_coeff * rayleigh_density * rayleigh_phase;
        let in_scattered_m = mie_scattering_coeff * mie_density * mie_phase;

        scattered_light += (in_scattered_r + in_scattered_m) * transmittance_to_sun * transmittance_to_camera * step_size;
    }

    var result: RaymarchResult;
    result.scattered_light = scattered_light;
    result.transmittance = transmittance_to_camera;
    return result;
}

// =============== MAIN SKY FUNCTION ===============
fn get_sky_color(camera_pos_world: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let cam_alt = length(camera_pos_world);
    let up = camera_pos_world / cam_alt;

    let atm_int = ray_sphere_intersect(camera_pos_world, view_dir, atmosphere.atmosphere_radius);
    if atm_int.y < 0.0 { return vec3<f32>(0.0); }

    let planet_int = ray_sphere_intersect(camera_pos_world, view_dir, atmosphere.planet_radius);
    var ray_len = atm_int.y;
    var hit_ground = false;
    if planet_int.y > 0.0 {
        ray_len = planet_int.x;
        hit_ground = true;
    }

    let sun_rad = sky.sun_color * atmosphere.sun_intensity;

    // ==================== GROUND (raymarch for atmospheric bands) ====================
    if hit_ground {
        let ground_pos = camera_pos_world + view_dir * ray_len;
        let ground_norm = normalize(ground_pos);
        let ground_ndotl = max(0.0, dot(ground_norm, atmosphere.sun_direction));

        // Raymarch
        let march = raymarch_to_ground(camera_pos_world, view_dir, ray_len, atmosphere.sun_direction);

        // Direct sun on ground
        let sun_trans = get_transmittance_to_sun(ground_pos, atmosphere.sun_direction);
        let direct = ground_albedo * sun_rad * sun_trans * ground_ndotl * 0.03;

        // Use scattered light as sky ambient
        let sky_ambient = ground_albedo * march.scattered_light * sun_rad * 20.0;

        // Horizon glow
        let horizon_factor = exp(-abs(atmosphere.sun_direction.y) * 4.0);
        let horizon_glow = ground_albedo * march.scattered_light * sun_rad * horizon_factor * 15.0;

        // Combine ground lighting
        let ground_color = direct + sky_ambient + horizon_glow;

        // Final: ground color + in-scattered light (scattered_light already has correct transmittance)
        return ground_color + march.scattered_light * sun_rad;
    }

    // ==================== SKY (use LUT for performance) ====================
    var color = get_scattering_color(camera_pos_world, view_dir);

    // SUN DISK
    let sun_cos = dot(view_dir, atmosphere.sun_direction);
    if sun_cos > SUN_ANGULAR_RADIUS_COS {
        let trans = get_transmittance(camera_pos_world, view_dir);

        // Limb darkening
        let t = (sun_cos - SUN_ANGULAR_RADIUS_COS) / (1.0 - SUN_ANGULAR_RADIUS_COS);
        let limb = 1.0 - 0.3 * (1.0 - sqrt(t));

        return sun_rad * trans * limb;
    }

    // Night ambient
    let night_factor = 1.0 - smoothstep(-0.2, 0.0, atmosphere.sun_direction.y);
    color += night_ambient_color * night_factor;

    return color;
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.screen_uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    out.clip_position = vec4<f32>(out.screen_uv.x * 2.0 - 1.0, out.screen_uv.y * -2.0 + 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, scene_sampler, in.screen_uv);
    if depth <= 0.0 {
        let clip_pos = vec4(in.screen_uv.x * 2.0 - 1.0, (1.0 - in.screen_uv.y) * 2.0 - 1.0, 1.0, 1.0);
        let world_pos_h = camera.inverse_view_projection_matrix * clip_pos;
        let world_pos = world_pos_h.xyz / world_pos_h.w;
        let view_dir = normalize(world_pos - camera.view_position);

        let camera_world_pos = vec3(0.0, atmosphere.planet_radius + 1.0, 0.0);
        let sky_color = get_sky_color(camera_world_pos, view_dir);

        let tonemapped = sky_color / (sky_color + vec3<f32>(1.0));
        let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2));
        
        return vec4<f32>(gamma_corrected, 1.0);
    } else {
        return vec4(0.0);
    }
}