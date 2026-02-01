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

struct ScatteringSlice {
    slice: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var<uniform> atmosphere: AtmosphereParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var atmosphere_sampler: sampler;
@group(0) @binding(3) var<uniform> scattering_slice: ScatteringSlice;

const PI: f32 = 3.14159265;
const TRANSMITTANCE_LUT_WIDTH: f32 = 256.0;
const TRANSMITTANCE_LUT_HEIGHT: f32 = 64.0;
const SCATTERING_LUT_WIDTH: f32 = 256.0;
const SCATTERING_LUT_HEIGHT: f32 = 128.0;
const SCATTERING_LUT_DEPTH: f32 = 32.0;
const IRRADIANCE_LUT_WIDTH: f32 = 64.0;
const IRRADIANCE_LUT_HEIGHT: f32 = 16.0;

const transmittance_samples: i32 = 64;
const scattering_samples: i32 = 32;
const irradiance_samples: i32 = 16;

struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var out: VsOut;
    out.position = vec4<f32>(positions[vid], 0.0, 1.0);
    return out;
}

fn ray_sphere_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_radius: f32) -> vec2<f32> {
    let b = dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;
    var delta = b * b - c;
    if delta < 0.0 { return vec2<f32>(-1.0); }
    delta = sqrt(delta);
    return vec2<f32>(-b - delta, -b + delta);
}

fn uv_to_altitude_mu(uv: vec2<f32>, planet_radius: f32, atmosphere_radius: f32) -> vec2<f32> {
    let alt_range = atmosphere_radius - planet_radius;
    let altitude = planet_radius + uv.x * alt_range;
    let mu = uv.y * 2.0 - 1.0;
    return vec2<f32>(altitude, mu);
}

fn altitude_mu_to_uv(altitude: f32, mu: f32, planet_radius: f32, atmosphere_radius: f32) -> vec2<f32> {
    let alt_range = atmosphere_radius - planet_radius;
    let u = (altitude - planet_radius) / alt_range;
    let v = (mu + 1.0) * 0.5;
    return vec2<f32>(u, v);
}

fn calculate_optical_depth(ray_origin: vec3<f32>, ray_dir: vec3<f32>, ray_length: f32, atm: AtmosphereParams) -> vec3<f32> {
    var optical_depth = vec3<f32>(0.0);
    let step_size = ray_length / f32(transmittance_samples);
    for (var i = 0; i < transmittance_samples; i = i + 1) {
        let sample_pos = ray_origin + ray_dir * (f32(i) + 0.5) * step_size;
        let height = length(sample_pos) - atm.planet_radius;
        if height < 0.0 { break; }
        let rayleigh_density = exp(-height / atm.rayleigh_scale_height);
        let mie_density = exp(-height / atm.mie_scale_height);
        let mie_extinction = atm.mie_scattering_coeff + atm.mie_absorption_coeff;
        optical_depth += (atm.rayleigh_scattering_coeff * rayleigh_density + mie_extinction * mie_density) * step_size;
    }
    return optical_depth;
}

fn mie_phase_function(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta)) / ((2.0 + g2) * pow(denom, 1.5));
}

fn rayleigh_phase_function(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

@fragment
fn fs_transmittance(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(pos.x / TRANSMITTANCE_LUT_WIDTH, pos.y / TRANSMITTANCE_LUT_HEIGHT);
    let params = uv_to_altitude_mu(uv, atmosphere.planet_radius, atmosphere.atmosphere_radius);
    let altitude = params.x;
    let mu = params.y;

    let ray_origin = vec3<f32>(0.0, altitude, 0.0);
    let ray_dir = vec3<f32>(sqrt(max(0.0, 1.0 - mu * mu)), mu, 0.0);
    let intersection = ray_sphere_intersect(ray_origin, ray_dir, atmosphere.atmosphere_radius);
    let ray_length = intersection.y;

    let optical_depth = calculate_optical_depth(ray_origin, ray_dir, ray_length, atmosphere);
    let transmittance = exp(-optical_depth);

    return vec4<f32>(transmittance, 1.0);
}

@fragment
fn fs_scattering(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let atm = atmosphere;
    let u = pos.x / SCATTERING_LUT_WIDTH;
    let v = pos.y / SCATTERING_LUT_HEIGHT;

    let alt_range = atm.atmosphere_radius - atm.planet_radius;
    let altitude = atm.planet_radius + u * alt_range;
    let mu_s = v * 2.0 - 1.0;
    let mu_v = (f32(scattering_slice.slice) + 0.5) / SCATTERING_LUT_DEPTH * 2.0 - 1.0;

    if mu_s < -0.15 {
        return vec4<f32>(0.0);
    }

    let ray_origin = vec3<f32>(0.0, altitude, 0.0);

    let sin_v = sqrt(max(0.0, 1.0 - mu_v * mu_v));
    let view_dir = vec3<f32>(sin_v, mu_v, 0.0);

    let sin_s = sqrt(max(0.0, 1.0 - mu_s * mu_s));
    let sun_dir = vec3<f32>(sin_s, mu_s, 0.0);

    let mu_v_s = dot(view_dir, sun_dir);

    let intersect_atmo = ray_sphere_intersect(ray_origin, view_dir, atm.atmosphere_radius);
    let intersect_planet = ray_sphere_intersect(ray_origin, view_dir, atm.planet_radius);

    var ray_length = intersect_atmo.y;
    if intersect_planet.x > 0.0 {
        ray_length = min(ray_length, intersect_planet.x);
    }

    let step_size = ray_length / f32(scattering_samples);
    let rayleigh_phase = rayleigh_phase_function(mu_v_s);
    let mie_phase = mie_phase_function(mu_v_s, atm.mie_preferred_scattering_dir);

    var rayleigh_scatter = vec3<f32>(0.0);
    var mie_scatter = vec3<f32>(0.0);

    for (var i = 0; i < scattering_samples; i = i + 1) {
        let sample_pos = ray_origin + view_dir * (f32(i) + 0.5) * step_size;
        let sample_altitude = length(sample_pos);
        if sample_altitude <= atm.planet_radius { break; }
        let sample_height = sample_altitude - atm.planet_radius;

        let sample_up = normalize(sample_pos);
        let mu_to_camera = dot(sample_up, -view_dir);
        let uv_to_camera = altitude_mu_to_uv(sample_altitude, mu_to_camera, atm.planet_radius, atm.atmosphere_radius);
        let transmittance_to_camera = textureSampleLevel(transmittance_lut, atmosphere_sampler, uv_to_camera, 0.0).rgb;

        let mu_to_sun = dot(sample_up, sun_dir);
        let uv_to_sun = altitude_mu_to_uv(sample_altitude, mu_to_sun, atm.planet_radius, atm.atmosphere_radius);
        let transmittance_to_sun = textureSampleLevel(transmittance_lut, atmosphere_sampler, uv_to_sun, 0.0).rgb;

        let rayleigh_density = exp(-sample_height / atm.rayleigh_scale_height);
        let mie_density = exp(-sample_height / atm.mie_scale_height);

        rayleigh_scatter += atm.rayleigh_scattering_coeff * rayleigh_density * rayleigh_phase * transmittance_to_sun * transmittance_to_camera * step_size;
        mie_scatter += atm.mie_scattering_coeff * mie_density * mie_phase * transmittance_to_sun * transmittance_to_camera * step_size;
    }

    let sun_radiance = atm.sun_intensity;
    let horizon_fade = smoothstep(-0.15, 0.0, mu_s);
    let combined = (rayleigh_scatter + mie_scatter) * sun_radiance * horizon_fade;
    return vec4<f32>(combined, 1.0);
}

@fragment
fn fs_irradiance(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let atm = atmosphere;
    let uv = vec2<f32>(pos.x / IRRADIANCE_LUT_WIDTH, pos.y / IRRADIANCE_LUT_HEIGHT);
    let params = uv_to_altitude_mu(uv, atm.planet_radius, atm.atmosphere_radius);
    let altitude = params.x;
    let mu_s = params.y;

    if mu_s < -0.2 {
        return vec4<f32>(atm.night_ambient_color, 1.0);
    }

    let ray_origin = vec3<f32>(0.0, altitude, 0.0);
    let sun_dir = normalize(vec3<f32>(sqrt(max(0.0, 1.0 - mu_s * mu_s)), mu_s, 0.0));

    var irradiance = vec3<f32>(0.0);
    let N = irradiance_samples;
    let invN = 1.0 / f32(N);

    for (var i = 0; i < N; i = i + 1) {
        for (var j = 0; j < N; j = j + 1) {
            let u = (f32(i) + 0.5) * invN;
            let v = (f32(j) + 0.5) * invN;

            let phi = 2.0 * PI * u;
            let cos_theta = sqrt(v);
            let sin_theta = sqrt(1.0 - v);

            let L = vec3<f32>(cos(phi) * sin_theta, cos_theta, sin(phi) * sin_theta);
            let LdotUp = L.y;

            let intersect_atmo = ray_sphere_intersect(ray_origin, L, atm.atmosphere_radius);
            let intersect_planet = ray_sphere_intersect(ray_origin, L, atm.planet_radius);
            var ray_len = intersect_atmo.y;
            if intersect_planet.x > 0.0 { ray_len = min(ray_len, intersect_planet.x); }

            let step = ray_len / f32(scattering_samples);
            var rayleigh = vec3<f32>(0.0);
            var mie = vec3<f32>(0.0);

            for (var k = 0; k < scattering_samples; k = k + 1) {
                let pos_sample = ray_origin + L * (f32(k) + 0.5) * step;
                let h = length(pos_sample) - atm.planet_radius;
                if h < 0.0 { break; }

                let density_r = exp(-h / atm.rayleigh_scale_height);
                let density_m = exp(-h / atm.mie_scale_height);

                let mu_cam = dot(normalize(pos_sample), -L);
                let uv_cam = altitude_mu_to_uv(length(pos_sample), mu_cam, atm.planet_radius, atm.atmosphere_radius);
                let t_cam = textureSampleLevel(transmittance_lut, atmosphere_sampler, uv_cam, 0.0).rgb;

                let mu_sun = dot(normalize(pos_sample), sun_dir);
                let uv_sun = altitude_mu_to_uv(length(pos_sample), mu_sun, atm.planet_radius, atm.atmosphere_radius);
                let t_sun = textureSampleLevel(transmittance_lut, atmosphere_sampler, uv_sun, 0.0).rgb;

                let phase_r = rayleigh_phase_function(dot(L, sun_dir));
                let phase_m = mie_phase_function(dot(L, sun_dir), atm.mie_preferred_scattering_dir);

                rayleigh += atm.rayleigh_scattering_coeff * density_r * phase_r * t_sun * t_cam * step;
                mie += atm.mie_scattering_coeff * density_m * phase_m * t_sun * t_cam * step;
            }

            var ground = vec3<f32>(0.0);
            if intersect_planet.x > 0.0 {
                let hit = ray_origin + L * intersect_planet.x;
                let n_g = normalize(hit);
                let v_g = -L;
                let t_g = textureSampleLevel(transmittance_lut, atmosphere_sampler,
                    altitude_mu_to_uv(length(hit), dot(n_g, v_g), atm.planet_radius, atm.atmosphere_radius), 0.0).rgb;
                ground = atm.ground_albedo * atm.ground_brightness * t_g * max(dot(n_g, sun_dir), 0.0) * atm.sun_intensity;
            }

            irradiance += (rayleigh + mie) * LdotUp + ground * LdotUp;
        }
    }

    irradiance *= atm.sun_intensity * invN * invN * 2.0 * PI;
    let horizon_fade = smoothstep(-0.2, 0.0, mu_s);
    irradiance *= horizon_fade;

    return vec4<f32>(irradiance, 1.0);
}
