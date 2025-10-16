struct Constants {
    // sky
    planet_radius: f32,          // 0x00
    atmosphere_radius: f32,      // 0x04
    sky_light_samples: u32,      // 0x08
    _pad0: f32,                 // 0x0C - padding to 16 bytes

    // SSR block 1
    ssr_coarse_steps: u32,       // 0x10
    ssr_binary_search_steps: u32,// 0x14
    ssr_linear_step_size: f32,   // 0x18
    ssr_thickness: f32,          // 0x1C

    // SSR block 2
    ssr_max_distance: f32,       // 0x20
    ssr_roughness_fade_start: f32,// 0x24
    ssr_roughness_fade_end: f32, // 0x28
    _pad1: f32,                  // 0x2C - padding to 16 bytes

    // SSGI block 1
    ssgi_num_rays: u32,          // 0x30
    ssgi_num_steps: u32,         // 0x34
    ssgi_ray_step_size: f32,     // 0x38
    ssgi_thickness: f32,         // 0x3C

    // SSGI block 2
    ssgi_blend_factor: f32,      // 0x40
    evsm_c: f32,                 // 0x44
    pcf_radius: u32,             // 0x48
    ssgi_intensity: f32,         // 0x4C

    // Final padding to align total struct size to 16 bytes
    _padding: vec4<f32>,         // 0x50 - 16 bytes padding
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> sky: SkyUniforms;

@group(1) @binding(0) var<uniform> constants: Constants;

const PI = 3.14159265;

// Atmospheric scattering parameters
const rayleigh_scattering_coeff = vec3(5.5e-6, 13.0e-6, 22.4e-6);
const rayleigh_scale_height = 8e3;

const mie_scattering_coeff = 21e-6;
const mie_scale_height = 1.2e3;
const mie_preferred_scattering_dir = 0.76;

fn ray_sphere_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_radius: f32) -> vec2<f32> {
    let b = dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;
    var delta = b * b - c;
    if (delta < 0.0) {
        return vec2(-1.0);
    }
    delta = sqrt(delta);
    return vec2(-b - delta, -b + delta);
}

// Calculates transmittance (how much light is NOT scattered) from a point to the sun
fn get_transmittance_to_sun(sample_pos: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let dist_to_atmosphere = ray_sphere_intersect(sample_pos, sun_dir, constants.atmosphere_radius).y;
    let num_light_samples = i32(constants.sky_light_samples);
    let light_step_size = dist_to_atmosphere / f32(num_light_samples);
    var optical_depth = vec3(0.0);

    for (var j = 0; j < num_light_samples; j = j + 1) {
        let light_pos = sample_pos + sun_dir * (f32(j) + 0.5) * light_step_size;
        let height = length(light_pos) - constants.planet_radius;
        if (height < 0.0) { // Below ground
            return vec3(0.0);
        }

        let rayleigh_density = exp(-height / rayleigh_scale_height);
        let mie_density = exp(-height / mie_scale_height);
        
        optical_depth += (rayleigh_scattering_coeff * rayleigh_density + mie_scattering_coeff * mie_density) * light_step_size;
    }
    return exp(-optical_depth);
}

fn get_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let camera_pos = vec3(0.0, constants.planet_radius + 1.0, 0.0);
    let dist_to_atmosphere = ray_sphere_intersect(camera_pos, view_dir, constants.atmosphere_radius).y;
    
    let num_samples = i32(constants.sky_light_samples);
    let step_size = dist_to_atmosphere / f32(num_samples);

    var transmittance_to_camera = vec3(1.0);
    var scattered_light = vec3(0.0);

    for (var i = 0; i < num_samples; i = i + 1) {
        let sample_pos = camera_pos + view_dir * (f32(i) + 0.5) * step_size;
        let height = length(sample_pos) - constants.planet_radius;

        if (height < 0.0) { break; }

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

    let sun_radiance = sky.sun_color * sky.sun_intensity;
    var final_color = scattered_light * sun_radiance;

    // Sun disk
    let sun_cos_theta = dot(view_dir, sun_dir);
    if (sun_cos_theta > 0.9998) {
        final_color += sun_radiance * transmittance_to_camera;
    }
    
    return final_color;
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
    let clip_pos = vec4(in.screen_uv.x * 2.0 - 1.0, (1.0 - in.screen_uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let world_pos_h = camera.inverse_view_projection_matrix * clip_pos;
    let world_pos = world_pos_h.xyz / world_pos_h.w;
    let view_dir = normalize(world_pos - camera.view_position);
    
    let color = get_sky_color(view_dir, sky.sun_direction);
    return vec4(color, 1.0);
}