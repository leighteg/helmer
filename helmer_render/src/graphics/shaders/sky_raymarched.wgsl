const PI: f32 = 3.14159265;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
};

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

struct Constants {
    // lighting
    shade_mode: u32,
    shade_smooth: u32,
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
    _pad_light: vec4<u32>,
    prev_view_proj: mat4x4<f32>,
    frame_index: u32,
    _padding: vec3<u32>,
    _pad_end: vec4<u32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> sky: SkyUniforms;
@group(0) @binding(2) var scene_sampler: sampler;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;

@group(1) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(1) var scattering_lut: texture_3d<f32>;
@group(1) @binding(2) var irradiance_lut: texture_2d<f32>;
@group(1) @binding(3) var atmosphere_sampler: sampler;
@group(1) @binding(4) var<uniform> atmosphere: AtmosphereParams;

@group(2) @binding(0) var<uniform> constants: Constants;

fn ray_sphere_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_radius: f32) -> vec2<f32> {
    let b = dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;
    var delta = b * b - c;
    if delta < 0.0 { return vec2<f32>(-1.0); }
    delta = sqrt(delta);
    return vec2<f32>(-b - delta, -b + delta);
}

// transmittance calculation with ozone absorption
fn get_transmittance_to_sun(sample_pos: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let dist_to_atmosphere = ray_sphere_intersect(sample_pos, sun_dir, constants.atmosphere_radius).y;
    let num_light_samples = max(1, i32(constants.sky_light_samples));
    let light_step_size = dist_to_atmosphere / f32(num_light_samples);
    var optical_depth = vec3(0.0);

    for (var j = 0; j < num_light_samples; j = j + 1) {
        let light_pos = sample_pos + sun_dir * (f32(j) + 0.5) * light_step_size;
        let height = length(light_pos) - constants.planet_radius;
        if height < 0.0 {
            return vec3(0.0);
        }

        let rayleigh_density = exp(-height / atmosphere.rayleigh_scale_height);
        let mie_density = exp(-height / atmosphere.mie_scale_height);
        
        // Add ozone absorption for better sunset/sunrise colors
        let ozone_density = max(
            0.0,
            1.0 - abs(height - atmosphere.ozone_center_height) / atmosphere.ozone_falloff,
        );
        let mie_ext = atmosphere.mie_scattering_coeff + atmosphere.mie_absorption_coeff;

        optical_depth += (atmosphere.rayleigh_scattering_coeff * rayleigh_density
            + mie_ext * mie_density
            + atmosphere.ozone_absorption_coeff * ozone_density)
            * light_step_size;
    }
    return exp(-optical_depth);
}

// Cornette-Shanks phase function for Mie scattering
fn mie_phase_function(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta)) / ((2.0 + g2) * pow(denom, 1.5));
}

// Rayleigh phase function
fn rayleigh_phase_function(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn get_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let camera_pos = vec3(0.0, constants.planet_radius + 1.0, 0.0);
    
    // Check if ray hits ground
    let ground_intersection = ray_sphere_intersect(camera_pos, view_dir, constants.planet_radius);
    let hits_ground = ground_intersection.x > 0.0;

    let dist_to_atmosphere = ray_sphere_intersect(camera_pos, view_dir, constants.atmosphere_radius).y;

    let num_samples = max(1, i32(constants.sky_light_samples));
    let ray_length = select(dist_to_atmosphere, ground_intersection.x, hits_ground);
    let step_size = ray_length / f32(num_samples);

    var transmittance_to_camera = vec3(1.0);
    var scattered_light = vec3(0.0);

    let cos_theta = dot(view_dir, sun_dir);
    let rayleigh_phase = rayleigh_phase_function(cos_theta);
    let mie_phase = mie_phase_function(cos_theta, atmosphere.mie_preferred_scattering_dir);

    for (var i = 0; i < num_samples; i = i + 1) {
        let sample_pos = camera_pos + view_dir * (f32(i) + 0.5) * step_size;
        let height = length(sample_pos) - constants.planet_radius;

        if height < 0.0 { break; }

        let rayleigh_density = exp(-height / atmosphere.rayleigh_scale_height);
        let mie_density = exp(-height / atmosphere.mie_scale_height);
        
        // Include ozone in camera transmittance
        let ozone_density = max(
            0.0,
            1.0 - abs(height - atmosphere.ozone_center_height) / atmosphere.ozone_falloff,
        );
        let mie_ext = atmosphere.mie_scattering_coeff + atmosphere.mie_absorption_coeff;

        let optical_depth_step = (atmosphere.rayleigh_scattering_coeff * rayleigh_density
            + mie_ext * mie_density
            + atmosphere.ozone_absorption_coeff * ozone_density)
            * step_size;
        transmittance_to_camera *= exp(-optical_depth_step);

        let transmittance_to_sun = get_transmittance_to_sun(sample_pos, sun_dir);

        let in_scattered_r = atmosphere.rayleigh_scattering_coeff * rayleigh_density * rayleigh_phase;
        let in_scattered_m = atmosphere.mie_scattering_coeff * mie_density * mie_phase;

        scattered_light += (in_scattered_r + in_scattered_m) * transmittance_to_sun * transmittance_to_camera * step_size;
    }

    let sun_radiance = sky.sun_color * sky.sun_intensity;
    let ground_albedo = sky.ground_albedo * sky.ground_brightness;
    var final_color = scattered_light * sun_radiance;
    
    // ground contribution
    if hits_ground {
        let ground_point = camera_pos + view_dir * ground_intersection.x;
        let ground_normal = normalize(ground_point);
        let ground_ndotl = max(0.0, dot(ground_normal, sun_dir));
        
        // Direct sun on ground
        let sun_transmittance = get_transmittance_to_sun(ground_point, sun_dir);
        let ground_direct = ground_albedo * sun_radiance * sun_transmittance * ground_ndotl * 0.03;
        
        // Use actual scattered light as ambient
        // If sky has color, ground should have color
        let sky_ambient = ground_albedo * scattered_light * sun_radiance * 20.0;
        
        // Horizon glow during sunrise/sunset
        let horizon_factor = exp(-abs(sun_dir.y) * 4.0);
        let horizon_glow = ground_albedo * scattered_light * sun_radiance * horizon_factor * 15.0;
        
        // Combine ground lighting
        let ground_color = ground_direct + sky_ambient + horizon_glow;
        final_color = ground_color * transmittance_to_camera + scattered_light * sun_radiance;
    }

    // Sun disk
    let sun_cos_theta = dot(view_dir, sun_dir);
    if sun_cos_theta > sky.sun_angular_radius_cos {
        // Limb darkening effect for more realistic sun
        let limb = 1.0 - 0.3 * (1.0 - sqrt(sun_cos_theta / sky.sun_angular_radius_cos));
        final_color += sun_radiance * transmittance_to_camera * limb;
    }
    
    // Add subtle night sky ambient for twilight (when sun is below horizon)
    let sun_height_factor = smoothstep(-0.2, 0.0, sun_dir.y);
    let night_ambient = sky.night_ambient_color * (1.0 - sun_height_factor);
    final_color += night_ambient;

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
    let depth = textureSample(depth_tex, scene_sampler, in.screen_uv).x;
    if depth <= 0.0 {
        let clip_pos = vec4(in.screen_uv.x * 2.0 - 1.0, (1.0 - in.screen_uv.y) * 2.0 - 1.0, 1.0, 1.0);
        let world_pos_h = camera.inverse_view_projection_matrix * clip_pos;
        let world_pos = world_pos_h.xyz / world_pos_h.w;
        let view_dir = normalize(world_pos - camera.view_position);

        let color = get_sky_color(view_dir, sky.sun_direction);
        return vec4(color, 1.0);
    } else {
        return vec4(0.0);
    }
}
