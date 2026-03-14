enable wgpu_mesh_shader;

const PI: f32 = 3.14159265;
const SCATTERING_LUT_DEPTH: f32 = 32.0;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
};

var<workgroup> mesh_output: MeshOutput;

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
    // general
    mip_bias: f32,

    // lighting
    shade_mode: u32,
    shade_smooth: u32,
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
    _final_padding: vec4<f32>,
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
@group(1) @binding(1) var scattering_lut: texture_2d_array<f32>;
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

fn altitude_mu_to_uv(altitude: f32, mu: f32, planet_radius: f32, atmosphere_radius: f32) -> vec2<f32> {
    let alt_range = atmosphere_radius - planet_radius;
    let u = (altitude - planet_radius) / alt_range;
    let v = (mu + 1.0) * 0.5;
    return clamp(vec2<f32>(u, v), vec2<f32>(0.0), vec2<f32>(1.0));
}

fn scattering_lut_coords(altitude: f32, mu_s: f32, mu_v: f32, planet_radius: f32, atmosphere_radius: f32) -> vec3<f32> {
    let alt_range = atmosphere_radius - planet_radius;
    let u = (altitude - planet_radius) / alt_range;
    let v = (mu_s + 1.0) * 0.5;
    let w = (mu_v + 1.0) * 0.5;
    return clamp(vec3<f32>(u, v, w), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sample_scattering_lut(coords: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(coords, vec3<f32>(0.0), vec3<f32>(1.0));
    let layer_f = clamped.z * (SCATTERING_LUT_DEPTH - 1.0);
    let layer0 = floor(layer_f);
    let layer1 = min(layer0 + 1.0, SCATTERING_LUT_DEPTH - 1.0);
    let t = layer_f - layer0;
    let uv = clamped.xy;
    let layer0_i = i32(layer0);
    let layer1_i = i32(layer1);
    let c0 = textureSampleLevel(scattering_lut, atmosphere_sampler, uv, layer0_i, 0.0).rgb;
    let c1 = textureSampleLevel(scattering_lut, atmosphere_sampler, uv, layer1_i, 0.0).rgb;
    return mix(c0, c1, t);
}

fn ozone_density(height: f32) -> f32 {
    return max(0.0, 1.0 - abs(height - atmosphere.ozone_center_height) / atmosphere.ozone_falloff);
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
    let samples = max(1, i32(constants.sky_light_samples));
    let step = ray_len / f32(samples);
    var od = vec3<f32>(0.0);

    for (var i = 0; i < samples; i = i + 1) {
        let p = sample_pos + sun_dir * (f32(i) + 0.5) * step;
        let h = length(p) - atmosphere.planet_radius;
        if h < 0.0 { return vec3<f32>(0.0); }

        let rd = exp(-h / atmosphere.rayleigh_scale_height);
        let md = exp(-h / atmosphere.mie_scale_height);
        let odens = ozone_density(h);
        let mie_ext = atmosphere.mie_scattering_coeff + atmosphere.mie_absorption_coeff;

        od += (atmosphere.rayleigh_scattering_coeff * rd + vec3<f32>(mie_ext) * md + atmosphere.ozone_absorption_coeff * odens) * step;
    }
    return exp(-od);
}

// =============== LUT SAMPLING ===============
fn get_transmittance(world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let altitude = length(world_pos);
    let up = world_pos / altitude;
    let mu = dot(view_dir, up);
    let uv = altitude_mu_to_uv(altitude, mu, atmosphere.planet_radius, atmosphere.atmosphere_radius);
    return textureSampleLevel(transmittance_lut, atmosphere_sampler, uv, 0.0).rgb;
}

fn get_scattering_color(world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let altitude = length(world_pos);
    let up = world_pos / altitude;
    let mu_s = dot(atmosphere.sun_direction, up);
    let mu_v = dot(view_dir, up);

    let coords = scattering_lut_coords(altitude, mu_s, mu_v, atmosphere.planet_radius, atmosphere.atmosphere_radius);

    return sample_scattering_lut(coords);
}

// =============== RAYMARCH FOR GROUND (creates atmospheric bands) ===============
struct RaymarchResult {
    scattered_light: vec3<f32>,
    transmittance: vec3<f32>,
};

fn raymarch_to_ground(camera_pos: vec3<f32>, view_dir: vec3<f32>, ray_len: f32, sun_dir: vec3<f32>) -> RaymarchResult {
    let num_samples = max(1, i32(constants.sky_light_samples));
    let step_size = ray_len / f32(num_samples);

    var transmittance_to_camera = vec3<f32>(1.0);
    var scattered_light = vec3<f32>(0.0);

    let cos_theta = dot(view_dir, sun_dir);
    let rayleigh_phase = rayleigh_phase_function(cos_theta);
    let mie_phase = mie_phase_function(cos_theta, atmosphere.mie_preferred_scattering_dir);

    for (var i = 0; i < num_samples; i = i + 1) {
        let sample_pos = camera_pos + view_dir * (f32(i) + 0.5) * step_size;
        let height = length(sample_pos) - atmosphere.planet_radius;

        if height < 0.0 { break; }

        let rayleigh_density = exp(-height / atmosphere.rayleigh_scale_height);
        let mie_density = exp(-height / atmosphere.mie_scale_height);
        let ozone_dens = ozone_density(height);
        let mie_ext = atmosphere.mie_scattering_coeff + atmosphere.mie_absorption_coeff;

        let optical_depth_step = (atmosphere.rayleigh_scattering_coeff * rayleigh_density + vec3<f32>(mie_ext) * mie_density + atmosphere.ozone_absorption_coeff * ozone_dens) * step_size;
        transmittance_to_camera *= exp(-optical_depth_step);

        let transmittance_to_sun = get_transmittance_to_sun(sample_pos, sun_dir);

        let in_scattered_r = atmosphere.rayleigh_scattering_coeff * rayleigh_density * rayleigh_phase;
        let in_scattered_m = atmosphere.mie_scattering_coeff * mie_density * mie_phase;

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
    let ground_albedo = sky.ground_albedo * sky.ground_brightness;

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
    if sun_cos > sky.sun_angular_radius_cos {
        let trans = get_transmittance(camera_pos_world, view_dir);

        // Limb darkening
        let t = (sun_cos - sky.sun_angular_radius_cos) / (1.0 - sky.sun_angular_radius_cos);
        let limb = 1.0 - 0.3 * (1.0 - sqrt(t));

        return sun_rad * trans * limb;
    }

    // Night ambient
    let night_factor = 1.0 - smoothstep(-0.2, 0.0, atmosphere.sun_direction.y);
    color += sky.night_ambient_color * night_factor;

    return color;
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.screen_uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    out.clip_position = vec4<f32>(out.screen_uv.x * 2.0 - 1.0, out.screen_uv.y * -2.0 + 1.0, 0.0, 1.0);
    return out;
}

fn write_vertex(index: u32) {
    let screen_uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    mesh_output.vertices[index].screen_uv = screen_uv;
    mesh_output.vertices[index].clip_position = vec4<f32>(
        screen_uv.x * 2.0 - 1.0,
        screen_uv.y * -2.0 + 1.0,
        0.0,
        1.0
    );
}

@mesh(mesh_output)
@workgroup_size(1)
fn ms_main() {
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    write_vertex(0u);
    write_vertex(1u);
    write_vertex(2u);
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, scene_sampler, in.screen_uv).x;
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
