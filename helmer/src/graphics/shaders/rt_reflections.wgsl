const PI: f32 = 3.14159265;
const MAX_STACK: u32 = 128u;
const EPSILON: f32 = 1.0e-4;
const RT_FLAG_DIRECT_LIGHTING: u32 = 1u << 0u;
const RT_FLAG_SHADOWS: u32 = 1u << 1u;
const RT_FLAG_USE_TEXTURES: u32 = 1u << 2u;
const RT_FLAG_SHADE_SMOOTH: u32 = 1u << 3u;

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
    _padding0: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    ground_albedo: vec3<f32>,
    ground_brightness: f32,
    night_ambient_color: vec3<f32>,
    sun_angular_radius_cos: f32,
}

struct LightData {
    position: vec3<f32>,
    light_type: u32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    _padding: f32,
}

struct MaterialData {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,
    albedo_idx: i32,
    normal_idx: i32,
    metallic_roughness_idx: i32,
    emission_idx: i32,
    emission_color: vec3<f32>,
    _padding: f32,
    alpha_mode: u32,
    alpha_cutoff: f32,
    _pad_alpha0: u32,
    _pad_alpha1: u32,
}

struct RtBvhNode {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    left: u32,
    right: u32,
    first_index: u32,
    index_count: u32,
}

struct RtTriangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    _pad0: vec2<f32>,
}

struct RtBlasDesc {
    node_offset: u32,
    node_count: u32,
    index_offset: u32,
    index_count: u32,
    tri_offset: u32,
    tri_count: u32,
    _pad0: vec2<u32>,
}

struct RtInstance {
    model: mat4x4<f32>,
    inv_model: mat4x4<f32>,
    blas_index: u32,
    material_id: u32,
    _pad0: vec2<u32>,
}

struct RtConstants {
    rng_frame_index: u32,
    accumulation_frame: u32,
    max_bounces: u32,
    samples_per_frame: u32,
    light_count: u32,
    flags: u32,
    reset: u32,
    width: u32,
    height: u32,
    tlas_node_count: u32,
    tlas_index_count: u32,
    instance_count: u32,
    blas_desc_count: u32,
    direct_light_samples: u32,
    max_accumulation_frames: u32,
    sky_view_samples: u32,
    sky_sun_samples: u32,
    exposure: f32,
    env_intensity: f32,
    firefly_clamp: f32,
    shadow_bias: f32,
    ray_bias: f32,
    min_roughness: f32,
    normal_map_strength: f32,
    throughput_cutoff: f32,
    sky_multi_scatter_strength: f32,
    sky_multi_scatter_power: f32,
    texture_array_layers: u32,
}

struct RtReflectionParams {
    params0: vec4<f32>,
    params1: vec4<u32>,
    params2: vec4<u32>,
}

struct ShaderConstants {
    mip_bias: f32,
    shade_mode: u32,
    shade_smooth: u32,
    light_model: u32,
    skylight_contribution: u32,
    planet_radius: f32,
    atmosphere_radius: f32,
    sky_light_samples: u32,
    _pad0: u32,
    ssr_coarse_steps: u32,
    ssr_binary_search_steps: u32,
    ssr_linear_step_size: f32,
    ssr_thickness: f32,
    ssr_max_distance: f32,
    ssr_roughness_fade_start: f32,
    ssr_roughness_fade_end: f32,
    _pad1: u32,
    ssgi_num_rays: u32,
    ssgi_num_steps: u32,
    ssgi_ray_step_size: f32,
    ssgi_thickness: f32,
    ssgi_blend_factor: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    evsm_c: f32,
    pcf_radius: u32,
    pcf_min_scale: f32,
    pcf_max_scale: f32,
    pcf_max_distance: f32,
    ssgi_intensity: f32,
    _final_padding0: f32,
    _final_padding1: f32,
    ssr_jitter_strength: f32,
    _pad_after_ssr_jitter0: f32,
    _pad_after_ssr_jitter1: f32,
    _pad_after_ssr_jitter2: f32,
    rayleigh_scattering_coeff_x: f32,
    rayleigh_scattering_coeff_y: f32,
    rayleigh_scattering_coeff_z: f32,
    rayleigh_scale_height: f32,
    mie_scattering_coeff: f32,
    mie_absorption_coeff: f32,
    mie_scale_height: f32,
    mie_preferred_scattering_dir: f32,
    ozone_absorption_coeff_x: f32,
    ozone_absorption_coeff_y: f32,
    ozone_absorption_coeff_z: f32,
    ozone_center_height: f32,
    ozone_falloff: f32,
    sun_angular_radius_cos: f32,
    _pad_atmo0_0: f32,
    _pad_atmo0_1: f32,
    night_ambient_color_x: f32,
    night_ambient_color_y: f32,
    night_ambient_color_z: f32,
    _pad_atmo1: f32,
    sky_ground_albedo_x: f32,
    sky_ground_albedo_y: f32,
    sky_ground_albedo_z: f32,
    sky_ground_brightness: f32,
    _pad_end0: f32,
    _pad_end1: f32,
    _pad_end2: f32,
}

struct HitInfo {
    pos: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    material_id: u32,
    instance_index: u32,
    tri_index: u32,
    hit: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> constants: RtConstants;
@group(0) @binding(2) var<storage, read> tlas_nodes: array<RtBvhNode>;
@group(0) @binding(3) var<storage, read> tlas_indices: array<u32>;
@group(0) @binding(4) var<storage, read> instances: array<RtInstance>;
@group(0) @binding(5) var<storage, read> blas_nodes: array<RtBvhNode>;
@group(0) @binding(6) var<storage, read> blas_indices: array<u32>;
@group(0) @binding(7) var<storage, read> blas_triangles: array<RtTriangle>;
@group(0) @binding(8) var<storage, read> blas_descs: array<RtBlasDesc>;
@group(0) @binding(9) var<storage, read> lights: array<LightData>;
@group(0) @binding(10) var<storage, read> materials: array<MaterialData>;
@group(0) @binding(11) var<uniform> sky: SkyUniforms;
@group(0) @binding(12) var<uniform> shader_constants: ShaderConstants;

@group(1) @binding(0) var gbuf_normal: texture_2d<f32>;
@group(1) @binding(1) var gbuf_mra: texture_2d<f32>;
@group(1) @binding(2) var depth_tex: texture_2d<f32>;
@group(1) @binding(3) var history_tex: texture_2d<f32>;
@group(1) @binding(4) var<uniform> reflection_params: RtReflectionParams;

@group(2) @binding(0) var reflection_out: texture_storage_2d<rgba16float, write>;

fn wang_hash(seed: u32) -> u32 {
    var v = seed ^ 61u;
    v = v ^ (v >> 16u);
    v = v * 9u;
    v = v ^ (v >> 4u);
    v = v * 0x27d4eb2du;
    v = v ^ (v >> 15u);
    return v;
}

fn rand(state: ptr<function, u32>) -> f32 {
    (*state) = wang_hash(*state);
    return f32(*state) / 4294967296.0;
}

fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let t0 = (bmin - origin) * inv_dir;
    let t1 = (bmax - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let near = max(max(tmin.x, tmin.y), tmin.z);
    let far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(near, far);
}

fn intersect_triangle(origin: vec3<f32>, dir: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> vec3<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = cross(dir, e2);
    let det = dot(e1, p);
    if abs(det) < EPSILON {
        return vec3<f32>(-1.0);
    }
    let inv_det = 1.0 / det;
    let t = origin - v0;
    let u = dot(t, p) * inv_det;
    if u < 0.0 || u > 1.0 {
        return vec3<f32>(-1.0);
    }
    let q = cross(t, e1);
    let v = dot(dir, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return vec3<f32>(-1.0);
    }
    let t_hit = dot(e2, q) * inv_det;
    return vec3<f32>(t_hit, u, v);
}

fn cosine_sample_hemisphere(r1: f32, r2: f32) -> vec3<f32> {
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn make_tangent(n: vec3<f32>) -> vec3<f32> {
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
}

fn max_component(v: vec3<f32>) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn vec3_from_scalars(x: f32, y: f32, z: f32) -> vec3<f32> {
    return vec3<f32>(x, y, z);
}

fn ray_sphere_intersect(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let h = b * b - c;
    if h < 0.0 {
        return vec2(-1.0);
    }
    let sqrt_h = sqrt(h);
    return vec2(-b - sqrt_h, -b + sqrt_h);
}

fn rayleigh_phase_function(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn mie_phase_function(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta))
        / ((2.0 + g2) * pow(denom, 1.5));
}

fn ozone_density(height: f32, center: f32, falloff: f32) -> f32 {
    let safe_falloff = max(falloff, EPSILON);
    return max(0.0, 1.0 - abs(height - center) / safe_falloff);
}

fn transmittance_to_sun(
    sample_pos: vec3<f32>,
    sun_dir: vec3<f32>,
    planet_radius: f32,
    atmosphere_radius: f32,
) -> vec3<f32> {
    let planet_hit = ray_sphere_intersect(sample_pos, sun_dir, planet_radius);
    if planet_hit.x > 0.0 {
        return vec3(0.0);
    }
    let atmo_hit = ray_sphere_intersect(sample_pos, sun_dir, atmosphere_radius);
    if atmo_hit.y <= 0.0 {
        return vec3(1.0);
    }

    let steps = max(1u, constants.sky_sun_samples);
    let step_size = atmo_hit.y / f32(steps);
    let rayleigh_coeff = vec3_from_scalars(
        shader_constants.rayleigh_scattering_coeff_x,
        shader_constants.rayleigh_scattering_coeff_y,
        shader_constants.rayleigh_scattering_coeff_z,
    );
    let ozone_coeff = vec3_from_scalars(
        shader_constants.ozone_absorption_coeff_x,
        shader_constants.ozone_absorption_coeff_y,
        shader_constants.ozone_absorption_coeff_z,
    );
    let rayleigh_scale = max(shader_constants.rayleigh_scale_height, EPSILON);
    let mie_scale = max(shader_constants.mie_scale_height, EPSILON);
    let ozone_falloff = max(shader_constants.ozone_falloff, EPSILON);
    let mie_ext = shader_constants.mie_scattering_coeff + shader_constants.mie_absorption_coeff;

    var optical_depth = vec3(0.0);
    var t = 0.0;
    for (var i = 0u; i < steps; i = i + 1u) {
        let pos = sample_pos + sun_dir * (t + 0.5 * step_size);
        let height = length(pos) - planet_radius;
        if height < 0.0 {
            return vec3(0.0);
        }

        let rayleigh_density = exp(-height / rayleigh_scale);
        let mie_density = exp(-height / mie_scale);
        let ozone = ozone_density(height, shader_constants.ozone_center_height, ozone_falloff);

        optical_depth += (rayleigh_coeff * rayleigh_density
            + vec3<f32>(mie_ext) * mie_density
            + ozone_coeff * ozone)
            * step_size;
        t += step_size;
    }

    return exp(-optical_depth);
}

fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let view_dir = normalize(dir);
    let planet_radius = max(shader_constants.planet_radius, EPSILON);
    let atmosphere_radius = max(shader_constants.atmosphere_radius, planet_radius + EPSILON);
    let sun_dir = normalize(sky.sun_direction);
    let sun_radiance = sky.sun_color * sky.sun_intensity;
    let rayleigh_coeff = vec3_from_scalars(
        shader_constants.rayleigh_scattering_coeff_x,
        shader_constants.rayleigh_scattering_coeff_y,
        shader_constants.rayleigh_scattering_coeff_z,
    );
    let ozone_coeff = vec3_from_scalars(
        shader_constants.ozone_absorption_coeff_x,
        shader_constants.ozone_absorption_coeff_y,
        shader_constants.ozone_absorption_coeff_z,
    );
    let night_ambient = vec3_from_scalars(
        shader_constants.night_ambient_color_x,
        shader_constants.night_ambient_color_y,
        shader_constants.night_ambient_color_z,
    );
    let ground_albedo =
        vec3_from_scalars(
            shader_constants.sky_ground_albedo_x,
            shader_constants.sky_ground_albedo_y,
            shader_constants.sky_ground_albedo_z,
        ) * shader_constants.sky_ground_brightness;
    let planet_center = vec3<f32>(0.0, -planet_radius, 0.0);
    let origin = camera.view_position - planet_center;

    let atmo_hit = ray_sphere_intersect(origin, view_dir, atmosphere_radius);
    if atmo_hit.y < 0.0 {
        var color = night_ambient;
        if dot(view_dir, sun_dir) > shader_constants.sun_angular_radius_cos {
            color += sun_radiance;
        }
        return color * constants.env_intensity;
    }

    var t_start = max(atmo_hit.x, 0.0);
    var t_end = atmo_hit.y;
    let ground_hit = ray_sphere_intersect(origin, view_dir, planet_radius);
    var hits_ground = false;
    if ground_hit.x > 0.0 {
        t_end = min(t_end, ground_hit.x);
        hits_ground = true;
    }

    if t_end <= t_start {
        var color = night_ambient;
        if dot(view_dir, sun_dir) > shader_constants.sun_angular_radius_cos {
            color += sun_radiance;
        }
        return color * constants.env_intensity;
    }

    let steps = max(1u, constants.sky_view_samples);
    let step_size = (t_end - t_start) / f32(steps);
    let rayleigh_scale = max(shader_constants.rayleigh_scale_height, EPSILON);
    let mie_scale = max(shader_constants.mie_scale_height, EPSILON);
    let ozone_falloff = max(shader_constants.ozone_falloff, EPSILON);
    let mie_ext = shader_constants.mie_scattering_coeff + shader_constants.mie_absorption_coeff;
    let cos_theta = dot(view_dir, sun_dir);
    let rayleigh_phase = rayleigh_phase_function(cos_theta);
    let mie_phase = mie_phase_function(cos_theta, shader_constants.mie_preferred_scattering_dir);

    var transmittance = vec3(1.0);
    var scattered = vec3(0.0);
    var t = t_start;
    for (var i = 0u; i < steps; i = i + 1u) {
        let sample_pos = origin + view_dir * (t + 0.5 * step_size);
        let height = length(sample_pos) - planet_radius;
        if height < 0.0 {
            break;
        }

        let rayleigh_density = exp(-height / rayleigh_scale);
        let mie_density = exp(-height / mie_scale);
        let ozone = ozone_density(height, shader_constants.ozone_center_height, ozone_falloff);

        let extinction = (rayleigh_coeff * rayleigh_density
            + vec3<f32>(mie_ext) * mie_density
            + ozone_coeff * ozone)
            * step_size;

        let transmittance_mid = transmittance * exp(-extinction * 0.5);
        let trans_to_sun = transmittance_to_sun(
            sample_pos,
            sun_dir,
            planet_radius,
            atmosphere_radius,
        );
        let scattering = rayleigh_coeff * rayleigh_density * rayleigh_phase
            + vec3<f32>(shader_constants.mie_scattering_coeff) * mie_density * mie_phase;

        scattered += scattering * trans_to_sun * transmittance_mid * step_size;
        transmittance *= exp(-extinction);
        t += step_size;
    }

    var radiance = scattered * sun_radiance;
    let multi_strength = max(constants.sky_multi_scatter_strength, 0.0);
    let multi_power = max(constants.sky_multi_scatter_power, EPSILON);
    let multi_depth = pow(clamp(1.0 - luminance(transmittance), 0.0, 1.0), multi_power);
    radiance += scattered * sun_radiance * (multi_strength * multi_depth);

    if hits_ground {
        let ground_pos = origin + view_dir * t_end;
        let ground_normal = normalize(ground_pos);
        let n_dot_l = max(dot(ground_normal, sun_dir), 0.0);
        if n_dot_l > 0.0 {
            let ground_trans = transmittance_to_sun(
                ground_pos,
                sun_dir,
                planet_radius,
                atmosphere_radius,
            );
            let ground_radiance = ground_albedo * sun_radiance * ground_trans * (n_dot_l / PI);
            radiance += ground_radiance * transmittance;
        }
    }

    if dot(view_dir, sun_dir) > shader_constants.sun_angular_radius_cos {
        radiance += sun_radiance * transmittance;
    }

    radiance += night_ambient;
    return radiance * constants.env_intensity;
}

fn trace_shadow(origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> bool {
    if constants.tlas_node_count == 0u {
        return false;
    }
    let bias = max(constants.shadow_bias, EPSILON);
    let max_dist = max(max_t - bias, 0.0);
    let inv_dir = 1.0 / dir;
    var stack: array<u32, MAX_STACK>;
    var stack_ptr = 0u;
    stack[stack_ptr] = 0u;
    stack_ptr = stack_ptr + 1u;

    loop {
        if stack_ptr == 0u {
            break;
        }
        stack_ptr = stack_ptr - 1u;
        let node_index = stack[stack_ptr];
        if node_index >= constants.tlas_node_count {
            continue;
        }
        let node = tlas_nodes[node_index];
        let hit = intersect_aabb(origin, inv_dir, node.bounds_min.xyz, node.bounds_max.xyz);
        if hit.y < max(hit.x, 0.0) || hit.x > max_dist {
            continue;
        }
        if node.index_count > 0u {
            for (var i = 0u; i < node.index_count; i = i + 1u) {
                let instance_index = tlas_indices[node.first_index + i];
                if instance_index >= constants.instance_count {
                    continue;
                }
                let inst = instances[instance_index];
                if inst.blas_index >= constants.blas_desc_count {
                    continue;
                }
                let blas = blas_descs[inst.blas_index];
                if blas.node_count == 0u {
                    continue;
                }
                let local_origin = (inst.inv_model * vec4(origin, 1.0)).xyz;
                let local_dir = (inst.inv_model * vec4(dir, 0.0)).xyz;
                let inv_local_dir = 1.0 / local_dir;

                var blas_stack: array<u32, MAX_STACK>;
                var blas_ptr = 0u;
                blas_stack[blas_ptr] = blas.node_offset;
                blas_ptr = blas_ptr + 1u;
                loop {
                    if blas_ptr == 0u {
                        break;
                    }
                    blas_ptr = blas_ptr - 1u;
                    let blas_index = blas_stack[blas_ptr];
                    let bnode = blas_nodes[blas_index];
                    let bhit = intersect_aabb(local_origin, inv_local_dir, bnode.bounds_min.xyz, bnode.bounds_max.xyz);
                    if bhit.y < max(bhit.x, 0.0) || bhit.x > max_dist {
                        continue;
                    }
                    if bnode.index_count > 0u {
                        for (var j = 0u; j < bnode.index_count; j = j + 1u) {
                            let tri_idx = blas_indices[bnode.first_index + j];
                            if tri_idx < blas.tri_offset || tri_idx >= blas.tri_offset + blas.tri_count {
                                continue;
                            }
                            let tri = blas_triangles[tri_idx];
                            let res = intersect_triangle(local_origin, local_dir, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz);
                            if res.x > 0.0 && res.x < max_dist {
                                return true;
                            }
                        }
                    } else {
                        if blas_ptr + 2u < MAX_STACK {
                            blas_stack[blas_ptr] = bnode.left;
                            blas_ptr = blas_ptr + 1u;
                            blas_stack[blas_ptr] = bnode.right;
                            blas_ptr = blas_ptr + 1u;
                        }
                    }
                }
            }
        } else {
            if stack_ptr + 2u < MAX_STACK {
                stack[stack_ptr] = node.left;
                stack_ptr = stack_ptr + 1u;
                stack[stack_ptr] = node.right;
                stack_ptr = stack_ptr + 1u;
            }
        }
    }
    return false;
}

fn trace_primary(origin: vec3<f32>, dir: vec3<f32>) -> HitInfo {
    if constants.tlas_node_count == 0u {
        return HitInfo(vec3(0.0), vec3(0.0), vec2(0.0), 0u, 0u, 0u, 0u);
    }
    let inv_dir = 1.0 / dir;
    var stack: array<u32, MAX_STACK>;
    var stack_ptr = 0u;
    stack[stack_ptr] = 0u;
    stack_ptr = stack_ptr + 1u;

    var best_t = 1.0e30;
    var best_normal = vec3<f32>(0.0);
    var best_uv = vec2<f32>(0.0);
    var best_material = 0u;
    var best_instance = 0u;
    var best_tri = 0u;
    let use_smooth = (constants.flags & RT_FLAG_SHADE_SMOOTH) != 0u;

    loop {
        if stack_ptr == 0u {
            break;
        }
        stack_ptr = stack_ptr - 1u;
        let node_index = stack[stack_ptr];
        if node_index >= constants.tlas_node_count {
            continue;
        }
        let node = tlas_nodes[node_index];
        let hit = intersect_aabb(origin, inv_dir, node.bounds_min.xyz, node.bounds_max.xyz);
        if hit.y < max(hit.x, 0.0) || hit.x > best_t {
            continue;
        }
        if node.index_count > 0u {
            for (var i = 0u; i < node.index_count; i = i + 1u) {
                let instance_index = tlas_indices[node.first_index + i];
                if instance_index >= constants.instance_count {
                    continue;
                }
                let inst = instances[instance_index];
                if inst.blas_index >= constants.blas_desc_count {
                    continue;
                }
                let blas = blas_descs[inst.blas_index];
                if blas.node_count == 0u {
                    continue;
                }
                let local_origin = (inst.inv_model * vec4(origin, 1.0)).xyz;
                let local_dir = (inst.inv_model * vec4(dir, 0.0)).xyz;
                let inv_local_dir = 1.0 / local_dir;
                let normal_matrix = transpose(inst.inv_model);

                var blas_stack: array<u32, MAX_STACK>;
                var blas_ptr = 0u;
                blas_stack[blas_ptr] = blas.node_offset;
                blas_ptr = blas_ptr + 1u;

                loop {
                    if blas_ptr == 0u {
                        break;
                    }
                    blas_ptr = blas_ptr - 1u;
                    let blas_index = blas_stack[blas_ptr];
                    let bnode = blas_nodes[blas_index];
                    let bhit = intersect_aabb(local_origin, inv_local_dir, bnode.bounds_min.xyz, bnode.bounds_max.xyz);
                    if bhit.y < max(bhit.x, 0.0) || bhit.x > best_t {
                        continue;
                    }
                    if bnode.index_count > 0u {
                        for (var j = 0u; j < bnode.index_count; j = j + 1u) {
                            let tri_idx = blas_indices[bnode.first_index + j];
                            if tri_idx < blas.tri_offset || tri_idx >= blas.tri_offset + blas.tri_count {
                                continue;
                            }
                            let tri = blas_triangles[tri_idx];
                            let res = intersect_triangle(local_origin, local_dir, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz);
                            if res.x > 0.0 && res.x < best_t {
                                best_t = res.x;
                                let w = 1.0 - res.y - res.z;
                                let uv = tri.uv0 * w + tri.uv1 * res.y + tri.uv2 * res.z;
                                let face_local = normalize(cross(tri.v1.xyz - tri.v0.xyz, tri.v2.xyz - tri.v0.xyz));
                                let face_world = normalize((normal_matrix * vec4(face_local, 0.0)).xyz);
                                var normal_world = face_world;
                                if use_smooth {
                                    let n0_world = normalize((normal_matrix * vec4(tri.n0.xyz, 0.0)).xyz);
                                    let n1_world = normalize((normal_matrix * vec4(tri.n1.xyz, 0.0)).xyz);
                                    let n2_world = normalize((normal_matrix * vec4(tri.n2.xyz, 0.0)).xyz);
                                    let smooth_sum = n0_world * w + n1_world * res.y + n2_world * res.z;
                                    if length(smooth_sum) > 1.0e-5 {
                                        normal_world = normalize(smooth_sum);
                                        if dot(normal_world, face_world) < 0.0 {
                                            normal_world = -normal_world;
                                        }
                                    }
                                }
                                best_normal = normal_world;
                                best_uv = uv;
                                best_material = inst.material_id;
                                best_instance = instance_index;
                                best_tri = tri_idx;
                            }
                        }
                    } else {
                        if blas_ptr + 2u < MAX_STACK {
                            blas_stack[blas_ptr] = bnode.left;
                            blas_ptr = blas_ptr + 1u;
                            blas_stack[blas_ptr] = bnode.right;
                            blas_ptr = blas_ptr + 1u;
                        }
                    }
                }
            }
        } else {
            if stack_ptr + 2u < MAX_STACK {
                stack[stack_ptr] = node.left;
                stack_ptr = stack_ptr + 1u;
                stack[stack_ptr] = node.right;
                stack_ptr = stack_ptr + 1u;
            }
        }
    }

    if best_t < 1.0e29 {
        let best_hit = origin + dir * best_t;
        return HitInfo(best_hit, best_normal, best_uv, best_material, best_instance, best_tri, 1u);
    }
    return HitInfo(vec3(0.0), vec3(0.0), vec2(0.0), 0u, 0u, 0u, 0u);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, EPSILON);
}

fn sample_ggx(r1: f32, r2: f32, alpha: f32) -> vec3<f32> {
    let a2 = alpha * alpha;
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt((1.0 - r2) / max(1.0 + (a2 - 1.0) * r2, EPSILON));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn ggx_pdf(n_dot_h: f32, v_dot_h: f32, alpha: f32) -> f32 {
    let D = distribution_ggx(n_dot_h, alpha);
    return D * n_dot_h / max(4.0 * v_dot_h, EPSILON);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
}

fn evaluate_direct_lighting(
    pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    roughness: f32,
    f0: vec3<f32>,
    diffuse_color: vec3<f32>,
    seed: ptr<function, u32>,
) -> vec3<f32> {
    if constants.light_count == 0u {
        return vec3<f32>(0.0);
    }
    let use_shadows = (constants.flags & RT_FLAG_SHADOWS) != 0u;
    let sample_count = max(1u, constants.direct_light_samples);
    let inv_samples = 1.0 / f32(sample_count);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let alpha = max(roughness * roughness, 1.0e-4);
    var result = vec3<f32>(0.0);
    var s = 0u;
    loop {
        if s >= sample_count {
            break;
        }
        let light_idx =
            min(u32(rand(seed) * f32(constants.light_count)), constants.light_count - 1u);
        let light = lights[light_idx];
        var L = vec3<f32>(0.0);
        var radiance = vec3<f32>(0.0);
        var max_t = 1.0e30;
        if light.light_type == 0u {
            L = normalize(-light.direction);
            radiance = light.color * light.intensity;
            max_t = 1.0e30;
        } else {
            let to_light = light.position - pos;
            let dist_sq = dot(to_light, to_light);
            if dist_sq < EPSILON {
                s = s + 1u;
                continue;
            }
            let dist = sqrt(dist_sq);
            L = to_light / dist;
            radiance = light.color * light.intensity / max(dist_sq, 1.0);
            max_t = dist;
        }
        let n_dot_l = max(dot(normal, L), 0.0);
        if n_dot_l > 0.0 {
            var shadowed = false;
            if use_shadows {
                shadowed = trace_shadow(pos + normal * constants.shadow_bias, L, max_t);
            }
            if !shadowed {
                let H = normalize(L + view_dir);
                let n_dot_h = max(dot(normal, H), 0.0);
                let v_dot_h = max(dot(view_dir, H), 0.0);
                let F = fresnel_schlick(v_dot_h, f0);
                let D = distribution_ggx(n_dot_h, alpha);
                let G = geometry_smith(n_dot_v, n_dot_l, roughness);
                let spec = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, EPSILON);
                let diff = diffuse_color / PI;
                result += (diff + spec) * radiance * n_dot_l;
            }
        }
        s = s + 1u;
    }
    return result * inv_samples;
}

fn reconstruct_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world = camera.inverse_view_projection_matrix * ndc;
    return world.xyz / world.w;
}

@compute @workgroup_size(8, 8, 1)
fn rt_reflections(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_width = max(u32(reflection_params.params0.z), 1u);
    let out_height = max(u32(reflection_params.params0.w), 1u);
    if gid.x >= out_width || gid.y >= out_height {
        return;
    }

    let output_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let output_dims = vec2<f32>(f32(out_width), f32(out_height));
    let gbuffer_dims = textureDimensions(depth_tex);
    let gbuffer_dims_f = vec2<f32>(f32(gbuffer_dims.x), f32(gbuffer_dims.y));
    let uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / output_dims;
    let gbuffer_coord_f = uv * gbuffer_dims_f;
    let gbuffer_dims_i = vec2<i32>(i32(gbuffer_dims.x), i32(gbuffer_dims.y));
    let max_coord = max(gbuffer_dims_i - vec2<i32>(1), vec2<i32>(0));
    let gbuffer_coord = clamp(vec2<i32>(gbuffer_coord_f), vec2<i32>(0), max_coord);

    let depth = textureLoad(depth_tex, gbuffer_coord, 0).x;
    if depth <= 0.0 || depth >= 1.0 {
        textureStore(reflection_out, output_coord, vec4<f32>(0.0));
        return;
    }

    let normal_sample = textureLoad(gbuf_normal, gbuffer_coord, 0).xyz;
    var normal = normal_sample * 2.0 - 1.0;
    if length(normal) < 1.0e-3 {
        normal = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        normal = normalize(normal);
    }
    let mra = textureLoad(gbuf_mra, gbuffer_coord, 0);
    let surface_roughness = clamp(mra.g, constants.min_roughness, 1.0);
    let fade = 1.0 - smoothstep(
        shader_constants.ssr_roughness_fade_start,
        shader_constants.ssr_roughness_fade_end,
        surface_roughness,
    );
    if fade <= 0.0 {
        textureStore(reflection_out, output_coord, vec4<f32>(0.0));
        return;
    }

    let gbuffer_uv = (vec2<f32>(gbuffer_coord) + vec2<f32>(0.5)) / gbuffer_dims_f;
    let world_pos = reconstruct_world(gbuffer_uv, depth);
    let view_dir = normalize(camera.view_position - world_pos);

    let history_weight = clamp(reflection_params.params0.x, 0.0, 1.0);
    let history_depth_threshold = max(reflection_params.params0.y, 0.0);
    let interleave = max(reflection_params.params1.x, 1u);
    let sample_count = max(reflection_params.params1.y, 1u);
    let direct_samples = max(reflection_params.params1.z, 1u);
    let max_accumulation_frames = reflection_params.params1.w;
    let flags = reflection_params.params2.x;
    let use_direct = (flags & RT_FLAG_DIRECT_LIGHTING) != 0u;
    let use_shadows = (flags & RT_FLAG_SHADOWS) != 0u;
    let ray_bias = max(constants.ray_bias, EPSILON);
    let shadow_bias = max(constants.shadow_bias, EPSILON);

    let tangent = make_tangent(normal);
    let bitangent = normalize(cross(normal, tangent));
    let alpha = max(surface_roughness * surface_roughness, 1.0e-4);

    var history_valid = false;
    var history_coord = output_coord;
    if history_weight > 0.0 {
        let prev_clip = camera.prev_view_proj * vec4<f32>(world_pos, 1.0);
        if abs(prev_clip.w) > EPSILON {
            let prev_ndc = prev_clip.xyz / prev_clip.w;
            let prev_uv = vec2<f32>(
                prev_ndc.x * 0.5 + 0.5,
                1.0 - (prev_ndc.y * 0.5 + 0.5),
            );
            let in_bounds = prev_uv.x >= 0.0
                && prev_uv.x <= 1.0
                && prev_uv.y >= 0.0
                && prev_uv.y <= 1.0;
            let depth_delta = abs(prev_ndc.z - depth);
            if in_bounds && (history_depth_threshold == 0.0 || depth_delta <= history_depth_threshold)
            {
                let max_out_coord = max(
                    vec2<i32>(i32(out_width) - 1, i32(out_height) - 1),
                    vec2<i32>(0),
                );
                history_coord = clamp(
                    vec2<i32>(prev_uv * output_dims),
                    vec2<i32>(0),
                    max_out_coord,
                );
                history_valid = true;
            }
        }
    }

    if interleave > 1u {
        let phase = constants.rng_frame_index % interleave;
        if ((gid.x + gid.y + phase) % interleave) != 0u {
            if history_valid && history_weight > 0.0 {
                let prev = textureLoad(history_tex, history_coord, 0);
                var out_color = prev.xyz;
                let alpha_out = max(prev.a, fade);
                textureStore(reflection_out, output_coord, vec4<f32>(out_color, alpha_out));
                return;
            }
        }
    }

    var seed = wang_hash(gid.x * 1973u + gid.y * 9277u + constants.rng_frame_index * 26699u);
    var radiance = vec3<f32>(0.0);
    var sample = 0u;

    loop {
        if sample >= sample_count {
            break;
        }

        var dir = reflect(-view_dir, normal);
        if surface_roughness > constants.min_roughness {
            let r1 = rand(&seed);
            let r2 = rand(&seed);
            let h_local = sample_ggx(r1, r2, alpha);
            let H = normalize(tangent * h_local.x + bitangent * h_local.y + normal * h_local.z);
            dir = normalize(reflect(-view_dir, H));
        }
        if dot(normal, dir) <= 0.0 {
            sample = sample + 1u;
            continue;
        }

        let origin = world_pos + normal * ray_bias;
        let hit = trace_primary(origin, dir);
        var sample_radiance = vec3<f32>(0.0);
        if hit.hit == 0u {
            sample_radiance = sample_environment(dir);
        } else {
            let hit_pos = hit.pos;
            let normal_hit = hit.normal;
            var safe_normal = select(
                normalize(normal_hit),
                vec3<f32>(0.0, 1.0, 0.0),
                length(normal_hit) < 1.0e-3,
            );
            if dot(safe_normal, dir) > 0.0 {
                safe_normal = -safe_normal;
            }

            let mat = materials[hit.material_id];
            var albedo = mat.albedo.xyz;
            var metallic = mat.metallic;
            var roughness = mat.roughness;
            var ao = mat.ao;
            let emission = mat.emission_color * mat.emission_strength;

            roughness = clamp(roughness, constants.min_roughness, 1.0);
            var shading_normal = safe_normal;
            if dot(shading_normal, dir) > 0.0 {
                shading_normal = -shading_normal;
            }

            let V = normalize(-dir);
            let NdotV = max(dot(shading_normal, V), 0.0);
            let alpha_hit = max(roughness * roughness, 1.0e-4);
            let f0 = mix(vec3<f32>(0.04), albedo, metallic);
            let diffuse_color = albedo * (1.0 - metallic) * ao;

            var direct = vec3<f32>(0.0);
            if use_direct && constants.light_count > 0u {
                let inv_samples = 1.0 / f32(direct_samples);
                var s = 0u;
                loop {
                    if s >= direct_samples {
                        break;
                    }
                    let light_idx = min(
                        u32(rand(&seed) * f32(constants.light_count)),
                        constants.light_count - 1u,
                    );
                    let light = lights[light_idx];
                    var L = vec3<f32>(0.0);
                    var radiance_light = vec3<f32>(0.0);
                    var max_t = 1.0e30;
                    if light.light_type == 0u {
                        L = normalize(-light.direction);
                        radiance_light = light.color * light.intensity;
                    } else {
                        let to_light = light.position - hit_pos;
                        let dist_sq = dot(to_light, to_light);
                        if dist_sq > EPSILON {
                            let dist = sqrt(dist_sq);
                            L = to_light / dist;
                            max_t = dist;
                            radiance_light = light.color * light.intensity / max(dist_sq, 1.0);
                        }
                    }
                    let NdotL = max(dot(shading_normal, L), 0.0);
                    let shadowed = use_shadows
                        && trace_shadow(hit_pos + shading_normal * shadow_bias, L, max_t);
                    if NdotL > 0.0 && !shadowed {
                        let H = normalize(L + V);
                        let NdotH = max(dot(shading_normal, H), 0.0);
                        let VdotH = max(dot(V, H), 0.0);
                        let F = fresnel_schlick(VdotH, f0);
                        let D = distribution_ggx(NdotH, alpha_hit);
                        let G = geometry_smith(NdotV, NdotL, roughness);
                        let spec = (D * G * F) / max(4.0 * NdotV * NdotL, EPSILON);
                        let diff = diffuse_color / PI;
                        direct = direct + (diff + spec) * radiance_light * NdotL;
                    }
                    s = s + 1u;
                }
                direct = direct * f32(constants.light_count) * inv_samples;
            }

            sample_radiance = direct + emission;
        }

        if constants.firefly_clamp > 0.0 {
            let max_comp = max_component(sample_radiance);
            if max_comp > constants.firefly_clamp {
                sample_radiance =
                    sample_radiance * (constants.firefly_clamp / max(max_comp, EPSILON));
            }
        }
        radiance = radiance + sample_radiance;
        sample = sample + 1u;
    }

    radiance = radiance / f32(sample_count);
    radiance = radiance * constants.exposure;
    var out_color = radiance;
    if constants.reset == 0u && history_valid && history_weight > 0.0 {
        let prev = textureLoad(history_tex, history_coord, 0);
        var capped_frame = constants.accumulation_frame;
        if max_accumulation_frames > 0u {
            capped_frame = min(constants.accumulation_frame, max_accumulation_frames - 1u);
        }
        let denom = f32(capped_frame + 1u);
        let avg = (prev.xyz * f32(capped_frame) + radiance) / denom;
        out_color = mix(radiance, avg, history_weight);
    }
    out_color = max(out_color, vec3<f32>(0.0));
    textureStore(reflection_out, output_coord, vec4<f32>(out_color, fade));
}
