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

struct DdgiGridConstants {
    origin: vec3<f32>,
    spacing: f32,
    counts: vec3<u32>,
    probe_resolution: u32,
    max_distance: f32,
    normal_bias: f32,
    hysteresis: f32,
    update_stride: u32,
    frame_index: u32,
    reset: u32,
    total_probes: u32,
    _pad0: u32,
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

@group(1) @binding(0) var<uniform> ddgi: DdgiGridConstants;

@group(2) @binding(0) var irradiance_tex: texture_2d_array<f32>;
@group(2) @binding(1) var irradiance_out: texture_storage_2d_array<rgba16float, write>;
@group(2) @binding(2) var distance_tex: texture_2d_array<f32>;
@group(2) @binding(3) var distance_out: texture_storage_2d_array<rgba16float, write>;

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

fn sample_environment(view_dir: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(sky.sun_direction);
    let sun_dot = dot(normalize(view_dir), sun_dir);
    let sun_visible = select(0.0, 1.0, sun_dot > sky.sun_angular_radius_cos);
    let sun = sky.sun_color * sky.sun_intensity * sun_visible;
    return sky.night_ambient_color + sun;
}

fn evaluate_direct_lighting(
    pos: vec3<f32>,
    normal: vec3<f32>,
    seed: ptr<function, u32>,
) -> vec3<f32> {
    if constants.light_count == 0u {
        return vec3<f32>(0.0);
    }
    let use_shadows = (constants.flags & RT_FLAG_SHADOWS) != 0u;
    let sample_count = max(1u, constants.direct_light_samples);
    let inv_samples = 1.0 / f32(sample_count);
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
            let attenuation = 1.0 / (dist_sq + 1.0);
            radiance = light.color * light.intensity * attenuation;
            max_t = dist;
        }
        let n_dot_l = max(dot(normal, L), 0.0);
        if n_dot_l > 0.0 {
            var shadowed = false;
            if use_shadows {
                shadowed = trace_shadow(pos + L * constants.shadow_bias, L, max_t);
            }
            if !shadowed {
                result += radiance * n_dot_l;
            }
        }
        s = s + 1u;
    }
    return result * inv_samples;
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e * 2.0 - 1.0, 1.0 - abs(e.x * 2.0 - 1.0) - abs(e.y * 2.0 - 1.0));
    if v.z < 0.0 {
        let x = (1.0 - abs(v.y)) * select(-1.0, 1.0, v.x >= 0.0);
        let y = (1.0 - abs(v.x)) * select(-1.0, 1.0, v.y >= 0.0);
        v.x = x;
        v.y = y;
    }
    return normalize(v);
}

@compute @workgroup_size(8, 8, 1)
fn probe_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= ddgi.probe_resolution || gid.y >= ddgi.probe_resolution {
        return;
    }
    if gid.z >= ddgi.total_probes {
        return;
    }

    let update_probe =
        ddgi.reset != 0u
        || ddgi.update_stride <= 1u
        || (gid.z % ddgi.update_stride) == (ddgi.frame_index % ddgi.update_stride);
    if !update_probe {
        let prev_irr = textureLoad(irradiance_tex, vec2<i32>(gid.xy), i32(gid.z), 0);
        let prev_dist = textureLoad(distance_tex, vec2<i32>(gid.xy), i32(gid.z), 0);
        textureStore(irradiance_out, vec2<i32>(gid.xy), i32(gid.z), prev_irr);
        textureStore(distance_out, vec2<i32>(gid.xy), i32(gid.z), prev_dist);
        return;
    }

    let probe_xy = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let uv = probe_xy / f32(ddgi.probe_resolution);
    let dir = oct_decode(uv);

    let probes_per_layer = ddgi.counts.x * ddgi.counts.y;
    let probe_x = gid.z % ddgi.counts.x;
    let probe_y = (gid.z / ddgi.counts.x) % ddgi.counts.y;
    let probe_z = gid.z / probes_per_layer;
    let probe_pos = ddgi.origin
        + vec3<f32>(f32(probe_x), f32(probe_y), f32(probe_z)) * ddgi.spacing;

    let ray_origin = probe_pos + dir * ddgi.normal_bias;
    let hit = trace_primary(ray_origin, dir);

    let use_direct = (constants.flags & RT_FLAG_DIRECT_LIGHTING) != 0u;

    var radiance = vec3<f32>(0.0);
    var distance = ddgi.max_distance;
    if hit.hit != 0u {
        let dist = length(hit.pos - probe_pos);
        if dist <= ddgi.max_distance {
            var seed = wang_hash(
                gid.z * 73856093u + gid.x * 19349663u + gid.y * 83492791u + ddgi.frame_index * 29791u,
            );
            let mat = materials[hit.material_id];
            var albedo = mat.albedo.xyz;
            var metallic = mat.metallic;
            var ao = mat.ao;
            var emission = mat.emission_color * mat.emission_strength;
            var safe_normal =
                select(normalize(hit.normal), vec3<f32>(0.0, 1.0, 0.0), length(hit.normal) < 1.0e-3);
            if dot(safe_normal, dir) > 0.0 {
                safe_normal = -safe_normal;
            }
            let diffuse_color = albedo * (1.0 - metallic) * ao;
            let direct =
                select(vec3<f32>(0.0), evaluate_direct_lighting(hit.pos, safe_normal, &seed), use_direct);
            radiance = direct * diffuse_color * (1.0 / PI) + emission;
            distance = dist;
        } else {
            radiance = sample_environment(dir);
        }
    } else {
        radiance = sample_environment(dir);
    }

    let prev_irr = textureLoad(irradiance_tex, vec2<i32>(gid.xy), i32(gid.z), 0);
    let prev_dist = textureLoad(distance_tex, vec2<i32>(gid.xy), i32(gid.z), 0);
    let blend = select(ddgi.hysteresis, 0.0, ddgi.reset != 0u);

    let new_irr = vec4<f32>(radiance, 1.0);
    let new_dist = vec4<f32>(distance, distance * distance, 0.0, 0.0);
    let final_irr = mix(new_irr, prev_irr, blend);
    let final_dist = mix(new_dist, prev_dist, blend);

    textureStore(irradiance_out, vec2<i32>(gid.xy), i32(gid.z), final_irr);
    textureStore(distance_out, vec2<i32>(gid.xy), i32(gid.z), final_dist);
}
