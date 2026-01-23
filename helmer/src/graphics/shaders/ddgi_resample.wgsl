const EPSILON: f32 = 1.0e-5;
const PI: f32 = 3.14159265;

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

struct DdgiResampleConstants {
    indirect_scale: f32,
    specular_scale: f32,
    specular_confidence: f32,
    reflection_roughness_start: f32,
    reflection_roughness_end: f32,
    temporal_weight: f32,
    spatial_weight: f32,
    reservoir_mix: f32,
    diffuse_samples: u32,
    specular_samples: u32,
    spatial_samples: u32,
    spatial_radius: u32,
    history_depth_threshold: f32,
    min_candidate_weight: f32,
    specular_cone_angle: f32,
    visibility_normal_bias: f32,
    visibility_spacing_bias: f32,
    visibility_max_bias: f32,
    visibility_receiver_bias: f32,
    visibility_variance_scale: f32,
    visibility_bleed_min: f32,
    visibility_bleed_max: f32,
    _pad0: f32,
}

struct Reservoir {
    data0: vec4<f32>,
    data1: vec4<f32>,
}

@group(0) @binding(0) var normal_tex: texture_2d<f32>;
@group(0) @binding(1) var mra_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var irradiance_tex: texture_2d_array<f32>;
@group(0) @binding(4) var distance_tex: texture_2d_array<f32>;
@group(0) @binding(5) var scene_sampler: sampler;
@group(0) @binding(6) var point_sampler: sampler;

@group(1) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(1) var<uniform> ddgi: DdgiGridConstants;
@group(1) @binding(2) var<uniform> params: DdgiResampleConstants;

@group(2) @binding(0) var<storage, read> reservoirs_in: array<Reservoir>;
@group(2) @binding(1) var<storage, read_write> reservoirs_out: array<Reservoir>;

@group(3) @binding(0) var diffuse_out: texture_storage_2d<rgba16float, write>;
@group(3) @binding(1) var spec_out: texture_storage_2d<rgba16float, write>;

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

fn build_tangent_basis(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

fn sample_cosine_hemisphere(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - u1));
    let basis = build_tangent_basis(n);
    return normalize(basis[0] * x + basis[1] * y + basis[2] * z);
}

fn sample_cone(dir: vec3<f32>, cos_theta_max: f32, u1: f32, u2: f32) -> vec3<f32> {
    let cos_theta = mix(cos_theta_max, 1.0, u1);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * u2;
    let x = cos(phi) * sin_theta;
    let y = sin(phi) * sin_theta;
    let z = cos_theta;
    let basis = build_tangent_basis(dir);
    return normalize(basis[0] * x + basis[1] * y + basis[2] * z);
}

fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    var v = n / (abs(n.x) + abs(n.y) + abs(n.z));
    if v.z < 0.0 {
        let x = (1.0 - abs(v.y)) * select(-1.0, 1.0, v.x >= 0.0);
        let y = (1.0 - abs(v.x)) * select(-1.0, 1.0, v.y >= 0.0);
        v.x = x;
        v.y = y;
    }
    return v.xy * 0.5 + 0.5;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn probe_index(coord: vec3<u32>) -> u32 {
    let plane = ddgi.counts.x * ddgi.counts.y;
    return coord.x + coord.y * ddgi.counts.x + coord.z * plane;
}

fn sample_probe_irradiance(probe: u32, dir: vec3<f32>) -> vec3<f32> {
    let uv = oct_encode(dir);
    return textureSampleLevel(irradiance_tex, scene_sampler, uv, i32(probe), 0.0).rgb;
}

fn sample_probe_distance_moments(probe: u32, dir: vec3<f32>) -> vec2<f32> {
    let uv = oct_encode(dir);
    return textureSampleLevel(distance_tex, point_sampler, uv, i32(probe), 0.0).xy;
}

fn probe_visibility(dist: f32, moments: vec2<f32>, n_dot: f32, spacing: f32) -> f32 {
    let mean = max(moments.x, 0.0);
    let mean_sq = max(moments.y, mean * mean);
    if mean <= EPSILON && mean_sq <= EPSILON {
        return 1.0;
    }
    let normal_bias = max(params.visibility_normal_bias, 0.0);
    let spacing_bias = max(params.visibility_spacing_bias, 0.0);
    let max_bias = max(params.visibility_max_bias, 0.0);
    let base_bias = max(normal_bias, spacing * spacing_bias);
    let vis_bias = min(base_bias, spacing * max_bias);
    let receiver_scale = max(params.visibility_receiver_bias, 0.0);
    let n_dot_sat = clamp(n_dot, 0.0, 1.0);
    let receiver_bias = vis_bias * receiver_scale * (1.0 - n_dot_sat);
    let dist_biased = max(dist - receiver_bias, 0.0);
    if dist_biased <= mean {
        return 1.0;
    }
    let variance = max(mean_sq - mean * mean, 0.0);
    let variance_scale = max(params.visibility_variance_scale, 0.0);
    let vis_term = vis_bias * variance_scale;
    let min_variance = max(variance, vis_term * vis_term);
    let delta = dist_biased - mean;
    let p_max = min_variance / (min_variance + delta * delta);
    let bleed_min = clamp(params.visibility_bleed_min, 0.0, 0.999);
    let bleed_max = clamp(params.visibility_bleed_max, 0.0, 0.999);
    let bleed_lo = min(bleed_min, bleed_max);
    let bleed_hi = max(bleed_min, bleed_max);
    let bleed = mix(bleed_lo, bleed_hi, 1.0 - n_dot_sat);
    let denom = max(1.0 - bleed, EPSILON);
    return clamp((p_max - bleed) / denom, 0.0, 1.0);
}

fn reconstruct_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world = camera.inverse_view_projection_matrix * ndc;
    return world.xyz / world.w;
}

fn parallax_hit(origin: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec4<f32> {
    let dir_sign = select(vec3<f32>(-1.0), vec3<f32>(1.0), dir >= vec3<f32>(0.0));
    let inv_dir = dir_sign / max(abs(dir), vec3<f32>(EPSILON));
    let t0 = (bmin - origin) * inv_dir;
    let t1 = (bmax - origin) * inv_dir;
    let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if tmax < 0.0 || tmin > tmax {
        return vec4<f32>(origin + dir, 0.0);
    }
    let t = select(tmin, tmax, tmin < 0.0);
    return vec4<f32>(origin + dir * t, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn ddgi_resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(textureDimensions(depth_tex));
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let depth = textureLoad(depth_tex, coord, 0).x;
    let idx = gid.y * dims.x + gid.x;

    if depth <= 0.0 || depth >= 1.0 || ddgi.total_probes == 0u {
        reservoirs_out[idx].data0 = vec4<f32>(0.0);
        reservoirs_out[idx].data1 = vec4<f32>(0.0);
        textureStore(diffuse_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(spec_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let normal = normalize(textureLoad(normal_tex, coord, 0).xyz * 2.0 - 1.0);
    let mra = textureLoad(mra_tex, coord, 0);
    let roughness = mra.g;
    let uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / vec2<f32>(dims);
    let world_pos = reconstruct_world(uv, depth);
    let view_dir = normalize(camera.view_position - world_pos);
    let refl_dir = reflect(-view_dir, normal);

    let spacing = max(ddgi.spacing, EPSILON);
    let max_dist = max(ddgi.max_distance, spacing);
    let rel = (world_pos - ddgi.origin) / spacing;
    let counts_f = vec3<f32>(f32(ddgi.counts.x), f32(ddgi.counts.y), f32(ddgi.counts.z));
    let volume_half = vec3<f32>(spacing * 0.5);
    let volume_min = ddgi.origin - volume_half;
    let volume_max = ddgi.origin + (counts_f - vec3<f32>(1.0)) * spacing + volume_half;
    let parallax = parallax_hit(world_pos, refl_dir, volume_min, volume_max);
    let use_parallax = parallax.w > 0.5;
    let max_cell = max(counts_f - vec3<f32>(1.0), vec3<f32>(0.0));
    let cell_f = clamp(floor(rel), vec3<f32>(0.0), max_cell);
    let base = vec3<u32>(cell_f);
    let frac = clamp(rel - cell_f, vec3<f32>(0.0), vec3<f32>(1.0));
    let max_coord = vec3<u32>(
        max(ddgi.counts.x, 1u) - 1u,
        max(ddgi.counts.y, 1u) - 1u,
        max(ddgi.counts.z, 1u) - 1u,
    );

    var seed = wang_hash(gid.x * 1973u + gid.y * 9277u + ddgi.frame_index * 26699u);
    let diffuse_samples = params.diffuse_samples;
    let specular_samples = params.specular_samples;
    let spatial_samples = params.spatial_samples;
    let spatial_radius = i32(params.spatial_radius);
    let reservoir_mix = clamp(params.reservoir_mix, 0.0, 1.0);

    var res_radiance = vec3<f32>(0.0);
    var res_weight_sum = 0.0;
    var res_m = 0.0;
    var avg_sum = vec3<f32>(0.0);
    var avg_weight = 0.0;

    for (var z = 0u; z < 2u; z = z + 1u) {
        for (var y = 0u; y < 2u; y = y + 1u) {
            for (var x = 0u; x < 2u; x = x + 1u) {
                let coord_u = vec3<u32>(
                    min(base.x + x, max_coord.x),
                    min(base.y + y, max_coord.y),
                    min(base.z + z, max_coord.z),
                );
                let probe = probe_index(coord_u);
                let probe_pos =
                    ddgi.origin + vec3<f32>(f32(coord_u.x), f32(coord_u.y), f32(coord_u.z)) * spacing;
                let to_surface = world_pos - probe_pos;
                let dist = length(to_surface);
                if dist <= EPSILON {
                    continue;
                }
                let dir_to_surface = to_surface / dist;
                let dir_weight = max(dot(normal, -dir_to_surface), 0.0);
                if dir_weight <= 0.0 {
                    continue;
                }
                let dist_moments = sample_probe_distance_moments(probe, dir_to_surface);
                let visibility = probe_visibility(dist, dist_moments, dir_weight, spacing);
                let distance_weight = max(0.0, 1.0 - dist / max_dist);
                let wx = select(1.0 - frac.x, frac.x, x == 1u);
                let wy = select(1.0 - frac.y, frac.y, y == 1u);
                let wz = select(1.0 - frac.z, frac.z, z == 1u);
                let grid_weight = wx * wy * wz;
                let weight = grid_weight * visibility * distance_weight * dir_weight;
                if weight <= 0.0 {
                    continue;
                }
                var irradiance = vec3<f32>(0.0);
                for (var s = 0u; s < diffuse_samples; s = s + 1u) {
                    let u1 = rand(&seed);
                    let u2 = rand(&seed);
                    let sample_dir = sample_cosine_hemisphere(normal, u1, u2);
                    irradiance += sample_probe_irradiance(probe, sample_dir);
                }
                irradiance *= PI / f32(diffuse_samples);
                avg_sum += irradiance * weight;
                avg_weight += weight;

                let candidate_weight = max(luminance(irradiance) * weight, params.min_candidate_weight);
                res_m += 1.0;
                res_weight_sum += candidate_weight;
                if rand(&seed) < candidate_weight / res_weight_sum {
                    res_radiance = irradiance;
                }
            }
        }
    }

    if ddgi.reset == 0u && params.temporal_weight > 0.0 {
        let prev_clip = camera.prev_view_proj * vec4<f32>(world_pos, 1.0);
        if prev_clip.w > EPSILON {
            let prev_uv = prev_clip.xy / prev_clip.w * vec2<f32>(0.5, -0.5) + 0.5;
            if all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
                let prev_clamped = clamp(
                    prev_uv * vec2<f32>(dims),
                    vec2<f32>(0.0),
                    vec2<f32>(f32(dims.x - 1u), f32(dims.y - 1u)),
                );
                let prev_coord = vec2<u32>(prev_clamped);
                let prev_idx = prev_coord.y * dims.x + prev_coord.x;
                let prev = reservoirs_in[prev_idx];
                let prev_depth = prev.data1.x;
                let prev_m = max(prev.data1.y, 0.0);
                let depth_ok = abs(prev_depth - depth) < params.history_depth_threshold;
                if prev_m > 0.0 && depth_ok {
                    let prev_radiance = prev.data0.xyz;
                    let prev_weight = max(prev.data0.w / max(prev_m, 1.0), 0.0) * params.temporal_weight;
                    if prev_weight > 0.0 {
                        res_m += 1.0;
                        res_weight_sum += prev_weight;
                        if rand(&seed) < prev_weight / res_weight_sum {
                            res_radiance = prev_radiance;
                        }
                    }
                }
            }
        }
    }

    if ddgi.reset == 0u && params.spatial_weight > 0.0 && spatial_samples > 0u {
        let radius = max(spatial_radius, 0);
        for (var i = 0u; i < spatial_samples; i = i + 1u) {
            let rx = i32(rand(&seed) * f32(radius * 2 + 1)) - radius;
            let ry = i32(rand(&seed) * f32(radius * 2 + 1)) - radius;
            let neighbor = vec2<i32>(i32(gid.x) + rx, i32(gid.y) + ry);
            let clamped = vec2<u32>(
                u32(clamp(neighbor.x, 0, i32(dims.x) - 1)),
                u32(clamp(neighbor.y, 0, i32(dims.y) - 1)),
            );
            let n_idx = clamped.y * dims.x + clamped.x;
            let n_res = reservoirs_in[n_idx];
            let n_m = max(n_res.data1.y, 0.0);
            if n_m > 0.0 {
                let n_radiance = n_res.data0.xyz;
                let n_weight =
                    max(n_res.data0.w / max(n_m, 1.0), 0.0) * params.spatial_weight;
                if n_weight > 0.0 {
                    res_m += 1.0;
                    res_weight_sum += n_weight;
                    if rand(&seed) < n_weight / res_weight_sum {
                        res_radiance = n_radiance;
                    }
                }
            }
        }
    }

    if res_weight_sum <= 0.0 {
        if avg_weight > 0.0 {
            res_radiance = avg_sum / avg_weight;
            res_weight_sum = avg_weight;
            res_m = 1.0;
        }
    }

    let final_weight = res_weight_sum / max(res_m, 1.0);
    let avg_radiance = select(vec3<f32>(0.0), avg_sum / avg_weight, avg_weight > 0.0);
    let reservoir_radiance = res_radiance * final_weight;
    let diffuse = max(
        mix(avg_radiance, reservoir_radiance, reservoir_mix) * params.indirect_scale,
        vec3<f32>(0.0),
    );

    let roughness_weight = smoothstep(
        params.reflection_roughness_start,
        params.reflection_roughness_end,
        roughness,
    );
    let max_cone_angle = min(params.specular_cone_angle, PI * 0.5);
    let cone_angle = roughness_weight * roughness_weight * max_cone_angle;
    let cos_theta_max = cos(cone_angle);

    var spec_sum = vec3<f32>(0.0);
    var spec_weight = 0.0;
    for (var z = 0u; z < 2u; z = z + 1u) {
        for (var y = 0u; y < 2u; y = y + 1u) {
            for (var x = 0u; x < 2u; x = x + 1u) {
                let coord_u = vec3<u32>(
                    min(base.x + x, max_coord.x),
                    min(base.y + y, max_coord.y),
                    min(base.z + z, max_coord.z),
                );
                let probe = probe_index(coord_u);
                let probe_pos =
                    ddgi.origin + vec3<f32>(f32(coord_u.x), f32(coord_u.y), f32(coord_u.z)) * spacing;
                let to_surface = world_pos - probe_pos;
                let dist = length(to_surface);
                if dist <= EPSILON {
                    continue;
                }
                let dir_to_surface = to_surface / dist;
                let dir_weight = max(dot(normal, -dir_to_surface), 0.0);
                if dir_weight <= 0.0 {
                    continue;
                }
                let dist_moments = sample_probe_distance_moments(probe, dir_to_surface);
                let visibility = probe_visibility(dist, dist_moments, dir_weight, spacing);
                let distance_weight = max(0.0, 1.0 - dist / max_dist);
                let wx = select(1.0 - frac.x, frac.x, x == 1u);
                let wy = select(1.0 - frac.y, frac.y, y == 1u);
                let wz = select(1.0 - frac.z, frac.z, z == 1u);
                let grid_weight = wx * wy * wz;
                let weight = grid_weight * visibility * distance_weight * dir_weight;
                if weight <= 0.0 {
                    continue;
                }
                var sample_dir = refl_dir;
                if use_parallax {
                    let dir_to_hit = parallax.xyz - probe_pos;
                    let dir_len = length(dir_to_hit);
                    if dir_len > EPSILON {
                        sample_dir = dir_to_hit / dir_len;
                    }
                }
                var radiance = vec3<f32>(0.0);
                for (var s = 0u; s < specular_samples; s = s + 1u) {
                    let u1 = rand(&seed);
                    let u2 = rand(&seed);
                    let lobe_dir = sample_cone(sample_dir, cos_theta_max, u1, u2);
                    radiance += sample_probe_irradiance(probe, lobe_dir);
                }
                radiance /= f32(specular_samples);
                spec_sum += radiance * weight;
                spec_weight += weight;
            }
        }
    }

    let specular = select(
        vec3<f32>(0.0),
        spec_sum / max(spec_weight, params.min_candidate_weight),
        spec_weight > 0.0,
    );
    let spec_conf = clamp(
        spec_weight / max(spec_weight + params.min_candidate_weight, EPSILON),
        0.0,
        1.0,
    );
    let spec_alpha = clamp(roughness_weight * params.specular_confidence * spec_conf, 0.0, 1.0);
    let spec_color = max(specular * params.specular_scale, vec3<f32>(0.0));

    reservoirs_out[idx].data0 = vec4<f32>(res_radiance, res_weight_sum);
    reservoirs_out[idx].data1 = vec4<f32>(depth, res_m, 0.0, 0.0);

    textureStore(diffuse_out, coord, vec4<f32>(diffuse, 1.0));
    textureStore(spec_out, coord, vec4<f32>(spec_color, spec_alpha));
}
