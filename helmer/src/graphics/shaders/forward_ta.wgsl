//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const NUM_CASCADES: u32 = 4u;
const EVSM_C = 20.0;
const EPSILON: f32 = 0.00001;
const MAX_REFLECTION_LOD: f32 = 4.0;
const EMISSIVE_THRESHOLD: f32 = 0.01;

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

// --- Sun Disk ---
const SUN_ANGULAR_RADIUS: f32 = 0.00465;
const SUN_DISK_THRESHOLD: f32 = 0.99996;

// --- Night Sky ---
const night_ambient_color: vec3<f32> = vec3(0.0002, 0.0004, 0.0008);

//=============== STRUCTS ===============//
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

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
    @location(5) view_z: f32,
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
}

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

struct ModelPushConstant {
    model_matrix: mat4x4<f32>,
}

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

//=============== BINDINGS ===============//
// Scene data
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights_buffer: array<LightData>;
@group(0) @binding(2) var shadow_map: texture_2d_array<f32>;
@group(0) @binding(3) var shadow_sampler: sampler;
@group(0) @binding(4) var<uniform> shadow_uniforms: array<CascadeData, NUM_CASCADES>;
@group(0) @binding(5) var<uniform> sky: SkyUniforms;
@group(0) @binding(6) var<uniform> render_constants: Constants;

// Material data
@group(1) @binding(0) var<uniform> material: MaterialData;
@group(1) @binding(1) var albedo_textures: texture_2d_array<f32>;
@group(1) @binding(2) var normal_textures: texture_2d_array<f32>;
@group(1) @binding(3) var mr_textures: texture_2d_array<f32>;
@group(1) @binding(4) var texture_sampler: sampler;

@group(2) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(1) var scattering_lut: texture_3d<f32>;
@group(2) @binding(2) var irradiance_lut: texture_2d<f32>;
@group(2) @binding(3) var atmosphere_sampler: sampler;
@group(2) @binding(4) var<uniform> atmosphere: AtmosphereParams;

@vertex var<push_constant> model: ModelPushConstant;

//=============== UTILITY FUNCTIONS ===============//
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len < EPSILON {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return v / len;
}

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if abs(det) < EPSILON {
        return mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    }

    let inv_det = 1.0 / det;
    var res: mat3x3<f32>;
    res[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det;
    res[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    res[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    res[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    res[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    res[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det;
    res[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det;
    res[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det;
    res[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det;
    return res;
}

//=============== PBR FUNCTIONS ===============//
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
    var cascade_index = i32(NUM_CASCADES - 1u);
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

fn ozone_density(height: f32) -> f32 {
    return max(0.0, 1.0 - abs(height - ozone_center_height) / ozone_falloff);
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

    // ==================== GROUND ====================
    if hit_ground {
        let ground_pos = camera_pos_world + view_dir * ray_len;
        let ground_norm = normalize(ground_pos);
        let mu_s = dot(ground_norm, atmosphere.sun_direction);
        if mu_s < -0.1 { return vec3<f32>(0.0); }

        // --- Direct Sun (ozone-aware) ---
        let sun_trans = get_transmittance_to_sun(ground_pos, atmosphere.sun_direction);
        let sun_rad = sky.sun_color * atmosphere.sun_intensity;
        let direct = ground_albedo * sun_rad * sun_trans * max(0.0, mu_s);

        // --- Sky Ambient: Use scattering LUT as irradiance ---
        // LUT already includes sun_intensity → treat as incoming radiance
        let view_up = -view_dir; // from ground to sky
        let scattered_incoming = get_scattering_color(ground_pos, view_up);

        // Lambertian diffuse: albedo * incoming / π
        let ambient = ground_albedo * scattered_incoming / PI * 1.5;

        // --- Sunset Horizon Glow (boost only near horizon) ---
        let horizon_factor = exp(-abs(atmosphere.sun_direction.y) * 5.0);
        let glow = ground_albedo * scattered_incoming * horizon_factor * 5.0; // no /π here — artistic

        let fade = smoothstep(-0.15, 0.0, mu_s);
        return (direct + ambient + glow) * fade;
    }
    // ==================== SKY ====================
    var color = get_scattering_color(camera_pos_world, view_dir);

    // SUN DISK
    let sun_cos = dot(view_dir, atmosphere.sun_direction);
    if sun_cos > SUN_ANGULAR_RADIUS_COS {
        let trans = get_transmittance(camera_pos_world, view_dir);
        let sun_rad = sky.sun_color * atmosphere.sun_intensity;

        // Limb darkening (brighter center)
        let t = (sun_cos - SUN_ANGULAR_RADIUS_COS) / (1.0 - SUN_ANGULAR_RADIUS_COS);
        let limb = 1.0 - 0.3 * (1.0 - sqrt(t));

        return sun_rad * trans * limb;
    }

    // Night ambient
    let night_factor = 1.0 - smoothstep(-0.2, 0.0, atmosphere.sun_direction.y);
    color += night_ambient_color * night_factor;

    return color;
}

//=============== VERTEX SHADER ===============//
@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_position_vec4 = model.model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;
    out.clip_position = camera.projection_matrix * camera.view_matrix * world_position_vec4;

    let model_mat3 = mat3x3<f32>(
        model.model_matrix[0].xyz,
        model.model_matrix[1].xyz,
        model.model_matrix[2].xyz
    );
    let normal_matrix = transpose(mat3_inverse(model_mat3));

    let N = safe_normalize(normal_matrix * vertex.normal);
    let T = safe_normalize(normal_matrix * vertex.tangent.xyz);
    let B = cross(N, T) * vertex.tangent.w;

    out.world_normal = N;
    out.world_tangent = T;
    out.world_bitangent = B;
    out.tex_coord = vertex.tex_coord;
    
    // Calculate view space z for cascade selection
    let view_pos = camera.view_matrix * world_position_vec4;
    out.view_z = view_pos.z;

    return out;
}

//=============== FRAGMENT SHADER ===============//
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // --- Albedo Calculation (FIXED) ---
    var albedo_color: vec3<f32>;
    var alpha: f32;
    if material.albedo_idx >= 0i {
        let albedo_sample = textureSample(albedo_textures, texture_sampler, in.tex_coord, u32(material.albedo_idx));
        albedo_color = albedo_sample.rgb * material.albedo.rgb;
        alpha = albedo_sample.a * material.albedo.a;
    } else {
        albedo_color = material.albedo.rgb;
        alpha = material.albedo.a;
    }

    // --- Metallic/Roughness/AO Calculation (FIXED) ---
    var ao: f32;
    var metallic: f32;
    var roughness: f32;
    if material.metallic_roughness_idx >= 0i {
        let mr_sample = textureSample(mr_textures, texture_sampler, in.tex_coord, u32(material.metallic_roughness_idx));
        ao = mr_sample.r * material.ao;
        metallic = mr_sample.b * material.metallic;
        roughness = max(mr_sample.g * material.roughness, MIN_ROUGHNESS);
    } else {
        ao = material.ao;
        metallic = material.metallic;
        roughness = max(material.roughness, MIN_ROUGHNESS);
    }

    // --- Normal Calculation (Correct as-is) ---
    var N: vec3<f32>;
    if material.normal_idx >= 0i {
        let tangent_space_normal = textureSample(normal_textures, texture_sampler, in.tex_coord, u32(material.normal_idx)).xyz * 2.0 - 1.0;
        let T = safe_normalize(in.world_tangent);
        let B = safe_normalize(in.world_bitangent);
        let N_geom = safe_normalize(in.world_normal);
        let tbn = mat3x3<f32>(T, B, N_geom);
        N = safe_normalize(tbn * tangent_space_normal);
    } else {
        N = safe_normalize(in.world_normal);
    }

    // Add emission
    let emission = material.emission_color * material.emission_strength;

    let emissive_intensity = max(max(emission.r, emission.g), emission.b);
    if emissive_intensity > EMISSIVE_THRESHOLD {
        var color = albedo_color + emission;
        let tonemapped = color / (color + vec3(1.0));
        let gamma_corrected = pow(tonemapped, vec3(1.0 / 2.2));
        return vec4(gamma_corrected, alpha);
    }

    let shade_mode = render_constants.shade_mode;
    if shade_mode == 1u {
        var color = albedo_color + emission;
        let tonemapped = color / (color + vec3(1.0));
        let gamma_corrected = pow(tonemapped, vec3(1.0 / 2.2));
        return vec4(gamma_corrected, alpha);
    }

    let is_lighting_only = shade_mode == 2u;
    let effective_albedo = select(albedo_color, vec3(1.0), is_lighting_only);
    let effective_metallic = metallic;

    // Basic material properties
    let V = safe_normalize(camera.view_position - in.world_position);
    let R = reflect(-V, N);
    let F0 = mix(vec3<f32>(0.04), albedo_color, metallic);

    // Calculate lighting
    var Lo = vec3<f32>(0.0);

    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;
        var shadow_multiplier = 1.0;

        if light.light_type == 0u { // Directional
            L = safe_normalize(-light.direction);
            radiance = light.color * light.intensity;

            let NdotL = max(dot(N, L), 0.0);
            let bias_amount = 0.5 + 1.0 * (1.0 - NdotL);
            let biased_world_position = in.world_position + N * bias_amount;
            shadow_multiplier = calculate_shadow_factor(biased_world_position, in.view_z, N, L);
        } else { // Point
            let to_light_vector = light.position - in.world_position;
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
            let HdotV = max(dot(H, V), 0.0);

            // Cook-Torrance BRDF
            let NDF = distribution_ggx(NdotH, roughness);
            let G = geometry_smith(N, V, L, roughness);
            let F = fresnel_schlick(HdotV, F0);

            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + EPSILON;
            let specular = numerator / denominator;

            // Diffuse
            let kS = F;
            let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

            Lo += (kD * effective_albedo / PI + specular) * radiance * NdotL * shadow_multiplier;
        }
    }

    // --- INDIRECT LIGHTING ---
    var ambient = vec3<f32>(0.0);
    var specular_indirect_occluded = vec3<f32>(0.0);

    let skylight_contribution = render_constants.skylight_contribution;

    if skylight_contribution == 1u { // FULL
        let sky_visibility = ao;
        let up = vec3<f32>(0.0, 1.0, 0.0);

        // Sample precomputed irradiance (incident light)
        let diffuse_sky_color = get_irradiance(in.world_position, N);

        // Sample sky color for reflections
        var reflection_sky_color = get_sky_color(in.world_position, R);

        let F_ambient = fresnel_schlick(max(dot(N, V), 0.0), F0);
        let kS_ambient = F_ambient * (1.0 - roughness * 0.7);
        let kD_ambient = (vec3<f32>(1.0) - kS_ambient) * (1.0 - effective_metallic);

        // Energy-conserving ambient (added / PI for correct diffuse energy, fixes grey/washed-out look)
        let diffuse_contribution = kD_ambient * effective_albedo * diffuse_sky_color / PI;
        let specular_contribution = kS_ambient * reflection_sky_color;

        ambient = diffuse_contribution * sky_visibility;
        specular_indirect_occluded = specular_contribution * sky_visibility;
    } else if skylight_contribution == 2u { // STYLIZED FULL
        var sky_visibility = ao;
        sky_visibility = mix(1.0, sky_visibility, 0.5);

        let up = vec3<f32>(0.0, 1.0, 0.0);

        // Slightly bias normals upward for a dreamy, painterly look
        let biased_normal = normalize(mix(N, up, 0.5));
        let biased_reflection = reflect(-V, biased_normal);

        // --- Blend between sun scattering and general sky irradiance ---
        let sky_diffuse = get_irradiance(in.world_position, biased_normal) / PI;

        // Use sky color for specular
        var sky_specular = get_sky_color(in.world_position, biased_reflection);

        // Fresnel & energy conservation
        var F_ambient = fresnel_schlick(max(dot(N, V), 0.0), F0);
        F_ambient = mix(F_ambient, vec3<f32>(0.04), 0.5);

        let kS = F_ambient * (1.0 - roughness * 0.7);
        let kD = (vec3<f32>(1.0) - kS) * (1.0 - effective_metallic);

        let diffuse_contribution = kD * effective_albedo * sky_diffuse;
        let specular_contribution = kS * sky_specular;

        ambient = diffuse_contribution * sky_visibility;
        specular_indirect_occluded = specular_contribution * sky_visibility;
    } else if skylight_contribution == 3u { // SIMPLE
        var sky_visibility = mix(1.0, ao, 0.5);
        let up = vec3<f32>(0.0, 1.0, 0.0);

        let flat_sky_color = get_irradiance(in.world_position, up) / PI; // Sample irradiance straight up

        let flat_kD = vec3(1.0) - effective_metallic;

        let diffuse_contribution = flat_kD * effective_albedo * flat_sky_color;

        ambient = diffuse_contribution * sky_visibility;
        specular_indirect_occluded = vec3(0.0);
    }
    
    // --- FINAL COMPOSITION ---
    var final_hdr_color = ambient + Lo + specular_indirect_occluded + select(emission, vec3<f32>(0.0), is_lighting_only);

    let tonemapped = final_hdr_color / (final_hdr_color + vec3<f32>(1.0));
    let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, alpha);
}