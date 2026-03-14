//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const MAX_SHADOW_CASCADES: u32 = 4u;
const EPSILON: f32 = 0.00001;
const MAX_REFLECTION_LOD: f32 = 4.0;
const EMISSIVE_THRESHOLD: f32 = 0.01;

//=============== STRUCTS ===============//
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

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
}

struct InstanceInput {
    @location(6) model_matrix_col_0: vec4<f32>,
    @location(7) model_matrix_col_1: vec4<f32>,
    @location(8) model_matrix_col_2: vec4<f32>,
    @location(9) model_matrix_col_3: vec4<f32>,
    @location(10) skin_offset: u32,
    @location(11) skin_count: u32,
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
    _pad_light: vec4<u32>,
    prev_view_proj: mat4x4<f32>,
    frame_index: u32,
    _padding: vec3<u32>,
    _pad_end: vec4<u32>,
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
struct ShadowUniforms {
    cascade_count: u32,
    _pad0: vec3<u32>,
    cascades: array<CascadeData, MAX_SHADOW_CASCADES>,
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

//=============== BINDINGS ===============//
// Scene data
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights_buffer: array<LightData>;
@group(0) @binding(2) var shadow_map: texture_2d_array<f32>;
@group(0) @binding(3) var shadow_sampler: sampler;
@group(0) @binding(4) var<uniform> shadow_uniforms: ShadowUniforms;
@group(0) @binding(5) var<uniform> sky: SkyUniforms;
@group(0) @binding(6) var<uniform> render_constants: Constants;

// Material data
@group(1) @binding(0) var<uniform> material: MaterialData;
@group(1) @binding(1) var albedo_textures: texture_2d_array<f32>;
@group(1) @binding(2) var normal_textures: texture_2d_array<f32>;
@group(1) @binding(3) var mr_textures: texture_2d_array<f32>;
@group(1) @binding(4) var texture_sampler: sampler;

// Atmosphere data
@group(2) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(1) var scattering_lut: texture_3d<f32>;
@group(2) @binding(2) var irradiance_lut: texture_2d<f32>;
@group(2) @binding(3) var atmosphere_sampler: sampler;
@group(2) @binding(4) var<uniform> atmosphere: AtmosphereParams;

//=============== UTILITY FUNCTIONS ===============//
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

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len < EPSILON {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return v / len;
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
    current_depth = exp(render_constants.evsm_c * (current_depth - 1.0));

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

//=============== VERTEX SHADER ===============//
@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Reconstruct model_matrix from instance input
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_col_0,
        instance.model_matrix_col_1,
        instance.model_matrix_col_2,
        instance.model_matrix_col_3
    );

    let world_position_vec4 = model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;
    out.clip_position = camera.projection_matrix * camera.view_matrix * world_position_vec4;

    let model_mat3 = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    );
    let normal_matrix = transpose(mat3_inverse(model_mat3));
    let tangent_world = normalize(model_mat3 * vertex.tangent.xyz);

    let N = normalize(normal_matrix * vertex.normal);
    let T = normalize(tangent_world - dot(tangent_world, N) * N);
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
    // --- Albedo Calculation ---
    let albedo_sample = textureSample(albedo_textures, texture_sampler, in.tex_coord, u32(material.albedo_idx));
    let albedo_color = albedo_sample.rgb * material.albedo.rgb;
    let alpha = albedo_sample.a * material.albedo.a;

    // --- Metallic/Roughness/AO Calculation ---
    let mr_sample = textureSample(mr_textures, texture_sampler, in.tex_coord, u32(material.metallic_roughness_idx));
    let ao = mr_sample.r * material.ao;
    let metallic = mr_sample.b * material.metallic;
    let roughness = max(mr_sample.g * material.roughness, MIN_ROUGHNESS);

    // --- Normal Calculation ---
    let smooth_normal = safe_normalize(in.world_normal);
    var geom_normal = smooth_normal;
    if render_constants.shade_smooth == 0u {
        var flat_normal = safe_normalize(cross(dpdx(in.world_position), dpdy(in.world_position)));
        if dot(flat_normal, smooth_normal) < 0.0 {
            flat_normal = -flat_normal;
        }
        geom_normal = flat_normal;
    }
    let tangent_space_normal =
        textureSample(normal_textures, texture_sampler, in.tex_coord, u32(material.normal_idx)).xyz
        * 2.0
        - 1.0;
    let T = safe_normalize(in.world_tangent - geom_normal * dot(geom_normal, in.world_tangent));
    var B = safe_normalize(cross(geom_normal, T));
    if dot(B, in.world_bitangent) < 0.0 {
        B = -B;
    }
    let tbn = mat3x3<f32>(T, B, geom_normal);
    let N = safe_normalize(tbn * tangent_space_normal);

    // Add emission
    let emission = material.emission_color * material.emission_strength;

    let emissive_intensity = max(max(emission.r, emission.g), emission.b);
    if emissive_intensity > EMISSIVE_THRESHOLD {
        var color = albedo_color + emission;
        let tonemapped = color / (color + vec3(1.0));
        // let gamma_corrected = pow(tonemapped, vec3(1.0 / 2.2)); // Handled by sRGB output
        return vec4(tonemapped, alpha);
    }

    let shade_mode = render_constants.shade_mode;
    if shade_mode == 1u {
        var color = albedo_color + emission;
        let tonemapped = color / (color + vec3(1.0));
        // let gamma_corrected = pow(tonemapped, vec3(1.0 / 2.2));
        return vec4(tonemapped, alpha);
    }

    let is_lighting_only = shade_mode == 2u;
    let effective_albedo = select(albedo_color, vec3(1.0), is_lighting_only);
    let effective_metallic = metallic;

    // Basic material properties
    let V = normalize(camera.view_position - in.world_position);
    let R = reflect(-V, N);
    let F0 = mix(vec3<f32>(0.04), effective_albedo, effective_metallic);

    // Calculate lighting
    var Lo = vec3<f32>(0.0);

    let sun_height_factor = max(sky.sun_direction.y, 0.0);
    let sun_fade = pow(sun_height_factor, 1.5);

    let light_model = render_constants.light_model;
    let is_stylized = light_model == 1u;
    let is_simple = light_model == 2u;

    for (var i = 0u; i < camera.light_count; i = i + 1u) {
        let light = lights_buffer[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;
        var shadow_multiplier = 1.0;

        if light.light_type == 0u { // Directional
            L = normalize(-light.direction);
            radiance = light.color * light.intensity * sun_fade;

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
        if NdotL > 0.0 || is_stylized {
            let H = normalize(V + L);
            let NdotH = max(dot(N, H), 0.0);
            let HdotV = max(dot(H, V), 0.0);

            // Cook-Torrance BRDF
            let NDF = distribution_ggx(NdotH, roughness);
            let G = geometry_smith(N, V, L, roughness);
            let F = fresnel_schlick(HdotV, F0);

            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + EPSILON;
            let specular = select(numerator / denominator, vec3(0.0), is_simple);

            // Diffuse
            let kS = F;
            let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

            Lo += (kD * effective_albedo / PI + specular) * radiance * select(NdotL, 1.0, is_stylized) * shadow_multiplier;
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

        // Sample precomputed scattering for reflections
        let reflection_sky_color = get_irradiance(in.world_position, R) * 0.5 + get_transmittance(in.world_position, R) * 0.5;

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
        let sun_scatter = get_scattering_color(in.world_position, atmosphere.sun_direction);
        let sky_reflection = get_irradiance(in.world_position, biased_reflection);

        // Blend — 0.5 gives balanced, stylized but not sun-locked look
        let sky_specular = mix(sun_scatter, sky_reflection, 0.5);
        let sky_diffuse = get_irradiance(in.world_position, biased_normal) / PI;

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
    //let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2)); // Handled by sRGB output

    return vec4<f32>(tonemapped, alpha);
}
