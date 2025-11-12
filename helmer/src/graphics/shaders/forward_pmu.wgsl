//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MIN_ROUGHNESS: f32 = 0.04;
const NUM_CASCADES: u32 = 4u;
const EPSILON: f32 = 0.0001;
const MAX_REFLECTION_LOD: f32 = 4.0;
const EMISSIVE_THRESHOLD: f32 = 0.01;

const POISSON_DISK_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2(-0.94201624, -0.39906216),
    vec2(0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845),
    vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554),
    vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507),
    vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790)
);

//=============== STRUCTS ===============//

struct Constants {
    // general
    mip_bias: f32,

    // lighting
    shade_mode: u32,
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
    _final_padding: vec2<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct InstanceInput {
    @location(5) model_matrix_col_0: vec4<f32>,
    @location(6) model_matrix_col_1: vec4<f32>,
    @location(7) model_matrix_col_2: vec4<f32>,
    @location(8) model_matrix_col_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
    @location(5) view_space_depth: f32,
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
}

struct MaterialUniforms {
    albedo: vec4<f32>,
    emission_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,
    // No padding needed here, WGSL handles it
};

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

// GROUP 0: Scene-wide data
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> lights_buffer: array<LightData>;
@group(0) @binding(2) var shadow_map: texture_2d_array<f32>;
@group(0) @binding(3) var shadow_sampler: sampler;
@group(0) @binding(4) var<uniform> shadow_uniforms: array<CascadeData, NUM_CASCADES>;
@group(0) @binding(5) var<uniform> sky: SkyUniforms;
@group(0) @binding(6) var<uniform> render_constants: Constants;
// @group(0) @binding(7) var depth_tex: texture_depth_2d; // Not used in this shader
// @group(0) @binding(8) var scene_sampler: sampler; // Not used in this shader

// GROUP 1: Per-Material data
@group(1) @binding(0) var pbr_sampler: sampler;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_texture: texture_2d<f32>;
@group(1) @binding(3) var mr_texture: texture_2d<f32>;
@group(1) @binding(4) var emission_texture: texture_2d<f32>;
@group(1) @binding(5) var<uniform> material: MaterialUniforms;

// GROUP 2: Atmosphere
@group(2) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(1) var scattering_lut: texture_3d<f32>;
@group(2) @binding(2) var irradiance_lut: texture_2d<f32>;
@group(2) @binding(3) var atmosphere_sampler: sampler;
@group(2) @binding(4) var<uniform> atmosphere: AtmosphereParams;

// GROUP 3: Image-Based Lighting
@group(3) @binding(0) var brdf_lut: texture_2d<f32>;
@group(3) @binding(1) var irradiance_map: texture_cube<f32>;
@group(3) @binding(2) var prefiltered_env_map: texture_cube<f32>;
@group(3) @binding(3) var env_map_sampler: sampler;
@group(3) @binding(4) var brdf_lut_sampler: sampler;

//=============== UTILITY FUNCTIONS ===============//

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = determinant(m);
    if abs(det) < EPSILON {
        return mat3x3<f32>(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );
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

fn fresnel_schlick_roughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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

fn calculate_shadow_factor(
    world_pos: vec3<f32>,
    view_z: f32,
    N: vec3<f32>,
    L: vec3<f32>
) -> f32 {
    // 1. Determine cascade
    var cascade_index = i32(NUM_CASCADES - 1u);
    for (var i = 0i; i < i32(NUM_CASCADES); i = i + 1i) {
        if view_z > shadow_uniforms[i].split_depth.x {
            cascade_index = i;
            break;
        }
    }
    let cascade = shadow_uniforms[cascade_index];

    // 2. Project to shadow space
    let shadow_pos_clip = cascade.light_view_proj * vec4(world_pos, 1.0);
    if shadow_pos_clip.w < EPSILON {
        return 1.0;
    }

    let shadow_coord = shadow_pos_clip.xyz / shadow_pos_clip.w;
    let shadow_uv = vec2(shadow_coord.x * 0.5 + 0.5, shadow_coord.y * -0.5 + 0.5);

    // 3. Bounds check
    if any(shadow_uv < vec2(0.0)) || any(shadow_uv > vec2(1.0)) || shadow_coord.z < 0.0 || shadow_coord.z > 1.0 {
        return 1.0;
    }

    // 4. Calculate dynamic filter size
    let base_radius = f32(render_constants.pcf_radius);
    let dist_factor = clamp(view_z / render_constants.pcf_max_distance, 0.0, 1.0);
    let scale = mix(render_constants.pcf_min_scale, render_constants.pcf_max_scale, dist_factor);
    let filter_radius = base_radius * scale;

    // 5. Choose sample count based on quality settings
    // Use fewer samples for far cascades or distant geometry
    var sample_count = 16;
    if cascade_index > 1 || dist_factor > 0.7 {
        sample_count = 8; // Reduce samples for far/distant shadows
    }
    
    // 6. Sample setup - compute once
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map, 0u));
    let texel_size = filter_radius / shadow_map_size;

    var total_moments = vec2<f32>(0.0);
    
    // 7. Poisson disk sampling
    if sample_count == 16 {
        // 16-tap sampling
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[0] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[1] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[2] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[3] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[4] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[5] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[6] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[7] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[8] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[9] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[10] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[11] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[12] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[13] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[14] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[15] * texel_size, u32(cascade_index)).rg;
        total_moments *= (1.0 / 16.0);
    } else {
        // 8-tap sampling - every other sample
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[0] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[2] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[4] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[6] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[8] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[10] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[12] * texel_size, u32(cascade_index)).rg;
        total_moments += textureSample(shadow_map, shadow_sampler, shadow_uv + POISSON_DISK_16[14] * texel_size, u32(cascade_index)).rg;
        total_moments *= (1.0 / 8.0);
    }

    // 8. EVSM calculation
    return chebyshev_inequality(shadow_coord.z, total_moments, N, L);
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
    // Reconstruct model_matrix from instance input
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_col_0,
        instance.model_matrix_col_1,
        instance.model_matrix_col_2,
        instance.model_matrix_col_3
    );

    var out: VertexOutput;

    let world_position_vec4 = model_matrix * vec4<f32>(vertex.position, 1.0);
    out.world_position = world_position_vec4.xyz;

    let view_pos = camera.view_matrix * world_position_vec4;
    out.clip_position = camera.projection_matrix * view_pos;
    out.view_space_depth = view_pos.z;

    let model_mat3 = mat3x3<f32>(model_matrix[0].xyz, model_matrix[1].xyz, model_matrix[2].xyz);
    let normal_matrix = transpose(mat3_inverse(model_mat3));
    out.world_normal = normalize(normal_matrix * vertex.normal);

    let tangent_world = normalize(model_mat3 * vertex.tangent.xyz);
    out.world_tangent = normalize(tangent_world - dot(tangent_world, out.world_normal) * out.world_normal);
    out.world_bitangent = cross(out.world_normal, out.world_tangent) * vertex.tangent.w;
    out.tex_coord = vertex.tex_coord;
    return out;
}

//=============== FRAGMENT SHADER ===============//
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo_sample = textureSampleBias(albedo_texture, pbr_sampler, in.tex_coord, render_constants.mip_bias);
    let albedo = albedo_sample.rgb * material.albedo.rgb;
    let alpha = albedo_sample.a * material.albedo.a;

    let mr_sample = textureSampleBias(mr_texture, pbr_sampler, in.tex_coord, render_constants.mip_bias);
    let metallic = mr_sample.b * material.metallic;
    let roughness = max(mr_sample.g * material.roughness, MIN_ROUGHNESS);
    var ao = mr_sample.r * material.ao;
    ao = mix(ao, 1.0, 1.0 - smoothstep(0.0, 0.1, ao));

    var emission = material.emission_color * material.emission_strength;
    emission *= textureSampleBias(emission_texture, pbr_sampler, in.tex_coord, render_constants.mip_bias).rgb;

    let emissive_intensity = max(max(emission.r, emission.g), emission.b);
    if emissive_intensity > EMISSIVE_THRESHOLD {
        var color = albedo + emission;
        let tonemapped = color / (color + vec3(1.0));
        // let gamma_corrected = pow(tonemapped, vec3(1.0 / 2.2)); // Gamma is handled by sRGB format
        return vec4(tonemapped, alpha);
    }

    let shade_mode = render_constants.shade_mode;
    if shade_mode == 1u {
        var color = albedo + emission;
        let tonemapped = color / (color + vec3(1.0));
        // let gamma_corrected = pow(tonemapped, vec3(1.0 / 2.2));
        return vec4(tonemapped, alpha);
    }

    let is_lighting_only = shade_mode == 2u;
    let effective_albedo = select(albedo, vec3(1.0), is_lighting_only);
    let effective_metallic = metallic;

    let tangent_space_normal = textureSampleBias(normal_texture, pbr_sampler, in.tex_coord, render_constants.mip_bias).xyz * 2.0 - 1.0;
    let N_geom = normalize(in.world_normal);
    let T = normalize(in.world_tangent);
    let B = normalize(in.world_bitangent);
    let tbn = mat3x3<f32>(T, B, N_geom);
    let N = normalize(tbn * tangent_space_normal);

    let V = normalize(camera.view_position - in.world_position);
    let R = reflect(-V, N);
    let F0 = mix(vec3<f32>(0.04), effective_albedo, effective_metallic);

    // --- DIRECT LIGHTING ---
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
            let bias_amount = 0.001 + 0.005 * (1.0 - NdotL);
            let biased_world_position = in.world_position + N * bias_amount;
            shadow_multiplier = calculate_shadow_factor(biased_world_position, in.view_space_depth, N, L);
        } else { // Point
            let to_light = light.position - in.world_position;
            let dist_sq = dot(to_light, to_light);
            if dist_sq < EPSILON { continue; }
            L = to_light / sqrt(dist_sq);
            radiance = light.color * light.intensity / (dist_sq + 1.0);
        }

        let H = normalize(V + L);
        let NdotL = max(dot(N, L), 0.0);
        if NdotL > 0.0 || is_stylized {
            let NdotH = max(dot(N, H), 0.0);
            let NDF = distribution_ggx(NdotH, roughness);
            let G = geometry_smith(N, V, L, roughness);
            let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + EPSILON;
            var specular = numerator / denominator;
            specular = select(specular, vec3(0.0), is_simple);
            let kS = F;
            var kD = vec3<f32>(1.0) - kS;
            kD *= (1.0 - effective_metallic);
            let n_dot_l_factor = select(NdotL, 1.0, is_stylized);
            let final_radiance = radiance * n_dot_l_factor * shadow_multiplier;
            let diffuse_brdf = effective_albedo / PI;
            Lo += (kD * diffuse_brdf + specular) * final_radiance;
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

    // --- IBL ---
    let F_ibl = fresnel_schlick_roughness(max(dot(N, V), 0.0), F0, roughness);
    let kS_ibl = F_ibl;
    var kD_ibl = vec3(1.0) - kS_ibl;
    kD_ibl *= (1.0 - effective_metallic);

    let irradiance = textureSampleBias(irradiance_map, env_map_sampler, N, render_constants.mip_bias).rgb;
    let diffuse_indirect = irradiance * effective_albedo;

    let prefiltered_color = textureSampleLevel(prefiltered_env_map, env_map_sampler, R, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSampleBias(brdf_lut, brdf_lut_sampler, vec2<f32>(max(dot(N, V), 0.0), roughness), render_constants.mip_bias).rg;
    let specular_indirect = prefiltered_color * (F_ibl * brdf.x + brdf.y);

    ambient += kD_ibl * diffuse_indirect * ao;
    specular_indirect_occluded += specular_indirect * ao;

    // --- FINAL COMPOSITION ---
    var final_hdr_color = ambient + Lo + specular_indirect_occluded + select(emission, vec3<f32>(0.0), is_lighting_only);

    let tonemapped = final_hdr_color / (final_hdr_color + vec3<f32>(1.0));
    //let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2)); // Handled by sRGB output format

    return vec4<f32>(tonemapped, alpha);
}