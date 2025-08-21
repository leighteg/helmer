//=============== CONSTANTS ===============//
const PI: f32 = 3.14159265359;
const MAX_REFLECTION_LOD: f32 = 4.0;
const EMISSIVE_THRESHOLD: f32 = 0.01;
const EPSILON: f32 = 0.00001;

//=============== STRUCTS ===============//
struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    inverse_projection_matrix: mat4x4<f32>,
    inverse_view_projection_matrix: mat4x4<f32>,
    view_position: vec3<f32>,
    light_count: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

//=============== BINDINGS ===============//
// --- Pass Inputs ---
@group(0) @binding(0) var direct_lighting_tex: texture_2d<f32>;
@group(0) @binding(1) var ssgi_tex: texture_2d<f32>; // Now receives the upsampled result
@group(0) @binding(2) var ssr_tex: texture_2d<f32>;
@group(0) @binding(3) var albedo_tex: texture_2d<f32>;
@group(0) @binding(4) var emission_tex: texture_2d<f32>;
@group(0) @binding(5) var scene_sampler: sampler;
// --- G-Buffer for IBL ---
@group(0) @binding(6) var normal_tex: texture_2d<f32>;
@group(0) @binding(7) var mra_tex: texture_2d<f32>;
@group(0) @binding(8) var depth_tex: texture_depth_2d;

// --- IBL Textures ---
@group(1) @binding(0) var brdf_lut: texture_2d<f32>;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var prefiltered_env_map: texture_cube<f32>;
@group(1) @binding(3) var ibl_sampler: sampler;
@group(1) @binding(4) var brdf_lut_sampler: sampler;

// --- Scene Uniforms ---
@group(2) @binding(0) var<uniform> camera: CameraUniforms;

//=============== UTILITY & PBR FUNCTIONS ===============//
fn fresnel_schlick_roughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

//=============== SHADERS ===============//
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate a fullscreen triangle from vertex indices
    out.tex_coords = vec2<f32>(
        f32((in_vertex_index << 1u) & 2u),
        f32(in_vertex_index & 2u)
    );
    out.clip_position = vec4<f32>(
        out.tex_coords.x * 2.0 - 1.0,
        1.0 - out.tex_coords.y * 2.0,
        0.0,
        1.0
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo_sample = textureSample(albedo_tex, scene_sampler, in.tex_coords);
    let albedo = albedo_sample.rgb;
    let alpha = albedo_sample.a;
    let emission = textureSample(emission_tex, scene_sampler, in.tex_coords).rgb;

    let emission_strength = length(emission);
    if (emission_strength > EMISSIVE_THRESHOLD) {
        let tonemapped = emission / (emission + vec3<f32>(1.0));
        let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2));
        return vec4<f32>(gamma_corrected, alpha);
    }
    
    let direct_light = textureSample(direct_lighting_tex, scene_sampler, in.tex_coords).rgb;
    let indirect_diffuse_ssgi = textureSample(ssgi_tex, scene_sampler, in.tex_coords).rgb;
    let ssr_sample = textureSample(ssr_tex, scene_sampler, in.tex_coords);
    let indirect_specular_ssr = ssr_sample.rgb;
    let ssr_confidence = ssr_sample.a;

    // --- IBL Fallback Calculation ---
    let mra = textureSample(mra_tex, scene_sampler, in.tex_coords);
    let metallic = mra.r;
    let roughness = mra.g;
    let ao = mra.b;
    
    let packed_normal = textureSample(normal_tex, scene_sampler, in.tex_coords).xyz;
    let N = normalize(packed_normal * 2.0 - 1.0);

    let depth = textureSample(depth_tex, scene_sampler, in.tex_coords);
    let ndc = vec4<f32>(in.tex_coords.x * 2.0 - 1.0, (1.0 - in.tex_coords.y) * 2.0 - 1.0, depth, 1.0);
    let world_pos_h = camera.inverse_view_projection_matrix * ndc;
    let world_position = world_pos_h.xyz / world_pos_h.w;
    
    let V = normalize(camera.view_position - world_position);
    let R = reflect(-V, N);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Get IBL diffuse and specular
    let F_ibl = fresnel_schlick_roughness(max(dot(N, V), 0.0), F0, roughness);
    let kS_ibl = F_ibl;
    var kD_ibl = vec3(1.0) - kS_ibl;
    kD_ibl *= (1.0 - metallic);
    
    let irradiance = textureSample(irradiance_map, ibl_sampler, N).rgb;
    let diffuse_ibl = irradiance * albedo;

    let prefiltered_color = textureSampleLevel(prefiltered_env_map, ibl_sampler, R, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSample(brdf_lut, brdf_lut_sampler, vec2<f32>(max(dot(N, V), 0.0), roughness)).rg;
    let specular_ibl = prefiltered_color * (F_ibl * brdf.x + brdf.y);

    // --- Final Combination ---
    let total_indirect_diffuse = (indirect_diffuse_ssgi + diffuse_ibl) * kD_ibl;
    let total_indirect_specular = mix(specular_ibl, indirect_specular_ssr, ssr_confidence);
    let final_hdr_color = (direct_light + total_indirect_diffuse + total_indirect_specular) * ao;

    // Tonemapping and Gamma Correction
    let tonemapped = final_hdr_color / (final_hdr_color + vec3<f32>(1.0));
    let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, alpha);
}