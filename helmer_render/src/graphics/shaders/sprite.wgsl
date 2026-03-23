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
    _pad_after_frame: vec3<u32>,
    _padding: vec3<u32>,
    _pad_after_padding: u32,
    _pad_end: vec4<u32>,
};

struct SpritePassParams {
    viewport_size: vec2<f32>,
    viewport_inv_size: vec2<f32>,
    depth_flags: vec4<u32>,
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

struct LightData {
    position: vec3<f32>,
    light_type: u32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    _padding: f32,
};

const MAX_SPRITE_LIGHTS: u32 = 32u;

struct SpriteLights {
    lights: array<LightData, MAX_SPRITE_LIGHTS>,
};

struct SpriteInstanceIn {
    @location(0) origin_mode: vec4<f32>,
    @location(1) right_size_x: vec4<f32>,
    @location(2) up_size_y: vec4<f32>,
    @location(3) uv_rect: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) pivot_clip_min: vec4<f32>,
    @location(6) clip_max_layer: vec4<f32>,
    @location(7) instance_meta: vec4<u32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) clip_min: vec2<f32>,
    @location(3) clip_max: vec2<f32>,
    @location(4) @interpolate(flat) flags: u32,
    @location(5) @interpolate(flat) pick_id: u32,
    @location(6) @interpolate(flat) world_space: u32,
    @location(7) @interpolate(flat) blend_mode: u32,
    @location(8) world_pos: vec3<f32>,
    @location(9) world_normal: vec3<f32>,
    @location(10) lighting: vec3<f32>,
    @location(11) @interpolate(flat) lighting_mode: u32,
};

struct FsIn {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) clip_min: vec2<f32>,
    @location(3) clip_max: vec2<f32>,
    @location(4) @interpolate(flat) flags: u32,
    @location(5) @interpolate(flat) pick_id: u32,
    @location(6) @interpolate(flat) world_space: u32,
    @location(7) @interpolate(flat) blend_mode: u32,
    @location(8) world_pos: vec3<f32>,
    @location(9) world_normal: vec3<f32>,
    @location(10) lighting: vec3<f32>,
    @location(11) @interpolate(flat) lighting_mode: u32,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) pick: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> pass_params: SpritePassParams;
@group(0) @binding(2) var<uniform> sprite_lights: SpriteLights;
@group(0) @binding(3) var<uniform> sky: SkyUniforms;
@group(1) @binding(0) var sprite_tex: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;
@group(1) @binding(2) var scene_depth_tex: texture_2d<f32>;

const FLAG_CLIP_ENABLED: u32 = 1u;
const FLAG_TEXT_ALPHA: u32 = 1u << 1u;
const FLAG_DEPTH_WRITE: u32 = 1u << 3u;
const FLAG_QUAD_CORRECTION: u32 = 1u << 4u;
const BLEND_MODE_ADDITIVE: u32 = 2u;
const SPRITE_LIGHTING_FRAGMENT: u32 = 0u;
const SPRITE_LIGHTING_VERTEX: u32 = 1u;
const INSTANCE_LIGHTING_NONE: u32 = 0u;
const INSTANCE_LIGHTING_FRAGMENT: u32 = 1u;
const INSTANCE_LIGHTING_VERTEX: u32 = 2u;

fn safe_normalize(value: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(value, value);
    if (len_sq <= 1.0e-8) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return value * inverseSqrt(len_sq);
}

fn sprite_sun_fade() -> f32 {
    let sun_height_factor = max(sky.sun_direction.y, 0.0);
    return pow(sun_height_factor, 1.5);
}

fn sprite_ambient_light() -> vec3<f32> {
    let night_floor = max(
        sky.night_ambient_color * 72.0,
        vec3<f32>(0.035, 0.040, 0.055),
    );
    let day_ambient = vec3<f32>(0.24 * max(sky.ground_brightness, 0.25));
    return mix(night_floor, day_ambient, clamp(sprite_sun_fade(), 0.0, 1.0));
}

fn sprite_lighting_factor(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let N = safe_normalize(normal);
    let sun_fade = sprite_sun_fade();
    var lighting = sprite_ambient_light();
    let light_count = min(pass_params.depth_flags.y, MAX_SPRITE_LIGHTS);
    for (var i = 0u; i < light_count; i = i + 1u) {
        let light = sprite_lights.lights[i];
        if (light.intensity <= 0.0) {
            continue;
        }

        if (light.light_type == 0u) {
            let L = safe_normalize(-light.direction);
            let ndotl = 0.55 + 0.45 * max(dot(N, L), 0.0);
            lighting += light.color * (light.intensity * 0.14 * sun_fade) * ndotl;
        } else if (light.light_type == 1u || light.light_type == 2u) {
            let to_light = light.position - world_pos;
            let dist_sq = dot(to_light, to_light);
            if (dist_sq <= 1.0e-6) {
                continue;
            }
            let dist = sqrt(dist_sq);
            let radius = max(light.intensity * 5.5, 2.0);
            if (dist >= radius) {
                continue;
            }
            let L = to_light / dist;
            let ndotl = 0.25 + 0.75 * max(dot(N, L), 0.0);
            let attenuation = 1.0 - dist / radius;
            lighting += light.color * light.intensity * attenuation * attenuation * 0.45 * ndotl;
        }
    }
    return min(lighting, vec3<f32>(1.8));
}

fn sprite_corner(vertex_index: u32) -> vec2<f32> {
    switch vertex_index {
        case 0u: { return vec2<f32>(0.0, 0.0); }
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u: { return vec2<f32>(0.0, 1.0); }
        case 3u: { return vec2<f32>(0.0, 1.0); }
        case 4u: { return vec2<f32>(1.0, 0.0); }
        default: { return vec2<f32>(1.0, 1.0); }
    }
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, instance: SpriteInstanceIn) -> VsOut {
    let corner = sprite_corner(vertex_index);
    let pivot = instance.pivot_clip_min.xy;
    let size = vec2<f32>(instance.right_size_x.w, instance.up_size_y.w);
    let local = (corner - pivot) * size;

    let mode = instance.origin_mode.w;
    var out: VsOut;
    out.world_pos = vec3<f32>(0.0);
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.lighting = vec3<f32>(1.0);
    out.lighting_mode = INSTANCE_LIGHTING_NONE;

    var clip: vec4<f32>;
    if (mode > 0.5) {
        let quad_correction = vec3<f32>(
            instance.pivot_clip_min.z,
            instance.pivot_clip_min.w,
            instance.clip_max_layer.x,
        );
        let world = instance.origin_mode.xyz
            + instance.right_size_x.xyz * local.x
            + instance.up_size_y.xyz * local.y
            + select(
                vec3<f32>(0.0),
                quad_correction * (corner.x * corner.y),
                (instance.instance_meta.y & FLAG_QUAD_CORRECTION) != 0u,
            );
        clip = camera.projection_matrix * camera.view_matrix * vec4<f32>(world, 1.0);
        if ((instance.instance_meta.y & FLAG_DEPTH_WRITE) == 0u) {
            clip.z = clip.z - (instance.clip_max_layer.z * 1.0e-4 * clip.w);
        }
        out.world_pos = world;
        let world_dx = instance.right_size_x.xyz
            + select(
                vec3<f32>(0.0),
                quad_correction * corner.y,
                (instance.instance_meta.y & FLAG_QUAD_CORRECTION) != 0u,
            );
        let world_dy = instance.up_size_y.xyz
            + select(
                vec3<f32>(0.0),
                quad_correction * corner.x,
                (instance.instance_meta.y & FLAG_QUAD_CORRECTION) != 0u,
            );
        out.world_normal = safe_normalize(cross(world_dx, world_dy));
        if (instance.instance_meta.w != BLEND_MODE_ADDITIVE) {
            if (pass_params.depth_flags.z == SPRITE_LIGHTING_VERTEX) {
                out.lighting = sprite_lighting_factor(world, out.world_normal);
                out.lighting_mode = INSTANCE_LIGHTING_VERTEX;
            } else {
                out.lighting_mode = INSTANCE_LIGHTING_FRAGMENT;
            }
        }
        out.color = instance.color;
    } else {
        let quad_correction = vec2<f32>(
            instance.pivot_clip_min.z,
            instance.pivot_clip_min.w,
        );
        let pixel = instance.origin_mode.xy
            + instance.right_size_x.xy * local.x
            + instance.up_size_y.xy * local.y
            + select(
                vec2<f32>(0.0),
                quad_correction * (corner.x * corner.y),
                (instance.instance_meta.y & FLAG_QUAD_CORRECTION) != 0u,
            );
        let ndc = vec2<f32>(
            (pixel.x * pass_params.viewport_inv_size.x) * 2.0 - 1.0,
            1.0 - (pixel.y * pass_params.viewport_inv_size.y) * 2.0,
        );
        clip = vec4<f32>(ndc, instance.clip_max_layer.z, 1.0);
        out.color = instance.color;
    }

    let uv = mix(instance.uv_rect.xy, instance.uv_rect.zw, corner);

    out.clip_pos = clip;
    out.uv = uv;
    out.clip_min = instance.pivot_clip_min.zw;
    out.clip_max = instance.clip_max_layer.xy;
    out.flags = instance.instance_meta.y;
    out.pick_id = instance.instance_meta.z;
    out.world_space = select(0u, 1u, mode > 0.5);
    out.blend_mode = instance.instance_meta.w;
    return out;
}

@fragment
fn fs_main(in: FsIn, @builtin(position) frag_coord: vec4<f32>) -> FsOut {
    if ((in.flags & FLAG_CLIP_ENABLED) != 0u) {
        if (frag_coord.x < in.clip_min.x || frag_coord.y < in.clip_min.y
            || frag_coord.x > in.clip_max.x || frag_coord.y > in.clip_max.y) {
            discard;
        }
    }

    let sampled = textureSample(sprite_tex, sprite_sampler, in.uv);
    var color = sampled * in.color;
    if ((in.flags & FLAG_TEXT_ALPHA) != 0u) {
        color = vec4<f32>(in.color.rgb, sampled.r * in.color.a);
    }

    if (color.a <= 0.0001) {
        discard;
    }

    if (in.lighting_mode == INSTANCE_LIGHTING_VERTEX) {
        color = vec4<f32>(color.rgb * in.lighting, color.a);
    } else if (in.lighting_mode == INSTANCE_LIGHTING_FRAGMENT) {
        color = vec4<f32>(
            color.rgb * sprite_lighting_factor(in.world_pos, in.world_normal),
            color.a,
        );
    }

    if (pass_params.depth_flags.x != 0u && in.world_space != 0u) {
        let pixel = vec2<i32>(frag_coord.xy);
        let scene_depth = textureLoad(scene_depth_tex, pixel, 0).x;
        if (scene_depth > frag_coord.z + 1.0e-5) {
            discard;
        }
    }

    var out: FsOut;
    out.color = color;
    out.pick = select(0u, in.pick_id, color.a > 0.001);
    return out;
}
