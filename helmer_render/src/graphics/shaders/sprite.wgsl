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
};

struct FsIn {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) clip_min: vec2<f32>,
    @location(3) clip_max: vec2<f32>,
    @location(4) @interpolate(flat) flags: u32,
    @location(5) @interpolate(flat) pick_id: u32,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) pick: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> pass_params: SpritePassParams;
@group(1) @binding(0) var sprite_tex: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;

const FLAG_CLIP_ENABLED: u32 = 1u;
const FLAG_TEXT_ALPHA: u32 = 1u << 1u;

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

    var clip: vec4<f32>;
    if (mode > 0.5) {
        let world = instance.origin_mode.xyz
            + instance.right_size_x.xyz * local.x
            + instance.up_size_y.xyz * local.y;
        clip = camera.projection_matrix * camera.view_matrix * vec4<f32>(world, 1.0);
        clip.z = clip.z + (instance.clip_max_layer.z * 1.0e-4 * clip.w);
    } else {
        let pixel = instance.origin_mode.xy + local;
        let ndc = vec2<f32>(
            (pixel.x * pass_params.viewport_inv_size.x) * 2.0 - 1.0,
            1.0 - (pixel.y * pass_params.viewport_inv_size.y) * 2.0,
        );
        clip = vec4<f32>(ndc, instance.clip_max_layer.z, 1.0);
    }

    let uv = mix(instance.uv_rect.xy, instance.uv_rect.zw, corner);

    var out: VsOut;
    out.clip_pos = clip;
    out.uv = uv;
    out.color = instance.color;
    out.clip_min = instance.pivot_clip_min.zw;
    out.clip_max = instance.clip_max_layer.xy;
    out.flags = instance.instance_meta.y;
    out.pick_id = instance.instance_meta.z;
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

    var out: FsOut;
    out.color = color;
    out.pick = select(0u, in.pick_id, color.a > 0.001);
    return out;
}
