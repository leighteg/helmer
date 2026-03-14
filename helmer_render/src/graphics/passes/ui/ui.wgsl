struct UiPassParams {
    viewport_size: vec2<f32>,
    viewport_inv_size: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> params: UiPassParams;

@group(1) @binding(0)
var ui_tex: texture_2d<f32>;
@group(1) @binding(1)
var ui_samp: sampler;

const UI_FLAG_CLIP_ENABLED: u32 = 1u;
const UI_FLAG_TEXT_ALPHA: u32 = 1u << 1u;

struct VsIn {
    @location(0) rect: vec4<f32>,
    @location(1) uv_rect: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) clip_rect: vec4<f32>,
    @location(4) layer_flags: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) clip_rect: vec4<f32>,
    @location(3) @interpolate(flat) flags: u32,
    @location(4) screen_xy: vec2<f32>,
};

fn quad_corner(vid: u32) -> vec2<f32> {
    switch vid {
        case 0u: { return vec2<f32>(0.0, 0.0); }
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u: { return vec2<f32>(0.0, 1.0); }
        case 3u: { return vec2<f32>(0.0, 1.0); }
        case 4u: { return vec2<f32>(1.0, 0.0); }
        default: { return vec2<f32>(1.0, 1.0); }
    }
}

@vertex
fn vs_main(in: VsIn, @builtin(vertex_index) vid: u32) -> VsOut {
    let corner = quad_corner(vid);
    let px = in.rect.xy + corner * in.rect.zw;
    let ndc = vec2<f32>(
        px.x * (2.0 * params.viewport_inv_size.x) - 1.0,
        1.0 - px.y * (2.0 * params.viewport_inv_size.y)
    );

    var out: VsOut;
    out.clip_position = vec4<f32>(ndc, in.layer_flags.x, 1.0);
    out.uv = mix(in.uv_rect.xy, in.uv_rect.zw, corner);
    out.color = in.color;
    out.clip_rect = in.clip_rect;
    out.flags = u32(round(in.layer_flags.y));
    out.screen_xy = px;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if ((in.flags & UI_FLAG_CLIP_ENABLED) != 0u) {
        if (in.screen_xy.x < in.clip_rect.x
            || in.screen_xy.y < in.clip_rect.y
            || in.screen_xy.x > in.clip_rect.z
            || in.screen_xy.y > in.clip_rect.w) {
            discard;
        }
    }

    let sampled = textureSample(ui_tex, ui_samp, in.uv);
    if ((in.flags & UI_FLAG_TEXT_ALPHA) != 0u) {
        return vec4<f32>(in.color.rgb, in.color.a * sampled.r);
    }
    return sampled * in.color;
}
