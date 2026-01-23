struct DenoiseParams {
    radius: u32,
    depth_sigma: f32,
    normal_sigma: f32,
    color_sigma: f32,
}

@group(0) @binding(0) var reflection_in: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: DenoiseParams;
@group(0) @binding(4) var reflection_out: texture_storage_2d<rgba16float, write>;

fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    let n = encoded * 2.0 - 1.0;
    if length(n) < 1.0e-3 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(n);
}

@compute @workgroup_size(8, 8, 1)
fn rt_reflections_denoise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(reflection_in);
    if gid.x >= out_dims.x || gid.y >= out_dims.y {
        return;
    }

    let output_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let output_dims_f = vec2<f32>(f32(out_dims.x), f32(out_dims.y));
    let gbuffer_dims = textureDimensions(depth_tex);
    let gbuffer_dims_f = vec2<f32>(f32(gbuffer_dims.x), f32(gbuffer_dims.y));
    let uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / output_dims_f;
    let gbuffer_coord_f = uv * gbuffer_dims_f;
    let gbuffer_dims_i = vec2<i32>(i32(gbuffer_dims.x), i32(gbuffer_dims.y));
    let max_gbuf_coord = max(gbuffer_dims_i - vec2<i32>(1), vec2<i32>(0));
    let gbuffer_coord = clamp(vec2<i32>(gbuffer_coord_f), vec2<i32>(0), max_gbuf_coord);

    let center = textureLoad(reflection_in, output_coord, 0);
    let center_depth = textureLoad(depth_tex, gbuffer_coord, 0).x;
    if params.radius == 0u || center_depth <= 0.0 || center_depth >= 1.0 {
        textureStore(reflection_out, output_coord, center);
        return;
    }
    let center_normal = decode_normal(textureLoad(normal_tex, gbuffer_coord, 0).xyz);

    let radius_i = i32(params.radius);
    let max_out_coord = max(
        vec2<i32>(i32(out_dims.x) - 1, i32(out_dims.y) - 1),
        vec2<i32>(0),
    );
    let gbuffer_scale = gbuffer_dims_f / output_dims_f;

    var accum = vec3<f32>(0.0);
    var accum_alpha = 0.0;
    var accum_weight = 0.0;
    var accum_alpha_weight = 0.0;

    var y = -radius_i;
    loop {
        if y > radius_i {
            break;
        }
        var x = -radius_i;
        loop {
            if x > radius_i {
                break;
            }
            let sample_coord = clamp(
                output_coord + vec2<i32>(x, y),
                vec2<i32>(0),
                max_out_coord,
            );
            let sample_uv =
                (vec2<f32>(sample_coord) + vec2<f32>(0.5)) * gbuffer_scale;
            let sample_gbuf_coord =
                clamp(vec2<i32>(sample_uv), vec2<i32>(0), max_gbuf_coord);

            let sample = textureLoad(reflection_in, sample_coord, 0);
            let sample_depth = textureLoad(depth_tex, sample_gbuf_coord, 0).x;
            let sample_normal =
                decode_normal(textureLoad(normal_tex, sample_gbuf_coord, 0).xyz);

            let depth_diff = abs(sample_depth - center_depth);
            let normal_diff = 1.0 - max(dot(center_normal, sample_normal), 0.0);
            let color_diff = length(sample.rgb - center.rgb);

            let weight =
                exp(-depth_diff * params.depth_sigma)
                * exp(-normal_diff * params.normal_sigma)
                * exp(-color_diff * params.color_sigma);
            let weighted = weight * sample.a;
            accum = accum + sample.rgb * weighted;
            accum_weight = accum_weight + weighted;
            accum_alpha = accum_alpha + sample.a * weight;
            accum_alpha_weight = accum_alpha_weight + weight;

            x = x + 1;
        }
        y = y + 1;
    }

    let color = select(center.rgb, accum / accum_weight, accum_weight > 0.0);
    let alpha = select(center.a, accum_alpha / accum_alpha_weight, accum_alpha_weight > 0.0);
    textureStore(reflection_out, output_coord, vec4<f32>(color, clamp(alpha, 0.0, 1.0)));
}
