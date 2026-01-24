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

struct GizmoParams {
    origin: vec3<f32>,
    mode: u32,
    rotation: vec4<f32>,
    scale: vec3<f32>,
    size: f32,
    hover_axis: u32,
    active_axis: u32,
    ring_segments: u32,
    _pad0: u32,
    translate_params: vec4<f32>,
    scale_params: vec4<f32>,
    rotate_params: vec4<f32>,
    origin_params: vec4<f32>,
    axis_color_x: vec4<f32>,
    axis_color_y: vec4<f32>,
    axis_color_z: vec4<f32>,
    origin_color: vec4<f32>,
    highlight_params: vec4<f32>,
    selection_min: vec3<f32>,
    selection_enabled: u32,
    selection_max: vec3<f32>,
    selection_thickness: f32,
    selection_color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> gizmo: GizmoParams;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

struct Basis {
    right: vec3<f32>,
    up: vec3<f32>,
}

const AXIS_COUNT: u32 = 3u;
const ARROW_VERTS_PER_AXIS: u32 = 9u;
const SCALE_VERTS_PER_AXIS: u32 = 12u;
const ORIGIN_VERTS: u32 = 6u;
const SELECTION_EDGE_COUNT: u32 = 12u;
const SELECTION_VERTS_PER_EDGE: u32 = 6u;
const TWO_PI: f32 = 6.2831853;
const CENTER_AXIS_ID: u32 = 4u;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len < 1.0e-4) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return v / len;
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

fn axis_dir(axis_index: u32) -> vec3<f32> {
    if (axis_index == 0u) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    if (axis_index == 1u) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn axis_color(axis_index: u32) -> vec3<f32> {
    if (axis_index == 0u) {
        return gizmo.axis_color_x.xyz;
    }
    if (axis_index == 1u) {
        return gizmo.axis_color_y.xyz;
    }
    return gizmo.axis_color_z.xyz;
}

const SELECTION_EDGE_STARTS: array<u32, 12> = array<u32, 12>(
    0u, 0u, 0u, 1u, 1u, 2u, 2u, 3u, 4u, 4u, 5u, 6u
);
const SELECTION_EDGE_ENDS: array<u32, 12> = array<u32, 12>(
    1u, 2u, 4u, 3u, 5u, 3u, 6u, 7u, 5u, 6u, 7u, 7u
);

fn selection_corner(index: u32) -> vec3<f32> {
    let x = select(gizmo.selection_min.x, gizmo.selection_max.x, (index & 1u) == 1u);
    let y = select(gizmo.selection_min.y, gizmo.selection_max.y, (index & 2u) == 2u);
    let z = select(gizmo.selection_min.z, gizmo.selection_max.z, (index & 4u) == 4u);
    return vec3<f32>(x, y, z);
}

fn axis_side(axis_dir: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    var side = cross(view_dir, axis_dir);
    if (length(side) < 1.0e-4) {
        side = cross(axis_dir, vec3<f32>(0.0, 1.0, 0.0));
    }
    if (length(side) < 1.0e-4) {
        side = cross(axis_dir, vec3<f32>(1.0, 0.0, 0.0));
    }
    return safe_normalize(side);
}

fn billboard_basis(view_dir: vec3<f32>) -> Basis {
    var right = cross(vec3<f32>(0.0, 1.0, 0.0), view_dir);
    if (length(right) < 1.0e-4) {
        right = cross(vec3<f32>(1.0, 0.0, 0.0), view_dir);
    }
    right = safe_normalize(right);
    let up = safe_normalize(cross(view_dir, right));
    return Basis(right, up);
}

fn ring_basis(axis_dir: vec3<f32>) -> Basis {
    var right = cross(axis_dir, vec3<f32>(0.0, 1.0, 0.0));
    if (length(right) < 1.0e-4) {
        right = cross(axis_dir, vec3<f32>(1.0, 0.0, 0.0));
    }
    right = safe_normalize(right);
    let up = safe_normalize(cross(axis_dir, right));
    return Basis(right, up);
}

fn quad_corner_along(index: u32) -> vec2<f32> {
    if (index == 0u) {
        return vec2<f32>(0.0, -1.0);
    }
    if (index == 1u) {
        return vec2<f32>(1.0, -1.0);
    }
    if (index == 2u) {
        return vec2<f32>(1.0, 1.0);
    }
    if (index == 3u) {
        return vec2<f32>(0.0, -1.0);
    }
    if (index == 4u) {
        return vec2<f32>(1.0, 1.0);
    }
    return vec2<f32>(0.0, 1.0);
}

fn quad_corner_signed(index: u32) -> vec2<f32> {
    if (index == 0u) {
        return vec2<f32>(-1.0, -1.0);
    }
    if (index == 1u) {
        return vec2<f32>(1.0, -1.0);
    }
    if (index == 2u) {
        return vec2<f32>(1.0, 1.0);
    }
    if (index == 3u) {
        return vec2<f32>(-1.0, -1.0);
    }
    if (index == 4u) {
        return vec2<f32>(1.0, 1.0);
    }
    return vec2<f32>(-1.0, 1.0);
}

fn highlight_color(base: vec3<f32>, axis_id: u32) -> vec3<f32> {
    if (gizmo.active_axis == axis_id) {
        return mix(base, vec3<f32>(1.0, 1.0, 1.0), gizmo.highlight_params.y);
    }
    if (gizmo.hover_axis == axis_id) {
        return mix(base, vec3<f32>(1.0, 1.0, 1.0), gizmo.highlight_params.x);
    }
    return base;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var pos = gizmo.origin;
    var color = vec3<f32>(0.0, 0.0, 0.0);
    var alpha = 1.0;

    let axis_len = max(gizmo.size, 0.001);
    let view_dir = safe_normalize(camera.view_position - gizmo.origin);
    let ring_segments = max(gizmo.ring_segments, 3u);
    let ring_verts_per_axis = ring_segments * 6u;
    let translate_total = ARROW_VERTS_PER_AXIS * AXIS_COUNT;
    let scale_total = SCALE_VERTS_PER_AXIS * AXIS_COUNT;
    let rotate_total = ring_verts_per_axis * AXIS_COUNT;
    let selection_total = select(0u, SELECTION_EDGE_COUNT * SELECTION_VERTS_PER_EDGE, gizmo.selection_enabled != 0u);
    var gizmo_total = 0u;
    if (gizmo.mode == 1u) {
        gizmo_total = translate_total + ORIGIN_VERTS;
    } else if (gizmo.mode == 2u) {
        gizmo_total = rotate_total + ORIGIN_VERTS;
    } else if (gizmo.mode == 3u) {
        gizmo_total = scale_total + ORIGIN_VERTS;
    }

    if (vertex_index < gizmo_total) {
        if (gizmo.mode == 2u) {
            if (vertex_index < rotate_total) {
                let axis_index = vertex_index / ring_verts_per_axis;
                let local_index = vertex_index % ring_verts_per_axis;
                let axis_id = axis_index + 1u;

                let dir = quat_rotate(gizmo.rotation, axis_dir(axis_index));
                let basis = ring_basis(dir);
                let thickness = max(axis_len * gizmo.rotate_params.y, gizmo.rotate_params.z);
                let radius = axis_len * gizmo.rotate_params.x;

                let segment = local_index / 6u;
                let corner = local_index % 6u;
                let angle0 = TWO_PI * (f32(segment) / f32(ring_segments));
                let angle1 = TWO_PI * (f32(segment + 1u) / f32(ring_segments));

                let p0 = basis.right * cos(angle0) + basis.up * sin(angle0);
                let p1 = basis.right * cos(angle1) + basis.up * sin(angle1);
                let center0 = p0 * radius;
                let center1 = p1 * radius;
                var tangent = safe_normalize(center1 - center0);
                var side = cross(view_dir, tangent);
                if (length(side) < 1.0e-4) {
                    side = cross(dir, tangent);
                }
                side = safe_normalize(side);
                let offset = side * thickness;

                var ring_offset = vec3<f32>(0.0, 0.0, 0.0);
                if (corner == 0u) {
                    ring_offset = center0 - offset;
                } else if (corner == 1u) {
                    ring_offset = center0 + offset;
                } else if (corner == 2u) {
                    ring_offset = center1 + offset;
                } else if (corner == 3u) {
                    ring_offset = center0 - offset;
                } else if (corner == 4u) {
                    ring_offset = center1 + offset;
                } else {
                    ring_offset = center1 - offset;
                }

                pos = gizmo.origin + ring_offset;
                color = highlight_color(axis_color(axis_index), axis_id);
            } else if (vertex_index < rotate_total + ORIGIN_VERTS) {
                let local_index = vertex_index - rotate_total;
                let basis = billboard_basis(view_dir);
                let dot_size = max(axis_len * gizmo.origin_params.x, gizmo.origin_params.y);
                let corner = quad_corner_signed(local_index);
                pos = gizmo.origin
                    + basis.right * (corner.x * dot_size)
                    + basis.up * (corner.y * dot_size);
                color = highlight_color(gizmo.origin_color.xyz, CENTER_AXIS_ID);
            } else {
                alpha = 0.0;
            }
        } else if (gizmo.mode == 3u) {
            if (vertex_index < scale_total) {
                let axis_index = vertex_index / SCALE_VERTS_PER_AXIS;
                let local_index = vertex_index % SCALE_VERTS_PER_AXIS;
                let axis_id = axis_index + 1u;

                let dir = quat_rotate(gizmo.rotation, axis_dir(axis_index));
                let side = axis_side(dir, view_dir);
                let basis = billboard_basis(view_dir);

                let thickness = max(axis_len * gizmo.scale_params.x, gizmo.scale_params.y);
                let head_len = axis_len * gizmo.scale_params.z;
                let shaft_len = axis_len - head_len;

                if (local_index < 6u) {
                    let quad = quad_corner_along(local_index);
                    pos = gizmo.origin + dir * (quad.x * shaft_len) + side * (quad.y * thickness);
                } else {
                    let box_half = thickness * gizmo.scale_params.w;
                    let center = gizmo.origin + dir * axis_len;
                    let corner = quad_corner_signed(local_index - 6u);
                    pos = center
                        + basis.right * (corner.x * box_half)
                        + basis.up * (corner.y * box_half);
                }

                color = highlight_color(axis_color(axis_index), axis_id);
            } else if (vertex_index < scale_total + ORIGIN_VERTS) {
                let local_index = vertex_index - scale_total;
                let basis = billboard_basis(view_dir);
                let dot_size = max(axis_len * gizmo.origin_params.x, gizmo.origin_params.y);
                let corner = quad_corner_signed(local_index);
                pos = gizmo.origin
                    + basis.right * (corner.x * dot_size)
                    + basis.up * (corner.y * dot_size);
                color = highlight_color(gizmo.origin_color.xyz, CENTER_AXIS_ID);
            } else {
                alpha = 0.0;
            }
        } else if (gizmo.mode == 1u) {
            if (vertex_index < translate_total) {
                let axis_index = vertex_index / ARROW_VERTS_PER_AXIS;
                let local_index = vertex_index % ARROW_VERTS_PER_AXIS;
                let axis_id = axis_index + 1u;

                let dir = quat_rotate(gizmo.rotation, axis_dir(axis_index));
                let side = axis_side(dir, view_dir);

                let thickness = max(axis_len * gizmo.translate_params.x, gizmo.translate_params.y);
                let head_len = axis_len * gizmo.translate_params.z;
                let shaft_len = axis_len - head_len;
                let head_width = thickness * gizmo.translate_params.w;

                if (local_index < 6u) {
                    let quad = quad_corner_along(local_index);
                    pos = gizmo.origin + dir * (quad.x * shaft_len) + side * (quad.y * thickness);
                } else {
                    let base = gizmo.origin + dir * shaft_len;
                    let tip = gizmo.origin + dir * axis_len;
                    if (local_index == 6u) {
                        pos = base - side * head_width;
                    } else if (local_index == 7u) {
                        pos = base + side * head_width;
                    } else {
                        pos = tip;
                    }
                }

                color = highlight_color(axis_color(axis_index), axis_id);
            } else if (vertex_index < translate_total + ORIGIN_VERTS) {
                let local_index = vertex_index - translate_total;
                let basis = billboard_basis(view_dir);
                let dot_size = max(axis_len * gizmo.origin_params.x, gizmo.origin_params.y);
                let corner = quad_corner_signed(local_index);
                pos = gizmo.origin
                    + basis.right * (corner.x * dot_size)
                    + basis.up * (corner.y * dot_size);
                color = highlight_color(gizmo.origin_color.xyz, CENTER_AXIS_ID);
            } else {
                alpha = 0.0;
            }
        } else {
            alpha = 0.0;
        }
    } else if (vertex_index < gizmo_total + selection_total) {
        let local_index = vertex_index - gizmo_total;
        let edge_index = local_index / SELECTION_VERTS_PER_EDGE;
        let corner_index = local_index % SELECTION_VERTS_PER_EDGE;
        let start = selection_corner(SELECTION_EDGE_STARTS[edge_index]);
        let end = selection_corner(SELECTION_EDGE_ENDS[edge_index]);
        let start_world = gizmo.origin + quat_rotate(gizmo.rotation, start * gizmo.scale);
        let end_world = gizmo.origin + quat_rotate(gizmo.rotation, end * gizmo.scale);
        let edge_dir = safe_normalize(end_world - start_world);
        let side = axis_side(edge_dir, view_dir);
        let quad = quad_corner_along(corner_index);
        pos = mix(start_world, end_world, quad.x) + side * (quad.y * gizmo.selection_thickness);
        color = gizmo.selection_color.xyz;
        alpha = gizmo.selection_color.w;
    } else {
        alpha = 0.0;
    }

    let clip = camera.projection_matrix * camera.view_matrix * vec4<f32>(pos, 1.0);

    var out: VertexOut;
    out.clip_position = clip;
    out.color = vec4<f32>(color, alpha);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
