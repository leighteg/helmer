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

struct GizmoIcon {
    position: vec3<f32>,
    kind: u32,
    rotation: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
    size_params: vec4<f32>,
}

struct GizmoLine {
    start: vec3<f32>,
    _pad0: f32,
    end: vec3<f32>,
    _pad1: f32,
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
    plane_params: vec4<f32>,
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
    icon_meta: vec4<u32>,
    icon_line_params: vec4<f32>,
    outline_meta: vec4<u32>,
    outline_line_params: vec4<f32>,
    outline_color: vec4<f32>,
}

struct GizmoIconBuffer {
    icons: array<GizmoIcon>,
}

struct GizmoLineBuffer {
    lines: array<GizmoLine>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> gizmo: GizmoParams;
@group(1) @binding(1) var<storage, read> gizmo_icons: GizmoIconBuffer;
@group(1) @binding(2) var<storage, read> gizmo_outline_lines: GizmoLineBuffer;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

struct Basis {
    right: vec3<f32>,
    up: vec3<f32>,
}

const AXIS_COUNT: u32 = 3u;
const PLANE_COUNT: u32 = 3u;
const ARROW_VERTS_PER_AXIS: u32 = 9u;
const SCALE_VERTS_PER_AXIS: u32 = 12u;
const PLANE_VERTS_PER_HANDLE: u32 = 6u;
const ORIGIN_VERTS: u32 = 6u;
const SELECTION_EDGE_COUNT: u32 = 12u;
const SELECTION_VERTS_PER_EDGE: u32 = 6u;
const ICON_EDGE_COUNT: u32 = 16u;
const ICON_VERTS_PER_EDGE: u32 = 6u;
const ICON_VERTS_PER_GIZMO: u32 = ICON_EDGE_COUNT * ICON_VERTS_PER_EDGE;
const OUTLINE_VERTS_PER_LINE: u32 = 6u;
const TWO_PI: f32 = 6.2831853;
const CENTER_AXIS_ID: u32 = 4u;
const PLANE_XY_AXIS_ID: u32 = 5u;
const PLANE_XZ_AXIS_ID: u32 = 6u;
const PLANE_YZ_AXIS_ID: u32 = 7u;
const ICON_KIND_CAMERA: u32 = 0u;
const ICON_KIND_LIGHT_DIRECTIONAL: u32 = 1u;
const ICON_KIND_LIGHT_POINT: u32 = 2u;
const ICON_KIND_LIGHT_SPOT: u32 = 3u;
const ICON_KIND_AUDIO_EMITTER: u32 = 4u;

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

fn axis_color(axis_index: u32) -> vec4<f32> {
    if (axis_index == 0u) {
        return gizmo.axis_color_x;
    }
    if (axis_index == 1u) {
        return gizmo.axis_color_y;
    }
    return gizmo.axis_color_z;
}

fn plane_axis_id(plane_index: u32) -> u32 {
    if (plane_index == 0u) {
        return PLANE_XY_AXIS_ID;
    }
    if (plane_index == 1u) {
        return PLANE_XZ_AXIS_ID;
    }
    return PLANE_YZ_AXIS_ID;
}

fn plane_axis_u(plane_index: u32) -> vec3<f32> {
    if (plane_index == 0u || plane_index == 1u) {
        return axis_dir(0u);
    }
    return axis_dir(1u);
}

fn plane_axis_v(plane_index: u32) -> vec3<f32> {
    if (plane_index == 0u) {
        return axis_dir(1u);
    }
    return axis_dir(2u);
}

fn plane_color(plane_index: u32) -> vec4<f32> {
    if (plane_index == 0u) {
        return vec4<f32>((gizmo.axis_color_x.xyz + gizmo.axis_color_y.xyz) * 0.5, gizmo.plane_params.z);
    }
    if (plane_index == 1u) {
        return vec4<f32>((gizmo.axis_color_x.xyz + gizmo.axis_color_z.xyz) * 0.5, gizmo.plane_params.z);
    }
    return vec4<f32>((gizmo.axis_color_y.xyz + gizmo.axis_color_z.xyz) * 0.5, gizmo.plane_params.z);
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

fn plane_corner(index: u32) -> vec2<f32> {
    if (index == 0u) {
        return vec2<f32>(0.0, 0.0);
    }
    if (index == 1u) {
        return vec2<f32>(1.0, 0.0);
    }
    if (index == 2u) {
        return vec2<f32>(1.0, 1.0);
    }
    if (index == 3u) {
        return vec2<f32>(0.0, 0.0);
    }
    if (index == 4u) {
        return vec2<f32>(1.0, 1.0);
    }
    return vec2<f32>(0.0, 1.0);
}

struct IconEdge {
    start: vec3<f32>,
    end: vec3<f32>,
    valid: bool,
}

fn line_vertex(
    index: u32,
    start: vec3<f32>,
    end: vec3<f32>,
    view_dir: vec3<f32>,
    thickness: f32
) -> vec3<f32> {
    let edge_dir = safe_normalize(end - start);
    let side = axis_side(edge_dir, view_dir);
    let quad = quad_corner_along(index);
    return mix(start, end, quad.x) + side * (quad.y * thickness);
}

fn icon_edge(kind: u32, edge_index: u32, size: f32, params: vec4<f32>) -> IconEdge {
    var edge: IconEdge;
    edge.start = vec3<f32>(0.0, 0.0, 0.0);
    edge.end = vec3<f32>(0.0, 0.0, 0.0);
    edge.valid = true;

    if (kind == ICON_KIND_CAMERA) {
        let fov = clamp(params.x, 0.01, 3.12);
        let aspect = max(params.y, 0.01);
        let near_ratio = clamp(params.z, 0.02, 0.9);
        let far_dist = max(size, 0.001);
        let near_dist = far_dist * near_ratio;
        let tan_half = tan(fov * 0.5);
        let half_h_far = tan_half * far_dist;
        let half_w_far = half_h_far * aspect;
        let half_h_near = tan_half * near_dist;
        let half_w_near = half_h_near * aspect;
        let n0 = vec3<f32>(-half_w_near, -half_h_near, near_dist);
        let n1 = vec3<f32>(half_w_near, -half_h_near, near_dist);
        let n2 = vec3<f32>(half_w_near, half_h_near, near_dist);
        let n3 = vec3<f32>(-half_w_near, half_h_near, near_dist);
        let f0 = vec3<f32>(-half_w_far, -half_h_far, far_dist);
        let f1 = vec3<f32>(half_w_far, -half_h_far, far_dist);
        let f2 = vec3<f32>(half_w_far, half_h_far, far_dist);
        let f3 = vec3<f32>(-half_w_far, half_h_far, far_dist);

        if (edge_index == 0u) {
            edge.start = n0;
            edge.end = n1;
        } else if (edge_index == 1u) {
            edge.start = n1;
            edge.end = n2;
        } else if (edge_index == 2u) {
            edge.start = n2;
            edge.end = n3;
        } else if (edge_index == 3u) {
            edge.start = n3;
            edge.end = n0;
        } else if (edge_index == 4u) {
            edge.start = f0;
            edge.end = f1;
        } else if (edge_index == 5u) {
            edge.start = f1;
            edge.end = f2;
        } else if (edge_index == 6u) {
            edge.start = f2;
            edge.end = f3;
        } else if (edge_index == 7u) {
            edge.start = f3;
            edge.end = f0;
        } else if (edge_index == 8u) {
            edge.start = n0;
            edge.end = f0;
        } else if (edge_index == 9u) {
            edge.start = n1;
            edge.end = f1;
        } else if (edge_index == 10u) {
            edge.start = n2;
            edge.end = f2;
        } else if (edge_index == 11u) {
            edge.start = n3;
            edge.end = f3;
        } else if (edge_index == 12u) {
            edge.start = vec3<f32>(0.0, 0.0, 0.0);
            edge.end = vec3<f32>(0.0, 0.0, far_dist * 0.6);
        } else {
            edge.valid = false;
        }
    } else if (kind == ICON_KIND_LIGHT_SPOT) {
        let angle = clamp(params.x, 0.01, 3.12);
        let length = max(size, 0.001);
        let radius = tan(angle * 0.5) * length;
        let f0 = vec3<f32>(-radius, -radius, length);
        let f1 = vec3<f32>(radius, -radius, length);
        let f2 = vec3<f32>(radius, radius, length);
        let f3 = vec3<f32>(-radius, radius, length);

        if (edge_index == 0u) {
            edge.start = f0;
            edge.end = f1;
        } else if (edge_index == 1u) {
            edge.start = f1;
            edge.end = f2;
        } else if (edge_index == 2u) {
            edge.start = f2;
            edge.end = f3;
        } else if (edge_index == 3u) {
            edge.start = f3;
            edge.end = f0;
        } else if (edge_index == 4u) {
            edge.start = vec3<f32>(0.0, 0.0, 0.0);
            edge.end = f0;
        } else if (edge_index == 5u) {
            edge.start = vec3<f32>(0.0, 0.0, 0.0);
            edge.end = f1;
        } else if (edge_index == 6u) {
            edge.start = vec3<f32>(0.0, 0.0, 0.0);
            edge.end = f2;
        } else if (edge_index == 7u) {
            edge.start = vec3<f32>(0.0, 0.0, 0.0);
            edge.end = f3;
        } else if (edge_index == 8u) {
            edge.start = vec3<f32>(0.0, 0.0, 0.0);
            edge.end = vec3<f32>(0.0, 0.0, length);
        } else {
            edge.valid = false;
        }
    } else if (kind == ICON_KIND_LIGHT_POINT) {
        let radius = max(size, 0.001) * 0.4;
        if (edge_index == 0u) {
            edge.start = vec3<f32>(-radius, 0.0, 0.0);
            edge.end = vec3<f32>(radius, 0.0, 0.0);
        } else if (edge_index == 1u) {
            edge.start = vec3<f32>(0.0, -radius, 0.0);
            edge.end = vec3<f32>(0.0, radius, 0.0);
        } else if (edge_index == 2u) {
            edge.start = vec3<f32>(0.0, 0.0, -radius);
            edge.end = vec3<f32>(0.0, 0.0, radius);
        } else {
            edge.valid = false;
        }
    } else if (kind == ICON_KIND_LIGHT_DIRECTIONAL) {
        let length = max(size, 0.001) * 0.9;
        let head_len = max(size * 0.25, 0.001);
        let head_width = size * 0.2;
        let tip = vec3<f32>(0.0, 0.0, length);
        let base = vec3<f32>(0.0, 0.0, length - head_len);
        let left = base + vec3<f32>(head_width, 0.0, 0.0);
        let right = base - vec3<f32>(head_width, 0.0, 0.0);
        let disc = size * 0.2;
        let d0 = vec3<f32>(-disc, -disc, 0.0);
        let d1 = vec3<f32>(disc, -disc, 0.0);
        let d2 = vec3<f32>(disc, disc, 0.0);
        let d3 = vec3<f32>(-disc, disc, 0.0);

        if (edge_index == 0u) {
            edge.start = vec3<f32>(0.0, 0.0, 0.0);
            edge.end = tip;
        } else if (edge_index == 1u) {
            edge.start = tip;
            edge.end = left;
        } else if (edge_index == 2u) {
            edge.start = tip;
            edge.end = right;
        } else if (edge_index == 3u) {
            edge.start = d0;
            edge.end = d1;
        } else if (edge_index == 4u) {
            edge.start = d1;
            edge.end = d2;
        } else if (edge_index == 5u) {
            edge.start = d2;
            edge.end = d3;
        } else if (edge_index == 6u) {
            edge.start = d3;
            edge.end = d0;
        } else {
            edge.valid = false;
        }
    } else if (kind == ICON_KIND_AUDIO_EMITTER) {
        let half_h = max(size * 0.32, 0.001);
        let body_x = -max(size * 0.24, 0.001);
        let cone_x = max(size * 0.25, 0.001);
        let mid = half_h * 0.65;
        let p0 = vec3<f32>(body_x, -half_h, 0.0);
        let p1 = vec3<f32>(body_x, half_h, 0.0);
        let p2 = vec3<f32>(0.0, mid, 0.0);
        let p3 = vec3<f32>(cone_x, 0.0, 0.0);
        let p4 = vec3<f32>(0.0, -mid, 0.0);
        let r1 = max(size * 0.24, 0.001);
        let r2 = max(size * 0.36, 0.001);
        let r3 = max(size * 0.48, 0.001);
        let w1a = p3 + vec3<f32>(r1 * 0.72, r1 * 0.72, 0.0);
        let w1m = p3 + vec3<f32>(r1, 0.0, 0.0);
        let w1b = p3 + vec3<f32>(r1 * 0.72, -r1 * 0.72, 0.0);
        let w2a = p3 + vec3<f32>(r2 * 0.72, r2 * 0.72, 0.0);
        let w2m = p3 + vec3<f32>(r2, 0.0, 0.0);
        let w2b = p3 + vec3<f32>(r2 * 0.72, -r2 * 0.72, 0.0);
        let w3a = p3 + vec3<f32>(r3 * 0.72, r3 * 0.72, 0.0);
        let w3m = p3 + vec3<f32>(r3, 0.0, 0.0);
        let w3b = p3 + vec3<f32>(r3 * 0.72, -r3 * 0.72, 0.0);

        if (edge_index == 0u) {
            edge.start = p0;
            edge.end = p1;
        } else if (edge_index == 1u) {
            edge.start = p1;
            edge.end = p2;
        } else if (edge_index == 2u) {
            edge.start = p2;
            edge.end = p3;
        } else if (edge_index == 3u) {
            edge.start = p3;
            edge.end = p4;
        } else if (edge_index == 4u) {
            edge.start = p4;
            edge.end = p0;
        } else if (edge_index == 5u) {
            edge.start = w1a;
            edge.end = w1m;
        } else if (edge_index == 6u) {
            edge.start = w1m;
            edge.end = w1b;
        } else if (edge_index == 7u) {
            edge.start = w2a;
            edge.end = w2m;
        } else if (edge_index == 8u) {
            edge.start = w2m;
            edge.end = w2b;
        } else if (edge_index == 9u) {
            edge.start = w3a;
            edge.end = w3m;
        } else if (edge_index == 10u) {
            edge.start = w3m;
            edge.end = w3b;
        } else {
            edge.valid = false;
        }
    } else {
        edge.valid = false;
    }

    return edge;
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
    let plane_total = PLANE_COUNT * PLANE_VERTS_PER_HANDLE;
    let rotate_total = ring_verts_per_axis * AXIS_COUNT;
    let selection_total = select(0u, SELECTION_EDGE_COUNT * SELECTION_VERTS_PER_EDGE, gizmo.selection_enabled != 0u);
    let outline_total = gizmo.outline_meta.x * OUTLINE_VERTS_PER_LINE;
    let icon_total = gizmo.icon_meta.x * ICON_VERTS_PER_GIZMO;
    var gizmo_total = 0u;
    if (gizmo.mode == 1u) {
        gizmo_total = translate_total + plane_total + ORIGIN_VERTS;
    } else if (gizmo.mode == 2u) {
        gizmo_total = rotate_total + ORIGIN_VERTS;
    } else if (gizmo.mode == 3u) {
        gizmo_total = scale_total + plane_total + ORIGIN_VERTS;
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
                let axis = axis_color(axis_index);
                color = highlight_color(axis.xyz, axis_id);
                alpha = axis.w;
            } else if (vertex_index < rotate_total + ORIGIN_VERTS) {
                let local_index = vertex_index - rotate_total;
                let basis = billboard_basis(view_dir);
                let dot_size = max(axis_len * gizmo.origin_params.x, gizmo.origin_params.y);
                let corner = quad_corner_signed(local_index);
                pos = gizmo.origin
                    + basis.right * (corner.x * dot_size)
                    + basis.up * (corner.y * dot_size);
                let origin_color = gizmo.origin_color;
                color = highlight_color(origin_color.xyz, CENTER_AXIS_ID);
                alpha = origin_color.w;
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

                let axis = axis_color(axis_index);
                color = highlight_color(axis.xyz, axis_id);
                alpha = axis.w;
            } else if (vertex_index < scale_total + plane_total) {
                let local_index = vertex_index - scale_total;
                let plane_index = local_index / PLANE_VERTS_PER_HANDLE;
                let corner = local_index % PLANE_VERTS_PER_HANDLE;
                let corner_uv = plane_corner(corner);
                let plane_id = plane_axis_id(plane_index);
                let axis_u = quat_rotate(gizmo.rotation, plane_axis_u(plane_index));
                let axis_v = quat_rotate(gizmo.rotation, plane_axis_v(plane_index));
                let offset = axis_len * gizmo.plane_params.x;
                let plane_size = axis_len * gizmo.plane_params.y;
                let plane_pos = vec2<f32>(offset, offset) + corner_uv * plane_size;
                pos = gizmo.origin + axis_u * plane_pos.x + axis_v * plane_pos.y;
                let plane_color_rgba = plane_color(plane_index);
                color = highlight_color(plane_color_rgba.xyz, plane_id);
                alpha = plane_color_rgba.w;
            } else if (vertex_index < scale_total + plane_total + ORIGIN_VERTS) {
                let local_index = vertex_index - scale_total - plane_total;
                let basis = billboard_basis(view_dir);
                let dot_size = max(axis_len * gizmo.origin_params.x, gizmo.origin_params.y);
                let corner = quad_corner_signed(local_index);
                pos = gizmo.origin
                    + basis.right * (corner.x * dot_size)
                    + basis.up * (corner.y * dot_size);
                let origin_color = gizmo.origin_color;
                color = highlight_color(origin_color.xyz, CENTER_AXIS_ID);
                alpha = origin_color.w;
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

                let axis = axis_color(axis_index);
                color = highlight_color(axis.xyz, axis_id);
                alpha = axis.w;
            } else if (vertex_index < translate_total + plane_total) {
                let local_index = vertex_index - translate_total;
                let plane_index = local_index / PLANE_VERTS_PER_HANDLE;
                let corner = local_index % PLANE_VERTS_PER_HANDLE;
                let corner_uv = plane_corner(corner);
                let plane_id = plane_axis_id(plane_index);
                let axis_u = quat_rotate(gizmo.rotation, plane_axis_u(plane_index));
                let axis_v = quat_rotate(gizmo.rotation, plane_axis_v(plane_index));
                let offset = axis_len * gizmo.plane_params.x;
                let plane_size = axis_len * gizmo.plane_params.y;
                let plane_pos = vec2<f32>(offset, offset) + corner_uv * plane_size;
                pos = gizmo.origin + axis_u * plane_pos.x + axis_v * plane_pos.y;
                let plane_color_rgba = plane_color(plane_index);
                color = highlight_color(plane_color_rgba.xyz, plane_id);
                alpha = plane_color_rgba.w;
            } else if (vertex_index < translate_total + plane_total + ORIGIN_VERTS) {
                let local_index = vertex_index - translate_total - plane_total;
                let basis = billboard_basis(view_dir);
                let dot_size = max(axis_len * gizmo.origin_params.x, gizmo.origin_params.y);
                let corner = quad_corner_signed(local_index);
                pos = gizmo.origin
                    + basis.right * (corner.x * dot_size)
                    + basis.up * (corner.y * dot_size);
                let origin_color = gizmo.origin_color;
                color = highlight_color(origin_color.xyz, CENTER_AXIS_ID);
                alpha = origin_color.w;
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
        let segment_mid = (start_world + end_world) * 0.5;
        let segment_view = safe_normalize(camera.view_position - segment_mid);
        let side = axis_side(edge_dir, segment_view);
        let quad = quad_corner_along(corner_index);
        pos = mix(start_world, end_world, quad.x) + side * (quad.y * gizmo.selection_thickness);
        color = gizmo.selection_color.xyz;
        alpha = gizmo.selection_color.w;
    } else if (vertex_index < gizmo_total + selection_total + outline_total) {
        let local_index = vertex_index - gizmo_total - selection_total;
        let line_index = local_index / OUTLINE_VERTS_PER_LINE;
        let corner_index = local_index % OUTLINE_VERTS_PER_LINE;

        if (line_index < gizmo.outline_meta.x) {
            let line = gizmo_outline_lines.lines[line_index];
            let start_world = line.start;
            let end_world = line.end;
            let segment_mid = (start_world + end_world) * 0.5;
            let segment_view = safe_normalize(camera.view_position - segment_mid);
            let thickness = max(gizmo.size * gizmo.outline_line_params.x, gizmo.outline_line_params.y);
            pos = line_vertex(corner_index, start_world, end_world, segment_view, thickness);
            color = gizmo.outline_color.xyz;
            alpha = gizmo.outline_color.w;
        } else {
            alpha = 0.0;
        }
    } else if (vertex_index < gizmo_total + selection_total + outline_total + icon_total) {
        let local_index = vertex_index - gizmo_total - selection_total - outline_total;
        let icon_index = local_index / ICON_VERTS_PER_GIZMO;
        let icon_local = local_index % ICON_VERTS_PER_GIZMO;
        let edge_index = icon_local / ICON_VERTS_PER_EDGE;
        let corner_index = icon_local % ICON_VERTS_PER_EDGE;

        if (icon_index < gizmo.icon_meta.x) {
            let icon = gizmo_icons.icons[icon_index];
            let thickness = max(icon.size_params.x * gizmo.icon_line_params.x, gizmo.icon_line_params.y);
            let edge = icon_edge(icon.kind, edge_index, icon.size_params.x, icon.params);
            if (edge.valid) {
                let start_world = icon.position + quat_rotate(icon.rotation, edge.start);
                let end_world = icon.position + quat_rotate(icon.rotation, edge.end);
                let segment_mid = (start_world + end_world) * 0.5;
                let segment_view = safe_normalize(camera.view_position - segment_mid);
                pos = line_vertex(corner_index, start_world, end_world, segment_view, thickness);
                color = icon.color.xyz;
                alpha = icon.color.w;
            } else {
                alpha = 0.0;
            }
        } else {
            alpha = 0.0;
        }
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
