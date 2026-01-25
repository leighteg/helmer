use bevy_ecs::prelude::{Entity, Query, Res, ResMut, Resource, With};
use glam::{Mat4, Quat, Vec3};
use winit::event::MouseButton;

use helmer::graphics::common::renderer::{Aabb, GizmoAxis, GizmoData, GizmoMode, GizmoStyle};
use helmer::provided::components::{Camera, Transform};
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::render_system::RenderGizmoState;
use helmer_becs::{
    BevyActiveCamera, BevyAssetServer, BevyCamera, BevyInputManager, BevyMeshRenderer,
    BevyTransform,
};

use crate::editor::scene::{EditorEntity, EditorSceneState, WorldState};
use crate::editor::{
    EditorUndoState, EditorViewportState, request_begin_undo_group, request_end_undo_group,
};

#[derive(Resource, Debug, Clone)]
pub struct EditorGizmoState {
    pub mode: GizmoMode,
    pub hover_axis: GizmoAxis,
    pub active_axis: GizmoAxis,
    last_mouse_down: bool,
    drag: Option<GizmoDragState>,
}

impl Default for EditorGizmoState {
    fn default() -> Self {
        Self {
            mode: GizmoMode::Translate,
            hover_axis: GizmoAxis::None,
            active_axis: GizmoAxis::None,
            last_mouse_down: false,
            drag: None,
        }
    }
}

#[derive(Resource, Debug, Clone)]
pub struct EditorSelectionState {
    last_mouse_down: bool,
}

impl Default for EditorSelectionState {
    fn default() -> Self {
        Self {
            last_mouse_down: false,
        }
    }
}

#[derive(Resource, Debug, Clone)]
pub struct EditorGizmoSettings {
    pub size_scale: f32,
    pub size_min: f32,
    pub size_max: f32,
    pub axis_pick_radius_scale: f32,
    pub axis_pick_radius_min: f32,
    pub center_pick_radius_scale: f32,
    pub center_pick_radius_min: f32,
    pub rotate_pick_radius_scale: f32,
    pub rotate_pick_radius_min: f32,
    pub scale_min: f32,
    pub ring_segments: u32,
    pub translate_thickness_scale: f32,
    pub translate_thickness_min: f32,
    pub translate_head_length_scale: f32,
    pub translate_head_width_scale: f32,
    pub scale_thickness_scale: f32,
    pub scale_thickness_min: f32,
    pub scale_head_length_scale: f32,
    pub scale_box_scale: f32,
    pub rotate_radius_scale: f32,
    pub rotate_thickness_scale: f32,
    pub rotate_thickness_min: f32,
    pub origin_size_scale: f32,
    pub origin_size_min: f32,
    pub axis_color_x: [f32; 3],
    pub axis_color_y: [f32; 3],
    pub axis_color_z: [f32; 3],
    pub origin_color: [f32; 3],
    pub selection_thickness_scale: f32,
    pub selection_thickness_min: f32,
    pub selection_color: [f32; 3],
    pub hover_mix: f32,
    pub active_mix: f32,
}

impl Default for EditorGizmoSettings {
    fn default() -> Self {
        Self {
            size_scale: 0.12,
            size_min: 0.25,
            size_max: 100.0,
            axis_pick_radius_scale: 0.14,
            axis_pick_radius_min: 0.05,
            center_pick_radius_scale: 0.18,
            center_pick_radius_min: 0.06,
            rotate_pick_radius_scale: 0.08,
            rotate_pick_radius_min: 0.04,
            scale_min: 0.01,
            ring_segments: 32,
            translate_thickness_scale: 0.05,
            translate_thickness_min: 0.015,
            translate_head_length_scale: 0.22,
            translate_head_width_scale: 2.6,
            scale_thickness_scale: 0.05,
            scale_thickness_min: 0.015,
            scale_head_length_scale: 0.18,
            scale_box_scale: 2.0,
            rotate_radius_scale: 0.85,
            rotate_thickness_scale: 0.03,
            rotate_thickness_min: 0.01,
            origin_size_scale: 0.06,
            origin_size_min: 0.03,
            axis_color_x: [1.0, 0.2, 0.2],
            axis_color_y: [0.2, 1.0, 0.2],
            axis_color_z: [0.2, 0.4, 1.0],
            origin_color: [0.9, 0.9, 0.9],
            selection_thickness_scale: 0.03,
            selection_thickness_min: 0.01,
            selection_color: [1.0, 0.85, 0.2],
            hover_mix: 0.3,
            active_mix: 0.5,
        }
    }
}

impl EditorGizmoSettings {
    fn to_style(&self) -> GizmoStyle {
        GizmoStyle {
            ring_segments: self.ring_segments.max(3),
            translate_thickness_scale: self.translate_thickness_scale,
            translate_thickness_min: self.translate_thickness_min,
            translate_head_length_scale: self.translate_head_length_scale,
            translate_head_width_scale: self.translate_head_width_scale,
            scale_thickness_scale: self.scale_thickness_scale,
            scale_thickness_min: self.scale_thickness_min,
            scale_head_length_scale: self.scale_head_length_scale,
            scale_box_scale: self.scale_box_scale,
            rotate_radius_scale: self.rotate_radius_scale,
            rotate_thickness_scale: self.rotate_thickness_scale,
            rotate_thickness_min: self.rotate_thickness_min,
            origin_size_scale: self.origin_size_scale,
            origin_size_min: self.origin_size_min,
            axis_color_x: self.axis_color_x,
            axis_color_y: self.axis_color_y,
            axis_color_z: self.axis_color_z,
            origin_color: self.origin_color,
            selection_thickness_scale: self.selection_thickness_scale,
            selection_thickness_min: self.selection_thickness_min,
            selection_color: self.selection_color,
            hover_mix: self.hover_mix,
            active_mix: self.active_mix,
        }
    }

    fn size_bounds(&self) -> (f32, f32) {
        if self.size_min <= self.size_max {
            (self.size_min, self.size_max)
        } else {
            (self.size_max, self.size_min)
        }
    }

    pub fn sanitize(&mut self) {
        if self.size_min > self.size_max {
            std::mem::swap(&mut self.size_min, &mut self.size_max);
        }
        if self.ring_segments < 3 {
            self.ring_segments = 3;
        }
    }
}

#[derive(Debug, Clone)]
struct GizmoDragState {
    axis: GizmoAxis,
    mode: GizmoMode,
    start_transform: Transform,
    kind: GizmoDragKind,
}

#[derive(Debug, Clone)]
enum GizmoDragKind {
    AxisLine {
        axis_dir: Vec3,
        start_axis_param: f32,
    },
    AxisRotate {
        axis_dir: Vec3,
        start_vector: Vec3,
    },
    CenterTranslate {
        plane_normal: Vec3,
        start_hit: Vec3,
    },
    CenterScale {
        plane_normal: Vec3,
        start_distance: f32,
    },
    CenterRotate {
        plane_normal: Vec3,
        start_vector: Vec3,
    },
}

pub fn gizmo_system(
    mut state: ResMut<EditorGizmoState>,
    mut render_gizmo: ResMut<RenderGizmoState>,
    settings: Res<EditorGizmoSettings>,
    viewport_state: Res<EditorViewportState>,
    selection: Res<InspectorSelectedEntityResource>,
    scene_state: Res<EditorSceneState>,
    mut undo_state: ResMut<EditorUndoState>,
    input: Res<BevyInputManager>,
    asset_server: Res<BevyAssetServer>,
    mesh_query: Query<&BevyMeshRenderer>,
    mut transforms: Query<&mut BevyTransform>,
    camera_query: Query<(Entity, &BevyCamera), With<BevyActiveCamera>>,
) {
    if scene_state.world_state != WorldState::Edit && !viewport_state.gizmos_in_play {
        clear_gizmo(&mut state, &mut render_gizmo, &settings);
        return;
    }
    let allow_undo = scene_state.world_state == WorldState::Edit;

    let Some(entity) = selection.0 else {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        clear_gizmo(&mut state, &mut render_gizmo, &settings);
        return;
    };

    let Some((camera_entity, camera)) = camera_query.iter().next() else {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        clear_gizmo(&mut state, &mut render_gizmo, &settings);
        return;
    };

    let camera_transform = match transforms.get_mut(camera_entity) {
        Ok(transform) => transform.0,
        Err(_) => {
            if state.drag.is_some() && allow_undo {
                request_end_undo_group(&mut undo_state);
            }
            clear_gizmo(&mut state, &mut render_gizmo, &settings);
            return;
        }
    };

    let Ok(mut target_transform) = transforms.get_mut(entity) else {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        clear_gizmo(&mut state, &mut render_gizmo, &settings);
        return;
    };

    let input_manager = input.0.read();
    let wants_pointer = input_manager.egui_wants_pointer;
    let left_down = input_manager.is_mouse_button_active(MouseButton::Left);
    let left_pressed = left_down && !state.last_mouse_down;
    let left_released = !left_down && state.last_mouse_down;
    state.last_mouse_down = left_down;

    let inv_view_proj = camera_inv_view_proj(&camera.0, &camera_transform);
    let ray = if wants_pointer {
        None
    } else {
        screen_ray(
            input_manager.cursor_position,
            input_manager.window_size,
            inv_view_proj,
            camera_transform.position,
        )
    };

    let origin = target_transform.0.position;
    let rotation = target_transform.0.rotation;
    let (size_min, size_max) = settings.size_bounds();
    let distance = camera_transform.position.distance(origin).max(0.001);
    let gizmo_size = (distance * settings.size_scale).clamp(size_min, size_max);
    let mut view_dir = camera_transform.position - origin;
    if view_dir.length_squared() < 1.0e-6 {
        view_dir = camera_transform.forward();
    }
    let view_dir = view_dir.normalize_or_zero();

    if wants_pointer {
        let had_drag = state.drag.is_some();
        if left_released || had_drag {
            state.drag = None;
            state.active_axis = GizmoAxis::None;
        }
        if had_drag && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        state.hover_axis = GizmoAxis::None;
    } else if let Some((ray_origin, ray_dir)) = ray {
        if state.drag.is_none() && state.mode != GizmoMode::None {
            state.hover_axis = pick_gizmo(
                state.mode, ray_origin, ray_dir, origin, rotation, gizmo_size, &settings,
            );
        }

        if left_pressed
            && state.drag.is_none()
            && state.mode != GizmoMode::None
            && state.hover_axis != GizmoAxis::None
        {
            if let Some(drag_state) = begin_drag(
                state.mode,
                state.hover_axis,
                ray_origin,
                ray_dir,
                origin,
                view_dir,
                target_transform.0,
                &settings,
            ) {
                state.active_axis = drag_state.axis;
                state.drag = Some(drag_state);
                if allow_undo {
                    request_begin_undo_group(&mut undo_state, undo_label_for_mode(state.mode));
                }
            }
        }

        if let Some(drag) = state.drag.take() {
            if left_down {
                apply_drag(
                    &mut target_transform.0,
                    &drag,
                    ray_origin,
                    ray_dir,
                    &settings,
                );
                state.active_axis = drag.axis;
                state.drag = Some(drag);
            } else {
                state.active_axis = GizmoAxis::None;
                state.drag = None;
                if allow_undo {
                    request_end_undo_group(&mut undo_state);
                }
            }
        }
    } else if left_released {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        state.active_axis = GizmoAxis::None;
        state.drag = None;
    }

    if state.mode == GizmoMode::None {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        state.drag = None;
        state.hover_axis = GizmoAxis::None;
        state.active_axis = GizmoAxis::None;
    }

    let output_distance = camera_transform
        .position
        .distance(target_transform.0.position)
        .max(0.001);
    let output_size = (output_distance * settings.size_scale).clamp(size_min, size_max);

    let (selection_enabled, selection_min, selection_max) =
        match selection_bounds(&asset_server, &mesh_query, entity) {
            Some(bounds) => (true, bounds.min, bounds.max),
            None => (false, Vec3::ZERO, Vec3::ZERO),
        };

    render_gizmo.0 = GizmoData {
        mode: state.mode,
        position: target_transform.0.position,
        rotation: target_transform.0.rotation,
        scale: target_transform.0.scale,
        size: output_size,
        hover_axis: state.hover_axis,
        active_axis: state.active_axis,
        selection_enabled,
        selection_min,
        selection_max,
        style: settings.to_style(),
    };
}

pub fn selection_system(
    mut selection_state: ResMut<EditorSelectionState>,
    mut selection: ResMut<InspectorSelectedEntityResource>,
    gizmo_state: Res<EditorGizmoState>,
    viewport_state: Res<EditorViewportState>,
    scene_state: Res<EditorSceneState>,
    input: Res<BevyInputManager>,
    asset_server: Res<BevyAssetServer>,
    mesh_query: Query<(Entity, &BevyMeshRenderer, &BevyTransform), With<EditorEntity>>,
    camera_query: Query<(&BevyCamera, &BevyTransform), With<BevyActiveCamera>>,
) {
    if scene_state.world_state != WorldState::Edit && !viewport_state.gizmos_in_play {
        return;
    }

    let input_manager = input.0.read();
    let left_down = input_manager.is_mouse_button_active(MouseButton::Left);
    let left_pressed = left_down && !selection_state.last_mouse_down;
    selection_state.last_mouse_down = left_down;

    if input_manager.egui_wants_pointer || !left_pressed {
        return;
    }

    if gizmo_state.active_axis != GizmoAxis::None {
        return;
    }

    let Some((camera, camera_transform)) = camera_query.iter().next() else {
        return;
    };

    let inv_view_proj = camera_inv_view_proj(&camera.0, &camera_transform.0);
    let Some((ray_origin, ray_dir)) = screen_ray(
        input_manager.cursor_position,
        input_manager.window_size,
        inv_view_proj,
        camera_transform.0.position,
    ) else {
        return;
    };

    let asset_server = asset_server.0.lock();
    let mesh_aabb_map = asset_server.mesh_aabb_map.read();

    let mut best: Option<(Entity, f32)> = None;
    for (entity, renderer, transform) in mesh_query.iter() {
        let Some(bounds) = mesh_aabb_map.0.get(&renderer.0.mesh_id) else {
            continue;
        };
        let Some(distance) = ray_aabb_intersection(ray_origin, ray_dir, &transform.0, bounds)
        else {
            continue;
        };
        if best.map_or(true, |(_, best_distance)| distance < best_distance) {
            best = Some((entity, distance));
        }
    }

    selection.0 = best.map(|(entity, _)| entity);
}

fn clear_gizmo(
    state: &mut EditorGizmoState,
    render_gizmo: &mut RenderGizmoState,
    settings: &EditorGizmoSettings,
) {
    state.drag = None;
    state.hover_axis = GizmoAxis::None;
    state.active_axis = GizmoAxis::None;
    render_gizmo.0 = GizmoData {
        style: settings.to_style(),
        ..GizmoData::default()
    };
}

fn undo_label_for_mode(mode: GizmoMode) -> &'static str {
    match mode {
        GizmoMode::Translate => "Move",
        GizmoMode::Rotate => "Rotate",
        GizmoMode::Scale => "Scale",
        GizmoMode::None => "Edit",
    }
}

fn selection_bounds(
    asset_server: &BevyAssetServer,
    mesh_query: &Query<&BevyMeshRenderer>,
    entity: Entity,
) -> Option<Aabb> {
    let renderer = mesh_query.get(entity).ok()?;
    let asset_server = asset_server.0.lock();
    let mesh_aabb_map = asset_server.mesh_aabb_map.read();
    mesh_aabb_map.0.get(&renderer.0.mesh_id).copied()
}

fn camera_inv_view_proj(camera: &Camera, transform: &Transform) -> Mat4 {
    let forward = transform.forward().normalize_or_zero();
    let up = transform.up().normalize_or_zero();
    let view = Mat4::look_at_rh(transform.position, transform.position + forward, up);
    let projection = Mat4::perspective_infinite_reverse_rh(
        camera.fov_y_rad.max(0.001),
        camera.aspect_ratio.max(0.001),
        camera.near_plane.max(0.001),
    );
    let view_proj = projection * view;
    view_proj.inverse()
}

fn screen_ray(
    cursor: glam::DVec2,
    window_size: glam::UVec2,
    inv_view_proj: Mat4,
    camera_position: Vec3,
) -> Option<(Vec3, Vec3)> {
    if window_size.x == 0 || window_size.y == 0 {
        return None;
    }
    let x = (cursor.x as f32 / window_size.x as f32) * 2.0 - 1.0;
    let y = 1.0 - (cursor.y as f32 / window_size.y as f32) * 2.0;
    let ndc_near = Vec3::new(x, y, 1.0);
    let near = inv_view_proj * ndc_near.extend(1.0);
    if near.w.abs() < 1.0e-6 {
        return None;
    }
    let near_pos = near.truncate() / near.w;
    let dir = (near_pos - camera_position).normalize_or_zero();
    if dir.length_squared() < 1.0e-6 {
        return None;
    }
    Some((camera_position, dir))
}

fn ray_aabb_intersection(
    ray_origin: Vec3,
    ray_dir: Vec3,
    transform: &Transform,
    bounds: &Aabb,
) -> Option<f32> {
    let inv = transform.to_matrix().inverse();
    let local_origin = (inv * ray_origin.extend(1.0)).truncate();
    let local_dir = (inv * ray_dir.extend(0.0)).truncate();
    if local_dir.length_squared() < 1.0e-12 {
        return None;
    }

    let mut tmin: f32 = 0.0;
    let mut tmax: f32 = f32::MAX;

    let (origin, dir, min, max) = (local_origin.x, local_dir.x, bounds.min.x, bounds.max.x);
    if dir.abs() < 1.0e-6 {
        if origin < min || origin > max {
            return None;
        }
    } else {
        let mut t1 = (min - origin) / dir;
        let mut t2 = (max - origin) / dir;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        tmin = tmin.max(t1);
        tmax = tmax.min(t2);
        if tmin > tmax {
            return None;
        }
    }

    let (origin, dir, min, max) = (local_origin.y, local_dir.y, bounds.min.y, bounds.max.y);
    if dir.abs() < 1.0e-6 {
        if origin < min || origin > max {
            return None;
        }
    } else {
        let mut t1 = (min - origin) / dir;
        let mut t2 = (max - origin) / dir;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        tmin = tmin.max(t1);
        tmax = tmax.min(t2);
        if tmin > tmax {
            return None;
        }
    }

    let (origin, dir, min, max) = (local_origin.z, local_dir.z, bounds.min.z, bounds.max.z);
    if dir.abs() < 1.0e-6 {
        if origin < min || origin > max {
            return None;
        }
    } else {
        let mut t1 = (min - origin) / dir;
        let mut t2 = (max - origin) / dir;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        tmin = tmin.max(t1);
        tmax = tmax.min(t2);
        if tmin > tmax {
            return None;
        }
    }

    let hit_local = local_origin + local_dir * tmin.max(0.0);
    let hit_world = (transform.to_matrix() * hit_local.extend(1.0)).truncate();
    let distance = (hit_world - ray_origin).dot(ray_dir);
    if distance.is_finite() && distance >= 0.0 {
        Some(distance)
    } else {
        None
    }
}

fn pick_gizmo(
    mode: GizmoMode,
    ray_origin: Vec3,
    ray_dir: Vec3,
    origin: Vec3,
    rotation: Quat,
    size: f32,
    settings: &EditorGizmoSettings,
) -> GizmoAxis {
    if mode == GizmoMode::None {
        return GizmoAxis::None;
    }

    let axes = [
        (GizmoAxis::X, rotation * Vec3::X),
        (GizmoAxis::Y, rotation * Vec3::Y),
        (GizmoAxis::Z, rotation * Vec3::Z),
    ];

    let mut best_axis = GizmoAxis::None;
    let mut best_score = f32::MAX;

    let center_radius =
        (size * settings.center_pick_radius_scale).max(settings.center_pick_radius_min);
    let center_dist = ray_point_distance(ray_origin, ray_dir, origin);
    if center_dist <= center_radius {
        return GizmoAxis::Center;
    }

    match mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            let threshold =
                (size * settings.axis_pick_radius_scale).max(settings.axis_pick_radius_min);
            for (axis, dir) in axes {
                if let Some((axis_t, distance)) =
                    closest_point_on_axis(ray_origin, ray_dir, origin, dir)
                {
                    if axis_t < 0.0 || axis_t > size {
                        continue;
                    }
                    if distance <= threshold {
                        let score = distance / threshold;
                        if score < best_score {
                            best_score = score;
                            best_axis = axis;
                        }
                    }
                }
            }
        }
        GizmoMode::Rotate => {
            let ring_radius = size * settings.rotate_radius_scale;
            let band =
                (size * settings.rotate_pick_radius_scale).max(settings.rotate_pick_radius_min);
            for (axis, dir) in axes {
                if let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, dir) {
                    let dist = (hit - origin).length();
                    let delta = (dist - ring_radius).abs();
                    if delta <= band {
                        let score = delta / band;
                        if score < best_score {
                            best_score = score;
                            best_axis = axis;
                        }
                    }
                }
            }
        }
        GizmoMode::None => {}
    }

    best_axis
}

fn begin_drag(
    mode: GizmoMode,
    axis: GizmoAxis,
    ray_origin: Vec3,
    ray_dir: Vec3,
    origin: Vec3,
    view_dir: Vec3,
    start_transform: Transform,
    settings: &EditorGizmoSettings,
) -> Option<GizmoDragState> {
    if axis == GizmoAxis::Center {
        let plane_normal = view_dir.normalize_or_zero();
        match mode {
            GizmoMode::Translate => {
                let hit = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal)?;
                return Some(GizmoDragState {
                    axis,
                    mode,
                    start_transform,
                    kind: GizmoDragKind::CenterTranslate {
                        plane_normal,
                        start_hit: hit,
                    },
                });
            }
            GizmoMode::Scale => {
                let hit = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal)?;
                let start_distance = (hit - origin).length().max(settings.scale_min);
                return Some(GizmoDragState {
                    axis,
                    mode,
                    start_transform,
                    kind: GizmoDragKind::CenterScale {
                        plane_normal,
                        start_distance,
                    },
                });
            }
            GizmoMode::Rotate => {
                let hit = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal)?;
                let vec = (hit - origin).normalize_or_zero();
                if vec.length_squared() < 1.0e-6 {
                    return None;
                }
                return Some(GizmoDragState {
                    axis,
                    mode,
                    start_transform,
                    kind: GizmoDragKind::CenterRotate {
                        plane_normal,
                        start_vector: vec,
                    },
                });
            }
            GizmoMode::None => return None,
        }
    }

    let axis_dir = axis_direction(start_transform.rotation, axis).normalize_or_zero();
    match mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            let start_axis_param = closest_point_on_axis(ray_origin, ray_dir, origin, axis_dir)
                .map(|(axis_t, _)| axis_t)?;
            Some(GizmoDragState {
                axis,
                mode,
                start_transform,
                kind: GizmoDragKind::AxisLine {
                    axis_dir,
                    start_axis_param,
                },
            })
        }
        GizmoMode::Rotate => {
            let hit = intersect_ray_plane(ray_origin, ray_dir, origin, axis_dir)?;
            let vec = (hit - origin).normalize_or_zero();
            if vec.length_squared() < 1.0e-6 {
                return None;
            }
            Some(GizmoDragState {
                axis,
                mode,
                start_transform,
                kind: GizmoDragKind::AxisRotate {
                    axis_dir,
                    start_vector: vec,
                },
            })
        }
        GizmoMode::None => None,
    }
}

fn apply_drag(
    transform: &mut Transform,
    drag: &GizmoDragState,
    ray_origin: Vec3,
    ray_dir: Vec3,
    settings: &EditorGizmoSettings,
) {
    let origin = drag.start_transform.position;
    match drag.mode {
        GizmoMode::Translate => match drag.kind {
            GizmoDragKind::AxisLine {
                axis_dir,
                start_axis_param,
            } => {
                if let Some((axis_t, _)) =
                    closest_point_on_axis(ray_origin, ray_dir, origin, axis_dir)
                {
                    let delta = axis_t - start_axis_param;
                    transform.position = drag.start_transform.position + axis_dir * delta;
                }
            }
            GizmoDragKind::CenterTranslate {
                plane_normal,
                start_hit,
            } => {
                if let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal) {
                    let delta = hit - start_hit;
                    transform.position = drag.start_transform.position + delta;
                }
            }
            _ => {}
        },
        GizmoMode::Scale => match drag.kind {
            GizmoDragKind::AxisLine {
                axis_dir,
                start_axis_param,
            } => {
                if let Some((axis_t, _)) =
                    closest_point_on_axis(ray_origin, ray_dir, origin, axis_dir)
                {
                    let delta = axis_t - start_axis_param;
                    let mut scale = drag.start_transform.scale;
                    match drag.axis {
                        GizmoAxis::X => scale.x = (scale.x + delta).max(settings.scale_min),
                        GizmoAxis::Y => scale.y = (scale.y + delta).max(settings.scale_min),
                        GizmoAxis::Z => scale.z = (scale.z + delta).max(settings.scale_min),
                        _ => {}
                    }
                    transform.scale = scale;
                }
            }
            GizmoDragKind::CenterScale {
                plane_normal,
                start_distance,
            } => {
                if let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal) {
                    let distance = (hit - origin).length().max(settings.scale_min);
                    let factor = distance / start_distance;
                    let mut scale = drag.start_transform.scale * factor;
                    scale.x = scale.x.max(settings.scale_min);
                    scale.y = scale.y.max(settings.scale_min);
                    scale.z = scale.z.max(settings.scale_min);
                    transform.scale = scale;
                }
            }
            _ => {}
        },
        GizmoMode::Rotate => match drag.kind {
            GizmoDragKind::AxisRotate {
                axis_dir,
                start_vector,
            } => {
                if let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, axis_dir) {
                    let current_vec = (hit - origin).normalize_or_zero();
                    if current_vec.length_squared() > 1.0e-6 {
                        let mut angle = start_vector.angle_between(current_vec);
                        let cross = start_vector.cross(current_vec);
                        if axis_dir.dot(cross) < 0.0 {
                            angle = -angle;
                        }
                        let delta = Quat::from_axis_angle(axis_dir, angle);
                        transform.rotation = delta * drag.start_transform.rotation;
                    }
                }
            }
            GizmoDragKind::CenterRotate {
                plane_normal,
                start_vector,
            } => {
                if let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal) {
                    let current_vec = (hit - origin).normalize_or_zero();
                    if current_vec.length_squared() > 1.0e-6 {
                        let mut angle = start_vector.angle_between(current_vec);
                        let cross = start_vector.cross(current_vec);
                        if plane_normal.dot(cross) < 0.0 {
                            angle = -angle;
                        }
                        let delta = Quat::from_axis_angle(plane_normal, angle);
                        transform.rotation = delta * drag.start_transform.rotation;
                    }
                }
            }
            _ => {}
        },
        GizmoMode::None => {}
    }
}

fn ray_point_distance(ray_origin: Vec3, ray_dir: Vec3, point: Vec3) -> f32 {
    let to_point = point - ray_origin;
    let t = to_point.dot(ray_dir);
    if t < 0.0 {
        return to_point.length();
    }
    let closest = ray_origin + ray_dir * t;
    closest.distance(point)
}

fn axis_direction(rotation: Quat, axis: GizmoAxis) -> Vec3 {
    match axis {
        GizmoAxis::X => rotation * Vec3::X,
        GizmoAxis::Y => rotation * Vec3::Y,
        GizmoAxis::Z => rotation * Vec3::Z,
        _ => Vec3::ZERO,
    }
}

fn closest_point_on_axis(
    ray_origin: Vec3,
    ray_dir: Vec3,
    axis_origin: Vec3,
    axis_dir: Vec3,
) -> Option<(f32, f32)> {
    let d1 = ray_dir;
    let d2 = axis_dir;
    let w0 = ray_origin - axis_origin;
    let a = d1.dot(d1);
    let b = d1.dot(d2);
    let c = d2.dot(d2);
    let d = d1.dot(w0);
    let e = d2.dot(w0);
    let denom = a * c - b * b;
    if denom.abs() < 1.0e-6 {
        return None;
    }
    let ray_t = (b * e - c * d) / denom;
    let axis_t = (a * e - b * d) / denom;
    if ray_t < 0.0 {
        return None;
    }
    let closest_ray = ray_origin + d1 * ray_t;
    let closest_axis = axis_origin + d2 * axis_t;
    let distance = closest_ray.distance(closest_axis);
    Some((axis_t, distance))
}

fn intersect_ray_plane(
    ray_origin: Vec3,
    ray_dir: Vec3,
    plane_origin: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denom = plane_normal.dot(ray_dir);
    if denom.abs() < 1.0e-6 {
        return None;
    }
    let t = plane_normal.dot(plane_origin - ray_origin) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray_origin + ray_dir * t)
}
