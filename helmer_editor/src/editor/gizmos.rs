use bevy_ecs::prelude::{DetectChanges, Entity, Query, Ref, Res, ResMut, Resource, With};
use glam::{Mat4, Quat, Vec3};
use winit::event::MouseButton;

use helmer::graphics::common::renderer::{
    Aabb, GizmoAxis, GizmoData, GizmoIcon, GizmoIconKind, GizmoLine, GizmoMode, GizmoStyle,
    MeshLodPayload,
};
use helmer::provided::components::{Camera, LightType, Transform};
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::render_system::RenderGizmoState;
use helmer_becs::{
    BevyActiveCamera, BevyAssetServer, BevyCamera, BevyInputManager, BevyLight, BevyMeshRenderer,
    BevyTransform,
};

use crate::editor::scene::{EditorEntity, EditorSceneState, WorldState};
use crate::editor::{
    EditorPlayCamera, EditorUndoState, EditorViewportState, request_begin_undo_group,
    request_end_undo_group,
};
use std::collections::HashMap;

#[derive(Resource, Debug, Clone)]
pub struct EditorGizmoState {
    pub mode: GizmoMode,
    pub hover_axis: GizmoAxis,
    pub active_axis: GizmoAxis,
    icon_revision: u64,
    outline_revision: u64,
    last_mouse_down: bool,
    drag: Option<GizmoDragState>,
}

impl Default for EditorGizmoState {
    fn default() -> Self {
        Self {
            mode: GizmoMode::Translate,
            hover_axis: GizmoAxis::None,
            active_axis: GizmoAxis::None,
            icon_revision: 0,
            outline_revision: 0,
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

#[derive(Resource, Debug, Default)]
pub struct EditorMeshOutlineCache {
    entries: HashMap<usize, MeshOutlineEntry>,
}

#[derive(Debug)]
struct MeshOutlineEntry {
    max_lines: usize,
    lines: std::sync::Arc<[GizmoLine]>,
}

#[derive(Resource, Debug, Clone)]
pub struct EditorGizmoSettings {
    pub size_scale: f32,
    pub size_min: f32,
    pub size_max: f32,
    pub icon_size_scale: f32,
    pub icon_pick_radius_scale: f32,
    pub icon_pick_radius_min: f32,
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
    pub axis_alpha: f32,
    pub origin_alpha: f32,
    pub show_bounds_outline: bool,
    pub selection_thickness_scale: f32,
    pub selection_thickness_min: f32,
    pub selection_color: [f32; 3],
    pub selection_alpha: f32,
    pub show_mesh_outline: bool,
    pub outline_thickness_scale: f32,
    pub outline_thickness_min: f32,
    pub outline_max_lines: u32,
    pub outline_color: [f32; 3],
    pub outline_alpha: f32,
    pub icon_thickness_scale: f32,
    pub icon_thickness_min: f32,
    pub camera_icon_color: [f32; 3],
    pub camera_icon_alpha: f32,
    pub active_camera_icon_color: [f32; 3],
    pub light_icon_alpha: f32,
    pub hover_mix: f32,
    pub active_mix: f32,
}

impl Default for EditorGizmoSettings {
    fn default() -> Self {
        Self {
            size_scale: 0.12,
            size_min: 0.25,
            size_max: 100.0,
            icon_size_scale: 0.6,
            icon_pick_radius_scale: 0.75,
            icon_pick_radius_min: 0.05,
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
            axis_alpha: 1.0,
            origin_alpha: 1.0,
            show_bounds_outline: true,
            selection_thickness_scale: 0.03,
            selection_thickness_min: 0.01,
            selection_color: [1.0, 0.85, 0.2],
            selection_alpha: 1.0,
            show_mesh_outline: true,
            outline_thickness_scale: 0.02,
            outline_thickness_min: 0.006,
            outline_max_lines: 9999999,
            outline_color: [0.35, 0.85, 1.0],
            outline_alpha: 1.0,
            icon_thickness_scale: 0.025,
            icon_thickness_min: 0.008,
            camera_icon_color: [0.75, 0.9, 1.0],
            camera_icon_alpha: 0.9,
            active_camera_icon_color: [1.0, 0.6, 0.2],
            light_icon_alpha: 0.9,
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
            axis_alpha: self.axis_alpha,
            origin_alpha: self.origin_alpha,
            selection_thickness_scale: self.selection_thickness_scale,
            selection_thickness_min: self.selection_thickness_min,
            selection_color: self.selection_color,
            selection_alpha: self.selection_alpha,
            outline_thickness_scale: self.outline_thickness_scale,
            outline_thickness_min: self.outline_thickness_min,
            outline_color: self.outline_color,
            outline_alpha: self.outline_alpha,
            icon_thickness_scale: self.icon_thickness_scale,
            icon_thickness_min: self.icon_thickness_min,
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
        self.axis_alpha = self.axis_alpha.clamp(0.0, 1.0);
        self.origin_alpha = self.origin_alpha.clamp(0.0, 1.0);
        self.selection_alpha = self.selection_alpha.clamp(0.0, 1.0);
        self.outline_alpha = self.outline_alpha.clamp(0.0, 1.0);
        self.camera_icon_alpha = self.camera_icon_alpha.clamp(0.0, 1.0);
        self.light_icon_alpha = self.light_icon_alpha.clamp(0.0, 1.0);
        self.hover_mix = self.hover_mix.clamp(0.0, 1.0);
        self.active_mix = self.active_mix.clamp(0.0, 1.0);
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
    mut outline_cache: ResMut<EditorMeshOutlineCache>,
    viewport_state: Res<EditorViewportState>,
    selection: Res<InspectorSelectedEntityResource>,
    scene_state: Res<EditorSceneState>,
    mut undo_state: ResMut<EditorUndoState>,
    input: Res<BevyInputManager>,
    asset_server: Res<BevyAssetServer>,
    mesh_query: Query<&BevyMeshRenderer>,
    mut transforms: Query<&mut BevyTransform>,
    camera_query: Query<(Entity, Ref<BevyCamera>), With<BevyActiveCamera>>,
    camera_icon_query: Query<
        (Entity, Ref<BevyCamera>, Option<Ref<EditorPlayCamera>>),
        With<EditorEntity>,
    >,
    light_icon_query: Query<(Entity, Ref<BevyLight>), With<EditorEntity>>,
) {
    let show_gizmos = scene_state.world_state == WorldState::Edit || viewport_state.gizmos_in_play;
    if !show_gizmos {
        clear_gizmo(&mut state, &mut render_gizmo, &settings);
        return;
    }
    let allow_undo = scene_state.world_state == WorldState::Edit;
    let prev_icon_count = render_gizmo.0.icons.len();
    let prev_outline_lines = render_gizmo.0.outline_lines.clone();

    let Some((camera_entity, camera)) = camera_query.iter().next() else {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        clear_gizmo(&mut state, &mut render_gizmo, &settings);
        return;
    };

    let camera_transform = match transforms.get_mut(camera_entity) {
        Ok(transform) => transform,
        Err(_) => {
            if state.drag.is_some() && allow_undo {
                request_end_undo_group(&mut undo_state);
            }
            clear_gizmo(&mut state, &mut render_gizmo, &settings);
            return;
        }
    };
    let camera_transform_changed = camera_transform.is_changed();
    let camera_transform = camera_transform.0;

    let mut icons_dirty =
        settings.is_changed() || viewport_state.is_changed() || camera_transform_changed;
    let icons = collect_icon_gizmos(
        &viewport_state,
        &settings,
        &camera_transform,
        &mut transforms,
        &camera_icon_query,
        &light_icon_query,
        &mut icons_dirty,
    );
    if icons_dirty || icons.len() != prev_icon_count {
        bump_revision(&mut state.icon_revision);
    }

    let Some(entity) = selection.0 else {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        state.drag = None;
        state.hover_axis = GizmoAxis::None;
        state.active_axis = GizmoAxis::None;
        render_gizmo.0 = GizmoData {
            icons,
            icons_revision: state.icon_revision,
            outline_revision: state.outline_revision,
            style: settings.to_style(),
            ..GizmoData::default()
        };
        return;
    };

    let mut target_transform = match transforms.get_mut(entity) {
        Ok(transform) => transform.0,
        Err(_) => {
            if state.drag.is_some() && allow_undo {
                request_end_undo_group(&mut undo_state);
            }
            state.drag = None;
            state.hover_axis = GizmoAxis::None;
            state.active_axis = GizmoAxis::None;
            render_gizmo.0 = GizmoData {
                icons,
                icons_revision: state.icon_revision,
                outline_revision: state.outline_revision,
                style: settings.to_style(),
                ..GizmoData::default()
            };
            return;
        }
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

    let origin = target_transform.position;
    let rotation = target_transform.rotation;
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
                target_transform,
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
                apply_drag(&mut target_transform, &drag, ray_origin, ray_dir, &settings);
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
        .distance(target_transform.position)
        .max(0.001);
    let output_size = (output_distance * settings.size_scale).clamp(size_min, size_max);

    let (bounds_enabled, selection_min, selection_max) = if settings.show_bounds_outline {
        match selection_bounds(&asset_server, &mesh_query, entity) {
            Some(bounds) => (true, bounds.min, bounds.max),
            None => (false, Vec3::ZERO, Vec3::ZERO),
        }
    } else {
        (false, Vec3::ZERO, Vec3::ZERO)
    };
    let outline_lines = collect_mesh_outline_lines(
        &asset_server,
        &mesh_query,
        &settings,
        &mut outline_cache,
        entity,
    );
    let outline_equal = std::sync::Arc::ptr_eq(&prev_outline_lines, &outline_lines)
        || (prev_outline_lines.is_empty() && outline_lines.is_empty());
    if !outline_equal {
        bump_revision(&mut state.outline_revision);
    }
    let selection_enabled = settings.show_bounds_outline && bounds_enabled;

    if let Ok(mut transform) = transforms.get_mut(entity) {
        transform.0 = target_transform;
    }

    render_gizmo.0 = GizmoData {
        mode: state.mode,
        position: target_transform.position,
        rotation: target_transform.rotation,
        scale: target_transform.scale,
        size: output_size,
        hover_axis: state.hover_axis,
        active_axis: state.active_axis,
        selection_enabled,
        selection_min,
        selection_max,
        outline_lines,
        outline_revision: state.outline_revision,
        icons,
        icons_revision: state.icon_revision,
        style: settings.to_style(),
    };
}

pub fn selection_system(
    mut selection_state: ResMut<EditorSelectionState>,
    mut selection: ResMut<InspectorSelectedEntityResource>,
    gizmo_state: Res<EditorGizmoState>,
    settings: Res<EditorGizmoSettings>,
    viewport_state: Res<EditorViewportState>,
    scene_state: Res<EditorSceneState>,
    input: Res<BevyInputManager>,
    asset_server: Res<BevyAssetServer>,
    mesh_query: Query<(Entity, &BevyMeshRenderer, &BevyTransform), With<EditorEntity>>,
    transform_query: Query<&BevyTransform>,
    camera_icon_query: Query<(Entity, &BevyCamera, Option<&EditorPlayCamera>), With<EditorEntity>>,
    light_icon_query: Query<(Entity, &BevyLight), With<EditorEntity>>,
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

    if let Some((entity, distance)) = pick_icon_entity(
        &viewport_state,
        &settings,
        &camera_transform.0,
        ray_origin,
        ray_dir,
        &transform_query,
        &camera_icon_query,
        &light_icon_query,
    ) {
        if best.map_or(true, |(_, best_distance)| distance < best_distance) {
            best = Some((entity, distance));
        }
    }

    selection.0 = best.map(|(entity, _)| entity);
}

fn bump_revision(value: &mut u64) {
    *value = value.wrapping_add(1);
    if *value == 0 {
        // reserve 0 to indicate "no revision" for fallback hashing
        *value = 1;
    }
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

fn collect_mesh_outline_lines(
    asset_server: &BevyAssetServer,
    mesh_query: &Query<&BevyMeshRenderer>,
    settings: &EditorGizmoSettings,
    outline_cache: &mut EditorMeshOutlineCache,
    entity: Entity,
) -> std::sync::Arc<[GizmoLine]> {
    if !settings.show_mesh_outline {
        return std::sync::Arc::from(Vec::new());
    }
    let Ok(renderer) = mesh_query.get(entity) else {
        return std::sync::Arc::from(Vec::new());
    };
    let max_lines = settings.outline_max_lines as usize;
    if max_lines == 0 {
        return std::sync::Arc::from(Vec::new());
    }

    let mesh_id = renderer.0.mesh_id;
    let needs_rebuild = outline_cache
        .entries
        .get(&mesh_id)
        .map_or(true, |entry| entry.max_lines != max_lines);
    if needs_rebuild {
        let mesh = {
            let asset_server = asset_server.0.lock();
            asset_server.get_mesh(mesh_id)
        };
        let mesh = match mesh {
            Some(mesh) => mesh,
            None => return std::sync::Arc::from(Vec::new()),
        };
        let payload = match select_outline_lod(mesh.as_ref(), max_lines) {
            Some(payload) => payload,
            None => return std::sync::Arc::from(Vec::new()),
        };
        let lines = build_outline_lines(&payload, max_lines);
        outline_cache.entries.insert(
            mesh_id,
            MeshOutlineEntry {
                max_lines,
                lines: std::sync::Arc::from(lines),
            },
        );
    }

    outline_cache
        .entries
        .get(&mesh_id)
        .map(|entry| entry.lines.clone())
        .unwrap_or_else(|| std::sync::Arc::from(Vec::new()))
}

fn select_outline_lod(
    mesh: &helmer::runtime::asset_server::Mesh,
    max_lines: usize,
) -> Option<MeshLodPayload> {
    let lods = mesh.lods.read();
    let mut selected = lods.first()?;
    for lod in lods.iter().rev() {
        if lod.indices.len() <= max_lines {
            selected = lod;
            break;
        }
    }
    Some(selected.clone())
}

fn build_outline_lines(payload: &MeshLodPayload, max_lines: usize) -> Vec<GizmoLine> {
    if max_lines == 0 {
        return Vec::new();
    }
    let indices = payload.indices.as_ref();
    let vertices = payload.vertices.as_ref();
    if indices.len() < 3 || vertices.is_empty() {
        return Vec::new();
    }

    let edge_total = indices.len();
    let step = (edge_total / max_lines).max(1);
    let mut lines = Vec::with_capacity(max_lines.min(edge_total));
    let mut edge_index = 0usize;

    for tri in indices.chunks_exact(3) {
        let idx0 = tri[0] as usize;
        let idx1 = tri[1] as usize;
        let idx2 = tri[2] as usize;
        if idx0 >= vertices.len() || idx1 >= vertices.len() || idx2 >= vertices.len() {
            edge_index = edge_index.saturating_add(3);
            continue;
        }
        let p0 = Vec3::from_array(vertices[idx0].position);
        let p1 = Vec3::from_array(vertices[idx1].position);
        let p2 = Vec3::from_array(vertices[idx2].position);
        let edges = [(p0, p1), (p1, p2), (p2, p0)];
        for (start, end) in edges {
            if edge_index % step == 0 {
                lines.push(GizmoLine { start, end });
                if lines.len() >= max_lines {
                    return lines;
                }
            }
            edge_index = edge_index.saturating_add(1);
        }
    }

    lines
}

fn collect_icon_gizmos(
    viewport_state: &EditorViewportState,
    settings: &EditorGizmoSettings,
    camera_transform: &Transform,
    transforms: &mut Query<&mut BevyTransform>,
    camera_query: &Query<
        (Entity, Ref<BevyCamera>, Option<Ref<EditorPlayCamera>>),
        With<EditorEntity>,
    >,
    light_query: &Query<(Entity, Ref<BevyLight>), With<EditorEntity>>,
    icons_dirty: &mut bool,
) -> Vec<GizmoIcon> {
    let mut icons = Vec::new();
    if !viewport_state.show_camera_gizmos
        && !viewport_state.show_directional_light_gizmos
        && !viewport_state.show_point_light_gizmos
        && !viewport_state.show_spot_light_gizmos
    {
        return icons;
    }

    let icon_scale = settings.icon_size_scale.max(0.0);
    if icon_scale <= 0.0 {
        return icons;
    }
    let camera_alpha = settings.camera_icon_alpha.clamp(0.0, 1.0);
    let light_alpha = settings.light_icon_alpha.clamp(0.0, 1.0);
    let (size_min, size_max) = settings.size_bounds();
    let mut icon_min = size_min * icon_scale;
    let mut icon_max = size_max * icon_scale;
    if icon_min > icon_max {
        std::mem::swap(&mut icon_min, &mut icon_max);
    }

    if viewport_state.show_camera_gizmos {
        for (entity, camera, play_camera) in camera_query.iter() {
            let Ok(transform) = transforms.get_mut(entity) else {
                continue;
            };
            let transform_changed = transform.is_changed();
            let play_camera_changed = play_camera
                .as_ref()
                .map_or(false, |play_camera| play_camera.is_changed());
            if camera.is_changed() || transform_changed || play_camera_changed {
                *icons_dirty = true;
            }
            let distance = camera_transform
                .position
                .distance(transform.0.position)
                .max(0.001);
            let size = (distance * settings.size_scale * icon_scale).clamp(icon_min, icon_max);

            let near = camera.0.near_plane.max(0.001);
            let far = camera.0.far_plane.max(near + 0.001);
            let mut near_ratio = near / far;
            if !near_ratio.is_finite() {
                near_ratio = 0.1;
            }
            near_ratio = near_ratio.clamp(0.02, 0.9);

            let color = if play_camera.is_some() {
                sanitize_icon_color(settings.active_camera_icon_color)
            } else {
                sanitize_icon_color(settings.camera_icon_color)
            };

            let icon = GizmoIcon {
                position: transform.0.position,
                rotation: transform.0.rotation,
                size,
                color,
                alpha: camera_alpha,
                kind: GizmoIconKind::Camera,
                params: [camera.0.fov_y_rad, camera.0.aspect_ratio, near_ratio, 0.0],
            };
            icons.push(icon);
        }
    }

    if viewport_state.show_directional_light_gizmos
        || viewport_state.show_point_light_gizmos
        || viewport_state.show_spot_light_gizmos
    {
        for (entity, light) in light_query.iter() {
            let Ok(transform) = transforms.get_mut(entity) else {
                continue;
            };
            if light.is_changed() || transform.is_changed() {
                *icons_dirty = true;
            }

            let (kind, params, allowed) = match light.0.light_type {
                LightType::Directional => (
                    GizmoIconKind::LightDirectional,
                    [0.0; 4],
                    viewport_state.show_directional_light_gizmos,
                ),
                LightType::Point => (
                    GizmoIconKind::LightPoint,
                    [0.0; 4],
                    viewport_state.show_point_light_gizmos,
                ),
                LightType::Spot { angle } => (
                    GizmoIconKind::LightSpot,
                    [angle, 0.0, 0.0, 0.0],
                    viewport_state.show_spot_light_gizmos,
                ),
            };
            if !allowed {
                continue;
            }

            let distance = camera_transform
                .position
                .distance(transform.0.position)
                .max(0.001);
            let size = (distance * settings.size_scale * icon_scale).clamp(icon_min, icon_max);
            let color = sanitize_light_color(light.0.color);

            let icon = GizmoIcon {
                position: transform.0.position,
                rotation: transform.0.rotation,
                size,
                color,
                alpha: light_alpha,
                kind,
                params,
            };
            icons.push(icon);
        }
    }

    icons
}

fn pick_icon_entity(
    viewport_state: &EditorViewportState,
    settings: &EditorGizmoSettings,
    camera_transform: &Transform,
    ray_origin: Vec3,
    ray_dir: Vec3,
    transforms: &Query<&BevyTransform>,
    camera_query: &Query<(Entity, &BevyCamera, Option<&EditorPlayCamera>), With<EditorEntity>>,
    light_query: &Query<(Entity, &BevyLight), With<EditorEntity>>,
) -> Option<(Entity, f32)> {
    if !viewport_state.show_camera_gizmos
        && !viewport_state.show_directional_light_gizmos
        && !viewport_state.show_point_light_gizmos
        && !viewport_state.show_spot_light_gizmos
    {
        return None;
    }

    let icon_scale = settings.icon_size_scale.max(0.0);
    if icon_scale <= 0.0 {
        return None;
    }
    let pick_scale = settings.icon_pick_radius_scale.max(0.0);
    let pick_min = settings.icon_pick_radius_min.max(0.0);
    if pick_scale <= 0.0 && pick_min <= 0.0 {
        return None;
    }

    let (size_min, size_max) = settings.size_bounds();
    let mut icon_min = size_min * icon_scale;
    let mut icon_max = size_max * icon_scale;
    if icon_min > icon_max {
        std::mem::swap(&mut icon_min, &mut icon_max);
    }

    let mut best: Option<(Entity, f32)> = None;

    if viewport_state.show_camera_gizmos {
        for (entity, _, _) in camera_query.iter() {
            let Ok(transform) = transforms.get(entity) else {
                continue;
            };
            let distance = camera_transform
                .position
                .distance(transform.0.position)
                .max(0.001);
            let size = (distance * settings.size_scale * icon_scale).clamp(icon_min, icon_max);
            let radius = (size * pick_scale).max(pick_min);
            if let Some(hit) =
                ray_sphere_intersection(ray_origin, ray_dir, transform.0.position, radius)
            {
                if best.map_or(true, |(_, best_distance)| hit < best_distance) {
                    best = Some((entity, hit));
                }
            }
        }
    }

    if viewport_state.show_directional_light_gizmos
        || viewport_state.show_point_light_gizmos
        || viewport_state.show_spot_light_gizmos
    {
        for (entity, light) in light_query.iter() {
            let allowed = match light.0.light_type {
                LightType::Directional => viewport_state.show_directional_light_gizmos,
                LightType::Point => viewport_state.show_point_light_gizmos,
                LightType::Spot { .. } => viewport_state.show_spot_light_gizmos,
            };
            if !allowed {
                continue;
            }
            let Ok(transform) = transforms.get(entity) else {
                continue;
            };
            let distance = camera_transform
                .position
                .distance(transform.0.position)
                .max(0.001);
            let size = (distance * settings.size_scale * icon_scale).clamp(icon_min, icon_max);
            let radius = (size * pick_scale).max(pick_min);
            if let Some(hit) =
                ray_sphere_intersection(ray_origin, ray_dir, transform.0.position, radius)
            {
                if best.map_or(true, |(_, best_distance)| hit < best_distance) {
                    best = Some((entity, hit));
                }
            }
        }
    }

    best
}

fn sanitize_icon_color(color: [f32; 3]) -> Vec3 {
    let mut color = Vec3::from_array(color);
    if !color.is_finite() {
        return Vec3::ONE;
    }
    color = color.clamp(Vec3::ZERO, Vec3::ONE);
    color
}

fn sanitize_light_color(color: Vec3) -> Vec3 {
    if !color.is_finite() {
        return Vec3::ONE;
    }
    let max = color.max_element();
    let mut normalized = if max > 1.0 { color / max } else { color };
    normalized = normalized.clamp(Vec3::ZERO, Vec3::ONE);
    normalized
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

fn ray_sphere_intersection(
    ray_origin: Vec3,
    ray_dir: Vec3,
    center: Vec3,
    radius: f32,
) -> Option<f32> {
    let radius = radius.max(0.0);
    if radius <= 0.0 {
        return None;
    }
    let oc = ray_origin - center;
    let half_b = oc.dot(ray_dir);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = half_b * half_b - c;
    if discriminant < 0.0 {
        return None;
    }
    let t = -half_b - discriminant.sqrt();
    if t >= 0.0 { Some(t) } else { None }
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
