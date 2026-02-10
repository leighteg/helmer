use bevy_ecs::prelude::{DetectChanges, Entity, Query, Ref, Res, ResMut, Resource, With};
use bevy_ecs::query::Or;
use bevy_ecs::system::{ParamSet, SystemParam};
use glam::{Mat4, Quat, Vec3};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use helmer::animation::compute_global_matrices;
use helmer::graphics::common::renderer::{
    Aabb, GizmoAxis, GizmoData, GizmoIcon, GizmoIconKind, GizmoLine, GizmoMode, GizmoStyle,
    MeshLodPayload,
};
use helmer::provided::components::{Camera, LightType, Spline, Transform};
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::render_system::RenderGizmoState;
use helmer_becs::systems::scene_system::SceneChild;
use helmer_becs::{
    BevyActiveCamera, BevyAssetServer, BevyCamera, BevyInputManager, BevyLight, BevyMeshRenderer,
    BevyPoseOverride, BevySkinnedMeshRenderer, BevySpline, BevyTransform,
};

use crate::editor::scene::{EditorEntity, EditorSceneState, WorldState};
use crate::editor::{
    EditorPlayCamera, EditorUndoState, EditorViewportRuntime, EditorViewportState, PoseEditorState,
    ViewportRectPixels, request_begin_undo_group, request_end_undo_group,
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
    key_drag_active: bool,
    suppress_selection: bool,
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
            key_drag_active: false,
            suppress_selection: false,
        }
    }
}

impl EditorGizmoState {
    pub fn is_drag_active(&self) -> bool {
        self.drag.is_some() || self.key_drag_active
    }

    pub fn is_key_drag_active(&self) -> bool {
        self.key_drag_active
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineDrawPlane {
    WorldXZ,
    View,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplinePivotMode {
    Point,
    SplineOrigin,
}

impl Default for SplineDrawPlane {
    fn default() -> Self {
        SplineDrawPlane::WorldXZ
    }
}

#[derive(Resource, Debug, Clone)]
pub struct EditorSplineState {
    pub enabled: bool,
    pub add_mode: bool,
    pub insert_mode: bool,
    pub use_gizmo: bool,
    pub pivot_mode: SplinePivotMode,
    pub draw_plane: SplineDrawPlane,
    pub samples: u32,
    pub handle_size_scale: f32,
    pub handle_size_min: f32,
    pub pick_radius_scale: f32,
    pub pick_radius_min: f32,
    pub active_point: Option<usize>,
    pub hover_point: Option<usize>,
    pub dragging: bool,
    pub key_dragging: bool,
    pub saved_gizmo_mode: Option<GizmoMode>,
    point_drag: Option<SplinePointDrag>,
    drag_plane_origin: Vec3,
    drag_plane_normal: Vec3,
}

impl Default for EditorSplineState {
    fn default() -> Self {
        Self {
            enabled: false,
            add_mode: false,
            insert_mode: false,
            use_gizmo: true,
            pivot_mode: SplinePivotMode::SplineOrigin,
            draw_plane: SplineDrawPlane::WorldXZ,
            samples: 64,
            handle_size_scale: 0.04,
            handle_size_min: 0.05,
            pick_radius_scale: 0.03,
            pick_radius_min: 0.04,
            active_point: None,
            hover_point: None,
            dragging: false,
            key_dragging: false,
            saved_gizmo_mode: None,
            point_drag: None,
            drag_plane_origin: Vec3::ZERO,
            drag_plane_normal: Vec3::Y,
        }
    }
}

#[derive(Debug, Clone)]
struct SplinePointDrag {
    index: usize,
    start_local: Vec3,
    start_world: Vec3,
    pivot_world: Vec3,
    start_rotation: Quat,
    mode: GizmoMode,
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

#[derive(SystemParam)]
pub struct GizmoSystemParams<'w, 's> {
    state: ResMut<'w, EditorGizmoState>,
    render_gizmo: ResMut<'w, RenderGizmoState>,
    settings: Res<'w, EditorGizmoSettings>,
    outline_cache: ResMut<'w, EditorMeshOutlineCache>,
    spline_state: ResMut<'w, EditorSplineState>,
    pose_state: ResMut<'w, PoseEditorState>,
    viewport_state: Res<'w, EditorViewportState>,
    viewport_runtime: Res<'w, EditorViewportRuntime>,
    selection: Res<'w, InspectorSelectedEntityResource>,
    scene_state: Res<'w, EditorSceneState>,
    undo_state: ResMut<'w, EditorUndoState>,
    input: Res<'w, BevyInputManager>,
    asset_server: Res<'w, BevyAssetServer>,
    mesh_query: Query<'w, 's, &'static BevyMeshRenderer>,
    skinned_query: Query<'w, 's, &'static BevySkinnedMeshRenderer>,
    pose_override_query: Query<'w, 's, &'static mut BevyPoseOverride>,
    spatial_queries: ParamSet<
        'w,
        's,
        (
            Query<'w, 's, &'static mut BevyTransform>,
            Query<'w, 's, &'static BevyTransform>,
            Query<'w, 's, &'static mut BevySpline>,
            Query<
                'w,
                's,
                (Entity, &'static BevySpline, Option<&'static BevyTransform>),
                With<EditorEntity>,
            >,
        ),
    >,
    active_camera_query: Query<'w, 's, (Entity, Ref<'static, BevyCamera>), With<BevyActiveCamera>>,
    camera_component_query: Query<'w, 's, (Entity, Ref<'static, BevyCamera>)>,
    camera_icon_query: Query<
        'w,
        's,
        (
            Entity,
            Ref<'static, BevyCamera>,
            Option<Ref<'static, EditorPlayCamera>>,
        ),
        Or<(With<EditorEntity>, With<SceneChild>)>,
    >,
    light_icon_query: Query<
        'w,
        's,
        (Entity, Ref<'static, BevyLight>),
        Or<(With<EditorEntity>, With<SceneChild>)>,
    >,
}

pub fn gizmo_system(params: GizmoSystemParams) {
    let GizmoSystemParams {
        mut state,
        mut render_gizmo,
        settings,
        mut outline_cache,
        mut spline_state,
        mut pose_state,
        viewport_state,
        viewport_runtime,
        selection,
        scene_state,
        mut undo_state,
        input,
        asset_server,
        mesh_query,
        skinned_query,
        mut pose_override_query,
        mut spatial_queries,
        active_camera_query,
        camera_component_query,
        camera_icon_query,
        light_icon_query,
    } = params;

    state.suppress_selection = false;

    let show_gizmos = scene_state.world_state == WorldState::Edit || viewport_state.gizmos_in_play;
    if !show_gizmos {
        clear_gizmo(&mut state, &mut render_gizmo, &settings);
        return;
    }
    let allow_undo = scene_state.world_state == WorldState::Edit;
    let prev_icon_count = render_gizmo.0.icons.len();
    let prev_outline_lines = render_gizmo.0.outline_lines.clone();

    let selected_entity = selection.0;
    let show_spline_paths = viewport_state.show_spline_paths;
    let show_spline_points = viewport_state.show_spline_points;
    let mut global_spline_lines = Vec::new();
    if show_spline_paths {
        let samples = spline_state.samples.max(2);
        let spline_query = spatial_queries.p3();
        for (entity, spline, spline_transform) in spline_query.iter() {
            if spline_state.enabled && selected_entity == Some(entity) {
                continue;
            }
            let transform = spline_transform.map(|t| t.0).unwrap_or_default();
            let lines = build_spline_path_lines(&spline.0, &transform, samples);
            if !lines.is_empty() {
                global_spline_lines.extend(lines);
            }
        }
    }

    let input_manager = input.0.read();
    let (viewport_rect, runtime_camera_entity) =
        pick_viewport_interaction_target(input_manager.cursor_position, &viewport_runtime);

    let Some((camera_entity, camera)) = runtime_camera_entity
        .and_then(|entity| camera_component_query.get(entity).ok())
        .or_else(|| active_camera_query.iter().next())
    else {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        clear_gizmo(&mut state, &mut render_gizmo, &settings);
        return;
    };

    let (camera_transform, icons, icons_dirty) = {
        let mut transforms = spatial_queries.p0();
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
        (camera_transform, icons, icons_dirty)
    };
    if icons_dirty || icons.len() != prev_icon_count {
        bump_revision(&mut state.icon_revision);
    }

    let entity = match selected_entity {
        Some(entity) => entity,
        None => {
            if state.drag.is_some() && allow_undo {
                request_end_undo_group(&mut undo_state);
            }
            state.drag = None;
            state.hover_axis = GizmoAxis::None;
            state.active_axis = GizmoAxis::None;

            let outline_lines = if lines_equal(&prev_outline_lines, &global_spline_lines) {
                prev_outline_lines.clone()
            } else {
                bump_revision(&mut state.outline_revision);
                std::sync::Arc::from(global_spline_lines)
            };

            render_gizmo.0 = GizmoData {
                icons,
                icons_revision: state.icon_revision,
                outline_lines,
                outline_revision: state.outline_revision,
                style: settings.to_style(),
                ..GizmoData::default()
            };
            return;
        }
    };

    let mut target_transform = {
        let mut transforms = spatial_queries.p0();
        match transforms.get_mut(entity) {
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
        }
    };

    let mut pose_edit_active = false;
    let mut pose_world_transforms: Vec<Transform> = Vec::new();
    let mut pose_parent_indices: Vec<Option<usize>> = Vec::new();
    let mut selected_joint_world: Option<Transform> = None;
    let mut selected_parent_world: Option<Transform> = None;
    if pose_state.edit_mode && pose_state.active_entity == Some(entity.to_bits()) {
        if let Ok(skinned) = skinned_query.get(entity) {
            let skeleton = &skinned.0.skin.skeleton;
            if let Ok(mut pose_override) = pose_override_query.get_mut(entity) {
                if pose_override.0.pose.locals.len() != skeleton.joint_count() {
                    pose_override.0.pose.reset_to_bind(skeleton);
                }
                if !pose_override.0.enabled {
                    pose_override.0.enabled = true;
                }
                let mut globals = vec![Mat4::IDENTITY; skeleton.joint_count()];
                compute_global_matrices(skeleton, &pose_override.0.pose.locals, &mut globals);
                let entity_matrix = target_transform.to_matrix();
                pose_world_transforms.reserve(globals.len());
                for global in globals.iter() {
                    pose_world_transforms.push(Transform::from_matrix(entity_matrix * *global));
                }
                pose_parent_indices = skeleton.joints.iter().map(|joint| joint.parent).collect();
                pose_edit_active = true;
                if let Some(index) = pose_state.selected_joint {
                    if index < pose_world_transforms.len() {
                        selected_joint_world = pose_world_transforms.get(index).copied();
                        if let Some(parent) =
                            pose_parent_indices.get(index).and_then(|parent| *parent)
                        {
                            selected_parent_world = pose_world_transforms.get(parent).copied();
                        } else {
                            selected_parent_world = Some(target_transform);
                        }
                    } else {
                        pose_state.selected_joint = None;
                    }
                }
            }
        } else {
            pose_state.edit_mode = false;
            pose_state.active_entity = None;
            pose_state.selected_joint = None;
            pose_state.hover_joint = None;
        }
    }

    let mut spline_query = spatial_queries.p2();
    let mut spline = spline_query.get_mut(entity).ok();
    let spline_selected = spline.is_some();
    let spline_edit_active = spline_state.enabled && spline_selected && !pose_edit_active;
    if !spline_selected {
        spline_state.point_drag = None;
    }

    let pointer_in_viewport = viewport_rect
        .map(|rect| rect.contains(input_manager.cursor_position))
        .unwrap_or(false);
    let wants_pointer = input_manager.egui_wants_pointer;
    let wants_key = input_manager.egui_wants_key;
    let freecam_looking = input_manager.is_mouse_button_active(MouseButton::Right);
    let left_down = input_manager.is_mouse_button_active(MouseButton::Left);
    let left_pressed = left_down && !state.last_mouse_down;
    let left_released = !left_down && state.last_mouse_down;
    state.last_mouse_down = left_down;

    let inv_view_proj = camera_inv_view_proj(&camera.0, &camera_transform, viewport_rect);
    let allow_ui_raycast = pointer_in_viewport
        || state.drag.is_some()
        || state.key_drag_active
        || spline_state.key_dragging;
    let ray = if wants_pointer && !allow_ui_raycast {
        None
    } else {
        screen_ray(
            input_manager.cursor_position,
            viewport_rect,
            input_manager.window_size,
            inv_view_proj,
            camera_transform.position,
        )
    };

    if spline_edit_active
        && !wants_key
        && state.drag.is_none()
        && !spline_state.dragging
        && !spline_state.key_dragging
        && spline_state.point_drag.is_none()
    {
        if input_manager.was_just_pressed(KeyCode::KeyE) {
            if let (Some(active_index), Some(spline_mut)) =
                (spline_state.active_point, spline.as_mut())
            {
                if allow_undo {
                    request_begin_undo_group(&mut undo_state, "Spline");
                }
                if let Some((new_index, plane_origin, plane_normal)) = extrude_spline_point(
                    &mut spline_mut.0,
                    active_index,
                    &target_transform,
                    &camera_transform,
                    spline_state.draw_plane,
                    ray,
                ) {
                    spline_state.active_point = Some(new_index);
                    spline_state.hover_point = Some(new_index);
                    spline_state.point_drag = None;
                    spline_state.drag_plane_origin = plane_origin;
                    spline_state.drag_plane_normal = plane_normal;
                    spline_state.key_dragging = true;
                } else if allow_undo {
                    request_end_undo_group(&mut undo_state);
                }
            }
        } else if input_manager.was_just_pressed(KeyCode::KeyI) {
            if let (Some(active_index), Some(spline_mut)) =
                (spline_state.active_point, spline.as_mut())
            {
                if let Some((new_index, plane_origin, plane_normal)) = insert_spline_midpoint(
                    &mut spline_mut.0,
                    active_index,
                    &target_transform,
                    &camera_transform,
                    spline_state.draw_plane,
                ) {
                    if allow_undo {
                        request_begin_undo_group(&mut undo_state, "Spline");
                    }
                    spline_state.active_point = Some(new_index);
                    spline_state.hover_point = Some(new_index);
                    spline_state.point_drag = None;
                    spline_state.drag_plane_origin = plane_origin;
                    spline_state.drag_plane_normal = plane_normal;
                    spline_state.key_dragging = true;
                }
            }
        }
    }

    if !pose_edit_active {
        pose_state.hover_joint = None;
        pose_state.dragging = false;
    } else if wants_pointer && !allow_ui_raycast {
        pose_state.hover_joint = None;
        if !left_down {
            pose_state.dragging = false;
        }
    } else if let Some((ray_origin, ray_dir)) = ray {
        let pick_radius = (camera_transform
            .position
            .distance(target_transform.position)
            * pose_state.pick_radius_scale)
            .max(pose_state.pick_radius_min)
            .max(1.0e-4);
        let mut best_joint = None;
        let mut best_dist = f32::MAX;
        for (index, joint_transform) in pose_world_transforms.iter().enumerate() {
            let dist = ray_point_distance(ray_origin, ray_dir, joint_transform.position);
            if dist <= pick_radius && dist < best_dist {
                best_dist = dist;
                best_joint = Some(index);
            }
        }
        pose_state.hover_joint = best_joint;
        if left_pressed {
            if let Some(index) = best_joint {
                pose_state.selected_joint = Some(index);
            }
        }
    }

    let spline_point_world = if spline_selected {
        match (spline_state.active_point, spline.as_ref()) {
            (Some(index), Some(spline_ref)) if index < spline_ref.0.points.len() => Some(
                target_transform
                    .to_matrix()
                    .transform_point3(spline_ref.0.points[index]),
            ),
            _ => None,
        }
    } else {
        None
    };

    let spline_points_active = show_spline_points && spline_edit_active;
    let point_gizmo_active = spline_points_active
        && spline_state.use_gizmo
        && spline_point_world.is_some()
        && state.mode != GizmoMode::None;
    let bone_gizmo_active = pose_edit_active
        && pose_state.use_gizmo
        && selected_joint_world.is_some()
        && state.mode != GizmoMode::None;
    let lock_entity_gizmo = spline_points_active || pose_edit_active;
    if lock_entity_gizmo && !(point_gizmo_active || bone_gizmo_active) {
        state.drag = None;
        state.hover_axis = GizmoAxis::None;
        state.active_axis = GizmoAxis::None;
    }
    let mut point_world = target_transform.position;
    let mut pivot_world = target_transform.position;
    let mut gizmo_origin = target_transform.position;
    let mut rotation = target_transform.rotation;
    let mut edit_gizmo_active = false;
    if bone_gizmo_active {
        if let Some(joint_world) = selected_joint_world {
            point_world = joint_world.position;
            pivot_world = joint_world.position;
            gizmo_origin = joint_world.position;
            rotation = joint_world.rotation;
            edit_gizmo_active = true;
        }
    } else if point_gizmo_active {
        point_world = spline_point_world.unwrap_or(target_transform.position);
        pivot_world = match spline_state.pivot_mode {
            SplinePivotMode::Point => point_world,
            SplinePivotMode::SplineOrigin => target_transform.position,
        };
        gizmo_origin = if !matches!(state.mode, GizmoMode::Translate) {
            pivot_world
        } else {
            point_world
        };
        edit_gizmo_active = true;
    }
    let (size_min, size_max) = settings.size_bounds();
    let distance = camera_transform.position.distance(gizmo_origin).max(0.001);
    let gizmo_size = (distance * settings.size_scale).clamp(size_min, size_max);
    let mut view_dir = camera_transform.position - gizmo_origin;
    if view_dir.length_squared() < 1.0e-6 {
        view_dir = camera_transform.forward();
    }
    let view_dir = view_dir.normalize_or_zero();
    let drag_start_transform = if bone_gizmo_active {
        selected_joint_world.unwrap_or_else(|| Transform::new(gizmo_origin, rotation, Vec3::ONE))
    } else if point_gizmo_active {
        Transform::new(gizmo_origin, rotation, Vec3::ONE)
    } else {
        target_transform
    };

    if wants_pointer && !allow_ui_raycast {
        let had_drag = state.drag.is_some();
        if left_released || had_drag {
            state.drag = None;
            state.active_axis = GizmoAxis::None;
            state.key_drag_active = false;
            spline_state.point_drag = None;
        }
        if had_drag && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        state.hover_axis = GizmoAxis::None;
    } else if let Some((ray_origin, ray_dir)) = ray {
        let control_active = input_manager.is_key_active(KeyCode::ControlLeft)
            || input_manager.is_key_active(KeyCode::ControlRight);
        let shift_active = input_manager.is_key_active(KeyCode::ShiftLeft)
            || input_manager.is_key_active(KeyCode::ShiftRight);
        let key_mode = if !wants_key && !control_active && !freecam_looking {
            if input_manager.was_just_pressed(KeyCode::KeyG) {
                Some(GizmoMode::Translate)
            } else if input_manager.was_just_pressed(KeyCode::KeyR) {
                Some(GizmoMode::Rotate)
            } else if input_manager.was_just_pressed(KeyCode::KeyS) {
                Some(GizmoMode::Scale)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(mode) = key_mode {
            state.mode = mode;
            if shift_active
                && state.drag.is_none()
                && mode != GizmoMode::None
                && (!lock_entity_gizmo || point_gizmo_active || bone_gizmo_active)
            {
                if let Some(drag_state) = begin_drag(
                    mode,
                    GizmoAxis::Center,
                    ray_origin,
                    ray_dir,
                    gizmo_origin,
                    view_dir,
                    drag_start_transform,
                    &settings,
                ) {
                    state.active_axis = drag_state.axis;
                    state.drag = Some(drag_state);
                    state.key_drag_active = true;
                    if point_gizmo_active {
                        if let Some(index) = spline_state.active_point {
                            spline_state.point_drag = Some(SplinePointDrag {
                                index,
                                start_local: spline
                                    .as_ref()
                                    .and_then(|spline| spline.0.points.get(index))
                                    .copied()
                                    .unwrap_or(Vec3::ZERO),
                                start_world: point_world,
                                pivot_world,
                                start_rotation: rotation,
                                mode,
                            });
                        }
                    }
                    if bone_gizmo_active {
                        pose_state.dragging = true;
                    }
                    if allow_undo {
                        request_begin_undo_group(&mut undo_state, undo_label_for_mode(mode));
                    }
                }
            }
        }

        if state.drag.is_none()
            && state.mode != GizmoMode::None
            && (!lock_entity_gizmo || point_gizmo_active || bone_gizmo_active)
        {
            state.hover_axis = pick_gizmo(
                state.mode,
                ray_origin,
                ray_dir,
                gizmo_origin,
                rotation,
                gizmo_size,
                &settings,
            );
        }

        if left_pressed
            && state.drag.is_none()
            && state.mode != GizmoMode::None
            && state.hover_axis != GizmoAxis::None
            && (!lock_entity_gizmo || point_gizmo_active || bone_gizmo_active)
        {
            if let Some(drag_state) = begin_drag(
                state.mode,
                state.hover_axis,
                ray_origin,
                ray_dir,
                gizmo_origin,
                view_dir,
                drag_start_transform,
                &settings,
            ) {
                state.active_axis = drag_state.axis;
                state.drag = Some(drag_state);
                state.key_drag_active = false;
                if point_gizmo_active {
                    if let Some(index) = spline_state.active_point {
                        spline_state.point_drag = Some(SplinePointDrag {
                            index,
                            start_local: spline
                                .as_ref()
                                .and_then(|spline| spline.0.points.get(index))
                                .copied()
                                .unwrap_or(Vec3::ZERO),
                            start_world: point_world,
                            pivot_world,
                            start_rotation: rotation,
                            mode: state.mode,
                        });
                    }
                }
                if bone_gizmo_active {
                    pose_state.dragging = true;
                }
                if allow_undo {
                    request_begin_undo_group(&mut undo_state, undo_label_for_mode(state.mode));
                }
            }
        }

        if let Some(drag) = state.drag.take() {
            let mut did_apply = false;
            if state.key_drag_active || left_down {
                if let Some(point_drag) = spline_state.point_drag.as_ref() {
                    let mut drag_transform =
                        Transform::new(drag.start_transform.position, rotation, Vec3::ONE);
                    apply_drag(&mut drag_transform, &drag, ray_origin, ray_dir, &settings);
                    let new_world = match point_drag.mode {
                        GizmoMode::Translate => drag_transform.position,
                        GizmoMode::Rotate => {
                            let delta =
                                drag_transform.rotation * point_drag.start_rotation.inverse();
                            point_drag.pivot_world
                                + delta * (point_drag.start_world - point_drag.pivot_world)
                        }
                        GizmoMode::Scale => {
                            let scale = drag_transform.scale;
                            let local_offset = point_drag.start_rotation.inverse()
                                * (point_drag.start_world - point_drag.pivot_world);
                            let scaled_local = Vec3::new(
                                local_offset.x * scale.x,
                                local_offset.y * scale.y,
                                local_offset.z * scale.z,
                            );
                            point_drag.pivot_world + point_drag.start_rotation * scaled_local
                        }
                        GizmoMode::None => point_drag.start_world,
                    };
                    if let Some(spline_mut) = spline.as_mut() {
                        let inv_spline_matrix = target_transform.to_matrix().inverse();
                        let local_hit = inv_spline_matrix.transform_point3(new_world);
                        if point_drag.index < spline_mut.0.points.len() {
                            spline_mut.0.points[point_drag.index] = local_hit;
                        }
                    }
                    did_apply = true;
                } else if bone_gizmo_active {
                    if let (Some(index), Some(parent_world)) =
                        (pose_state.selected_joint, selected_parent_world)
                    {
                        let mut drag_transform = drag.start_transform;
                        apply_drag(&mut drag_transform, &drag, ray_origin, ray_dir, &settings);
                        let parent_matrix = parent_world.to_matrix();
                        let local_matrix = parent_matrix.inverse() * drag_transform.to_matrix();
                        let local = Transform::from_matrix(local_matrix);
                        if let Ok(mut pose_override) = pose_override_query.get_mut(entity) {
                            if index < pose_override.0.pose.locals.len() {
                                pose_override.0.pose.locals[index] = local;
                            }
                        }
                        did_apply = true;
                    }
                } else if !lock_entity_gizmo {
                    apply_drag(&mut target_transform, &drag, ray_origin, ray_dir, &settings);
                    did_apply = true;
                }
            }

            if state.key_drag_active {
                if did_apply {
                    state.active_axis = drag.axis;
                    state.drag = Some(drag);
                }
                if left_pressed {
                    state.suppress_selection = true;
                    state.active_axis = GizmoAxis::None;
                    state.drag = None;
                    state.key_drag_active = false;
                    spline_state.point_drag = None;
                    pose_state.dragging = false;
                    if allow_undo {
                        request_end_undo_group(&mut undo_state);
                    }
                }
            } else if left_down {
                state.active_axis = drag.axis;
                state.drag = Some(drag);
            } else {
                state.active_axis = GizmoAxis::None;
                state.drag = None;
                state.key_drag_active = false;
                spline_state.point_drag = None;
                pose_state.dragging = false;
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
        state.key_drag_active = false;
        spline_state.point_drag = None;
        pose_state.dragging = false;
    }

    if spline_selected {
        let allow_plane_drag = spline_points_active
            && spline_edit_active
            && (!spline_state.use_gizmo
                || matches!(state.mode, GizmoMode::None)
                || spline_state.key_dragging);
        if !spline_points_active {
            if (spline_state.dragging || spline_state.key_dragging) && allow_undo {
                request_end_undo_group(&mut undo_state);
            }
            spline_state.dragging = false;
            spline_state.key_dragging = false;
            spline_state.hover_point = None;
        } else if wants_pointer && !allow_ui_raycast {
            if (spline_state.dragging || spline_state.key_dragging) && allow_undo {
                request_end_undo_group(&mut undo_state);
            }
            spline_state.dragging = false;
            spline_state.key_dragging = false;
            spline_state.hover_point = None;
        } else if let Some((ray_origin, ray_dir)) = ray {
            let plane_normal = spline_plane_normal(spline_state.draw_plane, &camera_transform);
            let pick_radius = (distance * spline_state.pick_radius_scale)
                .max(spline_state.pick_radius_min)
                .max(1.0e-4);
            let spline_matrix = target_transform.to_matrix();
            let inv_spline_matrix = spline_matrix.inverse();

            let hover_point = if let Some(spline_ref) = spline.as_ref() {
                let mut closest = None;
                let mut closest_dist = f32::MAX;
                for (index, point) in spline_ref.0.points.iter().enumerate() {
                    let world_point = spline_matrix.transform_point3(*point);
                    let dist = ray_point_distance(ray_origin, ray_dir, world_point);
                    if dist <= pick_radius && dist < closest_dist {
                        closest = Some(index);
                        closest_dist = dist;
                    }
                }
                closest
            } else {
                None
            };

            spline_state.hover_point = hover_point;

            if left_pressed && !spline_state.dragging {
                if let Some(index) = hover_point {
                    spline_state.active_point = Some(index);
                    if allow_plane_drag {
                        spline_state.dragging = true;
                        spline_state.drag_plane_origin = spline_matrix.transform_point3(
                            spline
                                .as_ref()
                                .map(|spline| spline.0.points[index])
                                .unwrap_or(Vec3::ZERO),
                        );
                        spline_state.drag_plane_normal = plane_normal;
                        if allow_undo {
                            request_begin_undo_group(&mut undo_state, "Spline");
                        }
                    }
                } else if spline_edit_active && spline_state.add_mode {
                    if let Some(hit) = intersect_ray_plane(
                        ray_origin,
                        ray_dir,
                        target_transform.position,
                        plane_normal,
                    ) {
                        let local_hit = inv_spline_matrix.transform_point3(hit);
                        if let Some(spline_mut) = spline.as_mut() {
                            let insert_index = if spline_state.insert_mode {
                                spline_state.active_point.map(|idx| idx + 1)
                            } else {
                                None
                            };
                            if let Some(index) = insert_index {
                                let index = index.min(spline_mut.0.points.len());
                                spline_mut.0.points.insert(index, local_hit);
                                spline_state.active_point = Some(index);
                            } else {
                                spline_mut.0.points.push(local_hit);
                                spline_state.active_point = Some(spline_mut.0.points.len() - 1);
                            }
                        }
                        if allow_undo {
                            request_begin_undo_group(&mut undo_state, "Spline");
                            request_end_undo_group(&mut undo_state);
                        }
                    }
                } else if !spline_edit_active {
                    spline_state.active_point = None;
                }
            }

            if (spline_state.dragging || spline_state.key_dragging) && allow_plane_drag {
                if spline_state.key_dragging {
                    if let Some(hit) = intersect_ray_plane(
                        ray_origin,
                        ray_dir,
                        spline_state.drag_plane_origin,
                        spline_state.drag_plane_normal,
                    ) {
                        let local_hit = inv_spline_matrix.transform_point3(hit);
                        if let Some(spline_mut) = spline.as_mut() {
                            if let Some(index) = spline_state.active_point {
                                if index < spline_mut.0.points.len() {
                                    spline_mut.0.points[index] = local_hit;
                                }
                            }
                        }
                    }
                    if left_pressed {
                        spline_state.key_dragging = false;
                        if allow_undo {
                            request_end_undo_group(&mut undo_state);
                        }
                    }
                } else if left_down {
                    if let Some(hit) = intersect_ray_plane(
                        ray_origin,
                        ray_dir,
                        spline_state.drag_plane_origin,
                        spline_state.drag_plane_normal,
                    ) {
                        let local_hit = inv_spline_matrix.transform_point3(hit);
                        if let Some(spline_mut) = spline.as_mut() {
                            if let Some(index) = spline_state.active_point {
                                if index < spline_mut.0.points.len() {
                                    spline_mut.0.points[index] = local_hit;
                                }
                            }
                        }
                    }
                } else {
                    spline_state.dragging = false;
                    if allow_undo {
                        request_end_undo_group(&mut undo_state);
                    }
                }
            } else if !allow_plane_drag {
                spline_state.dragging = false;
                spline_state.key_dragging = false;
            }
        }
    } else {
        spline_state.dragging = false;
        spline_state.key_dragging = false;
        spline_state.hover_point = None;
        spline_state.active_point = None;
    }

    if state.mode == GizmoMode::None {
        if state.drag.is_some() && allow_undo {
            request_end_undo_group(&mut undo_state);
        }
        state.drag = None;
        state.hover_axis = GizmoAxis::None;
        state.active_axis = GizmoAxis::None;
        spline_state.point_drag = None;
    }

    let output_distance = camera_transform.position.distance(gizmo_origin).max(0.001);
    let output_size = (output_distance * settings.size_scale).clamp(size_min, size_max);

    let (bounds_enabled, selection_min, selection_max) = if settings.show_bounds_outline {
        match selection_bounds(&asset_server, &mesh_query, entity) {
            Some(bounds) => (true, bounds.min, bounds.max),
            None => (false, Vec3::ZERO, Vec3::ZERO),
        }
    } else {
        (false, Vec3::ZERO, Vec3::ZERO)
    };
    let mut combined_lines = global_spline_lines;
    let mesh_outline_lines = collect_mesh_outline_lines(
        &asset_server,
        &mesh_query,
        &settings,
        &mut outline_cache,
        entity,
    );
    if !mesh_outline_lines.is_empty() {
        let matrix = target_transform.to_matrix();
        combined_lines.extend(mesh_outline_lines.iter().map(|line| GizmoLine {
            start: matrix.transform_point3(line.start),
            end: matrix.transform_point3(line.end),
        }));
    }
    if show_spline_paths && spline_selected && spline_state.enabled {
        if let Some(spline_ref) = spline.as_ref() {
            let samples = spline_state.samples.max(2);
            let path_lines = build_spline_path_lines(&spline_ref.0, &target_transform, samples);
            if !path_lines.is_empty() {
                combined_lines.extend(path_lines);
            }
        }
    }
    if spline_points_active {
        if let Some(spline_ref) = spline.as_ref() {
            let handle_size =
                (distance * spline_state.handle_size_scale).max(spline_state.handle_size_min);
            let handle_lines =
                build_spline_handle_lines(&spline_ref.0, &target_transform, handle_size);
            if !handle_lines.is_empty() {
                combined_lines.extend(handle_lines);
            }
        }
    }
    if pose_edit_active {
        if pose_state.show_bones {
            for (index, joint_transform) in pose_world_transforms.iter().enumerate() {
                let Some(parent) = pose_parent_indices.get(index).and_then(|parent| *parent) else {
                    continue;
                };
                let Some(parent_transform) = pose_world_transforms.get(parent) else {
                    continue;
                };
                combined_lines.push(GizmoLine {
                    start: parent_transform.position,
                    end: joint_transform.position,
                });
            }
        }
        if pose_state.show_joint_handles {
            for (index, joint_transform) in pose_world_transforms.iter().enumerate() {
                let distance = camera_transform
                    .position
                    .distance(joint_transform.position)
                    .max(0.001);
                let mut handle_size =
                    (distance * pose_state.handle_size_scale).max(pose_state.handle_size_min);
                if pose_state.selected_joint == Some(index) {
                    handle_size *= 1.35;
                } else if pose_state.hover_joint == Some(index) {
                    handle_size *= 1.15;
                }
                let basis = joint_transform.rotation;
                let x = basis * Vec3::X * handle_size;
                let y = basis * Vec3::Y * handle_size;
                let z = basis * Vec3::Z * handle_size;
                let origin = joint_transform.position;
                combined_lines.push(GizmoLine {
                    start: origin - x,
                    end: origin + x,
                });
                combined_lines.push(GizmoLine {
                    start: origin - y,
                    end: origin + y,
                });
                combined_lines.push(GizmoLine {
                    start: origin - z,
                    end: origin + z,
                });
            }
        }
    }
    if !combined_lines.is_empty() {
        let gizmo_scale = if edit_gizmo_active {
            Vec3::ONE
        } else {
            target_transform.scale
        };
        let gizmo_matrix =
            Mat4::from_scale_rotation_translation(gizmo_scale, rotation, gizmo_origin);
        let inv_gizmo = gizmo_matrix.inverse();
        for line in combined_lines.iter_mut() {
            line.start = inv_gizmo.transform_point3(line.start);
            line.end = inv_gizmo.transform_point3(line.end);
        }
    }
    let outline_lines = if lines_equal(&prev_outline_lines, &combined_lines) {
        prev_outline_lines.clone()
    } else {
        bump_revision(&mut state.outline_revision);
        std::sync::Arc::from(combined_lines)
    };
    let selection_enabled = settings.show_bounds_outline && bounds_enabled;

    drop(spline);
    drop(spline_query);

    if !edit_gizmo_active {
        let mut transforms = spatial_queries.p0();
        if let Ok(mut transform) = transforms.get_mut(entity) {
            transform.0 = target_transform;
        }
    }

    render_gizmo.0 = GizmoData {
        mode: state.mode,
        position: gizmo_origin,
        rotation,
        scale: if edit_gizmo_active {
            Vec3::ONE
        } else {
            target_transform.scale
        },
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

#[derive(SystemParam)]
pub struct SelectionSystemParams<'w, 's> {
    selection_state: ResMut<'w, EditorSelectionState>,
    selection: ResMut<'w, InspectorSelectedEntityResource>,
    gizmo_state: Res<'w, EditorGizmoState>,
    spline_state: Res<'w, EditorSplineState>,
    pose_state: Res<'w, PoseEditorState>,
    settings: Res<'w, EditorGizmoSettings>,
    viewport_state: Res<'w, EditorViewportState>,
    viewport_runtime: Res<'w, EditorViewportRuntime>,
    scene_state: Res<'w, EditorSceneState>,
    input: Res<'w, BevyInputManager>,
    asset_server: Res<'w, BevyAssetServer>,
    mesh_query: Query<
        'w,
        's,
        (Entity, &'static BevyMeshRenderer, &'static BevyTransform),
        Or<(With<EditorEntity>, With<SceneChild>)>,
    >,
    skinned_query: Query<
        'w,
        's,
        (
            Entity,
            &'static BevySkinnedMeshRenderer,
            &'static BevyTransform,
        ),
        Or<(With<EditorEntity>, With<SceneChild>)>,
    >,
    spline_query: Query<
        'w,
        's,
        (Entity, &'static BevySpline, Option<&'static BevyTransform>),
        With<EditorEntity>,
    >,
    transform_query: Query<'w, 's, &'static BevyTransform>,
    camera_icon_query: Query<
        'w,
        's,
        (
            Entity,
            &'static BevyCamera,
            Option<&'static EditorPlayCamera>,
        ),
        Or<(With<EditorEntity>, With<SceneChild>)>,
    >,
    light_icon_query:
        Query<'w, 's, (Entity, &'static BevyLight), Or<(With<EditorEntity>, With<SceneChild>)>>,
    active_camera_query: Query<
        'w,
        's,
        (Entity, &'static BevyCamera, &'static BevyTransform),
        With<BevyActiveCamera>,
    >,
    camera_query: Query<'w, 's, (Entity, &'static BevyCamera, &'static BevyTransform)>,
}

pub fn selection_system(params: SelectionSystemParams) {
    let SelectionSystemParams {
        mut selection_state,
        mut selection,
        gizmo_state,
        spline_state,
        pose_state,
        settings,
        viewport_state,
        viewport_runtime,
        scene_state,
        input,
        asset_server,
        mesh_query,
        skinned_query,
        spline_query,
        transform_query,
        camera_icon_query,
        light_icon_query,
        active_camera_query,
        camera_query,
    } = params;
    if scene_state.world_state != WorldState::Edit && !viewport_state.gizmos_in_play {
        return;
    }

    if spline_state.dragging || spline_state.hover_point.is_some() {
        return;
    }
    if spline_state.enabled && spline_state.add_mode {
        return;
    }
    if pose_state.edit_mode
        && selection
            .0
            .map(|entity| pose_state.active_entity == Some(entity.to_bits()))
            .unwrap_or(false)
        && (pose_state.dragging || pose_state.hover_joint.is_some())
    {
        return;
    }

    let input_manager = input.0.read();
    let (viewport_rect, runtime_camera_entity) =
        pick_viewport_interaction_target(input_manager.cursor_position, &viewport_runtime);
    let pointer_in_viewport = viewport_rect
        .map(|rect| rect.contains(input_manager.cursor_position))
        .unwrap_or(false);
    let left_down = input_manager.is_mouse_button_active(MouseButton::Left);
    let left_pressed = left_down && !selection_state.last_mouse_down;
    selection_state.last_mouse_down = left_down;

    if !pointer_in_viewport || !left_pressed {
        return;
    }

    if gizmo_state.suppress_selection
        || gizmo_state.key_drag_active
        || gizmo_state.drag.is_some()
        || gizmo_state.active_axis != GizmoAxis::None
    {
        return;
    }

    let Some((_, camera, camera_transform)) = runtime_camera_entity
        .and_then(|entity| camera_query.get(entity).ok())
        .or_else(|| active_camera_query.iter().next())
    else {
        return;
    };

    let inv_view_proj = camera_inv_view_proj(&camera.0, &camera_transform.0, viewport_rect);
    let Some((ray_origin, ray_dir)) = screen_ray(
        input_manager.cursor_position,
        viewport_rect,
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
    for (entity, renderer, transform) in skinned_query.iter() {
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

    if viewport_state.show_spline_paths {
        let samples = spline_state.samples.max(2);
        let mut best_spline: Option<(Entity, f32)> = None;
        for (entity, spline, spline_transform) in spline_query.iter() {
            let transform = spline_transform.map(|t| t.0).unwrap_or_default();
            let lines = build_spline_path_lines(&spline.0, &transform, samples);
            for line in lines {
                let mid = (line.start + line.end) * 0.5;
                let distance_scale = camera_transform.0.position.distance(mid).max(0.001);
                let pick_radius = (distance_scale * spline_state.pick_radius_scale)
                    .max(spline_state.pick_radius_min)
                    .max(1.0e-4);
                let Some((dist, ray_t)) =
                    ray_segment_distance(ray_origin, ray_dir, line.start, line.end)
                else {
                    continue;
                };
                if dist <= pick_radius && best_spline.map_or(true, |(_, best_t)| ray_t < best_t) {
                    best_spline = Some((entity, ray_t));
                }
            }
        }
        if let Some((entity, ray_t)) = best_spline {
            if best.map_or(true, |(_, best_distance)| ray_t < best_distance) {
                best = Some((entity, ray_t));
            }
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

fn lines_equal(prev: &std::sync::Arc<[GizmoLine]>, current: &[GizmoLine]) -> bool {
    if prev.len() != current.len() {
        return false;
    }
    prev.iter().zip(current.iter()).all(|(a, b)| a == b)
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
        Or<(With<EditorEntity>, With<SceneChild>)>,
    >,
    light_query: &Query<(Entity, Ref<BevyLight>), Or<(With<EditorEntity>, With<SceneChild>)>>,
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
    camera_query: &Query<
        (Entity, &BevyCamera, Option<&EditorPlayCamera>),
        Or<(With<EditorEntity>, With<SceneChild>)>,
    >,
    light_query: &Query<(Entity, &BevyLight), Or<(With<EditorEntity>, With<SceneChild>)>>,
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

fn pick_viewport_interaction_target(
    cursor: glam::DVec2,
    viewport_runtime: &EditorViewportRuntime,
) -> (Option<ViewportRectPixels>, Option<Entity>) {
    let hovered_pane = viewport_runtime
        .pane_requests
        .iter()
        .find(|pane| pane.pointer_over)
        .or_else(|| {
            viewport_runtime
                .pane_requests
                .iter()
                .find(|pane| pane.viewport_rect.contains(cursor))
        });
    let active_pane = viewport_runtime.active_pane_id.and_then(|pane_id| {
        viewport_runtime
            .pane_requests
            .iter()
            .find(|pane| pane.pane_id == pane_id)
    });

    let viewport_rect = hovered_pane
        .map(|pane| pane.viewport_rect)
        .or_else(|| active_pane.map(|pane| pane.viewport_rect))
        .or(viewport_runtime.main_rect_pixels)
        .or_else(|| {
            viewport_runtime
                .pane_requests
                .first()
                .map(|pane| pane.viewport_rect)
        });
    let camera_entity = hovered_pane
        .map(|pane| pane.camera_entity)
        .or_else(|| active_pane.map(|pane| pane.camera_entity))
        .or(viewport_runtime.active_camera_entity)
        .or_else(|| {
            viewport_runtime
                .pane_requests
                .first()
                .map(|pane| pane.camera_entity)
        });

    (viewport_rect, camera_entity)
}

fn camera_inv_view_proj(
    camera: &Camera,
    transform: &Transform,
    viewport_rect: Option<ViewportRectPixels>,
) -> Mat4 {
    let forward = transform.forward().normalize_or_zero();
    let up = transform.up().normalize_or_zero();
    let view = Mat4::look_at_rh(transform.position, transform.position + forward, up);
    let aspect = viewport_rect
        .map(|rect| rect.aspect_ratio())
        .filter(|aspect| aspect.is_finite() && *aspect > 0.0)
        .unwrap_or(camera.aspect_ratio.max(0.001));
    let projection = Mat4::perspective_infinite_reverse_rh(
        camera.fov_y_rad.max(0.001),
        aspect,
        camera.near_plane.max(0.001),
    );
    let view_proj = projection * view;
    view_proj.inverse()
}

fn screen_ray(
    cursor: glam::DVec2,
    viewport_rect: Option<ViewportRectPixels>,
    window_size: glam::UVec2,
    inv_view_proj: Mat4,
    camera_position: Vec3,
) -> Option<(Vec3, Vec3)> {
    let (cursor_x, cursor_y, width, height) = if let Some(rect) = viewport_rect {
        if !rect.contains(cursor) {
            return None;
        }
        (
            cursor.x as f32 - rect.min_x,
            cursor.y as f32 - rect.min_y,
            rect.width(),
            rect.height(),
        )
    } else {
        if window_size.x == 0 || window_size.y == 0 {
            return None;
        }
        (
            cursor.x as f32,
            cursor.y as f32,
            window_size.x as f32,
            window_size.y as f32,
        )
    };

    if width <= 0.0 || height <= 0.0 {
        return None;
    }

    let x = (cursor_x / width) * 2.0 - 1.0;
    let y = 1.0 - (cursor_y / height) * 2.0;
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

fn ray_segment_distance(
    ray_origin: Vec3,
    ray_dir: Vec3,
    start: Vec3,
    end: Vec3,
) -> Option<(f32, f32)> {
    let dir = ray_dir.normalize_or_zero();
    if dir.length_squared() < 1.0e-6 {
        return None;
    }
    let seg = end - start;
    let seg_len_sq = seg.length_squared();
    if seg_len_sq < 1.0e-8 {
        let dist = ray_point_distance(ray_origin, dir, start);
        let ray_t = (start - ray_origin).dot(dir).max(0.0);
        return Some((dist, ray_t));
    }

    let w = start - ray_origin;
    let uv = dir.dot(seg);
    let vv = seg_len_sq;
    let uw = dir.dot(w);
    let vw = seg.dot(w);
    let denom = vv - uv * uv;
    let mut t = if denom.abs() > 1.0e-6 {
        (uw * uv - vw) / denom
    } else {
        0.0
    };
    t = t.clamp(0.0, 1.0);
    let closest_seg = start + seg * t;
    let mut ray_t = (closest_seg - ray_origin).dot(dir);
    if ray_t < 0.0 {
        ray_t = 0.0;
    }
    let closest_ray = ray_origin + dir * ray_t;
    Some((closest_ray.distance(closest_seg), ray_t))
}

fn spline_plane_normal(plane: SplineDrawPlane, camera_transform: &Transform) -> Vec3 {
    match plane {
        SplineDrawPlane::WorldXZ => Vec3::Y,
        SplineDrawPlane::View => camera_transform.forward().normalize_or_zero(),
    }
}

fn extrude_spline_point(
    spline: &mut Spline,
    active_index: usize,
    spline_transform: &Transform,
    camera_transform: &Transform,
    draw_plane: SplineDrawPlane,
    ray: Option<(Vec3, Vec3)>,
) -> Option<(usize, Vec3, Vec3)> {
    let count = spline.points.len();
    if count == 0 {
        return None;
    }
    let active_index = active_index.min(count.saturating_sub(1));
    let spline_matrix = spline_transform.to_matrix();
    let inv_spline_matrix = spline_matrix.inverse();
    let active_world = spline_matrix.transform_point3(spline.points[active_index]);

    let plane_normal = spline_plane_normal(draw_plane, camera_transform).normalize_or_zero();
    let new_world = ray
        .and_then(|(ray_origin, ray_dir)| {
            intersect_ray_plane(ray_origin, ray_dir, active_world, plane_normal)
        })
        .unwrap_or(active_world);
    let new_local = inv_spline_matrix.transform_point3(new_world);
    let insert_index = (active_index + 1).min(spline.points.len());
    spline.points.insert(insert_index, new_local);
    Some((insert_index, new_world, plane_normal))
}

fn insert_spline_midpoint(
    spline: &mut Spline,
    active_index: usize,
    spline_transform: &Transform,
    camera_transform: &Transform,
    draw_plane: SplineDrawPlane,
) -> Option<(usize, Vec3, Vec3)> {
    let count = spline.points.len();
    if count < 2 {
        return None;
    }
    let active_index = active_index.min(count.saturating_sub(1));
    let next_index = if active_index + 1 < count {
        active_index + 1
    } else if spline.closed {
        0
    } else {
        return None;
    };

    let a = spline.points[active_index];
    let b = spline.points[next_index];
    let midpoint = (a + b) * 0.5;
    let insert_index = (active_index + 1).min(spline.points.len());
    spline.points.insert(insert_index, midpoint);

    let spline_matrix = spline_transform.to_matrix();
    let new_world = spline_matrix.transform_point3(midpoint);
    let plane_normal = spline_plane_normal(draw_plane, camera_transform).normalize_or_zero();
    Some((insert_index, new_world, plane_normal))
}

fn build_spline_path_lines(spline: &Spline, transform: &Transform, samples: u32) -> Vec<GizmoLine> {
    let mut lines = Vec::new();
    if !spline.is_valid() {
        return lines;
    }

    let matrix = transform.to_matrix();
    let steps = samples.max(2);
    let mut prev = matrix.transform_point3(spline.sample(0.0));
    for i in 1..steps {
        let t = i as f32 / (steps.saturating_sub(1) as f32);
        let current = matrix.transform_point3(spline.sample(t));
        lines.push(GizmoLine {
            start: prev,
            end: current,
        });
        prev = current;
    }
    if spline.closed {
        let first = matrix.transform_point3(spline.sample(0.0));
        lines.push(GizmoLine {
            start: prev,
            end: first,
        });
    }

    lines
}

fn build_spline_handle_lines(
    spline: &Spline,
    transform: &Transform,
    handle_size: f32,
) -> Vec<GizmoLine> {
    let mut lines = Vec::new();
    let handle = handle_size.max(0.0);
    if handle <= 0.0 {
        return lines;
    }

    let matrix = transform.to_matrix();
    let half = handle * 0.5;
    for point in &spline.points {
        let p = matrix.transform_point3(*point);
        lines.push(GizmoLine {
            start: p - Vec3::X * half,
            end: p + Vec3::X * half,
        });
        lines.push(GizmoLine {
            start: p - Vec3::Y * half,
            end: p + Vec3::Y * half,
        });
        lines.push(GizmoLine {
            start: p - Vec3::Z * half,
            end: p + Vec3::Z * half,
        });
    }

    lines
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
