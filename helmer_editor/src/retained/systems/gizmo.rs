use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Resource, World};
use glam::{Mat4, Quat, Vec3};

use helmer::graphics::common::renderer::{
    GizmoAxis, GizmoData, GizmoIcon, GizmoIconKind, GizmoMode,
};
use helmer::provided::components::{Camera, LightType, Transform};
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::render_system::RenderGizmoState;
use helmer_becs::{BevyActiveCamera, BevyCamera, BevyInputManager, BevyLight, BevyTransform};
use helmer_ui::UiRect;

use crate::retained::panes::{ViewportPaneMode, ViewportPaneSurfaceKey};
use crate::retained::shell::EditorRetainedPaneInteractionState;
use crate::retained::state::{
    EditorRetainedGizmoSnapSettings, EditorRetainedViewportStates, EditorSceneState,
    RetainedEditorViewportCamera, WorldState,
};

use super::viewport::{
    can_control_surface, ensure_retained_editor_camera, resolve_gameplay_camera,
    surface_under_pointer,
};

const GIZMO_SIZE_SCALE: f32 = 0.12;
const GIZMO_SIZE_MIN: f32 = 0.25;
const GIZMO_SIZE_MAX: f32 = 100.0;
const GIZMO_SCALE_MIN: f32 = 0.01;

const CENTER_PICK_RADIUS_SCALE: f32 = 0.18;
const CENTER_PICK_RADIUS_MIN: f32 = 0.06;
const AXIS_PICK_RADIUS_SCALE: f32 = 0.14;
const AXIS_PICK_RADIUS_MIN: f32 = 0.05;
const ROTATE_PICK_RADIUS_SCALE: f32 = 0.08;
const ROTATE_PICK_RADIUS_MIN: f32 = 0.04;
const ROTATE_RADIUS_SCALE: f32 = 0.85;
const PLANE_OFFSET_SCALE: f32 = 0.22;
const PLANE_SIZE_SCALE: f32 = 0.20;
const PLANE_PICK_PADDING_SCALE: f32 = 0.08;
const PLANE_PICK_PADDING_MIN: f32 = 0.02;

#[derive(Resource, Debug, Clone)]
struct RetainedGizmoInteractionState {
    hover_axis: GizmoAxis,
    active_axis: GizmoAxis,
    last_left_down: bool,
    drag_surface: Option<ViewportPaneSurfaceKey>,
    drag: Option<RetainedGizmoDragState>,
}

impl Default for RetainedGizmoInteractionState {
    fn default() -> Self {
        Self {
            hover_axis: GizmoAxis::None,
            active_axis: GizmoAxis::None,
            last_left_down: false,
            drag_surface: None,
            drag: None,
        }
    }
}

#[derive(Debug, Clone)]
struct RetainedGizmoDragState {
    axis: GizmoAxis,
    mode: GizmoMode,
    start_transform: Transform,
    kind: RetainedGizmoDragKind,
}

#[derive(Debug, Clone)]
enum RetainedGizmoDragKind {
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
    PlaneScale {
        plane_normal: Vec3,
        axis_u: Vec3,
        axis_v: Vec3,
        start_u: f32,
        start_v: f32,
    },
    CenterRotate {
        plane_normal: Vec3,
        start_vector: Vec3,
    },
}

#[derive(Debug, Clone, Copy)]
struct DragSnapState {
    enabled: bool,
    translate_step: f32,
    rotate_step_radians: f32,
    scale_step: f32,
}

pub fn retained_gizmo_state_system(world: &mut World) {
    let mut interaction_state = world
        .remove_resource::<RetainedGizmoInteractionState>()
        .unwrap_or_default();

    let (
        cursor,
        pointer,
        window_size,
        left_down,
        right_down,
        wants_pointer,
        control_active,
        shift_active,
        alt_active,
    ) = if let Some(input_manager) = world.get_resource::<BevyInputManager>() {
        let input = input_manager.0.read();
        (
            input.cursor_position,
            glam::Vec2::new(
                input.cursor_position.x as f32,
                input.cursor_position.y as f32,
            ),
            input.window_size,
            input
                .active_mouse_buttons
                .contains(&winit::event::MouseButton::Left),
            input
                .active_mouse_buttons
                .contains(&winit::event::MouseButton::Right),
            input.egui_wants_pointer,
            input.is_key_active(winit::keyboard::KeyCode::ControlLeft)
                || input.is_key_active(winit::keyboard::KeyCode::ControlRight),
            input.is_key_active(winit::keyboard::KeyCode::ShiftLeft)
                || input.is_key_active(winit::keyboard::KeyCode::ShiftRight),
            input.is_key_active(winit::keyboard::KeyCode::AltLeft)
                || input.is_key_active(winit::keyboard::KeyCode::AltRight),
        )
    } else {
        (
            glam::DVec2::ZERO,
            glam::Vec2::ZERO,
            glam::UVec2::ZERO,
            false,
            false,
            false,
            false,
            false,
            false,
        )
    };

    let left_pressed = left_down && !interaction_state.last_left_down;
    let left_released = !left_down && interaction_state.last_left_down;

    let surfaces = world
        .get_resource::<EditorRetainedPaneInteractionState>()
        .map(|state| state.viewport_surfaces.clone())
        .unwrap_or_default();
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);

    interaction_state.drag_surface = interaction_state.drag_surface.filter(|surface| {
        surfaces.contains_key(surface) && can_control_surface(*surface, world_state)
    });

    let hovered_surface = surface_under_pointer(&surfaces, pointer)
        .filter(|surface| can_control_surface(*surface, world_state));
    let target_surface = interaction_state
        .drag_surface
        .or(hovered_surface)
        .filter(|surface| can_control_surface(*surface, world_state));

    let viewport_rect = target_surface.and_then(|surface| surfaces.get(&surface).copied());
    let pointer_in_viewport = viewport_rect
        .map(|rect| point_in_rect(rect, pointer))
        .unwrap_or(false);
    let pointer_over_active_viewport =
        target_surface.is_some() && target_surface == hovered_surface;

    let viewport_states = world
        .get_resource::<EditorRetainedViewportStates>()
        .cloned()
        .unwrap_or_default();
    let active_viewport_state = target_surface
        .map(|surface| viewport_states.state_for(surface.tab_id))
        .or_else(|| viewport_states.states.values().next().cloned())
        .unwrap_or_default();

    let show_gizmos = world_state == WorldState::Edit || active_viewport_state.gizmos_in_play;
    let gizmo_mode = active_viewport_state.gizmo_mode;

    let mut snap_settings = world
        .get_resource::<EditorRetainedGizmoSnapSettings>()
        .cloned()
        .unwrap_or_default();
    snap_settings.sanitize();

    let editor_camera = ensure_retained_editor_camera(world);
    let gameplay_camera = resolve_gameplay_camera(world, editor_camera);
    let interaction_camera = target_surface.and_then(|surface| match surface.mode {
        ViewportPaneMode::Edit => Some(editor_camera),
        ViewportPaneMode::Play => match world_state {
            WorldState::Edit => None,
            WorldState::Play => gameplay_camera.or(Some(editor_camera)),
        },
    });
    let ray_camera = interaction_camera.or(Some(editor_camera));

    let camera_snapshot = ray_camera.and_then(|entity| camera_snapshot_for_entity(world, entity));

    let selected_entity = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selection| selection.0)
        .filter(|entity| world.get::<BevyTransform>(*entity).is_some());

    let mut selected_transform = selected_entity.and_then(|entity| {
        world
            .get::<BevyTransform>(entity)
            .map(|transform| transform.0)
    });

    if !show_gizmos || selected_transform.is_none() {
        interaction_state.hover_axis = GizmoAxis::None;
        interaction_state.active_axis = GizmoAxis::None;
        interaction_state.drag = None;
        interaction_state.drag_surface = None;
    } else {
        let allow_raycast = pointer_over_active_viewport
            || (!wants_pointer && pointer_in_viewport)
            || interaction_state.drag.is_some();
        let ray = camera_snapshot.and_then(|(camera, camera_transform)| {
            if wants_pointer && !allow_raycast {
                return None;
            }
            let inv_view_proj = camera_inv_view_proj(&camera, &camera_transform, viewport_rect);
            screen_ray(
                cursor,
                viewport_rect,
                window_size,
                inv_view_proj,
                camera_transform.position,
                interaction_state.drag.is_some(),
            )
        });

        let transform = selected_transform.unwrap_or_default();
        let camera_position = camera_snapshot
            .map(|(_, camera_transform)| camera_transform.position)
            .unwrap_or(Vec3::ZERO);
        let distance = camera_position.distance(transform.position).max(1.0e-3);
        let gizmo_size = (distance * GIZMO_SIZE_SCALE).clamp(GIZMO_SIZE_MIN, GIZMO_SIZE_MAX);
        let view_dir = (camera_position - transform.position).normalize_or_zero();

        if let Some((ray_origin, ray_dir)) = ray {
            let mut snap_enabled = snap_settings.enabled;
            if snap_settings.ctrl_toggles && control_active {
                snap_enabled = !snap_enabled;
            }
            let mut snap_scale = 1.0f32;
            if snap_settings.shift_fine && shift_active {
                snap_scale *= snap_settings.fine_scale.max(0.001);
            }
            if snap_settings.alt_coarse && alt_active {
                snap_scale *= snap_settings.coarse_scale.max(1.0);
            }
            let snap_state = DragSnapState {
                enabled: snap_enabled,
                translate_step: snap_settings.translate_step.max(0.0) * snap_scale,
                rotate_step_radians: snap_settings.rotate_step_degrees.max(0.0).to_radians()
                    * snap_scale,
                scale_step: snap_settings.scale_step.max(0.0) * snap_scale,
            };

            if interaction_state.drag.is_none() && gizmo_mode != GizmoMode::None && !right_down {
                interaction_state.hover_axis = pick_gizmo(
                    gizmo_mode,
                    ray_origin,
                    ray_dir,
                    transform.position,
                    transform.rotation,
                    gizmo_size,
                );
            }

            if left_pressed
                && interaction_state.drag.is_none()
                && gizmo_mode != GizmoMode::None
                && interaction_state.hover_axis != GizmoAxis::None
                && !right_down
            {
                if let Some(drag) = begin_drag(
                    gizmo_mode,
                    interaction_state.hover_axis,
                    ray_origin,
                    ray_dir,
                    transform.position,
                    view_dir,
                    transform,
                ) {
                    interaction_state.active_axis = drag.axis;
                    interaction_state.drag = Some(drag);
                    interaction_state.drag_surface = target_surface;
                }
            }

            if let Some(drag) = interaction_state.drag.take() {
                if left_down {
                    let mut next_transform = transform;
                    apply_drag(&mut next_transform, &drag, ray_origin, ray_dir, snap_state);
                    if !transform_approx_eq(next_transform, transform) {
                        selected_transform = Some(next_transform);
                    }
                    interaction_state.active_axis = drag.axis;
                    interaction_state.drag = Some(drag);
                    if interaction_state.drag_surface.is_none() {
                        interaction_state.drag_surface = target_surface;
                    }
                } else {
                    interaction_state.active_axis = GizmoAxis::None;
                    interaction_state.drag = None;
                    interaction_state.drag_surface = None;
                }
            }
        } else if left_released {
            interaction_state.active_axis = GizmoAxis::None;
            interaction_state.drag = None;
            interaction_state.drag_surface = None;
            interaction_state.hover_axis = GizmoAxis::None;
        }
    }

    if let (Some(entity), Some(transform)) = (selected_entity, selected_transform)
        && let Some(mut transform_component) = world.get_mut::<BevyTransform>(entity)
        && !transform_approx_eq(transform_component.0, transform)
    {
        transform_component.0 = transform;
        mark_scene_dirty(world);
    }

    let mut gizmo = GizmoData::default();
    gizmo.mode = gizmo_mode;

    if show_gizmos
        && let Some(entity) = selected_entity
        && let Some(transform) = world
            .get::<BevyTransform>(entity)
            .map(|component| component.0)
    {
        let camera_position = camera_snapshot
            .map(|(_, camera_transform)| camera_transform.position)
            .unwrap_or(Vec3::ZERO);
        let distance = camera_position.distance(transform.position).max(0.25);
        let size = (distance * GIZMO_SIZE_SCALE).clamp(GIZMO_SIZE_MIN, GIZMO_SIZE_MAX);
        let extent = transform.scale.abs().max(glam::Vec3::splat(0.25)) * 0.5;
        gizmo.position = transform.position;
        gizmo.rotation = transform.rotation;
        gizmo.scale = transform.scale;
        gizmo.size = size;
        gizmo.hover_axis = interaction_state.hover_axis;
        gizmo.active_axis = interaction_state.active_axis;
        let _ = extent;
        gizmo.selection_enabled = false;
        gizmo.selection_min = glam::Vec3::ZERO;
        gizmo.selection_max = glam::Vec3::ZERO;
    } else {
        gizmo.hover_axis = GizmoAxis::None;
        gizmo.active_axis = GizmoAxis::None;
    }

    if show_gizmos {
        append_icon_gizmos(world, &active_viewport_state, &mut gizmo);
    }

    if let Some(mut render_gizmo) = world.get_resource_mut::<RenderGizmoState>() {
        render_gizmo.0 = gizmo;
    } else {
        world.insert_resource(RenderGizmoState(gizmo));
    }

    interaction_state.last_left_down = left_down;
    world.insert_resource(interaction_state);
}

fn append_icon_gizmos(
    world: &World,
    viewport_state: &crate::retained::state::EditorRetainedViewportState,
    gizmo: &mut GizmoData,
) {
    for entity_ref in world.iter_entities() {
        let entity = entity_ref.id();

        if viewport_state.show_camera_gizmos
            && world.get::<RetainedEditorViewportCamera>(entity).is_none()
            && let (Some(camera), Some(transform)) = (
                world.get::<BevyCamera>(entity),
                world.get::<BevyTransform>(entity),
            )
        {
            let active = world.get::<BevyActiveCamera>(entity).is_some();
            let color = if active {
                glam::Vec3::new(1.0, 0.6, 0.2)
            } else {
                glam::Vec3::new(0.75, 0.9, 1.0)
            };
            gizmo.icons.push(GizmoIcon {
                position: transform.0.position,
                rotation: transform.0.rotation,
                size: 0.9,
                color,
                alpha: if active { 1.0 } else { 0.9 },
                kind: GizmoIconKind::Camera,
                params: [
                    camera.0.fov_y_rad,
                    camera.0.aspect_ratio,
                    camera.0.near_plane,
                    camera.0.far_plane,
                ],
            });
        }

        if let (Some(light), Some(transform)) = (
            world.get::<BevyLight>(entity),
            world.get::<BevyTransform>(entity),
        ) {
            let kind = match light.0.light_type {
                LightType::Directional if viewport_state.show_directional_light_gizmos => {
                    Some(GizmoIconKind::LightDirectional)
                }
                LightType::Point if viewport_state.show_point_light_gizmos => {
                    Some(GizmoIconKind::LightPoint)
                }
                LightType::Spot { .. } if viewport_state.show_spot_light_gizmos => {
                    Some(GizmoIconKind::LightSpot)
                }
                _ => None,
            };
            if let Some(kind) = kind {
                gizmo.icons.push(GizmoIcon {
                    position: transform.0.position,
                    rotation: transform.0.rotation,
                    size: 0.85,
                    color: light.0.color.clamp(glam::Vec3::ZERO, glam::Vec3::ONE),
                    alpha: 0.9,
                    kind,
                    params: [light.0.intensity, 0.0, 0.0, 0.0],
                });
            }
        }
    }
}

fn camera_snapshot_for_entity(world: &World, entity: Entity) -> Option<(Camera, Transform)> {
    let camera = world.get::<BevyCamera>(entity)?;
    let transform = world.get::<BevyTransform>(entity)?;
    Some((camera.0.clone(), transform.0))
}

fn camera_inv_view_proj(
    camera: &Camera,
    transform: &Transform,
    viewport_rect: Option<UiRect>,
) -> Mat4 {
    let forward = transform.forward().normalize_or_zero();
    let up = transform.up().normalize_or_zero();
    let view = Mat4::look_at_rh(transform.position, transform.position + forward, up);
    let aspect = viewport_rect
        .map(|rect| rect.width / rect.height.max(1.0))
        .filter(|aspect| aspect.is_finite() && *aspect > 0.0)
        .unwrap_or(camera.aspect_ratio.max(0.001));
    let projection = Mat4::perspective_infinite_reverse_rh(
        camera.fov_y_rad.max(0.001),
        aspect,
        camera.near_plane.max(0.001),
    );
    (projection * view).inverse()
}

fn screen_ray(
    cursor: glam::DVec2,
    viewport_rect: Option<UiRect>,
    window_size: glam::UVec2,
    inv_view_proj: Mat4,
    camera_position: Vec3,
    allow_outside_viewport: bool,
) -> Option<(Vec3, Vec3)> {
    let (cursor_x, cursor_y, width, height) = if let Some(rect) = viewport_rect {
        let width = rect.width.max(1.0);
        let height = rect.height.max(1.0);
        let local_x = cursor.x as f32 - rect.x;
        let local_y = cursor.y as f32 - rect.y;
        let outside = local_x < 0.0 || local_x > width || local_y < 0.0 || local_y > height;
        if outside && !allow_outside_viewport {
            return None;
        }
        (
            local_x.clamp(0.0, width),
            local_y.clamp(0.0, height),
            width,
            height,
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

fn pick_gizmo(
    mode: GizmoMode,
    ray_origin: Vec3,
    ray_dir: Vec3,
    origin: Vec3,
    rotation: Quat,
    size: f32,
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

    let center_radius = (size * CENTER_PICK_RADIUS_SCALE).max(CENTER_PICK_RADIUS_MIN);
    let center_dist = ray_point_distance(ray_origin, ray_dir, origin);
    if center_dist <= center_radius {
        return GizmoAxis::Center;
    }

    match mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            let plane_offset = size * PLANE_OFFSET_SCALE.max(0.0);
            let plane_size = size * PLANE_SIZE_SCALE.max(0.0);
            let plane_padding = (size * PLANE_PICK_PADDING_SCALE)
                .max(PLANE_PICK_PADDING_MIN)
                .max(0.0);
            if plane_size > 0.0 {
                let min = plane_offset - plane_padding;
                let max = plane_offset + plane_size + plane_padding;
                for axis in [GizmoAxis::XY, GizmoAxis::XZ, GizmoAxis::YZ] {
                    let Some((axis_u, axis_v, normal)) = plane_axes(rotation, axis) else {
                        continue;
                    };
                    let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, normal) else {
                        continue;
                    };
                    let rel = hit - origin;
                    let u = rel.dot(axis_u);
                    let v = rel.dot(axis_v);
                    if u >= min && u <= max && v >= min && v <= max {
                        let center = plane_offset + plane_size * 0.5;
                        let score = ((u - center).abs() + (v - center).abs())
                            / (plane_size + plane_padding * 2.0).max(1.0e-4);
                        if score < best_score {
                            best_score = score;
                            best_axis = axis;
                        }
                    }
                }
            }

            let threshold = (size * AXIS_PICK_RADIUS_SCALE).max(AXIS_PICK_RADIUS_MIN);
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
            let ring_radius = size * ROTATE_RADIUS_SCALE;
            let band = (size * ROTATE_PICK_RADIUS_SCALE).max(ROTATE_PICK_RADIUS_MIN);
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
) -> Option<RetainedGizmoDragState> {
    if axis == GizmoAxis::Center {
        let plane_normal = view_dir.normalize_or_zero();
        match mode {
            GizmoMode::Translate => {
                let hit = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal)?;
                return Some(RetainedGizmoDragState {
                    axis,
                    mode,
                    start_transform,
                    kind: RetainedGizmoDragKind::CenterTranslate {
                        plane_normal,
                        start_hit: hit,
                    },
                });
            }
            GizmoMode::Scale => {
                let hit = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal)?;
                let start_distance = (hit - origin).length().max(GIZMO_SCALE_MIN);
                return Some(RetainedGizmoDragState {
                    axis,
                    mode,
                    start_transform,
                    kind: RetainedGizmoDragKind::CenterScale {
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
                return Some(RetainedGizmoDragState {
                    axis,
                    mode,
                    start_transform,
                    kind: RetainedGizmoDragKind::CenterRotate {
                        plane_normal,
                        start_vector: vec,
                    },
                });
            }
            GizmoMode::None => return None,
        }
    }

    if let Some((axis_u, axis_v, plane_normal)) = plane_axes(start_transform.rotation, axis) {
        match mode {
            GizmoMode::Translate => {
                let hit = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal)?;
                return Some(RetainedGizmoDragState {
                    axis,
                    mode,
                    start_transform,
                    kind: RetainedGizmoDragKind::CenterTranslate {
                        plane_normal,
                        start_hit: hit,
                    },
                });
            }
            GizmoMode::Scale => {
                let hit = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal)?;
                let rel = hit - origin;
                let start_u = rel.dot(axis_u).abs().max(1.0e-4);
                let start_v = rel.dot(axis_v).abs().max(1.0e-4);
                return Some(RetainedGizmoDragState {
                    axis,
                    mode,
                    start_transform,
                    kind: RetainedGizmoDragKind::PlaneScale {
                        plane_normal,
                        axis_u,
                        axis_v,
                        start_u,
                        start_v,
                    },
                });
            }
            GizmoMode::Rotate | GizmoMode::None => return None,
        }
    }

    let axis_dir = axis_direction(start_transform.rotation, axis).normalize_or_zero();
    match mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            let start_axis_param = closest_point_on_axis(ray_origin, ray_dir, origin, axis_dir)
                .map(|(axis_t, _)| axis_t)?;
            Some(RetainedGizmoDragState {
                axis,
                mode,
                start_transform,
                kind: RetainedGizmoDragKind::AxisLine {
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
            Some(RetainedGizmoDragState {
                axis,
                mode,
                start_transform,
                kind: RetainedGizmoDragKind::AxisRotate {
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
    drag: &RetainedGizmoDragState,
    ray_origin: Vec3,
    ray_dir: Vec3,
    snap: DragSnapState,
) {
    let origin = drag.start_transform.position;
    match drag.mode {
        GizmoMode::Translate => match drag.kind {
            RetainedGizmoDragKind::AxisLine {
                axis_dir,
                start_axis_param,
            } => {
                if let Some((axis_t, _)) =
                    closest_point_on_axis(ray_origin, ray_dir, origin, axis_dir)
                {
                    let mut delta = axis_t - start_axis_param;
                    if snap.enabled && snap.translate_step > 1.0e-6 {
                        delta = snap_to_step(delta, snap.translate_step);
                    }
                    transform.position = drag.start_transform.position + axis_dir * delta;
                }
            }
            RetainedGizmoDragKind::CenterTranslate {
                plane_normal,
                start_hit,
            } => {
                if let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal) {
                    let mut delta = hit - start_hit;
                    if snap.enabled && snap.translate_step > 1.0e-6 {
                        delta = snap_vec3(delta, snap.translate_step);
                    }
                    transform.position = drag.start_transform.position + delta;
                }
            }
            _ => {}
        },
        GizmoMode::Scale => match drag.kind {
            RetainedGizmoDragKind::AxisLine {
                axis_dir,
                start_axis_param,
            } => {
                if let Some((axis_t, _)) =
                    closest_point_on_axis(ray_origin, ray_dir, origin, axis_dir)
                {
                    let delta = axis_t - start_axis_param;
                    let mut scale = drag.start_transform.scale;
                    match drag.axis {
                        GizmoAxis::X => scale.x = (scale.x + delta).max(GIZMO_SCALE_MIN),
                        GizmoAxis::Y => scale.y = (scale.y + delta).max(GIZMO_SCALE_MIN),
                        GizmoAxis::Z => scale.z = (scale.z + delta).max(GIZMO_SCALE_MIN),
                        _ => {}
                    }
                    if snap.enabled && snap.scale_step > 1.0e-6 {
                        scale = snap_scale_vec3(scale, snap.scale_step, GIZMO_SCALE_MIN);
                    }
                    transform.scale = scale;
                }
            }
            RetainedGizmoDragKind::CenterScale {
                plane_normal,
                start_distance,
            } => {
                if let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal) {
                    let distance = (hit - origin).length().max(GIZMO_SCALE_MIN);
                    let factor = distance / start_distance;
                    let mut scale = drag.start_transform.scale * factor;
                    scale.x = scale.x.max(GIZMO_SCALE_MIN);
                    scale.y = scale.y.max(GIZMO_SCALE_MIN);
                    scale.z = scale.z.max(GIZMO_SCALE_MIN);
                    if snap.enabled && snap.scale_step > 1.0e-6 {
                        scale = snap_scale_vec3(scale, snap.scale_step, GIZMO_SCALE_MIN);
                    }
                    transform.scale = scale;
                }
            }
            RetainedGizmoDragKind::PlaneScale {
                plane_normal,
                axis_u,
                axis_v,
                start_u,
                start_v,
            } => {
                if let Some(hit) = intersect_ray_plane(ray_origin, ray_dir, origin, plane_normal) {
                    let rel = hit - origin;
                    let current_u = rel.dot(axis_u).abs().max(1.0e-4);
                    let current_v = rel.dot(axis_v).abs().max(1.0e-4);
                    let factor_u = (current_u / start_u.max(1.0e-4)).max(1.0e-4);
                    let factor_v = (current_v / start_v.max(1.0e-4)).max(1.0e-4);
                    let mut scale = drag.start_transform.scale;
                    match drag.axis {
                        GizmoAxis::XY => {
                            scale.x = (scale.x * factor_u).max(GIZMO_SCALE_MIN);
                            scale.y = (scale.y * factor_v).max(GIZMO_SCALE_MIN);
                        }
                        GizmoAxis::XZ => {
                            scale.x = (scale.x * factor_u).max(GIZMO_SCALE_MIN);
                            scale.z = (scale.z * factor_v).max(GIZMO_SCALE_MIN);
                        }
                        GizmoAxis::YZ => {
                            scale.y = (scale.y * factor_u).max(GIZMO_SCALE_MIN);
                            scale.z = (scale.z * factor_v).max(GIZMO_SCALE_MIN);
                        }
                        _ => {}
                    }
                    if snap.enabled && snap.scale_step > 1.0e-6 {
                        scale = snap_scale_vec3(scale, snap.scale_step, GIZMO_SCALE_MIN);
                    }
                    transform.scale = scale;
                }
            }
            _ => {}
        },
        GizmoMode::Rotate => match drag.kind {
            RetainedGizmoDragKind::AxisRotate {
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
                        if snap.enabled && snap.rotate_step_radians > 1.0e-6 {
                            angle = snap_to_step(angle, snap.rotate_step_radians);
                        }
                        let delta = Quat::from_axis_angle(axis_dir, angle);
                        transform.rotation = delta * drag.start_transform.rotation;
                    }
                }
            }
            RetainedGizmoDragKind::CenterRotate {
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
                        if snap.enabled && snap.rotate_step_radians > 1.0e-6 {
                            angle = snap_to_step(angle, snap.rotate_step_radians);
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

fn axis_direction(rotation: Quat, axis: GizmoAxis) -> Vec3 {
    match axis {
        GizmoAxis::X => rotation * Vec3::X,
        GizmoAxis::Y => rotation * Vec3::Y,
        GizmoAxis::Z => rotation * Vec3::Z,
        _ => Vec3::ZERO,
    }
}

fn plane_axes(rotation: Quat, axis: GizmoAxis) -> Option<(Vec3, Vec3, Vec3)> {
    let x = rotation * Vec3::X;
    let y = rotation * Vec3::Y;
    let z = rotation * Vec3::Z;
    match axis {
        GizmoAxis::XY => Some((x, y, z)),
        GizmoAxis::XZ => Some((x, z, y)),
        GizmoAxis::YZ => Some((y, z, x)),
        _ => None,
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
    Some((axis_t, closest_ray.distance(closest_axis)))
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

fn ray_point_distance(ray_origin: Vec3, ray_dir: Vec3, point: Vec3) -> f32 {
    let to_point = point - ray_origin;
    let t = to_point.dot(ray_dir);
    if t < 0.0 {
        return to_point.length();
    }
    let closest = ray_origin + ray_dir * t;
    closest.distance(point)
}

fn snap_to_step(value: f32, step: f32) -> f32 {
    if !value.is_finite() || !step.is_finite() || step <= 1.0e-6 {
        return value;
    }
    (value / step).round() * step
}

fn snap_vec3(value: Vec3, step: f32) -> Vec3 {
    Vec3::new(
        snap_to_step(value.x, step),
        snap_to_step(value.y, step),
        snap_to_step(value.z, step),
    )
}

fn snap_scale_vec3(scale: Vec3, step: f32, min_scale: f32) -> Vec3 {
    Vec3::new(
        snap_to_step(scale.x, step).max(min_scale),
        snap_to_step(scale.y, step).max(min_scale),
        snap_to_step(scale.z, step).max(min_scale),
    )
}

fn transform_approx_eq(a: Transform, b: Transform) -> bool {
    a.position.distance_squared(b.position) <= 1.0e-12
        && a.rotation.dot(b.rotation).abs() >= 1.0 - 1.0e-6
        && a.scale.distance_squared(b.scale) <= 1.0e-12
}

fn point_in_rect(rect: UiRect, point: glam::Vec2) -> bool {
    point.x >= rect.x
        && point.x <= rect.x + rect.width
        && point.y >= rect.y
        && point.y <= rect.y + rect.height
}

fn mark_scene_dirty(world: &mut World) {
    if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
        scene.dirty = true;
    }
}
