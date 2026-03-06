use std::collections::HashMap;

use bevy_ecs::prelude::{Resource, World};
use bevy_ecs::{entity::Entity, name::Name};
use helmer::graphics::common::renderer::{
    GizmoAxis, RenderViewportGizmoOptions, RenderViewportRequest,
};
use helmer::provided::components::{ActiveCamera, Camera, Transform};
use helmer::runtime::runtime::{RuntimeCursorGrabMode, RuntimeCursorStateSnapshot};
use helmer_becs::DeltaTime;
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::render_system::{RenderGizmoState, RenderViewportRequests};
use helmer_becs::{
    BevyActiveCamera, BevyCamera, BevyInputManager, BevyRuntimeCursorState, BevyTransform,
    BevyWrapper,
};
use helmer_ui::UiRect;

use crate::retained::panes::{
    ViewportPaneMode, ViewportPaneSurfaceKey, retained_viewport_id, retained_viewport_preview_id,
};
use crate::retained::shell::EditorRetainedPaneInteractionState;
use crate::retained::state::{
    EditorRetainedViewportStates, EditorSceneState, FREECAM_BOOST_MULTIPLIER_MAX,
    FREECAM_BOOST_MULTIPLIER_MIN, FREECAM_MOVE_ACCEL_MAX, FREECAM_MOVE_ACCEL_MIN,
    FREECAM_MOVE_DECEL_MAX, FREECAM_MOVE_DECEL_MIN, FREECAM_ORBIT_DISTANCE_MAX,
    FREECAM_ORBIT_DISTANCE_MIN, FREECAM_ORBIT_PAN_SENSITIVITY_MAX,
    FREECAM_ORBIT_PAN_SENSITIVITY_MIN, FREECAM_PAN_SENSITIVITY_MAX, FREECAM_PAN_SENSITIVITY_MIN,
    FREECAM_SENSITIVITY_MAX, FREECAM_SENSITIVITY_MIN, FREECAM_SMOOTHING_MAX, FREECAM_SMOOTHING_MIN,
    FREECAM_SPEED_MAX_MAX, FREECAM_SPEED_MAX_MIN, FREECAM_SPEED_MIN_MAX, FREECAM_SPEED_MIN_MIN,
    FREECAM_SPEED_STEP_MAX, FREECAM_SPEED_STEP_MIN, RetainedEditorViewportCamera, WorldState,
};

pub fn retained_active_camera_system(world: &mut World) {
    let editor_camera = ensure_retained_editor_camera(world);
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);
    let camera_entities = world
        .query::<(
            Entity,
            &BevyCamera,
            &BevyTransform,
            Option<&BevyActiveCamera>,
        )>()
        .iter(world)
        .map(|(entity, _, _, active)| {
            (
                entity,
                active.is_some(),
                world.get::<RetainedEditorViewportCamera>(entity).is_some(),
            )
        })
        .collect::<Vec<_>>();

    if camera_entities.is_empty() {
        world
            .entity_mut(editor_camera)
            .insert(BevyWrapper(ActiveCamera {}));
        return;
    }

    let preferred = if world_state == WorldState::Edit {
        Some(editor_camera)
    } else {
        camera_entities
            .iter()
            .find_map(|(entity, is_active, is_editor)| {
                (*is_active && !*is_editor).then_some(*entity)
            })
            .or_else(|| {
                camera_entities
                    .iter()
                    .find_map(|(entity, _is_active, is_editor)| (!*is_editor).then_some(*entity))
            })
            .or_else(|| camera_entities.first().map(|(entity, _, _)| *entity))
    };

    let Some(preferred) = preferred else {
        return;
    };

    for (entity, is_active, _) in camera_entities {
        if entity == preferred {
            if !is_active {
                world
                    .entity_mut(entity)
                    .insert(BevyWrapper(ActiveCamera {}));
            }
            continue;
        }
        if is_active {
            world.entity_mut(entity).remove::<BevyActiveCamera>();
        }
    }
}

pub(crate) fn ensure_retained_editor_camera(world: &mut World) -> Entity {
    if let Some((entity, _, _)) = world
        .query::<(Entity, &RetainedEditorViewportCamera, &BevyCamera)>()
        .iter(world)
        .find(|(entity, _, _)| world.get::<BevyTransform>(*entity).is_some())
    {
        return entity;
    }

    world
        .spawn((
            RetainedEditorViewportCamera,
            Name::new("Editor Camera"),
            BevyTransform::default(),
            BevyCamera::default(),
        ))
        .id()
}

pub(crate) fn camera_snapshot_for_entity(
    world: &World,
    entity: Entity,
) -> Option<(Camera, Transform)> {
    let camera = world.get::<BevyCamera>(entity)?;
    let transform = world.get::<BevyTransform>(entity)?;
    Some((camera.0.clone(), transform.0))
}

pub(crate) fn resolve_gameplay_camera(world: &World, editor_camera: Entity) -> Option<Entity> {
    let mut fallback: Option<Entity> = None;
    for entity_ref in world.iter_entities() {
        let entity = entity_ref.id();
        if world.get::<BevyCamera>(entity).is_none() || world.get::<BevyTransform>(entity).is_none()
        {
            continue;
        }
        if entity == editor_camera || world.get::<RetainedEditorViewportCamera>(entity).is_some() {
            continue;
        }
        if fallback.is_none() {
            fallback = Some(entity);
        }
        if world.get::<BevyActiveCamera>(entity).is_some() {
            return Some(entity);
        }
    }
    fallback
}

fn resolve_preview_camera_entity(world: &World, editor_camera: Entity) -> Option<Entity> {
    world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0)
        .filter(|entity| {
            *entity != editor_camera
                && world.get::<RetainedEditorViewportCamera>(*entity).is_none()
                && world.get::<BevyCamera>(*entity).is_some()
                && world.get::<BevyTransform>(*entity).is_some()
        })
}

#[derive(Resource, Debug, Clone)]
pub struct RetainedViewportControlState {
    speed: f32,
    move_velocity: glam::Vec3,
    sensitivity: f32,
    look_smoothing: f32,
    move_accel: f32,
    move_decel: f32,
    speed_step: f32,
    speed_min: f32,
    speed_max: f32,
    boost_multiplier: f32,
    fov_lerp_speed: f32,
    is_looking: bool,
    is_middle_looking: bool,
    middle_orbit_distance: f32,
    configured_orbit_distance: f32,
    middle_orbit_focus: Option<glam::Vec3>,
    look_start_cursor_position: Option<glam::DVec2>,
    look_surface: Option<ViewportPaneSurfaceKey>,
    smoothed_cursor_delta: glam::DVec2,
    current_fov_multiplier: f32,
    active_entity: Option<Entity>,
    last_cursor_position: glam::DVec2,
}

impl Default for RetainedViewportControlState {
    fn default() -> Self {
        Self {
            speed: 0.0,
            move_velocity: glam::Vec3::ZERO,
            sensitivity: 0.0,
            look_smoothing: 0.0,
            move_accel: 0.0,
            move_decel: 0.0,
            speed_step: 0.0,
            speed_min: 0.0,
            speed_max: 0.0,
            boost_multiplier: 0.0,
            fov_lerp_speed: 8.0,
            is_looking: false,
            is_middle_looking: false,
            middle_orbit_distance: 5.0,
            configured_orbit_distance: 5.0,
            middle_orbit_focus: None,
            look_start_cursor_position: None,
            look_surface: None,
            smoothed_cursor_delta: glam::DVec2::ZERO,
            current_fov_multiplier: 1.0,
            active_entity: None,
            last_cursor_position: glam::DVec2::ZERO,
        }
    }
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct RetainedCursorControlState {
    pub freecam_capture_active: bool,
    pub script_capture_allowed: bool,
    pub script_capture_suspended: bool,
    pub script_policy: Option<RuntimeCursorStateSnapshot>,
}

impl RetainedCursorControlState {
    pub fn effective_policy(&self) -> RuntimeCursorStateSnapshot {
        if self.freecam_capture_active {
            RuntimeCursorStateSnapshot {
                visible: false,
                grab_mode: RuntimeCursorGrabMode::Locked,
            }
        } else if self.script_capture_allowed && !self.script_capture_suspended {
            self.script_policy.unwrap_or_default()
        } else {
            RuntimeCursorStateSnapshot::default()
        }
    }
}

impl Default for RetainedCursorControlState {
    fn default() -> Self {
        Self {
            freecam_capture_active: false,
            script_capture_allowed: false,
            script_capture_suspended: false,
            script_policy: None,
        }
    }
}

pub fn retained_viewport_camera_controls_system(world: &mut World) {
    let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
        return;
    };
    let input = input_manager.0.read();
    let cursor_position = glam::DVec2::new(input.cursor_position.x, input.cursor_position.y);
    let pointer = glam::Vec2::new(cursor_position.x as f32, cursor_position.y as f32);
    let raw_mouse_delta = input.mouse_motion;
    let wheel_delta = input.mouse_wheel_unfiltered.y;
    let right_mouse_down = input
        .active_mouse_buttons
        .contains(&winit::event::MouseButton::Right);
    let middle_mouse_down = input
        .active_mouse_buttons
        .contains(&winit::event::MouseButton::Middle);
    let wants_pointer = input.egui_wants_pointer;
    let shift_active = input.is_key_active(winit::keyboard::KeyCode::ShiftLeft)
        || input.is_key_active(winit::keyboard::KeyCode::ShiftRight);
    let move_forward = input.is_key_active(winit::keyboard::KeyCode::KeyW);
    let move_back = input.is_key_active(winit::keyboard::KeyCode::KeyS);
    let move_left = input.is_key_active(winit::keyboard::KeyCode::KeyA);
    let move_right = input.is_key_active(winit::keyboard::KeyCode::KeyD);
    let move_up = input.is_key_active(winit::keyboard::KeyCode::Space)
        || input.is_key_active(winit::keyboard::KeyCode::KeyE);
    let move_down = input.is_key_active(winit::keyboard::KeyCode::KeyC)
        || input.is_key_active(winit::keyboard::KeyCode::KeyQ);
    drop(input);

    let dt = world
        .get_resource::<DeltaTime>()
        .map(|delta| delta.0.max(0.0))
        .unwrap_or(0.0);

    let surfaces = world
        .get_resource::<EditorRetainedPaneInteractionState>()
        .map(|state| state.viewport_surfaces.clone())
        .unwrap_or_default();
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);
    let viewport_states = world
        .get_resource::<EditorRetainedViewportStates>()
        .cloned()
        .unwrap_or_default();

    let mut control_state = world
        .remove_resource::<RetainedViewportControlState>()
        .unwrap_or_default();
    let mut cursor_control_state = world
        .remove_resource::<RetainedCursorControlState>()
        .unwrap_or_default();
    let hovered_surface = surface_under_pointer(&surfaces, pointer)
        .filter(|surface| can_control_surface(*surface, world_state));

    if control_state.speed == 0.0 {
        control_state.speed = 1.0;
    }

    control_state.look_surface = control_state.look_surface.filter(|surface| {
        surfaces.contains_key(surface) && can_control_surface(*surface, world_state)
    });

    let target_surface = control_state
        .look_surface
        .or(hovered_surface)
        .filter(|surface| can_control_surface(*surface, world_state));
    let pointer_in_viewport = target_surface
        .and_then(|surface| surfaces.get(&surface))
        .map(|rect| {
            pointer.x >= rect.x
                && pointer.x <= rect.x + rect.width
                && pointer.y >= rect.y
                && pointer.y <= rect.y + rect.height
        })
        .unwrap_or(false);
    let pointer_over_active_viewport =
        target_surface.is_some() && hovered_surface == target_surface;

    let mut viewport_state = target_surface
        .map(|surface| viewport_states.state_for(surface.tab_id))
        .unwrap_or_default();
    viewport_state.sanitize();

    control_state.sensitivity = viewport_state
        .freecam_sensitivity
        .clamp(FREECAM_SENSITIVITY_MIN, FREECAM_SENSITIVITY_MAX);
    control_state.look_smoothing = viewport_state
        .freecam_smoothing
        .clamp(FREECAM_SMOOTHING_MIN, FREECAM_SMOOTHING_MAX);
    control_state.move_accel = viewport_state
        .freecam_move_accel
        .clamp(FREECAM_MOVE_ACCEL_MIN, FREECAM_MOVE_ACCEL_MAX);
    control_state.move_decel = viewport_state
        .freecam_move_decel
        .clamp(FREECAM_MOVE_DECEL_MIN, FREECAM_MOVE_DECEL_MAX);
    control_state.speed_step = viewport_state
        .freecam_speed_step
        .clamp(FREECAM_SPEED_STEP_MIN, FREECAM_SPEED_STEP_MAX);
    control_state.speed_min = viewport_state
        .freecam_speed_min
        .clamp(FREECAM_SPEED_MIN_MIN, FREECAM_SPEED_MIN_MAX);
    control_state.speed_max = viewport_state
        .freecam_speed_max
        .clamp(FREECAM_SPEED_MAX_MIN, FREECAM_SPEED_MAX_MAX);
    if control_state.speed_min > control_state.speed_max {
        std::mem::swap(&mut control_state.speed_min, &mut control_state.speed_max);
    }
    control_state.boost_multiplier = viewport_state
        .freecam_boost_multiplier
        .clamp(FREECAM_BOOST_MULTIPLIER_MIN, FREECAM_BOOST_MULTIPLIER_MAX);
    let configured_orbit_distance = viewport_state
        .orbit_distance
        .clamp(FREECAM_ORBIT_DISTANCE_MIN, FREECAM_ORBIT_DISTANCE_MAX);
    let configured_orbit_sensitivity = viewport_state.freecam_orbit_pan_sensitivity.clamp(
        FREECAM_ORBIT_PAN_SENSITIVITY_MIN,
        FREECAM_ORBIT_PAN_SENSITIVITY_MAX,
    );
    let configured_pan_sensitivity = viewport_state
        .freecam_pan_sensitivity
        .clamp(FREECAM_PAN_SENSITIVITY_MIN, FREECAM_PAN_SENSITIVITY_MAX);
    let orbit_distance_config_changed = !control_state.configured_orbit_distance.is_finite()
        || (control_state.configured_orbit_distance - configured_orbit_distance).abs() > 0.0001;
    if orbit_distance_config_changed {
        control_state.configured_orbit_distance = configured_orbit_distance;
        control_state.middle_orbit_distance = configured_orbit_distance;
    }
    if !control_state.speed.is_finite() || control_state.speed <= 0.0 {
        control_state.speed = control_state.speed_min.max(1.0);
    }
    control_state.speed = control_state
        .speed
        .clamp(control_state.speed_min, control_state.speed_max);

    let editor_camera = ensure_retained_editor_camera(world);
    let gameplay_camera = resolve_gameplay_camera(world, editor_camera);
    let active_camera_entity = target_surface.and_then(|surface| match surface.mode {
        ViewportPaneMode::Edit => Some(editor_camera),
        ViewportPaneMode::Play => match world_state {
            WorldState::Play => gameplay_camera.or(Some(editor_camera)),
            WorldState::Edit => None,
        },
    });
    let orbit_selected_focus = if viewport_state.orbit_selected_entity {
        world
            .get_resource::<InspectorSelectedEntityResource>()
            .and_then(|selection| selection.0)
            .and_then(|entity| world.get::<BevyTransform>(entity).map(|t| t.0.position))
    } else {
        None
    };

    let can_control_active_camera = active_camera_entity.is_some();
    let allow_viewport_look_input = can_control_active_camera
        && (control_state.is_looking
            || control_state.is_middle_looking
            || pointer_over_active_viewport
            || (!wants_pointer && pointer_in_viewport));

    const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const BOOST_AMOUNT: f32 = 1.15;

    let mut yaw_delta = 0.0;
    let mut pitch_delta = 0.0;
    let mut middle_pan_delta = glam::DVec2::ZERO;
    let mut started_middle_look = false;
    let was_looking = control_state.is_looking;

    let gizmo_blocking = world
        .get_resource::<RenderGizmoState>()
        .is_some_and(|state| state.0.active_axis != GizmoAxis::None);

    let can_pointer_look = !gizmo_blocking
        && allow_viewport_look_input
        && (pointer_in_viewport
            || pointer_over_active_viewport
            || control_state.is_looking
            || control_state.is_middle_looking);
    let next_look_delta = |state: &mut RetainedViewportControlState| -> glam::DVec2 {
        let mut cursor_delta = raw_mouse_delta;
        if cursor_delta.length_squared() <= f64::EPSILON {
            cursor_delta = cursor_position - state.last_cursor_position;
        }
        state.last_cursor_position = cursor_position;
        let cursor_delta = if cursor_delta.length_squared() <= 0.25 {
            glam::DVec2::ZERO
        } else {
            cursor_delta.clamp_length_max(320.0)
        };
        if state.look_smoothing > 0.0 {
            let smoothing_seconds = state.look_smoothing.max(0.0001) as f64;
            let alpha = (1.0 - (-(dt as f64) / smoothing_seconds).exp()).clamp(0.0, 1.0);
            let smoothed_cursor_delta = state.smoothed_cursor_delta;
            state.smoothed_cursor_delta += (cursor_delta - smoothed_cursor_delta) * alpha;
        } else {
            state.smoothed_cursor_delta = cursor_delta;
        }
        if state.smoothed_cursor_delta.length_squared() <= 0.01 {
            glam::DVec2::ZERO
        } else {
            state.smoothed_cursor_delta
        }
    };

    if can_pointer_look && right_mouse_down {
        if !control_state.is_looking {
            control_state.look_start_cursor_position = Some(cursor_position);
            control_state.last_cursor_position = cursor_position;
            control_state.smoothed_cursor_delta = glam::DVec2::ZERO;
            control_state.look_surface = target_surface;
            control_state.is_looking = true;
            control_state.is_middle_looking = false;
        } else {
            let cursor_delta = next_look_delta(&mut control_state);
            yaw_delta -= cursor_delta.x as f32 * control_state.sensitivity / 100.0;
            pitch_delta += cursor_delta.y as f32 * control_state.sensitivity / 100.0;
        }
    } else {
        control_state.is_looking = false;
        if !control_state.is_middle_looking {
            control_state.smoothed_cursor_delta = glam::DVec2::ZERO;
        }
    }

    if can_pointer_look && middle_mouse_down && !right_mouse_down {
        if !control_state.is_middle_looking {
            control_state.is_middle_looking = true;
            control_state.last_cursor_position = cursor_position;
            control_state.smoothed_cursor_delta = glam::DVec2::ZERO;
            control_state.middle_orbit_focus = orbit_selected_focus;
            control_state.look_surface = target_surface;
            started_middle_look = true;
        } else {
            let cursor_delta = next_look_delta(&mut control_state);
            if shift_active {
                middle_pan_delta = cursor_delta;
            } else {
                yaw_delta -= cursor_delta.x as f32 * configured_orbit_sensitivity;
                pitch_delta += cursor_delta.y as f32 * configured_orbit_sensitivity;
            }
        }
    } else {
        control_state.is_middle_looking = false;
        if !control_state.is_looking {
            control_state.smoothed_cursor_delta = glam::DVec2::ZERO;
        }
    }

    if !right_mouse_down && !middle_mouse_down {
        control_state.look_surface = hovered_surface;
    }

    if gizmo_blocking {
        control_state.is_looking = false;
        control_state.is_middle_looking = false;
        control_state.look_surface = None;
        control_state.smoothed_cursor_delta = glam::DVec2::ZERO;
        control_state.move_velocity = glam::Vec3::ZERO;
    }
    let restore_cursor_position =
        if was_looking && !control_state.is_looking && !control_state.is_middle_looking {
            control_state.look_start_cursor_position.take()
        } else {
            None
        };
    let keyboard_active = control_state.is_looking && !gizmo_blocking;
    if !wants_pointer && keyboard_active && pointer_in_viewport {
        control_state.speed += wheel_delta * control_state.speed_step;
    }

    let mut orbit_distance_input_changed = orbit_distance_config_changed;
    let orbit_distance_before_input = if control_state.middle_orbit_distance.is_finite() {
        control_state.middle_orbit_distance
    } else {
        control_state.configured_orbit_distance
    };
    let apply_orbit_zoom = |distance: &mut f32, wheel: f32| -> bool {
        if wheel.abs() <= f32::EPSILON {
            return false;
        }
        let zoom_factor = (1.0 - wheel * 0.12).clamp(0.1, 10.0);
        *distance *= zoom_factor;
        true
    };
    if !wants_pointer && control_state.is_middle_looking && pointer_in_viewport {
        if apply_orbit_zoom(&mut control_state.middle_orbit_distance, wheel_delta) {
            orbit_distance_input_changed = true;
        }
    }
    if !wants_pointer
        && !control_state.is_looking
        && !control_state.is_middle_looking
        && pointer_in_viewport
    {
        if apply_orbit_zoom(&mut control_state.middle_orbit_distance, wheel_delta) {
            orbit_distance_input_changed = true;
        }
    }

    control_state.speed = control_state
        .speed
        .clamp(control_state.speed_min, control_state.speed_max);
    if !control_state.middle_orbit_distance.is_finite() {
        control_state.middle_orbit_distance = control_state.configured_orbit_distance;
    }
    control_state.middle_orbit_distance = control_state
        .middle_orbit_distance
        .clamp(FREECAM_ORBIT_DISTANCE_MIN, FREECAM_ORBIT_DISTANCE_MAX);
    let mut speed = control_state.speed;
    let boost_active = keyboard_active && shift_active;
    if boost_active {
        speed *= control_state.boost_multiplier;
    }

    if let Some(camera_entity) = active_camera_entity {
        let mut camera_query = world.query::<(Entity, &mut BevyTransform, &mut BevyCamera)>();
        if let Ok((entity, mut transform, mut camera)) = camera_query.get_mut(world, camera_entity)
        {
            if control_state.active_entity != Some(entity) {
                control_state.active_entity = Some(entity);
                control_state.current_fov_multiplier = 1.0;
                control_state.is_looking = false;
                control_state.is_middle_looking = false;
                control_state.look_start_cursor_position = None;
                control_state.last_cursor_position = cursor_position;
                control_state.smoothed_cursor_delta = glam::DVec2::ZERO;
                control_state.move_velocity = glam::Vec3::ZERO;
                control_state.middle_orbit_focus = None;
            }

            let transform = &mut transform.0;
            let camera = &mut camera.0;
            let previous_forward = transform.forward().normalize_or_zero();
            let middle_orbit_active = control_state.is_middle_looking && !control_state.is_looking;
            let middle_pan_active = middle_orbit_active && shift_active;

            let (mut yaw, mut pitch) = extract_yaw_pitch(transform.rotation);
            yaw += yaw_delta;
            pitch += pitch_delta;
            pitch = pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);

            let yaw_rot = glam::Quat::from_axis_angle(glam::Vec3::Y, yaw);
            let pitch_rot = glam::Quat::from_axis_angle(glam::Vec3::X, pitch);
            let orientation = yaw_rot * pitch_rot;
            transform.rotation = orientation;

            let forward = orientation * glam::Vec3::Z;
            let right = orientation * -glam::Vec3::X;

            if !middle_orbit_active {
                let fallback_focus = control_state
                    .middle_orbit_focus
                    .filter(|focus| focus.is_finite())
                    .unwrap_or(transform.position + forward * orbit_distance_before_input);
                let focus = orbit_selected_focus.unwrap_or(fallback_focus);
                control_state.middle_orbit_focus =
                    if focus.is_finite() { Some(focus) } else { None };
            }

            if orbit_distance_input_changed && !control_state.is_looking && !middle_orbit_active {
                if let Some(orbit_focus) = control_state
                    .middle_orbit_focus
                    .filter(|focus| focus.is_finite())
                {
                    let orbit_distance = control_state.middle_orbit_distance;
                    transform.position = orbit_focus - forward * orbit_distance;
                    control_state.move_velocity = glam::Vec3::ZERO;
                }
            }

            if middle_orbit_active {
                if viewport_state.orbit_selected_entity && !middle_pan_active {
                    if let Some(selected_focus) = orbit_selected_focus {
                        if started_middle_look {
                            let selected_distance = transform.position.distance(selected_focus);
                            if selected_distance.is_finite() {
                                control_state.middle_orbit_distance = selected_distance
                                    .clamp(FREECAM_ORBIT_DISTANCE_MIN, FREECAM_ORBIT_DISTANCE_MAX);
                            }
                        }
                        control_state.middle_orbit_focus = Some(selected_focus);
                    }
                }
                let anchor_forward = if previous_forward.length_squared() > 1.0e-6 {
                    previous_forward
                } else {
                    forward
                };
                let orbit_distance = control_state.middle_orbit_distance;
                let mut orbit_focus = control_state
                    .middle_orbit_focus
                    .filter(|focus| focus.is_finite())
                    .unwrap_or(transform.position + anchor_forward * orbit_distance);
                if middle_pan_active && middle_pan_delta.length_squared() > f64::EPSILON {
                    let up = orientation * glam::Vec3::Y;
                    let pan_speed = (orbit_distance * configured_pan_sensitivity).max(1.0e-4);
                    let pan = right * (-(middle_pan_delta.x as f32) * pan_speed)
                        + up * ((middle_pan_delta.y as f32) * pan_speed);
                    if pan.is_finite() {
                        orbit_focus += pan;
                    }
                }
                control_state.middle_orbit_focus = Some(orbit_focus);
                transform.position = orbit_focus - forward * orbit_distance;
                control_state.move_velocity = glam::Vec3::ZERO;
            }

            let mut move_input = glam::Vec3::ZERO;
            if keyboard_active {
                if move_forward {
                    move_input += forward;
                }
                if move_back {
                    move_input -= forward;
                }
                if move_left {
                    move_input -= right;
                }
                if move_right {
                    move_input += right;
                }
                if move_up {
                    move_input += glam::Vec3::Y;
                }
                if move_down {
                    move_input -= glam::Vec3::Y;
                }
            }

            let move_input_len = if middle_orbit_active {
                0.0
            } else {
                move_input.length()
            };
            let move_dir = if move_input_len > 1.0e-6 {
                move_input / move_input_len
            } else {
                glam::Vec3::ZERO
            };
            let move_scale = move_input_len.min(1.0);
            let target_velocity = move_dir * (speed * move_scale);
            let response = if move_input_len > 1.0e-6 {
                control_state.move_accel
            } else {
                control_state.move_decel
            };
            let alpha = (1.0 - (-(response * dt.max(0.0))).exp()).clamp(0.0, 1.0);
            let current_velocity = control_state.move_velocity;
            control_state.move_velocity =
                current_velocity + (target_velocity - current_velocity) * alpha;
            if !control_state.move_velocity.is_finite() {
                control_state.move_velocity = glam::Vec3::ZERO;
            }
            if move_input_len <= 1.0e-6 && control_state.move_velocity.length_squared() < 1.0e-6 {
                control_state.move_velocity = glam::Vec3::ZERO;
            }
            transform.position += control_state.move_velocity * dt;

            let target_multiplier = if boost_active { BOOST_AMOUNT } else { 1.0 };
            let safe_multiplier = control_state
                .current_fov_multiplier
                .clamp(0.01, BOOST_AMOUNT);
            let base_fov = camera.fov_y_rad / safe_multiplier;
            let t = 1.0 - (-control_state.fov_lerp_speed * dt).exp();
            control_state.current_fov_multiplier +=
                (target_multiplier - control_state.current_fov_multiplier) * t;
            camera.fov_y_rad = base_fov * control_state.current_fov_multiplier;
        } else {
            control_state.active_entity = None;
            control_state.move_velocity = glam::Vec3::ZERO;
        }
    } else {
        control_state.active_entity = None;
        control_state.move_velocity = glam::Vec3::ZERO;
    }

    if let (Some(surface), Some(mut states)) = (
        target_surface,
        world.get_resource_mut::<EditorRetainedViewportStates>(),
    ) {
        let state = states.state_for_mut(surface.tab_id);
        state.orbit_distance = control_state
            .middle_orbit_distance
            .clamp(FREECAM_ORBIT_DISTANCE_MIN, FREECAM_ORBIT_DISTANCE_MAX);
        state.sanitize();
    }

    control_state.last_cursor_position = cursor_position;
    sync_retained_cursor_control(
        &control_state,
        &mut cursor_control_state,
        world.get_resource::<BevyRuntimeCursorState>(),
        restore_cursor_position,
    );
    world.insert_resource(control_state);
    world.insert_resource(cursor_control_state);
}

pub(crate) fn can_control_surface(
    surface: ViewportPaneSurfaceKey,
    world_state: WorldState,
) -> bool {
    !matches!(surface.mode, ViewportPaneMode::Play) || world_state == WorldState::Play
}

fn sync_retained_cursor_control(
    control_state: &RetainedViewportControlState,
    cursor_control_state: &mut RetainedCursorControlState,
    runtime_cursor_state: Option<&BevyRuntimeCursorState>,
    restore_cursor_position: Option<glam::DVec2>,
) {
    cursor_control_state.freecam_capture_active =
        control_state.is_looking || control_state.is_middle_looking;

    if let Some(runtime_cursor_state) = runtime_cursor_state {
        runtime_cursor_state
            .0
            .set(cursor_control_state.effective_policy());
        if !control_state.is_looking
            && !control_state.is_middle_looking
            && let Some(restore_cursor_position) = restore_cursor_position
        {
            runtime_cursor_state
                .0
                .request_warp(restore_cursor_position.x, restore_cursor_position.y);
        }
    }
}

pub(crate) fn surface_under_pointer(
    surfaces: &HashMap<ViewportPaneSurfaceKey, UiRect>,
    pointer: glam::Vec2,
) -> Option<ViewportPaneSurfaceKey> {
    surfaces.iter().find_map(|(key, rect)| {
        (pointer.x >= rect.x
            && pointer.x <= rect.x + rect.width
            && pointer.y >= rect.y
            && pointer.y <= rect.y + rect.height)
            .then_some(*key)
    })
}

fn extract_yaw_pitch(rot: glam::Quat) -> (f32, f32) {
    let forward = rot * glam::Vec3::Z;
    let yaw = forward.x.atan2(forward.z);
    let pitch = (-forward.y).asin();
    (yaw, pitch)
}

pub fn retained_viewport_requests_system(world: &mut World) {
    let viewport_surfaces = world
        .get_resource::<EditorRetainedPaneInteractionState>()
        .map(|state| state.viewport_surfaces.clone())
        .unwrap_or_default();

    let mut requests = Vec::new();
    if !viewport_surfaces.is_empty() {
        let editor_camera = ensure_retained_editor_camera(world);
        let viewport_states = world
            .get_resource::<EditorRetainedViewportStates>()
            .cloned()
            .unwrap_or_default();
        let world_state = world
            .get_resource::<EditorSceneState>()
            .map(|scene| scene.world_state)
            .unwrap_or(WorldState::Edit);
        let gameplay_camera = resolve_gameplay_camera(world, editor_camera);
        let preview_camera = resolve_preview_camera_entity(world, editor_camera);

        for (surface_key, surface) in viewport_surfaces {
            let viewport_state = viewport_states.state_for(surface_key.tab_id);
            let should_render_main = match surface_key.mode {
                ViewportPaneMode::Edit => true,
                ViewportPaneMode::Play => world_state == WorldState::Play,
            };
            if !should_render_main {
                continue;
            }
            let surface_size = [
                surface.width.max(1.0).round() as u32,
                surface.height.max(1.0).round() as u32,
            ];
            let target_size = viewport_state.resolution.target_size(surface_size);
            let viewport_id = retained_viewport_id(surface_key.mode, surface_key.tab_id);
            let graph_template = viewport_state
                .graph_template
                .clone()
                .or_else(|| Some("debug-graph".to_string()));
            let gizmo_options = RenderViewportGizmoOptions {
                show_gizmos: world_state == WorldState::Edit || viewport_state.gizmos_in_play,
                show_camera_gizmos: viewport_state.show_camera_gizmos,
                show_directional_light_gizmos: viewport_state.show_directional_light_gizmos,
                show_point_light_gizmos: viewport_state.show_point_light_gizmos,
                show_spot_light_gizmos: viewport_state.show_spot_light_gizmos,
            };

            let main_camera_entity = match surface_key.mode {
                ViewportPaneMode::Edit => editor_camera,
                ViewportPaneMode::Play => {
                    if world_state == WorldState::Play {
                        gameplay_camera.unwrap_or(editor_camera)
                    } else {
                        editor_camera
                    }
                }
            };

            if let Some((camera_component, camera_transform)) =
                camera_snapshot_for_entity(world, main_camera_entity)
            {
                requests.push(RenderViewportRequest {
                    id: viewport_id,
                    camera_transform,
                    camera_component,
                    texture_handle: viewport_id,
                    texture_is_managed: false,
                    target_size,
                    temporal_history: true,
                    immediate_resize: false,
                    graph_template: graph_template.clone(),
                    gizmo_options,
                });
            }

            if matches!(surface_key.mode, ViewportPaneMode::Edit)
                && let Some(preview_camera_entity) = preview_camera
                && preview_camera_entity != main_camera_entity
                && let Some((camera_component, camera_transform)) =
                    camera_snapshot_for_entity(world, preview_camera_entity)
            {
                let aspect = if camera_component.aspect_ratio.is_finite()
                    && camera_component.aspect_ratio > 0.01
                {
                    camera_component.aspect_ratio
                } else {
                    (surface.width / surface.height.max(1.0)).max(0.1)
                };
                let preview_width = (surface.width * viewport_state.preview_width_norm)
                    .clamp(72.0, surface.width.max(72.0));
                let preview_height = (preview_width / aspect).clamp(48.0, surface.height.max(48.0));
                let preview_target = [
                    preview_width.max(1.0).round() as u32,
                    preview_height.max(1.0).round() as u32,
                ];
                let preview_viewport_id = retained_viewport_preview_id(surface_key.tab_id);
                requests.push(RenderViewportRequest {
                    id: preview_viewport_id,
                    camera_transform,
                    camera_component,
                    texture_handle: preview_viewport_id,
                    texture_is_managed: false,
                    target_size: preview_target,
                    temporal_history: false,
                    immediate_resize: false,
                    graph_template: graph_template,
                    gizmo_options,
                });
            }
        }
    }

    if let Some(mut viewport_requests) = world.get_resource_mut::<RenderViewportRequests>() {
        viewport_requests.0 = requests;
    } else {
        world.insert_resource(RenderViewportRequests(requests));
    }
}
