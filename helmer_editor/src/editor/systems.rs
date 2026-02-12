use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use bevy_ecs::prelude::{Changed, Entity, Or, Query, Res, ResMut, With, World};
use bevy_ecs::{component::Component, name::Name};
use egui::{Id, Order, Pos2, Rect, Ui, Vec2};
use glam::{DVec2, Quat, Vec3};
use helmer::graphics::common::renderer::{RenderViewportGizmoOptions, RenderViewportRequest};
use helmer::graphics::render_graphs::template_for_graph;
use helmer::provided::components::{
    Camera, Light, MeshRenderer, PoseOverride, SkinnedMeshRenderer, Transform,
};
use helmer::runtime::asset_server::{Handle, Material, Mesh};
use helmer_becs::egui_integration::{
    EguiInputPassthrough, EguiResource, EguiWindowChrome, EguiWindowSpec,
};
use helmer_becs::physics::components::{ColliderShape, DynamicRigidBody, FixedCollider};
use helmer_becs::physics::physics_resource::PhysicsResource;
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::render_system::{
    RenderGraphResource, RenderSyncRequest, RenderViewportRequests,
};
use helmer_becs::systems::scene_system::{
    EntityParent, SceneChild, SceneRoot, SceneSpawnedChildren, build_default_animator,
};
use helmer_becs::{
    AudioBackendResource, BevyAnimator, BevyAudioEmitter, BevyAudioListener, BevyCamera,
    BevyEntityFollower, BevyInputManager, BevyLight, BevyLookAt, BevyMeshRenderer,
    BevyPoseOverride, BevyRuntimeCursorState, BevySkinnedMeshRenderer, BevySpline,
    BevySplineFollower, BevyTransform, BevyWrapper, DeltaTime, DraggedFile,
};
use walkdir::WalkDir;
use winit::{event::MouseButton, keyboard::KeyCode};

use crate::editor::{
    EditorCursorControlState, EditorLayout, EditorLayoutState, EditorPaneViewportState,
    EditorPlayCamera, EditorTimelineState, EditorViewportCamera, EditorViewportRuntime,
    EditorViewportState, FREECAM_SENSITIVITY_MAX, FREECAM_SENSITIVITY_MIN, FREECAM_SMOOTHING_MAX,
    FREECAM_SMOOTHING_MIN, LayoutDragEdges, LayoutDragMode, LayoutSaveRequest, NormalizedRect,
    PlayViewportKind, VIEWPORT_ID_EDITOR, VIEWPORT_ID_GAMEPLAY, VIEWPORT_ID_PREVIEW,
    ViewportRectPixels, activate_play_camera, activate_viewport_camera,
    activate_viewport_camera_for_pane,
    assets::{
        AssetBrowserState, EditorAssetCache, EditorAudio, EditorMesh, EditorSkinnedMesh,
        MeshSource, PrimitiveKind, SceneAssetPath, cached_scene_handle, scan_asset_entries,
    },
    capture_layout,
    commands::{AssetCreateKind, EditorCommand, EditorCommandQueue, SpawnKind},
    dynamic::DynamicComponents,
    ensure_play_camera, ensure_viewport_camera,
    gizmos::EditorGizmoState,
    layout_window_ids, mark_undo_clean,
    project::{
        EditorProject, create_project, default_animation_template, default_material_template,
        default_rust_script_template_full, default_scene_template, default_script_template_full,
        load_project, rust_script_manifest_template, rust_script_sdk_dependency_path,
        sanitize_rust_crate_name, save_recent_projects,
    },
    push_undo_snapshot, redo_action, reset_undo_history, save_layouts,
    scene::{
        AnimationClipData, EditorEntity, EditorRenderRefresh, EditorSceneState,
        PendingSceneChildAnimations, PendingSceneChildPoseOverrides, PendingSceneChildRenderer,
        PendingSkinnedMeshAsset, SceneChildRendererKind, WorldState,
        apply_animation_data_to_timeline, apply_custom_clips_to_animator,
        next_available_scene_path, normalize_path, pose_from_serialized, read_scene_document,
        reset_editor_scene, restore_scene_transforms_from_document, serialize_scene,
        spawn_default_camera, spawn_default_light, spawn_scene_from_document, write_scene_document,
    },
    scripting::{
        ScriptComponent, ScriptRegistry, ScriptRuntime, is_script_path, load_script_asset,
        script_registry_key_for_path,
    },
    set_play_camera, set_viewport_audio_listener_enabled,
    ui::{
        AssetDragState, EditorPaneManagerState, EditorPaneWorkspaceState, EditorUiState,
        EditorWorkspaceState, EntityDragState, InspectorPinnedEntityResource, MiddleDragUiState,
        close_editor_window, close_pane_workspace_window, draw_editor_window,
        draw_pane_manager_window, draw_pane_workspace_window, ensure_default_pane_workspace,
        spawn_play_viewport_pane,
    },
    undo_action,
    visual_scripting::default_visual_script_template_full,
    watch::configure_file_watcher,
};

pub fn editor_ui_system(world: &mut World) {
    if let Some(mut viewport_runtime) = world.get_resource_mut::<EditorViewportRuntime>() {
        viewport_runtime.begin_frame();
    }
    if let Some(mut passthrough) = world.get_resource_mut::<EguiInputPassthrough>() {
        *passthrough = EguiInputPassthrough::default();
    }
    crate::editor::drain_runtime_log_queue(world);
    crate::editor::sync_console_diagnostics(world);

    ensure_default_pane_workspace(world);

    let editor_windows = world
        .get_resource::<EditorWorkspaceState>()
        .map(|state| {
            state
                .windows
                .iter()
                .map(|window| (window.id, window.title.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let editor_tab_dragging = world
        .get_resource::<EditorWorkspaceState>()
        .is_some_and(|state| state.dragging.is_some());
    let pane_windows = world
        .get_resource::<EditorPaneWorkspaceState>()
        .map(|state| {
            state
                .windows
                .iter()
                .map(|window| (window.id.clone(), window.layout_managed))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let pane_tab_dragging = world
        .get_resource::<EditorPaneWorkspaceState>()
        .is_some_and(|state| state.dragging.is_some());
    let pane_manager_open = world
        .get_resource::<EditorPaneManagerState>()
        .map(|state| state.open)
        .unwrap_or(false);

    if let Some(mut workspace) = world.get_resource_mut::<EditorWorkspaceState>() {
        workspace.drop_handled = false;
    }
    if let Some(mut workspace) = world.get_resource_mut::<EditorPaneWorkspaceState>() {
        workspace.drop_handled = false;
    }

    let mut egui_res = world
        .get_resource_mut::<EguiResource>()
        .expect("EguiResource missing");

    egui_res.inspector_ui = false;
    egui_res.disable_window_drag = editor_tab_dragging || pane_tab_dragging;
    for (window_id, layout_managed) in pane_windows {
        let mut chrome = EguiWindowChrome::pane_dock();
        // detached pane windows should keep compact chrome, but still allow native title-bar drag
        if !layout_managed {
            chrome.disable_native_drag = false;
        }
        egui_res
            .window_chrome_overrides
            .insert(window_id.clone(), chrome);
        egui_res.window_order_overrides.insert(
            window_id.clone(),
            if layout_managed {
                Order::Middle
            } else {
                Order::Foreground
            },
        );
        let close_id = window_id.clone();
        egui_res.close_actions.insert(
            window_id.clone(),
            Box::new(move |world: &mut World| {
                close_pane_workspace_window(world, &close_id);
            }),
        );
        let pane_window_id = window_id.clone();
        egui_res.windows.push((
            Box::new(move |ui: &mut Ui, world: &mut World, _| {
                draw_pane_workspace_window(ui, world, &pane_window_id);
            }),
            EguiWindowSpec {
                id: window_id,
                title: String::new(),
            },
        ));
    }

    if pane_manager_open {
        egui_res.close_actions.insert(
            "Pane Manager".to_string(),
            Box::new(|world: &mut World| {
                if let Some(mut state) = world.get_resource_mut::<EditorPaneManagerState>() {
                    state.open = false;
                }
            }),
        );
        egui_res
            .window_order_overrides
            .insert("Pane Manager".to_string(), Order::Foreground);
        if let Some(screen_rect) = egui_res.last_screen_rect {
            let size = Vec2::new(280.0, 320.0);
            let rect = Rect::from_center_size(screen_rect.center(), size);
            egui_res
                .window_rect_overrides
                .insert("Pane Manager".to_string(), rect);
        }
        egui_res.windows.push((
            Box::new(|ui: &mut Ui, world: &mut World, _| {
                draw_pane_manager_window(ui, world);
            }),
            EguiWindowSpec {
                id: "Pane Manager".to_string(),
                title: "Pane Manager".to_string(),
            },
        ));
    }

    for (window_id, title) in editor_windows {
        let window_key = format!("editor_window_{}", window_id);
        let is_layout_window = layout_window_ids()
            .iter()
            .any(|layout_id| *layout_id == title.as_str());
        egui_res.window_order_overrides.insert(
            window_key.clone(),
            if is_layout_window {
                Order::Middle
            } else {
                Order::Foreground
            },
        );
        egui_res.close_actions.insert(
            window_key.clone(),
            Box::new(move |world: &mut World| {
                close_editor_window(world, window_id);
            }),
        );
        egui_res.windows.push((
            Box::new(move |ui: &mut Ui, world: &mut World, _| {
                draw_editor_window(ui, world, window_id);
            }),
            EguiWindowSpec {
                id: window_key,
                title,
            },
        ));
    }
}

pub fn editor_layout_apply_system(world: &mut World) {
    ensure_default_pane_workspace(world);

    let project_open = world
        .get_resource::<crate::editor::EditorProject>()
        .and_then(|project| project.root.as_ref())
        .is_some();
    let (screen_rect, pixels_per_point) = {
        let Some(egui_res) = world.get_resource::<EguiResource>() else {
            return;
        };
        let Some(result) = screen_rect_from_egui(egui_res) else {
            return;
        };
        result
    };

    let (
        layout,
        clear_active,
        activate_default,
        allow_layout_edit,
        apply_requested,
        layout_active,
        last_screen_rect,
        project_open_changed,
        active_name,
        last_active_layout,
        default_runtime_layout,
    ) = match world.get_resource::<EditorLayoutState>() {
        Some(state) => {
            let project_open_changed = state
                .last_project_open
                .map(|prev| prev != project_open)
                .unwrap_or(true);
            if !project_open {
                (
                    None,
                    false,
                    None,
                    state.allow_layout_edit,
                    state.apply_requested,
                    true,
                    state.last_screen_rect,
                    project_open_changed,
                    state.active.clone(),
                    state.last_active_layout.clone(),
                    state.default_runtime_layout.clone(),
                )
            } else {
                let mut clear_active = false;
                let mut activate_default = None;
                let layout = match state.active.as_ref() {
                    Some(name) => {
                        let layout = state.layouts.get(name).cloned();
                        if layout.is_none() {
                            clear_active = true;
                        }
                        layout
                    }
                    None => {
                        if project_open_changed {
                            if let Some(default) = state.layouts.get("Default") {
                                activate_default = Some(default.name.clone());
                                Some(default.clone())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                };
                let layout_active = state.active.is_some() || activate_default.is_some();
                (
                    layout,
                    clear_active,
                    activate_default,
                    state.allow_layout_edit,
                    state.apply_requested,
                    layout_active,
                    state.last_screen_rect,
                    project_open_changed,
                    state.active.clone(),
                    state.last_active_layout.clone(),
                    state.default_runtime_layout.clone(),
                )
            }
        }
        None => return,
    };

    let effective_active = activate_default.clone().or_else(|| active_name.clone());
    let active_is_default = effective_active.as_deref() == Some("Default");
    let mut layout = layout;
    if project_open && active_is_default {
        layout = default_runtime_layout
            .clone()
            .or_else(|| Some(crate::editor::default_layout()));
    }

    let has_layout = layout.is_some();
    let enforce_layout = !project_open || has_layout;
    let active_changed =
        activate_default.is_some() || active_name.as_deref() != last_active_layout.as_deref();
    let hard_apply = if !project_open {
        true
    } else if enforce_layout {
        apply_requested
            || activate_default.is_some()
            || active_changed
            || last_screen_rect.is_none()
            || project_open_changed
    } else {
        false
    };

    if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
        egui_res.layout_active = layout_active;
        egui_res.layout_allow_move = allow_layout_edit;
        egui_res.layout_force_positions = layout_active;
        egui_res.layout_resizing_window = None;

        if enforce_layout {
            if !project_open {
                let project_rect = centered_project_rect(screen_rect);
                let project_rect =
                    round_rect_to_pixels(project_rect, pixels_per_point, screen_rect);
                for id in layout_window_ids() {
                    let collapsed = *id != "Project";
                    if *id == "Project" {
                        egui_res
                            .window_rect_overrides
                            .insert((*id).to_string(), project_rect);
                        egui_res
                            .window_positions
                            .insert((*id).to_string(), project_rect.min);
                    }
                    egui_res
                        .window_collapsed_overrides
                        .insert((*id).to_string(), collapsed);
                }
                egui_res
                    .window_order_overrides
                    .insert("Project".to_string(), Order::Foreground);
            } else if let Some(layout) = layout {
                let rects = layout
                    .windows
                    .iter()
                    .map(|(id, window)| {
                        let rect = window.rect.to_rect(screen_rect);
                        let rect = round_rect_to_pixels(rect, pixels_per_point, screen_rect);
                        (id.clone(), rect)
                    })
                    .collect::<Vec<_>>();
                for (id, rect) in rects {
                    if let Some(window) = layout.windows.get(&id) {
                        let collapsed = window.collapsed;
                        egui_res.window_rect_overrides.insert(id.clone(), rect);
                        egui_res.window_positions.insert(id.clone(), rect.min);
                        if !allow_layout_edit || hard_apply {
                            egui_res.window_collapsed_overrides.insert(id, collapsed);
                        }
                    }
                }
            }
            egui_res.suppress_snap = true;
        }
    }

    if let Some(mut state) = world.get_resource_mut::<EditorLayoutState>() {
        if clear_active {
            state.active = None;
        }
        if let Some(name) = activate_default {
            state.active = Some(name);
        }
        state.apply_requested = false;
        state.last_screen_rect = Some(screen_rect);
        state.last_project_open = Some(project_open);
        state.last_active_layout = state.active.clone();
        state.layout_applied_this_frame = hard_apply;
        let active_name = state.active.clone();
        let active_is_default = active_name.as_deref() == Some("Default");
        if active_is_default {
            if active_changed || state.default_runtime_layout.is_none() {
                state.default_runtime_layout = Some(crate::editor::default_layout());
            }
        } else {
            state.default_runtime_layout = None;
        }
        if hard_apply && has_layout && !state.layout_verify_pending {
            state.layout_verify_pending = true;
            state.layout_verify_attempts = 3;
        }
        if hard_apply {
            state.layout_dragging_window = None;
            state.layout_drag_mode = LayoutDragMode::None;
        }
    }
}

pub fn editor_layout_save_system(world: &mut World) {
    let (request, active_name) = match world.get_resource_mut::<EditorLayoutState>() {
        Some(mut state) => (state.save_request.take(), state.active.clone()),
        None => return,
    };

    let Some(request) = request else {
        return;
    };

    let Some(egui_res) = world.get_resource::<EguiResource>() else {
        return;
    };
    let Some((screen_rect, _)) = screen_rect_from_egui(egui_res) else {
        if let Some(mut state) = world.get_resource_mut::<EditorLayoutState>() {
            state.save_request = Some(request);
        }
        return;
    };

    let layout_name = match request {
        LayoutSaveRequest::SaveActive => active_name,
        LayoutSaveRequest::SaveAs(name) => Some(name),
    };

    let Some(layout_name) = layout_name else {
        if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
            state.status = Some("No active layout to save".to_string());
        }
        return;
    };

    let layout = capture_layout(
        layout_name.clone(),
        &egui_res.window_rects,
        &egui_res.window_collapsed,
        screen_rect,
    );
    if layout.windows.is_empty() {
        if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
            state.status = Some("No visible windows to save".to_string());
        }
        return;
    }

    let save_error = world.resource_scope::<EditorLayoutState, _>(|_world, mut state| {
        state.layouts.insert(layout_name.clone(), layout);
        state.active = Some(layout_name.clone());
        state.last_screen_rect = Some(screen_rect);
        state.apply_requested = false;
        save_layouts(&state).err()
    });

    if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
        match save_error {
            Some(err) => {
                state.status = Some(format!("Failed to save layout: {}", err));
            }
            None => {
                state.status = Some(format!("Layout saved: {}", layout_name));
            }
        }
    }
}

pub fn editor_layout_update_system(world: &mut World) {
    let project_open = world
        .get_resource::<crate::editor::EditorProject>()
        .and_then(|project| project.root.as_ref())
        .is_some();
    if !project_open {
        return;
    }

    let (
        ctx,
        screen_rect,
        pixels_per_point,
        window_rects,
        window_content_rects,
        window_collapsed,
        pointer_down,
        pointer_pressed,
        pointer_released,
        pointer_pos,
        grab_radius,
        top_layer_id,
    ) = {
        let Some(egui_res) = world.get_resource::<EguiResource>() else {
            return;
        };
        let Some((screen_rect, pixels_per_point)) = screen_rect_from_egui(egui_res) else {
            return;
        };
        let window_rects = egui_res.window_rects.clone();
        let window_content_rects = egui_res.window_content_rects.clone();
        let window_collapsed = egui_res.window_collapsed.clone();
        let (pointer_down, pointer_pressed, pointer_released, pointer_pos) =
            egui_res.ctx.input(|input| {
                (
                    input.pointer.button_down(egui::PointerButton::Primary),
                    input.pointer.button_pressed(egui::PointerButton::Primary),
                    input.pointer.button_released(egui::PointerButton::Primary),
                    input.pointer.interact_pos(),
                )
            });
        let top_layer_id = pointer_pos.and_then(|pos| egui_res.ctx.layer_id_at(pos));
        let style = egui_res.ctx.style();
        let grab_radius = style
            .interaction
            .resize_grab_radius_side
            .max(style.interaction.resize_grab_radius_corner);
        let ctx = egui_res.ctx.clone();
        (
            ctx,
            screen_rect,
            pixels_per_point,
            window_rects,
            window_content_rects,
            window_collapsed,
            pointer_down,
            pointer_pressed,
            pointer_released,
            pointer_pos,
            grab_radius,
            top_layer_id,
        )
    };
    let gizmo_drag_active = world
        .get_resource::<EditorGizmoState>()
        .map(|state| state.is_drag_active())
        .unwrap_or(false);
    let asset_drag_active = world
        .get_resource::<AssetDragState>()
        .map(|state| state.active)
        .unwrap_or(false);
    let entity_drag_active = world
        .get_resource::<EntityDragState>()
        .map(|state| state.active)
        .unwrap_or(false);
    let middle_drag_active = world
        .get_resource::<MiddleDragUiState>()
        .map(|state| state.active)
        .unwrap_or(false);
    let pane_tab_drag_active = world
        .get_resource::<EditorPaneWorkspaceState>()
        .is_some_and(|workspace| workspace.dragging.is_some());

    let mut state = world
        .get_resource_mut::<EditorLayoutState>()
        .expect("EditorLayoutState missing");
    let live_reflow = state.live_reflow;
    let allow_layout_edit = state.allow_layout_edit;

    if state.layout_verify_pending {
        let attempts_left = state.layout_verify_attempts;
        let active_name = state.active.clone();
        if attempts_left == 0 {
            state.layout_verify_pending = false;
        } else if let Some(active_name) = active_name {
            let layout = if active_name == "Default" {
                if state.default_runtime_layout.is_none() {
                    state.default_runtime_layout = Some(crate::editor::default_layout());
                }
                state.default_runtime_layout.as_ref()
            } else {
                state.layouts.get(&active_name)
            };
            if let Some(layout) = layout {
                if layout_matches_window_rects(
                    layout,
                    &window_rects,
                    &window_collapsed,
                    screen_rect,
                    pixels_per_point,
                ) {
                    state.layout_verify_pending = false;
                } else {
                    state.layout_verify_attempts = attempts_left.saturating_sub(1);
                    if state.layout_verify_attempts == 0 {
                        state.layout_verify_pending = false;
                    } else {
                        state.apply_requested = true;
                    }
                }
            } else {
                state.layout_verify_pending = false;
            }
        } else {
            state.layout_verify_pending = false;
        }
    }

    let Some(active_name) = state.active.clone() else {
        state.layout_applied_this_frame = false;
        state.layout_dragging_window = None;
        state.layout_drag_mode = LayoutDragMode::None;
        state.layout_drag_start_pos = None;
        state.layout_drag_start_rect = None;
        state.layout_drag_start_layout = None;
        state.layout_drag_edges = LayoutDragEdges::default();
        state.last_screen_rect = Some(screen_rect);
        return;
    };

    let active_is_default = active_name == "Default";
    if state.layout_applied_this_frame {
        state.layout_applied_this_frame = false;
        state.layout_dragging_window = None;
        state.layout_drag_mode = LayoutDragMode::None;
        state.layout_drag_start_pos = None;
        state.layout_drag_start_rect = None;
        state.layout_drag_start_layout = None;
        state.layout_drag_edges = LayoutDragEdges::default();
        state.last_screen_rect = Some(screen_rect);
        return;
    }

    if screen_rect_changed(state.last_screen_rect, screen_rect, pixels_per_point) {
        state.last_screen_rect = Some(screen_rect);
        state.layout_dragging_window = None;
        state.layout_drag_mode = LayoutDragMode::None;
        state.layout_drag_start_pos = None;
        state.layout_drag_start_rect = None;
        state.layout_drag_start_layout = None;
        state.layout_drag_edges = LayoutDragEdges::default();
        return;
    }

    let layout = if active_is_default {
        if state.default_runtime_layout.is_none() {
            state.default_runtime_layout = Some(crate::editor::default_layout());
        }
        state.default_runtime_layout.clone()
    } else {
        state.layouts.get(&active_name).cloned()
    };
    let Some(layout) = layout else {
        state.last_screen_rect = Some(screen_rect);
        return;
    };

    let layout_rects = layout_rects_for_screen(&layout, screen_rect, pixels_per_point);
    if pane_tab_drag_active {
        state.layout_dragging_window = None;
        state.layout_drag_mode = LayoutDragMode::None;
        state.layout_drag_start_pos = None;
        state.layout_drag_start_rect = None;
        state.layout_drag_start_layout = None;
        state.layout_drag_edges = LayoutDragEdges::default();
        state.last_screen_rect = Some(screen_rect);
        return;
    }

    let external_drag_active = ctx.dragged_id().is_some()
        || gizmo_drag_active
        || asset_drag_active
        || entity_drag_active
        || middle_drag_active;

    if state.layout_dragging_window.is_none() && external_drag_active {
        state.last_screen_rect = Some(screen_rect);
        return;
    }

    if allow_layout_edit {
        let target_layout = if active_is_default {
            state.default_runtime_layout.as_mut()
        } else {
            state.layouts.get_mut(&active_name)
        };
        if let Some(layout) = target_layout {
            let mut updated = false;
            for (id, window) in layout.windows.iter_mut() {
                if let Some(collapsed) = window_collapsed.get(id) {
                    if window.collapsed != *collapsed {
                        window.collapsed = *collapsed;
                        updated = true;
                    }
                }
            }
            if updated {
                state.last_screen_rect = Some(screen_rect);
            }
        }
    }

    let preferred_window_id = top_layer_id.and_then(|layer| {
        layout_window_ids().iter().find_map(|id| {
            if Id::new(*id) == layer.id {
                Some((*id).to_string())
            } else {
                None
            }
        })
    });
    let pointer_blocked_by_non_layout = top_layer_id.is_some() && preferred_window_id.is_none();

    if pointer_pressed
        && state.layout_dragging_window.is_none()
        && allow_layout_edit
        && !pointer_blocked_by_non_layout
    {
        if let Some(pos) = pointer_pos {
            if let Some((id, mode, edges)) = pick_layout_drag_target(
                &ctx,
                &window_rects,
                &window_content_rects,
                &window_collapsed,
                pos,
                grab_radius,
                preferred_window_id.as_deref(),
            ) {
                state.layout_dragging_window = Some(id.clone());
                state.layout_drag_mode = mode;
                state.layout_drag_start_pos = Some(pos);
                state.layout_drag_start_layout = Some(layout_rects.clone());
                state.layout_drag_start_rect = layout_rects.get(&id).copied();
                state.layout_drag_edges = edges;
            }
        }
    }

    let Some(drag_id) = state.layout_dragging_window.clone() else {
        state.last_screen_rect = Some(screen_rect);
        return;
    };

    let drag_mode = state.layout_drag_mode;
    let drag_edges = state.layout_drag_edges;
    let start_layout = state
        .layout_drag_start_layout
        .clone()
        .unwrap_or_else(|| layout_rects.clone());
    let Some(start_rect) = state
        .layout_drag_start_rect
        .or_else(|| start_layout.get(&drag_id).copied())
    else {
        state.layout_dragging_window = None;
        state.layout_drag_mode = LayoutDragMode::None;
        state.layout_drag_start_pos = None;
        state.layout_drag_start_rect = None;
        state.layout_drag_start_layout = None;
        state.layout_drag_edges = LayoutDragEdges::default();
        state.last_screen_rect = Some(screen_rect);
        return;
    };

    let start_pos = state.layout_drag_start_pos.unwrap_or(start_rect.min);
    let pos = pointer_pos.unwrap_or(start_pos);
    let delta = pos - start_pos;
    let finalize = pointer_released || !pointer_down;

    let mut updated_rects = start_layout.clone();
    match drag_mode {
        LayoutDragMode::Move => {
            let moved = translate_rect(start_rect, delta, screen_rect);
            updated_rects.insert(drag_id.clone(), moved);
        }
        LayoutDragMode::Resize => {
            let resized = resize_rect(start_rect, delta, drag_edges, screen_rect);
            updated_rects.insert(drag_id.clone(), resized);
            if live_reflow || finalize {
                let threshold = edge_threshold(pixels_per_point);
                apply_reflow_edges(
                    &start_layout,
                    &mut updated_rects,
                    &drag_id,
                    start_rect,
                    resized,
                    drag_edges,
                    threshold,
                );
            }
        }
        LayoutDragMode::None => {}
    }

    for rect in updated_rects.values_mut() {
        clamp_rect_to_screen(rect, screen_rect);
        *rect = round_rect_to_pixels(*rect, pixels_per_point, screen_rect);
    }

    let target_layout = if active_is_default {
        state.default_runtime_layout.as_mut()
    } else {
        state.layouts.get_mut(&active_name)
    };
    if let Some(layout) = target_layout {
        for (id, window) in layout.windows.iter_mut() {
            if let Some(rect) = updated_rects.get(id) {
                window.rect = NormalizedRect::from_rect(*rect, screen_rect);
            }
            if let Some(collapsed) = window_collapsed.get(id) {
                window.collapsed = *collapsed;
            }
        }
    }

    state.last_screen_rect = Some(screen_rect);

    if finalize {
        state.layout_dragging_window = None;
        state.layout_drag_mode = LayoutDragMode::None;
        state.layout_drag_start_pos = None;
        state.layout_drag_start_rect = None;
        state.layout_drag_start_layout = None;
        state.layout_drag_edges = LayoutDragEdges::default();
    }
}

fn screen_rect_from_egui(egui_res: &EguiResource) -> Option<(Rect, f32)> {
    let pixels_per_point = egui_res
        .render_data
        .as_ref()
        .map(|data| data.screen_descriptor.pixels_per_point)
        .unwrap_or(1.0);
    let pixels_per_point = if pixels_per_point > 0.0 {
        pixels_per_point
    } else {
        1.0
    };
    if let Some(rect) = egui_res.last_screen_rect {
        return Some((rect, pixels_per_point));
    }
    let render_data = egui_res.render_data.as_ref()?;
    let size = render_data.screen_descriptor.size_in_pixels;
    let width = size[0] as f32 / pixels_per_point;
    let height = size[1] as f32 / pixels_per_point;
    Some((
        Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(width, height)),
        pixels_per_point,
    ))
}

fn round_rect_to_pixels(rect: Rect, pixels_per_point: f32, screen_rect: Rect) -> Rect {
    if pixels_per_point <= 0.0 {
        return rect;
    }
    let inv = 1.0 / pixels_per_point;
    let mut min_x = (rect.min.x * pixels_per_point).round() * inv;
    let mut min_y = (rect.min.y * pixels_per_point).round() * inv;
    let mut max_x = (rect.max.x * pixels_per_point).round() * inv;
    let mut max_y = (rect.max.y * pixels_per_point).round() * inv;
    if max_x < min_x {
        std::mem::swap(&mut min_x, &mut max_x);
    }
    if max_y < min_y {
        std::mem::swap(&mut min_y, &mut max_y);
    }
    let min_x = min_x.max(screen_rect.min.x);
    let min_y = min_y.max(screen_rect.min.y);
    let max_x = max_x.min(screen_rect.max.x);
    let max_y = max_y.min(screen_rect.max.y);
    let min_x = min_x.min(max_x);
    let min_y = min_y.min(max_y);
    Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, max_y))
}

fn layout_rects_for_screen(
    layout: &EditorLayout,
    screen_rect: Rect,
    pixels_per_point: f32,
) -> HashMap<String, Rect> {
    let mut rects = HashMap::new();
    for (id, window) in layout.windows.iter() {
        let rect = window.rect.to_rect(screen_rect);
        rects.insert(id.clone(), rect);
    }

    if pixels_per_point <= 0.0 || rects.is_empty() {
        return rects;
    }

    let epsilon = 0.51_f32;
    let mut x_edges = Vec::with_capacity(rects.len() * 2 + 2);
    let mut y_edges = Vec::with_capacity(rects.len() * 2 + 2);
    for rect in rects.values() {
        x_edges.push(rect.min.x * pixels_per_point);
        x_edges.push(rect.max.x * pixels_per_point);
        y_edges.push(rect.min.y * pixels_per_point);
        y_edges.push(rect.max.y * pixels_per_point);
    }
    x_edges.push(screen_rect.min.x * pixels_per_point);
    x_edges.push(screen_rect.max.x * pixels_per_point);
    y_edges.push(screen_rect.min.y * pixels_per_point);
    y_edges.push(screen_rect.max.y * pixels_per_point);

    let x_clusters = build_edge_clusters(x_edges, epsilon);
    let y_clusters = build_edge_clusters(y_edges, epsilon);

    for rect in rects.values_mut() {
        let min_x = snap_edge(rect.min.x * pixels_per_point, &x_clusters, epsilon);
        let max_x = snap_edge(rect.max.x * pixels_per_point, &x_clusters, epsilon);
        let min_y = snap_edge(rect.min.y * pixels_per_point, &y_clusters, epsilon);
        let max_y = snap_edge(rect.max.y * pixels_per_point, &y_clusters, epsilon);

        let mut min_x = (min_x / pixels_per_point).max(screen_rect.min.x);
        let mut max_x = (max_x / pixels_per_point).min(screen_rect.max.x);
        let mut min_y = (min_y / pixels_per_point).max(screen_rect.min.y);
        let mut max_y = (max_y / pixels_per_point).min(screen_rect.max.y);
        if max_x < min_x {
            std::mem::swap(&mut min_x, &mut max_x);
        }
        if max_y < min_y {
            std::mem::swap(&mut min_y, &mut max_y);
        }
        *rect = Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, max_y));
    }

    rects
}

fn layout_matches_window_rects(
    layout: &EditorLayout,
    window_rects: &HashMap<String, Rect>,
    window_collapsed: &HashMap<String, bool>,
    screen_rect: Rect,
    pixels_per_point: f32,
) -> bool {
    if layout.windows.is_empty() {
        return false;
    }
    let pixels_per_point = if pixels_per_point > 0.0 {
        pixels_per_point
    } else {
        1.0
    };
    let threshold = (1.5 / pixels_per_point).max(edge_threshold(pixels_per_point));
    for (id, window) in layout.windows.iter() {
        let Some(actual) = window_rects.get(id) else {
            return false;
        };
        let expected = round_rect_to_pixels(
            window.rect.to_rect(screen_rect),
            pixels_per_point,
            screen_rect,
        );
        let collapsed = window_collapsed.get(id).copied().unwrap_or(false);
        if collapsed {
            if (actual.min.x - expected.min.x).abs() > threshold
                || (actual.min.y - expected.min.y).abs() > threshold
                || (actual.max.x - expected.max.x).abs() > threshold
            {
                return false;
            }
        } else if (actual.min.x - expected.min.x).abs() > threshold
            || (actual.min.y - expected.min.y).abs() > threshold
            || (actual.max.x - expected.max.x).abs() > threshold
            || (actual.max.y - expected.max.y).abs() > threshold
        {
            return false;
        }
    }
    true
}

fn layout_title_bar_height(ctx: &egui::Context, collapsed: bool) -> f32 {
    let style = ctx.style();
    let font_id = egui::TextStyle::Heading.resolve(style.as_ref());
    let title_height = ctx
        .fonts_mut(|fonts| fonts.row_height(&font_id))
        .max(style.spacing.interact_size.y);
    let frame = egui::Frame::window(style.as_ref());
    let title_bar_inner_height = title_height + frame.inner_margin.sum().y;
    let title_content_spacing = if collapsed { 0.0 } else { frame.stroke.width };
    frame.outer_margin.topf() + frame.stroke.width + title_bar_inner_height + title_content_spacing
}

fn drag_edges_for_pos(rect: Rect, pos: Pos2, grab_radius: f32) -> LayoutDragEdges {
    let mut edges = LayoutDragEdges::default();
    if grab_radius <= 0.0 {
        return edges;
    }
    if !rect.expand(grab_radius).contains(pos) {
        return edges;
    }
    let left = (pos.x - rect.min.x).abs() <= grab_radius;
    let right = (pos.x - rect.max.x).abs() <= grab_radius;
    let top = (pos.y - rect.min.y).abs() <= grab_radius;
    let bottom = (pos.y - rect.max.y).abs() <= grab_radius;
    let center = rect.center();

    if left && right {
        edges.left = pos.x <= center.x;
        edges.right = pos.x > center.x;
    } else {
        edges.left = left;
        edges.right = right;
    }

    if top && bottom {
        edges.top = pos.y <= center.y;
        edges.bottom = pos.y > center.y;
    } else {
        edges.top = top;
        edges.bottom = bottom;
    }

    edges
}

fn hit_test_layout_window(
    ctx: &egui::Context,
    rect: Rect,
    content_rect: Option<Rect>,
    collapsed: bool,
    pos: Pos2,
    grab_radius: f32,
) -> Option<(LayoutDragMode, LayoutDragEdges)> {
    let edges = drag_edges_for_pos(rect, pos, grab_radius);
    let in_rect = rect.contains(pos);
    let in_content = content_rect.is_some_and(|content| content.contains(pos));
    let title_bar_height = layout_title_bar_height(ctx, collapsed);

    if edges.any() {
        if edges.top && in_rect && pos.y <= rect.min.y + title_bar_height {
            // keep title-bar move dominant except when truly on the top edge
            let top_resize_band = (grab_radius * 0.35).clamp(1.0, 3.0);
            if (pos.y - rect.min.y).abs() <= top_resize_band {
                return Some((LayoutDragMode::Resize, edges));
            }
        } else {
            return Some((LayoutDragMode::Resize, edges));
        }
    }

    if rect.contains(pos) {
        if in_content {
            return None;
        }
        if pos.y <= rect.min.y + title_bar_height {
            return Some((LayoutDragMode::Move, LayoutDragEdges::default()));
        }
    }

    None
}

fn pick_layout_drag_target(
    ctx: &egui::Context,
    window_rects: &HashMap<String, Rect>,
    window_content_rects: &HashMap<String, Rect>,
    window_collapsed: &HashMap<String, bool>,
    pos: Pos2,
    grab_radius: f32,
    preferred_id: Option<&str>,
) -> Option<(String, LayoutDragMode, LayoutDragEdges)> {
    if let Some(id) = preferred_id {
        if let Some(rect) = window_rects.get(id) {
            let content_rect = window_content_rects.get(id).copied();
            let collapsed = window_collapsed.get(id).copied().unwrap_or(false);
            if let Some(hit) =
                hit_test_layout_window(ctx, *rect, content_rect, collapsed, pos, grab_radius)
            {
                return Some((id.to_string(), hit.0, hit.1));
            }
            if rect.contains(pos) || content_rect.is_some_and(|content| content.contains(pos)) {
                return None;
            }
        }
    }

    for id in layout_window_ids() {
        let Some(rect) = window_rects.get(*id) else {
            continue;
        };
        let content_rect = window_content_rects.get(*id).copied();
        let collapsed = window_collapsed.get(*id).copied().unwrap_or(false);
        if let Some(hit) =
            hit_test_layout_window(ctx, *rect, content_rect, collapsed, pos, grab_radius)
        {
            return Some(((*id).to_string(), hit.0, hit.1));
        }
    }

    None
}

fn translate_rect(rect: Rect, delta: Vec2, screen_rect: Rect) -> Rect {
    let size = rect.size();
    let mut min = rect.min + delta;
    let max_x = (screen_rect.max.x - size.x).max(screen_rect.min.x);
    let max_y = (screen_rect.max.y - size.y).max(screen_rect.min.y);
    min.x = min.x.max(screen_rect.min.x).min(max_x);
    min.y = min.y.max(screen_rect.min.y).min(max_y);
    Rect::from_min_size(min, size)
}

fn resize_rect(rect: Rect, delta: Vec2, edges: LayoutDragEdges, screen_rect: Rect) -> Rect {
    let mut rect = rect;
    if edges.left {
        rect.min.x += delta.x;
    }
    if edges.right {
        rect.max.x += delta.x;
    }
    if edges.top {
        rect.min.y += delta.y;
    }
    if edges.bottom {
        rect.max.y += delta.y;
    }

    let min_size = Vec2::splat(32.0);
    if rect.width() < min_size.x {
        if edges.left && !edges.right {
            rect.min.x = rect.max.x - min_size.x;
        } else {
            rect.max.x = rect.min.x + min_size.x;
        }
    }
    if rect.height() < min_size.y {
        if edges.top && !edges.bottom {
            rect.min.y = rect.max.y - min_size.y;
        } else {
            rect.max.y = rect.min.y + min_size.y;
        }
    }

    clamp_rect_to_screen(&mut rect, screen_rect);
    rect
}

fn edge_threshold(pixels_per_point: f32) -> f32 {
    if pixels_per_point > 0.0 {
        0.75 / pixels_per_point
    } else {
        0.75
    }
}

fn apply_reflow_edges(
    base: &HashMap<String, Rect>,
    updated: &mut HashMap<String, Rect>,
    changed_id: &str,
    start_rect: Rect,
    new_rect: Rect,
    edges: LayoutDragEdges,
    threshold: f32,
) {
    if edges.left {
        let old_edge = start_rect.min.x;
        let new_edge = new_rect.min.x;
        for (id, rect) in base.iter() {
            if id == changed_id {
                continue;
            }
            if let Some(target) = updated.get_mut(id) {
                if (rect.min.x - old_edge).abs() <= threshold {
                    target.min.x = new_edge;
                }
                if (rect.max.x - old_edge).abs() <= threshold {
                    target.max.x = new_edge;
                }
            }
        }
    }

    if edges.right {
        let old_edge = start_rect.max.x;
        let new_edge = new_rect.max.x;
        for (id, rect) in base.iter() {
            if id == changed_id {
                continue;
            }
            if let Some(target) = updated.get_mut(id) {
                if (rect.min.x - old_edge).abs() <= threshold {
                    target.min.x = new_edge;
                }
                if (rect.max.x - old_edge).abs() <= threshold {
                    target.max.x = new_edge;
                }
            }
        }
    }

    if edges.top {
        let old_edge = start_rect.min.y;
        let new_edge = new_rect.min.y;
        for (id, rect) in base.iter() {
            if id == changed_id {
                continue;
            }
            if let Some(target) = updated.get_mut(id) {
                if (rect.min.y - old_edge).abs() <= threshold {
                    target.min.y = new_edge;
                }
                if (rect.max.y - old_edge).abs() <= threshold {
                    target.max.y = new_edge;
                }
            }
        }
    }

    if edges.bottom {
        let old_edge = start_rect.max.y;
        let new_edge = new_rect.max.y;
        for (id, rect) in base.iter() {
            if id == changed_id {
                continue;
            }
            if let Some(target) = updated.get_mut(id) {
                if (rect.min.y - old_edge).abs() <= threshold {
                    target.min.y = new_edge;
                }
                if (rect.max.y - old_edge).abs() <= threshold {
                    target.max.y = new_edge;
                }
            }
        }
    }
}

fn centered_project_rect(screen_rect: Rect) -> Rect {
    let width = screen_rect.width().max(1.0);
    let height = screen_rect.height().max(1.0);
    let mut size = Vec2::new(width * 0.6, height * 0.7);
    size.x = size.x.max(420.0).min(width);
    size.y = size.y.max(320.0).min(height);
    Rect::from_center_size(screen_rect.center(), size)
}

fn screen_rect_changed(prev: Option<Rect>, current: Rect, pixels_per_point: f32) -> bool {
    let Some(prev) = prev else {
        return true;
    };
    let epsilon = if pixels_per_point > 0.0 {
        0.5 / pixels_per_point
    } else {
        0.5
    };
    (prev.min.x - current.min.x).abs() > epsilon
        || (prev.min.y - current.min.y).abs() > epsilon
        || (prev.width() - current.width()).abs() > epsilon
        || (prev.height() - current.height()).abs() > epsilon
}

fn clamp_rect_to_screen(rect: &mut Rect, screen_rect: Rect) {
    rect.min.x = rect.min.x.max(screen_rect.min.x).min(screen_rect.max.x);
    rect.max.x = rect.max.x.max(screen_rect.min.x).min(screen_rect.max.x);
    rect.min.y = rect.min.y.max(screen_rect.min.y).min(screen_rect.max.y);
    rect.max.y = rect.max.y.max(screen_rect.min.y).min(screen_rect.max.y);
    if rect.max.x < rect.min.x {
        rect.max.x = rect.min.x;
    }
    if rect.max.y < rect.min.y {
        rect.max.y = rect.min.y;
    }
}

#[derive(Debug, Clone, Copy)]
struct EdgeCluster {
    min: f32,
    max: f32,
    snapped: f32,
}

fn build_edge_clusters(mut edges: Vec<f32>, epsilon: f32) -> Vec<EdgeCluster> {
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut clusters: Vec<(f32, f32, f32, u32)> = Vec::new();
    for edge in edges {
        if let Some((_min, max, sum, count)) = clusters.last_mut() {
            if edge - *max <= epsilon {
                *max = edge;
                *sum += edge;
                *count += 1;
                continue;
            }
        }
        clusters.push((edge, edge, edge, 1));
    }

    clusters
        .into_iter()
        .map(|(min, max, sum, count)| {
            let mean = sum / count as f32;
            EdgeCluster {
                min,
                max,
                snapped: mean.round(),
            }
        })
        .collect()
}

fn snap_edge(edge: f32, clusters: &[EdgeCluster], epsilon: f32) -> f32 {
    for cluster in clusters {
        if edge >= cluster.min - epsilon && edge <= cluster.max + epsilon {
            return cluster.snapped;
        }
    }
    edge.round()
}

pub fn editor_physics_state_system(
    scene_state: Res<EditorSceneState>,
    mut phys: ResMut<PhysicsResource>,
) {
    let should_run = scene_state.world_state == WorldState::Play;
    if phys.running != should_run {
        phys.running = should_run;
    }
}

pub fn editor_command_system(world: &mut World) {
    let commands = {
        let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() else {
            return;
        };
        std::mem::take(&mut queue.commands)
    };

    for command in commands {
        match command {
            EditorCommand::CreateProject { name, path } => {
                handle_create_project(world, &name, &path);
            }
            EditorCommand::OpenProject { path } => {
                handle_open_project(world, &path);
            }
            EditorCommand::NewScene => {
                handle_new_scene(world);
            }
            EditorCommand::OpenScene { path } => {
                handle_open_scene(world, &path);
            }
            EditorCommand::SaveScene => {
                handle_save_scene(world);
            }
            EditorCommand::SaveSceneAs { path } => {
                handle_save_scene_as(world, &path);
            }
            EditorCommand::CreateEntity { kind } => {
                handle_create_entity(world, kind);
                push_undo_snapshot(world, "Create Entity");
            }
            EditorCommand::ImportAsset {
                source_path,
                destination_dir,
            } => {
                handle_import_asset(world, &source_path, destination_dir.as_deref());
            }
            EditorCommand::CreateAsset {
                directory,
                name,
                kind,
            } => {
                handle_create_asset(world, &directory, &name, kind);
            }
            EditorCommand::DeleteEntity { entity } => {
                let to_delete = collect_entity_subtree(world, entity);
                let existed = !to_delete.is_empty();
                let removed: HashSet<Entity> = to_delete.iter().copied().collect();
                for target in to_delete {
                    if world.get_entity(target).is_ok() {
                        world.despawn(target);
                    }
                }

                if let Some(mut spawned) = world.get_resource_mut::<SceneSpawnedChildren>() {
                    for removed_entity in removed.iter().copied() {
                        spawned.spawned_scenes.remove(&removed_entity);
                    }
                    spawned.spawned_scenes.retain(|root_entity, children| {
                        if removed.contains(root_entity) {
                            return false;
                        }
                        children.retain(|child| !removed.contains(child));
                        true
                    });
                }

                if let Some(mut selection) = world.get_resource_mut::<
                    helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource,
                >() {
                    if selection.0.is_some_and(|selected| removed.contains(&selected)) {
                        selection.0 = None;
                    }
                }
                if let Some(mut pinned) = world.get_resource_mut::<InspectorPinnedEntityResource>()
                {
                    if pinned.0.is_some_and(|selected| removed.contains(&selected)) {
                        pinned.0 = None;
                    }
                }
                if existed {
                    push_undo_snapshot(world, "Delete Entity");
                }
            }
            EditorCommand::SetActiveCamera { entity } => {
                if handle_set_active_camera(world, entity) {
                    push_undo_snapshot(world, "Set Active Camera");
                }
            }
            EditorCommand::TogglePlayMode => {
                handle_toggle_play(world);
            }
            EditorCommand::Undo => {
                let label = undo_action(world);
                if let Some(label) = label {
                    set_status(world, format!("Undo {}", label));
                } else {
                    set_status(world, "Nothing to undo".to_string());
                }
            }
            EditorCommand::Redo => {
                let label = redo_action(world);
                if let Some(label) = label {
                    set_status(world, format!("Redo {}", label));
                } else {
                    set_status(world, "Nothing to redo".to_string());
                }
            }
            EditorCommand::CloseProject => {
                handle_close_project(world);
            }
        }
    }
}

fn collect_entity_subtree(world: &mut World, root: Entity) -> Vec<Entity> {
    if world.get_entity(root).is_err() {
        return Vec::new();
    }

    let mut children_by_parent: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut relation_query = world.query::<(Entity, &EntityParent)>();
    for (entity, relation) in relation_query.iter(world) {
        children_by_parent
            .entry(relation.parent)
            .or_default()
            .push(entity);
    }

    let mut ordered = Vec::new();
    let mut visited: HashSet<Entity> = HashSet::new();
    let mut stack = vec![root];

    while let Some(entity) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }
        ordered.push(entity);
        if let Some(children) = children_by_parent.get(&entity) {
            for child in children.iter().copied() {
                stack.push(child);
            }
        }
    }

    ordered.reverse();
    ordered
}

pub fn asset_scan_system(mut state: ResMut<AssetBrowserState>) {
    let root = match state.root.as_ref() {
        Some(root) => root.clone(),
        None => return,
    };

    let now = Instant::now();
    if !state.refresh_requested && now.duration_since(state.last_scan) < state.scan_interval {
        return;
    }

    state.entries = scan_asset_entries(&root, &state.filter);
    state.refresh_requested = false;
    state.last_scan = now;
}

pub fn drag_drop_system(
    mut dragged: ResMut<DraggedFile>,
    mut queue: ResMut<EditorCommandQueue>,
    assets: Res<AssetBrowserState>,
) {
    if let Some(path) = dragged.0.take() {
        let destination_dir = assets.current_dir.clone().or_else(|| {
            assets.selected.as_ref().and_then(|selected| {
                if selected.is_dir() {
                    Some(selected.clone())
                } else {
                    selected.parent().map(|parent| parent.to_path_buf())
                }
            })
        });

        queue.push(EditorCommand::ImportAsset {
            source_path: path,
            destination_dir,
        });
    }
}

pub fn editor_shortcut_system(
    input_manager: Res<BevyInputManager>,
    mut queue: ResMut<EditorCommandQueue>,
) {
    let input_manager = input_manager.0.read();

    if input_manager.egui_wants_key {
        return;
    }

    let control = input_manager.is_key_active(KeyCode::ControlLeft)
        || input_manager.is_key_active(KeyCode::ControlRight);
    let shift = input_manager.is_key_active(KeyCode::ShiftLeft)
        || input_manager.is_key_active(KeyCode::ShiftRight);
    if control && input_manager.just_pressed.contains(&KeyCode::KeyN) {
        queue.push(EditorCommand::NewScene);
    }
    if control && input_manager.just_pressed.contains(&KeyCode::KeyS) {
        queue.push(EditorCommand::SaveScene);
    }
    if control && input_manager.just_pressed.contains(&KeyCode::KeyZ) {
        if shift {
            queue.push(EditorCommand::Redo);
        } else {
            queue.push(EditorCommand::Undo);
        }
    }
    if control && input_manager.just_pressed.contains(&KeyCode::KeyY) {
        queue.push(EditorCommand::Redo);
    }
    if input_manager.just_pressed.contains(&KeyCode::F5) {
        queue.push(EditorCommand::TogglePlayMode);
    }
}

pub fn pane_manager_toggle_system(
    mut pane_manager: ResMut<EditorPaneManagerState>,
    egui_res: Res<EguiResource>,
) {
    let toggle = egui_res
        .ctx
        .input(|input| input.key_pressed(egui::Key::Tab) && input.modifiers.ctrl);
    if toggle {
        pane_manager.open = !pane_manager.open;
        egui_res.ctx.memory_mut(|mem| {
            if let Some(id) = mem.focused() {
                mem.surrender_focus(id);
            }
        });
    }
}

pub fn scene_dirty_system(
    mut scene_state: ResMut<EditorSceneState>,
    query_core: Query<
        (),
        (
            Or<(With<EditorEntity>, With<SceneChild>)>,
            Or<(
                Changed<BevyTransform>,
                Changed<BevyMeshRenderer>,
                Changed<BevySkinnedMeshRenderer>,
                Changed<EditorMesh>,
                Changed<EditorSkinnedMesh>,
                Changed<BevyLight>,
                Changed<BevyCamera>,
                Changed<BevyPoseOverride>,
                Changed<BevySpline>,
                Changed<BevySplineFollower>,
                Changed<BevyLookAt>,
                Changed<BevyEntityFollower>,
                Changed<EntityParent>,
            )>,
        ),
    >,
    query_core_extra: Query<
        (),
        (
            Or<(With<EditorEntity>, With<SceneChild>)>,
            Or<(
                Changed<BevyAudioEmitter>,
                Changed<BevyAudioListener>,
                Changed<EditorAudio>,
                Changed<Name>,
                Changed<SceneRoot>,
                Changed<SceneAssetPath>,
                Changed<SceneChild>,
            )>,
        ),
    >,
    query_extra: Query<
        (),
        (
            Or<(With<EditorEntity>, With<SceneChild>)>,
            Or<(Changed<ScriptComponent>, Changed<DynamicComponents>)>,
        ),
    >,
    pose_child_query: Query<(), (With<SceneChild>, Changed<BevyPoseOverride>)>,
) {
    if scene_state.world_state == WorldState::Play {
        return;
    }

    if !query_core.is_empty()
        || !query_core_extra.is_empty()
        || !query_extra.is_empty()
        || !pose_child_query.is_empty()
    {
        scene_state.dirty = true;
    }
}

pub fn apply_scene_child_animations_system(
    mut pending: ResMut<PendingSceneChildAnimations>,
    mut timeline: ResMut<EditorTimelineState>,
    project: Res<EditorProject>,
    scene_roots: Query<(Entity, &SceneAssetPath), With<SceneRoot>>,
    child_query: Query<(Entity, &SceneChild)>,
    mut animator_query: Query<&mut BevyAnimator>,
    skinned_query: Query<&BevySkinnedMeshRenderer>,
    name_query: Query<&Name>,
) {
    if pending.entries.is_empty() {
        return;
    }

    let root = project.root.as_deref();
    let mut root_map: HashMap<String, Entity> = HashMap::new();
    for (entity, path) in scene_roots.iter() {
        let normalized = normalize_path(path.path.to_string_lossy().as_ref(), root);
        root_map.insert(normalized, entity);
    }
    let mut child_map: HashMap<(Entity, usize), Entity> = HashMap::new();
    for (entity, child) in child_query.iter() {
        child_map
            .entry((child.scene_root, child.scene_node_index))
            .or_insert(entity);
    }

    let mut remaining = Vec::new();
    for entry in pending.entries.drain(..) {
        let Some(root_entity) = root_map.get(&entry.scene_path).copied() else {
            remaining.push(entry);
            continue;
        };
        let Some(child_entity) = child_map
            .get(&(root_entity, entry.scene_node_index))
            .copied()
        else {
            remaining.push(entry);
            continue;
        };

        let skeleton = skinned_query
            .get(child_entity)
            .map(|skinned| skinned.0.skin.skeleton.as_ref())
            .ok();
        let name = name_query
            .get(child_entity)
            .map(|name| name.to_string())
            .unwrap_or_else(|_| format!("Entity {}", child_entity.index()));

        apply_animation_data_to_timeline(
            &mut timeline,
            child_entity,
            name,
            &entry.animation,
            skeleton,
        );

        if let Ok(mut animator) = animator_query.get_mut(child_entity) {
            let custom_clips = entry
                .animation
                .clips
                .iter()
                .map(|clip| clip.to_clip_for_skeleton(skeleton))
                .collect::<Vec<_>>();
            apply_custom_clips_to_animator(&mut animator, &custom_clips);
        }
    }

    pending.entries = remaining;
}

pub fn apply_scene_child_pose_overrides_system(
    mut commands: bevy_ecs::prelude::Commands,
    mut pending: ResMut<PendingSceneChildPoseOverrides>,
    project: Res<EditorProject>,
    scene_roots: Query<(Entity, &SceneAssetPath), With<SceneRoot>>,
    child_query: Query<(Entity, &SceneChild)>,
    skinned_query: Query<&BevySkinnedMeshRenderer>,
    mut pose_query: Query<&mut BevyPoseOverride>,
) {
    if pending.entries.is_empty() {
        return;
    }

    let root = project.root.as_deref();
    let mut root_map: HashMap<String, Entity> = HashMap::new();
    for (entity, path) in scene_roots.iter() {
        let normalized = normalize_path(path.path.to_string_lossy().as_ref(), root);
        root_map.insert(normalized, entity);
    }
    let mut child_map: HashMap<(Entity, usize), Entity> = HashMap::new();
    for (entity, child) in child_query.iter() {
        child_map
            .entry((child.scene_root, child.scene_node_index))
            .or_insert(entity);
    }

    let mut remaining = Vec::new();
    for entry in pending.entries.drain(..) {
        let Some(root_entity) = root_map.get(&entry.scene_path).copied() else {
            remaining.push(entry);
            continue;
        };
        let Some(child_entity) = child_map
            .get(&(root_entity, entry.scene_node_index))
            .copied()
        else {
            remaining.push(entry);
            continue;
        };

        let skeleton = match skinned_query.get(child_entity) {
            Ok(skinned) => skinned.0.skin.skeleton.clone(),
            Err(_) => {
                remaining.push(entry);
                continue;
            }
        };
        let pose = pose_from_serialized(&entry.pose.locals, Some(skeleton.as_ref()));
        let pose_override = PoseOverride {
            enabled: entry.pose.enabled,
            pose,
        };

        if let Ok(mut existing) = pose_query.get_mut(child_entity) {
            existing.0 = pose_override;
        } else {
            commands
                .entity(child_entity)
                .try_insert(BevyPoseOverride(pose_override));
        }
    }

    pending.entries = remaining;
}

pub fn pending_scene_child_renderer_system(world: &mut World) {
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|state| state.world_state);
    if world_state != Some(WorldState::Edit) {
        return;
    }

    let asset_server = {
        let Some(asset_server) = world.get_resource::<helmer_becs::BevyAssetServer>() else {
            return;
        };
        asset_server.0.clone()
    };

    let mut pending_entries = Vec::new();
    {
        let mut query = world.query::<(Entity, &PendingSceneChildRenderer, &SceneChild)>();
        for (entity, pending, scene_child) in query.iter(world) {
            pending_entries.push((entity, *pending, *scene_child));
        }
    }

    if pending_entries.is_empty() {
        return;
    }

    for (entity, pending, scene_child) in pending_entries {
        let scene_handle = {
            let Some(scene_root) = world.get::<SceneRoot>(scene_child.scene_root) else {
                world
                    .entity_mut(entity)
                    .remove::<PendingSceneChildRenderer>();
                continue;
            };
            scene_root.0.clone()
        };

        let scene = {
            let asset_server = asset_server.lock();
            asset_server.request_scene_assets(&scene_handle, Some(0), 1.0);
            asset_server.get_scene(&scene_handle)
        };
        let Some(scene) = scene else {
            continue;
        };
        let Some(node) = scene.nodes.get(scene_child.scene_node_index) else {
            world
                .entity_mut(entity)
                .remove::<PendingSceneChildRenderer>();
            continue;
        };

        let mesh_id = node.mesh.id;
        let material_id = node.material.id;
        let skin_index = node.skin_index;
        let mut kind = pending.kind;
        if kind == SceneChildRendererKind::Auto {
            kind = if skin_index.is_some() {
                SceneChildRendererKind::Skinned
            } else {
                SceneChildRendererKind::Mesh
            };
        }

        match kind {
            SceneChildRendererKind::Auto => {}
            SceneChildRendererKind::None => {
                let mut entity_mut = world.entity_mut(entity);
                entity_mut.remove::<BevyMeshRenderer>();
                entity_mut.remove::<BevySkinnedMeshRenderer>();
                entity_mut.remove::<PendingSkinnedMeshAsset>();
                entity_mut.remove::<BevyAnimator>();
                entity_mut.remove::<BevyPoseOverride>();
                entity_mut.remove::<PendingSceneChildRenderer>();
            }
            SceneChildRendererKind::Mesh => {
                let mesh_renderer =
                    MeshRenderer::new(mesh_id, material_id, pending.casts_shadow, pending.visible);
                let mut entity_mut = world.entity_mut(entity);
                entity_mut.remove::<BevySkinnedMeshRenderer>();
                entity_mut.remove::<PendingSkinnedMeshAsset>();
                entity_mut.remove::<BevyAnimator>();
                entity_mut.remove::<BevyPoseOverride>();
                entity_mut.insert(BevyWrapper(mesh_renderer));
                entity_mut.remove::<PendingSceneChildRenderer>();
            }
            SceneChildRendererKind::Skinned => {
                let Some(skin_index) = skin_index else {
                    let mesh_renderer = MeshRenderer::new(
                        mesh_id,
                        material_id,
                        pending.casts_shadow,
                        pending.visible,
                    );
                    let mut entity_mut = world.entity_mut(entity);
                    entity_mut.remove::<BevySkinnedMeshRenderer>();
                    entity_mut.remove::<PendingSkinnedMeshAsset>();
                    entity_mut.remove::<BevyAnimator>();
                    entity_mut.remove::<BevyPoseOverride>();
                    entity_mut.insert(BevyWrapper(mesh_renderer));
                    entity_mut.remove::<PendingSceneChildRenderer>();
                    continue;
                };
                let Some(skin) = scene.skins.read().get(skin_index).cloned() else {
                    continue;
                };
                let anim_lib = scene.animations.read().get(skin_index).cloned();
                let skinned = SkinnedMeshRenderer::new(
                    mesh_id,
                    material_id,
                    skin,
                    pending.casts_shadow,
                    pending.visible,
                );

                let has_animator = world.get::<BevyAnimator>(entity).is_some();
                {
                    let mut entity_mut = world.entity_mut(entity);
                    entity_mut.remove::<BevyMeshRenderer>();
                    entity_mut.remove::<PendingSkinnedMeshAsset>();
                    entity_mut.insert(BevySkinnedMeshRenderer(skinned));
                    entity_mut.remove::<PendingSceneChildRenderer>();
                }

                if !has_animator {
                    if let Some(anim_lib) = anim_lib {
                        world
                            .entity_mut(entity)
                            .insert(BevyAnimator(build_default_animator(anim_lib)));
                    }
                }
            }
        }
    }
}

pub fn pending_skinned_mesh_system(world: &mut World) {
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|state| state.world_state);
    if world_state != Some(WorldState::Edit) {
        return;
    }

    let asset_server = {
        let Some(asset_server) = world.get_resource::<helmer_becs::BevyAssetServer>() else {
            return;
        };
        asset_server.0.clone()
    };

    let mut pending_entries = Vec::new();
    {
        let mut query = world.query::<(Entity, &PendingSkinnedMeshAsset)>();
        for (entity, pending) in query.iter(world) {
            pending_entries.push((entity, pending.clone()));
        }
    }

    if pending_entries.is_empty() {
        return;
    }

    let mut applied_any = false;
    for (entity, pending) in pending_entries {
        let scene = {
            let asset_server = asset_server.lock();
            asset_server.get_scene(&pending.scene_handle)
        };
        let Some(scene) = scene else {
            continue;
        };
        let desired_index = pending.node_index.or_else(|| {
            world
                .get::<EditorSkinnedMesh>(entity)
                .and_then(|skinned| skinned.node_index)
        });
        let node_index = if let Some(index) = desired_index {
            Some(index)
        } else {
            scene
                .nodes
                .iter()
                .position(|node| node.skin_index.is_some())
        };

        let Some(node_index) = node_index else {
            world.entity_mut(entity).remove::<PendingSkinnedMeshAsset>();
            set_status(world, "Scene has no skinned nodes.".to_string());
            continue;
        };

        if pending.node_index.is_none() {
            world.entity_mut(entity).insert(PendingSkinnedMeshAsset {
                scene_handle: pending.scene_handle,
                node_index: Some(node_index),
            });
            if let Some(mut editor_skinned) = world.get_mut::<EditorSkinnedMesh>(entity) {
                editor_skinned.node_index = Some(node_index);
            }
        }

        let Some(node) = scene.nodes.get(node_index) else {
            continue;
        };
        let Some(skin_index) = node.skin_index else {
            continue;
        };
        let Some(skin) = scene.skins.read().get(skin_index).cloned() else {
            continue;
        };

        let (casts_shadow, visible) = world
            .get::<BevyMeshRenderer>(entity)
            .map(|renderer| (renderer.0.casts_shadow, renderer.0.visible))
            .or_else(|| {
                world
                    .get::<EditorSkinnedMesh>(entity)
                    .map(|skinned| (skinned.casts_shadow, skinned.visible))
            })
            .unwrap_or((true, true));

        let skinned =
            SkinnedMeshRenderer::new(node.mesh.id, node.material.id, skin, casts_shadow, visible);

        let has_animator = world.get::<BevyAnimator>(entity).is_some();
        {
            let mut entity_mut = world.entity_mut(entity);
            entity_mut.remove::<BevyMeshRenderer>();
            entity_mut.remove::<EditorMesh>();
            entity_mut.remove::<PendingSkinnedMeshAsset>();
            entity_mut.insert(BevySkinnedMeshRenderer(skinned));
        }

        if !has_animator {
            if let Some(anim_lib) = scene.animations.read().get(skin_index).cloned() {
                world
                    .entity_mut(entity)
                    .insert(BevyAnimator(build_default_animator(anim_lib)));
            }
        }

        applied_any = true;
    }

    if applied_any {
        push_undo_snapshot(world, "Skinned Mesh");
    }
}

pub fn editor_render_refresh_system(world: &mut World) {
    let pending = world
        .get_resource::<EditorRenderRefresh>()
        .map(|refresh| refresh.pending)
        .unwrap_or(false);
    if !pending {
        return;
    }

    if let Some(mut refresh) = world.get_resource_mut::<EditorRenderRefresh>() {
        refresh.pending = false;
    }

    let entities: Vec<Entity> = world
        .query_filtered::<Entity, Or<(With<BevyMeshRenderer>, With<BevySkinnedMeshRenderer>)>>()
        .iter(world)
        .collect();
    for entity in entities {
        if let Some(mut transform) = world.get_mut::<BevyTransform>(entity) {
            let current = transform.0;
            transform.0 = current;
        }
    }
}

pub fn script_registry_system(mut registry: ResMut<ScriptRegistry>, project: Res<EditorProject>) {
    let Some(root) = project.root.as_ref() else {
        return;
    };
    let scripts_root = project.config.as_ref().map(|cfg| cfg.scripts_root(root));

    let mut dirty_paths = registry.take_dirty_paths();
    if !dirty_paths.is_empty() {
        let mut updated = 0;
        let mut removed = 0;
        for path in dirty_paths.drain() {
            if !is_script_path(&path) {
                continue;
            }

            let Some(script_key) = script_registry_key_for_path(&path) else {
                continue;
            };

            if script_key.exists() {
                registry
                    .scripts
                    .insert(script_key.clone(), load_script_asset(&script_key));
                updated += 1;
            } else if registry.scripts.remove(&script_key).is_some() {
                removed += 1;
            }
        }

        if updated > 0 || removed > 0 {
            registry.status = Some(format!(
                "Reloaded {} script(s), removed {}",
                updated, removed
            ));
        }
        return;
    }

    let now = Instant::now();
    if now.duration_since(registry.last_scan) < registry.scan_interval {
        return;
    }

    registry.last_scan = now;

    let Some(scripts_root) = scripts_root else {
        return;
    };

    if !scripts_root.exists() {
        let stale = registry
            .scripts
            .keys()
            .filter(|path| path.starts_with(&scripts_root))
            .cloned()
            .collect::<Vec<_>>();
        if !stale.is_empty() {
            for path in stale {
                registry.scripts.remove(&path);
            }
            registry.status = Some("Removed stale script cache entries".to_string());
        }
        return;
    }

    let mut discovered = HashSet::new();
    let mut updated = 0;
    let mut removed = 0;

    for entry in WalkDir::new(&scripts_root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        let path = entry.path().to_path_buf();
        if !is_script_path(&path) {
            continue;
        }

        let Some(script_key) = script_registry_key_for_path(&path) else {
            continue;
        };
        if !discovered.insert(script_key.clone()) {
            continue;
        }

        let next_asset = load_script_asset(&script_key);
        let reload = match registry.scripts.get(&script_key) {
            Some(existing) => {
                next_asset.modified > existing.modified
                    || next_asset.error != existing.error
                    || next_asset.language != existing.language
            }
            None => true,
        };

        if reload {
            registry.scripts.insert(script_key.clone(), next_asset);
            updated += 1;
        }
    }

    let stale = registry
        .scripts
        .keys()
        .filter(|path| path.starts_with(&scripts_root) && !discovered.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    for path in stale {
        if registry.scripts.remove(&path).is_some() {
            removed += 1;
        }
    }

    if updated > 0 || removed > 0 {
        registry.status = Some(format!(
            "Updated {} script(s), removed {}",
            updated, removed
        ));
    }
}

fn handle_create_project(world: &mut World, name: &str, path: &Path) {
    match create_project(path, name) {
        Ok(config) => {
            {
                let mut project = world
                    .get_resource_mut::<EditorProject>()
                    .expect("EditorProject missing");
                project.root = Some(path.to_path_buf());
                project.config = Some(config);
            }

            record_recent_project(world, path);

            let project_snapshot = world
                .get_resource::<EditorProject>()
                .cloned()
                .expect("EditorProject missing");
            initialize_project_state(world, &project_snapshot);
            set_status(world, format!("Project created at {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to create project: {}", err));
        }
    }
}

fn handle_open_project(world: &mut World, path: &Path) {
    match load_project(path) {
        Ok(config) => {
            {
                let mut project = world
                    .get_resource_mut::<EditorProject>()
                    .expect("EditorProject missing");
                project.root = Some(path.to_path_buf());
                project.config = Some(config);
            }

            record_recent_project(world, path);

            let project_snapshot = world
                .get_resource::<EditorProject>()
                .cloned()
                .expect("EditorProject missing");
            initialize_project_state(world, &project_snapshot);
            set_status(world, format!("Project opened at {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to open project: {}", err));
        }
    }
}

fn handle_close_project(world: &mut World) {
    if let Some(mut project) = world.get_resource_mut::<EditorProject>() {
        project.root = None;
        project.config = None;
    }

    if let Some(mut workspace) = world.get_resource_mut::<EditorWorkspaceState>() {
        workspace.windows.clear();
        workspace.dragging = None;
        workspace.last_focused_window = None;
        workspace.drop_handled = false;
        workspace.next_window_id = 1;
        workspace.next_tab_id = 1;
    }

    if let Some(mut assets) = world.get_resource_mut::<AssetBrowserState>() {
        assets.root = None;
        assets.entries.clear();
        assets.expanded.clear();
        assets.selected = None;
        assets.current_dir = None;
        assets.refresh_requested = true;
    }

    if let Some(mut watcher) = world.get_resource_mut::<crate::editor::watch::FileWatchState>() {
        watcher.root = None;
        watcher.watcher = None;
        watcher.receiver = None;
        watcher.pending_paths.clear();
        watcher.status = Some("File watcher disabled".to_string());
    }

    if let Some(mut registry) = world.get_resource_mut::<ScriptRegistry>() {
        registry.scripts.clear();
        registry.dirty_paths.clear();
        registry.status = None;
    }
    if let Some(mut runtime) = world.get_resource_mut::<ScriptRuntime>() {
        runtime.clear_all();
    }

    handle_new_scene(world);
    set_status(world, "Project closed".to_string());
}

fn record_recent_project(world: &mut World, path: &Path) {
    let normalized = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
        state.recent_projects.retain(|entry| entry != &normalized);
        state.recent_projects.insert(0, normalized.clone());
        const MAX_RECENT_PROJECTS: usize = 8;
        if state.recent_projects.len() > MAX_RECENT_PROJECTS {
            state.recent_projects.truncate(MAX_RECENT_PROJECTS);
        }

        if let Err(err) = save_recent_projects(&state.recent_projects) {
            state.status = Some(format!("Failed to save recent projects: {}", err));
        }
    }
}

fn initialize_project_state(world: &mut World, project: &EditorProject) {
    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = world
            .get_resource::<helmer_becs::BevyAssetServer>()
            .expect("AssetServer missing");
        ensure_default_material(project, &mut cache, asset_server);
    });

    if let Some(mut asset_state) = world.get_resource_mut::<AssetBrowserState>() {
        if let Some(root) = project.root.as_ref() {
            asset_state.root = Some(root.to_path_buf());
            asset_state.expanded.insert(root.to_path_buf());
            asset_state.selected = Some(root.to_path_buf());
            asset_state.current_dir = Some(root.to_path_buf());
            asset_state.refresh_requested = true;
        }
    }

    if let Some(root) = project.root.as_ref() {
        if let Some(mut watcher) = world.get_resource_mut::<crate::editor::watch::FileWatchState>()
        {
            configure_file_watcher(&mut watcher, root);
        }
    }

    handle_new_scene(world);
}

fn handle_new_scene(world: &mut World) {
    reset_editor_scene(world);
    spawn_default_camera(world);
    spawn_default_light(world);

    if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
        scene_state.path = None;
        scene_state.name = "Untitled".to_string();
        scene_state.dirty = false;
        scene_state.play_backup = None;
        scene_state.play_selected_index = None;
    }

    if let Some(mut selection) = world.get_resource_mut::<
        helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource,
    >() {
        selection.0 = None;
    }
    if let Some(mut pinned) = world.get_resource_mut::<InspectorPinnedEntityResource>() {
        pinned.0 = None;
    }

    reset_undo_history(world);

    if let Some(mut render_sync) = world.get_resource_mut::<RenderSyncRequest>() {
        render_sync.request_with_epoch(3);
    }
    if let Some(mut refresh) = world.get_resource_mut::<EditorRenderRefresh>() {
        refresh.pending = true;
    }
}

fn handle_open_scene(world: &mut World, path: &Path) {
    match read_scene_document(path) {
        Ok(document) => {
            let project_snapshot = world
                .get_resource::<EditorProject>()
                .cloned()
                .expect("EditorProject missing");

            reset_editor_scene(world);
            if let Some(mut pinned) = world.get_resource_mut::<InspectorPinnedEntityResource>() {
                pinned.0 = None;
            }

            let created_entities =
                world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
                    let asset_server = {
                        let asset_server = world
                            .get_resource::<helmer_becs::BevyAssetServer>()
                            .expect("AssetServer missing");
                        helmer_becs::BevyAssetServer(asset_server.0.clone())
                    };
                    spawn_scene_from_document(
                        world,
                        &document,
                        &project_snapshot,
                        &mut cache,
                        &asset_server,
                    )
                });

            restore_scene_transforms_from_document(world, &document, &created_entities);

            if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
                scene_state.path = Some(path.to_path_buf());
                scene_state.name = scene_display_name(path);
                scene_state.dirty = false;
                scene_state.play_backup = None;
            }

            if let Some(mut render_sync) = world.get_resource_mut::<RenderSyncRequest>() {
                render_sync.request_with_epoch(3);
            }
            if let Some(mut refresh) = world.get_resource_mut::<EditorRenderRefresh>() {
                refresh.pending = true;
            }

            reset_undo_history(world);
            set_status(world, format!("Scene loaded from {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to load scene: {}", err));
        }
    }
}

fn handle_save_scene(world: &mut World) {
    let scene_path = world
        .get_resource::<EditorSceneState>()
        .and_then(|scene| scene.path.clone());

    if let Some(path) = scene_path {
        handle_save_scene_as(world, &path);
        return;
    }

    if let Some(path) = next_available_scene_path(
        world
            .get_resource::<EditorProject>()
            .expect("Project missing"),
    ) {
        handle_save_scene_as(world, &path);
    } else {
        set_status(world, "Unable to allocate a scene file name".to_string());
    }
}

fn handle_save_scene_as(world: &mut World, path: &Path) {
    let project = world
        .get_resource::<EditorProject>()
        .cloned()
        .expect("EditorProject missing");
    let (document, _) = serialize_scene(world, &project);

    match write_scene_document(path, &document) {
        Ok(()) => {
            if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
                scene_state.path = Some(path.to_path_buf());
                scene_state.name = scene_display_name(path);
                scene_state.dirty = false;
            }

            if let Some(mut asset_state) = world.get_resource_mut::<AssetBrowserState>() {
                asset_state.refresh_requested = true;
            }

            mark_undo_clean(world);
            set_status(world, format!("Scene saved to {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to save scene: {}", err));
        }
    }
}

fn handle_create_entity(world: &mut World, kind: SpawnKind) {
    match kind {
        SpawnKind::Empty => {
            world.spawn((EditorEntity, BevyTransform::default(), Name::new("Empty")));
        }
        SpawnKind::Camera => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                BevyCamera::default(),
                Name::new("Camera"),
            ));
        }
        SpawnKind::FreecamCamera => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                BevyCamera::default(),
                Freecam {},
                Name::new("Freecam Camera"),
            ));
        }
        SpawnKind::DirectionalLight => {
            world.spawn((
                EditorEntity,
                BevyWrapper(Transform::default()),
                BevyWrapper(Light::directional(glam::vec3(1.0, 1.0, 1.0), 25.0)),
                Name::new("Directional Light"),
            ));
        }
        SpawnKind::PointLight => {
            world.spawn((
                EditorEntity,
                BevyWrapper(Transform::default()),
                BevyWrapper(Light::point(glam::vec3(1.0, 1.0, 1.0), 10.0)),
                Name::new("Point Light"),
            ));
        }
        SpawnKind::SpotLight => {
            world.spawn((
                EditorEntity,
                BevyWrapper(Transform::default()),
                BevyWrapper(Light::spot(
                    glam::vec3(1.0, 1.0, 1.0),
                    10.0,
                    45.0_f32.to_radians(),
                )),
                Name::new("Spot Light"),
            ));
        }
        SpawnKind::Primitive(kind) => {
            spawn_primitive(world, kind);
        }
        SpawnKind::DynamicBodyCuboid => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                DynamicRigidBody { mass: 1.0 },
                ColliderShape::Cuboid,
                Name::new("Dynamic Body (Box)"),
            ));
        }
        SpawnKind::DynamicBodySphere => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                DynamicRigidBody { mass: 1.0 },
                ColliderShape::Sphere,
                Name::new("Dynamic Body (Sphere)"),
            ));
        }
        SpawnKind::FixedColliderCuboid => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                FixedCollider,
                ColliderShape::Cuboid,
                Name::new("Fixed Collider (Box)"),
            ));
        }
        SpawnKind::FixedColliderSphere => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                FixedCollider,
                ColliderShape::Sphere,
                Name::new("Fixed Collider (Sphere)"),
            ));
        }
        SpawnKind::SceneAsset(path) => {
            spawn_scene_asset(world, &path);
        }
        SpawnKind::MeshAsset(path) => {
            spawn_mesh_asset(world, &path);
        }
    }

    if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
        scene_state.dirty = true;
    }
}

fn handle_import_asset(world: &mut World, source: &Path, destination: Option<&Path>) {
    let Some(project) = world.get_resource::<EditorProject>() else {
        set_status(world, "Open a project before importing".to_string());
        return;
    };

    let Some(root) = project.root.as_ref() else {
        set_status(world, "Open a project before importing".to_string());
        return;
    };

    let target_dir = destination
        .map(|path| path.to_path_buf())
        .or_else(|| guess_import_dir(project, root, source))
        .unwrap_or_else(|| root.join("assets"));

    if !target_dir.exists() {
        if let Err(err) = fs::create_dir_all(&target_dir) {
            set_status(world, format!("Failed to create target dir: {}", err));
            return;
        }
    }

    let Some(file_name) = source.file_name() else {
        set_status(world, "Invalid source file".to_string());
        return;
    };

    let mut target_path = target_dir.join(file_name);
    target_path = unique_path(&target_path);

    match fs::copy(source, &target_path) {
        Ok(_) => {
            if let Some(mut asset_state) = world.get_resource_mut::<AssetBrowserState>() {
                asset_state.refresh_requested = true;
            }
            set_status(
                world,
                format!("Imported asset to {}", target_path.display()),
            );
        }
        Err(err) => {
            set_status(world, format!("Import failed: {}", err));
        }
    }
}

fn handle_create_asset(world: &mut World, directory: &Path, name: &str, kind: AssetCreateKind) {
    let target_path = match kind {
        AssetCreateKind::Folder => directory.join(name),
        AssetCreateKind::Scene => directory.join(format!("{}.hscene.ron", name)),
        AssetCreateKind::Material => directory.join(format!("{}.ron", name)),
        AssetCreateKind::Script => directory.join(format!("{}.luau", name)),
        AssetCreateKind::VisualScript => directory.join(format!("{}.hvs", name)),
        AssetCreateKind::RustScript => directory.join(name),
        AssetCreateKind::Animation => directory.join(format!("{}.hanim.ron", name)),
    };

    let target_path = unique_path(&target_path);

    let result = match kind {
        AssetCreateKind::Folder => fs::create_dir_all(&target_path).map_err(|err| err.to_string()),
        AssetCreateKind::Scene => {
            fs::write(&target_path, default_scene_template()).map_err(|err| err.to_string())
        }
        AssetCreateKind::Material => {
            fs::write(&target_path, default_material_template()).map_err(|err| err.to_string())
        }
        AssetCreateKind::Script => {
            fs::write(&target_path, default_script_template_full()).map_err(|err| err.to_string())
        }
        AssetCreateKind::VisualScript => {
            fs::write(&target_path, default_visual_script_template_full())
                .map_err(|err| err.to_string())
        }
        AssetCreateKind::RustScript => (|| -> Result<(), String> {
            let crate_name = sanitize_rust_crate_name(name);
            let src_dir = target_path.join("src");
            fs::create_dir_all(&src_dir).map_err(|err| err.to_string())?;
            let project_root = world
                .get_resource::<EditorProject>()
                .and_then(|project| project.root.clone());
            let sdk_path = rust_script_sdk_dependency_path(project_root.as_deref(), &target_path)?;
            fs::write(
                target_path.join("Cargo.toml"),
                rust_script_manifest_template(&crate_name, &sdk_path),
            )
            .map_err(|err| err.to_string())?;
            fs::write(src_dir.join("lib.rs"), default_rust_script_template_full())
                .map_err(|err| err.to_string())
        })(),
        AssetCreateKind::Animation => {
            fs::write(&target_path, default_animation_template()).map_err(|err| err.to_string())
        }
    };

    match result {
        Ok(()) => {
            if let Some(mut asset_state) = world.get_resource_mut::<AssetBrowserState>() {
                asset_state.refresh_requested = true;
            }
            set_status(world, format!("Created {}", target_path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to create asset: {}", err));
        }
    }
}

fn handle_set_active_camera(world: &mut World, entity: Entity) -> bool {
    if world.get::<BevyCamera>(entity).is_none() {
        set_status(world, "Selected entity has no camera".to_string());
        return false;
    }

    if world.get::<EditorViewportCamera>(entity).is_some() {
        set_status(
            world,
            "Viewport camera is managed by the editor".to_string(),
        );
        return false;
    }

    set_play_camera(world, entity);
    if let Some(mut viewport_state) = world.get_resource_mut::<EditorViewportState>() {
        viewport_state.play_mode_view = PlayViewportKind::Gameplay;
    }

    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);
    if world_state == WorldState::Play {
        activate_play_camera(world);
    }

    set_status(world, "Game camera updated".to_string());
    true
}

fn apply_viewport_graph(world: &mut World) {
    let template_name = world
        .get_resource::<EditorViewportState>()
        .map(|state| state.graph_template.clone());
    let Some(template_name) = template_name else {
        return;
    };
    let Some(template) = template_for_graph(&template_name) else {
        return;
    };
    if let Some(mut graph_res) = world.get_resource_mut::<RenderGraphResource>() {
        graph_res.0 = (template.build)();
    }
}

fn has_camera(world: &World, entity: Entity) -> bool {
    world.get::<BevyCamera>(entity).is_some() && world.get::<BevyTransform>(entity).is_some()
}

fn viewport_gizmo_options(
    world_state: WorldState,
    gizmos_in_play: bool,
    show_camera_gizmos: bool,
    show_directional_light_gizmos: bool,
    show_point_light_gizmos: bool,
    show_spot_light_gizmos: bool,
) -> RenderViewportGizmoOptions {
    RenderViewportGizmoOptions {
        show_gizmos: world_state == WorldState::Edit || gizmos_in_play,
        show_camera_gizmos,
        show_directional_light_gizmos,
        show_point_light_gizmos,
        show_spot_light_gizmos,
    }
}

fn viewport_gizmo_options_from_state(
    world_state: WorldState,
    viewport_state: &EditorViewportState,
) -> RenderViewportGizmoOptions {
    viewport_gizmo_options(
        world_state,
        viewport_state.gizmos_in_play,
        viewport_state.show_camera_gizmos,
        viewport_state.show_directional_light_gizmos,
        viewport_state.show_point_light_gizmos,
        viewport_state.show_spot_light_gizmos,
    )
}

fn viewport_request_for_entity(
    world: &World,
    entity: Entity,
    id: u64,
    texture_id: egui::TextureId,
    viewport_rect: ViewportRectPixels,
    target_size_override: Option<[u32; 2]>,
    temporal_history: bool,
    immediate_resize: bool,
    graph_template: Option<String>,
    gizmo_options: RenderViewportGizmoOptions,
) -> Option<RenderViewportRequest> {
    let transform = world.get::<BevyTransform>(entity)?.0;
    let mut camera = world.get::<BevyCamera>(entity)?.0;
    let target_size = target_size_override.unwrap_or_else(|| viewport_rect.target_size());
    let aspect_ratio = target_size[0].max(1) as f32 / target_size[1].max(1) as f32;
    if aspect_ratio.is_finite() && aspect_ratio > 0.0 {
        camera.aspect_ratio = aspect_ratio;
    }
    Some(RenderViewportRequest {
        id,
        camera_transform: transform,
        camera_component: camera,
        egui_texture_id: texture_id,
        target_size,
        temporal_history,
        immediate_resize,
        graph_template,
        gizmo_options,
    })
}

pub fn editor_viewport_camera_mode_system(world: &mut World) {
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);
    let play_mode = world
        .get_resource::<EditorViewportState>()
        .map(|state| state.play_mode_view)
        .unwrap_or(PlayViewportKind::Editor);
    let runtime_camera = world
        .get_resource::<EditorViewportRuntime>()
        .map(|runtime| {
            let active_pane_camera = runtime.active_pane_id.and_then(|pane_id| {
                runtime
                    .pane_requests
                    .iter()
                    .find(|pane| pane.pane_id == pane_id)
                    .map(|pane| pane.camera_entity)
            });
            active_pane_camera
                .or(runtime.active_camera_entity)
                .or_else(|| runtime.pane_requests.first().map(|pane| pane.camera_entity))
        })
        .flatten()
        .filter(|entity| has_camera(world, *entity));

    let viewport_entity = ensure_viewport_camera(world);
    let desired = match world_state {
        WorldState::Edit => runtime_camera.unwrap_or(viewport_entity),
        WorldState::Play => {
            if let Some(entity) = runtime_camera {
                entity
            } else if play_mode == PlayViewportKind::Gameplay {
                ensure_play_camera(world).unwrap_or(viewport_entity)
            } else {
                viewport_entity
            }
        }
    };
    let desired = if has_camera(world, desired) {
        desired
    } else {
        viewport_entity
    };

    let current = world
        .query::<(Entity, &helmer_becs::BevyActiveCamera)>()
        .iter(world)
        .next()
        .map(|(entity, _)| entity);
    if current == Some(desired) {
        return;
    }

    if let Some(viewport_camera) = world.get::<EditorViewportCamera>(desired).copied() {
        activate_viewport_camera_for_pane(world, viewport_camera.pane_id);
    } else {
        set_play_camera(world, desired);
        activate_play_camera(world);
    }
}

pub fn editor_viewport_render_requests_system(world: &mut World) {
    let runtime = world
        .get_resource::<EditorViewportRuntime>()
        .cloned()
        .unwrap_or_default();
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);
    let viewport_state = world
        .get_resource::<EditorViewportState>()
        .cloned()
        .unwrap_or_default();
    let default_graph_template = (!viewport_state.graph_template.is_empty())
        .then_some(viewport_state.graph_template.clone());
    let default_gizmo_options = viewport_gizmo_options_from_state(world_state, &viewport_state);

    if !runtime.pane_requests.is_empty() {
        let active_pane_id = runtime.active_pane_id;
        let mut pane_requests = runtime.pane_requests;
        if let Some(active_pane_id) = active_pane_id {
            if let Some(index) = pane_requests
                .iter()
                .position(|pane| pane.pane_id == active_pane_id)
            {
                pane_requests.swap(0, index);
            }
        }

        let mut requests = Vec::new();
        let active_pane_request = pane_requests.first();
        let main_display_entity = active_pane_request
            .map(|pane| pane.camera_entity)
            .or_else(|| pane_requests.first().map(|pane| pane.camera_entity));
        let preview_graph_template = active_pane_request
            .and_then(|pane| {
                (!pane.graph_template.is_empty()).then_some(pane.graph_template.clone())
            })
            .or_else(|| default_graph_template.clone());
        let preview_gizmo_options = active_pane_request
            .map(|pane| {
                viewport_gizmo_options(
                    world_state,
                    pane.gizmos_in_play,
                    pane.show_camera_gizmos,
                    pane.show_directional_light_gizmos,
                    pane.show_point_light_gizmos,
                    pane.show_spot_light_gizmos,
                )
            })
            .unwrap_or(default_gizmo_options);
        let preview_entity = runtime.preview_camera_entity;
        let preview_texture_id = runtime.preview_texture_id;
        let preview_rect = runtime.preview_rect_pixels;
        for pane in pane_requests {
            let graph_template = (!pane.graph_template.is_empty()).then_some(pane.graph_template);
            let gizmo_options = viewport_gizmo_options(
                world_state,
                pane.gizmos_in_play,
                pane.show_camera_gizmos,
                pane.show_directional_light_gizmos,
                pane.show_point_light_gizmos,
                pane.show_spot_light_gizmos,
            );
            if let Some(request) = viewport_request_for_entity(
                world,
                pane.camera_entity,
                pane.pane_id,
                pane.texture_id,
                pane.viewport_rect,
                Some(pane.target_size),
                pane.temporal_history,
                pane.immediate_resize,
                graph_template,
                gizmo_options,
            ) {
                requests.push(request);
            }
        }
        if let (Some(preview_entity), Some(texture_id), Some(preview_rect)) =
            (preview_entity, preview_texture_id, preview_rect)
        {
            if Some(preview_entity) != main_display_entity && has_camera(world, preview_entity) {
                if let Some(request) = viewport_request_for_entity(
                    world,
                    preview_entity,
                    VIEWPORT_ID_PREVIEW,
                    texture_id,
                    preview_rect,
                    None,
                    false,
                    false,
                    preview_graph_template,
                    preview_gizmo_options,
                ) {
                    requests.push(request);
                }
            }
        }
        if let Some(mut viewport_requests) = world.get_resource_mut::<RenderViewportRequests>() {
            viewport_requests.0 = requests;
        }
        return;
    }

    let play_mode = viewport_state.play_mode_view;

    let mut requests = Vec::new();
    let main_rect = runtime.main_rect_pixels;
    if let Some(main_rect) = main_rect {
        let editor_entity = ensure_viewport_camera(world);
        let play_entity = if world_state == WorldState::Play {
            ensure_play_camera(world)
        } else {
            None
        };
        let main_target_size = runtime
            .main_target_size
            .unwrap_or_else(|| main_rect.target_size());

        let wants_gameplay_main =
            world_state == WorldState::Play && play_mode == PlayViewportKind::Gameplay;
        let main_view = if wants_gameplay_main {
            match (play_entity, runtime.gameplay_texture_id) {
                (Some(entity), Some(texture_id)) => {
                    Some((entity, texture_id, VIEWPORT_ID_GAMEPLAY))
                }
                _ => runtime
                    .editor_texture_id
                    .map(|texture_id| (editor_entity, texture_id, VIEWPORT_ID_EDITOR)),
            }
        } else {
            runtime
                .editor_texture_id
                .map(|texture_id| (editor_entity, texture_id, VIEWPORT_ID_EDITOR))
                .or_else(|| {
                    if world_state == WorldState::Play {
                        match (play_entity, runtime.gameplay_texture_id) {
                            (Some(entity), Some(texture_id)) => {
                                Some((entity, texture_id, VIEWPORT_ID_GAMEPLAY))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
        };

        let main_display_entity = if let Some((main_entity, texture_id, viewport_id)) = main_view {
            if let Some(request) = viewport_request_for_entity(
                world,
                main_entity,
                viewport_id,
                texture_id,
                main_rect,
                Some(main_target_size),
                true,
                runtime.main_resize_immediate,
                default_graph_template.clone(),
                default_gizmo_options,
            ) {
                requests.push(request);
            }
            main_entity
        } else {
            editor_entity
        };

        if let (Some(preview_entity), Some(texture_id), Some(preview_rect)) = (
            runtime.preview_camera_entity,
            runtime.preview_texture_id,
            runtime.preview_rect_pixels,
        ) {
            if preview_entity != main_display_entity && has_camera(world, preview_entity) {
                if let Some(request) = viewport_request_for_entity(
                    world,
                    preview_entity,
                    VIEWPORT_ID_PREVIEW,
                    texture_id,
                    preview_rect,
                    None,
                    false,
                    false,
                    default_graph_template.clone(),
                    default_gizmo_options,
                ) {
                    requests.push(request);
                }
            }
        }
    }

    if let Some(mut viewport_requests) = world.get_resource_mut::<RenderViewportRequests>() {
        viewport_requests.0 = requests;
    }
}

fn handle_toggle_play(world: &mut World) {
    let state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);

    match state {
        WorldState::Edit => {
            if let Some(audio) = world.get_resource::<AudioBackendResource>() {
                audio.0.clear_emitters();
            }
            let project = world
                .get_resource::<EditorProject>()
                .cloned()
                .expect("Project missing");
            let (document, entity_order) = serialize_scene(world, &project);

            let selected_entity = world
                .get_resource::<InspectorSelectedEntityResource>()
                .and_then(|selection| selection.0);
            let selection_index = selected_entity
                .and_then(|entity| entity_order.iter().position(|ordered| *ordered == entity));

            if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
                scene_state.play_backup = Some(document);
                scene_state.play_selected_index = selection_index;
                scene_state.world_state = WorldState::Play;
            }
            activate_play_camera(world);
            set_viewport_audio_listener_enabled(world, false);
            apply_viewport_graph(world);
            spawn_play_viewport_pane(world);

            set_status(world, "Play mode".to_string());
        }
        WorldState::Play => {
            if let Some(audio) = world.get_resource::<AudioBackendResource>() {
                audio.0.clear_emitters();
            }
            if let Some(mut phys_res) = world.get_resource_mut::<PhysicsResource>() {
                phys_res.running = false;
            }

            let (backup, selection_index) =
                if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
                    scene_state.world_state = WorldState::Edit;
                    (
                        scene_state.play_backup.take(),
                        scene_state.play_selected_index.take(),
                    )
                } else {
                    (None, None)
                };

            if let Some(document) = backup {
                let project_snapshot = world
                    .get_resource::<EditorProject>()
                    .cloned()
                    .expect("Project missing");
                reset_editor_scene(world);

                let created_entities =
                    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
                        let asset_server = {
                            let asset_server = world
                                .get_resource::<helmer_becs::BevyAssetServer>()
                                .expect("AssetServer missing");
                            helmer_becs::BevyAssetServer(asset_server.0.clone())
                        };
                        spawn_scene_from_document(
                            world,
                            &document,
                            &project_snapshot,
                            &mut cache,
                            &asset_server,
                        )
                    });
                restore_scene_transforms_from_document(world, &document, &created_entities);

                if let Some(index) = selection_index {
                    if let Some(&entity) = created_entities.get(index) {
                        if let Some(mut selection) =
                            world.get_resource_mut::<InspectorSelectedEntityResource>()
                        {
                            selection.0 = Some(entity);
                        }
                    }
                }
            }

            if let Some(mut render_sync) = world.get_resource_mut::<RenderSyncRequest>() {
                render_sync.request_with_epoch(3);
            }
            if let Some(mut refresh) = world.get_resource_mut::<EditorRenderRefresh>() {
                refresh.pending = true;
            }

            activate_viewport_camera(world);
            set_viewport_audio_listener_enabled(world, true);
            apply_viewport_graph(world);

            set_status(world, "Edit mode".to_string());
        }
    }
}

fn spawn_primitive(world: &mut World, kind: PrimitiveKind) {
    let project = world
        .get_resource::<EditorProject>()
        .cloned()
        .expect("Project missing");

    let material_path = project
        .config
        .as_ref()
        .and_then(|config| {
            project.root.as_ref().map(|root| {
                config
                    .materials_root(root)
                    .join("default.ron")
                    .strip_prefix(root)
                    .ok()
                    .map(|path| path.to_string_lossy().replace('\\', "/"))
            })
        })
        .flatten();

    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = world
            .get_resource::<helmer_becs::BevyAssetServer>()
            .expect("AssetServer missing");
        let material_handle = ensure_default_material(&project, &mut cache, asset_server);
        let mesh_handle = load_primitive_mesh(kind, &mut cache, asset_server);

        let Some(material_handle) = material_handle else {
            set_status(world, "Default material missing".to_string());
            return;
        };

        world.spawn((
            EditorEntity,
            BevyTransform::default(),
            BevyWrapper(MeshRenderer::new(
                mesh_handle.id,
                material_handle.id,
                true,
                true,
            )),
            EditorMesh {
                source: MeshSource::Primitive(kind),
                material_path,
            },
        ));
    });
}

fn spawn_scene_asset(world: &mut World, path: &Path) {
    let asset_server = world
        .get_resource::<helmer_becs::BevyAssetServer>()
        .map(|server| helmer_becs::BevyAssetServer(server.0.clone()))
        .expect("AssetServer missing");
    let handle = if let Some(mut cache) = world.get_resource_mut::<EditorAssetCache>() {
        cached_scene_handle(&mut cache, &asset_server, path)
    } else {
        asset_server.0.lock().load_scene(path)
    };

    world.spawn((
        EditorEntity,
        BevyTransform::default(),
        SceneRoot(handle),
        SceneAssetPath {
            path: path.to_path_buf(),
        },
    ));
}

fn spawn_mesh_asset(world: &mut World, path: &Path) {
    let project = world
        .get_resource::<EditorProject>()
        .cloned()
        .expect("Project missing");

    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = world
            .get_resource::<helmer_becs::BevyAssetServer>()
            .expect("AssetServer missing");
        let material_handle = ensure_default_material(&project, &mut cache, asset_server);
        let mesh_handle = load_mesh_asset(path, &mut cache, asset_server);

        let Some(material_handle) = material_handle else {
            set_status(world, "Default material missing".to_string());
            return;
        };

        world.spawn((
            EditorEntity,
            BevyTransform::default(),
            BevyWrapper(MeshRenderer::new(
                mesh_handle.id,
                material_handle.id,
                true,
                true,
            )),
            EditorMesh {
                source: MeshSource::Asset {
                    path: path.to_string_lossy().replace('\\', "/"),
                },
                material_path: None,
            },
        ));
    });
}

fn ensure_default_material(
    project: &EditorProject,
    cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Option<Handle<Material>> {
    if let Some(handle) = cache.default_material {
        return Some(handle);
    }

    let root = project.root.as_ref()?;
    let config = project.config.as_ref()?;
    let default_path = config.materials_root(root).join("default.ron");
    let handle = asset_server.0.lock().load_material(&default_path);

    let relative = default_path
        .strip_prefix(root)
        .ok()
        .map(|path| path.to_string_lossy().replace('\\', "/"));

    cache.default_material = Some(handle);
    if let Some(relative) = relative {
        cache.material_handles.insert(relative, handle);
    }

    Some(handle)
}

fn load_mesh_asset(
    path: &Path,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Handle<Mesh> {
    let key = path.to_string_lossy().replace('\\', "/");
    if let Some(handle) = asset_cache.mesh_handles.get(&key).copied() {
        return handle;
    }

    let handle = asset_server.0.lock().load_mesh(path);
    asset_cache.mesh_handles.insert(key, handle);
    handle
}

fn load_primitive_mesh(
    kind: PrimitiveKind,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Handle<Mesh> {
    if let Some(handle) = asset_cache.primitive_meshes.get(&kind).copied() {
        return handle;
    }

    let mesh_asset = match kind {
        PrimitiveKind::Cube => helmer::provided::components::MeshAsset::cube("cube".to_string()),
        PrimitiveKind::UvSphere(segments, rings) => {
            helmer::provided::components::MeshAsset::uv_sphere(
                "uv sphere".to_string(),
                segments,
                rings,
            )
        }
        PrimitiveKind::Plane => helmer::provided::components::MeshAsset::plane("plane".to_string()),
    };

    let handle = asset_server
        .0
        .lock()
        .add_mesh(mesh_asset.vertices.unwrap(), mesh_asset.indices);
    asset_cache.primitive_meshes.insert(kind, handle);
    handle
}

fn guess_import_dir(project: &EditorProject, root: &Path, source: &Path) -> Option<PathBuf> {
    let config = project.config.as_ref()?;
    let ext = source
        .extension()
        .and_then(|ext| ext.to_str())?
        .to_ascii_lowercase();
    let file_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase())
        .unwrap_or_default();

    let target = match ext.as_str() {
        "glb" | "gltf" => config.models_root(root),
        "ktx2" | "png" | "jpg" | "jpeg" | "tga" => config.textures_root(root),
        "lua" | "luau" | "hvs" | "rs" => config.scripts_root(root),
        "toml" if file_name == "cargo.toml" => config.scripts_root(root),
        "ron" => config.materials_root(root),
        _ => config.assets_root(root),
    };

    Some(target)
}

fn unique_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }

    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("file");
    let extension = path.extension().and_then(|ext| ext.to_str());
    let parent = path.parent().unwrap_or_else(|| Path::new("."));

    for idx in 1..=999u32 {
        let file_name = match extension {
            Some(ext) => format!("{}_{}.{}", stem, idx, ext),
            None => format!("{}_{}", stem, idx),
        };
        let candidate = parent.join(file_name);
        if !candidate.exists() {
            return candidate;
        }
    }

    path.to_path_buf()
}

fn scene_display_name(path: &Path) -> String {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Scene");
    if let Some(stripped) = file_name.strip_suffix(".hscene.ron") {
        return stripped.to_string();
    }
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("Scene")
        .to_string()
}

fn set_status(world: &mut World, message: String) {
    if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
        state.status = Some(message.clone());
    }
    crate::editor::push_console_status(world, message);
}

//================================================================================
// Freecam System
//================================================================================

#[derive(Default)]
pub struct FreecamState {
    speed: f32,
    sensitivity: f32,
    look_smoothing: f32,
    fov_lerp_speed: f32,
    is_looking: bool,
    look_start_cursor_position: Option<DVec2>,
    look_pane_id: Option<u64>,
    last_cursor_position: DVec2,
    smoothed_cursor_delta: DVec2,
    current_fov_multiplier: f32,
    active_entity: Option<Entity>,
}

#[derive(Component, Default)]
pub struct Freecam;

pub fn freecam_system(
    mut state: bevy_ecs::system::Local<FreecamState>,
    input: Res<BevyInputManager>,
    egui_res: Res<EguiResource>,
    time: Res<DeltaTime>,
    gizmo_state: Res<EditorGizmoState>,
    scene_state: Res<EditorSceneState>,
    viewport_state: Res<EditorViewportState>,
    pane_viewport_state: Option<Res<EditorPaneViewportState>>,
    viewport_runtime: Res<EditorViewportRuntime>,
    runtime_cursor_state: Option<Res<BevyRuntimeCursorState>>,
    mut cursor_control_state: ResMut<EditorCursorControlState>,
    mut viewport_query: bevy_ecs::prelude::Query<
        (Entity, &mut BevyTransform, &mut BevyCamera),
        (
            bevy_ecs::prelude::With<EditorViewportCamera>,
            bevy_ecs::prelude::Without<EditorPlayCamera>,
        ),
    >,
    mut play_query: bevy_ecs::prelude::Query<
        (Entity, &mut BevyTransform, &mut BevyCamera),
        (
            bevy_ecs::prelude::With<EditorPlayCamera>,
            bevy_ecs::prelude::With<Freecam>,
            bevy_ecs::prelude::Without<EditorViewportCamera>,
        ),
    >,
    viewport_camera_candidates: bevy_ecs::prelude::Query<
        Entity,
        (
            bevy_ecs::prelude::With<EditorViewportCamera>,
            bevy_ecs::prelude::Without<EditorPlayCamera>,
        ),
    >,
    play_freecam_camera_candidates: bevy_ecs::prelude::Query<
        Entity,
        (
            bevy_ecs::prelude::With<EditorPlayCamera>,
            bevy_ecs::prelude::With<Freecam>,
            bevy_ecs::prelude::Without<EditorViewportCamera>,
        ),
    >,
) {
    if state.speed == 0.0 {
        state.speed = 1.0;
        state.sensitivity = viewport_state
            .freecam_sensitivity
            .clamp(FREECAM_SENSITIVITY_MIN, FREECAM_SENSITIVITY_MAX);
        state.look_smoothing = viewport_state
            .freecam_smoothing
            .clamp(FREECAM_SMOOTHING_MIN, FREECAM_SMOOTHING_MAX);
        state.fov_lerp_speed = 8.0;
        state.current_fov_multiplier = 1.0;
    }

    let dt = time.0;
    let input_manager = &input.0.read();
    let raw_mouse_delta = input_manager.mouse_motion;
    let egui_pixels_per_point = egui_res.ctx.pixels_per_point() as f64;
    let egui_cursor_position = egui_res.ctx.input(|input| {
        input
            .pointer
            .interact_pos()
            .or_else(|| input.pointer.hover_pos())
            .map(|pos| {
                DVec2::new(
                    pos.x as f64 * egui_pixels_per_point,
                    pos.y as f64 * egui_pixels_per_point,
                )
            })
    });
    let cursor_position = if state.is_looking {
        input_manager.cursor_position
    } else {
        egui_cursor_position.unwrap_or(input_manager.cursor_position)
    };
    let right_mouse_down = input_manager.is_mouse_button_active(MouseButton::Right)
        || egui_res
            .ctx
            .input(|input| input.pointer.button_down(egui::PointerButton::Secondary));
    let wants_pointer = input_manager.egui_wants_pointer;
    let gizmo_blocking = gizmo_state.is_drag_active();
    let hovered_pane_request = viewport_runtime
        .pane_requests
        .iter()
        .find(|pane| pane.pointer_over)
        .or_else(|| {
            viewport_runtime
                .pane_requests
                .iter()
                .find(|pane| pane.viewport_rect.contains(cursor_position))
        });
    let active_pane_request = viewport_runtime.active_pane_id.and_then(|pane_id| {
        viewport_runtime
            .pane_requests
            .iter()
            .find(|pane| pane.pane_id == pane_id)
    });
    let look_pane_request = state.look_pane_id.and_then(|pane_id| {
        viewport_runtime
            .pane_requests
            .iter()
            .find(|pane| pane.pane_id == pane_id)
    });
    let target_pane_request = look_pane_request
        .or(active_pane_request)
        .or(hovered_pane_request);
    let active_camera_entity = look_pane_request
        .map(|pane| pane.camera_entity)
        .or_else(|| active_pane_request.map(|pane| pane.camera_entity))
        .or_else(|| hovered_pane_request.map(|pane| pane.camera_entity))
        .or(viewport_runtime.active_camera_entity)
        .or_else(|| {
            viewport_runtime
                .pane_requests
                .first()
                .map(|pane| pane.camera_entity)
        });
    let can_control_active_camera = active_camera_entity.is_some_and(|entity| {
        viewport_camera_candidates.get(entity).is_ok()
            || play_freecam_camera_candidates.get(entity).is_ok()
    });
    let freecam_settings_pane_id = state
        .look_pane_id
        .or_else(|| active_pane_request.map(|pane| pane.pane_id))
        .or_else(|| hovered_pane_request.map(|pane| pane.pane_id))
        .or(viewport_runtime.active_pane_id);
    let mut configured_sensitivity = viewport_state.freecam_sensitivity;
    let mut configured_smoothing = viewport_state.freecam_smoothing;
    if let (Some(pane_id), Some(pane_state)) =
        (freecam_settings_pane_id, pane_viewport_state.as_deref())
    {
        if let Some(pane_settings) = pane_state.settings.get(&pane_id) {
            configured_sensitivity = pane_settings.freecam_sensitivity;
            configured_smoothing = pane_settings.freecam_smoothing;
        }
    }
    state.sensitivity =
        configured_sensitivity.clamp(FREECAM_SENSITIVITY_MIN, FREECAM_SENSITIVITY_MAX);
    state.look_smoothing = configured_smoothing.clamp(FREECAM_SMOOTHING_MIN, FREECAM_SMOOTHING_MAX);
    let pointer_rect = target_pane_request
        .map(|pane| pane.viewport_rect)
        .or(viewport_runtime.main_rect_pixels)
        .or_else(|| {
            viewport_runtime
                .pane_requests
                .first()
                .map(|pane| pane.viewport_rect)
        });
    let pointer_in_viewport = pointer_rect
        .map(|rect| rect.contains(cursor_position))
        .unwrap_or(false);
    let pointer_over_active_viewport = target_pane_request
        .map(|pane| pane.pointer_over)
        .unwrap_or(false)
        || viewport_runtime.pointer_over_main;
    let allow_viewport_look_input = can_control_active_camera
        && (state.is_looking
            || pointer_over_active_viewport
            || (!wants_pointer && pointer_in_viewport));

    const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const BOOST_AMOUNT: f32 = 1.15;
    const CONTROLLER_SENSITIVITY: f32 = 2.0;

    let maybe_gamepad_id = input_manager.first_gamepad_id();
    let was_looking = state.is_looking;

    let mut yaw_delta = 0.0;
    let mut pitch_delta = 0.0;

    if !gizmo_blocking
        && allow_viewport_look_input
        && right_mouse_down
        && (pointer_in_viewport || pointer_over_active_viewport || state.is_looking)
    {
        if !state.is_looking {
            let capture_pane_request = active_pane_request
                .or(hovered_pane_request)
                .or_else(|| viewport_runtime.pane_requests.first());
            state.look_start_cursor_position = Some(cursor_position);
            state.last_cursor_position = cursor_position;
            state.smoothed_cursor_delta = DVec2::ZERO;
            state.look_pane_id = capture_pane_request.map(|pane| pane.pane_id);
            state.is_looking = true;
        } else {
            let mut cursor_delta = raw_mouse_delta;
            if cursor_delta.length_squared() <= f64::EPSILON {
                cursor_delta = cursor_position - state.last_cursor_position;
            }
            state.last_cursor_position = cursor_position;
            let cursor_delta = if cursor_delta.length_squared() <= 0.25 {
                DVec2::ZERO
            } else {
                cursor_delta
            };
            if state.look_smoothing > 0.0 {
                let smoothing_seconds = state.look_smoothing.max(0.0001) as f64;
                let alpha = (1.0 - (-(dt as f64) / smoothing_seconds).exp()).clamp(0.0, 1.0);
                let smoothed_cursor_delta = state.smoothed_cursor_delta;
                state.smoothed_cursor_delta += (cursor_delta - smoothed_cursor_delta) * alpha;
            } else {
                state.smoothed_cursor_delta = cursor_delta;
            }
            let cursor_delta = if state.smoothed_cursor_delta.length_squared() <= 0.01 {
                DVec2::ZERO
            } else {
                state.smoothed_cursor_delta
            };

            yaw_delta -= cursor_delta.x as f32 * state.sensitivity / 100.0;
            pitch_delta += cursor_delta.y as f32 * state.sensitivity / 100.0;
        }
    } else {
        state.is_looking = false;
        state.look_pane_id = None;
        state.smoothed_cursor_delta = DVec2::ZERO;
    }

    if gizmo_blocking {
        state.is_looking = false;
        state.look_pane_id = None;
        state.smoothed_cursor_delta = DVec2::ZERO;
    }
    let restore_cursor_position = if was_looking && !state.is_looking {
        state.look_start_cursor_position.take()
    } else {
        None
    };

    let keyboard_active = state.is_looking && !gizmo_blocking;

    if !gizmo_blocking {
        if let Some(gamepad_id) = maybe_gamepad_id {
            yaw_delta -= input_manager.get_controller_axis(gamepad_id, gilrs::Axis::RightStickX)
                * CONTROLLER_SENSITIVITY
                * dt;
            pitch_delta -= input_manager.get_controller_axis(gamepad_id, gilrs::Axis::RightStickY)
                * CONTROLLER_SENSITIVITY
                * dt;
        }
    }

    if !wants_pointer && keyboard_active && pointer_in_viewport {
        state.speed += input_manager.mouse_wheel.y * 2.0;
    }

    if !gizmo_blocking {
        if let Some(gamepad_id) = maybe_gamepad_id {
            if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::RightTrigger) {
                state.speed += 10.0 * dt;
            }
            if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::LeftTrigger) {
                state.speed -= 10.0 * dt;
            }
        }
    }

    state.speed = state.speed.max(0.5);
    let mut speed = state.speed;

    let mut boost_active = keyboard_active
        && (input_manager.is_key_active(KeyCode::ShiftLeft)
            || input_manager.is_key_active(KeyCode::ShiftRight));

    if !gizmo_blocking {
        if let Some(gamepad_id) = maybe_gamepad_id {
            if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::LeftThumb) {
                boost_active = true;
                speed *= 2.5;
            }
        }
    }

    if boost_active {
        speed *= 2.5;
    }

    let mut apply_freecam =
        |entity: Entity, transform: &mut BevyTransform, camera: &mut BevyCamera| {
            if state.active_entity != Some(entity) {
                state.active_entity = Some(entity);
                state.current_fov_multiplier = 1.0;
                state.is_looking = false;
                state.look_start_cursor_position = None;
                state.look_pane_id = None;
                state.last_cursor_position = cursor_position;
                state.smoothed_cursor_delta = DVec2::ZERO;
            }

            let transform = &mut transform.0;
            let camera = &mut camera.0;

            let (mut yaw, mut pitch) = extract_yaw_pitch(transform.rotation);

            yaw += yaw_delta;
            pitch += pitch_delta;

            pitch = pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);

            let yaw_rot = Quat::from_axis_angle(Vec3::Y, yaw);
            let pitch_rot = Quat::from_axis_angle(Vec3::X, pitch);
            let orientation = yaw_rot * pitch_rot;

            transform.rotation = orientation;

            let forward = orientation * Vec3::Z;
            let right = orientation * -Vec3::X;

            let mut velocity = Vec3::ZERO;

            if keyboard_active {
                for key in &input_manager.active_keys {
                    match key {
                        KeyCode::KeyW => velocity += forward,
                        KeyCode::KeyS => velocity -= forward,
                        KeyCode::KeyA => velocity -= right,
                        KeyCode::KeyD => velocity += right,
                        KeyCode::Space => velocity += Vec3::Y,
                        KeyCode::KeyC => velocity -= Vec3::Y,
                        _ => {}
                    }
                }
            }

            if !gizmo_blocking {
                if let Some(gamepad_id) = maybe_gamepad_id {
                    let lx = input_manager.get_controller_axis(gamepad_id, gilrs::Axis::LeftStickX);
                    let ly = input_manager.get_controller_axis(gamepad_id, gilrs::Axis::LeftStickY);
                    velocity += right * lx;
                    velocity += forward * ly;

                    let up = input_manager.get_right_trigger_value(gamepad_id);
                    let down = input_manager.get_left_trigger_value(gamepad_id);
                    velocity += Vec3::Y * up;
                    velocity -= Vec3::Y * down;
                }
            }

            if let Some(norm_velocity) = velocity.try_normalize() {
                transform.position += norm_velocity * speed * dt;
            }

            let target_multiplier = if boost_active { BOOST_AMOUNT } else { 1.0 };
            let safe_multiplier = state.current_fov_multiplier.clamp(0.01, BOOST_AMOUNT);
            let base_fov = camera.fov_y_rad / safe_multiplier;
            let t = 1.0 - (-state.fov_lerp_speed * dt).exp();
            state.current_fov_multiplier += (target_multiplier - state.current_fov_multiplier) * t;
            camera.fov_y_rad = base_fov * state.current_fov_multiplier;
        };

    if let Some(active_entity) = active_camera_entity {
        if let Ok((entity, mut transform, mut camera)) = viewport_query.get_mut(active_entity) {
            apply_freecam(entity, &mut transform, &mut camera);
            sync_editor_cursor_control(
                &state,
                &mut cursor_control_state,
                runtime_cursor_state.as_deref(),
                restore_cursor_position,
            );
            return;
        }
        if let Ok((entity, mut transform, mut camera)) = play_query.get_mut(active_entity) {
            apply_freecam(entity, &mut transform, &mut camera);
            sync_editor_cursor_control(
                &state,
                &mut cursor_control_state,
                runtime_cursor_state.as_deref(),
                restore_cursor_position,
            );
            return;
        }
    }

    match scene_state.world_state {
        WorldState::Edit => {
            if let Some((entity, mut transform, mut camera)) = viewport_query.iter_mut().next() {
                apply_freecam(entity, &mut transform, &mut camera);
            }
        }
        WorldState::Play => {
            if viewport_state.play_mode_view == PlayViewportKind::Editor {
                if let Some((entity, mut transform, mut camera)) = viewport_query.iter_mut().next()
                {
                    apply_freecam(entity, &mut transform, &mut camera);
                }
            } else if let Some((entity, mut transform, mut camera)) = play_query.iter_mut().next() {
                apply_freecam(entity, &mut transform, &mut camera);
            }
        }
    }

    sync_editor_cursor_control(
        &state,
        &mut cursor_control_state,
        runtime_cursor_state.as_deref(),
        restore_cursor_position,
    );
}

fn extract_yaw_pitch(rot: Quat) -> (f32, f32) {
    let forward = rot * Vec3::Z;
    let yaw = forward.x.atan2(forward.z);
    let pitch = (-forward.y).asin();
    (yaw, pitch)
}

fn sync_editor_cursor_control(
    freecam_state: &FreecamState,
    cursor_control_state: &mut EditorCursorControlState,
    runtime_cursor_state: Option<&BevyRuntimeCursorState>,
    restore_cursor_position: Option<DVec2>,
) {
    cursor_control_state.freecam_capture_active = freecam_state.is_looking;

    if let Some(runtime_cursor_state) = runtime_cursor_state {
        runtime_cursor_state
            .0
            .set(cursor_control_state.effective_policy());
        if !freecam_state.is_looking {
            if let Some(restore_cursor_position) = restore_cursor_position {
                runtime_cursor_state
                    .0
                    .request_warp(restore_cursor_position.x, restore_cursor_position.y);
            }
        }
    }
}
