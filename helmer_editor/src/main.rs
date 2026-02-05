use std::{env, path::PathBuf, sync::OnceLock};

use bevy_ecs::schedule::IntoScheduleConfigs;
use helmer::graphics::render_graphs::template_for_graph;
use helmer::runtime::asset_server::AssetServer;
use helmer_becs::systems::render_system::RenderGizmoState;
use helmer_becs::{AudioBackendResource, egui_integration::EguiResource, helmer_becs_init};

use helmer_editor::editor::{
    AnimatorUiState, AssetBrowserState, AssetDragState, EditorAssetCache, EditorAudioDeviceCache,
    EditorCommand, EditorCommandQueue, EditorGizmoSettings, EditorGizmoState,
    EditorMeshOutlineCache, EditorPaneAutoState, EditorPaneManagerState, EditorPaneVisibility,
    EditorProject, EditorRenderRefresh, EditorSceneState, EditorSelectionState, EditorSplineState,
    EditorTimelineState, EditorUiState, EditorUndoState, EditorViewportState, EditorWorkspaceState,
    EntityDragState, FileWatchState, HierarchyUiState, InspectorNameEditState,
    InspectorPinnedEntityResource, MaterialEditorCache, MiddleDragUiState,
    PendingSceneChildAnimations, PendingSceneChildPoseOverrides, PoseEditorState, ScriptInputState,
    ScriptRegistry, ScriptRunState, ScriptRuntime, activate_viewport_camera,
    apply_scene_child_animations_system, apply_scene_child_pose_overrides_system,
    asset_scan_system, drag_drop_system, editor_command_system, editor_layout_apply_system,
    editor_layout_save_system, editor_layout_update_system, editor_physics_state_system,
    editor_render_refresh_system, editor_shortcut_system, editor_ui_system,
    editor_undo_request_system, file_watch_system, freecam_system, gizmo_system, load_layout_state,
    load_recent_projects, pane_manager_toggle_system, pending_skinned_mesh_system,
    scene_dirty_system, script_execution_system, script_registry_system, selection_system,
    set_viewport_audio_listener_enabled, timeline_playback_system,
};

static PROJECT_ARG: OnceLock<Option<PathBuf>> = OnceLock::new();
static DEFAULT_PROJECTS_ROOT: OnceLock<PathBuf> = OnceLock::new();

fn main() {
    #[cfg(target_os = "linux")]
    unsafe {
        env::set_var("HELMER_FORCE_UNIX_BACKEND", "x11")
    };

    let project_arg = env::args().nth(1).map(PathBuf::from);
    let default_projects_root = env::current_dir()
        .ok()
        .map(|path| path.join("projects"))
        .unwrap_or_else(|| PathBuf::from("./projects"));

    let _ = PROJECT_ARG.set(project_arg);
    let _ = DEFAULT_PROJECTS_ROOT.set(default_projects_root);

    helmer_becs_init(editor_init);
}

fn editor_init(
    world: &mut bevy_ecs::world::World,
    schedule: &mut bevy_ecs::schedule::Schedule,
    _asset_server: &AssetServer,
) {
    world.insert_resource(EditorProject::default());
    world.insert_resource(EditorSceneState::default());
    world.insert_resource(EditorRenderRefresh::default());
    world.insert_resource(EditorUndoState::default());
    world.insert_resource(PendingSceneChildAnimations::default());
    world.insert_resource(PendingSceneChildPoseOverrides::default());
    world.insert_resource(EditorAssetCache::default());
    world.insert_resource(AssetBrowserState::default());
    world.insert_resource(AssetDragState::default());
    world.insert_resource(EntityDragState::default());
    world.insert_resource(MiddleDragUiState::default());
    world.insert_resource(EditorCommandQueue::default());
    world.insert_resource(ScriptRegistry::default());
    world.insert_resource(FileWatchState::default());
    world.insert_resource(ScriptRuntime::default());
    world.insert_resource(ScriptRunState::default());
    world.insert_resource(ScriptInputState::default());
    world.insert_resource(HierarchyUiState::default());
    world.insert_resource(AnimatorUiState::default());
    world.insert_resource(PoseEditorState::default());
    world.insert_resource(MaterialEditorCache::default());
    world.insert_resource(EditorWorkspaceState::default());
    world.insert_resource(load_layout_state());
    world.insert_resource(InspectorNameEditState::default());
    world.insert_resource(InspectorPinnedEntityResource::default());
    world.insert_resource(EditorGizmoState::default());
    world.insert_resource(EditorGizmoSettings::default());
    world.insert_resource(EditorMeshOutlineCache::default());
    world.insert_resource(EditorSelectionState::default());
    world.insert_resource(EditorSplineState::default());
    world.insert_resource(EditorTimelineState::default());
    world.insert_resource(EditorViewportState::default());
    world.insert_resource(RenderGizmoState::default());
    world.insert_resource(EditorPaneVisibility::default());
    world.insert_resource(EditorPaneManagerState::default());
    world.insert_resource(EditorPaneAutoState::default());
    world.insert_resource(EditorAudioDeviceCache::default());

    if let Some(audio) = world.get_resource::<AudioBackendResource>() {
        audio.0.set_enabled(true);
        audio.0.clear_emitters();
    }

    activate_viewport_camera(world);
    set_viewport_audio_listener_enabled(world, true);

    let projects_root = DEFAULT_PROJECTS_ROOT
        .get()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("./projects"));

    world.insert_resource(EditorUiState {
        project_name: "NewProject".to_string(),
        project_path: projects_root.to_string_lossy().into_owned(),
        open_project_path: projects_root.to_string_lossy().into_owned(),
        status: None,
        recent_projects: load_recent_projects(),
    });

    if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
        egui_res.inspector_ui = false;
        egui_res.snap_enabled = true;
        egui_res.snap_distance = 12.0;
    }

    let graph_template = world
        .get_resource::<EditorViewportState>()
        .map(|state| state.graph_template.clone())
        .unwrap_or_else(|| "debug-graph".to_string());
    if let Some(mut graph_res) =
        world.get_resource_mut::<helmer_becs::systems::render_system::RenderGraphResource>()
    {
        if let Some(template) = template_for_graph(&graph_template) {
            graph_res.0 = (template.build)();
        }
    }

    if let Some(path) = PROJECT_ARG.get().and_then(|path| path.clone()) {
        world
            .get_resource_mut::<EditorCommandQueue>()
            .expect("EditorCommandQueue missing")
            .push(EditorCommand::OpenProject { path });
    }

    schedule.add_systems(
        editor_command_system
            .before(helmer_becs::systems::scene_system::scene_spawning_system)
            .before(helmer_becs::systems::animation_system::skinning_system)
            .before(helmer_becs::systems::render_system::render_data_system),
    );
    schedule.add_systems(
        editor_physics_state_system.before(helmer_becs::physics::systems::cleanup_physics_system),
    );
    schedule.add_systems(file_watch_system);
    schedule.add_systems(asset_scan_system);
    schedule.add_systems(drag_drop_system);
    schedule.add_systems(editor_shortcut_system);
    schedule.add_systems(scene_dirty_system);
    schedule.add_systems(
        apply_scene_child_animations_system
            .after(helmer_becs::systems::scene_system::scene_spawning_system),
    );
    schedule.add_systems(
        apply_scene_child_pose_overrides_system
            .after(helmer_becs::systems::scene_system::scene_spawning_system),
    );
    schedule.add_systems(
        pending_skinned_mesh_system
            .after(helmer_becs::systems::scene_system::scene_spawning_system)
            .before(helmer_becs::systems::animation_system::skinning_system)
            .before(helmer_becs::systems::render_system::render_data_system),
    );
    schedule.add_systems(script_registry_system);
    schedule.add_systems(script_execution_system);
    schedule.add_systems(editor_layout_apply_system.before(editor_ui_system));
    schedule.add_systems(editor_ui_system.before(helmer_becs::egui_integration::egui_system));
    schedule
        .add_systems(pane_manager_toggle_system.after(helmer_becs::egui_integration::egui_system));
    schedule
        .add_systems(editor_layout_update_system.after(helmer_becs::egui_integration::egui_system));
    schedule.add_systems(editor_layout_save_system.after(editor_layout_update_system));
    schedule.add_systems(timeline_playback_system);
    schedule
        .add_systems(gizmo_system.before(helmer_becs::systems::render_system::render_data_system));
    schedule.add_systems(
        editor_render_refresh_system.after(helmer_becs::systems::render_system::render_data_system),
    );
    schedule.add_systems(selection_system.after(gizmo_system));
    schedule.add_systems(
        editor_undo_request_system
            .after(selection_system)
            .after(editor_ui_system),
    );
    schedule.add_systems(freecam_system);
}
