use std::{env, path::PathBuf, sync::OnceLock};

use bevy_ecs::schedule::IntoScheduleConfigs;
use helmer::runtime::asset_server::AssetServer;
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::render_system::{RenderGizmoState, RenderMainSceneToSwapchain};
use helmer_becs::{AudioBackendResource, BevyRuntimeConfig, helmer_becs_init};
use helmer_editor_runtime::editor_commands::EditorCommand;

mod retained;
use retained::{
    AssetBrowserState, EditorAssetClipboardState, EditorCommandQueue, EditorConsoleState,
    EditorPaneWorkspaceState, EditorProject, EditorProjectLauncherState,
    EditorRetainedDockingState, EditorRetainedGizmoSnapSettings,
    EditorRetainedGraphInteractionState, EditorRetainedGraphRenderer, EditorRetainedGraphState,
    EditorRetainedLayoutCatalog, EditorRetainedLayoutState, EditorRetainedPaneInteractionState,
    EditorRetainedUiMode, EditorRetainedViewportStates, EditorSceneState, EditorTimelineState,
    EditorUndoState, FileWatchState, RetainedCursorControlState, RetainedViewportControlState,
    ScriptRegistry, editor_retained_shell_system, install_runtime_log_listener,
    retained_active_camera_system, retained_asset_browser_refresh_system, retained_command_system,
    retained_console_runtime_log_system, retained_file_watch_system, retained_gizmo_state_system,
    retained_physics_state_system, retained_script_registry_system,
    retained_timeline_playback_system, retained_viewport_camera_controls_system,
    retained_viewport_requests_system, retained_workspace_seed_system,
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

    install_runtime_log_listener();
    helmer_becs_init(editor_init);
}

fn editor_init(
    world: &mut bevy_ecs::world::World,
    schedule: &mut bevy_ecs::schedule::Schedule,
    _asset_server: &AssetServer,
) {
    world.insert_resource(EditorProject::default());
    world.insert_resource(AssetBrowserState::default());
    world.insert_resource(EditorConsoleState::default());
    world.insert_resource(EditorTimelineState::default());
    world.insert_resource(EditorPaneWorkspaceState::default());
    world.insert_resource(FileWatchState::default());
    world.insert_resource(EditorSceneState::default());
    world.insert_resource(EditorUndoState::default());
    world.insert_resource(EditorCommandQueue::default());
    world.insert_resource(EditorAssetClipboardState::default());
    world.insert_resource(ScriptRegistry::default());

    world.insert_resource(EditorRetainedUiMode { enabled: true });
    world.insert_resource(EditorRetainedLayoutState::default());
    world.insert_resource(EditorRetainedLayoutCatalog::default());
    world.insert_resource(EditorRetainedDockingState::default());
    world.insert_resource(EditorRetainedGraphState::default());
    world.insert_resource(EditorRetainedGraphRenderer::default());
    world.insert_resource(EditorRetainedGraphInteractionState::default());
    world.insert_resource(EditorRetainedPaneInteractionState::default());
    world.insert_resource(EditorRetainedViewportStates::default());
    world.insert_resource(EditorRetainedGizmoSnapSettings::default());
    world.insert_resource(RetainedViewportControlState::default());
    world.insert_resource(RetainedCursorControlState::default());

    world.insert_resource(InspectorSelectedEntityResource::default());
    world.insert_resource(RenderGizmoState::default());
    world.insert_resource(RenderMainSceneToSwapchain(false));

    if let Some(mut runtime_config) = world.get_resource_mut::<BevyRuntimeConfig>() {
        runtime_config.0.egui = false;
        runtime_config.0.render_config.ui_pass = true;
        runtime_config.0.render_config.egui_pass = false;
    }

    if let Some(audio) = world.get_resource::<AudioBackendResource>() {
        audio.0.set_enabled(true);
        audio.0.clear_emitters();
    }

    let projects_root = DEFAULT_PROJECTS_ROOT
        .get()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("./projects"));
    world.insert_resource(EditorProjectLauncherState::from_projects_root(
        projects_root,
    ));
    if let Some(path) = PROJECT_ARG.get().and_then(|path| path.clone())
        && let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>()
    {
        queue.push(EditorCommand::OpenProject { path });
    }

    schedule.add_systems(retained_workspace_seed_system.before(editor_retained_shell_system));
    schedule.add_systems(
        retained_active_camera_system
            .before(helmer_becs::systems::render_system::render_data_system),
    );
    schedule.add_systems(
        retained_file_watch_system
            .before(retained_script_registry_system)
            .before(retained_asset_browser_refresh_system),
    );
    schedule.add_systems(retained_script_registry_system.before(editor_retained_shell_system));
    schedule
        .add_systems(retained_asset_browser_refresh_system.before(editor_retained_shell_system));
    schedule.add_systems(retained_console_runtime_log_system.before(editor_retained_shell_system));
    schedule.add_systems(retained_command_system.before(editor_retained_shell_system));
    schedule.add_systems(retained_timeline_playback_system);
    schedule.add_systems(
        retained_physics_state_system.before(helmer_becs::physics::systems::physics_step_system),
    );
    schedule.add_systems(
        retained_viewport_camera_controls_system
            .after(editor_retained_shell_system)
            .before(retained_viewport_requests_system)
            .before(helmer_becs::systems::render_system::render_data_system),
    );
    schedule.add_systems(
        retained_gizmo_state_system
            .after(retained_viewport_camera_controls_system)
            .before(helmer_becs::systems::render_system::render_data_system),
    );
    schedule.add_systems(
        retained_viewport_requests_system
            .after(retained_viewport_camera_controls_system)
            .after(editor_retained_shell_system)
            .before(helmer_becs::systems::render_system::render_data_system),
    );
    schedule.add_systems(
        editor_retained_shell_system
            .before(helmer_becs::systems::render_system::render_data_system)
            .before(helmer_becs::ui_integration::ui_system),
    );
}
