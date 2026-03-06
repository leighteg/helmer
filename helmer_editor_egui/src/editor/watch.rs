use bevy_ecs::prelude::{Res, ResMut};
use helmer_becs::BevySystemProfiler;
pub use helmer_editor_runtime::file_watch::{FileWatchState, configure_file_watcher};
use helmer_editor_runtime::{file_watch::poll_file_watcher, project::ProjectConfig};

use crate::editor::{assets::AssetBrowserState, project::EditorProject, scripting::ScriptRegistry};

pub fn file_watch_system(
    mut state: ResMut<FileWatchState>,
    mut assets: ResMut<AssetBrowserState>,
    mut scripts: ResMut<ScriptRegistry>,
    project: Res<EditorProject>,
    system_profiler: Option<Res<BevySystemProfiler>>,
) {
    let _system_scope = system_profiler.as_ref().and_then(|profiler| {
        profiler
            .0
            .begin_scope("helmer_editor_egui::editor::file_watch_system")
    });

    let Some(project_root) = project.root.as_ref() else {
        return;
    };
    let assets_root = project
        .config
        .as_ref()
        .map(|cfg| cfg.assets_root(project_root))
        .unwrap_or_else(|| project_root.join("assets"));
    let scripts_root = project
        .config
        .as_ref()
        .map(|cfg| cfg.scripts_root(project_root))
        .unwrap_or_else(|| assets_root.join("scripts"));
    let config_path = ProjectConfig::config_path(project_root);

    let poll = poll_file_watcher(&mut state, &assets_root, &scripts_root, &config_path);
    if !poll.assets_changed && poll.script_paths.is_empty() {
        return;
    }

    if poll.assets_changed {
        assets.refresh_requested = true;
    }
    if !poll.script_paths.is_empty() {
        scripts.mark_dirty_paths_owned(poll.script_paths);
    }
}
