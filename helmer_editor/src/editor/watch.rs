use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        mpsc::{Receiver, channel},
    },
};

use bevy_ecs::prelude::{Res, ResMut, Resource};
use helmer_becs::BevySystemProfiler;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};

use crate::editor::{assets::AssetBrowserState, scripting::ScriptRegistry};

#[derive(Resource, Default)]
pub struct FileWatchState {
    pub root: Option<PathBuf>,
    pub watcher: Option<Mutex<RecommendedWatcher>>,
    pub receiver: Option<Mutex<Receiver<notify::Result<Event>>>>,
    pub pending_paths: HashSet<PathBuf>,
    pub status: Option<String>,
}

pub fn configure_file_watcher(state: &mut FileWatchState, root: &Path) {
    let (tx, rx) = channel();
    match notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    }) {
        Ok(mut watcher) => {
            if let Err(err) = watcher.watch(root, RecursiveMode::Recursive) {
                state.status = Some(format!("Watcher failed: {}", err));
                state.watcher = None;
                state.receiver = None;
                state.root = None;
                return;
            }

            state.root = Some(root.to_path_buf());
            state.watcher = Some(Mutex::new(watcher));
            state.receiver = Some(Mutex::new(rx));
            state.status = Some("File watcher active".to_string());
        }
        Err(err) => {
            state.status = Some(format!("Watcher init failed: {}", err));
            state.watcher = None;
            state.receiver = None;
            state.root = None;
        }
    }
}

pub fn file_watch_system(
    mut state: ResMut<FileWatchState>,
    mut assets: ResMut<AssetBrowserState>,
    mut scripts: ResMut<ScriptRegistry>,
    system_profiler: Option<Res<BevySystemProfiler>>,
) {
    let _system_scope = system_profiler.as_ref().and_then(|profiler| {
        profiler
            .0
            .begin_scope("helmer_editor::editor::file_watch_system")
    });

    let events = {
        let Some(receiver) = state.receiver.as_ref() else {
            return;
        };
        let Ok(receiver) = receiver.lock() else {
            return;
        };
        receiver.try_iter().collect::<Vec<_>>()
    };

    for event in events {
        match event {
            Ok(event) => {
                for path in event.paths {
                    state.pending_paths.insert(path);
                }
            }
            Err(err) => {
                state.status = Some(format!("Watcher error: {}", err));
            }
        }
    }

    if state.pending_paths.is_empty() {
        return;
    }

    assets.refresh_requested = true;
    scripts.mark_dirty_paths(&state.pending_paths);
    state.pending_paths.clear();
}
