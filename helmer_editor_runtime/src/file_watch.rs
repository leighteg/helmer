use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        mpsc::{Receiver, TryRecvError, channel},
    },
};

use helmer_becs::ecs::prelude::Resource;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};

use crate::project::ProjectConfig;

pub const MAX_WATCH_EVENTS_PER_TICK: usize = 512;

#[derive(Resource, Default)]
pub struct FileWatchState {
    pub root: Option<PathBuf>,
    pub watcher: Option<Mutex<RecommendedWatcher>>,
    pub receiver: Option<Mutex<Receiver<notify::Result<Event>>>>,
    pub status: Option<String>,
}

#[derive(Debug, Default)]
pub struct FileWatchPollOutput {
    pub assets_changed: bool,
    pub script_paths: HashSet<PathBuf>,
    pub status_update: Option<String>,
}

pub fn configure_file_watcher(
    state: &mut FileWatchState,
    root: &Path,
    config: Option<&ProjectConfig>,
) {
    let (tx, rx) = channel();
    match notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    }) {
        Ok(mut watcher) => {
            let mut watch_roots = Vec::<PathBuf>::new();
            if let Some(config) = config {
                let assets_root = config.assets_root(root);
                let scripts_root = config.scripts_root(root);
                if assets_root.starts_with(&scripts_root) && scripts_root != assets_root {
                    watch_roots.push(scripts_root);
                } else if scripts_root.starts_with(&assets_root) {
                    watch_roots.push(assets_root);
                } else {
                    watch_roots.push(assets_root);
                    if scripts_root != watch_roots[0] {
                        watch_roots.push(scripts_root);
                    }
                }
            }
            if watch_roots.is_empty() {
                watch_roots.push(root.to_path_buf());
            }

            let mut watched_count = 0usize;
            for watch_root in watch_roots {
                if !watch_root.exists() {
                    continue;
                }
                if let Err(err) = watcher.watch(&watch_root, RecursiveMode::Recursive) {
                    state.status = Some(format!("Watcher failed: {}", err));
                    state.watcher = None;
                    state.receiver = None;
                    state.root = None;
                    return;
                }
                watched_count += 1;
            }

            if watched_count == 0 {
                state.status = Some("Watcher failed: no valid roots to watch".to_string());
                state.watcher = None;
                state.receiver = None;
                state.root = None;
                return;
            }

            if let Err(err) = watcher.watch(
                &ProjectConfig::config_path(root),
                RecursiveMode::NonRecursive,
            ) {
                state.status = Some(format!("Watcher failed: {}", err));
                state.watcher = None;
                state.receiver = None;
                state.root = None;
                return;
            }

            state.root = Some(root.to_path_buf());
            state.watcher = Some(Mutex::new(watcher));
            state.receiver = Some(Mutex::new(rx));
            state.status = Some(format!("File watcher active ({} roots)", watched_count + 1));
        }
        Err(err) => {
            state.status = Some(format!("Watcher init failed: {}", err));
            state.watcher = None;
            state.receiver = None;
            state.root = None;
        }
    }
}

pub fn poll_file_watcher(
    state: &mut FileWatchState,
    assets_root: &Path,
    scripts_root: &Path,
    config_path: &Path,
) -> FileWatchPollOutput {
    let Some(receiver) = state.receiver.as_ref() else {
        return FileWatchPollOutput::default();
    };
    let Ok(receiver) = receiver.lock() else {
        return FileWatchPollOutput::default();
    };

    let mut output = FileWatchPollOutput::default();
    let mut watcher_disconnected = false;
    let mut events_processed = 0usize;

    loop {
        if events_processed >= MAX_WATCH_EVENTS_PER_TICK {
            output.status_update =
                Some("File watcher backlog detected; draining incrementally".to_string());
            break;
        }

        let event = match receiver.try_recv() {
            Ok(event) => event,
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                output.status_update = Some("Watcher disconnected".to_string());
                watcher_disconnected = true;
                break;
            }
        };
        events_processed += 1;

        match event {
            Ok(event) => {
                for path in event.paths {
                    if path_has_ignored_component(&path) {
                        continue;
                    }
                    if path == config_path {
                        output.assets_changed = true;
                        continue;
                    }
                    let in_assets_root = path.starts_with(assets_root);
                    let in_scripts_root = path.starts_with(scripts_root);
                    if in_assets_root || in_scripts_root {
                        output.assets_changed = true;
                    }
                    if in_scripts_root && path_might_affect_script_registry(&path) {
                        output.script_paths.insert(path);
                    }
                }
            }
            Err(err) => {
                output.status_update = Some(format!("Watcher error: {}", err));
            }
        }
    }

    drop(receiver);

    if watcher_disconnected {
        state.watcher = None;
        state.receiver = None;
    }

    if let Some(status) = output.status_update.as_ref() {
        state.status = Some(status.clone());
    }

    output
}

fn path_has_ignored_component(path: &Path) -> bool {
    path.components().any(|component| {
        let std::path::Component::Normal(part) = component else {
            return false;
        };
        let Some(name) = part.to_str() else {
            return false;
        };
        name.starts_with('.') || is_ignored_watch_component(name)
    })
}

fn is_ignored_watch_component(name: &str) -> bool {
    name.eq_ignore_ascii_case("target")
        || name.eq_ignore_ascii_case(".git")
        || name.eq_ignore_ascii_case(".helmer")
        || name.eq_ignore_ascii_case("node_modules")
}

fn path_might_affect_script_registry(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if file_name.eq_ignore_ascii_case("cargo.toml") || file_name.eq_ignore_ascii_case("build.rs") {
        return true;
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => {
            ext.eq_ignore_ascii_case("lua")
                || ext.eq_ignore_ascii_case("luau")
                || ext.eq_ignore_ascii_case("hvs")
                || ext.eq_ignore_ascii_case("rs")
        }
        None => true,
    }
}
