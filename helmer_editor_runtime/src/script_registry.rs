use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime},
};

use helmer_becs::ecs::prelude::Resource;
use walkdir::WalkDir;

use crate::{project::ProjectConfig, scripting as runtime_scripting};

#[derive(Debug, Clone)]
pub struct ScriptAsset {
    pub language: String,
    pub source: String,
    pub modified: SystemTime,
    pub error: Option<String>,
}

#[derive(Resource)]
pub struct ScriptRegistry {
    pub scripts: HashMap<PathBuf, ScriptAsset>,
    pub dirty_paths: HashSet<PathBuf>,
    pub last_scan: Instant,
    pub scan_interval: Duration,
    pub status: Option<String>,
}

impl Default for ScriptRegistry {
    fn default() -> Self {
        Self {
            scripts: HashMap::new(),
            dirty_paths: HashSet::new(),
            last_scan: Instant::now(),
            scan_interval: Duration::from_secs(2),
            status: None,
        }
    }
}

impl ScriptRegistry {
    pub fn mark_dirty_paths(&mut self, paths: &HashSet<PathBuf>) {
        self.dirty_paths.extend(paths.iter().cloned());
    }

    pub fn mark_dirty_paths_owned(&mut self, paths: HashSet<PathBuf>) {
        self.dirty_paths.extend(paths);
    }

    pub fn take_dirty_paths(&mut self) -> HashSet<PathBuf> {
        std::mem::take(&mut self.dirty_paths)
    }
}

pub fn load_script_asset_from_disk(path: &Path) -> ScriptAsset {
    let language = runtime_scripting::script_language_from_path(path);
    if language == "rust" {
        let Some(manifest_path) = runtime_scripting::resolve_rust_script_manifest(path) else {
            return ScriptAsset {
                language,
                source: String::new(),
                modified: SystemTime::now(),
                error: Some("Rust script manifest not found".to_string()),
            };
        };

        let modified = rust_script_modified_time(&manifest_path);
        return match fs::read_to_string(&manifest_path) {
            Ok(source) => ScriptAsset {
                language,
                source,
                modified,
                error: None,
            },
            Err(err) => ScriptAsset {
                language,
                source: String::new(),
                modified,
                error: Some(err.to_string()),
            },
        };
    }

    let modified = fs::metadata(path)
        .and_then(|meta| meta.modified())
        .unwrap_or_else(|_| SystemTime::now());

    match fs::read_to_string(path) {
        Ok(source) => ScriptAsset {
            language,
            source,
            modified,
            error: None,
        },
        Err(err) => ScriptAsset {
            language,
            source: String::new(),
            modified,
            error: Some(err.to_string()),
        },
    }
}

pub fn update_script_registry(
    registry: &mut ScriptRegistry,
    project_root: &Path,
    project_config: Option<&ProjectConfig>,
) {
    update_script_registry_with_loader(
        registry,
        project_root,
        project_config,
        load_script_asset_from_disk,
    );
}

pub fn update_script_registry_with_loader<F>(
    registry: &mut ScriptRegistry,
    project_root: &Path,
    project_config: Option<&ProjectConfig>,
    mut load_script: F,
) where
    F: FnMut(&Path) -> ScriptAsset,
{
    let scripts_root = project_config
        .map(|cfg| cfg.scripts_root(project_root))
        .unwrap_or_else(|| project_root.join("assets").join("scripts"));

    let dirty_paths = registry.take_dirty_paths();
    if !dirty_paths.is_empty() {
        let mut dirty_keys = HashSet::<PathBuf>::new();
        for path in dirty_paths {
            if let Some(script_key) = runtime_scripting::script_registry_key_for_path(&path) {
                dirty_keys.insert(script_key);
            }
        }

        let mut updated = 0usize;
        let mut removed = 0usize;
        for script_key in dirty_keys {
            if script_key.exists() {
                let should_reload = match registry.scripts.get(&script_key) {
                    Some(existing) if existing.language != "rust" => fs::metadata(&script_key)
                        .and_then(|meta| meta.modified())
                        .map(|modified| modified > existing.modified)
                        .unwrap_or(true),
                    _ => true,
                };
                if !should_reload {
                    continue;
                }

                registry
                    .scripts
                    .insert(script_key.clone(), load_script(&script_key));
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
    let mut updated = 0usize;
    let mut removed = 0usize;

    for entry in WalkDir::new(&scripts_root)
        .into_iter()
        .filter_entry(should_visit_script_walk_entry)
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        if !should_consider_script_scan_path(entry.path()) {
            continue;
        }

        let Some(script_key) = runtime_scripting::script_registry_key_for_path(entry.path()) else {
            continue;
        };
        if !discovered.insert(script_key.clone()) {
            continue;
        }

        let should_attempt_reload = match registry.scripts.get(&script_key) {
            Some(existing) => fs::metadata(&script_key)
                .and_then(|meta| meta.modified())
                .map(|modified| modified > existing.modified)
                .unwrap_or(true),
            None => true,
        };
        if !should_attempt_reload {
            continue;
        }

        let next_asset = load_script(&script_key);
        let reload = match registry.scripts.get(&script_key) {
            Some(existing) => {
                next_asset.modified > existing.modified
                    || next_asset.error != existing.error
                    || next_asset.language != existing.language
                    || next_asset.source != existing.source
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

fn rust_script_modified_time(manifest_path: &Path) -> SystemTime {
    let mut latest = fs::metadata(manifest_path)
        .and_then(|meta| meta.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let Some(root) = manifest_path.parent() else {
        return latest;
    };

    let src_root = root.join("src");
    if src_root.exists() {
        for entry in WalkDir::new(&src_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            if let Ok(modified) = fs::metadata(entry.path()).and_then(|meta| meta.modified()) {
                if modified > latest {
                    latest = modified;
                }
            }
        }
    }

    let build_rs = root.join("build.rs");
    if let Ok(modified) = fs::metadata(&build_rs).and_then(|meta| meta.modified()) {
        if modified > latest {
            latest = modified;
        }
    }

    latest
}

fn should_visit_script_walk_entry(entry: &walkdir::DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }

    let Some(name) = entry.file_name().to_str() else {
        return true;
    };

    if name.starts_with('.') {
        return false;
    }

    if entry.file_type().is_dir() {
        return !is_ignored_script_dir_name(name);
    }

    true
}

fn should_consider_script_scan_path(path: &Path) -> bool {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("cargo.toml"))
    {
        return true;
    }

    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            ext.eq_ignore_ascii_case("lua")
                || ext.eq_ignore_ascii_case("luau")
                || ext.eq_ignore_ascii_case("hvs")
        })
}

fn is_ignored_script_dir_name(name: &str) -> bool {
    name.eq_ignore_ascii_case("target")
        || name.eq_ignore_ascii_case("node_modules")
        || name.eq_ignore_ascii_case(".git")
        || name.eq_ignore_ascii_case(".helmer")
}
