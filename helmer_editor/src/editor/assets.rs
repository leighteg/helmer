use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use bevy_ecs::prelude::{Component, Resource};
use egui::Pos2;
use helmer::runtime::asset_server::{Handle, Material, Mesh};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveKind {
    Cube,
    UvSphere(u32, u32),
    Plane,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeshSource {
    Primitive(PrimitiveKind),
    Asset { path: String },
}

#[derive(Component, Debug, Clone)]
pub struct EditorMesh {
    pub source: MeshSource,
    pub material_path: Option<String>,
}

#[derive(Component, Debug, Clone)]
pub struct SceneAssetPath {
    pub path: PathBuf,
}

#[derive(Resource, Default)]
pub struct EditorAssetCache {
    pub default_material: Option<Handle<Material>>,
    pub material_handles: HashMap<String, Handle<Material>>,
    pub mesh_handles: HashMap<String, Handle<Mesh>>,
    pub primitive_meshes: HashMap<PrimitiveKind, Handle<Mesh>>,
}

#[derive(Debug, Clone)]
pub struct AssetEntry {
    pub path: PathBuf,
    pub depth: usize,
    pub is_dir: bool,
}

#[derive(Resource)]
pub struct AssetBrowserState {
    pub root: Option<PathBuf>,
    pub entries: Vec<AssetEntry>,
    pub expanded: HashSet<PathBuf>,
    pub selected: Option<PathBuf>,
    pub selected_paths: HashSet<PathBuf>,
    pub selection_anchor: Option<PathBuf>,
    pub selection_drag_start: Option<Pos2>,
    pub current_dir: Option<PathBuf>,
    pub last_click_path: Option<PathBuf>,
    pub last_click_time: f64,
    pub last_scan: Instant,
    pub scan_interval: Duration,
    pub refresh_requested: bool,
    pub filter: String,
    pub status: Option<String>,
    pub rename_path: Option<PathBuf>,
    pub rename_buffer: String,
    pub tile_size: f32,
}

impl Default for AssetBrowserState {
    fn default() -> Self {
        Self {
            root: None,
            entries: Vec::new(),
            expanded: HashSet::new(),
            selected: None,
            selected_paths: HashSet::new(),
            selection_anchor: None,
            selection_drag_start: None,
            current_dir: None,
            last_click_path: None,
            last_click_time: 0.0,
            last_scan: Instant::now(),
            scan_interval: Duration::from_secs(1),
            refresh_requested: true,
            filter: String::new(),
            status: None,
            rename_path: None,
            rename_buffer: String::new(),
            tile_size: 120.0,
        }
    }
}

pub fn scan_asset_entries(root: &Path, filter: &str) -> Vec<AssetEntry> {
    let mut entries = Vec::new();

    if !root.exists() {
        return entries;
    }

    let lower_filter = filter.to_ascii_lowercase();

    for entry in WalkDir::new(root)
        .min_depth(0)
        .sort_by_file_name()
        .into_iter()
        .filter_map(Result::ok)
    {
        let path = entry.path();
        let depth = entry.depth();
        let is_dir = entry.file_type().is_dir();

        if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            if name.starts_with('.') {
                continue;
            }

            if !lower_filter.is_empty() && !is_dir {
                let name_lower = name.to_ascii_lowercase();
                if !name_lower.contains(&lower_filter) {
                    continue;
                }
            }
        }

        entries.push(AssetEntry {
            path: path.to_path_buf(),
            depth,
            is_dir,
        });
    }

    entries
}

pub fn is_entry_visible(entry: &AssetEntry, root: &Path, expanded: &HashSet<PathBuf>) -> bool {
    if entry.depth == 0 {
        return true;
    }

    let mut current = entry.path.as_path();
    while let Some(parent) = current.parent() {
        if parent == root {
            return expanded.contains(parent);
        }
        if !expanded.contains(parent) {
            return false;
        }
        current = parent;
    }

    true
}
