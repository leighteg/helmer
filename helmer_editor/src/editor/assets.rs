use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use bevy_ecs::prelude::{Component, Resource};
use egui::Pos2;
use helmer::audio::AudioLoadMode;
use helmer::provided::components::MeshAsset;
use helmer::runtime::asset_server::{AssetKind, Handle, Material, Mesh, Scene, Texture};
use helmer_becs::BevyAssetServer;
use serde::{Deserialize, Serialize};
use walkdir::{DirEntry, WalkDir};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveKind {
    Cube,
    UvSphere(u32, u32),
    Icosphere(u32),
    Cylinder(u32, u32),
    Capsule(u32, u32),
    Plane,
}

impl PrimitiveKind {
    pub const DEFAULT_UV_SPHERE_SEGMENTS: u32 = 12;
    pub const DEFAULT_UV_SPHERE_RINGS: u32 = 12;
    pub const DEFAULT_ICOSPHERE_SUBDIVISIONS: u32 = 2;
    pub const DEFAULT_CYLINDER_RADIAL_SEGMENTS: u32 = 24;
    pub const DEFAULT_CYLINDER_HEIGHT_SEGMENTS: u32 = 1;
    pub const DEFAULT_CAPSULE_SEGMENTS: u32 = 24;
    pub const DEFAULT_CAPSULE_RINGS: u32 = 8;

    pub const fn default_uv_sphere() -> Self {
        Self::UvSphere(
            Self::DEFAULT_UV_SPHERE_SEGMENTS,
            Self::DEFAULT_UV_SPHERE_RINGS,
        )
    }

    pub const fn default_icosphere() -> Self {
        Self::Icosphere(Self::DEFAULT_ICOSPHERE_SUBDIVISIONS)
    }

    pub const fn default_cylinder() -> Self {
        Self::Cylinder(
            Self::DEFAULT_CYLINDER_RADIAL_SEGMENTS,
            Self::DEFAULT_CYLINDER_HEIGHT_SEGMENTS,
        )
    }

    pub const fn default_capsule() -> Self {
        Self::Capsule(Self::DEFAULT_CAPSULE_SEGMENTS, Self::DEFAULT_CAPSULE_RINGS)
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Cube => "Cube",
            Self::UvSphere(_, _) => "UV Sphere",
            Self::Icosphere(_) => "Icosphere",
            Self::Cylinder(_, _) => "Cylinder",
            Self::Capsule(_, _) => "Capsule",
            Self::Plane => "Plane",
        }
    }

    pub fn from_source_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "cube" => Some(Self::Cube),
            "uv sphere" | "uv_sphere" | "uvsphere" => Some(Self::default_uv_sphere()),
            "icosphere" | "ico sphere" | "ico_sphere" => Some(Self::default_icosphere()),
            "cylinder" => Some(Self::default_cylinder()),
            "capsule" => Some(Self::default_capsule()),
            "plane" => Some(Self::Plane),
            _ => None,
        }
    }

    pub fn to_mesh_asset(self) -> MeshAsset {
        match self {
            Self::Cube => MeshAsset::cube("cube".to_string()),
            Self::UvSphere(segments, rings) => {
                MeshAsset::uv_sphere("uv sphere".to_string(), segments.max(3), rings.max(3))
            }
            Self::Icosphere(subdivisions) => {
                MeshAsset::icosphere("icosphere".to_string(), subdivisions.min(6))
            }
            Self::Cylinder(radial_segments, height_segments) => MeshAsset::cylinder(
                "cylinder".to_string(),
                radial_segments.max(3),
                height_segments.max(1),
            ),
            Self::Capsule(segments, rings) => {
                MeshAsset::capsule("capsule".to_string(), segments.max(3), rings.max(2))
            }
            Self::Plane => MeshAsset::plane("plane".to_string()),
        }
    }
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
pub struct EditorSkinnedMesh {
    pub scene_path: Option<String>,
    pub node_index: Option<usize>,
    pub casts_shadow: bool,
    pub visible: bool,
}

#[derive(Component, Debug, Clone)]
pub struct EditorAudio {
    pub path: Option<String>,
    pub streaming: bool,
}

#[derive(Component, Debug, Clone)]
pub struct EditorSprite {
    pub texture_path: Option<String>,
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
    pub scene_handles: HashMap<String, Handle<Scene>>,
    pub audio_handles: HashMap<String, Handle<helmer::audio::AudioClip>>,
    pub texture_handles: HashMap<String, Handle<Texture>>,
}

pub fn scene_cache_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn audio_cache_key(path: &Path, streaming: bool) -> String {
    let mut key = scene_cache_key(path);
    key.push_str(if streaming { "|stream" } else { "|static" });
    key
}

pub fn cached_scene_handle(
    cache: &mut EditorAssetCache,
    asset_server: &BevyAssetServer,
    path: &Path,
) -> Handle<Scene> {
    let key = scene_cache_key(path);
    if let Some(handle) = cache.scene_handles.get(&key) {
        return handle.clone();
    }
    let handle = asset_server.0.lock().load_scene(path);
    cache.scene_handles.insert(key, handle.clone());
    handle
}

pub fn cached_audio_handle(
    cache: &mut EditorAssetCache,
    asset_server: &BevyAssetServer,
    path: &Path,
    streaming: bool,
) -> Handle<helmer::audio::AudioClip> {
    let key = audio_cache_key(path, streaming);
    if let Some(handle) = cache.audio_handles.get(&key) {
        return handle.clone();
    }
    let mode = if streaming {
        AudioLoadMode::Streaming
    } else {
        AudioLoadMode::Static
    };
    let handle = asset_server.0.lock().load_audio(path, mode);
    cache.audio_handles.insert(key, handle.clone());
    handle
}

pub fn cached_texture_handle(
    cache: &mut EditorAssetCache,
    asset_server: &BevyAssetServer,
    path: &Path,
) -> Handle<Texture> {
    let key = scene_cache_key(path);
    if let Some(handle) = cache.texture_handles.get(&key) {
        return handle.clone();
    }
    let handle = asset_server.0.lock().load_texture(path, AssetKind::Albedo);
    cache.texture_handles.insert(key, handle.clone());
    handle
}

#[derive(Debug, Clone)]
pub struct AssetEntry {
    pub path: PathBuf,
    pub depth: usize,
    pub is_dir: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetRenameSelection {
    All,
    FileStem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetRenameView {
    Tree,
    Grid,
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
    pub rename_pending_selection: Option<AssetRenameSelection>,
    pub rename_view: AssetRenameView,
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
            rename_pending_selection: None,
            rename_view: AssetRenameView::Grid,
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
        .filter_entry(should_visit_asset_walk_entry)
        .filter_map(Result::ok)
    {
        let path = entry.path();
        let depth = entry.depth();
        let is_dir = entry.file_type().is_dir();

        if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
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

fn should_visit_asset_walk_entry(entry: &DirEntry) -> bool {
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
        return !is_ignored_asset_dir_name(name);
    }

    true
}

fn is_ignored_asset_dir_name(name: &str) -> bool {
    name.eq_ignore_ascii_case("target")
        || name.eq_ignore_ascii_case("node_modules")
        || name.eq_ignore_ascii_case(".git")
        || name.eq_ignore_ascii_case(".helmer")
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
