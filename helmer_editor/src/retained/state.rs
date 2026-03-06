use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use bevy_ecs::{
    entity::Entity,
    prelude::{Component, Resource, World},
};
use helmer::graphics::common::renderer::GizmoMode;
use helmer::runtime::runtime::{
    RuntimeLogEntry, RuntimeLogLevel, RuntimeLogListener, set_runtime_log_listener,
};
use helmer_becs::{
    BevyCamera, BevyLight, BevyMeshRenderer, BevySkinnedMeshRenderer, BevySpriteRenderer,
    BevyText2d, BevyTransform,
    physics::components::{ColliderShape, DynamicRigidBody},
};
pub use helmer_editor_runtime::editor_commands::{
    AssetCreateKind, EditorCommand, EditorCommandQueue, SpawnKind,
};
pub use helmer_editor_runtime::file_watch::{FileWatchState, configure_file_watcher};
pub use helmer_editor_runtime::project::{EditorProject, load_recent_projects};
use helmer_editor_runtime::project::{load_project_config, push_recent_project};
pub use helmer_editor_runtime::scene_state::{EditorEntity, WorldState};
pub use helmer_editor_runtime::script_registry::ScriptRegistry;
use helmer_editor_runtime::undo::EditorUndoState as RuntimeUndoState;
use walkdir::WalkDir;

use super::workspace::{
    EditorPaneArea, EditorPaneAreaRect, EditorPaneKind, EditorPaneTab, EditorPaneWindow,
    EditorPaneWorkspaceState,
};

const MAX_ASSET_SCAN_ENTRIES: usize = 20_000;
const MAX_PENDING_RUNTIME_LOGS: usize = 4_096;

#[derive(Debug, Clone, Default)]
pub struct RetainedPlayBackup {
    pub entities: Vec<RetainedPlayEntitySnapshot>,
    pub selected_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RetainedPlayEntitySnapshot {
    pub source: Entity,
    pub name: Option<String>,
    pub editor_entity: bool,
    pub transform: Option<BevyTransform>,
    pub camera: Option<BevyCamera>,
    pub active_camera: bool,
    pub light: Option<BevyLight>,
    pub mesh_renderer: Option<BevyMeshRenderer>,
    pub skinned_mesh_renderer: Option<BevySkinnedMeshRenderer>,
    pub sprite_renderer: Option<BevySpriteRenderer>,
    pub text_2d: Option<BevyText2d>,
    pub dynamic_body: Option<DynamicRigidBody>,
    pub fixed_collider: bool,
    pub collider_shape: Option<ColliderShape>,
    pub parent: Option<Entity>,
}

pub type EditorSceneState =
    helmer_editor_runtime::scene_state::EditorSceneState<RetainedPlayBackup>;

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct RetainedEditorViewportCamera;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainedViewportResolutionPreset {
    Canvas,
    R640x360,
    R854x480,
    R1280x720,
    R1600x900,
    R1920x1080,
    R2560x1440,
    R3840x2160,
}

impl RetainedViewportResolutionPreset {
    pub const ALL: [Self; 8] = [
        Self::Canvas,
        Self::R640x360,
        Self::R854x480,
        Self::R1280x720,
        Self::R1600x900,
        Self::R1920x1080,
        Self::R2560x1440,
        Self::R3840x2160,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Canvas => "Canvas (Auto)",
            Self::R640x360 => "640x360",
            Self::R854x480 => "854x480",
            Self::R1280x720 => "1280x720",
            Self::R1600x900 => "1600x900",
            Self::R1920x1080 => "1920x1080",
            Self::R2560x1440 => "2560x1440",
            Self::R3840x2160 => "3840x2160",
        }
    }

    pub fn target_size(self, surface_size: [u32; 2]) -> [u32; 2] {
        match self {
            Self::Canvas => surface_size,
            Self::R640x360 => [640, 360],
            Self::R854x480 => [854, 480],
            Self::R1280x720 => [1280, 720],
            Self::R1600x900 => [1600, 900],
            Self::R1920x1080 => [1920, 1080],
            Self::R2560x1440 => [2560, 1440],
            Self::R3840x2160 => [3840, 2160],
        }
    }
}

impl Default for RetainedViewportResolutionPreset {
    fn default() -> Self {
        Self::Canvas
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainedViewportFreecamSpeed {
    Slow,
    Normal,
    Fast,
}

impl Default for RetainedViewportFreecamSpeed {
    fn default() -> Self {
        Self::Normal
    }
}

pub const FREECAM_SENSITIVITY_DEFAULT: f32 = 0.12;
pub const FREECAM_SENSITIVITY_MIN: f32 = 0.01;
pub const FREECAM_SENSITIVITY_MAX: f32 = 2.0;
pub const FREECAM_SMOOTHING_DEFAULT: f32 = 0.04;
pub const FREECAM_SMOOTHING_MIN: f32 = 0.0;
pub const FREECAM_SMOOTHING_MAX: f32 = 0.25;
pub const FREECAM_MOVE_ACCEL_DEFAULT: f32 = 18.0;
pub const FREECAM_MOVE_ACCEL_MIN: f32 = 0.5;
pub const FREECAM_MOVE_ACCEL_MAX: f32 = 160.0;
pub const FREECAM_MOVE_DECEL_DEFAULT: f32 = 12.0;
pub const FREECAM_MOVE_DECEL_MIN: f32 = 0.5;
pub const FREECAM_MOVE_DECEL_MAX: f32 = 200.0;
pub const FREECAM_SPEED_STEP_DEFAULT: f32 = 2.0;
pub const FREECAM_SPEED_STEP_MIN: f32 = 0.05;
pub const FREECAM_SPEED_STEP_MAX: f32 = 40.0;
pub const FREECAM_SPEED_MIN_DEFAULT: f32 = 0.5;
pub const FREECAM_SPEED_MIN_MIN: f32 = 0.05;
pub const FREECAM_SPEED_MIN_MAX: f32 = 20.0;
pub const FREECAM_SPEED_MAX_DEFAULT: f32 = 80.0;
pub const FREECAM_SPEED_MAX_MIN: f32 = 1.0;
pub const FREECAM_SPEED_MAX_MAX: f32 = 500.0;
pub const FREECAM_BOOST_MULTIPLIER_DEFAULT: f32 = 2.5;
pub const FREECAM_BOOST_MULTIPLIER_MIN: f32 = 1.0;
pub const FREECAM_BOOST_MULTIPLIER_MAX: f32 = 10.0;
pub const FREECAM_ORBIT_DISTANCE_DEFAULT: f32 = 5.0;
pub const FREECAM_ORBIT_DISTANCE_MIN: f32 = 0.25;
pub const FREECAM_ORBIT_DISTANCE_MAX: f32 = 5000.0;
pub const FREECAM_ORBIT_PAN_SENSITIVITY_DEFAULT: f32 = 0.0020;
pub const FREECAM_ORBIT_PAN_SENSITIVITY_MIN: f32 = 0.0001;
pub const FREECAM_ORBIT_PAN_SENSITIVITY_MAX: f32 = 0.05;
pub const FREECAM_PAN_SENSITIVITY_DEFAULT: f32 = 0.0008;
pub const FREECAM_PAN_SENSITIVITY_MIN: f32 = 0.00005;
pub const FREECAM_PAN_SENSITIVITY_MAX: f32 = 0.05;

pub const GIZMO_SNAP_TRANSLATE_STEP_DEFAULT: f32 = 0.5;
pub const GIZMO_SNAP_ROTATE_STEP_DEGREES_DEFAULT: f32 = 15.0;
pub const GIZMO_SNAP_SCALE_STEP_DEFAULT: f32 = 0.1;
pub const GIZMO_SNAP_FINE_SCALE_DEFAULT: f32 = 0.25;
pub const GIZMO_SNAP_COARSE_SCALE_DEFAULT: f32 = 4.0;

#[derive(Resource, Debug, Clone)]
pub struct EditorRetainedGizmoSnapSettings {
    pub enabled: bool,
    pub ctrl_toggles: bool,
    pub shift_fine: bool,
    pub alt_coarse: bool,
    pub translate_step: f32,
    pub rotate_step_degrees: f32,
    pub scale_step: f32,
    pub fine_scale: f32,
    pub coarse_scale: f32,
}

impl Default for EditorRetainedGizmoSnapSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            ctrl_toggles: true,
            shift_fine: true,
            alt_coarse: true,
            translate_step: GIZMO_SNAP_TRANSLATE_STEP_DEFAULT,
            rotate_step_degrees: GIZMO_SNAP_ROTATE_STEP_DEGREES_DEFAULT,
            scale_step: GIZMO_SNAP_SCALE_STEP_DEFAULT,
            fine_scale: GIZMO_SNAP_FINE_SCALE_DEFAULT,
            coarse_scale: GIZMO_SNAP_COARSE_SCALE_DEFAULT,
        }
    }
}

impl EditorRetainedGizmoSnapSettings {
    pub fn sanitize(&mut self) {
        self.translate_step = self.translate_step.max(0.0);
        self.rotate_step_degrees = self.rotate_step_degrees.max(0.0);
        self.scale_step = self.scale_step.max(0.0);
        self.fine_scale = self.fine_scale.max(0.001);
        self.coarse_scale = self.coarse_scale.max(1.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainedPlayViewportKind {
    Editor,
    Gameplay,
}

impl RetainedPlayViewportKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Editor => "Edit",
            Self::Gameplay => "Game",
        }
    }
}

impl Default for RetainedPlayViewportKind {
    fn default() -> Self {
        Self::Gameplay
    }
}

#[derive(Resource, Debug, Clone)]
pub struct EditorRetainedViewportState {
    pub resolution: RetainedViewportResolutionPreset,
    pub play_mode_view: RetainedPlayViewportKind,
    pub graph_template: Option<String>,
    pub execute_scripts_in_edit_mode: bool,
    pub gizmos_in_play: bool,
    pub show_camera_gizmos: bool,
    pub show_directional_light_gizmos: bool,
    pub show_point_light_gizmos: bool,
    pub show_spot_light_gizmos: bool,
    pub show_spline_paths: bool,
    pub show_spline_points: bool,
    pub show_navigation_gizmo: bool,
    pub gizmo_mode: GizmoMode,
    pub freecam_speed: RetainedViewportFreecamSpeed,
    pub freecam_sensitivity: f32,
    pub freecam_smoothing: f32,
    pub freecam_move_accel: f32,
    pub freecam_move_decel: f32,
    pub freecam_speed_step: f32,
    pub freecam_speed_min: f32,
    pub freecam_speed_max: f32,
    pub freecam_boost_multiplier: f32,
    pub freecam_orbit_pan_sensitivity: f32,
    pub freecam_pan_sensitivity: f32,
    pub orbit_selected_entity: bool,
    pub orbit_distance: f32,
    pub preview_position_norm: [f32; 2],
    pub preview_width_norm: f32,
}

impl Default for EditorRetainedViewportState {
    fn default() -> Self {
        Self {
            resolution: RetainedViewportResolutionPreset::Canvas,
            play_mode_view: RetainedPlayViewportKind::Gameplay,
            graph_template: Some("debug-graph".to_string()),
            execute_scripts_in_edit_mode: false,
            gizmos_in_play: false,
            show_camera_gizmos: true,
            show_directional_light_gizmos: true,
            show_point_light_gizmos: true,
            show_spot_light_gizmos: true,
            show_spline_paths: true,
            show_spline_points: true,
            show_navigation_gizmo: true,
            gizmo_mode: GizmoMode::Translate,
            freecam_speed: RetainedViewportFreecamSpeed::Normal,
            freecam_sensitivity: FREECAM_SENSITIVITY_DEFAULT,
            freecam_smoothing: FREECAM_SMOOTHING_DEFAULT,
            freecam_move_accel: FREECAM_MOVE_ACCEL_DEFAULT,
            freecam_move_decel: FREECAM_MOVE_DECEL_DEFAULT,
            freecam_speed_step: FREECAM_SPEED_STEP_DEFAULT,
            freecam_speed_min: FREECAM_SPEED_MIN_DEFAULT,
            freecam_speed_max: FREECAM_SPEED_MAX_DEFAULT,
            freecam_boost_multiplier: FREECAM_BOOST_MULTIPLIER_DEFAULT,
            freecam_orbit_pan_sensitivity: FREECAM_ORBIT_PAN_SENSITIVITY_DEFAULT,
            freecam_pan_sensitivity: FREECAM_PAN_SENSITIVITY_DEFAULT,
            orbit_selected_entity: true,
            orbit_distance: FREECAM_ORBIT_DISTANCE_DEFAULT,
            preview_position_norm: [0.03, 0.74],
            preview_width_norm: 0.28,
        }
    }
}

impl EditorRetainedViewportState {
    pub fn sanitize(&mut self) {
        self.freecam_sensitivity = self
            .freecam_sensitivity
            .clamp(FREECAM_SENSITIVITY_MIN, FREECAM_SENSITIVITY_MAX);
        self.freecam_smoothing = self
            .freecam_smoothing
            .clamp(FREECAM_SMOOTHING_MIN, FREECAM_SMOOTHING_MAX);
        self.freecam_move_accel = self
            .freecam_move_accel
            .clamp(FREECAM_MOVE_ACCEL_MIN, FREECAM_MOVE_ACCEL_MAX);
        self.freecam_move_decel = self
            .freecam_move_decel
            .clamp(FREECAM_MOVE_DECEL_MIN, FREECAM_MOVE_DECEL_MAX);
        self.freecam_speed_step = self
            .freecam_speed_step
            .clamp(FREECAM_SPEED_STEP_MIN, FREECAM_SPEED_STEP_MAX);
        self.freecam_speed_min = self
            .freecam_speed_min
            .clamp(FREECAM_SPEED_MIN_MIN, FREECAM_SPEED_MIN_MAX);
        self.freecam_speed_max = self
            .freecam_speed_max
            .clamp(FREECAM_SPEED_MAX_MIN, FREECAM_SPEED_MAX_MAX);
        if self.freecam_speed_min > self.freecam_speed_max {
            std::mem::swap(&mut self.freecam_speed_min, &mut self.freecam_speed_max);
        }
        self.freecam_boost_multiplier = self
            .freecam_boost_multiplier
            .clamp(FREECAM_BOOST_MULTIPLIER_MIN, FREECAM_BOOST_MULTIPLIER_MAX);
        self.orbit_distance = self
            .orbit_distance
            .clamp(FREECAM_ORBIT_DISTANCE_MIN, FREECAM_ORBIT_DISTANCE_MAX);
        self.freecam_orbit_pan_sensitivity = self.freecam_orbit_pan_sensitivity.clamp(
            FREECAM_ORBIT_PAN_SENSITIVITY_MIN,
            FREECAM_ORBIT_PAN_SENSITIVITY_MAX,
        );
        self.freecam_pan_sensitivity = self
            .freecam_pan_sensitivity
            .clamp(FREECAM_PAN_SENSITIVITY_MIN, FREECAM_PAN_SENSITIVITY_MAX);
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct EditorRetainedViewportStates {
    pub states: HashMap<u64, EditorRetainedViewportState>,
}

impl EditorRetainedViewportStates {
    pub fn state_for(&self, tab_id: u64) -> EditorRetainedViewportState {
        self.states.get(&tab_id).cloned().unwrap_or_default()
    }

    pub fn state_for_mut(&mut self, tab_id: u64) -> &mut EditorRetainedViewportState {
        self.states.entry(tab_id).or_default()
    }
}

#[derive(Debug, Clone)]
pub struct RetainedSceneUndoSnapshot {
    pub path: Option<PathBuf>,
    pub name: String,
    pub dirty: bool,
    pub world_state: WorldState,
}

#[derive(Debug, Clone)]
pub struct RetainedUndoEntry {
    pub label: Option<String>,
    pub scene: RetainedSceneUndoSnapshot,
}

impl RetainedUndoEntry {
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

pub type EditorUndoState = RuntimeUndoState<RetainedUndoEntry, ()>;

#[derive(Resource, Debug, Clone)]
pub struct EditorProjectLauncherState {
    pub project_name: String,
    pub project_path: String,
    pub open_project_path: String,
    pub status: Option<String>,
    pub recent_projects: Vec<PathBuf>,
}

impl EditorProjectLauncherState {
    pub fn from_projects_root(projects_root: PathBuf) -> Self {
        let default_root = projects_root.to_string_lossy().into_owned();
        Self {
            project_name: "NewProject".to_string(),
            project_path: default_root.clone(),
            open_project_path: default_root,
            status: None,
            recent_projects: load_recent_projects(),
        }
    }

    pub fn normalize_inputs(&mut self) {
        if self.project_name.trim().is_empty() {
            self.project_name = "NewProject".to_string();
        }
        if self.project_path.trim().is_empty() {
            self.project_path = "./projects".to_string();
        }
        if self.open_project_path.trim().is_empty() {
            self.open_project_path = self.project_path.clone();
        }
    }

    pub fn desired_open_path(&self) -> PathBuf {
        PathBuf::from(self.open_project_path.trim())
    }

    pub fn desired_create_name(&self) -> String {
        let trimmed = self.project_name.trim();
        if trimmed.is_empty() {
            "NewProject".to_string()
        } else {
            trimmed.to_string()
        }
    }

    pub fn desired_create_path(&self) -> PathBuf {
        PathBuf::from(self.project_path.trim()).join(self.desired_create_name())
    }
}

impl Default for EditorProjectLauncherState {
    fn default() -> Self {
        Self::from_projects_root(PathBuf::from("./projects"))
    }
}

pub fn record_recent_project(
    launcher: &mut EditorProjectLauncherState,
    path: &Path,
) -> Result<(), String> {
    push_recent_project(&mut launcher.recent_projects, path)
}

#[derive(Debug, Clone)]
pub struct AssetEntry {
    pub path: PathBuf,
    pub depth: usize,
    pub is_dir: bool,
}

#[derive(Resource, Debug, Clone)]
pub struct AssetBrowserState {
    pub root: Option<PathBuf>,
    pub entries: Vec<AssetEntry>,
    pub expanded: HashSet<PathBuf>,
    pub selected: Option<PathBuf>,
    pub selected_paths: HashSet<PathBuf>,
    pub selection_anchor: Option<PathBuf>,
    pub current_dir: Option<PathBuf>,
    pub last_click_path: Option<PathBuf>,
    pub last_click_instant: Option<Instant>,
    pub refresh_requested: bool,
    pub filter: String,
    pub location_dropdown_open: bool,
    pub grid_scroll: f32,
    pub tile_size: f32,
    pub status: Option<String>,
    pub last_scan: Instant,
    pub scan_interval: Duration,
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
            current_dir: None,
            last_click_path: None,
            last_click_instant: None,
            refresh_requested: true,
            filter: String::new(),
            location_dropdown_open: false,
            grid_scroll: 0.0,
            tile_size: 112.0,
            status: None,
            last_scan: Instant::now(),
            scan_interval: Duration::from_secs(2),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssetClipboardMode {
    #[default]
    Copy,
    Cut,
}

#[derive(Resource, Debug, Clone, Default)]
pub struct EditorAssetClipboardState {
    pub mode: AssetClipboardMode,
    pub paths: Vec<PathBuf>,
}

pub fn is_entry_visible(entry: &AssetEntry, root: &Path, expanded: &HashSet<PathBuf>) -> bool {
    if !entry.path.starts_with(root) {
        return false;
    }
    if entry.path == root || entry.depth <= 1 {
        return true;
    }

    let mut current = entry.path.parent();
    while let Some(parent) = current {
        if parent == root {
            return true;
        }
        if !expanded.contains(parent) {
            return false;
        }
        current = parent.parent();
    }

    false
}

pub fn refresh_asset_browser(state: &mut AssetBrowserState) {
    state.last_scan = Instant::now();

    let Some(root) = state.root.clone() else {
        state.entries.clear();
        state.current_dir = None;
        state.selected = None;
        state.selected_paths.clear();
        state.selection_anchor = None;
        state.last_click_path = None;
        state.location_dropdown_open = false;
        state.grid_scroll = 0.0;
        state.tile_size = state.tile_size.clamp(64.0, 220.0);
        state.status = Some("No project assets root".to_string());
        state.refresh_requested = false;
        return;
    };

    if !root.exists() {
        state.entries.clear();
        state.current_dir = Some(root.clone());
        state.selected = Some(root.clone());
        state.selected_paths.clear();
        state.selected_paths.insert(root.clone());
        state.selection_anchor = Some(root.clone());
        state.location_dropdown_open = false;
        state.grid_scroll = 0.0;
        state.tile_size = state.tile_size.clamp(64.0, 220.0);
        state.status = Some(format!(
            "Assets root does not exist: {}",
            root.to_string_lossy()
        ));
        state.refresh_requested = false;
        return;
    }

    match scan_asset_entries(&root) {
        Ok(entries) => {
            state.entries = entries;
            state.expanded.insert(root.clone());
            if !state
                .current_dir
                .as_ref()
                .map(|path| path.starts_with(&root))
                .unwrap_or(false)
            {
                state.current_dir = Some(root.clone());
            }
            if !state
                .selected
                .as_ref()
                .map(|path| path.starts_with(&root))
                .unwrap_or(false)
            {
                state.selected = Some(root.clone());
            }
            state.selected_paths.retain(|path| path.starts_with(&root));
            if state.selected_paths.is_empty() {
                if let Some(selected) = state.selected.clone() {
                    state.selected_paths.insert(selected.clone());
                    state.selection_anchor = Some(selected);
                }
            }
            state.grid_scroll = state.grid_scroll.max(0.0);
            state.tile_size = state.tile_size.clamp(64.0, 220.0);
            state.status = None;
        }
        Err(err) => {
            state.entries.clear();
            state.status = Some(err);
        }
    }

    state.refresh_requested = false;
}

fn scan_asset_entries(root: &Path) -> Result<Vec<AssetEntry>, String> {
    let mut entries = Vec::new();

    for entry in WalkDir::new(root).follow_links(false).sort_by_file_name() {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path().to_path_buf();
        if path != root && path_has_hidden_component(root, &path) {
            continue;
        }
        let depth = path
            .strip_prefix(root)
            .map(|relative| relative.components().count())
            .unwrap_or(0);

        entries.push(AssetEntry {
            path,
            depth,
            is_dir: entry.file_type().is_dir(),
        });

        if entries.len() >= MAX_ASSET_SCAN_ENTRIES {
            break;
        }
    }

    entries.sort_by(|lhs, rhs| lhs.path.cmp(&rhs.path));
    Ok(entries)
}

fn path_has_hidden_component(root: &Path, path: &Path) -> bool {
    let components = path
        .strip_prefix(root)
        .map(|relative| relative.components().collect::<Vec<_>>())
        .unwrap_or_else(|_| path.components().collect::<Vec<_>>());
    components.into_iter().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|name| name.starts_with('.'))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorConsoleLevel {
    Trace,
    Debug,
    Log,
    Info,
    Warn,
    Error,
}

impl EditorConsoleLevel {
    pub fn from_runtime_level(level: RuntimeLogLevel) -> Self {
        match level {
            RuntimeLogLevel::Trace => Self::Trace,
            RuntimeLogLevel::Debug => Self::Debug,
            RuntimeLogLevel::Info => Self::Info,
            RuntimeLogLevel::Warn => Self::Warn,
            RuntimeLogLevel::Error => Self::Error,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EditorConsoleEntry {
    pub sequence: u64,
    pub level: EditorConsoleLevel,
    pub target: String,
    pub message: String,
}

#[derive(Debug, Clone, Resource)]
pub struct EditorConsoleState {
    pub entries: VecDeque<EditorConsoleEntry>,
    pub max_entries: usize,
    pub show_trace: bool,
    pub show_debug: bool,
    pub show_log: bool,
    pub show_info: bool,
    pub show_warn: bool,
    pub show_error: bool,
    pub search: String,
    pub auto_scroll: bool,
    pub scroll: f32,
    next_sequence: u64,
}

impl Default for EditorConsoleState {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries: 5_000,
            show_trace: true,
            show_debug: true,
            show_log: true,
            show_info: true,
            show_warn: true,
            show_error: true,
            search: String::new(),
            auto_scroll: true,
            scroll: 0.0,
            next_sequence: 1,
        }
    }
}

impl EditorConsoleState {
    pub fn clear(&mut self) {
        self.entries.clear();
        self.scroll = 0.0;
    }

    pub fn push(
        &mut self,
        level: EditorConsoleLevel,
        target: impl Into<String>,
        message: impl Into<String>,
    ) {
        let message = message.into().trim().to_string();
        if message.is_empty() {
            return;
        }

        self.entries.push_back(EditorConsoleEntry {
            sequence: self.next_sequence,
            level,
            target: target.into(),
            message,
        });
        self.next_sequence = self.next_sequence.saturating_add(1);

        let max_entries = self.max_entries.max(1);
        while self.entries.len() > max_entries {
            self.entries.pop_front();
        }
    }
}

static RUNTIME_LOG_QUEUE: OnceLock<Arc<Mutex<VecDeque<RuntimeLogEntry>>>> = OnceLock::new();

fn runtime_log_queue() -> Arc<Mutex<VecDeque<RuntimeLogEntry>>> {
    RUNTIME_LOG_QUEUE
        .get_or_init(|| Arc::new(Mutex::new(VecDeque::new())))
        .clone()
}

pub fn install_runtime_log_listener() {
    let queue = runtime_log_queue();
    let listener: Arc<RuntimeLogListener> = Arc::new(move |entry: RuntimeLogEntry| {
        if let Ok(mut pending) = queue.lock() {
            if pending.len() >= MAX_PENDING_RUNTIME_LOGS {
                pending.pop_front();
            }
            pending.push_back(entry);
        }
    });
    let _ = set_runtime_log_listener(listener);
}

pub fn drain_runtime_log_entries() -> Vec<RuntimeLogEntry> {
    let queue = runtime_log_queue();
    let Ok(mut pending) = queue.lock() else {
        return Vec::new();
    };
    pending.drain(..).collect()
}

#[derive(Debug, Clone, Default)]
pub struct TimelineTrackGroupSummary {
    pub id: u64,
    pub name: String,
}

#[derive(Debug, Clone, Resource)]
pub struct EditorTimelineState {
    pub playing: bool,
    pub current_time: f32,
    pub duration: f32,
    pub playback_rate: f32,
    pub groups: Vec<TimelineTrackGroupSummary>,
    pub selected: Vec<u64>,
}

impl Default for EditorTimelineState {
    fn default() -> Self {
        Self {
            playing: false,
            current_time: 0.0,
            duration: 5.0,
            playback_rate: 1.0,
            groups: Vec::new(),
            selected: Vec::new(),
        }
    }
}

pub fn initialize_editor_project(world: &mut World, project_root: PathBuf) {
    let config = load_project_config(&project_root).ok();

    if let Some(mut project) = world.get_resource_mut::<EditorProject>() {
        project.root = Some(project_root.clone());
        project.config = config.clone();
    }

    if let Some(mut watch) = world.get_resource_mut::<FileWatchState>() {
        configure_file_watcher(&mut watch, &project_root, config.as_ref());
    }

    if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
        scene.path = None;
        scene.name = "Untitled".to_string();
        scene.dirty = false;
        scene.world_state = WorldState::Edit;
        scene.play_backup = None;
        scene.play_selected_index = None;
    }

    if let Some(mut scripts) = world.get_resource_mut::<ScriptRegistry>() {
        scripts.scripts.clear();
        scripts.dirty_paths.clear();
        scripts.last_scan = Instant::now()
            .checked_sub(scripts.scan_interval)
            .unwrap_or_else(Instant::now);
        scripts.status = Some("Queued full script scan".to_string());
    }

    if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
        browser.root = Some(project_root.clone());
        browser.current_dir = Some(project_root.clone());
        browser.selected = Some(project_root.clone());
        browser.selection_anchor = Some(project_root.clone());
        browser.selected_paths.clear();
        browser.selected_paths.insert(project_root.clone());
        browser.expanded.insert(project_root.clone());
        browser.refresh_requested = true;
    }

    let watch_status = world
        .get_resource::<FileWatchState>()
        .and_then(|watch| watch.status.clone());

    if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
        console.push(
            EditorConsoleLevel::Info,
            "editor.project",
            format!("Opened project {}", project_root.to_string_lossy()),
        );
        if let Some(status) = watch_status {
            console.push(EditorConsoleLevel::Info, "editor.watch", status);
        }
    }

    reset_scene_undo_history(world, Some("Open Project"));
}

pub fn capture_scene_undo_snapshot(world: &World) -> Option<RetainedSceneUndoSnapshot> {
    let scene = world.get_resource::<EditorSceneState>()?;
    Some(RetainedSceneUndoSnapshot {
        path: scene.path.clone(),
        name: scene.name.clone(),
        dirty: scene.dirty,
        world_state: scene.world_state,
    })
}

pub fn apply_scene_undo_snapshot(world: &mut World, snapshot: &RetainedSceneUndoSnapshot) {
    let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() else {
        return;
    };
    scene.path = snapshot.path.clone();
    scene.name = snapshot.name.clone();
    scene.dirty = snapshot.dirty;
    scene.world_state = snapshot.world_state;
}

pub fn push_scene_undo_entry(world: &mut World, label: impl Into<String>) {
    let Some(snapshot) = capture_scene_undo_snapshot(world) else {
        return;
    };
    let Some(mut undo) = world.get_resource_mut::<EditorUndoState>() else {
        return;
    };

    let truncate_to = undo.cursor + 1;
    if truncate_to < undo.entries.len() {
        undo.entries.truncate(truncate_to);
    }
    undo.entries.push(RetainedUndoEntry {
        label: Some(label.into()),
        scene: snapshot,
    });
    undo.cursor = undo.entries.len().saturating_sub(1);
    undo.enforce_cap();
}

pub fn reset_scene_undo_history(world: &mut World, label: Option<&str>) {
    let Some(snapshot) = capture_scene_undo_snapshot(world) else {
        return;
    };
    let Some(mut undo) = world.get_resource_mut::<EditorUndoState>() else {
        return;
    };
    undo.entries.clear();
    undo.entries.push(RetainedUndoEntry {
        label: label.map(str::to_string),
        scene: snapshot,
    });
    undo.cursor = 0;
    undo.pending_group = None;
    undo.pending_commit = false;
}

pub fn ensure_default_pane_workspace(workspace: &mut EditorPaneWorkspaceState) {
    if workspace.initialized && !workspace.windows.is_empty() {
        return;
    }

    workspace.initialized = true;
    workspace.windows.clear();

    push_window(workspace, "Project", true, &[EditorPaneKind::Project]);
    push_window(workspace, "Toolbar", true, &[EditorPaneKind::Toolbar]);
    push_window(
        workspace,
        "Viewport",
        true,
        &[EditorPaneKind::Viewport, EditorPaneKind::PlayViewport],
    );
    push_window(
        workspace,
        "Content Browser",
        true,
        &[EditorPaneKind::ContentBrowser, EditorPaneKind::Console],
    );
    push_window(workspace, "Hierarchy", true, &[EditorPaneKind::Hierarchy]);
    push_window(workspace, "Inspector", true, &[EditorPaneKind::Inspector]);

    workspace.last_focused_window = workspace.windows.first().map(|window| window.id.clone());
    workspace.last_focused_area = workspace
        .windows
        .first()
        .and_then(|window| window.areas.first())
        .map(|area| area.id);
    workspace.dragging = None;
    workspace.drop_handled = false;
}

fn push_window(
    workspace: &mut EditorPaneWorkspaceState,
    id: &str,
    layout_managed: bool,
    kinds: &[EditorPaneKind],
) {
    let area_id = workspace.next_area_id;
    workspace.next_area_id = workspace.next_area_id.saturating_add(1);

    let tabs = kinds
        .iter()
        .copied()
        .map(|kind| EditorPaneTab::from_builtin(workspace, kind))
        .collect::<Vec<_>>();

    workspace.windows.push(EditorPaneWindow {
        id: id.to_string(),
        title: id.to_string(),
        areas: vec![EditorPaneArea {
            id: area_id,
            rect: EditorPaneAreaRect::full(),
            tabs,
            active: 0,
        }],
        layout_managed,
    });
}
