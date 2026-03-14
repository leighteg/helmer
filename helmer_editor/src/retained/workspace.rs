use std::path::PathBuf;

use helmer_becs::ecs::prelude::Resource;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditorPaneKind {
    Toolbar,
    Viewport,
    PlayViewport,
    Project,
    Hierarchy,
    Inspector,
    History,
    Timeline,
    ContentBrowser,
    Console,
    AudioMixer,
    Profiler,
    MaterialEditor,
    VisualScriptEditor,
}

impl EditorPaneKind {
    pub const ALL: [Self; 14] = [
        Self::Toolbar,
        Self::Viewport,
        Self::PlayViewport,
        Self::Project,
        Self::Hierarchy,
        Self::Inspector,
        Self::History,
        Self::Timeline,
        Self::ContentBrowser,
        Self::Console,
        Self::AudioMixer,
        Self::Profiler,
        Self::MaterialEditor,
        Self::VisualScriptEditor,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Toolbar => "Toolbar",
            Self::Viewport => "Viewport",
            Self::PlayViewport => "Play Viewport",
            Self::Project => "Project",
            Self::Hierarchy => "Hierarchy",
            Self::Inspector => "Inspector",
            Self::History => "History",
            Self::Timeline => "Timeline",
            Self::ContentBrowser => "Content Browser",
            Self::Console => "Console",
            Self::AudioMixer => "Audio Mixer",
            Self::Profiler => "Profiler",
            Self::MaterialEditor => "Material",
            Self::VisualScriptEditor => "Visual Script",
        }
    }

    pub fn default_layout_window_id(self) -> Option<&'static str> {
        match self {
            Self::Toolbar => Some("Toolbar"),
            Self::Viewport => Some("Viewport"),
            Self::PlayViewport => Some("Viewport"),
            Self::Project => Some("Project"),
            Self::Hierarchy => Some("Hierarchy"),
            Self::Inspector => Some("Inspector"),
            Self::History => Some("History"),
            Self::ContentBrowser => Some("Content Browser"),
            Self::Console => Some("Content Browser"),
            Self::Timeline
            | Self::AudioMixer
            | Self::Profiler
            | Self::MaterialEditor
            | Self::VisualScriptEditor => None,
        }
    }

    pub fn persistence_key(self) -> &'static str {
        match self {
            Self::Toolbar => "toolbar",
            Self::Viewport => "viewport",
            Self::PlayViewport => "play_viewport",
            Self::Project => "project",
            Self::Hierarchy => "hierarchy",
            Self::Inspector => "inspector",
            Self::History => "history",
            Self::Timeline => "timeline",
            Self::ContentBrowser => "content_browser",
            Self::Console => "console",
            Self::AudioMixer => "audio_mixer",
            Self::Profiler => "profiler",
            Self::MaterialEditor => "material_editor",
            Self::VisualScriptEditor => "visual_script_editor",
        }
    }

    pub fn from_persistence_key(key: &str) -> Option<Self> {
        match key {
            "toolbar" => Some(Self::Toolbar),
            "viewport" => Some(Self::Viewport),
            "play_viewport" => Some(Self::PlayViewport),
            "project" => Some(Self::Project),
            "hierarchy" => Some(Self::Hierarchy),
            "inspector" => Some(Self::Inspector),
            "history" => Some(Self::History),
            "timeline" => Some(Self::Timeline),
            "content_browser" => Some(Self::ContentBrowser),
            "console" => Some(Self::Console),
            "audio_mixer" => Some(Self::AudioMixer),
            "profiler" => Some(Self::Profiler),
            "material_editor" => Some(Self::MaterialEditor),
            "visual_script_editor" => Some(Self::VisualScriptEditor),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Resource)]
pub struct EditorPaneWorkspaceState {
    pub initialized: bool,
    pub next_window_id: u64,
    pub next_tab_id: u64,
    pub next_area_id: u64,
    pub windows: Vec<EditorPaneWindow>,
    pub last_focused_window: Option<String>,
    pub last_focused_area: Option<u64>,
    pub dragging: Option<EditorPaneTabDrag>,
    pub drop_handled: bool,
}

impl Default for EditorPaneWorkspaceState {
    fn default() -> Self {
        Self {
            initialized: false,
            next_window_id: 1,
            next_tab_id: 1,
            next_area_id: 1,
            windows: Vec::new(),
            last_focused_window: None,
            last_focused_area: None,
            dragging: None,
            drop_handled: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EditorPaneWindow {
    pub id: String,
    pub title: String,
    pub areas: Vec<EditorPaneArea>,
    pub layout_managed: bool,
}

#[derive(Debug, Clone)]
pub struct EditorPaneArea {
    pub id: u64,
    pub rect: EditorPaneAreaRect,
    pub tabs: Vec<EditorPaneTab>,
    pub active: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EditorPaneAreaRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl EditorPaneAreaRect {
    pub fn full() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        }
    }

    pub fn to_host_rect(
        self,
        host_x: f32,
        host_y: f32,
        host_width: f32,
        host_height: f32,
    ) -> (f32, f32, f32, f32) {
        let width = host_width.max(1.0);
        let height = host_height.max(1.0);
        (
            host_x + self.x * width,
            host_y + self.y * height,
            self.w * width,
            self.h * height,
        )
    }
}

#[derive(Debug, Clone)]
pub struct EditorPaneTab {
    pub id: u64,
    pub title: String,
    pub kind: EditorPaneKind,
    pub asset_path: Option<PathBuf>,
}

impl EditorPaneTab {
    pub fn from_builtin(workspace: &mut EditorPaneWorkspaceState, kind: EditorPaneKind) -> Self {
        let tab = Self {
            id: workspace.next_tab_id,
            title: kind.label().to_string(),
            kind,
            asset_path: None,
        };
        workspace.next_tab_id += 1;
        tab
    }

    pub fn duplicate_with_new_id(&self, workspace: &mut EditorPaneWorkspaceState) -> Self {
        let mut tab = self.clone();
        tab.id = workspace.next_tab_id;
        workspace.next_tab_id += 1;
        tab
    }
}

#[derive(Debug, Clone)]
pub struct EditorPaneTabDrag {
    pub tab: EditorPaneTab,
    pub source_window_id: String,
    pub source_area_id: u64,
    pub source_was_single_tab: bool,
}
