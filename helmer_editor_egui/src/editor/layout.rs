use std::{collections::HashMap, env, fs, path::PathBuf};

use bevy_ecs::prelude::Resource;
use egui::{Pos2, Rect, Vec2};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

const LAYOUTS_FILE_NAME: &str = "layouts.ron";

const fn default_layout_move_enabled() -> bool {
    true
}

const fn default_layout_resize_enabled() -> bool {
    true
}

const fn default_live_reflow_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl NormalizedRect {
    pub fn from_rect(rect: Rect, screen: Rect) -> Self {
        let width = screen.width().max(1.0);
        let height = screen.height().max(1.0);
        Self {
            x: clamp01((rect.min.x - screen.min.x) / width),
            y: clamp01((rect.min.y - screen.min.y) / height),
            w: clamp01(rect.width() / width),
            h: clamp01(rect.height() / height),
        }
    }

    pub fn to_rect(&self, screen: Rect) -> Rect {
        let width = screen.width();
        let height = screen.height();
        let min = Pos2::new(
            screen.min.x + self.x * width,
            screen.min.y + self.y * height,
        );
        let size = Vec2::new(self.w * width, self.h * height);
        Rect::from_min_size(min, size)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutWindow {
    pub rect: NormalizedRect,
    #[serde(default)]
    pub collapsed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaneWorkspaceLayout {
    #[serde(default)]
    pub windows: Vec<PaneWorkspaceWindowLayout>,
    #[serde(default)]
    pub last_focused_window: Option<String>,
    #[serde(default)]
    pub last_focused_area: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaneWorkspaceWindowLayout {
    pub id: String,
    #[serde(default)]
    pub layout_managed: bool,
    #[serde(default)]
    pub rect: Option<NormalizedRect>,
    #[serde(default)]
    pub collapsed: bool,
    #[serde(default)]
    pub areas: Vec<PaneWorkspaceAreaLayout>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaneWorkspaceAreaLayout {
    pub id: u64,
    pub rect: NormalizedRect,
    #[serde(default)]
    pub tabs: Vec<PaneWorkspaceTabLayout>,
    #[serde(default)]
    pub active: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaneWorkspaceTabLayout {
    pub id: u64,
    pub title: String,
    pub kind: String,
    #[serde(default)]
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorLayout {
    pub name: String,
    pub windows: HashMap<String, LayoutWindow>,
    #[serde(default)]
    pub pane_workspace: Option<PaneWorkspaceLayout>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct LayoutsFile {
    active: Option<String>,
    layouts: Vec<EditorLayout>,
    #[serde(default = "default_layout_move_enabled")]
    allow_layout_move: bool,
    #[serde(default = "default_layout_resize_enabled")]
    allow_layout_resize: bool,
    #[serde(default = "default_live_reflow_enabled")]
    live_reflow: bool,
}

#[derive(Debug, Clone)]
pub enum LayoutSaveRequest {
    SaveActive,
    SaveAs(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutDragMode {
    None,
    Resize,
    Move,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LayoutDragEdges {
    pub left: bool,
    pub right: bool,
    pub top: bool,
    pub bottom: bool,
}

impl LayoutDragEdges {
    pub fn any(&self) -> bool {
        self.left || self.right || self.top || self.bottom
    }
}

#[derive(Resource, Debug, Clone)]
pub struct EditorLayoutState {
    pub active: Option<String>,
    pub layouts: HashMap<String, EditorLayout>,
    pub new_layout_name: String,
    pub rename_layout_name: String,
    pub apply_requested: bool,
    pub save_request: Option<LayoutSaveRequest>,
    pub last_screen_rect: Option<Rect>,
    pub last_project_open: Option<bool>,
    pub last_active_layout: Option<String>,
    pub default_runtime_layout: Option<EditorLayout>,
    pub allow_layout_move: bool,
    pub allow_layout_resize: bool,
    pub live_reflow: bool,
    pub layout_applied_this_frame: bool,
    pub layout_verify_pending: bool,
    pub layout_verify_attempts: u8,
    pub layout_dragging_window: Option<String>,
    pub layout_drag_mode: LayoutDragMode,
    pub layout_drag_start_pos: Option<Pos2>,
    pub layout_drag_start_rect: Option<Rect>,
    pub layout_drag_start_layout: Option<HashMap<String, Rect>>,
    pub layout_drag_edges: LayoutDragEdges,
}

impl Default for EditorLayoutState {
    fn default() -> Self {
        Self {
            active: None,
            layouts: HashMap::new(),
            new_layout_name: String::new(),
            rename_layout_name: String::new(),
            apply_requested: false,
            save_request: None,
            last_screen_rect: None,
            last_project_open: None,
            last_active_layout: None,
            default_runtime_layout: None,
            allow_layout_move: default_layout_move_enabled(),
            allow_layout_resize: default_layout_resize_enabled(),
            live_reflow: default_live_reflow_enabled(),
            layout_applied_this_frame: false,
            layout_verify_pending: false,
            layout_verify_attempts: 0,
            layout_dragging_window: None,
            layout_drag_mode: LayoutDragMode::None,
            layout_drag_start_pos: None,
            layout_drag_start_rect: None,
            layout_drag_start_layout: None,
            layout_drag_edges: LayoutDragEdges::default(),
        }
    }
}

pub fn default_layout() -> EditorLayout {
    let top: f32 = 0.085;
    let center_top = top;
    let bottom: f32 = 0.28;
    let side: f32 = 0.22;
    let center_height = (1.0_f32 - center_top - bottom).max(0.2_f32);
    let left_height = (1.0_f32 - bottom).max(0.2_f32);
    let center_width = (1.0_f32 - side - side).max(0.2_f32);
    let toolbar_x = (1.0_f32 - center_width) * 0.5_f32;

    let mut windows = HashMap::new();
    windows.insert(
        "Toolbar".to_string(),
        LayoutWindow {
            rect: NormalizedRect {
                x: toolbar_x,
                y: 0.0,
                w: center_width,
                h: top,
            },
            collapsed: false,
        },
    );
    windows.insert(
        "Hierarchy".to_string(),
        LayoutWindow {
            rect: NormalizedRect {
                x: 0.0,
                y: 0.0,
                w: side,
                h: left_height,
            },
            collapsed: false,
        },
    );
    windows.insert(
        "Inspector".to_string(),
        LayoutWindow {
            rect: NormalizedRect {
                x: 1.0 - side,
                y: 0.0,
                w: side,
                h: 1.0,
            },
            collapsed: false,
        },
    );
    windows.insert(
        "Content Browser".to_string(),
        LayoutWindow {
            rect: NormalizedRect {
                x: 0.0,
                y: 1.0 - bottom,
                w: 1.0 - side,
                h: bottom,
            },
            collapsed: false,
        },
    );
    windows.insert(
        "Viewport".to_string(),
        LayoutWindow {
            rect: NormalizedRect {
                x: side,
                y: center_top,
                w: center_width,
                h: center_height,
            },
            collapsed: false,
        },
    );

    let pane_workspace = PaneWorkspaceLayout {
        windows: vec![
            PaneWorkspaceWindowLayout {
                id: "Toolbar".to_string(),
                layout_managed: true,
                rect: None,
                collapsed: false,
                areas: vec![PaneWorkspaceAreaLayout {
                    id: 1,
                    rect: NormalizedRect {
                        x: 0.0,
                        y: 0.0,
                        w: 1.0,
                        h: 1.0,
                    },
                    tabs: vec![PaneWorkspaceTabLayout {
                        id: 1,
                        title: "Toolbar".to_string(),
                        kind: "toolbar".to_string(),
                        path: None,
                    }],
                    active: 0,
                }],
            },
            PaneWorkspaceWindowLayout {
                id: "Viewport".to_string(),
                layout_managed: true,
                rect: None,
                collapsed: false,
                areas: vec![PaneWorkspaceAreaLayout {
                    id: 2,
                    rect: NormalizedRect {
                        x: 0.0,
                        y: 0.0,
                        w: 1.0,
                        h: 1.0,
                    },
                    tabs: vec![
                        PaneWorkspaceTabLayout {
                            id: 2,
                            title: "Viewport".to_string(),
                            kind: "viewport".to_string(),
                            path: None,
                        },
                        PaneWorkspaceTabLayout {
                            id: 3,
                            title: "Play Viewport".to_string(),
                            kind: "play_viewport".to_string(),
                            path: None,
                        },
                    ],
                    active: 0,
                }],
            },
            PaneWorkspaceWindowLayout {
                id: "Content Browser".to_string(),
                layout_managed: true,
                rect: None,
                collapsed: false,
                areas: vec![PaneWorkspaceAreaLayout {
                    id: 3,
                    rect: NormalizedRect {
                        x: 0.0,
                        y: 0.0,
                        w: 1.0,
                        h: 1.0,
                    },
                    tabs: vec![
                        PaneWorkspaceTabLayout {
                            id: 4,
                            title: "Content Browser".to_string(),
                            kind: "content_browser".to_string(),
                            path: None,
                        },
                        PaneWorkspaceTabLayout {
                            id: 5,
                            title: "Console".to_string(),
                            kind: "console".to_string(),
                            path: None,
                        },
                    ],
                    active: 0,
                }],
            },
            PaneWorkspaceWindowLayout {
                id: "Hierarchy".to_string(),
                layout_managed: true,
                rect: None,
                collapsed: false,
                areas: vec![PaneWorkspaceAreaLayout {
                    id: 4,
                    rect: NormalizedRect {
                        x: 0.0,
                        y: 0.0,
                        w: 1.0,
                        h: 1.0,
                    },
                    tabs: vec![PaneWorkspaceTabLayout {
                        id: 6,
                        title: "Hierarchy".to_string(),
                        kind: "hierarchy".to_string(),
                        path: None,
                    }],
                    active: 0,
                }],
            },
            PaneWorkspaceWindowLayout {
                id: "Inspector".to_string(),
                layout_managed: true,
                rect: None,
                collapsed: false,
                areas: vec![PaneWorkspaceAreaLayout {
                    id: 5,
                    rect: NormalizedRect {
                        x: 0.0,
                        y: 0.0,
                        w: 1.0,
                        h: 1.0,
                    },
                    tabs: vec![PaneWorkspaceTabLayout {
                        id: 7,
                        title: "Inspector".to_string(),
                        kind: "inspector".to_string(),
                        path: None,
                    }],
                    active: 0,
                }],
            },
        ],
        last_focused_window: Some("Viewport".to_string()),
        last_focused_area: Some(2),
    };
    EditorLayout {
        name: "Default".to_string(),
        windows,
        pane_workspace: Some(pane_workspace),
    }
}

pub fn load_layout_state() -> EditorLayoutState {
    let mut state = EditorLayoutState::default();
    let mut loaded = false;

    if let Some(path) = layouts_path() {
        if let Ok(data) = fs::read_to_string(&path) {
            if let Ok(file) = ron::de::from_str::<LayoutsFile>(&data) {
                loaded = true;
                for layout in file.layouts {
                    state.layouts.insert(layout.name.clone(), layout);
                }
                state.active = file.active;
                state.allow_layout_move = file.allow_layout_move;
                state.allow_layout_resize = file.allow_layout_resize;
                state.live_reflow = file.live_reflow;
            }
        }
    }

    let default = default_layout();
    let default_name = default.name.clone();
    state.layouts.insert(default_name.clone(), default);

    if !loaded {
        state.active = Some(default_name);
        state.apply_requested = true;
        return state;
    }

    if let Some(active) = state.active.clone() {
        if !state.layouts.contains_key(&active) {
            state.active = Some(default_name);
            state.apply_requested = true;
        }
    }
    state
}

pub fn save_layouts(state: &EditorLayoutState) -> Result<(), String> {
    let Some(path) = layouts_path() else {
        return Ok(());
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }

    let mut layouts = state.layouts.values().cloned().collect::<Vec<_>>();
    layouts.sort_by(|a, b| a.name.cmp(&b.name));

    let payload = LayoutsFile {
        active: state.active.clone(),
        layouts,
        allow_layout_move: state.allow_layout_move,
        allow_layout_resize: state.allow_layout_resize,
        live_reflow: state.live_reflow,
    };

    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(4)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(&payload, pretty).map_err(|err| err.to_string())?;
    fs::write(path, data).map_err(|err| err.to_string())
}

pub fn capture_layout(
    name: String,
    window_rects: &HashMap<String, Rect>,
    window_collapsed: &HashMap<String, bool>,
    screen_rect: Rect,
    pane_workspace: Option<PaneWorkspaceLayout>,
) -> EditorLayout {
    let mut windows = HashMap::new();
    for id in layout_window_ids() {
        if let Some(rect) = window_rects.get(*id) {
            let collapsed = window_collapsed.get(*id).copied().unwrap_or(false);
            windows.insert(
                (*id).to_string(),
                LayoutWindow {
                    rect: NormalizedRect::from_rect(*rect, screen_rect),
                    collapsed,
                },
            );
        }
    }

    EditorLayout {
        name,
        windows,
        pane_workspace,
    }
}

pub fn layout_window_ids() -> &'static [&'static str] {
    &[
        "Toolbar",
        "Viewport",
        "Content Browser",
        "Hierarchy",
        "Inspector",
        "Project",
        "History",
    ]
}

fn layouts_path() -> Option<PathBuf> {
    let home = env::var("HOME").or_else(|_| env::var("USERPROFILE")).ok()?;
    Some(
        PathBuf::from(home)
            .join(".helmer_editor")
            .join(LAYOUTS_FILE_NAME),
    )
}

fn clamp01(value: f32) -> f32 {
    if value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}
