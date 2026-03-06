use bevy_ecs::{
    entity::Entity,
    name::Name,
    prelude::{Resource, World},
};
use glam::{Vec2, Vec3};
use helmer::graphics::common::renderer::GizmoMode;
use helmer::graphics::render_graphs::graph_templates;
use helmer::provided::components::Light;
use helmer_becs::physics::components::{ColliderShape, DynamicRigidBody, FixedCollider};
use helmer_becs::systems::scene_system::{EntityParent, SceneChild};
use helmer_becs::{
    BevyActiveCamera, BevyCamera, BevyInputManager, BevyLight, BevyMeshRenderer,
    BevySkinnedMeshRenderer, BevySpriteRenderer, BevyText2d, BevyTransform, BevyWrapper,
    provided::ui::inspector::InspectorSelectedEntityResource, ui_integration::UiRuntimeState,
};
use helmer_ui::{
    RetainedUi, RetainedUiNode, UiButton, UiButtonVariant, UiColor, UiDimension, UiId, UiLabel,
    UiLayoutBuilder, UiRect, UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle,
    UiWidget, estimate_char_advance,
};
use helmer_ui_docking::{DockAxis, DockLeafLayout, DockSplitHandleLayout, DockTab};
use helmer_ui_graph::{GraphInteractionInput, GraphPreviewEdge, GraphState};
use helmer_ui_layout::{LayoutRect, WindowFrame};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};
use walkdir::WalkDir;

use super::workspace::{EditorPaneKind, EditorPaneTab, EditorPaneWindow, EditorPaneWorkspaceState};
use super::{
    docking::{EditorDockWorkspace, EditorRetainedDockingState},
    graph::{
        EditorRetainedGraphInteractionState, EditorRetainedGraphRenderer, EditorRetainedGraphState,
    },
    layout::{EditorRetainedLayoutCatalog, EditorRetainedLayoutState},
    panes::{
        build_audio_mixer_pane, build_console_pane, build_content_browser_pane,
        build_hierarchy_pane, build_history_pane, build_inspector_pane, build_material_editor_pane,
        build_profiler_pane, build_project_pane, build_timeline_pane, build_toolbar_pane,
        build_viewport_pane, build_visual_scripting_pane, ensure_seed_graph, graph_id_for_tab,
    },
    shell_actions::{
        apply_audio_mixer_action, apply_console_action, apply_content_browser_action,
        apply_hierarchy_action, apply_history_action, apply_inspector_action,
        apply_material_editor_action, apply_profiler_action, apply_project_action,
        apply_timeline_action, apply_toolbar_action, apply_viewport_action,
        extend_content_browser_drag_selection, reparent_hierarchy_entity,
    },
    shell_data::{
        build_audio_mixer_pane_data, build_console_pane_data, build_content_browser_pane_data,
        build_hierarchy_pane_data, build_history_pane_data, build_inspector_pane_data,
        build_material_editor_pane_data, build_profiler_pane_data, build_project_pane_data,
        build_timeline_pane_data, build_toolbar_pane_data, build_viewport_pane_data,
    },
    state::{
        AssetBrowserState, AssetClipboardMode, AssetCreateKind, EditorAssetClipboardState,
        EditorCommand, EditorCommandQueue, EditorConsoleLevel, EditorConsoleState, EditorEntity,
        EditorProject, EditorRetainedGizmoSnapSettings, EditorRetainedViewportStates,
        EditorSceneState, FREECAM_BOOST_MULTIPLIER_DEFAULT, FREECAM_MOVE_ACCEL_DEFAULT,
        FREECAM_MOVE_DECEL_DEFAULT, FREECAM_ORBIT_DISTANCE_DEFAULT,
        FREECAM_ORBIT_PAN_SENSITIVITY_DEFAULT, FREECAM_PAN_SENSITIVITY_DEFAULT,
        FREECAM_SENSITIVITY_DEFAULT, FREECAM_SMOOTHING_DEFAULT, FREECAM_SPEED_MAX_DEFAULT,
        FREECAM_SPEED_MIN_DEFAULT, FREECAM_SPEED_STEP_DEFAULT, RetainedEditorViewportCamera,
        RetainedViewportFreecamSpeed, RetainedViewportResolutionPreset, SpawnKind,
        ensure_default_pane_workspace,
    },
};
use crate::retained::panes::{
    AudioMixerPaneAction, ConsolePaneAction, ConsolePaneTextField, ContentBrowserPaneAction,
    ContentBrowserPaneTextField, HierarchyPaneAction, HistoryPaneAction, InspectorPaneAction,
    InspectorPaneDragAction, InspectorPaneTextField, MaterialEditorPaneAction, ProfilerPaneAction,
    ProjectPaneAction, ProjectPaneTextField, TimelinePaneAction, ToolbarPaneAction,
    ViewportPaneAction, ViewportPaneMode, ViewportPaneSurfaceKey, rgb_to_hsv,
};

const RETAINED_SHELL_ROOT: UiId = UiId::from_raw(0x9e5a_73d6_01bf_82a1);
const WINDOW_TITLE_BAR_HEIGHT: f32 = 18.0;
const WINDOW_RESIZE_GRAB_RADIUS: f32 = 6.0;
const WINDOW_TAB_STRIP_HEIGHT: f32 = 20.0;
const WINDOW_TAB_MIN_WIDTH: f32 = 72.0;
const WINDOW_TAB_MAX_WIDTH: f32 = 220.0;
const WINDOW_CONTROL_SIZE: f32 = 10.0;
const WINDOW_CONTROL_GAP: f32 = 2.0;
const WINDOW_CONTROL_MARGIN: f32 = 3.0;
const CONTENT_BROWSER_DOUBLE_CLICK_THRESHOLD: Duration = Duration::from_millis(350);
const PROJECT_TEXT_FIELD_FONT_SIZE: f32 = 12.0;
const CONTENT_BROWSER_TEXT_FIELD_FONT_SIZE: f32 = 11.0;
const INSPECTOR_NAME_TEXT_FIELD_FONT_SIZE: f32 = 12.0;
const INSPECTOR_NUMERIC_TEXT_FIELD_FONT_SIZE: f32 = 11.0;
const CONSOLE_TEXT_FIELD_FONT_SIZE: f32 = 11.0;

#[derive(Clone, Copy, Debug)]
pub struct InspectorLightColorPickerState {
    pub entity: Entity,
    pub hue: f32,
}

#[derive(Resource, Clone, Debug)]
pub struct EditorRetainedUiMode {
    pub enabled: bool,
}

impl Default for EditorRetainedUiMode {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Resource, Clone, Debug, Default)]
struct HierarchyClipboardState {
    entity: Option<Entity>,
    cut: bool,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct EditorRetainedPaneInteractionState {
    pub hierarchy_click_targets: HashMap<UiId, Entity>,
    pub hierarchy_actions: HashMap<UiId, HierarchyPaneAction>,
    pub hierarchy_drop_surface_hits: HashSet<UiId>,
    pub hierarchy_expanded: HashSet<Entity>,
    pub hierarchy_scroll: f32,
    hierarchy_drag: Option<HierarchyDragState>,
    pub project_actions: HashMap<UiId, ProjectPaneAction>,
    project_text_fields: HashMap<UiId, ProjectPaneTextField>,
    pub project_text_cursors: HashMap<ProjectPaneTextField, usize>,
    pub focused_project_text_field: Option<ProjectPaneTextField>,
    pub content_browser_actions: HashMap<UiId, ContentBrowserPaneAction>,
    content_browser_text_fields: HashMap<UiId, ContentBrowserPaneTextField>,
    pub content_browser_text_cursors: HashMap<ContentBrowserPaneTextField, usize>,
    pub focused_content_browser_text_field: Option<ContentBrowserPaneTextField>,
    content_browser_viewports: Vec<UiRect>,
    content_browser_scroll_regions: Vec<UiRect>,
    content_browser_scroll_max: f32,
    content_browser_drag_select_active: bool,
    content_browser_drag_select_additive: bool,
    content_browser_entry_drag: Option<ContentEntryDragState>,
    pub inspector_actions: HashMap<UiId, InspectorPaneAction>,
    inspector_drag_actions: HashMap<UiId, InspectorPaneDragAction>,
    inspector_drag: Option<InspectorDragState>,
    inspector_text_fields: HashMap<UiId, InspectorPaneTextField>,
    pub inspector_text_cursors: HashMap<InspectorPaneTextField, usize>,
    pub focused_inspector_text_field: Option<InspectorPaneTextField>,
    pub inspector_light_color_picker: Option<InspectorLightColorPickerState>,
    pub console_actions: HashMap<UiId, ConsolePaneAction>,
    console_text_fields: HashMap<UiId, ConsolePaneTextField>,
    pub console_text_cursors: HashMap<ConsolePaneTextField, usize>,
    pub focused_console_text_field: Option<ConsolePaneTextField>,
    console_viewports: Vec<UiRect>,
    console_scroll_regions: Vec<UiRect>,
    console_scroll_max: f32,
    pub timeline_actions: HashMap<UiId, TimelinePaneAction>,
    pub toolbar_actions: HashMap<UiId, ToolbarPaneAction>,
    pub viewport_actions: HashMap<UiId, ViewportPaneAction>,
    pub viewport_surfaces: HashMap<ViewportPaneSurfaceKey, UiRect>,
    viewport_preview_drag: Option<ViewportPreviewDragState>,
    dock_tab_actions: HashMap<UiId, DockTabClickAction>,
    dock_tab_context_targets: HashMap<UiId, DockTabContextTarget>,
    dock_tab_bar_context_targets: HashMap<UiId, DockTabBarContextTarget>,
    dock_split_actions: HashMap<UiId, DockSplitHandleAction>,
    pub history_actions: HashMap<UiId, HistoryPaneAction>,
    pub audio_mixer_actions: HashMap<UiId, AudioMixerPaneAction>,
    pub profiler_actions: HashMap<UiId, ProfilerPaneAction>,
    pub material_editor_actions: HashMap<UiId, MaterialEditorPaneAction>,
    context_menu_actions: HashMap<UiId, ContextMenuAction>,
    context_menu_drag_actions: HashMap<UiId, ContextMenuDragAction>,
    context_menu_drag: Option<ContextMenuDragState>,
    pub window_title_hits: HashMap<UiId, UiId>,
    window_controls: HashMap<UiId, WindowControlAction>,
    collapsed_windows: HashSet<UiId>,
    collapsed_window_heights: HashMap<UiId, f32>,
    pub dragged_window: Option<UiId>,
    dragged_window_mode: WindowDragMode,
    dragged_window_edges: WindowResizeEdges,
    dragged_window_start_rect: Option<LayoutRect>,
    dragged_pointer_start: Vec2,
    dock_tab_drag: Option<DockTabDragState>,
    dock_split_drag: Option<DockSplitDragState>,
    primary_pointer_down_previous: bool,
    secondary_pointer_down_previous: bool,
    pointer_down_previous: bool,
    context_menu: Option<ContextMenuState>,
    context_menu_bounds: Option<UiRect>,
    last_project_open: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
struct HierarchyDragState {
    entity: Entity,
    pointer_start: Vec2,
    active: bool,
}

#[derive(Clone, Debug)]
struct ContentEntryDragState {
    primary_path: PathBuf,
    pointer_start: Vec2,
    active: bool,
}

#[derive(Clone, Copy, Debug)]
struct InspectorDragState {
    action: InspectorPaneDragAction,
    pointer_start: Vec2,
    last_pointer: Vec2,
    active: bool,
}

#[derive(Clone, Copy, Debug)]
struct ContextMenuDragState {
    action: ContextMenuDragAction,
    pointer_start: Vec2,
    last_pointer: Vec2,
    active: bool,
}

#[derive(Clone, Copy, Debug)]
enum ViewportPreviewDragKind {
    Move,
    Resize,
}

#[derive(Clone, Copy, Debug)]
struct ViewportPreviewDragState {
    tab_id: u64,
    mode: ViewportPaneMode,
    kind: ViewportPreviewDragKind,
    pointer_start: Vec2,
    scene_rect: UiRect,
    start_position_norm: [f32; 2],
    start_width_norm: f32,
    aspect_ratio: f32,
}

#[derive(Clone, Debug)]
struct RetainedLeafView {
    id: UiId,
    rect: UiRect,
    tabs: Vec<EditorPaneTab>,
    active_tab_id: Option<UiId>,
    active_tab: Option<EditorPaneTab>,
}

#[derive(Clone, Debug)]
struct RetainedWindowView {
    id: UiId,
    title: String,
    rect: LayoutRect,
    leaves: Vec<RetainedLeafView>,
    split_handles: Vec<DockSplitHandleLayout>,
}

#[derive(Clone, Debug)]
struct GraphRenderPayload {
    graph: GraphState,
    preview: Option<GraphPreviewEdge>,
}

#[derive(Clone, Copy, Debug)]
struct DockTabClickAction {
    workspace_id: UiId,
    tab_id: UiId,
}

#[derive(Clone, Copy, Debug)]
struct DockTabContextTarget {
    workspace_id: UiId,
    tab_id: UiId,
}

#[derive(Clone, Copy, Debug)]
struct DockTabBarContextTarget {
    workspace_id: UiId,
}

#[derive(Clone, Debug)]
struct DockSplitHandleAction {
    workspace_id: UiId,
    path: Vec<u8>,
    axis: DockAxis,
    parent_rect: UiRect,
}

#[derive(Clone, Debug)]
enum ContextMenuAction {
    Noop,
    HierarchyAddEntity,
    HierarchySpawn(SpawnKind),
    HierarchySetActiveCamera(Entity),
    HierarchySelectSceneRoot(Entity),
    HierarchyDeleteEntity(Entity),
    HierarchyDeleteHierarchy(Entity),
    HierarchyUnparent(Entity),
    HierarchyFocus(Entity),
    HierarchyCopy(Entity),
    HierarchyCut(Entity),
    HierarchyPaste(Option<Entity>),
    HierarchyDuplicate(Entity),
    HierarchyRename(Entity),
    DockCloseTab {
        workspace_id: UiId,
        tab_id: UiId,
    },
    DockDetachTab {
        workspace_id: UiId,
        tab_id: UiId,
    },
    DockOpenTab {
        workspace_id: UiId,
        kind: EditorPaneKind,
    },
    DockOpenTabInNewWindow {
        workspace_id: UiId,
        kind: EditorPaneKind,
    },
    ContentOpen {
        path: PathBuf,
        is_dir: bool,
    },
    ContentOpenScene(PathBuf),
    ContentAddModelToScene(PathBuf),
    ContentOpenInFileBrowser(PathBuf),
    ContentCopyPaths {
        anchor: Option<PathBuf>,
        mode: AssetClipboardMode,
    },
    ContentPaste {
        target: Option<PathBuf>,
    },
    ContentDuplicatePaths {
        anchor: Option<PathBuf>,
    },
    ContentDeletePaths {
        anchor: Option<PathBuf>,
    },
    ContentCreateAsset {
        directory: PathBuf,
        name: &'static str,
        kind: AssetCreateKind,
    },
    ContentSelectFolder(PathBuf),
    ContentRefresh,
    InspectorAddTransform(Entity),
    InspectorAddCamera(Entity),
    InspectorAddDirectionalLight(Entity),
    InspectorAddPointLight(Entity),
    InspectorAddSpotLight(Entity),
    InspectorAddMeshRenderer(Entity),
    ViewportSetResolution {
        tab_id: u64,
        preset: RetainedViewportResolutionPreset,
    },
    ViewportSetGraphTemplate {
        tab_id: u64,
        template: Option<String>,
    },
    ViewportToggleExecuteScriptsInEditMode {
        tab_id: u64,
    },
    ViewportToggleGizmosInPlay {
        tab_id: u64,
    },
    ViewportToggleShowCameraGizmos {
        tab_id: u64,
    },
    ViewportToggleShowDirectionalLightGizmos {
        tab_id: u64,
    },
    ViewportToggleShowPointLightGizmos {
        tab_id: u64,
    },
    ViewportToggleShowSpotLightGizmos {
        tab_id: u64,
    },
    ViewportToggleShowSplinePaths {
        tab_id: u64,
    },
    ViewportToggleShowSplinePoints {
        tab_id: u64,
    },
    ViewportToggleShowNavigationGizmo {
        tab_id: u64,
    },
    ViewportSetGizmoMode {
        tab_id: u64,
        mode: GizmoMode,
    },
    ViewportToggleGizmoSnapEnabled,
    ViewportToggleGizmoSnapCtrlToggles,
    ViewportToggleGizmoSnapShiftFine,
    ViewportToggleGizmoSnapAltCoarse,
    ViewportSetFreecamSpeed {
        tab_id: u64,
        speed: RetainedViewportFreecamSpeed,
    },
    ViewportResetFreecam {
        tab_id: u64,
    },
    ViewportAdjustFreecamSensitivity {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustFreecamSmoothing {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustFreecamMoveAccel {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustFreecamMoveDecel {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustFreecamSpeedStep {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustFreecamSpeedMin {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustFreecamSpeedMax {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustFreecamBoostMultiplier {
        tab_id: u64,
        delta: f32,
    },
    ViewportSetOrbitSelected {
        tab_id: u64,
        orbit_selected_entity: bool,
    },
    ViewportAdjustOrbitDistance {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustOrbitPanSensitivity {
        tab_id: u64,
        delta: f32,
    },
    ViewportAdjustPanSensitivity {
        tab_id: u64,
        delta: f32,
    },
    ViewportResetView {
        tab_id: u64,
    },
    ViewportFrameSelection {
        tab_id: u64,
    },
    LayoutTileColumns(usize),
    LayoutTileRows(usize),
    LayoutCascade,
    LayoutNormalize,
    LayoutActivate(String),
    LayoutDeactivate,
    LayoutSaveActive,
    LayoutSaveAsNew,
    LayoutDeleteActive,
    LayoutToggleMove,
    LayoutToggleResize,
    LayoutToggleLiveReflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContextMenuDragAction {
    ViewportFreecamSensitivity { tab_id: u64 },
    ViewportFreecamSmoothing { tab_id: u64 },
    ViewportFreecamMoveAccel { tab_id: u64 },
    ViewportFreecamMoveDecel { tab_id: u64 },
    ViewportFreecamSpeedStep { tab_id: u64 },
    ViewportFreecamSpeedMin { tab_id: u64 },
    ViewportFreecamSpeedMax { tab_id: u64 },
    ViewportFreecamBoostMultiplier { tab_id: u64 },
    ViewportOrbitDistance { tab_id: u64 },
    ViewportOrbitPanSensitivity { tab_id: u64 },
    ViewportPanSensitivity { tab_id: u64 },
    GizmoSnapTranslateStep,
    GizmoSnapRotateStepDegrees,
    GizmoSnapScaleStep,
    GizmoSnapFineScale,
    GizmoSnapCoarseScale,
}

#[derive(Clone, Copy, Debug)]
struct ViewportMenuTarget {
    tab_id: u64,
    mode: ViewportPaneMode,
}

#[derive(Clone, Debug)]
enum ContextMenuKind {
    HierarchyEntity(Entity),
    HierarchySurface,
    InspectorAddComponent(Entity),
    ViewportCanvas(ViewportMenuTarget),
    ViewportRender(ViewportMenuTarget),
    ViewportScripting(ViewportMenuTarget),
    ViewportGizmos(ViewportMenuTarget),
    ViewportFreecam(ViewportMenuTarget),
    ViewportOrbit(ViewportMenuTarget),
    ViewportAdvanced(ViewportMenuTarget),
    DockTab { workspace_id: UiId, tab_id: UiId },
    DockTabBar { workspace_id: UiId },
    ContentEntry { path: PathBuf, is_dir: bool },
    ContentSurface,
    ToolbarLayout,
}

#[derive(Clone, Debug)]
struct ContextMenuState {
    position: Vec2,
    kind: ContextMenuKind,
    open_submenu: Option<usize>,
}

#[derive(Clone, Debug)]
enum ContextMenuEntryRow {
    Action(String, ContextMenuAction),
    Submenu(String, Vec<(String, ContextMenuAction)>),
    DragValue(ContextMenuDragValueRow),
    Separator,
}

#[derive(Clone, Debug)]
struct ContextMenuDragValueRow {
    label: String,
    value_text: String,
    action: ContextMenuDragAction,
}

#[derive(Clone, Copy, Debug, Default)]
struct HierarchyContextInfo {
    can_unparent: bool,
    can_set_active_camera: bool,
    scene_root: Option<Entity>,
    has_children: bool,
}

#[derive(Clone, Copy, Debug)]
struct DockTabDragState {
    source_workspace_id: UiId,
    tab_id: UiId,
    pointer_start: Vec2,
    active: bool,
}

#[derive(Clone, Debug)]
struct DockSplitDragState {
    workspace_id: UiId,
    path: Vec<u8>,
    axis: DockAxis,
    parent_rect: UiRect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DockDropZone {
    Center,
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy, Debug)]
struct DockDropTarget {
    window_id: UiId,
    leaf_index: usize,
    leaf_focus_tab: Option<UiId>,
    zone: DockDropZone,
}

#[derive(Clone, Copy, Debug)]
enum WindowControlAction {
    ToggleCollapsed(UiId),
    Close(UiId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum WindowDragMode {
    #[default]
    None,
    Move,
    Resize,
}

#[derive(Clone, Copy, Debug, Default)]
struct WindowResizeEdges {
    left: bool,
    right: bool,
    top: bool,
    bottom: bool,
}

impl WindowResizeEdges {
    fn any(self) -> bool {
        self.left || self.right || self.top || self.bottom
    }
}

pub fn editor_retained_shell_system(world: &mut World) {
    let mode_enabled = world
        .get_resource::<EditorRetainedUiMode>()
        .map(|mode| mode.enabled)
        .unwrap_or(false);

    let Some(mut ui_runtime_state) = world.get_resource_mut::<UiRuntimeState>() else {
        return;
    };
    if !mode_enabled {
        ui_runtime_state
            .runtime_mut()
            .retained_mut()
            .remove_subtree(RETAINED_SHELL_ROOT);
        drop(ui_runtime_state);

        if let Some(mut pane_interaction_state) =
            world.get_resource_mut::<EditorRetainedPaneInteractionState>()
        {
            pane_interaction_state.hierarchy_click_targets.clear();
            pane_interaction_state.hierarchy_actions.clear();
            pane_interaction_state.hierarchy_drop_surface_hits.clear();
            pane_interaction_state.hierarchy_expanded.clear();
            pane_interaction_state.hierarchy_scroll = 0.0;
            pane_interaction_state.hierarchy_drag = None;
            pane_interaction_state.project_actions.clear();
            pane_interaction_state.project_text_fields.clear();
            pane_interaction_state.project_text_cursors.clear();
            pane_interaction_state.focused_project_text_field = None;
            pane_interaction_state.content_browser_actions.clear();
            pane_interaction_state.content_browser_text_fields.clear();
            pane_interaction_state.content_browser_text_cursors.clear();
            pane_interaction_state.focused_content_browser_text_field = None;
            pane_interaction_state.content_browser_viewports.clear();
            pane_interaction_state
                .content_browser_scroll_regions
                .clear();
            pane_interaction_state.content_browser_scroll_max = 0.0;
            pane_interaction_state.content_browser_drag_select_active = false;
            pane_interaction_state.content_browser_drag_select_additive = false;
            pane_interaction_state.content_browser_entry_drag = None;
            pane_interaction_state.inspector_actions.clear();
            pane_interaction_state.inspector_drag_actions.clear();
            pane_interaction_state.inspector_drag = None;
            pane_interaction_state.inspector_text_fields.clear();
            pane_interaction_state.inspector_text_cursors.clear();
            pane_interaction_state.focused_inspector_text_field = None;
            pane_interaction_state.inspector_light_color_picker = None;
            pane_interaction_state.console_actions.clear();
            pane_interaction_state.console_text_fields.clear();
            pane_interaction_state.console_text_cursors.clear();
            pane_interaction_state.focused_console_text_field = None;
            pane_interaction_state.console_viewports.clear();
            pane_interaction_state.console_scroll_regions.clear();
            pane_interaction_state.console_scroll_max = 0.0;
            pane_interaction_state.timeline_actions.clear();
            pane_interaction_state.toolbar_actions.clear();
            pane_interaction_state.viewport_actions.clear();
            pane_interaction_state.viewport_surfaces.clear();
            pane_interaction_state.viewport_preview_drag = None;
            pane_interaction_state.dock_tab_actions.clear();
            pane_interaction_state.dock_tab_context_targets.clear();
            pane_interaction_state.dock_tab_bar_context_targets.clear();
            pane_interaction_state.dock_split_actions.clear();
            pane_interaction_state.history_actions.clear();
            pane_interaction_state.audio_mixer_actions.clear();
            pane_interaction_state.profiler_actions.clear();
            pane_interaction_state.material_editor_actions.clear();
            pane_interaction_state.context_menu_actions.clear();
            pane_interaction_state.context_menu_drag_actions.clear();
            pane_interaction_state.context_menu_drag = None;
            pane_interaction_state.window_title_hits.clear();
            pane_interaction_state.window_controls.clear();
            pane_interaction_state.collapsed_windows.clear();
            pane_interaction_state.collapsed_window_heights.clear();
            pane_interaction_state.dragged_window = None;
            pane_interaction_state.dragged_window_mode = WindowDragMode::None;
            pane_interaction_state.dragged_window_edges = WindowResizeEdges::default();
            pane_interaction_state.dragged_window_start_rect = None;
            pane_interaction_state.dragged_pointer_start = Vec2::ZERO;
            pane_interaction_state.dock_tab_drag = None;
            pane_interaction_state.dock_split_drag = None;
            pane_interaction_state.primary_pointer_down_previous = false;
            pane_interaction_state.secondary_pointer_down_previous = false;
            pane_interaction_state.pointer_down_previous = false;
            pane_interaction_state.context_menu = None;
            pane_interaction_state.context_menu_bounds = None;
            pane_interaction_state.last_project_open = None;
        }
        if let Some(mut graph_interaction_state) =
            world.get_resource_mut::<EditorRetainedGraphInteractionState>()
        {
            graph_interaction_state.frames.clear();
            graph_interaction_state.controllers.clear();
            graph_interaction_state.pointer_down_previous = false;
        }
        return;
    }
    drop(ui_runtime_state);

    let (
        input_viewport_size,
        pointer_position,
        pointer_down,
        secondary_pointer_down,
        scroll_delta,
        input_shift_down,
        input_ctrl_down,
        input_just_pressed_keys,
    ) = world
        .get_resource::<BevyInputManager>()
        .map(|input| {
            let input = input.0.read();
            let shift_down = input
                .active_keys
                .contains(&winit::keyboard::KeyCode::ShiftLeft)
                || input
                    .active_keys
                    .contains(&winit::keyboard::KeyCode::ShiftRight);
            let ctrl_down = input
                .active_keys
                .contains(&winit::keyboard::KeyCode::ControlLeft)
                || input
                    .active_keys
                    .contains(&winit::keyboard::KeyCode::ControlRight);
            (
                Vec2::new(
                    input.window_size.x.max(1) as f32,
                    input.window_size.y.max(1) as f32,
                ),
                Some(Vec2::new(
                    input.cursor_position.x as f32,
                    input.cursor_position.y as f32,
                )),
                input
                    .active_mouse_buttons
                    .contains(&winit::event::MouseButton::Left),
                input
                    .active_mouse_buttons
                    .contains(&winit::event::MouseButton::Right)
                    || input
                        .active_mouse_buttons
                        .contains(&winit::event::MouseButton::Middle),
                input.mouse_wheel,
                shift_down,
                ctrl_down,
                input.just_pressed.iter().copied().collect::<Vec<_>>(),
            )
        })
        .unwrap_or((
            Vec2::new(1.0, 1.0),
            None,
            false,
            false,
            Vec2::ZERO,
            false,
            false,
            Vec::new(),
        ));

    let mut viewport_size = input_viewport_size;
    if viewport_size.x < 64.0 || viewport_size.y < 64.0 {
        if let Some(layout_state) = world.get_resource::<EditorRetainedLayoutState>()
            && layout_state.workspace_bounds.width >= 64.0
            && layout_state.workspace_bounds.height >= 64.0
        {
            viewport_size = Vec2::new(
                layout_state.workspace_bounds.width,
                layout_state.workspace_bounds.height,
            );
        }
    }
    if viewport_size.x < 64.0 || viewport_size.y < 64.0 {
        viewport_size = Vec2::new(1280.0, 720.0);
    }

    let ui_interaction = world
        .get_resource::<UiRuntimeState>()
        .map(|state| state.runtime().interaction())
        .unwrap_or_default();
    let project_open = world
        .get_resource::<EditorProject>()
        .is_some_and(|project| project.root.is_some());

    let mut pane_windows = world
        .get_resource_mut::<EditorPaneWorkspaceState>()
        .map(|mut workspace| {
            ensure_default_pane_workspace(&mut workspace);
            workspace.windows.clone()
        })
        .unwrap_or_default();
    let valid_viewport_tabs: HashSet<u64> = pane_windows
        .iter()
        .flat_map(|window| window.areas.iter())
        .flat_map(|area| area.tabs.iter())
        .filter(|tab| {
            matches!(
                tab.kind,
                EditorPaneKind::Viewport | EditorPaneKind::PlayViewport
            )
        })
        .map(|tab| tab.id)
        .collect();
    if let Some(mut viewport_states) = world.get_resource_mut::<EditorRetainedViewportStates>() {
        viewport_states
            .states
            .retain(|tab_id, _| valid_viewport_tabs.contains(tab_id));
    }
    let (
        previous_window_title_hits,
        previous_hierarchy_drop_surface_hits,
        mut hierarchy_drag_state,
        mut hierarchy_expanded,
        mut hierarchy_scroll,
        mut inspector_drag_state,
        mut context_menu_drag_state,
        mut project_text_cursors,
        mut focused_project_text_field,
        mut content_browser_text_cursors,
        mut focused_content_browser_text_field,
        previous_content_browser_viewports,
        previous_content_browser_scroll_regions,
        mut previous_content_browser_scroll_max,
        mut inspector_text_cursors,
        mut focused_inspector_text_field,
        mut inspector_light_color_picker,
        mut console_text_cursors,
        mut focused_console_text_field,
        previous_console_viewports,
        previous_console_scroll_regions,
        mut previous_console_scroll_max,
        previous_viewport_surfaces,
        mut viewport_preview_drag_state,
        mut content_browser_drag_select_active,
        mut content_browser_drag_select_additive,
        mut content_browser_entry_drag,
        mut previous_collapsed_windows,
        mut previous_collapsed_window_heights,
        mut dragged_window,
        mut dragged_window_mode,
        mut dragged_window_edges,
        mut dragged_window_start_rect,
        mut dragged_pointer_start,
        mut dock_tab_drag,
        mut dock_split_drag,
        primary_pointer_down_previous,
        secondary_pointer_down_previous,
        pointer_down_previous,
        mut context_menu_state,
        mut context_menu_bounds,
        last_project_open,
    ) = world
        .get_resource::<EditorRetainedPaneInteractionState>()
        .map(|state| {
            (
                state.window_title_hits.clone(),
                state.hierarchy_drop_surface_hits.clone(),
                state.hierarchy_drag,
                state.hierarchy_expanded.clone(),
                state.hierarchy_scroll,
                state.inspector_drag,
                state.context_menu_drag,
                state.project_text_cursors.clone(),
                state.focused_project_text_field,
                state.content_browser_text_cursors.clone(),
                state.focused_content_browser_text_field,
                state.content_browser_viewports.clone(),
                state.content_browser_scroll_regions.clone(),
                state.content_browser_scroll_max,
                state.inspector_text_cursors.clone(),
                state.focused_inspector_text_field,
                state.inspector_light_color_picker,
                state.console_text_cursors.clone(),
                state.focused_console_text_field,
                state.console_viewports.clone(),
                state.console_scroll_regions.clone(),
                state.console_scroll_max,
                state.viewport_surfaces.clone(),
                state.viewport_preview_drag,
                state.content_browser_drag_select_active,
                state.content_browser_drag_select_additive,
                state.content_browser_entry_drag.clone(),
                state.collapsed_windows.clone(),
                state.collapsed_window_heights.clone(),
                state.dragged_window,
                state.dragged_window_mode,
                state.dragged_window_edges,
                state.dragged_window_start_rect,
                state.dragged_pointer_start,
                state.dock_tab_drag,
                state.dock_split_drag.clone(),
                state.primary_pointer_down_previous,
                state.secondary_pointer_down_previous,
                state.pointer_down_previous,
                state.context_menu.clone(),
                state.context_menu_bounds,
                state.last_project_open,
            )
        })
        .unwrap_or((
            HashMap::new(),
            HashSet::new(),
            None,
            HashSet::new(),
            0.0,
            None,
            None,
            HashMap::new(),
            None,
            HashMap::new(),
            None,
            Vec::new(),
            Vec::new(),
            0.0,
            HashMap::new(),
            None,
            None,
            HashMap::new(),
            None,
            Vec::new(),
            Vec::new(),
            0.0,
            HashMap::new(),
            None,
            false,
            false,
            None,
            HashSet::new(),
            HashMap::new(),
            None,
            WindowDragMode::None,
            WindowResizeEdges::default(),
            None,
            Vec2::ZERO,
            None,
            None,
            false,
            false,
            false,
            None,
            None,
            None,
        ));
    let primary_pointer_pressed = pointer_down && !primary_pointer_down_previous;
    let primary_pointer_released = !pointer_down && primary_pointer_down_previous;
    let secondary_pointer_pressed = secondary_pointer_down && !secondary_pointer_down_previous;
    let window_drag_pointer_down = pointer_down || secondary_pointer_down;
    let pointer_pressed = window_drag_pointer_down && !pointer_down_previous;
    let pointer_released = !window_drag_pointer_down && pointer_down_previous;

    if primary_pointer_pressed
        && let (Some(pointer), Some(hovered)) = (pointer_position, ui_interaction.hovered)
        && let Some(action) = world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .and_then(|state| state.dock_tab_actions.get(&hovered).copied())
    {
        dock_tab_drag = Some(DockTabDragState {
            source_workspace_id: action.workspace_id,
            tab_id: action.tab_id,
            pointer_start: pointer,
            active: false,
        });
    }

    if pointer_down && let Some(mut drag) = dock_tab_drag {
        if !drag.active
            && let Some(pointer) = pointer_position
            && (pointer - drag.pointer_start).length_squared() >= 36.0
        {
            drag.active = true;
        }
        dock_tab_drag = Some(drag);
    }

    if primary_pointer_pressed
        && let Some(hovered) = ui_interaction.hovered
        && let Some(action) = world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .and_then(|state| state.dock_split_actions.get(&hovered).cloned())
    {
        dock_split_drag = Some(DockSplitDragState {
            workspace_id: action.workspace_id,
            path: action.path,
            axis: action.axis,
            parent_rect: action.parent_rect,
        });
    }

    if pointer_down
        && let Some(pointer) = pointer_position
        && let Some(drag) = dock_split_drag.as_ref()
    {
        let ratio = match drag.axis {
            DockAxis::Horizontal => {
                if drag.parent_rect.height <= f32::EPSILON {
                    None
                } else {
                    Some((pointer.y - drag.parent_rect.y) / drag.parent_rect.height)
                }
            }
            DockAxis::Vertical => {
                if drag.parent_rect.width <= f32::EPSILON {
                    None
                } else {
                    Some((pointer.x - drag.parent_rect.x) / drag.parent_rect.width)
                }
            }
        };
        if let Some(ratio) = ratio {
            let workspace_id = drag.workspace_id;
            let path = drag.path.clone();
            world.resource_scope::<EditorRetainedDockingState, _>(|_world, mut docking_state| {
                if let Some(workspace) = docking_state.workspace_mut(workspace_id) {
                    workspace.docking.set_split_ratio(&path, ratio);
                }
            });
        }
    }

    let hovered_content_action = ui_interaction.hovered.and_then(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .and_then(|state| state.content_browser_actions.get(&hovered).cloned())
    });
    let hovered_hierarchy_action = ui_interaction.hovered.and_then(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .and_then(|state| state.hierarchy_actions.get(&hovered).cloned())
    });
    let hovered_viewport_action = ui_interaction.hovered.and_then(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .and_then(|state| state.viewport_actions.get(&hovered).cloned())
    });
    let hovered_content_hit = ui_interaction.hovered.is_some_and(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .is_some_and(|state| state.content_browser_actions.contains_key(&hovered))
    });
    let hovered_console_hit = ui_interaction.hovered.is_some_and(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .is_some_and(|state| state.console_actions.contains_key(&hovered))
    });
    let hovered_hierarchy_hit = ui_interaction.hovered.is_some_and(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .is_some_and(|state| state.hierarchy_actions.contains_key(&hovered))
    });
    let pointer_active_tab_kind = pointer_position
        .and_then(|pointer| pane_kind_under_pointer(world, &pane_windows, pointer, project_open));
    let pointer_in_previous_content_viewport = pointer_position.is_some_and(|pointer| {
        previous_content_browser_viewports
            .iter()
            .any(|rect| point_in_ui_rect(*rect, pointer))
    });
    let pointer_in_previous_content_scroll_region = pointer_position.is_some_and(|pointer| {
        previous_content_browser_scroll_regions
            .iter()
            .any(|rect| point_in_ui_rect(*rect, pointer))
    });
    let pointer_in_previous_console_viewport = pointer_position.is_some_and(|pointer| {
        previous_console_viewports
            .iter()
            .any(|rect| point_in_ui_rect(*rect, pointer))
    });
    let pointer_in_previous_console_scroll_region = pointer_position.is_some_and(|pointer| {
        previous_console_scroll_regions
            .iter()
            .any(|rect| point_in_ui_rect(*rect, pointer))
    });

    let scroll_lines = if scroll_delta.y.abs() > f32::EPSILON {
        scroll_delta.y
    } else {
        scroll_delta.x
    };

    if scroll_lines.abs() > f32::EPSILON
        && (hovered_content_hit
            || pointer_active_tab_kind == Some(EditorPaneKind::ContentBrowser)
            || pointer_in_previous_content_viewport
            || pointer_in_previous_content_scroll_region)
        && let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>()
    {
        let next_scroll = (browser.grid_scroll - scroll_lines * 36.0).max(0.0);
        browser.grid_scroll = next_scroll;
    }
    if scroll_lines.abs() > f32::EPSILON
        && (hovered_console_hit
            || pointer_active_tab_kind == Some(EditorPaneKind::Console)
            || pointer_in_previous_console_viewport
            || pointer_in_previous_console_scroll_region)
        && let Some(mut console) = world.get_resource_mut::<EditorConsoleState>()
    {
        console.auto_scroll = false;
        console.scroll = (console.scroll - scroll_lines * 28.0).max(0.0);
    }
    if scroll_lines.abs() > f32::EPSILON
        && (hovered_hierarchy_hit
            || (hovered_hierarchy_action.is_none()
                && pointer_active_tab_kind == Some(EditorPaneKind::Hierarchy)))
    {
        hierarchy_scroll = (hierarchy_scroll - scroll_lines * 28.0).max(0.0);
    }

    if pointer_down {
        let (slider_hit, grid_scrollbar_hit, console_scrollbar_hit) =
            world
                .get_resource::<EditorRetainedPaneInteractionState>()
                .map(|state| {
                    [ui_interaction.active, ui_interaction.hovered]
                        .into_iter()
                        .flatten()
                        .fold((None, None, None), |mut acc, hit| {
                            if let Some(action) = state.content_browser_actions.get(&hit) {
                                match action {
                                    ContentBrowserPaneAction::TileSizeSlider => acc.0 = Some(hit),
                                    ContentBrowserPaneAction::GridScrollbar => acc.1 = Some(hit),
                                    _ => {}
                                }
                            }
                            if state.console_actions.get(&hit).is_some_and(|action| {
                                matches!(action, ConsolePaneAction::LogScrollbar)
                            }) {
                                acc.2 = Some(hit);
                            }
                            acc
                        })
                })
                .unwrap_or((None, None, None));
        if let Some(hit) = slider_hit {
            update_content_browser_tile_size_from_pointer(world, Some(hit), pointer_position);
        }
        if let Some(hit) = grid_scrollbar_hit {
            update_content_browser_grid_scroll_from_pointer(
                world,
                Some(hit),
                pointer_position,
                previous_content_browser_scroll_max,
            );
        }
        if let Some(hit) = console_scrollbar_hit {
            update_console_scroll_from_pointer(
                world,
                Some(hit),
                pointer_position,
                previous_console_scroll_max,
            );
        }
    }

    if primary_pointer_pressed
        && let Some(pointer) = pointer_position
        && let Some(action) = hovered_viewport_action.as_ref()
    {
        let (tab_id, mode, kind) = match action {
            ViewportPaneAction::PreviewMove { tab_id, mode } => {
                (*tab_id, *mode, ViewportPreviewDragKind::Move)
            }
            ViewportPaneAction::PreviewResize { tab_id, mode } => {
                (*tab_id, *mode, ViewportPreviewDragKind::Resize)
            }
            _ => (0, ViewportPaneMode::Edit, ViewportPreviewDragKind::Move),
        };
        if matches!(
            action,
            ViewportPaneAction::PreviewMove { .. } | ViewportPaneAction::PreviewResize { .. }
        ) {
            let surface_key = ViewportPaneSurfaceKey { tab_id, mode };
            if let Some(scene_rect) = previous_viewport_surfaces.get(&surface_key).copied() {
                let viewport_state = world
                    .get_resource::<EditorRetainedViewportStates>()
                    .map(|states| states.state_for(tab_id))
                    .unwrap_or_default();
                let aspect_ratio = retained_preview_camera_aspect_ratio(world);
                viewport_preview_drag_state = Some(ViewportPreviewDragState {
                    tab_id,
                    mode,
                    kind,
                    pointer_start: pointer,
                    scene_rect,
                    start_position_norm: viewport_state.preview_position_norm,
                    start_width_norm: viewport_state.preview_width_norm,
                    aspect_ratio,
                });
            }
        }
    }

    if pointer_down
        && let Some(pointer) = pointer_position
        && let Some(drag) = viewport_preview_drag_state
    {
        if !matches!(drag.mode, ViewportPaneMode::Edit) {
            // preview camera overlay exists only for edit viewport tabs
            viewport_preview_drag_state = None;
        } else {
            let delta = pointer - drag.pointer_start;
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(drag.tab_id);
                let aspect = if drag.aspect_ratio.is_finite() && drag.aspect_ratio > 0.01 {
                    drag.aspect_ratio
                } else {
                    16.0 / 9.0
                };
                let start_rect = retained_preview_rect_from_state(
                    drag.scene_rect,
                    drag.start_position_norm,
                    drag.start_width_norm,
                    aspect,
                );
                match drag.kind {
                    ViewportPreviewDragKind::Move => {
                        let max_offset_x = (drag.scene_rect.width - start_rect.width).max(0.0);
                        let max_offset_y = (drag.scene_rect.height - start_rect.height).max(0.0);
                        let new_x = (start_rect.x + delta.x)
                            .clamp(drag.scene_rect.x, drag.scene_rect.x + max_offset_x);
                        let new_y = (start_rect.y + delta.y)
                            .clamp(drag.scene_rect.y, drag.scene_rect.y + max_offset_y);
                        let pos_norm_x = if max_offset_x <= f32::EPSILON {
                            0.0
                        } else {
                            ((new_x - drag.scene_rect.x) / max_offset_x).clamp(0.0, 1.0)
                        };
                        let pos_norm_y = if max_offset_y <= f32::EPSILON {
                            0.0
                        } else {
                            ((new_y - drag.scene_rect.y) / max_offset_y).clamp(0.0, 1.0)
                        };
                        state.preview_position_norm = [pos_norm_x, pos_norm_y];
                    }
                    ViewportPreviewDragKind::Resize => {
                        let min_preview_w = drag.scene_rect.width.min(120.0).max(72.0);
                        let resize_delta = delta.x + delta.y * aspect;
                        let max_w_from_pos = (drag.scene_rect.right() - start_rect.x)
                            .min((drag.scene_rect.bottom() - start_rect.y).max(1.0) * aspect);
                        let preview_w = (start_rect.width + resize_delta * 0.5)
                            .clamp(min_preview_w, max_w_from_pos.max(min_preview_w));
                        let preview_h = (preview_w / aspect).max(1.0);
                        state.preview_width_norm =
                            (preview_w / drag.scene_rect.width.max(1.0)).clamp(0.05, 0.95);
                        let offset_x_den = (drag.scene_rect.width - preview_w).max(1.0);
                        let offset_y_den = (drag.scene_rect.height - preview_h).max(1.0);
                        state.preview_position_norm = [
                            ((start_rect.x - drag.scene_rect.x) / offset_x_den).clamp(0.0, 1.0),
                            ((start_rect.y - drag.scene_rect.y) / offset_y_den).clamp(0.0, 1.0),
                        ];
                    }
                }
            }
        }
    }
    if primary_pointer_released {
        viewport_preview_drag_state = None;
    }

    if primary_pointer_pressed
        && let Some(action) = hovered_content_action.as_ref()
        && matches!(
            action,
            ContentBrowserPaneAction::GridSurface
                | ContentBrowserPaneAction::SelectEntry { .. }
                | ContentBrowserPaneAction::SelectFolder(_)
        )
    {
        content_browser_drag_select_active = true;
        content_browser_drag_select_additive = input_ctrl_down || input_shift_down;
        if let Some(pointer) = pointer_position {
            let drag_path = match action {
                ContentBrowserPaneAction::SelectEntry { path, .. } => Some(path.clone()),
                ContentBrowserPaneAction::SelectFolder(path) => Some(path.clone()),
                _ => None,
            };
            if let Some(path) = drag_path {
                content_browser_entry_drag = Some(ContentEntryDragState {
                    primary_path: path,
                    pointer_start: pointer,
                    active: false,
                });
            }
        }
        if matches!(action, ContentBrowserPaneAction::GridSurface)
            && !content_browser_drag_select_additive
            && let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>()
        {
            browser.selected = None;
            browser.selected_paths.clear();
            browser.selection_anchor = None;
        }
    }

    if pointer_down
        && let Some(mut drag) = content_browser_entry_drag.clone()
        && !drag.active
        && let Some(pointer) = pointer_position
        && (pointer - drag.pointer_start).length_squared() >= 36.0
    {
        drag.active = true;
        content_browser_entry_drag = Some(drag);
    }

    if pointer_down
        && content_browser_drag_select_active
        && let Some(ContentBrowserPaneAction::SelectEntry { path, .. }) =
            hovered_content_action.as_ref()
    {
        extend_content_browser_drag_selection(
            world,
            path.clone(),
            content_browser_drag_select_additive,
        );
    }
    if primary_pointer_released {
        if let Some(drag) = content_browser_entry_drag.as_ref()
            && drag.active
            && pointer_active_tab_kind == Some(EditorPaneKind::Inspector)
        {
            apply_content_drop_to_inspector(world, &drag.primary_path);
        }
        content_browser_entry_drag = None;
        content_browser_drag_select_active = false;
        content_browser_drag_select_additive = false;
        dock_split_drag = None;
    }

    let hovered_hierarchy_hit = ui_interaction.hovered.filter(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .is_some_and(|state| state.hierarchy_click_targets.contains_key(hovered))
    });
    let hovered_hierarchy_entity = ui_interaction.hovered.and_then(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .and_then(|state| state.hierarchy_click_targets.get(&hovered).copied())
    });
    let hovered_hierarchy_surface = ui_interaction
        .hovered
        .is_some_and(|hovered| previous_hierarchy_drop_surface_hits.contains(&hovered));
    if scroll_lines.abs() > f32::EPSILON && hovered_hierarchy_entity.is_some() {
        hierarchy_scroll = (hierarchy_scroll - scroll_lines * 28.0).max(0.0);
    }

    if primary_pointer_pressed
        && let (Some(pointer), Some(entity)) = (pointer_position, hovered_hierarchy_entity)
    {
        hierarchy_drag_state = Some(HierarchyDragState {
            entity,
            pointer_start: pointer,
            active: false,
        });
    }

    if pointer_down
        && let Some(mut drag) = hierarchy_drag_state
        && !drag.active
        && let Some(pointer) = pointer_position
        && (pointer - drag.pointer_start).length_squared() >= 36.0
    {
        drag.active = true;
        hierarchy_drag_state = Some(drag);
    }

    if primary_pointer_released
        && let Some(drag) = hierarchy_drag_state.take()
        && drag.active
    {
        let mut did_reparent = false;
        if pointer_active_tab_kind == Some(EditorPaneKind::Inspector) {
            if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>()
            {
                selected.0 = Some(drag.entity);
                did_reparent = true;
            }
        }
        if !did_reparent && let Some(target_entity) = hovered_hierarchy_entity {
            let sibling_drop = hovered_hierarchy_hit
                .and_then(|hit| {
                    world
                        .get_resource::<UiRuntimeState>()
                        .and_then(|state| state.runtime().layout_rect(hit))
                })
                .zip(pointer_position)
                .is_some_and(|(rect, pointer)| {
                    if rect.height <= f32::EPSILON {
                        false
                    } else {
                        let rel_y = ((pointer.y - rect.y) / rect.height).clamp(0.0, 1.0);
                        rel_y < 0.30 || rel_y > 0.70
                    }
                });
            let parent = if sibling_drop {
                world
                    .get::<EntityParent>(target_entity)
                    .map(|relation| relation.parent)
            } else {
                Some(target_entity)
            };
            if target_entity != drag.entity
                && parent != Some(drag.entity)
                && reparent_hierarchy_entity(world, drag.entity, parent)
            {
                did_reparent = true;
            }
        }
        if !did_reparent
            && (hovered_hierarchy_surface
                || pointer_active_tab_kind == Some(EditorPaneKind::Hierarchy))
        {
            let _ = reparent_hierarchy_entity(world, drag.entity, None);
        }
    }
    if primary_pointer_released {
        hierarchy_drag_state = None;
    }

    let hovered_inspector_drag_action = ui_interaction.hovered.and_then(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .and_then(|state| state.inspector_drag_actions.get(&hovered).copied())
    });

    if primary_pointer_pressed
        && let (Some(pointer), Some(action)) = (pointer_position, hovered_inspector_drag_action)
    {
        inspector_drag_state = Some(InspectorDragState {
            action,
            pointer_start: pointer,
            last_pointer: pointer,
            active: false,
        });
    }
    if pointer_down
        && let Some(mut drag) = inspector_drag_state
        && let Some(pointer) = pointer_position
    {
        let immediate = inspector_drag_starts_immediately(drag.action);
        if !drag.active && (immediate || (pointer - drag.pointer_start).length_squared() >= 9.0) {
            drag.active = true;
            drag.last_pointer = pointer;
        }
        if drag.active {
            let delta_pixels = pointer.x - drag.last_pointer.x;
            if inspector_drag_uses_pointer_position(drag.action) {
                apply_inspector_drag_delta(
                    world,
                    drag.action,
                    delta_pixels,
                    Some(pointer),
                    &mut inspector_light_color_picker,
                );
            } else if delta_pixels.abs() > f32::EPSILON {
                apply_inspector_drag_delta(
                    world,
                    drag.action,
                    delta_pixels,
                    Some(pointer),
                    &mut inspector_light_color_picker,
                );
            }
            drag.last_pointer = pointer;
        }
        inspector_drag_state = Some(drag);
    }
    if primary_pointer_released {
        inspector_drag_state = None;
    }

    let hovered_context_menu_drag_action = ui_interaction.hovered.and_then(|hovered| {
        world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .and_then(|state| state.context_menu_drag_actions.get(&hovered).copied())
    });

    if primary_pointer_pressed
        && let (Some(pointer), Some(action)) = (pointer_position, hovered_context_menu_drag_action)
    {
        context_menu_drag_state = Some(ContextMenuDragState {
            action,
            pointer_start: pointer,
            last_pointer: pointer,
            active: false,
        });
    }
    if pointer_down
        && let Some(mut drag) = context_menu_drag_state
        && let Some(pointer) = pointer_position
    {
        if !drag.active && (pointer - drag.pointer_start).length_squared() >= 9.0 {
            drag.active = true;
            drag.last_pointer = pointer;
        }
        if drag.active {
            let delta_pixels = pointer.x - drag.last_pointer.x;
            if delta_pixels.abs() > f32::EPSILON {
                apply_context_menu_drag_delta(world, drag.action, delta_pixels);
            }
            drag.last_pointer = pointer;
        }
        context_menu_drag_state = Some(drag);
    }
    if primary_pointer_released {
        context_menu_drag_state = None;
    }

    if let Some(clicked) = ui_interaction.clicked {
        let menu_was_open = context_menu_state.is_some();
        let mut context_menu_opened_this_click = false;
        let (
            clicked_entity,
            hierarchy_action,
            project_action,
            project_text_field,
            content_browser_text_field,
            content_action,
            inspector_text_field,
            inspector_action,
            console_text_field,
            console_action,
            timeline_action,
            toolbar_action,
            viewport_action,
            dock_tab_action,
            history_action,
            audio_mixer_action,
            profiler_action,
            material_editor_action,
            window_control_action,
            context_menu_action,
        ) = world
            .get_resource::<EditorRetainedPaneInteractionState>()
            .map(|state| {
                (
                    state.hierarchy_click_targets.get(&clicked).copied(),
                    state.hierarchy_actions.get(&clicked).cloned(),
                    state.project_actions.get(&clicked).cloned(),
                    state.project_text_fields.get(&clicked).copied(),
                    state.content_browser_text_fields.get(&clicked).copied(),
                    state.content_browser_actions.get(&clicked).cloned(),
                    state.inspector_text_fields.get(&clicked).copied(),
                    state.inspector_actions.get(&clicked).copied(),
                    state.console_text_fields.get(&clicked).copied(),
                    state.console_actions.get(&clicked).cloned(),
                    state.timeline_actions.get(&clicked).cloned(),
                    state.toolbar_actions.get(&clicked).cloned(),
                    state.viewport_actions.get(&clicked).cloned(),
                    state.dock_tab_actions.get(&clicked).copied(),
                    state.history_actions.get(&clicked).cloned(),
                    state.audio_mixer_actions.get(&clicked).cloned(),
                    state.profiler_actions.get(&clicked).cloned(),
                    state.material_editor_actions.get(&clicked).cloned(),
                    state.window_controls.get(&clicked).copied(),
                    state.context_menu_actions.get(&clicked).cloned(),
                )
            })
            .unwrap_or((
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None,
            ));

        let clicked_rect = world
            .get_resource::<UiRuntimeState>()
            .and_then(|state| state.runtime().layout_rect(clicked));

        if let Some(field) = project_text_field {
            focused_project_text_field = Some(field);
            focused_content_browser_text_field = None;
            focused_inspector_text_field = None;
            focused_console_text_field = None;
            let text = project_text_field_value(world, field);
            let cursor = click_cursor_for_text_field(
                pointer_position,
                clicked_rect,
                &text,
                PROJECT_TEXT_FIELD_FONT_SIZE,
            );
            project_text_cursors.insert(field, cursor);
        } else if let Some(field) = content_browser_text_field {
            focused_content_browser_text_field = Some(field);
            focused_project_text_field = None;
            focused_inspector_text_field = None;
            focused_console_text_field = None;
            let text = content_browser_text_field_value(world, field);
            let cursor = click_cursor_for_text_field(
                pointer_position,
                clicked_rect,
                &text,
                CONTENT_BROWSER_TEXT_FIELD_FONT_SIZE,
            );
            content_browser_text_cursors.insert(field, cursor);
        } else if let Some(field) = inspector_text_field {
            focused_inspector_text_field = Some(field);
            focused_project_text_field = None;
            focused_content_browser_text_field = None;
            focused_console_text_field = None;
            let text = inspector_text_field_value(world, field);
            let font_size = if matches!(field, InspectorPaneTextField::Name) {
                INSPECTOR_NAME_TEXT_FIELD_FONT_SIZE
            } else {
                INSPECTOR_NUMERIC_TEXT_FIELD_FONT_SIZE
            };
            let cursor =
                click_cursor_for_text_field(pointer_position, clicked_rect, &text, font_size);
            inspector_text_cursors.insert(field, cursor);
        } else if let Some(field) = console_text_field {
            focused_console_text_field = Some(field);
            focused_project_text_field = None;
            focused_content_browser_text_field = None;
            focused_inspector_text_field = None;
            let text = console_text_field_value(world, field);
            let cursor = click_cursor_for_text_field(
                pointer_position,
                clicked_rect,
                &text,
                CONSOLE_TEXT_FIELD_FONT_SIZE,
            );
            console_text_cursors.insert(field, cursor);
        } else {
            focused_project_text_field = None;
            focused_content_browser_text_field = None;
            focused_inspector_text_field = None;
            focused_console_text_field = None;
        }

        if let Some(entity) = clicked_entity {
            if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>()
            {
                selected.0 = Some(entity);
            }
            hierarchy_expand_ancestors(world, &mut hierarchy_expanded, entity);
        }
        if let Some(action) = hierarchy_action {
            match action {
                HierarchyPaneAction::AddEntity => {
                    if let Some(pointer) = pointer_position {
                        context_menu_state = Some(ContextMenuState {
                            position: pointer,
                            kind: ContextMenuKind::HierarchySurface,
                            open_submenu: None,
                        });
                        context_menu_bounds = None;
                        context_menu_opened_this_click = true;
                    } else {
                        apply_hierarchy_action(world, action);
                    }
                }
                HierarchyPaneAction::ToggleExpanded(entity) => {
                    if !hierarchy_expanded.insert(entity) {
                        hierarchy_expanded.remove(&entity);
                    }
                }
                HierarchyPaneAction::ListSurface => {}
            }
        }
        if let Some(action) = window_control_action {
            world.resource_scope::<EditorRetainedLayoutState, _>(|_world, mut layout_state| {
                match action {
                    WindowControlAction::ToggleCollapsed(window_id) => {
                        if previous_collapsed_windows.remove(&window_id) {
                            if let Some(height) =
                                previous_collapsed_window_heights.remove(&window_id)
                                && let Some(frame) = layout_state.windows.window_mut(window_id)
                            {
                                frame.rect.height = height.max(WINDOW_TITLE_BAR_HEIGHT + 2.0);
                            }
                        } else {
                            if let Some(frame) = layout_state.windows.window(window_id) {
                                previous_collapsed_window_heights.insert(
                                    window_id,
                                    frame.rect.height.max(WINDOW_TITLE_BAR_HEIGHT + 2.0),
                                );
                            }
                            previous_collapsed_windows.insert(window_id);
                        }
                    }
                    WindowControlAction::Close(window_id) => {
                        if let Some(frame) = layout_state.windows.window_mut(window_id) {
                            frame.visible = false;
                        }
                    }
                }
            });
        }
        if let Some(action) = project_action {
            apply_project_action(world, action);
        }
        if let Some(action) = content_action {
            let double_click = content_action_double_clicked(world, &action);
            apply_content_browser_action(
                world,
                action,
                input_shift_down,
                input_ctrl_down,
                double_click,
            );
        }
        if let Some(action) = inspector_action {
            match action {
                InspectorPaneAction::ToggleLightColorPicker(entity) => {
                    if inspector_light_color_picker.is_some_and(|picker| picker.entity == entity) {
                        inspector_light_color_picker = None;
                    } else {
                        let hue = light_picker_hue_from_world(world, entity, 0.0);
                        inspector_light_color_picker =
                            Some(InspectorLightColorPickerState { entity, hue });
                    }
                }
                InspectorPaneAction::OpenAddComponentMenu(entity) => {
                    let menu_position = clicked_rect
                        .map(|rect| Vec2::new(rect.x + 4.0, rect.y + rect.height + 2.0))
                        .or(pointer_position);
                    if let Some(position) = menu_position {
                        context_menu_state = Some(ContextMenuState {
                            position,
                            kind: ContextMenuKind::InspectorAddComponent(entity),
                            open_submenu: None,
                        });
                        context_menu_bounds = None;
                        context_menu_opened_this_click = true;
                    }
                }
                _ => apply_inspector_action(world, action),
            }
        }
        if let Some(action) = console_action {
            apply_console_action(world, action);
        }
        if let Some(action) = timeline_action {
            apply_timeline_action(world, action);
        }
        if let Some(action) = toolbar_action {
            match action {
                ToolbarPaneAction::OpenLayoutMenu => {
                    let menu_position = clicked_rect
                        .map(|rect| Vec2::new(rect.x + 4.0, rect.y + rect.height + 2.0))
                        .or(pointer_position);
                    if let Some(position) = menu_position {
                        context_menu_state = Some(ContextMenuState {
                            position,
                            kind: ContextMenuKind::ToolbarLayout,
                            open_submenu: None,
                        });
                        context_menu_bounds = None;
                        context_menu_opened_this_click = true;
                    }
                }
                _ => apply_toolbar_action(world, action),
            }
        }
        if let Some(action) = viewport_action {
            let viewport_menu_kind = match action {
                ViewportPaneAction::OpenCanvasMenu { tab_id, mode } => {
                    Some(ContextMenuKind::ViewportCanvas(ViewportMenuTarget {
                        tab_id,
                        mode,
                    }))
                }
                ViewportPaneAction::OpenRenderMenu { tab_id, mode } => {
                    Some(ContextMenuKind::ViewportRender(ViewportMenuTarget {
                        tab_id,
                        mode,
                    }))
                }
                ViewportPaneAction::OpenScriptingMenu { tab_id, mode } => {
                    Some(ContextMenuKind::ViewportScripting(ViewportMenuTarget {
                        tab_id,
                        mode,
                    }))
                }
                ViewportPaneAction::OpenGizmosMenu { tab_id, mode } => {
                    Some(ContextMenuKind::ViewportGizmos(ViewportMenuTarget {
                        tab_id,
                        mode,
                    }))
                }
                ViewportPaneAction::OpenFreecamMenu { tab_id, mode } => {
                    Some(ContextMenuKind::ViewportFreecam(ViewportMenuTarget {
                        tab_id,
                        mode,
                    }))
                }
                ViewportPaneAction::OpenOrbitMenu { tab_id, mode } => {
                    Some(ContextMenuKind::ViewportOrbit(ViewportMenuTarget {
                        tab_id,
                        mode,
                    }))
                }
                ViewportPaneAction::OpenAdvancedMenu { tab_id, mode } => {
                    Some(ContextMenuKind::ViewportAdvanced(ViewportMenuTarget {
                        tab_id,
                        mode,
                    }))
                }
                ViewportPaneAction::TogglePlayMode { .. }
                | ViewportPaneAction::PreviewMove { .. }
                | ViewportPaneAction::PreviewResize { .. } => None,
            };
            if let Some(kind) = viewport_menu_kind {
                let menu_position = clicked_rect
                    .map(|rect| Vec2::new(rect.x + 2.0, rect.y + rect.height + 2.0))
                    .or(pointer_position);
                if let Some(position) = menu_position {
                    context_menu_state = Some(ContextMenuState {
                        position,
                        kind,
                        open_submenu: None,
                    });
                    context_menu_bounds = None;
                    context_menu_opened_this_click = true;
                }
            } else {
                apply_viewport_action(world, action);
            }
        }
        if let Some(action) = dock_tab_action {
            world.resource_scope::<EditorRetainedDockingState, _>(|_world, mut docking_state| {
                if let Some(workspace) = docking_state.workspace_mut(action.workspace_id) {
                    workspace.docking.activate_tab(action.tab_id);
                }
            });
            world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
                activate_workspace_tab(&mut workspace, action.workspace_id, action.tab_id);
            });
        }
        if let Some(action) = history_action {
            apply_history_action(world, action);
        }
        if let Some(action) = audio_mixer_action {
            apply_audio_mixer_action(world, action);
        }
        if let Some(action) = profiler_action {
            apply_profiler_action(world, action);
        }
        if let Some(action) = material_editor_action {
            apply_material_editor_action(world, action);
        }
        if let Some(action) = context_menu_action {
            apply_context_menu_action(world, action);
            context_menu_state = None;
            context_menu_bounds = None;
        } else if menu_was_open && !context_menu_opened_this_click {
            let clicked_inside_menu = pointer_position
                .zip(context_menu_bounds)
                .is_some_and(|(pointer, bounds)| point_in_ui_rect(bounds, pointer));
            if !clicked_inside_menu {
                context_menu_state = None;
                context_menu_bounds = None;
            }
        }
    }

    if secondary_pointer_pressed {
        if let Some(hovered) = ui_interaction.hovered {
            let (
                clicked_entity,
                hierarchy_action,
                content_action,
                dock_tab_action,
                dock_tab_context,
                dock_tab_bar,
            ) = world
                .get_resource::<EditorRetainedPaneInteractionState>()
                .map(|state| {
                    (
                        state.hierarchy_click_targets.get(&hovered).copied(),
                        state.hierarchy_actions.get(&hovered).cloned(),
                        state.content_browser_actions.get(&hovered).cloned(),
                        state.dock_tab_actions.get(&hovered).copied(),
                        state.dock_tab_context_targets.get(&hovered).copied(),
                        state.dock_tab_bar_context_targets.get(&hovered).copied(),
                    )
                })
                .unwrap_or((None, None, None, None, None, None));

            if let Some(entity) = clicked_entity
                && let Some(mut selected) =
                    world.get_resource_mut::<InspectorSelectedEntityResource>()
            {
                selected.0 = Some(entity);
            }
            if let Some(action) = content_action.clone() {
                apply_content_browser_action(world, action, false, false, false);
            }
            if let Some(action) = dock_tab_action {
                world.resource_scope::<EditorRetainedDockingState, _>(
                    |_world, mut docking_state| {
                        if let Some(workspace) = docking_state.workspace_mut(action.workspace_id) {
                            workspace.docking.activate_tab(action.tab_id);
                        }
                    },
                );
                world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
                    activate_workspace_tab(&mut workspace, action.workspace_id, action.tab_id);
                });
            }

            if let Some(pointer) = pointer_position {
                context_menu_state = if let Some(entity) = clicked_entity {
                    Some(ContextMenuState {
                        position: pointer,
                        kind: ContextMenuKind::HierarchyEntity(entity),
                        open_submenu: None,
                    })
                } else if hierarchy_action.as_ref().is_some_and(|action| {
                    matches!(
                        action,
                        HierarchyPaneAction::ListSurface | HierarchyPaneAction::AddEntity
                    )
                }) {
                    Some(ContextMenuState {
                        position: pointer,
                        kind: ContextMenuKind::HierarchySurface,
                        open_submenu: None,
                    })
                } else if let Some(target) = dock_tab_context {
                    Some(ContextMenuState {
                        position: pointer,
                        kind: ContextMenuKind::DockTab {
                            workspace_id: target.workspace_id,
                            tab_id: target.tab_id,
                        },
                        open_submenu: None,
                    })
                } else if let Some(target) = dock_tab_bar {
                    Some(ContextMenuState {
                        position: pointer,
                        kind: ContextMenuKind::DockTabBar {
                            workspace_id: target.workspace_id,
                        },
                        open_submenu: None,
                    })
                } else {
                    match content_action {
                        Some(ContentBrowserPaneAction::SelectEntry { path, is_dir, .. }) => {
                            Some(ContextMenuState {
                                position: pointer,
                                kind: ContextMenuKind::ContentEntry { path, is_dir },
                                open_submenu: None,
                            })
                        }
                        Some(ContentBrowserPaneAction::GridSurface)
                        | Some(ContentBrowserPaneAction::SelectFolder(_)) => {
                            Some(ContextMenuState {
                                position: pointer,
                                kind: ContextMenuKind::ContentSurface,
                                open_submenu: None,
                            })
                        }
                        _ => None,
                    }
                };
                context_menu_bounds = None;
            }
        } else {
            context_menu_state = None;
            context_menu_bounds = None;
        }
    }

    if let Some(field) = focused_project_text_field {
        apply_project_text_field_input(
            world,
            field,
            &mut project_text_cursors,
            &input_just_pressed_keys,
            input_shift_down,
            input_ctrl_down,
        );
    }
    if let Some(field) = focused_content_browser_text_field {
        apply_content_browser_text_field_input(
            world,
            field,
            &mut content_browser_text_cursors,
            &input_just_pressed_keys,
            input_shift_down,
            input_ctrl_down,
        );
    }
    if let Some(field) = focused_inspector_text_field {
        apply_inspector_text_field_input(
            world,
            field,
            &mut inspector_text_cursors,
            &input_just_pressed_keys,
            input_shift_down,
            input_ctrl_down,
        );
    }
    if let Some(field) = focused_console_text_field {
        apply_console_text_field_input(
            world,
            field,
            &mut console_text_cursors,
            &input_just_pressed_keys,
            input_shift_down,
            input_ctrl_down,
        );
    }
    sync_inspector_light_color_picker(world, &mut inspector_light_color_picker);

    let graph_renderer = world
        .get_resource::<EditorRetainedGraphRenderer>()
        .map(|renderer| renderer.0.clone())
        .unwrap_or_default();

    let project_data = build_project_pane_data(world);
    let hierarchy_data = build_hierarchy_pane_data(world);
    let inspector_data = build_inspector_pane_data(world, inspector_light_color_picker);
    let timeline_data = build_timeline_pane_data(world);
    let content_browser_data = build_content_browser_pane_data(world);
    let content_browser_current_dir = content_current_directory(world);
    let content_drag_selection_count = content_browser_entry_drag
        .as_ref()
        .map(|drag| {
            world
                .get_resource::<AssetBrowserState>()
                .map(|state| {
                    if state.selected_paths.contains(&drag.primary_path) {
                        state.selected_paths.len().max(1)
                    } else {
                        1
                    }
                })
                .unwrap_or(1)
        })
        .unwrap_or(1);
    let console_data = build_console_pane_data(world);
    let toolbar_data = build_toolbar_pane_data(world);
    let history_data = build_history_pane_data(world);
    let audio_mixer_data = build_audio_mixer_pane_data(world);
    let profiler_data = build_profiler_pane_data(world);
    let material_editor_data = build_material_editor_pane_data(world);
    let hierarchy_context_info: HashMap<Entity, HierarchyContextInfo> = hierarchy_data
        .entries
        .iter()
        .map(|entry| {
            (
                entry.entity,
                HierarchyContextInfo {
                    can_unparent: world.get::<EntityParent>(entry.entity).is_some(),
                    can_set_active_camera: world
                        .get::<helmer_becs::BevyCamera>(entry.entity)
                        .is_some(),
                    scene_root: world
                        .get::<SceneChild>(entry.entity)
                        .map(|scene_child| scene_child.scene_root),
                    has_children: entry.has_children,
                },
            )
        })
        .collect();
    let (layout_active, allow_layout_move, allow_layout_resize, live_reflow) = world
        .get_resource::<EditorRetainedLayoutCatalog>()
        .map(|catalog| {
            (
                catalog.active.is_some(),
                catalog.allow_layout_move,
                catalog.allow_layout_resize,
                catalog.live_reflow,
            )
        })
        .unwrap_or((false, true, true, false));

    let mut retained_windows = Vec::new();
    let mut active_dock_drop_target: Option<DockDropTarget> = None;
    world.resource_scope::<EditorRetainedLayoutState, _>(|world, mut layout_state| {
        world.resource_scope::<EditorRetainedDockingState, _>(|_world, mut docking_state| {
            layout_state.sync_workspace_bounds(LayoutRect::new(
                0.0,
                0.0,
                viewport_size.x,
                viewport_size.y,
            ));
            for (index, pane_window) in pane_windows.iter().enumerate() {
                let window_id = pane_window_ui_id(pane_window);
                if !project_open && pane_window.id.eq_ignore_ascii_case("project") {
                    previous_collapsed_windows.remove(&window_id);
                    previous_collapsed_window_heights.remove(&window_id);
                }
                let bounds = layout_state.workspace_bounds;
                let fallback =
                    seeded_window_frame(pane_window, index, viewport_size, project_open, bounds);
                let window_exists = layout_state.windows.contains(window_id);
                let frame = layout_state.ensure_window(window_id, fallback);
                if !window_exists || last_project_open != Some(project_open) {
                    frame.visible = if project_open {
                        default_window_visible_for_open_project(pane_window)
                    } else {
                        pane_window.id.eq_ignore_ascii_case("project")
                    };
                }
                frame.locked = !project_open && pane_window.id.eq_ignore_ascii_case("project");
                if frame.locked {
                    let centered = centered_project_launcher_rect(viewport_size, bounds);
                    frame.rect = centered;
                }
                if previous_collapsed_windows.contains(&window_id) {
                    frame.rect.height = WINDOW_TITLE_BAR_HEIGHT + 2.0;
                }
                frame.rect = frame.constrained_rect().clamp_inside(bounds);

                let initial_tab = pane_window
                    .areas
                    .iter()
                    .find(|area| !area.tabs.is_empty())
                    .and_then(|area| {
                        area.tabs
                            .get(area.active.min(area.tabs.len().saturating_sub(1)))
                    })
                    .or_else(|| pane_window.areas.first().and_then(|area| area.tabs.first()))
                    .map(|tab| DockTab::new(UiId::from_raw(tab.id), tab.title.clone()))
                    .unwrap_or_else(|| DockTab::new(window_id.child("empty-tab"), "Pane"));

                let workspace = docking_state.ensure_workspace(window_id, initial_tab);
                sync_workspace_tabs(workspace, pane_window);
            }

            if project_open
                && let Some(drag) = dock_tab_drag
                && drag.active
                && let Some(pointer) = pointer_position
            {
                active_dock_drop_target =
                    resolve_dock_drop_target(&layout_state, &docking_state, pointer);
                if let Some(target) = active_dock_drop_target {
                    layout_state.windows.bring_to_front(target.window_id);
                } else if let Some(window_id) =
                    top_visible_window_at_pointer(&layout_state, pointer)
                {
                    layout_state.windows.bring_to_front(window_id);
                }
            } else {
                active_dock_drop_target = None;
            }

            if primary_pointer_released
                && let Some(drag) = dock_tab_drag
                && drag.active
            {
                let fallback_target = pointer_position
                    .and_then(|pointer| top_visible_window_at_pointer(&layout_state, pointer))
                    .map(|window_id| DockDropTarget {
                        window_id,
                        leaf_index: 0,
                        leaf_focus_tab: None,
                        zone: DockDropZone::Center,
                    });
                let drop_target = active_dock_drop_target.or(fallback_target);

                if let Some(target) = drop_target {
                    if drag.source_workspace_id != target.window_id
                        && let Some((moved_tab, updated_windows)) =
                            move_workspace_tab_between_windows(
                                _world,
                                drag.source_workspace_id,
                                target.window_id,
                                drag.tab_id,
                            )
                    {
                        pane_windows = updated_windows;
                        if let Some(source_workspace) =
                            docking_state.workspace_mut(drag.source_workspace_id)
                        {
                            source_workspace
                                .docking
                                .close_tab(UiId::from_raw(moved_tab.id));
                        }
                        if let Some(target_workspace) =
                            docking_state.workspace_mut(target.window_id)
                        {
                            apply_dock_tab_drop_to_workspace(
                                target_workspace,
                                target,
                                DockTab::new(UiId::from_raw(moved_tab.id), moved_tab.title),
                            );
                        }
                    } else if let Some(target_workspace) =
                        docking_state.workspace_mut(target.window_id)
                    {
                        let tab_count = dock_workspace_tab_count(target_workspace);
                        if tab_count > 1 {
                            let tab_title = find_pane_tab_title(&pane_windows, drag.tab_id)
                                .unwrap_or_else(|| "Pane".to_string());
                            target_workspace.docking.close_tab(drag.tab_id);
                            let mut adjusted_target = target;
                            if adjusted_target.leaf_focus_tab == Some(drag.tab_id) {
                                adjusted_target.leaf_focus_tab = None;
                            }
                            apply_dock_tab_drop_to_workspace(
                                target_workspace,
                                adjusted_target,
                                DockTab::new(drag.tab_id, tab_title),
                            );
                        } else {
                            target_workspace.docking.activate_tab(drag.tab_id);
                        }
                    }
                }
            }
            if primary_pointer_released {
                dock_tab_drag = None;
                active_dock_drop_target = None;
            }

            if pointer_pressed
                && let Some(pointer) = pointer_position
                && let Some(window_id) = top_visible_window_at_pointer(&layout_state, pointer)
            {
                layout_state.windows.bring_to_front(window_id);
            }

            if pointer_pressed
                && dragged_window.is_none()
                && let Some(pointer) = pointer_position
                && let Some((window_id, mode, edges, start_rect)) = pick_window_drag_target(
                    &layout_state,
                    pointer,
                    ui_interaction.hovered,
                    &previous_window_title_hits,
                    window_drag_pointer_down,
                )
            {
                dragged_window = Some(window_id);
                dragged_window_mode = mode;
                dragged_window_edges = edges;
                dragged_window_start_rect = Some(start_rect);
                dragged_pointer_start = pointer;
                layout_state.windows.bring_to_front(window_id);
            }

            if window_drag_pointer_down
                && let (Some(window_id), Some(start_rect), Some(pointer)) =
                    (dragged_window, dragged_window_start_rect, pointer_position)
            {
                let (min_size, max_size, locked) = layout_state
                    .windows
                    .window(window_id)
                    .map(|frame| (frame.min_size, frame.max_size, frame.locked))
                    .unwrap_or((Vec2::new(96.0, 72.0), None, false));
                let allow_drag_mode = match dragged_window_mode {
                    WindowDragMode::Move => !layout_active || allow_layout_move,
                    WindowDragMode::Resize => !layout_active || allow_layout_resize,
                    WindowDragMode::None => true,
                };

                if !locked && allow_drag_mode {
                    let delta = pointer - dragged_pointer_start;
                    let bounds = layout_state.workspace_bounds;
                    let new_rect = match dragged_window_mode {
                        WindowDragMode::Move => move_window_rect(start_rect, delta, bounds),
                        WindowDragMode::Resize => resize_window_rect(
                            start_rect,
                            delta,
                            dragged_window_edges,
                            min_size,
                            max_size,
                            bounds,
                        ),
                        WindowDragMode::None => start_rect,
                    };
                    let current_rect = layout_state
                        .windows
                        .window(window_id)
                        .map(|frame| frame.rect)
                        .unwrap_or(start_rect);

                    if dragged_window_mode == WindowDragMode::Resize && layout_active && live_reflow
                    {
                        layout_state.windows.resize_with_reflow(
                            window_id,
                            current_rect,
                            new_rect,
                            bounds,
                            8.0,
                        );
                    } else if let Some(frame) = layout_state.windows.window_mut(window_id) {
                        frame.rect = new_rect;
                    }
                    layout_state.windows.bring_to_front(window_id);
                }
            }

            if pointer_released || !window_drag_pointer_down {
                dragged_window = None;
                dragged_window_mode = WindowDragMode::None;
                dragged_window_edges = WindowResizeEdges::default();
                dragged_window_start_rect = None;
            }

            let ordered_windows = layout_state
                .windows
                .ordered_visible_windows()
                .into_iter()
                .map(|(window_id, frame)| (window_id, frame.rect))
                .collect::<Vec<_>>();

            for (window_id, window_rect) in ordered_windows {
                let Some(pane_window) = pane_windows
                    .iter()
                    .find(|pane_window| pane_window_ui_id(pane_window) == window_id)
                else {
                    continue;
                };
                let (leaves, split_handles) = if !project_open {
                    let active_tab = active_tab_for_window(pane_window);
                    let tabs = active_tab.clone().into_iter().collect::<Vec<_>>();
                    let active_tab_id = active_tab.as_ref().map(|tab| UiId::from_raw(tab.id));
                    (
                        vec![RetainedLeafView {
                            id: window_id.child("leaf").child(0_u64),
                            rect: UiRect {
                                x: window_rect.x,
                                y: window_rect.y + WINDOW_TITLE_BAR_HEIGHT,
                                width: window_rect.width,
                                height: (window_rect.height - WINDOW_TITLE_BAR_HEIGHT).max(1.0),
                            },
                            tabs,
                            active_tab_id,
                            active_tab,
                        }],
                        Vec::new(),
                    )
                } else {
                    let Some(workspace) = docking_state.workspace(window_id) else {
                        continue;
                    };

                    let content_rect = UiRect {
                        x: 0.0,
                        y: WINDOW_TITLE_BAR_HEIGHT,
                        width: window_rect.width,
                        height: (window_rect.height - WINDOW_TITLE_BAR_HEIGHT).max(1.0),
                    };
                    let dock_layout = workspace.docking.layout(content_rect);
                    let split_handles = workspace.docking.split_handles(content_rect, 7.0);

                    let tab_lookup = pane_tab_lookup(pane_window);
                    let leaves = dock_layout
                        .iter()
                        .enumerate()
                        .map(|(leaf_index, leaf)| RetainedLeafView {
                            id: window_id.child("leaf").child(leaf_index as u64),
                            rect: UiRect {
                                x: window_rect.x + leaf.rect.x,
                                y: window_rect.y + leaf.rect.y,
                                width: leaf.rect.width,
                                height: leaf.rect.height,
                            },
                            tabs: leaf
                                .tabs
                                .iter()
                                .filter_map(|tab_id| tab_lookup.get(tab_id).cloned())
                                .collect(),
                            active_tab_id: leaf.active.or_else(|| leaf.tabs.first().copied()),
                            active_tab: leaf_active_tab(leaf, &tab_lookup),
                        })
                        .collect();
                    (leaves, split_handles)
                };

                retained_windows.push(RetainedWindowView {
                    id: window_id,
                    title: pane_window.title.clone(),
                    rect: window_rect,
                    leaves,
                    split_handles,
                });
            }
        });
    });

    if retained_windows.is_empty() {
        let fallback_id = RETAINED_SHELL_ROOT.child("fallback-window");
        let width = (viewport_size.x - 48.0)
            .max(320.0)
            .min(viewport_size.x.max(1.0));
        let height = (viewport_size.y - 48.0)
            .max(220.0)
            .min(viewport_size.y.max(1.0));
        let rect = LayoutRect::new(
            ((viewport_size.x - width) * 0.5).max(0.0),
            ((viewport_size.y - height) * 0.5).max(0.0),
            width.max(1.0),
            height.max(1.0),
        );
        retained_windows.push(RetainedWindowView {
            id: fallback_id,
            title: "Editor Workspace".to_string(),
            rect,
            leaves: vec![RetainedLeafView {
                id: fallback_id.child("leaf").child(0_u64),
                rect: UiRect {
                    x: rect.x,
                    y: rect.y + WINDOW_TITLE_BAR_HEIGHT,
                    width: rect.width,
                    height: (rect.height - WINDOW_TITLE_BAR_HEIGHT).max(1.0),
                },
                tabs: Vec::new(),
                active_tab_id: None,
                active_tab: None,
            }],
            split_handles: Vec::new(),
        });
    }

    if let Some(window_snapshot) = world
        .get_resource::<EditorRetainedLayoutState>()
        .map(|layout| layout.windows.snapshot())
        && let Some(mut catalog) = world.get_resource_mut::<EditorRetainedLayoutCatalog>()
    {
        catalog.ensure_default(window_snapshot);
    }

    let active_graph_ids = collect_visible_visual_graph_ids(&retained_windows);
    world.resource_scope::<EditorRetainedGraphInteractionState, _>(
        |world, mut graph_interactions| {
            let pointer_pressed = pointer_down && !graph_interactions.pointer_down_previous;
            let pointer_released = !pointer_down && graph_interactions.pointer_down_previous;
            let graph_input = GraphInteractionInput {
                pointer_position,
                pointer_down,
                pointer_pressed,
                pointer_released,
                scroll_delta,
                hovered: ui_interaction.hovered,
                active: ui_interaction.active,
                clicked: ui_interaction.clicked,
            };

            world.resource_scope::<EditorRetainedGraphState, _>(|_world, mut graph_state| {
                for graph_id in &active_graph_ids {
                    let graph = graph_state.ensure_graph(*graph_id);
                    ensure_seed_graph(graph, *graph_id);

                    if let Some(frame) = graph_interactions.frames.get(graph_id).cloned() {
                        let controller =
                            graph_interactions.controllers.entry(*graph_id).or_default();
                        let _ = controller.update(&frame, graph, graph_input);
                    }
                }
                graph_state.active_graph = active_graph_ids.first().copied();
            });

            graph_interactions.pointer_down_previous = pointer_down;
        },
    );
    let graph_render_payloads = build_graph_render_payloads(world, &active_graph_ids);

    let mut next_hierarchy_click_targets = HashMap::new();
    let mut next_hierarchy_actions = HashMap::new();
    let mut next_hierarchy_drop_surface_hits = HashSet::new();
    let mut next_project_actions = HashMap::new();
    let mut next_project_text_fields = HashMap::new();
    let mut next_content_browser_actions = HashMap::new();
    let mut next_content_browser_text_fields = HashMap::new();
    let mut next_inspector_actions = HashMap::new();
    let mut next_inspector_drag_actions = HashMap::new();
    let mut next_inspector_text_fields = HashMap::new();
    let mut next_console_actions = HashMap::new();
    let mut next_console_text_fields = HashMap::new();
    let mut next_timeline_actions = HashMap::new();
    let mut next_toolbar_actions = HashMap::new();
    let mut next_viewport_actions = HashMap::new();
    let mut next_viewport_surfaces = HashMap::new();
    let mut next_dock_tab_actions = HashMap::new();
    let mut next_dock_tab_context_targets = HashMap::new();
    let mut next_dock_tab_bar_context_targets = HashMap::new();
    let mut next_dock_split_actions = HashMap::new();
    let mut next_history_actions = HashMap::new();
    let mut next_audio_mixer_actions = HashMap::new();
    let mut next_profiler_actions = HashMap::new();
    let mut next_material_editor_actions = HashMap::new();
    let mut next_context_menu_actions = HashMap::new();
    let mut next_context_menu_drag_actions = HashMap::new();
    let mut next_window_title_hits = HashMap::new();
    let mut next_window_controls = HashMap::new();
    let mut next_graph_frames = HashMap::new();
    let mut current_content_browser_viewports = Vec::new();
    let mut current_content_browser_scroll_regions = Vec::new();
    let mut current_console_viewports = Vec::new();
    let mut current_console_scroll_regions = Vec::new();
    let mut observed_hierarchy_scroll_max: Option<f32> = None;
    let mut observed_content_browser_scroll_max: Option<f32> = None;
    let mut observed_console_scroll_max: Option<f32> = None;
    let mut current_context_menu_bounds: Option<UiRect> = None;
    let toolbar_layout_rows = toolbar_layout_context_rows(world);
    let hierarchy_clipboard_has_entity = world
        .get_resource::<HierarchyClipboardState>()
        .and_then(|clipboard| clipboard.entity)
        .is_some_and(|entity| world.get_entity(entity).is_ok());
    let asset_clipboard_has_paths = world
        .get_resource::<EditorAssetClipboardState>()
        .is_some_and(|clipboard| !clipboard.paths.is_empty());
    let viewport_states_snapshot = world
        .get_resource::<EditorRetainedViewportStates>()
        .cloned()
        .unwrap_or_default();
    let gizmo_snap_snapshot = world
        .get_resource::<EditorRetainedGizmoSnapSettings>()
        .cloned()
        .unwrap_or_default();
    let hierarchy_label_lookup: HashMap<Entity, String> = hierarchy_data
        .entries
        .iter()
        .map(|entry| (entry.entity, entry.label.clone()))
        .collect();
    let viewport_pane_data: HashMap<
        (ViewportPaneMode, u64),
        crate::retained::panes::ViewportPaneData,
    > = pane_windows
        .iter()
        .flat_map(|window| window.areas.iter())
        .flat_map(|area| area.tabs.iter())
        .filter_map(|tab| match tab.kind {
            EditorPaneKind::Viewport => Some((ViewportPaneMode::Edit, tab.id)),
            EditorPaneKind::PlayViewport => Some((ViewportPaneMode::Play, tab.id)),
            _ => None,
        })
        .map(|(mode, tab_id)| {
            (
                (mode, tab_id),
                build_viewport_pane_data(world, mode, tab_id),
            )
        })
        .collect();

    let Some(mut ui_runtime_state) = world.get_resource_mut::<UiRuntimeState>() else {
        return;
    };
    let retained = ui_runtime_state.runtime_mut().retained_mut();
    retained.remove_subtree(RETAINED_SHELL_ROOT);

    let canvas_id = RETAINED_SHELL_ROOT.child("canvas");
    retained.upsert(RetainedUiNode::new(
        RETAINED_SHELL_ROOT,
        UiWidget::Container,
        absolute_style(
            0.0,
            0.0,
            viewport_size.x,
            viewport_size.y,
            UiVisualStyle::default(),
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        canvas_id,
        UiWidget::Container,
        absolute_style(
            0.0,
            0.0,
            viewport_size.x,
            viewport_size.y,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.02, 0.02, 0.03, 0.45)),
                border_color: None,
                border_width: 0.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.set_roots([RETAINED_SHELL_ROOT]);
    retained.set_children(RETAINED_SHELL_ROOT, [canvas_id]);

    let mut canvas_children = Vec::new();

    for window in &retained_windows {
        let window_id = window.id.child("shell");
        let title_bar_id = window_id.child("title-bar");
        let title_hit_id = window_id.child("title-hit");
        let title_label_id = window_id.child("title-label");
        let collapse_id = window_id.child("title-collapse");
        let close_id = window_id.child("title-close");
        let collapsed = previous_collapsed_windows.contains(&window.id);
        let is_project_window = window.id == UiId::from_str("pane-window-Project");
        let can_close = project_open || !is_project_window;
        let can_collapse = project_open || !is_project_window;
        let control_count = (can_close as u8 + can_collapse as u8) as f32;
        let controls_inner_width = if control_count > 0.0 {
            control_count * WINDOW_CONTROL_SIZE + (control_count - 1.0) * WINDOW_CONTROL_GAP
        } else {
            0.0
        };
        let controls_width = if control_count > 0.0 {
            controls_inner_width + WINDOW_CONTROL_MARGIN * 2.0
        } else {
            0.0
        };
        let title_hit_x = 0.0;
        let title_width = (window.rect.width - controls_width).max(1.0);

        retained.upsert(RetainedUiNode::new(
            window_id,
            UiWidget::Container,
            absolute_style(
                window.rect.x,
                window.rect.y,
                window.rect.width,
                window.rect.height,
                UiVisualStyle {
                    background: if collapsed {
                        None
                    } else {
                        Some(UiColor::rgba(0.12, 0.13, 0.16, 0.88))
                    },
                    border_color: if collapsed {
                        None
                    } else {
                        Some(UiColor::rgba(0.30, 0.34, 0.40, 0.92))
                    },
                    border_width: if collapsed { 0.0 } else { 1.0 },
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));

        retained.upsert(RetainedUiNode::new(
            title_bar_id,
            UiWidget::Container,
            absolute_style(
                0.0,
                0.0,
                window.rect.width,
                WINDOW_TITLE_BAR_HEIGHT.max(1.0),
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.18, 0.21, 0.27, 0.94)),
                    border_color: None,
                    border_width: 0.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            title_hit_id,
            UiWidget::HitBox,
            absolute_style(
                title_hit_x,
                0.0,
                title_width,
                WINDOW_TITLE_BAR_HEIGHT.max(1.0),
                UiVisualStyle::default(),
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            title_label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(if project_open {
                    String::new()
                } else {
                    window.title.clone()
                }),
                style: UiTextStyle {
                    color: UiColor::rgba(0.92, 0.95, 1.0, 1.0),
                    font_size: 10.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                8.0,
                0.0,
                (title_width - 12.0).max(1.0),
                WINDOW_TITLE_BAR_HEIGHT.max(1.0),
                UiVisualStyle::default(),
            ),
        ));
        next_window_title_hits.insert(title_hit_id, window.id);

        let mut window_children = vec![title_bar_id];
        let mut title_children = vec![title_hit_id, title_label_id];

        let mut control_x =
            (window.rect.width - WINDOW_CONTROL_MARGIN - controls_inner_width).max(0.0);
        let control_y = ((WINDOW_TITLE_BAR_HEIGHT - WINDOW_CONTROL_SIZE) * 0.5 - 1.0).max(0.0);
        if can_collapse {
            let collapse_bg_id = collapse_id.child("bg");
            let collapse_label_id = collapse_id.child("label");
            retained.upsert(RetainedUiNode::new(
                collapse_bg_id,
                UiWidget::Container,
                absolute_style(
                    control_x,
                    control_y,
                    WINDOW_CONTROL_SIZE,
                    WINDOW_CONTROL_SIZE,
                    UiVisualStyle {
                        background: None,
                        border_color: None,
                        border_width: 0.0,
                        corner_radius: 0.0,
                        clip: true,
                    },
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                collapse_label_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from(if collapsed { "+" } else { "-" }),
                    style: UiTextStyle {
                        color: UiColor::rgba(0.93, 0.96, 1.0, 1.0),
                        font_size: 10.0,
                        align_h: UiTextAlign::Center,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                absolute_style(
                    control_x,
                    control_y,
                    WINDOW_CONTROL_SIZE,
                    WINDOW_CONTROL_SIZE,
                    UiVisualStyle::default(),
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                collapse_id,
                UiWidget::HitBox,
                absolute_style(
                    control_x,
                    control_y,
                    WINDOW_CONTROL_SIZE,
                    WINDOW_CONTROL_SIZE,
                    UiVisualStyle::default(),
                ),
            ));
            next_window_controls
                .insert(collapse_id, WindowControlAction::ToggleCollapsed(window.id));
            title_children.push(collapse_bg_id);
            title_children.push(collapse_label_id);
            title_children.push(collapse_id);
            control_x += WINDOW_CONTROL_SIZE + WINDOW_CONTROL_GAP;
        }
        if can_close {
            let close_bg_id = close_id.child("bg");
            let close_label_id = close_id.child("label");
            retained.upsert(RetainedUiNode::new(
                close_bg_id,
                UiWidget::Container,
                absolute_style(
                    control_x,
                    control_y,
                    WINDOW_CONTROL_SIZE,
                    WINDOW_CONTROL_SIZE,
                    UiVisualStyle {
                        background: None,
                        border_color: None,
                        border_width: 0.0,
                        corner_radius: 0.0,
                        clip: true,
                    },
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                close_label_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from("x"),
                    style: UiTextStyle {
                        color: UiColor::rgba(0.88, 0.90, 0.95, 1.0),
                        font_size: 10.0,
                        align_h: UiTextAlign::Center,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                absolute_style(
                    control_x,
                    control_y,
                    WINDOW_CONTROL_SIZE,
                    WINDOW_CONTROL_SIZE,
                    UiVisualStyle::default(),
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                close_id,
                UiWidget::HitBox,
                absolute_style(
                    control_x,
                    control_y,
                    WINDOW_CONTROL_SIZE,
                    WINDOW_CONTROL_SIZE,
                    UiVisualStyle::default(),
                ),
            ));
            next_window_controls.insert(close_id, WindowControlAction::Close(window.id));
            title_children.push(close_bg_id);
            title_children.push(close_label_id);
            title_children.push(close_id);
        }
        retained.set_children(title_bar_id, title_children);

        if collapsed {
            retained.set_children(window_id, window_children);
            canvas_children.push(window_id);
            continue;
        }

        for (leaf_index, leaf) in window.leaves.iter().enumerate() {
            let leaf_id = leaf.id;
            let local_leaf_x = (leaf.rect.x - window.rect.x).max(0.0);
            let local_leaf_y = (leaf.rect.y - window.rect.y).max(0.0);
            let draw_tab_strip = project_open && !leaf.tabs.is_empty();
            let tab_strip_height = if draw_tab_strip {
                WINDOW_TAB_STRIP_HEIGHT.min((leaf.rect.height - 1.0).max(1.0))
            } else {
                0.0
            };
            let content_height = (leaf.rect.height - tab_strip_height).max(1.0);
            let content_host_id = leaf_id.child("content-host");

            retained.upsert(RetainedUiNode::new(
                leaf_id,
                UiWidget::Container,
                absolute_style(
                    local_leaf_x,
                    local_leaf_y,
                    leaf.rect.width,
                    leaf.rect.height,
                    UiVisualStyle {
                        background: Some(UiColor::rgba(0.12, 0.14, 0.18, 0.90)),
                        border_color: Some(UiColor::rgba(0.22, 0.26, 0.32, 0.86)),
                        border_width: 1.0,
                        corner_radius: 0.0,
                        clip: true,
                    },
                ),
            ));

            let mut leaf_children = Vec::new();
            let mut drop_overlay_id = None;
            if dock_tab_drag.is_some_and(|drag| drag.active)
                && let Some(target) = active_dock_drop_target
                && target.window_id == window.id
                && target.leaf_index == leaf_index
            {
                let drop_zone_rect = dock_drop_zone_rect(
                    UiRect {
                        x: 0.0,
                        y: tab_strip_height,
                        width: leaf.rect.width.max(1.0),
                        height: content_height.max(1.0),
                    },
                    target.zone,
                );
                let overlay_id = leaf_id.child("drop-overlay");
                retained.upsert(RetainedUiNode::new(
                    overlay_id,
                    UiWidget::Container,
                    absolute_style(
                        drop_zone_rect.x,
                        drop_zone_rect.y,
                        drop_zone_rect.width,
                        drop_zone_rect.height,
                        UiVisualStyle {
                            background: Some(UiColor::rgba(0.32, 0.48, 0.76, 0.34)),
                            border_color: Some(UiColor::rgba(0.62, 0.80, 0.98, 0.88)),
                            border_width: 1.0,
                            corner_radius: 0.0,
                            clip: false,
                        },
                    ),
                ));
                drop_overlay_id = Some(overlay_id);
            }
            if draw_tab_strip {
                let tab_bar_id = leaf_id.child("tab-bar");
                let tab_bar_hit_id = tab_bar_id.child("context-hit");
                retained.upsert(RetainedUiNode::new(
                    tab_bar_id,
                    UiWidget::Container,
                    absolute_style(
                        0.0,
                        0.0,
                        leaf.rect.width,
                        tab_strip_height,
                        UiVisualStyle {
                            background: Some(UiColor::rgba(0.16, 0.20, 0.26, 0.96)),
                            border_color: Some(UiColor::rgba(0.24, 0.30, 0.38, 0.92)),
                            border_width: 1.0,
                            corner_radius: 0.0,
                            clip: true,
                        },
                    ),
                ));
                retained.upsert(RetainedUiNode::new(
                    tab_bar_hit_id,
                    UiWidget::HitBox,
                    absolute_style(
                        0.0,
                        0.0,
                        leaf.rect.width.max(1.0),
                        tab_strip_height.max(1.0),
                        UiVisualStyle::default(),
                    ),
                ));
                next_dock_tab_bar_context_targets.insert(
                    tab_bar_hit_id,
                    DockTabBarContextTarget {
                        workspace_id: window.id,
                    },
                );

                let mut tab_children = vec![tab_bar_hit_id];
                let mut tab_x = 6.0;
                for tab in &leaf.tabs {
                    let tab_id = UiId::from_raw(tab.id);
                    let selected = Some(tab_id) == leaf.active_tab_id;
                    let button_id = tab_bar_id.child("tab").child(tab.id);
                    let title = tab.title.clone();
                    let estimated_width = (title.chars().count() as f32 * 7.0) + 26.0;
                    let tab_width =
                        estimated_width.clamp(WINDOW_TAB_MIN_WIDTH, WINDOW_TAB_MAX_WIDTH);
                    if tab_x + tab_width > (leaf.rect.width - 6.0).max(10.0) {
                        break;
                    }

                    retained.upsert(RetainedUiNode::new(
                        button_id,
                        UiWidget::Button(UiButton {
                            text: UiTextValue::from(title),
                            variant: if selected {
                                UiButtonVariant::Secondary
                            } else {
                                UiButtonVariant::Ghost
                            },
                            enabled: true,
                            style: UiTextStyle {
                                color: UiColor::rgba(0.93, 0.96, 1.0, 1.0),
                                font_size: 11.0,
                                align_h: UiTextAlign::Center,
                                align_v: UiTextAlign::Center,
                                wrap: false,
                            },
                        }),
                        absolute_style(
                            tab_x,
                            2.0,
                            tab_width,
                            (tab_strip_height - 4.0).max(1.0),
                            UiVisualStyle {
                                background: Some(if selected {
                                    UiColor::rgba(0.24, 0.32, 0.44, 0.96)
                                } else {
                                    UiColor::rgba(0.18, 0.22, 0.30, 0.90)
                                }),
                                border_color: Some(if selected {
                                    UiColor::rgba(0.56, 0.72, 0.95, 0.96)
                                } else {
                                    UiColor::rgba(0.34, 0.42, 0.54, 0.92)
                                }),
                                border_width: 1.0,
                                corner_radius: 0.0,
                                clip: false,
                            },
                        ),
                    ));
                    next_dock_tab_actions.insert(
                        button_id,
                        DockTabClickAction {
                            workspace_id: window.id,
                            tab_id,
                        },
                    );
                    next_dock_tab_context_targets.insert(
                        button_id,
                        DockTabContextTarget {
                            workspace_id: window.id,
                            tab_id,
                        },
                    );
                    tab_children.push(button_id);
                    tab_x += tab_width + 4.0;
                }
                retained.set_children(tab_bar_id, tab_children);
                leaf_children.push(tab_bar_id);
            }

            retained.upsert(RetainedUiNode::new(
                content_host_id,
                UiWidget::Container,
                absolute_style(
                    0.0,
                    tab_strip_height,
                    leaf.rect.width,
                    content_height,
                    UiVisualStyle {
                        background: None,
                        border_color: None,
                        border_width: 0.0,
                        corner_radius: 0.0,
                        clip: true,
                    },
                ),
            ));

            let mut content_children = Vec::new();
            if let Some(tab) = leaf.active_tab.as_ref() {
                match tab.kind {
                    EditorPaneKind::Toolbar => {
                        let pane_id = content_host_id.child("toolbar-pane");
                        let frame = build_toolbar_pane(
                            retained,
                            pane_id,
                            UiRect {
                                x: 0.0,
                                y: 0.0,
                                width: leaf.rect.width.max(1.0),
                                height: content_height.max(1.0),
                            },
                            &toolbar_data,
                        );
                        next_toolbar_actions.extend(frame.actions);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::Viewport => {
                        let pane_id = content_host_id.child("viewport-pane");
                        let viewport_data = viewport_pane_data
                            .get(&(ViewportPaneMode::Edit, tab.id))
                            .cloned()
                            .unwrap_or_else(|| crate::retained::panes::ViewportPaneData {
                                tab_id: tab.id,
                                ..Default::default()
                            });
                        let frame = build_viewport_pane(
                            retained,
                            pane_id,
                            UiRect {
                                x: 0.0,
                                y: 0.0,
                                width: leaf.rect.width.max(1.0),
                                height: content_height.max(1.0),
                            },
                            ViewportPaneMode::Edit,
                            &viewport_data,
                        );
                        next_viewport_actions.extend(frame.actions);
                        next_viewport_surfaces.insert(
                            ViewportPaneSurfaceKey {
                                tab_id: tab.id,
                                mode: ViewportPaneMode::Edit,
                            },
                            UiRect {
                                x: leaf.rect.x + frame.scene_rect.x,
                                y: leaf.rect.y + tab_strip_height + frame.scene_rect.y,
                                width: frame.scene_rect.width.max(1.0),
                                height: frame.scene_rect.height.max(1.0),
                            },
                        );
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::PlayViewport => {
                        let pane_id = content_host_id.child("play-viewport-pane");
                        let viewport_data = viewport_pane_data
                            .get(&(ViewportPaneMode::Play, tab.id))
                            .cloned()
                            .unwrap_or_else(|| crate::retained::panes::ViewportPaneData {
                                tab_id: tab.id,
                                ..Default::default()
                            });
                        let frame = build_viewport_pane(
                            retained,
                            pane_id,
                            UiRect {
                                x: 0.0,
                                y: 0.0,
                                width: leaf.rect.width.max(1.0),
                                height: content_height.max(1.0),
                            },
                            ViewportPaneMode::Play,
                            &viewport_data,
                        );
                        next_viewport_actions.extend(frame.actions);
                        next_viewport_surfaces.insert(
                            ViewportPaneSurfaceKey {
                                tab_id: tab.id,
                                mode: ViewportPaneMode::Play,
                            },
                            UiRect {
                                x: leaf.rect.x + frame.scene_rect.x,
                                y: leaf.rect.y + tab_strip_height + frame.scene_rect.y,
                                width: frame.scene_rect.width.max(1.0),
                                height: frame.scene_rect.height.max(1.0),
                            },
                        );
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::Project => {
                        let pane_id = content_host_id.child("project-pane");
                        let frame = build_project_pane(
                            retained,
                            pane_id,
                            UiRect {
                                x: 0.0,
                                y: 0.0,
                                width: leaf.rect.width.max(1.0),
                                height: content_height.max(1.0),
                            },
                            &project_data,
                        );
                        next_project_actions.extend(frame.actions);
                        next_project_text_fields.extend(frame.text_fields);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::Hierarchy => {
                        let pane_id = content_host_id.child("hierarchy-pane");
                        let frame = build_hierarchy_pane(
                            retained,
                            pane_id,
                            UiRect {
                                x: 0.0,
                                y: 0.0,
                                width: leaf.rect.width.max(1.0),
                                height: content_height.max(1.0),
                            },
                            &hierarchy_data,
                        );
                        observed_hierarchy_scroll_max = Some(
                            observed_hierarchy_scroll_max
                                .unwrap_or(0.0)
                                .max(frame.scroll_max.max(0.0)),
                        );
                        next_hierarchy_click_targets.extend(frame.click_targets);
                        next_hierarchy_actions.extend(frame.actions);
                        next_hierarchy_drop_surface_hits.extend(frame.drop_surface_hits);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::Inspector => {
                        let pane_id = content_host_id.child("inspector-pane");
                        let frame = build_inspector_pane(
                            retained,
                            pane_id,
                            UiRect {
                                x: 0.0,
                                y: 0.0,
                                width: leaf.rect.width.max(1.0),
                                height: content_height.max(1.0),
                            },
                            &inspector_data,
                        );
                        next_inspector_actions.extend(frame.actions);
                        next_inspector_drag_actions.extend(frame.drag_actions);
                        next_inspector_text_fields.extend(frame.text_fields);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::History => {
                        let pane_id = content_host_id.child("history-pane");
                        let frame = build_history_pane(retained, pane_id, &history_data);
                        next_history_actions.extend(frame.actions);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::Timeline => {
                        let pane_id = content_host_id.child("timeline-pane");
                        let frame = build_timeline_pane(retained, pane_id, &timeline_data);
                        next_timeline_actions.extend(frame.actions);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::ContentBrowser => {
                        let pane_id = content_host_id.child("content-browser-pane");
                        current_content_browser_viewports.push(UiRect {
                            x: window.rect.x,
                            y: window.rect.y + WINDOW_TITLE_BAR_HEIGHT,
                            width: window.rect.width.max(1.0),
                            height: (window.rect.height - WINDOW_TITLE_BAR_HEIGHT).max(1.0),
                        });
                        let frame = build_content_browser_pane(
                            retained,
                            pane_id,
                            UiRect {
                                x: 0.0,
                                y: 0.0,
                                width: leaf.rect.width.max(1.0),
                                height: content_height.max(1.0),
                            },
                            &content_browser_data,
                        );
                        observed_content_browser_scroll_max = Some(
                            observed_content_browser_scroll_max
                                .unwrap_or(0.0)
                                .max(frame.grid_scroll_max.max(0.0)),
                        );
                        if let Some(region) = frame.grid_scroll_region {
                            current_content_browser_scroll_regions.push(UiRect {
                                x: leaf.rect.x + region.x,
                                y: leaf.rect.y + tab_strip_height + region.y,
                                width: region.width,
                                height: region.height,
                            });
                        }
                        next_content_browser_actions.extend(frame.actions);
                        next_content_browser_text_fields.extend(frame.text_fields);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::Console => {
                        let pane_id = content_host_id.child("console-pane");
                        current_console_viewports.push(UiRect {
                            x: window.rect.x,
                            y: window.rect.y + WINDOW_TITLE_BAR_HEIGHT,
                            width: window.rect.width.max(1.0),
                            height: (window.rect.height - WINDOW_TITLE_BAR_HEIGHT).max(1.0),
                        });
                        let frame = build_console_pane(
                            retained,
                            pane_id,
                            UiRect {
                                x: 0.0,
                                y: 0.0,
                                width: leaf.rect.width.max(1.0),
                                height: content_height.max(1.0),
                            },
                            &console_data,
                        );
                        observed_console_scroll_max = Some(
                            observed_console_scroll_max
                                .unwrap_or(0.0)
                                .max(frame.log_scroll_max.max(0.0)),
                        );
                        if let Some(region) = frame.log_scroll_region {
                            current_console_scroll_regions.push(UiRect {
                                x: leaf.rect.x + region.x,
                                y: leaf.rect.y + tab_strip_height + region.y,
                                width: region.width,
                                height: region.height,
                            });
                        }
                        next_console_actions.extend(frame.actions);
                        next_console_text_fields.extend(frame.text_fields);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::AudioMixer => {
                        let pane_id = content_host_id.child("audio-mixer-pane");
                        let frame = build_audio_mixer_pane(retained, pane_id, &audio_mixer_data);
                        next_audio_mixer_actions.extend(frame.actions);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::Profiler => {
                        let pane_id = content_host_id.child("profiler-pane");
                        let frame = build_profiler_pane(retained, pane_id, &profiler_data);
                        next_profiler_actions.extend(frame.actions);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::MaterialEditor => {
                        let pane_id = content_host_id.child("material-editor-pane");
                        let frame =
                            build_material_editor_pane(retained, pane_id, &material_editor_data);
                        next_material_editor_actions.extend(frame.actions);
                        content_children.push(pane_id);
                    }
                    EditorPaneKind::VisualScriptEditor => {
                        let graph_id = graph_id_for_tab(tab.id, tab.asset_path.as_deref());
                        let (graph, preview) = graph_render_payloads
                            .get(&graph_id)
                            .map(|payload| (payload.graph.clone(), payload.preview))
                            .unwrap_or_else(|| (GraphState::default(), None));

                        let graph_root = RETAINED_SHELL_ROOT
                            .child("graph")
                            .child(graph_id)
                            .child(tab.id);
                        let graph_rect = inset_rect(
                            UiRect {
                                x: leaf.rect.x,
                                y: leaf.rect.y + tab_strip_height,
                                width: leaf.rect.width,
                                height: content_height,
                            },
                            4.0,
                        );
                        let frame = build_visual_scripting_pane(
                            retained,
                            graph_root,
                            graph_rect,
                            &graph_renderer,
                            &graph,
                            preview,
                        );
                        next_graph_frames.insert(graph_id, frame);
                        canvas_children.push(graph_root);

                        let subtitle_id = content_host_id.child("visual-subtitle");
                        retained.upsert(RetainedUiNode::new(
                            subtitle_id,
                            UiWidget::Label(UiLabel {
                                text: UiTextValue::from("Visual Scripting (Retained Graph)"),
                                style: UiTextStyle {
                                    color: UiColor::rgba(0.86, 0.90, 0.98, 1.0),
                                    font_size: 11.0,
                                    align_h: UiTextAlign::Start,
                                    align_v: UiTextAlign::Center,
                                    wrap: false,
                                },
                            }),
                            absolute_style(
                                10.0,
                                6.0,
                                (leaf.rect.width - 20.0).max(1.0),
                                16.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        content_children.push(subtitle_id);
                    }
                }
            } else {
                let pane_id = content_host_id.child("empty-pane");
                build_placeholder_pane(retained, pane_id, "No active pane tab");
                content_children.push(pane_id);
            }

            retained.set_children(content_host_id, content_children);
            leaf_children.push(content_host_id);
            if let Some(overlay_id) = drop_overlay_id {
                leaf_children.push(overlay_id);
            }
            retained.set_children(leaf_id, leaf_children);
            window_children.push(leaf_id);
        }

        for (split_index, split) in window.split_handles.iter().enumerate() {
            let split_id = window_id.child("split").child(split_index as u64);
            let bar_id = split_id.child("bar");
            let hit_id = split_id.child("hit");
            let hovered = ui_interaction.hovered == Some(hit_id);
            let active = ui_interaction.active == Some(hit_id);
            let bar_color = if active {
                UiColor::rgba(0.68, 0.82, 0.98, 0.96)
            } else if hovered {
                UiColor::rgba(0.56, 0.74, 0.95, 0.92)
            } else {
                UiColor::rgba(0.34, 0.44, 0.58, 0.78)
            };

            retained.upsert(RetainedUiNode::new(
                bar_id,
                UiWidget::Container,
                absolute_style(
                    split.handle_rect.x,
                    split.handle_rect.y,
                    split.handle_rect.width.max(1.0),
                    split.handle_rect.height.max(1.0),
                    UiVisualStyle {
                        background: Some(bar_color),
                        border_color: Some(UiColor::rgba(0.16, 0.22, 0.30, 0.92)),
                        border_width: 1.0,
                        corner_radius: 0.0,
                        clip: false,
                    },
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                hit_id,
                UiWidget::HitBox,
                absolute_style(
                    split.handle_rect.x,
                    split.handle_rect.y,
                    split.handle_rect.width.max(1.0),
                    split.handle_rect.height.max(1.0),
                    UiVisualStyle::default(),
                ),
            ));
            next_dock_split_actions.insert(
                hit_id,
                DockSplitHandleAction {
                    workspace_id: window.id,
                    path: split.path.clone(),
                    axis: split.axis,
                    parent_rect: UiRect {
                        x: window.rect.x + split.parent_rect.x,
                        y: window.rect.y + split.parent_rect.y,
                        width: split.parent_rect.width,
                        height: split.parent_rect.height,
                    },
                },
            );
            window_children.push(bar_id);
            window_children.push(hit_id);
        }

        retained.set_children(window_id, window_children);
        canvas_children.push(window_id);
    }

    if let Some(drag) = dock_tab_drag
        && drag.active
        && let Some(pointer) = pointer_position
    {
        let preview_title =
            find_pane_tab_title(&pane_windows, drag.tab_id).unwrap_or_else(|| "Pane".to_string());
        let preview_width = (preview_title.chars().count() as f32 * 7.0 + 34.0).clamp(88.0, 240.0);
        let preview_height = 26.0;
        let shadow_id = RETAINED_SHELL_ROOT.child("tab-drag-shadow");
        let preview_id = RETAINED_SHELL_ROOT.child("tab-drag-preview");
        let label_id = preview_id.child("label");

        retained.upsert(RetainedUiNode::new(
            shadow_id,
            UiWidget::Container,
            absolute_style(
                pointer.x + 16.0,
                pointer.y + 18.0,
                preview_width,
                preview_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.01, 0.01, 0.02, 0.42)),
                    border_color: None,
                    border_width: 0.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            preview_id,
            UiWidget::Container,
            absolute_style(
                pointer.x + 12.0,
                pointer.y + 12.0,
                preview_width,
                preview_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.20, 0.28, 0.40, 0.96)),
                    border_color: Some(UiColor::rgba(0.58, 0.74, 0.96, 0.96)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(preview_title),
                style: UiTextStyle {
                    color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
                    font_size: 11.0,
                    align_h: UiTextAlign::Center,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                6.0,
                0.0,
                (preview_width - 12.0).max(1.0),
                preview_height,
                UiVisualStyle::default(),
            ),
        ));
        retained.set_children(preview_id, [label_id]);
        canvas_children.push(shadow_id);
        canvas_children.push(preview_id);
    }

    if let Some(drag) = hierarchy_drag_state
        && drag.active
        && let Some(pointer) = pointer_position
    {
        let preview_title = hierarchy_label_lookup
            .get(&drag.entity)
            .cloned()
            .unwrap_or_else(|| format!("Entity {}", drag.entity.to_bits()));
        let preview_width = (preview_title.chars().count() as f32 * 7.0 + 30.0).clamp(96.0, 260.0);
        let preview_height = 24.0;
        let shadow_id = RETAINED_SHELL_ROOT.child("hierarchy-drag-shadow");
        let preview_id = RETAINED_SHELL_ROOT.child("hierarchy-drag-preview");
        let label_id = preview_id.child("label");

        retained.upsert(RetainedUiNode::new(
            shadow_id,
            UiWidget::Container,
            absolute_style(
                pointer.x + 14.0,
                pointer.y + 16.0,
                preview_width,
                preview_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.01, 0.01, 0.02, 0.42)),
                    border_color: None,
                    border_width: 0.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            preview_id,
            UiWidget::Container,
            absolute_style(
                pointer.x + 10.0,
                pointer.y + 10.0,
                preview_width,
                preview_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.24, 0.32, 0.46, 0.96)),
                    border_color: Some(UiColor::rgba(0.60, 0.76, 0.97, 0.95)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(preview_title),
                style: UiTextStyle {
                    color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
                    font_size: 11.0,
                    align_h: UiTextAlign::Center,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                6.0,
                0.0,
                (preview_width - 12.0).max(1.0),
                preview_height,
                UiVisualStyle::default(),
            ),
        ));
        retained.set_children(preview_id, [label_id]);
        canvas_children.push(shadow_id);
        canvas_children.push(preview_id);
    }
    if let Some(drag) = content_browser_entry_drag.as_ref()
        && drag.active
        && pointer_down
        && let Some(pointer) = pointer_position
    {
        let mut preview_title = drag
            .primary_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| drag.primary_path.to_string_lossy().to_string());
        let selection_count = content_drag_selection_count.max(1);
        if selection_count > 1 {
            preview_title = format!("{preview_title} (+{})", selection_count.saturating_sub(1));
        }
        let preview_width = (preview_title.chars().count() as f32 * 7.0 + 30.0).clamp(96.0, 280.0);
        let preview_height = 24.0;
        let shadow_id = RETAINED_SHELL_ROOT.child("content-drag-shadow");
        let preview_id = RETAINED_SHELL_ROOT.child("content-drag-preview");
        let label_id = preview_id.child("label");

        retained.upsert(RetainedUiNode::new(
            shadow_id,
            UiWidget::Container,
            absolute_style(
                pointer.x + 14.0,
                pointer.y + 16.0,
                preview_width,
                preview_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.01, 0.01, 0.02, 0.42)),
                    border_color: None,
                    border_width: 0.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            preview_id,
            UiWidget::Container,
            absolute_style(
                pointer.x + 10.0,
                pointer.y + 10.0,
                preview_width,
                preview_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.24, 0.32, 0.46, 0.96)),
                    border_color: Some(UiColor::rgba(0.60, 0.76, 0.97, 0.95)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(preview_title),
                style: UiTextStyle {
                    color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
                    font_size: 11.0,
                    align_h: UiTextAlign::Center,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                6.0,
                0.0,
                (preview_width - 12.0).max(1.0),
                preview_height,
                UiVisualStyle::default(),
            ),
        ));
        retained.set_children(preview_id, [label_id]);
        canvas_children.push(shadow_id);
        canvas_children.push(preview_id);
    }

    if let Some(menu_state) = context_menu_state.as_ref() {
        let entries: Vec<ContextMenuEntryRow> = match &menu_state.kind {
            ContextMenuKind::HierarchyEntity(entity) => {
                let mut entries = Vec::new();
                let context = hierarchy_context_info
                    .get(entity)
                    .copied()
                    .unwrap_or_default();
                entries.push(ContextMenuEntryRow::Action(
                    "Copy".to_string(),
                    ContextMenuAction::HierarchyCopy(*entity),
                ));
                entries.push(ContextMenuEntryRow::Action(
                    "Cut".to_string(),
                    ContextMenuAction::HierarchyCut(*entity),
                ));
                if hierarchy_clipboard_has_entity {
                    entries.push(ContextMenuEntryRow::Action(
                        "Paste".to_string(),
                        ContextMenuAction::HierarchyPaste(Some(*entity)),
                    ));
                }
                entries.push(ContextMenuEntryRow::Action(
                    "Duplicate".to_string(),
                    ContextMenuAction::HierarchyDuplicate(*entity),
                ));
                entries.push(ContextMenuEntryRow::Action(
                    "Rename".to_string(),
                    ContextMenuAction::HierarchyRename(*entity),
                ));
                entries.push(ContextMenuEntryRow::Separator);
                if context.can_unparent {
                    entries.push(ContextMenuEntryRow::Action(
                        "Unparent Entity".to_string(),
                        ContextMenuAction::HierarchyUnparent(*entity),
                    ));
                }
                entries.push(ContextMenuEntryRow::Action(
                    "Focus".to_string(),
                    ContextMenuAction::HierarchyFocus(*entity),
                ));
                if context.can_set_active_camera {
                    entries.push(ContextMenuEntryRow::Action(
                        "Set Game Camera".to_string(),
                        ContextMenuAction::HierarchySetActiveCamera(*entity),
                    ));
                }
                if let Some(scene_root) = context.scene_root {
                    entries.push(ContextMenuEntryRow::Action(
                        "Select Scene Root".to_string(),
                        ContextMenuAction::HierarchySelectSceneRoot(scene_root),
                    ));
                }
                entries.push(ContextMenuEntryRow::Separator);
                entries.push(ContextMenuEntryRow::Action(
                    "Delete Entity".to_string(),
                    ContextMenuAction::HierarchyDeleteEntity(*entity),
                ));
                if context.has_children {
                    entries.push(ContextMenuEntryRow::Action(
                        "Delete Hierarchy".to_string(),
                        ContextMenuAction::HierarchyDeleteHierarchy(*entity),
                    ));
                }
                entries
            }
            ContextMenuKind::HierarchySurface => {
                let mut entries = Vec::new();
                if hierarchy_clipboard_has_entity {
                    entries.push(ContextMenuEntryRow::Action(
                        "Paste".to_string(),
                        ContextMenuAction::HierarchyPaste(None),
                    ));
                    entries.push(ContextMenuEntryRow::Separator);
                }
                entries.extend(
                    hierarchy_spawn_primary_context_entries()
                        .into_iter()
                        .map(|(label, action)| ContextMenuEntryRow::Action(label, action)),
                );
                entries.push(ContextMenuEntryRow::Separator);
                entries.push(ContextMenuEntryRow::Submenu(
                    "Physics".to_string(),
                    hierarchy_spawn_physics_context_entries(),
                ));
                entries.push(ContextMenuEntryRow::Submenu(
                    "Provided".to_string(),
                    hierarchy_spawn_provided_context_entries(),
                ));
                entries
            }
            ContextMenuKind::InspectorAddComponent(entity) => {
                let mut entries = Vec::new();
                let matches_selected = inspector_data.entity == Some(*entity);
                let has_transform = matches_selected && inspector_data.has_transform;
                let has_camera = matches_selected && inspector_data.has_camera;
                let has_light = matches_selected && inspector_data.has_light;
                let has_mesh_renderer = matches_selected && inspector_data.has_mesh_renderer;
                if !has_transform {
                    entries.push(ContextMenuEntryRow::Action(
                        "Transform".to_string(),
                        ContextMenuAction::InspectorAddTransform(*entity),
                    ));
                }
                if !has_camera {
                    entries.push(ContextMenuEntryRow::Action(
                        "Camera".to_string(),
                        ContextMenuAction::InspectorAddCamera(*entity),
                    ));
                }
                if !has_light {
                    entries.push(ContextMenuEntryRow::Submenu(
                        "Light".to_string(),
                        vec![
                            (
                                "Directional".to_string(),
                                ContextMenuAction::InspectorAddDirectionalLight(*entity),
                            ),
                            (
                                "Point".to_string(),
                                ContextMenuAction::InspectorAddPointLight(*entity),
                            ),
                            (
                                "Spot".to_string(),
                                ContextMenuAction::InspectorAddSpotLight(*entity),
                            ),
                        ],
                    ));
                }
                if !has_mesh_renderer {
                    entries.push(ContextMenuEntryRow::Action(
                        "Mesh Renderer".to_string(),
                        ContextMenuAction::InspectorAddMeshRenderer(*entity),
                    ));
                }
                if entries.is_empty() {
                    entries.push(ContextMenuEntryRow::Action(
                        "No Components Available".to_string(),
                        ContextMenuAction::Noop,
                    ));
                }
                entries
            }
            ContextMenuKind::DockTab {
                workspace_id,
                tab_id,
            } => {
                let mut entries = vec![
                    ContextMenuEntryRow::Action(
                        "Close".to_string(),
                        ContextMenuAction::DockCloseTab {
                            workspace_id: *workspace_id,
                            tab_id: *tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        "Detach".to_string(),
                        ContextMenuAction::DockDetachTab {
                            workspace_id: *workspace_id,
                            tab_id: *tab_id,
                        },
                    ),
                ];
                entries.push(ContextMenuEntryRow::Separator);
                entries.push(ContextMenuEntryRow::Submenu(
                    "New Pane".to_string(),
                    dock_spawn_tab_entries(*workspace_id),
                ));
                entries.push(ContextMenuEntryRow::Separator);
                entries.push(ContextMenuEntryRow::Submenu(
                    "New Window".to_string(),
                    dock_spawn_window_entries(*workspace_id),
                ));
                entries
            }
            ContextMenuKind::DockTabBar { workspace_id } => {
                let mut entries = vec![ContextMenuEntryRow::Submenu(
                    "New Pane".to_string(),
                    dock_spawn_tab_entries(*workspace_id),
                )];
                entries.push(ContextMenuEntryRow::Separator);
                entries.push(ContextMenuEntryRow::Submenu(
                    "New Window".to_string(),
                    dock_spawn_window_entries(*workspace_id),
                ));
                entries
            }
            ContextMenuKind::ContentEntry { path, is_dir } => {
                content_entry_context_rows(path, *is_dir, asset_clipboard_has_paths)
            }
            ContextMenuKind::ContentSurface => content_surface_context_rows(
                content_browser_current_dir.as_deref(),
                asset_clipboard_has_paths,
            ),
            ContextMenuKind::ViewportCanvas(target) => {
                let _mode = target.mode;
                let viewport_menu_state = viewport_states_snapshot.state_for(target.tab_id);
                RetainedViewportResolutionPreset::ALL
                    .into_iter()
                    .map(|preset| {
                        ContextMenuEntryRow::Action(
                            checked_label(viewport_menu_state.resolution == preset, preset.label()),
                            ContextMenuAction::ViewportSetResolution {
                                tab_id: target.tab_id,
                                preset,
                            },
                        )
                    })
                    .collect()
            }
            ContextMenuKind::ViewportRender(target) => {
                let viewport_menu_state = viewport_states_snapshot.state_for(target.tab_id);
                let mut rows = vec![ContextMenuEntryRow::Action(
                    checked_label(
                        viewport_menu_state.graph_template.is_none(),
                        "Default Graph",
                    ),
                    ContextMenuAction::ViewportSetGraphTemplate {
                        tab_id: target.tab_id,
                        template: None,
                    },
                )];
                for template in graph_templates() {
                    let selected = viewport_menu_state
                        .graph_template
                        .as_deref()
                        .is_some_and(|name| name == template.name);
                    rows.push(ContextMenuEntryRow::Action(
                        checked_label(selected, template.label),
                        ContextMenuAction::ViewportSetGraphTemplate {
                            tab_id: target.tab_id,
                            template: Some(template.name.to_string()),
                        },
                    ));
                }
                rows
            }
            ContextMenuKind::ViewportScripting(target) => {
                let viewport_menu_state = viewport_states_snapshot.state_for(target.tab_id);
                vec![ContextMenuEntryRow::Action(
                    checked_label(
                        viewport_menu_state.execute_scripts_in_edit_mode,
                        "Execute Scripts in Edit Mode",
                    ),
                    ContextMenuAction::ViewportToggleExecuteScriptsInEditMode {
                        tab_id: target.tab_id,
                    },
                )]
            }
            ContextMenuKind::ViewportGizmos(target) => {
                let viewport_menu_state = viewport_states_snapshot.state_for(target.tab_id);
                vec![
                    ContextMenuEntryRow::Action(
                        checked_label(viewport_menu_state.gizmos_in_play, "Show Gizmos in Play"),
                        ContextMenuAction::ViewportToggleGizmosInPlay {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(viewport_menu_state.show_camera_gizmos, "Show Camera Gizmos"),
                        ContextMenuAction::ViewportToggleShowCameraGizmos {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.show_directional_light_gizmos,
                            "Show Directional Light Gizmos",
                        ),
                        ContextMenuAction::ViewportToggleShowDirectionalLightGizmos {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.show_point_light_gizmos,
                            "Show Point Light Gizmos",
                        ),
                        ContextMenuAction::ViewportToggleShowPointLightGizmos {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.show_spot_light_gizmos,
                            "Show Spot Light Gizmos",
                        ),
                        ContextMenuAction::ViewportToggleShowSpotLightGizmos {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(viewport_menu_state.show_spline_paths, "Show Spline Paths"),
                        ContextMenuAction::ViewportToggleShowSplinePaths {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(viewport_menu_state.show_spline_points, "Show Spline Points"),
                        ContextMenuAction::ViewportToggleShowSplinePoints {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.show_navigation_gizmo,
                            "Show Navigation Gizmo",
                        ),
                        ContextMenuAction::ViewportToggleShowNavigationGizmo {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Separator,
                    ContextMenuEntryRow::Action(
                        checked_label(viewport_menu_state.gizmo_mode == GizmoMode::None, "Select"),
                        ContextMenuAction::ViewportSetGizmoMode {
                            tab_id: target.tab_id,
                            mode: GizmoMode::None,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.gizmo_mode == GizmoMode::Translate,
                            "Move",
                        ),
                        ContextMenuAction::ViewportSetGizmoMode {
                            tab_id: target.tab_id,
                            mode: GizmoMode::Translate,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.gizmo_mode == GizmoMode::Rotate,
                            "Rotate",
                        ),
                        ContextMenuAction::ViewportSetGizmoMode {
                            tab_id: target.tab_id,
                            mode: GizmoMode::Rotate,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(viewport_menu_state.gizmo_mode == GizmoMode::Scale, "Scale"),
                        ContextMenuAction::ViewportSetGizmoMode {
                            tab_id: target.tab_id,
                            mode: GizmoMode::Scale,
                        },
                    ),
                    ContextMenuEntryRow::Separator,
                    ContextMenuEntryRow::Action(
                        checked_label(gizmo_snap_snapshot.enabled, "Enable Snapping"),
                        ContextMenuAction::ViewportToggleGizmoSnapEnabled,
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(gizmo_snap_snapshot.ctrl_toggles, "Ctrl Inverts Snap"),
                        ContextMenuAction::ViewportToggleGizmoSnapCtrlToggles,
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(gizmo_snap_snapshot.shift_fine, "Shift Fine Step"),
                        ContextMenuAction::ViewportToggleGizmoSnapShiftFine,
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(gizmo_snap_snapshot.alt_coarse, "Alt Coarse Step"),
                        ContextMenuAction::ViewportToggleGizmoSnapAltCoarse,
                    ),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Move Step",
                        format!("{:.3}", gizmo_snap_snapshot.translate_step),
                        ContextMenuDragAction::GizmoSnapTranslateStep,
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Rotate Step (deg)",
                        format!("{:.2}", gizmo_snap_snapshot.rotate_step_degrees),
                        ContextMenuDragAction::GizmoSnapRotateStepDegrees,
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Scale Step",
                        format!("{:.3}", gizmo_snap_snapshot.scale_step),
                        ContextMenuDragAction::GizmoSnapScaleStep,
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Fine Mult",
                        format!("{:.3}", gizmo_snap_snapshot.fine_scale),
                        ContextMenuDragAction::GizmoSnapFineScale,
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Coarse Mult",
                        format!("{:.3}", gizmo_snap_snapshot.coarse_scale),
                        ContextMenuDragAction::GizmoSnapCoarseScale,
                    )),
                ]
            }
            ContextMenuKind::ViewportFreecam(target) => {
                let viewport_menu_state = viewport_states_snapshot.state_for(target.tab_id);
                vec![
                    ContextMenuEntryRow::Action(
                        "Defaults".to_string(),
                        ContextMenuAction::ViewportResetFreecam {
                            tab_id: target.tab_id,
                        },
                    ),
                    ContextMenuEntryRow::Separator,
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.freecam_speed == RetainedViewportFreecamSpeed::Slow,
                            "Slow Profile",
                        ),
                        ContextMenuAction::ViewportSetFreecamSpeed {
                            tab_id: target.tab_id,
                            speed: RetainedViewportFreecamSpeed::Slow,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.freecam_speed
                                == RetainedViewportFreecamSpeed::Normal,
                            "Normal Profile",
                        ),
                        ContextMenuAction::ViewportSetFreecamSpeed {
                            tab_id: target.tab_id,
                            speed: RetainedViewportFreecamSpeed::Normal,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(
                            viewport_menu_state.freecam_speed == RetainedViewportFreecamSpeed::Fast,
                            "Fast Profile",
                        ),
                        ContextMenuAction::ViewportSetFreecamSpeed {
                            tab_id: target.tab_id,
                            speed: RetainedViewportFreecamSpeed::Fast,
                        },
                    ),
                    ContextMenuEntryRow::Separator,
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Sensitivity",
                        format!("{:.3}", viewport_menu_state.freecam_sensitivity),
                        ContextMenuDragAction::ViewportFreecamSensitivity {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Smoothing",
                        format!("{:.4}", viewport_menu_state.freecam_smoothing),
                        ContextMenuDragAction::ViewportFreecamSmoothing {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Acceleration",
                        format!("{:.2}", viewport_menu_state.freecam_move_accel),
                        ContextMenuDragAction::ViewportFreecamMoveAccel {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Deceleration",
                        format!("{:.2}", viewport_menu_state.freecam_move_decel),
                        ContextMenuDragAction::ViewportFreecamMoveDecel {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Boost Mult",
                        format!("{:.2}", viewport_menu_state.freecam_boost_multiplier),
                        ContextMenuDragAction::ViewportFreecamBoostMultiplier {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::Separator,
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Adjust Step",
                        format!("{:.2}", viewport_menu_state.freecam_speed_step),
                        ContextMenuDragAction::ViewportFreecamSpeedStep {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Min Speed",
                        format!("{:.2}", viewport_menu_state.freecam_speed_min),
                        ContextMenuDragAction::ViewportFreecamSpeedMin {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Max Speed",
                        format!("{:.2}", viewport_menu_state.freecam_speed_max),
                        ContextMenuDragAction::ViewportFreecamSpeedMax {
                            tab_id: target.tab_id,
                        },
                    )),
                ]
            }
            ContextMenuKind::ViewportOrbit(target) => {
                let viewport_menu_state = viewport_states_snapshot.state_for(target.tab_id);
                vec![
                    ContextMenuEntryRow::Action(
                        checked_label(viewport_menu_state.orbit_selected_entity, "Orbit Selected"),
                        ContextMenuAction::ViewportSetOrbitSelected {
                            tab_id: target.tab_id,
                            orbit_selected_entity: true,
                        },
                    ),
                    ContextMenuEntryRow::Action(
                        checked_label(!viewport_menu_state.orbit_selected_entity, "Orbit Camera"),
                        ContextMenuAction::ViewportSetOrbitSelected {
                            tab_id: target.tab_id,
                            orbit_selected_entity: false,
                        },
                    ),
                    ContextMenuEntryRow::Separator,
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Orbit Sensitivity",
                        format!("{:.4}", viewport_menu_state.freecam_orbit_pan_sensitivity),
                        ContextMenuDragAction::ViewportOrbitPanSensitivity {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Pan Sensitivity",
                        format!("{:.5}", viewport_menu_state.freecam_pan_sensitivity),
                        ContextMenuDragAction::ViewportPanSensitivity {
                            tab_id: target.tab_id,
                        },
                    )),
                    ContextMenuEntryRow::DragValue(drag_value_row(
                        "Orbit Distance",
                        format!("{:.2}", viewport_menu_state.orbit_distance),
                        ContextMenuDragAction::ViewportOrbitDistance {
                            tab_id: target.tab_id,
                        },
                    )),
                ]
            }
            ContextMenuKind::ViewportAdvanced(target) => vec![
                ContextMenuEntryRow::Action(
                    "Reset View".to_string(),
                    ContextMenuAction::ViewportResetView {
                        tab_id: target.tab_id,
                    },
                ),
                ContextMenuEntryRow::Action(
                    "Frame Selection".to_string(),
                    ContextMenuAction::ViewportFrameSelection {
                        tab_id: target.tab_id,
                    },
                ),
            ],
            ContextMenuKind::ToolbarLayout => toolbar_layout_rows.clone(),
        };

        if !entries.is_empty() {
            let menu_id = RETAINED_SHELL_ROOT.child("context-menu");
            let row_height = 22.0;
            let separator_height = 8.0;
            let menu_width = 212.0;
            let submenu_width = 196.0;
            let menu_height = 8.0
                + entries
                    .iter()
                    .map(|entry| match entry {
                        ContextMenuEntryRow::Action(_, _)
                        | ContextMenuEntryRow::Submenu(_, _)
                        | ContextMenuEntryRow::DragValue(_) => row_height,
                        ContextMenuEntryRow::Separator => separator_height,
                    })
                    .sum::<f32>();
            let menu_x = menu_state
                .position
                .x
                .clamp(4.0, (viewport_size.x - menu_width - 4.0).max(4.0));
            let menu_y = menu_state
                .position
                .y
                .clamp(4.0, (viewport_size.y - menu_height - 4.0).max(4.0));
            current_context_menu_bounds = Some(UiRect {
                x: menu_x,
                y: menu_y,
                width: menu_width,
                height: menu_height,
            });
            let mut open_submenu = menu_state.open_submenu;
            let mut submenu_hovered_parent = None::<usize>;
            let mut submenu_bounds = None::<UiRect>;
            let merge_bounds = |lhs: UiRect, rhs: UiRect| UiRect {
                x: lhs.x.min(rhs.x),
                y: lhs.y.min(rhs.y),
                width: (lhs.x + lhs.width).max(rhs.x + rhs.width) - lhs.x.min(rhs.x),
                height: (lhs.y + lhs.height).max(rhs.y + rhs.height) - lhs.y.min(rhs.y),
            };

            retained.upsert(RetainedUiNode::new(
                menu_id,
                UiWidget::Container,
                absolute_style(
                    menu_x,
                    menu_y,
                    menu_width,
                    menu_height,
                    UiVisualStyle {
                        background: Some(UiColor::rgba(0.11, 0.14, 0.18, 0.97)),
                        border_color: Some(UiColor::rgba(0.44, 0.53, 0.66, 0.95)),
                        border_width: 1.0,
                        corner_radius: 0.0,
                        clip: true,
                    },
                ),
            ));

            let mut menu_children = Vec::new();
            let mut cursor_y = 4.0;
            for (index, entry) in entries.into_iter().enumerate() {
                match entry {
                    ContextMenuEntryRow::Action(label, action) => {
                        let row_id = menu_id.child("row").child(index as u64);
                        let label_id = row_id.child("label");
                        let hit_id = row_id.child("hit");
                        let hovered = ui_interaction
                            .hovered
                            .is_some_and(|id| id == hit_id || id == row_id || id == label_id);
                        let active = ui_interaction
                            .active
                            .is_some_and(|id| id == hit_id || id == row_id || id == label_id);
                        let row_background = if active {
                            UiColor::rgba(0.30, 0.42, 0.60, 0.95)
                        } else if hovered {
                            UiColor::rgba(0.25, 0.34, 0.50, 0.95)
                        } else {
                            UiColor::rgba(0.16, 0.20, 0.28, 0.92)
                        };
                        let row_border = if active || hovered {
                            UiColor::rgba(0.60, 0.76, 0.97, 0.94)
                        } else {
                            UiColor::rgba(0.30, 0.38, 0.50, 0.86)
                        };
                        retained.upsert(RetainedUiNode::new(
                            row_id,
                            UiWidget::Container,
                            absolute_style(
                                4.0,
                                cursor_y,
                                menu_width - 8.0,
                                row_height - 1.0,
                                UiVisualStyle {
                                    background: Some(row_background),
                                    border_color: Some(row_border),
                                    border_width: 1.0,
                                    corner_radius: 0.0,
                                    clip: false,
                                },
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            label_id,
                            UiWidget::Label(UiLabel {
                                text: UiTextValue::from(label),
                                style: UiTextStyle {
                                    color: UiColor::rgba(0.94, 0.97, 1.0, 1.0),
                                    font_size: 11.0,
                                    align_h: UiTextAlign::Start,
                                    align_v: UiTextAlign::Center,
                                    wrap: false,
                                },
                            }),
                            absolute_style(
                                10.0,
                                cursor_y,
                                menu_width - 20.0,
                                row_height - 1.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            hit_id,
                            UiWidget::HitBox,
                            absolute_style(
                                4.0,
                                cursor_y,
                                menu_width - 8.0,
                                row_height - 1.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        next_context_menu_actions.insert(row_id, action.clone());
                        next_context_menu_actions.insert(label_id, action.clone());
                        next_context_menu_actions.insert(hit_id, action);
                        menu_children.push(row_id);
                        menu_children.push(label_id);
                        menu_children.push(hit_id);
                        cursor_y += row_height;
                    }
                    ContextMenuEntryRow::Submenu(label, submenu_entries) => {
                        let row_id = menu_id.child("row").child(index as u64);
                        let label_id = row_id.child("label");
                        let arrow_id = row_id.child("arrow");
                        let hit_id = row_id.child("hit");
                        let hovered = ui_interaction.hovered.is_some_and(|id| {
                            id == hit_id || id == row_id || id == label_id || id == arrow_id
                        });
                        let active = ui_interaction.active.is_some_and(|id| {
                            id == hit_id || id == row_id || id == label_id || id == arrow_id
                        });
                        if hovered || active {
                            open_submenu = Some(index);
                            submenu_hovered_parent = Some(index);
                        }
                        let row_background = if active {
                            UiColor::rgba(0.30, 0.42, 0.60, 0.95)
                        } else if hovered || open_submenu == Some(index) {
                            UiColor::rgba(0.25, 0.34, 0.50, 0.95)
                        } else {
                            UiColor::rgba(0.16, 0.20, 0.28, 0.92)
                        };
                        let row_border = if active || hovered || open_submenu == Some(index) {
                            UiColor::rgba(0.60, 0.76, 0.97, 0.94)
                        } else {
                            UiColor::rgba(0.30, 0.38, 0.50, 0.86)
                        };
                        retained.upsert(RetainedUiNode::new(
                            row_id,
                            UiWidget::Container,
                            absolute_style(
                                4.0,
                                cursor_y,
                                menu_width - 8.0,
                                row_height - 1.0,
                                UiVisualStyle {
                                    background: Some(row_background),
                                    border_color: Some(row_border),
                                    border_width: 1.0,
                                    corner_radius: 0.0,
                                    clip: false,
                                },
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            label_id,
                            UiWidget::Label(UiLabel {
                                text: UiTextValue::from(label),
                                style: UiTextStyle {
                                    color: UiColor::rgba(0.94, 0.97, 1.0, 1.0),
                                    font_size: 11.0,
                                    align_h: UiTextAlign::Start,
                                    align_v: UiTextAlign::Center,
                                    wrap: false,
                                },
                            }),
                            absolute_style(
                                10.0,
                                cursor_y,
                                menu_width - 30.0,
                                row_height - 1.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            arrow_id,
                            UiWidget::Label(UiLabel {
                                text: UiTextValue::from(">"),
                                style: UiTextStyle {
                                    color: UiColor::rgba(0.80, 0.88, 0.98, 1.0),
                                    font_size: 11.0,
                                    align_h: UiTextAlign::Center,
                                    align_v: UiTextAlign::Center,
                                    wrap: false,
                                },
                            }),
                            absolute_style(
                                menu_width - 18.0,
                                cursor_y,
                                8.0,
                                row_height - 1.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            hit_id,
                            UiWidget::HitBox,
                            absolute_style(
                                4.0,
                                cursor_y,
                                menu_width - 8.0,
                                row_height - 1.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        menu_children.push(row_id);
                        menu_children.push(label_id);
                        menu_children.push(arrow_id);
                        menu_children.push(hit_id);

                        if open_submenu == Some(index) {
                            let submenu_id = menu_id.child("submenu").child(index as u64);
                            let submenu_height = 8.0 + submenu_entries.len() as f32 * row_height;
                            let default_right_x = menu_x + menu_width - 1.0;
                            let submenu_x =
                                if default_right_x + submenu_width + 4.0 <= viewport_size.x {
                                    default_right_x
                                } else {
                                    (menu_x - submenu_width + 1.0).max(4.0)
                                };
                            let submenu_y = (menu_y + cursor_y - 1.0)
                                .clamp(4.0, (viewport_size.y - submenu_height - 4.0).max(4.0));
                            let submenu_rect = UiRect {
                                x: submenu_x,
                                y: submenu_y,
                                width: submenu_width,
                                height: submenu_height,
                            };
                            submenu_bounds = Some(submenu_rect);
                            if let Some(bounds) = current_context_menu_bounds {
                                current_context_menu_bounds =
                                    Some(merge_bounds(bounds, submenu_rect));
                            }

                            retained.upsert(RetainedUiNode::new(
                                submenu_id,
                                UiWidget::Container,
                                absolute_style(
                                    submenu_x,
                                    submenu_y,
                                    submenu_width,
                                    submenu_height,
                                    UiVisualStyle {
                                        background: Some(UiColor::rgba(0.11, 0.14, 0.18, 0.98)),
                                        border_color: Some(UiColor::rgba(0.44, 0.53, 0.66, 0.95)),
                                        border_width: 1.0,
                                        corner_radius: 0.0,
                                        clip: true,
                                    },
                                ),
                            ));

                            let mut submenu_children = Vec::new();
                            for (item_index, (item_label, item_action)) in
                                submenu_entries.into_iter().enumerate()
                            {
                                let item_row_y = 4.0 + item_index as f32 * row_height;
                                let item_row_id = submenu_id.child("row").child(item_index as u64);
                                let item_label_id = item_row_id.child("label");
                                let item_hit_id = item_row_id.child("hit");
                                let item_hovered = ui_interaction.hovered.is_some_and(|id| {
                                    id == item_hit_id || id == item_row_id || id == item_label_id
                                });
                                let item_active = ui_interaction.active.is_some_and(|id| {
                                    id == item_hit_id || id == item_row_id || id == item_label_id
                                });
                                let item_background = if item_active {
                                    UiColor::rgba(0.30, 0.42, 0.60, 0.95)
                                } else if item_hovered {
                                    UiColor::rgba(0.25, 0.34, 0.50, 0.95)
                                } else {
                                    UiColor::rgba(0.16, 0.20, 0.28, 0.92)
                                };
                                let item_border = if item_active || item_hovered {
                                    UiColor::rgba(0.60, 0.76, 0.97, 0.94)
                                } else {
                                    UiColor::rgba(0.30, 0.38, 0.50, 0.86)
                                };
                                retained.upsert(RetainedUiNode::new(
                                    item_row_id,
                                    UiWidget::Container,
                                    absolute_style(
                                        4.0,
                                        item_row_y,
                                        submenu_width - 8.0,
                                        row_height - 1.0,
                                        UiVisualStyle {
                                            background: Some(item_background),
                                            border_color: Some(item_border),
                                            border_width: 1.0,
                                            corner_radius: 0.0,
                                            clip: false,
                                        },
                                    ),
                                ));
                                retained.upsert(RetainedUiNode::new(
                                    item_label_id,
                                    UiWidget::Label(UiLabel {
                                        text: UiTextValue::from(item_label),
                                        style: UiTextStyle {
                                            color: UiColor::rgba(0.94, 0.97, 1.0, 1.0),
                                            font_size: 11.0,
                                            align_h: UiTextAlign::Start,
                                            align_v: UiTextAlign::Center,
                                            wrap: false,
                                        },
                                    }),
                                    absolute_style(
                                        10.0,
                                        item_row_y,
                                        submenu_width - 20.0,
                                        row_height - 1.0,
                                        UiVisualStyle::default(),
                                    ),
                                ));
                                retained.upsert(RetainedUiNode::new(
                                    item_hit_id,
                                    UiWidget::HitBox,
                                    absolute_style(
                                        4.0,
                                        item_row_y,
                                        submenu_width - 8.0,
                                        row_height - 1.0,
                                        UiVisualStyle::default(),
                                    ),
                                ));
                                next_context_menu_actions.insert(item_row_id, item_action.clone());
                                next_context_menu_actions
                                    .insert(item_label_id, item_action.clone());
                                next_context_menu_actions.insert(item_hit_id, item_action);
                                submenu_children.push(item_row_id);
                                submenu_children.push(item_label_id);
                                submenu_children.push(item_hit_id);
                            }
                            retained.set_children(submenu_id, submenu_children);
                            canvas_children.push(submenu_id);
                        }

                        cursor_y += row_height;
                    }
                    ContextMenuEntryRow::DragValue(row) => {
                        let row_id = menu_id.child("drag").child(index as u64);
                        let label_id = row_id.child("label");
                        let value_box_id = row_id.child("value-box");
                        let value_label_id = row_id.child("value");
                        let hit_id = row_id.child("hit");
                        let hovered = ui_interaction.hovered.is_some_and(|id| {
                            id == hit_id
                                || id == row_id
                                || id == label_id
                                || id == value_box_id
                                || id == value_label_id
                        });
                        let dragging_this = context_menu_drag_state
                            .is_some_and(|drag| drag.action == row.action && drag.active);
                        let active = ui_interaction.active.is_some_and(|id| {
                            id == hit_id
                                || id == row_id
                                || id == label_id
                                || id == value_box_id
                                || id == value_label_id
                        }) || dragging_this;
                        let row_background = if active {
                            UiColor::rgba(0.29, 0.41, 0.57, 0.96)
                        } else if hovered {
                            UiColor::rgba(0.22, 0.31, 0.45, 0.95)
                        } else {
                            UiColor::rgba(0.14, 0.18, 0.25, 0.94)
                        };
                        let row_border = if active || hovered {
                            UiColor::rgba(0.58, 0.73, 0.92, 0.96)
                        } else {
                            UiColor::rgba(0.28, 0.35, 0.46, 0.90)
                        };
                        retained.upsert(RetainedUiNode::new(
                            row_id,
                            UiWidget::Container,
                            absolute_style(
                                4.0,
                                cursor_y,
                                menu_width - 8.0,
                                row_height - 1.0,
                                UiVisualStyle {
                                    background: Some(row_background),
                                    border_color: Some(row_border),
                                    border_width: 1.0,
                                    corner_radius: 0.0,
                                    clip: false,
                                },
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            label_id,
                            UiWidget::Label(UiLabel {
                                text: UiTextValue::from(row.label),
                                style: UiTextStyle {
                                    color: UiColor::rgba(0.88, 0.93, 1.0, 1.0),
                                    font_size: 11.0,
                                    align_h: UiTextAlign::Start,
                                    align_v: UiTextAlign::Center,
                                    wrap: false,
                                },
                            }),
                            absolute_style(
                                10.0,
                                cursor_y,
                                menu_width - 96.0,
                                row_height - 1.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            value_box_id,
                            UiWidget::Container,
                            absolute_style(
                                menu_width - 86.0,
                                cursor_y + 2.0,
                                74.0,
                                row_height - 5.0,
                                UiVisualStyle {
                                    background: Some(UiColor::rgba(0.10, 0.13, 0.18, 0.95)),
                                    border_color: Some(UiColor::rgba(0.34, 0.43, 0.56, 0.92)),
                                    border_width: 1.0,
                                    corner_radius: 0.0,
                                    clip: false,
                                },
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            value_label_id,
                            UiWidget::Label(UiLabel {
                                text: UiTextValue::from(row.value_text),
                                style: UiTextStyle {
                                    color: UiColor::rgba(0.95, 0.98, 1.0, 1.0),
                                    font_size: 10.5,
                                    align_h: UiTextAlign::Center,
                                    align_v: UiTextAlign::Center,
                                    wrap: false,
                                },
                            }),
                            absolute_style(
                                menu_width - 86.0,
                                cursor_y + 2.0,
                                74.0,
                                row_height - 5.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        retained.upsert(RetainedUiNode::new(
                            hit_id,
                            UiWidget::HitBox,
                            absolute_style(
                                4.0,
                                cursor_y,
                                menu_width - 8.0,
                                row_height - 1.0,
                                UiVisualStyle::default(),
                            ),
                        ));
                        next_context_menu_drag_actions.insert(row_id, row.action);
                        next_context_menu_drag_actions.insert(label_id, row.action);
                        next_context_menu_drag_actions.insert(value_box_id, row.action);
                        next_context_menu_drag_actions.insert(value_label_id, row.action);
                        next_context_menu_drag_actions.insert(hit_id, row.action);
                        menu_children.push(row_id);
                        menu_children.push(label_id);
                        menu_children.push(value_box_id);
                        menu_children.push(value_label_id);
                        menu_children.push(hit_id);
                        cursor_y += row_height;
                    }
                    ContextMenuEntryRow::Separator => {
                        let separator_id = menu_id.child("separator").child(index as u64);
                        retained.upsert(RetainedUiNode::new(
                            separator_id,
                            UiWidget::Container,
                            absolute_style(
                                8.0,
                                cursor_y + (separator_height * 0.5),
                                menu_width - 16.0,
                                1.0,
                                UiVisualStyle {
                                    background: Some(UiColor::rgba(0.30, 0.38, 0.49, 0.80)),
                                    border_color: None,
                                    border_width: 0.0,
                                    corner_radius: 0.0,
                                    clip: false,
                                },
                            ),
                        ));
                        menu_children.push(separator_id);
                        cursor_y += separator_height;
                    }
                }
            }
            if let Some(active_submenu) = open_submenu {
                let pointer_over_submenu = pointer_position
                    .zip(submenu_bounds)
                    .is_some_and(|(pointer, bounds)| point_in_ui_rect(bounds, pointer));
                if submenu_hovered_parent != Some(active_submenu) && !pointer_over_submenu {
                    open_submenu = None;
                }
            }
            if let Some(menu_state) = context_menu_state.as_mut() {
                menu_state.open_submenu = open_submenu;
            }
            retained.set_children(menu_id, menu_children);
            canvas_children.push(menu_id);
        }
    }

    retained.set_children(canvas_id, canvas_children);
    drop(ui_runtime_state);

    if let Some(max_scroll) = observed_hierarchy_scroll_max {
        hierarchy_scroll = hierarchy_scroll.clamp(0.0, max_scroll.max(0.0));
    }
    if let Some(max_scroll) = observed_content_browser_scroll_max
        && let Some(mut browser_state) = world.get_resource_mut::<AssetBrowserState>()
    {
        browser_state.grid_scroll = browser_state.grid_scroll.clamp(0.0, max_scroll.max(0.0));
        previous_content_browser_scroll_max = max_scroll.max(0.0);
    }
    if let Some(max_scroll) = observed_console_scroll_max
        && let Some(mut console_state) = world.get_resource_mut::<EditorConsoleState>()
    {
        if console_state.auto_scroll {
            console_state.scroll = max_scroll.max(0.0);
        } else {
            console_state.scroll = console_state.scroll.clamp(0.0, max_scroll.max(0.0));
        }
        previous_console_scroll_max = max_scroll.max(0.0);
    }

    if let Some(mut pane_interaction_state) =
        world.get_resource_mut::<EditorRetainedPaneInteractionState>()
    {
        pane_interaction_state.hierarchy_click_targets = next_hierarchy_click_targets;
        pane_interaction_state.hierarchy_actions = next_hierarchy_actions;
        pane_interaction_state.hierarchy_drop_surface_hits = next_hierarchy_drop_surface_hits;
        pane_interaction_state.hierarchy_expanded = hierarchy_expanded;
        pane_interaction_state.hierarchy_scroll = hierarchy_scroll;
        pane_interaction_state.hierarchy_drag = hierarchy_drag_state;
        pane_interaction_state.project_actions = next_project_actions;
        pane_interaction_state.project_text_fields = next_project_text_fields;
        pane_interaction_state.project_text_cursors = project_text_cursors;
        pane_interaction_state.focused_project_text_field = focused_project_text_field;
        pane_interaction_state.content_browser_actions = next_content_browser_actions;
        pane_interaction_state.content_browser_text_fields = next_content_browser_text_fields;
        pane_interaction_state.content_browser_text_cursors = content_browser_text_cursors;
        pane_interaction_state.focused_content_browser_text_field =
            focused_content_browser_text_field;
        pane_interaction_state.content_browser_viewports = current_content_browser_viewports;
        pane_interaction_state.content_browser_scroll_regions =
            current_content_browser_scroll_regions;
        pane_interaction_state.content_browser_scroll_max = previous_content_browser_scroll_max;
        pane_interaction_state.content_browser_drag_select_active =
            content_browser_drag_select_active;
        pane_interaction_state.content_browser_drag_select_additive =
            content_browser_drag_select_additive;
        pane_interaction_state.content_browser_entry_drag = content_browser_entry_drag;
        pane_interaction_state.inspector_actions = next_inspector_actions;
        pane_interaction_state.inspector_drag_actions = next_inspector_drag_actions;
        pane_interaction_state.inspector_drag = inspector_drag_state;
        pane_interaction_state.inspector_text_fields = next_inspector_text_fields;
        pane_interaction_state.inspector_text_cursors = inspector_text_cursors;
        pane_interaction_state.focused_inspector_text_field = focused_inspector_text_field;
        pane_interaction_state.inspector_light_color_picker = inspector_light_color_picker;
        pane_interaction_state.console_actions = next_console_actions;
        pane_interaction_state.console_text_fields = next_console_text_fields;
        pane_interaction_state.console_text_cursors = console_text_cursors;
        pane_interaction_state.focused_console_text_field = focused_console_text_field;
        pane_interaction_state.console_viewports = current_console_viewports;
        pane_interaction_state.console_scroll_regions = current_console_scroll_regions;
        pane_interaction_state.console_scroll_max = previous_console_scroll_max;
        pane_interaction_state.timeline_actions = next_timeline_actions;
        pane_interaction_state.toolbar_actions = next_toolbar_actions;
        pane_interaction_state.viewport_actions = next_viewport_actions;
        pane_interaction_state.viewport_surfaces = next_viewport_surfaces;
        pane_interaction_state.dock_tab_actions = next_dock_tab_actions;
        pane_interaction_state.dock_tab_context_targets = next_dock_tab_context_targets;
        pane_interaction_state.dock_tab_bar_context_targets = next_dock_tab_bar_context_targets;
        pane_interaction_state.dock_split_actions = next_dock_split_actions;
        pane_interaction_state.history_actions = next_history_actions;
        pane_interaction_state.audio_mixer_actions = next_audio_mixer_actions;
        pane_interaction_state.profiler_actions = next_profiler_actions;
        pane_interaction_state.material_editor_actions = next_material_editor_actions;
        pane_interaction_state.context_menu_actions = next_context_menu_actions;
        pane_interaction_state.context_menu_drag_actions = next_context_menu_drag_actions;
        pane_interaction_state.context_menu_drag = context_menu_drag_state;
        pane_interaction_state.window_title_hits = next_window_title_hits;
        pane_interaction_state.window_controls = next_window_controls;
        pane_interaction_state.collapsed_windows = previous_collapsed_windows;
        pane_interaction_state.collapsed_window_heights = previous_collapsed_window_heights;
        pane_interaction_state.dragged_window = dragged_window;
        pane_interaction_state.dragged_window_mode = dragged_window_mode;
        pane_interaction_state.dragged_window_edges = dragged_window_edges;
        pane_interaction_state.dragged_window_start_rect = dragged_window_start_rect;
        pane_interaction_state.dragged_pointer_start = dragged_pointer_start;
        pane_interaction_state.dock_tab_drag = dock_tab_drag;
        pane_interaction_state.dock_split_drag = dock_split_drag;
        pane_interaction_state.viewport_preview_drag = viewport_preview_drag_state;
        pane_interaction_state.primary_pointer_down_previous = pointer_down;
        pane_interaction_state.secondary_pointer_down_previous = secondary_pointer_down;
        pane_interaction_state.pointer_down_previous = window_drag_pointer_down;
        pane_interaction_state.context_menu = context_menu_state;
        pane_interaction_state.context_menu_bounds = current_context_menu_bounds;
        pane_interaction_state.last_project_open = Some(project_open);
    }

    if let Some(mut graph_interaction_state) =
        world.get_resource_mut::<EditorRetainedGraphInteractionState>()
    {
        graph_interaction_state.frames = next_graph_frames;
        let active: HashSet<UiId> = active_graph_ids.into_iter().collect();
        graph_interaction_state
            .controllers
            .retain(|graph_id, _| active.contains(graph_id));
    }
}

fn content_action_double_clicked(world: &mut World, action: &ContentBrowserPaneAction) -> bool {
    let ContentBrowserPaneAction::SelectEntry { path, .. } = action else {
        return false;
    };

    let Some(mut browser_state) = world.get_resource_mut::<AssetBrowserState>() else {
        return false;
    };

    let now = Instant::now();
    let is_double_click = browser_state.last_click_path.as_ref() == Some(path)
        && browser_state
            .last_click_instant
            .is_some_and(|last| now.duration_since(last) <= CONTENT_BROWSER_DOUBLE_CLICK_THRESHOLD);
    browser_state.last_click_path = Some(path.clone());
    browser_state.last_click_instant = Some(now);
    is_double_click
}

fn apply_context_menu_action(world: &mut World, action: ContextMenuAction) {
    match action {
        ContextMenuAction::Noop => {}
        ContextMenuAction::HierarchyAddEntity => {
            apply_hierarchy_action(world, HierarchyPaneAction::AddEntity);
        }
        ContextMenuAction::HierarchySpawn(kind) => {
            if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
                queue.push(EditorCommand::CreateEntity { kind });
            }
        }
        ContextMenuAction::HierarchySetActiveCamera(entity) => {
            if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
                queue.push(EditorCommand::SetActiveCamera { entity });
            }
        }
        ContextMenuAction::HierarchySelectSceneRoot(scene_root) => {
            if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>()
            {
                selected.0 = Some(scene_root);
            }
        }
        ContextMenuAction::HierarchyDeleteEntity(entity) => {
            if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
                queue.push(EditorCommand::DeleteEntity { entity });
            }
        }
        ContextMenuAction::HierarchyDeleteHierarchy(entity) => {
            delete_hierarchy_subtree(world, entity);
        }
        ContextMenuAction::HierarchyUnparent(entity) => {
            let _ = reparent_hierarchy_entity(world, entity, None);
        }
        ContextMenuAction::HierarchyFocus(entity) => {
            if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>()
            {
                selected.0 = Some(entity);
            }
            let mut expanded_set = world
                .get_resource::<EditorRetainedPaneInteractionState>()
                .map(|state| state.hierarchy_expanded.clone())
                .unwrap_or_default();
            hierarchy_expand_ancestors(world, &mut expanded_set, entity);
            if let Some(mut interaction) =
                world.get_resource_mut::<EditorRetainedPaneInteractionState>()
            {
                interaction.hierarchy_expanded = expanded_set;
            }
        }
        ContextMenuAction::HierarchyCopy(entity) => {
            if world.get_entity(entity).is_ok() {
                if let Some(mut clipboard) = world.get_resource_mut::<HierarchyClipboardState>() {
                    clipboard.entity = Some(entity);
                    clipboard.cut = false;
                } else {
                    world.insert_resource(HierarchyClipboardState {
                        entity: Some(entity),
                        cut: false,
                    });
                }
            }
        }
        ContextMenuAction::HierarchyCut(entity) => {
            if world.get_entity(entity).is_ok() {
                if let Some(mut clipboard) = world.get_resource_mut::<HierarchyClipboardState>() {
                    clipboard.entity = Some(entity);
                    clipboard.cut = true;
                } else {
                    world.insert_resource(HierarchyClipboardState {
                        entity: Some(entity),
                        cut: true,
                    });
                }
            }
        }
        ContextMenuAction::HierarchyPaste(parent) => {
            paste_hierarchy_clipboard(world, parent);
        }
        ContextMenuAction::HierarchyDuplicate(entity) => {
            let _ = duplicate_entity_shallow(world, entity, None);
        }
        ContextMenuAction::HierarchyRename(entity) => {
            if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>()
            {
                selected.0 = Some(entity);
            }
            let cursor = world
                .get::<Name>(entity)
                .map(|name| name.as_str().chars().count())
                .unwrap_or(0);
            if let Some(mut interaction) =
                world.get_resource_mut::<EditorRetainedPaneInteractionState>()
            {
                interaction.focused_inspector_text_field = Some(InspectorPaneTextField::Name);
                interaction
                    .inspector_text_cursors
                    .insert(InspectorPaneTextField::Name, cursor);
            }
        }
        ContextMenuAction::DockCloseTab {
            workspace_id,
            tab_id,
        } => {
            close_workspace_tab(world, workspace_id, tab_id);
        }
        ContextMenuAction::DockDetachTab {
            workspace_id,
            tab_id,
        } => {
            detach_workspace_tab(world, workspace_id, tab_id);
        }
        ContextMenuAction::DockOpenTab { workspace_id, kind } => {
            open_builtin_tab_in_workspace(world, workspace_id, kind);
        }
        ContextMenuAction::DockOpenTabInNewWindow { workspace_id, kind } => {
            open_builtin_tab_in_new_window(world, workspace_id, kind);
        }
        ContextMenuAction::ContentOpen { path, is_dir } => {
            apply_content_browser_action(
                world,
                ContentBrowserPaneAction::SelectEntry {
                    path,
                    is_dir,
                    index: 0,
                },
                false,
                false,
                true,
            );
        }
        ContextMenuAction::ContentOpenScene(path) => {
            if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
                queue.push(EditorCommand::OpenScene { path });
            }
        }
        ContextMenuAction::ContentAddModelToScene(path) => {
            if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
                queue.push(EditorCommand::CreateEntity {
                    kind: SpawnKind::SceneAsset(path),
                });
            }
        }
        ContextMenuAction::ContentOpenInFileBrowser(path) => {
            if let Err(err) = open_in_file_browser(&path) {
                set_asset_status(world, format!("Open in file browser failed: {err}"));
            }
        }
        ContextMenuAction::ContentCopyPaths { anchor, mode } => {
            set_asset_clipboard(world, anchor.as_deref(), mode);
        }
        ContextMenuAction::ContentPaste { target } => {
            let destination = resolve_content_paste_target(world, target.as_deref());
            let _ = paste_assets_to_directory(world, destination.as_deref());
        }
        ContextMenuAction::ContentDuplicatePaths { anchor } => {
            let paths = selected_assets_for_action(world, anchor.as_deref());
            if paths.is_empty() {
                set_asset_status(world, "No assets selected");
            } else {
                duplicate_assets(world, &paths);
            }
        }
        ContextMenuAction::ContentDeletePaths { anchor } => {
            let paths = selected_assets_for_action(world, anchor.as_deref());
            if paths.is_empty() {
                set_asset_status(world, "No assets selected");
            } else {
                delete_assets(world, &paths);
            }
        }
        ContextMenuAction::ContentCreateAsset {
            directory,
            name,
            kind,
        } => {
            if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
                queue.push(EditorCommand::CreateAsset {
                    directory,
                    name: name.to_string(),
                    kind,
                });
            }
        }
        ContextMenuAction::ContentSelectFolder(path) => {
            apply_content_browser_action(
                world,
                ContentBrowserPaneAction::SelectFolder(path),
                false,
                false,
                false,
            );
        }
        ContextMenuAction::ContentRefresh => {
            apply_content_browser_action(
                world,
                ContentBrowserPaneAction::Refresh,
                false,
                false,
                false,
            );
        }
        ContextMenuAction::InspectorAddTransform(entity) => {
            let mut changed = false;
            if world.get::<BevyTransform>(entity).is_none() {
                world.entity_mut(entity).insert(BevyTransform::default());
                changed = true;
            }
            if changed && let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                scene.dirty = true;
            }
        }
        ContextMenuAction::InspectorAddCamera(entity) => {
            let mut changed = false;
            if world.get::<BevyTransform>(entity).is_none() {
                world.entity_mut(entity).insert(BevyTransform::default());
            }
            if world.get::<BevyCamera>(entity).is_none() {
                world.entity_mut(entity).insert(BevyCamera::default());
                changed = true;
            }
            if changed && let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                scene.dirty = true;
            }
        }
        ContextMenuAction::InspectorAddDirectionalLight(entity) => {
            let mut changed = false;
            if world.get::<BevyTransform>(entity).is_none() {
                world.entity_mut(entity).insert(BevyTransform::default());
            }
            if world.get::<BevyLight>(entity).is_none() {
                world
                    .entity_mut(entity)
                    .insert(BevyWrapper(Light::directional(
                        glam::vec3(1.0, 1.0, 1.0),
                        25.0,
                    )));
                changed = true;
            }
            if changed && let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                scene.dirty = true;
            }
        }
        ContextMenuAction::InspectorAddPointLight(entity) => {
            let mut changed = false;
            if world.get::<BevyTransform>(entity).is_none() {
                world.entity_mut(entity).insert(BevyTransform::default());
            }
            if world.get::<BevyLight>(entity).is_none() {
                world
                    .entity_mut(entity)
                    .insert(BevyWrapper(Light::point(glam::vec3(1.0, 1.0, 1.0), 10.0)));
                changed = true;
            }
            if changed && let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                scene.dirty = true;
            }
        }
        ContextMenuAction::InspectorAddSpotLight(entity) => {
            let mut changed = false;
            if world.get::<BevyTransform>(entity).is_none() {
                world.entity_mut(entity).insert(BevyTransform::default());
            }
            if world.get::<BevyLight>(entity).is_none() {
                world.entity_mut(entity).insert(BevyWrapper(Light::spot(
                    glam::vec3(1.0, 1.0, 1.0),
                    10.0,
                    45.0_f32.to_radians(),
                )));
                changed = true;
            }
            if changed && let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                scene.dirty = true;
            }
        }
        ContextMenuAction::InspectorAddMeshRenderer(entity) => {
            let mut changed = false;
            if world.get::<BevyMeshRenderer>(entity).is_none() {
                if world.get::<BevyTransform>(entity).is_none() {
                    world.entity_mut(entity).insert(BevyTransform::default());
                }
                world.entity_mut(entity).insert(BevyWrapper(
                    helmer::provided::components::MeshRenderer::new(0, 0, true, true),
                ));
                changed = true;
            }
            if changed && let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                scene.dirty = true;
            }
        }
        ContextMenuAction::ViewportSetResolution { tab_id, preset } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                viewport_states.state_for_mut(tab_id).resolution = preset;
            }
        }
        ContextMenuAction::ViewportSetGraphTemplate { tab_id, template } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                viewport_states.state_for_mut(tab_id).graph_template = template;
            }
        }
        ContextMenuAction::ViewportToggleExecuteScriptsInEditMode { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.execute_scripts_in_edit_mode = !state.execute_scripts_in_edit_mode;
            }
        }
        ContextMenuAction::ViewportToggleGizmosInPlay { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.gizmos_in_play = !state.gizmos_in_play;
            }
        }
        ContextMenuAction::ViewportToggleShowCameraGizmos { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.show_camera_gizmos = !state.show_camera_gizmos;
            }
        }
        ContextMenuAction::ViewportToggleShowDirectionalLightGizmos { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.show_directional_light_gizmos = !state.show_directional_light_gizmos;
            }
        }
        ContextMenuAction::ViewportToggleShowPointLightGizmos { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.show_point_light_gizmos = !state.show_point_light_gizmos;
            }
        }
        ContextMenuAction::ViewportToggleShowSpotLightGizmos { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.show_spot_light_gizmos = !state.show_spot_light_gizmos;
            }
        }
        ContextMenuAction::ViewportToggleShowSplinePaths { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.show_spline_paths = !state.show_spline_paths;
            }
        }
        ContextMenuAction::ViewportToggleShowSplinePoints { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.show_spline_points = !state.show_spline_points;
            }
        }
        ContextMenuAction::ViewportToggleShowNavigationGizmo { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.show_navigation_gizmo = !state.show_navigation_gizmo;
            }
        }
        ContextMenuAction::ViewportSetGizmoMode { tab_id, mode } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                viewport_states.state_for_mut(tab_id).gizmo_mode = mode;
            }
        }
        ContextMenuAction::ViewportToggleGizmoSnapEnabled => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.enabled = !settings.enabled;
                settings.sanitize();
            }
        }
        ContextMenuAction::ViewportToggleGizmoSnapCtrlToggles => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.ctrl_toggles = !settings.ctrl_toggles;
                settings.sanitize();
            }
        }
        ContextMenuAction::ViewportToggleGizmoSnapShiftFine => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.shift_fine = !settings.shift_fine;
                settings.sanitize();
            }
        }
        ContextMenuAction::ViewportToggleGizmoSnapAltCoarse => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.alt_coarse = !settings.alt_coarse;
                settings.sanitize();
            }
        }
        ContextMenuAction::ViewportSetFreecamSpeed { tab_id, speed } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                apply_freecam_speed_profile(state, speed);
            }
        }
        ContextMenuAction::ViewportResetFreecam { tab_id } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_speed = RetainedViewportFreecamSpeed::Normal;
                state.freecam_sensitivity = FREECAM_SENSITIVITY_DEFAULT;
                state.freecam_smoothing = FREECAM_SMOOTHING_DEFAULT;
                state.freecam_move_accel = FREECAM_MOVE_ACCEL_DEFAULT;
                state.freecam_move_decel = FREECAM_MOVE_DECEL_DEFAULT;
                state.freecam_speed_step = FREECAM_SPEED_STEP_DEFAULT;
                state.freecam_speed_min = FREECAM_SPEED_MIN_DEFAULT;
                state.freecam_speed_max = FREECAM_SPEED_MAX_DEFAULT;
                state.freecam_boost_multiplier = FREECAM_BOOST_MULTIPLIER_DEFAULT;
                state.orbit_distance = FREECAM_ORBIT_DISTANCE_DEFAULT;
                state.freecam_orbit_pan_sensitivity = FREECAM_ORBIT_PAN_SENSITIVITY_DEFAULT;
                state.freecam_pan_sensitivity = FREECAM_PAN_SENSITIVITY_DEFAULT;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustFreecamSensitivity { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_sensitivity += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustFreecamSmoothing { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_smoothing += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustFreecamMoveAccel { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_move_accel += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustFreecamMoveDecel { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_move_decel += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustFreecamSpeedStep { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_speed_step += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustFreecamSpeedMin { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_speed_min += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustFreecamSpeedMax { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_speed_max += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustFreecamBoostMultiplier { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_boost_multiplier += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportSetOrbitSelected {
            tab_id,
            orbit_selected_entity,
        } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                viewport_states.state_for_mut(tab_id).orbit_selected_entity = orbit_selected_entity;
            }
        }
        ContextMenuAction::ViewportAdjustOrbitDistance { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.orbit_distance += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustOrbitPanSensitivity { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_orbit_pan_sensitivity += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportAdjustPanSensitivity { tab_id, delta } => {
            if let Some(mut viewport_states) =
                world.get_resource_mut::<EditorRetainedViewportStates>()
            {
                let state = viewport_states.state_for_mut(tab_id);
                state.freecam_pan_sensitivity += delta;
                state.sanitize();
            }
        }
        ContextMenuAction::ViewportResetView { tab_id: _ } => {
            reset_active_viewport_camera(world);
        }
        ContextMenuAction::ViewportFrameSelection { tab_id: _ } => {
            frame_selection_in_active_viewport(world);
        }
        ContextMenuAction::LayoutTileColumns(columns) => {
            world.resource_scope::<EditorRetainedLayoutState, _>(|_world, mut layout_state| {
                let bounds = layout_state.workspace_bounds;
                layout_state.windows.tile_columns(bounds, columns.max(1));
            });
        }
        ContextMenuAction::LayoutTileRows(rows) => {
            world.resource_scope::<EditorRetainedLayoutState, _>(|_world, mut layout_state| {
                let bounds = layout_state.workspace_bounds;
                layout_state.windows.tile_rows(bounds, rows.max(1));
            });
        }
        ContextMenuAction::LayoutCascade => {
            world.resource_scope::<EditorRetainedLayoutState, _>(|_world, mut layout_state| {
                let bounds = layout_state.workspace_bounds;
                layout_state.windows.cascade(bounds, Vec2::new(24.0, 24.0));
            });
        }
        ContextMenuAction::LayoutNormalize => {
            world.resource_scope::<EditorRetainedLayoutState, _>(|_world, mut layout_state| {
                let bounds = layout_state.workspace_bounds;
                layout_state.windows.normalize_to_bounds(bounds);
            });
        }
        ContextMenuAction::LayoutActivate(name) => {
            world.resource_scope::<EditorRetainedLayoutState, _>(|world, mut layout_state| {
                world.resource_scope::<EditorRetainedLayoutCatalog, _>(
                    |_world, mut layout_catalog| {
                        if layout_catalog.active.as_deref() == Some(name.as_str()) {
                            layout_catalog.active = None;
                            return;
                        }
                        if let Some(snapshot) = layout_catalog.layouts.get(&name).cloned() {
                            layout_state.windows.restore_snapshot(&snapshot.windows);
                            let bounds = layout_state.workspace_bounds;
                            layout_state.windows.normalize_to_bounds(bounds);
                            layout_catalog.active = Some(name);
                        }
                    },
                );
            });
        }
        ContextMenuAction::LayoutDeactivate => {
            if let Some(mut layout_catalog) =
                world.get_resource_mut::<EditorRetainedLayoutCatalog>()
            {
                layout_catalog.active = None;
            }
        }
        ContextMenuAction::LayoutSaveActive => {
            let snapshot = world
                .get_resource::<EditorRetainedLayoutState>()
                .map(|layout_state| layout_state.windows.snapshot());
            if let Some(snapshot) = snapshot
                && let Some(mut layout_catalog) =
                    world.get_resource_mut::<EditorRetainedLayoutCatalog>()
            {
                layout_catalog.save_active(snapshot);
            }
        }
        ContextMenuAction::LayoutSaveAsNew => {
            let snapshot = world
                .get_resource::<EditorRetainedLayoutState>()
                .map(|layout_state| layout_state.windows.snapshot());
            if let Some(snapshot) = snapshot
                && let Some(mut layout_catalog) =
                    world.get_resource_mut::<EditorRetainedLayoutCatalog>()
            {
                layout_catalog.save_as_new(snapshot);
            }
        }
        ContextMenuAction::LayoutDeleteActive => {
            if let Some(mut layout_catalog) =
                world.get_resource_mut::<EditorRetainedLayoutCatalog>()
                && let Some(active) = layout_catalog.active.clone()
            {
                if active.eq_ignore_ascii_case("default") {
                    return;
                }
                layout_catalog.layouts.remove(&active);
                layout_catalog.active = None;
            }
        }
        ContextMenuAction::LayoutToggleMove => {
            if let Some(mut layout_catalog) =
                world.get_resource_mut::<EditorRetainedLayoutCatalog>()
            {
                layout_catalog.allow_layout_move = !layout_catalog.allow_layout_move;
            }
        }
        ContextMenuAction::LayoutToggleResize => {
            if let Some(mut layout_catalog) =
                world.get_resource_mut::<EditorRetainedLayoutCatalog>()
            {
                layout_catalog.allow_layout_resize = !layout_catalog.allow_layout_resize;
            }
        }
        ContextMenuAction::LayoutToggleLiveReflow => {
            if let Some(mut layout_catalog) =
                world.get_resource_mut::<EditorRetainedLayoutCatalog>()
            {
                layout_catalog.live_reflow = !layout_catalog.live_reflow;
            }
        }
    }
}

fn apply_freecam_speed_profile(
    state: &mut crate::retained::state::EditorRetainedViewportState,
    speed: RetainedViewportFreecamSpeed,
) {
    state.freecam_speed = speed;
    match speed {
        RetainedViewportFreecamSpeed::Slow => {
            state.freecam_speed_step = 1.0;
            state.freecam_speed_min = 0.25;
            state.freecam_speed_max = 20.0;
            state.freecam_boost_multiplier = 2.0;
        }
        RetainedViewportFreecamSpeed::Normal => {
            state.freecam_speed_step = FREECAM_SPEED_STEP_DEFAULT;
            state.freecam_speed_min = FREECAM_SPEED_MIN_DEFAULT;
            state.freecam_speed_max = FREECAM_SPEED_MAX_DEFAULT;
            state.freecam_boost_multiplier = FREECAM_BOOST_MULTIPLIER_DEFAULT;
        }
        RetainedViewportFreecamSpeed::Fast => {
            state.freecam_speed_step = 4.0;
            state.freecam_speed_min = 1.0;
            state.freecam_speed_max = 180.0;
            state.freecam_boost_multiplier = 3.0;
        }
    }
    state.sanitize();
}

fn checked_label(checked: bool, label: impl AsRef<str>) -> String {
    let prefix = if checked { "[x]" } else { "[ ]" };
    format!("{prefix} {}", label.as_ref())
}

fn drag_value_row(
    label: impl AsRef<str>,
    value_text: impl Into<String>,
    action: ContextMenuDragAction,
) -> ContextMenuDragValueRow {
    ContextMenuDragValueRow {
        label: label.as_ref().to_string(),
        value_text: value_text.into(),
        action,
    }
}

fn toolbar_layout_context_rows(world: &World) -> Vec<ContextMenuEntryRow> {
    let mut rows = Vec::new();
    if let Some(layout_catalog) = world.get_resource::<EditorRetainedLayoutCatalog>() {
        let mut names = layout_catalog.layouts.keys().cloned().collect::<Vec<_>>();
        names.sort_by_key(|name| name.to_ascii_lowercase());

        if !names.is_empty() {
            let layout_entries = names
                .into_iter()
                .map(|name| {
                    let selected = layout_catalog.active.as_deref() == Some(name.as_str());
                    (
                        checked_label(selected, &name),
                        ContextMenuAction::LayoutActivate(name),
                    )
                })
                .collect::<Vec<_>>();
            rows.push(ContextMenuEntryRow::Submenu(
                "Layouts".to_string(),
                layout_entries,
            ));
        }

        rows.push(ContextMenuEntryRow::Submenu(
            "Preferences".to_string(),
            vec![
                (
                    checked_label(layout_catalog.allow_layout_move, "Allow Move"),
                    ContextMenuAction::LayoutToggleMove,
                ),
                (
                    checked_label(layout_catalog.allow_layout_resize, "Allow Resize"),
                    ContextMenuAction::LayoutToggleResize,
                ),
                (
                    checked_label(layout_catalog.live_reflow, "Live Reflow"),
                    ContextMenuAction::LayoutToggleLiveReflow,
                ),
            ],
        ));
        rows.push(ContextMenuEntryRow::Separator);
        rows.push(ContextMenuEntryRow::Submenu(
            "Manage".to_string(),
            vec![
                (
                    "Save Active Layout".to_string(),
                    ContextMenuAction::LayoutSaveActive,
                ),
                (
                    "Save As New Layout".to_string(),
                    ContextMenuAction::LayoutSaveAsNew,
                ),
                (
                    "Delete Active Layout".to_string(),
                    ContextMenuAction::LayoutDeleteActive,
                ),
                (
                    "Deactivate Layout".to_string(),
                    ContextMenuAction::LayoutDeactivate,
                ),
            ],
        ));
        rows.push(ContextMenuEntryRow::Separator);
    }
    rows.push(ContextMenuEntryRow::Submenu(
        "Arrange".to_string(),
        vec![
            (
                "Tile Windows (2 Columns)".to_string(),
                ContextMenuAction::LayoutTileColumns(2),
            ),
            (
                "Tile Windows (2 Rows)".to_string(),
                ContextMenuAction::LayoutTileRows(2),
            ),
            (
                "Cascade Windows".to_string(),
                ContextMenuAction::LayoutCascade,
            ),
            (
                "Normalize Bounds".to_string(),
                ContextMenuAction::LayoutNormalize,
            ),
        ],
    ));
    rows
}

fn dock_spawn_tab_entries(workspace_id: UiId) -> Vec<(String, ContextMenuAction)> {
    [
        EditorPaneKind::Toolbar,
        EditorPaneKind::Viewport,
        EditorPaneKind::PlayViewport,
        EditorPaneKind::Project,
        EditorPaneKind::Hierarchy,
        EditorPaneKind::Inspector,
        EditorPaneKind::ContentBrowser,
        EditorPaneKind::Console,
        EditorPaneKind::Timeline,
        EditorPaneKind::History,
        EditorPaneKind::AudioMixer,
        EditorPaneKind::Profiler,
    ]
    .into_iter()
    .map(|kind| {
        (
            kind.label().to_string(),
            ContextMenuAction::DockOpenTab { workspace_id, kind },
        )
    })
    .collect()
}

fn dock_spawn_window_entries(workspace_id: UiId) -> Vec<(String, ContextMenuAction)> {
    [
        EditorPaneKind::Toolbar,
        EditorPaneKind::Viewport,
        EditorPaneKind::PlayViewport,
        EditorPaneKind::Project,
        EditorPaneKind::Hierarchy,
        EditorPaneKind::Inspector,
        EditorPaneKind::ContentBrowser,
        EditorPaneKind::Console,
        EditorPaneKind::Timeline,
        EditorPaneKind::History,
        EditorPaneKind::AudioMixer,
        EditorPaneKind::Profiler,
    ]
    .into_iter()
    .map(|kind| {
        (
            kind.label().to_string(),
            ContextMenuAction::DockOpenTabInNewWindow { workspace_id, kind },
        )
    })
    .collect()
}

fn hierarchy_spawn_primary_context_entries() -> Vec<(String, ContextMenuAction)> {
    vec![
        (
            "Add Empty".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::Empty),
        ),
        (
            "Add Camera".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::Camera),
        ),
        (
            "Add Directional Light".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::DirectionalLight),
        ),
        (
            "Add Point Light".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::PointLight),
        ),
        (
            "Add Spot Light".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::SpotLight),
        ),
        (
            "Add Cube".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::Primitive(
                helmer_editor_runtime::editor_commands::PrimitiveKind::Cube,
            )),
        ),
        (
            "Add UV Sphere".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::Primitive(
                helmer_editor_runtime::editor_commands::PrimitiveKind::default_uv_sphere(),
            )),
        ),
        (
            "Add Icosphere".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::Primitive(
                helmer_editor_runtime::editor_commands::PrimitiveKind::default_icosphere(),
            )),
        ),
        (
            "Add Cylinder".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::Primitive(
                helmer_editor_runtime::editor_commands::PrimitiveKind::default_cylinder(),
            )),
        ),
        (
            "Add Capsule".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::Primitive(
                helmer_editor_runtime::editor_commands::PrimitiveKind::default_capsule(),
            )),
        ),
        (
            "Add Plane".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::Primitive(
                helmer_editor_runtime::editor_commands::PrimitiveKind::Plane,
            )),
        ),
    ]
}

fn hierarchy_spawn_physics_context_entries() -> Vec<(String, ContextMenuAction)> {
    vec![
        (
            "Add Dynamic Body (Box)".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::DynamicBodyCuboid),
        ),
        (
            "Add Dynamic Body (Sphere)".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::DynamicBodySphere),
        ),
        (
            "Add Fixed Collider (Box)".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::FixedColliderCuboid),
        ),
        (
            "Add Fixed Collider (Sphere)".to_string(),
            ContextMenuAction::HierarchySpawn(SpawnKind::FixedColliderSphere),
        ),
    ]
}

fn hierarchy_spawn_provided_context_entries() -> Vec<(String, ContextMenuAction)> {
    vec![(
        "Add Freecam Camera".to_string(),
        ContextMenuAction::HierarchySpawn(SpawnKind::FreecamCamera),
    )]
}

fn content_entry_context_rows(
    path: &PathBuf,
    is_dir: bool,
    allow_paste: bool,
) -> Vec<ContextMenuEntryRow> {
    let mut entries = Vec::new();
    if is_dir {
        entries.push(ContextMenuEntryRow::Action(
            "Open Folder".to_string(),
            ContextMenuAction::ContentOpen {
                path: path.clone(),
                is_dir: true,
            },
        ));
    } else {
        if is_scene_file(path) {
            entries.push(ContextMenuEntryRow::Action(
                "Open Scene".to_string(),
                ContextMenuAction::ContentOpenScene(path.clone()),
            ));
        }
        if is_model_file(path) {
            entries.push(ContextMenuEntryRow::Action(
                "Add to Scene".to_string(),
                ContextMenuAction::ContentAddModelToScene(path.clone()),
            ));
        }
        if is_material_file(path) {
            entries.push(ContextMenuEntryRow::Action(
                "Open Material".to_string(),
                ContextMenuAction::ContentOpen {
                    path: path.clone(),
                    is_dir: false,
                },
            ));
        } else if is_animation_file(path) {
            entries.push(ContextMenuEntryRow::Action(
                "Open Animation".to_string(),
                ContextMenuAction::ContentOpen {
                    path: path.clone(),
                    is_dir: false,
                },
            ));
        } else if is_script_file(path) {
            entries.push(ContextMenuEntryRow::Action(
                "Open Script".to_string(),
                ContextMenuAction::ContentOpen {
                    path: path.clone(),
                    is_dir: false,
                },
            ));
        }
        entries.push(ContextMenuEntryRow::Action(
            "Open".to_string(),
            ContextMenuAction::ContentOpen {
                path: path.clone(),
                is_dir: false,
            },
        ));
    }

    entries.push(ContextMenuEntryRow::Action(
        "Open in File Browser".to_string(),
        ContextMenuAction::ContentOpenInFileBrowser(path.clone()),
    ));

    if is_dir {
        entries.push(ContextMenuEntryRow::Submenu(
            "Create".to_string(),
            content_create_submenu_entries(path),
        ));
    }
    entries.push(ContextMenuEntryRow::Separator);
    entries.push(ContextMenuEntryRow::Action(
        "Copy".to_string(),
        ContextMenuAction::ContentCopyPaths {
            anchor: Some(path.clone()),
            mode: AssetClipboardMode::Copy,
        },
    ));
    entries.push(ContextMenuEntryRow::Action(
        "Cut".to_string(),
        ContextMenuAction::ContentCopyPaths {
            anchor: Some(path.clone()),
            mode: AssetClipboardMode::Cut,
        },
    ));
    if allow_paste {
        entries.push(ContextMenuEntryRow::Action(
            "Paste".to_string(),
            ContextMenuAction::ContentPaste {
                target: if is_dir {
                    Some(path.clone())
                } else {
                    path.parent().map(Path::to_path_buf)
                },
            },
        ));
    }
    entries.push(ContextMenuEntryRow::Separator);
    entries.push(ContextMenuEntryRow::Action(
        "Duplicate".to_string(),
        ContextMenuAction::ContentDuplicatePaths {
            anchor: Some(path.clone()),
        },
    ));
    entries.push(ContextMenuEntryRow::Action(
        "Delete".to_string(),
        ContextMenuAction::ContentDeletePaths {
            anchor: Some(path.clone()),
        },
    ));
    if is_dir {
        entries.push(ContextMenuEntryRow::Separator);
        entries.push(ContextMenuEntryRow::Action(
            "Set As Directory".to_string(),
            ContextMenuAction::ContentSelectFolder(path.clone()),
        ));
    }
    entries.push(ContextMenuEntryRow::Separator);
    entries.push(ContextMenuEntryRow::Action(
        "Refresh Browser".to_string(),
        ContextMenuAction::ContentRefresh,
    ));
    entries
}

fn content_surface_context_rows(
    current_dir: Option<&Path>,
    allow_paste: bool,
) -> Vec<ContextMenuEntryRow> {
    let mut entries = Vec::new();
    if let Some(current_dir) = current_dir {
        entries.push(ContextMenuEntryRow::Action(
            "Open in File Browser".to_string(),
            ContextMenuAction::ContentOpenInFileBrowser(current_dir.to_path_buf()),
        ));
        entries.push(ContextMenuEntryRow::Separator);
        entries.push(ContextMenuEntryRow::Action(
            "Copy Selected".to_string(),
            ContextMenuAction::ContentCopyPaths {
                anchor: None,
                mode: AssetClipboardMode::Copy,
            },
        ));
        entries.push(ContextMenuEntryRow::Action(
            "Cut Selected".to_string(),
            ContextMenuAction::ContentCopyPaths {
                anchor: None,
                mode: AssetClipboardMode::Cut,
            },
        ));
        if allow_paste {
            entries.push(ContextMenuEntryRow::Action(
                "Paste".to_string(),
                ContextMenuAction::ContentPaste {
                    target: Some(current_dir.to_path_buf()),
                },
            ));
        }
        entries.push(ContextMenuEntryRow::Separator);
        entries.push(ContextMenuEntryRow::Action(
            "Duplicate Selected".to_string(),
            ContextMenuAction::ContentDuplicatePaths { anchor: None },
        ));
        entries.push(ContextMenuEntryRow::Action(
            "Delete Selected".to_string(),
            ContextMenuAction::ContentDeletePaths { anchor: None },
        ));
        entries.push(ContextMenuEntryRow::Separator);
        entries.push(ContextMenuEntryRow::Submenu(
            "Create".to_string(),
            content_create_submenu_entries(current_dir),
        ));
        entries.push(ContextMenuEntryRow::Separator);
    }
    entries.push(ContextMenuEntryRow::Action(
        "Refresh Browser".to_string(),
        ContextMenuAction::ContentRefresh,
    ));
    entries
}

fn content_create_submenu_entries(directory: &Path) -> Vec<(String, ContextMenuAction)> {
    content_create_context_entries(directory)
        .into_iter()
        .map(|(label, action)| (label.to_string(), action))
        .collect()
}

fn content_create_context_entries(directory: &Path) -> Vec<(&'static str, ContextMenuAction)> {
    vec![
        (
            "New Folder",
            ContextMenuAction::ContentCreateAsset {
                directory: directory.to_path_buf(),
                name: "new_folder",
                kind: AssetCreateKind::Folder,
            },
        ),
        (
            "New Scene",
            ContextMenuAction::ContentCreateAsset {
                directory: directory.to_path_buf(),
                name: "new_scene",
                kind: AssetCreateKind::Scene,
            },
        ),
        (
            "New Material",
            ContextMenuAction::ContentCreateAsset {
                directory: directory.to_path_buf(),
                name: "new_material",
                kind: AssetCreateKind::Material,
            },
        ),
        (
            "New Script",
            ContextMenuAction::ContentCreateAsset {
                directory: directory.to_path_buf(),
                name: "new_script",
                kind: AssetCreateKind::Script,
            },
        ),
        (
            "New Rust Script",
            ContextMenuAction::ContentCreateAsset {
                directory: directory.to_path_buf(),
                name: "new_rust_script",
                kind: AssetCreateKind::RustScript,
            },
        ),
        (
            "New Visual Script",
            ContextMenuAction::ContentCreateAsset {
                directory: directory.to_path_buf(),
                name: "new_visual_script",
                kind: AssetCreateKind::VisualScript,
            },
        ),
        (
            "New Visual Script (3P)",
            ContextMenuAction::ContentCreateAsset {
                directory: directory.to_path_buf(),
                name: "new_visual_script_third_person",
                kind: AssetCreateKind::VisualScriptThirdPerson,
            },
        ),
        (
            "New Animation",
            ContextMenuAction::ContentCreateAsset {
                directory: directory.to_path_buf(),
                name: "new_animation",
                kind: AssetCreateKind::Animation,
            },
        ),
    ]
}

fn selected_assets_for_action(world: &World, anchor: Option<&Path>) -> Vec<PathBuf> {
    let Some(browser) = world.get_resource::<AssetBrowserState>() else {
        return Vec::new();
    };

    let mut selected = if let Some(anchor) = anchor {
        if browser.selected_paths.contains(anchor) && !browser.selected_paths.is_empty() {
            browser.selected_paths.iter().cloned().collect::<Vec<_>>()
        } else {
            vec![anchor.to_path_buf()]
        }
    } else if !browser.selected_paths.is_empty() {
        browser.selected_paths.iter().cloned().collect::<Vec<_>>()
    } else {
        browser.selected.clone().into_iter().collect::<Vec<_>>()
    };

    selected.sort();
    selected.dedup();
    selected
}

fn content_current_directory(world: &World) -> Option<PathBuf> {
    let browser = world.get_resource::<AssetBrowserState>()?;
    browser.current_dir.clone().or_else(|| browser.root.clone())
}

fn set_asset_clipboard(world: &mut World, anchor: Option<&Path>, mode: AssetClipboardMode) {
    let paths = selected_assets_for_action(world, anchor);
    if paths.is_empty() {
        set_asset_status(world, "No assets selected");
        return;
    }

    if let Some(mut clipboard) = world.get_resource_mut::<EditorAssetClipboardState>() {
        clipboard.mode = mode;
        clipboard.paths = paths.clone();
    }

    let verb = if mode == AssetClipboardMode::Cut {
        "Cut"
    } else {
        "Copied"
    };
    set_asset_status(world, format!("{verb} {} item(s)", paths.len()));
}

fn resolve_content_paste_target(world: &World, target: Option<&Path>) -> Option<PathBuf> {
    if let Some(target) = target {
        if target.is_dir() {
            return Some(target.to_path_buf());
        }
        if target.is_file() {
            return target.parent().map(Path::to_path_buf);
        }
        if target.extension().is_none() {
            return Some(target.to_path_buf());
        }
        return target.parent().map(Path::to_path_buf);
    }

    content_current_directory(world)
}

fn paste_assets_to_directory(world: &mut World, destination: Option<&Path>) -> bool {
    let Some(target_dir) = destination
        .map(Path::to_path_buf)
        .or_else(|| content_current_directory(world))
    else {
        set_asset_status(world, "Paste target directory is not available");
        return false;
    };
    if !target_dir.exists() || !target_dir.is_dir() {
        set_asset_status(world, "Paste target directory does not exist");
        return false;
    }

    let (mode, paths) = world
        .get_resource::<EditorAssetClipboardState>()
        .map(|clipboard| (clipboard.mode, clipboard.paths.clone()))
        .unwrap_or((AssetClipboardMode::Copy, Vec::new()));
    if paths.is_empty() {
        set_asset_status(world, "Clipboard is empty");
        return false;
    }

    let mut pasted = Vec::new();
    let mut failed = 0usize;
    let mut moved_sources = Vec::new();
    for source in &paths {
        let result = match mode {
            AssetClipboardMode::Copy => copy_asset_to_directory(source, &target_dir),
            AssetClipboardMode::Cut => move_asset_to_directory(world, source, &target_dir),
        };
        match result {
            Ok(path) => {
                if mode == AssetClipboardMode::Cut {
                    moved_sources.push(source.clone());
                }
                pasted.push(path);
            }
            Err(_) => failed = failed.saturating_add(1),
        }
    }

    if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
        browser.refresh_requested = true;
    }

    if mode == AssetClipboardMode::Cut
        && let Some(mut clipboard) = world.get_resource_mut::<EditorAssetClipboardState>()
    {
        clipboard
            .paths
            .retain(|path| !moved_sources.iter().any(|source| source == path));
    }

    if pasted.is_empty() {
        set_asset_status(world, "Paste failed");
        return false;
    }

    if failed == 0 {
        set_asset_status(world, format!("Pasted {} item(s)", pasted.len()));
    } else {
        set_asset_status(
            world,
            format!("Pasted {} item(s), {} failed", pasted.len(), failed),
        );
    }
    true
}

fn copy_asset_to_directory(source: &Path, target_dir: &Path) -> Result<PathBuf, String> {
    if !source.exists() {
        return Err("Source asset does not exist".to_string());
    }
    fs::create_dir_all(target_dir).map_err(|err| format!("Copy failed: {err}"))?;
    let file_name = source
        .file_name()
        .ok_or_else(|| "Source asset has no file name".to_string())?;
    let target_path = unique_asset_path(&target_dir.join(file_name));
    if source.is_dir() {
        copy_dir_recursive(source, &target_path)?;
    } else {
        fs::copy(source, &target_path).map_err(|err| format!("Copy failed: {err}"))?;
    }
    Ok(target_path)
}

fn move_asset_to_directory(
    world: &World,
    source: &Path,
    target_dir: &Path,
) -> Result<PathBuf, String> {
    if !source.exists() {
        return Err("Source asset does not exist".to_string());
    }
    if let Some(browser) = world.get_resource::<AssetBrowserState>()
        && browser.root.as_deref() == Some(source)
    {
        return Err("Cannot move project assets root".to_string());
    }
    if source.is_dir() && target_dir.starts_with(source) {
        return Err("Cannot move a directory into itself".to_string());
    }
    fs::create_dir_all(target_dir).map_err(|err| format!("Move failed: {err}"))?;
    let file_name = source
        .file_name()
        .ok_or_else(|| "Source asset has no file name".to_string())?;
    let target_path = unique_asset_path(&target_dir.join(file_name));
    fs::rename(source, &target_path).map_err(|err| format!("Move failed: {err}"))?;
    Ok(target_path)
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<(), String> {
    for entry in WalkDir::new(source).min_depth(0).into_iter().flatten() {
        let relative = entry
            .path()
            .strip_prefix(source)
            .map_err(|_| "Copy failed".to_string())?;
        let target_path = target.join(relative);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&target_path).map_err(|err| format!("Copy failed: {err}"))?;
        } else {
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent).map_err(|err| format!("Copy failed: {err}"))?;
            }
            fs::copy(entry.path(), &target_path).map_err(|err| format!("Copy failed: {err}"))?;
        }
    }
    Ok(())
}

fn duplicate_assets(world: &mut World, paths: &[PathBuf]) {
    let mut success = 0usize;
    let mut failed = 0usize;
    for path in paths {
        if let Some(browser) = world.get_resource::<AssetBrowserState>()
            && browser.root.as_deref() == Some(path.as_path())
        {
            failed = failed.saturating_add(1);
            continue;
        }
        let Some(parent) = path.parent() else {
            failed = failed.saturating_add(1);
            continue;
        };
        if copy_asset_to_directory(path, parent).is_ok() {
            success = success.saturating_add(1);
        } else {
            failed = failed.saturating_add(1);
        }
    }

    if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
        browser.refresh_requested = true;
    }

    if success == 0 {
        set_asset_status(world, "Duplicate failed");
    } else if failed == 0 {
        set_asset_status(world, format!("Duplicated {success} item(s)"));
    } else {
        set_asset_status(
            world,
            format!("Duplicated {success} item(s), {failed} failed"),
        );
    }
}

fn delete_assets(world: &mut World, paths: &[PathBuf]) {
    let mut delete_paths = paths.to_vec();
    delete_paths.sort();
    delete_paths.dedup();
    delete_paths.sort_by_key(|path| std::cmp::Reverse(path.components().count()));

    let root = world
        .get_resource::<AssetBrowserState>()
        .and_then(|browser| browser.root.clone());
    let mut success = 0usize;
    let mut failed = 0usize;

    for path in &delete_paths {
        if root.as_deref() == Some(path.as_path()) {
            failed = failed.saturating_add(1);
            continue;
        }
        let result = if path.is_dir() {
            fs::remove_dir_all(path)
        } else {
            fs::remove_file(path)
        };
        if result.is_ok() {
            success = success.saturating_add(1);
        } else {
            failed = failed.saturating_add(1);
        }
    }

    if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
        browser.refresh_requested = true;
        browser
            .selected_paths
            .retain(|path| !delete_paths.contains(path));
        if browser
            .selected
            .as_ref()
            .is_some_and(|selected| delete_paths.contains(selected))
        {
            browser.selected = browser.current_dir.clone().or_else(|| browser.root.clone());
        }
    }

    if success == 0 {
        set_asset_status(world, "Delete failed");
    } else if failed == 0 {
        set_asset_status(world, format!("Deleted {success} item(s)"));
    } else {
        set_asset_status(world, format!("Deleted {success} item(s), {failed} failed"));
    }
}

fn unique_asset_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("asset");
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{ext}"))
        .unwrap_or_default();

    for index in 1..=999u32 {
        let candidate = parent.join(format!("{stem}_{index}{extension}"));
        if !candidate.exists() {
            return candidate;
        }
    }

    path.to_path_buf()
}

fn set_asset_status(world: &mut World, message: impl Into<String>) {
    let message = message.into();
    if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
        browser.status = Some(message.clone());
    }
    if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
        console.push(
            asset_status_level(&message),
            "editor.assets",
            message.clone(),
        );
    }
}

fn asset_status_level(message: &str) -> EditorConsoleLevel {
    let lower = message.to_ascii_lowercase();
    if lower.contains("failed") || lower.contains("cannot") || lower.contains("error") {
        EditorConsoleLevel::Error
    } else if lower.contains("warn") {
        EditorConsoleLevel::Warn
    } else {
        EditorConsoleLevel::Info
    }
}

fn open_in_file_browser(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        if path.is_dir() {
            Command::new("explorer")
                .arg(path)
                .spawn()
                .map(|_| ())
                .map_err(|err| err.to_string())
        } else {
            Command::new("explorer")
                .arg(format!("/select,{}", path.display()))
                .spawn()
                .map(|_| ())
                .map_err(|err| err.to_string())
        }
    }
    #[cfg(target_os = "macos")]
    {
        if path.is_dir() {
            Command::new("open")
                .arg(path)
                .spawn()
                .map(|_| ())
                .map_err(|err| err.to_string())
        } else {
            Command::new("open")
                .arg("-R")
                .arg(path)
                .spawn()
                .map(|_| ())
                .map_err(|err| err.to_string())
        }
    }
    #[cfg(target_os = "linux")]
    {
        let target = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(path)
        };
        Command::new("xdg-open")
            .arg(target)
            .spawn()
            .map(|_| ())
            .map_err(|err| err.to_string())
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        let _ = path;
        Err("Unsupported platform".to_string())
    }
}

fn is_scene_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".hscene.ron"))
}

fn is_animation_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".hanim.ron"))
}

fn is_model_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "glb" | "gltf"))
}

fn is_material_file(path: &Path) -> bool {
    if !path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ron"))
    {
        return false;
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    !(name.eq_ignore_ascii_case("helmer_project.ron")
        || name.ends_with(".hscene.ron")
        || name.ends_with(".hanim.ron"))
}

fn is_script_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "luau" | "lua" | "hvs" | "rs" | "wasm"
            )
        })
}

fn apply_content_drop_to_inspector(world: &mut World, path: &Path) {
    let Some(entity) = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0)
    else {
        return;
    };

    let mut message = None::<String>;
    if is_model_file(path) {
        if world.get::<BevyMeshRenderer>(entity).is_none() {
            world.entity_mut(entity).insert(BevyWrapper(
                helmer::provided::components::MeshRenderer::new(0, 0, true, true),
            ));
            if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                scene.dirty = true;
            }
            message = Some(format!(
                "Dropped {} on inspector: Mesh Renderer component added",
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("model")
            ));
        }
    } else if is_scene_file(path) {
        message = Some(format!(
            "Dropped {} on inspector (scene assignment in retained inspector is pending)",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("scene")
        ));
    }

    if let Some(message) = message
        && let Some(mut console) = world.get_resource_mut::<EditorConsoleState>()
    {
        console.push(EditorConsoleLevel::Info, "editor.inspector.drop", message);
    }
}

fn apply_context_menu_drag_delta(
    world: &mut World,
    drag_action: ContextMenuDragAction,
    delta_pixels: f32,
) {
    let action = match drag_action {
        ContextMenuDragAction::ViewportFreecamSensitivity { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustFreecamSensitivity {
                tab_id,
                delta: delta_pixels * 0.01,
            })
        }
        ContextMenuDragAction::ViewportFreecamSmoothing { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustFreecamSmoothing {
                tab_id,
                delta: delta_pixels * 0.0025,
            })
        }
        ContextMenuDragAction::ViewportFreecamMoveAccel { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustFreecamMoveAccel {
                tab_id,
                delta: delta_pixels * 0.5,
            })
        }
        ContextMenuDragAction::ViewportFreecamMoveDecel { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustFreecamMoveDecel {
                tab_id,
                delta: delta_pixels * 0.5,
            })
        }
        ContextMenuDragAction::ViewportFreecamSpeedStep { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustFreecamSpeedStep {
                tab_id,
                delta: delta_pixels * 0.05,
            })
        }
        ContextMenuDragAction::ViewportFreecamSpeedMin { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustFreecamSpeedMin {
                tab_id,
                delta: delta_pixels * 0.05,
            })
        }
        ContextMenuDragAction::ViewportFreecamSpeedMax { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustFreecamSpeedMax {
                tab_id,
                delta: delta_pixels * 0.5,
            })
        }
        ContextMenuDragAction::ViewportFreecamBoostMultiplier { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustFreecamBoostMultiplier {
                tab_id,
                delta: delta_pixels * 0.05,
            })
        }
        ContextMenuDragAction::ViewportOrbitDistance { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustOrbitDistance {
                tab_id,
                delta: delta_pixels * 0.1,
            })
        }
        ContextMenuDragAction::ViewportOrbitPanSensitivity { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustOrbitPanSensitivity {
                tab_id,
                delta: delta_pixels * 0.0001,
            })
        }
        ContextMenuDragAction::ViewportPanSensitivity { tab_id } => {
            Some(ContextMenuAction::ViewportAdjustPanSensitivity {
                tab_id,
                delta: delta_pixels * 0.00005,
            })
        }
        ContextMenuDragAction::GizmoSnapTranslateStep => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.translate_step += delta_pixels * 0.01;
                settings.sanitize();
            }
            None
        }
        ContextMenuDragAction::GizmoSnapRotateStepDegrees => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.rotate_step_degrees += delta_pixels * 0.5;
                settings.sanitize();
            }
            None
        }
        ContextMenuDragAction::GizmoSnapScaleStep => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.scale_step += delta_pixels * 0.01;
                settings.sanitize();
            }
            None
        }
        ContextMenuDragAction::GizmoSnapFineScale => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.fine_scale += delta_pixels * 0.01;
                settings.sanitize();
            }
            None
        }
        ContextMenuDragAction::GizmoSnapCoarseScale => {
            if let Some(mut settings) = world.get_resource_mut::<EditorRetainedGizmoSnapSettings>()
            {
                settings.coarse_scale += delta_pixels * 0.05;
                settings.sanitize();
            }
            None
        }
    };
    if let Some(action) = action {
        apply_context_menu_action(world, action);
    }
}

fn inspector_drag_starts_immediately(action: InspectorPaneDragAction) -> bool {
    matches!(
        action,
        InspectorPaneDragAction::LightColorSv { .. }
            | InspectorPaneDragAction::LightColorHue { .. }
    )
}

fn inspector_drag_uses_pointer_position(action: InspectorPaneDragAction) -> bool {
    matches!(
        action,
        InspectorPaneDragAction::LightColorSv { .. }
            | InspectorPaneDragAction::LightColorHue { .. }
    )
}

fn light_color_for_entity(world: &World, entity: Entity) -> Option<[f32; 3]> {
    world.get::<BevyLight>(entity).map(|light| {
        [
            light.0.color.x.clamp(0.0, 1.0),
            light.0.color.y.clamp(0.0, 1.0),
            light.0.color.z.clamp(0.0, 1.0),
        ]
    })
}

fn light_picker_hue_from_world(world: &World, entity: Entity, fallback: f32) -> f32 {
    let Some(color) = light_color_for_entity(world, entity) else {
        return fallback.rem_euclid(1.0);
    };
    let hsv = rgb_to_hsv(color);
    if hsv[1] > 0.0001 {
        hsv[0]
    } else {
        fallback.rem_euclid(1.0)
    }
}

fn sync_inspector_light_color_picker(
    world: &World,
    picker_state: &mut Option<InspectorLightColorPickerState>,
) {
    let Some(mut picker) = *picker_state else {
        return;
    };
    let selected = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0);
    if selected != Some(picker.entity) {
        *picker_state = None;
        return;
    }
    if world.get::<BevyLight>(picker.entity).is_none() {
        *picker_state = None;
        return;
    }
    picker.hue = light_picker_hue_from_world(world, picker.entity, picker.hue);
    *picker_state = Some(picker);
}

fn apply_inspector_drag_delta(
    world: &mut World,
    drag_action: InspectorPaneDragAction,
    delta_pixels: f32,
    pointer_position: Option<Vec2>,
    picker_state: &mut Option<InspectorLightColorPickerState>,
) {
    let action = match drag_action {
        InspectorPaneDragAction::Transform {
            entity,
            field,
            sensitivity,
        } => InspectorPaneAction::AdjustTransform {
            entity,
            field,
            delta: delta_pixels * sensitivity,
        },
        InspectorPaneDragAction::Camera {
            entity,
            field,
            sensitivity,
        } => InspectorPaneAction::AdjustCamera {
            entity,
            field,
            delta: delta_pixels * sensitivity,
        },
        InspectorPaneDragAction::Light {
            entity,
            field,
            sensitivity,
        } => InspectorPaneAction::AdjustLight {
            entity,
            field,
            delta: delta_pixels * sensitivity,
        },
        InspectorPaneDragAction::LightColorSv { entity, surface_id } => {
            let Some(pointer) = pointer_position else {
                return;
            };
            let Some(rect) = world
                .get_resource::<UiRuntimeState>()
                .and_then(|state| state.runtime().layout_rect(surface_id))
            else {
                return;
            };
            if rect.width <= f32::EPSILON || rect.height <= f32::EPSILON {
                return;
            }
            let hue = picker_state
                .as_ref()
                .filter(|picker| picker.entity == entity)
                .map(|picker| picker.hue)
                .unwrap_or_else(|| light_picker_hue_from_world(world, entity, 0.0));
            let saturation = ((pointer.x - rect.x) / rect.width).clamp(0.0, 1.0);
            let value = (1.0 - (pointer.y - rect.y) / rect.height).clamp(0.0, 1.0);
            InspectorPaneAction::SetLightColor {
                entity,
                color: crate::retained::panes::hsv_to_rgb(hue, saturation, value),
            }
        }
        InspectorPaneDragAction::LightColorHue { entity, surface_id } => {
            let Some(pointer) = pointer_position else {
                return;
            };
            let Some(rect) = world
                .get_resource::<UiRuntimeState>()
                .and_then(|state| state.runtime().layout_rect(surface_id))
            else {
                return;
            };
            if rect.width <= f32::EPSILON {
                return;
            }
            let mut hue = ((pointer.x - rect.x) / rect.width).clamp(0.0, 1.0);
            if !hue.is_finite() {
                hue = 0.0;
            }
            if let Some(mut picker) = picker_state
                .as_ref()
                .filter(|picker| picker.entity == entity)
                .copied()
            {
                picker.hue = hue;
                *picker_state = Some(picker);
            } else {
                *picker_state = Some(InspectorLightColorPickerState { entity, hue });
            }
            let hsv = light_color_for_entity(world, entity)
                .map(rgb_to_hsv)
                .unwrap_or([hue, 0.0, 1.0]);
            InspectorPaneAction::SetLightColor {
                entity,
                color: crate::retained::panes::hsv_to_rgb(hue, hsv[1], hsv[2]),
            }
        }
    };
    apply_inspector_action(world, action);
    sync_inspector_light_color_picker(world, picker_state);
}

fn update_content_browser_tile_size_from_pointer(
    world: &mut World,
    hovered: Option<UiId>,
    pointer_position: Option<Vec2>,
) {
    let Some(hit_id) = hovered else {
        return;
    };
    let Some(pointer) = pointer_position else {
        return;
    };
    let Some(rect) = world
        .get_resource::<UiRuntimeState>()
        .and_then(|state| state.runtime().layout_rect(hit_id))
    else {
        return;
    };
    if rect.width <= f32::EPSILON {
        return;
    }

    let normalized = ((pointer.x - rect.x) / rect.width).clamp(0.0, 1.0);
    let tile_size = 64.0 + normalized * (220.0 - 64.0);
    if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
        browser.tile_size = tile_size;
    }
}

fn update_content_browser_grid_scroll_from_pointer(
    world: &mut World,
    hovered: Option<UiId>,
    pointer_position: Option<Vec2>,
    max_scroll: f32,
) {
    let Some(hit_id) = hovered else {
        return;
    };
    let Some(pointer) = pointer_position else {
        return;
    };
    let Some(rect) = world
        .get_resource::<UiRuntimeState>()
        .and_then(|state| state.runtime().layout_rect(hit_id))
    else {
        return;
    };
    if rect.height <= f32::EPSILON {
        return;
    }

    let clamped_max_scroll = max_scroll.max(0.0);
    let normalized = ((pointer.y - rect.y) / rect.height).clamp(0.0, 1.0);
    let scroll = normalized * clamped_max_scroll;
    if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
        browser.grid_scroll = scroll.clamp(0.0, clamped_max_scroll);
    }
}

fn update_console_scroll_from_pointer(
    world: &mut World,
    hovered: Option<UiId>,
    pointer_position: Option<Vec2>,
    max_scroll: f32,
) {
    let Some(hit_id) = hovered else {
        return;
    };
    let Some(pointer) = pointer_position else {
        return;
    };
    let Some(rect) = world
        .get_resource::<UiRuntimeState>()
        .and_then(|state| state.runtime().layout_rect(hit_id))
    else {
        return;
    };
    if rect.height <= f32::EPSILON {
        return;
    }

    let clamped_max_scroll = max_scroll.max(0.0);
    let normalized = ((pointer.y - rect.y) / rect.height).clamp(0.0, 1.0);
    let scroll = normalized * clamped_max_scroll;
    if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
        console.auto_scroll = false;
        console.scroll = scroll.clamp(0.0, clamped_max_scroll);
    }
}

fn active_scene_camera_entity(world: &mut World) -> Option<Entity> {
    let mut fallback = None;
    let mut query = world.query::<(Entity, Option<&BevyActiveCamera>, &BevyCamera)>();
    for (entity, active, _camera) in query.iter(world) {
        if fallback.is_none() {
            fallback = Some(entity);
        }
        if active.is_some() {
            return Some(entity);
        }
    }
    fallback
}

fn reset_active_viewport_camera(world: &mut World) {
    let Some(camera_entity) = active_scene_camera_entity(world) else {
        return;
    };
    if let Some(mut transform) = world.get_mut::<BevyTransform>(camera_entity) {
        transform.0.position = Vec3::new(0.0, 0.0, 0.0);
        transform.0.rotation = glam::Quat::IDENTITY;
        transform.0.scale = Vec3::ONE;
    }
}

fn frame_selection_in_active_viewport(world: &mut World) {
    let selected = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selection| selection.0)
        .and_then(|entity| {
            world
                .get::<BevyTransform>(entity)
                .map(|transform| transform.0)
        });
    let Some(selected_transform) = selected else {
        return;
    };
    let Some(camera_entity) = active_scene_camera_entity(world) else {
        return;
    };
    if let Some(mut camera_transform) = world.get_mut::<BevyTransform>(camera_entity) {
        camera_transform.0.position = selected_transform.position + Vec3::new(0.0, 2.5, 6.0);
    }
}

fn pane_window_ui_id(pane_window: &EditorPaneWindow) -> UiId {
    UiId::from_str(&format!("pane-window-{}", pane_window.id))
}

fn find_pane_tab_title(pane_windows: &[EditorPaneWindow], tab_id: UiId) -> Option<String> {
    for window in pane_windows {
        for area in &window.areas {
            if let Some(tab) = area.tabs.iter().find(|tab| tab.id == tab_id.0) {
                return Some(tab.title.clone());
            }
        }
    }
    None
}

fn default_window_visible_for_open_project(pane_window: &EditorPaneWindow) -> bool {
    pane_window.id.eq_ignore_ascii_case("toolbar")
        || pane_window.id.eq_ignore_ascii_case("viewport")
        || pane_window.id.eq_ignore_ascii_case("content browser")
        || pane_window.id.eq_ignore_ascii_case("hierarchy")
        || pane_window.id.eq_ignore_ascii_case("inspector")
}

fn centered_project_launcher_rect(viewport_size: Vec2, bounds: LayoutRect) -> LayoutRect {
    let viewport_width = viewport_size.x.max(1.0);
    let viewport_height = viewport_size.y.max(1.0);
    let width = if viewport_width <= 420.0 {
        viewport_width
    } else {
        (viewport_width * 0.60).clamp(420.0, viewport_width)
    };
    let height = if viewport_height <= 320.0 {
        viewport_height
    } else {
        (viewport_height * 0.70).clamp(320.0, viewport_height)
    };
    LayoutRect::new(
        ((viewport_width - width) * 0.5).max(0.0),
        ((viewport_height - height) * 0.5).max(0.0),
        width,
        height,
    )
    .clamp_inside(bounds)
}

fn seeded_window_frame(
    pane_window: &EditorPaneWindow,
    index: usize,
    viewport_size: Vec2,
    project_open: bool,
    bounds: LayoutRect,
) -> WindowFrame {
    let side_width = (viewport_size.x * 0.22).max(220.0);
    let top_height = (viewport_size.y * 0.085).max(40.0);
    let bottom_height = (viewport_size.y * 0.28).max(170.0);
    let center_width = (viewport_size.x - side_width * 2.0).max(320.0);
    let center_height = (viewport_size.y - top_height - bottom_height).max(220.0);
    let left_height = (viewport_size.y - bottom_height).max(240.0);
    let toolbar_x = ((viewport_size.x - center_width) * 0.5).max(0.0);

    let rect = if !project_open && pane_window.id.eq_ignore_ascii_case("project") {
        centered_project_launcher_rect(viewport_size, bounds)
    } else if pane_window.id.eq_ignore_ascii_case("toolbar") {
        LayoutRect::new(toolbar_x, 0.0, center_width, top_height)
    } else if pane_window.id.eq_ignore_ascii_case("viewport") {
        LayoutRect::new(side_width, top_height, center_width, center_height)
    } else if pane_window.id.eq_ignore_ascii_case("content browser") {
        LayoutRect::new(
            0.0,
            (viewport_size.y - bottom_height).max(0.0),
            (viewport_size.x - side_width).max(320.0),
            bottom_height,
        )
    } else if pane_window.id.eq_ignore_ascii_case("project") {
        LayoutRect::new(16.0, 16.0, 430.0, 420.0)
    } else if pane_window.id.eq_ignore_ascii_case("hierarchy") {
        LayoutRect::new(0.0, 0.0, side_width, left_height)
    } else if pane_window.id.eq_ignore_ascii_case("inspector") {
        LayoutRect::new(
            (viewport_size.x - side_width).max(0.0),
            0.0,
            side_width,
            viewport_size.y,
        )
    } else if pane_window.id.eq_ignore_ascii_case("timeline") {
        LayoutRect::new(
            side_width,
            (viewport_size.y - 236.0).max(16.0),
            (viewport_size.x - side_width * 2.0).max(320.0),
            220.0,
        )
    } else if pane_window.id.eq_ignore_ascii_case("history") {
        LayoutRect::new((viewport_size.x - 460.0).max(20.0), 20.0, 440.0, 320.0)
    } else if pane_window.id.eq_ignore_ascii_case("audio mixer") {
        LayoutRect::new(
            (viewport_size.x - 520.0).max(20.0),
            (viewport_size.y - 360.0).max(20.0),
            500.0,
            340.0,
        )
    } else if pane_window.id.eq_ignore_ascii_case("profiler") {
        LayoutRect::new(
            side_width,
            (viewport_size.y - 360.0).max(20.0),
            (viewport_size.x - side_width * 2.0).max(360.0),
            340.0,
        )
    } else if pane_window.id.eq_ignore_ascii_case("material") {
        LayoutRect::new((viewport_size.x - 460.0).max(20.0), 20.0, 440.0, 340.0)
    } else if pane_window.id.eq_ignore_ascii_case("visual script") {
        let width = (viewport_size.x - 120.0).max(500.0);
        let height = (viewport_size.y - 140.0).max(360.0);
        LayoutRect::new(
            ((viewport_size.x - width) * 0.5).max(0.0),
            ((viewport_size.y - height) * 0.5).max(0.0),
            width,
            height,
        )
    } else {
        LayoutRect::new(
            24.0 + (index as f32 * 24.0),
            24.0 + (index as f32 * 24.0),
            420.0,
            320.0,
        )
    };

    WindowFrame {
        rect: rect.clamp_inside(bounds),
        min_size: Vec2::new(220.0, 140.0),
        max_size: None,
        visible: if project_open {
            default_window_visible_for_open_project(pane_window)
        } else {
            pane_window.id.eq_ignore_ascii_case("project")
        },
        locked: false,
    }
}

fn top_visible_window_at_pointer(
    layout_state: &EditorRetainedLayoutState,
    pointer: Vec2,
) -> Option<UiId> {
    let ordered = layout_state.windows.ordered_visible_windows();
    for (window_id, frame) in ordered.into_iter().rev() {
        if point_in_layout_rect(frame.rect, pointer) {
            return Some(window_id);
        }
    }
    None
}

fn pane_kind_under_pointer(
    world: &World,
    pane_windows: &[EditorPaneWindow],
    pointer: Vec2,
    project_open: bool,
) -> Option<EditorPaneKind> {
    let layout_state = world.get_resource::<EditorRetainedLayoutState>()?;
    let window_id = top_visible_window_at_pointer(layout_state, pointer)?;
    let pane_window = pane_windows
        .iter()
        .find(|pane_window| pane_window_ui_id(pane_window) == window_id)?;

    if !project_open {
        return active_tab_for_window(pane_window).map(|tab| tab.kind);
    }

    let docking_state = world.get_resource::<EditorRetainedDockingState>()?;
    let frame = layout_state.windows.window(window_id)?;
    if let Some(workspace) = docking_state.workspace(window_id) {
        let content_rect = UiRect {
            x: frame.rect.x,
            y: frame.rect.y + WINDOW_TITLE_BAR_HEIGHT,
            width: frame.rect.width.max(1.0),
            height: (frame.rect.height - WINDOW_TITLE_BAR_HEIGHT).max(1.0),
        };
        let leaves = workspace.docking.layout(content_rect);
        if let Some(leaf) = leaves
            .iter()
            .find(|leaf| point_in_ui_rect(leaf.rect, pointer))
        {
            let tab_lookup = pane_tab_lookup(pane_window);
            if let Some(tab_id) = leaf.active.or_else(|| leaf.tabs.first().copied()) {
                if let Some(tab) = tab_lookup.get(&tab_id) {
                    return Some(tab.kind);
                }
            }
        }
    }

    active_tab_for_window(pane_window).map(|tab| tab.kind)
}

fn resolve_dock_drop_target(
    layout_state: &EditorRetainedLayoutState,
    docking_state: &EditorRetainedDockingState,
    pointer: Vec2,
) -> Option<DockDropTarget> {
    let ordered = layout_state.windows.ordered_visible_windows();
    for (window_id, frame) in ordered.into_iter().rev() {
        if !point_in_layout_rect(frame.rect, pointer) {
            continue;
        }

        let content_rect = UiRect {
            x: frame.rect.x,
            y: frame.rect.y + WINDOW_TITLE_BAR_HEIGHT,
            width: frame.rect.width.max(1.0),
            height: (frame.rect.height - WINDOW_TITLE_BAR_HEIGHT).max(1.0),
        };
        let Some(workspace) = docking_state.workspace(window_id) else {
            return Some(DockDropTarget {
                window_id,
                leaf_index: 0,
                leaf_focus_tab: None,
                zone: DockDropZone::Center,
            });
        };

        let leaves = workspace.docking.layout(content_rect);
        if leaves.is_empty() {
            return Some(DockDropTarget {
                window_id,
                leaf_index: 0,
                leaf_focus_tab: None,
                zone: DockDropZone::Center,
            });
        }

        let mut best: Option<(f32, DockDropTarget)> = None;
        for (leaf_index, leaf) in leaves.iter().enumerate() {
            let leaf_rect = leaf.rect;
            let clamped_x = pointer.x.clamp(leaf_rect.x, leaf_rect.x + leaf_rect.width);
            let clamped_y = pointer.y.clamp(leaf_rect.y, leaf_rect.y + leaf_rect.height);
            let mut distance_sq = (pointer.x - clamped_x).powi(2) + (pointer.y - clamped_y).powi(2);
            if point_in_ui_rect(leaf_rect, pointer) {
                distance_sq -= 1_000_000.0;
            }

            let target = DockDropTarget {
                window_id,
                leaf_index,
                leaf_focus_tab: leaf.active.or_else(|| leaf.tabs.first().copied()),
                zone: dock_drop_zone_for_pointer(leaf_rect, pointer),
            };
            match best {
                Some((best_distance, _)) if best_distance <= distance_sq => {}
                _ => {
                    best = Some((distance_sq, target));
                }
            }
        }

        if let Some((_, target)) = best {
            return Some(target);
        }

        return Some(DockDropTarget {
            window_id,
            leaf_index: 0,
            leaf_focus_tab: None,
            zone: DockDropZone::Center,
        });
    }
    None
}

fn apply_dock_tab_drop_to_workspace(
    workspace: &mut EditorDockWorkspace,
    target: DockDropTarget,
    tab: DockTab,
) {
    if let Some(focus_tab) = target.leaf_focus_tab {
        let _ = workspace.docking.activate_tab(focus_tab);
    }

    match target.zone {
        DockDropZone::Center => workspace.docking.add_tab_to_focused(tab),
        DockDropZone::Left => {
            workspace
                .docking
                .split_focused(DockAxis::Vertical, 0.5, tab, false);
        }
        DockDropZone::Right => {
            workspace
                .docking
                .split_focused(DockAxis::Vertical, 0.5, tab, true);
        }
        DockDropZone::Top => {
            workspace
                .docking
                .split_focused(DockAxis::Horizontal, 0.5, tab, false);
        }
        DockDropZone::Bottom => {
            workspace
                .docking
                .split_focused(DockAxis::Horizontal, 0.5, tab, true);
        }
    }
}

fn dock_workspace_tab_count(workspace: &EditorDockWorkspace) -> usize {
    workspace
        .docking
        .layout(UiRect {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        })
        .into_iter()
        .map(|leaf| leaf.tabs.len())
        .sum()
}

fn dock_drop_zone_for_pointer(area: UiRect, pointer: Vec2) -> DockDropZone {
    let (left, right, top, bottom, center) = dock_drop_zone_rects(area);
    let mut best = (DockDropZone::Center, f32::INFINITY);
    for (zone, rect) in [
        (DockDropZone::Left, left),
        (DockDropZone::Right, right),
        (DockDropZone::Top, top),
        (DockDropZone::Bottom, bottom),
        (DockDropZone::Center, center),
    ] {
        if !point_in_ui_rect(rect, pointer) {
            continue;
        }
        let center = Vec2::new(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
        let distance_sq = (pointer - center).length_squared();
        if distance_sq < best.1 {
            best = (zone, distance_sq);
        }
    }
    best.0
}

fn dock_drop_zone_rect(area: UiRect, zone: DockDropZone) -> UiRect {
    let (left, right, top, bottom, center) = dock_drop_zone_rects(area);
    match zone {
        DockDropZone::Center => center,
        DockDropZone::Left => left,
        DockDropZone::Right => right,
        DockDropZone::Top => top,
        DockDropZone::Bottom => bottom,
    }
}

fn dock_drop_zone_rects(area: UiRect) -> (UiRect, UiRect, UiRect, UiRect, UiRect) {
    let inset_amount = 6.0;
    let inset_width = (area.width - inset_amount * 2.0).max(1.0);
    let inset_height = (area.height - inset_amount * 2.0).max(1.0);
    let inset = UiRect {
        x: area.x + inset_amount.min(area.width * 0.45),
        y: area.y + inset_amount.min(area.height * 0.45),
        width: inset_width,
        height: inset_height,
    };
    let edge_x = ((inset.width * 0.20).clamp(16.0, 64.0)).min(inset.width * 0.40);
    let edge_y = ((inset.height * 0.20).clamp(16.0, 64.0)).min(inset.height * 0.40);

    let left = UiRect {
        x: inset.x,
        y: inset.y,
        width: edge_x.max(1.0),
        height: inset.height.max(1.0),
    };
    let right = UiRect {
        x: (inset.x + inset.width - edge_x).max(inset.x),
        y: inset.y,
        width: edge_x.max(1.0),
        height: inset.height.max(1.0),
    };
    let top = UiRect {
        x: inset.x + edge_x,
        y: inset.y,
        width: (inset.width - edge_x * 2.0).max(1.0),
        height: edge_y.max(1.0),
    };
    let bottom = UiRect {
        x: inset.x + edge_x,
        y: (inset.y + inset.height - edge_y).max(inset.y),
        width: (inset.width - edge_x * 2.0).max(1.0),
        height: edge_y.max(1.0),
    };
    let center = UiRect {
        x: inset.x + edge_x,
        y: inset.y + edge_y,
        width: (inset.width - edge_x * 2.0).max(1.0),
        height: (inset.height - edge_y * 2.0).max(1.0),
    };
    (left, right, top, bottom, center)
}

fn point_in_ui_rect(rect: UiRect, point: Vec2) -> bool {
    point.x >= rect.x
        && point.x <= rect.x + rect.width
        && point.y >= rect.y
        && point.y <= rect.y + rect.height
}

fn retained_preview_camera_aspect_ratio(world: &World) -> f32 {
    world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0)
        .filter(|entity| {
            world.get::<BevyCamera>(*entity).is_some()
                && world.get::<BevyTransform>(*entity).is_some()
                && world.get::<RetainedEditorViewportCamera>(*entity).is_none()
        })
        .and_then(|entity| {
            world
                .get::<BevyCamera>(entity)
                .map(|camera| camera.0.aspect_ratio)
        })
        .filter(|aspect| aspect.is_finite() && *aspect > 0.01)
        .unwrap_or(16.0 / 9.0)
}

fn retained_preview_rect_from_state(
    scene_rect: UiRect,
    position_norm: [f32; 2],
    width_norm: f32,
    aspect_ratio: f32,
) -> UiRect {
    let aspect = if aspect_ratio.is_finite() && aspect_ratio > 0.01 {
        aspect_ratio
    } else {
        16.0 / 9.0
    };
    let min_preview_w = scene_rect.width.min(120.0).max(72.0);
    let mut preview_w = (scene_rect.width * width_norm).max(min_preview_w);
    let mut preview_h = (preview_w / aspect).max(48.0);
    let max_preview_h = scene_rect.height.max(48.0);
    if preview_h > max_preview_h {
        preview_h = max_preview_h;
        preview_w = (preview_h * aspect).max(min_preview_w);
    }
    preview_w = preview_w.min(scene_rect.width.max(1.0));
    preview_h = preview_h.min(scene_rect.height.max(1.0));

    let max_offset_x = (scene_rect.width - preview_w).max(0.0);
    let max_offset_y = (scene_rect.height - preview_h).max(0.0);
    let mut x = scene_rect.x + position_norm[0].clamp(0.0, 1.0) * max_offset_x;
    let mut y = scene_rect.y + position_norm[1].clamp(0.0, 1.0) * max_offset_y;
    if !x.is_finite() || !y.is_finite() {
        x = scene_rect.x;
        y = scene_rect.y;
    }
    x = x.clamp(scene_rect.x, scene_rect.right() - preview_w);
    y = y.clamp(scene_rect.y, scene_rect.bottom() - preview_h);
    UiRect {
        x,
        y,
        width: preview_w,
        height: preview_h,
    }
}

fn pick_window_drag_target(
    layout_state: &EditorRetainedLayoutState,
    pointer: Vec2,
    hovered: Option<UiId>,
    title_hits: &HashMap<UiId, UiId>,
    allow_resize: bool,
) -> Option<(UiId, WindowDragMode, WindowResizeEdges, LayoutRect)> {
    let ordered = layout_state.windows.ordered_visible_windows();

    if allow_resize {
        let mut best_resize: Option<(f32, usize, UiId, WindowResizeEdges, LayoutRect)> = None;
        for (z_index, (window_id, frame)) in ordered.iter().copied().enumerate() {
            if frame.locked {
                continue;
            }
            let expanded = expand_layout_rect(frame.rect, WINDOW_RESIZE_GRAB_RADIUS);
            if !point_in_layout_rect(expanded, pointer) {
                continue;
            }
            let edges = drag_edges_for_pos(frame.rect, pointer, WINDOW_RESIZE_GRAB_RADIUS);
            if !edges.any() {
                continue;
            }
            let mut edge_distance = f32::INFINITY;
            if edges.left {
                edge_distance = edge_distance.min((pointer.x - frame.rect.x).abs());
            }
            if edges.right {
                edge_distance = edge_distance.min((pointer.x - frame.rect.right()).abs());
            }
            if edges.top {
                edge_distance = edge_distance.min((pointer.y - frame.rect.y).abs());
            }
            if edges.bottom {
                edge_distance = edge_distance.min((pointer.y - frame.rect.bottom()).abs());
            }
            match best_resize {
                Some((best_dist, best_z, _, _, _))
                    if best_dist < edge_distance
                        || ((best_dist - edge_distance).abs() <= f32::EPSILON
                            && best_z >= z_index) => {}
                _ => {
                    best_resize = Some((edge_distance, z_index, window_id, edges, frame.rect));
                }
            }
        }
        if let Some((_, _, window_id, edges, rect)) = best_resize {
            return Some((window_id, WindowDragMode::Resize, edges, rect));
        }
    }

    for (window_id, frame) in ordered.into_iter().rev() {
        if frame.locked || !point_in_layout_rect(frame.rect, pointer) {
            continue;
        }
        let hovered_is_title = hovered
            .and_then(|id| title_hits.get(&id).copied())
            .is_some_and(|hit_window_id| hit_window_id == window_id);
        if hovered.is_none() || hovered_is_title {
            return Some((
                window_id,
                WindowDragMode::Move,
                WindowResizeEdges::default(),
                frame.rect,
            ));
        }
        return None;
    }
    None
}

fn active_tab_for_window(pane_window: &EditorPaneWindow) -> Option<EditorPaneTab> {
    pane_window
        .areas
        .iter()
        .find(|area| !area.tabs.is_empty())
        .and_then(|area| {
            area.tabs
                .get(area.active.min(area.tabs.len().saturating_sub(1)))
        })
        .or_else(|| pane_window.areas.first().and_then(|area| area.tabs.first()))
        .cloned()
}

fn activate_workspace_tab(
    workspace: &mut EditorPaneWorkspaceState,
    workspace_id: UiId,
    tab_id: UiId,
) {
    let Some(window_index) = workspace
        .windows
        .iter()
        .position(|window| pane_window_ui_id(window) == workspace_id)
    else {
        return;
    };
    let Some(area_index) = workspace.windows[window_index]
        .areas
        .iter()
        .position(|area| area.tabs.iter().any(|tab| tab.id == tab_id.0))
    else {
        return;
    };
    let Some(tab_index) = workspace.windows[window_index].areas[area_index]
        .tabs
        .iter()
        .position(|tab| tab.id == tab_id.0)
    else {
        return;
    };

    workspace.windows[window_index].areas[area_index].active = tab_index;
    workspace.last_focused_window = Some(workspace.windows[window_index].id.clone());
    workspace.last_focused_area = Some(workspace.windows[window_index].areas[area_index].id);
}

fn extract_workspace_tab(
    workspace: &mut EditorPaneWorkspaceState,
    workspace_id: UiId,
    tab_id: UiId,
) -> Option<(EditorPaneTab, Option<UiId>)> {
    let window_index = workspace
        .windows
        .iter()
        .position(|window| pane_window_ui_id(window) == workspace_id)?;

    let mut moved_tab = None;
    for area in &mut workspace.windows[window_index].areas {
        if let Some(tab_index) = area.tabs.iter().position(|tab| tab.id == tab_id.0) {
            moved_tab = Some(area.tabs.remove(tab_index));
            area.active = area.active.min(area.tabs.len().saturating_sub(1));
            break;
        }
    }
    let moved_tab = moved_tab?;

    let mut removed_window_ui_id = None;
    let remove_source_window = workspace.windows[window_index]
        .areas
        .iter()
        .all(|area| area.tabs.is_empty())
        && !workspace.windows[window_index].layout_managed;
    if remove_source_window {
        let removed = workspace.windows.remove(window_index);
        removed_window_ui_id = Some(pane_window_ui_id(&removed));
    }

    let focused_exists = workspace
        .last_focused_window
        .as_ref()
        .is_some_and(|focused| {
            workspace
                .windows
                .iter()
                .any(|window| window.id.as_str() == focused.as_str())
        });
    if !focused_exists {
        workspace.last_focused_window = workspace.windows.first().map(|window| window.id.clone());
        workspace.last_focused_area = workspace
            .windows
            .first()
            .and_then(|window| window.areas.first())
            .map(|area| area.id);
    }

    Some((moved_tab, removed_window_ui_id))
}

fn close_workspace_tab(world: &mut World, workspace_id: UiId, tab_id: UiId) {
    let mut removed_window_ui_id = None;
    let closed = world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        if let Some((_removed_tab, removed_window)) =
            extract_workspace_tab(&mut workspace, workspace_id, tab_id)
        {
            removed_window_ui_id = removed_window;
            true
        } else {
            false
        }
    });
    if !closed {
        return;
    }

    world.resource_scope::<EditorRetainedDockingState, _>(|_world, mut docking_state| {
        if let Some(workspace) = docking_state.workspace_mut(workspace_id) {
            workspace.docking.close_tab(tab_id);
        }
        if let Some(removed_window) = removed_window_ui_id {
            docking_state.workspaces.remove(&removed_window);
        }
    });
    if let Some(removed_window) = removed_window_ui_id {
        world.resource_scope::<EditorRetainedLayoutState, _>(|_world, mut layout_state| {
            layout_state.remove_window(removed_window);
        });
    }
}

fn detach_workspace_tab(world: &mut World, workspace_id: UiId, tab_id: UiId) {
    let mut moved = None;
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let Some((tab, removed_source_window)) =
            extract_workspace_tab(&mut workspace, workspace_id, tab_id)
        else {
            return;
        };

        let window_serial = workspace.next_window_id.max(1);
        workspace.next_window_id = workspace.next_window_id.saturating_add(1);
        let new_window_id = format!("Pane {window_serial}");
        let new_area_id = workspace.next_area_id.max(1);
        workspace.next_area_id = workspace.next_area_id.saturating_add(1);

        workspace.windows.push(super::workspace::EditorPaneWindow {
            id: new_window_id.clone(),
            title: tab.title.clone(),
            areas: vec![super::workspace::EditorPaneArea {
                id: new_area_id,
                rect: super::workspace::EditorPaneAreaRect::full(),
                tabs: vec![tab.clone()],
                active: 0,
            }],
            layout_managed: false,
        });
        workspace.last_focused_window = Some(new_window_id.clone());
        workspace.last_focused_area = Some(new_area_id);

        moved = Some((
            tab,
            UiId::from_str(&format!("pane-window-{new_window_id}")),
            removed_source_window,
        ));
    });

    let Some((tab, detached_workspace_id, removed_source_window)) = moved else {
        return;
    };
    let tab_id = UiId::from_raw(tab.id);

    world.resource_scope::<EditorRetainedDockingState, _>(|_world, mut docking_state| {
        if let Some(source_workspace) = docking_state.workspace_mut(workspace_id) {
            source_workspace.docking.close_tab(tab_id);
        }
        if let Some(removed_window) = removed_source_window {
            docking_state.workspaces.remove(&removed_window);
        }
        let detached = docking_state.ensure_workspace(
            detached_workspace_id,
            DockTab::new(tab_id, tab.title.clone()),
        );
        detached.docking.activate_tab(tab_id);
    });
    world.resource_scope::<EditorRetainedLayoutState, _>(|_world, mut layout_state| {
        let bounds = layout_state.workspace_bounds;
        let detached_rect =
            LayoutRect::new(bounds.x + 72.0, bounds.y + 72.0, 420.0, 300.0).clamp_inside(bounds);
        layout_state.windows.add_window(
            detached_workspace_id,
            WindowFrame {
                rect: detached_rect,
                min_size: Vec2::new(220.0, 140.0),
                max_size: None,
                visible: true,
                locked: false,
            },
        );
        layout_state.windows.bring_to_front(detached_workspace_id);
        if let Some(removed_window) = removed_source_window {
            layout_state.remove_window(removed_window);
        }
    });
}

fn open_builtin_tab_in_workspace(world: &mut World, workspace_id: UiId, kind: EditorPaneKind) {
    let mut opened = None;
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| pane_window_ui_id(window) == workspace_id)
        else {
            return;
        };

        let new_tab = EditorPaneTab::from_builtin(&mut workspace, kind);
        let new_tab_id = UiId::from_raw(new_tab.id);
        let new_tab_title = new_tab.title.clone();
        let area_id = if workspace.windows[window_index].areas.is_empty() {
            let area_id = workspace.next_area_id.max(1);
            workspace.next_area_id = workspace.next_area_id.saturating_add(1);
            workspace.windows[window_index]
                .areas
                .push(super::workspace::EditorPaneArea {
                    id: area_id,
                    rect: super::workspace::EditorPaneAreaRect::full(),
                    tabs: Vec::new(),
                    active: 0,
                });
            area_id
        } else {
            workspace.windows[window_index].areas[0].id
        };

        let area = &mut workspace.windows[window_index].areas[0];
        area.tabs.push(new_tab);
        area.active = area.tabs.len().saturating_sub(1);
        workspace.last_focused_window = Some(workspace.windows[window_index].id.clone());
        workspace.last_focused_area = Some(area_id);
        opened = Some((new_tab_id, new_tab_title));
    });

    if let Some((tab_id, title)) = opened {
        world.resource_scope::<EditorRetainedDockingState, _>(|_world, mut docking_state| {
            if let Some(workspace) = docking_state.workspace_mut(workspace_id) {
                workspace
                    .docking
                    .add_tab_to_focused(DockTab::new(tab_id, title.clone()));
                workspace.docking.activate_tab(tab_id);
            } else {
                let workspace =
                    docking_state.ensure_workspace(workspace_id, DockTab::new(tab_id, title));
                workspace.docking.activate_tab(tab_id);
            }
        });
    }
}

fn open_builtin_tab_in_new_window(world: &mut World, _workspace_id: UiId, kind: EditorPaneKind) {
    let mut opened = None;
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let new_tab = EditorPaneTab::from_builtin(&mut workspace, kind);
        let new_tab_id = UiId::from_raw(new_tab.id);
        let new_tab_title = new_tab.title.clone();
        let window_serial = workspace.next_window_id.max(1);
        workspace.next_window_id = workspace.next_window_id.saturating_add(1);
        let new_window_id = format!("Pane {window_serial}");
        let new_area_id = workspace.next_area_id.max(1);
        workspace.next_area_id = workspace.next_area_id.saturating_add(1);

        workspace.windows.push(super::workspace::EditorPaneWindow {
            id: new_window_id.clone(),
            title: new_tab_title.clone(),
            areas: vec![super::workspace::EditorPaneArea {
                id: new_area_id,
                rect: super::workspace::EditorPaneAreaRect::full(),
                tabs: vec![new_tab],
                active: 0,
            }],
            layout_managed: false,
        });
        workspace.last_focused_window = Some(new_window_id.clone());
        workspace.last_focused_area = Some(new_area_id);
        opened = Some((
            UiId::from_str(&format!("pane-window-{new_window_id}")),
            new_tab_id,
            new_tab_title,
        ));
    });

    let Some((new_workspace_id, tab_id, title)) = opened else {
        return;
    };

    world.resource_scope::<EditorRetainedDockingState, _>(|_world, mut docking_state| {
        let workspace =
            docking_state.ensure_workspace(new_workspace_id, DockTab::new(tab_id, title));
        workspace.docking.activate_tab(tab_id);
    });
    world.resource_scope::<EditorRetainedLayoutState, _>(|_world, mut layout_state| {
        let bounds = layout_state.workspace_bounds;
        let rect =
            LayoutRect::new(bounds.x + 84.0, bounds.y + 84.0, 420.0, 300.0).clamp_inside(bounds);
        layout_state.windows.add_window(
            new_workspace_id,
            WindowFrame {
                rect,
                min_size: Vec2::new(220.0, 140.0),
                max_size: None,
                visible: true,
                locked: false,
            },
        );
        layout_state.windows.bring_to_front(new_workspace_id);
    });
}

fn point_in_layout_rect(rect: LayoutRect, point: Vec2) -> bool {
    point.x >= rect.x
        && point.x <= rect.x + rect.width
        && point.y >= rect.y
        && point.y <= rect.y + rect.height
}

fn expand_layout_rect(rect: LayoutRect, amount: f32) -> LayoutRect {
    let amount = amount.max(0.0);
    LayoutRect::new(
        rect.x - amount,
        rect.y - amount,
        rect.width + amount * 2.0,
        rect.height + amount * 2.0,
    )
}

fn drag_edges_for_pos(rect: LayoutRect, point: Vec2, grab_radius: f32) -> WindowResizeEdges {
    let mut edges = WindowResizeEdges::default();
    if grab_radius <= 0.0 {
        return edges;
    }
    if !point_in_layout_rect(expand_layout_rect(rect, grab_radius), point) {
        return edges;
    }

    let left = (point.x - rect.x).abs() <= grab_radius;
    let right = (point.x - (rect.x + rect.width)).abs() <= grab_radius;
    let top = (point.y - rect.y).abs() <= grab_radius;
    let bottom = (point.y - (rect.y + rect.height)).abs() <= grab_radius;
    let center_x = rect.x + rect.width * 0.5;
    let center_y = rect.y + rect.height * 0.5;

    if left && right {
        edges.left = point.x <= center_x;
        edges.right = point.x > center_x;
    } else {
        edges.left = left;
        edges.right = right;
    }

    if top && bottom {
        edges.top = point.y <= center_y;
        edges.bottom = point.y > center_y;
    } else {
        edges.top = top;
        edges.bottom = bottom;
    }

    edges
}

fn move_window_rect(start_rect: LayoutRect, delta: Vec2, bounds: LayoutRect) -> LayoutRect {
    LayoutRect::new(
        start_rect.x + delta.x,
        start_rect.y + delta.y,
        start_rect.width,
        start_rect.height,
    )
    .clamp_inside(bounds)
}

fn resize_window_rect(
    start_rect: LayoutRect,
    delta: Vec2,
    edges: WindowResizeEdges,
    min_size: Vec2,
    max_size: Option<Vec2>,
    bounds: LayoutRect,
) -> LayoutRect {
    let mut left = start_rect.x;
    let mut right = start_rect.x + start_rect.width;
    let mut top = start_rect.y;
    let mut bottom = start_rect.y + start_rect.height;

    if edges.left {
        left += delta.x;
    }
    if edges.right {
        right += delta.x;
    }
    if edges.top {
        top += delta.y;
    }
    if edges.bottom {
        bottom += delta.y;
    }

    let min_width = min_size.x.max(96.0);
    let min_height = min_size.y.max(72.0);
    let max_width = max_size
        .map(|size| size.x.max(min_width))
        .unwrap_or(f32::INFINITY);
    let max_height = max_size
        .map(|size| size.y.max(min_height))
        .unwrap_or(f32::INFINITY);

    let mut width = (right - left).max(1.0);
    let mut height = (bottom - top).max(1.0);
    if width < min_width {
        if edges.left && !edges.right {
            left = right - min_width;
        } else {
            right = left + min_width;
        }
        width = min_width;
    }
    if height < min_height {
        if edges.top && !edges.bottom {
            top = bottom - min_height;
        } else {
            bottom = top + min_height;
        }
        height = min_height;
    }
    if width > max_width {
        if edges.left && !edges.right {
            left = right - max_width;
        } else {
            right = left + max_width;
        }
    }
    if height > max_height {
        if edges.top && !edges.bottom {
            top = bottom - max_height;
        } else {
            bottom = top + max_height;
        }
    }

    LayoutRect::new(left, top, (right - left).max(1.0), (bottom - top).max(1.0))
        .clamp_inside(bounds)
}

fn collect_visible_visual_graph_ids(retained_windows: &[RetainedWindowView]) -> Vec<UiId> {
    let mut ids = Vec::new();
    let mut seen = HashSet::new();

    for window in retained_windows {
        for leaf in &window.leaves {
            let Some(tab) = leaf.active_tab.as_ref() else {
                continue;
            };
            if tab.kind != EditorPaneKind::VisualScriptEditor {
                continue;
            }
            let graph_id = graph_id_for_tab(tab.id, tab.asset_path.as_deref());
            if seen.insert(graph_id) {
                ids.push(graph_id);
            }
        }
    }

    ids
}

fn build_graph_render_payloads(
    world: &World,
    graph_ids: &[UiId],
) -> HashMap<UiId, GraphRenderPayload> {
    let mut payloads = HashMap::new();

    let Some(graph_state) = world.get_resource::<EditorRetainedGraphState>() else {
        return payloads;
    };
    let Some(graph_interactions) = world.get_resource::<EditorRetainedGraphInteractionState>()
    else {
        return payloads;
    };

    for graph_id in graph_ids {
        let Some(graph) = graph_state.graph(*graph_id).cloned() else {
            continue;
        };
        let preview = graph_interactions
            .controllers
            .get(graph_id)
            .and_then(|controller| controller.preview_edge());
        payloads.insert(*graph_id, GraphRenderPayload { graph, preview });
    }

    payloads
}

fn move_workspace_tab_between_windows(
    world: &mut World,
    source_workspace_id: UiId,
    target_workspace_id: UiId,
    tab_id: UiId,
) -> Option<(EditorPaneTab, Vec<EditorPaneWindow>)> {
    if source_workspace_id == target_workspace_id {
        return None;
    }

    let mut workspace = world.get_resource_mut::<EditorPaneWorkspaceState>()?;
    let source_window_index = workspace
        .windows
        .iter()
        .position(|window| pane_window_ui_id(window) == source_workspace_id)?;
    let target_window_index = workspace
        .windows
        .iter()
        .position(|window| pane_window_ui_id(window) == target_workspace_id)?;
    if workspace.windows[target_window_index]
        .areas
        .iter()
        .any(|area| area.tabs.iter().any(|tab| tab.id == tab_id.0))
    {
        return None;
    }

    let mut moved_tab = None;
    {
        let source_window = workspace.windows.get_mut(source_window_index)?;
        for area in &mut source_window.areas {
            if let Some(tab_index) = area.tabs.iter().position(|tab| tab.id == tab_id.0) {
                moved_tab = Some(area.tabs.remove(tab_index));
                area.active = area.active.min(area.tabs.len().saturating_sub(1));
                break;
            }
        }
    }
    let moved_tab = moved_tab?;

    if workspace
        .windows
        .get(target_window_index)
        .is_some_and(|window| window.areas.is_empty())
    {
        let area_id = workspace.next_area_id;
        workspace.next_area_id = workspace.next_area_id.saturating_add(1);
        if let Some(target_window) = workspace.windows.get_mut(target_window_index) {
            target_window.areas.push(super::workspace::EditorPaneArea {
                id: area_id,
                rect: super::workspace::EditorPaneAreaRect::full(),
                tabs: Vec::new(),
                active: 0,
            });
        }
    }
    {
        let target_window = workspace.windows.get_mut(target_window_index)?;
        let target_area = target_window.areas.first_mut()?;
        target_area.tabs.push(moved_tab.clone());
        target_area.active = target_area.tabs.len().saturating_sub(1);
    }

    workspace.last_focused_window = workspace
        .windows
        .get(target_window_index)
        .map(|window| window.id.clone());
    workspace.last_focused_area = workspace
        .windows
        .get(target_window_index)
        .and_then(|window| window.areas.first())
        .map(|area| area.id);

    Some((moved_tab, workspace.windows.clone()))
}

fn sync_workspace_tabs(workspace: &mut EditorDockWorkspace, pane_window: &EditorPaneWindow) {
    let mut existing_tabs = HashSet::new();
    for leaf in workspace.docking.layout(UiRect {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    }) {
        for tab in leaf.tabs {
            existing_tabs.insert(tab);
        }
    }

    for area in &pane_window.areas {
        for tab in &area.tabs {
            let tab_id = UiId::from_raw(tab.id);
            if existing_tabs.contains(&tab_id) {
                continue;
            }
            workspace
                .docking
                .add_tab_to_focused(DockTab::new(tab_id, tab.title.clone()));
            existing_tabs.insert(tab_id);
        }
    }

    if let Some(active_tab) = pane_window
        .areas
        .iter()
        .find(|area| !area.tabs.is_empty())
        .and_then(|area| {
            area.tabs
                .get(area.active.min(area.tabs.len().saturating_sub(1)))
        })
    {
        let _ = workspace
            .docking
            .activate_tab(UiId::from_raw(active_tab.id));
    } else if workspace.docking.focused_tab().is_none()
        && let Some(fallback_tab) = pane_window.areas.iter().find_map(|area| area.tabs.first())
    {
        let _ = workspace
            .docking
            .activate_tab(UiId::from_raw(fallback_tab.id));
    }
}

fn pane_tab_lookup(pane_window: &EditorPaneWindow) -> HashMap<UiId, EditorPaneTab> {
    let mut lookup = HashMap::new();
    for area in &pane_window.areas {
        for tab in &area.tabs {
            lookup.insert(UiId::from_raw(tab.id), tab.clone());
        }
    }
    lookup
}

fn leaf_active_tab(
    leaf: &DockLeafLayout,
    lookup: &HashMap<UiId, EditorPaneTab>,
) -> Option<EditorPaneTab> {
    let tab_id = leaf.active.or_else(|| leaf.tabs.first().copied())?;
    lookup.get(&tab_id).cloned()
}

fn build_placeholder_pane(retained: &mut RetainedUi, root_id: UiId, message: &str) {
    let background_id = root_id.child("background");
    let label_id = root_id.child("label");

    retained.upsert(RetainedUiNode::new(
        root_id,
        UiWidget::Container,
        fill_style(UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        background_id,
        UiWidget::Container,
        fill_style(UiVisualStyle {
            background: Some(UiColor::rgba(0.11, 0.12, 0.16, 0.84)),
            border_color: Some(UiColor::rgba(0.24, 0.28, 0.36, 0.88)),
            border_width: 1.0,
            corner_radius: 0.0,
            clip: true,
        }),
    ));
    retained.upsert(RetainedUiNode::new(
        label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(message.to_string()),
            style: UiTextStyle {
                color: UiColor::rgba(0.84, 0.88, 0.94, 1.0),
                font_size: 12.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, 8.0, 620.0, 20.0, UiVisualStyle::default()),
    ));

    retained.set_children(root_id, [background_id, label_id]);
}

fn delete_hierarchy_subtree(world: &mut World, root: Entity) {
    if world.get_entity(root).is_err() {
        return;
    }

    let mut children: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut relation_query = world.query::<(Entity, &EntityParent)>();
    for (entity, relation) in relation_query.iter(world) {
        children.entry(relation.parent).or_default().push(entity);
    }

    fn collect_postorder(
        node: Entity,
        children: &HashMap<Entity, Vec<Entity>>,
        out: &mut Vec<Entity>,
        visited: &mut HashSet<Entity>,
    ) {
        if !visited.insert(node) {
            return;
        }
        if let Some(nodes) = children.get(&node) {
            for child in nodes {
                collect_postorder(*child, children, out, visited);
            }
        }
        out.push(node);
    }

    let mut ordered = Vec::new();
    let mut visited = HashSet::new();
    collect_postorder(root, &children, &mut ordered, &mut visited);

    for entity in ordered {
        if world.get_entity(entity).is_ok() {
            let _ = world.despawn(entity);
        }
    }

    let clear_selected = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0)
        .is_some_and(|entity| world.get_entity(entity).is_err());
    if clear_selected
        && let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>()
    {
        selected.0 = None;
    }
    if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
        scene.dirty = true;
    }
}

fn duplicate_entity_shallow(
    world: &mut World,
    source: Entity,
    parent_override: Option<Entity>,
) -> Option<Entity> {
    if world.get_entity(source).is_err() {
        return None;
    }

    let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut relations = world.query::<(Entity, &EntityParent)>();
    for (entity, relation) in relations.iter(world) {
        if entity != relation.parent {
            child_map.entry(relation.parent).or_default().push(entity);
        }
    }
    for children in child_map.values_mut() {
        children.sort_by_key(|entity| entity.to_bits());
    }

    fn duplicate_node(
        world: &mut World,
        source: Entity,
        parent: Option<Entity>,
        child_map: &HashMap<Entity, Vec<Entity>>,
        visited: &mut HashSet<Entity>,
        name_copy_suffix: bool,
    ) -> Option<Entity> {
        if !visited.insert(source) {
            return None;
        }
        if world.get_entity(source).is_err() {
            return None;
        }

        let name_value = world.get::<Name>(source).map(|name| {
            if name_copy_suffix {
                format!("{} Copy", name.as_str())
            } else {
                name.as_str().to_string()
            }
        });
        let editor_entity = world.get::<EditorEntity>(source).copied();
        let transform = world.get::<BevyTransform>(source).copied();
        let camera = world.get::<BevyCamera>(source).copied();
        let light = world.get::<BevyLight>(source).copied();
        let mesh_renderer = world.get::<BevyMeshRenderer>(source).copied();
        let skinned_mesh = world.get::<BevySkinnedMeshRenderer>(source).cloned();
        let sprite = world.get::<BevySpriteRenderer>(source).copied();
        let text_2d = world.get::<BevyText2d>(source).cloned();
        let dynamic_body = world.get::<DynamicRigidBody>(source).copied();
        let fixed_collider = world.get::<FixedCollider>(source).is_some();
        let collider_shape = world.get::<ColliderShape>(source).copied();

        let new_entity = world.spawn_empty().id();
        if let Some(editor_entity) = editor_entity {
            world.entity_mut(new_entity).insert(editor_entity);
        }
        if let Some(name) = name_value {
            world.entity_mut(new_entity).insert(Name::new(name));
        }
        if let Some(transform) = transform {
            world.entity_mut(new_entity).insert(transform);
        }
        if let Some(camera) = camera {
            world.entity_mut(new_entity).insert(camera);
        }
        if let Some(light) = light {
            world.entity_mut(new_entity).insert(light);
        }
        if let Some(mesh_renderer) = mesh_renderer {
            world.entity_mut(new_entity).insert(mesh_renderer);
        }
        if let Some(skinned_mesh) = skinned_mesh {
            world.entity_mut(new_entity).insert(skinned_mesh);
        }
        if let Some(sprite) = sprite {
            world.entity_mut(new_entity).insert(sprite);
        }
        if let Some(text_2d) = text_2d {
            world.entity_mut(new_entity).insert(text_2d);
        }
        if let Some(dynamic_body) = dynamic_body {
            world.entity_mut(new_entity).insert(dynamic_body);
        }
        if fixed_collider {
            world.entity_mut(new_entity).insert(FixedCollider);
        }
        if let Some(collider_shape) = collider_shape {
            world.entity_mut(new_entity).insert(collider_shape);
        }

        if let Some(parent) = parent.filter(|parent| *parent != new_entity) {
            let _ = reparent_hierarchy_entity(world, new_entity, Some(parent));
        }

        if let Some(children) = child_map.get(&source) {
            for child in children {
                let _ = duplicate_node(world, *child, Some(new_entity), child_map, visited, false);
            }
        }

        Some(new_entity)
    }

    let source_parent =
        parent_override.or_else(|| world.get::<EntityParent>(source).map(|r| r.parent));
    let mut visited = HashSet::new();
    let new_entity = duplicate_node(world, source, source_parent, &child_map, &mut visited, true)?;

    if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>() {
        selected.0 = Some(new_entity);
    }
    let mut expanded = world
        .get_resource::<EditorRetainedPaneInteractionState>()
        .map(|interaction| interaction.hierarchy_expanded.clone())
        .unwrap_or_default();
    hierarchy_expand_ancestors(world, &mut expanded, new_entity);
    if let Some(mut interaction) = world.get_resource_mut::<EditorRetainedPaneInteractionState>() {
        interaction.hierarchy_expanded = expanded;
    }
    if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
        scene.dirty = true;
    }
    Some(new_entity)
}

fn paste_hierarchy_clipboard(world: &mut World, parent: Option<Entity>) {
    let Some((entity, cut)) = world
        .get_resource::<HierarchyClipboardState>()
        .and_then(|clipboard| clipboard.entity.map(|entity| (entity, clipboard.cut)))
    else {
        return;
    };
    if world.get_entity(entity).is_err() {
        if let Some(mut clipboard) = world.get_resource_mut::<HierarchyClipboardState>() {
            clipboard.entity = None;
            clipboard.cut = false;
        }
        return;
    }

    if cut {
        let _ = reparent_hierarchy_entity(world, entity, parent);
        if let Some(mut clipboard) = world.get_resource_mut::<HierarchyClipboardState>() {
            clipboard.entity = None;
            clipboard.cut = false;
        }
        if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
            scene.dirty = true;
        }
        if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>() {
            selected.0 = Some(entity);
        }
        return;
    }

    let _ = duplicate_entity_shallow(world, entity, parent);
}

fn hierarchy_expand_ancestors(world: &World, expanded: &mut HashSet<Entity>, entity: Entity) {
    let mut current = Some(entity);
    let mut remaining = 2048usize;
    while let Some(node) = current {
        expanded.insert(node);
        if remaining == 0 {
            break;
        }
        remaining -= 1;
        current = world
            .get::<EntityParent>(node)
            .map(|relation| relation.parent)
            .filter(|parent| *parent != node);
    }
}

fn click_cursor_for_text_field(
    pointer: Option<Vec2>,
    field_rect: Option<UiRect>,
    text: &str,
    font_size: f32,
) -> usize {
    let fallback = text.chars().count();
    let Some(pointer) = pointer else {
        return fallback;
    };
    let Some(rect) = field_rect else {
        return fallback;
    };
    let text_start_x = rect.x + 6.0;
    let local_x = (pointer.x - text_start_x).max(0.0);
    caret_from_local_x(text, local_x, font_size)
}

fn caret_from_local_x(text: &str, local_x: f32, font_size: f32) -> usize {
    let mut x = 0.0;
    for (index, ch) in text.chars().enumerate() {
        let advance = estimate_char_advance(ch, font_size).max(1.0);
        if local_x <= x + advance * 0.5 {
            return index;
        }
        x += advance;
    }
    text.chars().count()
}

fn project_text_field_value(world: &World, field: ProjectPaneTextField) -> String {
    let Some(launcher) = world.get_resource::<super::state::EditorProjectLauncherState>() else {
        return String::new();
    };
    match field {
        ProjectPaneTextField::OpenPath => launcher.open_project_path.clone(),
        ProjectPaneTextField::CreateName => launcher.project_name.clone(),
        ProjectPaneTextField::CreateLocation => launcher.project_path.clone(),
    }
}

fn content_browser_text_field_value(world: &World, field: ContentBrowserPaneTextField) -> String {
    let Some(state) = world.get_resource::<AssetBrowserState>() else {
        return String::new();
    };
    match field {
        ContentBrowserPaneTextField::Filter => state.filter.clone(),
    }
}

fn console_text_field_value(world: &World, field: ConsolePaneTextField) -> String {
    let Some(state) = world.get_resource::<EditorConsoleState>() else {
        return String::new();
    };
    match field {
        ConsolePaneTextField::Search => state.search.clone(),
    }
}

fn inspector_text_field_value(world: &World, field: InspectorPaneTextField) -> String {
    let Some(entity) = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0)
    else {
        return String::new();
    };

    match field {
        InspectorPaneTextField::Name => world
            .get::<Name>(entity)
            .map(|name| name.as_str().to_string())
            .unwrap_or_default(),
        InspectorPaneTextField::Transform(component_field) => world
            .get::<helmer_becs::BevyTransform>(entity)
            .map(|transform| match component_field {
                crate::retained::panes::InspectorTransformField::PositionX => {
                    transform.0.position.x
                }
                crate::retained::panes::InspectorTransformField::PositionY => {
                    transform.0.position.y
                }
                crate::retained::panes::InspectorTransformField::PositionZ => {
                    transform.0.position.z
                }
                crate::retained::panes::InspectorTransformField::RotationX => transform
                    .0
                    .rotation
                    .to_euler(glam::EulerRot::YXZ)
                    .0
                    .to_degrees(),
                crate::retained::panes::InspectorTransformField::RotationY => transform
                    .0
                    .rotation
                    .to_euler(glam::EulerRot::YXZ)
                    .1
                    .to_degrees(),
                crate::retained::panes::InspectorTransformField::RotationZ => transform
                    .0
                    .rotation
                    .to_euler(glam::EulerRot::YXZ)
                    .2
                    .to_degrees(),
                crate::retained::panes::InspectorTransformField::ScaleX => transform.0.scale.x,
                crate::retained::panes::InspectorTransformField::ScaleY => transform.0.scale.y,
                crate::retained::panes::InspectorTransformField::ScaleZ => transform.0.scale.z,
            })
            .map(numeric_field_text)
            .unwrap_or_default(),
        InspectorPaneTextField::Camera(component_field) => world
            .get::<helmer_becs::BevyCamera>(entity)
            .map(|camera| match component_field {
                crate::retained::panes::InspectorCameraField::FovDeg => {
                    camera.0.fov_y_rad.to_degrees()
                }
                crate::retained::panes::InspectorCameraField::Aspect => camera.0.aspect_ratio,
                crate::retained::panes::InspectorCameraField::Near => camera.0.near_plane,
                crate::retained::panes::InspectorCameraField::Far => camera.0.far_plane,
            })
            .map(numeric_field_text)
            .unwrap_or_default(),
        InspectorPaneTextField::Light(component_field) => world
            .get::<helmer_becs::BevyLight>(entity)
            .map(|light| match component_field {
                crate::retained::panes::InspectorLightField::Intensity => light.0.intensity,
                crate::retained::panes::InspectorLightField::SpotAngleDeg => {
                    match light.0.light_type {
                        helmer::provided::components::LightType::Spot { angle } => {
                            angle.to_degrees()
                        }
                        _ => 45.0,
                    }
                }
            })
            .map(numeric_field_text)
            .unwrap_or_default(),
    }
}

fn apply_project_text_field_input(
    world: &mut World,
    field: ProjectPaneTextField,
    cursors: &mut HashMap<ProjectPaneTextField, usize>,
    just_pressed: &[winit::keyboard::KeyCode],
    shift_down: bool,
    ctrl_down: bool,
) {
    let Some(mut launcher) = world.get_resource_mut::<super::state::EditorProjectLauncherState>()
    else {
        return;
    };

    let target = match field {
        ProjectPaneTextField::OpenPath => &mut launcher.open_project_path,
        ProjectPaneTextField::CreateName => &mut launcher.project_name,
        ProjectPaneTextField::CreateLocation => &mut launcher.project_path,
    };

    let cursor = cursors
        .entry(field)
        .or_insert_with(|| target.chars().count());
    let changed = apply_text_field_input(target, cursor, just_pressed, shift_down, ctrl_down);

    if changed {
        launcher.status = None;
    }
}

fn apply_content_browser_text_field_input(
    world: &mut World,
    field: ContentBrowserPaneTextField,
    cursors: &mut HashMap<ContentBrowserPaneTextField, usize>,
    just_pressed: &[winit::keyboard::KeyCode],
    shift_down: bool,
    ctrl_down: bool,
) {
    let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() else {
        return;
    };

    let target = match field {
        ContentBrowserPaneTextField::Filter => &mut browser.filter,
    };

    let cursor = cursors
        .entry(field)
        .or_insert_with(|| target.chars().count());
    let changed = apply_text_field_input(target, cursor, just_pressed, shift_down, ctrl_down);

    if changed {
        browser.refresh_requested = true;
    }
}

fn apply_inspector_text_field_input(
    world: &mut World,
    field: InspectorPaneTextField,
    cursors: &mut HashMap<InspectorPaneTextField, usize>,
    just_pressed: &[winit::keyboard::KeyCode],
    shift_down: bool,
    ctrl_down: bool,
) {
    let Some(entity) = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0)
    else {
        return;
    };

    match field {
        InspectorPaneTextField::Name => {
            let mut value = world
                .get::<Name>(entity)
                .map(|name| name.as_str().to_string())
                .unwrap_or_default();
            let cursor = cursors
                .entry(InspectorPaneTextField::Name)
                .or_insert_with(|| value.chars().count());
            let changed =
                apply_text_field_input(&mut value, cursor, just_pressed, shift_down, ctrl_down);

            if changed {
                let trimmed = value.trim().to_string();
                if trimmed.is_empty() {
                    world.entity_mut(entity).remove::<Name>();
                } else {
                    world.entity_mut(entity).insert(Name::new(trimmed));
                }
            }
        }
        InspectorPaneTextField::Transform(component_field) => {
            let Some(current) = world
                .get::<helmer_becs::BevyTransform>(entity)
                .map(|value| match component_field {
                    crate::retained::panes::InspectorTransformField::PositionX => {
                        value.0.position.x
                    }
                    crate::retained::panes::InspectorTransformField::PositionY => {
                        value.0.position.y
                    }
                    crate::retained::panes::InspectorTransformField::PositionZ => {
                        value.0.position.z
                    }
                    crate::retained::panes::InspectorTransformField::RotationX => value
                        .0
                        .rotation
                        .to_euler(glam::EulerRot::YXZ)
                        .0
                        .to_degrees(),
                    crate::retained::panes::InspectorTransformField::RotationY => value
                        .0
                        .rotation
                        .to_euler(glam::EulerRot::YXZ)
                        .1
                        .to_degrees(),
                    crate::retained::panes::InspectorTransformField::RotationZ => value
                        .0
                        .rotation
                        .to_euler(glam::EulerRot::YXZ)
                        .2
                        .to_degrees(),
                    crate::retained::panes::InspectorTransformField::ScaleX => value.0.scale.x,
                    crate::retained::panes::InspectorTransformField::ScaleY => value.0.scale.y,
                    crate::retained::panes::InspectorTransformField::ScaleZ => value.0.scale.z,
                })
            else {
                return;
            };
            let cursor = cursors
                .entry(field)
                .or_insert_with(|| numeric_field_text(current).chars().count());
            if let Some(next) =
                apply_numeric_field_input(current, cursor, just_pressed, shift_down, ctrl_down)
            {
                apply_inspector_action(
                    world,
                    InspectorPaneAction::SetTransformValue {
                        entity,
                        field: component_field,
                        value: next,
                    },
                );
            }
        }
        InspectorPaneTextField::Camera(component_field) => {
            let Some(current) =
                world
                    .get::<helmer_becs::BevyCamera>(entity)
                    .map(|value| match component_field {
                        crate::retained::panes::InspectorCameraField::FovDeg => {
                            value.0.fov_y_rad.to_degrees()
                        }
                        crate::retained::panes::InspectorCameraField::Aspect => {
                            value.0.aspect_ratio
                        }
                        crate::retained::panes::InspectorCameraField::Near => value.0.near_plane,
                        crate::retained::panes::InspectorCameraField::Far => value.0.far_plane,
                    })
            else {
                return;
            };
            let cursor = cursors
                .entry(field)
                .or_insert_with(|| numeric_field_text(current).chars().count());
            if let Some(next) =
                apply_numeric_field_input(current, cursor, just_pressed, shift_down, ctrl_down)
            {
                apply_inspector_action(
                    world,
                    InspectorPaneAction::SetCameraValue {
                        entity,
                        field: component_field,
                        value: next,
                    },
                );
            }
        }
        InspectorPaneTextField::Light(component_field) => {
            let Some(current) =
                world
                    .get::<helmer_becs::BevyLight>(entity)
                    .map(|value| match component_field {
                        crate::retained::panes::InspectorLightField::Intensity => value.0.intensity,
                        crate::retained::panes::InspectorLightField::SpotAngleDeg => {
                            match value.0.light_type {
                                helmer::provided::components::LightType::Spot { angle } => {
                                    angle.to_degrees()
                                }
                                _ => 45.0,
                            }
                        }
                    })
            else {
                return;
            };
            let cursor = cursors
                .entry(field)
                .or_insert_with(|| numeric_field_text(current).chars().count());
            if let Some(next) =
                apply_numeric_field_input(current, cursor, just_pressed, shift_down, ctrl_down)
            {
                apply_inspector_action(
                    world,
                    InspectorPaneAction::SetLightValue {
                        entity,
                        field: component_field,
                        value: next,
                    },
                );
            }
        }
    }
}

fn apply_numeric_field_input(
    current: f32,
    cursor: &mut usize,
    just_pressed: &[winit::keyboard::KeyCode],
    shift_down: bool,
    ctrl_down: bool,
) -> Option<f32> {
    let mut text = numeric_field_text(current);
    *cursor = (*cursor).min(text.chars().count());
    let changed = apply_text_field_input_with_validator(
        &mut text,
        cursor,
        just_pressed,
        shift_down,
        ctrl_down,
        |ch| ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E',
    );
    if !changed {
        return None;
    }
    text.parse::<f32>().ok()
}

fn apply_console_text_field_input(
    world: &mut World,
    field: ConsolePaneTextField,
    cursors: &mut HashMap<ConsolePaneTextField, usize>,
    just_pressed: &[winit::keyboard::KeyCode],
    shift_down: bool,
    ctrl_down: bool,
) {
    let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() else {
        return;
    };

    let target = match field {
        ConsolePaneTextField::Search => &mut console.search,
    };
    let cursor = cursors
        .entry(field)
        .or_insert_with(|| target.chars().count());
    let _ = apply_text_field_input(target, cursor, just_pressed, shift_down, ctrl_down);
}

fn numeric_field_text(value: f32) -> String {
    let mut text = format!("{value:.6}");
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    if text.is_empty() {
        text.push('0');
    }
    text
}

fn apply_text_field_input(
    text: &mut String,
    cursor: &mut usize,
    just_pressed: &[winit::keyboard::KeyCode],
    shift_down: bool,
    ctrl_down: bool,
) -> bool {
    apply_text_field_input_with_validator(text, cursor, just_pressed, shift_down, ctrl_down, |_| {
        true
    })
}

fn apply_text_field_input_with_validator<F>(
    text: &mut String,
    cursor: &mut usize,
    just_pressed: &[winit::keyboard::KeyCode],
    shift_down: bool,
    ctrl_down: bool,
    mut allow_char: F,
) -> bool
where
    F: FnMut(char) -> bool,
{
    use winit::keyboard::KeyCode;

    let mut changed = false;
    *cursor = (*cursor).min(text.chars().count());
    for key in just_pressed {
        match key {
            KeyCode::ArrowLeft => {
                if *cursor > 0 {
                    *cursor -= 1;
                }
            }
            KeyCode::ArrowRight => {
                let len = text.chars().count();
                if *cursor < len {
                    *cursor += 1;
                }
            }
            KeyCode::Home => {
                *cursor = 0;
            }
            KeyCode::End => {
                *cursor = text.chars().count();
            }
            KeyCode::Backspace => {
                if *cursor > 0 && remove_char_at(text, *cursor - 1) {
                    *cursor -= 1;
                    changed = true;
                }
            }
            KeyCode::Delete if ctrl_down => {
                if !text.is_empty() {
                    text.clear();
                    *cursor = 0;
                    changed = true;
                }
            }
            KeyCode::Delete => {
                if remove_char_at(text, *cursor) {
                    changed = true;
                }
            }
            _ => {
                if let Some(ch) = keycode_to_char(*key, shift_down)
                    && allow_char(ch)
                {
                    insert_char_at(text, *cursor, ch);
                    *cursor += 1;
                    changed = true;
                }
            }
        }
    }
    changed
}

fn insert_char_at(text: &mut String, char_index: usize, ch: char) {
    let byte_index = byte_index_for_char_index(text, char_index);
    text.insert(byte_index, ch);
}

fn remove_char_at(text: &mut String, char_index: usize) -> bool {
    let len = text.chars().count();
    if char_index >= len {
        return false;
    }
    let start = byte_index_for_char_index(text, char_index);
    let end = byte_index_for_char_index(text, char_index + 1);
    text.replace_range(start..end, "");
    true
}

fn byte_index_for_char_index(text: &str, char_index: usize) -> usize {
    if char_index == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_index)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

fn keycode_to_char(key: winit::keyboard::KeyCode, shift_down: bool) -> Option<char> {
    use winit::keyboard::KeyCode;

    let ch = match key {
        KeyCode::Space => ' ',
        KeyCode::KeyA => {
            if shift_down {
                'A'
            } else {
                'a'
            }
        }
        KeyCode::KeyB => {
            if shift_down {
                'B'
            } else {
                'b'
            }
        }
        KeyCode::KeyC => {
            if shift_down {
                'C'
            } else {
                'c'
            }
        }
        KeyCode::KeyD => {
            if shift_down {
                'D'
            } else {
                'd'
            }
        }
        KeyCode::KeyE => {
            if shift_down {
                'E'
            } else {
                'e'
            }
        }
        KeyCode::KeyF => {
            if shift_down {
                'F'
            } else {
                'f'
            }
        }
        KeyCode::KeyG => {
            if shift_down {
                'G'
            } else {
                'g'
            }
        }
        KeyCode::KeyH => {
            if shift_down {
                'H'
            } else {
                'h'
            }
        }
        KeyCode::KeyI => {
            if shift_down {
                'I'
            } else {
                'i'
            }
        }
        KeyCode::KeyJ => {
            if shift_down {
                'J'
            } else {
                'j'
            }
        }
        KeyCode::KeyK => {
            if shift_down {
                'K'
            } else {
                'k'
            }
        }
        KeyCode::KeyL => {
            if shift_down {
                'L'
            } else {
                'l'
            }
        }
        KeyCode::KeyM => {
            if shift_down {
                'M'
            } else {
                'm'
            }
        }
        KeyCode::KeyN => {
            if shift_down {
                'N'
            } else {
                'n'
            }
        }
        KeyCode::KeyO => {
            if shift_down {
                'O'
            } else {
                'o'
            }
        }
        KeyCode::KeyP => {
            if shift_down {
                'P'
            } else {
                'p'
            }
        }
        KeyCode::KeyQ => {
            if shift_down {
                'Q'
            } else {
                'q'
            }
        }
        KeyCode::KeyR => {
            if shift_down {
                'R'
            } else {
                'r'
            }
        }
        KeyCode::KeyS => {
            if shift_down {
                'S'
            } else {
                's'
            }
        }
        KeyCode::KeyT => {
            if shift_down {
                'T'
            } else {
                't'
            }
        }
        KeyCode::KeyU => {
            if shift_down {
                'U'
            } else {
                'u'
            }
        }
        KeyCode::KeyV => {
            if shift_down {
                'V'
            } else {
                'v'
            }
        }
        KeyCode::KeyW => {
            if shift_down {
                'W'
            } else {
                'w'
            }
        }
        KeyCode::KeyX => {
            if shift_down {
                'X'
            } else {
                'x'
            }
        }
        KeyCode::KeyY => {
            if shift_down {
                'Y'
            } else {
                'y'
            }
        }
        KeyCode::KeyZ => {
            if shift_down {
                'Z'
            } else {
                'z'
            }
        }
        KeyCode::Digit0 | KeyCode::Numpad0 => {
            if shift_down {
                ')'
            } else {
                '0'
            }
        }
        KeyCode::Digit1 | KeyCode::Numpad1 => {
            if shift_down {
                '!'
            } else {
                '1'
            }
        }
        KeyCode::Digit2 | KeyCode::Numpad2 => {
            if shift_down {
                '@'
            } else {
                '2'
            }
        }
        KeyCode::Digit3 | KeyCode::Numpad3 => {
            if shift_down {
                '#'
            } else {
                '3'
            }
        }
        KeyCode::Digit4 | KeyCode::Numpad4 => {
            if shift_down {
                '$'
            } else {
                '4'
            }
        }
        KeyCode::Digit5 | KeyCode::Numpad5 => {
            if shift_down {
                '%'
            } else {
                '5'
            }
        }
        KeyCode::Digit6 | KeyCode::Numpad6 => {
            if shift_down {
                '^'
            } else {
                '6'
            }
        }
        KeyCode::Digit7 | KeyCode::Numpad7 => {
            if shift_down {
                '&'
            } else {
                '7'
            }
        }
        KeyCode::Digit8 | KeyCode::Numpad8 => {
            if shift_down {
                '*'
            } else {
                '8'
            }
        }
        KeyCode::Digit9 | KeyCode::Numpad9 => {
            if shift_down {
                '('
            } else {
                '9'
            }
        }
        KeyCode::Minus | KeyCode::NumpadSubtract => {
            if shift_down {
                '_'
            } else {
                '-'
            }
        }
        KeyCode::Equal | KeyCode::NumpadAdd => {
            if shift_down {
                '+'
            } else {
                '='
            }
        }
        KeyCode::Period | KeyCode::NumpadDecimal => {
            if shift_down {
                '>'
            } else {
                '.'
            }
        }
        KeyCode::Comma => {
            if shift_down {
                '<'
            } else {
                ','
            }
        }
        KeyCode::Slash | KeyCode::NumpadDivide => {
            if shift_down {
                '?'
            } else {
                '/'
            }
        }
        KeyCode::Backslash => {
            if shift_down {
                '|'
            } else {
                '\\'
            }
        }
        KeyCode::Semicolon => {
            if shift_down {
                ':'
            } else {
                ';'
            }
        }
        KeyCode::Quote => {
            if shift_down {
                '"'
            } else {
                '\''
            }
        }
        KeyCode::BracketLeft => {
            if shift_down {
                '{'
            } else {
                '['
            }
        }
        KeyCode::BracketRight => {
            if shift_down {
                '}'
            } else {
                ']'
            }
        }
        KeyCode::Backquote => {
            if shift_down {
                '~'
            } else {
                '`'
            }
        }
        _ => return None,
    };
    Some(ch)
}

fn inset_rect(rect: UiRect, inset: f32) -> UiRect {
    let inset = inset.max(0.0);
    UiRect {
        x: rect.x + inset,
        y: rect.y + inset,
        width: (rect.width - inset * 2.0).max(1.0),
        height: (rect.height - inset * 2.0).max(1.0),
    }
}

fn fill_style(visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .width(UiDimension::percent(1.0))
            .height(UiDimension::percent(1.0))
            .build(),
        visual,
    }
}

fn absolute_style(x: f32, y: f32, width: f32, height: f32, visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .position_type(helmer_ui::UiPositionType::Absolute)
            .left(UiDimension::points(x))
            .top(UiDimension::points(y))
            .width(UiDimension::points(width.max(0.0)))
            .height(UiDimension::points(height.max(0.0)))
            .build(),
        visual,
    }
}
