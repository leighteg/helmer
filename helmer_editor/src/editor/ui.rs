use std::{
    any::Any,
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Component, Entity, Resource, With, World};
use egui::containers::menu::{MenuButton, MenuConfig};
use egui::{
    Align, Align2, Color32, ComboBox, DragValue, FontId, Id, Layout, Modifiers, Order,
    PointerButton, PopupCloseBehavior, Pos2, Rect, Response, RichText, Sense, Stroke, StrokeKind,
    TextStyle, Ui, Vec2,
};
use glam::{DVec2, EulerRot, Mat3, Quat, Vec3};
use helmer::animation::{AnimationClip, Pose, Skeleton};
use helmer::audio::{AudioBus, AudioLoadMode, AudioPlaybackState};
use helmer::graphics::common::config::SkinningMode;
use helmer::graphics::{
    common::renderer::GizmoMode,
    render_graphs::{graph_templates, template_for_graph},
};
use helmer::provided::components::{
    AudioEmitter, AudioListener, EntityFollower, Light, LightType, LookAt, MeshRenderer,
    PoseOverride, SkinnedMeshRenderer, Spline, SplineFollower, SplineMode,
};
use helmer::runtime::asset_server::{Handle, Material, MaterialFile, Mesh, Scene};
use helmer_becs::egui_integration::{EguiInputPassthrough, EguiResource};
use helmer_becs::physics::components::{
    CharacterController, CharacterControllerInput, CharacterControllerOutput, ColliderProperties,
    ColliderPropertyInheritance, ColliderShape, DynamicRigidBody, FixedCollider, KinematicMode,
    KinematicRigidBody, MeshColliderKind, MeshColliderLod, PhysicsCombineRule, PhysicsHandle,
    PhysicsJoint, PhysicsJointKind, PhysicsPointProjection, PhysicsPointProjectionHit,
    PhysicsQueryFilter, PhysicsRayCast, PhysicsRayCastHit, PhysicsShapeCast, PhysicsShapeCastHit,
    PhysicsWorldDefaults, RigidBodyProperties, RigidBodyPropertyInheritance,
};
use helmer_becs::physics::physics_resource::PhysicsResource;
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::scene_system::{
    EntityParent, SceneChild, SceneRoot, SceneSpawnedChildren, build_default_animator,
};
use helmer_becs::{
    AudioBackendResource, BevyActiveCamera, BevyAnimator, BevyAssetServer, BevyAudioEmitter,
    BevyAudioListener, BevyCamera, BevyEntityFollower, BevyLight, BevyLookAt, BevyMeshRenderer,
    BevyPoseOverride, BevyRuntimeConfig, BevySkinnedMeshRenderer, BevySpline, BevySplineFollower,
    BevyTransform, BevyWrapper,
};
use ron::ser::PrettyConfig;
use walkdir::WalkDir;

use crate::editor::{
    EditorLayoutState, EditorPlayCamera, EditorSplineState, EditorTimelineState, EditorUndoState,
    EditorViewportCamera, EditorViewportRuntime, EditorViewportState, EditorViewportTextures,
    Freecam, LayoutSaveRequest, PlayViewportKind, UndoEntry, ViewportRectPixels,
    ViewportResolutionPreset,
    assets::{
        AssetBrowserState, AssetEntry, EditorAssetCache, EditorAudio, EditorMesh,
        EditorSkinnedMesh, MeshSource, PrimitiveKind, SceneAssetPath, cached_audio_handle,
        cached_scene_handle, is_entry_visible,
    },
    begin_material_undo_group, begin_undo_group,
    commands::{AssetCreateKind, EditorCommand, EditorCommandQueue, SpawnKind},
    dynamic::{DynamicComponent, DynamicComponents, DynamicField, DynamicValue, DynamicValueKind},
    end_material_undo_group, end_undo_group, enforce_undo_cap, ensure_play_camera,
    ensure_viewport_camera_for_pane,
    gizmos::{EditorGizmoSettings, EditorGizmoState},
    project::{
        EditorProject, ProjectConfig, default_rust_script_template_full,
        default_script_template_full, rust_script_manifest_template,
        rust_script_sdk_dependency_path, sanitize_rust_crate_name, save_project_config,
    },
    push_undo_snapshot, save_layouts,
    scene::{
        EditorEntity, EditorSceneState, PendingSkinnedMeshAsset, WorldState,
        animation_asset_from_group, apply_custom_clips_to_animator,
        merge_animation_asset_into_timeline, next_available_scene_path,
        read_animation_asset_document, write_animation_asset_document,
    },
    scripting::{
        ScriptComponent, ScriptEditCommand, ScriptEditModeState, ScriptEntry, ScriptInstanceKey,
        ScriptRegistry, ScriptRuntime, is_script_path, normalize_script_language,
        script_language_from_path,
    },
    timeline::{
        CameraKey, CameraTrack, ClipSegment, ClipTrack, JointKey, JointTrack, LightKey, LightTrack,
        PoseKey, PoseTrack, SplineKey, SplineTrack, TimelineClipExpandRequest, TimelineDragSelect,
        TimelineDragSelectMode, TimelineDragSelectPending, TimelineInterpolation,
        TimelineSelection, TimelineTrack, TimelineTrackGroup, TransformKey, TransformTrack,
        build_clip_from_clip_track, build_clip_from_pose_track, build_pose_track_from_clip,
        build_pose_track_from_clip_segment,
    },
};

#[derive(Default, Debug, Clone, Resource)]
pub struct EditorUiState {
    pub project_name: String,
    pub project_path: String,
    pub open_project_path: String,
    pub status: Option<String>,
    pub recent_projects: Vec<PathBuf>,
}

#[derive(Debug, Clone, Resource)]
pub struct EditorPaneVisibility {
    pub toolbar: bool,
    pub viewport: bool,
    pub project: bool,
    pub hierarchy: bool,
    pub inspector: bool,
    pub history: bool,
    pub timeline: bool,
    pub content_browser: bool,
    pub console: bool,
    pub audio_mixer: bool,
}

impl Default for EditorPaneVisibility {
    fn default() -> Self {
        Self {
            toolbar: true,
            viewport: true,
            project: true,
            hierarchy: true,
            inspector: true,
            history: false,
            timeline: false,
            content_browser: true,
            console: true,
            audio_mixer: false,
        }
    }
}

#[derive(Debug, Clone, Resource, Default)]
pub struct EditorPaneManagerState {
    pub open: bool,
}

#[derive(Debug, Clone, Resource)]
pub struct EditorPaneAutoState {
    pub project_auto_hide: bool,
    pub last_project_open: bool,
}

impl Default for EditorPaneAutoState {
    fn default() -> Self {
        Self {
            project_auto_hide: true,
            last_project_open: false,
        }
    }
}

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
}

impl EditorPaneKind {
    pub const ALL: [Self; 11] = [
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
            Self::Timeline | Self::AudioMixer => None,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorPaneViewportSettings {
    pub graph_template: String,
    pub gizmos_in_play: bool,
    pub show_camera_gizmos: bool,
    pub show_directional_light_gizmos: bool,
    pub show_point_light_gizmos: bool,
    pub show_spot_light_gizmos: bool,
    pub show_spline_paths: bool,
    pub show_spline_points: bool,
}

impl EditorPaneViewportSettings {
    pub fn from_viewport_state(state: &EditorViewportState) -> Self {
        Self {
            graph_template: state.graph_template.clone(),
            gizmos_in_play: state.gizmos_in_play,
            show_camera_gizmos: state.show_camera_gizmos,
            show_directional_light_gizmos: state.show_directional_light_gizmos,
            show_point_light_gizmos: state.show_point_light_gizmos,
            show_spot_light_gizmos: state.show_spot_light_gizmos,
            show_spline_paths: state.show_spline_paths,
            show_spline_points: state.show_spline_points,
        }
    }
}

#[derive(Debug, Clone, Resource, Default)]
pub struct EditorPaneViewportState {
    pub resolutions: HashMap<u64, ViewportResolutionPreset>,
    pub settings: HashMap<u64, EditorPaneViewportSettings>,
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

    pub fn to_rect(self, host: Rect) -> Rect {
        let width = host.width().max(1.0);
        let height = host.height().max(1.0);
        Rect::from_min_size(
            Pos2::new(host.min.x + self.x * width, host.min.y + self.y * height),
            Vec2::new(self.w * width, self.h * height),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitAxis {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaneDropZone {
    Center,
    Left,
    Right,
    Top,
    Bottom,
}

impl PaneDropZone {
    fn from_split_axis(axis: SplitAxis) -> Self {
        match axis {
            SplitAxis::Vertical => Self::Right,
            SplitAxis::Horizontal => Self::Bottom,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EditorPaneTab {
    pub id: u64,
    pub title: String,
    pub kind: EditorPaneKind,
}

#[derive(Debug, Clone)]
pub struct EditorPaneTabDrag {
    pub tab: EditorPaneTab,
    pub source_window_id: String,
    pub source_area_id: u64,
    pub source_was_single_tab: bool,
}

#[derive(Debug, Clone, Resource)]
pub struct EditorAudioDeviceCache {
    pub hosts: Vec<helmer::audio::AudioHostId>,
    pub devices: Vec<helmer::audio::AudioOutputDevice>,
    pub last_host: Option<helmer::audio::AudioHostId>,
    pub show_system_devices: bool,
    pub pending_output: Option<helmer::audio::AudioOutputSettings>,
    pub pending_dirty: bool,
    pub new_bus_name: String,
}

impl Default for EditorAudioDeviceCache {
    fn default() -> Self {
        Self {
            hosts: Vec::new(),
            devices: Vec::new(),
            last_host: None,
            show_system_devices: false,
            pending_output: None,
            pending_dirty: false,
            new_bus_name: String::new(),
        }
    }
}

#[derive(Default, Debug, Clone, Resource)]
pub struct MaterialEditorCache {
    pub entries: HashMap<PathBuf, MaterialEditorEntry>,
}

#[derive(Default, Debug, Clone)]
pub struct MaterialEditorEntry {
    pub data: Option<MaterialFile>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Resource)]
pub struct EditorWorkspaceState {
    pub next_window_id: u64,
    pub next_tab_id: u64,
    pub windows: Vec<EditorTabWindow>,
    pub last_focused_window: Option<u64>,
    pub dragging: Option<EditorTabDrag>,
    pub drop_handled: bool,
}

impl Default for EditorWorkspaceState {
    fn default() -> Self {
        Self {
            next_window_id: 1,
            next_tab_id: 1,
            windows: Vec::new(),
            last_focused_window: None,
            dragging: None,
            drop_handled: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EditorTabWindow {
    pub id: u64,
    pub title: String,
    pub tabs: Vec<EditorTab>,
    pub active: usize,
}

#[derive(Debug, Clone)]
pub struct EditorTab {
    pub id: u64,
    pub title: String,
    pub content: EditorTabContent,
}

#[derive(Debug, Clone)]
pub enum EditorTabContent {
    Material { path: PathBuf },
}

#[derive(Debug, Clone)]
pub struct EditorTabDrag {
    pub tab: EditorTab,
    pub source_window_id: u64,
    pub source_was_single_tab: bool,
}

#[derive(Default, Debug, Clone, Resource)]
pub struct InspectorNameEditState {
    pub entity: Option<Entity>,
    pub buffer: String,
}

#[derive(Default, Debug, Clone, Resource)]
pub struct InspectorPinnedEntityResource(pub Option<Entity>);

#[derive(Default, Debug, Resource)]
pub struct AssetDragState {
    pub active: bool,
    pub path: Option<PathBuf>,
}

impl AssetDragState {
    pub fn start_drag(&mut self, path: PathBuf) {
        self.active = true;
        self.path = Some(path);
    }

    pub fn stop_drag(&mut self) {
        self.active = false;
        self.path = None;
    }
}

#[derive(Default, Debug, Resource)]
pub struct EntityDragState {
    pub active: bool,
    pub entity: Option<Entity>,
}

impl EntityDragState {
    pub fn start_drag(&mut self, entity: Entity) {
        self.active = true;
        self.entity = Some(entity);
    }

    pub fn stop_drag(&mut self) {
        self.active = false;
        self.entity = None;
    }
}

#[derive(Default, Debug, Resource)]
pub struct MiddleDragUiState {
    pub active: bool,
    pub locked_window_id: Option<Id>,
    pub start_pos: Pos2,
    pub delta: Vec2,
}

#[derive(Debug, Clone, Resource)]
pub struct HierarchyUiState {
    pub rename_entity: Option<Entity>,
    pub rename_buffer: String,
    pub rename_request_focus: bool,
    pub expanded_entities: HashSet<Entity>,
    pub new_dynamic_component_name: String,
    pub new_dynamic_field_name: String,
    pub new_dynamic_field_kind: DynamicValueKind,
}

impl Default for HierarchyUiState {
    fn default() -> Self {
        Self {
            rename_entity: None,
            rename_buffer: String::new(),
            rename_request_focus: false,
            expanded_entities: HashSet::new(),
            new_dynamic_component_name: String::new(),
            new_dynamic_field_name: String::new(),
            new_dynamic_field_kind: DynamicValueKind::Float,
        }
    }
}

#[derive(Debug, Clone, Resource)]
pub struct AnimatorUiState {
    pub new_float_name: String,
    pub new_float_value: f32,
    pub new_bool_name: String,
    pub new_bool_value: bool,
    pub new_trigger_name: String,
}

impl Default for AnimatorUiState {
    fn default() -> Self {
        Self {
            new_float_name: String::new(),
            new_float_value: 0.0,
            new_bool_name: String::new(),
            new_bool_value: false,
            new_trigger_name: String::new(),
        }
    }
}

#[derive(Debug, Clone, Resource)]
pub struct PoseEditorState {
    pub joint_filter: String,
    pub selected_joint: Option<usize>,
    pub hover_joint: Option<usize>,
    pub edit_mode: bool,
    pub active_entity: Option<u64>,
    pub use_gizmo: bool,
    pub show_bones: bool,
    pub show_joint_handles: bool,
    pub handle_size_scale: f32,
    pub handle_size_min: f32,
    pub pick_radius_scale: f32,
    pub pick_radius_min: f32,
    pub dragging: bool,
}

impl Default for PoseEditorState {
    fn default() -> Self {
        Self {
            joint_filter: String::new(),
            selected_joint: None,
            hover_joint: None,
            edit_mode: false,
            active_entity: None,
            use_gizmo: true,
            show_bones: true,
            show_joint_handles: true,
            handle_size_scale: 0.03,
            handle_size_min: 0.05,
            pick_radius_scale: 0.025,
            pick_radius_min: 0.04,
            dragging: false,
        }
    }
}

pub fn draw_toolbar(ui: &mut Ui, world: &mut World) {
    draw_entity_drag_indicator(ui, world);

    let (scene_name, scene_dirty, world_state) = {
        let Some(scene) = world.get_resource::<EditorSceneState>() else {
            return;
        };
        (scene.name.clone(), scene.dirty, scene.world_state)
    };

    let project_loaded = world
        .get_resource::<EditorProject>()
        .and_then(|project| project.root.as_ref())
        .is_some();

    let (can_undo, undo_label, can_redo, redo_label) = world
        .get_resource::<EditorUndoState>()
        .map(|state| {
            (
                state.can_undo(),
                state.undo_label().map(|label| label.to_string()),
                state.can_redo(),
                state.redo_label().map(|label| label.to_string()),
            )
        })
        .unwrap_or((false, None, false, None));

    with_middle_drag_blocked(ui, world, |ui, world| {
        ui.horizontal(|ui| {
            let undo_button = ui.add_enabled(can_undo, egui::Button::new("Undo"));
            let undo_button = match undo_label.as_deref() {
                Some(label) => undo_button.on_hover_text(format!("Undo {}", label)),
                None => undo_button,
            };
            if undo_button.clicked() {
                push_command(world, EditorCommand::Undo);
            }

            let redo_button = ui.add_enabled(can_redo, egui::Button::new("Redo"));
            let redo_button = match redo_label.as_deref() {
                Some(label) => redo_button.on_hover_text(format!("Redo {}", label)),
                None => redo_button,
            };
            if redo_button.clicked() {
                push_command(world, EditorCommand::Redo);
            }

            if ui.button("New Scene").clicked() {
                push_command(world, EditorCommand::NewScene);
            }

            if ui.button("Save").clicked() {
                push_command(world, EditorCommand::SaveScene);
            }

            if ui.button("Save As").clicked() {
                if project_loaded {
                    if let Some(path) = next_available_scene_path(
                        world
                            .get_resource::<EditorProject>()
                            .expect("Project missing"),
                    ) {
                        push_command(world, EditorCommand::SaveSceneAs { path });
                    } else {
                        set_status(world, "Unable to allocate a scene file name".to_string());
                    }
                } else {
                    set_status(world, "Open a project before saving".to_string());
                }
            }

            let play_label = match world_state {
                WorldState::Edit => "Play",
                WorldState::Play => "Stop",
            };
            if ui.button(play_label).clicked() {
                push_command(world, EditorCommand::TogglePlayMode);
            }

            draw_layout_menu(ui, world);

            ui.separator();

            let dirty_marker = if scene_dirty { "*" } else { "" };
            ui.label(format!("Scene: {}{}", scene_name, dirty_marker));

            if let Some(status) = world
                .get_resource::<EditorUiState>()
                .and_then(|state| state.status.clone())
            {
                ui.label(RichText::new(status).small());
            }
        });
    });
}

fn draw_layout_menu(ui: &mut Ui, world: &mut World) {
    let Some(mut layout_state) = world.get_resource_mut::<EditorLayoutState>() else {
        return;
    };

    let mut status_message: Option<String> = None;

    ui.menu_button("Layout", |ui| {
        ui.set_min_width(220.0);
        let previous_allow_edit = layout_state.allow_layout_edit;
        let mut names = layout_state.layouts.keys().cloned().collect::<Vec<_>>();
        names.sort();

        for name in names {
            let is_active = layout_state.active.as_deref() == Some(name.as_str());
            let label = if is_active {
                format!("[x] {}", name)
            } else {
                format!("[ ] {}", name)
            };
            if ui.selectable_label(is_active, label).clicked() {
                if name == "Default" {
                    let default_layout = crate::editor::default_layout();
                    layout_state
                        .layouts
                        .insert(default_layout.name.clone(), default_layout);
                }
                if is_active {
                    layout_state.active = None;
                    layout_state.apply_requested = false;
                    layout_state.last_screen_rect = None;
                } else {
                    layout_state.active = Some(name.clone());
                    layout_state.apply_requested = true;
                }
                if let Err(err) = save_layouts(&layout_state) {
                    status_message = Some(format!("Failed to save layouts: {}", err));
                }
                ui.close_menu();
            }
        }

        ui.separator();
        if ui
            .checkbox(
                &mut layout_state.allow_layout_edit,
                "Allow moving/resizing while layout active",
            )
            .changed()
            && previous_allow_edit
            && !layout_state.allow_layout_edit
        {
            layout_state.apply_requested = true;
        }
        ui.checkbox(&mut layout_state.live_reflow, "Live reflow while resizing");
        ui.separator();
        let can_save_active = layout_state.active.is_some();
        if ui
            .add_enabled(can_save_active, egui::Button::new("Save Active Layout"))
            .clicked()
        {
            layout_state.save_request = Some(LayoutSaveRequest::SaveActive);
            ui.close_menu();
        }

        ui.horizontal(|ui| {
            ui.label("Save As:");
            ui.text_edit_singleline(&mut layout_state.new_layout_name);
            if ui.button("Save").clicked() {
                let name = layout_state.new_layout_name.trim().to_string();
                if name.is_empty() {
                    status_message = Some("Layout name cannot be empty".to_string());
                } else if layout_state.layouts.contains_key(&name) {
                    status_message = Some("Layout name already exists".to_string());
                } else {
                    layout_state.save_request = Some(LayoutSaveRequest::SaveAs(name));
                    layout_state.new_layout_name.clear();
                    ui.close_menu();
                }
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Rename:");
            ui.text_edit_singleline(&mut layout_state.rename_layout_name);
            if ui
                .add_enabled(
                    layout_state.active.is_some(),
                    egui::Button::new("Rename Active"),
                )
                .clicked()
            {
                let Some(active_name) = layout_state.active.clone() else {
                    status_message = Some("No active layout to rename".to_string());
                    ui.close_menu();
                    return;
                };
                if active_name == "Default" {
                    status_message = Some("Default layout cannot be renamed".to_string());
                    ui.close_menu();
                    return;
                }
                let new_name = layout_state.rename_layout_name.trim().to_string();
                if new_name.is_empty() {
                    status_message = Some("New layout name cannot be empty".to_string());
                    ui.close_menu();
                    return;
                }
                if new_name == active_name {
                    status_message = Some("Layout already uses that name".to_string());
                    ui.close_menu();
                    return;
                }
                if layout_state.layouts.contains_key(&new_name) {
                    status_message = Some("Layout name already exists".to_string());
                    ui.close_menu();
                    return;
                }
                if let Some(mut layout) = layout_state.layouts.remove(&active_name) {
                    layout.name = new_name.clone();
                    layout_state.layouts.insert(new_name.clone(), layout);
                    layout_state.active = Some(new_name);
                    layout_state.rename_layout_name.clear();
                    if let Err(err) = save_layouts(&layout_state) {
                        status_message = Some(format!("Failed to save layouts: {}", err));
                    }
                } else {
                    status_message = Some("Active layout not found".to_string());
                }
                ui.close_menu();
            }
        });

        if ui
            .add_enabled(
                layout_state.active.is_some(),
                egui::Button::new("Delete Active"),
            )
            .clicked()
        {
            let Some(active_name) = layout_state.active.clone() else {
                status_message = Some("No active layout to delete".to_string());
                ui.close_menu();
                return;
            };
            if active_name == "Default" {
                status_message = Some("Default layout cannot be deleted".to_string());
                ui.close_menu();
                return;
            }
            layout_state.layouts.remove(&active_name);
            layout_state.active = None;
            layout_state.apply_requested = false;
            layout_state.last_screen_rect = None;
            if let Err(err) = save_layouts(&layout_state) {
                status_message = Some(format!("Failed to save layouts: {}", err));
            }
            ui.close_menu();
        }
    });

    drop(layout_state);
    if let Some(message) = status_message {
        set_status(world, message);
    }
}

pub fn draw_history_window(ui: &mut Ui, world: &mut World) {
    with_middle_drag_blocked(ui, world, |ui, world| {
        ui.heading("History");

        let mut max_entries = world
            .get_resource::<EditorUndoState>()
            .map(|state| state.max_entries)
            .unwrap_or(128);
        let mut cap_changed = false;
        ui.horizontal(|ui| {
            ui.label("Cap");
            let response = ui.add(DragValue::new(&mut max_entries).clamp_range(1..=4096));
            if response.changed() {
                cap_changed = true;
            }
        });
        if cap_changed {
            if let Some(mut state) = world.get_resource_mut::<EditorUndoState>() {
                state.max_entries = max_entries.max(1);
                enforce_undo_cap(&mut state);
            }
        }

        let (entries, cursor, max_entries) = world
            .get_resource::<EditorUndoState>()
            .map(|state| (state.entries.clone(), state.cursor, state.max_entries))
            .unwrap_or((Vec::new(), 0, max_entries));
        let project = world.get_resource::<EditorProject>().cloned();

        let can_undo = world
            .get_resource::<EditorUndoState>()
            .map(|state| state.can_undo())
            .unwrap_or(false);
        let can_redo = world
            .get_resource::<EditorUndoState>()
            .map(|state| state.can_redo())
            .unwrap_or(false);

        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_undo, egui::Button::new("Undo"))
                .clicked()
            {
                push_command(world, EditorCommand::Undo);
            }
            if ui
                .add_enabled(can_redo, egui::Button::new("Redo"))
                .clicked()
            {
                push_command(world, EditorCommand::Redo);
            }
        });

        if entries.is_empty() {
            ui.label("No history entries");
            return;
        }

        ui.label(format!(
            "Entries: {} / {} (Current: {})",
            entries.len(),
            max_entries,
            cursor + 1
        ));
        ui.separator();

        let available_height = ui.available_height();
        egui::ScrollArea::vertical()
            .id_source("history_entries")
            .auto_shrink([false, false])
            .max_height(available_height)
            .show(ui, |ui| {
                for (index, entry) in entries.iter().enumerate().rev() {
                    let label = history_entry_label(entry, &project);
                    let is_current = index == cursor;
                    let _ = ui.selectable_label(is_current, label);
                }
            });
    });
}

fn console_level_color(level: crate::editor::EditorConsoleLevel) -> Color32 {
    match level {
        crate::editor::EditorConsoleLevel::Trace => Color32::from_rgb(130, 130, 130),
        crate::editor::EditorConsoleLevel::Debug => Color32::from_rgb(145, 155, 175),
        crate::editor::EditorConsoleLevel::Log => Color32::from_rgb(170, 170, 170),
        crate::editor::EditorConsoleLevel::Info => Color32::from_rgb(145, 175, 235),
        crate::editor::EditorConsoleLevel::Warn => Color32::from_rgb(235, 185, 90),
        crate::editor::EditorConsoleLevel::Error => Color32::from_rgb(225, 95, 95),
    }
}

pub fn draw_console_window(ui: &mut Ui, world: &mut World) {
    with_middle_drag_blocked(ui, world, |ui, world| {
        let mut clear_requested = false;
        if let Some(mut state) = world.get_resource_mut::<crate::editor::EditorConsoleState>() {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Clear").clicked() {
                    clear_requested = true;
                }
                ui.checkbox(&mut state.auto_scroll, "Auto-scroll");
                ui.separator();
                ui.label("Level:");
                ui.checkbox(&mut state.show_log, "Log");
                ui.checkbox(&mut state.show_debug, "Debug");
                ui.checkbox(&mut state.show_info, "Info");
                ui.checkbox(&mut state.show_warn, "Warn");
                ui.checkbox(&mut state.show_error, "Error");
                ui.checkbox(&mut state.show_trace, "Trace");
                ui.separator();
                ui.label("Search");
                ui.text_edit_singleline(&mut state.search);
            });
            ui.separator();

            let search = state.search.trim().to_ascii_lowercase();
            let has_search = !search.is_empty();
            let entries = state.entries.iter().cloned().collect::<Vec<_>>();
            let auto_scroll = state.auto_scroll;
            egui::ScrollArea::vertical()
                .id_salt("editor_console_entries")
                .auto_shrink([false, false])
                .stick_to_bottom(auto_scroll)
                .show(ui, |ui| {
                    let mut shown = 0usize;
                    for entry in entries.iter() {
                        if !state.level_enabled(entry.level) {
                            continue;
                        }
                        if has_search {
                            let target = entry.target.to_ascii_lowercase();
                            let message = entry.message.to_ascii_lowercase();
                            if !target.contains(&search) && !message.contains(&search) {
                                continue;
                            }
                        }
                        shown += 1;
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                RichText::new(format!("#{:06}", entry.sequence))
                                    .small()
                                    .color(Color32::from_gray(110)),
                            );
                            ui.colored_label(
                                console_level_color(entry.level),
                                format!("[{}]", entry.level.label()),
                            );
                            if !entry.target.is_empty() {
                                ui.label(
                                    RichText::new(format!("[{}]", entry.target))
                                        .small()
                                        .color(Color32::from_gray(140)),
                                );
                            }
                            ui.label(RichText::new(&entry.message).small().monospace());
                        });
                    }
                    if shown == 0 {
                        ui.label(RichText::new("No console output").small());
                    }
                });
        } else {
            ui.label("Console state missing");
        }

        if clear_requested {
            if let Some(mut state) = world.get_resource_mut::<crate::editor::EditorConsoleState>() {
                state.clear();
            }
        }
    });
}

fn align_output_buffer_frames_ui(frames: u32) -> u32 {
    if frames == 0 {
        return 0;
    }
    let alignment = 256u32;
    ((frames + alignment - 1) / alignment) * alignment
}

pub fn draw_audio_mixer_window(ui: &mut Ui, world: &mut World) {
    with_middle_drag_blocked(ui, world, |ui, world| {
        let audio = match world
            .get_resource::<AudioBackendResource>()
            .map(|backend| backend.0.clone())
        {
            Some(audio) => audio,
            None => {
                ui.label("Audio backend not available.");
                return;
            }
        };

        ui.heading("Audio Mixer");
        ui.separator();

        let mut enabled = audio.enabled();
        if ui.checkbox(&mut enabled, "enabled").changed() {
            audio.set_enabled(enabled);
        }

        ui.separator();
        ui.label("output");
        {
            let mut cache = world
                .get_resource_mut::<EditorAudioDeviceCache>()
                .expect("EditorAudioDeviceCache missing");
            let current_output = audio.output_settings();
            let mut pending_output = if cache.pending_output.is_none() || !cache.pending_dirty {
                current_output.clone()
            } else {
                cache
                    .pending_output
                    .clone()
                    .expect("pending output missing")
            };
            let mut pending_dirty = cache.pending_dirty;

            if cache.hosts.is_empty() {
                cache.hosts = audio.available_output_hosts();
            }

            let mut host_changed = false;
            let host_label = pending_output
                .host_id
                .map(|host| host.label())
                .unwrap_or_else(|| "System Default".to_string());
            ComboBox::from_label("host")
                .selected_text(host_label)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(pending_output.host_id.is_none(), "System Default")
                        .clicked()
                    {
                        pending_output.host_id = None;
                        host_changed = true;
                    }
                    for host in cache.hosts.iter().copied() {
                        let label = host.label();
                        if ui
                            .selectable_label(pending_output.host_id == Some(host), &label)
                            .clicked()
                        {
                            pending_output.host_id = Some(host);
                            host_changed = true;
                        }
                    }
                });
            if host_changed {
                pending_output.device_name = None;
                pending_output.device_index = None;
                cache.devices.clear();
                cache.last_host = None;
                pending_dirty = true;
            }

            ui.horizontal(|ui| {
                if ui.button("refresh devices").clicked() {
                    cache.devices.clear();
                    cache.last_host = None;
                }
                ui.checkbox(&mut cache.show_system_devices, "show system devices");
            });

            let needs_refresh =
                cache.devices.is_empty() || cache.last_host != pending_output.host_id;
            if needs_refresh {
                cache.devices = audio.available_output_devices(pending_output.host_id);
                cache.last_host = pending_output.host_id;
            }

            let mut name_counts: HashMap<String, usize> = HashMap::new();
            let mut filtered_devices: Vec<helmer::audio::AudioOutputDevice> = Vec::new();
            for device in cache.devices.iter() {
                let lower = device.name.to_lowercase();
                let is_alias = matches!(
                    lower.as_str(),
                    "default" | "sysdefault" | "samplerate" | "speexrate" | "jack" | "pulse"
                );
                let is_virtual = lower.contains("dmix")
                    || lower.contains("dsnoop")
                    || lower.contains("front")
                    || lower.contains("surround")
                    || lower.contains("iec958")
                    || lower.contains("spdif")
                    || lower.contains("null")
                    || lower.contains("plug")
                    || lower.contains("hw:")
                    || lower.contains("hdmi");
                if !cache.show_system_devices && (is_alias || is_virtual) {
                    continue;
                }
                *name_counts.entry(device.name.clone()).or_insert(0) += 1;
                filtered_devices.push(device.clone());
            }

            let current_device_label = pending_output
                .device_index
                .and_then(|index| {
                    filtered_devices
                        .iter()
                        .find(|device| device.index == index)
                        .map(|device| {
                            if name_counts.get(&device.name).copied().unwrap_or(0) > 1 {
                                format!("{} [{}]", device.name, device.index)
                            } else {
                                device.name.clone()
                            }
                        })
                })
                .or_else(|| pending_output.device_name.clone())
                .unwrap_or_else(|| "default".to_string());

            ComboBox::from_label("device")
                .selected_text(current_device_label)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(
                            pending_output.device_name.is_none()
                                && pending_output.device_index.is_none(),
                            "default",
                        )
                        .clicked()
                    {
                        pending_output.device_name = None;
                        pending_output.device_index = None;
                        pending_dirty = true;
                    }
                    for device in filtered_devices.iter() {
                        let label = if name_counts.get(&device.name).copied().unwrap_or(0) > 1 {
                            format!("{} [{}]", device.name, device.index)
                        } else {
                            device.name.clone()
                        };
                        if ui
                            .selectable_label(
                                pending_output.device_index == Some(device.index),
                                &label,
                            )
                            .clicked()
                        {
                            pending_output.device_name = Some(device.name.clone());
                            pending_output.device_index = Some(device.index);
                            pending_dirty = true;
                        }
                    }
                });

            let mut sample_rate = pending_output.sample_rate.max(1);
            let mut channels = pending_output.channels.max(1);
            let mut buffer_frames = pending_output.buffer_frames.unwrap_or(0);
            let sample_rate_response = ui.add(
                DragValue::new(&mut sample_rate)
                    .speed(100.0)
                    .prefix("sample rate: "),
            );
            if sample_rate_response.changed() {
                pending_output.sample_rate = sample_rate;
                pending_dirty = true;
            }
            let channels_response =
                ui.add(egui::Slider::new(&mut channels, 1..=16).text("channels"));
            if channels_response.changed() {
                pending_output.channels = channels;
                pending_dirty = true;
            }
            let buffer_response = ui.add(
                DragValue::new(&mut buffer_frames)
                    .speed(256.0)
                    .prefix("buffer frames: "),
            );
            if buffer_response.changed() {
                let aligned = align_output_buffer_frames_ui(buffer_frames);
                pending_output.buffer_frames = if aligned == 0 { None } else { Some(aligned) };
                pending_dirty = true;
            }
            if ui.button("apply output").clicked() {
                match audio.reconfigure(pending_output.clone()) {
                    Ok(()) => {
                        pending_dirty = false;
                        pending_output = audio.output_settings();
                    }
                    Err(err) => {
                        ui.colored_label(Color32::RED, format!("output error: {}", err));
                    }
                }
            }
            if let Some(err) = audio.last_error() {
                ui.colored_label(Color32::RED, err);
            }

            cache.pending_output = Some(pending_output);
            cache.pending_dirty = pending_dirty;
        }

        ui.separator();
        ui.label("spatialization");
        let mut head_width = audio.head_width();
        if ui
            .add(
                DragValue::new(&mut head_width)
                    .speed(0.01)
                    .prefix("head width: "),
            )
            .changed()
        {
            audio.set_head_width(head_width);
        }
        let mut speed = audio.speed_of_sound();
        if ui
            .add(
                DragValue::new(&mut speed)
                    .speed(1.0)
                    .prefix("speed of sound: "),
            )
            .changed()
        {
            audio.set_speed_of_sound(speed);
        }

        ui.separator();
        ui.label("streaming");
        let (stream_buffer, stream_chunk) = audio.streaming_config();
        let mut stream_buffer = stream_buffer as u32;
        let mut stream_chunk = stream_chunk as u32;
        let stream_changed = ui
            .add(
                DragValue::new(&mut stream_buffer)
                    .speed(64.0)
                    .prefix("stream buffer frames: "),
            )
            .changed()
            || ui
                .add(
                    DragValue::new(&mut stream_chunk)
                        .speed(64.0)
                        .prefix("stream chunk frames: "),
                )
                .changed();
        if stream_changed {
            audio.set_streaming_config(stream_buffer as usize, stream_chunk as usize);
        }

        ui.separator();
        ui.label("asset budgets");
        let mb = 1024.0 * 1024.0;
        #[cfg(target_arch = "wasm32")]
        let asset_server = world.get_non_send_resource::<BevyAssetServer>();
        #[cfg(not(target_arch = "wasm32"))]
        let asset_server = world.get_resource::<BevyAssetServer>();
        if let Some(asset_server) = asset_server {
            let mut server = asset_server.0.lock();
            let mut audio_budget_mb = server.audio_budget_bytes() as f32 / mb as f32;
            let audio_cache_mb = server.audio_cache_usage_bytes() as f32 / mb as f32;
            if ui
                .add(
                    DragValue::new(&mut audio_budget_mb)
                        .speed(1.0)
                        .prefix("audio budget (MiB): "),
                )
                .changed()
            {
                let budget_bytes = (audio_budget_mb.max(0.0) * mb as f32) as usize;
                server.set_audio_budget_bytes(budget_bytes);
            }
            ui.label(format!("audio cache usage: {:.1} MiB", audio_cache_mb));
        } else {
            ui.label("audio asset budgets unavailable");
        }

        ui.separator();
        ui.label("buses");
        {
            let mut cache = world
                .get_resource_mut::<EditorAudioDeviceCache>()
                .expect("EditorAudioDeviceCache missing");
            ui.horizontal(|ui| {
                ui.label("new bus:");
                ui.text_edit_singleline(&mut cache.new_bus_name);
                if ui.button("Add").clicked() {
                    let name = cache.new_bus_name.trim().to_string();
                    if name.is_empty() {
                        audio.create_custom_bus(None);
                    } else {
                        audio.create_custom_bus(Some(name));
                    }
                    cache.new_bus_name.clear();
                }
            });
            let buses = audio.bus_list();
            for bus in buses {
                let mut volume = audio.bus_volume(bus);
                let mut remove_requested = false;
                ui.horizontal(|ui| {
                    match bus {
                        AudioBus::Custom(_) => {
                            let mut name = audio.bus_name(bus);
                            let font_id = egui::TextStyle::Body.resolve(ui.style());
                            let galley = ui.painter().layout_no_wrap(
                                name.clone(),
                                font_id,
                                ui.visuals().text_color(),
                            );
                            let desired = galley.size().x;
                            let width = (desired + 16.0).clamp(60.0, 200.0);
                            if ui
                                .add(egui::TextEdit::singleline(&mut name).desired_width(width))
                                .changed()
                            {
                                audio.set_bus_name(bus, name);
                            }
                            if ui.button("Remove").clicked() {
                                remove_requested = true;
                            }
                        }
                        _ => {
                            ui.label(audio.bus_name(bus));
                        }
                    }
                    if ui.add(egui::Slider::new(&mut volume, 0.0..=2.0)).changed() {
                        audio.set_bus_volume(bus, volume);
                    }
                });
                if remove_requested {
                    audio.remove_bus(bus);
                }
            }
        }

        ui.separator();
        ui.label("scenes");
        let mut scenes: Vec<(u64, String)> = Vec::new();
        for (entity, _) in world.query::<(Entity, &SceneRoot)>().iter(world) {
            let name = world
                .get::<Name>(entity)
                .map(|n| n.as_str().to_string())
                .unwrap_or_else(|| format!("Scene {}", entity.to_bits()));
            scenes.push((entity.to_bits(), name));
        }
        scenes.sort_by(|a, b| a.1.cmp(&b.1));
        if scenes.is_empty() {
            ui.label("no active scenes");
        } else {
            for (scene_id, name) in scenes {
                let mut volume = audio.scene_volume(scene_id);
                ui.horizontal(|ui| {
                    ui.label(name);
                    if ui.add(egui::Slider::new(&mut volume, 0.0..=2.0)).changed() {
                        audio.set_scene_volume(scene_id, volume);
                    }
                });
            }
        }
    });
}

pub fn draw_pane_manager_window(ui: &mut Ui, world: &mut World) {
    with_middle_drag_blocked(ui, world, |ui, world| {
        ui.horizontal(|ui| {
            if ui.button("Reset To Default").clicked() {
                reset_pane_workspace(world);
            }
            if ui.button("Close").clicked() {
                if let Some(mut state) = world.get_resource_mut::<EditorPaneManagerState>() {
                    state.open = false;
                }
            }
        });
        ui.separator();

        ui.label("Create Pane");
        ui.horizontal_wrapped(|ui| {
            for kind in EditorPaneKind::ALL {
                if ui.button(kind.label()).clicked() {
                    spawn_pane_workspace_tab(world, kind);
                }
            }
        });
        ui.separator();

        if let Some(workspace) = world.get_resource::<EditorPaneWorkspaceState>() {
            ui.separator();
            ui.label(format!("Windows: {}", workspace.windows.len()));
            egui::ScrollArea::vertical()
                .id_salt("pane_manager_window_list")
                .max_height(160.0)
                .show(ui, |ui| {
                    for window in workspace.windows.iter() {
                        let tab_count: usize =
                            window.areas.iter().map(|area| area.tabs.len()).sum();
                        let kind = if window.layout_managed {
                            "Layout"
                        } else {
                            "Detached"
                        };
                        ui.label(format!(
                            "{} [{}] - {} tabs in {} areas",
                            window.id,
                            kind,
                            tab_count,
                            window.areas.len()
                        ));
                    }
                });
        }
    });
}

fn history_entry_label(entry: &UndoEntry, project: &Option<EditorProject>) -> String {
    match entry {
        UndoEntry::Scene(snapshot) => snapshot
            .label
            .clone()
            .unwrap_or_else(|| "Scene".to_string()),
        UndoEntry::Material(snapshot) => {
            let path_label = project_relative_path(project, &snapshot.path);
            let base_label = snapshot.label.as_deref().unwrap_or("Material");
            if base_label.contains(&path_label) {
                base_label.to_string()
            } else {
                format!("{} ({})", base_label, path_label)
            }
        }
    }
}

fn ensure_viewport_texture_ids(
    ui: &Ui,
    world: &mut World,
) -> Option<(egui::TextureId, egui::TextureId, egui::TextureId)> {
    let mut textures = world.get_resource_mut::<EditorViewportTextures>()?;
    let placeholder = egui::ColorImage::filled([1, 1], Color32::BLACK);
    let options = egui::TextureOptions::LINEAR;

    let editor_id = {
        textures
            .editor
            .get_or_insert_with(|| {
                ui.ctx().load_texture(
                    "helmer_editor/viewport/editor",
                    placeholder.clone(),
                    options,
                )
            })
            .id()
    };
    let gameplay_id = {
        textures
            .gameplay
            .get_or_insert_with(|| {
                ui.ctx().load_texture(
                    "helmer_editor/viewport/gameplay",
                    placeholder.clone(),
                    options,
                )
            })
            .id()
    };
    let preview_id = {
        textures
            .preview
            .get_or_insert_with(|| {
                ui.ctx()
                    .load_texture("helmer_editor/viewport/preview", placeholder, options)
            })
            .id()
    };
    Some((editor_id, gameplay_id, preview_id))
}

fn viewport_rect_pixels_from_ui_rect(
    rect: Rect,
    pixels_per_point: f32,
) -> Option<ViewportRectPixels> {
    if !pixels_per_point.is_finite() || pixels_per_point <= 0.0 {
        return None;
    }
    ViewportRectPixels::new(
        rect.min.x * pixels_per_point,
        rect.min.y * pixels_per_point,
        rect.max.x * pixels_per_point,
        rect.max.y * pixels_per_point,
    )
}

fn fit_rect_to_aspect(container: Rect, aspect_ratio: f32) -> Rect {
    let container_w = container.width().max(1.0);
    let container_h = container.height().max(1.0);
    let aspect = if aspect_ratio.is_finite() && aspect_ratio > 0.0 {
        aspect_ratio
    } else {
        container_w / container_h
    };

    let container_aspect = container_w / container_h;
    if container_aspect > aspect {
        let width = container_h * aspect;
        let x = container.center().x - width * 0.5;
        Rect::from_min_max(
            Pos2::new(x, container.min.y),
            Pos2::new(x + width, container.max.y),
        )
    } else {
        let height = container_w / aspect;
        let y = container.center().y - height * 0.5;
        Rect::from_min_max(
            Pos2::new(container.min.x, y),
            Pos2::new(container.max.x, y + height),
        )
    }
}

fn first_camera_with_component<T: Component>(world: &mut World) -> Option<Entity> {
    let mut query = world.query_filtered::<(Entity, &BevyCamera), With<T>>();
    query.iter(world).map(|(entity, _)| entity).next()
}

fn resolve_preview_camera_for_viewport(world: &mut World) -> Option<Entity> {
    let state_pin = world
        .get_resource::<EditorViewportState>()
        .and_then(|state| state.pinned_camera);
    let inspector_pin = world
        .get_resource::<InspectorPinnedEntityResource>()
        .and_then(|state| state.0);
    let selected = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|state| state.0);

    let mut candidates = [state_pin, inspector_pin, selected]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    candidates.dedup();

    candidates.into_iter().find(|entity| {
        world.get::<BevyCamera>(*entity).is_some() && world.get::<BevyTransform>(*entity).is_some()
    })
}

fn ensure_pane_viewport_texture_id(
    ui: &Ui,
    world: &mut World,
    pane_id: u64,
) -> Option<egui::TextureId> {
    let mut textures = world.get_resource_mut::<EditorViewportTextures>()?;
    let placeholder = egui::ColorImage::filled([1, 1], Color32::BLACK);
    let options = egui::TextureOptions::LINEAR;
    let id = textures
        .pane_textures
        .entry(pane_id)
        .or_insert_with(|| {
            ui.ctx().load_texture(
                format!("helmer_editor/viewport/pane_{}", pane_id),
                placeholder,
                options,
            )
        })
        .id();
    Some(id)
}

pub fn draw_viewport_pane(ui: &mut Ui, world: &mut World, pane_id: u64, play_viewport: bool) {
    with_middle_drag_blocked(ui, world, |ui, world| {
        let world_state = world
            .get_resource::<EditorSceneState>()
            .map(|state| state.world_state)
            .unwrap_or(WorldState::Edit);
        let mut play_mode_view = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.play_mode_view)
            .unwrap_or_default();
        let show_options_panel = false;
        let fallback_viewport_state = world
            .get_resource::<EditorViewportState>()
            .cloned()
            .unwrap_or_default();
        let fallback_resolution = fallback_viewport_state.render_resolution;
        let fallback_pane_settings =
            EditorPaneViewportSettings::from_viewport_state(&fallback_viewport_state);
        let (mut render_resolution, pane_settings) =
            if let Some(mut pane_state) = world.get_resource_mut::<EditorPaneViewportState>() {
                let render_resolution = pane_state
                    .resolutions
                    .entry(pane_id)
                    .or_insert(fallback_resolution)
                    .to_owned();
                let pane_settings = pane_state
                    .settings
                    .entry(pane_id)
                    .or_insert_with(|| fallback_pane_settings.clone())
                    .clone();
                (render_resolution, pane_settings)
            } else {
                (fallback_resolution, fallback_pane_settings)
            };
        let previous_render_resolution = render_resolution;
        let templates = graph_templates();
        let mut graph_template = if pane_settings.graph_template.is_empty() {
            templates
                .first()
                .map(|template| template.name.to_string())
                .unwrap_or_else(|| "default-graph".to_string())
        } else {
            pane_settings.graph_template.clone()
        };
        let previous_graph_template = graph_template.clone();
        let mut pinned_camera = world
            .get_resource::<EditorViewportState>()
            .and_then(|state| state.pinned_camera);
        let mut gizmos_in_play = pane_settings.gizmos_in_play;
        let mut execute_scripts_in_edit_mode = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.execute_scripts_in_edit_mode)
            .unwrap_or(false);
        let mut show_camera_gizmos = pane_settings.show_camera_gizmos;
        let mut show_directional_light_gizmos = pane_settings.show_directional_light_gizmos;
        let mut show_point_light_gizmos = pane_settings.show_point_light_gizmos;
        let mut show_spot_light_gizmos = pane_settings.show_spot_light_gizmos;
        let mut show_spline_paths = pane_settings.show_spline_paths;
        let mut show_spline_points = pane_settings.show_spline_points;
        let mut gizmo_mode = world
            .get_resource::<EditorGizmoState>()
            .map(|state| state.mode)
            .unwrap_or(GizmoMode::None);
        let mut preview_position_norm = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.preview_position_norm)
            .unwrap_or([0.03, 0.74]);
        let mut preview_width_norm = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.preview_width_norm)
            .unwrap_or(0.28);
        let script_registry_status = world
            .get_resource::<ScriptRegistry>()
            .and_then(|registry| registry.status.clone());
        let (rust_status, script_errors, script_error_count) =
            if let Some(runtime) = world.get_resource::<ScriptRuntime>() {
                (
                    runtime.rust_status.clone(),
                    runtime.errors.iter().take(3).cloned().collect::<Vec<_>>(),
                    runtime.errors.len(),
                )
            } else {
                (None, Vec::new(), 0)
            };
        let menu_max_height = (ui.ctx().content_rect().height() * 0.62).clamp(170.0, 520.0);
        let menu_max_width = (ui.ctx().content_rect().width() * 0.26).clamp(150.0, 340.0);
        let advanced_menu_max_height =
            (ui.ctx().content_rect().height() * 0.78).clamp(220.0, 760.0);
        let advanced_menu_width = (ui.ctx().content_rect().width() * 0.36).clamp(220.0, 460.0);
        let viewport_menu_config =
            MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside);
        let advanced_menu_config =
            MenuConfig::new().close_behavior(PopupCloseBehavior::IgnoreClicks);
        let mut graph_selection_changed = false;
        let selected_camera = world
            .get_resource::<InspectorSelectedEntityResource>()
            .and_then(|selection| selection.0)
            .filter(|entity| world.get::<BevyCamera>(*entity).is_some());

        let Some(texture_id) = ensure_pane_viewport_texture_id(ui, world, pane_id) else {
            ui.label("Viewport texture allocation failed.");
            return;
        };
        let preview_texture_id = ensure_viewport_texture_ids(ui, world).map(|(_, _, id)| id);

        let editor_camera_entity = ensure_viewport_camera_for_pane(world, pane_id);
        let play_camera_entity = if play_viewport && world_state == WorldState::Play {
            ensure_play_camera(world)
        } else {
            None
        };
        let camera_entity = if play_viewport {
            play_camera_entity.unwrap_or(editor_camera_entity)
        } else {
            editor_camera_entity
        };

        ui.horizontal_wrapped(|ui| {
            if play_viewport {
                ui.label("Play Viewport");
            } else {
                ui.label("Viewport");
            }
            if play_viewport && world_state == WorldState::Play {
                ui.separator();
                ui.selectable_value(&mut play_mode_view, PlayViewportKind::Gameplay, "Game");
                ui.selectable_value(&mut play_mode_view, PlayViewportKind::Editor, "Edit");
            }
            ui.separator();
            ComboBox::from_id_salt(("pane_viewport_resolution_preset", pane_id, play_viewport))
                .selected_text(render_resolution.label())
                .show_ui(ui, |ui| {
                    for preset in ViewportResolutionPreset::ALL {
                        ui.selectable_value(&mut render_resolution, preset, preset.label());
                    }
                });
            ui.separator();
            MenuButton::new("Render")
                .config(viewport_menu_config.clone())
                .ui(ui, |ui| {
                    ui.set_max_width(menu_max_width);
                    egui::ScrollArea::vertical()
                        .auto_shrink([true, true])
                        .max_height(menu_max_height)
                        .show(ui, |ui| {
                            ui.label("Render Graph");
                            for template in templates.iter() {
                                if ui
                                    .selectable_value(
                                        &mut graph_template,
                                        template.name.to_string(),
                                        template.label,
                                    )
                                    .changed()
                                {
                                    graph_selection_changed = true;
                                    //ui.close_menu();
                                }
                            }
                        });
                });
            MenuButton::new("Scripting")
                .config(viewport_menu_config.clone())
                .ui(ui, |ui| {
                    ui.set_max_width(menu_max_width);
                    egui::ScrollArea::vertical()
                        .auto_shrink([true, true])
                        .max_height(menu_max_height)
                        .show(ui, |ui| {
                            ui.checkbox(
                                &mut execute_scripts_in_edit_mode,
                                "Execute Scripts in Edit Mode",
                            );
                            if world_state == WorldState::Edit
                                && !execute_scripts_in_edit_mode
                                && ui.button("Stop All Edit Scripts").clicked()
                            {
                                if let Some(mut edit_state) =
                                    world.get_resource_mut::<ScriptEditModeState>()
                                {
                                    edit_state.queue(ScriptEditCommand::StopAll);
                                }
                            }
                            if let Some(status) = script_registry_status.as_ref() {
                                ui.label(format!("Registry: {}", status));
                            }
                            if let Some(status) = rust_status.as_ref() {
                                ui.label(format!("Rust: {}", status));
                            }
                            if script_error_count > 0 {
                                ui.colored_label(
                                    Color32::from_rgb(180, 60, 60),
                                    format!("Script Errors: {}", script_error_count),
                                );
                                for error in &script_errors {
                                    ui.label(error);
                                }
                                if script_error_count > script_errors.len() {
                                    ui.label(format!(
                                        "... {} more",
                                        script_error_count - script_errors.len()
                                    ));
                                }
                            }
                        });
                });
            MenuButton::new("Gizmos")
                .config(viewport_menu_config.clone())
                .ui(ui, |ui| {
                    ui.set_max_width(menu_max_width);
                    egui::ScrollArea::vertical()
                        .auto_shrink([true, true])
                        .max_height(menu_max_height)
                        .show(ui, |ui| {
                            ui.checkbox(&mut gizmos_in_play, "Show Gizmos in Play");
                            ui.checkbox(&mut show_camera_gizmos, "Show Camera Gizmos");
                            ui.checkbox(
                                &mut show_directional_light_gizmos,
                                "Show Directional Light Gizmos",
                            );
                            ui.checkbox(&mut show_point_light_gizmos, "Show Point Light Gizmos");
                            ui.checkbox(&mut show_spot_light_gizmos, "Show Spot Light Gizmos");
                            ui.checkbox(&mut show_spline_paths, "Show Spline Paths");
                            ui.checkbox(&mut show_spline_points, "Show Spline Points");
                            ui.separator();
                            ui.horizontal_wrapped(|ui| {
                                ui.selectable_value(&mut gizmo_mode, GizmoMode::None, "Select");
                                ui.selectable_value(&mut gizmo_mode, GizmoMode::Translate, "Move");
                                ui.selectable_value(&mut gizmo_mode, GizmoMode::Rotate, "Rotate");
                                ui.selectable_value(&mut gizmo_mode, GizmoMode::Scale, "Scale");
                            });
                        });
                });
            MenuButton::new("Advanced")
                .config(advanced_menu_config)
                .ui(ui, |ui| {
                    ui.set_max_width(advanced_menu_width);
                    egui::ScrollArea::vertical()
                        .auto_shrink([true, false])
                        .max_height(advanced_menu_max_height)
                        .show(ui, |ui| {
                            ui.collapsing("Gizmo Style", |ui| {
                                if let Some(mut gizmo_settings) =
                                    world.get_resource_mut::<EditorGizmoSettings>()
                                {
                                    if ui.button("Defaults").clicked() {
                                        *gizmo_settings = EditorGizmoSettings::default();
                                    }

                                    if gizmo_settings.size_min > gizmo_settings.size_max {
                                        let size_min = gizmo_settings.size_min;
                                        gizmo_settings.size_min = gizmo_settings.size_max;
                                        gizmo_settings.size_max = size_min;
                                    }

                                    ui.collapsing("General", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Size Scale",
                                            &mut gizmo_settings.size_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        let size_max_limit = gizmo_settings.size_max;
                                        edit_float_range(
                                            ui,
                                            "Size Min",
                                            &mut gizmo_settings.size_min,
                                            0.01,
                                            0.0..=size_max_limit,
                                        );
                                        let size_min_limit = gizmo_settings.size_min;
                                        edit_float_range(
                                            ui,
                                            "Size Max",
                                            &mut gizmo_settings.size_max,
                                            1.0,
                                            size_min_limit..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Picking", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Axis Pick Scale",
                                            &mut gizmo_settings.axis_pick_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Axis Pick Min",
                                            &mut gizmo_settings.axis_pick_radius_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Center Pick Scale",
                                            &mut gizmo_settings.center_pick_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Center Pick Min",
                                            &mut gizmo_settings.center_pick_radius_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Rotate Pick Scale",
                                            &mut gizmo_settings.rotate_pick_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Rotate Pick Min",
                                            &mut gizmo_settings.rotate_pick_radius_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Icon Pick Scale",
                                            &mut gizmo_settings.icon_pick_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Icon Pick Min",
                                            &mut gizmo_settings.icon_pick_radius_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Translate", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Thickness Scale",
                                            &mut gizmo_settings.translate_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Min",
                                            &mut gizmo_settings.translate_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Head Length Scale",
                                            &mut gizmo_settings.translate_head_length_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Head Width Scale",
                                            &mut gizmo_settings.translate_head_width_scale,
                                            0.05,
                                            0.0..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Rotate", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Ring Radius Scale",
                                            &mut gizmo_settings.rotate_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Ring Thickness Scale",
                                            &mut gizmo_settings.rotate_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Ring Thickness Min",
                                            &mut gizmo_settings.rotate_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_u32_range(
                                            ui,
                                            "Ring Segments",
                                            &mut gizmo_settings.ring_segments,
                                            1.0,
                                            3..=u32::MAX,
                                        );
                                    });

                                    ui.collapsing("Scale", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Thickness Scale",
                                            &mut gizmo_settings.scale_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Min",
                                            &mut gizmo_settings.scale_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Head Length Scale",
                                            &mut gizmo_settings.scale_head_length_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Box Scale",
                                            &mut gizmo_settings.scale_box_scale,
                                            0.05,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Scale Min",
                                            &mut gizmo_settings.scale_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Origin", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Size Scale",
                                            &mut gizmo_settings.origin_size_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Size Min",
                                            &mut gizmo_settings.origin_size_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Colors", |ui| {
                                        edit_color(ui, "Axis X", &mut gizmo_settings.axis_color_x);
                                        edit_color(ui, "Axis Y", &mut gizmo_settings.axis_color_y);
                                        edit_color(ui, "Axis Z", &mut gizmo_settings.axis_color_z);
                                        edit_color(ui, "Origin", &mut gizmo_settings.origin_color);
                                        edit_float_range(
                                            ui,
                                            "Axis Alpha",
                                            &mut gizmo_settings.axis_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Origin Alpha",
                                            &mut gizmo_settings.origin_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                    });

                                    ui.collapsing("Bounds Outline", |ui| {
                                        ui.checkbox(
                                            &mut gizmo_settings.show_bounds_outline,
                                            "Show Bounds Outline",
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Scale",
                                            &mut gizmo_settings.selection_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Min",
                                            &mut gizmo_settings.selection_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_color(
                                            ui,
                                            "Color",
                                            &mut gizmo_settings.selection_color,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Alpha",
                                            &mut gizmo_settings.selection_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                    });

                                    ui.collapsing("Mesh Outline", |ui| {
                                        ui.checkbox(
                                            &mut gizmo_settings.show_mesh_outline,
                                            "Show Mesh Outline",
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Scale",
                                            &mut gizmo_settings.outline_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Min",
                                            &mut gizmo_settings.outline_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_u32_range(
                                            ui,
                                            "Max Lines",
                                            &mut gizmo_settings.outline_max_lines,
                                            1.0,
                                            0..=u32::MAX,
                                        );
                                        edit_color(ui, "Color", &mut gizmo_settings.outline_color);
                                        edit_float_range(
                                            ui,
                                            "Alpha",
                                            &mut gizmo_settings.outline_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                    });

                                    ui.collapsing("Icons", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Icon Size Scale",
                                            &mut gizmo_settings.icon_size_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Line Thickness Scale",
                                            &mut gizmo_settings.icon_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Line Thickness Min",
                                            &mut gizmo_settings.icon_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Camera Alpha",
                                            &mut gizmo_settings.camera_icon_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Light Alpha",
                                            &mut gizmo_settings.light_icon_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                        edit_color(
                                            ui,
                                            "Camera Color",
                                            &mut gizmo_settings.camera_icon_color,
                                        );
                                        edit_color(
                                            ui,
                                            "Active Camera Color",
                                            &mut gizmo_settings.active_camera_icon_color,
                                        );
                                    });

                                    ui.collapsing("Highlight", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Hover Mix",
                                            &mut gizmo_settings.hover_mix,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Active Mix",
                                            &mut gizmo_settings.active_mix,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                    });

                                    gizmo_settings.sanitize();
                                } else {
                                    ui.label("Gizmo settings unavailable.");
                                }
                            });

                            ui.separator();
                            ui.collapsing("Skinning", |ui| {
                                if let Some(mut runtime_config) =
                                    world.get_resource_mut::<BevyRuntimeConfig>()
                                {
                                    let mut render_config = runtime_config.0.render_config;
                                    ui.label("Mode");
                                    ui.horizontal_wrapped(|ui| {
                                        ui.selectable_value(
                                            &mut render_config.skinning_mode,
                                            SkinningMode::Auto,
                                            "Auto",
                                        );
                                        ui.selectable_value(
                                            &mut render_config.skinning_mode,
                                            SkinningMode::Gpu,
                                            "GPU",
                                        );
                                        ui.selectable_value(
                                            &mut render_config.skinning_mode,
                                            SkinningMode::Cpu,
                                            "CPU",
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Palette Capacity");
                                        ui.add(
                                            DragValue::new(
                                                &mut render_config.skin_palette_capacity,
                                            )
                                            .range(0..=u32::MAX)
                                            .speed(1),
                                        );
                                    });
                                    let _ = edit_float(
                                        ui,
                                        "Palette Growth",
                                        &mut render_config.skin_palette_growth,
                                        0.01,
                                    );
                                    ui.horizontal(|ui| {
                                        ui.label("CPU Vertex Budget");
                                        ui.add(
                                            DragValue::new(
                                                &mut render_config.cpu_skinning_vertex_budget,
                                            )
                                            .range(0..=u32::MAX)
                                            .speed(1),
                                        );
                                    });
                                    runtime_config.0.render_config = render_config;
                                } else {
                                    ui.label("Runtime config unavailable.");
                                }
                            });

                            ui.separator();
                            ui.collapsing("Camera", |ui| {
                                let camera_snapshot = world
                                    .get::<BevyCamera>(camera_entity)
                                    .map(|camera| camera.0);
                                let transform_snapshot = world
                                    .get::<BevyTransform>(camera_entity)
                                    .map(|transform| transform.0);
                                if let (Some(mut camera_data), Some(mut transform_data)) =
                                    (camera_snapshot, transform_snapshot)
                                {
                                    let mut camera_changed = false;
                                    let mut transform_changed = false;

                                    let mut fov = camera_data.fov_y_rad.to_degrees();
                                    let fov_response = edit_float(ui, "FOV (deg)", &mut fov, 0.25);
                                    if fov_response.changed {
                                        camera_data.fov_y_rad = fov.to_radians();
                                        camera_changed = true;
                                    }

                                    let mut near = camera_data.near_plane;
                                    if edit_float(ui, "Near", &mut near, 0.01).changed {
                                        camera_data.near_plane = near;
                                        camera_changed = true;
                                    }

                                    let mut far = camera_data.far_plane;
                                    if edit_float(ui, "Far", &mut far, 1.0).changed {
                                        camera_data.far_plane = far;
                                        camera_changed = true;
                                    }

                                    let mut position = transform_data.position;
                                    if edit_vec3(ui, "Position", &mut position, 0.1).changed {
                                        transform_data.position = position;
                                        transform_changed = true;
                                    }

                                    if camera_changed {
                                        if let Some(mut camera) =
                                            world.get_mut::<BevyCamera>(camera_entity)
                                        {
                                            camera.0 = camera_data;
                                        }
                                    }
                                    if transform_changed {
                                        if let Some(mut transform) =
                                            world.get_mut::<BevyTransform>(camera_entity)
                                        {
                                            transform.0 = transform_data;
                                        }
                                    }
                                } else {
                                    ui.label("Viewport camera missing.");
                                }
                            });
                        });
                });
            ui.separator();
            if let Some(selected_camera) = selected_camera {
                if ui
                    .button("P")
                    .on_hover_text("Pin selected camera")
                    .clicked()
                {
                    pinned_camera = Some(selected_camera);
                }
            }
            if pinned_camera.is_some() && ui.button("U").on_hover_text("Unpin camera").clicked() {
                pinned_camera = None;
            }
            if let Some(entity) = pinned_camera {
                ui.label(format!("Pinned: {}", entity_display_name(world, entity)));
            }
        });
        ui.separator();
        let resolution_preset_changed = render_resolution != previous_render_resolution;

        let available = ui.available_size_before_wrap();
        let viewport_size = Vec2::new(available.x.max(128.0), available.y.max(128.0));
        let (main_rect, main_response) =
            ui.allocate_exact_size(viewport_size, Sense::click_and_drag());

        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let pixels_per_point = ui.ctx().pixels_per_point();
        let canvas_target_size = [
            (main_rect.width() * pixels_per_point).round().max(1.0) as u32,
            (main_rect.height() * pixels_per_point).round().max(1.0) as u32,
        ];
        let target_size = render_resolution.target_size(canvas_target_size);
        let target_aspect = (target_size[0] as f32 / target_size[1].max(1) as f32).max(0.1);
        let scene_rect = fit_rect_to_aspect(main_rect, target_aspect);

        ui.painter()
            .rect_filled(main_rect, 6.0, Color32::from_rgb(8, 10, 12));
        ui.painter()
            .image(texture_id, scene_rect, uv, Color32::WHITE);
        if play_viewport && world_state != WorldState::Play {
            ui.painter().text(
                scene_rect.center(),
                Align2::CENTER_CENTER,
                "play mode inactive",
                FontId::proportional(13.0),
                Color32::from_gray(210),
            );
        }

        let mut preview_rect_pixels = None;
        let mut shown_preview_camera = None;
        let mut preview_pointer_blocked = false;
        let preview_host_active = world
            .get_resource::<EditorViewportRuntime>()
            .is_some_and(|runtime| runtime.active_pane_id == Some(pane_id));
        if preview_host_active {
            if let (Some(preview_texture_id), Some(preview_entity)) = (
                preview_texture_id,
                resolve_preview_camera_for_viewport(world),
            ) {
                if camera_entity != preview_entity {
                    let preview_aspect = world
                        .get::<BevyCamera>(preview_entity)
                        .map(|camera| camera.0.aspect_ratio)
                        .unwrap_or(16.0 / 9.0)
                        .max(0.1);
                    let scene_w = scene_rect.width().max(1.0);
                    let scene_h = scene_rect.height().max(1.0);
                    let min_preview_w = scene_w.min(120.0).max(72.0);
                    let mut preview_w = (scene_w * preview_width_norm).max(min_preview_w);
                    let mut preview_h = (preview_w / preview_aspect).max(48.0);
                    let max_preview_h = scene_h.max(48.0);
                    if preview_h > max_preview_h {
                        preview_h = max_preview_h;
                        preview_w = (preview_h * preview_aspect).max(min_preview_w);
                    }
                    preview_w = preview_w.min(scene_w);
                    preview_h = preview_h.min(scene_h);

                    let max_offset_x = (scene_w - preview_w).max(0.0);
                    let max_offset_y = (scene_h - preview_h).max(0.0);
                    let mut preview_x =
                        scene_rect.min.x + preview_position_norm[0].clamp(0.0, 1.0) * max_offset_x;
                    let mut preview_y =
                        scene_rect.min.y + preview_position_norm[1].clamp(0.0, 1.0) * max_offset_y;
                    if !preview_x.is_finite() || !preview_y.is_finite() {
                        preview_x = scene_rect.min.x;
                        preview_y = scene_rect.min.y;
                    }
                    preview_x = preview_x.clamp(scene_rect.min.x, scene_rect.max.x - preview_w);
                    preview_y = preview_y.clamp(scene_rect.min.y, scene_rect.max.y - preview_h);

                    let mut preview_rect = Rect::from_min_size(
                        Pos2::new(preview_x, preview_y),
                        Vec2::new(preview_w, preview_h),
                    );
                    let preview_move_id = ui.id().with(("pane_viewport_preview_move", pane_id));
                    let preview_resize_id = ui.id().with(("pane_viewport_preview_resize", pane_id));
                    let resize_handle = 14.0;
                    let resize_rect = Rect::from_min_max(
                        Pos2::new(
                            preview_rect.max.x - resize_handle,
                            preview_rect.max.y - resize_handle,
                        ),
                        preview_rect.max,
                    );
                    let move_response =
                        ui.interact(preview_rect, preview_move_id, Sense::click_and_drag());
                    let resize_response = ui.interact(
                        resize_rect.expand(2.0),
                        preview_resize_id,
                        Sense::click_and_drag(),
                    );
                    preview_pointer_blocked = move_response.hovered() || resize_response.hovered();
                    let pointer_delta = ui.ctx().input(|input| input.pointer.delta());

                    if resize_response.dragged_by(PointerButton::Primary) {
                        let resize_delta = pointer_delta.x + pointer_delta.y * preview_aspect;
                        let min_w = scene_w.min(96.0).max(48.0);
                        let max_w_from_pos = (scene_rect.max.x - preview_rect.min.x)
                            .min((scene_rect.max.y - preview_rect.min.y).max(1.0) * preview_aspect);
                        preview_w = (preview_rect.width() + resize_delta * 0.5)
                            .clamp(min_w.min(max_w_from_pos), max_w_from_pos.max(min_w));
                        preview_h = (preview_w / preview_aspect).max(1.0);
                        preview_rect =
                            Rect::from_min_size(preview_rect.min, Vec2::new(preview_w, preview_h));
                    } else if move_response.dragged_by(PointerButton::Primary) {
                        let mut min = preview_rect.min + pointer_delta;
                        min.x = min
                            .x
                            .clamp(scene_rect.min.x, scene_rect.max.x - preview_rect.width());
                        min.y = min
                            .y
                            .clamp(scene_rect.min.y, scene_rect.max.y - preview_rect.height());
                        preview_rect = Rect::from_min_size(min, preview_rect.size());
                    }

                    let offset_x_den = (scene_w - preview_rect.width()).max(1.0);
                    let offset_y_den = (scene_h - preview_rect.height()).max(1.0);
                    preview_position_norm = [
                        ((preview_rect.min.x - scene_rect.min.x) / offset_x_den).clamp(0.0, 1.0),
                        ((preview_rect.min.y - scene_rect.min.y) / offset_y_den).clamp(0.0, 1.0),
                    ];
                    preview_width_norm = (preview_rect.width() / scene_w).clamp(0.05, 0.95);

                    ui.painter().rect_filled(
                        preview_rect.expand(2.0),
                        4.0,
                        Color32::from_black_alpha(200),
                    );
                    ui.painter()
                        .image(preview_texture_id, preview_rect, uv, Color32::WHITE);
                    ui.painter().text(
                        preview_rect.min + Vec2::new(6.0, 6.0),
                        Align2::LEFT_TOP,
                        entity_display_name(world, preview_entity),
                        FontId::proportional(11.0),
                        Color32::from_rgb(235, 235, 235),
                    );
                    let handle_color = if resize_response.hovered()
                        || resize_response.dragged_by(PointerButton::Primary)
                    {
                        Color32::from_rgb(220, 220, 220)
                    } else {
                        Color32::from_gray(180)
                    };
                    ui.painter().rect_filled(
                        resize_rect.shrink(1.0),
                        2.0,
                        Color32::from_black_alpha(150),
                    );
                    ui.painter().line_segment(
                        [
                            resize_rect.left_bottom() + Vec2::new(3.0, -3.0),
                            resize_rect.right_top() + Vec2::new(-3.0, 3.0),
                        ],
                        Stroke::new(1.5, handle_color),
                    );

                    preview_rect_pixels =
                        viewport_rect_pixels_from_ui_rect(preview_rect, pixels_per_point);
                    shown_preview_camera = Some(preview_entity);
                }
            }
        }

        let pointer_pos = ui.ctx().input(|input| {
            input
                .pointer
                .interact_pos()
                .or_else(|| input.pointer.hover_pos())
        });
        let pointer_over = pointer_pos.is_some_and(|pointer_pos| scene_rect.contains(pointer_pos))
            && main_response.hovered()
            && !preview_pointer_blocked
            && pointer_pos
                .and_then(|pos| ui.ctx().layer_id_at(pos))
                .map_or(true, |layer| layer == ui.layer_id());

        let rect_pixels = viewport_rect_pixels_from_ui_rect(scene_rect, pixels_per_point);
        if let Some(rect_pixels) = rect_pixels {
            let should_render = !play_viewport || world_state == WorldState::Play;
            if should_render {
                if let Some(mut runtime) = world.get_resource_mut::<EditorViewportRuntime>() {
                    runtime
                        .pane_requests
                        .push(crate::editor::EditorViewportPaneRequest {
                            pane_id,
                            camera_entity,
                            texture_id,
                            viewport_rect: rect_pixels,
                            pointer_over,
                            target_size,
                            temporal_history: true,
                            immediate_resize: resolution_preset_changed,
                            graph_template: graph_template.clone(),
                            gizmos_in_play,
                            show_camera_gizmos,
                            show_directional_light_gizmos,
                            show_point_light_gizmos,
                            show_spot_light_gizmos,
                            show_spline_paths,
                            show_spline_points,
                        });

                    let secondary_pressed_here = ui.ctx().input(|input| {
                        input.pointer.button_pressed(PointerButton::Secondary)
                            && input
                                .pointer
                                .interact_pos()
                                .or_else(|| input.pointer.hover_pos())
                                .is_some_and(|pos| scene_rect.contains(pos))
                    });
                    let clicked_here = (main_response.clicked_by(PointerButton::Primary)
                        || main_response.clicked_by(PointerButton::Secondary)
                        || secondary_pressed_here)
                        && pointer_over;
                    if clicked_here {
                        runtime.active_pane_id = Some(pane_id);
                        runtime.keyboard_focus = true;
                    }
                    let right_down_here = ui.ctx().input(|input| {
                        input.pointer.button_down(PointerButton::Secondary)
                            && input
                                .pointer
                                .interact_pos()
                                .or_else(|| input.pointer.hover_pos())
                                .is_some_and(|pos| scene_rect.contains(pos))
                    });
                    if right_down_here {
                        runtime.active_pane_id = Some(pane_id);
                        runtime.keyboard_focus = true;
                        runtime.active_camera_entity = Some(camera_entity);
                        runtime.main_rect_pixels = Some(rect_pixels);
                        runtime.pointer_over_main = true;
                    }
                    if runtime.active_pane_id.is_none() {
                        runtime.active_pane_id = Some(pane_id);
                        runtime.keyboard_focus = false;
                    }
                    if runtime.active_pane_id == Some(pane_id) {
                        runtime.active_camera_entity = Some(camera_entity);
                        runtime.main_rect_pixels = Some(rect_pixels);
                        runtime.pointer_over_main = pointer_over;
                        runtime.preview_texture_id = preview_texture_id;
                        runtime.preview_rect_pixels = preview_rect_pixels;
                        runtime.preview_camera_entity = shown_preview_camera;
                        if ui.ctx().input(|input| input.pointer.any_pressed()) && !pointer_over {
                            runtime.keyboard_focus = false;
                        }
                    }
                }
            }
        }

        let (active_pane, keyboard_focus) = world
            .get_resource::<EditorViewportRuntime>()
            .map(|runtime| {
                (
                    runtime.active_pane_id == Some(pane_id),
                    runtime.keyboard_focus,
                )
            })
            .unwrap_or((false, false));
        let pointer_passthrough = pointer_over;
        let keyboard_passthrough = active_pane && keyboard_focus;
        if pointer_passthrough || keyboard_passthrough {
            if let Some(mut passthrough) = world.get_resource_mut::<EguiInputPassthrough>() {
                passthrough.pointer |= pointer_passthrough;
                passthrough.keyboard |= keyboard_passthrough;
            }
        }

        if let Some(mut gizmo_state) = world.get_resource_mut::<EditorGizmoState>() {
            gizmo_state.mode = gizmo_mode;
        }

        if let Some(mut viewport_state) = world.get_resource_mut::<EditorViewportState>() {
            viewport_state.graph_template = graph_template.clone();
            viewport_state.play_mode_view = play_mode_view;
            viewport_state.render_resolution = render_resolution;
            viewport_state.pinned_camera = pinned_camera;
            viewport_state.preview_position_norm = preview_position_norm;
            viewport_state.preview_width_norm = preview_width_norm;
            viewport_state.gizmos_in_play = gizmos_in_play;
            viewport_state.execute_scripts_in_edit_mode = execute_scripts_in_edit_mode;
            viewport_state.show_camera_gizmos = show_camera_gizmos;
            viewport_state.show_directional_light_gizmos = show_directional_light_gizmos;
            viewport_state.show_point_light_gizmos = show_point_light_gizmos;
            viewport_state.show_spot_light_gizmos = show_spot_light_gizmos;
            viewport_state.show_spline_paths = show_spline_paths;
            viewport_state.show_spline_points = show_spline_points;
            viewport_state.show_options_panel = show_options_panel;
        }

        if graph_selection_changed || previous_graph_template != graph_template {
            if let Some(template) = template_for_graph(&graph_template) {
                if let Some(mut graph_res) =
                    world.get_resource_mut::<helmer_becs::systems::render_system::RenderGraphResource>()
                {
                    graph_res.0 = (template.build)();
                }
                if let Some(mut render_sync) =
                    world
                        .get_resource_mut::<helmer_becs::systems::render_system::RenderSyncRequest>(
                        )
                {
                    render_sync.request_with_epoch(3);
                }
                if let Some(mut refresh) =
                    world.get_resource_mut::<crate::editor::scene::EditorRenderRefresh>()
                {
                    refresh.pending = true;
                }
            }
        }

        if let Some(mut pane_state) = world.get_resource_mut::<EditorPaneViewportState>() {
            if resolution_preset_changed {
                pane_state.resolutions.insert(pane_id, render_resolution);
            }
            pane_state.settings.insert(
                pane_id,
                EditorPaneViewportSettings {
                    graph_template: graph_template.clone(),
                    gizmos_in_play,
                    show_camera_gizmos,
                    show_directional_light_gizmos,
                    show_point_light_gizmos,
                    show_spot_light_gizmos,
                    show_spline_paths,
                    show_spline_points,
                },
            );
        }
    });
}

pub fn draw_viewport_window(ui: &mut Ui, world: &mut World) {
    with_middle_drag_blocked(ui, world, |ui, world| {
        let world_state = world
            .get_resource::<EditorSceneState>()
            .map(|state| state.world_state)
            .unwrap_or(WorldState::Edit);
        let mut play_mode_view = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.play_mode_view)
            .unwrap_or_default();
        let show_options_panel = false;
        let mut render_resolution = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.render_resolution)
            .unwrap_or_default();
        let previous_render_resolution = render_resolution;
        let templates = graph_templates();
        let mut graph_template = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.graph_template.clone())
            .unwrap_or_else(|| {
                templates
                    .first()
                    .map(|template| template.name.to_string())
                    .unwrap_or_else(|| "default-graph".to_string())
            });
        let previous_graph_template = graph_template.clone();
        let mut pinned_camera = world
            .get_resource::<EditorViewportState>()
            .and_then(|state| state.pinned_camera);
        let mut gizmos_in_play = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.gizmos_in_play)
            .unwrap_or(false);
        let mut execute_scripts_in_edit_mode = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.execute_scripts_in_edit_mode)
            .unwrap_or(false);
        let mut show_camera_gizmos = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.show_camera_gizmos)
            .unwrap_or(true);
        let mut show_directional_light_gizmos = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.show_directional_light_gizmos)
            .unwrap_or(true);
        let mut show_point_light_gizmos = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.show_point_light_gizmos)
            .unwrap_or(true);
        let mut show_spot_light_gizmos = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.show_spot_light_gizmos)
            .unwrap_or(true);
        let mut show_spline_paths = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.show_spline_paths)
            .unwrap_or(true);
        let mut show_spline_points = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.show_spline_points)
            .unwrap_or(true);
        let mut gizmo_mode = world
            .get_resource::<EditorGizmoState>()
            .map(|state| state.mode)
            .unwrap_or(GizmoMode::None);
        let mut preview_position_norm = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.preview_position_norm)
            .unwrap_or([0.03, 0.74]);
        let mut preview_width_norm = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.preview_width_norm)
            .unwrap_or(0.28);
        let script_registry_status = world
            .get_resource::<ScriptRegistry>()
            .and_then(|registry| registry.status.clone());
        let (rust_status, script_errors, script_error_count) =
            if let Some(runtime) = world.get_resource::<ScriptRuntime>() {
                (
                    runtime.rust_status.clone(),
                    runtime.errors.iter().take(3).cloned().collect::<Vec<_>>(),
                    runtime.errors.len(),
                )
            } else {
                (None, Vec::new(), 0)
            };
        let menu_max_height = (ui.ctx().content_rect().height() * 0.62).clamp(170.0, 520.0);
        let menu_max_width = (ui.ctx().content_rect().width() * 0.26).clamp(150.0, 340.0);
        let advanced_menu_max_height =
            (ui.ctx().content_rect().height() * 0.78).clamp(220.0, 760.0);
        let advanced_menu_width = (ui.ctx().content_rect().width() * 0.36).clamp(220.0, 460.0);
        let viewport_menu_config =
            MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside);
        let advanced_menu_config =
            MenuConfig::new().close_behavior(PopupCloseBehavior::IgnoreClicks);
        let mut graph_selection_changed = false;

        let selected_camera = world
            .get_resource::<InspectorSelectedEntityResource>()
            .and_then(|selection| selection.0)
            .filter(|entity| world.get::<BevyCamera>(*entity).is_some());

        ui.horizontal_wrapped(|ui| {
            ui.label("Viewport");
            if world_state == WorldState::Play {
                ui.separator();
                ui.selectable_value(&mut play_mode_view, PlayViewportKind::Gameplay, "Game");
                ui.selectable_value(&mut play_mode_view, PlayViewportKind::Editor, "Edit");
            }
            ui.separator();
            ComboBox::from_id_source("viewport_resolution_preset")
                .selected_text(render_resolution.label())
                .show_ui(ui, |ui| {
                    for preset in ViewportResolutionPreset::ALL {
                        ui.selectable_value(&mut render_resolution, preset, preset.label());
                    }
                });
            ui.separator();
            MenuButton::new("Render")
                .config(viewport_menu_config.clone())
                .ui(ui, |ui| {
                    ui.set_max_width(menu_max_width);
                    egui::ScrollArea::vertical()
                        .auto_shrink([true, true])
                        .max_height(menu_max_height)
                        .show(ui, |ui| {
                            ui.label("Render Graph");
                            for template in templates.iter() {
                                if ui
                                    .selectable_value(
                                        &mut graph_template,
                                        template.name.to_string(),
                                        template.label,
                                    )
                                    .changed()
                                {
                                    graph_selection_changed = true;
                                    //ui.close_menu();
                                }
                            }
                        });
                });
            MenuButton::new("Scripting")
                .config(viewport_menu_config.clone())
                .ui(ui, |ui| {
                    ui.set_max_width(menu_max_width);
                    egui::ScrollArea::vertical()
                        .auto_shrink([true, true])
                        .max_height(menu_max_height)
                        .show(ui, |ui| {
                            ui.checkbox(
                                &mut execute_scripts_in_edit_mode,
                                "Execute Scripts in Edit Mode",
                            );
                            if world_state == WorldState::Edit
                                && !execute_scripts_in_edit_mode
                                && ui.button("Stop All Edit Scripts").clicked()
                            {
                                if let Some(mut edit_state) =
                                    world.get_resource_mut::<ScriptEditModeState>()
                                {
                                    edit_state.queue(ScriptEditCommand::StopAll);
                                }
                            }
                            if let Some(status) = script_registry_status.as_ref() {
                                ui.label(format!("Registry: {}", status));
                            }
                            if let Some(status) = rust_status.as_ref() {
                                ui.label(format!("Rust: {}", status));
                            }
                            if script_error_count > 0 {
                                ui.colored_label(
                                    Color32::from_rgb(180, 60, 60),
                                    format!("Script Errors: {}", script_error_count),
                                );
                                for error in &script_errors {
                                    ui.label(error);
                                }
                                if script_error_count > script_errors.len() {
                                    ui.label(format!(
                                        "... {} more",
                                        script_error_count - script_errors.len()
                                    ));
                                }
                            }
                        });
                });
            MenuButton::new("Gizmos")
                .config(viewport_menu_config.clone())
                .ui(ui, |ui| {
                    ui.set_max_width(menu_max_width);
                    egui::ScrollArea::vertical()
                        .auto_shrink([true, true])
                        .max_height(menu_max_height)
                        .show(ui, |ui| {
                            ui.checkbox(&mut gizmos_in_play, "Show Gizmos in Play");
                            ui.checkbox(&mut show_camera_gizmos, "Show Camera Gizmos");
                            ui.checkbox(
                                &mut show_directional_light_gizmos,
                                "Show Directional Light Gizmos",
                            );
                            ui.checkbox(&mut show_point_light_gizmos, "Show Point Light Gizmos");
                            ui.checkbox(&mut show_spot_light_gizmos, "Show Spot Light Gizmos");
                            ui.checkbox(&mut show_spline_paths, "Show Spline Paths");
                            ui.checkbox(&mut show_spline_points, "Show Spline Points");
                            ui.separator();
                            ui.horizontal_wrapped(|ui| {
                                ui.selectable_value(&mut gizmo_mode, GizmoMode::None, "Select");
                                ui.selectable_value(&mut gizmo_mode, GizmoMode::Translate, "Move");
                                ui.selectable_value(&mut gizmo_mode, GizmoMode::Rotate, "Rotate");
                                ui.selectable_value(&mut gizmo_mode, GizmoMode::Scale, "Scale");
                            });
                        });
                });
            MenuButton::new("Advanced")
                .config(advanced_menu_config)
                .ui(ui, |ui| {
                    ui.set_max_width(advanced_menu_width);
                    egui::ScrollArea::vertical()
                        .auto_shrink([true, false])
                        .max_height(advanced_menu_max_height)
                        .show(ui, |ui| {
                            ui.collapsing("Gizmo Style", |ui| {
                                if let Some(mut gizmo_settings) =
                                    world.get_resource_mut::<EditorGizmoSettings>()
                                {
                                    if ui.button("Defaults").clicked() {
                                        *gizmo_settings = EditorGizmoSettings::default();
                                    }

                                    if gizmo_settings.size_min > gizmo_settings.size_max {
                                        let size_min = gizmo_settings.size_min;
                                        gizmo_settings.size_min = gizmo_settings.size_max;
                                        gizmo_settings.size_max = size_min;
                                    }

                                    ui.collapsing("General", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Size Scale",
                                            &mut gizmo_settings.size_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        let size_max_limit = gizmo_settings.size_max;
                                        edit_float_range(
                                            ui,
                                            "Size Min",
                                            &mut gizmo_settings.size_min,
                                            0.01,
                                            0.0..=size_max_limit,
                                        );
                                        let size_min_limit = gizmo_settings.size_min;
                                        edit_float_range(
                                            ui,
                                            "Size Max",
                                            &mut gizmo_settings.size_max,
                                            1.0,
                                            size_min_limit..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Picking", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Axis Pick Scale",
                                            &mut gizmo_settings.axis_pick_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Axis Pick Min",
                                            &mut gizmo_settings.axis_pick_radius_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Center Pick Scale",
                                            &mut gizmo_settings.center_pick_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Center Pick Min",
                                            &mut gizmo_settings.center_pick_radius_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Rotate Pick Scale",
                                            &mut gizmo_settings.rotate_pick_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Rotate Pick Min",
                                            &mut gizmo_settings.rotate_pick_radius_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Icon Pick Scale",
                                            &mut gizmo_settings.icon_pick_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Icon Pick Min",
                                            &mut gizmo_settings.icon_pick_radius_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Translate", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Thickness Scale",
                                            &mut gizmo_settings.translate_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Min",
                                            &mut gizmo_settings.translate_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Head Length Scale",
                                            &mut gizmo_settings.translate_head_length_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Head Width Scale",
                                            &mut gizmo_settings.translate_head_width_scale,
                                            0.05,
                                            0.0..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Rotate", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Ring Radius Scale",
                                            &mut gizmo_settings.rotate_radius_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Ring Thickness Scale",
                                            &mut gizmo_settings.rotate_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Ring Thickness Min",
                                            &mut gizmo_settings.rotate_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_u32_range(
                                            ui,
                                            "Ring Segments",
                                            &mut gizmo_settings.ring_segments,
                                            1.0,
                                            3..=u32::MAX,
                                        );
                                    });

                                    ui.collapsing("Scale", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Thickness Scale",
                                            &mut gizmo_settings.scale_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Min",
                                            &mut gizmo_settings.scale_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Head Length Scale",
                                            &mut gizmo_settings.scale_head_length_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Box Scale",
                                            &mut gizmo_settings.scale_box_scale,
                                            0.05,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Scale Min",
                                            &mut gizmo_settings.scale_min,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Origin", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Size Scale",
                                            &mut gizmo_settings.origin_size_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Size Min",
                                            &mut gizmo_settings.origin_size_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                    });

                                    ui.collapsing("Colors", |ui| {
                                        edit_color(ui, "Axis X", &mut gizmo_settings.axis_color_x);
                                        edit_color(ui, "Axis Y", &mut gizmo_settings.axis_color_y);
                                        edit_color(ui, "Axis Z", &mut gizmo_settings.axis_color_z);
                                        edit_color(ui, "Origin", &mut gizmo_settings.origin_color);
                                        edit_float_range(
                                            ui,
                                            "Axis Alpha",
                                            &mut gizmo_settings.axis_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Origin Alpha",
                                            &mut gizmo_settings.origin_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                    });

                                    ui.collapsing("Bounds Outline", |ui| {
                                        ui.checkbox(
                                            &mut gizmo_settings.show_bounds_outline,
                                            "Show Bounds Outline",
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Scale",
                                            &mut gizmo_settings.selection_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Min",
                                            &mut gizmo_settings.selection_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_color(
                                            ui,
                                            "Color",
                                            &mut gizmo_settings.selection_color,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Alpha",
                                            &mut gizmo_settings.selection_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                    });

                                    ui.collapsing("Mesh Outline", |ui| {
                                        ui.checkbox(
                                            &mut gizmo_settings.show_mesh_outline,
                                            "Show Mesh Outline",
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Scale",
                                            &mut gizmo_settings.outline_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Thickness Min",
                                            &mut gizmo_settings.outline_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_u32_range(
                                            ui,
                                            "Max Lines",
                                            &mut gizmo_settings.outline_max_lines,
                                            1.0,
                                            0..=u32::MAX,
                                        );
                                        edit_color(ui, "Color", &mut gizmo_settings.outline_color);
                                        edit_float_range(
                                            ui,
                                            "Alpha",
                                            &mut gizmo_settings.outline_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                    });

                                    ui.collapsing("Icons", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Icon Size Scale",
                                            &mut gizmo_settings.icon_size_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Line Thickness Scale",
                                            &mut gizmo_settings.icon_thickness_scale,
                                            0.01,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Line Thickness Min",
                                            &mut gizmo_settings.icon_thickness_min,
                                            0.005,
                                            0.0..=f32::MAX,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Camera Alpha",
                                            &mut gizmo_settings.camera_icon_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Light Alpha",
                                            &mut gizmo_settings.light_icon_alpha,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                        edit_color(
                                            ui,
                                            "Camera Color",
                                            &mut gizmo_settings.camera_icon_color,
                                        );
                                        edit_color(
                                            ui,
                                            "Active Camera Color",
                                            &mut gizmo_settings.active_camera_icon_color,
                                        );
                                    });

                                    ui.collapsing("Highlight", |ui| {
                                        edit_float_range(
                                            ui,
                                            "Hover Mix",
                                            &mut gizmo_settings.hover_mix,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                        edit_float_range(
                                            ui,
                                            "Active Mix",
                                            &mut gizmo_settings.active_mix,
                                            0.01,
                                            0.0..=1.0,
                                        );
                                    });

                                    gizmo_settings.sanitize();
                                } else {
                                    ui.label("Gizmo settings unavailable.");
                                }
                            });

                            ui.separator();
                            ui.collapsing("Skinning", |ui| {
                                if let Some(mut runtime_config) =
                                    world.get_resource_mut::<BevyRuntimeConfig>()
                                {
                                    let mut render_config = runtime_config.0.render_config;
                                    ui.label("Mode");
                                    ui.horizontal_wrapped(|ui| {
                                        ui.selectable_value(
                                            &mut render_config.skinning_mode,
                                            SkinningMode::Auto,
                                            "Auto",
                                        );
                                        ui.selectable_value(
                                            &mut render_config.skinning_mode,
                                            SkinningMode::Gpu,
                                            "GPU",
                                        );
                                        ui.selectable_value(
                                            &mut render_config.skinning_mode,
                                            SkinningMode::Cpu,
                                            "CPU",
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Palette Capacity");
                                        ui.add(
                                            DragValue::new(
                                                &mut render_config.skin_palette_capacity,
                                            )
                                            .range(0..=u32::MAX)
                                            .speed(1),
                                        );
                                    });
                                    let _ = edit_float(
                                        ui,
                                        "Palette Growth",
                                        &mut render_config.skin_palette_growth,
                                        0.01,
                                    );
                                    ui.horizontal(|ui| {
                                        ui.label("CPU Vertex Budget");
                                        ui.add(
                                            DragValue::new(
                                                &mut render_config.cpu_skinning_vertex_budget,
                                            )
                                            .range(0..=u32::MAX)
                                            .speed(1),
                                        );
                                    });
                                    runtime_config.0.render_config = render_config;
                                } else {
                                    ui.label("Runtime config unavailable.");
                                }
                            });

                            ui.separator();
                            ui.collapsing("Camera", |ui| {
                                let mut camera_query = world.query_filtered::<
                                (&mut BevyCamera, &mut BevyTransform),
                                With<EditorViewportCamera>,
                            >();
                                if let Some((mut camera, mut transform)) =
                                    camera_query.iter_mut(world).next()
                                {
                                    let camera = &mut camera.0;
                                    let transform = &mut transform.0;

                                    let mut fov = camera.fov_y_rad.to_degrees();
                                    let fov_response = edit_float(ui, "FOV (deg)", &mut fov, 0.25);
                                    if fov_response.changed {
                                        camera.fov_y_rad = fov.to_radians();
                                    }
                                    let _ = edit_float(ui, "Near", &mut camera.near_plane, 0.01);
                                    let _ = edit_float(ui, "Far", &mut camera.far_plane, 1.0);

                                    let mut position = transform.position;
                                    let position_response =
                                        edit_vec3(ui, "Position", &mut position, 0.1);
                                    if position_response.changed {
                                        transform.position = position;
                                    }
                                } else {
                                    ui.label("Viewport camera missing.");
                                }
                            });
                        });
                });
            ui.separator();
            if let Some(selected_camera) = selected_camera {
                if ui
                    .button("P")
                    .on_hover_text("Pin selected camera")
                    .clicked()
                {
                    pinned_camera = Some(selected_camera);
                }
            }
            if pinned_camera.is_some() && ui.button("U").on_hover_text("Unpin camera").clicked() {
                pinned_camera = None;
            }
            if let Some(entity) = pinned_camera {
                ui.label(format!("Pinned: {}", entity_display_name(world, entity)));
            }
        });
        ui.separator();
        let resolution_preset_changed = render_resolution != previous_render_resolution;

        if let Some((editor_texture_id, gameplay_texture_id, preview_texture_id)) =
            ensure_viewport_texture_ids(ui, world)
        {
            let editor_camera_entity = first_camera_with_component::<EditorViewportCamera>(world);
            let gameplay_camera_entity = first_camera_with_component::<EditorPlayCamera>(world);
            let main_camera_entity = if world_state == WorldState::Play
                && play_mode_view == PlayViewportKind::Gameplay
            {
                gameplay_camera_entity.or(editor_camera_entity)
            } else {
                editor_camera_entity
            };
            let main_texture_id = if world_state == WorldState::Play
                && play_mode_view == PlayViewportKind::Gameplay
                && gameplay_camera_entity.is_some()
            {
                gameplay_texture_id
            } else {
                editor_texture_id
            };

            let available = ui.available_size_before_wrap();
            let viewport_size = Vec2::new(available.x.max(128.0), available.y.max(128.0));
            let (main_rect, main_response) =
                ui.allocate_exact_size(viewport_size, Sense::click_and_drag());

            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            let pixels_per_point = ui.ctx().pixels_per_point();
            let canvas_target_size = [
                (main_rect.width() * pixels_per_point).round().max(1.0) as u32,
                (main_rect.height() * pixels_per_point).round().max(1.0) as u32,
            ];
            let main_target_size = render_resolution.target_size(canvas_target_size);
            let main_target_aspect =
                (main_target_size[0] as f32 / main_target_size[1].max(1) as f32).max(0.1);
            let scene_rect = fit_rect_to_aspect(main_rect, main_target_aspect);
            ui.painter()
                .rect_filled(main_rect, 6.0, Color32::from_rgb(8, 10, 12));
            ui.painter()
                .image(main_texture_id, scene_rect, uv, Color32::WHITE);
            let mode_label = if world_state == WorldState::Play {
                format!("{} view", play_mode_view.as_str())
            } else {
                "editor view".to_string()
            };
            ui.painter().text(
                scene_rect.min + Vec2::new(10.0, 10.0),
                Align2::LEFT_TOP,
                mode_label,
                FontId::proportional(12.0),
                Color32::from_rgb(220, 230, 240),
            );

            let mut preview_rect_pixels = None;
            let mut shown_preview_camera = None;
            let mut preview_pointer_blocked = false;
            let preview_camera = resolve_preview_camera_for_viewport(world);
            if let Some(preview_entity) = preview_camera {
                if main_camera_entity != Some(preview_entity) {
                    let preview_aspect = world
                        .get::<BevyCamera>(preview_entity)
                        .map(|camera| camera.0.aspect_ratio)
                        .unwrap_or(16.0 / 9.0)
                        .max(0.1);
                    let scene_w = scene_rect.width().max(1.0);
                    let scene_h = scene_rect.height().max(1.0);
                    let min_preview_w = scene_w.min(120.0).max(72.0);
                    let mut preview_w = (scene_w * preview_width_norm).max(min_preview_w);
                    let mut preview_h = (preview_w / preview_aspect).max(48.0);
                    let max_preview_h = scene_h.max(48.0);
                    if preview_h > max_preview_h {
                        preview_h = max_preview_h;
                        preview_w = (preview_h * preview_aspect).max(min_preview_w);
                    }
                    preview_w = preview_w.min(scene_w);
                    preview_h = preview_h.min(scene_h);

                    let max_offset_x = (scene_w - preview_w).max(0.0);
                    let max_offset_y = (scene_h - preview_h).max(0.0);
                    let mut preview_x =
                        scene_rect.min.x + preview_position_norm[0].clamp(0.0, 1.0) * max_offset_x;
                    let mut preview_y =
                        scene_rect.min.y + preview_position_norm[1].clamp(0.0, 1.0) * max_offset_y;
                    if !preview_x.is_finite() || !preview_y.is_finite() {
                        preview_x = scene_rect.min.x;
                        preview_y = scene_rect.min.y;
                    }
                    preview_x = preview_x.clamp(scene_rect.min.x, scene_rect.max.x - preview_w);
                    preview_y = preview_y.clamp(scene_rect.min.y, scene_rect.max.y - preview_h);

                    let mut preview_rect = Rect::from_min_size(
                        Pos2::new(preview_x, preview_y),
                        Vec2::new(preview_w, preview_h),
                    );
                    let preview_move_id = ui.id().with("viewport_preview_move");
                    let preview_resize_id = ui.id().with("viewport_preview_resize");
                    let resize_handle = 14.0;
                    let resize_rect = Rect::from_min_max(
                        Pos2::new(
                            preview_rect.max.x - resize_handle,
                            preview_rect.max.y - resize_handle,
                        ),
                        preview_rect.max,
                    );
                    let move_response =
                        ui.interact(preview_rect, preview_move_id, Sense::click_and_drag());
                    let resize_response = ui.interact(
                        resize_rect.expand(2.0),
                        preview_resize_id,
                        Sense::click_and_drag(),
                    );
                    preview_pointer_blocked = move_response.hovered() || resize_response.hovered();
                    let pointer_delta = ui.ctx().input(|input| input.pointer.delta());

                    if resize_response.dragged_by(PointerButton::Primary) {
                        let resize_delta = pointer_delta.x + pointer_delta.y * preview_aspect;
                        let min_w = scene_w.min(96.0).max(48.0);
                        let max_w_from_pos = (scene_rect.max.x - preview_rect.min.x)
                            .min((scene_rect.max.y - preview_rect.min.y).max(1.0) * preview_aspect);
                        preview_w = (preview_rect.width() + resize_delta * 0.5)
                            .clamp(min_w.min(max_w_from_pos), max_w_from_pos.max(min_w));
                        preview_h = (preview_w / preview_aspect).max(1.0);
                        preview_rect =
                            Rect::from_min_size(preview_rect.min, Vec2::new(preview_w, preview_h));
                    } else if move_response.dragged_by(PointerButton::Primary) {
                        let mut min = preview_rect.min + pointer_delta;
                        min.x = min
                            .x
                            .clamp(scene_rect.min.x, scene_rect.max.x - preview_rect.width());
                        min.y = min
                            .y
                            .clamp(scene_rect.min.y, scene_rect.max.y - preview_rect.height());
                        preview_rect = Rect::from_min_size(min, preview_rect.size());
                    }

                    let offset_x_den = (scene_w - preview_rect.width()).max(1.0);
                    let offset_y_den = (scene_h - preview_rect.height()).max(1.0);
                    preview_position_norm = [
                        ((preview_rect.min.x - scene_rect.min.x) / offset_x_den).clamp(0.0, 1.0),
                        ((preview_rect.min.y - scene_rect.min.y) / offset_y_den).clamp(0.0, 1.0),
                    ];
                    preview_width_norm = (preview_rect.width() / scene_w).clamp(0.05, 0.95);

                    ui.painter().rect_filled(
                        preview_rect.expand(2.0),
                        4.0,
                        Color32::from_black_alpha(200),
                    );
                    ui.painter()
                        .image(preview_texture_id, preview_rect, uv, Color32::WHITE);
                    ui.painter().text(
                        preview_rect.min + Vec2::new(6.0, 6.0),
                        Align2::LEFT_TOP,
                        entity_display_name(world, preview_entity),
                        FontId::proportional(11.0),
                        Color32::from_rgb(235, 235, 235),
                    );
                    let handle_color = if resize_response.hovered()
                        || resize_response.dragged_by(PointerButton::Primary)
                    {
                        Color32::from_rgb(220, 220, 220)
                    } else {
                        Color32::from_gray(180)
                    };
                    ui.painter().rect_filled(
                        resize_rect.shrink(1.0),
                        2.0,
                        Color32::from_black_alpha(150),
                    );
                    ui.painter().line_segment(
                        [
                            resize_rect.left_bottom() + Vec2::new(3.0, -3.0),
                            resize_rect.right_top() + Vec2::new(-3.0, 3.0),
                        ],
                        Stroke::new(1.5, handle_color),
                    );

                    preview_rect_pixels =
                        viewport_rect_pixels_from_ui_rect(preview_rect, pixels_per_point);
                    shown_preview_camera = Some(preview_entity);
                }
            }

            let pointer_pos = ui.ctx().input(|input| {
                input
                    .pointer
                    .interact_pos()
                    .or_else(|| input.pointer.hover_pos())
            });
            let pointer_over_main = ui
                .ctx()
                .input(|input| {
                    input
                        .pointer
                        .interact_pos()
                        .or_else(|| input.pointer.hover_pos())
                })
                .is_some_and(|pointer_pos| scene_rect.contains(pointer_pos))
                && main_response.hovered()
                && !preview_pointer_blocked
                && pointer_pos
                    .and_then(|pos| ui.ctx().layer_id_at(pos))
                    .map_or(true, |layer| layer == ui.layer_id());
            let mut keyboard_focus = false;
            if let Some(mut runtime) = world.get_resource_mut::<EditorViewportRuntime>() {
                runtime.editor_texture_id = Some(editor_texture_id);
                runtime.gameplay_texture_id = Some(gameplay_texture_id);
                runtime.preview_texture_id = Some(preview_texture_id);
                runtime.main_target_size = Some(main_target_size);
                runtime.main_resize_immediate = resolution_preset_changed;
                runtime.main_rect_pixels =
                    viewport_rect_pixels_from_ui_rect(scene_rect, pixels_per_point);
                runtime.preview_rect_pixels = preview_rect_pixels;
                runtime.preview_camera_entity = shown_preview_camera;
                runtime.pointer_over_main = pointer_over_main;
                if main_response.clicked_by(PointerButton::Primary)
                    || main_response.clicked_by(PointerButton::Secondary)
                {
                    if pointer_over_main {
                        runtime.keyboard_focus = true;
                    }
                } else if ui.ctx().input(|input| input.pointer.any_pressed()) && !pointer_over_main
                {
                    runtime.keyboard_focus = false;
                }
                keyboard_focus = runtime.keyboard_focus;
            }

            if let Some(mut passthrough) = world.get_resource_mut::<EguiInputPassthrough>() {
                passthrough.pointer = pointer_over_main;
                passthrough.keyboard = keyboard_focus;
            }
        }

        if let Some(mut gizmo_state) = world.get_resource_mut::<EditorGizmoState>() {
            gizmo_state.mode = gizmo_mode;
        }

        if let Some(mut viewport_state) = world.get_resource_mut::<EditorViewportState>() {
            viewport_state.graph_template = graph_template.clone();
            viewport_state.play_mode_view = play_mode_view;
            viewport_state.render_resolution = render_resolution;
            viewport_state.pinned_camera = pinned_camera;
            viewport_state.preview_position_norm = preview_position_norm;
            viewport_state.preview_width_norm = preview_width_norm;
            viewport_state.gizmos_in_play = gizmos_in_play;
            viewport_state.execute_scripts_in_edit_mode = execute_scripts_in_edit_mode;
            viewport_state.show_camera_gizmos = show_camera_gizmos;
            viewport_state.show_directional_light_gizmos = show_directional_light_gizmos;
            viewport_state.show_point_light_gizmos = show_point_light_gizmos;
            viewport_state.show_spot_light_gizmos = show_spot_light_gizmos;
            viewport_state.show_spline_paths = show_spline_paths;
            viewport_state.show_spline_points = show_spline_points;
            viewport_state.show_options_panel = show_options_panel;
        }

        if graph_selection_changed || previous_graph_template != graph_template {
            if let Some(template) = template_for_graph(&graph_template) {
                if let Some(mut graph_res) =
                    world.get_resource_mut::<helmer_becs::systems::render_system::RenderGraphResource>()
                {
                    graph_res.0 = (template.build)();
                }
                if let Some(mut render_sync) =
                    world
                        .get_resource_mut::<helmer_becs::systems::render_system::RenderSyncRequest>(
                        )
                {
                    render_sync.request_with_epoch(3);
                }
                if let Some(mut refresh) =
                    world.get_resource_mut::<crate::editor::scene::EditorRenderRefresh>()
                {
                    refresh.pending = true;
                }
            }
        }
    });
}

pub fn draw_project_window(ui: &mut Ui, world: &mut World) {
    let project_snapshot = world.get_resource::<EditorProject>().cloned();
    let project_loaded = project_snapshot
        .as_ref()
        .and_then(|project| project.root.as_ref())
        .is_some();

    with_middle_drag_blocked(ui, world, |ui, world| {
        if !project_loaded {
            ui.label("No project loaded");
            ui.separator();

            let mut open_request: Option<PathBuf> = None;
            let mut create_request: Option<(String, PathBuf)> = None;
            let mut browse_requested = false;

            world.resource_scope::<EditorUiState, _>(|_world, mut state| {
                if state.project_name.is_empty() {
                    state.project_name = "NewProject".to_string();
                }
                if state.project_path.is_empty() {
                    state.project_path = "./projects".to_string();
                }
                if state.open_project_path.is_empty() {
                    state.open_project_path = state.project_path.clone();
                }

                ui.heading("Open Project");
                ui.horizontal(|ui| {
                    ui.label("Path:");
                    ui.text_edit_singleline(&mut state.open_project_path);
                    if ui.button("Browse...").clicked() {
                        browse_requested = true;
                    }
                });

                if ui.button("Open").clicked() {
                    open_request = Some(PathBuf::from(state.open_project_path.clone()));
                }

                ui.separator();
                ui.heading("Recent Projects");
                if state.recent_projects.is_empty() {
                    ui.label("No recent projects yet.");
                } else {
                    for path in state.recent_projects.clone() {
                        if ui.button(path.display().to_string()).clicked() {
                            open_request = Some(path);
                        }
                    }
                }

                ui.separator();
                ui.heading("Create Project");
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(&mut state.project_name);
                });
                ui.horizontal(|ui| {
                    ui.label("Location:");
                    ui.text_edit_singleline(&mut state.project_path);
                });

                if ui.button("Create Project").clicked() {
                    let create_path = Path::new(&state.project_path).join(&state.project_name);
                    create_request = Some((state.project_name.clone(), create_path));
                }
            });

            if browse_requested {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
                        state.open_project_path = path.to_string_lossy().into_owned();
                    }
                    open_request = Some(path);
                }
            }

            if let Some(path) = open_request {
                push_command(world, EditorCommand::OpenProject { path });
            }
            if let Some((name, path)) = create_request {
                push_command(world, EditorCommand::CreateProject { name, path });
            }
        } else {
            let (project_root, project_name) = project_snapshot
                .as_ref()
                .map(|project| {
                    (
                        project.root.clone(),
                        project
                            .config
                            .as_ref()
                            .map(|cfg| cfg.name.clone())
                            .unwrap_or_else(|| "<unknown>".to_string()),
                    )
                })
                .unwrap_or((None, "<unknown>".to_string()));

            if let Some(root) = project_root {
                ui.label(format!("Project: {}", project_name));
                ui.label(root.display().to_string());
                ui.separator();
            }

            let mut save_request: Option<(PathBuf, ProjectConfig)> = None;
            let mut close_requested = false;

            world.resource_scope::<EditorProject, _>(|_world, mut project| {
                let root = match project.root.clone() {
                    Some(root) => root,
                    None => return,
                };
                let Some(config) = project.config.as_mut() else {
                    return;
                };

                ui.heading("Project Preferences");
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(&mut config.name);
                });
                ui.horizontal(|ui| {
                    ui.label("Assets Dir:");
                    ui.text_edit_singleline(&mut config.assets_dir);
                });
                ui.horizontal(|ui| {
                    ui.label("Models Dir:");
                    ui.text_edit_singleline(&mut config.models_dir);
                });
                ui.horizontal(|ui| {
                    ui.label("Textures Dir:");
                    ui.text_edit_singleline(&mut config.textures_dir);
                });
                ui.horizontal(|ui| {
                    ui.label("Materials Dir:");
                    ui.text_edit_singleline(&mut config.materials_dir);
                });
                ui.horizontal(|ui| {
                    ui.label("Scenes Dir:");
                    ui.text_edit_singleline(&mut config.scenes_dir);
                });
                ui.horizontal(|ui| {
                    ui.label("Scripts Dir:");
                    ui.text_edit_singleline(&mut config.scripts_dir);
                });

                ui.horizontal(|ui| {
                    if ui.button("Save Preferences").clicked() {
                        save_request = Some((root.clone(), config.clone()));
                    }
                    if ui.button("Close Project").clicked() {
                        close_requested = true;
                    }
                });
            });

            if let Some((root, config)) = save_request {
                match save_project_config(&root, &config) {
                    Ok(()) => {
                        set_status(world, "Project preferences saved".to_string());
                    }
                    Err(err) => {
                        set_status(world, format!("Failed to save preferences: {}", err));
                    }
                }
            }

            if close_requested {
                push_command(world, EditorCommand::CloseProject);
            }
        }
    });
}

pub fn draw_scene_window(ui: &mut Ui, world: &mut World) {
    with_middle_drag_blocked(ui, world, |ui, world| {
        let entries = collect_hierarchy_entries(world);

        egui::Frame::none().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button("Add", |ui| {
                    if ui.button("Empty").clicked() {
                        push_command(
                            world,
                            EditorCommand::CreateEntity {
                                kind: SpawnKind::Empty,
                            },
                        );
                        ui.close_menu();
                    }
                    if ui.button("Camera").clicked() {
                        push_command(
                            world,
                            EditorCommand::CreateEntity {
                                kind: SpawnKind::Camera,
                            },
                        );
                        ui.close_menu();
                    }
                    if ui.button("Directional Light").clicked() {
                        push_command(
                            world,
                            EditorCommand::CreateEntity {
                                kind: SpawnKind::DirectionalLight,
                            },
                        );
                        ui.close_menu();
                    }
                    if ui.button("Point Light").clicked() {
                        push_command(
                            world,
                            EditorCommand::CreateEntity {
                                kind: SpawnKind::PointLight,
                            },
                        );
                        ui.close_menu();
                    }
                    if ui.button("Spot Light").clicked() {
                        push_command(
                            world,
                            EditorCommand::CreateEntity {
                                kind: SpawnKind::SpotLight,
                            },
                        );
                        ui.close_menu();
                    }
                    if ui.button("Cube").clicked() {
                        push_command(
                            world,
                            EditorCommand::CreateEntity {
                                kind: SpawnKind::Primitive(PrimitiveKind::Cube),
                            },
                        );
                        ui.close_menu();
                    }
                    if ui.button("UV Sphere").clicked() {
                        push_command(
                            world,
                            EditorCommand::CreateEntity {
                                kind: SpawnKind::Primitive(PrimitiveKind::UvSphere(12, 12)),
                            },
                        );
                        ui.close_menu();
                    }
                    if ui.button("Plane").clicked() {
                        push_command(
                            world,
                            EditorCommand::CreateEntity {
                                kind: SpawnKind::Primitive(PrimitiveKind::Plane),
                            },
                        );
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Physics", |ui| {
                        if ui.button("Dynamic Body (Box)").clicked() {
                            push_command(
                                world,
                                EditorCommand::CreateEntity {
                                    kind: SpawnKind::DynamicBodyCuboid,
                                },
                            );
                            ui.close_menu();
                        }
                        if ui.button("Dynamic Body (Sphere)").clicked() {
                            push_command(
                                world,
                                EditorCommand::CreateEntity {
                                    kind: SpawnKind::DynamicBodySphere,
                                },
                            );
                            ui.close_menu();
                        }
                        if ui.button("Fixed Collider (Box)").clicked() {
                            push_command(
                                world,
                                EditorCommand::CreateEntity {
                                    kind: SpawnKind::FixedColliderCuboid,
                                },
                            );
                            ui.close_menu();
                        }
                        if ui.button("Fixed Collider (Sphere)").clicked() {
                            push_command(
                                world,
                                EditorCommand::CreateEntity {
                                    kind: SpawnKind::FixedColliderSphere,
                                },
                            );
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Provided", |ui| {
                        if ui.button("Freecam Camera").clicked() {
                            push_command(
                                world,
                                EditorCommand::CreateEntity {
                                    kind: SpawnKind::FreecamCamera,
                                },
                            );
                            ui.close_menu();
                        }
                    });
                });
            });
            ui.separator();
            draw_hierarchy_panel(ui, world, &entries);
        });
    });
}

pub fn draw_inspector_window(ui: &mut Ui, world: &mut World) {
    with_middle_drag_blocked(ui, world, |ui, world| {
        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(6, 0))
            .show(ui, |ui| {
                let selection = world
                    .get_resource::<InspectorSelectedEntityResource>()
                    .and_then(|selection| selection.0);
                let pinned = world
                    .get_resource::<InspectorPinnedEntityResource>()
                    .and_then(|pinned| pinned.0);
                let entity = selection.or(pinned);

                if let Some(entity) = selection {
                    if let Some(mut pinned) =
                        world.get_resource_mut::<InspectorPinnedEntityResource>()
                    {
                        pinned.0 = Some(entity);
                    }
                }

                let Some(entity) = entity else {
                    ui.label("Select an entity to inspect");
                    return;
                };
                if world.get_entity(entity).is_err() {
                    if selection.is_some() {
                        set_selection(world, None);
                    }
                    if let Some(mut pinned) =
                        world.get_resource_mut::<InspectorPinnedEntityResource>()
                    {
                        if pinned.0 == Some(entity) {
                            pinned.0 = None;
                        }
                    }
                    if selection.is_some() {
                        ui.label("Selected entity is no longer available");
                    } else {
                        ui.label("Select an entity to inspect");
                    }
                    return;
                }

                draw_inspector_header(ui, world, entity);
                ui.separator();

                egui::ScrollArea::both()
                    .id_salt("inspector_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        draw_inspector_panel(ui, world, entity);
                    });
            });
    });
}

fn default_pane_kind_for_window(window_id: &str) -> Option<EditorPaneKind> {
    match window_id {
        "Toolbar" => Some(EditorPaneKind::Toolbar),
        "Viewport" => Some(EditorPaneKind::Viewport),
        "Project" => Some(EditorPaneKind::Project),
        "Hierarchy" => Some(EditorPaneKind::Hierarchy),
        "Inspector" => Some(EditorPaneKind::Inspector),
        "History" => Some(EditorPaneKind::History),
        "Content Browser" => Some(EditorPaneKind::ContentBrowser),
        _ => None,
    }
}

fn make_pane_tab(workspace: &mut EditorPaneWorkspaceState, kind: EditorPaneKind) -> EditorPaneTab {
    let tab = EditorPaneTab {
        id: workspace.next_tab_id,
        title: kind.label().to_string(),
        kind,
    };
    workspace.next_tab_id += 1;
    tab
}

fn next_pane_area_id(workspace: &mut EditorPaneWorkspaceState) -> u64 {
    let id = workspace.next_area_id;
    workspace.next_area_id += 1;
    id
}

fn make_pane_area(
    workspace: &mut EditorPaneWorkspaceState,
    rect: EditorPaneAreaRect,
    tabs: Vec<EditorPaneTab>,
) -> EditorPaneArea {
    EditorPaneArea {
        id: next_pane_area_id(workspace),
        rect,
        tabs,
        active: 0,
    }
}

fn normalize_pane_area_rect(rect: &mut EditorPaneAreaRect) {
    const MIN_SPAN: f32 = 0.04;
    if !rect.x.is_finite() || !rect.y.is_finite() || !rect.w.is_finite() || !rect.h.is_finite() {
        *rect = EditorPaneAreaRect::full();
        return;
    }

    rect.w = rect.w.max(MIN_SPAN);
    rect.h = rect.h.max(MIN_SPAN);
    rect.x = rect.x.clamp(0.0, (1.0 - MIN_SPAN).max(0.0));
    rect.y = rect.y.clamp(0.0, (1.0 - MIN_SPAN).max(0.0));
    if rect.x + rect.w > 1.0 {
        rect.w = (1.0 - rect.x).max(MIN_SPAN);
    }
    if rect.y + rect.h > 1.0 {
        rect.h = (1.0 - rect.y).max(MIN_SPAN);
    }
}

fn split_pane_area_rect(
    rect: EditorPaneAreaRect,
    zone: PaneDropZone,
) -> Option<(EditorPaneAreaRect, EditorPaneAreaRect)> {
    const SPLIT_RATIO: f32 = 0.5;
    const MIN_SPAN: f32 = 0.06;
    let mut rect = rect;
    normalize_pane_area_rect(&mut rect);
    match zone {
        PaneDropZone::Center => None,
        PaneDropZone::Left | PaneDropZone::Right => {
            if rect.w < MIN_SPAN * 2.0 {
                return None;
            }
            let inserted_width = (rect.w * SPLIT_RATIO).clamp(MIN_SPAN, rect.w - MIN_SPAN);
            let existing_width = rect.w - inserted_width;
            if inserted_width <= 0.0 || existing_width <= 0.0 {
                return None;
            }
            let existing = if zone == PaneDropZone::Right {
                EditorPaneAreaRect {
                    x: rect.x,
                    y: rect.y,
                    w: existing_width,
                    h: rect.h,
                }
            } else {
                EditorPaneAreaRect {
                    x: rect.x + inserted_width,
                    y: rect.y,
                    w: existing_width,
                    h: rect.h,
                }
            };
            let inserted = if zone == PaneDropZone::Right {
                EditorPaneAreaRect {
                    x: rect.x + existing_width,
                    y: rect.y,
                    w: inserted_width,
                    h: rect.h,
                }
            } else {
                EditorPaneAreaRect {
                    x: rect.x,
                    y: rect.y,
                    w: inserted_width,
                    h: rect.h,
                }
            };
            let mut existing = existing;
            let mut inserted = inserted;
            normalize_pane_area_rect(&mut existing);
            normalize_pane_area_rect(&mut inserted);
            Some((existing, inserted))
        }
        PaneDropZone::Top | PaneDropZone::Bottom => {
            if rect.h < MIN_SPAN * 2.0 {
                return None;
            }
            let inserted_height = (rect.h * SPLIT_RATIO).clamp(MIN_SPAN, rect.h - MIN_SPAN);
            let existing_height = rect.h - inserted_height;
            if inserted_height <= 0.0 || existing_height <= 0.0 {
                return None;
            }
            let existing = if zone == PaneDropZone::Bottom {
                EditorPaneAreaRect {
                    x: rect.x,
                    y: rect.y,
                    w: rect.w,
                    h: existing_height,
                }
            } else {
                EditorPaneAreaRect {
                    x: rect.x,
                    y: rect.y + inserted_height,
                    w: rect.w,
                    h: existing_height,
                }
            };
            let inserted = if zone == PaneDropZone::Bottom {
                EditorPaneAreaRect {
                    x: rect.x,
                    y: rect.y + existing_height,
                    w: rect.w,
                    h: inserted_height,
                }
            } else {
                EditorPaneAreaRect {
                    x: rect.x,
                    y: rect.y,
                    w: rect.w,
                    h: inserted_height,
                }
            };
            let mut existing = existing;
            let mut inserted = inserted;
            normalize_pane_area_rect(&mut existing);
            normalize_pane_area_rect(&mut inserted);
            Some((existing, inserted))
        }
    }
}

fn split_pane_area_with_tab(
    workspace: &mut EditorPaneWorkspaceState,
    window_index: usize,
    area_index: usize,
    zone: PaneDropZone,
    tab: EditorPaneTab,
) -> Option<u64> {
    let rect = workspace
        .windows
        .get(window_index)?
        .areas
        .get(area_index)?
        .rect;
    let (existing_rect, inserted_rect) = split_pane_area_rect(rect, zone)?;
    workspace.windows[window_index].areas[area_index].rect = existing_rect;
    let area = make_pane_area(workspace, inserted_rect, vec![tab]);
    let area_id = area.id;
    workspace.windows[window_index].areas.push(area);
    Some(area_id)
}

fn pane_rect_nearly_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.0005
}

fn pane_rects_share_vertical_edge(a: EditorPaneAreaRect, b: EditorPaneAreaRect) -> bool {
    pane_rect_nearly_eq(a.y, b.y)
        && pane_rect_nearly_eq(a.h, b.h)
        && (pane_rect_nearly_eq(a.x + a.w, b.x) || pane_rect_nearly_eq(b.x + b.w, a.x))
}

fn pane_rects_share_horizontal_edge(a: EditorPaneAreaRect, b: EditorPaneAreaRect) -> bool {
    pane_rect_nearly_eq(a.x, b.x)
        && pane_rect_nearly_eq(a.w, b.w)
        && (pane_rect_nearly_eq(a.y + a.h, b.y) || pane_rect_nearly_eq(b.y + b.h, a.y))
}

fn pane_rects_match(a: EditorPaneAreaRect, b: EditorPaneAreaRect) -> bool {
    pane_rect_nearly_eq(a.x, b.x)
        && pane_rect_nearly_eq(a.y, b.y)
        && pane_rect_nearly_eq(a.w, b.w)
        && pane_rect_nearly_eq(a.h, b.h)
}

fn pane_rect_union(a: EditorPaneAreaRect, b: EditorPaneAreaRect) -> EditorPaneAreaRect {
    let min_x = a.x.min(b.x);
    let min_y = a.y.min(b.y);
    let max_x = (a.x + a.w).max(b.x + b.w);
    let max_y = (a.y + a.h).max(b.y + b.h);
    EditorPaneAreaRect {
        x: min_x,
        y: min_y,
        w: (max_x - min_x).max(0.0),
        h: (max_y - min_y).max(0.0),
    }
}

fn compact_pane_window_areas(window: &mut EditorPaneWindow) {
    for area in window.areas.iter_mut() {
        normalize_pane_area_rect(&mut area.rect);
    }

    for area in window.areas.iter_mut() {
        if area.active >= area.tabs.len() {
            area.active = area.tabs.len().saturating_sub(1);
        }
    }

    let mut cursor = 0usize;
    while cursor < window.areas.len() {
        if !window.areas[cursor].tabs.is_empty() {
            cursor += 1;
            continue;
        }

        let empty_rect = window.areas[cursor].rect;
        let merge_target = window
            .areas
            .iter()
            .enumerate()
            .filter(|(index, area)| *index != cursor && !area.tabs.is_empty())
            .find_map(|(index, area)| {
                if pane_rects_share_vertical_edge(empty_rect, area.rect)
                    || pane_rects_share_horizontal_edge(empty_rect, area.rect)
                {
                    Some((index, pane_rect_union(area.rect, empty_rect)))
                } else {
                    None
                }
            });

        if let Some((target_index, union_rect)) = merge_target {
            window.areas[target_index].rect = union_rect;
        }
        window.areas.remove(cursor);
    }

    // If rects became identical due drag/drop edge-cases, collapse them into tabs.
    let mut i = 0usize;
    while i < window.areas.len() {
        let mut j = i + 1;
        while j < window.areas.len() {
            if pane_rects_match(window.areas[i].rect, window.areas[j].rect) {
                let source_active = window.areas[j].active;
                let insert_at = window.areas[i].tabs.len();
                let mut moved_tabs = std::mem::take(&mut window.areas[j].tabs);
                window.areas[i].tabs.append(&mut moved_tabs);
                if insert_at < window.areas[i].tabs.len() {
                    let active_offset =
                        source_active.min(window.areas[i].tabs.len() - insert_at - 1);
                    window.areas[i].active = insert_at + active_offset;
                } else if window.areas[i].active >= window.areas[i].tabs.len() {
                    window.areas[i].active = window.areas[i].tabs.len().saturating_sub(1);
                }
                window.areas.remove(j);
                continue;
            }
            j += 1;
        }
        i += 1;
    }

    for area in window.areas.iter_mut() {
        normalize_pane_area_rect(&mut area.rect);
    }
}

fn compact_pane_workspace(workspace: &mut EditorPaneWorkspaceState) {
    let mut index = 0usize;
    while index < workspace.windows.len() {
        compact_pane_window_areas(&mut workspace.windows[index]);

        let all_empty = workspace.windows[index]
            .areas
            .iter()
            .all(|area| area.tabs.is_empty());

        if all_empty {
            if workspace.windows[index].layout_managed {
                workspace.windows[index].areas.clear();
                let area_id = next_pane_area_id(workspace);
                workspace.windows[index].areas.push(EditorPaneArea {
                    id: area_id,
                    rect: EditorPaneAreaRect::full(),
                    tabs: Vec::new(),
                    active: 0,
                });
                index += 1;
            } else {
                let removed_window_id = workspace.windows[index].id.clone();
                workspace.windows.remove(index);
                if workspace
                    .last_focused_window
                    .as_deref()
                    .is_some_and(|focused| focused == removed_window_id)
                {
                    workspace.last_focused_window = None;
                    workspace.last_focused_area = None;
                }
            }
        } else {
            index += 1;
        }
    }

    if let Some(focused_window_id) = workspace.last_focused_window.clone() {
        if let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| window.id == focused_window_id)
        {
            let focused_area_valid = workspace.last_focused_area.is_some_and(|area_id| {
                workspace.windows[window_index]
                    .areas
                    .iter()
                    .any(|area| area.id == area_id)
            });
            if !focused_area_valid {
                workspace.last_focused_area = workspace.windows[window_index]
                    .areas
                    .first()
                    .map(|area| area.id);
            }
        } else {
            workspace.last_focused_window =
                workspace.windows.first().map(|window| window.id.clone());
            workspace.last_focused_area = workspace
                .windows
                .first()
                .and_then(|window| window.areas.first())
                .map(|area| area.id);
        }
    } else {
        workspace.last_focused_window = workspace.windows.first().map(|window| window.id.clone());
        workspace.last_focused_area = workspace
            .windows
            .first()
            .and_then(|window| window.areas.first())
            .map(|area| area.id);
    }
}

fn next_pane_window_id(workspace: &mut EditorPaneWorkspaceState) -> String {
    let id = workspace.next_window_id;
    workspace.next_window_id += 1;
    format!("pane_window_{}", id)
}

fn refresh_pane_window_titles(workspace: &mut EditorPaneWorkspaceState) {
    for (index, window) in workspace.windows.iter_mut().enumerate() {
        if window.layout_managed {
            window.title = window.id.clone();
            continue;
        }
        if window.areas.is_empty() {
            window.title = format!("Pane {}", index + 1);
            continue;
        }
        let mut best_title: Option<String> = None;
        for area in window.areas.iter_mut() {
            if area.tabs.is_empty() {
                continue;
            }
            if area.active >= area.tabs.len() {
                area.active = area.tabs.len().saturating_sub(1);
            }
            best_title = Some(area.tabs[area.active].title.clone());
            break;
        }
        window.title = best_title.unwrap_or_else(|| format!("Pane {}", index + 1));
    }
}

pub fn ensure_default_pane_workspace(world: &mut World) {
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        if workspace.initialized {
            return;
        }
        if !workspace.windows.is_empty() {
            workspace.initialized = true;
            return;
        }

        for window_id in crate::editor::layout_window_ids() {
            let mut tabs = Vec::new();
            if let Some(kind) = default_pane_kind_for_window(window_id) {
                tabs.push(make_pane_tab(&mut workspace, kind));
            }
            if *window_id == "Content Browser" {
                tabs.push(make_pane_tab(&mut workspace, EditorPaneKind::Console));
            }
            let area = make_pane_area(&mut workspace, EditorPaneAreaRect::full(), tabs);
            workspace.windows.push(EditorPaneWindow {
                id: (*window_id).to_string(),
                title: (*window_id).to_string(),
                areas: vec![area],
                layout_managed: true,
            });
        }

        workspace.initialized = true;
        workspace.last_focused_window = Some("Viewport".to_string());
        workspace.last_focused_area = workspace
            .windows
            .iter()
            .find(|window| window.id == "Viewport")
            .and_then(|window| window.areas.first())
            .map(|area| area.id);
        refresh_pane_window_titles(&mut workspace);
    });
}

fn reset_pane_workspace(world: &mut World) {
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        workspace.initialized = false;
        workspace.windows.clear();
        workspace.last_focused_window = None;
        workspace.last_focused_area = None;
        workspace.dragging = None;
        workspace.drop_handled = false;
    });
    ensure_default_pane_workspace(world);
}

pub fn spawn_pane_workspace_tab(world: &mut World, kind: EditorPaneKind) {
    open_pane_workspace_tab(world, kind, None, None, None);
}

pub fn spawn_play_viewport_pane(world: &mut World) {
    ensure_default_pane_workspace(world);
    let reused_existing =
        world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
            let mut found: Option<(usize, usize, usize)> = None;
            for (window_index, window) in workspace.windows.iter().enumerate() {
                for (area_index, area) in window.areas.iter().enumerate() {
                    if let Some(tab_index) = area
                        .tabs
                        .iter()
                        .position(|tab| tab.kind == EditorPaneKind::PlayViewport)
                    {
                        found = Some((window_index, area_index, tab_index));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }

            if let Some((window_index, area_index, tab_index)) = found {
                workspace.windows[window_index].areas[area_index].active = tab_index;
                workspace.last_focused_window = Some(workspace.windows[window_index].id.clone());
                workspace.last_focused_area =
                    Some(workspace.windows[window_index].areas[area_index].id);
                refresh_pane_window_titles(&mut workspace);
                true
            } else {
                false
            }
        });
    if !reused_existing {
        open_pane_workspace_tab(world, EditorPaneKind::PlayViewport, None, None, None);
    }
}

pub(crate) fn close_pane_workspace_window(world: &mut World, window_id: &str) {
    let removed_tab_ids =
        world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
            let mut removed_tab_ids = Vec::new();
            if let Some(index) = workspace
                .windows
                .iter()
                .position(|window| window.id == window_id)
            {
                removed_tab_ids.extend(
                    workspace.windows[index]
                        .areas
                        .iter()
                        .flat_map(|area| area.tabs.iter().map(|tab| tab.id)),
                );
                workspace.windows.remove(index);
            }

            if workspace
                .dragging
                .as_ref()
                .map(|dragging| dragging.source_window_id == window_id)
                .unwrap_or(false)
            {
                workspace.dragging = None;
                workspace.drop_handled = false;
            }

            if workspace
                .last_focused_window
                .as_deref()
                .is_some_and(|focused| focused == window_id)
            {
                workspace.last_focused_window = None;
                workspace.last_focused_area = None;
            }

            compact_pane_workspace(&mut workspace);
            refresh_pane_window_titles(&mut workspace);
            removed_tab_ids
        });
    if let Some(mut pane_viewport_state) = world.get_resource_mut::<EditorPaneViewportState>() {
        for tab_id in removed_tab_ids {
            pane_viewport_state.resolutions.remove(&tab_id);
            pane_viewport_state.settings.remove(&tab_id);
        }
    }
}

fn open_pane_workspace_tab(
    world: &mut World,
    kind: EditorPaneKind,
    target_window_id: Option<&str>,
    target_area_id: Option<u64>,
    split_axis: Option<SplitAxis>,
) {
    ensure_default_pane_workspace(world);

    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let tab = make_pane_tab(&mut workspace, kind);
        let preferred_layout_window = target_window_id
            .map(|id| id.to_string())
            .or_else(|| kind.default_layout_window_id().map(|id| id.to_string()));
        let target_id = preferred_layout_window
            .or_else(|| workspace.last_focused_window.clone())
            .or_else(|| workspace.windows.first().map(|window| window.id.clone()));

        if let Some(target_id) = target_id {
            if let Some(window_index) = workspace
                .windows
                .iter()
                .position(|window| window.id == target_id)
            {
                let preferred_area = target_area_id.or_else(|| {
                    let focused = workspace.last_focused_area?;
                    workspace.windows[window_index]
                        .areas
                        .iter()
                        .any(|area| area.id == focused)
                        .then_some(focused)
                });

                let area_index = preferred_area
                    .and_then(|area_id| {
                        workspace.windows[window_index]
                            .areas
                            .iter()
                            .position(|area| area.id == area_id)
                    })
                    .or_else(|| {
                        (!workspace.windows[window_index].areas.is_empty()).then_some(0usize)
                    });

                let area_index = if let Some(area_index) = area_index {
                    area_index
                } else {
                    let area =
                        make_pane_area(&mut workspace, EditorPaneAreaRect::full(), Vec::new());
                    workspace.windows[window_index].areas.push(area);
                    workspace.windows[window_index]
                        .areas
                        .len()
                        .saturating_sub(1)
                };

                if let Some(axis) = split_axis {
                    if workspace.windows[window_index].areas[area_index]
                        .tabs
                        .is_empty()
                    {
                        workspace.windows[window_index].areas[area_index]
                            .tabs
                            .push(tab);
                        let new_index = workspace.windows[window_index].areas[area_index]
                            .tabs
                            .len()
                            .saturating_sub(1);
                        workspace.windows[window_index].areas[area_index].active = new_index;
                        workspace.last_focused_window =
                            Some(workspace.windows[window_index].id.clone());
                        workspace.last_focused_area =
                            Some(workspace.windows[window_index].areas[area_index].id);
                        compact_pane_workspace(&mut workspace);
                        refresh_pane_window_titles(&mut workspace);
                        return;
                    }

                    let zone = PaneDropZone::from_split_axis(axis);
                    if let Some(new_area_id) = split_pane_area_with_tab(
                        &mut workspace,
                        window_index,
                        area_index,
                        zone,
                        tab.clone(),
                    ) {
                        workspace.last_focused_window =
                            Some(workspace.windows[window_index].id.clone());
                        workspace.last_focused_area = Some(new_area_id);
                    } else {
                        workspace.windows[window_index].areas[area_index]
                            .tabs
                            .push(tab);
                        let new_index = workspace.windows[window_index].areas[area_index]
                            .tabs
                            .len()
                            .saturating_sub(1);
                        workspace.windows[window_index].areas[area_index].active = new_index;
                        workspace.last_focused_window =
                            Some(workspace.windows[window_index].id.clone());
                        workspace.last_focused_area =
                            Some(workspace.windows[window_index].areas[area_index].id);
                    }
                } else {
                    workspace.windows[window_index].areas[area_index]
                        .tabs
                        .push(tab);
                    let new_index = workspace.windows[window_index].areas[area_index]
                        .tabs
                        .len()
                        .saturating_sub(1);
                    workspace.windows[window_index].areas[area_index].active = new_index;
                    workspace.last_focused_window =
                        Some(workspace.windows[window_index].id.clone());
                    workspace.last_focused_area =
                        Some(workspace.windows[window_index].areas[area_index].id);
                }
                compact_pane_workspace(&mut workspace);
                refresh_pane_window_titles(&mut workspace);
                return;
            }
        }

        let window_id = next_pane_window_id(&mut workspace);
        let area = make_pane_area(&mut workspace, EditorPaneAreaRect::full(), vec![tab]);
        let area_id = area.id;
        workspace.windows.push(EditorPaneWindow {
            id: window_id.clone(),
            title: "Pane".to_string(),
            areas: vec![area],
            layout_managed: false,
        });
        workspace.last_focused_window = Some(window_id);
        workspace.last_focused_area = Some(area_id);
        compact_pane_workspace(&mut workspace);
        refresh_pane_window_titles(&mut workspace);
    });
}

fn spawn_pane_workspace_window(world: &mut World, kind: EditorPaneKind, pointer_pos: Option<Pos2>) {
    let mut spawned_window_id: Option<String> = None;
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        workspace.initialized = true;
        let tab = make_pane_tab(&mut workspace, kind);
        let window_id = next_pane_window_id(&mut workspace);
        let area = make_pane_area(&mut workspace, EditorPaneAreaRect::full(), vec![tab]);
        let area_id = area.id;
        workspace.windows.push(EditorPaneWindow {
            id: window_id.clone(),
            title: "Pane".to_string(),
            areas: vec![area],
            layout_managed: false,
        });
        workspace.last_focused_window = Some(window_id.clone());
        workspace.last_focused_area = Some(area_id);
        spawned_window_id = Some(window_id);
        compact_pane_workspace(&mut workspace);
        refresh_pane_window_titles(&mut workspace);
    });

    if let (Some(window_id), Some(pointer_pos)) = (spawned_window_id, pointer_pos) {
        if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
            egui_res
                .window_positions
                .insert(window_id, pointer_pos - Vec2::new(120.0, 14.0));
        }
    }
}

fn draw_pane_spawn_new_window_menu(ui: &mut Ui, world: &mut World, pointer_pos: Option<Pos2>) {
    ui.menu_button("New Window", |ui| {
        for kind in EditorPaneKind::ALL {
            if ui.button(kind.label()).clicked() {
                spawn_pane_workspace_window(world, kind, pointer_pos);
                ui.close_menu();
            }
        }
    });
}

fn draw_pane_area_context_menu(ui: &mut Ui, world: &mut World, window_id: &str, area_id: u64) {
    let spawn_pos = ui.ctx().input(|input| {
        input
            .pointer
            .hover_pos()
            .or_else(|| input.pointer.interact_pos())
    });
    draw_pane_spawn_menu(ui, world, window_id, Some(area_id), None);
    ui.separator();
    draw_pane_spawn_menu(
        ui,
        world,
        window_id,
        Some(area_id),
        Some(SplitAxis::Vertical),
    );
    draw_pane_spawn_menu(
        ui,
        world,
        window_id,
        Some(area_id),
        Some(SplitAxis::Horizontal),
    );
    ui.separator();
    draw_pane_spawn_new_window_menu(ui, world, spawn_pos);
}

fn pointer_over_viewport_scene_at(ui: &Ui, world: &World, pointer_pos: Option<Pos2>) -> bool {
    pointer_pos.is_some_and(|pointer_pos| {
        let pixels_per_point = ui.ctx().pixels_per_point().max(0.0001) as f64;
        let pointer_pixels = DVec2::new(
            pointer_pos.x as f64 * pixels_per_point,
            pointer_pos.y as f64 * pixels_per_point,
        );
        world
            .get_resource::<EditorViewportRuntime>()
            .is_some_and(|runtime| {
                runtime
                    .pane_requests
                    .iter()
                    .any(|request| request.viewport_rect.contains(pointer_pixels))
            })
    })
}

fn pointer_over_any_pane_window(ctx: &egui::Context, world: &World, pointer_pos: Pos2) -> bool {
    world
        .get_resource::<EditorPaneWorkspaceState>()
        .is_some_and(|workspace| {
            workspace.windows.iter().any(|window| {
                ctx.memory(|mem| mem.area_rect(Id::new(window.id.clone())))
                    .is_some_and(|rect| rect.contains(pointer_pos))
            })
        })
}

fn draw_pane_workspace_background_context_menu(ui: &mut Ui, world: &mut World) {
    let screen_rect = ui.ctx().viewport_rect();
    let area_id = Id::new("pane_workspace_background_menu_area");
    egui::Area::new(area_id)
        .order(Order::Background)
        .fixed_pos(screen_rect.min)
        .interactable(true)
        .show(ui.ctx(), |ui| {
            ui.set_min_size(screen_rect.size());
            let background_rect = ui.max_rect();
            let response = ui.interact(background_rect, area_id.with("response"), Sense::click());
            let pointer_pos = response.interact_pointer_pos().or_else(|| {
                ui.ctx().input(|input| {
                    input
                        .pointer
                        .hover_pos()
                        .or_else(|| input.pointer.interact_pos())
                })
            });
            response.context_menu(|ui| {
                draw_pane_spawn_new_window_menu(ui, world, pointer_pos);
            });
        });
}

pub fn draw_pane_workspace_window(ui: &mut Ui, world: &mut World, window_id: &str) {
    let middle_drag_active = world
        .get_resource::<MiddleDragUiState>()
        .map(|state| state.active)
        .unwrap_or(false);
    if middle_drag_active {
        ui.add_enabled_ui(false, |ui| {
            draw_pane_workspace_window_contents(ui, world, window_id);
        });
    } else {
        draw_pane_workspace_window_contents(ui, world, window_id);
    }
}

fn draw_pane_workspace_window_contents(ui: &mut Ui, world: &mut World, window_id: &str) {
    let (areas_snapshot, window_index, layout_managed) = {
        let Some(state) = world.get_resource::<EditorPaneWorkspaceState>() else {
            ui.label("Pane workspace missing");
            return;
        };
        let Some((index, window)) = state
            .windows
            .iter()
            .enumerate()
            .find(|(_, window)| window.id == window_id)
        else {
            ui.label("Pane window missing");
            return;
        };
        (window.areas.clone(), index, window.layout_managed)
    };

    let available = ui.available_size_before_wrap();
    let min_size = Vec2::new(180.0, 70.0);
    let host_size = Vec2::new(available.x.max(min_size.x), available.y.max(min_size.y));
    let (host_rect, host_response) = ui.allocate_exact_size(host_size, Sense::click());
    if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
        egui_res
            .window_content_rects
            .insert(window_id.to_string(), host_rect);
    }
    ui.painter()
        .rect_filled(host_rect, 0.0, Color32::from_rgb(22, 24, 28));
    ui.painter().rect_stroke(
        host_rect,
        0.0,
        Stroke::new(1.0, Color32::from_gray(56)),
        StrokeKind::Inside,
    );
    if window_index == 0 {
        draw_pane_workspace_background_context_menu(ui, world);
    }

    if areas_snapshot.is_empty() {
        open_pane_workspace_tab(world, EditorPaneKind::Viewport, Some(window_id), None, None);
    }

    for area in areas_snapshot.iter().cloned() {
        let area_rect = area.rect.to_rect(host_rect);
        if area_rect.width() < 24.0 || area_rect.height() < 24.0 {
            continue;
        }
        draw_pane_area(ui, world, window_id, area, area_rect);
    }

    let pointer_over_viewport_scene = pointer_over_viewport_scene_at(
        ui,
        world,
        ui.ctx().input(|input| {
            input
                .pointer
                .hover_pos()
                .or_else(|| input.pointer.interact_pos())
        }),
    );
    let window_has_active_viewport = areas_snapshot.iter().any(|area| {
        area.tabs
            .get(area.active.min(area.tabs.len().saturating_sub(1)))
            .is_some_and(|tab| {
                matches!(
                    tab.kind,
                    EditorPaneKind::Viewport | EditorPaneKind::PlayViewport
                )
            })
    });
    if !pointer_over_viewport_scene || !window_has_active_viewport {
        let spawn_pos = host_response.interact_pointer_pos().or_else(|| {
            ui.ctx().input(|input| {
                input
                    .pointer
                    .hover_pos()
                    .or_else(|| input.pointer.interact_pos())
            })
        });
        host_response.context_menu(|ui| {
            draw_pane_spawn_menu(ui, world, window_id, None, None);
            ui.separator();
            draw_pane_spawn_new_window_menu(ui, world, spawn_pos);
        });
    }

    if let Some(dragging) = world
        .get_resource::<EditorPaneWorkspaceState>()
        .and_then(|state| state.dragging.clone())
    {
        if let Some(pointer_pos) = ui.ctx().input(|input| input.pointer.hover_pos()) {
            let painter = ui.ctx().layer_painter(egui::LayerId::new(
                Order::Tooltip,
                ui.id().with("pane_tab_drag"),
            ));
            painter.rect_filled(
                Rect::from_min_size(pointer_pos + Vec2::new(12.0, 12.0), Vec2::new(180.0, 28.0)),
                6.0,
                ui.visuals().widgets.active.bg_fill,
            );
            painter.text(
                pointer_pos + Vec2::new(20.0, 26.0),
                Align2::LEFT_CENTER,
                dragging.tab.title,
                FontId::proportional(13.0),
                ui.visuals().text_color(),
            );
        }
    }

    if world
        .get_resource::<EditorPaneWorkspaceState>()
        .map(|state| state.dragging.is_some() && !state.drop_handled)
        .unwrap_or(false)
        && ui.ctx().input(|input| input.pointer.any_released())
    {
        let pointer_pos = ui.ctx().input(|input| input.pointer.hover_pos());
        let pointer_over_this_layer = pointer_pos
            .and_then(|pos| ui.ctx().layer_id_at(pos))
            .is_some_and(|layer_id| layer_id == ui.layer_id());
        let window_outer_rect = ui.ctx().memory(|mem| mem.area_rect(Id::new(window_id)));
        let target = pointer_pos
            .filter(|pos| host_rect.contains(*pos))
            .and_then(|pos| {
                areas_snapshot
                    .iter()
                    .filter_map(|area| {
                        let area_rect = area.rect.to_rect(host_rect);
                        if area_rect.width() <= 0.0 || area_rect.height() <= 0.0 {
                            return None;
                        }

                        let clamped = Pos2::new(
                            pos.x.clamp(area_rect.left(), area_rect.right()),
                            pos.y.clamp(area_rect.top(), area_rect.bottom()),
                        );
                        let dx = pos.x - clamped.x;
                        let dy = pos.y - clamped.y;
                        let distance_sq = dx * dx + dy * dy;
                        let empty_penalty = if area.tabs.is_empty() {
                            1_000_000.0
                        } else {
                            0.0
                        };
                        let probe_pos = if area_rect.contains(pos) {
                            pos
                        } else {
                            clamped
                        };
                        let zone = pane_drop_zone_for_pointer(area_rect, probe_pos);
                        Some((area.id, zone, distance_sq + empty_penalty))
                    })
                    .min_by(|a, b| a.2.total_cmp(&b.2))
                    .map(|(area_id, zone, _)| (area_id, zone))
            })
            .or_else(|| {
                match (pointer_pos, window_outer_rect) {
                    (Some(pos), Some(window_rect))
                        if pointer_over_this_layer && window_rect.contains(pos) =>
                    {
                        // dropping on a pane window title bar/border should dock as tab in that window
                        areas_snapshot
                            .first()
                            .map(|area| (area.id, PaneDropZone::Center))
                    }
                    _ => None,
                }
            });

        if let Some((target_area, zone)) = target {
            if zone == PaneDropZone::Center {
                accept_pane_tab_drop(world, window_id, target_area, None);
            } else {
                accept_pane_tab_split_drop(world, window_id, target_area, zone);
            }
        } else {
            let pointer_over_any_window =
                pointer_pos.is_some_and(|pos| pointer_over_any_pane_window(ui.ctx(), world, pos));
            if !pointer_over_any_window {
                drop_pane_tab_into_new_window(world, pointer_pos);
            }
        }
    }

    if layout_managed {
        world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
            if let Some(window_index) = workspace
                .windows
                .iter()
                .position(|window| window.id == window_id)
            {
                if workspace.windows[window_index].areas.is_empty() {
                    let area =
                        make_pane_area(&mut workspace, EditorPaneAreaRect::full(), Vec::new());
                    workspace.windows[window_index].areas.push(area);
                }
            }
        });
    }
}

#[derive(Clone)]
struct PaneTabDragPayload;

fn begin_pane_tab_drag(world: &mut World, window_id: &str, area_id: u64, tab_index: usize) {
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| window.id == window_id)
        else {
            return;
        };

        let Some(area_index) = workspace.windows[window_index]
            .areas
            .iter()
            .position(|area| area.id == area_id)
        else {
            return;
        };

        let (tab, source_was_single_tab) = {
            let area = &mut workspace.windows[window_index].areas[area_index];
            if tab_index >= area.tabs.len() {
                return;
            }
            if area.tabs.len() == 1 {
                (area.tabs[tab_index].clone(), true)
            } else {
                let tab = area.tabs.remove(tab_index);
                if area.active >= area.tabs.len() {
                    area.active = area.tabs.len().saturating_sub(1);
                }
                (tab, false)
            }
        };

        workspace.dragging = Some(EditorPaneTabDrag {
            tab,
            source_window_id: window_id.to_string(),
            source_area_id: area_id,
            source_was_single_tab,
        });
        workspace.drop_handled = false;
        refresh_pane_window_titles(&mut workspace);
    });
}

fn detach_pane_tab_to_new_window(
    world: &mut World,
    window_id: &str,
    area_id: u64,
    tab_index: usize,
) {
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| window.id == window_id)
        else {
            return;
        };

        let Some(area_index) = workspace.windows[window_index]
            .areas
            .iter()
            .position(|area| area.id == area_id)
        else {
            return;
        };

        let tab = {
            let area = &mut workspace.windows[window_index].areas[area_index];
            if tab_index >= area.tabs.len() {
                return;
            }
            let tab = area.tabs.remove(tab_index);
            if area.active >= area.tabs.len() {
                area.active = area.tabs.len().saturating_sub(1);
            }
            tab
        };

        let new_window_id = next_pane_window_id(&mut workspace);
        let area = make_pane_area(&mut workspace, EditorPaneAreaRect::full(), vec![tab]);
        let new_area_id = area.id;
        workspace.windows.push(EditorPaneWindow {
            id: new_window_id.clone(),
            title: "Pane".to_string(),
            areas: vec![area],
            layout_managed: false,
        });
        workspace.last_focused_window = Some(new_window_id);
        workspace.last_focused_area = Some(new_area_id);
        compact_pane_workspace(&mut workspace);
        refresh_pane_window_titles(&mut workspace);
    });
}

fn close_pane_tab_in_window(world: &mut World, window_id: &str, area_id: u64, tab_index: usize) {
    let removed_tab_id =
        world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
            let Some(window_index) = workspace
                .windows
                .iter()
                .position(|window| window.id == window_id)
            else {
                return None;
            };

            let Some(area_index) = workspace.windows[window_index]
                .areas
                .iter()
                .position(|area| area.id == area_id)
            else {
                return None;
            };

            let mut removed_tab_id = None;
            if tab_index < workspace.windows[window_index].areas[area_index].tabs.len() {
                removed_tab_id = workspace.windows[window_index].areas[area_index]
                    .tabs
                    .get(tab_index)
                    .map(|tab| tab.id);
                workspace.windows[window_index].areas[area_index]
                    .tabs
                    .remove(tab_index);
            }

            if workspace.windows[window_index].areas[area_index].active
                >= workspace.windows[window_index].areas[area_index].tabs.len()
            {
                workspace.windows[window_index].areas[area_index].active =
                    workspace.windows[window_index].areas[area_index]
                        .tabs
                        .len()
                        .saturating_sub(1);
            }

            compact_pane_workspace(&mut workspace);
            refresh_pane_window_titles(&mut workspace);
            removed_tab_id
        });
    if let Some(tab_id) = removed_tab_id {
        if let Some(mut pane_viewport_state) = world.get_resource_mut::<EditorPaneViewportState>() {
            pane_viewport_state.resolutions.remove(&tab_id);
            pane_viewport_state.settings.remove(&tab_id);
        }
    }
}

fn accept_pane_tab_drop(
    world: &mut World,
    target_window_id: &str,
    target_area_id: u64,
    insert_index: Option<usize>,
) {
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let Some(dragging) = workspace.dragging.take() else {
            return;
        };

        if dragging.source_was_single_tab
            && dragging.source_window_id == target_window_id
            && dragging.source_area_id == target_area_id
        {
            workspace.last_focused_window = Some(target_window_id.to_string());
            workspace.last_focused_area = Some(target_area_id);
            workspace.drop_handled = true;
            compact_pane_workspace(&mut workspace);
            refresh_pane_window_titles(&mut workspace);
            return;
        }

        if dragging.source_was_single_tab {
            remove_pane_tab_from_window_by_id(
                &mut workspace,
                &dragging.source_window_id,
                dragging.source_area_id,
                dragging.tab.id,
            );
        }

        let mut placed_window_id = target_window_id.to_string();
        let mut placed_area_id = target_area_id;
        if let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| window.id == target_window_id)
        {
            if let Some(area_index) = workspace.windows[window_index]
                .areas
                .iter()
                .position(|area| area.id == target_area_id)
            {
                let area = &mut workspace.windows[window_index].areas[area_index];
                let index = insert_index.unwrap_or(area.tabs.len()).min(area.tabs.len());
                area.tabs.insert(index, dragging.tab);
                area.active = index;
            } else {
                if let Some(area) = workspace.windows[window_index].areas.first_mut() {
                    let index = insert_index.unwrap_or(area.tabs.len()).min(area.tabs.len());
                    area.tabs.insert(index, dragging.tab);
                    area.active = index;
                    placed_area_id = area.id;
                } else {
                    let area = make_pane_area(
                        &mut workspace,
                        EditorPaneAreaRect::full(),
                        vec![dragging.tab],
                    );
                    placed_area_id = area.id;
                    workspace.windows[window_index].areas.push(area);
                }
            }
        } else {
            let new_window_id = next_pane_window_id(&mut workspace);
            let area = make_pane_area(
                &mut workspace,
                EditorPaneAreaRect::full(),
                vec![dragging.tab],
            );
            placed_area_id = area.id;
            workspace.windows.push(EditorPaneWindow {
                id: new_window_id.clone(),
                title: "Pane".to_string(),
                areas: vec![area],
                layout_managed: false,
            });
            placed_window_id = new_window_id;
        }

        workspace.last_focused_window = Some(placed_window_id);
        workspace.last_focused_area = Some(placed_area_id);
        workspace.drop_handled = true;

        compact_pane_workspace(&mut workspace);
        refresh_pane_window_titles(&mut workspace);
    });
}

fn accept_pane_tab_split_drop(
    world: &mut World,
    target_window_id: &str,
    target_area_id: u64,
    zone: PaneDropZone,
) {
    if zone == PaneDropZone::Center {
        accept_pane_tab_drop(world, target_window_id, target_area_id, None);
        return;
    }

    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let Some(dragging) = workspace.dragging.take() else {
            return;
        };

        let splitting_source_area = dragging.source_was_single_tab
            && dragging.source_window_id == target_window_id
            && dragging.source_area_id == target_area_id;
        if splitting_source_area {
            if let Some(window_index) = workspace
                .windows
                .iter()
                .position(|window| window.id == target_window_id)
            {
                if let Some(area_index) = workspace.windows[window_index]
                    .areas
                    .iter()
                    .position(|area| area.id == target_area_id)
                {
                    let duplicate_tab = make_pane_tab(&mut workspace, dragging.tab.kind);
                    if let Some(new_area_id) = split_pane_area_with_tab(
                        &mut workspace,
                        window_index,
                        area_index,
                        zone,
                        duplicate_tab,
                    ) {
                        workspace.last_focused_window = Some(target_window_id.to_string());
                        workspace.last_focused_area = Some(new_area_id);
                    } else {
                        workspace.last_focused_window = Some(target_window_id.to_string());
                        workspace.last_focused_area = Some(target_area_id);
                    }
                    workspace.drop_handled = true;
                    compact_pane_workspace(&mut workspace);
                    refresh_pane_window_titles(&mut workspace);
                    return;
                }
            }
        }

        if dragging.source_was_single_tab {
            remove_pane_tab_from_window_by_id(
                &mut workspace,
                &dragging.source_window_id,
                dragging.source_area_id,
                dragging.tab.id,
            );
        }

        let mut placed_window_id = target_window_id.to_string();
        let mut placed_area_id = target_area_id;
        if let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| window.id == target_window_id)
        {
            if let Some(area_index) = workspace.windows[window_index]
                .areas
                .iter()
                .position(|area| area.id == target_area_id)
            {
                if workspace.windows[window_index].areas[area_index]
                    .tabs
                    .is_empty()
                {
                    let area = &mut workspace.windows[window_index].areas[area_index];
                    area.tabs.push(dragging.tab);
                    area.active = area.tabs.len().saturating_sub(1);
                    placed_area_id = area.id;
                } else if let Some(new_area_id) = split_pane_area_with_tab(
                    &mut workspace,
                    window_index,
                    area_index,
                    zone,
                    dragging.tab.clone(),
                ) {
                    placed_area_id = new_area_id;
                } else {
                    let area = &mut workspace.windows[window_index].areas[area_index];
                    area.tabs.push(dragging.tab);
                    area.active = area.tabs.len().saturating_sub(1);
                    placed_area_id = area.id;
                }
            } else {
                if let Some(area_index) =
                    (!workspace.windows[window_index].areas.is_empty()).then_some(0usize)
                {
                    if workspace.windows[window_index].areas[area_index]
                        .tabs
                        .is_empty()
                    {
                        let area = &mut workspace.windows[window_index].areas[area_index];
                        area.tabs.push(dragging.tab);
                        area.active = area.tabs.len().saturating_sub(1);
                        placed_area_id = area.id;
                    } else if let Some(new_area_id) = split_pane_area_with_tab(
                        &mut workspace,
                        window_index,
                        area_index,
                        zone,
                        dragging.tab.clone(),
                    ) {
                        placed_area_id = new_area_id;
                    } else {
                        let area = &mut workspace.windows[window_index].areas[area_index];
                        area.tabs.push(dragging.tab);
                        area.active = area.tabs.len().saturating_sub(1);
                        placed_area_id = area.id;
                    }
                } else {
                    let area = make_pane_area(
                        &mut workspace,
                        EditorPaneAreaRect::full(),
                        vec![dragging.tab],
                    );
                    placed_area_id = area.id;
                    workspace.windows[window_index].areas.push(area);
                }
            }
        } else {
            let new_window_id = next_pane_window_id(&mut workspace);
            let area = make_pane_area(
                &mut workspace,
                EditorPaneAreaRect::full(),
                vec![dragging.tab],
            );
            placed_area_id = area.id;
            workspace.windows.push(EditorPaneWindow {
                id: new_window_id.clone(),
                title: "Pane".to_string(),
                areas: vec![area],
                layout_managed: false,
            });
            placed_window_id = new_window_id;
        }

        workspace.last_focused_window = Some(placed_window_id);
        workspace.last_focused_area = Some(placed_area_id);
        workspace.drop_handled = true;

        compact_pane_workspace(&mut workspace);
        refresh_pane_window_titles(&mut workspace);
    });
}

fn drop_pane_tab_into_new_window(world: &mut World, pointer_pos: Option<Pos2>) {
    let mut spawned_window_id: Option<String> = None;
    world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
        let Some(dragging) = workspace.dragging.take() else {
            return;
        };
        workspace.initialized = true;

        if dragging.source_was_single_tab {
            remove_pane_tab_from_window_by_id(
                &mut workspace,
                &dragging.source_window_id,
                dragging.source_area_id,
                dragging.tab.id,
            );
        }

        let new_window_id = next_pane_window_id(&mut workspace);
        let area = make_pane_area(
            &mut workspace,
            EditorPaneAreaRect::full(),
            vec![dragging.tab],
        );
        let new_area_id = area.id;
        workspace.windows.push(EditorPaneWindow {
            id: new_window_id.clone(),
            title: "Pane".to_string(),
            areas: vec![area],
            layout_managed: false,
        });
        workspace.last_focused_window = Some(new_window_id.clone());
        workspace.last_focused_area = Some(new_area_id);
        workspace.drop_handled = true;
        spawned_window_id = Some(new_window_id);

        compact_pane_workspace(&mut workspace);
        refresh_pane_window_titles(&mut workspace);
    });

    if let (Some(window_id), Some(pointer_pos)) = (spawned_window_id, pointer_pos) {
        if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
            // bias the window origin up/left so the dropped tab stays under the cursor
            egui_res
                .window_positions
                .insert(window_id, pointer_pos - Vec2::new(120.0, 14.0));
        }
    }
}

fn remove_pane_tab_from_window_by_id(
    workspace: &mut EditorPaneWorkspaceState,
    window_id: &str,
    area_id: u64,
    tab_id: u64,
) {
    let Some(window_index) = workspace
        .windows
        .iter()
        .position(|window| window.id == window_id)
    else {
        return;
    };

    let Some(area_index) = workspace.windows[window_index]
        .areas
        .iter()
        .position(|area| area.id == area_id)
    else {
        return;
    };

    if let Some(index) = workspace.windows[window_index].areas[area_index]
        .tabs
        .iter()
        .position(|tab| tab.id == tab_id)
    {
        workspace.windows[window_index].areas[area_index]
            .tabs
            .remove(index);
        if workspace.windows[window_index].areas[area_index].active
            >= workspace.windows[window_index].areas[area_index].tabs.len()
        {
            workspace.windows[window_index].areas[area_index].active =
                workspace.windows[window_index].areas[area_index]
                    .tabs
                    .len()
                    .saturating_sub(1);
        }
    }
    compact_pane_workspace(workspace);
}

fn pane_drop_zone_geometry(area_rect: Rect) -> (Rect, f32, f32) {
    let inset = area_rect.shrink(6.0);
    let edge_x = (inset.width() * 0.20)
        .clamp(16.0, 64.0)
        .min(inset.width() * 0.40);
    let edge_y = (inset.height() * 0.20)
        .clamp(16.0, 64.0)
        .min(inset.height() * 0.40);
    (inset, edge_x, edge_y)
}

fn pane_drop_zone_rects(area_rect: Rect) -> [(PaneDropZone, Rect); 5] {
    let (inset, edge_x, edge_y) = pane_drop_zone_geometry(area_rect);

    let left = Rect::from_min_max(
        Pos2::new(inset.min.x, inset.min.y),
        Pos2::new(inset.min.x + edge_x, inset.max.y),
    );
    let right = Rect::from_min_max(
        Pos2::new(inset.max.x - edge_x, inset.min.y),
        Pos2::new(inset.max.x, inset.max.y),
    );
    let top = Rect::from_min_max(
        Pos2::new(inset.min.x + edge_x, inset.min.y),
        Pos2::new(inset.max.x - edge_x, inset.min.y + edge_y),
    );
    let bottom = Rect::from_min_max(
        Pos2::new(inset.min.x + edge_x, inset.max.y - edge_y),
        Pos2::new(inset.max.x - edge_x, inset.max.y),
    );
    let center = Rect::from_min_max(
        Pos2::new(inset.min.x + edge_x, inset.min.y + edge_y),
        Pos2::new(inset.max.x - edge_x, inset.max.y - edge_y),
    );

    [
        (PaneDropZone::Left, left),
        (PaneDropZone::Right, right),
        (PaneDropZone::Top, top),
        (PaneDropZone::Bottom, bottom),
        (PaneDropZone::Center, center),
    ]
}

fn pane_drop_zone_for_pointer(area_rect: Rect, pointer_pos: Pos2) -> PaneDropZone {
    let (inset, edge_x, edge_y) = pane_drop_zone_geometry(area_rect);
    if !inset.contains(pointer_pos) {
        return PaneDropZone::Center;
    }

    let left_dist = (pointer_pos.x - inset.left()).max(0.0);
    let right_dist = (inset.right() - pointer_pos.x).max(0.0);
    let top_dist = (pointer_pos.y - inset.top()).max(0.0);
    let bottom_dist = (inset.bottom() - pointer_pos.y).max(0.0);

    let mut candidates: [(PaneDropZone, f32); 4] = [
        (PaneDropZone::Left, f32::INFINITY),
        (PaneDropZone::Right, f32::INFINITY),
        (PaneDropZone::Top, f32::INFINITY),
        (PaneDropZone::Bottom, f32::INFINITY),
    ];
    if left_dist <= edge_x {
        candidates[0].1 = left_dist;
    }
    if right_dist <= edge_x {
        candidates[1].1 = right_dist;
    }
    if top_dist <= edge_y {
        candidates[2].1 = top_dist;
    }
    if bottom_dist <= edge_y {
        candidates[3].1 = bottom_dist;
    }

    let (zone, distance) = candidates
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((PaneDropZone::Center, f32::INFINITY));

    if distance.is_finite() {
        zone
    } else {
        PaneDropZone::Center
    }
}

fn paint_pane_drop_targets(ui: &Ui, area_rect: Rect, hovered_zone: PaneDropZone) {
    for (zone, rect) in pane_drop_zone_rects(area_rect) {
        let is_hovered = zone == hovered_zone;
        let fill = if is_hovered {
            Color32::from_rgba_premultiplied(90, 140, 220, 120)
        } else {
            Color32::from_rgba_premultiplied(60, 75, 100, 80)
        };
        let stroke = if is_hovered {
            Stroke::new(1.5, Color32::from_rgb(155, 205, 255))
        } else {
            Stroke::new(1.0, Color32::from_gray(120))
        };
        ui.painter().rect_filled(rect, 4.0, fill);
        ui.painter()
            .rect_stroke(rect, 4.0, stroke, StrokeKind::Inside);
    }
}

fn draw_pane_area(
    ui: &mut Ui,
    world: &mut World,
    window_id: &str,
    area: EditorPaneArea,
    area_rect: Rect,
) {
    let frame = egui::Frame::none()
        .fill(Color32::from_rgb(17, 18, 22))
        .stroke(Stroke::new(1.0, Color32::from_gray(62)))
        .inner_margin(egui::Margin::same(4));
    ui.allocate_ui_at_rect(area_rect, |ui| {
        frame.show(ui, |ui| {
            let pane_clip_rect = ui.max_rect();
            ui.set_clip_rect(ui.clip_rect().intersect(pane_clip_rect));
            let area_response = ui.interact(
                ui.max_rect(),
                ui.id().with(("pane_area_bg", window_id, area.id)),
                Sense::click(),
            );

            if area_response.clicked_by(PointerButton::Primary) {
                world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
                    workspace.last_focused_window = Some(window_id.to_string());
                    workspace.last_focused_area = Some(area.id);
                });
            }

            let active_kind = area
                .tabs
                .get(area.active.min(area.tabs.len().saturating_sub(1)))
                .map(|tab| tab.kind);

            let mut activate_tab: Option<usize> = None;
            let mut close_tab: Option<usize> = None;
            let mut detach_tab: Option<usize> = None;
            let mut drag_tab: Option<usize> = None;
            let mut drop_on_tab: Option<usize> = None;
            let mut did_interact = false;
            let mut tab_row_bottom = area_response.rect.min.y;
            let mut tab_rects: Vec<Rect> = Vec::new();

            let tab_bar = egui::Frame::none().show(ui, |ui| {
                let old_spacing = ui.spacing().item_spacing;
                ui.spacing_mut().item_spacing = Vec2::new(4.0, old_spacing.y);
                ui.set_min_width(ui.available_width());
                ui.horizontal_wrapped(|ui| {
                    for (index, tab) in area.tabs.iter().enumerate() {
                        let selected = index == area.active;
                        let visuals = ui.visuals().clone();
                        let text_color = if selected {
                            visuals.selection.stroke.color
                        } else {
                            visuals.text_color()
                        };
                        let galley = ui.painter().layout_no_wrap(
                            tab.title.clone(),
                            FontId::proportional(13.0),
                            text_color,
                        );

                        let padding = Vec2::new(10.0, 4.0);
                        let close_width = 24.0;
                        let tab_height = ui
                            .spacing()
                            .interact_size
                            .y
                            .max(galley.size().y + padding.y * 2.0);
                        let tab_width = galley.size().x + padding.x * 2.0 + close_width;
                        let (rect, response) = ui.allocate_exact_size(
                            Vec2::new(tab_width, tab_height),
                            Sense::click_and_drag(),
                        );
                        tab_rects.push(rect);
                        tab_row_bottom = tab_row_bottom.max(rect.max.y);

                        let base_fill = if selected {
                            visuals.selection.bg_fill
                        } else if response.hovered() {
                            visuals.widgets.hovered.bg_fill
                        } else {
                            visuals.widgets.inactive.bg_fill
                        };
                        let stroke = if selected {
                            visuals.selection.stroke
                        } else {
                            visuals.widgets.inactive.bg_stroke
                        };
                        ui.painter().rect_filled(rect, 6.0, base_fill);
                        ui.painter()
                            .rect_stroke(rect, 6.0, stroke, StrokeKind::Inside);

                        ui.painter().galley(rect.min + padding, galley, text_color);

                        let close_rect = Rect::from_min_size(
                            Pos2::new(rect.max.x - close_width, rect.min.y),
                            Vec2::new(close_width, rect.height()),
                        );
                        let close_color = if response.hovered() {
                            visuals.strong_text_color()
                        } else {
                            visuals.text_color()
                        };
                        ui.painter().text(
                            close_rect.center(),
                            Align2::CENTER_CENTER,
                            "x",
                            FontId::proportional(12.0),
                            close_color,
                        );
                        if response.clicked() {
                            if let Some(pointer_pos) = response.interact_pointer_pos() {
                                if close_rect.contains(pointer_pos) {
                                    close_tab = Some(index);
                                } else {
                                    activate_tab = Some(index);
                                }
                            } else {
                                activate_tab = Some(index);
                            }
                            did_interact = true;
                        }
                        if response.drag_started() {
                            drag_tab = Some(index);
                            did_interact = true;
                        }
                        response.dnd_set_drag_payload(PaneTabDragPayload);
                        if typed_dnd_release_payload::<PaneTabDragPayload>(&response).is_some() {
                            drop_on_tab = Some(index);
                            did_interact = true;
                        }
                        response.context_menu(|ui| {
                            let spawn_pos = ui.ctx().input(|input| {
                                input
                                    .pointer
                                    .hover_pos()
                                    .or_else(|| input.pointer.interact_pos())
                            });
                            if ui.button("Close").clicked() {
                                close_tab = Some(index);
                                did_interact = true;
                                ui.close_menu();
                            }
                            if ui.button("Detach").clicked() {
                                detach_tab = Some(index);
                                did_interact = true;
                                ui.close_menu();
                            }
                            ui.separator();
                            draw_pane_spawn_menu(ui, world, window_id, Some(area.id), None);
                            ui.separator();
                            draw_pane_spawn_new_window_menu(ui, world, spawn_pos);
                        });
                    }
                });
                ui.spacing_mut().item_spacing = old_spacing;
            });

            tab_row_bottom = tab_row_bottom.max(tab_bar.response.rect.max.y);
            let tab_bar_rect = Rect::from_min_max(
                Pos2::new(area_response.rect.min.x, area_response.rect.min.y),
                Pos2::new(
                    area_response.rect.max.x,
                    tab_row_bottom.min(area_response.rect.max.y),
                ),
            );
            if tab_bar_rect.width() > 1.0 && tab_bar_rect.height() > 1.0 {
                let mut tab_rects_sorted = tab_rects.clone();
                tab_rects_sorted.sort_by(|a, b| a.min.x.total_cmp(&b.min.x));

                let mut blank_rects: Vec<Rect> = Vec::new();
                if tab_rects_sorted.is_empty() {
                    blank_rects.push(tab_bar_rect);
                } else {
                    let mut cursor_x = tab_bar_rect.left();
                    for tab_rect in tab_rects_sorted {
                        let gap_left = cursor_x.max(tab_bar_rect.left());
                        let gap_right = tab_rect.left().min(tab_bar_rect.right());
                        if gap_right - gap_left > 1.0 {
                            blank_rects.push(Rect::from_min_max(
                                Pos2::new(gap_left, tab_bar_rect.top()),
                                Pos2::new(gap_right, tab_bar_rect.bottom()),
                            ));
                        }
                        cursor_x = cursor_x.max(tab_rect.right().min(tab_bar_rect.right()));
                    }
                    if tab_bar_rect.right() - cursor_x > 1.0 {
                        blank_rects.push(Rect::from_min_max(
                            Pos2::new(cursor_x, tab_bar_rect.top()),
                            Pos2::new(tab_bar_rect.right(), tab_bar_rect.bottom()),
                        ));
                    }
                }

                for (blank_index, blank_rect) in blank_rects.into_iter().enumerate() {
                    let blank_response = ui.interact(
                        blank_rect,
                        ui.id().with((
                            "pane_tab_bar_blank_context",
                            window_id,
                            area.id,
                            blank_index,
                        )),
                        Sense::click(),
                    );
                    blank_response.context_menu(|ui| {
                        draw_pane_area_context_menu(ui, world, window_id, area.id);
                    });
                }
            }

            if let Some(kind) = active_kind {
                if !matches!(
                    kind,
                    EditorPaneKind::Viewport | EditorPaneKind::PlayViewport
                ) {
                    let content_context_rect = Rect::from_min_max(
                        Pos2::new(
                            area_response.rect.min.x,
                            (tab_row_bottom + 1.0).min(area_response.rect.max.y),
                        ),
                        area_response.rect.max,
                    );
                    let content_context_response = ui.interact(
                        content_context_rect,
                        ui.id()
                            .with(("pane_area_content_context", window_id, area.id)),
                        Sense::click_and_drag(),
                    );
                    content_context_response.context_menu(|ui| {
                        draw_pane_area_context_menu(ui, world, window_id, area.id);
                    });
                }
            }

            let split_target_rect = Rect::from_min_max(
                Pos2::new(
                    area_response.rect.min.x,
                    (tab_row_bottom + 4.0).min(area_response.rect.max.y),
                ),
                area_response.rect.max,
            );
            let pane_drop_zone = world
                .get_resource::<EditorPaneWorkspaceState>()
                .and_then(|workspace| workspace.dragging.as_ref())
                .and_then(|_| ui.ctx().input(|input| input.pointer.hover_pos()))
                .filter(|pos| split_target_rect.contains(*pos))
                .map(|pos| pane_drop_zone_for_pointer(split_target_rect, pos));

            if let Some(index) = drag_tab {
                begin_pane_tab_drag(world, window_id, area.id, index);
            }
            if let Some(index) = detach_tab {
                detach_pane_tab_to_new_window(world, window_id, area.id, index);
            }
            if let Some(index) = close_tab {
                close_pane_tab_in_window(world, window_id, area.id, index);
            }
            if let Some(index) = activate_tab {
                world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
                    let mut focused_area = None;
                    if let Some(window_index) =
                        workspace.windows.iter().position(|w| w.id == window_id)
                    {
                        if let Some(area_index) = workspace.windows[window_index]
                            .areas
                            .iter()
                            .position(|a| a.id == area.id)
                        {
                            let area = &mut workspace.windows[window_index].areas[area_index];
                            if index < area.tabs.len() {
                                area.active = index;
                            }
                            focused_area = Some(area.id);
                        }
                    }
                    if let Some(area_id) = focused_area {
                        workspace.last_focused_window = Some(window_id.to_string());
                        workspace.last_focused_area = Some(area_id);
                    }
                    refresh_pane_window_titles(&mut workspace);
                });
            }
            if did_interact {
                world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
                    workspace.last_focused_window = Some(window_id.to_string());
                    workspace.last_focused_area = Some(area.id);
                });
            }

            if let Some(index) = drop_on_tab {
                accept_pane_tab_drop(world, window_id, area.id, Some(index));
            } else {
                let released_on_tab_bar =
                    typed_dnd_release_payload::<PaneTabDragPayload>(&tab_bar.response).is_some();
                let released_on_area =
                    typed_dnd_release_payload::<PaneTabDragPayload>(&area_response).is_some();
                if released_on_tab_bar || released_on_area {
                    if let Some(zone) = pane_drop_zone {
                        let effective_zone = if area.tabs.is_empty() {
                            PaneDropZone::Center
                        } else {
                            zone
                        };
                        match effective_zone {
                            PaneDropZone::Center => {
                                accept_pane_tab_drop(world, window_id, area.id, None);
                            }
                            split_zone => {
                                accept_pane_tab_split_drop(world, window_id, area.id, split_zone);
                            }
                        }
                    } else {
                        accept_pane_tab_drop(world, window_id, area.id, None);
                    }
                }
            }

            ui.separator();

            let active_tab =
                world.resource_scope::<EditorPaneWorkspaceState, _>(|_world, mut workspace| {
                    let window = workspace
                        .windows
                        .iter_mut()
                        .find(|window| window.id == window_id)?;
                    let area = window
                        .areas
                        .iter_mut()
                        .find(|target| target.id == area.id)?;
                    if area.tabs.is_empty() {
                        return None;
                    }
                    if area.active >= area.tabs.len() {
                        area.active = area.tabs.len().saturating_sub(1);
                    }
                    Some(area.tabs[area.active].clone())
                });

            if let Some(active_tab) = active_tab {
                ui.push_id((window_id, area.id, active_tab.id), |ui| {
                    match active_tab.kind {
                        EditorPaneKind::Toolbar => draw_toolbar(ui, world),
                        EditorPaneKind::Viewport => {
                            draw_viewport_pane(ui, world, active_tab.id, false)
                        }
                        EditorPaneKind::PlayViewport => {
                            draw_viewport_pane(ui, world, active_tab.id, true)
                        }
                        EditorPaneKind::Project => draw_project_window(ui, world),
                        EditorPaneKind::Hierarchy => draw_scene_window(ui, world),
                        EditorPaneKind::Inspector => draw_inspector_window(ui, world),
                        EditorPaneKind::History => draw_history_window(ui, world),
                        EditorPaneKind::Timeline => draw_timeline_window(ui, world),
                        EditorPaneKind::ContentBrowser => draw_assets_window(ui, world),
                        EditorPaneKind::Console => draw_console_window(ui, world),
                        EditorPaneKind::AudioMixer => draw_audio_mixer_window(ui, world),
                    }
                });
            } else {
                let empty_body_rect = Rect::from_min_max(
                    Pos2::new(
                        area_response.rect.min.x,
                        (tab_row_bottom + 1.0).min(area_response.rect.max.y),
                    ),
                    area_response.rect.max,
                );
                if empty_body_rect.width() > 1.0 && empty_body_rect.height() > 1.0 {
                    ui.allocate_ui_at_rect(empty_body_rect, |ui| {
                        let empty_body_response = ui.allocate_rect(ui.max_rect(), Sense::click());
                        empty_body_response.context_menu(|ui| {
                            draw_pane_area_context_menu(ui, world, window_id, area.id);
                        });
                        ui.with_layout(
                            Layout::centered_and_justified(egui::Direction::TopDown),
                            |ui| {
                                ui.label("Right-click to add panes");
                            },
                        );
                    });
                } else {
                    area_response.context_menu(|ui| {
                        draw_pane_area_context_menu(ui, world, window_id, area.id);
                    });
                    ui.with_layout(
                        Layout::centered_and_justified(egui::Direction::TopDown),
                        |ui| {
                            ui.label("Right-click to add panes");
                        },
                    );
                }
            }

            if let Some(zone) = pane_drop_zone {
                paint_pane_drop_targets(ui, split_target_rect, zone);
            }
        });
    });
}

fn draw_pane_spawn_menu(
    ui: &mut Ui,
    world: &mut World,
    window_id: &str,
    area_id: Option<u64>,
    split_axis: Option<SplitAxis>,
) {
    let label = match split_axis {
        Some(SplitAxis::Vertical) => "Split Right",
        Some(SplitAxis::Horizontal) => "Split Down",
        None => "New Pane",
    };
    ui.menu_button(label, |ui| {
        for kind in EditorPaneKind::ALL {
            if ui.button(kind.label()).clicked() {
                open_pane_workspace_tab(world, kind, Some(window_id), area_id, split_axis);
                ui.close_menu();
            }
        }
    });
}

pub fn draw_editor_window(ui: &mut Ui, world: &mut World, window_id: u64) {
    bring_window_to_front_if_dragging(ui, world);
    drag_egui_window_on_middle_click(ui, world, &format!("editor_window_{}", window_id));

    let middle_drag_active = world
        .get_resource::<MiddleDragUiState>()
        .map(|state| state.active)
        .unwrap_or(false);
    if middle_drag_active {
        ui.add_enabled_ui(false, |ui| {
            draw_editor_window_contents(ui, world, window_id);
        });
    } else {
        draw_editor_window_contents(ui, world, window_id);
    }
}

fn draw_editor_window_contents(ui: &mut Ui, world: &mut World, window_id: u64) {
    let (tabs_snapshot, active_index, window_index, window_count) = {
        let Some(state) = world.get_resource::<EditorWorkspaceState>() else {
            ui.label("No editors available.");
            return;
        };
        let Some((index, window)) = state
            .windows
            .iter()
            .enumerate()
            .find(|(_, window)| window.id == window_id)
        else {
            ui.label("Editor window missing.");
            return;
        };
        (
            window.tabs.clone(),
            window.active,
            index,
            state.windows.len(),
        )
    };
    let pane_window_count = world
        .get_resource::<EditorPaneWorkspaceState>()
        .map(|state| state.windows.len())
        .unwrap_or(0);
    if window_index == 0 && pane_window_count == 0 {
        draw_pane_workspace_background_context_menu(ui, world);
    }

    if tabs_snapshot.is_empty() {
        world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
            if let Some(index) = workspace
                .windows
                .iter()
                .position(|window| window.id == window_id)
            {
                let is_drag_source = workspace
                    .dragging
                    .as_ref()
                    .map(|drag| drag.source_window_id == window_id)
                    .unwrap_or(false);
                if !is_drag_source {
                    workspace.windows.remove(index);
                    refresh_editor_window_titles(&mut workspace);
                }
            }
        });
        return;
    }

    let mut activate_tab: Option<usize> = None;
    let mut close_tab: Option<usize> = None;
    let mut detach_tab: Option<usize> = None;
    let mut drag_tab: Option<usize> = None;
    let mut interacted = false;

    let mut drop_on_tab: Option<usize> = None;
    let can_drag_tabs = true;
    let tab_bar = egui::Frame::none().show(ui, |ui| {
        let old_spacing = ui.spacing().item_spacing;
        ui.spacing_mut().item_spacing = Vec2::new(4.0, old_spacing.y);
        ui.horizontal_wrapped(|ui| {
            for (index, tab) in tabs_snapshot.iter().enumerate() {
                let selected = index == active_index;
                let visuals = ui.visuals().clone();
                let text_color = if selected {
                    visuals.selection.stroke.color
                } else {
                    visuals.text_color()
                };
                let galley = ui.painter().layout_no_wrap(
                    tab.title.clone(),
                    FontId::proportional(13.0),
                    text_color,
                );

                let padding = Vec2::new(10.0, 4.0);
                let close_width = 24.0;
                let tab_height = ui
                    .spacing()
                    .interact_size
                    .y
                    .max(galley.size().y + padding.y * 2.0);
                let tab_width = galley.size().x + padding.x * 2.0 + close_width;

                let (rect, response) = ui.allocate_exact_size(
                    Vec2::new(tab_width, tab_height),
                    if can_drag_tabs {
                        Sense::click_and_drag()
                    } else {
                        Sense::click()
                    },
                );

                let base_fill = if selected {
                    visuals.selection.bg_fill
                } else if response.hovered() {
                    visuals.widgets.hovered.bg_fill
                } else {
                    visuals.widgets.inactive.bg_fill
                };
                let stroke = if selected {
                    visuals.selection.stroke
                } else {
                    visuals.widgets.inactive.bg_stroke
                };
                ui.painter().rect_filled(rect, 6.0, base_fill);
                ui.painter()
                    .rect_stroke(rect, 6.0, stroke, StrokeKind::Inside);

                let text_pos = rect.min + padding;
                ui.painter().galley(text_pos, galley, text_color);

                let close_rect = Rect::from_min_size(
                    Pos2::new(rect.max.x - close_width, rect.min.y),
                    Vec2::new(close_width, rect.height()),
                );
                let close_center = close_rect.center();
                let close_hovered = ui
                    .ctx()
                    .input(|input| input.pointer.hover_pos())
                    .map(|pos| close_rect.contains(pos))
                    .unwrap_or(false);
                if close_hovered {
                    let close_bg = visuals.warn_fg_color.linear_multiply(0.48);
                    let close_rect = close_rect.shrink(2.0);
                    ui.painter().rect_filled(close_rect, 4.0, close_bg);
                }
                ui.painter().text(
                    close_center,
                    Align2::CENTER_CENTER,
                    "x",
                    FontId::proportional(14.0),
                    visuals.text_color(),
                );

                if can_drag_tabs {
                    response.dnd_set_drag_payload(TabDragPayload);
                }

                if response.clicked() {
                    if let Some(pointer_pos) = response.interact_pointer_pos() {
                        if close_rect.contains(pointer_pos) {
                            close_tab = Some(index);
                        } else {
                            activate_tab = Some(index);
                        }
                    } else {
                        activate_tab = Some(index);
                    }
                    interacted = true;
                }

                if can_drag_tabs && response.drag_started() {
                    drag_tab = Some(index);
                    interacted = true;
                }

                if typed_dnd_release_payload::<TabDragPayload>(&response).is_some() {
                    drop_on_tab = Some(index);
                    interacted = true;
                }

                response.context_menu(|ui| {
                    if ui.button("Close").clicked() {
                        close_tab = Some(index);
                        interacted = true;
                        ui.close_menu();
                    }
                    if ui.button("Detach").clicked() {
                        detach_tab = Some(index);
                        interacted = true;
                        ui.close_menu();
                    }
                });
            }
        });
        ui.spacing_mut().item_spacing = old_spacing;
    });

    if drop_on_tab.is_some() {
        accept_tab_drop(world, window_id, drop_on_tab);
        interacted = true;
    } else if typed_dnd_release_payload::<TabDragPayload>(&tab_bar.response).is_some() {
        accept_tab_drop(world, window_id, None);
        interacted = true;
    }

    ui.separator();

    if let Some(index) = drag_tab {
        begin_tab_drag(world, window_id, index);
    }

    if let Some(index) = detach_tab {
        detach_tab_to_new_window(world, window_id, index);
    }

    if let Some(index) = close_tab {
        close_tab_in_window(world, window_id, index);
    }

    let active_tab = world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
        let window_index = workspace
            .windows
            .iter()
            .position(|window| window.id == window_id)?;

        if interacted {
            workspace.last_focused_window = Some(window_id);
        }

        if let Some(index) = activate_tab {
            if index < workspace.windows[window_index].tabs.len() {
                workspace.windows[window_index].active = index;
            }
        }

        let is_drag_source_empty = workspace.windows[window_index].tabs.is_empty()
            && workspace
                .dragging
                .as_ref()
                .map(|drag| drag.source_window_id == window_id)
                .unwrap_or(false);
        if workspace.windows[window_index].tabs.is_empty() && !is_drag_source_empty {
            workspace.windows.remove(window_index);
            refresh_editor_window_titles(&mut workspace);
            return None;
        }

        if workspace.windows[window_index].tabs.is_empty() {
            return None;
        }

        if workspace.windows[window_index].active >= workspace.windows[window_index].tabs.len() {
            workspace.windows[window_index].active =
                workspace.windows[window_index].tabs.len().saturating_sub(1);
        }

        Some(workspace.windows[window_index].tabs[workspace.windows[window_index].active].clone())
    });

    let Some(active_tab) = active_tab else {
        if world
            .get_resource::<EditorWorkspaceState>()
            .and_then(|state| state.dragging.as_ref())
            .map(|drag| drag.source_window_id == window_id)
            .unwrap_or(false)
        {
            ui.label("Drop tab to close this editor.");
        } else {
            ui.label("No tabs open.");
        }
        return;
    };

    if let Some(dragging) = world
        .get_resource::<EditorWorkspaceState>()
        .and_then(|state| state.dragging.clone())
    {
        if let Some(pointer_pos) = ui.ctx().input(|input| input.pointer.hover_pos()) {
            let painter = ui.ctx().layer_painter(egui::LayerId::new(
                Order::Tooltip,
                ui.id().with("editor_tab_drag"),
            ));
            painter.rect_filled(
                Rect::from_min_size(pointer_pos + Vec2::new(12.0, 12.0), Vec2::new(160.0, 28.0)),
                6.0,
                ui.visuals().widgets.active.bg_fill,
            );
            painter.text(
                pointer_pos + Vec2::new(20.0, 26.0),
                Align2::LEFT_CENTER,
                dragging.tab.title,
                FontId::proportional(13.0),
                ui.visuals().text_color(),
            );
        }
    }

    if world
        .get_resource::<EditorWorkspaceState>()
        .map(|state| state.dragging.is_some() && !state.drop_handled)
        .unwrap_or(false)
        && ui.ctx().input(|input| input.pointer.any_released())
    {
        let over_this_window = ui
            .ctx()
            .input(|input| input.pointer.hover_pos())
            .and_then(|pos| ui.ctx().layer_id_at(pos))
            .map(|layer_id| layer_id == ui.layer_id())
            .unwrap_or(false);
        if over_this_window {
            accept_tab_drop(world, window_id, None);
        } else if window_index + 1 == window_count {
            drop_tab_into_new_window(world);
        }
    }

    let project = world.get_resource::<EditorProject>().cloned();
    match active_tab.content {
        EditorTabContent::Material { path } => {
            draw_material_editor_tab(ui, world, &project, &path);
        }
    }
}

pub fn draw_hierarchy_window(ui: &mut Ui, world: &mut World) {
    draw_scene_window(ui, world);
}

#[derive(Default)]
struct HierarchyData {
    labels: HashMap<Entity, String>,
    children: HashMap<Entity, Vec<Entity>>,
    roots: Vec<Entity>,
}

fn collect_hierarchy_entries(world: &mut World) -> HierarchyData {
    let mut rows: Vec<(Entity, String, Option<Entity>)> = Vec::new();
    let mut query = world.query::<(
        Entity,
        Option<&Name>,
        Option<&SceneRoot>,
        Option<&SceneChild>,
        Option<&SceneAssetPath>,
        Option<&EditorEntity>,
        Option<&EntityParent>,
        Option<&EditorPlayCamera>,
    )>();

    for (
        entity,
        name,
        scene_root,
        scene_child,
        scene_asset,
        editor_entity,
        relation,
        active_camera,
    ) in query.iter(world)
    {
        if editor_entity.is_none() && scene_child.is_none() {
            continue;
        }

        let mut label = name
            .map(|name| name.to_string())
            .unwrap_or_else(|| format!("Entity {}", entity.to_bits()));

        let camera = world.get::<BevyCamera>(entity);
        let light = world.get::<BevyLight>(entity);
        let mesh = world.get::<BevyMeshRenderer>(entity);
        let skinned = world.get::<BevySkinnedMeshRenderer>(entity);
        let editor_skinned = world.get::<EditorSkinnedMesh>(entity);
        let editor_mesh = world.get::<EditorMesh>(entity);
        let script = world.get::<ScriptComponent>(entity);
        let dynamic = world.get::<DynamicComponents>(entity);

        let mut tags = Vec::new();
        if camera.is_some() {
            if active_camera.is_some() {
                tags.push("Camera*");
            } else {
                tags.push("Camera");
            }
        }
        if light.is_some() {
            tags.push("Light");
        }
        if mesh.is_some() {
            let mesh_tag = editor_mesh
                .map(|mesh| match &mesh.source {
                    MeshSource::Primitive(PrimitiveKind::Cube) => "Cube",
                    MeshSource::Primitive(PrimitiveKind::UvSphere(_, _)) => "UV Sphere",
                    MeshSource::Primitive(PrimitiveKind::Plane) => "Plane",
                    MeshSource::Asset { .. } => "Mesh",
                })
                .unwrap_or("Mesh");
            tags.push(mesh_tag);
        }
        if skinned.is_some() || editor_skinned.is_some() {
            tags.push("Skinned");
        }
        if scene_root.is_some() {
            if let Some(scene) = scene_asset {
                if let Some(name) = scene.path.file_name().and_then(|name| name.to_str()) {
                    label = format!("{} ({})", label, name);
                }
            }
            tags.push("Scene");
        }
        if scene_child.is_some() {
            tags.push("Scene Child");
        }
        if script.is_some() {
            tags.push("Script");
        }
        if dynamic.is_some() {
            tags.push("Dynamic");
        }

        if !tags.is_empty() {
            label.push_str(" [");
            label.push_str(&tags.join(", "));
            label.push(']');
        }

        rows.push((entity, label, relation.map(|relation| relation.parent)));
    }

    rows.sort_by_key(|(entity, _, _)| entity.to_bits());
    let included: HashSet<Entity> = rows.iter().map(|(entity, _, _)| *entity).collect();
    let mut data = HierarchyData::default();

    for (entity, label, parent) in rows {
        data.labels.insert(entity, label);
        let valid_parent = parent.filter(|parent| *parent != entity && included.contains(parent));
        if let Some(parent) = valid_parent {
            data.children.entry(parent).or_default().push(entity);
        } else {
            data.roots.push(entity);
        }
    }

    for children in data.children.values_mut() {
        children.sort_by_key(|entity| entity.to_bits());
    }
    data.roots.sort_by_key(|entity| entity.to_bits());
    data
}

fn draw_hierarchy_panel(ui: &mut Ui, world: &mut World, data: &HierarchyData) {
    let selection = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selection| selection.0);
    if let Some(selected) = selection {
        expand_hierarchy_ancestors(world, selected);
    }

    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
    egui::ScrollArea::vertical()
        .id_salt("hierarchy_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let mut visited = HashSet::new();
            let mut drop_handled = false;
            for root in data.roots.iter().copied() {
                draw_hierarchy_row(
                    ui,
                    world,
                    data,
                    root,
                    0,
                    selection,
                    &mut visited,
                    &mut drop_handled,
                );
            }

            let drop_rect = ui.max_rect();
            let drop_response = ui.interact(
                drop_rect,
                ui.id().with("hierarchy_background_drop"),
                Sense::hover(),
            );
            if !drop_handled {
                if let Some(payload) =
                    typed_dnd_release_payload::<EntityDragPayload>(&drop_response)
                {
                    reparent_entity_in_hierarchy(world, payload.entity, None);
                }
            }
            highlight_drop_target(ui, &drop_response);
        });
}

fn draw_hierarchy_row(
    ui: &mut Ui,
    world: &mut World,
    data: &HierarchyData,
    entity: Entity,
    depth: usize,
    selection: Option<Entity>,
    visited: &mut HashSet<Entity>,
    drop_handled: &mut bool,
) {
    if !visited.insert(entity) {
        return;
    }

    const INDENT: f32 = 14.0;
    const TOGGLE_WIDTH: f32 = 14.0;

    let label = data
        .labels
        .get(&entity)
        .cloned()
        .unwrap_or_else(|| format!("Entity {}", entity.to_bits()));
    let children = data.children.get(&entity).cloned().unwrap_or_default();
    let can_expand = !children.is_empty();
    let is_selected = selection == Some(entity);
    let mut expanded = if can_expand {
        world
            .get_resource::<HierarchyUiState>()
            .map(|state| state.expanded_entities.contains(&entity))
            .unwrap_or(false)
    } else {
        false
    };
    let is_renaming = world
        .get_resource::<HierarchyUiState>()
        .map(|state| state.rename_entity == Some(entity))
        .unwrap_or(false);

    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * INDENT);
        if can_expand {
            let toggle_label = if expanded { "v" } else { ">" };
            let toggle_response = ui.add_sized(
                Vec2::new(TOGGLE_WIDTH, 0.0),
                egui::Button::new(toggle_label).frame(false),
            );
            if toggle_response.clicked() {
                expanded = !expanded;
                world.resource_scope::<HierarchyUiState, _>(|_world, mut ui_state| {
                    if expanded {
                        ui_state.expanded_entities.insert(entity);
                    } else {
                        ui_state.expanded_entities.remove(&entity);
                    }
                });
            }
        } else {
            ui.add_space(TOGGLE_WIDTH);
        }

        if is_renaming {
            world.resource_scope::<HierarchyUiState, _>(|world, mut ui_state| {
                let response = ui.text_edit_singleline(&mut ui_state.rename_buffer);
                if ui_state.rename_request_focus {
                    response.request_focus();
                    ui_state.rename_request_focus = false;
                }
                let commit = response.lost_focus()
                    || (response.has_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter)));
                if commit {
                    apply_entity_name(world, entity, ui_state.rename_buffer.trim());
                    ui_state.rename_entity = None;
                }
            });
        } else {
            let response = ui.add_sized(
                Vec2::new(ui.available_width(), 0.0),
                egui::Button::new(label)
                    .wrap()
                    .selected(is_selected)
                    .sense(Sense::click_and_drag()),
            );

            let drag_started = response.drag_started_by(PointerButton::Primary);
            let drag_stopped = response.drag_stopped_by(PointerButton::Primary);
            if drag_started || drag_stopped {
                if let Some(mut drag_state) = world.get_resource_mut::<EntityDragState>() {
                    if drag_started {
                        drag_state.start_drag(entity);
                    }
                    if drag_stopped {
                        drag_state.stop_drag();
                    }
                }
            }
            response.dnd_set_drag_payload(EntityDragPayload { entity });

            if response.clicked() {
                set_selection(world, Some(entity));
            }
            if response.double_clicked() {
                focus_entity_in_view(world, entity);
            }

            response.context_menu(|ui| {
                if ui.button("Rename").clicked() {
                    begin_rename(world, entity);
                    ui.close_menu();
                }
                if ui.button("Delete").clicked() {
                    push_command(world, EditorCommand::DeleteEntity { entity });
                    ui.close_menu();
                }
                if ui.button("Focus").clicked() {
                    focus_entity_in_view(world, entity);
                    ui.close_menu();
                }
                if world.get::<BevyCamera>(entity).is_some()
                    && ui.button("Set Game Camera").clicked()
                {
                    push_command(world, EditorCommand::SetActiveCamera { entity });
                    ui.close_menu();
                }
                if let Some(scene_child) = world.get::<SceneChild>(entity) {
                    if ui.button("Select Scene Root").clicked() {
                        set_selection(world, Some(scene_child.scene_root));
                        ui.close_menu();
                    }
                }
            });

            if let Some(payload) = typed_dnd_release_payload::<EntityDragPayload>(&response) {
                *drop_handled = true;
                if payload.entity != entity {
                    reparent_entity_in_hierarchy(world, payload.entity, Some(entity));
                }
            }
            highlight_drop_target(ui, &response);

            let is_dragging = world
                .get_resource::<EntityDragState>()
                .map(|state| state.active && state.entity == Some(entity))
                .unwrap_or(false);
            if is_dragging {
                let stroke = Stroke::new(
                    ui.visuals().selection.stroke.width.max(1.5),
                    ui.visuals().selection.stroke.color,
                );
                let rounding = ui.visuals().widgets.active.rounding();
                ui.painter()
                    .rect_stroke(response.rect, rounding, stroke, StrokeKind::Inside);
            }
        }
    });

    if expanded {
        for child in children {
            draw_hierarchy_row(
                ui,
                world,
                data,
                child,
                depth + 1,
                selection,
                visited,
                drop_handled,
            );
        }
    }
}

fn expand_hierarchy_ancestors(world: &mut World, entity: Entity) {
    let mut ancestors = Vec::new();
    let mut current = entity;
    let mut visited = HashSet::new();
    while let Some(parent) = world
        .get::<EntityParent>(current)
        .map(|relation| relation.parent)
    {
        if !visited.insert(parent) {
            break;
        }
        ancestors.push(parent);
        current = parent;
    }
    if ancestors.is_empty() {
        return;
    }
    world.resource_scope::<HierarchyUiState, _>(|_world, mut ui_state| {
        for ancestor in ancestors {
            ui_state.expanded_entities.insert(ancestor);
        }
    });
}

fn remove_scene_child_mapping(world: &mut World, scene_root: Entity, child_entity: Entity) {
    if let Some(mut spawned) = world.get_resource_mut::<SceneSpawnedChildren>() {
        if let Some(children) = spawned.spawned_scenes.get_mut(&scene_root) {
            children.retain(|entity| *entity != child_entity);
        }
    }
}

fn ensure_scene_child_mapping(world: &mut World, scene_root: Entity, child_entity: Entity) {
    if let Some(mut spawned) = world.get_resource_mut::<SceneSpawnedChildren>() {
        let children = spawned.spawned_scenes.entry(scene_root).or_default();
        if !children.contains(&child_entity) {
            children.push(child_entity);
        }
    }
}

fn is_descendant_of(world: &World, candidate: Entity, ancestor: Entity) -> bool {
    let mut current = Some(candidate);
    let mut visited = HashSet::new();
    while let Some(entity) = current {
        if !visited.insert(entity) {
            return false;
        }
        if entity == ancestor {
            return true;
        }
        current = world
            .get::<EntityParent>(entity)
            .map(|relation| relation.parent);
    }
    false
}

fn reparent_entity_in_hierarchy(
    world: &mut World,
    child: Entity,
    new_parent: Option<Entity>,
) -> bool {
    if world.get_entity(child).is_err() {
        return false;
    }

    if let Some(parent) = new_parent {
        if parent == child {
            set_status(world, "Cannot parent an entity to itself".to_string());
            return false;
        }
        if world.get_entity(parent).is_err() {
            return false;
        }
        if is_descendant_of(world, parent, child) {
            set_status(
                world,
                "Cannot parent an entity to one of its descendants".to_string(),
            );
            return false;
        }
    }

    let current_parent = world
        .get::<EntityParent>(child)
        .map(|relation| relation.parent);
    if current_parent == new_parent {
        return false;
    }

    let previous_scene_child = world.get::<SceneChild>(child).copied();
    let child_transform = world
        .get::<BevyTransform>(child)
        .map(|transform| transform.0)
        .unwrap_or_default();

    if let Some(parent) = new_parent {
        let parent_matrix = world
            .get::<BevyTransform>(parent)
            .map(|transform| transform.0.to_matrix())
            .unwrap_or(glam::Mat4::IDENTITY);
        world.entity_mut(child).insert(EntityParent {
            parent,
            local_transform: parent_matrix.inverse() * child_transform.to_matrix(),
            last_written: child_transform,
        });
    } else {
        world.entity_mut(child).remove::<EntityParent>();
    }

    if let Some(scene_child) = previous_scene_child {
        let keep_scene_link = new_parent
            .and_then(|parent| {
                if world.get::<SceneRoot>(parent).is_some() {
                    Some(parent)
                } else {
                    world
                        .get::<SceneChild>(parent)
                        .map(|child| child.scene_root)
                }
            })
            .map(|scene_root| scene_root == scene_child.scene_root)
            .unwrap_or(false);
        if keep_scene_link {
            ensure_scene_child_mapping(world, scene_child.scene_root, child);
        } else {
            remove_scene_child_mapping(world, scene_child.scene_root, child);
            let mut entity = world.entity_mut(child);
            entity.remove::<SceneChild>();
            entity.insert(EditorEntity);
        }
    }

    if let Some(parent) = new_parent {
        world.resource_scope::<HierarchyUiState, _>(|_world, mut ui_state| {
            ui_state.expanded_entities.insert(parent);
        });
    }

    push_undo_snapshot(
        world,
        if new_parent.is_some() {
            "Reparent Entity"
        } else {
            "Unparent Entity"
        },
    );
    true
}

fn draw_inspector_header(ui: &mut Ui, world: &mut World, entity: Entity) {
    let name_snapshot = world
        .get::<Name>(entity)
        .map(|name| name.to_string())
        .unwrap_or_default();

    let mut commit_name = false;
    world.resource_scope::<InspectorNameEditState, _>(|_world, mut name_state| {
        if name_state.entity != Some(entity) {
            name_state.entity = Some(entity);
            name_state.buffer = name_snapshot.clone();
        }

        ui.horizontal(|ui| {
            ui.label("Name");
            let response = ui.text_edit_singleline(&mut name_state.buffer);
            if response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                response.surrender_focus();
                commit_name = true;
            } else if response.lost_focus() {
                commit_name = true;
            }
        });
    });

    if commit_name {
        if let Some(mut name_state) = world.get_resource_mut::<InspectorNameEditState>() {
            let trimmed = name_state.buffer.trim().to_string();
            name_state.buffer = trimmed.clone();
            apply_entity_name(world, entity, &trimmed);
        }
    }

    ui.horizontal(|ui| {
        ui.label(format!("ID: {}", entity.to_bits()));
        if ui.button("Delete").clicked() {
            push_command(world, EditorCommand::DeleteEntity { entity });
        }
    });
}

fn draw_inspector_panel(ui: &mut Ui, world: &mut World, entity: Entity) {
    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);

    let selected_asset = world
        .get_resource::<AssetBrowserState>()
        .and_then(|state| state.selected.clone());
    let project = world.get_resource::<EditorProject>().cloned();

    if world.get::<BevyTransform>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Transform");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyTransform>();
            push_undo_snapshot(world, "Remove Transform");
        } else if let Some(transform) = world
            .get::<BevyTransform>(entity)
            .map(|transform| transform.0)
        {
            let mut position = transform.position;
            let position_response = edit_vec3(ui, "Position", &mut position, 0.1);
            begin_edit_undo(world, "Move", position_response);
            if position_response.changed {
                if let Some(mut transform) = world.get_mut::<BevyTransform>(entity) {
                    transform.0.position = position;
                }
            }
            end_edit_undo(world, position_response);

            let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
            let mut rotation = Vec3::new(yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees());
            let rotation_response = edit_vec3(ui, "Rotation", &mut rotation, 0.5);
            begin_edit_undo(world, "Rotate", rotation_response);
            if rotation_response.changed {
                if let Some(mut transform) = world.get_mut::<BevyTransform>(entity) {
                    transform.0.rotation = Quat::from_euler(
                        EulerRot::YXZ,
                        rotation.x.to_radians(),
                        rotation.y.to_radians(),
                        rotation.z.to_radians(),
                    );
                }
            }
            end_edit_undo(world, rotation_response);

            let mut scale = transform.scale;
            let scale_response = edit_vec3(ui, "Scale", &mut scale, 0.05);
            begin_edit_undo(world, "Scale", scale_response);
            if scale_response.changed {
                if let Some(mut transform) = world.get_mut::<BevyTransform>(entity) {
                    transform.0.scale = scale;
                }
            }
            end_edit_undo(world, scale_response);
        }
        ui.separator();
    }

    if world.get::<BevyCamera>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Camera");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyCamera>();
            world.entity_mut(entity).remove::<EditorPlayCamera>();
            world.entity_mut(entity).remove::<BevyActiveCamera>();
            world.entity_mut(entity).remove::<Freecam>();
            push_undo_snapshot(world, "Remove Camera");
        } else {
            let is_active = world.get::<EditorPlayCamera>(entity).is_some();
            if !is_active {
                if ui.button("Set Game Camera").clicked() {
                    push_command(world, EditorCommand::SetActiveCamera { entity });
                }
            } else {
                ui.label("Game Camera");
            }

            if let Some(camera) = world.get::<BevyCamera>(entity).map(|camera| camera.0) {
                let mut fov = camera.fov_y_rad.to_degrees();
                let fov_response = edit_float(ui, "FOV (deg)", &mut fov, 0.25);
                begin_edit_undo(world, "Camera", fov_response);
                if fov_response.changed {
                    if let Some(mut camera) = world.get_mut::<BevyCamera>(entity) {
                        camera.0.fov_y_rad = fov.to_radians();
                    }
                }
                end_edit_undo(world, fov_response);

                let mut aspect_ratio = camera.aspect_ratio;
                let aspect_response = edit_float(ui, "Aspect Ratio", &mut aspect_ratio, 0.01);
                begin_edit_undo(world, "Camera", aspect_response);
                if aspect_response.changed {
                    if let Some(mut camera) = world.get_mut::<BevyCamera>(entity) {
                        camera.0.aspect_ratio = aspect_ratio;
                    }
                }
                end_edit_undo(world, aspect_response);

                let mut near_plane = camera.near_plane;
                let near_response = edit_float(ui, "Near", &mut near_plane, 0.01);
                begin_edit_undo(world, "Camera", near_response);
                if near_response.changed {
                    if let Some(mut camera) = world.get_mut::<BevyCamera>(entity) {
                        camera.0.near_plane = near_plane;
                    }
                }
                end_edit_undo(world, near_response);

                let mut far_plane = camera.far_plane;
                let far_response = edit_float(ui, "Far", &mut far_plane, 1.0);
                begin_edit_undo(world, "Camera", far_response);
                if far_response.changed {
                    if let Some(mut camera) = world.get_mut::<BevyCamera>(entity) {
                        camera.0.far_plane = far_plane;
                    }
                }
                end_edit_undo(world, far_response);
            }
        }
        ui.separator();
    }

    if world.get::<Freecam>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Freecam");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });
        if remove {
            world.entity_mut(entity).remove::<Freecam>();
            push_undo_snapshot(world, "Remove Freecam");
        }
        ui.separator();
    }

    if world.get::<BevyLight>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Light");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyLight>();
            push_undo_snapshot(world, "Remove Light");
        } else if let Some(light) = world.get::<BevyLight>(entity).map(|light| light.0) {
            let mut light_type = light.light_type;
            let mut type_changed = false;
            let current_label = match light_type {
                LightType::Directional => "Directional",
                LightType::Point => "Point",
                LightType::Spot { .. } => "Spot",
            };

            ComboBox::from_id_source(format!("light_kind_{}", entity.to_bits()))
                .selected_text(current_label)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(
                            matches!(light_type, LightType::Directional),
                            "Directional",
                        )
                        .clicked()
                    {
                        light_type = LightType::Directional;
                        type_changed = true;
                    }
                    if ui
                        .selectable_label(matches!(light_type, LightType::Point), "Point")
                        .clicked()
                    {
                        light_type = LightType::Point;
                        type_changed = true;
                    }
                    if ui
                        .selectable_label(matches!(light_type, LightType::Spot { .. }), "Spot")
                        .clicked()
                    {
                        let angle = match light_type {
                            LightType::Spot { angle } => angle,
                            _ => 45.0_f32.to_radians(),
                        };
                        light_type = LightType::Spot { angle };
                        type_changed = true;
                    }
                });

            if type_changed {
                if let Some(mut light) = world.get_mut::<BevyLight>(entity) {
                    light.0.light_type = light_type;
                }
                push_undo_snapshot(world, "Light Type");
            }

            let mut color = [light.color.x, light.color.y, light.color.z];
            let color_response = ui.horizontal(|ui| {
                ui.label("Color");
                ui.color_edit_button_rgb(&mut color)
            });
            let color_response = EditResponse::from_response(&color_response.inner);
            begin_edit_undo(world, "Light", color_response);
            if color_response.changed {
                if let Some(mut light) = world.get_mut::<BevyLight>(entity) {
                    light.0.color = Vec3::new(color[0], color[1], color[2]);
                }
            }
            end_edit_undo(world, color_response);

            let mut intensity = light.intensity;
            let intensity_response = edit_float(ui, "Intensity", &mut intensity, 0.1);
            begin_edit_undo(world, "Light", intensity_response);
            if intensity_response.changed {
                if let Some(mut light) = world.get_mut::<BevyLight>(entity) {
                    light.0.intensity = intensity;
                }
            }
            end_edit_undo(world, intensity_response);

            if let LightType::Spot { angle } = light_type {
                let mut angle_deg = angle.to_degrees();
                let angle_response = edit_float(ui, "Spot Angle", &mut angle_deg, 0.5);
                begin_edit_undo(world, "Light", angle_response);
                if angle_response.changed {
                    if let Some(mut light) = world.get_mut::<BevyLight>(entity) {
                        light.0.light_type = LightType::Spot {
                            angle: angle_deg.to_radians(),
                        };
                    }
                }
                end_edit_undo(world, angle_response);
            }
        }
        ui.separator();
    }

    if world.get::<BevyAudioListener>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Audio Listener");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyAudioListener>();
            push_undo_snapshot(world, "Remove Audio Listener");
        } else if let Some(listener) = world.get::<BevyAudioListener>(entity).map(|l| l.0) {
            let mut enabled = listener.enabled;
            let response = ui.checkbox(&mut enabled, "Enabled");
            let response = EditResponse::from_response(&response);
            begin_edit_undo(world, "Audio Listener", response);
            if response.changed {
                if let Some(mut listener) = world.get_mut::<BevyAudioListener>(entity) {
                    listener.0.enabled = enabled;
                }
            }
            end_edit_undo(world, response);
        }
        ui.separator();
    }

    if world.get::<BevyAudioEmitter>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Audio Emitter");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyAudioEmitter>();
            world.entity_mut(entity).remove::<EditorAudio>();
            push_undo_snapshot(world, "Remove Audio Emitter");
        } else if let Some(emitter_snapshot) = world.get::<BevyAudioEmitter>(entity).map(|e| e.0) {
            let mut emitter = emitter_snapshot;
            let mut editor_audio =
                world
                    .get::<EditorAudio>(entity)
                    .cloned()
                    .unwrap_or(EditorAudio {
                        path: None,
                        streaming: false,
                    });

            let mut clip_applied = false;

            let clip_label = editor_audio
                .path
                .as_deref()
                .map(|path| format!("Clip: {}", path))
                .unwrap_or_else(|| "Clip: <none>".to_string());

            let clip_button = ui.button(clip_label);
            if clip_button.clicked() {
                if let Some(path) = selected_asset.as_ref().filter(|p| is_audio_file(p)) {
                    apply_audio_emitter_from_asset(
                        world,
                        entity,
                        &project,
                        path,
                        editor_audio.streaming,
                    );
                    clip_applied = true;
                    push_undo_snapshot(world, "Audio Clip");
                }
            }
            if let Some(payload) = typed_dnd_release_payload::<AssetDragPayload>(&clip_button) {
                if let Some(path) = payload_primary_path(&payload) {
                    if is_audio_file(path) {
                        apply_audio_emitter_from_asset(
                            world,
                            entity,
                            &project,
                            path,
                            editor_audio.streaming,
                        );
                        clip_applied = true;
                        push_undo_snapshot(world, "Audio Clip");
                    }
                }
            }
            highlight_drop_target(ui, &clip_button);

            ui.horizontal(|ui| {
                if ui.button("Clear").clicked() {
                    editor_audio.path = None;
                    emitter.clip_id = None;
                    if let Some(mut audio) = world.get_mut::<EditorAudio>(entity) {
                        audio.path = None;
                    } else {
                        world.entity_mut(entity).insert(EditorAudio {
                            path: None,
                            streaming: editor_audio.streaming,
                        });
                    }
                    if let Some(mut emitter_comp) = world.get_mut::<BevyAudioEmitter>(entity) {
                        emitter_comp.0.clip_id = None;
                    }
                    push_undo_snapshot(world, "Clear Audio Clip");
                }
                let mut streaming = editor_audio.streaming;
                if ui.checkbox(&mut streaming, "Streaming").changed() {
                    editor_audio.streaming = streaming;
                    if let Some(path) = editor_audio.path.clone() {
                        let resolved = resolve_asset_path(project.as_ref(), &path);
                        apply_audio_emitter_from_asset(
                            world, entity, &project, &resolved, streaming,
                        );
                        clip_applied = true;
                    } else {
                        world.entity_mut(entity).insert(EditorAudio {
                            path: None,
                            streaming,
                        });
                    }
                    push_undo_snapshot(world, "Audio Streaming");
                }
            });

            let bus_options: Vec<(String, AudioBus)> = world
                .get_resource::<AudioBackendResource>()
                .map(|backend| {
                    let audio = backend.0.clone();
                    audio
                        .bus_list()
                        .into_iter()
                        .map(|bus| (audio.bus_name(bus), bus))
                        .collect()
                })
                .unwrap_or_else(|| {
                    vec![
                        ("Master".to_string(), AudioBus::Master),
                        ("Music".to_string(), AudioBus::Music),
                        ("SFX".to_string(), AudioBus::Sfx),
                        ("UI".to_string(), AudioBus::Ui),
                        ("Ambience".to_string(), AudioBus::Ambience),
                        ("World".to_string(), AudioBus::World),
                    ]
                });
            let bus_label = bus_options
                .iter()
                .find(|(_, bus)| *bus == emitter.bus)
                .map(|(label, _)| label.as_str())
                .unwrap_or("Custom");
            ComboBox::from_id_source(format!("audio_bus_{}", entity.to_bits()))
                .selected_text(bus_label)
                .show_ui(ui, |ui| {
                    for (label, bus) in &bus_options {
                        if ui.selectable_label(emitter.bus == *bus, label).clicked() {
                            emitter.bus = *bus;
                        }
                    }
                });

            let mut volume = emitter.volume;
            let volume_response = edit_float(ui, "Volume", &mut volume, 0.05);
            begin_edit_undo(world, "Audio", volume_response);
            if volume_response.changed {
                emitter.volume = volume;
            }
            end_edit_undo(world, volume_response);

            let mut pitch = emitter.pitch;
            let pitch_response = edit_float(ui, "Pitch", &mut pitch, 0.01);
            begin_edit_undo(world, "Audio", pitch_response);
            if pitch_response.changed {
                emitter.pitch = pitch;
            }
            end_edit_undo(world, pitch_response);

            let looping_response = ui.checkbox(&mut emitter.looping, "Looping");
            if looping_response.changed() {
                push_undo_snapshot(world, "Audio Looping");
            }

            let spatial_response = ui.checkbox(&mut emitter.spatial, "Spatialize");
            if spatial_response.changed() {
                push_undo_snapshot(world, "Audio Spatial");
            }

            let mut min_distance = emitter.min_distance;
            let min_response = edit_float(ui, "Min Distance", &mut min_distance, 0.1);
            begin_edit_undo(world, "Audio", min_response);
            if min_response.changed {
                emitter.min_distance = min_distance.max(0.01);
            }
            end_edit_undo(world, min_response);

            let mut max_distance = emitter.max_distance;
            let max_response = edit_float(ui, "Max Distance", &mut max_distance, 0.5);
            begin_edit_undo(world, "Audio", max_response);
            if max_response.changed {
                emitter.max_distance = max_distance.max(emitter.min_distance + 0.01);
            }
            end_edit_undo(world, max_response);

            let mut rolloff = emitter.rolloff;
            let rolloff_response = edit_float(ui, "Rolloff", &mut rolloff, 0.05);
            begin_edit_undo(world, "Audio", rolloff_response);
            if rolloff_response.changed {
                emitter.rolloff = rolloff.max(0.0);
            }
            end_edit_undo(world, rolloff_response);

            let mut spatial_blend = emitter.spatial_blend;
            let blend_response = edit_float(ui, "Spatial Blend", &mut spatial_blend, 0.05);
            begin_edit_undo(world, "Audio", blend_response);
            if blend_response.changed {
                emitter.spatial_blend = spatial_blend.clamp(0.0, 1.0);
            }
            end_edit_undo(world, blend_response);

            ui.horizontal(|ui| {
                if ui.button("Play").clicked() {
                    emitter.playback_state = AudioPlaybackState::Playing;
                    emitter.play_on_spawn = false;
                    push_undo_snapshot(world, "Audio Play");
                }
                if ui.button("Pause").clicked() {
                    emitter.playback_state = AudioPlaybackState::Paused;
                    push_undo_snapshot(world, "Audio Pause");
                }
                if ui.button("Stop").clicked() {
                    emitter.playback_state = AudioPlaybackState::Stopped;
                    push_undo_snapshot(world, "Audio Stop");
                }
                ui.checkbox(&mut emitter.play_on_spawn, "Play On Spawn");
            });

            if clip_applied {
                if let Some(current) = world.get::<BevyAudioEmitter>(entity) {
                    emitter.clip_id = current.0.clip_id;
                }
            }

            if let Some(mut emitter_comp) = world.get_mut::<BevyAudioEmitter>(entity) {
                emitter_comp.0 = emitter;
            } else {
                world.entity_mut(entity).insert(BevyWrapper(emitter));
            }
        }
        ui.separator();
    }

    if world.get::<BevyMeshRenderer>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Mesh Renderer");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyMeshRenderer>();
            world.entity_mut(entity).remove::<EditorMesh>();
            push_undo_snapshot(world, "Remove Mesh");
        } else {
            let mesh_state = world
                .get::<EditorMesh>(entity)
                .cloned()
                .unwrap_or(EditorMesh {
                    source: MeshSource::Primitive(PrimitiveKind::Cube),
                    material_path: None,
                });
            let mut mesh_source = mesh_state.source.clone();
            let mut material_path = mesh_state.material_path.clone();
            let mut casts_shadow = world
                .get::<BevyMeshRenderer>(entity)
                .map(|renderer| renderer.0.casts_shadow)
                .unwrap_or(true);
            let mut visible = world
                .get::<BevyMeshRenderer>(entity)
                .map(|renderer| renderer.0.visible)
                .unwrap_or(true);

            let mesh_label = match &mesh_source {
                MeshSource::Primitive(PrimitiveKind::Cube) => "Mesh: Cube".to_string(),
                MeshSource::Primitive(PrimitiveKind::UvSphere(_, _)) => {
                    "Mesh: UV Sphere".to_string()
                }
                MeshSource::Primitive(PrimitiveKind::Plane) => "Mesh: Plane".to_string(),
                MeshSource::Asset { path } => {
                    let relative = project_relative_path(&project, Path::new(path));
                    format!("Mesh: {}", relative)
                }
            };

            let material_label = match material_path.as_deref() {
                Some(path) => format!(
                    "Material: {}",
                    project_relative_path(&project, Path::new(path))
                ),
                None => "Material: <default>".to_string(),
            };

            let mut mesh_changed = false;
            let mut material_changed = false;
            let mut uv_edit_response = EditResponse::default();

            let mesh_source_button = ui.menu_button(mesh_label, |ui| {
                let (mut segments, mut rings) = match mesh_source {
                    MeshSource::Primitive(PrimitiveKind::UvSphere(segments, rings)) => {
                        (segments, rings)
                    }
                    _ => (12, 12),
                };

                if ui.button("Cube").clicked() {
                    mesh_source = MeshSource::Primitive(PrimitiveKind::Cube);
                    mesh_changed = true;
                    ui.close_menu();
                }
                if ui.button("UV Sphere").clicked() {
                    mesh_source = MeshSource::Primitive(PrimitiveKind::UvSphere(segments, rings));
                    mesh_changed = true;
                    ui.close_menu();
                }
                if ui.button("Plane").clicked() {
                    mesh_source = MeshSource::Primitive(PrimitiveKind::Plane);
                    mesh_changed = true;
                    ui.close_menu();
                }

                if let Some(path) = selected_asset.as_ref() {
                    if is_model_file(path) {
                        if ui.button("Use Selected Asset").clicked() {
                            mesh_source = mesh_source_from_path(&project, path);
                            mesh_changed = true;
                            ui.close_menu();
                        }
                    }
                }

                if ui.button("Browse...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Model", &["glb", "gltf"])
                        .pick_file()
                    {
                        mesh_source = mesh_source_from_path(&project, &path);
                        mesh_changed = true;
                    }
                    ui.close_menu();
                }
            });

            if let Some(payload) =
                typed_dnd_release_payload::<AssetDragPayload>(&mesh_source_button.response)
            {
                if let Some(path) = payload_primary_path(&payload) {
                    if is_model_file(path) {
                        mesh_source = mesh_source_from_path(&project, path);
                        mesh_changed = true;
                    }
                }
            }
            highlight_drop_target(ui, &mesh_source_button.response);

            if let MeshSource::Primitive(PrimitiveKind::UvSphere(segments, rings)) = mesh_source {
                let mut next_segments = segments;
                let mut next_rings = rings;
                ui.indent("uv_sphere_settings", |ui| {
                    ui.label("UV Sphere");
                    ui.horizontal(|ui| {
                        ui.label("Segments");
                        let response = ui.add(
                            egui::DragValue::new(&mut next_segments)
                                .range(3..=128)
                                .speed(1),
                        );
                        uv_edit_response.merge(EditResponse::from_response(&response));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Rings");
                        let response = ui.add(
                            egui::DragValue::new(&mut next_rings)
                                .range(3..=128)
                                .speed(1),
                        );
                        uv_edit_response.merge(EditResponse::from_response(&response));
                    });
                });
                begin_edit_undo(world, "Mesh", uv_edit_response);
                if next_segments != segments || next_rings != rings {
                    mesh_source =
                        MeshSource::Primitive(PrimitiveKind::UvSphere(next_segments, next_rings));
                    mesh_changed = true;
                }
            }

            let material_button = ui.menu_button(material_label, |ui| {
                if ui.button("Use Default").clicked() {
                    material_path = None;
                    material_changed = true;
                    ui.close_menu();
                }
                if let Some(path) = selected_asset.as_ref() {
                    if is_material_file(path) {
                        if ui.button("Use Selected Material").clicked() {
                            material_path = material_path_from_project(&project, path);
                            material_changed = true;
                            ui.close_menu();
                        }
                    }
                }

                if ui.button("Browse...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Material", &["ron"])
                        .pick_file()
                    {
                        material_path = material_path_from_project(&project, &path);
                        material_changed = true;
                    }
                    ui.close_menu();
                }
            });

            if let Some(path) = material_path
                .as_deref()
                .map(|path| resolve_asset_path(project.as_ref(), path))
            {
                if ui.button("Edit Material").clicked() {
                    open_material_editor_tab(world, path);
                }
            }

            if let Some(payload) =
                typed_dnd_release_payload::<AssetDragPayload>(&material_button.response)
            {
                if let Some(path) = payload_primary_path(&payload) {
                    if is_material_file(path) {
                        material_path = material_path_from_project(&project, path);
                        material_changed = true;
                    }
                }
            }
            highlight_drop_target(ui, &material_button.response);

            if ui.checkbox(&mut casts_shadow, "Casts Shadow").changed() {
                if let Some(mut renderer) = world.get_mut::<BevyMeshRenderer>(entity) {
                    renderer.0.casts_shadow = casts_shadow;
                }
                mark_entity_render_dirty(world, entity);
                push_undo_snapshot(world, "Mesh");
            }
            if ui.checkbox(&mut visible, "Visible").changed() {
                if let Some(mut renderer) = world.get_mut::<BevyMeshRenderer>(entity) {
                    renderer.0.visible = visible;
                }
                mark_entity_render_dirty(world, entity);
                push_undo_snapshot(world, "Mesh");
            }

            if mesh_changed || material_changed {
                apply_mesh_renderer(
                    world,
                    entity,
                    &project,
                    mesh_source,
                    material_path,
                    casts_shadow,
                    visible,
                );
            }

            let uv_editing = uv_edit_response.changed
                || uv_edit_response.drag_started
                || uv_edit_response.drag_released
                || uv_edit_response.lost_focus;
            if uv_editing {
                end_edit_undo(world, uv_edit_response);
            }
            if (mesh_changed || material_changed) && !uv_editing {
                let label = if mesh_changed { "Mesh" } else { "Material" };
                push_undo_snapshot(world, label);
            }
        }
        ui.separator();
    }

    let has_skinned_panel = world.get::<BevySkinnedMeshRenderer>(entity).is_some()
        || world.get::<PendingSkinnedMeshAsset>(entity).is_some()
        || world.get::<EditorSkinnedMesh>(entity).is_some();
    if has_skinned_panel {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Skinned Mesh Renderer");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevySkinnedMeshRenderer>();
            world.entity_mut(entity).remove::<BevyAnimator>();
            world.entity_mut(entity).remove::<BevyPoseOverride>();
            world.entity_mut(entity).remove::<PendingSkinnedMeshAsset>();
            world.entity_mut(entity).remove::<EditorSkinnedMesh>();
            push_undo_snapshot(world, "Remove Skinned Mesh");
        } else {
            if world.get::<EditorSkinnedMesh>(entity).is_none() {
                let (casts_shadow, visible) = world
                    .get::<BevySkinnedMeshRenderer>(entity)
                    .map(|renderer| (renderer.0.casts_shadow, renderer.0.visible))
                    .unwrap_or((true, true));
                world.entity_mut(entity).insert(EditorSkinnedMesh {
                    scene_path: None,
                    node_index: None,
                    casts_shadow,
                    visible,
                });
            }

            let pending = world.get::<PendingSkinnedMeshAsset>(entity).is_some();
            if pending {
                ui.label("Status: Pending");
            }

            if let Some(skinned) = world.get::<EditorSkinnedMesh>(entity) {
                let path_label = skinned
                    .scene_path
                    .as_deref()
                    .map(|path| format!("Source: {}", path))
                    .unwrap_or_else(|| "Source: <none>".to_string());
                ui.add(egui::Label::new(path_label).wrap_mode(egui::TextWrapMode::Extend));
            }

            let mut skinned_changed = false;
            let skinned_source_button = ui.menu_button("Skinned Source", |ui| {
                if let Some(path) = selected_asset.as_ref().filter(|path| is_model_file(path)) {
                    if ui.button("Use Selected Asset").clicked() {
                        if apply_skinned_mesh_renderer_from_asset(world, entity, &project, path) {
                            skinned_changed = true;
                        }
                        ui.close_menu();
                    }
                } else {
                    ui.label("Select a glb/gltf asset");
                }

                if ui.button("Browse...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Scene", &["glb", "gltf"])
                        .pick_file()
                    {
                        if apply_skinned_mesh_renderer_from_asset(world, entity, &project, &path) {
                            skinned_changed = true;
                        }
                    }
                    ui.close_menu();
                }

                if ui.button("Clear").clicked() {
                    if let Some(mut skinned) = world.get_mut::<EditorSkinnedMesh>(entity) {
                        skinned.scene_path = None;
                        skinned.node_index = None;
                    }
                    world.entity_mut(entity).remove::<BevySkinnedMeshRenderer>();
                    world.entity_mut(entity).remove::<BevyAnimator>();
                    world.entity_mut(entity).remove::<BevyPoseOverride>();
                    world.entity_mut(entity).remove::<PendingSkinnedMeshAsset>();
                    skinned_changed = true;
                    ui.close_menu();
                }
            });

            if let Some(payload) =
                typed_dnd_release_payload::<AssetDragPayload>(&skinned_source_button.response)
            {
                if let Some(path) = payload_primary_path(&payload) {
                    if is_model_file(path) {
                        if apply_skinned_mesh_renderer_from_asset(world, entity, &project, path) {
                            skinned_changed = true;
                        }
                    }
                }
            }
            highlight_drop_target(ui, &skinned_source_button.response);

            if let Some(skinned_renderer) = world.get::<BevySkinnedMeshRenderer>(entity).cloned() {
                ui.label(format!("Mesh ID: {}", skinned_renderer.0.mesh_id));
                ui.label(format!("Material ID: {}", skinned_renderer.0.material_id));
                ui.label(format!(
                    "Skin: {} ({} joints)",
                    skinned_renderer.0.skin.name,
                    skinned_renderer.0.skin.skeleton.joint_count()
                ));
            }

            if let Some(skinned) = world.get::<EditorSkinnedMesh>(entity).cloned() {
                let mut casts_shadow = world
                    .get::<BevySkinnedMeshRenderer>(entity)
                    .map(|renderer| renderer.0.casts_shadow)
                    .unwrap_or(skinned.casts_shadow);
                if ui.checkbox(&mut casts_shadow, "Casts Shadow").changed() {
                    if let Some(mut renderer) = world.get_mut::<BevySkinnedMeshRenderer>(entity) {
                        renderer.0.casts_shadow = casts_shadow;
                    }
                    if let Some(mut editor_skinned) = world.get_mut::<EditorSkinnedMesh>(entity) {
                        editor_skinned.casts_shadow = casts_shadow;
                    }
                    skinned_changed = true;
                }

                let mut visible = world
                    .get::<BevySkinnedMeshRenderer>(entity)
                    .map(|renderer| renderer.0.visible)
                    .unwrap_or(skinned.visible);
                if ui.checkbox(&mut visible, "Visible").changed() {
                    if let Some(mut renderer) = world.get_mut::<BevySkinnedMeshRenderer>(entity) {
                        renderer.0.visible = visible;
                    }
                    if let Some(mut editor_skinned) = world.get_mut::<EditorSkinnedMesh>(entity) {
                        editor_skinned.visible = visible;
                    }
                    skinned_changed = true;
                }
            }

            if skinned_changed {
                mark_entity_render_dirty(world, entity);
                push_undo_snapshot(world, "Skinned Mesh");
            }
        }
        ui.separator();
    }

    if world.get::<BevyAnimator>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Animator");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyAnimator>();
            push_undo_snapshot(world, "Remove Animator");
        } else {
            let mut animator_changed = false;
            ui.horizontal(|ui| {
                ui.label("Animation Asset");
                let apply_button = ui.button("Apply...");
                if apply_button.clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Animation", &["ron"])
                        .pick_file()
                    {
                        apply_animation_asset_to_entity(world, entity, &path);
                    }
                }
                if let Some(payload) = typed_dnd_release_payload::<AssetDragPayload>(&apply_button)
                {
                    if let Some(path) = payload_primary_path(&payload) {
                        if is_animation_file(path) {
                            apply_animation_asset_to_entity(world, entity, path);
                        } else {
                            set_status(world, "Drop a .hanim.ron animation asset".to_string());
                        }
                    }
                }
                highlight_drop_target(ui, &apply_button);
            });
            world.resource_scope::<AnimatorUiState, _>(|world, mut ui_state| {
                let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) else {
                    return;
                };

                let mut enabled = animator.0.enabled;
                if ui.checkbox(&mut enabled, "Enabled").changed() {
                    animator.0.enabled = enabled;
                    animator_changed = true;
                }

                let mut time_scale = animator.0.time_scale;
                let time_response = edit_float(ui, "Time Scale", &mut time_scale, 0.01);
                if time_response.changed {
                    animator.0.time_scale = time_scale.max(0.0);
                    animator_changed = true;
                }

                ui.collapsing("Layers", |ui| {
                    for (layer_index, layer) in animator.0.layers.iter_mut().enumerate() {
                        let header = format!("Layer {}: {}", layer_index, layer.name);
                        ui.collapsing(header, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Name");
                                let response = ui.text_edit_singleline(&mut layer.name);
                                if response.changed() {
                                    animator_changed = true;
                                }
                            });

                            let weight_response = edit_float(ui, "Weight", &mut layer.weight, 0.01);
                            if weight_response.changed {
                                animator_changed = true;
                            }

                            if ui.checkbox(&mut layer.additive, "Additive").changed() {
                                animator_changed = true;
                            }

                            let state_name = layer
                                .state_machine
                                .states
                                .get(layer.state_machine.current_state)
                                .map(|s| s.name.as_str())
                                .unwrap_or("<missing>");
                            ui.label(format!("State: {}", state_name));

                            if let Some(state) = layer
                                .state_machine
                                .states
                                .get(layer.state_machine.current_state)
                            {
                                if let Some(node) = layer.graph.nodes.get_mut(state.node) {
                                    if let helmer::animation::AnimationNode::Clip(clip_node) = node
                                    {
                                        let clip_names: Vec<String> = layer
                                            .graph
                                            .library
                                            .clips
                                            .iter()
                                            .map(|clip| clip.name.clone())
                                            .collect();
                                        if !clip_names.is_empty() {
                                            let current = clip_node
                                                .clip_index
                                                .min(clip_names.len().saturating_sub(1));
                                            let mut selected = current;
                                            ComboBox::from_id_source(format!(
                                                "anim_clip_{}_{}",
                                                entity.to_bits(),
                                                layer_index
                                            ))
                                            .selected_text(
                                                clip_names
                                                    .get(current)
                                                    .map(|s| s.as_str())
                                                    .unwrap_or("<clip>"),
                                            )
                                            .show_ui(
                                                ui,
                                                |ui| {
                                                    for (i, name) in clip_names.iter().enumerate() {
                                                        if ui
                                                            .selectable_label(i == current, name)
                                                            .clicked()
                                                        {
                                                            selected = i;
                                                        }
                                                    }
                                                },
                                            );
                                            if selected != current {
                                                clip_node.clip_index = selected;
                                                animator_changed = true;
                                            }
                                        }

                                        let speed_response =
                                            edit_float(ui, "Speed", &mut clip_node.speed, 0.01);
                                        if speed_response.changed {
                                            animator_changed = true;
                                        }

                                        if ui.checkbox(&mut clip_node.looping, "Looping").changed()
                                        {
                                            animator_changed = true;
                                        }

                                        let offset_response = edit_float(
                                            ui,
                                            "Time Offset",
                                            &mut clip_node.time_offset,
                                            0.01,
                                        );
                                        if offset_response.changed {
                                            animator_changed = true;
                                        }
                                    } else {
                                        ui.label("Non-clip node");
                                    }
                                }
                            }
                        });
                    }
                });

                ui.collapsing("Parameters", |ui| {
                    ui.label("Floats");
                    let mut remove_float = None;
                    let mut float_keys: Vec<String> =
                        animator.0.parameters.floats.keys().cloned().collect();
                    float_keys.sort();
                    for key in float_keys {
                        let mut value = animator
                            .0
                            .parameters
                            .floats
                            .get(&key)
                            .copied()
                            .unwrap_or(0.0);
                        ui.horizontal(|ui| {
                            ui.label(&key);
                            let response = ui.add(DragValue::new(&mut value).speed(0.05));
                            if response.changed() {
                                animator.0.parameters.set_float(key.clone(), value);
                                animator_changed = true;
                            }
                            if ui.button("X").clicked() {
                                remove_float = Some(key.clone());
                            }
                        });
                    }
                    if let Some(key) = remove_float {
                        animator.0.parameters.floats.remove(&key);
                        animator_changed = true;
                    }
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut ui_state.new_float_name);
                        ui.add(DragValue::new(&mut ui_state.new_float_value).speed(0.05));
                        if ui.button("Add Float").clicked() {
                            let name = ui_state.new_float_name.trim();
                            if !name.is_empty() {
                                animator
                                    .0
                                    .parameters
                                    .set_float(name.to_string(), ui_state.new_float_value);
                                ui_state.new_float_name.clear();
                                animator_changed = true;
                            }
                        }
                    });

                    ui.separator();
                    ui.label("Bools");
                    let mut remove_bool = None;
                    let mut bool_keys: Vec<String> =
                        animator.0.parameters.bools.keys().cloned().collect();
                    bool_keys.sort();
                    for key in bool_keys {
                        let mut value = animator
                            .0
                            .parameters
                            .bools
                            .get(&key)
                            .copied()
                            .unwrap_or(false);
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut value, &key);
                            if ui.button("X").clicked() {
                                remove_bool = Some(key.clone());
                            }
                        });
                        if animator
                            .0
                            .parameters
                            .bools
                            .get(&key)
                            .copied()
                            .unwrap_or(false)
                            != value
                        {
                            animator.0.parameters.set_bool(key.clone(), value);
                            animator_changed = true;
                        }
                    }
                    if let Some(key) = remove_bool {
                        animator.0.parameters.bools.remove(&key);
                        animator_changed = true;
                    }
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut ui_state.new_bool_name);
                        ui.checkbox(&mut ui_state.new_bool_value, "Value");
                        if ui.button("Add Bool").clicked() {
                            let name = ui_state.new_bool_name.trim();
                            if !name.is_empty() {
                                animator
                                    .0
                                    .parameters
                                    .set_bool(name.to_string(), ui_state.new_bool_value);
                                ui_state.new_bool_name.clear();
                                animator_changed = true;
                            }
                        }
                    });

                    ui.separator();
                    ui.label("Triggers");
                    let mut trigger_keys: Vec<String> =
                        animator.0.parameters.triggers.iter().cloned().collect();
                    trigger_keys.sort();
                    for key in trigger_keys {
                        ui.horizontal(|ui| {
                            ui.label(&key);
                            if ui.button("Fire").clicked() {
                                animator.0.parameters.trigger(key.clone());
                            }
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut ui_state.new_trigger_name);
                        if ui.button("Add Trigger").clicked() {
                            let name = ui_state.new_trigger_name.trim();
                            if !name.is_empty() {
                                animator.0.parameters.trigger(name.to_string());
                                ui_state.new_trigger_name.clear();
                            }
                        }
                    });
                });
            });
            if animator_changed {
                push_undo_snapshot(world, "Animator");
            }
        }
        ui.separator();
    }

    if let Some(skinned) = world.get::<BevySkinnedMeshRenderer>(entity) {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Pose Override");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyPoseOverride>();
            push_undo_snapshot(world, "Remove Pose Override");
        } else {
            let skeleton = skinned.0.skin.skeleton.clone();
            let mut enabled = world
                .get::<BevyPoseOverride>(entity)
                .map(|pose| pose.0.enabled)
                .unwrap_or(false);
            if ui.checkbox(&mut enabled, "Enable Pose Override").changed() {
                if enabled {
                    world
                        .entity_mut(entity)
                        .insert(BevyPoseOverride(PoseOverride::new(&skeleton)));
                } else if let Some(mut pose_override) = world.get_mut::<BevyPoseOverride>(entity) {
                    pose_override.0.enabled = false;
                }
                push_undo_snapshot(world, "Pose Override");
            }

            if enabled {
                world.resource_scope::<PoseEditorState, _>(|world, mut pose_state| {
                    let mut pose_changed = false;
                    let Some(mut pose_override) = world.get_mut::<BevyPoseOverride>(entity) else {
                        return;
                    };

                    if pose_override.0.pose.locals.len() != skeleton.joint_count() {
                        pose_override.0.pose.reset_to_bind(&skeleton);
                    }

                    let mut edit_mode =
                        pose_state.edit_mode && pose_state.active_entity == Some(entity.to_bits());
                    if ui.checkbox(&mut edit_mode, "Edit in Viewport").changed() {
                        pose_state.edit_mode = edit_mode;
                        pose_state.active_entity = if edit_mode {
                            Some(entity.to_bits())
                        } else {
                            pose_state.selected_joint = None;
                            pose_state.hover_joint = None;
                            None
                        };
                        if edit_mode && !pose_override.0.enabled {
                            pose_override.0.enabled = true;
                            pose_changed = true;
                        }
                    }

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut pose_state.use_gizmo, "Use Gizmo");
                        ui.checkbox(&mut pose_state.show_bones, "Show Bones");
                        ui.checkbox(&mut pose_state.show_joint_handles, "Show Joint Handles");
                    });

                    ui.horizontal(|ui| {
                        ui.label("Handle Size");
                        ui.add(DragValue::new(&mut pose_state.handle_size_scale).speed(0.005));
                        ui.label("Min");
                        ui.add(DragValue::new(&mut pose_state.handle_size_min).speed(0.01));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Pick Radius");
                        ui.add(DragValue::new(&mut pose_state.pick_radius_scale).speed(0.005));
                        ui.label("Min");
                        ui.add(DragValue::new(&mut pose_state.pick_radius_min).speed(0.01));
                    });

                    ui.horizontal(|ui| {
                        if ui.button("Reset Pose").clicked() {
                            pose_override.0.pose.reset_to_bind(&skeleton);
                            pose_changed = true;
                        }
                        if ui.button("Clear Selection").clicked() {
                            pose_state.selected_joint = None;
                            pose_state.hover_joint = None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Filter");
                        ui.text_edit_singleline(&mut pose_state.joint_filter);
                    });

                    ui.collapsing("Joints", |ui| {
                        let filter = pose_state.joint_filter.trim().to_lowercase();
                        if filter.is_empty() {
                            let children = build_joint_children(&skeleton);
                            for &root in skeleton.root_joints.iter() {
                                draw_joint_tree(
                                    ui,
                                    &skeleton,
                                    &children,
                                    root,
                                    0,
                                    &filter,
                                    &mut pose_state,
                                );
                            }
                        } else {
                            for (index, joint) in skeleton.joints.iter().enumerate() {
                                if joint.name.to_lowercase().contains(&filter) {
                                    ui.push_id(index, |ui| {
                                        if ui
                                            .selectable_label(
                                                pose_state.selected_joint == Some(index),
                                                &joint.name,
                                            )
                                            .clicked()
                                        {
                                            pose_state.selected_joint = Some(index);
                                        }
                                    });
                                }
                            }
                        }
                    });

                    if let Some(index) = pose_state.selected_joint {
                        if let Some(joint) = skeleton.joints.get(index) {
                            ui.separator();
                            ui.label(format!("Selected: {}", joint.name));
                        }
                        let mut transform = pose_override
                            .0
                            .pose
                            .locals
                            .get(index)
                            .copied()
                            .unwrap_or_default();
                        let response = edit_vec3(ui, "Position", &mut transform.position, 0.05);
                        if response.changed {
                            pose_override.0.pose.locals[index].position = transform.position;
                            pose_changed = true;
                        }
                        let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
                        let mut rotation =
                            Vec3::new(yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees());
                        let rotation_response = edit_vec3(ui, "Rotation", &mut rotation, 0.5);
                        if rotation_response.changed {
                            pose_override.0.pose.locals[index].rotation = Quat::from_euler(
                                EulerRot::YXZ,
                                rotation.x.to_radians(),
                                rotation.y.to_radians(),
                                rotation.z.to_radians(),
                            );
                            pose_changed = true;
                        }
                        let scale_response = edit_vec3(ui, "Scale", &mut transform.scale, 0.05);
                        if scale_response.changed {
                            pose_override.0.pose.locals[index].scale = transform.scale;
                            pose_changed = true;
                        }
                        if ui.button("Reset Joint").clicked() {
                            if let Some(joint) = skeleton.joints.get(index) {
                                pose_override.0.pose.locals[index] = joint.bind_transform;
                                pose_changed = true;
                            }
                        }
                    }
                    if pose_changed {
                        push_undo_snapshot(world, "Pose Override");
                    }
                });
            }
        }
        ui.separator();
    }

    if world.get::<BevySpline>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Spline");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevySpline>();
            push_undo_snapshot(world, "Remove Spline");
        } else {
            let mut spline_changed = false;
            {
                let Some(mut spline) = world.get_mut::<BevySpline>(entity) else {
                    return;
                };

                let mut mode = spline.0.mode;
                let mode_label = match mode {
                    SplineMode::Linear => "Linear",
                    SplineMode::CatmullRom => "Catmull-Rom",
                    SplineMode::Bezier => "Bezier",
                };
                ComboBox::from_id_source(format!("spline_mode_{}", entity.to_bits()))
                    .selected_text(mode_label)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(matches!(mode, SplineMode::Linear), "Linear")
                            .clicked()
                        {
                            mode = SplineMode::Linear;
                        }
                        if ui
                            .selectable_label(matches!(mode, SplineMode::CatmullRom), "Catmull-Rom")
                            .clicked()
                        {
                            mode = SplineMode::CatmullRom;
                        }
                        if ui
                            .selectable_label(matches!(mode, SplineMode::Bezier), "Bezier")
                            .clicked()
                        {
                            mode = SplineMode::Bezier;
                        }
                    });
                if mode != spline.0.mode {
                    spline.0.mode = mode;
                    spline_changed = true;
                }

                let mut closed = spline.0.closed;
                if ui.checkbox(&mut closed, "Closed").changed() {
                    spline.0.closed = closed;
                    spline_changed = true;
                }

                let tension_response = edit_float(ui, "Catmull Alpha", &mut spline.0.tension, 0.01);
                if tension_response.changed {
                    spline_changed = true;
                }

                ui.horizontal(|ui| {
                    if ui.button("Add Point").clicked() {
                        spline.0.points.push(Vec3::ZERO);
                        spline_changed = true;
                    }
                    if ui.button("Clear").clicked() {
                        spline.0.points.clear();
                        spline_changed = true;
                    }
                    if ui.button("Reverse").clicked() {
                        spline.0.points.reverse();
                        spline_changed = true;
                    }
                });

                ui.collapsing("Points", |ui| {
                    let mut remove_index = None;
                    for (index, point) in spline.0.points.iter_mut().enumerate() {
                        let label = format!("Point {}", index);
                        let response = edit_vec3(ui, &label, point, 0.05);
                        if response.changed {
                            spline_changed = true;
                        }
                        if ui.button("Remove").clicked() {
                            remove_index = Some(index);
                        }
                        ui.separator();
                    }
                    if let Some(index) = remove_index {
                        if index < spline.0.points.len() {
                            spline.0.points.remove(index);
                            spline_changed = true;
                        }
                    }
                });
            }
            if spline_changed {
                push_undo_snapshot(world, "Spline");
            }

            world.resource_scope::<EditorSplineState, _>(|world, mut spline_state| {
                ui.collapsing("Spline Tool", |ui| {
                    let mut enabled = spline_state.enabled;
                    if ui.checkbox(&mut enabled, "Edit in Viewport").changed() {
                        spline_state.enabled = enabled;
                    }

                    ui.checkbox(&mut spline_state.add_mode, "Add Points on Click");
                    ui.checkbox(&mut spline_state.insert_mode, "Insert After Selection");
                    ui.checkbox(&mut spline_state.use_gizmo, "Use Gizmo Handles");
                    if ui.button("Insert Midpoint").clicked() {
                        if let Some(active_index) = spline_state.active_point {
                            if let Some(mut spline) = world.get_mut::<BevySpline>(entity) {
                                let count = spline.0.points.len();
                                if count >= 2 {
                                    let active_index = active_index.min(count.saturating_sub(1));
                                    let next_index = if active_index + 1 < count {
                                        active_index + 1
                                    } else if spline.0.closed {
                                        0
                                    } else {
                                        active_index
                                    };
                                    if next_index != active_index {
                                        let a = spline.0.points[active_index];
                                        let b = spline.0.points[next_index];
                                        let insert_index =
                                            (active_index + 1).min(spline.0.points.len());
                                        spline.0.points.insert(insert_index, (a + b) * 0.5);
                                        spline_state.active_point = Some(insert_index);
                                        push_undo_snapshot(world, "Spline");
                                    }
                                }
                            }
                        }
                    }
                    if spline_state.use_gizmo {
                        if let Some(mut gizmo_state) = world.get_resource_mut::<EditorGizmoState>()
                        {
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Gizmo Mode");
                                ui.selectable_value(
                                    &mut gizmo_state.mode,
                                    GizmoMode::None,
                                    "Select",
                                );
                                ui.selectable_value(
                                    &mut gizmo_state.mode,
                                    GizmoMode::Translate,
                                    "Move",
                                );
                                ui.selectable_value(
                                    &mut gizmo_state.mode,
                                    GizmoMode::Rotate,
                                    "Rotate",
                                );
                                ui.selectable_value(
                                    &mut gizmo_state.mode,
                                    GizmoMode::Scale,
                                    "Scale/Resize",
                                );
                            });
                        }
                    }

                    ui.horizontal(|ui| {
                        ui.label("Pivot");
                        let mut pivot = spline_state.pivot_mode;
                        ComboBox::from_id_source("spline_pivot_mode")
                            .selected_text(match pivot {
                                crate::editor::SplinePivotMode::Point => "Point",
                                crate::editor::SplinePivotMode::SplineOrigin => "Spline Origin",
                            })
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(
                                        matches!(pivot, crate::editor::SplinePivotMode::Point),
                                        "Point",
                                    )
                                    .clicked()
                                {
                                    pivot = crate::editor::SplinePivotMode::Point;
                                }
                                if ui
                                    .selectable_label(
                                        matches!(
                                            pivot,
                                            crate::editor::SplinePivotMode::SplineOrigin
                                        ),
                                        "Spline Origin",
                                    )
                                    .clicked()
                                {
                                    pivot = crate::editor::SplinePivotMode::SplineOrigin;
                                }
                            });
                        spline_state.pivot_mode = pivot;
                    });

                    let plane_label = match spline_state.draw_plane {
                        crate::editor::SplineDrawPlane::WorldXZ => "World XZ",
                        crate::editor::SplineDrawPlane::View => "View Plane",
                    };
                    ComboBox::from_id_source("spline_draw_plane")
                        .selected_text(plane_label)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    matches!(
                                        spline_state.draw_plane,
                                        crate::editor::SplineDrawPlane::WorldXZ
                                    ),
                                    "World XZ",
                                )
                                .clicked()
                            {
                                spline_state.draw_plane = crate::editor::SplineDrawPlane::WorldXZ;
                            }
                            if ui
                                .selectable_label(
                                    matches!(
                                        spline_state.draw_plane,
                                        crate::editor::SplineDrawPlane::View
                                    ),
                                    "View Plane",
                                )
                                .clicked()
                            {
                                spline_state.draw_plane = crate::editor::SplineDrawPlane::View;
                            }
                        });

                    ui.horizontal(|ui| {
                        ui.label("Samples");
                        ui.add(
                            DragValue::new(&mut spline_state.samples)
                                .range(8..=512)
                                .speed(1),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Handle Scale");
                        ui.add(DragValue::new(&mut spline_state.handle_size_scale).speed(0.01));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Handle Min");
                        ui.add(DragValue::new(&mut spline_state.handle_size_min).speed(0.01));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Pick Scale");
                        ui.add(DragValue::new(&mut spline_state.pick_radius_scale).speed(0.01));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Pick Min");
                        ui.add(DragValue::new(&mut spline_state.pick_radius_min).speed(0.01));
                    });
                });
            });
        }
        ui.separator();
    }

    if world.get::<BevySplineFollower>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Spline Follower");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevySplineFollower>();
            push_undo_snapshot(world, "Remove Spline Follower");
        } else {
            let pinned_entity = world
                .get_resource::<InspectorPinnedEntityResource>()
                .and_then(|res| res.0);
            let pinned_has_spline = pinned_entity
                .map(|pinned| world.get::<BevySpline>(pinned).is_some())
                .unwrap_or(false);
            let target_name = world
                .get::<BevySplineFollower>(entity)
                .and_then(|follower| follower.0.spline_entity)
                .and_then(|id| world.get::<Name>(Entity::from_bits(id)))
                .map(|name| name.to_string())
                .unwrap_or_else(|| "<self>".to_string());

            let mut dropped_entity: Option<Entity> = None;
            let mut drop_hint: Option<bool> = None;
            let drop_response = ui.add_sized(
                Vec2::new(ui.available_width(), 0.0),
                egui::Button::new("Drop Spline Here").sense(Sense::hover()),
            );
            if let Some(payload) = drop_response.dnd_hover_payload::<EntityDragPayload>() {
                drop_hint = Some(world.get::<BevySpline>(payload.entity).is_some());
            }
            if let Some(payload) = typed_dnd_release_payload::<EntityDragPayload>(&drop_response) {
                dropped_entity = Some(payload.entity);
            }
            if let Some(valid) = drop_hint {
                let color = if valid {
                    ui.visuals().selection.bg_fill
                } else {
                    ui.visuals().widgets.inactive.bg_fill
                };
                ui.painter().rect_filled(drop_response.rect, 4.0, color);
                ui.painter().text(
                    drop_response.rect.center(),
                    Align2::CENTER_CENTER,
                    if valid {
                        "Drop Spline Here"
                    } else {
                        "Entity has no Spline"
                    },
                    FontId::proportional(12.0),
                    ui.visuals().text_color(),
                );
            }
            let dropped_valid = dropped_entity
                .map(|entity| world.get::<BevySpline>(entity).is_some())
                .unwrap_or(false);

            let mut follower_changed = false;
            {
                let Some(mut follower) = world.get_mut::<BevySplineFollower>(entity) else {
                    return;
                };

                let speed_response = edit_float(ui, "Speed", &mut follower.0.speed, 0.01);
                if speed_response.changed {
                    follower_changed = true;
                }

                let t_response = edit_float(ui, "T", &mut follower.0.t, 0.01);
                if t_response.changed {
                    follower_changed = true;
                }

                if ui.checkbox(&mut follower.0.looped, "Looped").changed() {
                    follower_changed = true;
                }

                if ui
                    .checkbox(&mut follower.0.follow_rotation, "Follow Rotation")
                    .changed()
                {
                    follower_changed = true;
                }

                let up_response = edit_vec3(ui, "Up", &mut follower.0.up, 0.01);
                if up_response.changed {
                    follower_changed = true;
                }

                let offset_response = edit_vec3(ui, "Offset", &mut follower.0.offset, 0.01);
                if offset_response.changed {
                    follower_changed = true;
                }

                ui.horizontal(|ui| {
                    ui.label("Length Samples");
                    ui.add(
                        DragValue::new(&mut follower.0.length_samples)
                            .range(4..=512)
                            .speed(1),
                    );
                });

                ui.label(format!("Target: {}", target_name));

                if ui.button("Clear Target").clicked() {
                    follower.0.spline_entity = None;
                    follower_changed = true;
                }

                if let Some(pinned) = pinned_entity {
                    if ui
                        .add_enabled(pinned_has_spline, egui::Button::new("Use Pinned Spline"))
                        .clicked()
                    {
                        follower.0.spline_entity = Some(pinned.to_bits());
                        follower_changed = true;
                    }
                }

                if dropped_valid {
                    follower.0.spline_entity = dropped_entity.map(|entity| entity.to_bits());
                    follower_changed = true;
                }
            }
            if follower_changed {
                push_undo_snapshot(world, "Spline Follower");
            }
        }
        ui.separator();
    }

    if world.get::<BevyLookAt>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Look At");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyLookAt>();
            push_undo_snapshot(world, "Remove Look At");
        } else {
            let pinned_entity = world
                .get_resource::<InspectorPinnedEntityResource>()
                .and_then(|res| res.0);
            let pinned_has_transform = pinned_entity
                .map(|pinned| world.get::<BevyTransform>(pinned).is_some())
                .unwrap_or(false);
            let target_name = world
                .get::<BevyLookAt>(entity)
                .and_then(|look_at| look_at.0.target_entity)
                .and_then(|id| world.get::<Name>(Entity::from_bits(id)))
                .map(|name| name.to_string())
                .unwrap_or_else(|| "<none>".to_string());

            let mut dropped_entity: Option<Entity> = None;
            let mut drop_hint: Option<bool> = None;
            let drop_response = ui.add_sized(
                Vec2::new(ui.available_width(), 0.0),
                egui::Button::new("Drop Target Here").sense(Sense::hover()),
            );
            highlight_drop_target(ui, &drop_response);
            if let Some(payload) = drop_response.dnd_hover_payload::<EntityDragPayload>() {
                drop_hint = Some(world.get::<BevyTransform>(payload.entity).is_some());
            }
            if let Some(payload) = typed_dnd_release_payload::<EntityDragPayload>(&drop_response) {
                dropped_entity = Some(payload.entity);
            }
            if let Some(valid) = drop_hint {
                let color = if valid {
                    ui.visuals().selection.bg_fill
                } else {
                    ui.visuals().widgets.inactive.bg_fill
                };
                ui.painter().rect_filled(drop_response.rect, 4.0, color);
                ui.painter().text(
                    drop_response.rect.center(),
                    Align2::CENTER_CENTER,
                    if valid {
                        "Drop Target Here"
                    } else {
                        "Entity has no Transform"
                    },
                    FontId::proportional(12.0),
                    ui.visuals().text_color(),
                );
            }
            let dropped_valid = dropped_entity
                .map(|entity| world.get::<BevyTransform>(entity).is_some())
                .unwrap_or(false);

            let mut look_changed = false;
            {
                let Some(mut look_at) = world.get_mut::<BevyLookAt>(entity) else {
                    return;
                };

                let offset_response =
                    edit_vec3(ui, "Target Offset", &mut look_at.0.target_offset, 0.05);
                if offset_response.changed {
                    look_changed = true;
                }

                if ui
                    .checkbox(
                        &mut look_at.0.offset_in_target_space,
                        "Offset in Target Space",
                    )
                    .changed()
                {
                    look_changed = true;
                }

                let up_response = edit_vec3(ui, "Up", &mut look_at.0.up, 0.05);
                if up_response.changed {
                    look_changed = true;
                }

                let smooth_response = edit_float(
                    ui,
                    "Rotation Smooth Time",
                    &mut look_at.0.rotation_smooth_time,
                    0.01,
                );
                if smooth_response.changed {
                    if look_at.0.rotation_smooth_time < 0.0 {
                        look_at.0.rotation_smooth_time = 0.0;
                    }
                    look_changed = true;
                }

                ui.label(format!("Target: {}", target_name));

                if ui.button("Clear Target").clicked() {
                    look_at.0.target_entity = None;
                    look_changed = true;
                }

                if let Some(pinned) = pinned_entity {
                    if ui
                        .add_enabled(pinned_has_transform, egui::Button::new("Use Pinned Entity"))
                        .clicked()
                    {
                        look_at.0.target_entity = Some(pinned.to_bits());
                        look_changed = true;
                    }
                }

                if dropped_valid {
                    look_at.0.target_entity = dropped_entity.map(|entity| entity.to_bits());
                    look_changed = true;
                }
            }

            if look_changed {
                push_undo_snapshot(world, "Look At");
            }
        }
        ui.separator();
    }

    if world.get::<BevyEntityFollower>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Entity Follower");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<BevyEntityFollower>();
            push_undo_snapshot(world, "Remove Entity Follower");
        } else {
            let pinned_entity = world
                .get_resource::<InspectorPinnedEntityResource>()
                .and_then(|res| res.0);
            let pinned_has_transform = pinned_entity
                .map(|pinned| world.get::<BevyTransform>(pinned).is_some())
                .unwrap_or(false);
            let target_name = world
                .get::<BevyEntityFollower>(entity)
                .and_then(|follower| follower.0.target_entity)
                .and_then(|id| world.get::<Name>(Entity::from_bits(id)))
                .map(|name| name.to_string())
                .unwrap_or_else(|| "<none>".to_string());

            let mut dropped_entity: Option<Entity> = None;
            let mut drop_hint: Option<bool> = None;
            let drop_response = ui.add_sized(
                Vec2::new(ui.available_width(), 0.0),
                egui::Button::new("Drop Target Here").sense(Sense::hover()),
            );
            highlight_drop_target(ui, &drop_response);
            if let Some(payload) = drop_response.dnd_hover_payload::<EntityDragPayload>() {
                drop_hint = Some(world.get::<BevyTransform>(payload.entity).is_some());
            }
            if let Some(payload) = typed_dnd_release_payload::<EntityDragPayload>(&drop_response) {
                dropped_entity = Some(payload.entity);
            }
            if let Some(valid) = drop_hint {
                let color = if valid {
                    ui.visuals().selection.bg_fill
                } else {
                    ui.visuals().widgets.inactive.bg_fill
                };
                ui.painter().rect_filled(drop_response.rect, 4.0, color);
                ui.painter().text(
                    drop_response.rect.center(),
                    Align2::CENTER_CENTER,
                    if valid {
                        "Drop Target Here"
                    } else {
                        "Entity has no Transform"
                    },
                    FontId::proportional(12.0),
                    ui.visuals().text_color(),
                );
            }
            let dropped_valid = dropped_entity
                .map(|entity| world.get::<BevyTransform>(entity).is_some())
                .unwrap_or(false);

            let mut follower_changed = false;
            {
                let Some(mut follower) = world.get_mut::<BevyEntityFollower>(entity) else {
                    return;
                };

                let offset_response =
                    edit_vec3(ui, "Position Offset", &mut follower.0.position_offset, 0.05);
                if offset_response.changed {
                    follower_changed = true;
                }

                if ui
                    .checkbox(
                        &mut follower.0.offset_in_target_space,
                        "Offset in Target Space",
                    )
                    .changed()
                {
                    follower_changed = true;
                }

                if ui
                    .checkbox(&mut follower.0.follow_rotation, "Follow Rotation")
                    .changed()
                {
                    follower_changed = true;
                }

                let pos_smooth_response = edit_float(
                    ui,
                    "Position Smooth Time",
                    &mut follower.0.position_smooth_time,
                    0.01,
                );
                if pos_smooth_response.changed {
                    if follower.0.position_smooth_time < 0.0 {
                        follower.0.position_smooth_time = 0.0;
                    }
                    follower_changed = true;
                }

                ui.horizontal(|ui| {
                    ui.label("Rotation Smooth Time");
                    let rot_smooth_response = ui.add_enabled(
                        follower.0.follow_rotation,
                        egui::DragValue::new(&mut follower.0.rotation_smooth_time).speed(0.01),
                    );
                    let rot_smooth_response = EditResponse::from_response(&rot_smooth_response);
                    if rot_smooth_response.changed {
                        if follower.0.rotation_smooth_time < 0.0 {
                            follower.0.rotation_smooth_time = 0.0;
                        }
                        follower_changed = true;
                    }
                });

                ui.label(format!("Target: {}", target_name));

                if ui.button("Clear Target").clicked() {
                    follower.0.target_entity = None;
                    follower_changed = true;
                }

                if let Some(pinned) = pinned_entity {
                    if ui
                        .add_enabled(pinned_has_transform, egui::Button::new("Use Pinned Entity"))
                        .clicked()
                    {
                        follower.0.target_entity = Some(pinned.to_bits());
                        follower_changed = true;
                    }
                }

                if dropped_valid {
                    follower.0.target_entity = dropped_entity.map(|entity| entity.to_bits());
                    follower_changed = true;
                }
            }

            if follower_changed {
                push_undo_snapshot(world, "Entity Follower");
            }
        }
        ui.separator();
    }

    if world.get::<DynamicRigidBody>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Dynamic Rigid Body");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<DynamicRigidBody>();
            remove_body_dependent_physics_components(world, entity);
            push_undo_snapshot(world, "Remove Dynamic Body");
        } else if let Some(body) = world.get::<DynamicRigidBody>(entity).copied() {
            let mut mass = body.mass;
            let mass_response = edit_float(ui, "Mass", &mut mass, 0.1);
            begin_edit_undo(world, "Physics", mass_response);
            if mass_response.changed {
                if let Some(mut body) = world.get_mut::<DynamicRigidBody>(entity) {
                    body.mass = mass.max(0.0);
                }
            }
            end_edit_undo(world, mass_response);
        }
        ui.separator();
    }

    if world.get::<KinematicRigidBody>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Kinematic Rigid Body");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<KinematicRigidBody>();
            remove_body_dependent_physics_components(world, entity);
            push_undo_snapshot(world, "Remove Kinematic Body");
        } else if let Some(mode) = world.get::<KinematicRigidBody>(entity).copied() {
            let mut selected_mode = mode.mode;
            let mut changed = false;
            ComboBox::from_id_source(format!("kinematic_mode_{}", entity.to_bits()))
                .selected_text(match selected_mode {
                    KinematicMode::PositionBased => "Position Based",
                    KinematicMode::VelocityBased => "Velocity Based",
                })
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(
                            matches!(selected_mode, KinematicMode::PositionBased),
                            "Position Based",
                        )
                        .clicked()
                    {
                        selected_mode = KinematicMode::PositionBased;
                        changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(selected_mode, KinematicMode::VelocityBased),
                            "Velocity Based",
                        )
                        .clicked()
                    {
                        selected_mode = KinematicMode::VelocityBased;
                        changed = true;
                    }
                });
            if changed {
                if let Some(mut body) = world.get_mut::<KinematicRigidBody>(entity) {
                    body.mode = selected_mode;
                }
                push_undo_snapshot(world, "Kinematic Mode");
            }
        }
        ui.separator();
    }

    if world.get::<FixedCollider>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Fixed Collider");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<FixedCollider>();
            remove_body_dependent_physics_components(world, entity);
            push_undo_snapshot(world, "Remove Fixed Collider");
        }
        ui.separator();
    }

    if world.get::<PhysicsWorldDefaults>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Physics World Defaults");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });
        if remove {
            world.entity_mut(entity).remove::<PhysicsWorldDefaults>();
            push_undo_snapshot(world, "Remove Physics World Defaults");
        } else if let Some(mut defaults) = world.get_mut::<PhysicsWorldDefaults>(entity) {
            let mut changed = false;
            changed |= edit_vec3(ui, "Gravity", &mut defaults.gravity, 0.1).changed;

            ui.collapsing("Rigid Body Defaults", |ui| {
                changed |= edit_float(
                    ui,
                    "Linear Damping",
                    &mut defaults.rigid_body_properties.linear_damping,
                    0.01,
                )
                .changed;
                changed |= edit_float(
                    ui,
                    "Angular Damping",
                    &mut defaults.rigid_body_properties.angular_damping,
                    0.01,
                )
                .changed;
                changed |= edit_float(
                    ui,
                    "Gravity Scale",
                    &mut defaults.rigid_body_properties.gravity_scale,
                    0.01,
                )
                .changed;
                changed |= ui
                    .checkbox(
                        &mut defaults.rigid_body_properties.ccd_enabled,
                        "CCD Enabled",
                    )
                    .changed();
                changed |= ui
                    .checkbox(&mut defaults.rigid_body_properties.can_sleep, "Can Sleep")
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut defaults.rigid_body_properties.sleeping,
                        "Start Sleeping",
                    )
                    .changed();
                ui.horizontal(|ui| {
                    ui.label("Dominance Group");
                    changed |= ui
                        .add(
                            DragValue::new(&mut defaults.rigid_body_properties.dominance_group)
                                .speed(1.0),
                        )
                        .changed();
                });
                changed |= ui
                    .checkbox(
                        &mut defaults.rigid_body_properties.lock_translation_x,
                        "Lock Translation X",
                    )
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut defaults.rigid_body_properties.lock_translation_y,
                        "Lock Translation Y",
                    )
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut defaults.rigid_body_properties.lock_translation_z,
                        "Lock Translation Z",
                    )
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut defaults.rigid_body_properties.lock_rotation_x,
                        "Lock Rotation X",
                    )
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut defaults.rigid_body_properties.lock_rotation_y,
                        "Lock Rotation Y",
                    )
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut defaults.rigid_body_properties.lock_rotation_z,
                        "Lock Rotation Z",
                    )
                    .changed();
                changed |= edit_vec3(
                    ui,
                    "Linear Velocity",
                    &mut defaults.rigid_body_properties.linear_velocity,
                    0.05,
                )
                .changed;
                changed |= edit_vec3(
                    ui,
                    "Angular Velocity",
                    &mut defaults.rigid_body_properties.angular_velocity,
                    0.05,
                )
                .changed;
            });

            ui.collapsing("Collider Defaults", |ui| {
                changed |= edit_float(
                    ui,
                    "Friction",
                    &mut defaults.collider_properties.friction,
                    0.01,
                )
                .changed;
                changed |= edit_float(
                    ui,
                    "Restitution",
                    &mut defaults.collider_properties.restitution,
                    0.01,
                )
                .changed;
                changed |= edit_float(
                    ui,
                    "Density",
                    &mut defaults.collider_properties.density,
                    0.01,
                )
                .changed;
                changed |= ui
                    .checkbox(&mut defaults.collider_properties.is_sensor, "Sensor")
                    .changed();
                changed |= ui
                    .checkbox(&mut defaults.collider_properties.enabled, "Enabled")
                    .changed();
                changed |= edit_vec3(
                    ui,
                    "Offset Position",
                    &mut defaults.collider_properties.translation_offset,
                    0.01,
                )
                .changed;
                let mut rot = defaults
                    .collider_properties
                    .rotation_offset
                    .to_euler(EulerRot::XYZ);
                let mut rot_deg =
                    Vec3::new(rot.0.to_degrees(), rot.1.to_degrees(), rot.2.to_degrees());
                let rot_response = edit_vec3(ui, "Offset Rotation", &mut rot_deg, 0.5);
                if rot_response.changed {
                    defaults.collider_properties.rotation_offset = Quat::from_euler(
                        EulerRot::XYZ,
                        rot_deg.x.to_radians(),
                        rot_deg.y.to_radians(),
                        rot_deg.z.to_radians(),
                    );
                    changed = true;
                }
                changed |= edit_u32_range(
                    ui,
                    "Collision Memberships",
                    &mut defaults.collider_properties.collision_memberships,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
                changed |= edit_u32_range(
                    ui,
                    "Collision Filter",
                    &mut defaults.collider_properties.collision_filter,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
                changed |= edit_u32_range(
                    ui,
                    "Solver Memberships",
                    &mut defaults.collider_properties.solver_memberships,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
                changed |= edit_u32_range(
                    ui,
                    "Solver Filter",
                    &mut defaults.collider_properties.solver_filter,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;

                let mut friction_rule = defaults.collider_properties.friction_combine_rule;
                let mut restitution_rule = defaults.collider_properties.restitution_combine_rule;
                ComboBox::from_id_source(format!(
                    "world_defaults_friction_rule_{}",
                    entity.to_bits()
                ))
                .selected_text(match friction_rule {
                    PhysicsCombineRule::Average => "Average",
                    PhysicsCombineRule::Min => "Min",
                    PhysicsCombineRule::Multiply => "Multiply",
                    PhysicsCombineRule::Max => "Max",
                })
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(
                            matches!(friction_rule, PhysicsCombineRule::Average),
                            "Average",
                        )
                        .clicked()
                    {
                        friction_rule = PhysicsCombineRule::Average;
                        changed = true;
                    }
                    if ui
                        .selectable_label(matches!(friction_rule, PhysicsCombineRule::Min), "Min")
                        .clicked()
                    {
                        friction_rule = PhysicsCombineRule::Min;
                        changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(friction_rule, PhysicsCombineRule::Multiply),
                            "Multiply",
                        )
                        .clicked()
                    {
                        friction_rule = PhysicsCombineRule::Multiply;
                        changed = true;
                    }
                    if ui
                        .selectable_label(matches!(friction_rule, PhysicsCombineRule::Max), "Max")
                        .clicked()
                    {
                        friction_rule = PhysicsCombineRule::Max;
                        changed = true;
                    }
                });
                ComboBox::from_id_source(format!(
                    "world_defaults_restitution_rule_{}",
                    entity.to_bits()
                ))
                .selected_text(match restitution_rule {
                    PhysicsCombineRule::Average => "Average",
                    PhysicsCombineRule::Min => "Min",
                    PhysicsCombineRule::Multiply => "Multiply",
                    PhysicsCombineRule::Max => "Max",
                })
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(
                            matches!(restitution_rule, PhysicsCombineRule::Average),
                            "Average",
                        )
                        .clicked()
                    {
                        restitution_rule = PhysicsCombineRule::Average;
                        changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(restitution_rule, PhysicsCombineRule::Min),
                            "Min",
                        )
                        .clicked()
                    {
                        restitution_rule = PhysicsCombineRule::Min;
                        changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(restitution_rule, PhysicsCombineRule::Multiply),
                            "Multiply",
                        )
                        .clicked()
                    {
                        restitution_rule = PhysicsCombineRule::Multiply;
                        changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(restitution_rule, PhysicsCombineRule::Max),
                            "Max",
                        )
                        .clicked()
                    {
                        restitution_rule = PhysicsCombineRule::Max;
                        changed = true;
                    }
                });
                defaults.collider_properties.friction_combine_rule = friction_rule;
                defaults.collider_properties.restitution_combine_rule = restitution_rule;
            });

            if changed {
                push_undo_snapshot(world, "Physics World Defaults");
            }
        }
        ui.separator();
    }

    if world.get::<RigidBodyProperties>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Rigid Body Properties");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });
        if remove {
            world.entity_mut(entity).remove::<RigidBodyProperties>();
            world
                .entity_mut(entity)
                .remove::<RigidBodyPropertyInheritance>();
            push_undo_snapshot(world, "Remove Rigid Body Properties");
        } else if let Some(current_properties) = world.get::<RigidBodyProperties>(entity).copied() {
            let mut properties = current_properties;
            let mut inheritance = world
                .get::<RigidBodyPropertyInheritance>(entity)
                .copied()
                .unwrap_or_else(RigidBodyPropertyInheritance::none);
            let world_defaults = default_rigid_body_properties_for_world(world);
            let mut changed = false;
            changed |= edit_inherited_float(
                ui,
                "Linear Damping",
                &mut properties.linear_damping,
                &mut inheritance.linear_damping,
                world_defaults.linear_damping,
                0.01,
            );
            changed |= edit_inherited_float(
                ui,
                "Angular Damping",
                &mut properties.angular_damping,
                &mut inheritance.angular_damping,
                world_defaults.angular_damping,
                0.01,
            );
            changed |= edit_inherited_float(
                ui,
                "Gravity Scale",
                &mut properties.gravity_scale,
                &mut inheritance.gravity_scale,
                world_defaults.gravity_scale,
                0.01,
            );
            changed |= edit_inherited_bool(
                ui,
                "CCD Enabled",
                &mut properties.ccd_enabled,
                &mut inheritance.ccd_enabled,
                world_defaults.ccd_enabled,
            );
            changed |= edit_inherited_bool(
                ui,
                "Can Sleep",
                &mut properties.can_sleep,
                &mut inheritance.can_sleep,
                world_defaults.can_sleep,
            );
            changed |= edit_inherited_bool(
                ui,
                "Start Sleeping",
                &mut properties.sleeping,
                &mut inheritance.sleeping,
                world_defaults.sleeping,
            );
            changed |= edit_inherited_i8(
                ui,
                "Dominance Group",
                &mut properties.dominance_group,
                &mut inheritance.dominance_group,
                world_defaults.dominance_group,
                1.0,
            );
            changed |= edit_inherited_bool(
                ui,
                "Lock Translation X",
                &mut properties.lock_translation_x,
                &mut inheritance.lock_translation_x,
                world_defaults.lock_translation_x,
            );
            changed |= edit_inherited_bool(
                ui,
                "Lock Translation Y",
                &mut properties.lock_translation_y,
                &mut inheritance.lock_translation_y,
                world_defaults.lock_translation_y,
            );
            changed |= edit_inherited_bool(
                ui,
                "Lock Translation Z",
                &mut properties.lock_translation_z,
                &mut inheritance.lock_translation_z,
                world_defaults.lock_translation_z,
            );
            changed |= edit_inherited_bool(
                ui,
                "Lock Rotation X",
                &mut properties.lock_rotation_x,
                &mut inheritance.lock_rotation_x,
                world_defaults.lock_rotation_x,
            );
            changed |= edit_inherited_bool(
                ui,
                "Lock Rotation Y",
                &mut properties.lock_rotation_y,
                &mut inheritance.lock_rotation_y,
                world_defaults.lock_rotation_y,
            );
            changed |= edit_inherited_bool(
                ui,
                "Lock Rotation Z",
                &mut properties.lock_rotation_z,
                &mut inheritance.lock_rotation_z,
                world_defaults.lock_rotation_z,
            );
            changed |= edit_inherited_vec3(
                ui,
                "Linear Velocity",
                &mut properties.linear_velocity,
                &mut inheritance.linear_velocity,
                world_defaults.linear_velocity,
                0.05,
            );
            changed |= edit_inherited_vec3(
                ui,
                "Angular Velocity",
                &mut properties.angular_velocity,
                &mut inheritance.angular_velocity,
                world_defaults.angular_velocity,
                0.05,
            );
            if changed {
                if let Some(mut props) = world.get_mut::<RigidBodyProperties>(entity) {
                    *props = properties;
                }
                if let Some(mut inheritance_component) =
                    world.get_mut::<RigidBodyPropertyInheritance>(entity)
                {
                    *inheritance_component = inheritance;
                } else {
                    world.entity_mut(entity).insert(inheritance);
                }
                push_undo_snapshot(world, "Rigid Body Properties");
            }
        }
        ui.separator();
    }

    if world.get::<ColliderProperties>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Collider Properties");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });
        if remove {
            world.entity_mut(entity).remove::<ColliderProperties>();
            world
                .entity_mut(entity)
                .remove::<ColliderPropertyInheritance>();
            push_undo_snapshot(world, "Remove Collider Properties");
        } else if let Some(current_properties) = world.get::<ColliderProperties>(entity).copied() {
            let mut properties = current_properties;
            let mut inheritance = world
                .get::<ColliderPropertyInheritance>(entity)
                .copied()
                .unwrap_or_else(ColliderPropertyInheritance::none);
            let world_defaults = default_collider_properties_for_world(world);
            let mut changed = false;
            changed |= edit_inherited_float(
                ui,
                "Friction",
                &mut properties.friction,
                &mut inheritance.friction,
                world_defaults.friction,
                0.01,
            );
            changed |= edit_inherited_float(
                ui,
                "Restitution",
                &mut properties.restitution,
                &mut inheritance.restitution,
                world_defaults.restitution,
                0.01,
            );
            changed |= edit_inherited_float(
                ui,
                "Density",
                &mut properties.density,
                &mut inheritance.density,
                world_defaults.density,
                0.01,
            );
            changed |= edit_inherited_bool(
                ui,
                "Sensor",
                &mut properties.is_sensor,
                &mut inheritance.is_sensor,
                world_defaults.is_sensor,
            );
            changed |= edit_inherited_bool(
                ui,
                "Enabled",
                &mut properties.enabled,
                &mut inheritance.enabled,
                world_defaults.enabled,
            );
            changed |= edit_inherited_vec3(
                ui,
                "Offset Position",
                &mut properties.translation_offset,
                &mut inheritance.translation_offset,
                world_defaults.translation_offset,
                0.01,
            );
            changed |= edit_inherited_quat_euler(
                ui,
                "Offset Rotation",
                &mut properties.rotation_offset,
                &mut inheritance.rotation_offset,
                world_defaults.rotation_offset,
                0.5,
            );
            changed |= edit_inherited_u32(
                ui,
                "Collision Memberships",
                &mut properties.collision_memberships,
                &mut inheritance.collision_memberships,
                world_defaults.collision_memberships,
            );
            changed |= edit_inherited_u32(
                ui,
                "Collision Filter",
                &mut properties.collision_filter,
                &mut inheritance.collision_filter,
                world_defaults.collision_filter,
            );
            changed |= edit_inherited_u32(
                ui,
                "Solver Memberships",
                &mut properties.solver_memberships,
                &mut inheritance.solver_memberships,
                world_defaults.solver_memberships,
            );
            changed |= edit_inherited_u32(
                ui,
                "Solver Filter",
                &mut properties.solver_filter,
                &mut inheritance.solver_filter,
                world_defaults.solver_filter,
            );

            let mut friction_rule = properties.friction_combine_rule;
            let mut restitution_rule = properties.restitution_combine_rule;
            if inheritance.friction_combine_rule {
                friction_rule = world_defaults.friction_combine_rule;
                if properties.friction_combine_rule != world_defaults.friction_combine_rule {
                    changed = true;
                }
            }
            if inheritance.restitution_combine_rule {
                restitution_rule = world_defaults.restitution_combine_rule;
                if properties.restitution_combine_rule != world_defaults.restitution_combine_rule {
                    changed = true;
                }
            }

            ui.horizontal(|ui| {
                changed |= ui
                    .checkbox(&mut inheritance.friction_combine_rule, "Use World")
                    .changed();
                ui.label("Friction Combine Rule");
                ui.add_enabled_ui(!inheritance.friction_combine_rule, |ui| {
                    ComboBox::from_id_source(format!("friction_rule_{}", entity.to_bits()))
                        .selected_text(match friction_rule {
                            PhysicsCombineRule::Average => "Average",
                            PhysicsCombineRule::Min => "Min",
                            PhysicsCombineRule::Multiply => "Multiply",
                            PhysicsCombineRule::Max => "Max",
                        })
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    matches!(friction_rule, PhysicsCombineRule::Average),
                                    "Average",
                                )
                                .clicked()
                            {
                                friction_rule = PhysicsCombineRule::Average;
                                changed = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(friction_rule, PhysicsCombineRule::Min),
                                    "Min",
                                )
                                .clicked()
                            {
                                friction_rule = PhysicsCombineRule::Min;
                                changed = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(friction_rule, PhysicsCombineRule::Multiply),
                                    "Multiply",
                                )
                                .clicked()
                            {
                                friction_rule = PhysicsCombineRule::Multiply;
                                changed = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(friction_rule, PhysicsCombineRule::Max),
                                    "Max",
                                )
                                .clicked()
                            {
                                friction_rule = PhysicsCombineRule::Max;
                                changed = true;
                            }
                        });
                });
            });

            ui.horizontal(|ui| {
                changed |= ui
                    .checkbox(&mut inheritance.restitution_combine_rule, "Use World")
                    .changed();
                ui.label("Restitution Combine Rule");
                ui.add_enabled_ui(!inheritance.restitution_combine_rule, |ui| {
                    ComboBox::from_id_source(format!("restitution_rule_{}", entity.to_bits()))
                        .selected_text(match restitution_rule {
                            PhysicsCombineRule::Average => "Average",
                            PhysicsCombineRule::Min => "Min",
                            PhysicsCombineRule::Multiply => "Multiply",
                            PhysicsCombineRule::Max => "Max",
                        })
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    matches!(restitution_rule, PhysicsCombineRule::Average),
                                    "Average",
                                )
                                .clicked()
                            {
                                restitution_rule = PhysicsCombineRule::Average;
                                changed = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(restitution_rule, PhysicsCombineRule::Min),
                                    "Min",
                                )
                                .clicked()
                            {
                                restitution_rule = PhysicsCombineRule::Min;
                                changed = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(restitution_rule, PhysicsCombineRule::Multiply),
                                    "Multiply",
                                )
                                .clicked()
                            {
                                restitution_rule = PhysicsCombineRule::Multiply;
                                changed = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(restitution_rule, PhysicsCombineRule::Max),
                                    "Max",
                                )
                                .clicked()
                            {
                                restitution_rule = PhysicsCombineRule::Max;
                                changed = true;
                            }
                        });
                });
            });

            if inheritance.friction_combine_rule {
                friction_rule = world_defaults.friction_combine_rule;
            }
            if inheritance.restitution_combine_rule {
                restitution_rule = world_defaults.restitution_combine_rule;
            }
            properties.friction_combine_rule = friction_rule;
            properties.restitution_combine_rule = restitution_rule;

            if changed {
                if let Some(mut props) = world.get_mut::<ColliderProperties>(entity) {
                    *props = properties;
                }
                if let Some(mut inheritance_component) =
                    world.get_mut::<ColliderPropertyInheritance>(entity)
                {
                    *inheritance_component = inheritance;
                } else {
                    world.entity_mut(entity).insert(inheritance);
                }
                push_undo_snapshot(world, "Collider Properties");
            }
        }
        ui.separator();
    }

    if world.get::<ColliderShape>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Collider Shape");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<ColliderShape>();
            push_undo_snapshot(world, "Remove Collider");
        } else if let Some(shape) = world.get::<ColliderShape>(entity).copied() {
            let mut current = shape;
            let mut shape_changed = false;
            let label = match current {
                ColliderShape::Cuboid => "Box",
                ColliderShape::Sphere => "Sphere",
                ColliderShape::CapsuleY => "Capsule",
                ColliderShape::CylinderY => "Cylinder",
                ColliderShape::ConeY => "Cone",
                ColliderShape::RoundCuboid { .. } => "Round Box",
                ColliderShape::Mesh { kind, .. } => match kind {
                    MeshColliderKind::TriMesh => "Mesh",
                    MeshColliderKind::ConvexHull => "Mesh Convex Hull",
                },
            };
            ComboBox::from_id_source(format!("collider_shape_{}", entity.to_bits()))
                .selected_text(label)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(matches!(current, ColliderShape::Cuboid), "Box")
                        .clicked()
                    {
                        current = ColliderShape::Cuboid;
                        shape_changed = true;
                    }
                    if ui
                        .selectable_label(matches!(current, ColliderShape::Sphere), "Sphere")
                        .clicked()
                    {
                        current = ColliderShape::Sphere;
                        shape_changed = true;
                    }
                    if ui
                        .selectable_label(matches!(current, ColliderShape::CapsuleY), "Capsule")
                        .clicked()
                    {
                        current = ColliderShape::CapsuleY;
                        shape_changed = true;
                    }
                    if ui
                        .selectable_label(matches!(current, ColliderShape::CylinderY), "Cylinder")
                        .clicked()
                    {
                        current = ColliderShape::CylinderY;
                        shape_changed = true;
                    }
                    if ui
                        .selectable_label(matches!(current, ColliderShape::ConeY), "Cone")
                        .clicked()
                    {
                        current = ColliderShape::ConeY;
                        shape_changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(current, ColliderShape::RoundCuboid { .. }),
                            "Round Box",
                        )
                        .clicked()
                    {
                        current = ColliderShape::RoundCuboid {
                            border_radius: 0.05,
                        };
                        shape_changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(
                                current,
                                ColliderShape::Mesh {
                                    kind: MeshColliderKind::TriMesh,
                                    ..
                                }
                            ),
                            "Mesh",
                        )
                        .clicked()
                    {
                        let mesh_id = world
                            .get::<BevyMeshRenderer>(entity)
                            .map(|renderer| renderer.0.mesh_id)
                            .or_else(|| {
                                world
                                    .get::<BevySkinnedMeshRenderer>(entity)
                                    .map(|renderer| renderer.0.mesh_id)
                            });
                        current = ColliderShape::Mesh {
                            mesh_id,
                            lod: MeshColliderLod::Lowest,
                            kind: MeshColliderKind::TriMesh,
                        };
                        shape_changed = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(
                                current,
                                ColliderShape::Mesh {
                                    kind: MeshColliderKind::ConvexHull,
                                    ..
                                }
                            ),
                            "Mesh Convex Hull",
                        )
                        .clicked()
                    {
                        let mesh_id = world
                            .get::<BevyMeshRenderer>(entity)
                            .map(|renderer| renderer.0.mesh_id)
                            .or_else(|| {
                                world
                                    .get::<BevySkinnedMeshRenderer>(entity)
                                    .map(|renderer| renderer.0.mesh_id)
                            });
                        current = ColliderShape::Mesh {
                            mesh_id,
                            lod: MeshColliderLod::Lowest,
                            kind: MeshColliderKind::ConvexHull,
                        };
                        shape_changed = true;
                    }
                });

            if let ColliderShape::RoundCuboid { border_radius } = &mut current {
                let response = edit_float(ui, "Border Radius", border_radius, 0.01);
                if response.changed {
                    *border_radius = border_radius.max(0.0);
                    shape_changed = true;
                }
            }

            if let ColliderShape::Mesh { mesh_id, lod, kind } = &mut current {
                let mesh_source_label = mesh_id
                    .map(|id| format!("Collider Mesh: #{}", id))
                    .unwrap_or_else(|| "Collider Mesh: Entity Mesh".to_string());

                let mesh_source_button = ui.menu_button(mesh_source_label, |ui| {
                    if ui.button("Use Entity Mesh").clicked() {
                        *mesh_id = world
                            .get::<BevyMeshRenderer>(entity)
                            .map(|renderer| renderer.0.mesh_id)
                            .or_else(|| {
                                world
                                    .get::<BevySkinnedMeshRenderer>(entity)
                                    .map(|renderer| renderer.0.mesh_id)
                            });
                        shape_changed = true;
                        ui.close_menu();
                    }

                    if let Some(path) = selected_asset.as_ref().filter(|path| is_model_file(path)) {
                        if ui.button("Use Selected Asset").clicked() {
                            if let Some(id) = collider_mesh_id_from_asset(world, &project, path) {
                                *mesh_id = Some(id);
                                shape_changed = true;
                            }
                            ui.close_menu();
                        }
                    }

                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Model", &["glb", "gltf"])
                            .pick_file()
                        {
                            if let Some(id) = collider_mesh_id_from_asset(world, &project, &path) {
                                *mesh_id = Some(id);
                                shape_changed = true;
                            }
                        }
                        ui.close_menu();
                    }

                    if ui.button("Clear Mesh Override").clicked() {
                        *mesh_id = None;
                        shape_changed = true;
                        ui.close_menu();
                    }
                });

                if let Some(payload) =
                    typed_dnd_release_payload::<AssetDragPayload>(&mesh_source_button.response)
                {
                    if let Some(path) = payload_primary_path(&payload) {
                        if let Some(id) = collider_mesh_id_from_asset(world, &project, path) {
                            *mesh_id = Some(id);
                            shape_changed = true;
                        }
                    }
                }
                highlight_drop_target(ui, &mesh_source_button.response);

                ComboBox::from_id_source(format!("mesh_collider_lod_{}", entity.to_bits()))
                    .selected_text(match lod {
                        MeshColliderLod::Lod0 => "LOD0",
                        MeshColliderLod::Lod1 => "LOD1",
                        MeshColliderLod::Lod2 => "LOD2",
                        MeshColliderLod::Lowest => "Lowest",
                        MeshColliderLod::Specific(_) => "Specific",
                    })
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(matches!(lod, MeshColliderLod::Lod0), "LOD0")
                            .clicked()
                        {
                            *lod = MeshColliderLod::Lod0;
                            shape_changed = true;
                        }
                        if ui
                            .selectable_label(matches!(lod, MeshColliderLod::Lod1), "LOD1")
                            .clicked()
                        {
                            *lod = MeshColliderLod::Lod1;
                            shape_changed = true;
                        }
                        if ui
                            .selectable_label(matches!(lod, MeshColliderLod::Lod2), "LOD2")
                            .clicked()
                        {
                            *lod = MeshColliderLod::Lod2;
                            shape_changed = true;
                        }
                        if ui
                            .selectable_label(matches!(lod, MeshColliderLod::Lowest), "Lowest")
                            .clicked()
                        {
                            *lod = MeshColliderLod::Lowest;
                            shape_changed = true;
                        }
                    });

                ComboBox::from_id_source(format!("mesh_collider_kind_{}", entity.to_bits()))
                    .selected_text(match kind {
                        MeshColliderKind::TriMesh => "TriMesh",
                        MeshColliderKind::ConvexHull => "Convex Hull",
                    })
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(matches!(kind, MeshColliderKind::TriMesh), "TriMesh")
                            .clicked()
                        {
                            *kind = MeshColliderKind::TriMesh;
                            shape_changed = true;
                        }
                        if ui
                            .selectable_label(
                                matches!(kind, MeshColliderKind::ConvexHull),
                                "Convex Hull",
                            )
                            .clicked()
                        {
                            *kind = MeshColliderKind::ConvexHull;
                            shape_changed = true;
                        }
                    });
            }

            if shape_changed {
                if let Some(mut shape) = world.get_mut::<ColliderShape>(entity) {
                    *shape = current;
                }
                push_undo_snapshot(world, "Collider Shape");
            }
        }
        ui.separator();
    }

    if world.get::<PhysicsJoint>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Physics Joint");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });
        if remove {
            world.entity_mut(entity).remove::<PhysicsJoint>();
            push_undo_snapshot(world, "Remove Joint");
        } else {
            let pinned_entity = world
                .get_resource::<InspectorPinnedEntityResource>()
                .and_then(|res| res.0);
            if let Some(mut joint) = world.get_mut::<PhysicsJoint>(entity) {
                let mut changed = false;
                ComboBox::from_id_source(format!("joint_kind_{}", entity.to_bits()))
                    .selected_text(match joint.kind {
                        PhysicsJointKind::Fixed => "Fixed",
                        PhysicsJointKind::Spherical => "Spherical",
                        PhysicsJointKind::Revolute => "Revolute",
                        PhysicsJointKind::Prismatic => "Prismatic",
                    })
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(
                                matches!(joint.kind, PhysicsJointKind::Fixed),
                                "Fixed",
                            )
                            .clicked()
                        {
                            joint.kind = PhysicsJointKind::Fixed;
                            changed = true;
                        }
                        if ui
                            .selectable_label(
                                matches!(joint.kind, PhysicsJointKind::Spherical),
                                "Spherical",
                            )
                            .clicked()
                        {
                            joint.kind = PhysicsJointKind::Spherical;
                            changed = true;
                        }
                        if ui
                            .selectable_label(
                                matches!(joint.kind, PhysicsJointKind::Revolute),
                                "Revolute",
                            )
                            .clicked()
                        {
                            joint.kind = PhysicsJointKind::Revolute;
                            changed = true;
                        }
                        if ui
                            .selectable_label(
                                matches!(joint.kind, PhysicsJointKind::Prismatic),
                                "Prismatic",
                            )
                            .clicked()
                        {
                            joint.kind = PhysicsJointKind::Prismatic;
                            changed = true;
                        }
                    });

                changed |= ui
                    .checkbox(&mut joint.contacts_enabled, "Contacts Enabled")
                    .changed();
                changed |= edit_vec3(ui, "Anchor A", &mut joint.local_anchor1, 0.05).changed;
                changed |= edit_vec3(ui, "Anchor B", &mut joint.local_anchor2, 0.05).changed;
                changed |= edit_vec3(ui, "Axis A", &mut joint.local_axis1, 0.05).changed;
                changed |= edit_vec3(ui, "Axis B", &mut joint.local_axis2, 0.05).changed;
                changed |= ui
                    .checkbox(&mut joint.limit_enabled, "Use Limits")
                    .changed();
                if joint.limit_enabled {
                    let mut min_limit = joint.limits[0];
                    let mut max_limit = joint.limits[1];
                    changed |= edit_float(ui, "Limit Min", &mut min_limit, 0.01).changed;
                    changed |= edit_float(ui, "Limit Max", &mut max_limit, 0.01).changed;
                    joint.limits = [min_limit, max_limit.max(min_limit)];
                }
                changed |= ui
                    .checkbox(&mut joint.motor.enabled, "Motor Enabled")
                    .changed();
                if joint.motor.enabled {
                    changed |= edit_float(
                        ui,
                        "Motor Target Position",
                        &mut joint.motor.target_position,
                        0.01,
                    )
                    .changed;
                    changed |= edit_float(
                        ui,
                        "Motor Target Velocity",
                        &mut joint.motor.target_velocity,
                        0.01,
                    )
                    .changed;
                    changed |=
                        edit_float(ui, "Motor Stiffness", &mut joint.motor.stiffness, 0.01).changed;
                    changed |=
                        edit_float(ui, "Motor Damping", &mut joint.motor.damping, 0.01).changed;
                    changed |=
                        edit_float(ui, "Motor Max Force", &mut joint.motor.max_force, 0.1).changed;
                }

                ui.horizontal(|ui| {
                    if ui.button("Clear Target").clicked() {
                        joint.target = None;
                        changed = true;
                    }
                    if let Some(pinned) = pinned_entity {
                        if ui.button("Use Pinned Entity").clicked() {
                            joint.target = Some(pinned);
                            changed = true;
                        }
                    }
                });
                let target_label = joint
                    .target
                    .and_then(|target| world.get::<Name>(target))
                    .map(|name| name.to_string())
                    .unwrap_or_else(|| "<none>".to_string());
                ui.label(format!("Target: {}", target_label));

                if changed {
                    push_undo_snapshot(world, "Joint");
                }
            }
        }
        ui.separator();
    }

    if world.get::<CharacterController>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Character Controller");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });
        if remove {
            world.entity_mut(entity).remove::<CharacterController>();
            world
                .entity_mut(entity)
                .remove::<CharacterControllerInput>();
            world
                .entity_mut(entity)
                .remove::<CharacterControllerOutput>();
            push_undo_snapshot(world, "Remove Character Controller");
        } else if let Some(mut controller) = world.get_mut::<CharacterController>(entity) {
            let mut changed = false;
            changed |= edit_vec3(ui, "Up", &mut controller.up, 0.05).changed;
            changed |= edit_float(ui, "Offset", &mut controller.offset, 0.001).changed;
            changed |= ui.checkbox(&mut controller.slide, "Slide").changed();
            changed |= edit_float(
                ui,
                "Autostep Max Height",
                &mut controller.autostep_max_height,
                0.01,
            )
            .changed;
            changed |= edit_float(
                ui,
                "Autostep Min Width",
                &mut controller.autostep_min_width,
                0.01,
            )
            .changed;
            changed |= ui
                .checkbox(
                    &mut controller.autostep_include_dynamic_bodies,
                    "Autostep Include Dynamic Bodies",
                )
                .changed();
            changed |= edit_float(
                ui,
                "Max Slope Climb Angle",
                &mut controller.max_slope_climb_angle,
                0.01,
            )
            .changed;
            changed |= edit_float(
                ui,
                "Min Slope Slide Angle",
                &mut controller.min_slope_slide_angle,
                0.01,
            )
            .changed;
            changed |=
                edit_float(ui, "Snap To Ground", &mut controller.snap_to_ground, 0.01).changed;
            changed |= edit_float(
                ui,
                "Normal Nudge Factor",
                &mut controller.normal_nudge_factor,
                0.0001,
            )
            .changed;
            changed |= ui
                .checkbox(
                    &mut controller.apply_impulses_to_dynamic_bodies,
                    "Apply Impulses To Dynamics",
                )
                .changed();
            changed |=
                edit_float(ui, "Character Mass", &mut controller.character_mass, 0.1).changed;

            if let Some(mut input) = world.get_mut::<CharacterControllerInput>(entity) {
                changed |= edit_vec3(
                    ui,
                    "Desired Translation",
                    &mut input.desired_translation,
                    0.05,
                )
                .changed;
            } else {
                world
                    .entity_mut(entity)
                    .insert(CharacterControllerInput::default());
                changed = true;
            }
            if world.get::<CharacterControllerOutput>(entity).is_none() {
                world
                    .entity_mut(entity)
                    .insert(CharacterControllerOutput::default());
                changed = true;
            } else if let Some(output) = world.get::<CharacterControllerOutput>(entity) {
                ui.label(format!(
                    "Grounded: {} | Sliding: {} | Contacts: {}",
                    output.grounded, output.sliding_down_slope, output.collision_count
                ));
            }

            if changed {
                push_undo_snapshot(world, "Character Controller");
            }
        }
        ui.separator();
    }

    if world.get::<PhysicsRayCast>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Ray Cast Query");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<PhysicsRayCast>();
            world.entity_mut(entity).remove::<PhysicsRayCastHit>();
            push_undo_snapshot(world, "Remove Ray Cast Query");
        } else if let Some(mut query_data) = world.get_mut::<PhysicsRayCast>(entity) {
            let mut changed = false;
            changed |= edit_vec3(ui, "Origin", &mut query_data.origin, 0.05).changed;
            changed |= edit_vec3(ui, "Direction", &mut query_data.direction, 0.05).changed;
            changed |= edit_float(ui, "Max TOI", &mut query_data.max_toi, 0.1).changed;
            changed |= ui.checkbox(&mut query_data.solid, "Solid").changed();
            changed |= ui
                .checkbox(&mut query_data.exclude_self, "Exclude Self")
                .changed();
            changed |= edit_u32_range(
                ui,
                "Filter Flags",
                &mut query_data.filter.flags,
                1.0,
                0..=u32::MAX,
            )
            .changed;
            changed |= ui
                .checkbox(&mut query_data.filter.use_groups, "Use Groups")
                .changed();
            if query_data.filter.use_groups {
                changed |= edit_u32_range(
                    ui,
                    "Group Memberships",
                    &mut query_data.filter.groups_memberships,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
                changed |= edit_u32_range(
                    ui,
                    "Group Filter",
                    &mut query_data.filter.groups_filter,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
            }
            if let Some(hit) = world.get::<PhysicsRayCastHit>(entity) {
                ui.label(format!("Hit: {} | TOI: {:.4}", hit.has_hit, hit.toi));
                if let Some(hit_entity) = hit.hit_entity {
                    ui.label(format!(
                        "Hit Entity: {}",
                        entity_display_name(world, hit_entity)
                    ));
                }
            }
            if changed {
                push_undo_snapshot(world, "Ray Cast Query");
            }
        }
        ui.separator();
    }

    if world.get::<PhysicsPointProjection>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Point Projection Query");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<PhysicsPointProjection>();
            world
                .entity_mut(entity)
                .remove::<PhysicsPointProjectionHit>();
            push_undo_snapshot(world, "Remove Point Projection Query");
        } else if let Some(mut query_data) = world.get_mut::<PhysicsPointProjection>(entity) {
            let mut changed = false;
            changed |= edit_vec3(ui, "Point", &mut query_data.point, 0.05).changed;
            changed |= ui.checkbox(&mut query_data.solid, "Solid").changed();
            changed |= ui
                .checkbox(&mut query_data.exclude_self, "Exclude Self")
                .changed();
            changed |= edit_u32_range(
                ui,
                "Filter Flags",
                &mut query_data.filter.flags,
                1.0,
                0..=u32::MAX,
            )
            .changed;
            changed |= ui
                .checkbox(&mut query_data.filter.use_groups, "Use Groups")
                .changed();
            if query_data.filter.use_groups {
                changed |= edit_u32_range(
                    ui,
                    "Group Memberships",
                    &mut query_data.filter.groups_memberships,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
                changed |= edit_u32_range(
                    ui,
                    "Group Filter",
                    &mut query_data.filter.groups_filter,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
            }
            if let Some(hit) = world.get::<PhysicsPointProjectionHit>(entity) {
                ui.label(format!(
                    "Hit: {} | Distance: {:.4} | Inside: {}",
                    hit.has_hit, hit.distance, hit.is_inside
                ));
            }
            if changed {
                push_undo_snapshot(world, "Point Projection Query");
            }
        }
        ui.separator();
    }

    if world.get::<PhysicsShapeCast>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Shape Cast Query");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<PhysicsShapeCast>();
            world.entity_mut(entity).remove::<PhysicsShapeCastHit>();
            push_undo_snapshot(world, "Remove Shape Cast Query");
        } else if let Some(mut query_data) = world.get_mut::<PhysicsShapeCast>(entity) {
            let mut changed = false;
            changed |= edit_vec3(ui, "Scale", &mut query_data.scale, 0.05).changed;
            changed |= edit_vec3(ui, "Position", &mut query_data.position, 0.05).changed;
            changed |= edit_vec3(ui, "Velocity", &mut query_data.velocity, 0.05).changed;
            let mut rotation = query_data.rotation.to_euler(EulerRot::XYZ);
            let mut rotation_deg = Vec3::new(
                rotation.0.to_degrees(),
                rotation.1.to_degrees(),
                rotation.2.to_degrees(),
            );
            let rotation_response = edit_vec3(ui, "Rotation", &mut rotation_deg, 0.5);
            if rotation_response.changed {
                query_data.rotation = Quat::from_euler(
                    EulerRot::XYZ,
                    rotation_deg.x.to_radians(),
                    rotation_deg.y.to_radians(),
                    rotation_deg.z.to_radians(),
                );
                changed = true;
            }
            changed |= edit_float(
                ui,
                "Max Time Of Impact",
                &mut query_data.max_time_of_impact,
                0.1,
            )
            .changed;
            changed |=
                edit_float(ui, "Target Distance", &mut query_data.target_distance, 0.01).changed;
            changed |= ui
                .checkbox(&mut query_data.stop_at_penetration, "Stop At Penetration")
                .changed();
            changed |= ui
                .checkbox(
                    &mut query_data.compute_impact_geometry_on_penetration,
                    "Compute Penetration Geometry",
                )
                .changed();
            changed |= ui
                .checkbox(&mut query_data.exclude_self, "Exclude Self")
                .changed();
            changed |= edit_u32_range(
                ui,
                "Filter Flags",
                &mut query_data.filter.flags,
                1.0,
                0..=u32::MAX,
            )
            .changed;
            changed |= ui
                .checkbox(&mut query_data.filter.use_groups, "Use Groups")
                .changed();
            if query_data.filter.use_groups {
                changed |= edit_u32_range(
                    ui,
                    "Group Memberships",
                    &mut query_data.filter.groups_memberships,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
                changed |= edit_u32_range(
                    ui,
                    "Group Filter",
                    &mut query_data.filter.groups_filter,
                    1.0,
                    0..=u32::MAX,
                )
                .changed;
            }
            if let Some(hit) = world.get::<PhysicsShapeCastHit>(entity) {
                ui.label(format!(
                    "Hit: {} | TOI: {:.4} | Status: {:?}",
                    hit.has_hit, hit.toi, hit.status
                ));
            }
            if changed {
                push_undo_snapshot(world, "Shape Cast Query");
            }
        }
        ui.separator();
    }

    if world.get::<SceneRoot>(entity).is_some() || world.get::<SceneAssetPath>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Scene Asset");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<SceneRoot>();
            world.entity_mut(entity).remove::<SceneAssetPath>();
            push_undo_snapshot(world, "Remove Scene Asset");
        } else {
            let mut scene_asset_changed = false;
            if let Some(scene_path) = world.get::<SceneAssetPath>(entity) {
                let path_label = if scene_path.path.as_os_str().is_empty() {
                    "Path: <none>".to_string()
                } else {
                    format!(
                        "Path: {}",
                        project_relative_path(&project, &scene_path.path)
                    )
                };
                ui.add(egui::Label::new(path_label).wrap_mode(egui::TextWrapMode::Extend));
            }

            let scene_asset_button = ui.menu_button("Scene Source", |ui| {
                if let Some(path) = selected_asset.as_ref().filter(|path| is_model_file(path)) {
                    if ui.button("Use Selected Asset").clicked() {
                        apply_scene_asset(world, entity, path);
                        scene_asset_changed = true;
                        ui.close_menu();
                    }
                } else {
                    ui.label("Select a scene asset");
                }

                if ui.button("Browse...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Scene", &["glb", "gltf"])
                        .pick_file()
                    {
                        apply_scene_asset(world, entity, &path);
                        scene_asset_changed = true;
                    }
                    ui.close_menu();
                }
            });

            if let Some(payload) =
                typed_dnd_release_payload::<AssetDragPayload>(&scene_asset_button.response)
            {
                if let Some(path) = payload_primary_path(&payload) {
                    if try_apply_scene_asset_path(world, entity, path) {
                        scene_asset_changed = true;
                    }
                }
            }
            highlight_drop_target(ui, &scene_asset_button.response);
            if scene_asset_changed {
                push_undo_snapshot(world, "Scene Asset");
            }
        }
        ui.separator();
    }

    if let Some(script_component) = world.get::<ScriptComponent>(entity).cloned() {
        let mut remove_component = false;
        ui.horizontal(|ui| {
            ui.heading("Scripts");
            if ui.button("Remove").clicked() {
                remove_component = true;
            }
        });

        if remove_component {
            world.entity_mut(entity).remove::<ScriptComponent>();
            push_undo_snapshot(world, "Remove Scripts");
        } else {
            let mut scripts = script_component.scripts;
            let mut remove_indices = Vec::new();
            let mut scripts_changed = false;
            let mut script_edit_response = EditResponse::default();

            let world_state = world
                .get_resource::<EditorSceneState>()
                .map(|state| state.world_state)
                .unwrap_or(WorldState::Edit);
            let execute_scripts_in_edit_mode = world
                .get_resource::<EditorViewportState>()
                .map(|state| state.execute_scripts_in_edit_mode)
                .unwrap_or(false);

            for (index, script) in scripts.iter_mut().enumerate() {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(format!("Script {}", index + 1));
                    if ui.button("Remove").clicked() {
                        remove_indices.push(index);
                        scripts_changed = true;
                    }
                });

                let mut path_string = script
                    .path
                    .as_ref()
                    .map(|path| path.to_string_lossy().to_string())
                    .unwrap_or_default();
                let path_response = ui.text_edit_singleline(&mut path_string);
                let mut edit_response = EditResponse::from_response(&path_response);
                if path_response.has_focus()
                    && ui.input(|input| input.key_pressed(egui::Key::Enter))
                {
                    edit_response.lost_focus = true;
                }
                begin_edit_undo(world, "Scripts", edit_response);

                let mut updated_path = script.path.clone();
                if edit_response.changed {
                    let trimmed = path_string.trim();
                    updated_path = if trimmed.is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(trimmed))
                    };
                }
                script_edit_response.merge(edit_response);

                if let Some(payload) = typed_dnd_release_payload::<AssetDragPayload>(&path_response)
                {
                    if let Some(path) = payload_primary_path(&payload) {
                        if is_script_file(path) {
                            updated_path = Some(path.clone());
                            path_string = path.to_string_lossy().to_string();
                            scripts_changed = true;
                        } else {
                            set_status(world, "Select a script asset to assign".to_string());
                        }
                    }
                }

                highlight_drop_target(ui, &path_response);
                script.path = updated_path;
                let normalized_language =
                    normalize_script_language(&script.language, script.path.as_deref());
                if script.language != normalized_language {
                    script.language = normalized_language;
                    scripts_changed = true;
                }

                if ui.button("Browse...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Script", &["lua", "luau", "rs", "toml"])
                        .pick_file()
                    {
                        if is_script_file(&path) {
                            script.path = Some(path.clone());
                            script.language = script_language_from_path(&path);
                            scripts_changed = true;
                        } else {
                            set_status(
                                world,
                                "Selected file is not a supported script".to_string(),
                            );
                        }
                    }
                }

                if let Some(path) = selected_asset.as_ref() {
                    if is_script_file(path) && ui.button("Use Selected Script").clicked() {
                        script.path = Some(path.clone());
                        script.language = script_language_from_path(path);
                        scripts_changed = true;
                    }
                }

                ui.horizontal(|ui| {
                    if ui.button("Create Lua Script").clicked() {
                        if let Some(project) = project.as_ref() {
                            if let Some(path) = create_script_asset(world, project) {
                                script.path = Some(path.clone());
                                script.language = script_language_from_path(&path);
                                scripts_changed = true;
                            }
                        } else {
                            set_status(world, "Open a project before creating scripts".to_string());
                        }
                    }
                    if ui.button("Create Rust Script").clicked() {
                        if let Some(project) = project.as_ref() {
                            if let Some(path) = create_rust_script_asset(world, project) {
                                script.path = Some(path.clone());
                                script.language = script_language_from_path(&path);
                                scripts_changed = true;
                            }
                        } else {
                            set_status(world, "Open a project before creating scripts".to_string());
                        }
                    }
                });

                ui.label(format!("Language: {}", script.language));

                let script_key = ScriptInstanceKey {
                    entity,
                    script_index: index,
                };
                let runtime_running = world
                    .get_resource::<ScriptRuntime>()
                    .map(|runtime| runtime.contains_instance(script_key))
                    .unwrap_or(false);
                let manual_running = world
                    .get_resource::<ScriptEditModeState>()
                    .map(|state| state.running_in_edit.contains(&script_key))
                    .unwrap_or(false);

                if world_state == WorldState::Edit {
                    let can_run = script.path.is_some();
                    if execute_scripts_in_edit_mode {
                        ui.horizontal(|ui| {
                            ui.label(if runtime_running {
                                "Status: Running (auto)"
                            } else {
                                "Status: Auto (idle)"
                            });
                            if ui
                                .add_enabled(can_run, egui::Button::new("Restart"))
                                .clicked()
                            {
                                if let Some(mut state) =
                                    world.get_resource_mut::<ScriptEditModeState>()
                                {
                                    state.queue(ScriptEditCommand::Restart(script_key));
                                }
                            }
                        });
                    } else {
                        ui.horizontal(|ui| {
                            let running = runtime_running || manual_running;
                            ui.label(if running {
                                "Status: Running"
                            } else {
                                "Status: Stopped"
                            });

                            if ui.add_enabled(can_run, egui::Button::new("Run")).clicked() {
                                if let Some(mut state) =
                                    world.get_resource_mut::<ScriptEditModeState>()
                                {
                                    state.queue(ScriptEditCommand::Run(script_key));
                                }
                            }
                            if ui
                                .add_enabled(can_run, egui::Button::new("Restart"))
                                .clicked()
                            {
                                if let Some(mut state) =
                                    world.get_resource_mut::<ScriptEditModeState>()
                                {
                                    state.queue(ScriptEditCommand::Restart(script_key));
                                }
                            }
                            if ui.add_enabled(running, egui::Button::new("Stop")).clicked() {
                                if let Some(mut state) =
                                    world.get_resource_mut::<ScriptEditModeState>()
                                {
                                    state.queue(ScriptEditCommand::Stop(script_key));
                                }
                            }
                        });
                    }
                } else {
                    ui.label(if runtime_running {
                        "Status: Running (play mode)"
                    } else {
                        "Status: Waiting for Play mode"
                    });
                }
            }

            for index in remove_indices.into_iter().rev() {
                if index < scripts.len() {
                    scripts.remove(index);
                }
            }

            if ui.button("Add Script").clicked() {
                scripts.push(ScriptEntry::new());
                scripts_changed = true;
            }

            if scripts.is_empty() {
                world.entity_mut(entity).remove::<ScriptComponent>();
                scripts_changed = true;
            } else {
                world.entity_mut(entity).insert(ScriptComponent { scripts });
            }

            end_edit_undo(world, script_edit_response);
            if scripts_changed {
                push_undo_snapshot(world, "Scripts");
            }
        }
        ui.separator();
    }

    if world.get::<DynamicComponents>(entity).is_some() {
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("Dynamic Components");
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });

        if remove {
            world.entity_mut(entity).remove::<DynamicComponents>();
            push_undo_snapshot(world, "Remove Dynamic Components");
        } else {
            draw_dynamic_components_section(ui, world, entity);
        }
        ui.separator();
    }

    draw_add_component_menu(ui, world, entity, &project, selected_asset);
}

pub fn draw_assets_window(ui: &mut Ui, world: &mut World) {
    let root = match world
        .get_resource::<AssetBrowserState>()
        .and_then(|state| state.root.clone())
    {
        Some(root) => root,
        None => {
            with_middle_drag_blocked(ui, world, |ui, _world| {
                ui.label("Open a project to browse assets.");
            });
            return;
        }
    };

    with_middle_drag_blocked(ui, world, |ui, world| {
        {
            let mut state = world
                .get_resource_mut::<AssetBrowserState>()
                .expect("AssetBrowserState missing");
            ui.horizontal(|ui| {
                ui.label("Filter:");
                if ui.text_edit_singleline(&mut state.filter).changed() {
                    state.refresh_requested = true;
                }
                if ui.button("Refresh").clicked() {
                    state.refresh_requested = true;
                }
                ui.separator();
                ui.add(egui::Slider::new(&mut state.tile_size, 64.0..=220.0).text("Tile Size"));
                state.tile_size = state.tile_size.clamp(64.0, 220.0);
            });

            if state
                .current_dir
                .as_ref()
                .map(|path| !path.exists())
                .unwrap_or(true)
            {
                state.current_dir = Some(root.clone());
            }
            let current_dir = state.current_dir.clone();
            if let Some(current_dir) = current_dir.as_ref() {
                let location_label = asset_path_label(&root, current_dir);
                let breadcrumb_dirs = asset_breadcrumb_dirs(&root, current_dir);
                let mut selected_dir: Option<PathBuf> = None;
                ui.horizontal(|ui| {
                    ui.label("Location:");
                    ComboBox::from_id_source("asset_location_dropdown")
                        .selected_text(location_label)
                        .show_ui(ui, |ui| {
                            for entry in breadcrumb_dirs.iter() {
                                let label = asset_path_label(&root, entry);
                                if ui
                                    .selectable_label(
                                        entry.as_path() == current_dir.as_path(),
                                        label,
                                    )
                                    .clicked()
                                {
                                    selected_dir = Some(entry.clone());
                                }
                            }
                        });
                });

                if let Some(selected_dir) = selected_dir {
                    state.current_dir = Some(selected_dir.clone());
                    state.selected = Some(selected_dir.clone());
                    state.selected_paths.clear();
                    state.selected_paths.insert(selected_dir.clone());
                    state.selection_anchor = Some(selected_dir);
                }
            }

            if let Some(status) = state.status.clone() {
                ui.label(RichText::new(status).small());
            }
        }

        ui.separator();
        let content_height = ui.available_height();
        ui.horizontal(|ui| {
            let sidebar_width = 220.0;
            ui.allocate_ui_with_layout(
                Vec2::new(sidebar_width, content_height),
                Layout::top_down(Align::Min),
                |ui| {
                    ui.heading("Folders");
                    draw_asset_tree(ui, world, &root);
                },
            );
            ui.separator();
            ui.allocate_ui_with_layout(
                Vec2::new(ui.available_width(), content_height),
                Layout::top_down(Align::Min),
                |ui| {
                    draw_asset_grid(ui, world, &root);
                },
            );
        });
    });
}

#[derive(Clone)]
struct AssetDragPayload {
    paths: Vec<PathBuf>,
}

#[derive(Clone)]
struct EntityDragPayload {
    entity: Entity,
}

fn typed_dnd_release_payload<Payload>(response: &Response) -> Option<Arc<Payload>>
where
    Payload: Any + Send + Sync,
{
    if response.dnd_hover_payload::<Payload>().is_some() {
        response.dnd_release_payload::<Payload>()
    } else {
        None
    }
}

fn highlight_drop_target(ui: &Ui, response: &Response) {
    if response.dnd_hover_payload::<AssetDragPayload>().is_some()
        || response.dnd_hover_payload::<EntityDragPayload>().is_some()
    {
        let border = Stroke::new(
            ui.visuals().selection.stroke.width.max(1.5),
            ui.visuals().selection.stroke.color,
        );
        let rounding = ui.visuals().widgets.active.rounding();
        ui.painter().rect_stroke(
            response.rect.expand(4.0),
            rounding,
            border,
            StrokeKind::Inside,
        );
    }
}

fn draw_entity_drag_indicator(ui: &Ui, world: &World) {
    let Some(state) = world.get_resource::<EntityDragState>() else {
        return;
    };
    if !state.active {
        return;
    }
    let Some(entity) = state.entity else {
        return;
    };
    let Some(pointer_pos) = ui.ctx().input(|input| input.pointer.latest_pos()) else {
        return;
    };

    let name = world
        .get::<Name>(entity)
        .map(|name| name.to_string())
        .unwrap_or_else(|| format!("Entity {}", entity.index()));
    let text = format!("Dragging: {}", name);
    let font = FontId::proportional(12.0);
    let padding = Vec2::new(8.0, 4.0);
    let galley = ui
        .painter()
        .layout_no_wrap(text.clone(), font.clone(), Color32::WHITE);
    let rect = Rect::from_min_size(
        pointer_pos + Vec2::new(12.0, 12.0),
        galley.size() + padding * 2.0,
    );
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        Order::Foreground,
        Id::new("entity_drag_indicator"),
    ));
    painter.rect_filled(rect, 4.0, ui.visuals().selection.bg_fill);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        font,
        ui.visuals().text_color(),
    );
}

fn try_apply_scene_asset_path(world: &mut World, entity: Entity, path: &Path) -> bool {
    if is_model_file(path) {
        apply_scene_asset(world, entity, path);
        true
    } else {
        set_status(world, "Select a model asset to add".to_string());
        false
    }
}

fn ensure_scene_asset_placeholder(world: &mut World, entity: Entity) {
    if world.get::<SceneAssetPath>(entity).is_none() {
        world.entity_mut(entity).insert(SceneAssetPath {
            path: PathBuf::new(),
        });
    }
}

fn bring_window_to_front_if_dragging(ui: &Ui, world: &World) {
    let dragging = world
        .get_resource::<AssetDragState>()
        .map(|state| state.active)
        .unwrap_or(false)
        || world
            .get_resource::<EntityDragState>()
            .map(|state| state.active)
            .unwrap_or(false);
    if !dragging {
        return;
    }
    let pointer_pos = ui.ctx().input(|input| input.pointer.latest_pos());
    if let Some(pointer_pos) = pointer_pos {
        if let Some(layer_id) = ui.ctx().layer_id_at(pointer_pos) {
            if layer_id == ui.layer_id() {
                ui.ctx().move_to_top(layer_id);
            }
        }
    }
}

fn update_middle_drag_state(ui: &Ui, world: &mut World) {
    let (middle_down, just_pressed, pos, delta) = ui.ctx().input(|input| {
        (
            input.pointer.button_down(PointerButton::Middle),
            input.pointer.button_pressed(PointerButton::Middle),
            input.pointer.latest_pos(),
            input.pointer.delta(),
        )
    });

    let mut state = world
        .get_resource_mut::<MiddleDragUiState>()
        .expect("MiddleDragUiState missing");

    let was_active = state.active;
    state.delta = if middle_down { delta } else { Vec2::ZERO };
    state.active = middle_down;

    if just_pressed && !was_active {
        if let Some(pos) = pos {
            state.start_pos = pos;
            state.locked_window_id = ui.ctx().layer_id_at(pos).map(|layer| layer.id);
        } else {
            state.locked_window_id = None;
        }
    } else if !middle_down {
        state.locked_window_id = None;
    }
}

fn with_middle_drag_blocked(
    ui: &mut Ui,
    world: &mut World,
    add_contents: impl FnOnce(&mut Ui, &mut World),
) {
    let active = world
        .get_resource::<MiddleDragUiState>()
        .map(|state| state.active)
        .unwrap_or(false);
    if active {
        ui.add_enabled_ui(false, |ui| add_contents(ui, world));
    } else {
        add_contents(ui, world);
    }
}

fn drag_egui_window_on_middle_click(ui: &Ui, world: &mut World, window_id: &str) {
    update_middle_drag_state(ui, world);
    let (active, delta, locked_window_id) = {
        let drag_state = world
            .get_resource::<MiddleDragUiState>()
            .expect("MiddleDragUiState missing");
        (
            drag_state.active,
            drag_state.delta,
            drag_state.locked_window_id,
        )
    };
    if !active || delta == Vec2::ZERO {
        return;
    }
    let rect = ui.ctx().memory(|mem| mem.area_rect(Id::new(window_id)));
    let Some(rect) = rect else {
        return;
    };
    let locked_ok = locked_window_id
        .map(|id| id == Id::new(window_id))
        .unwrap_or(false);
    if !locked_ok {
        return;
    }
    if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
        let entry = egui_res
            .window_positions
            .entry(window_id.to_string())
            .or_insert(rect.min);
        *entry += delta;
        egui_res.window_dragging.insert(window_id.to_string());
    }
}

fn draw_asset_tree(ui: &mut Ui, world: &mut World, root: &Path) {
    let (entries, expanded, current_dir, rename_path) = {
        let state = world
            .get_resource::<AssetBrowserState>()
            .expect("AssetBrowserState missing");
        (
            state.entries.clone(),
            state.expanded.clone(),
            state.current_dir.clone(),
            state.rename_path.clone(),
        )
    };

    egui::ScrollArea::vertical()
        .id_salt("asset_tree_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for entry in entries.iter().filter(|entry| entry.is_dir) {
                if !is_entry_visible(entry, root, &expanded) {
                    continue;
                }

                let depth_indent = (entry.depth as f32) * 12.0;
                let name = asset_display_name(&entry.path);
                let is_expanded = expanded.contains(&entry.path);
                let is_current = current_dir.as_ref() == Some(&entry.path);
                let is_renaming = rename_path.as_ref() == Some(&entry.path);

                ui.horizontal(|ui| {
                    ui.add_space(depth_indent);
                    let arrow = if is_expanded { "v" } else { ">" };
                    if ui.small_button(arrow).clicked() {
                        toggle_expand(world, entry.path.clone());
                    }

                    if is_renaming {
                        asset_rename_editor(ui, world, &entry.path);
                        return;
                    }

                    let response = ui.add(
                        egui::Button::selectable(is_current, format!("[DIR] {}", name))
                            .sense(Sense::click_and_drag()),
                    );

                    if let Some(mut drag_state) = world.get_resource_mut::<AssetDragState>() {
                        if response.drag_started() {
                            drag_state.start_drag(entry.path.clone());
                        }
                        if response.drag_stopped() {
                            drag_state.stop_drag();
                        }
                    }

                    response.dnd_set_drag_payload(AssetDragPayload {
                        paths: vec![entry.path.clone()],
                    });

                    highlight_drop_target(ui, &response);

                    if response.clicked() {
                        set_current_dir(world, entry.path.clone());
                    }

                    if response.double_clicked() {
                        set_current_dir(world, entry.path.clone());
                        toggle_expand(world, entry.path.clone());
                    }

                    if let Some(payload) = typed_dnd_release_payload::<AssetDragPayload>(&response)
                    {
                        move_assets(world, &payload.paths, &entry.path);
                    }

                    response.context_menu(|ui| {
                        asset_dir_menu(world, ui, &entry.path);
                    });
                });
            }
        });
}

fn draw_asset_grid(ui: &mut Ui, world: &mut World, root: &Path) {
    let (entries, current_dir, selected_paths, rename_path, tile_size) = {
        let state = world
            .get_resource::<AssetBrowserState>()
            .expect("AssetBrowserState missing");
        (
            state.entries.clone(),
            state
                .current_dir
                .clone()
                .unwrap_or_else(|| root.to_path_buf()),
            state.selected_paths.clone(),
            state.rename_path.clone(),
            state.tile_size,
        )
    };

    let tile_size = tile_size.clamp(64.0, 220.0);
    let mut items: Vec<AssetEntry> = entries
        .into_iter()
        .filter(|entry| entry.path.parent() == Some(current_dir.as_path()))
        .collect();

    items.sort_by(|a, b| {
        b.is_dir.cmp(&a.is_dir).then_with(|| {
            asset_display_name(&a.path)
                .to_ascii_lowercase()
                .cmp(&asset_display_name(&b.path).to_ascii_lowercase())
        })
    });

    let old_spacing = ui.spacing().item_spacing;
    ui.spacing_mut().item_spacing = Vec2::new(12.0, 12.0);

    egui::ScrollArea::vertical()
        .id_salt("asset_grid_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let background_id = ui.id().with("asset_grid_background");
            let background_rect = ui.available_rect_before_wrap();
            let background_response =
                ui.interact(background_rect, background_id, Sense::click_and_drag());
            background_response.context_menu(|ui| {
                if ui.button("Open in File Browser").clicked() {
                    open_in_file_browser(world, &current_dir);
                    ui.close_menu();
                }
                ui.separator();
                if let Some(selection) = selected_assets_for_action(world, None) {
                    if ui.button("Duplicate Selected").clicked() {
                        duplicate_assets(world, &selection);
                        ui.close_menu();
                    }
                    if ui.button("Delete Selected").clicked() {
                        delete_assets(world, &selection);
                        ui.close_menu();
                    }
                    ui.separator();
                }
                asset_create_menu(world, ui, &current_dir);
            });
            if background_response.clicked_by(PointerButton::Primary) {
                if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
                    state.selected = None;
                    state.selected_paths.clear();
                    state.selection_anchor = None;
                    state.selection_drag_start = None;
                    state.rename_path = None;
                    state.rename_buffer.clear();
                }
            }
            if background_response.drag_started_by(PointerButton::Primary) {
                let modifiers = ui.input(|input| input.modifiers);
                if let Some(pointer_pos) = ui.input(|input| input.pointer.interact_pos()) {
                    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
                        state.selection_drag_start = Some(pointer_pos);
                        if !(modifiers.command || modifiers.ctrl) {
                            state.selected = None;
                            state.selected_paths.clear();
                            state.selection_anchor = None;
                        }
                        state.rename_path = None;
                        state.rename_buffer.clear();
                    }
                }
            }

            let spacing = ui.spacing().item_spacing.x;
            let mut available_width = ui.available_width();
            if !available_width.is_finite() {
                available_width = tile_size;
            }
            let available_width = available_width.max(tile_size);
            let columns = ((available_width + spacing) / (tile_size + spacing))
                .floor()
                .max(1.0) as usize;

            let mut tile_rects = Vec::new();
            let mut index = 0;
            while index < items.len() {
                ui.horizontal(|ui| {
                    for _ in 0..columns {
                        if index >= items.len() {
                            break;
                        }
                        let entry = &items[index];
                        let is_selected = selected_paths.contains(&entry.path);
                        let is_renaming = rename_path.as_ref() == Some(&entry.path);
                        let rect = draw_asset_tile(
                            ui,
                            world,
                            entry,
                            is_selected,
                            is_renaming,
                            tile_size,
                            index,
                            &items,
                        );
                        tile_rects.push((entry.path.clone(), rect));
                        index += 1;
                    }
                });
            }

            if background_response.dragged_by(PointerButton::Primary)
                || background_response.drag_stopped_by(PointerButton::Primary)
            {
                let pointer_pos = ui.input(|input| {
                    input
                        .pointer
                        .interact_pos()
                        .or_else(|| input.pointer.hover_pos())
                });
                let drag_start = world
                    .get_resource::<AssetBrowserState>()
                    .and_then(|state| state.selection_drag_start);
                if let (Some(pointer_pos), Some(start)) = (pointer_pos, drag_start) {
                    let rect = Rect::from_two_pos(start, pointer_pos);
                    let modifiers = ui.input(|input| input.modifiers);
                    update_drag_selection(world, &tile_rects, rect, modifiers);
                    paint_selection_rect(ui, rect);
                }
                if background_response.drag_stopped_by(PointerButton::Primary) {
                    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
                        state.selection_drag_start = None;
                    }
                }
            }
        });

    ui.spacing_mut().item_spacing = old_spacing;
}

fn draw_asset_tile(
    ui: &mut Ui,
    world: &mut World,
    entry: &AssetEntry,
    is_selected: bool,
    is_renaming: bool,
    tile_size: f32,
    index: usize,
    items: &[AssetEntry],
) -> Rect {
    let tile_size = Vec2::new(tile_size, tile_size);
    let sense = if is_renaming {
        Sense::click()
    } else {
        Sense::click_and_drag()
    };
    let (rect, response) = ui.allocate_exact_size(tile_size, sense);

    if let Some(mut drag_state) = world.get_resource_mut::<AssetDragState>() {
        if response.drag_started_by(PointerButton::Primary) {
            drag_state.start_drag(entry.path.clone());
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            drag_state.stop_drag();
        }
    }

    let mut bg_color = ui.visuals().widgets.inactive.bg_fill;
    if is_selected {
        bg_color = ui.visuals().selection.bg_fill;
    }
    if entry.is_dir && response.dnd_hover_payload::<AssetDragPayload>().is_some() {
        bg_color = ui.visuals().widgets.hovered.bg_fill;
    }

    ui.painter().rect_filled(rect, 6.0, bg_color);

    if !is_renaming {
        response.dnd_set_drag_payload(asset_drag_payload(world, &entry.path));
    }

    if is_renaming {
        ui.allocate_ui_at_rect(rect.shrink(6.0), |ui| {
            ui.vertical_centered(|ui| {
                ui.label("Rename");
                asset_rename_editor(ui, world, &entry.path);
            });
        });
    } else {
        ui.allocate_ui_at_rect(rect.shrink(8.0), |ui| {
            ui.with_layout(Layout::top_down(Align::Center), |ui| {
                let thumb = (tile_size.x * 0.55).clamp(32.0, tile_size.x - 16.0);
                let thumb_size = Vec2::new(thumb, thumb);
                let (thumb_rect, _) = ui.allocate_exact_size(thumb_size, Sense::hover());
                let thumb_color = asset_thumbnail_color(entry);
                ui.painter().rect_filled(thumb_rect, 6.0, thumb_color);
                ui.painter().text(
                    thumb_rect.center(),
                    Align2::CENTER_CENTER,
                    asset_thumbnail_tag(entry),
                    FontId::proportional(12.0),
                    Color32::WHITE,
                );
                ui.add_space(6.0);
                let name = asset_display_name(&entry.path);
                ui.label(RichText::new(name).small());
            });
        });
    }

    let mut double_clicked = response.double_clicked();
    if response.clicked_by(PointerButton::Primary) {
        update_asset_selection(world, &entry.path, index, items);
        let click_time = ui.input(|input| input.time);
        if register_asset_click(world, &entry.path, click_time) {
            double_clicked = true;
        }
    }

    if double_clicked {
        on_asset_double_click(world, entry);
    }

    if entry.is_dir {
        if let Some(payload) = typed_dnd_release_payload::<AssetDragPayload>(&response) {
            move_assets(world, &payload.paths, &entry.path);
        }
    }

    response.context_menu(|ui| {
        if entry.is_dir {
            asset_dir_menu(world, ui, &entry.path);
        } else {
            asset_file_menu(world, ui, &entry.path);
        }
    });

    if response.dragged_by(PointerButton::Primary) {
        if let Some(pointer_pos) = ui.ctx().input(|input| input.pointer.hover_pos()) {
            let selection_count = if is_selected {
                world
                    .get_resource::<AssetBrowserState>()
                    .map(|state| state.selected_paths.len())
                    .unwrap_or(1)
            } else {
                1
            };
            paint_asset_drag_preview(
                ui,
                entry,
                tile_size,
                is_selected,
                selection_count,
                pointer_pos,
            );
        }
    }

    rect
}

fn paint_asset_drag_preview(
    ui: &Ui,
    entry: &AssetEntry,
    tile_size: Vec2,
    is_selected: bool,
    selection_count: usize,
    pointer_pos: Pos2,
) {
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        Order::Tooltip,
        Id::new("asset_drag_preview"),
    ));
    let scale = 1.08;
    let drag_size = tile_size * scale;
    let rect = Rect::from_center_size(pointer_pos, drag_size);
    let shadow_offset = Vec2::new(6.0, 6.0);
    painter.rect_filled(
        rect.translate(shadow_offset),
        8.0,
        Color32::from_black_alpha(70),
    );

    let mut bg_color = if is_selected {
        ui.visuals().selection.bg_fill
    } else {
        ui.visuals().widgets.inactive.bg_fill
    };
    bg_color = bg_color.gamma_multiply(1.05);
    let bg_color = Color32::from_rgba_unmultiplied(bg_color.r(), bg_color.g(), bg_color.b(), 170);

    let stack_layers = selection_count.clamp(1, 5);
    if stack_layers > 1 {
        let max_offset = 10.0;
        let step = max_offset / (stack_layers as f32 - 1.0);
        for layer in (1..stack_layers).rev() {
            let offset = Vec2::new(step * layer as f32, step * layer as f32);
            let layer_rect = rect.translate(offset);
            let layer_color = Color32::from_rgba_unmultiplied(
                bg_color.r(),
                bg_color.g(),
                bg_color.b(),
                (140u8).saturating_sub(layer as u8 * 12),
            );
            painter.rect_filled(layer_rect, 8.0, layer_color);
        }
    }

    painter.rect_filled(rect, 8.0, bg_color);

    let content_rect = rect.shrink(8.0);
    let thumb = (drag_size.x * 0.55).clamp(32.0, drag_size.x - 16.0);
    let thumb_rect = Rect::from_center_size(
        Pos2::new(content_rect.center().x, content_rect.top() + thumb * 0.5),
        Vec2::new(thumb, thumb),
    );
    let thumb_color = asset_thumbnail_color(entry);
    painter.rect_filled(thumb_rect, 6.0, thumb_color);
    painter.text(
        thumb_rect.center(),
        Align2::CENTER_CENTER,
        asset_thumbnail_tag(entry),
        FontId::proportional(12.0),
        Color32::WHITE,
    );

    let name = asset_display_name(&entry.path);
    painter.text(
        Pos2::new(content_rect.center().x, content_rect.bottom() - 4.0),
        Align2::CENTER_BOTTOM,
        name,
        FontId::proportional(12.0),
        ui.visuals().text_color(),
    );
}

fn paint_selection_rect(ui: &Ui, rect: Rect) {
    let fill = ui.visuals().selection.bg_fill.gamma_multiply(0.35);
    let stroke = ui.visuals().selection.stroke;
    ui.painter().rect_filled(rect, 2.0, fill);
    ui.painter()
        .rect_stroke(rect, 2.0, stroke, StrokeKind::Inside);
}

fn asset_display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unknown>")
        .to_string()
}

fn asset_path_label(root: &Path, path: &Path) -> String {
    let root_label = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Project");

    match path.strip_prefix(root) {
        Ok(relative) => {
            if relative.as_os_str().is_empty() {
                root_label.to_string()
            } else {
                format!(
                    "{}/{}",
                    root_label,
                    relative.to_string_lossy().replace('\\', "/")
                )
            }
        }
        Err(_) => path.to_string_lossy().replace('\\', "/"),
    }
}

fn asset_breadcrumb_dirs(root: &Path, current: &Path) -> Vec<PathBuf> {
    if let Ok(relative) = current.strip_prefix(root) {
        let mut path = root.to_path_buf();
        let mut dirs = vec![path.clone()];
        for component in relative.components() {
            path = path.join(component);
            dirs.push(path.clone());
        }
        dirs
    } else {
        vec![current.to_path_buf()]
    }
}

fn asset_tag_for(entry: &AssetEntry) -> &'static str {
    if entry.is_dir {
        return "[DIR]";
    }

    if is_script_file(&entry.path) {
        return if script_language_from_path(&entry.path) == "rust" {
            "[RSCRIPT]"
        } else {
            "[SCRIPT]"
        };
    }

    let Some(ext) = entry.path.extension().and_then(|ext| ext.to_str()) else {
        return "[FILE]";
    };

    match ext.to_ascii_lowercase().as_str() {
        "ron" => {
            let name = entry
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if name.eq_ignore_ascii_case("helmer_project.ron") {
                "[CFG]"
            } else if name.ends_with(".hanim.ron") {
                "[ANIM]"
            } else if name.ends_with(".hscene.ron") {
                "[SCENE]"
            } else {
                "[MAT]"
            }
        }
        "glb" | "gltf" => "[MODEL]",
        "ktx2" | "png" | "jpg" | "jpeg" | "tga" => "[TEX]",
        _ => "[FILE]",
    }
}

fn asset_thumbnail_tag(entry: &AssetEntry) -> &'static str {
    match asset_tag_for(entry) {
        "[DIR]" => "DIR",
        "[SCENE]" => "SCN",
        "[MAT]" => "MAT",
        "[ANIM]" => "ANIM",
        "[SCRIPT]" => "LUA",
        "[RSCRIPT]" => "RUST",
        "[MODEL]" => "MOD",
        "[TEX]" => "TEX",
        "[CFG]" => "CFG",
        _ => "FILE",
    }
}

fn asset_thumbnail_color(entry: &AssetEntry) -> Color32 {
    if entry.is_dir {
        return Color32::from_rgb(60, 92, 120);
    }

    let Some(ext) = entry.path.extension().and_then(|ext| ext.to_str()) else {
        return Color32::from_rgb(80, 80, 80);
    };

    if is_script_file(&entry.path) {
        return if script_language_from_path(&entry.path) == "rust" {
            Color32::from_rgb(165, 92, 48)
        } else {
            Color32::from_rgb(60, 110, 70)
        };
    }

    match ext.to_ascii_lowercase().as_str() {
        "ron" => {
            let name = entry
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if name.ends_with(".hanim.ron") {
                Color32::from_rgb(70, 110, 150)
            } else {
                Color32::from_rgb(120, 90, 60)
            }
        }
        "glb" | "gltf" => Color32::from_rgb(90, 70, 130),
        "ktx2" | "png" | "jpg" | "jpeg" | "tga" => Color32::from_rgb(120, 80, 80),
        _ => Color32::from_rgb(70, 70, 90),
    }
}

fn asset_dir_menu(world: &mut World, ui: &mut Ui, path: &Path) {
    if ui.button("Open").clicked() {
        set_current_dir(world, path.to_path_buf());
        ui.close_menu();
    }
    if ui.button("Open in File Browser").clicked() {
        open_in_file_browser(world, path);
        ui.close_menu();
    }
    asset_create_menu(world, ui, path);
    ui.separator();
    if ui.button("Duplicate").clicked() {
        if let Some(selection) = selected_assets_for_action(world, Some(path)) {
            duplicate_assets(world, &selection);
        }
        ui.close_menu();
    }
    if ui.button("Rename").clicked() {
        begin_asset_rename(world, path);
        ui.close_menu();
    }
    if ui.button("Delete").clicked() {
        if let Some(selection) = selected_assets_for_action(world, Some(path)) {
            delete_assets(world, &selection);
        }
        ui.close_menu();
    }
}

fn asset_file_menu(world: &mut World, ui: &mut Ui, path: &Path) {
    if is_scene_file(path) && ui.button("Open Scene").clicked() {
        push_command(
            world,
            EditorCommand::OpenScene {
                path: path.to_path_buf(),
            },
        );
        ui.close_menu();
    }

    if is_model_file(path) && ui.button("Add to Scene").clicked() {
        push_command(
            world,
            EditorCommand::CreateEntity {
                kind: SpawnKind::SceneAsset(path.to_path_buf()),
            },
        );
        ui.close_menu();
    }

    if is_material_file(path) && ui.button("Open in Editor").clicked() {
        open_material_editor_tab(world, path.to_path_buf());
        ui.close_menu();
    }

    if is_animation_file(path) && ui.button("Open in Editor").clicked() {
        open_in_external_editor(world, path);
        ui.close_menu();
    }

    if is_script_file(path) && ui.button("Open in Editor").clicked() {
        open_in_external_editor(world, path);
        ui.close_menu();
    }

    if ui.button("Open in File Browser").clicked() {
        open_in_file_browser(world, path);
        ui.close_menu();
    }

    if ui.button("Duplicate").clicked() {
        if let Some(selection) = selected_assets_for_action(world, Some(path)) {
            duplicate_assets(world, &selection);
        }
        ui.close_menu();
    }

    if ui.button("Rename").clicked() {
        begin_asset_rename(world, path);
        ui.close_menu();
    }
    if ui.button("Delete").clicked() {
        if let Some(selection) = selected_assets_for_action(world, Some(path)) {
            delete_assets(world, &selection);
        }
        ui.close_menu();
    }
}

fn asset_create_menu(world: &mut World, ui: &mut Ui, path: &Path) {
    if ui.button("New Folder").clicked() {
        push_command(
            world,
            EditorCommand::CreateAsset {
                directory: path.to_path_buf(),
                name: "new_folder".to_string(),
                kind: AssetCreateKind::Folder,
            },
        );
        ui.close_menu();
    }
    if ui.button("New Scene").clicked() {
        push_command(
            world,
            EditorCommand::CreateAsset {
                directory: path.to_path_buf(),
                name: "new_scene".to_string(),
                kind: AssetCreateKind::Scene,
            },
        );
        ui.close_menu();
    }
    if ui.button("New Material").clicked() {
        push_command(
            world,
            EditorCommand::CreateAsset {
                directory: path.to_path_buf(),
                name: "new_material".to_string(),
                kind: AssetCreateKind::Material,
            },
        );
        ui.close_menu();
    }
    if ui.button("New Script").clicked() {
        push_command(
            world,
            EditorCommand::CreateAsset {
                directory: path.to_path_buf(),
                name: "new_script".to_string(),
                kind: AssetCreateKind::Script,
            },
        );
        ui.close_menu();
    }
    if ui.button("New Rust Script").clicked() {
        push_command(
            world,
            EditorCommand::CreateAsset {
                directory: path.to_path_buf(),
                name: "new_rust_script".to_string(),
                kind: AssetCreateKind::RustScript,
            },
        );
        ui.close_menu();
    }
    if ui.button("New Animation").clicked() {
        push_command(
            world,
            EditorCommand::CreateAsset {
                directory: path.to_path_buf(),
                name: "new_animation".to_string(),
                kind: AssetCreateKind::Animation,
            },
        );
        ui.close_menu();
    }
}

fn open_in_file_browser(world: &mut World, path: &Path) {
    if let Err(err) = open_in_file_browser_inner(path) {
        set_status(world, format!("Open in file browser failed: {}", err));
    }
}

fn open_in_file_browser_inner(path: &Path) -> Result<(), String> {
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
        Err("Unsupported platform".to_string())
    }
}

fn open_in_external_editor(world: &mut World, path: &Path) {
    if try_open_in_vscode(path) {
        return;
    }
    if !open_with_default_app(path) {
        set_status(world, format!("Unable to open {}", path.display()));
    }
}

fn try_open_in_vscode(path: &Path) -> bool {
    Command::new("code").arg(path).spawn().is_ok()
}

fn open_with_default_app(path: &Path) -> bool {
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(path)
            .spawn()
            .is_ok()
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(path).spawn().is_ok()
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open").arg(path).spawn().is_ok()
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

fn on_asset_double_click(world: &mut World, entry: &AssetEntry) {
    if entry.is_dir {
        set_current_dir(world, entry.path.clone());
        toggle_expand(world, entry.path.clone());
    } else if is_scene_file(&entry.path) {
        push_command(
            world,
            EditorCommand::OpenScene {
                path: entry.path.clone(),
            },
        );
    } else if is_model_file(&entry.path) {
        push_command(
            world,
            EditorCommand::CreateEntity {
                kind: SpawnKind::SceneAsset(entry.path.clone()),
            },
        );
    } else if is_material_file(&entry.path) {
        open_material_editor_tab(world, entry.path.clone());
    } else if is_animation_file(&entry.path) {
        open_in_external_editor(world, &entry.path);
    } else if is_script_file(&entry.path) {
        open_in_external_editor(world, &entry.path);
    }
}

fn open_material_editor_tab(world: &mut World, path: PathBuf) {
    world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
        for window in workspace.windows.iter_mut() {
            if let Some((index, _)) = window
                .tabs
                .iter()
                .enumerate()
                .find(|(_, tab)| matches!(&tab.content, EditorTabContent::Material { path: tab_path } if tab_path == &path))
            {
                window.active = index;
                workspace.last_focused_window = Some(window.id);
                return;
            }
        }

        let title = asset_display_name(&path);
        let tab = EditorTab {
            id: workspace.next_tab_id,
            title,
            content: EditorTabContent::Material { path },
        };
        workspace.next_tab_id += 1;

        let target_window_id = workspace.windows.last().map(|window| window.id);
        if let Some(target_window_id) = target_window_id {
            if let Some(window) = workspace
                .windows
                .iter_mut()
                .find(|window| window.id == target_window_id)
            {
                window.tabs.push(tab);
                window.active = window.tabs.len().saturating_sub(1);
                workspace.last_focused_window = Some(window.id);
                return;
            }
        }

        let window_id = workspace.next_window_id;
        workspace.next_window_id += 1;
        workspace.windows.push(EditorTabWindow {
            id: window_id,
            title: format!("Editor {}", window_id),
            tabs: vec![tab],
            active: 0,
        });
        refresh_editor_window_titles(&mut workspace);
        workspace.last_focused_window = Some(window_id);
    });
}

#[derive(Clone)]
struct TabDragPayload;

fn begin_tab_drag(world: &mut World, window_id: u64, tab_index: usize) {
    world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
        let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| window.id == window_id)
        else {
            return;
        };

        let (tab, source_was_single_tab) = {
            let window = &mut workspace.windows[window_index];
            if tab_index >= window.tabs.len() {
                return;
            }
            if window.tabs.len() == 1 {
                (window.tabs[tab_index].clone(), true)
            } else {
                let tab = window.tabs.remove(tab_index);
                if window.active >= window.tabs.len() {
                    window.active = window.tabs.len().saturating_sub(1);
                }
                (tab, false)
            }
        };

        workspace.dragging = Some(EditorTabDrag {
            tab,
            source_window_id: window_id,
            source_was_single_tab,
        });
        workspace.drop_handled = false;
    });
}

fn detach_tab_to_new_window(world: &mut World, window_id: u64, tab_index: usize) {
    world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
        let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| window.id == window_id)
        else {
            return;
        };

        let tab = {
            let window = &mut workspace.windows[window_index];
            if tab_index >= window.tabs.len() {
                return;
            }
            let tab = window.tabs.remove(tab_index);
            if window.active >= window.tabs.len() {
                window.active = window.tabs.len().saturating_sub(1);
            }
            tab
        };

        let new_window_id = workspace.next_window_id;
        workspace.next_window_id += 1;
        workspace.windows.push(EditorTabWindow {
            id: new_window_id,
            title: format!("Editor {}", new_window_id),
            tabs: vec![tab],
            active: 0,
        });
        refresh_editor_window_titles(&mut workspace);
        workspace.last_focused_window = Some(window_id);
    });
}

fn close_tab_in_window(world: &mut World, window_id: u64, tab_index: usize) {
    world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
        let Some(window) = workspace
            .windows
            .iter_mut()
            .find(|window| window.id == window_id)
        else {
            return;
        };
        if tab_index < window.tabs.len() {
            window.tabs.remove(tab_index);
        }

        if window.active >= window.tabs.len() {
            window.active = window.tabs.len().saturating_sub(1);
        }
    });
}

pub(crate) fn close_editor_window(world: &mut World, window_id: u64) {
    world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
        if workspace
            .dragging
            .as_ref()
            .map(|dragging| dragging.source_window_id == window_id)
            .unwrap_or(false)
        {
            workspace.dragging = None;
            workspace.drop_handled = false;
        }

        if let Some(index) = workspace
            .windows
            .iter()
            .position(|window| window.id == window_id)
        {
            workspace.windows.remove(index);
            refresh_editor_window_titles(&mut workspace);
        }

        workspace.last_focused_window = workspace.windows.last().map(|window| window.id);
    });
}

fn accept_tab_drop(world: &mut World, target_window_id: u64, insert_index: Option<usize>) {
    world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
        let Some(dragging) = workspace.dragging.take() else {
            return;
        };

        if dragging.source_was_single_tab && dragging.source_window_id == target_window_id {
            workspace.last_focused_window = Some(target_window_id);
            workspace.drop_handled = true;
            return;
        }

        if dragging.source_was_single_tab {
            remove_tab_from_window_by_id(
                &mut workspace,
                dragging.source_window_id,
                dragging.tab.id,
            );
        }

        let mut placed_window_id = target_window_id;
        if let Some(window) = workspace
            .windows
            .iter_mut()
            .find(|window| window.id == target_window_id)
        {
            let index = insert_index
                .unwrap_or(window.tabs.len())
                .min(window.tabs.len());
            window.tabs.insert(index, dragging.tab);
            window.active = index;
        } else {
            let new_window_id = workspace.next_window_id;
            workspace.next_window_id += 1;
            workspace.windows.push(EditorTabWindow {
                id: new_window_id,
                title: format!("Editor {}", new_window_id),
                tabs: vec![dragging.tab],
                active: 0,
            });
            placed_window_id = new_window_id;
        }

        workspace.last_focused_window = Some(placed_window_id);
        workspace.drop_handled = true;

        if let Some(index) = workspace
            .windows
            .iter()
            .position(|window| window.id == dragging.source_window_id)
        {
            if workspace.windows[index].tabs.is_empty() {
                workspace.windows.remove(index);
            }
        }
        refresh_editor_window_titles(&mut workspace);
    });
}

fn drop_tab_into_new_window(world: &mut World) {
    world.resource_scope::<EditorWorkspaceState, _>(|_world, mut workspace| {
        let Some(dragging) = workspace.dragging.take() else {
            return;
        };

        if dragging.source_was_single_tab {
            remove_tab_from_window_by_id(
                &mut workspace,
                dragging.source_window_id,
                dragging.tab.id,
            );
        }

        let new_window_id = workspace.next_window_id;
        workspace.next_window_id += 1;
        workspace.windows.push(EditorTabWindow {
            id: new_window_id,
            title: format!("Editor {}", new_window_id),
            tabs: vec![dragging.tab],
            active: 0,
        });
        workspace.last_focused_window = Some(new_window_id);
        workspace.drop_handled = true;

        if let Some(index) = workspace
            .windows
            .iter()
            .position(|window| window.id == dragging.source_window_id)
        {
            if workspace.windows[index].tabs.is_empty() {
                workspace.windows.remove(index);
            }
        }
        refresh_editor_window_titles(&mut workspace);
    });
}

fn refresh_editor_window_titles(workspace: &mut EditorWorkspaceState) {
    for (index, window) in workspace.windows.iter_mut().enumerate() {
        window.title = format!("Editor {}", index + 1);
    }
}

fn remove_tab_from_window_by_id(workspace: &mut EditorWorkspaceState, window_id: u64, tab_id: u64) {
    let Some(window) = workspace
        .windows
        .iter_mut()
        .find(|window| window.id == window_id)
    else {
        return;
    };
    if let Some(index) = window.tabs.iter().position(|tab| tab.id == tab_id) {
        window.tabs.remove(index);
        if window.active >= window.tabs.len() {
            window.active = window.tabs.len().saturating_sub(1);
        }
    }
    if window.tabs.is_empty() {
        if let Some(window_index) = workspace
            .windows
            .iter()
            .position(|window| window.id == window_id)
        {
            workspace.windows.remove(window_index);
        }
    }
}

fn asset_rename_editor(ui: &mut Ui, world: &mut World, path: &Path) {
    let mut finalize: Option<String> = None;
    let mut cancel = false;

    world.resource_scope::<AssetBrowserState, _>(|_world, mut state| {
        let response = ui.text_edit_singleline(&mut state.rename_buffer);
        if response.lost_focus() || ui.input(|input| input.key_pressed(egui::Key::Enter)) {
            finalize = Some(state.rename_buffer.trim().to_string());
        }
        if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
            cancel = true;
        }
    });

    if cancel {
        clear_asset_rename(world);
        return;
    }

    if let Some(name) = finalize {
        apply_asset_rename(world, path, &name);
    }
}

fn begin_asset_rename(world: &mut World, path: &Path) {
    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
        state.rename_path = Some(path.to_path_buf());
        state.rename_buffer = asset_display_name(path);
    }
}

fn clear_asset_rename(world: &mut World) {
    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
        state.rename_path = None;
        state.rename_buffer.clear();
    }
}

fn apply_asset_rename(world: &mut World, path: &Path, new_name: &str) {
    let new_name = new_name.trim();
    if new_name.is_empty() {
        clear_asset_rename(world);
        return;
    }

    let Some(parent) = path.parent() else {
        clear_asset_rename(world);
        return;
    };

    let target_path = parent.join(new_name);
    if target_path == path {
        clear_asset_rename(world);
        return;
    }

    if let Some(state) = world.get_resource::<AssetBrowserState>() {
        if state.root.as_deref() == Some(path) {
            set_status(world, "Cannot rename project root".to_string());
            clear_asset_rename(world);
            return;
        }
    }

    if target_path.exists() {
        set_status(world, "An item with that name already exists".to_string());
        clear_asset_rename(world);
        return;
    }

    match fs::rename(path, &target_path) {
        Ok(()) => {
            remap_asset_state_paths(world, path, &target_path);
            if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
                state.refresh_requested = true;
            }
            set_status(world, format!("Renamed to {}", target_path.display()));
        }
        Err(err) => {
            set_status(world, format!("Rename failed: {}", err));
        }
    }

    clear_asset_rename(world);
}

fn delete_asset(world: &mut World, path: &Path) {
    if let Some(state) = world.get_resource::<AssetBrowserState>() {
        if state.root.as_deref() == Some(path) {
            set_status(world, "Cannot delete project root".to_string());
            return;
        }
    }

    let result = if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };

    match result {
        Ok(()) => {
            if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
                if state.selected.as_deref() == Some(path) {
                    state.selected = None;
                }
                state.selected_paths.remove(path);
                if state.selection_anchor.as_deref() == Some(path) {
                    state.selection_anchor = None;
                }
                if state.current_dir.as_deref() == Some(path) {
                    state.current_dir = state.root.clone();
                }
                state.refresh_requested = true;
            }
            set_status(world, format!("Deleted {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Delete failed: {}", err));
        }
    }
}

fn delete_assets(world: &mut World, paths: &[PathBuf]) {
    let mut any_failed = false;
    for path in paths {
        if let Some(state) = world.get_resource::<AssetBrowserState>() {
            if state.root.as_deref() == Some(path.as_path()) {
                set_status(world, "Cannot delete project root".to_string());
                return;
            }
        }
        let result = if path.is_dir() {
            fs::remove_dir_all(path)
        } else {
            fs::remove_file(path)
        };
        if result.is_err() {
            any_failed = true;
        }
    }

    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
        for path in paths {
            if state.selected.as_deref() == Some(path) {
                state.selected = None;
            }
            state.selected_paths.remove(path);
            if state.selection_anchor.as_deref() == Some(path) {
                state.selection_anchor = None;
            }
            if state.current_dir.as_deref() == Some(path) {
                state.current_dir = state.root.clone();
            }
        }
        state.refresh_requested = true;
    }

    if any_failed {
        set_status(world, "Delete failed for one or more items".to_string());
    } else {
        set_status(world, format!("Deleted {} item(s)", paths.len()));
    }
}

fn move_asset(world: &mut World, source: &Path, target_dir: &Path) {
    if source == target_dir {
        return;
    }

    if let Some(state) = world.get_resource::<AssetBrowserState>() {
        if state.root.as_deref() == Some(source) {
            set_status(world, "Cannot move project root".to_string());
            return;
        }
    }

    if source.is_dir() && target_dir.starts_with(source) {
        set_status(world, "Cannot move a folder into itself".to_string());
        return;
    }

    let Some(file_name) = source.file_name() else {
        return;
    };

    let mut target_path = target_dir.join(file_name);
    target_path = unique_path(&target_path);

    match fs::rename(source, &target_path) {
        Ok(()) => {
            remap_asset_state_paths(world, source, &target_path);
            if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
                state.refresh_requested = true;
            }
            set_status(world, format!("Moved to {}", target_path.display()));
        }
        Err(err) => {
            set_status(world, format!("Move failed: {}", err));
        }
    }
}

fn move_assets(world: &mut World, sources: &[PathBuf], target_dir: &Path) {
    for source in sources {
        move_asset(world, source, target_dir);
    }
}

fn duplicate_assets(world: &mut World, paths: &[PathBuf]) {
    let mut any_failed = false;
    for path in paths {
        if duplicate_asset(world, path).is_err() {
            any_failed = true;
        }
    }
    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
        state.refresh_requested = true;
    }
    if any_failed {
        set_status(world, "Duplicate failed for one or more items".to_string());
    } else {
        set_status(world, format!("Duplicated {} item(s)", paths.len()));
    }
}

fn duplicate_asset(world: &mut World, path: &Path) -> Result<(), String> {
    if let Some(state) = world.get_resource::<AssetBrowserState>() {
        if state.root.as_deref() == Some(path) {
            return Err("Cannot duplicate project root".to_string());
        }
    }
    let Some(parent) = path.parent() else {
        return Err("Asset has no parent directory".to_string());
    };
    let file_name = path
        .file_name()
        .ok_or_else(|| "Asset has no name".to_string())?;
    let target_path = unique_path(&parent.join(file_name));
    if path.is_dir() {
        copy_dir_recursive(path, &target_path)
    } else {
        fs::copy(path, &target_path)
            .map(|_| ())
            .map_err(|err| format!("Duplicate failed: {}", err))
    }
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<(), String> {
    for entry in WalkDir::new(source)
        .min_depth(0)
        .into_iter()
        .filter_map(Result::ok)
    {
        let relative = entry
            .path()
            .strip_prefix(source)
            .map_err(|_| "Failed to copy directory".to_string())?;
        let target_path = target.join(relative);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&target_path).map_err(|err| format!("Duplicate failed: {}", err))?;
        } else {
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent).map_err(|err| format!("Duplicate failed: {}", err))?;
            }
            fs::copy(entry.path(), &target_path)
                .map(|_| ())
                .map_err(|err| format!("Duplicate failed: {}", err))?;
        }
    }
    Ok(())
}

fn remap_asset_state_paths(world: &mut World, from: &Path, to: &Path) {
    let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() else {
        return;
    };

    state.selected = state
        .selected
        .clone()
        .map(|path| remap_asset_path(&path, from, to).unwrap_or(path));

    let mut remapped_selected = HashSet::new();
    for path in state.selected_paths.iter() {
        remapped_selected.insert(remap_asset_path(path, from, to).unwrap_or_else(|| path.clone()));
    }
    state.selected_paths = remapped_selected;
    state.selection_anchor = state
        .selection_anchor
        .clone()
        .and_then(|path| remap_asset_path(&path, from, to));

    state.current_dir = state
        .current_dir
        .clone()
        .map(|path| remap_asset_path(&path, from, to).unwrap_or(path));

    let mut remapped = HashSet::new();
    for path in state.expanded.iter() {
        remapped.insert(remap_asset_path(path, from, to).unwrap_or_else(|| path.clone()));
    }
    state.expanded = remapped;
}

fn remap_asset_path(path: &Path, from: &Path, to: &Path) -> Option<PathBuf> {
    if path == from {
        return Some(to.to_path_buf());
    }
    let relative = path.strip_prefix(from).ok()?;
    Some(to.join(relative))
}

fn begin_rename(world: &mut World, entity: Entity) {
    world.resource_scope::<HierarchyUiState, _>(|world, mut ui_state| {
        ui_state.rename_entity = Some(entity);
        ui_state.rename_buffer = world
            .get::<Name>(entity)
            .map(|name| name.to_string())
            .unwrap_or_else(|| format!("Entity {}", entity.to_bits()));
        ui_state.rename_request_focus = true;
    });
    set_selection(world, Some(entity));
}

fn apply_entity_name(world: &mut World, entity: Entity, name: &str) {
    let trimmed = name.trim();
    let current = world
        .get::<Name>(entity)
        .map(|name| name.to_string())
        .unwrap_or_default();
    if trimmed == current {
        return;
    }
    if trimmed.is_empty() {
        world.entity_mut(entity).remove::<Name>();
    } else {
        world
            .entity_mut(entity)
            .insert(Name::new(trimmed.to_string()));
    }
    if let Some(mut name_state) = world.get_resource_mut::<InspectorNameEditState>() {
        if name_state.entity == Some(entity) {
            name_state.buffer = trimmed.to_string();
        }
    }
    push_undo_snapshot(world, "Rename");
}

fn entity_display_name(world: &World, entity: Entity) -> String {
    world
        .get::<Name>(entity)
        .map(|name| name.to_string())
        .unwrap_or_else(|| format!("Entity {}", entity.index()))
}

fn draw_material_editor_tab(
    ui: &mut Ui,
    world: &mut World,
    project: &Option<EditorProject>,
    path: &Path,
) {
    let path = path.to_path_buf();
    world.resource_scope::<MaterialEditorCache, _>(|world, mut cache| {
        let entry = cache
            .entries
            .entry(path.clone())
            .or_insert_with(MaterialEditorEntry::default);
        let mut reload_requested = false;

        ui.horizontal(|ui| {
            ui.heading("Material");
            if ui.button("Reload").clicked() {
                reload_requested = true;
            }
        });

        if reload_requested || entry.data.is_none() {
            match load_material_file(&path) {
                Ok(data) => {
                    entry.data = Some(data);
                    entry.error = None;
                }
                Err(err) => {
                    entry.data = None;
                    entry.error = Some(err);
                }
            }
        }

        let path_label = project_relative_path(project, &path);
        ui.horizontal(|ui| {
            ui.label("File:");
            ui.add(egui::Label::new(path_label).wrap_mode(egui::TextWrapMode::Extend));
        });

        if let Some(error) = entry.error.as_ref() {
            ui.label(RichText::new(error).small());
        }

        let Some(data) = entry.data.as_mut() else {
            return;
        };

        let mut changed = false;
        let mut edit_response = EditResponse::default();
        let undo_label = format!("Material {}", project_relative_path(project, &path));

        ui.separator();
        ui.label("Surface");
        let albedo_response = ui.horizontal(|ui| {
            ui.label("Albedo");
            let mut albedo = data.albedo;
            let response = ui.color_edit_button_rgba_unmultiplied(&mut albedo);
            if response.changed() {
                data.albedo = albedo;
                changed = true;
            }
            response
        });
        let mut albedo_response = EditResponse::from_response(&albedo_response.inner);
        if albedo_response.changed {
            albedo_response.lost_focus = true;
        }
        edit_response.merge(albedo_response);

        let metallic_response =
            edit_float_range(ui, "Metallic", &mut data.metallic, 0.01, 0.0..=1.0);
        changed |= metallic_response.changed;
        edit_response.merge(metallic_response);

        let roughness_response =
            edit_float_range(ui, "Roughness", &mut data.roughness, 0.01, 0.0..=1.0);
        changed |= roughness_response.changed;
        edit_response.merge(roughness_response);

        let ao_response = edit_float_range(ui, "AO", &mut data.ao, 0.01, 0.0..=1.0);
        changed |= ao_response.changed;
        edit_response.merge(ao_response);

        ui.separator();
        ui.label("Emission");
        let emission_color_response = ui.horizontal(|ui| {
            ui.label("Color");
            let mut color = data.emission_color;
            let response = ui.color_edit_button_rgb(&mut color);
            if response.changed() {
                data.emission_color = color;
                changed = true;
            }
            response
        });
        let mut emission_color_response =
            EditResponse::from_response(&emission_color_response.inner);
        if emission_color_response.changed {
            emission_color_response.lost_focus = true;
        }
        edit_response.merge(emission_color_response);

        let emission_strength_response = edit_float_range(
            ui,
            "Strength",
            &mut data.emission_strength,
            0.05,
            0.0..=100.0,
        );
        changed |= emission_strength_response.changed;
        edit_response.merge(emission_strength_response);

        ui.separator();
        ui.label("Textures");
        let albedo_texture_response = edit_material_texture(ui, "Albedo", &mut data.albedo_texture);
        changed |= albedo_texture_response.changed;
        edit_response.merge(albedo_texture_response);

        let normal_texture_response = edit_material_texture(ui, "Normal", &mut data.normal_texture);
        changed |= normal_texture_response.changed;
        edit_response.merge(normal_texture_response);

        let metallic_texture_response = edit_material_texture(
            ui,
            "Metallic/Roughness",
            &mut data.metallic_roughness_texture,
        );
        changed |= metallic_texture_response.changed;
        edit_response.merge(metallic_texture_response);

        let emission_texture_response =
            edit_material_texture(ui, "Emission", &mut data.emission_texture);
        changed |= emission_texture_response.changed;
        edit_response.merge(emission_texture_response);

        begin_material_edit_undo(world, &path, &undo_label, edit_response);

        if changed {
            match save_material_file(&path, data) {
                Ok(()) => {
                    entry.error = None;
                    set_status(world, format!("Saved material {}", path.display()));
                    refresh_material_usage(world, project, &path);
                }
                Err(err) => {
                    entry.error = Some(err.clone());
                    set_status(world, format!("Failed to save material: {}", err));
                }
            }
        }

        end_material_edit_undo(world, edit_response);
    });
}

fn default_collider_shape_for_entity(world: &World, entity: Entity) -> ColliderShape {
    if let Some(renderer) = world.get::<BevyMeshRenderer>(entity) {
        return ColliderShape::Mesh {
            mesh_id: Some(renderer.0.mesh_id),
            lod: MeshColliderLod::Lowest,
            kind: MeshColliderKind::TriMesh,
        };
    }

    if let Some(renderer) = world.get::<BevySkinnedMeshRenderer>(entity) {
        return ColliderShape::Mesh {
            mesh_id: Some(renderer.0.mesh_id),
            lod: MeshColliderLod::Lowest,
            kind: MeshColliderKind::TriMesh,
        };
    }

    ColliderShape::Cuboid
}

fn physics_world_defaults_for_world(world: &mut World) -> PhysicsWorldDefaults {
    let mut selected: Option<(u64, PhysicsWorldDefaults)> = None;
    let mut query = world.query::<(Entity, &PhysicsWorldDefaults)>();
    for (entity, defaults) in query.iter(world) {
        let key = entity.to_bits();
        let should_replace = selected
            .as_ref()
            .map(|(selected_key, _)| key < *selected_key)
            .unwrap_or(true);
        if should_replace {
            selected = Some((key, *defaults));
        }
    }
    if let Some((_, defaults)) = selected {
        return defaults;
    }

    world
        .get_resource::<PhysicsResource>()
        .map(|phys| PhysicsWorldDefaults {
            gravity: phys.gravity,
            collider_properties: phys.default_collider_properties,
            rigid_body_properties: phys.default_rigid_body_properties,
        })
        .unwrap_or_default()
}

fn default_collider_properties_for_world(world: &mut World) -> ColliderProperties {
    physics_world_defaults_for_world(world).collider_properties
}

fn default_rigid_body_properties_for_world(world: &mut World) -> RigidBodyProperties {
    physics_world_defaults_for_world(world).rigid_body_properties
}

fn remove_body_dependent_physics_components(world: &mut World, entity: Entity) {
    let has_any_body = world.get::<DynamicRigidBody>(entity).is_some()
        || world.get::<KinematicRigidBody>(entity).is_some()
        || world.get::<FixedCollider>(entity).is_some();
    if has_any_body {
        return;
    }

    world.entity_mut(entity).remove::<ColliderShape>();
    world.entity_mut(entity).remove::<ColliderProperties>();
    world
        .entity_mut(entity)
        .remove::<ColliderPropertyInheritance>();
    world.entity_mut(entity).remove::<RigidBodyProperties>();
    world
        .entity_mut(entity)
        .remove::<RigidBodyPropertyInheritance>();
    world.entity_mut(entity).remove::<PhysicsHandle>();
}

fn collider_mesh_id_from_asset(
    world: &mut World,
    project: &Option<EditorProject>,
    path: &Path,
) -> Option<usize> {
    if !is_model_file(path) {
        set_status(world, "Collider mesh requires a model asset".to_string());
        return None;
    }

    let cache_key = project_relative_path(project, path);
    let mesh_id = world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = world.get_resource::<BevyAssetServer>()?;
        Some(load_mesh_asset(&cache_key, &mut cache, asset_server, project.as_ref()).id)
    });

    if mesh_id.is_none() {
        set_status(world, "Asset server missing for collider mesh".to_string());
    }

    mesh_id
}

fn ensure_physics_defaults(world: &mut World, entity: Entity) {
    if world.get::<ColliderProperties>(entity).is_none() {
        let defaults = default_collider_properties_for_world(world);
        world.entity_mut(entity).insert(defaults);
    }
    if world.get::<ColliderPropertyInheritance>(entity).is_none() {
        world
            .entity_mut(entity)
            .insert(ColliderPropertyInheritance::default());
    }
    if world.get::<RigidBodyProperties>(entity).is_none() {
        let defaults = default_rigid_body_properties_for_world(world);
        world.entity_mut(entity).insert(defaults);
    }
    if world.get::<RigidBodyPropertyInheritance>(entity).is_none() {
        world
            .entity_mut(entity)
            .insert(RigidBodyPropertyInheritance::default());
    }
}

fn draw_add_component_menu(
    ui: &mut Ui,
    world: &mut World,
    entity: Entity,
    project: &Option<EditorProject>,
    selected_asset: Option<PathBuf>,
) {
    let has_transform = world.get::<BevyTransform>(entity).is_some();
    let has_camera = world.get::<BevyCamera>(entity).is_some();
    let has_light = world.get::<BevyLight>(entity).is_some();
    let has_mesh = world.get::<BevyMeshRenderer>(entity).is_some();
    let has_skinned = world.get::<BevySkinnedMeshRenderer>(entity).is_some()
        || world.get::<EditorSkinnedMesh>(entity).is_some()
        || world.get::<PendingSkinnedMeshAsset>(entity).is_some();
    let has_scene = world.get::<SceneRoot>(entity).is_some();
    let has_dynamic = world.get::<DynamicComponents>(entity).is_some();
    let has_freecam = world.get::<Freecam>(entity).is_some();
    let has_dynamic_body = world.get::<DynamicRigidBody>(entity).is_some();
    let has_fixed_collider = world.get::<FixedCollider>(entity).is_some();
    let has_kinematic_body = world.get::<KinematicRigidBody>(entity).is_some();
    let has_joint = world.get::<PhysicsJoint>(entity).is_some();
    let has_character_controller = world.get::<CharacterController>(entity).is_some();
    let has_ray_cast_query = world.get::<PhysicsRayCast>(entity).is_some();
    let has_point_projection_query = world.get::<PhysicsPointProjection>(entity).is_some();
    let has_shape_cast_query = world.get::<PhysicsShapeCast>(entity).is_some();
    let has_world_defaults = world.get::<PhysicsWorldDefaults>(entity).is_some();
    let any_world_defaults = {
        let mut query = world.query::<&PhysicsWorldDefaults>();
        query.iter(world).next().is_some()
    };
    let has_spline = world.get::<BevySpline>(entity).is_some();
    let has_spline_follower = world.get::<BevySplineFollower>(entity).is_some();
    let has_look_at = world.get::<BevyLookAt>(entity).is_some();
    let has_entity_follower = world.get::<BevyEntityFollower>(entity).is_some();
    let has_audio_emitter = world.get::<BevyAudioEmitter>(entity).is_some();
    let has_audio_listener = world.get::<BevyAudioListener>(entity).is_some();
    let collider_shape = world.get::<ColliderShape>(entity).copied();

    let selected_mesh_source = selected_asset
        .as_ref()
        .filter(|path| is_model_file(path))
        .map(|path| mesh_source_from_path(project, path));
    let selected_material = selected_asset
        .as_ref()
        .filter(|path| is_material_file(path))
        .and_then(|path| material_path_from_project(project, path));
    let selected_audio_asset = selected_asset
        .as_ref()
        .filter(|path| is_audio_file(path))
        .cloned();

    ui.menu_button("Add Component", |ui| {
        if !has_transform && ui.button("Transform").clicked() {
            world.entity_mut(entity).insert(BevyTransform::default());
            push_undo_snapshot(world, "Add Transform");
            ui.close_menu();
        }
        if !has_camera && ui.button("Camera").clicked() {
            ensure_transform(world, entity);
            world.entity_mut(entity).insert(BevyCamera::default());
            push_undo_snapshot(world, "Add Camera");
            ui.close_menu();
        }
        if !has_light {
            ui.menu_button("Light", |ui| {
                if ui.button("Directional").clicked() {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevyWrapper(Light::directional(Vec3::ONE, 25.0)));
                    push_undo_snapshot(world, "Add Light");
                    ui.close_menu();
                }
                if ui.button("Point").clicked() {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevyWrapper(Light::point(Vec3::ONE, 10.0)));
                    push_undo_snapshot(world, "Add Light");
                    ui.close_menu();
                }
                if ui.button("Spot").clicked() {
                    ensure_transform(world, entity);
                    world.entity_mut(entity).insert(BevyWrapper(Light::spot(
                        Vec3::ONE,
                        10.0,
                        45.0_f32.to_radians(),
                    )));
                    push_undo_snapshot(world, "Add Light");
                    ui.close_menu();
                }
            });
        }
        if !has_audio_emitter {
            let audio_button = ui.button("Audio Emitter");
            if audio_button.clicked() {
                ensure_transform(world, entity);
                world
                    .entity_mut(entity)
                    .insert(BevyWrapper(AudioEmitter::default()));
                world.entity_mut(entity).insert(EditorAudio {
                    path: None,
                    streaming: false,
                });
                if let Some(path) = selected_audio_asset.as_ref() {
                    apply_audio_emitter_from_asset(world, entity, project, path, false);
                }
                push_undo_snapshot(world, "Add Audio Emitter");
                ui.close_menu();
            }
            if let Some(payload) = typed_dnd_release_payload::<AssetDragPayload>(&audio_button) {
                if let Some(path) = payload_primary_path(&payload) {
                    if is_audio_file(path) {
                        ensure_transform(world, entity);
                        world
                            .entity_mut(entity)
                            .insert(BevyWrapper(AudioEmitter::default()));
                        apply_audio_emitter_from_asset(world, entity, project, path, false);
                        push_undo_snapshot(world, "Add Audio Emitter");
                    }
                }
                ui.close_menu();
            }
            highlight_drop_target(ui, &audio_button);
        }
        if !has_audio_listener && ui.button("Audio Listener").clicked() {
            ensure_transform(world, entity);
            world
                .entity_mut(entity)
                .insert(BevyWrapper(AudioListener::default()));
            push_undo_snapshot(world, "Add Audio Listener");
            ui.close_menu();
        }
        if !has_mesh && ui.button("Mesh Renderer").clicked() {
            ensure_transform(world, entity);
            let source = selected_mesh_source
                .clone()
                .unwrap_or(MeshSource::Primitive(PrimitiveKind::Cube));
            apply_mesh_renderer(
                world,
                entity,
                project,
                source,
                selected_material.clone(),
                true,
                true,
            );
            push_undo_snapshot(world, "Add Mesh");
            ui.close_menu();
        }
        if !has_skinned {
            let skinned_button = ui.button("Skinned Mesh Renderer");
            if skinned_button.clicked() {
                ensure_transform(world, entity);
                if world.get::<EditorSkinnedMesh>(entity).is_none() {
                    world.entity_mut(entity).insert(EditorSkinnedMesh {
                        scene_path: None,
                        node_index: None,
                        casts_shadow: true,
                        visible: true,
                    });
                }
                if let Some(path) = selected_asset.as_ref().filter(|path| is_model_file(path)) {
                    apply_skinned_mesh_renderer_from_asset(world, entity, project, path);
                } else {
                    set_status(
                        world,
                        "Skinned mesh component added. Assign a glb/gltf asset in the Inspector."
                            .to_string(),
                    );
                }
                push_undo_snapshot(world, "Add Skinned Mesh");
                ui.close_menu();
            }
            if let Some(payload) = typed_dnd_release_payload::<AssetDragPayload>(&skinned_button) {
                if let Some(path) = payload_primary_path(&payload) {
                    if is_model_file(path) {
                        ensure_transform(world, entity);
                        if world.get::<EditorSkinnedMesh>(entity).is_none() {
                            world.entity_mut(entity).insert(EditorSkinnedMesh {
                                scene_path: None,
                                node_index: None,
                                casts_shadow: true,
                                visible: true,
                            });
                        }
                        apply_skinned_mesh_renderer_from_asset(world, entity, project, path);
                        push_undo_snapshot(world, "Add Skinned Mesh");
                    }
                }
                ui.close_menu();
            }
            highlight_drop_target(ui, &skinned_button);
        }
        if !has_spline && ui.button("Spline").clicked() {
            ensure_transform(world, entity);
            world
                .entity_mut(entity)
                .insert(BevySpline(Spline::default()));
            push_undo_snapshot(world, "Add Spline");
            ui.close_menu();
        }
        if !has_spline_follower && ui.button("Spline Follower").clicked() {
            ensure_transform(world, entity);
            world
                .entity_mut(entity)
                .insert(BevySplineFollower(SplineFollower::default()));
            push_undo_snapshot(world, "Add Spline Follower");
            ui.close_menu();
        }
        if !has_look_at && ui.button("Look At").clicked() {
            ensure_transform(world, entity);
            world
                .entity_mut(entity)
                .insert(BevyLookAt(LookAt::default()));
            push_undo_snapshot(world, "Add Look At");
            ui.close_menu();
        }
        if !has_entity_follower && ui.button("Entity Follower").clicked() {
            ensure_transform(world, entity);
            world
                .entity_mut(entity)
                .insert(BevyEntityFollower(EntityFollower::default()));
            push_undo_snapshot(world, "Add Entity Follower");
            ui.close_menu();
        }
        if !has_scene {
            let scene_asset_button = ui.button("Scene Asset");
            if scene_asset_button.clicked() {
                ensure_scene_asset_placeholder(world, entity);
                if let Some(path) = selected_asset.as_ref().filter(|path| is_model_file(path)) {
                    try_apply_scene_asset_path(world, entity, path);
                }
                push_undo_snapshot(world, "Add Scene Asset");
                ui.close_menu();
            }
            if let Some(payload) =
                typed_dnd_release_payload::<AssetDragPayload>(&scene_asset_button)
            {
                if let Some(path) = payload_primary_path(&payload) {
                    if try_apply_scene_asset_path(world, entity, path) {
                        push_undo_snapshot(world, "Add Scene Asset");
                    }
                }
                ui.close_menu();
            }
            highlight_drop_target(ui, &scene_asset_button);
        }
        if ui.button("Script").clicked() {
            add_script_component(world, entity);
            push_undo_snapshot(world, "Add Script");
            ui.close_menu();
        }
        ui.menu_button("Provided", |ui| {
            if !has_freecam && ui.button("Freecam Controller").clicked() {
                ensure_transform(world, entity);
                if world.get::<BevyCamera>(entity).is_none() {
                    world.entity_mut(entity).insert(BevyCamera::default());
                }
                world.entity_mut(entity).insert(Freecam::default());
                push_undo_snapshot(world, "Add Freecam");
                ui.close_menu();
            }
        });
        ui.menu_button("Physics", |ui| {
            if !has_world_defaults {
                let add_world_defaults_button =
                    ui.add_enabled(!any_world_defaults, egui::Button::new("World Defaults"));
                if add_world_defaults_button.clicked() {
                    let defaults = physics_world_defaults_for_world(world);
                    world.entity_mut(entity).insert(defaults);
                    push_undo_snapshot(world, "Add World Defaults");
                    ui.close_menu();
                }
                if any_world_defaults {
                    ui.label("World Defaults already exists on another entity");
                }
                ui.separator();
            }

            let has_any_body = has_dynamic_body || has_fixed_collider || has_kinematic_body;
            if !has_any_body {
                if ui.button("Dynamic Body").clicked() {
                    ensure_transform(world, entity);
                    let shape = default_collider_shape_for_entity(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(DynamicRigidBody::default())
                        .insert(shape);
                    ensure_physics_defaults(world, entity);
                    push_undo_snapshot(world, "Add Dynamic Body");
                    ui.close_menu();
                }
                if ui.button("Kinematic Body (Position)").clicked() {
                    ensure_transform(world, entity);
                    let shape = default_collider_shape_for_entity(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(KinematicRigidBody {
                            mode: KinematicMode::PositionBased,
                        })
                        .insert(shape);
                    ensure_physics_defaults(world, entity);
                    push_undo_snapshot(world, "Add Kinematic Body");
                    ui.close_menu();
                }
                if ui.button("Kinematic Body (Velocity)").clicked() {
                    ensure_transform(world, entity);
                    let shape = default_collider_shape_for_entity(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(KinematicRigidBody {
                            mode: KinematicMode::VelocityBased,
                        })
                        .insert(shape);
                    ensure_physics_defaults(world, entity);
                    push_undo_snapshot(world, "Add Kinematic Body");
                    ui.close_menu();
                }
                if ui.button("Fixed Collider").clicked() {
                    ensure_transform(world, entity);
                    let shape = default_collider_shape_for_entity(world, entity);
                    world.entity_mut(entity).insert(FixedCollider).insert(shape);
                    ensure_physics_defaults(world, entity);
                    push_undo_snapshot(world, "Add Fixed Collider");
                    ui.close_menu();
                }
            } else {
                ui.label("Body already present");
            }

            ui.separator();
            ui.label("Collider Shape");
            if ui
                .selectable_label(matches!(collider_shape, Some(ColliderShape::Cuboid)), "Box")
                .clicked()
            {
                world.entity_mut(entity).insert(ColliderShape::Cuboid);
                push_undo_snapshot(world, "Collider Shape");
                ui.close_menu();
            }
            if ui
                .selectable_label(
                    matches!(collider_shape, Some(ColliderShape::Sphere)),
                    "Sphere",
                )
                .clicked()
            {
                world.entity_mut(entity).insert(ColliderShape::Sphere);
                push_undo_snapshot(world, "Collider Shape");
                ui.close_menu();
            }
            if ui
                .selectable_label(
                    matches!(collider_shape, Some(ColliderShape::CapsuleY)),
                    "Capsule",
                )
                .clicked()
            {
                world.entity_mut(entity).insert(ColliderShape::CapsuleY);
                push_undo_snapshot(world, "Collider Shape");
                ui.close_menu();
            }
            if ui
                .selectable_label(
                    matches!(collider_shape, Some(ColliderShape::CylinderY)),
                    "Cylinder",
                )
                .clicked()
            {
                world.entity_mut(entity).insert(ColliderShape::CylinderY);
                push_undo_snapshot(world, "Collider Shape");
                ui.close_menu();
            }
            if ui
                .selectable_label(matches!(collider_shape, Some(ColliderShape::ConeY)), "Cone")
                .clicked()
            {
                world.entity_mut(entity).insert(ColliderShape::ConeY);
                push_undo_snapshot(world, "Collider Shape");
                ui.close_menu();
            }
            if ui
                .selectable_label(
                    matches!(collider_shape, Some(ColliderShape::RoundCuboid { .. })),
                    "Round Box",
                )
                .clicked()
            {
                world.entity_mut(entity).insert(ColliderShape::RoundCuboid {
                    border_radius: 0.05,
                });
                push_undo_snapshot(world, "Collider Shape");
                ui.close_menu();
            }
            if ui
                .selectable_label(
                    matches!(
                        collider_shape,
                        Some(ColliderShape::Mesh {
                            kind: MeshColliderKind::TriMesh,
                            ..
                        })
                    ),
                    "Mesh (Lowest LOD)",
                )
                .clicked()
            {
                let mesh_id = world
                    .get::<BevyMeshRenderer>(entity)
                    .map(|renderer| renderer.0.mesh_id)
                    .or_else(|| {
                        world
                            .get::<BevySkinnedMeshRenderer>(entity)
                            .map(|r| r.0.mesh_id)
                    });
                world.entity_mut(entity).insert(ColliderShape::Mesh {
                    mesh_id,
                    lod: MeshColliderLod::Lowest,
                    kind: MeshColliderKind::TriMesh,
                });
                push_undo_snapshot(world, "Collider Shape");
                ui.close_menu();
            }
            if ui
                .selectable_label(
                    matches!(
                        collider_shape,
                        Some(ColliderShape::Mesh {
                            kind: MeshColliderKind::ConvexHull,
                            ..
                        })
                    ),
                    "Mesh Convex Hull",
                )
                .clicked()
            {
                let mesh_id = world
                    .get::<BevyMeshRenderer>(entity)
                    .map(|renderer| renderer.0.mesh_id)
                    .or_else(|| {
                        world
                            .get::<BevySkinnedMeshRenderer>(entity)
                            .map(|r| r.0.mesh_id)
                    });
                world.entity_mut(entity).insert(ColliderShape::Mesh {
                    mesh_id,
                    lod: MeshColliderLod::Lowest,
                    kind: MeshColliderKind::ConvexHull,
                });
                push_undo_snapshot(world, "Collider Shape");
                ui.close_menu();
            }

            ui.separator();
            if !world.get::<ColliderProperties>(entity).is_some()
                && ui.button("Collider Properties").clicked()
            {
                let defaults = default_collider_properties_for_world(world);
                world
                    .entity_mut(entity)
                    .insert(defaults)
                    .insert(ColliderPropertyInheritance::default());
                push_undo_snapshot(world, "Add Collider Properties");
                ui.close_menu();
            }
            if !world.get::<RigidBodyProperties>(entity).is_some()
                && ui.button("Rigid Body Properties").clicked()
            {
                let defaults = default_rigid_body_properties_for_world(world);
                world
                    .entity_mut(entity)
                    .insert(defaults)
                    .insert(RigidBodyPropertyInheritance::default());
                push_undo_snapshot(world, "Add Rigid Body Properties");
                ui.close_menu();
            }
            if !has_joint && ui.button("Joint").clicked() {
                world.entity_mut(entity).insert(PhysicsJoint::default());
                push_undo_snapshot(world, "Add Joint");
                ui.close_menu();
            }
            if !has_character_controller && ui.button("Character Controller").clicked() {
                world
                    .entity_mut(entity)
                    .insert(CharacterController::default())
                    .insert(CharacterControllerInput::default())
                    .insert(CharacterControllerOutput::default());
                push_undo_snapshot(world, "Add Character Controller");
                ui.close_menu();
            }
            if !has_ray_cast_query && ui.button("Ray Cast Query").clicked() {
                world
                    .entity_mut(entity)
                    .insert(PhysicsRayCast::default())
                    .insert(PhysicsRayCastHit::default());
                push_undo_snapshot(world, "Add Ray Cast Query");
                ui.close_menu();
            }
            if !has_point_projection_query && ui.button("Point Projection Query").clicked() {
                world
                    .entity_mut(entity)
                    .insert(PhysicsPointProjection::default())
                    .insert(PhysicsPointProjectionHit::default());
                push_undo_snapshot(world, "Add Point Query");
                ui.close_menu();
            }
            if !has_shape_cast_query && ui.button("Shape Cast Query").clicked() {
                world
                    .entity_mut(entity)
                    .insert(PhysicsShapeCast::default())
                    .insert(PhysicsShapeCastHit::default());
                push_undo_snapshot(world, "Add Shape Cast Query");
                ui.close_menu();
            }
        });
        if !has_dynamic && ui.button("Dynamic Components").clicked() {
            world
                .entity_mut(entity)
                .insert(DynamicComponents::default());
            push_undo_snapshot(world, "Add Dynamic Components");
            ui.close_menu();
        }
    });
}

fn draw_dynamic_components_section(ui: &mut Ui, world: &mut World, entity: Entity) {
    let Some(dynamic_snapshot) = world.get::<DynamicComponents>(entity).cloned() else {
        return;
    };

    let mut dynamic = dynamic_snapshot.clone();
    let mut edit_response = EditResponse::default();
    let mut discrete_changed = false;

    world.resource_scope::<HierarchyUiState, _>(|world, mut ui_state| {
        if dynamic.components.is_empty() {
            ui.label("No dynamic components yet.");
        }

        let mut remove_component_index = None;
        for (component_index, component) in dynamic.components.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label("Component");
                let mut name = component.name.clone();
                let response = ui.text_edit_singleline(&mut name);
                let mut field_response = EditResponse::from_response(&response);
                if response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                    field_response.lost_focus = true;
                }
                begin_edit_undo(world, "Dynamic", field_response);
                if field_response.changed {
                    component.name = name;
                }
                edit_response.merge(field_response);

                if ui.button("Remove").clicked() {
                    remove_component_index = Some(component_index);
                    discrete_changed = true;
                }
            });

            let mut remove_field_index = None;
            for (field_index, field) in component.fields.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label("Field");
                    let mut name = field.name.clone();
                    let response = ui.text_edit_singleline(&mut name);
                    let mut name_response = EditResponse::from_response(&response);
                    if response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter))
                    {
                        name_response.lost_focus = true;
                    }
                    begin_edit_undo(world, "Dynamic", name_response);
                    if name_response.changed {
                        field.name = name;
                    }
                    edit_response.merge(name_response);

                    ui.label(dynamic_value_label(&field.value));
                    let mut value = field.value.clone();
                    let value_response = edit_dynamic_value(ui, &mut value);
                    begin_edit_undo(world, "Dynamic", value_response);
                    if value_response.changed {
                        field.value = value;
                    }
                    edit_response.merge(value_response);

                    if ui.button("Remove").clicked() {
                        remove_field_index = Some(field_index);
                        discrete_changed = true;
                    }
                });
            }

            if let Some(remove_index) = remove_field_index {
                if remove_index < component.fields.len() {
                    component.fields.remove(remove_index);
                }
            }

            let mut add_field = None;
            ui.horizontal(|ui| {
                ui.label("New Field");
                ui.text_edit_singleline(&mut ui_state.new_dynamic_field_name);
                ComboBox::from_id_source(format!(
                    "dyn_field_kind_{}_{}",
                    entity.to_bits(),
                    component_index
                ))
                .selected_text(dynamic_value_kind_label(ui_state.new_dynamic_field_kind))
                .show_ui(ui, |ui| {
                    for kind in [
                        DynamicValueKind::Bool,
                        DynamicValueKind::Float,
                        DynamicValueKind::Int,
                        DynamicValueKind::Vec3,
                        DynamicValueKind::String,
                    ] {
                        if ui
                            .selectable_label(
                                ui_state.new_dynamic_field_kind == kind,
                                dynamic_value_kind_label(kind),
                            )
                            .clicked()
                        {
                            ui_state.new_dynamic_field_kind = kind;
                        }
                    }
                });

                if ui.button("Add").clicked() {
                    let name = ui_state.new_dynamic_field_name.trim();
                    if !name.is_empty() {
                        add_field = Some(DynamicField {
                            name: name.to_string(),
                            value: ui_state.new_dynamic_field_kind.default_value(),
                        });
                        ui_state.new_dynamic_field_name.clear();
                        discrete_changed = true;
                    }
                }
            });

            if let Some(field) = add_field {
                component.fields.push(field);
            }

            ui.separator();
        }

        if let Some(remove_index) = remove_component_index {
            if remove_index < dynamic.components.len() {
                dynamic.components.remove(remove_index);
            }
        }

        ui.horizontal(|ui| {
            ui.label("New Component");
            ui.text_edit_singleline(&mut ui_state.new_dynamic_component_name);
            if ui.button("Add").clicked() {
                let name = ui_state.new_dynamic_component_name.trim();
                if !name.is_empty() {
                    dynamic
                        .components
                        .push(DynamicComponent::new(name.to_string()));
                    ui_state.new_dynamic_component_name.clear();
                    discrete_changed = true;
                }
            }
        });
    });

    if dynamic != dynamic_snapshot {
        world.entity_mut(entity).insert(dynamic);
    }

    end_edit_undo(world, edit_response);
    if discrete_changed {
        push_undo_snapshot(world, "Dynamic");
    }
}

#[derive(Default, Copy, Clone)]
struct EditResponse {
    changed: bool,
    drag_started: bool,
    drag_released: bool,
    lost_focus: bool,
}

impl EditResponse {
    fn from_response(response: &Response) -> Self {
        Self {
            changed: response.changed(),
            drag_started: response.drag_started(),
            drag_released: response.drag_stopped(),
            lost_focus: response.lost_focus(),
        }
    }

    fn merge(&mut self, other: Self) {
        self.changed |= other.changed;
        self.drag_started |= other.drag_started;
        self.drag_released |= other.drag_released;
        self.lost_focus |= other.lost_focus;
    }
}

fn begin_edit_undo(world: &mut World, label: &str, response: EditResponse) {
    if response.drag_started || response.changed {
        begin_undo_group(world, label);
    }
}

fn end_edit_undo(world: &mut World, response: EditResponse) {
    if response.drag_released || response.lost_focus {
        end_undo_group(world);
    }
}

fn begin_material_edit_undo(world: &mut World, path: &Path, label: &str, response: EditResponse) {
    if response.drag_started || response.changed {
        begin_material_undo_group(world, path, label);
    }
}

fn end_material_edit_undo(world: &mut World, response: EditResponse) {
    if response.drag_released || response.lost_focus {
        end_material_undo_group(world);
    }
}

fn edit_float(ui: &mut Ui, label: &str, value: &mut f32, speed: f32) -> EditResponse {
    let mut response = EditResponse::default();
    ui.horizontal(|ui| {
        ui.label(label);
        let drag_response = ui.add(DragValue::new(value).speed(speed));
        response.merge(EditResponse::from_response(&drag_response));
    });
    response
}

fn edit_float_range(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    speed: f32,
    range: std::ops::RangeInclusive<f32>,
) -> EditResponse {
    let mut response = EditResponse::default();
    ui.horizontal(|ui| {
        ui.label(label);
        let drag_response = ui.add(DragValue::new(value).speed(speed).range(range));
        response.merge(EditResponse::from_response(&drag_response));
    });
    response
}

fn edit_u32_range(
    ui: &mut Ui,
    label: &str,
    value: &mut u32,
    speed: f32,
    range: std::ops::RangeInclusive<u32>,
) -> EditResponse {
    let mut response = EditResponse::default();
    ui.horizontal(|ui| {
        ui.label(label);
        let drag_response = ui.add(DragValue::new(value).speed(speed).range(range));
        response.merge(EditResponse::from_response(&drag_response));
    });
    response
}

fn edit_color(ui: &mut Ui, label: &str, color: &mut [f32; 3]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        if ui.color_edit_button_rgb(color).changed() {
            changed = true;
        }
    });
    changed
}

fn edit_material_texture(ui: &mut Ui, label: &str, value: &mut Option<String>) -> EditResponse {
    let mut response = EditResponse::default();
    let mut buffer = value.clone().unwrap_or_default();

    ui.horizontal(|ui| {
        ui.label(label);
        let text_response = ui.text_edit_singleline(&mut buffer);
        response.merge(EditResponse::from_response(&text_response));
        if text_response.changed() {
            let trimmed = buffer.trim();
            if trimmed.is_empty() {
                *value = None;
            } else {
                *value = Some(trimmed.to_string());
            }
        }
        if ui.button("Browse...").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Texture", &["ktx2", "png", "jpg", "jpeg", "tga"])
                .pick_file()
            {
                *value = Some(path.to_string_lossy().to_string());
                response.changed = true;
                response.lost_focus = true;
            }
        }
        if ui.button("Clear").clicked() {
            *value = None;
            response.changed = true;
            response.lost_focus = true;
        }
    });

    response
}

fn edit_vec3(ui: &mut Ui, label: &str, value: &mut Vec3, speed: f32) -> EditResponse {
    let mut response = EditResponse::default();
    ui.horizontal(|ui| {
        ui.label(label);
        let x_response = ui.add(DragValue::new(&mut value.x).speed(speed));
        response.merge(EditResponse::from_response(&x_response));
        let y_response = ui.add(DragValue::new(&mut value.y).speed(speed));
        response.merge(EditResponse::from_response(&y_response));
        let z_response = ui.add(DragValue::new(&mut value.z).speed(speed));
        response.merge(EditResponse::from_response(&z_response));
    });
    response
}

fn edit_vec3_inline(ui: &mut Ui, value: &mut Vec3, speed: f32) -> EditResponse {
    let mut response = EditResponse::default();
    let x_response = ui.add(DragValue::new(&mut value.x).speed(speed));
    response.merge(EditResponse::from_response(&x_response));
    let y_response = ui.add(DragValue::new(&mut value.y).speed(speed));
    response.merge(EditResponse::from_response(&y_response));
    let z_response = ui.add(DragValue::new(&mut value.z).speed(speed));
    response.merge(EditResponse::from_response(&z_response));
    response
}

fn edit_inherited_float(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    inherit: &mut bool,
    world_value: f32,
    speed: f32,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.checkbox(inherit, "Use World").changed();
        ui.label(label);
        if *inherit {
            if *value != world_value {
                *value = world_value;
                changed = true;
            }
            ui.label(format!("= {:.4}", world_value));
        } else if ui.add(DragValue::new(value).speed(speed)).changed() {
            changed = true;
        }
    });
    changed
}

fn edit_inherited_bool(
    ui: &mut Ui,
    label: &str,
    value: &mut bool,
    inherit: &mut bool,
    world_value: bool,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.checkbox(inherit, "Use World").changed();
        if *inherit {
            if *value != world_value {
                *value = world_value;
                changed = true;
            }
            ui.label(format!("{} = {}", label, world_value));
        } else if ui.checkbox(value, label).changed() {
            changed = true;
        }
    });
    changed
}

fn edit_inherited_i8(
    ui: &mut Ui,
    label: &str,
    value: &mut i8,
    inherit: &mut bool,
    world_value: i8,
    speed: f32,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.checkbox(inherit, "Use World").changed();
        ui.label(label);
        if *inherit {
            if *value != world_value {
                *value = world_value;
                changed = true;
            }
            ui.label(format!("= {}", world_value));
        } else if ui.add(DragValue::new(value).speed(speed)).changed() {
            changed = true;
        }
    });
    changed
}

fn edit_inherited_u32(
    ui: &mut Ui,
    label: &str,
    value: &mut u32,
    inherit: &mut bool,
    world_value: u32,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.checkbox(inherit, "Use World").changed();
        ui.label(label);
        if *inherit {
            if *value != world_value {
                *value = world_value;
                changed = true;
            }
            ui.label(format!("= {}", world_value));
        } else if ui
            .add(DragValue::new(value).speed(1.0).range(0..=u32::MAX))
            .changed()
        {
            changed = true;
        }
    });
    changed
}

fn edit_inherited_vec3(
    ui: &mut Ui,
    label: &str,
    value: &mut Vec3,
    inherit: &mut bool,
    world_value: Vec3,
    speed: f32,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.checkbox(inherit, "Use World").changed();
        if *inherit {
            if *value != world_value {
                *value = world_value;
                changed = true;
            }
            ui.label(format!(
                "{} = [{:.3}, {:.3}, {:.3}]",
                label, world_value.x, world_value.y, world_value.z
            ));
        } else {
            ui.label(label);
            changed |= edit_vec3_inline(ui, value, speed).changed;
        }
    });
    changed
}

fn edit_inherited_quat_euler(
    ui: &mut Ui,
    label: &str,
    value: &mut Quat,
    inherit: &mut bool,
    world_value: Quat,
    speed: f32,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.checkbox(inherit, "Use World").changed();
        if *inherit {
            if *value != world_value {
                *value = world_value;
                changed = true;
            }
            let world_rot = world_value.to_euler(EulerRot::XYZ);
            ui.label(format!(
                "{} = [{:.1}, {:.1}, {:.1}]",
                label,
                world_rot.0.to_degrees(),
                world_rot.1.to_degrees(),
                world_rot.2.to_degrees()
            ));
        } else {
            ui.label(label);
            let rot = value.to_euler(EulerRot::XYZ);
            let mut rot_deg = Vec3::new(rot.0.to_degrees(), rot.1.to_degrees(), rot.2.to_degrees());
            if edit_vec3_inline(ui, &mut rot_deg, speed).changed {
                *value = Quat::from_euler(
                    EulerRot::XYZ,
                    rot_deg.x.to_radians(),
                    rot_deg.y.to_radians(),
                    rot_deg.z.to_radians(),
                );
                changed = true;
            }
        }
    });
    changed
}

fn edit_dynamic_value(ui: &mut Ui, value: &mut DynamicValue) -> EditResponse {
    match value {
        DynamicValue::Bool(value) => {
            let response = ui.checkbox(value, "");
            EditResponse::from_response(&response)
        }
        DynamicValue::Float(value) => {
            let response = ui.add(DragValue::new(value).speed(0.1));
            EditResponse::from_response(&response)
        }
        DynamicValue::Int(value) => {
            let response = ui.add(DragValue::new(value).speed(1.0));
            EditResponse::from_response(&response)
        }
        DynamicValue::Vec3(value) => {
            let mut vec = Vec3::new(value[0], value[1], value[2]);
            let response = edit_vec3_inline(ui, &mut vec, 0.1);
            if response.changed {
                *value = [vec.x, vec.y, vec.z];
            }
            response
        }
        DynamicValue::String(value) => {
            let response = ui.text_edit_singleline(value);
            EditResponse::from_response(&response)
        }
    }
}

fn dynamic_value_label(value: &DynamicValue) -> &'static str {
    match value {
        DynamicValue::Bool(_) => "Bool",
        DynamicValue::Float(_) => "Float",
        DynamicValue::Int(_) => "Int",
        DynamicValue::Vec3(_) => "Vec3",
        DynamicValue::String(_) => "String",
    }
}

fn dynamic_value_kind_label(kind: DynamicValueKind) -> &'static str {
    match kind {
        DynamicValueKind::Bool => "Bool",
        DynamicValueKind::Float => "Float",
        DynamicValueKind::Int => "Int",
        DynamicValueKind::Vec3 => "Vec3",
        DynamicValueKind::String => "String",
    }
}

fn ensure_transform(world: &mut World, entity: Entity) {
    if world.get::<BevyTransform>(entity).is_none() {
        world.entity_mut(entity).insert(BevyTransform::default());
    }
}

fn mesh_source_from_path(project: &Option<EditorProject>, path: &Path) -> MeshSource {
    MeshSource::Asset {
        path: project_relative_path(project, path),
    }
}

fn material_path_from_project(project: &Option<EditorProject>, path: &Path) -> Option<String> {
    Some(project_relative_path(project, path))
}

fn project_relative_path(project: &Option<EditorProject>, path: &Path) -> String {
    project
        .as_ref()
        .and_then(|project| {
            project
                .root
                .as_ref()
                .and_then(|root| path.strip_prefix(root).ok())
        })
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|| path.to_string_lossy().replace('\\', "/"))
}

fn resolve_asset_path(project: Option<&EditorProject>, path: &str) -> PathBuf {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        return candidate.to_path_buf();
    }

    if let Some(project) = project {
        if let Some(root) = project.root.as_ref() {
            return root.join(path);
        }
    }

    candidate.to_path_buf()
}

fn mark_entity_render_dirty(world: &mut World, entity: Entity) {
    if let Some(mut transform) = world.get_mut::<BevyTransform>(entity) {
        let current = transform.0;
        transform.0 = current;
    }
}

fn apply_mesh_renderer(
    world: &mut World,
    entity: Entity,
    project: &Option<EditorProject>,
    source: MeshSource,
    material_path: Option<String>,
    casts_shadow: bool,
    visible: bool,
) {
    let project_ref = project.as_ref();
    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = match world.get_resource::<BevyAssetServer>() {
            Some(server) => server,
            None => return,
        };

        let material_handle = material_path
            .as_ref()
            .and_then(|path| load_material_handle(path, &mut cache, asset_server, project_ref));
        let material_handle = material_handle.or_else(|| {
            project_ref
                .and_then(|project| ensure_default_material(project, &mut cache, asset_server))
        });
        let Some(material_handle) = material_handle else {
            set_status(world, "Material missing".to_string());
            return;
        };

        let mesh_handle = match &source {
            MeshSource::Primitive(kind) => {
                Some(load_primitive_mesh(*kind, &mut cache, asset_server))
            }
            MeshSource::Asset { path } => {
                Some(load_mesh_asset(path, &mut cache, asset_server, project_ref))
            }
        };

        let Some(mesh_handle) = mesh_handle else {
            set_status(world, "Mesh missing".to_string());
            return;
        };

        world.entity_mut(entity).insert((
            BevyWrapper(MeshRenderer::new(
                mesh_handle.id,
                material_handle.id,
                casts_shadow,
                visible,
            )),
            EditorMesh {
                source,
                material_path,
            },
        ));
        mark_entity_render_dirty(world, entity);
    });
}

fn apply_audio_emitter_from_asset(
    world: &mut World,
    entity: Entity,
    project: &Option<EditorProject>,
    path: &Path,
    streaming: bool,
) {
    let Some(asset_server) = world
        .get_resource::<BevyAssetServer>()
        .map(|server| BevyAssetServer(server.0.clone()))
    else {
        set_status(world, "Asset server missing".to_string());
        return;
    };
    let path_str = path.to_string_lossy();
    let resolved_path = resolve_asset_path(project.as_ref(), path_str.as_ref());
    let relative_path = project_relative_path(project, path);

    let handle = if let Some(mut cache) = world.get_resource_mut::<EditorAssetCache>() {
        cached_audio_handle(&mut cache, &asset_server, &resolved_path, streaming)
    } else {
        asset_server.0.lock().load_audio(
            &resolved_path,
            if streaming {
                AudioLoadMode::Streaming
            } else {
                AudioLoadMode::Static
            },
        )
    };

    let mut emitter = world
        .get::<BevyAudioEmitter>(entity)
        .map(|emitter| emitter.0)
        .unwrap_or_default();
    emitter.clip_id = Some(handle.id);

    world.entity_mut(entity).insert((
        BevyWrapper(emitter),
        EditorAudio {
            path: Some(relative_path),
            streaming,
        },
    ));
}

fn apply_skinned_mesh_renderer_from_asset(
    world: &mut World,
    entity: Entity,
    project: &Option<EditorProject>,
    path: &Path,
) -> bool {
    let Some(asset_server) = world
        .get_resource::<BevyAssetServer>()
        .map(|server| BevyAssetServer(server.0.clone()))
    else {
        set_status(world, "Asset server missing".to_string());
        return false;
    };
    let path_str = path.to_string_lossy();
    let resolved_path = resolve_asset_path(project.as_ref(), path_str.as_ref());
    let relative_path = project_relative_path(project, path);

    let (scene_handle, scene) = {
        let handle = if let Some(mut cache) = world.get_resource_mut::<EditorAssetCache>() {
            cached_scene_handle(&mut cache, &asset_server, &resolved_path)
        } else {
            asset_server.0.lock().load_scene(&resolved_path)
        };
        let scene = asset_server.0.lock().get_scene(&handle);
        (handle, scene)
    };

    let existing_node_index = world
        .get::<EditorSkinnedMesh>(entity)
        .and_then(|skinned| skinned.node_index);
    let (casts_shadow, visible) = world
        .get::<BevyMeshRenderer>(entity)
        .map(|renderer| (renderer.0.casts_shadow, renderer.0.visible))
        .or_else(|| {
            world
                .get::<EditorSkinnedMesh>(entity)
                .map(|skinned| (skinned.casts_shadow, skinned.visible))
        })
        .unwrap_or((true, true));

    world.entity_mut(entity).insert(EditorSkinnedMesh {
        scene_path: Some(relative_path),
        node_index: existing_node_index,
        casts_shadow,
        visible,
    });

    let Some(scene) = scene else {
        world.entity_mut(entity).insert(PendingSkinnedMeshAsset {
            scene_handle: scene_handle,
            node_index: existing_node_index,
        });
        set_status(
            world,
            format!(
                "Scene not ready yet for {}. Will apply when loaded.",
                resolved_path.display()
            ),
        );
        return true;
    };

    let mut skinned_nodes = Vec::new();
    for (index, node) in scene.nodes.iter().enumerate() {
        if node.skin_index.is_some() {
            skinned_nodes.push((index, node));
        }
    }

    let node_index = existing_node_index
        .filter(|index| {
            scene
                .nodes
                .get(*index)
                .map(|node| node.skin_index.is_some())
                .unwrap_or(false)
        })
        .or_else(|| skinned_nodes.first().map(|(index, _)| *index));

    let Some(node_index) = node_index else {
        set_status(world, "Scene has no skinned nodes.".to_string());
        return false;
    };

    let Some(node) = scene.nodes.get(node_index) else {
        set_status(world, "Selected skinned node missing.".to_string());
        return false;
    };

    if skinned_nodes.len() > 1 {
        set_status(
            world,
            format!("Multiple skinned nodes found; using node {}.", node_index),
        );
    }

    let Some(skin_index) = node.skin_index else {
        set_status(world, "Selected node has no skin.".to_string());
        return false;
    };

    if let Some(mut skinned) = world.get_mut::<EditorSkinnedMesh>(entity) {
        skinned.node_index = Some(node_index);
    }

    let Some(skin) = scene.skins.read().get(skin_index).cloned() else {
        world.entity_mut(entity).insert(PendingSkinnedMeshAsset {
            scene_handle,
            node_index: Some(node_index),
        });
        set_status(
            world,
            "Skin data not ready yet; will apply when loaded.".to_string(),
        );
        return true;
    };

    apply_skinned_mesh_renderer_from_scene_node(world, entity, &scene, node, skin_index, skin)
}

fn apply_skinned_mesh_renderer_from_scene_node(
    world: &mut World,
    entity: Entity,
    scene: &Scene,
    node: &helmer::runtime::asset_server::SceneNode,
    skin_index: usize,
    skin: std::sync::Arc<helmer::animation::Skin>,
) -> bool {
    let (casts_shadow, visible) = world
        .get::<BevyMeshRenderer>(entity)
        .map(|renderer| (renderer.0.casts_shadow, renderer.0.visible))
        .or_else(|| {
            world
                .get::<EditorSkinnedMesh>(entity)
                .map(|skinned| (skinned.casts_shadow, skinned.visible))
        })
        .unwrap_or((true, true));

    let skinned =
        SkinnedMeshRenderer::new(node.mesh.id, node.material.id, skin, casts_shadow, visible);

    world.entity_mut(entity).remove::<BevyMeshRenderer>();
    world.entity_mut(entity).remove::<EditorMesh>();
    world.entity_mut(entity).remove::<PendingSkinnedMeshAsset>();
    world
        .entity_mut(entity)
        .insert(BevySkinnedMeshRenderer(skinned));
    mark_entity_render_dirty(world, entity);

    if world.get::<BevyAnimator>(entity).is_none() {
        if let Some(anim_lib) = scene.animations.read().get(skin_index).cloned() {
            world
                .entity_mut(entity)
                .insert(BevyAnimator(build_default_animator(anim_lib)));
        }
    }

    true
}

pub fn refresh_material_usage(world: &mut World, project: &Option<EditorProject>, path: &Path) {
    let material_key = project_relative_path(project, path);
    let mut targets = Vec::new();

    let mut query = world.query::<(Entity, &EditorMesh, &BevyMeshRenderer)>();
    for (entity, mesh, renderer) in query.iter(world) {
        if mesh.material_path.as_deref() == Some(material_key.as_str()) {
            targets.push((
                entity,
                mesh.source.clone(),
                mesh.material_path.clone(),
                renderer.0.casts_shadow,
                renderer.0.visible,
            ));
        }
    }

    if let Some(mut cache) = world.get_resource_mut::<EditorAssetCache>() {
        cache.material_handles.remove(&material_key);
    }

    for (entity, source, material_path, casts_shadow, visible) in targets {
        apply_mesh_renderer(
            world,
            entity,
            project,
            source,
            material_path,
            casts_shadow,
            visible,
        );
    }
}

fn load_material_file(path: &Path) -> Result<MaterialFile, String> {
    let data = fs::read_to_string(path).map_err(|err| err.to_string())?;
    ron::de::from_str(&data).map_err(|err| err.to_string())
}

fn save_material_file(path: &Path, data: &MaterialFile) -> Result<(), String> {
    let pretty = PrettyConfig::new().compact_arrays(false);
    let payload = ron::ser::to_string_pretty(data, pretty).map_err(|err| err.to_string())?;
    fs::write(path, payload).map_err(|err| err.to_string())
}

fn apply_scene_asset(world: &mut World, entity: Entity, path: &Path) {
    let Some(asset_server) = world
        .get_resource::<BevyAssetServer>()
        .map(|server| BevyAssetServer(server.0.clone()))
    else {
        return;
    };
    let handle = if let Some(mut cache) = world.get_resource_mut::<EditorAssetCache>() {
        cached_scene_handle(&mut cache, &asset_server, path)
    } else {
        asset_server.0.lock().load_scene(path)
    };
    world.entity_mut(entity).insert(SceneRoot(handle));
    world.entity_mut(entity).insert(SceneAssetPath {
        path: path.to_path_buf(),
    });
}

fn add_script_component(world: &mut World, entity: Entity) {
    let entry = ScriptEntry::new();
    if let Some(mut scripts) = world.get_mut::<ScriptComponent>(entity) {
        scripts.scripts.push(entry);
    } else {
        world.entity_mut(entity).insert(ScriptComponent {
            scripts: vec![entry],
        });
    }
}

fn create_script_asset(world: &mut World, project: &EditorProject) -> Option<PathBuf> {
    let root = project.root.as_ref()?;
    let config = project.config.as_ref()?;
    let scripts_root = config.scripts_root(root);

    if let Err(err) = fs::create_dir_all(&scripts_root) {
        set_status(world, format!("Failed to create scripts dir: {}", err));
        return None;
    }

    let candidate = scripts_root.join("script.luau");
    let path = unique_path(&candidate);
    if let Err(err) = fs::write(&path, default_script_template_full()) {
        set_status(world, format!("Failed to write script: {}", err));
        return None;
    }

    if let Some(mut assets) = world.get_resource_mut::<AssetBrowserState>() {
        assets.refresh_requested = true;
    }

    Some(path)
}

fn create_rust_script_asset(world: &mut World, project: &EditorProject) -> Option<PathBuf> {
    let root = project.root.as_ref()?;
    let config = project.config.as_ref()?;
    let scripts_root = config.scripts_root(root);

    if let Err(err) = fs::create_dir_all(&scripts_root) {
        set_status(world, format!("Failed to create scripts dir: {}", err));
        return None;
    }

    let crate_dir = unique_path(&scripts_root.join("rust_script"));
    let src_dir = crate_dir.join("src");
    if let Err(err) = fs::create_dir_all(&src_dir) {
        set_status(world, format!("Failed to create Rust script dir: {}", err));
        return None;
    }

    let crate_name = crate_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(sanitize_rust_crate_name)
        .unwrap_or_else(|| "rust_script".to_string());
    let sdk_path = match rust_script_sdk_dependency_path(Some(root.as_path()), &crate_dir) {
        Ok(path) => path,
        Err(err) => {
            set_status(world, format!("Failed to prepare Rust script SDK: {}", err));
            return None;
        }
    };
    let manifest_path = crate_dir.join("Cargo.toml");
    if let Err(err) = fs::write(
        &manifest_path,
        rust_script_manifest_template(&crate_name, &sdk_path),
    ) {
        set_status(
            world,
            format!("Failed to write Rust script manifest: {}", err),
        );
        return None;
    }

    if let Err(err) = fs::write(src_dir.join("lib.rs"), default_rust_script_template_full()) {
        set_status(
            world,
            format!("Failed to write Rust script source: {}", err),
        );
        return None;
    }

    if let Some(mut assets) = world.get_resource_mut::<AssetBrowserState>() {
        assets.refresh_requested = true;
    }

    Some(manifest_path)
}

fn unique_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }

    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("file");
    let extension = path.extension().and_then(|ext| ext.to_str());
    let parent = path.parent().unwrap_or_else(|| Path::new("."));

    for idx in 1..=999u32 {
        let file_name = match extension {
            Some(ext) => format!("{}_{}.{}", stem, idx, ext),
            None => format!("{}_{}", stem, idx),
        };
        let candidate = parent.join(file_name);
        if !candidate.exists() {
            return candidate;
        }
    }

    path.to_path_buf()
}

fn ensure_default_material(
    project: &EditorProject,
    cache: &mut EditorAssetCache,
    asset_server: &BevyAssetServer,
) -> Option<Handle<Material>> {
    if let Some(handle) = cache.default_material {
        return Some(handle);
    }

    let root = project.root.as_ref()?;
    let config = project.config.as_ref()?;
    let default_path = config.materials_root(root).join("default.ron");
    let handle = asset_server.0.lock().load_material(&default_path);

    let relative = default_path
        .strip_prefix(root)
        .ok()
        .map(|path| path.to_string_lossy().replace('\\', "/"));

    cache.default_material = Some(handle);
    if let Some(relative) = relative {
        cache.material_handles.insert(relative, handle);
    }

    Some(handle)
}

fn load_material_handle(
    path: &str,
    cache: &mut EditorAssetCache,
    asset_server: &BevyAssetServer,
    project: Option<&EditorProject>,
) -> Option<Handle<Material>> {
    if let Some(handle) = cache.material_handles.get(path).copied() {
        return Some(handle);
    }

    let full_path = resolve_asset_path(project, path);
    let handle = asset_server.0.lock().load_material(full_path);
    cache.material_handles.insert(path.to_string(), handle);
    Some(handle)
}

fn load_mesh_asset(
    path: &str,
    cache: &mut EditorAssetCache,
    asset_server: &BevyAssetServer,
    project: Option<&EditorProject>,
) -> Handle<Mesh> {
    if let Some(handle) = cache.mesh_handles.get(path).copied() {
        return handle;
    }

    let full_path = resolve_asset_path(project, path);
    let handle = asset_server.0.lock().load_mesh(full_path);
    cache.mesh_handles.insert(path.to_string(), handle);
    handle
}

fn load_primitive_mesh(
    kind: PrimitiveKind,
    cache: &mut EditorAssetCache,
    asset_server: &BevyAssetServer,
) -> Handle<Mesh> {
    if let Some(handle) = cache.primitive_meshes.get(&kind).copied() {
        return handle;
    }

    let mesh_asset = match kind {
        PrimitiveKind::Cube => helmer::provided::components::MeshAsset::cube("cube".to_string()),
        PrimitiveKind::UvSphere(segments, rings) => {
            helmer::provided::components::MeshAsset::uv_sphere(
                "uv sphere".to_string(),
                segments,
                rings,
            )
        }
        PrimitiveKind::Plane => helmer::provided::components::MeshAsset::plane("plane".to_string()),
    };

    let handle = asset_server
        .0
        .lock()
        .add_mesh(mesh_asset.vertices.unwrap(), mesh_asset.indices);
    cache.primitive_meshes.insert(kind, handle);
    handle
}

fn focus_entity_in_view(world: &mut World, entity: Entity) {
    let target = world
        .get::<BevyTransform>(entity)
        .map(|transform| transform.0.position)
        .unwrap_or(Vec3::ZERO);
    let scale = world
        .get::<BevyTransform>(entity)
        .map(|transform| transform.0.scale)
        .unwrap_or(Vec3::ONE);
    let max_scale = scale.x.max(scale.y).max(scale.z).max(1.0);
    let distance = (max_scale * 2.5).max(3.0);

    let mut active_camera = None;
    let mut query = world.query::<(Entity, &BevyCamera, Option<&BevyActiveCamera>)>();
    for (candidate, _, active) in query.iter(world) {
        if active.is_some() {
            active_camera = Some(candidate);
            break;
        }
        if active_camera.is_none() {
            active_camera = Some(candidate);
        }
    }

    let Some(camera_entity) = active_camera else {
        return;
    };

    if world.get::<BevyTransform>(camera_entity).is_none() {
        world
            .entity_mut(camera_entity)
            .insert(BevyTransform::default());
    }

    if let Some(mut transform) = world.get_mut::<BevyTransform>(camera_entity) {
        let transform = &mut transform.0;
        let direction = target - transform.position;
        let mut forward = if direction.length_squared() > 0.0001 {
            direction.normalize()
        } else {
            transform.forward()
        };
        if forward.length_squared() < 0.0001 {
            forward = Vec3::Z;
        }
        transform.position = target - forward * distance;
        transform.rotation = look_rotation(forward, Vec3::Y);
    }
}

fn look_rotation(forward: Vec3, up: Vec3) -> Quat {
    let forward = forward.normalize_or_zero();
    let mut right = up.cross(forward);
    if right.length_squared() < 0.0001 {
        right = Vec3::X;
    }
    right = right.normalize_or_zero();
    let up = forward.cross(right).normalize_or_zero();
    let basis = Mat3::from_cols(right, up, forward);
    Quat::from_mat3(&basis)
}

fn is_scene_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("ron"))
        .unwrap_or(false)
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.ends_with(".hscene.ron"))
            .unwrap_or(false)
}

fn is_animation_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("ron"))
        .unwrap_or(false)
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.ends_with(".hanim.ron"))
            .unwrap_or(false)
}

fn ensure_animation_asset_extension(path: &Path) -> PathBuf {
    let path_str = path.to_string_lossy();
    if path_str.ends_with(".hanim.ron") {
        return path.to_path_buf();
    }
    if path_str.ends_with(".ron") {
        let trimmed = path_str.trim_end_matches(".ron");
        return PathBuf::from(format!("{}.hanim.ron", trimmed));
    }
    PathBuf::from(format!("{}.hanim.ron", path_str))
}

fn is_model_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "glb" | "gltf"))
        .unwrap_or(false)
}

fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "wav" | "ogg" | "flac" | "mp3" | "aiff" | "aif" | "aifc"
            )
        })
        .unwrap_or(false)
}

fn is_material_file(path: &Path) -> bool {
    let is_ron = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("ron"))
        .unwrap_or(false);
    if !is_ron {
        return false;
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    !(name.eq_ignore_ascii_case("helmer_project.ron")
        || name.ends_with(".hscene.ron")
        || name.ends_with(".hanim.ron"))
}

fn is_script_file(path: &Path) -> bool {
    is_script_path(path)
}

fn push_command(world: &mut World, command: EditorCommand) {
    if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
        queue.push(command);
    }
}

fn set_status(world: &mut World, message: String) {
    if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
        state.status = Some(message.clone());
    }
    crate::editor::push_console_status(world, message);
}

fn set_selection(world: &mut World, entity: Option<Entity>) {
    if let Some(mut selection) = world.get_resource_mut::<InspectorSelectedEntityResource>() {
        selection.0 = entity;
    }
}

fn apply_animation_asset_to_entity(world: &mut World, entity: Entity, path: &Path) {
    let document = match read_animation_asset_document(path) {
        Ok(document) => document,
        Err(err) => {
            set_status(world, format!("Failed to load animation asset: {}", err));
            return;
        }
    };

    let skeleton = world
        .get::<BevySkinnedMeshRenderer>(entity)
        .map(|skinned| skinned.0.skin.skeleton.clone());
    let skeleton_ref = skeleton.as_deref();
    let name = world
        .get::<Name>(entity)
        .map(|name| name.to_string())
        .unwrap_or_else(|| format!("Entity {}", entity.index()));

    let clips = if let Some(mut timeline) = world.get_resource_mut::<EditorTimelineState>() {
        merge_animation_asset_into_timeline(&mut timeline, entity, name, &document, skeleton_ref)
    } else {
        Vec::new()
    };

    if let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) {
        apply_custom_clips_to_animator(&mut animator, &clips);
    }

    push_undo_snapshot(world, "Apply Animation Asset");
    set_status(world, format!("Applied animation asset {}", path.display()));
}

fn set_current_dir(world: &mut World, path: PathBuf) {
    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
        state.current_dir = Some(path.clone());
        state.selected = Some(path.clone());
        state.selected_paths.clear();
        state.selected_paths.insert(path.clone());
        state.selection_anchor = Some(path);
        state.selection_drag_start = None;
    }
}

fn set_selected_asset(world: &mut World, path: PathBuf) {
    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
        state.selected = Some(path.clone());
        state.selected_paths.clear();
        state.selected_paths.insert(path.clone());
        state.selection_anchor = Some(path);
        state.selection_drag_start = None;
    }
}

fn register_asset_click(world: &mut World, path: &Path, time: f64) -> bool {
    let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() else {
        return false;
    };

    let is_double =
        state.last_click_path.as_deref() == Some(path) && (time - state.last_click_time) <= 0.35;

    state.last_click_path = Some(path.to_path_buf());
    state.last_click_time = time;
    is_double
}

fn update_asset_selection(world: &mut World, path: &Path, index: usize, items: &[AssetEntry]) {
    let modifiers = world
        .get_resource::<EguiResource>()
        .map(|res| res.ctx.input(|input| input.modifiers))
        .unwrap_or_default();
    let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() else {
        return;
    };
    let additive = modifiers.command || modifiers.ctrl;
    let shift = modifiers.shift;

    if shift {
        let anchor_path = state
            .selection_anchor
            .clone()
            .or_else(|| state.selected.clone());
        if let Some(anchor_path) = anchor_path {
            if let Some(anchor_index) = items.iter().position(|entry| entry.path == anchor_path) {
                let (start, end) = if anchor_index <= index {
                    (anchor_index, index)
                } else {
                    (index, anchor_index)
                };
                let mut next = if additive {
                    state.selected_paths.clone()
                } else {
                    HashSet::new()
                };
                for entry in &items[start..=end] {
                    next.insert(entry.path.clone());
                }
                state.selected_paths = next;
                state.selected = Some(path.to_path_buf());
                if state.selection_anchor.is_none() {
                    state.selection_anchor = Some(anchor_path);
                }
                return;
            }
        }
    }

    if additive {
        if state.selected_paths.contains(path) {
            state.selected_paths.remove(path);
            if state.selected.as_deref() == Some(path) {
                state.selected = state.selected_paths.iter().next().cloned();
            }
        } else {
            state.selected_paths.insert(path.to_path_buf());
            state.selected = Some(path.to_path_buf());
        }
        state.selection_anchor = state.selected.clone();
    } else {
        state.selected_paths.clear();
        state.selected_paths.insert(path.to_path_buf());
        state.selected = Some(path.to_path_buf());
        state.selection_anchor = Some(path.to_path_buf());
    }
}

fn update_drag_selection(
    world: &mut World,
    tile_rects: &[(PathBuf, Rect)],
    selection_rect: Rect,
    modifiers: Modifiers,
) {
    let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() else {
        return;
    };
    let additive = modifiers.command || modifiers.ctrl;
    let mut next = if additive {
        state.selected_paths.clone()
    } else {
        HashSet::new()
    };
    let mut last_selected: Option<PathBuf> = None;
    for (path, rect) in tile_rects {
        if selection_rect.intersects(*rect) {
            next.insert(path.clone());
            last_selected = Some(path.clone());
        }
    }

    state.selected_paths = next;
    if let Some(last_selected) = last_selected {
        state.selected = Some(last_selected.clone());
        if !additive || state.selection_anchor.is_none() {
            state.selection_anchor = Some(last_selected);
        }
    } else if !additive {
        state.selected = None;
        state.selection_anchor = None;
    }
}

fn asset_drag_payload(world: &World, path: &Path) -> AssetDragPayload {
    let mut paths = Vec::new();
    if let Some(state) = world.get_resource::<AssetBrowserState>() {
        if state.selected_paths.contains(path) && state.selected_paths.len() > 1 {
            paths.extend(state.selected_paths.iter().cloned());
        } else {
            paths.push(path.to_path_buf());
        }
    } else {
        paths.push(path.to_path_buf());
    }
    AssetDragPayload { paths }
}

fn payload_primary_path(payload: &AssetDragPayload) -> Option<&PathBuf> {
    payload.paths.first()
}

fn selected_assets_for_action(world: &World, path: Option<&Path>) -> Option<Vec<PathBuf>> {
    let state = world.get_resource::<AssetBrowserState>()?;
    if let Some(path) = path {
        if state.selected_paths.contains(path) {
            let mut paths = state.selected_paths.iter().cloned().collect::<Vec<_>>();
            paths.sort();
            return Some(paths);
        }
        return Some(vec![path.to_path_buf()]);
    }
    if state.selected_paths.is_empty() {
        None
    } else {
        let mut paths = state.selected_paths.iter().cloned().collect::<Vec<_>>();
        paths.sort();
        Some(paths)
    }
}

fn toggle_expand(world: &mut World, path: PathBuf) {
    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
        if state.expanded.contains(&path) {
            state.expanded.remove(&path);
        } else {
            state.expanded.insert(path);
        }
    }
}

pub fn draw_timeline_window(ui: &mut Ui, world: &mut World) {
    ui.heading("Timeline");

    let pinned = world
        .get_resource::<InspectorPinnedEntityResource>()
        .and_then(|res| res.0);
    let selected = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|res| res.0);
    let active_entity = pinned.or(selected);

    world.resource_scope::<EditorTimelineState, _>(|world, mut timeline| {
        timeline.recompute_duration();
        if timeline.current_time > timeline.duration.max(0.01) {
            timeline.current_time = timeline.duration.max(0.01);
        }
        if timeline.view_offset > timeline.duration.max(0.01) {
            timeline.view_offset = (timeline.duration - 0.01).max(0.0);
        }

        let EditorTimelineState {
            playing,
            loop_playback,
            playback_rate,
            current_time,
            duration,
            auto_duration,
            frame_rate,
            snap_to_frame,
            smart_key,
            pixels_per_second,
            view_offset,
            new_clip_index,
            new_clip_looping,
            new_clip_speed,
            new_clip_name,
            groups,
            selected,
            apply_requested,
            next_id,
            middle_drag_active,
            selection_drag,
            selection_drag_pending,
            pending_clip_expand,
        } = &mut *timeline;

        let mut alloc_id = || {
            let id = *next_id;
            *next_id = next_id.saturating_add(1);
            id
        };

        ui.horizontal(|ui| {
            if ui.button(if *playing { "Pause" } else { "Play" }).clicked() {
                *playing = !*playing;
            }
            if ui.button("Stop").clicked() {
                *playing = false;
                *current_time = 0.0;
                *apply_requested = true;
            }
            ui.checkbox(loop_playback, "Loop");
            ui.checkbox(snap_to_frame, "Snap");
            ui.checkbox(smart_key, "Smart Key");
            ui.checkbox(auto_duration, "Auto Duration");
        });

            ui.horizontal(|ui| {
                ui.label("Time");
                let mut time_value = *current_time;
                let response = ui.add(DragValue::new(&mut time_value).speed(0.01));
                if response.changed() {
                    *current_time = snap_time(time_value, *snap_to_frame, *frame_rate);
                    *apply_requested = true;
                }
                ui.label("Duration");
                let mut duration_value = *duration;
                if ui
                    .add(DragValue::new(&mut duration_value).speed(0.05).clamp_range(0.0..=f32::MAX))
                    .changed()
                {
                    *duration = duration_value.max(0.01);
                }
                ui.label("Speed");
                ui.add(DragValue::new(playback_rate).speed(0.05).clamp_range(0.01..=10.0));
                ui.label("FPS");
                ui.add(DragValue::new(frame_rate).speed(1.0).clamp_range(1.0..=240.0));
                ui.label("Zoom");
                ui.add(DragValue::new(pixels_per_second).speed(5.0).clamp_range(10.0..=1000.0));
            });

            if let Some(entity) = active_entity {
                let name = world
                    .get::<Name>(entity)
                    .map(|name| name.as_str().to_string())
                    .unwrap_or_else(|| format!("Entity {}", entity.index()));
                ui.label(format!("Active: {}", name));

                let has_transform = world.get::<BevyTransform>(entity).is_some();
                let has_camera = world.get::<BevyCamera>(entity).is_some();
                let has_light = world.get::<BevyLight>(entity).is_some();
                let has_skinned = world.get::<BevySkinnedMeshRenderer>(entity).is_some();
                let has_spline = world.get::<BevySpline>(entity).is_some();
                let has_animator = world.get::<BevyAnimator>(entity).is_some();

                let group_index = if let Some(index) =
                    groups.iter().position(|group| group.entity == entity.to_bits())
                {
                    index
                } else {
                    groups.push(TimelineTrackGroup {
                        entity: entity.to_bits(),
                        name: name.clone(),
                        tracks: Vec::new(),
                        custom_clips: Vec::new(),
                    });
                    groups.len().saturating_sub(1)
                };

                {
                    let group = &mut groups[group_index];
                    ui.horizontal(|ui| {
                        if ui.button("Export Anim Asset").clicked() {
                            let project = world.get_resource::<EditorProject>().cloned();
                            let mut dialog =
                                rfd::FileDialog::new().add_filter("Animation", &["ron"]);
                            if let Some(project) = project.as_ref() {
                                if let (Some(root), Some(config)) =
                                    (project.root.as_ref(), project.config.as_ref())
                                {
                                    dialog = dialog.set_directory(config.assets_root(root));
                                }
                            }
                            let default_name =
                                format!("{}.hanim.ron", name.replace(' ', "_").to_lowercase());
                            if let Some(path) = dialog.set_file_name(default_name).save_file() {
                                let path = ensure_animation_asset_extension(&path);
                                let skeleton = world
                                    .get::<BevySkinnedMeshRenderer>(entity)
                                    .map(|skinned| skinned.0.skin.skeleton.as_ref());
                                let doc = animation_asset_from_group(group, skeleton);
                                match write_animation_asset_document(&path, &doc) {
                                    Ok(()) => {
                                        if let Some(mut assets) =
                                            world.get_resource_mut::<AssetBrowserState>()
                                        {
                                            assets.refresh_requested = true;
                                        }
                                        set_status(
                                            world,
                                            format!("Saved animation asset {}", path.display()),
                                        );
                                    }
                                    Err(err) => {
                                        set_status(
                                            world,
                                            format!("Failed to save animation asset: {}", err),
                                        );
                                    }
                                }
                            }
                        }
                    });
                }

                let selected_joint = world
                    .get_resource::<PoseEditorState>()
                    .and_then(|state| {
                        if state.active_entity == Some(entity.to_bits()) {
                            state.selected_joint
                        } else {
                            None
                        }
                    });
                let selected_joint_name = if let (Some(joint_index), Some(skinned)) =
                    (selected_joint, world.get::<BevySkinnedMeshRenderer>(entity))
                {
                    skinned
                        .0
                        .skin
                        .skeleton
                        .joints
                        .get(joint_index)
                        .map(|joint| joint.name.clone())
                } else {
                    None
                };

                ui.horizontal(|ui| {
                    if has_transform && ui.button("Add Transform Track").clicked() {
                        let group = &mut groups[group_index];
                        let count = group
                            .tracks
                            .iter()
                            .filter(|track| matches!(track, TimelineTrack::Transform(_)))
                            .count();
                        group.tracks.push(TimelineTrack::Transform(TransformTrack {
                            id: alloc_id(),
                            name: format!("Transform Track {}", count + 1),
                            enabled: true,
                            translation_interpolation: TimelineInterpolation::Linear,
                            rotation_interpolation: TimelineInterpolation::Linear,
                            scale_interpolation: TimelineInterpolation::Linear,
                            keys: Vec::new(),
                        }));
                    }
                    if has_camera && ui.button("Add Camera Track").clicked() {
                        let group = &mut groups[group_index];
                        let count = group
                            .tracks
                            .iter()
                            .filter(|track| matches!(track, TimelineTrack::Camera(_)))
                            .count();
                        group.tracks.push(TimelineTrack::Camera(CameraTrack {
                            id: alloc_id(),
                            name: format!("Camera Track {}", count + 1),
                            enabled: true,
                            interpolation: TimelineInterpolation::Linear,
                            keys: Vec::new(),
                        }));
                    }
                    if has_light && ui.button("Add Light Track").clicked() {
                        let group = &mut groups[group_index];
                        let count = group
                            .tracks
                            .iter()
                            .filter(|track| matches!(track, TimelineTrack::Light(_)))
                            .count();
                        group.tracks.push(TimelineTrack::Light(LightTrack {
                            id: alloc_id(),
                            name: format!("Light Track {}", count + 1),
                            enabled: true,
                            interpolation: TimelineInterpolation::Linear,
                            keys: Vec::new(),
                        }));
                    }
                    if has_skinned && ui.button("Add Pose Track").clicked() {
                        let group = &mut groups[group_index];
                        group.tracks.push(TimelineTrack::Pose(PoseTrack {
                            id: alloc_id(),
                            name: format!("Pose Track {}", group.tracks.len() + 1),
                            enabled: true,
                            weight: 1.0,
                            additive: false,
                            translation_interpolation: TimelineInterpolation::Linear,
                            rotation_interpolation: TimelineInterpolation::Linear,
                            scale_interpolation: TimelineInterpolation::Linear,
                            keys: Vec::new(),
                        }));
                    }
                    if has_skinned {
                        let can_add_joint = selected_joint.is_some();
                        if ui
                            .add_enabled(can_add_joint, egui::Button::new("Add Joint Track"))
                            .clicked()
                        {
                            let joint_index = selected_joint.unwrap_or(0);
                            let joint_label = selected_joint_name
                                .clone()
                                .unwrap_or_else(|| format!("Joint {}", joint_index));
                            let group = &mut groups[group_index];
                            let count = group
                                .tracks
                                .iter()
                                .filter(|track| {
                                    matches!(track, TimelineTrack::Joint(track) if track.joint_index == joint_index)
                                })
                                .count();
                            group.tracks.push(TimelineTrack::Joint(JointTrack {
                                id: alloc_id(),
                                name: if count == 0 {
                                    format!("Joint: {}", joint_label)
                                } else {
                                    format!("Joint: {} ({})", joint_label, count + 1)
                                },
                                enabled: true,
                                joint_index,
                                weight: 1.0,
                                additive: false,
                                translation_interpolation: TimelineInterpolation::Linear,
                                rotation_interpolation: TimelineInterpolation::Linear,
                                scale_interpolation: TimelineInterpolation::Linear,
                                keys: Vec::new(),
                            }));
                        }
                    }
                    if has_spline && ui.button("Add Spline Track").clicked() {
                        let group = &mut groups[group_index];
                        group.tracks.push(TimelineTrack::Spline(SplineTrack {
                            id: alloc_id(),
                            name: format!("Spline Track {}", group.tracks.len() + 1),
                            enabled: true,
                            interpolation: TimelineInterpolation::Linear,
                            keys: Vec::new(),
                        }));
                    }
                    if has_animator && ui.button("Add Clip Track").clicked() {
                        let group = &mut groups[group_index];
                        group.tracks.push(TimelineTrack::Clip(ClipTrack {
                            id: alloc_id(),
                            name: format!("Clip Track {}", group.tracks.len() + 1),
                            enabled: true,
                            weight: 1.0,
                            additive: false,
                            segments: Vec::new(),
                        }));
                    }
                });

                ui.horizontal(|ui| {
                    if ui.button("Keyframe").clicked() {
                        let key_time = snap_time(*current_time, *snap_to_frame, *frame_rate);
                        let mut any_changed = false;
                        let group = &mut groups[group_index];

                        if has_transform {
                            if let Some(transform) = world.get::<BevyTransform>(entity) {
                                let index = ensure_transform_track(group, &mut alloc_id);
                                if let TimelineTrack::Transform(track) =
                                    &mut group.tracks[index]
                                {
                                    if upsert_transform_key(
                                        track,
                                        key_time,
                                        transform.0,
                                        *smart_key,
                                        &mut alloc_id,
                                    ) {
                                        any_changed = true;
                                    }
                                }
                            }
                        }
                        if has_camera {
                            if let Some(camera) = world.get::<BevyCamera>(entity) {
                                let index = ensure_camera_track(group, &mut alloc_id);
                                if let TimelineTrack::Camera(track) = &mut group.tracks[index] {
                                    if upsert_camera_key(
                                        track,
                                        key_time,
                                        camera.0,
                                        *smart_key,
                                        &mut alloc_id,
                                    ) {
                                        any_changed = true;
                                    }
                                }
                            }
                        }
                        if has_light {
                            if let Some(light) = world.get::<BevyLight>(entity) {
                                let index = ensure_light_track(group, &mut alloc_id);
                                if let TimelineTrack::Light(track) = &mut group.tracks[index] {
                                    if upsert_light_key(
                                        track,
                                        key_time,
                                        light.0,
                                        *smart_key,
                                        &mut alloc_id,
                                    ) {
                                        any_changed = true;
                                    }
                                }
                            }
                        }
                        if has_skinned {
                            let pose_track_exists = group
                                .tracks
                                .iter()
                                .any(|track| matches!(track, TimelineTrack::Pose(_)));
                            let pose_edit_active = world
                                .get_resource::<PoseEditorState>()
                                .map(|state| {
                                    state.edit_mode
                                        && state.active_entity == Some(entity.to_bits())
                                })
                                .unwrap_or(false);
                            let pose_override_enabled = world
                                .get::<BevyPoseOverride>(entity)
                                .map(|pose| pose.0.enabled)
                                .unwrap_or(false);
                            if pose_track_exists || pose_edit_active || pose_override_enabled {
                                if let Some(skinned) = world.get::<BevySkinnedMeshRenderer>(entity)
                                {
                                    let skeleton = &skinned.0.skin.skeleton;
                                    let pose = world
                                        .get::<BevyPoseOverride>(entity)
                                        .map(|pose| pose.0.pose.clone())
                                        .unwrap_or_else(|| PoseOverride::new(skeleton).pose);
                                    let index = ensure_pose_track(group, &mut alloc_id);
                                    if let TimelineTrack::Pose(track) =
                                        &mut group.tracks[index]
                                    {
                                        if upsert_pose_key(
                                            track,
                                            key_time,
                                            pose,
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            any_changed = true;
                                        }
                                    }
                                }
                            }
                            if let Some(joint_index) = selected_joint {
                                if let Some(transform) =
                                    current_joint_transform(world, entity, joint_index)
                                {
                                    let joint_label = selected_joint_name
                                        .clone()
                                        .unwrap_or_else(|| format!("Joint {}", joint_index));
                                    let index =
                                        ensure_joint_track(group, joint_index, joint_label, &mut alloc_id);
                                    if let TimelineTrack::Joint(track) =
                                        &mut group.tracks[index]
                                    {
                                        if upsert_joint_key(
                                            track,
                                            key_time,
                                            transform,
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            any_changed = true;
                                        }
                                    }
                                }
                            }
                        }
                        if has_spline {
                            let spline_track_exists = group
                                .tracks
                                .iter()
                                .any(|track| matches!(track, TimelineTrack::Spline(_)));
                            if spline_track_exists {
                                if let Some(spline) = world.get::<BevySpline>(entity) {
                                    let index = ensure_spline_track(group, &mut alloc_id);
                                    if let TimelineTrack::Spline(track) =
                                        &mut group.tracks[index]
                                    {
                                        if upsert_spline_key(
                                            track,
                                            key_time,
                                            spline.0.clone(),
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            any_changed = true;
                                        }
                                    }
                                }
                            }
                        }

                        if any_changed {
                            *apply_requested = true;
                        }
                    }
                });

                if has_skinned && has_animator {
                    if let Some(animator) = world.get::<BevyAnimator>(entity) {
                        if let Some(layer) = animator.0.layers.first() {
                            let clips = &layer.graph.library.clips;
                            if !clips.is_empty() {
                                ui.horizontal(|ui| {
                                    ui.label("Fork Clip");
                                    let current = (*new_clip_index).min(clips.len().saturating_sub(1));
                                    let mut selected = current;
                                    ComboBox::from_id_source(format!(
                                        "timeline_fork_clip_{}",
                                        entity.to_bits()
                                    ))
                                    .selected_text(
                                        clips
                                            .get(current)
                                            .map(|clip| clip.name.as_str())
                                            .unwrap_or("<clip>"),
                                    )
                                    .show_ui(ui, |ui| {
                                        for (i, clip) in clips.iter().enumerate() {
                                            if ui
                                                .selectable_label(i == current, clip.name.as_str())
                                                .clicked()
                                            {
                                                selected = i;
                                            }
                                        }
                                    });
                                    *new_clip_index = selected;

                                    if ui.button("Fork to Pose Track").clicked() {
                                        if let Some(skinned) =
                                            world.get::<BevySkinnedMeshRenderer>(entity)
                                        {
                                            let skeleton = &skinned.0.skin.skeleton;
                                            let clip = clips
                                                .get(selected)
                                                .map(|clip| clip.as_ref())
                                                .unwrap_or_else(|| clips[0].as_ref());
                                            let track_id = alloc_id();
                                            let track_name =
                                                format!("{} (Fork)", clip.name.as_str());
                                            let track = build_pose_track_from_clip(
                                                track_id,
                                                track_name,
                                                clip,
                                                skeleton,
                                                &mut alloc_id,
                                            );
                                            let group = &mut groups[group_index];
                                            group.tracks.push(TimelineTrack::Pose(track));
                                            *apply_requested = true;
                                        }
                                    }
                                });
                            }
                        }
                    }
                }

                let group = &mut groups[group_index];

                ui.separator();
                ui.collapsing("Tracks", |ui| {
                    let duration_limit = (*duration).max(0.01);
                    let snap_enabled = *snap_to_frame;
                    let fps = *frame_rate;
                    let snap_time_to_frame = |time: f32| snap_time(time, snap_enabled, fps);

                    let mut remove_track: Option<u64> = None;
                    let mut baked_clips: Vec<AnimationClip> = Vec::new();
                    for track in group.tracks.iter_mut() {
                        let track_id = track.id();
                        ui.push_id(track_id, |ui| {
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label(track.name());
                                if ui.button("Remove Track").clicked() {
                                    remove_track = Some(track.id());
                                }
                            });

                            match track {
                            TimelineTrack::Pose(track) => {
                                ui.checkbox(&mut track.enabled, "Enabled");
                                ui.checkbox(&mut track.additive, "Additive");
                                ui.add(DragValue::new(&mut track.weight).speed(0.05).clamp_range(0.0..=1.0));

                                let mut interp_changed = false;
                                ui.horizontal(|ui| {
                                    ui.label("Interp Pos");
                                    let mut interp = track.translation_interpolation;
                                    ComboBox::from_id_source(format!("pose_interp_pos_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.translation_interpolation {
                                        track.translation_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Interp Rot");
                                    let mut interp = track.rotation_interpolation;
                                    ComboBox::from_id_source(format!("pose_interp_rot_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.rotation_interpolation {
                                        track.rotation_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Interp Scale");
                                    let mut interp = track.scale_interpolation;
                                    ComboBox::from_id_source(format!("pose_interp_scale_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.scale_interpolation {
                                        track.scale_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                if interp_changed {
                                    *apply_requested = true;
                                }

                                if ui.button("Add Keyframe").clicked() {
                                    if let Some(skinned) = world.get::<BevySkinnedMeshRenderer>(entity) {
                                        let skeleton = &skinned.0.skin.skeleton;
                                        let pose = world
                                            .get::<BevyPoseOverride>(entity)
                                            .map(|pose| pose.0.pose.clone())
                                            .unwrap_or_else(|| PoseOverride::new(skeleton).pose);
                                        let key_time =
                                            snap_time_to_frame(*current_time).clamp(0.0, duration_limit);
                                        if upsert_pose_key(
                                            track,
                                            key_time,
                                            pose,
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            *apply_requested = true;
                                        }
                                    }
                                }

                                ui.collapsing("Keyframes", |ui| {
                                    let mut remove_key: Option<u64> = None;
                                    for key in track.keys.iter_mut() {
                                        let selected_key = selection_contains(
                                            selected,
                                            TimelineSelection::Key {
                                                track_id: track.id,
                                                key_id: key.id,
                                            },
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .selectable_label(
                                                    selected_key,
                                                    format!("Key {}", key.id),
                                                )
                                                .clicked()
                                            {
                                                apply_selection_click(
                                                    selected,
                                                    TimelineSelection::Key {
                                                        track_id: track.id,
                                                        key_id: key.id,
                                                    },
                                                    ui.ctx().input(|input| input.modifiers),
                                                );
                                            }
                                            let mut time = key.time;
                                            if ui
                                                .add(DragValue::new(&mut time).speed(0.01))
                                                .changed()
                                            {
                                                key.time = snap_time_to_frame(time).clamp(0.0, duration_limit);
                                                *apply_requested = true;
                                            }
                                            if ui.button("Capture").clicked() {
                                                if let Some(skinned) = world.get::<BevySkinnedMeshRenderer>(entity) {
                                                    let skeleton = &skinned.0.skin.skeleton;
                                                    let pose = world
                                                        .get::<BevyPoseOverride>(entity)
                                                        .map(|pose| pose.0.pose.clone())
                                                        .unwrap_or_else(|| PoseOverride::new(skeleton).pose);
                                                    key.pose = pose;
                                                    *apply_requested = true;
                                                }
                                            }
                                            if ui.button("Delete").clicked() {
                                                remove_key = Some(key.id);
                                            }
                                        });
                                    }
                                    if let Some(key_id) = remove_key {
                                        track.keys.retain(|key| key.id != key_id);
                                        *apply_requested = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Clip Name");
                                    ui.text_edit_singleline(new_clip_name);
                                    if ui.button("Bake Clip").clicked() {
                                        if let Some(skinned) = world.get::<BevySkinnedMeshRenderer>(entity) {
                                            let skeleton = &skinned.0.skin.skeleton;
                                            if let Some(clip) = build_clip_from_pose_track(
                                                new_clip_name.clone(),
                                                track,
                                                skeleton,
                                            ) {
                                                baked_clips.push(clip);
                                                *apply_requested = true;
                                            }
                                        }
                                    }
                                });
                            }
                            TimelineTrack::Joint(track) => {
                                ui.checkbox(&mut track.enabled, "Enabled");
                                ui.checkbox(&mut track.additive, "Additive");
                                ui.add(
                                    DragValue::new(&mut track.weight)
                                        .speed(0.05)
                                        .clamp_range(0.0..=1.0),
                                );

                                let joint_label = world
                                    .get::<BevySkinnedMeshRenderer>(entity)
                                    .and_then(|skinned| {
                                        skinned
                                            .0
                                            .skin
                                            .skeleton
                                            .joints
                                            .get(track.joint_index)
                                            .map(|joint| joint.name.as_str())
                                    })
                                    .map(|name| format!("Joint: {}", name))
                                    .unwrap_or_else(|| format!("Joint: {}", track.joint_index));
                                ui.label(joint_label);

                                let mut interp_changed = false;
                                ui.horizontal(|ui| {
                                    ui.label("Interp Pos");
                                    let mut interp = track.translation_interpolation;
                                    ComboBox::from_id_source(format!("joint_interp_pos_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.translation_interpolation {
                                        track.translation_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Interp Rot");
                                    let mut interp = track.rotation_interpolation;
                                    ComboBox::from_id_source(format!("joint_interp_rot_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.rotation_interpolation {
                                        track.rotation_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Interp Scale");
                                    let mut interp = track.scale_interpolation;
                                    ComboBox::from_id_source(format!("joint_interp_scale_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.scale_interpolation {
                                        track.scale_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                if interp_changed {
                                    *apply_requested = true;
                                }

                                if ui.button("Add Keyframe").clicked() {
                                    let key_time =
                                        snap_time_to_frame(*current_time).clamp(0.0, duration_limit);
                                    if let Some(transform) =
                                        current_joint_transform(world, entity, track.joint_index)
                                    {
                                        if upsert_joint_key(
                                            track,
                                            key_time,
                                            transform,
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            *apply_requested = true;
                                        }
                                    }
                                }

                                ui.collapsing("Keyframes", |ui| {
                                    let mut remove_key: Option<u64> = None;
                                    for key in track.keys.iter_mut() {
                                        let selected_key = selection_contains(
                                            selected,
                                            TimelineSelection::Key {
                                                track_id: track.id,
                                                key_id: key.id,
                                            },
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .selectable_label(
                                                    selected_key,
                                                    format!("Key {}", key.id),
                                                )
                                                .clicked()
                                            {
                                                apply_selection_click(
                                                    selected,
                                                    TimelineSelection::Key {
                                                        track_id: track.id,
                                                        key_id: key.id,
                                                    },
                                                    ui.ctx().input(|input| input.modifiers),
                                                );
                                            }
                                            let mut time = key.time;
                                            if ui
                                                .add(DragValue::new(&mut time).speed(0.01))
                                                .changed()
                                            {
                                                key.time = snap_time_to_frame(time)
                                                    .clamp(0.0, duration_limit);
                                                *apply_requested = true;
                                            }
                                            if ui.button("Capture").clicked() {
                                                if let Some(transform) = current_joint_transform(
                                                    world,
                                                    entity,
                                                    track.joint_index,
                                                ) {
                                                    key.transform = transform;
                                                    *apply_requested = true;
                                                }
                                            }
                                            if ui.button("Delete").clicked() {
                                                remove_key = Some(key.id);
                                            }
                                        });
                                    }
                                    if let Some(key_id) = remove_key {
                                        track.keys.retain(|key| key.id != key_id);
                                        *apply_requested = true;
                                    }
                                });
                            }
                            TimelineTrack::Transform(track) => {
                                ui.checkbox(&mut track.enabled, "Enabled");

                                let mut interp_changed = false;
                                ui.horizontal(|ui| {
                                    ui.label("Interp Pos");
                                    let mut interp = track.translation_interpolation;
                                    ComboBox::from_id_source(format!("transform_interp_pos_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.translation_interpolation {
                                        track.translation_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Interp Rot");
                                    let mut interp = track.rotation_interpolation;
                                    ComboBox::from_id_source(format!("transform_interp_rot_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.rotation_interpolation {
                                        track.rotation_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Interp Scale");
                                    let mut interp = track.scale_interpolation;
                                    ComboBox::from_id_source(format!("transform_interp_scale_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.scale_interpolation {
                                        track.scale_interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                if interp_changed {
                                    *apply_requested = true;
                                }

                                if ui.button("Add Keyframe").clicked() {
                                    if let Some(transform) = world.get::<BevyTransform>(entity) {
                                        let key_time =
                                            snap_time_to_frame(*current_time).clamp(0.0, duration_limit);
                                        if upsert_transform_key(
                                            track,
                                            key_time,
                                            transform.0,
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            *apply_requested = true;
                                        }
                                    }
                                }

                                ui.collapsing("Keyframes", |ui| {
                                    let mut remove_key: Option<u64> = None;
                                    for key in track.keys.iter_mut() {
                                        let selected_key = selection_contains(
                                            selected,
                                            TimelineSelection::Key {
                                                track_id: track.id,
                                                key_id: key.id,
                                            },
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .selectable_label(selected_key, format!("Key {}", key.id))
                                                .clicked()
                                            {
                                                apply_selection_click(
                                                    selected,
                                                    TimelineSelection::Key {
                                                        track_id: track.id,
                                                        key_id: key.id,
                                                    },
                                                    ui.ctx().input(|input| input.modifiers),
                                                );
                                            }
                                            let mut time = key.time;
                                            if ui
                                                .add(DragValue::new(&mut time).speed(0.01))
                                                .changed()
                                            {
                                                key.time =
                                                    snap_time_to_frame(time).clamp(0.0, duration_limit);
                                                *apply_requested = true;
                                            }
                                            if ui.button("Capture").clicked() {
                                                if let Some(transform) = world.get::<BevyTransform>(entity) {
                                                    key.transform = transform.0;
                                                    *apply_requested = true;
                                                }
                                            }
                                            if ui.button("Delete").clicked() {
                                                remove_key = Some(key.id);
                                            }
                                        });
                                    }
                                    if let Some(key_id) = remove_key {
                                        track.keys.retain(|key| key.id != key_id);
                                        *apply_requested = true;
                                    }
                                });
                            }
                            TimelineTrack::Camera(track) => {
                                ui.checkbox(&mut track.enabled, "Enabled");

                                let mut interp = track.interpolation;
                                ComboBox::from_id_source(format!("camera_interp_{}", track.id))
                                    .selected_text(interpolation_label(interp))
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_label(
                                                matches!(interp, TimelineInterpolation::Linear),
                                                "Linear",
                                            )
                                            .clicked()
                                        {
                                            interp = TimelineInterpolation::Linear;
                                        }
                                        if ui
                                            .selectable_label(
                                                matches!(interp, TimelineInterpolation::Step),
                                                "Step",
                                            )
                                            .clicked()
                                        {
                                            interp = TimelineInterpolation::Step;
                                        }
                                    });
                                if interp != track.interpolation {
                                    track.interpolation = interp;
                                    *apply_requested = true;
                                }

                                if ui.button("Add Keyframe").clicked() {
                                    if let Some(camera) = world.get::<BevyCamera>(entity) {
                                        let key_time =
                                            snap_time_to_frame(*current_time).clamp(0.0, duration_limit);
                                        if upsert_camera_key(
                                            track,
                                            key_time,
                                            camera.0,
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            *apply_requested = true;
                                        }
                                    }
                                }

                                ui.collapsing("Keyframes", |ui| {
                                    let mut remove_key: Option<u64> = None;
                                    for key in track.keys.iter_mut() {
                                        let selected_key = selection_contains(
                                            selected,
                                            TimelineSelection::Key {
                                                track_id: track.id,
                                                key_id: key.id,
                                            },
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .selectable_label(selected_key, format!("Key {}", key.id))
                                                .clicked()
                                            {
                                                apply_selection_click(
                                                    selected,
                                                    TimelineSelection::Key {
                                                        track_id: track.id,
                                                        key_id: key.id,
                                                    },
                                                    ui.ctx().input(|input| input.modifiers),
                                                );
                                            }
                                            let mut time = key.time;
                                            if ui
                                                .add(DragValue::new(&mut time).speed(0.01))
                                                .changed()
                                            {
                                                key.time =
                                                    snap_time_to_frame(time).clamp(0.0, duration_limit);
                                                *apply_requested = true;
                                            }
                                            if ui.button("Capture").clicked() {
                                                if let Some(camera) = world.get::<BevyCamera>(entity) {
                                                    key.camera = camera.0;
                                                    *apply_requested = true;
                                                }
                                            }
                                            if ui.button("Delete").clicked() {
                                                remove_key = Some(key.id);
                                            }
                                        });
                                    }
                                    if let Some(key_id) = remove_key {
                                        track.keys.retain(|key| key.id != key_id);
                                        *apply_requested = true;
                                    }
                                });
                            }
                            TimelineTrack::Light(track) => {
                                ui.checkbox(&mut track.enabled, "Enabled");

                                let mut interp = track.interpolation;
                                ComboBox::from_id_source(format!("light_interp_{}", track.id))
                                    .selected_text(interpolation_label(interp))
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_label(
                                                matches!(interp, TimelineInterpolation::Linear),
                                                "Linear",
                                            )
                                            .clicked()
                                        {
                                            interp = TimelineInterpolation::Linear;
                                        }
                                        if ui
                                            .selectable_label(
                                                matches!(interp, TimelineInterpolation::Step),
                                                "Step",
                                            )
                                            .clicked()
                                        {
                                            interp = TimelineInterpolation::Step;
                                        }
                                    });
                                if interp != track.interpolation {
                                    track.interpolation = interp;
                                    *apply_requested = true;
                                }

                                if ui.button("Add Keyframe").clicked() {
                                    if let Some(light) = world.get::<BevyLight>(entity) {
                                        let key_time =
                                            snap_time_to_frame(*current_time).clamp(0.0, duration_limit);
                                        if upsert_light_key(
                                            track,
                                            key_time,
                                            light.0,
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            *apply_requested = true;
                                        }
                                    }
                                }

                                ui.collapsing("Keyframes", |ui| {
                                    let mut remove_key: Option<u64> = None;
                                    for key in track.keys.iter_mut() {
                                        let selected_key = selection_contains(
                                            selected,
                                            TimelineSelection::Key {
                                                track_id: track.id,
                                                key_id: key.id,
                                            },
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .selectable_label(selected_key, format!("Key {}", key.id))
                                                .clicked()
                                            {
                                                apply_selection_click(
                                                    selected,
                                                    TimelineSelection::Key {
                                                        track_id: track.id,
                                                        key_id: key.id,
                                                    },
                                                    ui.ctx().input(|input| input.modifiers),
                                                );
                                            }
                                            let mut time = key.time;
                                            if ui
                                                .add(DragValue::new(&mut time).speed(0.01))
                                                .changed()
                                            {
                                                key.time =
                                                    snap_time_to_frame(time).clamp(0.0, duration_limit);
                                                *apply_requested = true;
                                            }
                                            if ui.button("Capture").clicked() {
                                                if let Some(light) = world.get::<BevyLight>(entity) {
                                                    key.light = light.0;
                                                    *apply_requested = true;
                                                }
                                            }
                                            if ui.button("Delete").clicked() {
                                                remove_key = Some(key.id);
                                            }
                                        });
                                    }
                                    if let Some(key_id) = remove_key {
                                        track.keys.retain(|key| key.id != key_id);
                                        *apply_requested = true;
                                    }
                                });
                            }
                            TimelineTrack::Spline(track) => {
                                ui.checkbox(&mut track.enabled, "Enabled");

                                let mut interp_changed = false;
                                ui.horizontal(|ui| {
                                    ui.label("Interpolation");
                                    let mut interp = track.interpolation;
                                    ComboBox::from_id_source(format!("spline_interp_{}", track.id))
                                        .selected_text(interpolation_label(interp))
                                        .show_ui(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Linear),
                                                    "Linear",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Linear;
                                            }
                                            if ui
                                                .selectable_label(
                                                    matches!(interp, TimelineInterpolation::Step),
                                                    "Step",
                                                )
                                                .clicked()
                                            {
                                                interp = TimelineInterpolation::Step;
                                            }
                                        });
                                    if interp != track.interpolation {
                                        track.interpolation = interp;
                                        interp_changed = true;
                                    }
                                });
                                if interp_changed {
                                    *apply_requested = true;
                                }

                                if ui.button("Add Keyframe").clicked() {
                                    if let Some(spline) = world.get::<BevySpline>(entity) {
                                        let key_time =
                                            snap_time_to_frame(*current_time).clamp(0.0, duration_limit);
                                        if upsert_spline_key(
                                            track,
                                            key_time,
                                            spline.0.clone(),
                                            *smart_key,
                                            &mut alloc_id,
                                        ) {
                                            *apply_requested = true;
                                        }
                                    }
                                }

                                ui.collapsing("Keyframes", |ui| {
                                    let mut remove_key: Option<u64> = None;
                                    for key in track.keys.iter_mut() {
                                        let selected_key = selection_contains(
                                            selected,
                                            TimelineSelection::Key {
                                                track_id: track.id,
                                                key_id: key.id,
                                            },
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .selectable_label(
                                                    selected_key,
                                                    format!("Key {}", key.id),
                                                )
                                                .clicked()
                                            {
                                                apply_selection_click(
                                                    selected,
                                                    TimelineSelection::Key {
                                                        track_id: track.id,
                                                        key_id: key.id,
                                                    },
                                                    ui.ctx().input(|input| input.modifiers),
                                                );
                                            }
                                            let mut time = key.time;
                                            if ui
                                                .add(DragValue::new(&mut time).speed(0.01))
                                                .changed()
                                            {
                                                key.time = snap_time_to_frame(time).clamp(0.0, duration_limit);
                                                *apply_requested = true;
                                            }
                                            if ui.button("Capture").clicked() {
                                                if let Some(spline) = world.get::<BevySpline>(entity) {
                                                    key.spline = spline.0.clone();
                                                    *apply_requested = true;
                                                }
                                            }
                                            if ui.button("Delete").clicked() {
                                                remove_key = Some(key.id);
                                            }
                                        });
                                    }
                                    if let Some(key_id) = remove_key {
                                        track.keys.retain(|key| key.id != key_id);
                                        *apply_requested = true;
                                    }
                                });
                            }
                            TimelineTrack::Clip(track) => {
                                ui.checkbox(&mut track.enabled, "Enabled");
                                ui.checkbox(&mut track.additive, "Additive");
                                ui.add(DragValue::new(&mut track.weight).speed(0.05).clamp_range(0.0..=1.0));

                                let clip_names = world
                                    .get::<BevyAnimator>(entity)
                                    .and_then(|animator| animator.0.layers.first().map(|layer| layer.graph.library.clips.clone()))
                                    .unwrap_or_default();
                                if !clip_names.is_empty() {
                                    let current = (*new_clip_index).min(clip_names.len().saturating_sub(1));
                                    let mut selected = current;
                                    ComboBox::from_id_source(format!("timeline_clip_picker_{}", track.id))
                                        .selected_text(clip_names
                                            .get(current)
                                            .map(|clip| clip.name.as_str())
                                            .unwrap_or("<clip>"))
                                        .show_ui(ui, |ui| {
                                            for (i, clip) in clip_names.iter().enumerate() {
                                                if ui
                                                    .selectable_label(i == current, clip.name.as_str())
                                                    .clicked()
                                                {
                                                    selected = i;
                                                }
                                            }
                                        });
                                    *new_clip_index = selected;
                                }
                                ui.add(DragValue::new(new_clip_speed).speed(0.05).clamp_range(0.01..=10.0));
                                ui.checkbox(new_clip_looping, "Loop Segment");

                                if ui.button("Add Segment").clicked() {
                                    if let Some(animator) = world.get::<BevyAnimator>(entity) {
                                        if let Some(layer) = animator.0.layers.first() {
                                            if let Some(clip) = layer.graph.library.clip(*new_clip_index) {
                                                let duration = (clip.duration / new_clip_speed.max(0.01)).max(0.01);
                                                let clip_name = clip.name.clone();
                                                track.segments.push(ClipSegment {
                                                    id: alloc_id(),
                                                    start: *current_time,
                                                    duration,
                                                    clip_name,
                                                    speed: *new_clip_speed,
                                                    looping: *new_clip_looping,
                                                });
                                                track.segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
                                                *apply_requested = true;
                                            }
                                        }
                                    }
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Bake Name");
                                    ui.text_edit_singleline(new_clip_name);
                                    if ui.button("Bake Arrangement").clicked() {
                                        if let (Some(skinned), Some(animator)) = (
                                            world.get::<BevySkinnedMeshRenderer>(entity),
                                            world.get::<BevyAnimator>(entity),
                                        ) {
                                            let skeleton = &skinned.0.skin.skeleton;
                                            if let Some(clip) = build_clip_from_clip_track(
                                                new_clip_name.clone(),
                                                track,
                                                animator,
                                                skeleton,
                                                *frame_rate,
                                            ) {
                                                baked_clips.push(clip);
                                                *apply_requested = true;
                                            } else {
                                                set_status(
                                                    world,
                                                    "Unable to bake arrangement (missing clips?)"
                                                        .to_string(),
                                                );
                                            }
                                        } else {
                                            set_status(
                                                world,
                                                "Select a skinned entity with an animator to bake"
                                                    .to_string(),
                                            );
                                        }
                                    }
                                });

                                ui.collapsing("Segments", |ui| {
                                    let mut remove_segment: Option<u64> = None;
                                    for segment in track.segments.iter_mut() {
                                        let selected_segment = selection_contains(
                                            selected,
                                            TimelineSelection::Clip {
                                                track_id: track.id,
                                                segment_id: segment.id,
                                            },
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .selectable_label(
                                                    selected_segment,
                                                    format!("Seg {}", segment.id),
                                                )
                                                .clicked()
                                            {
                                                apply_selection_click(
                                                    selected,
                                                    TimelineSelection::Clip {
                                                        track_id: track.id,
                                                        segment_id: segment.id,
                                                    },
                                                    ui.ctx().input(|input| input.modifiers),
                                                );
                                            }
                                            let mut start = segment.start;
                                            if ui
                                                .add(DragValue::new(&mut start).speed(0.01))
                                                .changed()
                                            {
                                                let max_start = (duration_limit - segment.duration).max(0.0);
                                                segment.start = snap_time_to_frame(start).clamp(0.0, max_start);
                                                *apply_requested = true;
                                            }
                                            let mut duration_value = segment.duration;
                                            if ui
                                                .add(DragValue::new(&mut duration_value).speed(0.01))
                                                .changed()
                                            {
                                                segment.duration = duration_value.max(0.01);
                                                *apply_requested = true;
                                            }
                                            if ui.button("Delete").clicked() {
                                                remove_segment = Some(segment.id);
                                            }
                                        });
                                    }
                                    if let Some(segment_id) = remove_segment {
                                        track.segments.retain(|segment| segment.id != segment_id);
                                        *apply_requested = true;
                                    }
                                });
                            }
                        }
                        });
                    }
                    if let Some(track_id) = remove_track {
                        group.tracks.retain(|track| track.id() != track_id);
                    }

                    if !baked_clips.is_empty() {
                        if let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) {
                            for layer in animator.0.layers.iter_mut() {
                                let mut library = (*layer.graph.library).clone();
                                for clip in baked_clips.iter() {
                                    library.upsert_clip(clip.clone());
                                }
                                layer.graph.library = std::sync::Arc::new(library);
                            }
                        }

                        for clip in baked_clips {
                            if let Some(existing) = group
                                .custom_clips
                                .iter_mut()
                                .find(|existing| existing.name == clip.name)
                            {
                                *existing = clip;
                            } else {
                                group.custom_clips.push(clip);
                            }
                        }
                    }
                });

                ui.separator();
                draw_timeline_canvas(
                    ui,
                    group,
                    current_time,
                    (*duration).max(0.01),
                    pixels_per_second,
                    view_offset,
                    middle_drag_active,
                    *snap_to_frame,
                    *frame_rate,
                    selected,
                    apply_requested,
                    selection_drag,
                    selection_drag_pending,
                    pending_clip_expand,
                );

                if let Some(request) = pending_clip_expand.take() {
                    let mut target_segment: Option<ClipSegment> = None;
                    for track in group.tracks.iter() {
                        if track.id() == request.track_id {
                            if let TimelineTrack::Clip(track) = track {
                                target_segment = track
                                    .segments
                                    .iter()
                                    .find(|segment| segment.id == request.segment_id)
                                    .cloned();
                            }
                            break;
                        }
                    }

                    if let Some(segment) = target_segment {
                        if let (Some(skinned), Some(animator)) = (
                            world.get::<BevySkinnedMeshRenderer>(entity),
                            world.get::<BevyAnimator>(entity),
                        ) {
                            let skeleton = &skinned.0.skin.skeleton;
                            let name = format!("{} (Expanded)", segment.clip_name);
                            if let Some(track) = build_pose_track_from_clip_segment(
                                alloc_id(),
                                name,
                                &segment,
                                animator,
                                skeleton,
                                *frame_rate,
                                &mut alloc_id,
                            ) {
                                group.tracks.push(TimelineTrack::Pose(track));
                                *apply_requested = true;
                            }
                        } else {
                            set_status(
                                world,
                                "Select a skinned entity with an animator to expand clips"
                                    .to_string(),
                            );
                        }
                    }
                }
            } else {
                ui.label("Select or pin an entity to edit timeline tracks.");
            }

            if !*middle_drag_active {
                drag_egui_window_on_middle_click(ui, world, "Timeline");
            }
        });
}

fn timeline_handle_key(
    ui: &mut Ui,
    painter: &egui::Painter,
    track_id: u64,
    key_id: u64,
    key_time: f32,
    key_rect: Rect,
    color: Color32,
    row_timeline_rect: Rect,
    selected: &mut Vec<TimelineSelection>,
    current_time: &mut f32,
    apply_requested: &mut bool,
    clicked_key: &mut bool,
    visible_start: f32,
    pixels_per_second: f32,
    snap_to_frame: bool,
    frame_rate: f32,
    duration: f32,
    allow_interactions: bool,
) -> (Option<f32>, bool) {
    let selected_key = selection_contains(selected, TimelineSelection::Key { track_id, key_id });
    painter.rect_filled(key_rect, 2.0, color);
    if selected_key {
        painter.rect_stroke(
            key_rect,
            2.0,
            Stroke::new(1.5, Color32::from_gray(240)),
            StrokeKind::Inside,
        );
    }
    if !allow_interactions {
        return (None, false);
    }
    let response = ui.interact(
        key_rect,
        ui.make_persistent_id((track_id, key_id, "key")),
        Sense::click_and_drag(),
    );
    let modifiers = ui.ctx().input(|input| input.modifiers);
    if response.clicked() {
        apply_selection_click(
            selected,
            TimelineSelection::Key { track_id, key_id },
            modifiers,
        );
        *clicked_key = true;
    } else if response.secondary_clicked() {
        if !selected_key {
            apply_selection_click(
                selected,
                TimelineSelection::Key { track_id, key_id },
                modifiers,
            );
        }
        *clicked_key = true;
    }
    let mut delete = false;
    response.context_menu(|ui| {
        if ui.button("Set Playhead").clicked() {
            *current_time = key_time;
            *apply_requested = true;
            ui.close_menu();
        }
        if ui.button("Delete").clicked() {
            delete = true;
            ui.close_menu();
        }
    });
    if response.dragged() {
        if let Some(pos) = response.interact_pointer_pos() {
            let time = visible_start + (pos.x - row_timeline_rect.left()) / pixels_per_second;
            let snapped = snap_time(time, snap_to_frame, frame_rate).clamp(0.0, duration);
            *apply_requested = true;
            *clicked_key = true;
            return (Some(snapped), delete);
        }
    }
    (None, delete)
}

fn draw_timeline_canvas(
    ui: &mut Ui,
    group: &mut TimelineTrackGroup,
    current_time: &mut f32,
    duration: f32,
    pixels_per_second: &mut f32,
    view_offset: &mut f32,
    middle_drag_active: &mut bool,
    snap_to_frame: bool,
    frame_rate: f32,
    selected: &mut Vec<TimelineSelection>,
    apply_requested: &mut bool,
    selection_drag: &mut Option<TimelineDragSelect>,
    selection_drag_pending: &mut Option<TimelineDragSelectPending>,
    pending_clip_expand: &mut Option<TimelineClipExpandRequest>,
) {
    let track_count = group.tracks.len().max(1);
    let row_height = 24.0f32;
    let ruler_height = 18.0f32;
    let label_width = 160.0f32;
    let height = ruler_height + row_height * track_count as f32;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
    let painter = ui.painter_at(rect);

    let ruler_rect = Rect::from_min_max(
        Pos2::new(rect.min.x + label_width, rect.min.y),
        Pos2::new(rect.max.x, rect.min.y + ruler_height),
    );
    let timeline_rect = Rect::from_min_max(
        Pos2::new(rect.min.x + label_width, rect.min.y + ruler_height),
        rect.max,
    );

    let (middle_down, middle_pressed, middle_released, pointer_pos, pointer_delta) =
        ui.ctx().input(|input| {
            (
                input.pointer.button_down(PointerButton::Middle),
                input.pointer.button_pressed(PointerButton::Middle),
                input.pointer.button_released(PointerButton::Middle),
                input.pointer.latest_pos(),
                input.pointer.delta(),
            )
        });

    if middle_pressed {
        *middle_drag_active = pointer_pos
            .map(|pos| timeline_rect.contains(pos) || ruler_rect.contains(pos))
            .unwrap_or(false);
    }
    if middle_released || !middle_down {
        *middle_drag_active = false;
    }
    if *middle_drag_active && middle_down {
        if pointer_delta.x.abs() > 0.0 {
            *view_offset = (*view_offset - pointer_delta.x / *pixels_per_second).max(0.0);
        }
    }

    if let Some(pointer_pos) = ui.ctx().pointer_hover_pos() {
        if timeline_rect.contains(pointer_pos) || ruler_rect.contains(pointer_pos) {
            let scroll = ui.ctx().input(|input| input.raw_scroll_delta);
            if scroll.y.abs() > 0.0 {
                let cursor_time =
                    *view_offset + (pointer_pos.x - timeline_rect.left()) / *pixels_per_second;
                let zoom = (scroll.y / 200.0).exp();
                *pixels_per_second = (*pixels_per_second * zoom).clamp(10.0, 2000.0);
                *view_offset = (cursor_time
                    - (pointer_pos.x - timeline_rect.left()) / *pixels_per_second)
                    .max(0.0);
            }
            if scroll.x.abs() > 0.0 {
                *view_offset = (*view_offset - scroll.x / *pixels_per_second).max(0.0);
            }
        }
    }

    painter.rect_filled(ruler_rect, 0.0, Color32::from_gray(26));

    let visible_start = *view_offset;
    let visible_end = visible_start + timeline_rect.width() / *pixels_per_second;
    let desired_px = 80.0;
    let mut tick_step = 1.0;
    for candidate in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
        if candidate * *pixels_per_second >= desired_px {
            tick_step = candidate;
            break;
        }
    }
    let mut t = (visible_start / tick_step).floor() * tick_step;
    while t <= visible_end + tick_step {
        let x = timeline_rect.left() + (t - visible_start) * *pixels_per_second;
        if x >= timeline_rect.left() - 1.0 && x <= timeline_rect.right() + 1.0 {
            painter.line_segment(
                [
                    Pos2::new(x, timeline_rect.top()),
                    Pos2::new(x, timeline_rect.bottom()),
                ],
                Stroke::new(1.0, Color32::from_gray(60)),
            );
            painter.line_segment(
                [
                    Pos2::new(x, ruler_rect.bottom()),
                    Pos2::new(x, ruler_rect.bottom() - 6.0),
                ],
                Stroke::new(1.0, Color32::from_gray(130)),
            );
            painter.text(
                Pos2::new(x + 2.0, ruler_rect.center().y),
                Align2::LEFT_CENTER,
                format!("{:.2}", t),
                FontId::proportional(11.0),
                Color32::from_gray(190),
            );
        }
        t += tick_step;
    }

    let time_x = timeline_rect.left() + (*current_time - visible_start) * *pixels_per_second;
    painter.line_segment(
        [
            Pos2::new(time_x, ruler_rect.top()),
            Pos2::new(time_x, timeline_rect.bottom()),
        ],
        Stroke::new(2.0, Color32::from_rgb(200, 160, 90)),
    );

    let modifiers = ui.ctx().input(|input| input.modifiers);
    let (primary_down, primary_pressed) = ui.ctx().input(|input| {
        (
            input.pointer.button_down(PointerButton::Primary),
            input.pointer.button_pressed(PointerButton::Primary),
        )
    });
    let pointer_in_timeline = pointer_pos
        .map(|pos| timeline_rect.contains(pos))
        .unwrap_or(false);
    let pointer_in_ruler = pointer_pos
        .map(|pos| ruler_rect.contains(pos))
        .unwrap_or(false);

    let mut clicked_key = false;
    let allow_interactions = !*middle_drag_active;
    let mut playhead_dragging = false;
    let playhead_handle_rect = Rect::from_min_max(
        Pos2::new(time_x - 4.0, ruler_rect.top()),
        Pos2::new(time_x + 4.0, ruler_rect.bottom()),
    );
    let pointer_over_playhead = pointer_pos
        .map(|pos| playhead_handle_rect.contains(pos))
        .unwrap_or(false);
    let ruler_response = if allow_interactions {
        Some(ui.interact(
            ruler_rect,
            ui.make_persistent_id("timeline_ruler"),
            Sense::click(),
        ))
    } else {
        None
    };
    if allow_interactions {
        let response = ui.interact(
            playhead_handle_rect,
            ui.make_persistent_id("timeline_playhead"),
            Sense::click_and_drag(),
        );
        if response.dragged() || response.drag_started() {
            playhead_dragging = true;
            if let Some(pos) = response.interact_pointer_pos() {
                let time = visible_start + (pos.x - timeline_rect.left()) / *pixels_per_second;
                *current_time = snap_time(time, snap_to_frame, frame_rate).clamp(0.0, duration);
                *apply_requested = true;
            }
        }
    }
    if let Some(response) = ruler_response {
        if response.clicked() && !playhead_dragging {
            if let Some(pos) = response.interact_pointer_pos() {
                let time = visible_start + (pos.x - timeline_rect.left()) / *pixels_per_second;
                *current_time = snap_time(time, snap_to_frame, frame_rate).clamp(0.0, duration);
                *apply_requested = true;
            }
        }
    }

    let mut key_hits: Vec<(TimelineSelection, Rect)> = Vec::new();
    let mut pointer_over_key = false;
    let mut pointer_over_segment = false;
    let mut record_key_hit = |track_id: u64, key_id: u64, rect: Rect| {
        key_hits.push((TimelineSelection::Key { track_id, key_id }, rect));
        if pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false) {
            pointer_over_key = true;
        }
    };

    for (index, track) in group.tracks.iter_mut().enumerate() {
        let row_top = rect.min.y + ruler_height + index as f32 * row_height;
        let row_rect = Rect::from_min_max(
            Pos2::new(rect.min.x, row_top),
            Pos2::new(rect.max.x, row_top + row_height),
        );
        let label_rect = Rect::from_min_max(
            Pos2::new(row_rect.min.x, row_rect.min.y),
            Pos2::new(row_rect.min.x + label_width, row_rect.max.y),
        );
        let row_timeline_rect =
            Rect::from_min_max(Pos2::new(label_rect.max.x, row_rect.min.y), row_rect.max);

        let mut pointer_on_key = false;
        let mut pointer_on_segment = false;
        painter.rect_filled(row_rect, 0.0, Color32::from_gray(20));
        match track {
            TimelineTrack::Pose(track) => {
                painter.text(
                    label_rect.left_center() + Vec2::new(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    track.name.as_str(),
                    FontId::proportional(12.0),
                    Color32::from_gray(220),
                );
                let mut remove_key: Option<u64> = None;
                for key in track.keys.iter_mut() {
                    let x =
                        row_timeline_rect.left() + (key.time - visible_start) * *pixels_per_second;
                    let rect = Rect::from_center_size(
                        Pos2::new(x, row_rect.center().y),
                        Vec2::new(8.0, 8.0),
                    );
                    if pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false) {
                        pointer_on_key = true;
                    }
                    record_key_hit(track.id, key.id, rect);
                    let (new_time, delete) = timeline_handle_key(
                        ui,
                        &painter,
                        track.id,
                        key.id,
                        key.time,
                        rect,
                        Color32::from_rgb(100, 200, 240),
                        row_timeline_rect,
                        selected,
                        current_time,
                        apply_requested,
                        &mut clicked_key,
                        visible_start,
                        *pixels_per_second,
                        snap_to_frame,
                        frame_rate,
                        duration,
                        allow_interactions,
                    );
                    if let Some(time) = new_time {
                        key.time = time;
                    }
                    if delete {
                        remove_key = Some(key.id);
                    }
                }
                if let Some(key_id) = remove_key {
                    track.keys.retain(|key| key.id != key_id);
                    *apply_requested = true;
                }
                track
                    .keys
                    .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            }
            TimelineTrack::Joint(track) => {
                painter.text(
                    label_rect.left_center() + Vec2::new(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    track.name.as_str(),
                    FontId::proportional(12.0),
                    Color32::from_gray(220),
                );
                let mut remove_key: Option<u64> = None;
                for key in track.keys.iter_mut() {
                    let x =
                        row_timeline_rect.left() + (key.time - visible_start) * *pixels_per_second;
                    let rect = Rect::from_center_size(
                        Pos2::new(x, row_rect.center().y),
                        Vec2::new(8.0, 8.0),
                    );
                    if pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false) {
                        pointer_on_key = true;
                    }
                    record_key_hit(track.id, key.id, rect);
                    let (new_time, delete) = timeline_handle_key(
                        ui,
                        &painter,
                        track.id,
                        key.id,
                        key.time,
                        rect,
                        Color32::from_rgb(200, 140, 240),
                        row_timeline_rect,
                        selected,
                        current_time,
                        apply_requested,
                        &mut clicked_key,
                        visible_start,
                        *pixels_per_second,
                        snap_to_frame,
                        frame_rate,
                        duration,
                        allow_interactions,
                    );
                    if let Some(time) = new_time {
                        key.time = time;
                    }
                    if delete {
                        remove_key = Some(key.id);
                    }
                }
                if let Some(key_id) = remove_key {
                    track.keys.retain(|key| key.id != key_id);
                    *apply_requested = true;
                }
                track
                    .keys
                    .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            }
            TimelineTrack::Transform(track) => {
                painter.text(
                    label_rect.left_center() + Vec2::new(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    track.name.as_str(),
                    FontId::proportional(12.0),
                    Color32::from_gray(220),
                );
                let mut remove_key: Option<u64> = None;
                for key in track.keys.iter_mut() {
                    let x =
                        row_timeline_rect.left() + (key.time - visible_start) * *pixels_per_second;
                    let rect = Rect::from_center_size(
                        Pos2::new(x, row_rect.center().y),
                        Vec2::new(8.0, 8.0),
                    );
                    if pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false) {
                        pointer_on_key = true;
                    }
                    record_key_hit(track.id, key.id, rect);
                    let (new_time, delete) = timeline_handle_key(
                        ui,
                        &painter,
                        track.id,
                        key.id,
                        key.time,
                        rect,
                        Color32::from_rgb(240, 180, 80),
                        row_timeline_rect,
                        selected,
                        current_time,
                        apply_requested,
                        &mut clicked_key,
                        visible_start,
                        *pixels_per_second,
                        snap_to_frame,
                        frame_rate,
                        duration,
                        allow_interactions,
                    );
                    if let Some(time) = new_time {
                        key.time = time;
                    }
                    if delete {
                        remove_key = Some(key.id);
                    }
                }
                if let Some(key_id) = remove_key {
                    track.keys.retain(|key| key.id != key_id);
                    *apply_requested = true;
                }
                track
                    .keys
                    .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            }
            TimelineTrack::Camera(track) => {
                painter.text(
                    label_rect.left_center() + Vec2::new(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    track.name.as_str(),
                    FontId::proportional(12.0),
                    Color32::from_gray(220),
                );
                let mut remove_key: Option<u64> = None;
                for key in track.keys.iter_mut() {
                    let x =
                        row_timeline_rect.left() + (key.time - visible_start) * *pixels_per_second;
                    let rect = Rect::from_center_size(
                        Pos2::new(x, row_rect.center().y),
                        Vec2::new(8.0, 8.0),
                    );
                    if pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false) {
                        pointer_on_key = true;
                    }
                    record_key_hit(track.id, key.id, rect);
                    let (new_time, delete) = timeline_handle_key(
                        ui,
                        &painter,
                        track.id,
                        key.id,
                        key.time,
                        rect,
                        Color32::from_rgb(120, 170, 240),
                        row_timeline_rect,
                        selected,
                        current_time,
                        apply_requested,
                        &mut clicked_key,
                        visible_start,
                        *pixels_per_second,
                        snap_to_frame,
                        frame_rate,
                        duration,
                        allow_interactions,
                    );
                    if let Some(time) = new_time {
                        key.time = time;
                    }
                    if delete {
                        remove_key = Some(key.id);
                    }
                }
                if let Some(key_id) = remove_key {
                    track.keys.retain(|key| key.id != key_id);
                    *apply_requested = true;
                }
                track
                    .keys
                    .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            }
            TimelineTrack::Light(track) => {
                painter.text(
                    label_rect.left_center() + Vec2::new(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    track.name.as_str(),
                    FontId::proportional(12.0),
                    Color32::from_gray(220),
                );
                let mut remove_key: Option<u64> = None;
                for key in track.keys.iter_mut() {
                    let x =
                        row_timeline_rect.left() + (key.time - visible_start) * *pixels_per_second;
                    let rect = Rect::from_center_size(
                        Pos2::new(x, row_rect.center().y),
                        Vec2::new(8.0, 8.0),
                    );
                    if pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false) {
                        pointer_on_key = true;
                    }
                    record_key_hit(track.id, key.id, rect);
                    let (new_time, delete) = timeline_handle_key(
                        ui,
                        &painter,
                        track.id,
                        key.id,
                        key.time,
                        rect,
                        Color32::from_rgb(220, 220, 120),
                        row_timeline_rect,
                        selected,
                        current_time,
                        apply_requested,
                        &mut clicked_key,
                        visible_start,
                        *pixels_per_second,
                        snap_to_frame,
                        frame_rate,
                        duration,
                        allow_interactions,
                    );
                    if let Some(time) = new_time {
                        key.time = time;
                    }
                    if delete {
                        remove_key = Some(key.id);
                    }
                }
                if let Some(key_id) = remove_key {
                    track.keys.retain(|key| key.id != key_id);
                    *apply_requested = true;
                }
                track
                    .keys
                    .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            }
            TimelineTrack::Spline(track) => {
                painter.text(
                    label_rect.left_center() + Vec2::new(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    track.name.as_str(),
                    FontId::proportional(12.0),
                    Color32::from_gray(220),
                );
                let mut remove_key: Option<u64> = None;
                for key in track.keys.iter_mut() {
                    let x =
                        row_timeline_rect.left() + (key.time - visible_start) * *pixels_per_second;
                    let rect = Rect::from_center_size(
                        Pos2::new(x, row_rect.center().y),
                        Vec2::new(8.0, 8.0),
                    );
                    if pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false) {
                        pointer_on_key = true;
                    }
                    record_key_hit(track.id, key.id, rect);
                    let (new_time, delete) = timeline_handle_key(
                        ui,
                        &painter,
                        track.id,
                        key.id,
                        key.time,
                        rect,
                        Color32::from_rgb(180, 160, 240),
                        row_timeline_rect,
                        selected,
                        current_time,
                        apply_requested,
                        &mut clicked_key,
                        visible_start,
                        *pixels_per_second,
                        snap_to_frame,
                        frame_rate,
                        duration,
                        allow_interactions,
                    );
                    if let Some(time) = new_time {
                        key.time = time;
                    }
                    if delete {
                        remove_key = Some(key.id);
                    }
                }
                if let Some(key_id) = remove_key {
                    track.keys.retain(|key| key.id != key_id);
                    *apply_requested = true;
                }
                track
                    .keys
                    .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            }
            TimelineTrack::Clip(track) => {
                painter.text(
                    label_rect.left_center() + Vec2::new(4.0, 0.0),
                    Align2::LEFT_CENTER,
                    track.name.as_str(),
                    FontId::proportional(12.0),
                    Color32::from_gray(220),
                );
                let mut remove_segment: Option<u64> = None;
                for segment in track.segments.iter_mut() {
                    let start_x = row_timeline_rect.left()
                        + (segment.start - visible_start) * *pixels_per_second;
                    let end_x = row_timeline_rect.left()
                        + (segment.start + segment.duration - visible_start) * *pixels_per_second;
                    let seg_rect = Rect::from_min_max(
                        Pos2::new(start_x, row_rect.min.y + 4.0),
                        Pos2::new(end_x.max(start_x + 6.0), row_rect.max.y - 4.0),
                    );
                    if pointer_pos
                        .map(|pos| seg_rect.contains(pos))
                        .unwrap_or(false)
                    {
                        pointer_on_segment = true;
                        pointer_over_segment = true;
                    }
                    painter.rect_filled(seg_rect, 2.0, Color32::from_rgb(120, 220, 140));
                    let selected_segment = selection_contains(
                        selected,
                        TimelineSelection::Clip {
                            track_id: track.id,
                            segment_id: segment.id,
                        },
                    );
                    if selected_segment {
                        painter.rect_stroke(
                            seg_rect,
                            2.0,
                            Stroke::new(1.5, Color32::from_gray(240)),
                            StrokeKind::Inside,
                        );
                    }
                    if allow_interactions {
                        let response = ui.interact(
                            seg_rect,
                            ui.make_persistent_id((track.id, segment.id, "segment")),
                            Sense::click_and_drag(),
                        );
                        if response.clicked() || response.secondary_clicked() {
                            apply_selection_click(
                                selected,
                                TimelineSelection::Clip {
                                    track_id: track.id,
                                    segment_id: segment.id,
                                },
                                ui.ctx().input(|input| input.modifiers),
                            );
                            clicked_key = true;
                        }
                        if response.double_clicked() {
                            *pending_clip_expand = Some(TimelineClipExpandRequest {
                                track_id: track.id,
                                segment_id: segment.id,
                            });
                            clicked_key = true;
                        }
                        response.context_menu(|ui| {
                            if ui.button("Set Playhead").clicked() {
                                *current_time = segment.start;
                                *apply_requested = true;
                                ui.close_menu();
                            }
                            if ui.button("Delete").clicked() {
                                remove_segment = Some(segment.id);
                                ui.close_menu();
                            }
                        });
                        if response.dragged() {
                            if let Some(pos) = response.interact_pointer_pos() {
                                let time = visible_start
                                    + (pos.x - row_timeline_rect.left()) / *pixels_per_second;
                                let max_start = (duration - segment.duration).max(0.0);
                                segment.start = snap_time(time, snap_to_frame, frame_rate)
                                    .clamp(0.0, max_start);
                                *apply_requested = true;
                                clicked_key = true;
                            }
                        }
                    }
                }
                if let Some(segment_id) = remove_segment {
                    track.segments.retain(|segment| segment.id != segment_id);
                    *apply_requested = true;
                }
            }
        }

        let pointer_on_item = pointer_on_key || pointer_on_segment;
        let pointer_on_playhead = pointer_pos
            .map(|pos| playhead_handle_rect.contains(pos))
            .unwrap_or(false);
        if allow_interactions
            && selection_drag.is_none()
            && !pointer_on_item
            && !playhead_dragging
            && !pointer_on_playhead
        {
            let response = ui.interact(
                row_timeline_rect,
                ui.make_persistent_id((track.id(), "row")),
                Sense::click(),
            );
            if response.clicked() && !clicked_key {
                if let Some(pos) = response.interact_pointer_pos() {
                    let time =
                        visible_start + (pos.x - row_timeline_rect.left()) / *pixels_per_second;
                    *current_time = snap_time(time, snap_to_frame, frame_rate).clamp(0.0, duration);
                    *apply_requested = true;
                }
            }
        }
    }

    let pointer_over_item = pointer_over_key || pointer_over_segment;
    if allow_interactions {
        if let Some(drag) = selection_drag.as_mut() {
            if primary_down {
                if let Some(pos) = pointer_pos {
                    drag.current = glam::Vec2::new(pos.x, pos.y);
                }
                apply_drag_selection(selected, drag, &key_hits);
            } else {
                *selection_drag = None;
            }
        } else {
            let drag_threshold = 4.0f32;
            if let Some(pending) = selection_drag_pending.as_mut() {
                if primary_down {
                    if let Some(pos) = pointer_pos {
                        let current = glam::Vec2::new(pos.x, pos.y);
                        let delta = current - pending.start;
                        if delta.length() >= drag_threshold {
                            *selection_drag = Some(TimelineDragSelect {
                                start: pending.start,
                                current,
                                mode: pending.mode,
                                base_selection: pending.base_selection.clone(),
                            });
                            *selection_drag_pending = None;
                            if let Some(drag) = selection_drag.as_ref() {
                                apply_drag_selection(selected, drag, &key_hits);
                            }
                        }
                    }
                } else {
                    *selection_drag_pending = None;
                }
            } else if primary_pressed
                && pointer_in_timeline
                && !pointer_in_ruler
                && !pointer_over_item
                && !clicked_key
                && !pointer_over_playhead
            {
                if let Some(pos) = pointer_pos {
                    let mode = if modifiers.ctrl || modifiers.command {
                        TimelineDragSelectMode::Toggle
                    } else if modifiers.shift {
                        TimelineDragSelectMode::Add
                    } else {
                        TimelineDragSelectMode::Replace
                    };
                    *selection_drag_pending = Some(TimelineDragSelectPending {
                        start: glam::Vec2::new(pos.x, pos.y),
                        mode,
                        base_selection: selected.clone(),
                    });
                }
            }
        }
    } else {
        *selection_drag = None;
        *selection_drag_pending = None;
    }

    if let Some(drag) = selection_drag.as_ref() {
        let rect = selection_drag_rect(drag);
        painter.rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(80, 160, 255, 24));
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0, Color32::from_rgb(120, 180, 255)),
            StrokeKind::Inside,
        );
    }
}

fn snap_time(time: f32, snap_to_frame: bool, frame_rate: f32) -> f32 {
    if snap_to_frame && frame_rate > 0.0 {
        let frame = (time * frame_rate).round();
        frame / frame_rate
    } else {
        time
    }
}

fn selection_contains(selected: &[TimelineSelection], target: TimelineSelection) -> bool {
    selected.iter().any(|entry| *entry == target)
}

fn selection_add(selected: &mut Vec<TimelineSelection>, target: TimelineSelection) {
    if !selection_contains(selected, target) {
        selected.push(target);
    }
}

fn selection_remove(selected: &mut Vec<TimelineSelection>, target: TimelineSelection) {
    selected.retain(|entry| *entry != target);
}

fn selection_set_single(selected: &mut Vec<TimelineSelection>, target: TimelineSelection) {
    selected.clear();
    selected.push(target);
}

fn selection_toggle(selected: &mut Vec<TimelineSelection>, target: TimelineSelection) {
    if selection_contains(selected, target) {
        selection_remove(selected, target);
    } else {
        selection_add(selected, target);
    }
}

fn apply_selection_click(
    selected: &mut Vec<TimelineSelection>,
    target: TimelineSelection,
    modifiers: Modifiers,
) {
    let toggle = modifiers.ctrl || modifiers.command;
    let additive = modifiers.shift;
    if toggle {
        selection_toggle(selected, target);
    } else if additive {
        selection_add(selected, target);
    } else {
        selection_set_single(selected, target);
    }
}

fn selection_drag_rect(drag: &TimelineDragSelect) -> Rect {
    let start = Pos2::new(drag.start.x, drag.start.y);
    let current = Pos2::new(drag.current.x, drag.current.y);
    Rect::from_two_pos(start, current)
}

fn apply_drag_selection(
    selected: &mut Vec<TimelineSelection>,
    drag: &TimelineDragSelect,
    key_hits: &[(TimelineSelection, Rect)],
) {
    let rect = selection_drag_rect(drag);
    let hits: Vec<TimelineSelection> = key_hits
        .iter()
        .filter_map(|(selection, hit_rect)| {
            if rect.intersects(*hit_rect) {
                Some(*selection)
            } else {
                None
            }
        })
        .collect();
    match drag.mode {
        TimelineDragSelectMode::Replace => {
            selected.clear();
            selected.extend(hits);
        }
        TimelineDragSelectMode::Add => {
            *selected = drag.base_selection.clone();
            for hit in hits {
                selection_add(selected, hit);
            }
        }
        TimelineDragSelectMode::Toggle => {
            *selected = drag.base_selection.clone();
            for hit in hits {
                selection_toggle(selected, hit);
            }
        }
    }
}

fn interpolation_label(value: TimelineInterpolation) -> &'static str {
    match value {
        TimelineInterpolation::Linear => "Linear",
        TimelineInterpolation::Step => "Step",
    }
}

const KEY_TIME_EPS: f32 = 1e-4;
const FLOAT_EPS: f32 = 1e-4;
const ROT_EPS: f32 = 1e-4;

fn key_index_at_time<T, F>(keys: &[T], time: f32, mut get_time: F) -> Option<usize>
where
    F: FnMut(&T) -> f32,
{
    keys.iter()
        .position(|key| (get_time(key) - time).abs() <= KEY_TIME_EPS)
}

fn last_key_before_time<T, F>(keys: &[T], time: f32, mut get_time: F) -> Option<&T>
where
    F: FnMut(&T) -> f32,
{
    for key in keys.iter().rev() {
        if get_time(key) <= time + KEY_TIME_EPS {
            return Some(key);
        }
    }
    None
}

fn quat_differs(a: Quat, b: Quat) -> bool {
    (1.0 - a.dot(b).abs()) > ROT_EPS
}

fn vec3_differs(a: Vec3, b: Vec3) -> bool {
    (a - b).length_squared() > (FLOAT_EPS * FLOAT_EPS)
}

fn transform_differs(
    a: &helmer::provided::components::Transform,
    b: &helmer::provided::components::Transform,
) -> bool {
    vec3_differs(a.position, b.position)
        || vec3_differs(a.scale, b.scale)
        || quat_differs(a.rotation, b.rotation)
}

fn camera_differs(
    a: &helmer::provided::components::Camera,
    b: &helmer::provided::components::Camera,
) -> bool {
    (a.fov_y_rad - b.fov_y_rad).abs() > FLOAT_EPS
        || (a.aspect_ratio - b.aspect_ratio).abs() > FLOAT_EPS
        || (a.near_plane - b.near_plane).abs() > FLOAT_EPS
        || (a.far_plane - b.far_plane).abs() > FLOAT_EPS
}

fn light_differs(a: &Light, b: &Light) -> bool {
    if std::mem::discriminant(&a.light_type) != std::mem::discriminant(&b.light_type) {
        return true;
    }
    let angle_differs = match (a.light_type, b.light_type) {
        (LightType::Spot { angle: a_angle }, LightType::Spot { angle: b_angle }) => {
            (a_angle - b_angle).abs() > FLOAT_EPS
        }
        _ => false,
    };
    angle_differs || vec3_differs(a.color, b.color) || (a.intensity - b.intensity).abs() > FLOAT_EPS
}

fn pose_differs(a: &Pose, b: &Pose) -> bool {
    if a.locals.len() != b.locals.len() {
        return true;
    }
    for (left, right) in a.locals.iter().zip(b.locals.iter()) {
        if transform_differs(left, right) {
            return true;
        }
    }
    false
}

fn spline_differs(a: &Spline, b: &Spline) -> bool {
    if a.points.len() != b.points.len() || a.closed != b.closed || a.mode != b.mode {
        return true;
    }
    if (a.tension - b.tension).abs() > FLOAT_EPS {
        return true;
    }
    for (pa, pb) in a.points.iter().zip(b.points.iter()) {
        if vec3_differs(*pa, *pb) {
            return true;
        }
    }
    false
}

fn upsert_pose_key(
    track: &mut PoseTrack,
    time: f32,
    pose: Pose,
    smart_key: bool,
    alloc_id: &mut impl FnMut() -> u64,
) -> bool {
    if let Some(index) = key_index_at_time(&track.keys, time, |key| key.time) {
        track.keys[index].pose = pose;
        return true;
    }
    if smart_key {
        if let Some(prev) = last_key_before_time(&track.keys, time, |key| key.time) {
            if !pose_differs(&prev.pose, &pose) {
                return false;
            }
        }
    }
    track.keys.push(PoseKey {
        id: alloc_id(),
        time,
        pose,
    });
    track
        .keys
        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    true
}

fn upsert_joint_key(
    track: &mut JointTrack,
    time: f32,
    transform: helmer::provided::components::Transform,
    smart_key: bool,
    alloc_id: &mut impl FnMut() -> u64,
) -> bool {
    if let Some(index) = key_index_at_time(&track.keys, time, |key| key.time) {
        track.keys[index].transform = transform;
        return true;
    }
    if smart_key {
        if let Some(prev) = last_key_before_time(&track.keys, time, |key| key.time) {
            if !transform_differs(&prev.transform, &transform) {
                return false;
            }
        }
    }
    track.keys.push(JointKey {
        id: alloc_id(),
        time,
        transform,
    });
    track
        .keys
        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    true
}

fn upsert_transform_key(
    track: &mut TransformTrack,
    time: f32,
    transform: helmer::provided::components::Transform,
    smart_key: bool,
    alloc_id: &mut impl FnMut() -> u64,
) -> bool {
    if let Some(index) = key_index_at_time(&track.keys, time, |key| key.time) {
        track.keys[index].transform = transform;
        return true;
    }
    if smart_key {
        if let Some(prev) = last_key_before_time(&track.keys, time, |key| key.time) {
            if !transform_differs(&prev.transform, &transform) {
                return false;
            }
        }
    }
    track.keys.push(TransformKey {
        id: alloc_id(),
        time,
        transform,
    });
    track
        .keys
        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    true
}

fn upsert_camera_key(
    track: &mut CameraTrack,
    time: f32,
    camera: helmer::provided::components::Camera,
    smart_key: bool,
    alloc_id: &mut impl FnMut() -> u64,
) -> bool {
    if let Some(index) = key_index_at_time(&track.keys, time, |key| key.time) {
        track.keys[index].camera = camera;
        return true;
    }
    if smart_key {
        if let Some(prev) = last_key_before_time(&track.keys, time, |key| key.time) {
            if !camera_differs(&prev.camera, &camera) {
                return false;
            }
        }
    }
    track.keys.push(CameraKey {
        id: alloc_id(),
        time,
        camera,
    });
    track
        .keys
        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    true
}

fn upsert_light_key(
    track: &mut LightTrack,
    time: f32,
    light: Light,
    smart_key: bool,
    alloc_id: &mut impl FnMut() -> u64,
) -> bool {
    if let Some(index) = key_index_at_time(&track.keys, time, |key| key.time) {
        track.keys[index].light = light;
        return true;
    }
    if smart_key {
        if let Some(prev) = last_key_before_time(&track.keys, time, |key| key.time) {
            if !light_differs(&prev.light, &light) {
                return false;
            }
        }
    }
    track.keys.push(LightKey {
        id: alloc_id(),
        time,
        light,
    });
    track
        .keys
        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    true
}

fn upsert_spline_key(
    track: &mut SplineTrack,
    time: f32,
    spline: Spline,
    smart_key: bool,
    alloc_id: &mut impl FnMut() -> u64,
) -> bool {
    if let Some(index) = key_index_at_time(&track.keys, time, |key| key.time) {
        track.keys[index].spline = spline;
        return true;
    }
    if smart_key {
        if let Some(prev) = last_key_before_time(&track.keys, time, |key| key.time) {
            if !spline_differs(&prev.spline, &spline) {
                return false;
            }
        }
    }
    track.keys.push(SplineKey {
        id: alloc_id(),
        time,
        spline,
    });
    track
        .keys
        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    true
}

fn ensure_transform_track(
    group: &mut TimelineTrackGroup,
    alloc_id: &mut impl FnMut() -> u64,
) -> usize {
    if let Some(index) = group
        .tracks
        .iter()
        .position(|track| matches!(track, TimelineTrack::Transform(_)))
    {
        return index;
    }
    let count = group
        .tracks
        .iter()
        .filter(|track| matches!(track, TimelineTrack::Transform(_)))
        .count();
    group.tracks.push(TimelineTrack::Transform(TransformTrack {
        id: alloc_id(),
        name: format!("Transform Track {}", count + 1),
        enabled: true,
        translation_interpolation: TimelineInterpolation::Linear,
        rotation_interpolation: TimelineInterpolation::Linear,
        scale_interpolation: TimelineInterpolation::Linear,
        keys: Vec::new(),
    }));
    group.tracks.len().saturating_sub(1)
}

fn ensure_camera_track(
    group: &mut TimelineTrackGroup,
    alloc_id: &mut impl FnMut() -> u64,
) -> usize {
    if let Some(index) = group
        .tracks
        .iter()
        .position(|track| matches!(track, TimelineTrack::Camera(_)))
    {
        return index;
    }
    let count = group
        .tracks
        .iter()
        .filter(|track| matches!(track, TimelineTrack::Camera(_)))
        .count();
    group.tracks.push(TimelineTrack::Camera(CameraTrack {
        id: alloc_id(),
        name: format!("Camera Track {}", count + 1),
        enabled: true,
        interpolation: TimelineInterpolation::Linear,
        keys: Vec::new(),
    }));
    group.tracks.len().saturating_sub(1)
}

fn ensure_light_track(group: &mut TimelineTrackGroup, alloc_id: &mut impl FnMut() -> u64) -> usize {
    if let Some(index) = group
        .tracks
        .iter()
        .position(|track| matches!(track, TimelineTrack::Light(_)))
    {
        return index;
    }
    let count = group
        .tracks
        .iter()
        .filter(|track| matches!(track, TimelineTrack::Light(_)))
        .count();
    group.tracks.push(TimelineTrack::Light(LightTrack {
        id: alloc_id(),
        name: format!("Light Track {}", count + 1),
        enabled: true,
        interpolation: TimelineInterpolation::Linear,
        keys: Vec::new(),
    }));
    group.tracks.len().saturating_sub(1)
}

fn ensure_pose_track(group: &mut TimelineTrackGroup, alloc_id: &mut impl FnMut() -> u64) -> usize {
    if let Some(index) = group
        .tracks
        .iter()
        .position(|track| matches!(track, TimelineTrack::Pose(_)))
    {
        return index;
    }
    let count = group
        .tracks
        .iter()
        .filter(|track| matches!(track, TimelineTrack::Pose(_)))
        .count();
    group.tracks.push(TimelineTrack::Pose(PoseTrack {
        id: alloc_id(),
        name: format!("Pose Track {}", count + 1),
        enabled: true,
        weight: 1.0,
        additive: false,
        translation_interpolation: TimelineInterpolation::Linear,
        rotation_interpolation: TimelineInterpolation::Linear,
        scale_interpolation: TimelineInterpolation::Linear,
        keys: Vec::new(),
    }));
    group.tracks.len().saturating_sub(1)
}

fn ensure_joint_track(
    group: &mut TimelineTrackGroup,
    joint_index: usize,
    joint_label: String,
    alloc_id: &mut impl FnMut() -> u64,
) -> usize {
    if let Some(index) = group.tracks.iter().position(
        |track| matches!(track, TimelineTrack::Joint(track) if track.joint_index == joint_index),
    ) {
        return index;
    }
    let count = group
        .tracks
        .iter()
        .filter(|track| matches!(track, TimelineTrack::Joint(track) if track.joint_index == joint_index))
        .count();
    group.tracks.push(TimelineTrack::Joint(JointTrack {
        id: alloc_id(),
        name: if count == 0 {
            format!("Joint: {}", joint_label)
        } else {
            format!("Joint: {} ({})", joint_label, count + 1)
        },
        enabled: true,
        joint_index,
        weight: 1.0,
        additive: false,
        translation_interpolation: TimelineInterpolation::Linear,
        rotation_interpolation: TimelineInterpolation::Linear,
        scale_interpolation: TimelineInterpolation::Linear,
        keys: Vec::new(),
    }));
    group.tracks.len().saturating_sub(1)
}

fn ensure_spline_track(
    group: &mut TimelineTrackGroup,
    alloc_id: &mut impl FnMut() -> u64,
) -> usize {
    if let Some(index) = group
        .tracks
        .iter()
        .position(|track| matches!(track, TimelineTrack::Spline(_)))
    {
        return index;
    }
    let count = group
        .tracks
        .iter()
        .filter(|track| matches!(track, TimelineTrack::Spline(_)))
        .count();
    group.tracks.push(TimelineTrack::Spline(SplineTrack {
        id: alloc_id(),
        name: format!("Spline Track {}", count + 1),
        enabled: true,
        interpolation: TimelineInterpolation::Linear,
        keys: Vec::new(),
    }));
    group.tracks.len().saturating_sub(1)
}

fn current_joint_transform(
    world: &World,
    entity: Entity,
    joint_index: usize,
) -> Option<helmer::provided::components::Transform> {
    let skinned = world.get::<BevySkinnedMeshRenderer>(entity)?;
    let skeleton = &skinned.0.skin.skeleton;
    let pose = world
        .get::<BevyPoseOverride>(entity)
        .map(|pose| pose.0.pose.clone())
        .unwrap_or_else(|| PoseOverride::new(skeleton).pose);
    pose.locals.get(joint_index).copied().or_else(|| {
        skeleton
            .joints
            .get(joint_index)
            .map(|joint| joint.bind_transform)
    })
}

fn build_joint_children(skeleton: &Skeleton) -> Vec<Vec<usize>> {
    let mut children = vec![Vec::new(); skeleton.joints.len()];
    for (index, joint) in skeleton.joints.iter().enumerate() {
        if let Some(parent) = joint.parent {
            if let Some(list) = children.get_mut(parent) {
                list.push(index);
            }
        }
    }
    children
}

fn joint_subtree_matches(
    skeleton: &Skeleton,
    children: &[Vec<usize>],
    index: usize,
    filter: &str,
) -> bool {
    if filter.is_empty() {
        return true;
    }
    if skeleton
        .joints
        .get(index)
        .map(|joint| joint.name.to_lowercase().contains(filter))
        .unwrap_or(false)
    {
        return true;
    }
    if let Some(kids) = children.get(index) {
        for &child in kids {
            if joint_subtree_matches(skeleton, children, child, filter) {
                return true;
            }
        }
    }
    false
}

fn draw_joint_tree(
    ui: &mut Ui,
    skeleton: &Skeleton,
    children: &[Vec<usize>],
    index: usize,
    depth: usize,
    filter: &str,
    pose_state: &mut PoseEditorState,
) {
    if !joint_subtree_matches(skeleton, children, index, filter) {
        return;
    }
    let name = skeleton
        .joints
        .get(index)
        .map(|joint| joint.name.as_str())
        .unwrap_or("<joint>");
    ui.push_id(index, |ui| {
        ui.horizontal(|ui| {
            ui.add_space(depth as f32 * 10.0);
            if ui
                .selectable_label(pose_state.selected_joint == Some(index), name)
                .clicked()
            {
                pose_state.selected_joint = Some(index);
            }
        });
    });
    if let Some(kids) = children.get(index) {
        for &child in kids {
            draw_joint_tree(ui, skeleton, children, child, depth + 1, filter, pose_state);
        }
    }
}
