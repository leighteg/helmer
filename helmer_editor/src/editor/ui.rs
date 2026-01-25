use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Entity, Resource, With, World};
use egui::{
    Align, Align2, Color32, ComboBox, DragValue, FontId, Id, Layout, Modifiers, Order,
    PointerButton, Pos2, Rect, Response, RichText, Sense, Stroke, StrokeKind, Ui, Vec2,
};
use glam::{EulerRot, Mat3, Quat, Vec3};
use helmer::graphics::{
    common::renderer::GizmoMode,
    render_graphs::{graph_templates, template_for_graph},
};
use helmer::provided::components::{Light, LightType, MeshRenderer};
use helmer::runtime::asset_server::{Handle, Material, MaterialFile, Mesh};
use helmer_becs::egui_integration::EguiResource;
use helmer_becs::physics::components::{ColliderShape, DynamicRigidBody, FixedCollider};
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::scene_system::SceneRoot;
use helmer_becs::{
    BevyActiveCamera, BevyAssetServer, BevyCamera, BevyLight, BevyMeshRenderer, BevyTransform,
    BevyWrapper,
};
use ron::ser::PrettyConfig;
use walkdir::WalkDir;

use crate::editor::{
    EditorPlayCamera, EditorUndoState, EditorViewportCamera, EditorViewportState, Freecam,
    UndoEntry,
    assets::{
        AssetBrowserState, AssetEntry, EditorAssetCache, EditorMesh, MeshSource, PrimitiveKind,
        SceneAssetPath, is_entry_visible,
    },
    begin_material_undo_group, begin_undo_group,
    commands::{AssetCreateKind, EditorCommand, EditorCommandQueue, SpawnKind},
    dynamic::{DynamicComponent, DynamicComponents, DynamicField, DynamicValue, DynamicValueKind},
    end_material_undo_group, end_undo_group, enforce_undo_cap,
    gizmos::{EditorGizmoSettings, EditorGizmoState},
    project::{EditorProject, ProjectConfig, default_script_template_full, save_project_config},
    push_undo_snapshot,
    scene::{EditorEntity, EditorSceneState, WorldState, next_available_scene_path},
    scripting::{ScriptComponent, ScriptEntry},
};

#[derive(Default, Debug, Clone, Resource)]
pub struct EditorUiState {
    pub project_name: String,
    pub project_path: String,
    pub open_project_path: String,
    pub status: Option<String>,
    pub recent_projects: Vec<PathBuf>,
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
            new_dynamic_component_name: String::new(),
            new_dynamic_field_name: String::new(),
            new_dynamic_field_kind: DynamicValueKind::Float,
        }
    }
}

pub fn draw_toolbar(ui: &mut Ui, world: &mut World) {
    bring_window_to_front_if_dragging(ui, world);
    drag_egui_window_on_middle_click(ui, world, "Toolbar");

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
        });

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
}

pub fn draw_history_window(ui: &mut Ui, world: &mut World) {
    bring_window_to_front_if_dragging(ui, world);
    drag_egui_window_on_middle_click(ui, world, "History");

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

pub fn draw_viewport_window(ui: &mut Ui, world: &mut World) {
    bring_window_to_front_if_dragging(ui, world);
    drag_egui_window_on_middle_click(ui, world, "Viewport");

    with_middle_drag_blocked(ui, world, |ui, world| {
        ui.heading("Viewport");

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
        let previous_template = graph_template.clone();

        let mut gizmos_in_play = world
            .get_resource::<EditorViewportState>()
            .map(|state| state.gizmos_in_play)
            .unwrap_or(false);

        let selected_label = templates
            .iter()
            .find(|template| template.name == graph_template)
            .map(|template| template.label)
            .unwrap_or(graph_template.as_str());

        ui.label("Render Graph");
        ComboBox::from_id_source("viewport_graph_template")
            .selected_text(selected_label)
            .show_ui(ui, |ui| {
                for template in templates {
                    if ui
                        .selectable_label(template.name == graph_template, template.label)
                        .clicked()
                    {
                        graph_template = template.name.to_string();
                    }
                }
            });

        if let Some(mut viewport_state) = world.get_resource_mut::<EditorViewportState>() {
            viewport_state.graph_template = graph_template.clone();
            viewport_state.gizmos_in_play = gizmos_in_play;
        }

        if previous_template != graph_template {
            if let Some(template) = template_for_graph(&graph_template) {
                if let Some(mut graph_res) =
                world.get_resource_mut::<helmer_becs::systems::render_system::RenderGraphResource>()
            {
                graph_res.0 = (template.build)();
            }
            }
        }

        ui.separator();

        ui.heading("Gizmos");
        ui.checkbox(&mut gizmos_in_play, "Show Gizmos in Play");
        if let Some(mut viewport_state) = world.get_resource_mut::<EditorViewportState>() {
            viewport_state.gizmos_in_play = gizmos_in_play;
        }

        let mut gizmo_mode = world
            .get_resource::<EditorGizmoState>()
            .map(|state| state.mode)
            .unwrap_or(GizmoMode::None);

        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut gizmo_mode, GizmoMode::None, "Select");
            ui.selectable_value(&mut gizmo_mode, GizmoMode::Translate, "Move");
            ui.selectable_value(&mut gizmo_mode, GizmoMode::Rotate, "Rotate");
            ui.selectable_value(&mut gizmo_mode, GizmoMode::Scale, "Scale/Resize");
        });

        if let Some(mut gizmo_state) = world.get_resource_mut::<EditorGizmoState>() {
            gizmo_state.mode = gizmo_mode;
        }

        ui.separator();
        if let Some(mut gizmo_settings) = world.get_resource_mut::<EditorGizmoSettings>() {
            ui.collapsing("Gizmo Settings", |ui| {
                if ui.button("Defaults").clicked() {
                    *gizmo_settings = EditorGizmoSettings::default();
                }

                if gizmo_settings.size_min > gizmo_settings.size_max {
                    let size_min = gizmo_settings.size_min;
                    gizmo_settings.size_min = gizmo_settings.size_max;
                    gizmo_settings.size_max = size_min;
                }

                ui.label("Sizing");
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

                ui.separator();
                ui.label("Picking");
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
                    "Scale Min",
                    &mut gizmo_settings.scale_min,
                    0.01,
                    0.0..=f32::MAX,
                );

                ui.separator();
                ui.label("Translate");
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

                ui.separator();
                ui.label("Rotate");
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

                ui.separator();
                ui.label("Scale");
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

                ui.separator();
                ui.label("Origin");
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

                ui.separator();
                ui.label("Colors");
                edit_color(ui, "Axis X", &mut gizmo_settings.axis_color_x);
                edit_color(ui, "Axis Y", &mut gizmo_settings.axis_color_y);
                edit_color(ui, "Axis Z", &mut gizmo_settings.axis_color_z);
                edit_color(ui, "Origin", &mut gizmo_settings.origin_color);

                ui.separator();
                ui.label("Selection Outline");
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
                edit_color(ui, "Color", &mut gizmo_settings.selection_color);

                ui.separator();
                ui.label("Highlight");
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
        }

        ui.separator();

        ui.heading("Camera");
        let mut camera_query = world
            .query_filtered::<(&mut BevyCamera, &mut BevyTransform), With<EditorViewportCamera>>();
        if let Some((mut camera, mut transform)) = camera_query.iter_mut(world).next() {
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
            let position_response = edit_vec3(ui, "Position", &mut position, 0.1);
            if position_response.changed {
                transform.position = position;
            }
        } else {
            ui.label("Viewport camera missing.");
        }
    });
}

pub fn draw_project_window(ui: &mut Ui, world: &mut World) {
    bring_window_to_front_if_dragging(ui, world);
    drag_egui_window_on_middle_click(ui, world, "Project");

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
    bring_window_to_front_if_dragging(ui, world);
    drag_egui_window_on_middle_click(ui, world, "Hierarchy");

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
    bring_window_to_front_if_dragging(ui, world);
    drag_egui_window_on_middle_click(ui, world, "Inspector");

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

                if response.dnd_release_payload::<TabDragPayload>().is_some() {
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
    } else if tab_bar
        .response
        .dnd_release_payload::<TabDragPayload>()
        .is_some()
    {
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
                Id::new("editor_tab_drag"),
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

fn collect_hierarchy_entries(world: &mut World) -> Vec<(Entity, String)> {
    let mut entries: Vec<(Entity, String)> = Vec::new();
    let mut query = world.query::<(
        Entity,
        Option<&Name>,
        Option<&BevyCamera>,
        Option<&BevyLight>,
        Option<&BevyMeshRenderer>,
        Option<&EditorPlayCamera>,
        Option<&SceneRoot>,
        Option<&SceneAssetPath>,
        Option<&EditorMesh>,
        Option<&ScriptComponent>,
        Option<&DynamicComponents>,
    )>();

    for (
        entity,
        name,
        camera,
        light,
        mesh,
        active_camera,
        scene_root,
        scene_asset,
        editor_mesh,
        script,
        dynamic,
    ) in query.iter(world)
    {
        if world.get::<EditorEntity>(entity).is_none() {
            continue;
        }

        let mut label = name
            .map(|name| name.to_string())
            .unwrap_or_else(|| format!("Entity {}", entity.to_bits()));

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
        if scene_root.is_some() {
            if let Some(scene) = scene_asset {
                if let Some(name) = scene.path.file_name().and_then(|name| name.to_str()) {
                    label = format!("{} ({})", label, name);
                }
            }
            tags.push("Scene");
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

        entries.push((entity, label));
    }

    entries.sort_by_key(|(entity, _)| entity.to_bits());
    entries
}

fn draw_hierarchy_panel(ui: &mut Ui, world: &mut World, entries: &[(Entity, String)]) {
    let selection = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selection| selection.0);

    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
    egui::ScrollArea::vertical()
        .id_salt("hierarchy_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                for (entity, label) in entries.iter() {
                    let is_selected = selection == Some(*entity);
                    let is_renaming = world
                        .get_resource::<HierarchyUiState>()
                        .map(|state| state.rename_entity == Some(*entity))
                        .unwrap_or(false);

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
                                apply_entity_name(world, *entity, ui_state.rename_buffer.trim());
                                ui_state.rename_entity = None;
                            }
                        });
                        continue;
                    }

                    let response = ui.add_sized(
                        Vec2::new(ui.available_width(), 0.0),
                        egui::Button::new(label).wrap().selected(is_selected),
                    );

                    if response.clicked() {
                        set_selection(world, Some(*entity));
                    }

                    if response.double_clicked() {
                        focus_entity_in_view(world, *entity);
                    }

                    response.context_menu(|ui| {
                        if ui.button("Rename").clicked() {
                            begin_rename(world, *entity);
                            ui.close_menu();
                        }
                        if ui.button("Delete").clicked() {
                            push_command(world, EditorCommand::DeleteEntity { entity: *entity });
                            ui.close_menu();
                        }
                        if ui.button("Set Game Camera").clicked() {
                            push_command(world, EditorCommand::SetActiveCamera { entity: *entity });
                            ui.close_menu();
                        }
                    });
                }
            });
        });
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

            if let Some(payload) = mesh_source_button
                .response
                .dnd_release_payload::<AssetDragPayload>()
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

            if let Some(payload) = material_button
                .response
                .dnd_release_payload::<AssetDragPayload>()
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
                push_undo_snapshot(world, "Mesh");
            }
            if ui.checkbox(&mut visible, "Visible").changed() {
                if let Some(mut renderer) = world.get_mut::<BevyMeshRenderer>(entity) {
                    renderer.0.visible = visible;
                }
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
            push_undo_snapshot(world, "Remove Dynamic Body");
        } else if let Some(body) = world.get::<DynamicRigidBody>(entity) {
            let mut mass = body.mass;
            let mass_response = edit_float(ui, "Mass", &mut mass, 0.1);
            begin_edit_undo(world, "Physics", mass_response);
            if mass_response.changed {
                if let Some(mut body) = world.get_mut::<DynamicRigidBody>(entity) {
                    body.mass = mass;
                }
            }
            end_edit_undo(world, mass_response);
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
            push_undo_snapshot(world, "Remove Fixed Collider");
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
                });
            if shape_changed {
                if let Some(mut shape) = world.get_mut::<ColliderShape>(entity) {
                    *shape = current;
                }
                push_undo_snapshot(world, "Collider Shape");
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

            if let Some(payload) = scene_asset_button
                .response
                .dnd_release_payload::<AssetDragPayload>()
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

                if let Some(payload) = path_response.dnd_release_payload::<AssetDragPayload>() {
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

                if ui.button("Browse...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Lua Script", &["lua"])
                        .pick_file()
                    {
                        script.path = Some(path);
                        scripts_changed = true;
                    }
                }

                if let Some(path) = selected_asset.as_ref() {
                    if is_script_file(path) && ui.button("Use Selected Script").clicked() {
                        script.path = Some(path.clone());
                        scripts_changed = true;
                    }
                }

                if ui.button("Create Script Asset").clicked() {
                    if let Some(project) = project.as_ref() {
                        if let Some(path) = create_script_asset(world, project) {
                            script.path = Some(path);
                            scripts_changed = true;
                        }
                    } else {
                        set_status(world, "Open a project before creating scripts".to_string());
                    }
                }

                ui.label(format!("Language: {}", script.language));
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
    bring_window_to_front_if_dragging(ui, world);
    drag_egui_window_on_middle_click(ui, world, "Content Browser");

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

fn highlight_drop_target(ui: &Ui, response: &Response) {
    if response.dnd_hover_payload::<AssetDragPayload>().is_some() {
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

                    if let Some(payload) = response.dnd_release_payload::<AssetDragPayload>() {
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
            let background_id = Id::new("asset_grid_background");
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
        if let Some(payload) = response.dnd_release_payload::<AssetDragPayload>() {
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
            } else if name.ends_with(".hscene.ron") {
                "[SCENE]"
            } else {
                "[MAT]"
            }
        }
        "lua" => "[SCRIPT]",
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
        "[SCRIPT]" => "LUA",
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

    match ext.to_ascii_lowercase().as_str() {
        "ron" => Color32::from_rgb(120, 90, 60),
        "lua" => Color32::from_rgb(60, 110, 70),
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
    let has_scene = world.get::<SceneRoot>(entity).is_some();
    let has_dynamic = world.get::<DynamicComponents>(entity).is_some();
    let has_freecam = world.get::<Freecam>(entity).is_some();
    let has_dynamic_body = world.get::<DynamicRigidBody>(entity).is_some();
    let has_fixed_collider = world.get::<FixedCollider>(entity).is_some();
    let collider_shape = world.get::<ColliderShape>(entity).copied();

    let selected_mesh_source = selected_asset
        .as_ref()
        .filter(|path| is_model_file(path))
        .map(|path| mesh_source_from_path(project, path));
    let selected_material = selected_asset
        .as_ref()
        .filter(|path| is_material_file(path))
        .and_then(|path| material_path_from_project(project, path));

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
            if let Some(payload) = scene_asset_button.dnd_release_payload::<AssetDragPayload>() {
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
            if !has_dynamic_body && !has_fixed_collider {
                if ui.button("Dynamic Body (Box)").clicked() {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(DynamicRigidBody { mass: 1.0 });
                    world.entity_mut(entity).insert(ColliderShape::Cuboid);
                    push_undo_snapshot(world, "Add Dynamic Body");
                    ui.close_menu();
                }
                if ui.button("Dynamic Body (Sphere)").clicked() {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(DynamicRigidBody { mass: 1.0 });
                    world.entity_mut(entity).insert(ColliderShape::Sphere);
                    push_undo_snapshot(world, "Add Dynamic Body");
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Fixed Collider (Box)").clicked() {
                    ensure_transform(world, entity);
                    world.entity_mut(entity).insert(FixedCollider);
                    world.entity_mut(entity).insert(ColliderShape::Cuboid);
                    push_undo_snapshot(world, "Add Fixed Collider");
                    ui.close_menu();
                }
                if ui.button("Fixed Collider (Sphere)").clicked() {
                    ensure_transform(world, entity);
                    world.entity_mut(entity).insert(FixedCollider);
                    world.entity_mut(entity).insert(ColliderShape::Sphere);
                    push_undo_snapshot(world, "Add Fixed Collider");
                    ui.close_menu();
                }
            } else if has_dynamic_body {
                ui.label("Dynamic body already present.");
            } else if has_fixed_collider {
                ui.label("Fixed collider already present.");
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
    });
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
    let Some(asset_server) = world.get_resource::<BevyAssetServer>() else {
        return;
    };
    let handle = asset_server.0.lock().load_scene(path);
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

    let candidate = scripts_root.join("script.lua");
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

fn is_model_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "glb" | "gltf"))
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
    !(name.eq_ignore_ascii_case("helmer_project.ron") || name.ends_with(".hscene.ron"))
}

fn is_script_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("lua"))
        .unwrap_or(false)
}

fn push_command(world: &mut World, command: EditorCommand) {
    if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
        queue.push(command);
    }
}

fn set_status(world: &mut World, message: String) {
    if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
        state.status = Some(message);
    }
}

fn set_selection(world: &mut World, entity: Option<Entity>) {
    if let Some(mut selection) = world.get_resource_mut::<InspectorSelectedEntityResource>() {
        selection.0 = entity;
    }
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
