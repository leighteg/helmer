use std::path::PathBuf;

use helmer_becs::ecs::prelude::World;
use helmer_becs::systems::scene_system::EntityParent;
use helmer_becs::{
    ActiveCamera, AudioBackendResource, BecsSystemProfiler, Camera, Light, MeshRenderer, Transform,
    components::LightType,
};
use helmer_editor_runtime::project::load_project_config;

use super::state::{
    AssetBrowserState, EditorCommand, EditorCommandQueue, EditorConsoleLevel, EditorConsoleState,
    EditorProject, EditorProjectLauncherState, EditorSceneState, EditorTimelineState, SpawnKind,
};
use crate::retained::panes::{
    AudioMixerPaneAction, ConsolePaneAction, ConsolePaneLevel, ContentBrowserPaneAction,
    HierarchyPaneAction, HistoryPaneAction, InspectorCameraField, InspectorLightField,
    InspectorLightKind, InspectorPaneAction, InspectorTransformField, MaterialEditorPaneAction,
    ProfilerPaneAction, ProjectPaneAction, TimelinePaneAction, ToolbarPaneAction,
    ViewportPaneAction,
};
use crate::retained::shell::EditorRetainedPaneInteractionState;

const TIMELINE_STEP_SECONDS: f32 = 1.0 / 30.0;

pub fn apply_project_action(world: &mut World, action: ProjectPaneAction) {
    match action {
        ProjectPaneAction::BrowseOpenProject => {
            if let Some(path) = rfd::FileDialog::new().pick_folder() {
                if let Some(mut launcher) = world.get_resource_mut::<EditorProjectLauncherState>() {
                    launcher.open_project_path = path.to_string_lossy().into_owned();
                    launcher.status = None;
                }
                push_command(world, EditorCommand::OpenProject { path });
            }
        }
        ProjectPaneAction::OpenInputPath => {
            let path =
                world
                    .get_resource_mut::<EditorProjectLauncherState>()
                    .map(|mut launcher| {
                        launcher.normalize_inputs();
                        launcher.status = None;
                        launcher.desired_open_path()
                    });
            if let Some(path) = path {
                push_command(world, EditorCommand::OpenProject { path });
            }
        }
        ProjectPaneAction::OpenRecentProject(path) => {
            if let Some(mut launcher) = world.get_resource_mut::<EditorProjectLauncherState>() {
                launcher.open_project_path = path.to_string_lossy().into_owned();
                launcher.status = None;
            }
            push_command(world, EditorCommand::OpenProject { path });
        }
        ProjectPaneAction::BrowseCreateLocation => {
            if let Some(path) = rfd::FileDialog::new().pick_folder()
                && let Some(mut launcher) = world.get_resource_mut::<EditorProjectLauncherState>()
            {
                launcher.project_path = path.to_string_lossy().into_owned();
                launcher.status = None;
            }
        }
        ProjectPaneAction::CreateFromInputs => {
            let request =
                world
                    .get_resource_mut::<EditorProjectLauncherState>()
                    .map(|mut launcher| {
                        launcher.normalize_inputs();
                        launcher.status = None;
                        let name = launcher.desired_create_name();
                        let path = launcher.desired_create_path();
                        (name, path)
                    });
            if let Some((name, path)) = request {
                push_command(world, EditorCommand::CreateProject { name, path });
            }
        }
        ProjectPaneAction::RefreshAssets => {
            if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
                state.refresh_requested = true;
            }
            if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
                console.push(
                    EditorConsoleLevel::Info,
                    "editor.project",
                    "Asset refresh requested",
                );
            }
        }
        ProjectPaneAction::ReloadProjectConfig => {
            let Some(root) = world
                .get_resource::<EditorProject>()
                .and_then(|project| project.root.clone())
            else {
                return;
            };

            let config = load_project_config(&root).ok();

            if let Some(mut project) = world.get_resource_mut::<EditorProject>() {
                project.config = config.clone();
            }

            if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
                browser.root = Some(root.clone());
                if browser.current_dir.is_none()
                    || !browser
                        .current_dir
                        .as_ref()
                        .is_some_and(|path| path.starts_with(&root))
                {
                    browser.current_dir = Some(root.clone());
                }
                browser.refresh_requested = true;
            }

            if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
                if config.is_some() {
                    console.push(
                        EditorConsoleLevel::Info,
                        "editor.project",
                        format!("Reloaded project config from {}", root.to_string_lossy()),
                    );
                } else {
                    console.push(
                        EditorConsoleLevel::Warn,
                        "editor.project",
                        format!(
                            "Project config missing or invalid at {}",
                            root.to_string_lossy()
                        ),
                    );
                }
            }
        }
        ProjectPaneAction::CloseProject => {
            push_command(world, EditorCommand::CloseProject);
        }
    }
}

pub fn apply_toolbar_action(world: &mut World, action: ToolbarPaneAction) {
    match action {
        ToolbarPaneAction::NewScene => push_command(world, EditorCommand::NewScene),
        ToolbarPaneAction::SaveScene => push_command(world, EditorCommand::SaveScene),
        ToolbarPaneAction::TogglePlayMode => push_command(world, EditorCommand::TogglePlayMode),
        ToolbarPaneAction::Undo => push_command(world, EditorCommand::Undo),
        ToolbarPaneAction::Redo => push_command(world, EditorCommand::Redo),
        ToolbarPaneAction::OpenLayoutMenu => {}
    }
}

pub fn apply_viewport_action(world: &mut World, action: ViewportPaneAction) {
    match action {
        ViewportPaneAction::TogglePlayMode { .. } => {
            push_command(world, EditorCommand::TogglePlayMode)
        }
        ViewportPaneAction::OpenCanvasMenu { .. }
        | ViewportPaneAction::OpenRenderMenu { .. }
        | ViewportPaneAction::OpenScriptingMenu { .. }
        | ViewportPaneAction::OpenGizmosMenu { .. }
        | ViewportPaneAction::OpenFreecamMenu { .. }
        | ViewportPaneAction::OpenOrbitMenu { .. }
        | ViewportPaneAction::OpenAdvancedMenu { .. }
        | ViewportPaneAction::PreviewMove { .. }
        | ViewportPaneAction::PreviewResize { .. } => {}
    }
}

pub fn apply_history_action(world: &mut World, action: HistoryPaneAction) {
    match action {
        HistoryPaneAction::Undo => push_command(world, EditorCommand::Undo),
        HistoryPaneAction::Redo => push_command(world, EditorCommand::Redo),
    }
}

pub fn apply_inspector_action(world: &mut World, action: InspectorPaneAction) {
    let mut changed = false;
    match action {
        InspectorPaneAction::SetActiveCamera(entity) => {
            push_command(world, EditorCommand::SetActiveCamera { entity });
        }
        InspectorPaneAction::DeleteEntity(entity) => {
            push_command(world, EditorCommand::DeleteEntity { entity });
        }
        InspectorPaneAction::OpenAddComponentMenu(_entity) => {}
        InspectorPaneAction::RemoveTransform(entity) => {
            world.entity_mut(entity).remove::<Transform>();
            changed = true;
        }
        InspectorPaneAction::RemoveCamera(entity) => {
            world.entity_mut(entity).remove::<Camera>();
            world.entity_mut(entity).remove::<ActiveCamera>();
            changed = true;
        }
        InspectorPaneAction::RemoveLight(entity) => {
            world.entity_mut(entity).remove::<Light>();
            changed = true;
        }
        InspectorPaneAction::RemoveMeshRenderer(entity) => {
            world.entity_mut(entity).remove::<MeshRenderer>();
            changed = true;
        }
        InspectorPaneAction::AdjustTransform {
            entity,
            field,
            delta,
        } => {
            if let Some(mut transform) = world.get_mut::<Transform>(entity) {
                match field {
                    InspectorTransformField::PositionX => transform.position.x += delta,
                    InspectorTransformField::PositionY => transform.position.y += delta,
                    InspectorTransformField::PositionZ => transform.position.z += delta,
                    InspectorTransformField::RotationX => {
                        let (x, y, z) = transform.rotation.to_euler(glam::EulerRot::YXZ);
                        transform.rotation = glam::Quat::from_euler(
                            glam::EulerRot::YXZ,
                            (x.to_degrees() + delta).to_radians(),
                            y,
                            z,
                        );
                    }
                    InspectorTransformField::RotationY => {
                        let (x, y, z) = transform.rotation.to_euler(glam::EulerRot::YXZ);
                        transform.rotation = glam::Quat::from_euler(
                            glam::EulerRot::YXZ,
                            x,
                            (y.to_degrees() + delta).to_radians(),
                            z,
                        );
                    }
                    InspectorTransformField::RotationZ => {
                        let (x, y, z) = transform.rotation.to_euler(glam::EulerRot::YXZ);
                        transform.rotation = glam::Quat::from_euler(
                            glam::EulerRot::YXZ,
                            x,
                            y,
                            (z.to_degrees() + delta).to_radians(),
                        );
                    }
                    InspectorTransformField::ScaleX => {
                        transform.scale.x = (transform.scale.x + delta).max(0.001)
                    }
                    InspectorTransformField::ScaleY => {
                        transform.scale.y = (transform.scale.y + delta).max(0.001)
                    }
                    InspectorTransformField::ScaleZ => {
                        transform.scale.z = (transform.scale.z + delta).max(0.001)
                    }
                }
                changed = true;
            }
        }
        InspectorPaneAction::AdjustCamera {
            entity,
            field,
            delta,
        } => {
            if let Some(mut camera) = world.get_mut::<Camera>(entity) {
                match field {
                    InspectorCameraField::FovDeg => {
                        let next = camera.fov_y_rad.to_degrees() + delta;
                        camera.fov_y_rad = next.clamp(5.0, 175.0).to_radians();
                    }
                    InspectorCameraField::Aspect => {
                        camera.aspect_ratio = (camera.aspect_ratio + delta).clamp(0.1, 8.0);
                    }
                    InspectorCameraField::Near => {
                        camera.near_plane = (camera.near_plane + delta).clamp(0.001, 500.0);
                        if camera.far_plane <= camera.near_plane {
                            camera.far_plane = camera.near_plane + 0.1;
                        }
                    }
                    InspectorCameraField::Far => {
                        camera.far_plane = (camera.far_plane + delta).max(0.01);
                        if camera.far_plane <= camera.near_plane {
                            camera.far_plane = camera.near_plane + 0.1;
                        }
                    }
                }
                changed = true;
            }
        }
        InspectorPaneAction::SetLightType { entity, kind } => {
            if let Some(mut light) = world.get_mut::<Light>(entity) {
                light.light_type = match kind {
                    InspectorLightKind::Directional => LightType::Directional,
                    InspectorLightKind::Point => LightType::Point,
                    InspectorLightKind::Spot => {
                        let angle = match light.light_type {
                            LightType::Spot { angle } => angle,
                            _ => 45.0_f32.to_radians(),
                        };
                        LightType::Spot { angle }
                    }
                };
                changed = true;
            }
        }
        InspectorPaneAction::AdjustLight {
            entity,
            field,
            delta,
        } => {
            if let Some(mut light) = world.get_mut::<Light>(entity) {
                match field {
                    InspectorLightField::Intensity => {
                        light.intensity = (light.intensity + delta).max(0.0);
                    }
                    InspectorLightField::SpotAngleDeg => {
                        let current_deg = match light.light_type {
                            LightType::Spot { angle } => angle.to_degrees(),
                            _ => 45.0,
                        };
                        light.light_type = LightType::Spot {
                            angle: (current_deg + delta).clamp(1.0, 170.0).to_radians(),
                        };
                    }
                }
                changed = true;
            }
        }
        InspectorPaneAction::ToggleMeshCastsShadow(entity) => {
            if let Some(mut mesh) = world.get_mut::<MeshRenderer>(entity) {
                mesh.casts_shadow = !mesh.casts_shadow;
                changed = true;
            }
        }
        InspectorPaneAction::ToggleMeshVisible(entity) => {
            if let Some(mut mesh) = world.get_mut::<MeshRenderer>(entity) {
                mesh.visible = !mesh.visible;
                changed = true;
            }
        }
        InspectorPaneAction::SetTransformValue {
            entity,
            field,
            value,
        } => {
            if let Some(mut transform) = world.get_mut::<Transform>(entity) {
                match field {
                    InspectorTransformField::PositionX => transform.position.x = value,
                    InspectorTransformField::PositionY => transform.position.y = value,
                    InspectorTransformField::PositionZ => transform.position.z = value,
                    InspectorTransformField::RotationX => {
                        let (_, y, z) = transform.rotation.to_euler(glam::EulerRot::YXZ);
                        transform.rotation =
                            glam::Quat::from_euler(glam::EulerRot::YXZ, value.to_radians(), y, z);
                    }
                    InspectorTransformField::RotationY => {
                        let (x, _, z) = transform.rotation.to_euler(glam::EulerRot::YXZ);
                        transform.rotation =
                            glam::Quat::from_euler(glam::EulerRot::YXZ, x, value.to_radians(), z);
                    }
                    InspectorTransformField::RotationZ => {
                        let (x, y, _) = transform.rotation.to_euler(glam::EulerRot::YXZ);
                        transform.rotation =
                            glam::Quat::from_euler(glam::EulerRot::YXZ, x, y, value.to_radians());
                    }
                    InspectorTransformField::ScaleX => transform.scale.x = value.max(0.001),
                    InspectorTransformField::ScaleY => transform.scale.y = value.max(0.001),
                    InspectorTransformField::ScaleZ => transform.scale.z = value.max(0.001),
                }
                changed = true;
            }
        }
        InspectorPaneAction::SetCameraValue {
            entity,
            field,
            value,
        } => {
            if let Some(mut camera) = world.get_mut::<Camera>(entity) {
                match field {
                    InspectorCameraField::FovDeg => {
                        camera.fov_y_rad = value.clamp(5.0, 175.0).to_radians();
                    }
                    InspectorCameraField::Aspect => {
                        camera.aspect_ratio = value.clamp(0.1, 8.0);
                    }
                    InspectorCameraField::Near => {
                        camera.near_plane = value.clamp(0.001, 500.0);
                        if camera.far_plane <= camera.near_plane {
                            camera.far_plane = camera.near_plane + 0.1;
                        }
                    }
                    InspectorCameraField::Far => {
                        camera.far_plane = value.max(0.01);
                        if camera.far_plane <= camera.near_plane {
                            camera.far_plane = camera.near_plane + 0.1;
                        }
                    }
                }
                changed = true;
            }
        }
        InspectorPaneAction::SetLightValue {
            entity,
            field,
            value,
        } => {
            if let Some(mut light) = world.get_mut::<Light>(entity) {
                match field {
                    InspectorLightField::Intensity => {
                        light.intensity = value.max(0.0);
                    }
                    InspectorLightField::SpotAngleDeg => {
                        light.light_type = LightType::Spot {
                            angle: value.clamp(1.0, 170.0).to_radians(),
                        };
                    }
                }
                changed = true;
            }
        }
        InspectorPaneAction::SetLightColor { entity, color } => {
            if let Some(mut light) = world.get_mut::<Light>(entity) {
                light.color = glam::Vec3::new(
                    color[0].clamp(0.0, 1.0),
                    color[1].clamp(0.0, 1.0),
                    color[2].clamp(0.0, 1.0),
                );
                changed = true;
            }
        }
        InspectorPaneAction::ToggleLightColorPicker(_) => {}
    }

    if changed && let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
        scene.dirty = true;
    }
}

pub fn apply_audio_mixer_action(world: &mut World, action: AudioMixerPaneAction) {
    let _ = action;
    if let Some(audio) = world.get_resource::<AudioBackendResource>() {
        let next = !audio.enabled();
        audio.set_enabled(next);
        if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
            console.push(
                EditorConsoleLevel::Info,
                "editor.audio",
                if next {
                    "Audio backend enabled"
                } else {
                    "Audio backend disabled"
                },
            );
        }
    }
}

pub fn apply_profiler_action(world: &mut World, action: ProfilerPaneAction) {
    let Some(profiler) = world.get_resource::<BecsSystemProfiler>() else {
        return;
    };
    match action {
        ProfilerPaneAction::ToggleEnabled => {
            profiler.set_enabled(!profiler.enabled());
        }
        ProfilerPaneAction::Reset => {
            profiler.reset_all();
        }
    }
}

pub fn apply_material_editor_action(world: &mut World, action: MaterialEditorPaneAction) {
    let _ = action;
    let Some(data) = world
        .get_resource::<AssetBrowserState>()
        .and_then(|assets| assets.selected.as_ref())
        .map(|path| path.to_string_lossy().to_string())
    else {
        return;
    };

    if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
        console.push(
            EditorConsoleLevel::Info,
            "editor.material",
            format!("Material selection refreshed: {data}"),
        );
    }
}

pub fn apply_timeline_action(world: &mut World, action: TimelinePaneAction) {
    let Some(mut timeline) = world.get_resource_mut::<EditorTimelineState>() else {
        return;
    };

    let duration = timeline.duration.max(0.0);
    match action {
        TimelinePaneAction::TogglePlayPause => {
            timeline.playing = !timeline.playing;
        }
        TimelinePaneAction::Stop => {
            timeline.playing = false;
            timeline.current_time = 0.0;
        }
        TimelinePaneAction::RewindStep => {
            timeline.current_time = (timeline.current_time - TIMELINE_STEP_SECONDS).max(0.0);
        }
        TimelinePaneAction::ForwardStep => {
            timeline.current_time = (timeline.current_time + TIMELINE_STEP_SECONDS).min(duration);
        }
    }
}

pub fn apply_content_browser_action(
    world: &mut World,
    action: ContentBrowserPaneAction,
    shift_down: bool,
    ctrl_down: bool,
    double_click: bool,
) {
    let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() else {
        return;
    };

    match action {
        ContentBrowserPaneAction::NavigateUp => {
            let Some(root) = state.root.clone() else {
                return;
            };
            let Some(current) = state.current_dir.clone() else {
                return;
            };
            let Some(parent) = current.parent().map(PathBuf::from) else {
                return;
            };
            if !parent.starts_with(&root) {
                return;
            }

            state.current_dir = Some(parent.clone());
            state.selected = Some(parent.clone());
            state.selection_anchor = Some(parent.clone());
            state.selected_paths.clear();
            state.selected_paths.insert(parent.clone());
            state.expanded.insert(parent);
            state.location_dropdown_open = false;
            state.grid_scroll = 0.0;
        }
        ContentBrowserPaneAction::Refresh => {
            state.refresh_requested = true;
        }
        ContentBrowserPaneAction::ToggleLocationDropdown => {
            state.location_dropdown_open = !state.location_dropdown_open;
        }
        ContentBrowserPaneAction::TileSizeSlider => {}
        ContentBrowserPaneAction::SelectLocation(path) => {
            if !path.exists() {
                return;
            }
            state.current_dir = Some(path.clone());
            state.selected = Some(path.clone());
            state.selection_anchor = Some(path.clone());
            state.last_click_path = Some(path.clone());
            state.selected_paths.clear();
            state.selected_paths.insert(path.clone());
            state.expanded.insert(path);
            state.location_dropdown_open = false;
            state.grid_scroll = 0.0;
        }
        ContentBrowserPaneAction::SelectFolder(path) => {
            state.current_dir = Some(path.clone());
            state.selected = Some(path.clone());
            state.selection_anchor = Some(path.clone());
            state.last_click_path = Some(path.clone());
            state.selected_paths.clear();
            state.selected_paths.insert(path.clone());
            state.expanded.insert(path);
            state.location_dropdown_open = false;
            state.grid_scroll = 0.0;
        }
        ContentBrowserPaneAction::ToggleFolderExpanded(path) => {
            if state.expanded.contains(&path) {
                state.expanded.remove(&path);
            } else {
                state.expanded.insert(path);
            }
        }
        ContentBrowserPaneAction::SelectEntry {
            path,
            is_dir,
            index,
        } => {
            select_content_browser_entry(&mut state, path.clone(), index, shift_down, ctrl_down);
            if double_click {
                open_content_browser_entry(&mut state, path, is_dir);
            }
        }
        ContentBrowserPaneAction::GridSurface => {}
        ContentBrowserPaneAction::GridScrollbar => {}
    }
}

pub fn extend_content_browser_drag_selection(world: &mut World, path: PathBuf, additive: bool) {
    let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() else {
        return;
    };

    if !additive {
        state.selected_paths.clear();
    }
    state.selected_paths.insert(path.clone());
    state.selected = Some(path.clone());
    if state.selection_anchor.is_none() {
        state.selection_anchor = Some(path.clone());
    }
    state.last_click_path = Some(path);
}

fn select_content_browser_entry(
    state: &mut AssetBrowserState,
    path: PathBuf,
    index: usize,
    shift_down: bool,
    ctrl_down: bool,
) {
    let visible = visible_grid_paths(state);
    if visible.is_empty() {
        state.selected = Some(path.clone());
        state.selection_anchor = Some(path.clone());
        state.last_click_path = Some(path.clone());
        state.selected_paths.clear();
        state.selected_paths.insert(path);
        return;
    }

    let selected_index = visible
        .get(index)
        .filter(|candidate| **candidate == path)
        .and_then(|_| Some(index))
        .or_else(|| visible.iter().position(|candidate| *candidate == path))
        .unwrap_or(0);

    if shift_down {
        let anchor_path = state
            .selection_anchor
            .clone()
            .or_else(|| state.selected.clone())
            .unwrap_or_else(|| path.clone());
        if let Some(anchor_index) = visible
            .iter()
            .position(|candidate| *candidate == anchor_path)
        {
            let (start, end) = if anchor_index <= selected_index {
                (anchor_index, selected_index)
            } else {
                (selected_index, anchor_index)
            };
            let mut next = if ctrl_down {
                state.selected_paths.clone()
            } else {
                std::collections::HashSet::new()
            };
            for entry in &visible[start..=end] {
                next.insert(entry.clone());
            }
            state.selected_paths = next;
            state.selected = Some(path.clone());
            if state.selection_anchor.is_none() {
                state.selection_anchor = Some(anchor_path);
            }
            state.last_click_path = Some(path);
            return;
        }
    }

    if ctrl_down {
        if state.selected_paths.contains(&path) {
            state.selected_paths.remove(&path);
            if state.selected.as_ref() == Some(&path) {
                state.selected = state.selected_paths.iter().next().cloned();
            }
        } else {
            state.selected_paths.insert(path.clone());
            state.selected = Some(path.clone());
        }
        state.selection_anchor = state.selected.clone();
        state.last_click_path = Some(path);
        return;
    }

    state.selected_paths.clear();
    state.selected_paths.insert(path.clone());
    state.selected = Some(path.clone());
    state.selection_anchor = Some(path.clone());
    state.last_click_path = Some(path);
}

fn open_content_browser_entry(state: &mut AssetBrowserState, path: PathBuf, is_dir: bool) {
    if is_dir {
        state.current_dir = Some(path.clone());
        state.expanded.insert(path.clone());
        state.selected = Some(path.clone());
        state.selection_anchor = Some(path.clone());
        state.selected_paths.clear();
        state.selected_paths.insert(path);
    } else if state.current_dir.is_none() {
        state.current_dir = path.parent().map(PathBuf::from);
    }
}

fn visible_grid_paths(state: &AssetBrowserState) -> Vec<PathBuf> {
    let Some(root) = state.root.as_ref() else {
        return Vec::new();
    };

    let filter = state.filter.trim().to_ascii_lowercase();
    let has_filter = !filter.is_empty();
    let current_dir = state.current_dir.as_ref();

    let mut visible = Vec::new();
    for entry in &state.entries {
        if entry.path == *root {
            continue;
        }
        if let Some(dir) = current_dir {
            if entry.path.parent() != Some(dir.as_path()) {
                continue;
            }
        } else if entry.path.parent() != Some(root.as_path()) {
            continue;
        }

        if has_filter {
            let label = entry
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_ascii_lowercase)
                .unwrap_or_default();
            if !label.contains(&filter) {
                continue;
            }
        }

        visible.push(entry.path.clone());
    }

    visible
}

pub fn apply_hierarchy_action(world: &mut World, action: HierarchyPaneAction) {
    match action {
        HierarchyPaneAction::AddEntity => {
            push_command(
                world,
                EditorCommand::CreateEntity {
                    kind: SpawnKind::Empty,
                },
            );
        }
        HierarchyPaneAction::ToggleExpanded(entity) => {
            if let Some(mut interaction) =
                world.get_resource_mut::<EditorRetainedPaneInteractionState>()
            {
                if !interaction.hierarchy_expanded.insert(entity) {
                    interaction.hierarchy_expanded.remove(&entity);
                }
            }
        }
        HierarchyPaneAction::ListSurface => {}
    }
}

pub fn reparent_hierarchy_entity(
    world: &mut World,
    child: helmer_becs::ecs::entity::Entity,
    new_parent: Option<helmer_becs::ecs::entity::Entity>,
) -> bool {
    if world.get_entity(child).is_err() {
        return false;
    }

    if let Some(parent) = new_parent {
        if parent == child || world.get_entity(parent).is_err() {
            return false;
        }
        if hierarchy_is_descendant_of(world, parent, child) {
            return false;
        }
    }

    let current_parent = world
        .get::<EntityParent>(child)
        .map(|relation| relation.parent);
    if current_parent == new_parent {
        return false;
    }

    let child_transform = world.get::<Transform>(child).copied().unwrap_or_default();

    if let Some(parent) = new_parent {
        let parent_matrix = world
            .get::<Transform>(parent)
            .map(|transform| transform.to_matrix())
            .unwrap_or(glam::Mat4::IDENTITY);
        world.entity_mut(child).insert(EntityParent {
            parent,
            local_transform: parent_matrix.inverse() * child_transform.to_matrix(),
            last_written: child_transform,
        });
    } else {
        world.entity_mut(child).remove::<EntityParent>();
    }

    if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
        scene.dirty = true;
    }
    if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
        console.push(
            EditorConsoleLevel::Info,
            "editor.hierarchy",
            match new_parent {
                Some(parent) => format!(
                    "Reparented entity {} to {}",
                    child.to_bits(),
                    parent.to_bits()
                ),
                None => format!("Unparented entity {}", child.to_bits()),
            },
        );
    }
    true
}

fn hierarchy_is_descendant_of(
    world: &World,
    candidate: helmer_becs::ecs::entity::Entity,
    ancestor: helmer_becs::ecs::entity::Entity,
) -> bool {
    let mut current = Some(candidate);
    let mut visited = std::collections::HashSet::new();
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

pub fn apply_console_action(world: &mut World, action: ConsolePaneAction) {
    let Some(mut state) = world.get_resource_mut::<EditorConsoleState>() else {
        return;
    };

    match action {
        ConsolePaneAction::Clear => state.clear(),
        ConsolePaneAction::ToggleAutoScroll => state.auto_scroll = !state.auto_scroll,
        ConsolePaneAction::ToggleLevel(level) => match level {
            ConsolePaneLevel::Trace => state.show_trace = !state.show_trace,
            ConsolePaneLevel::Debug => state.show_debug = !state.show_debug,
            ConsolePaneLevel::Log => state.show_log = !state.show_log,
            ConsolePaneLevel::Info => state.show_info = !state.show_info,
            ConsolePaneLevel::Warn => state.show_warn = !state.show_warn,
            ConsolePaneLevel::Error => state.show_error = !state.show_error,
        },
        ConsolePaneAction::LogSurface => {}
        ConsolePaneAction::LogScrollbar => {}
    }
}

pub fn push_command(world: &mut World, command: EditorCommand) {
    if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
        queue.push(command);
    }
}
