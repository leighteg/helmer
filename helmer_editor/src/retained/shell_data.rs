use helmer_becs::ecs::{entity::Entity, name::Name, prelude::World};
use helmer_becs::systems::scene_system::{EntityParent, SceneChild, SceneRoot};
use helmer_becs::{
    ActiveCamera, AudioBackendResource, AudioEmitter, AudioListener, BecsSystemProfiler, Camera,
    Light, MeshRenderer, SkinnedMeshRenderer, SpriteRenderer, Text2d, Transform,
    components::LightType, provided::ui::inspector::InspectorSelectedEntityResource,
};
use std::collections::{HashMap, HashSet};

use super::state::{
    AssetBrowserState, EditorConsoleLevel, EditorConsoleState, EditorEntity, EditorProject,
    EditorProjectLauncherState, EditorRetainedViewportStates, EditorSceneState,
    EditorTimelineState, EditorUndoState, RetainedEditorViewportCamera, WorldState,
    is_entry_visible,
};
use crate::retained::panes::{
    AudioMixerPaneData, ConsolePaneData, ConsolePaneEntry, ConsolePaneLevel,
    ContentBrowserPaneData, ContentBrowserPaneEntry, ContentBrowserPaneLocationEntry,
    HierarchyPaneData, HierarchyPaneEntry, HistoryPaneData, HistoryPaneEntry, InspectorLightKind,
    InspectorPaneData, MaterialEditorPaneData, ProfilerPaneData, ProfilerPaneRow, ProjectPaneData,
    TimelinePaneData, ToolbarPaneData, ViewportPaneData, ViewportPaneMode,
    retained_viewport_preview_texture_slot, retained_viewport_texture_slot, rgb_to_hsv,
};
use crate::retained::shell::{EditorRetainedPaneInteractionState, InspectorLightColorPickerState};

pub fn build_project_pane_data(world: &World) -> ProjectPaneData {
    let mut data = ProjectPaneData::default();
    if let Some(project) = world.get_resource::<EditorProject>() {
        data.project_loaded = project.root.is_some();
        if let Some(config) = project.config.as_ref() {
            data.project_name = config.name.clone();
        }
        data.root_path = project
            .root
            .as_ref()
            .map(|path| path.to_string_lossy().to_string());
        data.assets_root = project.root.as_ref().map(|root| {
            project
                .config
                .as_ref()
                .map(|config| config.assets_root(root))
                .unwrap_or_else(|| root.join("assets"))
                .to_string_lossy()
                .to_string()
        });
    }
    if let Some(launcher) = world.get_resource::<EditorProjectLauncherState>() {
        data.open_project_path = launcher.open_project_path.clone();
        data.create_project_name = launcher.project_name.clone();
        data.create_project_location = launcher.project_path.clone();
        data.status = launcher.status.clone();
        data.recent_projects = launcher
            .recent_projects
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect();
    }
    if let Some(interaction) = world.get_resource::<EditorRetainedPaneInteractionState>() {
        data.focused_field = interaction.focused_project_text_field;
        data.text_cursors = interaction.project_text_cursors.clone();
    }
    data
}

pub fn build_toolbar_pane_data(world: &World) -> ToolbarPaneData {
    let project_label = world
        .get_resource::<EditorProject>()
        .and_then(|project| {
            project
                .config
                .as_ref()
                .map(|config| config.name.clone())
                .or_else(|| {
                    project
                        .root
                        .as_ref()
                        .map(|root| root.to_string_lossy().to_string())
                })
        })
        .unwrap_or_else(|| "<none>".to_string());

    let (scene_label, world_label) = world
        .get_resource::<EditorSceneState>()
        .map(|scene| {
            let world_label = match scene.world_state {
                WorldState::Edit => "Edit".to_string(),
                WorldState::Play => "Play".to_string(),
            };
            let scene_label = if scene.dirty {
                format!("{}*", scene.name)
            } else {
                scene.name.clone()
            };
            (scene_label, world_label)
        })
        .unwrap_or_else(|| ("Untitled".to_string(), "Edit".to_string()));

    let (can_undo, can_redo) = world
        .get_resource::<EditorUndoState>()
        .map(|undo| (undo.can_undo(), undo.can_redo()))
        .unwrap_or((false, false));

    ToolbarPaneData {
        project_label,
        scene_label,
        world_label,
        can_undo,
        can_redo,
    }
}

pub fn build_viewport_pane_data(
    world: &World,
    mode: ViewportPaneMode,
    tab_id: u64,
) -> ViewportPaneData {
    let mut entity_count = 0usize;
    let mut camera_count = 0usize;
    let mut active_camera = "<none>".to_string();

    let mut editor_camera_entity: Option<Entity> = None;
    let mut editor_camera_label: Option<String> = None;
    for entity_ref in world.iter_entities() {
        entity_count += 1;
        let entity = entity_ref.id();
        if world.get::<Camera>(entity).is_some() {
            camera_count += 1;
            if world.get::<RetainedEditorViewportCamera>(entity).is_some()
                && editor_camera_entity.is_none()
            {
                editor_camera_entity = Some(entity);
                editor_camera_label = Some(entity_display_label(world, entity));
            }
            if active_camera == "<none>" {
                active_camera = entity_display_label(world, entity);
            }
        }
    }

    if mode == ViewportPaneMode::Edit
        && let Some(label) = editor_camera_label.clone()
    {
        active_camera = label;
    }

    let (scene_name, world_mode, world_state) = world
        .get_resource::<EditorSceneState>()
        .map(|scene| {
            let world_mode = match scene.world_state {
                WorldState::Edit => "Edit".to_string(),
                WorldState::Play => "Play".to_string(),
            };
            (scene.name.clone(), world_mode, scene.world_state)
        })
        .unwrap_or_else(|| ("Untitled".to_string(), "Edit".to_string(), WorldState::Edit));

    let title = match mode {
        ViewportPaneMode::Edit => format!("Editor Viewport ({} cameras)", camera_count),
        ViewportPaneMode::Play => format!("Play Viewport ({} cameras)", camera_count),
    };
    let viewport_state = world
        .get_resource::<EditorRetainedViewportStates>()
        .map(|states| states.state_for(tab_id))
        .unwrap_or_default();
    let play_mode_active = world_state == WorldState::Play;
    let resolution_label = viewport_state.resolution.label().to_string();

    let gameplay_camera_label = world
        .iter_entities()
        .map(|entity_ref| entity_ref.id())
        .filter(|entity| {
            world.get::<Camera>(*entity).is_some()
                && world.get::<Transform>(*entity).is_some()
                && Some(*entity) != editor_camera_entity
                && world.get::<RetainedEditorViewportCamera>(*entity).is_none()
        })
        .max_by_key(|entity| world.get::<ActiveCamera>(*entity).is_some() as u8)
        .map(|entity| entity_display_label(world, entity));

    if mode == ViewportPaneMode::Play && play_mode_active {
        active_camera = gameplay_camera_label
            .or(editor_camera_label)
            .unwrap_or(active_camera);
    } else if mode == ViewportPaneMode::Play {
        active_camera = "play mode inactive".to_string();
    }

    let preview_entity = matches!(mode, ViewportPaneMode::Edit)
        .then(|| {
            world
                .get_resource::<InspectorSelectedEntityResource>()
                .and_then(|selected| selected.0)
                .filter(|entity| {
                    world.get::<Camera>(*entity).is_some()
                        && world.get::<Transform>(*entity).is_some()
                        && Some(*entity) != editor_camera_entity
                })
        })
        .flatten();
    let preview_camera_name = preview_entity.map(|entity| entity_display_label(world, entity));
    let preview_aspect_ratio = preview_entity
        .and_then(|entity| {
            world
                .get::<Camera>(entity)
                .map(|camera| camera.aspect_ratio)
        })
        .filter(|aspect| aspect.is_finite() && *aspect > 0.01)
        .unwrap_or(16.0 / 9.0);
    let preview_texture_id = preview_camera_name
        .as_ref()
        .and_then(|_| retained_viewport_preview_texture_slot(tab_id));

    let should_render_main = matches!(mode, ViewportPaneMode::Edit) || play_mode_active;

    ViewportPaneData {
        tab_id,
        title,
        world_mode,
        play_mode_active,
        scene_name,
        active_camera,
        entity_count,
        resolution_label,
        texture_id: should_render_main
            .then(|| retained_viewport_texture_slot(mode, tab_id))
            .flatten(),
        preview_texture_id,
        preview_camera_name,
        preview_position_norm: viewport_state.preview_position_norm,
        preview_width_norm: viewport_state.preview_width_norm,
        preview_aspect_ratio,
    }
}

pub fn build_history_pane_data(world: &World) -> HistoryPaneData {
    let Some(undo) = world.get_resource::<EditorUndoState>() else {
        return HistoryPaneData::default();
    };

    let entries = undo
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| HistoryPaneEntry {
            label: entry
                .label()
                .map(str::to_string)
                .unwrap_or_else(|| format!("History Entry {}", index + 1)),
            active: index == undo.cursor,
        })
        .collect::<Vec<_>>();

    HistoryPaneData {
        can_undo: undo.can_undo(),
        can_redo: undo.can_redo(),
        undo_label: undo
            .undo_label(|entry| entry.label())
            .map(|value| value.to_string()),
        redo_label: undo
            .redo_label(|entry| entry.label())
            .map(|value| value.to_string()),
        entries,
    }
}

pub fn build_audio_mixer_pane_data(world: &World) -> AudioMixerPaneData {
    let enabled = world
        .get_resource::<AudioBackendResource>()
        .map(|audio| audio.enabled())
        .unwrap_or(false);

    let mut emitter_count = 0usize;
    let mut listener_count = 0usize;
    for entity_ref in world.iter_entities() {
        let entity = entity_ref.id();
        if world.get::<AudioEmitter>(entity).is_some() {
            emitter_count += 1;
        }
        if world.get::<AudioListener>(entity).is_some() {
            listener_count += 1;
        }
    }

    AudioMixerPaneData {
        enabled,
        emitter_count,
        listener_count,
    }
}

pub fn build_profiler_pane_data(world: &World) -> ProfilerPaneData {
    let Some(profiler) = world.get_resource::<BecsSystemProfiler>() else {
        return ProfilerPaneData::default();
    };

    let mut snapshots = profiler.snapshots();
    snapshots.sort_by(|lhs, rhs| rhs.avg_us.total_cmp(&lhs.avg_us));

    let rows = snapshots
        .into_iter()
        .take(18)
        .map(|snapshot| ProfilerPaneRow {
            label: snapshot.name.to_string(),
            value: format!(
                "avg {:.3}ms | last {:.3}ms | calls {}",
                snapshot.avg_us / 1000.0,
                snapshot.last_us as f64 / 1000.0,
                snapshot.calls
            ),
        })
        .collect::<Vec<_>>();

    ProfilerPaneData {
        enabled: profiler.enabled(),
        rows,
    }
}

pub fn build_material_editor_pane_data(world: &World) -> MaterialEditorPaneData {
    let selected_asset = world
        .get_resource::<AssetBrowserState>()
        .and_then(|assets| assets.selected.as_ref())
        .map(|path| path.to_string_lossy().to_string());

    let status = selected_asset.as_ref().map(|path| {
        if path.to_ascii_lowercase().ends_with(".material")
            || path.to_ascii_lowercase().ends_with(".mat")
        {
            "Material asset selected".to_string()
        } else {
            "Selected asset is not a material".to_string()
        }
    });

    MaterialEditorPaneData {
        selected_asset,
        status,
    }
}

pub fn build_hierarchy_pane_data(world: &World) -> HierarchyPaneData {
    let selected = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0);
    let (expanded_entities, hierarchy_scroll) = world
        .get_resource::<EditorRetainedPaneInteractionState>()
        .map(|state| (state.hierarchy_expanded.clone(), state.hierarchy_scroll))
        .unwrap_or_default();

    let entities: Vec<Entity> = world
        .iter_entities()
        .map(|entity_ref| entity_ref.id())
        .filter(|entity| {
            world.get::<EditorEntity>(*entity).is_some()
                || world.get::<SceneChild>(*entity).is_some()
                || world.get::<SceneRoot>(*entity).is_some()
        })
        .collect();
    let included: HashSet<Entity> = entities.iter().copied().collect();
    let mut children: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut roots = Vec::new();

    for entity in entities.iter().copied() {
        let parent = world
            .get::<EntityParent>(entity)
            .map(|relation| relation.parent);
        let valid_parent = parent.filter(|parent| *parent != entity && included.contains(parent));
        if let Some(parent) = valid_parent {
            children.entry(parent).or_default().push(entity);
        } else {
            roots.push(entity);
        }
    }

    roots.sort_by_key(|entity| entity.to_bits());
    for child_list in children.values_mut() {
        child_list.sort_by_key(|entity| entity.to_bits());
    }

    let mut entries = Vec::new();
    let mut stack = roots
        .into_iter()
        .rev()
        .map(|entity| (entity, 0usize))
        .collect::<Vec<_>>();
    while let Some((entity, depth)) = stack.pop() {
        let has_children = children
            .get(&entity)
            .is_some_and(|children| !children.is_empty());
        let is_expanded = expanded_entities.contains(&entity);
        entries.push(HierarchyPaneEntry {
            entity,
            label: hierarchy_entity_label(world, entity),
            depth,
            has_children,
            expanded: is_expanded,
        });
        if entries.len() >= 256 {
            break;
        }
        if has_children
            && is_expanded
            && let Some(child_list) = children.get(&entity)
        {
            for child in child_list.iter().rev().copied() {
                stack.push((child, depth.saturating_add(1)));
            }
        }
    }

    HierarchyPaneData {
        selected,
        entries,
        scroll: hierarchy_scroll,
    }
}

pub fn build_inspector_pane_data(
    world: &World,
    picker_state: Option<InspectorLightColorPickerState>,
) -> InspectorPaneData {
    let selected = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selected| selected.0);

    let Some(entity) = selected else {
        return InspectorPaneData::default();
    };

    let mut data = InspectorPaneData {
        entity: Some(entity),
        entity_label: entity_display_label(world, entity),
        name_value: world
            .get::<Name>(entity)
            .map(|name| name.as_str().to_string())
            .unwrap_or_default(),
        ..InspectorPaneData::default()
    };

    if let Some(transform) = world.get::<Transform>(entity) {
        let transform = *transform;
        data.has_transform = true;
        data.transform_position = [
            transform.position.x,
            transform.position.y,
            transform.position.z,
        ];
        let (rot_x, rot_y, rot_z) = transform.rotation.to_euler(glam::EulerRot::YXZ);
        data.transform_rotation = [rot_x.to_degrees(), rot_y.to_degrees(), rot_z.to_degrees()];
        data.transform_scale = [transform.scale.x, transform.scale.y, transform.scale.z];
    }
    if let Some(camera) = world.get::<Camera>(entity) {
        let camera = *camera;
        data.has_camera = true;
        data.camera_active = world.get::<helmer_becs::ActiveCamera>(entity).is_some();
        data.camera_fov_deg = camera.fov_y_rad.to_degrees();
        data.camera_aspect_ratio = camera.aspect_ratio;
        data.camera_near = camera.near_plane;
        data.camera_far = camera.far_plane;
    }
    if let Some(light) = world.get::<Light>(entity) {
        let light = *light;
        data.has_light = true;
        data.light_kind = match light.light_type {
            LightType::Directional => InspectorLightKind::Directional,
            LightType::Point => InspectorLightKind::Point,
            LightType::Spot { .. } => InspectorLightKind::Spot,
        };
        data.light_color = [light.color.x, light.color.y, light.color.z];
        data.light_color_hsv = rgb_to_hsv(data.light_color);
        if let Some(picker) = picker_state.filter(|picker| picker.entity == entity) {
            data.light_color_picker_open = true;
            if data.light_color_hsv[1] <= 0.0001 {
                data.light_color_hsv[0] = picker.hue.rem_euclid(1.0);
            }
        }
        data.light_intensity = light.intensity;
        data.light_spot_angle_deg = match light.light_type {
            LightType::Spot { angle } => angle.to_degrees(),
            _ => 45.0,
        };
    }
    if let Some(mesh_renderer) = world.get::<MeshRenderer>(entity) {
        let mesh_renderer = *mesh_renderer;
        data.has_mesh_renderer = true;
        data.mesh_id = mesh_renderer.mesh_id;
        data.material_id = mesh_renderer.material_id;
        data.mesh_casts_shadow = mesh_renderer.casts_shadow;
        data.mesh_visible = mesh_renderer.visible;
    }

    if let Some(interaction) = world.get_resource::<EditorRetainedPaneInteractionState>() {
        data.focused_field = interaction.focused_inspector_text_field;
        data.text_cursors = interaction.inspector_text_cursors.clone();
    }

    data
}

pub fn build_timeline_pane_data(world: &World) -> TimelinePaneData {
    let Some(timeline) = world.get_resource::<EditorTimelineState>() else {
        return TimelinePaneData::default();
    };

    TimelinePaneData {
        playing: timeline.playing,
        current_time: timeline.current_time,
        duration: timeline.duration,
        playback_rate: timeline.playback_rate,
        track_group_count: timeline.groups.len(),
        selected_key_count: timeline.selected.len(),
    }
}

pub fn build_content_browser_pane_data(world: &World) -> ContentBrowserPaneData {
    let Some(state) = world.get_resource::<AssetBrowserState>() else {
        return ContentBrowserPaneData::default();
    };

    let root = state.root.as_ref();
    let current_dir = state.current_dir.as_ref();

    let mut data = ContentBrowserPaneData {
        root_path: root.map(|path| content_browser_display_path(path, path)),
        current_dir: root
            .zip(current_dir)
            .map(|(root, path)| content_browser_display_path(root, path))
            .or_else(|| current_dir.map(|path| path.to_string_lossy().to_string())),
        filter: state.filter.clone(),
        status: state.status.clone(),
        sidebar_entries: Vec::new(),
        grid_entries: Vec::new(),
        location_entries: Vec::new(),
        can_navigate_up: content_browser_can_navigate_up(state),
        location_dropdown_open: state.location_dropdown_open,
        grid_scroll: state.grid_scroll,
        tile_size: state.tile_size,
        focused_field: None,
        text_cursors: HashMap::new(),
    };

    let Some(root) = root else {
        return data;
    };
    let filter = state.filter.trim().to_ascii_lowercase();
    let has_filter = !filter.is_empty();

    for entry in state.entries.iter().filter(|entry| entry.is_dir) {
        if entry.path != *root && content_browser_path_is_hidden(root, &entry.path) {
            continue;
        }
        if entry.path != *root && !is_entry_visible(entry, root, &state.expanded) {
            continue;
        }

        let label = if entry.path == *root {
            content_browser_display_path(root, &entry.path)
        } else {
            entry
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
                .unwrap_or_else(|| entry.path.to_string_lossy().to_string())
        };
        let depth = entry.depth;
        let selected = state.current_dir.as_ref() == Some(&entry.path);
        let has_children = state.entries.iter().any(|candidate| {
            candidate.is_dir && candidate.path.parent() == Some(entry.path.as_path())
        });

        data.sidebar_entries.push(ContentBrowserPaneEntry {
            path: entry.path.clone(),
            label,
            depth,
            is_dir: true,
            selected,
            has_children,
            expanded: state.expanded.contains(&entry.path),
        });
    }

    let active_dir = current_dir.cloned().unwrap_or_else(|| root.clone());
    for path in content_browser_location_entries(root, &active_dir) {
        let label = content_browser_display_path(root, &path);
        data.location_entries.push(ContentBrowserPaneLocationEntry {
            selected: path == active_dir,
            path,
            label,
        });
    }

    for entry in state.entries.iter() {
        if entry.path == *root {
            continue;
        }
        if content_browser_path_is_hidden(root, &entry.path) {
            continue;
        }
        if let Some(dir) = current_dir {
            if entry.path.parent() != Some(dir.as_path()) {
                continue;
            }
        } else if entry.path.parent() != Some(root.as_path()) {
            continue;
        }

        let label = entry
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| entry.path.to_string_lossy().to_string());
        if has_filter && !label.to_ascii_lowercase().contains(&filter) {
            continue;
        }

        let selected = state.selected.as_ref() == Some(&entry.path)
            || state.selected_paths.contains(&entry.path);
        data.grid_entries.push(ContentBrowserPaneEntry {
            path: entry.path.clone(),
            label,
            depth: 0,
            is_dir: entry.is_dir,
            selected,
            has_children: false,
            expanded: false,
        });
    }

    if let Some(interaction) = world.get_resource::<EditorRetainedPaneInteractionState>() {
        data.focused_field = interaction.focused_content_browser_text_field;
        data.text_cursors = interaction.content_browser_text_cursors.clone();
    }

    data
}

fn content_browser_location_entries(
    root: &std::path::Path,
    current: &std::path::Path,
) -> Vec<std::path::PathBuf> {
    if !current.starts_with(root) {
        return vec![root.to_path_buf()];
    }

    let mut paths = Vec::new();
    let mut cursor = Some(current);
    while let Some(path) = cursor {
        paths.push(path.to_path_buf());
        if path == root {
            break;
        }
        cursor = path.parent();
    }
    paths.reverse();
    paths
}

fn content_browser_display_path(root: &std::path::Path, path: &std::path::Path) -> String {
    if !path.starts_with(root) {
        return path.to_string_lossy().to_string();
    }

    let root_label = root
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| root.to_string_lossy().to_string());

    if path == root {
        return root_label;
    }

    let relative = path
        .strip_prefix(root)
        .ok()
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|| path.to_string_lossy().to_string());
    if relative.is_empty() {
        root_label
    } else {
        format!("{root_label}/{relative}")
    }
}

fn content_browser_path_is_hidden(root: &std::path::Path, path: &std::path::Path) -> bool {
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

pub fn build_console_pane_data(world: &World) -> ConsolePaneData {
    let Some(state) = world.get_resource::<EditorConsoleState>() else {
        return ConsolePaneData::default();
    };

    let mut entries = state
        .entries
        .iter()
        .rev()
        .take(240)
        .map(|entry| ConsolePaneEntry {
            sequence: entry.sequence,
            level: map_console_level(entry.level),
            target: entry.target.clone(),
            message: entry.message.clone(),
        })
        .collect::<Vec<_>>();
    entries.reverse();

    let mut data = ConsolePaneData {
        auto_scroll: state.auto_scroll,
        show_trace: state.show_trace,
        show_debug: state.show_debug,
        show_log: state.show_log,
        show_info: state.show_info,
        show_warn: state.show_warn,
        show_error: state.show_error,
        search: state.search.clone(),
        entries,
        scroll: state.scroll,
        focused_field: None,
        text_cursors: HashMap::new(),
    };
    if let Some(interaction) = world.get_resource::<EditorRetainedPaneInteractionState>() {
        data.focused_field = interaction.focused_console_text_field;
        data.text_cursors = interaction.console_text_cursors.clone();
    }
    data
}

fn content_browser_can_navigate_up(state: &AssetBrowserState) -> bool {
    let Some(root) = state.root.as_ref() else {
        return false;
    };
    let Some(current) = state.current_dir.as_ref() else {
        return false;
    };
    current != root
        && current
            .parent()
            .is_some_and(|parent| parent == root || parent.starts_with(root))
}

fn map_console_level(level: EditorConsoleLevel) -> ConsolePaneLevel {
    match level {
        EditorConsoleLevel::Trace => ConsolePaneLevel::Trace,
        EditorConsoleLevel::Debug => ConsolePaneLevel::Debug,
        EditorConsoleLevel::Log => ConsolePaneLevel::Log,
        EditorConsoleLevel::Info => ConsolePaneLevel::Info,
        EditorConsoleLevel::Warn => ConsolePaneLevel::Warn,
        EditorConsoleLevel::Error => ConsolePaneLevel::Error,
    }
}

fn entity_display_label(world: &World, entity: Entity) -> String {
    if let Some(name) = world.get::<Name>(entity) {
        return format!("{} ({})", name.as_str(), entity.to_bits());
    }
    format!("Entity {}", entity.to_bits())
}

fn hierarchy_entity_label(world: &World, entity: Entity) -> String {
    let mut label = world
        .get::<Name>(entity)
        .map(|name| name.as_str().to_string())
        .unwrap_or_else(|| format!("Entity {}", entity.to_bits()));

    let mut tags = Vec::new();
    if world.get::<Camera>(entity).is_some() {
        if world.get::<ActiveCamera>(entity).is_some() {
            tags.push("Camera*");
        } else {
            tags.push("Camera");
        }
    }
    if world.get::<Light>(entity).is_some() {
        tags.push("Light");
    }
    if world.get::<MeshRenderer>(entity).is_some() {
        tags.push("Mesh");
    }
    if world.get::<SkinnedMeshRenderer>(entity).is_some() {
        tags.push("Skinned");
    }
    if world.get::<SpriteRenderer>(entity).is_some() {
        tags.push("Sprite");
    }
    if world.get::<Text2d>(entity).is_some() {
        tags.push("Text2D");
    }
    if world.get::<SceneRoot>(entity).is_some() {
        tags.push("Scene");
    }
    if world.get::<SceneChild>(entity).is_some() {
        tags.push("Scene Child");
    }

    if !tags.is_empty() {
        label.push_str(" [");
        label.push_str(&tags.join(", "));
        label.push(']');
    }
    label
}
