use std::{
    fs,
    path::{Path, PathBuf},
};

use bevy_ecs::prelude::{With, World};
use bevy_ecs::{entity::Entity, name::Name};
use helmer::provided::components::{ActiveCamera, Light, Transform};
use helmer_becs::physics::components::{ColliderShape, DynamicRigidBody, FixedCollider};
use helmer_becs::physics::physics_resource::PhysicsResource;
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::scene_system::{EntityParent, SceneChild, SceneRoot};
use helmer_becs::{
    BevyActiveCamera, BevyCamera, BevyLight, BevyMeshRenderer, BevySkinnedMeshRenderer,
    BevySpriteRenderer, BevyText2d, BevyTransform, BevyWrapper, DeltaTime,
};
use helmer_editor_runtime::{
    file_watch::poll_file_watcher,
    project::{create_project, load_project},
    scene_state::scene_display_name,
    script_registry::update_script_registry,
};

use crate::retained::shell::EditorRetainedPaneInteractionState;
use crate::retained::state::{
    AssetBrowserState, AssetCreateKind, EditorCommand, EditorCommandQueue, EditorConsoleLevel,
    EditorConsoleState, EditorEntity, EditorProject, EditorProjectLauncherState, EditorSceneState,
    EditorTimelineState, EditorUndoState, FileWatchState, RetainedPlayBackup,
    RetainedPlayEntitySnapshot, ScriptRegistry, SpawnKind, WorldState, apply_scene_undo_snapshot,
    configure_file_watcher, drain_runtime_log_entries, ensure_default_pane_workspace,
    initialize_editor_project, push_scene_undo_entry, record_recent_project, refresh_asset_browser,
    reset_scene_undo_history,
};
use crate::retained::workspace::EditorPaneWorkspaceState;

pub fn retained_workspace_seed_system(world: &mut World) {
    if let Some(mut workspace) = world.get_resource_mut::<EditorPaneWorkspaceState>() {
        ensure_default_pane_workspace(&mut workspace);
    }
}

pub fn retained_asset_browser_refresh_system(world: &mut World) {
    let should_refresh = world
        .get_resource::<AssetBrowserState>()
        .map(|state| state.refresh_requested || state.last_scan.elapsed() >= state.scan_interval)
        .unwrap_or(false);

    if !should_refresh {
        return;
    }

    if let Some(mut state) = world.get_resource_mut::<AssetBrowserState>() {
        refresh_asset_browser(&mut state);
    }
}

pub fn retained_console_runtime_log_system(world: &mut World) {
    let drained = drain_runtime_log_entries();
    if drained.is_empty() {
        return;
    }

    if let Some(mut state) = world.get_resource_mut::<EditorConsoleState>() {
        for entry in drained {
            state.push(
                EditorConsoleLevel::from_runtime_level(entry.level),
                entry.target,
                entry.message,
            );
        }
    }
}

pub fn retained_file_watch_system(world: &mut World) {
    let Some((project_root, project_config, assets_root, scripts_root, config_path)) =
        world.get_resource::<EditorProject>().and_then(|project| {
            project.root.as_ref().map(|root| {
                let config = project.config.clone();
                let assets_root = config
                    .as_ref()
                    .map(|cfg| cfg.assets_root(root))
                    .unwrap_or_else(|| root.join("assets"));
                let scripts_root = config
                    .as_ref()
                    .map(|cfg| cfg.scripts_root(root))
                    .unwrap_or_else(|| assets_root.join("scripts"));
                let config_path = helmer_editor_runtime::project::ProjectConfig::config_path(root);
                (root.clone(), config, assets_root, scripts_root, config_path)
            })
        })
    else {
        return;
    };

    let should_configure_watcher = world
        .get_resource::<FileWatchState>()
        .map(|state| state.root.as_ref() != Some(&project_root) || state.receiver.is_none())
        .unwrap_or(false);

    if should_configure_watcher {
        if let Some(mut watch) = world.get_resource_mut::<FileWatchState>() {
            configure_file_watcher(&mut watch, &project_root, project_config.as_ref());
        }
    }

    let poll = world
        .get_resource_mut::<FileWatchState>()
        .map(|mut watch| poll_file_watcher(&mut watch, &assets_root, &scripts_root, &config_path));

    let Some(poll) = poll else {
        return;
    };

    if poll.assets_changed {
        if let Some(mut assets) = world.get_resource_mut::<AssetBrowserState>() {
            assets.refresh_requested = true;
        }
    }

    if !poll.script_paths.is_empty() {
        if let Some(mut scripts) = world.get_resource_mut::<ScriptRegistry>() {
            scripts.mark_dirty_paths_owned(poll.script_paths);
        }
    }

    if let Some(status) = poll.status_update {
        if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
            let level = status_level(&status);
            console.push(level, "editor.watch", status);
        }
    }
}

pub fn retained_script_registry_system(world: &mut World) {
    let Some((project_root, project_config)) =
        world.get_resource::<EditorProject>().and_then(|project| {
            project
                .root
                .as_ref()
                .map(|root| (root.clone(), project.config.clone()))
        })
    else {
        return;
    };

    let Some(mut registry) = world.get_resource_mut::<ScriptRegistry>() else {
        return;
    };

    update_script_registry(&mut registry, &project_root, project_config.as_ref());
}

pub fn retained_command_system(world: &mut World) {
    let commands = {
        let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() else {
            return;
        };
        std::mem::take(&mut queue.commands)
    };

    for command in commands {
        match command {
            EditorCommand::CreateProject { name, path } => match create_project(&path, &name) {
                Ok(_config) => {
                    initialize_editor_project(world, path.clone());
                    reset_editor_scene_entities(world);
                    spawn_default_scene_entities(world);
                    sync_recent_project(world, &path);
                    update_launcher_after_project_open(world, &path, None);
                    set_console_status(
                        world,
                        EditorConsoleLevel::Info,
                        format!("Project created at {}", path.to_string_lossy()),
                    );
                }
                Err(err) => {
                    set_launcher_status(world, format!("Failed to create project: {err}"));
                    set_console_status(
                        world,
                        EditorConsoleLevel::Error,
                        format!("Failed to create project: {err}"),
                    );
                }
            },
            EditorCommand::OpenProject { path } => match load_project(&path) {
                Ok(_config) => {
                    initialize_editor_project(world, path.clone());
                    reset_editor_scene_entities(world);
                    spawn_default_scene_entities(world);
                    sync_recent_project(world, &path);
                    update_launcher_after_project_open(world, &path, None);
                    set_console_status(
                        world,
                        EditorConsoleLevel::Info,
                        format!("Project opened at {}", path.to_string_lossy()),
                    );
                }
                Err(err) => {
                    set_launcher_status(world, format!("Failed to open project: {err}"));
                    set_console_status(
                        world,
                        EditorConsoleLevel::Error,
                        format!("Failed to open project: {err}"),
                    );
                }
            },
            EditorCommand::CloseProject => {
                close_project(world);
            }
            EditorCommand::NewScene => {
                reset_editor_scene_entities(world);
                spawn_default_scene_entities(world);
                if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                    scene.path = None;
                    scene.name = "Untitled".to_string();
                    scene.dirty = false;
                    scene.world_state = WorldState::Edit;
                }
                push_scene_undo_entry(world, "New Scene");
            }
            EditorCommand::OpenScene { path } => {
                if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                    scene.path = Some(path.clone());
                    scene.name = scene_display_name(&path);
                    scene.dirty = false;
                    scene.world_state = WorldState::Edit;
                }
                push_scene_undo_entry(world, "Open Scene");
            }
            EditorCommand::SaveScene => {
                let default_path = default_scene_save_path(world);
                if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                    if scene.path.is_none() {
                        scene.path = default_path;
                    }
                    scene.dirty = false;
                }
                push_scene_undo_entry(world, "Save Scene");
            }
            EditorCommand::SaveSceneAs { path } => {
                if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                    scene.path = Some(path.clone());
                    scene.name = scene_display_name(&path);
                    scene.dirty = false;
                }
                push_scene_undo_entry(world, "Save Scene As");
            }
            EditorCommand::TogglePlayMode => {
                let world_state = world
                    .get_resource::<EditorSceneState>()
                    .map(|scene| scene.world_state)
                    .unwrap_or(WorldState::Edit);
                match world_state {
                    WorldState::Edit => {
                        let backup = capture_play_backup(world);
                        let selected_index = backup.selected_index;
                        if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                            scene.play_backup = Some(backup);
                            scene.play_selected_index = selected_index;
                            scene.world_state = WorldState::Play;
                        }
                    }
                    WorldState::Play => {
                        let backup =
                            if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
                                scene.world_state = WorldState::Edit;
                                scene.play_selected_index = None;
                                scene.play_backup.take()
                            } else {
                                None
                            };
                        if let Some(backup) = backup {
                            restore_play_backup(world, backup);
                        }
                        if let Some(mut phys) = world.get_resource_mut::<PhysicsResource>() {
                            phys.running = false;
                        }
                    }
                }
            }
            EditorCommand::Undo => {
                let label = apply_undo(world);
                set_console_status(
                    world,
                    EditorConsoleLevel::Info,
                    label.unwrap_or_else(|| "Nothing to undo".to_string()),
                );
            }
            EditorCommand::Redo => {
                let label = apply_redo(world);
                set_console_status(
                    world,
                    EditorConsoleLevel::Info,
                    label.unwrap_or_else(|| "Nothing to redo".to_string()),
                );
            }
            EditorCommand::CreateEntity { kind } => {
                let label = create_entity(world, kind);
                if let Some(label) = label {
                    push_scene_undo_entry(world, format!("Create {label}"));
                    mark_scene_dirty(world);
                    set_console_status(
                        world,
                        EditorConsoleLevel::Info,
                        format!("Created entity: {label}"),
                    );
                }
            }
            EditorCommand::DeleteEntity { entity } => {
                if delete_entity(world, entity) {
                    push_scene_undo_entry(world, "Delete Entity");
                    mark_scene_dirty(world);
                    set_console_status(
                        world,
                        EditorConsoleLevel::Info,
                        format!("Deleted entity {}", entity.to_bits()),
                    );
                } else {
                    set_console_status(
                        world,
                        EditorConsoleLevel::Warn,
                        "Delete failed: entity no longer exists",
                    );
                }
            }
            EditorCommand::SetActiveCamera { entity } => {
                if set_active_camera(world, entity) {
                    push_scene_undo_entry(world, "Set Active Camera");
                    mark_scene_dirty(world);
                    set_console_status(
                        world,
                        EditorConsoleLevel::Info,
                        format!("Active camera set to entity {}", entity.to_bits()),
                    );
                } else {
                    set_console_status(world, EditorConsoleLevel::Warn, "Set active camera failed");
                }
            }
            EditorCommand::ImportAsset {
                source_path,
                destination_dir,
            } => match import_asset(world, &source_path, destination_dir.as_deref()) {
                Ok(path) => set_console_status(
                    world,
                    EditorConsoleLevel::Info,
                    format!("Imported asset to {}", path.to_string_lossy()),
                ),
                Err(err) => {
                    set_console_status(world, EditorConsoleLevel::Error, err);
                }
            },
            EditorCommand::CreateAsset {
                directory,
                name,
                kind,
            } => match create_asset(world, &directory, &name, kind) {
                Ok(path) => {
                    set_console_status(
                        world,
                        EditorConsoleLevel::Info,
                        format!("Created asset {}", path.to_string_lossy()),
                    );
                    if let Some(mut assets) = world.get_resource_mut::<AssetBrowserState>() {
                        assets.refresh_requested = true;
                        assets.current_dir = Some(directory.clone());
                        assets.selected = Some(path.clone());
                        assets.selection_anchor = Some(path.clone());
                        assets.selected_paths.clear();
                        assets.selected_paths.insert(path);
                    }
                }
                Err(err) => {
                    set_console_status(world, EditorConsoleLevel::Error, err);
                }
            },
        }
    }
}

pub fn retained_timeline_playback_system(world: &mut World) {
    let delta_seconds = world
        .get_resource::<DeltaTime>()
        .map(|dt| dt.0)
        .unwrap_or(0.0)
        .max(0.0);

    let Some(mut timeline) = world.get_resource_mut::<EditorTimelineState>() else {
        return;
    };

    if !timeline.playing || delta_seconds <= 0.0 {
        return;
    }

    let playback_rate = timeline.playback_rate.max(0.0);
    timeline.current_time += delta_seconds * playback_rate;

    if timeline.duration <= 0.0 {
        return;
    }

    if timeline.current_time >= timeline.duration {
        timeline.current_time = timeline.duration;
        timeline.playing = false;
    }
}

pub fn retained_physics_state_system(world: &mut World) {
    let should_run = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state == WorldState::Play)
        .unwrap_or(false);

    if let Some(mut phys) = world.get_resource_mut::<PhysicsResource>()
        && phys.running != should_run
    {
        phys.running = should_run;
    }
}

fn mark_scene_dirty(world: &mut World) {
    if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
        scene.dirty = true;
    }
}

fn reset_editor_scene_entities(world: &mut World) {
    let editor_entities = world
        .query_filtered::<Entity, With<EditorEntity>>()
        .iter(world)
        .collect::<Vec<_>>();
    for entity in editor_entities {
        let _ = world.despawn(entity);
    }

    let scene_children = world
        .query_filtered::<Entity, With<SceneChild>>()
        .iter(world)
        .collect::<Vec<_>>();
    for entity in scene_children {
        let _ = world.despawn(entity);
    }

    let scene_roots = world
        .query_filtered::<Entity, With<SceneRoot>>()
        .iter(world)
        .collect::<Vec<_>>();
    for entity in scene_roots {
        let _ = world.despawn(entity);
    }

    if let Some(mut selection) = world.get_resource_mut::<InspectorSelectedEntityResource>() {
        selection.0 = None;
    }

    if let Some(mut timeline) = world.get_resource_mut::<EditorTimelineState>() {
        timeline.playing = false;
        timeline.current_time = 0.0;
        timeline.groups.clear();
        timeline.selected.clear();
    }
}

fn spawn_default_scene_entities(world: &mut World) {
    world.spawn((
        EditorEntity,
        BevyTransform::default(),
        BevyCamera::default(),
        BevyWrapper(ActiveCamera {}),
        Name::new("Scene Camera"),
    ));

    let light_entity = world
        .spawn((
            EditorEntity,
            BevyWrapper(Transform {
                position: glam::Vec3::new(0.0, 4.0, 0.0),
                rotation: glam::Quat::from_euler(
                    glam::EulerRot::YXZ,
                    20.0f32.to_radians(),
                    -50.0f32.to_radians(),
                    20.0f32.to_radians(),
                ),
                scale: glam::Vec3::ONE,
            }),
            BevyWrapper(Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0)),
            Name::new("Directional Light"),
        ))
        .id();

    if let Some(mut interaction) = world.get_resource_mut::<EditorRetainedPaneInteractionState>() {
        interaction.hierarchy_expanded.insert(light_entity);
    }
}

fn capture_play_backup(world: &World) -> RetainedPlayBackup {
    let entities = world
        .iter_entities()
        .map(|entity_ref| entity_ref.id())
        .filter(|entity| {
            world.get::<EditorEntity>(*entity).is_some()
                || world.get::<SceneChild>(*entity).is_some()
                || world.get::<SceneRoot>(*entity).is_some()
        })
        .collect::<Vec<_>>();
    let included = entities
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let selected_entity = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selection| selection.0);
    let selected_index =
        selected_entity.and_then(|entity| entities.iter().position(|id| *id == entity));

    let entities = entities
        .into_iter()
        .map(|entity| RetainedPlayEntitySnapshot {
            source: entity,
            name: world
                .get::<Name>(entity)
                .map(|name| name.as_str().to_string()),
            editor_entity: world.get::<EditorEntity>(entity).is_some(),
            transform: world.get::<BevyTransform>(entity).copied(),
            camera: world.get::<BevyCamera>(entity).copied(),
            active_camera: world.get::<BevyActiveCamera>(entity).is_some(),
            light: world.get::<BevyLight>(entity).copied(),
            mesh_renderer: world.get::<BevyMeshRenderer>(entity).copied(),
            skinned_mesh_renderer: world.get::<BevySkinnedMeshRenderer>(entity).cloned(),
            sprite_renderer: world.get::<BevySpriteRenderer>(entity).copied(),
            text_2d: world.get::<BevyText2d>(entity).cloned(),
            dynamic_body: world.get::<DynamicRigidBody>(entity).copied(),
            fixed_collider: world.get::<FixedCollider>(entity).is_some(),
            collider_shape: world.get::<ColliderShape>(entity).copied(),
            parent: world
                .get::<EntityParent>(entity)
                .map(|relation| relation.parent)
                .filter(|parent| *parent != entity && included.contains(parent)),
        })
        .collect::<Vec<_>>();

    RetainedPlayBackup {
        entities,
        selected_index,
    }
}

fn restore_play_backup(world: &mut World, backup: RetainedPlayBackup) {
    reset_editor_scene_entities(world);
    if backup.entities.is_empty() {
        if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>() {
            selected.0 = None;
        }
        return;
    }

    let source_to_index = backup
        .entities
        .iter()
        .enumerate()
        .map(|(index, snapshot)| (snapshot.source, index))
        .collect::<std::collections::HashMap<_, _>>();

    let mut spawned = Vec::with_capacity(backup.entities.len());
    for snapshot in &backup.entities {
        let entity = world.spawn_empty().id();
        if snapshot.editor_entity {
            world.entity_mut(entity).insert(EditorEntity);
        }
        if let Some(name) = snapshot.name.as_ref() {
            world.entity_mut(entity).insert(Name::new(name.clone()));
        }
        if let Some(transform) = snapshot.transform {
            world.entity_mut(entity).insert(transform);
        }
        if let Some(camera) = snapshot.camera {
            world.entity_mut(entity).insert(camera);
        }
        if snapshot.active_camera {
            world
                .entity_mut(entity)
                .insert(BevyWrapper(ActiveCamera {}));
        }
        if let Some(light) = snapshot.light {
            world.entity_mut(entity).insert(light);
        }
        if let Some(mesh_renderer) = snapshot.mesh_renderer {
            world.entity_mut(entity).insert(mesh_renderer);
        }
        if let Some(skinned_mesh_renderer) = snapshot.skinned_mesh_renderer.as_ref() {
            world
                .entity_mut(entity)
                .insert(skinned_mesh_renderer.clone());
        }
        if let Some(sprite_renderer) = snapshot.sprite_renderer {
            world.entity_mut(entity).insert(sprite_renderer);
        }
        if let Some(text_2d) = snapshot.text_2d.as_ref() {
            world.entity_mut(entity).insert(text_2d.clone());
        }
        if let Some(dynamic_body) = snapshot.dynamic_body {
            world.entity_mut(entity).insert(dynamic_body);
        }
        if snapshot.fixed_collider {
            world.entity_mut(entity).insert(FixedCollider);
        }
        if let Some(collider_shape) = snapshot.collider_shape {
            world.entity_mut(entity).insert(collider_shape);
        }
        spawned.push(entity);
    }

    for (child_index, snapshot) in backup.entities.iter().enumerate() {
        let Some(parent_source) = snapshot.parent else {
            continue;
        };
        let Some(parent_index) = source_to_index.get(&parent_source).copied() else {
            continue;
        };
        let Some(&child_entity) = spawned.get(child_index) else {
            continue;
        };
        let Some(&parent_entity) = spawned.get(parent_index) else {
            continue;
        };
        if child_entity == parent_entity {
            continue;
        }

        let child_transform = world
            .get::<BevyTransform>(child_entity)
            .map(|transform| transform.0)
            .unwrap_or_default();
        let parent_matrix = world
            .get::<BevyTransform>(parent_entity)
            .map(|transform| transform.0.to_matrix())
            .unwrap_or(glam::Mat4::IDENTITY);
        world.entity_mut(child_entity).insert(EntityParent {
            parent: parent_entity,
            local_transform: parent_matrix.inverse() * child_transform.to_matrix(),
            last_written: child_transform,
        });
    }

    if let Some(mut selected) = world.get_resource_mut::<InspectorSelectedEntityResource>() {
        selected.0 = backup
            .selected_index
            .and_then(|index| spawned.get(index).copied());
    }
}

fn create_entity(world: &mut World, kind: SpawnKind) -> Option<String> {
    let label = match &kind {
        SpawnKind::Empty => "Empty".to_string(),
        SpawnKind::Camera => "Camera".to_string(),
        SpawnKind::FreecamCamera => "Freecam Camera".to_string(),
        SpawnKind::DirectionalLight => "Directional Light".to_string(),
        SpawnKind::PointLight => "Point Light".to_string(),
        SpawnKind::SpotLight => "Spot Light".to_string(),
        SpawnKind::Primitive(kind) => format!("Primitive ({})", kind.display_name()),
        SpawnKind::DynamicBodyCuboid => "Dynamic Body (Box)".to_string(),
        SpawnKind::DynamicBodySphere => "Dynamic Body (Sphere)".to_string(),
        SpawnKind::FixedColliderCuboid => "Fixed Collider (Box)".to_string(),
        SpawnKind::FixedColliderSphere => "Fixed Collider (Sphere)".to_string(),
        SpawnKind::SceneAsset(path) => {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("Scene");
            format!("Scene Asset ({name})")
        }
        SpawnKind::MeshAsset(path) => {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("Mesh");
            format!("Mesh Asset ({name})")
        }
    };

    match kind {
        SpawnKind::Empty => {
            world.spawn((EditorEntity, BevyTransform::default(), Name::new("Empty")));
        }
        SpawnKind::Camera | SpawnKind::FreecamCamera => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                BevyCamera::default(),
                Name::new("Camera"),
            ));
        }
        SpawnKind::DirectionalLight => {
            world.spawn((
                EditorEntity,
                BevyWrapper(Transform::default()),
                BevyWrapper(Light::directional(glam::vec3(1.0, 1.0, 1.0), 25.0)),
                Name::new("Directional Light"),
            ));
        }
        SpawnKind::PointLight => {
            world.spawn((
                EditorEntity,
                BevyWrapper(Transform::default()),
                BevyWrapper(Light::point(glam::vec3(1.0, 1.0, 1.0), 10.0)),
                Name::new("Point Light"),
            ));
        }
        SpawnKind::SpotLight => {
            world.spawn((
                EditorEntity,
                BevyWrapper(Transform::default()),
                BevyWrapper(Light::spot(
                    glam::vec3(1.0, 1.0, 1.0),
                    10.0,
                    45.0_f32.to_radians(),
                )),
                Name::new("Spot Light"),
            ));
        }
        SpawnKind::Primitive(kind) => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                Name::new(format!("{} Primitive", kind.display_name())),
            ));
        }
        SpawnKind::DynamicBodyCuboid => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                DynamicRigidBody { mass: 1.0 },
                ColliderShape::Cuboid,
                Name::new("Dynamic Body (Box)"),
            ));
        }
        SpawnKind::DynamicBodySphere => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                DynamicRigidBody { mass: 1.0 },
                ColliderShape::Sphere,
                Name::new("Dynamic Body (Sphere)"),
            ));
        }
        SpawnKind::FixedColliderCuboid => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                FixedCollider,
                ColliderShape::Cuboid,
                Name::new("Fixed Collider (Box)"),
            ));
        }
        SpawnKind::FixedColliderSphere => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                FixedCollider,
                ColliderShape::Sphere,
                Name::new("Fixed Collider (Sphere)"),
            ));
        }
        SpawnKind::SceneAsset(path) => {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("Scene");
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                Name::new(format!("Scene Asset ({name})")),
            ));
        }
        SpawnKind::MeshAsset(path) => {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("Mesh");
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                Name::new(format!("Mesh Asset ({name})")),
            ));
        }
    }

    Some(label)
}

fn delete_entity(world: &mut World, entity: Entity) -> bool {
    if world.get_entity(entity).is_err() {
        return false;
    }
    world.despawn(entity)
}

fn set_active_camera(world: &mut World, entity: Entity) -> bool {
    if world.get::<BevyCamera>(entity).is_none() {
        return false;
    }

    let camera_entities = world
        .query::<(Entity, &BevyCamera, Option<&BevyActiveCamera>)>()
        .iter(world)
        .map(|(camera_entity, _, active)| (camera_entity, active.is_some()))
        .collect::<Vec<_>>();

    for (camera_entity, is_active) in camera_entities {
        if camera_entity == entity {
            if !is_active {
                world
                    .entity_mut(camera_entity)
                    .insert(BevyWrapper(ActiveCamera {}));
            }
            continue;
        }
        if is_active {
            world.entity_mut(camera_entity).remove::<BevyActiveCamera>();
        }
    }

    true
}

fn import_asset(
    world: &mut World,
    source_path: &Path,
    destination_dir: Option<&Path>,
) -> Result<PathBuf, String> {
    if !source_path.exists() {
        return Err(format!(
            "Import failed: source does not exist ({})",
            source_path.to_string_lossy()
        ));
    }

    let Some(project) = world.get_resource::<EditorProject>() else {
        return Err("Import failed: open a project first".to_string());
    };
    let Some(project_root) = project.root.as_ref() else {
        return Err("Import failed: open a project first".to_string());
    };

    let target_dir = destination_dir
        .map(PathBuf::from)
        .or_else(|| {
            project
                .config
                .as_ref()
                .map(|cfg| cfg.assets_root(project_root))
        })
        .unwrap_or_else(|| project_root.join("assets"));

    fs::create_dir_all(&target_dir).map_err(|err| err.to_string())?;

    let Some(file_name) = source_path.file_name() else {
        return Err("Import failed: source path has no file name".to_string());
    };
    let target_path = unique_path(&target_dir.join(file_name));
    fs::copy(source_path, &target_path).map_err(|err| err.to_string())?;

    if let Some(mut assets) = world.get_resource_mut::<AssetBrowserState>() {
        assets.refresh_requested = true;
    }

    Ok(target_path)
}

fn create_asset(
    world: &mut World,
    directory: &Path,
    name: &str,
    kind: AssetCreateKind,
) -> Result<PathBuf, String> {
    fs::create_dir_all(directory).map_err(|err| err.to_string())?;

    let target_path = unique_path(&asset_target_path(directory, name, &kind));
    match kind {
        AssetCreateKind::Folder => {
            fs::create_dir_all(&target_path).map_err(|err| err.to_string())?;
        }
        AssetCreateKind::Scene => {
            fs::write(&target_path, default_scene_template()).map_err(|err| err.to_string())?;
        }
        AssetCreateKind::Material => {
            fs::write(&target_path, default_material_template()).map_err(|err| err.to_string())?;
        }
        AssetCreateKind::Script => {
            fs::write(&target_path, default_script_template()).map_err(|err| err.to_string())?;
        }
        AssetCreateKind::VisualScript => {
            fs::write(&target_path, default_visual_script_template())
                .map_err(|err| err.to_string())?;
        }
        AssetCreateKind::VisualScriptThirdPerson => {
            fs::write(&target_path, default_visual_script_third_person_template())
                .map_err(|err| err.to_string())?;
        }
        AssetCreateKind::RustScript => {
            fs::create_dir_all(target_path.join("src")).map_err(|err| err.to_string())?;
            let crate_name = sanitize_crate_name(name);
            let manifest = format!(
                "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lib]\ncrate-type = [\"cdylib\"]\n",
                crate_name
            );
            fs::write(target_path.join("Cargo.toml"), manifest).map_err(|err| err.to_string())?;
            fs::write(
                target_path.join("src").join("lib.rs"),
                "pub fn on_start() {}\npub fn on_update(_dt: f32) {}\n",
            )
            .map_err(|err| err.to_string())?;
        }
        AssetCreateKind::Animation => {
            fs::write(&target_path, default_animation_template()).map_err(|err| err.to_string())?;
        }
    }

    if let Some(mut assets) = world.get_resource_mut::<AssetBrowserState>() {
        assets.refresh_requested = true;
    }

    Ok(target_path)
}

fn asset_target_path(directory: &Path, name: &str, kind: &AssetCreateKind) -> PathBuf {
    match kind {
        AssetCreateKind::Folder => directory.join(name),
        AssetCreateKind::Scene => directory.join(format!("{name}.hscene.ron")),
        AssetCreateKind::Material => directory.join(format!("{name}.ron")),
        AssetCreateKind::Script => directory.join(format!("{name}.luau")),
        AssetCreateKind::VisualScript | AssetCreateKind::VisualScriptThirdPerson => {
            directory.join(format!("{name}.hvs"))
        }
        AssetCreateKind::RustScript => directory.join(name),
        AssetCreateKind::Animation => directory.join(format!("{name}.hanim.ron")),
    }
}

fn unique_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }

    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("asset");
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{ext}"))
        .unwrap_or_default();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));

    for index in 1..=999u32 {
        let candidate = parent.join(format!("{stem}_{index}{extension}"));
        if !candidate.exists() {
            return candidate;
        }
    }

    path.to_path_buf()
}

fn sanitize_crate_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len().max(1));
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else if ch == '-' || ch.is_whitespace() {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("script_module");
    }
    if out.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

fn default_scene_template() -> &'static str {
    "(\n    version: 1,\n    entities: [],\n    scene_child_animations: [],\n    scene_child_pose_overrides: [],\n)\n"
}

fn default_material_template() -> &'static str {
    "(\n    base_color: (1.0, 1.0, 1.0, 1.0),\n    metallic: 0.0,\n    roughness: 0.8,\n)\n"
}

fn default_script_template() -> &'static str {
    "function on_start()\nend\n\nfunction on_update(dt)\nend\n"
}

fn default_visual_script_template() -> &'static str {
    "{\n  \"version\": 1,\n  \"nodes\": [],\n  \"connections\": []\n}\n"
}

fn default_visual_script_third_person_template() -> &'static str {
    "{\n  \"version\": 1,\n  \"template\": \"third_person\",\n  \"nodes\": [],\n  \"connections\": []\n}\n"
}

fn default_animation_template() -> &'static str {
    "(\n    version: 1,\n    name: \"Animation\",\n    tracks: [],\n    clips: [],\n)\n"
}

fn status_level(message: &str) -> EditorConsoleLevel {
    let message = message.to_ascii_lowercase();
    if message.contains("error")
        || message.contains("failed")
        || message.contains("unable")
        || message.contains("missing")
    {
        EditorConsoleLevel::Error
    } else if message.contains("warn") {
        EditorConsoleLevel::Warn
    } else if message.contains("debug") {
        EditorConsoleLevel::Debug
    } else {
        EditorConsoleLevel::Info
    }
}

fn default_scene_save_path(world: &World) -> Option<PathBuf> {
    let project = world.get_resource::<EditorProject>()?;
    let root = project.root.as_ref()?;
    let scenes_root = project
        .config
        .as_ref()
        .map(|cfg| cfg.scenes_root(root))
        .unwrap_or_else(|| root.join("assets").join("scenes"));
    Some(scenes_root.join("untitled_scene.hsf"))
}

fn apply_undo(world: &mut World) -> Option<String> {
    let (snapshot, label) = {
        let mut undo = world.get_resource_mut::<EditorUndoState>()?;
        if !undo.can_undo() {
            return None;
        }
        let label = undo
            .entries
            .get(undo.cursor)
            .and_then(|entry| entry.label().map(str::to_string));
        undo.cursor = undo.cursor.saturating_sub(1);
        let snapshot = undo
            .entries
            .get(undo.cursor)
            .cloned()
            .map(|entry| entry.scene);
        (snapshot, label)
    };

    if let Some(snapshot) = snapshot.as_ref() {
        apply_scene_undo_snapshot(world, snapshot);
    }
    label
}

fn apply_redo(world: &mut World) -> Option<String> {
    let (snapshot, label) = {
        let mut undo = world.get_resource_mut::<EditorUndoState>()?;
        if !undo.can_redo() {
            return None;
        }
        undo.cursor += 1;
        let snapshot = undo
            .entries
            .get(undo.cursor)
            .cloned()
            .map(|entry| entry.scene);
        let label = undo
            .entries
            .get(undo.cursor)
            .and_then(|entry| entry.label().map(str::to_string));
        (snapshot, label)
    };

    if let Some(snapshot) = snapshot.as_ref() {
        apply_scene_undo_snapshot(world, snapshot);
    }
    label
}

fn sync_recent_project(world: &mut World, path: &Path) {
    if let Some(mut launcher) = world.get_resource_mut::<EditorProjectLauncherState>() {
        if let Err(err) = record_recent_project(&mut launcher, path) {
            launcher.status = Some(format!("Failed to save recent projects: {err}"));
        }
    }
}

fn update_launcher_after_project_open(world: &mut World, path: &Path, status: Option<String>) {
    if let Some(mut launcher) = world.get_resource_mut::<EditorProjectLauncherState>() {
        launcher.open_project_path = path.to_string_lossy().into_owned();
        launcher.status = status;
    }
}

fn set_launcher_status(world: &mut World, status: impl Into<String>) {
    if let Some(mut launcher) = world.get_resource_mut::<EditorProjectLauncherState>() {
        launcher.status = Some(status.into());
    }
}

fn close_project(world: &mut World) {
    let previous_root = world
        .get_resource::<EditorProject>()
        .and_then(|project| project.root.clone());

    if let Some(mut project) = world.get_resource_mut::<EditorProject>() {
        project.root = None;
        project.config = None;
    }
    if let Some(mut watch) = world.get_resource_mut::<FileWatchState>() {
        watch.root = None;
        watch.watcher = None;
        watch.receiver = None;
        watch.status = Some("File watcher stopped".to_string());
    }
    if let Some(mut scripts) = world.get_resource_mut::<ScriptRegistry>() {
        scripts.scripts.clear();
        scripts.dirty_paths.clear();
        scripts.status = Some("Closed project; script cache cleared".to_string());
    }
    if let Some(mut browser) = world.get_resource_mut::<AssetBrowserState>() {
        browser.root = None;
        browser.entries.clear();
        browser.expanded.clear();
        browser.selected = None;
        browser.selected_paths.clear();
        browser.selection_anchor = None;
        browser.current_dir = None;
        browser.last_click_path = None;
        browser.refresh_requested = false;
    }
    if let Some(mut scene) = world.get_resource_mut::<EditorSceneState>() {
        scene.path = None;
        scene.name = "Untitled".to_string();
        scene.dirty = false;
        scene.world_state = WorldState::Edit;
        scene.play_backup = None;
        scene.play_selected_index = None;
    }
    reset_editor_scene_entities(world);
    spawn_default_scene_entities(world);
    if let Some(mut launcher) = world.get_resource_mut::<EditorProjectLauncherState>() {
        if let Some(path) = previous_root {
            launcher.open_project_path = path.to_string_lossy().into_owned();
        }
        launcher.status = Some("Project closed".to_string());
    }
    reset_scene_undo_history(world, Some("Close Project"));
    set_console_status(world, EditorConsoleLevel::Info, "Closed project");
}

fn set_console_status(world: &mut World, level: EditorConsoleLevel, message: impl Into<String>) {
    if let Some(mut console) = world.get_resource_mut::<EditorConsoleState>() {
        console.push(level, "editor.runtime", message.into());
    }
}
