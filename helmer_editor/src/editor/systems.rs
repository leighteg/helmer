use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use bevy_ecs::prelude::{Changed, Entity, Or, Query, Res, ResMut, World};
use bevy_ecs::{component::Component, name::Name};
use egui::Ui;
use glam::{DVec2, Quat, Vec3};
use helmer::graphics::render_graphs::template_for_graph;
use helmer::provided::components::{Light, MeshRenderer, Transform};
use helmer::runtime::asset_server::{Handle, Material, Mesh};
use helmer_becs::egui_integration::EguiResource;
use helmer_becs::physics::components::{ColliderShape, DynamicRigidBody, FixedCollider};
use helmer_becs::physics::physics_resource::PhysicsResource;
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;
use helmer_becs::systems::render_system::RenderGraphResource;
use helmer_becs::systems::scene_system::SceneRoot;
use helmer_becs::{
    BevyCamera, BevyInputManager, BevyLight, BevyMeshRenderer, BevyTransform, BevyWrapper,
    DeltaTime, DraggedFile,
};
use winit::{event::MouseButton, keyboard::KeyCode};

use crate::editor::{
    EditorPlayCamera, EditorViewportCamera, EditorViewportState, activate_play_camera,
    activate_viewport_camera,
    assets::{
        AssetBrowserState, EditorAssetCache, EditorMesh, MeshSource, PrimitiveKind, SceneAssetPath,
        scan_asset_entries,
    },
    commands::{AssetCreateKind, EditorCommand, EditorCommandQueue, SpawnKind},
    dynamic::DynamicComponents,
    project::{
        EditorProject, create_project, default_material_template, default_scene_template,
        default_script_template, load_project, save_recent_projects,
    },
    scene::{
        EditorEntity, EditorSceneState, WorldState, next_available_scene_path, read_scene_document,
        reset_editor_scene, restore_scene_transforms_from_document, serialize_scene,
        spawn_default_camera, spawn_default_light, spawn_scene_from_document, write_scene_document,
    },
    scripting::{ScriptComponent, ScriptRegistry, load_script_asset},
    set_play_camera,
    ui::{
        EditorUiState, draw_assets_window, draw_project_window, draw_scene_window, draw_toolbar,
        draw_viewport_window,
    },
    watch::configure_file_watcher,
};

pub fn editor_ui_system(mut egui_res: ResMut<EguiResource>) {
    egui_res.inspector_ui = false;
    egui_res.windows.push((
        Box::new(|ui: &mut Ui, world: &mut World, _| {
            draw_toolbar(ui, world);
        }),
        "Toolbar".to_string(),
    ));

    egui_res.windows.push((
        Box::new(|ui: &mut Ui, world: &mut World, _| {
            draw_viewport_window(ui, world);
        }),
        "Viewport".to_string(),
    ));

    egui_res.windows.push((
        Box::new(|ui: &mut Ui, world: &mut World, _| {
            draw_project_window(ui, world);
        }),
        "Project".to_string(),
    ));

    egui_res.windows.push((
        Box::new(|ui: &mut Ui, world: &mut World, _| {
            draw_scene_window(ui, world);
        }),
        "Hierarchy Inspector".to_string(),
    ));

    egui_res.windows.push((
        Box::new(|ui: &mut Ui, world: &mut World, _| {
            draw_assets_window(ui, world);
        }),
        "Content Browser".to_string(),
    ));
}

pub fn editor_physics_state_system(
    scene_state: Res<EditorSceneState>,
    mut phys: ResMut<PhysicsResource>,
) {
    let should_run = scene_state.world_state == WorldState::Play;
    if phys.running != should_run {
        phys.running = should_run;
    }
}

pub fn editor_command_system(world: &mut World) {
    let commands = {
        let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() else {
            return;
        };
        std::mem::take(&mut queue.commands)
    };

    for command in commands {
        match command {
            EditorCommand::CreateProject { name, path } => {
                handle_create_project(world, &name, &path);
            }
            EditorCommand::OpenProject { path } => {
                handle_open_project(world, &path);
            }
            EditorCommand::NewScene => {
                handle_new_scene(world);
            }
            EditorCommand::OpenScene { path } => {
                handle_open_scene(world, &path);
            }
            EditorCommand::SaveScene => {
                handle_save_scene(world);
            }
            EditorCommand::SaveSceneAs { path } => {
                handle_save_scene_as(world, &path);
            }
            EditorCommand::CreateEntity { kind } => {
                handle_create_entity(world, kind);
            }
            EditorCommand::ImportAsset {
                source_path,
                destination_dir,
            } => {
                handle_import_asset(world, &source_path, destination_dir.as_deref());
            }
            EditorCommand::CreateAsset {
                directory,
                name,
                kind,
            } => {
                handle_create_asset(world, &directory, &name, kind);
            }
            EditorCommand::DeleteEntity { entity } => {
                if world.get_entity(entity).is_ok() {
                    world.despawn(entity);
                }
                if let Some(mut selection) = world.get_resource_mut::<
                    helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource,
                >() {
                    if selection.0 == Some(entity) {
                        selection.0 = None;
                    }
                }
            }
            EditorCommand::SetActiveCamera { entity } => {
                handle_set_active_camera(world, entity);
            }
            EditorCommand::TogglePlayMode => {
                handle_toggle_play(world);
            }
            EditorCommand::CloseProject => {
                handle_close_project(world);
            }
        }
    }
}

pub fn asset_scan_system(mut state: ResMut<AssetBrowserState>) {
    let root = match state.root.as_ref() {
        Some(root) => root.clone(),
        None => return,
    };

    let now = Instant::now();
    if !state.refresh_requested && now.duration_since(state.last_scan) < state.scan_interval {
        return;
    }

    state.entries = scan_asset_entries(&root, &state.filter);
    state.refresh_requested = false;
    state.last_scan = now;
}

pub fn drag_drop_system(
    mut dragged: ResMut<DraggedFile>,
    mut queue: ResMut<EditorCommandQueue>,
    assets: Res<AssetBrowserState>,
) {
    if let Some(path) = dragged.0.take() {
        let destination_dir = assets.current_dir.clone().or_else(|| {
            assets.selected.as_ref().and_then(|selected| {
                if selected.is_dir() {
                    Some(selected.clone())
                } else {
                    selected.parent().map(|parent| parent.to_path_buf())
                }
            })
        });

        queue.push(EditorCommand::ImportAsset {
            source_path: path,
            destination_dir,
        });
    }
}

pub fn editor_shortcut_system(
    input_manager: Res<BevyInputManager>,
    mut queue: ResMut<EditorCommandQueue>,
) {
    let input_manager = input_manager.0.read();

    if input_manager.egui_wants_key {
        return;
    }

    let control = input_manager.is_key_active(KeyCode::ControlLeft);
    if control && input_manager.just_pressed.contains(&KeyCode::KeyN) {
        queue.push(EditorCommand::NewScene);
    }
    if control && input_manager.just_pressed.contains(&KeyCode::KeyS) {
        queue.push(EditorCommand::SaveScene);
    }
    if input_manager.just_pressed.contains(&KeyCode::F5) {
        queue.push(EditorCommand::TogglePlayMode);
    }
}

pub fn scene_dirty_system(
    mut scene_state: ResMut<EditorSceneState>,
    query: Query<
        (),
        (
            bevy_ecs::prelude::With<EditorEntity>,
            Or<(
                Changed<BevyTransform>,
                Changed<BevyMeshRenderer>,
                Changed<EditorMesh>,
                Changed<BevyLight>,
                Changed<BevyCamera>,
                Changed<Name>,
                Changed<SceneRoot>,
                Changed<SceneAssetPath>,
                Changed<ScriptComponent>,
                Changed<DynamicComponents>,
            )>,
        ),
    >,
) {
    if scene_state.world_state == WorldState::Play {
        return;
    }

    if !query.is_empty() {
        scene_state.dirty = true;
    }
}

pub fn script_registry_system(mut registry: ResMut<ScriptRegistry>, project: Res<EditorProject>) {
    let Some(root) = project.root.as_ref() else {
        return;
    };

    let mut dirty_paths = registry.take_dirty_paths();
    if !dirty_paths.is_empty() {
        let mut updated = 0;
        for path in dirty_paths.drain() {
            if !path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("lua"))
                .unwrap_or(false)
            {
                continue;
            }

            if path.exists() {
                registry
                    .scripts
                    .insert(path.clone(), load_script_asset(&path));
                updated += 1;
            } else {
                registry.scripts.remove(&path);
            }
        }

        if updated > 0 {
            registry.status = Some(format!("Reloaded {} script(s)", updated));
        }
        return;
    }

    let now = Instant::now();
    if now.duration_since(registry.last_scan) < registry.scan_interval {
        return;
    }

    registry.last_scan = now;

    let scripts_root = project.config.as_ref().map(|cfg| cfg.scripts_root(root));

    let Some(scripts_root) = scripts_root else {
        return;
    };

    if !scripts_root.exists() {
        return;
    }

    let mut updated = 0;

    if let Ok(entries) = fs::read_dir(&scripts_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("lua"))
                .unwrap_or(false)
            {
                continue;
            }

            let reload = match registry.scripts.get(&path) {
                Some(asset) => match fs::metadata(&path).and_then(|meta| meta.modified()) {
                    Ok(modified) => modified > asset.modified,
                    Err(_) => false,
                },
                None => true,
            };

            if reload {
                registry
                    .scripts
                    .insert(path.clone(), load_script_asset(&path));
                updated += 1;
            }
        }
    }

    if updated > 0 {
        registry.status = Some(format!("Updated {} script(s)", updated));
    }
}

fn handle_create_project(world: &mut World, name: &str, path: &Path) {
    match create_project(path, name) {
        Ok(config) => {
            {
                let mut project = world
                    .get_resource_mut::<EditorProject>()
                    .expect("EditorProject missing");
                project.root = Some(path.to_path_buf());
                project.config = Some(config);
            }

            record_recent_project(world, path);

            let project_snapshot = world
                .get_resource::<EditorProject>()
                .cloned()
                .expect("EditorProject missing");
            initialize_project_state(world, &project_snapshot);
            set_status(world, format!("Project created at {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to create project: {}", err));
        }
    }
}

fn handle_open_project(world: &mut World, path: &Path) {
    match load_project(path) {
        Ok(config) => {
            {
                let mut project = world
                    .get_resource_mut::<EditorProject>()
                    .expect("EditorProject missing");
                project.root = Some(path.to_path_buf());
                project.config = Some(config);
            }

            record_recent_project(world, path);

            let project_snapshot = world
                .get_resource::<EditorProject>()
                .cloned()
                .expect("EditorProject missing");
            initialize_project_state(world, &project_snapshot);
            set_status(world, format!("Project opened at {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to open project: {}", err));
        }
    }
}

fn handle_close_project(world: &mut World) {
    if let Some(mut project) = world.get_resource_mut::<EditorProject>() {
        project.root = None;
        project.config = None;
    }

    if let Some(mut assets) = world.get_resource_mut::<AssetBrowserState>() {
        assets.root = None;
        assets.entries.clear();
        assets.expanded.clear();
        assets.selected = None;
        assets.current_dir = None;
        assets.refresh_requested = true;
    }

    if let Some(mut watcher) = world.get_resource_mut::<crate::editor::watch::FileWatchState>() {
        watcher.root = None;
        watcher.watcher = None;
        watcher.receiver = None;
        watcher.pending_paths.clear();
        watcher.status = Some("File watcher disabled".to_string());
    }

    if let Some(mut registry) = world.get_resource_mut::<ScriptRegistry>() {
        registry.scripts.clear();
        registry.dirty_paths.clear();
        registry.status = None;
    }

    handle_new_scene(world);
    set_status(world, "Project closed".to_string());
}

fn record_recent_project(world: &mut World, path: &Path) {
    let normalized = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
        state.recent_projects.retain(|entry| entry != &normalized);
        state.recent_projects.insert(0, normalized.clone());
        const MAX_RECENT_PROJECTS: usize = 8;
        if state.recent_projects.len() > MAX_RECENT_PROJECTS {
            state.recent_projects.truncate(MAX_RECENT_PROJECTS);
        }

        if let Err(err) = save_recent_projects(&state.recent_projects) {
            state.status = Some(format!("Failed to save recent projects: {}", err));
        }
    }
}

fn initialize_project_state(world: &mut World, project: &EditorProject) {
    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = world
            .get_resource::<helmer_becs::BevyAssetServer>()
            .expect("AssetServer missing");
        ensure_default_material(project, &mut cache, asset_server);
    });

    if let Some(mut asset_state) = world.get_resource_mut::<AssetBrowserState>() {
        if let Some(root) = project.root.as_ref() {
            asset_state.root = Some(root.to_path_buf());
            asset_state.expanded.insert(root.to_path_buf());
            asset_state.selected = Some(root.to_path_buf());
            asset_state.current_dir = Some(root.to_path_buf());
            asset_state.refresh_requested = true;
        }
    }

    if let Some(root) = project.root.as_ref() {
        if let Some(mut watcher) = world.get_resource_mut::<crate::editor::watch::FileWatchState>()
        {
            configure_file_watcher(&mut watcher, root);
        }
    }

    handle_new_scene(world);
}

fn handle_new_scene(world: &mut World) {
    reset_editor_scene(world);
    spawn_default_camera(world);
    spawn_default_light(world);

    if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
        scene_state.path = None;
        scene_state.name = "Untitled".to_string();
        scene_state.dirty = false;
        scene_state.play_backup = None;
        scene_state.play_selected_index = None;
    }

    if let Some(mut selection) = world.get_resource_mut::<
        helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource,
    >() {
        selection.0 = None;
    }
}

fn handle_open_scene(world: &mut World, path: &Path) {
    match read_scene_document(path) {
        Ok(document) => {
            let project_snapshot = world
                .get_resource::<EditorProject>()
                .cloned()
                .expect("EditorProject missing");

            reset_editor_scene(world);

            let created_entities =
                world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
                    let asset_server = {
                        let asset_server = world
                            .get_resource::<helmer_becs::BevyAssetServer>()
                            .expect("AssetServer missing");
                        helmer_becs::BevyAssetServer(asset_server.0.clone())
                    };
                    spawn_scene_from_document(
                        world,
                        &document,
                        &project_snapshot,
                        &mut cache,
                        &asset_server,
                    )
                });

            restore_scene_transforms_from_document(world, &document, &created_entities);

            if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
                scene_state.path = Some(path.to_path_buf());
                scene_state.name = scene_display_name(path);
                scene_state.dirty = false;
                scene_state.play_backup = None;
            }

            set_status(world, format!("Scene loaded from {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to load scene: {}", err));
        }
    }
}

fn handle_save_scene(world: &mut World) {
    let scene_path = world
        .get_resource::<EditorSceneState>()
        .and_then(|scene| scene.path.clone());

    if let Some(path) = scene_path {
        handle_save_scene_as(world, &path);
        return;
    }

    if let Some(path) = next_available_scene_path(
        world
            .get_resource::<EditorProject>()
            .expect("Project missing"),
    ) {
        handle_save_scene_as(world, &path);
    } else {
        set_status(world, "Unable to allocate a scene file name".to_string());
    }
}

fn handle_save_scene_as(world: &mut World, path: &Path) {
    let project = world
        .get_resource::<EditorProject>()
        .cloned()
        .expect("EditorProject missing");
    let (document, _) = serialize_scene(world, &project);

    match write_scene_document(path, &document) {
        Ok(()) => {
            if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
                scene_state.path = Some(path.to_path_buf());
                scene_state.name = scene_display_name(path);
                scene_state.dirty = false;
            }

            if let Some(mut asset_state) = world.get_resource_mut::<AssetBrowserState>() {
                asset_state.refresh_requested = true;
            }

            set_status(world, format!("Scene saved to {}", path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to save scene: {}", err));
        }
    }
}

fn handle_create_entity(world: &mut World, kind: SpawnKind) {
    match kind {
        SpawnKind::Empty => {
            world.spawn((EditorEntity, BevyTransform::default(), Name::new("Empty")));
        }
        SpawnKind::Camera => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                BevyCamera::default(),
                Name::new("Camera"),
            ));
        }
        SpawnKind::FreecamCamera => {
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                BevyCamera::default(),
                Freecam {},
                Name::new("Freecam Camera"),
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
            spawn_primitive(world, kind);
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
            spawn_scene_asset(world, &path);
        }
        SpawnKind::MeshAsset(path) => {
            spawn_mesh_asset(world, &path);
        }
    }

    if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
        scene_state.dirty = true;
    }
}

fn handle_import_asset(world: &mut World, source: &Path, destination: Option<&Path>) {
    let Some(project) = world.get_resource::<EditorProject>() else {
        set_status(world, "Open a project before importing".to_string());
        return;
    };

    let Some(root) = project.root.as_ref() else {
        set_status(world, "Open a project before importing".to_string());
        return;
    };

    let target_dir = destination
        .map(|path| path.to_path_buf())
        .or_else(|| guess_import_dir(project, root, source))
        .unwrap_or_else(|| root.join("assets"));

    if !target_dir.exists() {
        if let Err(err) = fs::create_dir_all(&target_dir) {
            set_status(world, format!("Failed to create target dir: {}", err));
            return;
        }
    }

    let Some(file_name) = source.file_name() else {
        set_status(world, "Invalid source file".to_string());
        return;
    };

    let mut target_path = target_dir.join(file_name);
    target_path = unique_path(&target_path);

    match fs::copy(source, &target_path) {
        Ok(_) => {
            if let Some(mut asset_state) = world.get_resource_mut::<AssetBrowserState>() {
                asset_state.refresh_requested = true;
            }
            set_status(
                world,
                format!("Imported asset to {}", target_path.display()),
            );
        }
        Err(err) => {
            set_status(world, format!("Import failed: {}", err));
        }
    }
}

fn handle_create_asset(world: &mut World, directory: &Path, name: &str, kind: AssetCreateKind) {
    let target_path = match kind {
        AssetCreateKind::Folder => directory.join(name),
        AssetCreateKind::Scene => directory.join(format!("{}.hscene.ron", name)),
        AssetCreateKind::Material => directory.join(format!("{}.ron", name)),
        AssetCreateKind::Script => directory.join(format!("{}.lua", name)),
    };

    let target_path = unique_path(&target_path);

    let result = match kind {
        AssetCreateKind::Folder => fs::create_dir_all(&target_path).map_err(|err| err.to_string()),
        AssetCreateKind::Scene => {
            fs::write(&target_path, default_scene_template()).map_err(|err| err.to_string())
        }
        AssetCreateKind::Material => {
            fs::write(&target_path, default_material_template()).map_err(|err| err.to_string())
        }
        AssetCreateKind::Script => {
            fs::write(&target_path, default_script_template()).map_err(|err| err.to_string())
        }
    };

    match result {
        Ok(()) => {
            if let Some(mut asset_state) = world.get_resource_mut::<AssetBrowserState>() {
                asset_state.refresh_requested = true;
            }
            set_status(world, format!("Created {}", target_path.display()));
        }
        Err(err) => {
            set_status(world, format!("Failed to create asset: {}", err));
        }
    }
}

fn handle_set_active_camera(world: &mut World, entity: Entity) {
    if world.get::<BevyCamera>(entity).is_none() {
        set_status(world, "Selected entity has no camera".to_string());
        return;
    }

    if world.get::<EditorViewportCamera>(entity).is_some() {
        set_status(
            world,
            "Viewport camera is managed by the editor".to_string(),
        );
        return;
    }

    set_play_camera(world, entity);

    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);
    if world_state == WorldState::Play {
        activate_play_camera(world);
    }

    set_status(world, "Game camera updated".to_string());
}

fn apply_viewport_graph(world: &mut World) {
    let template_name = world
        .get_resource::<EditorViewportState>()
        .map(|state| state.graph_template.clone());
    let Some(template_name) = template_name else {
        return;
    };
    let Some(template) = template_for_graph(&template_name) else {
        return;
    };
    if let Some(mut graph_res) = world.get_resource_mut::<RenderGraphResource>() {
        graph_res.0 = (template.build)();
    }
}

fn handle_toggle_play(world: &mut World) {
    let state = world
        .get_resource::<EditorSceneState>()
        .map(|scene| scene.world_state)
        .unwrap_or(WorldState::Edit);

    match state {
        WorldState::Edit => {
            let project = world
                .get_resource::<EditorProject>()
                .cloned()
                .expect("Project missing");
            let (document, entity_order) = serialize_scene(world, &project);

            let selected_entity = world
                .get_resource::<InspectorSelectedEntityResource>()
                .and_then(|selection| selection.0);
            let selection_index = selected_entity
                .and_then(|entity| entity_order.iter().position(|ordered| *ordered == entity));

            if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
                scene_state.play_backup = Some(document);
                scene_state.play_selected_index = selection_index;
                scene_state.world_state = WorldState::Play;
            }
            activate_play_camera(world);
            apply_viewport_graph(world);

            set_status(world, "Play mode".to_string());
        }
        WorldState::Play => {
            if let Some(mut phys_res) = world.get_resource_mut::<PhysicsResource>() {
                phys_res.running = false;
            }

            let (backup, selection_index) =
                if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
                    scene_state.world_state = WorldState::Edit;
                    (
                        scene_state.play_backup.take(),
                        scene_state.play_selected_index.take(),
                    )
                } else {
                    (None, None)
                };

            if let Some(document) = backup {
                let project_snapshot = world
                    .get_resource::<EditorProject>()
                    .cloned()
                    .expect("Project missing");
                reset_editor_scene(world);

                let created_entities =
                    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
                        let asset_server = {
                            let asset_server = world
                                .get_resource::<helmer_becs::BevyAssetServer>()
                                .expect("AssetServer missing");
                            helmer_becs::BevyAssetServer(asset_server.0.clone())
                        };
                        spawn_scene_from_document(
                            world,
                            &document,
                            &project_snapshot,
                            &mut cache,
                            &asset_server,
                        )
                    });
                restore_scene_transforms_from_document(world, &document, &created_entities);

                if let Some(index) = selection_index {
                    if let Some(&entity) = created_entities.get(index) {
                        if let Some(mut selection) =
                            world.get_resource_mut::<InspectorSelectedEntityResource>()
                        {
                            selection.0 = Some(entity);
                        }
                    }
                }
            }

            activate_viewport_camera(world);
            apply_viewport_graph(world);

            set_status(world, "Edit mode".to_string());
        }
    }
}

fn spawn_primitive(world: &mut World, kind: PrimitiveKind) {
    let project = world
        .get_resource::<EditorProject>()
        .cloned()
        .expect("Project missing");

    let material_path = project
        .config
        .as_ref()
        .and_then(|config| {
            project.root.as_ref().map(|root| {
                config
                    .materials_root(root)
                    .join("default.ron")
                    .strip_prefix(root)
                    .ok()
                    .map(|path| path.to_string_lossy().replace('\\', "/"))
            })
        })
        .flatten();

    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = world
            .get_resource::<helmer_becs::BevyAssetServer>()
            .expect("AssetServer missing");
        let material_handle = ensure_default_material(&project, &mut cache, asset_server);
        let mesh_handle = load_primitive_mesh(kind, &mut cache, asset_server);

        let Some(material_handle) = material_handle else {
            set_status(world, "Default material missing".to_string());
            return;
        };

        world.spawn((
            EditorEntity,
            BevyTransform::default(),
            BevyWrapper(MeshRenderer::new(
                mesh_handle.id,
                material_handle.id,
                true,
                true,
            )),
            EditorMesh {
                source: MeshSource::Primitive(kind),
                material_path,
            },
        ));
    });
}

fn spawn_scene_asset(world: &mut World, path: &Path) {
    let asset_server = world
        .get_resource::<helmer_becs::BevyAssetServer>()
        .expect("AssetServer missing");
    let handle = asset_server.0.lock().load_scene(path);

    world.spawn((
        EditorEntity,
        BevyTransform::default(),
        SceneRoot(handle),
        SceneAssetPath {
            path: path.to_path_buf(),
        },
    ));
}

fn spawn_mesh_asset(world: &mut World, path: &Path) {
    let project = world
        .get_resource::<EditorProject>()
        .cloned()
        .expect("Project missing");

    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = world
            .get_resource::<helmer_becs::BevyAssetServer>()
            .expect("AssetServer missing");
        let material_handle = ensure_default_material(&project, &mut cache, asset_server);
        let mesh_handle = load_mesh_asset(path, &mut cache, asset_server);

        let Some(material_handle) = material_handle else {
            set_status(world, "Default material missing".to_string());
            return;
        };

        world.spawn((
            EditorEntity,
            BevyTransform::default(),
            BevyWrapper(MeshRenderer::new(
                mesh_handle.id,
                material_handle.id,
                true,
                true,
            )),
            EditorMesh {
                source: MeshSource::Asset {
                    path: path.to_string_lossy().replace('\\', "/"),
                },
                material_path: None,
            },
        ));
    });
}

fn ensure_default_material(
    project: &EditorProject,
    cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
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

fn load_mesh_asset(
    path: &Path,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Handle<Mesh> {
    let key = path.to_string_lossy().replace('\\', "/");
    if let Some(handle) = asset_cache.mesh_handles.get(&key).copied() {
        return handle;
    }

    let handle = asset_server.0.lock().load_mesh(path);
    asset_cache.mesh_handles.insert(key, handle);
    handle
}

fn load_primitive_mesh(
    kind: PrimitiveKind,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Handle<Mesh> {
    if let Some(handle) = asset_cache.primitive_meshes.get(&kind).copied() {
        return handle;
    }

    let mesh_asset = match kind {
        PrimitiveKind::Cube => helmer::provided::components::MeshAsset::cube("cube".to_string()),
        PrimitiveKind::Plane => helmer::provided::components::MeshAsset::plane("plane".to_string()),
    };

    let handle = asset_server
        .0
        .lock()
        .add_mesh(mesh_asset.vertices.unwrap(), mesh_asset.indices);
    asset_cache.primitive_meshes.insert(kind, handle);
    handle
}

fn guess_import_dir(project: &EditorProject, root: &Path, source: &Path) -> Option<PathBuf> {
    let config = project.config.as_ref()?;
    let ext = source
        .extension()
        .and_then(|ext| ext.to_str())?
        .to_ascii_lowercase();

    let target = match ext.as_str() {
        "glb" | "gltf" => config.models_root(root),
        "ktx2" | "png" | "jpg" | "jpeg" | "tga" => config.textures_root(root),
        "lua" => config.scripts_root(root),
        "ron" => config.materials_root(root),
        _ => config.assets_root(root),
    };

    Some(target)
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

fn scene_display_name(path: &Path) -> String {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Scene");
    if let Some(stripped) = file_name.strip_suffix(".hscene.ron") {
        return stripped.to_string();
    }
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("Scene")
        .to_string()
}

fn set_status(world: &mut World, message: String) {
    if let Some(mut state) = world.get_resource_mut::<EditorUiState>() {
        state.status = Some(message);
    }
}

//================================================================================
// Freecam System
//================================================================================

#[derive(Default)]
pub struct FreecamState {
    speed: f32,
    sensitivity: f32,
    fov_lerp_speed: f32,
    is_looking: bool,
    last_cursor_position: DVec2,
    current_fov_multiplier: f32,
    active_entity: Option<Entity>,
}

#[derive(Component, Default)]
pub struct Freecam;

pub fn freecam_system(
    mut state: bevy_ecs::system::Local<FreecamState>,
    input: Res<BevyInputManager>,
    time: Res<DeltaTime>,
    scene_state: Res<EditorSceneState>,
    mut viewport_query: bevy_ecs::prelude::Query<
        (Entity, &mut BevyTransform, &mut BevyCamera),
        (
            bevy_ecs::prelude::With<EditorViewportCamera>,
            bevy_ecs::prelude::Without<EditorPlayCamera>,
        ),
    >,
    mut play_query: bevy_ecs::prelude::Query<
        (Entity, &mut BevyTransform, &mut BevyCamera),
        (
            bevy_ecs::prelude::With<EditorPlayCamera>,
            bevy_ecs::prelude::With<Freecam>,
            bevy_ecs::prelude::Without<EditorViewportCamera>,
        ),
    >,
) {
    if state.speed == 0.0 {
        state.speed = 1.0;
        state.sensitivity = 0.3;
        state.fov_lerp_speed = 8.0;
        state.current_fov_multiplier = 1.0;
    }

    let dt = time.0;
    let input_manager = &input.0.read();
    let wants_pointer = input_manager.egui_wants_pointer;

    const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const BOOST_AMOUNT: f32 = 1.15;
    const CONTROLLER_SENSITIVITY: f32 = 2.0;

    let maybe_gamepad_id = input_manager.first_gamepad_id();

    let mut yaw_delta = 0.0;
    let mut pitch_delta = 0.0;

    if !wants_pointer && input_manager.is_mouse_button_active(MouseButton::Right) {
        if !state.is_looking {
            state.last_cursor_position = input_manager.cursor_position;
            state.is_looking = true;
        } else {
            let cursor_delta = input_manager.cursor_position - state.last_cursor_position;
            state.last_cursor_position = input_manager.cursor_position;

            yaw_delta -= cursor_delta.x as f32 * state.sensitivity / 100.0;
            pitch_delta += cursor_delta.y as f32 * state.sensitivity / 100.0;
        }
    } else {
        state.is_looking = false;
    }

    if let Some(gamepad_id) = maybe_gamepad_id {
        yaw_delta -= input_manager.get_controller_axis(gamepad_id, gilrs::Axis::RightStickX)
            * CONTROLLER_SENSITIVITY
            * dt;
        pitch_delta -= input_manager.get_controller_axis(gamepad_id, gilrs::Axis::RightStickY)
            * CONTROLLER_SENSITIVITY
            * dt;
    }

    if !wants_pointer {
        state.speed += input_manager.mouse_wheel.y * 2.0;
    }

    if let Some(gamepad_id) = maybe_gamepad_id {
        if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::RightTrigger) {
            state.speed += 10.0 * dt;
        }
        if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::LeftTrigger) {
            state.speed -= 10.0 * dt;
        }
    }

    state.speed = state.speed.max(0.5);
    let mut speed = state.speed;

    let mut boost_active = input_manager.is_key_active(KeyCode::ShiftLeft);

    if let Some(gamepad_id) = maybe_gamepad_id {
        if input_manager.is_controller_button_active(gamepad_id, gilrs::Button::LeftThumb) {
            boost_active = true;
            speed *= 2.5;
        }
    }

    if boost_active {
        speed *= 2.5;
    }

    let mut apply_freecam =
        |entity: Entity, transform: &mut BevyTransform, camera: &mut BevyCamera| {
            if state.active_entity != Some(entity) {
                state.active_entity = Some(entity);
                state.current_fov_multiplier = 1.0;
                state.is_looking = false;
                state.last_cursor_position = input_manager.cursor_position;
            }

            let transform = &mut transform.0;
            let camera = &mut camera.0;

            let (mut yaw, mut pitch) = extract_yaw_pitch(transform.rotation);

            yaw += yaw_delta;
            pitch += pitch_delta;

            pitch = pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);

            let yaw_rot = Quat::from_axis_angle(Vec3::Y, yaw);
            let pitch_rot = Quat::from_axis_angle(Vec3::X, pitch);
            let orientation = yaw_rot * pitch_rot;

            transform.rotation = orientation;

            let forward = orientation * Vec3::Z;
            let right = orientation * -Vec3::X;

            let mut velocity = Vec3::ZERO;

            for key in &input_manager.active_keys {
                match key {
                    KeyCode::KeyW => velocity += forward,
                    KeyCode::KeyS => velocity -= forward,
                    KeyCode::KeyA => velocity -= right,
                    KeyCode::KeyD => velocity += right,
                    KeyCode::Space => velocity += Vec3::Y,
                    KeyCode::KeyC => velocity -= Vec3::Y,
                    _ => {}
                }
            }

            if let Some(gamepad_id) = maybe_gamepad_id {
                let lx = input_manager.get_controller_axis(gamepad_id, gilrs::Axis::LeftStickX);
                let ly = input_manager.get_controller_axis(gamepad_id, gilrs::Axis::LeftStickY);
                velocity += right * lx;
                velocity += forward * ly;

                let up = input_manager.get_right_trigger_value(gamepad_id);
                let down = input_manager.get_left_trigger_value(gamepad_id);
                velocity += Vec3::Y * up;
                velocity -= Vec3::Y * down;
            }

            if let Some(norm_velocity) = velocity.try_normalize() {
                transform.position += norm_velocity * speed * dt;
            }

            let target_multiplier = if boost_active { BOOST_AMOUNT } else { 1.0 };
            let safe_multiplier = state.current_fov_multiplier.clamp(0.01, BOOST_AMOUNT);
            let base_fov = camera.fov_y_rad / safe_multiplier;
            let t = 1.0 - (-state.fov_lerp_speed * dt).exp();
            state.current_fov_multiplier += (target_multiplier - state.current_fov_multiplier) * t;
            camera.fov_y_rad = base_fov * state.current_fov_multiplier;
        };

    match scene_state.world_state {
        WorldState::Edit => {
            for (entity, mut transform, mut camera) in viewport_query.iter_mut() {
                apply_freecam(entity, &mut transform, &mut camera);
            }
        }
        WorldState::Play => {
            for (entity, mut transform, mut camera) in play_query.iter_mut() {
                apply_freecam(entity, &mut transform, &mut camera);
            }
        }
    }
}

fn extract_yaw_pitch(rot: Quat) -> (f32, f32) {
    let forward = rot * Vec3::Z;
    let yaw = forward.x.atan2(forward.z);
    let pitch = (-forward.y).asin();
    (yaw, pitch)
}
