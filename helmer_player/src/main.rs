use std::{
    env,
    ffi::OsString,
    fs,
    path::{Component, Path, PathBuf},
    process,
    sync::OnceLock,
};

use bevy_ecs::{
    entity::Entity,
    prelude::{With, Without},
    schedule::{IntoScheduleConfigs, Schedule},
    world::World,
};
use helmer::runtime::asset_server::AssetServer;
use helmer_becs::{
    BevyAssetServer, BevyCamera, BevyInputManager, BevyLight, BevyTransform, helmer_becs_init,
    physics::systems::{
        apply_persistent_forces_system, apply_queued_impulses_system, apply_transient_forces_system,
    },
    systems::{
        animation_system::skinning_system,
        render_system::RenderMainSceneToSwapchain,
        render_system::render_data_system,
        scene_system::{SceneRoot, scene_spawning_system},
    },
};
use helmer_build_runtime::PackSetReader;
use helmer_editor::editor::{
    EditorAssetCache, EditorCommand, EditorCommandQueue, EditorConsoleState,
    EditorCursorControlState, EditorEntity, EditorGizmoState, EditorProject, EditorRenderRefresh,
    EditorSceneState, EditorTimelineState, EditorUndoState, EditorViewportCamera,
    EditorViewportRuntime, EditorViewportState, PendingSceneChildAnimations,
    PendingSceneChildPoseOverrides, PlayViewportKind, RustScriptRuntimeBuildPolicy, SceneAssetPath,
    ScriptEditModeState, ScriptInputState, ScriptRegistry, ScriptRunState, ScriptRuntime,
    ViewportRectPixels, WorldState, activate_play_camera, apply_scene_child_animations_system,
    apply_scene_child_pose_overrides_system, cached_scene_handle, ensure_play_camera,
    freecam_system, pending_scene_child_renderer_system, pending_skinned_mesh_system,
    read_scene_document, reset_editor_scene, restore_scene_transforms_from_document,
    script_execution_system, script_registry_system, set_play_camera, spawn_default_camera,
    spawn_default_light, spawn_scene_from_document,
};
use helmer_editor_runtime::{
    bundle::{BUILD_LAUNCH_MANIFEST_VERSION, BuildLaunchManifest, resolve_manifest_relative_path},
    project::{PROJECT_FILE_NAME, ProjectConfig},
};
use rayon::prelude::*;
use ron::ser::PrettyConfig;

static PLAYER_BOOTSTRAP: OnceLock<PlayerBootstrap> = OnceLock::new();

#[derive(Debug, Clone)]
struct PlayerArgs {
    manifest_path: PathBuf,
    key_override: Option<String>,
}

#[derive(Debug, Clone)]
struct PlayerBootstrap {
    project_root: PathBuf,
    startup_scene: PathBuf,
    project_config: ProjectConfig,
}

fn main() {
    #[cfg(target_os = "linux")]
    unsafe {
        if env::var_os("HELMER_FORCE_UNIX_BACKEND").is_none() {
            env::set_var("HELMER_FORCE_UNIX_BACKEND", "x11");
        }
    }

    if let Err(err) = run() {
        eprintln!("helmer_player: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args_os().skip(1).collect())?;
    let launch_manifest = read_launch_manifest(&args.manifest_path)?;
    let pack_key = args
        .key_override
        .unwrap_or_else(|| launch_manifest.pack_key.clone());

    let pack_manifest_path =
        resolve_manifest_relative_path(&args.manifest_path, &launch_manifest.pack_manifest);
    let project_root = materialize_project_assets(
        &launch_manifest,
        &args.manifest_path,
        &pack_manifest_path,
        &pack_key,
    )?;
    let startup_scene = resolve_startup_scene_path(
        &project_root,
        &launch_manifest.project_config,
        &launch_manifest.startup_scene,
    )?;

    let bootstrap = PlayerBootstrap {
        project_root,
        startup_scene,
        project_config: launch_manifest.project_config.clone(),
    };
    PLAYER_BOOTSTRAP
        .set(bootstrap)
        .map_err(|_| "player bootstrap was already initialized".to_string())?;

    helmer_becs_init(player_init);
    Ok(())
}

fn player_init(world: &mut World, schedule: &mut Schedule, _asset_server: &AssetServer) {
    let bootstrap = PLAYER_BOOTSTRAP
        .get()
        .cloned()
        .expect("player bootstrap missing");

    insert_player_resources(world, &bootstrap);
    if let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() {
        queue.push(EditorCommand::OpenScene {
            path: bootstrap.startup_scene.clone(),
        });
    }
    ensure_runtime_scene_defaults(world);

    schedule.add_systems(
        runtime_scene_command_system
            .before(scene_spawning_system)
            .before(skinning_system)
            .before(render_data_system),
    );
    schedule.add_systems(player_viewport_runtime_system.before(freecam_system));
    schedule.add_systems(
        freecam_system
            .after(helmer_becs::egui_integration::egui_system)
            .before(render_data_system),
    );
    schedule.add_systems(script_registry_system);
    schedule.add_systems(
        script_execution_system
            .before(apply_transient_forces_system)
            .before(apply_persistent_forces_system)
            .before(apply_queued_impulses_system),
    );
    schedule.add_systems(apply_scene_child_animations_system.after(scene_spawning_system));
    schedule.add_systems(apply_scene_child_pose_overrides_system.after(scene_spawning_system));
    schedule.add_systems(
        pending_scene_child_renderer_system
            .after(scene_spawning_system)
            .before(pending_skinned_mesh_system)
            .before(skinning_system)
            .before(render_data_system),
    );
    schedule.add_systems(
        pending_skinned_mesh_system
            .after(scene_spawning_system)
            .before(skinning_system)
            .before(render_data_system),
    );
}

fn insert_player_resources(world: &mut World, bootstrap: &PlayerBootstrap) {
    let mut scene_state = EditorSceneState::default();
    scene_state.world_state = WorldState::Play;

    let mut viewport_state = EditorViewportState::default();
    viewport_state.play_mode_view = PlayViewportKind::Gameplay;
    viewport_state.execute_scripts_in_edit_mode = false;

    world.insert_resource(EditorProject {
        root: Some(bootstrap.project_root.clone()),
        config: Some(bootstrap.project_config.clone()),
    });
    world.insert_resource(scene_state);
    world.insert_resource(EditorRenderRefresh::default());
    world.insert_resource(EditorUndoState::default());
    world.insert_resource(PendingSceneChildAnimations::default());
    world.insert_resource(PendingSceneChildPoseOverrides::default());
    world.insert_resource(EditorAssetCache::default());
    world.insert_resource(EditorConsoleState::default());
    world.insert_resource(EditorCommandQueue::default());
    world.insert_resource(ScriptRegistry::default());
    world.insert_resource(ScriptRuntime::default());
    world.insert_resource(RustScriptRuntimeBuildPolicy {
        allow_runtime_build: false,
    });
    world.insert_resource(ScriptRunState::default());
    world.insert_resource(ScriptEditModeState::default());
    world.insert_resource(ScriptInputState::default());
    world.insert_resource(EditorTimelineState::default());
    world.insert_resource(EditorViewportRuntime::default());
    world.insert_resource(EditorCursorControlState::default());
    world.insert_resource(EditorGizmoState::default());
    world.insert_resource(RenderMainSceneToSwapchain(true));
    world.insert_resource(viewport_state);
}

fn player_viewport_runtime_system(world: &mut World) {
    let (window_width, window_height) = world
        .get_resource::<BevyInputManager>()
        .map(|input| {
            let input = input.0.read();
            (input.window_size.x.max(1), input.window_size.y.max(1))
        })
        .unwrap_or((1, 1));

    let active_camera = ensure_play_camera(world).or_else(|| {
        world
            .query_filtered::<Entity, (
                With<helmer_editor::editor::EditorPlayCamera>,
                Without<EditorViewportCamera>,
            )>()
            .iter(world)
            .next()
    });

    if let Some(mut runtime) = world.get_resource_mut::<EditorViewportRuntime>() {
        runtime.main_rect_pixels =
            ViewportRectPixels::new(0.0, 0.0, window_width as f32, window_height as f32);
        runtime.main_target_size = Some([window_width, window_height]);
        runtime.main_resize_immediate = false;
        runtime.pointer_over_main = true;
        runtime.keyboard_focus = true;
        runtime.active_camera_entity = active_camera;
    }
}

fn runtime_scene_command_system(world: &mut World) {
    let commands = {
        let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() else {
            return;
        };
        std::mem::take(&mut queue.commands)
    };

    for command in commands {
        match command {
            EditorCommand::OpenScene { path } => {
                if let Err(err) = load_runtime_scene(world, &path) {
                    eprintln!(
                        "helmer_player: failed to switch scene {}: {}",
                        path.to_string_lossy(),
                        err
                    );
                    ensure_runtime_scene_defaults(world);
                }
            }
            EditorCommand::SetActiveCamera { entity } => {
                set_play_camera(world, entity);
                let _ = activate_play_camera(world);
            }
            _ => {}
        }
    }
}

fn load_runtime_scene(world: &mut World, path: &Path) -> Result<(), String> {
    if is_hscene_path(path) {
        let document = read_scene_document(path)?;
        let project_snapshot = world
            .get_resource::<EditorProject>()
            .cloned()
            .ok_or_else(|| "EditorProject resource is missing".to_string())?;

        reset_editor_scene(world);
        let created_entities = world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
            let asset_server = world
                .get_resource::<BevyAssetServer>()
                .expect("BevyAssetServer resource is missing");
            let asset_server = BevyAssetServer(asset_server.0.clone());
            spawn_scene_from_document(
                world,
                &document,
                &project_snapshot,
                &mut cache,
                &asset_server,
            )
        });
        restore_scene_transforms_from_document(world, &document, &created_entities);
    } else if is_gltf_path(path) {
        reset_editor_scene(world);
        let scene_path = path.to_path_buf();
        world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
            let asset_server = world
                .get_resource::<BevyAssetServer>()
                .expect("BevyAssetServer resource is missing");
            let asset_server = BevyAssetServer(asset_server.0.clone());
            let handle = cached_scene_handle(&mut cache, &asset_server, &scene_path);
            world.spawn((
                EditorEntity,
                BevyTransform::default(),
                SceneRoot(handle),
                SceneAssetPath {
                    path: scene_path.clone(),
                },
            ));
        });
    } else {
        return Err(format!(
            "unsupported scene format for '{}'",
            path.to_string_lossy()
        ));
    }

    ensure_runtime_scene_defaults(world);

    if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
        scene_state.path = Some(path.to_path_buf());
        scene_state.name = scene_display_name(path);
        scene_state.dirty = false;
        scene_state.play_backup = None;
        scene_state.play_selected_index = None;
    }

    Ok(())
}

fn ensure_runtime_scene_defaults(world: &mut World) {
    let has_scene_camera = world
        .query_filtered::<Entity, (
            With<BevyCamera>,
            Without<helmer_editor::editor::EditorViewportCamera>,
        )>()
        .iter(world)
        .next()
        .is_some();
    if !has_scene_camera {
        spawn_default_camera(world);
    }

    let has_light = world.query::<&BevyLight>().iter(world).next().is_some();
    if !has_light {
        spawn_default_light(world);
    }

    let _ = activate_play_camera(world);
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

fn materialize_project_assets(
    manifest: &BuildLaunchManifest,
    launch_manifest_path: &Path,
    pack_manifest_path: &Path,
    pack_key: &str,
) -> Result<PathBuf, String> {
    let bundle_root = launch_manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let cache_key = format!(
        "{}-{}",
        manifest.created_unix_ms,
        sanitize_cache_component(&manifest.key_fingerprint)
    );
    let project_root = bundle_root.join(".helmer_runtime").join(format!(
        "{}-{}",
        sanitize_cache_component(&manifest.project_name),
        cache_key
    ));
    let ready_marker = project_root.join(".ready");
    if ready_marker.exists() {
        return Ok(project_root);
    }

    if project_root.exists() {
        fs::remove_dir_all(&project_root).map_err(|err| {
            format!(
                "failed to clear stale runtime cache {}: {}",
                project_root.to_string_lossy(),
                err
            )
        })?;
    }
    fs::create_dir_all(&project_root).map_err(|err| {
        format!(
            "failed to create runtime cache root {}: {}",
            project_root.to_string_lossy(),
            err
        )
    })?;

    let reader = PackSetReader::open(pack_manifest_path, pack_key)?;
    let assets_root = project_root.join(&manifest.project_config.assets_dir);
    fs::create_dir_all(&assets_root).map_err(|err| {
        format!(
            "failed to create runtime assets root {}: {}",
            assets_root.to_string_lossy(),
            err
        )
    })?;

    let assets = reader.list_assets();
    assets.par_iter().try_for_each(|asset| {
        let safe_relative = safe_asset_relative_path(asset)?;
        let bytes = reader.read_asset(asset)?;
        let output_path = assets_root.join(safe_relative);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create extracted asset directory {}: {}",
                    parent.to_string_lossy(),
                    err
                )
            })?;
        }
        fs::write(&output_path, bytes).map_err(|err| {
            format!(
                "failed to write extracted asset {}: {}",
                output_path.to_string_lossy(),
                err
            )
        })?;
        Ok::<(), String>(())
    })?;

    write_project_config_file(&project_root, &manifest.project_config)?;
    fs::write(&ready_marker, b"ready").map_err(|err| {
        format!(
            "failed to write runtime cache marker {}: {}",
            ready_marker.to_string_lossy(),
            err
        )
    })?;

    Ok(project_root)
}

fn write_project_config_file(project_root: &Path, config: &ProjectConfig) -> Result<(), String> {
    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(4)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(config, pretty).map_err(|err| err.to_string())?;
    let path = project_root.join(PROJECT_FILE_NAME);
    fs::write(&path, data).map_err(|err| {
        format!(
            "failed to write extracted project config {}: {}",
            path.to_string_lossy(),
            err
        )
    })
}

fn resolve_startup_scene_path(
    project_root: &Path,
    config: &ProjectConfig,
    startup_scene: &str,
) -> Result<PathBuf, String> {
    let trimmed = startup_scene.trim();
    if trimmed.is_empty() {
        return Err("launch manifest startup_scene is empty".to_string());
    }

    let candidate = PathBuf::from(trimmed);
    let candidates = if candidate.is_absolute() {
        vec![candidate]
    } else {
        vec![
            project_root.join(&candidate),
            project_root.join(&config.assets_dir).join(&candidate),
            project_root.join(&config.scenes_dir).join(&candidate),
        ]
    };

    for path in candidates {
        if path.is_file() {
            return Ok(path);
        }
    }

    Err(format!(
        "startup scene '{}' was not found under extracted project {}",
        startup_scene,
        project_root.to_string_lossy()
    ))
}

fn safe_asset_relative_path(asset: &str) -> Result<PathBuf, String> {
    let mut safe = PathBuf::new();
    for component in Path::new(asset).components() {
        match component {
            Component::Normal(segment) => safe.push(segment),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("unsafe asset path in pack: '{asset}'"));
            }
        }
    }

    if safe.as_os_str().is_empty() {
        return Err("asset path in pack is empty".to_string());
    }

    Ok(safe)
}

fn sanitize_cache_component(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    output.trim_matches('_').to_string()
}

fn read_launch_manifest(path: &Path) -> Result<BuildLaunchManifest, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read launch manifest {}: {}",
            path.to_string_lossy(),
            err
        )
    })?;
    let manifest: BuildLaunchManifest = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse launch manifest {}: {}",
            path.to_string_lossy(),
            err
        )
    })?;

    if manifest.version > BUILD_LAUNCH_MANIFEST_VERSION {
        return Err(format!(
            "launch manifest version {} is newer than supported version {}",
            manifest.version, BUILD_LAUNCH_MANIFEST_VERSION
        ));
    }

    Ok(manifest)
}

fn is_hscene_path(path: &Path) -> bool {
    let path_lc = path.to_string_lossy().to_ascii_lowercase();
    path_lc.ends_with(".hscene.ron") || path_lc.ends_with(".hscene") || path_lc.ends_with(".scene")
}

fn is_gltf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            ext == "glb" || ext == "gltf"
        })
        .unwrap_or(false)
}

fn parse_args(args: Vec<OsString>) -> Result<PlayerArgs, String> {
    let mut manifest_path: Option<PathBuf> = None;
    let mut key_override: Option<String> = None;

    let mut index = 0usize;
    while index < args.len() {
        let value = args[index].to_string_lossy().to_string();
        match value.as_str() {
            "--manifest" => {
                index += 1;
                let next = args
                    .get(index)
                    .ok_or_else(|| "--manifest expects a path".to_string())?;
                manifest_path = Some(PathBuf::from(next));
            }
            "--key" => {
                index += 1;
                let next = args
                    .get(index)
                    .ok_or_else(|| "--key expects a value".to_string())?;
                key_override = Some(next.to_string_lossy().into_owned());
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => {
                if manifest_path.is_none() && !other.starts_with('-') {
                    manifest_path = Some(PathBuf::from(other));
                } else {
                    return Err(format!("unknown argument '{other}'"));
                }
            }
        }
        index += 1;
    }

    let manifest_path = manifest_path.unwrap_or_else(default_launch_manifest_path);
    Ok(PlayerArgs {
        manifest_path,
        key_override,
    })
}

fn default_launch_manifest_path() -> PathBuf {
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("helmer_player"));
    let stem = exe
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("helmer_project");
    exe.with_file_name(format!("{stem}.launch.json"))
}

fn print_usage() {
    eprintln!("Usage:\n  helmer_player [--manifest <launch_manifest.json>] [--key <pack_key>]");
}
