use std::{
    collections::{HashMap, HashSet},
    ffi::{CStr, c_char, c_void},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender, channel},
    },
    thread,
    time::{Duration, Instant, SystemTime},
};

use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Component, Entity, Resource, World};
use bevy_ecs::query::With;
use glam::{DVec2, Quat, Vec2, Vec3};
use helmer_editor_runtime::scripting as runtime_scripting;
use helmer_script_sdk::{
    EntityId as RustScriptEntityId, Quat as RustScriptQuat, SCRIPT_API_ABI_VERSION,
    SCRIPT_PLUGIN_ABI_VERSION, ScriptApi as RustScriptApi, ScriptBytes as RustScriptBytes,
    ScriptBytesView as RustScriptBytesView, ScriptPlugin as RustScriptPluginApi,
    Transform as RustScriptTransform, TransformPatch as RustScriptTransformPatch,
    Vec3 as RustScriptVec3,
};
use libloading::{Library, Symbol};
use mlua::{Function, Lua, MultiValue, RegistryKey, Table, UserData, Value, Variadic};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};
use winit::{event::MouseButton, keyboard::KeyCode};

use helmer::audio::{AudioBus, AudioPlaybackState};
use helmer::provided::components::{
    AudioEmitter, AudioListener, EntityFollower, Light, LightType, LookAt, MeshAsset, MeshRenderer,
    Spline, SplineFollower, SplineMode,
};
use helmer::runtime::asset_server::{Handle, Material, Mesh};
use helmer::runtime::input_manager::InputManager;
use helmer::runtime::runtime::RuntimeCursorGrabMode;
use helmer_becs::physics::components::{
    CharacterController, CharacterControllerInput, CharacterControllerOutput, ColliderProperties,
    ColliderPropertyInheritance, ColliderShape, DynamicRigidBody, FixedCollider, KinematicMode,
    KinematicRigidBody, MeshColliderKind, MeshColliderLod, PhysicsCombineRule, PhysicsHandle,
    PhysicsJoint, PhysicsJointKind, PhysicsPointProjection, PhysicsPointProjectionHit,
    PhysicsQueryFilter, PhysicsRayCast, PhysicsRayCastHit, PhysicsShapeCast, PhysicsShapeCastHit,
    PhysicsShapeCastStatus, PhysicsWorldDefaults, PointForce, PointImpulse, RigidBodyForces,
    RigidBodyImpulseQueue, RigidBodyProperties, RigidBodyPropertyInheritance,
    RigidBodyTransientForces,
};
use helmer_becs::physics::physics_resource::PhysicsResource;
use helmer_becs::systems::scene_system::SceneRoot;
use helmer_becs::{
    AudioBackendResource, BevyActiveCamera, BevyAnimator, BevyAssetServer, BevyAudioEmitter,
    BevyAudioListener, BevyCamera, BevyEntityFollower, BevyInputManager, BevyLight, BevyLookAt,
    BevyMeshRenderer, BevySpline, BevySplineFollower, BevyTransform, BevyWrapper, DeltaTime,
};

use crate::editor::{
    EditorPlayCamera, activate_play_camera,
    assets::{
        EditorAssetCache, EditorAudio, EditorMesh, MeshSource, PrimitiveKind, SceneAssetPath,
        cached_audio_handle, cached_scene_handle,
    },
    commands::{EditorCommand, EditorCommandQueue},
    dynamic::{DynamicComponent, DynamicComponents, DynamicField, DynamicValue},
    project::EditorProject,
    scene::{EditorEntity, EditorSceneState, WorldState, reset_scene_root_instance},
    set_play_camera,
    viewport::{
        EditorCursorControlState, EditorViewportRuntime, EditorViewportState, PlayViewportKind,
    },
    visual_scripting::{
        VisualScriptApiTable, VisualScriptEvent, VisualScriptHost, VisualScriptProgram,
        VisualScriptRuntimeState, compile_visual_script_runtime_source,
    },
};

const RUST_SCRIPT_PLUGIN_SYMBOL: &[u8] = b"helmer_get_script_plugin\0";

#[derive(Debug, Clone)]
pub struct ScriptEntry {
    pub path: Option<PathBuf>,
    pub language: String,
}

impl ScriptEntry {
    pub fn new() -> Self {
        Self {
            path: None,
            language: "luau".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptEditCommand {
    Run(ScriptInstanceKey),
    Stop(ScriptInstanceKey),
    Restart(ScriptInstanceKey),
    StopAll,
}

#[derive(Resource, Default)]
pub struct ScriptEditModeState {
    pub running_in_edit: HashSet<ScriptInstanceKey>,
    pub pending_commands: Vec<ScriptEditCommand>,
}

impl ScriptEditModeState {
    pub fn queue(&mut self, command: ScriptEditCommand) {
        self.pending_commands.push(command);
    }
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct RustScriptRuntimeBuildPolicy {
    pub allow_runtime_build: bool,
}

impl Default for RustScriptRuntimeBuildPolicy {
    fn default() -> Self {
        Self {
            allow_runtime_build: true,
        }
    }
}

#[derive(Component, Debug, Clone)]
pub struct ScriptComponent {
    pub scripts: Vec<ScriptEntry>,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct ScriptSpawned {
    pub owner: Entity,
    pub script_index: usize,
}

#[derive(Debug, Clone)]
pub struct ScriptAsset {
    pub language: String,
    pub source: String,
    pub modified: SystemTime,
    pub error: Option<String>,
}

#[derive(Resource)]
pub struct ScriptRegistry {
    pub scripts: HashMap<PathBuf, ScriptAsset>,
    pub dirty_paths: HashSet<PathBuf>,
    pub last_scan: Instant,
    pub scan_interval: Duration,
    pub status: Option<String>,
}

impl Default for ScriptRegistry {
    fn default() -> Self {
        Self {
            scripts: HashMap::new(),
            dirty_paths: HashSet::new(),
            last_scan: Instant::now(),
            scan_interval: Duration::from_millis(500),
            status: None,
        }
    }
}

impl ScriptRegistry {
    pub fn mark_dirty_paths(&mut self, paths: &HashSet<PathBuf>) {
        for path in paths {
            self.dirty_paths.insert(path.clone());
        }
    }

    pub fn take_dirty_paths(&mut self) -> HashSet<PathBuf> {
        std::mem::take(&mut self.dirty_paths)
    }
}

pub fn is_script_path(path: &Path) -> bool {
    runtime_scripting::is_script_path(path)
}

pub fn is_lua_script_path(path: &Path) -> bool {
    runtime_scripting::is_lua_script_path(path)
}

pub fn is_visual_script_path(path: &Path) -> bool {
    runtime_scripting::is_visual_script_path(path)
}

pub fn is_rust_script_path(path: &Path) -> bool {
    runtime_scripting::is_rust_script_path(path)
}

pub fn resolve_rust_script_manifest(path: &Path) -> Option<PathBuf> {
    runtime_scripting::resolve_rust_script_manifest(path)
}

pub fn script_registry_key_for_path(path: &Path) -> Option<PathBuf> {
    runtime_scripting::script_registry_key_for_path(path)
}

pub fn script_language_from_path(path: &Path) -> String {
    runtime_scripting::script_language_from_path(path)
}

pub fn normalize_script_language(language: &str, path: Option<&Path>) -> String {
    let normalized = language.trim().to_ascii_lowercase();
    if normalized == "lua" || normalized == "luau" || normalized == "visual" || normalized == "rust"
    {
        return normalized;
    }
    if let Some(path) = path {
        return script_language_from_path(path);
    }
    "luau".to_string()
}

fn rust_script_modified_time(manifest_path: &Path) -> SystemTime {
    let mut latest = fs::metadata(manifest_path)
        .and_then(|meta| meta.modified())
        .unwrap_or_else(|_| SystemTime::UNIX_EPOCH);

    let Some(root) = manifest_path.parent() else {
        return latest;
    };

    let src_root = root.join("src");
    if src_root.exists() {
        for entry in walkdir::WalkDir::new(&src_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            if let Ok(modified) = fs::metadata(entry.path()).and_then(|meta| meta.modified()) {
                if modified > latest {
                    latest = modified;
                }
            }
        }
    }

    let build_rs = root.join("build.rs");
    if let Ok(modified) = fs::metadata(&build_rs).and_then(|meta| meta.modified()) {
        if modified > latest {
            latest = modified;
        }
    }

    latest
}

pub fn load_script_asset(path: &Path) -> ScriptAsset {
    let language = script_language_from_path(path);
    if language == "visual" {
        let modified = fs::metadata(path)
            .and_then(|meta| meta.modified())
            .unwrap_or_else(|_| SystemTime::now());

        match fs::read_to_string(path) {
            Ok(source) => {
                let error =
                    compile_visual_script_runtime_source(&source, &path.to_string_lossy()).err();
                ScriptAsset {
                    language,
                    source,
                    modified,
                    error,
                }
            }
            Err(err) => ScriptAsset {
                language,
                source: String::new(),
                modified,
                error: Some(err.to_string()),
            },
        }
    } else if language == "rust" {
        let Some(manifest_path) = resolve_rust_script_manifest(path) else {
            return ScriptAsset {
                language,
                source: String::new(),
                modified: SystemTime::now(),
                error: Some("Rust script manifest not found".to_string()),
            };
        };

        let modified = rust_script_modified_time(&manifest_path);
        match fs::read_to_string(&manifest_path) {
            Ok(source) => ScriptAsset {
                language,
                source,
                modified,
                error: None,
            },
            Err(err) => ScriptAsset {
                language,
                source: String::new(),
                modified,
                error: Some(err.to_string()),
            },
        }
    } else {
        let modified = fs::metadata(path)
            .and_then(|meta| meta.modified())
            .unwrap_or_else(|_| SystemTime::now());

        match fs::read_to_string(path) {
            Ok(source) => ScriptAsset {
                language,
                source,
                modified,
                error: None,
            },
            Err(err) => ScriptAsset {
                language,
                source: String::new(),
                modified,
                error: Some(err.to_string()),
            },
        }
    }
}

#[derive(Debug)]
pub struct ScriptInstance {
    pub path: PathBuf,
    pub env_key: RegistryKey,
    pub modified: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScriptInstanceKey {
    pub entity: Entity,
    pub script_index: usize,
}

#[derive(Debug)]
struct RustBuildResult {
    manifest_path: PathBuf,
    source_modified: SystemTime,
    artifact_path: Option<PathBuf>,
    output: String,
    error: Option<String>,
}

#[allow(dead_code)]
pub struct RustLoadedPlugin {
    manifest_path: PathBuf,
    source_library_path: PathBuf,
    loaded_library_path: PathBuf,
    version_id: u64,
    source_modified: SystemTime,
    plugin_ptr: *const RustScriptPluginApi,
    library: Arc<Mutex<Library>>,
}

// SAFETY: Rust script plugins are only accessed from the main ECS thread
unsafe impl Send for RustLoadedPlugin {}
// SAFETY: Rust script plugins are only accessed from the main ECS thread
unsafe impl Sync for RustLoadedPlugin {}

pub struct RustScriptHostContext {
    world_ptr: usize,
    owner: Entity,
    script_index: usize,
}

// SAFETY: Script host context is only mutated on the main ECS thread
unsafe impl Send for RustScriptHostContext {}
// SAFETY: Script host context is only mutated on the main ECS thread
unsafe impl Sync for RustScriptHostContext {}

#[allow(dead_code)]
pub struct RustScriptInstance {
    manifest_path: PathBuf,
    plugin_version_id: u64,
    plugin_instance_ptr: *mut c_void,
    plugin: Arc<RustLoadedPlugin>,
    host_context: Box<RustScriptHostContext>,
    host_api: Box<RustScriptApi>,
}

// SAFETY: Script instances are only created, updated, and destroyed on the main ECS thread
unsafe impl Send for RustScriptInstance {}
// SAFETY: Script instances are only created, updated, and destroyed on the main ECS thread
unsafe impl Sync for RustScriptInstance {}

#[derive(Debug)]
pub struct VisualScriptInstance {
    pub path: PathBuf,
    pub program: VisualScriptProgram,
    pub state: VisualScriptRuntimeState,
    pub modified: SystemTime,
}

#[derive(Resource)]
pub struct ScriptRuntime {
    pub lua: Arc<Mutex<Lua>>,
    pub instances: HashMap<ScriptInstanceKey, ScriptInstance>,
    pub visual_instances: HashMap<ScriptInstanceKey, VisualScriptInstance>,
    pub rust_instances: HashMap<ScriptInstanceKey, RustScriptInstance>,
    rust_plugins: HashMap<PathBuf, Arc<RustLoadedPlugin>>,
    rust_inflight_builds: HashSet<PathBuf>,
    rust_build_sender: Sender<RustBuildResult>,
    rust_build_receiver: Arc<Mutex<Receiver<RustBuildResult>>>,
    rust_build_errors: HashMap<PathBuf, String>,
    pub rust_status: Option<String>,
    next_rust_plugin_version: u64,
    pub errors: Vec<String>,
}

impl Default for ScriptRuntime {
    fn default() -> Self {
        let (rust_build_sender, rust_build_receiver) = channel();
        Self {
            lua: Arc::new(Mutex::new(Lua::new())),
            instances: HashMap::new(),
            visual_instances: HashMap::new(),
            rust_instances: HashMap::new(),
            rust_plugins: HashMap::new(),
            rust_inflight_builds: HashSet::new(),
            rust_build_sender,
            rust_build_receiver: Arc::new(Mutex::new(rust_build_receiver)),
            rust_build_errors: HashMap::new(),
            rust_status: None,
            next_rust_plugin_version: 1,
            errors: Vec::new(),
        }
    }
}

impl ScriptRuntime {
    pub fn clear_all(&mut self) {
        self.instances.clear();
        self.visual_instances.clear();
        self.rust_instances.clear();
        self.rust_plugins.clear();
        self.rust_inflight_builds.clear();
        self.rust_build_errors.clear();
        self.rust_status = None;
        self.errors.clear();
    }

    pub fn contains_instance(&self, key: ScriptInstanceKey) -> bool {
        self.instances.contains_key(&key)
            || self.visual_instances.contains_key(&key)
            || self.rust_instances.contains_key(&key)
    }

    fn request_rust_build(&mut self, manifest_path: &Path) {
        if self.rust_inflight_builds.contains(manifest_path) {
            return;
        }

        self.rust_inflight_builds
            .insert(manifest_path.to_path_buf());
        let sender = self.rust_build_sender.clone();
        let manifest = manifest_path.to_path_buf();
        thread::spawn(move || {
            let _ = sender.send(build_rust_script_library(&manifest));
        });
    }

    fn take_rust_build_results(&self) -> Vec<RustBuildResult> {
        let Ok(receiver) = self.rust_build_receiver.lock() else {
            return Vec::new();
        };
        receiver.try_iter().collect::<Vec<_>>()
    }
}

#[derive(Resource)]
pub struct ScriptRunState {
    pub last_state: WorldState,
    pub last_executing: bool,
}

impl Default for ScriptRunState {
    fn default() -> Self {
        Self {
            last_state: WorldState::Edit,
            last_executing: false,
        }
    }
}

#[derive(Resource, Default)]
pub struct ScriptInputState {
    pub last_keys: HashSet<KeyCode>,
    pub last_mouse_buttons: HashSet<MouseButton>,
    pub last_gamepad_buttons: HashMap<usize, HashSet<gilrs::Button>>,
    pub last_cursor_position: DVec2,
    pub gamepad_map: HashMap<usize, gilrs::GamepadId>,
    pub gamepad_ids: Vec<usize>,
}

impl ScriptInputState {
    pub fn refresh_gamepad_cache(&mut self, input: &InputManager) {
        self.gamepad_map.clear();
        self.gamepad_ids.clear();

        for id in input.controller_states.keys() {
            let id_value = usize::from(*id);
            self.gamepad_map.insert(id_value, *id);
            self.gamepad_ids.push(id_value);
        }

        self.gamepad_ids.sort_unstable();
    }

    pub fn sync_last_state(&mut self, input: &InputManager) {
        self.last_keys.clear();
        self.last_keys.extend(input.active_keys.iter().copied());

        self.last_mouse_buttons.clear();
        self.last_mouse_buttons
            .extend(input.active_mouse_buttons.iter().copied());

        self.last_cursor_position = input.cursor_position;

        let mut active_ids = HashSet::new();
        for (id, state) in input.controller_states.iter() {
            let id_value = usize::from(*id);
            active_ids.insert(id_value);
            let entry = self.last_gamepad_buttons.entry(id_value).or_default();
            entry.clear();
            entry.extend(state.active_buttons.iter().copied());
        }

        self.last_gamepad_buttons
            .retain(|id, _| active_ids.contains(id));
    }

    pub fn reset(&mut self, input: Option<&InputManager>) {
        self.last_keys.clear();
        self.last_mouse_buttons.clear();
        self.last_gamepad_buttons.clear();
        self.last_cursor_position = DVec2::ZERO;

        let Some(input) = input else {
            return;
        };

        self.last_keys.extend(input.active_keys.iter().copied());
        self.last_mouse_buttons
            .extend(input.active_mouse_buttons.iter().copied());
        self.last_cursor_position = input.cursor_position;

        for (id, state) in input.controller_states.iter() {
            let id_value = usize::from(*id);
            let entry = self.last_gamepad_buttons.entry(id_value).or_default();
            entry.clear();
            entry.extend(state.active_buttons.iter().copied());
        }
    }
}

pub fn script_execution_system(world: &mut World) {
    let current_state = world
        .get_resource::<EditorSceneState>()
        .map(|state| state.world_state)
        .unwrap_or(WorldState::Edit);
    let execute_scripts_in_edit_mode = world
        .get_resource::<EditorViewportState>()
        .map(|state| state.execute_scripts_in_edit_mode)
        .unwrap_or(false);
    let allow_runtime_rust_build = world
        .get_resource::<RustScriptRuntimeBuildPolicy>()
        .map(|policy| policy.allow_runtime_build)
        .unwrap_or(true);
    let dt = world
        .get_resource::<DeltaTime>()
        .map(|time| time.0)
        .unwrap_or(0.0);

    let script_assets = match world.get_resource::<ScriptRegistry>() {
        Some(registry) => registry.scripts.clone(),
        None => return,
    };

    let scripts = {
        let mut query = world.query::<(Entity, &ScriptComponent)>();
        let mut entries = Vec::new();
        for (entity, script_component) in query.iter(world) {
            for (script_index, script) in script_component.scripts.iter().enumerate() {
                let Some(path) = script.path.as_ref() else {
                    continue;
                };

                let language = normalize_script_language(&script.language, Some(path));
                let runtime_path = if language == "rust" {
                    resolve_rust_script_manifest(path).unwrap_or_else(|| path.clone())
                } else {
                    path.clone()
                };

                let asset = script_assets
                    .get(&runtime_path)
                    .cloned()
                    .unwrap_or_else(|| load_script_asset(&runtime_path));
                entries.push((
                    ScriptInstanceKey {
                        entity,
                        script_index,
                    },
                    script.clone(),
                    runtime_path,
                    asset,
                ));
            }
        }
        entries
    };
    let scripts_by_key = scripts
        .iter()
        .map(|(key, script, runtime_path, asset)| {
            (*key, (script.clone(), runtime_path.clone(), asset.clone()))
        })
        .collect::<HashMap<_, _>>();
    let script_keys = scripts_by_key.keys().copied().collect::<HashSet<_>>();

    let mut force_reload = HashSet::new();
    if let Some(mut edit_mode) = world.get_resource_mut::<ScriptEditModeState>() {
        let pending = std::mem::take(&mut edit_mode.pending_commands);
        for command in pending {
            match command {
                ScriptEditCommand::Run(key) => {
                    if script_keys.contains(&key) {
                        edit_mode.running_in_edit.insert(key);
                    }
                }
                ScriptEditCommand::Stop(key) => {
                    edit_mode.running_in_edit.remove(&key);
                }
                ScriptEditCommand::Restart(key) => {
                    if script_keys.contains(&key) {
                        edit_mode.running_in_edit.insert(key);
                        force_reload.insert(key);
                    }
                }
                ScriptEditCommand::StopAll => {
                    edit_mode.running_in_edit.clear();
                }
            }
        }
        edit_mode
            .running_in_edit
            .retain(|key| script_keys.contains(key));
    }

    let desired_active = if current_state == WorldState::Play
        || (current_state == WorldState::Edit && execute_scripts_in_edit_mode)
    {
        script_keys.clone()
    } else {
        world
            .get_resource::<ScriptEditModeState>()
            .map(|edit_mode| {
                edit_mode
                    .running_in_edit
                    .iter()
                    .copied()
                    .filter(|key| script_keys.contains(key))
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default()
    };
    let is_executing = !desired_active.is_empty();

    let input_manager = world
        .get_resource::<BevyInputManager>()
        .map(|input| input.0.clone());

    if let Some(input_manager) = input_manager.as_ref() {
        let input_manager = input_manager.read();
        if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
            input_state.refresh_gamepad_cache(&input_manager);
        }
    }

    let (last_state, last_executing) = world
        .get_resource::<ScriptRunState>()
        .map(|state| (state.last_state, state.last_executing))
        .unwrap_or((WorldState::Edit, false));
    let lifecycle_changed = last_state != current_state || last_executing != is_executing;

    world.resource_scope::<ScriptRuntime, _>(|world, mut runtime| {
        runtime.errors.clear();

        let completed_builds = runtime.take_rust_build_results();
        for result in completed_builds {
            runtime.rust_inflight_builds.remove(&result.manifest_path);
            for warning in rust_build_warning_lines(&result.output) {
                tracing::warn!(
                    target: "script.rust.build",
                    "{}: {}",
                    result.manifest_path.to_string_lossy(),
                    warning
                );
            }

            if let Some(error) = result.error {
                let mut message = error;
                if !result.output.trim().is_empty() {
                    message.push('\n');
                    message.push_str(&truncate_diagnostic(&result.output));
                }
                tracing::error!(
                    target: "script.rust.build",
                    "{}: {}",
                    result.manifest_path.to_string_lossy(),
                    message
                );
                runtime
                    .rust_build_errors
                    .insert(result.manifest_path.clone(), message);
                runtime.rust_status = Some(format!(
                    "Rust build failed: {}",
                    result.manifest_path.to_string_lossy()
                ));
                continue;
            }

            let Some(artifact_path) = result.artifact_path.clone() else {
                let message = format!(
                    "No build artifact produced for {}",
                    result.manifest_path.to_string_lossy()
                );
                tracing::error!(target: "script.rust.build", "{message}");
                runtime
                    .rust_build_errors
                    .insert(result.manifest_path.clone(), message.clone());
                runtime.rust_status = Some(message);
                continue;
            };

            let version_id = runtime.next_rust_plugin_version;
            runtime.next_rust_plugin_version = runtime.next_rust_plugin_version.saturating_add(1);
            match load_rust_plugin(
                &result.manifest_path,
                &artifact_path,
                result.source_modified,
                version_id,
            ) {
                Ok(plugin) => {
                    runtime
                        .rust_plugins
                        .insert(result.manifest_path.clone(), Arc::new(plugin));
                    runtime.rust_build_errors.remove(&result.manifest_path);
                    tracing::info!(
                        target: "script.rust.build",
                        "Rust script loaded: {}",
                        result.manifest_path.to_string_lossy()
                    );
                    runtime.rust_status = Some(format!(
                        "Rust script loaded: {}",
                        result.manifest_path.to_string_lossy()
                    ));
                }
                Err(err) => {
                    let message = format!("Failed to load Rust script plugin: {}", err);
                    tracing::error!(
                        target: "script.rust.build",
                        "{}: {}",
                        result.manifest_path.to_string_lossy(),
                        message
                    );
                    runtime
                        .rust_build_errors
                        .insert(result.manifest_path.clone(), message.clone());
                    runtime.rust_status = Some(message);
                }
            }
        }

        let lua = runtime.lua.clone();
        let lua = match lua.lock() {
            Ok(lua) => lua,
            Err(_) => {
                if let Some(mut run_state) = world.get_resource_mut::<ScriptRunState>() {
                    run_state.last_state = current_state;
                    run_state.last_executing = is_executing;
                }
                return;
            }
        };

        let world_ptr = world as *mut World as usize;

        if lifecycle_changed {
            if let Some(input_manager) = input_manager.as_ref() {
                let input_manager = input_manager.read();
                if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
                    input_state.reset(Some(&input_manager));
                }
            } else if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
                input_state.reset(None);
            }
        }

        if last_state != current_state {
            let drained = runtime.instances.drain().collect::<Vec<_>>();
            for (key, instance) in drained {
                let _ = call_script_function0(&lua, &instance, "on_stop");
                let _ = lua.remove_registry_value(instance.env_key);
                despawn_script_owned(world, key.entity, key.script_index);
            }

            let drained_rust = runtime.rust_instances.drain().collect::<Vec<_>>();
            for (key, instance) in drained_rust {
                stop_rust_script_instance(world, key, instance);
            }

            let drained_visual = runtime.visual_instances.drain().collect::<Vec<_>>();
            for (key, instance) in drained_visual {
                stop_visual_script_instance(world, world_ptr, key, instance);
            }
        }

        let mut active_scripts = HashSet::new();
        for key in desired_active.iter().copied() {
            let Some((script, runtime_path, asset)) = scripts_by_key.get(&key) else {
                continue;
            };
            active_scripts.insert(key);

            if let Some(error) = asset.error.as_ref() {
                if let Some(old_instance) = runtime.instances.remove(&key) {
                    let _ = call_script_function0(&lua, &old_instance, "on_stop");
                    let _ = lua.remove_registry_value(old_instance.env_key);
                    despawn_script_owned(world, key.entity, key.script_index);
                }
                if let Some(old_instance) = runtime.visual_instances.remove(&key) {
                    stop_visual_script_instance(world, world_ptr, key, old_instance);
                }
                if let Some(old_instance) = runtime.rust_instances.remove(&key) {
                    stop_rust_script_instance(world, key, old_instance);
                }
                push_script_runtime_error(
                    &mut runtime,
                    format!("{}: {}", script_path_label(script), error),
                );
                continue;
            }

            let language = asset.language.as_str();
            if language == "rust" {
                if let Some(old_instance) = runtime.instances.remove(&key) {
                    let _ = call_script_function0(&lua, &old_instance, "on_stop");
                    let _ = lua.remove_registry_value(old_instance.env_key);
                    despawn_script_owned(world, key.entity, key.script_index);
                }
                if let Some(old_instance) = runtime.visual_instances.remove(&key) {
                    stop_visual_script_instance(world, world_ptr, key, old_instance);
                }

                let manifest_path = runtime_path.clone();
                if allow_runtime_rust_build {
                    let build_needed = runtime
                        .rust_plugins
                        .get(&manifest_path)
                        .map(|plugin| asset.modified > plugin.source_modified)
                        .unwrap_or(true);
                    if build_needed {
                        runtime.request_rust_build(&manifest_path);
                    }
                } else {
                    let prebuilt_library_path =
                        match resolve_prebuilt_rust_plugin_path(world, &manifest_path) {
                            Ok(path) => path,
                            Err(err) => {
                                runtime
                                    .rust_build_errors
                                    .insert(manifest_path.clone(), err.clone());
                                push_script_runtime_error(
                                    &mut runtime,
                                    format!("{}: {}", script_path_label(script), err),
                                );
                                continue;
                            }
                        };
                    let prebuilt_modified = fs::metadata(&prebuilt_library_path)
                        .and_then(|meta| meta.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH);
                    let prebuilt_needs_load = runtime
                        .rust_plugins
                        .get(&manifest_path)
                        .map(|plugin| {
                            plugin.source_library_path != prebuilt_library_path
                                || plugin.source_modified < prebuilt_modified
                        })
                        .unwrap_or(true);
                    if prebuilt_needs_load {
                        let version_id = runtime.next_rust_plugin_version;
                        runtime.next_rust_plugin_version =
                            runtime.next_rust_plugin_version.saturating_add(1);
                        match load_rust_plugin(
                            &manifest_path,
                            &prebuilt_library_path,
                            prebuilt_modified,
                            version_id,
                        ) {
                            Ok(plugin) => {
                                runtime
                                    .rust_plugins
                                    .insert(manifest_path.clone(), Arc::new(plugin));
                                runtime.rust_build_errors.remove(&manifest_path);
                            }
                            Err(err) => {
                                let message =
                                    format!("Failed to load prebuilt Rust script plugin: {}", err);
                                runtime
                                    .rust_build_errors
                                    .insert(manifest_path.clone(), message.clone());
                                push_script_runtime_error(
                                    &mut runtime,
                                    format!("{}: {}", script_path_label(script), message),
                                );
                                continue;
                            }
                        }
                    }
                }

                let plugin = runtime.rust_plugins.get(&manifest_path).cloned();
                let Some(plugin) = plugin else {
                    if let Some(error) = runtime.rust_build_errors.get(&manifest_path).cloned() {
                        push_script_runtime_error(
                            &mut runtime,
                            format!("{}: {}", script_path_label(script), error),
                        );
                    }
                    continue;
                };

                let reload = if force_reload.contains(&key) {
                    true
                } else {
                    match runtime.rust_instances.get(&key) {
                        Some(instance) => {
                            instance.manifest_path != manifest_path
                                || instance.plugin_version_id != plugin.version_id
                        }
                        None => true,
                    }
                };

                if reload {
                    let mut carry_state: Option<Vec<u8>> = None;
                    if let Some(mut old_instance) = runtime.rust_instances.remove(&key) {
                        let plugin_changed = old_instance.plugin_version_id != plugin.version_id;
                        let same_script = old_instance.manifest_path == manifest_path;
                        if plugin_changed && same_script && !force_reload.contains(&key) {
                            match take_rust_script_state(&mut old_instance) {
                                Ok(state) => carry_state = state,
                                Err(err) => push_script_runtime_error(&mut runtime, err),
                            }
                        }
                        stop_rust_script_instance(world, key, old_instance);
                    }
                    match create_rust_script_instance(
                        world_ptr,
                        key.entity,
                        key.script_index,
                        &manifest_path,
                        plugin,
                    ) {
                        Ok(mut instance) => {
                            if let Err(err) = call_rust_script_start(&mut instance) {
                                push_script_runtime_error(&mut runtime, err);
                            }
                            if let Some(state) = carry_state.as_deref() {
                                match call_rust_script_load_state(&mut instance, state) {
                                    Ok(true) => {}
                                    Ok(false) => push_script_runtime_error(
                                        &mut runtime,
                                        format!(
                                            "Rust script rejected carried state: {}",
                                            manifest_path.to_string_lossy()
                                        ),
                                    ),
                                    Err(err) => push_script_runtime_error(&mut runtime, err),
                                }
                            }
                            runtime.rust_instances.insert(key, instance);
                        }
                        Err(err) => {
                            push_script_runtime_error(&mut runtime, err);
                            continue;
                        }
                    }
                }

                if let Some(instance) = runtime.rust_instances.get_mut(&key) {
                    instance.host_context.world_ptr = world_ptr;
                    if let Err(err) = call_rust_script_update(instance, dt) {
                        push_script_runtime_error(&mut runtime, err);
                    }
                }

                continue;
            }

            if language == "visual" {
                if let Some(old_instance) = runtime.instances.remove(&key) {
                    let _ = call_script_function0(&lua, &old_instance, "on_stop");
                    let _ = lua.remove_registry_value(old_instance.env_key);
                    despawn_script_owned(world, key.entity, key.script_index);
                }
                if let Some(old_instance) = runtime.rust_instances.remove(&key) {
                    stop_rust_script_instance(world, key, old_instance);
                }

                let reload = if force_reload.contains(&key) {
                    true
                } else {
                    match runtime.visual_instances.get(&key) {
                        Some(instance) => visual_script_needs_reload(instance, runtime_path, asset),
                        None => true,
                    }
                };

                if reload {
                    if let Some(old_instance) = runtime.visual_instances.remove(&key) {
                        stop_visual_script_instance(world, world_ptr, key, old_instance);
                    }

                    match load_visual_script_instance(runtime_path, asset) {
                        Ok(mut instance) => {
                            if let Err(err) = run_visual_script_event(
                                world_ptr,
                                key,
                                &mut instance,
                                VisualScriptEvent::Start,
                                0.0,
                            ) {
                                push_script_runtime_error(
                                    &mut runtime,
                                    format!("{}: {}", script_path_label(script), err),
                                );
                            }
                            runtime.visual_instances.insert(key, instance);
                        }
                        Err(err) => {
                            push_script_runtime_error(
                                &mut runtime,
                                format!("{}: {}", script_path_label(script), err),
                            );
                            continue;
                        }
                    }
                }

                if let Some(instance) = runtime.visual_instances.get_mut(&key) {
                    if let Err(err) = run_visual_script_event(
                        world_ptr,
                        key,
                        instance,
                        VisualScriptEvent::Update,
                        dt,
                    ) {
                        push_script_runtime_error(
                            &mut runtime,
                            format!("{}: {}", script_path_label(script), err),
                        );
                    }
                }

                continue;
            }

            if let Some(old_instance) = runtime.rust_instances.remove(&key) {
                stop_rust_script_instance(world, key, old_instance);
            }
            if let Some(old_instance) = runtime.visual_instances.remove(&key) {
                stop_visual_script_instance(world, world_ptr, key, old_instance);
            }

            let reload = if force_reload.contains(&key) {
                true
            } else {
                match runtime.instances.get(&key) {
                    Some(instance) => script_needs_reload(instance, runtime_path, asset),
                    None => true,
                }
            };

            if reload {
                if let Some(old_instance) = runtime.instances.remove(&key) {
                    let _ = call_script_function0(&lua, &old_instance, "on_stop");
                    let _ = lua.remove_registry_value(old_instance.env_key);
                    despawn_script_owned(world, key.entity, key.script_index);
                }

                match load_script_instance(
                    &lua,
                    world_ptr,
                    key.entity,
                    key.script_index,
                    runtime_path,
                    asset,
                ) {
                    Ok(instance) => {
                        runtime.instances.insert(key, instance);
                        if let Some(instance) = runtime.instances.get(&key) {
                            if let Err(err) = call_script_function0(&lua, instance, "on_start") {
                                push_script_runtime_error(&mut runtime, err);
                            }
                        }
                    }
                    Err(err) => {
                        push_script_runtime_error(&mut runtime, err);
                        continue;
                    }
                }
            }

            if let Some(instance) = runtime.instances.get(&key) {
                if let Err(err) = call_script_function1(&lua, instance, "on_update", dt) {
                    push_script_runtime_error(&mut runtime, err);
                }
            }
        }

        let inactive = runtime
            .instances
            .keys()
            .copied()
            .filter(|key| !active_scripts.contains(key))
            .collect::<Vec<_>>();
        for key in inactive {
            if let Some(instance) = runtime.instances.remove(&key) {
                let _ = call_script_function0(&lua, &instance, "on_stop");
                let _ = lua.remove_registry_value(instance.env_key);
                despawn_script_owned(world, key.entity, key.script_index);
            }
        }

        let inactive_rust = runtime
            .rust_instances
            .keys()
            .copied()
            .filter(|key| !active_scripts.contains(key))
            .collect::<Vec<_>>();
        for key in inactive_rust {
            if let Some(instance) = runtime.rust_instances.remove(&key) {
                stop_rust_script_instance(world, key, instance);
            }
        }

        let inactive_visual = runtime
            .visual_instances
            .keys()
            .copied()
            .filter(|key| !active_scripts.contains(key))
            .collect::<Vec<_>>();
        for key in inactive_visual {
            if let Some(instance) = runtime.visual_instances.remove(&key) {
                stop_visual_script_instance(world, world_ptr, key, instance);
            }
        }

        if let Some(mut run_state) = world.get_resource_mut::<ScriptRunState>() {
            run_state.last_state = current_state;
            run_state.last_executing = is_executing;
        }
    });

    if is_executing {
        if let Some(input_manager) = input_manager {
            let input_manager = input_manager.read();
            if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
                input_state.sync_last_state(&input_manager);
            }
        }
    } else if lifecycle_changed {
        if let Some(mut cursor_control) = world.get_resource_mut::<EditorCursorControlState>() {
            cursor_control.script_policy = None;
        }
        if let Some(input_manager) = input_manager {
            let input_manager = input_manager.read();
            if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
                input_state.reset(Some(&input_manager));
            }
        }
    }
}

fn script_path_label(script: &ScriptEntry) -> String {
    script
        .path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|| "<unassigned>".to_string())
}

fn push_script_runtime_error(runtime: &mut ScriptRuntime, message: impl Into<String>) {
    let message = message.into();
    tracing::error!(target: "script", "{message}");
    runtime.errors.push(message);
}

fn script_needs_reload(
    instance: &ScriptInstance,
    runtime_path: &Path,
    asset: &ScriptAsset,
) -> bool {
    if instance.path != runtime_path {
        return true;
    }

    asset.modified > instance.modified
}

fn visual_script_needs_reload(
    instance: &VisualScriptInstance,
    runtime_path: &Path,
    asset: &ScriptAsset,
) -> bool {
    if instance.path != runtime_path {
        return true;
    }

    asset.modified > instance.modified
}

struct VisualScriptExecutionHost {
    context: RustScriptHostContext,
}

impl VisualScriptExecutionHost {
    fn new(world_ptr: usize, owner: Entity, script_index: usize) -> Self {
        Self {
            context: RustScriptHostContext {
                world_ptr,
                owner,
                script_index,
            },
        }
    }
}

impl VisualScriptHost for VisualScriptExecutionHost {
    fn invoke_api(
        &mut self,
        table: VisualScriptApiTable,
        function: &str,
        args: &[JsonValue],
    ) -> Result<JsonValue, String> {
        let args_json = serde_json::to_string(args).map_err(|err| err.to_string())?;
        invoke_json_with_context(&self.context, table.as_str(), function, &args_json)
    }

    fn log(&mut self, message: &str) {
        emit_visual_script_log(self.context.owner, self.context.script_index, message);
    }
}

fn load_visual_script_instance(
    runtime_path: &Path,
    asset: &ScriptAsset,
) -> Result<VisualScriptInstance, String> {
    if let Some(error) = asset.error.as_ref() {
        return Err(format!("{}: {}", runtime_path.to_string_lossy(), error));
    }

    let program =
        compile_visual_script_runtime_source(&asset.source, &runtime_path.to_string_lossy())?;
    Ok(VisualScriptInstance {
        path: runtime_path.to_path_buf(),
        program,
        state: VisualScriptRuntimeState::default(),
        modified: asset.modified,
    })
}

fn run_visual_script_event(
    world_ptr: usize,
    key: ScriptInstanceKey,
    instance: &mut VisualScriptInstance,
    event: VisualScriptEvent,
    dt: f32,
) -> Result<(), String> {
    let mut host = VisualScriptExecutionHost::new(world_ptr, key.entity, key.script_index);
    instance.program.execute_event(
        event,
        key.entity.to_bits(),
        dt,
        &mut instance.state,
        &mut host,
    )
}

fn stop_visual_script_instance(
    world: &mut World,
    world_ptr: usize,
    key: ScriptInstanceKey,
    mut instance: VisualScriptInstance,
) {
    let _ = run_visual_script_event(world_ptr, key, &mut instance, VisualScriptEvent::Stop, 0.0);
    despawn_script_owned(world, key.entity, key.script_index);
}

fn load_script_instance(
    lua: &Lua,
    world_ptr: usize,
    entity: Entity,
    script_index: usize,
    runtime_path: &Path,
    asset: &ScriptAsset,
) -> Result<ScriptInstance, String> {
    if let Some(error) = asset.error.as_ref() {
        return Err(format!("{}: {}", runtime_path.to_string_lossy(), error));
    }

    let env = lua.create_table().map_err(|err| err.to_string())?;
    let _ = env.set("entity_id", entity.to_bits());
    register_script_api(lua, &env, world_ptr, entity, script_index).map_err(|err| {
        format!(
            "Failed to register script API for {}: {}",
            runtime_path.to_string_lossy(),
            err
        )
    })?;
    apply_lua_globals_fallback(lua, &env).map_err(|err| {
        format!(
            "Failed to apply Lua globals for {}: {}",
            runtime_path.to_string_lossy(),
            err
        )
    })?;

    let env_key = lua
        .create_registry_value(env.clone())
        .map_err(|err| err.to_string())?;

    let chunk = lua
        .load(&asset.source)
        .set_name(runtime_path.to_string_lossy().to_string())
        .set_environment(env);
    if let Err(err) = chunk.exec() {
        let _ = lua.remove_registry_value(env_key);
        return Err(format!("{}: {}", runtime_path.to_string_lossy(), err));
    }

    Ok(ScriptInstance {
        path: runtime_path.to_path_buf(),
        env_key,
        modified: asset.modified,
    })
}

fn build_rust_script_library(manifest_path: &Path) -> RustBuildResult {
    let source_modified = rust_script_modified_time(manifest_path);
    let mut output = String::new();

    let target_library_path = match rust_target_library_path(manifest_path) {
        Ok(path) => path,
        Err(err) => {
            return RustBuildResult {
                manifest_path: manifest_path.to_path_buf(),
                source_modified,
                artifact_path: None,
                output,
                error: Some(err),
            };
        }
    };

    let Some(root) = manifest_path.parent() else {
        return RustBuildResult {
            manifest_path: manifest_path.to_path_buf(),
            source_modified,
            artifact_path: None,
            output,
            error: Some("Rust script manifest has no parent directory".to_string()),
        };
    };

    let build_output = Command::new("cargo")
        .arg("build")
        .arg("--manifest-path")
        .arg(manifest_path)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .output();

    match build_output {
        Ok(command_output) => {
            let stdout = String::from_utf8_lossy(&command_output.stdout);
            let stderr = String::from_utf8_lossy(&command_output.stderr);
            output = format!("{}\n{}", stdout, stderr);

            if !command_output.status.success() {
                return RustBuildResult {
                    manifest_path: manifest_path.to_path_buf(),
                    source_modified,
                    artifact_path: None,
                    output: output.clone(),
                    error: Some(format!(
                        "cargo build failed for {}",
                        manifest_path.to_string_lossy()
                    )),
                };
            }

            if !target_library_path.exists() {
                return RustBuildResult {
                    manifest_path: manifest_path.to_path_buf(),
                    source_modified,
                    artifact_path: None,
                    output: output.clone(),
                    error: Some(format!(
                        "Expected build artifact missing: {}",
                        target_library_path.to_string_lossy()
                    )),
                };
            }

            RustBuildResult {
                manifest_path: manifest_path.to_path_buf(),
                source_modified,
                artifact_path: Some(target_library_path),
                output,
                error: None,
            }
        }
        Err(err) => RustBuildResult {
            manifest_path: manifest_path.to_path_buf(),
            source_modified,
            artifact_path: None,
            output,
            error: Some(format!(
                "Failed to run cargo build for {}: {}",
                manifest_path.to_string_lossy(),
                err
            )),
        },
    }
}

fn rust_target_library_path(manifest_path: &Path) -> Result<PathBuf, String> {
    runtime_scripting::rust_target_library_path(manifest_path)
}

fn resolve_prebuilt_rust_plugin_path(
    world: &World,
    manifest_path: &Path,
) -> Result<PathBuf, String> {
    let Some(project) = world.get_resource::<EditorProject>() else {
        return Err("EditorProject resource is missing".to_string());
    };
    let Some(project_root) = project.root.as_ref() else {
        return Err("Project root is not set for Rust script execution".to_string());
    };
    let Some(project_config) = project.config.as_ref() else {
        return Err("Project config is not set for Rust script execution".to_string());
    };

    let assets_root = project_config.assets_root(project_root);
    let manifest_relative = manifest_path.strip_prefix(&assets_root).map_err(|_| {
        format!(
            "Rust script manifest is outside project assets root: {}",
            manifest_path.to_string_lossy()
        )
    })?;
    let Some(prebuilt_relative) =
        runtime_scripting::rust_prebuilt_plugin_relative_path(manifest_relative)
    else {
        return Err(format!(
            "Rust script manifest path is not valid for prebuilt plugin lookup: {}",
            manifest_relative.to_string_lossy()
        ));
    };

    let prebuilt_path = assets_root.join(prebuilt_relative);
    if !prebuilt_path.is_file() {
        return Err(format!(
            "Prebuilt Rust script plugin not found: {}. Rebuild the project with helmer_build.",
            prebuilt_path.to_string_lossy()
        ));
    }

    Ok(prebuilt_path)
}

fn load_rust_plugin(
    manifest_path: &Path,
    source_library_path: &Path,
    source_modified: SystemTime,
    version_id: u64,
) -> Result<RustLoadedPlugin, String> {
    let temp_root = std::env::temp_dir().join("helmer_editor_rust_plugins");
    fs::create_dir_all(&temp_root).map_err(|err| err.to_string())?;

    let artifact_name = source_library_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("script");
    let sanitized = sanitize_file_name(&manifest_path.to_string_lossy());
    let load_path = temp_root.join(format!("{}_v{}_{}", sanitized, version_id, artifact_name));
    fs::copy(source_library_path, &load_path).map_err(|err| {
        format!(
            "Failed to copy Rust plugin artifact {} -> {}: {}",
            source_library_path.to_string_lossy(),
            load_path.to_string_lossy(),
            err
        )
    })?;

    let library = unsafe { Library::new(&load_path) }
        .map_err(|err| format!("Library load failed: {}", err))?;
    let get_plugin: Symbol<unsafe extern "C" fn() -> *const RustScriptPluginApi> = unsafe {
        library
            .get(RUST_SCRIPT_PLUGIN_SYMBOL)
            .map_err(|err| format!("Missing plugin symbol: {}", err))?
    };
    let plugin_ptr = unsafe { get_plugin() };
    if plugin_ptr.is_null() {
        return Err("Rust plugin returned null vtable pointer".to_string());
    }

    let plugin = unsafe { &*plugin_ptr };
    if plugin.abi_version != SCRIPT_PLUGIN_ABI_VERSION {
        return Err(format!(
            "Rust plugin ABI mismatch: expected {}, got {}",
            SCRIPT_PLUGIN_ABI_VERSION, plugin.abi_version
        ));
    }

    Ok(RustLoadedPlugin {
        manifest_path: manifest_path.to_path_buf(),
        source_library_path: source_library_path.to_path_buf(),
        loaded_library_path: load_path,
        version_id,
        source_modified,
        plugin_ptr,
        library: Arc::new(Mutex::new(library)),
    })
}

fn sanitize_file_name(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    if output.is_empty() {
        "script".to_string()
    } else {
        output
    }
}

fn truncate_diagnostic(value: &str) -> String {
    const MAX_LINES: usize = 16;
    const MAX_CHARS: usize = 2048;

    let mut out = String::new();
    for (index, line) in value.lines().enumerate() {
        if index >= MAX_LINES || out.len() >= MAX_CHARS {
            out.push_str("\n...");
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
    }
    out
}

fn rust_build_warning_lines(output: &str) -> Vec<String> {
    const MAX_WARNINGS: usize = 24;
    let mut warnings = Vec::new();
    let mut seen = HashSet::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed.to_ascii_lowercase().contains("warning:") {
            continue;
        }
        let entry = trimmed.to_string();
        if seen.insert(entry.clone()) {
            warnings.push(entry);
            if warnings.len() >= MAX_WARNINGS {
                break;
            }
        }
    }
    warnings
}

fn create_rust_script_instance(
    world_ptr: usize,
    owner: Entity,
    script_index: usize,
    manifest_path: &Path,
    plugin: Arc<RustLoadedPlugin>,
) -> Result<RustScriptInstance, String> {
    let mut host_context = Box::new(RustScriptHostContext {
        world_ptr,
        owner,
        script_index,
    });
    let host_api = Box::new(RustScriptApi {
        abi_version: SCRIPT_API_ABI_VERSION,
        user_data: (&mut *host_context as *mut RustScriptHostContext).cast::<c_void>(),
        log: rust_api_log,
        spawn_entity: rust_api_spawn_entity,
        entity_exists: rust_api_entity_exists,
        delete_entity: rust_api_delete_entity,
        get_transform: rust_api_get_transform,
        set_transform: rust_api_set_transform,
        invoke_json: rust_api_invoke_json,
        free_bytes: rust_api_free_bytes,
    });

    let plugin_api = rust_plugin_api(&plugin)?;
    let plugin_instance_ptr = unsafe { (plugin_api.create)(&*host_api, owner.to_bits()) };
    if plugin_instance_ptr.is_null() {
        return Err(format!(
            "Rust script plugin failed to create instance for {}",
            manifest_path.to_string_lossy()
        ));
    }

    Ok(RustScriptInstance {
        manifest_path: manifest_path.to_path_buf(),
        plugin_version_id: plugin.version_id,
        plugin_instance_ptr,
        plugin,
        host_context,
        host_api,
    })
}

fn rust_plugin_api(plugin: &RustLoadedPlugin) -> Result<&RustScriptPluginApi, String> {
    if plugin.plugin_ptr.is_null() {
        return Err("Rust script plugin table pointer is null".to_string());
    }
    Ok(unsafe { &*plugin.plugin_ptr })
}

fn call_rust_script_start(instance: &mut RustScriptInstance) -> Result<(), String> {
    if instance.plugin_instance_ptr.is_null() {
        return Err("Rust script instance pointer is null".to_string());
    }
    let plugin_api = rust_plugin_api(&instance.plugin)?;
    unsafe { (plugin_api.on_start)(instance.plugin_instance_ptr) };
    Ok(())
}

fn call_rust_script_update(instance: &mut RustScriptInstance, dt: f32) -> Result<(), String> {
    if instance.plugin_instance_ptr.is_null() {
        return Err("Rust script instance pointer is null".to_string());
    }
    let plugin_api = rust_plugin_api(&instance.plugin)?;
    unsafe { (plugin_api.on_update)(instance.plugin_instance_ptr, dt) };
    Ok(())
}

fn call_rust_script_stop(instance: &mut RustScriptInstance) -> Result<(), String> {
    if instance.plugin_instance_ptr.is_null() {
        return Ok(());
    }
    let plugin_api = rust_plugin_api(&instance.plugin)?;
    unsafe { (plugin_api.on_stop)(instance.plugin_instance_ptr) };
    Ok(())
}

fn take_rust_script_state(instance: &mut RustScriptInstance) -> Result<Option<Vec<u8>>, String> {
    if instance.plugin_instance_ptr.is_null() {
        return Ok(None);
    }
    let plugin_api = rust_plugin_api(&instance.plugin)?;
    let mut state_bytes = RustScriptBytes::default();
    let saved = unsafe { (plugin_api.save_state)(instance.plugin_instance_ptr, &mut state_bytes) };
    if saved == 0 {
        return Ok(None);
    }
    if state_bytes.ptr.is_null() && state_bytes.len > 0 {
        return Err("Rust script save_state returned invalid state buffer".to_string());
    }

    let payload = if state_bytes.ptr.is_null() || state_bytes.len == 0 {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(state_bytes.ptr as *const u8, state_bytes.len).to_vec()
        }
    };

    unsafe { (plugin_api.free_state)(state_bytes) };
    Ok(Some(payload))
}

fn call_rust_script_load_state(
    instance: &mut RustScriptInstance,
    state: &[u8],
) -> Result<bool, String> {
    if instance.plugin_instance_ptr.is_null() {
        return Err("Rust script instance pointer is null".to_string());
    }
    let plugin_api = rust_plugin_api(&instance.plugin)?;
    let state_view = RustScriptBytesView {
        ptr: state.as_ptr(),
        len: state.len(),
    };
    let loaded = unsafe { (plugin_api.load_state)(instance.plugin_instance_ptr, state_view) };
    Ok(loaded != 0)
}

fn stop_rust_script_instance(
    world: &mut World,
    key: ScriptInstanceKey,
    mut instance: RustScriptInstance,
) {
    instance.host_context.world_ptr = world as *mut World as usize;
    let _ = call_rust_script_stop(&mut instance);
    if !instance.plugin_instance_ptr.is_null() {
        if let Ok(plugin_api) = rust_plugin_api(&instance.plugin) {
            unsafe { (plugin_api.destroy)(instance.plugin_instance_ptr) };
        }
        instance.plugin_instance_ptr = std::ptr::null_mut();
    }
    despawn_script_owned(world, key.entity, key.script_index);
}

unsafe fn rust_context_from_user_data<'a>(
    user_data: *mut c_void,
) -> Option<&'a mut RustScriptHostContext> {
    if user_data.is_null() {
        return None;
    }
    Some(unsafe { &mut *(user_data as *mut RustScriptHostContext) })
}

unsafe fn rust_world_from_context<'a>(context: &RustScriptHostContext) -> Option<&'a mut World> {
    if context.world_ptr == 0 {
        return None;
    }
    Some(unsafe { &mut *(context.world_ptr as *mut World) })
}

#[derive(Clone, Copy)]
enum ScriptLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

fn script_log_level_from_message(message: &str) -> ScriptLogLevel {
    let lower = message.trim_start().to_ascii_lowercase();
    if lower.starts_with("error:")
        || lower.starts_with("err:")
        || lower.starts_with("[error]")
        || lower.starts_with("[err]")
    {
        ScriptLogLevel::Error
    } else if lower.starts_with("warning:")
        || lower.starts_with("warn:")
        || lower.starts_with("[warning]")
        || lower.starts_with("[warn]")
    {
        ScriptLogLevel::Warn
    } else if lower.starts_with("debug:") || lower.starts_with("[debug]") {
        ScriptLogLevel::Debug
    } else if lower.starts_with("trace:") || lower.starts_with("[trace]") {
        ScriptLogLevel::Trace
    } else {
        ScriptLogLevel::Info
    }
}

fn emit_rust_script_log(owner: Entity, script_index: usize, message: &str) {
    let formatted = format!(
        "[rust-script {}:{}] {}",
        owner.index(),
        script_index,
        message
    );
    match script_log_level_from_message(message) {
        ScriptLogLevel::Trace => tracing::trace!(target: "script.rust", "{formatted}"),
        ScriptLogLevel::Debug => tracing::debug!(target: "script.rust", "{formatted}"),
        ScriptLogLevel::Info => tracing::info!(target: "script.rust", "{formatted}"),
        ScriptLogLevel::Warn => tracing::warn!(target: "script.rust", "{formatted}"),
        ScriptLogLevel::Error => tracing::error!(target: "script.rust", "{formatted}"),
    }
}

fn emit_lua_script_log(message: &str) {
    match script_log_level_from_message(message) {
        ScriptLogLevel::Trace => tracing::trace!(target: "script.lua", "{message}"),
        ScriptLogLevel::Debug => tracing::debug!(target: "script.lua", "{message}"),
        ScriptLogLevel::Info => tracing::info!(target: "script.lua", "{message}"),
        ScriptLogLevel::Warn => tracing::warn!(target: "script.lua", "{message}"),
        ScriptLogLevel::Error => tracing::error!(target: "script.lua", "{message}"),
    }
}

fn emit_visual_script_log(owner: Entity, script_index: usize, message: &str) {
    let formatted = format!(
        "[visual-script {}:{}] {}",
        owner.index(),
        script_index,
        message
    );
    match script_log_level_from_message(message) {
        ScriptLogLevel::Trace => tracing::trace!(target: "script.visual", "{formatted}"),
        ScriptLogLevel::Debug => tracing::debug!(target: "script.visual", "{formatted}"),
        ScriptLogLevel::Info => tracing::info!(target: "script.visual", "{formatted}"),
        ScriptLogLevel::Warn => tracing::warn!(target: "script.visual", "{formatted}"),
        ScriptLogLevel::Error => tracing::error!(target: "script.visual", "{formatted}"),
    }
}

unsafe extern "C" fn rust_api_log(user_data: *mut c_void, message: *const c_char) {
    if message.is_null() {
        return;
    }
    let Some(context) = (unsafe { rust_context_from_user_data(user_data) }) else {
        return;
    };
    let text = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .to_string();
    emit_rust_script_log(context.owner, context.script_index, &text);
}

unsafe extern "C" fn rust_api_spawn_entity(
    user_data: *mut c_void,
    name: *const c_char,
) -> RustScriptEntityId {
    let Some(context) = (unsafe { rust_context_from_user_data(user_data) }) else {
        return 0;
    };
    let Some(world) = (unsafe { rust_world_from_context(context) }) else {
        return 0;
    };

    let mut entity = world.spawn((
        EditorEntity,
        BevyTransform::default(),
        ScriptSpawned {
            owner: context.owner,
            script_index: context.script_index,
        },
    ));
    if !name.is_null() {
        let value = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .trim()
            .to_string();
        if !value.is_empty() {
            entity.insert(Name::new(value));
        }
    }
    entity.id().to_bits()
}

unsafe extern "C" fn rust_api_entity_exists(
    user_data: *mut c_void,
    entity_id: RustScriptEntityId,
) -> u8 {
    let Some(context) = (unsafe { rust_context_from_user_data(user_data) }) else {
        return 0;
    };
    let Some(world) = (unsafe { rust_world_from_context(context) }) else {
        return 0;
    };
    if lookup_editor_entity(world, entity_id).is_some() {
        1
    } else {
        0
    }
}

unsafe extern "C" fn rust_api_delete_entity(
    user_data: *mut c_void,
    entity_id: RustScriptEntityId,
) -> u8 {
    let Some(context) = (unsafe { rust_context_from_user_data(user_data) }) else {
        return 0;
    };
    let Some(world) = (unsafe { rust_world_from_context(context) }) else {
        return 0;
    };
    let Some(entity) = lookup_editor_entity(world, entity_id) else {
        return 0;
    };
    if world.despawn(entity) { 1 } else { 0 }
}

unsafe extern "C" fn rust_api_get_transform(
    user_data: *mut c_void,
    entity_id: RustScriptEntityId,
    out_transform: *mut RustScriptTransform,
) -> u8 {
    if out_transform.is_null() {
        return 0;
    }
    let Some(context) = (unsafe { rust_context_from_user_data(user_data) }) else {
        return 0;
    };
    let Some(world) = (unsafe { rust_world_from_context(context) }) else {
        return 0;
    };
    let Some(entity) = lookup_editor_entity(world, entity_id) else {
        return 0;
    };
    let Some(transform) = world.get::<BevyTransform>(entity) else {
        return 0;
    };

    unsafe {
        *out_transform = RustScriptTransform {
            position: rust_vec3_from_glam(transform.0.position),
            rotation: rust_quat_from_glam(transform.0.rotation),
            scale: rust_vec3_from_glam(transform.0.scale),
        };
    }
    1
}

unsafe extern "C" fn rust_api_set_transform(
    user_data: *mut c_void,
    entity_id: RustScriptEntityId,
    patch: *const RustScriptTransformPatch,
) -> u8 {
    if patch.is_null() {
        return 0;
    }
    let Some(context) = (unsafe { rust_context_from_user_data(user_data) }) else {
        return 0;
    };
    let Some(world) = (unsafe { rust_world_from_context(context) }) else {
        return 0;
    };
    let Some(entity) = lookup_editor_entity(world, entity_id) else {
        return 0;
    };
    ensure_transform(world, entity);
    let Some(mut transform) = world.get_mut::<BevyTransform>(entity) else {
        return 0;
    };

    let patch = unsafe { &*patch };
    if patch.has_position != 0 {
        transform.0.position = glam_vec3_from_rust(patch.position);
    }
    if patch.has_rotation != 0 {
        transform.0.rotation = glam_quat_from_rust(patch.rotation);
    }
    if patch.has_scale != 0 {
        transform.0.scale = glam_vec3_from_rust(patch.scale);
    }
    1
}

unsafe extern "C" fn rust_api_invoke_json(
    user_data: *mut c_void,
    table_name: *const c_char,
    function_name: *const c_char,
    args_json: *const c_char,
    out_result: *mut RustScriptBytes,
) -> u8 {
    if out_result.is_null() {
        return 0;
    }

    let response =
        match unsafe { rust_invoke_json(user_data, table_name, function_name, args_json) } {
            Ok(result) => serde_json::json!({
                "ok": true,
                "result": result
            }),
            Err(error) => serde_json::json!({
                "ok": false,
                "error": error
            }),
        };

    let payload = serde_json::to_vec(&response)
        .unwrap_or_else(|_| b"{\"ok\":false,\"error\":\"response serialization failed\"}".to_vec());
    unsafe {
        *out_result = rust_script_bytes_from_vec(payload);
    }
    1
}

unsafe extern "C" fn rust_api_free_bytes(_user_data: *mut c_void, value: RustScriptBytes) {
    if value.ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(value.ptr, value.len, value.cap);
    }
}

fn rust_script_bytes_from_vec(mut value: Vec<u8>) -> RustScriptBytes {
    let out = RustScriptBytes {
        ptr: value.as_mut_ptr(),
        len: value.len(),
        cap: value.capacity(),
    };
    std::mem::forget(value);
    out
}

unsafe fn rust_invoke_json(
    user_data: *mut c_void,
    table_name: *const c_char,
    function_name: *const c_char,
    args_json: *const c_char,
) -> Result<JsonValue, String> {
    if table_name.is_null() || function_name.is_null() {
        return Err("Missing table/function for JSON bridge call".to_string());
    }

    let Some(context) = (unsafe { rust_context_from_user_data(user_data) }) else {
        return Err("Missing script host context".to_string());
    };

    let table_name = unsafe { CStr::from_ptr(table_name) }
        .to_string_lossy()
        .to_string();
    let function_name = unsafe { CStr::from_ptr(function_name) }
        .to_string_lossy()
        .to_string();
    let args_json = if args_json.is_null() {
        "[]".to_string()
    } else {
        unsafe { CStr::from_ptr(args_json) }
            .to_string_lossy()
            .to_string()
    };

    invoke_json_with_context(context, &table_name, &function_name, &args_json)
}

fn invoke_json_with_context(
    context: &RustScriptHostContext,
    table_name: &str,
    function_name: &str,
    args_json: &str,
) -> Result<JsonValue, String> {
    let table_name = table_name.trim().to_ascii_lowercase();
    let function_name = function_name.trim();
    if function_name.is_empty() {
        return Err("Function name cannot be empty".to_string());
    }

    let lua = Lua::new();
    let api_table = match table_name.as_str() {
        "ecs" => build_ecs_table(&lua, context.world_ptr, context.owner, context.script_index)
            .map_err(|err| err.to_string())?,
        "input" => build_input_table(&lua, context.world_ptr).map_err(|err| err.to_string())?,
        _ => {
            return Err(format!(
                "Unknown script API table '{}', expected 'ecs' or 'input'",
                table_name
            ));
        }
    };

    let args = rust_json_args_to_lua_multivalue(&lua, args_json)?;
    let value: Value = api_table
        .get(function_name)
        .map_err(|err| err.to_string())?;

    match value {
        Value::Function(function) => {
            let result: MultiValue = function.call(args).map_err(|err| err.to_string())?;
            rust_lua_multivalue_to_json(result)
        }
        value => {
            if !args.is_empty() {
                return Err(format!(
                    "Table field '{}.{}' is not callable",
                    table_name, function_name
                ));
            }
            rust_lua_value_to_json(value)
        }
    }
}

fn rust_json_args_to_lua_multivalue(lua: &Lua, args_json: &str) -> Result<MultiValue, String> {
    let parsed = if args_json.trim().is_empty() {
        JsonValue::Array(Vec::new())
    } else {
        serde_json::from_str::<JsonValue>(args_json).map_err(|err| err.to_string())?
    };

    let mut out = MultiValue::new();
    match parsed {
        JsonValue::Array(values) => {
            for value in values {
                out.push_back(rust_json_to_lua_value(lua, value)?);
            }
        }
        JsonValue::Null => {}
        value => out.push_back(rust_json_to_lua_value(lua, value)?),
    }
    Ok(out)
}

fn rust_json_to_lua_value(lua: &Lua, value: JsonValue) -> Result<Value, String> {
    match value {
        JsonValue::Null => Ok(Value::Nil),
        JsonValue::Bool(value) => Ok(Value::Boolean(value)),
        JsonValue::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(Value::Integer(value))
            } else if let Some(value) = value.as_f64() {
                Ok(Value::Number(value))
            } else {
                Ok(Value::Nil)
            }
        }
        JsonValue::String(value) => lua
            .create_string(&value)
            .map(Value::String)
            .map_err(|err| err.to_string()),
        JsonValue::Array(values) => {
            let table = lua.create_table().map_err(|err| err.to_string())?;
            for (index, value) in values.into_iter().enumerate() {
                table
                    .set(index + 1, rust_json_to_lua_value(lua, value)?)
                    .map_err(|err| err.to_string())?;
            }
            Ok(Value::Table(table))
        }
        JsonValue::Object(values) => {
            let table = lua.create_table().map_err(|err| err.to_string())?;
            for (key, value) in values {
                table
                    .set(key, rust_json_to_lua_value(lua, value)?)
                    .map_err(|err| err.to_string())?;
            }
            Ok(Value::Table(table))
        }
    }
}

fn rust_lua_multivalue_to_json(values: MultiValue) -> Result<JsonValue, String> {
    let values = values.into_vec();
    if values.is_empty() {
        return Ok(JsonValue::Null);
    }
    if values.len() == 1 {
        return rust_lua_value_to_json(values.into_iter().next().expect("len checked above"));
    }

    let mut result = Vec::with_capacity(values.len());
    for value in values {
        result.push(rust_lua_value_to_json(value)?);
    }
    Ok(JsonValue::Array(result))
}

fn rust_lua_value_to_json(value: Value) -> Result<JsonValue, String> {
    match value {
        Value::Nil => Ok(JsonValue::Null),
        Value::Boolean(value) => Ok(JsonValue::Bool(value)),
        Value::Integer(value) => Ok(JsonValue::Number(JsonNumber::from(value))),
        Value::Number(value) => Ok(JsonNumber::from_f64(value)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null)),
        Value::String(value) => Ok(JsonValue::String(value.to_string_lossy().to_string())),
        Value::UserData(userdata) => {
            if let Ok(key) = userdata.borrow::<LuaKey>() {
                return Ok(JsonValue::String(lua_key_to_handle_name(*key)));
            }
            if let Ok(button) = userdata.borrow::<LuaMouseButton>() {
                return Ok(JsonValue::String(lua_mouse_button_to_handle_name(button.0)));
            }
            if let Ok(button) = userdata.borrow::<LuaGamepadButton>() {
                return Ok(JsonValue::String(lua_gamepad_button_to_handle_name(
                    button.0,
                )));
            }
            if let Ok(axis) = userdata.borrow::<LuaGamepadAxis>() {
                return Ok(JsonValue::String(lua_gamepad_axis_to_handle_name(axis.0)));
            }
            Ok(JsonValue::Null)
        }
        Value::Table(table) => rust_lua_table_to_json(table),
        Value::Error(error) => Ok(JsonValue::String(error.to_string())),
        _ => Ok(JsonValue::Null),
    }
}

fn lua_key_to_handle_name(key: LuaKey) -> String {
    match key {
        LuaKey::AnyShift => "Shift".to_string(),
        LuaKey::AnyCtrl => "Ctrl".to_string(),
        LuaKey::AnyAlt => "Alt".to_string(),
        LuaKey::AnySuper => "Super".to_string(),
        LuaKey::Code(code) => format!("{:?}", code),
    }
}

fn lua_mouse_button_to_handle_name(button: MouseButton) -> String {
    match button {
        MouseButton::Left => "Left".to_string(),
        MouseButton::Right => "Right".to_string(),
        MouseButton::Middle => "Middle".to_string(),
        MouseButton::Back => "Back".to_string(),
        MouseButton::Forward => "Forward".to_string(),
        MouseButton::Other(value) => format!("Button{}", value),
    }
}

fn lua_gamepad_button_to_handle_name(button: gilrs::Button) -> String {
    format!("{:?}", button)
}

fn lua_gamepad_axis_to_handle_name(axis: gilrs::Axis) -> String {
    format!("{:?}", axis)
}

fn rust_lua_table_to_json(table: Table) -> Result<JsonValue, String> {
    let mut entries = Vec::new();
    let mut array_like = true;
    let mut array_count = 0usize;
    let mut max_index = 0usize;

    for pair in table.pairs::<Value, Value>() {
        let (key, value) = pair.map_err(|err| err.to_string())?;
        match &key {
            Value::Integer(index) if *index > 0 => {
                array_count += 1;
                max_index = max_index.max(*index as usize);
            }
            _ => array_like = false,
        }
        entries.push((key, value));
    }

    if array_like && max_index == array_count {
        let mut output = vec![JsonValue::Null; max_index];
        for (key, value) in entries {
            if let Value::Integer(index) = key {
                let index = (index as usize).saturating_sub(1);
                if index < output.len() {
                    output[index] = rust_lua_value_to_json(value)?;
                }
            }
        }
        return Ok(JsonValue::Array(output));
    }

    let mut output = JsonMap::new();
    for (key, value) in entries {
        let key = match key {
            Value::String(value) => value.to_string_lossy().to_string(),
            Value::Integer(value) => value.to_string(),
            Value::Number(value) => value.to_string(),
            Value::Boolean(value) => value.to_string(),
            _ => continue,
        };
        output.insert(key, rust_lua_value_to_json(value)?);
    }
    Ok(JsonValue::Object(output))
}

fn rust_vec3_from_glam(value: Vec3) -> RustScriptVec3 {
    RustScriptVec3 {
        x: value.x,
        y: value.y,
        z: value.z,
    }
}

fn rust_quat_from_glam(value: Quat) -> RustScriptQuat {
    RustScriptQuat {
        x: value.x,
        y: value.y,
        z: value.z,
        w: value.w,
    }
}

fn glam_vec3_from_rust(value: RustScriptVec3) -> Vec3 {
    Vec3::new(value.x, value.y, value.z)
}

fn glam_quat_from_rust(value: RustScriptQuat) -> Quat {
    Quat::from_xyzw(value.x, value.y, value.z, value.w)
}

fn apply_lua_globals_fallback(lua: &Lua, env: &Table) -> mlua::Result<()> {
    let globals = lua.globals();
    let metatable = lua.create_table()?;
    metatable.set("__index", globals)?;
    env.set_metatable(Some(metatable))?;
    Ok(())
}

fn call_script_function0(lua: &Lua, instance: &ScriptInstance, name: &str) -> Result<(), String> {
    let env: Table = lua
        .registry_value(&instance.env_key)
        .map_err(|err| err.to_string())?;
    let func: Option<Function> = env.get(name).ok();
    let Some(func) = func else {
        return Ok(());
    };
    func.call::<()>(())
        .map_err(|err| format!("{}:{}: {}", instance.path.to_string_lossy(), name, err))
}

fn call_script_function1(
    lua: &Lua,
    instance: &ScriptInstance,
    name: &str,
    dt: f32,
) -> Result<(), String> {
    let env: Table = lua
        .registry_value(&instance.env_key)
        .map_err(|err| err.to_string())?;
    let func: Option<Function> = env.get(name).ok();
    let Some(func) = func else {
        return Ok(());
    };
    func.call::<()>((dt,))
        .map_err(|err| format!("{}:{}: {}", instance.path.to_string_lossy(), name, err))
}

fn register_script_api(
    lua: &Lua,
    env: &Table,
    world_ptr: usize,
    owner: Entity,
    script_index: usize,
) -> mlua::Result<()> {
    let print_fn = lua.create_function(|_, args: Variadic<Value>| {
        let mut parts = Vec::with_capacity(args.len());
        for value in args {
            parts.push(lua_value_to_string(value));
        }
        emit_lua_script_log(&parts.join(" "));
        Ok(())
    })?;
    env.set("print", print_fn)?;

    let ecs = build_ecs_table(lua, world_ptr, owner, script_index)?;
    env.set("ecs", ecs)?;

    let input = build_input_table(lua, world_ptr)?;
    env.set("input", input)?;
    Ok(())
}

fn build_ecs_table(
    lua: &Lua,
    world_ptr: usize,
    owner: Entity,
    script_index: usize,
) -> mlua::Result<Table> {
    let ecs = lua.create_table()?;
    let owner_id = owner.to_bits();

    let world_ptr_list = world_ptr;
    ecs.set(
        "list_entities",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_list as *mut World) };
            let mut query = world.query_filtered::<Entity, With<EditorEntity>>();
            let table = lua.create_table()?;
            for (index, entity) in query.iter(world).enumerate() {
                table.set(index + 1, entity.to_bits())?;
            }
            Ok(table)
        })?,
    )?;

    let world_ptr_spawn = world_ptr;
    ecs.set(
        "spawn_entity",
        lua.create_function(move |_, name: Option<String>| {
            let world = unsafe { &mut *(world_ptr_spawn as *mut World) };
            let owner = Entity::from_bits(owner_id);
            let mut entity = world.spawn((
                EditorEntity,
                BevyTransform::default(),
                ScriptSpawned {
                    owner,
                    script_index,
                },
            ));
            if let Some(name) = name {
                if !name.is_empty() {
                    entity.insert(Name::new(name));
                }
            }
            Ok(entity.id().to_bits())
        })?,
    )?;

    let world_ptr_exists = world_ptr;
    ecs.set(
        "entity_exists",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_exists as *mut World) };
            Ok(lookup_editor_entity(world, entity_id).is_some())
        })?,
    )?;

    let world_ptr_find = world_ptr;
    ecs.set(
        "find_entity_by_name",
        lua.create_function(move |_, name: String| {
            let world = unsafe { &mut *(world_ptr_find as *mut World) };
            let mut query = world.query::<(Entity, &Name)>();
            for (entity, entity_name) in query.iter(world) {
                if world.get::<EditorEntity>(entity).is_some() && entity_name.as_str() == name {
                    return Ok(Some(entity.to_bits()));
                }
            }
            Ok(None)
        })?,
    )?;

    let world_ptr_get_name = world_ptr;
    ecs.set(
        "get_entity_name",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_name as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            Ok(world
                .get::<Name>(entity)
                .map(|name| name.as_str().to_string()))
        })?,
    )?;

    let world_ptr_set_name = world_ptr;
    ecs.set(
        "set_entity_name",
        lua.create_function(move |_, (entity_id, name): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_set_name as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            if name.is_empty() {
                world.entity_mut(entity).remove::<Name>();
            } else {
                world.entity_mut(entity).insert(Name::new(name));
            }
            Ok(true)
        })?,
    )?;

    let world_ptr_delete = world_ptr;
    ecs.set(
        "delete_entity",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_delete as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            world.despawn(entity);
            Ok(true)
        })?,
    )?;

    let world_ptr_has = world_ptr;
    ecs.set(
        "has_component",
        lua.create_function(move |_, (entity_id, component): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_has as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let component = component.to_ascii_lowercase();
            let has = match component.as_str() {
                "name" => world.get::<Name>(entity).is_some(),
                "transform" => world.get::<BevyTransform>(entity).is_some(),
                "camera" => world.get::<BevyCamera>(entity).is_some(),
                "light" => world.get::<BevyLight>(entity).is_some(),
                "mesh" | "mesh_renderer" => world.get::<BevyMeshRenderer>(entity).is_some(),
                "spline" => world.get::<BevySpline>(entity).is_some(),
                "spline_follower" => world.get::<BevySplineFollower>(entity).is_some(),
                "look_at" => world.get::<BevyLookAt>(entity).is_some(),
                "entity_follower" => world.get::<BevyEntityFollower>(entity).is_some(),
                "animator" => world.get::<BevyAnimator>(entity).is_some(),
                "scene" => world.get::<SceneRoot>(entity).is_some(),
                "audio" => {
                    world.get::<BevyAudioEmitter>(entity).is_some()
                        || world.get::<BevyAudioListener>(entity).is_some()
                }
                "audio_emitter" => world.get::<BevyAudioEmitter>(entity).is_some(),
                "audio_listener" => world.get::<BevyAudioListener>(entity).is_some(),
                "script" => world
                    .get::<ScriptComponent>(entity)
                    .map(|scripts| !scripts.scripts.is_empty())
                    .unwrap_or(false),
                "dynamic" => world.get::<DynamicComponents>(entity).is_some(),
                "physics" => has_physics_component(world, entity),
                "collider_shape" => world.get::<ColliderShape>(entity).is_some(),
                "dynamic_rigid_body" => world.get::<DynamicRigidBody>(entity).is_some(),
                "kinematic_rigid_body" => world.get::<KinematicRigidBody>(entity).is_some(),
                "fixed_collider" => world.get::<FixedCollider>(entity).is_some(),
                "collider_properties" => world.get::<ColliderProperties>(entity).is_some(),
                "collider_inheritance" => {
                    world.get::<ColliderPropertyInheritance>(entity).is_some()
                }
                "rigid_body_properties" => world.get::<RigidBodyProperties>(entity).is_some(),
                "rigid_body_inheritance" => {
                    world.get::<RigidBodyPropertyInheritance>(entity).is_some()
                }
                "physics_joint" => world.get::<PhysicsJoint>(entity).is_some(),
                "character_controller" => world.get::<CharacterController>(entity).is_some(),
                "character_controller_input" => {
                    world.get::<CharacterControllerInput>(entity).is_some()
                }
                "character_controller_output" => {
                    world.get::<CharacterControllerOutput>(entity).is_some()
                }
                "physics_ray_cast" => world.get::<PhysicsRayCast>(entity).is_some(),
                "physics_ray_cast_hit" => world.get::<PhysicsRayCastHit>(entity).is_some(),
                "physics_point_projection" => world.get::<PhysicsPointProjection>(entity).is_some(),
                "physics_point_projection_hit" => {
                    world.get::<PhysicsPointProjectionHit>(entity).is_some()
                }
                "physics_shape_cast" => world.get::<PhysicsShapeCast>(entity).is_some(),
                "physics_shape_cast_hit" => world.get::<PhysicsShapeCastHit>(entity).is_some(),
                "physics_world_defaults" => world.get::<PhysicsWorldDefaults>(entity).is_some(),
                _ => false,
            };
            Ok(has)
        })?,
    )?;

    let world_ptr_add = world_ptr;
    ecs.set(
        "add_component",
        lua.create_function(move |_, (entity_id, component): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_add as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let component = component.to_ascii_lowercase();
            match component.as_str() {
                "transform" => {
                    ensure_transform(world, entity);
                }
                "camera" => {
                    ensure_transform(world, entity);
                    world.entity_mut(entity).insert(BevyCamera::default());
                }
                "light" => {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevyWrapper(Light::point(Vec3::ONE, 10.0)));
                }
                "spline" => {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevySpline(Spline::default()));
                }
                "spline_follower" => {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevySplineFollower(SplineFollower::default()));
                }
                "look_at" => {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevyLookAt(LookAt::default()));
                }
                "entity_follower" => {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevyEntityFollower(EntityFollower::default()));
                }
                "audio_emitter" => {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevyWrapper(AudioEmitter::default()));
                    world.entity_mut(entity).insert(EditorAudio {
                        path: None,
                        streaming: false,
                    });
                }
                "audio_listener" => {
                    ensure_transform(world, entity);
                    world
                        .entity_mut(entity)
                        .insert(BevyWrapper(AudioListener::default()));
                }
                "dynamic" => {
                    world
                        .entity_mut(entity)
                        .insert(DynamicComponents::default());
                }
                "physics" => {
                    ensure_transform(world, entity);
                    world.entity_mut(entity).insert(ColliderShape::default());
                    set_physics_body_kind(world, entity, PhysicsBodyKind::Fixed);
                }
                "collider_shape" => {
                    ensure_transform(world, entity);
                    world.entity_mut(entity).insert(ColliderShape::default());
                }
                "dynamic_rigid_body" => {
                    ensure_transform(world, entity);
                    set_physics_body_kind(world, entity, PhysicsBodyKind::Dynamic { mass: 1.0 });
                }
                "kinematic_rigid_body" => {
                    ensure_transform(world, entity);
                    set_physics_body_kind(
                        world,
                        entity,
                        PhysicsBodyKind::Kinematic {
                            mode: KinematicMode::default(),
                        },
                    );
                }
                "fixed_collider" => {
                    ensure_transform(world, entity);
                    set_physics_body_kind(world, entity, PhysicsBodyKind::Fixed);
                }
                "collider_properties" => {
                    world
                        .entity_mut(entity)
                        .insert(ColliderProperties::default());
                }
                "collider_inheritance" => {
                    world
                        .entity_mut(entity)
                        .insert(ColliderPropertyInheritance::default());
                }
                "rigid_body_properties" => {
                    world
                        .entity_mut(entity)
                        .insert(RigidBodyProperties::default());
                }
                "rigid_body_inheritance" => {
                    world
                        .entity_mut(entity)
                        .insert(RigidBodyPropertyInheritance::default());
                }
                "physics_joint" => {
                    world.entity_mut(entity).insert(PhysicsJoint::default());
                }
                "character_controller" => {
                    world
                        .entity_mut(entity)
                        .insert(CharacterController::default());
                    world
                        .entity_mut(entity)
                        .insert(CharacterControllerInput::default());
                    world
                        .entity_mut(entity)
                        .insert(CharacterControllerOutput::default());
                }
                "character_controller_input" => {
                    world
                        .entity_mut(entity)
                        .insert(CharacterControllerInput::default());
                }
                "character_controller_output" => {
                    world
                        .entity_mut(entity)
                        .insert(CharacterControllerOutput::default());
                }
                "physics_ray_cast" => {
                    world.entity_mut(entity).insert(PhysicsRayCast::default());
                    world
                        .entity_mut(entity)
                        .insert(PhysicsRayCastHit::default());
                }
                "physics_ray_cast_hit" => {
                    world
                        .entity_mut(entity)
                        .insert(PhysicsRayCastHit::default());
                }
                "physics_point_projection" => {
                    world
                        .entity_mut(entity)
                        .insert(PhysicsPointProjection::default());
                    world
                        .entity_mut(entity)
                        .insert(PhysicsPointProjectionHit::default());
                }
                "physics_point_projection_hit" => {
                    world
                        .entity_mut(entity)
                        .insert(PhysicsPointProjectionHit::default());
                }
                "physics_shape_cast" => {
                    world.entity_mut(entity).insert(PhysicsShapeCast::default());
                    world
                        .entity_mut(entity)
                        .insert(PhysicsShapeCastHit::default());
                }
                "physics_shape_cast_hit" => {
                    world
                        .entity_mut(entity)
                        .insert(PhysicsShapeCastHit::default());
                }
                "physics_world_defaults" => {
                    world
                        .entity_mut(entity)
                        .insert(PhysicsWorldDefaults::default());
                }
                _ => return Ok(false),
            }
            Ok(true)
        })?,
    )?;

    let world_ptr_remove = world_ptr;
    ecs.set(
        "remove_component",
        lua.create_function(move |_, (entity_id, component): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_remove as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let component = component.to_ascii_lowercase();
            match component.as_str() {
                "name" => {
                    world.entity_mut(entity).remove::<Name>();
                }
                "transform" => {
                    world.entity_mut(entity).remove::<BevyTransform>();
                }
                "camera" => {
                    world.entity_mut(entity).remove::<BevyCamera>();
                    world.entity_mut(entity).remove::<EditorPlayCamera>();
                    world.entity_mut(entity).remove::<BevyActiveCamera>();
                }
                "light" => {
                    world.entity_mut(entity).remove::<BevyLight>();
                }
                "mesh" | "mesh_renderer" => {
                    world.entity_mut(entity).remove::<BevyMeshRenderer>();
                    world.entity_mut(entity).remove::<EditorMesh>();
                }
                "spline" => {
                    world.entity_mut(entity).remove::<BevySpline>();
                }
                "spline_follower" => {
                    world.entity_mut(entity).remove::<BevySplineFollower>();
                }
                "look_at" => {
                    world.entity_mut(entity).remove::<BevyLookAt>();
                }
                "entity_follower" => {
                    world.entity_mut(entity).remove::<BevyEntityFollower>();
                }
                "animator" => {
                    world.entity_mut(entity).remove::<BevyAnimator>();
                }
                "scene" => {
                    reset_scene_root_instance(world, entity);
                    world.entity_mut(entity).remove::<SceneRoot>();
                    world.entity_mut(entity).remove::<SceneAssetPath>();
                }
                "audio" => {
                    world.entity_mut(entity).remove::<BevyAudioEmitter>();
                    world.entity_mut(entity).remove::<EditorAudio>();
                    world.entity_mut(entity).remove::<BevyAudioListener>();
                }
                "audio_emitter" => {
                    world.entity_mut(entity).remove::<BevyAudioEmitter>();
                    world.entity_mut(entity).remove::<EditorAudio>();
                }
                "audio_listener" => {
                    world.entity_mut(entity).remove::<BevyAudioListener>();
                }
                "script" => {
                    world.entity_mut(entity).remove::<ScriptComponent>();
                }
                "dynamic" => {
                    world.entity_mut(entity).remove::<DynamicComponents>();
                }
                "physics" => {
                    remove_all_physics_components(world, entity);
                }
                "collider_shape" => {
                    world.entity_mut(entity).remove::<ColliderShape>();
                    world.entity_mut(entity).remove::<PhysicsHandle>();
                }
                "dynamic_rigid_body" => {
                    world.entity_mut(entity).remove::<DynamicRigidBody>();
                    world.entity_mut(entity).remove::<PhysicsHandle>();
                }
                "kinematic_rigid_body" => {
                    world.entity_mut(entity).remove::<KinematicRigidBody>();
                    world.entity_mut(entity).remove::<PhysicsHandle>();
                }
                "fixed_collider" => {
                    world.entity_mut(entity).remove::<FixedCollider>();
                    world.entity_mut(entity).remove::<PhysicsHandle>();
                }
                "collider_properties" => {
                    world.entity_mut(entity).remove::<ColliderProperties>();
                }
                "collider_inheritance" => {
                    world
                        .entity_mut(entity)
                        .remove::<ColliderPropertyInheritance>();
                }
                "rigid_body_properties" => {
                    world.entity_mut(entity).remove::<RigidBodyProperties>();
                }
                "rigid_body_inheritance" => {
                    world
                        .entity_mut(entity)
                        .remove::<RigidBodyPropertyInheritance>();
                }
                "physics_joint" => {
                    world.entity_mut(entity).remove::<PhysicsJoint>();
                }
                "character_controller" => {
                    world.entity_mut(entity).remove::<CharacterController>();
                    world
                        .entity_mut(entity)
                        .remove::<CharacterControllerInput>();
                    world
                        .entity_mut(entity)
                        .remove::<CharacterControllerOutput>();
                }
                "character_controller_input" => {
                    world
                        .entity_mut(entity)
                        .remove::<CharacterControllerInput>();
                }
                "character_controller_output" => {
                    world
                        .entity_mut(entity)
                        .remove::<CharacterControllerOutput>();
                }
                "physics_ray_cast" => {
                    world.entity_mut(entity).remove::<PhysicsRayCast>();
                    world.entity_mut(entity).remove::<PhysicsRayCastHit>();
                }
                "physics_ray_cast_hit" => {
                    world.entity_mut(entity).remove::<PhysicsRayCastHit>();
                }
                "physics_point_projection" => {
                    world.entity_mut(entity).remove::<PhysicsPointProjection>();
                    world
                        .entity_mut(entity)
                        .remove::<PhysicsPointProjectionHit>();
                }
                "physics_point_projection_hit" => {
                    world
                        .entity_mut(entity)
                        .remove::<PhysicsPointProjectionHit>();
                }
                "physics_shape_cast" => {
                    world.entity_mut(entity).remove::<PhysicsShapeCast>();
                    world.entity_mut(entity).remove::<PhysicsShapeCastHit>();
                }
                "physics_shape_cast_hit" => {
                    world.entity_mut(entity).remove::<PhysicsShapeCastHit>();
                }
                "physics_world_defaults" => {
                    world.entity_mut(entity).remove::<PhysicsWorldDefaults>();
                }
                _ => return Ok(false),
            }
            Ok(true)
        })?,
    )?;

    let world_ptr_get_transform = world_ptr;
    ecs.set(
        "get_transform",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_transform as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(transform) = world.get::<BevyTransform>(entity).map(|t| t.0) else {
                return Ok(None);
            };
            let table = lua.create_table()?;
            table.set("position", vec3_to_table(lua, transform.position)?)?;
            table.set("rotation", quat_to_table(lua, transform.rotation)?)?;
            table.set("scale", vec3_to_table(lua, transform.scale)?)?;
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_set_transform = world_ptr;
    ecs.set(
        "set_transform",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_transform as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };

            let mut transform = world
                .get::<BevyTransform>(entity)
                .map(|t| t.0)
                .unwrap_or_default();

            if let Ok(table) = data.get::<Table>("position") {
                if let Some(vec) = table_to_vec3(&table) {
                    transform.position = vec;
                }
            }
            if let Ok(table) = data.get::<Table>("rotation") {
                if let Some(rot) = table_to_quat(&table) {
                    transform.rotation = rot;
                }
            }
            if let Ok(table) = data.get::<Table>("scale") {
                if let Some(vec) = table_to_vec3(&table) {
                    transform.scale = vec;
                }
            }

            world.entity_mut(entity).insert(BevyWrapper(transform));
            Ok(true)
        })?,
    )?;

    let world_ptr_get_spline = world_ptr;
    ecs.set(
        "get_spline",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_spline as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(spline) = world.get::<BevySpline>(entity) else {
                return Ok(None);
            };
            let table = lua.create_table()?;
            let points_table = lua.create_table()?;
            for (index, point) in spline.0.points.iter().enumerate() {
                points_table.set(index + 1, vec3_to_table(lua, *point)?)?;
            }
            table.set("points", points_table)?;
            table.set("closed", spline.0.closed)?;
            table.set("tension", spline.0.tension)?;
            table.set("mode", spline_mode_to_str(spline.0.mode))?;
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_set_spline = world_ptr;
    ecs.set(
        "set_spline",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_spline as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut spline) = world.get_mut::<BevySpline>(entity) else {
                return Ok(false);
            };

            if let Ok(points) = data.get::<Table>("points") {
                let mut parsed = Vec::new();
                let len = points.len().unwrap_or(0);
                for i in 1..=len {
                    if let Ok(point_table) = points.get::<Table>(i) {
                        if let Some(point) = table_to_vec3(&point_table) {
                            parsed.push(point);
                        }
                    }
                }
                spline.0.points = parsed;
            }
            if let Ok(closed) = data.get::<bool>("closed") {
                spline.0.closed = closed;
            }
            if let Ok(tension) = data.get::<f32>("tension") {
                spline.0.tension = tension;
            }
            if let Ok(mode) = data.get::<String>("mode") {
                if let Some(parsed) = spline_mode_from_str(&mode) {
                    spline.0.mode = parsed;
                }
            }
            Ok(true)
        })?,
    )?;

    let world_ptr_add_spline_point = world_ptr;
    ecs.set(
        "add_spline_point",
        lua.create_function(move |_, (entity_id, point): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_add_spline_point as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut spline) = world.get_mut::<BevySpline>(entity) else {
                return Ok(false);
            };
            let Some(vec) = table_to_vec3(&point) else {
                return Ok(false);
            };
            spline.0.points.push(vec);
            Ok(true)
        })?,
    )?;

    let world_ptr_set_spline_point = world_ptr;
    ecs.set(
        "set_spline_point",
        lua.create_function(move |_, (entity_id, index, point): (u64, usize, Table)| {
            let world = unsafe { &mut *(world_ptr_set_spline_point as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut spline) = world.get_mut::<BevySpline>(entity) else {
                return Ok(false);
            };
            let Some(vec) = table_to_vec3(&point) else {
                return Ok(false);
            };
            let idx = index.saturating_sub(1);
            if idx >= spline.0.points.len() {
                return Ok(false);
            }
            spline.0.points[idx] = vec;
            Ok(true)
        })?,
    )?;

    let world_ptr_remove_spline_point = world_ptr;
    ecs.set(
        "remove_spline_point",
        lua.create_function(move |_, (entity_id, index): (u64, usize)| {
            let world = unsafe { &mut *(world_ptr_remove_spline_point as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut spline) = world.get_mut::<BevySpline>(entity) else {
                return Ok(false);
            };
            let idx = index.saturating_sub(1);
            if idx >= spline.0.points.len() {
                return Ok(false);
            }
            spline.0.points.remove(idx);
            Ok(true)
        })?,
    )?;

    let world_ptr_sample_spline = world_ptr;
    ecs.set(
        "sample_spline",
        lua.create_function(move |lua, (entity_id, t): (u64, f32)| {
            let world = unsafe { &mut *(world_ptr_sample_spline as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(spline) = world.get::<BevySpline>(entity) else {
                return Ok(None);
            };
            let pos = spline.0.sample(t);
            Ok(Some(vec3_to_table(lua, pos)?))
        })?,
    )?;

    let world_ptr_spline_length = world_ptr;
    ecs.set(
        "spline_length",
        lua.create_function(move |_, (entity_id, samples): (u64, Option<u32>)| {
            let world = unsafe { &mut *(world_ptr_spline_length as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(spline) = world.get::<BevySpline>(entity) else {
                return Ok(None);
            };
            let samples = samples.unwrap_or(64).max(2) as usize;
            Ok(Some(spline.0.approx_length(samples)))
        })?,
    )?;

    let world_ptr_follow_spline = world_ptr;
    ecs.set(
        "follow_spline",
        lua.create_function(
            move |_,
                  (entity_id, spline_id, speed, looped): (
                u64,
                Option<u64>,
                Option<f32>,
                Option<bool>,
            )| {
                let world = unsafe { &mut *(world_ptr_follow_spline as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let follower = world
                    .get::<BevySplineFollower>(entity)
                    .cloned()
                    .map(|f| f.0)
                    .unwrap_or_else(SplineFollower::default);
                let mut follower = follower;
                follower.spline_entity = spline_id;
                if let Some(value) = speed {
                    follower.speed = value;
                }
                if let Some(value) = looped {
                    follower.looped = value;
                }
                world
                    .entity_mut(entity)
                    .insert(BevySplineFollower(follower));
                Ok(true)
            },
        )?,
    )?;

    let world_ptr_set_anim_enabled = world_ptr;
    ecs.set(
        "set_animator_enabled",
        lua.create_function(move |_, (entity_id, enabled): (u64, bool)| {
            let world = unsafe { &mut *(world_ptr_set_anim_enabled as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) else {
                return Ok(false);
            };
            animator.0.enabled = enabled;
            Ok(true)
        })?,
    )?;

    let world_ptr_set_anim_time = world_ptr;
    ecs.set(
        "set_animator_time_scale",
        lua.create_function(move |_, (entity_id, time_scale): (u64, f32)| {
            let world = unsafe { &mut *(world_ptr_set_anim_time as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) else {
                return Ok(false);
            };
            animator.0.time_scale = time_scale.max(0.0);
            Ok(true)
        })?,
    )?;

    let world_ptr_set_anim_float = world_ptr;
    ecs.set(
        "set_animator_param_float",
        lua.create_function(move |_, (entity_id, name, value): (u64, String, f32)| {
            let world = unsafe { &mut *(world_ptr_set_anim_float as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) else {
                return Ok(false);
            };
            animator.0.parameters.set_float(name, value);
            Ok(true)
        })?,
    )?;

    let world_ptr_set_anim_bool = world_ptr;
    ecs.set(
        "set_animator_param_bool",
        lua.create_function(move |_, (entity_id, name, value): (u64, String, bool)| {
            let world = unsafe { &mut *(world_ptr_set_anim_bool as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) else {
                return Ok(false);
            };
            animator.0.parameters.set_bool(name, value);
            Ok(true)
        })?,
    )?;

    let world_ptr_trigger_anim = world_ptr;
    ecs.set(
        "trigger_animator",
        lua.create_function(move |_, (entity_id, name): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_trigger_anim as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) else {
                return Ok(false);
            };
            animator.0.parameters.trigger(name);
            Ok(true)
        })?,
    )?;

    let world_ptr_anim_clips = world_ptr;
    ecs.set(
        "get_animator_clips",
        lua.create_function(move |lua, (entity_id, layer_index): (u64, Option<usize>)| {
            let world = unsafe { &mut *(world_ptr_anim_clips as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(animator) = world.get::<BevyAnimator>(entity) else {
                return Ok(None);
            };
            let layer_index = layer_index.unwrap_or(0);
            let Some(layer) = animator.0.layers.get(layer_index) else {
                return Ok(None);
            };
            let table = lua.create_table()?;
            for (index, clip) in layer.graph.library.clips.iter().enumerate() {
                table.set(index + 1, clip.name.clone())?;
            }
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_play_clip = world_ptr;
    ecs.set(
        "play_anim_clip",
        lua.create_function(
            move |_, (entity_id, name, layer_index): (u64, String, Option<usize>)| {
                let world = unsafe { &mut *(world_ptr_play_clip as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(mut animator) = world.get_mut::<BevyAnimator>(entity) else {
                    return Ok(false);
                };
                let layer_index = layer_index.unwrap_or(0);
                let Some(layer) = animator.0.layers.get_mut(layer_index) else {
                    return Ok(false);
                };
                let Some(clip_index) = layer.graph.library.clip_index(&name) else {
                    return Ok(false);
                };
                if let Some(state) = layer
                    .state_machine
                    .states
                    .get(layer.state_machine.current_state)
                {
                    if let Some(node) = layer.graph.nodes.get_mut(state.node) {
                        if let helmer::animation::AnimationNode::Clip(clip_node) = node {
                            clip_node.clip_index = clip_index;
                            layer.state_machine.state_time = 0.0;
                            return Ok(true);
                        }
                    }
                }
                Ok(false)
            },
        )?,
    )?;

    let world_ptr_get_light = world_ptr;
    ecs.set(
        "get_light",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_light as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(light) = world.get::<BevyLight>(entity).map(|l| l.0) else {
                return Ok(None);
            };
            let table = lua.create_table()?;
            table.set("type", light_type_name(light.light_type))?;
            table.set("color", vec3_to_table(lua, light.color)?)?;
            table.set("intensity", light.intensity)?;
            if let LightType::Spot { angle } = light.light_type {
                table.set("angle", angle)?;
            }
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_set_light = world_ptr;
    ecs.set(
        "set_light",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_light as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };

            let mut light = world
                .get::<BevyLight>(entity)
                .map(|l| l.0)
                .unwrap_or_else(|| Light::point(Vec3::ONE, 10.0));

            if let Ok(kind) = data.get::<String>("type") {
                light.light_type = parse_light_type(&kind, light.light_type);
            }

            if let Ok(table) = data.get::<Table>("color") {
                if let Some(vec) = table_to_vec3(&table) {
                    light.color = vec;
                }
            }

            if let Ok(intensity) = data.get::<f32>("intensity") {
                light.intensity = intensity;
            }

            if let LightType::Spot { ref mut angle } = light.light_type {
                if let Ok(value) = data.get::<f32>("angle") {
                    *angle = value;
                }
            }

            ensure_transform(world, entity);
            world.entity_mut(entity).insert(BevyWrapper(light));
            Ok(true)
        })?,
    )?;

    let world_ptr_get_camera = world_ptr;
    ecs.set(
        "get_camera",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_camera as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(camera) = world.get::<BevyCamera>(entity).map(|c| c.0) else {
                return Ok(None);
            };
            let table = lua.create_table()?;
            table.set("fov_y_rad", camera.fov_y_rad)?;
            table.set("aspect_ratio", camera.aspect_ratio)?;
            table.set("near_plane", camera.near_plane)?;
            table.set("far_plane", camera.far_plane)?;
            table.set("active", world.get::<EditorPlayCamera>(entity).is_some())?;
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_set_camera = world_ptr;
    ecs.set(
        "set_camera",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_camera as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };

            let mut camera = world
                .get::<BevyCamera>(entity)
                .map(|c| c.0)
                .unwrap_or_default();

            if let Ok(value) = data.get::<f32>("fov_y_rad") {
                camera.fov_y_rad = value;
            }
            if let Ok(value) = data.get::<f32>("aspect_ratio") {
                camera.aspect_ratio = value;
            }
            if let Ok(value) = data.get::<f32>("near_plane") {
                camera.near_plane = value;
            }
            if let Ok(value) = data.get::<f32>("far_plane") {
                camera.far_plane = value;
            }

            ensure_transform(world, entity);
            world.entity_mut(entity).insert(BevyWrapper(camera));

            if let Ok(active) = data.get::<bool>("active") {
                if active {
                    set_active_camera(world, entity);
                }
            }

            Ok(true)
        })?,
    )?;

    let world_ptr_set_active = world_ptr;
    ecs.set(
        "set_active_camera",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_set_active as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            if world.get::<BevyCamera>(entity).is_none() {
                return Ok(false);
            }
            set_active_camera(world, entity);
            Ok(true)
        })?,
    )?;

    let world_ptr_get_viewport_mode = world_ptr;
    ecs.set(
        "get_viewport_mode",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_get_viewport_mode as *mut World) };
            let mode = world
                .get_resource::<EditorViewportState>()
                .map(|state| state.play_mode_view.as_str().to_string())
                .unwrap_or_else(|| PlayViewportKind::default().as_str().to_string());
            Ok(mode)
        })?,
    )?;

    let world_ptr_set_viewport_mode = world_ptr;
    ecs.set(
        "set_viewport_mode",
        lua.create_function(move |_, mode: String| {
            let world = unsafe { &mut *(world_ptr_set_viewport_mode as *mut World) };
            let Some(mode) = PlayViewportKind::parse(&mode) else {
                return Ok(false);
            };
            if let Some(mut viewport_state) = world.get_resource_mut::<EditorViewportState>() {
                viewport_state.play_mode_view = mode;
                return Ok(true);
            }
            Ok(false)
        })?,
    )?;

    let world_ptr_get_viewport_preview = world_ptr;
    ecs.set(
        "get_viewport_preview_camera",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_get_viewport_preview as *mut World) };
            Ok(world
                .get_resource::<EditorViewportState>()
                .and_then(|state| state.pinned_camera)
                .map(|entity| entity.to_bits()))
        })?,
    )?;

    let world_ptr_set_viewport_preview = world_ptr;
    ecs.set(
        "set_viewport_preview_camera",
        lua.create_function(move |_, value: Value| {
            let world = unsafe { &mut *(world_ptr_set_viewport_preview as *mut World) };
            let pinned_camera = match value {
                Value::Nil => None,
                Value::Integer(raw_id) => {
                    if raw_id < 0 {
                        return Ok(false);
                    }
                    let Some(entity) = lookup_editor_entity(world, raw_id as u64) else {
                        return Ok(false);
                    };
                    if world.get::<BevyCamera>(entity).is_none() {
                        return Ok(false);
                    }
                    Some(entity)
                }
                Value::Number(raw_id) => {
                    if !raw_id.is_finite() || raw_id < 0.0 {
                        return Ok(false);
                    }
                    let raw_id = raw_id as u64;
                    let Some(entity) = lookup_editor_entity(world, raw_id) else {
                        return Ok(false);
                    };
                    if world.get::<BevyCamera>(entity).is_none() {
                        return Ok(false);
                    }
                    Some(entity)
                }
                _ => return Ok(false),
            };

            if let Some(mut viewport_state) = world.get_resource_mut::<EditorViewportState>() {
                viewport_state.pinned_camera = pinned_camera;
                return Ok(true);
            }
            Ok(false)
        })?,
    )?;

    let world_ptr_get_mesh = world_ptr;
    ecs.set(
        "get_mesh_renderer",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_mesh as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(mesh_renderer) = world.get::<BevyMeshRenderer>(entity).map(|r| r.0) else {
                return Ok(None);
            };

            let table = lua.create_table()?;
            table.set("casts_shadow", mesh_renderer.casts_shadow)?;
            table.set("visible", mesh_renderer.visible)?;

            if let Some(editor_mesh) = world.get::<EditorMesh>(entity) {
                match &editor_mesh.source {
                    MeshSource::Primitive(PrimitiveKind::Cube) => {
                        table.set("source", "Cube")?;
                    }
                    MeshSource::Primitive(PrimitiveKind::UvSphere(_, _)) => {
                        table.set("source", "UV Sphere")?;
                    }
                    MeshSource::Primitive(PrimitiveKind::Plane) => {
                        table.set("source", "Plane")?;
                    }
                    MeshSource::Asset { path } => {
                        table.set("source", path.clone())?;
                    }
                }

                if let Some(material) = &editor_mesh.material_path {
                    table.set("material", material.clone())?;
                }
            }

            Ok(Some(table))
        })?,
    )?;

    let world_ptr_get_mesh_source_path = world_ptr;
    ecs.set(
        "get_mesh_renderer_source_path",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_mesh_source_path as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(editor_mesh) = world.get::<EditorMesh>(entity) else {
                return Ok(None);
            };
            match &editor_mesh.source {
                MeshSource::Asset { path } => Ok(Some(path.clone())),
                MeshSource::Primitive(_) => Ok(None),
            }
        })?,
    )?;

    let world_ptr_get_mesh_material_path = world_ptr;
    ecs.set(
        "get_mesh_renderer_material_path",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_mesh_material_path as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            Ok(world
                .get::<EditorMesh>(entity)
                .and_then(|mesh| mesh.material_path.clone()))
        })?,
    )?;

    let world_ptr_set_mesh = world_ptr;
    ecs.set(
        "set_mesh_renderer",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_mesh as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };

            let project = world.get_resource::<EditorProject>().cloned();
            let existing_mesh = world.get::<EditorMesh>(entity).cloned();
            let existing_renderer = world.get::<BevyMeshRenderer>(entity).map(|r| r.0);

            let mut source = existing_mesh
                .as_ref()
                .map(|mesh| mesh.source.clone())
                .unwrap_or(MeshSource::Primitive(PrimitiveKind::Cube));
            let mut material_path = existing_mesh.and_then(|mesh| mesh.material_path.clone());
            let mut casts_shadow = existing_renderer
                .map(|renderer| renderer.casts_shadow)
                .unwrap_or(true);
            let mut visible = existing_renderer
                .map(|renderer| renderer.visible)
                .unwrap_or(true);

            if let Ok(value) = data.get::<Value>("source") {
                if let Some(parsed) = parse_mesh_source(value, project.as_ref()) {
                    source = parsed;
                }
            }

            if let Ok(value) = data.get::<Value>("material") {
                material_path = parse_material_path(value, project.as_ref());
            }

            if let Ok(value) = data.get::<bool>("casts_shadow") {
                casts_shadow = value;
            }
            if let Ok(value) = data.get::<bool>("visible") {
                visible = value;
            }

            ensure_transform(world, entity);
            Ok(apply_mesh_renderer(
                world,
                entity,
                project.as_ref(),
                source,
                material_path,
                casts_shadow,
                visible,
            ))
        })?,
    )?;

    let world_ptr_set_mesh_source_path = world_ptr;
    ecs.set(
        "set_mesh_renderer_source_path",
        lua.create_function(move |_, (entity_id, path): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_set_mesh_source_path as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let trimmed = path.trim();
            if trimmed.is_empty() {
                return Ok(false);
            }

            let project = world.get_resource::<EditorProject>().cloned();
            let source = MeshSource::Asset {
                path: normalize_project_path(project.as_ref(), Path::new(trimmed)),
            };
            let material_path = world
                .get::<EditorMesh>(entity)
                .and_then(|mesh| mesh.material_path.clone());
            let existing_renderer = world
                .get::<BevyMeshRenderer>(entity)
                .map(|renderer| renderer.0);
            let casts_shadow = existing_renderer
                .map(|renderer| renderer.casts_shadow)
                .unwrap_or(true);
            let visible = existing_renderer
                .map(|renderer| renderer.visible)
                .unwrap_or(true);

            ensure_transform(world, entity);
            Ok(apply_mesh_renderer(
                world,
                entity,
                project.as_ref(),
                source,
                material_path,
                casts_shadow,
                visible,
            ))
        })?,
    )?;

    let world_ptr_set_mesh_material_path = world_ptr;
    ecs.set(
        "set_mesh_renderer_material_path",
        lua.create_function(move |_, (entity_id, path): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_set_mesh_material_path as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let project = world.get_resource::<EditorProject>().cloned();
            let source = world
                .get::<EditorMesh>(entity)
                .map(|mesh| mesh.source.clone())
                .unwrap_or(MeshSource::Primitive(PrimitiveKind::Cube));
            let material_path = if path.trim().is_empty() {
                None
            } else {
                Some(normalize_project_path(
                    project.as_ref(),
                    Path::new(path.trim()),
                ))
            };
            let existing_renderer = world
                .get::<BevyMeshRenderer>(entity)
                .map(|renderer| renderer.0);
            let casts_shadow = existing_renderer
                .map(|renderer| renderer.casts_shadow)
                .unwrap_or(true);
            let visible = existing_renderer
                .map(|renderer| renderer.visible)
                .unwrap_or(true);

            ensure_transform(world, entity);
            Ok(apply_mesh_renderer(
                world,
                entity,
                project.as_ref(),
                source,
                material_path,
                casts_shadow,
                visible,
            ))
        })?,
    )?;

    let world_ptr_get_scene = world_ptr;
    ecs.set(
        "get_scene_asset",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_scene as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            Ok(world
                .get::<SceneAssetPath>(entity)
                .map(|asset| asset.path.to_string_lossy().to_string()))
        })?,
    )?;

    let world_ptr_set_scene = world_ptr;
    ecs.set(
        "set_scene_asset",
        lua.create_function(move |_, (entity_id, path): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_set_scene as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let project = world.get_resource::<EditorProject>().cloned();
            let resolved = resolve_project_path(project.as_ref(), Path::new(&path));
            let asset_server = match world.get_resource::<BevyAssetServer>() {
                Some(server) => BevyAssetServer(server.0.clone()),
                None => return Ok(false),
            };
            let handle = if let Some(mut cache) = world.get_resource_mut::<EditorAssetCache>() {
                cached_scene_handle(&mut cache, &asset_server, &resolved)
            } else {
                asset_server.0.lock().load_scene(&resolved)
            };
            reset_scene_root_instance(world, entity);
            world.entity_mut(entity).insert(SceneRoot(handle));
            world
                .entity_mut(entity)
                .insert(SceneAssetPath { path: resolved });
            Ok(true)
        })?,
    )?;

    let world_ptr_open_scene = world_ptr;
    ecs.set(
        "open_scene",
        lua.create_function(move |_, path: String| {
            let world = unsafe { &mut *(world_ptr_open_scene as *mut World) };
            let trimmed = path.trim();
            if trimmed.is_empty() {
                return Ok(false);
            }

            let resolved = {
                let project = world.get_resource::<EditorProject>();
                resolve_project_path(project, Path::new(trimmed))
            };

            let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() else {
                return Ok(false);
            };
            queue.push(EditorCommand::OpenScene { path: resolved });
            Ok(true)
        })?,
    )?;

    let world_ptr_switch_scene = world_ptr;
    ecs.set(
        "switch_scene",
        lua.create_function(move |_, path: String| {
            let world = unsafe { &mut *(world_ptr_switch_scene as *mut World) };
            let trimmed = path.trim();
            if trimmed.is_empty() {
                return Ok(false);
            }

            let resolved = {
                let project = world.get_resource::<EditorProject>();
                resolve_project_path(project, Path::new(trimmed))
            };

            let Some(mut queue) = world.get_resource_mut::<EditorCommandQueue>() else {
                return Ok(false);
            };
            queue.push(EditorCommand::OpenScene { path: resolved });
            Ok(true)
        })?,
    )?;

    let world_ptr_get_script = world_ptr;
    ecs.set(
        "get_script",
        lua.create_function(move |lua, (entity_id, index): (u64, Option<Value>)| {
            let world = unsafe { &mut *(world_ptr_get_script as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(scripts) = world.get::<ScriptComponent>(entity) else {
                return Ok(None);
            };
            let selected_index = match index {
                Some(Value::Integer(raw)) => Some((raw.max(1) as usize).saturating_sub(1)),
                Some(Value::Number(raw)) if raw.is_finite() => {
                    Some((raw.round().max(1.0) as usize).saturating_sub(1))
                }
                Some(Value::Nil) | None => None,
                _ => return Ok(None),
            };
            let script = if let Some(index) = selected_index {
                scripts.scripts.get(index)
            } else {
                scripts.scripts.first()
            };
            let Some(script) = script else {
                return Ok(None);
            };
            let Some(path) = script.path.as_ref() else {
                return Ok(None);
            };
            let table = lua.create_table()?;
            table.set("path", path.to_string_lossy().to_string())?;
            table.set("language", script.language.clone())?;
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_get_script_path = world_ptr;
    ecs.set(
        "get_script_path",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_script_path as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(scripts) = world.get::<ScriptComponent>(entity) else {
                return Ok(None);
            };
            let Some(script) = scripts.scripts.first() else {
                return Ok(None);
            };
            Ok(script
                .path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()))
        })?,
    )?;

    let world_ptr_get_script_language = world_ptr;
    ecs.set(
        "get_script_language",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_script_language as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(scripts) = world.get::<ScriptComponent>(entity) else {
                return Ok(None);
            };
            let Some(script) = scripts.scripts.first() else {
                return Ok(None);
            };
            Ok(Some(script.language.clone()))
        })?,
    )?;

    let world_ptr_set_script = world_ptr;
    ecs.set(
        "set_script",
        lua.create_function(
            move |_, (entity_id, path, language): (u64, String, Option<String>)| {
                let world = unsafe { &mut *(world_ptr_set_script as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let project = world.get_resource::<EditorProject>().cloned();
                let resolved = resolve_project_path(project.as_ref(), Path::new(&path));
                let language = normalize_script_language(
                    &language.unwrap_or_default(),
                    Some(resolved.as_path()),
                );
                if let Some(mut scripts) = world.get_mut::<ScriptComponent>(entity) {
                    if scripts.scripts.is_empty() {
                        scripts.scripts.push(ScriptEntry {
                            path: Some(resolved),
                            language,
                        });
                    } else {
                        scripts.scripts[0].path = Some(resolved);
                        scripts.scripts[0].language = language;
                    }
                } else {
                    world.entity_mut(entity).insert(ScriptComponent {
                        scripts: vec![ScriptEntry {
                            path: Some(resolved),
                            language,
                        }],
                    });
                }
                Ok(true)
            },
        )?,
    )?;

    let world_ptr_list_dyn = world_ptr;
    ecs.set(
        "list_dynamic_components",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_list_dyn as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(dynamic) = world.get::<DynamicComponents>(entity) else {
                return Ok(None);
            };
            let list = lua.create_table()?;
            for (index, component) in dynamic.components.iter().enumerate() {
                let table = lua.create_table()?;
                table.set("name", component.name.clone())?;
                let fields = lua.create_table()?;
                for field in &component.fields {
                    fields.set(field.name.clone(), dynamic_value_to_lua(lua, &field.value)?)?;
                }
                table.set("fields", fields)?;
                list.set(index + 1, table)?;
            }
            Ok(Some(list))
        })?,
    )?;

    let world_ptr_get_dyn = world_ptr;
    ecs.set(
        "get_dynamic_component",
        lua.create_function(move |lua, (entity_id, name): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_get_dyn as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(dynamic) = world.get::<DynamicComponents>(entity) else {
                return Ok(None);
            };
            let Some(component) = dynamic.components.iter().find(|comp| comp.name == name) else {
                return Ok(None);
            };
            let fields = lua.create_table()?;
            for field in &component.fields {
                fields.set(field.name.clone(), dynamic_value_to_lua(lua, &field.value)?)?;
            }
            Ok(Some(fields))
        })?,
    )?;

    let world_ptr_set_dyn = world_ptr;
    ecs.set(
        "set_dynamic_component",
        lua.create_function(move |_, (entity_id, name, fields): (u64, String, Table)| {
            let world = unsafe { &mut *(world_ptr_set_dyn as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(index) = ensure_dynamic_component_index(world, entity, &name) else {
                return Ok(false);
            };
            let Some(mut dynamic) = world.get_mut::<DynamicComponents>(entity) else {
                return Ok(false);
            };
            apply_dynamic_fields(&mut dynamic.components[index], fields);
            Ok(true)
        })?,
    )?;

    let world_ptr_get_dyn_field = world_ptr;
    ecs.set(
        "get_dynamic_field",
        lua.create_function(
            move |lua, (entity_id, comp_name, field_name): (u64, String, String)| {
                let world = unsafe { &mut *(world_ptr_get_dyn_field as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(None);
                };
                let Some(dynamic) = world.get::<DynamicComponents>(entity) else {
                    return Ok(None);
                };
                let Some(component) = dynamic
                    .components
                    .iter()
                    .find(|comp| comp.name == comp_name)
                else {
                    return Ok(None);
                };
                let Some(field) = component
                    .fields
                    .iter()
                    .find(|field| field.name == field_name)
                else {
                    return Ok(None);
                };
                Ok(Some(dynamic_value_to_lua(lua, &field.value)?))
            },
        )?,
    )?;

    let world_ptr_set_dyn_field = world_ptr;
    ecs.set(
        "set_dynamic_field",
        lua.create_function(
            move |_, (entity_id, comp_name, field_name, value): (u64, String, String, Value)| {
                let world = unsafe { &mut *(world_ptr_set_dyn_field as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(value) = lua_value_to_dynamic(value) else {
                    return Ok(false);
                };

                let Some(index) = ensure_dynamic_component_index(world, entity, &comp_name) else {
                    return Ok(false);
                };
                let Some(mut dynamic) = world.get_mut::<DynamicComponents>(entity) else {
                    return Ok(false);
                };
                let component = &mut dynamic.components[index];
                if let Some(field) = component
                    .fields
                    .iter_mut()
                    .find(|field| field.name == field_name)
                {
                    field.value = value;
                } else {
                    component.fields.push(DynamicField {
                        name: field_name,
                        value,
                    });
                }
                Ok(true)
            },
        )?,
    )?;

    let world_ptr_remove_dyn = world_ptr;
    ecs.set(
        "remove_dynamic_component",
        lua.create_function(move |_, (entity_id, name): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_remove_dyn as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let Some(mut dynamic) = world.get_mut::<DynamicComponents>(entity) else {
                return Ok(false);
            };
            let before = dynamic.components.len();
            dynamic
                .components
                .retain(|component| component.name != name);
            Ok(before != dynamic.components.len())
        })?,
    )?;

    let world_ptr_remove_dyn_field = world_ptr;
    ecs.set(
        "remove_dynamic_field",
        lua.create_function(
            move |_, (entity_id, comp_name, field_name): (u64, String, String)| {
                let world = unsafe { &mut *(world_ptr_remove_dyn_field as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(mut dynamic) = world.get_mut::<DynamicComponents>(entity) else {
                    return Ok(false);
                };
                if let Some(component) = dynamic
                    .components
                    .iter_mut()
                    .find(|component| component.name == comp_name)
                {
                    let before = component.fields.len();
                    component.fields.retain(|field| field.name != field_name);
                    return Ok(before != component.fields.len());
                }
                Ok(false)
            },
        )?,
    )?;

    let world_ptr_get_audio_emitter = world_ptr;
    ecs.set(
        "get_audio_emitter",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_audio_emitter as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(emitter) = world
                .get::<BevyAudioEmitter>(entity)
                .map(|emitter| emitter.0)
            else {
                return Ok(None);
            };
            let editor_audio = world
                .get::<EditorAudio>(entity)
                .cloned()
                .unwrap_or(EditorAudio {
                    path: None,
                    streaming: false,
                });

            let table = lua.create_table()?;
            match editor_audio.path {
                Some(path) => table.set("path", path)?,
                None => table.set("path", Value::Nil)?,
            }
            table.set("streaming", editor_audio.streaming)?;
            table.set("bus", audio_bus_to_lua(lua, emitter.bus)?)?;
            table.set("volume", emitter.volume)?;
            table.set("pitch", emitter.pitch)?;
            table.set("looping", emitter.looping)?;
            table.set("spatial", emitter.spatial)?;
            table.set("min_distance", emitter.min_distance)?;
            table.set("max_distance", emitter.max_distance)?;
            table.set("rolloff", emitter.rolloff)?;
            table.set("spatial_blend", emitter.spatial_blend)?;
            table.set(
                "playback_state",
                audio_playback_state_name(emitter.playback_state),
            )?;
            table.set("play_on_spawn", emitter.play_on_spawn)?;
            if let Some(clip_id) = emitter.clip_id {
                table.set("clip_id", clip_id as u64)?;
            }
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_get_audio_emitter_path = world_ptr;
    ecs.set(
        "get_audio_emitter_path",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_audio_emitter_path as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            Ok(world
                .get::<EditorAudio>(entity)
                .and_then(|audio| audio.path.clone()))
        })?,
    )?;

    let world_ptr_set_audio_emitter = world_ptr;
    ecs.set(
        "set_audio_emitter",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_audio_emitter as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };

            let mut emitter = world
                .get::<BevyAudioEmitter>(entity)
                .map(|emitter| emitter.0)
                .unwrap_or_default();
            let mut editor_audio =
                world
                    .get::<EditorAudio>(entity)
                    .cloned()
                    .unwrap_or(EditorAudio {
                        path: None,
                        streaming: false,
                    });
            let project = world.get_resource::<EditorProject>().cloned();

            if let Ok(bus) = data.get::<Value>("bus") {
                if let Some(parsed) =
                    audio_bus_from_value(bus, world.get_resource::<AudioBackendResource>())
                {
                    emitter.bus = parsed;
                }
            }
            if let Ok(volume) = data.get::<f32>("volume") {
                emitter.volume = volume.max(0.0);
            }
            if let Ok(pitch) = data.get::<f32>("pitch") {
                emitter.pitch = pitch.max(0.001);
            }
            if let Ok(looping) = data.get::<bool>("looping") {
                emitter.looping = looping;
            }
            if let Ok(spatial) = data.get::<bool>("spatial") {
                emitter.spatial = spatial;
            }
            if let Ok(min_distance) = data.get::<f32>("min_distance") {
                emitter.min_distance = min_distance.max(0.0);
            }
            if let Ok(max_distance) = data.get::<f32>("max_distance") {
                emitter.max_distance = max_distance.max(0.0);
            }
            if emitter.max_distance < emitter.min_distance {
                emitter.max_distance = emitter.min_distance;
            }
            if let Ok(rolloff) = data.get::<f32>("rolloff") {
                emitter.rolloff = rolloff.max(0.0);
            }
            if let Ok(spatial_blend) = data.get::<f32>("spatial_blend") {
                emitter.spatial_blend = spatial_blend.clamp(0.0, 1.0);
            }
            if let Ok(play_on_spawn) = data.get::<bool>("play_on_spawn") {
                emitter.play_on_spawn = play_on_spawn;
            }
            if let Ok(playback_state_value) = data.get::<Value>("playback_state") {
                if let Some(state) = audio_playback_state_from_value(playback_state_value) {
                    emitter.playback_state = state;
                }
            }
            if let Ok(streaming) = data.get::<bool>("streaming") {
                editor_audio.streaming = streaming;
            }

            if let Ok(path_value) = data.get::<Value>("path") {
                match path_value {
                    Value::Nil => {
                        editor_audio.path = None;
                    }
                    Value::String(path) => {
                        let path = path.to_string_lossy().to_string();
                        if path.trim().is_empty() {
                            editor_audio.path = None;
                        } else {
                            let normalized =
                                normalize_project_path(project.as_ref(), Path::new(path.trim()));
                            editor_audio.path = Some(normalized);
                        }
                    }
                    _ => {}
                }
            }

            if !apply_audio_emitter_asset(world, project.as_ref(), &mut emitter, &editor_audio) {
                emitter.clip_id = None;
            }

            ensure_transform(world, entity);
            world
                .entity_mut(entity)
                .insert((BevyWrapper(emitter), editor_audio));
            Ok(true)
        })?,
    )?;

    let world_ptr_set_audio_emitter_path = world_ptr;
    ecs.set(
        "set_audio_emitter_path",
        lua.create_function(move |_, (entity_id, path): (u64, String)| {
            let world = unsafe { &mut *(world_ptr_set_audio_emitter_path as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };

            let mut emitter = world
                .get::<BevyAudioEmitter>(entity)
                .map(|emitter| emitter.0)
                .unwrap_or_default();
            let mut editor_audio =
                world
                    .get::<EditorAudio>(entity)
                    .cloned()
                    .unwrap_or(EditorAudio {
                        path: None,
                        streaming: false,
                    });
            let project = world.get_resource::<EditorProject>().cloned();

            editor_audio.path = if path.trim().is_empty() {
                None
            } else {
                Some(normalize_project_path(
                    project.as_ref(),
                    Path::new(path.trim()),
                ))
            };

            if !apply_audio_emitter_asset(world, project.as_ref(), &mut emitter, &editor_audio) {
                emitter.clip_id = None;
            }

            ensure_transform(world, entity);
            world
                .entity_mut(entity)
                .insert((BevyWrapper(emitter), editor_audio));
            Ok(true)
        })?,
    )?;

    let world_ptr_get_audio_listener = world_ptr;
    ecs.set(
        "get_audio_listener",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_audio_listener as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(listener) = world
                .get::<BevyAudioListener>(entity)
                .map(|listener| listener.0)
            else {
                return Ok(None);
            };
            let table = lua.create_table()?;
            table.set("enabled", listener.enabled)?;
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_set_audio_listener = world_ptr;
    ecs.set(
        "set_audio_listener",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_audio_listener as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let mut listener = world
                .get::<BevyAudioListener>(entity)
                .map(|listener| listener.0)
                .unwrap_or_default();
            if let Ok(enabled) = data.get::<bool>("enabled") {
                listener.enabled = enabled;
            }
            ensure_transform(world, entity);
            world.entity_mut(entity).insert(BevyWrapper(listener));
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_enabled = world_ptr;
    ecs.set(
        "set_audio_enabled",
        lua.create_function(move |_, enabled: bool| {
            let world = unsafe { &mut *(world_ptr_audio_enabled as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            audio.0.set_enabled(enabled);
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_is_enabled = world_ptr;
    ecs.set(
        "get_audio_enabled",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_audio_is_enabled as *mut World) };
            Ok(world
                .get_resource::<AudioBackendResource>()
                .map(|audio| audio.0.enabled())
                .unwrap_or(false))
        })?,
    )?;

    let world_ptr_audio_buses = world_ptr;
    ecs.set(
        "list_audio_buses",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_audio_buses as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(lua.create_table()?);
            };
            let table = lua.create_table()?;
            let buses = audio.0.bus_list();
            for (index, bus) in buses.iter().enumerate() {
                table.set(index + 1, audio_bus_to_lua(lua, *bus)?)?;
            }
            Ok(table)
        })?,
    )?;

    let world_ptr_audio_create_bus = world_ptr;
    ecs.set(
        "create_audio_bus",
        lua.create_function(move |lua, name: Option<String>| {
            let world = unsafe { &mut *(world_ptr_audio_create_bus as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(Value::Nil);
            };
            let bus = audio.0.create_custom_bus(name);
            audio_bus_to_lua(lua, bus)
        })?,
    )?;

    let world_ptr_audio_remove_bus = world_ptr;
    ecs.set(
        "remove_audio_bus",
        lua.create_function(move |_, bus: Value| {
            let world = unsafe { &mut *(world_ptr_audio_remove_bus as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            let Some(bus) = audio_bus_from_value(bus, Some(audio)) else {
                return Ok(false);
            };
            audio.0.remove_bus(bus);
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_bus_name = world_ptr;
    ecs.set(
        "get_audio_bus_name",
        lua.create_function(move |_, bus: Value| {
            let world = unsafe { &mut *(world_ptr_audio_bus_name as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(None);
            };
            let Some(bus) = audio_bus_from_value(bus, Some(audio)) else {
                return Ok(None);
            };
            Ok(Some(audio.0.bus_name(bus)))
        })?,
    )?;

    let world_ptr_audio_set_bus_name = world_ptr;
    ecs.set(
        "set_audio_bus_name",
        lua.create_function(move |_, (bus, name): (Value, String)| {
            let world = unsafe { &mut *(world_ptr_audio_set_bus_name as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            let Some(bus) = audio_bus_from_value(bus, Some(audio)) else {
                return Ok(false);
            };
            audio.0.set_bus_name(bus, name);
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_set_bus_volume = world_ptr;
    ecs.set(
        "set_audio_bus_volume",
        lua.create_function(move |_, (bus, volume): (Value, f32)| {
            let world = unsafe { &mut *(world_ptr_audio_set_bus_volume as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            let Some(bus) = audio_bus_from_value(bus, Some(audio)) else {
                return Ok(false);
            };
            audio.0.set_bus_volume(bus, volume.max(0.0));
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_get_bus_volume = world_ptr;
    ecs.set(
        "get_audio_bus_volume",
        lua.create_function(move |_, bus: Value| {
            let world = unsafe { &mut *(world_ptr_audio_get_bus_volume as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(None);
            };
            let Some(bus) = audio_bus_from_value(bus, Some(audio)) else {
                return Ok(None);
            };
            Ok(Some(audio.0.bus_volume(bus)))
        })?,
    )?;

    let world_ptr_audio_set_scene_volume = world_ptr;
    ecs.set(
        "set_audio_scene_volume",
        lua.create_function(move |_, (scene_id, volume): (u64, f32)| {
            let world = unsafe { &mut *(world_ptr_audio_set_scene_volume as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            audio.0.set_scene_volume(scene_id, volume.max(0.0));
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_get_scene_volume = world_ptr;
    ecs.set(
        "get_audio_scene_volume",
        lua.create_function(move |_, scene_id: u64| {
            let world = unsafe { &mut *(world_ptr_audio_get_scene_volume as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(None);
            };
            Ok(Some(audio.0.scene_volume(scene_id)))
        })?,
    )?;

    let world_ptr_audio_clear = world_ptr;
    ecs.set(
        "clear_audio_emitters",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_audio_clear as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            audio.0.clear_emitters();
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_set_head_width = world_ptr;
    ecs.set(
        "set_audio_head_width",
        lua.create_function(move |_, width: f32| {
            let world = unsafe { &mut *(world_ptr_audio_set_head_width as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            audio.0.set_head_width(width);
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_get_head_width = world_ptr;
    ecs.set(
        "get_audio_head_width",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_audio_get_head_width as *mut World) };
            Ok(world
                .get_resource::<AudioBackendResource>()
                .map(|audio| audio.0.head_width()))
        })?,
    )?;

    let world_ptr_audio_set_speed = world_ptr;
    ecs.set(
        "set_audio_speed_of_sound",
        lua.create_function(move |_, speed: f32| {
            let world = unsafe { &mut *(world_ptr_audio_set_speed as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            audio.0.set_speed_of_sound(speed);
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_get_speed = world_ptr;
    ecs.set(
        "get_audio_speed_of_sound",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_audio_get_speed as *mut World) };
            Ok(world
                .get_resource::<AudioBackendResource>()
                .map(|audio| audio.0.speed_of_sound()))
        })?,
    )?;

    let world_ptr_audio_set_streaming = world_ptr;
    ecs.set(
        "set_audio_streaming_config",
        lua.create_function(move |_, (buffer_frames, chunk_frames): (usize, usize)| {
            let world = unsafe { &mut *(world_ptr_audio_set_streaming as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(false);
            };
            audio.0.set_streaming_config(buffer_frames, chunk_frames);
            Ok(true)
        })?,
    )?;

    let world_ptr_audio_get_streaming = world_ptr;
    ecs.set(
        "get_audio_streaming_config",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_audio_get_streaming as *mut World) };
            let Some(audio) = world.get_resource::<AudioBackendResource>() else {
                return Ok(None);
            };
            let (buffer_frames, chunk_frames) = audio.0.streaming_config();
            let table = lua.create_table()?;
            table.set("buffer_frames", buffer_frames as u64)?;
            table.set("chunk_frames", chunk_frames as u64)?;
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_get_physics = world_ptr;
    ecs.set(
        "get_physics",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_physics as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            if !has_physics_component(world, entity) {
                return Ok(None);
            }
            Ok(Some(physics_entity_to_table(lua, world, entity)?))
        })?,
    )?;

    let world_ptr_set_physics = world_ptr;
    ecs.set(
        "set_physics",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_physics as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            ensure_transform(world, entity);
            apply_physics_patch(world, entity, &data);
            Ok(true)
        })?,
    )?;

    let world_ptr_clear_physics = world_ptr;
    ecs.set(
        "clear_physics",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_clear_physics as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            remove_all_physics_components(world, entity);
            Ok(true)
        })?,
    )?;

    let world_ptr_get_physics_world_defaults = world_ptr;
    ecs.set(
        "get_physics_world_defaults",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_physics_world_defaults as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(defaults) = world.get::<PhysicsWorldDefaults>(entity).copied() else {
                return Ok(None);
            };
            Ok(Some(physics_world_defaults_to_table(lua, defaults)?))
        })?,
    )?;

    let world_ptr_set_physics_world_defaults = world_ptr;
    ecs.set(
        "set_physics_world_defaults",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_physics_world_defaults as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            let mut defaults = world
                .get::<PhysicsWorldDefaults>(entity)
                .copied()
                .unwrap_or_default();
            patch_physics_world_defaults(&mut defaults, &data);
            world.entity_mut(entity).insert(defaults);
            Ok(true)
        })?,
    )?;

    let world_ptr_get_character_output = world_ptr;
    ecs.set(
        "get_character_controller_output",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_character_output as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(output) = world.get::<CharacterControllerOutput>(entity).copied() else {
                return Ok(None);
            };
            Ok(Some(character_output_to_table(lua, output)?))
        })?,
    )?;

    let world_ptr_get_ray_hit = world_ptr;
    ecs.set(
        "get_physics_ray_cast_hit",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_ray_hit as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(hit) = world.get::<PhysicsRayCastHit>(entity).copied() else {
                return Ok(None);
            };
            Ok(Some(ray_cast_hit_to_table(lua, hit)?))
        })?,
    )?;

    let world_ptr_get_point_hit = world_ptr;
    ecs.set(
        "get_physics_point_projection_hit",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_point_hit as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(hit) = world.get::<PhysicsPointProjectionHit>(entity).copied() else {
                return Ok(None);
            };
            Ok(Some(point_projection_hit_to_table(lua, hit)?))
        })?,
    )?;

    let world_ptr_get_shape_hit = world_ptr;
    ecs.set(
        "get_physics_shape_cast_hit",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_shape_hit as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(hit) = world.get::<PhysicsShapeCastHit>(entity).copied() else {
                return Ok(None);
            };
            Ok(Some(shape_cast_hit_to_table(lua, hit)?))
        })?,
    )?;

    let world_ptr_ray_cast = world_ptr;
    ecs.set(
        "ray_cast",
        lua.create_function(
            move |lua,
                  (origin, direction, max_toi, solid, filter, exclude_entity): (
                Table,
                Table,
                Option<f32>,
                Option<bool>,
                Option<Table>,
                Option<u64>,
            )| {
                let world = unsafe { &mut *(world_ptr_ray_cast as *mut World) };
                let mut hit = PhysicsRayCastHit::default();
                let Some(origin) = table_to_vec3(&origin) else {
                    return ray_cast_hit_to_table(lua, hit);
                };
                let Some(direction) = table_to_vec3(&direction) else {
                    return ray_cast_hit_to_table(lua, hit);
                };

                let mut query_filter = PhysicsQueryFilter::default();
                if let Some(filter) = filter {
                    patch_query_filter(&mut query_filter, &filter);
                }

                let exclude_entity =
                    exclude_entity.and_then(|entity_id| lookup_editor_entity(world, entity_id));

                if let Some(phys) = world.get_resource::<PhysicsResource>() {
                    hit = phys.cast_ray(
                        origin,
                        direction,
                        max_toi.unwrap_or(10_000.0),
                        solid.unwrap_or(true),
                        query_filter,
                        exclude_entity,
                    );
                }

                ray_cast_hit_to_table(lua, hit)
            },
        )?,
    )?;

    let world_ptr_get_velocity = world_ptr;
    ecs.set(
        "get_physics_velocity",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_velocity as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(table) = physics_velocity_to_table(lua, world, entity)? else {
                return Ok(None);
            };
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_set_velocity = world_ptr;
    ecs.set(
        "set_physics_velocity",
        lua.create_function(move |_, (entity_id, data): (u64, Table)| {
            let world = unsafe { &mut *(world_ptr_set_velocity as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            Ok(set_physics_velocity(world, entity, &data))
        })?,
    )?;

    let world_ptr_add_force = world_ptr;
    ecs.set(
        "add_force",
        lua.create_function(
            move |_, (entity_id, force, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_add_force as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(force) = table_to_vec3(&force) else {
                    return Ok(false);
                };
                Ok(add_transient_force(
                    world,
                    entity,
                    force,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_add_torque = world_ptr;
    ecs.set(
        "add_torque",
        lua.create_function(
            move |_, (entity_id, torque, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_add_torque as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(torque) = table_to_vec3(&torque) else {
                    return Ok(false);
                };
                Ok(add_transient_torque(
                    world,
                    entity,
                    torque,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_add_force_at_point = world_ptr;
    ecs.set(
        "add_force_at_point",
        lua.create_function(
            move |_, (entity_id, force, point, wake_up): (u64, Table, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_add_force_at_point as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(force) = table_to_vec3(&force) else {
                    return Ok(false);
                };
                let Some(point) = table_to_vec3(&point) else {
                    return Ok(false);
                };
                Ok(add_transient_force_at_point(
                    world,
                    entity,
                    force,
                    point,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_add_persistent_force = world_ptr;
    ecs.set(
        "add_persistent_force",
        lua.create_function(
            move |_, (entity_id, force, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_add_persistent_force as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(force) = table_to_vec3(&force) else {
                    return Ok(false);
                };
                Ok(add_persistent_force(
                    world,
                    entity,
                    force,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_set_persistent_force = world_ptr;
    ecs.set(
        "set_persistent_force",
        lua.create_function(
            move |_, (entity_id, force, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_set_persistent_force as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(force) = table_to_vec3(&force) else {
                    return Ok(false);
                };
                Ok(set_persistent_force(
                    world,
                    entity,
                    force,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_add_persistent_torque = world_ptr;
    ecs.set(
        "add_persistent_torque",
        lua.create_function(
            move |_, (entity_id, torque, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_add_persistent_torque as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(torque) = table_to_vec3(&torque) else {
                    return Ok(false);
                };
                Ok(add_persistent_torque(
                    world,
                    entity,
                    torque,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_set_persistent_torque = world_ptr;
    ecs.set(
        "set_persistent_torque",
        lua.create_function(
            move |_, (entity_id, torque, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_set_persistent_torque as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(torque) = table_to_vec3(&torque) else {
                    return Ok(false);
                };
                Ok(set_persistent_torque(
                    world,
                    entity,
                    torque,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_add_persistent_force_at_point = world_ptr;
    ecs.set(
        "add_persistent_force_at_point",
        lua.create_function(
            move |_, (entity_id, force, point, wake_up): (u64, Table, Table, Option<bool>)| {
                let world =
                    unsafe { &mut *(world_ptr_add_persistent_force_at_point as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(force) = table_to_vec3(&force) else {
                    return Ok(false);
                };
                let Some(point) = table_to_vec3(&point) else {
                    return Ok(false);
                };
                Ok(add_persistent_force_at_point(
                    world,
                    entity,
                    force,
                    point,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_clear_persistent_forces = world_ptr;
    ecs.set(
        "clear_persistent_forces",
        lua.create_function(move |_, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_clear_persistent_forces as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(false);
            };
            Ok(clear_persistent_forces(world, entity))
        })?,
    )?;

    let world_ptr_apply_impulse = world_ptr;
    ecs.set(
        "apply_impulse",
        lua.create_function(
            move |_, (entity_id, impulse, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_apply_impulse as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(impulse) = table_to_vec3(&impulse) else {
                    return Ok(false);
                };
                Ok(queue_physics_impulse(
                    world,
                    entity,
                    impulse,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_apply_angular_impulse = world_ptr;
    ecs.set(
        "apply_angular_impulse",
        lua.create_function(
            move |_, (entity_id, impulse, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_apply_angular_impulse as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(impulse) = table_to_vec3(&impulse) else {
                    return Ok(false);
                };
                Ok(queue_physics_angular_impulse(
                    world,
                    entity,
                    impulse,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_apply_torque_impulse = world_ptr;
    ecs.set(
        "apply_torque_impulse",
        lua.create_function(
            move |_, (entity_id, impulse, wake_up): (u64, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_apply_torque_impulse as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(impulse) = table_to_vec3(&impulse) else {
                    return Ok(false);
                };
                Ok(queue_physics_angular_impulse(
                    world,
                    entity,
                    impulse,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_apply_impulse_at_point = world_ptr;
    ecs.set(
        "apply_impulse_at_point",
        lua.create_function(
            move |_, (entity_id, impulse, point, wake_up): (u64, Table, Table, Option<bool>)| {
                let world = unsafe { &mut *(world_ptr_apply_impulse_at_point as *mut World) };
                let Some(entity) = lookup_editor_entity(world, entity_id) else {
                    return Ok(false);
                };
                let Some(impulse) = table_to_vec3(&impulse) else {
                    return Ok(false);
                };
                let Some(point) = table_to_vec3(&point) else {
                    return Ok(false);
                };
                Ok(queue_physics_impulse_at_point(
                    world,
                    entity,
                    impulse,
                    point,
                    wake_up.unwrap_or(true),
                ))
            },
        )?,
    )?;

    let world_ptr_set_phys_running = world_ptr;
    ecs.set(
        "set_physics_running",
        lua.create_function(move |_, running: bool| {
            let world = unsafe { &mut *(world_ptr_set_phys_running as *mut World) };
            let Some(mut phys) = world.get_resource_mut::<PhysicsResource>() else {
                return Ok(false);
            };
            phys.running = running;
            Ok(true)
        })?,
    )?;

    let world_ptr_get_phys_running = world_ptr;
    ecs.set(
        "get_physics_running",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_get_phys_running as *mut World) };
            Ok(world
                .get_resource::<PhysicsResource>()
                .map(|phys| phys.running)
                .unwrap_or(false))
        })?,
    )?;

    let world_ptr_set_phys_gravity = world_ptr;
    ecs.set(
        "set_physics_gravity",
        lua.create_function(move |_, gravity: Table| {
            let world = unsafe { &mut *(world_ptr_set_phys_gravity as *mut World) };
            let Some(gravity) = table_to_vec3(&gravity) else {
                return Ok(false);
            };
            let Some(mut phys) = world.get_resource_mut::<PhysicsResource>() else {
                return Ok(false);
            };
            phys.gravity = gravity;
            Ok(true)
        })?,
    )?;

    let world_ptr_get_phys_gravity = world_ptr;
    ecs.set(
        "get_physics_gravity",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_get_phys_gravity as *mut World) };
            let Some(phys) = world.get_resource::<PhysicsResource>() else {
                return Ok(None);
            };
            Ok(Some(vec3_to_table(lua, phys.gravity)?))
        })?,
    )?;

    Ok(ecs)
}

fn has_physics_component(world: &World, entity: Entity) -> bool {
    world.get::<ColliderShape>(entity).is_some()
        || world.get::<DynamicRigidBody>(entity).is_some()
        || world.get::<KinematicRigidBody>(entity).is_some()
        || world.get::<FixedCollider>(entity).is_some()
        || world.get::<ColliderProperties>(entity).is_some()
        || world.get::<ColliderPropertyInheritance>(entity).is_some()
        || world.get::<RigidBodyProperties>(entity).is_some()
        || world.get::<RigidBodyPropertyInheritance>(entity).is_some()
        || world.get::<PhysicsJoint>(entity).is_some()
        || world.get::<CharacterController>(entity).is_some()
        || world.get::<CharacterControllerInput>(entity).is_some()
        || world.get::<CharacterControllerOutput>(entity).is_some()
        || world.get::<PhysicsRayCast>(entity).is_some()
        || world.get::<PhysicsRayCastHit>(entity).is_some()
        || world.get::<PhysicsPointProjection>(entity).is_some()
        || world.get::<PhysicsPointProjectionHit>(entity).is_some()
        || world.get::<PhysicsShapeCast>(entity).is_some()
        || world.get::<PhysicsShapeCastHit>(entity).is_some()
        || world.get::<PhysicsWorldDefaults>(entity).is_some()
}

fn set_physics_body_kind(world: &mut World, entity: Entity, body_kind: PhysicsBodyKind) {
    world.entity_mut(entity).remove::<DynamicRigidBody>();
    world.entity_mut(entity).remove::<KinematicRigidBody>();
    world.entity_mut(entity).remove::<FixedCollider>();
    world.entity_mut(entity).remove::<PhysicsHandle>();

    match body_kind {
        PhysicsBodyKind::Dynamic { mass } => {
            world.entity_mut(entity).insert(DynamicRigidBody {
                mass: mass.max(0.0),
            });
        }
        PhysicsBodyKind::Kinematic { mode } => {
            world.entity_mut(entity).insert(KinematicRigidBody { mode });
        }
        PhysicsBodyKind::Fixed => {
            world.entity_mut(entity).insert(FixedCollider);
        }
    }
}

fn remove_all_physics_components(world: &mut World, entity: Entity) {
    world.entity_mut(entity).remove::<ColliderShape>();
    world.entity_mut(entity).remove::<DynamicRigidBody>();
    world.entity_mut(entity).remove::<KinematicRigidBody>();
    world.entity_mut(entity).remove::<FixedCollider>();
    world.entity_mut(entity).remove::<ColliderProperties>();
    world
        .entity_mut(entity)
        .remove::<ColliderPropertyInheritance>();
    world.entity_mut(entity).remove::<RigidBodyProperties>();
    world
        .entity_mut(entity)
        .remove::<RigidBodyPropertyInheritance>();
    world.entity_mut(entity).remove::<PhysicsJoint>();
    world.entity_mut(entity).remove::<CharacterController>();
    world
        .entity_mut(entity)
        .remove::<CharacterControllerInput>();
    world
        .entity_mut(entity)
        .remove::<CharacterControllerOutput>();
    world.entity_mut(entity).remove::<PhysicsRayCast>();
    world.entity_mut(entity).remove::<PhysicsRayCastHit>();
    world.entity_mut(entity).remove::<PhysicsPointProjection>();
    world
        .entity_mut(entity)
        .remove::<PhysicsPointProjectionHit>();
    world.entity_mut(entity).remove::<PhysicsShapeCast>();
    world.entity_mut(entity).remove::<PhysicsShapeCastHit>();
    world.entity_mut(entity).remove::<PhysicsWorldDefaults>();
    world.entity_mut(entity).remove::<RigidBodyForces>();
    world
        .entity_mut(entity)
        .remove::<RigidBodyTransientForces>();
    world.entity_mut(entity).remove::<RigidBodyImpulseQueue>();
    world.entity_mut(entity).remove::<PhysicsHandle>();
}

fn audio_bus_to_lua(lua: &Lua, bus: AudioBus) -> mlua::Result<Value> {
    match bus {
        AudioBus::Master => Ok(Value::String(lua.create_string("Master")?)),
        AudioBus::Music => Ok(Value::String(lua.create_string("Music")?)),
        AudioBus::Sfx => Ok(Value::String(lua.create_string("Sfx")?)),
        AudioBus::Ui => Ok(Value::String(lua.create_string("Ui")?)),
        AudioBus::Ambience => Ok(Value::String(lua.create_string("Ambience")?)),
        AudioBus::World => Ok(Value::String(lua.create_string("World")?)),
        AudioBus::Custom(id) => Ok(Value::Integer(id as i64)),
    }
}

fn audio_bus_from_value(value: Value, backend: Option<&AudioBackendResource>) -> Option<AudioBus> {
    match value {
        Value::Integer(id) => {
            if id > 0 {
                Some(AudioBus::Custom(id as u32))
            } else {
                None
            }
        }
        Value::Number(id) => {
            if id > 0.0 {
                Some(AudioBus::Custom(id as u32))
            } else {
                None
            }
        }
        Value::String(raw) => {
            let raw = raw.to_string_lossy().to_string();
            let normalized = normalize_name(&raw);
            let direct = match normalized.as_str() {
                "master" => Some(AudioBus::Master),
                "music" => Some(AudioBus::Music),
                "sfx" => Some(AudioBus::Sfx),
                "ui" => Some(AudioBus::Ui),
                "ambience" | "ambient" => Some(AudioBus::Ambience),
                "world" => Some(AudioBus::World),
                _ => None,
            };
            if direct.is_some() {
                return direct;
            }
            if let Some(custom) = normalized.strip_prefix("custom") {
                if let Ok(id) = custom.parse::<u32>() {
                    if id > 0 {
                        return Some(AudioBus::Custom(id));
                    }
                }
            }
            if let Some(audio) = backend {
                let candidate = normalize_name(&raw);
                for bus in audio.0.bus_list() {
                    let name = audio.0.bus_name(bus);
                    if normalize_name(&name) == candidate {
                        return Some(bus);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn audio_playback_state_name(state: AudioPlaybackState) -> &'static str {
    match state {
        AudioPlaybackState::Playing => "Playing",
        AudioPlaybackState::Paused => "Paused",
        AudioPlaybackState::Stopped => "Stopped",
    }
}

fn audio_playback_state_from_value(value: Value) -> Option<AudioPlaybackState> {
    match value {
        Value::Integer(value) => match value {
            0 => Some(AudioPlaybackState::Playing),
            1 => Some(AudioPlaybackState::Paused),
            2 => Some(AudioPlaybackState::Stopped),
            _ => None,
        },
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "playing" | "play" => Some(AudioPlaybackState::Playing),
                "paused" | "pause" => Some(AudioPlaybackState::Paused),
                "stopped" | "stop" => Some(AudioPlaybackState::Stopped),
                _ => None,
            }
        }
        _ => None,
    }
}

fn apply_audio_emitter_asset(
    world: &mut World,
    project: Option<&EditorProject>,
    emitter: &mut AudioEmitter,
    editor_audio: &EditorAudio,
) -> bool {
    let Some(path) = editor_audio.path.as_ref() else {
        emitter.clip_id = None;
        return true;
    };

    let resolved = resolve_project_path(project, Path::new(path));
    let mut loaded_id = None;
    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let Some(asset_server) = world.get_resource::<BevyAssetServer>() else {
            return;
        };
        let handle =
            cached_audio_handle(&mut cache, asset_server, &resolved, editor_audio.streaming);
        loaded_id = Some(handle.id);
    });
    emitter.clip_id = loaded_id;
    loaded_id.is_some()
}

fn combine_rule_name(rule: PhysicsCombineRule) -> &'static str {
    match rule {
        PhysicsCombineRule::Average => "Average",
        PhysicsCombineRule::Min => "Min",
        PhysicsCombineRule::Multiply => "Multiply",
        PhysicsCombineRule::Max => "Max",
    }
}

fn combine_rule_from_value(value: Value) -> Option<PhysicsCombineRule> {
    match value {
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "average" => Some(PhysicsCombineRule::Average),
                "min" | "minimum" => Some(PhysicsCombineRule::Min),
                "multiply" | "mul" => Some(PhysicsCombineRule::Multiply),
                "max" | "maximum" => Some(PhysicsCombineRule::Max),
                _ => None,
            }
        }
        Value::Integer(value) => match value {
            0 => Some(PhysicsCombineRule::Average),
            1 => Some(PhysicsCombineRule::Min),
            2 => Some(PhysicsCombineRule::Multiply),
            3 => Some(PhysicsCombineRule::Max),
            _ => None,
        },
        _ => None,
    }
}

fn kinematic_mode_name(mode: KinematicMode) -> &'static str {
    match mode {
        KinematicMode::PositionBased => "PositionBased",
        KinematicMode::VelocityBased => "VelocityBased",
    }
}

fn kinematic_mode_from_value(value: Value) -> Option<KinematicMode> {
    match value {
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "positionbased" | "position" => Some(KinematicMode::PositionBased),
                "velocitybased" | "velocity" => Some(KinematicMode::VelocityBased),
                _ => None,
            }
        }
        Value::Integer(value) => match value {
            0 => Some(KinematicMode::PositionBased),
            1 => Some(KinematicMode::VelocityBased),
            _ => None,
        },
        _ => None,
    }
}

fn joint_kind_name(kind: PhysicsJointKind) -> &'static str {
    match kind {
        PhysicsJointKind::Fixed => "Fixed",
        PhysicsJointKind::Spherical => "Spherical",
        PhysicsJointKind::Revolute => "Revolute",
        PhysicsJointKind::Prismatic => "Prismatic",
    }
}

fn joint_kind_from_value(value: Value) -> Option<PhysicsJointKind> {
    match value {
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "fixed" => Some(PhysicsJointKind::Fixed),
                "spherical" => Some(PhysicsJointKind::Spherical),
                "revolute" => Some(PhysicsJointKind::Revolute),
                "prismatic" => Some(PhysicsJointKind::Prismatic),
                _ => None,
            }
        }
        Value::Integer(value) => match value {
            0 => Some(PhysicsJointKind::Fixed),
            1 => Some(PhysicsJointKind::Spherical),
            2 => Some(PhysicsJointKind::Revolute),
            3 => Some(PhysicsJointKind::Prismatic),
            _ => None,
        },
        _ => None,
    }
}

fn mesh_collider_lod_to_lua(lua: &Lua, lod: MeshColliderLod) -> mlua::Result<Value> {
    match lod {
        MeshColliderLod::Lod0 => Ok(Value::String(lua.create_string("Lod0")?)),
        MeshColliderLod::Lod1 => Ok(Value::String(lua.create_string("Lod1")?)),
        MeshColliderLod::Lod2 => Ok(Value::String(lua.create_string("Lod2")?)),
        MeshColliderLod::Lowest => Ok(Value::String(lua.create_string("Lowest")?)),
        MeshColliderLod::Specific(index) => Ok(Value::Integer(index as i64)),
    }
}

fn mesh_collider_lod_from_value(value: Value) -> Option<MeshColliderLod> {
    match value {
        Value::Integer(index) => {
            if index >= 0 && index <= u8::MAX as i64 {
                Some(MeshColliderLod::Specific(index as u8))
            } else {
                None
            }
        }
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "lod0" => Some(MeshColliderLod::Lod0),
                "lod1" => Some(MeshColliderLod::Lod1),
                "lod2" => Some(MeshColliderLod::Lod2),
                "lowest" => Some(MeshColliderLod::Lowest),
                _ => {
                    if let Some(rest) = normalized.strip_prefix("lod") {
                        if let Ok(index) = rest.parse::<u8>() {
                            return Some(MeshColliderLod::Specific(index));
                        }
                    }
                    None
                }
            }
        }
        _ => None,
    }
}

fn mesh_collider_kind_name(kind: MeshColliderKind) -> &'static str {
    match kind {
        MeshColliderKind::TriMesh => "TriMesh",
        MeshColliderKind::ConvexHull => "ConvexHull",
    }
}

fn mesh_collider_kind_from_value(value: Value) -> Option<MeshColliderKind> {
    match value {
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "trimesh" | "mesh" => Some(MeshColliderKind::TriMesh),
                "convexhull" | "convex" => Some(MeshColliderKind::ConvexHull),
                _ => None,
            }
        }
        Value::Integer(value) => match value {
            0 => Some(MeshColliderKind::TriMesh),
            1 => Some(MeshColliderKind::ConvexHull),
            _ => None,
        },
        _ => None,
    }
}

fn shape_cast_status_name(status: PhysicsShapeCastStatus) -> &'static str {
    match status {
        PhysicsShapeCastStatus::NoHit => "NoHit",
        PhysicsShapeCastStatus::Converged => "Converged",
        PhysicsShapeCastStatus::OutOfIterations => "OutOfIterations",
        PhysicsShapeCastStatus::Failed => "Failed",
        PhysicsShapeCastStatus::PenetratingOrWithinTargetDist => "PenetratingOrWithinTargetDist",
    }
}

fn lookup_editor_entity_ref(world: &World, entity_id: u64) -> Option<Entity> {
    let entity = Entity::from_bits(entity_id);
    if world.get_entity(entity).is_err() {
        return None;
    }
    if world.get::<EditorEntity>(entity).is_none() {
        return None;
    }
    Some(entity)
}

fn physics_body_kind_from_world(world: &World, entity: Entity) -> Option<PhysicsBodyKind> {
    if let Some(body) = world.get::<DynamicRigidBody>(entity) {
        return Some(PhysicsBodyKind::Dynamic { mass: body.mass });
    }
    if let Some(body) = world.get::<KinematicRigidBody>(entity) {
        return Some(PhysicsBodyKind::Kinematic { mode: body.mode });
    }
    if world.get::<FixedCollider>(entity).is_some() {
        return Some(PhysicsBodyKind::Fixed);
    }
    None
}

fn physics_body_kind_to_table(lua: &Lua, kind: PhysicsBodyKind) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    match kind {
        PhysicsBodyKind::Dynamic { mass } => {
            table.set("type", "Dynamic")?;
            table.set("mass", mass)?;
        }
        PhysicsBodyKind::Kinematic { mode } => {
            table.set("type", "Kinematic")?;
            table.set("mode", kinematic_mode_name(mode))?;
        }
        PhysicsBodyKind::Fixed => {
            table.set("type", "Fixed")?;
        }
    }
    Ok(table)
}

fn physics_body_kind_from_value(value: Value) -> Option<PhysicsBodyKind> {
    match value {
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "dynamic" => Some(PhysicsBodyKind::Dynamic { mass: 1.0 }),
                "kinematic" => Some(PhysicsBodyKind::Kinematic {
                    mode: KinematicMode::default(),
                }),
                "fixed" => Some(PhysicsBodyKind::Fixed),
                _ => None,
            }
        }
        Value::Table(table) => {
            let kind = table
                .get::<String>("type")
                .ok()
                .or_else(|| table.get::<String>("kind").ok())
                .map(|kind| normalize_name(&kind));
            match kind.as_deref() {
                Some("dynamic") => {
                    let mass = table.get::<f32>("mass").unwrap_or(1.0);
                    Some(PhysicsBodyKind::Dynamic {
                        mass: mass.max(0.0),
                    })
                }
                Some("kinematic") => {
                    let mode = table
                        .get::<Value>("mode")
                        .ok()
                        .and_then(kinematic_mode_from_value)
                        .unwrap_or_default();
                    Some(PhysicsBodyKind::Kinematic { mode })
                }
                Some("fixed") => Some(PhysicsBodyKind::Fixed),
                _ => None,
            }
        }
        _ => None,
    }
}

fn collider_shape_to_table(lua: &Lua, shape: ColliderShape) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    match shape {
        ColliderShape::Cuboid => {
            table.set("type", "Cuboid")?;
        }
        ColliderShape::Sphere => {
            table.set("type", "Sphere")?;
        }
        ColliderShape::CapsuleY => {
            table.set("type", "CapsuleY")?;
        }
        ColliderShape::CylinderY => {
            table.set("type", "CylinderY")?;
        }
        ColliderShape::ConeY => {
            table.set("type", "ConeY")?;
        }
        ColliderShape::RoundCuboid { border_radius } => {
            table.set("type", "RoundCuboid")?;
            table.set("border_radius", border_radius)?;
        }
        ColliderShape::Mesh { mesh_id, lod, kind } => {
            table.set("type", "Mesh")?;
            if let Some(mesh_id) = mesh_id {
                table.set("mesh_id", mesh_id as u64)?;
            } else {
                table.set("mesh_id", Value::Nil)?;
            }
            table.set("lod", mesh_collider_lod_to_lua(lua, lod)?)?;
            table.set("kind", mesh_collider_kind_name(kind))?;
        }
    }
    Ok(table)
}

fn collider_shape_from_value(
    value: Value,
    current: Option<ColliderShape>,
) -> Option<ColliderShape> {
    match value {
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "cuboid" | "box" => Some(ColliderShape::Cuboid),
                "sphere" | "ball" => Some(ColliderShape::Sphere),
                "capsuley" | "capsule" => Some(ColliderShape::CapsuleY),
                "cylindery" | "cylinder" => Some(ColliderShape::CylinderY),
                "coney" | "cone" => Some(ColliderShape::ConeY),
                "roundcuboid" => Some(ColliderShape::RoundCuboid {
                    border_radius: 0.05,
                }),
                "mesh" => Some(ColliderShape::Mesh {
                    mesh_id: None,
                    lod: MeshColliderLod::Lod0,
                    kind: MeshColliderKind::TriMesh,
                }),
                _ => None,
            }
        }
        Value::Table(table) => {
            let kind = table
                .get::<String>("type")
                .ok()
                .or_else(|| table.get::<String>("shape").ok())
                .map(|value| normalize_name(&value));
            match kind.as_deref() {
                Some("cuboid") | Some("box") => Some(ColliderShape::Cuboid),
                Some("sphere") | Some("ball") => Some(ColliderShape::Sphere),
                Some("capsuley") | Some("capsule") => Some(ColliderShape::CapsuleY),
                Some("cylindery") | Some("cylinder") => Some(ColliderShape::CylinderY),
                Some("coney") | Some("cone") => Some(ColliderShape::ConeY),
                Some("roundcuboid") => {
                    let current_radius = match current {
                        Some(ColliderShape::RoundCuboid { border_radius }) => border_radius,
                        _ => 0.05,
                    };
                    let border_radius = table.get::<f32>("border_radius").unwrap_or(current_radius);
                    Some(ColliderShape::RoundCuboid {
                        border_radius: border_radius.max(0.0),
                    })
                }
                Some("mesh") => {
                    let (mesh_id, lod, mesh_kind) = match current {
                        Some(ColliderShape::Mesh { mesh_id, lod, kind }) => (mesh_id, lod, kind),
                        _ => (None, MeshColliderLod::Lod0, MeshColliderKind::TriMesh),
                    };
                    let parsed_mesh_id = match table.get::<Value>("mesh_id") {
                        Ok(Value::Integer(id)) => {
                            if id >= 0 {
                                Some(id as usize)
                            } else {
                                None
                            }
                        }
                        Ok(Value::Number(id)) => {
                            if id >= 0.0 {
                                Some(id as usize)
                            } else {
                                None
                            }
                        }
                        Ok(Value::Nil) => None,
                        _ => mesh_id,
                    };
                    let parsed_lod = table
                        .get::<Value>("lod")
                        .ok()
                        .and_then(mesh_collider_lod_from_value)
                        .unwrap_or(lod);
                    let parsed_kind = table
                        .get::<Value>("kind")
                        .ok()
                        .and_then(mesh_collider_kind_from_value)
                        .unwrap_or(mesh_kind);
                    Some(ColliderShape::Mesh {
                        mesh_id: parsed_mesh_id,
                        lod: parsed_lod,
                        kind: parsed_kind,
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn query_filter_to_table(lua: &Lua, filter: PhysicsQueryFilter) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("flags", filter.flags)?;
    table.set("groups_memberships", filter.groups_memberships)?;
    table.set("groups_filter", filter.groups_filter)?;
    table.set("use_groups", filter.use_groups)?;
    table.set("exclude_fixed_flag", PhysicsQueryFilter::EXCLUDE_FIXED)?;
    table.set(
        "exclude_kinematic_flag",
        PhysicsQueryFilter::EXCLUDE_KINEMATIC,
    )?;
    table.set("exclude_dynamic_flag", PhysicsQueryFilter::EXCLUDE_DYNAMIC)?;
    table.set("exclude_sensors_flag", PhysicsQueryFilter::EXCLUDE_SENSORS)?;
    table.set("exclude_solids_flag", PhysicsQueryFilter::EXCLUDE_SOLIDS)?;
    Ok(table)
}

fn patch_query_filter(filter: &mut PhysicsQueryFilter, data: &Table) {
    if let Ok(flags) = data.get::<u32>("flags") {
        filter.flags = flags;
    }
    if let Ok(value) = data.get::<u32>("groups_memberships") {
        filter.groups_memberships = value;
    }
    if let Ok(value) = data.get::<u32>("groups_filter") {
        filter.groups_filter = value;
    }
    if let Ok(value) = data.get::<bool>("use_groups") {
        filter.use_groups = value;
    }
}

fn collider_properties_to_table(lua: &Lua, props: ColliderProperties) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("friction", props.friction)?;
    table.set("restitution", props.restitution)?;
    table.set("density", props.density)?;
    table.set("is_sensor", props.is_sensor)?;
    table.set("enabled", props.enabled)?;
    table.set("collision_memberships", props.collision_memberships)?;
    table.set("collision_filter", props.collision_filter)?;
    table.set("solver_memberships", props.solver_memberships)?;
    table.set("solver_filter", props.solver_filter)?;
    table.set(
        "friction_combine_rule",
        combine_rule_name(props.friction_combine_rule),
    )?;
    table.set(
        "restitution_combine_rule",
        combine_rule_name(props.restitution_combine_rule),
    )?;
    table.set(
        "translation_offset",
        vec3_to_table(lua, props.translation_offset)?,
    )?;
    table.set(
        "rotation_offset",
        quat_to_table(lua, props.rotation_offset)?,
    )?;
    Ok(table)
}

fn patch_collider_properties(props: &mut ColliderProperties, data: &Table) {
    if let Ok(value) = data.get::<f32>("friction") {
        props.friction = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("restitution") {
        props.restitution = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("density") {
        props.density = value.max(0.0);
    }
    if let Ok(value) = data.get::<bool>("is_sensor") {
        props.is_sensor = value;
    }
    if let Ok(value) = data.get::<bool>("enabled") {
        props.enabled = value;
    }
    if let Ok(value) = data.get::<u32>("collision_memberships") {
        props.collision_memberships = value;
    }
    if let Ok(value) = data.get::<u32>("collision_filter") {
        props.collision_filter = value;
    }
    if let Ok(value) = data.get::<u32>("solver_memberships") {
        props.solver_memberships = value;
    }
    if let Ok(value) = data.get::<u32>("solver_filter") {
        props.solver_filter = value;
    }
    if let Ok(value) = data.get::<Value>("friction_combine_rule") {
        if let Some(rule) = combine_rule_from_value(value) {
            props.friction_combine_rule = rule;
        }
    }
    if let Ok(value) = data.get::<Value>("restitution_combine_rule") {
        if let Some(rule) = combine_rule_from_value(value) {
            props.restitution_combine_rule = rule;
        }
    }
    if let Ok(value) = data.get::<Table>("translation_offset") {
        if let Some(offset) = table_to_vec3(&value) {
            props.translation_offset = offset;
        }
    }
    if let Ok(value) = data.get::<Table>("rotation_offset") {
        if let Some(offset) = table_to_quat(&value) {
            props.rotation_offset = offset;
        }
    }
}

fn collider_inheritance_to_table(
    lua: &Lua,
    inheritance: ColliderPropertyInheritance,
) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("friction", inheritance.friction)?;
    table.set("restitution", inheritance.restitution)?;
    table.set("density", inheritance.density)?;
    table.set("is_sensor", inheritance.is_sensor)?;
    table.set("enabled", inheritance.enabled)?;
    table.set("collision_memberships", inheritance.collision_memberships)?;
    table.set("collision_filter", inheritance.collision_filter)?;
    table.set("solver_memberships", inheritance.solver_memberships)?;
    table.set("solver_filter", inheritance.solver_filter)?;
    table.set("friction_combine_rule", inheritance.friction_combine_rule)?;
    table.set(
        "restitution_combine_rule",
        inheritance.restitution_combine_rule,
    )?;
    table.set("translation_offset", inheritance.translation_offset)?;
    table.set("rotation_offset", inheritance.rotation_offset)?;
    Ok(table)
}

fn patch_collider_inheritance(inheritance: &mut ColliderPropertyInheritance, data: &Table) {
    if let Ok(value) = data.get::<bool>("friction") {
        inheritance.friction = value;
    }
    if let Ok(value) = data.get::<bool>("restitution") {
        inheritance.restitution = value;
    }
    if let Ok(value) = data.get::<bool>("density") {
        inheritance.density = value;
    }
    if let Ok(value) = data.get::<bool>("is_sensor") {
        inheritance.is_sensor = value;
    }
    if let Ok(value) = data.get::<bool>("enabled") {
        inheritance.enabled = value;
    }
    if let Ok(value) = data.get::<bool>("collision_memberships") {
        inheritance.collision_memberships = value;
    }
    if let Ok(value) = data.get::<bool>("collision_filter") {
        inheritance.collision_filter = value;
    }
    if let Ok(value) = data.get::<bool>("solver_memberships") {
        inheritance.solver_memberships = value;
    }
    if let Ok(value) = data.get::<bool>("solver_filter") {
        inheritance.solver_filter = value;
    }
    if let Ok(value) = data.get::<bool>("friction_combine_rule") {
        inheritance.friction_combine_rule = value;
    }
    if let Ok(value) = data.get::<bool>("restitution_combine_rule") {
        inheritance.restitution_combine_rule = value;
    }
    if let Ok(value) = data.get::<bool>("translation_offset") {
        inheritance.translation_offset = value;
    }
    if let Ok(value) = data.get::<bool>("rotation_offset") {
        inheritance.rotation_offset = value;
    }
}

fn rigid_body_properties_to_table(lua: &Lua, props: RigidBodyProperties) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("linear_damping", props.linear_damping)?;
    table.set("angular_damping", props.angular_damping)?;
    table.set("gravity_scale", props.gravity_scale)?;
    table.set("ccd_enabled", props.ccd_enabled)?;
    table.set("can_sleep", props.can_sleep)?;
    table.set("sleeping", props.sleeping)?;
    table.set("dominance_group", props.dominance_group as i64)?;
    table.set("lock_translation_x", props.lock_translation_x)?;
    table.set("lock_translation_y", props.lock_translation_y)?;
    table.set("lock_translation_z", props.lock_translation_z)?;
    table.set("lock_rotation_x", props.lock_rotation_x)?;
    table.set("lock_rotation_y", props.lock_rotation_y)?;
    table.set("lock_rotation_z", props.lock_rotation_z)?;
    table.set(
        "linear_velocity",
        vec3_to_table(lua, props.linear_velocity)?,
    )?;
    table.set(
        "angular_velocity",
        vec3_to_table(lua, props.angular_velocity)?,
    )?;
    Ok(table)
}

fn patch_rigid_body_properties(props: &mut RigidBodyProperties, data: &Table) {
    if let Ok(value) = data.get::<f32>("linear_damping") {
        props.linear_damping = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("angular_damping") {
        props.angular_damping = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("gravity_scale") {
        props.gravity_scale = value;
    }
    if let Ok(value) = data.get::<bool>("ccd_enabled") {
        props.ccd_enabled = value;
    }
    if let Ok(value) = data.get::<bool>("can_sleep") {
        props.can_sleep = value;
    }
    if let Ok(value) = data.get::<bool>("sleeping") {
        props.sleeping = value;
    }
    if let Ok(value) = data.get::<i64>("dominance_group") {
        props.dominance_group = value.clamp(i8::MIN as i64, i8::MAX as i64) as i8;
    }
    if let Ok(value) = data.get::<bool>("lock_translation_x") {
        props.lock_translation_x = value;
    }
    if let Ok(value) = data.get::<bool>("lock_translation_y") {
        props.lock_translation_y = value;
    }
    if let Ok(value) = data.get::<bool>("lock_translation_z") {
        props.lock_translation_z = value;
    }
    if let Ok(value) = data.get::<bool>("lock_rotation_x") {
        props.lock_rotation_x = value;
    }
    if let Ok(value) = data.get::<bool>("lock_rotation_y") {
        props.lock_rotation_y = value;
    }
    if let Ok(value) = data.get::<bool>("lock_rotation_z") {
        props.lock_rotation_z = value;
    }
    if let Ok(value) = data.get::<Table>("linear_velocity") {
        if let Some(velocity) = table_to_vec3(&value) {
            props.linear_velocity = velocity;
        }
    }
    if let Ok(value) = data.get::<Table>("angular_velocity") {
        if let Some(velocity) = table_to_vec3(&value) {
            props.angular_velocity = velocity;
        }
    }
}

fn rigid_body_inheritance_to_table(
    lua: &Lua,
    inheritance: RigidBodyPropertyInheritance,
) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("linear_damping", inheritance.linear_damping)?;
    table.set("angular_damping", inheritance.angular_damping)?;
    table.set("gravity_scale", inheritance.gravity_scale)?;
    table.set("ccd_enabled", inheritance.ccd_enabled)?;
    table.set("can_sleep", inheritance.can_sleep)?;
    table.set("sleeping", inheritance.sleeping)?;
    table.set("dominance_group", inheritance.dominance_group)?;
    table.set("lock_translation_x", inheritance.lock_translation_x)?;
    table.set("lock_translation_y", inheritance.lock_translation_y)?;
    table.set("lock_translation_z", inheritance.lock_translation_z)?;
    table.set("lock_rotation_x", inheritance.lock_rotation_x)?;
    table.set("lock_rotation_y", inheritance.lock_rotation_y)?;
    table.set("lock_rotation_z", inheritance.lock_rotation_z)?;
    table.set("linear_velocity", inheritance.linear_velocity)?;
    table.set("angular_velocity", inheritance.angular_velocity)?;
    Ok(table)
}

fn patch_rigid_body_inheritance(inheritance: &mut RigidBodyPropertyInheritance, data: &Table) {
    if let Ok(value) = data.get::<bool>("linear_damping") {
        inheritance.linear_damping = value;
    }
    if let Ok(value) = data.get::<bool>("angular_damping") {
        inheritance.angular_damping = value;
    }
    if let Ok(value) = data.get::<bool>("gravity_scale") {
        inheritance.gravity_scale = value;
    }
    if let Ok(value) = data.get::<bool>("ccd_enabled") {
        inheritance.ccd_enabled = value;
    }
    if let Ok(value) = data.get::<bool>("can_sleep") {
        inheritance.can_sleep = value;
    }
    if let Ok(value) = data.get::<bool>("sleeping") {
        inheritance.sleeping = value;
    }
    if let Ok(value) = data.get::<bool>("dominance_group") {
        inheritance.dominance_group = value;
    }
    if let Ok(value) = data.get::<bool>("lock_translation_x") {
        inheritance.lock_translation_x = value;
    }
    if let Ok(value) = data.get::<bool>("lock_translation_y") {
        inheritance.lock_translation_y = value;
    }
    if let Ok(value) = data.get::<bool>("lock_translation_z") {
        inheritance.lock_translation_z = value;
    }
    if let Ok(value) = data.get::<bool>("lock_rotation_x") {
        inheritance.lock_rotation_x = value;
    }
    if let Ok(value) = data.get::<bool>("lock_rotation_y") {
        inheritance.lock_rotation_y = value;
    }
    if let Ok(value) = data.get::<bool>("lock_rotation_z") {
        inheritance.lock_rotation_z = value;
    }
    if let Ok(value) = data.get::<bool>("linear_velocity") {
        inheritance.linear_velocity = value;
    }
    if let Ok(value) = data.get::<bool>("angular_velocity") {
        inheritance.angular_velocity = value;
    }
}

fn joint_to_table(lua: &Lua, joint: PhysicsJoint) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    match joint.target {
        Some(target) => table.set("target", target.to_bits())?,
        None => table.set("target", Value::Nil)?,
    }
    table.set("kind", joint_kind_name(joint.kind))?;
    table.set("contacts_enabled", joint.contacts_enabled)?;
    table.set("local_anchor1", vec3_to_table(lua, joint.local_anchor1)?)?;
    table.set("local_anchor2", vec3_to_table(lua, joint.local_anchor2)?)?;
    table.set("local_axis1", vec3_to_table(lua, joint.local_axis1)?)?;
    table.set("local_axis2", vec3_to_table(lua, joint.local_axis2)?)?;
    table.set("limit_enabled", joint.limit_enabled)?;

    let limits = lua.create_table()?;
    limits.set("min", joint.limits[0])?;
    limits.set("max", joint.limits[1])?;
    limits.set(1, joint.limits[0])?;
    limits.set(2, joint.limits[1])?;
    table.set("limits", limits)?;

    let motor = lua.create_table()?;
    motor.set("enabled", joint.motor.enabled)?;
    motor.set("target_position", joint.motor.target_position)?;
    motor.set("target_velocity", joint.motor.target_velocity)?;
    motor.set("stiffness", joint.motor.stiffness)?;
    motor.set("damping", joint.motor.damping)?;
    motor.set("max_force", joint.motor.max_force)?;
    table.set("motor", motor)?;
    Ok(table)
}

fn patch_joint(joint: &mut PhysicsJoint, data: &Table, world: &World) {
    if let Ok(value) = data.get::<Value>("target") {
        match value {
            Value::Nil => {
                joint.target = None;
            }
            Value::Integer(id) => {
                joint.target = lookup_editor_entity_ref(world, id as u64);
            }
            Value::Number(id) => {
                joint.target = lookup_editor_entity_ref(world, id as u64);
            }
            _ => {}
        }
    }
    if let Ok(value) = data.get::<Value>("kind") {
        if let Some(kind) = joint_kind_from_value(value) {
            joint.kind = kind;
        }
    }
    if let Ok(value) = data.get::<bool>("contacts_enabled") {
        joint.contacts_enabled = value;
    }
    if let Ok(value) = data.get::<Table>("local_anchor1") {
        if let Some(anchor) = table_to_vec3(&value) {
            joint.local_anchor1 = anchor;
        }
    }
    if let Ok(value) = data.get::<Table>("local_anchor2") {
        if let Some(anchor) = table_to_vec3(&value) {
            joint.local_anchor2 = anchor;
        }
    }
    if let Ok(value) = data.get::<Table>("local_axis1") {
        if let Some(axis) = table_to_vec3(&value) {
            joint.local_axis1 = axis;
        }
    }
    if let Ok(value) = data.get::<Table>("local_axis2") {
        if let Some(axis) = table_to_vec3(&value) {
            joint.local_axis2 = axis;
        }
    }
    if let Ok(value) = data.get::<bool>("limit_enabled") {
        joint.limit_enabled = value;
    }
    if let Ok(value) = data.get::<Value>("limits") {
        if let Value::Table(table) = value {
            if let Ok(min) = table.get::<f32>("min") {
                joint.limits[0] = min;
            } else if let Ok(min) = table.get::<f32>(1) {
                joint.limits[0] = min;
            }
            if let Ok(max) = table.get::<f32>("max") {
                joint.limits[1] = max;
            } else if let Ok(max) = table.get::<f32>(2) {
                joint.limits[1] = max;
            }
        }
    }
    if let Ok(motor) = data.get::<Table>("motor") {
        if let Ok(value) = motor.get::<bool>("enabled") {
            joint.motor.enabled = value;
        }
        if let Ok(value) = motor.get::<f32>("target_position") {
            joint.motor.target_position = value;
        }
        if let Ok(value) = motor.get::<f32>("target_velocity") {
            joint.motor.target_velocity = value;
        }
        if let Ok(value) = motor.get::<f32>("stiffness") {
            joint.motor.stiffness = value.max(0.0);
        }
        if let Ok(value) = motor.get::<f32>("damping") {
            joint.motor.damping = value.max(0.0);
        }
        if let Ok(value) = motor.get::<f32>("max_force") {
            joint.motor.max_force = value.max(0.0);
        }
    }
}

fn character_controller_to_table(
    lua: &Lua,
    controller: CharacterController,
) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("up", vec3_to_table(lua, controller.up)?)?;
    table.set("offset", controller.offset)?;
    table.set("slide", controller.slide)?;
    table.set("autostep_max_height", controller.autostep_max_height)?;
    table.set("autostep_min_width", controller.autostep_min_width)?;
    table.set(
        "autostep_include_dynamic_bodies",
        controller.autostep_include_dynamic_bodies,
    )?;
    table.set("max_slope_climb_angle", controller.max_slope_climb_angle)?;
    table.set("min_slope_slide_angle", controller.min_slope_slide_angle)?;
    table.set("snap_to_ground", controller.snap_to_ground)?;
    table.set("normal_nudge_factor", controller.normal_nudge_factor)?;
    table.set(
        "apply_impulses_to_dynamic_bodies",
        controller.apply_impulses_to_dynamic_bodies,
    )?;
    table.set("character_mass", controller.character_mass)?;
    Ok(table)
}

fn patch_character_controller(controller: &mut CharacterController, data: &Table) {
    if let Ok(value) = data.get::<Table>("up") {
        if let Some(up) = table_to_vec3(&value) {
            controller.up = up;
        }
    }
    if let Ok(value) = data.get::<f32>("offset") {
        controller.offset = value.max(0.0);
    }
    if let Ok(value) = data.get::<bool>("slide") {
        controller.slide = value;
    }
    if let Ok(value) = data.get::<f32>("autostep_max_height") {
        controller.autostep_max_height = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("autostep_min_width") {
        controller.autostep_min_width = value.max(0.0);
    }
    if let Ok(value) = data.get::<bool>("autostep_include_dynamic_bodies") {
        controller.autostep_include_dynamic_bodies = value;
    }
    if let Ok(value) = data.get::<f32>("max_slope_climb_angle") {
        controller.max_slope_climb_angle = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("min_slope_slide_angle") {
        controller.min_slope_slide_angle = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("snap_to_ground") {
        controller.snap_to_ground = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("normal_nudge_factor") {
        controller.normal_nudge_factor = value.max(0.0);
    }
    if let Ok(value) = data.get::<bool>("apply_impulses_to_dynamic_bodies") {
        controller.apply_impulses_to_dynamic_bodies = value;
    }
    if let Ok(value) = data.get::<f32>("character_mass") {
        controller.character_mass = value.max(0.0);
    }
}

fn character_input_to_table(lua: &Lua, input: CharacterControllerInput) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set(
        "desired_translation",
        vec3_to_table(lua, input.desired_translation)?,
    )?;
    Ok(table)
}

fn patch_character_input(input: &mut CharacterControllerInput, data: &Table) {
    if let Ok(value) = data.get::<Table>("desired_translation") {
        if let Some(v) = table_to_vec3(&value) {
            input.desired_translation = v;
        }
    }
}

fn character_output_to_table(lua: &Lua, output: CharacterControllerOutput) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set(
        "effective_translation",
        vec3_to_table(lua, output.effective_translation)?,
    )?;
    table.set("grounded", output.grounded)?;
    table.set("sliding_down_slope", output.sliding_down_slope)?;
    table.set("collision_count", output.collision_count)?;
    Ok(table)
}

fn ray_cast_to_table(lua: &Lua, request: PhysicsRayCast) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("origin", vec3_to_table(lua, request.origin)?)?;
    table.set("direction", vec3_to_table(lua, request.direction)?)?;
    table.set("max_toi", request.max_toi)?;
    table.set("solid", request.solid)?;
    table.set("filter", query_filter_to_table(lua, request.filter)?)?;
    table.set("exclude_self", request.exclude_self)?;
    Ok(table)
}

fn patch_ray_cast(request: &mut PhysicsRayCast, data: &Table) {
    if let Ok(value) = data.get::<Table>("origin") {
        if let Some(origin) = table_to_vec3(&value) {
            request.origin = origin;
        }
    }
    if let Ok(value) = data.get::<Table>("direction") {
        if let Some(direction) = table_to_vec3(&value) {
            request.direction = direction;
        }
    }
    if let Ok(value) = data.get::<f32>("max_toi") {
        request.max_toi = value.max(0.0);
    }
    if let Ok(value) = data.get::<bool>("solid") {
        request.solid = value;
    }
    if let Ok(filter) = data.get::<Table>("filter") {
        patch_query_filter(&mut request.filter, &filter);
    }
    if let Ok(value) = data.get::<bool>("exclude_self") {
        request.exclude_self = value;
    }
}

fn ray_cast_hit_to_table(lua: &Lua, hit: PhysicsRayCastHit) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("has_hit", hit.has_hit)?;
    if let Some(entity) = hit.hit_entity {
        table.set("hit_entity", entity.to_bits())?;
    } else {
        table.set("hit_entity", Value::Nil)?;
    }
    table.set("point", vec3_to_table(lua, hit.point)?)?;
    table.set("normal", vec3_to_table(lua, hit.normal)?)?;
    table.set("toi", hit.toi)?;
    Ok(table)
}

fn point_projection_to_table(lua: &Lua, request: PhysicsPointProjection) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("point", vec3_to_table(lua, request.point)?)?;
    table.set("solid", request.solid)?;
    table.set("filter", query_filter_to_table(lua, request.filter)?)?;
    table.set("exclude_self", request.exclude_self)?;
    Ok(table)
}

fn patch_point_projection(request: &mut PhysicsPointProjection, data: &Table) {
    if let Ok(value) = data.get::<Table>("point") {
        if let Some(point) = table_to_vec3(&value) {
            request.point = point;
        }
    }
    if let Ok(value) = data.get::<bool>("solid") {
        request.solid = value;
    }
    if let Ok(filter) = data.get::<Table>("filter") {
        patch_query_filter(&mut request.filter, &filter);
    }
    if let Ok(value) = data.get::<bool>("exclude_self") {
        request.exclude_self = value;
    }
}

fn point_projection_hit_to_table(lua: &Lua, hit: PhysicsPointProjectionHit) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("has_hit", hit.has_hit)?;
    if let Some(entity) = hit.hit_entity {
        table.set("hit_entity", entity.to_bits())?;
    } else {
        table.set("hit_entity", Value::Nil)?;
    }
    table.set("projected_point", vec3_to_table(lua, hit.projected_point)?)?;
    table.set("is_inside", hit.is_inside)?;
    table.set("distance", hit.distance)?;
    Ok(table)
}

fn shape_cast_to_table(lua: &Lua, request: PhysicsShapeCast) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("shape", collider_shape_to_table(lua, request.shape)?)?;
    table.set("scale", vec3_to_table(lua, request.scale)?)?;
    table.set("position", vec3_to_table(lua, request.position)?)?;
    table.set("rotation", quat_to_table(lua, request.rotation)?)?;
    table.set("velocity", vec3_to_table(lua, request.velocity)?)?;
    table.set("max_time_of_impact", request.max_time_of_impact)?;
    table.set("target_distance", request.target_distance)?;
    table.set("stop_at_penetration", request.stop_at_penetration)?;
    table.set(
        "compute_impact_geometry_on_penetration",
        request.compute_impact_geometry_on_penetration,
    )?;
    table.set("filter", query_filter_to_table(lua, request.filter)?)?;
    table.set("exclude_self", request.exclude_self)?;
    Ok(table)
}

fn patch_shape_cast(request: &mut PhysicsShapeCast, data: &Table) {
    if let Ok(value) = data.get::<Value>("shape") {
        if let Some(shape) = collider_shape_from_value(value, Some(request.shape)) {
            request.shape = shape;
        }
    }
    if let Ok(value) = data.get::<Table>("scale") {
        if let Some(scale) = table_to_vec3(&value) {
            request.scale = scale;
        }
    }
    if let Ok(value) = data.get::<Table>("position") {
        if let Some(position) = table_to_vec3(&value) {
            request.position = position;
        }
    }
    if let Ok(value) = data.get::<Table>("rotation") {
        if let Some(rotation) = table_to_quat(&value) {
            request.rotation = rotation;
        }
    }
    if let Ok(value) = data.get::<Table>("velocity") {
        if let Some(velocity) = table_to_vec3(&value) {
            request.velocity = velocity;
        }
    }
    if let Ok(value) = data.get::<f32>("max_time_of_impact") {
        request.max_time_of_impact = value.max(0.0);
    }
    if let Ok(value) = data.get::<f32>("target_distance") {
        request.target_distance = value.max(0.0);
    }
    if let Ok(value) = data.get::<bool>("stop_at_penetration") {
        request.stop_at_penetration = value;
    }
    if let Ok(value) = data.get::<bool>("compute_impact_geometry_on_penetration") {
        request.compute_impact_geometry_on_penetration = value;
    }
    if let Ok(filter) = data.get::<Table>("filter") {
        patch_query_filter(&mut request.filter, &filter);
    }
    if let Ok(value) = data.get::<bool>("exclude_self") {
        request.exclude_self = value;
    }
}

fn shape_cast_hit_to_table(lua: &Lua, hit: PhysicsShapeCastHit) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("has_hit", hit.has_hit)?;
    if let Some(entity) = hit.hit_entity {
        table.set("hit_entity", entity.to_bits())?;
    } else {
        table.set("hit_entity", Value::Nil)?;
    }
    table.set("toi", hit.toi)?;
    table.set("witness1", vec3_to_table(lua, hit.witness1)?)?;
    table.set("witness2", vec3_to_table(lua, hit.witness2)?)?;
    table.set("normal1", vec3_to_table(lua, hit.normal1)?)?;
    table.set("normal2", vec3_to_table(lua, hit.normal2)?)?;
    table.set("status", shape_cast_status_name(hit.status))?;
    Ok(table)
}

fn physics_world_defaults_to_table(
    lua: &Lua,
    defaults: PhysicsWorldDefaults,
) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("gravity", vec3_to_table(lua, defaults.gravity)?)?;
    table.set(
        "collider_properties",
        collider_properties_to_table(lua, defaults.collider_properties)?,
    )?;
    table.set(
        "rigid_body_properties",
        rigid_body_properties_to_table(lua, defaults.rigid_body_properties)?,
    )?;
    Ok(table)
}

fn patch_physics_world_defaults(defaults: &mut PhysicsWorldDefaults, data: &Table) {
    if let Ok(value) = data.get::<Table>("gravity") {
        if let Some(gravity) = table_to_vec3(&value) {
            defaults.gravity = gravity;
        }
    }
    if let Ok(value) = data.get::<Table>("collider_properties") {
        let mut props = defaults.collider_properties;
        patch_collider_properties(&mut props, &value);
        defaults.collider_properties = props;
    }
    if let Ok(value) = data.get::<Table>("rigid_body_properties") {
        let mut props = defaults.rigid_body_properties;
        patch_rigid_body_properties(&mut props, &value);
        defaults.rigid_body_properties = props;
    }
}

fn physics_entity_to_table(lua: &Lua, world: &World, entity: Entity) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    if let Some(shape) = world.get::<ColliderShape>(entity).copied() {
        table.set("collider_shape", collider_shape_to_table(lua, shape)?)?;
    }
    if let Some(body_kind) = physics_body_kind_from_world(world, entity) {
        table.set("body_kind", physics_body_kind_to_table(lua, body_kind)?)?;
    }
    if let Some(props) = world.get::<ColliderProperties>(entity).copied() {
        table.set(
            "collider_properties",
            collider_properties_to_table(lua, props)?,
        )?;
    }
    if let Some(value) = world.get::<ColliderPropertyInheritance>(entity).copied() {
        table.set(
            "collider_inheritance",
            collider_inheritance_to_table(lua, value)?,
        )?;
    }
    if let Some(props) = world.get::<RigidBodyProperties>(entity).copied() {
        table.set(
            "rigid_body_properties",
            rigid_body_properties_to_table(lua, props)?,
        )?;
    }
    if let Some(value) = world.get::<RigidBodyPropertyInheritance>(entity).copied() {
        table.set(
            "rigid_body_inheritance",
            rigid_body_inheritance_to_table(lua, value)?,
        )?;
    }
    if let Some(joint) = world.get::<PhysicsJoint>(entity).copied() {
        table.set("joint", joint_to_table(lua, joint)?)?;
    }
    if let Some(controller) = world.get::<CharacterController>(entity).copied() {
        table.set(
            "character_controller",
            character_controller_to_table(lua, controller)?,
        )?;
    }
    if let Some(input) = world.get::<CharacterControllerInput>(entity).copied() {
        table.set("character_input", character_input_to_table(lua, input)?)?;
    }
    if let Some(output) = world.get::<CharacterControllerOutput>(entity).copied() {
        table.set("character_output", character_output_to_table(lua, output)?)?;
    }
    if let Some(request) = world.get::<PhysicsRayCast>(entity).copied() {
        table.set("ray_cast", ray_cast_to_table(lua, request)?)?;
    }
    if let Some(hit) = world.get::<PhysicsRayCastHit>(entity).copied() {
        table.set("ray_cast_hit", ray_cast_hit_to_table(lua, hit)?)?;
    }
    if let Some(request) = world.get::<PhysicsPointProjection>(entity).copied() {
        table.set("point_projection", point_projection_to_table(lua, request)?)?;
    }
    if let Some(hit) = world.get::<PhysicsPointProjectionHit>(entity).copied() {
        table.set(
            "point_projection_hit",
            point_projection_hit_to_table(lua, hit)?,
        )?;
    }
    if let Some(request) = world.get::<PhysicsShapeCast>(entity).copied() {
        table.set("shape_cast", shape_cast_to_table(lua, request)?)?;
    }
    if let Some(hit) = world.get::<PhysicsShapeCastHit>(entity).copied() {
        table.set("shape_cast_hit", shape_cast_hit_to_table(lua, hit)?)?;
    }
    if let Some(defaults) = world.get::<PhysicsWorldDefaults>(entity).copied() {
        table.set(
            "world_defaults",
            physics_world_defaults_to_table(lua, defaults)?,
        )?;
    }
    table.set("has_handle", world.get::<PhysicsHandle>(entity).is_some())?;
    Ok(table)
}

fn apply_physics_patch(world: &mut World, entity: Entity, data: &Table) {
    if let Ok(value) = data.get::<Value>("collider_shape") {
        if !matches!(value, Value::Nil) {
            let current = world.get::<ColliderShape>(entity).copied();
            if let Some(shape) = collider_shape_from_value(value, current) {
                world.entity_mut(entity).insert(shape);
                world.entity_mut(entity).remove::<PhysicsHandle>();
            }
        }
    }
    if let Ok(value) = data.get::<Value>("body_kind") {
        if let Some(kind) = physics_body_kind_from_value(value) {
            set_physics_body_kind(world, entity, kind);
        }
    }
    if let Ok(value) = data.get::<Table>("collider_properties") {
        let mut props = world
            .get::<ColliderProperties>(entity)
            .copied()
            .unwrap_or_default();
        patch_collider_properties(&mut props, &value);
        world.entity_mut(entity).insert(props);
    }
    if let Ok(value) = data.get::<Table>("collider_inheritance") {
        let mut inheritance = world
            .get::<ColliderPropertyInheritance>(entity)
            .copied()
            .unwrap_or_default();
        patch_collider_inheritance(&mut inheritance, &value);
        world.entity_mut(entity).insert(inheritance);
    }
    if let Ok(value) = data.get::<Table>("rigid_body_properties") {
        let mut props = world
            .get::<RigidBodyProperties>(entity)
            .copied()
            .unwrap_or_default();
        patch_rigid_body_properties(&mut props, &value);
        world.entity_mut(entity).insert(props);
    }
    if let Ok(value) = data.get::<Table>("rigid_body_inheritance") {
        let mut inheritance = world
            .get::<RigidBodyPropertyInheritance>(entity)
            .copied()
            .unwrap_or_default();
        patch_rigid_body_inheritance(&mut inheritance, &value);
        world.entity_mut(entity).insert(inheritance);
    }
    if let Ok(value) = data.get::<Table>("joint") {
        let mut joint = world
            .get::<PhysicsJoint>(entity)
            .copied()
            .unwrap_or_default();
        patch_joint(&mut joint, &value, world);
        world.entity_mut(entity).insert(joint);
    }
    if let Ok(value) = data.get::<Table>("character_controller") {
        let mut controller = world
            .get::<CharacterController>(entity)
            .copied()
            .unwrap_or_default();
        patch_character_controller(&mut controller, &value);
        world.entity_mut(entity).insert(controller);
        world
            .entity_mut(entity)
            .insert(CharacterControllerOutput::default());
        if world.get::<CharacterControllerInput>(entity).is_none() {
            world
                .entity_mut(entity)
                .insert(CharacterControllerInput::default());
        }
    }
    if let Ok(value) = data.get::<Table>("character_input") {
        let mut input = world
            .get::<CharacterControllerInput>(entity)
            .copied()
            .unwrap_or_default();
        patch_character_input(&mut input, &value);
        world.entity_mut(entity).insert(input);
    }
    if let Ok(value) = data.get::<Table>("ray_cast") {
        let mut request = world
            .get::<PhysicsRayCast>(entity)
            .copied()
            .unwrap_or_default();
        patch_ray_cast(&mut request, &value);
        world.entity_mut(entity).insert(request);
        if world.get::<PhysicsRayCastHit>(entity).is_none() {
            world
                .entity_mut(entity)
                .insert(PhysicsRayCastHit::default());
        }
    }
    if let Ok(value) = data.get::<Table>("point_projection") {
        let mut request = world
            .get::<PhysicsPointProjection>(entity)
            .copied()
            .unwrap_or_default();
        patch_point_projection(&mut request, &value);
        world.entity_mut(entity).insert(request);
        if world.get::<PhysicsPointProjectionHit>(entity).is_none() {
            world
                .entity_mut(entity)
                .insert(PhysicsPointProjectionHit::default());
        }
    }
    if let Ok(value) = data.get::<Table>("shape_cast") {
        let mut request = world
            .get::<PhysicsShapeCast>(entity)
            .copied()
            .unwrap_or_default();
        patch_shape_cast(&mut request, &value);
        world.entity_mut(entity).insert(request);
        if world.get::<PhysicsShapeCastHit>(entity).is_none() {
            world
                .entity_mut(entity)
                .insert(PhysicsShapeCastHit::default());
        }
    }
    if let Ok(value) = data.get::<Table>("world_defaults") {
        let mut defaults = world
            .get::<PhysicsWorldDefaults>(entity)
            .copied()
            .unwrap_or_default();
        patch_physics_world_defaults(&mut defaults, &value);
        world.entity_mut(entity).insert(defaults);
    }
}

fn physics_velocity_to_table(
    lua: &Lua,
    world: &mut World,
    entity: Entity,
) -> mlua::Result<Option<Table>> {
    let mut linear = None;
    let mut angular = None;

    if let Some(handle) = world.get::<PhysicsHandle>(entity).copied() {
        if let Some(phys) = world.get_resource::<PhysicsResource>() {
            if let Some(body) = phys.rigid_body_set.get(handle.rigid_body) {
                let linvel = body.linvel();
                linear = Some(Vec3::new(linvel.x, linvel.y, linvel.z));
                let angvel = body.angvel();
                angular = Some(Vec3::new(angvel.x, angvel.y, angvel.z));
            }
        }
    }

    if (linear.is_none() || angular.is_none()) && world.get::<RigidBodyProperties>(entity).is_some()
    {
        if let Some(props) = world.get::<RigidBodyProperties>(entity) {
            if linear.is_none() {
                linear = Some(props.linear_velocity);
            }
            if angular.is_none() {
                angular = Some(props.angular_velocity);
            }
        }
    }

    if linear.is_none() && angular.is_none() {
        return Ok(None);
    }

    let table = lua.create_table()?;
    if let Some(value) = linear {
        table.set("linear", vec3_to_table(lua, value)?)?;
    }
    if let Some(value) = angular {
        table.set("angular", vec3_to_table(lua, value)?)?;
    }
    Ok(Some(table))
}

fn set_physics_velocity(world: &mut World, entity: Entity, data: &Table) -> bool {
    let linear = data
        .get::<Table>("linear")
        .ok()
        .and_then(|value| table_to_vec3(&value));
    let angular = data
        .get::<Table>("angular")
        .ok()
        .and_then(|value| table_to_vec3(&value));
    let wake_up = data.get::<bool>("wake_up").unwrap_or(true);
    if linear.is_none() && angular.is_none() {
        return false;
    }

    let handle = world.get::<PhysicsHandle>(entity).copied();
    if let Some(handle) = handle {
        if let Some(mut phys) = world.get_resource_mut::<PhysicsResource>() {
            if let Some(body) = phys.rigid_body_set.get_mut(handle.rigid_body) {
                if let Some(linear) = linear {
                    let mut linvel = body.linvel();
                    linvel.x = linear.x;
                    linvel.y = linear.y;
                    linvel.z = linear.z;
                    body.set_linvel(linvel, wake_up);
                }
                if let Some(angular) = angular {
                    let mut angvel = body.angvel();
                    angvel.x = angular.x;
                    angvel.y = angular.y;
                    angvel.z = angular.z;
                    body.set_angvel(angvel, wake_up);
                }
            }
        }
    }

    if let Some(mut props) = world.get_mut::<RigidBodyProperties>(entity) {
        if let Some(linear) = linear {
            props.linear_velocity = linear;
        }
        if let Some(angular) = angular {
            props.angular_velocity = angular;
        }
    } else {
        let mut props = RigidBodyProperties::default();
        if let Some(linear) = linear {
            props.linear_velocity = linear;
        }
        if let Some(angular) = angular {
            props.angular_velocity = angular;
        }
        world.entity_mut(entity).insert(props);
    }

    true
}

#[inline]
fn is_valid_non_zero_vec3(value: Vec3) -> bool {
    value.is_finite() && value.length_squared() > 1.0e-12
}

#[inline]
fn is_valid_vec3(value: Vec3) -> bool {
    value.is_finite()
}

#[inline]
fn push_capped<T>(values: &mut Vec<T>, value: T, max_items: usize) {
    if values.len() + 1 > max_items {
        let overflow = values.len() + 1 - max_items;
        values.drain(0..overflow);
    }
    values.push(value);
}

fn add_transient_force(world: &mut World, entity: Entity, force: Vec3, wake_up: bool) -> bool {
    if !is_valid_non_zero_vec3(force) || !has_physics_component(world, entity) {
        return false;
    }
    let mut forces = world
        .get::<RigidBodyTransientForces>(entity)
        .cloned()
        .unwrap_or_default();
    forces.force += force;
    forces.force_wake_up |= wake_up;
    world.entity_mut(entity).insert(forces);
    true
}

fn add_transient_torque(world: &mut World, entity: Entity, torque: Vec3, wake_up: bool) -> bool {
    if !is_valid_non_zero_vec3(torque) || !has_physics_component(world, entity) {
        return false;
    }
    let mut forces = world
        .get::<RigidBodyTransientForces>(entity)
        .cloned()
        .unwrap_or_default();
    forces.torque += torque;
    forces.torque_wake_up |= wake_up;
    world.entity_mut(entity).insert(forces);
    true
}

fn add_transient_force_at_point(
    world: &mut World,
    entity: Entity,
    force: Vec3,
    point: Vec3,
    wake_up: bool,
) -> bool {
    if !is_valid_non_zero_vec3(force) || !point.is_finite() || !has_physics_component(world, entity)
    {
        return false;
    }

    const MAX_TRANSIENT_POINT_FORCES: usize = 1024;

    let mut forces = world
        .get::<RigidBodyTransientForces>(entity)
        .cloned()
        .unwrap_or_default();
    push_capped(
        &mut forces.point_forces,
        PointForce {
            force,
            point,
            wake_up,
        },
        MAX_TRANSIENT_POINT_FORCES,
    );
    world.entity_mut(entity).insert(forces);
    true
}

fn add_persistent_force(world: &mut World, entity: Entity, force: Vec3, wake_up: bool) -> bool {
    if !is_valid_non_zero_vec3(force) || !has_physics_component(world, entity) {
        return false;
    }
    let mut forces = world
        .get::<RigidBodyForces>(entity)
        .cloned()
        .unwrap_or_default();
    forces.force += force;
    forces.force_wake_up |= wake_up;
    world.entity_mut(entity).insert(forces);
    true
}

fn set_persistent_force(world: &mut World, entity: Entity, force: Vec3, wake_up: bool) -> bool {
    if !is_valid_vec3(force) || !has_physics_component(world, entity) {
        return false;
    }
    let mut forces = world
        .get::<RigidBodyForces>(entity)
        .cloned()
        .unwrap_or_default();
    forces.force = force;
    forces.force_wake_up = force != Vec3::ZERO && wake_up;
    world.entity_mut(entity).insert(forces);
    true
}

fn add_persistent_torque(world: &mut World, entity: Entity, torque: Vec3, wake_up: bool) -> bool {
    if !is_valid_non_zero_vec3(torque) || !has_physics_component(world, entity) {
        return false;
    }
    let mut forces = world
        .get::<RigidBodyForces>(entity)
        .cloned()
        .unwrap_or_default();
    forces.torque += torque;
    forces.torque_wake_up |= wake_up;
    world.entity_mut(entity).insert(forces);
    true
}

fn set_persistent_torque(world: &mut World, entity: Entity, torque: Vec3, wake_up: bool) -> bool {
    if !is_valid_vec3(torque) || !has_physics_component(world, entity) {
        return false;
    }
    let mut forces = world
        .get::<RigidBodyForces>(entity)
        .cloned()
        .unwrap_or_default();
    forces.torque = torque;
    forces.torque_wake_up = torque != Vec3::ZERO && wake_up;
    world.entity_mut(entity).insert(forces);
    true
}

fn add_persistent_force_at_point(
    world: &mut World,
    entity: Entity,
    force: Vec3,
    point: Vec3,
    wake_up: bool,
) -> bool {
    if !is_valid_non_zero_vec3(force) || !point.is_finite() || !has_physics_component(world, entity)
    {
        return false;
    }

    const MAX_PERSISTENT_POINT_FORCES: usize = 1024;

    let mut forces = world
        .get::<RigidBodyForces>(entity)
        .cloned()
        .unwrap_or_default();
    push_capped(
        &mut forces.point_forces,
        PointForce {
            force,
            point,
            wake_up,
        },
        MAX_PERSISTENT_POINT_FORCES,
    );
    world.entity_mut(entity).insert(forces);
    true
}

fn clear_persistent_forces(world: &mut World, entity: Entity) -> bool {
    if !has_physics_component(world, entity) {
        return false;
    }
    world.entity_mut(entity).remove::<RigidBodyForces>();
    true
}

fn queue_physics_impulse(world: &mut World, entity: Entity, impulse: Vec3, wake_up: bool) -> bool {
    if !is_valid_non_zero_vec3(impulse) || !has_physics_component(world, entity) {
        return false;
    }
    let mut queue = world
        .get::<RigidBodyImpulseQueue>(entity)
        .cloned()
        .unwrap_or_default();
    queue.impulse += impulse;
    queue.impulse_wake_up |= wake_up;
    world.entity_mut(entity).insert(queue);
    true
}

fn queue_physics_angular_impulse(
    world: &mut World,
    entity: Entity,
    impulse: Vec3,
    wake_up: bool,
) -> bool {
    if !is_valid_non_zero_vec3(impulse) || !has_physics_component(world, entity) {
        return false;
    }
    let mut queue = world
        .get::<RigidBodyImpulseQueue>(entity)
        .cloned()
        .unwrap_or_default();
    queue.angular_impulse += impulse;
    queue.angular_impulse_wake_up |= wake_up;
    world.entity_mut(entity).insert(queue);
    true
}

fn queue_physics_impulse_at_point(
    world: &mut World,
    entity: Entity,
    impulse: Vec3,
    point: Vec3,
    wake_up: bool,
) -> bool {
    if !is_valid_non_zero_vec3(impulse)
        || !point.is_finite()
        || !has_physics_component(world, entity)
    {
        return false;
    }

    const MAX_POINT_IMPULSES: usize = 1024;

    let mut queue = world
        .get::<RigidBodyImpulseQueue>(entity)
        .cloned()
        .unwrap_or_default();
    push_capped(
        &mut queue.point_impulses,
        PointImpulse {
            impulse,
            point,
            wake_up,
        },
        MAX_POINT_IMPULSES,
    );
    world.entity_mut(entity).insert(queue);
    true
}

fn gameplay_input_capture_allowed(world: &World) -> bool {
    let in_play_mode = world
        .get_resource::<EditorSceneState>()
        .map(|state| state.world_state == WorldState::Play)
        .unwrap_or(false);
    if !in_play_mode {
        return true;
    }

    if let Some(runtime) = world.get_resource::<EditorViewportRuntime>() {
        if !runtime.pane_requests.is_empty() {
            let pointer_over_play_viewport = runtime.pane_requests.iter().any(|pane| {
                pane.pointer_over && world.get::<EditorPlayCamera>(pane.camera_entity).is_some()
            });
            if pointer_over_play_viewport {
                return true;
            }

            if runtime.keyboard_focus {
                if let Some(active_pane) = runtime.active_pane_id.and_then(|pane_id| {
                    runtime
                        .pane_requests
                        .iter()
                        .find(|pane| pane.pane_id == pane_id)
                }) {
                    return world
                        .get::<EditorPlayCamera>(active_pane.camera_entity)
                        .is_some();
                }
            }

            return false;
        }
    }

    world
        .get_resource::<EditorViewportState>()
        .is_some_and(|state| state.play_mode_view == PlayViewportKind::Gameplay)
}

fn build_input_table(lua: &Lua, world_ptr: usize) -> mlua::Result<Table> {
    let input = lua.create_table()?;

    let keys = lua.create_table()?;
    fill_key_constants(lua, &keys)?;
    input.set("keys", keys)?;

    let mouse_buttons = lua.create_table()?;
    fill_mouse_button_constants(lua, &mouse_buttons)?;
    input.set("mouse_buttons", mouse_buttons)?;

    let gamepad_buttons = lua.create_table()?;
    fill_gamepad_button_constants(lua, &gamepad_buttons)?;
    input.set("gamepad_buttons", gamepad_buttons)?;

    let gamepad_axes = lua.create_table()?;
    fill_gamepad_axis_constants(lua, &gamepad_axes)?;
    input.set("gamepad_axes", gamepad_axes)?;

    input.set(
        "key",
        lua.create_function(|lua, name: String| {
            let Some(key) = parse_key_name(&name) else {
                return Ok(None);
            };
            Ok(Some(lua.create_userdata(key)?))
        })?,
    )?;

    input.set(
        "mouse_button",
        lua.create_function(|lua, name: String| {
            let Some(button) = parse_mouse_button_name(&name) else {
                return Ok(None);
            };
            Ok(Some(lua.create_userdata(LuaMouseButton(button))?))
        })?,
    )?;

    input.set(
        "gamepad_button",
        lua.create_function(|lua, name: String| {
            let Some(button) = parse_gamepad_button_name(&name) else {
                return Ok(None);
            };
            Ok(Some(lua.create_userdata(LuaGamepadButton(button))?))
        })?,
    )?;

    let gamepad_axis_handle_fn = lua.create_function(|lua, name: String| {
        let Some(axis) = parse_gamepad_axis_name(&name) else {
            return Ok(None);
        };
        Ok(Some(lua.create_userdata(LuaGamepadAxis(axis))?))
    })?;
    input.set("gamepad_axis_handle", gamepad_axis_handle_fn.clone())?;
    input.set("gamepad_axis_ref", gamepad_axis_handle_fn)?;

    let world_ptr_key = world_ptr;
    input.set(
        "key_down",
        lua.create_function(move |_, key: Value| {
            let world = unsafe { &mut *(world_ptr_key as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(false);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_key {
                return Ok(false);
            }
            let Some(key) = lua_key_from_value(key) else {
                return Ok(false);
            };
            Ok(lua_key_is_active(&input_manager, key))
        })?,
    )?;

    let world_ptr_key_pressed = world_ptr;
    input.set(
        "key_pressed",
        lua.create_function(move |_, key: Value| {
            let world = unsafe { &mut *(world_ptr_key_pressed as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(false);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_key {
                return Ok(false);
            }
            let Some(key) = lua_key_from_value(key) else {
                return Ok(false);
            };
            Ok(lua_key_is_just_pressed(&input_manager, key))
        })?,
    )?;

    let world_ptr_key_released = world_ptr;
    input.set(
        "key_released",
        lua.create_function(move |_, key: Value| {
            let world = unsafe { &mut *(world_ptr_key_released as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(false);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_key {
                return Ok(false);
            }
            let Some(key) = lua_key_from_value(key) else {
                return Ok(false);
            };
            let Some(input_state) = world.get_resource::<ScriptInputState>() else {
                return Ok(false);
            };
            Ok(lua_key_is_released(&input_manager, input_state, key))
        })?,
    )?;

    let world_ptr_mouse = world_ptr;
    input.set(
        "mouse_down",
        lua.create_function(move |_, button: Value| {
            let world = unsafe { &mut *(world_ptr_mouse as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(false);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_pointer {
                return Ok(false);
            }
            let Some(button) = lua_mouse_button_from_value(button) else {
                return Ok(false);
            };
            Ok(input_manager.is_mouse_button_active(button))
        })?,
    )?;

    let world_ptr_mouse_pressed = world_ptr;
    input.set(
        "mouse_pressed",
        lua.create_function(move |_, button: Value| {
            let world = unsafe { &mut *(world_ptr_mouse_pressed as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(false);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_pointer {
                return Ok(false);
            }
            let Some(button) = lua_mouse_button_from_value(button) else {
                return Ok(false);
            };
            let Some(input_state) = world.get_resource::<ScriptInputState>() else {
                return Ok(false);
            };
            let is_down = input_manager.is_mouse_button_active(button);
            let was_down = input_state.last_mouse_buttons.contains(&button);
            Ok(is_down && !was_down)
        })?,
    )?;

    let world_ptr_mouse_released = world_ptr;
    input.set(
        "mouse_released",
        lua.create_function(move |_, button: Value| {
            let world = unsafe { &mut *(world_ptr_mouse_released as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(false);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_pointer {
                return Ok(false);
            }
            let Some(button) = lua_mouse_button_from_value(button) else {
                return Ok(false);
            };
            let Some(input_state) = world.get_resource::<ScriptInputState>() else {
                return Ok(false);
            };
            let is_down = input_manager.is_mouse_button_active(button);
            let was_down = input_state.last_mouse_buttons.contains(&button);
            Ok(!is_down && was_down)
        })?,
    )?;

    let world_ptr_cursor = world_ptr;
    input.set(
        "cursor",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_cursor as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(None);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_pointer {
                return Ok(None);
            }
            let position = Vec2::new(
                input_manager.cursor_position.x as f32,
                input_manager.cursor_position.y as f32,
            );
            Ok(Some(vec2_to_table(lua, position)?))
        })?,
    )?;

    let world_ptr_cursor_delta = world_ptr;
    input.set(
        "cursor_delta",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_cursor_delta as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(None);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_pointer {
                return Ok(None);
            }
            let Some(input_state) = world.get_resource::<ScriptInputState>() else {
                return Ok(None);
            };
            let delta = input_manager.cursor_position - input_state.last_cursor_position;
            let delta = Vec2::new(delta.x as f32, delta.y as f32);
            Ok(Some(vec2_to_table(lua, delta)?))
        })?,
    )?;

    let world_ptr_wheel = world_ptr;
    input.set(
        "wheel",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_wheel as *mut World) };
            if !gameplay_input_capture_allowed(world) {
                return Ok(None);
            }
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_pointer {
                return Ok(None);
            }
            Ok(Some(vec2_to_table(lua, input_manager.mouse_wheel)?))
        })?,
    )?;

    let world_ptr_window = world_ptr;
    input.set(
        "window_size",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_window as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
            let size = Vec2::new(
                input_manager.window_size.x as f32,
                input_manager.window_size.y as f32,
            );
            Ok(Some(vec2_to_table(lua, size)?))
        })?,
    )?;

    let world_ptr_scale = world_ptr;
    input.set(
        "scale_factor",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_scale as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
            Ok(Some(input_manager.scale_factor))
        })?,
    )?;

    let world_ptr_mods = world_ptr;
    input.set(
        "modifiers",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_mods as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
            if input_manager.egui_wants_key || !gameplay_input_capture_allowed(world) {
                let table = lua.create_table()?;
                table.set("shift", false)?;
                table.set("ctrl", false)?;
                table.set("alt", false)?;
                table.set("super", false)?;
                return Ok(Some(table));
            }

            let shift = input_manager.is_key_active(KeyCode::ShiftLeft)
                || input_manager.is_key_active(KeyCode::ShiftRight);
            let ctrl = input_manager.is_key_active(KeyCode::ControlLeft)
                || input_manager.is_key_active(KeyCode::ControlRight);
            let alt = input_manager.is_key_active(KeyCode::AltLeft)
                || input_manager.is_key_active(KeyCode::AltRight);
            let super_key = input_manager.is_key_active(KeyCode::SuperLeft)
                || input_manager.is_key_active(KeyCode::SuperRight);

            let table = lua.create_table()?;
            table.set("shift", shift)?;
            table.set("ctrl", ctrl)?;
            table.set("alt", alt)?;
            table.set("super", super_key)?;
            Ok(Some(table))
        })?,
    )?;

    let world_ptr_wants_keyboard = world_ptr;
    input.set(
        "wants_keyboard",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_wants_keyboard as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            Ok(input_manager.egui_wants_key || !gameplay_input_capture_allowed(world))
        })?,
    )?;

    let world_ptr_wants_pointer = world_ptr;
    input.set(
        "wants_pointer",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_wants_pointer as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            Ok(input_manager.egui_wants_pointer || !gameplay_input_capture_allowed(world))
        })?,
    )?;

    let world_ptr_cursor_grab_mode = world_ptr;
    input.set(
        "cursor_grab_mode",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_cursor_grab_mode as *mut World) };
            let Some(cursor_control) = world.get_resource::<EditorCursorControlState>() else {
                return Ok(RuntimeCursorGrabMode::None.as_str().to_string());
            };
            Ok(cursor_control
                .effective_policy()
                .grab_mode
                .as_str()
                .to_string())
        })?,
    )?;

    let world_ptr_set_cursor_visible = world_ptr;
    input.set(
        "set_cursor_visible",
        lua.create_function(move |_, visible: bool| {
            let world = unsafe { &mut *(world_ptr_set_cursor_visible as *mut World) };
            let Some(mut cursor_control) = world.get_resource_mut::<EditorCursorControlState>()
            else {
                return Ok(false);
            };
            let mut policy = cursor_control.script_policy.unwrap_or_default();
            policy.visible = visible;
            cursor_control.script_policy = Some(policy);
            Ok(true)
        })?,
    )?;

    let world_ptr_set_cursor_grab = world_ptr;
    input.set(
        "set_cursor_grab",
        lua.create_function(move |_, mode: Value| {
            let world = unsafe { &mut *(world_ptr_set_cursor_grab as *mut World) };
            let Some(grab_mode) = parse_cursor_grab_mode(mode) else {
                return Ok(false);
            };
            let Some(mut cursor_control) = world.get_resource_mut::<EditorCursorControlState>()
            else {
                return Ok(false);
            };
            let mut policy = cursor_control.script_policy.unwrap_or_default();
            policy.grab_mode = grab_mode;
            cursor_control.script_policy = Some(policy);
            Ok(true)
        })?,
    )?;

    let world_ptr_reset_cursor_control = world_ptr;
    input.set(
        "reset_cursor_control",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_reset_cursor_control as *mut World) };
            let Some(mut cursor_control) = world.get_resource_mut::<EditorCursorControlState>()
            else {
                return Ok(false);
            };
            cursor_control.script_policy = None;
            Ok(true)
        })?,
    )?;

    let world_ptr_gamepad_ids = world_ptr;
    input.set(
        "gamepad_ids",
        lua.create_function(move |lua, ()| {
            let world = unsafe { &mut *(world_ptr_gamepad_ids as *mut World) };
            let Some(input_state) = world.get_resource::<ScriptInputState>() else {
                return Ok(lua.create_table()?);
            };
            let list = lua.create_table()?;
            for (index, id) in input_state.gamepad_ids.iter().enumerate() {
                list.set(index + 1, *id as u64)?;
            }
            Ok(list)
        })?,
    )?;

    let world_ptr_gamepad_count = world_ptr;
    input.set(
        "gamepad_count",
        lua.create_function(move |_, ()| {
            let world = unsafe { &mut *(world_ptr_gamepad_count as *mut World) };
            let Some(input_state) = world.get_resource::<ScriptInputState>() else {
                return Ok(0_u32);
            };
            Ok(input_state.gamepad_ids.len() as u32)
        })?,
    )?;

    let world_ptr_gamepad_axis = world_ptr;
    input.set(
        "gamepad_axis",
        lua.create_function(move |_, (axis, id): (Value, Option<u64>)| {
            let world = unsafe { &mut *(world_ptr_gamepad_axis as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(0.0);
            };
            let input_manager = input_manager.0.read();
            let Some(axis) = lua_gamepad_axis_from_value(axis) else {
                return Ok(0.0);
            };
            let Some(gamepad_id) =
                resolve_gamepad_id(world.get_resource::<ScriptInputState>(), &input_manager, id)
            else {
                return Ok(0.0);
            };
            Ok(input_manager.get_controller_axis(gamepad_id, axis))
        })?,
    )?;

    let world_ptr_gamepad_down = world_ptr;
    input.set(
        "gamepad_button_down",
        lua.create_function(move |_, (button, id): (Value, Option<u64>)| {
            let world = unsafe { &mut *(world_ptr_gamepad_down as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            let Some(button) = lua_gamepad_button_from_value(button) else {
                return Ok(false);
            };
            let Some(gamepad_id) =
                resolve_gamepad_id(world.get_resource::<ScriptInputState>(), &input_manager, id)
            else {
                return Ok(false);
            };
            Ok(input_manager.is_controller_button_active(gamepad_id, button))
        })?,
    )?;

    let world_ptr_gamepad_pressed = world_ptr;
    input.set(
        "gamepad_button_pressed",
        lua.create_function(move |_, (button, id): (Value, Option<u64>)| {
            let world = unsafe { &mut *(world_ptr_gamepad_pressed as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            let Some(button) = lua_gamepad_button_from_value(button) else {
                return Ok(false);
            };
            let Some(gamepad_id) =
                resolve_gamepad_id(world.get_resource::<ScriptInputState>(), &input_manager, id)
            else {
                return Ok(false);
            };

            let Some(input_state) = world.get_resource::<ScriptInputState>() else {
                return Ok(false);
            };
            let id_value = usize::from(gamepad_id);
            let was_down = input_state
                .last_gamepad_buttons
                .get(&id_value)
                .map(|buttons| buttons.contains(&button))
                .unwrap_or(false);
            let is_down = input_manager.is_controller_button_active(gamepad_id, button);
            Ok(is_down && !was_down)
        })?,
    )?;

    let world_ptr_gamepad_released = world_ptr;
    input.set(
        "gamepad_button_released",
        lua.create_function(move |_, (button, id): (Value, Option<u64>)| {
            let world = unsafe { &mut *(world_ptr_gamepad_released as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(false);
            };
            let input_manager = input_manager.0.read();
            let Some(button) = lua_gamepad_button_from_value(button) else {
                return Ok(false);
            };
            let Some(gamepad_id) =
                resolve_gamepad_id(world.get_resource::<ScriptInputState>(), &input_manager, id)
            else {
                return Ok(false);
            };

            let Some(input_state) = world.get_resource::<ScriptInputState>() else {
                return Ok(false);
            };
            let id_value = usize::from(gamepad_id);
            let was_down = input_state
                .last_gamepad_buttons
                .get(&id_value)
                .map(|buttons| buttons.contains(&button))
                .unwrap_or(false);
            let is_down = input_manager.is_controller_button_active(gamepad_id, button);
            Ok(!is_down && was_down)
        })?,
    )?;

    let world_ptr_trigger = world_ptr;
    input.set(
        "gamepad_trigger",
        lua.create_function(move |_, (side, id): (Value, Option<u64>)| {
            let world = unsafe { &mut *(world_ptr_trigger as *mut World) };
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(0.0);
            };
            let input_manager = input_manager.0.read();
            let Some(side) = parse_trigger_side(side) else {
                return Ok(0.0);
            };
            let Some(gamepad_id) =
                resolve_gamepad_id(world.get_resource::<ScriptInputState>(), &input_manager, id)
            else {
                return Ok(0.0);
            };

            let value = match side {
                TriggerSide::Left => input_manager.get_left_trigger_value(gamepad_id),
                TriggerSide::Right => input_manager.get_right_trigger_value(gamepad_id),
            };
            Ok(value)
        })?,
    )?;

    Ok(input)
}

#[derive(Clone, Copy, Debug)]
enum LuaKey {
    Code(KeyCode),
    AnyShift,
    AnyCtrl,
    AnyAlt,
    AnySuper,
}

impl UserData for LuaKey {}

#[derive(Clone, Copy, Debug)]
struct LuaMouseButton(MouseButton);

impl UserData for LuaMouseButton {}

#[derive(Clone, Copy, Debug)]
struct LuaGamepadButton(gilrs::Button);

impl UserData for LuaGamepadButton {}

#[derive(Clone, Copy, Debug)]
struct LuaGamepadAxis(gilrs::Axis);

impl UserData for LuaGamepadAxis {}

#[derive(Clone, Copy, Debug)]
enum TriggerSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug)]
enum PhysicsBodyKind {
    Dynamic { mass: f32 },
    Kinematic { mode: KinematicMode },
    Fixed,
}

fn fill_key_constants(lua: &Lua, table: &Table) -> mlua::Result<()> {
    add_key_constant(lua, table, "Shift", LuaKey::AnyShift)?;
    add_key_constant(lua, table, "Ctrl", LuaKey::AnyCtrl)?;
    add_key_constant(lua, table, "Control", LuaKey::AnyCtrl)?;
    add_key_constant(lua, table, "Alt", LuaKey::AnyAlt)?;
    add_key_constant(lua, table, "Super", LuaKey::AnySuper)?;

    add_key_constant(lua, table, "LeftShift", LuaKey::Code(KeyCode::ShiftLeft))?;
    add_key_constant(lua, table, "RightShift", LuaKey::Code(KeyCode::ShiftRight))?;
    add_key_constant(lua, table, "LeftCtrl", LuaKey::Code(KeyCode::ControlLeft))?;
    add_key_constant(lua, table, "RightCtrl", LuaKey::Code(KeyCode::ControlRight))?;
    add_key_constant(lua, table, "LeftAlt", LuaKey::Code(KeyCode::AltLeft))?;
    add_key_constant(lua, table, "RightAlt", LuaKey::Code(KeyCode::AltRight))?;
    add_key_constant(lua, table, "LeftSuper", LuaKey::Code(KeyCode::SuperLeft))?;
    add_key_constant(lua, table, "RightSuper", LuaKey::Code(KeyCode::SuperRight))?;

    add_key_constant(lua, table, "Space", LuaKey::Code(KeyCode::Space))?;
    add_key_constant(lua, table, "Enter", LuaKey::Code(KeyCode::Enter))?;
    add_key_constant(lua, table, "Escape", LuaKey::Code(KeyCode::Escape))?;
    add_key_constant(lua, table, "Tab", LuaKey::Code(KeyCode::Tab))?;
    add_key_constant(lua, table, "Backspace", LuaKey::Code(KeyCode::Backspace))?;
    add_key_constant(lua, table, "Delete", LuaKey::Code(KeyCode::Delete))?;
    add_key_constant(lua, table, "Insert", LuaKey::Code(KeyCode::Insert))?;
    add_key_constant(lua, table, "Home", LuaKey::Code(KeyCode::Home))?;
    add_key_constant(lua, table, "End", LuaKey::Code(KeyCode::End))?;
    add_key_constant(lua, table, "PageUp", LuaKey::Code(KeyCode::PageUp))?;
    add_key_constant(lua, table, "PageDown", LuaKey::Code(KeyCode::PageDown))?;
    add_key_constant(lua, table, "Up", LuaKey::Code(KeyCode::ArrowUp))?;
    add_key_constant(lua, table, "Down", LuaKey::Code(KeyCode::ArrowDown))?;
    add_key_constant(lua, table, "Left", LuaKey::Code(KeyCode::ArrowLeft))?;
    add_key_constant(lua, table, "Right", LuaKey::Code(KeyCode::ArrowRight))?;

    add_key_constant(lua, table, "CapsLock", LuaKey::Code(KeyCode::CapsLock))?;
    add_key_constant(lua, table, "NumLock", LuaKey::Code(KeyCode::NumLock))?;
    add_key_constant(lua, table, "ScrollLock", LuaKey::Code(KeyCode::ScrollLock))?;
    add_key_constant(
        lua,
        table,
        "PrintScreen",
        LuaKey::Code(KeyCode::PrintScreen),
    )?;
    add_key_constant(lua, table, "Pause", LuaKey::Code(KeyCode::Pause))?;
    add_key_constant(
        lua,
        table,
        "ContextMenu",
        LuaKey::Code(KeyCode::ContextMenu),
    )?;

    add_key_constant(lua, table, "Minus", LuaKey::Code(KeyCode::Minus))?;
    add_key_constant(lua, table, "Equal", LuaKey::Code(KeyCode::Equal))?;
    add_key_constant(lua, table, "Comma", LuaKey::Code(KeyCode::Comma))?;
    add_key_constant(lua, table, "Period", LuaKey::Code(KeyCode::Period))?;
    add_key_constant(lua, table, "Slash", LuaKey::Code(KeyCode::Slash))?;
    add_key_constant(lua, table, "Backslash", LuaKey::Code(KeyCode::Backslash))?;
    add_key_constant(lua, table, "Semicolon", LuaKey::Code(KeyCode::Semicolon))?;
    add_key_constant(lua, table, "Quote", LuaKey::Code(KeyCode::Quote))?;
    add_key_constant(lua, table, "Backquote", LuaKey::Code(KeyCode::Backquote))?;
    add_key_constant(
        lua,
        table,
        "LeftBracket",
        LuaKey::Code(KeyCode::BracketLeft),
    )?;
    add_key_constant(
        lua,
        table,
        "RightBracket",
        LuaKey::Code(KeyCode::BracketRight),
    )?;

    for letter in b'A'..=b'Z' {
        let name = (letter as char).to_string();
        if let Some(code) = keycode_from_letter(letter.to_ascii_lowercase()) {
            add_key_constant(lua, table, &name, LuaKey::Code(code))?;
            let alias = format!("Key{}", name);
            add_key_constant(lua, table, &alias, LuaKey::Code(code))?;
        }
    }

    for digit in 0..=9 {
        let name = digit.to_string();
        if let Some(code) = keycode_from_digit(b'0' + digit) {
            add_key_constant(lua, table, &name, LuaKey::Code(code))?;
            let alias = format!("Digit{}", digit);
            add_key_constant(lua, table, &alias, LuaKey::Code(code))?;
        }

        if let Some(code) = keycode_from_numpad_digit(b'0' + digit) {
            let alias = format!("Numpad{}", digit);
            add_key_constant(lua, table, &alias, LuaKey::Code(code))?;
            let short = format!("Num{}", digit);
            add_key_constant(lua, table, &short, LuaKey::Code(code))?;
        }
    }

    for num in 1..=24 {
        if let Some(code) = keycode_from_function(num) {
            let name = format!("F{}", num);
            add_key_constant(lua, table, &name, LuaKey::Code(code))?;
        }
    }

    add_key_constant(lua, table, "NumpadAdd", LuaKey::Code(KeyCode::NumpadAdd))?;
    add_key_constant(
        lua,
        table,
        "NumpadSub",
        LuaKey::Code(KeyCode::NumpadSubtract),
    )?;
    add_key_constant(
        lua,
        table,
        "NumpadMul",
        LuaKey::Code(KeyCode::NumpadMultiply),
    )?;
    add_key_constant(lua, table, "NumpadDiv", LuaKey::Code(KeyCode::NumpadDivide))?;
    add_key_constant(
        lua,
        table,
        "NumpadEnter",
        LuaKey::Code(KeyCode::NumpadEnter),
    )?;
    add_key_constant(
        lua,
        table,
        "NumpadDecimal",
        LuaKey::Code(KeyCode::NumpadDecimal),
    )?;

    Ok(())
}

fn fill_mouse_button_constants(lua: &Lua, table: &Table) -> mlua::Result<()> {
    add_mouse_button_constant(lua, table, "Left", MouseButton::Left)?;
    add_mouse_button_constant(lua, table, "Right", MouseButton::Right)?;
    add_mouse_button_constant(lua, table, "Middle", MouseButton::Middle)?;
    add_mouse_button_constant(lua, table, "Back", MouseButton::Back)?;
    add_mouse_button_constant(lua, table, "Forward", MouseButton::Forward)?;
    add_mouse_button_constant(lua, table, "Primary", MouseButton::Left)?;
    add_mouse_button_constant(lua, table, "Secondary", MouseButton::Right)?;
    Ok(())
}

fn fill_gamepad_button_constants(lua: &Lua, table: &Table) -> mlua::Result<()> {
    use gilrs::Button;

    add_gamepad_button_constant(lua, table, "South", Button::South)?;
    add_gamepad_button_constant(lua, table, "East", Button::East)?;
    add_gamepad_button_constant(lua, table, "North", Button::North)?;
    add_gamepad_button_constant(lua, table, "West", Button::West)?;
    add_gamepad_button_constant(lua, table, "A", Button::South)?;
    add_gamepad_button_constant(lua, table, "B", Button::East)?;
    add_gamepad_button_constant(lua, table, "X", Button::West)?;
    add_gamepad_button_constant(lua, table, "Y", Button::North)?;

    add_gamepad_button_constant(lua, table, "LeftTrigger", Button::LeftTrigger)?;
    add_gamepad_button_constant(lua, table, "RightTrigger", Button::RightTrigger)?;
    add_gamepad_button_constant(lua, table, "LeftTrigger2", Button::LeftTrigger2)?;
    add_gamepad_button_constant(lua, table, "RightTrigger2", Button::RightTrigger2)?;
    add_gamepad_button_constant(lua, table, "LB", Button::LeftTrigger)?;
    add_gamepad_button_constant(lua, table, "RB", Button::RightTrigger)?;
    add_gamepad_button_constant(lua, table, "LT", Button::LeftTrigger2)?;
    add_gamepad_button_constant(lua, table, "RT", Button::RightTrigger2)?;

    add_gamepad_button_constant(lua, table, "Select", Button::Select)?;
    add_gamepad_button_constant(lua, table, "Start", Button::Start)?;
    add_gamepad_button_constant(lua, table, "Mode", Button::Mode)?;
    add_gamepad_button_constant(lua, table, "Back", Button::Select)?;
    add_gamepad_button_constant(lua, table, "Menu", Button::Start)?;
    add_gamepad_button_constant(lua, table, "Guide", Button::Mode)?;

    add_gamepad_button_constant(lua, table, "LeftThumb", Button::LeftThumb)?;
    add_gamepad_button_constant(lua, table, "RightThumb", Button::RightThumb)?;
    add_gamepad_button_constant(lua, table, "LS", Button::LeftThumb)?;
    add_gamepad_button_constant(lua, table, "RS", Button::RightThumb)?;
    add_gamepad_button_constant(lua, table, "L3", Button::LeftThumb)?;
    add_gamepad_button_constant(lua, table, "R3", Button::RightThumb)?;

    add_gamepad_button_constant(lua, table, "DPadUp", Button::DPadUp)?;
    add_gamepad_button_constant(lua, table, "DPadDown", Button::DPadDown)?;
    add_gamepad_button_constant(lua, table, "DPadLeft", Button::DPadLeft)?;
    add_gamepad_button_constant(lua, table, "DPadRight", Button::DPadRight)?;

    add_gamepad_button_constant(lua, table, "C", Button::C)?;
    add_gamepad_button_constant(lua, table, "Z", Button::Z)?;

    Ok(())
}

fn fill_gamepad_axis_constants(lua: &Lua, table: &Table) -> mlua::Result<()> {
    use gilrs::Axis;

    add_gamepad_axis_constant(lua, table, "LeftX", Axis::LeftStickX)?;
    add_gamepad_axis_constant(lua, table, "LeftY", Axis::LeftStickY)?;
    add_gamepad_axis_constant(lua, table, "RightX", Axis::RightStickX)?;
    add_gamepad_axis_constant(lua, table, "RightY", Axis::RightStickY)?;
    add_gamepad_axis_constant(lua, table, "LeftZ", Axis::LeftZ)?;
    add_gamepad_axis_constant(lua, table, "RightZ", Axis::RightZ)?;
    add_gamepad_axis_constant(lua, table, "DPadX", Axis::DPadX)?;
    add_gamepad_axis_constant(lua, table, "DPadY", Axis::DPadY)?;
    add_gamepad_axis_constant(lua, table, "LeftStickX", Axis::LeftStickX)?;
    add_gamepad_axis_constant(lua, table, "LeftStickY", Axis::LeftStickY)?;
    add_gamepad_axis_constant(lua, table, "RightStickX", Axis::RightStickX)?;
    add_gamepad_axis_constant(lua, table, "RightStickY", Axis::RightStickY)?;

    Ok(())
}

fn add_key_constant(lua: &Lua, table: &Table, name: &str, key: LuaKey) -> mlua::Result<()> {
    table.set(name, lua.create_userdata(key)?)?;
    Ok(())
}

fn add_mouse_button_constant(
    lua: &Lua,
    table: &Table,
    name: &str,
    button: MouseButton,
) -> mlua::Result<()> {
    table.set(name, lua.create_userdata(LuaMouseButton(button))?)?;
    Ok(())
}

fn add_gamepad_button_constant(
    lua: &Lua,
    table: &Table,
    name: &str,
    button: gilrs::Button,
) -> mlua::Result<()> {
    table.set(name, lua.create_userdata(LuaGamepadButton(button))?)?;
    Ok(())
}

fn add_gamepad_axis_constant(
    lua: &Lua,
    table: &Table,
    name: &str,
    axis: gilrs::Axis,
) -> mlua::Result<()> {
    table.set(name, lua.create_userdata(LuaGamepadAxis(axis))?)?;
    Ok(())
}

fn lua_key_from_value(value: Value) -> Option<LuaKey> {
    match value {
        Value::UserData(userdata) => userdata.borrow::<LuaKey>().ok().map(|key| *key),
        Value::String(value) => parse_key_name(&value.to_string_lossy()),
        Value::Integer(value) => {
            if value >= 0 && value <= 9 {
                keycode_from_digit(b'0' + value as u8).map(LuaKey::Code)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn lua_mouse_button_from_value(value: Value) -> Option<MouseButton> {
    match value {
        Value::UserData(userdata) => userdata
            .borrow::<LuaMouseButton>()
            .ok()
            .map(|button| button.0),
        Value::String(value) => parse_mouse_button_name(&value.to_string_lossy()),
        Value::Integer(value) => mouse_button_from_int(value),
        _ => None,
    }
}

fn lua_gamepad_button_from_value(value: Value) -> Option<gilrs::Button> {
    match value {
        Value::UserData(userdata) => userdata
            .borrow::<LuaGamepadButton>()
            .ok()
            .map(|button| button.0),
        Value::String(value) => parse_gamepad_button_name(&value.to_string_lossy()),
        _ => None,
    }
}

fn lua_gamepad_axis_from_value(value: Value) -> Option<gilrs::Axis> {
    match value {
        Value::UserData(userdata) => userdata.borrow::<LuaGamepadAxis>().ok().map(|axis| axis.0),
        Value::String(value) => parse_gamepad_axis_name(&value.to_string_lossy()),
        _ => None,
    }
}

fn lua_key_is_active(input: &InputManager, key: LuaKey) -> bool {
    match key {
        LuaKey::Code(code) => input.is_key_active(code),
        LuaKey::AnyShift => {
            input.is_key_active(KeyCode::ShiftLeft) || input.is_key_active(KeyCode::ShiftRight)
        }
        LuaKey::AnyCtrl => {
            input.is_key_active(KeyCode::ControlLeft) || input.is_key_active(KeyCode::ControlRight)
        }
        LuaKey::AnyAlt => {
            input.is_key_active(KeyCode::AltLeft) || input.is_key_active(KeyCode::AltRight)
        }
        LuaKey::AnySuper => {
            input.is_key_active(KeyCode::SuperLeft) || input.is_key_active(KeyCode::SuperRight)
        }
    }
}

fn lua_key_is_just_pressed(input: &InputManager, key: LuaKey) -> bool {
    match key {
        LuaKey::Code(code) => input.was_just_pressed(code),
        LuaKey::AnyShift => {
            input.was_just_pressed(KeyCode::ShiftLeft)
                || input.was_just_pressed(KeyCode::ShiftRight)
        }
        LuaKey::AnyCtrl => {
            input.was_just_pressed(KeyCode::ControlLeft)
                || input.was_just_pressed(KeyCode::ControlRight)
        }
        LuaKey::AnyAlt => {
            input.was_just_pressed(KeyCode::AltLeft) || input.was_just_pressed(KeyCode::AltRight)
        }
        LuaKey::AnySuper => {
            input.was_just_pressed(KeyCode::SuperLeft)
                || input.was_just_pressed(KeyCode::SuperRight)
        }
    }
}

fn lua_key_is_released(input: &InputManager, state: &ScriptInputState, key: LuaKey) -> bool {
    let was_down = match key {
        LuaKey::Code(code) => state.last_keys.contains(&code),
        LuaKey::AnyShift => {
            state.last_keys.contains(&KeyCode::ShiftLeft)
                || state.last_keys.contains(&KeyCode::ShiftRight)
        }
        LuaKey::AnyCtrl => {
            state.last_keys.contains(&KeyCode::ControlLeft)
                || state.last_keys.contains(&KeyCode::ControlRight)
        }
        LuaKey::AnyAlt => {
            state.last_keys.contains(&KeyCode::AltLeft)
                || state.last_keys.contains(&KeyCode::AltRight)
        }
        LuaKey::AnySuper => {
            state.last_keys.contains(&KeyCode::SuperLeft)
                || state.last_keys.contains(&KeyCode::SuperRight)
        }
    };

    let is_down = lua_key_is_active(input, key);
    was_down && !is_down
}

fn parse_key_name(name: &str) -> Option<LuaKey> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.len() == 1 {
        let byte = trimmed.as_bytes()[0];
        if byte.is_ascii_alphabetic() {
            return keycode_from_letter(byte.to_ascii_lowercase()).map(LuaKey::Code);
        }
        if byte.is_ascii_digit() {
            return keycode_from_digit(byte).map(LuaKey::Code);
        }
        return match byte {
            b' ' => Some(LuaKey::Code(KeyCode::Space)),
            b'`' => Some(LuaKey::Code(KeyCode::Backquote)),
            b'-' => Some(LuaKey::Code(KeyCode::Minus)),
            b'=' => Some(LuaKey::Code(KeyCode::Equal)),
            b'[' => Some(LuaKey::Code(KeyCode::BracketLeft)),
            b']' => Some(LuaKey::Code(KeyCode::BracketRight)),
            b'\\' => Some(LuaKey::Code(KeyCode::Backslash)),
            b';' => Some(LuaKey::Code(KeyCode::Semicolon)),
            b'\'' => Some(LuaKey::Code(KeyCode::Quote)),
            b',' => Some(LuaKey::Code(KeyCode::Comma)),
            b'.' => Some(LuaKey::Code(KeyCode::Period)),
            b'/' => Some(LuaKey::Code(KeyCode::Slash)),
            _ => None,
        };
    }

    let normalized = normalize_name(trimmed);
    if normalized.is_empty() {
        return None;
    }

    match normalized.as_str() {
        "shift" => return Some(LuaKey::AnyShift),
        "ctrl" | "control" => return Some(LuaKey::AnyCtrl),
        "alt" | "option" => return Some(LuaKey::AnyAlt),
        "super" | "meta" | "command" | "cmd" | "win" | "windows" => return Some(LuaKey::AnySuper),
        "space" => return Some(LuaKey::Code(KeyCode::Space)),
        "enter" | "return" => return Some(LuaKey::Code(KeyCode::Enter)),
        "escape" | "esc" => return Some(LuaKey::Code(KeyCode::Escape)),
        "tab" => return Some(LuaKey::Code(KeyCode::Tab)),
        "backspace" => return Some(LuaKey::Code(KeyCode::Backspace)),
        "delete" | "del" => return Some(LuaKey::Code(KeyCode::Delete)),
        "insert" | "ins" => return Some(LuaKey::Code(KeyCode::Insert)),
        "home" => return Some(LuaKey::Code(KeyCode::Home)),
        "end" => return Some(LuaKey::Code(KeyCode::End)),
        "pageup" | "pgup" => return Some(LuaKey::Code(KeyCode::PageUp)),
        "pagedown" | "pgdn" => return Some(LuaKey::Code(KeyCode::PageDown)),
        "up" | "arrowup" => return Some(LuaKey::Code(KeyCode::ArrowUp)),
        "down" | "arrowdown" => return Some(LuaKey::Code(KeyCode::ArrowDown)),
        "left" | "arrowleft" => return Some(LuaKey::Code(KeyCode::ArrowLeft)),
        "right" | "arrowright" => return Some(LuaKey::Code(KeyCode::ArrowRight)),
        "capslock" => return Some(LuaKey::Code(KeyCode::CapsLock)),
        "numlock" => return Some(LuaKey::Code(KeyCode::NumLock)),
        "scrolllock" => return Some(LuaKey::Code(KeyCode::ScrollLock)),
        "printscreen" | "prtsc" => return Some(LuaKey::Code(KeyCode::PrintScreen)),
        "pause" => return Some(LuaKey::Code(KeyCode::Pause)),
        "contextmenu" | "menu" => return Some(LuaKey::Code(KeyCode::ContextMenu)),
        "lshift" | "shiftleft" | "leftshift" => return Some(LuaKey::Code(KeyCode::ShiftLeft)),
        "rshift" | "shiftright" | "rightshift" => return Some(LuaKey::Code(KeyCode::ShiftRight)),
        "lctrl" | "ctrlleft" | "leftctrl" => return Some(LuaKey::Code(KeyCode::ControlLeft)),
        "rctrl" | "ctrlright" | "rightctrl" => return Some(LuaKey::Code(KeyCode::ControlRight)),
        "lalt" | "altleft" | "leftalt" => return Some(LuaKey::Code(KeyCode::AltLeft)),
        "ralt" | "altright" | "rightalt" => return Some(LuaKey::Code(KeyCode::AltRight)),
        "lsuper" | "superleft" | "leftsuper" | "lmeta" | "metaleft" => {
            return Some(LuaKey::Code(KeyCode::SuperLeft));
        }
        "rsuper" | "superright" | "rightsuper" | "rmeta" | "metaright" => {
            return Some(LuaKey::Code(KeyCode::SuperRight));
        }
        "minus" => return Some(LuaKey::Code(KeyCode::Minus)),
        "equal" | "equals" => return Some(LuaKey::Code(KeyCode::Equal)),
        "comma" => return Some(LuaKey::Code(KeyCode::Comma)),
        "period" | "dot" => return Some(LuaKey::Code(KeyCode::Period)),
        "slash" => return Some(LuaKey::Code(KeyCode::Slash)),
        "backslash" => return Some(LuaKey::Code(KeyCode::Backslash)),
        "semicolon" => return Some(LuaKey::Code(KeyCode::Semicolon)),
        "quote" | "apostrophe" => return Some(LuaKey::Code(KeyCode::Quote)),
        "backquote" | "grave" => return Some(LuaKey::Code(KeyCode::Backquote)),
        "leftbracket" | "lbracket" => return Some(LuaKey::Code(KeyCode::BracketLeft)),
        "rightbracket" | "rbracket" => return Some(LuaKey::Code(KeyCode::BracketRight)),
        "numpadadd" => return Some(LuaKey::Code(KeyCode::NumpadAdd)),
        "numpadsub" | "numpadminus" => return Some(LuaKey::Code(KeyCode::NumpadSubtract)),
        "numpadmul" | "numpadmultiply" => return Some(LuaKey::Code(KeyCode::NumpadMultiply)),
        "numpaddiv" | "numpaddivide" => return Some(LuaKey::Code(KeyCode::NumpadDivide)),
        "numpadenter" => return Some(LuaKey::Code(KeyCode::NumpadEnter)),
        "numpaddecimal" | "numpaddot" => return Some(LuaKey::Code(KeyCode::NumpadDecimal)),
        _ => {}
    }

    if let Some(rest) = normalized.strip_prefix("key") {
        if rest.len() == 1 {
            let byte = rest.as_bytes()[0];
            if byte.is_ascii_alphabetic() {
                return keycode_from_letter(byte).map(LuaKey::Code);
            }
        }
    }

    if let Some(rest) = normalized.strip_prefix("digit") {
        if rest.len() == 1 {
            let byte = rest.as_bytes()[0];
            if byte.is_ascii_digit() {
                return keycode_from_digit(byte).map(LuaKey::Code);
            }
        }
    }

    if let Some(rest) = normalized.strip_prefix("numpad") {
        if rest.len() == 1 {
            let byte = rest.as_bytes()[0];
            if byte.is_ascii_digit() {
                return keycode_from_numpad_digit(byte).map(LuaKey::Code);
            }
        }
    }

    if let Some(rest) = normalized.strip_prefix("f") {
        if let Ok(number) = rest.parse::<u8>() {
            return keycode_from_function(number).map(LuaKey::Code);
        }
    }

    None
}

fn parse_mouse_button_name(name: &str) -> Option<MouseButton> {
    let normalized = normalize_name(name);
    match normalized.as_str() {
        "left" | "primary" | "button1" => Some(MouseButton::Left),
        "right" | "secondary" | "button2" => Some(MouseButton::Right),
        "middle" | "button3" => Some(MouseButton::Middle),
        "back" | "button4" => Some(MouseButton::Back),
        "forward" | "button5" => Some(MouseButton::Forward),
        _ => {
            if let Some(rest) = normalized.strip_prefix("button") {
                if let Ok(value) = rest.parse::<i64>() {
                    return mouse_button_from_int(value);
                }
            }
            None
        }
    }
}

fn parse_gamepad_button_name(name: &str) -> Option<gilrs::Button> {
    let normalized = normalize_name(name);
    match normalized.as_str() {
        "south" | "a" => Some(gilrs::Button::South),
        "east" | "b" => Some(gilrs::Button::East),
        "north" | "y" => Some(gilrs::Button::North),
        "west" | "x" => Some(gilrs::Button::West),
        "lefttrigger" | "lb" | "l1" => Some(gilrs::Button::LeftTrigger),
        "righttrigger" | "rb" | "r1" => Some(gilrs::Button::RightTrigger),
        "lefttrigger2" | "lt" | "l2" => Some(gilrs::Button::LeftTrigger2),
        "righttrigger2" | "rt" | "r2" => Some(gilrs::Button::RightTrigger2),
        "leftthumb" | "leftstick" | "ls" | "l3" => Some(gilrs::Button::LeftThumb),
        "rightthumb" | "rightstick" | "rs" | "r3" => Some(gilrs::Button::RightThumb),
        "select" | "back" => Some(gilrs::Button::Select),
        "start" | "menu" => Some(gilrs::Button::Start),
        "mode" | "guide" => Some(gilrs::Button::Mode),
        "dpadup" | "up" => Some(gilrs::Button::DPadUp),
        "dpaddown" | "down" => Some(gilrs::Button::DPadDown),
        "dpadleft" | "left" => Some(gilrs::Button::DPadLeft),
        "dpadright" | "right" => Some(gilrs::Button::DPadRight),
        "c" => Some(gilrs::Button::C),
        "z" => Some(gilrs::Button::Z),
        _ => None,
    }
}

fn parse_gamepad_axis_name(name: &str) -> Option<gilrs::Axis> {
    let normalized = normalize_name(name);
    match normalized.as_str() {
        "leftx" | "lx" | "leftstickx" => Some(gilrs::Axis::LeftStickX),
        "lefty" | "ly" | "leftsticky" => Some(gilrs::Axis::LeftStickY),
        "rightx" | "rx" | "rightstickx" => Some(gilrs::Axis::RightStickX),
        "righty" | "ry" | "rightsticky" => Some(gilrs::Axis::RightStickY),
        "leftz" | "lz" => Some(gilrs::Axis::LeftZ),
        "rightz" | "rz" => Some(gilrs::Axis::RightZ),
        "dpadx" | "dx" => Some(gilrs::Axis::DPadX),
        "dpady" | "dy" => Some(gilrs::Axis::DPadY),
        _ => None,
    }
}

fn parse_cursor_grab_mode(value: Value) -> Option<RuntimeCursorGrabMode> {
    match value {
        Value::Nil => Some(RuntimeCursorGrabMode::None),
        Value::Boolean(value) => Some(if value {
            RuntimeCursorGrabMode::Locked
        } else {
            RuntimeCursorGrabMode::None
        }),
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "none" | "free" | "off" | "release" | "released" => {
                    Some(RuntimeCursorGrabMode::None)
                }
                "confined" | "confine" | "clip" | "clipped" => {
                    Some(RuntimeCursorGrabMode::Confined)
                }
                "locked" | "lock" | "capture" | "captured" => Some(RuntimeCursorGrabMode::Locked),
                _ => None,
            }
        }
        _ => None,
    }
}

fn parse_trigger_side(value: Value) -> Option<TriggerSide> {
    match value {
        Value::String(value) => {
            let normalized = normalize_name(&value.to_string_lossy());
            match normalized.as_str() {
                "left" | "lt" | "l2" | "lefttrigger" | "lefttrigger2" => Some(TriggerSide::Left),
                "right" | "rt" | "r2" | "righttrigger" | "righttrigger2" => {
                    Some(TriggerSide::Right)
                }
                _ => None,
            }
        }
        Value::UserData(userdata) => {
            if let Ok(button) = userdata.borrow::<LuaGamepadButton>() {
                return match button.0 {
                    gilrs::Button::LeftTrigger | gilrs::Button::LeftTrigger2 => {
                        Some(TriggerSide::Left)
                    }
                    gilrs::Button::RightTrigger | gilrs::Button::RightTrigger2 => {
                        Some(TriggerSide::Right)
                    }
                    _ => None,
                };
            }
            if let Ok(axis) = userdata.borrow::<LuaGamepadAxis>() {
                return match axis.0 {
                    gilrs::Axis::LeftZ => Some(TriggerSide::Left),
                    gilrs::Axis::RightZ => Some(TriggerSide::Right),
                    _ => None,
                };
            }
            None
        }
        _ => None,
    }
}

fn resolve_gamepad_id(
    input_state: Option<&ScriptInputState>,
    input_manager: &InputManager,
    id: Option<u64>,
) -> Option<gilrs::GamepadId> {
    if let Some(id) = id {
        return input_state.and_then(|state| state.gamepad_map.get(&(id as usize)).copied());
    }

    input_manager.first_gamepad_id()
}

fn normalize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for byte in name.bytes() {
        let normalized = byte.to_ascii_lowercase();
        if normalized.is_ascii_alphanumeric() {
            out.push(normalized as char);
        }
    }
    out
}

fn mouse_button_from_int(value: i64) -> Option<MouseButton> {
    if value <= 0 {
        return None;
    }
    if value > u16::MAX as i64 {
        return None;
    }
    let value = value as u16;
    Some(match value {
        1 => MouseButton::Left,
        2 => MouseButton::Right,
        3 => MouseButton::Middle,
        4 => MouseButton::Back,
        5 => MouseButton::Forward,
        _ => MouseButton::Other(value),
    })
}

fn keycode_from_letter(letter: u8) -> Option<KeyCode> {
    match letter {
        b'a' => Some(KeyCode::KeyA),
        b'b' => Some(KeyCode::KeyB),
        b'c' => Some(KeyCode::KeyC),
        b'd' => Some(KeyCode::KeyD),
        b'e' => Some(KeyCode::KeyE),
        b'f' => Some(KeyCode::KeyF),
        b'g' => Some(KeyCode::KeyG),
        b'h' => Some(KeyCode::KeyH),
        b'i' => Some(KeyCode::KeyI),
        b'j' => Some(KeyCode::KeyJ),
        b'k' => Some(KeyCode::KeyK),
        b'l' => Some(KeyCode::KeyL),
        b'm' => Some(KeyCode::KeyM),
        b'n' => Some(KeyCode::KeyN),
        b'o' => Some(KeyCode::KeyO),
        b'p' => Some(KeyCode::KeyP),
        b'q' => Some(KeyCode::KeyQ),
        b'r' => Some(KeyCode::KeyR),
        b's' => Some(KeyCode::KeyS),
        b't' => Some(KeyCode::KeyT),
        b'u' => Some(KeyCode::KeyU),
        b'v' => Some(KeyCode::KeyV),
        b'w' => Some(KeyCode::KeyW),
        b'x' => Some(KeyCode::KeyX),
        b'y' => Some(KeyCode::KeyY),
        b'z' => Some(KeyCode::KeyZ),
        _ => None,
    }
}

fn keycode_from_digit(digit: u8) -> Option<KeyCode> {
    match digit {
        b'0' => Some(KeyCode::Digit0),
        b'1' => Some(KeyCode::Digit1),
        b'2' => Some(KeyCode::Digit2),
        b'3' => Some(KeyCode::Digit3),
        b'4' => Some(KeyCode::Digit4),
        b'5' => Some(KeyCode::Digit5),
        b'6' => Some(KeyCode::Digit6),
        b'7' => Some(KeyCode::Digit7),
        b'8' => Some(KeyCode::Digit8),
        b'9' => Some(KeyCode::Digit9),
        _ => None,
    }
}

fn keycode_from_numpad_digit(digit: u8) -> Option<KeyCode> {
    match digit {
        b'0' => Some(KeyCode::Numpad0),
        b'1' => Some(KeyCode::Numpad1),
        b'2' => Some(KeyCode::Numpad2),
        b'3' => Some(KeyCode::Numpad3),
        b'4' => Some(KeyCode::Numpad4),
        b'5' => Some(KeyCode::Numpad5),
        b'6' => Some(KeyCode::Numpad6),
        b'7' => Some(KeyCode::Numpad7),
        b'8' => Some(KeyCode::Numpad8),
        b'9' => Some(KeyCode::Numpad9),
        _ => None,
    }
}

fn keycode_from_function(number: u8) -> Option<KeyCode> {
    match number {
        1 => Some(KeyCode::F1),
        2 => Some(KeyCode::F2),
        3 => Some(KeyCode::F3),
        4 => Some(KeyCode::F4),
        5 => Some(KeyCode::F5),
        6 => Some(KeyCode::F6),
        7 => Some(KeyCode::F7),
        8 => Some(KeyCode::F8),
        9 => Some(KeyCode::F9),
        10 => Some(KeyCode::F10),
        11 => Some(KeyCode::F11),
        12 => Some(KeyCode::F12),
        13 => Some(KeyCode::F13),
        14 => Some(KeyCode::F14),
        15 => Some(KeyCode::F15),
        16 => Some(KeyCode::F16),
        17 => Some(KeyCode::F17),
        18 => Some(KeyCode::F18),
        19 => Some(KeyCode::F19),
        20 => Some(KeyCode::F20),
        21 => Some(KeyCode::F21),
        22 => Some(KeyCode::F22),
        23 => Some(KeyCode::F23),
        24 => Some(KeyCode::F24),
        _ => None,
    }
}

fn lua_value_to_string(value: Value) -> String {
    match value {
        Value::Nil => "nil".to_string(),
        Value::Boolean(value) => value.to_string(),
        Value::Integer(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.to_string_lossy().to_string(),
        Value::Table(_) => "<table>".to_string(),
        Value::Function(_) => "<function>".to_string(),
        Value::UserData(_) => "<userdata>".to_string(),
        Value::LightUserData(_) => "<lightuserdata>".to_string(),
        Value::Vector(_) => "<vector>".to_string(),
        Value::Buffer(_) => "<buffer>".to_string(),
        Value::Thread(_) => "<thread>".to_string(),
        Value::Error(err) => err.to_string(),
        Value::Other(_) => "<other>".to_string(),
    }
}

fn lookup_editor_entity(world: &mut World, entity_id: u64) -> Option<Entity> {
    let entity = Entity::from_bits(entity_id);
    if world.get_entity(entity).is_err() {
        return None;
    }
    if world.get::<EditorEntity>(entity).is_none() {
        return None;
    }
    Some(entity)
}

fn despawn_script_owned(world: &mut World, owner: Entity, script_index: usize) {
    let spawned: Vec<Entity> = world
        .query::<(Entity, &ScriptSpawned)>()
        .iter(world)
        .filter_map(|(entity, marker)| {
            if marker.owner == owner && marker.script_index == script_index {
                Some(entity)
            } else {
                None
            }
        })
        .collect();

    for entity in spawned {
        let _ = world.despawn(entity);
    }
}

fn ensure_transform(world: &mut World, entity: Entity) {
    if world.get::<BevyTransform>(entity).is_none() {
        world.entity_mut(entity).insert(BevyTransform::default());
    }
}

fn vec2_to_table(lua: &Lua, value: Vec2) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("x", value.x)?;
    table.set("y", value.y)?;
    table.set(1, value.x)?;
    table.set(2, value.y)?;
    Ok(table)
}

fn vec3_to_table(lua: &Lua, value: Vec3) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("x", value.x)?;
    table.set("y", value.y)?;
    table.set("z", value.z)?;
    table.set(1, value.x)?;
    table.set(2, value.y)?;
    table.set(3, value.z)?;
    Ok(table)
}

fn quat_to_table(lua: &Lua, value: Quat) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("x", value.x)?;
    table.set("y", value.y)?;
    table.set("z", value.z)?;
    table.set("w", value.w)?;
    table.set(1, value.x)?;
    table.set(2, value.y)?;
    table.set(3, value.z)?;
    table.set(4, value.w)?;
    Ok(table)
}

fn table_to_vec3(table: &Table) -> Option<Vec3> {
    let x = table
        .get::<f32>("x")
        .ok()
        .or_else(|| table.get::<f32>(1).ok())?;
    let y = table
        .get::<f32>("y")
        .ok()
        .or_else(|| table.get::<f32>(2).ok())?;
    let z = table
        .get::<f32>("z")
        .ok()
        .or_else(|| table.get::<f32>(3).ok())?;
    Some(Vec3::new(x, y, z))
}

fn table_to_quat(table: &Table) -> Option<Quat> {
    let x = table
        .get::<f32>("x")
        .ok()
        .or_else(|| table.get::<f32>(1).ok())?;
    let y = table
        .get::<f32>("y")
        .ok()
        .or_else(|| table.get::<f32>(2).ok())?;
    let z = table
        .get::<f32>("z")
        .ok()
        .or_else(|| table.get::<f32>(3).ok())?;
    let w = table
        .get::<f32>("w")
        .ok()
        .or_else(|| table.get::<f32>(4).ok())?;
    Some(Quat::from_xyzw(x, y, z, w))
}

fn light_type_name(kind: LightType) -> &'static str {
    match kind {
        LightType::Directional => "Directional",
        LightType::Point => "Point",
        LightType::Spot { .. } => "Spot",
    }
}

fn spline_mode_to_str(mode: SplineMode) -> &'static str {
    match mode {
        SplineMode::Linear => "linear",
        SplineMode::CatmullRom => "catmullrom",
        SplineMode::Bezier => "bezier",
    }
}

fn spline_mode_from_str(mode: &str) -> Option<SplineMode> {
    match mode.to_ascii_lowercase().as_str() {
        "linear" => Some(SplineMode::Linear),
        "catmullrom" | "catmull_rom" | "catmull-rom" => Some(SplineMode::CatmullRom),
        "bezier" => Some(SplineMode::Bezier),
        _ => None,
    }
}

fn parse_light_type(value: &str, current: LightType) -> LightType {
    match value.to_ascii_lowercase().as_str() {
        "directional" => LightType::Directional,
        "point" => LightType::Point,
        "spot" => match current {
            LightType::Spot { angle } => LightType::Spot { angle },
            _ => LightType::Spot {
                angle: 45.0_f32.to_radians(),
            },
        },
        _ => current,
    }
}

fn set_active_camera(world: &mut World, entity: Entity) {
    set_play_camera(world, entity);
    if let Some(mut viewport_state) = world.get_resource_mut::<EditorViewportState>() {
        viewport_state.play_mode_view = PlayViewportKind::Gameplay;
    }
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|state| state.world_state)
        .unwrap_or(WorldState::Edit);
    if world_state == WorldState::Play {
        activate_play_camera(world);
    }
}

fn parse_mesh_source(value: Value, project: Option<&EditorProject>) -> Option<MeshSource> {
    match value {
        Value::String(value) => {
            let raw = value.to_string_lossy().to_string();
            if raw.eq_ignore_ascii_case("cube") {
                Some(MeshSource::Primitive(PrimitiveKind::Cube))
            } else if raw.eq_ignore_ascii_case("uv sphere") {
                Some(MeshSource::Primitive(PrimitiveKind::UvSphere(12, 12)))
            } else if raw.eq_ignore_ascii_case("plane") {
                Some(MeshSource::Primitive(PrimitiveKind::Plane))
            } else {
                Some(MeshSource::Asset {
                    path: normalize_project_path(project, Path::new(&raw)),
                })
            }
        }
        Value::Table(table) => {
            if let Ok(kind) = table.get::<String>("primitive") {
                if kind.eq_ignore_ascii_case("cube") {
                    return Some(MeshSource::Primitive(PrimitiveKind::Cube));
                }
                if kind.eq_ignore_ascii_case("uv sphere") {
                    return Some(MeshSource::Primitive(PrimitiveKind::UvSphere(12, 12)));
                }
                if kind.eq_ignore_ascii_case("plane") {
                    return Some(MeshSource::Primitive(PrimitiveKind::Plane));
                }
            }
            if let Ok(path) = table.get::<String>("path") {
                return Some(MeshSource::Asset {
                    path: normalize_project_path(project, Path::new(&path)),
                });
            }
            None
        }
        _ => None,
    }
}

fn parse_material_path(value: Value, project: Option<&EditorProject>) -> Option<String> {
    match value {
        Value::Nil => None,
        Value::String(value) => {
            let raw = value.to_string_lossy().to_string();
            Some(normalize_project_path(project, Path::new(&raw)))
        }
        _ => None,
    }
}

fn normalize_project_path(project: Option<&EditorProject>, path: &Path) -> String {
    project
        .and_then(|project| {
            project
                .root
                .as_ref()
                .and_then(|root| path.strip_prefix(root).ok())
        })
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|| path.to_string_lossy().replace('\\', "/"))
}

fn resolve_project_path(project: Option<&EditorProject>, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    if let Some(project) = project {
        if let Some(root) = project.root.as_ref() {
            return root.join(path);
        }
    }
    path.to_path_buf()
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
    project: Option<&EditorProject>,
    source: MeshSource,
    material_path: Option<String>,
    casts_shadow: bool,
    visible: bool,
) -> bool {
    let mut applied = false;
    world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let Some(asset_server) = world.get_resource::<BevyAssetServer>() else {
            return;
        };

        let material_handle = material_path
            .as_ref()
            .and_then(|path| load_material_handle(path, &mut cache, asset_server, project))
            .or_else(|| {
                project
                    .and_then(|project| ensure_default_material(project, &mut cache, asset_server))
            });

        let Some(material_handle) = material_handle else {
            return;
        };

        let mesh_handle = match &source {
            MeshSource::Primitive(kind) => {
                Some(load_primitive_mesh(*kind, &mut cache, asset_server))
            }
            MeshSource::Asset { path } => {
                Some(load_mesh_asset(path, &mut cache, asset_server, project))
            }
        };

        let Some(mesh_handle) = mesh_handle else {
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
        applied = true;
    });

    applied
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

    let full_path = resolve_project_path(project, Path::new(path));
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

    let full_path = resolve_project_path(project, Path::new(path));
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
        PrimitiveKind::Cube => MeshAsset::cube("cube".to_string()),
        PrimitiveKind::UvSphere(segments, rings) => {
            MeshAsset::uv_sphere("uv sphere".to_string(), segments, rings)
        }
        PrimitiveKind::Plane => MeshAsset::plane("plane".to_string()),
    };

    let handle = asset_server
        .0
        .lock()
        .add_mesh(mesh_asset.vertices.unwrap(), mesh_asset.indices);
    cache.primitive_meshes.insert(kind, handle);
    handle
}

fn ensure_dynamic_component_index(world: &mut World, entity: Entity, name: &str) -> Option<usize> {
    if world.get::<DynamicComponents>(entity).is_none() {
        world
            .entity_mut(entity)
            .insert(DynamicComponents::default());
    }
    let mut dynamic = world.get_mut::<DynamicComponents>(entity)?;
    let index = match dynamic
        .components
        .iter()
        .position(|component| component.name == name)
    {
        Some(index) => index,
        None => {
            dynamic
                .components
                .push(DynamicComponent::new(name.to_string()));
            dynamic.components.len() - 1
        }
    };

    Some(index)
}

fn apply_dynamic_fields(component: &mut DynamicComponent, fields: Table) {
    let mut entries = Vec::new();
    for pair in fields.pairs::<Value, Value>() {
        if let Ok((Value::String(key), value)) = pair {
            if let Some(value) = lua_value_to_dynamic(value) {
                entries.push((key.to_string_lossy().to_string(), value));
            }
        }
    }

    for (name, value) in entries {
        if let Some(field) = component.fields.iter_mut().find(|field| field.name == name) {
            field.value = value;
        } else {
            component.fields.push(DynamicField { name, value });
        }
    }
}

fn lua_value_to_dynamic(value: Value) -> Option<DynamicValue> {
    match value {
        Value::Boolean(value) => Some(DynamicValue::Bool(value)),
        Value::Integer(value) => {
            if value >= i32::MIN as i64 && value <= i32::MAX as i64 {
                Some(DynamicValue::Int(value as i32))
            } else {
                Some(DynamicValue::Float(value as f32))
            }
        }
        Value::Number(value) => Some(DynamicValue::Float(value as f32)),
        Value::String(value) => Some(DynamicValue::String(value.to_string_lossy().to_string())),
        Value::Table(table) => {
            table_to_vec3(&table).map(|vec| DynamicValue::Vec3([vec.x, vec.y, vec.z]))
        }
        _ => None,
    }
}

fn dynamic_value_to_lua(lua: &Lua, value: &DynamicValue) -> mlua::Result<Value> {
    match value {
        DynamicValue::Bool(value) => Ok(Value::Boolean(*value)),
        DynamicValue::Float(value) => Ok(Value::Number(*value as f64)),
        DynamicValue::Int(value) => Ok(Value::Integer(*value as i64)),
        DynamicValue::Vec3(value) => {
            let table = lua.create_table()?;
            table.set("x", value[0])?;
            table.set("y", value[1])?;
            table.set("z", value[2])?;
            table.set(1, value[0])?;
            table.set(2, value[1])?;
            table.set(3, value[2])?;
            Ok(Value::Table(table))
        }
        DynamicValue::String(value) => Ok(Value::String(lua.create_string(value)?)),
    }
}
