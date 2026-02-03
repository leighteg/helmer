use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Component, Entity, Resource, World};
use bevy_ecs::query::With;
use glam::{DVec2, Quat, Vec2, Vec3};
use mlua::{Function, Lua, RegistryKey, Table, UserData, Value, Variadic};
use winit::{event::MouseButton, keyboard::KeyCode};

use helmer::provided::components::{
    EntityFollower, Light, LightType, LookAt, MeshAsset, MeshRenderer, Spline, SplineFollower,
    SplineMode,
};
use helmer::runtime::asset_server::{Handle, Material, Mesh};
use helmer::runtime::input_manager::InputManager;
use helmer_becs::systems::scene_system::SceneRoot;
use helmer_becs::{
    BevyActiveCamera, BevyAnimator, BevyAssetServer, BevyCamera, BevyEntityFollower,
    BevyInputManager, BevyLight, BevyLookAt, BevyMeshRenderer, BevySpline, BevySplineFollower,
    BevyTransform, BevyWrapper, DeltaTime,
};

use crate::editor::{
    EditorPlayCamera, activate_play_camera,
    assets::{
        EditorAssetCache, EditorMesh, MeshSource, PrimitiveKind, SceneAssetPath,
        cached_scene_handle,
    },
    dynamic::{DynamicComponent, DynamicComponents, DynamicField, DynamicValue},
    project::EditorProject,
    scene::{EditorEntity, EditorSceneState, WorldState},
    set_play_camera,
};

#[derive(Debug, Clone)]
pub struct ScriptEntry {
    pub path: Option<PathBuf>,
    pub language: String,
}

impl ScriptEntry {
    pub fn new() -> Self {
        Self {
            path: None,
            language: "lua".to_string(),
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

pub fn load_script_asset(path: &Path) -> ScriptAsset {
    let modified = fs::metadata(path)
        .and_then(|meta| meta.modified())
        .unwrap_or_else(|_| SystemTime::now());

    match fs::read_to_string(path) {
        Ok(source) => ScriptAsset {
            source,
            modified,
            error: None,
        },
        Err(err) => ScriptAsset {
            source: String::new(),
            modified,
            error: Some(err.to_string()),
        },
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

#[derive(Resource)]
pub struct ScriptRuntime {
    pub lua: Arc<Mutex<Lua>>,
    pub instances: HashMap<ScriptInstanceKey, ScriptInstance>,
    pub errors: Vec<String>,
}

impl Default for ScriptRuntime {
    fn default() -> Self {
        Self {
            lua: Arc::new(Mutex::new(Lua::new())),
            instances: HashMap::new(),
            errors: Vec::new(),
        }
    }
}

#[derive(Resource)]
pub struct ScriptRunState {
    pub last_state: WorldState,
}

impl Default for ScriptRunState {
    fn default() -> Self {
        Self {
            last_state: WorldState::Edit,
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
                let asset = script_assets
                    .get(path)
                    .cloned()
                    .unwrap_or_else(|| load_script_asset(path));
                entries.push((
                    ScriptInstanceKey {
                        entity,
                        script_index,
                    },
                    script.clone(),
                    asset,
                ));
            }
        }
        entries
    };

    let input_manager = world
        .get_resource::<BevyInputManager>()
        .map(|input| input.0.clone());

    if let Some(input_manager) = input_manager.as_ref() {
        let input_manager = input_manager.read();
        if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
            input_state.refresh_gamepad_cache(&input_manager);
        }
    }

    let last_state = world
        .get_resource::<ScriptRunState>()
        .map(|state| state.last_state)
        .unwrap_or(WorldState::Edit);

    world.resource_scope::<ScriptRuntime, _>(|world, mut runtime| {
        let lua = runtime.lua.clone();
        let lua = match lua.lock() {
            Ok(lua) => lua,
            Err(_) => return,
        };

        let world_ptr = world as *mut World as usize;

        if last_state != current_state {
            if let Some(input_manager) = input_manager.as_ref() {
                let input_manager = input_manager.read();
                if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
                    input_state.reset(Some(&input_manager));
                }
            } else if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
                input_state.reset(None);
            }

            match (last_state, current_state) {
                (WorldState::Edit, WorldState::Play) => {
                    runtime.instances.clear();
                    runtime.errors.clear();
                    for (key, script, asset) in &scripts {
                        if asset.error.is_some() {
                            continue;
                        }
                        if let Some(instance) = load_script_instance(
                            &lua,
                            world_ptr,
                            key.entity,
                            key.script_index,
                            script,
                            asset,
                        ) {
                            runtime.instances.insert(*key, instance);
                            let _ =
                                call_script_function0(&lua, &runtime.instances[key], "on_start");
                        }
                    }
                }
                (WorldState::Play, WorldState::Edit) => {
                    let drained = runtime.instances.drain().collect::<Vec<_>>();
                    for (key, instance) in drained {
                        let _ = call_script_function0(&lua, &instance, "on_stop");
                        let _ = lua.remove_registry_value(instance.env_key);
                        despawn_script_owned(world, key.entity, key.script_index);
                    }
                    runtime.errors.clear();
                }
                _ => {}
            }
        }

        if current_state != WorldState::Play {
            if last_state != current_state {
                if let Some(mut run_state) = world.get_resource_mut::<ScriptRunState>() {
                    run_state.last_state = current_state;
                }
            }
            return;
        }

        let mut active_scripts = HashSet::new();
        for (key, script, asset) in &scripts {
            active_scripts.insert(*key);
            if asset.error.is_some() {
                continue;
            }

            let reload = match runtime.instances.get(key) {
                Some(instance) => script_needs_reload(instance, script, asset),
                None => true,
            };

            if reload {
                if let Some(old_instance) = runtime.instances.remove(key) {
                    let _ = call_script_function0(&lua, &old_instance, "on_stop");
                    let _ = lua.remove_registry_value(old_instance.env_key);
                    despawn_script_owned(world, key.entity, key.script_index);
                }

                if let Some(instance) = load_script_instance(
                    &lua,
                    world_ptr,
                    key.entity,
                    key.script_index,
                    script,
                    asset,
                ) {
                    runtime.instances.insert(*key, instance);
                    let _ = call_script_function0(&lua, &runtime.instances[key], "on_start");
                }
            }

            if let Some(instance) = runtime.instances.get(key) {
                if let Err(err) = call_script_function1(&lua, instance, "on_update", dt) {
                    runtime.errors.push(err);
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

        if last_state != current_state {
            if let Some(mut run_state) = world.get_resource_mut::<ScriptRunState>() {
                run_state.last_state = current_state;
            }
        }
    });

    if let Some(input_manager) = input_manager {
        let input_manager = input_manager.read();
        if let Some(mut input_state) = world.get_resource_mut::<ScriptInputState>() {
            input_state.sync_last_state(&input_manager);
        }
    }
}

fn script_needs_reload(
    instance: &ScriptInstance,
    script: &ScriptEntry,
    asset: &ScriptAsset,
) -> bool {
    let Some(path) = script.path.as_ref() else {
        return true;
    };
    if instance.path != *path {
        return true;
    }

    asset.modified > instance.modified
}

fn load_script_instance(
    lua: &Lua,
    world_ptr: usize,
    entity: Entity,
    script_index: usize,
    script: &ScriptEntry,
    asset: &ScriptAsset,
) -> Option<ScriptInstance> {
    if asset.error.is_some() {
        return None;
    }

    let path = script.path.as_ref()?;
    let env = lua.create_table().ok()?;
    let _ = env.set("entity_id", entity.to_bits());
    if register_script_api(lua, &env, world_ptr, entity, script_index).is_err() {
        return None;
    }
    if apply_lua_globals_fallback(lua, &env).is_err() {
        return None;
    }

    let env_key = lua.create_registry_value(env.clone()).ok()?;

    let chunk = lua
        .load(&asset.source)
        .set_name(path.to_string_lossy().to_string())
        .set_environment(env);
    if chunk.exec().is_err() {
        let _ = lua.remove_registry_value(env_key);
        return None;
    }

    Some(ScriptInstance {
        path: path.clone(),
        env_key,
        modified: asset.modified,
    })
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
    func.call::<()>(()).map_err(|err| err.to_string())
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
    func.call::<()>((dt,)).map_err(|err| err.to_string())
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
        tracing::info!("{}", parts.join(" "));
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
                "script" => world
                    .get::<ScriptComponent>(entity)
                    .map(|scripts| !scripts.scripts.is_empty())
                    .unwrap_or(false),
                "dynamic" => world.get::<DynamicComponents>(entity).is_some(),
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
                "dynamic" => {
                    world
                        .entity_mut(entity)
                        .insert(DynamicComponents::default());
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
                    world.entity_mut(entity).remove::<SceneRoot>();
                    world.entity_mut(entity).remove::<SceneAssetPath>();
                }
                "script" => {
                    world.entity_mut(entity).remove::<ScriptComponent>();
                }
                "dynamic" => {
                    world.entity_mut(entity).remove::<DynamicComponents>();
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
            world.entity_mut(entity).insert(SceneRoot(handle));
            world
                .entity_mut(entity)
                .insert(SceneAssetPath { path: resolved });
            Ok(true)
        })?,
    )?;

    let world_ptr_get_script = world_ptr;
    ecs.set(
        "get_script",
        lua.create_function(move |lua, entity_id: u64| {
            let world = unsafe { &mut *(world_ptr_get_script as *mut World) };
            let Some(entity) = lookup_editor_entity(world, entity_id) else {
                return Ok(None);
            };
            let Some(scripts) = world.get::<ScriptComponent>(entity) else {
                return Ok(None);
            };
            let Some(script) = scripts.scripts.first() else {
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
                let language = language.unwrap_or_else(|| "lua".to_string());
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

    Ok(ecs)
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

    input.set(
        "gamepad_axis",
        lua.create_function(|lua, name: String| {
            let Some(axis) = parse_gamepad_axis_name(&name) else {
                return Ok(None);
            };
            Ok(Some(lua.create_userdata(LuaGamepadAxis(axis))?))
        })?,
    )?;

    let world_ptr_key = world_ptr;
    input.set(
        "key_down",
        lua.create_function(move |_, key: Value| {
            let world = unsafe { &mut *(world_ptr_key as *mut World) };
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
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
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
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
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
            let Some(input_manager) = world.get_resource::<BevyInputManager>() else {
                return Ok(None);
            };
            let input_manager = input_manager.0.read();
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
            if input_manager.egui_wants_key {
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
            Ok(input_manager.egui_wants_key)
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
            Ok(input_manager.egui_wants_pointer)
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
