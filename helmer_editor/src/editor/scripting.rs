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
use glam::{Quat, Vec3};
use mlua::{Function, Lua, RegistryKey, Table, Value, Variadic};

use helmer::provided::components::{Light, LightType, MeshAsset, MeshRenderer};
use helmer::runtime::asset_server::{Handle, Material, Mesh};
use helmer_becs::systems::scene_system::SceneRoot;
use helmer_becs::{
    BevyActiveCamera, BevyAssetServer, BevyCamera, BevyLight, BevyMeshRenderer, BevyTransform,
    BevyWrapper, DeltaTime,
};

use crate::editor::{
    EditorPlayCamera, activate_play_camera,
    assets::{EditorAssetCache, EditorMesh, MeshSource, PrimitiveKind, SceneAssetPath},
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
            let handle = {
                let Some(asset_server) = world.get_resource::<BevyAssetServer>() else {
                    return Ok(false);
                };
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
