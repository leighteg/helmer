use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use bevy_ecs::prelude::Resource;
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

const PROJECT_FILE_NAME: &str = "helmer_project.ron";
const DEFAULT_MATERIAL_FILE: &str = "default.ron";
const LUAURC_FILE_NAME: &str = ".luaurc";
const LUAURC_GENERATED_MARKER: &str = "\"__generated_by_helmer__\": true";
const HELMER_LUAU_API_FILE_NAME: &str = "helmer_api.d.luau";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub name: String,
    pub version: u32,
    pub vscode_config_dir: String,
    pub assets_dir: String,
    pub models_dir: String,
    pub textures_dir: String,
    pub materials_dir: String,
    pub scenes_dir: String,
    pub scripts_dir: String,
}

impl ProjectConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            version: 1,
            vscode_config_dir: ".vscode".to_string(),
            assets_dir: "assets".to_string(),
            models_dir: "assets/models".to_string(),
            textures_dir: "assets/textures".to_string(),
            materials_dir: "assets/materials".to_string(),
            scenes_dir: "assets/scenes".to_string(),
            scripts_dir: "assets/scripts".to_string(),
        }
    }

    pub fn vscode_config_root(&self, root: &Path) -> PathBuf {
        root.join(&self.vscode_config_dir)
    }

    pub fn config_path(root: &Path) -> PathBuf {
        root.join(PROJECT_FILE_NAME)
    }

    pub fn assets_root(&self, root: &Path) -> PathBuf {
        root.join(&self.assets_dir)
    }

    pub fn models_root(&self, root: &Path) -> PathBuf {
        root.join(&self.models_dir)
    }

    pub fn textures_root(&self, root: &Path) -> PathBuf {
        root.join(&self.textures_dir)
    }

    pub fn materials_root(&self, root: &Path) -> PathBuf {
        root.join(&self.materials_dir)
    }

    pub fn scenes_root(&self, root: &Path) -> PathBuf {
        root.join(&self.scenes_dir)
    }

    pub fn scripts_root(&self, root: &Path) -> PathBuf {
        root.join(&self.scripts_dir)
    }
}

#[derive(Resource, Default, Clone)]
pub struct EditorProject {
    pub root: Option<PathBuf>,
    pub config: Option<ProjectConfig>,
}

pub fn create_project(root: &Path, name: &str) -> Result<ProjectConfig, String> {
    fs::create_dir_all(root).map_err(|err| err.to_string())?;

    let config = ProjectConfig::new(name.to_string());
    ensure_project_layout(root, &config)?;
    write_project_config(root, &config)?;
    write_default_material(root, &config)?;

    Ok(config)
}

pub fn load_project(root: &Path) -> Result<ProjectConfig, String> {
    let config_path = ProjectConfig::config_path(root);
    let config = if config_path.exists() {
        let data = fs::read_to_string(&config_path).map_err(|err| err.to_string())?;
        ron::de::from_str::<ProjectConfig>(&data).map_err(|err| err.to_string())?
    } else {
        let name = root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("helmer Project")
            .to_string();
        let config = ProjectConfig::new(name);
        write_project_config(root, &config)?;
        config
    };

    ensure_project_layout(root, &config)?;
    write_default_material(root, &config)?;

    Ok(config)
}

pub fn save_project_config(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    ensure_project_layout(root, config)?;
    write_project_config(root, config)?;
    write_default_material(root, config)?;
    Ok(())
}

fn write_project_config(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    let config_path = ProjectConfig::config_path(root);
    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(4)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(config, pretty).map_err(|err| err.to_string())?;
    fs::write(config_path, data).map_err(|err| err.to_string())
}

fn ensure_project_layout(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    fs::create_dir_all(config.assets_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.models_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.textures_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.materials_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.scenes_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.scripts_root(root)).map_err(|err| err.to_string())?;
    write_luau_support_files(root, config)?;
    Ok(())
}

fn write_default_material(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    let materials_root = config.materials_root(root);
    let default_path = materials_root.join(DEFAULT_MATERIAL_FILE);
    if default_path.exists() {
        return Ok(());
    }

    let template = default_material_template();
    fs::write(default_path, template).map_err(|err| err.to_string())
}

fn write_luau_support_files(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    let scripts_root = config.scripts_root(root);
    fs::create_dir_all(&scripts_root).map_err(|err| err.to_string())?;

    let api_path = scripts_root.join(HELMER_LUAU_API_FILE_NAME);
    fs::write(&api_path, helmer_luau_api_types()).map_err(|err| err.to_string())?;

    let luaurc_path = root.join(LUAURC_FILE_NAME);
    let should_write_luaurc = match fs::read_to_string(&luaurc_path) {
        Ok(existing) => existing.contains(LUAURC_GENERATED_MARKER),
        Err(err) if err.kind() == ErrorKind::NotFound => true,
        Err(err) => return Err(err.to_string()),
    };

    if should_write_luaurc {
        fs::write(&luaurc_path, luau_rc_template(config)).map_err(|err| err.to_string())?;

        let vscode_settings_path = config.vscode_config_root(root);
        fs::create_dir_all(&vscode_settings_path).map_err(|err| err.to_string())?;
        fs::write(
            &vscode_settings_path.join("settings.json"),
            luau_vscode_settings_template(),
        )
        .map_err(|err| err.to_string())?;
    }

    Ok(())
}

fn luau_rc_template(config: &ProjectConfig) -> String {
    let scripts_glob = format!(
        "{}/**",
        config
            .scripts_dir
            .trim_start_matches("./")
            .replace('\\', "/")
    );
    format!(
        "{{\n  \"__generated_by_helmer__\": true,\n  \"languageMode\": \"strict\",\n  \"paths\": [\n    \"{}\"\n  ],\n  \"globals\": [\n    \"ecs\",\n    \"input\",\n    \"entity_id\"\n  ]\n}}\n",
        scripts_glob
    )
}

fn luau_vscode_settings_template() -> &'static str {
    r#"{
    "luau-lsp.plugin.enabled": true,
    "luau-lsp.platform.type": "standard",
    "luau-lsp.sourcemap.enabled": false,
    "luau-lsp.sourcemap.autogenerate": false,
    "luau-lsp.types.definitionFiles": {
        "@helmer": "assets/scripts/helmer_api.d.luau",
    }
}"#
}

fn helmer_luau_api_types() -> &'static str {
    r#"-- Generated by helmer_editor. Update by reopening or saving project preferences.

type EntityId = number

type Vec2 = { x: number, y: number, [number]: number }
type Vec3 = { x: number, y: number, z: number, [number]: number }
type Quat = { x: number, y: number, z: number, w: number, [number]: number }

type Transform = {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}

type TransformPatch = {
    position: Vec3?,
    rotation: Quat?,
    scale: Vec3?,
}

type SplineData = {
    points: { Vec3 },
    closed: boolean,
    tension: number,
    mode: string,
}

type SplinePatch = {
    points: { Vec3 }?,
    closed: boolean?,
    tension: number?,
    mode: string?,
}

type LightData = {
    type: string,
    color: Vec3,
    intensity: number,
    angle: number?,
}

type LightPatch = {
    type: string?,
    color: Vec3?,
    intensity: number?,
    angle: number?,
}

type CameraData = {
    fov_y_rad: number,
    aspect_ratio: number,
    near_plane: number,
    far_plane: number,
    active: boolean,
}

type CameraPatch = {
    fov_y_rad: number?,
    aspect_ratio: number?,
    near_plane: number?,
    far_plane: number?,
    active: boolean?,
}

type MeshRendererData = {
    source: string,
    material: string?,
    casts_shadow: boolean,
    visible: boolean,
}

type MeshRendererPatch = {
    source: string?,
    material: string?,
    casts_shadow: boolean?,
    visible: boolean?,
}

type ScriptData = {
    path: string,
    language: string,
}

type DynamicFieldValue = boolean | number | string | Vec3
type DynamicFields = { [string]: DynamicFieldValue }

type DynamicComponentData = {
    name: string,
    fields: DynamicFields,
}

type HelmerEcs = {
    list_entities: () -> { EntityId },
    spawn_entity: (name: string?) -> EntityId,
    entity_exists: (id: EntityId) -> boolean,
    find_entity_by_name: (name: string) -> EntityId?,
    get_entity_name: (id: EntityId) -> string?,
    set_entity_name: (id: EntityId, name: string) -> boolean,
    delete_entity: (id: EntityId) -> boolean,
    has_component: (id: EntityId, component: string) -> boolean,
    add_component: (id: EntityId, component: string) -> boolean,
    remove_component: (id: EntityId, component: string) -> boolean,

    get_transform: (id: EntityId) -> Transform?,
    set_transform: (id: EntityId, data: TransformPatch) -> boolean,

    get_spline: (id: EntityId) -> SplineData?,
    set_spline: (id: EntityId, data: SplinePatch) -> boolean,
    add_spline_point: (id: EntityId, point: Vec3) -> boolean,
    set_spline_point: (id: EntityId, index: number, point: Vec3) -> boolean,
    remove_spline_point: (id: EntityId, index: number) -> boolean,
    sample_spline: (id: EntityId, t: number) -> Vec3?,
    spline_length: (id: EntityId, samples: number?) -> number?,
    follow_spline: (id: EntityId, spline_id: EntityId?, speed: number?, looped: boolean?) -> boolean,

    set_animator_enabled: (id: EntityId, enabled: boolean) -> boolean,
    set_animator_time_scale: (id: EntityId, time_scale: number) -> boolean,
    set_animator_param_float: (id: EntityId, name: string, value: number) -> boolean,
    set_animator_param_bool: (id: EntityId, name: string, value: boolean) -> boolean,
    trigger_animator: (id: EntityId, name: string) -> boolean,
    get_animator_clips: (id: EntityId, layer_index: number?) -> { string }?,
    play_anim_clip: (id: EntityId, name: string, layer_index: number?) -> boolean,

    get_light: (id: EntityId) -> LightData?,
    set_light: (id: EntityId, data: LightPatch) -> boolean,

    get_camera: (id: EntityId) -> CameraData?,
    set_camera: (id: EntityId, data: CameraPatch) -> boolean,
    set_active_camera: (id: EntityId) -> boolean,

    get_mesh_renderer: (id: EntityId) -> MeshRendererData?,
    set_mesh_renderer: (id: EntityId, data: MeshRendererPatch) -> boolean,

    get_scene_asset: (id: EntityId) -> string?,
    set_scene_asset: (id: EntityId, path: string) -> boolean,

    get_script: (id: EntityId) -> ScriptData?,
    set_script: (id: EntityId, path: string, language: string?) -> boolean,

    list_dynamic_components: (id: EntityId) -> { DynamicComponentData }?,
    get_dynamic_component: (id: EntityId, name: string) -> DynamicFields?,
    set_dynamic_component: (id: EntityId, name: string, fields: DynamicFields) -> boolean,
    get_dynamic_field: (id: EntityId, component_name: string, field_name: string) -> DynamicFieldValue?,
    set_dynamic_field: (id: EntityId, component_name: string, field_name: string, value: DynamicFieldValue) -> boolean,
    remove_dynamic_component: (id: EntityId, name: string) -> boolean,
    remove_dynamic_field: (id: EntityId, component_name: string, field_name: string) -> boolean,
}

type InputHandle = any
type InputModifiers = { shift: boolean, ctrl: boolean, alt: boolean, super: boolean }

type HelmerInput = {
    keys: { [string]: InputHandle },
    mouse_buttons: { [string]: InputHandle },
    gamepad_buttons: { [string]: InputHandle },
    gamepad_axes: { [string]: InputHandle },

    key: (name: string) -> InputHandle?,
    mouse_button: (name: string) -> InputHandle?,
    gamepad_button: (name: string) -> InputHandle?,
    gamepad_axis_handle: (name: string) -> InputHandle?,
    gamepad_axis_ref: (name: string) -> InputHandle?,

    key_down: (key: InputHandle | string | number) -> boolean,
    key_pressed: (key: InputHandle | string | number) -> boolean,
    key_released: (key: InputHandle | string | number) -> boolean,

    mouse_down: (button: InputHandle | string | number) -> boolean,
    mouse_pressed: (button: InputHandle | string | number) -> boolean,
    mouse_released: (button: InputHandle | string | number) -> boolean,

    cursor: () -> Vec2?,
    cursor_delta: () -> Vec2?,
    wheel: () -> Vec2?,
    window_size: () -> Vec2?,
    scale_factor: () -> number?,
    modifiers: () -> InputModifiers?,
    wants_keyboard: () -> boolean,
    wants_pointer: () -> boolean,

    gamepad_ids: () -> { number },
    gamepad_count: () -> number,
    gamepad_axis: (axis: InputHandle | string, gamepad_id: number?) -> number,
    gamepad_button_down: (button: InputHandle | string, gamepad_id: number?) -> boolean,
    gamepad_button_pressed: (button: InputHandle | string, gamepad_id: number?) -> boolean,
    gamepad_button_released: (button: InputHandle | string, gamepad_id: number?) -> boolean,
    gamepad_trigger: (side: InputHandle | string, gamepad_id: number?) -> number,
}

declare entity_id: EntityId
declare ecs: HelmerEcs
declare input: HelmerInput
declare print: (...any) -> ()
"#
}

pub fn default_material_template() -> &'static str {
    "(\n    albedo: (1.0, 1.0, 1.0, 1.0),\n    metallic: 0.0,\n    roughness: 0.8,\n    ao: 1.0,\n    emission_strength: 0.0,\n    emission_color: (0.0, 0.0, 0.0),\n    albedo_texture: None,\n    normal_texture: None,\n    metallic_roughness_texture: None,\n    emission_texture: None,\n)\n"
}

pub fn default_scene_template() -> &'static str {
    "(\n    version: 1,\n    entities: [],\n)\n"
}

pub fn default_animation_template() -> &'static str {
    "(\n    version: 1,\n    name: \"New Animation\",\n    tracks: [],\n    clips: [],\n)\n"
}

pub fn default_script_template_simple() -> &'static str {
    r#"--!strict
local mover: number = -1
local t = 0.0
local speed = 10.0

function on_start()
    mover = ecs.spawn_entity("Orbiting Cube")
    ecs.set_mesh_renderer(mover, {
        source = "Cube",
        casts_shadow = true,
        visible = true,
    })
    ecs.set_transform(mover, { position = { x = 0.0, y = 0.5, z = 0.0 } })
end

function on_update(dt: number)
    if mover == nil then return end
    t = t + (dt * speed)
    local radius = 2.0
    local x = math.cos(t) * radius
    local y = math.cos(t / 2) * radius
    local z = math.sin(t) * radius
    ecs.set_transform(mover, { position = { x = x, y = y, z = z } })
end
"#
}

pub fn default_script_template_full() -> &'static str {
    r#"--!strict
-- helmer Luau API
-- Entry points: on_start(), on_update(dt), on_stop()
-- Globals:
--   entity_id : u64 id of the entity that owns this script
--   print(...) : log values to the editor console
-- input table:
--   input.key_down(key) -> bool
--   input.key_pressed(key) -> bool
--   input.key_released(key) -> bool
--   input.mouse_down(button) -> bool
--   input.mouse_pressed(button) -> bool
--   input.mouse_released(button) -> bool
--   input.cursor() -> {x,y}
--   input.cursor_delta() -> {x,y}
--   input.wheel() -> {x,y}
--   input.window_size() -> {x,y}
--   input.scale_factor() -> number
--   input.modifiers() -> {shift, ctrl, alt, super}
--   input.wants_keyboard() -> bool
--   input.wants_pointer() -> bool
--   input.gamepad_ids() -> {id, ...}
--   input.gamepad_count() -> number
--   input.gamepad_axis(axis, id?) -> number
--   input.gamepad_button_down(button, id?) -> bool
--   input.gamepad_button_pressed(button, id?) -> bool
--   input.gamepad_button_released(button, id?) -> bool
--   input.gamepad_trigger(side, id?) -> number
--   input.key(name) -> key|nil
--   input.mouse_button(name) -> button|nil
--   input.gamepad_button(name) -> button|nil
--   input.gamepad_axis(name) -> axis|nil
-- input constants:
--   input.keys.<Name>, input.mouse_buttons.<Name>, input.gamepad_buttons.<Name>, input.gamepad_axes.<Name>
-- ecs table:
--   ecs.list_entities() -> {id, ...}
--   ecs.spawn_entity(name?) -> id
--   ecs.entity_exists(id) -> bool
--   ecs.find_entity_by_name(name) -> id|nil
--   ecs.get_entity_name(id) -> string|nil
--   ecs.set_entity_name(id, name) -> bool (empty name removes Name)
--   ecs.delete_entity(id) -> bool
--   ecs.has_component(id, "name"|"transform"|"camera"|"light"|"mesh"|"mesh_renderer"|"scene"|"script"|"dynamic") -> bool
--   ecs.add_component(id, "transform"|"camera"|"light"|"dynamic") -> bool
--   ecs.remove_component(id, "name"|"transform"|"camera"|"light"|"mesh"|"mesh_renderer"|"scene"|"script"|"dynamic") -> bool
--   ecs.get_transform(id) -> {position={x,y,z}, rotation={x,y,z,w}, scale={x,y,z}}|nil
--   ecs.set_transform(id, {position?, rotation?, scale?}) -> bool
--   ecs.get_spline(id) -> {points={{x,y,z}}, closed, tension, mode}|nil
--   ecs.set_spline(id, {points?, closed?, tension?, mode?}) -> bool
--   ecs.add_spline_point(id, point) -> bool
--   ecs.set_spline_point(id, index, point) -> bool
--   ecs.remove_spline_point(id, index) -> bool
--   ecs.sample_spline(id, t) -> point|nil
--   ecs.spline_length(id, samples?) -> number|nil
--   ecs.follow_spline(id, spline_id?, speed?, looped?) -> bool
--   ecs.set_animator_enabled(id, enabled) -> bool
--   ecs.set_animator_time_scale(id, value) -> bool
--   ecs.set_animator_param_float(id, name, value) -> bool
--   ecs.set_animator_param_bool(id, name, value) -> bool
--   ecs.trigger_animator(id, name) -> bool
--   ecs.get_animator_clips(id, layer?) -> {name, ...}|nil
--   ecs.play_anim_clip(id, name, layer?) -> bool
--   ecs.get_light(id) -> {type, color={x,y,z}, intensity, angle?}|nil
--   ecs.set_light(id, {type?, color?, intensity?, angle?}) -> bool
--   ecs.get_camera(id) -> {fov_y_rad, aspect_ratio, near_plane, far_plane, active}|nil
--   ecs.set_camera(id, {fov_y_rad?, aspect_ratio?, near_plane?, far_plane?, active?}) -> bool
--   ecs.set_active_camera(id) -> bool
--   ecs.get_mesh_renderer(id) -> {source, material?, casts_shadow, visible}|nil
--   ecs.set_mesh_renderer(id, {source?, material?, casts_shadow?, visible?}) -> bool
--   ecs.get_scene_asset(id) -> path|nil
--   ecs.set_scene_asset(id, path) -> bool
--   ecs.get_script(id) -> {path, language}|nil
--   ecs.set_script(id, path, language?) -> bool
--   ecs.list_dynamic_components(id) -> [{name, fields={...}}, ...]|nil
--   ecs.get_dynamic_component(id, name) -> fields|nil
--   ecs.set_dynamic_component(id, name, fields_table) -> bool
--   ecs.get_dynamic_field(id, comp_name, field_name) -> value|nil
--   ecs.set_dynamic_field(id, comp_name, field_name, value) -> bool
--   ecs.remove_dynamic_component(id, name) -> bool
--   ecs.remove_dynamic_field(id, comp_name, field_name) -> bool

local mover: number = -1
local t = 0.0
local origin = { x = 0.0, y = 0.0, z = 0.0 }

local speed = 5.0

function on_start()
    local owner_transform = ecs.get_transform(entity_id)
    if owner_transform ~= nil and owner_transform.position ~= nil then
        origin.x = owner_transform.position.x or 0.0
        origin.y = owner_transform.position.y or 0.0
        origin.z = owner_transform.position.z or 0.0
    end

    mover = ecs.spawn_entity("Orbiting Cube")
    ecs.set_mesh_renderer(mover, {
        source = "Cube",
        casts_shadow = true,
        visible = true,
    })
    ecs.set_transform(mover, { position = { x = origin.x, y = origin.y, z = origin.z } })
end

function on_update(dt: number)
    if mover == nil then return end

    local owner_transform = ecs.get_transform(entity_id)
    if owner_transform ~= nil and owner_transform.position ~= nil then
        origin.x = owner_transform.position.x or 0.0
        origin.y = owner_transform.position.y or 0.0
        origin.z = owner_transform.position.z or 0.0
    end

    t = t + (dt * speed)
    local radius = 2.0
    local x = origin.x + math.cos(t) * radius
    local y = origin.y + math.cos(t / 2) * radius
    local z = origin.z + math.sin(t) * radius
    ecs.set_transform(mover, { position = { x = x, y = y, z = z } })
end
"#
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RecentProjectsFile {
    paths: Vec<String>,
}

pub fn load_recent_projects() -> Vec<PathBuf> {
    let Some(path) = recent_projects_path() else {
        return Vec::new();
    };

    let data = match fs::read_to_string(&path) {
        Ok(data) => data,
        Err(_) => return Vec::new(),
    };

    let parsed = ron::de::from_str::<RecentProjectsFile>(&data).unwrap_or_default();
    parsed
        .paths
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .collect()
}

pub fn save_recent_projects(paths: &[PathBuf]) -> Result<(), String> {
    let Some(path) = recent_projects_path() else {
        return Ok(());
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }

    let payload = RecentProjectsFile {
        paths: paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
    };

    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(4)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(&payload, pretty).map_err(|err| err.to_string())?;
    fs::write(path, data).map_err(|err| err.to_string())
}

fn recent_projects_path() -> Option<PathBuf> {
    let home = env::var("HOME").or_else(|_| env::var("USERPROFILE")).ok()?;
    Some(
        PathBuf::from(home)
            .join(".helmer_editor")
            .join("recent_projects.ron"),
    )
}
