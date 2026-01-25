use std::{
    env, fs,
    path::{Path, PathBuf},
};

use bevy_ecs::prelude::Resource;
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

const PROJECT_FILE_NAME: &str = "helmer_project.ron";
const DEFAULT_MATERIAL_FILE: &str = "default.ron";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub name: String,
    pub version: u32,
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
            assets_dir: "assets".to_string(),
            models_dir: "assets/models".to_string(),
            textures_dir: "assets/textures".to_string(),
            materials_dir: "assets/materials".to_string(),
            scenes_dir: "assets/scenes".to_string(),
            scripts_dir: "assets/scripts".to_string(),
        }
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

pub fn default_material_template() -> &'static str {
    "(\n    albedo: (1.0, 1.0, 1.0, 1.0),\n    metallic: 0.0,\n    roughness: 0.8,\n    ao: 1.0,\n    emission_strength: 0.0,\n    emission_color: (0.0, 0.0, 0.0),\n    albedo_texture: None,\n    normal_texture: None,\n    metallic_roughness_texture: None,\n    emission_texture: None,\n)\n"
}

pub fn default_scene_template() -> &'static str {
    "(\n    version: 1,\n    entities: [],\n)\n"
}

pub fn default_script_template_simple() -> &'static str {
    r#"local mover = nil
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

function on_update(dt)
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
    r#"-- helmer Lua API (editor)
-- Entry points: on_start(), on_update(dt)
-- Globals:
--   entity_id : u64 id of the entity that owns this script
--   print(...) : log values to the editor console
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

local mover = nil
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

function on_update(dt)
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
