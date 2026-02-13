use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

pub const PROJECT_FILE_NAME: &str = "helmer_project.ron";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    #[serde(default = "default_project_name")]
    pub name: String,
    #[serde(default = "default_project_version")]
    pub version: u32,
    #[serde(default)]
    pub startup_scene: Option<String>,
    #[serde(default = "default_vscode_config_dir")]
    pub vscode_config_dir: String,
    #[serde(default = "default_assets_dir")]
    pub assets_dir: String,
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
    #[serde(default = "default_textures_dir")]
    pub textures_dir: String,
    #[serde(default = "default_materials_dir")]
    pub materials_dir: String,
    #[serde(default = "default_scenes_dir")]
    pub scenes_dir: String,
    #[serde(default = "default_scripts_dir")]
    pub scripts_dir: String,
}

impl ProjectConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            version: default_project_version(),
            startup_scene: None,
            vscode_config_dir: default_vscode_config_dir(),
            assets_dir: default_assets_dir(),
            models_dir: default_models_dir(),
            textures_dir: default_textures_dir(),
            materials_dir: default_materials_dir(),
            scenes_dir: default_scenes_dir(),
            scripts_dir: default_scripts_dir(),
        }
    }

    pub fn config_path(root: &Path) -> PathBuf {
        root.join(PROJECT_FILE_NAME)
    }

    pub fn vscode_config_root(&self, root: &Path) -> PathBuf {
        root.join(&self.vscode_config_dir)
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

#[derive(Debug, Clone)]
pub struct ProjectLayout {
    pub root: PathBuf,
    pub assets_root: PathBuf,
    pub models_root: PathBuf,
    pub textures_root: PathBuf,
    pub materials_root: PathBuf,
    pub scenes_root: PathBuf,
    pub scripts_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ProjectDescriptor {
    pub root: PathBuf,
    pub config: ProjectConfig,
    pub layout: ProjectLayout,
}

impl ProjectDescriptor {
    pub fn from_root(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        let config = load_project_config(&root)?;
        let layout = resolve_project_layout(&root, &config);
        Ok(Self {
            root,
            config,
            layout,
        })
    }
}

pub fn resolve_project_layout(root: &Path, config: &ProjectConfig) -> ProjectLayout {
    ProjectLayout {
        root: root.to_path_buf(),
        assets_root: config.assets_root(root),
        models_root: config.models_root(root),
        textures_root: config.textures_root(root),
        materials_root: config.materials_root(root),
        scenes_root: config.scenes_root(root),
        scripts_root: config.scripts_root(root),
    }
}

pub fn load_project_config(root: &Path) -> Result<ProjectConfig, String> {
    let config_path = ProjectConfig::config_path(root);
    if config_path.exists() {
        let raw = fs::read_to_string(&config_path).map_err(|err| err.to_string())?;
        return ron::de::from_str::<ProjectConfig>(&raw).map_err(|err| err.to_string());
    }

    let name = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("helmer Project")
        .to_string();
    Ok(ProjectConfig::new(name))
}

pub fn normalize_asset_relative_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn default_project_name() -> String {
    "helmer Project".to_string()
}

fn default_project_version() -> u32 {
    1
}

fn default_vscode_config_dir() -> String {
    ".vscode".to_string()
}

fn default_assets_dir() -> String {
    "assets".to_string()
}

fn default_models_dir() -> String {
    "assets/models".to_string()
}

fn default_textures_dir() -> String {
    "assets/textures".to_string()
}

fn default_materials_dir() -> String {
    "assets/materials".to_string()
}

fn default_scenes_dir() -> String {
    "assets/scenes".to_string()
}

fn default_scripts_dir() -> String {
    "assets/scripts".to_string()
}
