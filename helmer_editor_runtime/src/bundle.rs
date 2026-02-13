use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::project::ProjectConfig;

pub const BUILD_LAUNCH_MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildLaunchManifest {
    pub version: u32,
    pub created_unix_ms: u64,
    pub project_name: String,
    pub project_config: ProjectConfig,
    pub startup_scene: String,
    pub pack_manifest: String,
    pub pack_key: String,
    pub key_fingerprint: String,
}

impl BuildLaunchManifest {
    pub fn resolve_pack_manifest_path(&self, manifest_path: &Path) -> PathBuf {
        resolve_manifest_relative_path(manifest_path, &self.pack_manifest)
    }
}

pub fn resolve_manifest_relative_path(manifest_path: &Path, value: &str) -> PathBuf {
    let candidate = PathBuf::from(value);
    if candidate.is_absolute() {
        candidate
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(candidate)
    }
}
