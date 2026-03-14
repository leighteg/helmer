use std::path::{Path, PathBuf};

use helmer_becs::ecs::prelude::{Component, Resource};

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EditorEntity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldState {
    Edit,
    Play,
}

#[derive(Resource, Debug, Clone)]
pub struct EditorSceneState<P = ()> {
    pub path: Option<PathBuf>,
    pub name: String,
    pub dirty: bool,
    pub world_state: WorldState,
    pub play_backup: Option<P>,
    pub play_selected_index: Option<usize>,
}

impl<P> Default for EditorSceneState<P> {
    fn default() -> Self {
        Self {
            path: None,
            name: "Untitled".to_string(),
            dirty: false,
            world_state: WorldState::Edit,
            play_backup: None,
            play_selected_index: None,
        }
    }
}

#[derive(Resource, Debug, Default, Clone)]
pub struct EditorRenderRefresh {
    pub pending: bool,
}

pub fn scene_display_name(path: &Path) -> String {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Scene");
    if let Some(stripped) = file_name.strip_suffix(".hscene.ron") {
        return stripped.to_string();
    }
    path.file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Scene")
        .to_string()
}
