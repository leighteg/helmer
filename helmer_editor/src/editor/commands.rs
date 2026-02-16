use std::path::PathBuf;

use bevy_ecs::prelude::{Entity, Resource};

use crate::editor::assets::PrimitiveKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetCreateKind {
    Folder,
    Scene,
    Material,
    Script,
    VisualScript,
    VisualScriptThirdPerson,
    RustScript,
    Animation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnKind {
    Empty,
    Camera,
    FreecamCamera,
    DirectionalLight,
    PointLight,
    SpotLight,
    Primitive(PrimitiveKind),
    DynamicBodyCuboid,
    DynamicBodySphere,
    FixedColliderCuboid,
    FixedColliderSphere,
    SceneAsset(PathBuf),
    MeshAsset(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorCommand {
    CreateProject {
        name: String,
        path: PathBuf,
    },
    OpenProject {
        path: PathBuf,
    },
    NewScene,
    OpenScene {
        path: PathBuf,
    },
    SaveScene,
    SaveSceneAs {
        path: PathBuf,
    },
    CreateEntity {
        kind: SpawnKind,
    },
    ImportAsset {
        source_path: PathBuf,
        destination_dir: Option<PathBuf>,
    },
    CreateAsset {
        directory: PathBuf,
        name: String,
        kind: AssetCreateKind,
    },
    DeleteEntity {
        entity: Entity,
    },
    SetActiveCamera {
        entity: Entity,
    },
    TogglePlayMode,
    Undo,
    Redo,
    CloseProject,
}

#[derive(Resource, Default)]
pub struct EditorCommandQueue {
    pub commands: Vec<EditorCommand>,
}

impl EditorCommandQueue {
    pub fn push(&mut self, command: EditorCommand) {
        self.commands.push(command);
    }
}
