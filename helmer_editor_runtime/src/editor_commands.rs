use std::path::PathBuf;

use helmer_becs::components::MeshAsset;
use helmer_becs::ecs::prelude::{Entity, Resource};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveKind {
    Cube,
    UvSphere(u32, u32),
    Icosphere(u32),
    Cylinder(u32, u32),
    Capsule(u32, u32),
    Plane,
}

impl PrimitiveKind {
    pub const DEFAULT_UV_SPHERE_SEGMENTS: u32 = 12;
    pub const DEFAULT_UV_SPHERE_RINGS: u32 = 12;
    pub const DEFAULT_ICOSPHERE_SUBDIVISIONS: u32 = 2;
    pub const DEFAULT_CYLINDER_RADIAL_SEGMENTS: u32 = 24;
    pub const DEFAULT_CYLINDER_HEIGHT_SEGMENTS: u32 = 1;
    pub const DEFAULT_CAPSULE_SEGMENTS: u32 = 24;
    pub const DEFAULT_CAPSULE_RINGS: u32 = 8;

    pub const fn default_uv_sphere() -> Self {
        Self::UvSphere(
            Self::DEFAULT_UV_SPHERE_SEGMENTS,
            Self::DEFAULT_UV_SPHERE_RINGS,
        )
    }

    pub const fn default_icosphere() -> Self {
        Self::Icosphere(Self::DEFAULT_ICOSPHERE_SUBDIVISIONS)
    }

    pub const fn default_cylinder() -> Self {
        Self::Cylinder(
            Self::DEFAULT_CYLINDER_RADIAL_SEGMENTS,
            Self::DEFAULT_CYLINDER_HEIGHT_SEGMENTS,
        )
    }

    pub const fn default_capsule() -> Self {
        Self::Capsule(Self::DEFAULT_CAPSULE_SEGMENTS, Self::DEFAULT_CAPSULE_RINGS)
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Cube => "Cube",
            Self::UvSphere(_, _) => "UV Sphere",
            Self::Icosphere(_) => "Icosphere",
            Self::Cylinder(_, _) => "Cylinder",
            Self::Capsule(_, _) => "Capsule",
            Self::Plane => "Plane",
        }
    }

    pub fn from_source_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "cube" => Some(Self::Cube),
            "uv sphere" | "uv_sphere" | "uvsphere" => Some(Self::default_uv_sphere()),
            "icosphere" | "ico sphere" | "ico_sphere" => Some(Self::default_icosphere()),
            "cylinder" => Some(Self::default_cylinder()),
            "capsule" => Some(Self::default_capsule()),
            "plane" => Some(Self::Plane),
            _ => None,
        }
    }

    pub fn to_mesh_asset(self) -> MeshAsset {
        match self {
            Self::Cube => MeshAsset::cube("cube".to_string()),
            Self::UvSphere(segments, rings) => {
                MeshAsset::uv_sphere("uv sphere".to_string(), segments.max(3), rings.max(3))
            }
            Self::Icosphere(subdivisions) => {
                MeshAsset::icosphere("icosphere".to_string(), subdivisions.min(6))
            }
            Self::Cylinder(radial_segments, height_segments) => MeshAsset::cylinder(
                "cylinder".to_string(),
                radial_segments.max(3),
                height_segments.max(1),
            ),
            Self::Capsule(segments, rings) => {
                MeshAsset::capsule("capsule".to_string(), segments.max(3), rings.max(2))
            }
            Self::Plane => MeshAsset::plane("plane".to_string()),
        }
    }
}

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
