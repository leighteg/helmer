use std::{
    fs,
    path::{Path, PathBuf},
};

use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Component, Entity, Resource, World};
use bevy_ecs::query::{With, Without};
use helmer::{
    provided::components::{Camera, Light, LightType, MeshRenderer, Transform},
    runtime::asset_server::Handle,
};
use helmer_becs::physics::components::{ColliderShape, DynamicRigidBody, FixedCollider};
use helmer_becs::{
    BevyCamera, BevyLight, BevyMeshRenderer, BevyTransform, BevyWrapper,
    systems::scene_system::SceneRoot,
};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

use crate::editor::{
    EditorPlayCamera, EditorViewportCamera, Freecam,
    assets::{EditorAssetCache, EditorMesh, MeshSource, PrimitiveKind, SceneAssetPath},
    dynamic::{DynamicComponent, DynamicComponents},
    project::EditorProject,
    scripting::{ScriptComponent, ScriptEntry},
};

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EditorEntity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldState {
    Edit,
    Play,
}

#[derive(Resource, Debug, Clone)]
pub struct EditorSceneState {
    pub path: Option<PathBuf>,
    pub name: String,
    pub dirty: bool,
    pub world_state: WorldState,
    pub play_backup: Option<SceneDocument>,
    pub play_selected_index: Option<usize>,
}

impl Default for EditorSceneState {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDocument {
    pub version: u32,
    pub entities: Vec<SceneEntityData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntityData {
    pub name: Option<String>,
    pub transform: SerializedTransform,
    pub components: SceneComponents,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SceneComponents {
    pub mesh: Option<MeshComponentData>,
    pub light: Option<LightComponentData>,
    pub camera: Option<CameraComponentData>,
    pub scene: Option<SceneAssetData>,
    pub scripts: Vec<ScriptComponentData>,
    pub dynamic: Vec<DynamicComponent>,
    #[serde(default)]
    pub freecam: bool,
    #[serde(default)]
    pub physics: Option<PhysicsComponentData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsComponentData {
    pub collider_shape: SceneColliderShape,
    pub body_kind: PhysicsBodyKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsBodyKind {
    Dynamic { mass: f32 },
    Fixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneColliderShape {
    Cuboid,
    Sphere,
}

impl From<ColliderShape> for SceneColliderShape {
    fn from(shape: ColliderShape) -> Self {
        match shape {
            ColliderShape::Cuboid => SceneColliderShape::Cuboid,
            ColliderShape::Sphere => SceneColliderShape::Sphere,
        }
    }
}

impl SceneColliderShape {
    pub fn to_collider_shape(&self) -> ColliderShape {
        match self {
            SceneColliderShape::Cuboid => ColliderShape::Cuboid,
            SceneColliderShape::Sphere => ColliderShape::Sphere,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshComponentData {
    pub source: MeshSource,
    pub material: Option<String>,
    pub casts_shadow: bool,
    pub visible: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LightKind {
    Directional,
    Point,
    Spot { angle: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightComponentData {
    pub kind: LightKind,
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraComponentData {
    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAssetData {
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptComponentData {
    pub path: String,
    pub language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTransform {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl From<&Transform> for SerializedTransform {
    fn from(transform: &Transform) -> Self {
        Self {
            position: transform.position.to_array(),
            rotation: transform.rotation.to_array(),
            scale: transform.scale.to_array(),
        }
    }
}

impl SerializedTransform {
    pub fn to_transform(&self) -> Transform {
        Transform {
            position: glam::Vec3::from_array(self.position),
            rotation: glam::Quat::from_array(self.rotation),
            scale: glam::Vec3::from_array(self.scale),
        }
    }
}

pub fn reset_editor_scene(world: &mut World) {
    let entities: Vec<Entity> = world
        .query_filtered::<Entity, With<EditorEntity>>()
        .iter(world)
        .collect();

    for entity in entities {
        world.despawn(entity);
    }
}

pub fn spawn_default_camera(world: &mut World) -> Entity {
    world
        .spawn((
            EditorEntity,
            BevyTransform::default(),
            BevyCamera::default(),
            EditorPlayCamera,
            Name::new("Scene Camera"),
        ))
        .id()
}

pub fn spawn_default_light(world: &mut World) -> Entity {
    world
        .spawn((
            EditorEntity,
            BevyWrapper(Transform {
                position: glam::Vec3::new(0.0, 4.0, 0.0),
                rotation: glam::Quat::from_euler(
                    glam::EulerRot::YXZ,
                    20.0f32.to_radians(),
                    -50.0f32.to_radians(),
                    20.0f32.to_radians(),
                ),
                scale: glam::Vec3::ONE,
            }),
            BevyWrapper(Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0)),
            Name::new("Directional Light"),
        ))
        .id()
}

pub fn ensure_active_camera(world: &mut World) {
    let mut active_found = false;
    for _ in world
        .query::<(&BevyCamera, &EditorPlayCamera)>()
        .iter(world)
    {
        active_found = true;
        break;
    }

    if active_found {
        return;
    }

    if let Some((entity, _)) = world
        .query_filtered::<(Entity, &BevyCamera), Without<EditorViewportCamera>>()
        .iter(world)
        .next()
    {
        world.entity_mut(entity).insert(EditorPlayCamera);
    }
}

pub fn serialize_scene(world: &mut World, project: &EditorProject) -> (SceneDocument, Vec<Entity>) {
    let root = project.root.as_deref();

    let mut entities = Vec::new();
    let mut entity_order = Vec::new();
    let mut query = world.query_filtered::<(
        Entity,
        Option<&Name>,
        Option<&BevyTransform>,
        Option<&BevyMeshRenderer>,
        Option<&EditorMesh>,
        Option<&BevyLight>,
        Option<&BevyCamera>,
        Option<&EditorPlayCamera>,
        Option<&SceneRoot>,
        Option<&SceneAssetPath>,
        Option<&ScriptComponent>,
    ), With<EditorEntity>>();

    for (
        entity,
        name,
        transform,
        mesh_renderer,
        editor_mesh,
        light,
        camera,
        active_camera,
        scene_root,
        scene_asset,
        script,
    ) in query.iter(world)
    {
        let transform = transform.map(|t| t.0).unwrap_or_default();
        let serialized_transform = SerializedTransform::from(&transform);

        let mesh = if let (Some(mesh_renderer), Some(editor_mesh)) = (mesh_renderer, editor_mesh) {
            Some(MeshComponentData {
                source: editor_mesh.source.clone(),
                material: editor_mesh
                    .material_path
                    .as_ref()
                    .map(|path| normalize_path(path, root)),
                casts_shadow: mesh_renderer.0.casts_shadow,
                visible: mesh_renderer.0.visible,
            })
        } else {
            None
        };

        let light = light.map(|light| LightComponentData {
            kind: match light.0.light_type {
                LightType::Directional => LightKind::Directional,
                LightType::Point => LightKind::Point,
                LightType::Spot { angle } => LightKind::Spot { angle },
            },
            color: [light.0.color.x, light.0.color.y, light.0.color.z],
            intensity: light.0.intensity,
        });

        let camera = camera.map(|camera| CameraComponentData {
            fov_y_rad: camera.0.fov_y_rad,
            aspect_ratio: camera.0.aspect_ratio,
            near_plane: camera.0.near_plane,
            far_plane: camera.0.far_plane,
            active: active_camera.is_some(),
        });

        let scene = if scene_root.is_some() {
            scene_asset.map(|asset| SceneAssetData {
                path: normalize_path(asset.path.to_string_lossy().as_ref(), root),
            })
        } else {
            None
        };

        let scripts = script
            .map(|scripts| {
                scripts
                    .scripts
                    .iter()
                    .filter_map(|script| {
                        let path = script.path.as_ref()?;
                        Some(ScriptComponentData {
                            path: normalize_path(path.to_string_lossy().as_ref(), root),
                            language: script.language.clone(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let dynamic = world
            .get::<DynamicComponents>(entity)
            .map(|components| components.components.clone())
            .unwrap_or_default();
        let freecam = world.get::<Freecam>(entity).is_some();
        let physics = world
            .get::<ColliderShape>(entity)
            .copied()
            .and_then(|shape| {
                if let Some(body) = world.get::<DynamicRigidBody>(entity) {
                    Some(PhysicsComponentData {
                        collider_shape: SceneColliderShape::from(shape),
                        body_kind: PhysicsBodyKind::Dynamic { mass: body.mass },
                    })
                } else if world.get::<FixedCollider>(entity).is_some() {
                    Some(PhysicsComponentData {
                        collider_shape: SceneColliderShape::from(shape),
                        body_kind: PhysicsBodyKind::Fixed,
                    })
                } else {
                    None
                }
            });

        let components = SceneComponents {
            mesh,
            light,
            camera,
            scene,
            scripts,
            dynamic,
            freecam,
            physics,
        };

        entities.push(SceneEntityData {
            name: name.map(|name| name.to_string()),
            transform: serialized_transform,
            components,
        });
        entity_order.push(entity);
    }

    (
        SceneDocument {
            version: 1,
            entities,
        },
        entity_order,
    )
}

pub fn write_scene_document(path: &Path, document: &SceneDocument) -> Result<(), String> {
    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(6)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(document, pretty).map_err(|err| err.to_string())?;
    fs::write(path, data).map_err(|err| err.to_string())
}

pub fn read_scene_document(path: &Path) -> Result<SceneDocument, String> {
    let data = fs::read_to_string(path).map_err(|err| err.to_string())?;
    ron::de::from_str::<SceneDocument>(&data).map_err(|err| err.to_string())
}

pub fn spawn_scene_from_document(
    world: &mut World,
    document: &SceneDocument,
    project: &EditorProject,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Vec<Entity> {
    let mut created = Vec::new();
    let root = project.root.as_deref();

    let mut any_play_camera = false;

    for entity_data in &document.entities {
        let transform = entity_data.transform.to_transform();
        let mut entity = world.spawn((EditorEntity, BevyWrapper(transform)));

        if let Some(name) = &entity_data.name {
            entity.insert(Name::new(name.clone()));
        }

        if let Some(mesh) = &entity_data.components.mesh {
            if let Some(mesh_renderer) =
                build_mesh_renderer(mesh, asset_cache, asset_server, project)
            {
                entity.insert(mesh_renderer);
                entity.insert(EditorMesh {
                    source: mesh.source.clone(),
                    material_path: mesh.material.clone(),
                });
            }
        }

        if let Some(light) = &entity_data.components.light {
            entity.insert(BevyWrapper(serde_light_to_component(light)));
        }

        if let Some(camera) = &entity_data.components.camera {
            entity.insert(BevyWrapper(Camera {
                fov_y_rad: camera.fov_y_rad,
                aspect_ratio: camera.aspect_ratio,
                near_plane: camera.near_plane,
                far_plane: camera.far_plane,
            }));
            if camera.active {
                any_play_camera = true;
                entity.insert(EditorPlayCamera);
            }
        }

        if entity_data.components.freecam {
            entity.insert(Freecam::default());
        }

        if let Some(scene) = &entity_data.components.scene {
            let path = resolve_path(&scene.path, root);
            let handle = asset_server.0.lock().load_scene(path);
            entity.insert(SceneRoot(handle));
            entity.insert(SceneAssetPath {
                path: resolve_path(&scene.path, root),
            });
        }

        if !entity_data.components.scripts.is_empty() {
            let scripts = entity_data
                .components
                .scripts
                .iter()
                .filter_map(|script| {
                    let path = script.path.trim();
                    if path.is_empty() {
                        return None;
                    }
                    Some(ScriptEntry {
                        path: Some(resolve_path(path, root)),
                        language: script.language.clone(),
                    })
                })
                .collect::<Vec<_>>();
            if !scripts.is_empty() {
                entity.insert(ScriptComponent { scripts });
            }
        }

        if !entity_data.components.dynamic.is_empty() {
            entity.insert(DynamicComponents {
                components: entity_data.components.dynamic.clone(),
            });
        }

        if let Some(physics) = &entity_data.components.physics {
            entity.insert(physics.collider_shape.to_collider_shape());
            match physics.body_kind.clone() {
                PhysicsBodyKind::Dynamic { mass } => {
                    entity.insert(DynamicRigidBody { mass });
                }
                PhysicsBodyKind::Fixed => {
                    entity.insert(FixedCollider);
                }
            }
        }

        created.push(entity.id());
    }

    if !any_play_camera {
        ensure_active_camera(world);
    }

    created
}

pub fn restore_scene_transforms_from_document(
    world: &mut World,
    document: &SceneDocument,
    entities: &[Entity],
) {
    for (entity, entity_data) in entities.iter().zip(document.entities.iter()) {
        if let Some(mut transform) = world.get_mut::<BevyTransform>(*entity) {
            transform.0 = entity_data.transform.to_transform();
        }
    }
}

fn build_mesh_renderer(
    mesh: &MeshComponentData,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
    project: &EditorProject,
) -> Option<BevyMeshRenderer> {
    let material_handle = mesh
        .material
        .as_ref()
        .and_then(|path| load_material_handle(path, asset_cache, asset_server, project))
        .or_else(|| asset_cache.default_material);

    let Some(material_handle) = material_handle else {
        return None;
    };

    let mesh_handle = match &mesh.source {
        MeshSource::Primitive(kind) => Some(load_primitive_mesh(*kind, asset_cache, asset_server)),
        MeshSource::Asset { path } => {
            Some(load_mesh_asset(path, asset_cache, asset_server, project))
        }
    }?;

    Some(BevyWrapper(MeshRenderer::new(
        mesh_handle.id,
        material_handle.id,
        mesh.casts_shadow,
        mesh.visible,
    )))
}

fn load_material_handle(
    path: &str,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
    project: &EditorProject,
) -> Option<Handle<helmer::runtime::asset_server::Material>> {
    if let Some(handle) = asset_cache.material_handles.get(path).copied() {
        return Some(handle);
    }

    let root = project.root.as_deref()?;
    let full_path = resolve_path(path, Some(root));
    let handle = asset_server.0.lock().load_material(full_path);
    asset_cache
        .material_handles
        .insert(path.to_string(), handle);
    Some(handle)
}

fn load_mesh_asset(
    path: &str,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
    project: &EditorProject,
) -> Handle<helmer::runtime::asset_server::Mesh> {
    if let Some(handle) = asset_cache.mesh_handles.get(path).copied() {
        return handle;
    }

    let root = project.root.as_deref();
    let full_path = resolve_path(path, root);
    let handle = asset_server.0.lock().load_mesh(full_path);
    asset_cache.mesh_handles.insert(path.to_string(), handle);
    handle
}

fn load_primitive_mesh(
    kind: PrimitiveKind,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Handle<helmer::runtime::asset_server::Mesh> {
    if let Some(handle) = asset_cache.primitive_meshes.get(&kind).copied() {
        return handle;
    }

    let mesh_asset = match kind {
        PrimitiveKind::Cube => helmer::provided::components::MeshAsset::cube("cube".to_string()),
        PrimitiveKind::UvSphere(segments, rings) => {
            helmer::provided::components::MeshAsset::uv_sphere(
                "uv sphere".to_string(),
                segments,
                rings,
            )
        }
        PrimitiveKind::Plane => helmer::provided::components::MeshAsset::plane("plane".to_string()),
    };

    let handle = asset_server
        .0
        .lock()
        .add_mesh(mesh_asset.vertices.unwrap(), mesh_asset.indices);
    asset_cache.primitive_meshes.insert(kind, handle);
    handle
}

fn serde_light_to_component(light: &LightComponentData) -> Light {
    match light.kind {
        LightKind::Directional => Light::directional(
            glam::vec3(light.color[0], light.color[1], light.color[2]),
            light.intensity,
        ),
        LightKind::Point => Light::point(
            glam::vec3(light.color[0], light.color[1], light.color[2]),
            light.intensity,
        ),
        LightKind::Spot { angle } => Light::spot(
            glam::vec3(light.color[0], light.color[1], light.color[2]),
            light.intensity,
            angle,
        ),
    }
}

pub fn default_scene_path(project: &EditorProject) -> Option<PathBuf> {
    let root = project.root.as_ref()?;
    let config = project.config.as_ref()?;
    Some(config.scenes_root(root).join("untitled.hscene.ron"))
}

pub fn next_available_scene_path(project: &EditorProject) -> Option<PathBuf> {
    let root = project.root.as_ref()?;
    let config = project.config.as_ref()?;
    let scenes_root = config.scenes_root(root);

    for idx in 1..=999u32 {
        let candidate = scenes_root.join(format!("scene_{:03}.hscene.ron", idx));
        if !candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

fn normalize_path(path: &str, root: Option<&Path>) -> String {
    if let Some(root) = root {
        if let Ok(relative) = Path::new(path).strip_prefix(root) {
            return relative.to_string_lossy().replace('\\', "/");
        }
    }
    path.replace('\\', "/")
}

fn resolve_path(path: &str, root: Option<&Path>) -> PathBuf {
    if let Some(root) = root {
        let candidate = root.join(path);
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from(path)
}
