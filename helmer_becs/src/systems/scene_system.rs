use bevy_ecs::{
    component::Component,
    prelude::{Commands, Entity, Query, Res, Without},
    query::With,
    system::Local,
};
use glam::Mat4;
use hashbrown::{HashMap, HashSet};
use helmer::{
    provided::components::{MeshRenderer, Transform},
    runtime::asset_server::{AssetServer, Handle, Scene},
};
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::info;

use crate::{BevyAssetServer, BevyMeshRenderer, BevyTransform};

#[derive(Default)]
pub struct SceneSpawningLocal {
    pub spawned_scenes: HashMap<Entity, Vec<Entity>>,
}

//================================================================================
// Bevy Wrapper Components
//================================================================================

#[derive(Component)]
pub struct SceneRoot(pub Handle<Scene>);

#[derive(Component)]
pub struct SceneChild {
    pub scene_root: Entity,
}

#[derive(Component)]
pub struct SpawnedScene;

//================================================================================
// The Bevy System
//================================================================================

/// An ECS system that finds entities with a `SceneRoot` component
/// and spawns the corresponding scene's nodes as new entities.
pub fn scene_spawning_system(
    mut commands: Commands,
    mut local: Local<SceneSpawningLocal>,
    asset_server: Res<BevyAssetServer>,
    scene_root_query: Query<(Entity, &SceneRoot), Without<SpawnedScene>>,
    scene_child_query: Query<(Entity, &SceneChild)>,
    transforms: Query<&BevyTransform>,
) {
    for (root_entity, scene_root) in scene_root_query.iter() {
        if let Some(scene) = asset_server.0.lock().get_scene(&scene_root.0) {
            info!(
                "Spawning scene {} for entity {}",
                scene_root.0.id, root_entity
            );

            let root_transform = transforms.get(root_entity).map(|t| t.0).unwrap_or_default();

            let mut spawned_children = Vec::new();

            for node in &scene.nodes {
                let final_matrix = root_transform.to_matrix() * node.transform;
                let child_entity = commands.spawn_empty().id();
                spawned_children.push(child_entity);
                commands.entity(child_entity).insert(BevyTransform {
                    0: Transform::from_matrix(final_matrix),
                });
                commands.entity(child_entity).insert(BevyMeshRenderer {
                    0: MeshRenderer::new(node.mesh.id, node.material.id, true, true),
                });
                commands.entity(child_entity).insert(SceneChild {
                    scene_root: root_entity,
                });
            }
            local.spawned_scenes.insert(root_entity, spawned_children);
            commands.entity(root_entity).insert(SpawnedScene);

            info!(
                "Spawned scene {} for entity {}",
                scene_root.0.id, root_entity
            );
        }
    }

    for (spawned_scene_entity, childen) in local.spawned_scenes.iter() {
        if !commands.get_entity(*spawned_scene_entity).is_ok() {
            for child in childen {
                commands.entity(*child).despawn();
            }
        }
    }
}
