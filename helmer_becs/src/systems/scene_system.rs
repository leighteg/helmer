use bevy_ecs::{
    component::Component,
    prelude::{Commands, Entity, Query, Res, Without},
    query::With,
};
use glam::Mat4;
use helmer::{
    provided::components::{MeshRenderer, Transform},
    runtime::asset_server::{AssetServer, Handle, Scene},
};
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::info;

use crate::{BevyAssetServer, BevyMeshRenderer, BevyTransform};

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
    asset_server: Res<BevyAssetServer>,
    query: Query<(Entity, &SceneRoot), Without<SpawnedScene>>,
    transforms: Query<&BevyTransform>,
) {
    for (root_entity, scene_root) in query.iter() {
        if let Some(scene) = asset_server.0.lock().get_scene(&scene_root.0) {
            info!(
                "Spawning scene with {} nodes for entity {}",
                scene.nodes.len(),
                root_entity
            );

            let root_transform = transforms.get(root_entity).map(|t| t.0).unwrap_or_default();

            for node in &scene.nodes {
                let final_matrix = root_transform.to_matrix() * node.transform;
                let child_entity = commands.spawn_empty().id();
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
            commands.entity(root_entity).insert(SpawnedScene);

            info!(
                "Spawned scene with {} nodes for entity {}",
                scene.nodes.len(),
                root_entity
            );
        }
    }
}
