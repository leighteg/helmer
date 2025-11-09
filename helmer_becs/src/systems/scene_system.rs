use crate::{BevyAssetServer, BevyMeshRenderer, BevyTransform};
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
    pub local_transform: Mat4, // Transform in parent's local space
    pub last_written: Mat4,    // What was wrote to BevyTransform last frame
}

#[derive(Component)]
pub struct SpawnedScene;

//================================================================================
// The Systems
//================================================================================

/// An ECS system that finds entities with a `SceneRoot` component
/// and spawns the corresponding scene's nodes as new entities.
pub fn scene_spawning_system(
    mut commands: Commands,
    mut local: Local<SceneSpawningLocal>,
    asset_server: Res<BevyAssetServer>,
    scene_root_query: Query<(Entity, &SceneRoot), Without<SpawnedScene>>,
    root_transforms: Query<&BevyTransform, With<SceneRoot>>,
) {
    for (root_entity, scene_root) in scene_root_query.iter() {
        if let Some(scene) = asset_server.0.lock().get_scene(&scene_root.0) {
            info!(
                "Spawning scene {} for entity {}",
                scene_root.0.id, root_entity
            );

            let parent_matrix = root_transforms
                .get(root_entity)
                .map(|t| t.0.to_matrix())
                .unwrap_or(Mat4::IDENTITY);

            let mut spawned_children = Vec::new();

            for node in &scene.nodes {
                let child_entity = commands.spawn_empty().id();
                spawned_children.push(child_entity);

                // Compute initial world transform
                let world_matrix = parent_matrix * node.transform;

                commands.entity(child_entity).insert(BevyTransform {
                    0: Transform::from_matrix(world_matrix),
                });

                commands.entity(child_entity).insert(BevyMeshRenderer {
                    0: MeshRenderer::new(node.mesh.id, node.material.id, true, true),
                });

                commands.entity(child_entity).insert(SceneChild {
                    scene_root: root_entity,
                    local_transform: node.transform,
                    last_written: world_matrix,
                });
            }

            local
                .spawned_scenes
                .insert(root_entity, spawned_children.clone());
            commands.entity(root_entity).insert(SpawnedScene);

            info!("Spawned {} children for scene", spawned_children.len());
        }
    }

    local
        .spawned_scenes
        .retain(|spawned_scene_entity, children| {
            if commands.get_entity(*spawned_scene_entity).is_err() {
                for child in children {
                    if let Ok(mut entity_commands) = commands.get_entity(*child) {
                        entity_commands.despawn();
                    }
                }
                false
            } else {
                true
            }
        });
}

/// An ECS system that finds entities with a `SceneChild` component
/// and makes it's transform local to the corresponding scene root entity.
pub fn update_scene_child_transforms(
    mut child_query: Query<(&mut SceneChild, &mut BevyTransform), Without<SceneRoot>>,
    root_query: Query<&BevyTransform, With<SceneRoot>>,
) {
    for (mut child, mut child_transform) in child_query.iter_mut() {
        if let Ok(root_transform) = root_query.get(child.scene_root) {
            let parent_matrix = root_transform.0.to_matrix();
            let current_world = child_transform.0.to_matrix();

            // Check if BevyTransform was modified externally
            if (current_world - child.last_written).abs_diff_eq(Mat4::ZERO, 0.0001) == false {
                // BevyTransform changed! Update local transform
                child.local_transform = parent_matrix.inverse() * current_world;
            }

            // Compute world transform from parent and local
            let world_matrix = parent_matrix * child.local_transform;
            child_transform.0 = Transform::from_matrix(world_matrix);
            child.last_written = world_matrix;
        }
    }
}
