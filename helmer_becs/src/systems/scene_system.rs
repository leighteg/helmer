use crate::{BevyAssetServer, BevyRuntimeProfiling, BevyTransform, BevyWrapper};
use bevy_ecs::{
    component::Component,
    name::Name,
    prelude::{Changed, Commands, Entity, ParamSet, Query, Res, ResMut, Resource, Without},
    query::With,
    system::Local,
};
use glam::Mat4;
use hashbrown::{HashMap, HashSet};
use helmer::{
    provided::components::{MeshRenderer, Transform},
    runtime::asset_server::{Handle, Scene},
};
use std::time::Instant;
use tracing::info;

#[derive(Resource, Default)]
pub struct SceneSpawnedChildren {
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
    pub local_transform: Mat4,   // Transform in parent's local space
    pub last_written: Transform, // Last world transform written to BevyTransform
}

#[derive(Component)]
pub struct SpawnedScene;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SceneTuning {
    pub transform_epsilon: f32,
}

impl Default for SceneTuning {
    fn default() -> Self {
        Self {
            transform_epsilon: 0.0001,
        }
    }
}

#[inline]
fn transform_approx_eq(a: &Transform, b: &Transform, epsilon: f32) -> bool {
    a.position.abs_diff_eq(b.position, epsilon)
        && a.scale.abs_diff_eq(b.scale, epsilon)
        && a.rotation.dot(b.rotation).abs() >= 1.0 - epsilon
}

#[derive(Clone, Copy)]
struct RootMatrices {
    parent: Mat4,
    inverse: Mat4,
}

#[derive(Default)]
pub struct SceneChildUpdateCache {
    changed_roots: HashSet<Entity>,
    root_matrices: HashMap<Entity, RootMatrices>,
}

//================================================================================
// The Systems
//================================================================================

/// An ECS system that finds entities with a `SceneRoot` component
/// and spawns the corresponding scene's nodes as new entities.
pub fn scene_spawning_system(
    mut commands: Commands,
    mut scene_children: ResMut<SceneSpawnedChildren>,
    asset_server: Res<BevyAssetServer>,
    scene_root_query: Query<(Entity, &SceneRoot), Without<SpawnedScene>>,
    root_transforms: Query<&BevyTransform, With<SceneRoot>>,
    profiling_res: Option<Res<BevyRuntimeProfiling>>,
) {
    let profiling = profiling_res.as_ref().map(|p| p.0.clone());
    let profiling_start = profiling.as_ref().and_then(|profiling| {
        if profiling.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            Some(Instant::now())
        } else {
            None
        }
    });

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

            let mut spawned_children = Vec::with_capacity(scene.nodes.len());

            for node in &scene.nodes {
                // Compute initial world transform
                let world_matrix = parent_matrix * node.transform;
                let world_transform = Transform::from_matrix(world_matrix);

                let child_entity = commands
                    .spawn((
                        BevyWrapper(world_transform),
                        BevyWrapper(MeshRenderer::new(
                            node.mesh.id,
                            node.material.id,
                            true,
                            true,
                        )),
                        SceneChild {
                            scene_root: root_entity,
                            local_transform: node.transform,
                            last_written: world_transform,
                        },
                        Name::new(format!(
                            "scene {} child {}",
                            scene_root.0.id,
                            spawned_children.len() + 1
                        )),
                    ))
                    .id();
                spawned_children.push(child_entity);
            }

            let child_count = spawned_children.len();
            scene_children
                .spawned_scenes
                .insert(root_entity, spawned_children);
            commands.entity(root_entity).insert(SpawnedScene);

            info!("Spawned {} children for scene", child_count);
        }
    }

    scene_children
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

    if let (Some(profiling), Some(start)) = (profiling.as_ref(), profiling_start) {
        profiling.ecs_scene_spawn_us.store(
            start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// An ECS system that finds entities with a `SceneChild` component
/// and makes it's transform local to the corresponding scene root entity.
pub fn update_scene_child_transforms(
    scene_children: Res<SceneSpawnedChildren>,
    scene_tuning: Res<crate::BevySceneTuning>,
    mut cache: Local<SceneChildUpdateCache>,
    changed_roots: Query<(Entity, &BevyTransform), (With<SceneRoot>, Changed<BevyTransform>)>,
    root_query: Query<&BevyTransform, With<SceneRoot>>,
    mut child_queries: ParamSet<(
        Query<(&mut SceneChild, &mut BevyTransform), Without<SceneRoot>>,
        Query<
            (Entity, &mut SceneChild, &BevyTransform),
            (Without<SceneRoot>, Changed<BevyTransform>),
        >,
    )>,
    profiling_res: Option<Res<BevyRuntimeProfiling>>,
) {
    let profiling = profiling_res.as_ref().map(|p| p.0.clone());
    let profiling_start = profiling.as_ref().and_then(|profiling| {
        if profiling.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            Some(Instant::now())
        } else {
            None
        }
    });

    let epsilon = scene_tuning.0.transform_epsilon.max(0.0);
    cache.changed_roots.clear();
    cache.root_matrices.clear();

    for (root_entity, root_transform) in changed_roots.iter() {
        let parent = root_transform.0.to_matrix();
        let inverse = parent.inverse();
        cache.changed_roots.insert(root_entity);
        cache
            .root_matrices
            .insert(root_entity, RootMatrices { parent, inverse });
    }

    if !cache.changed_roots.is_empty() {
        let mut child_query = child_queries.p0();
        for (root_entity, root_matrices) in cache.root_matrices.iter() {
            let Some(children) = scene_children.spawned_scenes.get(root_entity) else {
                continue;
            };
            for &child_entity in children {
                let Ok((mut child, mut child_transform)) = child_query.get_mut(child_entity) else {
                    continue;
                };

                let current_transform = child_transform.0;
                if !transform_approx_eq(&current_transform, &child.last_written, epsilon) {
                    let current_world = current_transform.to_matrix();
                    child.local_transform = root_matrices.inverse * current_world;
                }

                let world_transform =
                    Transform::from_matrix(root_matrices.parent * child.local_transform);
                if !transform_approx_eq(&world_transform, &current_transform, epsilon) {
                    child_transform.0 = world_transform;
                }
                if !transform_approx_eq(&world_transform, &child.last_written, epsilon) {
                    child.last_written = world_transform;
                }
            }
        }
    }

    let mut external_query = child_queries.p1();
    for (_entity, mut child, child_transform) in external_query.iter_mut() {
        if cache.changed_roots.contains(&child.scene_root) {
            continue;
        }

        let current_transform = child_transform.0;
        if transform_approx_eq(&current_transform, &child.last_written, epsilon) {
            continue;
        }

        let root_matrices = if let Some(root_matrices) = cache.root_matrices.get(&child.scene_root)
        {
            *root_matrices
        } else {
            let Ok(root_transform) = root_query.get(child.scene_root) else {
                continue;
            };
            let parent = root_transform.0.to_matrix();
            let inverse = parent.inverse();
            let root_matrices = RootMatrices { parent, inverse };
            cache.root_matrices.insert(child.scene_root, root_matrices);
            root_matrices
        };

        child.local_transform = root_matrices.inverse * current_transform.to_matrix();
        child.last_written = current_transform;
    }

    if let (Some(profiling), Some(start)) = (profiling.as_ref(), profiling_start) {
        profiling.ecs_scene_update_us.store(
            start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}
