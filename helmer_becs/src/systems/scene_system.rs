use crate::{
    BevyAnimator, BevyAssetServerParam, BevyMeshRenderer, BevyRuntimeProfiling,
    BevySkinnedMeshRenderer, BevySystemProfiler, BevyTransform, BevyWrapper,
};
use bevy_ecs::{
    component::Component,
    name::Name,
    prelude::{Changed, Commands, Entity, ParamSet, Query, Res, ResMut, Resource, Without, World},
    query::With,
    system::Local,
};
use glam::Mat4;
use hashbrown::{HashMap, HashSet};
use helmer::animation::{
    AnimationGraph, AnimationLayer, AnimationLibrary, AnimationNode, AnimationParameters,
    AnimationState, AnimationStateMachine, Animator, BlendMode, BlendNode, ClipNode,
};
use helmer::provided::components::{MeshRenderer, SkinnedMeshRenderer, Transform};
use helmer::runtime::asset_server::{Handle, Scene};
use std::sync::Arc;
use tracing::info;
use web_time::Instant;

#[derive(Resource, Default)]
pub struct SceneSpawnedChildren {
    pub spawned_scenes: HashMap<Entity, Vec<Entity>>,
}

//================================================================================
// Bevy Wrapper Components
//================================================================================
#[derive(Component)]
pub struct SceneRoot(pub Handle<Scene>);

#[derive(Component, Clone, Copy, Debug)]
pub struct SceneChild {
    pub scene_root: Entity,
    pub scene_node_index: usize,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct EntityParent {
    pub parent: Entity,
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
    changed_entities: HashSet<Entity>,
    enqueued_entities: HashSet<Entity>,
    root_matrices: HashMap<Entity, RootMatrices>,
    parent_to_children: HashMap<Entity, Vec<Entity>>,
    entity_to_parent: HashMap<Entity, Entity>,
    stack: Vec<(Entity, Mat4)>,
}

pub fn build_default_animator(library: Arc<AnimationLibrary>) -> Animator {
    let mut nodes = Vec::new();
    if !library.clips.is_empty() {
        nodes.push(AnimationNode::Clip(ClipNode {
            clip_index: 0,
            speed: 1.0,
            looping: true,
            time_offset: 0.0,
        }));
    } else {
        nodes.push(AnimationNode::Blend(BlendNode {
            children: Vec::new(),
            normalize: true,
            mode: BlendMode::Linear,
        }));
    }

    let graph = AnimationGraph { library, nodes };
    let state = AnimationState {
        name: "Default".to_string(),
        node: 0,
    };
    let state_machine = AnimationStateMachine::new(vec![state], Vec::new());
    let layer = AnimationLayer {
        name: "Base".to_string(),
        weight: 1.0,
        additive: false,
        mask: Vec::new(),
        graph,
        state_machine,
    };
    Animator {
        layers: vec![layer],
        parameters: AnimationParameters::default(),
        enabled: true,
        time_scale: 1.0,
    }
}

#[derive(Component)]
pub struct PendingSkinnedMesh {
    pub skin_index: usize,
}

//================================================================================
// The Systems
//================================================================================

/// An ECS system that finds entities with a `SceneRoot` component
/// and spawns the corresponding scene's nodes as new entities.
pub fn scene_spawning_system(
    mut commands: Commands,
    mut scene_children: ResMut<SceneSpawnedChildren>,
    asset_server: BevyAssetServerParam<'_>,
    scene_root_query: Query<(Entity, &SceneRoot), Without<SpawnedScene>>,
    root_transforms: Query<&BevyTransform, With<SceneRoot>>,
    profiling_res: Option<Res<BevyRuntimeProfiling>>,
    system_profiler: Option<Res<BevySystemProfiler>>,
) {
    let _system_scope = system_profiler.as_ref().and_then(|profiler| {
        profiler
            .0
            .begin_scope("helmer_becs::systems::scene_spawning_system")
    });

    let profiling = profiling_res.as_ref().map(|p| p.0.clone());
    let profiling_start = profiling.as_ref().and_then(|profiling| {
        if profiling.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            Some(Instant::now())
        } else {
            None
        }
    });

    for (root_entity, scene_root) in scene_root_query.iter() {
        let scene = {
            let asset_server_guard = asset_server.0.lock();
            let scene = asset_server_guard.get_scene(&scene_root.0);
            if scene.is_some() {
                // proactively request scene assets so they're seen immediately
                asset_server_guard.request_scene_assets(&scene_root.0, Some(0), 1.0);
            }
            scene
        };

        if let Some(scene) = scene {
            info!(
                "Spawning scene {} for entity {}",
                scene_root.0.id, root_entity
            );

            let parent_matrix = root_transforms
                .get(root_entity)
                .map(|t| t.0.to_matrix())
                .unwrap_or(Mat4::IDENTITY);

            let mut spawned_children = Vec::with_capacity(scene.nodes.len());
            let mut node_parent_map: HashMap<usize, Option<usize>> =
                HashMap::with_capacity(scene.nodes.len());
            let mut node_entity_map: HashMap<usize, Entity> =
                HashMap::with_capacity(scene.nodes.len());
            let mut node_world_map: HashMap<usize, Mat4> =
                HashMap::with_capacity(scene.nodes.len());

            for node in scene.nodes.iter() {
                node_parent_map
                    .entry(node.node_index)
                    .or_insert(node.parent_node_index);
            }

            for (scene_node_index, node) in scene.nodes.iter().enumerate() {
                // Compute initial world transform
                let world_matrix = parent_matrix * node.transform;
                let world_transform = Transform::from_matrix(world_matrix);
                let mut parent_entity = root_entity;
                let mut parent_world_matrix = parent_matrix;

                let mut ancestor = node.parent_node_index;
                while let Some(ancestor_node_index) = ancestor {
                    if let Some(mapped_parent) = node_entity_map.get(&ancestor_node_index).copied()
                    {
                        parent_entity = mapped_parent;
                        parent_world_matrix = node_world_map
                            .get(&ancestor_node_index)
                            .copied()
                            .unwrap_or(parent_matrix);
                        break;
                    }
                    ancestor = node_parent_map.get(&ancestor_node_index).copied().flatten();
                }

                let local_transform = parent_world_matrix.inverse() * world_matrix;

                let skin = node
                    .skin_index
                    .and_then(|idx| scene.skins.read().get(idx).cloned());

                let mut entity_commands = if let Some(skin) = skin {
                    let skinned =
                        SkinnedMeshRenderer::new(node.mesh.id, node.material.id, skin, true, true);
                    let mut commands = commands.spawn((
                        BevyWrapper(world_transform),
                        BevySkinnedMeshRenderer(skinned),
                        EntityParent {
                            parent: parent_entity,
                            local_transform,
                            last_written: world_transform,
                        },
                        SceneChild {
                            scene_root: root_entity,
                            scene_node_index,
                        },
                        Name::new(format!(
                            "scene {} child {}",
                            scene_root.0.id,
                            spawned_children.len() + 1
                        )),
                    ));
                    if let Some(anim_lib) = node
                        .skin_index
                        .and_then(|idx| scene.animations.read().get(idx).cloned())
                    {
                        commands.insert(BevyAnimator(build_default_animator(anim_lib)));
                    }
                    commands
                } else {
                    let mut commands = commands.spawn((
                        BevyWrapper(world_transform),
                        BevyWrapper(MeshRenderer::new(
                            node.mesh.id,
                            node.material.id,
                            true,
                            true,
                        )),
                        EntityParent {
                            parent: parent_entity,
                            local_transform,
                            last_written: world_transform,
                        },
                        SceneChild {
                            scene_root: root_entity,
                            scene_node_index,
                        },
                        Name::new(format!(
                            "scene {} child {}",
                            scene_root.0.id,
                            spawned_children.len() + 1
                        )),
                    ));
                    if let Some(skin_index) = node.skin_index {
                        commands.insert(PendingSkinnedMesh { skin_index });
                    }
                    commands
                };

                let child_entity = entity_commands.id();
                spawned_children.push(child_entity);
                node_entity_map
                    .entry(node.node_index)
                    .or_insert(child_entity);
                node_world_map
                    .entry(node.node_index)
                    .or_insert(world_matrix);
            }

            let child_count = spawned_children.len();
            scene_children
                .spawned_scenes
                .insert(root_entity, spawned_children);
            commands.entity(root_entity).try_insert(SpawnedScene);

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
                children.retain(|child| commands.get_entity(*child).is_ok());
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

pub fn scene_child_skinning_system(
    mut commands: Commands,
    asset_server: BevyAssetServerParam<'_>,
    scene_root_query: Query<&SceneRoot>,
    pending_query: Query<(
        Entity,
        &SceneChild,
        &PendingSkinnedMesh,
        Option<&BevyMeshRenderer>,
    )>,
    system_profiler: Option<Res<BevySystemProfiler>>,
) {
    let _system_scope = system_profiler.as_ref().and_then(|profiler| {
        profiler
            .0
            .begin_scope("helmer_becs::systems::scene_child_skinning_system")
    });

    let asset_server = asset_server.0.lock();

    for (entity, child, pending, mesh_renderer) in pending_query.iter() {
        let Ok(scene_root) = scene_root_query.get(child.scene_root) else {
            continue;
        };
        let Some(scene) = asset_server.get_scene(&scene_root.0) else {
            continue;
        };
        let Some(node) = scene.nodes.get(child.scene_node_index) else {
            continue;
        };
        if node.skin_index != Some(pending.skin_index) {
            continue;
        }
        let Some(skin) = scene.skins.read().get(pending.skin_index).cloned() else {
            continue;
        };

        let (casts_shadow, visible) = mesh_renderer
            .map(|renderer| (renderer.0.casts_shadow, renderer.0.visible))
            .unwrap_or((true, true));

        let skinned =
            SkinnedMeshRenderer::new(node.mesh.id, node.material.id, skin, casts_shadow, visible);

        let mut entity_commands = commands.entity(entity);
        entity_commands.try_remove::<BevyMeshRenderer>();
        entity_commands.try_insert(BevySkinnedMeshRenderer(skinned));
        entity_commands.try_remove::<PendingSkinnedMesh>();

        if let Some(anim_lib) = scene.animations.read().get(pending.skin_index).cloned() {
            entity_commands.try_insert(BevyAnimator(build_default_animator(anim_lib)));
        }
    }
}

/// Flush deferred commands so downstream systems can see spawned scene entities in the same frame.
pub fn apply_scene_commands_system(world: &mut World) {
    let _system_scope = world
        .get_resource::<BevySystemProfiler>()
        .and_then(|profiler| {
            profiler
                .0
                .begin_scope("helmer_becs::systems::apply_scene_commands_system")
        });
    world.flush();
}

/// Synchronize entity transforms for hierarchical relations.
///
/// `EntityParent` stores each child's local transform and the last world-space transform that was
/// authored by this system. External edits to a child update local space. External edits to a
/// parent are propagated recursively down the subtree
pub fn update_scene_child_transforms(
    scene_tuning: Res<crate::BevySceneTuning>,
    mut cache: Local<SceneChildUpdateCache>,
    mut queries: ParamSet<(
        Query<Entity, Changed<BevyTransform>>,
        Query<&BevyTransform>,
        Query<(Entity, &EntityParent)>,
        Query<(&mut EntityParent, &mut BevyTransform)>,
        Query<(Entity, &EntityParent, &BevyTransform), Changed<BevyTransform>>,
    )>,
    profiling_res: Option<Res<BevyRuntimeProfiling>>,
    system_profiler: Option<Res<BevySystemProfiler>>,
) {
    let _system_scope = system_profiler.as_ref().and_then(|profiler| {
        profiler
            .0
            .begin_scope("helmer_becs::systems::update_scene_child_transforms")
    });

    let profiling = profiling_res.as_ref().map(|p| p.0.clone());
    let profiling_start = profiling.as_ref().and_then(|profiling| {
        if profiling.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            Some(Instant::now())
        } else {
            None
        }
    });

    let epsilon = scene_tuning.0.transform_epsilon.max(0.0);
    cache.changed_entities.clear();
    cache.enqueued_entities.clear();
    cache.root_matrices.clear();
    cache.parent_to_children.clear();
    cache.entity_to_parent.clear();
    cache.stack.clear();

    {
        let changed_transforms = queries.p0();
        for entity in changed_transforms.iter() {
            cache.changed_entities.insert(entity);
        }
    }

    {
        let parent_snapshot = queries.p2();
        for (entity, parent) in parent_snapshot.iter() {
            cache
                .parent_to_children
                .entry(parent.parent)
                .or_default()
                .push(entity);
            cache.entity_to_parent.insert(entity, parent.parent);
        }
    }

    let mut changed_child_updates: Vec<(Entity, Entity, Transform)> = Vec::new();
    {
        let changed_children = queries.p4();
        for (entity, relation, child_transform) in changed_children.iter() {
            if !cache.changed_entities.contains(&entity) {
                continue;
            }

            let current_transform = child_transform.0;
            if transform_approx_eq(&current_transform, &relation.last_written, epsilon) {
                continue;
            }

            changed_child_updates.push((entity, relation.parent, current_transform));
        }
    }

    for (_, parent_entity, _) in changed_child_updates.iter().copied() {
        if cache.root_matrices.contains_key(&parent_entity) {
            continue;
        }
        let parent = {
            let transform_query = queries.p1();
            let Ok(parent_transform) = transform_query.get(parent_entity) else {
                continue;
            };
            parent_transform.0.to_matrix()
        };
        let inverse = parent.inverse();
        cache
            .root_matrices
            .insert(parent_entity, RootMatrices { parent, inverse });
    }

    if !changed_child_updates.is_empty() {
        let mut relation_query = queries.p3();
        for (entity, parent_entity, current_transform) in changed_child_updates {
            let Some(parent_matrices) = cache.root_matrices.get(&parent_entity).copied() else {
                continue;
            };
            let Ok((mut relation, _)) = relation_query.get_mut(entity) else {
                continue;
            };
            relation.local_transform = parent_matrices.inverse * current_transform.to_matrix();
            relation.last_written = current_transform;
        }
    }

    let changed_entities: Vec<Entity> = cache.changed_entities.iter().copied().collect();
    for entity in changed_entities {
        if !cache.parent_to_children.contains_key(&entity) {
            continue;
        }
        let mut ancestor = cache.entity_to_parent.get(&entity).copied();
        let mut has_changed_ancestor = false;
        let mut visited_ancestors = HashSet::new();
        while let Some(parent_entity) = ancestor {
            if !visited_ancestors.insert(parent_entity) {
                break;
            }
            if cache.changed_entities.contains(&parent_entity) {
                has_changed_ancestor = true;
                break;
            }
            ancestor = cache.entity_to_parent.get(&parent_entity).copied();
        }
        if has_changed_ancestor {
            continue;
        }

        let parent_matrices = if let Some(existing) = cache.root_matrices.get(&entity) {
            *existing
        } else {
            let parent = {
                let transform_query = queries.p1();
                let Ok(parent_transform) = transform_query.get(entity) else {
                    continue;
                };
                parent_transform.0.to_matrix()
            };
            let inverse = parent.inverse();
            let matrices = RootMatrices { parent, inverse };
            cache.root_matrices.insert(entity, matrices);
            matrices
        };

        if cache.enqueued_entities.insert(entity) {
            cache.stack.push((entity, parent_matrices.parent));
        }
    }

    {
        let mut relation_query = queries.p3();
        while let Some((parent_entity, parent_matrix)) = cache.stack.pop() {
            let parent_inverse = parent_matrix.inverse();
            let Some(children) = cache.parent_to_children.get(&parent_entity).cloned() else {
                continue;
            };

            for child_entity in children {
                let Ok((mut relation, mut child_transform)) = relation_query.get_mut(child_entity)
                else {
                    continue;
                };

                let current_transform = child_transform.0;
                if !transform_approx_eq(&current_transform, &relation.last_written, epsilon) {
                    relation.local_transform = parent_inverse * current_transform.to_matrix();
                }

                let world_transform =
                    Transform::from_matrix(parent_matrix * relation.local_transform);
                if !transform_approx_eq(&world_transform, &current_transform, epsilon) {
                    child_transform.0 = world_transform;
                }
                if !transform_approx_eq(&world_transform, &relation.last_written, epsilon) {
                    relation.last_written = world_transform;
                }

                if cache.parent_to_children.contains_key(&child_entity)
                    && cache.enqueued_entities.insert(child_entity)
                {
                    cache
                        .stack
                        .push((child_entity, world_transform.to_matrix()));
                }
            }
        }
    }

    if let (Some(profiling), Some(start)) = (profiling.as_ref(), profiling_start) {
        profiling.ecs_scene_update_us.store(
            start.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}
