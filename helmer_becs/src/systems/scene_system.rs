use crate::{
    BevyAnimator, BevyAssetServerParam, BevyMeshRenderer, BevyRuntimeProfiling,
    BevySkinnedMeshRenderer, BevyTransform, BevyWrapper,
};
use bevy_ecs::{
    component::Component,
    name::Name,
    prelude::{Changed, Commands, Entity, ParamSet, Query, Res, ResMut, Resource, Without},
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

#[derive(Component)]
pub struct SceneChild {
    pub scene_root: Entity,
    pub local_transform: Mat4,   // Transform in parent's local space
    pub last_written: Transform, // Last world transform written to BevyTransform
    pub scene_node_index: usize,
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

            for (scene_node_index, node) in scene.nodes.iter().enumerate() {
                // Compute initial world transform
                let world_matrix = parent_matrix * node.transform;
                let world_transform = Transform::from_matrix(world_matrix);

                let skin = node
                    .skin_index
                    .and_then(|idx| scene.skins.read().get(idx).cloned());

                let mut entity_commands = if let Some(skin) = skin {
                    let skinned =
                        SkinnedMeshRenderer::new(node.mesh.id, node.material.id, skin, true, true);
                    let mut commands = commands.spawn((
                        BevyWrapper(world_transform),
                        BevySkinnedMeshRenderer(skinned),
                        SceneChild {
                            scene_root: root_entity,
                            local_transform: node.transform,
                            last_written: world_transform,
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
                        SceneChild {
                            scene_root: root_entity,
                            local_transform: node.transform,
                            last_written: world_transform,
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
) {
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
