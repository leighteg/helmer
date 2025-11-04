use crate::ecs::{
    component::Component,
    ecs_core::{ECSCore, Entity},
    system::System,
};
use helmer::{
    provided::components::{MeshRenderer, Transform},
    runtime::{
        asset_server::{AssetServer, Handle, Scene},
        input_manager::InputManager,
    },
};
use parking_lot::Mutex;
use proc::Component as ComponentDerive;
use std::{any::TypeId, sync::Arc};
use tracing::info;

/// Add to an entity to request a scene to be loaded and spawned.
#[derive(ComponentDerive, Debug, Clone)]
pub struct SceneRoot(pub Handle<Scene>);

/// Added to each entity that is spawned as part of a scene.
#[derive(ComponentDerive, Debug, Clone)]
pub struct SceneChild {
    pub scene_root: Entity,
}

/// A marker component added to a SceneRoot entity after its scene has been spawned.
#[derive(ComponentDerive, Debug, Clone)]
struct SpawnedScene;

/// An ECS system that finds entities with a `SceneRoot` component
/// and spawns the corresponding scene's nodes as new entities.
pub struct SceneSpawningSystem;

impl System for SceneSpawningSystem {
    fn name(&self) -> &str {
        "SceneSpawningSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let asset_server = ecs
            .get_resource::<Arc<Mutex<AssetServer>>>()
            .unwrap()
            .clone();
        let mut spawn_commands = Vec::new();

        // Step 1: Get a list of all entities that could potentially be spawned.
        let candidate_entities = ecs
            .component_pool
            .get_entities_with_all(&[TypeId::of::<SceneRoot>()]);

        // Step 2: Loop through the list of candidates to check their status.
        for entity in candidate_entities {
            // Filter out entities that have already been spawned.
            if ecs.component_pool.get::<SpawnedScene>(entity).is_none() {
                // Get the actual SceneRoot component for the valid entity.
                if let Some(scene_root) = ecs.component_pool.get::<SceneRoot>(entity) {
                    // Check if the scene asset is finished loading.
                    if let Some(scene) = asset_server.lock().get_scene(&scene_root.0) {
                        spawn_commands.push((entity, (scene, scene_root.0.id)));
                    }
                }
            }
        }

        // Step 3: Execute the collected spawn commands.
        for (root_entity, scene) in spawn_commands {
            info!("Spawning scene {} for entity {}", scene.1, root_entity);

            let root_transform = ecs
                .get_component::<Transform>(root_entity)
                .cloned()
                .unwrap_or_default();

            for node in &scene.0.nodes {
                let child_entity = ecs.create_entity();
                let final_matrix = root_transform.to_matrix() * node.transform;
                ecs.add_component(child_entity, Transform::from_matrix(final_matrix));
                ecs.add_component(
                    child_entity,
                    MeshRenderer::new(node.mesh.id, node.material.id, true, true),
                );
                ecs.add_component(
                    child_entity,
                    SceneChild {
                        scene_root: root_entity,
                    },
                );
            }
            ecs.add_component(root_entity, SpawnedScene {});

            info!("Spawned scene {} for entity {}", scene.1, root_entity);
        }
    }
}
