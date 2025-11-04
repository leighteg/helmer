use std::collections::{HashMap, HashSet};

use bevy_ecs::{
    entity::Entity,
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use glam::Vec3;
use helmer::provided::components::{MeshRenderer, Transform};
use helmer_becs::{
    BevyMeshRenderer, BevyTransform, DeltaTime,
    egui_integration::EguiResource,
    physics::components::{ColliderShape, DynamicRigidBody},
};

#[derive(Resource, Default)]
pub struct MeshRendererStore {
    pub mesh_renderers: HashMap<String, MeshRenderer>,
}

#[derive(Resource, Default)]
pub struct SpawnerSystemResource {
    pub max_entities: usize,
    pub spawn_iters: usize,
    pub spawned_entities: HashSet<Entity>,
    pub despawn_all: bool,
    pub cube_scale: f32,
}

pub fn spawner_system(
    mut commands: Commands,
    mesh_renderer_store: Res<MeshRendererStore>,
    mut spawner_system_resource: ResMut<SpawnerSystemResource>,
    mut egui_res: ResMut<EguiResource>,
) {
    // --- UI ---
    egui_res.windows.push((
        Box::new(move |ui, world, _input_arc| {
            if let Some(mut spawner_system_resource) =
                world.get_resource_mut::<SpawnerSystemResource>()
            {
                ui.label(format!(
                    "spawned entity count: {}",
                    spawner_system_resource.spawned_entities.len()
                ));
                ui.add(
                    egui::DragValue::new(&mut spawner_system_resource.max_entities)
                        .prefix("max entities: "),
                );
                ui.add(
                    egui::DragValue::new(&mut spawner_system_resource.spawn_iters)
                        .prefix("spawn iters: "),
                );
                ui.add(
                    egui::DragValue::new(&mut spawner_system_resource.cube_scale)
                        .prefix("cube scale: "),
                );

                if ui.button("destroy spawned entities").clicked() {
                    spawner_system_resource.despawn_all = true;
                }
            }
        }),
        "spawn system".to_string(),
    ));

    // --- ECS logic ---

    if spawner_system_resource.despawn_all {
        for entity in spawner_system_resource.spawned_entities.iter() {
            commands.entity(*entity).despawn();
        }
        spawner_system_resource.spawned_entities.clear();
        spawner_system_resource.despawn_all = false;
    }

    if spawner_system_resource.spawned_entities.len() > spawner_system_resource.max_entities {
        let mut ran = false;
        let mut entity_to_remove: Option<Entity> = None;
        for entity in spawner_system_resource.spawned_entities.iter() {
            if ran {
                break;
            }

            commands.entity(*entity).despawn();
            entity_to_remove = Some(*entity);
            ran = true;
        }

        if let Some(entity_to_remove) = entity_to_remove {
            spawner_system_resource
                .spawned_entities
                .remove(&entity_to_remove);
        }

        return;
    }

    if let Some(mesh_renderer) = mesh_renderer_store.mesh_renderers.get("default") {
        for i in 0..spawner_system_resource.spawn_iters {
            let new_entity = commands.spawn((
                BevyTransform {
                    0: Transform {
                        position: Vec3::from_array([0.0, 10.0, 5.0]),
                        scale: Vec3::from_array([spawner_system_resource.cube_scale; 3]),
                        ..Default::default()
                    },
                },
                BevyMeshRenderer { 0: *mesh_renderer },
                ColliderShape::Cuboid,
                DynamicRigidBody { mass: 1.0 },
            ));
            spawner_system_resource
                .spawned_entities
                .insert(new_entity.id());
        }
    }
}
