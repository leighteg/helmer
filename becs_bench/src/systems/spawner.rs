use std::collections::{HashMap, HashSet};

use bevy_ecs::{
    entity::Entity,
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use helmer::provided::components::{MeshRenderer, Transform};
use helmer_becs::{
    BevyMeshRenderer, BevyTransform,
    egui_integration::EguiResource,
    physics::components::{ColliderShape, DynamicRigidBody},
};

#[derive(Resource, Default)]
pub struct MeshRendererStore {
    pub mesh_renderers: HashMap<String, MeshRenderer>,
}

#[derive(Resource, Default)]
pub struct SpawnerSystemResource {
    pub spawn_iters: usize,
    pub spawned_entities: HashSet<Entity>,
    pub despawn_all: bool,
}

pub fn spawner_system(
    mut commands: Commands,
    mesh_renderer_store: Res<MeshRendererStore>,
    mut spawner_system_resource: ResMut<SpawnerSystemResource>,
    mut egui_res: ResMut<EguiResource>,
) {
    if spawner_system_resource.despawn_all {
        for entity in spawner_system_resource.spawned_entities.iter() {
            commands.entity(*entity).despawn();
        }
        spawner_system_resource.despawn_all = false;
    }

    /*if spawner_system_resource.spawned_count > 3000 {
        return;
    }*/

    if let Some(mesh_renderer) = mesh_renderer_store.mesh_renderers.get("default") {
        for i in 0..spawner_system_resource.spawn_iters {
            let new_entity = commands.spawn((
                BevyTransform {
                    0: Transform::from_position([0.0, 10.0, 5.0]),
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
                    egui::DragValue::new(&mut spawner_system_resource.spawn_iters)
                        .prefix("spawn iters: "),
                );

                if ui.button("destroy spawned entities").clicked() {
                    spawner_system_resource.despawn_all = true;
                }
            }
        }),
        "spawn system".to_string(),
    ));
}
