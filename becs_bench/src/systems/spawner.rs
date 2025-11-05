use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

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
use rand::Rng;

use crate::systems::config_toggle::HideToggleProof;

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
    pub mesh_scale: f32,
    pub rand_xz_range: Range<f32>,
    pub rand_y_range: Range<f32>,
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
                    egui::DragValue::new(&mut spawner_system_resource.mesh_scale)
                        .range(1..=100)
                        .prefix("mesh scale: "),
                );

                ui.separator();

                ui.add(
                    egui::DragValue::new(&mut spawner_system_resource.rand_xz_range.start)
                        .prefix("spawn x/z min: "),
                );
                ui.add(
                    egui::DragValue::new(&mut spawner_system_resource.rand_xz_range.end)
                        .prefix("spawn x/z max: "),
                );

                ui.add(
                    egui::DragValue::new(&mut spawner_system_resource.rand_y_range.start)
                        .prefix("spawn y min: "),
                );
                ui.add(
                    egui::DragValue::new(&mut spawner_system_resource.rand_y_range.end)
                        .prefix("spawn y max: "),
                );

                ui.separator();

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

    let mut rng = rand::rng();

    let mut collider = ColliderShape::Cuboid;

    let mesh_renderer = if rng.random_bool(0.5) {
        if let Some(mesh_renderer) = mesh_renderer_store.mesh_renderers.get("default cube") {
            mesh_renderer
        } else {
            return;
        }
    } else {
        if let Some(mesh_renderer) = mesh_renderer_store.mesh_renderers.get("default sphere") {
            collider = ColliderShape::Sphere;
            mesh_renderer
        } else {
            return;
        }
    };

    for i in 0..spawner_system_resource.spawn_iters {
        let (x, y, z) = if (spawner_system_resource.rand_xz_range.start == 0.0
            && spawner_system_resource.rand_xz_range.end == 0.0)
            || (spawner_system_resource.rand_y_range.start == 0.0
                && spawner_system_resource.rand_y_range.end == 0.0)
        {
            (0.0, 0.0, 0.0)
        } else {
            (
                rng.random_range(spawner_system_resource.rand_xz_range.clone()),
                rng.random_range(spawner_system_resource.rand_y_range.clone()),
                rng.random_range(spawner_system_resource.rand_xz_range.clone()),
            )
        };

        let new_entity = commands.spawn((
            BevyTransform {
                0: Transform {
                    position: Vec3::from_array([x, y, z]),
                    scale: Vec3::from_array([spawner_system_resource.mesh_scale; 3]),
                    ..Default::default()
                },
            },
            BevyMeshRenderer { 0: *mesh_renderer },
            collider,
            DynamicRigidBody {
                mass: spawner_system_resource.mesh_scale * 10.0,
            },
            HideToggleProof {},
        ));
        spawner_system_resource
            .spawned_entities
            .insert(new_entity.id());

        rng.reseed();
    }
}
