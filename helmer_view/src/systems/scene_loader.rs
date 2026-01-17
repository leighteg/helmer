use std::path::PathBuf;

use bevy_ecs::{
    entity::Entity,
    resource::Resource,
    system::{Commands, ResMut},
};
use glam::Quat;
use helmer::provided::components::Transform;
use helmer_becs::{
    BevyAssetServer, BevyTransform, DraggedFile, egui_integration::EguiResource,
    systems::scene_system::SceneRoot,
};

#[derive(Default, Resource)]
pub struct SceneLoaderResource {
    spawned_scenes: Vec<SpawnedScene>,
}

#[derive(Clone)]
pub struct SpawnedScene {
    entity: Entity,
    path: PathBuf,
}

pub fn scene_loader_system(
    mut scene_loader_res: ResMut<SceneLoaderResource>,
    mut commands: Commands,
    mut egui_res: ResMut<EguiResource>,
    mut dragged_file_res: ResMut<DraggedFile>,
    asset_server: ResMut<BevyAssetServer>,
) {
    // --- UI ---
    egui_res.windows.push((
        Box::new(move |ui, world, _input_arc| {
            ui.heading("Scene Loader");
            ui.separator();

            world.resource_scope::<SceneLoaderResource, _>(|world, mut scene_loader_res| {
                let mut scenes_to_remove = Vec::new();

                for (idx, scene) in scene_loader_res.spawned_scenes.iter().enumerate() {
                    ui.group(|ui| {
                        ui.label(format!("Scene: {:?}", scene.path));
                        ui.separator();

                        // Query transform for this specific entity
                        if let Some(mut bevy_transform) =
                            world.get_mut::<BevyTransform>(scene.entity)
                        {
                            let transform = &mut bevy_transform.0;

                            ui.label("Position:");
                            ui.horizontal(|ui| {
                                ui.label("X:");
                                ui.add(egui::DragValue::new(&mut transform.position.x).speed(0.1));
                                ui.label("Y:");
                                ui.add(egui::DragValue::new(&mut transform.position.y).speed(0.1));
                                ui.label("Z:");
                                ui.add(egui::DragValue::new(&mut transform.position.z).speed(0.1));
                            });

                            ui.label("Rotation:");
                            ui.horizontal(|ui| {
                                // Convert current quaternion to Euler angles (in radians)
                                let mut euler = transform.rotation.to_euler(glam::EulerRot::XYZ);

                                ui.label("X:");
                                ui.drag_angle(&mut euler.0);
                                ui.label("Y:");
                                ui.drag_angle(&mut euler.1);
                                ui.label("Z:");
                                ui.drag_angle(&mut euler.2);

                                // Convert back to quaternion
                                transform.rotation = Quat::from_euler(
                                    glam::EulerRot::XYZ,
                                    euler.0,
                                    euler.1,
                                    euler.2,
                                );
                            });

                            ui.label("Scale:");
                            ui.horizontal(|ui| {
                                ui.label("X:");
                                ui.add(egui::DragValue::new(&mut transform.scale.x).speed(0.01));
                                ui.label("Y:");
                                ui.add(egui::DragValue::new(&mut transform.scale.y).speed(0.01));
                                ui.label("Z:");
                                ui.add(egui::DragValue::new(&mut transform.scale.z).speed(0.01));
                            });
                        }

                        ui.separator();
                        if ui.button("Remove Scene").clicked() {
                            scenes_to_remove.push(idx);
                        }
                    });

                    ui.add_space(10.0);
                }

                // Remove scenes in reverse order to maintain valid indices
                for idx in scenes_to_remove.into_iter().rev() {
                    let scene = scene_loader_res.spawned_scenes.remove(idx);
                    world.commands().entity(scene.entity).despawn();
                }
            });

            ui.separator();
            ui.label("Drag and drop a .gltf/.glb file containing a scene or model to load");
            ui.separator();
            ui.collapsing("keybinds", |ui| {
                ui.group(|ui| {
                    ui.label("ctrl + I: toggle inspector");
                    ui.label("ctrl + G: toggle tunables/config UI");
                    ui.label("U: toggle UI pass");
                    ui.label("F11: toggle fullscreen");
                    ui.label("alt + F4: quit");
                });
            });
            ui.collapsing("miscellaneous keybinds", |ui| {
                ui.group(|ui| {
                    ui.label("1: lit view");
                    ui.label("2: lighting view");
                    ui.label("0: unlit view");
                    ui.separator();
                    ui.label("Z: toggle shadow pass");
                    ui.label("G: toggle GI pass");
                    ui.label("R: toggle reflection pass");
                    ui.label("H: toggle sky pass");
                    ui.label("U: toggle UI pass");
                    ui.label("F: toggle frustum culling");
                    ui.label("L: toggle LODs");
                });
            });
        }),
        "Scene Loader".to_string(),
    ));

    // Handle new file drag
    if let Some(dragged_file_path) = dragged_file_res.0.take() {
        let path_string = dragged_file_path.clone();
        let scene_handle = asset_server.0.lock().load_scene(dragged_file_path);

        let scene_root_entity = commands.spawn((
            BevyTransform {
                0: Transform::default(),
            },
            SceneRoot(scene_handle),
        ));

        scene_loader_res.spawned_scenes.push(SpawnedScene {
            entity: scene_root_entity.id(),
            path: path_string,
        });
    }
}
