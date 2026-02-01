use std::{path::PathBuf, sync::Arc};

use bevy_ecs::{
    entity::Entity,
    resource::Resource,
    system::{Commands, ResMut},
};
use glam::Quat;
use helmer::provided::components::Transform;
use helmer_becs::{
    BevyAssetServerParam, BevyTransform, DraggedFile,
    egui_integration::{EguiResource, EguiWindowSpec},
    systems::scene_system::SceneRoot,
};
use parking_lot::Mutex;

#[derive(Resource)]
pub struct SceneLoaderResource {
    spawned_scenes: Vec<SpawnedScene>,
    file_picker_in_flight: bool,
    // None = pending, Some = completed (picked or canceled)
    file_picker_result: Arc<Mutex<Option<FilePickResult>>>,
}

impl Default for SceneLoaderResource {
    fn default() -> Self {
        Self {
            spawned_scenes: Vec::new(),
            file_picker_in_flight: false,
            file_picker_result: Arc::new(Mutex::new(None)),
        }
    }
}

#[derive(Clone)]
pub struct SpawnedScene {
    entity: Entity,
    path: PathBuf,
}

enum FilePickResult {
    Canceled,
    Picked(PickedFile),
}

struct PickedFile {
    path: PathBuf,
    bytes: Option<Vec<u8>>,
}

#[cfg(not(target_arch = "wasm32"))]
async fn pick_scene_file() -> Option<PickedFile> {
    let file = rfd::AsyncFileDialog::new()
        .add_filter("Scene/Model", &["glb", "gltf"])
        .pick_file()
        .await?;
    Some(PickedFile {
        path: file.path().to_owned(),
        bytes: None,
    })
}

#[cfg(target_arch = "wasm32")]
async fn pick_scene_file() -> Option<PickedFile> {
    let file = rfd::AsyncFileDialog::new()
        .add_filter("Scene/Model", &["glb", "gltf"])
        .pick_file()
        .await?;
    let bytes = file.read().await;
    Some(PickedFile {
        path: PathBuf::from(file.file_name()),
        bytes: Some(bytes),
    })
}

fn spawn_scene_file_picker(result_slot: Arc<Mutex<Option<FilePickResult>>>) {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move {
            let picked = pick_scene_file().await;
            let result = match picked {
                Some(path) => FilePickResult::Picked(path),
                None => FilePickResult::Canceled,
            };
            *result_slot.lock() = Some(result);
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(move || {
            let picked = pollster::block_on(pick_scene_file());
            let result = match picked {
                Some(path) => FilePickResult::Picked(path),
                None => FilePickResult::Canceled,
            };
            *result_slot.lock() = Some(result);
        });
    }
}

pub fn scene_loader_system(
    mut scene_loader_res: ResMut<SceneLoaderResource>,
    mut commands: Commands,
    mut egui_res: ResMut<EguiResource>,
    mut dragged_file_res: ResMut<DraggedFile>,
    asset_server: BevyAssetServerParam,
) {
    // --- UI ---
    egui_res.windows.push((
        Box::new(|ui, world, _input_arc| {
            ui.heading("Scene Loader");
            if ui.button("browse").clicked() {
                let result_slot = world.resource_scope::<SceneLoaderResource, _>(
                    |_world, mut scene_loader_res| {
                        if scene_loader_res.file_picker_in_flight {
                            return None;
                        }
                        scene_loader_res.file_picker_in_flight = true;
                        *scene_loader_res.file_picker_result.lock() = None;
                        Some(scene_loader_res.file_picker_result.clone())
                    },
                );

                if let Some(result_slot) = result_slot {
                    spawn_scene_file_picker(result_slot);
                }
            }
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
        EguiWindowSpec {
            id: "Scene Loader".to_string(),
            title: "Scene Loader".to_string(),
        },
    ));

    if scene_loader_res.file_picker_in_flight {
        let result = {
            let mut guard = scene_loader_res.file_picker_result.lock();
            guard.take()
        };
        if let Some(result) = result {
            scene_loader_res.file_picker_in_flight = false;
            if let FilePickResult::Picked(picked) = result {
                let PickedFile { path, bytes } = picked;
                #[cfg(not(target_arch = "wasm32"))]
                let _ = bytes;
                #[cfg(target_arch = "wasm32")]
                if let Some(bytes) = bytes {
                    asset_server.0.lock().store_virtual_asset(&path, bytes);
                }
                spawn_scene_from_path(&mut commands, &asset_server, &mut scene_loader_res, path);
            }
        }
    }

    // Handle new file drag
    if let Some(dragged_file_path) = dragged_file_res.0.take() {
        spawn_scene_from_path(
            &mut commands,
            &asset_server,
            &mut scene_loader_res,
            dragged_file_path,
        );
    }
}

fn spawn_scene_from_path<'w>(
    commands: &mut Commands,
    asset_server: &BevyAssetServerParam<'w>,
    scene_loader_res: &mut SceneLoaderResource,
    path: PathBuf,
) {
    let scene_handle = asset_server.0.lock().load_scene(path.clone());

    let scene_root_entity = commands.spawn((
        BevyTransform {
            0: Transform::default(),
        },
        SceneRoot(scene_handle),
    ));

    scene_loader_res.spawned_scenes.push(SpawnedScene {
        entity: scene_root_entity.id(),
        path,
    });
}
