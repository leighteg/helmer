use std::{path::PathBuf, sync::Arc, time::Duration};

use glam::Quat;
use helmer_becs::ecs::{
    entity::Entity,
    resource::Resource,
    system::{Commands, ParamSet, Query, Res, ResMut},
};
use helmer_becs::{
    DraggedFile,
    components::Transform,
    systems::scene_system::SceneRoot,
    ui_integration::{UiResource, UiWindowSpec},
};
use helmer_ui::{DragValue, UiContext, UiTextAlign, UiTextStyle, label};
use parking_lot::Mutex;
use web_time::Instant;
use winit::keyboard::KeyCode;

const SCENE_LOADER_WINDOW_ID: &str = "scene_loader_window";
const SCENE_LOADER_REOPEN_HINT_WINDOW_ID: &str = "scene_loader_reopen_hint_window";

#[derive(Resource)]
pub struct SceneLoaderResource {
    spawned_scenes: Vec<SpawnedScene>,
    file_picker_in_flight: bool,
    file_picker_result: Arc<Mutex<Option<FilePickResult>>>,
    reopen_hint_until: Option<Instant>,
    ui_shared: Arc<Mutex<SceneLoaderUiShared>>,
}

impl Default for SceneLoaderResource {
    fn default() -> Self {
        Self {
            spawned_scenes: Vec::new(),
            file_picker_in_flight: false,
            file_picker_result: Arc::new(Mutex::new(None)),
            reopen_hint_until: None,
            ui_shared: Arc::new(Mutex::new(SceneLoaderUiShared::default())),
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

#[derive(Default)]
struct SceneLoaderUiShared {
    scenes: Vec<SceneLoaderUiScene>,
    actions: Vec<SceneLoaderUiAction>,
}

#[derive(Clone)]
struct SceneLoaderUiScene {
    entity: Entity,
    path: PathBuf,
    transform: Transform,
}

enum SceneLoaderUiAction {
    BrowseScene,
    RemoveScene(Entity),
    SetTransform(Entity, Transform),
    ClosedByChrome,
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

fn show_scene_loader_reopen_hint(scene_loader_res: &mut SceneLoaderResource) {
    scene_loader_res.reopen_hint_until = Some(Instant::now() + Duration::from_secs(3));
}

pub fn scene_loader_system(
    mut scene_loader_res: ResMut<SceneLoaderResource>,
    mut commands: Commands,
    ui_res: Res<UiResource>,
    mut dragged_file_res: ResMut<DraggedFile>,
    asset_server: helmer_becs::AssetServerParam,
    input_manager: Option<Res<helmer_becs::InputManagerResource>>,
    mut transform_queries: ParamSet<(
        Query<&helmer_becs::Transform>,
        Query<&mut helmer_becs::Transform>,
    )>,
) {
    let mut viewport_width = 1280.0f32;
    let mut tab_pressed = false;
    if let Some(input_manager) = input_manager.as_ref() {
        let input = input_manager.0.read();
        viewport_width = input.window_size.x.max(1) as f32;
        tab_pressed = input.just_pressed.contains(&KeyCode::Tab);
    }

    let ui_actions = {
        let mut shared = scene_loader_res.ui_shared.lock();
        std::mem::take(&mut shared.actions)
    };
    for action in ui_actions {
        match action {
            SceneLoaderUiAction::BrowseScene => {
                if !scene_loader_res.file_picker_in_flight {
                    scene_loader_res.file_picker_in_flight = true;
                    *scene_loader_res.file_picker_result.lock() = None;
                    let result_slot = scene_loader_res.file_picker_result.clone();
                    spawn_scene_file_picker(result_slot);
                }
            }
            SceneLoaderUiAction::RemoveScene(entity) => {
                if let Some(idx) = scene_loader_res
                    .spawned_scenes
                    .iter()
                    .position(|scene| scene.entity == entity)
                {
                    let scene = scene_loader_res.spawned_scenes.remove(idx);
                    commands.entity(scene.entity).despawn();
                }
            }
            SceneLoaderUiAction::SetTransform(entity, transform) => {
                if let Ok(mut transform_component) = transform_queries.p1().get_mut(entity) {
                    *transform_component = transform;
                }
            }
            SceneLoaderUiAction::ClosedByChrome => {
                show_scene_loader_reopen_hint(&mut scene_loader_res);
            }
        }
    }

    if tab_pressed {
        if ui_res.toggle_window(SCENE_LOADER_WINDOW_ID) {
            scene_loader_res.reopen_hint_until = None;
        } else {
            show_scene_loader_reopen_hint(&mut scene_loader_res);
        }
    }

    if let Some(until) = scene_loader_res.reopen_hint_until
        && Instant::now() >= until
    {
        scene_loader_res.reopen_hint_until = None;
    }
    let show_reopen_hint = scene_loader_res.reopen_hint_until.is_some();

    {
        let mut shared = scene_loader_res.ui_shared.lock();
        shared.scenes.clear();
        shared.scenes.reserve(scene_loader_res.spawned_scenes.len());
        for scene in scene_loader_res.spawned_scenes.iter() {
            let transform = transform_queries
                .p0()
                .get(scene.entity)
                .copied()
                .unwrap_or_default();
            shared.scenes.push(SceneLoaderUiScene {
                entity: scene.entity,
                path: scene.path.clone(),
                transform,
            });
        }
    }

    {
        let ui_shared_for_close = scene_loader_res.ui_shared.clone();
        ui_res.set_close_action(SCENE_LOADER_WINDOW_ID.to_string(), move || {
            ui_shared_for_close
                .lock()
                .actions
                .push(SceneLoaderUiAction::ClosedByChrome);
        });
    }

    {
        let ui_shared_for_window = scene_loader_res.ui_shared.clone();
        ui_res.add_window(
            UiWindowSpec {
                id: SCENE_LOADER_WINDOW_ID.to_string(),
                title: "Scene Loader".to_string(),
                position: [16.0, 16.0],
                size: [460.0, 720.0],
                closable: true,
                ..UiWindowSpec::default()
            },
            move |ui: &mut UiContext, _input_arc| {
                ui.heading("Scene Loader");
                let mut shared = ui_shared_for_window.lock();
                let mut actions = Vec::new();

                if ui.button("Browse Scene").clicked() {
                    actions.push(SceneLoaderUiAction::BrowseScene);
                }

                ui.separator();

                if shared.scenes.is_empty() {
                    ui.muted_label("No scenes loaded yet.");
                }

                for scene in shared.scenes.iter_mut() {
                    let scene_label = scene
                        .path
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| scene.path.to_string_lossy().to_string());

                    ui.push_id(scene.entity.to_bits(), |ui| {
                        ui.collapsing(scene_label, |ui| {
                            ui.label_with_style(
                                format!("Path: {}", scene.path.to_string_lossy()),
                                UiTextStyle {
                                    color: ui.theme().muted_text,
                                    font_size: 10.5,
                                    align_h: UiTextAlign::Start,
                                    align_v: UiTextAlign::Start,
                                    wrap: true,
                                },
                            );
                            ui.separator();

                            ui.label("Position:");
                            ui.horizontal_wrapped(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    if ui
                                        .add(
                                            DragValue::new(&mut scene.transform.position.x)
                                                .speed(0.1),
                                        )
                                        .changed()
                                    {
                                        actions.push(SceneLoaderUiAction::SetTransform(
                                            scene.entity,
                                            scene.transform,
                                        ));
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Y:");
                                    if ui
                                        .add(
                                            DragValue::new(&mut scene.transform.position.y)
                                                .speed(0.1),
                                        )
                                        .changed()
                                    {
                                        actions.push(SceneLoaderUiAction::SetTransform(
                                            scene.entity,
                                            scene.transform,
                                        ));
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Z:");
                                    if ui
                                        .add(
                                            DragValue::new(&mut scene.transform.position.z)
                                                .speed(0.1),
                                        )
                                        .changed()
                                    {
                                        actions.push(SceneLoaderUiAction::SetTransform(
                                            scene.entity,
                                            scene.transform,
                                        ));
                                    }
                                });
                            });

                            ui.label("Rotation:");
                            ui.horizontal_wrapped(|ui| {
                                let mut euler =
                                    scene.transform.rotation.to_euler(glam::EulerRot::XYZ);
                                let mut changed = false;

                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    changed |= ui.drag_angle(&mut euler.0).changed();
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Y:");
                                    changed |= ui.drag_angle(&mut euler.1).changed();
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Z:");
                                    changed |= ui.drag_angle(&mut euler.2).changed();
                                });

                                if changed {
                                    scene.transform.rotation = Quat::from_euler(
                                        glam::EulerRot::XYZ,
                                        euler.0,
                                        euler.1,
                                        euler.2,
                                    );
                                    actions.push(SceneLoaderUiAction::SetTransform(
                                        scene.entity,
                                        scene.transform,
                                    ));
                                }
                            });

                            ui.label("Scale:");
                            ui.horizontal_wrapped(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    if ui
                                        .add(
                                            DragValue::new(&mut scene.transform.scale.x)
                                                .speed(0.01),
                                        )
                                        .changed()
                                    {
                                        actions.push(SceneLoaderUiAction::SetTransform(
                                            scene.entity,
                                            scene.transform,
                                        ));
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Y:");
                                    if ui
                                        .add(
                                            DragValue::new(&mut scene.transform.scale.y)
                                                .speed(0.01),
                                        )
                                        .changed()
                                    {
                                        actions.push(SceneLoaderUiAction::SetTransform(
                                            scene.entity,
                                            scene.transform,
                                        ));
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Z:");
                                    if ui
                                        .add(
                                            DragValue::new(&mut scene.transform.scale.z)
                                                .speed(0.01),
                                        )
                                        .changed()
                                    {
                                        actions.push(SceneLoaderUiAction::SetTransform(
                                            scene.entity,
                                            scene.transform,
                                        ));
                                    }
                                });
                            });

                            ui.separator();
                            if ui.button("Remove Scene").clicked() {
                                actions.push(SceneLoaderUiAction::RemoveScene(scene.entity));
                            }
                        });
                    });

                    ui.add_space(8.0);
                }

                shared.actions.extend(actions);

                ui.separator();
                ui.wrapping_label(
                    "Drag and drop a .gltf/.glb file containing a scene or model to load",
                );
                ui.separator();
                ui.collapsing("keybinds", |ui| {
                    ui.group(|ui| {
                        ui.add(label("ctrl + I: toggle inspector").wrap(true));
                        ui.add(label("ctrl + G: toggle tunables/config UI").wrap(true));
                        ui.add(label("U: toggle UI pass").wrap(true));
                        ui.add(label("F11: toggle fullscreen").wrap(true));
                        ui.add(label("alt + F4: quit").wrap(true));
                    });
                });
                ui.collapsing("miscellaneous keybinds", |ui| {
                    ui.group(|ui| {
                        ui.add(label("1: lit view").wrap(true));
                        ui.add(label("2: lighting view").wrap(true));
                        ui.add(label("0: unlit view").wrap(true));
                        ui.separator();
                        ui.add(label("Z: toggle shadow pass").wrap(true));
                        ui.add(label("G: toggle GI pass").wrap(true));
                        ui.add(label("R: toggle reflection pass").wrap(true));
                        ui.add(label("H: toggle sky pass").wrap(true));
                        ui.add(label("U: toggle UI pass").wrap(true));
                        ui.add(label("F: toggle frustum culling").wrap(true));
                        ui.add(label("L: toggle LODs").wrap(true));
                    });
                });
            },
        );
    }

    ui_res.add_window(
        UiWindowSpec {
            id: SCENE_LOADER_REOPEN_HINT_WINDOW_ID.to_string(),
            title: String::new(),
            position: [((viewport_width - 460.0) * 0.5).max(12.0), 10.0],
            size: [460.0, 42.0],
            min_size: [460.0, 42.0],
            movable: false,
            resizable: false,
            closable: false,
            collapsible: false,
            input_passthrough: true,
            scrollable: false,
            ..UiWindowSpec::default()
        },
        |ui, _input_arc| {
            ui.label_with_style(
                "Scene Loader closed. Press Tab to reopen",
                UiTextStyle {
                    color: ui.theme().text,
                    font_size: 12.5,
                    align_h: UiTextAlign::Center,
                    align_v: UiTextAlign::Center,
                    wrap: true,
                },
            );
        },
    );
    if show_reopen_hint {
        ui_res.open_window(SCENE_LOADER_REOPEN_HINT_WINDOW_ID);
    } else {
        ui_res.close_window(SCENE_LOADER_REOPEN_HINT_WINDOW_ID);
    }

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
    asset_server: &helmer_becs::AssetServerParam<'w>,
    scene_loader_res: &mut SceneLoaderResource,
    path: PathBuf,
) {
    let scene_handle = asset_server.0.lock().load_scene(path.clone());

    let scene_root_entity = commands.spawn((Transform::default(), SceneRoot(scene_handle)));

    scene_loader_res.spawned_scenes.push(SpawnedScene {
        entity: scene_root_entity.id(),
        path,
    });
}
