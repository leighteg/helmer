use bevy_ecs::{
    entity::Entity,
    resource::Resource,
    system::{Commands, Local, ResMut},
};
use helmer::provided::components::Transform;
use helmer_becs::{
    BevyAssetServer, BevyTransform, DraggedFile, egui_integration::EguiResource,
    systems::scene_system::SceneRoot,
};

#[derive(Default, Resource)]
pub struct SceneLoaderResource {
    spawned_entity: Option<Entity>,
}

pub fn scene_loader_system(
    mut local: ResMut<SceneLoaderResource>,
    mut commands: Commands,
    mut egui_res: ResMut<EguiResource>,
    mut dragged_file_res: ResMut<DraggedFile>,
    asset_server: ResMut<BevyAssetServer>,
) {
    // --- UI ---
    egui_res.windows.push((
        Box::new(move |ui, world, _input_arc| {
            world.resource_scope::<DraggedFile, _>(|world, dragged_file_res| {});

            world.resource_scope::<SceneLoaderResource, _>(|world, scene_loader_res| {
                if let Some(spawned_entity) = scene_loader_res.spawned_entity {
                    ui.label("scene loaded");
                }
            });
        }),
        "scene loader".to_string(),
    ));

    if let Some(dragged_file_path) = dragged_file_res.0.take() {
        let scene_handle = asset_server.0.lock().load_scene(dragged_file_path);

        if let Some(prev_spawned_entity) = local.spawned_entity {
            commands.entity(prev_spawned_entity).despawn();
        }

        let scene_root_entity = commands.spawn((
            BevyTransform {
                0: Transform::default(),
            },
            SceneRoot(scene_handle),
        ));
        local.spawned_entity = Some(scene_root_entity.id());
    }
}
