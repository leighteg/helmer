use becs_bench::systems::{config_toggle::config_toggle_system, freecam::freecam_system};
use helmer::provided::components::ActiveCamera;
use helmer_becs::{BevyActiveCamera, BevyCamera, BevyTransform, helmer_becs_init};

use crate::systems::model_loader::{SceneLoaderResource, scene_loader_system};

pub mod systems;

fn main() {
    helmer_becs_init(|world, schedule, asset_server| {
        let camera_entity = world.spawn((
            BevyTransform::default(),
            BevyCamera::default(),
            BevyActiveCamera { 0: ActiveCamera {} },
        ));

        // resources init
        world.insert_resource(SceneLoaderResource::default());

        // systems init
        schedule.add_systems(config_toggle_system);
        schedule.add_systems(freecam_system);
        schedule.add_systems(scene_loader_system);
    });
}
