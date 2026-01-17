use std::env;

use becs_bench::systems::{config_toggle::config_toggle_system, freecam::freecam_system};
use glam::Quat;
use helmer::provided::components::{ActiveCamera, Light, Transform};
use helmer_becs::{BevyActiveCamera, BevyCamera, BevyLight, BevyTransform, helmer_becs_init};

use crate::systems::scene_loader::{SceneLoaderResource, scene_loader_system};

pub mod systems;

fn main() {
    #[cfg(target_os = "linux")]
    unsafe {
        env::set_var("HELMER_FORCE_UNIX_BACKEND", "x11")
    };

    helmer_becs_init(|world, schedule, asset_server| {
        let camera_entity = world.spawn((
            BevyTransform::default(),
            BevyCamera::default(),
            BevyActiveCamera { 0: ActiveCamera {} },
        ));

        let sun_rotation = Quat::from_euler(
            glam::EulerRot::YXZ,
            20.0f32.to_radians(),  // Y rotation - very slight side angle
            -50.0f32.to_radians(), // X rotation - steeper downward angle
            20.0f32.to_radians(),  // Z rotation - no roll
        );

        let sun_entity = world.spawn((
            BevyTransform {
                0: Transform {
                    position: glam::Vec3::new(0.0, 0.0, 0.0),
                    rotation: sun_rotation,
                    scale: glam::Vec3::ONE,
                },
            },
            BevyLight {
                0: Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0),
            },
        ));

        // resources init
        world.insert_resource(SceneLoaderResource::default());

        // systems init
        schedule.add_systems(config_toggle_system);
        schedule.add_systems(freecam_system);
        schedule.add_systems(scene_loader_system);
    });
}
