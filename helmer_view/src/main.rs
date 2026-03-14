#[cfg(target_os = "linux")]
use std::env;

use becs_bench::systems::{config_toggle::config_toggle_system, freecam::freecam_system};
use glam::{Quat, Vec3};
use helmer_becs::{
    components::{ActiveCamera, Light, Transform},
    helmer_becs_init_with_runtime,
};
#[cfg(target_arch = "wasm32")]
use helmer_window::runtime::wasm_harness::WasmHarnessConfig;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

use crate::systems::scene_loader::{SceneLoaderResource, scene_loader_system};

pub mod systems;

fn run() {
    #[cfg(target_os = "linux")]
    unsafe {
        env::set_var("HELMER_FORCE_UNIX_BACKEND", "x11")
    };

    helmer_becs_init_with_runtime(
        |world, schedule, asset_server| {
            let camera_pos = Vec3::new(0.0, 1.5, -5.0);
            let camera_forward = (Vec3::ZERO - camera_pos).normalize_or_zero();
            let camera_rot = Quat::from_rotation_arc(Vec3::Z, camera_forward);
            let camera_entity = world.spawn((
                Transform {
                    position: camera_pos,
                    rotation: camera_rot,
                    scale: Vec3::ONE,
                },
                helmer_becs::Camera::default(),
                ActiveCamera,
            ));

            let sun_rotation = Quat::from_euler(
                glam::EulerRot::YXZ,
                20.0f32.to_radians(),
                -50.0f32.to_radians(),
                20.0f32.to_radians(),
            );

            let sun_entity = world.spawn((
                Transform {
                    position: glam::Vec3::new(0.0, 0.0, 0.0),
                    rotation: sun_rotation,
                    scale: glam::Vec3::ONE,
                },
                Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0),
            ));

            // resources init
            world.insert_resource(SceneLoaderResource::default());

            // systems init
            schedule.add_systems(config_toggle_system);
            schedule.add_systems(scene_loader_system);
            schedule.add_systems(freecam_system);
        },
        |runtime| {
            #[cfg(target_arch = "wasm32")]
            {
                let mut config = WasmHarnessConfig::default();
                config.mount_id = Some("helmer-root".to_string());
                config.canvas_id = Some("helmer-canvas".to_string());
                if let Err(err) = config.apply_to_runtime(runtime) {
                    eprintln!("wasm harness setup failed: {}", err);
                }
            }
        },
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    run();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_start() {
    run();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
