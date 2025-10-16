use std::sync::Arc;

use bevy_ecs::{component::Component, resource::Resource, schedule::Schedule, world::World};
use helmer::{
    provided::components::{ActiveCamera, Camera, Light, MeshRenderer, Transform},
    runtime::{asset_server::AssetServer, config::RuntimeConfig, runtime::Runtime},
};
use parking_lot::Mutex;

use crate::{
    egui_integration::EguiResource,
    systems::render_system::{RenderPacket, render_data_system},
};

/// A generic wrapper to turn existing data structures into Bevy Components or Resources.
#[derive(Component, Resource, Clone, Copy, Debug, Default)]
pub struct BevyWrapper<T>(pub T);

// Component Wrappers
pub type BevyTransform = BevyWrapper<Transform>;
pub type BevyCamera = BevyWrapper<Camera>;
pub type BevyActiveCamera = BevyWrapper<ActiveCamera>;
pub type BevyMeshRenderer = BevyWrapper<MeshRenderer>;
pub type BevyLight = BevyWrapper<Light>;

// Resource Wrappers
#[derive(Resource)]
pub struct BevyAssetServer(pub Arc<Mutex<AssetServer>>);
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct BevyRuntimeConfig(pub RuntimeConfig);

pub mod egui_integration;
pub mod systems;

pub fn helmer_becs_init(init_callback: fn(&mut World, &mut Schedule, &AssetServer)) {
    let world = World::new();
    let schedule = Schedule::default();

    let mut runtime: Runtime<(World, Schedule)> = Runtime::new(
        (world, schedule),
        move |runtime, (world, schedule)| {
            world.insert_resource::<BevyRuntimeConfig>(BevyRuntimeConfig(
                runtime.config.as_ref().clone(),
            ));
            world.insert_resource::<BevyAssetServer>(BevyAssetServer(
                runtime.asset_server.as_ref().unwrap().clone(),
            ));
            world.insert_resource::<RenderPacket>(RenderPacket::default());

            schedule.add_systems(render_data_system);

            init_callback(
                world,
                schedule,
                &runtime.asset_server.as_ref().unwrap().lock(),
            );
        },
        |dt, input_manager, (world, schedule)| {
            schedule.run(world);

            let render_data = {
                if let Some(mut packet) = world.get_resource_mut::<RenderPacket>() {
                    packet.0.take()
                } else {
                    None
                }
            };

            let egui_data = {
                if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
                    egui_res.render_data.take()
                } else {
                    None
                }
            };

            (render_data, egui_data)
        },
        |new_size, (world, schedule)| {},
    );
    runtime.init();
}
