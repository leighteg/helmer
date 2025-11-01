use std::sync::Arc;

use bevy_ecs::{component::Component, resource::Resource, schedule::Schedule, world::World};
use helmer::{
    provided::components::{ActiveCamera, Camera, Light, MeshRenderer, Transform},
    runtime::{
        asset_server::AssetServer, config::RuntimeConfig, input_manager::InputManager,
        runtime::Runtime,
    },
};
use parking_lot::{Mutex, RwLock};

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
#[derive(Resource)]
pub struct BevyInputManager(pub Arc<RwLock<InputManager>>);
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct BevyRuntimeConfig(pub RuntimeConfig);

// resources
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct DeltaTime(pub f32);

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
            world.insert_resource::<BevyInputManager>(BevyInputManager(
                runtime.input_manager.clone(),
            ));
            world.insert_resource::<DeltaTime>(DeltaTime(1.0));
            world.insert_resource::<RenderPacket>(RenderPacket::default());

            schedule.add_systems(render_data_system);

            init_callback(
                world,
                schedule,
                &runtime.asset_server.as_ref().unwrap().lock(),
            );
        },
        |dt, (world, schedule)| {
            world.resource_mut::<DeltaTime>().0 = dt;

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
        |new_size, (world, _schedule)| {
            for (mut camera, _) in world
                .query::<(&mut BevyCamera, &BevyActiveCamera)>()
                .iter_mut(world)
            {
                camera.0.aspect_ratio = new_size.width as f32 / new_size.height as f32;
            }
        },
    );
    runtime.init();
}
