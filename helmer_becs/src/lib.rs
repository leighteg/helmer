use bevy_ecs::{component::Component, schedule::Schedule, world::World};
use helmer::runtime::{asset_server::AssetServer, runtime::Runtime};

use crate::systems::render_system::{RenderPacket, render_collection_system};

#[derive(Component)]
struct AnyComponent<T>(T);

pub mod systems;

pub fn helmer_becs_init(init_callback: fn(&mut World, &mut Schedule, &AssetServer)) {
    let world = World::new();
    let scheduler = Schedule::default();

    let mut runtime: Runtime<(World, Schedule)> = Runtime::new(
        (world, scheduler),
        move |runtime, (world, scheduler)| {
            world.register_resource::<RenderPacket>();

            scheduler.add_systems(render_collection_system);

            init_callback(
                world,
                scheduler,
                &runtime.asset_server.as_ref().unwrap().lock(),
            );
        },
        |dt, input_manager, (world, schedule)| (None, None),
        |new_size, (world, schedule)| {},
    );
    runtime.init();
}
