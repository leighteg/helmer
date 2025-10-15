use bevy_ecs::{schedule::Schedule, world::World};
use helmer::runtime::{asset_server::AssetServer, runtime::Runtime};

pub fn helmer_ecs_init(init_callback: fn(&mut World, &mut Schedule, &AssetServer)) {
    let world = World::new();
    let scheduler = Schedule::default();

    let mut runtime = Runtime::new(
        (world, scheduler),
        move |runtime, (world, scheduler)| {
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
