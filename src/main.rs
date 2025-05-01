mod ecs;

use helmer_rs::{ecs::component::Component, runtime::Runtime};
use proc::Component;

#[derive(Debug, Component)]
struct Position(i64, i64);


fn main() {
    let mut runtime = Runtime::new();
    let entity = runtime.ecs.create_entity();
    runtime.ecs.add_component(entity, Position(0,0));
    println!("sigma entity components: {:?}", runtime.ecs.get_components(entity));
    println!("getting sigma's position comp: {:?}", runtime.ecs.get_component::<Position>(entity))
}