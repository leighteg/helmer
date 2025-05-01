mod ecs;

use ecs::{component::Component, ecs_core::ECSCore};
use proc::Component;

#[derive(Debug, Component)]
struct Position(i64, i64);


fn main() {
    let mut ecs_core: ECSCore = ECSCore::new();
    let entity = ecs_core.create_entity();
    ecs_core.add_component(entity, Position(0,0));
    println!("sigma entity components: {:?}", ecs_core.get_components(entity));
    println!("getting sigma's position comp: {:?}", ecs_core.get_components_of_type::<Position>(entity))
}