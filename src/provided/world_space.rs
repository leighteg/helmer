use crate::ecs::component::Component;
use proc::Component;

#[derive(Debug, Component)]
pub struct Position(pub f32, pub f32, pub f32);

#[derive(Debug, Component)]
pub struct Rotation(pub f32, pub f32, pub f32);

#[derive(Debug, Component)]
pub struct Scale(pub u32, pub u32, pub u32);