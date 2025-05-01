use crate::ecs::ecs_core::ECSCore;

pub struct Runtime {
    ecs: ECSCore,
    window: Option<winit::window::Window>,
}