use super::ecs_core::ECSCore;

pub trait System {
    fn name(&self) -> &str;
    fn run(&mut self, ecs: &mut ECSCore);
}