use super::ecs_core::ECSCore;

pub enum AccessType {
    Read,
    Write,
}
pub trait System: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, ecs: &mut ECSCore);
}