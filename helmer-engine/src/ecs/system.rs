use std::sync::Arc;

use crate::runtime::input_manager::{self, InputManager};

use super::ecs_core::ECSCore;

pub enum AccessType {
    Read,
    Write,
}
pub trait System: Send + Sync {
    fn name(&self) -> &str;
    fn run(&mut self, dt: f32, ecs: &mut ECSCore, input_manager: &InputManager);
}