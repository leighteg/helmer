use proc::System;
use crate::{ecs::system::System, runtime::input_manager::InputManager};

pub struct RendererSystem {}

impl System for RendererSystem {
    fn name(&self) -> &str {
        "RendererSystem"
    }
    fn run(&mut self, dt: f32, ecs: &mut crate::ecs::ecs_core::ECSCore, input_manager: &InputManager) {
        
    }
}