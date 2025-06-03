use proc::System;
use crate::ecs::system::System;

pub struct RendererSystem {}

impl System for RendererSystem {
    fn name(&self) -> &str {
        "RendererSystem"
    }
    fn run(&self, ecs: &mut crate::ecs::ecs_core::ECSCore) {
        
    }
}