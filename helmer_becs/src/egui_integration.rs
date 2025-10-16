use bevy_ecs::{resource::Resource, world::World};
use egui::{ClippedPrimitive, Context, RawInput, TexturesDelta};
use helmer::{graphics::renderer::renderer::EguiRenderData, runtime::{config::RuntimeConfig, input_manager::InputManager}};
use std::sync::{Arc, atomic::Ordering, mpsc};
use winit::keyboard::KeyCode;

#[derive(Default, Resource)]
pub struct EguiResource {
    pub ctx: Context,
    pub render_data: Option<EguiRenderData>,
    pub windows: Vec<(
        Box<dyn FnMut(&mut egui::Ui, &World, &InputManager) + Send + Sync>,
        String,
    )>,
    pub stats_ui: bool,
}

pub struct EguiSystem {}