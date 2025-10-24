use bevy_ecs::{resource::Resource, world::World};
use egui::Context;
use helmer::{graphics::renderer_common::common::EguiRenderData, runtime::input_manager::InputManager};

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