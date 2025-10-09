use helmer_engine::{ecs::{ecs_core::ECSCore, system::System}, runtime::{egui_integration::EguiResource, input_manager::InputManager}};

pub struct InspectorSystem {}

impl System for InspectorSystem {
    fn name(&self) -> &str {
        "inspector system"
    }

    fn run(&mut self, dt: f32, ecs: &mut ECSCore, input_manager: &InputManager) {
        ecs.resource_scope::<EguiResource, _>(|ecs, egui_resouce| {
            egui_resouce.windows.push((
                |ui, ecs, _| {
                    ui.label(format!(
                        "{} resources registered",
                        ecs.resource_pool.resources.len()
                    ));
                },
                "inspector".to_string(),
            ));
        });
    }
}
