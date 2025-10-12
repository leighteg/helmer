use helmer_engine::{
    ecs::{ecs_core::ECSCore, system::System},
    provided::components::Transform,
    runtime::{egui_integration::EguiResource, input_manager::InputManager},
};

pub struct InspectorSystem {}

impl System for InspectorSystem {
    fn name(&self) -> &str {
        "inspector system"
    }

    fn run(&mut self, dt: f32, ecs: &mut ECSCore, input_manager: &InputManager) {
        ecs.resource_scope::<EguiResource, _>(|ecs, egui_resouce| {
            egui_resouce.windows.push((
                Box::new(move |ui, ecs, _| {
                    for entity in ecs.get_all_entities() {
                        ui.heading(format!("entity {} -", entity));

                        for component in ecs.get_components_mut(entity) {
                            ui.heading(format!("{} -", component.short_name()));

                            if let Some(transform) =
                                component.as_any_mut().downcast_mut::<Transform>()
                            {
                                ui.label("position");
                                ui.add(
                                    egui::DragValue::new(&mut transform.position.x).prefix("x: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut transform.position.y).prefix("y: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut transform.position.z).prefix("z: "),
                                );

                                ui.label("rotation");
                                ui.add(
                                    egui::DragValue::new(&mut transform.rotation.x).prefix("x: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut transform.rotation.y).prefix("y: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut transform.rotation.z).prefix("z: "),
                                );

                                ui.label("scale");
                                ui.add(egui::DragValue::new(&mut transform.scale.x).prefix("x: "));
                                ui.add(egui::DragValue::new(&mut transform.scale.y).prefix("y: "));
                                ui.add(egui::DragValue::new(&mut transform.scale.z).prefix("z: "));
                            }
                        }

                        ui.separator();
                    }
                }),
                "inspector".to_string(),
            ));
        });
    }
}
