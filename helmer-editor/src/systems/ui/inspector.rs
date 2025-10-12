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

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        ecs.resource_scope::<EguiResource, _>(|ecs, egui_resource| {
            egui_resource.windows.push((
                Box::new(move |ui, ecs, _| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            let mut entities = ecs.get_all_entities();
                            entities.sort();
                            
                            for entity in entities {
                                let header_label = format!("Entity {}", entity);

                                egui::CollapsingHeader::new(header_label)
                                    .default_open(false)
                                    .show(ui, |ui| {
                                        for component in ecs.get_components_mut(entity) {
                                            let comp_label = component.short_name();

                                            egui::CollapsingHeader::new(comp_label)
                                                .default_open(false)
                                                .show(ui, |ui| {
                                                    if let Some(transform) =
                                                        component.as_any_mut().downcast_mut::<Transform>()
                                                    {
                                                        ui.label("Position");
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.position.x)
                                                                .prefix("x: "),
                                                        );
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.position.y)
                                                                .prefix("y: "),
                                                        );
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.position.z)
                                                                .prefix("z: "),
                                                        );

                                                        ui.label("Rotation");
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.rotation.x)
                                                                .prefix("x: "),
                                                        );
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.rotation.y)
                                                                .prefix("y: "),
                                                        );
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.rotation.z)
                                                                .prefix("z: "),
                                                        );

                                                        ui.label("Scale");
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.scale.x)
                                                                .prefix("x: "),
                                                        );
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.scale.y)
                                                                .prefix("y: "),
                                                        );
                                                        ui.add(
                                                            egui::DragValue::new(&mut transform.scale.z)
                                                                .prefix("z: "),
                                                        );
                                                    } else {
                                                        ui.label("No inspector available for this component.");
                                                    }
                                                });
                                        }
                                    });
                            }
                        });
                }),
                "inspector".to_string(),
            ));
        });
    }
}
