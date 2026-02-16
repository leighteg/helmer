use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::name::Name;
use bevy_ecs::prelude::ReflectComponent;
use bevy_ecs::resource::Resource;
use bevy_reflect::{PartialReflect, Reflect, ReflectMut, TypeRegistry};
use egui::Id;
use helmer::provided::components::{Light, LightType};
use std::any::TypeId;

use crate::egui_integration::{EguiResource, EguiWindowSpec};
use crate::{BevyActiveCamera, BevyCamera, BevyLight, BevyMeshRenderer, BevyTransform};

#[derive(Resource, Default)]
pub struct InspectorSelectedEntityResource(pub Option<Entity>);

pub struct InspectorUI {}

impl InspectorUI {
    pub fn add_window(egui_res: &mut EguiResource) {
        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                fn reflect_ui(
                    registry: &TypeRegistry,
                    value: &mut dyn PartialReflect,
                    ui: &mut egui::Ui,
                ) {
                    let type_id = value.get_represented_type_info().unwrap().type_id();
                    let _registration = registry.get(type_id).expect("Unregistered type");

                    // Handle primitives via downcast
                    if let Some(v) = value.try_downcast_mut::<bool>() {
                        ui.checkbox(v, "");
                        return;
                    }
                    if let Some(v) = value.try_downcast_mut::<f32>() {
                        ui.add(egui::DragValue::new(v).speed(0.1));
                        return;
                    }
                    if let Some(v) = value.try_downcast_mut::<f64>() {
                        ui.add(egui::DragValue::new(v).speed(0.1));
                        return;
                    }
                    if let Some(v) = value.try_downcast_mut::<i32>() {
                        ui.add(egui::DragValue::new(v));
                        return;
                    }
                    if let Some(v) = value.try_downcast_mut::<u32>() {
                        ui.add(egui::DragValue::new(v));
                        return;
                    }
                    if let Some(v) = value.try_downcast_mut::<i64>() {
                        ui.add(egui::DragValue::new(v));
                        return;
                    }
                    if let Some(v) = value.try_downcast_mut::<u64>() {
                        ui.add(egui::DragValue::new(v));
                        return;
                    }
                    if let Some(v) = value.try_downcast_mut::<String>() {
                        ui.text_edit_singleline(v);
                        return;
                    }

                    // Handle arrays
                    if let ReflectMut::Array(array) = value.reflect_mut() {
                        ui.collapsing("Array", |ui| {
                            for i in 0..array.len() {
                                ui.horizontal(|ui| {
                                    ui.label(format!("{}:", i));
                                    reflect_ui(registry, array.get_mut(i).unwrap(), ui);
                                });
                            }
                        });
                        return;
                    }

                    // Handle lists (e.g., Vec)
                    if let ReflectMut::List(list) = value.reflect_mut() {
                        ui.collapsing("List", |ui| {
                            for i in 0..list.len() {
                                ui.horizontal(|ui| {
                                    ui.label(format!("{}:", i));
                                    reflect_ui(registry, list.get_mut(i).unwrap(), ui);
                                });
                            }
                        });
                        return;
                    }

                    // Handle enums
                    if let ReflectMut::Enum(enum_ref) = value.reflect_mut() {
                        let current_index = enum_ref.variant_index();
                        let current_name = enum_ref.variant_name().to_string();
                        let enum_info = enum_ref.get_represented_enum_info().unwrap();

                        let variant_names: Vec<String> = (0..enum_info.variant_len())
                            .map(|i| enum_info.variant_at(i).unwrap().name().to_string())
                            .collect();
                        let mut selected = current_index;

                        egui::ComboBox::from_id_salt("enum_variant") // (uses parent ID stack)
                            .selected_text(&current_name)
                            .show_ui(ui, |ui| {
                                for (i, name) in variant_names.iter().enumerate() {
                                    ui.selectable_value(&mut selected, i, name);
                                }
                            });

                        // Show fields
                        let field_count = enum_ref.field_len();
                        for i in 0..field_count {
                            let field_name = enum_ref
                                .name_at(i)
                                .unwrap_or(&format!("field {}", i))
                                .to_string();
                            ui.horizontal(|ui| {
                                ui.label(&field_name);
                            });
                        }
                        return;
                    }

                    // Handle structs
                    if let ReflectMut::Struct(struct_ref) = value.reflect_mut() {
                        let field_count = struct_ref.field_len();
                        for i in 0..field_count {
                            if let Some(name) = struct_ref.name_at(i) {
                                let name = name.to_string();
                                ui.horizontal(|ui| {
                                    ui.label(&name);
                                });
                            }
                        }
                        return;
                    }

                    // Handle tuple structs
                    if let ReflectMut::TupleStruct(tuple_struct) = value.reflect_mut() {
                        for i in 0..tuple_struct.field_len() {
                            ui.horizontal(|ui| {
                                ui.label(format!("field {}", i));
                                if let Some(field) = tuple_struct.field_mut(i) {
                                    reflect_ui(registry, field, ui);
                                }
                            });
                        }
                        return;
                    }

                    // Handle tuples
                    if let ReflectMut::Tuple(tuple) = value.reflect_mut() {
                        for i in 0..tuple.field_len() {
                            if let Some(field) = tuple.field_mut(i) {
                                reflect_ui(registry, field, ui);
                            }
                        }
                        return;
                    }

                    // Fallback
                    ui.label("Unsupported type");
                }

                // Use AppTypeRegistry instead of TypeRegistry
                let app_registry =
                    if let Some(r) = world.get_resource::<bevy_ecs::prelude::AppTypeRegistry>() {
                        r.clone()
                    } else {
                        ui.label("AppTypeRegistry resource is missing");
                        return;
                    };

                // Collect entity and component info first, before any mutable borrows
                let mut entities: Vec<Entity> = world
                    .query::<bevy_ecs::world::EntityRef>()
                    .iter(world)
                    .map(|e| e.id())
                    .collect();
                entities.sort_by_key(|e| e.to_bits());

                // Collect component info for all entities
                let entity_components: Vec<(
                    Entity,
                    Vec<(bevy_ecs::component::ComponentId, String, std::any::TypeId)>,
                )> = entities
                    .iter()
                    .map(|&entity| {
                        let entity_ref = world.entity(entity);
                        let component_ids: Vec<bevy_ecs::component::ComponentId> = entity_ref
                            .archetype()
                            .components()
                            .into_iter()
                            .copied()
                            .collect();

                        let components_info: Vec<_> = component_ids
                            .into_iter()
                            .filter_map(|component_id| {
                                world.components().get_info(component_id).and_then(|info| {
                                    info.type_id().map(|type_id| {
                                        // Match known custom types for real names (bypass placeholder)
                                        let short_name = if type_id == TypeId::of::<BevyTransform>()
                                        {
                                            "Transform".to_string()
                                        } else if type_id == TypeId::of::<BevyCamera>() {
                                            "Camera".to_string()
                                        } else if type_id == TypeId::of::<BevyActiveCamera>() {
                                            "Active Camera".to_string()
                                        } else if type_id == TypeId::of::<BevyLight>() {
                                            match world
                                                .entity(entity)
                                                .get::<BevyLight>()
                                                .unwrap()
                                                .0
                                                .light_type
                                            {
                                                LightType::Directional => {
                                                    "Directional Light".to_string()
                                                }
                                                LightType::Spot { angle } => {
                                                    "Spot Light".to_string()
                                                }
                                                LightType::Point => "Point Light".to_string(),
                                            }
                                        } else if type_id == TypeId::of::<BevyMeshRenderer>() {
                                            "Mesh Renderer".to_string()
                                        } else {
                                            // Fallback for reflectable/unknown (safe now, as knowns are handled)
                                            info.name()
                                                .rsplit("::")
                                                .next()
                                                .unwrap_or("Unknown")
                                                .to_string()
                                        };
                                        (component_id, short_name, type_id)
                                    })
                                })
                            })
                            .collect();

                        (entity, components_info)
                    })
                    .collect();

                // Get the currently selected entity (if any)
                let selected_entity = world
                    .get_resource::<InspectorSelectedEntityResource>()
                    .and_then(|res| res.0);

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (entity, components) in entity_components {
                        let is_selected = Some(entity) == selected_entity;

                        // Give each entity its own ID scope
                        ui.push_id(entity, |ui| {
                            // Draw a highlight frame if this entity is selected
                            let frame = if is_selected {
                                egui::Frame::group(ui.style())
                                    .fill(ui.visuals().selection.bg_fill)
                                    .stroke(egui::Stroke::new(
                                        1.0,
                                        ui.visuals().selection.stroke.color,
                                    ))
                            } else {
                                egui::Frame::none()
                            };

                            frame.show(ui, |ui| {
                                // Entity header (expandable)
                                let header_title = if let Some(name) = world.entity(entity).get::<Name>() {
                                    &name.to_string()
                                } else if let Some(_) = world.entity(entity).get::<BevyActiveCamera>() {
                                    "Active Camera"
                                } else if let Some(light) = world.entity(entity).get::<BevyLight>() {
                                    match light.0.light_type {
                                        LightType::Directional => "Directional Light",
                                        LightType::Point => "Spot Light",
                                        LightType::Spot { angle: _ } => "Spot Light",
                                    }
                                } else { &{
                                    format!(
                                    "Entity {}",
                                    entity.to_bits()
                                )
                                }};

                                let header = egui::CollapsingHeader::new(header_title)
                                .default_open(is_selected); // auto-open if selected

                                let response = header.show(ui, |ui| {
                                    for (component_id, short_name, type_id) in components {
                                    let unique_id = Id::new((entity, type_id));  // ← Unique per entity + component type

                                    // Check if this is a reflectable component
                                    let reflect_component_opt = {
                                        let registry = app_registry.read();
                                        registry.get(type_id).and_then(|reg| reg.data::<ReflectComponent>().cloned())
                                    };

                                    if let Some(reflect_component) = reflect_component_opt {
                                        // Reflectable component: Use reflection system
                                        let mut entity_mut = world.entity_mut(entity);
                                        if let Some(mut component_reflect) = reflect_component.reflect_mut(&mut entity_mut) {
                                            let registry = app_registry.read();
                                            egui::CollapsingHeader::new(&short_name)
                                                .id_salt(unique_id)  // ← Make header ID unique
                                                .show(ui, |ui| {
                                                    ui.push_id(unique_id, |ui| {  // ← Push ID for children
                                                        reflect_ui(&registry, component_reflect.as_partial_reflect_mut(), ui);
                                                    });
                                                });
                                        }
                                    } else {
                                        // Non-reflectable: Handle custom types via TypeId matching
                                        let mut entity_mut = world.entity_mut(entity);

                                        if type_id == TypeId::of::<BevyTransform>() {
                                            if let Some(mut transform_wrapper) = entity_mut.get_mut::<BevyTransform>() {
                                                egui::CollapsingHeader::new(&short_name)
                                                    .id_salt(unique_id)  // ← Make header ID unique
                                                    .show(ui, |ui| {
                                                        ui.push_id(unique_id, |ui| {  // ← Push ID for children
                                                            let t = &mut transform_wrapper.0;

                                                            ui.label("Position:");
                                                            ui.horizontal(|ui| {
                                                                ui.add(egui::DragValue::new(&mut t.position.x).prefix("x: ").speed(0.1));
                                                                ui.add(egui::DragValue::new(&mut t.position.y).prefix("y: ").speed(0.1));
                                                                ui.add(egui::DragValue::new(&mut t.position.z).prefix("z: ").speed(0.1));
                                                            });

                                                            ui.label("Rotation:");
                                                            ui.horizontal(|ui| {
                                                                ui.add(egui::DragValue::new(&mut t.rotation.x).prefix("x: ").speed(0.01));
                                                                ui.add(egui::DragValue::new(&mut t.rotation.y).prefix("y: ").speed(0.01));
                                                                ui.add(egui::DragValue::new(&mut t.rotation.z).prefix("z: ").speed(0.01));
                                                                ui.add(egui::DragValue::new(&mut t.rotation.w).prefix("w: ").speed(0.01));
                                                            });

                                                            ui.label("Scale:");
                                                            ui.horizontal(|ui| {
                                                                ui.add(egui::DragValue::new(&mut t.scale.x).prefix("x: ").speed(0.1));
                                                                ui.add(egui::DragValue::new(&mut t.scale.y).prefix("y: ").speed(0.1));
                                                                ui.add(egui::DragValue::new(&mut t.scale.z).prefix("z: ").speed(0.1));
                                                            });
                                                        });
                                                    });
                                            }
                                        } else if type_id == TypeId::of::<BevyCamera>() {
                                            if let Some(mut camera_wrapper) = entity_mut.get_mut::<BevyCamera>() {
                                                egui::CollapsingHeader::new(&short_name)
                                                    .id_salt(unique_id)  // ← Make header ID unique
                                                    .show(ui, |ui| {
                                                        ui.push_id(unique_id, |ui| {  // ← Push ID for children
                                                            let c = &mut camera_wrapper.0;

                                                            ui.horizontal(|ui| {
                                                                ui.label("FOV:");
                                                                ui.add(egui::DragValue::new(&mut c.fov_y_rad).speed(0.1).suffix("°"));
                                                            });
                                                            ui.horizontal(|ui| {
                                                                ui.label("Aspect Ratio:");
                                                                ui.add(egui::DragValue::new(&mut c.aspect_ratio).speed(0.01));
                                                            });
                                                            ui.horizontal(|ui| {
                                                                ui.label("Near:");
                                                                ui.add(egui::DragValue::new(&mut c.near_plane).speed(0.01));
                                                            });
                                                            ui.horizontal(|ui| {
                                                                ui.label("Far:");
                                                                ui.add(egui::DragValue::new(&mut c.far_plane).speed(1.0));
                                                            });
                                                        });
                                                    });
                                            }
                                        } else if type_id == TypeId::of::<BevyLight>() {
                                            if let Some(mut light_wrapper) = entity_mut.get_mut::<BevyLight>() {
                                                egui::CollapsingHeader::new(&short_name)
                                                    .id_salt(unique_id)  // ← Make header ID unique
                                                    .show(ui, |ui| {
                                                        ui.push_id(unique_id, |ui| {  // ← Push ID for children
                                                            let l = &mut light_wrapper.0;

                                                            ui.horizontal(|ui| {
                                                                ui.label("Color:");
                                                                let mut color = [l.color.x, l.color.y, l.color.z];
                                                                if ui.color_edit_button_rgb(&mut color).changed() {
                                                                    l.color.x = color[0];
                                                                    l.color.y = color[1];
                                                                    l.color.z = color[2];
                                                                }
                                                            });
                                                            ui.horizontal(|ui| {
                                                                ui.label("Intensity:");
                                                                ui.add(egui::DragValue::new(&mut l.intensity).speed(0.1));
                                                            });
                                                        });
                                                    });
                                            }
                                        } else if type_id == TypeId::of::<BevyMeshRenderer>() {
                                            if let Some(mut mesh_wrapper) = entity_mut.get_mut::<BevyMeshRenderer>() {
                                                egui::CollapsingHeader::new(&short_name)
                                                    .id_salt(unique_id)  // make header ID unique
                                                    .show(ui, |ui| {
                                                        ui.push_id(unique_id, |ui| {  // push ID for children
                                                            let mesh_renderer = &mut mesh_wrapper.0;

                                                            ui.label("mesh id");
                                                            ui.add(egui::DragValue::new(&mut mesh_renderer.mesh_id));

                                                            ui.label("material id");
                                                            ui.add(egui::DragValue::new(&mut mesh_renderer.material_id));

                                                            ui.checkbox(&mut mesh_renderer.casts_shadow, "casts shadow");

                                                            ui.checkbox(&mut mesh_renderer.visible, "visible");
                                                        });
                                                    });
                                            }
                                        } else {
                                            // Unknown non-reflectable component: Optional fallback
                                            egui::CollapsingHeader::new(&short_name)
                                                .id_salt(unique_id)  // ← Make header ID unique
                                                .show(ui, |ui| {
                                                    ui.push_id(unique_id, |ui| {  // ← Push ID for children
                                                        ui.label("Non-reflectable component (no UI available)");
                                                    });
                                                });
                                        }
                                    }
                                }
                                });

                                // Auto-scroll to this entity if it's selected
                                if is_selected {
                                    ui.scroll_to_rect(
                                        response.header_response.rect,
                                        Some(egui::Align::Center),
                                    );
                                }

                                // allow clicking header to select/deselect entity
                                if response.header_response.clicked() {
                                    if let Some(mut sel) =
                                        world.get_resource_mut::<InspectorSelectedEntityResource>()
                                    {
                                        if sel.0 == Some(entity) {
                                            sel.0 = None;
                                        } else {
                                            sel.0 = Some(entity);
                                        }
                                    }
                                }
                            });
                        });
                    }
                });
            }),
            EguiWindowSpec {
                id: "inspector".to_string(),
                title: "inspector".to_string(),
            },
        ));
    }
}
