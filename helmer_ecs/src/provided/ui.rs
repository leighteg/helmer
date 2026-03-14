use std::sync::{Arc, atomic::Ordering};

use crate::components::{ActiveCamera, Camera, Light, LightType, Transform};
use egui::ComboBox;
use helmer_render::graphics::{
    common::{config::RenderConfig, renderer::ShaderConstants},
    render_graphs::{graph_templates, template_for_graph},
};
use helmer_render::runtime::RuntimeConfig;
use helmer_window::runtime::runtime::PerformanceMetrics;

use crate::{
    ecs::ecs_core::ECSCore, egui_integration::EguiResource,
    physics::physics_resource::PhysicsResource, systems::renderer_system::RenderGraphResource,
};

pub struct StatsUI {}

fn toggle_debug_flag(ui: &mut egui::Ui, flags: &mut u32, mask: u32, label: &str) {
    let mut enabled = (*flags & mask) != 0;
    if ui.checkbox(&mut enabled, label).changed() {
        if enabled {
            *flags |= mask;
        } else {
            *flags &= !mask;
        }
    }
}

impl StatsUI {
    pub fn run(ecs: &mut ECSCore) {
        ecs.resource_scope::<EguiResource, _>(|ecs, egui_res| {
            egui_res.windows.push((
                Box::new(move |ui, ecs, input| {
                    let (fps, tps) = {
                        let pm = ecs.get_resource::<Arc<PerformanceMetrics>>().unwrap();
                        (
                            pm.fps.load(Ordering::Relaxed),
                            pm.tps.load(Ordering::Relaxed),
                        )
                    };

                    ui.label(format!("FPS: {}", fps));
                    ui.label(format!("TPS: {}", tps));
                }),
                "helmer metrics".to_string(),
            ));

            /*egui_res.windows.push((
                Box::new(move |ui, ecs, input| {
                    ecs.resource_scope::<EguiResource, _>(|ecs, egui_res: &mut EguiResource| {
                        ui.checkbox(&mut egui_res.stats_ui, "stats ui");
                    });
                }),
                "runtime config".to_string(),
            ));*/

            egui_res.windows.push((
                Box::new(move |ui, ecs, input| {
                    ecs.resource_scope::<RuntimeConfig, _>(
                        |ecs, runtime_cfg: &mut RuntimeConfig| {
                            let render_cfg = &mut runtime_cfg.render_config;

                            if ui.button("default").clicked() {
                                let mut new_cfg = RenderConfig::default();
                                new_cfg.egui_pass = true;
                                *render_cfg = new_cfg;
                            }
                            ui.separator();

                            let shade_mode_labels = ["lit", "unlit", "lighting"];
                            ComboBox::from_label("shade mode")
                                .selected_text(
                                    *shade_mode_labels
                                        .get(render_cfg.shader_constants.shade_mode as usize)
                                        .unwrap_or(&"???"),
                                )
                                .show_ui(ui, |ui| {
                                    for (i, label) in shade_mode_labels.iter().enumerate() {
                                        ui.selectable_value(
                                            &mut render_cfg.shader_constants.shade_mode,
                                            i as u32,
                                            *label,
                                        );
                                    }
                                });

                            if render_cfg.shader_constants.shade_mode != 1 {
                                let light_model_labels = ["PBR lit", "stylized lit"];
                                ComboBox::from_label("lighting model")
                                    .selected_text(
                                        *light_model_labels
                                            .get(render_cfg.shader_constants.light_model as usize)
                                            .unwrap_or(&"???"),
                                    )
                                    .show_ui(ui, |ui| {
                                        for (i, label) in light_model_labels.iter().enumerate() {
                                            ui.selectable_value(
                                                &mut render_cfg.shader_constants.light_model,
                                                i as u32,
                                                *label,
                                            );
                                        }
                                    });

                                let sky_light_contribution_model_labels =
                                    ["none", "full", "stylized full", "simple"];
                                ComboBox::from_label("sky light contribution")
                                    .selected_text(
                                        *sky_light_contribution_model_labels
                                            .get(
                                                render_cfg.shader_constants.skylight_contribution
                                                    as usize,
                                            )
                                            .unwrap_or(&"???"),
                                    )
                                    .show_ui(ui, |ui| {
                                        for (i, label) in
                                            sky_light_contribution_model_labels.iter().enumerate()
                                        {
                                            ui.selectable_value(
                                                &mut render_cfg
                                                    .shader_constants
                                                    .skylight_contribution,
                                                i as u32,
                                                *label,
                                            );
                                        }
                                    });
                            }

                            ui.separator();
                            ui.heading("Culling & LOD");
                            ui.checkbox(&mut render_cfg.frustum_culling, "Frustum Culling");
                            ui.checkbox(
                                &mut render_cfg.occlusion_culling,
                                "Occlusion Culling (Hi-Z)",
                            );
                            ui.checkbox(&mut render_cfg.lod, "LOD");

                            ui.separator();
                            ui.heading("Lighting Limits");
                            ui.add(
                                egui::Slider::new(&mut render_cfg.max_lights_forward, 0..=1024)
                                    .text("Max Lights (Forward)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut render_cfg.max_lights_deferred, 0..=4096)
                                    .text("Max Lights (Deferred)"),
                            );

                            ui.separator();
                            ui.heading("Shader Constants");
                            shader_constants_ui(ui, &mut render_cfg.shader_constants);
                        },
                    );
                }),
                "render config".to_string(),
            ));

            egui_res.windows.push((
                Box::new(move |ui, ecs, _input| {
                    let templates = graph_templates();
                    let fallback_template =
                        templates.first().expect("render graph templates missing");
                    let mut active_name = ecs
                        .get_resource::<RenderGraphResource>()
                        .map(|res| res.0.name)
                        .unwrap_or(fallback_template.name);

                    ui.heading("render graph");
                    ui.separator();
                    ui.heading("graph selection");

                    if let Some(mut graph_res) = ecs.get_resource_mut::<RenderGraphResource>() {
                        let mut selected_name = active_name;
                        ComboBox::from_label("graph template")
                            .selected_text(
                                template_for_graph(selected_name)
                                    .unwrap_or(fallback_template)
                                    .label,
                            )
                            .show_ui(ui, |ui| {
                                for template in templates {
                                    ui.selectable_value(
                                        &mut selected_name,
                                        template.name,
                                        template.label,
                                    );
                                }
                            });

                        if selected_name != active_name {
                            if let Some(template) = template_for_graph(selected_name) {
                                graph_res.0 = (template.build)();
                                active_name = selected_name;
                            }
                        }

                        ui.label(format!("active graph: {}", graph_res.0.name));
                    } else {
                        ui.label("render graph resource unavailable");
                    }

                    let active_template =
                        template_for_graph(active_name).unwrap_or(fallback_template);

                    ui.separator();
                    ui.heading("passes");
                    ecs.resource_scope::<RuntimeConfig, _>(
                        |ecs, runtime_cfg: &mut RuntimeConfig| {
                            let render_cfg = &mut runtime_cfg.render_config;
                            for pass in active_template.pass_toggles {
                                let mut enabled = pass.toggle.get(render_cfg);
                                if ui.checkbox(&mut enabled, pass.label).changed() {
                                    pass.toggle.set(render_cfg, enabled);
                                }
                            }

                            if !active_template.debug_flags.is_empty() {
                                ui.separator();
                                ui.heading("debug composite");
                                let flags = &mut render_cfg.debug_flags;
                                for flag in active_template.debug_flags {
                                    toggle_debug_flag(ui, flags, flag.mask, flag.label);
                                }

                                let all_mask = active_template
                                    .debug_flags
                                    .iter()
                                    .fold(0u32, |acc, flag| acc | flag.mask);
                                ui.horizontal(|ui| {
                                    if ui.button("all").clicked() {
                                        *flags = all_mask;
                                    }
                                    if ui.button("none").clicked() {
                                        *flags = 0;
                                    }
                                });
                            }
                        },
                    );
                }),
                "render graph".to_string(),
            ));

            egui_res.windows.push((
                Box::new(move |ui, ecs, input| {
                    ecs.component_pool
                        .query_exact_mut_for_each::<(Transform, Camera, ActiveCamera), _>(
                            |(transform, camera, _)| {
                                ui.heading("active camera");

                                ui.label("position");
                                ui.add(
                                    egui::DragValue::new(&mut transform.position.x)
                                        .speed(0.1)
                                        .prefix("x: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut transform.position.y)
                                        .speed(0.1)
                                        .prefix("y: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut transform.position.z)
                                        .speed(0.1)
                                        .prefix("z: "),
                                );

                                ui.label("rotation");
                                ui.drag_angle(&mut transform.rotation.x);
                                ui.drag_angle(&mut transform.rotation.y);
                                ui.drag_angle(&mut transform.rotation.z);

                                ui.add(
                                    egui::DragValue::new(&mut camera.fov_y_rad)
                                        .speed(0.1)
                                        .prefix("fov: "),
                                );

                                ui.separator();
                            },
                        );

                    ecs.component_pool
                        .query_exact_mut_for_each::<(Transform, Light), _>(|(transform, light)| {
                            if light.light_type != LightType::Directional {
                                return;
                            }

                            ui.heading("directional light");

                            ui.label("rotation");
                            ui.drag_angle(&mut transform.rotation.x);
                            ui.drag_angle(&mut transform.rotation.y);
                            ui.drag_angle(&mut transform.rotation.z);

                            ui.add(
                                egui::DragValue::new(&mut light.intensity)
                                    .speed(0.1)
                                    .prefix("intensity: "),
                            );

                            ui.separator();
                        });

                    ecs.resource_scope::<PhysicsResource, _>(|ecs, physics_resource| {
                        ui.heading("physics");

                        ui.label("gravity");
                        ui.add(
                            egui::DragValue::new(&mut physics_resource.gravity.x)
                                .speed(0.1)
                                .prefix("x: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut physics_resource.gravity.y)
                                .speed(0.1)
                                .prefix("y: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut physics_resource.gravity.z)
                                .speed(0.1)
                                .prefix("z: "),
                        );

                        ui.separator();
                    });
                }),
                "scene".to_string(),
            ));
        });
    }
}

fn shader_constants_ui(ui: &mut egui::Ui, c: &mut ShaderConstants) {
    ui.collapsing("sky", |ui| {
        ui.add(egui::Slider::new(&mut c.planet_radius, 0.0..=9999e3).text("planet radius"));
        ui.add(egui::Slider::new(&mut c.atmosphere_radius, 0.0..=9999e3).text("atmosphere radius"));
        ui.add(egui::Slider::new(&mut c.sky_light_samples, 0..=64).text("light samples"));
    });

    ui.collapsing("SSR", |ui| {
        ui.add(
            egui::Slider::new(&mut c.ssr_coarse_steps, 1..=2048)
                .smart_aim(false)
                .step_by(1.0)
                .text("Coarse Steps"),
        );
        ui.add(
            egui::Slider::new(&mut c.ssr_binary_search_steps, 1..=2048)
                .smart_aim(false)
                .step_by(1.0)
                .text("Binary Search Steps"),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssr_linear_step_size)
                .speed(0.001)
                .range(0.0..=1000.0)
                .prefix("Step Size: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssr_thickness)
                .speed(0.001)
                .range(0.0..=1000.0)
                .prefix("Thickness: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssr_max_distance)
                .speed(0.1)
                .range(0.0..=10000.0)
                .prefix("Max Distance: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssr_roughness_fade_start)
                .speed(0.01)
                .prefix("Roughness Fade Start: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssr_roughness_fade_end)
                .speed(0.01)
                .prefix("Roughness Fade End: "),
        );
    });

    ui.collapsing("SSGI", |ui| {
        ui.add(
            egui::Slider::new(&mut c.ssgi_num_rays, 1..=512)
                .smart_aim(false)
                .step_by(1.0)
                .text("Num Rays"),
        );
        ui.add(
            egui::Slider::new(&mut c.ssgi_num_steps, 1..=512)
                .smart_aim(false)
                .step_by(1.0)
                .text("Num Steps"),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssgi_ray_step_size)
                .speed(0.001)
                .prefix("Ray Step Size: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssgi_thickness)
                .speed(0.001)
                .prefix("Thickness: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssgi_blend_factor)
                .speed(0.01)
                .range(0.0..=1000.0)
                .prefix("Blend Factor: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.ssgi_intensity)
                .speed(0.5)
                .range(0.0..=1000.0)
                .prefix("Intensity: "),
        );
    });

    ui.collapsing("shadows", |ui| {
        ui.add(
            egui::DragValue::new(&mut c.evsm_c)
                .speed(0.05)
                .range(0.0..=1000.0)
                .prefix("EVSM C: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.pcf_radius)
                .speed(0.1)
                .range(0..=100)
                .prefix("PCF radius: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.pcf_min_scale)
                .speed(0.1)
                .range(0..=100)
                .prefix("PCF min scale: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.pcf_max_scale)
                .speed(0.1)
                .range(0..=100)
                .prefix("PCF max scale: "),
        );
        ui.add(
            egui::DragValue::new(&mut c.pcf_max_distance)
                .speed(0.1)
                .range(0..=10000)
                .prefix("PCF max distance: "),
        );
    });
}
