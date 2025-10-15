use std::sync::{Arc, atomic::Ordering};

use helmer::{
    graphics::{config::RenderConfig, renderer::renderer::ShaderConstants},
    provided::components::{Light, LightType, Transform},
    runtime::{config::RuntimeConfig, runtime::PerformanceMetrics},
};

use crate::{
    ecs::{ecs_core::ECSCore, system::System},
    egui_integration::EguiResource,
};

pub struct StatsUI {}

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

            egui_res.windows.push((
                Box::new(move |ui, ecs, input| {
                    ecs.resource_scope::<EguiResource, _>(|ecs, egui_res: &mut EguiResource| {
                        ui.checkbox(&mut egui_res.stats_ui, "stats ui");
                    });
                }),
                "runtime config".to_string(),
            ));

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

                            ui.heading("Render Passes");
                            ui.checkbox(&mut render_cfg.shadow_pass, "Shadow Pass");
                            ui.checkbox(&mut render_cfg.direct_lighting_pass, "Direct Lighting");
                            ui.checkbox(&mut render_cfg.sky_pass, "Sky Pass");
                            ui.checkbox(&mut render_cfg.ssgi_pass, "SSGI Pass");
                            ui.checkbox(&mut render_cfg.ssgi_denoise_pass, "SSGI Denoise Pass");
                            ui.checkbox(&mut render_cfg.ssr_pass, "SSR Pass");

                            ui.separator();
                            ui.heading("Culling & LOD");
                            ui.checkbox(&mut render_cfg.frustum_culling, "Frustum Culling");
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
                Box::new(move |ui, ecs, input| {
                    ecs.component_pool
                        .query_exact_mut_for_each::<(Transform, Light), _>(|(transform, light)| {
                            if light.light_type != LightType::Directional {
                                return;
                            }

                            ui.label("directional light");
                            ui.drag_angle(&mut transform.rotation.x);
                            ui.drag_angle(&mut transform.rotation.y);
                            ui.drag_angle(&mut transform.rotation.z);

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

    ui.collapsing("EVSM", |ui| {
        ui.add(
            egui::DragValue::new(&mut c.evsm_c)
                .speed(0.05)
                .range(0.0..=1000.0)
                .prefix("EVSM C: "),
        );
    });
}
