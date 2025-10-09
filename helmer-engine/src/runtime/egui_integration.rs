use crate::{
    ecs::{ecs_core::ECSCore, system::System},
    graphics::{config::RenderConfig, renderer::renderer::ShaderConstants},
    provided::{components::{Light, LightType, Transform}, ui::StatsUI},
    runtime::{
        config::RuntimeConfig,
        egui_integration,
        input_manager::InputManager,
        runtime::{PerformanceMetrics, RenderMessage},
    },
};
use egui::{ClippedPrimitive, Context, RawInput, TexturesDelta};
use winit::keyboard::KeyCode;
use std::sync::{Arc, atomic::Ordering, mpsc};

pub struct EguiRenderData {
    pub primitives: Vec<ClippedPrimitive>,
    pub textures_delta: TexturesDelta,
    pub screen_descriptor: egui_wgpu::ScreenDescriptor,
}

#[derive(Default)]
pub struct EguiResource {
    pub ctx: Context,
    pub render_data: Option<EguiRenderData>,
    pub windows: Vec<(fn(&mut egui::Ui, &mut ECSCore, &InputManager), String)>,
    pub stats_ui: bool,
}

pub struct EguiSystem {}

impl System for EguiSystem {
    fn name(&self) -> &str {
        "EguiSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, input: &InputManager) {
        ecs.resource_scope::<RuntimeConfig, _>(|ecs, runtime_cfg| {
            ecs.resource_scope::<EguiResource, _>(|ecs, egui_res| {
                if !runtime_cfg.egui {
                    runtime_cfg.render_config.egui_pass = false;
                    return;
                }
                else if !runtime_cfg.render_config.egui_pass {
                    runtime_cfg.render_config.egui_pass = true;
                }

                if input.is_key_active(KeyCode::ControlLeft) && input.was_just_pressed(KeyCode::KeyI) {
                    egui_res.stats_ui = !egui_res.stats_ui;
                }

                let raw_input = input.build_egui_raw_input(input.window_size);
                let ctx = &egui_res.ctx;

                let full_output = ctx.run(raw_input, |ctx| {
                    if egui_res.stats_ui {
                        StatsUI::run(ecs);
                    }

                    for (elements, name) in egui_res.windows.clone() {
                        egui::Window::new(name).show(ctx, |ui| {
                            elements(ui, ecs, input);
                        });
                    }
                    egui_res.windows.clear();
                });

                let window_size = input.window_size;
                let pixels_per_point = input.scale_factor as f32;
                let primitives = ctx.tessellate(full_output.shapes, pixels_per_point);

                egui_res.render_data = Some(EguiRenderData {
                    primitives,
                    textures_delta: full_output.textures_delta,
                    screen_descriptor: egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [window_size.x, window_size.y],
                        pixels_per_point,
                    },
                });
            });
        });
    }
}
