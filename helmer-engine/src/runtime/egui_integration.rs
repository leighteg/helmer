use crate::{
    ecs::{ecs_core::ECSCore, system::System},
    runtime::{
        egui_integration,
        input_manager::InputManager,
        runtime::{PerformanceMetrics, RenderMessage},
    },
};
use egui::{ClippedPrimitive, Context, RawInput, TexturesDelta};
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
}

pub struct EguiSystem {}

impl System for EguiSystem {
    fn name(&self) -> &str {
        "EguiSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, input: &InputManager) {
        let (fps, tps) = {
            let pm = ecs.get_resource::<PerformanceMetrics>().unwrap();
            (
                pm.fps.load(Ordering::Relaxed),
                pm.tps.load(Ordering::Relaxed),
            )
        };

        let egui_resource = if let Some(res) = ecs.get_resource_mut::<EguiResource>() {
            res
        } else {
            return;
        };

        let pixels_per_point = egui_resource.ctx.pixels_per_point();

        // Build input from InputManager's persistent state every tick
        let raw_input = input.build_egui_raw_input(input.window_size);

        let full_output = egui_resource.ctx.run(raw_input, |ctx| {
            egui::Window::new("Engine Stats").show(ctx, |ui| {
                ui.label(format!("FPS: {}", fps));
                ui.label(format!("TPS: {}", tps));
            });
        });

        let window_size = input.window_size;
        let pixels_per_point = egui_resource.ctx.pixels_per_point();
        let primitives = egui_resource
            .ctx
            .tessellate(full_output.shapes, pixels_per_point);

        egui_resource.render_data = Some(EguiRenderData {
            primitives,
            textures_delta: full_output.textures_delta,
            screen_descriptor: egui_wgpu::ScreenDescriptor {
                size_in_pixels: [window_size.x, window_size.y],
                pixels_per_point,
            },
        });
    }
}
