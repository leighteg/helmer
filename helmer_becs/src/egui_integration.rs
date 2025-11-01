use bevy_ecs::prelude::*;
use egui::Context;
use helmer::{
    graphics::renderer_common::common::EguiRenderData, runtime::input_manager::InputManager,
};
use parking_lot::RwLock;
use std::sync::Arc;
use winit::keyboard::KeyCode;

use crate::{BevyInputManager, BevyRuntimeConfig, provided::ui::StatsUI};

#[derive(Resource, Default)]
pub struct EguiResource {
    pub ctx: Context,
    pub render_data: Option<EguiRenderData>,
    pub windows: Vec<(
        Box<dyn FnMut(&mut egui::Ui, &mut World, &Arc<RwLock<InputManager>>) + Send + Sync>,
        String,
    )>,
    pub accepting_input: bool,
    pub stats_ui: bool,
}

pub fn egui_system(world: &mut World) {
    let input_arc = world
        .get_resource::<BevyInputManager>()
        .expect("InputManager resource not found")
        .0
        .clone();

    let mut runtime_cfg = world
        .get_resource_mut::<BevyRuntimeConfig>()
        .expect("RuntimeConfig resource not found");

    if !runtime_cfg.0.egui {
        runtime_cfg.0.render_config.egui_pass = false;
        return;
    }

    runtime_cfg.0.render_config.egui_pass = true;

    let (do_toggle, raw_input, window_size, pixels_per_point) = {
        let input = input_arc.read();
        (
            input.is_key_active(KeyCode::ControlLeft) && input.was_just_pressed(KeyCode::KeyI),
            input.build_egui_raw_input(input.window_size),
            input.window_size,
            input.scale_factor as f32,
        )
    };

    let ctx = {
        let mut egui_res = world
            .get_resource_mut::<EguiResource>()
            .expect("EguiResource resource not found");

        if do_toggle {
            egui_res.stats_ui = !egui_res.stats_ui;
        }

        egui_res.accepting_input = true;

        if egui_res.stats_ui {
            StatsUI::add_windows(&mut egui_res);
        }

        egui_res.ctx.clone()
    };

    let full_output = ctx.run(raw_input, |ctx| {
        let mut egui_res = world
            .get_resource_mut::<EguiResource>()
            .expect("EguiResource resource not found");

        let windows = std::mem::take(&mut egui_res.windows);

        drop(egui_res);

        for (mut elements, name) in windows {
            egui::Window::new(name).show(ctx, |ui| {
                elements(ui, world, &input_arc);
            });
        }
    });

    let primitives = ctx.tessellate(full_output.shapes, pixels_per_point);

    let mut egui_res = world
        .get_resource_mut::<EguiResource>()
        .expect("EguiResource resource not found");

    egui_res.render_data = Some(EguiRenderData {
        primitives,
        textures_delta: full_output.textures_delta,
        screen_descriptor: egui_wgpu::ScreenDescriptor {
            size_in_pixels: [window_size.x, window_size.y],
            pixels_per_point,
        },
    });
}
