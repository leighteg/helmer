use bevy_ecs::prelude::*;
use egui::Context;
use helmer::{graphics::common::renderer::EguiRenderData, runtime::input_manager::InputManager};
use parking_lot::RwLock;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use winit::keyboard::KeyCode;

use crate::{
    BevyInputManager, BevyRuntimeConfig,
    provided::ui::{inspector::InspectorUI, stats::StatsUI},
};

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
    pub inspector_ui: bool,
    pub render_graph_passes_state: RenderGraphPassesUiState,
}

pub struct RenderGraphPassesUiState {
    pub filter: String,
    pub sort_by_time: bool,
    pub show_disabled: bool,
}

impl Default for RenderGraphPassesUiState {
    fn default() -> Self {
        Self {
            filter: String::new(),
            sort_by_time: false,
            show_disabled: true,
        }
    }
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

    let (do_toggle_stats, do_toggle_inspector, raw_input, window_size, pixels_per_point) = {
        let input = input_arc.read();

        let is_control_active = input.is_key_active(KeyCode::ControlLeft);
        (
            is_control_active && input.was_just_pressed(KeyCode::KeyG),
            is_control_active && input.was_just_pressed(KeyCode::KeyI),
            input.build_egui_raw_input(input.window_size),
            input.window_size,
            input.scale_factor as f32,
        )
    };

    let ctx = {
        let mut egui_res = world
            .get_resource_mut::<EguiResource>()
            .expect("EguiResource resource not found");

        if do_toggle_stats {
            egui_res.stats_ui = !egui_res.stats_ui;
        }
        if do_toggle_inspector {
            egui_res.inspector_ui = !egui_res.inspector_ui;
        }

        egui_res.accepting_input = true;

        if egui_res.stats_ui {
            StatsUI::add_windows(&mut egui_res);
        }
        if egui_res.inspector_ui {
            InspectorUI::add_window(&mut egui_res);
        }

        egui_res.ctx.clone()
    };

    let full_output = ctx.run(raw_input, |ctx| {
        let mut egui_res = world
            .get_resource_mut::<EguiResource>()
            .expect("EguiResource resource not found");

        let windows = std::mem::take(&mut egui_res.windows);

        drop(egui_res);

        let screen_rect = ctx.available_rect();

        for (mut elements, name) in windows {
            egui::Window::new(name.clone())
                .constrain_to(screen_rect)
                .show(ctx, |ui| {
                    elements(ui, world, &input_arc);
                });
        }
    });

    let primitives = ctx.tessellate(full_output.shapes, pixels_per_point);
    let textures_delta = full_output.textures_delta;

    let mut egui_res = world
        .get_resource_mut::<EguiResource>()
        .expect("EguiResource resource not found");

    static EGUI_TEXTURE_VERSION: AtomicU64 = AtomicU64::new(1);
    let textures_changed = !textures_delta.set.is_empty() || !textures_delta.free.is_empty();
    let version = if textures_changed {
        EGUI_TEXTURE_VERSION
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1)
    } else {
        EGUI_TEXTURE_VERSION.load(Ordering::Relaxed)
    };

    egui_res.render_data = Some(EguiRenderData {
        version,
        primitives,
        textures_delta,
        screen_descriptor: egui_wgpu::ScreenDescriptor {
            size_in_pixels: [window_size.x, window_size.y],
            pixels_per_point,
        },
    });
}
