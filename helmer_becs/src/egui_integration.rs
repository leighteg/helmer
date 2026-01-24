use bevy_ecs::prelude::*;
use egui::{Context, Id, Pos2};
use helmer::{graphics::common::renderer::EguiRenderData, runtime::input_manager::InputManager};
use parking_lot::RwLock;
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use winit::keyboard::KeyCode;

use crate::{
    BevyInputManager, BevyRuntimeConfig,
    provided::ui::{inspector::InspectorUI, stats::StatsUI},
};

pub struct EguiWindowSpec {
    pub id: String,
    pub title: String,
}

#[derive(Resource, Default)]
pub struct EguiResource {
    pub ctx: Context,
    pub render_data: Option<EguiRenderData>,
    pub windows: Vec<(
        Box<dyn FnMut(&mut egui::Ui, &mut World, &Arc<RwLock<InputManager>>) + Send + Sync>,
        EguiWindowSpec,
    )>,
    pub close_actions: HashMap<String, Box<dyn FnMut(&mut World) + Send + Sync>>,
    pub accepting_input: bool,
    pub stats_ui: bool,
    pub inspector_ui: bool,
    pub render_graph_passes_state: RenderGraphPassesUiState,
    pub window_positions: HashMap<String, Pos2>,
    pub window_dragging: HashSet<String>,
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

    let (do_toggle_stats, do_toggle_inspector, mut raw_input, window_size, pixels_per_point) = {
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
        egui_res.window_dragging.clear();

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
        let mut close_actions = std::mem::take(&mut egui_res.close_actions);

        drop(egui_res);

        let screen_rect = ctx.available_rect();

        for (mut elements, spec) in windows {
            let window_id = egui::Id::new(spec.id.clone());
            if let Some(on_close) = close_actions.get_mut(&spec.id) {
                let mut open = true;
                let mut window = egui::Window::new(spec.title.clone())
                    .id(window_id)
                    .constrain_to(screen_rect)
                    .open(&mut open);
                if let Some(pos) = world
                    .get_resource::<EguiResource>()
                    .and_then(|res| res.window_positions.get(&spec.id).copied())
                {
                    window = window.current_pos(pos);
                }
                if let Some(inner) = window.show(ctx, |ui| {
                    elements(ui, world, &input_arc);
                }) {
                    if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
                        if !egui_res.window_dragging.contains(&spec.id) {
                            egui_res
                                .window_positions
                                .insert(spec.id.clone(), inner.response.rect.min);
                        }
                    }
                }
                if !open {
                    on_close(world);
                }
            } else {
                let mut window = egui::Window::new(spec.title.clone())
                    .id(window_id)
                    .constrain_to(screen_rect);
                if let Some(pos) = world
                    .get_resource::<EguiResource>()
                    .and_then(|res| res.window_positions.get(&spec.id).copied())
                {
                    window = window.current_pos(pos);
                }
                if let Some(inner) = window.show(ctx, |ui| {
                    elements(ui, world, &input_arc);
                }) {
                    if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
                        if !egui_res.window_dragging.contains(&spec.id) {
                            egui_res
                                .window_positions
                                .insert(spec.id.clone(), inner.response.rect.min);
                        }
                    }
                }
            }
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
