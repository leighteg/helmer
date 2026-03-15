use std::{
    cmp::Ordering as CmpOrdering,
    collections::VecDeque,
    sync::{Arc, atomic::Ordering},
};

use crate::components::LightType;
use bevy_ecs::prelude::{Entity, World};
use egui::{Align2, Color32, ComboBox, Stroke, StrokeKind, TextStyle, pos2, vec2};
use hashbrown::HashSet;
use helmer_render::graphics::{
    backend::binding_backend::BindingBackendChoice,
    common::{
        config::{RenderConfig, TransparentSortMode},
        constants::MAX_SHADOW_CASCADES,
        renderer::{
            OCCLUSION_STATUS_DISABLED, OCCLUSION_STATUS_NO_GBUFFER, OCCLUSION_STATUS_NO_HIZ,
            OCCLUSION_STATUS_NO_INSTANCES, OCCLUSION_STATUS_RAN, RenderControl, RenderMessage,
            ShaderConstants, WgpuBackend,
        },
    },
    graph::definition::resource_id::ResourceKind,
    render_graphs::{graph_templates, template_for_graph},
};
use helmer_window::runtime::input_manager::InputManager;
use parking_lot::RwLock;
use tracing::warn;

use crate::{
    ActiveCamera, AudioBackendResource, BecsAssetServer, BecsLodTuning, BecsPerformanceMetrics,
    BecsRenderSender, BecsRenderWorkerTuning, BecsRendererStats, BecsRuntimeConfig,
    BecsRuntimeProfiling, BecsRuntimeTuning, BecsSceneTuning, BecsStreamingTuning, Camera,
    DebugGraphHistory, Light, ProfilingHistory, Transform,
    egui_integration::{EguiResource, EguiWindowSpec},
    physics::physics_resource::PhysicsResource,
    systems::render_system::{RenderGraphResource, RenderObjectCount},
    ui_integration::UiPerfStats,
};

fn draw_history_plot(
    ui: &mut egui::Ui,
    height: f32,
    series: &[(&str, &std::collections::VecDeque<f64>, Color32)],
) {
    if series.is_empty() {
        return;
    }

    let desired_size = vec2(ui.available_width().max(120.0), height);
    let (rect, _response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    let visuals = ui.visuals().widgets.noninteractive;
    let frame_rect = rect.shrink(2.0);
    let painter = ui.painter_at(rect);
    painter.rect(
        rect,
        6.0,
        visuals.bg_fill,
        Stroke::new(1.0, visuals.bg_stroke.color),
        StrokeKind::Outside,
    );
    painter.rect_stroke(
        frame_rect,
        4.0,
        Stroke::new(1.0, visuals.bg_stroke.color.gamma_multiply(0.6)),
        StrokeKind::Outside,
    );

    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut max_len = 0usize;
    for (_, data, _) in series.iter() {
        if data.is_empty() {
            continue;
        }
        let (local_min, local_max) = data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(*v), mx.max(*v))
            });
        min_val = min_val.min(local_min);
        max_val = max_val.max(local_max);
        max_len = max_len.max(data.len());
    }

    if !min_val.is_finite() || !max_val.is_finite() || max_len < 2 {
        return;
    }

    let mut span = max_val - min_val;
    if span.abs() < f64::EPSILON {
        span = 1.0;
    }
    let denom = ((max_len - 1) as f32).max(1.0);
    let plot_rect = frame_rect.shrink(6.0);

    for (_, data, color) in series.iter() {
        if data.len() < 2 {
            continue;
        }

        let mut last: Option<egui::Pos2> = None;
        for (idx, v) in data.iter().enumerate() {
            let x_t = idx as f32 / denom;
            let y_t = ((*v - min_val) / span) as f32;
            let pos = pos2(
                egui::lerp(plot_rect.x_range(), x_t),
                egui::lerp(plot_rect.y_range(), 1.0 - y_t),
            );

            if let Some(prev) = last {
                painter.line_segment([prev, pos], Stroke::new(1.5, *color));
            }
            last = Some(pos);
        }
    }

    let mut legend_y = plot_rect.top() + 4.0;
    for (label, _, color) in series.iter() {
        painter.text(
            pos2(plot_rect.left() + 6.0, legend_y),
            Align2::LEFT_TOP,
            *label,
            TextStyle::Small.resolve(ui.style()),
            *color,
        );
        legend_y += 14.0;
    }
}

fn push_history(deque: &mut VecDeque<f64>, value: f64, max_samples: usize) {
    if max_samples == 0 {
        deque.clear();
        return;
    }
    deque.push_back(value);
    while deque.len() > max_samples {
        deque.pop_front();
    }
}

fn clear_profiling_history(history: &mut ProfilingHistory) {
    history.main_event_ms.clear();
    history.main_update_ms.clear();
    history.logic_frame_ms.clear();
    history.logic_asset_ms.clear();
    history.logic_input_ms.clear();
    history.logic_tick_ms.clear();
    history.logic_schedule_ms.clear();
    history.logic_render_send_ms.clear();
    history.ecs_render_data_ms.clear();
    history.ecs_scene_spawn_ms.clear();
    history.ecs_scene_update_ms.clear();
    history.render_thread_frame_ms.clear();
    history.render_thread_messages_ms.clear();
    history.render_thread_upload_ms.clear();
    history.render_thread_render_ms.clear();
    history.render_prepare_globals_ms.clear();
    history.render_streaming_plan_ms.clear();
    history.render_occlusion_ms.clear();
    history.render_graph_ms.clear();
    history.render_graph_pass_ms.clear();
    history.render_graph_encoder_create_ms.clear();
    history.render_graph_encoder_finish_ms.clear();
    history.render_graph_overhead_ms.clear();
    history.render_resource_mgmt_ms.clear();
    history.render_acquire_ms.clear();
    history.render_submit_ms.clear();
    history.render_present_ms.clear();
    history.ui_system_ms.clear();
    history.ui_run_frame_ms.clear();
    history.ui_interaction_ms.clear();
    history.ui_scroll_metrics_ms.clear();
    history.ui_render_data_convert_ms.clear();
    history.render_ui_build_ms.clear();
    history.render_pass_ms.clear();
    history.render_pass_last_ms.clear();
    history.render_pass_order.clear();
    history.audio_mix_ms.clear();
    history.audio_callback_ms.clear();
    history.audio_emitters.clear();
    history.audio_streaming_emitters.clear();
}

fn micros_to_ms(value: u64) -> f64 {
    value as f64 / 1000.0
}

fn micros_f64_to_ms(value: f64) -> f64 {
    value / 1000.0
}

fn pass_color(name: &str) -> Color32 {
    let mut hash: u32 = 2166136261;
    for byte in name.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    let r = ((hash >> 16) & 0xFF) as u8;
    let g = ((hash >> 8) & 0xFF) as u8;
    let b = (hash & 0xFF) as u8;
    Color32::from_rgb(r / 2 + 64, g / 2 + 64, b / 2 + 64)
}

fn drag_usize(ui: &mut egui::Ui, value: &mut usize, label: &str, speed: f32) -> bool {
    let mut raw = *value as u64;
    let changed = ui
        .add(egui::DragValue::new(&mut raw).speed(speed).prefix(label))
        .changed();
    if changed {
        *value = raw as usize;
    }
    changed
}

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

fn window_spec(title: &str) -> EguiWindowSpec {
    EguiWindowSpec {
        id: title.to_string(),
        title: title.to_string(),
    }
}

pub fn draw_runtime_profiling_panel(ui: &mut egui::Ui, world: &mut World) {
    let runtime_profiling = world
        .get_resource::<BecsRuntimeProfiling>()
        .map(|profiling| profiling.0.clone());
    let renderer_stats = world
        .get_resource::<BecsRendererStats>()
        .map(|stats| stats.0.clone());
    let ui_perf_stats = world.get_resource::<UiPerfStats>().cloned();
    let audio_stats = world
        .get_resource::<AudioBackendResource>()
        .map(|audio| audio.0.stats());

    egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.heading("profiling");
                        ui.separator();

                        let mut profiling_enabled = runtime_profiling
                            .as_ref()
                            .map(|profiling| profiling.enabled.load(Ordering::Relaxed))
                            .unwrap_or(false);
                        let mut history_samples = runtime_profiling
                            .as_ref()
                            .map(|profiling| profiling.history_samples.load(Ordering::Relaxed))
                            .unwrap_or(0);
                        let mut render_pass_plot_limit = runtime_profiling
                            .as_ref()
                            .map(|profiling| {
                                profiling.render_pass_plot_limit.load(Ordering::Relaxed)
                            })
                            .unwrap_or(0);

                        let enabled_changed =
                            ui.checkbox(&mut profiling_enabled, "enabled").changed();
                        let history_changed = ui
                            .add(
                                egui::DragValue::new(&mut history_samples)
                                    .speed(1.0)
                                    .prefix("history samples: "),
                            )
                            .changed();
                        let plot_limit_changed = ui
                            .add(
                                egui::DragValue::new(&mut render_pass_plot_limit)
                                    .speed(1.0)
                                    .prefix("render pass plot limit: "),
                            )
                            .changed();
                        let clear_clicked = ui.button("clear history").clicked();

                        if let Some(profiling) = runtime_profiling.as_ref() {
                            if enabled_changed {
                                profiling
                                    .enabled
                                    .store(profiling_enabled, Ordering::Relaxed);
                            }
                            if history_changed {
                                profiling
                                    .history_samples
                                    .store(history_samples, Ordering::Relaxed);
                            }
                            if plot_limit_changed {
                                profiling
                                    .render_pass_plot_limit
                                    .store(render_pass_plot_limit, Ordering::Relaxed);
                            }
                        }

                        if let Some(stats) = renderer_stats.as_ref() {
                            stats
                                .profiling_enabled
                                .store(profiling_enabled, Ordering::Relaxed);
                        }

                        let history_limit = history_samples as usize;

                        let mut history = world
                            .get_resource_mut::<ProfilingHistory>()
                            .expect("ProfilingHistory resource not found");

                        if clear_clicked {
                            clear_profiling_history(&mut history);
                        }

                        if profiling_enabled && history_limit > 0 {
                            if let Some(profiling) = runtime_profiling.as_ref() {
                                let main_event_ms =
                                    micros_to_ms(profiling.main_event_us.load(Ordering::Relaxed));
                                let main_update_ms =
                                    micros_to_ms(profiling.main_update_us.load(Ordering::Relaxed));
                                let logic_frame_ms =
                                    micros_to_ms(profiling.logic_frame_us.load(Ordering::Relaxed));
                                let logic_asset_ms =
                                    micros_to_ms(profiling.logic_asset_us.load(Ordering::Relaxed));
                                let logic_input_ms =
                                    micros_to_ms(profiling.logic_input_us.load(Ordering::Relaxed));
                                let logic_tick_ms =
                                    micros_to_ms(profiling.logic_tick_us.load(Ordering::Relaxed));
                                let logic_schedule_ms = micros_to_ms(
                                    profiling.logic_schedule_us.load(Ordering::Relaxed),
                                );
                                let logic_render_send_ms = micros_to_ms(
                                    profiling.logic_render_send_us.load(Ordering::Relaxed),
                                );
                                let ecs_render_data_ms = micros_to_ms(
                                    profiling.ecs_render_data_us.load(Ordering::Relaxed),
                                );
                                let ecs_scene_spawn_ms = micros_to_ms(
                                    profiling.ecs_scene_spawn_us.load(Ordering::Relaxed),
                                );
                                let ecs_scene_update_ms = micros_to_ms(
                                    profiling.ecs_scene_update_us.load(Ordering::Relaxed),
                                );
                                let render_thread_frame_ms = micros_to_ms(
                                    profiling.render_thread_frame_us.load(Ordering::Relaxed),
                                );
                                let render_thread_messages_ms = micros_to_ms(
                                    profiling.render_thread_messages_us.load(Ordering::Relaxed),
                                );
                                let render_thread_upload_ms = micros_to_ms(
                                    profiling.render_thread_upload_us.load(Ordering::Relaxed),
                                );
                                let render_thread_render_ms = micros_to_ms(
                                    profiling.render_thread_render_us.load(Ordering::Relaxed),
                                );

                                push_history(
                                    &mut history.main_event_ms,
                                    main_event_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.main_update_ms,
                                    main_update_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.logic_frame_ms,
                                    logic_frame_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.logic_asset_ms,
                                    logic_asset_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.logic_input_ms,
                                    logic_input_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.logic_tick_ms,
                                    logic_tick_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.logic_schedule_ms,
                                    logic_schedule_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.logic_render_send_ms,
                                    logic_render_send_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.ecs_render_data_ms,
                                    ecs_render_data_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.ecs_scene_spawn_ms,
                                    ecs_scene_spawn_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.ecs_scene_update_ms,
                                    ecs_scene_update_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_thread_frame_ms,
                                    render_thread_frame_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_thread_messages_ms,
                                    render_thread_messages_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_thread_upload_ms,
                                    render_thread_upload_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_thread_render_ms,
                                    render_thread_render_ms,
                                    history_limit,
                                );
                            }

                            if let Some(stats) = renderer_stats.as_ref() {
                                let render_prepare_globals_ms = micros_to_ms(
                                    stats.render_prepare_globals_us.load(Ordering::Relaxed),
                                );
                                let render_streaming_plan_ms = micros_to_ms(
                                    stats.render_streaming_plan_us.load(Ordering::Relaxed),
                                );
                                let render_occlusion_ms =
                                    micros_to_ms(stats.render_occlusion_us.load(Ordering::Relaxed));
                                let render_graph_ms =
                                    micros_to_ms(stats.render_graph_us.load(Ordering::Relaxed));
                                let render_graph_pass_ms = micros_to_ms(
                                    stats.render_graph_pass_us.load(Ordering::Relaxed),
                                );
                                let render_graph_encoder_create_ms = micros_to_ms(
                                    stats.render_graph_encoder_create_us.load(Ordering::Relaxed),
                                );
                                let render_graph_encoder_finish_ms = micros_to_ms(
                                    stats.render_graph_encoder_finish_us.load(Ordering::Relaxed),
                                );
                                let render_graph_overhead_ms = micros_to_ms(
                                    stats.render_graph_overhead_us.load(Ordering::Relaxed),
                                );
                                let render_resource_mgmt_ms = micros_to_ms(
                                    stats.render_resource_mgmt_us.load(Ordering::Relaxed),
                                );
                                let render_acquire_ms =
                                    micros_to_ms(stats.render_acquire_us.load(Ordering::Relaxed));
                                let render_submit_ms =
                                    micros_to_ms(stats.render_submit_us.load(Ordering::Relaxed));
                                let render_present_ms =
                                    micros_to_ms(stats.render_present_us.load(Ordering::Relaxed));

                                push_history(
                                    &mut history.render_prepare_globals_ms,
                                    render_prepare_globals_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_streaming_plan_ms,
                                    render_streaming_plan_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_occlusion_ms,
                                    render_occlusion_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_graph_ms,
                                    render_graph_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_graph_pass_ms,
                                    render_graph_pass_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_graph_encoder_create_ms,
                                    render_graph_encoder_create_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_graph_encoder_finish_ms,
                                    render_graph_encoder_finish_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_graph_overhead_ms,
                                    render_graph_overhead_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_resource_mgmt_ms,
                                    render_resource_mgmt_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_acquire_ms,
                                    render_acquire_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_submit_ms,
                                    render_submit_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_present_ms,
                                    render_present_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.render_ui_build_ms,
                                    micros_to_ms(stats.render_ui_build_us.load(Ordering::Relaxed)),
                                    history_limit,
                                );
                            }

                            if let Some(stats) = renderer_stats.as_ref() {
                                let pass_timings = stats.pass_timings.read();
                                if !pass_timings.is_empty() {
                                    let mut seen = HashSet::new();
                                    history.render_pass_order = pass_timings
                                        .iter()
                                        .map(|timing| timing.name.clone())
                                        .collect();
                                    for timing in pass_timings.iter() {
                                        let ms = micros_to_ms(timing.duration_us);
                                        let entry = history
                                            .render_pass_ms
                                            .entry(timing.name.clone())
                                            .or_insert_with(VecDeque::new);
                                        push_history(entry, ms, history_limit);
                                        history.render_pass_last_ms.insert(timing.name.clone(), ms);
                                        seen.insert(timing.name.clone());
                                    }
                                    history.render_pass_ms.retain(|name, _| seen.contains(name));
                                    history
                                        .render_pass_last_ms
                                        .retain(|name, _| seen.contains(name));
                                }
                            }
                            if let Some(ui_perf) = ui_perf_stats.as_ref() {
                                push_history(
                                    &mut history.ui_system_ms,
                                    micros_to_ms(ui_perf.frame_us),
                                    history_limit,
                                );
                                push_history(
                                    &mut history.ui_run_frame_ms,
                                    micros_to_ms(ui_perf.run_frame_us),
                                    history_limit,
                                );
                                push_history(
                                    &mut history.ui_interaction_ms,
                                    micros_to_ms(ui_perf.interaction_us),
                                    history_limit,
                                );
                                push_history(
                                    &mut history.ui_scroll_metrics_ms,
                                    micros_to_ms(ui_perf.scroll_metrics_us),
                                    history_limit,
                                );
                                push_history(
                                    &mut history.ui_render_data_convert_ms,
                                    micros_to_ms(ui_perf.render_data_convert_us),
                                    history_limit,
                                );
                            }


                            if let Some(audio) = audio_stats {
                                let mix_ms = audio.mix_time_us as f64 / 1000.0;
                                let callback_ms = audio.callback_time_us as f64 / 1000.0;
                                push_history(&mut history.audio_mix_ms, mix_ms, history_limit);
                                push_history(
                                    &mut history.audio_callback_ms,
                                    callback_ms,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.audio_emitters,
                                    audio.active_emitters as f64,
                                    history_limit,
                                );
                                push_history(
                                    &mut history.audio_streaming_emitters,
                                    audio.streaming_emitters as f64,
                                    history_limit,
                                );
                            }
                        }

                        ui.separator();
                        ui.collapsing("main thread", |ui| {
                            let event_ms = history.main_event_ms.back().copied().unwrap_or(0.0);
                            let update_ms = history.main_update_ms.back().copied().unwrap_or(0.0);
                            ui.label(format!("event loop: {:.3} ms", event_ms));
                            ui.label(format!("update: {:.3} ms", update_ms));
                            draw_history_plot(
                                ui,
                                120.0,
                                &[
                                    (
                                        "event",
                                        &history.main_event_ms,
                                        Color32::from_rgb(240, 190, 90),
                                    ),
                                    (
                                        "update",
                                        &history.main_update_ms,
                                        Color32::from_rgb(90, 160, 240),
                                    ),
                                ],
                            );
                        });

                        ui.separator();
                        ui.collapsing("logic thread", |ui| {
                            let frame_ms = history.logic_frame_ms.back().copied().unwrap_or(0.0);
                            let tick_ms = history.logic_tick_ms.back().copied().unwrap_or(0.0);
                            let schedule_ms =
                                history.logic_schedule_ms.back().copied().unwrap_or(0.0);
                            let render_send_ms =
                                history.logic_render_send_ms.back().copied().unwrap_or(0.0);
                            ui.label(format!("frame: {:.3} ms", frame_ms));
                            ui.label(format!("tick: {:.3} ms", tick_ms));
                            ui.label(format!("schedule: {:.3} ms", schedule_ms));
                            ui.label(format!("render send: {:.3} ms", render_send_ms));
                            draw_history_plot(
                                ui,
                                140.0,
                                &[
                                    (
                                        "frame",
                                        &history.logic_frame_ms,
                                        Color32::from_rgb(90, 210, 120),
                                    ),
                                    (
                                        "tick",
                                        &history.logic_tick_ms,
                                        Color32::from_rgb(230, 170, 80),
                                    ),
                                    (
                                        "schedule",
                                        &history.logic_schedule_ms,
                                        Color32::from_rgb(90, 150, 240),
                                    ),
                                ],
                            );
                            draw_history_plot(
                                ui,
                                140.0,
                                &[
                                    (
                                        "asset",
                                        &history.logic_asset_ms,
                                        Color32::from_rgb(230, 210, 90),
                                    ),
                                    (
                                        "input",
                                        &history.logic_input_ms,
                                        Color32::from_rgb(80, 200, 200),
                                    ),
                                    (
                                        "render send",
                                        &history.logic_render_send_ms,
                                        Color32::from_rgb(220, 90, 90),
                                    ),
                                ],
                            );
                        });

                        ui.separator();
                        ui.collapsing("ecs systems", |ui| {
                            let render_data_ms =
                                history.ecs_render_data_ms.back().copied().unwrap_or(0.0);
                            let scene_spawn_ms =
                                history.ecs_scene_spawn_ms.back().copied().unwrap_or(0.0);
                            let scene_update_ms =
                                history.ecs_scene_update_ms.back().copied().unwrap_or(0.0);
                            ui.label(format!("render data: {:.3} ms", render_data_ms));
                            ui.label(format!("scene spawn: {:.3} ms", scene_spawn_ms));
                            ui.label(format!("scene update: {:.3} ms", scene_update_ms));
                            draw_history_plot(
                                ui,
                                140.0,
                                &[
                                    (
                                        "render data",
                                        &history.ecs_render_data_ms,
                                        Color32::from_rgb(100, 200, 255),
                                    ),
                                    (
                                        "scene spawn",
                                        &history.ecs_scene_spawn_ms,
                                        Color32::from_rgb(200, 140, 60),
                                    ),
                                    (
                                        "scene update",
                                        &history.ecs_scene_update_ms,
                                        Color32::from_rgb(120, 220, 140),
                                    ),
                                ],
                            );
                        });

                        ui.separator();
                        ui.collapsing("render thread", |ui| {
                            let frame_ms = history
                                .render_thread_frame_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            let messages_ms = history
                                .render_thread_messages_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            let upload_ms = history
                                .render_thread_upload_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            let render_ms = history
                                .render_thread_render_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            ui.label(format!("frame: {:.3} ms", frame_ms));
                            ui.label(format!("messages: {:.3} ms", messages_ms));
                            ui.label(format!("uploads: {:.3} ms", upload_ms));
                            ui.label(format!("render: {:.3} ms", render_ms));
                            draw_history_plot(
                                ui,
                                140.0,
                                &[
                                    (
                                        "frame",
                                        &history.render_thread_frame_ms,
                                        Color32::from_rgb(140, 220, 160),
                                    ),
                                    (
                                        "messages",
                                        &history.render_thread_messages_ms,
                                        Color32::from_rgb(90, 150, 255),
                                    ),
                                    (
                                        "uploads",
                                        &history.render_thread_upload_ms,
                                        Color32::from_rgb(240, 170, 90),
                                    ),
                                    (
                                        "render",
                                        &history.render_thread_render_ms,
                                        Color32::from_rgb(220, 90, 90),
                                    ),
                                ],
                            );
                        });

                        ui.separator();
                        ui.collapsing("audio", |ui| {
                            let mix_ms = history.audio_mix_ms.back().copied().unwrap_or(0.0);
                            let callback_ms =
                                history.audio_callback_ms.back().copied().unwrap_or(0.0);
                            let emitters = history.audio_emitters.back().copied().unwrap_or(0.0);
                            let streaming = history
                                .audio_streaming_emitters
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            ui.label(format!("mix: {:.3} ms", mix_ms));
                            ui.label(format!("callback: {:.3} ms", callback_ms));
                            ui.label(format!(
                                "emitters: {:.0} (streaming {:.0})",
                                emitters, streaming
                            ));
                            if let Some(audio) = audio_stats {
                                if audio.measured_sample_rate != audio.sample_rate {
                                    ui.label(format!(
                                        "output: {} Hz (measured {} Hz), {} ch, buffer {}",
                                        audio.sample_rate,
                                        audio.measured_sample_rate,
                                        audio.channels,
                                        audio.buffer_frames
                                    ));
                                } else {
                                    ui.label(format!(
                                        "output: {} Hz, {} ch, buffer {}",
                                        audio.sample_rate, audio.channels, audio.buffer_frames
                                    ));
                                }
                            }
                            draw_history_plot(
                                ui,
                                140.0,
                                &[
                                    (
                                        "mix ms",
                                        &history.audio_mix_ms,
                                        Color32::from_rgb(160, 210, 120),
                                    ),
                                    (
                                        "callback ms",
                                        &history.audio_callback_ms,
                                        Color32::from_rgb(220, 160, 90),
                                    ),
                                ],
                            );
                        });

                        ui.separator();
                        ui.collapsing("renderer internals", |ui| {
                            let globals_ms = history
                                .render_prepare_globals_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            let streaming_ms = history
                                .render_streaming_plan_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            let occlusion_ms =
                                history.render_occlusion_ms.back().copied().unwrap_or(0.0);
                            let graph_ms = history.render_graph_ms.back().copied().unwrap_or(0.0);
                            let resource_ms = history
                                .render_resource_mgmt_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            let acquire_ms =
                                history.render_acquire_ms.back().copied().unwrap_or(0.0);
                            let submit_ms = history.render_submit_ms.back().copied().unwrap_or(0.0);
                            let present_ms =
                                history.render_present_ms.back().copied().unwrap_or(0.0);
                            ui.label(format!("prepare globals: {:.3} ms", globals_ms));
                            ui.label(format!("streaming plan: {:.3} ms", streaming_ms));
                            ui.label(format!("occlusion: {:.3} ms", occlusion_ms));
                            ui.label(format!("render graph: {:.3} ms", graph_ms));
                            ui.label(format!("resource mgmt: {:.3} ms", resource_ms));
                            draw_history_plot(
                                ui,
                                160.0,
                                &[
                                    (
                                        "prepare globals",
                                        &history.render_prepare_globals_ms,
                                        Color32::from_rgb(120, 210, 120),
                                    ),
                                    (
                                        "streaming plan",
                                        &history.render_streaming_plan_ms,
                                        Color32::from_rgb(90, 170, 255),
                                    ),
                                    (
                                        "occlusion",
                                        &history.render_occlusion_ms,
                                        Color32::from_rgb(240, 120, 120),
                                    ),
                                    (
                                        "render graph",
                                        &history.render_graph_ms,
                                        Color32::from_rgb(200, 180, 80),
                                    ),
                                    (
                                        "resource mgmt",
                                        &history.render_resource_mgmt_ms,
                                        Color32::from_rgb(120, 200, 200),
                                    ),
                                ],
                            );

                            ui.separator();
                            ui.label(format!("swapchain acquire: {:.3} ms", acquire_ms));
                            ui.label(format!("queue submit: {:.3} ms", submit_ms));
                            ui.label(format!("present: {:.3} ms", present_ms));
                            draw_history_plot(
                                ui,
                                140.0,
                                &[
                                    (
                                        "acquire",
                                        &history.render_acquire_ms,
                                        Color32::from_rgb(180, 150, 240),
                                    ),
                                    (
                                        "submit",
                                        &history.render_submit_ms,
                                        Color32::from_rgb(160, 200, 140),
                                    ),
                                    (
                                        "present",
                                        &history.render_present_ms,
                                        Color32::from_rgb(240, 140, 100),
                                    ),
                                ],
                            );
                        });

            ui.separator();
            ui.collapsing("ui internals", |ui| {
                if !profiling_enabled {
                    ui.label("runtime profiling is disabled; UI timings are not sampled.");
                }
                if let Some(ui_perf) = ui_perf_stats.as_ref() {
                    ui.label(format!(
                        "ui_system: {:.3} ms (avg {:.3} / peak {:.3})",
                                    micros_to_ms(ui_perf.frame_us),
                                    micros_f64_to_ms(ui_perf.frame_ema_us),
                                    micros_to_ms(ui_perf.frame_peak_us),
                                ));
                                ui.label(format!(
                                    "run_frame: {:.3} ms (avg {:.3} / peak {:.3})",
                                    micros_to_ms(ui_perf.run_frame_us),
                                    micros_f64_to_ms(ui_perf.run_frame_ema_us),
                                    micros_to_ms(ui_perf.run_frame_peak_us),
                                ));
                                ui.label(format!(
                                    "interaction: {:.3} ms | scroll metrics: {:.3} ms | convert: {:.3} ms",
                                    micros_to_ms(ui_perf.interaction_us),
                                    micros_to_ms(ui_perf.scroll_metrics_us),
                                    micros_to_ms(ui_perf.render_data_convert_us),
                                ));
                                ui.label(format!(
                                    "windows: {} visible / {} total | built cmds: {} | active cmds: {} | converted: {} | layout rects: {} | ui revision: {}",
                                    ui_perf.windows_visible,
                                    ui_perf.windows_total,
                                    ui_perf.built_draw_commands,
                                    ui_perf.draw_commands,
                                    if ui_perf.converted_this_frame {
                                        "yes"
                                    } else {
                                        "no"
                                    },
                                    ui_perf.layout_rects,
                                    ui_perf.ui_revision,
                                ));
                            } else {
                                ui.label("ui system metrics unavailable");
                            }

                            if let Some(stats) = renderer_stats.as_ref() {
                                ui.separator();
                                ui.label(format!(
                                    "build_ui_draw_data: {:.3} ms (rebuilt this frame: {})",
                                    micros_to_ms(stats.render_ui_build_us.load(Ordering::Relaxed)),
                                    stats.render_ui_rebuilt.load(Ordering::Relaxed) != 0,
                                ));
                                ui.label(format!(
                                    "render UI payload: commands {} | instances {} | batches {} | textures {}",
                                    stats.render_ui_command_count.load(Ordering::Relaxed),
                                    stats.render_ui_instance_count.load(Ordering::Relaxed),
                                    stats.render_ui_batch_count.load(Ordering::Relaxed),
                                    stats.render_ui_texture_count.load(Ordering::Relaxed),
                                ));
                            }

                            draw_history_plot(
                                ui,
                                170.0,
                                &[
                                    (
                                        "ui_system",
                                        &history.ui_system_ms,
                                        Color32::from_rgb(120, 200, 255),
                                    ),
                                    (
                                        "ui_run_frame",
                                        &history.ui_run_frame_ms,
                                        Color32::from_rgb(120, 255, 170),
                                    ),
                                    (
                                        "renderer_ui_build",
                                        &history.render_ui_build_ms,
                                        Color32::from_rgb(255, 190, 120),
                                    ),
                                ],
                            );
                        });
                        ui.separator();
                        ui.collapsing("render graph execution breakdown", |ui| {
                            let pass_ms =
                                history.render_graph_pass_ms.back().copied().unwrap_or(0.0);
                            let create_ms = history
                                .render_graph_encoder_create_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            let finish_ms = history
                                .render_graph_encoder_finish_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);
                            let overhead_ms = history
                                .render_graph_overhead_ms
                                .back()
                                .copied()
                                .unwrap_or(0.0);

                            ui.label(format!("pass encode: {:.3} ms", pass_ms));
                            ui.label(format!("encoder create: {:.3} ms", create_ms));
                            ui.label(format!("encoder finish: {:.3} ms", finish_ms));
                            ui.label(format!("graph overhead: {:.3} ms", overhead_ms));

                            draw_history_plot(
                                ui,
                                160.0,
                                &[
                                    (
                                        "pass encode",
                                        &history.render_graph_pass_ms,
                                        Color32::from_rgb(90, 170, 240),
                                    ),
                                    (
                                        "encoder create",
                                        &history.render_graph_encoder_create_ms,
                                        Color32::from_rgb(220, 150, 80),
                                    ),
                                    (
                                        "encoder finish",
                                        &history.render_graph_encoder_finish_ms,
                                        Color32::from_rgb(200, 110, 110),
                                    ),
                                    (
                                        "overhead",
                                        &history.render_graph_overhead_ms,
                                        Color32::from_rgb(120, 210, 200),
                                    ),
                                ],
                            );
                        });

                        ui.separator();
                        ui.collapsing("render graph internals", |ui| {
                            let plot_limit = render_pass_plot_limit as usize;
                            let mut entries: Vec<(String, f64)> = history
                                .render_pass_last_ms
                                .iter()
                                .map(|(name, ms)| (name.clone(), *ms))
                                .collect();
                            if entries.is_empty() {
                                ui.label("render pass timings unavailable");
                                return;
                            }

                            entries.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal)
                            });
                            let total_ms: f64 = entries.iter().map(|(_, ms)| *ms).sum();
                            ui.label(format!("total (sum of passes): {:.3} ms", total_ms));

                            if plot_limit > 0 {
                                let mut plot_names = Vec::new();
                                for (name, _) in entries.iter().take(plot_limit) {
                                    plot_names.push(name.clone());
                                }
                                let mut series = Vec::new();
                                for name in &plot_names {
                                    if let Some(hist) = history.render_pass_ms.get(name) {
                                        series.push((name.as_str(), hist, pass_color(name)));
                                    }
                                }
                                draw_history_plot(ui, 180.0, &series);
                            } else {
                                ui.label("render pass plot limit is 0; adjust to show graphs");
                            }

                            ui.separator();
                            egui::ScrollArea::vertical()
                                .auto_shrink([false, false])
                                .max_height(240.0)
                                .show(ui, |ui| {
                                    egui::Grid::new("render_pass_stats").striped(true).show(
                                        ui,
                                        |ui| {
                                            ui.label("pass");
                                            ui.label("last ms");
                                            ui.label("%");
                                            ui.end_row();
                                            for (name, ms) in entries.iter() {
                                                ui.label(name);
                                                ui.label(format!("{:.3}", ms));
                                                if total_ms > 0.0 {
                                                    ui.label(format!(
                                                        "{:.1}%",
                                                        (ms / total_ms) * 100.0
                                                    ));
                                                } else {
                                                    ui.label("-");
                                                }
                                                ui.end_row();
                                            }
                                        },
                                    );
                                });
                        });
                    });
}

pub struct StatsUI {}

impl StatsUI {
    pub fn add_windows(egui_res: &mut EguiResource) {
        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                let (fps, tps) = {
                    let pm = &world
                        .get_resource::<BecsPerformanceMetrics>()
                        .expect("PerformanceMetrics resource not found")
                        .0;
                    (
                        pm.fps.load(Ordering::Relaxed),
                        pm.tps.load(Ordering::Relaxed),
                    )
                };

                let render_object_count = world
                    .get_resource::<RenderObjectCount>()
                    .map(|count| count.0);

                let (vram_used_mb, vram_soft_mb, vram_hard_mb) = world
                    .get_resource::<BecsRendererStats>()
                    .map(|stats| {
                        let mb = 1024.0 * 1024.0;
                        let used = stats.0.vram_used_bytes.load(Ordering::Relaxed) as f64 / mb;
                        let soft =
                            stats.0.vram_soft_limit_bytes.load(Ordering::Relaxed) as f64 / mb;
                        let hard =
                            stats.0.vram_hard_limit_bytes.load(Ordering::Relaxed) as f64 / mb;
                        (used, soft, hard)
                    })
                    .unwrap_or((0.0, 0.0, 0.0));

                let (mesh_bytes, tex_bytes, mat_bytes, audio_bytes) = {
                    #[cfg(target_arch = "wasm32")]
                    let asset_server = world.get_non_send_resource::<BecsAssetServer>();
                    #[cfg(not(target_arch = "wasm32"))]
                    let asset_server = world.get_resource::<BecsAssetServer>();
                    asset_server
                        .map(|srv| {
                            let srv = srv.0.lock();
                            let (mesh, tex, mat) = srv.cache_usage_bytes();
                            let audio = srv.audio_cache_usage_bytes();
                            (mesh, tex, mat, audio)
                        })
                        .unwrap_or((0, 0, 0, 0))
                };

                let history_samples = world
                    .get_resource::<BecsRuntimeProfiling>()
                    .map(|profiling| profiling.0.history_samples.load(Ordering::Relaxed) as usize)
                    .unwrap_or(0);

                let mut history = world
                    .get_resource_mut::<DebugGraphHistory>()
                    .expect("DebugGraphHistory resource not found");

                let mb_div = 1024.0 * 1024.0;
                push_history(&mut history.fps, fps as f64, history_samples);
                push_history(&mut history.tps, tps as f64, history_samples);
                push_history(&mut history.vram_bytes, vram_used_mb, history_samples);
                push_history(
                    &mut history.mesh_bytes,
                    mesh_bytes as f64 / mb_div,
                    history_samples,
                );
                push_history(
                    &mut history.texture_bytes,
                    tex_bytes as f64 / mb_div,
                    history_samples,
                );
                push_history(
                    &mut history.material_bytes,
                    mat_bytes as f64 / mb_div,
                    history_samples,
                );
                push_history(
                    &mut history.audio_bytes,
                    audio_bytes as f64 / mb_div,
                    history_samples,
                );

                ui.label(format!("FPS: {}  | TPS: {}", fps, tps));
                if let Some(render_object_count) = render_object_count {
                    ui.label(format!("render objects: {}", render_object_count));
                }
                ui.label(format!(
                    "VRAM: {:.1} MiB (soft {:.0} / hard {:.0})",
                    vram_used_mb, vram_soft_mb, vram_hard_mb
                ));
                ui.label(format!(
                    "Asset cache (MiB): meshes {:.1}  textures {:.1}  materials {:.1}  audio {:.1}",
                    mesh_bytes as f64 / mb_div,
                    tex_bytes as f64 / mb_div,
                    mat_bytes as f64 / mb_div,
                    audio_bytes as f64 / mb_div
                ));

                draw_history_plot(
                    ui,
                    120.0,
                    &[
                        ("FPS", &history.fps, Color32::from_rgb(90, 210, 120)),
                        ("TPS", &history.tps, Color32::from_rgb(90, 160, 240)),
                    ],
                );

                draw_history_plot(
                    ui,
                    140.0,
                    &[
                        (
                            "VRAM MiB",
                            &history.vram_bytes,
                            Color32::from_rgb(70, 150, 255),
                        ),
                        (
                            "mesh cache MiB",
                            &history.mesh_bytes,
                            Color32::from_rgb(230, 180, 80),
                        ),
                        (
                            "texture cache MiB",
                            &history.texture_bytes,
                            Color32::from_rgb(220, 90, 160),
                        ),
                        (
                            "material cache MiB",
                            &history.material_bytes,
                            Color32::from_rgb(150, 210, 220),
                        ),
                        (
                            "audio cache MiB",
                            &history.audio_bytes,
                            Color32::from_rgb(160, 200, 120),
                        ),
                    ],
                );
            }),
            window_spec("helmer metrics"),
        ));
        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                draw_runtime_profiling_panel(ui, world);
            }),
            window_spec("profiling"),
        ));

        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                ui.heading("runtime tuning");
                ui.separator();

                if let Some(runtime_tuning) = world.get_resource::<BecsRuntimeTuning>() {
                    let tuning = &runtime_tuning.0;
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.collapsing("timing", |ui| {
                                let mut target_tickrate = tuning.load_target_tickrate();
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut target_tickrate)
                                            .speed(1.0)
                                            .prefix("target tickrate: "),
                                    )
                                    .changed()
                                {
                                    tuning.store_target_tickrate(target_tickrate);
                                }

                                let mut target_fps = tuning.load_target_fps().unwrap_or(60.0);
                                let mut fps_enabled = tuning.load_target_fps().is_some();
                                if ui.checkbox(&mut fps_enabled, "cap render fps").changed() {
                                    if fps_enabled {
                                        tuning.store_target_fps(Some(target_fps));
                                    } else {
                                        tuning.store_target_fps(None);
                                    }
                                }
                                if ui
                                    .add_enabled(
                                        fps_enabled,
                                        egui::DragValue::new(&mut target_fps)
                                            .speed(1.0)
                                            .prefix("target fps: "),
                                    )
                                    .changed()
                                {
                                    tuning.store_target_fps(Some(target_fps));
                                }
                            });

                            ui.separator();
                            ui.collapsing("thread budgets", |ui| {
                                let mut task_worker_count =
                                    tuning.load_task_worker_count().min(u32::MAX as usize) as u32;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut task_worker_count)
                                            .speed(1.0)
                                            .prefix("task workers: "),
                                    )
                                    .changed()
                                {
                                    tuning.store_task_worker_count(task_worker_count as usize);
                                }

                                let mut render_message_capacity =
                                    tuning.render_message_capacity.load(Ordering::Relaxed) as u32;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut render_message_capacity)
                                            .speed(1.0)
                                            .prefix("render message capacity: "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .render_message_capacity
                                        .store(render_message_capacity as usize, Ordering::Relaxed);
                                }

                                let mut asset_stream_capacity =
                                    tuning.asset_stream_queue_capacity.load(Ordering::Relaxed)
                                        as u32;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut asset_stream_capacity)
                                            .speed(1.0)
                                            .prefix("asset stream queue cap: "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .asset_stream_queue_capacity
                                        .store(asset_stream_capacity as usize, Ordering::Relaxed);
                                }

                                let mut asset_worker_capacity =
                                    tuning.asset_worker_queue_capacity.load(Ordering::Relaxed)
                                        as u32;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut asset_worker_capacity)
                                            .speed(1.0)
                                            .prefix("asset worker queue cap: "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .asset_worker_queue_capacity
                                        .store(asset_worker_capacity as usize, Ordering::Relaxed);
                                }
                            });

                            ui.separator();
                            ui.collapsing("asset uploads", |ui| {
                                let mb = 1024.0 * 1024.0;
                                let mut max_pending_uploads =
                                    tuning.max_pending_asset_uploads.load(Ordering::Relaxed) as u32;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut max_pending_uploads)
                                            .speed(1.0)
                                            .prefix("max pending uploads: "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .max_pending_asset_uploads
                                        .store(max_pending_uploads as usize, Ordering::Relaxed);
                                }

                                let mut max_pending_mb =
                                    tuning.max_pending_asset_bytes.load(Ordering::Relaxed) as f32
                                        / mb;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut max_pending_mb)
                                            .speed(8.0)
                                            .prefix("max pending MiB: "),
                                    )
                                    .changed()
                                {
                                    tuning.max_pending_asset_bytes.store(
                                        (max_pending_mb.max(0.0) * mb) as usize,
                                        Ordering::Relaxed,
                                    );
                                }

                                let mut uploads_per_frame =
                                    tuning.asset_uploads_per_frame.load(Ordering::Relaxed) as u32;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut uploads_per_frame)
                                            .speed(1.0)
                                            .prefix("uploads per frame: "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .asset_uploads_per_frame
                                        .store(uploads_per_frame as usize, Ordering::Relaxed);
                                }
                            });

                            ui.separator();
                            ui.collapsing("wgpu cleanup", |ui| {
                                let mut poll_interval =
                                    tuning.wgpu_poll_interval_frames.load(Ordering::Relaxed);
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut poll_interval)
                                            .speed(1.0)
                                            .prefix("poll interval (frames): "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .wgpu_poll_interval_frames
                                        .store(poll_interval, Ordering::Relaxed);
                                }

                                let mut poll_mode = tuning.wgpu_poll_mode.load(Ordering::Relaxed);
                                let poll_labels = ["off", "poll", "wait"];
                                let selected =
                                    poll_labels.get(poll_mode as usize).unwrap_or(&"unknown");
                                ComboBox::from_label("poll mode")
                                    .selected_text(*selected)
                                    .show_ui(ui, |ui| {
                                        for (idx, label) in poll_labels.iter().enumerate() {
                                            ui.selectable_value(&mut poll_mode, idx as u32, *label);
                                        }
                                    });
                                if tuning.wgpu_poll_mode.load(Ordering::Relaxed) != poll_mode {
                                    tuning.wgpu_poll_mode.store(poll_mode, Ordering::Relaxed);
                                }
                            });

                            ui.separator();
                            ui.collapsing("input + window", |ui| {
                                let mut pixels_per_line =
                                    tuning.pixels_per_line.load(Ordering::Relaxed);
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut pixels_per_line)
                                            .speed(1.0)
                                            .prefix("pixels per line: "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .pixels_per_line
                                        .store(pixels_per_line, Ordering::Relaxed);
                                }

                                let mut title_update_ms =
                                    tuning.title_update_ms.load(Ordering::Relaxed);
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut title_update_ms)
                                            .speed(1.0)
                                            .prefix("title update ms: "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .title_update_ms
                                        .store(title_update_ms, Ordering::Relaxed);
                                }

                                let mut resize_debounce_ms =
                                    tuning.resize_debounce_ms.load(Ordering::Relaxed);
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut resize_debounce_ms)
                                            .speed(1.0)
                                            .prefix("resize debounce ms: "),
                                    )
                                    .changed()
                                {
                                    tuning
                                        .resize_debounce_ms
                                        .store(resize_debounce_ms, Ordering::Relaxed);
                                }
                            });
                        });
                } else {
                    ui.label("runtime tuning unavailable");
                }
            }),
            window_spec("runtime tuning"),
        ));

        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                ui.heading("render worker");
                ui.separator();

                if let Some(mut tuning_res) = world.get_resource_mut::<BecsRenderWorkerTuning>() {
                    let tuning = &mut tuning_res.0;
                    if ui.button("reset render worker tuning").clicked() {
                        *tuning = Default::default();
                    }

                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.collapsing("worker queues", |ui| {
                                drag_usize(
                                    ui,
                                    &mut tuning.change_queue_capacity,
                                    "change queue capacity: ",
                                    1.0,
                                );
                            });

                            ui.separator();
                            ui.collapsing("partitioning", |ui| {
                                ui.add(
                                    egui::DragValue::new(&mut tuning.cell_size)
                                        .speed(0.5)
                                        .prefix("cell size: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut tuning.full_radius)
                                        .speed(1.0)
                                        .prefix("full radius: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut tuning.hlod_radius)
                                        .speed(1.0)
                                        .prefix("hlod radius: "),
                                );
                                drag_usize(ui, &mut tuning.max_full_cells, "max full cells: ", 1.0);
                                drag_usize(ui, &mut tuning.max_hlod_cells, "max hlod cells: ", 1.0);
                            });

                            ui.separator();
                            ui.collapsing("hlod proxies", |ui| {
                                ui.checkbox(&mut tuning.hlod_enabled, "hlod enabled");
                                drag_usize(
                                    ui,
                                    &mut tuning.hlod_min_cell_objects,
                                    "min cell objects: ",
                                    1.0,
                                );
                                drag_usize(
                                    ui,
                                    &mut tuning.hlod_proxy_mesh_id,
                                    "proxy mesh id: ",
                                    1.0,
                                );
                                drag_usize(
                                    ui,
                                    &mut tuning.hlod_proxy_material_id,
                                    "proxy material id: ",
                                    1.0,
                                );
                                ui.add(
                                    egui::DragValue::new(&mut tuning.hlod_proxy_scale)
                                        .speed(0.05)
                                        .prefix("proxy scale: "),
                                );
                                drag_usize(
                                    ui,
                                    &mut tuning.hlod_proxy_id_base,
                                    "proxy id base: ",
                                    1.0,
                                );
                                ui.add(
                                    egui::DragValue::new(&mut tuning.hlod_hysteresis)
                                        .speed(0.01)
                                        .prefix("hlod hysteresis: "),
                                );
                            });

                            ui.separator();
                            ui.collapsing("culling thresholds", |ui| {
                                ui.add(
                                    egui::DragValue::new(&mut tuning.cull_camera_move_threshold)
                                        .speed(0.01)
                                        .prefix("cull move threshold: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut tuning.cull_camera_rot_threshold)
                                        .speed(0.001)
                                        .prefix("cull rot threshold: "),
                                );
                                ui.add(
                                    egui::DragValue::new(
                                        &mut tuning.streaming_camera_move_threshold,
                                    )
                                    .speed(0.01)
                                    .prefix("stream move threshold: "),
                                );
                                ui.add(
                                    egui::DragValue::new(
                                        &mut tuning.streaming_camera_rot_threshold,
                                    )
                                    .speed(0.001)
                                    .prefix("stream rot threshold: "),
                                );
                                ui.add(
                                    egui::DragValue::new(
                                        &mut tuning.streaming_motion_speed_threshold,
                                    )
                                    .speed(0.01)
                                    .prefix("motion speed threshold: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut tuning.streaming_motion_budget_scale)
                                        .speed(0.01)
                                        .prefix("motion budget scale: "),
                                );
                            });

                            ui.separator();
                            ui.collapsing("streaming budgets", |ui| {
                                drag_usize(
                                    ui,
                                    &mut tuning.streaming_object_budget,
                                    "streaming object budget: ",
                                    1.0,
                                );
                                drag_usize(
                                    ui,
                                    &mut tuning.streaming_request_budget,
                                    "streaming request budget: ",
                                    1.0,
                                );
                            });

                            ui.separator();
                            ui.collapsing("per-object toggles", |ui| {
                                ui.checkbox(&mut tuning.per_object_culling, "per-object culling");
                                ui.checkbox(&mut tuning.per_object_lod, "per-object LOD");
                            });
                        });
                } else {
                    ui.label("render worker tuning unavailable");
                }
            }),
            window_spec("render worker"),
        ));

        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                let sender = world
                    .get_resource::<BecsRenderSender>()
                    .map(|s| s.0.clone());
                {
                    let mut runtime_cfg = world
                        .get_resource_mut::<BecsRuntimeConfig>()
                        .expect("RuntimeConfig resource not found");
                    let mut wgpu_experimental = runtime_cfg.0.wgpu_experimental_features;
                    let mut wgpu_backend = runtime_cfg.0.wgpu_backend;
                    let mut binding_backend = runtime_cfg.0.binding_backend;
                    let mut request_recreate = false;
                    let mut wgpu_settings_changed = false;

                    {
                        let render_cfg = &mut runtime_cfg.0.render_config;

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

                        let mut shade_smooth = render_cfg.shader_constants.shade_smooth != 0;
                        if ui.checkbox(&mut shade_smooth, "shade smooth").changed() {
                            render_cfg.shader_constants.shade_smooth =
                                if shade_smooth { 1 } else { 0 };
                        }

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
                                            &mut render_cfg.shader_constants.skylight_contribution,
                                            i as u32,
                                            *label,
                                        );
                                    }
                                });
                        }

                        ui.separator();
                        ui.heading("general");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.shader_constants.mip_bias)
                                .speed(0.1)
                                .prefix("mip bias: "),
                        );

                        ui.separator();
                        ui.heading("transparency");
                        ui.checkbox(&mut render_cfg.transparent_pass, "transparent pass");
                        let sort_labels = ["none", "back-to-front", "front-to-back"];
                        let sort_idx = match render_cfg.transparent_sort_mode {
                            TransparentSortMode::None => 0,
                            TransparentSortMode::BackToFront => 1,
                            TransparentSortMode::FrontToBack => 2,
                        };
                        ComboBox::from_label("transparent sort")
                            .selected_text(sort_labels[sort_idx])
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut render_cfg.transparent_sort_mode,
                                    TransparentSortMode::None,
                                    sort_labels[0],
                                );
                                ui.selectable_value(
                                    &mut render_cfg.transparent_sort_mode,
                                    TransparentSortMode::BackToFront,
                                    sort_labels[1],
                                );
                                ui.selectable_value(
                                    &mut render_cfg.transparent_sort_mode,
                                    TransparentSortMode::FrontToBack,
                                    sort_labels[2],
                                );
                            });
                        ui.checkbox(
                            &mut render_cfg.transparent_shadows,
                            "transparent shadows (blend as mask)",
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.alpha_cutoff_default)
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .prefix("alpha cutoff default: "),
                        );

                        ui.separator();
                        ui.heading("camera");
                        ui.checkbox(&mut render_cfg.freeze_render_camera, "freeze render camera");
                        ui.label("streaming continues from logical camera");

                        ui.separator();
                        ui.heading("viewport");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.viewport_resize_debounce_ms)
                                .speed(1.0)
                                .range(0..=1000)
                                .prefix("resize debounce ms: "),
                        );
                        ui.label("0 = immediate resize apply");

                        ui.separator();
                        ui.heading("text/sprite quality");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_world_pixels_per_unit)
                                .speed(0.1)
                                .range(0.01..=4096.0)
                                .prefix("pixels/world unit: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_world_raster_oversample)
                                .speed(0.05)
                                .range(0.01..=64.0)
                                .prefix("raster oversample: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_world_raster_min_scale)
                                .speed(0.1)
                                .range(0.000001..=4096.0)
                                .prefix("min raster scale: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_world_raster_max_scale)
                                .speed(0.5)
                                .range(0.000001..=32768.0)
                                .prefix("max raster scale: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_world_glyph_max_raster_px)
                                .speed(1.0)
                                .range(1..=u32::MAX)
                                .prefix("max glyph raster px: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_world_raster_scale_step)
                                .speed(0.01)
                                .range(0.0..=32.0)
                                .prefix("raster scale step: "),
                        );
                        ui.checkbox(
                            &mut render_cfg.text_world_auto_quality,
                            "auto scale by VRAM budget",
                        );
                        ui.add(
                            egui::DragValue::new(
                                &mut render_cfg.text_world_auto_quality_min_factor,
                            )
                            .speed(0.01)
                            .range(0.0..=8.0)
                            .prefix("auto quality min: "),
                        );
                        ui.add(
                            egui::DragValue::new(
                                &mut render_cfg.text_world_auto_quality_max_factor,
                            )
                            .speed(0.01)
                            .range(0.0..=8.0)
                            .prefix("auto quality max: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_layout_scale)
                                .speed(0.05)
                                .range(0.000001..=16.0)
                                .prefix("layout scale: "),
                        );
                        ui.checkbox(
                            &mut render_cfg.text_screen_layout_quantize,
                            "screen layout quantize",
                        );
                        ui.checkbox(
                            &mut render_cfg.text_world_layout_quantize,
                            "world layout quantize",
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_min_font_px)
                                .speed(0.001)
                                .range(0.000001..=16.0)
                                .prefix("text min font px: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_glyph_atlas_padding_px)
                                .speed(1.0)
                                .range(0..=1024)
                                .prefix("glyph atlas padding px: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.text_glyph_uv_inset_px)
                                .speed(0.01)
                                .range(0.0..=64.0)
                                .prefix("glyph uv inset px: "),
                        );
                        render_cfg.text_min_font_px = render_cfg.text_min_font_px.max(f32::EPSILON);
                        render_cfg.text_world_pixels_per_unit = render_cfg
                            .text_world_pixels_per_unit
                            .max(render_cfg.text_min_font_px);
                        render_cfg.text_world_raster_oversample =
                            render_cfg.text_world_raster_oversample.max(f32::EPSILON);
                        render_cfg.text_world_raster_min_scale = render_cfg
                            .text_world_raster_min_scale
                            .max(render_cfg.text_min_font_px);
                        render_cfg.text_world_raster_max_scale = render_cfg
                            .text_world_raster_max_scale
                            .max(render_cfg.text_world_raster_min_scale);
                        render_cfg.text_world_glyph_max_raster_px =
                            render_cfg.text_world_glyph_max_raster_px.max(1);
                        render_cfg.text_world_raster_scale_step =
                            render_cfg.text_world_raster_scale_step.max(0.0);
                        render_cfg.text_world_auto_quality_min_factor =
                            render_cfg.text_world_auto_quality_min_factor.max(0.0);
                        render_cfg.text_world_auto_quality_max_factor = render_cfg
                            .text_world_auto_quality_max_factor
                            .max(render_cfg.text_world_auto_quality_min_factor);
                        render_cfg.text_layout_scale =
                            render_cfg.text_layout_scale.max(f32::EPSILON);
                        render_cfg.text_glyph_uv_inset_px =
                            render_cfg.text_glyph_uv_inset_px.max(0.0);

                        ui.separator();
                        ui.heading("Culling & LOD");
                        ui.checkbox(&mut render_cfg.frustum_culling, "Frustum Culling");
                        ui.checkbox(
                            &mut render_cfg.occlusion_culling,
                            "Occlusion Culling (Hi-Z)",
                        );
                        ui.checkbox(&mut render_cfg.lod, "LOD");
                        ui.checkbox(&mut render_cfg.gpu_driven, "GPU-driven (indirect)");
                        ui.checkbox(
                            &mut render_cfg.gpu_multi_draw_indirect,
                            "multi-draw indirect",
                        );
                        ui.checkbox(&mut render_cfg.render_bundles, "Render bundles (cached)");
                        ui.label("bundle invalidation is automatic");
                        ui.checkbox(
                            &mut render_cfg.deterministic_rendering,
                            "deterministic render/resource management",
                        );
                        ui.checkbox(
                            &mut render_cfg.use_dont_care_load_ops,
                            "use DontCare load ops for fullscreen passes",
                        );
                        ui.checkbox(
                            &mut render_cfg.use_transient_textures,
                            "use transient textures",
                        );
                        ui.checkbox(
                            &mut render_cfg.use_transient_aliasing,
                            "alias transient textures",
                        );
                        ui.checkbox(&mut render_cfg.use_mesh_shaders, "use mesh shaders");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.gpu_cull_depth_bias)
                                .speed(0.0001)
                                .prefix("gpu depth bias: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.gpu_cull_rect_pad)
                                .speed(0.1)
                                .prefix("gpu rect pad: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.cull_interval_frames)
                                .speed(1.0)
                                .prefix("cull interval frames: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.lod_interval_frames)
                                .speed(1.0)
                                .prefix("lod interval frames: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.streaming_interval_frames)
                                .speed(1.0)
                                .prefix("streaming interval frames: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.streaming_scan_budget)
                                .speed(1.0)
                                .prefix("streaming scan budget: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.streaming_request_budget)
                                .speed(1.0)
                                .prefix("streaming request budget: "),
                        );
                        ui.checkbox(
                            &mut render_cfg.streaming_allow_full_scan,
                            "allow full streaming scan fallback",
                        );
                        ui.label("interval = 0 means only update on changes");

                        ui.separator();
                        ui.heading("Transform Deltas");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.transform_epsilon)
                                .speed(0.0001)
                                .range(0.0..=1.0)
                                .prefix("position/scale epsilon: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.rotation_epsilon)
                                .speed(0.0001)
                                .range(0.0..=1.0)
                                .prefix("rotation epsilon: "),
                        );

                        ui.separator();
                        ui.heading("Frame & Shadows");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.frames_in_flight)
                                .speed(1.0)
                                .prefix("frames in flight: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.shadow_map_resolution)
                                .speed(64.0)
                                .prefix("shadow map resolution: "),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut render_cfg.shadow_cascade_count,
                                1..=(MAX_SHADOW_CASCADES as u32),
                            )
                            .text("shadow cascade count"),
                        );
                        ui.collapsing("shadow cascade splits", |ui| {
                            for (idx, split) in
                                render_cfg.shadow_cascade_splits.iter_mut().enumerate()
                            {
                                ui.add(
                                    egui::DragValue::new(split)
                                        .speed(0.1)
                                        .prefix(format!("split {idx}: ")),
                                );
                            }
                        });

                        ui.separator();
                        ui.heading("Ray Tracing");
                        ui.checkbox(&mut render_cfg.rt_accumulation, "accumulation");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.rt_max_accumulation_frames)
                                .speed(1.0)
                                .prefix("max accumulation frames: "),
                        );
                        ui.add(
                            egui::Slider::new(&mut render_cfg.rt_samples_per_frame, 1..=64)
                                .text("samples per frame"),
                        );
                        ui.add(
                            egui::Slider::new(&mut render_cfg.rt_max_bounces, 1..=16)
                                .text("max bounces"),
                        );
                        ui.add(
                            egui::Slider::new(&mut render_cfg.rt_direct_light_samples, 1..=8)
                                .text("direct light samples"),
                        );
                        ui.checkbox(&mut render_cfg.rt_direct_lighting, "direct lighting");
                        ui.checkbox(&mut render_cfg.rt_shadows, "shadows");
                        ui.checkbox(&mut render_cfg.rt_use_textures, "use material textures");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.rt_exposure)
                                .speed(0.05)
                                .range(0.0..=10.0)
                                .prefix("exposure: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.rt_env_intensity)
                                .speed(0.05)
                                .range(0.0..=10.0)
                                .prefix("env intensity: "),
                        );
                        ui.collapsing("sky ", |ui| {
                            ui.add(
                                egui::Slider::new(&mut render_cfg.rt_sky_view_samples, 1..=64)
                                    .text("sky view samples"),
                            );
                            ui.add(
                                egui::Slider::new(&mut render_cfg.rt_sky_sun_samples, 1..=64)
                                    .text("sky sun samples"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_sky_multi_scatter_strength)
                                    .speed(0.05)
                                    .range(0.0..=4.0)
                                    .prefix("multi scatter strength: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_sky_multi_scatter_power)
                                    .speed(0.05)
                                    .range(0.1..=8.0)
                                    .prefix("multi scatter power: "),
                            );
                        });
                        ui.collapsing("stability", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_camera_pos_epsilon)
                                    .speed(0.0001)
                                    .range(0.0..=1.0)
                                    .prefix("camera pos epsilon: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_camera_rot_epsilon)
                                    .speed(0.0001)
                                    .range(0.0..=1.0)
                                    .prefix("camera rot epsilon: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_scene_pos_quantize)
                                    .speed(0.0001)
                                    .range(0.0..=1.0)
                                    .prefix("scene pos quantize: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_scene_rot_quantize)
                                    .speed(0.0001)
                                    .range(0.0..=1.0)
                                    .prefix("scene rot quantize: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_scene_scale_quantize)
                                    .speed(0.0001)
                                    .range(0.0..=1.0)
                                    .prefix("scene scale quantize: "),
                            );
                        });
                        ui.collapsing("quality", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_firefly_clamp)
                                    .speed(0.1)
                                    .range(0.0..=100.0)
                                    .prefix("firefly clamp: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_min_roughness)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("min roughness: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_normal_map_strength)
                                    .speed(0.05)
                                    .range(0.0..=2.0)
                                    .prefix("normal strength: "),
                            );
                        });
                        ui.collapsing("transparency", |ui| {
                            ui.add(
                                egui::Slider::new(&mut render_cfg.rt_transparency_max_skip, 1..=64)
                                    .text("max transparent skips"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_alpha_cutoff_min)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("alpha min: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_alpha_cutoff_scale)
                                    .speed(0.01)
                                    .range(0.0..=4.0)
                                    .prefix("alpha cutoff scale: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_alpha_cutoff_bias)
                                    .speed(0.01)
                                    .range(-1.0..=1.0)
                                    .prefix("alpha cutoff bias: "),
                            );
                        });
                        ui.collapsing("skinning", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_skinning_bounds_pad)
                                    .speed(0.01)
                                    .prefix("rt bounds pad: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_skinning_motion_scale)
                                    .speed(0.01)
                                    .prefix("rt motion scale: "),
                            );
                        });
                        ui.collapsing("ray bias", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_shadow_bias)
                                    .speed(0.0001)
                                    .range(0.0..=0.01)
                                    .prefix("shadow bias: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_ray_bias)
                                    .speed(0.0001)
                                    .range(0.0..=0.01)
                                    .prefix("ray bias: "),
                            );
                        });
                        ui.collapsing("performance", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_throughput_cutoff)
                                    .speed(0.001)
                                    .range(0.0..=1.0)
                                    .prefix("throughput cutoff: "),
                            );
                        });
                        ui.collapsing("texture arrays", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_texture_array_budget_bytes)
                                    .speed(1024.0 * 1024.0)
                                    .prefix("array budget bytes: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_texture_array_min_resolution,
                                )
                                .speed(1.0)
                                .prefix("min resolution: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_texture_array_max_resolution,
                                )
                                .speed(1.0)
                                .prefix("max resolution: "),
                            );
                        });
                        ui.collapsing("resolution scaling", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_resolution_scale)
                                    .speed(0.01)
                                    .prefix("resolution scale: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_ray_budget)
                                    .speed(1000.0)
                                    .prefix("ray budget: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_min_pixel_budget)
                                    .speed(64.0)
                                    .prefix("min pixel budget: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_complexity_base)
                                    .speed(256.0)
                                    .prefix("complexity base: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_complexity_exponent)
                                    .speed(0.05)
                                    .prefix("complexity exponent: "),
                            );
                        });
                        ui.collapsing("adaptive scaling", |ui| {
                            ui.checkbox(&mut render_cfg.rt_adaptive_scale, "enabled");
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_adaptive_scale_min)
                                    .speed(0.01)
                                    .prefix("min scale: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_adaptive_scale_max)
                                    .speed(0.01)
                                    .prefix("max scale: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_adaptive_scale_down)
                                    .speed(0.01)
                                    .prefix("scale down: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_adaptive_scale_up)
                                    .speed(0.01)
                                    .prefix("scale up: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_adaptive_scale_recovery_frames,
                                )
                                .speed(1.0)
                                .prefix("recovery frames: "),
                            );
                        });
                        ui.collapsing("acceleration structures", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_blas_leaf_size)
                                    .speed(1.0)
                                    .prefix("blas leaf size: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_tlas_leaf_size)
                                    .speed(1.0)
                                    .prefix("tlas leaf size: "),
                            );
                        });

                        ui.separator();
                        ui.heading("Reflections");
                        ui.checkbox(&mut render_cfg.rt_reflections, "use RT reflections");
                        ui.label("(RT > SSR) + DDGI specular");
                        ui.collapsing("rt reflection quality", |ui| {
                            ui.add(
                                egui::Slider::new(
                                    &mut render_cfg.rt_reflection_samples_per_frame,
                                    1..=64,
                                )
                                .text("samples per frame"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut render_cfg.rt_reflection_direct_light_samples,
                                    1..=16,
                                )
                                .text("direct light samples"),
                            );
                            ui.checkbox(
                                &mut render_cfg.rt_reflection_direct_lighting,
                                "direct lighting",
                            );
                            ui.checkbox(&mut render_cfg.rt_reflection_shadows, "shadows");
                        });
                        ui.collapsing("rt reflection accumulation", |ui| {
                            ui.checkbox(&mut render_cfg.rt_reflection_accumulation, "accumulation");
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_reflection_max_accumulation_frames,
                                )
                                .speed(1.0)
                                .prefix("max accumulation frames: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_reflection_history_weight)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("history weight: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_reflection_history_depth_threshold,
                                )
                                .speed(0.001)
                                .range(0.0..=1.0)
                                .prefix("history depth thresh: "),
                            );
                        });
                        ui.collapsing("rt reflection scaling", |ui| {
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_reflection_resolution_scale,
                                )
                                .speed(0.01)
                                .range(0.05..=1.0)
                                .prefix("resolution scale: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_reflection_ray_budget)
                                    .speed(1000.0)
                                    .prefix("ray budget: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_reflection_min_pixel_budget,
                                )
                                .speed(64.0)
                                .prefix("min pixel budget: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_reflection_complexity_base)
                                    .speed(256.0)
                                    .prefix("complexity base: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_reflection_complexity_exponent,
                                )
                                .speed(0.05)
                                .prefix("complexity exponent: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_reflection_interleave)
                                    .speed(1.0)
                                    .range(1..=8)
                                    .prefix("interleave: "),
                            );
                        });
                        ui.collapsing("rt reflection denoise", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.rt_reflection_denoise_radius)
                                    .speed(1.0)
                                    .range(0..=8)
                                    .prefix("radius: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_reflection_denoise_depth_sigma,
                                )
                                .speed(1.0)
                                .range(0.0..=200.0)
                                .prefix("depth sigma: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_reflection_denoise_normal_sigma,
                                )
                                .speed(1.0)
                                .range(0.0..=200.0)
                                .prefix("normal sigma: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.rt_reflection_denoise_color_sigma,
                                )
                                .speed(0.5)
                                .range(0.0..=100.0)
                                .prefix("color sigma: "),
                            );
                        });

                        ui.separator();
                        ui.heading("DDGI Resampling");
                        ui.checkbox(&mut render_cfg.ddgi_pass, "enable ddgi");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.ddgi_intensity)
                                .speed(0.05)
                                .range(0.0..=10.0)
                                .prefix("indirect scale: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.ddgi_hysteresis)
                                .speed(0.01)
                                .range(0.0..=0.99)
                                .prefix("hysteresis: "),
                        );
                        ui.collapsing("grid", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_probe_count_x)
                                    .speed(1.0)
                                    .range(1..=512)
                                    .prefix("count x: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_probe_count_y)
                                    .speed(1.0)
                                    .range(1..=512)
                                    .prefix("count y: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_probe_count_z)
                                    .speed(1.0)
                                    .range(1..=512)
                                    .prefix("count z: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_probe_spacing)
                                    .speed(0.1)
                                    .range(0.1..=1000.0)
                                    .prefix("spacing: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_probe_resolution)
                                    .speed(1.0)
                                    .range(1..=64)
                                    .prefix("probe resolution: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_probe_update_stride)
                                    .speed(1.0)
                                    .range(1..=64)
                                    .prefix("update stride: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_max_distance)
                                    .speed(0.5)
                                    .range(0.1..=5000.0)
                                    .prefix("max distance: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_normal_bias)
                                    .speed(0.01)
                                    .range(0.0..=5.0)
                                    .prefix("normal bias: "),
                            );
                        });
                        ui.collapsing("reservoir reuse", |ui| {
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.ddgi_reservoir_temporal_weight,
                                )
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .prefix("temporal weight: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_reservoir_spatial_weight)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("spatial weight: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_resample_spatial_samples)
                                    .speed(1.0)
                                    .range(0..=16)
                                    .prefix("spatial samples: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_resample_spatial_radius)
                                    .speed(1.0)
                                    .range(0..=16)
                                    .prefix("spatial radius: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_resample_reservoir_mix)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("reservoir mix: "),
                            );
                        });
                        ui.collapsing("resample quality", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_resample_diffuse_samples)
                                    .speed(1.0)
                                    .range(1..=64)
                                    .prefix("diffuse samples: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.ddgi_resample_specular_samples,
                                )
                                .speed(1.0)
                                .range(1..=64)
                                .prefix("specular samples: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.ddgi_resample_history_depth_threshold,
                                )
                                .speed(0.001)
                                .range(0.0..=1.0)
                                .prefix("history depth thresh: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.ddgi_resample_min_candidate_weight,
                                )
                                .speed(0.00001)
                                .range(0.0..=0.1)
                                .prefix("min candidate weight: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.ddgi_resample_specular_cone_angle,
                                )
                                .speed(0.01)
                                .range(0.0..=3.14159)
                                .prefix("spec cone angle (rad): "),
                            );
                        });
                        ui.collapsing("visibility", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_visibility_normal_bias)
                                    .speed(0.01)
                                    .range(0.0..=5.0)
                                    .prefix("normal bias: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_visibility_spacing_bias)
                                    .speed(0.001)
                                    .range(0.0..=1.0)
                                    .prefix("spacing bias: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_visibility_max_bias)
                                    .speed(0.01)
                                    .range(0.0..=2.0)
                                    .prefix("max bias: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_visibility_receiver_bias)
                                    .speed(0.01)
                                    .range(0.0..=2.0)
                                    .prefix("receiver bias: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.ddgi_visibility_variance_scale,
                                )
                                .speed(0.01)
                                .range(0.0..=5.0)
                                .prefix("variance scale: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_visibility_bleed_min)
                                    .speed(0.005)
                                    .range(0.0..=1.0)
                                    .prefix("bleed min: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_visibility_bleed_max)
                                    .speed(0.005)
                                    .range(0.0..=1.0)
                                    .prefix("bleed max: "),
                            );
                        });
                        ui.collapsing("reflections fallback", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_reflection_strength)
                                    .speed(0.05)
                                    .range(0.0..=2.0)
                                    .prefix("strength: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_reflection_fallback_mix)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("fallback mix: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_specular_scale)
                                    .speed(0.05)
                                    .range(0.0..=5.0)
                                    .prefix("specular scale: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_specular_confidence)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("specular confidence: "),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut render_cfg.ddgi_reflection_roughness_start,
                                )
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .prefix("roughness start: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut render_cfg.ddgi_reflection_roughness_end)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .prefix("roughness end: "),
                            );
                        });

                        ui.separator();
                        ui.heading("wgpu");
                        let mut selected_backend = wgpu_backend;
                        ComboBox::from_label("backend")
                            .selected_text(wgpu_backend.label())
                            .show_ui(ui, |ui| {
                                for backend in [
                                    WgpuBackend::Auto,
                                    WgpuBackend::Vulkan,
                                    WgpuBackend::Dx12,
                                    WgpuBackend::Metal,
                                    WgpuBackend::Gl,
                                ] {
                                    ui.selectable_value(
                                        &mut selected_backend,
                                        backend,
                                        backend.label(),
                                    );
                                }
                            });
                        if selected_backend != wgpu_backend {
                            wgpu_backend = selected_backend;
                            wgpu_settings_changed = true;
                        }

                        let mut selected_binding = binding_backend;
                        ComboBox::from_label("binding backend")
                            .selected_text(binding_backend.label())
                            .show_ui(ui, |ui| {
                                for backend in [
                                    BindingBackendChoice::Auto,
                                    BindingBackendChoice::BindlessModern,
                                    BindingBackendChoice::BindlessFallback,
                                    BindingBackendChoice::BindGroups,
                                ] {
                                    ui.selectable_value(
                                        &mut selected_binding,
                                        backend,
                                        backend.label(),
                                    );
                                }
                            });
                        if selected_binding != binding_backend {
                            binding_backend = selected_binding;
                            wgpu_settings_changed = true;
                        }

                        if ui
                            .checkbox(&mut wgpu_experimental, "enable experimental wgpu features")
                            .changed()
                        {
                            wgpu_settings_changed = true;
                        }

                        if ui.button("recreate device").clicked() {
                            request_recreate = true;
                        }
                        if wgpu_settings_changed {
                            ui.label("changes require device recreation");
                        }

                        ui.separator();
                        ui.heading("Occlusion Stability");
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.occlusion_stable_pos_epsilon)
                                .speed(0.0001)
                                .range(0.0..=10.0)
                                .prefix("position epsilon: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut render_cfg.occlusion_stable_rot_epsilon)
                                .speed(0.0001)
                                .range(0.0..=1.0)
                                .prefix("rotation epsilon: "),
                        );

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
                    }

                    runtime_cfg.0.wgpu_experimental_features = wgpu_experimental;
                    runtime_cfg.0.wgpu_backend = wgpu_backend;
                    runtime_cfg.0.binding_backend = binding_backend;
                    if request_recreate {
                        if let Some(sender) = sender.as_ref() {
                            if sender
                                .try_send(RenderMessage::Control(RenderControl::RecreateDevice {
                                    backend: wgpu_backend,
                                    binding_backend,
                                    allow_experimental_features: wgpu_experimental,
                                }))
                                .is_err()
                            {
                                warn!("failed to queue RecreateDevice control message");
                            }
                        } else {
                            warn!("recreate device requested but render sender unavailable");
                        }
                    }
                }

                if let Some(mut lod_tuning) = world.get_resource_mut::<BecsLodTuning>() {
                    ui.separator();
                    ui.heading("LOD tuning");
                    ui.collapsing("LOD tuning", |ui| {
                        let tuning = &mut lod_tuning.0;
                        ui.add(
                            egui::DragValue::new(&mut tuning.lod0_distance)
                                .speed(0.5)
                                .prefix("LOD0 distance: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut tuning.lod1_distance)
                                .speed(0.5)
                                .prefix("LOD1 distance: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut tuning.lod2_distance)
                                .speed(0.5)
                                .prefix("LOD2 distance: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut tuning.hysteresis)
                                .speed(0.01)
                                .range(0.0..=0.9)
                                .prefix("hysteresis: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut tuning.smoothing)
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .prefix("smoothing: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut tuning.min_change_frames)
                                .speed(1.0)
                                .prefix("min change frames: "),
                        );
                    });
                }
            }),
            window_spec("render config"),
        ));

        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                ui.heading("occlusion diagnostics");
                ui.separator();

                let stats = world.get_resource::<BecsRendererStats>();
                if let Some(stats) = stats {
                    let status = stats.0.occlusion_status.load(Ordering::Relaxed);
                    let status_label = match status {
                        OCCLUSION_STATUS_DISABLED => "disabled",
                        OCCLUSION_STATUS_NO_GBUFFER => "skipped (gbuffer off)",
                        OCCLUSION_STATUS_NO_INSTANCES => "skipped (no instances)",
                        OCCLUSION_STATUS_NO_HIZ => "skipped (no hi-z)",
                        OCCLUSION_STATUS_RAN => "running",
                        _ => "unknown",
                    };
                    let last_frame = stats.0.occlusion_last_frame.load(Ordering::Relaxed);
                    let instance_count = stats.0.occlusion_instance_count.load(Ordering::Relaxed);
                    let camera_stable =
                        stats.0.occlusion_camera_stable.load(Ordering::Relaxed) != 0;

                    ui.label(format!("status: {}", status_label));
                    if last_frame == u32::MAX {
                        ui.label("last run frame: never");
                    } else {
                        ui.label(format!("last run frame: {}", last_frame));
                    }
                    ui.label(format!("instances tested: {}", instance_count));
                    ui.label(format!("camera stable: {}", camera_stable));

                    match status {
                        OCCLUSION_STATUS_DISABLED => {
                            ui.label("hint: enable occlusion culling in render config");
                        }
                        OCCLUSION_STATUS_NO_GBUFFER => {
                            ui.label("hint: gbuffer must be enabled for hi-z");
                        }
                        OCCLUSION_STATUS_NO_HIZ => {
                            ui.label("hint: hi-z not ready yet; check debug graph + toggles");
                        }
                        _ => {}
                    }
                } else {
                    ui.label("renderer stats unavailable");
                }
            }),
            window_spec("occlusion culling"),
        ));

        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                ui.heading("gpu driven diagnostics");
                ui.separator();

                let stats = world.get_resource::<BecsRendererStats>();
                if let Some(stats) = stats {
                    let draw_count = stats.0.gpu_draw_count.load(Ordering::Relaxed);
                    let mesh_count = stats.0.gpu_mesh_count.load(Ordering::Relaxed);
                    let instance_capacity = stats.0.gpu_instance_capacity.load(Ordering::Relaxed);
                    let visible_capacity = stats.0.gpu_visible_capacity.load(Ordering::Relaxed);
                    let shadow_capacity = stats.0.gpu_shadow_capacity.load(Ordering::Relaxed);
                    let total_capacity = stats.0.gpu_total_capacity.load(Ordering::Relaxed);
                    let fallbacks = stats.0.gpu_fallbacks.load(Ordering::Relaxed);

                    ui.label(format!("draws: {}", draw_count));
                    ui.label(format!("meshes: {}", mesh_count));
                    ui.label(format!("instance capacity: {}", instance_capacity));
                    ui.label(format!("visible capacity: {}", visible_capacity));
                    ui.label(format!("shadow capacity: {}", shadow_capacity));
                    ui.label(format!("total capacity: {}", total_capacity));
                    ui.label(format!("fallback frames: {}", fallbacks));
                } else {
                    ui.label("renderer stats unavailable");
                }
            }),
            window_spec("gpu driven"),
        ));

        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                let sender = world
                    .get_resource::<BecsRenderSender>()
                    .map(|s| s.0.clone());
                let asset_server = {
                    #[cfg(target_arch = "wasm32")]
                    let asset_server = world.get_non_send_resource::<BecsAssetServer>();
                    #[cfg(not(target_arch = "wasm32"))]
                    let asset_server = world.get_resource::<BecsAssetServer>();
                    asset_server.map(|s| s.0.clone())
                };
                let stats = world.get_resource::<BecsRendererStats>();
                let mb_bytes: usize = 1024 * 1024;
                let mb_f = mb_bytes as f32;
                let mb_u64 = mb_bytes as u64;

                let (mut soft_mb, mut hard_mb, mut idle_frames) = stats
                    .map(|s| {
                        (
                            s.0.vram_soft_limit_bytes.load(Ordering::Relaxed) as f32 / mb_f,
                            s.0.vram_hard_limit_bytes.load(Ordering::Relaxed) as f32 / mb_f,
                            s.0.idle_frames.load(Ordering::Relaxed),
                        )
                    })
                    .unwrap_or((0.0, 0.0, 120));
                let kind_count = ResourceKind::Transient as usize + 1;
                let mut per_kind_soft_mb = vec![soft_mb; kind_count];
                let mut per_kind_hard_mb = vec![hard_mb; kind_count];
                if let Some(stats) = stats {
                    for idx in 0..kind_count {
                        let soft = stats.0.vram_soft_limit_per_kind[idx].load(Ordering::Relaxed)
                            as f32
                            / mb_f;
                        let hard = stats.0.vram_hard_limit_per_kind[idx].load(Ordering::Relaxed)
                            as f32
                            / mb_f;
                        if soft > 0.0 {
                            per_kind_soft_mb[idx] = soft;
                        }
                        if hard > 0.0 {
                            per_kind_hard_mb[idx] = hard;
                        }
                    }
                }

                ui.heading("render graph");
                ui.separator();

                let templates = graph_templates();
                let fallback_template = templates.first().expect("render graph templates missing");
                let mut active_name = world
                    .get_resource::<RenderGraphResource>()
                    .map(|res| res.0.name)
                    .unwrap_or(fallback_template.name);

                ui.heading("graph selection");
                if let Some(mut graph_res) = world.get_resource_mut::<RenderGraphResource>() {
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

                let active_template = template_for_graph(active_name).unwrap_or(fallback_template);

                ui.separator();
                ui.heading("passes");
                if let Some(mut runtime_cfg) = world.get_resource_mut::<BecsRuntimeConfig>() {
                    let render_cfg = &mut runtime_cfg.0.render_config;
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
                } else {
                    ui.label("runtime config unavailable");
                }

                ui.separator();
                ui.label("gpu budgets");
                let mut gpu_changed = false;
                gpu_changed |= ui
                    .add(
                        egui::DragValue::new(&mut soft_mb)
                            .speed(8.0)
                            .prefix("soft MiB: "),
                    )
                    .changed();
                gpu_changed |= ui
                    .add(
                        egui::DragValue::new(&mut hard_mb)
                            .speed(8.0)
                            .prefix("hard MiB: "),
                    )
                    .changed();
                gpu_changed |= ui
                    .add(
                        egui::DragValue::new(&mut idle_frames).prefix("idle frames before evict: "),
                    )
                    .changed();
                let kind_labels = [
                    (ResourceKind::Mesh, "mesh"),
                    (ResourceKind::Material, "material"),
                    (ResourceKind::Texture, "texture"),
                    (ResourceKind::TextureView, "texture view"),
                    (ResourceKind::Sampler, "sampler"),
                    (ResourceKind::Buffer, "buffer"),
                    (ResourceKind::External, "external"),
                    (ResourceKind::Transient, "transient"),
                ];
                ui.collapsing("per-kind budgets", |ui| {
                    for (kind, label) in kind_labels.iter() {
                        let idx = *kind as usize;
                        gpu_changed |= ui
                            .add(
                                egui::DragValue::new(&mut per_kind_soft_mb[idx])
                                    .speed(4.0)
                                    .prefix(format!("{label} soft MiB: ")),
                            )
                            .changed();
                        gpu_changed |= ui
                            .add(
                                egui::DragValue::new(&mut per_kind_hard_mb[idx])
                                    .speed(4.0)
                                    .prefix(format!("{label} hard MiB: ")),
                            )
                            .changed();
                    }
                });

                if (gpu_changed || ui.button("apply gpu budget").clicked()) && sender.is_some() {
                    let mut per_kind_soft_bytes = Vec::with_capacity(kind_count);
                    let mut per_kind_hard_bytes = Vec::with_capacity(kind_count);
                    for idx in 0..kind_count {
                        let soft = (per_kind_soft_mb[idx].max(0.0) as u64) * mb_u64;
                        let hard = (per_kind_hard_mb[idx].max(0.0) as u64) * mb_u64;
                        per_kind_soft_bytes.push(soft);
                        per_kind_hard_bytes.push(hard.max(soft));
                    }
                    let _ = sender.as_ref().unwrap().try_send(RenderMessage::Control(
                        RenderControl::SetGpuBudget {
                            soft_limit_bytes: (soft_mb.max(1.0) as u64) * mb_u64,
                            hard_limit_bytes: (hard_mb.max(soft_mb) as u64) * mb_u64,
                            idle_frames: Some(idle_frames),
                            per_kind_soft: Some(per_kind_soft_bytes),
                            per_kind_hard: Some(per_kind_hard_bytes),
                        },
                    ));
                }

                ui.separator();
                ui.label("asset budgets");
                if let Some(server_arc) = asset_server.as_ref() {
                    let mut server = server_arc.lock();
                    let (mesh_b, tex_b, mat_b) = server.budgets_bytes();
                    let mut scene_buf_mb = server.scene_buffer_budget_bytes() as f32 / mb_f;
                    let mut mesh_mb = mesh_b as f32 / mb_f;
                    let mut tex_mb = tex_b as f32 / mb_f;
                    let mut mat_mb = mat_b as f32 / mb_f;
                    let mut budget_changed = false;
                    budget_changed |= ui
                        .add(
                            egui::DragValue::new(&mut scene_buf_mb)
                                .speed(4.0)
                                .prefix("scene buffers MiB: "),
                        )
                        .changed();
                    budget_changed |= ui
                        .add(
                            egui::DragValue::new(&mut mesh_mb)
                                .speed(4.0)
                                .prefix("mesh MiB: "),
                        )
                        .changed();
                    budget_changed |= ui
                        .add(
                            egui::DragValue::new(&mut tex_mb)
                                .speed(8.0)
                                .prefix("texture MiB: "),
                        )
                        .changed();
                    budget_changed |= ui
                        .add(
                            egui::DragValue::new(&mut mat_mb)
                                .speed(2.0)
                                .prefix("material MiB: "),
                        )
                        .changed();

                    if budget_changed {
                        server.set_scene_buffer_budget_bytes(
                            (scene_buf_mb.max(0.0) as usize).saturating_mul(mb_bytes),
                        );
                        server.set_budget_bytes(
                            (mesh_mb.max(0.0) as usize).saturating_mul(mb_bytes),
                            (tex_mb.max(0.0) as usize).saturating_mul(mb_bytes),
                            (mat_mb.max(0.0) as usize).saturating_mul(mb_bytes),
                        );
                    }
                    let (mesh_u, tex_u, mat_u) = server.cache_usage_bytes();
                    let scene_u = server.scene_buffer_usage_bytes();
                    ui.label(format!(
                        "cache usage: mesh {:.1} MiB, tex {:.1} MiB, mat {:.1} MiB",
                        mesh_u as f32 / mb_f,
                        tex_u as f32 / mb_f,
                        mat_u as f32 / mb_f
                    ));
                    ui.label(format!("scene buffers: {:.1} MiB", scene_u as f32 / mb_f));

                    ui.separator();
                    ui.label("asset limits");

                    let (mut asset_creation_limit_per_frame, mut streaming_upload_limit_per_frame) =
                        server.limits();
                    let mut limit_changed = false;
                    limit_changed |= ui
                        .add(
                            egui::DragValue::new(&mut asset_creation_limit_per_frame)
                                .speed(0.1)
                                .prefix("asset creation limit: "),
                        )
                        .changed();
                    limit_changed |= ui
                        .add(
                            egui::DragValue::new(&mut streaming_upload_limit_per_frame)
                                .speed(0.1)
                                .prefix("streaming upload per-frame limit: "),
                        )
                        .changed();

                    if limit_changed {
                        server.set_limits(
                            asset_creation_limit_per_frame,
                            streaming_upload_limit_per_frame,
                        );
                    }

                    ui.separator();
                    ui.label("streaming backlog");
                    let mut backlog_limit = server.streaming_backlog_limit();
                    if ui
                        .add(
                            egui::DragValue::new(&mut backlog_limit)
                                .speed(32.0)
                                .prefix("backlog limit: "),
                        )
                        .changed()
                    {
                        server.set_streaming_backlog_limit(backlog_limit);
                    }

                    ui.separator();
                    ui.label("cache eviction");
                    let mut cache_idle_ms = server.cache_idle_ms();
                    if ui
                        .add(
                            egui::DragValue::new(&mut cache_idle_ms)
                                .speed(10.0)
                                .prefix("cache idle ms: "),
                        )
                        .changed()
                    {
                        server.set_cache_idle_ms(cache_idle_ms);
                    }
                    let mut cache_eviction_limit = server.cache_eviction_limit();
                    if ui
                        .add(
                            egui::DragValue::new(&mut cache_eviction_limit)
                                .speed(4.0)
                                .prefix("cache evict limit: "),
                        )
                        .changed()
                    {
                        server.set_cache_eviction_limit(cache_eviction_limit);
                    }

                    ui.separator();
                    ui.label("asset streaming tuning");
                    let mut asset_tuning = server.asset_streaming_tuning();
                    let mut asset_tuning_changed = false;
                    if ui.button("reset asset streaming tuning").clicked() {
                        asset_tuning = Default::default();
                        asset_tuning_changed = true;
                    }
                    ui.collapsing("low-res textures", |ui| {
                        asset_tuning_changed |= ui
                            .add(
                                egui::DragValue::new(&mut asset_tuning.low_res_max_dim)
                                    .speed(1.0)
                                    .prefix("low-res max dim: "),
                            )
                            .changed();
                    });
                    ui.collapsing("mesh LOD generation", |ui| {
                        asset_tuning_changed |= ui
                            .add(
                                egui::DragValue::new(&mut asset_tuning.lod_safe_vertex_limit)
                                    .speed(10_000.0)
                                    .prefix("safe vertex limit: "),
                            )
                            .changed();
                        asset_tuning_changed |= ui
                            .add(
                                egui::DragValue::new(&mut asset_tuning.lod_safe_index_limit)
                                    .speed(10_000.0)
                                    .prefix("safe index limit: "),
                            )
                            .changed();
                        asset_tuning_changed |= ui
                            .add(
                                egui::DragValue::new(&mut asset_tuning.lod_simplification_error)
                                    .speed(0.001)
                                    .prefix("simplification error: "),
                            )
                            .changed();
                        ui.separator();
                        asset_tuning_changed |= ui
                            .add(
                                egui::DragValue::new(&mut asset_tuning.lod_targets[0])
                                    .speed(0.01)
                                    .prefix("lod target 1: "),
                            )
                            .changed();
                        asset_tuning_changed |= ui
                            .add(
                                egui::DragValue::new(&mut asset_tuning.lod_targets[1])
                                    .speed(0.01)
                                    .prefix("lod target 2: "),
                            )
                            .changed();
                        asset_tuning_changed |= ui
                            .add(
                                egui::DragValue::new(&mut asset_tuning.lod_targets[2])
                                    .speed(0.01)
                                    .prefix("lod target 3: "),
                            )
                            .changed();
                    });
                    if asset_tuning_changed {
                        server.set_asset_streaming_tuning(asset_tuning);
                    }

                    ui.separator();
                    ui.label("streaming + eviction tuning");
                    if let Some(mut tuning_res) = world.get_resource_mut::<BecsStreamingTuning>() {
                        let tuning = &mut tuning_res.0;
                        let mut tuning_changed = false;

                        if ui.button("reset streaming tuning").clicked() {
                            *tuning = Default::default();
                            tuning_changed = true;
                        }

                        ui.collapsing("caps", |ui| {
                            ui.label("none");
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_none.global)
                                        .speed(128.0)
                                        .prefix("global: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_none.mesh)
                                        .speed(128.0)
                                        .prefix("mesh: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_none.material)
                                        .speed(128.0)
                                        .prefix("material: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_none.texture)
                                        .speed(128.0)
                                        .prefix("texture: "),
                                )
                                .changed();

                            ui.separator();
                            ui.label("soft");
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_soft.global)
                                        .speed(128.0)
                                        .prefix("global: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_soft.mesh)
                                        .speed(128.0)
                                        .prefix("mesh: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_soft.material)
                                        .speed(128.0)
                                        .prefix("material: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_soft.texture)
                                        .speed(128.0)
                                        .prefix("texture: "),
                                )
                                .changed();

                            ui.separator();
                            ui.label("hard");
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_hard.global)
                                        .speed(128.0)
                                        .prefix("global: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_hard.mesh)
                                        .speed(128.0)
                                        .prefix("mesh: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_hard.material)
                                        .speed(128.0)
                                        .prefix("material: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.caps_hard.texture)
                                        .speed(128.0)
                                        .prefix("texture: "),
                                )
                                .changed();
                        });

                        ui.collapsing("priority floors + thresholds", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_size_bias)
                                        .speed(0.05)
                                        .prefix("priority size bias: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_distance_bias)
                                        .speed(0.05)
                                        .prefix("priority distance bias: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_lod_bias)
                                        .speed(0.05)
                                        .prefix("priority lod bias: "),
                                )
                                .changed();
                            ui.separator();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_floor_none)
                                        .speed(0.001)
                                        .prefix("floor none: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_floor_soft)
                                        .speed(0.001)
                                        .prefix("floor soft: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_floor_hard)
                                        .speed(0.001)
                                        .prefix("floor hard: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_near)
                                        .speed(0.01)
                                        .prefix("priority near: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_critical)
                                        .speed(0.01)
                                        .prefix("priority critical: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.upgrade_priority_soft)
                                        .speed(0.01)
                                        .prefix("upgrade priority soft: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.low_res_priority_soft)
                                        .speed(0.01)
                                        .prefix("low-res priority soft: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.low_res_priority_hard)
                                        .speed(0.01)
                                        .prefix("low-res priority hard: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.evict_retry_priority)
                                        .speed(0.01)
                                        .prefix("evict retry priority: "),
                                )
                                .changed();
                        });

                        ui.collapsing("sprite streaming", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning.sprite_screen_priority_multiplier,
                                    )
                                    .speed(0.05)
                                    .prefix("screen priority multiplier: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning.sprite_screen_distance_bias_min,
                                    )
                                    .speed(0.01)
                                    .prefix("screen distance bias min: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning.sprite_sequence_prefetch_frames,
                                    )
                                    .speed(1.0)
                                    .prefix("sequence prefetch frames: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning.sprite_sequence_prefetch_priority_scale,
                                    )
                                    .speed(0.01)
                                    .prefix("sequence prefetch scale: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning
                                            .sprite_sequence_pingpong_prefetch_priority_scale,
                                    )
                                    .speed(0.01)
                                    .prefix("pingpong prefetch scale: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.sprite_texture_priority_scale)
                                        .speed(0.01)
                                        .prefix("base texture scale: "),
                                )
                                .changed();
                        });

                        ui.collapsing("timing", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.pressure_release_frames)
                                        .speed(1.0)
                                        .prefix("pressure release frames: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.soft_upgrade_delay_frames)
                                        .speed(1.0)
                                        .prefix("soft upgrade delay: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.upgrade_cooldown_frames)
                                        .speed(1.0)
                                        .prefix("upgrade cooldown: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.inflight_cooldown_frames)
                                        .speed(1.0)
                                        .prefix("inflight cooldown: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.recent_evict_frames)
                                        .speed(1.0)
                                        .prefix("recent evict frames: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.evict_retry_frames)
                                        .speed(1.0)
                                        .prefix("evict retry frames: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.evict_soft_grace_frames)
                                        .speed(1.0)
                                        .prefix("evict soft grace: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.evict_hard_grace_frames)
                                        .speed(1.0)
                                        .prefix("evict hard grace: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.hard_idle_frames_before_evict)
                                        .speed(1.0)
                                        .prefix("hard idle frames: "),
                                )
                                .changed();
                        });

                        ui.collapsing("boosts + prediction", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.prediction_frames)
                                        .speed(0.5)
                                        .prefix("prediction frames: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.prediction_motion_epsilon)
                                        .speed(1.0e-6)
                                        .prefix("prediction motion epsilon: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.prediction_distance_threshold)
                                        .speed(0.01)
                                        .prefix("prediction distance margin: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.resident_priority_boost)
                                        .speed(0.01)
                                        .prefix("resident boost: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.inflight_priority_boost)
                                        .speed(0.01)
                                        .prefix("inflight boost: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.shadow_priority_boost)
                                        .speed(0.01)
                                        .prefix("shadow boost: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.priority_bump_factor)
                                        .speed(0.01)
                                        .prefix("priority bump factor: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.recent_evict_penalty)
                                        .speed(0.01)
                                        .prefix("recent evict penalty: "),
                                )
                                .changed();
                        });

                        ui.collapsing("lod bias + hard lod", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.lod_bias_soft)
                                        .speed(1.0)
                                        .prefix("lod bias soft: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.lod_bias_hard)
                                        .speed(1.0)
                                        .prefix("lod bias hard: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(egui::Checkbox::new(
                                    &mut tuning.force_lowest_lod_hard,
                                    "force lowest LOD on hard pressure",
                                ))
                                .changed();
                            tuning_changed |= ui
                                .add(egui::Checkbox::new(
                                    &mut tuning.force_low_res_hard,
                                    "force low-res textures on hard pressure",
                                ))
                                .changed();
                        });

                        ui.collapsing("pool eviction limits", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.evict_unplanned_idle_frames)
                                        .speed(1.0)
                                        .prefix("unplanned idle frames: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.pool_idle_frames_before_evict)
                                        .speed(1.0)
                                        .prefix("idle frames before evict: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning.pool_streaming_min_residency_frames,
                                    )
                                    .speed(1.0)
                                    .prefix("min residency frames: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.pool_max_evictions_per_tick)
                                        .speed(1.0)
                                        .prefix("max evictions per tick: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.pool_eviction_scan_budget)
                                        .speed(64.0)
                                        .prefix("eviction scan budget: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.pool_eviction_purge_budget)
                                        .speed(8.0)
                                        .prefix("eviction purge budget: "),
                                )
                                .changed();
                        });

                        ui.collapsing("asset id remap", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.asset_map_initial_capacity)
                                        .speed(128.0)
                                        .prefix("initial capacity: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.asset_map_max_load_factor)
                                        .speed(0.01)
                                        .prefix("max load factor: "),
                                )
                                .changed();
                        });

                        ui.collapsing("transient heap", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning.transient_heap_initial_capacity,
                                    )
                                    .speed(64.0)
                                    .prefix("initial capacity: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning.transient_heap_max_load_factor,
                                    )
                                    .speed(0.01)
                                    .prefix("max load factor: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut tuning.transient_heap_max_free_per_desc,
                                    )
                                    .speed(4.0)
                                    .prefix("max free per desc: "),
                                )
                                .changed();
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.transient_heap_max_total_free)
                                        .speed(64.0)
                                        .prefix("max total free: "),
                                )
                                .changed();
                        });

                        ui.collapsing("render graph execution", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.graph_encoder_batch_size)
                                        .speed(1.0)
                                        .prefix("encoder batch size: "),
                                )
                                .changed();
                            ui.label("0 = one encoder per level");
                        });

                        ui.collapsing("evict protection", |ui| {
                            tuning_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut tuning.evict_soft_protect_priority)
                                        .speed(0.01)
                                        .prefix("soft protect priority: "),
                                )
                                .changed();
                        });

                        if (tuning_changed || ui.button("apply streaming tuning").clicked())
                            && sender.is_some()
                        {
                            let _ = sender.as_ref().unwrap().try_send(RenderMessage::Control(
                                RenderControl::SetStreamingTuning(*tuning),
                            ));
                        }
                    }

                    ui.separator();

                    ui.horizontal(|ui| {
                        if ui.button("reupload cached assets").clicked() {
                            server.reupload_cached_assets();
                        }
                        if ui.button("evict gpu + restream").clicked() {
                            if let Some(s) = sender.as_ref() {
                                let _ =
                                    s.try_send(RenderMessage::Control(RenderControl::EvictAll {
                                        restream_assets: true,
                                    }));
                            }
                            server.reupload_cached_assets();
                        }
                        if ui.button("evict gpu only").clicked() {
                            if let Some(s) = sender.as_ref() {
                                let _ =
                                    s.try_send(RenderMessage::Control(RenderControl::EvictAll {
                                        restream_assets: false,
                                    }));
                            }
                        }
                    });
                } else {
                    ui.label("asset server unavailable");
                }
            }),
            window_spec("render graph"),
        ));

        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc| {
                ui.heading("render graph passes");
                ui.separator();

                let sender = world
                    .get_resource::<BecsRenderSender>()
                    .map(|s| s.0.clone());
                let (pass_timings, profiling_enabled) = world
                    .get_resource::<BecsRendererStats>()
                    .map(|stats| {
                        (
                            stats.0.pass_timings.read().clone(),
                            stats.0.profiling_enabled.load(Ordering::Relaxed),
                        )
                    })
                    .unwrap_or_default();

                let (pass_filter, sort_passes_by_time, show_disabled_passes) = world
                    .resource_scope::<EguiResource, _>(|_, mut egui_res| {
                        let state = &mut egui_res.render_graph_passes_state;

                        if pass_timings.is_empty() {
                            ui.label("render pass timings unavailable");
                        } else {
                            if !profiling_enabled {
                                ui.label("profiling disabled; timings may be stale");
                            }

                            ui.horizontal(|ui| {
                                ui.label("filter:");
                                ui.add(
                                    egui::TextEdit::singleline(&mut state.filter)
                                        .id_source("render_graph_pass_filter"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(egui::Checkbox::new(
                                    &mut state.sort_by_time,
                                    "sort by last duration",
                                ));
                                ui.add(egui::Checkbox::new(
                                    &mut state.show_disabled,
                                    "show disabled",
                                ));
                            });
                        }

                        (
                            state.filter.clone(),
                            state.sort_by_time,
                            state.show_disabled,
                        )
                    });

                let filter_lower = pass_filter.trim().to_lowercase();
                let mut rows = pass_timings;
                if !filter_lower.is_empty() {
                    rows.retain(|pass| pass.name.to_lowercase().contains(&filter_lower));
                }
                if sort_passes_by_time {
                    rows.sort_by(|a, b| {
                        b.duration_us
                            .cmp(&a.duration_us)
                            .then_with(|| a.order.cmp(&b.order))
                    });
                } else {
                    rows.sort_by(|a, b| a.order.cmp(&b.order));
                }

                if !rows.is_empty() {
                    ui.horizontal(|ui| {
                        let Some(sender) = sender.as_ref() else {
                            ui.label("render sender unavailable");
                            return;
                        };
                        if ui.button("enable all").clicked() {
                            for pass in rows.iter() {
                                if !pass.enabled {
                                    let _ = sender.try_send(RenderMessage::Control(
                                        RenderControl::SetPassEnabled {
                                            pass: pass.name.clone(),
                                            enabled: true,
                                        },
                                    ));
                                }
                            }
                        }
                        if ui.button("disable all").clicked() {
                            for pass in rows.iter() {
                                if pass.enabled {
                                    let _ = sender.try_send(RenderMessage::Control(
                                        RenderControl::SetPassEnabled {
                                            pass: pass.name.clone(),
                                            enabled: false,
                                        },
                                    ));
                                }
                            }
                        }
                    });

                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            egui::Grid::new("render_pass_list")
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.label("order");
                                    ui.label("pass");
                                    ui.label("enabled");
                                    ui.label("last ms");
                                    ui.end_row();

                                    for pass in rows.iter() {
                                        if !show_disabled_passes && !pass.enabled {
                                            continue;
                                        }
                                        ui.label(pass.order.to_string());
                                        ui.label(&pass.name);
                                        let mut enabled = pass.enabled;
                                        if ui.checkbox(&mut enabled, "").changed() {
                                            if let Some(sender) = sender.as_ref() {
                                                let _ = sender.try_send(RenderMessage::Control(
                                                    RenderControl::SetPassEnabled {
                                                        pass: pass.name.clone(),
                                                        enabled,
                                                    },
                                                ));
                                            }
                                        }
                                        ui.label(format!("{:.3}", micros_to_ms(pass.duration_us)));
                                        ui.end_row();
                                    }
                                });
                        });
                }
            }),
            window_spec("render graph passes"),
        ));

        egui_res.windows.push((
            Box::new(move |ui, world, _input_arc: &Arc<RwLock<InputManager>>| {
                let mut camera_query =
                    world.query::<(&mut Transform, &mut Camera, &ActiveCamera)>();

                for (mut transform, mut camera, _) in camera_query.iter_mut(world) {
                    let transform = &mut *transform;
                    let camera = &mut *camera;

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
                }

                let mut light_query = world.query::<(&mut Transform, &mut Light)>();

                for (mut transform, mut light) in light_query.iter_mut(world) {
                    let transform = &mut *transform;
                    let light = &mut *light;

                    if light.light_type != LightType::Directional {
                        continue;
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
                }

                if let Some(mut scene_tuning) = world.get_resource_mut::<BecsSceneTuning>() {
                    ui.collapsing("scene tuning", |ui| {
                        ui.add(
                            egui::DragValue::new(&mut scene_tuning.0.transform_epsilon)
                                .speed(0.0001)
                                .range(0.0..=1.0)
                                .prefix("transform epsilon: "),
                        );
                    });
                    ui.separator();
                }

                let mut physics_resource = world
                    .get_resource_mut::<PhysicsResource>()
                    .expect("PhysicsResource resource not found");

                ui.collapsing("physics", |ui| {
                    if ui.button("default").clicked() {
                        let default_res = PhysicsResource::default();
                        physics_resource.gravity = default_res.gravity;
                        physics_resource.integration_parameters =
                            default_res.integration_parameters;
                    }

                    ui.separator();

                    ui.checkbox(&mut physics_resource.running, "running");

                    ui.separator();

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

                    ui.add(
                        egui::DragValue::new(
                            &mut physics_resource.integration_parameters.max_ccd_substeps,
                        )
                        .prefix("max_ccd_substeps: "),
                    );
                    ui.add(
                        egui::DragValue::new(
                            &mut physics_resource
                                .integration_parameters
                                .num_internal_pgs_iterations,
                        )
                        .prefix("num_internal_pgs_iterations: "),
                    );
                    ui.add(
                        egui::DragValue::new(
                            &mut physics_resource
                                .integration_parameters
                                .num_internal_stabilization_iterations,
                        )
                        .prefix("num_internal_stabilization_iterations: "),
                    );
                    ui.add(
                        egui::DragValue::new(
                            &mut physics_resource
                                .integration_parameters
                                .num_solver_iterations,
                        )
                        .prefix("num_solver_iterations: "),
                    );
                    ui.add(
                        egui::DragValue::new(
                            &mut physics_resource
                                .integration_parameters
                                .contact_softness
                                .damping_ratio,
                        )
                        .prefix("contact_damping_ratio: "),
                    );
                    ui.add(
                        egui::DragValue::new(
                            &mut physics_resource
                                .integration_parameters
                                .contact_softness
                                .natural_frequency,
                        )
                        .prefix("contact_natural_frequency: "),
                    );
                    ui.add(
                        egui::DragValue::new(
                            &mut physics_resource
                                .integration_parameters
                                .normalized_allowed_linear_error,
                        )
                        .prefix("normalized_allowed_linear_error: "),
                    );
                });
            }),
            window_spec("scene"),
        ));
    }
}

fn shader_constants_ui(ui: &mut egui::Ui, c: &mut ShaderConstants) {
    ui.collapsing("sky", |ui| {
        ui.add(egui::Slider::new(&mut c.planet_radius, 0.0..=9999e3).text("planet radius"));
        ui.add(egui::Slider::new(&mut c.atmosphere_radius, 0.0..=9999e3).text("atmosphere radius"));
        ui.add(egui::Slider::new(&mut c.sky_light_samples, 0..=64).text("light samples"));
        ui.separator();

        ui.collapsing("scattering", |ui| {
            ui.horizontal(|ui| {
                ui.label("rayleigh coeff");
                ui.add(egui::DragValue::new(&mut c.rayleigh_scattering_coeff[0]).speed(1e-7));
                ui.add(egui::DragValue::new(&mut c.rayleigh_scattering_coeff[1]).speed(1e-7));
                ui.add(egui::DragValue::new(&mut c.rayleigh_scattering_coeff[2]).speed(1e-7));
            });
            ui.add(
                egui::DragValue::new(&mut c.rayleigh_scale_height)
                    .speed(10.0)
                    .range(0.0..=200_000.0)
                    .prefix("rayleigh height: "),
            );
            ui.add(
                egui::DragValue::new(&mut c.mie_scattering_coeff)
                    .speed(1e-7)
                    .range(0.0..=1.0e-3)
                    .prefix("mie scatter: "),
            );
            ui.add(
                egui::DragValue::new(&mut c.mie_absorption_coeff)
                    .speed(1e-7)
                    .range(0.0..=1.0e-3)
                    .prefix("mie absorb: "),
            );
            ui.add(
                egui::DragValue::new(&mut c.mie_scale_height)
                    .speed(10.0)
                    .range(0.0..=200_000.0)
                    .prefix("mie height: "),
            );
            ui.add(
                egui::DragValue::new(&mut c.mie_preferred_scattering_dir)
                    .speed(0.01)
                    .range(0.0..=0.99)
                    .prefix("mie dir: "),
            );
        });

        ui.collapsing("ozone", |ui| {
            ui.horizontal(|ui| {
                ui.label("ozone absorb");
                ui.add(egui::DragValue::new(&mut c.ozone_absorption_coeff[0]).speed(1e-7));
                ui.add(egui::DragValue::new(&mut c.ozone_absorption_coeff[1]).speed(1e-7));
                ui.add(egui::DragValue::new(&mut c.ozone_absorption_coeff[2]).speed(1e-7));
            });
            ui.add(
                egui::DragValue::new(&mut c.ozone_center_height)
                    .speed(10.0)
                    .range(0.0..=200_000.0)
                    .prefix("ozone center: "),
            );
            ui.add(
                egui::DragValue::new(&mut c.ozone_falloff)
                    .speed(10.0)
                    .range(0.0..=200_000.0)
                    .prefix("ozone falloff: "),
            );
        });

        ui.collapsing("sun & ground", |ui| {
            ui.horizontal(|ui| {
                ui.label("night ambient");
                ui.add(egui::DragValue::new(&mut c.night_ambient_color[0]).speed(0.0001));
                ui.add(egui::DragValue::new(&mut c.night_ambient_color[1]).speed(0.0001));
                ui.add(egui::DragValue::new(&mut c.night_ambient_color[2]).speed(0.0001));
            });
            ui.add(
                egui::DragValue::new(&mut c.sun_angular_radius_cos)
                    .speed(0.00001)
                    .range(0.99..=1.0)
                    .prefix("sun cos: "),
            );
            ui.horizontal(|ui| {
                ui.label("ground albedo");
                ui.add(egui::DragValue::new(&mut c.sky_ground_albedo[0]).speed(0.01));
                ui.add(egui::DragValue::new(&mut c.sky_ground_albedo[1]).speed(0.01));
                ui.add(egui::DragValue::new(&mut c.sky_ground_albedo[2]).speed(0.01));
            });
            ui.add(
                egui::DragValue::new(&mut c.sky_ground_brightness)
                    .speed(0.01)
                    .range(0.0..=10.0)
                    .prefix("ground brightness: "),
            );
        });
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
        ui.add(
            egui::DragValue::new(&mut c.ssr_jitter_strength)
                .speed(0.01)
                .range(0.0..=1.0)
                .prefix("Jitter Strength: "),
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
