use bevy_ecs::prelude::*;
use egui::collapsing_header::CollapsingState;
use egui::{Context, Frame, Order, Pos2, Rect, Shadow, TextStyle, Vec2};
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
    pub window_rects: HashMap<String, Rect>,
    pub window_rect_overrides: HashMap<String, Rect>,
    pub window_collapsed: HashMap<String, bool>,
    pub window_collapsed_overrides: HashMap<String, bool>,
    pub window_order_overrides: HashMap<String, Order>,
    pub window_dragging: HashSet<String>,
    pub last_screen_rect: Option<Rect>,
    pub snap_enabled: bool,
    pub snap_distance: f32,
    pub suppress_snap: bool,
    pub layout_active: bool,
    pub layout_allow_move: bool,
    pub layout_force_positions: bool,
    pub layout_resizing_window: Option<String>,
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

fn content_size_for_outer_rect(
    ctx: &Context,
    frame: &Frame,
    collapsed: bool,
    outer_rect: Rect,
) -> Vec2 {
    let style = ctx.style();
    let font_id = TextStyle::Heading.resolve(style.as_ref());
    let title_height = ctx
        .fonts_mut(|fonts| fonts.row_height(&font_id))
        .max(style.spacing.interact_size.y);
    let title_bar_inner_height = title_height + frame.inner_margin.sum().y;
    let title_content_spacing = if collapsed { 0.0 } else { frame.stroke.width };
    let margins =
        frame.total_margin().sum() + Vec2::new(0.0, title_bar_inner_height + title_content_spacing);
    let mut size = outer_rect.size() - margins;
    if size.x < 1.0 {
        size.x = 1.0;
    }
    if size.y < 1.0 {
        size.y = 1.0;
    }
    size
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

    if pixels_per_point > 0.0 {
        let size = Vec2::new(
            window_size.x as f32 / pixels_per_point,
            window_size.y as f32 / pixels_per_point,
        );
        raw_input.screen_rect = Some(Rect::from_min_size(Pos2::new(0.0, 0.0), size));
    }

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
    if pixels_per_point > 0.0 {
        ctx.set_pixels_per_point(pixels_per_point);
    }

    let mut last_screen_rect: Option<Rect> = None;
    let full_output = ctx.run(raw_input, |ctx| {
        let mut egui_res = world
            .get_resource_mut::<EguiResource>()
            .expect("EguiResource resource not found");

        let windows = std::mem::take(&mut egui_res.windows);
        let mut close_actions = std::mem::take(&mut egui_res.close_actions);
        let mut window_positions = std::mem::take(&mut egui_res.window_positions);
        let window_rects_prev = std::mem::take(&mut egui_res.window_rects);
        let window_collapsed_prev = std::mem::take(&mut egui_res.window_collapsed);
        let window_rect_overrides = std::mem::take(&mut egui_res.window_rect_overrides);
        let window_collapsed_overrides = std::mem::take(&mut egui_res.window_collapsed_overrides);
        let window_order_overrides = std::mem::take(&mut egui_res.window_order_overrides);
        let snap_enabled = egui_res.snap_enabled;
        let snap_distance = egui_res.snap_distance;
        let suppress_snap = egui_res.suppress_snap;
        let layout_active = egui_res.layout_active;
        let layout_force_positions = egui_res.layout_force_positions;
        egui_res.suppress_snap = false;

        drop(egui_res);

        let screen_rect = ctx.viewport_rect();
        last_screen_rect = Some(screen_rect);
        let mut window_rects = HashMap::new();
        let mut window_collapsed = HashMap::new();
        let layout_frame = if layout_active {
            let mut frame = Frame::window(&ctx.style());
            frame.shadow = Shadow::NONE;
            Some(frame)
        } else {
            None
        };

        for (mut elements, spec) in windows {
            let window_id = egui::Id::new(spec.id.clone());
            let mut open = true;
            let has_close = close_actions.get_mut(&spec.id).is_some();
            let is_layout_window = layout_active && window_rect_overrides.contains_key(&spec.id);

            let collapsed = window_collapsed_overrides
                .get(&spec.id)
                .copied()
                .or_else(|| window_collapsed_prev.get(&spec.id).copied())
                .unwrap_or(false);
            if window_collapsed_overrides.contains_key(&spec.id) {
                let mut collapsing = CollapsingState::load_with_default_open(
                    ctx,
                    window_id.with("collapsing"),
                    !collapsed,
                );
                collapsing.set_open(!collapsed);
                collapsing.store(ctx);
            }

            let mut window = egui::Window::new(spec.title.clone())
                .id(window_id)
                .constrain_to(screen_rect);

            if let Some(order) = window_order_overrides.get(&spec.id) {
                window = window.order(*order);
            }

            if is_layout_window {
                window = window.movable(false).resizable(false);
            }

            if let Some(frame) = layout_frame {
                window = window.frame(frame);
            }

            if let Some(rect) = window_rect_overrides.get(&spec.id) {
                let frame = layout_frame.unwrap_or_else(|| Frame::window(&ctx.style()));
                let content_size = content_size_for_outer_rect(&ctx, &frame, collapsed, *rect);
                window = window.current_pos(rect.min).fixed_size(content_size);
                if layout_force_positions {
                    window_positions.insert(spec.id.clone(), rect.min);
                }
            } else if let Some(pos) = window_positions.get(&spec.id).copied() {
                window = window.current_pos(pos);
            }

            if has_close {
                window = window.open(&mut open);
            }

            if let Some(inner) = window.show(ctx, |ui| {
                elements(ui, world, &input_arc);
            }) {
                window_collapsed.insert(spec.id.clone(), inner.inner.is_none());
                window_rects.insert(spec.id.clone(), inner.response.rect);
                if let Some(egui_res) = world.get_resource::<EguiResource>() {
                    if !egui_res.window_dragging.contains(&spec.id) {
                        window_positions.insert(spec.id.clone(), inner.response.rect.min);
                    }
                }
            }

            if has_close && !open {
                if let Some(on_close) = close_actions.get_mut(&spec.id) {
                    on_close(world);
                }
            }
        }

        if snap_enabled && !suppress_snap && snap_distance > 0.0 {
            let should_snap = ctx.input(|input| input.pointer.any_released());
            if should_snap {
                if let Some((window_id, rect)) =
                    find_moved_window(&window_rects_prev, &window_rects)
                {
                    let snapped = snap_window_rect(
                        &window_id,
                        rect,
                        &window_rects,
                        screen_rect,
                        snap_distance,
                    );
                    if snapped.min != rect.min {
                        window_positions.insert(window_id.clone(), snapped.min);
                        window_rects.insert(window_id, snapped);
                    }
                }
            }
        }

        if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
            egui_res.window_positions = window_positions;
            egui_res.window_rects = window_rects;
            egui_res.window_collapsed = window_collapsed;
            egui_res.last_screen_rect = last_screen_rect;
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

fn find_moved_window(
    previous: &HashMap<String, Rect>,
    current: &HashMap<String, Rect>,
) -> Option<(String, Rect)> {
    let mut best: Option<(String, Rect, f32)> = None;
    for (id, rect) in current {
        let Some(prev) = previous.get(id) else {
            continue;
        };
        let delta = (rect.min - prev.min).length();
        if delta <= 0.5 {
            continue;
        }
        let replace = match best {
            Some((_, _, best_delta)) => delta > best_delta,
            None => true,
        };
        if replace {
            best = Some((id.clone(), *rect, delta));
        }
    }
    best.map(|(id, rect, _)| (id, rect))
}

fn snap_window_rect(
    moving_id: &str,
    rect: Rect,
    windows: &HashMap<String, Rect>,
    screen_rect: Rect,
    distance: f32,
) -> Rect {
    if distance <= 0.0 {
        return rect;
    }

    let mut best_x = rect.min.x;
    let mut best_y = rect.min.y;
    let mut best_dx = distance + 1.0;
    let mut best_dy = distance + 1.0;

    let width = rect.width();
    let height = rect.height();

    let mut consider_x = |candidate: f32, delta: f32| {
        if delta <= distance && delta < best_dx {
            best_dx = delta;
            best_x = candidate;
        }
    };

    let mut consider_y = |candidate: f32, delta: f32| {
        if delta <= distance && delta < best_dy {
            best_dy = delta;
            best_y = candidate;
        }
    };

    consider_x(screen_rect.min.x, (rect.min.x - screen_rect.min.x).abs());
    consider_x(
        screen_rect.max.x - width,
        (rect.max.x - screen_rect.max.x).abs(),
    );
    consider_y(screen_rect.min.y, (rect.min.y - screen_rect.min.y).abs());
    consider_y(
        screen_rect.max.y - height,
        (rect.max.y - screen_rect.max.y).abs(),
    );

    for (id, other) in windows {
        if id == moving_id {
            continue;
        }

        if ranges_overlap(rect.min.y, rect.max.y, other.min.y, other.max.y) {
            consider_x(other.min.x, (rect.min.x - other.min.x).abs());
            consider_x(other.max.x - width, (rect.max.x - other.max.x).abs());
            consider_x(other.min.x - width, (rect.max.x - other.min.x).abs());
            consider_x(other.max.x, (rect.min.x - other.max.x).abs());
        }

        if ranges_overlap(rect.min.x, rect.max.x, other.min.x, other.max.x) {
            consider_y(other.min.y, (rect.min.y - other.min.y).abs());
            consider_y(other.max.y - height, (rect.max.y - other.max.y).abs());
            consider_y(other.min.y - height, (rect.max.y - other.min.y).abs());
            consider_y(other.max.y, (rect.min.y - other.max.y).abs());
        }
    }

    Rect::from_min_size(Pos2::new(best_x, best_y), Vec2::new(width, height))
}

fn ranges_overlap(min_a: f32, max_a: f32, min_b: f32, max_b: f32) -> bool {
    min_a < max_b && max_a > min_b
}
