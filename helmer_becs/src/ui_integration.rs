use std::{sync::Arc, time::Instant};

#[cfg(not(target_arch = "wasm32"))]
use arboard::Clipboard;
use bevy_ecs::prelude::{Res, ResMut, Resource};
use glam::Vec2;
use hashbrown::{HashMap, HashSet};
use helmer::{
    graphics::common::renderer::{
        TextAlignH, TextAlignV, UiRenderCommand, UiRenderData, UiRenderImage, UiRenderRect,
        UiRenderText,
    },
    runtime::input_manager::InputManager,
};
use helmer_ui::{
    UiContext, UiDragInputSnapshot, UiDrawCommand, UiId, UiInputSnapshot, UiKeyInput, UiTextAlign,
    UiWindowOptions,
};
use parking_lot::RwLock;
use winit::{event::MouseButton, keyboard::KeyCode};

use crate::{BevyInputManager, BevyRuntimeProfiling, egui_integration::EguiInputPassthrough};

pub struct UiWindowSpec {
    pub id: String,
    pub title: String,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub min_size: [f32; 2],
    pub movable: bool,
    pub resizable: bool,
    pub closable: bool,
    pub collapsible: bool,
    pub input_passthrough: bool,
    pub scrollable: bool,
    pub title_bar_height: f32,
    pub resize_handle_size: f32,
}

impl Default for UiWindowSpec {
    fn default() -> Self {
        Self {
            id: "window".to_string(),
            title: String::new(),
            position: [16.0, 16.0],
            size: [420.0, 520.0],
            min_size: [220.0, 140.0],
            movable: true,
            resizable: true,
            closable: true,
            collapsible: true,
            input_passthrough: false,
            scrollable: true,
            title_bar_height: 30.0,
            resize_handle_size: 14.0,
        }
    }
}

pub type UiWindowFn = Box<dyn FnMut(&mut UiContext, &Arc<RwLock<InputManager>>) + Send + Sync>;
pub type UiWindowCloseFn = Box<dyn FnMut() + Send + Sync>;
type WindowContentMarker = (usize, Option<UiId>, Option<UiId>, Option<UiId>, bool);

#[derive(Clone, Copy, Debug)]
struct UiWindowFrameState {
    position: Vec2,
    size: Vec2,
    collapsed: bool,
    content_scroll: f32,
    content_max_scroll: f32,
}

#[derive(Clone, Debug)]
enum ActiveWindowInteraction {
    Move {
        window_id: String,
        pointer_offset: Vec2,
        button: WindowDragPointerButton,
    },
    Resize {
        window_id: String,
        pointer_anchor: Vec2,
        position_anchor: Vec2,
        size_anchor: Vec2,
        region: ResizeRegion,
    },
    Scrollbar {
        window_id: String,
        pointer_anchor_y: f32,
        scroll_anchor: f32,
        max_scroll: f32,
        track_height: f32,
        thumb_height: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowDragPointerButton {
    Primary,
    Secondary,
}

#[derive(Clone, Copy, Debug)]
enum ResizeRegion {
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

#[derive(Clone, Copy, Debug)]
struct ScrollbarGeometry {
    track_min: Vec2,
    track_max: Vec2,
    track_height: f32,
    thumb_top: f32,
    thumb_height: f32,
}

#[derive(Resource)]
pub struct UiResource {
    windows: parking_lot::Mutex<Vec<(UiWindowFn, UiWindowSpec)>>,
    close_actions: parking_lot::Mutex<HashMap<String, UiWindowCloseFn>>,
    closed_windows: parking_lot::RwLock<HashSet<String>>,
}

impl Default for UiResource {
    fn default() -> Self {
        Self {
            windows: parking_lot::Mutex::new(Vec::new()),
            close_actions: parking_lot::Mutex::new(HashMap::new()),
            closed_windows: parking_lot::RwLock::new(HashSet::new()),
        }
    }
}

impl UiResource {
    pub fn add_window<F>(&self, spec: UiWindowSpec, draw_fn: F)
    where
        F: FnMut(&mut UiContext, &Arc<RwLock<InputManager>>) + Send + Sync + 'static,
    {
        let mut windows = self.windows.lock();
        if let Some(existing) = windows
            .iter_mut()
            .find(|(_, existing)| existing.id == spec.id)
        {
            *existing = (Box::new(draw_fn), spec);
        } else {
            windows.push((Box::new(draw_fn), spec));
        }
    }

    pub fn set_close_action<F>(&self, id: impl Into<String>, action: F)
    where
        F: FnMut() + Send + Sync + 'static,
    {
        self.close_actions
            .lock()
            .insert(id.into(), Box::new(action));
    }

    pub fn is_window_closed(&self, id: &str) -> bool {
        self.closed_windows.read().contains(id)
    }

    pub fn open_window(&self, id: &str) {
        self.closed_windows.write().remove(id);
    }

    pub fn close_window(&self, id: &str) {
        self.closed_windows.write().insert(id.to_string());
    }

    pub fn toggle_window(&self, id: &str) -> bool {
        if self.is_window_closed(id) {
            self.open_window(id);
            true
        } else {
            self.close_window(id);
            false
        }
    }
}

#[derive(Resource, Default)]
pub struct UiRuntimeState {
    runtime: helmer_ui::UiRuntime,
    window_states: HashMap<String, UiWindowFrameState>,
    active_window_interaction: Option<ActiveWindowInteraction>,
    pointer_down_previous: bool,
    secondary_pointer_down_previous: bool,
    close_requests: Vec<usize>,
    window_content_markers: Vec<WindowContentMarker>,
}

#[derive(Resource, Clone, Default)]
pub struct UiRenderState {
    pub data: UiRenderData,
    pub revision: u64,
    pub command_hash: u64,
    pub wants_pointer_input: bool,
    pub wants_keyboard_input: bool,
}

#[derive(Resource, Clone, Default)]
pub struct UiRenderFrameOutput {
    pub data: UiRenderData,
    pub revision: u64,
    pub command_hash: u64,
    pub wants_pointer_input: bool,
    pub wants_keyboard_input: bool,
}

#[derive(Resource, Default)]
pub struct UiClipboard {
    #[cfg(not(target_arch = "wasm32"))]
    clipboard: Option<Clipboard>,
    cached_text: Option<String>,
}

impl UiClipboard {
    pub fn read_text(&mut self) -> Option<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.clipboard.is_none() {
                self.clipboard = Clipboard::new().ok();
            }
            if let Some(clipboard) = self.clipboard.as_mut() {
                if let Ok(text) = clipboard.get_text() {
                    self.cached_text = Some(text.clone());
                    return Some(text);
                }
            }
        }
        self.cached_text.clone()
    }

    pub fn write_text(&mut self, text: &str) {
        self.cached_text = Some(text.to_string());
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.clipboard.is_none() {
                self.clipboard = Clipboard::new().ok();
            }
            if let Some(clipboard) = self.clipboard.as_mut() {
                let _ = clipboard.set_text(text.to_string());
            }
        }
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub struct UiPerfStats {
    pub frame_us: u64,
    pub prepare_windows_us: u64,
    pub drag_input_us: u64,
    pub interaction_us: u64,
    pub run_frame_us: u64,
    pub scroll_metrics_us: u64,
    pub render_data_convert_us: u64,
    pub built_draw_commands: u32,
    pub draw_commands: u32,
    pub layout_rects: u32,
    pub windows_total: u32,
    pub windows_visible: u32,
    pub converted_this_frame: bool,
    pub ui_revision: u64,
    pub frame_ema_us: f64,
    pub run_frame_ema_us: f64,
    pub frame_peak_us: u64,
    pub run_frame_peak_us: u64,
}

impl UiPerfStats {
    fn update_ema(ema: &mut f64, sample: u64) {
        let sample = sample as f64;
        if *ema <= f64::EPSILON {
            *ema = sample;
        } else {
            *ema = (*ema * 0.90) + (sample * 0.10);
        }
    }

    fn record_frame(
        &mut self,
        frame_us: u64,
        prepare_windows_us: u64,
        drag_input_us: u64,
        interaction_us: u64,
        run_frame_us: u64,
        scroll_metrics_us: u64,
        render_data_convert_us: u64,
        built_draw_commands: usize,
        draw_commands: usize,
        layout_rects: usize,
        windows_total: usize,
        windows_visible: usize,
        converted_this_frame: bool,
        ui_revision: u64,
    ) {
        self.frame_us = frame_us;
        self.prepare_windows_us = prepare_windows_us;
        self.drag_input_us = drag_input_us;
        self.interaction_us = interaction_us;
        self.run_frame_us = run_frame_us;
        self.scroll_metrics_us = scroll_metrics_us;
        self.render_data_convert_us = render_data_convert_us;
        self.built_draw_commands = built_draw_commands as u32;
        self.draw_commands = draw_commands as u32;
        self.layout_rects = layout_rects as u32;
        self.windows_total = windows_total as u32;
        self.windows_visible = windows_visible as u32;
        self.converted_this_frame = converted_this_frame;
        self.ui_revision = ui_revision;
        Self::update_ema(&mut self.frame_ema_us, frame_us);
        Self::update_ema(&mut self.run_frame_ema_us, run_frame_us);
        self.frame_peak_us = self.frame_peak_us.max(frame_us);
        self.run_frame_peak_us = self.run_frame_peak_us.max(run_frame_us);
    }
}

fn keycode_to_digit_char(key: KeyCode) -> Option<char> {
    match key {
        KeyCode::Digit0 | KeyCode::Numpad0 => Some('0'),
        KeyCode::Digit1 | KeyCode::Numpad1 => Some('1'),
        KeyCode::Digit2 | KeyCode::Numpad2 => Some('2'),
        KeyCode::Digit3 | KeyCode::Numpad3 => Some('3'),
        KeyCode::Digit4 | KeyCode::Numpad4 => Some('4'),
        KeyCode::Digit5 | KeyCode::Numpad5 => Some('5'),
        KeyCode::Digit6 | KeyCode::Numpad6 => Some('6'),
        KeyCode::Digit7 | KeyCode::Numpad7 => Some('7'),
        KeyCode::Digit8 | KeyCode::Numpad8 => Some('8'),
        KeyCode::Digit9 | KeyCode::Numpad9 => Some('9'),
        _ => None,
    }
}

fn collect_drag_input(
    input_arc: &Arc<RwLock<InputManager>>,
    egui_pointer_passthrough: bool,
    egui_keyboard_passthrough: bool,
    pointer_down_previous: bool,
    clipboard: Option<&mut UiClipboard>,
) -> UiDragInputSnapshot {
    let input = input_arc.read();
    let egui_blocks_pointer = input.egui_wants_pointer && !egui_pointer_passthrough;
    let egui_blocks_keyboard = input.egui_wants_key && !egui_keyboard_passthrough;
    let pointer_down =
        !egui_blocks_pointer && input.active_mouse_buttons.contains(&MouseButton::Left);
    let pointer_pressed = pointer_down && !pointer_down_previous;
    let pointer_released = !pointer_down && pointer_down_previous;

    let ctrl = !egui_blocks_keyboard
        && (input.active_keys.contains(&KeyCode::ControlLeft)
            || input.active_keys.contains(&KeyCode::ControlRight));
    let shift = !egui_blocks_keyboard
        && (input.active_keys.contains(&KeyCode::ShiftLeft)
            || input.active_keys.contains(&KeyCode::ShiftRight));

    let mut just_pressed = Vec::new();
    if !egui_blocks_keyboard {
        for key in &input.just_pressed {
            if ctrl {
                match key {
                    KeyCode::KeyA => {
                        just_pressed.push(UiKeyInput::SelectAll);
                        continue;
                    }
                    KeyCode::KeyC => {
                        just_pressed.push(UiKeyInput::Copy);
                        continue;
                    }
                    KeyCode::KeyX => {
                        just_pressed.push(UiKeyInput::Cut);
                        continue;
                    }
                    KeyCode::KeyV => {
                        just_pressed.push(UiKeyInput::Paste);
                        continue;
                    }
                    _ => {}
                }
            }

            match key {
                KeyCode::Enter | KeyCode::NumpadEnter => just_pressed.push(UiKeyInput::Enter),
                KeyCode::Escape => just_pressed.push(UiKeyInput::Escape),
                KeyCode::ArrowLeft => just_pressed.push(UiKeyInput::ArrowLeft),
                KeyCode::ArrowRight => just_pressed.push(UiKeyInput::ArrowRight),
                KeyCode::ArrowUp => just_pressed.push(UiKeyInput::ArrowUp),
                KeyCode::ArrowDown => just_pressed.push(UiKeyInput::ArrowDown),
                KeyCode::Home => just_pressed.push(UiKeyInput::Home),
                KeyCode::End => just_pressed.push(UiKeyInput::End),
                KeyCode::Backspace => just_pressed.push(UiKeyInput::Backspace),
                KeyCode::Delete => just_pressed.push(UiKeyInput::Delete),
                KeyCode::Minus | KeyCode::NumpadSubtract => {
                    just_pressed.push(UiKeyInput::Char('-'))
                }
                KeyCode::Period | KeyCode::NumpadDecimal => {
                    just_pressed.push(UiKeyInput::Char('.'))
                }
                KeyCode::KeyE => just_pressed.push(UiKeyInput::Char(if shift { 'E' } else { 'e' })),
                key => {
                    if let Some(digit) = keycode_to_digit_char(*key) {
                        just_pressed.push(UiKeyInput::Char(digit));
                    }
                }
            }
        }
    }
    drop(input);

    let clipboard_text = if just_pressed
        .iter()
        .any(|key| matches!(key, UiKeyInput::Paste))
    {
        clipboard.and_then(|clipboard| clipboard.read_text())
    } else {
        None
    };

    let input = input_arc.read();
    UiDragInputSnapshot {
        pointer_down,
        pointer_pressed,
        pointer_released,
        cursor_pos: Vec2::new(
            input.cursor_position.x as f32,
            input.cursor_position.y as f32,
        ),
        just_pressed,
        ctrl,
        shift,
        clipboard_text,
    }
}

pub fn ui_system(
    input: Option<Res<BevyInputManager>>,
    egui_passthrough: Option<Res<EguiInputPassthrough>>,
    mut ui_runtime_state: ResMut<UiRuntimeState>,
    ui_res: Res<UiResource>,
    mut ui_render_state: ResMut<UiRenderFrameOutput>,
    mut ui_perf: ResMut<UiPerfStats>,
    runtime_profiling: Option<Res<BevyRuntimeProfiling>>,
    mut clipboard: Option<ResMut<UiClipboard>>,
) {
    let profiling_enabled = runtime_profiling
        .as_ref()
        .map(|profiling| {
            profiling
                .0
                .enabled
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .unwrap_or(false);
    let ui_system_start = if profiling_enabled {
        Some(Instant::now())
    } else {
        None
    };

    let Some(input_arc) = input.map(|input| input.0.clone()) else {
        if !profiling_enabled {
            *ui_perf = UiPerfStats::default();
        }
        return;
    };

    let passthrough = egui_passthrough.as_deref().copied().unwrap_or_default();

    let (snapshot, secondary_pointer_down, egui_blocks_pointer) = {
        let input = input_arc.read();
        let egui_blocks_pointer = input.egui_wants_pointer && !passthrough.pointer;
        (
            UiInputSnapshot {
                viewport_size: Vec2::new(
                    input.window_size.x.max(1) as f32,
                    input.window_size.y.max(1) as f32,
                ),
                pointer_position: if egui_blocks_pointer {
                    None
                } else {
                    Some(Vec2::new(
                        input.cursor_position.x as f32,
                        input.cursor_position.y as f32,
                    ))
                },
                pointer_down: !egui_blocks_pointer
                    && input.active_mouse_buttons.contains(&MouseButton::Left),
                scroll_delta: if egui_blocks_pointer {
                    Vec2::ZERO
                } else {
                    Vec2::new(
                        input.mouse_wheel_unfiltered.x,
                        input.mouse_wheel_unfiltered.y,
                    )
                },
            },
            !egui_blocks_pointer
                && (input.active_mouse_buttons.contains(&MouseButton::Right)
                    || input.active_mouse_buttons.contains(&MouseButton::Middle)),
            egui_blocks_pointer,
        )
    };

    let mut runtime = std::mem::take(&mut ui_runtime_state.runtime);
    let mut window_states = std::mem::take(&mut ui_runtime_state.window_states);
    let mut active_window_interaction =
        std::mem::take(&mut ui_runtime_state.active_window_interaction);
    let pointer_down_previous = ui_runtime_state.pointer_down_previous;
    let secondary_pointer_down_previous = ui_runtime_state.secondary_pointer_down_previous;
    let mut close_requests = std::mem::take(&mut ui_runtime_state.close_requests);
    let mut window_content_markers = std::mem::take(&mut ui_runtime_state.window_content_markers);
    close_requests.clear();
    window_content_markers.clear();

    let mut windows = ui_res.windows.lock();
    let closed_windows = ui_res.closed_windows.read();

    let prepare_windows_start = if profiling_enabled {
        Some(Instant::now())
    } else {
        None
    };
    prepare_window_states(
        &mut window_states,
        windows.as_slice(),
        snapshot.viewport_size,
    );
    let prepare_windows_us = prepare_windows_start
        .map(|start| start.elapsed().as_micros() as u64)
        .unwrap_or(0);
    let visible_window_count = windows
        .iter()
        .filter(|(_, spec)| !closed_windows.contains(spec.id.as_str()))
        .count();
    close_requests.reserve(visible_window_count);
    window_content_markers.reserve(visible_window_count);

    let drag_input_start = if profiling_enabled {
        Some(Instant::now())
    } else {
        None
    };
    let drag_input = collect_drag_input(
        &input_arc,
        passthrough.pointer,
        passthrough.keyboard,
        pointer_down_previous,
        clipboard.as_deref_mut(),
    );
    let drag_input_us = drag_input_start
        .map(|start| start.elapsed().as_micros() as u64)
        .unwrap_or(0);

    let interaction_start = if profiling_enabled {
        Some(Instant::now())
    } else {
        None
    };
    update_window_interaction(
        &runtime,
        &snapshot,
        windows.as_slice(),
        &closed_windows,
        &mut window_states,
        &mut active_window_interaction,
        pointer_down_previous,
        secondary_pointer_down,
        secondary_pointer_down_previous,
    );
    let interaction_us = interaction_start
        .map(|start| start.elapsed().as_micros() as u64)
        .unwrap_or(0);

    let has_visible_content = visible_window_count > 0 || !runtime.retained().roots().is_empty();
    if !has_visible_content && active_window_interaction.is_none() {
        let mut render_data_convert_us = 0u64;
        let had_render_content = !ui_render_state.data.commands.is_empty()
            || ui_render_state.command_hash != 0
            || ui_render_state.wants_pointer_input;
        if had_render_content {
            let clear_start = if profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            ui_render_state.data.commands.clear();
            ui_render_state.command_hash = 0;
            ui_render_state.revision = ui_render_state.revision.wrapping_add(1);
            render_data_convert_us = clear_start
                .map(|start| start.elapsed().as_micros() as u64)
                .unwrap_or(0);
        }
        ui_render_state.wants_pointer_input = false;
        ui_render_state.wants_keyboard_input = false;
        let layout_rect_count = runtime.layout_rect_count();

        ui_runtime_state.runtime = runtime;
        ui_runtime_state.window_states = window_states;
        ui_runtime_state.active_window_interaction = active_window_interaction;
        ui_runtime_state.pointer_down_previous = snapshot.pointer_down;
        ui_runtime_state.secondary_pointer_down_previous = secondary_pointer_down;
        ui_runtime_state.close_requests = close_requests;
        ui_runtime_state.window_content_markers = window_content_markers;

        if profiling_enabled {
            ui_perf.record_frame(
                ui_system_start
                    .map(|start| start.elapsed().as_micros() as u64)
                    .unwrap_or(0),
                prepare_windows_us,
                drag_input_us,
                interaction_us,
                0,
                0,
                render_data_convert_us,
                0,
                ui_render_state.data.commands.len(),
                layout_rect_count,
                windows.len(),
                visible_window_count,
                render_data_convert_us > 0,
                ui_render_state.revision,
            );
        } else {
            *ui_perf = UiPerfStats::default();
        }
        return;
    }

    let input_arc_for_windows = input_arc.clone();

    let run_frame_start = if profiling_enabled {
        Some(Instant::now())
    } else {
        None
    };
    let hash_draw_commands = active_window_interaction.is_none();
    let mut output = runtime.run_frame_with_options(snapshot, hash_draw_commands, |ui| {
        ui.set_drag_input_snapshot(drag_input.clone());
        for (window_index, (draw_fn, spec)) in windows.iter_mut().enumerate() {
            if closed_windows.contains(spec.id.as_str()) {
                continue;
            }
            let state = window_states
                .get(&spec.id)
                .copied()
                .unwrap_or(UiWindowFrameState {
                    position: Vec2::new(spec.position[0], spec.position[1]),
                    size: Vec2::new(spec.size[0], spec.size[1]),
                    collapsed: false,
                    content_scroll: 0.0,
                    content_max_scroll: 0.0,
                });
            let options = UiWindowOptions {
                title: if spec.title.is_empty() {
                    None
                } else {
                    Some(spec.title.clone())
                },
                position: state.position,
                size: state.size,
                movable: spec.movable,
                resizable: spec.resizable,
                closable: spec.closable,
                collapsible: spec.collapsible,
                collapsed: state.collapsed,
                min_size: Vec2::new(spec.min_size[0], spec.min_size[1]),
                title_bar_height: spec.title_bar_height,
                resize_handle_size: spec.resize_handle_size,
                scroll_y: if spec.scrollable {
                    state.content_scroll
                } else {
                    0.0
                },
                max_scroll_y: if spec.scrollable {
                    state.content_max_scroll
                } else {
                    0.0
                },
            };
            let response = ui.window(spec.id.as_str(), options, |ui| {
                draw_fn(ui, &input_arc_for_windows);
            });
            if let Some(state) = window_states.get_mut(&spec.id) {
                state.collapsed = response.collapsed;
                clamp_window_state_to_viewport(state, spec, snapshot.viewport_size);
            }
            window_content_markers.push((
                window_index,
                response.content_viewport_id,
                response.content_inner_id,
                response.content_extent_id,
                spec.scrollable,
            ));
            if response.close_requested && spec.closable {
                close_requests.push(window_index);
            }
        }
    });
    let run_frame_us = run_frame_start
        .map(|start| start.elapsed().as_micros() as u64)
        .unwrap_or(0);

    if let Some(text) = runtime.take_drag_clipboard_write()
        && let Some(clipboard) = clipboard.as_deref_mut()
    {
        clipboard.write_text(&text);
    }

    let scroll_metrics_start = if profiling_enabled {
        Some(Instant::now())
    } else {
        None
    };
    for (window_index, viewport_id, inner_id, extent_id, scrollable) in
        window_content_markers.iter().copied()
    {
        let Some((_, spec)) = windows.get(window_index) else {
            continue;
        };
        if !scrollable {
            if let Some(state) = window_states.get_mut(&spec.id) {
                state.content_max_scroll = 0.0;
                state.content_scroll = 0.0;
            }
            continue;
        }
        if let (Some(viewport_id), Some(inner_id)) = (viewport_id, inner_id) {
            let viewport_rect = runtime.layout_rect(viewport_id);
            let viewport_height = viewport_rect.map(|rect| rect.height).unwrap_or(0.0);
            let inner_height = runtime
                .layout_rect(inner_id)
                .map(|rect| rect.height)
                .unwrap_or(0.0);
            let layout_max_scroll = (inner_height - viewport_height).max(0.0);
            let extent_max_scroll =
                if let (Some(viewport), Some(extent_id)) = (viewport_rect, extent_id) {
                    runtime
                        .layout_rect(extent_id)
                        .map(|extent| {
                            let inner_top = runtime
                                .layout_rect(inner_id)
                                .map(|inner| inner.y)
                                .unwrap_or(viewport.y);
                            let extent_bottom = extent.bottom().max(inner_top);
                            let content_height = (extent_bottom - inner_top).max(0.0);
                            (content_height - viewport.height.max(0.0)).max(0.0)
                        })
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
            let has_extent_measure = matches!((viewport_rect, extent_id), (Some(_), Some(_)));
            let max_scroll = if has_extent_measure {
                extent_max_scroll.max(0.0)
            } else {
                layout_max_scroll.max(extent_max_scroll).max(0.0)
            };
            if let Some(state) = window_states.get_mut(&spec.id) {
                state.content_max_scroll = max_scroll;
                if state.content_max_scroll <= 1.0 {
                    state.content_max_scroll = 0.0;
                    state.content_scroll = 0.0;
                } else {
                    state.content_scroll =
                        state.content_scroll.clamp(0.0, state.content_max_scroll);
                }
            }
        } else {
            if let Some(state) = window_states.get_mut(&spec.id) {
                state.content_max_scroll = 0.0;
                state.content_scroll = 0.0;
            }
        }
    }
    let scroll_metrics_us = scroll_metrics_start
        .map(|start| start.elapsed().as_micros() as u64)
        .unwrap_or(0);

    drop(closed_windows);

    if !close_requests.is_empty() {
        let mut close_actions = ui_res.close_actions.lock();
        let mut closed_set = ui_res.closed_windows.write();
        for window_index in close_requests.iter().copied() {
            let Some((_, spec)) = windows.get(window_index) else {
                continue;
            };
            let window_id = spec.id.as_str();
            closed_set.insert(spec.id.clone());
            window_states.remove(window_id);
            if let Some(close_action) = close_actions.get_mut(window_id) {
                close_action();
            }
            if matches!(
                &active_window_interaction,
                Some(ActiveWindowInteraction::Move { window_id: active, .. })
                    if active == window_id
            ) || matches!(
                &active_window_interaction,
                Some(ActiveWindowInteraction::Resize { window_id: active, .. })
                    if active == window_id
            ) || matches!(
                &active_window_interaction,
                Some(ActiveWindowInteraction::Scrollbar { window_id: active, .. })
                    if active == window_id
            ) {
                active_window_interaction = None;
            }
        }
    }

    if egui_blocks_pointer {
        // keep interaction capture stable while egui is actively capturing this frame
        active_window_interaction = None;
    }

    let active_pointer_capture = active_window_interaction.is_some();
    let layout_rect_count = runtime.layout_rect_count();
    let built_draw_commands = output.draw_commands.len();

    let mut render_data_convert_us = 0u64;
    let next_pointer_capture = output.interaction.pointer_captured || active_pointer_capture;
    ui_render_state.wants_pointer_input = next_pointer_capture;
    ui_render_state.wants_keyboard_input = false;

    let force_update = active_pointer_capture;
    if force_update {
        let convert_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        draw_commands_to_render_data_into(&mut ui_render_state.data, &mut output.draw_commands);
        render_data_convert_us = convert_start
            .map(|start| start.elapsed().as_micros() as u64)
            .unwrap_or(0);
        if output.command_hash_valid {
            ui_render_state.command_hash = output.command_hash;
        } else {
            ui_render_state.command_hash = ui_render_state.command_hash.wrapping_add(1);
        }
        ui_render_state.revision = ui_render_state.revision.wrapping_add(1);
    } else if output.command_hash_valid {
        let next_hash = output.command_hash;
        if ui_render_state.command_hash != next_hash {
            let convert_start = if profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            draw_commands_to_render_data_into(&mut ui_render_state.data, &mut output.draw_commands);
            render_data_convert_us = convert_start
                .map(|start| start.elapsed().as_micros() as u64)
                .unwrap_or(0);
            ui_render_state.command_hash = next_hash;
            ui_render_state.revision = ui_render_state.revision.wrapping_add(1);
        }
    } else {
        let convert_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        draw_commands_to_render_data_into(&mut ui_render_state.data, &mut output.draw_commands);
        render_data_convert_us = convert_start
            .map(|start| start.elapsed().as_micros() as u64)
            .unwrap_or(0);
        ui_render_state.command_hash = ui_render_state.command_hash.wrapping_add(1);
        ui_render_state.revision = ui_render_state.revision.wrapping_add(1);
    }
    let ui_revision = ui_render_state.revision;
    let draw_commands_count = ui_render_state.data.commands.len();
    runtime.recycle_draw_commands(output.draw_commands);

    ui_runtime_state.runtime = runtime;
    ui_runtime_state.window_states = window_states;
    ui_runtime_state.active_window_interaction = active_window_interaction;
    ui_runtime_state.pointer_down_previous = snapshot.pointer_down;
    ui_runtime_state.secondary_pointer_down_previous = secondary_pointer_down;
    ui_runtime_state.close_requests = close_requests;
    ui_runtime_state.window_content_markers = window_content_markers;

    if profiling_enabled {
        ui_perf.record_frame(
            ui_system_start
                .map(|start| start.elapsed().as_micros() as u64)
                .unwrap_or(0),
            prepare_windows_us,
            drag_input_us,
            interaction_us,
            run_frame_us,
            scroll_metrics_us,
            render_data_convert_us,
            built_draw_commands,
            draw_commands_count,
            layout_rect_count,
            windows.len(),
            visible_window_count,
            render_data_convert_us > 0,
            ui_revision,
        );
    } else {
        *ui_perf = UiPerfStats::default();
    }
}

fn prepare_window_states(
    window_states: &mut HashMap<String, UiWindowFrameState>,
    windows: &[(UiWindowFn, UiWindowSpec)],
    viewport_size: Vec2,
) {
    let mut valid_ids: HashSet<&str> = HashSet::with_capacity(windows.len());
    for (_, spec) in windows {
        valid_ids.insert(spec.id.as_str());
    }

    window_states.retain(|id, _| valid_ids.contains(id.as_str()));

    for (_, spec) in windows {
        let state = window_states
            .entry(spec.id.clone())
            .or_insert_with(|| UiWindowFrameState {
                position: Vec2::new(spec.position[0], spec.position[1]),
                size: Vec2::new(spec.size[0], spec.size[1]),
                collapsed: false,
                content_scroll: 0.0,
                content_max_scroll: 0.0,
            });
        clamp_window_state_to_viewport(state, spec, viewport_size);
    }
}

fn update_window_interaction(
    runtime: &helmer_ui::UiRuntime,
    snapshot: &UiInputSnapshot,
    windows: &[(UiWindowFn, UiWindowSpec)],
    closed_windows: &HashSet<String>,
    window_states: &mut HashMap<String, UiWindowFrameState>,
    active_window_interaction: &mut Option<ActiveWindowInteraction>,
    pointer_down_previous: bool,
    secondary_pointer_down: bool,
    secondary_pointer_down_previous: bool,
) {
    let window_exists = |window_id: &str| {
        window_states.contains_key(window_id) && !closed_windows.contains(window_id)
    };
    if let Some(active) = active_window_interaction.as_ref() {
        let window_id = match active {
            ActiveWindowInteraction::Move { window_id, .. } => window_id.as_str(),
            ActiveWindowInteraction::Resize { window_id, .. } => window_id.as_str(),
            ActiveWindowInteraction::Scrollbar { window_id, .. } => window_id.as_str(),
        };
        if !window_exists(window_id) {
            *active_window_interaction = None;
        }
    }

    let pointer_pressed = snapshot.pointer_down && !pointer_down_previous;
    let pointer_released = !snapshot.pointer_down && pointer_down_previous;
    let secondary_pointer_pressed = secondary_pointer_down && !secondary_pointer_down_previous;
    let secondary_pointer_released = !secondary_pointer_down && secondary_pointer_down_previous;
    let pointer = snapshot.pointer_position;
    let pointer_over_interactive = pointer
        .map(|pointer| runtime.is_pointer_over_interactive(pointer))
        .unwrap_or(false);

    if snapshot.scroll_delta.y.abs() > f32::EPSILON {
        if let Some(pointer) = pointer {
            for (_, spec) in windows.iter().rev() {
                if closed_windows.contains(spec.id.as_str()) {
                    continue;
                }
                let Some(state) = window_states.get_mut(&spec.id) else {
                    continue;
                };
                let visible_size = window_visible_size(spec, *state);
                if !contains_point(state.position, visible_size, pointer) {
                    continue;
                }
                if spec.input_passthrough && !pointer_over_interactive {
                    continue;
                }
                if !spec.scrollable {
                    continue;
                }
                if state.content_max_scroll > 1.0 {
                    let unclamped =
                        (state.content_scroll - snapshot.scroll_delta.y * 34.0).max(0.0);
                    state.content_scroll = unclamped.min(state.content_max_scroll.max(0.0));
                } else {
                    state.content_scroll = 0.0;
                }
                break;
            }
        }
    }

    if pointer_released
        && matches!(
            active_window_interaction.as_ref(),
            Some(ActiveWindowInteraction::Move {
                button: WindowDragPointerButton::Primary,
                ..
            }) | Some(ActiveWindowInteraction::Resize { .. })
                | Some(ActiveWindowInteraction::Scrollbar { .. })
        )
    {
        *active_window_interaction = None;
    }
    if secondary_pointer_released
        && matches!(
            active_window_interaction.as_ref(),
            Some(ActiveWindowInteraction::Move {
                button: WindowDragPointerButton::Secondary,
                ..
            })
        )
    {
        *active_window_interaction = None;
    }

    if pointer_pressed {
        *active_window_interaction = None;
        if let Some(pointer) = pointer {
            for (_, spec) in windows.iter().rev() {
                if closed_windows.contains(spec.id.as_str()) {
                    continue;
                }
                let Some(state) = window_states.get(&spec.id).copied() else {
                    continue;
                };
                let visible_size = window_visible_size(spec, state);
                if !contains_point(state.position, visible_size, pointer) {
                    continue;
                }
                if spec.input_passthrough && !pointer_over_interactive {
                    continue;
                }
                if spec.scrollable
                    && let Some(scrollbar) = window_scrollbar_geometry(spec, state, visible_size)
                {
                    let pointer_in_track = pointer.x >= scrollbar.track_min.x
                        && pointer.x <= scrollbar.track_max.x
                        && pointer.y >= scrollbar.track_min.y
                        && pointer.y <= scrollbar.track_max.y;
                    if pointer_in_track {
                        let pointer_in_thumb = pointer.y >= scrollbar.thumb_top
                            && pointer.y <= scrollbar.thumb_top + scrollbar.thumb_height;
                        let mut scroll_anchor = state.content_scroll;
                        if !pointer_in_thumb {
                            let travel = (scrollbar.track_height - scrollbar.thumb_height).max(1.0);
                            let centered =
                                (pointer.y - scrollbar.track_min.y - scrollbar.thumb_height * 0.5)
                                    .clamp(0.0, travel);
                            let t = centered / travel;
                            scroll_anchor = t * state.content_max_scroll;
                            if let Some(state_mut) = window_states.get_mut(&spec.id) {
                                state_mut.content_scroll =
                                    scroll_anchor.clamp(0.0, state.content_max_scroll);
                            }
                        }
                        *active_window_interaction = Some(ActiveWindowInteraction::Scrollbar {
                            window_id: spec.id.clone(),
                            pointer_anchor_y: pointer.y,
                            scroll_anchor,
                            max_scroll: state.content_max_scroll,
                            track_height: scrollbar.track_height,
                            thumb_height: scrollbar.thumb_height,
                        });
                        break;
                    }
                }
                let handle_size = spec.resize_handle_size.max(6.0);
                if spec.resizable
                    && let Some(region) =
                        resize_region_at_point(state.position, visible_size, handle_size, pointer)
                {
                    *active_window_interaction = Some(ActiveWindowInteraction::Resize {
                        window_id: spec.id.clone(),
                        pointer_anchor: pointer,
                        position_anchor: state.position,
                        size_anchor: state.size,
                        region,
                    });
                    break;
                }
                let in_title_bar = contains_point(
                    state.position,
                    Vec2::new(visible_size.x, spec.title_bar_height.max(18.0)),
                    pointer,
                );
                if spec.movable && in_title_bar {
                    *active_window_interaction = Some(ActiveWindowInteraction::Move {
                        window_id: spec.id.clone(),
                        pointer_offset: pointer - state.position,
                        button: WindowDragPointerButton::Primary,
                    });
                    break;
                }
                if spec.movable && !pointer_over_interactive {
                    *active_window_interaction = Some(ActiveWindowInteraction::Move {
                        window_id: spec.id.clone(),
                        pointer_offset: pointer - state.position,
                        button: WindowDragPointerButton::Primary,
                    });
                }
                // consume this press for the top-most containing window
                break;
            }
        }
    }

    if secondary_pointer_pressed {
        *active_window_interaction = None;
        if let Some(pointer) = pointer {
            for (_, spec) in windows.iter().rev() {
                if closed_windows.contains(spec.id.as_str()) {
                    continue;
                }
                let Some(state) = window_states.get(&spec.id).copied() else {
                    continue;
                };
                let visible_size = window_visible_size(spec, state);
                if !contains_point(state.position, visible_size, pointer) {
                    continue;
                }
                if spec.input_passthrough && !pointer_over_interactive {
                    continue;
                }
                let in_title_bar = contains_point(
                    state.position,
                    Vec2::new(visible_size.x, spec.title_bar_height.max(18.0)),
                    pointer,
                );
                if spec.movable && (in_title_bar || !pointer_over_interactive) {
                    *active_window_interaction = Some(ActiveWindowInteraction::Move {
                        window_id: spec.id.clone(),
                        pointer_offset: pointer - state.position,
                        button: WindowDragPointerButton::Secondary,
                    });
                }
                // consume this press for the top-most containing window
                break;
            }
        }
    }

    let active_pointer_down = match active_window_interaction.as_ref() {
        Some(ActiveWindowInteraction::Move {
            button: WindowDragPointerButton::Primary,
            ..
        }) => snapshot.pointer_down,
        Some(ActiveWindowInteraction::Move {
            button: WindowDragPointerButton::Secondary,
            ..
        }) => secondary_pointer_down,
        Some(ActiveWindowInteraction::Resize { .. })
        | Some(ActiveWindowInteraction::Scrollbar { .. }) => snapshot.pointer_down,
        None => false,
    };
    if !active_pointer_down {
        return;
    }

    let Some(pointer) = pointer else {
        return;
    };

    match active_window_interaction.clone() {
        Some(ActiveWindowInteraction::Move {
            window_id,
            pointer_offset,
            button: _,
        }) => {
            if let Some((_, spec)) = windows.iter().find(|(_, spec)| {
                spec.id == window_id && !closed_windows.contains(spec.id.as_str())
            }) {
                if let Some(state) = window_states.get_mut(&window_id) {
                    state.position = pointer - pointer_offset;
                    clamp_window_state_to_viewport(state, spec, snapshot.viewport_size);
                }
            }
        }
        Some(ActiveWindowInteraction::Resize {
            window_id,
            pointer_anchor,
            position_anchor,
            size_anchor,
            region,
        }) => {
            if let Some((_, spec)) = windows.iter().find(|(_, spec)| {
                spec.id == window_id && !closed_windows.contains(spec.id.as_str())
            }) {
                if let Some(state) = window_states.get_mut(&window_id) {
                    apply_resize_region(
                        state,
                        spec,
                        snapshot.viewport_size,
                        position_anchor,
                        size_anchor,
                        pointer - pointer_anchor,
                        region,
                    );
                    clamp_window_state_to_viewport(state, spec, snapshot.viewport_size);
                }
            }
        }
        Some(ActiveWindowInteraction::Scrollbar {
            window_id,
            pointer_anchor_y,
            scroll_anchor,
            max_scroll,
            track_height,
            thumb_height,
        }) => {
            if let Some(state) = window_states.get_mut(&window_id) {
                let travel = (track_height - thumb_height).max(1.0);
                let delta = pointer.y - pointer_anchor_y;
                let t = delta / travel;
                state.content_scroll = (scroll_anchor + t * max_scroll).clamp(0.0, max_scroll);
            }
        }
        None => {}
    }
}

fn clamp_window_state_to_viewport(
    state: &mut UiWindowFrameState,
    spec: &UiWindowSpec,
    viewport_size: Vec2,
) {
    let min_size = Vec2::new(spec.min_size[0].max(1.0), spec.min_size[1].max(1.0));
    state.size.x = state
        .size
        .x
        .max(min_size.x)
        .min(viewport_size.x.max(min_size.x));
    state.size.y = state
        .size
        .y
        .max(min_size.y)
        .min(viewport_size.y.max(min_size.y));
    let visible_size = window_visible_size(spec, *state);
    state.position.x = state
        .position
        .x
        .clamp(0.0, (viewport_size.x - visible_size.x).max(0.0));
    state.position.y = state
        .position
        .y
        .clamp(0.0, (viewport_size.y - visible_size.y).max(0.0));
}

fn window_visible_size(spec: &UiWindowSpec, state: UiWindowFrameState) -> Vec2 {
    if state.collapsed {
        Vec2::new(state.size.x, spec.title_bar_height.max(18.0))
    } else {
        state.size
    }
}

fn window_scrollbar_geometry(
    spec: &UiWindowSpec,
    state: UiWindowFrameState,
    visible_size: Vec2,
) -> Option<ScrollbarGeometry> {
    if state.collapsed || state.content_max_scroll <= 0.0 {
        return None;
    }
    let has_title_bar = !spec.title.is_empty();
    let viewport_height = (visible_size.y
        - if has_title_bar {
            spec.title_bar_height.max(18.0) + 1.0
        } else {
            0.0
        })
    .max(1.0);
    if viewport_height <= 1.0 {
        return None;
    }

    let track_inset = 3.0;
    let track_width = 6.0;
    let track_height = (viewport_height - track_inset * 2.0).max(8.0);
    let visible_ratio =
        (viewport_height / (viewport_height + state.content_max_scroll.max(0.0))).clamp(0.05, 1.0);
    let thumb_height = (track_height * visible_ratio).clamp(14.0, track_height);
    let travel = (track_height - thumb_height).max(0.0);
    let t = if state.content_max_scroll > 0.0 {
        (state.content_scroll / state.content_max_scroll).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let content_top = state.position.y
        + if has_title_bar {
            spec.title_bar_height.max(18.0) + 1.0
        } else {
            0.0
        };
    let track_left = state.position.x + visible_size.x - track_inset - track_width;
    let track_top = content_top + track_inset;
    Some(ScrollbarGeometry {
        track_min: Vec2::new(track_left, track_top),
        track_max: Vec2::new(track_left + track_width, track_top + track_height),
        track_height,
        thumb_top: track_top + travel * t,
        thumb_height,
    })
}

fn resize_region_at_point(
    position: Vec2,
    size: Vec2,
    handle_size: f32,
    pointer: Vec2,
) -> Option<ResizeRegion> {
    let corner_hs = handle_size.max(6.0);
    let edge_hs = (corner_hs * 0.36).clamp(3.0, 6.0);
    let x = pointer.x;
    let y = pointer.y;
    let near_left = (x - position.x).abs() <= edge_hs;
    let near_right = (x - (position.x + size.x)).abs() <= edge_hs;
    let near_top = (y - position.y).abs() <= edge_hs;
    let near_bottom = (y - (position.y + size.y)).abs() <= edge_hs;
    let left = near_left && y >= position.y - corner_hs && y <= position.y + size.y + corner_hs;
    let right = near_right && y >= position.y - corner_hs && y <= position.y + size.y + corner_hs;
    let top = near_top && x >= position.x - corner_hs && x <= position.x + size.x + corner_hs;
    let bottom = near_bottom && x >= position.x - corner_hs && x <= position.x + size.x + corner_hs;

    let corner_left = (x - position.x).abs() <= corner_hs;
    let corner_right = (x - (position.x + size.x)).abs() <= corner_hs;
    let corner_top = (y - position.y).abs() <= corner_hs;
    let corner_bottom = (y - (position.y + size.y)).abs() <= corner_hs;

    if corner_top && corner_left {
        return Some(ResizeRegion::TopLeft);
    }
    if corner_top && corner_right {
        return Some(ResizeRegion::TopRight);
    }
    if corner_bottom && corner_left {
        return Some(ResizeRegion::BottomLeft);
    }
    if corner_bottom && corner_right {
        return Some(ResizeRegion::BottomRight);
    }
    if left {
        return Some(ResizeRegion::Left);
    }
    if right {
        return Some(ResizeRegion::Right);
    }
    if top {
        return Some(ResizeRegion::Top);
    }
    if bottom {
        return Some(ResizeRegion::Bottom);
    }
    None
}

fn apply_resize_region(
    state: &mut UiWindowFrameState,
    spec: &UiWindowSpec,
    viewport_size: Vec2,
    position_anchor: Vec2,
    size_anchor: Vec2,
    pointer_delta: Vec2,
    region: ResizeRegion,
) {
    let mut left = position_anchor.x;
    let mut top = position_anchor.y;
    let mut right = position_anchor.x + size_anchor.x;
    let mut bottom = position_anchor.y + size_anchor.y;

    match region {
        ResizeRegion::Left => {
            left += pointer_delta.x;
        }
        ResizeRegion::Right => {
            right += pointer_delta.x;
        }
        ResizeRegion::Top => {
            top += pointer_delta.y;
        }
        ResizeRegion::Bottom => {
            bottom += pointer_delta.y;
        }
        ResizeRegion::TopLeft => {
            left += pointer_delta.x;
            top += pointer_delta.y;
        }
        ResizeRegion::TopRight => {
            right += pointer_delta.x;
            top += pointer_delta.y;
        }
        ResizeRegion::BottomLeft => {
            left += pointer_delta.x;
            bottom += pointer_delta.y;
        }
        ResizeRegion::BottomRight => {
            right += pointer_delta.x;
            bottom += pointer_delta.y;
        }
    }

    let min_size = Vec2::new(spec.min_size[0].max(1.0), spec.min_size[1].max(1.0));
    if right - left < min_size.x {
        match region {
            ResizeRegion::Left | ResizeRegion::TopLeft | ResizeRegion::BottomLeft => {
                left = right - min_size.x
            }
            _ => right = left + min_size.x,
        }
    }
    if bottom - top < min_size.y {
        match region {
            ResizeRegion::Top | ResizeRegion::TopLeft | ResizeRegion::TopRight => {
                top = bottom - min_size.y
            }
            _ => bottom = top + min_size.y,
        }
    }

    left = left.clamp(0.0, viewport_size.x.max(0.0));
    top = top.clamp(0.0, viewport_size.y.max(0.0));
    right = right.clamp(0.0, viewport_size.x.max(0.0));
    bottom = bottom.clamp(0.0, viewport_size.y.max(0.0));

    if right <= left {
        right = (left + min_size.x).min(viewport_size.x.max(min_size.x));
        left = (right - min_size.x).max(0.0);
    }
    if bottom <= top {
        bottom = (top + min_size.y).min(viewport_size.y.max(min_size.y));
        top = (bottom - min_size.y).max(0.0);
    }

    state.position = Vec2::new(left, top);
    state.size = Vec2::new((right - left).max(1.0), (bottom - top).max(1.0));
}

fn contains_point(position: Vec2, size: Vec2, pointer: Vec2) -> bool {
    pointer.x >= position.x
        && pointer.y >= position.y
        && pointer.x <= position.x + size.x
        && pointer.y <= position.y + size.y
}

fn draw_commands_to_render_data_into(data: &mut UiRenderData, commands: &mut Vec<UiDrawCommand>) {
    data.commands.clear();
    data.commands.reserve(commands.len());

    for command in commands.drain(..) {
        match command {
            UiDrawCommand::Rect(rect) => {
                data.commands.push(UiRenderCommand::Rect(UiRenderRect {
                    id: rect.id.0,
                    rect: [rect.rect.x, rect.rect.y, rect.rect.width, rect.rect.height],
                    color: rect.color.to_array(),
                    clip_rect: rect
                        .clip
                        .map(|clip| [clip.x, clip.y, clip.right(), clip.bottom()]),
                    layer: rect.layer,
                }));
            }
            UiDrawCommand::Image(image) => {
                data.commands.push(UiRenderCommand::Image(UiRenderImage {
                    id: image.id.0,
                    rect: [
                        image.rect.x,
                        image.rect.y,
                        image.rect.width,
                        image.rect.height,
                    ],
                    texture_id: image.texture_id,
                    tint: image.tint.to_array(),
                    uv_min: image.uv_min,
                    uv_max: image.uv_max,
                    clip_rect: image
                        .clip
                        .map(|clip| [clip.x, clip.y, clip.right(), clip.bottom()]),
                    layer: image.layer,
                }));
            }
            UiDrawCommand::Text(text) => {
                let align_h = match text.align_h {
                    UiTextAlign::Start => TextAlignH::Left,
                    UiTextAlign::Center => TextAlignH::Center,
                    UiTextAlign::End => TextAlignH::Right,
                };
                let align_v = match text.align_v {
                    UiTextAlign::Start => TextAlignV::Top,
                    UiTextAlign::Center => TextAlignV::Center,
                    UiTextAlign::End => TextAlignV::Bottom,
                };

                data.commands.push(UiRenderCommand::Text(UiRenderText {
                    id: text.id.0,
                    rect: [text.rect.x, text.rect.y, text.rect.width, text.rect.height],
                    text: text.text.into_string(),
                    color: text.color.to_array(),
                    font_size: text.font_size,
                    align_h,
                    align_v,
                    wrap: text.wrap,
                    cursor: text.cursor,
                    show_caret: text.show_caret,
                    caret_color: text.caret_color.map(|color| color.to_array()),
                    selection: text.selection.map(|(start, end)| [start, end]),
                    selection_color: text.selection_color.map(|color| color.to_array()),
                    layer: text.layer,
                    clip_rect: text
                        .clip
                        .map(|clip| [clip.x, clip.y, clip.right(), clip.bottom()]),
                }));
            }
        }
    }
}

pub fn publish_ui_render_state(world: &mut bevy_ecs::world::World) {
    world.resource_scope::<UiRenderState, _>(|world, mut published| {
        if let Some(mut pending) = world.get_resource_mut::<UiRenderFrameOutput>() {
            std::mem::swap(&mut published.data, &mut pending.data);
            published.revision = pending.revision;
            published.command_hash = pending.command_hash;
            published.wants_pointer_input = pending.wants_pointer_input;
            published.wants_keyboard_input = pending.wants_keyboard_input;
        }
    });
}
