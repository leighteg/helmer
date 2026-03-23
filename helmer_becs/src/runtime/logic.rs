use crossbeam_channel::Sender;
use std::{
    path::PathBuf,
    sync::{Arc, mpsc},
    time::Duration,
};

use bevy_ecs::{schedule::Schedule, world::World};
use helmer_render::{
    graphics::common::renderer::RenderMessage,
    runtime::{RuntimeConfig, RuntimeProfiling, RuntimeTuning},
};
use helmer_window::runtime::{
    input_manager::InputManager,
    runtime::{PerformanceMetrics, Runtime, RuntimeWindowControl},
};
use parking_lot::RwLock;

use crate::{
    components::{ActiveCamera, Camera},
    egui_integration::{EguiInputPassthrough, EguiResource},
    resources::{BecsRuntimeConfig, DeltaTime, DraggedFile},
    systems::render_system::RenderPacket,
    ui_integration::UiRenderState,
};

pub(super) enum BecsLogicEvent {
    CloseRequested(Option<mpsc::Sender<()>>),
    Started {
        state: helmer_window::event::WindowState,
    },
    Resized(helmer_window::event::WindowState),
    DroppedFile(PathBuf),
    Tick(f32),
}

pub(super) struct BecsLogicState {
    world: World,
    schedule: Schedule,
    input_manager: Arc<RwLock<InputManager>>,
    metrics: Arc<PerformanceMetrics>,
    runtime: Runtime,
    runtime_tuning: Arc<RuntimeTuning>,
    runtime_profiling: Arc<RuntimeProfiling>,
    runtime_window_control: Arc<RuntimeWindowControl>,
    render_sender: Sender<RenderMessage>,
    render_control_sender: Sender<RenderMessage>,
    logic_clock: helmer::runtime::LogicClock,
    fixed_timestep: bool,
    pending_resize: Option<(helmer_window::event::WindowState, web_time::Instant)>,
    last_window_state: Option<helmer_window::event::WindowState>,
    tps_window_start: web_time::Instant,
    tps_steps_accum: u64,
    applied_task_workers: usize,
}

impl BecsLogicState {
    pub(super) fn new(
        world: World,
        schedule: Schedule,
        input_manager: Arc<RwLock<InputManager>>,
        metrics: Arc<PerformanceMetrics>,
        runtime: Runtime,
        runtime_tuning: Arc<RuntimeTuning>,
        runtime_profiling: Arc<RuntimeProfiling>,
        runtime_window_control: Arc<RuntimeWindowControl>,
        render_sender: Sender<RenderMessage>,
        render_control_sender: Sender<RenderMessage>,
    ) -> Self {
        let fixed_timestep = RuntimeConfig::default().fixed_timestep;
        let target_tickrate = RuntimeConfig::default().target_tickrate;
        let applied_task_workers = if runtime.context().is_single_threaded() {
            0
        } else {
            runtime.context().task_pool().worker_count()
        };
        Self {
            world,
            schedule,
            input_manager,
            metrics,
            runtime,
            runtime_tuning,
            runtime_profiling,
            runtime_window_control,
            render_sender,
            render_control_sender,
            logic_clock: helmer::runtime::LogicClock::new(target_tickrate, fixed_timestep),
            fixed_timestep,
            pending_resize: None,
            last_window_state: None,
            tps_window_start: web_time::Instant::now(),
            tps_steps_accum: 0,
            applied_task_workers,
        }
    }

    fn apply_window_resize(&mut self, state: helmer_window::event::WindowState) {
        for (mut camera, _) in self
            .world
            .query::<(&mut Camera, &ActiveCamera)>()
            .iter_mut(&mut self.world)
        {
            camera.aspect_ratio = state.width as f32 / state.height as f32;
        }
        let resize =
            RenderMessage::Resize(winit::dpi::PhysicalSize::new(state.width, state.height));
        let _ = self.render_control_sender.send(resize);
        self.last_window_state = Some(state);
    }

    fn should_debounce_resize(&self, state: helmer_window::event::WindowState) -> bool {
        let debounce_ms = self
            .runtime_tuning
            .resize_debounce_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        if debounce_ms == 0 {
            return false;
        }
        if self.runtime_window_control.fullscreen() || self.runtime_window_control.maximized() {
            return false;
        }
        if let Some(previous) = self.last_window_state {
            let width_delta = state.width.abs_diff(previous.width);
            let height_delta = state.height.abs_diff(previous.height);
            let max_dim = previous
                .width
                .max(previous.height)
                .max(state.width.max(state.height))
                .max(1);
            let snap_threshold = ((max_dim as f32) * 0.30).round() as u32;
            if width_delta >= snap_threshold || height_delta >= snap_threshold {
                return false;
            }
        }
        true
    }

    fn handle_tick(&mut self, dt: f32) {
        let now = web_time::Instant::now();
        self.runtime_window_control.set_title_update_ms(
            self.runtime_tuning
                .title_update_ms
                .load(std::sync::atomic::Ordering::Relaxed)
                .max(1),
        );
        self.runtime_window_control
            .set_target_tickrate(self.runtime_tuning.load_target_tickrate());
        self.runtime_window_control
            .set_target_fps(self.runtime_tuning.load_target_fps());

        let requested_task_workers = self.runtime_tuning.load_task_worker_count();
        let desired_task_workers = if self.runtime.context().is_single_threaded() {
            0
        } else {
            requested_task_workers
        };
        if desired_task_workers != self.applied_task_workers {
            match self
                .runtime
                .context()
                .task_pool()
                .set_worker_count(desired_task_workers)
            {
                Ok(()) => {
                    self.applied_task_workers = desired_task_workers;
                }
                Err(err) => {
                    tracing::warn!(
                        requested_workers = requested_task_workers,
                        applied_workers = desired_task_workers,
                        reason = %err,
                        "failed to apply runtime task worker tuning"
                    );
                }
            }
        }

        let target_tickrate = self.runtime_tuning.load_target_tickrate();
        if let Some(runtime_config) = self.world.get_resource::<BecsRuntimeConfig>() {
            if runtime_config.0.fixed_timestep != self.fixed_timestep {
                self.fixed_timestep = runtime_config.0.fixed_timestep;
                self.logic_clock =
                    helmer::runtime::LogicClock::new(target_tickrate, self.fixed_timestep);
            } else {
                self.logic_clock.set_tickrate(target_tickrate, now);
            }
        } else {
            self.logic_clock.set_tickrate(target_tickrate, now);
        }

        if let Some((state, queued_at)) = self.pending_resize {
            let debounce_ms = self
                .runtime_tuning
                .resize_debounce_ms
                .load(std::sync::atomic::Ordering::Relaxed);
            if debounce_ms == 0
                || now.saturating_duration_since(queued_at)
                    >= Duration::from_millis(debounce_ms as u64)
            {
                self.apply_window_resize(state);
                self.pending_resize = None;
            }
        }

        let max_steps = self
            .runtime_tuning
            .max_logic_steps_per_frame
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1);
        let frame = if self.fixed_timestep {
            self.logic_clock.advance(now, max_steps)
        } else {
            helmer::runtime::LogicFrame {
                steps: 1,
                dt: dt.max(0.0),
            }
        };
        let step_count = if self.fixed_timestep { frame.steps } else { 1 };
        self.tps_steps_accum = self.tps_steps_accum.saturating_add(step_count as u64);
        let tps_elapsed = now.saturating_duration_since(self.tps_window_start);
        if tps_elapsed >= Duration::from_secs(1) {
            let seconds = tps_elapsed.as_secs_f64().max(f64::EPSILON);
            let tps = ((self.tps_steps_accum as f64) / seconds)
                .round()
                .clamp(0.0, u32::MAX as f64) as u32;
            self.metrics.set_tps(tps);
            self.tps_steps_accum = 0;
            self.tps_window_start = now;
        }
        if step_count == 0 {
            return;
        }

        let mut latest_render_delta = None;
        let mut latest_egui_data = None;

        for _ in 0..step_count {
            let step_dt = if self.fixed_timestep { frame.dt } else { dt };
            let _ = self
                .runtime
                .tick(std::time::Duration::from_secs_f32(step_dt.max(0.0)));

            {
                let mut input_manager = self.input_manager.write();
                input_manager.process_events();
                input_manager.ui_wants_pointer = self
                    .world
                    .get_resource::<UiRenderState>()
                    .map(|state| state.wants_pointer_input)
                    .unwrap_or(false);

                self.world
                    .resource_scope::<BecsRuntimeConfig, _>(|world, runtime_config| {
                        let runtime_config = runtime_config.0;
                        world.resource_scope::<EguiResource, _>(|world, egui_resource| {
                            let passthrough = world
                                .get_resource::<EguiInputPassthrough>()
                                .copied()
                                .unwrap_or_default();
                            if runtime_config.egui {
                                input_manager.egui_wants_pointer =
                                    egui_resource.ctx.wants_pointer_input() && !passthrough.pointer;
                                input_manager.egui_wants_key =
                                    egui_resource.ctx.wants_keyboard_input()
                                        && !passthrough.keyboard;
                            } else if egui_resource.accepting_input {
                                input_manager.clear_egui_state();
                            }
                        });
                    });
            }

            self.world.resource_mut::<DeltaTime>().0 = step_dt;
            let profiling_enabled = self
                .runtime_profiling
                .enabled
                .load(std::sync::atomic::Ordering::Relaxed);
            let schedule_start = if profiling_enabled {
                Some(web_time::Instant::now())
            } else {
                None
            };
            self.schedule.run(&mut self.world);
            crate::ui_integration::publish_ui_render_state(&mut self.world);
            if let Some(start) = schedule_start {
                self.runtime_profiling.logic_schedule_us.store(
                    start.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

            latest_render_delta = self
                .world
                .get_resource_mut::<RenderPacket>()
                .and_then(|mut packet| packet.0.take());
            latest_egui_data = self
                .world
                .get_resource_mut::<EguiResource>()
                .and_then(|mut egui_res| egui_res.render_data.take());

            self.input_manager.write().prepare_for_next_frame();
        }

        if let Some(delta) = latest_render_delta {
            let _ = self
                .render_sender
                .try_send(RenderMessage::RenderDelta(delta));
        }
        if let Some(data) = latest_egui_data {
            let _ = self.render_sender.try_send(RenderMessage::EguiData(data));
        }
    }

    pub(super) fn handle_event(&mut self, event: BecsLogicEvent) {
        match event {
            BecsLogicEvent::CloseRequested(done) => {
                let _ = self.runtime.shutdown();
                if let Some(done) = done {
                    let _ = done.send(());
                }
            }
            BecsLogicEvent::Started { state } => {
                self.pending_resize = None;
                self.last_window_state = Some(state);
            }
            BecsLogicEvent::Resized(state) => {
                if self.should_debounce_resize(state) {
                    self.pending_resize = Some((state, web_time::Instant::now()));
                } else {
                    self.pending_resize = None;
                    self.apply_window_resize(state);
                }
            }
            BecsLogicEvent::DroppedFile(path) => {
                if let Some(mut dragged_file_res) = self.world.get_resource_mut::<DraggedFile>() {
                    dragged_file_res.0 = Some(path);
                }
            }
            BecsLogicEvent::Tick(dt) => self.handle_tick(dt),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn run_becs_logic_thread(
    mut state: BecsLogicState,
    receiver: mpsc::Receiver<BecsLogicEvent>,
) {
    let _high_resolution_timer = helmer::runtime::HighResolutionTimerGuard::new(1);

    let mut last_tick = web_time::Instant::now();
    let mut next_tick_deadline = last_tick;
    loop {
        let now = web_time::Instant::now();
        if now >= next_tick_deadline {
            let dt = now.saturating_duration_since(last_tick).as_secs_f32();
            last_tick = now;
            state.handle_tick(dt.max(0.0));
            let tick_interval =
                Duration::from_secs_f32(1.0 / state.runtime_tuning.load_target_tickrate().max(1.0));
            while next_tick_deadline <= now {
                next_tick_deadline += tick_interval;
            }
            continue;
        }

        let timeout = next_tick_deadline - now;
        match receiver.recv_timeout(timeout) {
            Ok(event) => {
                let should_stop = matches!(event, BecsLogicEvent::CloseRequested(_));
                state.handle_event(event);
                if should_stop {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}
