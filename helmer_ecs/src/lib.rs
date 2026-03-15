use crate::{
    components::{
        ActiveCamera, Camera, Light, MeshRenderer, SkinnedMeshRenderer, Spline, SplineFollower,
        Transform,
    },
    ecs::component::Component,
    egui_integration::{EguiResource, EguiSystem},
    physics::{
        physics_resource::PhysicsResource,
        systems::{
            CleanupPhysicsSystem, PhysicsStepSystem, SyncEntitiesToPhysicsSystem,
            SyncPhysicsToEntitiesSystem,
        },
    },
    systems::{
        renderer_system::{RenderDataSystem, RenderPacket},
        scene_system::SceneSpawningSystem,
        spline_system::SplineFollowSystem,
    },
};

use std::{
    any::TypeId,
    collections::HashSet,
    sync::{Arc, mpsc},
    time::Duration,
};

use helmer_animation::Animator;
use helmer_asset::runtime::asset_server::AssetServer;
use helmer_render::runtime::{RuntimeConfig, RuntimeTuning};
use helmer_window::runtime::{input_manager::InputManager, runtime::Runtime};
use helmer_window::service::{WindowCallbacks, WindowService};
use parking_lot::RwLock;

use crate::ecs::{ecs_core::ECSCore, system_scheduler::SystemScheduler};

impl Component for Transform {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for MeshRenderer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for Camera {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for ActiveCamera {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for Light {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for SkinnedMeshRenderer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for Animator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for Spline {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for SplineFollower {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub mod components;
pub mod ecs;
pub mod egui_integration;
pub mod physics;
pub mod provided;
pub mod systems;

enum EcsLogicEvent {
    CloseRequested(Option<mpsc::Sender<()>>),
    Started {
        state: helmer_window::event::WindowState,
    },
    Resized(helmer_window::event::WindowState),
    Tick(f32),
}

struct EcsLogicState {
    ecs_core: ECSCore,
    scheduler: SystemScheduler,
    input_manager: Arc<RwLock<InputManager>>,
    metrics: Arc<helmer::runtime::PerformanceMetrics>,
    runtime: Runtime,
    runtime_tuning: Arc<RuntimeTuning>,
    runtime_window_control: Arc<helmer_window::runtime::runtime::RuntimeWindowControl>,
    render_sender:
        crossbeam_channel::Sender<helmer_render::graphics::common::renderer::RenderMessage>,
    logic_clock: helmer::runtime::LogicClock,
    fixed_timestep: bool,
    pending_resize: Option<(helmer_window::event::WindowState, web_time::Instant)>,
    last_window_state: Option<helmer_window::event::WindowState>,
    tps_window_start: web_time::Instant,
    tps_steps_accum: u64,
    applied_task_workers: usize,
}

impl EcsLogicState {
    fn new(
        ecs_core: ECSCore,
        scheduler: SystemScheduler,
        input_manager: Arc<RwLock<InputManager>>,
        metrics: Arc<helmer::runtime::PerformanceMetrics>,
        runtime: Runtime,
        runtime_tuning: Arc<RuntimeTuning>,
        runtime_window_control: Arc<helmer_window::runtime::runtime::RuntimeWindowControl>,
        render_sender: crossbeam_channel::Sender<
            helmer_render::graphics::common::renderer::RenderMessage,
        >,
    ) -> Self {
        let fixed_timestep = RuntimeConfig::default().fixed_timestep;
        let target_tickrate = RuntimeConfig::default().target_tickrate;
        let applied_task_workers = if runtime.context().is_single_threaded() {
            0
        } else {
            runtime.context().task_pool().worker_count()
        };
        Self {
            ecs_core,
            scheduler,
            input_manager,
            metrics,
            runtime,
            runtime_tuning,
            runtime_window_control,
            render_sender,
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
        self.ecs_core
            .component_pool
            .query_mut_for_each::<(Camera, ActiveCamera), _>(|_, (camera, _)| {
                camera.aspect_ratio = state.width as f32 / state.height as f32;
            });
        let resize = helmer_render::graphics::common::renderer::RenderMessage::Resize(
            winit::dpi::PhysicalSize::new(state.width, state.height),
        );
        if self.runtime.context().is_single_threaded() {
            let _ = self.render_sender.try_send(resize);
        } else {
            let _ = self.render_sender.send(resize);
        }
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
        if let Some(runtime_config) = self.ecs_core.get_resource::<RuntimeConfig>() {
            if runtime_config.fixed_timestep != self.fixed_timestep {
                self.fixed_timestep = runtime_config.fixed_timestep;
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
                self.ecs_core
                    .resource_scope::<RuntimeConfig, _>(|ecs_core, runtime_config| {
                        ecs_core.resource_scope::<EguiResource, _>(|_ecs, egui_resource| {
                            if runtime_config.egui {
                                if input_manager.active_mouse_buttons.is_empty() {
                                    input_manager.egui_wants_pointer =
                                        egui_resource.ctx.wants_pointer_input();
                                }
                                input_manager.egui_wants_key =
                                    egui_resource.ctx.wants_keyboard_input();
                            } else if egui_resource.accepting_input {
                                input_manager.clear_egui_state();
                            }
                        });
                    });
                self.scheduler
                    .run_all(step_dt, &mut self.ecs_core, &input_manager);
            }

            latest_render_delta = self
                .ecs_core
                .get_resource_mut::<RenderPacket>()
                .and_then(|packet| packet.0.take());
            latest_egui_data = self
                .ecs_core
                .get_resource_mut::<EguiResource>()
                .and_then(|egui_res| egui_res.render_data.take());

            self.input_manager.write().prepare_for_next_frame();
        }

        if let Some(delta) = latest_render_delta {
            let _ = self.render_sender.try_send(
                helmer_render::graphics::common::renderer::RenderMessage::RenderDelta(delta),
            );
        }
        if let Some(data) = latest_egui_data {
            let _ = self
                .render_sender
                .try_send(helmer_render::graphics::common::renderer::RenderMessage::EguiData(data));
        }
    }

    fn handle_event(&mut self, event: EcsLogicEvent) {
        match event {
            EcsLogicEvent::CloseRequested(done) => {
                let _ = self.runtime.shutdown();
                if let Some(done) = done {
                    let _ = done.send(());
                }
            }
            EcsLogicEvent::Started { state } => {
                self.pending_resize = None;
                self.last_window_state = Some(state);
            }
            EcsLogicEvent::Resized(state) => {
                if self.should_debounce_resize(state) {
                    self.pending_resize = Some((state, web_time::Instant::now()));
                } else {
                    self.pending_resize = None;
                    self.apply_window_resize(state);
                }
            }
            EcsLogicEvent::Tick(dt) => self.handle_tick(dt),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_ecs_logic_thread(mut state: EcsLogicState, receiver: mpsc::Receiver<EcsLogicEvent>) {
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
                let should_stop = matches!(event, EcsLogicEvent::CloseRequested(_));
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

pub fn helmer_ecs_init(init_callback: fn(&mut ECSCore, &mut SystemScheduler, &AssetServer)) {
    helmer::runtime::init_runtime_tracing();

    let mut ecs_core = ECSCore::new();
    let mut scheduler = SystemScheduler::new();
    let input_manager = Arc::new(RwLock::new(InputManager::new()));
    let metrics = Arc::new(helmer::runtime::PerformanceMetrics::default());

    let mut runtime = Runtime::builder()
        .build()
        .expect("failed to build helmer runtime");
    runtime
        .context()
        .resources()
        .insert(helmer::runtime::RuntimePerformanceMetricsResource(
            metrics.clone(),
        ));
    runtime
        .register_extension(helmer_input::extension::InputExtension::with_manager(
            input_manager.clone(),
        ))
        .expect("failed to register helmer_input extension");
    runtime
        .register_extension(helmer_render::extension::RenderExtension::new())
        .expect("failed to register helmer_render extension");
    runtime
        .register_extension(helmer_asset::extension::AssetExtension::new())
        .expect("failed to register helmer_asset extension");
    runtime
        .register_extension(helmer_audio::extension::AudioExtension::new())
        .expect("failed to register helmer_audio extension");
    runtime
        .register_extension(helmer_window::extension::WindowExtension::new(
            WindowService::with_input_manager(
                helmer_window::config::RuntimeConfig::default(),
                input_manager.clone(),
            )
            .with_metrics(metrics.clone()),
        ))
        .expect("failed to register helmer_window extension");
    runtime.start().expect("failed to start helmer runtime");

    let resources = runtime.context().resources();
    let asset_server = resources
        .get::<helmer_asset::extension::AssetServerResource>()
        .map(|resource| resource.0.clone())
        .expect("AssetExtension did not register AssetServerResource");
    let render_sender = resources
        .get::<helmer_render::extension::RenderMessageSender>()
        .map(|resource| resource.0.clone())
        .expect("RenderExtension did not register RenderMessageSender");
    let runtime_tuning = resources
        .get::<helmer_render::extension::RenderRuntimeTuningResource>()
        .map(|resource| resource.0.clone())
        .expect("RenderExtension did not register RuntimeTuning");
    let runtime_window_control = resources
        .get::<helmer_window::extension::WindowControlResource>()
        .map(|resource| resource.0.clone())
        .expect("WindowExtension did not register RuntimeWindowControl");
    let window_service = resources
        .get::<helmer_window::extension::WindowMainThreadServiceResource>()
        .and_then(|resource| resource.0.lock().take())
        .expect("WindowExtension did not register WindowService");

    ecs_core.add_resource(RuntimeConfig::default());
    ecs_core.add_resource(asset_server.clone());
    ecs_core.add_resource(input_manager.clone());
    ecs_core.add_resource(RenderPacket::default());
    ecs_core.add_resource(PhysicsResource::new());
    ecs_core.add_resource(EguiResource::default());
    ecs_core.add_resource(metrics.clone());

    scheduler.register_system(
        SceneSpawningSystem {},
        25,
        vec![],
        HashSet::from([TypeId::of::<Transform>()]),
        HashSet::from([TypeId::of::<Transform>()]),
    );
    scheduler.register_system(
        SyncEntitiesToPhysicsSystem {},
        20,
        vec![],
        HashSet::from([TypeId::of::<Transform>()]),
        HashSet::from([TypeId::of::<Transform>()]),
    );
    scheduler.register_system(
        PhysicsStepSystem {},
        10,
        vec![],
        HashSet::from([TypeId::of::<Transform>()]),
        HashSet::from([TypeId::of::<Transform>()]),
    );
    scheduler.register_system(
        SyncPhysicsToEntitiesSystem {},
        5,
        vec![],
        HashSet::from([TypeId::of::<Transform>()]),
        HashSet::from([TypeId::of::<Transform>()]),
    );
    scheduler.register_system(
        CleanupPhysicsSystem::default(),
        4,
        vec![],
        HashSet::from([TypeId::of::<Transform>()]),
        HashSet::from([TypeId::of::<Transform>()]),
    );
    scheduler.register_system(
        SplineFollowSystem {},
        3,
        vec![],
        HashSet::from([TypeId::of::<Transform>()]),
        HashSet::from([TypeId::of::<Transform>()]),
    );
    scheduler.register_system(
        EguiSystem {},
        1,
        vec![],
        HashSet::new(),
        HashSet::from([TypeId::of::<EguiResource>()]),
    );
    scheduler.register_system(
        RenderDataSystem::new(),
        0,
        vec![],
        HashSet::from([TypeId::of::<Transform>()]),
        HashSet::from([TypeId::of::<Transform>()]),
    );

    init_callback(&mut ecs_core, &mut scheduler, &asset_server.lock());
    #[cfg(not(target_arch = "wasm32"))]
    let render_sender_for_window_events = render_sender.clone();

    let logic_state = EcsLogicState::new(
        ecs_core,
        scheduler,
        input_manager,
        metrics,
        runtime,
        runtime_tuning,
        runtime_window_control,
        render_sender,
    );

    #[cfg(not(target_arch = "wasm32"))]
    let threaded_logic = !logic_state.runtime.context().is_single_threaded();
    #[cfg(not(target_arch = "wasm32"))]
    let (mut logic_sender, mut logic_thread, mut inline_logic_state) = if threaded_logic {
        let (sender, receiver) = mpsc::channel::<EcsLogicEvent>();
        let join = std::thread::Builder::new()
            .name("helmer-logic".to_string())
            .spawn(move || run_ecs_logic_thread(logic_state, receiver))
            .expect("failed to spawn helmer logic thread");
        (Some(sender), Some(join), None)
    } else {
        (None, None, Some(logic_state))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let logic_sender_for_events = logic_sender.clone();
    #[cfg(not(target_arch = "wasm32"))]
    let mut pending_render_bootstrap: Option<(
        Arc<winit::window::Window>,
        helmer_window::event::WindowState,
    )> = None;
    #[cfg(not(target_arch = "wasm32"))]
    let mut next_render_bootstrap_retry_at: Option<web_time::Instant> = None;

    #[cfg(target_arch = "wasm32")]
    let mut inline_logic_state = Some(logic_state);

    let mut callbacks = WindowCallbacks {
        on_event: Some(Box::new(move |event| {
            let logic_event = match &event.kind {
                helmer_window::event::WindowRuntimeEventKind::CloseRequested => {
                    Some(EcsLogicEvent::CloseRequested(None))
                }
                helmer_window::event::WindowRuntimeEventKind::Started { window, state } => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        pending_render_bootstrap = Some((window.clone(), *state));
                        next_render_bootstrap_retry_at = None;
                    }
                    Some(EcsLogicEvent::Started { state: *state })
                }
                helmer_window::event::WindowRuntimeEventKind::Resized(state) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if let Some((_, pending_state)) = pending_render_bootstrap.as_mut() {
                            *pending_state = *state;
                        }
                    }
                    Some(EcsLogicEvent::Resized(*state))
                }
                helmer_window::event::WindowRuntimeEventKind::Tick { dt } => {
                    Some(EcsLogicEvent::Tick(*dt))
                }
                _ => None,
            };

            #[cfg(not(target_arch = "wasm32"))]
            {
                let now = web_time::Instant::now();
                let retry_allowed = match next_render_bootstrap_retry_at {
                    Some(deadline) => now >= deadline,
                    None => true,
                };
                if retry_allowed && let Some((window, state)) = pending_render_bootstrap.as_ref() {
                    match helmer_render::extension::create_native_render_bootstrap_message(
                        window.clone(),
                        winit::dpi::PhysicalSize::new(state.width, state.height),
                    ) {
                        Ok(message) => match render_sender_for_window_events.send(message) {
                            Ok(()) => {
                                tracing::debug!(
                                    width = state.width,
                                    height = state.height,
                                    scale = state.scale_factor,
                                    "sent main-thread render bootstrap message"
                                );
                                pending_render_bootstrap = None;
                                next_render_bootstrap_retry_at = None;
                            }
                            Err(err) => {
                                tracing::warn!(
                                    "failed to send main-thread render bootstrap message: {err}"
                                );
                                pending_render_bootstrap = None;
                                next_render_bootstrap_retry_at = None;
                            }
                        },
                        Err(err) => {
                            let backoff_ms = 50u64;
                            next_render_bootstrap_retry_at =
                                Some(now + Duration::from_millis(backoff_ms));
                            tracing::warn!("failed to create main-thread render bootstrap: {err}");
                        }
                    }
                }
            }

            if let Some(logic_event) = logic_event {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(sender) = logic_sender_for_events.as_ref() {
                        if matches!(logic_event, EcsLogicEvent::Tick(_)) {
                            return;
                        }
                        if matches!(logic_event, EcsLogicEvent::CloseRequested(_)) {
                            let (done_tx, done_rx) = mpsc::channel();
                            let _ = sender.send(EcsLogicEvent::CloseRequested(Some(done_tx)));
                            let _ = done_rx.recv_timeout(Duration::from_secs(2));
                            return;
                        }
                        if sender.send(logic_event).is_err() {
                            tracing::warn!("logic event channel disconnected");
                        }
                    } else if let Some(state) = inline_logic_state.as_mut() {
                        state.handle_event(logic_event);
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(state) = inline_logic_state.as_mut() {
                        state.handle_event(logic_event);
                    }
                }
            }
        })),
    };

    window_service
        .run(std::mem::take(&mut callbacks))
        .expect("failed to run window service");

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(sender) = logic_sender.take() {
            let _ = sender.send(EcsLogicEvent::CloseRequested(None));
        }
        if let Some(join) = logic_thread.take() {
            let _ = join.join();
        }
    }
}
