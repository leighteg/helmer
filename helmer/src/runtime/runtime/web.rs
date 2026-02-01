use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use egui::ViewportId;
use glam::{DVec2, UVec2};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber, info, warn};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use wasm_bindgen_futures::spawn_local;
use web_time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use super::common::{LogicClock, PerformanceMetrics, RuntimeProfiling, RuntimeTuning};
use crate::{
    graphics::{
        backend::binding_backend::BindingBackendChoice,
        common::renderer::{
            AssetStreamingRequest, EguiRenderData, EguiTextureCache, RenderControl, RenderDelta,
            RenderMessage, RendererStats, WgpuBackend, apply_egui_delta, build_full_egui_delta,
            initialize_renderer, render_message_payload_bytes,
        },
        renderer::GraphRenderer,
    },
    runtime::{
        asset_server::AssetServer,
        config::RuntimeConfig,
        input_manager::{InputEvent, InputManager},
    },
};

struct RenderInit {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
}

type RenderInitResult = Result<WebRenderState, String>;

#[derive(Default)]
struct ConsoleVisitor {
    message: Option<String>,
    fields: Vec<String>,
}

impl Visit for ConsoleVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{value:?}"));
        } else {
            self.fields.push(format!("{}={value:?}", field.name()));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else {
            self.fields.push(format!("{}={}", field.name(), value));
        }
    }
}

struct ConsoleLayer;

impl<S> Layer<S> for ConsoleLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let mut visitor = ConsoleVisitor::default();
        event.record(&mut visitor);

        let mut message = visitor.message.unwrap_or_default();
        if !visitor.fields.is_empty() {
            if !message.is_empty() {
                message.push(' ');
            }
            message.push_str(&visitor.fields.join(" "));
        }
        if message.is_empty() {
            message = meta.name().to_string();
        }

        let full = if meta.target().is_empty() {
            message
        } else {
            format!("[{}] {}", meta.target(), message)
        };

        match *meta.level() {
            Level::ERROR => web_sys::console::error_1(&full.into()),
            Level::WARN => web_sys::console::warn_1(&full.into()),
            Level::INFO => web_sys::console::info_1(&full.into()),
            _ => web_sys::console::log_1(&full.into()),
        }
    }
}

fn log_warn(message: &str) {
    web_sys::console::warn_1(&message.into());
    warn!("{}", message);
}

struct RuntimeCallbacks<T: 'static> {
    init: Option<Box<dyn FnOnce(&mut Runtime<T>, &mut T)>>,
    tick: Arc<dyn Fn(f32, &mut T) -> (Option<RenderDelta>, Option<EguiRenderData>)>,
    resize: Arc<dyn Fn(PhysicalSize<u32>, &mut T)>,
    dropped_file: Arc<dyn Fn(PathBuf, &mut T)>,
}

#[derive(Debug, Clone)]
struct WindowSettingsCache {
    attributes: WindowAttributes,
    minimized: Option<bool>,
}

impl WindowSettingsCache {
    fn new(title: impl Into<String>) -> Self {
        let mut attributes = Window::default_attributes();
        attributes.title = title.into();
        Self {
            attributes,
            minimized: None,
        }
    }

    fn clone_attributes(&self) -> WindowAttributes {
        self.attributes.clone()
    }

    fn set_attributes(&mut self, attributes: WindowAttributes) {
        self.attributes = attributes;
    }

    fn title(&self) -> &str {
        &self.attributes.title
    }

    fn set_title(&mut self, title: String) {
        self.attributes.title = title;
    }

    fn set_inner_size(&mut self, size: PhysicalSize<u32>) {
        self.attributes.inner_size = Some(size.into());
    }

    fn set_outer_position(&mut self, position: PhysicalPosition<i32>) {
        self.attributes.position = Some(position.into());
    }

    fn sync_geometry_from_window(&mut self, window: &Window) {
        self.attributes.inner_size = Some(window.inner_size().into());
        if let Ok(position) = window.outer_position() {
            self.attributes.position = Some(position.into());
        }
    }

    fn apply_post_create(&self, window: &Window) {
        if let Some(minimized) = self.minimized {
            window.set_minimized(minimized);
        }
    }
}

struct WebRenderState {
    renderer: GraphRenderer,
    surface_size: PhysicalSize<u32>,
    last_render: Instant,
    asset_backlog: VecDeque<RenderMessage>,
    asset_backlog_bytes: usize,
    immediate_backlog: VecDeque<(RenderMessage, usize)>,
    upload_batch: Vec<RenderMessage>,
    upload_bytes: Vec<usize>,
    poll_frame: u32,
}

impl WebRenderState {
    fn new(renderer: GraphRenderer, surface_size: PhysicalSize<u32>) -> Self {
        Self {
            renderer,
            surface_size,
            last_render: Instant::now(),
            asset_backlog: VecDeque::new(),
            asset_backlog_bytes: 0,
            immediate_backlog: VecDeque::new(),
            upload_batch: Vec::new(),
            upload_bytes: Vec::new(),
            poll_frame: 0,
        }
    }

    fn render_frame(
        &mut self,
        frame_start: Instant,
        render_receiver: &Receiver<RenderMessage>,
        asset_receiver: &Receiver<RenderMessage>,
        tuning: &RuntimeTuning,
        metrics: &PerformanceMetrics,
        profiling: &RuntimeProfiling,
        target_fps: Option<f32>,
    ) -> bool {
        let profiling_enabled = profiling.enabled.load(Ordering::Relaxed);
        let mut should_render = false;
        let messages_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };

        let dt = frame_start.duration_since(self.last_render).as_secs_f32();
        if dt > 0.0 {
            let fps = (1.0 / dt).round() as u32;
            metrics.fps.store(fps, Ordering::Relaxed);
        }

        let max_pending = tuning.max_pending_asset_uploads.load(Ordering::Relaxed);
        let max_pending_bytes = tuning.max_pending_asset_bytes.load(Ordering::Relaxed);
        let backlog_enabled = max_pending > 0 && max_pending_bytes > 0;
        if !backlog_enabled && !self.asset_backlog.is_empty() {
            for message in self.asset_backlog.drain(..) {
                tuning.release_asset_upload(render_message_payload_bytes(&message));
            }
            self.asset_backlog_bytes = 0;
        }

        loop {
            match render_receiver.try_recv() {
                Ok(message) => match message {
                    message @ RenderMessage::CreateMesh { .. }
                    | message @ RenderMessage::CreateTexture { .. }
                    | message @ RenderMessage::CreateMaterial(_) => {
                        let message_bytes = render_message_payload_bytes(&message);
                        let fits_bytes = self.asset_backlog_bytes.saturating_add(message_bytes)
                            <= max_pending_bytes;
                        if backlog_enabled && self.asset_backlog.len() < max_pending && fits_bytes {
                            self.asset_backlog_bytes =
                                self.asset_backlog_bytes.saturating_add(message_bytes);
                            self.asset_backlog.push_back(message);
                        } else {
                            self.immediate_backlog.push_back((message, message_bytes));
                        }
                        should_render = true;
                    }
                    RenderMessage::RenderData(data) => {
                        should_render = true;
                        self.renderer
                            .process_message(RenderMessage::RenderData(data));
                    }
                    RenderMessage::RenderDelta(delta) => {
                        should_render = true;
                        self.renderer
                            .process_message(RenderMessage::RenderDelta(delta));
                    }
                    RenderMessage::Resize(size) => {
                        should_render = true;
                        self.surface_size = size;
                        self.renderer.process_message(RenderMessage::Resize(size));
                    }
                    RenderMessage::Control(ctrl) => {
                        should_render = true;
                        match ctrl {
                            RenderControl::RecreateDevice { .. } => {
                                warn!(
                                    "device recreation is not supported on web yet; request ignored"
                                );
                            }
                            _ => self.renderer.process_message(RenderMessage::Control(ctrl)),
                        }
                    }
                    RenderMessage::Shutdown => {
                        self.renderer.process_message(RenderMessage::Shutdown);
                        return false;
                    }
                    other => {
                        should_render = true;
                        self.renderer.process_message(other);
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return false,
            }
        }

        loop {
            if backlog_enabled && self.asset_backlog.len() >= max_pending {
                break;
            }
            match asset_receiver.try_recv() {
                Ok(message) => match message {
                    message @ RenderMessage::CreateMesh { .. }
                    | message @ RenderMessage::CreateTexture { .. }
                    | message @ RenderMessage::CreateMaterial(_) => {
                        let message_bytes = render_message_payload_bytes(&message);
                        let fits_bytes = self.asset_backlog_bytes.saturating_add(message_bytes)
                            <= max_pending_bytes;
                        if backlog_enabled && fits_bytes {
                            self.asset_backlog_bytes =
                                self.asset_backlog_bytes.saturating_add(message_bytes);
                            self.asset_backlog.push_back(message);
                        } else {
                            self.immediate_backlog.push_back((message, message_bytes));
                        }
                        should_render = true;
                    }
                    other => {
                        should_render = true;
                        self.renderer.process_message(other);
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        let uploads_per_frame = tuning.asset_uploads_per_frame.load(Ordering::Relaxed);
        let mut uploads_this_frame = 0usize;
        let upload_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        self.upload_batch.clear();
        self.upload_bytes.clear();
        while uploads_this_frame < uploads_per_frame {
            if let Some((msg, message_bytes)) = self.immediate_backlog.pop_front() {
                self.upload_batch.push(msg);
                self.upload_bytes.push(message_bytes);
                uploads_this_frame += 1;
                continue;
            }
            if let Some(msg) = self.asset_backlog.pop_front() {
                let message_bytes = render_message_payload_bytes(&msg);
                self.asset_backlog_bytes = self.asset_backlog_bytes.saturating_sub(message_bytes);
                self.upload_batch.push(msg);
                self.upload_bytes.push(message_bytes);
                uploads_this_frame += 1;
                continue;
            }
            break;
        }
        if !self.upload_batch.is_empty() {
            self.renderer.process_asset_batch(&mut self.upload_batch);
            for bytes in self.upload_bytes.drain(..) {
                tuning.release_asset_upload(bytes);
            }
            should_render = true;
        }
        if let Some(start) = upload_start {
            profiling
                .render_thread_upload_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if let Some(start) = messages_start {
            profiling
                .render_thread_messages_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if let Some(target_fps) = target_fps {
            let frame_duration = Duration::from_secs_f32(1.0 / target_fps);
            let time_since_last_render = frame_start.duration_since(self.last_render);
            if should_render || time_since_last_render >= frame_duration {
                let render_start = if profiling_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                if let Err(e) = self.renderer.render() {
                    warn!("render error: {}", e);
                }
                if let Some(start) = render_start {
                    profiling
                        .render_thread_render_us
                        .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
                self.last_render = frame_start;
            }
        } else {
            let render_start = if profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            if let Err(e) = self.renderer.render() {
                warn!("render error: {}", e);
            }
            if let Some(start) = render_start {
                profiling
                    .render_thread_render_us
                    .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            self.last_render = frame_start;
        }

        let poll_interval = tuning.wgpu_poll_interval_frames.load(Ordering::Relaxed);
        let poll_mode = tuning.wgpu_poll_mode.load(Ordering::Relaxed);
        if poll_interval > 0 && poll_mode != 0 && (self.poll_frame % poll_interval == 0) {
            let poll_type = match poll_mode {
                2 => wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                },
                1 => wgpu::PollType::Poll,
                _ => wgpu::PollType::Poll,
            };
            self.renderer.poll_device(poll_type);
        }
        self.poll_frame = self.poll_frame.wrapping_add(1);

        if profiling_enabled {
            profiling
                .render_thread_frame_us
                .store(frame_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        true
    }
}

struct LogicState {
    pending_render_delta: Option<RenderDelta>,
    egui_cache: EguiTextureCache,
    egui_pending_full_upload: bool,
    egui_pending_free: Vec<egui::TextureId>,
}

pub struct Runtime<T: 'static = ()> {
    pub input_manager: Arc<RwLock<InputManager>>,
    pub asset_server: Option<Arc<Mutex<AssetServer>>>,
    asset_base_path: Option<String>,
    opfs_enabled: bool,

    pub render_thread_sender: Sender<RenderMessage>,
    render_receiver: Receiver<RenderMessage>,
    asset_receiver: Receiver<RenderMessage>,
    asset_stream_sender: Sender<AssetStreamingRequest>,

    egui_winit_state: Option<egui_winit::State>,

    window: Option<Arc<Window>>,
    window_settings: WindowSettingsCache,

    user_state: Option<Arc<Mutex<T>>>,
    callbacks: RuntimeCallbacks<T>,

    last_title_update: Instant,

    pub metrics: Arc<PerformanceMetrics>,
    pub config: Arc<RuntimeConfig>,
    pub renderer_stats: Arc<RendererStats>,
    pub tuning: Arc<RuntimeTuning>,
    pub profiling: Arc<RuntimeProfiling>,

    has_init: Arc<AtomicBool>,

    new_window_size: Option<PhysicalSize<u32>>,
    resize_triggered: bool,
    last_resize: Instant,

    logic_clock: LogicClock,
    logic_state: LogicState,
    render_state: Option<WebRenderState>,
    render_init_receiver: Option<Receiver<RenderInitResult>>,
}

impl<T: 'static> Runtime<T> {
    pub fn new(
        user_state: T,
        init_callback: impl FnOnce(&mut Runtime<T>, &mut T) + 'static,
        tick_callback: impl Fn(f32, &mut T) -> (Option<RenderDelta>, Option<EguiRenderData>) + 'static,
        resize_callback: impl Fn(PhysicalSize<u32>, &mut T) + 'static,
        dropped_file_callback: impl Fn(PathBuf, &mut T) + 'static,
    ) -> Self {
        let console_layer = ConsoleLayer;
        tracing_subscriber::registry()
            .with(console_layer)
            .with(tracing_subscriber::EnvFilter::new("helmer"))
            .try_init()
            .ok();

        info!("2026 leighton [https://leighteg.dev]");

        let tuning = Arc::new(RuntimeTuning::default());
        let profiling = Arc::new(RuntimeProfiling::default());
        let render_capacity = tuning.render_message_capacity.load(Ordering::Relaxed);
        let (render_sender, render_receiver) = bounded(render_capacity);
        let (asset_sender, asset_receiver) = bounded(render_capacity);
        let asset_stream_capacity = tuning.asset_stream_queue_capacity.load(Ordering::Relaxed);
        let (asset_stream_sender, _asset_stream_receiver) = bounded(asset_stream_capacity);

        let config = Arc::new(RuntimeConfig::default());
        let target_tickrate = tuning.load_target_tickrate();
        let logic_clock = LogicClock::new(target_tickrate, config.fixed_timestep);

        let input_manager = Arc::new(RwLock::new(InputManager::new()));
        crate::runtime::input_manager::register_web_input_manager(Arc::clone(&input_manager));

        Self {
            input_manager,
            asset_server: None,
            asset_base_path: None,
            opfs_enabled: true,

            render_thread_sender: render_sender,
            render_receiver,
            asset_receiver,
            asset_stream_sender,

            egui_winit_state: None,

            window: None,
            window_settings: WindowSettingsCache::new("helmer engine"),

            user_state: Some(Arc::new(Mutex::new(user_state))),
            callbacks: RuntimeCallbacks {
                init: Some(Box::new(init_callback)),
                tick: Arc::new(tick_callback),
                resize: Arc::new(resize_callback),
                dropped_file: Arc::new(dropped_file_callback),
            },

            last_title_update: Instant::now(),

            metrics: Arc::new(PerformanceMetrics::default()),
            config,
            renderer_stats: Arc::new(RendererStats::default()),
            tuning,
            profiling,

            has_init: Arc::new(AtomicBool::new(false)),

            new_window_size: None,
            resize_triggered: false,
            last_resize: Instant::now(),

            logic_clock,
            logic_state: LogicState {
                pending_render_delta: None,
                egui_cache: EguiTextureCache::default(),
                egui_pending_full_upload: false,
                egui_pending_free: Vec::new(),
            },
            render_state: None,
            render_init_receiver: None,
        }
    }

    pub fn init(&mut self) {
        let event_loop: EventLoop<()> = EventLoop::new().unwrap();
        let _ = event_loop.run_app(self);
    }

    pub fn set_asset_base_path(&mut self, path: impl Into<String>) {
        let path = path.into();
        self.asset_base_path = Some(path.clone());
        if let Some(asset_server) = self.asset_server.as_ref() {
            asset_server.lock().set_asset_base_path(path);
        }
    }

    pub fn clear_asset_base_path(&mut self) {
        self.asset_base_path = None;
        if let Some(asset_server) = self.asset_server.as_ref() {
            asset_server.lock().clear_asset_base_path();
        }
    }

    pub fn set_opfs_enabled(&mut self, enabled: bool) {
        self.opfs_enabled = enabled;
        if let Some(asset_server) = self.asset_server.as_ref() {
            asset_server.lock().set_opfs_enabled(enabled);
        }
    }

    pub fn configure_web_window(
        &mut self,
        canvas: Option<web_sys::HtmlCanvasElement>,
        append: bool,
        prevent_default: bool,
        focusable: bool,
    ) {
        use winit::platform::web::WindowAttributesExtWebSys;

        let mut attributes = self.window_settings.clone_attributes();
        attributes = attributes
            .with_canvas(canvas)
            .with_append(append)
            .with_prevent_default(prevent_default)
            .with_focusable(focusable);
        self.window_settings.set_attributes(attributes);
    }

    fn build_window_attributes(&self) -> WindowAttributes {
        self.window_settings.clone_attributes()
    }

    fn attach_window(&mut self, window: Arc<Window>) -> PhysicalSize<u32> {
        let new_size = window.inner_size();
        self.window = Some(Arc::clone(&window));
        self.window_settings.sync_geometry_from_window(&window);
        self.window_settings.apply_post_create(&window);

        let mut input = self.input_manager.write();
        input.window_size = UVec2::new(new_size.width, new_size.height);
        input.scale_factor = window.scale_factor();
        input.clear_queues();
        input.clear_egui_state();
        drop(input);

        self.egui_winit_state = Some(egui_winit::State::new(
            egui::Context::default(),
            ViewportId::default(),
            &window,
            None,
            None,
            None,
        ));

        self.new_window_size = None;
        self.resize_triggered = false;
        self.last_resize = Instant::now();
        self.logic_clock.reset(Instant::now());

        new_size
    }

    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> (Arc<Window>, PhysicalSize<u32>) {
        let window = Arc::new(
            event_loop
                .create_window(self.build_window_attributes())
                .unwrap(),
        );
        let size = self.attach_window(Arc::clone(&window));
        (window, size)
    }

    fn enqueue_renderer_init(&mut self, window: Arc<Window>, size: PhysicalSize<u32>) {
        if self.render_state.is_some() {
            return;
        }

        if self.render_init_receiver.is_some() {
            return;
        }

        let (sender, receiver) = bounded(1);
        self.render_init_receiver = Some(receiver);

        let asset_stream_sender = self.asset_stream_sender.clone();
        let renderer_stats = Arc::clone(&self.renderer_stats);
        let allow_experimental_features = self.config.wgpu_experimental_features;
        let backend_choice = self.config.wgpu_backend;
        let binding_backend_choice = self.config.binding_backend;
        let target_tickrate = self.tuning.load_target_tickrate();
        let render_size = if size.width == 0 || size.height == 0 {
            warn!(
                "window size is zero ({}x{}), falling back to 1x1 for renderer init",
                size.width, size.height
            );
            PhysicalSize::new(size.width.max(1), size.height.max(1))
        } else {
            size
        };

        spawn_local(async move {
            let init_with_backend = |choice: WgpuBackend| {
                let window = Arc::clone(&window);
                let asset_stream_sender = asset_stream_sender.clone();
                let renderer_stats = Arc::clone(&renderer_stats);
                async move {
                    let init = create_render_init(&window, choice).await?;
                    initialize_renderer(
                        init.instance,
                        init.surface,
                        render_size,
                        target_tickrate,
                        asset_stream_sender,
                        renderer_stats,
                        allow_experimental_features,
                        choice,
                        binding_backend_choice,
                    )
                    .await
                    .map(|renderer| WebRenderState::new(renderer, render_size))
                    .map_err(|err| err.to_string())
                }
            };

            let init_result = match init_with_backend(backend_choice).await {
                Ok(state) => Ok(state),
                Err(primary_err) if backend_choice != WgpuBackend::Gl => {
                    log_warn(&format!(
                        "renderer init failed with {} backend: {}; retrying with WebGL",
                        backend_choice.label(),
                        primary_err
                    ));
                    match init_with_backend(WgpuBackend::Gl).await {
                        Ok(state) => Ok(state),
                        Err(fallback_err) => {
                            log_warn(&format!(
                                "renderer init failed with WebGL fallback: {}",
                                fallback_err
                            ));
                            Err(format!(
                                "{primary_err}; WebGL fallback failed: {fallback_err}"
                            ))
                        }
                    }
                }
                Err(err) => Err(err),
            };

            let _ = sender.send(init_result);
        });
    }

    fn handle_render_init(&mut self) {
        let Some(receiver) = &self.render_init_receiver else {
            return;
        };
        match receiver.try_recv() {
            Ok(Ok(state)) => {
                self.render_state = Some(state);
                self.render_init_receiver = None;
            }
            Ok(Err(err)) => {
                warn!("failed to initialize renderer: {}", err);
                self.render_init_receiver = None;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                warn!("renderer init channel disconnected");
                self.render_init_receiver = None;
            }
        }
    }

    fn run_logic(&mut self, frame_start: Instant) {
        let Some(asset_server) = self.asset_server.as_ref() else {
            return;
        };

        let desired_tickrate = self.tuning.load_target_tickrate();
        self.logic_clock.set_tickrate(desired_tickrate, frame_start);
        let max_steps = self
            .tuning
            .max_logic_steps_per_frame
            .load(Ordering::Relaxed);
        let logic_frame = self.logic_clock.advance(frame_start, max_steps);
        if logic_frame.steps > 0 {
            let tps = (1.0 / logic_frame.dt).round() as u32;
            self.metrics.tps.store(tps, Ordering::Relaxed);
        }

        let profiling_enabled = self.profiling.enabled.load(Ordering::Relaxed);
        let (asset_us, input_us) = if profiling_enabled {
            let start = Instant::now();
            asset_server.lock().update();
            let asset_us = start.elapsed().as_micros() as u64;
            let start = Instant::now();
            self.input_manager.write().process_events();
            let input_us = start.elapsed().as_micros() as u64;
            (asset_us, input_us)
        } else {
            asset_server.lock().update();
            self.input_manager.write().process_events();
            (0, 0)
        };

        if self.has_init.load(Ordering::Relaxed) && logic_frame.steps > 0 {
            let tick_callback = Arc::clone(&self.callbacks.tick);
            let user_state = Arc::clone(self.user_state.as_ref().unwrap());
            let sender = self.render_thread_sender.clone();

            for _ in 0..logic_frame.steps {
                let tick_start = if profiling_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                let mut user_state_guard = user_state.lock();
                let (render_delta, egui_render_data) =
                    tick_callback(logic_frame.dt, &mut *user_state_guard);
                drop(user_state_guard);
                if let Some(start) = tick_start {
                    self.profiling
                        .logic_tick_us
                        .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                }

                if let Some(delta) = render_delta {
                    if let Some(pending) = self.logic_state.pending_render_delta.as_mut() {
                        pending.merge_from(delta);
                    } else {
                        self.logic_state.pending_render_delta = Some(delta);
                    }
                }

                if let Some(delta) = self.logic_state.pending_render_delta.take() {
                    let send_start = if profiling_enabled {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    match sender.try_send(RenderMessage::RenderDelta(delta)) {
                        Ok(_) => {}
                        Err(TrySendError::Full(RenderMessage::RenderDelta(delta))) => {
                            self.logic_state.pending_render_delta = Some(delta);
                        }
                        Err(TrySendError::Full(_)) => {}
                        Err(TrySendError::Disconnected(_)) => {
                            warn!("render queue disconnected");
                            break;
                        }
                    }
                    if let Some(start) = send_start {
                        self.profiling
                            .logic_render_send_us
                            .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    }
                }

                if let Some(mut data) = egui_render_data {
                    let textures_changed =
                        !data.textures_delta.set.is_empty() || !data.textures_delta.free.is_empty();
                    apply_egui_delta(&mut self.logic_state.egui_cache, &data.textures_delta);
                    if !data.textures_delta.free.is_empty() {
                        self.logic_state
                            .egui_pending_free
                            .extend_from_slice(&data.textures_delta.free);
                    }

                    let mut textures_delta = if self.logic_state.egui_pending_full_upload {
                        build_full_egui_delta(&self.logic_state.egui_cache)
                            .unwrap_or_else(|| data.textures_delta.clone())
                    } else {
                        data.textures_delta.clone()
                    };

                    if !self.logic_state.egui_pending_free.is_empty() {
                        for id in &self.logic_state.egui_pending_free {
                            if !textures_delta.free.contains(id) {
                                textures_delta.free.push(*id);
                            }
                        }
                    }

                    data.textures_delta = textures_delta;

                    match sender.try_send(RenderMessage::EguiData(data)) {
                        Ok(_) => {
                            self.logic_state.egui_pending_full_upload = false;
                            self.logic_state.egui_pending_free.clear();
                        }
                        Err(TrySendError::Full(_)) => {
                            if textures_changed || self.logic_state.egui_pending_full_upload {
                                self.logic_state.egui_pending_full_upload = true;
                            }
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            warn!("render queue disconnected");
                            break;
                        }
                    }
                }
            }
        }

        self.input_manager.write().prepare_for_next_frame();

        if profiling_enabled {
            self.profiling
                .logic_asset_us
                .store(asset_us, Ordering::Relaxed);
            self.profiling
                .logic_input_us
                .store(input_us, Ordering::Relaxed);
            self.profiling
                .logic_frame_us
                .store(frame_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    fn tick_frame(&mut self) {
        let frame_start = Instant::now();
        self.run_logic(frame_start);

        if let Some(render_state) = self.render_state.as_mut() {
            let keep_alive = render_state.render_frame(
                frame_start,
                &self.render_receiver,
                &self.asset_receiver,
                &self.tuning,
                &self.metrics,
                &self.profiling,
                self.tuning.load_target_fps(),
            );
            if !keep_alive {
                self.render_state = None;
            }
        }
    }
}

impl<T: 'static> ApplicationHandler for Runtime<T> {
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(window) = &self.window {
            if window_id != window.id() {
                return;
            }
        }
        let profiling_enabled = self.profiling.enabled.load(Ordering::Relaxed);
        let event_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };

        if let WindowEvent::RedrawRequested = event {
            self.tick_frame();
            if let Some(start) = event_start {
                self.profiling
                    .main_event_us
                    .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            return;
        }

        self.input_manager
            .read()
            .update_egui_state_from_winit(&event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    self.window_settings.set_inner_size(new_size);
                    self.new_window_size = Some(new_size);
                    self.resize_triggered = true;
                }
            }
            WindowEvent::Moved(position) => {
                self.window_settings.set_outer_position(position);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input_manager.read().push_event(InputEvent::Keyboard {
                        key: code,
                        pressed: event.state.is_pressed(),
                    });
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input_manager
                    .read()
                    .push_event(InputEvent::MouseButton {
                        button,
                        pressed: state.is_pressed(),
                    });
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input_manager
                    .read()
                    .push_event(InputEvent::CursorMoved(DVec2::new(position.x, position.y)));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => glam::vec2(x, y),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        let pixels_per_line = self.tuning.pixels_per_line.load(Ordering::Relaxed);
                        let pixels_per_line = (pixels_per_line as f32).max(1.0);
                        glam::vec2(
                            pos.x as f32 / pixels_per_line,
                            pos.y as f32 / pixels_per_line,
                        )
                    }
                };

                self.input_manager
                    .read()
                    .push_event(InputEvent::MouseWheel(scroll_delta));
            }
            WindowEvent::Focused(is_focused) => {
                if !is_focused {
                    let mut input_manager_guard = self.input_manager.write();
                    input_manager_guard.clear_egui_state();
                    input_manager_guard.clear_queues();
                }
            }
            WindowEvent::DroppedFile(path) => {
                if let Some(user_state) = &self.user_state {
                    let user_state_clone = Arc::clone(user_state);
                    let mut state = user_state_clone.lock();
                    (self.callbacks.dropped_file)(path, &mut state);
                }
            }
            _ => {}
        }

        if let Some(start) = event_start {
            self.profiling
                .main_event_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let (window, size) = if let Some(window) = &self.window {
            (Arc::clone(window), window.inner_size())
        } else {
            self.create_window(event_loop)
        };

        if self.asset_server.is_none() {
            let capacity = self
                .tuning
                .render_message_capacity
                .load(Ordering::Relaxed)
                .max(1);
            let asset_stream_capacity = self
                .tuning
                .asset_stream_queue_capacity
                .load(Ordering::Relaxed)
                .max(1);
            let asset_worker_capacity = self
                .tuning
                .asset_worker_queue_capacity
                .load(Ordering::Relaxed)
                .max(1);
            let (sender, receiver) = bounded(capacity);
            self.render_thread_sender = sender;
            self.render_receiver = receiver;

            let (asset_sender, asset_receiver) = bounded(capacity);
            self.asset_receiver = asset_receiver;

            let (asset_stream_sender, asset_stream_receiver) = bounded(asset_stream_capacity);
            self.asset_stream_sender = asset_stream_sender;

            self.asset_server = Some(Arc::new(Mutex::new(AssetServer::new(
                asset_sender,
                asset_stream_receiver,
                asset_worker_capacity,
                Arc::clone(&self.tuning),
            ))));
            if let Some(asset_server) = self.asset_server.as_ref() {
                if let Some(base_path) = self.asset_base_path.as_ref() {
                    asset_server.lock().set_asset_base_path(base_path.clone());
                }
                asset_server.lock().set_opfs_enabled(self.opfs_enabled);
                #[cfg(target_arch = "wasm32")]
                crate::runtime::asset_server::set_web_asset_server(Arc::clone(asset_server));
            }

            if let Some(init_fn) = self.callbacks.init.take() {
                if let Some(user_state) = &self.user_state {
                    let user_state_clone = Arc::clone(user_state);
                    let mut state = user_state_clone.lock();
                    init_fn(self, &mut *state);
                    self.has_init.store(true, Ordering::Relaxed);
                }
            }
        }

        self.enqueue_renderer_init(window, size);

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.handle_render_init();

        let profiling_enabled = self.profiling.enabled.load(Ordering::Relaxed);
        let update_start = if profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let now = Instant::now();
        let title_ms = self.tuning.title_update_ms.load(Ordering::Relaxed);
        let update_interval = Duration::from_millis(title_ms as u64);

        if now.duration_since(self.last_title_update) >= update_interval {
            if let Some(window) = &self.window {
                let fps = self.metrics.fps.load(Ordering::Relaxed);
                let tps = self.metrics.tps.load(Ordering::Relaxed);

                self.window_settings
                    .set_title(format!("helmer engine | FPS: {} | TPS: {}", fps, tps));
                window.set_title(self.window_settings.title());
            }

            self.last_title_update = now;
        }

        if let Some(new_size) = self.new_window_size {
            let debounce_ms = self.tuning.resize_debounce_ms.load(Ordering::Relaxed);
            if self.last_resize.elapsed() > Duration::from_millis(debounce_ms as u64) {
                if !self.resize_triggered {
                    if let Err(err) = self
                        .render_thread_sender
                        .try_send(RenderMessage::Resize(new_size))
                    {
                        warn!("resize queue is full: {:?}", err);
                    }

                    let mut input = self.input_manager.write();
                    input.window_size = UVec2::new(new_size.width, new_size.height);
                    input.scale_factor = self.window.as_ref().unwrap().scale_factor();
                    drop(input);

                    if let Some(user_state) = &self.user_state {
                        let user_state_clone = Arc::clone(user_state);
                        let mut state = user_state_clone.lock();
                        (self.callbacks.resize)(new_size, &mut state);
                    }

                    self.new_window_size = None;
                    self.last_resize = now;
                }

                self.resize_triggered = false;
            }
        }

        if let Some(start) = update_start {
            self.profiling
                .main_update_us
                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

async fn make_wgpu_instance(backend_choice: WgpuBackend) -> wgpu::Instance {
    match backend_choice {
        WgpuBackend::Gl => wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        }),
        _ => {
            let desc = wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
                ..Default::default()
            };
            wgpu::util::new_instance_with_webgpu_detection(&desc).await
        }
    }
}

fn create_surface(
    instance: &wgpu::Instance,
    window: &Arc<Window>,
) -> Result<wgpu::Surface<'static>, String> {
    instance
        .create_surface(window.clone())
        .map_err(|err| err.to_string())
}

async fn create_render_init(
    window: &Arc<Window>,
    backend_choice: WgpuBackend,
) -> Result<RenderInit, String> {
    let instance = make_wgpu_instance(backend_choice).await;
    let surface = create_surface(&instance, window)?;
    Ok(RenderInit { instance, surface })
}
