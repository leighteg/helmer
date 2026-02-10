use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError, bounded};
use egui::ViewportId;
use glam::{DVec2, UVec2};
use parking_lot::{Mutex, RwLock};
use resvg::tiny_skia;
use std::{
    collections::VecDeque,
    env,
    num::NonZeroU32,
    path::PathBuf,
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
    thread::{self, JoinHandle},
    time::Duration,
};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use web_time::Instant;
use wgpu::{BackendOptions, Dx12BackendOptions};
#[cfg(target_os = "linux")]
use winit::platform::{wayland::EventLoopBuilderExtWayland, x11::EventLoopBuilderExtX11};
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use super::{
    RuntimeLogLayer,
    common::{LogicClock, PerformanceMetrics, RuntimeProfiling, RuntimeTuning},
};
use crate::{
    graphics::{
        backend::binding_backend::BindingBackendChoice,
        common::renderer::{
            AssetStreamingRequest, EguiRenderData, EguiTextureCache, RenderControl, RenderDelta,
            RenderMessage, RendererStats, WgpuBackend, apply_egui_delta, build_full_egui_delta,
            initialize_renderer, render_message_payload_bytes,
        },
    },
    runtime::{
        asset_server::AssetServer,
        config::RuntimeConfig,
        input_manager::{InputEvent, InputManager},
    },
};

const MACOS_RETINA_SCALE_FIX_MIN_X: u32 = 3840;
const MACOS_RETINA_SCALE_FIX_MIN_Y: u32 = 2160;
const DEFAULT_RUNTIME_LOG_FILTER: &str = "info,helmer=trace,helmer_becs=trace,helmer_editor=trace,script=trace,audio=trace,\
     wgpu=warn,wgpu_core=warn,wgpu_hal=warn,naga=warn,notify=info,mio=info,polling=info";

pub struct RuntimeCallbacks<T: Send + 'static> {
    init: Option<Box<dyn FnOnce(&mut Runtime<T>, &mut T) + Send>>,
    tick: Arc<dyn Fn(f32, &mut T) -> (Option<RenderDelta>, Option<EguiRenderData>) + Send + Sync>,
    resize: Arc<dyn Fn(PhysicalSize<u32>, &mut T) + Send + Sync>,
    dropped_file: Arc<dyn Fn(PathBuf, &mut T) + Send + Sync>,
}

#[derive(Debug, Clone, Copy)]
enum WindowRecreateRequest {
    BackendSwitch,
}

#[derive(Debug, Clone, Copy)]
enum RenderInitRequest {
    Create,
}

type RenderInitResult = Result<RenderInit, String>;

struct RenderInit {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
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

    fn set_fullscreen(&mut self, fullscreen: Option<winit::window::Fullscreen>) {
        self.attributes.fullscreen = fullscreen;
    }

    fn sync_geometry_from_window(&mut self, window: &Window) {
        self.attributes.inner_size = Some(window.inner_size().into());
        if let Ok(position) = window.outer_position() {
            self.attributes.position = Some(position.into());
        }
    }

    fn sync_from_window(&mut self, window: &Window) {
        self.sync_geometry_from_window(window);
        self.attributes.fullscreen = window.fullscreen();
        self.attributes.resizable = window.is_resizable();
        self.attributes.decorations = window.is_decorated();
        self.attributes.maximized = window.is_maximized();
        if let Some(visible) = window.is_visible() {
            self.attributes.visible = visible;
        }
        if let Some(minimized) = window.is_minimized() {
            self.minimized = Some(minimized);
        }
    }

    fn apply_post_create(&self, window: &Window) {
        if let Some(minimized) = self.minimized {
            window.set_minimized(minimized);
        }
    }
}

fn make_wgpu_instance() -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        // Keep all backends available so we can switch adapters at runtime.
        backends: wgpu::Backends::all(),
        backend_options: BackendOptions {
            dx12: Dx12BackendOptions {
                shader_compiler: wgpu::Dx12Compiler::StaticDxc,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    })
}

fn create_surface(
    instance: &wgpu::Instance,
    window: &Arc<Window>,
) -> Result<wgpu::Surface<'static>, String> {
    #[cfg(target_os = "windows")]
    {
        use raw_window_handle::{RawDisplayHandle, WindowsDisplayHandle};
        use winit::platform::windows::WindowExtWindows;

        let raw_window_handle = unsafe {
            window
                .window_handle_any_thread()
                .map_err(|err| err.to_string())?
                .as_raw()
        };
        let raw_display_handle = RawDisplayHandle::Windows(WindowsDisplayHandle::new());
        unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle,
                    raw_window_handle,
                })
                .map_err(|err| err.to_string())
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        instance
            .create_surface(window.clone())
            .map_err(|err| err.to_string())
    }
}

fn create_render_init(window: &Arc<Window>) -> Result<RenderInit, String> {
    let instance = make_wgpu_instance();
    let surface = create_surface(&instance, window)?;
    Ok(RenderInit { instance, surface })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowRecreatePhase {
    Idle,
    Requested,
    Ready,
}

#[derive(Debug, Clone, Copy)]
struct PendingRecreate {
    backend: WgpuBackend,
    binding_backend: BindingBackendChoice,
    allow_experimental_features: bool,
    needs_window_recreate: bool,
    window_phase: WindowRecreatePhase,
}

impl PendingRecreate {
    fn new(
        renderer_backend: wgpu::Backend,
        backend: WgpuBackend,
        binding_backend: BindingBackendChoice,
        allow_experimental_features: bool,
    ) -> Self {
        let needs_window_recreate = cfg!(target_os = "windows")
            && renderer_backend == wgpu::Backend::Gl
            && backend != WgpuBackend::Gl;
        Self {
            backend,
            binding_backend,
            allow_experimental_features,
            needs_window_recreate,
            window_phase: WindowRecreatePhase::Idle,
        }
    }
}

pub struct Runtime<T: Send + 'static = ()> {
    pub input_manager: Arc<RwLock<InputManager>>,
    pub asset_server: Option<Arc<Mutex<AssetServer>>>,
    asset_base_path: Option<String>,

    // logic thread
    logic_thread: Option<JoinHandle<()>>,
    logic_thread_state: Arc<AtomicBool>,

    // render thread
    render_thread: Option<JoinHandle<()>>,
    render_thread_state: Arc<AtomicBool>,
    pub render_thread_sender: Sender<RenderMessage>,
    window_recreate_sender: Sender<WindowRecreateRequest>,
    window_recreate_receiver: Receiver<WindowRecreateRequest>,
    render_init_request_sender: Sender<RenderInitRequest>,
    render_init_request_receiver: Receiver<RenderInitRequest>,
    render_init_sender: Sender<RenderInitResult>,
    render_init_receiver: Option<Receiver<RenderInitResult>>,

    // --- EGUI ---
    egui_winit_state: Option<egui_winit::State>,

    // Window management
    window: Option<Arc<Window>>,
    window_settings: WindowSettingsCache,

    pub user_state: Option<Arc<Mutex<T>>>,
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
}

impl<T: Send + 'static> Runtime<T> {
    pub fn new(
        user_state: T,
        init_callback: impl FnOnce(&mut Runtime<T>, &mut T) + Send + 'static,
        tick_callback: impl Fn(f32, &mut T) -> (Option<RenderDelta>, Option<EguiRenderData>)
        + Send
        + Sync
        + 'static,
        resize_callback: impl Fn(PhysicalSize<u32>, &mut T) + Send + Sync + 'static,
        dropped_file_callback: impl Fn(PathBuf, &mut T) + Send + Sync + 'static,
    ) -> Self {
        // -- TRACING SETUP

        #[cfg(windows)]
        colored::control::set_virtual_terminal(true).ok();

        let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_RUNTIME_LOG_FILTER));
        tracing_subscriber::registry()
            .with(RuntimeLogLayer)
            .with(tracing_subscriber::fmt::layer())
            .with(env_filter)
            .try_init()
            .unwrap();

        tracing::info!("2026 leighton [https://leighteg.dev]");

        // -- END TRACING SETUP

        let tuning = Arc::new(RuntimeTuning::default());
        let profiling = Arc::new(RuntimeProfiling::default());
        let render_capacity = tuning.render_message_capacity.load(Ordering::Relaxed);
        let (render_sender, _) = bounded(render_capacity);
        let (window_recreate_sender, window_recreate_receiver) = bounded(2);
        let (render_init_request_sender, render_init_request_receiver) = bounded(1);
        let (render_init_sender, render_init_receiver) = bounded(1);

        Self {
            input_manager: Arc::new(RwLock::new(InputManager::new())),
            asset_server: None,
            asset_base_path: None,

            logic_thread: None,
            logic_thread_state: Arc::new(AtomicBool::new(true)),

            render_thread: None,
            render_thread_state: Arc::new(AtomicBool::new(true)),
            render_thread_sender: render_sender,
            window_recreate_sender,
            window_recreate_receiver,
            render_init_request_sender,
            render_init_request_receiver,
            render_init_sender,
            render_init_receiver: Some(render_init_receiver),

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
            config: Arc::new(RuntimeConfig::default()),
            renderer_stats: Arc::new(RendererStats::default()),
            tuning,
            profiling,

            has_init: Arc::new(AtomicBool::new(false)),

            new_window_size: None,
            resize_triggered: false,
            last_resize: Instant::now(),
        }
    }

    pub fn init(&mut self) {
        let event_loop: EventLoop<()> = if let Ok(backend) = env::var("HELMER_FORCE_UNIX_BACKEND") {
            #[cfg(target_os = "linux")]
            let mut builder = EventLoop::builder();
            match backend.as_str() {
                #[cfg(target_os = "linux")]
                "x11" => builder.with_x11().build().expect("failed to force x11"),
                #[cfg(target_os = "linux")]
                "wayland" => builder
                    .with_wayland()
                    .build()
                    .expect("failed to force wayland"),
                _ => EventLoop::new().unwrap(),
            }
        } else {
            EventLoop::new().unwrap()
        };

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

    fn start_logic_thread(&mut self) {
        let input_manager = Arc::clone(&self.input_manager);
        let asset_server = Arc::clone(&self.asset_server.as_ref().unwrap());
        let state = Arc::clone(&self.logic_thread_state);
        let metrics = Arc::clone(&self.metrics);
        let profiling = Arc::clone(&self.profiling);
        let tick_callback = Arc::clone(&self.callbacks.tick);
        let user_state = Arc::clone(self.user_state.as_ref().unwrap());
        let has_init = Arc::clone(&self.has_init);
        let deterministic = self.config.fixed_timestep;
        let tuning = Arc::clone(&self.tuning);

        let sender = self.render_thread_sender.clone();

        self.logic_thread = Some(thread::spawn(move || {
            let mut target_tickrate = tuning.load_target_tickrate();
            let mut clock = LogicClock::new(target_tickrate, deterministic);
            let mut frame_duration = Duration::from_secs_f32(1.0 / target_tickrate);
            let mut pending_render_delta: Option<RenderDelta> = None;
            let mut egui_cache = EguiTextureCache::default();
            let mut egui_pending_full_upload = false;
            let mut egui_pending_free: Vec<egui::TextureId> = Vec::new();

            while state.load(Ordering::Relaxed) {
                let frame_start = Instant::now();
                let desired_tickrate = tuning.load_target_tickrate();
                if (desired_tickrate - target_tickrate).abs() > f32::EPSILON {
                    target_tickrate = desired_tickrate;
                    frame_duration = Duration::from_secs_f32(1.0 / target_tickrate);
                    clock.set_tickrate(target_tickrate, frame_start);
                }
                let max_steps = tuning.max_logic_steps_per_frame.load(Ordering::Relaxed);
                let logic_frame = clock.advance(frame_start, max_steps);
                if logic_frame.steps > 0 {
                    let tps = (1.0 / logic_frame.dt).round() as u32;
                    metrics.tps.store(tps, Ordering::Relaxed);
                }

                let profiling_enabled = profiling.enabled.load(Ordering::Relaxed);
                let (asset_us, input_us) = if profiling_enabled {
                    let start = Instant::now();
                    asset_server.lock().update();
                    let asset_us = start.elapsed().as_micros() as u64;
                    let start = Instant::now();
                    input_manager.write().process_events();
                    let input_us = start.elapsed().as_micros() as u64;
                    (asset_us, input_us)
                } else {
                    asset_server.lock().update();
                    input_manager.write().process_events();
                    (0, 0)
                };

                if has_init.load(Ordering::Relaxed) && logic_frame.steps > 0 {
                    // MAIN LOGIC LOOP EXECUTION
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
                            profiling
                                .logic_tick_us
                                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                        }
                        input_manager.write().clear_just_pressed();

                        if let Some(delta) = render_delta {
                            if let Some(pending) = pending_render_delta.as_mut() {
                                pending.merge_from(delta);
                            } else {
                                pending_render_delta = Some(delta);
                            }
                        }

                        if let Some(delta) = pending_render_delta.take() {
                            // Drop render frames if the channel is saturated to keep logic thread moving
                            let send_start = if profiling_enabled {
                                Some(Instant::now())
                            } else {
                                None
                            };
                            match sender.try_send(RenderMessage::RenderDelta(delta)) {
                                Ok(_) => {}
                                Err(TrySendError::Full(RenderMessage::RenderDelta(delta))) => {
                                    pending_render_delta = Some(delta);
                                }
                                Err(TrySendError::Full(_)) => {}
                                Err(TrySendError::Disconnected(_)) => {
                                    warn!("render thread disconnected");
                                    break;
                                }
                            }
                            if let Some(start) = send_start {
                                profiling
                                    .logic_render_send_us
                                    .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                            }
                        }

                        if let Some(mut data) = egui_render_data {
                            let textures_changed = !data.textures_delta.set.is_empty()
                                || !data.textures_delta.free.is_empty();
                            apply_egui_delta(&mut egui_cache, &data.textures_delta);
                            if !data.textures_delta.free.is_empty() {
                                egui_pending_free.extend_from_slice(&data.textures_delta.free);
                            }

                            let mut textures_delta = if egui_pending_full_upload {
                                build_full_egui_delta(&egui_cache)
                                    .unwrap_or_else(|| data.textures_delta.clone())
                            } else {
                                data.textures_delta.clone()
                            };

                            if !egui_pending_free.is_empty() {
                                for id in &egui_pending_free {
                                    if !textures_delta.free.contains(id) {
                                        textures_delta.free.push(*id);
                                    }
                                }
                            }

                            data.textures_delta = textures_delta;

                            match sender.try_send(RenderMessage::EguiData(data)) {
                                Ok(_) => {
                                    egui_pending_full_upload = false;
                                    egui_pending_free.clear();
                                }
                                Err(TrySendError::Full(_)) => {
                                    if textures_changed || egui_pending_full_upload {
                                        egui_pending_full_upload = true;
                                    }
                                }
                                Err(TrySendError::Disconnected(_)) => {
                                    warn!("render thread disconnected");
                                    break;
                                }
                            }
                        }
                    }
                }

                input_manager.write().prepare_for_next_frame();

                // tick rate limiting
                let logic_elapsed = frame_start.elapsed();
                let time_to_wait = frame_duration.saturating_sub(logic_elapsed);

                // Sleep for bulk of time (low CPU)
                if time_to_wait > Duration::from_millis(1) {
                    thread::sleep(time_to_wait - Duration::from_millis(1));
                }

                // Spin for final precision (high CPU)
                while frame_start.elapsed() < frame_duration {
                    thread::yield_now();
                }

                if profiling_enabled {
                    profiling.logic_asset_us.store(asset_us, Ordering::Relaxed);
                    profiling.logic_input_us.store(input_us, Ordering::Relaxed);
                    profiling
                        .logic_frame_us
                        .store(frame_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
            }

            info!("logic thread shutting down");
        }));

        info!("initialized logic thread");
    }

    fn start_render_thread(
        &mut self,
        render_receiver: Receiver<RenderMessage>,
        asset_receiver: Receiver<RenderMessage>,
        asset_stream_sender: crossbeam_channel::Sender<AssetStreamingRequest>,
        window_recreate_sender: Sender<WindowRecreateRequest>,
        render_init_request_sender: Sender<RenderInitRequest>,
        render_init_receiver: Receiver<RenderInitResult>,
        asset_server: Option<Arc<Mutex<AssetServer>>>,
    ) {
        let state = Arc::clone(&self.render_thread_state);
        let window = Arc::clone(self.window.as_ref().unwrap());
        let window_size = window.inner_size();
        let metrics = Arc::clone(&self.metrics);
        let renderer_stats = Arc::clone(&self.renderer_stats);
        let tuning = Arc::clone(&self.tuning);
        let profiling = Arc::clone(&self.profiling);
        let allow_experimental_features = self.config.wgpu_experimental_features;
        let backend_choice = self.config.wgpu_backend;
        let binding_backend_choice = self.config.binding_backend;

        let render_thread_handle = thread::spawn(move || {
            let mut _window = window;
            let wait_for_render_init =
                |receiver: &Receiver<RenderInitResult>| -> Option<RenderInit> {
                    loop {
                        match receiver.recv_timeout(Duration::from_millis(16)) {
                            Ok(Ok(init)) => return Some(init),
                            Ok(Err(err)) => {
                                warn!("failed to create surface: {}", err);
                                return None;
                            }
                            Err(RecvTimeoutError::Timeout) => {
                                if !state.load(Ordering::Relaxed) {
                                    return None;
                                }
                            }
                            Err(RecvTimeoutError::Disconnected) => {
                                warn!("render init channel disconnected");
                                return None;
                            }
                        }
                    }
                };

            let Some(render_init) = wait_for_render_init(&render_init_receiver) else {
                return;
            };
            let target_tickrate = tuning.load_target_tickrate();
            let mut renderer = pollster::block_on(async {
                initialize_renderer(
                    render_init.instance,
                    render_init.surface,
                    window_size,
                    target_tickrate,
                    asset_stream_sender.clone(),
                    Arc::clone(&renderer_stats),
                    allow_experimental_features,
                    backend_choice,
                    binding_backend_choice,
                )
                .await
                .unwrap()
            });

            let mut last_render = Instant::now();
            let mut asset_backlog: VecDeque<RenderMessage> = VecDeque::new();
            let mut asset_backlog_bytes: usize = 0;
            let mut immediate_backlog: VecDeque<(RenderMessage, usize)> = VecDeque::new();
            let mut upload_batch: Vec<RenderMessage> = Vec::new();
            let mut upload_bytes: Vec<usize> = Vec::new();
            let mut poll_frame: u32 = 0;
            let mut surface_size = window_size;
            let mut pending_recreate: Option<PendingRecreate> = None;

            while state.load(Ordering::Relaxed) {
                let frame_start = Instant::now();
                let profiling_enabled = profiling.enabled.load(Ordering::Relaxed);
                let mut should_render = false;
                let messages_start = if profiling_enabled {
                    Some(Instant::now())
                } else {
                    None
                };

                let dt = frame_start.duration_since(last_render).as_secs_f32();
                let fps = (1.0 / dt).round() as u32;
                metrics.fps.store(fps, Ordering::Relaxed);

                //renderer.resolve_pending_materials();

                let max_pending = tuning.max_pending_asset_uploads.load(Ordering::Relaxed);
                let max_pending_bytes = tuning.max_pending_asset_bytes.load(Ordering::Relaxed);
                let backlog_enabled = max_pending > 0 && max_pending_bytes > 0;
                if !backlog_enabled && !asset_backlog.is_empty() {
                    for message in asset_backlog.drain(..) {
                        tuning.release_asset_upload(render_message_payload_bytes(&message));
                    }
                    asset_backlog_bytes = 0;
                }

                // Drain priority messages without blocking on asset uploads
                loop {
                    match render_receiver.try_recv() {
                        Ok(message) => match message {
                            message @ RenderMessage::CreateMesh { .. }
                            | message @ RenderMessage::CreateTexture { .. }
                            | message @ RenderMessage::CreateMaterial(_) => {
                                let message_bytes = render_message_payload_bytes(&message);
                                let fits_bytes = asset_backlog_bytes.saturating_add(message_bytes)
                                    <= max_pending_bytes;
                                if backlog_enabled
                                    && asset_backlog.len() < max_pending
                                    && fits_bytes
                                {
                                    asset_backlog_bytes =
                                        asset_backlog_bytes.saturating_add(message_bytes);
                                    asset_backlog.push_back(message);
                                } else {
                                    immediate_backlog.push_back((message, message_bytes));
                                }
                                should_render = true;
                            }
                            RenderMessage::RenderData(data) => {
                                should_render = true;
                                renderer.process_message(RenderMessage::RenderData(data));
                            }
                            RenderMessage::RenderDelta(delta) => {
                                should_render = true;
                                renderer.process_message(RenderMessage::RenderDelta(delta));
                            }
                            RenderMessage::Resize(size) => {
                                should_render = true;
                                surface_size = size;
                                renderer.process_message(RenderMessage::Resize(size));
                            }
                            RenderMessage::WindowRecreated {
                                window: new_window,
                                size,
                            } => {
                                should_render = true;
                                _window = new_window;
                                surface_size = size;
                                if let Some(pending) = pending_recreate.as_mut() {
                                    if pending.needs_window_recreate {
                                        pending.window_phase = WindowRecreatePhase::Ready;
                                    }
                                }
                            }
                            RenderMessage::Control(ctrl) => {
                                should_render = true;
                                match ctrl {
                                    RenderControl::RecreateDevice {
                                        backend,
                                        binding_backend,
                                        allow_experimental_features: allow,
                                    } => {
                                        pending_recreate = Some(PendingRecreate::new(
                                            renderer.adapter_backend(),
                                            backend,
                                            binding_backend,
                                            allow,
                                        ));
                                    }
                                    _ => renderer.process_message(RenderMessage::Control(ctrl)),
                                }
                            }
                            RenderMessage::Shutdown => {
                                renderer.process_message(RenderMessage::Shutdown);
                                return;
                            }
                            other => {
                                should_render = true;
                                renderer.process_message(other);
                            }
                        },
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }

                // Drain asset uploads into the backlog until the cap is hit
                loop {
                    if backlog_enabled && asset_backlog.len() >= max_pending {
                        break;
                    }
                    match asset_receiver.try_recv() {
                        Ok(message) => match message {
                            message @ RenderMessage::CreateMesh { .. }
                            | message @ RenderMessage::CreateTexture { .. }
                            | message @ RenderMessage::CreateMaterial(_) => {
                                let message_bytes = render_message_payload_bytes(&message);
                                let fits_bytes = asset_backlog_bytes.saturating_add(message_bytes)
                                    <= max_pending_bytes;
                                if backlog_enabled && fits_bytes {
                                    asset_backlog_bytes =
                                        asset_backlog_bytes.saturating_add(message_bytes);
                                    asset_backlog.push_back(message);
                                } else {
                                    immediate_backlog.push_back((message, message_bytes));
                                }
                                should_render = true;
                            }
                            other => {
                                should_render = true;
                                renderer.process_message(other);
                            }
                        },
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => break,
                    }
                }

                if let Some(request) = pending_recreate.as_mut() {
                    let mut wait_for_window = false;
                    if request.needs_window_recreate {
                        if request.window_phase == WindowRecreatePhase::Idle {
                            match window_recreate_sender
                                .try_send(WindowRecreateRequest::BackendSwitch)
                            {
                                Ok(()) => {
                                    request.window_phase = WindowRecreatePhase::Requested;
                                }
                                Err(TrySendError::Full(_)) => {
                                    warn!("window recreate request queue is full");
                                    request.window_phase = WindowRecreatePhase::Requested;
                                }
                                Err(TrySendError::Disconnected(_)) => {
                                    warn!("window recreate channel disconnected");
                                }
                            }
                        }
                        if request.window_phase != WindowRecreatePhase::Ready {
                            wait_for_window = true;
                        }
                    }
                    if !wait_for_window {
                        let request = pending_recreate.take().unwrap();
                        info!(
                            "recreating render device (backend {:?}, binding {:?})",
                            request.backend, request.binding_backend
                        );
                        renderer.prepare_for_recreate();
                        renderer.poll_device(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: None,
                        });
                        let snapshot = renderer.take_snapshot();
                        let _old_parts = renderer.into_parts();
                        drop(_old_parts);
                        if let Err(err) = render_init_request_sender.send(RenderInitRequest::Create)
                        {
                            warn!("render init request channel disconnected: {}", err);
                            return;
                        }
                        let Some(render_init) = wait_for_render_init(&render_init_receiver) else {
                            return;
                        };
                        let target_tickrate = tuning.load_target_tickrate();
                        let recreated = pollster::block_on(async {
                            initialize_renderer(
                                render_init.instance,
                                render_init.surface,
                                surface_size,
                                target_tickrate,
                                asset_stream_sender.clone(),
                                Arc::clone(&renderer_stats),
                                request.allow_experimental_features,
                                request.backend,
                                request.binding_backend,
                            )
                            .await
                        });
                        match recreated {
                            Ok(mut new_renderer) => {
                                new_renderer.restore_snapshot(snapshot);
                                renderer = new_renderer;
                                if let Some(asset_server) = asset_server.as_ref() {
                                    asset_server.lock().reupload_cached_assets();
                                }
                            }
                            Err(err) => {
                                warn!("failed to recreate render device: {}", err);
                                return;
                            }
                        }
                        for message in asset_backlog.drain(..) {
                            tuning.release_asset_upload(render_message_payload_bytes(&message));
                        }
                        for (_, bytes) in immediate_backlog.drain(..) {
                            tuning.release_asset_upload(bytes);
                        }
                        asset_backlog_bytes = 0;
                        should_render = true;
                    }
                }

                // Upload a limited batch of heavy assets each frame to avoid long stalls
                let uploads_per_frame = tuning.asset_uploads_per_frame.load(Ordering::Relaxed);
                let mut uploads_this_frame = 0usize;
                let upload_start = if profiling_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                upload_batch.clear();
                upload_bytes.clear();
                while uploads_this_frame < uploads_per_frame {
                    if let Some((msg, message_bytes)) = immediate_backlog.pop_front() {
                        upload_batch.push(msg);
                        upload_bytes.push(message_bytes);
                        uploads_this_frame += 1;
                        continue;
                    }
                    if let Some(msg) = asset_backlog.pop_front() {
                        let message_bytes = render_message_payload_bytes(&msg);
                        asset_backlog_bytes = asset_backlog_bytes.saturating_sub(message_bytes);
                        upload_batch.push(msg);
                        upload_bytes.push(message_bytes);
                        uploads_this_frame += 1;
                        continue;
                    }
                    break;
                }
                if !upload_batch.is_empty() {
                    renderer.process_asset_batch(&mut upload_batch);
                    for bytes in upload_bytes.drain(..) {
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

                if !state.load(Ordering::Relaxed) {
                    break;
                }

                if let Some(target_fps) = tuning.load_target_fps() {
                    // Render at target FPS if we have new data OR if enough time has passed
                    let frame_duration = Duration::from_secs_f32(1.0 / target_fps);
                    let time_since_last_render = frame_start.duration_since(last_render);
                    if should_render || time_since_last_render >= frame_duration {
                        let render_start = if profiling_enabled {
                            Some(Instant::now())
                        } else {
                            None
                        };
                        if let Err(e) = renderer.render() {
                            warn!("render error: {}", e);
                        }
                        if let Some(start) = render_start {
                            profiling
                                .render_thread_render_us
                                .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                        }
                        last_render = frame_start;
                    }

                    // Sleep to maintain frame rate
                    let elapsed = frame_start.elapsed();
                    if elapsed < frame_duration {
                        thread::sleep(frame_duration - elapsed);
                    }
                } else {
                    let render_start = if profiling_enabled {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    if let Err(e) = renderer.render() {
                        warn!("render error: {}", e);
                    }
                    if let Some(start) = render_start {
                        profiling
                            .render_thread_render_us
                            .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    }
                    last_render = frame_start;
                }

                let poll_interval = tuning.wgpu_poll_interval_frames.load(Ordering::Relaxed);
                let poll_mode = tuning.wgpu_poll_mode.load(Ordering::Relaxed);
                if poll_interval > 0 && poll_mode != 0 && (poll_frame % poll_interval == 0) {
                    let poll_type = match poll_mode {
                        2 => wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: None,
                        },
                        1 => wgpu::PollType::Poll,
                        _ => wgpu::PollType::Poll,
                    };
                    renderer.poll_device(poll_type);
                }
                poll_frame = poll_frame.wrapping_add(1);

                if profiling_enabled {
                    profiling
                        .render_thread_frame_us
                        .store(frame_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
            }

            info!("render thread shutting down");
        });

        info!("initialized render thread");
        self.render_thread = Some(render_thread_handle);
    }

    pub fn shutdown_threads(&mut self) {
        self.render_thread_state.store(false, Ordering::Relaxed);
        self.logic_thread_state.store(false, Ordering::Relaxed);

        let _ = self.render_thread_sender.try_send(RenderMessage::Shutdown);

        if let Some(handle) = self.render_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.logic_thread.take() {
            let _ = handle.join();
        }
    }

    fn build_window_attributes(&self) -> WindowAttributes {
        self.window_settings.clone_attributes()
    }

    fn attach_window(&mut self, window: Arc<Window>) -> PhysicalSize<u32> {
        let new_size = window.inner_size();
        self.window = Some(Arc::clone(&window));
        self.window_settings.sync_geometry_from_window(&window);
        self.draw_splash();
        self.window_settings.apply_post_create(&window);

        let mut input = self.input_manager.write();
        input.window_size = UVec2::new(new_size.width, new_size.height);
        #[cfg(target_os = "macos")]
        if new_size.width >= MACOS_RETINA_SCALE_FIX_MIN_X
            && new_size.height >= MACOS_RETINA_SCALE_FIX_MIN_Y
        {
            input.scale_factor = window.scale_factor();
        }
        #[cfg(not(target_os = "macos"))]
        {
            input.scale_factor = window.scale_factor();
        }
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

    fn recreate_window(&mut self, event_loop: &ActiveEventLoop) {
        let Some(old_window) = self.window.as_ref() else {
            return;
        };
        self.window_settings.sync_from_window(old_window);
        let (new_window, new_size) = self.create_window(event_loop);
        if let Err(err) = self
            .render_thread_sender
            .send(RenderMessage::WindowRecreated {
                window: new_window,
                size: new_size,
            })
        {
            warn!("failed to send window recreation to render thread: {}", err);
        }
    }

    pub fn draw_splash(&self) {
        let window_size: PhysicalSize<u32> = self.window.as_ref().unwrap().inner_size();

        // SPLASH SCREEN
        let context = softbuffer::Context::new(self.window.as_ref().unwrap().clone()).unwrap();
        let mut surface =
            softbuffer::Surface::new(&context, self.window.as_ref().unwrap().clone()).unwrap();

        surface
            .resize(
                NonZeroU32::new(window_size.width).unwrap(),
                NonZeroU32::new(window_size.height).unwrap(),
            )
            .unwrap();

        // Load and parse SVG
        const BRAND_SVG_DATA: &[u8] = include_bytes!("../../../../brand/helmer.svg");
        let svg_str = std::str::from_utf8(BRAND_SVG_DATA)
            .map_err(|_| "Failed to convert SVG bytes to string")
            .unwrap();

        let mut opt = resvg::usvg::Options::default();
        opt.dpi = 96.0;

        let tree = resvg::usvg::Tree::from_str(svg_str, &opt)
            .map_err(|_| "Failed to parse SVG")
            .unwrap();

        // Calculate scaling to fit 1/3 of window size while maintaining aspect ratio
        let svg_size = tree.size();
        let svg_width = svg_size.width();
        let svg_height = svg_size.height();

        let max_width = window_size.width as f32 / 3.0;
        let max_height = window_size.height as f32 / 3.0;

        let scale_x = max_width / svg_width;
        let scale_y = max_height / svg_height;
        let scale = scale_x.min(scale_y);

        let scaled_width = (svg_width * scale) as u32;
        let scaled_height = (svg_height * scale) as u32;

        // Center the SVG
        let offset_x = (window_size.width - scaled_width) / 2 - (window_size.width / 55);
        let offset_y = (window_size.height - scaled_height) / 2;

        // Create pixmap for rendering SVG
        if let Some(mut pixmap) = tiny_skia::Pixmap::new(scaled_width, scaled_height) {
            // Clear with transparent background
            pixmap.fill(tiny_skia::Color::TRANSPARENT);

            // Create transform for scaling
            let transform = tiny_skia::Transform::from_scale(scale, scale);

            // Render SVG to pixmap
            resvg::render(&tree, transform, &mut pixmap.as_mut());

            // Get buffer and fill with white background
            let mut buffer = surface.buffer_mut().unwrap();
            let white_u32 = 0xFFFFFFFF;
            buffer.fill(white_u32);

            // Copy pixmap to buffer with centering
            copy_pixmap_to_buffer(
                &pixmap,
                &mut buffer,
                window_size.width as usize,
                window_size.height as usize,
                offset_x,
                offset_y,
            );

            // Present the splash screen
            buffer.present().unwrap();
        }
    }
}

impl<T: Send + 'static> ApplicationHandler for Runtime<T> {
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
            if self.render_thread.is_none() {
                self.draw_splash();
                self.window.as_ref().unwrap().request_redraw();
            }
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
                self.shutdown_threads();
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
                    if code == KeyCode::F11 && event.state.is_pressed() {
                        let fullscreen = {
                            let window = self.window.as_mut().unwrap();
                            let fullscreen = if window.fullscreen().is_some() {
                                None
                            } else {
                                Some(winit::window::Fullscreen::Borderless(None))
                            };
                            window.set_fullscreen(fullscreen.clone());
                            fullscreen
                        };
                        self.window_settings.set_fullscreen(fullscreen);
                    } else {
                        self.input_manager.read().push_event(InputEvent::Keyboard {
                            key: code,
                            pressed: event.state.is_pressed(),
                        });
                    }
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
                    // clear InputManager's state when we unfocus the window
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
        if self.window.is_none() {
            let _ = self.create_window(event_loop);
        }

        if self.render_thread.is_none() && self.logic_thread.is_none() {
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

            let (asset_sender, asset_receiver) = bounded(capacity);
            let (asset_stream_sender, asset_stream_receiver) = bounded(asset_stream_capacity);

            self.asset_server = Some(Arc::new(Mutex::new(AssetServer::new(
                asset_sender,
                asset_stream_receiver,
                asset_worker_capacity,
                Arc::clone(&self.tuning),
            ))));
            if let Some(base_path) = self.asset_base_path.as_ref() {
                if let Some(asset_server) = self.asset_server.as_ref() {
                    asset_server.lock().set_asset_base_path(base_path.clone());
                }
            }

            let render_init_receiver = match self.render_init_receiver.take() {
                Some(receiver) => receiver,
                None => {
                    warn!("render init receiver missing");
                    return;
                }
            };
            let render_init_result = match self.window.as_ref() {
                Some(window) => create_render_init(window),
                None => Err("render init requires a window".to_string()),
            };
            if let Err(err) = self.render_init_sender.send(render_init_result) {
                warn!("failed to send render init: {}", err);
            }

            self.start_render_thread(
                receiver,
                asset_receiver,
                asset_stream_sender,
                self.window_recreate_sender.clone(),
                self.render_init_request_sender.clone(),
                render_init_receiver,
                self.asset_server.as_ref().map(Arc::clone),
            );
            self.start_logic_thread();

            if let Some(init_fn) = self.callbacks.init.take() {
                if let Some(user_state) = &self.user_state {
                    let user_state_clone = Arc::clone(user_state);
                    let mut state = user_state_clone.lock();
                    init_fn(self, &mut *state);

                    self.has_init.store(true, Ordering::Relaxed);
                }
            }
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let mut recreate_window = false;
        while self.window_recreate_receiver.try_recv().is_ok() {
            recreate_window = true;
        }
        if recreate_window {
            self.recreate_window(event_loop);
        }
        while self.render_init_request_receiver.try_recv().is_ok() {
            let render_init_result = match self.window.as_ref() {
                Some(window) => create_render_init(window),
                None => Err("render init requires a window".to_string()),
            };
            if let Err(err) = self.render_init_sender.send(render_init_result) {
                warn!("failed to send render init: {}", err);
            }
        }
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
                    // --- RESIZE LOGIC ---
                    let _ = self
                        .render_thread_sender
                        .send(RenderMessage::Resize(new_size));

                    let mut input = self.input_manager.write();
                    input.window_size = UVec2::new(new_size.width, new_size.height);
                    #[cfg(not(target_os = "macos"))]
                    {
                        input.scale_factor = self.window.as_ref().unwrap().scale_factor();
                    }
                    if new_size.width >= MACOS_RETINA_SCALE_FIX_MIN_X
                        && new_size.height >= MACOS_RETINA_SCALE_FIX_MIN_Y
                    {
                        input.scale_factor = self.window.as_ref().unwrap().scale_factor();
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        input.scale_factor = self.window.as_ref().unwrap().scale_factor();
                    }
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
    }
}

fn copy_pixmap_to_buffer(
    pixmap: &tiny_skia::Pixmap,
    buffer: &mut [u32],
    buffer_width: usize,
    buffer_height: usize,
    offset_x: u32,
    offset_y: u32,
) {
    let pixmap_width = pixmap.width() as usize;
    let pixmap_height = pixmap.height() as usize;
    let pixmap_data = pixmap.data();

    for y in 0..pixmap_height {
        let buffer_y = y + offset_y as usize;
        if buffer_y >= buffer_height {
            break;
        }

        for x in 0..pixmap_width {
            let buffer_x = x + offset_x as usize;
            if buffer_x >= buffer_width {
                break;
            }

            let pixmap_idx = (y * pixmap_width + x) * 4;
            let buffer_idx = buffer_y * buffer_width + buffer_x;

            if pixmap_idx + 3 < pixmap_data.len() && buffer_idx < buffer.len() {
                let r = pixmap_data[pixmap_idx] as u32;
                let g = pixmap_data[pixmap_idx + 1] as u32;
                let b = pixmap_data[pixmap_idx + 2] as u32;
                let a = pixmap_data[pixmap_idx + 3] as u32;

                // Only render non-transparent pixels
                if a > 0 {
                    // Blend with white background
                    if a == 255 {
                        buffer[buffer_idx] = 0xFF000000 | (r << 16) | (g << 8) | b;
                    } else {
                        let alpha_f = a as f32 / 255.0;
                        let inv_alpha = 1.0 - alpha_f;

                        let blended_r = ((r as f32 * alpha_f + 255.0 * inv_alpha) as u32).min(255);
                        let blended_g = ((g as f32 * alpha_f + 255.0 * inv_alpha) as u32).min(255);
                        let blended_b = ((b as f32 * alpha_f + 255.0 * inv_alpha) as u32).min(255);

                        buffer[buffer_idx] =
                            0xFF000000 | (blended_r << 16) | (blended_g << 8) | blended_b;
                    }
                }
            }
        }
    }
}
