use crate::config::RuntimeConfig;
use crate::event::{WindowRuntimeEvent, WindowRuntimeEventKind, WindowState};
use crate::state::{
    RuntimeCursorGrabMode, RuntimeCursorState, RuntimeWindowControl, RuntimeWindowControlSnapshot,
    RuntimeWindowTitleMode,
};
use glam::{DVec2, UVec2, Vec2};
use helmer::runtime::PerformanceMetrics;
use helmer_input::input_manager::{InputEvent, InputManager};
use parking_lot::RwLock;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;
use web_time::Instant;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

pub type WindowEventCallback = Box<dyn FnMut(&WindowRuntimeEvent) + 'static>;

#[derive(Default)]
pub struct WindowCallbacks {
    pub on_event: Option<WindowEventCallback>,
}

#[derive(Debug)]
pub enum WindowError {
    EventLoop(String),
    WindowCreation(String),
}

impl Display for WindowError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventLoop(message) => write!(f, "window event loop error: {message}"),
            Self::WindowCreation(message) => write!(f, "window creation error: {message}"),
        }
    }
}

impl std::error::Error for WindowError {}

pub struct WindowService {
    pub config: RuntimeConfig,
    pub input_manager: Arc<RwLock<InputManager>>,
    pub cursor_state: Arc<RuntimeCursorState>,
    pub window_control: Arc<RuntimeWindowControl>,
    pub metrics: Option<Arc<PerformanceMetrics>>,
    #[cfg(target_arch = "wasm32")]
    pub wasm_harness: Option<crate::wasm_harness::WasmHarnessConfig>,
}

impl WindowService {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            input_manager: Arc::new(RwLock::new(InputManager::new())),
            cursor_state: Arc::new(RuntimeCursorState::default()),
            window_control: Arc::new(RuntimeWindowControl::default()),
            metrics: None,
            #[cfg(target_arch = "wasm32")]
            wasm_harness: None,
        }
    }

    pub fn with_input_manager(
        config: RuntimeConfig,
        input_manager: Arc<RwLock<InputManager>>,
    ) -> Self {
        Self {
            config,
            input_manager,
            cursor_state: Arc::new(RuntimeCursorState::default()),
            window_control: Arc::new(RuntimeWindowControl::default()),
            metrics: None,
            #[cfg(target_arch = "wasm32")]
            wasm_harness: None,
        }
    }

    pub fn with_runtime_state(
        mut self,
        cursor_state: Arc<RuntimeCursorState>,
        window_control: Arc<RuntimeWindowControl>,
    ) -> Self {
        window_control.set_title_update_ms(self.config.title_update_ms);
        window_control.set_target_tickrate(self.config.target_tickrate);
        window_control.set_target_fps(self.config.target_fps);
        self.cursor_state = cursor_state;
        self.window_control = window_control;
        self
    }

    pub fn with_metrics(mut self, metrics: Arc<PerformanceMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    #[cfg(target_arch = "wasm32")]
    pub fn with_wasm_harness(mut self, config: crate::wasm_harness::WasmHarnessConfig) -> Self {
        self.wasm_harness = Some(config);
        self
    }

    pub fn run(self, callbacks: WindowCallbacks) -> Result<(), WindowError> {
        let event_loop = EventLoop::new().map_err(|err| WindowError::EventLoop(err.to_string()))?;
        #[cfg(target_arch = "wasm32")]
        {
            let app = NativeWindowApp::new(
                self.config,
                self.input_manager,
                self.cursor_state,
                self.window_control,
                self.metrics,
                callbacks,
                self.wasm_harness,
            );
            event_loop.spawn_app(app);
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut app = NativeWindowApp::new(
                self.config,
                self.input_manager,
                self.cursor_state,
                self.window_control,
                self.metrics,
                callbacks,
            );
            event_loop
                .run_app(&mut app)
                .map_err(|err| WindowError::EventLoop(err.to_string()))
        }
    }
}

struct NativeWindowApp {
    config: RuntimeConfig,
    input_manager: Arc<RwLock<InputManager>>,
    cursor_state: Arc<RuntimeCursorState>,
    window_control: Arc<RuntimeWindowControl>,
    metrics: Option<Arc<PerformanceMetrics>>,
    callbacks: WindowCallbacks,
    window: Option<Arc<Window>>,
    _high_resolution_timer: helmer::runtime::HighResolutionTimerGuard,
    last_frame: Instant,
    last_tick: Instant,
    next_tick_deadline: Instant,
    next_frame_deadline: Instant,
    last_title_update: Instant,
    last_window_title: String,
    applied_window_control: RuntimeWindowControlSnapshot,
    #[cfg(target_arch = "wasm32")]
    wasm_harness: Option<crate::wasm_harness::WasmHarnessConfig>,
}

impl NativeWindowApp {
    fn new(
        config: RuntimeConfig,
        input_manager: Arc<RwLock<InputManager>>,
        cursor_state: Arc<RuntimeCursorState>,
        window_control: Arc<RuntimeWindowControl>,
        metrics: Option<Arc<PerformanceMetrics>>,
        callbacks: WindowCallbacks,
        #[cfg(target_arch = "wasm32")] wasm_harness: Option<crate::wasm_harness::WasmHarnessConfig>,
    ) -> Self {
        let now = Instant::now();
        let applied_window_control = window_control.snapshot();
        Self {
            config,
            input_manager,
            cursor_state,
            window_control,
            metrics,
            callbacks,
            window: None,
            _high_resolution_timer: helmer::runtime::HighResolutionTimerGuard::new(1),
            last_frame: now,
            last_tick: now,
            next_tick_deadline: now,
            next_frame_deadline: now,
            last_title_update: now,
            last_window_title: String::new(),
            applied_window_control,
            #[cfg(target_arch = "wasm32")]
            wasm_harness,
        }
    }

    fn emit(&mut self, kind: WindowRuntimeEventKind) {
        if let Some(cb) = self.callbacks.on_event.as_mut() {
            cb(&WindowRuntimeEvent { kind });
        }
    }

    fn window_state(window: &Window) -> WindowState {
        let size = window.inner_size();
        WindowState {
            width: size.width.max(1),
            height: size.height.max(1),
            scale_factor: window.scale_factor(),
        }
    }

    fn target_frame_interval(snapshot: &RuntimeWindowControlSnapshot) -> Option<Duration> {
        snapshot
            .target_fps
            .map(|fps| Duration::from_secs_f32(1.0 / fps.max(1.0)))
    }

    fn tick_interval(snapshot: &RuntimeWindowControlSnapshot) -> Duration {
        Duration::from_secs_f32(1.0 / snapshot.target_tickrate.max(1.0))
    }

    #[cfg(target_arch = "wasm32")]
    fn resolve_wasm_canvas(&self) -> Option<HtmlCanvasElement> {
        if let Some(config) = self.wasm_harness.as_ref()
            && let Ok(canvas) = config.ensure_canvas()
        {
            return Some(canvas);
        }

        let window = web_sys::window()?;
        let document = window.document()?;
        document
            .get_element_by_id("helmer-canvas")
            .and_then(|element| element.dyn_into::<HtmlCanvasElement>().ok())
    }

    fn ensure_window(&mut self, event_loop: &ActiveEventLoop) -> Result<(), WindowError> {
        if self.window.is_some() {
            return Ok(());
        }

        let mut attributes: WindowAttributes = Window::default_attributes();
        attributes.title = self.config.title.clone();
        attributes.maximized = self.config.maximized;
        attributes.visible = self.config.visible;
        #[cfg(target_arch = "wasm32")]
        if let Some(canvas) = self.resolve_wasm_canvas() {
            attributes = attributes.with_canvas(Some(canvas));
        }

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .map_err(|err| WindowError::WindowCreation(err.to_string()))?,
        );
        let now = Instant::now();
        let state = Self::window_state(&window);
        {
            let mut input = self.input_manager.write();
            input.window_size = UVec2::new(state.width, state.height);
            input.scale_factor = state.scale_factor;
        }
        self.sync_window_control_from_window(&window);
        self.apply_runtime_window_control(&window);
        self.update_window_title(&window, now, true);
        self.apply_runtime_cursor_state(&window);
        self.emit(WindowRuntimeEventKind::Started {
            window: window.clone(),
            state,
        });
        self.last_frame = now;
        self.last_tick = now;
        self.next_tick_deadline = now;
        self.next_frame_deadline = now;
        self.last_title_update = now;
        window.request_redraw();
        self.window = Some(window);
        event_loop.set_control_flow(ControlFlow::Poll);
        Ok(())
    }

    fn sync_window_control_from_window(&mut self, window: &Window) {
        let fullscreen = window.fullscreen().is_some();
        let resizable = window.is_resizable();
        let decorations = window.is_decorated();
        let maximized = window.is_maximized();
        let minimized = window.is_minimized().unwrap_or(false);
        let visible = window.is_visible().unwrap_or(self.config.visible);

        self.window_control.set_fullscreen(fullscreen);
        self.window_control.set_resizable(resizable);
        self.window_control.set_decorations(decorations);
        self.window_control.set_maximized(maximized);
        self.window_control.set_minimized(minimized);
        self.window_control.set_visible(visible);

        self.applied_window_control.fullscreen = fullscreen;
        self.applied_window_control.resizable = resizable;
        self.applied_window_control.decorations = decorations;
        self.applied_window_control.maximized = maximized;
        self.applied_window_control.minimized = minimized;
        self.applied_window_control.visible = visible;
    }

    fn apply_runtime_window_control(&mut self, window: &Window) {
        let snapshot = self.window_control.snapshot();

        if self.applied_window_control.fullscreen != snapshot.fullscreen {
            window.set_fullscreen(if snapshot.fullscreen {
                Some(winit::window::Fullscreen::Borderless(None))
            } else {
                None
            });
            self.applied_window_control.fullscreen = snapshot.fullscreen;
        }
        if self.applied_window_control.resizable != snapshot.resizable {
            window.set_resizable(snapshot.resizable);
            self.applied_window_control.resizable = snapshot.resizable;
        }
        if self.applied_window_control.decorations != snapshot.decorations {
            window.set_decorations(snapshot.decorations);
            self.applied_window_control.decorations = snapshot.decorations;
        }
        if self.applied_window_control.maximized != snapshot.maximized {
            window.set_maximized(snapshot.maximized);
            self.applied_window_control.maximized = snapshot.maximized;
        }
        if self.applied_window_control.minimized != snapshot.minimized {
            window.set_minimized(snapshot.minimized);
            self.applied_window_control.minimized = snapshot.minimized;
        }
        if self.applied_window_control.visible != snapshot.visible {
            window.set_visible(snapshot.visible);
            self.applied_window_control.visible = snapshot.visible;
        }
    }

    fn compose_window_title(&self, snapshot: &RuntimeWindowControlSnapshot) -> String {
        let (fps, tps) = self
            .metrics
            .as_ref()
            .map(|metrics| {
                (
                    metrics.fps.load(std::sync::atomic::Ordering::Relaxed),
                    metrics.tps.load(std::sync::atomic::Ordering::Relaxed),
                )
            })
            .unwrap_or((0, 0));

        match snapshot.title_mode {
            RuntimeWindowTitleMode::Stats => format!("helmer engine | FPS: {} | TPS: {}", fps, tps),
            RuntimeWindowTitleMode::Custom => snapshot.custom_title.clone(),
            RuntimeWindowTitleMode::CustomWithStats => {
                if snapshot.custom_title.trim().is_empty() {
                    format!("FPS: {} | TPS: {}", fps, tps)
                } else {
                    format!("{} | FPS: {} | TPS: {}", snapshot.custom_title, fps, tps)
                }
            }
        }
    }

    fn update_window_title(&mut self, window: &Window, now: Instant, force: bool) {
        let snapshot = self.window_control.snapshot();
        let next_title = self.compose_window_title(&snapshot);
        let interval_ms = snapshot.title_update_ms.max(1);
        let should_refresh = force
            || match snapshot.title_mode {
                RuntimeWindowTitleMode::Stats | RuntimeWindowTitleMode::CustomWithStats => {
                    now.duration_since(self.last_title_update)
                        >= Duration::from_millis(interval_ms as u64)
                }
                RuntimeWindowTitleMode::Custom => next_title != self.last_window_title,
            };

        if should_refresh {
            if next_title != self.last_window_title {
                window.set_title(&next_title);
                self.last_window_title = next_title;
            }
            self.last_title_update = now;
        }
    }

    fn apply_runtime_cursor_state(&self, window: &Window) {
        if self.cursor_state.take_dirty() {
            let snapshot = self.cursor_state.snapshot();
            window.set_cursor_visible(snapshot.visible);
            let requested_mode = match snapshot.grab_mode {
                RuntimeCursorGrabMode::None => CursorGrabMode::None,
                RuntimeCursorGrabMode::Confined => CursorGrabMode::Confined,
                RuntimeCursorGrabMode::Locked => CursorGrabMode::Locked,
            };
            if window.set_cursor_grab(requested_mode).is_err() {
                if requested_mode == CursorGrabMode::Locked {
                    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                }
            }
        }
        if let Some((x, y)) = self.cursor_state.take_warp_request() {
            let _ = window.set_cursor_position(PhysicalPosition::new(x, y));
        }
    }
}

impl ApplicationHandler for NativeWindowApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(err) = self.ensure_window(event_loop) {
            self.emit(WindowRuntimeEventKind::CloseRequested);
            tracing::error!("{err}");
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref().map(Arc::clone) else {
            return;
        };
        if window.id() != window_id {
            return;
        }

        self.input_manager
            .read()
            .update_egui_state_from_winit(&event);

        match event {
            WindowEvent::CloseRequested => {
                self.emit(WindowRuntimeEventKind::CloseRequested);
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                let width = size.width.max(1);
                let height = size.height.max(1);
                let scale_factor = window.scale_factor();
                {
                    let mut input = self.input_manager.write();
                    input.window_size = UVec2::new(width, height);
                    input.scale_factor = scale_factor;
                }
                self.emit(WindowRuntimeEventKind::Resized(WindowState {
                    width,
                    height,
                    scale_factor,
                }));
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let PhysicalSize { width, height } = window.inner_size();
                let mut input = self.input_manager.write();
                input.window_size = UVec2::new(width.max(1), height.max(1));
                input.scale_factor = scale_factor;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::F11 && event.state.is_pressed() {
                        let fullscreen_enabled = window.fullscreen().is_some();
                        window.set_fullscreen(if fullscreen_enabled {
                            None
                        } else {
                            Some(winit::window::Fullscreen::Borderless(None))
                        });
                        self.sync_window_control_from_window(&window);
                    } else {
                        let pressed = event.state.is_pressed();
                        self.input_manager
                            .read()
                            .push_event(InputEvent::Keyboard { key, pressed });
                        self.emit(WindowRuntimeEventKind::Keyboard { key, pressed });
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state.is_pressed();
                self.input_manager
                    .read()
                    .push_event(InputEvent::MouseButton { button, pressed });
                self.emit(WindowRuntimeEventKind::MouseButton { button, pressed });
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cursor = DVec2::new(position.x, position.y);
                self.input_manager
                    .read()
                    .push_event(InputEvent::CursorMoved(cursor));
                self.emit(WindowRuntimeEventKind::MouseMoved(cursor));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let converted = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => Vec2::new(x, y),
                    winit::event::MouseScrollDelta::PixelDelta(position) => {
                        Vec2::new(position.x as f32 / 38.0, position.y as f32 / 38.0)
                    }
                };
                self.input_manager
                    .read()
                    .push_event(InputEvent::MouseWheel(converted));
                self.emit(WindowRuntimeEventKind::MouseWheel(converted));
            }
            WindowEvent::DroppedFile(path) => {
                self.emit(WindowRuntimeEventKind::DroppedFile(path));
            }
            WindowEvent::Focused(is_focused) => {
                if !is_focused {
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                    self.cursor_state.mark_dirty();
                    let mut input = self.input_manager.write();
                    input.clear_egui_state();
                    input.clear_queues();
                } else {
                    self.cursor_state.mark_dirty();
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                self.last_frame = now;
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let motion = DVec2::new(delta.0, delta.1);
            self.input_manager
                .read()
                .push_event(InputEvent::MouseMotion(motion));
            self.emit(WindowRuntimeEventKind::MouseMotion(motion));
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref().map(Arc::clone) {
            let mut now = Instant::now();
            self.apply_runtime_window_control(&window);
            self.apply_runtime_cursor_state(&window);
            self.update_window_title(&window, now, false);
            self.sync_window_control_from_window(&window);

            let snapshot = self.window_control.snapshot();
            if now >= self.next_tick_deadline {
                let dt = now.saturating_duration_since(self.last_tick).as_secs_f32();
                self.last_tick = now;
                self.emit(WindowRuntimeEventKind::Tick { dt });
                let interval = Self::tick_interval(&snapshot);
                let mut next_tick = self.next_tick_deadline + interval;
                if next_tick <= now {
                    next_tick = now + interval;
                }
                self.next_tick_deadline = next_tick;
                now = Instant::now();
            }

            if let Some(frame_interval) = Self::target_frame_interval(&snapshot) {
                if now >= self.next_frame_deadline {
                    window.request_redraw();
                    self.next_frame_deadline = now + frame_interval;
                }
                let next_deadline = self.next_tick_deadline.min(self.next_frame_deadline);
                event_loop.set_control_flow(ControlFlow::WaitUntil(next_deadline));
            } else {
                // uncapped frame mode should not force a busy main-thread poll loop
                event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_tick_deadline));
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}
