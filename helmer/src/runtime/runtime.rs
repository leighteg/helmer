use egui::ViewportId;
use glam::{DVec2, UVec2};
use parking_lot::{Mutex, RwLock};
use resvg::tiny_skia;
use std::{
    env,
    num::NonZeroU32,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use wgpu::{BackendOptions, Dx12BackendOptions};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::{
    graphics::renderer_common::common::{
        EguiRenderData, RenderData, RenderMessage, RenderTrait, initialize_renderer,
    },
    runtime::{
        asset_server::AssetServer,
        config::RuntimeConfig,
        input_manager::{InputEvent, InputManager},
    },
};

#[derive(Default)]
pub struct PerformanceMetrics {
    pub fps: AtomicU32,
    pub tps: AtomicU32,
}

pub struct RuntimeCallbacks<T: Send + 'static> {
    init: Option<Box<dyn FnOnce(&mut Runtime<T>, &mut T) + Send>>,
    tick: Arc<dyn Fn(f32, &mut T) -> (Option<RenderData>, Option<EguiRenderData>) + Send + Sync>,
    resize: Arc<dyn Fn(PhysicalSize<u32>, &mut T) + Send + Sync>,
}

pub struct Runtime<T: Send + 'static = ()> {
    pub input_manager: Arc<RwLock<InputManager>>,
    pub asset_server: Option<Arc<Mutex<AssetServer>>>,

    // logic thread
    logic_thread: Option<JoinHandle<()>>,
    logic_thread_state: Arc<AtomicBool>,
    target_tickrate: f32,

    // render thread
    render_thread: Option<JoinHandle<()>>,
    render_thread_state: Arc<AtomicBool>,
    pub render_thread_sender: mpsc::Sender<RenderMessage>,
    target_fps: Option<f32>,

    // --- EGUI ---
    egui_winit_state: Option<egui_winit::State>,

    // Window management
    window: Option<Arc<Window>>,

    pub user_state: Option<Arc<Mutex<T>>>,
    callbacks: RuntimeCallbacks<T>,

    last_title_update: Instant,

    pub metrics: Arc<PerformanceMetrics>,
    pub config: Arc<RuntimeConfig>,

    has_init: Arc<AtomicBool>,

    new_window_size: Option<PhysicalSize<u32>>,
    resize_triggered: bool,
    last_resize: Instant,
}

impl<T: Send + 'static> Runtime<T> {
    pub fn new(
        user_state: T,
        init_callback: impl FnOnce(&mut Runtime<T>, &mut T) + Send + 'static,
        tick_callback: impl Fn(f32, &mut T) -> (Option<RenderData>, Option<EguiRenderData>)
        + Send
        + Sync
        + 'static,
        resize_callback: impl Fn(PhysicalSize<u32>, &mut T) + Send + Sync + 'static,
    ) -> Self {
        // -- TRACING SETUP

        #[cfg(windows)]
        colored::control::set_virtual_terminal(true).ok();

        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(tracing_subscriber::EnvFilter::new("helmer"))
            .try_init()
            .unwrap();

        tracing::info!("2025 leighton");

        // -- END TRACING SETUP

        let (render_sender, _) = mpsc::channel();

        Self {
            input_manager: Arc::new(RwLock::new(InputManager::new())),
            asset_server: None,

            logic_thread: None,
            logic_thread_state: Arc::new(AtomicBool::new(true)),
            target_tickrate: 120.0,

            render_thread: None,
            render_thread_state: Arc::new(AtomicBool::new(true)),
            render_thread_sender: render_sender,
            target_fps: None,

            egui_winit_state: None,

            window: None,

            user_state: Some(Arc::new(Mutex::new(user_state))),
            callbacks: RuntimeCallbacks {
                init: Some(Box::new(init_callback)),
                tick: Arc::new(tick_callback),
                resize: Arc::new(resize_callback),
            },

            last_title_update: Instant::now(),

            metrics: Arc::new(PerformanceMetrics::default()),
            config: Arc::new(RuntimeConfig::default()),

            has_init: Arc::new(AtomicBool::new(false)),

            new_window_size: None,
            resize_triggered: false,
            last_resize: Instant::now(),
        }
    }

    pub fn init(&mut self) {
        let event_loop = EventLoop::new().unwrap();
        let _ = event_loop.run_app(self);
    }

    fn start_logic_thread(&mut self) {
        let input_manager = Arc::clone(&self.input_manager);
        let asset_server = Arc::clone(&self.asset_server.as_ref().unwrap());
        let state = Arc::clone(&self.logic_thread_state);
        let target_tickrate = self.target_tickrate;
        let metrics = Arc::clone(&self.metrics);
        let tick_callback = Arc::clone(&self.callbacks.tick);
        let user_state = Arc::clone(self.user_state.as_ref().unwrap());
        let has_init = Arc::clone(&self.has_init);

        let sender = self.render_thread_sender.clone();

        self.logic_thread = Some(thread::spawn(move || {
            let mut last_time = Instant::now();
            let frame_duration = Duration::from_secs_f32(1.0 / target_tickrate);

            while state.load(Ordering::Relaxed) {
                let frame_start = Instant::now();
                let dt = (frame_start - last_time).as_secs_f32();

                let tps = (1.0 / dt).round() as u32;
                metrics.tps.store(tps, Ordering::Relaxed);

                //const MAX_DELTA_TIME: f32 = 1.0 / 30.0;
                //let dt = dt.min(MAX_DELTA_TIME);

                asset_server.lock().update();
                input_manager.write().process_events();

                if has_init.load(Ordering::Relaxed) {
                    // MAIN LOGIC LOOP EXECUTION
                    let mut user_state_guard = user_state.lock();
                    let (render_data, egui_render_data) = tick_callback(dt, &mut *user_state_guard);
                    drop(user_state_guard);

                    if let Some(data) = render_data {
                        if sender.send(RenderMessage::RenderData(data)).is_err() {
                            warn!("render thread disconnected");
                            break;
                        }
                    }

                    if let Some(data) = egui_render_data {
                        if sender.send(RenderMessage::EguiData(data)).is_err() {
                            warn!("render thread disconnected");
                            break;
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

                last_time = frame_start;
            }

            info!("logic thread shutting down");
        }));

        info!("initialized logic thread");
    }

    fn start_render_thread(&mut self, render_receiver: mpsc::Receiver<RenderMessage>) {
        let target_tickrate = self.target_tickrate;
        let target_fps = self.target_fps; // Use target_fps for render thread
        let state = Arc::clone(&self.render_thread_state);
        let window = Arc::clone(self.window.as_ref().unwrap());
        let window_size = window.inner_size();
        let metrics = Arc::clone(&self.metrics);

        let backend_str = env::var("HELMER_BACKEND").unwrap_or_else(|_| "all".to_string());

        let backends = match backend_str.to_lowercase().as_str() {
            "vulkan" => wgpu::Backends::VULKAN,
            "gl" => wgpu::Backends::GL,
            "dx12" => wgpu::Backends::DX12,
            "metal" => wgpu::Backends::METAL,
            _ => wgpu::Backends::all(),
        };

        if backends != wgpu::Backends::all() {
            info!("selected renderer backend: {:?}", backends);
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            backend_options: BackendOptions {
                dx12: Dx12BackendOptions {
                    shader_compiler: wgpu::Dx12Compiler::StaticDxc,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let render_thread_handle = thread::spawn(move || {
            let mut renderer = pollster::block_on(async {
                initialize_renderer(instance, surface, window_size, target_tickrate)
                    .await
                    .unwrap()
            });

            let frame_duration = Duration::from_secs_f32(1.0 / target_fps.unwrap_or(60.0));
            let mut last_render = Instant::now();

            while state.load(Ordering::Relaxed) {
                let frame_start = Instant::now();
                let mut should_render = false;

                let dt = frame_start.duration_since(last_render).as_secs_f32();
                let fps = (1.0 / dt).round() as u32;
                metrics.fps.store(fps, Ordering::Relaxed);

                renderer.resolve_pending_materials();

                while let Ok(message) = render_receiver.try_recv() {
                    match message {
                        RenderMessage::RenderData(_) => {
                            should_render = true;
                        }
                        RenderMessage::Resize(_) => {
                            should_render = true;
                        }
                        RenderMessage::Shutdown => {
                            renderer.process_message(message);
                            return;
                        }
                        _ => {}
                    }

                    renderer.process_message(message);
                }

                if target_fps.is_some() {
                    // Render at target FPS if we have new data OR if enough time has passed
                    let time_since_last_render = frame_start.duration_since(last_render);
                    if should_render || time_since_last_render >= frame_duration {
                        if let Err(e) = renderer.render() {
                            warn!("render error: {}", e);
                        }
                        last_render = frame_start;
                    }

                    // Sleep to maintain frame rate
                    let elapsed = frame_start.elapsed();
                    if elapsed < frame_duration {
                        thread::sleep(frame_duration - elapsed);
                    }
                } else {
                    if let Err(e) = renderer.render() {
                        warn!("render error: {}", e);
                    }
                    last_render = frame_start;
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

        let _ = self.render_thread_sender.send(RenderMessage::Shutdown);

        if let Some(handle) = self.render_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.logic_thread.take() {
            let _ = handle.join();
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
        const BRAND_SVG_DATA: &[u8] = include_bytes!("../../../brand/helmer.svg");
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
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let WindowEvent::RedrawRequested = event {
            if self.render_thread.is_none() {
                self.draw_splash();
                self.window.as_ref().unwrap().request_redraw();
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
                    self.new_window_size = Some(new_size);
                    self.resize_triggered = true;
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == KeyCode::F11 && event.state.is_pressed() {
                        let window = self.window.as_mut().unwrap();
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None);
                        } else {
                            window
                                .set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                        }
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
                        const PIXELS_PER_LINE: f32 = 38.0;
                        glam::vec2(
                            pos.x as f32 / PIXELS_PER_LINE,
                            pos.y as f32 / PIXELS_PER_LINE,
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

            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let mut window = Window::default_attributes();
            window.title = "helmer engine".into();
            //window.fullscreen = Some(winit::window::Fullscreen::Borderless(None));

            self.window = Some(Arc::new(event_loop.create_window(window).unwrap()));

            self.draw_splash();

            self.input_manager.write().scale_factor = self.window.as_ref().unwrap().scale_factor();

            // --- Initialize egui-winit ---
            let egui_context = egui::Context::default();
            self.egui_winit_state = Some(egui_winit::State::new(
                egui_context,
                ViewportId::default(),
                &self.window.as_ref().unwrap(),
                None,
                None,
                None,
            ));
        }

        if self.render_thread.is_none() && self.logic_thread.is_none() {
            let (sender, receiver) = mpsc::channel();
            self.render_thread_sender = sender;

            self.asset_server = Some(Arc::new(Mutex::new(AssetServer::new(
                self.render_thread_sender.clone(),
            ))));

            self.start_render_thread(receiver);
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let update_interval = std::time::Duration::from_millis(200);

        if now.duration_since(self.last_title_update) >= update_interval {
            if let Some(window) = &self.window {
                let fps = self.metrics.fps.load(Ordering::Relaxed);
                let tps = self.metrics.tps.load(Ordering::Relaxed);

                let new_title = format!("helmer engine | FPS: {} | TPS: {}", fps, tps);
                window.set_title(&new_title);
            }

            self.last_title_update = now;
        }

        if let Some(new_size) = self.new_window_size {
            if self.last_resize.elapsed() > Duration::from_millis(500) {
                if !self.resize_triggered {
                    // --- RESIZE LOGIC ---
                    let _ = self
                        .render_thread_sender
                        .send(RenderMessage::Resize(new_size));

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
