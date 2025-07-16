use core::time;
use glam::{DVec2, UVec2, Vec2};
use hashbrown::HashMap;
use parking_lot::{Mutex, RwLock};
use resvg::{tiny_skia, usvg::Tree};
use std::{
    num::NonZeroU32,
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tracing::{info, warn};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{MouseScrollDelta, TouchPhase, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    platform::modifier_supplement::KeyEventExtModifierSupplement,
    window::{Window, WindowId},
};

use crate::{
    ecs::{ecs_core::ECSCore, system_scheduler::SystemScheduler},
    graphics::{
        renderer::renderer::{Material, RenderData, RenderLight, RenderObject, Renderer, Vertex},
        renderer_system::RenderPacket,
    },
    provided::components::{ActiveCamera, Camera, Light, MeshAsset, MeshRenderer, Transform},
    runtime::{asset_server::{AssetKind, AssetServer, MaterialGpuData}, input_manager::{InputEvent, InputManager}},
};

pub enum RenderMessage {
    RenderData(RenderData),
    Resize(PhysicalSize<u32>),
    Shutdown,

    // --- Asset Pipeline Messages ---
    
    CreateMesh {
        id: usize,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    },
    CreateTexture {
        id: usize,
        name: String, // The file path for deduplication
        kind: AssetKind,
        data: Vec<u8>,
        format: wgpu::TextureFormat,
    },
    CreateMaterial(MaterialGpuData),
}

#[derive(Clone, Default)]
struct ExtractedState {
    objects: HashMap<usize, Transform>,
    lights: HashMap<usize, Transform>,
    camera_transform: Transform,
    camera_component: Camera,
}

pub struct Runtime {
    pub ecs: Arc<RwLock<ECSCore>>,
    //scene_root: Arc<RwLock<SceneNode>>,
    input_manager: Arc<RwLock<InputManager>>,
    pub asset_server: Arc<Mutex<AssetServer>>,

    // logic thread
    logic_thread: Option<JoinHandle<()>>,
    logic_thread_state: Arc<AtomicBool>,
    target_tickrate: f32,

    // render thread
    render_thread: Option<JoinHandle<()>>,
    render_thread_state: Arc<AtomicBool>,
    pub render_thread_sender: mpsc::Sender<RenderMessage>,
    target_fps: Option<f32>,

    // Window management
    window: Option<Arc<Window>>,

    init_callback: fn(&mut Runtime),
}

impl Runtime {
    pub fn new(init_callback: fn(&mut Runtime)) -> Self {
        let (render_sender, render_receiver) = mpsc::channel();

        Self {
            ecs: Arc::new(RwLock::new(ECSCore::new())),

            input_manager: Arc::new(RwLock::new(InputManager::new())),
            asset_server: Arc::new(Mutex::new(AssetServer::new(render_sender.clone()))),

            logic_thread: None,
            logic_thread_state: Arc::new(AtomicBool::new(true)),
            target_tickrate: 60.0,

            render_thread: None,
            render_thread_state: Arc::new(AtomicBool::new(true)),
            render_thread_sender: render_sender,
            target_fps: None,

            window: None,

            init_callback,
        }
    }

    pub fn init(&mut self) {
        let event_loop = EventLoop::new().unwrap();
        let _ = event_loop.run_app(self);
    }

    fn start_logic_thread(&mut self) {
        let ecs = Arc::clone(&self.ecs);
        let input_manager = Arc::clone(&self.input_manager);
        let asset_server = Arc::clone(&self.asset_server);
        let state = Arc::clone(&self.logic_thread_state);
        let target_tickrate = self.target_tickrate;

        let sender = self.render_thread_sender.clone();

        self.logic_thread = Some(thread::spawn(move || {
            let mut last_time = Instant::now();
            let frame_duration = Duration::from_secs_f32(1.0 / target_tickrate);

            while state.load(Ordering::Relaxed) {
                let frame_start = Instant::now();
                let dt = (frame_start - last_time).as_secs_f32();
                
                const MAX_DELTA_TIME: f32 = 1.0 / 30.0;
                let dt = dt.min(MAX_DELTA_TIME);

                asset_server.lock().update();
                input_manager.write().process_events();

                // Run ECS systems
                {
                    let mut ecs_guard = ecs.write();
                    let input_manager_guard = input_manager.read();

                    let mut scheduler =
                        std::mem::replace(&mut ecs_guard.system_scheduler, SystemScheduler::new());

                    scheduler.run_all(dt, &mut ecs_guard, &input_manager_guard);
                    let _ = std::mem::replace(&mut ecs_guard.system_scheduler, scheduler);
                }

                // Send render data if available - use regular send, not try_send
                let render_data_to_send = {
                    let mut ecs_guard = ecs.write();
                    if let Some(packet) = ecs_guard.get_resource_mut::<RenderPacket>() {
                        packet.0.take()
                    } else {
                        None
                    }
                };

                if let Some(data) = render_data_to_send {
                    if sender.send(RenderMessage::RenderData(data)).is_err() {
                        warn!("render thread disconnected");
                        break;
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

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let render_thread_handle = thread::spawn(move || {
            let mut renderer = pollster::block_on(Renderer::new(
                instance,
                surface,
                window_size,
                target_tickrate,
            ))
            .unwrap();

            let frame_duration = Duration::from_secs_f32(1.0 / target_fps.unwrap_or(60.0));
            let mut last_render = Instant::now();

            while state.load(Ordering::Relaxed) {
                let frame_start = Instant::now();                
                let mut should_render = false;

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
        const BRAND_SVG_DATA: &[u8] = include_bytes!("../../brand/helmer.svg");
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
        let offset_x = (window_size.width - scaled_width) / 2;
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

impl ApplicationHandler for Runtime {
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.shutdown_threads();
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {}
            WindowEvent::Resized(new_size) => {
                let _ = self
                    .render_thread_sender
                    .send(RenderMessage::Resize(new_size));

                self.input_manager.write().window_size =
                    UVec2::new(new_size.width, new_size.height);

                if new_size.width > 0 && new_size.height > 0 {
                    let mut ecs_guard = self.ecs.write();
                    ecs_guard
                        .component_pool
                        .query_mut_for_each::<(Camera, ActiveCamera), _>(|_, (camera, _)| {
                            camera.aspect_ratio = new_size.width as f32 / new_size.height as f32;
                        });
                }
            }

            // INPUT HANDLING
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {
                let mut key_code: Option<KeyCode> = None;

                match event.physical_key {
                    PhysicalKey::Code(code) => key_code = Some(code),
                    _ => {}
                }

                if key_code.is_some() {
                    match key_code.unwrap() {
                        KeyCode::F11 => {
                            if event.state.is_pressed() {
                                let window: &mut Arc<Window> = self.window.as_mut().unwrap();

                                if window.fullscreen().is_some() {
                                    window.set_fullscreen(None);
                                } else {
                                    window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(None),
                                    ));
                                }
                            }
                        }
                        key_code => {
                            self.input_manager.read().push_event(InputEvent::Keyboard {
                                key: key_code,
                                pressed: event.state.is_pressed(),
                            });
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
            } => {
                self.input_manager
                    .read()
                    .push_event(InputEvent::MouseButton {
                        button,
                        pressed: state.is_pressed(),
                    });
            }
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {
                self.input_manager
                    .read()
                    .push_event(InputEvent::CursorMoved(DVec2::from_array([
                        position.x, position.y,
                    ])));
            }
            WindowEvent::MouseWheel {
                device_id,
                delta,
                phase,
            } => {
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
        }

        if self.render_thread.is_none() && self.logic_thread.is_none() {
            let (sender, receiver) = mpsc::channel();
            self.render_thread_sender = sender;

            self.asset_server = Arc::new(Mutex::new(AssetServer::new(self.render_thread_sender.clone())));

            self.start_render_thread(receiver);
            self.start_logic_thread();

            (self.init_callback)(self);
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}
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
