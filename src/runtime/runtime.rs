use core::time;
use glam::{DVec2, UVec2, Vec2};
use hashbrown::HashMap;
use resvg::{tiny_skia, usvg::Tree};
use std::{
    num::NonZeroU32,
    sync::{
        Arc, Barrier, Mutex, RwLock,
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
    runtime::input_manager::InputManager,
};

pub enum Message {
    Init(Window),
    Shutdown,
    RenderData(RenderData),
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

    // Threading
    logic_thread: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,

    // Communication
    render_receiver: mpsc::Receiver<Message>,

    frame_barrier: Arc<Barrier>,

    renderer: Option<Renderer>,
    target_fps: f32,

    // Window management
    window: Option<Arc<Window>>,
    initialized: bool,

    init_callback: fn(&mut Runtime),
}

impl Runtime {
    pub fn new(init_callback: fn(&mut Runtime)) -> Self {
        let (render_sender, render_receiver) = mpsc::channel();

        Self {
            ecs: Arc::new(RwLock::new(ECSCore::new())),

            input_manager: Arc::new(RwLock::new(InputManager::new())),

            logic_thread: None,
            running: Arc::new(AtomicBool::new(true)),

            render_receiver,

            frame_barrier: Arc::new(Barrier::new(2)),

            renderer: None,
            target_fps: 144.0,

            window: None,
            initialized: false,

            init_callback,
        }
    }

    pub fn init(&mut self) {
        // We need to move the sender into the logic thread
        let (render_sender, render_receiver) = mpsc::channel();
        self.render_receiver = render_receiver;

        self.start_logic_thread(render_sender);

        let event_loop = EventLoop::new().unwrap();
        let _ = event_loop.run_app(self);
    }

    fn start_logic_thread(&mut self, sender: mpsc::Sender<Message>) {
        let ecs = Arc::clone(&self.ecs);
        let input_manager = Arc::clone(&self.input_manager);
        //let scene_root = Arc::clone(&self.scene_root);
        let running = Arc::clone(&self.running);
        let frame_barrier = Arc::clone(&self.frame_barrier);
        let target_fps = self.target_fps;

        self.logic_thread = Some(thread::spawn(move || {
            let mut last_time = Instant::now();
            let frame_duration = Duration::from_secs_f32(1.0 / target_fps);

            // State tracking for interpolation.
            let mut previous_state: Option<ExtractedState> = None;

            while running.load(Ordering::Relaxed) {
                let now = Instant::now();
                let dt = (now - last_time).as_secs_f32();

                // Run ECS systems
                {
                    let mut ecs_guard = ecs.write().unwrap();
                    let input_manager_guard = input_manager.read().unwrap();

                    // 1. Temporarily take ownership of the scheduler, leaving a placeholder.
                    // This is a zero-cost operation that satisfies the borrow checker.
                    let mut scheduler = std::mem::replace(
                        &mut ecs_guard.system_scheduler,
                        SystemScheduler::new(), // A temporary, empty scheduler
                    );

                    // 2. Now we can call run_all. We pass the ECS data (without the real scheduler).
                    // The borrow checker is happy because `scheduler` and `ecs_guard` are separate variables.
                    scheduler.run_all(dt, &mut ecs_guard, &input_manager_guard);

                    // 3. Put the scheduler back where it belongs.
                    let _ = std::mem::replace(&mut ecs_guard.system_scheduler, scheduler);
                }

                let render_data_to_send = {
                    let mut ecs_guard = ecs.write().unwrap(); // A brief write lock to `.take()` the data
                    if let Some(packet) = ecs_guard.get_resource_mut::<RenderPacket>() {
                        packet.0.take() // .take() pulls the value out of the Option, leaving None
                    } else {
                        None
                    }
                };

                // Send render data if the system produced it
                if let Some(data) = render_data_to_send {
                    if sender.send(Message::RenderData(data)).is_err() {
                        warn!("frame dropped/render thread disconnected!");
                    }
                }

                input_manager.write().unwrap().prepare_for_next_frame();

                // Frame rate limiting
                let logic_elapsed = now.elapsed();
                let time_to_wait = frame_duration.saturating_sub(logic_elapsed);

                // Only sleep if we have more than a millisecond to spare.
                // This `sleep` is the coarse, low-CPU part.
                if time_to_wait > Duration::from_millis(1) {
                    // Sleep for most of the duration, but leave the last millisecond for spinning.
                    thread::sleep(time_to_wait - Duration::from_millis(1));
                }

                // This is the "spin" part for high-precision timing.
                // It will use 100% of its CPU core for the final moment to hit the target time exactly.
                while now.elapsed() < frame_duration {
                    // Hint to the OS that it can schedule other work.
                    thread::yield_now();
                }

                // Reset last_time for the next iteration's dt calculation
                last_time = now;

                /*let total_tick_duration = now.elapsed();
                tracing::info!(
                    "Target FPS: {:.2}, Actual Tick Time: {:.2}ms ({:.2} FPS)",
                    target_fps,
                    total_tick_duration.as_secs_f32() * 1000.0,
                    1.0 / total_tick_duration.as_secs_f32()
                );*/

                //frame_barrier.wait();
            }

            let _ = sender.send(Message::Shutdown);
            tracing::info!("Logic thread shutting down");
        }));

        info!("initialized logic thread");
    }

    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Relaxed);

        if let Some(handle) = self.logic_thread.take() {
            let _ = handle.join();
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
                self.shutdown();
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(window) = &self.window {
                    if let Some(renderer) = &mut self.renderer {
                        let mut latest_data: Option<RenderData> = None;
                        // Loop over `try_recv` to pull all pending messages from the channel.
                        while let Ok(message) = self.render_receiver.try_recv() {
                            match message {
                                Message::RenderData(data) => {
                                    // Keep overwriting our variable, so we only end up
                                    // with the very last, most recent RenderData packet.
                                    latest_data = Some(data);
                                }
                                Message::Shutdown => {
                                    self.running.store(false, Ordering::Relaxed);
                                    // Break the loop if we get a shutdown message
                                    break;
                                }
                                _ => {}
                            }
                        }

                        // After the loop, if we received any data at all, update the renderer
                        // with the newest available state.
                        if let Some(data) = latest_data {
                            renderer.update_render_data(data);
                        }

                        if let Err(e) = renderer.render() {
                            tracing::error!("Renderer error: {}", e);
                        }
                    }

                    if self.running.load(Ordering::Relaxed) == true {
                        //self.frame_barrier.wait();

                        window.request_redraw();
                    }
                }
            }
            WindowEvent::Resized(new_size) => {
                self.input_manager.write().unwrap().window_size =
                    UVec2::new(new_size.width, new_size.height);

                self.renderer.as_mut().unwrap().resize(new_size);

                if new_size.width > 0 && new_size.height > 0 {
                    let mut ecs_guard = self.ecs.write().unwrap();
                    ecs_guard
                        .component_pool
                        .query_mut_for_each::<(Camera, ActiveCamera), _>(|(camera, _)| {
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
                        _ => {
                            let mut input_manager_guard = self.input_manager.write().unwrap();
                            
                            if event.state.is_pressed() {
                                input_manager_guard.active_keys.insert(key_code.unwrap());
                            } else {
                                input_manager_guard.active_keys.remove(&key_code.unwrap());
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
            } => {
                let mut input_manager_guard = self.input_manager.write().unwrap();

                match state {
                    winit::event::ElementState::Pressed => {
                        input_manager_guard.active_mouse_buttons.insert(button);
                    }
                    winit::event::ElementState::Released => {
                        input_manager_guard.active_mouse_buttons.remove(&button);
                    }
                }
            }
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {
                let mut input_manager_guard = self.input_manager.write().unwrap();

                input_manager_guard.cursor_position = DVec2::from_array([position.x, position.y]);
            }
            WindowEvent::MouseWheel {
                device_id,
                delta,
                phase,
            } => {
                let mut input_manager_guard = self.input_manager.write().unwrap();

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

                input_manager_guard.add_scroll(scroll_delta);
            }
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let mut window = Window::default_attributes();
            window.title = "helmer engine — 2025 leighton".into();
            //window.fullscreen = Some(winit::window::Fullscreen::Borderless(None));

            self.window = Some(Arc::new(event_loop.create_window(window).unwrap()));

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

            // -----

            let window_size = self.window.as_ref().unwrap().inner_size();
            self.input_manager.write().unwrap().window_size =
                UVec2::new(window_size.width, window_size.height);
            if window_size.width > 0 && window_size.height > 0 {
                let mut ecs_guard = self.ecs.write().unwrap();
                ecs_guard
                    .component_pool
                    .query_mut_for_each::<(Camera, ActiveCamera), _>(|(camera, _)| {
                        camera.aspect_ratio = window_size.width as f32 / window_size.height as f32;
                    });
            }
        }

        if !self.initialized {
            self.renderer = Some(
                pollster::block_on(Renderer::new(
                    Arc::clone(self.window.as_ref().unwrap()),
                    self.target_fps,
                ))
                .unwrap(),
            );

            let cube_mesh = MeshAsset::cube("cube".into());
            let uv_sphere_mesh = MeshAsset::uv_sphere("uv sphere".into(), 32, 32);
            self.renderer.as_mut().unwrap().add_mesh(
                0,
                &cube_mesh.vertices.unwrap(),
                &cube_mesh.indices,
            );
            self.renderer.as_mut().unwrap().add_mesh(
                1,
                &uv_sphere_mesh.vertices.unwrap(),
                &uv_sphere_mesh.indices,
            );
            self.renderer
                .as_mut()
                .unwrap()
                .add_material(0, Material::default());

            (self.init_callback)(self);

            self.initialized = true;
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Main thread can do additional work here if needed
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
