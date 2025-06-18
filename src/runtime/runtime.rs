use core::time;
use std::{
    sync::{
        Arc, Barrier, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use glam::{DVec2, Vec2};
use tracing::info;
use winit::{
    application::ApplicationHandler, event::{MouseScrollDelta, TouchPhase, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, platform::modifier_supplement::KeyEventExtModifierSupplement, window::{Window, WindowId}
};

use crate::{
    ecs::{ecs_core::ECSCore, system_scheduler::SystemScheduler},
    graphics::renderer::renderer::{
        Material, RenderData, RenderLight, RenderObject, Renderer, Vertex,
    },
    provided::components::{Camera, Light, MeshAsset, MeshRenderer, Transform},
    runtime::input_manager::InputManager,
};

pub enum Message {
    Init(Window),
    Shutdown,
    RenderData(RenderData),
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

        self.logic_thread = Some(thread::spawn(move || {
            let mut last_time = Instant::now();
            let target_fps = 60.0;
            let frame_duration = Duration::from_secs_f32(1.0 / target_fps);

            while running.load(Ordering::Relaxed) {
                let now = Instant::now();
                let dt = (now - last_time).as_secs_f32();
                last_time = now;

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

                // Extract render data
                let render_data = {
                    let ecs_guard = ecs.read().unwrap();
                    let mut objects = Vec::new();
                    let mut lights = Vec::new();

                    // Collect renderable objects
                    for (entity, mesh) in
                        ecs_guard.get_all_entities_with_component::<MeshRenderer>()
                    {
                        if let Some(transform) = ecs_guard.get_component::<Transform>(entity) {
                            objects.push(RenderObject {
                                transform: *transform,
                                mesh_id: mesh.mesh_id,
                                material_id: mesh.material_id,
                            });
                        }
                    }

                    // Collect lights
                    for (entity, light) in ecs_guard.get_all_entities_with_component::<Light>() {
                        if let Some(transform) = ecs_guard.get_component::<Transform>(entity) {
                            lights.push(RenderLight {
                                transform: *transform,
                                color: light.color.into(),
                                intensity: light.intensity,
                                light_type: light.light_type,
                            });
                        }
                    }

                    let mut camera_transform = Transform { // default camera transform (if world contains no cameras)
                        position: glam::Vec3::new(0.0, 0.0, -3.0), // Camera 3 units back on Z axis
                        rotation: glam::Quat::IDENTITY, // Looking down negative Z (forward for camera)
                        scale: glam::Vec3::ONE,
                    };

                    let camera_entities = ecs_guard.component_pool.query_exact::<(Transform, Camera)>();
                    for (transform, _) in camera_entities {
                        camera_transform = *transform;
                    }

                    RenderData {
                        objects,
                        lights,
                        camera_transform,
                    }
                };

                // Send render data to main thread (non-blocking)
                if sender.send(Message::RenderData(render_data)).is_err() {
                    // Channel full or disconnected, skip this frame's render data
                    // This prevents the logic thread from blocking
                }

                //frame_barrier.wait();

                // Frame rate limiting
                let elapsed = now.elapsed();
                if elapsed < frame_duration {
                    thread::sleep(frame_duration - elapsed);
                }
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
                        match self.render_receiver.try_recv() {
                            Ok(Message::RenderData(data)) => {
                                renderer.update_render_data(data);
                            }
                            _ => {}
                        }

                        renderer.render().unwrap(); // literally everything lacks proper err handling. todo: refactor all non-critical unwraps
                    }

                    if self.running.load(Ordering::Relaxed) == true {
                        //self.frame_barrier.wait();

                        window.request_redraw();
                    }
                }
            }
            WindowEvent::Resized(new_size) => {
                self.renderer.as_mut().unwrap().resize(new_size);
            }

            // INPUT HANDLING
            WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                let mut key_code: Option<KeyCode> = None;

                match event.physical_key {
                    PhysicalKey::Code(code) => key_code = Some(code),
                    _ => {}
                }

                let mut input_manager_guard = self.input_manager.write().unwrap();

                if key_code.is_some() {
                    if event.state.is_pressed() {
                        input_manager_guard.active_keys.insert(key_code.unwrap());
                    }
                    else {
                        input_manager_guard.active_keys.remove(&key_code.unwrap());
                    }
                }
            }
            WindowEvent::MouseInput { device_id, state, button } => {
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
            WindowEvent::CursorMoved { device_id, position } => {
                let mut input_manager_guard = self.input_manager.write().unwrap();

                input_manager_guard.cursor_position = DVec2::from_array([position.x, position.y]);
            }
            WindowEvent::MouseWheel { device_id, delta, phase } => {
                let mut input_manager_guard = self.input_manager.write().unwrap();

                match phase {
                    TouchPhase::Started | TouchPhase::Moved => {
                        match delta {
                            MouseScrollDelta::LineDelta(x, y) => {
                                input_manager_guard.mouse_wheel = Vec2::from_array([x, y]);
                            }
                            _ => {}
                        }
                    }
                    TouchPhase::Cancelled | TouchPhase::Ended => {
                        input_manager_guard.mouse_wheel = Vec2::ZERO;
                    }
                }
            }
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let mut window = Window::default_attributes();
            window.title = "helmer engine — 2025 leighton tegland".into();
            
            self.window = Some(Arc::new(
                event_loop
                    .create_window(window)
                    .unwrap(),
            ));
        }

        if !self.initialized {
            self.renderer = Some(pollster::block_on(Renderer::new(Arc::clone(self.window.as_ref().unwrap()))).unwrap());

            let cube_mesh = MeshAsset::cube("cube mesh".into());
            self.renderer.as_mut().unwrap().add_mesh(
                0,
                &cube_mesh.vertices.unwrap(),
                &cube_mesh.indices,
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
