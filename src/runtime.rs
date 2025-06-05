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
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use crate::{
    ecs::ecs_core::ECSCore,
    graphics::renderer::renderer::{
        Material, RenderData, RenderLight, RenderObject, Renderer, Vertex,
    },
    provided::components::{Light, MeshAsset, MeshRenderer, Transform},
};

pub enum Message {
    Init(Window),
    Shutdown,
    RenderData(RenderData),
}

pub struct Runtime {
    pub ecs: Arc<RwLock<ECSCore>>,
    //scene_root: Arc<RwLock<SceneNode>>,

    // Threading
    logic_thread: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,

    // Communication
    render_receiver: mpsc::Receiver<Message>,

    frame_barrier: Arc<Barrier>,

    renderer: Option<Renderer>,

    // Window management
    window: Option<Window>,
    initialized: bool,

    init_callback: fn(&mut Runtime),
}

impl Runtime {
    pub fn new(init_callback: fn(&mut Runtime)) -> Self {
        let (render_sender, render_receiver) = mpsc::channel();

        Self {
            ecs: Arc::new(RwLock::new(ECSCore::new())),

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

                    ecs_guard.system_manager.run_all(&ecs);

                    // Update scene graph
                    /*
                    {
                        let mut root = scene_root.write().unwrap();
                        let identity = Transform {
                            position: [0.0, 0.0, 0.0],
                            rotation: [0.0, 0.0, 0.0, 1.0],
                            scale: [1.0, 1.0, 1.0],
                        };
                        root.update_transforms(&ecs_guard, &identity);
                    }
                    */
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

                    let mut camera_transform = Transform::default();
                    camera_transform.position = [0.0, 0.0, 3.0].into(); // Move camera back along Z
                    //camera_transform.rotation.y = 20.0;

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

                frame_barrier.wait();

                // Frame rate limiting
                let elapsed = now.elapsed();
                if elapsed < frame_duration {
                    thread::sleep(frame_duration - elapsed);
                }
            }

            let _ = sender.send(Message::Shutdown);
            tracing::info!("Logic thread shutting down");
        }));
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

                        renderer.render(window);
                    }

                    if self.running.load(Ordering::Relaxed) == true {
                        self.frame_barrier.wait();

                        window.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.window = Some(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );
        }

        if !self.initialized {
            self.renderer = Some(Renderer::new(&self.window.as_ref().unwrap()).unwrap());
            self.renderer
                .as_mut()
                .unwrap()
                .render(&self.window.as_mut().unwrap());
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
