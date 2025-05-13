use core::time;
use std::{
    sync::{Arc, Mutex, RwLock},
    thread,
};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::Window,
};

use crate::{ecs::ecs_core::ECSCore, graphics::renderer::Renderer};

pub struct Runtime {
    pub ecs: Arc<RwLock<ECSCore>>,
    window: Option<Window>,
    renderer: Arc<Mutex<Option<Renderer>>>,
    init_callback: fn(&mut Runtime),
    initialized: bool,
}

impl Runtime {
    pub fn new(init_callback: fn(&mut Runtime)) -> Self {
        Self {
            ecs: Arc::new(RwLock::new(ECSCore::new())),
            window: None,
            renderer: Arc::new(Mutex::new(None)),
            init_callback,
            initialized: false,
        }
    }

    pub fn init(&mut self) {
        // renderer init
        let renderer = Arc::clone(&self.renderer);
        let renderer_handle = thread::spawn(move || {
            let mut guard = renderer
                .lock()
                .expect("Failed to lock renderer mutex in init thread");
            // *guard gives mutable access to the Option<Renderer>
            *guard = Some(Renderer::new()); // Create Renderer and move it into the Option
            println!("Renderer initialized in separate thread.");
        });
        // event loop init
        let event_loop = EventLoop::new().unwrap();
        let _ = event_loop.run_app(self);
    }
}

impl ApplicationHandler for Runtime {
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {}
            WindowEvent::RedrawRequested => {
                let mut renderer_guard = self
                    .renderer
                    .lock()
                    .expect("Failed to lock renderer for drawing");

                if let Some(renderer_instance) = renderer_guard.as_mut() {
                    if let Some(window) = self.window.as_mut() {
                        // Or .as_ref() if draw_frame takes &Window
                        renderer_instance.draw_frame(window);
                    } else {
                        // This case should ideally not be hit if redraw is requested for an existing window
                        eprintln!("Window not found for drawing during RedrawRequested.");
                    }
                } else {
                    eprintln!("Renderer not initialized during RedrawRequested.");
                }
            }
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            self.window = Some(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );
        }

        if !self.initialized {
            let _ = (self.init_callback)(self);

            self.initialized = true;
        }

        let mut renderer_guard = self
            .renderer
            .lock()
            .expect("Failed to lock renderer for drawing");

        if let Some(renderer_instance) = renderer_guard.as_mut() {
            if let Some(window) = self.window.as_mut() {
                // Or .as_ref() if draw_frame takes &Window
                renderer_instance.draw_frame(window);
            } else {
                // This case should ideally not be hit if redraw is requested for an existing window
                eprintln!("Window not found for drawing during RedrawRequested.");
            }
        } else {
            eprintln!("Renderer not initialized during RedrawRequested.");
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.ecs.write().unwrap().system_manager.run_all(&self.ecs);
    }
}
