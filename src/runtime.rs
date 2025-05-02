use core::time;
use std::sync::{Arc, RwLock};
use winit::{application::ApplicationHandler, event_loop::EventLoop};

use crate::{ecs::ecs_core::ECSCore, renderer::mev_renderer::Renderer};

pub struct Runtime {
    pub ecs: Arc<RwLock<ECSCore>>,
    renderer: Renderer,
    init_callback: fn(&mut Runtime),
    initialized: bool,
}

impl Runtime {
    pub fn new(init_callback: fn(&mut Runtime)) -> Self {
        Self {
            ecs: Arc::new(RwLock::new(ECSCore::new())),
            renderer: Renderer::new(),
            init_callback,
            initialized: false,
        }
    }

    pub fn init(&mut self) {
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
        self.renderer.handle_window_event(event_loop, window_id, event, &mut self.ecs.write().unwrap());
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if !self.initialized {
            let _ = (self.init_callback)(self);

            self.initialized = true;
        }
        
        self.renderer.initialize(event_loop);
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.ecs.write().unwrap().system_manager.run_all(&self.ecs);
    }
}