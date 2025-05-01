use winit::{application::ApplicationHandler, event_loop::EventLoop};

use crate::{ecs::ecs_core::ECSCore, renderer::mev_renderer::Renderer};

pub struct Runtime {
    pub ecs: ECSCore,
    renderer: Renderer,
}

impl Runtime {
    pub fn new() -> Self {
        let event_loop = EventLoop::new().unwrap();
        let mut instance = Self {
            ecs: ECSCore::new(),
            renderer: Renderer::new(),
        };
        let _ = event_loop.run_app(&mut instance);
        instance
    }
}

impl ApplicationHandler for Runtime {
    fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            window_id: winit::window::WindowId,
            event: winit::event::WindowEvent,
        ) {
        self.renderer.handle_window_event(event_loop, window_id, event, &mut self.ecs);
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.renderer.initialize(event_loop);
    }
}