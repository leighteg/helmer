use crate::service::WindowService;
use crate::state::{RuntimeCursorState, RuntimeWindowControl};
use helmer::runtime::{RuntimeContext, RuntimeError, RuntimeExtension};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
pub struct WindowCursorStateResource(pub Arc<RuntimeCursorState>);

#[derive(Clone)]
pub struct WindowControlResource(pub Arc<RuntimeWindowControl>);

#[derive(Clone)]
pub struct WindowMainThreadServiceResource(pub Arc<Mutex<Option<WindowService>>>);

pub struct WindowExtension {
    service: Arc<Mutex<Option<WindowService>>>,
    started: Arc<AtomicBool>,
    cursor_state: Arc<RuntimeCursorState>,
    window_control: Arc<RuntimeWindowControl>,
}

impl WindowExtension {
    pub fn new(service: WindowService) -> Self {
        let cursor_state = Arc::new(RuntimeCursorState::default());
        let window_control = Arc::new(RuntimeWindowControl::default());
        let service = service.with_runtime_state(cursor_state.clone(), window_control.clone());
        Self {
            service: Arc::new(Mutex::new(Some(service))),
            started: Arc::new(AtomicBool::new(false)),
            cursor_state,
            window_control,
        }
    }

    pub fn started_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.started)
    }
}

impl RuntimeExtension for WindowExtension {
    fn name(&self) -> &'static str {
        "helmer_window"
    }

    fn on_register(&mut self, ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        ctx.resources()
            .insert(WindowCursorStateResource(self.cursor_state.clone()));
        ctx.resources()
            .insert(WindowControlResource(self.window_control.clone()));
        ctx.resources()
            .insert(WindowMainThreadServiceResource(Arc::clone(&self.service)));
        Ok(())
    }

    fn on_start(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        if self.started.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        Ok(())
    }

    fn on_stop(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        self.started.store(false, Ordering::Release);
        Ok(())
    }
}
