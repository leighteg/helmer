use crate::input_manager::InputManager;
use helmer::runtime::{RuntimeContext, RuntimeError, RuntimeExtension};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone)]
pub struct InputManagerResource(pub Arc<RwLock<InputManager>>);

pub struct InputExtension {
    manager: Arc<RwLock<InputManager>>,
}

impl Default for InputExtension {
    fn default() -> Self {
        Self {
            manager: Arc::new(RwLock::new(InputManager::new())),
        }
    }
}

impl InputExtension {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_manager(manager: Arc<RwLock<InputManager>>) -> Self {
        Self { manager }
    }

    pub fn manager(&self) -> Arc<RwLock<InputManager>> {
        Arc::clone(&self.manager)
    }
}

impl RuntimeExtension for InputExtension {
    fn name(&self) -> &'static str {
        "helmer_input"
    }

    fn on_register(&mut self, ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        ctx.resources()
            .insert(InputManagerResource(Arc::clone(&self.manager)));
        Ok(())
    }
}
