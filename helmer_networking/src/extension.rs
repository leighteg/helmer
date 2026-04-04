use std::sync::Arc;

use helmer::runtime::{RuntimeContext, RuntimeError, RuntimeExtension};

use crate::NetworkHub;

#[derive(Clone)]
pub struct NetworkingResource(pub Arc<NetworkHub>);

pub struct NetworkingExtension {
    hub: Arc<NetworkHub>,
}

impl NetworkingExtension {
    pub fn new() -> Self {
        Self {
            hub: Arc::new(NetworkHub::new()),
        }
    }

    pub fn hub(&self) -> Arc<NetworkHub> {
        Arc::clone(&self.hub)
    }
}

impl Default for NetworkingExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeExtension for NetworkingExtension {
    fn name(&self) -> &'static str {
        "helmer_networking"
    }

    fn on_register(&mut self, ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        ctx.resources()
            .insert(NetworkingResource(Arc::clone(&self.hub)));
        Ok(())
    }

    fn on_stop(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        self.hub.shutdown();
        Ok(())
    }
}
