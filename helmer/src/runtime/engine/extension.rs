use super::{RuntimeError, RuntimeResources, TaskPool, ThreadRegistry};
use std::sync::Arc;
use std::time::Duration;

pub struct RuntimeContext {
    pub(crate) resources: Arc<RuntimeResources>,
    pub(crate) threads: Arc<ThreadRegistry>,
    pub(crate) task_pool: Arc<TaskPool>,
    pub(crate) single_threaded: bool,
    pub(crate) available_parallelism: usize,
}

impl RuntimeContext {
    pub fn resources(&self) -> &RuntimeResources {
        &self.resources
    }

    pub fn threads(&self) -> &ThreadRegistry {
        &self.threads
    }

    pub fn task_pool(&self) -> &TaskPool {
        &self.task_pool
    }

    pub fn is_single_threaded(&self) -> bool {
        self.single_threaded
    }

    pub fn available_parallelism(&self) -> usize {
        self.available_parallelism
    }
}

pub trait RuntimeExtension: Send {
    fn name(&self) -> &'static str;

    fn on_register(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn on_start(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn on_tick(&mut self, _ctx: &RuntimeContext, _dt: Duration) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn on_stop(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        Ok(())
    }
}
