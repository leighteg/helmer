use super::{
    RuntimeContext, RuntimeError, RuntimeExtension, RuntimeResources, TaskPool, ThreadRegistry,
};
use std::env;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
use std::time::Duration;

pub struct RuntimeBuilder {
    worker_count: Option<usize>,
    single_threaded: Option<bool>,
    available_parallelism: usize,
}

fn parse_bool_env(value: &str) -> Option<bool> {
    let normalized = value.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "1" | "true" | "yes" | "on") {
        return Some(true);
    }
    if matches!(normalized.as_str(), "0" | "false" | "no" | "off") {
        return Some(false);
    }
    None
}

fn runtime_single_thread_env_override() -> Option<bool> {
    for key in [
        "HELMER_RUNTIME_SINGLE_THREADED",
        "HELMER_FORCE_SINGLE_THREADED",
    ] {
        if let Ok(value) = env::var(key)
            && let Some(parsed) = parse_bool_env(&value)
        {
            return Some(parsed);
        }
    }
    None
}

fn detect_available_parallelism() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .max(1)
    }
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self {
            worker_count: None,
            single_threaded: None,
            available_parallelism: detect_available_parallelism(),
        }
    }
}

impl RuntimeBuilder {
    /// Configure worker threads for the task pool.
    ///
    /// Values below `1` are clamped to `1`.
    pub fn with_worker_count(mut self, worker_count: usize) -> Self {
        self.worker_count = Some(worker_count.max(1));
        self
    }

    /// Force inline task execution with no worker threads.
    pub fn with_inline_tasks(mut self) -> Self {
        self.worker_count = Some(0);
        self
    }

    /// Configure whether runtime-managed subsystems should prefer single-thread execution.
    pub fn with_single_threaded(mut self, single_threaded: bool) -> Self {
        self.single_threaded = Some(single_threaded);
        if single_threaded {
            self.worker_count = Some(0);
        }
        self
    }

    pub fn build(self) -> Result<Runtime, RuntimeError> {
        let env_single = runtime_single_thread_env_override().unwrap_or(false);
        let single_threaded = self.single_threaded.unwrap_or_else(|| {
            cfg!(target_arch = "wasm32") || self.available_parallelism <= 1 || env_single
        }) || env_single;
        let default_worker_count = if single_threaded || cfg!(target_arch = "wasm32") {
            0
        } else {
            self.available_parallelism.saturating_sub(1).max(1)
        };
        let worker_count = if single_threaded || cfg!(target_arch = "wasm32") {
            0
        } else {
            self.worker_count.unwrap_or(default_worker_count)
        };
        Runtime::new_with_policy(worker_count, single_threaded, self.available_parallelism)
    }
}

pub struct Runtime {
    context: RuntimeContext,
    extensions: Vec<Box<dyn RuntimeExtension>>,
    started: bool,
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    pub fn new(worker_count: usize) -> Result<Self, RuntimeError> {
        let available_parallelism = detect_available_parallelism();
        let env_single = runtime_single_thread_env_override().unwrap_or(false);
        let single_threaded =
            cfg!(target_arch = "wasm32") || available_parallelism <= 1 || env_single;
        let worker_count = if single_threaded || cfg!(target_arch = "wasm32") {
            0
        } else {
            worker_count
        };
        Self::new_with_policy(worker_count, single_threaded, available_parallelism)
    }

    fn new_with_policy(
        worker_count: usize,
        single_threaded: bool,
        available_parallelism: usize,
    ) -> Result<Self, RuntimeError> {
        let resources = Arc::new(RuntimeResources::default());
        let threads = Arc::new(ThreadRegistry::default());
        let task_pool = Arc::new(TaskPool::new(worker_count)?);
        let context = RuntimeContext {
            resources,
            threads,
            task_pool,
            single_threaded,
            available_parallelism,
        };
        Ok(Self {
            context,
            extensions: Vec::new(),
            started: false,
        })
    }

    pub fn context(&self) -> &RuntimeContext {
        &self.context
    }

    pub fn register_extension<E>(&mut self, mut extension: E) -> Result<(), RuntimeError>
    where
        E: RuntimeExtension + 'static,
    {
        extension.on_register(&self.context)?;
        self.extensions.push(Box::new(extension));
        Ok(())
    }

    pub fn start(&mut self) -> Result<(), RuntimeError> {
        if self.started {
            return Err(RuntimeError::AlreadyStarted);
        }

        for extension in &mut self.extensions {
            extension
                .on_start(&self.context)
                .map_err(|err| RuntimeError::ExtensionStart {
                    extension: extension.name(),
                    reason: err.to_string(),
                })?;
        }

        self.started = true;
        Ok(())
    }

    pub fn tick(&mut self, dt: Duration) -> Result<(), RuntimeError> {
        if !self.started {
            return Err(RuntimeError::NotStarted);
        }
        for extension in &mut self.extensions {
            extension.on_tick(&self.context, dt)?;
        }
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<(), RuntimeError> {
        if !self.started {
            return Ok(());
        }

        for extension in self.extensions.iter_mut().rev() {
            extension
                .on_stop(&self.context)
                .map_err(|err| RuntimeError::ExtensionStop {
                    extension: extension.name(),
                    reason: err.to_string(),
                })?;
        }

        self.context.task_pool.shutdown()?;
        self.context.threads.join_all()?;
        self.started = false;
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

pub struct RuntimeHandle<'a> {
    runtime: &'a mut Runtime,
}

impl<'a> RuntimeHandle<'a> {
    pub fn new(runtime: &'a mut Runtime) -> Self {
        Self { runtime }
    }

    pub fn context(&self) -> &RuntimeContext {
        self.runtime.context()
    }

    pub fn tick(&mut self, dt: Duration) -> Result<(), RuntimeError> {
        self.runtime.tick(dt)
    }
}
