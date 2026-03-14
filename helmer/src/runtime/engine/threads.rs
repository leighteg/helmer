use super::RuntimeError;
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

pub struct ThreadHandle {
    name: String,
    join: Option<JoinHandle<()>>,
}

impl ThreadHandle {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn join(mut self) -> thread::Result<()> {
        if let Some(join) = self.join.take() {
            join.join()
        } else {
            Ok(())
        }
    }
}

#[derive(Default)]
pub struct ThreadRegistry {
    threads: Mutex<Vec<ThreadHandle>>,
}

impl ThreadRegistry {
    pub fn spawn_named<F>(&self, name: impl Into<String>, work: F) -> Result<(), RuntimeError>
    where
        F: FnOnce() + Send + 'static,
    {
        let name = name.into();
        let builder = thread::Builder::new().name(name.clone());
        let join = builder
            .spawn(work)
            .map_err(|err| RuntimeError::ExtensionStart {
                extension: "thread_registry",
                reason: err.to_string(),
            })?;

        if let Ok(mut guard) = self.threads.lock() {
            guard.push(ThreadHandle {
                name,
                join: Some(join),
            });
        }
        Ok(())
    }

    pub fn join_all(&self) -> Result<(), RuntimeError> {
        let mut handles = Vec::new();
        if let Ok(mut guard) = self.threads.lock() {
            handles.extend(guard.drain(..));
        }

        for handle in handles {
            if let Err(err) = handle.join() {
                return Err(RuntimeError::ExtensionStop {
                    extension: "thread_registry",
                    reason: format!("failed to join thread: {err:?}"),
                });
            }
        }
        Ok(())
    }
}
