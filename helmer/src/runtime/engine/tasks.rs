use super::RuntimeError;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
use std::thread::JoinHandle;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use tracing::warn;

type Job = Box<dyn FnOnce() + Send + 'static>;

#[cfg(not(target_arch = "wasm32"))]
struct TaskWorker {
    running: Arc<AtomicBool>,
    join: JoinHandle<()>,
}

pub struct TaskPool {
    #[cfg(not(target_arch = "wasm32"))]
    sender: Sender<Job>,
    #[cfg(not(target_arch = "wasm32"))]
    receiver: Arc<Mutex<Receiver<Job>>>,
    #[cfg(not(target_arch = "wasm32"))]
    workers: Mutex<Vec<TaskWorker>>,
    #[cfg(not(target_arch = "wasm32"))]
    next_worker_id: AtomicUsize,
    inline: AtomicBool,
}

impl TaskPool {
    #[cfg(target_arch = "wasm32")]
    fn inline() -> Self {
        Self {
            inline: AtomicBool::new(true),
        }
    }

    pub fn new(worker_count: usize) -> Result<Self, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            if worker_count > 0 {
                warn!(
                    requested_workers = worker_count,
                    "task pool threads are unavailable on wasm; using inline task execution"
                );
            }
            return Ok(Self::inline());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver): (Sender<Job>, Receiver<Job>) = mpsc::channel();
            let pool = Self {
                sender,
                receiver: Arc::new(Mutex::new(receiver)),
                workers: Mutex::new(Vec::new()),
                next_worker_id: AtomicUsize::new(0),
                inline: AtomicBool::new(true),
            };

            if let Err(err) = pool.set_worker_count(worker_count) {
                warn!(
                    requested_workers = worker_count,
                    reason = %err,
                    "failed to spawn task worker threads; using inline task execution"
                );
            }
            Ok(pool)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn spawn_worker(&self, workers: &mut Vec<TaskWorker>) -> Result<(), RuntimeError> {
        let worker_id = self.next_worker_id.fetch_add(1, Ordering::Relaxed);
        let receiver = Arc::clone(&self.receiver);
        let running = Arc::new(AtomicBool::new(true));
        let worker_running = Arc::clone(&running);
        let builder = thread::Builder::new().name(format!("helmer-task-{worker_id}"));
        let join = builder.spawn(move || {
            while worker_running.load(Ordering::Acquire) {
                let message = {
                    let Ok(guard) = receiver.lock() else {
                        break;
                    };
                    guard.recv_timeout(Duration::from_millis(25))
                };
                match message {
                    Ok(job) => job(),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });
        let join = join.map_err(|err| RuntimeError::ExtensionStart {
            extension: "task_pool",
            reason: format!("failed to spawn task worker thread: {err}"),
        })?;
        workers.push(TaskWorker { running, join });
        Ok(())
    }

    pub fn worker_count(&self) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            0
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.workers
                .lock()
                .map(|workers| workers.len())
                .unwrap_or(0)
        }
    }

    pub fn set_worker_count(&self, worker_count: usize) -> Result<(), RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            if worker_count > 0 {
                warn!(
                    requested_workers = worker_count,
                    "task pool threads are unavailable on wasm; keeping inline execution"
                );
            }
            self.inline.store(true, Ordering::Release);
            return Ok(());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let target = worker_count;
            self.inline.store(target == 0, Ordering::Release);

            let mut removed = Vec::new();
            {
                let mut workers = self
                    .workers
                    .lock()
                    .map_err(|_| RuntimeError::ExtensionStop {
                        extension: "task_pool",
                        reason: "task worker lock poisoned".to_string(),
                    })?;
                let current = workers.len();
                if target > current {
                    for _ in current..target {
                        self.spawn_worker(&mut workers)?;
                    }
                } else if target < current {
                    removed.extend(workers.drain(target..));
                }
            }

            if !removed.is_empty() {
                for worker in &removed {
                    worker.running.store(false, Ordering::Release);
                }
                let _ = thread::Builder::new()
                    .name("helmer-task-reaper".to_string())
                    .spawn(move || {
                        for worker in removed {
                            if let Err(err) = worker.join.join() {
                                warn!(
                                    reason = ?err,
                                    "failed to join retired task worker thread"
                                );
                            }
                        }
                    });
            }
            Ok(())
        }
    }

    pub fn spawn<F>(&self, job: F) -> Result<(), RuntimeError>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.inline.load(Ordering::Acquire) {
            job();
            return Ok(());
        }

        #[cfg(target_arch = "wasm32")]
        {
            return Err(RuntimeError::TaskPoolUnavailable);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.sender
                .send(Box::new(job))
                .map_err(|_| RuntimeError::TaskPoolUnavailable)
        }
    }

    pub fn shutdown(&self) -> Result<(), RuntimeError> {
        self.set_worker_count(0)
    }
}
