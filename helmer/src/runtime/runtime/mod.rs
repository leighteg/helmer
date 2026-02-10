mod common;
mod log_bridge;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

pub use common::{PerformanceMetrics, RuntimeProfiling, RuntimeTuning};
pub(crate) use log_bridge::RuntimeLogLayer;
pub use log_bridge::{
    RuntimeLogEntry, RuntimeLogLevel, RuntimeLogListener, set_runtime_log_listener,
};
#[cfg(not(target_arch = "wasm32"))]
pub use native::Runtime;
#[cfg(target_arch = "wasm32")]
pub use web::Runtime;
