mod common;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

pub use common::{PerformanceMetrics, RuntimeProfiling, RuntimeTuning};
#[cfg(not(target_arch = "wasm32"))]
pub use native::Runtime;
#[cfg(target_arch = "wasm32")]
pub use web::Runtime;
