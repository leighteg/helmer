pub mod clock;
mod engine;
pub mod log_bridge;
mod windows_timer;

pub use clock::{LogicClock, LogicFrame, PerformanceMetrics, RuntimePerformanceMetricsResource};
pub use engine::{
    Runtime, RuntimeBuilder, RuntimeContext, RuntimeError, RuntimeExtension, RuntimeHandle,
    RuntimeResources, TaskPool, ThreadHandle, ThreadRegistry,
};
pub use log_bridge::{
    RuntimeLogEntry, RuntimeLogLayer, RuntimeLogLevel, RuntimeLogListener, init_runtime_tracing,
    set_runtime_log_listener,
};
pub use windows_timer::HighResolutionTimerGuard;

// transitional compatibility for existing call sites using `helmer::runtime::runtime::*`
pub mod runtime {
    pub use super::clock::{
        LogicClock, LogicFrame, PerformanceMetrics, RuntimePerformanceMetricsResource,
    };
    pub use super::engine::{
        Runtime, RuntimeBuilder, RuntimeContext, RuntimeError, RuntimeExtension, RuntimeHandle,
        RuntimeResources, TaskPool, ThreadHandle, ThreadRegistry,
    };
    pub use super::log_bridge::{
        RuntimeLogEntry, RuntimeLogLayer, RuntimeLogLevel, RuntimeLogListener,
        init_runtime_tracing, set_runtime_log_listener,
    };
    pub use super::windows_timer::HighResolutionTimerGuard;
}
