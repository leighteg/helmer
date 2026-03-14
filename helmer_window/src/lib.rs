pub mod config;
pub mod event;
pub mod extension;
pub mod service;
pub mod state;
#[cfg(target_arch = "wasm32")]
pub mod wasm_harness;

// temporary compatibility export while downstream crates migrate away from `helmer_window::runtime::*` paths
pub mod runtime {
    pub mod config {
        pub use crate::config::RuntimeConfig;
    }
    pub mod input_manager {
        pub use helmer_input::input_manager::*;
    }
    pub mod runtime {
        pub use crate::state::{
            RuntimeCursorGrabMode, RuntimeCursorState, RuntimeCursorStateSnapshot,
            RuntimeWindowControl, RuntimeWindowTitleMode,
        };
        pub use helmer::runtime::runtime::{
            LogicClock, LogicFrame, PerformanceMetrics, Runtime, RuntimeBuilder, RuntimeContext,
            RuntimeError, RuntimeExtension, RuntimeHandle, RuntimeLogEntry, RuntimeLogLayer,
            RuntimeLogLevel, RuntimeLogListener, RuntimeResources, TaskPool, ThreadHandle,
            ThreadRegistry, init_runtime_tracing, set_runtime_log_listener,
        };
    }
    #[cfg(target_arch = "wasm32")]
    pub mod wasm_harness {
        pub use crate::wasm_harness::WasmHarnessConfig;
    }

    pub use crate::config::RuntimeConfig;
    pub use crate::event::{WindowRuntimeEvent, WindowRuntimeEventKind, WindowState};
    pub use crate::extension::{
        WindowControlResource, WindowCursorStateResource, WindowExtension,
        WindowMainThreadServiceResource,
    };
    pub use crate::service::{WindowCallbacks, WindowError, WindowService};
    pub use crate::state::{
        RuntimeCursorGrabMode, RuntimeCursorState, RuntimeCursorStateSnapshot,
        RuntimeWindowControl, RuntimeWindowTitleMode,
    };
    #[cfg(target_arch = "wasm32")]
    pub use crate::wasm_harness::WasmHarnessConfig;
    pub use helmer::runtime::{
        LogicClock, LogicFrame, PerformanceMetrics, Runtime, RuntimeBuilder, RuntimeContext,
        RuntimeError, RuntimeExtension, RuntimeHandle, RuntimeLogEntry, RuntimeLogLayer,
        RuntimeLogLevel, RuntimeLogListener, RuntimeResources, TaskPool, ThreadHandle,
        ThreadRegistry, init_runtime_tracing, set_runtime_log_listener,
    };
}
