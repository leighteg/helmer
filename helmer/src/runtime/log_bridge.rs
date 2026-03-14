use std::sync::{Arc, Once, OnceLock};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::{
    EnvFilter,
    layer::{Context, Layer, SubscriberExt},
    util::SubscriberInitExt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl RuntimeLogLevel {
    fn from_tracing(level: &Level) -> Self {
        match *level {
            Level::TRACE => Self::Trace,
            Level::DEBUG => Self::Debug,
            Level::INFO => Self::Info,
            Level::WARN => Self::Warn,
            Level::ERROR => Self::Error,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeLogEntry {
    pub level: RuntimeLogLevel,
    pub target: String,
    pub message: String,
}

pub type RuntimeLogListener = dyn Fn(RuntimeLogEntry) + Send + Sync + 'static;

static RUNTIME_LOG_LISTENER: OnceLock<Arc<RuntimeLogListener>> = OnceLock::new();
static RUNTIME_TRACING_INIT: Once = Once::new();

fn runtime_env_filter() -> EnvFilter {
    let default_directive = if cfg!(debug_assertions) {
        "debug"
    } else {
        "info"
    };

    for key in ["HELMER_LOG_FILTER", "RUST_LOG", "HELMER_LOG_LEVEL"] {
        if let Ok(value) = std::env::var(key)
            && let Ok(filter) = EnvFilter::try_new(value)
        {
            return filter;
        }
    }

    EnvFilter::try_new(default_directive).unwrap_or_else(|_| EnvFilter::new("info"))
}

pub fn set_runtime_log_listener(
    listener: Arc<RuntimeLogListener>,
) -> Result<(), Arc<RuntimeLogListener>> {
    RUNTIME_LOG_LISTENER.set(listener)
}

pub fn init_runtime_tracing() {
    RUNTIME_TRACING_INIT.call_once(|| {
        #[cfg(target_arch = "wasm32")]
        let _ = tracing_subscriber::registry()
            .with(runtime_env_filter())
            .with(tracing_subscriber::fmt::layer().without_time())
            .with(RuntimeLogLayer)
            .try_init();

        #[cfg(not(target_arch = "wasm32"))]
        let _ = tracing_subscriber::registry()
            .with(runtime_env_filter())
            .with(tracing_subscriber::fmt::layer())
            .with(RuntimeLogLayer)
            .try_init();
    });
}

fn emit_runtime_log(entry: RuntimeLogEntry) {
    #[cfg(target_arch = "wasm32")]
    {
        let formatted = if entry.target.is_empty() {
            entry.message.clone()
        } else {
            format!("{}: {}", entry.target, entry.message)
        };
        let js_message = formatted.into();
        match entry.level {
            RuntimeLogLevel::Trace | RuntimeLogLevel::Debug => {
                web_sys::console::debug_1(&js_message)
            }
            RuntimeLogLevel::Info => web_sys::console::info_1(&js_message),
            RuntimeLogLevel::Warn => web_sys::console::warn_1(&js_message),
            RuntimeLogLevel::Error => web_sys::console::error_1(&js_message),
        }
    }

    if let Some(listener) = RUNTIME_LOG_LISTENER.get() {
        listener(entry);
    }
}

#[derive(Default)]
struct RuntimeLogVisitor {
    message: Option<String>,
    fields: Vec<String>,
}

impl Visit for RuntimeLogVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{value:?}"));
        } else {
            self.fields.push(format!("{}={value:?}", field.name()));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else {
            self.fields.push(format!("{}={}", field.name(), value));
        }
    }
}

fn build_runtime_log_entry(event: &Event<'_>) -> RuntimeLogEntry {
    let meta = event.metadata();
    let mut visitor = RuntimeLogVisitor::default();
    event.record(&mut visitor);

    let mut message = visitor.message.unwrap_or_default();
    if !visitor.fields.is_empty() {
        if !message.is_empty() {
            message.push(' ');
        }
        message.push_str(&visitor.fields.join(" "));
    }
    if message.is_empty() {
        message = meta.name().to_string();
    }

    RuntimeLogEntry {
        level: RuntimeLogLevel::from_tracing(meta.level()),
        target: meta.target().to_string(),
        message,
    }
}

pub struct RuntimeLogLayer;

impl<S> Layer<S> for RuntimeLogLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        emit_runtime_log(build_runtime_log_entry(event));
    }
}
