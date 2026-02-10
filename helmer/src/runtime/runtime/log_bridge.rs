use std::sync::{Arc, OnceLock};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

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

pub fn set_runtime_log_listener(
    listener: Arc<RuntimeLogListener>,
) -> Result<(), Arc<RuntimeLogListener>> {
    RUNTIME_LOG_LISTENER.set(listener)
}

fn emit_runtime_log(entry: RuntimeLogEntry) {
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

pub(crate) struct RuntimeLogLayer;

impl<S> Layer<S> for RuntimeLogLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        emit_runtime_log(build_runtime_log_entry(event));
    }
}
