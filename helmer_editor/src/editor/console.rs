use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, OnceLock},
};

use bevy_ecs::prelude::{Resource, World};
use helmer::runtime::runtime::{
    RuntimeLogEntry, RuntimeLogLevel, RuntimeLogListener, set_runtime_log_listener,
};

use crate::editor::{
    scripting::{ScriptRegistry, ScriptRuntime},
    watch::FileWatchState,
};

const MAX_PENDING_RUNTIME_LOGS: usize = 4096;
const DEFAULT_MAX_CONSOLE_ENTRIES: usize = 5000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorConsoleLevel {
    Trace,
    Debug,
    Log,
    Info,
    Warn,
    Error,
}

impl EditorConsoleLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Trace => "Trace",
            Self::Debug => "Debug",
            Self::Log => "Log",
            Self::Info => "Info",
            Self::Warn => "Warn",
            Self::Error => "Error",
        }
    }

    pub fn from_runtime_level(level: RuntimeLogLevel) -> Self {
        match level {
            RuntimeLogLevel::Trace => Self::Trace,
            RuntimeLogLevel::Debug => Self::Debug,
            RuntimeLogLevel::Info => Self::Info,
            RuntimeLogLevel::Warn => Self::Warn,
            RuntimeLogLevel::Error => Self::Error,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EditorConsoleEntry {
    pub sequence: u64,
    pub level: EditorConsoleLevel,
    pub target: String,
    pub message: String,
}

#[derive(Debug, Clone, Resource)]
pub struct EditorConsoleState {
    pub entries: VecDeque<EditorConsoleEntry>,
    pub max_entries: usize,
    pub show_trace: bool,
    pub show_debug: bool,
    pub show_log: bool,
    pub show_info: bool,
    pub show_warn: bool,
    pub show_error: bool,
    pub search: String,
    pub auto_scroll: bool,
    next_sequence: u64,
    last_rust_status: Option<String>,
    last_registry_status: Option<String>,
    last_watcher_status: Option<String>,
}

impl Default for EditorConsoleState {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries: DEFAULT_MAX_CONSOLE_ENTRIES,
            show_trace: true,
            show_debug: true,
            show_log: true,
            show_info: true,
            show_warn: true,
            show_error: true,
            search: String::new(),
            auto_scroll: true,
            next_sequence: 1,
            last_rust_status: None,
            last_registry_status: None,
            last_watcher_status: None,
        }
    }
}

impl EditorConsoleState {
    pub fn clear(&mut self) {
        self.entries.clear();
        self.last_rust_status = None;
        self.last_registry_status = None;
        self.last_watcher_status = None;
    }

    pub fn level_enabled(&self, level: EditorConsoleLevel) -> bool {
        match level {
            EditorConsoleLevel::Trace => self.show_trace,
            EditorConsoleLevel::Debug => self.show_debug,
            EditorConsoleLevel::Log => self.show_log,
            EditorConsoleLevel::Info => self.show_info,
            EditorConsoleLevel::Warn => self.show_warn,
            EditorConsoleLevel::Error => self.show_error,
        }
    }

    fn push_internal(
        &mut self,
        level: EditorConsoleLevel,
        target: impl Into<String>,
        message: impl Into<String>,
    ) {
        let message = message.into().trim().to_string();
        if message.is_empty() {
            return;
        }
        let target = target.into().trim().to_string();
        self.entries.push_back(EditorConsoleEntry {
            sequence: self.next_sequence,
            level,
            target,
            message,
        });
        self.next_sequence = self.next_sequence.saturating_add(1);

        let max_entries = self.max_entries.max(1);
        while self.entries.len() > max_entries {
            self.entries.pop_front();
        }
    }
}

static RUNTIME_LOG_QUEUE: OnceLock<Arc<Mutex<VecDeque<RuntimeLogEntry>>>> = OnceLock::new();

fn runtime_log_queue() -> Arc<Mutex<VecDeque<RuntimeLogEntry>>> {
    RUNTIME_LOG_QUEUE
        .get_or_init(|| Arc::new(Mutex::new(VecDeque::new())))
        .clone()
}

pub fn install_runtime_log_listener() {
    let queue = runtime_log_queue();
    let listener: Arc<RuntimeLogListener> = Arc::new(move |entry: RuntimeLogEntry| {
        if let Ok(mut pending) = queue.lock() {
            if pending.len() >= MAX_PENDING_RUNTIME_LOGS {
                pending.pop_front();
            }
            pending.push_back(entry);
        }
    });
    let _ = set_runtime_log_listener(listener);
}

pub fn drain_runtime_log_queue(world: &mut World) {
    let drained = {
        let queue = runtime_log_queue();
        let Ok(mut pending) = queue.lock() else {
            return;
        };
        pending.drain(..).collect::<Vec<_>>()
    };

    if drained.is_empty() {
        return;
    }

    if let Some(mut state) = world.get_resource_mut::<EditorConsoleState>() {
        for entry in drained {
            state.push_internal(
                EditorConsoleLevel::from_runtime_level(entry.level),
                entry.target,
                entry.message,
            );
        }
    }
}

pub fn push_console_entry(
    world: &mut World,
    level: EditorConsoleLevel,
    target: impl Into<String>,
    message: impl Into<String>,
) {
    if let Some(mut state) = world.get_resource_mut::<EditorConsoleState>() {
        state.push_internal(level, target, message);
    }
}

pub fn push_console_status(world: &mut World, message: impl Into<String>) {
    let message = message.into();
    let level = status_level(&message);
    push_console_entry(world, level, "editor.status", message);
}

pub fn sync_console_diagnostics(world: &mut World) {
    let (rust_status, registry_status, watcher_status) = {
        let rust_status = world
            .get_resource::<ScriptRuntime>()
            .and_then(|runtime| runtime.rust_status.clone());
        let registry_status = world
            .get_resource::<ScriptRegistry>()
            .and_then(|registry| registry.status.clone());
        let watcher_status = world
            .get_resource::<FileWatchState>()
            .and_then(|watch| watch.status.clone());
        (rust_status, registry_status, watcher_status)
    };

    let Some(mut state) = world.get_resource_mut::<EditorConsoleState>() else {
        return;
    };

    if state.last_rust_status != rust_status {
        if let Some(status) = rust_status.as_ref() {
            state.push_internal(status_level(status), "script.rust", status.clone());
        }
        state.last_rust_status = rust_status;
    }

    if state.last_registry_status != registry_status {
        if let Some(status) = registry_status.as_ref() {
            state.push_internal(status_level(status), "script.registry", status.clone());
        }
        state.last_registry_status = registry_status;
    }

    if state.last_watcher_status != watcher_status {
        if let Some(status) = watcher_status.as_ref() {
            state.push_internal(status_level(status), "filewatch", status.clone());
        }
        state.last_watcher_status = watcher_status;
    }
}

fn status_level(message: &str) -> EditorConsoleLevel {
    let message = message.to_ascii_lowercase();
    if message.contains("error")
        || message.contains("failed")
        || message.contains("unable")
        || message.contains("missing")
    {
        EditorConsoleLevel::Error
    } else if message.contains("warn") {
        EditorConsoleLevel::Warn
    } else if message.contains("debug") {
        EditorConsoleLevel::Debug
    } else {
        EditorConsoleLevel::Info
    }
}
