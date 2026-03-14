use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum RuntimeError {
    AlreadyStarted,
    NotStarted,
    ExtensionStart {
        extension: &'static str,
        reason: String,
    },
    ExtensionStop {
        extension: &'static str,
        reason: String,
    },
    TaskPoolUnavailable,
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyStarted => write!(f, "runtime already started"),
            Self::NotStarted => write!(f, "runtime is not started"),
            Self::ExtensionStart { extension, reason } => {
                write!(f, "extension `{extension}` failed to start: {reason}")
            }
            Self::ExtensionStop { extension, reason } => {
                write!(f, "extension `{extension}` failed to stop: {reason}")
            }
            Self::TaskPoolUnavailable => write!(f, "task pool is not available"),
        }
    }
}

impl std::error::Error for RuntimeError {}
