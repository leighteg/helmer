use thiserror::Error;

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct NetworkConnectionId(pub u64);

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub enum NetworkLane {
    ReliableOrdered,
    UnreliableSequenced,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NetworkTransportStats {
    pub queued_reliable_frames: usize,
    pub queued_reliable_bytes: u64,
    pub smoothed_rtt_millis: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkServerConfig {
    pub reliable_bind: String,
    pub unreliable_bind: Option<String>,
    pub public_unreliable_addr: Option<String>,
}

impl Default for NetworkServerConfig {
    fn default() -> Self {
        Self {
            reliable_bind: "127.0.0.1:25472".to_owned(),
            unreliable_bind: Some("127.0.0.1:25473".to_owned()),
            public_unreliable_addr: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkClientConfig {
    pub reliable_url: String,
    pub unreliable_bind: Option<String>,
}

impl Default for NetworkClientConfig {
    fn default() -> Self {
        Self {
            reliable_url: "ws://127.0.0.1:25472".to_owned(),
            unreliable_bind: Some("0.0.0.0:0".to_owned()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetworkEvent {
    Listening {
        reliable_addr: String,
        unreliable_addr: Option<String>,
    },
    Connected {
        connection_id: NetworkConnectionId,
        unreliable_supported: bool,
    },
    Disconnected {
        connection_id: NetworkConnectionId,
        reason: String,
    },
    Packet {
        connection_id: NetworkConnectionId,
        lane: NetworkLane,
        payload: Vec<u8>,
    },
    Error {
        context: String,
        message: String,
    },
}

#[derive(Debug, Error)]
pub enum NetworkHubError {
    #[error("network backend is unavailable")]
    BackendUnavailable,
    #[error("network backend does not support this operation on this target")]
    Unsupported,
    #[error("network command failed: {0}")]
    Command(String),
}
