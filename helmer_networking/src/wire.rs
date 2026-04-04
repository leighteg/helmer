use rkyv::rancor::Error as WireCodecError;

use crate::{NetworkConnectionId, NetworkLane};

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub(crate) enum ReliableWireFrame {
    Hello(TransportHello),
    Payload(PacketWire),
    Ping { sent_at_millis: u64 },
    Pong { sent_at_millis: u64 },
    Disconnect { reason: String },
}

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub(crate) struct TransportHello {
    pub(crate) connection_id: NetworkConnectionId,
    pub(crate) unreliable_token: Option<u64>,
    pub(crate) unreliable_addr: Option<String>,
}

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub(crate) struct PacketWire {
    pub(crate) lane: NetworkLane,
    pub(crate) payload: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub(crate) struct DatagramWire {
    pub(crate) connection_id: NetworkConnectionId,
    pub(crate) token: u64,
    pub(crate) sequence: u64,
    pub(crate) payload: Vec<u8>,
}

pub(crate) fn encode_reliable_frame(frame: &ReliableWireFrame) -> Result<Vec<u8>, WireCodecError> {
    rkyv::to_bytes::<WireCodecError>(frame).map(|bytes| bytes.into_vec())
}

pub(crate) fn decode_reliable_frame(bytes: &[u8]) -> Result<ReliableWireFrame, WireCodecError> {
    rkyv::from_bytes::<ReliableWireFrame, WireCodecError>(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn encode_datagram(frame: &DatagramWire) -> Result<Vec<u8>, WireCodecError> {
    rkyv::to_bytes::<WireCodecError>(frame).map(|bytes| bytes.into_vec())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn decode_datagram(bytes: &[u8]) -> Result<DatagramWire, WireCodecError> {
    rkyv::from_bytes::<DatagramWire, WireCodecError>(bytes)
}
