#![cfg(target_arch = "wasm32")]

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use futures_channel::mpsc as futures_mpsc;
use futures_util::{SinkExt, StreamExt};
use gloo_net::websocket::{Message as WasmMessage, futures::WebSocket};
use wasm_bindgen_futures::spawn_local;

use crate::{
    NetworkClientConfig, NetworkConnectionId, NetworkEvent, NetworkHubError, NetworkLane,
    hub::push_event,
    wire::{PacketWire, ReliableWireFrame, decode_reliable_frame, encode_reliable_frame},
};

#[derive(Default)]
pub(crate) struct WasmClientState {
    pub(crate) connection_id: Option<NetworkConnectionId>,
    pub(crate) outgoing: Option<futures_mpsc::UnboundedSender<WasmMessage>>,
}

pub(crate) fn spawn_wasm_client(
    config: NetworkClientConfig,
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
    state: Arc<Mutex<WasmClientState>>,
) {
    spawn_local(async move {
        let ws = match WebSocket::open(&config.reliable_url) {
            Ok(ws) => ws,
            Err(err) => {
                push_event(
                    &events,
                    NetworkEvent::Error {
                        context: "wasm_connect".to_owned(),
                        message: err.to_string(),
                    },
                );
                return;
            }
        };

        let (mut sink, mut stream) = ws.split();
        let (outgoing_tx, mut outgoing_rx) = futures_mpsc::unbounded::<WasmMessage>();

        {
            let mut state_guard = state.lock().expect("network hub poisoned");
            state_guard.outgoing = Some(outgoing_tx);
        }

        spawn_local(async move {
            while let Some(message) = outgoing_rx.next().await {
                let _ = sink.send(message).await;
            }
        });

        while let Some(message) = stream.next().await {
            let bytes = match message {
                Ok(WasmMessage::Bytes(bytes)) => bytes,
                Ok(WasmMessage::Text(_)) => continue,
                Err(err) => {
                    push_event(
                        &events,
                        NetworkEvent::Error {
                            context: "wasm_read".to_owned(),
                            message: err.to_string(),
                        },
                    );
                    break;
                }
            };

            let frame: ReliableWireFrame = match decode_reliable_frame(&bytes) {
                Ok(frame) => frame,
                Err(err) => {
                    push_event(
                        &events,
                        NetworkEvent::Error {
                            context: "wasm_decode".to_owned(),
                            message: err.to_string(),
                        },
                    );
                    continue;
                }
            };

            match frame {
                ReliableWireFrame::Hello(hello) => {
                    let mut state_guard = state.lock().expect("network hub poisoned");
                    state_guard.connection_id = Some(hello.connection_id);
                    push_event(
                        &events,
                        NetworkEvent::Connected {
                            connection_id: hello.connection_id,
                            unreliable_supported: false,
                        },
                    );
                }
                ReliableWireFrame::Payload(packet) => {
                    let connection_id = state
                        .lock()
                        .expect("network hub poisoned")
                        .connection_id
                        .unwrap_or_default();
                    if connection_id != NetworkConnectionId::default() {
                        push_event(
                            &events,
                            NetworkEvent::Packet {
                                connection_id,
                                lane: packet.lane,
                                payload: packet.payload,
                            },
                        );
                    }
                }
                ReliableWireFrame::Ping { sent_at_millis } => {
                    let outgoing = state.lock().expect("network hub poisoned").outgoing.clone();
                    if let Some(outgoing) = outgoing
                        && let Ok(bytes) =
                            encode_reliable_frame(&ReliableWireFrame::Pong { sent_at_millis })
                    {
                        let _ = outgoing.unbounded_send(WasmMessage::Bytes(bytes));
                    }
                }
                ReliableWireFrame::Pong { .. } => {}
                ReliableWireFrame::Disconnect { reason } => {
                    let connection_id = {
                        let mut state_guard = state.lock().expect("network hub poisoned");
                        let id = state_guard.connection_id;
                        state_guard.connection_id = None;
                        state_guard.outgoing = None;
                        id
                    };
                    if let Some(connection_id) = connection_id {
                        push_event(
                            &events,
                            NetworkEvent::Disconnected {
                                connection_id,
                                reason,
                            },
                        );
                    }
                    break;
                }
            }
        }
    });
}

pub(crate) fn wasm_send(
    state: &Arc<Mutex<WasmClientState>>,
    connection_id: NetworkConnectionId,
    lane: NetworkLane,
    payload: Vec<u8>,
) -> Result<(), NetworkHubError> {
    let mut state = state.lock().expect("network hub poisoned");
    if state.connection_id != Some(connection_id) {
        return Err(NetworkHubError::Command(format!(
            "unknown wasm connection {}",
            connection_id.0
        )));
    }
    let outgoing = state
        .outgoing
        .as_mut()
        .ok_or(NetworkHubError::BackendUnavailable)?;
    let frame = ReliableWireFrame::Payload(PacketWire { lane, payload });
    let bytes =
        encode_reliable_frame(&frame).map_err(|err| NetworkHubError::Command(err.to_string()))?;
    outgoing
        .unbounded_send(WasmMessage::Bytes(bytes))
        .map_err(|err| NetworkHubError::Command(err.to_string()))
}

pub(crate) fn wasm_disconnect(
    state: &Arc<Mutex<WasmClientState>>,
    connection_id: NetworkConnectionId,
) {
    let mut state = state.lock().expect("network hub poisoned");
    if state.connection_id == Some(connection_id) {
        state.outgoing = None;
        state.connection_id = None;
    }
}

pub(crate) fn wasm_shutdown(state: &Arc<Mutex<WasmClientState>>) {
    let mut state = state.lock().expect("network hub poisoned");
    state.outgoing = None;
    state.connection_id = None;
}
