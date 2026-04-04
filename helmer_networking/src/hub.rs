use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;

use crate::{
    NetworkClientConfig, NetworkConnectionId, NetworkEvent, NetworkHubError, NetworkLane,
    NetworkServerConfig, NetworkTransportStats,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::native::{BackendCommand, NativeTelemetryMap, spawn_native_backend};
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmClientState, spawn_wasm_client, wasm_disconnect, wasm_send, wasm_shutdown};

pub struct NetworkHub {
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
    #[cfg(not(target_arch = "wasm32"))]
    commands: mpsc::UnboundedSender<BackendCommand>,
    #[cfg(not(target_arch = "wasm32"))]
    telemetry: NativeTelemetryMap,
    #[cfg(target_arch = "wasm32")]
    wasm_state: Arc<Mutex<WasmClientState>>,
}

impl NetworkHub {
    pub fn new() -> Self {
        let events = Arc::new(Mutex::new(VecDeque::new()));

        #[cfg(not(target_arch = "wasm32"))]
        {
            let telemetry = NativeTelemetryMap::default();
            let commands = spawn_native_backend(Arc::clone(&events), telemetry.clone());
            Self {
                events,
                commands,
                telemetry,
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                events,
                wasm_state: Arc::new(Mutex::new(WasmClientState::default())),
            }
        }
    }

    pub fn listen(&self, config: NetworkServerConfig) -> Result<(), NetworkHubError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.commands
                .send(BackendCommand::Listen(config))
                .map_err(|_| NetworkHubError::BackendUnavailable)
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = config;
            Err(NetworkHubError::Unsupported)
        }
    }

    pub fn connect(&self, config: NetworkClientConfig) -> Result<(), NetworkHubError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.commands
                .send(BackendCommand::Connect(config))
                .map_err(|_| NetworkHubError::BackendUnavailable)
        }

        #[cfg(target_arch = "wasm32")]
        {
            spawn_wasm_client(
                config,
                Arc::clone(&self.events),
                Arc::clone(&self.wasm_state),
            );
            Ok(())
        }
    }

    pub fn send(
        &self,
        connection_id: NetworkConnectionId,
        lane: NetworkLane,
        payload: Vec<u8>,
    ) -> Result<(), NetworkHubError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.commands
                .send(BackendCommand::Send {
                    connection_id,
                    lane,
                    payload,
                })
                .map_err(|_| NetworkHubError::BackendUnavailable)
        }

        #[cfg(target_arch = "wasm32")]
        {
            wasm_send(&self.wasm_state, connection_id, lane, payload)
        }
    }

    pub fn disconnect(&self, connection_id: NetworkConnectionId) -> Result<(), NetworkHubError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.commands
                .send(BackendCommand::Disconnect { connection_id })
                .map_err(|_| NetworkHubError::BackendUnavailable)
        }

        #[cfg(target_arch = "wasm32")]
        {
            wasm_disconnect(&self.wasm_state, connection_id);
            Ok(())
        }
    }

    pub fn shutdown(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.commands.send(BackendCommand::Shutdown);
        }

        #[cfg(target_arch = "wasm32")]
        {
            wasm_shutdown(&self.wasm_state);
        }
    }

    pub fn drain_events(&self) -> Vec<NetworkEvent> {
        let mut events = self.events.lock().expect("network hub poisoned");
        events.drain(..).collect()
    }

    pub fn connection_stats(
        &self,
        connection_id: NetworkConnectionId,
    ) -> Option<NetworkTransportStats> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.telemetry.snapshot(connection_id)
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = connection_id;
            None
        }
    }
}

impl Default for NetworkHub {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn push_event(events: &Arc<Mutex<VecDeque<NetworkEvent>>>, event: NetworkEvent) {
    events
        .lock()
        .expect("network hub poisoned")
        .push_back(event);
}
