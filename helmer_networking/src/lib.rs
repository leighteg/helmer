#![forbid(unsafe_code)]

mod api;
mod extension;
mod hub;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod wasm;
mod wire;

pub use api::{
    NetworkClientConfig, NetworkConnectionId, NetworkEvent, NetworkHubError, NetworkLane,
    NetworkServerConfig, NetworkTransportStats,
};
pub use extension::{NetworkingExtension, NetworkingResource};
pub use hub::NetworkHub;
