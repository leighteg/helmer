use crate::runtime::asset_server::AssetServer;
#[cfg(target_arch = "wasm32")]
use crate::runtime::asset_server::set_web_asset_server;
use helmer::runtime::{RuntimeContext, RuntimeError, RuntimeExtension};
use helmer_render::extension::{
    RenderAssetMessageSender, RenderRuntimeTuningResource, RenderStreamRequestReceiver,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone)]
pub struct AssetServerResource(pub Arc<Mutex<AssetServer>>);

pub struct AssetExtension {
    asset_base_path: Option<String>,
    server: Option<Arc<Mutex<AssetServer>>>,
}

impl Default for AssetExtension {
    fn default() -> Self {
        Self {
            asset_base_path: None,
            server: None,
        }
    }
}

impl AssetExtension {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_asset_base_path(path: impl Into<String>) -> Self {
        Self {
            asset_base_path: Some(path.into()),
            server: None,
        }
    }
}

impl RuntimeExtension for AssetExtension {
    fn name(&self) -> &'static str {
        "helmer_asset"
    }

    fn on_register(&mut self, ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        let asset_sender = ctx
            .resources()
            .get::<RenderAssetMessageSender>()
            .map(|sender| sender.0.clone())
            .ok_or_else(|| RuntimeError::ExtensionStart {
                extension: self.name(),
                reason: "missing render asset channel; register RenderExtension first".to_string(),
            })?;
        let stream_receiver = ctx
            .resources()
            .get::<RenderStreamRequestReceiver>()
            .map(|receiver| receiver.0.clone())
            .ok_or_else(|| RuntimeError::ExtensionStart {
                extension: self.name(),
                reason: "missing render stream request channel".to_string(),
            })?;
        let tuning = ctx
            .resources()
            .get::<RenderRuntimeTuningResource>()
            .map(|resource| Arc::clone(&resource.0))
            .ok_or_else(|| RuntimeError::ExtensionStart {
                extension: self.name(),
                reason: "missing runtime tuning resource".to_string(),
            })?;

        let worker_capacity = tuning
            .asset_worker_queue_capacity
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1);
        let server = Arc::new(Mutex::new(AssetServer::new(
            asset_sender,
            stream_receiver,
            worker_capacity,
            tuning,
        )));
        if let Some(path) = self.asset_base_path.as_ref() {
            server.lock().set_asset_base_path(path.clone());
        }
        #[cfg(target_arch = "wasm32")]
        set_web_asset_server(Arc::clone(&server));
        ctx.resources()
            .insert(AssetServerResource(Arc::clone(&server)));
        self.server = Some(server);
        Ok(())
    }

    fn on_tick(&mut self, _ctx: &RuntimeContext, _dt: Duration) -> Result<(), RuntimeError> {
        if let Some(server) = self.server.as_ref() {
            server.lock().update();
        }
        Ok(())
    }

    fn on_stop(&mut self, _ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        Ok(())
    }
}
