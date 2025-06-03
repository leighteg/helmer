use super::error::RendererError;
use std::sync::Arc;

// Wrapper around mev's core device and queue
// Arc is used because RenderDevice might be shared or cloned
// for creating resources or pipelines in different parts of the renderer.
#[derive(Clone)]
pub struct RenderDevice {
    #[allow(dead_code)] // instance might be needed later for certain operations
    instance: Arc<mev::Instance>,
    #[allow(dead_code)] // device might be needed for more direct operations
    device: Arc<mev::Device>,
    queue: Arc<mev::Queue>,
}

impl RenderDevice {
    pub fn new() -> Result<Self, RendererError> {
        let instance = mev::Instance::load().map_err(|e| {
            RendererError::DeviceInitialization(format!("Failed to load mev instance: {}", e))
        })?;

        // For simplicity, use the first available device and queue
        // A real application might allow selection or query capabilities
        let (device, mut queues) = instance
            .new_device(mev::DeviceDesc {
                idx: 0, // First adapter
                queues: &[0], // Request the first queue family, one queue
                features: mev::Features::SURFACE, // Basic feature for rendering to a window
            })
            .map_err(|e| {
                RendererError::DeviceInitialization(format!("Failed to create mev device: {}", e))
            })?;

        let queue = queues.pop().ok_or_else(|| {
            RendererError::DeviceInitialization("No queues returned from device".to_string())
        })?;

        tracing::info!("RenderDevice initialized successfully.");
        Ok(RenderDevice {
            instance: Arc::new(instance),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    pub fn queue(&self) -> &Arc<mev::Queue> {
        &self.queue
    }

    // pub fn device(&self) -> &Arc<mev::Device> { &self.device }
    // pub fn instance(&self) -> &Arc<mev::Instance> { &self.instance }
}