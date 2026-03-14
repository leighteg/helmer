use crate::audio::AudioBackend;
use helmer::runtime::{RuntimeContext, RuntimeError, RuntimeExtension};
use std::sync::Arc;

#[derive(Clone)]
pub struct RuntimeAudioBackendResource(pub Arc<AudioBackend>);

pub struct AudioExtension {
    backend: Arc<AudioBackend>,
}

impl Default for AudioExtension {
    fn default() -> Self {
        Self {
            backend: Arc::new(AudioBackend::new()),
        }
    }
}

impl AudioExtension {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_backend(backend: Arc<AudioBackend>) -> Self {
        Self { backend }
    }
}

impl RuntimeExtension for AudioExtension {
    fn name(&self) -> &'static str {
        "helmer_audio"
    }

    fn on_register(&mut self, ctx: &RuntimeContext) -> Result<(), RuntimeError> {
        ctx.resources()
            .insert(RuntimeAudioBackendResource(Arc::clone(&self.backend)));
        Ok(())
    }
}
