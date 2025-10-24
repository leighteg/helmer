use thiserror::Error;

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("Device initialization failed: {0}")]
    DeviceInitialization(String),

    #[error("Resource creation failed: {0}")]
    ResourceCreation(String),

    #[error("Render graph error: {0}")]
    RenderGraphError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    #[error("Pipeline creation error: {0}")]
    PipelineCreation(String),

    #[error("Resource not found in graph: {0}")]
    ResourceNotFound(String),

    #[error("Internal renderer error: {0}")]
    Internal(String),
}