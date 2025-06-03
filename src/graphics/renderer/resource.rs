use super::error::RendererError;
use std::sync::Arc;

// Simple generational index for handles, could be replaced with slotmap or similar.
// For now, a usize ID is fine for demonstration.
pub type ResourceId = usize;
pub type TextureHandle = usize; // Could be an index into a Vec<Texture> or similar
pub type BufferHandle = usize;  // Could be an index into a Vec<Buffer> or similar

#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub name: String,
    pub format: mev::PixelFormat,
    pub extent: mev::Extent3,
    pub mip_levels: u32,
    pub sample_count: u32,
    pub usage: mev::ImageUsage,
}

#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub name: String,
    pub size: u64, // in bytes
    pub usage: mev::BufferUsage,
    pub memory_type: mev::DataType,
}

// Represents a GPU texture resource managed by the renderer/graph
#[derive(Debug, Clone)]
pub struct Texture {
    pub id: ResourceId, // The graph's ID for this resource
    pub desc: TextureDesc,
    pub mev_image: Option<Arc<mev::Image>>, // The actual mev GPU resource, Option if not yet materialized
}

// Represents a GPU buffer resource managed by the renderer/graph
#[derive(Debug, Clone)]
pub struct Buffer {
    pub id: ResourceId, // The graph's ID for this resource
    pub desc: BufferDesc,
    pub mev_buffer: Option<Arc<mev::Buffer>>, // The actual mev GPU resource, Option if not yet materialized
}

// A container for actual mev resources, mapped by their graph ResourceId
// This is simplified. A real system might use a slotmap or more complex storage.
#[derive(Default)]
pub struct ExternalResourceMap {
    textures: std::collections::HashMap<ResourceId, Arc<mev::Image>>,
    // buffers: std::collections::HashMap<ResourceId, Arc<mev::Buffer>>,
}

impl ExternalResourceMap {
    pub fn add_texture(&mut self, id: ResourceId, image: Arc<mev::Image>) {
        self.textures.insert(id, image);
    }
    pub fn get_texture(&self, id: ResourceId) -> Option<&Arc<mev::Image>> {
        self.textures.get(&id)
    }
    // pub fn add_buffer(&mut self, id: ResourceId, buffer: Arc<mev::Buffer>) {
    //     self.buffers.insert(id, buffer);
    // }
    // pub fn get_buffer(&self, id: ResourceId) -> Option<&Arc<mev::Buffer>> {
    //     self.buffers.get(&id)
    // }
}