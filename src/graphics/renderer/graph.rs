use super::device::RenderDevice;
use super::error::RendererError;
use super::resource::{Buffer, BufferDesc, ResourceId, Texture, TextureDesc, ExternalResourceMap};
use std::sync::Arc;
use std::collections::HashMap;

// Parameters passed to a render pass execution closure
pub struct RenderPassExecutionParams<'a> {
    pub pass_name: &'a str,
    pub queue: &'a Arc<mev::Queue>,
    pub encoder: &'a mut mev::CommandEncoder,
    pub inputs: &'a [ResourceId],
    pub outputs: &'a [ResourceId],
    pub graph_resources: &'a GraphResources, // To resolve ResourceIds to actual resource descriptions
    pub external_resource_map: &'a ExternalResourceMap, // To get actual mev::Image/Buffer for external resources
    // pub dynamic_resource_cache: &'c mut DynamicResourceCache, // For graph-managed transient resources
}

// Type alias for the function/closure that executes a render pass
type RenderPassFn = Box<
    dyn for<'a> FnMut(RenderPassExecutionParams<'a>) -> Result<(), RendererError>,
>;

// Node in the render graph representing a single pass
struct RenderPassNode {
    name: String,
    inputs: Vec<ResourceId>,  // Resources this pass reads from
    outputs: Vec<ResourceId>, // Resources this pass writes to
    execute_fn: RenderPassFn,
}

// Stores descriptions of resources known to the graph.
// Actual GPU resources (mev::Image, mev::Buffer) are managed separately
// by a resource manager or passed in externally (like the backbuffer).
#[derive(Default)]
pub struct GraphResources {
    resources: Vec<GraphResourceVariant>, // Indexed by ResourceId
    name_to_id: HashMap<String, ResourceId>,
}

impl GraphResources {
    fn add(&mut self, name: String, resource: GraphResourceVariant) -> ResourceId {
        let id = self.resources.len();
        self.resources.push(resource);
        self.name_to_id.insert(name, id);
        id
    }

    pub fn get_texture_desc(&self, id: ResourceId) -> Option<&TextureDesc> {
        match self.resources.get(id) {
            Some(GraphResourceVariant::Texture(t)) => Some(&t.desc),
            _ => None,
        }
    }
     pub fn get_buffer_desc(&self, id: ResourceId) -> Option<&BufferDesc> {
        match self.resources.get(id) {
            Some(GraphResourceVariant::Buffer(b)) => Some(&b.desc),
            _ => None,
        }
    }
    pub fn get_texture_handle(&self, id: ResourceId) -> Option<ResourceId> {
        // For now, ResourceId is the handle to external map
        self.resources.get(id).and_then(|r| match r {
            GraphResourceVariant::ExternalTexture { original_id, .. } => Some(*original_id),
            _ => None, // Or handle internal textures differently
        })
    }
    // Add similar get_buffer_handle if needed
}

#[derive(Clone)] // May need Debug
enum GraphResourceVariant {
    Texture(Texture), // For textures the graph might manage internally (future)
    Buffer(Buffer),   // For buffers the graph might manage internally (future)
    ExternalTexture { // For resources provided from outside the graph (e.g. swapchain)
        name: String,
        original_id: ResourceId, // The ID used in the external map
        // We don't store the mev::Image here, just its description if needed
        // The actual mev::Image is retrieved from ExternalResourceMap during execution
    },
    // ExternalBuffer { ... }
}


pub struct RenderGraph {
    passes: Vec<RenderPassNode>,
    resources: GraphResources,
    external_resource_map: ExternalResourceMap, // Holds actual mev::Image for external resources
    next_external_id: ResourceId,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            passes: Vec::new(),
            resources: GraphResources::default(),
            external_resource_map: ExternalResourceMap::default(),
            next_external_id: 0,
        }
    }

    pub fn clear(&mut self) {
        self.passes.clear();
        self.resources = GraphResources::default(); // Clears internal resource tracking
        self.external_resource_map = ExternalResourceMap::default(); // Clears map of external mev objects
        self.next_external_id = 0;
    }
    
    fn new_external_id(&mut self) -> ResourceId {
        let id = self.next_external_id;
        self.next_external_id += 1;
        id
    }

    // For resources like the swapchain image that are managed outside the graph
    pub fn add_external_texture(&mut self, name: String, image: mev::Image) -> ResourceId {
        let external_id = self.new_external_id();
        self.external_resource_map.add_texture(external_id, Arc::new(image));
        
        let graph_resource = GraphResourceVariant::ExternalTexture {
            name: name.clone(),
            original_id: external_id,
        };
        self.resources.add(name, graph_resource)
    }

    // For resources the graph will manage (simplified for now)
    pub fn _create_texture(&mut self, desc: TextureDesc) -> ResourceId {
        let name = desc.name.clone();
        let id = self.resources.resources.len(); // Next available ID
        let texture_resource = Texture { id, desc, mev_image: None }; // Not materialized yet
        self.resources.add(name, GraphResourceVariant::Texture(texture_resource))
    }

    pub fn _create_buffer(&mut self, desc: BufferDesc) -> ResourceId {
        let name = desc.name.clone();
        let id = self.resources.resources.len();
        let buffer_resource = Buffer { id, desc, mev_buffer: None };
        self.resources.add(name, GraphResourceVariant::Buffer(buffer_resource))
    }

    pub fn add_pass(
        &mut self,
        name: String,
        inputs: Vec<ResourceId>,
        outputs: Vec<ResourceId>,
        execute_fn: impl for<'a> FnMut(RenderPassExecutionParams<'a>) -> Result<(), RendererError> + 'static,
    ) -> Result<(), RendererError> {
        // Basic validation: check if resource IDs are valid (simplified)
        for &id in inputs.iter().chain(outputs.iter()) {
            if id >= self.resources.resources.len() && id >= self.next_external_id { // Check both internal and external ranges
                 return Err(RendererError::RenderGraphError(format!(
                    "Resource ID {} out of bounds for pass '{}'", id, name
                )));
            }
        }

        self.passes.push(RenderPassNode {
            name,
            inputs,
            outputs,
            execute_fn: Box::new(execute_fn),
        });
        Ok(())
    }

    // Executes the graph.
    // In a more complex graph, this would involve:
    // 1. Culling unused passes.
    // 2. Allocating transient resources.
    // 3. Inserting barriers and resource transitions.
    // 4. Building one or more command buffers.
    // For now, it just executes passes in order.
    pub fn execute(
        &mut self,
        queue: &Arc<mev::Queue>,
        encoder: &mut mev::CommandEncoder,
        // Pass external resources if they are not stored directly in the graph
    ) -> Result<(), RendererError> {
        tracing::debug!("Executing RenderGraph with {} passes.", self.passes.len());

        // TODO: Resource lifetime management, barrier insertion, etc.
        // For now, assume passes are ordered correctly and handle their own transitions if simple.

        for pass_node in &mut self.passes {
            tracing::debug!("Executing pass: {}", pass_node.name);
            let params = RenderPassExecutionParams {
                pass_name: &pass_node.name,
                queue,
                encoder,
                inputs: &pass_node.inputs,
                outputs: &pass_node.outputs,
                graph_resources: &self.resources,
                external_resource_map: &self.external_resource_map,
                // dynamic_resource_cache: &mut dynamic_cache, // For graph-managed resources
            };
            (pass_node.execute_fn)(params).map_err(|e| {
                RendererError::RenderGraphError(format!(
                    "Error in pass '{}': {}",
                    pass_node.name, e
                ))
            })?;
        }
        tracing::debug!("RenderGraph execution finished.");
        Ok(())
    }
}