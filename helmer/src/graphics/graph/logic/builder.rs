use crate::graphics::graph::{
    definition::pass::PassNodeDefinition,
    logic::{
        graph::CompiledRenderGraph, pass::CompiledPassNode, resource_pool::pool::ResourcePool,
    },
};

pub struct RenderGraphBuilder {
    definitions: Vec<PassNodeDefinition>,
}

impl RenderGraphBuilder {
    pub fn new(defs: Vec<PassNodeDefinition>) -> Self {
        Self { definitions: defs }
    }

    pub fn compile(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &mut ResourcePool,
    ) -> CompiledRenderGraph {
        let mut compiled = Vec::<Box<dyn CompiledPassNode>>::new();

        for def in self.definitions {
            match def {
                PassNodeDefinition::Forward {
                    mesh_handles,
                    material_handles,
                    output_target,
                } => {
                    /*compiled.push(Box::new(ForwardPass::new(
                        mesh_handles,
                        material_handles,
                        output_target,
                    )));*/
                }

                PassNodeDefinition::Shadows { mesh_handles } => {
                    /*compiled.push(Box::new(ShadowPass::new(mesh_handles)));*/
                }
            }
        }

        // Initialize each pass
        for pass in compiled.iter_mut() {
            pass.initialize(device, queue, resources);
        }

        CompiledRenderGraph { passes: compiled }
    }
}
