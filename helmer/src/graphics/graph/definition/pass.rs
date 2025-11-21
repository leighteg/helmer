use crate::graphics::{
    graph::logic::resource_pool::pool::ResourceHandle, renderer_common::common::Mesh,
    renderers::forward_pmu::MaterialLowEnd,
};

pub type MeshHandle = ResourceHandle<Mesh>;
pub type MaterialHandle = ResourceHandle<MaterialLowEnd>;
pub type TargetId = ResourceHandle<wgpu::TextureView>;

pub enum PassNodeDefinition {
    Forward {
        mesh_handles: Vec<MeshHandle>,
        material_handles: Vec<MaterialHandle>,
        output_target: TargetId,
    },
    Shadows {
        mesh_handles: Vec<MeshHandle>,
    },
}
