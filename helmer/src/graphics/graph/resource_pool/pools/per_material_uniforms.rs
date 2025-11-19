use slotmap::SlotMap;

use crate::graphics::{graph::resource_pool::{error::ResourcePoolError, pool::{MaterialHandle, MeshHandle, ResourcePool, TextureHandle}, resource::Resource}, renderer_common::common::Mesh, renderers::forward_pmu::MaterialLowEnd};

#[derive(Default)]
pub struct PerMaterialUniformsPool {
    meshes: SlotMap<MeshHandle, Resource<Mesh>>,
    materials: SlotMap<MaterialHandle, Resource<MaterialLowEnd>>,

    textures: SlotMap<TextureHandle, Resource<wgpu::Texture>>,

    meshes_to_evict: Vec<MeshHandle>,
    materials_to_evict: Vec<MaterialHandle>,
}

impl ResourcePool for PerMaterialUniformsPool {
    fn add_mesh(&mut self, mesh: Mesh) -> MeshHandle {
        self.meshes.insert(Resource::new(mesh))
    }

    fn get_mesh(&self, handle: MeshHandle) -> Option<&Resource<Mesh>> {
        self.meshes.get(handle)
    }

    fn evict_mesh(&mut self, handle: MeshHandle) -> Result<(), ResourcePoolError> {
        todo!()
    }
    
    fn add_material(&mut self, material: crate::graphics::renderer_common::common::Material) -> MaterialHandle {
        todo!()
    }
    
    fn get_material(&self, handle: MaterialHandle) -> Option<&Resource<crate::graphics::renderer_common::common::Material>> {
        todo!()
    }
    
    fn evict_material(&mut self, handle: MaterialHandle) -> Result<(), ResourcePoolError> {
        todo!()
    }

    fn cleanup(&mut self) {
        for handle in self.meshes_to_evict.iter() {
            self.meshes.remove(*handle);
        }

        for handle in self.materials_to_evict.iter() {
            self.materials.remove(*handle);
        }
    }
}