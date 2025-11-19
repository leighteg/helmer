use slotmap::new_key_type;

use crate::graphics::{graph::resource_pool::{error::ResourcePoolError, resource::Resource}, renderer_common::common::{Material, Mesh}};

new_key_type! { pub struct MeshHandle; }
new_key_type! { pub struct MaterialHandle; }

new_key_type! { pub struct TextureHandle; }
new_key_type! { pub struct BufferHandle; }
new_key_type! { pub struct BindGroupHandle; }

pub trait ResourcePool {
    fn add_mesh(&mut self, mesh: Mesh) -> MeshHandle;
    fn get_mesh(&self, handle: MeshHandle) -> Option<&Resource<Mesh>>;
    fn evict_mesh(&mut self, handle: MeshHandle) -> Result<(), ResourcePoolError>;

    fn add_material(&mut self, material: Material) -> MaterialHandle;
    fn get_material(&self, handle: MaterialHandle) -> Option<&Resource<Material>>;
    fn evict_material(&mut self, handle: MaterialHandle) -> Result<(), ResourcePoolError>;

    fn cleanup(&mut self);
}