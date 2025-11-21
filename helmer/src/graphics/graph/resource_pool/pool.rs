use std::{marker::PhantomData, time::Instant};

use generational_arena::{Arena, Index};
use slotmap::new_key_type;

use crate::graphics::{
    graph::resource_pool::{error::ResourcePoolError, resource::Resource},
    renderer_common::common::{Material, Mesh},
    renderers::forward_pmu::MaterialLowEnd,
};

pub struct ResourceHandle<T> {
    index: Index,
    _phantom: PhantomData<T>,
}

impl<T> ResourceHandle<T> {
    pub fn from_index(index: Index) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    pub fn get_index(&self) -> Index {
        self.index
    }
}

#[derive(Default)]
pub struct ResourcePool {
    pub meshes: Arena<Resource<Mesh>>,
    meshes_to_evict: Vec<ResourceHandle<Mesh>>,

    pub materials: Arena<Resource<MaterialLowEnd>>,
    materials_to_evict: Vec<ResourceHandle<MaterialLowEnd>>,

    pub textures: Arena<Resource<wgpu::Texture>>,
    textures_to_evict: Vec<ResourceHandle<wgpu::Texture>>,

    pub bind_groups: Arena<Resource<wgpu::BindGroup>>,
    bind_groups_to_evict: Vec<ResourceHandle<wgpu::BindGroup>>,
}

impl ResourcePool {
    pub fn add_texture(&mut self, texture: wgpu::Texture) -> ResourceHandle<wgpu::Texture> {
        ResourceHandle::from_index(self.textures.insert(Resource::new(texture)))
    }

    pub fn get_texture(&mut self, handle: ResourceHandle<wgpu::Texture>) -> Option<&wgpu::Texture> {
        if let Some(texture) = self.textures.get_mut(handle.index) {
            texture.last_used = Instant::now();
            return Some(&texture.inner);
        }
        None
    }

    pub fn evict_texture(
        &mut self,
        handle: ResourceHandle<wgpu::Texture>,
    ) -> Option<wgpu::Texture> {
        if let Some(texture) = self.textures.remove(handle.index) {
            return Some(texture.inner);
        }
        None
    }
}
