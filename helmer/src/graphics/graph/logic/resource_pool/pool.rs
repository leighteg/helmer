use std::{marker::PhantomData, time::Instant};

use generational_arena::{Arena, Index};

use crate::graphics::{
    graph::logic::resource_pool::{error::ResourcePoolError, resource::Resource},
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
    meshes: Arena<Resource<Mesh>>,
    meshes_to_evict: Vec<ResourceHandle<Mesh>>,

    materials: Arena<Resource<MaterialLowEnd>>,
    materials_to_evict: Vec<ResourceHandle<MaterialLowEnd>>,

    textures: Arena<Resource<wgpu::Texture>>,
    textures_to_evict: Vec<ResourceHandle<wgpu::Texture>>,

    texture_views: Arena<Resource<wgpu::TextureView>>,
    texture_views_to_evict: Vec<ResourceHandle<wgpu::TextureView>>,

    bind_groups: Arena<Resource<wgpu::BindGroup>>,
    bind_groups_to_evict: Vec<ResourceHandle<wgpu::BindGroup>>,
}

impl ResourcePool {
    // Mesh methods
    pub fn add_mesh(&mut self, mesh: Mesh) -> ResourceHandle<Mesh> {
        ResourceHandle::from_index(self.meshes.insert(Resource::new(mesh)))
    }

    pub fn get_mesh(&mut self, handle: ResourceHandle<Mesh>) -> Option<&Mesh> {
        if let Some(resource) = self.meshes.get_mut(handle.index) {
            resource.last_used = Instant::now();
            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_mesh(&mut self, handle: ResourceHandle<Mesh>) {
        self.meshes_to_evict.push(handle);
    }
    // -----

    // Material methods
    pub fn add_material(&mut self, material: MaterialLowEnd) -> ResourceHandle<MaterialLowEnd> {
        ResourceHandle::from_index(self.materials.insert(Resource::new(material)))
    }

    pub fn get_material(&mut self, handle: ResourceHandle<Mesh>) -> Option<&MaterialLowEnd> {
        if let Some(resource) = self.materials.get_mut(handle.index) {
            resource.last_used = Instant::now();
            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_material(&mut self, handle: ResourceHandle<MaterialLowEnd>) {
        self.materials_to_evict.push(handle);
    }
    // -----

    // Texture methods
    pub fn add_texture(&mut self, texture: wgpu::Texture) -> ResourceHandle<wgpu::Texture> {
        ResourceHandle::from_index(self.textures.insert(Resource::new(texture)))
    }

    pub fn get_texture(&mut self, handle: ResourceHandle<wgpu::Texture>) -> Option<&wgpu::Texture> {
        if let Some(resource) = self.textures.get_mut(handle.index) {
            resource.last_used = Instant::now();
            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_texture(&mut self, handle: ResourceHandle<wgpu::Texture>) {
        self.textures_to_evict.push(handle);
    }
    // -----

    // Texture View methods
    pub fn add_texture_view(
        &mut self,
        texture_view: wgpu::TextureView,
    ) -> ResourceHandle<wgpu::TextureView> {
        ResourceHandle::from_index(self.texture_views.insert(Resource::new(texture_view)))
    }

    pub fn get_texture_view(
        &mut self,
        handle: ResourceHandle<wgpu::TextureView>,
    ) -> Option<&wgpu::TextureView> {
        if let Some(resource) = self.texture_views.get_mut(handle.index) {
            resource.last_used = Instant::now();
            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_texture_view(&mut self, handle: ResourceHandle<wgpu::TextureView>) {
        self.texture_views_to_evict.push(handle);
    }
    // -----

    // Bind Group methods
    pub fn add_bind_group(
        &mut self,
        bind_group: wgpu::BindGroup,
    ) -> ResourceHandle<wgpu::BindGroup> {
        ResourceHandle::from_index(self.bind_groups.insert(Resource::new(bind_group)))
    }

    pub fn get_bind_group(
        &mut self,
        handle: ResourceHandle<wgpu::BindGroup>,
    ) -> Option<&wgpu::BindGroup> {
        if let Some(resource) = self.bind_groups.get_mut(handle.index) {
            resource.last_used = Instant::now();
            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_bind_group(&mut self, handle: ResourceHandle<wgpu::BindGroup>) {
        self.bind_groups_to_evict.push(handle);
    }
    // -----

    pub fn free_resources(&mut self) {
        for mesh in &self.meshes_to_evict {
            self.meshes.remove(mesh.index);
        }

        for material in &self.materials_to_evict {
            self.materials.remove(material.index);
        }

        for texture in &self.textures_to_evict {
            self.textures.remove(texture.index);
        }

        for texture_view in &self.texture_views_to_evict {
            self.texture_views.remove(texture_view.index);
        }

        for bind_group in &self.bind_groups_to_evict {
            self.bind_groups.remove(bind_group.index);
        }
    }
}
