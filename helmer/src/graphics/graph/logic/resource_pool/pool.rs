use std::{
    marker::PhantomData,
    time::{Duration, Instant},
};

use generational_arena::{Arena, Index};

use crate::graphics::{
    graph::logic::resource_pool::{
        error::ResourcePoolError, evictable_pool::EvictablePool, resource::Resource,
        timer_wheel::TimerWheel,
    },
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

pub struct ResourcePool {
    wheel: TimerWheel,

    meshes: EvictablePool<Mesh>,
    materials: EvictablePool<MaterialLowEnd>,
    textures: EvictablePool<wgpu::Texture>,
    texture_views: EvictablePool<wgpu::TextureView>,
    bind_groups: EvictablePool<wgpu::BindGroup>,
}

impl ResourcePool {
    pub fn new(timeout: Duration, resolution: Duration) -> Self {
        Self {
            wheel: TimerWheel::new(timeout, resolution),

            meshes: EvictablePool::new(),
            materials: EvictablePool::new(),
            textures: EvictablePool::new(),
            texture_views: EvictablePool::new(),
            bind_groups: EvictablePool::new(),
        }
    }
    // Mesh methods
    pub fn add_mesh(&mut self, mesh: Mesh) -> ResourceHandle<Mesh> {
        ResourceHandle::from_index(self.meshes.arena.insert(Resource::new(mesh)))
    }

    pub fn get_mesh(&mut self, handle: ResourceHandle<Mesh>) -> Option<&Mesh> {
        let idx = handle.index;

        if let Some(resource) = self.meshes.arena.get_mut(idx) {
            resource.last_used = Instant::now();
            let idx_raw = idx.into_raw_parts().0;

            self.wheel.schedule(
                idx,
                resource.last_used,
                &mut self.meshes.wheel_index[idx_raw],
            );

            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_mesh(&mut self, handle: ResourceHandle<Mesh>) {
        self.meshes.arena.remove(handle.index);
    }
    // -----

    // Material methods
    pub fn add_material(&mut self, material: MaterialLowEnd) -> ResourceHandle<MaterialLowEnd> {
        ResourceHandle::from_index(self.materials.arena.insert(Resource::new(material)))
    }

    pub fn get_material(
        &mut self,
        handle: ResourceHandle<MaterialLowEnd>,
    ) -> Option<&MaterialLowEnd> {
        let idx = handle.index;

        if let Some(resource) = self.materials.arena.get_mut(idx) {
            resource.last_used = Instant::now();
            let idx_raw = idx.into_raw_parts().0;

            self.wheel.schedule(
                idx,
                resource.last_used,
                &mut self.materials.wheel_index[idx_raw],
            );

            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_material(&mut self, handle: ResourceHandle<MaterialLowEnd>) {
        self.materials.arena.remove(handle.get_index());
    }
    // -----

    // Texture methods
    pub fn add_texture(&mut self, texture: wgpu::Texture) -> ResourceHandle<wgpu::Texture> {
        ResourceHandle::from_index(self.textures.arena.insert(Resource::new(texture)))
    }

    pub fn get_texture(&mut self, handle: ResourceHandle<wgpu::Texture>) -> Option<&wgpu::Texture> {
        let idx = handle.index;

        if let Some(resource) = self.textures.arena.get_mut(idx) {
            resource.last_used = Instant::now();
            let idx_raw = idx.into_raw_parts().0;

            self.wheel.schedule(
                idx,
                resource.last_used,
                &mut self.textures.wheel_index[idx_raw],
            );

            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_texture(&mut self, handle: ResourceHandle<wgpu::Texture>) {
        self.textures.arena.remove(handle.index);
    }
    // -----

    // Texture View methods
    pub fn add_texture_view(
        &mut self,
        texture_view: wgpu::TextureView,
    ) -> ResourceHandle<wgpu::TextureView> {
        ResourceHandle::from_index(self.texture_views.arena.insert(Resource::new(texture_view)))
    }

    pub fn get_texture_view(
        &mut self,
        handle: ResourceHandle<wgpu::TextureView>,
    ) -> Option<&wgpu::TextureView> {
        let idx = handle.index;

        if let Some(resource) = self.texture_views.arena.get_mut(idx) {
            resource.last_used = Instant::now();
            let idx_raw = idx.into_raw_parts().0;

            self.wheel.schedule(
                idx,
                resource.last_used,
                &mut self.texture_views.wheel_index[idx_raw],
            );

            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_texture_view(&mut self, handle: ResourceHandle<wgpu::TextureView>) {
        self.texture_views.arena.remove(handle.get_index());
    }
    // -----

    // Bind Group methods
    pub fn add_bind_group(
        &mut self,
        bind_group: wgpu::BindGroup,
    ) -> ResourceHandle<wgpu::BindGroup> {
        ResourceHandle::from_index(self.bind_groups.arena.insert(Resource::new(bind_group)))
    }

    pub fn get_bind_group(
        &mut self,
        handle: ResourceHandle<wgpu::BindGroup>,
    ) -> Option<&wgpu::BindGroup> {
        let idx = handle.index;

        if let Some(resource) = self.bind_groups.arena.get_mut(idx) {
            resource.last_used = Instant::now();
            let idx_raw = idx.into_raw_parts().0;

            self.wheel.schedule(
                idx,
                resource.last_used,
                &mut self.bind_groups.wheel_index[idx_raw],
            );

            return Some(&resource.inner);
        }
        None
    }

    pub fn evict_bind_group(&mut self, handle: ResourceHandle<wgpu::BindGroup>) {
        self.bind_groups.arena.remove(handle.get_index());
    }
    // -----

    pub fn tick(&mut self) {
        let now = Instant::now();

        self.wheel.tick(&mut self.meshes, now);
        self.wheel.tick(&mut self.materials, now);
        self.wheel.tick(&mut self.textures, now);
        self.wheel.tick(&mut self.texture_views, now);
        self.wheel.tick(&mut self.bind_groups, now);
    }
}
