use std::{
    marker::PhantomData,
    time::{Duration, Instant},
};

use generational_arena::{Arena, Index};

use crate::graphics::{
    graph::logic::resource_pool::{
        error::ResourcePoolError, evictable_pool::EvictablePool, handle::ResourceHandle,
        resource::Resource, timer_wheel::TimerWheel,
    },
    renderer_common::common::{Material, Mesh},
    renderers::forward_pmu::MaterialLowEnd,
};

pub struct ResourcePool {
    wheel: TimerWheel,
    last_tick: Instant,

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
            last_tick: Instant::now(),

            meshes: EvictablePool::new(),
            materials: EvictablePool::new(),
            textures: EvictablePool::new(),
            texture_views: EvictablePool::new(),
            bind_groups: EvictablePool::new(),
        }
    }

    // generic methods
    fn get_resource<'a, T>(
        pool: &'a mut EvictablePool<T>,
        wheel: &'a mut TimerWheel,
        handle: ResourceHandle<T>,
    ) -> Option<&'a T> {
        let idx = handle.index;

        if let Some(resource) = pool.arena.get_mut(idx) {
            resource.last_used = Instant::now();
            let idx_raw = idx.into_raw_parts().0;

            wheel.schedule(idx, resource.last_used, &mut pool.wheel_index[idx_raw]);

            return Some(&resource.inner);
        }
        None
    }

    fn get_resource_mut<'a, T>(
        pool: &'a mut EvictablePool<T>,
        wheel: &'a mut TimerWheel,
        handle: ResourceHandle<T>,
    ) -> Option<&'a mut T> {
        let idx = handle.index;

        if let Some(resource) = pool.arena.get_mut(idx) {
            resource.last_used = Instant::now();
            let idx_raw = idx.into_raw_parts().0;

            wheel.schedule(idx, resource.last_used, &mut pool.wheel_index[idx_raw]);

            return Some(&mut resource.inner);
        }
        None
    }

    // Mesh methods
    pub fn add_mesh(&mut self, mesh: Mesh) -> ResourceHandle<Mesh> {
        ResourceHandle::from_index(self.meshes.insert(Resource::new(mesh)))
    }

    pub fn get_mesh(&mut self, handle: ResourceHandle<Mesh>) -> Option<&Mesh> {
        Self::get_resource(&mut self.meshes, &mut self.wheel, handle)
    }

    pub fn evict_mesh(&mut self, handle: ResourceHandle<Mesh>) {
        self.meshes.remove(handle.index);
    }
    // -----

    // Material methods
    pub fn add_material(&mut self, material: MaterialLowEnd) -> ResourceHandle<MaterialLowEnd> {
        ResourceHandle::from_index(self.materials.insert(Resource::new(material)))
    }

    pub fn get_material(
        &mut self,
        handle: ResourceHandle<MaterialLowEnd>,
    ) -> Option<&MaterialLowEnd> {
        Self::get_resource(&mut self.materials, &mut self.wheel, handle)
    }

    pub fn evict_material(&mut self, handle: ResourceHandle<MaterialLowEnd>) {
        self.materials.remove(handle.index);
    }
    // -----

    // Texture methods
    pub fn add_texture(&mut self, texture: wgpu::Texture) -> ResourceHandle<wgpu::Texture> {
        ResourceHandle::from_index(self.textures.insert(Resource::new(texture)))
    }

    pub fn get_texture(&mut self, handle: ResourceHandle<wgpu::Texture>) -> Option<&wgpu::Texture> {
        Self::get_resource(&mut self.textures, &mut self.wheel, handle)
    }

    pub fn evict_texture(&mut self, handle: ResourceHandle<wgpu::Texture>) {
        self.textures.remove(handle.index);
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
        Self::get_resource(&mut self.texture_views, &mut self.wheel, handle)
    }

    pub fn evict_texture_view(&mut self, handle: ResourceHandle<wgpu::TextureView>) {
        self.texture_views.remove(handle.index);
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
        Self::get_resource(&mut self.bind_groups, &mut self.wheel, handle)
    }

    pub fn evict_bind_group(&mut self, handle: ResourceHandle<wgpu::BindGroup>) {
        self.bind_groups.remove(handle.index);
    }
    // -----

    pub fn tick(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_tick);

        // Only tick if enough time has passed
        if elapsed >= self.wheel.resolution {
            self.wheel.tick(&mut self.meshes, now);
            self.wheel.tick(&mut self.materials, now);
            self.wheel.tick(&mut self.textures, now);
            self.wheel.tick(&mut self.texture_views, now);
            self.wheel.tick(&mut self.bind_groups, now);

            self.last_tick = now;
        }
    }
}
