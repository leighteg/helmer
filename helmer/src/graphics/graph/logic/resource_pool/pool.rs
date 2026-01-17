use std::{
    marker::PhantomData,
    time::{Duration, Instant},
};

use generational_arena::{Arena, Index};

use crate::graphics::{
    graph::logic::resource_pool::{
        error::ResourcePoolError,
        evictable_pool::EvictablePool,
        handle::ResourceHandle,
        resource::Resource,
        timer_wheel::{PoolId, TimerWheel},
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
    samplers: EvictablePool<wgpu::Sampler>,
    bind_group_layouts: EvictablePool<wgpu::BindGroupLayout>,
    bind_groups: EvictablePool<wgpu::BindGroup>,
    buffers: EvictablePool<wgpu::Buffer>,
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
            samplers: EvictablePool::new(),
            bind_group_layouts: EvictablePool::new(),
            bind_groups: EvictablePool::new(),
            buffers: EvictablePool::new(),
        }
    }

    // generic methods
    fn get_resource<'a, T>(
        pool: &'a mut EvictablePool<T>,
        wheel: &'a mut TimerWheel,
        handle: ResourceHandle<T>,
        pool_id: PoolId,
    ) -> Option<&'a T> {
        let now = Instant::now();
        let idx = handle.index;

        if let Some(resource) = pool.arena.get_mut(idx) {
            let previous_last_used = resource.last_used;
            resource.last_used = now;
            let idx_raw = idx.into_raw_parts().0;

            wheel.schedule(
                pool_id,
                idx,
                previous_last_used,
                now,
                &mut pool.wheel_index[idx_raw],
            );

            return Some(&resource.inner);
        }
        None
    }

    fn get_resource_mut<'a, T>(
        pool: &'a mut EvictablePool<T>,
        wheel: &'a mut TimerWheel,
        handle: ResourceHandle<T>,
        pool_id: PoolId,
    ) -> Option<&'a mut T> {
        let now = Instant::now();
        let idx = handle.index;

        if let Some(resource) = pool.arena.get_mut(idx) {
            let previous_last_used = resource.last_used;
            resource.last_used = now;
            let idx_raw = idx.into_raw_parts().0;

            wheel.schedule(
                pool_id,
                idx,
                previous_last_used,
                now,
                &mut pool.wheel_index[idx_raw],
            );

            return Some(&mut resource.inner);
        }
        None
    }

    // Mesh methods
    pub fn add_mesh(&mut self, mesh: Mesh) -> ResourceHandle<Mesh> {
        ResourceHandle::from_index(self.meshes.insert(Resource::new(mesh)))
    }

    pub fn get_mesh(&mut self, handle: ResourceHandle<Mesh>) -> Option<&Mesh> {
        Self::get_resource(&mut self.meshes, &mut self.wheel, handle, PoolId::Meshes)
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
        Self::get_resource(
            &mut self.materials,
            &mut self.wheel,
            handle,
            PoolId::Materials,
        )
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
        Self::get_resource(
            &mut self.textures,
            &mut self.wheel,
            handle,
            PoolId::Textures,
        )
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
        Self::get_resource(
            &mut self.texture_views,
            &mut self.wheel,
            handle,
            PoolId::TextureViews,
        )
    }

    pub fn evict_texture_view(&mut self, handle: ResourceHandle<wgpu::TextureView>) {
        self.texture_views.remove(handle.index);
    }
    // -----

    // Sampler methods
    pub fn add_sampler(&mut self, sampler: wgpu::Sampler) -> ResourceHandle<wgpu::Sampler> {
        ResourceHandle::from_index(self.samplers.insert(Resource::new(sampler)))
    }

    pub fn get_sampler(&mut self, handle: ResourceHandle<wgpu::Sampler>) -> Option<&wgpu::Sampler> {
        Self::get_resource(
            &mut self.samplers,
            &mut self.wheel,
            handle,
            PoolId::Samplers,
        )
    }

    pub fn evict_sampler(&mut self, handle: ResourceHandle<wgpu::Sampler>) {
        self.samplers.remove(handle.index);
    }
    // -----

    // Bind Group Layout methods
    pub fn add_bind_group_layout(
        &mut self,
        bind_group_layout: wgpu::BindGroupLayout,
    ) -> ResourceHandle<wgpu::BindGroupLayout> {
        ResourceHandle::from_index(
            self.bind_group_layouts
                .insert(Resource::new(bind_group_layout)),
        )
    }

    pub fn get_bind_group_layout(
        &mut self,
        handle: ResourceHandle<wgpu::BindGroupLayout>,
    ) -> Option<&wgpu::BindGroupLayout> {
        Self::get_resource(
            &mut self.bind_group_layouts,
            &mut self.wheel,
            handle,
            PoolId::BindGroupLayouts,
        )
    }

    pub fn evict_bind_group_layout(&mut self, handle: ResourceHandle<wgpu::BindGroupLayout>) {
        self.bind_group_layouts.remove(handle.index);
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
        Self::get_resource(
            &mut self.bind_groups,
            &mut self.wheel,
            handle,
            PoolId::BindGroups,
        )
    }

    pub fn evict_bind_group(&mut self, handle: ResourceHandle<wgpu::BindGroup>) {
        self.bind_groups.remove(handle.index);
    }
    // -----

    // Buffer methods
    pub fn add_buffer(&mut self, buffer: wgpu::Buffer) -> ResourceHandle<wgpu::Buffer> {
        ResourceHandle::from_index(self.buffers.insert(Resource::new(buffer)))
    }

    pub fn get_buffer(&mut self, handle: ResourceHandle<wgpu::Buffer>) -> Option<&wgpu::Buffer> {
        Self::get_resource(&mut self.buffers, &mut self.wheel, handle, PoolId::Buffers)
    }

    pub fn evict_buffer(&mut self, handle: ResourceHandle<wgpu::Buffer>) {
        self.buffers.remove(handle.index);
    }
    // -----

    pub fn tick(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_tick);

        // Only tick if enough time has passed
        if elapsed >= self.wheel.resolution {
            self.wheel.tick(
                now,
                &mut self.meshes,
                &mut self.materials,
                &mut self.textures,
                &mut self.texture_views,
                &mut self.samplers,
                &mut self.bind_group_layouts,
                &mut self.bind_groups,
                &mut self.buffers,
            );

            self.last_tick = now;
        }
    }
}
