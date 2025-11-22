use egui_wgpu::Renderer as EguiRenderer;
use glam::{Mat4, Quat, Vec3, Vec4Swizzles};
use std::{
    cell::RefCell,
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::info;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::{
    graphics::{graph::logic::{graph::CompiledRenderGraph, resource_pool::pool::ResourcePool}, renderer_common::{
        atmosphere::AtmospherePrecomputer,
        common::{
            Aabb, CASCADE_SPLITS, CameraUniforms, CascadeUniform, EguiRenderData, FRAMES_IN_FLIGHT,
            InstanceRaw, LightData, Mesh, MeshLod, ModelPushConstant, NUM_CASCADES, PbrConstants,
            RenderData, RenderMessage, RenderObject, RenderTrait, SHADOW_MAP_RESOLUTION,
            ShaderConstants, ShadowPipeline, ShadowUniforms, SkyUniforms, Vertex,
        },
        error::RendererError,
        mipmap::MipmapGenerator,
    }},
    provided::components::{Camera, LightType},
    runtime::asset_server::MaterialGpuData,
};

/// The render graph renderer
pub struct GraphRenderer {
    // WGPU Core
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    resources: ResourcePool,
    graph: CompiledRenderGraph,
}

impl GraphRenderer {
    pub async fn new(
        _instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        size: PhysicalSize<u32>,
        target_tickrate: f32,
    ) -> Result<Self, RendererError> {
        let required_features = wgpu::Features::empty();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Render Device"),
                required_features,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 0,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .map_err(|e| {
                RendererError::ResourceCreation(format!("Failed to create device: {}", e))
            })?;

        let device = Arc::new(device);
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mut renderer = Self {
            device,
            queue,
            surface,
            surface_config,

            resources: ResourcePool::new(Duration::from_secs(5), Duration::from_secs(2)),
            graph: CompiledRenderGraph::default(),
        };

        info!("initialized renderer");
        Ok(renderer)
    }

    pub fn render(&mut self) -> Result<(), RendererError> {
        let output_frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost) => {
                //self.resize(self.window_size);
                return Ok(());
            }
            Err(e) => return Err(RendererError::ResourceCreation(e.to_string())),
        };
        let output_view = output_frame.texture.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Command Encoder"),
            });

        self.graph.execute(&self.device, &self.queue, &mut encoder, &mut self.resources);

        // --- Submit and Present ---
        self.queue.submit(std::iter::once(encoder.finish()));
        output_frame.present();
        Ok(())
    }
}