use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::graphics::{
    graph::{
        definition::{render_pass::RenderPass, resource_id::ResourceId},
        logic::{
            gpu_resource_pool::GpuResourcePool, graph_context::RenderGraphContext,
            graph_exec_ctx::RenderGraphExecCtx,
        },
    },
    passes::{FrameGlobals, SwapchainFrameInput},
    renderer_common::common::{EguiRenderData, EguiTextureCache},
};

#[derive(Clone, Copy, Debug)]
pub struct EguiOutputs {
    pub swapchain: ResourceId,
}

#[derive(Clone)]
pub struct EguiPass {
    outputs: EguiOutputs,
    renderer: Arc<parking_lot::RwLock<Option<egui_wgpu::Renderer>>>,
    format: Arc<parking_lot::RwLock<wgpu::TextureFormat>>,
    last_texture_version: Arc<AtomicU64>,
    last_epoch: Arc<AtomicU64>,
    last_screen_size: Arc<parking_lot::RwLock<[u32; 2]>>,
}

fn sanitize_egui_delta(
    delta: egui::TexturesDelta,
    cache: &EguiTextureCache,
) -> egui::TexturesDelta {
    let mut set_map: HashMap<egui::TextureId, egui::epaint::ImageDelta> = HashMap::new();
    for (id, delta) in delta.set {
        let mut next = delta;
        if next.pos.is_some() {
            if let Some(tex) = cache.atlas.get(&id) {
                let size = tex.image.size();
                if size[0] > 0 && size[1] > 0 {
                    next = egui::epaint::ImageDelta::full(tex.image.clone(), tex.options);
                }
            }
        }
        set_map.insert(id, next);
    }

    let mut free = delta.free;
    free.sort_unstable();
    free.dedup();

    egui::TexturesDelta {
        set: set_map.into_iter().collect(),
        free,
    }
}

impl EguiPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        swapchain: ResourceId,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // ensure the swapchain logical resource exists in the pool
        let _ = pool.entry(swapchain);

        Self {
            outputs: EguiOutputs { swapchain },
            renderer: Arc::new(parking_lot::RwLock::new(None)),
            format: Arc::new(parking_lot::RwLock::new(surface_format)),
            last_texture_version: Arc::new(AtomicU64::new(0)),
            last_epoch: Arc::new(AtomicU64::new(0)),
            last_screen_size: Arc::new(parking_lot::RwLock::new([0, 0])),
        }
    }

    pub fn outputs(&self) -> EguiOutputs {
        self.outputs
    }

    fn ensure_renderer(&self, device: &wgpu::Device, format: wgpu::TextureFormat) -> bool {
        let needs_rebuild = self.renderer.read().is_none() || *self.format.read() != format;
        if needs_rebuild {
            *self.format.write() = format;
            *self.renderer.write() = Some(egui_wgpu::Renderer::new(
                device,
                format,
                egui_wgpu::RendererOptions::default(),
            ));
            self.last_texture_version.store(0, Ordering::Relaxed);
            return true;
        }
        false
    }
}

impl RenderPass for EguiPass {
    fn name(&self) -> &'static str {
        "EguiPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.rw(self.outputs.swapchain);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let swapchain = match ctx.rpctx.frame_inputs.get::<SwapchainFrameInput>() {
            Some(s) => s,
            None => return,
        };
        let egui_data = match ctx.rpctx.frame_inputs.get::<EguiRenderData>() {
            Some(d) => d,
            None => return,
        };
        let cached_delta = ctx.rpctx.frame_inputs.get::<EguiTextureCache>();

        let recreated = self.ensure_renderer(ctx.device(), swapchain.format);
        let mut renderer_guard = self.renderer.write();
        let renderer = match renderer_guard.as_mut() {
            Some(r) => r,
            None => return,
        };

        let incoming_epoch = cached_delta
            .as_ref()
            .map(|c| c.epoch)
            .unwrap_or_else(|| self.last_epoch.load(Ordering::Relaxed));
        let epoch_changed = incoming_epoch != self.last_epoch.load(Ordering::Relaxed);
        if epoch_changed {
            self.last_epoch.store(incoming_epoch, Ordering::Relaxed);
            self.last_texture_version.store(0, Ordering::Relaxed);
        }

        let last_version = self.last_texture_version.load(Ordering::Relaxed);
        let mut screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: egui_data.screen_descriptor.size_in_pixels,
            pixels_per_point: egui_data.screen_descriptor.pixels_per_point,
        };

        if let Some(globals) = ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            screen_descriptor.size_in_pixels =
                [globals.surface_size.width, globals.surface_size.height];
        }

        if *self.last_screen_size.read() != screen_descriptor.size_in_pixels {
            *self.last_screen_size.write() = screen_descriptor.size_in_pixels;
            self.last_texture_version.store(0, Ordering::Relaxed);
        }

        let need_upload = last_version != egui_data.version;
        let force_full_upload =
            recreated || epoch_changed || self.last_texture_version.load(Ordering::Relaxed) == 0;
        let mut upload_delta: Option<egui::TexturesDelta> = None;

        if force_full_upload {
            if let Some(cache) = cached_delta.as_ref() {
                if !cache.atlas.is_empty() {
                    let mut full = egui::TexturesDelta::default();
                    for (id, tex) in &cache.atlas {
                        full.set.push((
                            *id,
                            egui::epaint::ImageDelta::full(tex.image.clone(), tex.options),
                        ));
                    }
                    upload_delta = Some(full);
                }
            }
        }

        if upload_delta.is_none() && need_upload {
            if !egui_data.textures_delta.set.is_empty() || !egui_data.textures_delta.free.is_empty()
            {
                upload_delta = Some(egui_data.textures_delta.clone());
            } else if let Some(cache) = cached_delta.as_ref() {
                if !cache.atlas.is_empty() {
                    let mut full = egui::TexturesDelta::default();
                    for (id, tex) in &cache.atlas {
                        full.set.push((
                            *id,
                            egui::epaint::ImageDelta::full(tex.image.clone(), tex.options),
                        ));
                    }
                    upload_delta = Some(full);
                }
            }
        }

        if let Some(mut delta) = upload_delta.take() {
            for id in &egui_data.textures_delta.free {
                if !delta.free.contains(id) {
                    delta.free.push(*id);
                }
            }
            delta
                .set
                .retain(|(_, d)| d.image.height() > 0 && d.image.width() > 0);

            if let Some(cache) = cached_delta.as_ref() {
                delta = sanitize_egui_delta(delta, cache);
            }

            for (id, d) in &delta.set {
                renderer.update_texture(ctx.device(), ctx.queue(), *id, d);
            }
            for id in &delta.free {
                renderer.free_texture(id);
            }
            self.last_texture_version
                .store(egui_data.version, Ordering::Relaxed);
        } else if need_upload {
            // try again next frame with a forced upload
            self.last_texture_version.store(0, Ordering::Relaxed);
        }

        let device = ctx.rpctx.device;
        let queue = ctx.rpctx.queue;
        renderer.update_buffers(
            device,
            queue,
            ctx.encoder,
            &egui_data.primitives,
            &screen_descriptor,
        );

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/Egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &swapchain.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        renderer.render(
            &mut pass.forget_lifetime(),
            &egui_data.primitives,
            &screen_descriptor,
        );

        ctx.rpctx.mark_used(self.outputs.swapchain);
    }
}
