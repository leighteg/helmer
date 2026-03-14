use hashbrown::{HashMap, HashSet};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::graphics::{
    graph::{
        definition::{render_pass::RenderPass, resource_id::ResourceId},
        logic::{graph_context::RenderGraphContext, graph_exec_ctx::RenderGraphExecCtx},
    },
    passes::{FrameGlobals, SwapchainFrameInput, UiInstanceRaw},
};

#[derive(Clone)]
struct UiTextureBindGroupCacheEntry {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy, Debug)]
pub struct UiOutputs {
    pub swapchain: ResourceId,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiPassParamsGpu {
    viewport_size: [f32; 2],
    viewport_inv_size: [f32; 2],
}

#[derive(Clone)]
pub struct UiPass {
    outputs: UiOutputs,
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    params_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    texture_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    params_buffer: Arc<RwLock<Option<wgpu::Buffer>>>,
    params_bind_group: Arc<RwLock<Option<wgpu::BindGroup>>>,
    texture_bind_group_cache: Arc<RwLock<HashMap<u32, UiTextureBindGroupCacheEntry>>>,
    format: Arc<RwLock<wgpu::TextureFormat>>,
}

impl UiPass {
    pub fn new(swapchain: ResourceId, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            outputs: UiOutputs { swapchain },
            pipeline: Arc::new(RwLock::new(None)),
            params_bgl: Arc::new(RwLock::new(None)),
            texture_bgl: Arc::new(RwLock::new(None)),
            params_buffer: Arc::new(RwLock::new(None)),
            params_bind_group: Arc::new(RwLock::new(None)),
            texture_bind_group_cache: Arc::new(RwLock::new(HashMap::new())),
            format: Arc::new(RwLock::new(surface_format)),
        }
    }

    pub fn outputs(&self) -> UiOutputs {
        self.outputs
    }

    fn ensure_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let mut guard = self.params_buffer.write();
        if guard.is_none() {
            *guard = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("UiPass/Params"),
                size: std::mem::size_of::<UiPassParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        guard.as_ref().unwrap().clone()
    }

    fn ensure_pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        if self.pipeline.read().is_some() && *self.format.read() == format {
            return;
        }

        *self.format.write() = format;

        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UiPass/ParamsBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<UiPassParamsGpu>() as u64
                    ),
                },
                count: None,
            }],
        });

        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UiPass/TextureBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UiPass/PipelineLayout"),
            bind_group_layouts: &[&params_bgl, &texture_bgl],
            immediate_size: 0,
        });

        const INSTANCE_ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Float32x4,
            1 => Float32x4,
            2 => Float32x4,
            3 => Float32x4,
            4 => Float32x4,
        ];

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UiInstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRS,
        };

        let shader = device.create_shader_module(wgpu::include_wgsl!("ui.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UiPass/Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.params_bgl.write() = Some(params_bgl);
        *self.texture_bgl.write() = Some(texture_bgl);
        *self.params_bind_group.write() = None;
        self.texture_bind_group_cache.write().clear();
    }

    fn ensure_params_bind_group(
        &self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let mut guard = self.params_bind_group.write();
        if guard.is_none() {
            *guard = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("UiPass/ParamsBG"),
                layout: self.params_bgl.read().as_ref().unwrap(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                }],
            }));
        }
        guard.as_ref().unwrap().clone()
    }
}

impl RenderPass for UiPass {
    fn name(&self) -> &'static str {
        "UiPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.rw(self.outputs.swapchain);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(frame) => frame,
            None => return,
        };
        if !frame.render_config.ui_pass {
            return;
        }

        let swapchain = match ctx.rpctx.frame_inputs.get::<SwapchainFrameInput>() {
            Some(s) => s,
            None => return,
        };

        self.ensure_pipeline(ctx.device(), swapchain.format);

        let instance_buffer = match frame.ui_instances.as_ref() {
            Some(buffer) if buffer.count > 0 => buffer,
            _ => {
                ctx.rpctx.mark_used(self.outputs.swapchain);
                return;
            }
        };

        let viewport = [
            swapchain.size_in_pixels[0].max(1) as f32,
            swapchain.size_in_pixels[1].max(1) as f32,
        ];
        let params = UiPassParamsGpu {
            viewport_size: viewport,
            viewport_inv_size: [1.0 / viewport[0], 1.0 / viewport[1]],
        };
        let params_buffer = self.ensure_params_buffer(ctx.device());
        ctx.queue()
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        let params_bg = self.ensure_params_bind_group(ctx.device(), &params_buffer);

        let texture_layout = self.texture_bgl.read().as_ref().unwrap().clone();
        let mut texture_bind_groups: HashMap<u32, wgpu::BindGroup> = HashMap::new();
        let mut used_texture_slots: HashSet<u32> = HashSet::with_capacity(frame.ui_batches.len());
        let mut cache = self.texture_bind_group_cache.write();
        for batch in frame.ui_batches.iter() {
            if batch.instance_range.start >= batch.instance_range.end {
                continue;
            }
            if texture_bind_groups.contains_key(&batch.texture_slot) {
                continue;
            }
            used_texture_slots.insert(batch.texture_slot);
            let texture_slot = batch.texture_slot as usize;
            let view = frame
                .ui_textures
                .get(texture_slot)
                .unwrap_or(&frame.fallback_view)
                .clone();
            let sampler = frame.scene_sampler.clone();
            let needs_rebuild = cache
                .get(&batch.texture_slot)
                .map(|entry| entry.view != view || entry.sampler != sampler)
                .unwrap_or(true);
            if needs_rebuild {
                let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("UiPass/TextureBG"),
                    layout: &texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });
                cache.insert(
                    batch.texture_slot,
                    UiTextureBindGroupCacheEntry {
                        view,
                        sampler,
                        bind_group,
                    },
                );
            }
            if let Some(entry) = cache.get(&batch.texture_slot) {
                texture_bind_groups.insert(batch.texture_slot, entry.bind_group.clone());
            }
        }
        cache.retain(|slot, _| used_texture_slots.contains(slot));
        drop(cache);

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/UI"),
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

        pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
        pass.set_vertex_buffer(0, instance_buffer.buffer.slice(..));
        pass.set_bind_group(0, &params_bg, &[]);

        for batch in frame.ui_batches.iter() {
            if batch.instance_range.start >= batch.instance_range.end {
                continue;
            }
            let Some(bind_group) = texture_bind_groups.get(&batch.texture_slot) else {
                continue;
            };
            pass.set_bind_group(1, bind_group, &[]);
            pass.draw(0..6, batch.instance_range.clone());
        }

        ctx.rpctx.mark_used(self.outputs.swapchain);
    }
}
