use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// A modular, reusable utility for generating mipmap chains on the GPU.
/// It uses a series of render passes ("blits") to downscale a texture.
pub struct MipmapGenerator {
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipelines: Mutex<HashMap<wgpu::TextureFormat, Arc<wgpu::RenderPipeline>>>,
}

impl MipmapGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/mipmap.wgsl"));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mipmap Blit Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mipmap Blit BGL"),
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
            label: Some("Mipmap Blit Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        Self {
            shader,
            pipeline_layout,
            bind_group_layout,
            sampler,
            pipelines: Mutex::new(HashMap::new()),
        }
    }

    /// Generates a full mipmap chain for the given texture.
    pub fn generate_mips(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        mip_count: u32,
    ) {
        self.generate_mips_from(encoder, device, texture, 0, mip_count);
    }

    /// Generates mipmaps starting from the provided base level.
    pub fn generate_mips_from(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        start_level: u32,
        mip_count: u32,
    ) {
        if mip_count <= 1 {
            return;
        }
        let start_level = start_level.min(mip_count.saturating_sub(1));
        if start_level + 1 >= mip_count {
            return;
        }

        let format = texture.format();
        let pipeline = {
            let mut pipelines = self.pipelines.lock().expect("mipmap pipeline lock");
            pipelines
                .entry(format)
                .or_insert_with(|| {
                    Arc::new(
                        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some("Mipmap Blit Pipeline (runtime)"),
                            layout: Some(&self.pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &self.shader,
                                entry_point: Some("vs_main"),
                                buffers: &[],
                                compilation_options: Default::default(),
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &self.shader,
                                entry_point: Some("fs_main"),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format,
                                    blend: None,
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                                compilation_options: Default::default(),
                            }),
                            primitive: wgpu::PrimitiveState::default(),
                            depth_stencil: None,
                            multisample: wgpu::MultisampleState::default(),
                            multiview_mask: None,
                            cache: None,
                        }),
                    )
                })
                .clone()
        };

        for i in (start_level + 1)..mip_count {
            let src_mip = i - 1;
            let dst_mip = i;

            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Mip Src View (Level {})", src_mip)),
                base_mip_level: src_mip,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Mip Dst View (Level {})", dst_mip)),
                base_mip_level: dst_mip,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mip Blit Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Mipmap Blit Pass (Level {})", dst_mip)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&pipeline); // Use the cached format-specific pipeline
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1); // Draw a fullscreen triangle
        }
    }
}
