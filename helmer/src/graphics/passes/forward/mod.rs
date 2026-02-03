use parking_lot::RwLock;
use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

use crate::graphics::{
    backend::binding_backend::BindingBackendKind,
    common::renderer::{
        MeshDrawParams, MeshTaskTiling, ShaderConstants, Vertex, mesh_shader_visibility,
        mesh_task_tiling, transient_usage,
    },
    graph::{
        definition::{
            render_pass::RenderPass, resource_desc::ResourceDesc, resource_flags::ResourceFlags,
            resource_id::ResourceId,
        },
        logic::{
            gpu_resource_pool::GpuResourcePool, graph_context::RenderGraphContext,
            graph_exec_ctx::RenderGraphExecCtx,
        },
    },
    passes::FrameGlobals,
};

use super::gbuffer::GBufferInstanceRaw;

struct MaterialBindGroupCache {
    version: u64,
    groups: Arc<Vec<wgpu::BindGroup>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ForwardOutputs {
    pub color: ResourceId,
}

#[derive(Clone)]
pub struct ForwardPass {
    outputs: ForwardOutputs,
    extent: (u32, u32),
    pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    camera_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    material_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    constants_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    mesh_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    mesh_params: Arc<RwLock<Option<wgpu::Buffer>>>,
    mesh_params_capacity: Arc<AtomicU64>,
    texture_array_size: Arc<AtomicU32>,
    backend_kind: Arc<RwLock<Option<BindingBackendKind>>>,
    material_bind_groups: Arc<RwLock<Option<MaterialBindGroupCache>>>,
    depth: ResourceId,
    use_transient_textures: bool,
}

impl ForwardPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        depth: ResourceId,
        width: u32,
        height: u32,
        use_transient_textures: bool,
        use_transient_aliasing: bool,
    ) -> Self {
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            use_transient_textures,
        );
        let (desc, mut hints) = ResourceDesc::Texture2D {
            width,
            height,
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
        }
        .with_hints();
        if use_transient_aliasing {
            hints.flags |= ResourceFlags::TRANSIENT;
        }
        let color = pool.create_logical(desc, Some(hints), 0, None);

        Self {
            outputs: ForwardOutputs { color },
            extent: (width, height),
            pipeline: Arc::new(RwLock::new(None)),
            mesh_pipeline: Arc::new(RwLock::new(None)),
            camera_bgl: Arc::new(RwLock::new(None)),
            material_bgl: Arc::new(RwLock::new(None)),
            constants_bgl: Arc::new(RwLock::new(None)),
            mesh_bgl: Arc::new(RwLock::new(None)),
            mesh_params: Arc::new(RwLock::new(None)),
            mesh_params_capacity: Arc::new(AtomicU64::new(0)),
            texture_array_size: Arc::new(AtomicU32::new(1)),
            backend_kind: Arc::new(RwLock::new(None)),
            material_bind_groups: Arc::new(RwLock::new(None)),
            depth,
            use_transient_textures,
        }
    }

    pub fn outputs(&self) -> ForwardOutputs {
        self.outputs
    }

    fn ensure_target(
        &self,
        ctx: &mut RenderGraphExecCtx,
        extent: (u32, u32),
    ) -> Option<wgpu::TextureView> {
        let usage = transient_usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            self.use_transient_textures,
        );
        let desc = ResourceDesc::Texture2D {
            width: extent.0,
            height: extent.1,
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
        };
        let needs_create = match ctx.rpctx.pool.entry(self.outputs.color) {
            Some(entry) => {
                let tex_ok = entry.texture.as_ref().map_or(false, |t| {
                    let size = t.size();
                    size.width == extent.0 && size.height == extent.1
                });
                !tex_ok
            }
            None => true,
        };
        let view = if needs_create {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("Forward/Color"),
                size: wgpu::Extent3d {
                    width: extent.0,
                    height: extent.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            ctx.rpctx.pool.realize_texture(
                self.outputs.color,
                desc,
                texture,
                view.clone(),
                ctx.rpctx.frame_index,
            );
            Some(view)
        } else {
            ctx.rpctx
                .pool
                .entry(self.outputs.color)
                .and_then(|e| e.texture.as_ref())
                .map(|tex| {
                    tex.create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        ..Default::default()
                    })
                })
        };
        if let Some(entry) = ctx.rpctx.pool.entry_mut(self.outputs.color) {
            if let Some(ref v) = view {
                entry.texture_view = Some(v.clone());
            }
        }

        ctx.rpctx
            .pool
            .mark_resident(self.outputs.color, ctx.rpctx.frame_index);
        view
    }

    fn ensure_backend(&self, binding_backend: BindingBackendKind) {
        let mut current = self.backend_kind.write();
        if current.map_or(true, |kind| kind != binding_backend) {
            *self.pipeline.write() = None;
            *self.mesh_pipeline.write() = None;
            *self.camera_bgl.write() = None;
            *self.material_bgl.write() = None;
            *self.constants_bgl.write() = None;
            *self.mesh_bgl.write() = None;
            *self.material_bind_groups.write() = None;
            self.texture_array_size.store(1, Ordering::Relaxed);
            *current = Some(binding_backend);
        }
    }

    fn ensure_pipeline(
        &self,
        device: &wgpu::Device,
        binding_backend: BindingBackendKind,
        texture_array_size: u32,
    ) {
        self.ensure_backend(binding_backend);
        if binding_backend == BindingBackendKind::BindGroups {
            if self.pipeline.read().is_some() {
                return;
            }
        } else {
            let current_size = self.texture_array_size.load(Ordering::Relaxed);
            if self.pipeline.read().is_some() && current_size == texture_array_size {
                return;
            }
        }

        let array_size = texture_array_size.max(1);
        if binding_backend != BindingBackendKind::BindGroups {
            self.texture_array_size.store(array_size, Ordering::Relaxed);
        } else {
            self.texture_array_size.store(1, Ordering::Relaxed);
        }

        let mesh_stage = mesh_shader_visibility(device);
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Forward/CameraBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | mesh_stage,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // Leave size unspecified to avoid validation mismatches across backends
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | mesh_stage,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let material_entries = if binding_backend == BindingBackendKind::BindGroups {
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ]
        } else {
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(std::num::NonZeroU32::new(array_size).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ]
        };
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Forward/MaterialBGL"),
            entries: &material_entries,
        });

        let constants_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Forward/ConstantsBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ShaderConstants>() as u64
                    ),
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward/PipelineLayout"),
            bind_group_layouts: &[&camera_bgl, &material_bgl, &constants_bgl],
            immediate_size: 0,
        });

        let shader = if binding_backend == BindingBackendKind::BindGroups {
            device.create_shader_module(wgpu::include_wgsl!("forward_bindgroups.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("forward.wgsl"))
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Forward/Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), GBufferInstanceRaw::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    // additive blend to accumulate unlit forward color without masking lighting
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        *self.pipeline.write() = Some(pipeline);
        *self.camera_bgl.write() = Some(camera_bgl);
        *self.material_bgl.write() = Some(material_bgl);
        *self.constants_bgl.write() = Some(constants_bgl);
    }

    fn ensure_mesh_pipeline(
        &self,
        device: &wgpu::Device,
        binding_backend: BindingBackendKind,
        texture_array_size: u32,
    ) {
        if mesh_shader_visibility(device).is_empty() {
            return;
        }

        if binding_backend == BindingBackendKind::BindGroups {
            if self.mesh_pipeline.read().is_some() {
                return;
            }
        } else {
            let current_size = self.texture_array_size.load(Ordering::Relaxed);
            if self.mesh_pipeline.read().is_some() && current_size == texture_array_size {
                return;
            }
        }

        self.ensure_pipeline(device, binding_backend, texture_array_size);

        let (camera_bgl, material_bgl, constants_bgl) = (
            self.camera_bgl.read(),
            self.material_bgl.read(),
            self.constants_bgl.read(),
        );
        let (camera_bgl, material_bgl, constants_bgl) = (
            camera_bgl.as_ref().unwrap(),
            material_bgl.as_ref().unwrap(),
            constants_bgl.as_ref().unwrap(),
        );

        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Forward/MeshBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<MeshDrawParams>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::MESH,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward/MeshPipelineLayout"),
            bind_group_layouts: &[camera_bgl, material_bgl, constants_bgl, &mesh_bgl],
            immediate_size: 0,
        });

        let shader = if binding_backend == BindingBackendKind::BindGroups {
            device.create_shader_module(wgpu::include_wgsl!("forward_mesh_bindgroups.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("forward_mesh.wgsl"))
        };

        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("Forward/MeshPipeline"),
            layout: Some(&layout),
            task: None,
            mesh: wgpu::MeshState {
                module: &shader,
                entry_point: Some("ms_main"),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        *self.mesh_pipeline.write() = Some(pipeline);
        *self.mesh_bgl.write() = Some(mesh_bgl);
    }

    fn create_camera_bind_group(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<wgpu::BindGroup> {
        let camera_layout = self.camera_bgl.read();
        let camera_layout = camera_layout.as_ref()?;
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Forward/CameraBG"),
            layout: camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.skin_palette_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    fn create_constants_bind_group(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<wgpu::BindGroup> {
        let constants_layout = self.constants_bgl.read();
        let constants_layout = constants_layout.as_ref()?;
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Forward/ConstantsBG"),
            layout: constants_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.render_constants_buffer.as_entire_binding(),
            }],
        }))
    }

    fn create_material_bind_group_bindless(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<wgpu::BindGroup> {
        let material_layout = self.material_bgl.read();
        let material_layout = material_layout.as_ref()?;
        let array_size = self.texture_array_size.load(Ordering::Relaxed);
        let mut texture_views: Vec<&wgpu::TextureView> = Vec::with_capacity(array_size as usize);
        for view in frame.texture_views.iter() {
            texture_views.push(view);
        }
        while texture_views.len() < array_size as usize {
            texture_views.push(&frame.fallback_view);
        }

        Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Forward/MaterialBG"),
                layout: material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame
                            .material_buffer
                            .as_ref()
                            .expect("material buffer missing")
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureViewArray(&texture_views),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                    },
                ],
            }),
        )
    }

    fn ensure_material_bind_groups(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
    ) -> Option<Arc<Vec<wgpu::BindGroup>>> {
        let material_buffer = frame.material_buffer.as_ref()?;
        let material_textures = frame.material_textures.as_ref()?;
        let material_layout = self.material_bgl.read();
        let material_layout = material_layout.as_ref()?;
        let version = frame.material_bindings_version;
        if let Some(cache) = self.material_bind_groups.read().as_ref() {
            if cache.version == version && cache.groups.len() == material_textures.len() {
                return Some(cache.groups.clone());
            }
        }

        let mut groups = Vec::with_capacity(material_textures.len());
        for set in material_textures.iter() {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Forward/MaterialBG"),
                layout: material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: material_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&set.albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&set.normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&set.metallic_roughness),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&set.emission),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&frame.pbr_sampler),
                    },
                ],
            });
            groups.push(group);
        }
        let groups = Arc::new(groups);
        *self.material_bind_groups.write() = Some(MaterialBindGroupCache {
            version,
            groups: groups.clone(),
        });
        Some(groups)
    }
}

impl RenderPass for ForwardPass {
    fn name(&self) -> &'static str {
        "ForwardPass"
    }

    fn setup(&self, ctx: &mut RenderGraphContext) {
        ctx.read(self.depth);
        ctx.write(self.outputs.color);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        let binding_backend = frame.binding_backend;
        let use_mesh =
            frame.render_config.use_mesh_shaders && frame.device_caps.supports_mesh_pipeline();
        let use_indirect = frame.render_config.gpu_driven
            && frame.gbuffer_indirect.is_some()
            && !frame.gpu_draws.is_empty();
        let use_multi_draw = use_indirect
            && frame.render_config.gpu_multi_draw_indirect
            && frame.device_caps.supports_multi_draw_indirect()
            && binding_backend != BindingBackendKind::BindGroups;
        let has_materials = frame.material_buffer.is_some()
            && (binding_backend != BindingBackendKind::BindGroups
                || frame.material_textures.is_some());
        let draw_enabled = has_materials
            && frame.forward_instances.is_some()
            && (use_indirect || !frame.forward_batches.is_empty());
        if !draw_enabled {
            return;
        }

        if use_mesh {
            self.ensure_mesh_pipeline(ctx.device(), binding_backend, frame.texture_array_size);
        } else {
            self.ensure_pipeline(ctx.device(), binding_backend, frame.texture_array_size);
        }
        let depth_tex_size = ctx
            .rpctx
            .pool
            .entry(self.depth)
            .and_then(|e| e.texture.as_ref())
            .map(|t| t.size());
        let desired_extent = depth_tex_size
            .map(|s| (s.width, s.height))
            .unwrap_or(self.extent);
        let color_view = match self.ensure_target(ctx, desired_extent) {
            Some(v) => v,
            None => return,
        };
        let depth_view = match ctx.rpctx.pool.texture_view(self.depth) {
            Some(v) => v.clone(),
            None => return,
        };

        let camera_bg = match self.create_camera_bind_group(ctx.device(), frame.as_ref()) {
            Some(bg) => bg,
            None => return,
        };
        let constants_bg = match self.create_constants_bind_group(ctx.device(), frame.as_ref()) {
            Some(bg) => bg,
            None => return,
        };
        let material_bg = if binding_backend == BindingBackendKind::BindGroups {
            None
        } else {
            match self.create_material_bind_group_bindless(ctx.device(), frame.as_ref()) {
                Some(bg) => Some(bg),
                None => return,
            }
        };
        let material_groups = if binding_backend == BindingBackendKind::BindGroups {
            match self.ensure_material_bind_groups(ctx.device(), frame.as_ref()) {
                Some(groups) => Some(groups),
                None => return,
            }
        } else {
            None
        };

        let device = ctx.device().clone();
        let queue = ctx.queue().clone();

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderGraph/Forward"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
        pass.set_bind_group(0, &camera_bg, &[]);
        if binding_backend != BindingBackendKind::BindGroups {
            let material_bg = match material_bg.as_ref() {
                Some(bg) => bg,
                None => return,
            };
            pass.set_bind_group(1, material_bg, &[]);
        }
        pass.set_bind_group(2, &constants_bg, &[]);

        if use_mesh {
            let instances = match frame.forward_instances.as_ref() {
                Some(buf) => buf,
                None => return,
            };
            let mesh_pipeline = self.mesh_pipeline.read();
            let mesh_pipeline = match mesh_pipeline.as_ref() {
                Some(pipeline) => pipeline,
                None => return,
            };
            let mesh_bgl = self.mesh_bgl.read();
            let mesh_bgl = match mesh_bgl.as_ref() {
                Some(layout) => layout,
                None => return,
            };

            let params_size = std::mem::size_of::<MeshDrawParams>() as u64;
            let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
            let params_stride = wgpu::util::align_to(params_size, alignment);

            let mut direct_tilings: Vec<MeshTaskTiling> = Vec::new();
            let total_tiles = if use_indirect {
                frame
                    .gpu_draws
                    .iter()
                    .map(|draw| draw.mesh_task_count)
                    .sum()
            } else {
                let limits = &frame.device_caps.limits;
                direct_tilings.reserve(frame.forward_batches.len());
                let mut total = 0u32;
                for batch in frame.forward_batches.iter() {
                    let instance_count = batch
                        .instance_range
                        .end
                        .saturating_sub(batch.instance_range.start);
                    let tiling = mesh_task_tiling(limits, batch.meshlet_count, instance_count);
                    total = total.saturating_add(tiling.task_count);
                    direct_tilings.push(tiling);
                }
                total
            };

            if total_tiles == 0 {
                return;
            }

            let needed = params_stride * total_tiles as u64;
            let params_buffer = {
                let mut buffer_guard = self.mesh_params.write();
                let capacity = self.mesh_params_capacity.load(Ordering::Relaxed);
                if buffer_guard.is_none() || capacity < needed {
                    let new_capacity = needed.max(params_stride).next_power_of_two();
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Forward/MeshDrawParams"),
                        size: new_capacity,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    *buffer_guard = Some(buffer);
                    self.mesh_params_capacity
                        .store(new_capacity, Ordering::Relaxed);
                }
                buffer_guard.as_ref().unwrap().clone()
            };

            let occlusion_enabled = frame.render_config.occlusion_culling
                && frame.occlusion_camera_stable
                && frame.hiz_view.is_some();
            let flags =
                (frame.render_config.frustum_culling as u32) | ((occlusion_enabled as u32) << 1);
            let depth_bias = frame.render_config.gpu_cull_depth_bias.max(0.0);
            let rect_pad = frame.render_config.gpu_cull_rect_pad.max(0.0);

            let mut params_bytes = vec![0u8; needed as usize];
            let mut tile_cursor = 0u32;
            if use_indirect {
                for draw in frame.gpu_draws.iter() {
                    if draw.mesh_task_count == 0 {
                        continue;
                    }
                    let tile_meshlets = draw.mesh_task_tile_meshlets;
                    let tile_instances = draw.mesh_task_tile_instances;
                    if tile_meshlets == 0 || tile_instances == 0 {
                        continue;
                    }
                    let tiles_x = (draw.meshlet_count + tile_meshlets - 1) / tile_meshlets;
                    let tiles_y = (draw.instance_capacity + tile_instances - 1) / tile_instances;

                    for tile_y in 0..tiles_y {
                        let instance_base = draw.instance_base + tile_y * tile_instances;
                        let instance_count = draw
                            .instance_capacity
                            .saturating_sub(tile_y * tile_instances)
                            .min(tile_instances);
                        for tile_x in 0..tiles_x {
                            let meshlet_base = tile_x * tile_meshlets;
                            let meshlet_count = draw
                                .meshlet_count
                                .saturating_sub(meshlet_base)
                                .min(tile_meshlets);
                            let params = MeshDrawParams {
                                instance_base,
                                instance_count,
                                meshlet_base,
                                meshlet_count,
                                flags,
                                _pad0: 0,
                                _pad1: 0,
                                _pad2: 0,
                                depth_bias,
                                rect_pad,
                                _pad3: [0.0; 2],
                            };
                            let offset = (tile_cursor as u64) * params_stride;
                            let start = offset as usize;
                            params_bytes[start..start + params_size as usize]
                                .copy_from_slice(bytemuck::bytes_of(&params));
                            tile_cursor = tile_cursor.saturating_add(1);
                        }
                    }
                }
            } else {
                for (idx, batch) in frame.forward_batches.iter().enumerate() {
                    let tiling = direct_tilings[idx];
                    if tiling.task_count == 0 {
                        continue;
                    }
                    for tile_y in 0..tiling.tiles_y {
                        let instance_base =
                            batch.instance_range.start + tile_y * tiling.tile_instances;
                        let instance_count = batch
                            .instance_range
                            .end
                            .saturating_sub(
                                batch.instance_range.start + tile_y * tiling.tile_instances,
                            )
                            .min(tiling.tile_instances);
                        for tile_x in 0..tiling.tiles_x {
                            let meshlet_base = tile_x * tiling.tile_meshlets;
                            let meshlet_count = batch
                                .meshlet_count
                                .saturating_sub(meshlet_base)
                                .min(tiling.tile_meshlets);
                            let params = MeshDrawParams {
                                instance_base,
                                instance_count,
                                meshlet_base,
                                meshlet_count,
                                flags,
                                _pad0: 0,
                                _pad1: 0,
                                _pad2: 0,
                                depth_bias,
                                rect_pad,
                                _pad3: [0.0; 2],
                            };
                            let offset = (tile_cursor as u64) * params_stride;
                            let start = offset as usize;
                            params_bytes[start..start + params_size as usize]
                                .copy_from_slice(bytemuck::bytes_of(&params));
                            tile_cursor = tile_cursor.saturating_add(1);
                        }
                    }
                }
            }
            queue.write_buffer(&params_buffer, 0, &params_bytes);

            pass.set_pipeline(mesh_pipeline);
            pass.set_bind_group(0, &camera_bg, &[]);
            if binding_backend != BindingBackendKind::BindGroups {
                let material_bg = match material_bg.as_ref() {
                    Some(bg) => bg,
                    None => return,
                };
                pass.set_bind_group(1, material_bg, &[]);
            }
            pass.set_bind_group(2, &constants_bg, &[]);

            let hiz_view = frame.hiz_view.as_ref().unwrap_or(&frame.fallback_view);
            let task_stride = std::mem::size_of::<wgpu::util::DispatchIndirectArgs>() as u64;
            let mut params_tile_index = 0u32;
            if use_indirect {
                let mesh_tasks = match frame.gbuffer_mesh_tasks.as_ref() {
                    Some(buf) => buf,
                    None => return,
                };
                let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                    match material_groups.as_ref() {
                        Some(groups) => Some(groups),
                        None => return,
                    }
                } else {
                    None
                };
                for draw in frame.gpu_draws.iter() {
                    let draw_tiles = draw.mesh_task_count;
                    if draw_tiles == 0 {
                        continue;
                    }
                    let tile_meshlets = draw.mesh_task_tile_meshlets;
                    let tile_instances = draw.mesh_task_tile_instances;
                    if tile_meshlets == 0 || tile_instances == 0 {
                        params_tile_index = params_tile_index.saturating_add(draw_tiles);
                        continue;
                    }
                    let tiles_x = (draw.meshlet_count + tile_meshlets - 1) / tile_meshlets;
                    let tiles_y = (draw.instance_capacity + tile_instances - 1) / tile_instances;
                    if tiles_x == 0 || tiles_y == 0 {
                        params_tile_index = params_tile_index.saturating_add(draw_tiles);
                        continue;
                    }
                    let meshlet_descs = match ctx
                        .rpctx
                        .pool
                        .entry(draw.meshlet_descs)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                    };
                    let meshlet_vertices = match ctx
                        .rpctx
                        .pool
                        .entry(draw.meshlet_vertices)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                    };
                    let meshlet_indices = match ctx
                        .rpctx
                        .pool
                        .entry(draw.meshlet_indices)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                    };
                    let vertex_data = match ctx
                        .rpctx
                        .pool
                        .entry(draw.vertex)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(draw_tiles);
                            continue;
                        }
                    };

                    let mesh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Forward/MeshBG"),
                        layout: mesh_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: instances.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: meshlet_descs.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meshlet_vertices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: meshlet_indices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: vertex_data.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &params_buffer,
                                    offset: 0,
                                    size: wgpu::BufferSize::new(params_size),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(hiz_view),
                            },
                        ],
                    });

                    if let Some(groups) = material_groups.as_ref() {
                        let material_idx = draw.material_id as usize;
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        pass.set_bind_group(1, material_bg, &[]);
                    }
                    let mut draw_tile_index = 0u32;
                    for _tile_y in 0..tiles_y {
                        for _tile_x in 0..tiles_x {
                            let params_offset = (params_tile_index as u64) * params_stride;
                            if params_offset > u32::MAX as u64 {
                                return;
                            }
                            let task_offset =
                                draw.mesh_task_offset + (draw_tile_index as u64) * task_stride;
                            pass.set_bind_group(3, &mesh_bg, &[params_offset as u32]);
                            pass.draw_mesh_tasks_indirect(mesh_tasks, task_offset);
                            params_tile_index = params_tile_index.saturating_add(1);
                            draw_tile_index = draw_tile_index.saturating_add(1);
                        }
                    }
                }
            } else {
                for (idx, batch) in frame.forward_batches.iter().enumerate() {
                    let tiling = direct_tilings[idx];
                    if tiling.task_count == 0 {
                        continue;
                    }
                    if tiling.tiles_x == 0 || tiling.tiles_y == 0 {
                        params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                        continue;
                    }
                    let meshlet_descs = match ctx
                        .rpctx
                        .pool
                        .entry(batch.meshlet_descs)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                    };
                    let meshlet_vertices = match ctx
                        .rpctx
                        .pool
                        .entry(batch.meshlet_vertices)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                    };
                    let meshlet_indices = match ctx
                        .rpctx
                        .pool
                        .entry(batch.meshlet_indices)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                    };
                    let vertex_data = match ctx
                        .rpctx
                        .pool
                        .entry(batch.vertex)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => {
                            params_tile_index = params_tile_index.saturating_add(tiling.task_count);
                            continue;
                        }
                    };

                    let mesh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Forward/MeshBG"),
                        layout: mesh_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: instances.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: meshlet_descs.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meshlet_vertices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: meshlet_indices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: vertex_data.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &params_buffer,
                                    offset: 0,
                                    size: wgpu::BufferSize::new(params_size),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(hiz_view),
                            },
                        ],
                    });

                    if let Some(groups) = material_groups.as_ref() {
                        let material_idx = batch.material_id as usize;
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        pass.set_bind_group(1, material_bg, &[]);
                    }
                    for tile_y in 0..tiling.tiles_y {
                        for tile_x in 0..tiling.tiles_x {
                            let params_offset = (params_tile_index as u64) * params_stride;
                            if params_offset > u32::MAX as u64 {
                                return;
                            }
                            pass.set_bind_group(3, &mesh_bg, &[params_offset as u32]);
                            let meshlet_base = tile_x * tiling.tile_meshlets;
                            let meshlet_count = batch
                                .meshlet_count
                                .saturating_sub(meshlet_base)
                                .min(tiling.tile_meshlets);
                            let instance_base =
                                batch.instance_range.start + tile_y * tiling.tile_instances;
                            let instance_count = batch
                                .instance_range
                                .end
                                .saturating_sub(instance_base)
                                .min(tiling.tile_instances);
                            pass.draw_mesh_tasks(meshlet_count, instance_count, 1);
                            params_tile_index = params_tile_index.saturating_add(1);
                        }
                    }
                }
            }
        } else {
            pass.set_pipeline(self.pipeline.read().as_ref().unwrap());
            pass.set_bind_group(0, &camera_bg, &[]);
            if binding_backend != BindingBackendKind::BindGroups {
                let material_bg = match material_bg.as_ref() {
                    Some(bg) => bg,
                    None => return,
                };
                pass.set_bind_group(1, material_bg, &[]);
            }
            pass.set_bind_group(2, &constants_bg, &[]);

            pass.set_vertex_buffer(
                1,
                frame.forward_instances.as_ref().unwrap().buffer.slice(..),
            );

            if use_indirect {
                let indirect = frame.gbuffer_indirect.as_ref().unwrap();
                if use_multi_draw {
                    let stride = std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;
                    let draws = frame.gpu_draws.as_ref();
                    let mut i = 0usize;
                    while i < draws.len() {
                        let draw = &draws[i];
                        let vertex = match ctx
                            .rpctx
                            .pool
                            .entry(draw.vertex)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => {
                                i += 1;
                                continue;
                            }
                        };
                        let index = match ctx
                            .rpctx
                            .pool
                            .entry(draw.index)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => {
                                i += 1;
                                continue;
                            }
                        };
                        pass.set_vertex_buffer(0, vertex.slice(..));
                        pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);

                        let mut count = 1u32;
                        let mut next_offset = draw.indirect_offset + stride;
                        let mut j = i + 1;
                        while j < draws.len() {
                            let next = &draws[j];
                            if next.vertex != draw.vertex
                                || next.index != draw.index
                                || next.indirect_offset != next_offset
                            {
                                break;
                            }
                            count += 1;
                            next_offset += stride;
                            j += 1;
                        }
                        pass.multi_draw_indexed_indirect(indirect, draw.indirect_offset, count);
                        i = j;
                    }
                } else {
                    let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                        match material_groups.as_ref() {
                            Some(groups) => Some(groups),
                            None => return,
                        }
                    } else {
                        None
                    };
                    for draw in frame.gpu_draws.iter() {
                        let vertex = match ctx
                            .rpctx
                            .pool
                            .entry(draw.vertex)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => continue,
                        };
                        let index = match ctx
                            .rpctx
                            .pool
                            .entry(draw.index)
                            .and_then(|e| e.buffer.as_ref())
                        {
                            Some(buf) => buf,
                            None => continue,
                        };
                        if let Some(groups) = material_groups.as_ref() {
                            let material_idx = draw.material_id as usize;
                            let material_bg = groups
                                .get(material_idx)
                                .or_else(|| groups.first())
                                .expect("material bind groups empty");
                            pass.set_bind_group(1, material_bg, &[]);
                        }
                        pass.set_vertex_buffer(0, vertex.slice(..));
                        pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed_indirect(indirect, draw.indirect_offset);
                    }
                }
            } else {
                let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                    match material_groups.as_ref() {
                        Some(groups) => Some(groups),
                        None => return,
                    }
                } else {
                    None
                };
                for batch in frame.forward_batches.iter() {
                    let vertex = match ctx
                        .rpctx
                        .pool
                        .entry(batch.vertex)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => continue,
                    };
                    let index = match ctx
                        .rpctx
                        .pool
                        .entry(batch.index)
                        .and_then(|e| e.buffer.as_ref())
                    {
                        Some(buf) => buf,
                        None => continue,
                    };

                    if let Some(groups) = material_groups.as_ref() {
                        let material_idx = batch.material_id as usize;
                        let material_bg = groups
                            .get(material_idx)
                            .or_else(|| groups.first())
                            .expect("material bind groups empty");
                        pass.set_bind_group(1, material_bg, &[]);
                    }
                    pass.set_vertex_buffer(0, vertex.slice(..));
                    pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..batch.index_count, 0, batch.instance_range.clone());
                }
            }
        }
    }
}
