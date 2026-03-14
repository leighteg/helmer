use parking_lot::RwLock;
use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

use super::ForwardBlendMode;
use crate::graphics::{
    backend::binding_backend::BindingBackendKind,
    common::renderer::{
        GBufferInstanceRaw, LightData, MeshDrawParams, MeshTaskTiling, ShaderConstants,
        ShadowUniforms, SkyUniforms, Vertex, mesh_shader_visibility, mesh_task_tiling,
    },
    graph::{
        definition::{render_pass::RenderPass, resource_id::ResourceId},
        logic::{
            gpu_resource_pool::GpuResourcePool, graph_context::RenderGraphContext,
            graph_exec_ctx::RenderGraphExecCtx,
        },
    },
    passes::{FrameGlobals, SwapchainFrameInput},
};

struct MaterialBindGroupCache {
    version: u64,
    groups: Arc<Vec<wgpu::BindGroup>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ForwardOutputs {
    pub swapchain: ResourceId,
}

#[derive(Clone)]
pub struct ForwardPass {
    outputs: ForwardOutputs,
    pipeline_alpha: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    pipeline_premultiplied: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    pipeline_additive: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline_alpha: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline_premultiplied: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    mesh_pipeline_additive: Arc<RwLock<Option<wgpu::RenderPipeline>>>,
    camera_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    scene_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    material_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    mesh_bgl: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
    mesh_params: Arc<RwLock<Option<wgpu::Buffer>>>,
    mesh_params_capacity: Arc<AtomicU64>,
    texture_array_size: Arc<AtomicU32>,
    backend_kind: Arc<RwLock<Option<BindingBackendKind>>>,
    material_bind_groups: Arc<RwLock<Option<MaterialBindGroupCache>>>,
    format: Arc<RwLock<Option<wgpu::TextureFormat>>>,
    depth: ResourceId,
    shadow_map: ResourceId,
    use_uniform_lights: bool,
    max_uniform_lights: usize,
}

impl ForwardPass {
    pub fn new(
        pool: &mut GpuResourcePool,
        swapchain: ResourceId,
        depth: ResourceId,
        shadow_map: ResourceId,
        surface_format: wgpu::TextureFormat,
        supports_fragment_storage_buffers: bool,
    ) -> Self {
        let _ = pool.entry(swapchain);
        let use_uniform_lights = !supports_fragment_storage_buffers;
        let max_uniform_lights = crate::graphics::common::constants::WEBGL_MAX_LIGHTS;

        Self {
            outputs: ForwardOutputs { swapchain },
            pipeline_alpha: Arc::new(RwLock::new(None)),
            pipeline_premultiplied: Arc::new(RwLock::new(None)),
            pipeline_additive: Arc::new(RwLock::new(None)),
            mesh_pipeline_alpha: Arc::new(RwLock::new(None)),
            mesh_pipeline_premultiplied: Arc::new(RwLock::new(None)),
            mesh_pipeline_additive: Arc::new(RwLock::new(None)),
            camera_bgl: Arc::new(RwLock::new(None)),
            scene_bgl: Arc::new(RwLock::new(None)),
            material_bgl: Arc::new(RwLock::new(None)),
            mesh_bgl: Arc::new(RwLock::new(None)),
            mesh_params: Arc::new(RwLock::new(None)),
            mesh_params_capacity: Arc::new(AtomicU64::new(0)),
            texture_array_size: Arc::new(AtomicU32::new(1)),
            backend_kind: Arc::new(RwLock::new(None)),
            material_bind_groups: Arc::new(RwLock::new(None)),
            format: Arc::new(RwLock::new(Some(surface_format))),
            depth,
            shadow_map,
            use_uniform_lights,
            max_uniform_lights,
        }
    }

    pub fn outputs(&self) -> ForwardOutputs {
        self.outputs
    }

    fn ensure_backend(&self, binding_backend: BindingBackendKind) {
        let mut current = self.backend_kind.write();
        if current.map_or(true, |kind| kind != binding_backend) {
            *self.pipeline_alpha.write() = None;
            *self.pipeline_premultiplied.write() = None;
            *self.pipeline_additive.write() = None;
            *self.mesh_pipeline_alpha.write() = None;
            *self.mesh_pipeline_premultiplied.write() = None;
            *self.mesh_pipeline_additive.write() = None;
            *self.camera_bgl.write() = None;
            *self.scene_bgl.write() = None;
            *self.material_bgl.write() = None;
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
        format: wgpu::TextureFormat,
    ) {
        self.ensure_backend(binding_backend);

        let format_changed = self.format.read().map_or(true, |current| current != format);
        if format_changed {
            *self.pipeline_alpha.write() = None;
            *self.pipeline_premultiplied.write() = None;
            *self.pipeline_additive.write() = None;
            *self.mesh_pipeline_alpha.write() = None;
            *self.mesh_pipeline_premultiplied.write() = None;
            *self.mesh_pipeline_additive.write() = None;
            *self.format.write() = Some(format);
        }

        let array_size = texture_array_size.max(1);
        if binding_backend != BindingBackendKind::BindGroups {
            let current_size = self.texture_array_size.load(Ordering::Relaxed);
            if current_size != array_size {
                *self.pipeline_alpha.write() = None;
                *self.pipeline_premultiplied.write() = None;
                *self.pipeline_additive.write() = None;
                *self.mesh_pipeline_alpha.write() = None;
                *self.mesh_pipeline_premultiplied.write() = None;
                *self.mesh_pipeline_additive.write() = None;
                *self.material_bgl.write() = None;
                *self.material_bind_groups.write() = None;
                self.texture_array_size.store(array_size, Ordering::Relaxed);
            }
        } else {
            self.texture_array_size.store(1, Ordering::Relaxed);
        }

        if self.pipeline_alpha.read().is_some()
            && self.pipeline_premultiplied.read().is_some()
            && self.pipeline_additive.read().is_some()
        {
            return;
        }

        let mesh_stage = mesh_shader_visibility(device);
        let camera_bgl = if let Some(layout) = self.camera_bgl.read().clone() {
            layout
        } else {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            *self.camera_bgl.write() = Some(layout.clone());
            layout
        };

        let lights_binding = if self.use_uniform_lights {
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        (std::mem::size_of::<LightData>() * self.max_uniform_lights) as u64,
                    ),
                },
                count: None,
            }
        } else {
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };

        let scene_bgl = if let Some(layout) = self.scene_bgl.read().clone() {
            layout
        } else {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Forward/SceneBGL"),
                entries: &[
                    lights_binding,
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                ShadowUniforms,
                            >()
                                as u64),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<SkyUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                ShaderConstants,
                            >()
                                as u64),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            *self.scene_bgl.write() = Some(layout.clone());
            layout
        };

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
        let material_bgl = if let Some(layout) = self.material_bgl.read().clone() {
            layout
        } else {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Forward/MaterialBGL"),
                entries: &material_entries,
            });
            *self.material_bgl.write() = Some(layout.clone());
            layout
        };

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward/PipelineLayout"),
            bind_group_layouts: &[&camera_bgl, &scene_bgl, &material_bgl],
            immediate_size: 0,
        });

        let shader = if self.use_uniform_lights {
            if binding_backend == BindingBackendKind::BindGroups {
                device.create_shader_module(wgpu::include_wgsl!("forward_webgl_bindgroups.wgsl"))
            } else {
                device.create_shader_module(wgpu::include_wgsl!("forward_webgl.wgsl"))
            }
        } else if binding_backend == BindingBackendKind::BindGroups {
            device.create_shader_module(wgpu::include_wgsl!("forward_bindgroups.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("forward.wgsl"))
        };

        let blend_state = |mode: ForwardBlendMode| -> wgpu::BlendState {
            match mode {
                ForwardBlendMode::Alpha => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
                ForwardBlendMode::Premultiplied => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
                ForwardBlendMode::Additive => wgpu::BlendState {
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
                },
            }
        };

        let create_pipeline = |mode: ForwardBlendMode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        format,
                        blend: Some(blend_state(mode)),
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
            })
        };

        *self.pipeline_alpha.write() = Some(create_pipeline(ForwardBlendMode::Alpha));
        *self.pipeline_premultiplied.write() =
            Some(create_pipeline(ForwardBlendMode::Premultiplied));
        *self.pipeline_additive.write() = Some(create_pipeline(ForwardBlendMode::Additive));
    }

    fn ensure_mesh_pipeline(
        &self,
        device: &wgpu::Device,
        binding_backend: BindingBackendKind,
        texture_array_size: u32,
        format: wgpu::TextureFormat,
    ) {
        if self.use_uniform_lights || mesh_shader_visibility(device).is_empty() {
            return;
        }
        self.ensure_pipeline(device, binding_backend, texture_array_size, format);

        if self.mesh_pipeline_alpha.read().is_some()
            && self.mesh_pipeline_premultiplied.read().is_some()
            && self.mesh_pipeline_additive.read().is_some()
        {
            return;
        }

        let (camera_bgl, scene_bgl, material_bgl) = (
            self.camera_bgl.read(),
            self.scene_bgl.read(),
            self.material_bgl.read(),
        );
        let (camera_bgl, scene_bgl, material_bgl) = (
            camera_bgl.as_ref().unwrap(),
            scene_bgl.as_ref().unwrap(),
            material_bgl.as_ref().unwrap(),
        );

        let mesh_bgl = if let Some(layout) = self.mesh_bgl.read().clone() {
            layout
        } else {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                MeshDrawParams,
                            >()
                                as u64),
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
            *self.mesh_bgl.write() = Some(layout.clone());
            layout
        };

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward/MeshPipelineLayout"),
            bind_group_layouts: &[camera_bgl, scene_bgl, material_bgl, &mesh_bgl],
            immediate_size: 0,
        });

        let shader = if binding_backend == BindingBackendKind::BindGroups {
            device.create_shader_module(wgpu::include_wgsl!("forward_mesh_bindgroups.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("forward_mesh.wgsl"))
        };

        let blend_state = |mode: ForwardBlendMode| -> wgpu::BlendState {
            match mode {
                ForwardBlendMode::Alpha => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
                ForwardBlendMode::Premultiplied => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
                ForwardBlendMode::Additive => wgpu::BlendState {
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
                },
            }
        };

        let create_pipeline = |mode: ForwardBlendMode| {
            device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
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
                        format,
                        blend: Some(blend_state(mode)),
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
            })
        };

        *self.mesh_pipeline_alpha.write() = Some(create_pipeline(ForwardBlendMode::Alpha));
        *self.mesh_pipeline_premultiplied.write() =
            Some(create_pipeline(ForwardBlendMode::Premultiplied));
        *self.mesh_pipeline_additive.write() = Some(create_pipeline(ForwardBlendMode::Additive));
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

    fn create_scene_bind_group(
        &self,
        device: &wgpu::Device,
        frame: &FrameGlobals,
        shadow_view: &wgpu::TextureView,
    ) -> Option<wgpu::BindGroup> {
        let scene_layout = self.scene_bgl.read();
        let scene_layout = scene_layout.as_ref()?;
        let lights_buffer = frame.lights_buffer.clone().unwrap_or_else(|| {
            let size = if self.use_uniform_lights {
                (std::mem::size_of::<LightData>() * self.max_uniform_lights) as u64
            } else {
                std::mem::size_of::<LightData>() as u64
            };
            let usage = if self.use_uniform_lights {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE
            };
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Forward/EmptyLights"),
                size,
                usage,
                mapped_at_creation: false,
            })
        });
        let shadow_uniforms = frame.shadow_uniforms_buffer.as_ref()?;
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Forward/SceneBG"),
            layout: scene_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&frame.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: shadow_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: frame.sky_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: frame.render_constants_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&frame.ibl_brdf_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&frame.ibl_irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&frame.ibl_prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&frame.ibl_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&frame.brdf_lut_sampler),
                },
            ],
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
        ctx.read(self.shadow_map);
        ctx.rw(self.outputs.swapchain);
    }

    fn execute(&self, ctx: &mut RenderGraphExecCtx) {
        let frame = match ctx.rpctx.frame_inputs.get::<FrameGlobals>() {
            Some(f) => f,
            None => return,
        };
        if !frame.render_config.transparent_pass {
            return;
        }
        let binding_backend = frame.binding_backend;
        let mut use_mesh =
            frame.render_config.use_mesh_shaders && frame.device_caps.supports_mesh_pipeline();
        let has_materials = frame.material_buffer.is_some()
            && (binding_backend != BindingBackendKind::BindGroups
                || frame.material_textures.is_some());
        let draw_enabled = frame.device_caps.supports_vertex_storage()
            && frame.device_caps.supports_fragment_storage_buffers()
            && has_materials
            && frame.transparent_instances.is_some()
            && !frame.transparent_batches.is_empty();
        if !draw_enabled {
            return;
        }

        let swapchain = match ctx.rpctx.frame_inputs.get::<SwapchainFrameInput>() {
            Some(v) => v,
            None => return,
        };
        let format = swapchain.format;
        use_mesh = use_mesh && !self.use_uniform_lights;
        if use_mesh
            && frame
                .transparent_batches
                .iter()
                .any(|batch| batch.batch.meshlet_count == 0)
        {
            use_mesh = false;
        }

        if use_mesh {
            self.ensure_mesh_pipeline(
                ctx.device(),
                binding_backend,
                frame.texture_array_size,
                format,
            );
        } else {
            self.ensure_pipeline(
                ctx.device(),
                binding_backend,
                frame.texture_array_size,
                format,
            );
        }

        let depth_view = match ctx.rpctx.pool.texture_view(self.depth) {
            Some(v) => v.clone(),
            None => return,
        };
        let shadow_view = match ctx.rpctx.pool.texture_view(self.shadow_map) {
            Some(v) => v.clone(),
            None => return,
        };

        let camera_bg = match self.create_camera_bind_group(ctx.device(), frame.as_ref()) {
            Some(bg) => bg,
            None => return,
        };
        let scene_bg =
            match self.create_scene_bind_group(ctx.device(), frame.as_ref(), &shadow_view) {
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
                view: &swapchain.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
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

        if use_mesh {
            let instances = match frame.transparent_instances.as_ref() {
                Some(buf) => buf,
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

            let limits = &frame.device_caps.limits;
            let mut direct_tilings: Vec<MeshTaskTiling> =
                Vec::with_capacity(frame.transparent_batches.len());
            let mut total_tiles = 0u32;
            for batch in frame.transparent_batches.iter() {
                let batch = &batch.batch;
                let instance_count = batch
                    .instance_range
                    .end
                    .saturating_sub(batch.instance_range.start);
                let tiling = mesh_task_tiling(limits, batch.meshlet_count, instance_count);
                total_tiles = total_tiles.saturating_add(tiling.task_count);
                direct_tilings.push(tiling);
            }

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
            for (idx, batch) in frame.transparent_batches.iter().enumerate() {
                let batch = &batch.batch;
                let tiling = direct_tilings[idx];
                if tiling.task_count == 0 {
                    continue;
                }
                for tile_y in 0..tiling.tiles_y {
                    let instance_base = batch.instance_range.start + tile_y * tiling.tile_instances;
                    let instance_count = batch
                        .instance_range
                        .end
                        .saturating_sub(batch.instance_range.start + tile_y * tiling.tile_instances)
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
            queue.write_buffer(&params_buffer, 0, &params_bytes);

            let hiz_view = frame.hiz_view.as_ref().unwrap_or(&frame.fallback_view);
            let mut params_tile_index = 0u32;
            let mut current_blend: Option<ForwardBlendMode> = None;
            let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                match material_groups.as_ref() {
                    Some(groups) => Some(groups),
                    None => return,
                }
            } else {
                None
            };

            for (idx, batch) in frame.transparent_batches.iter().enumerate() {
                let blend_mode = batch.blend_mode;
                if current_blend != Some(blend_mode) {
                    let pipeline = match blend_mode {
                        ForwardBlendMode::Alpha => self.mesh_pipeline_alpha.read(),
                        ForwardBlendMode::Premultiplied => self.mesh_pipeline_premultiplied.read(),
                        ForwardBlendMode::Additive => self.mesh_pipeline_additive.read(),
                    };
                    let pipeline = match pipeline.as_ref() {
                        Some(p) => p,
                        None => return,
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &camera_bg, &[]);
                    pass.set_bind_group(1, &scene_bg, &[]);
                    if binding_backend != BindingBackendKind::BindGroups {
                        let material_bg = match material_bg.as_ref() {
                            Some(bg) => bg,
                            None => return,
                        };
                        pass.set_bind_group(2, material_bg, &[]);
                    }
                    current_blend = Some(blend_mode);
                }

                let batch = &batch.batch;
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
                    pass.set_bind_group(2, material_bg, &[]);
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
        } else {
            pass.set_vertex_buffer(
                1,
                frame
                    .transparent_instances
                    .as_ref()
                    .unwrap()
                    .buffer
                    .slice(..),
            );
            let material_groups = if binding_backend == BindingBackendKind::BindGroups {
                match material_groups.as_ref() {
                    Some(groups) => Some(groups),
                    None => return,
                }
            } else {
                None
            };
            let mut current_blend: Option<ForwardBlendMode> = None;
            for batch in frame.transparent_batches.iter() {
                let blend_mode = batch.blend_mode;
                if current_blend != Some(blend_mode) {
                    let pipeline = match blend_mode {
                        ForwardBlendMode::Alpha => self.pipeline_alpha.read(),
                        ForwardBlendMode::Premultiplied => self.pipeline_premultiplied.read(),
                        ForwardBlendMode::Additive => self.pipeline_additive.read(),
                    };
                    let pipeline = match pipeline.as_ref() {
                        Some(p) => p,
                        None => return,
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &camera_bg, &[]);
                    pass.set_bind_group(1, &scene_bg, &[]);
                    if binding_backend != BindingBackendKind::BindGroups {
                        let material_bg = match material_bg.as_ref() {
                            Some(bg) => bg,
                            None => return,
                        };
                        pass.set_bind_group(2, material_bg, &[]);
                    }
                    current_blend = Some(blend_mode);
                }

                let batch = &batch.batch;
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
                    pass.set_bind_group(2, material_bg, &[]);
                }
                pass.set_vertex_buffer(0, vertex.slice(..));
                pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..batch.index_count, 0, batch.instance_range.clone());
            }
        }
    }
}
