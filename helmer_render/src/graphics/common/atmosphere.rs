use crate::graphics::common::renderer::{ShaderConstants, SkyUniforms};

// --- Constants for LUTs ---
const TRANSMITTANCE_LUT_WIDTH: u32 = 256;
const TRANSMITTANCE_LUT_HEIGHT: u32 = 64;

const SCATTERING_LUT_WIDTH: u32 = 256;
const SCATTERING_LUT_HEIGHT: u32 = 128;
const SCATTERING_LUT_DEPTH: u32 = 32;

const IRRADIANCE_LUT_WIDTH: u32 = 64;
const IRRADIANCE_LUT_HEIGHT: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScatteringLutDimension {
    Volume3d,
    Array2d,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AtmosphereUniforms {
    planet_radius: f32,
    atmosphere_radius: f32,
    sun_intensity: f32,
    _padding: f32,
    sun_direction: [f32; 3],
    _padding2: f32,
    rayleigh_scattering_coeff: [f32; 3],
    rayleigh_scale_height: f32,
    mie_scattering_coeff: f32,
    mie_absorption_coeff: f32,
    mie_scale_height: f32,
    mie_preferred_scattering_dir: f32,
    ozone_absorption_coeff: [f32; 3],
    ozone_center_height: f32,
    ozone_falloff: f32,
    _pad_after_ozone: [f32; 3],
    _pad_atmo0: [f32; 3],
    _pad_after_atmo0: f32,
    ground_albedo: [f32; 3],
    ground_brightness: f32,
    night_ambient_color: [f32; 3],
    _pad_atmo1: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScatteringSliceUniform {
    slice: u32,
    _pad: [u32; 7],
}

pub struct AtmosphereLuts {
    pub transmittance_lut: wgpu::Texture,
    pub scattering_lut: wgpu::Texture,
    pub irradiance_lut: wgpu::Texture,
    pub transmittance_lut_view: wgpu::TextureView,
    pub scattering_lut_view: wgpu::TextureView,
    pub irradiance_lut_view: wgpu::TextureView,
}

impl AtmosphereLuts {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        allow_storage: bool,
        allow_render: bool,
        scattering_dimension: ScatteringLutDimension,
    ) -> Self {
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if allow_storage {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        if allow_render {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        let transmittance_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Transmittance LUT"),
            size: wgpu::Extent3d {
                width: TRANSMITTANCE_LUT_WIDTH,
                height: TRANSMITTANCE_LUT_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });

        let (scattering_label, scattering_dimension_desc, scattering_view_dimension) =
            match scattering_dimension {
                ScatteringLutDimension::Volume3d => (
                    "Scattering LUT (3D)",
                    wgpu::TextureDimension::D3,
                    wgpu::TextureViewDimension::D3,
                ),
                ScatteringLutDimension::Array2d => (
                    "Scattering LUT (2D Array)",
                    wgpu::TextureDimension::D2,
                    wgpu::TextureViewDimension::D2Array,
                ),
            };

        let scattering_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(scattering_label),
            size: wgpu::Extent3d {
                width: SCATTERING_LUT_WIDTH,
                height: SCATTERING_LUT_HEIGHT,
                depth_or_array_layers: SCATTERING_LUT_DEPTH,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: scattering_dimension_desc,
            format,
            usage,
            view_formats: &[],
        });

        let irradiance_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Irradiance LUT"),
            size: wgpu::Extent3d {
                width: IRRADIANCE_LUT_WIDTH,
                height: IRRADIANCE_LUT_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });

        let transmittance_lut_view = transmittance_lut.create_view(&Default::default());
        let scattering_lut_view = scattering_lut.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(scattering_view_dimension),
            ..Default::default()
        });
        let irradiance_lut_view = irradiance_lut.create_view(&Default::default());

        Self {
            transmittance_lut,
            scattering_lut,
            irradiance_lut,
            transmittance_lut_view,
            scattering_lut_view,
            irradiance_lut_view,
        }
    }
}

pub struct AtmospherePrecomputer {
    luts: AtmosphereLuts,
    atmosphere_uniform_buffer: wgpu::Buffer,

    pub sampling_bind_group_layout: wgpu::BindGroupLayout,
    pub sampling_bind_group: wgpu::BindGroup,

    // Separate bind groups for each precompute pass to avoid conflicts
    precompute: AtmospherePrecomputeMode,
    scattering_is_array: bool,
}

enum AtmospherePrecomputeMode {
    Compute(AtmospherePrecomputeCompute),
    Raster(AtmospherePrecomputeRaster),
}

struct AtmospherePrecomputeCompute {
    transmittance_bg: wgpu::BindGroup,
    scattering_bg: wgpu::BindGroup,
    irradiance_bg: wgpu::BindGroup,
    transmittance_pipeline: wgpu::ComputePipeline,
    scattering_pipeline: wgpu::ComputePipeline,
    irradiance_pipeline: wgpu::ComputePipeline,
}

struct AtmospherePrecomputeRaster {
    transmittance_bg: wgpu::BindGroup,
    scattering_bg: wgpu::BindGroup,
    irradiance_bg: wgpu::BindGroup,
    transmittance_pipeline: wgpu::RenderPipeline,
    scattering_pipeline: wgpu::RenderPipeline,
    irradiance_pipeline: wgpu::RenderPipeline,
    scattering_slice_buffer: wgpu::Buffer,
    scattering_slice_views: Vec<wgpu::TextureView>,
}

impl AtmospherePrecomputer {
    pub fn new(device: &wgpu::Device) -> Self {
        Self::new_with_options(device, true, wgpu::TextureFormat::Rgba16Float)
    }

    pub fn new_with_options(
        device: &wgpu::Device,
        supports_compute: bool,
        lut_format: wgpu::TextureFormat,
    ) -> Self {
        let scattering_dimension = if supports_compute {
            ScatteringLutDimension::Volume3d
        } else {
            ScatteringLutDimension::Array2d
        };
        let allow_render = !supports_compute;
        let luts = AtmosphereLuts::new(
            device,
            lut_format,
            supports_compute,
            allow_render,
            scattering_dimension,
        );

        let atmosphere_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atmosphere Uniform Buffer"),
            size: std::mem::size_of::<AtmosphereUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampling_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Atmosphere LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let scattering_view_dimension = match scattering_dimension {
            ScatteringLutDimension::Volume3d => wgpu::TextureViewDimension::D3,
            ScatteringLutDimension::Array2d => wgpu::TextureViewDimension::D2Array,
        };

        // --- Sampling Layout (Fragment Shaders) ---
        let sampling_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Atmosphere Sampling BGL"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: scattering_view_dimension,
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let sampling_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atmosphere Sampling BG"),
            layout: &sampling_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&luts.transmittance_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&luts.scattering_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&luts.irradiance_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampling_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: atmosphere_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let precompute = if supports_compute {
            // --- 1. Transmittance Pass Bind Group ---
            let transmittance_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Atmosphere Transmittance BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            // Uniforms
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            // Transmittance LUT (write)
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba16Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

            let transmittance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmosphere Transmittance BG"),
                layout: &transmittance_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atmosphere_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&luts.transmittance_lut_view),
                    },
                ],
            });

            // --- 2. Scattering Pass Bind Group ---
            let scattering_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Atmosphere Scattering BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            // Uniforms
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            // Transmittance LUT (read)
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            // Sampler
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            // Scattering LUT (write)
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba16Float,
                                view_dimension: wgpu::TextureViewDimension::D3,
                            },
                            count: None,
                        },
                    ],
                });

            let scattering_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmosphere Scattering BG"),
                layout: &scattering_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atmosphere_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&luts.transmittance_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampling_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&luts.scattering_lut_view),
                    },
                ],
            });

            // --- 3. Irradiance Pass Bind Group ---
            let irradiance_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Atmosphere Irradiance BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            // Uniforms
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            // Transmittance LUT (read)
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            // Sampler
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            // Irradiance LUT (write)
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba16Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

            let irradiance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmosphere Irradiance BG"),
                layout: &irradiance_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atmosphere_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&luts.transmittance_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampling_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&luts.irradiance_lut_view),
                    },
                ],
            });

            // --- Pipelines ---
            let precompute_shader = device
                .create_shader_module(wgpu::include_wgsl!("../shaders/atmosphere_precompute.wgsl"));

            let transmittance_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Transmittance Pipeline"),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Transmittance Layout"),
                            bind_group_layouts: &[&transmittance_bgl],
                            immediate_size: 0,
                        }),
                    ),
                    module: &precompute_shader,
                    entry_point: Some("compute_transmittance"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let scattering_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Scattering Pipeline"),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Scattering Layout"),
                            bind_group_layouts: &[&scattering_bgl],
                            immediate_size: 0,
                        }),
                    ),
                    module: &precompute_shader,
                    entry_point: Some("compute_scattering"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let irradiance_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Irradiance Pipeline"),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Irradiance Layout"),
                            bind_group_layouts: &[&irradiance_bgl],
                            immediate_size: 0,
                        }),
                    ),
                    module: &precompute_shader,
                    entry_point: Some("compute_irradiance"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            AtmospherePrecomputeMode::Compute(AtmospherePrecomputeCompute {
                transmittance_bg,
                scattering_bg,
                irradiance_bg,
                transmittance_pipeline,
                scattering_pipeline,
                irradiance_pipeline,
            })
        } else {
            let transmittance_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Atmosphere Transmittance Raster BGL"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            let transmittance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmosphere Transmittance Raster BG"),
                layout: &transmittance_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: atmosphere_uniform_buffer.as_entire_binding(),
                }],
            });

            let scattering_slice_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Atmosphere Scattering Slice Buffer"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let scattering_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Atmosphere Scattering Raster BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
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
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let scattering_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmosphere Scattering Raster BG"),
                layout: &scattering_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atmosphere_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&luts.transmittance_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampling_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scattering_slice_buffer.as_entire_binding(),
                    },
                ],
            });

            let irradiance_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Atmosphere Irradiance Raster BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
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
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

            let irradiance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmosphere Irradiance Raster BG"),
                layout: &irradiance_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atmosphere_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&luts.transmittance_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampling_sampler),
                    },
                ],
            });

            let raster_shader = device.create_shader_module(wgpu::include_wgsl!(
                "../shaders/atmosphere_precompute_raster.wgsl"
            ));

            let transmittance_pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Transmittance Raster Pipeline"),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Transmittance Raster Layout"),
                            bind_group_layouts: &[&transmittance_bgl],
                            immediate_size: 0,
                        }),
                    ),
                    vertex: wgpu::VertexState {
                        module: &raster_shader,
                        entry_point: Some("vs_fullscreen"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &raster_shader,
                        entry_point: Some("fs_transmittance"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: lut_format,
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
                });

            let scattering_pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Scattering Raster Pipeline"),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Scattering Raster Layout"),
                            bind_group_layouts: &[&scattering_bgl],
                            immediate_size: 0,
                        }),
                    ),
                    vertex: wgpu::VertexState {
                        module: &raster_shader,
                        entry_point: Some("vs_fullscreen"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &raster_shader,
                        entry_point: Some("fs_scattering"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: lut_format,
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
                });

            let irradiance_pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Irradiance Raster Pipeline"),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Irradiance Raster Layout"),
                            bind_group_layouts: &[&irradiance_bgl],
                            immediate_size: 0,
                        }),
                    ),
                    vertex: wgpu::VertexState {
                        module: &raster_shader,
                        entry_point: Some("vs_fullscreen"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &raster_shader,
                        entry_point: Some("fs_irradiance"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: lut_format,
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
                });

            let mut scattering_slice_views = Vec::with_capacity(SCATTERING_LUT_DEPTH as usize);
            for layer in 0..SCATTERING_LUT_DEPTH {
                scattering_slice_views.push(luts.scattering_lut.create_view(
                    &wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    },
                ));
            }

            AtmospherePrecomputeMode::Raster(AtmospherePrecomputeRaster {
                transmittance_bg,
                scattering_bg,
                irradiance_bg,
                transmittance_pipeline,
                scattering_pipeline,
                irradiance_pipeline,
                scattering_slice_buffer,
                scattering_slice_views,
            })
        };

        Self {
            luts,
            atmosphere_uniform_buffer,
            sampling_bind_group_layout,
            sampling_bind_group,
            precompute,
            scattering_is_array: matches!(scattering_dimension, ScatteringLutDimension::Array2d),
        }
    }
    pub fn precompute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        sky_uniforms: &SkyUniforms,
        shader_constants: &ShaderConstants,
    ) {
        let uniforms = AtmosphereUniforms {
            planet_radius: shader_constants.planet_radius,
            atmosphere_radius: shader_constants.atmosphere_radius,
            sun_intensity: sky_uniforms.sun_intensity,
            _padding: 0.0,
            sun_direction: sky_uniforms.sun_direction,
            _padding2: 0.0,
            rayleigh_scattering_coeff: shader_constants.rayleigh_scattering_coeff,
            rayleigh_scale_height: shader_constants.rayleigh_scale_height,
            mie_scattering_coeff: shader_constants.mie_scattering_coeff,
            mie_absorption_coeff: shader_constants.mie_absorption_coeff,
            mie_scale_height: shader_constants.mie_scale_height,
            mie_preferred_scattering_dir: shader_constants.mie_preferred_scattering_dir,
            ozone_absorption_coeff: shader_constants.ozone_absorption_coeff,
            ozone_center_height: shader_constants.ozone_center_height,
            ozone_falloff: shader_constants.ozone_falloff,
            _pad_after_ozone: [0.0; 3],
            _pad_atmo0: [0.0; 3],
            _pad_after_atmo0: 0.0,
            ground_albedo: shader_constants.sky_ground_albedo,
            ground_brightness: shader_constants.sky_ground_brightness,
            night_ambient_color: shader_constants.night_ambient_color,
            _pad_atmo1: 0.0,
        };
        queue.write_buffer(
            &self.atmosphere_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        match &self.precompute {
            AtmospherePrecomputeMode::Compute(precompute) => {
                // Pass 1: Transmittance
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Transmittance Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&precompute.transmittance_pipeline);
                    cpass.set_bind_group(0, &precompute.transmittance_bg, &[]);
                    cpass.dispatch_workgroups(
                        TRANSMITTANCE_LUT_WIDTH / 8,
                        TRANSMITTANCE_LUT_HEIGHT / 8,
                        1,
                    );
                }

                // Pass 2: Scattering
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Scattering Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&precompute.scattering_pipeline);
                    cpass.set_bind_group(0, &precompute.scattering_bg, &[]);
                    cpass.dispatch_workgroups(
                        SCATTERING_LUT_WIDTH / 8,
                        SCATTERING_LUT_HEIGHT / 8,
                        SCATTERING_LUT_DEPTH,
                    );
                }

                // Pass 3: Irradiance
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Irradiance Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&precompute.irradiance_pipeline);
                    cpass.set_bind_group(0, &precompute.irradiance_bg, &[]);
                    cpass.dispatch_workgroups(
                        IRRADIANCE_LUT_WIDTH / 8,
                        IRRADIANCE_LUT_HEIGHT / 8,
                        1,
                    );
                }
            }
            AtmospherePrecomputeMode::Raster(precompute) => {
                let clear_ops = wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                };

                // Pass 1: Transmittance
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Transmittance Raster Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.luts.transmittance_lut_view,
                            resolve_target: None,
                            ops: clear_ops,
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    rpass.set_pipeline(&precompute.transmittance_pipeline);
                    rpass.set_bind_group(0, &precompute.transmittance_bg, &[]);
                    rpass.draw(0..3, 0..1);
                }

                // Pass 2: Scattering (per-slice)
                for (layer, view) in precompute.scattering_slice_views.iter().enumerate() {
                    let slice = ScatteringSliceUniform {
                        slice: layer as u32,
                        _pad: [0; 7],
                    };
                    queue.write_buffer(
                        &precompute.scattering_slice_buffer,
                        0,
                        bytemuck::bytes_of(&slice),
                    );

                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Scattering Raster Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: clear_ops,
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    rpass.set_pipeline(&precompute.scattering_pipeline);
                    rpass.set_bind_group(0, &precompute.scattering_bg, &[]);
                    rpass.draw(0..3, 0..1);
                }

                // Pass 3: Irradiance
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Irradiance Raster Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.luts.irradiance_lut_view,
                            resolve_target: None,
                            ops: clear_ops,
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    rpass.set_pipeline(&precompute.irradiance_pipeline);
                    rpass.set_bind_group(0, &precompute.irradiance_bg, &[]);
                    rpass.draw(0..3, 0..1);
                }
            }
        }
    }

    pub fn uses_array_scattering(&self) -> bool {
        self.scattering_is_array
    }
}
