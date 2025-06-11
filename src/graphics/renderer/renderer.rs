use crate::{
    graphics::renderer::{device::RenderDevice, error::RendererError},
    provided::components::{LightType, Transform},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use mev::{Arguments, BufferDesc, DeviceRepr};
use std::collections::HashMap;
use std::sync::Arc;
use winit::window::Window;

const MAX_TEXTURE_COUNT: u32 = 256;
const TEXTURE_RESOLUTION: u32 = 1024;

const MAX_LIGHTS: usize = 32;

const FRAMES_IN_FLIGHT: usize = 3;

/// Represents a loaded mesh with its vertex and index buffers.
pub struct Mesh {
    pub vertex_buffer: mev::Buffer,
    pub index_buffer: mev::Buffer,
    pub index_count: u32,
}

pub struct TextureManager {
    pub textures: HashMap<String, usize>,
    pub next_texture_index: usize,
}

impl Default for TextureManager {
    fn default() -> Self {
        Self {
            textures: HashMap::new(),
            // Reserve index 0 for default textures, start assigning from 1.
            next_texture_index: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub albedo_texture_index: i32,
    pub normal_texture_index: i32,
    pub metallic_roughness_texture_index: i32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            ao: 1.0,
            // Point to the guaranteed default textures at index 0.
            albedo_texture_index: 0,
            normal_texture_index: 0,
            metallic_roughness_texture_index: 0,
        }
    }
}

pub struct TextureHandle {
    pub mev_image: mev::Image,
    pub format: mev::PixelFormat,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DefaultProperties {
    pub albedo: [f32; 4],
    pub normal: [f32; 3],
    pub _p1: f32,
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub _p2: f32,
}

#[derive(Debug, Clone)]
pub struct RenderObject {
    pub transform: Transform,
    pub mesh_id: usize,
    pub material_id: usize,
}

#[derive(Debug, Clone)]
pub struct RenderLight {
    pub transform: Transform,
    pub color: [f32; 3],
    pub intensity: f32,
    pub light_type: LightType,
}

pub struct RenderData {
    pub objects: Vec<RenderObject>,
    pub lights: Vec<RenderLight>,
    pub camera_transform: Transform,
}

#[derive(mev::Arguments)]
struct PbrArguments {
    #[mev(vertex, fragment)]
    camera_buffer: mev::Buffer,
    #[mev(fragment)]
    lights_buffer: mev::Buffer,
    #[mev(fragment)]
    material_buffer: mev::Buffer,
    #[mev(shader(fragment), sampled)]
    albedo_texture_array: mev::Image,
    #[mev(shader(fragment), sampled)]
    normal_texture_array: mev::Image,
    #[mev(shader(fragment), sampled)]
    metallic_roughness_texture_array: mev::Image,
    #[mev(fragment)]
    sampler: mev::Sampler,
    #[mev(fragment)]
    default_properties_buffer: mev::Buffer,
}

#[repr(C)]
#[derive(mev::DeviceRepr)]
struct CameraUniforms {
    view_matrix: [[f32; 4]; 4],
    projection_matrix: [[f32; 4]; 4],
    view_position: [f32; 3],
    light_count: u32,
}

#[repr(C)]
#[derive(mev::DeviceRepr, Clone, Copy, Pod, Zeroable)]
struct LightData {
    position: [f32; 3],
    light_type: u32,
    color: [f32; 3],
    intensity: f32,
    direction: [f32; 3],
    _padding: f32,
}

#[repr(C)]
#[derive(mev::DeviceRepr)]
struct PbrConstants {
    model_matrix: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    material_id: u32,
    _p: [u32; 3],
}

#[repr(C)]
#[derive(mev::DeviceRepr, Clone, Copy, Pod, Zeroable)]
struct MaterialShaderData {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    ao: f32,
    _p1: f32,
    albedo_texture_index: i32,
    normal_texture_index: i32,
    metallic_roughness_texture_index: i32,
    _p2: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, DeviceRepr, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
    pub tangent: [f32; 3],
}

impl From<([f32; 3], [f32; 3], [f32; 2], [f32; 3])> for Vertex {
    fn from(data: ([f32; 3], [f32; 3], [f32; 2], [f32; 3])) -> Self {
        Vertex {
            position: data.0,
            normal: data.1,
            tex_coord: data.2,
            tangent: data.3,
        }
    }
}

pub struct Renderer {
    mev_device: Arc<mev::Device>,
    surface: mev::Surface,
    queue: mev::Queue,
    shader_library: Option<mev::Library>,
    pipeline: Option<mev::RenderPipeline>,
    depth_texture: Option<mev::Image>,

    camera_buffers: Vec<mev::Buffer>,
    lights_buffers: Vec<mev::Buffer>,

    material_uniform_buffer: Option<mev::Buffer>,
    meshes: HashMap<usize, Mesh>,
    materials: HashMap<usize, Material>,
    textures: HashMap<usize, TextureHandle>,
    albedo_texture_array: Option<mev::Image>,
    normal_texture_array: Option<mev::Image>,
    metallic_roughness_texture_array: Option<mev::Image>,
    default_properties_buffer: Option<mev::Buffer>,
    texture_manager: TextureManager,
    sampler: Option<mev::Sampler>,

    frame_index: usize,

    current_render_data: Option<RenderData>,
    last_format: Option<mev::PixelFormat>,
    last_extent: Option<mev::Extent2>,
    pending_uploads: Vec<mev::CommandBuffer>,
}

fn prepare_texture_slice_upload(
    queue: &mut mev::Queue,
    data: &[u8],
    target_array: &mev::Image,
    layer_index: u32,
) -> Result<mev::CommandBuffer, mev::OutOfMemory> {
    // This part of your logic for padding the buffer is correct and should be kept.
    let bytes_per_pixel = 4;
    let align = 256u32;
    let bytes_per_row_unaligned = TEXTURE_RESOLUTION * bytes_per_pixel;
    let bytes_per_row = (bytes_per_row_unaligned + align - 1) & !(align - 1);
    let padded_data_size = (bytes_per_row * TEXTURE_RESOLUTION) as usize;

    let mut padded_data = vec![0; padded_data_size];
    for y in 0..TEXTURE_RESOLUTION {
        let src_offset = (y * bytes_per_row_unaligned) as usize;
        let dst_offset = (y * bytes_per_row) as usize;
        if src_offset + bytes_per_row_unaligned as usize > data.len() {
            continue;
        }
        let src_slice = &data[src_offset..src_offset + bytes_per_row_unaligned as usize];
        padded_data[dst_offset..dst_offset + bytes_per_row_unaligned as usize]
            .copy_from_slice(src_slice);
    }

    let scratch = queue
        .device()
        .new_buffer_init(mev::BufferInitDesc {
            data: &padded_data,
            usage: mev::BufferUsage::TRANSFER_SRC,
            memory: mev::Memory::Upload,
            name: &format!("tex-slice-upload-{}", layer_index),
        })
        .unwrap();

    let mut upload_encoder = queue.new_command_encoder().unwrap();

    // ========================================================================
    // --- CHANGE #1: Perform the initial layout transition on the texture ---
    // This only needs to happen once, but doing it before each copy is safe.
    upload_encoder.init_image(
        mev::PipelineStages::empty(),
        mev::PipelineStages::TRANSFER, // Transition TO Transfer-Ready
        target_array,
    );

    // --- CHANGE #2: Modify the copy_buffer_to_image call arguments ---
    upload_encoder.copy().copy_buffer_to_image(
        &scratch,
        0,                      // source_offset
        bytes_per_row as usize, // source_bytes_per_row
        padded_data_size,       // source_bytes_per_image
        target_array,
        mev::Offset3::ZERO, // destination_origin is ZERO
        mev::Extent3::new(TEXTURE_RESOLUTION, TEXTURE_RESOLUTION, 1), // copy_extent
        0..1,               // destination_mip_levels (copy 1 mip)
        layer_index,        // Use this parameter for the array slice index
    );
    // ========================================================================

    // We still need a barrier to make the texture readable by the shader.
    // Transition FROM Transfer-Ready TO Fragment-Shader-Ready.
    // Following your guidance on the (after, before) signature:
    upload_encoder.barrier(
        mev::PipelineStages::FRAGMENT_SHADER,
        mev::PipelineStages::TRANSFER,
    );

    upload_encoder.finish()
}

fn prepare_uniform_updates(
    queue: &mut mev::Queue,
    camera_buffer: &mev::Buffer,
    lights_buffer: &mev::Buffer,
    render_data: &RenderData,
    window_size: (u32, u32),
) -> Result<mev::CommandBuffer, mev::OutOfMemory> {
    let camera_transform = &render_data.camera_transform;
    let eye: Vec3 = camera_transform.position.into();
    let forward = camera_transform.forward().normalize_or_zero();
    let view_matrix = Mat4::look_at_rh(
        eye,
        eye + forward,
        camera_transform.up().normalize_or(Vec3::Y),
    );
    let projection_matrix = Mat4::perspective_rh(
        (45.0_f32).to_radians(),
        window_size.0 as f32 / window_size.1 as f32,
        0.1,
        1000.0,
    );

    let light_data: Vec<LightData> = render_data
        .lights
        .iter()
        .take(MAX_LIGHTS)
        .map(|light| LightData {
            position: light.transform.position.into(),
            light_type: match light.light_type {
                LightType::Directional => 0,
                LightType::Point => 1,
                LightType::Spot { .. } => 2,
            },
            color: light.color,
            intensity: light.intensity,
            direction: light.transform.forward().into(),
            _padding: 0.0,
        })
        .collect();

    let num_lights = light_data.len() as u32;

    let camera_uniforms = CameraUniforms {
        view_matrix: view_matrix.to_cols_array_2d(),
        projection_matrix: projection_matrix.to_cols_array_2d(),
        view_position: eye.to_array(),
        light_count: num_lights,
    };

    let mut encoder = queue.new_command_encoder().unwrap();
    {
        let mut copy_pass = encoder.copy();
        copy_pass.write_buffer(camera_buffer, &camera_uniforms.as_repr());
        copy_pass.write_buffer_slice(lights_buffer.slice(0..), &light_data);
    }
    encoder.finish()
}

impl Renderer {
    pub fn new(window: &Window) -> Result<Self, RendererError> {
        let instance = mev::Instance::load()
            .map_err(|e| RendererError::MevError(format!("Failed to load MEV instance: {}", e)))?;
        let (mev_device, mut queues) = instance
            .new_device(mev::DeviceDesc {
                idx: 0,
                queues: &[0],
                features: mev::Features::SURFACE,
            })
            .map_err(|e| RendererError::MevError(format!("Failed to create device: {}", e)))?;
        let queue = queues.pop().unwrap();
        let surface = queue
            .new_surface(window, window)
            .map_err(|e| RendererError::MevError(format!("Failed to create surface: {}", e)))?;
        Ok(Self {
            mev_device: mev_device.into(),
            surface,
            queue,
            shader_library: None,
            pipeline: None,
            depth_texture: None,
            camera_buffers: Vec::new(),
            lights_buffers: Vec::new(),
            material_uniform_buffer: None,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            albedo_texture_array: None,
            normal_texture_array: None,
            metallic_roughness_texture_array: None,
            default_properties_buffer: None,
            texture_manager: TextureManager::default(),
            sampler: None,
            frame_index: 0,
            current_render_data: None,
            last_format: None,
            last_extent: None,
            pending_uploads: Vec::new(),
        })
    }

    fn initialize_resources(&mut self) -> Result<(), mev::DeviceError> {
        self.shader_library = Some(
            self.queue
                .new_shader_library(mev::LibraryDesc {
                    name: "pbr",
                    input: mev::include_library!(
                        "../shaders/pbr.wgsl" as mev::ShaderLanguage::Wgsl
                    ),
                })
                .unwrap(),
        );
        self.sampler = Some(self.queue.new_sampler(mev::SamplerDesc {
            min_filter: mev::Filter::Linear,
            mag_filter: mev::Filter::Linear,
            address_mode: [mev::AddressMode::Repeat; 3],
            ..mev::SamplerDesc::new()
        })?);

        let default_props = DefaultProperties {
            albedo: [1.0, 1.0, 1.0, 1.0],
            normal: [0.5, 0.5, 1.0],
            _p1: 0.0,
            metallic: 0.0,
            roughness: 0.8,
            ao: 1.0,
            _p2: 0.0,
        };
        self.default_properties_buffer = Some(self.queue.new_buffer_init(mev::BufferInitDesc {
            data: bytemuck::bytes_of(&default_props),
            usage: mev::BufferUsage::UNIFORM,
            memory: mev::Memory::Device,
            name: "default-properties-buffer",
        })?);

        self.albedo_texture_array =
            Some(
                self.queue.new_image(mev::ImageDesc {
                    extent: mev::Extent3::new(
                        TEXTURE_RESOLUTION,
                        TEXTURE_RESOLUTION,
                        MAX_TEXTURE_COUNT,
                    )
                    .into(),
                    format: mev::PixelFormat::Rgba8Srgb,
                    usage: mev::ImageUsage::SAMPLED | mev::ImageUsage::TRANSFER_DST,
                    layers: 1,
                    levels: 1,
                    name: "albedo-texture-array",
                })?,
            );
        self.normal_texture_array =
            Some(
                self.queue.new_image(mev::ImageDesc {
                    extent: mev::Extent3::new(
                        TEXTURE_RESOLUTION,
                        TEXTURE_RESOLUTION,
                        MAX_TEXTURE_COUNT,
                    )
                    .into(),
                    format: mev::PixelFormat::Rgba8Unorm,
                    usage: mev::ImageUsage::SAMPLED | mev::ImageUsage::TRANSFER_DST,
                    layers: 1,
                    levels: 1,
                    name: "normal-texture-array",
                })?,
            );
        self.metallic_roughness_texture_array =
            Some(
                self.queue.new_image(mev::ImageDesc {
                    extent: mev::Extent3::new(
                        TEXTURE_RESOLUTION,
                        TEXTURE_RESOLUTION,
                        MAX_TEXTURE_COUNT,
                    )
                    .into(),
                    format: mev::PixelFormat::Rgba8Unorm,
                    usage: mev::ImageUsage::SAMPLED | mev::ImageUsage::TRANSFER_DST,
                    layers: 1,
                    levels: 1,
                    name: "metallic-roughness-texture-array",
                })?,
            );

        let resolution_area = (TEXTURE_RESOLUTION * TEXTURE_RESOLUTION) as usize;
        let white_pixel = [255u8, 255, 255, 255];
        let default_albedo_data: Vec<u8> = white_pixel
            .iter()
            .cycle()
            .take(resolution_area * 4)
            .copied()
            .collect();
        let albedo_upload_cbuf = prepare_texture_slice_upload(
            &mut self.queue,
            &default_albedo_data,
            self.albedo_texture_array.as_ref().unwrap(),
            0,
        )
        .unwrap();
        //self.pending_uploads.push(albedo_upload_cbuf);

        let flat_normal_pixel = [128u8, 128, 255, 255];
        let default_normal_data: Vec<u8> = flat_normal_pixel
            .iter()
            .cycle()
            .take(resolution_area * 4)
            .copied()
            .collect();
        let normal_upload_cbuf = prepare_texture_slice_upload(
            &mut self.queue,
            &default_normal_data,
            self.normal_texture_array.as_ref().unwrap(),
            0,
        )
        .unwrap();
        //self.pending_uploads.push(normal_upload_cbuf);

        let default_mr_pixel = [255u8, 204, 0, 255];
        let default_mr_data: Vec<u8> = default_mr_pixel
            .iter()
            .cycle()
            .take(resolution_area * 4)
            .copied()
            .collect();
        let mr_upload_cbuf = prepare_texture_slice_upload(
            &mut self.queue,
            &default_mr_data,
            self.metallic_roughness_texture_array.as_ref().unwrap(),
            0,
        )
        .unwrap();
        //self.pending_uploads.push(mr_upload_cbuf);

        let initial_uploads = vec![albedo_upload_cbuf, normal_upload_cbuf, mr_upload_cbuf];
        self.queue.submit(initial_uploads, true).unwrap();
        self.queue.wait_idle().unwrap();

        self.camera_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                self.queue.new_buffer(mev::BufferDesc {
                    size: std::mem::size_of::<CameraUniforms>(),
                    usage: mev::BufferUsage::UNIFORM | mev::BufferUsage::TRANSFER_DST,
                    memory: mev::Memory::Device,
                    name: &format!("camera-uniforms-{}", i),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.lights_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                self.queue.new_buffer(mev::BufferDesc {
                    size: std::mem::size_of::<LightData>() * MAX_LIGHTS,
                    usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_DST,
                    memory: mev::Memory::Device,
                    name: &format!("lights-buffer-{}", i),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.material_uniform_buffer = Some(self.queue.new_buffer(mev::BufferDesc {
            size: std::mem::size_of::<MaterialShaderData>() * 256,
            usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_DST,
            memory: mev::Memory::Device,
            name: "materials-buffer",
        })?);

        Ok(())
    }

    pub fn add_texture_to_array(
        &mut self,
        name: &str,
        data: &[u8],
        target_array: &mev::Image,
    ) -> Result<usize, RendererError> {
        if let Some(index) = self.texture_manager.textures.get(name) {
            return Ok(*index);
        }
        if self.texture_manager.next_texture_index >= MAX_TEXTURE_COUNT as usize {
            return Err(RendererError::ResourceCreation(
                "Texture array is full".to_string(),
            ));
        }

        let index = self.texture_manager.next_texture_index;
        self.texture_manager.next_texture_index += 1;
        let upload_cbuf =
            prepare_texture_slice_upload(&mut self.queue, data, target_array, index as u32)
                .unwrap();
        self.pending_uploads.push(upload_cbuf);
        self.texture_manager
            .textures
            .insert(name.to_string(), index);
        Ok(index)
    }

    pub fn add_mesh(
        &mut self,
        id: usize,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Result<(), RendererError> {
        let vertex_buffer = self
            .queue
            .new_buffer_init(mev::BufferInitDesc {
                data: bytemuck::cast_slice(vertices),
                usage: mev::BufferUsage::VERTEX,
                memory: mev::Memory::Device,
                name: &format!("mesh-vbo-{}", id),
            })
            .map_err(|e| {
                RendererError::MevError(format!("Failed to create vertex buffer: {}", e))
            })?;
        let index_buffer = self
            .queue
            .new_buffer_init(mev::BufferInitDesc {
                data: bytemuck::cast_slice(indices),
                usage: mev::BufferUsage::INDEX,
                memory: mev::Memory::Device,
                name: &format!("mesh-ibo-{}", id),
            })
            .map_err(|e| {
                RendererError::MevError(format!("Failed to create index buffer: {}", e))
            })?;
        self.meshes.insert(
            id,
            Mesh {
                vertex_buffer,
                index_buffer,
                index_count: indices.len() as u32,
            },
        );
        Ok(())
    }

    pub fn add_material(&mut self, id: usize, material: Material) -> Result<(), RendererError> {
        if id >= 256 {
            return Err(RendererError::ResourceCreation(format!(
                "Material ID {} is out of bounds. Maximum is 255.",
                id
            )));
        }
        let shader_data = MaterialShaderData {
            albedo: material.albedo,
            metallic: material.metallic,
            roughness: material.roughness,
            ao: material.ao,
            _p1: 0.0,
            albedo_texture_index: material.albedo_texture_index,
            normal_texture_index: material.normal_texture_index,
            metallic_roughness_texture_index: material.metallic_roughness_texture_index,
            _p2: 0,
        };
        let material_offset = id * std::mem::size_of::<MaterialShaderData>();
        let material_slice =
            self.material_uniform_buffer.as_ref().unwrap().slice(
                material_offset..(material_offset + std::mem::size_of::<MaterialShaderData>()),
            );

        let mut upload_encoder = self.queue.new_command_encoder().unwrap();
        upload_encoder
            .copy()
            .write_buffer_slice(material_slice, &[shader_data]);
        let upload_cbuf = upload_encoder.finish().unwrap();
        self.pending_uploads.push(upload_cbuf);
        self.materials.insert(id, material);
        Ok(())
    }

    // ✅ CORRECTED: Reverted to an associated function (`Self::`) to avoid the borrow conflict.
    fn create_or_update_pipeline(
        queue: &mev::Queue,
        shader_library: &Option<mev::Library>,
        pipeline: &mut Option<mev::RenderPipeline>,
        depth_texture: &mut Option<mev::Image>,
        last_format: &mut Option<mev::PixelFormat>,
        last_extent: &mut Option<mev::Extent2>,
        target_format: mev::PixelFormat,
        target_extent: mev::Extent2,
    ) -> Result<(), mev::DeviceError> {
        if pipeline.is_some()
            && *last_format == Some(target_format)
            && *last_extent == Some(target_extent)
        {
            return Ok(());
        }
        let library = shader_library.as_ref().unwrap();
        *depth_texture = Some(queue.new_image(mev::ImageDesc {
            extent: target_extent.into(),
            format: mev::PixelFormat::D32Float,
            usage: mev::ImageUsage::all(),
            layers: 1,
            levels: 1,
            name: "depth-buffer",
        })?);
        *pipeline = Some(
            queue
                .new_render_pipeline(mev::RenderPipelineDesc {
                    name: "pbr-pipeline",
                    vertex_shader: library.entry("vs_main"),
                    vertex_attributes: vec![
                        mev::VertexAttributeDesc {
                            format: mev::VertexFormat::Float32x3,
                            offset: 0,
                            buffer_index: 0,
                        },
                        mev::VertexAttributeDesc {
                            format: mev::VertexFormat::Float32x3,
                            offset: 12,
                            buffer_index: 0,
                        },
                        mev::VertexAttributeDesc {
                            format: mev::VertexFormat::Float32x2,
                            offset: 24,
                            buffer_index: 0,
                        },
                        mev::VertexAttributeDesc {
                            format: mev::VertexFormat::Float32x3,
                            offset: 32,
                            buffer_index: 0,
                        },
                    ],
                    vertex_layouts: vec![mev::VertexLayoutDesc {
                        stride: std::mem::size_of::<Vertex>() as u32,
                        step_mode: mev::VertexStepMode::Vertex,
                    }],
                    primitive_topology: mev::PrimitiveTopology::Triangle,
                    raster: Some(mev::RasterDesc {
                        fragment_shader: Some(library.entry("fs_main")),
                        color_targets: vec![mev::ColorTargetDesc {
                            format: target_format,
                            blend: None,
                        }],
                        depth_stencil: Some(mev::DepthStencilDesc {
                            format: mev::PixelFormat::D32Float,
                            write_enabled: true,
                            compare: mev::CompareFunction::Less,
                        }),
                        front_face: mev::FrontFace::CounterClockwise,
                        culling: mev::Culling::Back,
                    }),
                    arguments: &[PbrArguments::LAYOUT],
                    constants: PbrConstants::SIZE,
                })
                .unwrap(),
        );
        *last_format = Some(target_format);
        *last_extent = Some(target_extent);
        Ok(())
    }

    pub fn update_render_data(&mut self, render_data: RenderData) {
        self.current_render_data = Some(render_data);
    }

    pub fn render(&mut self, window: &Window) -> Result<(), RendererError> {
        if self.shader_library.is_none() {
            self.initialize_resources().map_err(|e| {
                RendererError::MevError(format!("Failed to initialize resources: {}", e))
            })?;
        }

        let Some(ref render_data) = self.current_render_data else {
            return Ok(());
        };

        let mut frame = self
            .surface
            .next_frame()
            .map_err(|e| RendererError::MevError(format!("Failed to get next frame: {}", e)))?;
        let target_format = frame.image().format();
        let target_extent = frame.image().extent().expect_2d();

        // ✅ CORRECTED: Call the associated function `Self::` and pass the mutable fields explicitly.
        // This resolves the borrow checker error.
        Self::create_or_update_pipeline(
            &self.queue,
            &self.shader_library,
            &mut self.pipeline,
            &mut self.depth_texture,
            &mut self.last_format,
            &mut self.last_extent,
            target_format,
            target_extent,
        )
        .map_err(|e| RendererError::MevError(format!("Failed to create pipeline: {}", e)))?;

        let current_camera_buffer = &self.camera_buffers[self.frame_index];
        let current_lights_buffer = &self.lights_buffers[self.frame_index];

        let mut command_buffers_for_frame = std::mem::take(&mut self.pending_uploads);

        let uniform_cbuf = prepare_uniform_updates(
            &mut self.queue,
            current_camera_buffer,
            current_lights_buffer,
            render_data,
            (window.inner_size().width, window.inner_size().height),
        )
        .unwrap();
        command_buffers_for_frame.push(uniform_cbuf);

        let mut encoder = self.queue.new_command_encoder().unwrap();
        encoder.init_image(
            mev::PipelineStages::COLOR_OUTPUT,
            mev::PipelineStages::empty(),
            frame.image(),
        );

        encoder.barrier(
            mev::PipelineStages::all(),
            mev::PipelineStages::FRAGMENT_SHADER,
        );

        {
            let mut render_pass = encoder.render(mev::RenderPassDesc {
                name: "main-pass",
                color_attachments: &[
                    mev::AttachmentDesc::new(frame.image()).clear(mev::ClearColor::DARK_GRAY)
                ],
                depth_stencil_attachment: Some(
                    mev::AttachmentDesc::new(self.depth_texture.as_ref().unwrap())
                        .clear(mev::ClearDepthStencil::default()),
                ),
            });
            render_pass.with_viewport(mev::Offset3::ZERO, target_extent.to_3d().cast_as_f32());
            render_pass.with_scissor(mev::Offset2::ZERO, target_extent);
            render_pass.with_pipeline(self.pipeline.as_ref().unwrap());
            render_pass.with_arguments(
                0,
                &PbrArguments {
                    camera_buffer: current_camera_buffer.clone(),
                    lights_buffer: current_lights_buffer.clone(),
                    material_buffer: self.material_uniform_buffer.as_ref().unwrap().clone(),
                    albedo_texture_array: self.albedo_texture_array.as_ref().unwrap().clone(),
                    normal_texture_array: self.normal_texture_array.as_ref().unwrap().clone(),
                    metallic_roughness_texture_array: self
                        .metallic_roughness_texture_array
                        .as_ref()
                        .unwrap()
                        .clone(),
                    sampler: self.sampler.as_ref().unwrap().clone(),
                    default_properties_buffer: self
                        .default_properties_buffer
                        .as_ref()
                        .unwrap()
                        .clone(),
                },
            );

            for object in &render_data.objects {
                let Some(mesh) = self.meshes.get(&object.mesh_id) else {
                    continue;
                };
                if !self.materials.contains_key(&object.material_id) {
                    continue;
                };

                let transform = &object.transform;
                let scale: Vec3 = transform.scale.into();

                // check to prevent NaN generation
                let is_scale_valid =
                    scale.x.abs() > 0.0001 && scale.y.abs() > 0.0001 && scale.z.abs() > 0.0001;

                let model_matrix = Mat4::from_scale_rotation_translation(
                    object.transform.scale.into(),
                    Quat::from(object.transform.rotation).normalize(),
                    object.transform.position.into(),
                ).to_cols_array_2d(); 
                
                // Use a safe fallback if the scale is zero
                let normal_matrix = if is_scale_valid {
                    // The inverse transpose of the model matrix is correct for normals
                    Mat4::from_mat3(
                    Mat3::from_mat4(Mat4::from_cols_array_2d(&model_matrix))
                        .inverse()
                        .transpose(),
                    ).to_cols_array_2d()
                } else {
                    // Fallback to a non-inverted matrix if scale is zero
                    model_matrix
                };

                render_pass.with_constants(&PbrConstants {
                    model_matrix,
                    normal_matrix,
                    material_id: object.material_id as u32,
                    _p: [0; 3],
                });
                render_pass.bind_vertex_buffers(0, &[mesh.vertex_buffer.clone()]);
                render_pass.bind_index_buffer(&mesh.index_buffer);
                render_pass.draw_indexed(0, 0..mesh.index_count, 0..1);
            }
        }
        self.queue
            .sync_frame(&mut frame, mev::PipelineStages::COLOR_OUTPUT);
        encoder.present(frame, mev::PipelineStages::COLOR_OUTPUT);
        let main_render_cbuf = encoder.finish().unwrap();
        command_buffers_for_frame.push(main_render_cbuf);

        window.pre_present_notify();
        self.queue.submit(command_buffers_for_frame, true).unwrap();

        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;

        Ok(())
    }
}
