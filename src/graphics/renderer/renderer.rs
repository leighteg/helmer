use crate::{
    graphics::renderer::{device::RenderDevice, error::RendererError},
    provided::components::{LightType, Transform},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3, Vec3Swizzles}; // Using glam for proper matrix and vector math
use mev::{Arguments, BufferDesc, DeviceRepr};
use std::collections::HashMap;
use std::sync::Arc;
use winit::window::Window;

// --- New and Modified Data Structures ---

/// Represents a loaded mesh with its vertex and index buffers.
pub struct Mesh {
    pub vertex_buffer: mev::Buffer,
    pub index_buffer: mev::Buffer,
    pub index_count: u32,
}

/// Represents a material with its properties and texture IDs.
#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub albedo_texture_id: Option<usize>, // Optional texture ID
    pub normal_texture_id: Option<usize>, // Optional texture ID
    pub metallic_roughness_texture_id: Option<usize>, // Optional texture ID
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: [1.0, 1.0, 1.0, 1.0], // Default to white, fully opaque
            metallic: 0.0,                // Default to non-metallic
            roughness: 0.8,               // Default to somewhat rough
            ao: 1.0,                      // Default to full ambient occlusion (no darkening)
            albedo_texture_id: None,
            normal_texture_id: None,
            metallic_roughness_texture_id: None,
        }
    }
}

/// A handle to a loaded texture.
pub struct TextureHandle {
    pub mev_image: mev::Image,
    pub format: mev::PixelFormat,
    pub width: u32,
    pub height: u32,
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

// MEV Arguments for PBR rendering
#[derive(mev::Arguments)]
struct PbrArguments {
    #[mev(vertex, fragment)]
    camera_buffer: mev::Buffer,
    #[mev(fragment)] // Lights are only used in fragment shader
    lights_buffer: mev::Buffer,
    #[mev(fragment)] // Materials are only used in fragment shader
    material_buffer: mev::Buffer,
    #[mev(fragment)]
    albedo_texture: mev::Image,
    #[mev(fragment)]
    normal_texture: mev::Image,
    #[mev(fragment)]
    metallic_roughness_texture: mev::Image,
    #[mev(fragment)]
    sampler: mev::Sampler,
}

// Constants for PBR shader
#[derive(mev::DeviceRepr)]
struct PbrConstants {
    model_matrix: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    material_id: u32,
}

// Camera uniform buffer
#[derive(mev::DeviceRepr)]
struct CameraUniforms {
    view_matrix: [[f32; 4]; 4],
    projection_matrix: [[f32; 4]; 4],
    view_position: [f32; 3],
    _padding: f32,
}

// Light data structure
#[derive(mev::DeviceRepr)]
struct LightData {
    position: [f32; 3],
    light_type: u32, // 0: directional, 1: point, 2: spot
    color: [f32; 3],
    intensity: f32,
    direction: [f32; 3],
    _padding: f32,
}

// Material data structure (aligned for shader, no padding needed for f32)
#[repr(C)] // Crucial for memory layout consistency with C/GPU
#[derive(mev::DeviceRepr, Clone, Copy)] // mev::DeviceRepr should handle the repr(C) correctly
struct MaterialShaderData {
    albedo: [f32; 4], // 16 bytes, naturally aligned
    metallic: f32,    // 4 bytes
    roughness: f32,   // 4 bytes
    ao: f32,          // 4 bytes
    // Total so far: 28 bytes.
    // Need to pad to 32 bytes to match GPU's 16-byte (or 32-byte) alignment for array elements.
    _padding: [f32; 1], // Add 4 bytes of padding (1 * 4 bytes = 4 bytes)
                        // This makes the total size 28 + 4 = 32 bytes.
}
// Derive Pod and Zeroable after the padding has been correctly applied.
// Assuming MaterialShaderData will be used in slices/buffers.
unsafe impl Pod for MaterialShaderData {}
unsafe impl Zeroable for MaterialShaderData {}

#[repr(C)] // Crucial for memory layout consistency with C/GPU
#[derive(Debug, Clone, Copy, DeviceRepr, Pod, Zeroable)] // Derive Pod and Zeroable
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
    device: RenderDevice, // Assuming RenderDevice wraps mev::Device
    surface: mev::Surface,
    queue: mev::Queue,

    // Rendering resources
    shader_library: Option<mev::Library>,
    pipeline: Option<mev::RenderPipeline>,
    depth_texture: Option<mev::Image>,

    // Uniform buffers
    camera_buffer: Option<mev::Buffer>,
    lights_buffer: Option<mev::Buffer>,
    // The material_buffer will now hold all MaterialShaderData
    material_uniform_buffer: Option<mev::Buffer>,

    // Asset storage
    meshes: HashMap<usize, Mesh>,
    materials: HashMap<usize, Material>,
    textures: HashMap<usize, TextureHandle>,

    // Default textures
    default_albedo_tex: Option<mev::Image>,
    default_normal_tex: Option<mev::Image>,
    default_metallic_roughness_tex: Option<mev::Image>,
    sampler: Option<mev::Sampler>,

    frame_count: u64,
    current_render_data: Option<RenderData>,
    last_format: Option<mev::PixelFormat>,

    // Store a command encoder for asset uploads to avoid re-creating it often
    asset_upload_encoder: Option<mev::CommandEncoder>,
}

impl Renderer {
    pub fn new(window: &Window) -> Result<Self, RendererError> {
        let instance = mev::Instance::load()
            .map_err(|e| RendererError::MevError(format!("Failed to load MEV instance: {}", e)))?;

        let (_device, mut queues) = instance
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

        tracing::info!("MEV Renderer initialized");

        Ok(Self {
            device: RenderDevice::new()?, // Assuming RenderDevice handles device creation and access
            surface,
            queue,
            shader_library: None,
            pipeline: None,
            depth_texture: None,
            camera_buffer: None,
            lights_buffer: None,
            material_uniform_buffer: None, // Renamed for clarity

            meshes: HashMap::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),

            default_albedo_tex: None,
            default_normal_tex: None,
            default_metallic_roughness_tex: None,
            sampler: None,

            frame_count: 0,
            current_render_data: None,
            last_format: None,
            asset_upload_encoder: None,
        })
    }

    fn initialize_resources(&mut self) -> Result<(), mev::DeviceError> {
        // Create shader library
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

        // Create sampler
        self.sampler = Some(self.queue.new_sampler(mev::SamplerDesc {
            min_filter: mev::Filter::Linear,
            mag_filter: mev::Filter::Linear,
            address_mode: [mev::AddressMode::Repeat; 3],
            ..mev::SamplerDesc::new()
        })?);

        // Create uniform buffers
        self.camera_buffer = Some(self.queue.new_buffer(mev::BufferDesc {
            size: std::mem::size_of::<CameraUniforms>(),
            usage: mev::BufferUsage::UNIFORM | mev::BufferUsage::TRANSFER_DST,
            memory: mev::Memory::Device,
            name: "camera-uniforms",
        })?);

        self.lights_buffer = Some(self.queue.new_buffer(mev::BufferDesc {
            size: std::mem::size_of::<LightData>() * 32, // Support up to 32 lights
            usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_DST, // Changed to STORAGE
            memory: mev::Memory::Device,
            name: "lights-buffer",
        })?);

        self.material_uniform_buffer = Some(self.queue.new_buffer(mev::BufferDesc {
            size: std::mem::size_of::<MaterialShaderData>() * 256, // Support up to 256 materials
            usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_DST, // Changed to STORAGE
            memory: mev::Memory::Device,
            name: "materials-buffer",
        })?);

        // Initialize asset upload encoder
        self.asset_upload_encoder = Some(self.queue.new_command_encoder()?);

        // Create default textures
        self.create_default_textures()?;

        Ok(())
    }

    fn create_default_textures(&mut self) -> Result<(), mev::DeviceError> {
        // Create default white albedo texture
        let white_pixel = [255u8, 255, 255, 255];
        self.default_albedo_tex = Some(self.create_texture_from_data(
            &white_pixel,
            1,
            1,
            mev::PixelFormat::Rgba8Srgb,
            "default-albedo",
        )?);

        // Create default normal texture (flat normal: 128, 128, 255, 255)
        let normal_pixel = [128u8, 128, 255, 255];
        self.default_normal_tex = Some(self.create_texture_from_data(
            &normal_pixel,
            1,
            1,
            mev::PixelFormat::Rgba8Unorm,
            "default-normal",
        )?);

        // Create default metallic-roughness texture (no metallic, medium roughness: 0, 128, 0, 255)
        let mr_pixel = [0u8, 128, 0, 255];
        self.default_metallic_roughness_tex = Some(self.create_texture_from_data(
            &mr_pixel,
            1,
            1,
            mev::PixelFormat::Rgba8Unorm,
            "default-metallic-roughness",
        )?);

        Ok(())
    }

    fn create_texture_from_data(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        format: mev::PixelFormat,
        name: &str,
    ) -> Result<mev::Image, mev::DeviceError> {
        let image = self.queue.new_image(mev::ImageDesc {
            extent: mev::Extent2::new(width, height).into(),
            format,
            usage: mev::ImageUsage::SAMPLED | mev::ImageUsage::TRANSFER_DST,
            layers: 1,
            levels: 1,
            name,
        })?;

        let scratch = self
            .queue
            .device()
            .new_buffer_init(mev::BufferInitDesc {
                data,
                usage: mev::BufferUsage::TRANSFER_SRC,
                memory: mev::Memory::Upload,
                name: "scratch",
            })
            .unwrap();

        // Use the persistent asset_upload_encoder
        let encoder = self.asset_upload_encoder.as_mut().unwrap();
        encoder.copy().init_image(
            mev::PipelineStages::empty(),
            mev::PipelineStages::TRANSFER,
            &image,
        );
        // Upload data after initialization
        encoder.copy().copy_buffer_to_image(
            &scratch,
            0,
            4 * image.extent().width() as usize,
            4 * image.extent().width() as usize * image.extent().height() as usize,
            &image,
            mev::Offset3::ZERO,
            image.extent().into_3d(),
            0..1,
            1,
        );

        Ok(image)
    }

    pub fn add_mesh(
        &mut self,
        id: usize,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Result<(), RendererError> {
        if self.meshes.contains_key(&id) {
            tracing::warn!("Mesh with ID {} already exists, overwriting.", id);
        }

        let vertex_buffer_size = std::mem::size_of_val(vertices);
        let index_buffer_size = std::mem::size_of_val(indices);

        if vertex_buffer_size == 0 || index_buffer_size == 0 {
            tracing::warn!(
                "Attempted to add mesh with empty vertices or indices for ID {}",
                id
            );
            // Potentially return an error or handle as appropriate
            // For now, let's prevent creating zero-sized buffers if mev doesn't like it
            return Err(RendererError::ResourceCreation(
                "Empty mesh data".to_string(),
            ));
        }

        let vertex_buffer = self
            .queue
            .new_buffer(mev::BufferDesc {
                size: vertex_buffer_size,
                usage: mev::BufferUsage::VERTEX | mev::BufferUsage::TRANSFER_DST,
                memory: mev::Memory::Device,
                name: &format!("mesh-{}-vertex-buffer", id),
            })
            .map_err(|e| {
                RendererError::MevError(format!("Failed to create vertex buffer: {}", e))
            })?;

        let index_buffer = self
            .queue
            .new_buffer(mev::BufferDesc {
                size: index_buffer_size,
                usage: mev::BufferUsage::INDEX | mev::BufferUsage::TRANSFER_DST,
                memory: mev::Memory::Device,
                name: &format!("mesh-{}-index-buffer", id),
            })
            .map_err(|e| {
                RendererError::MevError(format!("Failed to create index buffer: {}", e))
            })?;

        // Create a temporary encoder for immediate upload
        let mut upload_encoder = self.queue.new_command_encoder().map_err(|e| {
            RendererError::MevError(format!("Failed to create upload encoder: {}", e))
        })?;

        if !vertices.is_empty() {
            upload_encoder
                .copy()
                .write_buffer_slice(vertex_buffer.slice(0..vertex_buffer_size), vertices);
        }
        if !indices.is_empty() {
            upload_encoder
                .copy()
                .write_buffer_slice(index_buffer.slice(0..index_buffer_size), indices);
        }

        // Submit immediately
        let cbuf = upload_encoder.finish().map_err(|e| {
            RendererError::MevError(format!("Failed to finish upload command buffer: {}", e))
        })?;

        self.queue
            .submit(std::iter::once(cbuf), true)
            .map_err(|e| RendererError::MevError(format!("Failed to submit mesh upload: {}", e)))?;

        self.meshes.insert(
            id,
            Mesh {
                vertex_buffer,
                index_buffer,
                index_count: indices.len() as u32,
            },
        );
        tracing::info!("Added mesh with ID: {}", id);
        Ok(())
    }

    // New: Add a material to the renderer and upload its shader data
    pub fn add_material(&mut self, id: usize, material: Material) -> Result<(), RendererError> {
        let shader_data = MaterialShaderData {
            albedo: material.albedo,
            metallic: material.metallic,
            roughness: material.roughness,
            ao: material.ao,
            _padding: [0.0],
        };

        let material_offset = id * std::mem::size_of::<MaterialShaderData>();
        let material_slice =
            self.material_uniform_buffer.as_ref().unwrap().slice(
                material_offset..(material_offset + std::mem::size_of::<MaterialShaderData>()),
            );

        // Use the persistent asset_upload_encoder
        let encoder = self.asset_upload_encoder.as_mut().unwrap();
        encoder
            .copy()
            .write_buffer_slice(material_slice, &[shader_data]);

        self.materials.insert(id, material);
        tracing::info!("Added material with ID: {}", id);
        Ok(())
    }

    // New: Add a texture to the renderer
    pub fn add_texture(
        &mut self,
        id: usize,
        data: &[u8],
        width: u32,
        height: u32,
        format: mev::PixelFormat,
    ) -> Result<(), RendererError> {
        let mev_image = self
            .create_texture_from_data(data, width, height, format, &format!("texture-{}", id))
            .unwrap();
        self.textures.insert(
            id,
            TextureHandle {
                mev_image,
                format,
                width,
                height,
            },
        );
        tracing::info!("Added texture with ID: {}", id);
        Ok(())
    }

    fn create_or_update_pipeline(
        queue: &mev::Queue,
        shader_library: &Option<mev::Library>,
        pipeline: &mut Option<mev::RenderPipeline>,
        depth_texture: &mut Option<mev::Image>,
        last_format: &mut Option<mev::PixelFormat>,
        target_format: mev::PixelFormat,
        target_extent: mev::Extent2, // Pass extent to re-create depth texture
    ) -> Result<(), mev::DeviceError> {
        if pipeline.is_none() || *last_format != Some(target_format) {
            let library = shader_library.as_ref().unwrap();

            // Re-create depth texture with current frame dimensions
            *depth_texture = Some(queue.new_image(mev::ImageDesc {
                extent: target_extent.into(),
                format: mev::PixelFormat::D32Float,
                usage: mev::ImageUsage::SAMPLED, // Depth textures can also be sampled for future effects
                layers: 1,
                levels: 1,
                name: "depth-buffer",
            })?);

            *pipeline = Some(
                queue
                    .new_render_pipeline(mev::RenderPipelineDesc {
                        name: "pbr-pipeline",
                        vertex_shader: mev::Shader {
                            library: library.clone(),
                            entry: "vs_main".into(),
                        },
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
                            fragment_shader: Some(mev::Shader {
                                library: library.clone(),
                                entry: "fs_main".into(),
                            }),
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
        }

        Ok(())
    }

    fn update_uniforms(
        queue: &mut mev::Queue,
        camera_buffer: &mev::Buffer,
        lights_buffer: &mev::Buffer,
        render_data: &RenderData,
    ) -> Result<(), mev::DeviceError> {
        // Update camera uniforms using glam
        let camera_transform = &render_data.camera_transform;
        let eye: Vec3 = camera_transform.position.into();
        let center: Vec3 = (eye + camera_transform.forward()).into(); // Look forward from camera
        let up: Vec3 = camera_transform.up().into();

        let view_matrix = Mat4::look_at_rh(eye, center, up); // Right-handed view matrix
        let projection_matrix = Mat4::perspective_rh(
            45.0_f32.to_radians(),
            800.0 / 600.0, // Aspect ratio
            0.1,
            100.0,
        );

        let camera_uniforms = CameraUniforms {
            view_matrix: view_matrix.to_cols_array_2d(),
            projection_matrix: projection_matrix.to_cols_array_2d(),
            view_position: camera_transform.position.to_array(),
            _padding: 0.0,
        };

        // Command encoder for uniform updates
        let mut encoder = queue.new_command_encoder()?;
        {
            let mut copy_encoder = encoder.copy();
            copy_encoder.write_buffer(camera_buffer, &camera_uniforms.as_repr());

            // Update lights buffer
            let light_data: Vec<<LightData as DeviceRepr>::Repr> = render_data
                .lights
                .iter()
                .map(|light| {
                    let direction: Vec3 = match light.light_type {
                        LightType::Directional => light.transform.forward().into(),
                        _ => Vec3::ZERO, // Point and Spot lights don't use a global direction like directional
                    };
                    LightData {
                        position: light.transform.position.to_array(),
                        light_type: match light.light_type {
                            LightType::Directional => 0,
                            LightType::Point => 1,
                            LightType::Spot { angle: _ } => 2,
                        },
                        color: light.color,
                        intensity: light.intensity,
                        direction: direction.to_array(),
                        _padding: 0.0,
                    }
                    .as_repr()
                })
                .collect();

            // Ensure the lights buffer is large enough
            if !light_data.is_empty() {
                // If you have more lights than the buffer size allows, you might need to handle this
                // e.g., reallocate the buffer or cap the number of lights
                copy_encoder.write_buffer_slice(lights_buffer.slice(0..), &light_data);
            }
        }
        let cbuf = encoder.finish()?;
        queue.submit(std::iter::once(cbuf), true)?;

        Ok(())
    }

    // Use glam for matrix operations
    fn create_model_matrix(transform: &Transform) -> [[f32; 4]; 4] {
        let position: Vec3 = transform.position.into();
        let rotation = transform.rotation.into();
        let scale: Vec3 = transform.scale.into();

        Mat4::from_scale_rotation_translation(scale, rotation, position).to_cols_array_2d()
    }

    // Use glam for inverse transpose for normal matrix
    fn create_normal_matrix(model_matrix: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let model_mat4 = Mat4::from_cols_array_2d(model_matrix);
        // For uniform scaling, we can use just the upper 3x3 part
        // For non-uniform scaling, we need the inverse transpose of the 3x3 part
        let upper_3x3 = Mat3::from_mat4(model_mat4);
        let normal_mat3 = upper_3x3.inverse().transpose();
        Mat4::from_mat3(normal_mat3).to_cols_array_2d()
    }

    pub fn update_render_data(&mut self, render_data: RenderData) {
        self.current_render_data = Some(render_data);
    }

    pub fn render(&mut self, window: &Window) -> Result<(), RendererError> {
        self.frame_count += 1;

        if self.shader_library.is_none() {
            self.initialize_resources().map_err(|e| {
                RendererError::MevError(format!("Failed to initialize resources: {}", e))
            })?;
        }

        let Some(ref render_data) = self.current_render_data else {
            return Ok(());
        };

        // Submit any pending asset uploads before rendering
        if let Some(encoder) = self.asset_upload_encoder.take() {
            let cbuf = encoder.finish().map_err(|e| {
                RendererError::MevError(format!(
                    "Failed to finish asset upload command buffer: {}",
                    e
                ))
            })?;
            self.queue
                .submit(std::iter::once(cbuf), true)
                .map_err(|e| {
                    RendererError::MevError(format!("Failed to submit asset uploads: {}", e))
                })?;
            // Re-create the encoder for the next frame's asset uploads
            self.asset_upload_encoder = Some(self.queue.new_command_encoder().map_err(|e| {
                RendererError::MevError(format!(
                    "Failed to re-create asset upload command encoder: {}",
                    e
                ))
            })?);
        }

        let mut frame = self
            .surface
            .next_frame()
            .map_err(|e| RendererError::MevError(format!("Failed to get next frame: {}", e)))?;

        let target_format = frame.image().format();
        let target_extent = frame.image().extent().expect_2d();

        Self::create_or_update_pipeline(
            &self.queue,
            &self.shader_library,
            &mut self.pipeline,
            &mut self.depth_texture,
            &mut self.last_format,
            target_format,
            target_extent,
        )
        .map_err(|e| RendererError::MevError(format!("Failed to create pipeline: {}", e)))?;

        Self::update_uniforms(
            &mut self.queue,
            self.camera_buffer.as_ref().unwrap(),
            self.lights_buffer.as_ref().unwrap(),
            &render_data,
        )
        .map_err(|e| RendererError::MevError(format!("Failed to update uniforms: {}", e)))?;

        let mut encoder = self.queue.new_command_encoder().map_err(|e| {
            RendererError::MevError(format!("Failed to create command encoder: {}", e))
        })?;

        encoder.init_image(
            mev::PipelineStages::empty(),
            mev::PipelineStages::FRAGMENT_SHADER,
            frame.image(),
        );

        // Render pass
        {
            let mut render = encoder.render(mev::RenderPassDesc {
                name: "main-pass",
                color_attachments: &[
                    mev::AttachmentDesc::new(frame.image()).clear(mev::ClearColor::DARK_GRAY)
                ],
                depth_stencil_attachment: Some(
                    mev::AttachmentDesc::new(self.depth_texture.as_ref().unwrap())
                        .clear(mev::ClearDepthStencil::default()),
                ),
            });

            render.with_viewport(mev::Offset3::ZERO, target_extent.to_3d().cast_as_f32());
            render.with_scissor(mev::Offset2::ZERO, target_extent);
            render.with_pipeline(self.pipeline.as_ref().unwrap());

            // Bind global resources (camera, lights, material buffer, sampler)
            render.with_arguments(
                0,
                &PbrArguments {
                    camera_buffer: self.camera_buffer.as_ref().unwrap().clone(),
                    lights_buffer: self.lights_buffer.as_ref().unwrap().clone(),
                    material_buffer: self.material_uniform_buffer.as_ref().unwrap().clone(), // Bind the material uniform buffer
                    albedo_texture: self.default_albedo_tex.as_ref().unwrap().clone(), // These will be overwritten per object
                    normal_texture: self.default_normal_tex.as_ref().unwrap().clone(), // These will be overwritten per object
                    metallic_roughness_texture: self
                        .default_metallic_roughness_tex
                        .as_ref()
                        .unwrap()
                        .clone(), // These will be overwritten per object
                    sampler: self.sampler.as_ref().unwrap().clone(),
                },
            );

            // Render each object
            for object in &render_data.objects {
                let Some(mesh) = self.meshes.get(&object.mesh_id) else {
                    tracing::warn!(
                        "Mesh with ID {} not found, skipping object.",
                        object.mesh_id
                    );
                    continue;
                };
                let Some(material) = self.materials.get(&object.material_id) else {
                    tracing::warn!(
                        "Material with ID {} not found, skipping object.",
                        object.material_id
                    );
                    continue;
                };

                let model_matrix = Self::create_model_matrix(&object.transform);
                let normal_matrix = Self::create_normal_matrix(&model_matrix);

                render.with_constants(&PbrConstants {
                    model_matrix,
                    normal_matrix,
                    material_id: object.material_id as u32,
                });

                // Dynamically bind textures for the current material
                let albedo_texture = material
                    .albedo_texture_id
                    .and_then(|id| self.textures.get(&id))
                    .map(|h| h.mev_image.clone())
                    .unwrap_or_else(|| self.default_albedo_tex.as_ref().unwrap().clone());

                let normal_texture = material
                    .normal_texture_id
                    .and_then(|id| self.textures.get(&id))
                    .map(|h| h.mev_image.clone())
                    .unwrap_or_else(|| self.default_normal_tex.as_ref().unwrap().clone());

                let metallic_roughness_texture = material
                    .metallic_roughness_texture_id
                    .and_then(|id| self.textures.get(&id))
                    .map(|h| h.mev_image.clone())
                    .unwrap_or_else(|| {
                        self.default_metallic_roughness_tex
                            .as_ref()
                            .unwrap()
                            .clone()
                    });

                // Rebind arguments with the specific textures for this object
                render.with_arguments(
                    0,
                    &PbrArguments {
                        camera_buffer: self.camera_buffer.as_ref().unwrap().clone(),
                        lights_buffer: self.lights_buffer.as_ref().unwrap().clone(),
                        material_buffer: self.material_uniform_buffer.as_ref().unwrap().clone(),
                        albedo_texture,
                        normal_texture,
                        metallic_roughness_texture,
                        sampler: self.sampler.as_ref().unwrap().clone(),
                    },
                );

                // Bind mesh buffers and draw
                render.bind_vertex_buffers(0, &[mesh.vertex_buffer.clone()]);
                render.bind_index_buffer(&mesh.index_buffer);
                render.draw_indexed(0, 0..mesh.index_count, 0..1);
            }
        }

        self.queue
            .sync_frame(&mut frame, mev::PipelineStages::FRAGMENT_SHADER);
        encoder.present(frame, mev::PipelineStages::FRAGMENT_SHADER);

        let cbuf = encoder.finish().map_err(|e| {
            RendererError::MevError(format!("Failed to finish command buffer: {}", e))
        })?;

        window.pre_present_notify();
        self.queue
            .submit([cbuf], true)
            .map_err(|e| RendererError::MevError(format!("Failed to submit commands: {}", e)))?;

        tracing::debug!(
            "Rendered frame {} with {} objects and {} lights",
            self.frame_count,
            render_data.objects.len(),
            render_data.lights.len()
        );

        Ok(())
    }
}
