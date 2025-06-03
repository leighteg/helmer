use std::sync::Arc;
use mev::{Arguments, DeviceRepr};
use winit::window::Window;
use crate::{
    graphics::renderer::{device::RenderDevice, error::RendererError},
    provided::components::{LightType, Transform},
};

#[derive(Debug, Clone)]
pub struct RenderObject {
    pub transform: Transform,
    pub mesh_id: u32,
    pub material_id: u32,
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
    #[mev(vertex, fragment)]
    lights_buffer: mev::Buffer,
    #[mev(fragment)]
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

// Material data structure
#[derive(mev::DeviceRepr)]
struct MaterialData {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    ao: f32,
    _padding: f32,
}

// Vertex data structure
#[derive(mev::DeviceRepr)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coord: [f32; 2],
    tangent: [f32; 3],
}

pub struct Renderer {
    device: RenderDevice,
    surface: mev::Surface,
    queue: mev::Queue,
    
    // Rendering resources
    shader_library: Option<mev::Library>,
    pipeline: Option<mev::RenderPipeline>,
    depth_texture: Option<mev::Image>,
    
    // Uniform buffers
    camera_buffer: Option<mev::Buffer>,
    lights_buffer: Option<mev::Buffer>,
    material_buffer: Option<mev::Buffer>,
    
    // Vertex and index buffers
    vertex_buffer: Option<mev::Buffer>,
    index_buffer: Option<mev::Buffer>,
    
    // Textures and samplers
    default_albedo: Option<mev::Image>,
    default_normal: Option<mev::Image>,
    default_metallic_roughness: Option<mev::Image>,
    sampler: Option<mev::Sampler>,
    
    frame_count: u64,
    current_render_data: Option<RenderData>,
    last_format: Option<mev::PixelFormat>,
}

impl Renderer {
    pub fn new(window: &Window) -> Result<Self, RendererError> {
        // Initialize MEV instance and device
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
            device: RenderDevice::new()?,
            surface,
            queue,
            shader_library: None,
            pipeline: None,
            depth_texture: None,
            camera_buffer: None,
            lights_buffer: None,
            material_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
            default_albedo: None,
            default_normal: None,
            default_metallic_roughness: None,
            sampler: None,
            frame_count: 0,
            current_render_data: None,
            last_format: None,
        })
    }
    
    fn initialize_resources(&mut self) -> Result<(), mev::DeviceError> {
        // Create shader library
        self.shader_library = Some(self.queue.new_shader_library(mev::LibraryDesc {
            name: "pbr",
            input: mev::include_library!(
                "../shaders/pbr.wgsl" as mev::ShaderLanguage::Wgsl
            ),
        }).unwrap());
        
        // Create default textures
        self.create_default_textures()?;
        
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
            usage: mev::BufferUsage::UNIFORM | mev::BufferUsage::TRANSFER_DST,
            memory: mev::Memory::Device,
            name: "lights-buffer",
        })?);
        
        self.material_buffer = Some(self.queue.new_buffer(mev::BufferDesc {
            size: std::mem::size_of::<MaterialData>() * 256, // Support up to 256 materials
            usage: mev::BufferUsage::UNIFORM | mev::BufferUsage::TRANSFER_DST,
            memory: mev::Memory::Device,
            name: "materials-buffer",
        })?);
        
        Ok(())
    }
    
    fn create_default_textures(&mut self) -> Result<(), mev::DeviceError> {
        // Create default white albedo texture
        let white_pixel = [255u8, 255, 255, 255];
        self.default_albedo = Some(self.create_texture_from_data(
            &white_pixel,
            1, 1,
            mev::PixelFormat::Rgba8Srgb,
            "default-albedo"
        )?);
        
        // Create default normal texture (flat normal: 128, 128, 255, 255)
        let normal_pixel = [128u8, 128, 255, 255];
        self.default_normal = Some(self.create_texture_from_data(
            &normal_pixel,
            1, 1,
            mev::PixelFormat::Rgba8Unorm,
            "default-normal"
        )?);
        
        // Create default metallic-roughness texture (no metallic, medium roughness: 0, 128, 0, 255)
        let mr_pixel = [0u8, 128, 0, 255];
        self.default_metallic_roughness = Some(self.create_texture_from_data(
            &mr_pixel,
            1, 1,
            mev::PixelFormat::Rgba8Unorm,
            "default-metallic-roughness"
        )?);
        
        Ok(())
    }
    
    fn create_texture_from_data(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        format: mev::PixelFormat,
        name: &str
    ) -> Result<mev::Image, mev::DeviceError> {
        let image = self.queue.new_image(mev::ImageDesc {
            extent: mev::Extent2::new(width, height).into(),
            format,
            usage: mev::ImageUsage::SAMPLED | mev::ImageUsage::TRANSFER_DST,
            layers: 1,
            levels: 1,
            name,
        })?;
        
        // Upload texture data
        let mut encoder = self.queue.new_command_encoder()?;
        {
            let mut copy_encoder = encoder.copy();
            copy_encoder.init_image(
                mev::PipelineStages::empty(),
                mev::PipelineStages::TRANSFER,
                &image,
            );
        }
        let cbuf = encoder.finish()?;
        self.queue.submit(std::iter::once(cbuf), true)?;
        
        Ok(image)
    }
    
    fn create_or_update_pipeline(
        queue: &mev::Queue,
        shader_library: &Option<mev::Library>,
        pipeline: &mut Option<mev::RenderPipeline>,
        depth_texture: &mut Option<mev::Image>,
        last_format: &mut Option<mev::PixelFormat>,
        target_format: mev::PixelFormat
    ) -> Result<(), mev::DeviceError> {
        if pipeline.is_none() || *last_format != Some(target_format) {
            let library = shader_library.as_ref().unwrap();
            
            // Create depth texture if needed
            if depth_texture.is_none() {
                // This will be recreated with proper dimensions during render
                *depth_texture = Some(queue.new_image(mev::ImageDesc {
                    extent: mev::Extent2::new(1, 1).into(),
                    format: mev::PixelFormat::D32Float,
                    usage: mev::ImageUsage::SAMPLED | mev::ImageUsage::TRANSFER_DST,
                    layers: 1,
                    levels: 1,
                    name: "depth-buffer",
                })?);
            }
            
            *pipeline = Some(queue.new_render_pipeline(mev::RenderPipelineDesc {
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
            }).unwrap());
            
            *last_format = Some(target_format);
        }
        
        Ok(())
    }
    
    fn update_uniforms(
        queue: &mut mev::Queue,
        camera_buffer: &mev::Buffer,
        lights_buffer: &mev::Buffer,
        render_data: &RenderData
    ) -> Result<(), mev::DeviceError> {
        // Update camera uniforms
        let camera_uniforms = CameraUniforms {
            view_matrix: Self::create_view_matrix(&render_data.camera_transform),
            projection_matrix: Self::create_projection_matrix(800, 600),
            view_position: render_data.camera_transform.position,
            _padding: 0.0,
        };
        
        let mut encoder = queue.new_command_encoder()?;
        {
            let mut copy_encoder = encoder.copy();
            copy_encoder.write_buffer(
                camera_buffer,
                &camera_uniforms.as_repr(),
            );
            
            // Update lights buffer
            let light_data: Vec<<LightData as DeviceRepr>::Repr> = render_data.lights.iter().map(|light| {
                LightData {
                    position: light.transform.position,
                    light_type: match light.light_type {
                        LightType::Directional => 0,
                        LightType::Point => 1,
                        LightType::Spot {angle: _} => 2,
                    },
                    color: light.color,
                    intensity: light.intensity,
                    direction: [0.0, -1.0, 0.0],
                    _padding: 0.0,
                }.as_repr()
            }).collect();
            
            if !light_data.is_empty() {
                copy_encoder.write_buffer_slice(
                    lights_buffer.slice(0..),
                    &light_data,
                );
            }
        }
        let cbuf = encoder.finish()?;
        queue.submit(std::iter::once(cbuf), true)?;
        
        Ok(())
    }    
    
    fn create_view_matrix(camera_transform: &Transform) -> [[f32; 4]; 4] {
        // Simple view matrix creation - in a real implementation you'd use a proper math library
        let eye = camera_transform.position;
        let center = [0.0, 0.0, 0.0]; // Look at origin
        let up = [0.0, 1.0, 0.0];
        
        // This is a simplified look-at matrix - you should use a proper math library
        [
            [1.0, 0.0, 0.0, -eye[0]],
            [0.0, 1.0, 0.0, -eye[1]],
            [0.0, 0.0, 1.0, -eye[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
    
    fn create_projection_matrix(width: u32, height: u32) -> [[f32; 4]; 4] {
        let aspect = width as f32 / height as f32;
        let fov = 45.0_f32.to_radians();
        let near = 0.1;
        let far = 100.0;
        
        let f = 1.0 / (fov / 2.0).tan();
        
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }
    
    fn create_model_matrix(&self, transform: &Transform) -> [[f32; 4]; 4] {
        // Simple model matrix from transform
        [
            [transform.scale[0], 0.0, 0.0, transform.position[0]],
            [0.0, transform.scale[1], 0.0, transform.position[1]],
            [0.0, 0.0, transform.scale[2], transform.position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
    
    pub fn update_render_data(&mut self, render_data: RenderData) {
        self.current_render_data = Some(render_data);
    }
    
    pub fn render(&mut self, window: &Window) -> Result<(), RendererError> {
        self.frame_count += 1;
        
        if self.shader_library.is_none() {
            self.initialize_resources()
                .map_err(|e| RendererError::MevError(format!("Failed to initialize resources: {}", e)))?;
        }
        
        let Some(ref render_data) = self.current_render_data else {
            return Ok(());
        };
        
        let mut frame = self.surface.next_frame()
            .map_err(|e| RendererError::MevError(format!("Failed to get next frame: {}", e)))?;
        
        let target_format = frame.image().format();
        let target_extent = frame.image().extent().expect_2d();
        
        Self::create_or_update_pipeline(
            &self.queue,
            &self.shader_library,
            &mut self.pipeline,
            &mut self.depth_texture,
            &mut self.last_format,
            target_format
        ).map_err(|e| RendererError::MevError(format!("Failed to create pipeline: {}", e)))?;
        
        Self::update_uniforms(
            &mut self.queue,
            self.camera_buffer.as_ref().unwrap(),
            self.lights_buffer.as_ref().unwrap(),
            &render_data
        ).map_err(|e| RendererError::MevError(format!("Failed to update uniforms: {}", e)))?;
        
        let mut encoder = self.queue.new_command_encoder()
            .map_err(|e| RendererError::MevError(format!("Failed to create command encoder: {}", e)))?;
        
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
                    mev::AttachmentDesc::new(self.depth_texture.as_ref().unwrap()).clear(mev::ClearDepthStencil::default())
                ),
            });
            
            render.with_viewport(
                mev::Offset3::ZERO,
                target_extent.to_3d().cast_as_f32(),
            );
            render.with_scissor(mev::Offset2::ZERO, target_extent);
            render.with_pipeline(self.pipeline.as_ref().unwrap());
            
            // Bind global resources
            render.with_arguments(0, &PbrArguments {
                camera_buffer: self.camera_buffer.as_ref().unwrap().clone(),
                lights_buffer: self.lights_buffer.as_ref().unwrap().clone(),
                material_buffer: self.material_buffer.as_ref().unwrap().clone(),
                albedo_texture: self.default_albedo.as_ref().unwrap().clone(),
                normal_texture: self.default_normal.as_ref().unwrap().clone(),
                metallic_roughness_texture: self.default_metallic_roughness.as_ref().unwrap().clone(),
                sampler: self.sampler.as_ref().unwrap().clone(),
            });
            
            // Render each object
            for object in &render_data.objects {
                let model_matrix = self.create_model_matrix(&object.transform);
                let normal_matrix = model_matrix; // Simplified - should be inverse transpose
                
                render.with_constants(&PbrConstants {
                    model_matrix,
                    normal_matrix,
                    material_id: object.material_id,
                });
                
                // In a real implementation, you'd bind the actual mesh vertex/index buffers here
                // For now, we'll render a simple triangle as placeholder
                render.draw(0..3, 0..1);
            }
        }
        
        self.queue.sync_frame(&mut frame, mev::PipelineStages::FRAGMENT_SHADER);
        encoder.present(frame, mev::PipelineStages::FRAGMENT_SHADER);
        
        let cbuf = encoder.finish()
            .map_err(|e| RendererError::MevError(format!("Failed to finish command buffer: {}", e)))?;
        
        window.pre_present_notify();
        self.queue.submit([cbuf], true)
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