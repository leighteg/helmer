use std::time::Instant;
use std::collections::HashMap;
use mev::DeviceRepr;
use proc::Component;
use crate::ecs::component::Component;
use crate::ecs::ecs_core::ECSCore;
use crate::ecs::system::System;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};
use winit::event::WindowEvent;

// Components for rendering entities
#[derive(Clone, Debug, Component)]
pub struct RenderableComponent {
    pub visible: bool,
    pub shader_type: ShaderType,
    // Add more rendering properties as needed
}

// Adding required traits for HashMap key
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Triangle,
    Quad,
    Custom(String),
}

// Transform component for positioning
#[derive(Clone, Debug, Component)]
pub struct TransformComponent {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
}

// Main renderer that holds the mev graphics state
pub struct Renderer {
    queue: mev::Queue,
    window: Option<Window>,
    surface: Option<mev::Surface>,
    last_format: Option<mev::PixelFormat>,
    pipelines: HashMap<ShaderType, mev::RenderPipeline>,
    start_time: Instant,
    pub initialized: bool,
}

impl Renderer {
    pub fn new() -> Self {
        // Initialize mev
        let instance = mev::Instance::load().expect("Failed to initialize graphics");

        let (_device, mut queues) = instance
            .new_device(mev::DeviceDesc {
                idx: 0,
                queues: &[0],
                features: mev::Features::SURFACE,
            })
            .unwrap();
        
        let queue = queues.pop().unwrap();

        Self {
            queue,
            window: None,
            surface: None,
            last_format: None,
            pipelines: HashMap::new(),
            start_time: Instant::now(),
            initialized: false,
        }
    }

    pub fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(Window::default_attributes())
                .unwrap();
            
            let surface = self.queue.new_surface(&window, &window).unwrap();

            self.window = Some(window);
            self.surface = Some(surface);
            self.initialized = true;
        }

        self.window.as_ref().unwrap().request_redraw();
    }

    fn create_pipeline(&mut self, shader_type: &ShaderType, target_format: mev::PixelFormat) -> mev::RenderPipeline {
        // Choose the appropriate shader path based on shader type
        let shader_path = match shader_type {
            ShaderType::Triangle => "shaders/triangle.wgsl",
            ShaderType::Quad => "shaders/quad.wgsl",
            ShaderType::Custom(name) => name,
        };

        // Fixed macro call - using literal string directly
        let library = self
            .queue
            .new_shader_library(mev::LibraryDesc {
                name: "shader",
                input: mev::include_library!(
                    "shaders/triangle.wgsl" as mev::ShaderLanguage::Wgsl
                ), // Using direct literal instead of variable
            })
            .unwrap();

        self.queue
            .new_render_pipeline(mev::RenderPipelineDesc {
                name: "pipeline",
                vertex_shader: mev::Shader {
                    library: library.clone(),
                    entry: "vs_main".into(),
                },
                vertex_attributes: vec![],
                vertex_layouts: vec![],
                primitive_topology: mev::PrimitiveTopology::Triangle,
                raster: Some(mev::RasterDesc {
                    fragment_shader: Some(mev::Shader {
                        library,
                        entry: "fs_main".into(),
                    }),
                    color_targets: vec![mev::ColorTargetDesc {
                        format: target_format,
                        blend: Some(mev::BlendDesc::default()),
                    }],
                    depth_stencil: None,
                    front_face: mev::FrontFace::default(),
                    culling: mev::Culling::Back,
                }),
                arguments: &[],
                constants: RenderConstants::SIZE,
            })
            .unwrap()
    }

    pub fn render(&mut self, ecs: &ECSCore) {
        if !self.initialized || self.surface.is_none() {
            return;
        }

        let mut frame = self.surface.as_mut().unwrap().next_frame().unwrap();
        let target_format = frame.image().format();
        let target_extent = frame.image().extent();
        let elapsed_time = self.start_time.elapsed().as_secs_f32();

        // Check if we need to create or update pipelines
        if self.last_format != Some(target_format) {
            // Clear out old pipelines
            self.pipelines.clear();
            
            // Create default pipelines
            let triangle_pipeline = self.create_pipeline(&ShaderType::Triangle, target_format);
            self.pipelines.insert(ShaderType::Triangle, triangle_pipeline);
            
            let quad_pipeline = self.create_pipeline(&ShaderType::Quad, target_format);
            self.pipelines.insert(ShaderType::Quad, quad_pipeline);
            
            self.last_format = Some(target_format);
        }

        // Begin command encoding
        let mut encoder = self.queue.new_command_encoder().unwrap();
        encoder.init_image(
            mev::PipelineStages::empty(),
            mev::PipelineStages::FRAGMENT_SHADER,
            frame.image(),
        );

        {
            let mut render = encoder.render(mev::RenderPassDesc {
                name: "main",
                color_attachments: &[
                    mev::AttachmentDesc::new(frame.image()).clear(mev::ClearColor::DARK_GRAY)
                ],
                depth_stencil_attachment: None,
            });

            render.with_viewport(mev::Offset3::ZERO, target_extent.into_3d().cast_as_f32());
            render.with_scissor(mev::Offset2::ZERO, target_extent.into_2d());

            // Loop through all renderable entities and draw them
            let renderable_entities = ecs.get_all_components_of_type::<RenderableComponent>();
            
            for renderable in renderable_entities {
                if !renderable.visible {
                    continue;
                }
                
                // Get the appropriate pipeline
                if let Some(pipeline) = self.pipelines.get(&renderable.shader_type) {
                    render.with_pipeline(pipeline);
                    
                    // Set up constants for this draw
                    render.with_constants(&RenderConstants {
                        time: elapsed_time,
                        width: target_extent.width(),
                        height: target_extent.height(),
                    });
                    
                    // Draw the object
                    render.draw(0..3, 0..1);  // Adjust ranges based on your geometry
                }
            }
        }

        self.queue
            .sync_frame(&mut frame, mev::PipelineStages::FRAGMENT_SHADER);
        encoder.present(frame, mev::PipelineStages::FRAGMENT_SHADER);
        let cbuf = encoder.finish().unwrap();

        self.window.as_ref().unwrap().pre_present_notify();
        self.queue.submit([cbuf], true).unwrap();
    }

    pub fn handle_window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent, ecs: &mut ECSCore) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.render(ecs);
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }

    pub fn get_window(&self) -> Option<&Window> {
        self.window.as_ref()
    }
}

// Constants passed to shaders
#[derive(mev::DeviceRepr)]
pub struct RenderConstants {
    pub time: f32,
    pub width: u32,
    pub height: u32,
}

// Implement Clone for RenderSystem as required in runtime integration
#[derive(Clone)]
pub struct RenderSystem {
    renderer: Option<std::sync::Arc<std::sync::Mutex<Renderer>>>, // Use Arc and Mutex for cloning
    event_loop: Option<std::sync::Arc<EventLoop<()>>>,
}

impl RenderSystem {
    pub fn new() -> Self {
        Self {
            renderer: None,
            event_loop: None,
        }
    }

    pub fn set_event_loop(&mut self, event_loop: std::sync::Arc<EventLoop<()>>) {
        self.event_loop = Some(event_loop.clone());
        
        // Initialize the renderer with the event loop
        let renderer = Renderer::new();
        self.renderer = Some(std::sync::Arc::new(std::sync::Mutex::new(renderer)));
    }
    
    pub fn get_renderer(&self) -> Option<std::sync::MutexGuard<Renderer>> {
        self.renderer.as_ref().map(|r| r.lock().unwrap())
    }
    
    pub fn get_renderer_mut(&mut self) -> Option<std::sync::MutexGuard<Renderer>> {
        self.renderer.as_ref().map(|r| r.lock().unwrap())
    }
}

impl System for RenderSystem {
    fn name(&self) -> &str {
        "RenderSystem"
    }

    fn run(&mut self, ecs: &mut ECSCore) {
        if let Some(renderer) = &self.renderer {
            let mut renderer_guard = renderer.lock().unwrap();
            renderer_guard.render(ecs);
        }
    }
}