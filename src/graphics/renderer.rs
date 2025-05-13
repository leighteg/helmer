use glam::Mat4;
use hashbrown::HashMap;
use mev::mat4;
use winit::{event_loop::ActiveEventLoop, window::Window};

pub enum ShaderType {
    Basic,
    Texture,
}

pub struct RenderItem {
    pipeline_id: usize,
    mesh_id: usize,
    material_id: usize,
    world_matrix: Mat4,
}

pub struct Renderer {
    queue: mev::Queue,
    surface: Option<mev::Surface>,
    last_format: Option<mev::PixelFormat>,
    pipelines: HashMap<ShaderType, mev::RenderPipeline>,
    initialized: bool,

    render_items: HashMap<usize, RenderItem>,
}

impl Renderer {
    pub fn new() -> Self {
        // init
        
        let instance = mev::Instance::load().expect("failed to init graphics (mev)");

        let (_device, mut queues) = instance.new_device(mev::DeviceDesc {
            idx: 0,
            queues: &[0],
            features: mev::Features::SURFACE,
        }).unwrap();
        let queue = queues.pop().unwrap();

        // return fresh instance

        Self {
            queue,
            surface: None,
            last_format: None,
            pipelines: HashMap::new(),
            initialized: false,

            render_items: HashMap::new(),
        }
    }

    pub fn new_item(&mut self) {}

    pub fn draw_frame(&mut self, window: &mut Window) {
        if !self.initialized || self.surface.is_none() {
            self.surface = Some(self.queue.new_surface(&window, &window).unwrap());
            self.initialized = true;
        }

        let mut frame = self.surface.as_mut().unwrap().next_frame().unwrap();

        // Begin command encoding
        let mut encoder = self.queue.new_command_encoder().unwrap();
        encoder.init_image(
            mev::PipelineStages::empty(),
            mev::PipelineStages::FRAGMENT_SHADER,
            frame.image(),
        );

        self.queue.sync_frame(&mut frame, mev::PipelineStages::FRAGMENT_SHADER);
        encoder.present(frame, mev::PipelineStages::FRAGMENT_SHADER);
        let cbuf = encoder.finish().unwrap();
        
        window.pre_present_notify();
        self.queue.submit([cbuf], true).unwrap();

        window.request_redraw();
    }
}