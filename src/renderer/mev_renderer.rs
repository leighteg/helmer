use crate::ecs::ecs_core::ECSCore;

pub struct PrimitiveRenderer {
    pub queue: mev::Queue,
    pub surface: Option<mev::Surface>,
    pub window: Option<winit::window::Window>,
    pub pipeline: Option<mev::RenderPipeline>,
    pub last_format: Option<mev::PixelFormat>,
    pub start: std::time::Instant,
}

impl PrimitiveRenderer {
    pub fn render(&mut self, ecs: &ECSCore) {
        let Some(surface) = self.surface.as_mut() else { return };
        let mut frame = surface.next_frame().unwrap();
        let format = frame.image().format();
        let extent = frame.image().extent();

        let angle = self.start.elapsed().as_secs_f32() * 0.1;

        if self.pipeline.is_none() || self.last_format != Some(format) {
            let library = self.queue.new_shader_library(mev::LibraryDesc {
                name: "main",
                input: mev::include_library!(
                    "shaders/triangle.wgsl" as mev::ShaderLanguage::Wgsl
                ),
            }).unwrap();

            let pipeline = self.queue.new_render_pipeline(mev::RenderPipelineDesc {
                name: "main",
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
                        format,
                        blend: Some(mev::BlendDesc::default()),
                    }],
                    depth_stencil: None,
                    front_face: mev::FrontFace::default(),
                    culling: mev::Culling::Back,
                }),
                arguments: &[],
                constants: TriangleConstants::SIZE,
            }).unwrap();

            self.pipeline = Some(pipeline);
            self.last_format = Some(format);
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let mut encoder = self.queue.new_command_encoder().unwrap();
        encoder.init_image(mev::PipelineStages::empty(), mev::PipelineStages::FRAGMENT_SHADER, frame.image());

        {
            let mut render = encoder.render(mev::RenderPassDesc {
                name: "main",
                color_attachments: &[mev::AttachmentDesc::new(frame.image()).clear(mev::ClearColor::DARK_GRAY)],
                depth_stencil_attachment: None,
            });

            render.with_viewport(mev::Offset3::ZERO, extent.into_3d().cast_as_f32());
            render.with_scissor(mev::Offset2::ZERO, extent.into_2d());
            render.with_pipeline(pipeline);
            render.with_constants(&TriangleConstants {
                angle,
                width: extent.width(),
                height: extent.height(),
            });

            // Example: loop over all entities with RectRenderer + Position + Scale
            let rects = ecs.get_components_by_type::<RectRenderer>();
            let positions = ecs.get_components_by_type::<Position>();
            let scales = ecs.get_components_by_type::<Scale>();

            for ((rect, pos), scale) in rects.iter().zip(positions.iter()).zip(scales.iter()) {
                // TODO: You can store vertex buffers or draw via push constants.
                render.draw(0..3, 0..1);
            }
        }

        self.queue.sync_frame(&mut frame, mev::PipelineStages::FRAGMENT_SHADER);
        encoder.present(frame, mev::PipelineStages::FRAGMENT_SHADER);

        let cbuf = encoder.finish().unwrap();
        self.window.as_ref().unwrap().pre_present_notify();
        self.queue.submit([cbuf], true).unwrap();
    }
}