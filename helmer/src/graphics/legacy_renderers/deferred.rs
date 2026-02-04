use crate::{
    graphics::common::{
        atmosphere::AtmospherePrecomputer,
        constants::{CASCADE_SPLITS, FRAMES_IN_FLIGHT, NUM_CASCADES, SHADOW_MAP_RESOLUTION},
        error::RendererError,
        mipmap::MipmapGenerator,
        renderer::{
            Aabb, CameraUniforms, CascadeUniform, EguiRenderData, InstanceRaw, LightData, Material,
            MaterialShaderData, Mesh, MeshLod, MeshLodPayload, ModelPushConstant, PbrConstants,
            RenderData, RenderMessage, RenderTrait, ShaderConstants, ShadowPipeline,
            ShadowUniforms, SkyUniforms, TextureManager, Vertex, build_mip_uploads,
            calc_mip_level_count,
        },
        shadow_mapping::cevsm::CascadedEVSMPass,
    },
    provided::components::{Camera, LightType},
    runtime::asset_server::{AssetKind, MaterialGpuData},
};
use egui_wgpu::Renderer as EguiRenderer;
use glam::{Mat4, Quat, Vec3, Vec4Swizzles};
use hashbrown::HashMap;
use image::{GenericImageView, ImageFormat};
use std::{cell::RefCell, sync::Arc, time::Duration};
use tracing::{info, warn};
use web_time::Instant;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

// --- CONSTANTS ---
pub const MAX_LIGHTS: usize = 2048;
const MAX_TOTAL_TEXTURES: u32 = 4096;
const MAX_TOTAL_MATERIALS: usize = 4096;
const DEFAULT_TEXTURE_RESOLUTION: u32 = 1024;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GeometryInstanceRaw {
    model_matrix: [[f32; 4]; 4],
    material_id: u32,
}

impl GeometryInstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<GeometryInstanceRaw>() as wgpu::BufferAddress,
            // This buffer steps forward once per *instance*, not per vertex.
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // model_matrix (col 0-3) at locations 5-8
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5, // 0-4 are for Vertex
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // material_id at location 9
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Uint32, // Note: Uint32
                },
            ],
        }
    }
}

/// The high-end renderer using a deferred pipeline and bindless textures.
pub struct DeferredRenderer {
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    window_size: PhysicalSize<u32>,

    // generators
    mipmap_generator: MipmapGenerator,

    // precomputers
    atmosphere_precomputer: Option<AtmospherePrecomputer>,

    // modular pass pipelines
    shadow_pass: CascadedEVSMPass,

    // Maps an asset Handle ID (from AssetServer) to a final GPU texture array index
    handle_id_to_texture_index: HashMap<usize, usize>,

    // A queue for materials waiting for their textures to be loaded on the GPU
    pending_materials: Vec<MaterialGpuData>,

    // Core Pipelines
    geometry_pipeline: Option<wgpu::RenderPipeline>,
    sky_pipeline: Option<wgpu::RenderPipeline>,
    lighting_pipeline: Option<wgpu::RenderPipeline>,
    downsample_pipeline: Option<wgpu::RenderPipeline>,
    ssr_pipeline: Option<wgpu::RenderPipeline>,
    ssgi_pipeline: Option<wgpu::RenderPipeline>,
    ssgi_denoise_pipeline: Option<wgpu::RenderPipeline>,
    ssgi_upsample_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,

    // Bind Group Layouts
    scene_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    object_data_bind_group_layout: Option<wgpu::BindGroupLayout>,
    render_constants_bind_group_layout: Option<wgpu::BindGroupLayout>,
    downsample_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssr_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssgi_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssgi_blue_noise_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssgi_denoise_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssgi_upsample_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ssr_camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    lighting_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,
    sky_bind_group_layout: Option<wgpu::BindGroupLayout>,
    ibl_bind_group_layout: Option<wgpu::BindGroupLayout>,
    composite_inputs_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Bind Groups
    scene_data_bind_groups: Vec<wgpu::BindGroup>,
    object_data_bind_group: Option<wgpu::BindGroup>,
    render_constants_bind_group: Option<wgpu::BindGroup>,
    downsample_bind_group: Option<wgpu::BindGroup>,
    ssr_inputs_bind_group: Option<wgpu::BindGroup>,
    ssgi_inputs_bind_group: Option<wgpu::BindGroup>,
    ssgi_blue_noise_bind_group: Option<wgpu::BindGroup>,
    ssgi_denoise_inputs_bind_group: Option<wgpu::BindGroup>,
    ssgi_upsample_bind_group: Option<wgpu::BindGroup>,
    ssr_camera_bind_groups: Vec<wgpu::BindGroup>,
    lighting_inputs_bind_group: Option<wgpu::BindGroup>,
    sky_bind_groups: Vec<wgpu::BindGroup>,
    ibl_bind_group: Option<wgpu::BindGroup>,
    composite_inputs_bind_group: Option<wgpu::BindGroup>,

    // Main Render Targets & Depth (Full-res)
    main_depth_texture: Option<wgpu::Texture>,
    main_depth_texture_view: Option<wgpu::TextureView>,

    // G-Buffer Textures (Full-res)
    gbuf_normal_texture: Option<wgpu::Texture>,
    gbuf_albedo_texture: Option<wgpu::Texture>,
    gbuf_mra_texture: Option<wgpu::Texture>,
    gbuf_emission_texture: Option<wgpu::Texture>,
    gbuf_normal_texture_view: Option<wgpu::TextureView>,
    gbuf_albedo_texture_view: Option<wgpu::TextureView>,
    gbuf_mra_texture_view: Option<wgpu::TextureView>,
    gbuf_emission_texture_view: Option<wgpu::TextureView>,

    // Lighting & Reflection Textures (Full-res)
    direct_lighting_texture: Option<wgpu::Texture>,
    direct_lighting_texture_view: Option<wgpu::TextureView>,
    direct_lighting_diffuse_texture: Option<wgpu::Texture>,
    direct_lighting_diffuse_view: Option<wgpu::TextureView>,
    sky_texture: Option<wgpu::Texture>,
    sky_texture_view: Option<wgpu::TextureView>,
    ssr_texture: Option<wgpu::Texture>,
    ssr_texture_view: Option<wgpu::TextureView>,
    ssgi_upsampled_texture: Option<wgpu::Texture>,
    ssgi_upsampled_texture_view: Option<wgpu::TextureView>,
    history_texture: Option<wgpu::Texture>,
    history_texture_view: Option<wgpu::TextureView>,
    blue_noise_texture: Option<wgpu::Texture>,
    blue_noise_texture_view: Option<wgpu::TextureView>,
    blue_noise_sampler: Option<wgpu::Sampler>,

    // Half-Resolution Textures for SSGI
    depth_half_texture: Option<wgpu::Texture>,
    depth_half_view: Option<wgpu::TextureView>,
    normal_half_texture: Option<wgpu::Texture>,
    normal_half_view: Option<wgpu::TextureView>,
    albedo_half_texture: Option<wgpu::Texture>,
    albedo_half_view: Option<wgpu::TextureView>,
    direct_lighting_half_texture: Option<wgpu::Texture>,
    direct_lighting_half_view: Option<wgpu::TextureView>,
    direct_lighting_diffuse_half_texture: Option<wgpu::Texture>,
    direct_lighting_diffuse_half_view: Option<wgpu::TextureView>,
    ssgi_raw_half_texture: Option<wgpu::Texture>,
    ssgi_raw_half_view: Option<wgpu::TextureView>,
    ssgi_denoised_half_texture: Option<wgpu::Texture>,
    ssgi_denoised_half_view: Option<wgpu::TextureView>,
    ssgi_history_half_texture: Option<wgpu::Texture>,
    ssgi_history_half_view: Option<wgpu::TextureView>,

    // IBL Textures
    brdf_lut_texture: Option<wgpu::Texture>,
    brdf_lut_view: Option<wgpu::TextureView>,
    irradiance_map_texture: Option<wgpu::Texture>,
    irradiance_map_view: Option<wgpu::TextureView>,
    prefiltered_env_map_texture: Option<wgpu::Texture>,
    prefiltered_env_map_view: Option<wgpu::TextureView>,

    // instancing
    geometry_instance_buffer: RefCell<Option<wgpu::Buffer>>,
    geometry_instance_capacity: RefCell<usize>,

    // Buffers
    camera_buffers: Vec<wgpu::Buffer>,
    lights_buffers: Vec<wgpu::Buffer>,
    sky_uniforms_buffers: Vec<wgpu::Buffer>,
    material_uniform_buffer: Option<wgpu::Buffer>,
    render_constants_buffer: Option<wgpu::Buffer>,

    // Asset Storage
    meshes: HashMap<usize, Mesh>,
    materials: HashMap<usize, Material>,
    texture_manager: TextureManager,

    // Texture Collections
    albedo_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    normal_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    mr_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    emission_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,

    // Samplers
    pbr_sampler: Option<wgpu::Sampler>,
    gbuffer_sampler: Option<wgpu::Sampler>,
    scene_sampler: Option<wgpu::Sampler>,
    point_sampler: Option<wgpu::Sampler>,
    ibl_sampler: Option<wgpu::Sampler>,
    brdf_lut_sampler: Option<wgpu::Sampler>,

    // EGUI
    egui_renderer: EguiRenderer,
    current_egui_data: Option<EguiRenderData>,

    // State
    frame_index: usize,
    current_render_data: Option<Arc<RenderData>>,
    logic_frame_duration: Duration,
    last_timestamp: Option<Instant>,
    prev_view_proj: Mat4,
    prev_sky_uniforms: SkyUniforms,
    needs_atmosphere_precompute: bool,
}

impl DeferredRenderer {
    pub async fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        adapter: wgpu::Adapter,
        size: PhysicalSize<u32>,
        target_tickrate: f32,
    ) -> Result<Self, RendererError> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_features: wgpu::Features::FLOAT32_FILTERABLE
                    | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                    | wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                required_limits: wgpu::Limits {
                    max_immediate_size: 0,
                    max_binding_array_elements_per_shader_stage: MAX_TOTAL_TEXTURES,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .map_err(|e| {
                RendererError::ResourceCreation(format!("Failed to create device: {}", e))
            })?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mipmap_generator = MipmapGenerator::new(&device);

        let egui_renderer = EguiRenderer::new(
            &device,
            surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        let mut renderer = Self {
            adapter,
            instance,
            device: Arc::new(device),
            queue,
            surface,
            surface_config,
            window_size: size,
            mipmap_generator,
            atmosphere_precomputer: None,
            shadow_pass: CascadedEVSMPass::new(),
            handle_id_to_texture_index: HashMap::new(),
            pending_materials: Vec::new(),
            geometry_pipeline: None,
            sky_pipeline: None,
            lighting_pipeline: None,
            downsample_pipeline: None,
            ssr_pipeline: None,
            ssgi_pipeline: None,
            ssgi_denoise_pipeline: None,
            ssgi_upsample_pipeline: None,
            composite_pipeline: None,
            scene_data_bind_group_layout: None,
            object_data_bind_group_layout: None,
            render_constants_bind_group_layout: None,
            downsample_bind_group_layout: None,
            ssr_inputs_bind_group_layout: None,
            ssgi_inputs_bind_group_layout: None,
            ssgi_blue_noise_bind_group_layout: None,
            ssgi_denoise_inputs_bind_group_layout: None,
            ssgi_upsample_bind_group_layout: None,
            ssr_camera_bind_group_layout: None,
            lighting_inputs_bind_group_layout: None,
            sky_bind_group_layout: None,
            ibl_bind_group_layout: None,
            composite_inputs_bind_group_layout: None,
            scene_data_bind_groups: Vec::new(),
            object_data_bind_group: None,
            render_constants_bind_group: None,
            downsample_bind_group: None,
            ssr_inputs_bind_group: None,
            ssgi_inputs_bind_group: None,
            ssgi_blue_noise_bind_group: None,
            ssgi_denoise_inputs_bind_group: None,
            ssgi_upsample_bind_group: None,
            ssr_camera_bind_groups: Vec::new(),
            lighting_inputs_bind_group: None,
            sky_bind_groups: Vec::new(),
            ibl_bind_group: None,
            composite_inputs_bind_group: None,
            main_depth_texture: None,
            main_depth_texture_view: None,
            gbuf_normal_texture: None,
            gbuf_albedo_texture: None,
            gbuf_mra_texture: None,
            gbuf_emission_texture: None,
            gbuf_normal_texture_view: None,
            gbuf_albedo_texture_view: None,
            gbuf_mra_texture_view: None,
            gbuf_emission_texture_view: None,
            direct_lighting_texture: None,
            direct_lighting_texture_view: None,
            direct_lighting_diffuse_texture: None,
            direct_lighting_diffuse_view: None,
            sky_texture: None,
            sky_texture_view: None,
            ssr_texture: None,
            ssr_texture_view: None,
            ssgi_upsampled_texture: None,
            ssgi_upsampled_texture_view: None,
            history_texture: None,
            history_texture_view: None,
            blue_noise_texture: None,
            blue_noise_texture_view: None,
            blue_noise_sampler: None,
            depth_half_texture: None,
            depth_half_view: None,
            normal_half_texture: None,
            normal_half_view: None,
            albedo_half_texture: None,
            albedo_half_view: None,
            direct_lighting_half_texture: None,
            direct_lighting_half_view: None,
            direct_lighting_diffuse_half_texture: None,
            direct_lighting_diffuse_half_view: None,
            ssgi_raw_half_texture: None,
            ssgi_raw_half_view: None,
            ssgi_denoised_half_texture: None,
            ssgi_denoised_half_view: None,
            ssgi_history_half_texture: None,
            ssgi_history_half_view: None,
            brdf_lut_texture: None,
            brdf_lut_view: None,
            irradiance_map_texture: None,
            irradiance_map_view: None,
            prefiltered_env_map_texture: None,
            prefiltered_env_map_view: None,
            geometry_instance_buffer: RefCell::new(None),
            geometry_instance_capacity: RefCell::new(0),
            camera_buffers: Vec::new(),
            lights_buffers: Vec::new(),
            sky_uniforms_buffers: Vec::new(),
            material_uniform_buffer: None,
            render_constants_buffer: None,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            texture_manager: TextureManager::default(),
            albedo_textures: Vec::new(),
            normal_textures: Vec::new(),
            mr_textures: Vec::new(),
            emission_textures: Vec::new(),
            pbr_sampler: None,
            gbuffer_sampler: None,
            scene_sampler: None,
            point_sampler: None,
            ibl_sampler: None,
            brdf_lut_sampler: None,
            egui_renderer,
            current_egui_data: None,
            frame_index: 0,
            current_render_data: None,
            logic_frame_duration: Duration::from_secs_f32(1.0 / target_tickrate),
            last_timestamp: None,
            prev_view_proj: Mat4::IDENTITY,
            prev_sky_uniforms: SkyUniforms::default(),
            needs_atmosphere_precompute: true,
        };

        renderer.initialize_resources()?;
        Ok(renderer)
    }

    fn initialize_resources(&mut self) -> Result<(), RendererError> {
        let device = &self.device;

        self.camera_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("camera-uniforms-{}", i)),
                    size: std::mem::size_of::<CameraUniforms>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        self.lights_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("lights-buffer-{}", i)),
                    size: (std::mem::size_of::<LightData>() * MAX_LIGHTS) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        self.sky_uniforms_buffers = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("sky-uniforms-{}", i)),
                    size: std::mem::size_of::<SkyUniforms>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        self.material_uniform_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("materials-buffer"),
            size: (std::mem::size_of::<MaterialShaderData>() * MAX_TOTAL_MATERIALS)
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.render_constants_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Constants Uniform Buffer"),
            size: std::mem::size_of::<ShaderConstants>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.queue.write_buffer(
            self.render_constants_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&ShaderConstants::default()),
        );

        self.pbr_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PBR Filtering Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }));

        self.gbuffer_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("G-Buffer Non-Filtering Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        self.point_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Point Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        self.scene_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Scene Filtering Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        self.ibl_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }));

        self.brdf_lut_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BRDF LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        self.atmosphere_precomputer = Some(AtmospherePrecomputer::new(&self.device));

        self.create_blue_noise_texture();
        self.upload_default_textures();
        self.shadow_pass.create_shadow_resources(&self.device);
        self.create_pipelines_and_bind_groups();
        self.create_object_data_bind_group();
        self.shadow_pass.create_shadow_pipeline(
            &self.device,
            self.render_constants_bind_group_layout.as_ref().unwrap(),
        );

        // This must be called after pipelines are created, but before render
        self.resize(self.window_size);

        info!("initialized deferred (bindless) renderer");
        Ok(())
    }

    fn create_blue_noise_texture(&mut self) {
        let device = &self.device;

        let blue_noise_bytes = include_bytes!("../assets/LDR_LLL1_0.png");
        let blue_noise_image =
            image::load_from_memory_with_format(blue_noise_bytes, ImageFormat::Png)
                .expect("Failed to load blue noise texture");
        let blue_noise_rgba = blue_noise_image.to_rgba8();
        let dimensions = blue_noise_image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Blue Noise Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            texture.as_image_copy(),
            &blue_noise_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );

        self.blue_noise_texture_view = Some(texture.create_view(&Default::default()));
        self.blue_noise_texture = Some(texture);

        self.blue_noise_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blue Noise Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));
    }

    fn upload_default_textures(&mut self) {
        let white_pixel = [255u8, 255, 255, 255];
        let (tex, view) = self.create_and_upload_texture(
            "Default Albedo",
            &white_pixel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1,
            1,
        );
        self.albedo_textures.push((tex, view));

        let flat_normal_pixel = [128u8, 128, 255, 255];
        let (tex, view) = self.create_and_upload_texture(
            "Default Normal",
            &flat_normal_pixel,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        self.normal_textures.push((tex, view));

        let default_mr_pixel = [255u8, 204, 0, 255];
        let (tex, view) = self.create_and_upload_texture(
            "Default MR",
            &default_mr_pixel,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        self.mr_textures.push((tex, view));

        let (tex, view) = self.create_and_upload_texture(
            "Default Emission",
            &white_pixel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1,
            1,
        );
        self.emission_textures.push((tex, view));
    }

    fn create_and_upload_texture(
        &self,
        label: &str,
        data: &[u8],
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            texture_size,
        );

        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    fn upload_texture_slice(&self, data: &[u8], target_array: &wgpu::Texture, layer_index: u32) {
        let bytes_per_pixel = 4u32;
        let bytes_per_row = bytes_per_pixel * DEFAULT_TEXTURE_RESOLUTION;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target_array,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_index,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(DEFAULT_TEXTURE_RESOLUTION),
            },
            wgpu::Extent3d {
                width: DEFAULT_TEXTURE_RESOLUTION,
                height: DEFAULT_TEXTURE_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_uncompressed_texture_slice(
        &self,
        data: &[u8],
        target_array: &wgpu::Texture,
        layer_index: u32,
        bytes_per_pixel: u32,
    ) {
        let bytes_per_row = bytes_per_pixel * DEFAULT_TEXTURE_RESOLUTION;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target_array,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_index,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(DEFAULT_TEXTURE_RESOLUTION),
            },
            wgpu::Extent3d {
                width: DEFAULT_TEXTURE_RESOLUTION,
                height: DEFAULT_TEXTURE_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_compressed_texture_slice(
        &self,
        data: &[u8],
        target_array: &wgpu::Texture,
        layer_index: u32,
        format: wgpu::TextureFormat,
    ) {
        let block_dimensions = format.block_dimensions();
        let block_size_bytes = format.block_size(None).unwrap();
        let width_in_blocks = DEFAULT_TEXTURE_RESOLUTION / block_dimensions.0;
        let bytes_per_row = width_in_blocks * block_size_bytes as u32;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target_array,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_index,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(DEFAULT_TEXTURE_RESOLUTION / block_dimensions.1),
            },
            wgpu::Extent3d {
                width: DEFAULT_TEXTURE_RESOLUTION,
                height: DEFAULT_TEXTURE_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn add_texture(
        &mut self,
        data: &[u8],
        kind: AssetKind,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<usize, RendererError> {
        let target_list = match kind {
            AssetKind::Albedo => &mut self.albedo_textures,
            AssetKind::Normal => &mut self.normal_textures,
            AssetKind::MetallicRoughness => &mut self.mr_textures,
            AssetKind::Emission => &mut self.emission_textures,
        };

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let is_compressed = format.is_compressed();
        let full_mip_levels = calc_mip_level_count(width, height);
        let (uploads, used_bytes) =
            build_mip_uploads(format, width, height, data.len(), full_mip_levels);
        if uploads.is_empty() {
            return Err(RendererError::ResourceCreation(
                "Texture upload missing mip data.".into(),
            ));
        }
        if used_bytes != data.len() {
            warn!(
                "Deferred texture upload size mismatch (used {}, provided {}).",
                used_bytes,
                data.len()
            );
        }
        let provided_levels = uploads.len() as u32;
        let mut mip_level_count = full_mip_levels;
        if is_compressed && provided_levels < full_mip_levels {
            warn!(
                "Compressed deferred texture missing mip data ({} of {}).",
                provided_levels, full_mip_levels
            );
            mip_level_count = provided_levels.max(1);
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: {
                let mut usage =
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
                if !is_compressed {
                    usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
                }
                usage
            },
            label: Some("Bindless Texture"),
            view_formats: &[],
        });

        for upload in &uploads {
            let end = upload.offset + upload.size;
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: upload.level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data[upload.offset..end],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bytes_per_row),
                    rows_per_image: Some(upload.rows_per_image),
                },
                wgpu::Extent3d {
                    width: upload.width,
                    height: upload.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Generate the rest of the mip chain
        if !is_compressed && mip_level_count > 1 && provided_levels < mip_level_count {
            let start_level = provided_levels.saturating_sub(1);
            let mut mip_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Mipmap Generation Encoder"),
                    });

            self.mipmap_generator.generate_mips_from(
                &mut mip_encoder,
                &self.device,
                &texture,
                start_level,
                mip_level_count,
            );

            // Submit the mip generation commands immediately
            self.queue.submit(std::iter::once(mip_encoder.finish()));
        }

        let view = texture.create_view(&Default::default());
        target_list.push((texture, view));

        let new_index = target_list.len() - 1;
        self.create_object_data_bind_group();
        Ok(new_index)
    }

    pub fn add_mesh(
        &mut self,
        id: usize,
        lods: &[MeshLodPayload],
        bounds: Aabb,
    ) -> Result<(), RendererError> {
        let mut gpu_lods = Vec::new();
        for lod in lods {
            let vertex_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-vbo-{}-lod{}", id, lod.lod_index)),
                    contents: bytemuck::cast_slice(lod.vertices.as_ref()),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh-ibo-{}-lod{}", id, lod.lod_index)),
                    contents: bytemuck::cast_slice(lod.indices.as_ref()),
                    usage: wgpu::BufferUsages::INDEX,
                });
            gpu_lods.push(MeshLod {
                vertex_buffer,
                index_buffer,
                index_count: lod.indices.len() as u32,
            });
        }

        self.meshes.insert(
            id,
            Mesh {
                lods: gpu_lods,
                bounds,
            },
        );
        Ok(())
    }

    pub fn add_material(&mut self, id: usize, material: Material) -> Result<(), RendererError> {
        if id >= MAX_TOTAL_MATERIALS {
            return Err(RendererError::ResourceCreation(format!(
                "Material ID {} is out of bounds. Maximum is 255.",
                id
            )));
        }

        let shader_data = MaterialShaderData::from(&material);

        let material_offset =
            (id * std::mem::size_of::<MaterialShaderData>()) as wgpu::BufferAddress;
        self.queue.write_buffer(
            self.material_uniform_buffer.as_ref().unwrap(),
            material_offset,
            bytemuck::bytes_of(&shader_data),
        );

        self.materials.insert(id, material);
        Ok(())
    }

    fn create_render_target_textures(&mut self) {
        let device = &self.device;
        let (width, height) = (self.surface_config.width, self.surface_config.height);

        let full_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        // Ensure half_size dimensions are not zero
        let half_size = wgpu::Extent3d {
            width: (width / 2).max(1),
            height: (height / 2).max(1),
            depth_or_array_layers: 1,
        };
        let default_texture_desc = wgpu::TextureDescriptor {
            label: None,
            size: full_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::empty(),
            view_formats: &[],
        };

        // --- Full-Resolution Textures ---
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Main Depth Texture"),
            size: full_size,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            ..default_texture_desc
        });
        self.main_depth_texture_view = Some(depth_texture.create_view(&Default::default()));
        self.main_depth_texture = Some(depth_texture);

        let gbuffer_usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        let gbuf_normal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("G-Buffer Normal"),
            size: full_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: gbuffer_usage,
            ..default_texture_desc
        });
        self.gbuf_normal_texture_view = Some(gbuf_normal_texture.create_view(&Default::default()));
        self.gbuf_normal_texture = Some(gbuf_normal_texture);

        let gbuf_albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("G-Buffer Albedo"),
            size: full_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: gbuffer_usage,
            ..default_texture_desc
        });
        self.gbuf_albedo_texture_view = Some(gbuf_albedo_texture.create_view(&Default::default()));
        self.gbuf_albedo_texture = Some(gbuf_albedo_texture);

        let gbuf_mra_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("G-Buffer MRA"),
            size: full_size,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: gbuffer_usage,
            ..default_texture_desc
        });
        self.gbuf_mra_texture_view = Some(gbuf_mra_texture.create_view(&Default::default()));
        self.gbuf_mra_texture = Some(gbuf_mra_texture);

        let gbuf_emission_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("G-Buffer Emission"),
            size: full_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: gbuffer_usage,
            ..default_texture_desc
        });
        self.gbuf_emission_texture_view =
            Some(gbuf_emission_texture.create_view(&Default::default()));
        self.gbuf_emission_texture = Some(gbuf_emission_texture);

        let hdr_texture_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;

        let direct_lighting_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Direct Lighting Texture"),
            size: full_size,
            usage: hdr_texture_usage,
            ..default_texture_desc
        });
        self.direct_lighting_texture_view =
            Some(direct_lighting_texture.create_view(&Default::default()));
        self.direct_lighting_texture = Some(direct_lighting_texture);

        let direct_lighting_diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Direct Lighting Diffuse Texture"),
            size: full_size,
            usage: hdr_texture_usage,
            ..default_texture_desc
        });
        self.direct_lighting_diffuse_view =
            Some(direct_lighting_diffuse_texture.create_view(&Default::default()));
        self.direct_lighting_diffuse_texture = Some(direct_lighting_diffuse_texture);

        let sky_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Sky Texture"),
            size: full_size,
            usage: hdr_texture_usage,
            ..default_texture_desc
        });
        self.sky_texture_view = Some(sky_texture.create_view(&Default::default()));
        self.sky_texture = Some(sky_texture);

        let ssr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR Result Texture"),
            size: full_size,
            usage: hdr_texture_usage,
            ..default_texture_desc
        });
        self.ssr_texture_view = Some(ssr_texture.create_view(&Default::default()));
        self.ssr_texture = Some(ssr_texture);

        let ssgi_upsampled_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSGI Upsampled Texture"),
            size: full_size,
            usage: hdr_texture_usage,
            ..default_texture_desc
        });
        self.ssgi_upsampled_texture_view =
            Some(ssgi_upsampled_texture.create_view(&Default::default()));
        self.ssgi_upsampled_texture = Some(ssgi_upsampled_texture);

        let surface_format = self.surface.get_current_texture().unwrap().texture.format();

        let history_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("History Texture"),
            size: full_size,
            format: surface_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ..default_texture_desc
        });
        self.history_texture_view = Some(history_texture.create_view(&Default::default()));
        self.history_texture = Some(history_texture);

        // --- Half-Resolution Textures ---
        self.depth_half_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Half-Res"),
            size: half_size,
            format: wgpu::TextureFormat::R32Float,
            usage: hdr_texture_usage,
            ..default_texture_desc
        }));
        self.depth_half_view = Some(
            self.depth_half_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.normal_half_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normal Half-Res"),
            size: half_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: hdr_texture_usage,
            ..default_texture_desc
        }));
        self.normal_half_view = Some(
            self.normal_half_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.albedo_half_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Albedo Half-Res"),
            size: half_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: hdr_texture_usage,
            ..default_texture_desc
        }));
        self.albedo_half_view = Some(
            self.albedo_half_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.direct_lighting_half_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Direct Lighting Half-Res"),
            size: half_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: hdr_texture_usage,
            ..default_texture_desc
        }));
        self.direct_lighting_half_view = Some(
            self.direct_lighting_half_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.direct_lighting_diffuse_half_texture =
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Direct Lighting Diffuse Half-Res"),
                size: half_size,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: hdr_texture_usage,
                ..default_texture_desc
            }));
        self.direct_lighting_diffuse_half_view = Some(
            self.direct_lighting_diffuse_half_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.ssgi_raw_half_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSGI Raw Half-Res"),
            size: half_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: hdr_texture_usage,
            ..default_texture_desc
        }));
        self.ssgi_raw_half_view = Some(
            self.ssgi_raw_half_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.ssgi_denoised_half_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSGI Denoised Half-Res"),
            size: half_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: hdr_texture_usage,
            ..default_texture_desc
        }));
        self.ssgi_denoised_half_view = Some(
            self.ssgi_denoised_half_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.ssgi_history_half_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSGI History Half-Res"),
            size: half_size,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ..default_texture_desc
        }));
        self.ssgi_history_half_view = Some(
            self.ssgi_history_half_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        let brdf_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BRDF LUT"),
            size: wgpu::Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ..default_texture_desc
        });
        self.brdf_lut_view = Some(brdf_lut.create_view(&Default::default()));
        self.brdf_lut_texture = Some(brdf_lut);

        let irradiance_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Irradiance Map"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 6,
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ..default_texture_desc
        });
        self.irradiance_map_view = Some(irradiance_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        }));
        self.irradiance_map_texture = Some(irradiance_map);

        let prefiltered_env_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Prefiltered Environment Map"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 6,
            },
            mip_level_count: 5,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ..default_texture_desc
        });
        self.prefiltered_env_map_view = Some(prefiltered_env_map.create_view(
            &wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            },
        ));
        self.prefiltered_env_map_texture = Some(prefiltered_env_map);
    }

    fn create_pipelines_and_bind_groups(&mut self) {
        let device = &self.device;

        let g_buffer_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/g_buffer.wgsl"));
        let sky_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/sky_sampled.wgsl"));
        let downsample_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/downsample.wgsl"));
        let ssr_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/ssr.wgsl"));
        let ssgi_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/ssgi.wgsl"));
        let ssgi_denoise_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/ssgi_denoise.wgsl"));
        let ssgi_upsample_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/ssgi_upsample.wgsl"));
        let lighting_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/lighting.wgsl"));
        let composite_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/composite.wgsl"));

        let (
            scene_data_bind_group_layout,
            object_data_bind_group_layout,
            downsample_bind_group_layout,
            ssr_inputs_bind_group_layout,
            ssgi_inputs_bind_group_layout,
            ssgi_blue_noise_bind_group_layout,
            ssgi_denoise_inputs_bind_group_layout,
            ssgi_upsample_bind_group_layout,
            ssr_camera_bind_group_layout,
            lighting_inputs_bind_group_layout,
            sky_bind_group_layout,
            ibl_bind_group_layout,
            composite_inputs_bind_group_layout,
            render_constants_bind_group_layout,
        ) = self.create_bind_group_layouts();

        let atmosphere_lut_bind_group_layout = &self
            .atmosphere_precomputer
            .as_ref()
            .unwrap()
            .sampling_bind_group_layout;

        let geometry_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Geometry Pipeline Layout"),
                bind_group_layouts: &[
                    &scene_data_bind_group_layout,
                    &object_data_bind_group_layout,
                    &render_constants_bind_group_layout,
                ],
                immediate_size: 0,
            });

        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[
                &sky_bind_group_layout,
                atmosphere_lut_bind_group_layout,
                &render_constants_bind_group_layout,
            ],
            immediate_size: 0,
        });

        let downsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Downsample Pipeline Layout"),
                bind_group_layouts: &[&downsample_bind_group_layout],
                immediate_size: 0,
            });

        let ssr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Pipeline Layout"),
            bind_group_layouts: &[
                &ssr_inputs_bind_group_layout,
                &ssr_camera_bind_group_layout,
                &render_constants_bind_group_layout,
            ],
            immediate_size: 0,
        });

        let ssgi_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSGI Pipeline Layout"),
            bind_group_layouts: &[
                &ssgi_inputs_bind_group_layout,
                &ssr_camera_bind_group_layout,
                &ssgi_blue_noise_bind_group_layout,
                &render_constants_bind_group_layout,
            ],
            immediate_size: 0,
        });

        let ssgi_denoise_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSGI Denoise Pipeline Layout"),
                bind_group_layouts: &[
                    &ssgi_denoise_inputs_bind_group_layout,
                    &ssr_camera_bind_group_layout,
                ],
                immediate_size: 0,
            });

        let ssgi_upsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSGI Upsample Pipeline Layout"),
                bind_group_layouts: &[&ssgi_upsample_bind_group_layout],
                immediate_size: 0,
            });

        let lighting_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Lighting Pipeline Layout"),
                bind_group_layouts: &[
                    &lighting_inputs_bind_group_layout,
                    &scene_data_bind_group_layout,
                    atmosphere_lut_bind_group_layout,
                    &render_constants_bind_group_layout,
                ],
                immediate_size: 0,
            });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[
                    &composite_inputs_bind_group_layout,
                    &ibl_bind_group_layout,
                    &scene_data_bind_group_layout,
                    &render_constants_bind_group_layout,
                ],
                immediate_size: 0,
            });

        self.geometry_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Geometry Pipeline"),
                layout: Some(&geometry_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &g_buffer_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc(), GeometryInstanceRaw::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &g_buffer_shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(wgpu::TextureFormat::Rgba16Float.into()),
                        Some(wgpu::TextureFormat::Rgba16Float.into()),
                        Some(wgpu::TextureFormat::Rgba8Unorm.into()),
                        Some(wgpu::TextureFormat::Rgba16Float.into()),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        self.sky_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Sky Pipeline"),
                layout: Some(&sky_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &sky_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &sky_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::TextureFormat::Rgba16Float.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            }),
        );

        self.downsample_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Downsample Pipeline"),
                layout: Some(&downsample_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &downsample_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &downsample_shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(wgpu::TextureFormat::R32Float.into()),
                        Some(wgpu::TextureFormat::Rgba16Float.into()),
                        Some(wgpu::TextureFormat::Rgba16Float.into()),
                        Some(wgpu::TextureFormat::Rgba16Float.into()),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        self.ssr_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("SSR Pipeline"),
                layout: Some(&ssr_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ssr_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssr_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
        );

        self.ssgi_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("SSGI Pipeline"),
                layout: Some(&ssgi_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ssgi_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssgi_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
        );

        self.ssgi_denoise_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("SSGI Denoise Pipeline"),
                layout: Some(&ssgi_denoise_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ssgi_denoise_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssgi_denoise_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
            },
        ));

        self.ssgi_upsample_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("SSGI Upsample Pipeline"),
                layout: Some(&ssgi_upsample_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ssgi_upsample_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssgi_upsample_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::TextureFormat::Rgba16Float.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        self.lighting_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Lighting Pipeline"),
                layout: Some(&lighting_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &lighting_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &lighting_shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        self.composite_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Composite Pipeline"),
                layout: Some(&composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
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
            },
        ));

        self.scene_data_bind_group_layout = Some(scene_data_bind_group_layout);
        self.object_data_bind_group_layout = Some(object_data_bind_group_layout);
        self.downsample_bind_group_layout = Some(downsample_bind_group_layout);
        self.ssr_inputs_bind_group_layout = Some(ssr_inputs_bind_group_layout);
        self.ssgi_inputs_bind_group_layout = Some(ssgi_inputs_bind_group_layout);
        self.ssgi_blue_noise_bind_group_layout = Some(ssgi_blue_noise_bind_group_layout);
        self.ssgi_denoise_inputs_bind_group_layout = Some(ssgi_denoise_inputs_bind_group_layout);
        self.ssgi_upsample_bind_group_layout = Some(ssgi_upsample_bind_group_layout);
        self.ssr_camera_bind_group_layout = Some(ssr_camera_bind_group_layout);
        self.lighting_inputs_bind_group_layout = Some(lighting_inputs_bind_group_layout);
        self.sky_bind_group_layout = Some(sky_bind_group_layout);
        self.ibl_bind_group_layout = Some(ibl_bind_group_layout);
        self.composite_inputs_bind_group_layout = Some(composite_inputs_bind_group_layout);
        self.render_constants_bind_group_layout = Some(render_constants_bind_group_layout);
    }

    fn create_bind_group_layouts(
        &self,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        let device = &self.device;

        let scene_data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scene Data Bind Group Layout"),
                entries: &[
                    buffer_binding(
                        0,
                        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                    buffer_binding(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Storage { read_only: true },
                    ),
                    texture_binding(2, true, wgpu::TextureViewDimension::D2Array, false),
                    sampler_binding(3, true, wgpu::SamplerBindingType::Filtering),
                    buffer_binding(
                        4,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                    buffer_binding(
                        5,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                ],
            });

        let object_data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Object Data Bind Group Layout"),
                entries: &[
                    buffer_binding(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Storage { read_only: true },
                    ),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: Some(std::num::NonZeroU32::new(MAX_TOTAL_TEXTURES).unwrap()),
                    },
                    sampler_binding(2, true, wgpu::SamplerBindingType::Filtering),
                ],
            });

        let downsample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Downsample Bind Group Layout"),
                entries: &[
                    texture_binding(0, false, wgpu::TextureViewDimension::D2, true), // depth
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, false), // normal
                    texture_binding(2, false, wgpu::TextureViewDimension::D2, false), // albedo
                    texture_binding(3, false, wgpu::TextureViewDimension::D2, false), // diffuse lighting
                    sampler_binding(4, false, wgpu::SamplerBindingType::NonFiltering),
                ],
            });

        let ssr_inputs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSR Inputs Bind Group Layout"),
                entries: &[
                    texture_binding(0, false, wgpu::TextureViewDimension::D2, false),
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, false),
                    texture_binding(2, false, wgpu::TextureViewDimension::D2, true),
                    texture_binding(3, true, wgpu::TextureViewDimension::D2, false),
                    sampler_binding(4, false, wgpu::SamplerBindingType::NonFiltering),
                    sampler_binding(5, true, wgpu::SamplerBindingType::Filtering),
                ],
            });

        let ssgi_inputs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSGI Inputs Bind Group Layout"),
                entries: &[
                    texture_binding(0, true, wgpu::TextureViewDimension::D2, false), // normal_half
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, false), // depth_half
                    texture_binding(2, true, wgpu::TextureViewDimension::D2, false), // albedo_half
                    texture_binding(3, true, wgpu::TextureViewDimension::D2, false), // history_half
                    texture_binding(4, true, wgpu::TextureViewDimension::D2, false), // lighting_diffuse_half
                    sampler_binding(5, false, wgpu::SamplerBindingType::NonFiltering),
                    sampler_binding(6, true, wgpu::SamplerBindingType::Filtering),
                ],
            });

        let ssgi_blue_noise_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSGI Blue Noise Bind Group Layout"),
                entries: &[
                    texture_binding(0, false, wgpu::TextureViewDimension::D2, false),
                    sampler_binding(1, false, wgpu::SamplerBindingType::NonFiltering),
                ],
            });

        let ssgi_denoise_inputs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSGI Denoise Inputs Bind Group Layout"),
                entries: &[
                    texture_binding(0, true, wgpu::TextureViewDimension::D2, false), // t_noisy_input
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, false), // t_depth
                    texture_binding(2, true, wgpu::TextureViewDimension::D2, false), // t_normal
                    texture_binding(3, true, wgpu::TextureViewDimension::D2, false), // t_history
                    sampler_binding(4, true, wgpu::SamplerBindingType::Filtering),   // s_linear
                    sampler_binding(5, false, wgpu::SamplerBindingType::NonFiltering), // s_point
                ],
            });

        let ssgi_upsample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSGI Upsample Bind Group Layout"),
                entries: &[
                    texture_binding(0, true, wgpu::TextureViewDimension::D2, false), // denoised_ssgi_half
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, true), // full_res_depth
                    texture_binding(2, false, wgpu::TextureViewDimension::D2, false), // full_res_normal
                    sampler_binding(3, true, wgpu::SamplerBindingType::Filtering),
                    sampler_binding(4, false, wgpu::SamplerBindingType::NonFiltering),
                ],
            });

        let ssr_camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSR Camera Bind Group Layout"),
                entries: &[buffer_binding(
                    0,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Uniform,
                )],
            });

        let lighting_inputs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lighting Inputs Bind Group Layout"),
                entries: &[
                    texture_binding(0, false, wgpu::TextureViewDimension::D2, true), // depth
                    texture_binding(1, false, wgpu::TextureViewDimension::D2, false), // normal
                    texture_binding(2, false, wgpu::TextureViewDimension::D2, false), // albedo
                    texture_binding(3, false, wgpu::TextureViewDimension::D2, false), // mra
                    texture_binding(4, false, wgpu::TextureViewDimension::D2, false), // emission
                    sampler_binding(5, false, wgpu::SamplerBindingType::NonFiltering), // gbuffer sampler
                ],
            });

        let sky_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sky Bind Group Layout"),
                entries: &[
                    buffer_binding(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                    buffer_binding(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                    ),
                    sampler_binding(2, true, wgpu::SamplerBindingType::Filtering), // scene sampler
                    texture_binding(3, false, wgpu::TextureViewDimension::D2, true), // depth,
                ],
            });

        let ibl_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("IBL Bind Group Layout"),
                entries: &[
                    texture_binding(0, true, wgpu::TextureViewDimension::D2, false),
                    texture_binding(1, true, wgpu::TextureViewDimension::Cube, false),
                    texture_binding(2, true, wgpu::TextureViewDimension::Cube, false),
                    sampler_binding(3, true, wgpu::SamplerBindingType::Filtering),
                    sampler_binding(4, true, wgpu::SamplerBindingType::Filtering),
                ],
            });

        let composite_inputs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Inputs Bind Group Layout"),
                entries: &[
                    texture_binding(0, true, wgpu::TextureViewDimension::D2, false), // direct lighting
                    texture_binding(1, true, wgpu::TextureViewDimension::D2, false), // ssgi
                    texture_binding(2, true, wgpu::TextureViewDimension::D2, false), // ssr
                    texture_binding(3, true, wgpu::TextureViewDimension::D2, false), // albedo
                    texture_binding(4, true, wgpu::TextureViewDimension::D2, false), // emission
                    sampler_binding(5, true, wgpu::SamplerBindingType::Filtering),
                    texture_binding(6, true, wgpu::TextureViewDimension::D2, false), // normal
                    texture_binding(7, true, wgpu::TextureViewDimension::D2, false), // mra
                    texture_binding(8, false, wgpu::TextureViewDimension::D2, true), // depth
                    texture_binding(9, true, wgpu::TextureViewDimension::D2, false), // sky texture
                ],
            });

        let render_constants_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Constants Bind Group Layout"),
                entries: &[buffer_binding(
                    0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Uniform,
                )],
            });

        (
            scene_data_bind_group_layout,
            object_data_bind_group_layout,
            downsample_bind_group_layout,
            ssr_inputs_bind_group_layout,
            ssgi_inputs_bind_group_layout,
            ssgi_blue_noise_bind_group_layout,
            ssgi_denoise_inputs_bind_group_layout,
            ssgi_upsample_bind_group_layout,
            ssr_camera_bind_group_layout,
            lighting_inputs_bind_group_layout,
            sky_bind_group_layout,
            ibl_bind_group_layout,
            composite_inputs_bind_group_layout,
            render_constants_bind_group_layout,
        )
    }

    fn create_object_data_bind_group(&mut self) {
        let default_view = &self.albedo_textures[0].1;
        let mut all_views: Vec<&wgpu::TextureView> = Vec::new();
        all_views.extend(self.albedo_textures.iter().map(|(_, view)| view));
        all_views.extend(self.normal_textures.iter().map(|(_, view)| view));
        all_views.extend(self.mr_textures.iter().map(|(_, view)| view));
        all_views.extend(self.emission_textures.iter().map(|(_, view)| view));

        let current_len = all_views.len();
        if current_len < MAX_TOTAL_TEXTURES as usize {
            all_views.resize(MAX_TOTAL_TEXTURES as usize, default_view);
        }

        self.object_data_bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: self.object_data_bind_group_layout.as_ref().unwrap(),
                label: Some("Object Data Bind Group (Bindless)"),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self
                            .material_uniform_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureViewArray(&all_views),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            self.pbr_sampler.as_ref().unwrap(),
                        ),
                    },
                ],
            }),
        );
    }

    fn create_bind_groups(&mut self) {
        let device = &self.device;

        self.scene_data_bind_groups = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.scene_data_bind_group_layout.as_ref().unwrap(),
                    label: Some(&format!("Scene Data Bind Group {}", i)),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.camera_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.lights_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                self.shadow_pass.shadow_map_view.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(
                                self.shadow_pass.shadow_sampler.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self
                                .shadow_pass
                                .shadow_uniforms_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.sky_uniforms_buffers[i].as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        self.ssr_camera_bind_groups = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.ssr_camera_bind_group_layout.as_ref().unwrap(),
                    label: Some(&format!("SSR Camera Bind Group {}", i)),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.camera_buffers[i].as_entire_binding(),
                    }],
                })
            })
            .collect();

        self.sky_bind_groups = (0..FRAMES_IN_FLIGHT)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.sky_bind_group_layout.as_ref().unwrap(),
                    label: Some(&format!("Sky Bind Group {}", i)),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.camera_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.sky_uniforms_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                self.scene_sampler.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                self.main_depth_texture_view.as_ref().unwrap(),
                            ),
                        },
                    ],
                })
            })
            .collect();

        self.downsample_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Downsample Bind Group"),
            layout: self.downsample_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.main_depth_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        self.gbuf_normal_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.gbuf_albedo_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        self.direct_lighting_diffuse_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(self.point_sampler.as_ref().unwrap()),
                },
            ],
        }));

        self.ssr_inputs_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Inputs Bind Group"),
            layout: self.ssr_inputs_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.gbuf_normal_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        self.gbuf_mra_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.main_depth_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        self.history_texture_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(
                        self.gbuffer_sampler.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(self.scene_sampler.as_ref().unwrap()),
                },
            ],
        }));

        self.ssgi_inputs_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSGI Inputs Bind Group"),
            layout: self.ssgi_inputs_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.normal_half_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        self.depth_half_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.albedo_half_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        self.ssgi_history_half_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        self.direct_lighting_diffuse_half_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(self.point_sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(self.scene_sampler.as_ref().unwrap()),
                },
            ],
        }));

        self.ssgi_blue_noise_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSGI Blue Noise Bind Group"),
                layout: self.ssgi_blue_noise_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            self.blue_noise_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            self.blue_noise_sampler.as_ref().unwrap(),
                        ),
                    },
                ],
            }));

        self.ssgi_denoise_inputs_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSGI Denoise Inputs Bind Group"),
                layout: self.ssgi_denoise_inputs_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            self.ssgi_raw_half_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            self.depth_half_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            self.normal_half_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            self.ssgi_history_half_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(
                            self.scene_sampler.as_ref().unwrap(), // s_linear
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(
                            self.point_sampler.as_ref().unwrap(), // s_point
                        ),
                    },
                ],
            }));

        self.ssgi_upsample_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSGI Upsample Bind Group"),
                layout: self.ssgi_upsample_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            self.ssgi_denoised_half_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            self.main_depth_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_normal_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(
                            self.scene_sampler.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(
                            self.point_sampler.as_ref().unwrap(),
                        ),
                    },
                ],
            }));

        self.lighting_inputs_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Lighting Inputs Bind Group"),
                layout: self.lighting_inputs_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            self.main_depth_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_normal_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_albedo_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_mra_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_emission_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(
                            self.gbuffer_sampler.as_ref().unwrap(),
                        ),
                    },
                ],
            }));

        self.ibl_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL Bind Group"),
            layout: self.ibl_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.brdf_lut_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        self.irradiance_map_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.prefiltered_env_map_view.as_ref().unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(self.ibl_sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(
                        self.brdf_lut_sampler.as_ref().unwrap(),
                    ),
                },
            ],
        }));

        self.composite_inputs_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Composite Inputs Bind Group"),
                layout: self.composite_inputs_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            self.direct_lighting_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            self.ssgi_upsampled_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            self.ssr_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_albedo_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_emission_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(
                            self.scene_sampler.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_normal_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            self.gbuf_mra_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(
                            self.main_depth_texture_view.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(
                            self.sky_texture_view.as_ref().unwrap(),
                        ),
                    },
                ],
            }));

        self.render_constants_bind_group = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Constants Bind Group"),
                layout: self.render_constants_bind_group_layout.as_ref().unwrap(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .render_constants_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                }],
            }),
        );
    }

    fn calculate_uniforms(
        &self,
        render_data: &RenderData,
        alpha: f32,
    ) -> (CameraUniforms, Vec<LightData>, Mat4) {
        let camera = &render_data.camera_component;
        let eye = render_data
            .previous_camera_transform
            .position
            .lerp(render_data.current_camera_transform.position, alpha);
        let rotation = Quat::from(render_data.previous_camera_transform.rotation)
            .slerp(render_data.current_camera_transform.rotation, alpha);
        let forward = rotation * Vec3::Z;
        let up = rotation * Vec3::Y;
        let view_matrix = Mat4::look_at_rh(eye, eye + forward, up);
        let projection_matrix = Mat4::perspective_infinite_reverse_rh(
            camera.fov_y_rad,
            camera.aspect_ratio,
            camera.near_plane,
        );

        let current_view_proj = projection_matrix * view_matrix;
        let inv_proj = projection_matrix.inverse();
        let inv_view_proj = current_view_proj.inverse();

        let camera_uniforms = CameraUniforms {
            view_matrix: view_matrix.to_cols_array_2d(),
            projection_matrix: projection_matrix.to_cols_array_2d(),
            inverse_projection_matrix: inv_proj.to_cols_array_2d(),
            inverse_view_projection_matrix: inv_view_proj.to_cols_array_2d(),
            view_position: eye.to_array(),
            light_count: render_data.lights.len() as u32,
            _pad_light: [0; 4],
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            frame_index: self.frame_index as u32,
            _pad_after_frame: [0; 3],
            _padding: [0; 3],
            _pad_after_padding: 0,
            _pad_end: [0; 4],
        };

        let light_data: Vec<LightData> = render_data
            .lights
            .iter()
            .take(MAX_LIGHTS)
            .map(|light| {
                let position = light
                    .previous_transform
                    .position
                    .lerp(light.current_transform.position, alpha);
                let rotation = Quat::from(light.previous_transform.rotation)
                    .slerp(light.current_transform.rotation, alpha);
                let direction = (rotation * -Vec3::Z).normalize_or_zero();
                LightData {
                    position: position.into(),
                    light_type: match light.light_type {
                        LightType::Directional => 0,
                        LightType::Point => 1,
                        LightType::Spot { .. } => 2,
                    },
                    color: light.color,
                    intensity: light.intensity,
                    direction: direction.into(),
                    _padding: 0.0,
                }
            })
            .collect();

        (camera_uniforms, light_data, current_view_proj)
    }

    fn run_geometry_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_data: &RenderData,
        alpha: f32,
    ) {
        // --- 1. Build CPU-side instance data and batch info ---

        // (Key is (mesh_id, lod_index))
        // (Value is (instance_offset, instance_count))
        let mut batch_info: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
        let mut all_instances: Vec<GeometryInstanceRaw> = Vec::new();

        let mut sorted_objects = render_data.objects.iter().collect::<Vec<_>>();
        sorted_objects.sort_by_key(|obj| (obj.mesh_id, obj.lod_index, obj.material_id));

        for object in sorted_objects {
            if let (Some(mesh), true) = (
                self.meshes.get(&object.mesh_id),
                self.materials.contains_key(&object.material_id),
            ) {
                if mesh.lods.is_empty() {
                    continue;
                }
                let lod_index = object.lod_index.min(mesh.lods.len() - 1);
                let key = (object.mesh_id, lod_index);

                // Calculate interpolated transform
                let position = object
                    .previous_transform
                    .position
                    .lerp(object.current_transform.position, alpha);
                let rotation = Quat::from(object.previous_transform.rotation)
                    .slerp(object.current_transform.rotation, alpha);
                let scale = object
                    .previous_transform
                    .scale
                    .lerp(object.current_transform.scale, alpha);
                let model_matrix = Mat4::from_scale_rotation_translation(scale, rotation, position);

                let instance_data = GeometryInstanceRaw {
                    model_matrix: model_matrix.to_cols_array_2d(),
                    material_id: object.material_id as u32,
                };

                let current_offset = all_instances.len() as u32;
                all_instances.push(instance_data);

                let entry = batch_info.entry(key).or_insert((current_offset, 0));
                entry.1 += 1;
            }
        }

        let total_instances = all_instances.len();
        if total_instances == 0 {
            return;
        }

        // --- 2. Check and resize the GPU buffer if needed ---

        let mut capacity = self.geometry_instance_capacity.borrow_mut();
        let mut buffer = self.geometry_instance_buffer.borrow_mut();

        if total_instances > *capacity || buffer.is_none() {
            if let Some(old_buffer) = buffer.take() {
                old_buffer.destroy();
            }

            let new_capacity = (total_instances as f32 * 1.5).ceil() as usize;

            *buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Geometry Instance Buffer"),
                size: (new_capacity * std::mem::size_of::<GeometryInstanceRaw>())
                    as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            *capacity = new_capacity;
        }

        // --- 3. Upload all instance data in one go ---
        self.queue.write_buffer(
            buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&all_instances),
        );

        // --- 4. Run Geometry Pass ---
        let gbuffer_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: self.gbuf_normal_texture_view.as_ref().unwrap(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: self.gbuf_albedo_texture_view.as_ref().unwrap(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: self.gbuf_mra_texture_view.as_ref().unwrap(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: self.gbuf_emission_texture_view.as_ref().unwrap(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];

        let mut geometry_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Geometry Pass"),
            color_attachments: &gbuffer_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.main_depth_texture_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        geometry_pass.set_pipeline(self.geometry_pipeline.as_ref().unwrap());
        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;
        geometry_pass.set_bind_group(0, &self.scene_data_bind_groups[buffer_index], &[]);
        geometry_pass.set_bind_group(1, self.object_data_bind_group.as_ref().unwrap(), &[]);
        geometry_pass.set_bind_group(2, self.render_constants_bind_group.as_ref().unwrap(), &[]);

        // **Bind the one persistent buffer**
        geometry_pass.set_vertex_buffer(1, buffer.as_ref().unwrap().slice(..));

        // --- 5. Draw all batches ---
        for ((mesh_id, lod_index), (instance_offset, instance_count)) in batch_info {
            if let Some(mesh) = self.meshes.get(&mesh_id) {
                let lod = &mesh.lods[lod_index];

                geometry_pass.set_vertex_buffer(0, lod.vertex_buffer.slice(..));
                geometry_pass
                    .set_index_buffer(lod.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                let instance_range = instance_offset..(instance_offset + instance_count);
                geometry_pass.draw_indexed(0..lod.index_count, 0, instance_range);
            }
        }
    }

    fn run_sky_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Sky Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.sky_texture_view.as_ref().unwrap(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(self.sky_pipeline.as_ref().unwrap());
        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;
        pass.set_bind_group(0, &self.sky_bind_groups[buffer_index], &[]);
        pass.set_bind_group(
            1,
            &self
                .atmosphere_precomputer
                .as_ref()
                .unwrap()
                .sampling_bind_group,
            &[],
        );
        pass.set_bind_group(2, self.render_constants_bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }

    fn run_lighting_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Lighting Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: self.direct_lighting_texture_view.as_ref().unwrap(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: self.direct_lighting_diffuse_view.as_ref().unwrap(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(self.lighting_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.lighting_inputs_bind_group.as_ref().unwrap(), &[]);
        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;
        pass.set_bind_group(1, &self.scene_data_bind_groups[buffer_index], &[]);
        pass.set_bind_group(
            2,
            &self
                .atmosphere_precomputer
                .as_ref()
                .unwrap()
                .sampling_bind_group,
            &[],
        );
        pass.set_bind_group(3, self.render_constants_bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }

    fn run_downsample_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Downsample Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: self.depth_half_view.as_ref().unwrap(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: self.normal_half_view.as_ref().unwrap(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: self.albedo_half_view.as_ref().unwrap(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: self.direct_lighting_diffuse_half_view.as_ref().unwrap(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(self.downsample_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.downsample_bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }

    fn run_ssgi_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Pass (Half-Res)"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.ssgi_raw_half_view.as_ref().unwrap(),
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
        pass.set_pipeline(self.ssgi_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.ssgi_inputs_bind_group.as_ref().unwrap(), &[]);
        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;
        pass.set_bind_group(1, &self.ssr_camera_bind_groups[buffer_index], &[]);
        pass.set_bind_group(2, self.ssgi_blue_noise_bind_group.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, self.render_constants_bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }

    fn run_ssgi_denoise_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Denoise Pass (Half-Res)"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.ssgi_denoised_half_view.as_ref().unwrap(),
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
        pass.set_pipeline(self.ssgi_denoise_pipeline.as_ref().unwrap());
        pass.set_bind_group(
            0,
            self.ssgi_denoise_inputs_bind_group.as_ref().unwrap(),
            &[],
        );

        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;
        pass.set_bind_group(1, &self.ssr_camera_bind_groups[buffer_index], &[]);

        pass.draw(0..3, 0..1);
    }

    fn run_ssgi_upsample_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Upsample Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.ssgi_upsampled_texture_view.as_ref().unwrap(),
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
        pass.set_pipeline(self.ssgi_upsample_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.ssgi_upsample_bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }

    fn run_ssr_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.ssr_texture_view.as_ref().unwrap(),
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
        pass.set_pipeline(self.ssr_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.ssr_inputs_bind_group.as_ref().unwrap(), &[]);
        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;
        pass.set_bind_group(1, &self.ssr_camera_bind_groups[buffer_index], &[]);
        pass.set_bind_group(2, self.render_constants_bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }

    fn run_composite_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Composite Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(self.composite_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.composite_inputs_bind_group.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.ibl_bind_group.as_ref().unwrap(), &[]);
        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;
        pass.set_bind_group(2, &self.scene_data_bind_groups[buffer_index], &[]);
        pass.set_bind_group(3, self.render_constants_bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }

    fn sync_material_buffer(&mut self) {
        if self.materials.is_empty() {
            return;
        }

        // Pre-calculate the base offsets for each texture type.
        let albedo_offset = 0i32;
        let normal_offset = self.albedo_textures.len() as i32;
        let mr_offset = normal_offset + self.normal_textures.len() as i32;
        let emission_offset = mr_offset + self.mr_textures.len() as i32;

        // 1. Create a single, large buffer on the CPU (the staging buffer).
        let material_struct_size = std::mem::size_of::<MaterialShaderData>();
        let mut cpu_buffer = vec![0u8; MAX_TOTAL_MATERIALS * material_struct_size];

        // 2. Iterate through all loaded materials and write their data into the correct spot in the CPU buffer.
        for (id, material) in &self.materials {
            // Calculate the final, absolute texture indices for the shader.
            let shader_data = MaterialShaderData {
                albedo: material.albedo,
                metallic: material.metallic,
                roughness: material.roughness,
                ao: material.ao,
                emission_strength: material.emission_strength,
                emission_color: material.emission_color,
                albedo_idx: if material.albedo_texture_index == -1 {
                    -1
                } else {
                    albedo_offset + material.albedo_texture_index
                },
                normal_idx: if material.normal_texture_index == -1 {
                    -1
                } else {
                    normal_offset + material.normal_texture_index
                },
                metallic_roughness_idx: if material.metallic_roughness_texture_index == -1 {
                    -1
                } else {
                    mr_offset + material.metallic_roughness_texture_index
                },
                emission_idx: if material.emission_texture_index == -1 {
                    -1
                } else {
                    emission_offset + material.emission_texture_index
                },
                _padding: 0.0,
                alpha_mode: material.alpha_mode as u32,
                alpha_cutoff: material.alpha_cutoff.unwrap_or(0.0),
                _pad_alpha: [0; 2],
            };

            // Calculate the exact byte offset for this material in the buffer.
            let offset = *id * material_struct_size;

            // Write the raw bytes of the struct into the CPU buffer at the correct offset.
            if let Some(slice) = cpu_buffer.get_mut(offset..offset + material_struct_size) {
                slice.copy_from_slice(bytemuck::bytes_of(&shader_data));
            }
        }

        // 3. Upload the entire CPU buffer to the GPU in a single command.
        self.queue.write_buffer(
            self.material_uniform_buffer.as_ref().unwrap(),
            0,
            &cpu_buffer,
        );
    }
}

impl RenderTrait for DeferredRenderer {
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);

            // Recreate all screen-size dependent resources
            self.create_render_target_textures();
            self.create_bind_groups();
        }
    }

    fn render(&mut self) -> Result<(), RendererError> {
        self.sync_material_buffer();

        let Some(render_data) = &self.current_render_data else {
            return Ok(());
        };

        let now = Instant::now();
        let time_since_last_update = now.saturating_duration_since(render_data.timestamp);
        let alpha = (time_since_last_update.as_secs_f32()
            / self.logic_frame_duration.as_secs_f32())
        .clamp(0.0, 1.0);
        self.last_timestamp = Some(render_data.timestamp);

        let (camera_uniforms, light_data, new_view_proj) =
            self.calculate_uniforms(render_data, alpha);

        let buffer_index = self.frame_index % FRAMES_IN_FLIGHT;
        self.queue.write_buffer(
            &self.camera_buffers[buffer_index],
            0,
            bytemuck::bytes_of(&camera_uniforms),
        );
        self.queue.write_buffer(
            &self.lights_buffers[buffer_index],
            0,
            bytemuck::cast_slice(&light_data),
        );

        let directional_light = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional));

        let sky_cfg = &render_data.render_config.shader_constants;
        let (sun_dir, sun_color, sun_intensity) = if let Some(light) = directional_light {
            (
                (light.current_transform.rotation * Vec3::Z).normalize_or_zero(),
                light.color,
                light.intensity,
            )
        } else {
            (
                // Default sun
                Vec3::new(0.2, 0.8, 0.1).normalize(),
                Vec3::ONE.to_array(),
                100.0,
            )
        };

        let sky_uniforms = SkyUniforms {
            sun_direction: sun_dir.to_array(),
            _padding: 0.0,
            sun_color: sun_color,
            sun_intensity,
            ground_albedo: sky_cfg.sky_ground_albedo,
            ground_brightness: sky_cfg.sky_ground_brightness,
            night_ambient_color: sky_cfg.night_ambient_color,
            sun_angular_radius_cos: sky_cfg.sun_angular_radius_cos,
        };

        self.queue.write_buffer(
            &self.sky_uniforms_buffers[buffer_index],
            0,
            bytemuck::bytes_of(&sky_uniforms),
        );

        if sky_uniforms != self.prev_sky_uniforms {
            self.prev_sky_uniforms = sky_uniforms;

            self.needs_atmosphere_precompute = true;
        }

        self.prev_view_proj = new_view_proj;

        let output_frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost) => {
                self.resize(self.window_size);
                return Ok(());
            }
            Err(e) => {
                return Err(RendererError::ResourceCreation(format!(
                    "Surface error: {}",
                    e
                )));
            }
        };

        let output_view = output_frame.texture.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Command Encoder"),
            });

        // --- ATMOSPHERE PRECOMPUTATION PASS ---
        if self.needs_atmosphere_precompute {
            self.needs_atmosphere_precompute = false;

            if let (Some(atmo), Some(data)) = (
                self.atmosphere_precomputer.as_ref(),
                &self.current_render_data,
            ) {
                atmo.precompute(
                    &mut encoder,
                    &self.queue,
                    &sky_uniforms,
                    &data.render_config.shader_constants,
                );
            }
        }

        let camera_transform = &render_data.current_camera_transform;
        let eye = camera_transform.position;
        let forward = camera_transform.forward();
        let up = camera_transform.up();
        let static_camera_view = Mat4::look_at_rh(eye, eye + forward, up);

        if render_data.render_config.shadow_pass {
            self.shadow_pass.run_shadow_pass(
                &self.device,
                &self.queue,
                &mut encoder,
                self.render_constants_bind_group.as_ref().unwrap(),
                &self.meshes,
                render_data,
                &static_camera_view,
                alpha,
            );
        }

        self.run_geometry_pass(&mut encoder, render_data, alpha);

        if render_data.render_config.sky_pass {
            self.run_sky_pass(&mut encoder);
        }

        if render_data.render_config.direct_lighting_pass {
            self.run_lighting_pass(&mut encoder);
        }

        self.run_downsample_pass(&mut encoder);

        if render_data.render_config.ssgi_pass {
            self.run_ssgi_pass(&mut encoder);

            if render_data.render_config.ssgi_denoise_pass {
                self.run_ssgi_denoise_pass(&mut encoder);
            }

            encoder.copy_texture_to_texture(
                self.ssgi_denoised_half_texture
                    .as_ref()
                    .unwrap()
                    .as_image_copy(),
                self.ssgi_history_half_texture
                    .as_ref()
                    .unwrap()
                    .as_image_copy(),
                wgpu::Extent3d {
                    width: (self.surface_config.width / 2).max(1),
                    height: (self.surface_config.height / 2).max(1),
                    depth_or_array_layers: 1,
                },
            );

            self.run_ssgi_upsample_pass(&mut encoder);
        }

        if render_data.render_config.ssr_pass {
            self.run_ssr_pass(&mut encoder);
        }
        self.run_composite_pass(&mut encoder, &output_view);

        encoder.copy_texture_to_texture(
            output_frame.texture.as_image_copy(),
            self.history_texture.as_ref().unwrap().as_image_copy(),
            wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
        );

        if render_data.render_config.egui_pass {
            if let Some(egui_data) = &mut self.current_egui_data {
                let screen_descriptor = &mut egui_data.screen_descriptor;
                screen_descriptor.size_in_pixels =
                    [self.window_size.width, self.window_size.height];

                self.egui_renderer.update_buffers(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &egui_data.primitives,
                    &screen_descriptor,
                );

                {
                    let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Egui Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &output_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    self.egui_renderer.render(
                        &mut rpass.forget_lifetime(),
                        &egui_data.primitives,
                        &screen_descriptor,
                    );
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output_frame.present();

        self.frame_index = self.frame_index.wrapping_add(1);

        Ok(())
    }

    fn process_message(&mut self, message: RenderMessage) {
        match message {
            RenderMessage::CreateMesh {
                id,
                total_lods: _,
                lods,
                bounds,
            } => {
                self.add_mesh(id, &lods, bounds).unwrap();
            }
            RenderMessage::CreateTexture {
                id,
                name: _,
                kind,
                data,
                format,
                width,
                height,
            } => {
                if let Ok(gpu_index) = self.add_texture(data.as_ref(), kind, format, width, height)
                {
                    self.handle_id_to_texture_index.insert(id, gpu_index);
                }
            }
            RenderMessage::CreateMaterial(mat_data) => {
                self.pending_materials.push(mat_data);
            }
            RenderMessage::RenderData(data) => self.update_render_data(data),
            RenderMessage::RenderDelta(_) => {}
            RenderMessage::EguiData(data) => {
                // Upload textures immediately when message arrives
                for (id, delta) in &data.textures_delta.set {
                    self.egui_renderer
                        .update_texture(&self.device, &self.queue, *id, delta);
                }

                // Free old textures
                for id in &data.textures_delta.free {
                    self.egui_renderer.free_texture(id);
                }

                self.current_egui_data = Some(data);
            }
            RenderMessage::Control(_) => {}
            RenderMessage::Resize(size) => self.resize(size),
            RenderMessage::WindowRecreated { .. } => {}
            RenderMessage::Shutdown => {}
        }
    }

    fn update_render_data(&mut self, render_data: Arc<RenderData>) {
        if let Some(current_data) = &self.current_render_data {
            if current_data.render_config != render_data.render_config {
                if current_data.render_config.shader_constants
                    != render_data.render_config.shader_constants
                {
                    self.queue.write_buffer(
                        self.render_constants_buffer.as_ref().unwrap(),
                        0,
                        bytemuck::bytes_of(&render_data.render_config.shader_constants),
                    );

                    if current_data
                        .render_config
                        .shader_constants
                        .sky_light_samples
                        != render_data.render_config.shader_constants.sky_light_samples
                        || current_data.render_config.shader_constants.planet_radius
                            != render_data.render_config.shader_constants.planet_radius
                        || current_data
                            .render_config
                            .shader_constants
                            .atmosphere_radius
                            != render_data.render_config.shader_constants.atmosphere_radius
                    {
                        self.needs_atmosphere_precompute = true;
                    }
                } else if current_data.render_config.shadow_pass
                    != render_data.render_config.shadow_pass
                {
                    self.shadow_pass.create_shadow_resources(&self.device);
                    self.resize(self.window_size);
                } else if current_data.render_config.direct_lighting_pass
                    != render_data.render_config.direct_lighting_pass
                {
                    self.resize(self.window_size);
                } else if current_data.render_config.sky_pass != render_data.render_config.sky_pass
                {
                    self.resize(self.window_size);
                } else if current_data.render_config.ssr_pass != render_data.render_config.ssr_pass
                {
                    self.resize(self.window_size);
                } else if current_data.render_config.ssgi_pass
                    != render_data.render_config.ssgi_pass
                {
                    self.resize(self.window_size);
                }
            }
        }
        self.current_render_data = Some(render_data);
    }

    fn resolve_pending_materials(&mut self) {
        if self.pending_materials.is_empty() {
            return;
        }

        let mut newly_completed = Vec::new();

        self.pending_materials.retain(|mat_data| {
            let resolve = |id: Option<usize>| -> (bool, i32) {
                // Return i32
                match id {
                    // No texture for this slot. It's "ready" with a sentinel index.
                    None => (true, -1),
                    Some(handle_id) => {
                        if let Some(&gpu_index) = self.handle_id_to_texture_index.get(&handle_id) {
                            (true, gpu_index as i32)
                        } else {
                            // Texture asset is not loaded on GPU yet.
                            (false, -1)
                        }
                    }
                }
            };

            let (albedo_ready, albedo_idx) = resolve(mat_data.albedo_texture_id);
            let (normal_ready, normal_idx) = resolve(mat_data.normal_texture_id);
            let (mr_ready, mr_idx) = resolve(mat_data.metallic_roughness_texture_id);
            let (emission_ready, emission_idx) = resolve(mat_data.emission_texture_id);

            if albedo_ready && normal_ready && mr_ready && emission_ready {
                let final_material = Material {
                    albedo: mat_data.albedo,
                    metallic: mat_data.metallic,
                    roughness: mat_data.roughness,
                    ao: mat_data.ao,
                    emission_strength: mat_data.emission_strength,
                    emission_color: mat_data.emission_color,
                    albedo_texture_index: albedo_idx,
                    normal_texture_index: normal_idx,
                    metallic_roughness_texture_index: mr_idx,
                    emission_texture_index: emission_idx,
                    alpha_mode: mat_data.alpha_mode,
                    alpha_cutoff: mat_data.alpha_cutoff,
                };

                newly_completed.push((mat_data.id, final_material));
                return false;
            }

            true
        });

        for (id, material) in newly_completed {
            if let Err(e) = self.add_material(id, material) {
                warn!("Failed to add material {}: {}", id, e);
            }
        }
    }
}

// --- BIND GROUP HELPER FUNCTIONS ---

fn buffer_binding(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_binding(
    binding: u32,
    filterable: bool,
    view_dimension: wgpu::TextureViewDimension,
    is_depth: bool,
) -> wgpu::BindGroupLayoutEntry {
    let sample_type = if is_depth {
        wgpu::TextureSampleType::Depth
    } else {
        wgpu::TextureSampleType::Float { filterable }
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_binding(
    binding: u32,
    filtering: bool,
    _ty: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    let ty = if filtering {
        wgpu::SamplerBindingType::Filtering
    } else {
        wgpu::SamplerBindingType::NonFiltering
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}
