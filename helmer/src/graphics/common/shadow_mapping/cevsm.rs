use std::cell::RefCell;

use glam::{Mat4, Quat, Vec3, Vec4Swizzles as _};
use hashbrown::HashMap;

use crate::{
    graphics::common::{
        constants::{CASCADE_SPLITS, NUM_CASCADES, SHADOW_MAP_RESOLUTION},
        renderer::{
            Aabb, CascadeUniform, InstanceRaw, Mesh, RenderData, ShadowPipeline, ShadowUniforms,
            Vertex,
        },
    },
    provided::components::{Camera, LightType},
};

pub struct CascadedEVSMPass {
    shadow_pipeline: Option<ShadowPipeline>,

    shadow_instance_buffer: RefCell<Option<wgpu::Buffer>>,
    shadow_instance_capacity: RefCell<usize>,

    pub shadow_light_vp_buffer: Option<wgpu::Buffer>,
    pub shadow_uniforms_buffer: Option<wgpu::Buffer>,

    shadow_map_texture: Option<wgpu::Texture>,
    pub shadow_map_view: Option<wgpu::TextureView>,
    shadow_depth_texture: Option<wgpu::Texture>,
    pub shadow_depth_view: Option<wgpu::TextureView>,
    cascade_views: Option<Vec<wgpu::TextureView>>,

    pub shadow_sampler: Option<wgpu::Sampler>,
}

impl CascadedEVSMPass {
    pub fn new() -> Self {
        Self {
            shadow_pipeline: None,

            shadow_instance_buffer: RefCell::new(None),
            shadow_instance_capacity: RefCell::new(0),

            shadow_light_vp_buffer: None,
            shadow_uniforms_buffer: None,

            shadow_map_texture: None,
            shadow_map_view: None,
            shadow_depth_texture: None,
            shadow_depth_view: None,
            cascade_views: None,

            shadow_sampler: None,
        }
    }

    pub fn create_shadow_resources(&mut self, device: &wgpu::Device) {
        let shadow_texture_desc = wgpu::TextureDescriptor {
            label: Some("VSM Shadow Map Texture Array"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_RESOLUTION,
                height: SHADOW_MAP_RESOLUTION,
                depth_or_array_layers: NUM_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        self.shadow_map_texture = Some(device.create_texture(&shadow_texture_desc));
        self.shadow_map_view = Some(self.shadow_map_texture.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("VSM Shadow Map View"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            },
        ));

        self.shadow_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        let shadow_depth_desc = wgpu::TextureDescriptor {
            label: Some("Shadow Pass Depth Texture"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_RESOLUTION,
                height: SHADOW_MAP_RESOLUTION,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        self.shadow_depth_texture = Some(device.create_texture(&shadow_depth_desc));
        self.shadow_depth_view = Some(
            self.shadow_depth_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        );

        self.shadow_uniforms_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Uniforms Buffer"),
            size: std::mem::size_of::<ShadowUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let mut cascade_views = Vec::with_capacity(NUM_CASCADES);
        for i in 0..NUM_CASCADES {
            let cascade_view = self.shadow_map_texture.as_ref().unwrap().create_view(
                &wgpu::TextureViewDescriptor {
                    label: Some(&format!("Cascade View {}", i)),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                },
            );
            cascade_views.push(cascade_view);
        }
        self.cascade_views = Some(cascade_views);
    }

    pub fn create_shadow_pipeline(
        &mut self,
        device: &wgpu::Device,
        render_constants_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/shadow.wgsl"));

        let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
        let aligned_mat4_size = wgpu::util::align_to(mat4_size, alignment);

        self.shadow_light_vp_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Light VP (Dynamic Uniform)"),
            size: NUM_CASCADES as wgpu::BufferAddress * aligned_mat4_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(mat4_size),
                },
                count: None,
            }],
        });

        let mat4_size_val = std::mem::size_of::<[[f32; 4]; 4]>();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Light VP Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.shadow_light_vp_buffer.as_ref().unwrap(),
                    offset: 0,
                    size: wgpu::BufferSize::new(mat4_size_val as u64),
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, render_constants_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg32Float,
                    blend: None,
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
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        self.shadow_pipeline = Some(ShadowPipeline {
            pipeline,
            bind_group,
        });
    }

    pub fn run_shadow_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        render_constants_bind_group: &wgpu::BindGroup,
        meshes: &HashMap<usize, Mesh>,
        render_data: &RenderData,
        static_camera_view: &Mat4,
        alpha: f32,
    ) {
        const FAR_CASCADE_DISTANCE: f32 = 500.0;
        const MAX_SHADOW_CASTING_DISTANCE: f32 = FAR_CASCADE_DISTANCE * 1.5;
        const MAX_SHADOW_CASTING_DISTANCE_SQ: f32 =
            MAX_SHADOW_CASTING_DISTANCE * MAX_SHADOW_CASTING_DISTANCE;

        let shadow_light = render_data
            .lights
            .iter()
            .find(|l| matches!(l.light_type, LightType::Directional));

        if let (Some(light), Some(shadow_pipeline)) = (shadow_light, self.shadow_pipeline.as_ref())
        {
            // --- 1. Culling and Scene Bounds Calculation ---
            let camera_pos = render_data.current_camera_transform.position;
            let mut culled_objects = HashMap::new();
            let mut scene_bounds_min = Vec3::splat(f32::MAX);
            let mut scene_bounds_max = Vec3::splat(f32::MIN);

            let mut shadow_casters = Vec::new();

            for object in &render_data.objects {
                if object.casts_shadow {
                    let distance_sq = object
                        .current_transform
                        .position
                        .distance_squared(camera_pos);
                    let is_culled = distance_sq > MAX_SHADOW_CASTING_DISTANCE_SQ;
                    culled_objects.insert(object.id, is_culled);

                    if !is_culled {
                        shadow_casters.push(object); // Add to a list for sorting
                        if let Some(mesh) = meshes.get(&object.mesh_id) {
                            let model_matrix = Mat4::from_scale_rotation_translation(
                                object.current_transform.scale,
                                object.current_transform.rotation,
                                object.current_transform.position,
                            );
                            for &corner in &mesh.bounds.get_corners() {
                                let world_corner = (model_matrix * corner.extend(1.0)).xyz();
                                scene_bounds_min = scene_bounds_min.min(world_corner);
                                scene_bounds_max = scene_bounds_max.max(world_corner);
                            }
                        }
                    }
                }
            }
            let dynamic_scene_bounds = Aabb {
                min: scene_bounds_min,
                max: scene_bounds_max,
            };

            // --- 2. Cascade Calculation and Uniform Uploads ---
            let camera = &render_data.camera_component;
            let shadow_uniforms = self.calculate_cascades(
                camera,
                static_camera_view,
                light.current_transform.rotation,
                &dynamic_scene_bounds,
            );
            queue.write_buffer(
                self.shadow_uniforms_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&shadow_uniforms),
            );

            let mat4_size = std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress;
            let alignment =
                device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
            let aligned_mat4_size = wgpu::util::align_to(mat4_size, alignment);

            for i in 0..NUM_CASCADES {
                queue.write_buffer(
                    self.shadow_light_vp_buffer.as_ref().unwrap(),
                    (i as wgpu::BufferAddress) * aligned_mat4_size,
                    bytemuck::bytes_of(&shadow_uniforms.cascades[i].light_view_proj),
                );
            }

            // --- 3. Build CPU-side instance data and batch info ---
            // (Key is (mesh_id, lod_index))
            // (Value is (instance_offset, instance_count))
            let mut batch_info: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
            let mut all_instances: Vec<InstanceRaw> = Vec::new();

            // Sort objects by mesh/lod to create contiguous batches
            shadow_casters.sort_by_key(|obj| (obj.mesh_id, obj.lod_index));

            for object in shadow_casters {
                if let Some(mesh) = meshes.get(&object.mesh_id) {
                    if mesh.lods.is_empty() {
                        continue;
                    }
                    let lod_index = object.lod_index.min(mesh.lods.len() - 1);
                    let key = (object.mesh_id, lod_index);

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
                    let model_matrix =
                        Mat4::from_scale_rotation_translation(scale, rotation, position);

                    let instance_data = InstanceRaw {
                        model_matrix: model_matrix.to_cols_array_2d(),
                        skin_offset: 0,
                        skin_count: 0,
                        _pad0: [0; 2],
                    };

                    let current_offset = all_instances.len() as u32;
                    all_instances.push(instance_data);

                    let entry = batch_info.entry(key).or_insert((current_offset, 0));
                    entry.1 += 1;
                }
            }

            let total_instances = all_instances.len();
            if total_instances == 0 {
                return; // No shadows to cast
            }

            // --- 4. Check and resize the GPU buffer if needed ---
            let mut capacity = self.shadow_instance_capacity.borrow_mut();
            let mut buffer = self.shadow_instance_buffer.borrow_mut();

            if total_instances > *capacity || buffer.is_none() {
                if let Some(old_buffer) = buffer.take() {
                    old_buffer.destroy();
                }
                let new_capacity = (total_instances as f32 * 1.5).ceil() as usize;

                *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Shadow Instance Buffer"),
                    size: (new_capacity * std::mem::size_of::<InstanceRaw>())
                        as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                *capacity = new_capacity;
            }

            // --- 5. Upload all instance data in one go ---
            queue.write_buffer(
                buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&all_instances),
            );

            // --- 6. Run Render Pass ---
            for i in 0..NUM_CASCADES {
                let offset = (i as u32) * (aligned_mat4_size as u32);
                let cascade_view = &self.cascade_views.as_ref().unwrap()[i];
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("Shadow Pass {}", i)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &cascade_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 1.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.shadow_depth_view.as_ref().unwrap(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                shadow_pass.set_pipeline(&shadow_pipeline.pipeline);
                shadow_pass.set_bind_group(0, &shadow_pipeline.bind_group, &[offset]);
                shadow_pass.set_bind_group(1, render_constants_bind_group, &[]);

                // Bind the one persistent buffer
                shadow_pass.set_vertex_buffer(1, buffer.as_ref().unwrap().slice(..));

                // Draw all batches
                for ((mesh_id, lod_index), (instance_offset, instance_count)) in &batch_info {
                    if let Some(mesh) = meshes.get(mesh_id) {
                        let lod = &mesh.lods[*lod_index];

                        shadow_pass.set_vertex_buffer(0, lod.vertex_buffer.slice(..));
                        shadow_pass.set_index_buffer(
                            lod.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );

                        let instance_range = *instance_offset..(*instance_offset + *instance_count);
                        shadow_pass.draw_indexed(0..lod.index_count, 0, instance_range);
                    }
                }
            }
        }
    }

    fn calculate_cascades(
        &self,
        camera: &Camera,
        camera_view: &Mat4,
        light_rotation: Quat,
        scene_bounds: &Aabb,
    ) -> ShadowUniforms {
        let light_dir = (light_rotation * -Vec3::Z).normalize();
        let inv_camera_view = camera_view.inverse();
        let tan_half_fovy = (camera.fov_y_rad / 2.0).tan();
        let scene_corners = scene_bounds.get_corners();

        let mut uniforms = ShadowUniforms {
            cascade_count: NUM_CASCADES as u32,
            _pad0: [0; 3],
            _pad1: [0; 4],
            cascades: [CascadeUniform::default(); NUM_CASCADES],
        };

        for i in 0..NUM_CASCADES {
            let z_near = CASCADE_SPLITS[i];
            let z_far = CASCADE_SPLITS[i + 1];

            let h_near = 2.0 * tan_half_fovy * z_near;
            let w_near = h_near * camera.aspect_ratio;
            let h_far = 2.0 * tan_half_fovy * z_far;
            let w_far = h_far * camera.aspect_ratio;

            let corners_view = [
                Vec3::new(w_near / 2.0, h_near / 2.0, -z_near),
                Vec3::new(-w_near / 2.0, h_near / 2.0, -z_near),
                Vec3::new(w_near / 2.0, -h_near / 2.0, -z_near),
                Vec3::new(-w_near / 2.0, -h_near / 2.0, -z_near),
                Vec3::new(w_far / 2.0, h_far / 2.0, -z_far),
                Vec3::new(-w_far / 2.0, h_far / 2.0, -z_far),
                Vec3::new(w_far / 2.0, -h_far / 2.0, -z_far),
                Vec3::new(-w_far / 2.0, -h_far / 2.0, -z_far),
            ];

            let frustum_corners_world: [Vec3; 8] =
                std::array::from_fn(|i| (inv_camera_view * corners_view[i].extend(1.0)).xyz());

            let world_center = frustum_corners_world.iter().sum::<Vec3>() / 8.0;
            let light_view = Mat4::look_at_rh(world_center - light_dir, world_center, Vec3::Y);

            let mut cascade_min = Vec3::splat(f32::MAX);
            let mut cascade_max = Vec3::splat(f32::MIN);
            for corner in frustum_corners_world {
                let trf = light_view * corner.extend(1.0);
                cascade_min = cascade_min.min(trf.xyz());
                cascade_max = cascade_max.max(trf.xyz());
            }

            let mut scene_min_z = f32::MAX;
            let mut scene_max_z = f32::MIN;
            for corner in &scene_corners {
                let trf = light_view * corner.extend(1.0);
                scene_min_z = scene_min_z.min(trf.z);
                scene_max_z = scene_max_z.max(trf.z);
            }

            let light_proj = Mat4::orthographic_rh(
                cascade_min.x,
                cascade_max.x,
                cascade_min.y,
                cascade_max.y,
                -scene_max_z,
                -scene_min_z,
            );

            let final_light_vp = light_proj * light_view;
            uniforms.cascades[i] = CascadeUniform {
                light_view_proj: final_light_vp.to_cols_array_2d(),
                split_depth: [-z_far, 0.0, 0.0, 0.0],
            };
        }

        uniforms
    }
}
