use crate::{ecs::component::Component, graphics::renderer::renderer::Vertex};
use proc::Component;
use glam::{Quat, Vec2, Vec3}; // Using glam for mathematical types

// --- Core Transform Component ---
// This is fundamental for any spatial entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat, // Using Quat for rotations
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self { position, rotation, scale }
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::Z // Assuming Z is forward in local space
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X // Assuming X is right in local space
    }

    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y // Assuming Y is up in local space
    }
}


// --- Mesh Rendering Component ---
// This component links an entity to a specific mesh asset.
// The `u32` ID would refer to an ID managed by your asset system,
// which the renderer then uses to look up the `mev::Buffer`s.
#[derive(Component, Debug, Clone, Copy)]
pub struct MeshRenderer {
    pub mesh_id: usize,
    pub material_id: usize, // The ID of the material to use for this mesh instance
}

impl MeshRenderer {
    pub fn new(mesh_id: usize, material_id: usize) -> Self {
        Self { mesh_id, material_id }
    }
}

// --- Light Components ---
// Different types of lights will have different properties.
// We'll define a base `Light` component and then specific data components.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot { angle: f32 }, // Spot lights might have a cone angle
}

#[derive(Component, Debug, Clone, Copy)]
pub struct Light {
    pub light_type: LightType,
    pub color: Vec3,      // RGB color of the light
    pub intensity: f32,   // Brightness of the light
}

impl Light {
    pub fn directional(color: Vec3, intensity: f32) -> Self {
        Self {
            light_type: LightType::Directional,
            color,
            intensity,
        }
    }

    pub fn point(color: Vec3, intensity: f32) -> Self {
        Self {
            light_type: LightType::Point,
            color,
            intensity,
        }
    }

    pub fn spot(color: Vec3, intensity: f32, angle: f32) -> Self {
        Self {
            light_type: LightType::Spot { angle },
            color,
            intensity,
        }
    }
}


// --- Camera Component ---
// Defines a camera in your world. The ECS system would select one active camera
// to provide data for the renderer's camera uniforms.
#[derive(Component, Debug, Clone, Copy)]
pub struct Camera {
    pub fov_y_rad: f32,      // Vertical field of view in radians
    pub aspect_ratio: f32,   // Width / Height
    pub near_plane: f32,
    pub far_plane: f32,
    // Add other camera properties if needed, e.g., orthographic projection settings
}

impl Camera {
    pub fn new(fov_y_rad: f32, aspect_ratio: f32, near_plane: f32, far_plane: f32) -> Self {
        Self { fov_y_rad, aspect_ratio, near_plane, far_plane }
    }
}

// --- Material Asset Component ---
// This component acts as a descriptor for a material that can be loaded and used by the renderer.
// It doesn't reside on an entity directly but rather describes a material asset.
// You would register these with your asset manager.
#[derive(Component, Debug, Clone)] // Make it a Component if you store materials in your ECS
pub struct MaterialAsset {
    pub name: String,
    pub albedo: Vec3, // Base color
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32, // Ambient occlusion
    pub albedo_texture_path: Option<String>,
    pub normal_texture_path: Option<String>,
    pub metallic_roughness_texture_path: Option<String>,
    // Add other texture paths (e.g., emissive, height) as needed
}

impl MaterialAsset {
    pub fn new_pbr(
        name: String,
        albedo: Vec3,
        metallic: f32,
        roughness: f32,
        ao: f32,
        albedo_texture_path: Option<String>,
        normal_texture_path: Option<String>,
        metallic_roughness_texture_path: Option<String>,
    ) -> Self {
        Self {
            name,
            albedo,
            metallic,
            roughness,
            ao,
            albedo_texture_path,
            normal_texture_path,
            metallic_roughness_texture_path,
        }
    }
}

// --- Mesh Asset Component ---
// Similar to MaterialAsset, this describes a mesh asset.
// In your ECS components or asset definition file (where MeshAsset is defined)

// In your `provided::components` module or wherever MeshAsset is defined

#[derive(Component, Debug, Clone)]
pub struct MeshAsset {
    pub name: String,
    pub vertices: Option<Vec<Vertex>>, // Store Vec<Vertex> directly
    pub indices: Vec<u32>,
    pub mesh_file_path: Option<String>,
}

impl MeshAsset {
    // New constructor for already generated/loaded raw data
    pub fn new_raw(name: String, vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self {
            name,
            vertices: Some(vertices),
            indices,
            mesh_file_path: None,
        }
    }

    // Existing constructor for file-based assets (vertices/indices loaded later)
    pub fn from_file(name: String, mesh_file_path: String) -> Self {
        Self {
            name,
            vertices: None,
            indices: Vec::new(),
            mesh_file_path: Some(mesh_file_path),
        }
    }

    // --- Default Primitive Mesh Generation Methods ---
    // These now use `Self::new_raw` to create the MeshAsset

    pub fn cube(name: String) -> Self {
        let vertices = vec![
            // ... (your existing cube vertex data)
            Vertex { position: [-0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 1.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [-0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0], tangent: [1.0, 0.0, 0.0] },

            Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], tex_coord: [1.0, 1.0], tangent: [-1.0, 0.0, 0.0] },
            Vertex { position: [0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], tex_coord: [0.0, 1.0], tangent: [-1.0, 0.0, 0.0] },
            Vertex { position: [0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], tex_coord: [0.0, 0.0], tangent: [-1.0, 0.0, 0.0] },
            Vertex { position: [-0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], tex_coord: [1.0, 0.0], tangent: [-1.0, 0.0, 0.0] },

            Vertex { position: [-0.5, 0.5, 0.5], normal: [0.0, 1.0, 0.0], tex_coord: [0.0, 1.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [0.5, 0.5, 0.5], normal: [0.0, 1.0, 0.0], tex_coord: [1.0, 1.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [0.5, 0.5, -0.5], normal: [0.0, 1.0, 0.0], tex_coord: [1.0, 0.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [-0.5, 0.5, -0.5], normal: [0.0, 1.0, 0.0], tex_coord: [0.0, 0.0], tangent: [1.0, 0.0, 0.0] },

            Vertex { position: [-0.5, -0.5, 0.5], normal: [0.0, -1.0, 0.0], tex_coord: [0.0, 0.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [0.5, -0.5, 0.5], normal: [0.0, -1.0, 0.0], tex_coord: [1.0, 0.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0], tex_coord: [1.0, 1.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0], tex_coord: [0.0, 1.0], tangent: [1.0, 0.0, 0.0] },

            Vertex { position: [0.5, -0.5, 0.5], normal: [1.0, 0.0, 0.0], tex_coord: [0.0, 1.0], tangent: [0.0, 0.0, -1.0] },
            Vertex { position: [0.5, -0.5, -0.5], normal: [1.0, 0.0, 0.0], tex_coord: [1.0, 1.0], tangent: [0.0, 0.0, -1.0] },
            Vertex { position: [0.5, 0.5, -0.5], normal: [1.0, 0.0, 0.0], tex_coord: [1.0, 0.0], tangent: [0.0, 0.0, -1.0] },
            Vertex { position: [0.5, 0.5, 0.5], normal: [1.0, 0.0, 0.0], tex_coord: [0.0, 0.0], tangent: [0.0, 0.0, -1.0] },

            Vertex { position: [-0.5, -0.5, 0.5], normal: [-1.0, 0.0, 0.0], tex_coord: [1.0, 1.0], tangent: [0.0, 0.0, 1.0] },
            Vertex { position: [-0.5, -0.5, -0.5], normal: [-1.0, 0.0, 0.0], tex_coord: [0.0, 1.0], tangent: [0.0, 0.0, 1.0] },
            Vertex { position: [-0.5, 0.5, -0.5], normal: [-1.0, 0.0, 0.0], tex_coord: [0.0, 0.0], tangent: [0.0, 0.0, 1.0] },
            Vertex { position: [-0.5, 0.5, 0.5], normal: [-1.0, 0.0, 0.0], tex_coord: [1.0, 0.0], tangent: [0.0, 0.0, 1.0] },
        ];

        let indices = vec![
    0, 1, 2, 2, 3, 0,     // Front
    4, 7, 6, 6, 5, 4,     // Back (was 4,6,5,6,4,7)
    8, 9, 10, 10, 11, 8,  // Top
    12, 15, 14, 14, 13, 12, // Bottom (was 12,14,13,14,12,15)
    16, 17, 18, 18, 19, 16, // Right
    20, 23, 22, 22, 21, 20, // Left (was 20,22,21,22,20,23)
];
        Self::new_raw(name, vertices, indices)
    }

    pub fn plane(name: String) -> Self {
        let vertices = vec![
            Vertex { position: [-0.5, 0.0, 0.5], normal: [0.0, 1.0, 0.0], tex_coord: [0.0, 1.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [0.5, 0.0, 0.5], normal: [0.0, 1.0, 0.0], tex_coord: [1.0, 1.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [0.5, 0.0, -0.5], normal: [0.0, 1.0, 0.0], tex_coord: [1.0, 0.0], tangent: [1.0, 0.0, 0.0] },
            Vertex { position: [-0.5, 0.0, -0.5], normal: [0.0, 1.0, 0.0], tex_coord: [0.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        ];
        let indices = vec![
            0, 1, 2,
            2, 3, 0,
        ];
        Self::new_raw(name, vertices, indices)
    }

    pub fn uv_sphere(name: String, segments: u32, rings: u32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Vertices
        for r in 0..=rings {
            let v = r as f32 / rings as f32;
            let phi = v * std::f32::consts::PI;

            for s in 0..=segments {
                let u = s as f32 / segments as f32;
                let theta = u * std::f32::consts::TAU;

                let x = -phi.sin() * theta.sin();
                let y = phi.cos();
                let z = phi.sin() * theta.cos();

                let position = Vec3::new(x, y, z);
                let normal = position.normalize();
                let tex_coord = Vec2::new(u, 1.0 - v);

                let tangent = Vec3::new(-theta.cos(), 0.0, -theta.sin()).normalize();

                vertices.push(Vertex {
                    position: position.into(),
                    normal: normal.into(),
                    tex_coord: tex_coord.into(),
                    tangent: tangent.into(),
                });
            }
        }

        // Indices
        for r in 0..rings {
            for s in 0..segments {
                let p0 = r * (segments + 1) + s;
                let p1 = p0 + 1;
                let p2 = (r + 1) * (segments + 1) + s;
                let p3 = p2 + 1;
        
                // Counter-Clockwise (CCW) winding
                indices.push(p0);
                indices.push(p1);
                indices.push(p2);
        
                indices.push(p2);
                indices.push(p1);
                indices.push(p3);
            }
        }
        Self::new_raw(name, vertices, indices)
    }
}

// --- Asset ID Mapping Component ---
// This component would be used if you have a separate asset loading/management system
// that maps asset paths/names to generated runtime IDs (u32).
// Entities would have these components to refer to loaded assets.
#[derive(Component, Debug, Clone, Copy)]
pub struct MeshAssetId(pub u32);

#[derive(Component, Debug, Clone, Copy)]
pub struct MaterialAssetId(pub u32);

#[derive(Component, Debug, Clone, Copy)]
pub struct TextureAssetId(pub u32);


// --- Example ECS System to Collect Render Data ---
// This is a conceptual system that would run every frame to query your ECS
// and build the `RenderData` structure for the renderer.

/*
use arcana::edict::World; // Or whatever your ECS world type is

pub fn collect_render_data_system(
    world: &mut World, // Or immutable reference if you use queries
) {
    let mut render_objects: Vec<RenderObject> = Vec::new();
    let mut render_lights: Vec<RenderLight> = Vec::new();
    let mut camera_transform: Option<Transform> = None; // Assuming one main camera

    // Query for renderable entities
    // Example: Query all entities with Transform and MeshRenderer
    for (entity_id, (transform, mesh_renderer)) in world.query::<(Transform, MeshRenderer)>().iter() {
        render_objects.push(RenderObject {
            transform: *transform,
            mesh_id: mesh_renderer.mesh_id,
            material_id: mesh_renderer.material_id,
        });
    }

    // Query for lights
    // Example: Query all entities with Transform and Light
    for (entity_id, (transform, light)) in world.query::<(Transform, Light)>().iter() {
        render_lights.push(RenderLight {
            transform: *transform,
            color: light.color.into(), // Convert glam::Vec3 to [f32; 3]
            intensity: light.intensity,
            light_type: light.light_type,
        });
    }

    // Query for the main camera
    // Example: Find the first entity with Camera and Transform (or tag a main camera)
    if let Some((_, (transform, camera))) = world.query::<(Transform, Camera)>().iter().next() {
        // You might need to adjust the camera transform to be relative to the world
        // if your camera component stores local transform.
        camera_transform = Some(*transform);
    }


    // Now, update your renderer's data:
    if let Some(cam_transform) = camera_transform {
        let renderer = world.get_resource_mut::<crate::renderer::Renderer>().unwrap(); // Assuming Renderer is a resource
        renderer.update_render_data(crate::renderer::RenderData {
            objects: render_objects,
            lights: render_lights,
            camera_transform: cam_transform,
        });
    } else {
        tracing::warn!("No active camera found in ECS for rendering.");
    }
}
*/