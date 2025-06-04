use crate::ecs::component::Component;
use proc::Component;
use glam::{Vec3, Quat}; // Using glam for mathematical types

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
    pub mesh_id: u32,
    pub material_id: u32, // The ID of the material to use for this mesh instance
}

impl MeshRenderer {
    pub fn new(mesh_id: u32, material_id: u32) -> Self {
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
#[derive(Component, Debug, Clone)]
pub struct MeshAsset {
    pub name: String,
    pub vertex_data: Vec<u8>, // Raw vertex data, format depends on your `Vertex` struct
    pub index_data: Vec<u32>, // Raw index data
    // Or, more commonly, a path to a mesh file (e.g., GLTF, OBJ)
    pub mesh_file_path: Option<String>,
}

impl MeshAsset {
    pub fn from_raw_data(name: String, vertex_data: Vec<u8>, index_data: Vec<u32>) -> Self {
        Self {
            name,
            vertex_data,
            index_data,
            mesh_file_path: None,
        }
    }

    pub fn from_file(name: String, mesh_file_path: String) -> Self {
        Self {
            name,
            vertex_data: Vec::new(), // Will be loaded by asset system
            index_data: Vec::new(),  // Will be loaded by asset system
            mesh_file_path: Some(mesh_file_path),
        }
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