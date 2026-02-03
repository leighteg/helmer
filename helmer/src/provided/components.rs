use crate::animation::{Pose, Skeleton, Skin};
use crate::graphics::common::renderer::Vertex;
use glam::{Mat4, Quat, Vec2, Vec3};
use std::sync::Arc;

// --- Core Transform Component ---
// This is fundamental for any spatial entity.
#[derive(Debug, Clone, Copy)]
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
        Self {
            position,
            rotation,
            scale,
        }
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

    pub fn from_matrix(matrix: Mat4) -> Self {
        let transformed = matrix.to_scale_rotation_translation();
        Self {
            position: transformed.2,
            rotation: transformed.1,
            scale: transformed.0,
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    pub fn from_position(position: [f32; 3]) -> Self {
        Self {
            position: Vec3::from_array(position),
            rotation: Quat::default(),
            scale: Vec3::ONE,
        }
    }
}

// --- Mesh Rendering Component ---
// This component links an entity to a specific mesh asset.
// The `u32` ID would refer to an ID managed by your asset system,
// which the renderer then uses to look up the `mev::Buffer`s.
#[derive(Debug, Clone, Copy)]
pub struct MeshRenderer {
    pub mesh_id: usize,
    pub material_id: usize, // The ID of the material to use for this mesh instance
    pub casts_shadow: bool,
    pub visible: bool,
}

impl MeshRenderer {
    pub fn new(mesh_id: usize, material_id: usize, casts_shadow: bool, visible: bool) -> Self {
        Self {
            mesh_id,
            material_id,
            casts_shadow,
            visible,
        }
    }
}

// --- Skinned Mesh Rendering Component ---
// Uses a skin/skeleton to deform vertices.
#[derive(Debug, Clone)]
pub struct SkinnedMeshRenderer {
    pub mesh_id: usize,
    pub material_id: usize,
    pub casts_shadow: bool,
    pub visible: bool,
    pub skin: Arc<Skin>,
}

impl SkinnedMeshRenderer {
    pub fn new(
        mesh_id: usize,
        material_id: usize,
        skin: Arc<Skin>,
        casts_shadow: bool,
        visible: bool,
    ) -> Self {
        Self {
            mesh_id,
            material_id,
            casts_shadow,
            visible,
            skin,
        }
    }
}

// --- Pose Override Component ---
// Editor/runtime override for manual posing or timeline playback.
#[derive(Debug, Clone)]
pub struct PoseOverride {
    pub enabled: bool,
    pub pose: Pose,
}

impl PoseOverride {
    pub fn new(skeleton: &Skeleton) -> Self {
        Self {
            enabled: true,
            pose: Pose::from_skeleton(skeleton),
        }
    }

    pub fn reset_to_bind(&mut self, skeleton: &Skeleton) {
        self.pose.reset_to_bind(skeleton);
    }
}

impl Default for PoseOverride {
    fn default() -> Self {
        Self {
            enabled: false,
            pose: Pose { locals: Vec::new() },
        }
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub light_type: LightType,
    pub color: Vec3,    // RGB color of the light
    pub intensity: f32, // Brightness of the light
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub fov_y_rad: f32,    // Vertical field of view in radians
    pub aspect_ratio: f32, // Width / Height
    pub near_plane: f32,
    pub far_plane: f32,
    // Add other camera properties if needed, e.g., orthographic projection settings
}

impl Camera {
    pub fn new(fov_y_rad: f32, aspect_ratio: f32, near_plane: f32, far_plane: f32) -> Self {
        Self {
            fov_y_rad,
            aspect_ratio,
            near_plane,
            far_plane,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        let fov_45_degrees = std::f32::consts::FRAC_PI_4;
        Camera::new(fov_45_degrees, 1.7, 0.1, 100.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ActiveCamera {}

// --- Material Asset Component ---
// This component acts as a descriptor for a material that can be loaded and used by the renderer.
// It doesn't reside on an entity directly but rather describes a material asset.
// You would register these with your asset manager.
#[derive(Debug, Clone)] // Make it a Component if you store materials in your ECS
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

#[derive(Debug, Clone)]
pub struct MeshAsset {
    pub name: String,
    pub vertices: Option<Vec<Vertex>>, // Store Vec<Vertex> directly
    pub indices: Vec<u32>,
    pub mesh_file_path: Option<String>,
}

// --- Spline Components ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineMode {
    Linear,
    CatmullRom,
    Bezier,
}

#[derive(Debug, Clone)]
pub struct Spline {
    pub points: Vec<Vec3>,
    pub closed: bool,
    pub mode: SplineMode,
    pub tension: f32,
}

impl Default for Spline {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            closed: false,
            mode: SplineMode::CatmullRom,
            tension: 0.5,
        }
    }
}

impl Spline {
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn is_valid(&self) -> bool {
        self.points.len() >= 2
    }

    pub fn sample(&self, t: f32) -> Vec3 {
        if self.points.is_empty() {
            return Vec3::ZERO;
        }
        if self.points.len() == 1 {
            return self.points[0];
        }
        let t = t.clamp(0.0, 1.0);
        match self.mode {
            SplineMode::Linear => self.sample_linear(t),
            SplineMode::CatmullRom => self.sample_catmull_rom(t),
            SplineMode::Bezier => self.sample_bezier(t),
        }
    }

    pub fn sample_tangent(&self, t: f32) -> Vec3 {
        let eps = 1e-3;
        let t0 = (t - eps).clamp(0.0, 1.0);
        let t1 = (t + eps).clamp(0.0, 1.0);
        let p0 = self.sample(t0);
        let p1 = self.sample(t1);
        let dir = p1 - p0;
        if dir.length_squared() > 0.0 {
            dir.normalize()
        } else {
            Vec3::ZERO
        }
    }

    pub fn approx_length(&self, samples: usize) -> f32 {
        if self.points.len() < 2 || samples < 2 {
            return 0.0;
        }
        let mut length = 0.0;
        let mut prev = self.sample(0.0);
        let step = 1.0 / (samples.saturating_sub(1) as f32);
        for i in 1..samples {
            let t = i as f32 * step;
            let current = self.sample(t);
            length += current.distance(prev);
            prev = current;
        }
        length
    }

    fn sample_linear(&self, t: f32) -> Vec3 {
        let segment_count = if self.closed {
            self.points.len()
        } else {
            self.points.len().saturating_sub(1)
        };
        if segment_count == 0 {
            return self.points[0];
        }
        let f = t * segment_count as f32;
        let seg = f.floor() as usize;
        let local_t = f - seg as f32;
        let i0 = seg.min(segment_count - 1);
        let i1 = if self.closed {
            (i0 + 1) % self.points.len()
        } else {
            (i0 + 1).min(self.points.len() - 1)
        };
        self.points[i0].lerp(self.points[i1], local_t)
    }

    fn sample_catmull_rom(&self, t: f32) -> Vec3 {
        if self.points.len() < 4 {
            return self.sample_linear(t);
        }
        let segment_count = if self.closed {
            self.points.len()
        } else {
            self.points.len().saturating_sub(1)
        };
        if segment_count == 0 {
            return self.points[0];
        }
        let f = t * segment_count as f32;
        let seg = f.floor() as usize;
        let local_t = f - seg as f32;
        let idx = |i: isize| -> usize {
            if self.closed {
                let len = self.points.len() as isize;
                (i.rem_euclid(len)) as usize
            } else {
                i.clamp(0, (self.points.len() - 1) as isize) as usize
            }
        };
        let i1 = seg as isize;
        let p0 = self.points[idx(i1 - 1)];
        let p1 = self.points[idx(i1)];
        let p2 = self.points[idx(i1 + 1)];
        let p3 = self.points[idx(i1 + 2)];
        catmull_rom(p0, p1, p2, p3, local_t, self.tension)
    }

    fn sample_bezier(&self, t: f32) -> Vec3 {
        if self.points.len() < 4 {
            return self.sample_linear(t);
        }
        let segment_count = (self.points.len() - 1) / 3;
        if segment_count == 0 {
            return self.points[0];
        }
        let f = t * segment_count as f32;
        let seg = f.floor() as usize;
        let local_t = f - seg as f32;
        let base = (seg * 3).min(self.points.len() - 4);
        let p0 = self.points[base];
        let p1 = self.points[base + 1];
        let p2 = self.points[base + 2];
        let p3 = self.points[base + 3];
        cubic_bezier(p0, p1, p2, p3, local_t)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SplineFollower {
    pub spline_entity: Option<u64>,
    pub t: f32,
    pub speed: f32,
    pub looped: bool,
    pub follow_rotation: bool,
    pub up: Vec3,
    pub offset: Vec3,
    pub length_samples: u32,
}

impl Default for SplineFollower {
    fn default() -> Self {
        Self {
            spline_entity: None,
            t: 0.0,
            speed: 1.0,
            looped: true,
            follow_rotation: true,
            up: Vec3::Y,
            offset: Vec3::ZERO,
            length_samples: 32,
        }
    }
}

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32, tension: f32) -> Vec3 {
    let alpha = tension.clamp(0.0, 1.0);
    let eps = 1.0e-4;
    let d01 = (p1 - p0).length().max(eps).powf(alpha);
    let d12 = (p2 - p1).length().max(eps).powf(alpha);
    let d23 = (p3 - p2).length().max(eps).powf(alpha);

    let t0 = 0.0;
    let t1 = t0 + d01;
    let t2 = t1 + d12;
    let t3 = t2 + d23;

    let t = t1 + (t2 - t1) * t;

    let lerp_param = |a: Vec3, b: Vec3, ta: f32, tb: f32, tt: f32| -> Vec3 {
        let denom = (tb - ta).abs().max(eps);
        a * ((tb - tt) / denom) + b * ((tt - ta) / denom)
    };

    let a1 = lerp_param(p0, p1, t0, t1, t);
    let a2 = lerp_param(p1, p2, t1, t2, t);
    let a3 = lerp_param(p2, p3, t2, t3, t);

    let b1 = lerp_param(a1, a2, t0, t2, t);
    let b2 = lerp_param(a2, a3, t1, t3, t);

    lerp_param(b1, b2, t1, t2, t)
}

fn cubic_bezier(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    let uuu = uu * u;
    let ttt = tt * t;
    p0 * uuu + p1 * (3.0 * uu * t) + p2 * (3.0 * u * tt) + p3 * ttt
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
            Vertex {
                position: [-0.5, -0.5, 0.5],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.5],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.5],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.5],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, -0.5, -0.5],
                normal: [0.0, 0.0, -1.0],
                tex_coord: [1.0, 1.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.5, -0.5],
                normal: [0.0, 0.0, -1.0],
                tex_coord: [0.0, 1.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5, -0.5],
                normal: [0.0, 0.0, -1.0],
                tex_coord: [0.0, 0.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.5, -0.5],
                normal: [0.0, 0.0, -1.0],
                tex_coord: [1.0, 0.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.5],
                normal: [0.0, 1.0, 0.0],
                tex_coord: [0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.5],
                normal: [0.0, 1.0, 0.0],
                tex_coord: [1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5, -0.5],
                normal: [0.0, 1.0, 0.0],
                tex_coord: [1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.5, -0.5],
                normal: [0.0, 1.0, 0.0],
                tex_coord: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, -0.5, 0.5],
                normal: [0.0, -1.0, 0.0],
                tex_coord: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.5],
                normal: [0.0, -1.0, 0.0],
                tex_coord: [1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.5, -0.5],
                normal: [0.0, -1.0, 0.0],
                tex_coord: [1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, -0.5, -0.5],
                normal: [0.0, -1.0, 0.0],
                tex_coord: [0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.5],
                normal: [1.0, 0.0, 0.0],
                tex_coord: [0.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.5, -0.5],
                normal: [1.0, 0.0, 0.0],
                tex_coord: [1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5, -0.5],
                normal: [1.0, 0.0, 0.0],
                tex_coord: [1.0, 0.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.5],
                normal: [1.0, 0.0, 0.0],
                tex_coord: [0.0, 0.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, -0.5, 0.5],
                normal: [-1.0, 0.0, 0.0],
                tex_coord: [1.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, -0.5, -0.5],
                normal: [-1.0, 0.0, 0.0],
                tex_coord: [0.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.5, -0.5],
                normal: [-1.0, 0.0, 0.0],
                tex_coord: [0.0, 0.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.5],
                normal: [-1.0, 0.0, 0.0],
                tex_coord: [1.0, 0.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0, // Front
            4, 7, 6, 6, 5, 4, // Back (was 4,6,5,6,4,7)
            8, 9, 10, 10, 11, 8, // Top
            12, 15, 14, 14, 13, 12, // Bottom (was 12,14,13,14,12,15)
            16, 17, 18, 18, 19, 16, // Right
            20, 23, 22, 22, 21, 20, // Left (was 20,22,21,22,20,23)
        ];
        Self::new_raw(name, vertices, indices)
    }

    pub fn plane(name: String) -> Self {
        let vertices = vec![
            Vertex {
                position: [-0.5, 0.0, 0.5],
                normal: [0.0, 1.0, 0.0],
                tex_coord: [0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.0, 0.5],
                normal: [0.0, 1.0, 0.0],
                tex_coord: [1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.0, -0.5],
                normal: [0.0, 1.0, 0.0],
                tex_coord: [1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.0, -0.5],
                normal: [0.0, 1.0, 0.0],
                tex_coord: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                joints: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let indices = vec![0, 1, 2, 2, 3, 0];
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

                vertices.push(Vertex::new(
                    position.into(),
                    normal.into(),
                    tex_coord.into(),
                    tangent.extend(0.0).into(),
                ));
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
#[derive(Debug, Clone, Copy)]
pub struct MeshAssetId(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct MaterialAssetId(pub u32);

#[derive(Debug, Clone, Copy)]
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
