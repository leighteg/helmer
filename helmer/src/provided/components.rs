use crate::animation::{Pose, Skeleton, Skin};
use crate::audio::{AudioBus, AudioPlaybackState};
use crate::graphics::common::renderer::{
    AlphaMode, SpriteBlendMode, SpriteSheetAnimation, SpriteSpace, TextAlignH, TextAlignV,
    TextFontStyle, Vertex,
};
use glam::{Mat4, Quat, Vec2, Vec3};
use hashbrown::HashMap;
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

// --- Sprite Rendering Component ---
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteRenderer {
    pub color: [f32; 4],
    pub texture_id: Option<usize>,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub sheet_animation: SpriteSheetAnimation,
    pub pivot: [f32; 2],
    pub clip_rect: Option<[f32; 4]>,
    pub layer: f32,
    pub space: SpriteSpace,
    pub blend_mode: SpriteBlendMode,
    pub billboard: bool,
    pub visible: bool,
    pub pick_id: Option<u32>,
}

impl Default for SpriteRenderer {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            texture_id: None,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            sheet_animation: SpriteSheetAnimation::default(),
            pivot: [0.5, 0.5],
            clip_rect: None,
            layer: 0.0,
            space: SpriteSpace::World,
            blend_mode: SpriteBlendMode::Alpha,
            billboard: false,
            visible: true,
            pick_id: None,
        }
    }
}

// --- World/Screen Text Rendering Component ---
#[derive(Debug, Clone, PartialEq)]
pub struct Text2d {
    pub text: String,
    pub color: [f32; 4],
    pub font_path: Option<String>,
    pub font_family: Option<String>,
    pub font_size: f32,
    pub font_weight: f32,
    pub font_width: f32,
    pub font_style: TextFontStyle,
    pub line_height_scale: f32,
    pub letter_spacing: f32,
    pub word_spacing: f32,
    pub underline: bool,
    pub strikethrough: bool,
    pub max_width: Option<f32>,
    pub align_h: TextAlignH,
    pub align_v: TextAlignV,
    pub space: SpriteSpace,
    pub billboard: bool,
    pub blend_mode: SpriteBlendMode,
    pub layer: f32,
    pub clip_rect: Option<[f32; 4]>,
    pub visible: bool,
    pub pick_id: Option<u32>,
}

impl Default for Text2d {
    fn default() -> Self {
        Self {
            text: String::new(),
            color: [1.0, 1.0, 1.0, 1.0],
            font_path: None,
            font_family: None,
            font_size: 16.0,
            font_weight: 400.0,
            font_width: 1.0,
            font_style: TextFontStyle::Normal,
            line_height_scale: 1.0,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            underline: false,
            strikethrough: false,
            max_width: None,
            align_h: TextAlignH::Left,
            align_v: TextAlignV::Baseline,
            space: SpriteSpace::World,
            billboard: false,
            blend_mode: SpriteBlendMode::Alpha,
            layer: 0.0,
            clip_rect: None,
            visible: true,
            pick_id: None,
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

// --- Audio Components ---
#[derive(Debug, Clone, Copy)]
pub struct AudioListener {
    pub enabled: bool,
}

impl Default for AudioListener {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioEmitter {
    pub clip_id: Option<usize>,
    pub bus: AudioBus,
    pub volume: f32,
    pub pitch: f32,
    pub looping: bool,
    pub spatial: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: f32,
    pub spatial_blend: f32,
    pub playback_state: AudioPlaybackState,
    pub play_on_spawn: bool,
}

impl Default for AudioEmitter {
    fn default() -> Self {
        Self {
            clip_id: None,
            bus: AudioBus::Sfx,
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            spatial: true,
            min_distance: 1.0,
            max_distance: 50.0,
            rolloff: 1.0,
            spatial_blend: 1.0,
            playback_state: AudioPlaybackState::Playing,
            play_on_spawn: true,
        }
    }
}

// --- Material Asset Component ---
// This component acts as a descriptor for a material that can be loaded and used by the renderer.
// It doesn't reside on an entity directly but rather describes a material asset.
#[derive(Debug, Clone)]
pub struct MaterialAsset {
    pub name: String,
    pub albedo: Vec3, // Base color
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32, // Ambient occlusion
    pub albedo_texture_path: Option<String>,
    pub normal_texture_path: Option<String>,
    pub metallic_roughness_texture_path: Option<String>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: Option<f32>,
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
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: None,
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

// --- Look-at Component ---
// Rotates an entity to face a target, with optional smoothing.
#[derive(Debug, Clone, Copy)]
pub struct LookAt {
    pub target_entity: Option<u64>,
    pub target_offset: Vec3,
    pub offset_in_target_space: bool,
    pub up: Vec3,
    pub rotation_smooth_time: f32,
}

impl Default for LookAt {
    fn default() -> Self {
        Self {
            target_entity: None,
            target_offset: Vec3::ZERO,
            offset_in_target_space: true,
            up: Vec3::Y,
            rotation_smooth_time: 0.0,
        }
    }
}

// --- Entity Follow Component ---
// Moves (and optionally rotates) an entity to follow another entity.
#[derive(Debug, Clone, Copy)]
pub struct EntityFollower {
    pub target_entity: Option<u64>,
    pub position_offset: Vec3,
    pub offset_in_target_space: bool,
    pub follow_rotation: bool,
    pub position_smooth_time: f32,
    pub rotation_smooth_time: f32,
}

impl Default for EntityFollower {
    fn default() -> Self {
        Self {
            target_entity: None,
            position_offset: Vec3::ZERO,
            offset_in_target_space: true,
            follow_rotation: false,
            position_smooth_time: 0.0,
            rotation_smooth_time: 0.0,
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

fn icosphere_midpoint(
    a: u32,
    b: u32,
    vertices: &mut Vec<Vec3>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(idx) = cache.get(&key) {
        return *idx;
    }

    let midpoint = (vertices[a as usize] + vertices[b as usize]).normalize_or_zero();
    let idx = vertices.len() as u32;
    vertices.push(midpoint);
    cache.insert(key, idx);
    idx
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
            0, 1, 2, 2, 3, 0, 4, 7, 6, 6, 5, 4, 8, 9, 10, 10, 11, 8, 12, 15, 14, 14, 13, 12, 16,
            17, 18, 18, 19, 16, 20, 23, 22, 22, 21, 20,
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

    pub fn cylinder(name: String, radial_segments: u32, height_segments: u32) -> Self {
        let radial_segments = radial_segments.max(3);
        let height_segments = height_segments.max(1);
        let radius = 0.5f32;
        let half_height = 0.5f32;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for y in 0..=height_segments {
            let v = y as f32 / height_segments as f32;
            let y_pos = half_height - (v * 2.0 * half_height);

            for s in 0..=radial_segments {
                let u = s as f32 / radial_segments as f32;
                let theta = u * std::f32::consts::TAU;
                let (sin_theta, cos_theta) = theta.sin_cos();
                let x = sin_theta * radius;
                let z = cos_theta * radius;

                let normal = Vec3::new(sin_theta, 0.0, cos_theta);
                let tangent = Vec3::new(cos_theta, 0.0, -sin_theta);

                vertices.push(Vertex::new(
                    [x, y_pos, z],
                    normal.into(),
                    [u, v],
                    tangent.extend(0.0).into(),
                ));
            }
        }

        let stride = radial_segments + 1;
        for y in 0..height_segments {
            for s in 0..radial_segments {
                let p0 = y * stride + s;
                let p1 = p0 + 1;
                let p2 = (y + 1) * stride + s;
                let p3 = p2 + 1;

                indices.extend_from_slice(&[p0, p2, p1, p2, p3, p1]);
            }
        }

        let top_center = vertices.len() as u32;
        vertices.push(Vertex::new(
            [0.0, half_height, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
        ));
        for s in 0..=radial_segments {
            let u = s as f32 / radial_segments as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            vertices.push(Vertex::new(
                [sin_theta * radius, half_height, cos_theta * radius],
                [0.0, 1.0, 0.0],
                [0.5 + (sin_theta * 0.5), 0.5 - (cos_theta * 0.5)],
                [1.0, 0.0, 0.0, 1.0],
            ));
        }
        for s in 0..radial_segments {
            let current = top_center + 1 + s;
            let next = current + 1;
            indices.extend_from_slice(&[top_center, current, next]);
        }

        let bottom_center = vertices.len() as u32;
        vertices.push(Vertex::new(
            [0.0, -half_height, 0.0],
            [0.0, -1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
        ));
        for s in 0..=radial_segments {
            let u = s as f32 / radial_segments as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            vertices.push(Vertex::new(
                [sin_theta * radius, -half_height, cos_theta * radius],
                [0.0, -1.0, 0.0],
                [0.5 + (sin_theta * 0.5), 0.5 + (cos_theta * 0.5)],
                [1.0, 0.0, 0.0, 1.0],
            ));
        }
        for s in 0..radial_segments {
            let current = bottom_center + 1 + s;
            let next = current + 1;
            indices.extend_from_slice(&[bottom_center, next, current]);
        }

        Self::new_raw(name, vertices, indices)
    }

    pub fn icosphere(name: String, subdivisions: u32) -> Self {
        let subdivisions = subdivisions.min(6);
        let t = (1.0 + 5.0f32.sqrt()) * 0.5;
        let mut positions = vec![
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ];
        for position in &mut positions {
            *position = position.normalize_or_zero();
        }

        let mut faces = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];

        for _ in 0..subdivisions {
            let mut midpoint_cache = HashMap::new();
            let mut next_faces = Vec::with_capacity(faces.len() * 4);

            for [a, b, c] in faces {
                let ab = icosphere_midpoint(a, b, &mut positions, &mut midpoint_cache);
                let bc = icosphere_midpoint(b, c, &mut positions, &mut midpoint_cache);
                let ca = icosphere_midpoint(c, a, &mut positions, &mut midpoint_cache);

                next_faces.push([a, ab, ca]);
                next_faces.push([b, bc, ab]);
                next_faces.push([c, ca, bc]);
                next_faces.push([ab, bc, ca]);
            }

            faces = next_faces;
        }

        let vertices = positions
            .iter()
            .map(|position| {
                let normal = position.normalize_or_zero();
                let u = 0.5 + (-normal.x).atan2(normal.z) / std::f32::consts::TAU;
                let v = 0.5 - normal.y.asin() / std::f32::consts::PI;
                let tangent = if normal.y.abs() > 0.999 {
                    Vec3::X
                } else {
                    Vec3::new(normal.z, 0.0, -normal.x).normalize_or_zero()
                };
                Vertex::new(
                    normal.into(),
                    normal.into(),
                    [u, v],
                    tangent.extend(0.0).into(),
                )
            })
            .collect::<Vec<_>>();

        let indices = faces
            .iter()
            .flat_map(|face| [face[0], face[1], face[2]])
            .collect::<Vec<_>>();

        Self::new_raw(name, vertices, indices)
    }

    pub fn capsule(name: String, segments: u32, rings: u32) -> Self {
        let segments = segments.max(3);
        let rings = rings.max(2);
        let radius = 0.5f32;
        let half_length = 0.5f32;

        let mut ring_data = Vec::with_capacity((rings * 2 + 2) as usize);
        for i in 0..=rings {
            let t = i as f32 / rings as f32;
            let phi = t * std::f32::consts::FRAC_PI_2;
            ring_data.push((half_length + (radius * phi.cos()), radius * phi.sin()));
        }
        for i in 0..=rings {
            let t = i as f32 / rings as f32;
            let phi = std::f32::consts::FRAC_PI_2 + (t * std::f32::consts::FRAC_PI_2);
            ring_data.push((-half_length + (radius * phi.cos()), radius * phi.sin()));
        }

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let ring_count = ring_data.len() as u32;
        let stride = segments + 1;

        for (ring_index, (y, ring_radius)) in ring_data.iter().enumerate() {
            let v = ring_index as f32 / (ring_data.len().saturating_sub(1)) as f32;
            for s in 0..=segments {
                let u = s as f32 / segments as f32;
                let theta = u * std::f32::consts::TAU;
                let (sin_theta, cos_theta) = theta.sin_cos();

                let position = Vec3::new(sin_theta * *ring_radius, *y, cos_theta * *ring_radius);
                let normal = if *y > half_length {
                    (position - Vec3::new(0.0, half_length, 0.0)).normalize_or_zero()
                } else if *y < -half_length {
                    (position - Vec3::new(0.0, -half_length, 0.0)).normalize_or_zero()
                } else {
                    Vec3::new(sin_theta, 0.0, cos_theta)
                };
                let tangent = Vec3::new(cos_theta, 0.0, -sin_theta);

                vertices.push(Vertex::new(
                    position.into(),
                    normal.into(),
                    [u, v],
                    tangent.extend(0.0).into(),
                ));
            }
        }

        for r in 0..(ring_count - 1) {
            for s in 0..segments {
                let p0 = r * stride + s;
                let p1 = p0 + 1;
                let p2 = (r + 1) * stride + s;
                let p3 = p2 + 1;

                indices.extend_from_slice(&[p0, p2, p1, p2, p3, p1]);
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
