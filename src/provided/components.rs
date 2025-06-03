use crate::ecs::component::Component;
use proc::Component;

#[derive(Debug, Clone, Component)]
pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 4], // quaternion
    pub scale: [f32; 3],
}

#[derive(Debug, Clone, Component)]
pub struct MeshComponent {
    pub mesh_id: u32,
    pub material_id: u32,
}

#[derive(Debug, Clone, Component)]
pub struct LightComponent {
    pub color: [f32; 3],
    pub intensity: f32,
    pub light_type: LightType,
}

#[derive(Debug, Clone, Component)]
pub enum LightType {
    Directional,
    Point,
    Spot { angle: f32 },
}