use bevy_ecs::{entity::Entity, resource::Resource};
use glam::Vec3;
use hashbrown::HashMap;
use rapier3d::prelude::*;

use crate::physics::components::PhysicsHandle;

// This struct encapsulates the entire physics state.
// It will be stored in the ECS as a single resource.
#[derive(Resource)]
pub struct PhysicsResource {
    pub pipeline: PhysicsPipeline,
    pub gravity: Vec3,
    pub integration_parameters: IntegrationParameters,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline, // Important for ray-casting
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub physics_entities: HashMap<Entity, PhysicsHandle>,
    pub running: bool,
}

impl PhysicsResource {
    // A constructor to set up the default physics world.
    pub fn new() -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            gravity: Vec3::new(0.0, -98.1, 0.0),
            integration_parameters: IntegrationParameters::default(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            physics_entities: HashMap::new(),
            running: true,
        }
    }
}

// Implement Default for convenience
impl Default for PhysicsResource {
    fn default() -> Self {
        Self::new()
    }
}
