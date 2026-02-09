use bevy_ecs::{entity::Entity, resource::Resource};
use glam::Vec3;
use hashbrown::HashMap;
use rapier3d::prelude::*;

use crate::physics::components::{
    ColliderProperties, PhysicsHandle, PhysicsWorldDefaults, RigidBodyProperties,
};

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
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub physics_entities: HashMap<Entity, PhysicsHandle>,
    pub collider_entities: HashMap<ColliderHandle, Entity>,
    pub physics_joints: HashMap<Entity, ImpulseJointHandle>,
    pub default_collider_properties: ColliderProperties,
    pub default_rigid_body_properties: RigidBodyProperties,
    pub running: bool,
}

impl PhysicsResource {
    pub fn new() -> Self {
        let defaults = PhysicsWorldDefaults::default();
        Self {
            pipeline: PhysicsPipeline::new(),
            gravity: defaults.gravity,
            integration_parameters: IntegrationParameters::default(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            physics_entities: HashMap::new(),
            collider_entities: HashMap::new(),
            physics_joints: HashMap::new(),
            default_collider_properties: defaults.collider_properties,
            default_rigid_body_properties: defaults.rigid_body_properties,
            running: true,
        }
    }

    #[inline]
    pub fn query_pipeline(&self) -> QueryPipeline<'_> {
        self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            QueryFilter::default(),
        )
    }

    #[inline]
    pub fn query_pipeline_with_filter<'a>(&'a self, filter: QueryFilter<'a>) -> QueryPipeline<'a> {
        self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            filter,
        )
    }

    #[inline]
    pub fn entity_for_collider(&self, handle: ColliderHandle) -> Option<Entity> {
        self.collider_entities.get(&handle).copied()
    }
}

impl Default for PhysicsResource {
    fn default() -> Self {
        Self::new()
    }
}
