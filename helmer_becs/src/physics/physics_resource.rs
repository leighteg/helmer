use bevy_ecs::{entity::Entity, resource::Resource};
use glam::Vec3;
use hashbrown::HashMap;
use rapier3d::prelude::*;

use crate::physics::components::{
    ColliderProperties, PhysicsHandle, PhysicsQueryFilter, PhysicsRayCastHit, PhysicsWorldDefaults,
    RigidBodyProperties,
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

    pub fn cast_ray(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        solid: bool,
        filter: PhysicsQueryFilter,
        exclude_entity: Option<Entity>,
    ) -> PhysicsRayCastHit {
        let mut hit = PhysicsRayCastHit::default();
        if !self.running {
            return hit;
        }

        let direction = normalize_axis(direction, Vec3::ZERO);
        if direction.length_squared() <= 1.0e-8 {
            return hit;
        }

        let exclude_collider =
            exclude_entity.and_then(|entity| self.physics_entities.get(&entity).copied());
        let query_filter = build_query_filter(filter, exclude_collider);
        let pipeline = self.query_pipeline_with_filter(query_filter);

        let ray = Ray::new(to_vector(origin), to_vector(direction));
        if let Some((collider_handle, ray_hit)) =
            pipeline.cast_ray_and_get_normal(&ray, max_toi.max(0.0), solid)
        {
            hit.has_hit = true;
            hit.hit_entity = self.entity_for_collider(collider_handle);
            hit.toi = ray_hit.time_of_impact;
            hit.point = from_vector(ray.origin + ray.dir * ray_hit.time_of_impact);
            hit.normal = from_vector(ray_hit.normal);
        }

        hit
    }
}

fn build_query_filter(
    filter: PhysicsQueryFilter,
    exclude_collider: Option<PhysicsHandle>,
) -> QueryFilter<'static> {
    QueryFilter {
        flags: QueryFilterFlags::from_bits_retain(filter.flags),
        groups: if filter.use_groups {
            Some(InteractionGroups::new(
                Group::from_bits_retain(filter.groups_memberships),
                Group::from_bits_retain(filter.groups_filter),
                InteractionTestMode::And,
            ))
        } else {
            None
        },
        exclude_collider: exclude_collider.map(|handle| handle.collider),
        exclude_rigid_body: exclude_collider.map(|handle| handle.rigid_body),
        predicate: None,
    }
}

#[inline]
fn normalize_axis(axis: Vec3, fallback: Vec3) -> Vec3 {
    if axis.length_squared() > 1.0e-8 {
        axis.normalize()
    } else {
        fallback
    }
}

#[inline]
fn to_vector(v: Vec3) -> Vector {
    Vector::new(v.x, v.y, v.z)
}

#[inline]
fn from_vector(v: Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

impl Default for PhysicsResource {
    fn default() -> Self {
        Self::new()
    }
}
