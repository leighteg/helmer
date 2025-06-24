use crate::ecs::component::Component;
use proc::Component;
use rapier3d::prelude::{RigidBodyHandle, ColliderHandle};

// A component to mark an entity as a dynamic rigid body
#[derive(Component, Debug, Clone, Copy)]
pub struct DynamicRigidBody;

// A component to mark an entity as a fixed collider (e.g., the ground)
#[derive(Component, Debug, Clone, Copy)]
pub struct FixedCollider;

// A component that will store the Rapier handle after the body is created.
// This is how we map an ECS entity to a Rapier body.
#[derive(Component, Debug, Clone, Copy)]
pub struct PhysicsHandle {
    pub rigid_body: RigidBodyHandle,
    pub collider: ColliderHandle,
}