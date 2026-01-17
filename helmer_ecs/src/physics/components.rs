use crate::ecs::component::Component;
use proc::Component;
use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};

// A component that will store the Rapier handle after the body is created.
// This is how we map an ECS entity to a Rapier body.
#[derive(Component, Debug, Clone, Copy)]
pub struct PhysicsHandle {
    pub rigid_body: RigidBodyHandle,
    pub collider: ColliderHandle,
}

/// Defines the physical shape of an entity's collider. This allows different
/// entities to have different physical representations (e.g., a box vs. a sphere).
#[derive(Component, Debug, Clone, Copy)]
pub enum ColliderShape {
    Cuboid,
    Sphere,
    // Future shapes like Capsule, Cylinder, etc., can be added here.
}

// A component to mark an entity as a dynamic rigid body
#[derive(Component, Debug, Clone, Copy)]
pub struct DynamicRigidBody {
    pub mass: f32,
}

// A component to mark an entity as a fixed collider (e.g., the ground)
#[derive(Component, Debug, Clone, Copy)]
pub struct FixedCollider;
