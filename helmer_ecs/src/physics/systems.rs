use glam::{Quat, Vec3};
use helmer::provided::components::Transform;
use helmer::runtime::input_manager::InputManager;
use rapier3d::{
    math::Pose,
    prelude::{ColliderBuilder, RigidBodyBuilder},
};
use std::any::TypeId;
use tracing::warn;

use crate::{
    ecs::{
        ecs_core::{ECSCore, Entity},
        system::System,
    },
    physics::{
        // Import the new and updated component definitions
        components::{ColliderShape, DynamicRigidBody, FixedCollider, PhysicsHandle},
        physics_resource::PhysicsResource,
    },
};

//=====================================================================
// SyncEntitiesToPhysicsSystem: REWRITTEN
// This system now reads ColliderShape and DynamicRigidBody.mass
// to create physics bodies with specific properties.
//=====================================================================
pub struct SyncEntitiesToPhysicsSystem {}

impl System for SyncEntitiesToPhysicsSystem {
    fn name(&self) -> &str {
        "SyncEntitiesToPhysicsSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let mut new_handles_to_add: Vec<(Entity, PhysicsHandle)> = Vec::new();

        // --- Process Dynamic Bodies ---
        {
            // Pass 1: Collect data for dynamic bodies
            let mut entities_to_create: Vec<(Entity, Transform, DynamicRigidBody, ColliderShape)> =
                Vec::new();
            let entity_ids = ecs.component_pool.get_entities_with_all(&[
                TypeId::of::<Transform>(),
                TypeId::of::<DynamicRigidBody>(),
                TypeId::of::<ColliderShape>(),
            ]);

            for entity_id in entity_ids {
                if ecs.component_pool.get::<PhysicsHandle>(entity_id).is_some() {
                    continue;
                }
                // We know these components exist from the query, so we can unwrap them.
                let transform = *ecs.component_pool.get::<Transform>(entity_id).unwrap();
                let dynamic_body = *ecs
                    .component_pool
                    .get::<DynamicRigidBody>(entity_id)
                    .unwrap();
                let collider_shape = *ecs.component_pool.get::<ColliderShape>(entity_id).unwrap();
                entities_to_create.push((entity_id, transform, dynamic_body, collider_shape));
            }

            // Pass 2: Create dynamic bodies
            if let Some(physics_resource) = ecs.get_resource_mut::<PhysicsResource>() {
                for (entity_id, transform, dynamic_body, shape) in entities_to_create {
                    let iso = Pose::from_parts(transform.position, transform.rotation);

                    // Build the rigid body using the specified mass
                    let rigid_body = RigidBodyBuilder::dynamic()
                        .pose(iso)
                        .additional_mass(dynamic_body.mass)
                        .ccd_enabled(true)
                        .build();

                    // Build the collider based on the specified shape
                    let collider = match shape {
                        ColliderShape::Cuboid => ColliderBuilder::cuboid(
                            transform.scale.x * 0.5,
                            transform.scale.y * 0.5,
                            transform.scale.z * 0.5,
                        ),
                        ColliderShape::Sphere => ColliderBuilder::ball(transform.scale.x), // Assume uniform scale for sphere radius
                    }
                    .restitution(0.7)
                    .build();

                    let rigid_body_handle = physics_resource.rigid_body_set.insert(rigid_body);
                    let collider_handle = physics_resource.collider_set.insert_with_parent(
                        collider,
                        rigid_body_handle,
                        &mut physics_resource.rigid_body_set,
                    );

                    new_handles_to_add.push((
                        entity_id,
                        PhysicsHandle {
                            rigid_body: rigid_body_handle,
                            collider: collider_handle,
                        },
                    ));
                    physics_resource.physics_entities.insert(
                        entity_id,
                        new_handles_to_add[new_handles_to_add.len() - 1].1,
                    );
                }
            }
        }

        // --- Process Fixed Bodies ---
        {
            // Pass 1: Collect data for fixed bodies
            let mut entities_to_create: Vec<(Entity, Transform, ColliderShape)> = Vec::new();
            let entity_ids = ecs.component_pool.get_entities_with_all(&[
                TypeId::of::<Transform>(),
                TypeId::of::<FixedCollider>(),
                TypeId::of::<ColliderShape>(),
            ]);

            for entity_id in entity_ids {
                if ecs.component_pool.get::<PhysicsHandle>(entity_id).is_some() {
                    continue;
                }
                let transform = *ecs.component_pool.get::<Transform>(entity_id).unwrap();
                let collider_shape = *ecs.component_pool.get::<ColliderShape>(entity_id).unwrap();
                entities_to_create.push((entity_id, transform, collider_shape));
            }

            // Pass 2: Create fixed bodies
            if let Some(physics_resource) = ecs.get_resource_mut::<PhysicsResource>() {
                for (entity_id, transform, shape) in entities_to_create {
                    let iso = Pose::from_parts(transform.position, transform.rotation);

                    let rigid_body = RigidBodyBuilder::fixed().pose(iso).build();

                    let collider = match shape {
                        ColliderShape::Cuboid => ColliderBuilder::cuboid(
                            transform.scale.x * 0.5,
                            transform.scale.y * 0.5,
                            transform.scale.z * 0.5,
                        ),
                        ColliderShape::Sphere => ColliderBuilder::ball(transform.scale.x),
                    }
                    .build();

                    let rigid_body_handle = physics_resource.rigid_body_set.insert(rigid_body);
                    let collider_handle = physics_resource.collider_set.insert_with_parent(
                        collider,
                        rigid_body_handle,
                        &mut physics_resource.rigid_body_set,
                    );

                    new_handles_to_add.push((
                        entity_id,
                        PhysicsHandle {
                            rigid_body: rigid_body_handle,
                            collider: collider_handle,
                        },
                    ));
                    physics_resource.physics_entities.insert(
                        entity_id,
                        new_handles_to_add[new_handles_to_add.len() - 1].1,
                    );
                }
            }
        }

        // --- Pass 3: Add new handles to the ECS ---
        for (id, handle) in new_handles_to_add {
            ecs.component_pool.insert(id, handle);
        }
    }
}

//=====================================================================
// PhysicsStepSystem: Unchanged
//=====================================================================
pub struct PhysicsStepSystem {}

impl System for PhysicsStepSystem {
    fn name(&self) -> &str {
        "PhysicsStepSystem"
    }

    fn run(&mut self, dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let Some(phys) = ecs.get_resource_mut::<PhysicsResource>() else {
            return;
        };

        phys.integration_parameters.dt = dt;
        phys.integration_parameters.max_ccd_substeps = 4;
        phys.integration_parameters
            .num_internal_stabilization_iterations = 4; // Optional: helps with stability

        let gravity_vector = Vec3::new(phys.gravity.x, phys.gravity.y, phys.gravity.z);

        phys.pipeline.step(
            gravity_vector,
            &phys.integration_parameters,
            &mut phys.island_manager,
            &mut phys.broad_phase,
            &mut phys.narrow_phase,
            &mut phys.rigid_body_set,
            &mut phys.collider_set,
            &mut phys.impulse_joint_set,
            &mut phys.multibody_joint_set,
            &mut phys.ccd_solver,
            &(),
            &(),
        );
    }
}

//=====================================================================
// SyncPhysicsToEntitiesSystem: Unchanged
//=====================================================================
pub struct SyncPhysicsToEntitiesSystem {}

impl System for SyncPhysicsToEntitiesSystem {
    fn name(&self) -> &str {
        "SyncPhysicsToEntitiesSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let mut entities_with_handles: Vec<(Entity, PhysicsHandle)> = Vec::new();
        ecs.component_pool
            .query_for_each::<PhysicsHandle, _>(|entity_id, handle| {
                entities_with_handles.push((entity_id, *handle));
            });

        let mut updates: Vec<(Entity, Vec3, Quat)> = Vec::new();
        if let Some(physics_resource) = ecs.get_resource::<PhysicsResource>() {
            for (entity_id, handle) in &entities_with_handles {
                if let Some(rigid_body) = physics_resource.rigid_body_set.get(handle.rigid_body) {
                    if rigid_body.is_dynamic() {
                        let rb_pos = rigid_body.translation();
                        let rb_rot = rigid_body.rotation();

                        let new_pos = rb_pos;
                        let new_rot = *rb_rot;
                        updates.push((*entity_id, new_pos, new_rot));
                    }
                }
            }
        }

        for (entity_id, pos, rot) in updates {
            if let Some(transform) = ecs.component_pool.get_mut::<Transform>(entity_id) {
                transform.position = pos;
                transform.rotation = rot;
            }
        }
    }
}

//=====================================================================
// CleanupPhysicsSystem
// This system removes physics bodies from Rapier when their corresponding
// ECS entity has been destroyed.
//=====================================================================
pub struct CleanupPhysicsSystem {
    /// Maximum safe coordinate value before Rapier starts having issues
    /// Based on f32 precision and typical broad-phase limits
    pub max_coordinate: f32,
}

impl Default for CleanupPhysicsSystem {
    fn default() -> Self {
        Self {
            // rapier panics around -339 million
            max_coordinate: 100_000_000.0,
        }
    }
}

impl CleanupPhysicsSystem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_bounds(max_coordinate: f32) -> Self {
        Self { max_coordinate }
    }

    /// Check if a position is within safe Rapier bounds
    fn is_position_safe(&self, pos: &Vec3) -> bool {
        pos.x.abs() < self.max_coordinate
            && pos.y.abs() < self.max_coordinate
            && pos.z.abs() < self.max_coordinate
            && pos.x.is_finite()
            && pos.y.is_finite()
            && pos.z.is_finite()
    }
}

impl System for CleanupPhysicsSystem {
    fn name(&self) -> &str {
        "CleanupPhysicsSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let mut dead_entities: Vec<Entity> = Vec::new();
        let mut out_of_bounds_entities: Vec<Entity> = Vec::new();

        // Pass 1: Collect entities that need to be destroyed (separate scope to avoid borrow conflicts)
        {
            if let Some(physics_resource) = ecs.get_resource::<PhysicsResource>() {
                for (&entity_id, &handle) in physics_resource.physics_entities.iter() {
                    // Check if entity was destroyed
                    if !ecs.entity_exists(entity_id) {
                        dead_entities.push(entity_id);
                        continue;
                    }

                    // Check if physics body is out of bounds
                    if let Some(rigid_body) = physics_resource.rigid_body_set.get(handle.rigid_body)
                    {
                        let position = rigid_body.translation();

                        if !self.is_position_safe(&position) {
                            warn!(
                                "Entity {} is out of bounds at [{:.2}, {:.2}, {:.2}], destroying to prevent Rapier panic",
                                entity_id, position.x, position.y, position.z
                            );
                            out_of_bounds_entities.push(entity_id);
                        }
                    }
                }
            }
        } // End borrow scope

        // Pass 2: Destroy out-of-bounds ECS entities
        for entity_id in &out_of_bounds_entities {
            ecs.destroy_entity(*entity_id);
        }

        // Pass 3: Remove physics bodies and clean up
        let mut all_entities_to_cleanup = dead_entities;
        all_entities_to_cleanup.extend(out_of_bounds_entities);

        if !all_entities_to_cleanup.is_empty() {
            if let Some(physics_resource) = ecs.get_resource_mut::<PhysicsResource>() {
                for entity_id in all_entities_to_cleanup {
                    if let Some(handle) = physics_resource.physics_entities.remove(&entity_id) {
                        // Remove the rigid body and its associated colliders
                        physics_resource.rigid_body_set.remove(
                            handle.rigid_body,
                            &mut physics_resource.island_manager,
                            &mut physics_resource.collider_set,
                            &mut physics_resource.impulse_joint_set,
                            &mut physics_resource.multibody_joint_set,
                            true, // Remove associated colliders
                        );
                    }
                }
            }
        }
    }
}
