use glam::{Quat, Vec3};
use rapier3d::{
    math::Isometry,
    prelude::{ColliderBuilder, RigidBodyBuilder},
};
// Use `na` as an alias for nalgebra and explicitly import required types
use rapier3d::na::{self as nalgebra, Translation3};

use crate::{
    ecs::{ecs_core::{ECSCore, Entity}, system::System},
    physics::{
        // Import the new FixedCollider component
        components::{DynamicRigidBody, FixedCollider, PhysicsHandle},
        physics_resource::PhysicsResource,
    },
    provided::components::Transform,
    runtime::input_manager::InputManager,
};

//=====================================================================
// SyncEntitiesToPhysicsSystem: Updated to handle FixedColliders
//=====================================================================
pub struct SyncEntitiesToPhysicsSystem {}

impl System for SyncEntitiesToPhysicsSystem {
    fn name(&self) -> &str {
        "SyncEntitiesToPhysicsSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        // This system creates Rapier3D rigid bodies and colliders for entities
        // that have physics components (DynamicRigidBody, FixedCollider) but
        // do not yet have a PhysicsHandle.

        // --- Pass 1: Collect data with only immutable borrows. ---
        // We gather entities that need a physics body, separating dynamic and fixed ones.
        let mut dynamic_entities_to_create: Vec<(Entity, Transform)> = Vec::new();
        let mut fixed_entities_to_create: Vec<(Entity, Transform)> = Vec::new();

        // Collect entities needing a DYNAMIC rigid body
        let entities_with_dynamic = ecs.component_pool.get_entities_with_all(&[
            std::any::TypeId::of::<Transform>(),
            std::any::TypeId::of::<DynamicRigidBody>(),
        ]);

        for entity_id in entities_with_dynamic {
            // If the entity already has a physics handle, we can skip it.
            if ecs.component_pool.get::<PhysicsHandle>(entity_id).is_some() {
                continue;
            }
            // We know the transform exists from our query, so we can safely unwrap.
            if let Some(transform) = ecs.component_pool.get::<Transform>(entity_id) {
                dynamic_entities_to_create.push((entity_id, *transform));
            }
        }

        // Collect entities needing a FIXED rigid body (static collider)
        let entities_with_fixed = ecs.component_pool.get_entities_with_all(&[
            std::any::TypeId::of::<Transform>(),
            std::any::TypeId::of::<FixedCollider>(),
        ]);

        for entity_id in entities_with_fixed {
            // If the entity already has a physics handle, we can skip it.
            if ecs.component_pool.get::<PhysicsHandle>(entity_id).is_some() {
                continue;
            }
            if let Some(transform) = ecs.component_pool.get::<Transform>(entity_id) {
                fixed_entities_to_create.push((entity_id, *transform));
            }
        }

        // --- Pass 2: Mutably borrow the resource and create the bodies. ---
        // The immutable borrows from Pass 1 are now out of scope. We can now safely
        // borrow the physics resource mutably.
        let mut new_handles_to_add: Vec<(Entity, PhysicsHandle)> = Vec::new();
        if let Some(physics_resource) = ecs.get_resource_mut::<PhysicsResource>() {
            // Process DYNAMIC bodies
            for (entity_id, transform) in dynamic_entities_to_create {
                let translation = Translation3::new(transform.position.x, transform.position.y, transform.position.z);
                let rotation = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                    transform.rotation.w,
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                ));
                let iso = Isometry::from_parts(translation, rotation);

                let rigid_body = RigidBodyBuilder::dynamic().position(iso).build();
                let collider = ColliderBuilder::cuboid(transform.scale.x * 0.5, transform.scale.y * 0.5, transform.scale.z * 0.5)
                    .restitution(0.7)
                    .build();

                let rigid_body_handle = physics_resource.rigid_body_set.insert(rigid_body);
                let collider_handle = physics_resource.collider_set.insert_with_parent(
                    collider,
                    rigid_body_handle,
                    &mut physics_resource.rigid_body_set,
                );

                let handle = PhysicsHandle { rigid_body: rigid_body_handle, collider: collider_handle };
                new_handles_to_add.push((entity_id, handle));
            }

            // Process FIXED bodies
            for (entity_id, transform) in fixed_entities_to_create {
                let translation = Translation3::new(transform.position.x, transform.position.y, transform.position.z);
                let rotation = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                    transform.rotation.w,
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                ));
                let iso = Isometry::from_parts(translation, rotation);

                // Create a `fixed` rigid body for static geometry.
                let rigid_body = RigidBodyBuilder::fixed().position(iso).build();
                let collider = ColliderBuilder::cuboid(transform.scale.x * 0.5, transform.scale.y * 0.5, transform.scale.z * 0.5).build();

                let rigid_body_handle = physics_resource.rigid_body_set.insert(rigid_body);
                // Note: Fixed bodies don't need a parent, but Rapier's API supports it.
                // We can insert the collider and attach it to the body handle.
                let collider_handle = physics_resource.collider_set.insert_with_parent(
                    collider,
                    rigid_body_handle,
                    &mut physics_resource.rigid_body_set,
                );
                
                let handle = PhysicsHandle { rigid_body: rigid_body_handle, collider: collider_handle };
                new_handles_to_add.push((entity_id, handle));
            }
        }

        // --- Pass 3: Add the new components. ---
        // Now that the mutable borrow of the resource is out of scope, we can borrow the
        // component pool again to insert the new handles.
        for (id, handle) in new_handles_to_add {
            ecs.component_pool.insert(id, handle);
        }
    }
}

//=====================================================================
// PhysicsStepSystem: No Changes Needed
//=====================================================================
pub struct PhysicsStepSystem {}

impl System for PhysicsStepSystem {
    fn name(&self) -> &str {
        "PhysicsStepSystem"
    }

    fn run(&mut self, dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let Some(phys) = ecs.get_resource_mut::<PhysicsResource>() else { return; };
        phys.integration_parameters.dt = dt;

        let gravity_vector = nalgebra::Vector3::new(phys.gravity.x, phys.gravity.y, phys.gravity.z);

        phys.pipeline.step(
            &gravity_vector,
            &phys.integration_parameters,
            &mut phys.island_manager,
            &mut phys.broad_phase,
            &mut phys.narrow_phase,
            &mut phys.rigid_body_set,
            &mut phys.collider_set,
            &mut phys.impulse_joint_set,
            &mut phys.multibody_joint_set,
            &mut phys.ccd_solver,
            None,
            &(),
            &(),
        );
    }
}

//=====================================================================
// SyncPhysicsToEntitiesSystem: No Changes Needed
//=====================================================================
pub struct SyncPhysicsToEntitiesSystem {}

impl System for SyncPhysicsToEntitiesSystem {
    fn name(&self) -> &str {
        "SyncPhysicsToEntitiesSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        // --- Pass 1: Collect Entity IDs and Handles ---
        let mut entities_with_handles: Vec<(Entity, PhysicsHandle)> = Vec::new();
        ecs.component_pool.query_for_each::<PhysicsHandle, _>(|entity_id, handle| {
            entities_with_handles.push((entity_id, *handle));
        });

        // --- Pass 2: Collect Transform updates ---
        let mut updates: Vec<(Entity, Vec3, Quat)> = Vec::new();
        if let Some(physics_resource) = ecs.get_resource::<PhysicsResource>() {
            for (entity_id, handle) in &entities_with_handles {
                if let Some(rigid_body) = physics_resource.rigid_body_set.get(handle.rigid_body) {
                    // This check is important: only sync dynamic bodies back to the ECS.
                    // Fixed bodies do not move, so there's no need to update their Transform.
                    if rigid_body.is_dynamic() {
                        let rb_pos = rigid_body.translation();
                        let rb_rot = rigid_body.rotation();

                        let new_pos = Vec3::new(rb_pos.x, rb_pos.y, rb_pos.z);
                        let new_rot = Quat::from_xyzw(rb_rot.i, rb_rot.j, rb_rot.k, rb_rot.w);
                        updates.push((*entity_id, new_pos, new_rot));
                    }
                }
            }
        }

        // --- Pass 3: Apply updates with a mutable borrow ---
        for (entity_id, pos, rot) in updates {
            if let Some(transform) = ecs.component_pool.get_mut::<Transform>(entity_id) {
                transform.position = pos;
                transform.rotation = rot;
            }
        }
    }
}
