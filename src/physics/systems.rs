use glam::{Quat, Vec3};
use rapier3d::{
    math::Isometry,
    prelude::{ColliderBuilder, RigidBodyBuilder},
};
use rapier3d::na::{self as nalgebra, Translation3};
use std::any::TypeId;

use crate::{
    ecs::{ecs_core::{ECSCore, Entity}, system::System},
    physics::{
        // Import the new and updated component definitions
        components::{ColliderShape, DynamicRigidBody, FixedCollider, PhysicsHandle},
        physics_resource::PhysicsResource,
    },
    provided::components::Transform,
    runtime::input_manager::InputManager,
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
            let mut entities_to_create: Vec<(Entity, Transform, DynamicRigidBody, ColliderShape)> = Vec::new();
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
                let dynamic_body = *ecs.component_pool.get::<DynamicRigidBody>(entity_id).unwrap();
                let collider_shape = *ecs.component_pool.get::<ColliderShape>(entity_id).unwrap();
                entities_to_create.push((entity_id, transform, dynamic_body, collider_shape));
            }

            // Pass 2: Create dynamic bodies
            if let Some(physics_resource) = ecs.get_resource_mut::<PhysicsResource>() {
                for (entity_id, transform, dynamic_body, shape) in entities_to_create {
                    let translation = Translation3::new(transform.position.x, transform.position.y, transform.position.z);
                    let rotation = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                        transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z,
                    ));
                    let iso = Isometry::from_parts(translation, rotation);

                    // Build the rigid body using the specified mass
                    let rigid_body = RigidBodyBuilder::dynamic().position(iso).additional_mass(dynamic_body.mass).build();

                    // Build the collider based on the specified shape
                    let collider = match shape {
                        ColliderShape::Cuboid => ColliderBuilder::cuboid(transform.scale.x * 0.5, transform.scale.y * 0.5, transform.scale.z * 0.5),
                        ColliderShape::Sphere => ColliderBuilder::ball(transform.scale.x * 0.5), // Assume uniform scale for sphere radius
                    }
                    .restitution(0.7)
                    .build();
                    
                    let rigid_body_handle = physics_resource.rigid_body_set.insert(rigid_body);
                    let collider_handle = physics_resource.collider_set.insert_with_parent(
                        collider, rigid_body_handle, &mut physics_resource.rigid_body_set,
                    );

                    new_handles_to_add.push((entity_id, PhysicsHandle { rigid_body: rigid_body_handle, collider: collider_handle }));
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
                    let translation = Translation3::new(transform.position.x, transform.position.y, transform.position.z);
                    let rotation = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                        transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z,
                    ));
                    let iso = Isometry::from_parts(translation, rotation);

                    let rigid_body = RigidBodyBuilder::fixed().position(iso).build();

                    let collider = match shape {
                        ColliderShape::Cuboid => ColliderBuilder::cuboid(transform.scale.x * 0.5, transform.scale.y * 0.5, transform.scale.z * 0.5),
                        ColliderShape::Sphere => ColliderBuilder::ball(transform.scale.x * 0.5),
                    }
                    .build();
                    
                    let rigid_body_handle = physics_resource.rigid_body_set.insert(rigid_body);
                    let collider_handle = physics_resource.collider_set.insert_with_parent(
                        collider, rigid_body_handle, &mut physics_resource.rigid_body_set,
                    );

                    new_handles_to_add.push((entity_id, PhysicsHandle { rigid_body: rigid_body_handle, collider: collider_handle }));
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
// SyncPhysicsToEntitiesSystem: Unchanged
//=====================================================================
pub struct SyncPhysicsToEntitiesSystem {}

impl System for SyncPhysicsToEntitiesSystem {
    fn name(&self) -> &str {
        "SyncPhysicsToEntitiesSystem"
    }

    fn run(&mut self, _dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let mut entities_with_handles: Vec<(Entity, PhysicsHandle)> = Vec::new();
        ecs.component_pool.query_for_each::<PhysicsHandle, _>(|entity_id, handle| {
            entities_with_handles.push((entity_id, *handle));
        });

        let mut updates: Vec<(Entity, Vec3, Quat)> = Vec::new();
        if let Some(physics_resource) = ecs.get_resource::<PhysicsResource>() {
            for (entity_id, handle) in &entities_with_handles {
                if let Some(rigid_body) = physics_resource.rigid_body_set.get(handle.rigid_body) {
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
        
        for (entity_id, pos, rot) in updates {
            if let Some(transform) = ecs.component_pool.get_mut::<Transform>(entity_id) {
                transform.position = pos;
                transform.rotation = rot;
            }
        }
    }
}
