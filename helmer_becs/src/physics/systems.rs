use std::num::NonZero;

use bevy_ecs::{
    component::Component,
    prelude::{Entity, Local, With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use glam::{Quat, Vec3};
use hashbrown::{HashMap, HashSet};
use helmer::provided::components::Transform;
use rapier3d::{
    math::Isometry,
    na::{self as nalgebra, Translation3, UnitQuaternion, Vector3},
    prelude::{
        BroadPhaseMultiSap, CCDSolver, ColliderBuilder, ColliderSet, ImpulseJointSet,
        IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline,
        RigidBodyBuilder, RigidBodySet,
    },
};
use tracing::warn;

use crate::{
    BevyTransform, DeltaTime,
    physics::{
        components::{ColliderShape, DynamicRigidBody, FixedCollider, PhysicsHandle},
        physics_resource::PhysicsResource,
    },
};

//=====================================================================
// Cleanup State
//=====================================================================

pub struct CleanupState {
    max_coordinate: f32,
    // Reusable buffers to avoid allocations
    dead_entities: Vec<Entity>,
    out_of_bounds_entities: Vec<Entity>,
}

impl CleanupState {
    #[inline(always)]
    fn is_position_safe(&self, pos: &Vector3<f32>) -> bool {
        pos.x.abs() < self.max_coordinate
            && pos.y.abs() < self.max_coordinate
            && pos.z.abs() < self.max_coordinate
            && pos.x.is_finite()
            && pos.y.is_finite()
            && pos.z.is_finite()
    }
}

impl Default for CleanupState {
    fn default() -> Self {
        Self {
            max_coordinate: 100_000_000.0,
            dead_entities: Vec::with_capacity(64),
            out_of_bounds_entities: Vec::with_capacity(64),
        }
    }
}

//=====================================================================
// Helper Functions
//=====================================================================

#[inline]
fn build_isometry(position: Vec3, rotation: Quat) -> Isometry<f32> {
    let translation = Translation3::new(position.x, position.y, position.z);
    let rotation = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        rotation.w, rotation.x, rotation.y, rotation.z,
    ));
    Isometry::from_parts(translation, rotation)
}

#[inline]
fn build_collider(shape: &ColliderShape, scale: Vec3, is_dynamic: bool) -> ColliderBuilder {
    let mut builder = match shape {
        ColliderShape::Cuboid => {
            ColliderBuilder::cuboid(scale.x * 0.5, scale.y * 0.5, scale.z * 0.5)
        }
        ColliderShape::Sphere => ColliderBuilder::ball(scale.x),
    };

    if is_dynamic {
        builder = builder.restitution(0.7);
    }

    builder
}

//=====================================================================
// Systems
//=====================================================================

pub fn sync_entities_to_physics_system(
    mut commands: Commands,
    mut phys: ResMut<PhysicsResource>,
    dynamic_query: Query<
        (Entity, &BevyTransform, &DynamicRigidBody, &ColliderShape),
        Without<PhysicsHandle>,
    >,
    fixed_query: Query<
        (Entity, &BevyTransform, &ColliderShape),
        (With<FixedCollider>, Without<PhysicsHandle>),
    >,
) {
    // Pre-calculate sizes for batch allocation
    let dynamic_count = dynamic_query.iter().len();
    let fixed_count = fixed_query.iter().len();
    let total_count = dynamic_count + fixed_count;

    if total_count == 0 {
        return;
    }

    let phys_data = &mut *phys;
    let rigid_body_set = &mut phys_data.rigid_body_set;
    let collider_set = &mut phys_data.collider_set;
    let physics_entities = &mut phys_data.physics_entities;

    // Reserve capacity to avoid reallocations
    physics_entities.reserve(total_count);

    // Batch process dynamic bodies
    let mut batch_handles = Vec::with_capacity(total_count);

    for (entity, transform, dynamic_body, shape) in dynamic_query.iter() {
        let transform = transform.0;
        let iso = build_isometry(transform.position, transform.rotation);

        let rigid_body = RigidBodyBuilder::dynamic()
            .position(iso)
            .additional_mass(dynamic_body.mass)
            .ccd_enabled(true)
            .build();

        let collider = build_collider(shape, transform.scale, true).build();

        let rigid_body_handle = rigid_body_set.insert(rigid_body);
        let collider_handle =
            collider_set.insert_with_parent(collider, rigid_body_handle, rigid_body_set);

        let handle = PhysicsHandle {
            rigid_body: rigid_body_handle,
            collider: collider_handle,
        };

        batch_handles.push((entity, handle));
    }

    // Batch process fixed bodies
    for (entity, transform, shape) in fixed_query.iter() {
        let transform = transform.0;
        let iso = build_isometry(transform.position, transform.rotation);

        let rigid_body = RigidBodyBuilder::fixed().position(iso).build();
        let collider = build_collider(shape, transform.scale, false).build();

        let rigid_body_handle = rigid_body_set.insert(rigid_body);
        let collider_handle =
            collider_set.insert_with_parent(collider, rigid_body_handle, rigid_body_set);

        let handle = PhysicsHandle {
            rigid_body: rigid_body_handle,
            collider: collider_handle,
        };

        batch_handles.push((entity, handle));
    }

    // Batch insert all handles at once
    for (entity, handle) in batch_handles {
        commands.entity(entity).insert(handle);
        physics_entities.insert(entity, handle);
    }
}

#[inline]
pub fn physics_step_system(mut phys: ResMut<PhysicsResource>, time: Res<DeltaTime>) {
    if !phys.running {
        return;
    }
    
    let dt = time.0;
    let phys_data = &mut *phys;

    // Update integration parameters
    phys_data.integration_parameters.dt = dt;

    let gravity_vector = Vector3::new(
        phys_data.gravity.x,
        phys_data.gravity.y,
        phys_data.gravity.z,
    );

    // Run physics step
    phys_data.pipeline.step(
        &gravity_vector,
        &phys_data.integration_parameters,
        &mut phys_data.island_manager,
        &mut phys_data.broad_phase,
        &mut phys_data.narrow_phase,
        &mut phys_data.rigid_body_set,
        &mut phys_data.collider_set,
        &mut phys_data.impulse_joint_set,
        &mut phys_data.multibody_joint_set,
        &mut phys_data.ccd_solver,
        None,
        &(),
        &(),
    );
}

pub fn sync_physics_to_entities_system(
    phys: Res<PhysicsResource>,
    mut query: Query<(&PhysicsHandle, &mut BevyTransform)>,
) {
    let rigid_body_set = &phys.rigid_body_set;

    // Iterate and update transforms
    for (handle, mut transform) in query.iter_mut() {
        // SAFETY: We're accessing rigid_body_set immutably
        if let Some(rigid_body) = rigid_body_set.get(handle.rigid_body) {
            if rigid_body.is_dynamic() {
                let transform = &mut transform.0;
                let rb_pos = rigid_body.translation();
                let rb_rot = rigid_body.rotation();

                // Direct assignment is faster than creating intermediate structs
                transform.position.x = rb_pos.x;
                transform.position.y = rb_pos.y;
                transform.position.z = rb_pos.z;

                transform.rotation.x = rb_rot.i;
                transform.rotation.y = rb_rot.j;
                transform.rotation.z = rb_rot.k;
                transform.rotation.w = rb_rot.w;
            }
        }
    }
}

pub fn cleanup_physics_system(
    mut commands: Commands,
    mut phys: ResMut<PhysicsResource>,
    query: Query<Entity, With<PhysicsHandle>>,
    mut local_state: Local<CleanupState>,
) {
    // Clear reusable buffers
    local_state.dead_entities.clear();
    local_state.out_of_bounds_entities.clear();

    // Build live entities set efficiently
    let live_entities: HashSet<Entity> = query.iter().collect();

    // Find dead entities (removed from ECS but still in physics)
    for &entity in phys.physics_entities.keys() {
        if !live_entities.contains(&entity) {
            local_state.dead_entities.push(entity);
        }
    }

    // Find out-of-bounds entities
    for (&entity, &handle) in phys.physics_entities.iter() {
        if let Some(rigid_body) = phys.rigid_body_set.get(handle.rigid_body) {
            let position = rigid_body.translation();

            if !local_state.is_position_safe(position) {
                warn!(
                    "Entity {:?} is out of bounds at [{:.2}, {:.2}, {:.2}], destroying to prevent Rapier panic",
                    entity, position.x, position.y, position.z
                );
                local_state.out_of_bounds_entities.push(entity);
            }
        }
    }

    // Despawn out-of-bounds entities
    for &entity in &local_state.out_of_bounds_entities {
        commands.entity(entity).despawn();
    }

    // Combine all entities to cleanup
    // Move dead_entities into all_entities_to_cleanup, then append out_of_bounds
    let mut all_to_cleanup = std::mem::take(&mut local_state.dead_entities);
    all_to_cleanup.append(&mut local_state.out_of_bounds_entities);

    if all_to_cleanup.is_empty() {
        return;
    }

    // Remove from physics world
    let phys_data = &mut *phys;
    for &entity in &all_to_cleanup {
        if let Some(handle) = phys_data.physics_entities.remove(&entity) {
            phys_data.rigid_body_set.remove(
                handle.rigid_body,
                &mut phys_data.island_manager,
                &mut phys_data.collider_set,
                &mut phys_data.impulse_joint_set,
                &mut phys_data.multibody_joint_set,
                true,
            );
        }
    }
}

//=====================================================================
// Additional Broad-Phase Tuning Configuration
//=====================================================================

/// Configure broad-phase for optimal performance
/// Call this when initializing PhysicsResource
pub fn configure_broad_phase_optimal(broad_phase: &mut BroadPhaseMultiSap) {
    // Tune the broad phase grid for better spatial partitioning
    // Adjust these values based on your world size and entity density

    // For dense environments with many entities
    // broad_phase.set_grid_cell_size(10.0);

    // The default MultiSap is generally well-tuned, but you can experiment
    // with grid sizes based on your typical object sizes
}

/// Recommended integration parameters for performance
pub fn get_optimal_integration_parameters() -> IntegrationParameters {
    let mut params = IntegrationParameters::default();

    // Balance accuracy and performance
    params.max_ccd_substeps = 4; // Good balance for fast-moving objects
    params.num_internal_stabilization_iterations = 4; // Reasonable stability
    params.num_solver_iterations = NonZero::new(4).unwrap(); // Default is often good
    params.num_additional_friction_iterations = 2; // Reduce if not needed

    // Adjust allowed linear error for slight performance gain
    params.normalized_allowed_linear_error = 0.001; // Default is 0.001

    params
}
