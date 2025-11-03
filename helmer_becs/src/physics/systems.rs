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
}

impl CleanupState {
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
        }
    }
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
    let phys_data = &mut *phys;
    let rigid_body_set = &mut phys_data.rigid_body_set;
    let collider_set = &mut phys_data.collider_set;
    let physics_entities = &mut phys_data.physics_entities;

    // --- Process Dynamic Bodies ---
    for (entity, transform, dynamic_body, shape) in dynamic_query.iter() {
        let transform = transform.0;

        let translation = Translation3::new(
            transform.position.x,
            transform.position.y,
            transform.position.z,
        );
        let rotation = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            transform.rotation.w,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
        ));
        let iso = Isometry::from_parts(translation, rotation);

        let rigid_body = RigidBodyBuilder::dynamic()
            .position(iso)
            .additional_mass(dynamic_body.mass)
            .ccd_enabled(true)
            .build();

        let collider = match shape {
            ColliderShape::Cuboid => ColliderBuilder::cuboid(
                transform.scale.x * 0.5,
                transform.scale.y * 0.5,
                transform.scale.z * 0.5,
            ),
            ColliderShape::Sphere => ColliderBuilder::ball(transform.scale.x),
        }
        .restitution(0.7)
        .build();

        let rigid_body_handle = rigid_body_set.insert(rigid_body);
        let collider_handle =
            collider_set.insert_with_parent(collider, rigid_body_handle, rigid_body_set);

        let handle = PhysicsHandle {
            rigid_body: rigid_body_handle,
            collider: collider_handle,
        };
        commands.entity(entity).insert(handle);
        physics_entities.insert(entity, handle);
    }

    // --- Process Fixed Bodies ---
    for (entity, transform, shape) in fixed_query.iter() {
        let transform = transform.0;

        let translation = Translation3::new(
            transform.position.x,
            transform.position.y,
            transform.position.z,
        );
        let rotation = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            transform.rotation.w,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
        ));
        let iso = Isometry::from_parts(translation, rotation);

        let rigid_body = RigidBodyBuilder::fixed().position(iso).build();

        let collider = match shape {
            ColliderShape::Cuboid => ColliderBuilder::cuboid(
                transform.scale.x * 0.5,
                transform.scale.y * 0.5,
                transform.scale.z * 0.5,
            ),
            ColliderShape::Sphere => ColliderBuilder::ball(transform.scale.x),
        }
        .build();

        let rigid_body_handle = rigid_body_set.insert(rigid_body);
        let collider_handle =
            collider_set.insert_with_parent(collider, rigid_body_handle, rigid_body_set);

        let handle = PhysicsHandle {
            rigid_body: rigid_body_handle,
            collider: collider_handle,
        };
        commands.entity(entity).insert(handle);
        physics_entities.insert(entity, handle);
    }
}

pub fn physics_step_system(mut phys: ResMut<PhysicsResource>, time: Res<DeltaTime>) {
    let dt = time.0;

    let phys_data = &mut *phys;

    phys_data.integration_parameters.dt = dt;
    phys_data.integration_parameters.max_ccd_substeps = 4;
    phys_data
        .integration_parameters
        .num_internal_stabilization_iterations = 4;

    let gravity_vector = Vector3::new(
        phys_data.gravity.x,
        phys_data.gravity.y,
        phys_data.gravity.z,
    );

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
    for (handle, mut transform) in query.iter_mut() {
        let transform = &mut transform.0;

        if let Some(rigid_body) = phys.rigid_body_set.get(handle.rigid_body) {
            if rigid_body.is_dynamic() {
                let rb_pos = rigid_body.translation();
                transform.position = Vec3::new(rb_pos.x, rb_pos.y, rb_pos.z);
                let rb_rot = rigid_body.rotation();
                transform.rotation = Quat::from_xyzw(rb_rot.i, rb_rot.j, rb_rot.k, rb_rot.w);
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
    if local_state.max_coordinate == 0.0 {
        *local_state = CleanupState::default();
    }

    let mut dead_entities: Vec<Entity> = Vec::new();
    let mut out_of_bounds_entities: Vec<Entity> = Vec::new();

    let live_entities: HashSet<Entity> = query.iter().collect();

    for &entity in phys.physics_entities.keys() {
        if !live_entities.contains(&entity) {
            dead_entities.push(entity);
        }
    }

    for (&entity, &handle) in phys.physics_entities.iter() {
        if let Some(rigid_body) = phys.rigid_body_set.get(handle.rigid_body) {
            let position = rigid_body.translation();

            if !local_state.is_position_safe(position) {
                warn!(
                    "Entity {:?} is out of bounds at [{:.2}, {:.2}, {:.2}], destroying to prevent Rapier panic",
                    entity, position.x, position.y, position.z
                );
                out_of_bounds_entities.push(entity);
            }
        }
    }

    for &entity in &out_of_bounds_entities {
        commands.entity(entity).despawn();
    }

    let mut all_entities_to_cleanup = dead_entities;
    all_entities_to_cleanup.extend(out_of_bounds_entities);

    let phys_data = &mut *phys;
    let physics_entities = &mut phys_data.physics_entities;
    let rigid_body_set = &mut phys_data.rigid_body_set;
    let island_manager = &mut phys_data.island_manager;
    let collider_set = &mut phys_data.collider_set;
    let impulse_joint_set = &mut phys_data.impulse_joint_set;
    let multibody_joint_set = &mut phys_data.multibody_joint_set;

    for entity in all_entities_to_cleanup {
        let handle = physics_entities.remove(&entity);

        if let Some(handle) = handle {
            rigid_body_set.remove(
                handle.rigid_body,
                island_manager,
                collider_set,
                impulse_joint_set,
                multibody_joint_set,
                true,
            );
        }
    }
}
