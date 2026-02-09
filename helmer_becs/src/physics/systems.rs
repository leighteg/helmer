use bevy_ecs::{
    prelude::{Entity, Local, Or, RemovedComponents, With, Without},
    query::Changed,
    system::{Commands, Query, Res, ResMut},
};
use glam::{Quat, Vec3};
use hashbrown::{HashMap, HashSet};
use helmer::provided::components::Transform;
use rapier3d::{
    control::{CharacterAutostep, CharacterLength, KinematicCharacterController},
    math::{Pose, Vector},
    parry::query::{ShapeCastStatus, details::ShapeCastOptions},
    prelude::{
        CoefficientCombineRule, ColliderBuilder, DefaultBroadPhase, FixedJointBuilder, Group,
        IntegrationParameters, InteractionGroups, InteractionTestMode, JointAxis,
        PrismaticJointBuilder, QueryFilter, QueryFilterFlags, Ray, RevoluteJointBuilder,
        RigidBodyBuilder, RigidBodyType, SharedShape, SphericalJointBuilder,
    },
};
use tracing::warn;

use crate::{
    BevyAssetServer, BevyMeshRenderer, BevySkinnedMeshRenderer, BevyTransform, DeltaTime,
    physics::{
        components::{
            CharacterController, CharacterControllerInput, CharacterControllerOutput,
            ColliderProperties, ColliderPropertyInheritance, ColliderShape, DynamicRigidBody,
            FixedCollider, KinematicMode, KinematicRigidBody, MeshColliderKind, MeshColliderLod,
            PhysicsCombineRule, PhysicsHandle, PhysicsJoint, PhysicsJointKind,
            PhysicsPointProjection, PhysicsPointProjectionHit, PhysicsQueryFilter, PhysicsRayCast,
            PhysicsRayCastHit, PhysicsShapeCast, PhysicsShapeCastHit, PhysicsShapeCastStatus,
            PhysicsWorldDefaults, RigidBodyProperties, RigidBodyPropertyInheritance,
        },
        physics_resource::PhysicsResource,
    },
};

pub struct CleanupState {
    max_coordinate: f32,
    dead_entities: Vec<Entity>,
    out_of_bounds_entities: Vec<Entity>,
}

impl CleanupState {
    #[inline(always)]
    fn is_position_safe(&self, pos: &Vec3) -> bool {
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

#[derive(Clone, Copy)]
enum BodyKind {
    Dynamic { mass: f32 },
    Kinematic { mode: KinematicMode },
    Fixed,
}

#[inline]
fn build_isometry(position: Vec3, rotation: Quat) -> Pose {
    Pose::from_parts(position, rotation)
}

#[inline]
fn sanitize_scale(scale: Vec3) -> Vec3 {
    Vec3::new(
        scale.x.abs().max(0.0001),
        scale.y.abs().max(0.0001),
        scale.z.abs().max(0.0001),
    )
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

#[inline]
fn combine_rule(rule: PhysicsCombineRule) -> CoefficientCombineRule {
    match rule {
        PhysicsCombineRule::Average => CoefficientCombineRule::Average,
        PhysicsCombineRule::Min => CoefficientCombineRule::Min,
        PhysicsCombineRule::Multiply => CoefficientCombineRule::Multiply,
        PhysicsCombineRule::Max => CoefficientCombineRule::Max,
    }
}

fn select_world_defaults_component(
    query: &Query<(Entity, &PhysicsWorldDefaults)>,
) -> Option<PhysicsWorldDefaults> {
    let mut selected: Option<(u64, PhysicsWorldDefaults)> = None;
    for (entity, defaults) in query.iter() {
        let key = entity.to_bits();
        let should_replace = selected
            .as_ref()
            .map(|(selected_key, _)| key < *selected_key)
            .unwrap_or(true);
        if should_replace {
            selected = Some((key, *defaults));
        }
    }
    selected.map(|(_, defaults)| defaults)
}

fn effective_rigid_body_properties(
    entity_props: Option<RigidBodyProperties>,
    inheritance: Option<RigidBodyPropertyInheritance>,
    world_defaults: RigidBodyProperties,
) -> RigidBodyProperties {
    let mut props = entity_props.unwrap_or(world_defaults);
    let inheritance = inheritance.unwrap_or_else(RigidBodyPropertyInheritance::none);

    if inheritance.linear_damping {
        props.linear_damping = world_defaults.linear_damping;
    }
    if inheritance.angular_damping {
        props.angular_damping = world_defaults.angular_damping;
    }
    if inheritance.gravity_scale {
        props.gravity_scale = world_defaults.gravity_scale;
    }
    if inheritance.ccd_enabled {
        props.ccd_enabled = world_defaults.ccd_enabled;
    }
    if inheritance.can_sleep {
        props.can_sleep = world_defaults.can_sleep;
    }
    if inheritance.sleeping {
        props.sleeping = world_defaults.sleeping;
    }
    if inheritance.dominance_group {
        props.dominance_group = world_defaults.dominance_group;
    }
    if inheritance.lock_translation_x {
        props.lock_translation_x = world_defaults.lock_translation_x;
    }
    if inheritance.lock_translation_y {
        props.lock_translation_y = world_defaults.lock_translation_y;
    }
    if inheritance.lock_translation_z {
        props.lock_translation_z = world_defaults.lock_translation_z;
    }
    if inheritance.lock_rotation_x {
        props.lock_rotation_x = world_defaults.lock_rotation_x;
    }
    if inheritance.lock_rotation_y {
        props.lock_rotation_y = world_defaults.lock_rotation_y;
    }
    if inheritance.lock_rotation_z {
        props.lock_rotation_z = world_defaults.lock_rotation_z;
    }
    if inheritance.linear_velocity {
        props.linear_velocity = world_defaults.linear_velocity;
    }
    if inheritance.angular_velocity {
        props.angular_velocity = world_defaults.angular_velocity;
    }

    props
}

fn effective_collider_properties(
    entity_props: Option<ColliderProperties>,
    inheritance: Option<ColliderPropertyInheritance>,
    world_defaults: ColliderProperties,
) -> ColliderProperties {
    let mut props = entity_props.unwrap_or(world_defaults);
    let inheritance = inheritance.unwrap_or_else(ColliderPropertyInheritance::none);

    if inheritance.friction {
        props.friction = world_defaults.friction;
    }
    if inheritance.restitution {
        props.restitution = world_defaults.restitution;
    }
    if inheritance.density {
        props.density = world_defaults.density;
    }
    if inheritance.is_sensor {
        props.is_sensor = world_defaults.is_sensor;
    }
    if inheritance.enabled {
        props.enabled = world_defaults.enabled;
    }
    if inheritance.collision_memberships {
        props.collision_memberships = world_defaults.collision_memberships;
    }
    if inheritance.collision_filter {
        props.collision_filter = world_defaults.collision_filter;
    }
    if inheritance.solver_memberships {
        props.solver_memberships = world_defaults.solver_memberships;
    }
    if inheritance.solver_filter {
        props.solver_filter = world_defaults.solver_filter;
    }
    if inheritance.friction_combine_rule {
        props.friction_combine_rule = world_defaults.friction_combine_rule;
    }
    if inheritance.restitution_combine_rule {
        props.restitution_combine_rule = world_defaults.restitution_combine_rule;
    }
    if inheritance.translation_offset {
        props.translation_offset = world_defaults.translation_offset;
    }
    if inheritance.rotation_offset {
        props.rotation_offset = world_defaults.rotation_offset;
    }

    props
}

#[inline]
fn rigid_body_uses_world_defaults(
    entity_props: Option<RigidBodyProperties>,
    inheritance: Option<RigidBodyPropertyInheritance>,
) -> bool {
    entity_props.is_none()
        || inheritance
            .map(RigidBodyPropertyInheritance::any_inherited)
            .unwrap_or(false)
}

#[inline]
fn collider_uses_world_defaults(
    entity_props: Option<ColliderProperties>,
    inheritance: Option<ColliderPropertyInheritance>,
) -> bool {
    entity_props.is_none()
        || inheritance
            .map(ColliderPropertyInheritance::any_inherited)
            .unwrap_or(false)
}

#[inline]
fn resolve_body_kind(
    dynamic: Option<DynamicRigidBody>,
    kinematic: Option<KinematicRigidBody>,
    fixed: bool,
) -> Option<BodyKind> {
    if let Some(dynamic) = dynamic {
        Some(BodyKind::Dynamic {
            mass: dynamic.mass.max(0.0),
        })
    } else if let Some(kinematic) = kinematic {
        Some(BodyKind::Kinematic {
            mode: kinematic.mode,
        })
    } else if fixed {
        Some(BodyKind::Fixed)
    } else {
        None
    }
}

fn select_lod_index(lod: MeshColliderLod, lod_count: usize) -> usize {
    if lod_count == 0 {
        return 0;
    }

    let last = lod_count - 1;
    match lod {
        MeshColliderLod::Lod0 => 0,
        MeshColliderLod::Lod1 => 1.min(last),
        MeshColliderLod::Lod2 => 2.min(last),
        MeshColliderLod::Lowest => last,
        MeshColliderLod::Specific(index) => (index as usize).min(last),
    }
}

fn resolve_mesh_id(
    explicit_id: Option<usize>,
    mesh_renderer: Option<&BevyMeshRenderer>,
    skinned_renderer: Option<&BevySkinnedMeshRenderer>,
) -> Option<usize> {
    explicit_id
        .or_else(|| mesh_renderer.map(|renderer| renderer.0.mesh_id))
        .or_else(|| skinned_renderer.map(|renderer| renderer.0.mesh_id))
}

fn mesh_geometry_for_collider(
    mesh_id: usize,
    lod: MeshColliderLod,
    scale: Vec3,
    asset_server: Option<&BevyAssetServer>,
) -> Option<(Vec<Vector>, Vec<[u32; 3]>)> {
    let asset_server = asset_server?;
    let mesh = asset_server.0.lock().get_mesh(mesh_id)?;

    let lods = mesh.lods.read();
    if lods.is_empty() {
        return None;
    }

    let lod_index = select_lod_index(lod, lods.len());
    let lod = lods.get(lod_index)?;

    if lod.vertices.is_empty() || lod.indices.len() < 3 {
        return None;
    }

    let mut vertices = Vec::with_capacity(lod.vertices.len());
    for vertex in lod.vertices.iter() {
        let p = vertex.position;
        vertices.push(Vector::new(p[0] * scale.x, p[1] * scale.y, p[2] * scale.z));
    }

    let mut indices = Vec::with_capacity(lod.indices.len() / 3);
    for tri in lod.indices.chunks_exact(3) {
        indices.push([tri[0], tri[1], tri[2]]);
    }

    if indices.is_empty() {
        return None;
    }

    Some((vertices, indices))
}

fn build_shared_shape(
    shape: ColliderShape,
    scale: Vec3,
    mesh_renderer: Option<&BevyMeshRenderer>,
    skinned_renderer: Option<&BevySkinnedMeshRenderer>,
    asset_server: Option<&BevyAssetServer>,
) -> Option<SharedShape> {
    let scale = sanitize_scale(scale);

    match shape {
        ColliderShape::Cuboid => Some(SharedShape::cuboid(
            scale.x * 0.5,
            scale.y * 0.5,
            scale.z * 0.5,
        )),
        ColliderShape::Sphere => {
            let radius = scale.max_element() * 0.5;
            Some(SharedShape::ball(radius))
        }
        ColliderShape::CapsuleY => {
            let radius = (scale.x.min(scale.z) * 0.5).max(0.0001);
            let half_height = (scale.y * 0.5 - radius).max(0.0001);
            Some(SharedShape::capsule_y(half_height, radius))
        }
        ColliderShape::CylinderY => {
            let radius = (scale.x.min(scale.z) * 0.5).max(0.0001);
            let half_height = (scale.y * 0.5).max(0.0001);
            Some(SharedShape::cylinder(half_height, radius))
        }
        ColliderShape::ConeY => {
            let radius = (scale.x.min(scale.z) * 0.5).max(0.0001);
            let half_height = (scale.y * 0.5).max(0.0001);
            Some(SharedShape::cone(half_height, radius))
        }
        ColliderShape::RoundCuboid { border_radius } => {
            let hx = scale.x * 0.5;
            let hy = scale.y * 0.5;
            let hz = scale.z * 0.5;
            let max_border = hx.min(hy).min(hz).max(0.0);
            Some(SharedShape::round_cuboid(
                hx,
                hy,
                hz,
                border_radius.clamp(0.0, max_border),
            ))
        }
        ColliderShape::Mesh { mesh_id, lod, kind } => {
            let Some(mesh_id) = resolve_mesh_id(mesh_id, mesh_renderer, skinned_renderer) else {
                return None;
            };
            let Some((vertices, indices)) =
                mesh_geometry_for_collider(mesh_id, lod, scale, asset_server)
            else {
                return None;
            };

            match kind {
                MeshColliderKind::TriMesh => SharedShape::trimesh(vertices, indices).ok(),
                MeshColliderKind::ConvexHull => SharedShape::convex_hull(&vertices),
            }
        }
    }
}

fn build_collider(
    shape: ColliderShape,
    transform: Transform,
    collider_props: ColliderProperties,
    mesh_renderer: Option<&BevyMeshRenderer>,
    skinned_renderer: Option<&BevySkinnedMeshRenderer>,
    asset_server: Option<&BevyAssetServer>,
) -> Option<ColliderBuilder> {
    let shape = build_shared_shape(
        shape,
        transform.scale,
        mesh_renderer,
        skinned_renderer,
        asset_server,
    )
    .or_else(|| {
        warn!("Failed to build requested collider shape, falling back to cuboid");
        build_shared_shape(
            ColliderShape::Cuboid,
            transform.scale,
            mesh_renderer,
            skinned_renderer,
            asset_server,
        )
    })?;

    let collision_groups = InteractionGroups::new(
        Group::from_bits_retain(collider_props.collision_memberships),
        Group::from_bits_retain(collider_props.collision_filter),
        InteractionTestMode::And,
    );

    let solver_groups = InteractionGroups::new(
        Group::from_bits_retain(collider_props.solver_memberships),
        Group::from_bits_retain(collider_props.solver_filter),
        InteractionTestMode::And,
    );

    Some(
        ColliderBuilder::new(shape)
            .position(build_isometry(
                collider_props.translation_offset,
                collider_props.rotation_offset,
            ))
            .friction(collider_props.friction)
            .friction_combine_rule(combine_rule(collider_props.friction_combine_rule))
            .restitution(collider_props.restitution)
            .restitution_combine_rule(combine_rule(collider_props.restitution_combine_rule))
            .density(collider_props.density.max(0.0))
            .sensor(collider_props.is_sensor)
            .enabled(collider_props.enabled)
            .collision_groups(collision_groups)
            .solver_groups(solver_groups),
    )
}

fn build_rigid_body(
    body_kind: BodyKind,
    transform: Transform,
    body_props: RigidBodyProperties,
) -> RigidBodyBuilder {
    let mut builder = match body_kind {
        BodyKind::Dynamic { .. } => RigidBodyBuilder::dynamic(),
        BodyKind::Kinematic { mode } => match mode {
            KinematicMode::PositionBased => RigidBodyBuilder::kinematic_position_based(),
            KinematicMode::VelocityBased => RigidBodyBuilder::kinematic_velocity_based(),
        },
        BodyKind::Fixed => RigidBodyBuilder::fixed(),
    };

    builder = builder
        .pose(build_isometry(transform.position, transform.rotation))
        .gravity_scale(body_props.gravity_scale)
        .dominance_group(body_props.dominance_group)
        .enabled_translations(
            !body_props.lock_translation_x,
            !body_props.lock_translation_y,
            !body_props.lock_translation_z,
        )
        .enabled_rotations(
            !body_props.lock_rotation_x,
            !body_props.lock_rotation_y,
            !body_props.lock_rotation_z,
        )
        .linear_damping(body_props.linear_damping)
        .angular_damping(body_props.angular_damping)
        .linvel(body_props.linear_velocity)
        .angvel(body_props.angular_velocity)
        .can_sleep(body_props.can_sleep)
        .sleeping(body_props.sleeping)
        .ccd_enabled(body_props.ccd_enabled);

    if let BodyKind::Dynamic { mass } = body_kind {
        builder = builder.additional_mass(mass);
    }

    builder
}

fn insert_physics_for_entity(
    phys: &mut PhysicsResource,
    entity: Entity,
    transform: Transform,
    shape: ColliderShape,
    dynamic: Option<DynamicRigidBody>,
    kinematic: Option<KinematicRigidBody>,
    fixed: bool,
    body_props: RigidBodyProperties,
    collider_props: ColliderProperties,
    mesh_renderer: Option<&BevyMeshRenderer>,
    skinned_renderer: Option<&BevySkinnedMeshRenderer>,
    asset_server: Option<&BevyAssetServer>,
) -> Option<PhysicsHandle> {
    let body_kind = resolve_body_kind(dynamic, kinematic, fixed)?;

    let rigid_body = build_rigid_body(body_kind, transform, body_props).build();
    let collider = build_collider(
        shape,
        transform,
        collider_props,
        mesh_renderer,
        skinned_renderer,
        asset_server,
    )?
    .build();

    let rigid_body_handle = phys.rigid_body_set.insert(rigid_body);
    let collider_handle =
        phys.collider_set
            .insert_with_parent(collider, rigid_body_handle, &mut phys.rigid_body_set);

    let handle = PhysicsHandle {
        rigid_body: rigid_body_handle,
        collider: collider_handle,
    };

    phys.physics_entities.insert(entity, handle);
    phys.collider_entities.insert(collider_handle, entity);

    Some(handle)
}

fn remove_physics_entity(phys: &mut PhysicsResource, entity: Entity) {
    if let Some(joint_handle) = phys.physics_joints.remove(&entity) {
        let _ = phys.impulse_joint_set.remove(joint_handle, true);
    }

    if let Some(handle) = phys.physics_entities.remove(&entity) {
        phys.collider_entities.remove(&handle.collider);
        phys.rigid_body_set.remove(
            handle.rigid_body,
            &mut phys.island_manager,
            &mut phys.collider_set,
            &mut phys.impulse_joint_set,
            &mut phys.multibody_joint_set,
            true,
        );
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
fn point_projection_has_candidates(pipeline: &rapier3d::prelude::QueryPipeline<'_>) -> bool {
    if pipeline.bvh.is_empty() {
        return false;
    }

    pipeline.bvh.leaves(|_| true).any(|leaf| {
        pipeline
            .colliders
            .get_unknown_gen(leaf)
            .map(|(collider, handle)| pipeline.filter.test(pipeline.bodies, handle, collider))
            .unwrap_or(false)
    })
}

fn map_shape_cast_status(status: ShapeCastStatus) -> PhysicsShapeCastStatus {
    match status {
        ShapeCastStatus::Converged => PhysicsShapeCastStatus::Converged,
        ShapeCastStatus::OutOfIterations => PhysicsShapeCastStatus::OutOfIterations,
        ShapeCastStatus::Failed => PhysicsShapeCastStatus::Failed,
        ShapeCastStatus::PenetratingOrWithinTargetDist => {
            PhysicsShapeCastStatus::PenetratingOrWithinTargetDist
        }
    }
}

fn build_impulse_joint(joint: PhysicsJoint) -> rapier3d::prelude::GenericJoint {
    let anchor1 = to_vector(joint.local_anchor1);
    let anchor2 = to_vector(joint.local_anchor2);
    let axis1 = to_vector(normalize_axis(joint.local_axis1, Vec3::Y));
    let axis2 = to_vector(normalize_axis(joint.local_axis2, Vec3::Y));

    match joint.kind {
        PhysicsJointKind::Fixed => FixedJointBuilder::new()
            .contacts_enabled(joint.contacts_enabled)
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build()
            .into(),
        PhysicsJointKind::Spherical => {
            let mut builder = SphericalJointBuilder::new()
                .contacts_enabled(joint.contacts_enabled)
                .local_anchor1(anchor1)
                .local_anchor2(anchor2);

            if joint.limit_enabled {
                builder = builder
                    .limits(JointAxis::AngX, joint.limits)
                    .limits(JointAxis::AngY, joint.limits)
                    .limits(JointAxis::AngZ, joint.limits);
            }

            if joint.motor.enabled {
                builder = builder
                    .motor(
                        JointAxis::AngX,
                        joint.motor.target_position,
                        joint.motor.target_velocity,
                        joint.motor.stiffness,
                        joint.motor.damping,
                    )
                    .motor(
                        JointAxis::AngY,
                        joint.motor.target_position,
                        joint.motor.target_velocity,
                        joint.motor.stiffness,
                        joint.motor.damping,
                    )
                    .motor(
                        JointAxis::AngZ,
                        joint.motor.target_position,
                        joint.motor.target_velocity,
                        joint.motor.stiffness,
                        joint.motor.damping,
                    )
                    .motor_max_force(JointAxis::AngX, joint.motor.max_force)
                    .motor_max_force(JointAxis::AngY, joint.motor.max_force)
                    .motor_max_force(JointAxis::AngZ, joint.motor.max_force);
            }

            builder.build().into()
        }
        PhysicsJointKind::Revolute => {
            let mut builder = RevoluteJointBuilder::new(axis1)
                .contacts_enabled(joint.contacts_enabled)
                .local_anchor1(anchor1)
                .local_anchor2(anchor2);

            if joint.limit_enabled {
                builder = builder.limits(joint.limits);
            }

            if joint.motor.enabled {
                builder = builder
                    .motor(
                        joint.motor.target_position,
                        joint.motor.target_velocity,
                        joint.motor.stiffness,
                        joint.motor.damping,
                    )
                    .motor_max_force(joint.motor.max_force);
            }

            builder.build().into()
        }
        PhysicsJointKind::Prismatic => {
            let mut builder = PrismaticJointBuilder::new(axis1)
                .contacts_enabled(joint.contacts_enabled)
                .local_anchor1(anchor1)
                .local_anchor2(anchor2)
                .local_axis1(axis1)
                .local_axis2(axis2);

            if joint.limit_enabled {
                builder = builder.limits(joint.limits);
            }

            if joint.motor.enabled {
                builder = builder
                    .set_motor(
                        joint.motor.target_position,
                        joint.motor.target_velocity,
                        joint.motor.stiffness,
                        joint.motor.damping,
                    )
                    .motor_max_force(joint.motor.max_force);
            }

            builder.build().into()
        }
    }
}

pub fn sync_entities_to_physics_system(
    mut commands: Commands,
    mut phys: ResMut<PhysicsResource>,
    mut previous_defaults: Local<Option<PhysicsWorldDefaults>>,
    asset_server: Option<Res<BevyAssetServer>>,
    world_defaults_query: Query<(Entity, &PhysicsWorldDefaults)>,
    create_query: Query<
        (
            Entity,
            &BevyTransform,
            &ColliderShape,
            Option<&DynamicRigidBody>,
            Option<&KinematicRigidBody>,
            Option<&FixedCollider>,
            Option<&RigidBodyProperties>,
            Option<&RigidBodyPropertyInheritance>,
            Option<&ColliderProperties>,
            Option<&ColliderPropertyInheritance>,
            Option<&BevyMeshRenderer>,
            Option<&BevySkinnedMeshRenderer>,
        ),
        (
            Without<PhysicsHandle>,
            Or<(
                With<DynamicRigidBody>,
                With<KinematicRigidBody>,
                With<FixedCollider>,
            )>,
        ),
    >,
    changed_query: Query<
        Entity,
        (
            With<PhysicsHandle>,
            Or<(
                Changed<ColliderShape>,
                Changed<DynamicRigidBody>,
                Changed<KinematicRigidBody>,
                Changed<FixedCollider>,
                Changed<RigidBodyProperties>,
                Changed<RigidBodyPropertyInheritance>,
                Changed<ColliderProperties>,
                Changed<ColliderPropertyInheritance>,
            )>,
        ),
    >,
    changed_transform_query: Query<
        Entity,
        (
            With<PhysicsHandle>,
            Changed<BevyTransform>,
            Or<(With<FixedCollider>, With<KinematicRigidBody>)>,
        ),
    >,
    mut removed_rigid_body_props: RemovedComponents<RigidBodyProperties>,
    mut removed_rigid_inheritance: RemovedComponents<RigidBodyPropertyInheritance>,
    mut removed_collider_props: RemovedComponents<ColliderProperties>,
    mut removed_collider_inheritance: RemovedComponents<ColliderPropertyInheritance>,
    handle_query: Query<
        (
            Entity,
            &BevyTransform,
            Option<&ColliderShape>,
            Option<&DynamicRigidBody>,
            Option<&KinematicRigidBody>,
            Option<&FixedCollider>,
            Option<&RigidBodyProperties>,
            Option<&RigidBodyPropertyInheritance>,
            Option<&ColliderProperties>,
            Option<&ColliderPropertyInheritance>,
            Option<&BevyMeshRenderer>,
            Option<&BevySkinnedMeshRenderer>,
        ),
        With<PhysicsHandle>,
    >,
) {
    if !phys.running {
        return;
    }

    let asset_server = asset_server.as_deref();
    let mut world_defaults = PhysicsWorldDefaults {
        gravity: phys.gravity,
        collider_properties: phys.default_collider_properties,
        rigid_body_properties: phys.default_rigid_body_properties,
    };
    if let Some(component_defaults) = select_world_defaults_component(&world_defaults_query) {
        world_defaults = component_defaults;
    }

    let previous_defaults_value = previous_defaults.replace(world_defaults);
    let rigid_defaults_changed = previous_defaults_value
        .map(|previous| previous.rigid_body_properties != world_defaults.rigid_body_properties)
        .unwrap_or(false);
    let collider_defaults_changed = previous_defaults_value
        .map(|previous| previous.collider_properties != world_defaults.collider_properties)
        .unwrap_or(false);

    phys.gravity = world_defaults.gravity;
    phys.default_rigid_body_properties = world_defaults.rigid_body_properties;
    phys.default_collider_properties = world_defaults.collider_properties;

    let mut invalid_entities = Vec::new();
    for (entity, _, shape, dynamic, kinematic, fixed, _, _, _, _, _, _) in handle_query.iter() {
        let body_kind = resolve_body_kind(dynamic.copied(), kinematic.copied(), fixed.is_some());
        if body_kind.is_none() || shape.is_none() {
            invalid_entities.push(entity);
        }
    }

    for entity in invalid_entities {
        remove_physics_entity(&mut phys, entity);
        commands.entity(entity).remove::<PhysicsHandle>();
    }

    let mut rebuild_entities: HashSet<Entity> = HashSet::new();
    for entity in changed_query.iter() {
        rebuild_entities.insert(entity);
    }
    for entity in changed_transform_query.iter() {
        rebuild_entities.insert(entity);
    }
    for entity in removed_rigid_body_props.read() {
        rebuild_entities.insert(entity);
    }
    for entity in removed_rigid_inheritance.read() {
        rebuild_entities.insert(entity);
    }
    for entity in removed_collider_props.read() {
        rebuild_entities.insert(entity);
    }
    for entity in removed_collider_inheritance.read() {
        rebuild_entities.insert(entity);
    }
    if rigid_defaults_changed || collider_defaults_changed {
        for (
            entity,
            _,
            _,
            _dynamic,
            _kinematic,
            _fixed,
            body_props,
            body_inheritance,
            collider_props,
            collider_inheritance,
            _mesh_renderer,
            _skinned_renderer,
        ) in handle_query.iter()
        {
            let uses_world_body = rigid_defaults_changed
                && rigid_body_uses_world_defaults(body_props.copied(), body_inheritance.copied());
            let uses_world_collider = collider_defaults_changed
                && collider_uses_world_defaults(
                    collider_props.copied(),
                    collider_inheritance.copied(),
                );
            if uses_world_body || uses_world_collider {
                rebuild_entities.insert(entity);
            }
        }
    }

    for entity in rebuild_entities {
        if let Ok((
            _,
            transform,
            shape,
            dynamic,
            kinematic,
            fixed,
            body_props,
            body_inheritance,
            collider_props,
            collider_inheritance,
            mesh_renderer,
            skinned_renderer,
        )) = handle_query.get(entity)
        {
            remove_physics_entity(&mut phys, entity);
            commands.entity(entity).remove::<PhysicsHandle>();

            if let Some(shape) = shape.copied() {
                let body_props = effective_rigid_body_properties(
                    body_props.copied(),
                    body_inheritance.copied(),
                    world_defaults.rigid_body_properties,
                );
                let collider_props = effective_collider_properties(
                    collider_props.copied(),
                    collider_inheritance.copied(),
                    world_defaults.collider_properties,
                );
                if let Some(handle) = insert_physics_for_entity(
                    &mut phys,
                    entity,
                    transform.0,
                    shape,
                    dynamic.copied(),
                    kinematic.copied(),
                    fixed.is_some(),
                    body_props,
                    collider_props,
                    mesh_renderer,
                    skinned_renderer,
                    asset_server,
                ) {
                    commands.entity(entity).try_insert(handle);
                }
            }
        }
    }

    for (
        entity,
        transform,
        shape,
        dynamic,
        kinematic,
        fixed,
        body_props,
        body_inheritance,
        collider_props,
        collider_inheritance,
        mesh_renderer,
        skinned_renderer,
    ) in create_query.iter()
    {
        let body_props = effective_rigid_body_properties(
            body_props.copied(),
            body_inheritance.copied(),
            world_defaults.rigid_body_properties,
        );
        let collider_props = effective_collider_properties(
            collider_props.copied(),
            collider_inheritance.copied(),
            world_defaults.collider_properties,
        );

        if let Some(handle) = insert_physics_for_entity(
            &mut phys,
            entity,
            transform.0,
            *shape,
            dynamic.copied(),
            kinematic.copied(),
            fixed.is_some(),
            body_props,
            collider_props,
            mesh_renderer,
            skinned_renderer,
            asset_server,
        ) {
            commands.entity(entity).try_insert(handle);
        }
    }
}

pub fn sync_transforms_to_physics_system(
    mut phys: ResMut<PhysicsResource>,
    query: Query<
        (&PhysicsHandle, &BevyTransform, Option<&KinematicRigidBody>),
        (
            Changed<BevyTransform>,
            Or<(With<FixedCollider>, With<KinematicRigidBody>)>,
        ),
    >,
) {
    if !phys.running {
        return;
    }

    for (handle, transform, kinematic) in query.iter() {
        if let Some(rigid_body) = phys.rigid_body_set.get_mut(handle.rigid_body) {
            let iso = build_isometry(transform.0.position, transform.0.rotation);

            if matches!(kinematic, Some(k) if k.mode == KinematicMode::PositionBased) {
                rigid_body.set_next_kinematic_position(iso);
            } else {
                rigid_body.set_position(iso, true);
            }
        }
    }
}

pub fn sync_joints_to_physics_system(
    mut phys: ResMut<PhysicsResource>,
    changed_query: Query<(Entity, &PhysicsJoint), Changed<PhysicsJoint>>,
    all_query: Query<(Entity, &PhysicsJoint)>,
) {
    if !phys.running {
        return;
    }

    let stale_handles: Vec<Entity> = phys
        .physics_joints
        .iter()
        .filter_map(|(entity, handle)| {
            if phys.impulse_joint_set.get(*handle).is_none() {
                Some(*entity)
            } else {
                None
            }
        })
        .collect();

    for entity in stale_handles {
        phys.physics_joints.remove(&entity);
    }

    let mut changed_map: HashMap<Entity, PhysicsJoint> = HashMap::new();
    for (entity, joint) in changed_query.iter() {
        changed_map.insert(entity, *joint);
    }

    for (entity, joint) in changed_map {
        if let Some(handle) = phys.physics_joints.remove(&entity) {
            let _ = phys.impulse_joint_set.remove(handle, true);
        }

        let Some(target) = joint.target else {
            continue;
        };
        let Some(self_handle) = phys.physics_entities.get(&entity).copied() else {
            continue;
        };
        let Some(target_handle) = phys.physics_entities.get(&target).copied() else {
            continue;
        };
        if self_handle.rigid_body == target_handle.rigid_body {
            continue;
        }

        let data = build_impulse_joint(joint);
        let handle = phys.impulse_joint_set.insert(
            self_handle.rigid_body,
            target_handle.rigid_body,
            data,
            true,
        );
        phys.physics_joints.insert(entity, handle);
    }

    for (entity, joint) in all_query.iter() {
        if phys.physics_joints.contains_key(&entity) {
            continue;
        }

        let Some(target) = joint.target else {
            continue;
        };
        let Some(self_handle) = phys.physics_entities.get(&entity).copied() else {
            continue;
        };
        let Some(target_handle) = phys.physics_entities.get(&target).copied() else {
            continue;
        };
        if self_handle.rigid_body == target_handle.rigid_body {
            continue;
        }

        let data = build_impulse_joint(*joint);
        let handle = phys.impulse_joint_set.insert(
            self_handle.rigid_body,
            target_handle.rigid_body,
            data,
            true,
        );
        phys.physics_joints.insert(entity, handle);
    }
}

pub fn character_controller_system(
    mut phys: ResMut<PhysicsResource>,
    time: Res<DeltaTime>,
    mut query: Query<(
        Entity,
        &CharacterController,
        &CharacterControllerInput,
        &PhysicsHandle,
        &mut CharacterControllerOutput,
    )>,
) {
    if !phys.running {
        return;
    }

    let dt = time.0.max(1.0e-6);
    let crate::physics::physics_resource::PhysicsResource {
        broad_phase,
        narrow_phase,
        rigid_body_set,
        collider_set,
        ..
    } = &mut *phys;
    let mut query_pipeline = broad_phase.as_query_pipeline_mut(
        narrow_phase.query_dispatcher(),
        rigid_body_set,
        collider_set,
        QueryFilter::default(),
    );

    for (entity, controller, input, handle, mut output) in query.iter_mut() {
        *output = CharacterControllerOutput::default();

        let Some(current_pos) = query_pipeline
            .bodies
            .get(handle.rigid_body)
            .map(|body| (*body.position(), body.body_type()))
        else {
            continue;
        };

        if !current_pos.1.is_kinematic() {
            continue;
        }

        let Some(character_shape) = query_pipeline
            .colliders
            .get(handle.collider)
            .map(|collider| collider.shared_shape().clone())
        else {
            continue;
        };

        let up = normalize_axis(controller.up, Vec3::Y);
        let autostep =
            if controller.autostep_max_height > 0.0 && controller.autostep_min_width > 0.0 {
                Some(CharacterAutostep {
                    max_height: CharacterLength::Absolute(controller.autostep_max_height),
                    min_width: CharacterLength::Absolute(controller.autostep_min_width),
                    include_dynamic_bodies: controller.autostep_include_dynamic_bodies,
                })
            } else {
                None
            };
        let snap_to_ground = if controller.snap_to_ground > 0.0 {
            Some(CharacterLength::Absolute(controller.snap_to_ground))
        } else {
            None
        };

        let rapier_controller = KinematicCharacterController {
            up: to_vector(up),
            offset: CharacterLength::Absolute(controller.offset.max(0.0001)),
            slide: controller.slide,
            autostep,
            max_slope_climb_angle: controller.max_slope_climb_angle,
            min_slope_slide_angle: controller.min_slope_slide_angle,
            snap_to_ground,
            normal_nudge_factor: controller.normal_nudge_factor.max(0.0),
        };

        let mut collisions = Vec::new();
        let query_filter = QueryFilter::default()
            .exclude_collider(handle.collider)
            .exclude_rigid_body(handle.rigid_body);
        let effective_movement = rapier_controller.move_shape(
            dt,
            &query_pipeline.as_ref().with_filter(query_filter),
            &*character_shape,
            &current_pos.0,
            to_vector(input.desired_translation),
            |collision| collisions.push(collision),
        );

        if let Some(body) = query_pipeline.bodies.get_mut(handle.rigid_body) {
            let next_pos = Pose::from_translation(effective_movement.translation) * current_pos.0;
            if current_pos.1 == RigidBodyType::KinematicPositionBased {
                body.set_next_kinematic_position(next_pos);
            } else {
                body.set_linvel(effective_movement.translation / dt, true);
            }
        }

        if controller.apply_impulses_to_dynamic_bodies && !collisions.is_empty() {
            rapier_controller.solve_character_collision_impulses(
                dt,
                &mut query_pipeline,
                character_shape.as_ref(),
                controller.character_mass.max(0.0001),
                collisions.iter(),
            );
        }

        output.effective_translation = from_vector(effective_movement.translation);
        output.grounded = effective_movement.grounded;
        output.sliding_down_slope = effective_movement.is_sliding_down_slope;
        output.collision_count = collisions.len() as u32;

        if query_pipeline.bodies.get(handle.rigid_body).is_none() {
            warn!(
                "Character controller entity {:?} lost its rigid body during simulation.",
                entity
            );
        }
    }
}

#[inline]
pub fn physics_step_system(mut phys: ResMut<PhysicsResource>, time: Res<DeltaTime>) {
    if !phys.running {
        return;
    }

    let dt = time.0;
    let phys_data = &mut *phys;

    phys_data.integration_parameters.dt = dt;

    let gravity_vector = Vec3::new(
        phys_data.gravity.x,
        phys_data.gravity.y,
        phys_data.gravity.z,
    );

    phys_data.pipeline.step(
        gravity_vector,
        &phys_data.integration_parameters,
        &mut phys_data.island_manager,
        &mut phys_data.broad_phase,
        &mut phys_data.narrow_phase,
        &mut phys_data.rigid_body_set,
        &mut phys_data.collider_set,
        &mut phys_data.impulse_joint_set,
        &mut phys_data.multibody_joint_set,
        &mut phys_data.ccd_solver,
        &(),
        &(),
    );
}

pub fn sync_physics_to_entities_system(
    phys: Res<PhysicsResource>,
    mut query: Query<(&PhysicsHandle, &mut BevyTransform)>,
) {
    if !phys.running {
        return;
    }

    for (handle, mut transform) in query.iter_mut() {
        if let Some(rigid_body) = phys.rigid_body_set.get(handle.rigid_body) {
            if rigid_body.is_dynamic() || rigid_body.is_kinematic() {
                let transform = &mut transform.0;
                let rb_pos = rigid_body.translation();
                let rb_rot = rigid_body.rotation();

                transform.position.x = rb_pos.x;
                transform.position.y = rb_pos.y;
                transform.position.z = rb_pos.z;

                transform.rotation.x = rb_rot.x;
                transform.rotation.y = rb_rot.y;
                transform.rotation.z = rb_rot.z;
                transform.rotation.w = rb_rot.w;
            }
        }
    }
}

pub fn physics_scene_query_system(
    phys: Res<PhysicsResource>,
    asset_server: Option<Res<BevyAssetServer>>,
    mesh_query: Query<&BevyMeshRenderer>,
    skinned_query: Query<&BevySkinnedMeshRenderer>,
    mut ray_query: Query<(Entity, &PhysicsRayCast, &mut PhysicsRayCastHit)>,
    mut point_query: Query<(
        Entity,
        &PhysicsPointProjection,
        &mut PhysicsPointProjectionHit,
    )>,
    mut shape_query: Query<(Entity, &PhysicsShapeCast, &mut PhysicsShapeCastHit)>,
) {
    if !phys.running {
        return;
    }

    let asset_server = asset_server.as_deref();

    for (entity, request, mut hit) in ray_query.iter_mut() {
        *hit = PhysicsRayCastHit::default();

        let direction = normalize_axis(request.direction, Vec3::ZERO);
        if direction.length_squared() <= 1.0e-8 {
            continue;
        }

        let self_handle = if request.exclude_self {
            phys.physics_entities.get(&entity).copied()
        } else {
            None
        };
        let filter = build_query_filter(request.filter, self_handle);
        let pipeline = phys.query_pipeline_with_filter(filter);

        let ray = Ray::new(to_vector(request.origin), to_vector(direction));
        if let Some((collider_handle, ray_hit)) =
            pipeline.cast_ray_and_get_normal(&ray, request.max_toi.max(0.0), request.solid)
        {
            hit.has_hit = true;
            hit.hit_entity = phys.entity_for_collider(collider_handle);
            hit.toi = ray_hit.time_of_impact;
            hit.point = from_vector(ray.origin + ray.dir * ray_hit.time_of_impact);
            hit.normal = from_vector(ray_hit.normal);
        }
    }

    for (entity, request, mut hit) in point_query.iter_mut() {
        *hit = PhysicsPointProjectionHit::default();

        let self_handle = if request.exclude_self {
            phys.physics_entities.get(&entity).copied()
        } else {
            None
        };
        let filter = build_query_filter(request.filter, self_handle);
        let pipeline = phys.query_pipeline_with_filter(filter);

        if !point_projection_has_candidates(&pipeline) {
            continue;
        }

        if let Some((collider_handle, projection)) =
            pipeline.project_point(to_vector(request.point), f32::MAX, request.solid)
        {
            let projected = from_vector(projection.point);
            hit.has_hit = true;
            hit.hit_entity = phys.entity_for_collider(collider_handle);
            hit.projected_point = projected;
            hit.is_inside = projection.is_inside;
            hit.distance = request.point.distance(projected);
        }
    }

    for (entity, request, mut hit) in shape_query.iter_mut() {
        *hit = PhysicsShapeCastHit::default();

        let self_handle = if request.exclude_self {
            phys.physics_entities.get(&entity).copied()
        } else {
            None
        };
        let filter = build_query_filter(request.filter, self_handle);
        let pipeline = phys.query_pipeline_with_filter(filter);

        let mesh_renderer = mesh_query.get(entity).ok();
        let skinned_renderer = skinned_query.get(entity).ok();

        let Some(shape) = build_shared_shape(
            request.shape,
            request.scale,
            mesh_renderer,
            skinned_renderer,
            asset_server,
        ) else {
            continue;
        };

        let options = ShapeCastOptions {
            max_time_of_impact: request.max_time_of_impact.max(0.0),
            target_distance: request.target_distance.max(0.0),
            stop_at_penetration: request.stop_at_penetration,
            compute_impact_geometry_on_penetration: request.compute_impact_geometry_on_penetration,
        };

        let shape_pose = build_isometry(request.position, request.rotation);
        if let Some((collider_handle, cast_hit)) =
            pipeline.cast_shape(&shape_pose, to_vector(request.velocity), &*shape, options)
        {
            hit.has_hit = true;
            hit.hit_entity = phys.entity_for_collider(collider_handle);
            hit.toi = cast_hit.time_of_impact;
            hit.witness1 = from_vector(cast_hit.witness1);
            hit.witness2 = from_vector(cast_hit.witness2);
            hit.normal1 = from_vector(cast_hit.normal1);
            hit.normal2 = from_vector(cast_hit.normal2);
            hit.status = map_shape_cast_status(cast_hit.status);
        }
    }
}

pub fn cleanup_physics_system(
    mut commands: Commands,
    mut phys: ResMut<PhysicsResource>,
    query: Query<Entity, With<PhysicsHandle>>,
    mut local_state: Local<CleanupState>,
) {
    if !phys.running {
        return;
    }

    local_state.dead_entities.clear();
    local_state.out_of_bounds_entities.clear();

    let live_entities: HashSet<Entity> = query.iter().collect();

    for &entity in phys.physics_entities.keys() {
        if !live_entities.contains(&entity) {
            local_state.dead_entities.push(entity);
        }
    }

    for (&entity, &handle) in phys.physics_entities.iter() {
        if let Some(rigid_body) = phys.rigid_body_set.get(handle.rigid_body) {
            let position = rigid_body.translation();
            if !local_state.is_position_safe(&position) {
                warn!(
                    "Entity {:?} is out of bounds at [{:.2}, {:.2}, {:.2}], destroying to prevent Rapier panic",
                    entity, position.x, position.y, position.z
                );
                local_state.out_of_bounds_entities.push(entity);
            }
        }
    }

    for &entity in &local_state.out_of_bounds_entities {
        commands.entity(entity).try_despawn();
    }

    let mut all_to_cleanup = std::mem::take(&mut local_state.dead_entities);
    all_to_cleanup.append(&mut local_state.out_of_bounds_entities);

    for entity in all_to_cleanup {
        remove_physics_entity(&mut phys, entity);
    }

    let valid_colliders: HashSet<_> = phys.collider_set.iter().map(|(handle, _)| handle).collect();
    phys.collider_entities
        .retain(|handle, _| valid_colliders.contains(handle));

    let valid_joints: HashSet<_> = phys
        .impulse_joint_set
        .iter()
        .map(|(handle, _)| handle)
        .collect();
    phys.physics_joints
        .retain(|_, handle| valid_joints.contains(handle));
}

pub fn configure_broad_phase_optimal(_broad_phase: &mut DefaultBroadPhase) {}

pub fn get_optimal_integration_parameters() -> IntegrationParameters {
    let mut params = IntegrationParameters::default();

    params.max_ccd_substeps = 4;
    params.num_internal_stabilization_iterations = 4;
    params.num_solver_iterations = 4;
    params.normalized_allowed_linear_error = 0.001;

    params
}
