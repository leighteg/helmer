use bevy_ecs::{component::Component, entity::Entity};
use glam::{Quat, Vec3};
use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};

#[derive(Component, Debug, Clone, Copy)]
pub struct PhysicsHandle {
    pub rigid_body: RigidBodyHandle,
    pub collider: ColliderHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshColliderLod {
    Lod0,
    Lod1,
    Lod2,
    Lowest,
    Specific(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshColliderKind {
    TriMesh,
    ConvexHull,
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub enum ColliderShape {
    Cuboid,
    Sphere,
    CapsuleY,
    CylinderY,
    ConeY,
    RoundCuboid {
        border_radius: f32,
    },
    Mesh {
        mesh_id: Option<usize>,
        lod: MeshColliderLod,
        kind: MeshColliderKind,
    },
}

impl Default for ColliderShape {
    fn default() -> Self {
        Self::Cuboid
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsCombineRule {
    Average,
    Min,
    Multiply,
    Max,
}

impl Default for PhysicsCombineRule {
    fn default() -> Self {
        Self::Average
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct ColliderProperties {
    pub friction: f32,
    pub restitution: f32,
    pub density: f32,
    pub is_sensor: bool,
    pub enabled: bool,
    pub collision_memberships: u32,
    pub collision_filter: u32,
    pub solver_memberships: u32,
    pub solver_filter: u32,
    pub friction_combine_rule: PhysicsCombineRule,
    pub restitution_combine_rule: PhysicsCombineRule,
    pub translation_offset: Vec3,
    pub rotation_offset: Quat,
}

impl Default for ColliderProperties {
    fn default() -> Self {
        Self {
            friction: 0.7,
            restitution: 0.0,
            density: 1.0,
            is_sensor: false,
            enabled: true,
            collision_memberships: u32::MAX,
            collision_filter: u32::MAX,
            solver_memberships: u32::MAX,
            solver_filter: u32::MAX,
            friction_combine_rule: PhysicsCombineRule::Average,
            restitution_combine_rule: PhysicsCombineRule::Average,
            translation_offset: Vec3::ZERO,
            rotation_offset: Quat::IDENTITY,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColliderPropertyInheritance {
    pub friction: bool,
    pub restitution: bool,
    pub density: bool,
    pub is_sensor: bool,
    pub enabled: bool,
    pub collision_memberships: bool,
    pub collision_filter: bool,
    pub solver_memberships: bool,
    pub solver_filter: bool,
    pub friction_combine_rule: bool,
    pub restitution_combine_rule: bool,
    pub translation_offset: bool,
    pub rotation_offset: bool,
}

impl ColliderPropertyInheritance {
    pub fn all() -> Self {
        Self {
            friction: true,
            restitution: true,
            density: true,
            is_sensor: true,
            enabled: true,
            collision_memberships: true,
            collision_filter: true,
            solver_memberships: true,
            solver_filter: true,
            friction_combine_rule: true,
            restitution_combine_rule: true,
            translation_offset: true,
            rotation_offset: true,
        }
    }

    pub fn none() -> Self {
        Self {
            friction: false,
            restitution: false,
            density: false,
            is_sensor: false,
            enabled: false,
            collision_memberships: false,
            collision_filter: false,
            solver_memberships: false,
            solver_filter: false,
            friction_combine_rule: false,
            restitution_combine_rule: false,
            translation_offset: false,
            rotation_offset: false,
        }
    }

    pub fn any_inherited(self) -> bool {
        self.friction
            || self.restitution
            || self.density
            || self.is_sensor
            || self.enabled
            || self.collision_memberships
            || self.collision_filter
            || self.solver_memberships
            || self.solver_filter
            || self.friction_combine_rule
            || self.restitution_combine_rule
            || self.translation_offset
            || self.rotation_offset
    }
}

impl Default for ColliderPropertyInheritance {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct DynamicRigidBody {
    pub mass: f32,
}

impl Default for DynamicRigidBody {
    fn default() -> Self {
        Self { mass: 1.0 }
    }
}

#[derive(Component, Debug, Clone, Copy)]
pub struct FixedCollider;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KinematicMode {
    PositionBased,
    VelocityBased,
}

impl Default for KinematicMode {
    fn default() -> Self {
        Self::PositionBased
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct KinematicRigidBody {
    pub mode: KinematicMode,
}

impl Default for KinematicRigidBody {
    fn default() -> Self {
        Self {
            mode: KinematicMode::PositionBased,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct RigidBodyProperties {
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub ccd_enabled: bool,
    pub can_sleep: bool,
    pub sleeping: bool,
    pub dominance_group: i8,
    pub lock_translation_x: bool,
    pub lock_translation_y: bool,
    pub lock_translation_z: bool,
    pub lock_rotation_x: bool,
    pub lock_rotation_y: bool,
    pub lock_rotation_z: bool,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

impl Default for RigidBodyProperties {
    fn default() -> Self {
        Self {
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            ccd_enabled: true,
            can_sleep: true,
            sleeping: false,
            dominance_group: 0,
            lock_translation_x: false,
            lock_translation_y: false,
            lock_translation_z: false,
            lock_rotation_x: false,
            lock_rotation_y: false,
            lock_rotation_z: false,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointForce {
    pub force: Vec3,
    pub point: Vec3,
    pub wake_up: bool,
}

#[derive(Component, Debug, Clone, PartialEq)]
pub struct RigidBodyForces {
    pub force: Vec3,
    pub force_wake_up: bool,
    pub torque: Vec3,
    pub torque_wake_up: bool,
    pub point_forces: Vec<PointForce>,
}

impl Default for RigidBodyForces {
    fn default() -> Self {
        Self {
            force: Vec3::ZERO,
            force_wake_up: false,
            torque: Vec3::ZERO,
            torque_wake_up: false,
            point_forces: Vec::new(),
        }
    }
}

#[derive(Component, Debug, Clone, PartialEq)]
pub struct RigidBodyTransientForces {
    pub force: Vec3,
    pub force_wake_up: bool,
    pub torque: Vec3,
    pub torque_wake_up: bool,
    pub point_forces: Vec<PointForce>,
}

impl Default for RigidBodyTransientForces {
    fn default() -> Self {
        Self {
            force: Vec3::ZERO,
            force_wake_up: false,
            torque: Vec3::ZERO,
            torque_wake_up: false,
            point_forces: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointImpulse {
    pub impulse: Vec3,
    pub point: Vec3,
    pub wake_up: bool,
}

#[derive(Component, Debug, Clone, PartialEq)]
pub struct RigidBodyImpulseQueue {
    pub impulse: Vec3,
    pub impulse_wake_up: bool,
    pub angular_impulse: Vec3,
    pub angular_impulse_wake_up: bool,
    pub point_impulses: Vec<PointImpulse>,
}

impl Default for RigidBodyImpulseQueue {
    fn default() -> Self {
        Self {
            impulse: Vec3::ZERO,
            impulse_wake_up: false,
            angular_impulse: Vec3::ZERO,
            angular_impulse_wake_up: false,
            point_impulses: Vec::new(),
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct RigidBodyPropertyInheritance {
    pub linear_damping: bool,
    pub angular_damping: bool,
    pub gravity_scale: bool,
    pub ccd_enabled: bool,
    pub can_sleep: bool,
    pub sleeping: bool,
    pub dominance_group: bool,
    pub lock_translation_x: bool,
    pub lock_translation_y: bool,
    pub lock_translation_z: bool,
    pub lock_rotation_x: bool,
    pub lock_rotation_y: bool,
    pub lock_rotation_z: bool,
    pub linear_velocity: bool,
    pub angular_velocity: bool,
}

impl RigidBodyPropertyInheritance {
    pub fn all() -> Self {
        Self {
            linear_damping: true,
            angular_damping: true,
            gravity_scale: true,
            ccd_enabled: true,
            can_sleep: true,
            sleeping: true,
            dominance_group: true,
            lock_translation_x: true,
            lock_translation_y: true,
            lock_translation_z: true,
            lock_rotation_x: true,
            lock_rotation_y: true,
            lock_rotation_z: true,
            linear_velocity: true,
            angular_velocity: true,
        }
    }

    pub fn none() -> Self {
        Self {
            linear_damping: false,
            angular_damping: false,
            gravity_scale: false,
            ccd_enabled: false,
            can_sleep: false,
            sleeping: false,
            dominance_group: false,
            lock_translation_x: false,
            lock_translation_y: false,
            lock_translation_z: false,
            lock_rotation_x: false,
            lock_rotation_y: false,
            lock_rotation_z: false,
            linear_velocity: false,
            angular_velocity: false,
        }
    }

    pub fn any_inherited(self) -> bool {
        self.linear_damping
            || self.angular_damping
            || self.gravity_scale
            || self.ccd_enabled
            || self.can_sleep
            || self.sleeping
            || self.dominance_group
            || self.lock_translation_x
            || self.lock_translation_y
            || self.lock_translation_z
            || self.lock_rotation_x
            || self.lock_rotation_y
            || self.lock_rotation_z
            || self.linear_velocity
            || self.angular_velocity
    }
}

impl Default for RigidBodyPropertyInheritance {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PhysicsWorldDefaults {
    pub gravity: Vec3,
    pub collider_properties: ColliderProperties,
    pub rigid_body_properties: RigidBodyProperties,
}

impl Default for PhysicsWorldDefaults {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -98.1, 0.0),
            collider_properties: ColliderProperties::default(),
            rigid_body_properties: RigidBodyProperties::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsJointKind {
    Fixed,
    Spherical,
    Revolute,
    Prismatic,
}

impl Default for PhysicsJointKind {
    fn default() -> Self {
        Self::Fixed
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointMotor {
    pub enabled: bool,
    pub target_position: f32,
    pub target_velocity: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub max_force: f32,
}

impl Default for JointMotor {
    fn default() -> Self {
        Self {
            enabled: false,
            target_position: 0.0,
            target_velocity: 0.0,
            stiffness: 0.0,
            damping: 0.0,
            max_force: 0.0,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PhysicsJoint {
    pub target: Option<Entity>,
    pub kind: PhysicsJointKind,
    pub contacts_enabled: bool,
    pub local_anchor1: Vec3,
    pub local_anchor2: Vec3,
    pub local_axis1: Vec3,
    pub local_axis2: Vec3,
    pub limit_enabled: bool,
    pub limits: [f32; 2],
    pub motor: JointMotor,
}

impl Default for PhysicsJoint {
    fn default() -> Self {
        Self {
            target: None,
            kind: PhysicsJointKind::Fixed,
            contacts_enabled: true,
            local_anchor1: Vec3::ZERO,
            local_anchor2: Vec3::ZERO,
            local_axis1: Vec3::Y,
            local_axis2: Vec3::Y,
            limit_enabled: false,
            limits: [-1.0, 1.0],
            motor: JointMotor::default(),
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct CharacterController {
    pub up: Vec3,
    pub offset: f32,
    pub slide: bool,
    pub autostep_max_height: f32,
    pub autostep_min_width: f32,
    pub autostep_include_dynamic_bodies: bool,
    pub max_slope_climb_angle: f32,
    pub min_slope_slide_angle: f32,
    pub snap_to_ground: f32,
    pub normal_nudge_factor: f32,
    pub apply_impulses_to_dynamic_bodies: bool,
    pub character_mass: f32,
}

impl Default for CharacterController {
    fn default() -> Self {
        Self {
            up: Vec3::Y,
            offset: 0.01,
            slide: true,
            autostep_max_height: 0.0,
            autostep_min_width: 0.0,
            autostep_include_dynamic_bodies: false,
            max_slope_climb_angle: std::f32::consts::FRAC_PI_4,
            min_slope_slide_angle: std::f32::consts::FRAC_PI_4,
            snap_to_ground: 0.2,
            normal_nudge_factor: 1.0e-4,
            apply_impulses_to_dynamic_bodies: false,
            character_mass: 80.0,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct CharacterControllerInput {
    pub desired_translation: Vec3,
}

impl Default for CharacterControllerInput {
    fn default() -> Self {
        Self {
            desired_translation: Vec3::ZERO,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct CharacterControllerOutput {
    pub effective_translation: Vec3,
    pub grounded: bool,
    pub sliding_down_slope: bool,
    pub collision_count: u32,
}

impl Default for CharacterControllerOutput {
    fn default() -> Self {
        Self {
            effective_translation: Vec3::ZERO,
            grounded: false,
            sliding_down_slope: false,
            collision_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicsQueryFilter {
    pub flags: u32,
    pub groups_memberships: u32,
    pub groups_filter: u32,
    pub use_groups: bool,
}

impl PhysicsQueryFilter {
    pub const EXCLUDE_FIXED: u32 = 1 << 0;
    pub const EXCLUDE_KINEMATIC: u32 = 1 << 1;
    pub const EXCLUDE_DYNAMIC: u32 = 1 << 2;
    pub const EXCLUDE_SENSORS: u32 = 1 << 3;
    pub const EXCLUDE_SOLIDS: u32 = 1 << 4;
}

impl Default for PhysicsQueryFilter {
    fn default() -> Self {
        Self {
            flags: 0,
            groups_memberships: u32::MAX,
            groups_filter: u32::MAX,
            use_groups: false,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PhysicsRayCast {
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_toi: f32,
    pub solid: bool,
    pub filter: PhysicsQueryFilter,
    pub exclude_self: bool,
}

impl Default for PhysicsRayCast {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            direction: Vec3::NEG_Y,
            max_toi: 10_000.0,
            solid: true,
            filter: PhysicsQueryFilter::default(),
            exclude_self: true,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PhysicsRayCastHit {
    pub has_hit: bool,
    pub hit_entity: Option<Entity>,
    pub point: Vec3,
    pub normal: Vec3,
    pub toi: f32,
}

impl Default for PhysicsRayCastHit {
    fn default() -> Self {
        Self {
            has_hit: false,
            hit_entity: None,
            point: Vec3::ZERO,
            normal: Vec3::ZERO,
            toi: 0.0,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PhysicsPointProjection {
    pub point: Vec3,
    pub solid: bool,
    pub filter: PhysicsQueryFilter,
    pub exclude_self: bool,
}

impl Default for PhysicsPointProjection {
    fn default() -> Self {
        Self {
            point: Vec3::ZERO,
            solid: true,
            filter: PhysicsQueryFilter::default(),
            exclude_self: true,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PhysicsPointProjectionHit {
    pub has_hit: bool,
    pub hit_entity: Option<Entity>,
    pub projected_point: Vec3,
    pub is_inside: bool,
    pub distance: f32,
}

impl Default for PhysicsPointProjectionHit {
    fn default() -> Self {
        Self {
            has_hit: false,
            hit_entity: None,
            projected_point: Vec3::ZERO,
            is_inside: false,
            distance: 0.0,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PhysicsShapeCast {
    pub shape: ColliderShape,
    pub scale: Vec3,
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub max_time_of_impact: f32,
    pub target_distance: f32,
    pub stop_at_penetration: bool,
    pub compute_impact_geometry_on_penetration: bool,
    pub filter: PhysicsQueryFilter,
    pub exclude_self: bool,
}

impl Default for PhysicsShapeCast {
    fn default() -> Self {
        Self {
            shape: ColliderShape::Sphere,
            scale: Vec3::ONE,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            max_time_of_impact: 10_000.0,
            target_distance: 0.0,
            stop_at_penetration: false,
            compute_impact_geometry_on_penetration: true,
            filter: PhysicsQueryFilter::default(),
            exclude_self: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsShapeCastStatus {
    NoHit,
    Converged,
    OutOfIterations,
    Failed,
    PenetratingOrWithinTargetDist,
}

impl Default for PhysicsShapeCastStatus {
    fn default() -> Self {
        Self::NoHit
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PhysicsShapeCastHit {
    pub has_hit: bool,
    pub hit_entity: Option<Entity>,
    pub toi: f32,
    pub witness1: Vec3,
    pub witness2: Vec3,
    pub normal1: Vec3,
    pub normal2: Vec3,
    pub status: PhysicsShapeCastStatus,
}

impl Default for PhysicsShapeCastHit {
    fn default() -> Self {
        Self {
            has_hit: false,
            hit_entity: None,
            toi: 0.0,
            witness1: Vec3::ZERO,
            witness2: Vec3::ZERO,
            normal1: Vec3::ZERO,
            normal2: Vec3::ZERO,
            status: PhysicsShapeCastStatus::NoHit,
        }
    }
}
