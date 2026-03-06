use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Component, Entity, Resource, World};
use bevy_ecs::query::{With, Without};
use helmer::{
    animation::{AnimationChannel, AnimationClip, Interpolation, Keyframe, Pose, Skeleton},
    graphics::common::renderer::{
        SpriteAnimationPlayback, SpriteBlendMode, SpriteSheetAnimation, SpriteSpace, TextAlignH,
        TextAlignV, TextFontStyle,
    },
    provided::components::{
        AudioEmitter, AudioListener, Camera, EntityFollower, Light, LightType, LookAt,
        MeshRenderer, PoseOverride, SkinnedMeshRenderer, Spline, SplineFollower, SplineMode,
        SpriteImageSequence, SpriteRenderer, Text2d, Transform,
    },
    runtime::asset_server::{Handle, Scene},
};
use helmer_becs::physics::components::{
    CharacterController, CharacterControllerInput, ColliderProperties, ColliderPropertyInheritance,
    ColliderShape, DynamicRigidBody, FixedCollider, JointMotor, KinematicMode, KinematicRigidBody,
    MeshColliderKind, MeshColliderLod, PhysicsCombineRule, PhysicsJoint, PhysicsJointKind,
    PhysicsPointProjection, PhysicsQueryFilter, PhysicsRayCast, PhysicsShapeCast,
    PhysicsWorldDefaults, RigidBodyProperties, RigidBodyPropertyInheritance,
};
use helmer_becs::{
    BevyAnimator, BevyAudioEmitter, BevyAudioListener, BevyCamera, BevyEntityFollower, BevyLight,
    BevyLookAt, BevyMeshRenderer, BevyPoseOverride, BevySkinnedMeshRenderer, BevySpline,
    BevySplineFollower, BevySpriteImageSequence, BevySpriteRenderer, BevyText2d, BevyTransform,
    BevyWrapper,
    systems::scene_system::{
        EntityParent, SceneChild, SceneRoot, SceneSpawnedChildren, SpawnedScene,
        build_default_animator,
    },
};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

use crate::editor::{
    EditorPlayCamera, EditorTimelineState, EditorViewportCamera, FREECAM_ORBIT_DISTANCE_DEFAULT,
    Freecam,
    assets::{
        EditorAssetCache, EditorAudio, EditorMesh, EditorSkinnedMesh, EditorSprite, MeshSource,
        PrimitiveKind, SceneAssetPath, cached_audio_handle, cached_scene_handle,
        cached_texture_handle,
    },
    dynamic::{DynamicComponent, DynamicComponents},
    project::EditorProject,
    scripting::{ScriptComponent, ScriptEntry, ScriptInspectorField, normalize_script_language},
    timeline::{
        CameraKey, CameraTrack, ClipSegment, ClipTrack, JointKey, JointTrack, LightKey, LightTrack,
        PoseKey, PoseTrack, SplineKey, SplineTrack, TimelineInterpolation, TimelineTrack,
        TimelineTrackGroup, TransformKey, TransformTrack,
    },
};

pub use helmer_editor_runtime::scene_state::{EditorEntity, EditorRenderRefresh, WorldState};
pub type EditorSceneState = helmer_editor_runtime::scene_state::EditorSceneState<SceneDocument>;

#[derive(Component, Debug, Clone)]
pub struct PendingSkinnedMeshAsset {
    pub scene_handle: Handle<Scene>,
    pub node_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneDocument {
    pub version: u32,
    pub entities: Vec<SceneEntityData>,
    #[serde(default)]
    pub scene_child_animations: Vec<SceneChildAnimationData>,
    #[serde(default)]
    pub scene_child_pose_overrides: Vec<SceneChildPoseOverrideData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimationAssetDocument {
    pub version: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub tracks: Vec<AnimationTrackData>,
    #[serde(default)]
    pub clips: Vec<AnimationClipData>,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct PendingSceneChildAnimations {
    pub entries: Vec<SceneChildAnimationData>,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct PendingSceneChildPoseOverrides {
    pub entries: Vec<SceneChildPoseOverrideData>,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct PendingSceneChildRenderer {
    pub kind: SceneChildRendererKind,
    pub casts_shadow: bool,
    pub visible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntityData {
    pub name: Option<String>,
    pub transform: SerializedTransform,
    #[serde(default)]
    pub relation: Option<SceneEntityRelationData>,
    #[serde(default)]
    pub scene_child: Option<SceneChildLinkData>,
    #[serde(default)]
    pub scene_child_renderer: Option<SceneChildRendererData>,
    pub components: SceneComponents,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntityRelationData {
    pub parent: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneChildLinkData {
    pub scene_root: usize,
    pub scene_node_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneChildRendererKind {
    Auto,
    Mesh,
    Skinned,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SceneChildRendererData {
    pub kind: SceneChildRendererKind,
    pub casts_shadow: bool,
    pub visible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SceneComponents {
    pub mesh: Option<MeshComponentData>,
    #[serde(default)]
    pub skinned: Option<SkinnedMeshComponentData>,
    #[serde(default)]
    pub sprite: Option<SpriteComponentData>,
    #[serde(default)]
    pub text_2d: Option<Text2dComponentData>,
    pub light: Option<LightComponentData>,
    pub camera: Option<CameraComponentData>,
    pub scene: Option<SceneAssetData>,
    pub scripts: Vec<ScriptComponentData>,
    pub dynamic: Vec<DynamicComponent>,
    #[serde(default)]
    pub spline: Option<SplineComponentData>,
    #[serde(default)]
    pub spline_follower: Option<SplineFollowerData>,
    #[serde(default)]
    pub look_at: Option<LookAtData>,
    #[serde(default)]
    pub entity_follower: Option<EntityFollowerData>,
    #[serde(default)]
    pub animation: Option<AnimationComponentData>,
    #[serde(default)]
    pub pose_override: Option<PoseOverrideData>,
    #[serde(default)]
    pub freecam: bool,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freecam_settings: Option<SceneFreecamData>,
    #[serde(default)]
    pub physics: Option<PhysicsComponentData>,
    #[serde(default)]
    pub physics_world_defaults: Option<ScenePhysicsWorldDefaultsData>,
    #[serde(default)]
    pub audio_emitter: Option<AudioEmitterData>,
    #[serde(default)]
    pub audio_listener: Option<AudioListenerData>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SceneFreecamData {
    pub sensitivity: f32,
    pub smoothing: f32,
    pub move_accel: f32,
    pub move_decel: f32,
    pub speed_step: f32,
    pub speed_min: f32,
    pub speed_max: f32,
    pub boost_multiplier: f32,
    #[serde(default = "scene_freecam_orbit_distance_default")]
    pub orbit_distance: f32,
}

fn scene_freecam_orbit_distance_default() -> f32 {
    FREECAM_ORBIT_DISTANCE_DEFAULT
}

impl SceneFreecamData {
    fn from_component(component: Freecam) -> Self {
        Self {
            sensitivity: component.sensitivity,
            smoothing: component.smoothing,
            move_accel: component.move_accel,
            move_decel: component.move_decel,
            speed_step: component.speed_step,
            speed_min: component.speed_min,
            speed_max: component.speed_max,
            boost_multiplier: component.boost_multiplier,
            orbit_distance: component.orbit_distance,
        }
    }

    fn into_component(self) -> Freecam {
        let mut component = Freecam {
            sensitivity: self.sensitivity,
            smoothing: self.smoothing,
            move_accel: self.move_accel,
            move_decel: self.move_decel,
            speed_step: self.speed_step,
            speed_min: self.speed_min,
            speed_max: self.speed_max,
            boost_multiplier: self.boost_multiplier,
            orbit_distance: self.orbit_distance,
        };
        component.sanitize();
        component
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsComponentData {
    pub collider_shape: SceneColliderShape,
    pub body_kind: PhysicsBodyKind,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collider_properties: Option<SceneColliderPropertiesData>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collider_inheritance: Option<SceneColliderPropertyInheritanceData>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rigid_body_properties: Option<SceneRigidBodyPropertiesData>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rigid_body_inheritance: Option<SceneRigidBodyPropertyInheritanceData>,
    #[serde(default)]
    pub joint: Option<ScenePhysicsJointData>,
    #[serde(default)]
    pub character_controller: Option<SceneCharacterControllerData>,
    #[serde(default)]
    pub character_input: Option<SceneCharacterControllerInputData>,
    #[serde(default)]
    pub ray_cast: Option<ScenePhysicsRayCastData>,
    #[serde(default)]
    pub point_projection: Option<ScenePhysicsPointProjectionData>,
    #[serde(default)]
    pub shape_cast: Option<ScenePhysicsShapeCastData>,
}

impl Default for PhysicsComponentData {
    fn default() -> Self {
        Self {
            collider_shape: SceneColliderShape::Cuboid,
            body_kind: PhysicsBodyKind::Fixed,
            collider_properties: None,
            collider_inheritance: None,
            rigid_body_properties: None,
            rigid_body_inheritance: None,
            joint: None,
            character_controller: None,
            character_input: None,
            ray_cast: None,
            point_projection: None,
            shape_cast: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhysicsBodyKind {
    Dynamic { mass: f32 },
    Kinematic { mode: SceneKinematicMode },
    Fixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneKinematicMode {
    PositionBased,
    VelocityBased,
}

impl From<KinematicMode> for SceneKinematicMode {
    fn from(mode: KinematicMode) -> Self {
        match mode {
            KinematicMode::PositionBased => SceneKinematicMode::PositionBased,
            KinematicMode::VelocityBased => SceneKinematicMode::VelocityBased,
        }
    }
}

impl SceneKinematicMode {
    fn to_component(self) -> KinematicMode {
        match self {
            SceneKinematicMode::PositionBased => KinematicMode::PositionBased,
            SceneKinematicMode::VelocityBased => KinematicMode::VelocityBased,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneMeshColliderLod {
    Lod0,
    Lod1,
    Lod2,
    Lowest,
    Specific(u8),
}

impl From<MeshColliderLod> for SceneMeshColliderLod {
    fn from(lod: MeshColliderLod) -> Self {
        match lod {
            MeshColliderLod::Lod0 => SceneMeshColliderLod::Lod0,
            MeshColliderLod::Lod1 => SceneMeshColliderLod::Lod1,
            MeshColliderLod::Lod2 => SceneMeshColliderLod::Lod2,
            MeshColliderLod::Lowest => SceneMeshColliderLod::Lowest,
            MeshColliderLod::Specific(index) => SceneMeshColliderLod::Specific(index),
        }
    }
}

impl SceneMeshColliderLod {
    fn to_component(self) -> MeshColliderLod {
        match self {
            SceneMeshColliderLod::Lod0 => MeshColliderLod::Lod0,
            SceneMeshColliderLod::Lod1 => MeshColliderLod::Lod1,
            SceneMeshColliderLod::Lod2 => MeshColliderLod::Lod2,
            SceneMeshColliderLod::Lowest => MeshColliderLod::Lowest,
            SceneMeshColliderLod::Specific(index) => MeshColliderLod::Specific(index),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneMeshColliderKind {
    TriMesh,
    ConvexHull,
}

impl From<MeshColliderKind> for SceneMeshColliderKind {
    fn from(kind: MeshColliderKind) -> Self {
        match kind {
            MeshColliderKind::TriMesh => SceneMeshColliderKind::TriMesh,
            MeshColliderKind::ConvexHull => SceneMeshColliderKind::ConvexHull,
        }
    }
}

impl SceneMeshColliderKind {
    fn to_component(self) -> MeshColliderKind {
        match self {
            SceneMeshColliderKind::TriMesh => MeshColliderKind::TriMesh,
            SceneMeshColliderKind::ConvexHull => MeshColliderKind::ConvexHull,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SceneColliderShape {
    Cuboid,
    Sphere,
    CapsuleY,
    CylinderY,
    ConeY,
    RoundCuboid {
        border_radius: f32,
    },
    Mesh {
        #[serde(default)]
        mesh_id: Option<usize>,
        lod: SceneMeshColliderLod,
        kind: SceneMeshColliderKind,
    },
}

impl From<ColliderShape> for SceneColliderShape {
    fn from(shape: ColliderShape) -> Self {
        match shape {
            ColliderShape::Cuboid => SceneColliderShape::Cuboid,
            ColliderShape::Sphere => SceneColliderShape::Sphere,
            ColliderShape::CapsuleY => SceneColliderShape::CapsuleY,
            ColliderShape::CylinderY => SceneColliderShape::CylinderY,
            ColliderShape::ConeY => SceneColliderShape::ConeY,
            ColliderShape::RoundCuboid { border_radius } => {
                SceneColliderShape::RoundCuboid { border_radius }
            }
            ColliderShape::Mesh { mesh_id, lod, kind } => SceneColliderShape::Mesh {
                mesh_id,
                lod: lod.into(),
                kind: kind.into(),
            },
        }
    }
}

impl SceneColliderShape {
    pub fn to_collider_shape(&self) -> ColliderShape {
        match self {
            SceneColliderShape::Cuboid => ColliderShape::Cuboid,
            SceneColliderShape::Sphere => ColliderShape::Sphere,
            SceneColliderShape::CapsuleY => ColliderShape::CapsuleY,
            SceneColliderShape::CylinderY => ColliderShape::CylinderY,
            SceneColliderShape::ConeY => ColliderShape::ConeY,
            SceneColliderShape::RoundCuboid { border_radius } => ColliderShape::RoundCuboid {
                border_radius: *border_radius,
            },
            SceneColliderShape::Mesh { mesh_id, lod, kind } => ColliderShape::Mesh {
                mesh_id: *mesh_id,
                lod: lod.to_component(),
                kind: kind.to_component(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenePhysicsCombineRule {
    Average,
    Min,
    Multiply,
    Max,
}

impl From<PhysicsCombineRule> for ScenePhysicsCombineRule {
    fn from(rule: PhysicsCombineRule) -> Self {
        match rule {
            PhysicsCombineRule::Average => ScenePhysicsCombineRule::Average,
            PhysicsCombineRule::Min => ScenePhysicsCombineRule::Min,
            PhysicsCombineRule::Multiply => ScenePhysicsCombineRule::Multiply,
            PhysicsCombineRule::Max => ScenePhysicsCombineRule::Max,
        }
    }
}

impl ScenePhysicsCombineRule {
    fn to_component(self) -> PhysicsCombineRule {
        match self {
            ScenePhysicsCombineRule::Average => PhysicsCombineRule::Average,
            ScenePhysicsCombineRule::Min => PhysicsCombineRule::Min,
            ScenePhysicsCombineRule::Multiply => PhysicsCombineRule::Multiply,
            ScenePhysicsCombineRule::Max => PhysicsCombineRule::Max,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneColliderPropertiesData {
    pub friction: f32,
    pub restitution: f32,
    pub density: f32,
    pub is_sensor: bool,
    pub enabled: bool,
    pub collision_memberships: u32,
    pub collision_filter: u32,
    pub solver_memberships: u32,
    pub solver_filter: u32,
    pub friction_combine_rule: ScenePhysicsCombineRule,
    pub restitution_combine_rule: ScenePhysicsCombineRule,
    pub translation_offset: [f32; 3],
    pub rotation_offset: [f32; 4],
}

impl Default for SceneColliderPropertiesData {
    fn default() -> Self {
        Self::from(ColliderProperties::default())
    }
}

impl From<ColliderProperties> for SceneColliderPropertiesData {
    fn from(value: ColliderProperties) -> Self {
        Self {
            friction: value.friction,
            restitution: value.restitution,
            density: value.density,
            is_sensor: value.is_sensor,
            enabled: value.enabled,
            collision_memberships: value.collision_memberships,
            collision_filter: value.collision_filter,
            solver_memberships: value.solver_memberships,
            solver_filter: value.solver_filter,
            friction_combine_rule: value.friction_combine_rule.into(),
            restitution_combine_rule: value.restitution_combine_rule.into(),
            translation_offset: value.translation_offset.to_array(),
            rotation_offset: value.rotation_offset.to_array(),
        }
    }
}

impl SceneColliderPropertiesData {
    fn to_component(&self) -> ColliderProperties {
        ColliderProperties {
            friction: self.friction,
            restitution: self.restitution,
            density: self.density,
            is_sensor: self.is_sensor,
            enabled: self.enabled,
            collision_memberships: self.collision_memberships,
            collision_filter: self.collision_filter,
            solver_memberships: self.solver_memberships,
            solver_filter: self.solver_filter,
            friction_combine_rule: self.friction_combine_rule.to_component(),
            restitution_combine_rule: self.restitution_combine_rule.to_component(),
            translation_offset: glam::Vec3::from_array(self.translation_offset),
            rotation_offset: glam::Quat::from_array(self.rotation_offset),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneColliderPropertyInheritanceData {
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

impl From<ColliderPropertyInheritance> for SceneColliderPropertyInheritanceData {
    fn from(value: ColliderPropertyInheritance) -> Self {
        Self {
            friction: value.friction,
            restitution: value.restitution,
            density: value.density,
            is_sensor: value.is_sensor,
            enabled: value.enabled,
            collision_memberships: value.collision_memberships,
            collision_filter: value.collision_filter,
            solver_memberships: value.solver_memberships,
            solver_filter: value.solver_filter,
            friction_combine_rule: value.friction_combine_rule,
            restitution_combine_rule: value.restitution_combine_rule,
            translation_offset: value.translation_offset,
            rotation_offset: value.rotation_offset,
        }
    }
}

impl SceneColliderPropertyInheritanceData {
    fn to_component(self) -> ColliderPropertyInheritance {
        ColliderPropertyInheritance {
            friction: self.friction,
            restitution: self.restitution,
            density: self.density,
            is_sensor: self.is_sensor,
            enabled: self.enabled,
            collision_memberships: self.collision_memberships,
            collision_filter: self.collision_filter,
            solver_memberships: self.solver_memberships,
            solver_filter: self.solver_filter,
            friction_combine_rule: self.friction_combine_rule,
            restitution_combine_rule: self.restitution_combine_rule,
            translation_offset: self.translation_offset,
            rotation_offset: self.rotation_offset,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneRigidBodyPropertiesData {
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
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
}

impl Default for SceneRigidBodyPropertiesData {
    fn default() -> Self {
        Self::from(RigidBodyProperties::default())
    }
}

impl From<RigidBodyProperties> for SceneRigidBodyPropertiesData {
    fn from(value: RigidBodyProperties) -> Self {
        Self {
            linear_damping: value.linear_damping,
            angular_damping: value.angular_damping,
            gravity_scale: value.gravity_scale,
            ccd_enabled: value.ccd_enabled,
            can_sleep: value.can_sleep,
            sleeping: value.sleeping,
            dominance_group: value.dominance_group,
            lock_translation_x: value.lock_translation_x,
            lock_translation_y: value.lock_translation_y,
            lock_translation_z: value.lock_translation_z,
            lock_rotation_x: value.lock_rotation_x,
            lock_rotation_y: value.lock_rotation_y,
            lock_rotation_z: value.lock_rotation_z,
            linear_velocity: value.linear_velocity.to_array(),
            angular_velocity: value.angular_velocity.to_array(),
        }
    }
}

impl SceneRigidBodyPropertiesData {
    fn to_component(&self) -> RigidBodyProperties {
        RigidBodyProperties {
            linear_damping: self.linear_damping,
            angular_damping: self.angular_damping,
            gravity_scale: self.gravity_scale,
            ccd_enabled: self.ccd_enabled,
            can_sleep: self.can_sleep,
            sleeping: self.sleeping,
            dominance_group: self.dominance_group,
            lock_translation_x: self.lock_translation_x,
            lock_translation_y: self.lock_translation_y,
            lock_translation_z: self.lock_translation_z,
            lock_rotation_x: self.lock_rotation_x,
            lock_rotation_y: self.lock_rotation_y,
            lock_rotation_z: self.lock_rotation_z,
            linear_velocity: glam::Vec3::from_array(self.linear_velocity),
            angular_velocity: glam::Vec3::from_array(self.angular_velocity),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneRigidBodyPropertyInheritanceData {
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

impl From<RigidBodyPropertyInheritance> for SceneRigidBodyPropertyInheritanceData {
    fn from(value: RigidBodyPropertyInheritance) -> Self {
        Self {
            linear_damping: value.linear_damping,
            angular_damping: value.angular_damping,
            gravity_scale: value.gravity_scale,
            ccd_enabled: value.ccd_enabled,
            can_sleep: value.can_sleep,
            sleeping: value.sleeping,
            dominance_group: value.dominance_group,
            lock_translation_x: value.lock_translation_x,
            lock_translation_y: value.lock_translation_y,
            lock_translation_z: value.lock_translation_z,
            lock_rotation_x: value.lock_rotation_x,
            lock_rotation_y: value.lock_rotation_y,
            lock_rotation_z: value.lock_rotation_z,
            linear_velocity: value.linear_velocity,
            angular_velocity: value.angular_velocity,
        }
    }
}

impl SceneRigidBodyPropertyInheritanceData {
    fn to_component(self) -> RigidBodyPropertyInheritance {
        RigidBodyPropertyInheritance {
            linear_damping: self.linear_damping,
            angular_damping: self.angular_damping,
            gravity_scale: self.gravity_scale,
            ccd_enabled: self.ccd_enabled,
            can_sleep: self.can_sleep,
            sleeping: self.sleeping,
            dominance_group: self.dominance_group,
            lock_translation_x: self.lock_translation_x,
            lock_translation_y: self.lock_translation_y,
            lock_translation_z: self.lock_translation_z,
            lock_rotation_x: self.lock_rotation_x,
            lock_rotation_y: self.lock_rotation_y,
            lock_rotation_z: self.lock_rotation_z,
            linear_velocity: self.linear_velocity,
            angular_velocity: self.angular_velocity,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePhysicsWorldDefaultsData {
    pub gravity: [f32; 3],
    pub collider_properties: SceneColliderPropertiesData,
    pub rigid_body_properties: SceneRigidBodyPropertiesData,
}

impl From<PhysicsWorldDefaults> for ScenePhysicsWorldDefaultsData {
    fn from(value: PhysicsWorldDefaults) -> Self {
        Self {
            gravity: value.gravity.to_array(),
            collider_properties: value.collider_properties.into(),
            rigid_body_properties: value.rigid_body_properties.into(),
        }
    }
}

impl ScenePhysicsWorldDefaultsData {
    fn to_component(&self) -> PhysicsWorldDefaults {
        PhysicsWorldDefaults {
            gravity: glam::Vec3::from_array(self.gravity),
            collider_properties: self.collider_properties.to_component(),
            rigid_body_properties: self.rigid_body_properties.to_component(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenePhysicsJointKind {
    Fixed,
    Spherical,
    Revolute,
    Prismatic,
}

impl From<PhysicsJointKind> for ScenePhysicsJointKind {
    fn from(value: PhysicsJointKind) -> Self {
        match value {
            PhysicsJointKind::Fixed => ScenePhysicsJointKind::Fixed,
            PhysicsJointKind::Spherical => ScenePhysicsJointKind::Spherical,
            PhysicsJointKind::Revolute => ScenePhysicsJointKind::Revolute,
            PhysicsJointKind::Prismatic => ScenePhysicsJointKind::Prismatic,
        }
    }
}

impl ScenePhysicsJointKind {
    fn to_component(self) -> PhysicsJointKind {
        match self {
            ScenePhysicsJointKind::Fixed => PhysicsJointKind::Fixed,
            ScenePhysicsJointKind::Spherical => PhysicsJointKind::Spherical,
            ScenePhysicsJointKind::Revolute => PhysicsJointKind::Revolute,
            ScenePhysicsJointKind::Prismatic => PhysicsJointKind::Prismatic,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneJointMotorData {
    pub enabled: bool,
    pub target_position: f32,
    pub target_velocity: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub max_force: f32,
}

impl Default for SceneJointMotorData {
    fn default() -> Self {
        Self::from(JointMotor::default())
    }
}

impl From<JointMotor> for SceneJointMotorData {
    fn from(value: JointMotor) -> Self {
        Self {
            enabled: value.enabled,
            target_position: value.target_position,
            target_velocity: value.target_velocity,
            stiffness: value.stiffness,
            damping: value.damping,
            max_force: value.max_force,
        }
    }
}

impl SceneJointMotorData {
    fn to_component(&self) -> JointMotor {
        JointMotor {
            enabled: self.enabled,
            target_position: self.target_position,
            target_velocity: self.target_velocity,
            stiffness: self.stiffness,
            damping: self.damping,
            max_force: self.max_force,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePhysicsJointData {
    #[serde(default)]
    pub target_name: Option<String>,
    pub kind: ScenePhysicsJointKind,
    pub contacts_enabled: bool,
    pub local_anchor1: [f32; 3],
    pub local_anchor2: [f32; 3],
    pub local_axis1: [f32; 3],
    pub local_axis2: [f32; 3],
    pub limit_enabled: bool,
    pub limits: [f32; 2],
    #[serde(default)]
    pub motor: SceneJointMotorData,
}

impl ScenePhysicsJointData {
    fn from_component(joint: PhysicsJoint, target_name: Option<String>) -> Self {
        Self {
            target_name,
            kind: joint.kind.into(),
            contacts_enabled: joint.contacts_enabled,
            local_anchor1: joint.local_anchor1.to_array(),
            local_anchor2: joint.local_anchor2.to_array(),
            local_axis1: joint.local_axis1.to_array(),
            local_axis2: joint.local_axis2.to_array(),
            limit_enabled: joint.limit_enabled,
            limits: joint.limits,
            motor: joint.motor.into(),
        }
    }

    fn to_component(&self) -> PhysicsJoint {
        PhysicsJoint {
            target: None,
            kind: self.kind.to_component(),
            contacts_enabled: self.contacts_enabled,
            local_anchor1: glam::Vec3::from_array(self.local_anchor1),
            local_anchor2: glam::Vec3::from_array(self.local_anchor2),
            local_axis1: glam::Vec3::from_array(self.local_axis1),
            local_axis2: glam::Vec3::from_array(self.local_axis2),
            limit_enabled: self.limit_enabled,
            limits: self.limits,
            motor: self.motor.to_component(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneCharacterControllerData {
    pub up: [f32; 3],
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

impl From<CharacterController> for SceneCharacterControllerData {
    fn from(value: CharacterController) -> Self {
        Self {
            up: value.up.to_array(),
            offset: value.offset,
            slide: value.slide,
            autostep_max_height: value.autostep_max_height,
            autostep_min_width: value.autostep_min_width,
            autostep_include_dynamic_bodies: value.autostep_include_dynamic_bodies,
            max_slope_climb_angle: value.max_slope_climb_angle,
            min_slope_slide_angle: value.min_slope_slide_angle,
            snap_to_ground: value.snap_to_ground,
            normal_nudge_factor: value.normal_nudge_factor,
            apply_impulses_to_dynamic_bodies: value.apply_impulses_to_dynamic_bodies,
            character_mass: value.character_mass,
        }
    }
}

impl SceneCharacterControllerData {
    fn to_component(&self) -> CharacterController {
        CharacterController {
            up: glam::Vec3::from_array(self.up),
            offset: self.offset,
            slide: self.slide,
            autostep_max_height: self.autostep_max_height,
            autostep_min_width: self.autostep_min_width,
            autostep_include_dynamic_bodies: self.autostep_include_dynamic_bodies,
            max_slope_climb_angle: self.max_slope_climb_angle,
            min_slope_slide_angle: self.min_slope_slide_angle,
            snap_to_ground: self.snap_to_ground,
            normal_nudge_factor: self.normal_nudge_factor,
            apply_impulses_to_dynamic_bodies: self.apply_impulses_to_dynamic_bodies,
            character_mass: self.character_mass,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneCharacterControllerInputData {
    pub desired_translation: [f32; 3],
}

impl From<CharacterControllerInput> for SceneCharacterControllerInputData {
    fn from(value: CharacterControllerInput) -> Self {
        Self {
            desired_translation: value.desired_translation.to_array(),
        }
    }
}

impl SceneCharacterControllerInputData {
    fn to_component(&self) -> CharacterControllerInput {
        CharacterControllerInput {
            desired_translation: glam::Vec3::from_array(self.desired_translation),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePhysicsQueryFilterData {
    pub flags: u32,
    pub groups_memberships: u32,
    pub groups_filter: u32,
    pub use_groups: bool,
}

impl Default for ScenePhysicsQueryFilterData {
    fn default() -> Self {
        Self::from(PhysicsQueryFilter::default())
    }
}

impl From<PhysicsQueryFilter> for ScenePhysicsQueryFilterData {
    fn from(value: PhysicsQueryFilter) -> Self {
        Self {
            flags: value.flags,
            groups_memberships: value.groups_memberships,
            groups_filter: value.groups_filter,
            use_groups: value.use_groups,
        }
    }
}

impl ScenePhysicsQueryFilterData {
    fn to_component(&self) -> PhysicsQueryFilter {
        PhysicsQueryFilter {
            flags: self.flags,
            groups_memberships: self.groups_memberships,
            groups_filter: self.groups_filter,
            use_groups: self.use_groups,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePhysicsRayCastData {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    pub max_toi: f32,
    pub solid: bool,
    #[serde(default)]
    pub filter: ScenePhysicsQueryFilterData,
    #[serde(default)]
    pub exclude_self: bool,
}

impl From<PhysicsRayCast> for ScenePhysicsRayCastData {
    fn from(value: PhysicsRayCast) -> Self {
        Self {
            origin: value.origin.to_array(),
            direction: value.direction.to_array(),
            max_toi: value.max_toi,
            solid: value.solid,
            filter: value.filter.into(),
            exclude_self: value.exclude_self,
        }
    }
}

impl ScenePhysicsRayCastData {
    fn to_component(&self) -> PhysicsRayCast {
        PhysicsRayCast {
            origin: glam::Vec3::from_array(self.origin),
            direction: glam::Vec3::from_array(self.direction),
            max_toi: self.max_toi,
            solid: self.solid,
            filter: self.filter.to_component(),
            exclude_self: self.exclude_self,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePhysicsPointProjectionData {
    pub point: [f32; 3],
    pub solid: bool,
    #[serde(default)]
    pub filter: ScenePhysicsQueryFilterData,
    #[serde(default)]
    pub exclude_self: bool,
}

impl From<PhysicsPointProjection> for ScenePhysicsPointProjectionData {
    fn from(value: PhysicsPointProjection) -> Self {
        Self {
            point: value.point.to_array(),
            solid: value.solid,
            filter: value.filter.into(),
            exclude_self: value.exclude_self,
        }
    }
}

impl ScenePhysicsPointProjectionData {
    fn to_component(&self) -> PhysicsPointProjection {
        PhysicsPointProjection {
            point: glam::Vec3::from_array(self.point),
            solid: self.solid,
            filter: self.filter.to_component(),
            exclude_self: self.exclude_self,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePhysicsShapeCastData {
    pub shape: SceneColliderShape,
    pub scale: [f32; 3],
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub velocity: [f32; 3],
    pub max_time_of_impact: f32,
    pub target_distance: f32,
    pub stop_at_penetration: bool,
    pub compute_impact_geometry_on_penetration: bool,
    #[serde(default)]
    pub filter: ScenePhysicsQueryFilterData,
    #[serde(default)]
    pub exclude_self: bool,
}

impl From<PhysicsShapeCast> for ScenePhysicsShapeCastData {
    fn from(value: PhysicsShapeCast) -> Self {
        Self {
            shape: value.shape.into(),
            scale: value.scale.to_array(),
            position: value.position.to_array(),
            rotation: value.rotation.to_array(),
            velocity: value.velocity.to_array(),
            max_time_of_impact: value.max_time_of_impact,
            target_distance: value.target_distance,
            stop_at_penetration: value.stop_at_penetration,
            compute_impact_geometry_on_penetration: value.compute_impact_geometry_on_penetration,
            filter: value.filter.into(),
            exclude_self: value.exclude_self,
        }
    }
}

impl ScenePhysicsShapeCastData {
    fn to_component(&self) -> PhysicsShapeCast {
        PhysicsShapeCast {
            shape: self.shape.to_collider_shape(),
            scale: glam::Vec3::from_array(self.scale),
            position: glam::Vec3::from_array(self.position),
            rotation: glam::Quat::from_array(self.rotation),
            velocity: glam::Vec3::from_array(self.velocity),
            max_time_of_impact: self.max_time_of_impact,
            target_distance: self.target_distance,
            stop_at_penetration: self.stop_at_penetration,
            compute_impact_geometry_on_penetration: self.compute_impact_geometry_on_penetration,
            filter: self.filter.to_component(),
            exclude_self: self.exclude_self,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEmitterData {
    pub path: Option<String>,
    #[serde(default)]
    pub streaming: bool,
    pub bus: helmer::audio::AudioBus,
    pub volume: f32,
    pub pitch: f32,
    pub looping: bool,
    pub spatial: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: f32,
    #[serde(default)]
    pub spatial_blend: f32,
    #[serde(default)]
    pub playback_state: helmer::audio::AudioPlaybackState,
    #[serde(default)]
    pub play_on_spawn: bool,
}

impl Default for AudioEmitterData {
    fn default() -> Self {
        let defaults = helmer::provided::components::AudioEmitter::default();
        Self {
            path: None,
            streaming: false,
            bus: defaults.bus,
            volume: defaults.volume,
            pitch: defaults.pitch,
            looping: defaults.looping,
            spatial: defaults.spatial,
            min_distance: defaults.min_distance,
            max_distance: defaults.max_distance,
            rolloff: defaults.rolloff,
            spatial_blend: defaults.spatial_blend,
            playback_state: defaults.playback_state,
            play_on_spawn: defaults.play_on_spawn,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioListenerData {
    pub enabled: bool,
}

impl Default for AudioListenerData {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshComponentData {
    pub source: MeshSource,
    pub material: Option<String>,
    pub casts_shadow: bool,
    pub visible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkinnedMeshComponentData {
    pub scene_path: String,
    #[serde(default)]
    pub scene_node_index: Option<usize>,
    pub casts_shadow: bool,
    pub visible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpriteSpaceData {
    Screen,
    World,
}

impl From<SpriteSpace> for SpriteSpaceData {
    fn from(value: SpriteSpace) -> Self {
        match value {
            SpriteSpace::Screen => Self::Screen,
            SpriteSpace::World => Self::World,
        }
    }
}

impl Default for SpriteSpaceData {
    fn default() -> Self {
        Self::World
    }
}

impl SpriteSpaceData {
    fn to_space(self) -> SpriteSpace {
        match self {
            Self::Screen => SpriteSpace::Screen,
            Self::World => SpriteSpace::World,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpriteBlendModeData {
    Alpha,
    Premultiplied,
    Additive,
}

impl From<SpriteBlendMode> for SpriteBlendModeData {
    fn from(value: SpriteBlendMode) -> Self {
        match value {
            SpriteBlendMode::Alpha => Self::Alpha,
            SpriteBlendMode::Premultiplied => Self::Premultiplied,
            SpriteBlendMode::Additive => Self::Additive,
        }
    }
}

impl Default for SpriteBlendModeData {
    fn default() -> Self {
        Self::Alpha
    }
}

impl SpriteBlendModeData {
    fn to_blend_mode(self) -> SpriteBlendMode {
        match self {
            Self::Alpha => SpriteBlendMode::Alpha,
            Self::Premultiplied => SpriteBlendMode::Premultiplied,
            Self::Additive => SpriteBlendMode::Additive,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpriteAnimationPlaybackData {
    Loop,
    Once,
    PingPong,
}

impl From<SpriteAnimationPlayback> for SpriteAnimationPlaybackData {
    fn from(value: SpriteAnimationPlayback) -> Self {
        match value {
            SpriteAnimationPlayback::Loop => Self::Loop,
            SpriteAnimationPlayback::Once => Self::Once,
            SpriteAnimationPlayback::PingPong => Self::PingPong,
        }
    }
}

impl Default for SpriteAnimationPlaybackData {
    fn default() -> Self {
        Self::Loop
    }
}

impl SpriteAnimationPlaybackData {
    fn to_playback(self) -> SpriteAnimationPlayback {
        match self {
            Self::Loop => SpriteAnimationPlayback::Loop,
            Self::Once => SpriteAnimationPlayback::Once,
            Self::PingPong => SpriteAnimationPlayback::PingPong,
        }
    }
}

fn sprite_sheet_default_columns() -> u32 {
    1
}

fn sprite_sheet_default_rows() -> u32 {
    1
}

fn sprite_sheet_default_fps() -> f32 {
    12.0
}

fn sprite_sheet_default_uv_inset() -> [f32; 2] {
    [0.0, 0.0]
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpriteSheetAnimationData {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "sprite_sheet_default_columns")]
    pub columns: u32,
    #[serde(default = "sprite_sheet_default_rows")]
    pub rows: u32,
    #[serde(default)]
    pub start_frame: u32,
    #[serde(default)]
    pub frame_count: u32,
    #[serde(default = "sprite_sheet_default_fps")]
    pub fps: f32,
    #[serde(default)]
    pub playback: SpriteAnimationPlaybackData,
    #[serde(default)]
    pub phase: f32,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub paused_frame: u32,
    #[serde(default)]
    pub flip_x: bool,
    #[serde(default)]
    pub flip_y: bool,
    #[serde(default = "sprite_sheet_default_uv_inset")]
    pub frame_uv_inset: [f32; 2],
}

impl Default for SpriteSheetAnimationData {
    fn default() -> Self {
        Self::from_sheet(SpriteSheetAnimation::default())
    }
}

impl SpriteSheetAnimationData {
    fn from_sheet(value: SpriteSheetAnimation) -> Self {
        Self {
            enabled: value.enabled,
            columns: value.columns,
            rows: value.rows,
            start_frame: value.start_frame,
            frame_count: value.frame_count,
            fps: value.fps,
            playback: value.playback.into(),
            phase: value.phase,
            paused: value.paused,
            paused_frame: value.paused_frame,
            flip_x: value.flip_x,
            flip_y: value.flip_y,
            frame_uv_inset: value.frame_uv_inset,
        }
    }

    fn to_sheet(self) -> SpriteSheetAnimation {
        SpriteSheetAnimation {
            enabled: self.enabled,
            columns: self.columns.max(1),
            rows: self.rows.max(1),
            start_frame: self.start_frame,
            frame_count: self.frame_count,
            fps: if self.fps.is_finite() { self.fps } else { 12.0 },
            playback: self.playback.to_playback(),
            phase: if self.phase.is_finite() {
                self.phase
            } else {
                0.0
            },
            paused: self.paused,
            paused_frame: self.paused_frame,
            flip_x: self.flip_x,
            flip_y: self.flip_y,
            frame_uv_inset: [
                if self.frame_uv_inset[0].is_finite() {
                    self.frame_uv_inset[0].clamp(0.0, 0.49)
                } else {
                    0.0
                },
                if self.frame_uv_inset[1].is_finite() {
                    self.frame_uv_inset[1].clamp(0.0, 0.49)
                } else {
                    0.0
                },
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpriteImageSequenceData {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub textures: Vec<String>,
    #[serde(default)]
    pub start_frame: u32,
    #[serde(default)]
    pub frame_count: u32,
    #[serde(default = "sprite_sheet_default_fps")]
    pub fps: f32,
    #[serde(default)]
    pub playback: SpriteAnimationPlaybackData,
    #[serde(default)]
    pub phase: f32,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub paused_frame: u32,
    #[serde(default)]
    pub flip_x: bool,
    #[serde(default)]
    pub flip_y: bool,
}

impl Default for SpriteImageSequenceData {
    fn default() -> Self {
        Self {
            enabled: false,
            textures: Vec::new(),
            start_frame: 0,
            frame_count: 0,
            fps: sprite_sheet_default_fps(),
            playback: SpriteAnimationPlaybackData::default(),
            phase: 0.0,
            paused: false,
            paused_frame: 0,
            flip_x: false,
            flip_y: false,
        }
    }
}

impl SpriteImageSequenceData {
    fn from_component(component: &SpriteImageSequence, texture_paths: Vec<String>) -> Self {
        Self {
            enabled: component.enabled,
            textures: texture_paths,
            start_frame: component.start_frame,
            frame_count: component.frame_count,
            fps: component.fps,
            playback: component.playback.into(),
            phase: component.phase,
            paused: component.paused,
            paused_frame: component.paused_frame,
            flip_x: component.flip_x,
            flip_y: component.flip_y,
        }
    }

    fn has_component_state(&self) -> bool {
        self.enabled
            || !self.textures.is_empty()
            || self.start_frame != 0
            || self.frame_count != 0
            || (self.fps - sprite_sheet_default_fps()).abs() > f32::EPSILON
            || self.playback != SpriteAnimationPlaybackData::default()
            || self.phase != 0.0
            || self.paused
            || self.paused_frame != 0
            || self.flip_x
            || self.flip_y
    }

    fn to_component(&self, texture_ids: Vec<usize>) -> SpriteImageSequence {
        SpriteImageSequence {
            enabled: self.enabled,
            texture_ids,
            start_frame: self.start_frame,
            frame_count: self.frame_count,
            fps: if self.fps.is_finite() { self.fps } else { 12.0 },
            playback: self.playback.to_playback(),
            phase: if self.phase.is_finite() {
                self.phase
            } else {
                0.0
            },
            paused: self.paused,
            paused_frame: self.paused_frame,
            flip_x: self.flip_x,
            flip_y: self.flip_y,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlignHData {
    Left,
    Center,
    Right,
}

impl From<TextAlignH> for TextAlignHData {
    fn from(value: TextAlignH) -> Self {
        match value {
            TextAlignH::Left => Self::Left,
            TextAlignH::Center => Self::Center,
            TextAlignH::Right => Self::Right,
        }
    }
}

impl Default for TextAlignHData {
    fn default() -> Self {
        Self::Left
    }
}

impl TextAlignHData {
    fn to_align(self) -> TextAlignH {
        match self {
            Self::Left => TextAlignH::Left,
            Self::Center => TextAlignH::Center,
            Self::Right => TextAlignH::Right,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlignVData {
    Top,
    Center,
    Bottom,
    Baseline,
}

impl From<TextAlignV> for TextAlignVData {
    fn from(value: TextAlignV) -> Self {
        match value {
            TextAlignV::Top => Self::Top,
            TextAlignV::Center => Self::Center,
            TextAlignV::Bottom => Self::Bottom,
            TextAlignV::Baseline => Self::Baseline,
        }
    }
}

impl Default for TextAlignVData {
    fn default() -> Self {
        Self::Baseline
    }
}

impl TextAlignVData {
    fn to_align(self) -> TextAlignV {
        match self {
            Self::Top => TextAlignV::Top,
            Self::Center => TextAlignV::Center,
            Self::Bottom => TextAlignV::Bottom,
            Self::Baseline => TextAlignV::Baseline,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextFontStyleData {
    Normal,
    Italic,
    Oblique,
}

impl From<TextFontStyle> for TextFontStyleData {
    fn from(value: TextFontStyle) -> Self {
        match value {
            TextFontStyle::Normal => Self::Normal,
            TextFontStyle::Italic => Self::Italic,
            TextFontStyle::Oblique => Self::Oblique,
        }
    }
}

impl Default for TextFontStyleData {
    fn default() -> Self {
        Self::Normal
    }
}

impl TextFontStyleData {
    fn to_style(self) -> TextFontStyle {
        match self {
            Self::Normal => TextFontStyle::Normal,
            Self::Italic => TextFontStyle::Italic,
            Self::Oblique => TextFontStyle::Oblique,
        }
    }
}

fn sprite_default_color() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}

fn sprite_default_uv_min() -> [f32; 2] {
    [0.0, 0.0]
}

fn sprite_default_uv_max() -> [f32; 2] {
    [1.0, 1.0]
}

fn sprite_default_pivot() -> [f32; 2] {
    [0.5, 0.5]
}

fn scene_component_visible_default() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpriteComponentData {
    #[serde(default)]
    pub texture: Option<String>,
    #[serde(default = "sprite_default_color")]
    pub color: [f32; 4],
    #[serde(default = "sprite_default_uv_min")]
    pub uv_min: [f32; 2],
    #[serde(default = "sprite_default_uv_max")]
    pub uv_max: [f32; 2],
    #[serde(default)]
    pub sheet_animation: SpriteSheetAnimationData,
    #[serde(default)]
    pub image_sequence: SpriteImageSequenceData,
    #[serde(default = "sprite_default_pivot")]
    pub pivot: [f32; 2],
    #[serde(default)]
    pub clip_rect: Option<[f32; 4]>,
    #[serde(default)]
    pub layer: f32,
    #[serde(default)]
    pub space: SpriteSpaceData,
    #[serde(default)]
    pub blend_mode: SpriteBlendModeData,
    #[serde(default)]
    pub billboard: bool,
    #[serde(default = "scene_component_visible_default")]
    pub visible: bool,
    #[serde(default)]
    pub pick_id: Option<u32>,
}

impl Default for SpriteComponentData {
    fn default() -> Self {
        Self {
            texture: None,
            color: sprite_default_color(),
            uv_min: sprite_default_uv_min(),
            uv_max: sprite_default_uv_max(),
            sheet_animation: SpriteSheetAnimationData::default(),
            image_sequence: SpriteImageSequenceData::default(),
            pivot: sprite_default_pivot(),
            clip_rect: None,
            layer: 0.0,
            space: SpriteSpaceData::default(),
            blend_mode: SpriteBlendModeData::default(),
            billboard: false,
            visible: true,
            pick_id: None,
        }
    }
}

fn text_default_font_size() -> f32 {
    16.0
}

fn text_default_font_weight() -> f32 {
    400.0
}

fn text_default_font_width() -> f32 {
    1.0
}

fn text_default_line_height_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Text2dComponentData {
    #[serde(default)]
    pub text: String,
    #[serde(default = "sprite_default_color")]
    pub color: [f32; 4],
    #[serde(default)]
    pub font_path: Option<String>,
    #[serde(default)]
    pub font_family: Option<String>,
    #[serde(default = "text_default_font_size")]
    pub font_size: f32,
    #[serde(default = "text_default_font_weight")]
    pub font_weight: f32,
    #[serde(default = "text_default_font_width")]
    pub font_width: f32,
    #[serde(default)]
    pub font_style: TextFontStyleData,
    #[serde(default = "text_default_line_height_scale")]
    pub line_height_scale: f32,
    #[serde(default)]
    pub letter_spacing: f32,
    #[serde(default)]
    pub word_spacing: f32,
    #[serde(default)]
    pub underline: bool,
    #[serde(default)]
    pub strikethrough: bool,
    #[serde(default)]
    pub max_width: Option<f32>,
    #[serde(default)]
    pub align_h: TextAlignHData,
    #[serde(default)]
    pub align_v: TextAlignVData,
    #[serde(default)]
    pub space: SpriteSpaceData,
    #[serde(default)]
    pub billboard: bool,
    #[serde(default)]
    pub blend_mode: SpriteBlendModeData,
    #[serde(default)]
    pub layer: f32,
    #[serde(default)]
    pub clip_rect: Option<[f32; 4]>,
    #[serde(default = "scene_component_visible_default")]
    pub visible: bool,
    #[serde(default)]
    pub pick_id: Option<u32>,
}

impl Default for Text2dComponentData {
    fn default() -> Self {
        Self {
            text: String::new(),
            color: sprite_default_color(),
            font_path: None,
            font_family: None,
            font_size: text_default_font_size(),
            font_weight: text_default_font_weight(),
            font_width: text_default_font_width(),
            font_style: TextFontStyleData::default(),
            line_height_scale: text_default_line_height_scale(),
            letter_spacing: 0.0,
            word_spacing: 0.0,
            underline: false,
            strikethrough: false,
            max_width: None,
            align_h: TextAlignHData::default(),
            align_v: TextAlignVData::default(),
            space: SpriteSpaceData::default(),
            billboard: false,
            blend_mode: SpriteBlendModeData::default(),
            layer: 0.0,
            clip_rect: None,
            visible: true,
            pick_id: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SplineModeData {
    Linear,
    CatmullRom,
    Bezier,
}

impl From<SplineMode> for SplineModeData {
    fn from(mode: SplineMode) -> Self {
        match mode {
            SplineMode::Linear => Self::Linear,
            SplineMode::CatmullRom => Self::CatmullRom,
            SplineMode::Bezier => Self::Bezier,
        }
    }
}

impl SplineModeData {
    pub fn to_mode(self) -> SplineMode {
        match self {
            Self::Linear => SplineMode::Linear,
            Self::CatmullRom => SplineMode::CatmullRom,
            Self::Bezier => SplineMode::Bezier,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineComponentData {
    pub points: Vec<[f32; 3]>,
    pub closed: bool,
    pub mode: SplineModeData,
    pub tension: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineFollowerData {
    pub spline_name: Option<String>,
    pub t: f32,
    pub speed: f32,
    pub looped: bool,
    pub follow_rotation: bool,
    pub up: [f32; 3],
    pub offset: [f32; 3],
    pub length_samples: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LookAtData {
    pub target_name: Option<String>,
    pub target_offset: [f32; 3],
    pub offset_in_target_space: bool,
    pub up: [f32; 3],
    pub rotation_smooth_time: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityFollowerData {
    pub target_name: Option<String>,
    pub position_offset: [f32; 3],
    pub offset_in_target_space: bool,
    pub follow_rotation: bool,
    pub position_smooth_time: f32,
    pub rotation_smooth_time: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AnimationComponentData {
    #[serde(default)]
    pub tracks: Vec<AnimationTrackData>,
    #[serde(default)]
    pub clips: Vec<AnimationClipData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseOverrideData {
    pub enabled: bool,
    pub locals: Vec<SerializedTransform>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnimationTrackData {
    Pose(PoseTrackData),
    Joint(JointTrackData),
    Transform(TransformTrackData),
    Camera(CameraTrackData),
    Light(LightTrackData),
    Spline(SplineTrackData),
    Clip(ClipTrackData),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseTrackData {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub weight: f32,
    pub additive: bool,
    pub translation_interpolation: TimelineInterpolation,
    pub rotation_interpolation: TimelineInterpolation,
    pub scale_interpolation: TimelineInterpolation,
    pub keys: Vec<PoseKeyData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseKeyData {
    pub id: u64,
    pub time: f32,
    pub pose: Vec<SerializedTransform>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointTrackData {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub joint_index: usize,
    #[serde(default)]
    pub joint_name: Option<String>,
    pub weight: f32,
    pub additive: bool,
    pub translation_interpolation: TimelineInterpolation,
    pub rotation_interpolation: TimelineInterpolation,
    pub scale_interpolation: TimelineInterpolation,
    pub keys: Vec<JointKeyData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointKeyData {
    pub id: u64,
    pub time: f32,
    pub transform: SerializedTransform,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformTrackData {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub translation_interpolation: TimelineInterpolation,
    pub rotation_interpolation: TimelineInterpolation,
    pub scale_interpolation: TimelineInterpolation,
    pub keys: Vec<TransformKeyData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformKeyData {
    pub id: u64,
    pub time: f32,
    pub transform: SerializedTransform,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraTrackData {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub interpolation: TimelineInterpolation,
    pub keys: Vec<CameraKeyData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraKeyData {
    pub id: u64,
    pub time: f32,
    pub camera: CameraComponentData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightTrackData {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub interpolation: TimelineInterpolation,
    pub keys: Vec<LightKeyData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightKeyData {
    pub id: u64,
    pub time: f32,
    pub light: LightComponentData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineTrackData {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub interpolation: TimelineInterpolation,
    pub keys: Vec<SplineKeyData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineKeyData {
    pub id: u64,
    pub time: f32,
    pub spline: SplineComponentData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipTrackData {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub weight: f32,
    pub additive: bool,
    pub segments: Vec<ClipSegmentData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipSegmentData {
    pub id: u64,
    pub start: f32,
    pub duration: f32,
    pub clip_name: String,
    pub speed: f32,
    pub looping: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimationClipData {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannelData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnimationChannelData {
    Translation {
        target: usize,
        #[serde(default)]
        target_name: Option<String>,
        interpolation: Interpolation,
        keyframes: Vec<Vec3KeyframeData>,
    },
    Rotation {
        target: usize,
        #[serde(default)]
        target_name: Option<String>,
        interpolation: Interpolation,
        keyframes: Vec<QuatKeyframeData>,
    },
    Scale {
        target: usize,
        #[serde(default)]
        target_name: Option<String>,
        interpolation: Interpolation,
        keyframes: Vec<Vec3KeyframeData>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vec3KeyframeData {
    pub time: f32,
    pub value: [f32; 3],
    pub in_tangent: Option<[f32; 3]>,
    pub out_tangent: Option<[f32; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuatKeyframeData {
    pub time: f32,
    pub value: [f32; 4],
    pub in_tangent: Option<[f32; 4]>,
    pub out_tangent: Option<[f32; 4]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneChildAnimationData {
    pub scene_path: String,
    pub scene_node_index: usize,
    pub animation: AnimationComponentData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneChildPoseOverrideData {
    pub scene_path: String,
    pub scene_node_index: usize,
    pub pose: PoseOverrideData,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LightKind {
    Directional,
    Point,
    Spot { angle: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightComponentData {
    pub kind: LightKind,
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraComponentData {
    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneAssetData {
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptComponentData {
    pub path: String,
    pub language: String,
    #[serde(default)]
    pub inspector_fields: Vec<ScriptInspectorField>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializedTransform {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl From<&Transform> for SerializedTransform {
    fn from(transform: &Transform) -> Self {
        Self {
            position: transform.position.to_array(),
            rotation: transform.rotation.to_array(),
            scale: transform.scale.to_array(),
        }
    }
}

impl SerializedTransform {
    pub fn to_transform(&self) -> Transform {
        Transform {
            position: glam::Vec3::from_array(self.position),
            rotation: glam::Quat::from_array(self.rotation),
            scale: glam::Vec3::from_array(self.scale),
        }
    }
}

impl From<&Spline> for SplineComponentData {
    fn from(spline: &Spline) -> Self {
        Self {
            points: spline.points.iter().map(|point| point.to_array()).collect(),
            closed: spline.closed,
            mode: spline.mode.into(),
            tension: spline.tension,
        }
    }
}

impl SplineComponentData {
    pub fn to_spline(&self) -> Spline {
        Spline {
            points: self
                .points
                .iter()
                .map(|point| glam::Vec3::from_array(*point))
                .collect(),
            closed: self.closed,
            mode: self.mode.to_mode(),
            tension: self.tension,
        }
    }
}

impl AnimationClipData {
    pub fn from_clip(clip: &AnimationClip, skeleton: Option<&Skeleton>) -> Self {
        let channels = clip
            .channels
            .iter()
            .map(|channel| match channel {
                AnimationChannel::Translation {
                    target,
                    interpolation,
                    keyframes,
                } => AnimationChannelData::Translation {
                    target: *target,
                    target_name: skeleton
                        .and_then(|skeleton| skeleton.joints.get(*target))
                        .map(|joint| joint.name.clone()),
                    interpolation: *interpolation,
                    keyframes: keyframes
                        .iter()
                        .map(|key| Vec3KeyframeData {
                            time: key.time,
                            value: key.value.to_array(),
                            in_tangent: key.in_tangent.map(|tangent| tangent.to_array()),
                            out_tangent: key.out_tangent.map(|tangent| tangent.to_array()),
                        })
                        .collect(),
                },
                AnimationChannel::Rotation {
                    target,
                    interpolation,
                    keyframes,
                } => AnimationChannelData::Rotation {
                    target: *target,
                    target_name: skeleton
                        .and_then(|skeleton| skeleton.joints.get(*target))
                        .map(|joint| joint.name.clone()),
                    interpolation: *interpolation,
                    keyframes: keyframes
                        .iter()
                        .map(|key| QuatKeyframeData {
                            time: key.time,
                            value: key.value.to_array(),
                            in_tangent: key.in_tangent.map(|tangent| tangent.to_array()),
                            out_tangent: key.out_tangent.map(|tangent| tangent.to_array()),
                        })
                        .collect(),
                },
                AnimationChannel::Scale {
                    target,
                    interpolation,
                    keyframes,
                } => AnimationChannelData::Scale {
                    target: *target,
                    target_name: skeleton
                        .and_then(|skeleton| skeleton.joints.get(*target))
                        .map(|joint| joint.name.clone()),
                    interpolation: *interpolation,
                    keyframes: keyframes
                        .iter()
                        .map(|key| Vec3KeyframeData {
                            time: key.time,
                            value: key.value.to_array(),
                            in_tangent: key.in_tangent.map(|tangent| tangent.to_array()),
                            out_tangent: key.out_tangent.map(|tangent| tangent.to_array()),
                        })
                        .collect(),
                },
            })
            .collect();
        Self {
            name: clip.name.clone(),
            duration: clip.duration,
            channels,
        }
    }

    pub fn to_clip(&self) -> AnimationClip {
        self.to_clip_for_skeleton(None)
    }

    pub fn to_clip_for_skeleton(&self, skeleton: Option<&Skeleton>) -> AnimationClip {
        let channels =
            self.channels
                .iter()
                .filter_map(|channel| match channel {
                    AnimationChannelData::Translation {
                        target,
                        target_name,
                        interpolation,
                        keyframes,
                    } => resolve_channel_target(skeleton, *target, target_name.as_ref()).map(
                        |target| AnimationChannel::Translation {
                            target,
                            interpolation: *interpolation,
                            keyframes: keyframes
                                .iter()
                                .map(|key| {
                                    let mut frame =
                                        Keyframe::new(key.time, glam::Vec3::from_array(key.value));
                                    frame.in_tangent = key
                                        .in_tangent
                                        .map(|tangent| glam::Vec3::from_array(tangent));
                                    frame.out_tangent = key
                                        .out_tangent
                                        .map(|tangent| glam::Vec3::from_array(tangent));
                                    frame
                                })
                                .collect(),
                        },
                    ),
                    AnimationChannelData::Rotation {
                        target,
                        target_name,
                        interpolation,
                        keyframes,
                    } => resolve_channel_target(skeleton, *target, target_name.as_ref()).map(
                        |target| AnimationChannel::Rotation {
                            target,
                            interpolation: *interpolation,
                            keyframes: keyframes
                                .iter()
                                .map(|key| {
                                    let mut frame =
                                        Keyframe::new(key.time, glam::Quat::from_array(key.value));
                                    frame.in_tangent = key
                                        .in_tangent
                                        .map(|tangent| glam::Quat::from_array(tangent));
                                    frame.out_tangent = key
                                        .out_tangent
                                        .map(|tangent| glam::Quat::from_array(tangent));
                                    frame
                                })
                                .collect(),
                        },
                    ),
                    AnimationChannelData::Scale {
                        target,
                        target_name,
                        interpolation,
                        keyframes,
                    } => resolve_channel_target(skeleton, *target, target_name.as_ref()).map(
                        |target| AnimationChannel::Scale {
                            target,
                            interpolation: *interpolation,
                            keyframes: keyframes
                                .iter()
                                .map(|key| {
                                    let mut frame =
                                        Keyframe::new(key.time, glam::Vec3::from_array(key.value));
                                    frame.in_tangent = key
                                        .in_tangent
                                        .map(|tangent| glam::Vec3::from_array(tangent));
                                    frame.out_tangent = key
                                        .out_tangent
                                        .map(|tangent| glam::Vec3::from_array(tangent));
                                    frame
                                })
                                .collect(),
                        },
                    ),
                })
                .collect();

        AnimationClip {
            name: self.name.clone(),
            duration: self.duration,
            channels,
        }
    }
}

fn resolve_channel_target(
    skeleton: Option<&Skeleton>,
    target: usize,
    target_name: Option<&String>,
) -> Option<usize> {
    if let Some(skeleton) = skeleton {
        if let Some(name) = target_name {
            if let Some(index) = skeleton.joints.iter().position(|joint| joint.name == *name) {
                return Some(index);
            }
        }
        if target < skeleton.joint_count() {
            return Some(target);
        }
        None
    } else {
        Some(target)
    }
}

fn resolve_joint_index(
    skeleton: Option<&Skeleton>,
    joint_index: usize,
    joint_name: Option<&String>,
) -> Option<usize> {
    if let Some(skeleton) = skeleton {
        if let Some(name) = joint_name {
            if let Some(index) = skeleton.joints.iter().position(|joint| joint.name == *name) {
                return Some(index);
            }
        }
        if joint_index < skeleton.joint_count() {
            return Some(joint_index);
        }
        None
    } else {
        Some(joint_index)
    }
}

fn animation_component_from_group(
    group: &TimelineTrackGroup,
    skeleton: Option<&Skeleton>,
) -> Option<AnimationComponentData> {
    let tracks = group
        .tracks
        .iter()
        .map(|track| animation_track_to_data(track, skeleton))
        .collect::<Vec<_>>();
    let clips = group
        .custom_clips
        .iter()
        .map(|clip| AnimationClipData::from_clip(clip, skeleton))
        .collect::<Vec<_>>();
    if tracks.is_empty() && clips.is_empty() {
        None
    } else {
        Some(AnimationComponentData { tracks, clips })
    }
}

fn animation_track_to_data(
    track: &TimelineTrack,
    skeleton: Option<&Skeleton>,
) -> AnimationTrackData {
    match track {
        TimelineTrack::Pose(track) => AnimationTrackData::Pose(PoseTrackData {
            id: track.id,
            name: track.name.clone(),
            enabled: track.enabled,
            weight: track.weight,
            additive: track.additive,
            translation_interpolation: track.translation_interpolation,
            rotation_interpolation: track.rotation_interpolation,
            scale_interpolation: track.scale_interpolation,
            keys: track
                .keys
                .iter()
                .map(|key| PoseKeyData {
                    id: key.id,
                    time: key.time,
                    pose: key
                        .pose
                        .locals
                        .iter()
                        .map(SerializedTransform::from)
                        .collect(),
                })
                .collect(),
        }),
        TimelineTrack::Joint(track) => AnimationTrackData::Joint(JointTrackData {
            id: track.id,
            name: track.name.clone(),
            enabled: track.enabled,
            joint_index: track.joint_index,
            joint_name: skeleton
                .and_then(|skeleton| skeleton.joints.get(track.joint_index))
                .map(|joint| joint.name.clone()),
            weight: track.weight,
            additive: track.additive,
            translation_interpolation: track.translation_interpolation,
            rotation_interpolation: track.rotation_interpolation,
            scale_interpolation: track.scale_interpolation,
            keys: track
                .keys
                .iter()
                .map(|key| JointKeyData {
                    id: key.id,
                    time: key.time,
                    transform: SerializedTransform::from(&key.transform),
                })
                .collect(),
        }),
        TimelineTrack::Transform(track) => AnimationTrackData::Transform(TransformTrackData {
            id: track.id,
            name: track.name.clone(),
            enabled: track.enabled,
            translation_interpolation: track.translation_interpolation,
            rotation_interpolation: track.rotation_interpolation,
            scale_interpolation: track.scale_interpolation,
            keys: track
                .keys
                .iter()
                .map(|key| TransformKeyData {
                    id: key.id,
                    time: key.time,
                    transform: SerializedTransform::from(&key.transform),
                })
                .collect(),
        }),
        TimelineTrack::Camera(track) => AnimationTrackData::Camera(CameraTrackData {
            id: track.id,
            name: track.name.clone(),
            enabled: track.enabled,
            interpolation: track.interpolation,
            keys: track
                .keys
                .iter()
                .map(|key| CameraKeyData {
                    id: key.id,
                    time: key.time,
                    camera: CameraComponentData {
                        fov_y_rad: key.camera.fov_y_rad,
                        aspect_ratio: key.camera.aspect_ratio,
                        near_plane: key.camera.near_plane,
                        far_plane: key.camera.far_plane,
                        active: false,
                    },
                })
                .collect(),
        }),
        TimelineTrack::Light(track) => AnimationTrackData::Light(LightTrackData {
            id: track.id,
            name: track.name.clone(),
            enabled: track.enabled,
            interpolation: track.interpolation,
            keys: track
                .keys
                .iter()
                .map(|key| LightKeyData {
                    id: key.id,
                    time: key.time,
                    light: LightComponentData {
                        kind: match key.light.light_type {
                            LightType::Directional => LightKind::Directional,
                            LightType::Point => LightKind::Point,
                            LightType::Spot { angle } => LightKind::Spot { angle },
                        },
                        color: [key.light.color.x, key.light.color.y, key.light.color.z],
                        intensity: key.light.intensity,
                    },
                })
                .collect(),
        }),
        TimelineTrack::Spline(track) => AnimationTrackData::Spline(SplineTrackData {
            id: track.id,
            name: track.name.clone(),
            enabled: track.enabled,
            interpolation: track.interpolation,
            keys: track
                .keys
                .iter()
                .map(|key| SplineKeyData {
                    id: key.id,
                    time: key.time,
                    spline: SplineComponentData::from(&key.spline),
                })
                .collect(),
        }),
        TimelineTrack::Clip(track) => AnimationTrackData::Clip(ClipTrackData {
            id: track.id,
            name: track.name.clone(),
            enabled: track.enabled,
            weight: track.weight,
            additive: track.additive,
            segments: track
                .segments
                .iter()
                .map(|segment| ClipSegmentData {
                    id: segment.id,
                    start: segment.start,
                    duration: segment.duration,
                    clip_name: segment.clip_name.clone(),
                    speed: segment.speed,
                    looping: segment.looping,
                })
                .collect(),
        }),
    }
}

fn pose_to_serialized(pose: &Pose) -> Vec<SerializedTransform> {
    pose.locals.iter().map(SerializedTransform::from).collect()
}

pub(crate) fn pose_from_serialized(
    transforms: &[SerializedTransform],
    skeleton: Option<&Skeleton>,
) -> Pose {
    if let Some(skeleton) = skeleton {
        let mut locals = Vec::with_capacity(skeleton.joint_count());
        for index in 0..skeleton.joint_count() {
            let local = transforms
                .get(index)
                .map(SerializedTransform::to_transform)
                .unwrap_or_else(|| skeleton.joints[index].bind_transform);
            locals.push(local);
        }
        Pose { locals }
    } else {
        Pose {
            locals: transforms
                .iter()
                .map(SerializedTransform::to_transform)
                .collect(),
        }
    }
}

fn animation_track_from_data(
    data: &AnimationTrackData,
    skeleton: Option<&Skeleton>,
) -> Option<TimelineTrack> {
    match data {
        AnimationTrackData::Pose(track) => {
            let keys = track
                .keys
                .iter()
                .map(|key| PoseKey {
                    id: key.id,
                    time: key.time,
                    pose: pose_from_serialized(&key.pose, skeleton),
                })
                .collect();
            Some(TimelineTrack::Pose(PoseTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                weight: track.weight,
                additive: track.additive,
                translation_interpolation: track.translation_interpolation,
                rotation_interpolation: track.rotation_interpolation,
                scale_interpolation: track.scale_interpolation,
                keys,
            }))
        }
        AnimationTrackData::Joint(track) => {
            let joint_index =
                resolve_joint_index(skeleton, track.joint_index, track.joint_name.as_ref())?;
            let keys = track
                .keys
                .iter()
                .map(|key| JointKey {
                    id: key.id,
                    time: key.time,
                    transform: key.transform.to_transform(),
                })
                .collect();
            Some(TimelineTrack::Joint(JointTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                joint_index,
                weight: track.weight,
                additive: track.additive,
                translation_interpolation: track.translation_interpolation,
                rotation_interpolation: track.rotation_interpolation,
                scale_interpolation: track.scale_interpolation,
                keys,
            }))
        }
        AnimationTrackData::Transform(track) => {
            let keys = track
                .keys
                .iter()
                .map(|key| TransformKey {
                    id: key.id,
                    time: key.time,
                    transform: key.transform.to_transform(),
                })
                .collect();
            Some(TimelineTrack::Transform(TransformTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                translation_interpolation: track.translation_interpolation,
                rotation_interpolation: track.rotation_interpolation,
                scale_interpolation: track.scale_interpolation,
                keys,
            }))
        }
        AnimationTrackData::Camera(track) => {
            let keys = track
                .keys
                .iter()
                .map(|key| CameraKey {
                    id: key.id,
                    time: key.time,
                    camera: Camera {
                        fov_y_rad: key.camera.fov_y_rad,
                        aspect_ratio: key.camera.aspect_ratio,
                        near_plane: key.camera.near_plane,
                        far_plane: key.camera.far_plane,
                    },
                })
                .collect();
            Some(TimelineTrack::Camera(CameraTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                interpolation: track.interpolation,
                keys,
            }))
        }
        AnimationTrackData::Light(track) => {
            let keys = track
                .keys
                .iter()
                .map(|key| LightKey {
                    id: key.id,
                    time: key.time,
                    light: Light {
                        light_type: match key.light.kind {
                            LightKind::Directional => LightType::Directional,
                            LightKind::Point => LightType::Point,
                            LightKind::Spot { angle } => LightType::Spot { angle },
                        },
                        color: glam::Vec3::new(
                            key.light.color[0],
                            key.light.color[1],
                            key.light.color[2],
                        ),
                        intensity: key.light.intensity,
                    },
                })
                .collect();
            Some(TimelineTrack::Light(LightTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                interpolation: track.interpolation,
                keys,
            }))
        }
        AnimationTrackData::Spline(track) => {
            let keys = track
                .keys
                .iter()
                .map(|key| SplineKey {
                    id: key.id,
                    time: key.time,
                    spline: key.spline.to_spline(),
                })
                .collect();
            Some(TimelineTrack::Spline(SplineTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                interpolation: track.interpolation,
                keys,
            }))
        }
        AnimationTrackData::Clip(track) => {
            let segments = track
                .segments
                .iter()
                .map(|segment| ClipSegment {
                    id: segment.id,
                    start: segment.start,
                    duration: segment.duration,
                    clip_name: segment.clip_name.clone(),
                    speed: segment.speed,
                    looping: segment.looping,
                })
                .collect();
            Some(TimelineTrack::Clip(ClipTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                weight: track.weight,
                additive: track.additive,
                segments,
            }))
        }
    }
}

fn remap_track_ids(track: &mut TimelineTrack, alloc_id: &mut impl FnMut() -> u64) {
    match track {
        TimelineTrack::Pose(track) => {
            track.id = alloc_id();
            for key in &mut track.keys {
                key.id = alloc_id();
            }
        }
        TimelineTrack::Joint(track) => {
            track.id = alloc_id();
            for key in &mut track.keys {
                key.id = alloc_id();
            }
        }
        TimelineTrack::Transform(track) => {
            track.id = alloc_id();
            for key in &mut track.keys {
                key.id = alloc_id();
            }
        }
        TimelineTrack::Camera(track) => {
            track.id = alloc_id();
            for key in &mut track.keys {
                key.id = alloc_id();
            }
        }
        TimelineTrack::Light(track) => {
            track.id = alloc_id();
            for key in &mut track.keys {
                key.id = alloc_id();
            }
        }
        TimelineTrack::Spline(track) => {
            track.id = alloc_id();
            for key in &mut track.keys {
                key.id = alloc_id();
            }
        }
        TimelineTrack::Clip(track) => {
            track.id = alloc_id();
            for segment in &mut track.segments {
                segment.id = alloc_id();
            }
        }
    }
}

fn update_timeline_next_id(timeline: &mut EditorTimelineState) {
    let mut max_id = timeline.next_id;
    for group in &timeline.groups {
        for track in &group.tracks {
            max_id = max_id.max(track.id());
            match track {
                TimelineTrack::Pose(track) => {
                    for key in &track.keys {
                        max_id = max_id.max(key.id);
                    }
                }
                TimelineTrack::Joint(track) => {
                    for key in &track.keys {
                        max_id = max_id.max(key.id);
                    }
                }
                TimelineTrack::Transform(track) => {
                    for key in &track.keys {
                        max_id = max_id.max(key.id);
                    }
                }
                TimelineTrack::Camera(track) => {
                    for key in &track.keys {
                        max_id = max_id.max(key.id);
                    }
                }
                TimelineTrack::Light(track) => {
                    for key in &track.keys {
                        max_id = max_id.max(key.id);
                    }
                }
                TimelineTrack::Spline(track) => {
                    for key in &track.keys {
                        max_id = max_id.max(key.id);
                    }
                }
                TimelineTrack::Clip(track) => {
                    for segment in &track.segments {
                        max_id = max_id.max(segment.id);
                    }
                }
            }
        }
    }
    timeline.next_id = max_id.saturating_add(1);
}

pub(crate) fn apply_custom_clips_to_animator(animator: &mut BevyAnimator, clips: &[AnimationClip]) {
    if clips.is_empty() {
        return;
    }
    for layer in animator.0.layers.iter_mut() {
        let mut library = (*layer.graph.library).clone();
        for clip in clips {
            library.upsert_clip(clip.clone());
        }
        layer.graph.library = std::sync::Arc::new(library);
    }
}

pub(crate) fn apply_animation_data_to_timeline(
    timeline: &mut EditorTimelineState,
    entity: Entity,
    entity_name: String,
    data: &AnimationComponentData,
    skeleton: Option<&Skeleton>,
) {
    let group = timeline.ensure_group(entity.to_bits(), entity_name);
    group.tracks = data
        .tracks
        .iter()
        .filter_map(|track| animation_track_from_data(track, skeleton))
        .collect();
    group.custom_clips = data
        .clips
        .iter()
        .map(|clip| clip.to_clip_for_skeleton(skeleton))
        .collect();
    timeline.apply_requested = true;
    update_timeline_next_id(timeline);
}

pub(crate) fn animation_asset_from_group(
    group: &TimelineTrackGroup,
    skeleton: Option<&Skeleton>,
) -> AnimationAssetDocument {
    let tracks = group
        .tracks
        .iter()
        .map(|track| animation_track_to_data(track, skeleton))
        .collect();
    let clips = group
        .custom_clips
        .iter()
        .map(|clip| AnimationClipData::from_clip(clip, skeleton))
        .collect();
    AnimationAssetDocument {
        version: 1,
        name: group.name.clone(),
        tracks,
        clips,
    }
}

pub(crate) fn merge_animation_asset_into_timeline(
    timeline: &mut EditorTimelineState,
    entity: Entity,
    entity_name: String,
    asset: &AnimationAssetDocument,
    skeleton: Option<&Skeleton>,
) -> Vec<AnimationClip> {
    let group_index = timeline.ensure_group_index(entity.to_bits(), entity_name);
    let mut next_id = timeline.next_id;
    let mut alloc_id = || {
        let id = next_id;
        next_id = next_id.saturating_add(1);
        id
    };

    let mut clips = Vec::new();
    {
        let group = &mut timeline.groups[group_index];
        for track_data in &asset.tracks {
            if let Some(mut track) = animation_track_from_data(track_data, skeleton) {
                remap_track_ids(&mut track, &mut alloc_id);
                group.tracks.push(track);
            }
        }
        for clip_data in &asset.clips {
            let clip = clip_data.to_clip_for_skeleton(skeleton);
            if let Some(existing) = group
                .custom_clips
                .iter_mut()
                .find(|existing| existing.name == clip.name)
            {
                *existing = clip.clone();
            } else {
                group.custom_clips.push(clip.clone());
            }
            clips.push(clip);
        }
    }

    timeline.next_id = next_id;
    timeline.apply_requested = true;
    update_timeline_next_id(timeline);
    clips
}

pub fn reset_editor_scene(world: &mut World) {
    let entities: Vec<Entity> = world
        .query_filtered::<Entity, With<EditorEntity>>()
        .iter(world)
        .collect();

    for entity in entities {
        world.despawn(entity);
    }

    let scene_children: Vec<Entity> = world
        .query_filtered::<Entity, With<SceneChild>>()
        .iter(world)
        .collect();
    for entity in scene_children {
        world.despawn(entity);
    }

    if let Some(mut spawned) = world.get_resource_mut::<SceneSpawnedChildren>() {
        spawned.spawned_scenes.clear();
    }

    if let Some(mut timeline) = world.get_resource_mut::<EditorTimelineState>() {
        timeline.groups.clear();
        timeline.selected.clear();
        timeline.selection_drag = None;
        timeline.selection_drag_pending = None;
        timeline.pending_clip_expand = None;
        timeline.apply_requested = true;
        timeline.current_time = 0.0;
        timeline.next_id = 1;
    }
    if let Some(mut pending) = world.get_resource_mut::<PendingSceneChildAnimations>() {
        pending.entries.clear();
    }
    if let Some(mut pending) = world.get_resource_mut::<PendingSceneChildPoseOverrides>() {
        pending.entries.clear();
    }
}

pub fn reset_scene_root_instance(world: &mut World, scene_root: Entity) {
    let mut children_to_despawn: HashSet<Entity> = HashSet::new();

    if let Some(mut spawned) = world.get_resource_mut::<SceneSpawnedChildren>() {
        if let Some(children) = spawned.spawned_scenes.remove(&scene_root) {
            children_to_despawn.extend(children);
        }
    }

    let mut child_query = world.query::<(Entity, &SceneChild)>();
    for (child_entity, child_link) in child_query.iter(world) {
        if child_link.scene_root == scene_root {
            children_to_despawn.insert(child_entity);
        }
    }

    for child_entity in children_to_despawn {
        if child_entity == scene_root {
            continue;
        }
        let _ = world.despawn(child_entity);
    }

    if world.get_entity(scene_root).is_ok() {
        world.entity_mut(scene_root).remove::<SpawnedScene>();
    }
}

pub fn spawn_default_camera(world: &mut World) -> Entity {
    world
        .spawn((
            EditorEntity,
            BevyTransform::default(),
            BevyCamera::default(),
            EditorPlayCamera,
            Name::new("Scene Camera"),
        ))
        .id()
}

pub fn spawn_default_light(world: &mut World) -> Entity {
    world
        .spawn((
            EditorEntity,
            BevyWrapper(Transform {
                position: glam::Vec3::new(0.0, 4.0, 0.0),
                rotation: glam::Quat::from_euler(
                    glam::EulerRot::YXZ,
                    20.0f32.to_radians(),
                    -50.0f32.to_radians(),
                    20.0f32.to_radians(),
                ),
                scale: glam::Vec3::ONE,
            }),
            BevyWrapper(Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0)),
            Name::new("Directional Light"),
        ))
        .id()
}

pub fn ensure_active_camera(world: &mut World) {
    let mut active_found = false;
    for _ in world
        .query::<(&BevyCamera, &EditorPlayCamera)>()
        .iter(world)
    {
        active_found = true;
        break;
    }

    if active_found {
        return;
    }

    if let Some((entity, _)) = world
        .query_filtered::<(Entity, &BevyCamera), Without<EditorViewportCamera>>()
        .iter(world)
        .next()
    {
        world.entity_mut(entity).insert(EditorPlayCamera);
    }
}

pub fn serialize_scene(world: &mut World, project: &EditorProject) -> (SceneDocument, Vec<Entity>) {
    let root = project.root.as_deref();
    let timeline_groups = world
        .get_resource::<EditorTimelineState>()
        .map(|timeline| timeline.groups.clone());

    let mut entities = Vec::new();
    let mut entity_order = Vec::new();
    let mut saved_entities = HashSet::new();
    let mut pending_relations: Vec<Option<Entity>> = Vec::new();
    let mut pending_scene_children: Vec<Option<(Entity, usize)>> = Vec::new();
    let mut query = world.query::<(
        Entity,
        Option<&Name>,
        Option<&BevyTransform>,
        Option<&BevyMeshRenderer>,
        Option<&EditorMesh>,
        Option<&BevySkinnedMeshRenderer>,
        Option<&EditorSkinnedMesh>,
        Option<&BevyLight>,
        Option<&BevyCamera>,
        Option<&EditorPlayCamera>,
        Option<&EditorEntity>,
        Option<&SceneChild>,
        Option<&EntityParent>,
    )>();

    for (
        entity,
        name,
        transform,
        mesh_renderer,
        editor_mesh,
        skinned_renderer,
        editor_skinned,
        light,
        camera,
        active_camera,
        editor_entity,
        scene_child_meta,
        parent_relation,
    ) in query.iter(world)
    {
        if editor_entity.is_none() && scene_child_meta.is_none() {
            continue;
        }

        let audio_emitter = world.get::<BevyAudioEmitter>(entity);
        let editor_audio = world.get::<EditorAudio>(entity);
        let audio_listener = world.get::<BevyAudioListener>(entity);
        let scene_root = world.get::<SceneRoot>(entity);
        let scene_asset = world.get::<SceneAssetPath>(entity);
        let script = world.get::<ScriptComponent>(entity);
        let transform = transform.map(|t| t.0).unwrap_or_default();
        let serialized_transform = SerializedTransform::from(&transform);

        let mesh = if let (Some(mesh_renderer), Some(editor_mesh)) = (mesh_renderer, editor_mesh) {
            Some(MeshComponentData {
                source: editor_mesh.source.clone(),
                material: editor_mesh
                    .material_path
                    .as_ref()
                    .map(|path| normalize_path(path, root)),
                casts_shadow: mesh_renderer.0.casts_shadow,
                visible: mesh_renderer.0.visible,
            })
        } else {
            None
        };

        let skinned = editor_skinned.and_then(|skinned| {
            let path = skinned.scene_path.as_ref()?;
            if path.trim().is_empty() {
                return None;
            }
            let (casts_shadow, visible) = skinned_renderer
                .map(|renderer| (renderer.0.casts_shadow, renderer.0.visible))
                .unwrap_or((skinned.casts_shadow, skinned.visible));
            Some(SkinnedMeshComponentData {
                scene_path: normalize_path(path, root),
                scene_node_index: skinned.node_index,
                casts_shadow,
                visible,
            })
        });
        let sprite = world.get::<BevySpriteRenderer>(entity).map(|sprite| {
            let editor_sprite = world.get::<EditorSprite>(entity).cloned();
            let texture = editor_sprite
                .as_ref()
                .and_then(|editor| editor.texture_path.clone())
                .and_then(|path| {
                    let path = path.trim().to_string();
                    if path.is_empty() {
                        None
                    } else {
                        Some(normalize_path(&path, root))
                    }
                });
            let image_sequence = world
                .get::<BevySpriteImageSequence>(entity)
                .map(|sequence| {
                    let texture_paths = editor_sprite
                        .as_ref()
                        .map(|editor| editor.sequence_texture_paths.clone())
                        .unwrap_or_default()
                        .into_iter()
                        .filter_map(|path| {
                            let path = path.trim().to_string();
                            if path.is_empty() {
                                None
                            } else {
                                Some(normalize_path(&path, root))
                            }
                        })
                        .collect::<Vec<_>>();
                    SpriteImageSequenceData::from_component(&sequence.0, texture_paths)
                })
                .unwrap_or_default();
            SpriteComponentData {
                texture,
                color: sprite.0.color,
                uv_min: sprite.0.uv_min,
                uv_max: sprite.0.uv_max,
                sheet_animation: SpriteSheetAnimationData::from_sheet(sprite.0.sheet_animation),
                image_sequence,
                pivot: sprite.0.pivot,
                clip_rect: sprite.0.clip_rect,
                layer: sprite.0.layer,
                space: sprite.0.space.into(),
                blend_mode: sprite.0.blend_mode.into(),
                billboard: sprite.0.billboard,
                visible: sprite.0.visible,
                pick_id: sprite.0.pick_id,
            }
        });
        let text_2d = world
            .get::<BevyText2d>(entity)
            .map(|text| Text2dComponentData {
                text: text.0.text.clone(),
                color: text.0.color,
                font_path: text.0.font_path.as_ref().and_then(|path| {
                    let path = path.trim();
                    if path.is_empty() {
                        None
                    } else {
                        Some(normalize_path(path, root))
                    }
                }),
                font_family: text.0.font_family.clone(),
                font_size: text.0.font_size,
                font_weight: text.0.font_weight,
                font_width: text.0.font_width,
                font_style: text.0.font_style.into(),
                line_height_scale: text.0.line_height_scale,
                letter_spacing: text.0.letter_spacing,
                word_spacing: text.0.word_spacing,
                underline: text.0.underline,
                strikethrough: text.0.strikethrough,
                max_width: text.0.max_width,
                align_h: text.0.align_h.into(),
                align_v: text.0.align_v.into(),
                space: text.0.space.into(),
                billboard: text.0.billboard,
                blend_mode: text.0.blend_mode.into(),
                layer: text.0.layer,
                clip_rect: text.0.clip_rect,
                visible: text.0.visible,
                pick_id: text.0.pick_id,
            });
        let scene_child_renderer =
            if scene_child_meta.is_some() && mesh.is_none() && skinned.is_none() {
                if let Some(renderer) = skinned_renderer {
                    Some(SceneChildRendererData {
                        kind: SceneChildRendererKind::Skinned,
                        casts_shadow: renderer.0.casts_shadow,
                        visible: renderer.0.visible,
                    })
                } else if let Some(renderer) = mesh_renderer {
                    Some(SceneChildRendererData {
                        kind: SceneChildRendererKind::Mesh,
                        casts_shadow: renderer.0.casts_shadow,
                        visible: renderer.0.visible,
                    })
                } else {
                    Some(SceneChildRendererData {
                        kind: SceneChildRendererKind::None,
                        casts_shadow: true,
                        visible: true,
                    })
                }
            } else {
                None
            };

        let light = light.map(|light| LightComponentData {
            kind: match light.0.light_type {
                LightType::Directional => LightKind::Directional,
                LightType::Point => LightKind::Point,
                LightType::Spot { angle } => LightKind::Spot { angle },
            },
            color: [light.0.color.x, light.0.color.y, light.0.color.z],
            intensity: light.0.intensity,
        });

        let camera = camera.map(|camera| CameraComponentData {
            fov_y_rad: camera.0.fov_y_rad,
            aspect_ratio: camera.0.aspect_ratio,
            near_plane: camera.0.near_plane,
            far_plane: camera.0.far_plane,
            active: active_camera.is_some(),
        });

        let scene = if scene_root.is_some() {
            scene_asset.map(|asset| SceneAssetData {
                path: normalize_path(asset.path.to_string_lossy().as_ref(), root),
            })
        } else {
            None
        };

        let scripts = script
            .map(|scripts| {
                scripts
                    .scripts
                    .iter()
                    .filter_map(|script| {
                        let path = script.path.as_ref()?;
                        Some(ScriptComponentData {
                            path: normalize_path(path.to_string_lossy().as_ref(), root),
                            language: script.language.clone(),
                            inspector_fields: script.inspector_fields.clone(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let dynamic = world
            .get::<DynamicComponents>(entity)
            .map(|components| components.components.clone())
            .unwrap_or_default();
        let spline = world
            .get::<BevySpline>(entity)
            .map(|spline| SplineComponentData {
                points: spline
                    .0
                    .points
                    .iter()
                    .map(|point| point.to_array())
                    .collect(),
                closed: spline.0.closed,
                mode: spline.0.mode.into(),
                tension: spline.0.tension,
            });
        let spline_follower =
            world
                .get::<BevySplineFollower>(entity)
                .map(|follower| SplineFollowerData {
                    spline_name: follower
                        .0
                        .spline_entity
                        .and_then(|id| world.get::<Name>(Entity::from_bits(id)))
                        .map(|name| name.to_string()),
                    t: follower.0.t,
                    speed: follower.0.speed,
                    looped: follower.0.looped,
                    follow_rotation: follower.0.follow_rotation,
                    up: follower.0.up.to_array(),
                    offset: follower.0.offset.to_array(),
                    length_samples: follower.0.length_samples,
                });
        let look_at = world.get::<BevyLookAt>(entity).map(|look_at| LookAtData {
            target_name: look_at
                .0
                .target_entity
                .and_then(|id| world.get::<Name>(Entity::from_bits(id)))
                .map(|name| name.to_string()),
            target_offset: look_at.0.target_offset.to_array(),
            offset_in_target_space: look_at.0.offset_in_target_space,
            up: look_at.0.up.to_array(),
            rotation_smooth_time: look_at.0.rotation_smooth_time,
        });
        let entity_follower =
            world
                .get::<BevyEntityFollower>(entity)
                .map(|follower| EntityFollowerData {
                    target_name: follower
                        .0
                        .target_entity
                        .and_then(|id| world.get::<Name>(Entity::from_bits(id)))
                        .map(|name| name.to_string()),
                    position_offset: follower.0.position_offset.to_array(),
                    offset_in_target_space: follower.0.offset_in_target_space,
                    follow_rotation: follower.0.follow_rotation,
                    position_smooth_time: follower.0.position_smooth_time,
                    rotation_smooth_time: follower.0.rotation_smooth_time,
                });
        let skeleton = world
            .get::<BevySkinnedMeshRenderer>(entity)
            .map(|skinned| skinned.0.skin.skeleton.as_ref());
        let animation = timeline_groups
            .as_ref()
            .and_then(|groups| groups.iter().find(|group| group.entity == entity.to_bits()))
            .and_then(|group| animation_component_from_group(group, skeleton));
        let pose_override = world
            .get::<BevyPoseOverride>(entity)
            .map(|pose| PoseOverrideData {
                enabled: pose.0.enabled,
                locals: pose_to_serialized(&pose.0.pose),
            });
        let freecam_settings = world.get::<Freecam>(entity).copied().map(|component| {
            let mut component = component;
            component.sanitize();
            SceneFreecamData::from_component(component)
        });
        let freecam = freecam_settings.is_some();
        let physics = world
            .get::<ColliderShape>(entity)
            .copied()
            .and_then(|shape| {
                let body_kind = if let Some(body) = world.get::<DynamicRigidBody>(entity) {
                    Some(PhysicsBodyKind::Dynamic { mass: body.mass })
                } else if let Some(body) = world.get::<KinematicRigidBody>(entity) {
                    Some(PhysicsBodyKind::Kinematic {
                        mode: body.mode.into(),
                    })
                } else if world.get::<FixedCollider>(entity).is_some() {
                    Some(PhysicsBodyKind::Fixed)
                } else {
                    None
                }?;

                let collider_properties = world
                    .get::<ColliderProperties>(entity)
                    .copied()
                    .map(SceneColliderPropertiesData::from);
                let collider_inheritance = world
                    .get::<ColliderPropertyInheritance>(entity)
                    .copied()
                    .map(SceneColliderPropertyInheritanceData::from);
                let rigid_body_properties = world
                    .get::<RigidBodyProperties>(entity)
                    .copied()
                    .map(SceneRigidBodyPropertiesData::from);
                let rigid_body_inheritance = world
                    .get::<RigidBodyPropertyInheritance>(entity)
                    .copied()
                    .map(SceneRigidBodyPropertyInheritanceData::from);
                let joint = world.get::<PhysicsJoint>(entity).map(|joint| {
                    let target_name = joint
                        .target
                        .and_then(|target| world.get::<Name>(target))
                        .map(|name| name.to_string());
                    ScenePhysicsJointData::from_component(*joint, target_name)
                });
                let character_controller = world
                    .get::<CharacterController>(entity)
                    .copied()
                    .map(SceneCharacterControllerData::from);
                let character_input = world
                    .get::<CharacterControllerInput>(entity)
                    .copied()
                    .map(SceneCharacterControllerInputData::from);
                let ray_cast = world
                    .get::<PhysicsRayCast>(entity)
                    .copied()
                    .map(ScenePhysicsRayCastData::from);
                let point_projection = world
                    .get::<PhysicsPointProjection>(entity)
                    .copied()
                    .map(ScenePhysicsPointProjectionData::from);
                let shape_cast = world
                    .get::<PhysicsShapeCast>(entity)
                    .copied()
                    .map(ScenePhysicsShapeCastData::from);

                Some(PhysicsComponentData {
                    collider_shape: SceneColliderShape::from(shape),
                    body_kind,
                    collider_properties,
                    collider_inheritance,
                    rigid_body_properties,
                    rigid_body_inheritance,
                    joint,
                    character_controller,
                    character_input,
                    ray_cast,
                    point_projection,
                    shape_cast,
                })
            });
        let physics_world_defaults = world
            .get::<PhysicsWorldDefaults>(entity)
            .copied()
            .map(ScenePhysicsWorldDefaultsData::from);

        let audio_emitter = audio_emitter.map(|emitter| {
            let (path, streaming) = editor_audio
                .map(|audio| (audio.path.clone(), audio.streaming))
                .unwrap_or((None, false));
            AudioEmitterData {
                path: path.map(|path| normalize_path(&path, root)),
                streaming,
                bus: emitter.0.bus,
                volume: emitter.0.volume,
                pitch: emitter.0.pitch,
                looping: emitter.0.looping,
                spatial: emitter.0.spatial,
                min_distance: emitter.0.min_distance,
                max_distance: emitter.0.max_distance,
                rolloff: emitter.0.rolloff,
                spatial_blend: emitter.0.spatial_blend,
                playback_state: emitter.0.playback_state,
                play_on_spawn: emitter.0.play_on_spawn,
            }
        });

        let audio_listener = audio_listener.map(|listener| AudioListenerData {
            enabled: listener.0.enabled,
        });

        let components = SceneComponents {
            mesh,
            skinned,
            sprite,
            text_2d,
            light,
            camera,
            scene,
            scripts,
            dynamic,
            spline,
            spline_follower,
            look_at,
            entity_follower,
            animation,
            pose_override,
            freecam,
            freecam_settings,
            physics,
            physics_world_defaults,
            audio_emitter,
            audio_listener,
        };

        entities.push(SceneEntityData {
            name: name.map(|name| name.to_string()),
            transform: serialized_transform,
            relation: None,
            scene_child: None,
            scene_child_renderer,
            components,
        });
        pending_relations.push(parent_relation.map(|relation| relation.parent));
        pending_scene_children
            .push(scene_child_meta.map(|child| (child.scene_root, child.scene_node_index)));
        entity_order.push(entity);
        saved_entities.insert(entity.to_bits());
    }

    let mut entity_index_map: HashMap<Entity, usize> = HashMap::new();
    for (index, entity) in entity_order.iter().copied().enumerate() {
        entity_index_map.insert(entity, index);
    }

    for (index, entity_data) in entities.iter_mut().enumerate() {
        if let Some(parent_entity) = pending_relations.get(index).copied().flatten() {
            if let Some(parent_index) = entity_index_map.get(&parent_entity).copied() {
                entity_data.relation = Some(SceneEntityRelationData {
                    parent: parent_index,
                });
            }
        }
        if let Some((scene_root_entity, scene_node_index)) =
            pending_scene_children.get(index).copied().flatten()
        {
            if let Some(scene_root_index) = entity_index_map.get(&scene_root_entity).copied() {
                entity_data.scene_child = Some(SceneChildLinkData {
                    scene_root: scene_root_index,
                    scene_node_index,
                });
            }
        }
    }

    let mut scene_child_animations: Vec<SceneChildAnimationData> = Vec::new();
    if let Some(groups) = timeline_groups.as_ref() {
        for group in groups {
            if saved_entities.contains(&group.entity) {
                continue;
            }
            let entity = Entity::from_bits(group.entity);
            let Some(child) = world.get::<SceneChild>(entity) else {
                continue;
            };
            let Some(scene_root) = world.get::<SceneAssetPath>(child.scene_root) else {
                continue;
            };
            let skeleton = world
                .get::<BevySkinnedMeshRenderer>(entity)
                .map(|skinned| skinned.0.skin.skeleton.as_ref());
            let Some(animation) = animation_component_from_group(group, skeleton) else {
                continue;
            };
            let scene_path = normalize_path(scene_root.path.to_string_lossy().as_ref(), root);
            scene_child_animations.push(SceneChildAnimationData {
                scene_path,
                scene_node_index: child.scene_node_index,
                animation,
            });
        }
    }

    let mut scene_child_pose_overrides: Vec<SceneChildPoseOverrideData> = Vec::new();
    let mut pose_query = world.query::<(Entity, &SceneChild, &BevyPoseOverride)>();
    for (entity, child, pose_override) in pose_query.iter(world) {
        if saved_entities.contains(&entity.to_bits()) {
            continue;
        }
        let Some(scene_root) = world.get::<SceneAssetPath>(child.scene_root) else {
            continue;
        };
        let scene_path = normalize_path(scene_root.path.to_string_lossy().as_ref(), root);
        scene_child_pose_overrides.push(SceneChildPoseOverrideData {
            scene_path,
            scene_node_index: child.scene_node_index,
            pose: PoseOverrideData {
                enabled: pose_override.0.enabled,
                locals: pose_to_serialized(&pose_override.0.pose),
            },
        });
    }

    (
        SceneDocument {
            version: 1,
            entities,
            scene_child_animations,
            scene_child_pose_overrides,
        },
        entity_order,
    )
}

pub fn write_scene_document(path: &Path, document: &SceneDocument) -> Result<(), String> {
    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(6)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(document, pretty).map_err(|err| err.to_string())?;
    fs::write(path, data).map_err(|err| err.to_string())
}

pub fn read_scene_document(path: &Path) -> Result<SceneDocument, String> {
    let data = fs::read_to_string(path).map_err(|err| err.to_string())?;
    ron::de::from_str::<SceneDocument>(&data).map_err(|err| err.to_string())
}

pub fn write_animation_asset_document(
    path: &Path,
    document: &AnimationAssetDocument,
) -> Result<(), String> {
    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(6)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(document, pretty).map_err(|err| err.to_string())?;
    fs::write(path, data).map_err(|err| err.to_string())
}

pub fn read_animation_asset_document(path: &Path) -> Result<AnimationAssetDocument, String> {
    let data = fs::read_to_string(path).map_err(|err| err.to_string())?;
    ron::de::from_str::<AnimationAssetDocument>(&data).map_err(|err| err.to_string())
}

pub fn spawn_scene_from_document(
    world: &mut World,
    document: &SceneDocument,
    project: &EditorProject,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Vec<Entity> {
    let mut created = Vec::new();
    let root = project.root.as_deref();

    let mut any_play_camera = false;
    let mut pending_followers: Vec<(Entity, SplineFollowerData)> = Vec::new();
    let mut pending_look_ats: Vec<(Entity, LookAtData)> = Vec::new();
    let mut pending_entity_followers: Vec<(Entity, EntityFollowerData)> = Vec::new();
    let mut pending_joints: Vec<(Entity, String)> = Vec::new();
    let mut pending_parent_relations: Vec<(Entity, usize)> = Vec::new();
    let mut pending_scene_children: Vec<(Entity, SceneChildLinkData)> = Vec::new();
    let mut pending_scene_child_renderers: Vec<(Entity, SceneChildRendererData)> = Vec::new();

    for entity_data in &document.entities {
        let transform = entity_data.transform.to_transform();
        let mut entity = world.spawn((EditorEntity, BevyWrapper(transform)));
        let skinned_data = entity_data.components.skinned.clone();

        if let Some(name) = &entity_data.name {
            entity.insert(Name::new(name.clone()));
        }

        if let Some(mesh) = &entity_data.components.mesh {
            if let Some(mesh_renderer) =
                build_mesh_renderer(mesh, asset_cache, asset_server, project)
            {
                entity.insert(mesh_renderer);
                entity.insert(EditorMesh {
                    source: mesh.source.clone(),
                    material_path: mesh.material.clone(),
                });
            }
        }

        if let Some(skinned) = &skinned_data {
            entity.insert(EditorSkinnedMesh {
                scene_path: Some(skinned.scene_path.clone()),
                node_index: skinned.scene_node_index,
                casts_shadow: skinned.casts_shadow,
                visible: skinned.visible,
            });
        }

        if let Some(light) = &entity_data.components.light {
            entity.insert(BevyWrapper(serde_light_to_component(light)));
        }

        if let Some(camera) = &entity_data.components.camera {
            entity.insert(BevyWrapper(Camera {
                fov_y_rad: camera.fov_y_rad,
                aspect_ratio: camera.aspect_ratio,
                near_plane: camera.near_plane,
                far_plane: camera.far_plane,
            }));
            if camera.active {
                any_play_camera = true;
                entity.insert(EditorPlayCamera);
            }
        }

        if let Some(sprite) = &entity_data.components.sprite {
            let texture_path = sprite.texture.as_ref().and_then(|path| {
                let path = path.trim().to_string();
                if path.is_empty() { None } else { Some(path) }
            });
            let texture_id = texture_path.as_ref().map(|path| {
                let resolved = resolve_path(path, root);
                cached_texture_handle(asset_cache, asset_server, &resolved).id
            });
            let sequence_texture_paths = sprite
                .image_sequence
                .textures
                .iter()
                .filter_map(|path| {
                    let path = path.trim().to_string();
                    if path.is_empty() { None } else { Some(path) }
                })
                .collect::<Vec<_>>();
            let sequence_texture_ids = sequence_texture_paths
                .iter()
                .map(|path| {
                    let resolved = resolve_path(path, root);
                    cached_texture_handle(asset_cache, asset_server, &resolved).id
                })
                .collect::<Vec<_>>();
            entity.insert(BevyWrapper(SpriteRenderer {
                color: sprite.color,
                texture_id,
                uv_min: sprite.uv_min,
                uv_max: sprite.uv_max,
                sheet_animation: sprite.sheet_animation.to_sheet(),
                pivot: sprite.pivot,
                clip_rect: sprite.clip_rect,
                layer: sprite.layer,
                space: sprite.space.to_space(),
                blend_mode: sprite.blend_mode.to_blend_mode(),
                billboard: sprite.billboard,
                visible: sprite.visible,
                pick_id: sprite.pick_id,
            }));
            if sprite.image_sequence.has_component_state() {
                entity.insert(BevySpriteImageSequence(
                    sprite.image_sequence.to_component(sequence_texture_ids),
                ));
            }
            entity.insert(EditorSprite {
                texture_path: texture_path.clone(),
                sequence_texture_paths,
            });
        }

        if let Some(text) = &entity_data.components.text_2d {
            let font_path = text.font_path.as_ref().and_then(|path| {
                let path = path.trim();
                if path.is_empty() {
                    None
                } else {
                    Some(resolve_path(path, root).to_string_lossy().to_string())
                }
            });
            entity.insert(BevyText2d(Text2d {
                text: text.text.clone(),
                color: text.color,
                font_path,
                font_family: text.font_family.clone(),
                font_size: text.font_size,
                font_weight: text.font_weight,
                font_width: text.font_width,
                font_style: text.font_style.to_style(),
                line_height_scale: text.line_height_scale,
                letter_spacing: text.letter_spacing,
                word_spacing: text.word_spacing,
                underline: text.underline,
                strikethrough: text.strikethrough,
                max_width: text.max_width,
                align_h: text.align_h.to_align(),
                align_v: text.align_v.to_align(),
                space: text.space.to_space(),
                billboard: text.billboard,
                blend_mode: text.blend_mode.to_blend_mode(),
                layer: text.layer,
                clip_rect: text.clip_rect,
                visible: text.visible,
                pick_id: text.pick_id,
            }));
        }

        if let Some(listener) = &entity_data.components.audio_listener {
            entity.insert(BevyWrapper(AudioListener {
                enabled: listener.enabled,
            }));
        }

        if let Some(audio) = &entity_data.components.audio_emitter {
            let mut emitter = AudioEmitter {
                bus: audio.bus,
                volume: audio.volume,
                pitch: audio.pitch,
                looping: audio.looping,
                spatial: audio.spatial,
                min_distance: audio.min_distance,
                max_distance: audio.max_distance,
                rolloff: audio.rolloff,
                spatial_blend: audio.spatial_blend,
                playback_state: audio.playback_state,
                play_on_spawn: audio.play_on_spawn,
                clip_id: None,
            };

            if let Some(path) = audio.path.as_ref() {
                let resolved = resolve_path(path, root);
                let handle =
                    cached_audio_handle(asset_cache, asset_server, &resolved, audio.streaming);
                emitter.clip_id = Some(handle.id);
                entity.insert(EditorAudio {
                    path: Some(path.clone()),
                    streaming: audio.streaming,
                });
            } else {
                entity.insert(EditorAudio {
                    path: None,
                    streaming: audio.streaming,
                });
            }

            entity.insert(BevyWrapper(emitter));
        }

        if entity_data.components.freecam || entity_data.components.freecam_settings.is_some() {
            let component = entity_data
                .components
                .freecam_settings
                .map(SceneFreecamData::into_component)
                .unwrap_or_default();
            entity.insert(component);
        }

        if let Some(scene) = &entity_data.components.scene {
            let path = resolve_path(&scene.path, root);
            let handle = cached_scene_handle(asset_cache, asset_server, &path);
            entity.insert(SceneRoot(handle));
            entity.insert(SceneAssetPath {
                path: resolve_path(&scene.path, root),
            });
        }

        if !entity_data.components.scripts.is_empty() {
            let scripts = entity_data
                .components
                .scripts
                .iter()
                .filter_map(|script| {
                    let path = script.path.trim();
                    if path.is_empty() {
                        return None;
                    }
                    let resolved = resolve_path(path, root);
                    let mut entry = ScriptEntry {
                        path: Some(resolved.clone()),
                        language: normalize_script_language(
                            &script.language,
                            Some(resolved.as_path()),
                        ),
                        inspector_fields: script.inspector_fields.clone(),
                    };
                    entry.sanitize_inspector_fields();
                    Some(entry)
                })
                .collect::<Vec<_>>();
            if !scripts.is_empty() {
                entity.insert(ScriptComponent { scripts });
            }
        }

        if !entity_data.components.dynamic.is_empty() {
            entity.insert(DynamicComponents {
                components: entity_data.components.dynamic.clone(),
            });
        }

        if let Some(spline) = &entity_data.components.spline {
            let points = spline
                .points
                .iter()
                .map(|point| glam::Vec3::from_array(*point))
                .collect();
            entity.insert(BevySpline(Spline {
                points,
                closed: spline.closed,
                mode: spline.mode.to_mode(),
                tension: spline.tension,
            }));
        }

        if let Some(follower) = &entity_data.components.spline_follower {
            entity.insert(BevySplineFollower(SplineFollower {
                spline_entity: None,
                t: follower.t,
                speed: follower.speed,
                looped: follower.looped,
                follow_rotation: follower.follow_rotation,
                up: glam::Vec3::from_array(follower.up),
                offset: glam::Vec3::from_array(follower.offset),
                length_samples: follower.length_samples,
            }));
            pending_followers.push((entity.id(), follower.clone()));
        }

        if let Some(look_at) = &entity_data.components.look_at {
            entity.insert(BevyLookAt(LookAt {
                target_entity: None,
                target_offset: glam::Vec3::from_array(look_at.target_offset),
                offset_in_target_space: look_at.offset_in_target_space,
                up: glam::Vec3::from_array(look_at.up),
                rotation_smooth_time: look_at.rotation_smooth_time,
            }));
            pending_look_ats.push((entity.id(), look_at.clone()));
        }

        if let Some(follower) = &entity_data.components.entity_follower {
            entity.insert(BevyEntityFollower(EntityFollower {
                target_entity: None,
                position_offset: glam::Vec3::from_array(follower.position_offset),
                offset_in_target_space: follower.offset_in_target_space,
                follow_rotation: follower.follow_rotation,
                position_smooth_time: follower.position_smooth_time,
                rotation_smooth_time: follower.rotation_smooth_time,
            }));
            pending_entity_followers.push((entity.id(), follower.clone()));
        }

        if let Some(physics) = &entity_data.components.physics {
            entity.insert(physics.collider_shape.to_collider_shape());
            if let Some(collider_props) = physics.collider_properties.as_ref() {
                entity.insert(collider_props.to_component());
            }
            if let Some(collider_inheritance) = physics.collider_inheritance {
                entity.insert(collider_inheritance.to_component());
            }
            if let Some(rigid_body_props) = physics.rigid_body_properties.as_ref() {
                entity.insert(rigid_body_props.to_component());
            }
            if let Some(rigid_body_inheritance) = physics.rigid_body_inheritance {
                entity.insert(rigid_body_inheritance.to_component());
            }
            match physics.body_kind.clone() {
                PhysicsBodyKind::Dynamic { mass } => {
                    entity.insert(DynamicRigidBody { mass });
                }
                PhysicsBodyKind::Kinematic { mode } => {
                    entity.insert(KinematicRigidBody {
                        mode: mode.to_component(),
                    });
                }
                PhysicsBodyKind::Fixed => {
                    entity.insert(FixedCollider);
                }
            }

            if let Some(joint) = physics.joint.as_ref() {
                entity.insert(joint.to_component());
                if let Some(target_name) = joint.target_name.as_ref() {
                    pending_joints.push((entity.id(), target_name.clone()));
                }
            }

            let has_character_controller = physics.character_controller.is_some();
            if let Some(controller) = physics.character_controller.as_ref() {
                entity.insert(controller.to_component());
            }
            if let Some(input) = physics.character_input.as_ref() {
                entity.insert(input.to_component());
            } else if has_character_controller {
                entity.insert(CharacterControllerInput::default());
            }
            if has_character_controller {
                entity
                    .insert(helmer_becs::physics::components::CharacterControllerOutput::default());
            }
            if let Some(ray_cast) = physics.ray_cast.as_ref() {
                entity.insert(ray_cast.to_component());
                entity.insert(helmer_becs::physics::components::PhysicsRayCastHit::default());
            }
            if let Some(point_projection) = physics.point_projection.as_ref() {
                entity.insert(point_projection.to_component());
                entity
                    .insert(helmer_becs::physics::components::PhysicsPointProjectionHit::default());
            }
            if let Some(shape_cast) = physics.shape_cast.as_ref() {
                entity.insert(shape_cast.to_component());
                entity.insert(helmer_becs::physics::components::PhysicsShapeCastHit::default());
            }
        }

        if let Some(world_defaults) = &entity_data.components.physics_world_defaults {
            entity.insert(world_defaults.to_component());
        }

        let entity_id = entity.id();
        drop(entity);

        if let Some(relation) = entity_data.relation.as_ref() {
            pending_parent_relations.push((entity_id, relation.parent));
        }
        if let Some(scene_child) = entity_data.scene_child.as_ref() {
            pending_scene_children.push((entity_id, scene_child.clone()));
        }
        if let Some(scene_child_renderer) = entity_data.scene_child_renderer {
            pending_scene_child_renderers.push((entity_id, scene_child_renderer));
        }

        if let Some(skinned) = skinned_data {
            let path = resolve_path(&skinned.scene_path, root);
            let (scene_handle, scene) = {
                let handle = cached_scene_handle(asset_cache, asset_server, &path);
                let scene = asset_server.0.lock().get_scene(&handle);
                (handle, scene)
            };

            let mut node_index = skinned.scene_node_index;
            if let Some(scene) = scene.as_ref() {
                if node_index
                    .and_then(|index| scene.nodes.get(index))
                    .map(|node| node.skin_index.is_some())
                    != Some(true)
                {
                    node_index = scene
                        .nodes
                        .iter()
                        .position(|node| node.skin_index.is_some());
                }
            }

            if node_index.is_none() {
                world.entity_mut(entity_id).insert(PendingSkinnedMeshAsset {
                    scene_handle,
                    node_index,
                });
                continue;
            }

            let Some(scene) = scene else {
                world.entity_mut(entity_id).insert(PendingSkinnedMeshAsset {
                    scene_handle,
                    node_index,
                });
                continue;
            };

            let node_index = node_index.unwrap();
            let Some(node) = scene.nodes.get(node_index) else {
                world.entity_mut(entity_id).insert(PendingSkinnedMeshAsset {
                    scene_handle,
                    node_index: Some(node_index),
                });
                continue;
            };
            let Some(skin_index) = node.skin_index else {
                world.entity_mut(entity_id).insert(PendingSkinnedMeshAsset {
                    scene_handle,
                    node_index: Some(node_index),
                });
                continue;
            };
            let Some(skin) = scene.skins.read().get(skin_index).cloned() else {
                world.entity_mut(entity_id).insert(PendingSkinnedMeshAsset {
                    scene_handle,
                    node_index: Some(node_index),
                });
                continue;
            };

            let skinned_renderer = SkinnedMeshRenderer::new(
                node.mesh.id,
                node.material.id,
                skin,
                skinned.casts_shadow,
                skinned.visible,
            );

            {
                let mut entity_mut = world.entity_mut(entity_id);
                entity_mut.remove::<BevyMeshRenderer>();
                entity_mut.remove::<EditorMesh>();
                entity_mut.remove::<PendingSkinnedMeshAsset>();
                entity_mut.insert(BevySkinnedMeshRenderer(skinned_renderer));
            }

            if world.get::<BevyAnimator>(entity_id).is_none() {
                if let Some(anim_lib) = scene.animations.read().get(skin_index).cloned() {
                    world
                        .entity_mut(entity_id)
                        .insert(BevyAnimator(build_default_animator(anim_lib)));
                }
            }
        }

        if let Some(animation) = &entity_data.components.animation {
            let skeleton = world
                .get::<BevySkinnedMeshRenderer>(entity_id)
                .map(|skinned| skinned.0.skin.skeleton.clone());
            let skeleton_ref = skeleton.as_deref();
            if let Some(mut timeline) = world.get_resource_mut::<EditorTimelineState>() {
                let name = entity_data
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Entity {}", entity_id.index()));
                apply_animation_data_to_timeline(
                    &mut timeline,
                    entity_id,
                    name,
                    animation,
                    skeleton_ref,
                );
            }
            if let Some(mut animator) = world.get_mut::<BevyAnimator>(entity_id) {
                let custom_clips = animation
                    .clips
                    .iter()
                    .map(|clip| clip.to_clip_for_skeleton(skeleton_ref))
                    .collect::<Vec<_>>();
                apply_custom_clips_to_animator(&mut animator, &custom_clips);
            }
        }

        if let Some(pose_override) = &entity_data.components.pose_override {
            let skeleton = world
                .get::<BevySkinnedMeshRenderer>(entity_id)
                .map(|skinned| skinned.0.skin.skeleton.clone());
            let pose = pose_from_serialized(&pose_override.locals, skeleton.as_deref());
            world
                .entity_mut(entity_id)
                .insert(BevyPoseOverride(PoseOverride {
                    enabled: pose_override.enabled,
                    pose,
                }));
        }

        created.push(entity_id);
    }

    let mut restored_scene_children: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut restored_scene_links: HashMap<Entity, SceneChild> = HashMap::new();
    for (entity, scene_child) in pending_scene_children {
        let Some(scene_root) = created.get(scene_child.scene_root).copied() else {
            continue;
        };
        let scene_child_link = SceneChild {
            scene_root,
            scene_node_index: scene_child.scene_node_index,
        };
        world.entity_mut(entity).insert(scene_child_link);
        restored_scene_links.insert(entity, scene_child_link);
        restored_scene_children
            .entry(scene_root)
            .or_default()
            .push(entity);
    }

    if !restored_scene_links.is_empty() {
        let mut pending_renderer_map: HashMap<Entity, SceneChildRendererData> =
            pending_scene_child_renderers.into_iter().collect();
        for (entity, _) in restored_scene_links.iter() {
            let explicit = pending_renderer_map.remove(entity);
            let has_runtime_renderer = world.get::<BevyMeshRenderer>(*entity).is_some()
                || world.get::<BevySkinnedMeshRenderer>(*entity).is_some()
                || world.get::<PendingSkinnedMeshAsset>(*entity).is_some();
            let has_authored_renderer = world.get::<EditorMesh>(*entity).is_some()
                || world.get::<EditorSkinnedMesh>(*entity).is_some();

            let renderer_data = if let Some(explicit) = explicit {
                explicit
            } else if !has_runtime_renderer && !has_authored_renderer {
                SceneChildRendererData {
                    kind: SceneChildRendererKind::Auto,
                    casts_shadow: true,
                    visible: true,
                }
            } else {
                continue;
            };

            world.entity_mut(*entity).insert(PendingSceneChildRenderer {
                kind: renderer_data.kind,
                casts_shadow: renderer_data.casts_shadow,
                visible: renderer_data.visible,
            });
        }
    }

    if !restored_scene_children.is_empty() {
        if let Some(mut spawned) = world.get_resource_mut::<SceneSpawnedChildren>() {
            for (scene_root, children) in restored_scene_children.iter() {
                spawned.spawned_scenes.insert(*scene_root, children.clone());
            }
        } else {
            let mut spawned = SceneSpawnedChildren::default();
            for (scene_root, children) in restored_scene_children.iter() {
                spawned.spawned_scenes.insert(*scene_root, children.clone());
            }
            world.insert_resource(spawned);
        }
        for scene_root in restored_scene_children.keys().copied() {
            world.entity_mut(scene_root).insert(SpawnedScene);
        }
    }

    for (entity, parent_index) in pending_parent_relations {
        let Some(parent_entity) = created.get(parent_index).copied() else {
            continue;
        };
        if entity == parent_entity {
            continue;
        }
        let child_transform = world
            .get::<BevyTransform>(entity)
            .map(|transform| transform.0)
            .unwrap_or_default();
        let parent_matrix = world
            .get::<BevyTransform>(parent_entity)
            .map(|transform| transform.0.to_matrix())
            .unwrap_or(glam::Mat4::IDENTITY);
        world.entity_mut(entity).insert(EntityParent {
            parent: parent_entity,
            local_transform: parent_matrix.inverse() * child_transform.to_matrix(),
            last_written: child_transform,
        });
    }

    if !pending_followers.is_empty()
        || !pending_look_ats.is_empty()
        || !pending_entity_followers.is_empty()
        || !pending_joints.is_empty()
    {
        let mut name_map: HashMap<String, Entity> = HashMap::new();
        for entity in &created {
            if let Some(name) = world.get::<Name>(*entity) {
                name_map.insert(name.to_string(), *entity);
            }
        }
        for (entity, follower) in pending_followers {
            let Some(target_name) = follower.spline_name.as_ref() else {
                continue;
            };
            let Some(target_entity) = name_map.get(target_name) else {
                continue;
            };
            if let Some(mut component) = world.get_mut::<BevySplineFollower>(entity) {
                component.0.spline_entity = Some(target_entity.to_bits());
            }
        }

        for (entity, look_at) in pending_look_ats {
            let Some(target_name) = look_at.target_name.as_ref() else {
                continue;
            };
            let Some(target_entity) = name_map.get(target_name) else {
                continue;
            };
            if let Some(mut component) = world.get_mut::<BevyLookAt>(entity) {
                component.0.target_entity = Some(target_entity.to_bits());
            }
        }

        for (entity, follower) in pending_entity_followers {
            let Some(target_name) = follower.target_name.as_ref() else {
                continue;
            };
            let Some(target_entity) = name_map.get(target_name) else {
                continue;
            };
            if let Some(mut component) = world.get_mut::<BevyEntityFollower>(entity) {
                component.0.target_entity = Some(target_entity.to_bits());
            }
        }

        for (entity, target_name) in pending_joints {
            let Some(target_entity) = name_map.get(&target_name) else {
                continue;
            };
            if let Some(mut joint) = world.get_mut::<PhysicsJoint>(entity) {
                joint.target = Some(*target_entity);
            }
        }
    }

    if !any_play_camera {
        ensure_active_camera(world);
    }

    if let Some(mut pending) = world.get_resource_mut::<PendingSceneChildAnimations>() {
        pending.entries = document.scene_child_animations.clone();
    } else {
        world.insert_resource(PendingSceneChildAnimations {
            entries: document.scene_child_animations.clone(),
        });
    }

    if let Some(mut pending) = world.get_resource_mut::<PendingSceneChildPoseOverrides>() {
        pending.entries = document.scene_child_pose_overrides.clone();
    } else {
        world.insert_resource(PendingSceneChildPoseOverrides {
            entries: document.scene_child_pose_overrides.clone(),
        });
    }

    created
}

pub fn restore_scene_transforms_from_document(
    world: &mut World,
    document: &SceneDocument,
    entities: &[Entity],
) {
    for (entity, entity_data) in entities.iter().zip(document.entities.iter()) {
        if let Some(mut transform) = world.get_mut::<BevyTransform>(*entity) {
            transform.0 = entity_data.transform.to_transform();
        }
    }
}

fn build_mesh_renderer(
    mesh: &MeshComponentData,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
    project: &EditorProject,
) -> Option<BevyMeshRenderer> {
    let material_handle = mesh
        .material
        .as_ref()
        .and_then(|path| load_material_handle(path, asset_cache, asset_server, project))
        .or_else(|| asset_cache.default_material);

    let Some(material_handle) = material_handle else {
        return None;
    };

    let mesh_handle = match &mesh.source {
        MeshSource::Primitive(kind) => Some(load_primitive_mesh(*kind, asset_cache, asset_server)),
        MeshSource::Asset { path } => {
            Some(load_mesh_asset(path, asset_cache, asset_server, project))
        }
    }?;

    Some(BevyWrapper(MeshRenderer::new(
        mesh_handle.id,
        material_handle.id,
        mesh.casts_shadow,
        mesh.visible,
    )))
}

fn load_material_handle(
    path: &str,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
    project: &EditorProject,
) -> Option<Handle<helmer::runtime::asset_server::Material>> {
    if let Some(handle) = asset_cache.material_handles.get(path).copied() {
        return Some(handle);
    }

    let root = project.root.as_deref()?;
    let full_path = resolve_path(path, Some(root));
    let handle = asset_server.0.lock().load_material(full_path);
    asset_cache
        .material_handles
        .insert(path.to_string(), handle);
    Some(handle)
}

fn load_mesh_asset(
    path: &str,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
    project: &EditorProject,
) -> Handle<helmer::runtime::asset_server::Mesh> {
    if let Some(handle) = asset_cache.mesh_handles.get(path).copied() {
        return handle;
    }

    let root = project.root.as_deref();
    let full_path = resolve_path(path, root);
    let handle = asset_server.0.lock().load_mesh(full_path);
    asset_cache.mesh_handles.insert(path.to_string(), handle);
    handle
}

fn load_primitive_mesh(
    kind: PrimitiveKind,
    asset_cache: &mut EditorAssetCache,
    asset_server: &helmer_becs::BevyAssetServer,
) -> Handle<helmer::runtime::asset_server::Mesh> {
    if let Some(handle) = asset_cache.primitive_meshes.get(&kind).copied() {
        return handle;
    }

    let mesh_asset = kind.to_mesh_asset();

    let handle = asset_server
        .0
        .lock()
        .add_mesh(mesh_asset.vertices.unwrap(), mesh_asset.indices);
    asset_cache.primitive_meshes.insert(kind, handle);
    handle
}

fn serde_light_to_component(light: &LightComponentData) -> Light {
    match light.kind {
        LightKind::Directional => Light::directional(
            glam::vec3(light.color[0], light.color[1], light.color[2]),
            light.intensity,
        ),
        LightKind::Point => Light::point(
            glam::vec3(light.color[0], light.color[1], light.color[2]),
            light.intensity,
        ),
        LightKind::Spot { angle } => Light::spot(
            glam::vec3(light.color[0], light.color[1], light.color[2]),
            light.intensity,
            angle,
        ),
    }
}

pub fn default_scene_path(project: &EditorProject) -> Option<PathBuf> {
    let root = project.root.as_ref()?;
    let config = project.config.as_ref()?;
    Some(config.scenes_root(root).join("untitled.hscene.ron"))
}

pub fn next_available_scene_path(project: &EditorProject) -> Option<PathBuf> {
    let root = project.root.as_ref()?;
    let config = project.config.as_ref()?;
    let scenes_root = config.scenes_root(root);

    for idx in 1..=999u32 {
        let candidate = scenes_root.join(format!("scene_{:03}.hscene.ron", idx));
        if !candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

pub(crate) fn normalize_path(path: &str, root: Option<&Path>) -> String {
    if let Some(root) = root {
        if let Ok(relative) = Path::new(path).strip_prefix(root) {
            return relative.to_string_lossy().replace('\\', "/");
        }
    }
    path.replace('\\', "/")
}

fn resolve_path(path: &str, root: Option<&Path>) -> PathBuf {
    if let Some(root) = root {
        let candidate = root.join(path);
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from(path)
}
