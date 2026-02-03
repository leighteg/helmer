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
    provided::components::{
        Camera, EntityFollower, Light, LightType, LookAt, MeshRenderer, PoseOverride,
        SkinnedMeshRenderer, Spline, SplineFollower, SplineMode, Transform,
    },
    runtime::asset_server::{Handle, Scene},
};
use helmer_becs::physics::components::{ColliderShape, DynamicRigidBody, FixedCollider};
use helmer_becs::{
    BevyAnimator, BevyCamera, BevyEntityFollower, BevyLight, BevyLookAt, BevyMeshRenderer,
    BevyPoseOverride, BevySkinnedMeshRenderer, BevySpline, BevySplineFollower, BevyTransform,
    BevyWrapper,
    systems::scene_system::{SceneChild, SceneRoot, SceneSpawnedChildren, build_default_animator},
};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

use crate::editor::{
    EditorPlayCamera, EditorTimelineState, EditorViewportCamera, Freecam,
    assets::{
        EditorAssetCache, EditorMesh, EditorSkinnedMesh, MeshSource, PrimitiveKind, SceneAssetPath,
        cached_scene_handle,
    },
    dynamic::{DynamicComponent, DynamicComponents},
    project::EditorProject,
    scripting::{ScriptComponent, ScriptEntry},
    timeline::{
        CameraKey, CameraTrack, ClipSegment, ClipTrack, JointKey, JointTrack, LightKey, LightTrack,
        PoseKey, PoseTrack, SplineKey, SplineTrack, TimelineInterpolation, TimelineTrack,
        TimelineTrackGroup, TransformKey, TransformTrack,
    },
};

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EditorEntity;

#[derive(Component, Debug, Clone)]
pub struct PendingSkinnedMeshAsset {
    pub scene_handle: Handle<Scene>,
    pub node_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldState {
    Edit,
    Play,
}

#[derive(Resource, Debug, Clone)]
pub struct EditorSceneState {
    pub path: Option<PathBuf>,
    pub name: String,
    pub dirty: bool,
    pub world_state: WorldState,
    pub play_backup: Option<SceneDocument>,
    pub play_selected_index: Option<usize>,
}

impl Default for EditorSceneState {
    fn default() -> Self {
        Self {
            path: None,
            name: "Untitled".to_string(),
            dirty: false,
            world_state: WorldState::Edit,
            play_backup: None,
            play_selected_index: None,
        }
    }
}

#[derive(Resource, Debug, Default, Clone)]
pub struct EditorRenderRefresh {
    pub pending: bool,
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

#[derive(Resource, Debug, Default, Clone)]
pub struct PendingSceneChildAnimations {
    pub entries: Vec<SceneChildAnimationData>,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct PendingSceneChildPoseOverrides {
    pub entries: Vec<SceneChildPoseOverrideData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntityData {
    pub name: Option<String>,
    pub transform: SerializedTransform,
    pub components: SceneComponents,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SceneComponents {
    pub mesh: Option<MeshComponentData>,
    #[serde(default)]
    pub skinned: Option<SkinnedMeshComponentData>,
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
    pub physics: Option<PhysicsComponentData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsComponentData {
    pub collider_shape: SceneColliderShape,
    pub body_kind: PhysicsBodyKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhysicsBodyKind {
    Dynamic { mass: f32 },
    Fixed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SceneColliderShape {
    Cuboid,
    Sphere,
}

impl From<ColliderShape> for SceneColliderShape {
    fn from(shape: ColliderShape) -> Self {
        match shape {
            ColliderShape::Cuboid => SceneColliderShape::Cuboid,
            ColliderShape::Sphere => SceneColliderShape::Sphere,
        }
    }
}

impl SceneColliderShape {
    pub fn to_collider_shape(&self) -> ColliderShape {
        match self {
            SceneColliderShape::Cuboid => ColliderShape::Cuboid,
            SceneColliderShape::Sphere => ColliderShape::Sphere,
        }
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
        interpolation: Interpolation,
        keyframes: Vec<Vec3KeyframeData>,
    },
    Rotation {
        target: usize,
        interpolation: Interpolation,
        keyframes: Vec<QuatKeyframeData>,
    },
    Scale {
        target: usize,
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
    pub fn from_clip(clip: &AnimationClip) -> Self {
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
        let channels = self
            .channels
            .iter()
            .map(|channel| match channel {
                AnimationChannelData::Translation {
                    target,
                    interpolation,
                    keyframes,
                } => AnimationChannel::Translation {
                    target: *target,
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
                AnimationChannelData::Rotation {
                    target,
                    interpolation,
                    keyframes,
                } => AnimationChannel::Rotation {
                    target: *target,
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
                AnimationChannelData::Scale {
                    target,
                    interpolation,
                    keyframes,
                } => AnimationChannel::Scale {
                    target: *target,
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
            })
            .collect();

        AnimationClip {
            name: self.name.clone(),
            duration: self.duration,
            channels,
        }
    }
}

fn animation_component_from_group(group: &TimelineTrackGroup) -> Option<AnimationComponentData> {
    let tracks = group
        .tracks
        .iter()
        .map(animation_track_to_data)
        .collect::<Vec<_>>();
    let clips = group
        .custom_clips
        .iter()
        .map(AnimationClipData::from_clip)
        .collect::<Vec<_>>();
    if tracks.is_empty() && clips.is_empty() {
        None
    } else {
        Some(AnimationComponentData { tracks, clips })
    }
}

fn animation_track_to_data(track: &TimelineTrack) -> AnimationTrackData {
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
) -> TimelineTrack {
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
            TimelineTrack::Pose(PoseTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                weight: track.weight,
                additive: track.additive,
                translation_interpolation: track.translation_interpolation,
                rotation_interpolation: track.rotation_interpolation,
                scale_interpolation: track.scale_interpolation,
                keys,
            })
        }
        AnimationTrackData::Joint(track) => {
            let keys = track
                .keys
                .iter()
                .map(|key| JointKey {
                    id: key.id,
                    time: key.time,
                    transform: key.transform.to_transform(),
                })
                .collect();
            TimelineTrack::Joint(JointTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                joint_index: track.joint_index,
                weight: track.weight,
                additive: track.additive,
                translation_interpolation: track.translation_interpolation,
                rotation_interpolation: track.rotation_interpolation,
                scale_interpolation: track.scale_interpolation,
                keys,
            })
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
            TimelineTrack::Transform(TransformTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                translation_interpolation: track.translation_interpolation,
                rotation_interpolation: track.rotation_interpolation,
                scale_interpolation: track.scale_interpolation,
                keys,
            })
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
            TimelineTrack::Camera(CameraTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                interpolation: track.interpolation,
                keys,
            })
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
            TimelineTrack::Light(LightTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                interpolation: track.interpolation,
                keys,
            })
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
            TimelineTrack::Spline(SplineTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                interpolation: track.interpolation,
                keys,
            })
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
            TimelineTrack::Clip(ClipTrack {
                id: track.id,
                name: track.name.clone(),
                enabled: track.enabled,
                weight: track.weight,
                additive: track.additive,
                segments,
            })
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
        .map(|track| animation_track_from_data(track, skeleton))
        .collect();
    group.custom_clips = data.clips.iter().map(AnimationClipData::to_clip).collect();
    timeline.apply_requested = true;
    update_timeline_next_id(timeline);
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
        timeline.selected = None;
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
    let mut query = world.query_filtered::<(
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
        Option<&SceneRoot>,
        Option<&SceneAssetPath>,
        Option<&ScriptComponent>,
    ), With<EditorEntity>>();

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
        scene_root,
        scene_asset,
        script,
    ) in query.iter(world)
    {
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
        let animation = timeline_groups
            .as_ref()
            .and_then(|groups| groups.iter().find(|group| group.entity == entity.to_bits()))
            .and_then(animation_component_from_group);
        let pose_override = world
            .get::<BevyPoseOverride>(entity)
            .map(|pose| PoseOverrideData {
                enabled: pose.0.enabled,
                locals: pose_to_serialized(&pose.0.pose),
            });
        let freecam = world.get::<Freecam>(entity).is_some();
        let physics = world
            .get::<ColliderShape>(entity)
            .copied()
            .and_then(|shape| {
                if let Some(body) = world.get::<DynamicRigidBody>(entity) {
                    Some(PhysicsComponentData {
                        collider_shape: SceneColliderShape::from(shape),
                        body_kind: PhysicsBodyKind::Dynamic { mass: body.mass },
                    })
                } else if world.get::<FixedCollider>(entity).is_some() {
                    Some(PhysicsComponentData {
                        collider_shape: SceneColliderShape::from(shape),
                        body_kind: PhysicsBodyKind::Fixed,
                    })
                } else {
                    None
                }
            });

        let components = SceneComponents {
            mesh,
            skinned,
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
            physics,
        };

        entities.push(SceneEntityData {
            name: name.map(|name| name.to_string()),
            transform: serialized_transform,
            components,
        });
        entity_order.push(entity);
        saved_entities.insert(entity.to_bits());
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
            let Some(animation) = animation_component_from_group(group) else {
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

        if entity_data.components.freecam {
            entity.insert(Freecam::default());
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
                    Some(ScriptEntry {
                        path: Some(resolve_path(path, root)),
                        language: script.language.clone(),
                    })
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
            match physics.body_kind.clone() {
                PhysicsBodyKind::Dynamic { mass } => {
                    entity.insert(DynamicRigidBody { mass });
                }
                PhysicsBodyKind::Fixed => {
                    entity.insert(FixedCollider);
                }
            }
        }

        let entity_id = entity.id();
        drop(entity);

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
                    .map(AnimationClipData::to_clip)
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

    if !pending_followers.is_empty()
        || !pending_look_ats.is_empty()
        || !pending_entity_followers.is_empty()
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

    let mesh_asset = match kind {
        PrimitiveKind::Cube => helmer::provided::components::MeshAsset::cube("cube".to_string()),
        PrimitiveKind::UvSphere(segments, rings) => {
            helmer::provided::components::MeshAsset::uv_sphere(
                "uv sphere".to_string(),
                segments,
                rings,
            )
        }
        PrimitiveKind::Plane => helmer::provided::components::MeshAsset::plane("plane".to_string()),
    };

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
