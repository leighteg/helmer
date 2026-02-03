use bevy_ecs::prelude::{Commands, Entity, Query, Res, ResMut, Resource};
use glam::{Quat, Vec3};
use helmer::animation::{AnimationChannel, AnimationClip, Interpolation, Pose, Skeleton};
use helmer::provided::components::{Camera, Light, LightType, PoseOverride, Spline, Transform};
use helmer_becs::{BevyAnimator, BevyPoseOverride, BevySkinnedMeshRenderer, BevySpline, DeltaTime};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Resource)]
pub struct EditorTimelineState {
    pub playing: bool,
    pub loop_playback: bool,
    pub playback_rate: f32,
    pub current_time: f32,
    pub duration: f32,
    pub auto_duration: bool,
    pub frame_rate: f32,
    pub snap_to_frame: bool,
    pub smart_key: bool,
    pub pixels_per_second: f32,
    pub view_offset: f32,
    pub new_clip_index: usize,
    pub new_clip_looping: bool,
    pub new_clip_speed: f32,
    pub new_clip_name: String,
    pub groups: Vec<TimelineTrackGroup>,
    pub selected: Option<TimelineSelection>,
    pub apply_requested: bool,
    pub middle_drag_active: bool,
    pub(crate) next_id: u64,
}

impl Default for EditorTimelineState {
    fn default() -> Self {
        Self {
            playing: false,
            loop_playback: true,
            playback_rate: 1.0,
            current_time: 0.0,
            duration: 5.0,
            auto_duration: true,
            frame_rate: 30.0,
            snap_to_frame: true,
            smart_key: true,
            pixels_per_second: 120.0,
            view_offset: 0.0,
            new_clip_index: 0,
            new_clip_looping: true,
            new_clip_speed: 1.0,
            new_clip_name: "New Clip".to_string(),
            groups: Vec::new(),
            selected: None,
            apply_requested: true,
            middle_drag_active: false,
            next_id: 1,
        }
    }
}

impl EditorTimelineState {
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    pub fn request_apply(&mut self) {
        self.apply_requested = true;
    }

    pub fn ensure_group_index(&mut self, entity: u64, name: String) -> usize {
        if let Some(index) = self.groups.iter().position(|group| group.entity == entity) {
            return index;
        }
        self.groups.push(TimelineTrackGroup {
            entity,
            name,
            tracks: Vec::new(),
            custom_clips: Vec::new(),
        });
        self.groups.len().saturating_sub(1)
    }

    pub fn ensure_group(&mut self, entity: u64, name: String) -> &mut TimelineTrackGroup {
        let index = self.ensure_group_index(entity, name);
        &mut self.groups[index]
    }

    pub fn recompute_duration(&mut self) {
        if !self.auto_duration {
            return;
        }
        let mut max_time = 0.0f32;
        for group in &self.groups {
            for track in &group.tracks {
                max_time = max_time.max(track.end_time());
            }
        }
        if max_time > 0.0 {
            self.duration = max_time.max(self.duration);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TimelineTrackGroup {
    pub entity: u64,
    pub name: String,
    pub tracks: Vec<TimelineTrack>,
    pub custom_clips: Vec<AnimationClip>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimelineInterpolation {
    Step,
    Linear,
}

impl Default for TimelineInterpolation {
    fn default() -> Self {
        TimelineInterpolation::Linear
    }
}

#[derive(Debug, Clone)]
pub enum TimelineTrack {
    Pose(PoseTrack),
    Joint(JointTrack),
    Transform(TransformTrack),
    Camera(CameraTrack),
    Light(LightTrack),
    Spline(SplineTrack),
    Clip(ClipTrack),
}

impl TimelineTrack {
    pub fn id(&self) -> u64 {
        match self {
            TimelineTrack::Pose(track) => track.id,
            TimelineTrack::Joint(track) => track.id,
            TimelineTrack::Transform(track) => track.id,
            TimelineTrack::Camera(track) => track.id,
            TimelineTrack::Light(track) => track.id,
            TimelineTrack::Spline(track) => track.id,
            TimelineTrack::Clip(track) => track.id,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            TimelineTrack::Pose(track) => track.name.as_str(),
            TimelineTrack::Joint(track) => track.name.as_str(),
            TimelineTrack::Transform(track) => track.name.as_str(),
            TimelineTrack::Camera(track) => track.name.as_str(),
            TimelineTrack::Light(track) => track.name.as_str(),
            TimelineTrack::Spline(track) => track.name.as_str(),
            TimelineTrack::Clip(track) => track.name.as_str(),
        }
    }

    pub fn enabled(&self) -> bool {
        match self {
            TimelineTrack::Pose(track) => track.enabled,
            TimelineTrack::Joint(track) => track.enabled,
            TimelineTrack::Transform(track) => track.enabled,
            TimelineTrack::Camera(track) => track.enabled,
            TimelineTrack::Light(track) => track.enabled,
            TimelineTrack::Spline(track) => track.enabled,
            TimelineTrack::Clip(track) => track.enabled,
        }
    }

    pub fn end_time(&self) -> f32 {
        match self {
            TimelineTrack::Pose(track) => track.keys.last().map(|key| key.time).unwrap_or(0.0),
            TimelineTrack::Joint(track) => track.keys.last().map(|key| key.time).unwrap_or(0.0),
            TimelineTrack::Transform(track) => track.keys.last().map(|key| key.time).unwrap_or(0.0),
            TimelineTrack::Camera(track) => track.keys.last().map(|key| key.time).unwrap_or(0.0),
            TimelineTrack::Light(track) => track.keys.last().map(|key| key.time).unwrap_or(0.0),
            TimelineTrack::Spline(track) => track.keys.last().map(|key| key.time).unwrap_or(0.0),
            TimelineTrack::Clip(track) => track
                .segments
                .iter()
                .map(|segment| segment.start + segment.duration)
                .fold(0.0, f32::max),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoseTrack {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub weight: f32,
    pub additive: bool,
    pub translation_interpolation: TimelineInterpolation,
    pub rotation_interpolation: TimelineInterpolation,
    pub scale_interpolation: TimelineInterpolation,
    pub keys: Vec<PoseKey>,
}

#[derive(Debug, Clone)]
pub struct PoseKey {
    pub id: u64,
    pub time: f32,
    pub pose: Pose,
}

#[derive(Debug, Clone)]
pub struct JointTrack {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub joint_index: usize,
    pub weight: f32,
    pub additive: bool,
    pub translation_interpolation: TimelineInterpolation,
    pub rotation_interpolation: TimelineInterpolation,
    pub scale_interpolation: TimelineInterpolation,
    pub keys: Vec<JointKey>,
}

#[derive(Debug, Clone)]
pub struct JointKey {
    pub id: u64,
    pub time: f32,
    pub transform: Transform,
}

#[derive(Debug, Clone)]
pub struct TransformTrack {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub translation_interpolation: TimelineInterpolation,
    pub rotation_interpolation: TimelineInterpolation,
    pub scale_interpolation: TimelineInterpolation,
    pub keys: Vec<TransformKey>,
}

#[derive(Debug, Clone)]
pub struct TransformKey {
    pub id: u64,
    pub time: f32,
    pub transform: Transform,
}

#[derive(Debug, Clone)]
pub struct CameraTrack {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub interpolation: TimelineInterpolation,
    pub keys: Vec<CameraKey>,
}

#[derive(Debug, Clone)]
pub struct CameraKey {
    pub id: u64,
    pub time: f32,
    pub camera: Camera,
}

#[derive(Debug, Clone)]
pub struct LightTrack {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub interpolation: TimelineInterpolation,
    pub keys: Vec<LightKey>,
}

#[derive(Debug, Clone)]
pub struct LightKey {
    pub id: u64,
    pub time: f32,
    pub light: Light,
}

#[derive(Debug, Clone)]
pub struct SplineTrack {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub interpolation: TimelineInterpolation,
    pub keys: Vec<SplineKey>,
}

#[derive(Debug, Clone)]
pub struct SplineKey {
    pub id: u64,
    pub time: f32,
    pub spline: Spline,
}

#[derive(Debug, Clone)]
pub struct ClipTrack {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    pub weight: f32,
    pub additive: bool,
    pub segments: Vec<ClipSegment>,
}

#[derive(Debug, Clone)]
pub struct ClipSegment {
    pub id: u64,
    pub start: f32,
    pub duration: f32,
    pub clip_name: String,
    pub speed: f32,
    pub looping: bool,
}

#[derive(Debug, Clone)]
pub enum TimelineSelection {
    Key { track_id: u64, key_id: u64 },
    Clip { track_id: u64, segment_id: u64 },
}

pub fn timeline_playback_system(
    time: Res<DeltaTime>,
    mut timeline: ResMut<EditorTimelineState>,
    mut commands: Commands,
    mut pose_overrides: Query<&mut BevyPoseOverride>,
    mut transform_query: Query<&mut helmer_becs::BevyTransform>,
    mut camera_query: Query<&mut helmer_becs::BevyCamera>,
    mut light_query: Query<&mut helmer_becs::BevyLight>,
    mut spline_query: Query<&mut BevySpline>,
    skinned_query: Query<&BevySkinnedMeshRenderer>,
    animator_query: Query<&BevyAnimator>,
) {
    let dt = time.0;
    if timeline.playing {
        timeline.current_time += dt * timeline.playback_rate.max(0.0);
        if timeline.loop_playback && timeline.duration > 0.0 {
            timeline.current_time = timeline.current_time.rem_euclid(timeline.duration);
        } else {
            timeline.current_time = timeline.current_time.min(timeline.duration);
        }
    }

    if !timeline.playing && !timeline.apply_requested {
        return;
    }
    timeline.apply_requested = false;

    timeline.recompute_duration();

    let time_cursor = timeline.current_time;
    for group in &timeline.groups {
        let entity = Entity::from_bits(group.entity);

        let mut transform_sample = None;
        let mut camera_sample = None;
        let mut light_sample = None;
        let mut spline_sample = None;
        for track in &group.tracks {
            if !track.enabled() {
                continue;
            }
            match track {
                TimelineTrack::Transform(track) => {
                    if let Some(sample) = sample_transform_track(track, time_cursor) {
                        transform_sample = Some(sample);
                    }
                }
                TimelineTrack::Camera(track) => {
                    if let Some(sample) = sample_camera_track(track, time_cursor) {
                        camera_sample = Some(sample);
                    }
                }
                TimelineTrack::Light(track) => {
                    if let Some(sample) = sample_light_track(track, time_cursor) {
                        light_sample = Some(sample);
                    }
                }
                TimelineTrack::Spline(track) => {
                    if let Some(sample) = sample_spline_track(track, time_cursor) {
                        spline_sample = Some(sample);
                    }
                }
                _ => {}
            }
        }

        if let Some(sample) = transform_sample {
            if let Ok(mut transform) = transform_query.get_mut(entity) {
                transform.0 = sample;
            }
        }
        if let Some(sample) = camera_sample {
            if let Ok(mut camera) = camera_query.get_mut(entity) {
                camera.0 = sample;
            }
        }
        if let Some(sample) = light_sample {
            if let Ok(mut light) = light_query.get_mut(entity) {
                light.0 = sample;
            }
        }

        if let Ok(skinned) = skinned_query.get(entity) {
            let skeleton = &skinned.0.skin.skeleton;
            let mut base_pose = Pose::from_skeleton(skeleton);
            let mut has_pose = false;
            for track in &group.tracks {
                if !track.enabled() {
                    continue;
                }
                match track {
                    TimelineTrack::Pose(track) => {
                        if let Some(pose) = sample_pose_track(track, time_cursor) {
                            if track.additive {
                                apply_additive_pose(&mut base_pose, &pose, skeleton, track.weight);
                            } else {
                                blend_pose(&mut base_pose, &pose, track.weight);
                            }
                            has_pose = true;
                        }
                    }
                    TimelineTrack::Joint(track) => {
                        if let Some(sample) = sample_joint_track(track, time_cursor) {
                            apply_joint_sample(&mut base_pose, skeleton, track, sample);
                            has_pose = true;
                        }
                    }
                    TimelineTrack::Clip(track) => {
                        let animator = animator_query.get(entity).ok();
                        if let Some(pose) =
                            sample_clip_track(track, animator, skeleton, time_cursor)
                        {
                            if track.additive {
                                apply_additive_pose(&mut base_pose, &pose, skeleton, track.weight);
                            } else {
                                blend_pose(&mut base_pose, &pose, track.weight);
                            }
                            has_pose = true;
                        }
                    }
                    _ => {}
                }
            }

            if has_pose {
                if let Ok(mut override_pose) = pose_overrides.get_mut(entity) {
                    override_pose.0.enabled = true;
                    override_pose.0.pose = base_pose;
                } else {
                    commands
                        .entity(entity)
                        .try_insert(BevyPoseOverride(PoseOverride {
                            enabled: true,
                            pose: base_pose,
                        }));
                }
            }
        }

        if let Some(sample) = spline_sample {
            if let Ok(mut spline) = spline_query.get_mut(entity) {
                spline.0 = sample;
            }
        }
    }
}

fn sample_pose_track(track: &PoseTrack, time: f32) -> Option<Pose> {
    if track.keys.is_empty() {
        return None;
    }
    if track.keys.len() == 1 {
        return Some(track.keys[0].pose.clone());
    }
    let keys = track.keys.as_slice();
    if time <= keys[0].time {
        return Some(keys[0].pose.clone());
    }
    if time >= keys[keys.len() - 1].time {
        return Some(keys[keys.len() - 1].pose.clone());
    }
    for i in 0..keys.len() - 1 {
        let a = &keys[i];
        let b = &keys[i + 1];
        if time >= a.time && time <= b.time {
            let span = (b.time - a.time).max(0.0001);
            let t = ((time - a.time) / span).clamp(0.0, 1.0);
            return Some(interpolate_pose(track, &a.pose, &b.pose, t));
        }
    }
    None
}

fn sample_joint_track(track: &JointTrack, time: f32) -> Option<Transform> {
    if track.keys.is_empty() {
        return None;
    }
    if track.keys.len() == 1 {
        return Some(track.keys[0].transform);
    }
    if time <= track.keys[0].time {
        return Some(track.keys[0].transform);
    }
    if time >= track.keys[track.keys.len() - 1].time {
        return Some(track.keys[track.keys.len() - 1].transform);
    }
    for i in 0..track.keys.len() - 1 {
        let a = &track.keys[i];
        let b = &track.keys[i + 1];
        if time >= a.time && time <= b.time {
            let span = (b.time - a.time).max(0.0001);
            let t = ((time - a.time) / span).clamp(0.0, 1.0);
            let step_translation =
                matches!(track.translation_interpolation, TimelineInterpolation::Step);
            let step_rotation = matches!(track.rotation_interpolation, TimelineInterpolation::Step);
            let step_scale = matches!(track.scale_interpolation, TimelineInterpolation::Step);
            let position = if step_translation {
                a.transform.position
            } else {
                a.transform.position.lerp(b.transform.position, t)
            };
            let rotation = if step_rotation {
                a.transform.rotation
            } else {
                a.transform
                    .rotation
                    .slerp(b.transform.rotation, t)
                    .normalize()
            };
            let scale = if step_scale {
                a.transform.scale
            } else {
                a.transform.scale.lerp(b.transform.scale, t)
            };
            return Some(Transform {
                position,
                rotation,
                scale,
            });
        }
    }
    None
}

fn sample_spline_track(track: &SplineTrack, time: f32) -> Option<Spline> {
    if track.keys.is_empty() {
        return None;
    }
    if track.keys.len() == 1 {
        return Some(track.keys[0].spline.clone());
    }
    if time <= track.keys[0].time {
        return Some(track.keys[0].spline.clone());
    }
    if time >= track.keys[track.keys.len() - 1].time {
        return Some(track.keys[track.keys.len() - 1].spline.clone());
    }
    for i in 0..track.keys.len() - 1 {
        let a = &track.keys[i];
        let b = &track.keys[i + 1];
        if time >= a.time && time <= b.time {
            let span = (b.time - a.time).max(0.0001);
            let t = ((time - a.time) / span).clamp(0.0, 1.0);
            if matches!(track.interpolation, TimelineInterpolation::Step) {
                return Some(a.spline.clone());
            }
            if a.spline.points.len() != b.spline.points.len() {
                return Some(if t < 0.5 {
                    a.spline.clone()
                } else {
                    b.spline.clone()
                });
            }
            let mut out = a.spline.clone();
            out.mode = a.spline.mode;
            out.closed = if t < 0.5 {
                a.spline.closed
            } else {
                b.spline.closed
            };
            out.tension = a.spline.tension + (b.spline.tension - a.spline.tension) * t;
            for (idx, point) in out.points.iter_mut().enumerate() {
                *point = a.spline.points[idx].lerp(b.spline.points[idx], t);
            }
            return Some(out);
        }
    }
    None
}

fn sample_transform_track(track: &TransformTrack, time: f32) -> Option<Transform> {
    if track.keys.is_empty() {
        return None;
    }
    if track.keys.len() == 1 {
        return Some(track.keys[0].transform);
    }
    if time <= track.keys[0].time {
        return Some(track.keys[0].transform);
    }
    if time >= track.keys[track.keys.len() - 1].time {
        return Some(track.keys[track.keys.len() - 1].transform);
    }
    for i in 0..track.keys.len() - 1 {
        let a = &track.keys[i];
        let b = &track.keys[i + 1];
        if time >= a.time && time <= b.time {
            let span = (b.time - a.time).max(0.0001);
            let t = ((time - a.time) / span).clamp(0.0, 1.0);
            let step_translation =
                matches!(track.translation_interpolation, TimelineInterpolation::Step);
            let step_rotation = matches!(track.rotation_interpolation, TimelineInterpolation::Step);
            let step_scale = matches!(track.scale_interpolation, TimelineInterpolation::Step);
            let position = if step_translation {
                a.transform.position
            } else {
                a.transform.position.lerp(b.transform.position, t)
            };
            let rotation = if step_rotation {
                a.transform.rotation
            } else {
                a.transform
                    .rotation
                    .slerp(b.transform.rotation, t)
                    .normalize()
            };
            let scale = if step_scale {
                a.transform.scale
            } else {
                a.transform.scale.lerp(b.transform.scale, t)
            };
            return Some(Transform {
                position,
                rotation,
                scale,
            });
        }
    }
    None
}

fn sample_camera_track(track: &CameraTrack, time: f32) -> Option<Camera> {
    if track.keys.is_empty() {
        return None;
    }
    if track.keys.len() == 1 {
        return Some(track.keys[0].camera);
    }
    if time <= track.keys[0].time {
        return Some(track.keys[0].camera);
    }
    if time >= track.keys[track.keys.len() - 1].time {
        return Some(track.keys[track.keys.len() - 1].camera);
    }
    for i in 0..track.keys.len() - 1 {
        let a = &track.keys[i];
        let b = &track.keys[i + 1];
        if time >= a.time && time <= b.time {
            let span = (b.time - a.time).max(0.0001);
            let t = ((time - a.time) / span).clamp(0.0, 1.0);
            if matches!(track.interpolation, TimelineInterpolation::Step) {
                return Some(a.camera);
            }
            return Some(Camera {
                fov_y_rad: a.camera.fov_y_rad + (b.camera.fov_y_rad - a.camera.fov_y_rad) * t,
                aspect_ratio: a.camera.aspect_ratio
                    + (b.camera.aspect_ratio - a.camera.aspect_ratio) * t,
                near_plane: a.camera.near_plane + (b.camera.near_plane - a.camera.near_plane) * t,
                far_plane: a.camera.far_plane + (b.camera.far_plane - a.camera.far_plane) * t,
            });
        }
    }
    None
}

fn sample_light_track(track: &LightTrack, time: f32) -> Option<Light> {
    if track.keys.is_empty() {
        return None;
    }
    if track.keys.len() == 1 {
        return Some(track.keys[0].light);
    }
    if time <= track.keys[0].time {
        return Some(track.keys[0].light);
    }
    if time >= track.keys[track.keys.len() - 1].time {
        return Some(track.keys[track.keys.len() - 1].light);
    }
    for i in 0..track.keys.len() - 1 {
        let a = &track.keys[i];
        let b = &track.keys[i + 1];
        if time >= a.time && time <= b.time {
            let span = (b.time - a.time).max(0.0001);
            let t = ((time - a.time) / span).clamp(0.0, 1.0);
            if matches!(track.interpolation, TimelineInterpolation::Step) {
                return Some(a.light);
            }
            let type_matches = std::mem::discriminant(&a.light.light_type)
                == std::mem::discriminant(&b.light.light_type);
            if !type_matches {
                return Some(if t < 0.5 { a.light } else { b.light });
            }
            let angle = match (a.light.light_type, b.light.light_type) {
                (LightType::Spot { angle: a_angle }, LightType::Spot { angle: b_angle }) => {
                    Some(a_angle + (b_angle - a_angle) * t)
                }
                _ => None,
            };
            let light_type = match a.light.light_type {
                LightType::Directional => LightType::Directional,
                LightType::Point => LightType::Point,
                LightType::Spot { .. } => LightType::Spot {
                    angle: angle.unwrap_or(0.0),
                },
            };
            let color = a.light.color.lerp(b.light.color, t);
            let intensity = a.light.intensity + (b.light.intensity - a.light.intensity) * t;
            return Some(Light {
                light_type,
                color,
                intensity,
            });
        }
    }
    None
}

fn sample_clip_track(
    track: &ClipTrack,
    animator: Option<&BevyAnimator>,
    skeleton: &Skeleton,
    time: f32,
) -> Option<Pose> {
    let animator = animator?;
    let library = animator
        .0
        .layers
        .first()
        .map(|layer| &layer.graph.library)?;
    let segment = track
        .segments
        .iter()
        .find(|segment| time >= segment.start && time <= segment.start + segment.duration)?;
    let clip_index = library.clip_index(segment.clip_name.as_str())?;
    let clip = library.clip(clip_index)?;
    let mut pose = Pose::from_skeleton(skeleton);
    let mut clip_time = (time - segment.start) * segment.speed.max(0.0001);
    if segment.looping && clip.duration > 0.0 {
        clip_time = clip_time.rem_euclid(clip.duration);
    } else {
        clip_time = clip_time.clamp(0.0, clip.duration);
    }
    clip.sample_pose(clip_time, skeleton, &mut pose);
    Some(pose)
}

fn interpolate_pose(track: &PoseTrack, a: &Pose, b: &Pose, t: f32) -> Pose {
    let count = a.locals.len().min(b.locals.len());
    let mut out = Pose {
        locals: Vec::with_capacity(count),
    };
    let step_translation = matches!(track.translation_interpolation, TimelineInterpolation::Step);
    let step_rotation = matches!(track.rotation_interpolation, TimelineInterpolation::Step);
    let step_scale = matches!(track.scale_interpolation, TimelineInterpolation::Step);
    for idx in 0..count {
        let ta = a.locals[idx];
        let tb = b.locals[idx];
        let position = if step_translation {
            ta.position
        } else {
            ta.position.lerp(tb.position, t)
        };
        let rotation = if step_rotation {
            ta.rotation
        } else {
            ta.rotation.slerp(tb.rotation, t).normalize()
        };
        let scale = if step_scale {
            ta.scale
        } else {
            ta.scale.lerp(tb.scale, t)
        };
        out.locals.push(Transform {
            position,
            rotation,
            scale,
        });
    }
    out
}

fn blend_pose(target: &mut Pose, source: &Pose, weight: f32) {
    let weight = weight.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return;
    }
    let count = target.locals.len().min(source.locals.len());
    for idx in 0..count {
        let a = target.locals[idx];
        let b = source.locals[idx];
        target.locals[idx] = Transform {
            position: a.position.lerp(b.position, weight),
            rotation: a.rotation.slerp(b.rotation, weight).normalize(),
            scale: a.scale.lerp(b.scale, weight),
        };
    }
}

fn apply_joint_sample(
    base_pose: &mut Pose,
    skeleton: &Skeleton,
    track: &JointTrack,
    sample: Transform,
) {
    let joint_index = track.joint_index;
    if joint_index >= base_pose.locals.len() {
        return;
    }
    let weight = track.weight.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return;
    }
    let current = base_pose.locals[joint_index];
    if track.additive {
        let bind = skeleton
            .joints
            .get(joint_index)
            .map(|joint| joint.bind_transform)
            .unwrap_or_default();
        let delta_pos = sample.position - bind.position;
        let delta_scale = sample.scale - bind.scale;
        let delta_rot = bind.rotation.inverse() * sample.rotation;
        base_pose.locals[joint_index] = Transform {
            position: current.position + delta_pos * weight,
            rotation: current.rotation * Quat::IDENTITY.slerp(delta_rot, weight),
            scale: current.scale + delta_scale * weight,
        };
    } else {
        base_pose.locals[joint_index] = Transform {
            position: current.position.lerp(sample.position, weight),
            rotation: current.rotation.slerp(sample.rotation, weight).normalize(),
            scale: current.scale.lerp(sample.scale, weight),
        };
    }
}

fn apply_additive_pose(target: &mut Pose, additive: &Pose, skeleton: &Skeleton, weight: f32) {
    let weight = weight.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return;
    }
    let count = target.locals.len().min(additive.locals.len());
    for idx in 0..count {
        let base = skeleton
            .joints
            .get(idx)
            .map(|joint| joint.bind_transform)
            .unwrap_or_default();
        let add = additive.locals[idx];
        let delta_pos = add.position - base.position;
        let delta_scale = add.scale - base.scale;
        let delta_rot = base.rotation.inverse() * add.rotation;
        let current = target.locals[idx];
        target.locals[idx] = Transform {
            position: current.position + delta_pos * weight,
            rotation: current.rotation * Quat::IDENTITY.slerp(delta_rot, weight),
            scale: current.scale + delta_scale * weight,
        };
    }
}

pub fn build_clip_from_pose_track(
    name: String,
    track: &PoseTrack,
    skeleton: &Skeleton,
) -> Option<AnimationClip> {
    if track.keys.is_empty() {
        return None;
    }
    let joint_count = skeleton.joint_count();
    let mut translation_channels = vec![Vec::new(); joint_count];
    let mut rotation_channels = vec![Vec::new(); joint_count];
    let mut scale_channels = vec![Vec::new(); joint_count];

    for key in &track.keys {
        for joint_index in 0..joint_count {
            let transform = key
                .pose
                .locals
                .get(joint_index)
                .copied()
                .unwrap_or_default();
            translation_channels[joint_index].push((key.time, transform.position));
            rotation_channels[joint_index].push((key.time, transform.rotation));
            scale_channels[joint_index].push((key.time, transform.scale));
        }
    }

    let mut channels = Vec::new();
    let translation_interp = match track.translation_interpolation {
        TimelineInterpolation::Step => helmer::animation::Interpolation::Step,
        TimelineInterpolation::Linear => helmer::animation::Interpolation::Linear,
    };
    let rotation_interp = match track.rotation_interpolation {
        TimelineInterpolation::Step => helmer::animation::Interpolation::Step,
        TimelineInterpolation::Linear => helmer::animation::Interpolation::Linear,
    };
    let scale_interp = match track.scale_interpolation {
        TimelineInterpolation::Step => helmer::animation::Interpolation::Step,
        TimelineInterpolation::Linear => helmer::animation::Interpolation::Linear,
    };

    for joint_index in 0..joint_count {
        if !translation_channels[joint_index].is_empty() {
            channels.push(helmer::animation::AnimationChannel::Translation {
                target: joint_index,
                interpolation: translation_interp,
                keyframes: translation_channels[joint_index]
                    .iter()
                    .map(|(time, value)| helmer::animation::Keyframe::new(*time, *value))
                    .collect(),
            });
        }
        if !rotation_channels[joint_index].is_empty() {
            channels.push(helmer::animation::AnimationChannel::Rotation {
                target: joint_index,
                interpolation: rotation_interp,
                keyframes: rotation_channels[joint_index]
                    .iter()
                    .map(|(time, value)| helmer::animation::Keyframe::new(*time, *value))
                    .collect(),
            });
        }
        if !scale_channels[joint_index].is_empty() {
            channels.push(helmer::animation::AnimationChannel::Scale {
                target: joint_index,
                interpolation: scale_interp,
                keyframes: scale_channels[joint_index]
                    .iter()
                    .map(|(time, value)| helmer::animation::Keyframe::new(*time, *value))
                    .collect(),
            });
        }
    }

    let duration = track.keys.last().map(|key| key.time).unwrap_or(0.0);

    Some(AnimationClip {
        name,
        duration,
        channels,
    })
}

pub fn build_pose_track_from_clip<F: FnMut() -> u64>(
    id: u64,
    name: String,
    clip: &AnimationClip,
    skeleton: &Skeleton,
    alloc_id: &mut F,
) -> PoseTrack {
    let mut times = Vec::new();
    let mut translation_interp = TimelineInterpolation::Step;
    let mut rotation_interp = TimelineInterpolation::Step;
    let mut scale_interp = TimelineInterpolation::Step;
    let mut has_translation = false;
    let mut has_rotation = false;
    let mut has_scale = false;

    for channel in &clip.channels {
        match channel {
            AnimationChannel::Translation {
                interpolation,
                keyframes,
                ..
            } => {
                has_translation = true;
                if !matches!(interpolation, Interpolation::Step) {
                    translation_interp = TimelineInterpolation::Linear;
                }
                times.extend(keyframes.iter().map(|key| key.time));
            }
            AnimationChannel::Rotation {
                interpolation,
                keyframes,
                ..
            } => {
                has_rotation = true;
                if !matches!(interpolation, Interpolation::Step) {
                    rotation_interp = TimelineInterpolation::Linear;
                }
                times.extend(keyframes.iter().map(|key| key.time));
            }
            AnimationChannel::Scale {
                interpolation,
                keyframes,
                ..
            } => {
                has_scale = true;
                if !matches!(interpolation, Interpolation::Step) {
                    scale_interp = TimelineInterpolation::Linear;
                }
                times.extend(keyframes.iter().map(|key| key.time));
            }
        }
    }

    if !has_translation {
        translation_interp = TimelineInterpolation::Linear;
    }
    if !has_rotation {
        rotation_interp = TimelineInterpolation::Linear;
    }
    if !has_scale {
        scale_interp = TimelineInterpolation::Linear;
    }

    times.push(0.0);
    if clip.duration > 0.0 {
        times.push(clip.duration);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique_times: Vec<f32> = Vec::new();
    for time in times {
        if unique_times
            .last()
            .map(|last| (time - *last).abs() > 1.0e-4_f32)
            .unwrap_or(true)
        {
            unique_times.push(time);
        }
    }

    let mut keys = Vec::with_capacity(unique_times.len());
    for time in unique_times {
        let mut pose = Pose::from_skeleton(skeleton);
        clip.sample_pose(time, skeleton, &mut pose);
        keys.push(PoseKey {
            id: alloc_id(),
            time,
            pose,
        });
    }

    PoseTrack {
        id,
        name,
        enabled: true,
        weight: 1.0,
        additive: false,
        translation_interpolation: translation_interp,
        rotation_interpolation: rotation_interp,
        scale_interpolation: scale_interp,
        keys,
    }
}
