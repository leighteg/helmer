use glam::{Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::provided::components::Transform;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    Step,
    Linear,
    Cubic,
}

#[derive(Debug, Clone)]
pub struct Keyframe<T> {
    pub time: f32,
    pub value: T,
    pub in_tangent: Option<T>,
    pub out_tangent: Option<T>,
}

impl<T> Keyframe<T> {
    pub fn new(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            in_tangent: None,
            out_tangent: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AnimationChannel {
    Translation {
        target: usize,
        interpolation: Interpolation,
        keyframes: Vec<Keyframe<Vec3>>,
    },
    Rotation {
        target: usize,
        interpolation: Interpolation,
        keyframes: Vec<Keyframe<Quat>>,
    },
    Scale {
        target: usize,
        interpolation: Interpolation,
        keyframes: Vec<Keyframe<Vec3>>,
    },
}

impl AnimationChannel {
    pub fn target(&self) -> usize {
        match self {
            Self::Translation { target, .. } => *target,
            Self::Rotation { target, .. } => *target,
            Self::Scale { target, .. } => *target,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

impl AnimationClip {
    pub fn sample_pose(&self, time: f32, skeleton: &Skeleton, pose: &mut Pose) {
        pose.reset_to_bind(skeleton);
        if self.channels.is_empty() {
            return;
        }
        let time = if self.duration > 0.0 {
            time.rem_euclid(self.duration)
        } else {
            0.0
        };
        for channel in &self.channels {
            let target = channel.target();
            if target >= pose.locals.len() {
                continue;
            }
            let transform = &mut pose.locals[target];
            match channel {
                AnimationChannel::Translation {
                    interpolation,
                    keyframes,
                    ..
                } => {
                    let value = sample_vec3(keyframes, *interpolation, time);
                    transform.position = value;
                }
                AnimationChannel::Rotation {
                    interpolation,
                    keyframes,
                    ..
                } => {
                    let value = sample_quat(keyframes, *interpolation, time);
                    transform.rotation = value;
                }
                AnimationChannel::Scale {
                    interpolation,
                    keyframes,
                    ..
                } => {
                    let value = sample_vec3(keyframes, *interpolation, time);
                    transform.scale = value;
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Joint {
    pub name: String,
    pub parent: Option<usize>,
    pub bind_transform: Transform,
    pub inverse_bind: Mat4,
}

#[derive(Debug, Clone)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
    pub root_joints: Vec<usize>,
}

impl Skeleton {
    pub fn new(joints: Vec<Joint>) -> Self {
        let mut root_joints = Vec::new();
        for (idx, joint) in joints.iter().enumerate() {
            if joint.parent.is_none() {
                root_joints.push(idx);
            }
        }
        Self {
            joints,
            root_joints,
        }
    }

    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
}

#[derive(Debug, Clone)]
pub struct Skin {
    pub name: String,
    pub skeleton: Arc<Skeleton>,
    pub joint_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Pose {
    pub locals: Vec<Transform>,
}

impl Pose {
    pub fn from_skeleton(skeleton: &Skeleton) -> Self {
        let locals = skeleton
            .joints
            .iter()
            .map(|joint| joint.bind_transform)
            .collect();
        Self { locals }
    }

    pub fn reset_to_bind(&mut self, skeleton: &Skeleton) {
        if self.locals.len() != skeleton.joints.len() {
            self.locals = skeleton
                .joints
                .iter()
                .map(|joint| joint.bind_transform)
                .collect();
        } else {
            for (local, joint) in self.locals.iter_mut().zip(skeleton.joints.iter()) {
                *local = joint.bind_transform;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Linear,
    Additive,
}

#[derive(Debug, Clone)]
pub struct BlendChild {
    pub node: usize,
    pub weight: f32,
    pub weight_param: Option<String>,
    pub weight_scale: f32,
    pub weight_bias: f32,
}

impl BlendChild {
    fn resolved_weight(&self, params: Option<&AnimationParameters>) -> f32 {
        let base = self
            .weight_param
            .as_deref()
            .and_then(|name| params.map(|params| params.get_float(name, self.weight)))
            .unwrap_or(self.weight);
        (base * self.weight_scale) + self.weight_bias
    }
}

#[derive(Debug, Clone)]
pub struct BlendNode {
    pub children: Vec<BlendChild>,
    pub normalize: bool,
    pub mode: BlendMode,
}

#[derive(Debug, Clone)]
pub struct ClipNode {
    pub clip_index: usize,
    pub speed: f32,
    pub looping: bool,
    pub time_offset: f32,
}

#[derive(Debug, Clone)]
pub enum AnimationNode {
    Clip(ClipNode),
    Blend(BlendNode),
}

#[derive(Debug, Clone)]
pub struct AnimationGraph {
    pub library: Arc<AnimationLibrary>,
    pub nodes: Vec<AnimationNode>,
}

impl AnimationGraph {
    pub fn evaluate_node(
        &self,
        node_index: usize,
        skeleton: &Skeleton,
        time: f32,
        out: &mut Pose,
        scratch: &mut Pose,
    ) {
        self.evaluate_node_with_params(node_index, skeleton, time, out, scratch, None);
    }

    pub fn evaluate_node_with_params(
        &self,
        node_index: usize,
        skeleton: &Skeleton,
        time: f32,
        out: &mut Pose,
        scratch: &mut Pose,
        params: Option<&AnimationParameters>,
    ) {
        self.evaluate_node_internal(node_index, skeleton, time, out, scratch, params);
    }

    fn evaluate_node_internal(
        &self,
        node_index: usize,
        skeleton: &Skeleton,
        time: f32,
        out: &mut Pose,
        scratch: &mut Pose,
        params: Option<&AnimationParameters>,
    ) {
        let Some(node) = self.nodes.get(node_index) else {
            out.reset_to_bind(skeleton);
            return;
        };
        match node {
            AnimationNode::Clip(clip_node) => {
                if let Some(clip) = self.library.clip(clip_node.clip_index) {
                    let clip_time = time * clip_node.speed + clip_node.time_offset;
                    let clip_time = if clip_node.looping && clip.duration > 0.0 {
                        clip_time.rem_euclid(clip.duration)
                    } else {
                        clip_time.clamp(0.0, clip.duration)
                    };
                    clip.sample_pose(clip_time, skeleton, out);
                } else {
                    out.reset_to_bind(skeleton);
                }
            }
            AnimationNode::Blend(blend_node) => {
                out.reset_to_bind(skeleton);
                if blend_node.children.is_empty() {
                    return;
                }

                let mut total_weight = 0.0f32;
                for child in &blend_node.children {
                    total_weight += child.resolved_weight(params).max(0.0);
                }
                if total_weight <= f32::EPSILON {
                    return;
                }

                if blend_node.mode == BlendMode::Additive {
                    for child in &blend_node.children {
                        let weight = if blend_node.normalize {
                            (child.resolved_weight(params).max(0.0) / total_weight).clamp(0.0, 1.0)
                        } else {
                            child.resolved_weight(params).max(0.0)
                        };
                        if weight <= f32::EPSILON {
                            continue;
                        }
                        self.evaluate_node_internal(
                            child.node, skeleton, time, scratch, out, params,
                        );
                        apply_additive_pose(out, scratch, skeleton, weight, None);
                    }
                    return;
                }

                if blend_node.normalize {
                    let mut accumulated_weight = 0.0f32;
                    let mut initialized = false;
                    for child in &blend_node.children {
                        let weight =
                            (child.resolved_weight(params).max(0.0) / total_weight).clamp(0.0, 1.0);
                        if weight <= f32::EPSILON {
                            continue;
                        }
                        self.evaluate_node_internal(
                            child.node, skeleton, time, scratch, out, params,
                        );
                        if !initialized {
                            out.locals.clear();
                            out.locals.extend_from_slice(&scratch.locals);
                            accumulated_weight = weight;
                            initialized = true;
                            continue;
                        }
                        let blend_factor = weight / (accumulated_weight + weight);
                        blend_pose(out, scratch, blend_factor, None);
                        accumulated_weight += weight;
                    }
                    return;
                }

                for (index, child) in blend_node.children.iter().enumerate() {
                    let weight = child.resolved_weight(params).max(0.0);
                    if weight <= 0.0 {
                        continue;
                    }
                    self.evaluate_node_internal(child.node, skeleton, time, scratch, out, params);
                    if index == 0 {
                        out.locals.clear();
                        out.locals.extend_from_slice(&scratch.locals);
                    } else {
                        blend_pose(out, scratch, weight, None);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationComparison {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone)]
pub enum AnimationCondition {
    Bool {
        param: String,
        value: bool,
    },
    Float {
        param: String,
        comparison: AnimationComparison,
        value: f32,
    },
    Trigger {
        param: String,
    },
}

#[derive(Debug, Clone)]
pub struct AnimationTransition {
    pub from: usize,
    pub to: usize,
    pub duration: f32,
    pub exit_time: Option<f32>,
    pub conditions: Vec<AnimationCondition>,
    pub can_interrupt: bool,
}

#[derive(Debug, Clone)]
pub struct AnimationState {
    pub name: String,
    pub node: usize,
}

#[derive(Debug, Clone)]
struct ActiveTransition {
    from: usize,
    to: usize,
    elapsed: f32,
    duration: f32,
    can_interrupt: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct AnimationTransitionStatus {
    pub from: usize,
    pub to: usize,
    pub elapsed: f32,
    pub duration: f32,
    pub progress: f32,
    pub can_interrupt: bool,
}

#[derive(Debug, Clone)]
pub struct AnimationStateMachine {
    pub states: Vec<AnimationState>,
    pub transitions: Vec<AnimationTransition>,
    pub default_state: usize,
    pub current_state: usize,
    pub state_time: f32,
    active_transition: Option<ActiveTransition>,
}

impl AnimationStateMachine {
    pub fn new(states: Vec<AnimationState>, transitions: Vec<AnimationTransition>) -> Self {
        let default_state = 0;
        Self {
            states,
            transitions,
            default_state,
            current_state: default_state,
            state_time: 0.0,
            active_transition: None,
        }
    }

    pub fn reset(&mut self) {
        self.current_state = self.default_state.min(self.states.len().saturating_sub(1));
        self.state_time = 0.0;
        self.active_transition = None;
    }

    pub fn update(
        &mut self,
        dt: f32,
        params: &mut AnimationParameters,
        graph: &AnimationGraph,
    ) -> (usize, Option<(usize, f32)>) {
        self.state_time = (self.state_time + dt).max(0.0);
        if let Some(active) = self.active_transition.as_mut() {
            active.elapsed += dt;
            if active.can_interrupt {
                let active_to = active.to;
                let active_from = active.from;
                let active_duration = active.duration.max(0.0);
                let active_progress = if active_duration > 0.0 {
                    (active.elapsed / active_duration).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let candidate_state_time =
                    estimate_state_duration(graph, &self.states, active_to) * active_progress;
                if let Some(transition_index) =
                    self.find_transition_index(active_to, candidate_state_time, params, graph)
                {
                    let transition = self.transitions[transition_index].clone();
                    let duration = transition.duration.max(0.0);
                    if duration <= 0.0 {
                        self.current_state = transition.to;
                        self.state_time = 0.0;
                        self.active_transition = None;
                        return (self.current_state, None);
                    }
                    self.active_transition = Some(ActiveTransition {
                        from: transition.from,
                        to: transition.to,
                        elapsed: 0.0,
                        duration,
                        can_interrupt: transition.can_interrupt,
                    });
                    self.state_time = 0.0;
                    return (transition.from, Some((transition.to, 0.0)));
                }

                if active_progress >= 1.0 {
                    self.current_state = active_to;
                    self.state_time = 0.0;
                    self.active_transition = None;
                    return (self.current_state, None);
                }
                return (active_from, Some((active_to, active_progress)));
            }

            let t = if active.duration > 0.0 {
                (active.elapsed / active.duration).clamp(0.0, 1.0)
            } else {
                1.0
            };
            if t >= 1.0 {
                self.current_state = active.to;
                self.state_time = 0.0;
                self.active_transition = None;
                return (self.current_state, None);
            }
            return (active.from, Some((active.to, t)));
        }

        let current = self.current_state;
        if let Some(transition_index) =
            self.find_transition_index(current, self.state_time, params, graph)
        {
            let transition = self.transitions[transition_index].clone();
            let duration = transition.duration.max(0.0);
            if duration <= 0.0 {
                self.current_state = transition.to;
                self.state_time = 0.0;
                return (self.current_state, None);
            }
            self.active_transition = Some(ActiveTransition {
                from: transition.from,
                to: transition.to,
                elapsed: 0.0,
                duration,
                can_interrupt: transition.can_interrupt,
            });
            return (transition.from, Some((transition.to, 0.0)));
        }

        (self.current_state, None)
    }

    pub fn transition_status(&self) -> Option<AnimationTransitionStatus> {
        let active = self.active_transition.as_ref()?;
        let progress = if active.duration > 0.0 {
            (active.elapsed / active.duration).clamp(0.0, 1.0)
        } else {
            1.0
        };
        Some(AnimationTransitionStatus {
            from: active.from,
            to: active.to,
            elapsed: active.elapsed,
            duration: active.duration,
            progress,
            can_interrupt: active.can_interrupt,
        })
    }

    fn find_transition_index(
        &mut self,
        from_state: usize,
        state_time: f32,
        params: &mut AnimationParameters,
        graph: &AnimationGraph,
    ) -> Option<usize> {
        let state_duration = estimate_state_duration(graph, &self.states, from_state);
        for (index, transition) in self.transitions.iter().enumerate() {
            if transition.from != from_state {
                continue;
            }
            if let Some(exit_time) = transition.exit_time {
                if state_duration > 0.0 {
                    let normalized = state_time / state_duration;
                    if normalized < exit_time {
                        continue;
                    }
                }
            }
            if !conditions_met(transition.conditions.as_slice(), params) {
                continue;
            }
            return Some(index);
        }
        None
    }
}

#[derive(Debug, Clone, Default)]
pub struct AnimationParameters {
    pub floats: HashMap<String, f32>,
    pub bools: HashMap<String, bool>,
    pub triggers: HashSet<String>,
}

impl AnimationParameters {
    pub fn get_float(&self, name: &str, default: f32) -> f32 {
        self.floats.get(name).copied().unwrap_or(default)
    }

    pub fn get_bool(&self, name: &str, default: bool) -> bool {
        self.bools.get(name).copied().unwrap_or(default)
    }

    pub fn set_float(&mut self, name: impl Into<String>, value: f32) {
        self.floats.insert(name.into(), value);
    }

    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) {
        self.bools.insert(name.into(), value);
    }

    pub fn trigger(&mut self, name: impl Into<String>) {
        self.triggers.insert(name.into());
    }

    fn consume_trigger(&mut self, name: &str) -> bool {
        self.triggers.remove(name)
    }
}

#[derive(Debug, Clone)]
pub struct AnimationLayer {
    pub name: String,
    pub weight: f32,
    pub additive: bool,
    pub mask: Vec<f32>,
    pub graph: AnimationGraph,
    pub state_machine: AnimationStateMachine,
}

#[derive(Debug, Clone)]
pub struct Animator {
    pub layers: Vec<AnimationLayer>,
    pub parameters: AnimationParameters,
    pub enabled: bool,
    pub time_scale: f32,
}

impl Animator {
    pub fn evaluate(&mut self, skeleton: &Skeleton, dt: f32, out_pose: &mut Pose) {
        out_pose.reset_to_bind(skeleton);
        if !self.enabled {
            return;
        }
        let dt = dt * self.time_scale;
        let mut scratch = Pose::from_skeleton(skeleton);
        let mut layer_pose = Pose::from_skeleton(skeleton);

        for layer in &mut self.layers {
            if layer.weight <= 0.0 {
                continue;
            }
            let (from_state, transition) =
                layer
                    .state_machine
                    .update(dt, &mut self.parameters, &layer.graph);

            let from_node = layer.state_machine.states.get(from_state).map(|s| s.node);
            if let Some(node) = from_node {
                layer.graph.evaluate_node_with_params(
                    node,
                    skeleton,
                    layer.state_machine.state_time,
                    &mut layer_pose,
                    &mut scratch,
                    Some(&self.parameters),
                );
            } else {
                layer_pose.reset_to_bind(skeleton);
            }

            if let Some((to_state, t)) = transition {
                if let Some(state) = layer.state_machine.states.get(to_state) {
                    layer.graph.evaluate_node_with_params(
                        state.node,
                        skeleton,
                        layer.state_machine.state_time,
                        &mut scratch,
                        &mut layer_pose,
                        Some(&self.parameters),
                    );
                    blend_pose(&mut layer_pose, &scratch, t, None);
                }
            }

            if layer.additive {
                apply_additive_pose(
                    out_pose,
                    &layer_pose,
                    skeleton,
                    layer.weight,
                    Some(&layer.mask),
                );
            } else {
                blend_pose(out_pose, &layer_pose, layer.weight, Some(&layer.mask));
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AnimationLibrary {
    pub clips: Vec<Arc<AnimationClip>>,
    pub name_to_index: HashMap<String, usize>,
}

impl AnimationLibrary {
    pub fn add_clip(&mut self, clip: AnimationClip) -> usize {
        let index = self.clips.len();
        self.name_to_index.insert(clip.name.clone(), index);
        self.clips.push(Arc::new(clip));
        index
    }

    pub fn upsert_clip(&mut self, clip: AnimationClip) -> usize {
        if let Some(index) = self.name_to_index.get(&clip.name).copied() {
            self.clips[index] = Arc::new(clip);
            return index;
        }
        self.add_clip(clip)
    }

    pub fn clip(&self, index: usize) -> Option<&AnimationClip> {
        self.clips.get(index).map(|c| c.as_ref())
    }

    pub fn clip_index(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }
}

pub fn compute_global_matrices(skeleton: &Skeleton, locals: &[Transform], globals: &mut [Mat4]) {
    if globals.len() != skeleton.joints.len() {
        return;
    }

    let mut visited = vec![false; skeleton.joints.len()];
    for joint_index in 0..skeleton.joints.len() {
        compute_joint_global(skeleton, locals, globals, &mut visited, joint_index);
    }
}

fn compute_joint_global(
    skeleton: &Skeleton,
    locals: &[Transform],
    globals: &mut [Mat4],
    visited: &mut [bool],
    joint_index: usize,
) {
    if visited[joint_index] {
        return;
    }
    let joint = &skeleton.joints[joint_index];
    let local = locals
        .get(joint_index)
        .copied()
        .unwrap_or(joint.bind_transform);
    let local_matrix = local.to_matrix();
    let global = if let Some(parent) = joint.parent {
        compute_joint_global(skeleton, locals, globals, visited, parent);
        globals[parent] * local_matrix
    } else {
        local_matrix
    };
    globals[joint_index] = global;
    visited[joint_index] = true;
}

pub fn build_skin_palette(skeleton: &Skeleton, locals: &[Transform], out_palette: &mut Vec<Mat4>) {
    let joint_count = skeleton.joints.len();
    let mut globals = vec![Mat4::IDENTITY; joint_count];
    compute_global_matrices(skeleton, locals, &mut globals);
    out_palette.clear();
    out_palette.reserve(joint_count);
    for (joint, global) in skeleton.joints.iter().zip(globals.into_iter()) {
        out_palette.push(global * joint.inverse_bind);
    }
}

pub fn write_skin_palette(
    skeleton: &Skeleton,
    locals: &[Transform],
    globals: &mut Vec<Mat4>,
    out: &mut [Mat4],
) {
    let joint_count = skeleton.joints.len();
    if joint_count == 0 {
        return;
    }
    if globals.len() != joint_count {
        globals.resize(joint_count, Mat4::IDENTITY);
    }
    compute_global_matrices(skeleton, locals, globals);
    let count = joint_count.min(out.len());
    for i in 0..count {
        out[i] = globals[i] * skeleton.joints[i].inverse_bind;
    }
}

fn sample_vec3(keys: &[Keyframe<Vec3>], interpolation: Interpolation, time: f32) -> Vec3 {
    if keys.is_empty() {
        return Vec3::ZERO;
    }
    if keys.len() == 1 {
        return keys[0].value;
    }
    if time <= keys[0].time {
        return keys[0].value;
    }
    if time >= keys[keys.len() - 1].time {
        return keys[keys.len() - 1].value;
    }

    let (k0, k1) = match find_keyframe_pair(keys, time) {
        Some(pair) => pair,
        None => return keys[keys.len() - 1].value,
    };
    let dt = (k1.time - k0.time).max(f32::EPSILON);
    let t = ((time - k0.time) / dt).clamp(0.0, 1.0);
    match interpolation {
        Interpolation::Step => k0.value,
        Interpolation::Linear => k0.value.lerp(k1.value, t),
        Interpolation::Cubic => {
            let m0 = k0.out_tangent.unwrap_or(Vec3::ZERO);
            let m1 = k1.in_tangent.unwrap_or(Vec3::ZERO);
            hermite_vec3(k0.value, m0, k1.value, m1, t, dt)
        }
    }
}

fn sample_quat(keys: &[Keyframe<Quat>], interpolation: Interpolation, time: f32) -> Quat {
    if keys.is_empty() {
        return Quat::IDENTITY;
    }
    if keys.len() == 1 {
        return keys[0].value.normalize();
    }
    if time <= keys[0].time {
        return keys[0].value.normalize();
    }
    if time >= keys[keys.len() - 1].time {
        return keys[keys.len() - 1].value.normalize();
    }

    let (k0, k1) = match find_keyframe_pair(keys, time) {
        Some(pair) => pair,
        None => return keys[keys.len() - 1].value.normalize(),
    };
    let dt = (k1.time - k0.time).max(f32::EPSILON);
    let t = ((time - k0.time) / dt).clamp(0.0, 1.0);
    match interpolation {
        Interpolation::Step => k0.value.normalize(),
        Interpolation::Linear => k0.value.slerp(k1.value, t).normalize(),
        Interpolation::Cubic => {
            let m0 = k0.out_tangent.unwrap_or(Quat::IDENTITY);
            let m1 = k1.in_tangent.unwrap_or(Quat::IDENTITY);
            hermite_quat(k0.value, m0, k1.value, m1, t, dt)
        }
    }
}

fn hermite_vec3(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, t: f32, dt: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * p0 + h10 * (m0 * dt) + h01 * p1 + h11 * (m1 * dt)
}

fn hermite_quat(p0: Quat, m0: Quat, p1: Quat, m1: Quat, t: f32, dt: f32) -> Quat {
    let p0v = Vec4::from_array(p0.normalize().to_array());
    let p1v = Vec4::from_array(p1.normalize().to_array());
    let m0v = Vec4::from_array(m0.to_array());
    let m1v = Vec4::from_array(m1.to_array());
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    let v = p0v * h00 + m0v * (h10 * dt) + p1v * h01 + m1v * (h11 * dt);
    Quat::from_xyzw(v.x, v.y, v.z, v.w).normalize()
}

fn find_keyframe_pair<T>(keys: &[Keyframe<T>], time: f32) -> Option<(&Keyframe<T>, &Keyframe<T>)> {
    if keys.len() < 2 {
        return None;
    }
    let mut idx = 0;
    for i in 0..keys.len() - 1 {
        if time >= keys[i].time && time <= keys[i + 1].time {
            idx = i;
            break;
        }
    }
    Some((&keys[idx], &keys[idx + 1]))
}

fn blend_pose(target: &mut Pose, source: &Pose, weight: f32, mask: Option<&[f32]>) {
    let weight = weight.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return;
    }
    let count = target.locals.len().min(source.locals.len());
    for idx in 0..count {
        let joint_weight = mask
            .and_then(|mask| mask.get(idx).copied())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if joint_weight <= 0.0 {
            continue;
        }
        let w = weight * joint_weight;
        let a = target.locals[idx];
        let b = source.locals[idx];
        target.locals[idx] = Transform {
            position: a.position.lerp(b.position, w),
            rotation: a.rotation.slerp(b.rotation, w).normalize(),
            scale: a.scale.lerp(b.scale, w),
        };
    }
}

fn apply_additive_pose(
    target: &mut Pose,
    additive: &Pose,
    skeleton: &Skeleton,
    weight: f32,
    mask: Option<&[f32]>,
) {
    let weight = weight.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return;
    }
    let count = target.locals.len().min(additive.locals.len());
    for idx in 0..count {
        let joint_weight = mask
            .and_then(|mask| mask.get(idx).copied())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if joint_weight <= 0.0 {
            continue;
        }
        let w = weight * joint_weight;
        let base = skeleton
            .joints
            .get(idx)
            .map(|j| j.bind_transform)
            .unwrap_or_default();
        let add = additive.locals[idx];
        let delta_pos = add.position - base.position;
        let delta_scale = add.scale - base.scale;
        let delta_rot = base.rotation.inverse() * add.rotation;
        let current = target.locals[idx];
        target.locals[idx] = Transform {
            position: current.position + delta_pos * w,
            rotation: current.rotation * Quat::IDENTITY.slerp(delta_rot, w),
            scale: current.scale + delta_scale * w,
        };
    }
}

fn estimate_state_duration(
    graph: &AnimationGraph,
    states: &[AnimationState],
    state_index: usize,
) -> f32 {
    let Some(state) = states.get(state_index) else {
        return 0.0;
    };
    estimate_node_duration(graph, state.node)
}

fn estimate_node_duration(graph: &AnimationGraph, node_index: usize) -> f32 {
    let Some(node) = graph.nodes.get(node_index) else {
        return 0.0;
    };
    match node {
        AnimationNode::Clip(clip) => graph
            .library
            .clip(clip.clip_index)
            .map(|c| c.duration / clip.speed.max(0.0001))
            .unwrap_or(0.0),
        AnimationNode::Blend(blend) => blend
            .children
            .iter()
            .map(|child| estimate_node_duration(graph, child.node))
            .fold(0.0, f32::max),
    }
}

fn conditions_met(conditions: &[AnimationCondition], params: &mut AnimationParameters) -> bool {
    for condition in conditions {
        match condition {
            AnimationCondition::Bool { param, value } => {
                if params.get_bool(param, false) != *value {
                    return false;
                }
            }
            AnimationCondition::Float {
                param,
                comparison,
                value,
            } => {
                let current = params.get_float(param, 0.0);
                let pass = match comparison {
                    AnimationComparison::Less => current < *value,
                    AnimationComparison::LessEqual => current <= *value,
                    AnimationComparison::Greater => current > *value,
                    AnimationComparison::GreaterEqual => current >= *value,
                    AnimationComparison::Equal => (current - *value).abs() < f32::EPSILON,
                    AnimationComparison::NotEqual => (current - *value).abs() > f32::EPSILON,
                };
                if !pass {
                    return false;
                }
            }
            AnimationCondition::Trigger { param } => {
                if !params.consume_trigger(param) {
                    return false;
                }
            }
        }
    }
    true
}
