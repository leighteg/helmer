use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_void},
    ptr, slice,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

pub use serde_json::{Value as JsonValue, json};

pub const SCRIPT_API_ABI_VERSION: u32 = 1;
pub const SCRIPT_PLUGIN_ABI_VERSION: u32 = 1;

pub type EntityId = u64;

pub const ECS_FUNCTIONS: &[&str] = &[
    "add_component",
    "add_force",
    "add_force_at_point",
    "add_persistent_force",
    "add_persistent_force_at_point",
    "add_persistent_torque",
    "add_spline_point",
    "add_torque",
    "apply_angular_impulse",
    "apply_impulse",
    "apply_impulse_at_point",
    "apply_torque_impulse",
    "clear_audio_emitters",
    "clear_persistent_forces",
    "clear_physics",
    "create_audio_bus",
    "delete_entity",
    "entity_exists",
    "emit_event",
    "find_entity_by_name",
    "find_script_index",
    "follow_spline",
    "get_animator_clips",
    "get_animator_layer_weights",
    "get_animator_layer_weight",
    "get_animator_state",
    "get_animator_state_time",
    "get_animator_current_state",
    "get_animator_current_state_name",
    "get_animator_transition_active",
    "get_animator_transition_from",
    "get_animator_transition_to",
    "get_animator_transition_progress",
    "get_animator_transition_elapsed",
    "get_animator_transition_duration",
    "get_audio_bus_name",
    "get_audio_bus_volume",
    "get_audio_emitter",
    "get_audio_emitter_path",
    "get_audio_enabled",
    "get_audio_head_width",
    "get_audio_listener",
    "get_audio_scene_volume",
    "get_audio_speed_of_sound",
    "get_audio_streaming_config",
    "get_camera",
    "get_character_controller_output",
    "get_character_controller_desired_translation",
    "get_character_controller_effective_translation",
    "get_character_controller_remaining_translation",
    "get_character_controller_grounded",
    "get_character_controller_sliding_down_slope",
    "get_character_controller_collision_count",
    "get_character_controller_ground_normal",
    "get_character_controller_slope_angle",
    "get_character_controller_hit_normal",
    "get_character_controller_hit_point",
    "get_character_controller_hit_entity",
    "get_character_controller_stepped_up",
    "get_character_controller_step_height",
    "get_character_controller_platform_velocity",
    "get_collision_events",
    "get_collision_event_count",
    "get_collision_event_other",
    "get_collision_event_normal",
    "get_collision_event_point",
    "get_dynamic_component",
    "get_dynamic_field",
    "get_entity_follower",
    "get_entity_name",
    "get_light",
    "get_look_at",
    "get_mesh_renderer",
    "get_mesh_renderer_material_path",
    "get_mesh_renderer_source_path",
    "get_physics",
    "get_physics_gravity",
    "get_physics_point_projection_hit",
    "get_physics_ray_cast_hit",
    "get_physics_running",
    "get_physics_shape_cast_hit",
    "get_physics_velocity",
    "get_physics_world_defaults",
    "get_scene_asset",
    "get_script",
    "get_script_count",
    "get_script_field",
    "get_script_language",
    "get_script_path",
    "get_self_script_field",
    "get_spline",
    "get_transform",
    "get_transform_forward",
    "get_transform_right",
    "get_transform_up",
    "get_trigger_events",
    "get_trigger_event_count",
    "get_trigger_event_other",
    "get_trigger_event_normal",
    "get_trigger_event_point",
    "get_viewport_mode",
    "get_viewport_preview_camera",
    "has_component",
    "list_audio_buses",
    "list_dynamic_components",
    "list_entities",
    "list_script_fields",
    "list_self_script_fields",
    "open_scene",
    "play_anim_clip",
    "ray_cast",
    "ray_cast_has_hit",
    "ray_cast_hit_entity",
    "ray_cast_point",
    "ray_cast_normal",
    "ray_cast_toi",
    "sphere_cast",
    "sphere_cast_has_hit",
    "sphere_cast_hit_entity",
    "sphere_cast_point",
    "sphere_cast_normal",
    "sphere_cast_toi",
    "remove_audio_bus",
    "remove_component",
    "remove_dynamic_component",
    "remove_dynamic_field",
    "remove_spline_point",
    "sample_spline",
    "set_active_camera",
    "set_animator_enabled",
    "set_animator_blend_child",
    "set_animator_blend_node",
    "set_animator_param_bool",
    "set_animator_param_float",
    "set_animator_layer_weight",
    "set_animator_time_scale",
    "set_animator_transition",
    "set_audio_bus_name",
    "set_audio_bus_volume",
    "set_audio_emitter",
    "set_audio_emitter_path",
    "set_audio_enabled",
    "set_audio_head_width",
    "set_audio_listener",
    "set_audio_scene_volume",
    "set_audio_speed_of_sound",
    "set_audio_streaming_config",
    "set_camera",
    "set_dynamic_component",
    "set_dynamic_field",
    "set_entity_follower",
    "set_entity_name",
    "set_light",
    "set_look_at",
    "set_mesh_renderer",
    "set_mesh_renderer_material_path",
    "set_mesh_renderer_source_path",
    "set_persistent_force",
    "set_persistent_torque",
    "set_physics",
    "set_physics_gravity",
    "set_physics_running",
    "set_physics_velocity",
    "set_physics_world_defaults",
    "set_scene_asset",
    "set_script",
    "set_script_field",
    "set_self_script_field",
    "set_character_controller_desired_translation",
    "set_spline",
    "set_spline_point",
    "set_transform",
    "set_viewport_mode",
    "set_viewport_preview_camera",
    "spawn_entity",
    "spline_length",
    "switch_scene",
    "self_script_index",
    "trigger_animator",
];

pub const INPUT_FUNCTIONS: &[&str] = &[
    "bind_action",
    "cursor",
    "cursor_delta",
    "gamepad_axes",
    "gamepad_axis",
    "gamepad_axis_handle",
    "gamepad_axis_ref",
    "gamepad_button",
    "gamepad_button_down",
    "gamepad_button_pressed",
    "gamepad_button_released",
    "gamepad_buttons",
    "gamepad_count",
    "gamepad_ids",
    "gamepad_trigger",
    "key",
    "key_down",
    "key_pressed",
    "key_released",
    "keys",
    "modifiers",
    "mouse_button",
    "mouse_buttons",
    "mouse_down",
    "mouse_pressed",
    "mouse_released",
    "scale_factor",
    "cursor_grab_mode",
    "set_cursor_visible",
    "set_cursor_grab",
    "reset_cursor_control",
    "set_action_context",
    "wants_keyboard",
    "wants_pointer",
    "wheel",
    "window_size",
    "action_context",
    "action_value",
    "action_down",
    "action_pressed",
    "action_released",
    "unbind_action",
];

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TransformPatch {
    pub has_position: u8,
    pub position: Vec3,
    pub has_rotation: u8,
    pub rotation: Quat,
    pub has_scale: u8,
    pub scale: Vec3,
}

impl Default for TransformPatch {
    fn default() -> Self {
        Self {
            has_position: 0,
            position: Vec3::default(),
            has_rotation: 0,
            rotation: Quat::default(),
            has_scale: 0,
            scale: Vec3::default(),
        }
    }
}

impl TransformPatch {
    pub fn with_position(position: Vec3) -> Self {
        Self {
            has_position: 1,
            position,
            ..Self::default()
        }
    }

    pub fn with_rotation(rotation: Quat) -> Self {
        Self {
            has_rotation: 1,
            rotation,
            ..Self::default()
        }
    }

    pub fn with_scale(scale: Vec3) -> Self {
        Self {
            has_scale: 1,
            scale,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EcsComponentName {
    #[serde(rename = "name")]
    Name,
    #[serde(rename = "transform")]
    Transform,
    #[serde(rename = "camera")]
    Camera,
    #[serde(rename = "light")]
    Light,
    #[serde(rename = "mesh")]
    Mesh,
    #[serde(rename = "mesh_renderer")]
    MeshRenderer,
    #[serde(rename = "spline")]
    Spline,
    #[serde(rename = "spline_follower")]
    SplineFollower,
    #[serde(rename = "look_at")]
    LookAt,
    #[serde(rename = "entity_follower")]
    EntityFollower,
    #[serde(rename = "animator")]
    Animator,
    #[serde(rename = "scene")]
    Scene,
    #[serde(rename = "audio")]
    Audio,
    #[serde(rename = "audio_emitter")]
    AudioEmitter,
    #[serde(rename = "audio_listener")]
    AudioListener,
    #[serde(rename = "script")]
    Script,
    #[serde(rename = "dynamic")]
    Dynamic,
    #[serde(rename = "physics")]
    Physics,
    #[serde(rename = "collider_shape")]
    ColliderShape,
    #[serde(rename = "dynamic_rigid_body")]
    DynamicRigidBody,
    #[serde(rename = "kinematic_rigid_body")]
    KinematicRigidBody,
    #[serde(rename = "fixed_collider")]
    FixedCollider,
    #[serde(rename = "collider_properties")]
    ColliderProperties,
    #[serde(rename = "collider_inheritance")]
    ColliderInheritance,
    #[serde(rename = "rigid_body_properties")]
    RigidBodyProperties,
    #[serde(rename = "rigid_body_inheritance")]
    RigidBodyInheritance,
    #[serde(rename = "physics_joint")]
    PhysicsJoint,
    #[serde(rename = "character_controller")]
    CharacterController,
    #[serde(rename = "character_controller_input")]
    CharacterControllerInput,
    #[serde(rename = "character_controller_output")]
    CharacterControllerOutput,
    #[serde(rename = "physics_ray_cast")]
    PhysicsRayCast,
    #[serde(rename = "physics_ray_cast_hit")]
    PhysicsRayCastHit,
    #[serde(rename = "physics_point_projection")]
    PhysicsPointProjection,
    #[serde(rename = "physics_point_projection_hit")]
    PhysicsPointProjectionHit,
    #[serde(rename = "physics_shape_cast")]
    PhysicsShapeCast,
    #[serde(rename = "physics_shape_cast_hit")]
    PhysicsShapeCastHit,
    #[serde(rename = "physics_world_defaults")]
    PhysicsWorldDefaults,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplineMode {
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "catmullrom")]
    CatmullRom,
    #[serde(rename = "bezier")]
    Bezier,
}

impl Default for SplineMode {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SplineData {
    pub points: Vec<Vec3>,
    pub closed: bool,
    pub tension: f32,
    pub mode: SplineMode,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SplinePatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<Vec3>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tension: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<SplineMode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightKind {
    Directional,
    Point,
    Spot,
}

impl Default for LightKind {
    fn default() -> Self {
        Self::Point
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LightData {
    #[serde(rename = "type")]
    pub light_type: LightKind,
    pub color: Vec3,
    pub intensity: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LightPatch {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub light_type: Option<LightKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CameraData {
    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub active: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CameraPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fov_y_rad: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub near_plane: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub far_plane: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LookAtData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_entity: Option<EntityId>,
    pub target_offset: Vec3,
    pub offset_in_target_space: bool,
    pub up: Vec3,
    pub rotation_smooth_time: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LookAtPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_entity: Option<EntityId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_offset: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_in_target_space: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_smooth_time: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityFollowerData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_entity: Option<EntityId>,
    pub position_offset: Vec3,
    pub offset_in_target_space: bool,
    pub follow_rotation: bool,
    pub position_smooth_time: f32,
    pub rotation_smooth_time: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityFollowerPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_entity: Option<EntityId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_offset: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_in_target_space: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow_rotation: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_smooth_time: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_smooth_time: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshPrimitive {
    #[serde(rename = "Cube")]
    Cube,
    #[serde(rename = "UV Sphere")]
    UvSphere,
    #[serde(rename = "Plane")]
    Plane,
}

impl Default for MeshPrimitive {
    fn default() -> Self {
        Self::Cube
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MeshSource {
    Primitive(MeshPrimitive),
    AssetPath(String),
}

impl Default for MeshSource {
    fn default() -> Self {
        Self::Primitive(MeshPrimitive::Cube)
    }
}

impl MeshSource {
    pub fn asset(path: impl Into<String>) -> Self {
        Self::AssetPath(path.into())
    }

    pub fn cube() -> Self {
        Self::Primitive(MeshPrimitive::Cube)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshRendererData {
    pub source: MeshSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    pub casts_shadow: bool,
    pub visible: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshRendererPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<MeshSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub casts_shadow: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
}

impl MeshRendererPatch {
    pub fn cube() -> Self {
        Self {
            source: Some(MeshSource::cube()),
            material: None,
            casts_shadow: Some(true),
            visible: Some(true),
        }
    }

    pub fn with_source(mut self, source: MeshSource) -> Self {
        self.source = Some(source);
        self
    }

    pub fn with_material(mut self, material: impl Into<String>) -> Self {
        self.material = Some(material.into());
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScriptData {
    pub path: String,
    pub language: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NamedAudioBus {
    Master,
    Music,
    Sfx,
    Ui,
    Ambience,
    World,
}

impl Default for NamedAudioBus {
    fn default() -> Self {
        Self::Master
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AudioBus {
    Named(NamedAudioBus),
    Custom(i64),
}

impl Default for AudioBus {
    fn default() -> Self {
        Self::Named(NamedAudioBus::Master)
    }
}

impl AudioBus {
    pub const MASTER: Self = Self::Named(NamedAudioBus::Master);
    pub const MUSIC: Self = Self::Named(NamedAudioBus::Music);
    pub const SFX: Self = Self::Named(NamedAudioBus::Sfx);
    pub const UI: Self = Self::Named(NamedAudioBus::Ui);
    pub const AMBIENCE: Self = Self::Named(NamedAudioBus::Ambience);
    pub const WORLD: Self = Self::Named(NamedAudioBus::World);

    pub fn custom(id: i64) -> Self {
        Self::Custom(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioPlaybackState {
    Playing,
    Paused,
    Stopped,
}

impl Default for AudioPlaybackState {
    fn default() -> Self {
        Self::Playing
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AudioPlaybackStateValue {
    State(AudioPlaybackState),
    Index(i64),
    Name(String),
}

impl Default for AudioPlaybackStateValue {
    fn default() -> Self {
        Self::State(AudioPlaybackState::Playing)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioEmitterData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub streaming: bool,
    pub bus: AudioBus,
    pub volume: f32,
    pub pitch: f32,
    pub looping: bool,
    pub spatial: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: f32,
    pub spatial_blend: f32,
    pub playback_state: AudioPlaybackStateValue,
    pub play_on_spawn: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_id: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioEmitterPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bus: Option<AudioBus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub looping: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_distance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rolloff: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_blend: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub playback_state: Option<AudioPlaybackStateValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub play_on_spawn: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioListenerData {
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioListenerPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioStreamingConfig {
    pub buffer_frames: u32,
    pub chunk_frames: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DynamicFieldValue {
    Bool(bool),
    Number(f64),
    Text(String),
    Vec3(Vec3),
}

pub type DynamicFields = BTreeMap<String, DynamicFieldValue>;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DynamicComponentData {
    pub name: String,
    pub fields: DynamicFields,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct InputModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    #[serde(rename = "super")]
    pub super_key: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsCombineRule {
    Average,
    Min,
    Multiply,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KinematicMode {
    PositionBased,
    VelocityBased,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsJointKind {
    Fixed,
    Spherical,
    Revolute,
    Prismatic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshColliderKind {
    TriMesh,
    ConvexHull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshColliderLodPreset {
    Lod0,
    Lod1,
    Lod2,
    Lowest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MeshColliderLod {
    Preset(MeshColliderLodPreset),
    Specific(u8),
}

impl Default for MeshColliderLod {
    fn default() -> Self {
        Self::Preset(MeshColliderLodPreset::Lod0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsShapeCastStatus {
    NoHit,
    Converged,
    OutOfIterations,
    Failed,
    PenetratingOrWithinTargetDist,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PhysicsBodyKindData {
    Dynamic { mass: f32 },
    Kinematic { mode: KinematicMode },
    Fixed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PhysicsBodyKindValue {
    Data(PhysicsBodyKindData),
    Name(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ColliderShapeData {
    Cuboid,
    Sphere,
    CapsuleY,
    CylinderY,
    ConeY,
    RoundCuboid {
        border_radius: f32,
    },
    Mesh {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mesh_id: Option<u64>,
        lod: MeshColliderLod,
        kind: MeshColliderKind,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ColliderShapePatch {
    #[default]
    Cuboid,
    Sphere,
    CapsuleY,
    CylinderY,
    ConeY,
    RoundCuboid {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        border_radius: Option<f32>,
    },
    Mesh {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mesh_id: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lod: Option<MeshColliderLod>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kind: Option<MeshColliderKind>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ColliderShapeValue {
    Shape(ColliderShapePatch),
    Name(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsQueryFilterData {
    pub flags: u32,
    pub groups_memberships: u32,
    pub groups_filter: u32,
    pub use_groups: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_fixed_flag: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_kinematic_flag: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_dynamic_flag: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_sensors_flag: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_solids_flag: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsQueryFilterPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flags: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub groups_memberships: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub groups_filter: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_groups: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColliderPropertiesData {
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ColliderPropertiesPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub friction: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restitution: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_sensor: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision_memberships: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision_filter: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_memberships: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_filter: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub friction_combine_rule: Option<PhysicsCombineRule>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restitution_combine_rule: Option<PhysicsCombineRule>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translation_offset: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_offset: Option<Quat>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ColliderInheritanceData {
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ColliderInheritancePatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub friction: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restitution: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_sensor: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision_memberships: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision_filter: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_memberships: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_filter: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub friction_combine_rule: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restitution_combine_rule: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translation_offset: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_offset: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RigidBodyPropertiesData {
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub ccd_enabled: bool,
    pub can_sleep: bool,
    pub sleeping: bool,
    pub dominance_group: i16,
    pub lock_translation_x: bool,
    pub lock_translation_y: bool,
    pub lock_translation_z: bool,
    pub lock_rotation_x: bool,
    pub lock_rotation_y: bool,
    pub lock_rotation_z: bool,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RigidBodyPropertiesPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_damping: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_damping: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ccd_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub can_sleep: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sleeping: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dominance_group: Option<i16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_translation_x: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_translation_y: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_translation_z: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_rotation_x: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_rotation_y: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_rotation_z: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_velocity: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_velocity: Option<Vec3>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RigidBodyInheritanceData {
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RigidBodyInheritancePatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_damping: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_damping: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_scale: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ccd_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub can_sleep: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sleeping: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dominance_group: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_translation_x: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_translation_y: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_translation_z: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_rotation_x: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_rotation_y: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_rotation_z: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_velocity: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_velocity: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsJointLimitsData {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsJointMotorData {
    pub enabled: bool,
    pub target_position: f32,
    pub target_velocity: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub max_force: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsJointMotorPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_position: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_velocity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stiffness: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damping: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_force: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsJointData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<EntityId>,
    pub kind: PhysicsJointKind,
    pub contacts_enabled: bool,
    pub local_anchor1: Vec3,
    pub local_anchor2: Vec3,
    pub local_axis1: Vec3,
    pub local_axis2: Vec3,
    pub limit_enabled: bool,
    pub limits: PhysicsJointLimitsData,
    pub motor: PhysicsJointMotorData,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsJointPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<EntityId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<PhysicsJointKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contacts_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_anchor1: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_anchor2: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_axis1: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_axis2: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits: Option<PhysicsJointLimitsData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motor: Option<PhysicsJointMotorPatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CharacterControllerData {
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CharacterControllerPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slide: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autostep_max_height: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autostep_min_width: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autostep_include_dynamic_bodies: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_slope_climb_angle: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_slope_slide_angle: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snap_to_ground: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_nudge_factor: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_impulses_to_dynamic_bodies: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_mass: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CharacterControllerInputData {
    pub desired_translation: Vec3,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CharacterControllerInputPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desired_translation: Option<Vec3>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CharacterControllerOutputData {
    pub desired_translation: Vec3,
    pub effective_translation: Vec3,
    pub remaining_translation: Vec3,
    pub grounded: bool,
    pub sliding_down_slope: bool,
    pub collision_count: usize,
    pub ground_normal: Vec3,
    pub slope_angle: f32,
    pub hit_normal: Vec3,
    pub hit_point: Vec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_entity: Option<EntityId>,
    pub stepped_up: bool,
    pub step_height: f32,
    pub platform_velocity: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhysicsRayCastData {
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_toi: f32,
    pub solid: bool,
    pub filter: PhysicsQueryFilterData,
    pub exclude_self: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsRayCastPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_toi: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solid: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<PhysicsQueryFilterPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_self: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhysicsRayCastHitData {
    pub has_hit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_entity: Option<EntityId>,
    pub point: Vec3,
    pub normal: Vec3,
    pub toi: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhysicsPointProjectionData {
    pub point: Vec3,
    pub solid: bool,
    pub filter: PhysicsQueryFilterData,
    pub exclude_self: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsPointProjectionPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solid: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<PhysicsQueryFilterPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_self: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhysicsPointProjectionHitData {
    pub has_hit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_entity: Option<EntityId>,
    pub projected_point: Vec3,
    pub is_inside: bool,
    pub distance: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsShapeCastData {
    pub shape: ColliderShapeData,
    pub scale: Vec3,
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub max_time_of_impact: f32,
    pub target_distance: f32,
    pub stop_at_penetration: bool,
    pub compute_impact_geometry_on_penetration: bool,
    pub filter: PhysicsQueryFilterData,
    pub exclude_self: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsShapeCastPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<ColliderShapeValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<Quat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub velocity: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_time_of_impact: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_distance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_at_penetration: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_impact_geometry_on_penetration: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<PhysicsQueryFilterPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_self: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhysicsShapeCastHitData {
    pub has_hit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_entity: Option<EntityId>,
    pub toi: f32,
    pub witness1: Vec3,
    pub witness2: Vec3,
    pub normal1: Vec3,
    pub normal2: Vec3,
    pub status: PhysicsShapeCastStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsWorldDefaultsData {
    pub gravity: Vec3,
    pub collider_properties: ColliderPropertiesData,
    pub rigid_body_properties: RigidBodyPropertiesData,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsWorldDefaultsPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider_properties: Option<ColliderPropertiesPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid_body_properties: Option<RigidBodyPropertiesPatch>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsVelocityData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular: Option<Vec3>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsVelocityPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_up: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider_shape: Option<ColliderShapeData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_kind: Option<PhysicsBodyKindData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider_properties: Option<ColliderPropertiesData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider_inheritance: Option<ColliderInheritanceData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid_body_properties: Option<RigidBodyPropertiesData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid_body_inheritance: Option<RigidBodyInheritanceData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint: Option<PhysicsJointData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_controller: Option<CharacterControllerData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_input: Option<CharacterControllerInputData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_output: Option<CharacterControllerOutputData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_cast: Option<PhysicsRayCastData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_cast_hit: Option<PhysicsRayCastHitData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_projection: Option<PhysicsPointProjectionData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_projection_hit: Option<PhysicsPointProjectionHitData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape_cast: Option<PhysicsShapeCastData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape_cast_hit: Option<PhysicsShapeCastHitData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_defaults: Option<PhysicsWorldDefaultsData>,
    pub has_handle: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider_shape: Option<ColliderShapeValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_kind: Option<PhysicsBodyKindValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider_properties: Option<ColliderPropertiesPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider_inheritance: Option<ColliderInheritancePatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid_body_properties: Option<RigidBodyPropertiesPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid_body_inheritance: Option<RigidBodyInheritancePatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint: Option<PhysicsJointPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_controller: Option<CharacterControllerPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_input: Option<CharacterControllerInputPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_cast: Option<PhysicsRayCastPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_projection: Option<PhysicsPointProjectionPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape_cast: Option<PhysicsShapeCastPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_defaults: Option<PhysicsWorldDefaultsPatch>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ScriptBytes {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ScriptBytesView {
    pub ptr: *const u8,
    pub len: usize,
}

fn bytes_from_vec(mut value: Vec<u8>) -> ScriptBytes {
    let out = ScriptBytes {
        ptr: value.as_mut_ptr(),
        len: value.len(),
        cap: value.capacity(),
    };
    std::mem::forget(value);
    out
}

unsafe fn drop_script_bytes(value: ScriptBytes) {
    if value.ptr.is_null() {
        return;
    }
    // SAFETY: The buffer must originate from `bytes_from_vec` in this crate
    let _ = unsafe { Vec::from_raw_parts(value.ptr, value.len, value.cap) };
}

#[repr(C)]
pub struct ScriptApi {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub log: unsafe extern "C" fn(user_data: *mut c_void, message: *const c_char),
    pub spawn_entity: unsafe extern "C" fn(user_data: *mut c_void, name: *const c_char) -> EntityId,
    pub entity_exists: unsafe extern "C" fn(user_data: *mut c_void, entity_id: EntityId) -> u8,
    pub delete_entity: unsafe extern "C" fn(user_data: *mut c_void, entity_id: EntityId) -> u8,
    pub get_transform: unsafe extern "C" fn(
        user_data: *mut c_void,
        entity_id: EntityId,
        out_transform: *mut Transform,
    ) -> u8,
    pub set_transform: unsafe extern "C" fn(
        user_data: *mut c_void,
        entity_id: EntityId,
        patch: *const TransformPatch,
    ) -> u8,
    pub invoke_json: unsafe extern "C" fn(
        user_data: *mut c_void,
        table_name: *const c_char,
        function_name: *const c_char,
        args_json: *const c_char,
        out_result: *mut ScriptBytes,
    ) -> u8,
    pub free_bytes: unsafe extern "C" fn(user_data: *mut c_void, value: ScriptBytes),
}

#[repr(C)]
pub struct ScriptPlugin {
    pub abi_version: u32,
    pub create: unsafe extern "C" fn(api: *const ScriptApi, entity_id: EntityId) -> *mut c_void,
    pub destroy: unsafe extern "C" fn(instance: *mut c_void),
    pub on_start: unsafe extern "C" fn(instance: *mut c_void),
    pub on_update: unsafe extern "C" fn(instance: *mut c_void, dt: f32),
    pub on_stop: unsafe extern "C" fn(instance: *mut c_void),
    pub save_state: unsafe extern "C" fn(instance: *mut c_void, out_state: *mut ScriptBytes) -> u8,
    pub load_state: unsafe extern "C" fn(instance: *mut c_void, state: ScriptBytesView) -> u8,
    pub free_state: unsafe extern "C" fn(state: ScriptBytes),
}

#[derive(Deserialize)]
struct ApiInvokeResponse {
    ok: bool,
    #[serde(default)]
    result: Value,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Clone, Copy)]
pub struct Host {
    api: *const ScriptApi,
    entity_id: EntityId,
}

impl Host {
    pub fn entity_id(&self) -> EntityId {
        self.entity_id
    }

    pub fn log(&self, message: &str) {
        let c_message = to_c_string(message);
        // SAFETY: The host owns the API table and function pointers for the entire callback
        unsafe {
            ((*self.api).log)((*self.api).user_data, c_message.as_ptr());
        }
    }

    pub fn spawn_entity(&self, name: Option<&str>) -> EntityId {
        let c_name = name.map(to_c_string);
        let name_ptr = c_name
            .as_ref()
            .map(|name| name.as_ptr())
            .unwrap_or(ptr::null());
        // SAFETY: The host owns the API table and function pointers for the entire callback
        unsafe { ((*self.api).spawn_entity)((*self.api).user_data, name_ptr) }
    }

    pub fn entity_exists(&self, entity_id: EntityId) -> bool {
        // SAFETY: The host owns the API table and function pointers for the entire callback
        unsafe { ((*self.api).entity_exists)((*self.api).user_data, entity_id) != 0 }
    }

    pub fn delete_entity(&self, entity_id: EntityId) -> bool {
        // SAFETY: The host owns the API table and function pointers for the entire callback
        unsafe { ((*self.api).delete_entity)((*self.api).user_data, entity_id) != 0 }
    }

    pub fn get_transform(&self, entity_id: EntityId) -> Option<Transform> {
        let mut transform = Transform::default();
        // SAFETY: The host owns the API table and function pointers for the entire callback
        let ok = unsafe {
            ((*self.api).get_transform)((*self.api).user_data, entity_id, &mut transform)
        };
        if ok != 0 { Some(transform) } else { None }
    }

    pub fn set_transform(&self, entity_id: EntityId, patch: &TransformPatch) -> bool {
        // SAFETY: The host owns the API table and function pointers for the entire callback
        unsafe { ((*self.api).set_transform)((*self.api).user_data, entity_id, patch) != 0 }
    }

    pub fn set_position(&self, entity_id: EntityId, position: Vec3) -> bool {
        self.set_transform(entity_id, &TransformPatch::with_position(position))
    }

    pub fn call_api_value<S: Serialize>(
        &self,
        table_name: &str,
        function_name: &str,
        args: S,
    ) -> Result<Value, String> {
        let args_value = serde_json::to_value(args).map_err(|err| err.to_string())?;
        let args_value = normalize_call_args(args_value);
        let args_json = serde_json::to_string(&args_value).map_err(|err| err.to_string())?;
        let payload = self.invoke_json_raw(table_name, function_name, &args_json)?;
        let response: ApiInvokeResponse =
            serde_json::from_str(&payload).map_err(|err| err.to_string())?;
        if response.ok {
            Ok(response.result)
        } else {
            Err(response
                .error
                .unwrap_or_else(|| format!("{}:{} call failed", table_name, function_name)))
        }
    }

    pub fn call_api<R: DeserializeOwned, S: Serialize>(
        &self,
        table_name: &str,
        function_name: &str,
        args: S,
    ) -> Result<R, String> {
        let value = self.call_api_value(table_name, function_name, args)?;
        serde_json::from_value(value).map_err(|err| err.to_string())
    }

    pub fn ecs_call_value<S: Serialize>(
        &self,
        function_name: &str,
        args: S,
    ) -> Result<Value, String> {
        self.call_api_value("ecs", function_name, args)
    }

    pub fn ecs_call<R: DeserializeOwned, S: Serialize>(
        &self,
        function_name: &str,
        args: S,
    ) -> Result<R, String> {
        self.call_api("ecs", function_name, args)
    }

    pub fn input_call_value<S: Serialize>(
        &self,
        function_name: &str,
        args: S,
    ) -> Result<Value, String> {
        self.call_api_value("input", function_name, args)
    }

    pub fn input_call<R: DeserializeOwned, S: Serialize>(
        &self,
        function_name: &str,
        args: S,
    ) -> Result<R, String> {
        self.call_api("input", function_name, args)
    }

    pub fn list_entities(&self) -> Result<Vec<EntityId>, String> {
        self.ecs_call("list_entities", ())
    }

    pub fn find_entity_by_name(&self, name: &str) -> Result<Option<EntityId>, String> {
        self.ecs_call("find_entity_by_name", (name,))
    }

    pub fn get_entity_name(&self, entity_id: EntityId) -> Result<Option<String>, String> {
        self.ecs_call("get_entity_name", (entity_id,))
    }

    pub fn set_entity_name(&self, entity_id: EntityId, name: &str) -> Result<bool, String> {
        self.ecs_call("set_entity_name", (entity_id, name))
    }

    pub fn has_component(
        &self,
        entity_id: EntityId,
        component: EcsComponentName,
    ) -> Result<bool, String> {
        self.ecs_call("has_component", (entity_id, component))
    }

    pub fn add_component(
        &self,
        entity_id: EntityId,
        component: EcsComponentName,
    ) -> Result<bool, String> {
        self.ecs_call("add_component", (entity_id, component))
    }

    pub fn remove_component(
        &self,
        entity_id: EntityId,
        component: EcsComponentName,
    ) -> Result<bool, String> {
        self.ecs_call("remove_component", (entity_id, component))
    }

    pub fn get_spline(&self, entity_id: EntityId) -> Result<Option<SplineData>, String> {
        self.ecs_call("get_spline", (entity_id,))
    }

    pub fn set_spline(&self, entity_id: EntityId, patch: &SplinePatch) -> Result<bool, String> {
        self.ecs_call("set_spline", (entity_id, patch))
    }

    pub fn add_spline_point(&self, entity_id: EntityId, point: Vec3) -> Result<bool, String> {
        self.ecs_call("add_spline_point", (entity_id, point))
    }

    pub fn set_spline_point(
        &self,
        entity_id: EntityId,
        index: usize,
        point: Vec3,
    ) -> Result<bool, String> {
        self.ecs_call("set_spline_point", (entity_id, index, point))
    }

    pub fn remove_spline_point(&self, entity_id: EntityId, index: usize) -> Result<bool, String> {
        self.ecs_call("remove_spline_point", (entity_id, index))
    }

    pub fn sample_spline(&self, entity_id: EntityId, t: f32) -> Result<Option<Vec3>, String> {
        self.ecs_call("sample_spline", (entity_id, t))
    }

    pub fn spline_length(
        &self,
        entity_id: EntityId,
        samples: Option<usize>,
    ) -> Result<Option<f32>, String> {
        self.ecs_call("spline_length", (entity_id, samples))
    }

    pub fn follow_spline(
        &self,
        entity_id: EntityId,
        spline_id: Option<EntityId>,
        speed: Option<f32>,
        looped: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("follow_spline", (entity_id, spline_id, speed, looped))
    }

    pub fn get_transform_forward(&self, entity_id: EntityId) -> Result<Vec3, String> {
        self.ecs_call("get_transform_forward", (entity_id,))
    }

    pub fn get_transform_right(&self, entity_id: EntityId) -> Result<Vec3, String> {
        self.ecs_call("get_transform_right", (entity_id,))
    }

    pub fn get_transform_up(&self, entity_id: EntityId) -> Result<Vec3, String> {
        self.ecs_call("get_transform_up", (entity_id,))
    }

    pub fn get_look_at(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_look_at", (entity_id,))
    }

    pub fn get_look_at_data(&self, entity_id: EntityId) -> Result<Option<LookAtData>, String> {
        self.ecs_call("get_look_at", (entity_id,))
    }

    pub fn set_look_at<S: Serialize>(&self, entity_id: EntityId, patch: S) -> Result<bool, String> {
        self.ecs_call("set_look_at", (entity_id, patch))
    }

    pub fn set_look_at_data(
        &self,
        entity_id: EntityId,
        patch: &LookAtPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_look_at", (entity_id, patch))
    }

    pub fn get_entity_follower(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_entity_follower", (entity_id,))
    }

    pub fn get_entity_follower_data(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<EntityFollowerData>, String> {
        self.ecs_call("get_entity_follower", (entity_id,))
    }

    pub fn set_entity_follower<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_entity_follower", (entity_id, patch))
    }

    pub fn set_entity_follower_data(
        &self,
        entity_id: EntityId,
        patch: &EntityFollowerPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_entity_follower", (entity_id, patch))
    }

    pub fn set_animator_enabled(&self, entity_id: EntityId, enabled: bool) -> Result<bool, String> {
        self.ecs_call("set_animator_enabled", (entity_id, enabled))
    }

    pub fn set_animator_time_scale(
        &self,
        entity_id: EntityId,
        time_scale: f32,
    ) -> Result<bool, String> {
        self.ecs_call("set_animator_time_scale", (entity_id, time_scale))
    }

    pub fn set_animator_param_float(
        &self,
        entity_id: EntityId,
        name: &str,
        value: f32,
    ) -> Result<bool, String> {
        self.ecs_call("set_animator_param_float", (entity_id, name, value))
    }

    pub fn set_animator_param_bool(
        &self,
        entity_id: EntityId,
        name: &str,
        value: bool,
    ) -> Result<bool, String> {
        self.ecs_call("set_animator_param_bool", (entity_id, name, value))
    }

    pub fn trigger_animator(&self, entity_id: EntityId, name: &str) -> Result<bool, String> {
        self.ecs_call("trigger_animator", (entity_id, name))
    }

    pub fn get_animator_clips(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<Option<Vec<String>>, String> {
        self.ecs_call("get_animator_clips", (entity_id, layer_index))
    }

    pub fn get_animator_state(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<Option<Value>, String> {
        self.ecs_call("get_animator_state", (entity_id, layer_index))
    }

    pub fn get_animator_layer_weights(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_animator_layer_weights", (entity_id,))
    }

    pub fn get_animator_layer_weight(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<f32, String> {
        self.ecs_call("get_animator_layer_weight", (entity_id, layer_index))
    }

    pub fn get_animator_state_time(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<f32, String> {
        self.ecs_call("get_animator_state_time", (entity_id, layer_index))
    }

    pub fn get_animator_current_state(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<usize, String> {
        let state: u64 = self.ecs_call("get_animator_current_state", (entity_id, layer_index))?;
        Ok(state as usize)
    }

    pub fn get_animator_current_state_name(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<String, String> {
        self.ecs_call("get_animator_current_state_name", (entity_id, layer_index))
    }

    pub fn get_animator_transition_active(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<bool, String> {
        self.ecs_call("get_animator_transition_active", (entity_id, layer_index))
    }

    pub fn get_animator_transition_from(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<usize, String> {
        let state: u64 = self.ecs_call("get_animator_transition_from", (entity_id, layer_index))?;
        Ok(state as usize)
    }

    pub fn get_animator_transition_to(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<usize, String> {
        let state: u64 = self.ecs_call("get_animator_transition_to", (entity_id, layer_index))?;
        Ok(state as usize)
    }

    pub fn get_animator_transition_progress(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<f32, String> {
        self.ecs_call("get_animator_transition_progress", (entity_id, layer_index))
    }

    pub fn get_animator_transition_elapsed(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<f32, String> {
        self.ecs_call("get_animator_transition_elapsed", (entity_id, layer_index))
    }

    pub fn get_animator_transition_duration(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
    ) -> Result<f32, String> {
        self.ecs_call("get_animator_transition_duration", (entity_id, layer_index))
    }

    pub fn set_animator_layer_weight(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
        weight: f32,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_animator_layer_weight",
            (entity_id, layer_index, weight),
        )
    }

    pub fn set_animator_transition<S: Serialize>(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
        transition_index: usize,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_animator_transition",
            (entity_id, layer_index, transition_index, patch),
        )
    }

    pub fn set_animator_blend_node(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
        node_index: usize,
        normalize: Option<bool>,
        mode: Option<&str>,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_animator_blend_node",
            (entity_id, layer_index, node_index, normalize, mode),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_animator_blend_child(
        &self,
        entity_id: EntityId,
        layer_index: Option<usize>,
        node_index: usize,
        child_index: usize,
        weight: Option<f32>,
        weight_param: Option<&str>,
        weight_scale: Option<f32>,
        weight_bias: Option<f32>,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_animator_blend_child",
            (
                entity_id,
                layer_index,
                node_index,
                child_index,
                weight,
                weight_param,
                weight_scale,
                weight_bias,
            ),
        )
    }

    pub fn play_anim_clip(
        &self,
        entity_id: EntityId,
        clip_name: &str,
        layer_index: Option<usize>,
    ) -> Result<bool, String> {
        self.ecs_call("play_anim_clip", (entity_id, clip_name, layer_index))
    }

    pub fn get_light(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_light", (entity_id,))
    }

    pub fn get_light_data(&self, entity_id: EntityId) -> Result<Option<LightData>, String> {
        self.ecs_call("get_light", (entity_id,))
    }

    pub fn set_light<S: Serialize>(&self, entity_id: EntityId, patch: S) -> Result<bool, String> {
        self.ecs_call("set_light", (entity_id, patch))
    }

    pub fn set_light_data(&self, entity_id: EntityId, patch: &LightPatch) -> Result<bool, String> {
        self.ecs_call("set_light", (entity_id, patch))
    }

    pub fn get_camera(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_camera", (entity_id,))
    }

    pub fn get_camera_data(&self, entity_id: EntityId) -> Result<Option<CameraData>, String> {
        self.ecs_call("get_camera", (entity_id,))
    }

    pub fn set_camera<S: Serialize>(&self, entity_id: EntityId, patch: S) -> Result<bool, String> {
        self.ecs_call("set_camera", (entity_id, patch))
    }

    pub fn set_camera_data(
        &self,
        entity_id: EntityId,
        patch: &CameraPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_camera", (entity_id, patch))
    }

    pub fn set_active_camera(&self, entity_id: EntityId) -> Result<bool, String> {
        self.ecs_call("set_active_camera", (entity_id,))
    }

    pub fn get_viewport_mode(&self) -> Result<String, String> {
        self.ecs_call("get_viewport_mode", ())
    }

    pub fn set_viewport_mode(&self, mode: &str) -> Result<bool, String> {
        self.ecs_call("set_viewport_mode", (mode,))
    }

    pub fn get_viewport_preview_camera(&self) -> Result<Option<EntityId>, String> {
        self.ecs_call("get_viewport_preview_camera", ())
    }

    pub fn set_viewport_preview_camera(&self, entity_id: Option<EntityId>) -> Result<bool, String> {
        self.ecs_call("set_viewport_preview_camera", (entity_id,))
    }

    pub fn get_mesh_renderer(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_mesh_renderer", (entity_id,))
    }

    pub fn get_mesh_renderer_data(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<MeshRendererData>, String> {
        self.ecs_call("get_mesh_renderer", (entity_id,))
    }

    pub fn set_mesh_renderer<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_mesh_renderer", (entity_id, patch))
    }

    pub fn set_mesh_renderer_data(
        &self,
        entity_id: EntityId,
        patch: &MeshRendererPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_mesh_renderer", (entity_id, patch))
    }

    pub fn get_mesh_renderer_source_path(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<String>, String> {
        self.ecs_call("get_mesh_renderer_source_path", (entity_id,))
    }

    pub fn get_mesh_renderer_material_path(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<String>, String> {
        self.ecs_call("get_mesh_renderer_material_path", (entity_id,))
    }

    pub fn set_mesh_renderer_source_path(
        &self,
        entity_id: EntityId,
        path: &str,
    ) -> Result<bool, String> {
        self.ecs_call("set_mesh_renderer_source_path", (entity_id, path))
    }

    pub fn set_mesh_renderer_material_path(
        &self,
        entity_id: EntityId,
        path: &str,
    ) -> Result<bool, String> {
        self.ecs_call("set_mesh_renderer_material_path", (entity_id, path))
    }

    pub fn get_scene_asset(&self, entity_id: EntityId) -> Result<Option<String>, String> {
        self.ecs_call("get_scene_asset", (entity_id,))
    }

    pub fn set_scene_asset(&self, entity_id: EntityId, path: &str) -> Result<bool, String> {
        self.ecs_call("set_scene_asset", (entity_id, path))
    }

    pub fn open_scene(&self, path: &str) -> Result<bool, String> {
        self.ecs_call("open_scene", (path,))
    }

    pub fn switch_scene(&self, path: &str) -> Result<bool, String> {
        self.ecs_call("switch_scene", (path,))
    }

    pub fn self_script_index(&self) -> Result<Option<usize>, String> {
        let index: Option<u64> = self.ecs_call("self_script_index", ())?;
        Ok(from_lua_script_index(index))
    }

    pub fn get_script_count(&self, entity_id: EntityId) -> Result<usize, String> {
        let count: u64 = self.ecs_call("get_script_count", (entity_id,))?;
        Ok(usize::try_from(count).unwrap_or(usize::MAX))
    }

    pub fn find_script_index(
        &self,
        entity_id: EntityId,
        path: &str,
        language: Option<&str>,
    ) -> Result<Option<usize>, String> {
        let index: Option<u64> = self.ecs_call("find_script_index", (entity_id, path, language))?;
        Ok(from_lua_script_index(index))
    }

    pub fn get_script_data(&self, entity_id: EntityId) -> Result<Option<ScriptData>, String> {
        self.get_script_data_at(entity_id, None)
    }

    pub fn get_script_data_at(
        &self,
        entity_id: EntityId,
        script_index: Option<usize>,
    ) -> Result<Option<ScriptData>, String> {
        self.ecs_call("get_script", (entity_id, lua_script_index(script_index)))
    }

    pub fn get_script_path(
        &self,
        entity_id: EntityId,
        script_index: Option<usize>,
    ) -> Result<Option<String>, String> {
        self.ecs_call(
            "get_script_path",
            (entity_id, lua_script_index(script_index)),
        )
    }

    pub fn get_script_language(
        &self,
        entity_id: EntityId,
        script_index: Option<usize>,
    ) -> Result<Option<String>, String> {
        self.ecs_call(
            "get_script_language",
            (entity_id, lua_script_index(script_index)),
        )
    }

    pub fn list_script_fields(
        &self,
        entity_id: EntityId,
        script_index: Option<usize>,
    ) -> Result<Option<Value>, String> {
        self.ecs_call(
            "list_script_fields",
            (entity_id, lua_script_index(script_index)),
        )
    }

    pub fn get_script_field(
        &self,
        entity_id: EntityId,
        field_name: &str,
        script_index: Option<usize>,
    ) -> Result<Option<Value>, String> {
        self.ecs_call(
            "get_script_field",
            (entity_id, field_name, lua_script_index(script_index)),
        )
    }

    pub fn set_script_field<S: Serialize>(
        &self,
        entity_id: EntityId,
        field_name: &str,
        value: S,
        script_index: Option<usize>,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_script_field",
            (entity_id, field_name, value, lua_script_index(script_index)),
        )
    }

    pub fn list_self_script_fields(&self) -> Result<Option<Value>, String> {
        self.ecs_call("list_self_script_fields", ())
    }

    pub fn get_self_script_field(&self, field_name: &str) -> Result<Option<Value>, String> {
        self.ecs_call("get_self_script_field", (field_name,))
    }

    pub fn set_self_script_field<S: Serialize>(
        &self,
        field_name: &str,
        value: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_self_script_field", (field_name, value))
    }

    pub fn set_script_data(
        &self,
        entity_id: EntityId,
        path: &str,
        language: Option<&str>,
    ) -> Result<bool, String> {
        self.set_script_data_at(entity_id, path, language, None)
    }

    pub fn set_script_data_at(
        &self,
        entity_id: EntityId,
        path: &str,
        language: Option<&str>,
        script_index: Option<usize>,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_script",
            (entity_id, path, language, lua_script_index(script_index)),
        )
    }

    pub fn list_dynamic_components(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("list_dynamic_components", (entity_id,))
    }

    pub fn list_dynamic_components_data(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<Vec<DynamicComponentData>>, String> {
        self.ecs_call("list_dynamic_components", (entity_id,))
    }

    pub fn get_dynamic_component(
        &self,
        entity_id: EntityId,
        component_name: &str,
    ) -> Result<Option<Value>, String> {
        self.ecs_call("get_dynamic_component", (entity_id, component_name))
    }

    pub fn get_dynamic_component_data(
        &self,
        entity_id: EntityId,
        component_name: &str,
    ) -> Result<Option<DynamicFields>, String> {
        self.ecs_call("get_dynamic_component", (entity_id, component_name))
    }

    pub fn set_dynamic_component<S: Serialize>(
        &self,
        entity_id: EntityId,
        component_name: &str,
        fields: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_dynamic_component", (entity_id, component_name, fields))
    }

    pub fn set_dynamic_component_data(
        &self,
        entity_id: EntityId,
        component_name: &str,
        fields: &DynamicFields,
    ) -> Result<bool, String> {
        self.ecs_call("set_dynamic_component", (entity_id, component_name, fields))
    }

    pub fn get_dynamic_field(
        &self,
        entity_id: EntityId,
        component_name: &str,
        field_name: &str,
    ) -> Result<Option<Value>, String> {
        self.ecs_call("get_dynamic_field", (entity_id, component_name, field_name))
    }

    pub fn get_dynamic_field_data(
        &self,
        entity_id: EntityId,
        component_name: &str,
        field_name: &str,
    ) -> Result<Option<DynamicFieldValue>, String> {
        self.ecs_call("get_dynamic_field", (entity_id, component_name, field_name))
    }

    pub fn set_dynamic_field<S: Serialize>(
        &self,
        entity_id: EntityId,
        component_name: &str,
        field_name: &str,
        value: S,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_dynamic_field",
            (entity_id, component_name, field_name, value),
        )
    }

    pub fn set_dynamic_field_data(
        &self,
        entity_id: EntityId,
        component_name: &str,
        field_name: &str,
        value: DynamicFieldValue,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_dynamic_field",
            (entity_id, component_name, field_name, value),
        )
    }

    pub fn remove_dynamic_component(
        &self,
        entity_id: EntityId,
        component_name: &str,
    ) -> Result<bool, String> {
        self.ecs_call("remove_dynamic_component", (entity_id, component_name))
    }

    pub fn remove_dynamic_field(
        &self,
        entity_id: EntityId,
        component_name: &str,
        field_name: &str,
    ) -> Result<bool, String> {
        self.ecs_call(
            "remove_dynamic_field",
            (entity_id, component_name, field_name),
        )
    }

    pub fn get_physics(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_physics", (entity_id,))
    }

    pub fn get_physics_data(&self, entity_id: EntityId) -> Result<Option<PhysicsData>, String> {
        self.ecs_call("get_physics", (entity_id,))
    }

    pub fn set_physics<S: Serialize>(&self, entity_id: EntityId, patch: S) -> Result<bool, String> {
        self.ecs_call("set_physics", (entity_id, patch))
    }

    pub fn set_physics_data(
        &self,
        entity_id: EntityId,
        patch: &PhysicsPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_physics", (entity_id, patch))
    }

    pub fn clear_physics(&self, entity_id: EntityId) -> Result<bool, String> {
        self.ecs_call("clear_physics", (entity_id,))
    }

    pub fn get_physics_world_defaults(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_physics_world_defaults", (entity_id,))
    }

    pub fn get_physics_world_defaults_data(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<PhysicsWorldDefaultsData>, String> {
        self.ecs_call("get_physics_world_defaults", (entity_id,))
    }

    pub fn set_physics_world_defaults<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_physics_world_defaults", (entity_id, patch))
    }

    pub fn set_physics_world_defaults_data(
        &self,
        entity_id: EntityId,
        patch: &PhysicsWorldDefaultsPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_physics_world_defaults", (entity_id, patch))
    }

    pub fn get_character_controller_output(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<CharacterControllerOutputData>, String> {
        self.ecs_call("get_character_controller_output", (entity_id,))
    }

    pub fn set_character_controller_desired_translation(
        &self,
        entity_id: EntityId,
        desired_translation: Vec3,
    ) -> Result<bool, String> {
        self.ecs_call(
            "set_character_controller_desired_translation",
            (entity_id, desired_translation),
        )
    }

    pub fn get_character_controller_desired_translation(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<Vec3>, String> {
        self.ecs_call("get_character_controller_desired_translation", (entity_id,))
    }

    pub fn get_character_controller_effective_translation(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<Vec3>, String> {
        self.ecs_call(
            "get_character_controller_effective_translation",
            (entity_id,),
        )
    }

    pub fn get_character_controller_remaining_translation(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<Vec3>, String> {
        self.ecs_call(
            "get_character_controller_remaining_translation",
            (entity_id,),
        )
    }

    pub fn get_character_controller_grounded(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<bool>, String> {
        self.ecs_call("get_character_controller_grounded", (entity_id,))
    }

    pub fn get_character_controller_sliding_down_slope(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<bool>, String> {
        self.ecs_call("get_character_controller_sliding_down_slope", (entity_id,))
    }

    pub fn get_character_controller_collision_count(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<u32>, String> {
        self.ecs_call("get_character_controller_collision_count", (entity_id,))
    }

    pub fn get_character_controller_ground_normal(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<Vec3>, String> {
        self.ecs_call("get_character_controller_ground_normal", (entity_id,))
    }

    pub fn get_character_controller_slope_angle(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<f32>, String> {
        self.ecs_call("get_character_controller_slope_angle", (entity_id,))
    }

    pub fn get_character_controller_hit_normal(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<Vec3>, String> {
        self.ecs_call("get_character_controller_hit_normal", (entity_id,))
    }

    pub fn get_character_controller_hit_point(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<Vec3>, String> {
        self.ecs_call("get_character_controller_hit_point", (entity_id,))
    }

    pub fn get_character_controller_hit_entity(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<EntityId>, String> {
        self.ecs_call("get_character_controller_hit_entity", (entity_id,))
    }

    pub fn get_character_controller_stepped_up(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<bool>, String> {
        self.ecs_call("get_character_controller_stepped_up", (entity_id,))
    }

    pub fn get_character_controller_step_height(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<f32>, String> {
        self.ecs_call("get_character_controller_step_height", (entity_id,))
    }

    pub fn get_character_controller_platform_velocity(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<Vec3>, String> {
        self.ecs_call("get_character_controller_platform_velocity", (entity_id,))
    }

    pub fn get_collision_events(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
    ) -> Result<Option<Value>, String> {
        self.ecs_call("get_collision_events", (entity_id, phase))
    }

    pub fn get_collision_event_count(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
    ) -> Result<usize, String> {
        let count: u64 = self.ecs_call("get_collision_event_count", (entity_id, phase))?;
        Ok(usize::try_from(count).unwrap_or(usize::MAX))
    }

    pub fn get_collision_event_other(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
        event_index: Option<usize>,
    ) -> Result<Option<EntityId>, String> {
        let other: u64 = self.ecs_call(
            "get_collision_event_other",
            (entity_id, phase, lua_script_index(event_index)),
        )?;
        Ok((other > 0).then_some(other))
    }

    pub fn get_collision_event_normal(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
        event_index: Option<usize>,
    ) -> Result<Vec3, String> {
        self.ecs_call(
            "get_collision_event_normal",
            (entity_id, phase, lua_script_index(event_index)),
        )
    }

    pub fn get_collision_event_point(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
        event_index: Option<usize>,
    ) -> Result<Vec3, String> {
        self.ecs_call(
            "get_collision_event_point",
            (entity_id, phase, lua_script_index(event_index)),
        )
    }

    pub fn get_trigger_events(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
    ) -> Result<Option<Value>, String> {
        self.ecs_call("get_trigger_events", (entity_id, phase))
    }

    pub fn get_trigger_event_count(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
    ) -> Result<usize, String> {
        let count: u64 = self.ecs_call("get_trigger_event_count", (entity_id, phase))?;
        Ok(usize::try_from(count).unwrap_or(usize::MAX))
    }

    pub fn get_trigger_event_other(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
        event_index: Option<usize>,
    ) -> Result<Option<EntityId>, String> {
        let other: u64 = self.ecs_call(
            "get_trigger_event_other",
            (entity_id, phase, lua_script_index(event_index)),
        )?;
        Ok((other > 0).then_some(other))
    }

    pub fn get_trigger_event_normal(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
        event_index: Option<usize>,
    ) -> Result<Vec3, String> {
        self.ecs_call(
            "get_trigger_event_normal",
            (entity_id, phase, lua_script_index(event_index)),
        )
    }

    pub fn get_trigger_event_point(
        &self,
        entity_id: EntityId,
        phase: Option<&str>,
        event_index: Option<usize>,
    ) -> Result<Vec3, String> {
        self.ecs_call(
            "get_trigger_event_point",
            (entity_id, phase, lua_script_index(event_index)),
        )
    }

    pub fn emit_event(&self, name: &str, target_entity: Option<EntityId>) -> Result<bool, String> {
        self.ecs_call("emit_event", (name, target_entity))
    }

    pub fn get_physics_ray_cast_hit(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<PhysicsRayCastHitData>, String> {
        self.ecs_call("get_physics_ray_cast_hit", (entity_id,))
    }

    pub fn ray_cast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: Option<f32>,
        solid: Option<bool>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<PhysicsRayCastHitData, String> {
        self.ecs_call(
            "ray_cast",
            (origin, direction, max_toi, solid, filter, exclude_entity),
        )
    }

    pub fn ray_cast_has_hit(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: Option<f32>,
        solid: Option<bool>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<bool, String> {
        self.ecs_call(
            "ray_cast_has_hit",
            (origin, direction, max_toi, solid, filter, exclude_entity),
        )
    }

    pub fn ray_cast_hit_entity(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: Option<f32>,
        solid: Option<bool>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<Option<EntityId>, String> {
        let entity: u64 = self.ecs_call(
            "ray_cast_hit_entity",
            (origin, direction, max_toi, solid, filter, exclude_entity),
        )?;
        Ok((entity > 0).then_some(entity))
    }

    pub fn ray_cast_point(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: Option<f32>,
        solid: Option<bool>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<Vec3, String> {
        self.ecs_call(
            "ray_cast_point",
            (origin, direction, max_toi, solid, filter, exclude_entity),
        )
    }

    pub fn ray_cast_normal(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: Option<f32>,
        solid: Option<bool>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<Vec3, String> {
        self.ecs_call(
            "ray_cast_normal",
            (origin, direction, max_toi, solid, filter, exclude_entity),
        )
    }

    pub fn ray_cast_toi(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: Option<f32>,
        solid: Option<bool>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<f32, String> {
        self.ecs_call(
            "ray_cast_toi",
            (origin, direction, max_toi, solid, filter, exclude_entity),
        )
    }

    pub fn sphere_cast(
        &self,
        origin: Vec3,
        radius: f32,
        direction: Vec3,
        max_toi: Option<f32>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<PhysicsShapeCastHitData, String> {
        self.ecs_call(
            "sphere_cast",
            (origin, radius, direction, max_toi, filter, exclude_entity),
        )
    }

    pub fn sphere_cast_has_hit(
        &self,
        origin: Vec3,
        radius: f32,
        direction: Vec3,
        max_toi: Option<f32>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<bool, String> {
        self.ecs_call(
            "sphere_cast_has_hit",
            (origin, radius, direction, max_toi, filter, exclude_entity),
        )
    }

    pub fn sphere_cast_hit_entity(
        &self,
        origin: Vec3,
        radius: f32,
        direction: Vec3,
        max_toi: Option<f32>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<Option<EntityId>, String> {
        let entity: u64 = self.ecs_call(
            "sphere_cast_hit_entity",
            (origin, radius, direction, max_toi, filter, exclude_entity),
        )?;
        Ok((entity > 0).then_some(entity))
    }

    pub fn sphere_cast_point(
        &self,
        origin: Vec3,
        radius: f32,
        direction: Vec3,
        max_toi: Option<f32>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<Vec3, String> {
        self.ecs_call(
            "sphere_cast_point",
            (origin, radius, direction, max_toi, filter, exclude_entity),
        )
    }

    pub fn sphere_cast_normal(
        &self,
        origin: Vec3,
        radius: f32,
        direction: Vec3,
        max_toi: Option<f32>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<Vec3, String> {
        self.ecs_call(
            "sphere_cast_normal",
            (origin, radius, direction, max_toi, filter, exclude_entity),
        )
    }

    pub fn sphere_cast_toi(
        &self,
        origin: Vec3,
        radius: f32,
        direction: Vec3,
        max_toi: Option<f32>,
        filter: Option<PhysicsQueryFilterData>,
        exclude_entity: Option<EntityId>,
    ) -> Result<f32, String> {
        self.ecs_call(
            "sphere_cast_toi",
            (origin, radius, direction, max_toi, filter, exclude_entity),
        )
    }

    pub fn get_physics_point_projection_hit(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<PhysicsPointProjectionHitData>, String> {
        self.ecs_call("get_physics_point_projection_hit", (entity_id,))
    }

    pub fn get_physics_shape_cast_hit(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<PhysicsShapeCastHitData>, String> {
        self.ecs_call("get_physics_shape_cast_hit", (entity_id,))
    }

    pub fn get_physics_velocity(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_physics_velocity", (entity_id,))
    }

    pub fn get_physics_velocity_data(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<PhysicsVelocityData>, String> {
        self.ecs_call("get_physics_velocity", (entity_id,))
    }

    pub fn set_physics_velocity<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_physics_velocity", (entity_id, patch))
    }

    pub fn set_physics_velocity_data(
        &self,
        entity_id: EntityId,
        patch: &PhysicsVelocityPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_physics_velocity", (entity_id, patch))
    }

    pub fn add_force(
        &self,
        entity_id: EntityId,
        force: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("add_force", (entity_id, force, wake_up))
    }

    pub fn add_torque(
        &self,
        entity_id: EntityId,
        torque: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("add_torque", (entity_id, torque, wake_up))
    }

    pub fn add_force_at_point(
        &self,
        entity_id: EntityId,
        force: Vec3,
        point: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("add_force_at_point", (entity_id, force, point, wake_up))
    }

    pub fn add_persistent_force(
        &self,
        entity_id: EntityId,
        force: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("add_persistent_force", (entity_id, force, wake_up))
    }

    pub fn set_persistent_force(
        &self,
        entity_id: EntityId,
        force: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("set_persistent_force", (entity_id, force, wake_up))
    }

    pub fn add_persistent_torque(
        &self,
        entity_id: EntityId,
        torque: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("add_persistent_torque", (entity_id, torque, wake_up))
    }

    pub fn set_persistent_torque(
        &self,
        entity_id: EntityId,
        torque: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("set_persistent_torque", (entity_id, torque, wake_up))
    }

    pub fn add_persistent_force_at_point(
        &self,
        entity_id: EntityId,
        force: Vec3,
        point: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call(
            "add_persistent_force_at_point",
            (entity_id, force, point, wake_up),
        )
    }

    pub fn clear_persistent_forces(&self, entity_id: EntityId) -> Result<bool, String> {
        self.ecs_call("clear_persistent_forces", (entity_id,))
    }

    pub fn apply_impulse(
        &self,
        entity_id: EntityId,
        impulse: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("apply_impulse", (entity_id, impulse, wake_up))
    }

    pub fn apply_angular_impulse(
        &self,
        entity_id: EntityId,
        impulse: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("apply_angular_impulse", (entity_id, impulse, wake_up))
    }

    pub fn apply_torque_impulse(
        &self,
        entity_id: EntityId,
        impulse: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call("apply_torque_impulse", (entity_id, impulse, wake_up))
    }

    pub fn apply_impulse_at_point(
        &self,
        entity_id: EntityId,
        impulse: Vec3,
        point: Vec3,
        wake_up: Option<bool>,
    ) -> Result<bool, String> {
        self.ecs_call(
            "apply_impulse_at_point",
            (entity_id, impulse, point, wake_up),
        )
    }

    pub fn set_physics_running(&self, running: bool) -> Result<bool, String> {
        self.ecs_call("set_physics_running", (running,))
    }

    pub fn get_physics_running(&self) -> Result<bool, String> {
        self.ecs_call("get_physics_running", ())
    }

    pub fn set_physics_gravity<S: Serialize>(&self, value: S) -> Result<bool, String> {
        self.ecs_call("set_physics_gravity", (value,))
    }

    pub fn set_physics_gravity_vec3(&self, value: Vec3) -> Result<bool, String> {
        self.ecs_call("set_physics_gravity", (value,))
    }

    pub fn get_physics_gravity(&self) -> Result<Option<Value>, String> {
        self.ecs_call("get_physics_gravity", ())
    }

    pub fn get_physics_gravity_vec3(&self) -> Result<Option<Vec3>, String> {
        self.ecs_call("get_physics_gravity", ())
    }

    pub fn get_audio_emitter(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_audio_emitter", (entity_id,))
    }

    pub fn get_audio_emitter_data(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<AudioEmitterData>, String> {
        self.ecs_call("get_audio_emitter", (entity_id,))
    }

    pub fn set_audio_emitter<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_audio_emitter", (entity_id, patch))
    }

    pub fn set_audio_emitter_data(
        &self,
        entity_id: EntityId,
        patch: &AudioEmitterPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_audio_emitter", (entity_id, patch))
    }

    pub fn get_audio_emitter_path(&self, entity_id: EntityId) -> Result<Option<String>, String> {
        self.ecs_call("get_audio_emitter_path", (entity_id,))
    }

    pub fn set_audio_emitter_path(&self, entity_id: EntityId, path: &str) -> Result<bool, String> {
        self.ecs_call("set_audio_emitter_path", (entity_id, path))
    }

    pub fn get_audio_listener(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_audio_listener", (entity_id,))
    }

    pub fn get_audio_listener_data(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<AudioListenerData>, String> {
        self.ecs_call("get_audio_listener", (entity_id,))
    }

    pub fn set_audio_listener<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_audio_listener", (entity_id, patch))
    }

    pub fn set_audio_listener_data(
        &self,
        entity_id: EntityId,
        patch: &AudioListenerPatch,
    ) -> Result<bool, String> {
        self.ecs_call("set_audio_listener", (entity_id, patch))
    }

    pub fn set_audio_enabled(&self, enabled: bool) -> Result<bool, String> {
        self.ecs_call("set_audio_enabled", (enabled,))
    }

    pub fn get_audio_enabled(&self) -> Result<bool, String> {
        self.ecs_call("get_audio_enabled", ())
    }

    pub fn list_audio_buses(&self) -> Result<Value, String> {
        self.ecs_call_value("list_audio_buses", ())
    }

    pub fn list_audio_buses_typed(&self) -> Result<Vec<AudioBus>, String> {
        self.ecs_call("list_audio_buses", ())
    }

    pub fn create_audio_bus(&self, name: Option<&str>) -> Result<Option<i64>, String> {
        self.ecs_call("create_audio_bus", (name,))
    }

    pub fn create_audio_bus_typed(&self, name: Option<&str>) -> Result<Option<AudioBus>, String> {
        self.ecs_call("create_audio_bus", (name,))
    }

    pub fn remove_audio_bus<B: Serialize>(&self, bus: B) -> Result<bool, String> {
        self.ecs_call("remove_audio_bus", (bus,))
    }

    pub fn get_audio_bus_name<B: Serialize>(&self, bus: B) -> Result<Option<String>, String> {
        self.ecs_call("get_audio_bus_name", (bus,))
    }

    pub fn set_audio_bus_name<B: Serialize>(&self, bus: B, name: &str) -> Result<bool, String> {
        self.ecs_call("set_audio_bus_name", (bus, name))
    }

    pub fn set_audio_bus_volume<B: Serialize>(&self, bus: B, volume: f32) -> Result<bool, String> {
        self.ecs_call("set_audio_bus_volume", (bus, volume))
    }

    pub fn get_audio_bus_volume<B: Serialize>(&self, bus: B) -> Result<Option<f32>, String> {
        self.ecs_call("get_audio_bus_volume", (bus,))
    }

    pub fn set_audio_scene_volume(&self, scene_id: u64, volume: f32) -> Result<bool, String> {
        self.ecs_call("set_audio_scene_volume", (scene_id, volume))
    }

    pub fn get_audio_scene_volume(&self, scene_id: u64) -> Result<Option<f32>, String> {
        self.ecs_call("get_audio_scene_volume", (scene_id,))
    }

    pub fn clear_audio_emitters(&self) -> Result<bool, String> {
        self.ecs_call("clear_audio_emitters", ())
    }

    pub fn set_audio_head_width(&self, width: f32) -> Result<bool, String> {
        self.ecs_call("set_audio_head_width", (width,))
    }

    pub fn get_audio_head_width(&self) -> Result<Option<f32>, String> {
        self.ecs_call("get_audio_head_width", ())
    }

    pub fn set_audio_speed_of_sound(&self, speed: f32) -> Result<bool, String> {
        self.ecs_call("set_audio_speed_of_sound", (speed,))
    }

    pub fn get_audio_speed_of_sound(&self) -> Result<Option<f32>, String> {
        self.ecs_call("get_audio_speed_of_sound", ())
    }

    pub fn set_audio_streaming_config(
        &self,
        buffer_frames: u32,
        chunk_frames: u32,
    ) -> Result<bool, String> {
        self.ecs_call("set_audio_streaming_config", (buffer_frames, chunk_frames))
    }

    pub fn get_audio_streaming_config(&self) -> Result<Option<AudioStreamingConfig>, String> {
        self.ecs_call("get_audio_streaming_config", ())
    }

    pub fn input_keys(&self) -> Result<Value, String> {
        self.input_call_value("keys", ())
    }

    pub fn input_mouse_buttons(&self) -> Result<Value, String> {
        self.input_call_value("mouse_buttons", ())
    }

    pub fn input_gamepad_buttons(&self) -> Result<Value, String> {
        self.input_call_value("gamepad_buttons", ())
    }

    pub fn input_gamepad_axes(&self) -> Result<Value, String> {
        self.input_call_value("gamepad_axes", ())
    }

    pub fn input_key_handle(&self, name: &str) -> Result<Option<String>, String> {
        self.input_call("key", (name,))
    }

    pub fn input_mouse_button_handle(&self, name: &str) -> Result<Option<String>, String> {
        self.input_call("mouse_button", (name,))
    }

    pub fn input_gamepad_button_handle(&self, name: &str) -> Result<Option<String>, String> {
        self.input_call("gamepad_button", (name,))
    }

    pub fn input_gamepad_axis_handle(&self, name: &str) -> Result<Option<String>, String> {
        self.input_call("gamepad_axis_handle", (name,))
    }

    pub fn input_gamepad_axis_ref(&self, name: &str) -> Result<Option<String>, String> {
        self.input_call("gamepad_axis_ref", (name,))
    }

    pub fn input_bind_action<S: Serialize>(
        &self,
        action: &str,
        binding: S,
        context: Option<&str>,
        deadzone: Option<f32>,
    ) -> Result<bool, String> {
        self.input_call("bind_action", (action, binding, context, deadzone))
    }

    pub fn input_unbind_action(&self, action: &str, context: Option<&str>) -> Result<bool, String> {
        self.input_call("unbind_action", (action, Option::<Value>::None, context))
    }

    pub fn input_unbind_action_binding<S: Serialize>(
        &self,
        action: &str,
        binding: S,
        context: Option<&str>,
    ) -> Result<bool, String> {
        self.input_call("unbind_action", (action, Some(binding), context))
    }

    pub fn input_set_action_context(&self, context: Option<&str>) -> Result<bool, String> {
        self.input_call("set_action_context", (context,))
    }

    pub fn input_action_context(&self) -> Result<String, String> {
        self.input_call("action_context", ())
    }

    pub fn input_action_value(&self, action: &str) -> Result<f32, String> {
        self.input_call("action_value", (action,))
    }

    pub fn input_action_down(&self, action: &str) -> Result<bool, String> {
        self.input_call("action_down", (action,))
    }

    pub fn input_action_pressed(&self, action: &str) -> Result<bool, String> {
        self.input_call("action_pressed", (action,))
    }

    pub fn input_action_released(&self, action: &str) -> Result<bool, String> {
        self.input_call("action_released", (action,))
    }

    pub fn input_key_down<K: Serialize>(&self, key: K) -> Result<bool, String> {
        self.input_call("key_down", (key,))
    }

    pub fn input_key_pressed<K: Serialize>(&self, key: K) -> Result<bool, String> {
        self.input_call("key_pressed", (key,))
    }

    pub fn input_key_released<K: Serialize>(&self, key: K) -> Result<bool, String> {
        self.input_call("key_released", (key,))
    }

    pub fn input_mouse_down<B: Serialize>(&self, button: B) -> Result<bool, String> {
        self.input_call("mouse_down", (button,))
    }

    pub fn input_mouse_pressed<B: Serialize>(&self, button: B) -> Result<bool, String> {
        self.input_call("mouse_pressed", (button,))
    }

    pub fn input_mouse_released<B: Serialize>(&self, button: B) -> Result<bool, String> {
        self.input_call("mouse_released", (button,))
    }

    pub fn input_cursor(&self) -> Result<Option<Vec2>, String> {
        self.input_call("cursor", ())
    }

    pub fn input_cursor_delta(&self) -> Result<Option<Vec2>, String> {
        self.input_call("cursor_delta", ())
    }

    pub fn input_wheel(&self) -> Result<Option<Vec2>, String> {
        self.input_call("wheel", ())
    }

    pub fn input_window_size(&self) -> Result<Option<Vec2>, String> {
        self.input_call("window_size", ())
    }

    pub fn input_scale_factor(&self) -> Result<Option<f64>, String> {
        self.input_call("scale_factor", ())
    }

    pub fn input_modifiers(&self) -> Result<Option<InputModifiers>, String> {
        self.input_call("modifiers", ())
    }

    pub fn input_wants_keyboard(&self) -> Result<bool, String> {
        self.input_call("wants_keyboard", ())
    }

    pub fn input_wants_pointer(&self) -> Result<bool, String> {
        self.input_call("wants_pointer", ())
    }

    pub fn input_cursor_grab_mode(&self) -> Result<String, String> {
        self.input_call("cursor_grab_mode", ())
    }

    pub fn input_set_cursor_visible(&self, visible: bool) -> Result<bool, String> {
        self.input_call("set_cursor_visible", (visible,))
    }

    pub fn input_set_cursor_grab<S: Serialize>(&self, mode: S) -> Result<bool, String> {
        self.input_call("set_cursor_grab", (mode,))
    }

    pub fn input_reset_cursor_control(&self) -> Result<bool, String> {
        self.input_call("reset_cursor_control", ())
    }

    pub fn input_gamepad_ids(&self) -> Result<Vec<u64>, String> {
        self.input_call("gamepad_ids", ())
    }

    pub fn input_gamepad_count(&self) -> Result<u32, String> {
        self.input_call("gamepad_count", ())
    }

    pub fn input_gamepad_axis<A: Serialize>(
        &self,
        axis: A,
        gamepad_id: Option<u64>,
    ) -> Result<f32, String> {
        self.input_call("gamepad_axis", (axis, gamepad_id))
    }

    pub fn input_gamepad_button_down<B: Serialize>(
        &self,
        button: B,
        gamepad_id: Option<u64>,
    ) -> Result<bool, String> {
        self.input_call("gamepad_button_down", (button, gamepad_id))
    }

    pub fn input_gamepad_button_pressed<B: Serialize>(
        &self,
        button: B,
        gamepad_id: Option<u64>,
    ) -> Result<bool, String> {
        self.input_call("gamepad_button_pressed", (button, gamepad_id))
    }

    pub fn input_gamepad_button_released<B: Serialize>(
        &self,
        button: B,
        gamepad_id: Option<u64>,
    ) -> Result<bool, String> {
        self.input_call("gamepad_button_released", (button, gamepad_id))
    }

    pub fn input_gamepad_trigger<S: Serialize>(
        &self,
        side: S,
        gamepad_id: Option<u64>,
    ) -> Result<f32, String> {
        self.input_call("gamepad_trigger", (side, gamepad_id))
    }

    pub fn invoke_json_raw(
        &self,
        table_name: &str,
        function_name: &str,
        args_json: &str,
    ) -> Result<String, String> {
        let table_name = to_c_string(table_name);
        let function_name = to_c_string(function_name);
        let args_json = to_c_string(args_json);

        let mut out_result = ScriptBytes::default();
        // SAFETY: The host owns the API table and function pointers for the entire callback
        let ok = unsafe {
            ((*self.api).invoke_json)(
                (*self.api).user_data,
                table_name.as_ptr(),
                function_name.as_ptr(),
                args_json.as_ptr(),
                &mut out_result,
            )
        };
        if ok == 0 {
            return Err(format!(
                "{}:{} invocation failed",
                table_name.to_string_lossy(),
                function_name.to_string_lossy()
            ));
        }

        let bytes = if out_result.ptr.is_null() || out_result.len == 0 {
            Vec::new()
        } else {
            // SAFETY: The host owns this buffer until `free_bytes` is called
            unsafe { slice::from_raw_parts(out_result.ptr as *const u8, out_result.len).to_vec() }
        };

        // SAFETY: The host owns the API table and function pointers for the entire callback
        unsafe {
            ((*self.api).free_bytes)((*self.api).user_data, out_result);
        }

        String::from_utf8(bytes).map_err(|err| err.to_string())
    }
}

fn lua_script_index(index: Option<usize>) -> Option<u64> {
    index
        .and_then(|index| index.checked_add(1))
        .and_then(|index| u64::try_from(index).ok())
}

fn from_lua_script_index(index: Option<u64>) -> Option<usize> {
    index
        .and_then(|index| index.checked_sub(1))
        .and_then(|index| usize::try_from(index).ok())
}

fn normalize_call_args(args: Value) -> Value {
    match args {
        Value::Null => Value::Array(Vec::new()),
        Value::Array(_) => args,
        other => Value::Array(vec![other]),
    }
}

fn to_c_string(value: &str) -> CString {
    let sanitized = value.replace('\0', " ");
    CString::new(sanitized).unwrap_or_else(|_| CString::new("").expect("CString::new failed"))
}

pub fn encode_state<T: Serialize>(value: &T) -> Option<Vec<u8>> {
    serde_json::to_vec(value).ok()
}

pub fn decode_state<T: DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    serde_json::from_slice(bytes).ok()
}

pub trait Script: Default + Send + 'static {
    fn on_start(&mut self, _host: &Host) {}
    fn on_update(&mut self, _host: &Host, _dt: f32) {}
    fn on_stop(&mut self, _host: &Host) {}

    fn save_state(&self) -> Option<Vec<u8>> {
        None
    }

    fn load_state(&mut self, _state: &[u8]) -> bool {
        false
    }
}

struct ScriptState<T: Script> {
    script: T,
    api: *const ScriptApi,
    entity_id: EntityId,
}

impl<T: Script> ScriptState<T> {
    fn host(&self) -> Host {
        Host {
            api: self.api,
            entity_id: self.entity_id,
        }
    }
}

unsafe extern "C" fn create_instance<T: Script>(
    api: *const ScriptApi,
    entity_id: EntityId,
) -> *mut c_void {
    if api.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: The caller provides a valid API table for plugin callbacks
    let api_ref = unsafe { &*api };
    if api_ref.abi_version != SCRIPT_API_ABI_VERSION {
        return ptr::null_mut();
    }

    let state = ScriptState::<T> {
        script: T::default(),
        api,
        entity_id,
    };
    Box::into_raw(Box::new(state)) as *mut c_void
}

unsafe extern "C" fn destroy_instance<T: Script>(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    // SAFETY: The pointer was allocated in create_instance and is unique here
    unsafe {
        let _ = Box::from_raw(instance as *mut ScriptState<T>);
    }
}

unsafe extern "C" fn on_start_instance<T: Script>(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    // SAFETY: The pointer was allocated in create_instance and is valid during callbacks
    let state = unsafe { &mut *(instance as *mut ScriptState<T>) };
    state.script.on_start(&state.host());
}

unsafe extern "C" fn on_update_instance<T: Script>(instance: *mut c_void, dt: f32) {
    if instance.is_null() {
        return;
    }
    // SAFETY: The pointer was allocated in create_instance and is valid during callbacks
    let state = unsafe { &mut *(instance as *mut ScriptState<T>) };
    state.script.on_update(&state.host(), dt);
}

unsafe extern "C" fn on_stop_instance<T: Script>(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    // SAFETY: The pointer was allocated in create_instance and is valid during callbacks
    let state = unsafe { &mut *(instance as *mut ScriptState<T>) };
    state.script.on_stop(&state.host());
}

unsafe extern "C" fn save_state_instance<T: Script>(
    instance: *mut c_void,
    out_state: *mut ScriptBytes,
) -> u8 {
    if instance.is_null() || out_state.is_null() {
        return 0;
    }
    // SAFETY: The pointer was allocated in create_instance and is valid during callbacks
    let state = unsafe { &mut *(instance as *mut ScriptState<T>) };
    let Some(serialized) = state.script.save_state() else {
        return 0;
    };
    // SAFETY: `out_state` is validated non-null above
    unsafe {
        *out_state = bytes_from_vec(serialized);
    }
    1
}

unsafe extern "C" fn load_state_instance<T: Script>(
    instance: *mut c_void,
    state_bytes: ScriptBytesView,
) -> u8 {
    if instance.is_null() {
        return 0;
    }
    if state_bytes.ptr.is_null() && state_bytes.len > 0 {
        return 0;
    }

    // SAFETY: The pointer was allocated in create_instance and is valid during callbacks
    let state = unsafe { &mut *(instance as *mut ScriptState<T>) };
    let serialized = if state_bytes.ptr.is_null() || state_bytes.len == 0 {
        &[][..]
    } else {
        // SAFETY: `state_bytes.ptr` is checked above and remains valid for the callback duration
        unsafe { slice::from_raw_parts(state_bytes.ptr, state_bytes.len) }
    };

    if state.script.load_state(serialized) {
        1
    } else {
        0
    }
}

unsafe extern "C" fn free_state_buffer(buffer: ScriptBytes) {
    // SAFETY: The pointer is expected to originate from `save_state_instance`
    unsafe { drop_script_bytes(buffer) }
}

pub fn plugin_for<T: Script>() -> ScriptPlugin {
    ScriptPlugin {
        abi_version: SCRIPT_PLUGIN_ABI_VERSION,
        create: create_instance::<T>,
        destroy: destroy_instance::<T>,
        on_start: on_start_instance::<T>,
        on_update: on_update_instance::<T>,
        on_stop: on_stop_instance::<T>,
        save_state: save_state_instance::<T>,
        load_state: load_state_instance::<T>,
        free_state: free_state_buffer,
    }
}

#[macro_export]
macro_rules! export_script {
    ($script_ty:ty) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn helmer_get_script_plugin() -> *const $crate::ScriptPlugin {
            static PLUGIN: ::std::sync::OnceLock<$crate::ScriptPlugin> =
                ::std::sync::OnceLock::new();
            PLUGIN.get_or_init(|| $crate::plugin_for::<$script_ty>()) as *const $crate::ScriptPlugin
        }
    };
}
