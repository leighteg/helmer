use std::{
    any::Any,
    borrow::Cow,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use bevy_ecs::prelude::{Resource, World};
use egui::{Color32, ComboBox, DragValue, RichText, Sense, Stroke, TextEdit, Ui};
use egui_snarl::ui::{
    AnyPins, BackgroundPattern, Grid, NodeLayout, NodeLayoutKind, PinInfo, PinPlacement, PinShape,
    SelectionStyle, SnarlStyle, SnarlViewer, SnarlWidget, WireLayer, WireStyle,
};
use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};
use glam::DQuat;
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

use crate::editor::{AssetDragPayload, EditorProject, assets::PrimitiveKind};

pub const VISUAL_SCRIPT_EXTENSION: &str = "hvs";
pub const VISUAL_SCRIPT_FUNCTION_EXTENSION: &str = "hvsf";
const VISUAL_SCRIPT_VERSION: u32 = 1;
const MAX_API_ARGS: usize = 16;
const MAX_FUNCTION_IO_PORTS: usize = 16;
const MAX_FUNCTION_CALL_DEPTH: usize = 32;
const MAX_EXEC_STEPS_PER_EVENT: u32 = 10_000;
const MAX_LOOP_ITERATIONS: u32 = 4_096;
const PHYSICS_QUERY_FLAG_EXCLUDE_FIXED: u32 = 1 << 0;
const PHYSICS_QUERY_FLAG_EXCLUDE_KINEMATIC: u32 = 1 << 1;
const PHYSICS_QUERY_FLAG_EXCLUDE_DYNAMIC: u32 = 1 << 2;
const PHYSICS_QUERY_FLAG_EXCLUDE_SENSORS: u32 = 1 << 3;
const PHYSICS_QUERY_FLAG_EXCLUDE_SOLIDS: u32 = 1 << 4;
const VISUAL_MESH_PRIMITIVE_CHOICES: [&str; 6] = [
    "Cube",
    "UV Sphere",
    "Icosphere",
    "Cylinder",
    "Capsule",
    "Plane",
];

const PIN_COLOR_EVENT: Color32 = Color32::from_rgb(67, 160, 71);
const PIN_COLOR_EXEC: Color32 = Color32::from_rgb(74, 144, 226);
const PIN_COLOR_CONTROL: Color32 = Color32::from_rgb(255, 167, 38);
const PIN_COLOR_DATA: Color32 = Color32::from_rgb(234, 196, 53);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualValueType {
    Bool,
    Number,
    String,
    Entity,
    Array,
    Vec2,
    Vec3,
    Quat,
    Transform,
    Camera,
    Light,
    MeshRenderer,
    SpriteRenderer,
    Text2d,
    AudioEmitter,
    AudioListener,
    Script,
    LookAt,
    EntityFollower,
    AnimatorState,
    InputModifiers,
    AudioStreamingConfig,
    RuntimeTuning,
    RuntimeConfig,
    RenderConfig,
    ShaderConstants,
    StreamingTuning,
    RenderPasses,
    GpuBudget,
    AssetBudgets,
    WindowSettings,
    Spline,
    Physics,
    PhysicsVelocity,
    PhysicsWorldDefaults,
    CharacterControllerOutput,
    DynamicComponentFields,
    DynamicFieldValue,
    PhysicsQueryFilter,
    PhysicsRayCastHit,
    PhysicsPointProjectionHit,
    PhysicsShapeCastHit,
    #[serde(rename = "any", alias = "json")]
    #[default]
    Any,
}

impl VisualValueType {
    fn title(self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::Number => "Number",
            Self::String => "String",
            Self::Entity => "Entity",
            Self::Array => "Array",
            Self::Vec2 => "Vec2",
            Self::Vec3 => "Vec3",
            Self::Quat => "Quat",
            Self::Transform => "Transform",
            Self::Camera => "Camera",
            Self::Light => "Light",
            Self::MeshRenderer => "Mesh Renderer",
            Self::SpriteRenderer => "Sprite Renderer",
            Self::Text2d => "Text 2D",
            Self::AudioEmitter => "Audio Emitter",
            Self::AudioListener => "Audio Listener",
            Self::Script => "Script",
            Self::LookAt => "Look At",
            Self::EntityFollower => "Entity Follower",
            Self::AnimatorState => "Animator State",
            Self::InputModifiers => "Input Modifiers",
            Self::AudioStreamingConfig => "Audio Streaming Config",
            Self::RuntimeTuning => "Runtime Tuning",
            Self::RuntimeConfig => "Runtime Config",
            Self::RenderConfig => "Render Config",
            Self::ShaderConstants => "Shader Constants",
            Self::StreamingTuning => "Streaming Tuning",
            Self::RenderPasses => "Render Passes",
            Self::GpuBudget => "GPU Budget",
            Self::AssetBudgets => "Asset Budgets",
            Self::WindowSettings => "Window Settings",
            Self::Spline => "Spline",
            Self::Physics => "Physics",
            Self::PhysicsVelocity => "Physics Velocity",
            Self::PhysicsWorldDefaults => "Physics World Defaults",
            Self::CharacterControllerOutput => "Character Output",
            Self::DynamicComponentFields => "Dynamic Fields",
            Self::DynamicFieldValue => "Dynamic Value",
            Self::PhysicsQueryFilter => "Physics Query Filter",
            Self::PhysicsRayCastHit => "Ray Cast Hit",
            Self::PhysicsPointProjectionHit => "Point Projection Hit",
            Self::PhysicsShapeCastHit => "Shape Cast Hit",
            Self::Any => "Any",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualInspectorAssetKind {
    #[default]
    Any,
    Scene,
    Model,
    Material,
    Audio,
    Script,
    Animation,
}

impl VisualInspectorAssetKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::Any => "Any",
            Self::Scene => "Scene",
            Self::Model => "Model",
            Self::Material => "Material",
            Self::Audio => "Audio",
            Self::Script => "Script",
            Self::Animation => "Animation",
        }
    }
}

const VISUAL_VALUE_TYPE_CHOICES_NO_ANY: [VisualValueType; 42] = [
    VisualValueType::Bool,
    VisualValueType::Number,
    VisualValueType::String,
    VisualValueType::Entity,
    VisualValueType::Array,
    VisualValueType::Vec2,
    VisualValueType::Vec3,
    VisualValueType::Quat,
    VisualValueType::Transform,
    VisualValueType::Camera,
    VisualValueType::Light,
    VisualValueType::MeshRenderer,
    VisualValueType::SpriteRenderer,
    VisualValueType::Text2d,
    VisualValueType::AudioEmitter,
    VisualValueType::AudioListener,
    VisualValueType::Script,
    VisualValueType::LookAt,
    VisualValueType::EntityFollower,
    VisualValueType::AnimatorState,
    VisualValueType::InputModifiers,
    VisualValueType::AudioStreamingConfig,
    VisualValueType::RuntimeTuning,
    VisualValueType::RuntimeConfig,
    VisualValueType::RenderConfig,
    VisualValueType::ShaderConstants,
    VisualValueType::StreamingTuning,
    VisualValueType::RenderPasses,
    VisualValueType::GpuBudget,
    VisualValueType::AssetBudgets,
    VisualValueType::WindowSettings,
    VisualValueType::Spline,
    VisualValueType::Physics,
    VisualValueType::PhysicsVelocity,
    VisualValueType::PhysicsWorldDefaults,
    VisualValueType::CharacterControllerOutput,
    VisualValueType::DynamicComponentFields,
    VisualValueType::DynamicFieldValue,
    VisualValueType::PhysicsQueryFilter,
    VisualValueType::PhysicsRayCastHit,
    VisualValueType::PhysicsPointProjectionHit,
    VisualValueType::PhysicsShapeCastHit,
];

const VISUAL_ARRAY_ITEM_TYPE_CHOICES: [VisualValueType; 26] = [
    VisualValueType::Bool,
    VisualValueType::Number,
    VisualValueType::String,
    VisualValueType::Entity,
    VisualValueType::Vec2,
    VisualValueType::Vec3,
    VisualValueType::Quat,
    VisualValueType::Transform,
    VisualValueType::Camera,
    VisualValueType::Light,
    VisualValueType::MeshRenderer,
    VisualValueType::SpriteRenderer,
    VisualValueType::Text2d,
    VisualValueType::AudioEmitter,
    VisualValueType::AudioListener,
    VisualValueType::Script,
    VisualValueType::InputModifiers,
    VisualValueType::AudioStreamingConfig,
    VisualValueType::Spline,
    VisualValueType::Physics,
    VisualValueType::PhysicsVelocity,
    VisualValueType::PhysicsWorldDefaults,
    VisualValueType::CharacterControllerOutput,
    VisualValueType::DynamicFieldValue,
    VisualValueType::PhysicsRayCastHit,
    VisualValueType::PhysicsShapeCastHit,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisualVariableDefinition {
    #[serde(default)]
    pub id: u64,
    #[serde(default = "default_var_name")]
    pub name: String,
    #[serde(default)]
    pub value_type: VisualValueType,
    #[serde(default)]
    pub array_item_type: Option<VisualValueType>,
    #[serde(default)]
    pub default_value: String,
    #[serde(default)]
    pub inspector_exposed: bool,
    #[serde(default)]
    pub inspector_label: String,
    #[serde(default)]
    pub inspector_asset_kind: Option<VisualInspectorAssetKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct VisualFunctionIoDefinition {
    #[serde(default)]
    pub id: u64,
    #[serde(default = "default_function_io_name")]
    pub name: String,
    #[serde(default)]
    pub value_type: VisualValueType,
    #[serde(default)]
    pub array_item_type: Option<VisualValueType>,
    #[serde(default)]
    pub default_value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct VisualScriptFunctionDefinition {
    #[serde(default)]
    pub id: u64,
    #[serde(default = "default_function_name")]
    pub name: String,
    #[serde(default)]
    pub source_path: String,
    #[serde(default)]
    pub inputs: Vec<VisualFunctionIoDefinition>,
    #[serde(default)]
    pub outputs: Vec<VisualFunctionIoDefinition>,
    #[serde(default)]
    pub graph: VisualScriptGraphData,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VisualScriptFunctionAsset {
    #[serde(default = "default_visual_script_version")]
    pub version: u32,
    #[serde(default)]
    pub function: VisualScriptFunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct VisualScriptDocument {
    #[serde(default = "default_visual_script_version")]
    pub version: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub prelude: String,
    #[serde(default)]
    pub variables: Vec<VisualVariableDefinition>,
    #[serde(default)]
    pub functions: Vec<VisualScriptFunctionDefinition>,
    #[serde(default)]
    pub graph: VisualScriptGraphData,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct VisualScriptGraphData {
    #[serde(default)]
    pub nodes: Vec<VisualScriptNodeRecord>,
    #[serde(default)]
    pub wires: Vec<VisualScriptWireRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VisualScriptNodeRecord {
    pub id: u64,
    pub kind: VisualScriptNodeKind,
    #[serde(default)]
    pub pos: [i32; 2],
    #[serde(default = "default_true")]
    pub open: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisualScriptWireRecord {
    pub from_node: u64,
    pub from_pin: usize,
    pub to_node: u64,
    pub to_pin: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualScriptApiTable {
    #[default]
    Ecs,
    Input,
}

impl VisualScriptApiTable {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ecs => "ecs",
            Self::Input => "input",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Ecs => "ECS",
            Self::Input => "Input",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualApiOperation {
    #[default]
    EcsAddComponent,
    EcsAddForce,
    EcsAddForceAtPoint,
    EcsAddPersistentForce,
    EcsAddPersistentForceAtPoint,
    EcsAddPersistentTorque,
    EcsAddSplinePoint,
    EcsAddTorque,
    EcsApplyAngularImpulse,
    EcsApplyImpulse,
    EcsApplyImpulseAtPoint,
    EcsApplyTorqueImpulse,
    EcsClearAudioEmitters,
    EcsClearPersistentForces,
    EcsClearPhysics,
    EcsCreateAudioBus,
    EcsDeleteEntity,
    EcsEntityExists,
    EcsEmitEvent,
    EcsFindEntityByName,
    EcsFindScriptIndex,
    EcsFollowSpline,
    EcsGetAnimatorClips,
    EcsGetAnimatorLayerWeights,
    EcsGetAnimatorLayerWeight,
    EcsGetAnimatorState,
    EcsGetAnimatorStateTime,
    EcsGetAnimatorCurrentState,
    EcsGetAnimatorCurrentStateName,
    EcsGetAnimatorTransitionActive,
    EcsGetAnimatorTransitionFrom,
    EcsGetAnimatorTransitionTo,
    EcsGetAnimatorTransitionProgress,
    EcsGetAnimatorTransitionElapsed,
    EcsGetAnimatorTransitionDuration,
    EcsGetAudioBusName,
    EcsGetAudioBusVolume,
    EcsGetAudioEmitter,
    EcsGetAudioEmitterPath,
    EcsGetAudioEnabled,
    EcsGetAudioHeadWidth,
    EcsGetAudioListener,
    EcsGetAudioSceneVolume,
    EcsGetAudioSpeedOfSound,
    EcsGetAudioStreamingConfig,
    EcsGetCamera,
    EcsGetCharacterControllerOutput,
    EcsGetCharacterControllerDesiredTranslation,
    EcsGetCharacterControllerEffectiveTranslation,
    EcsGetCharacterControllerRemainingTranslation,
    EcsGetCharacterControllerGrounded,
    EcsGetCharacterControllerSlidingDownSlope,
    EcsGetCharacterControllerCollisionCount,
    EcsGetCharacterControllerGroundNormal,
    EcsGetCharacterControllerSlopeAngle,
    EcsGetCharacterControllerHitNormal,
    EcsGetCharacterControllerHitPoint,
    EcsGetCharacterControllerHitEntity,
    EcsGetCharacterControllerSteppedUp,
    EcsGetCharacterControllerStepHeight,
    EcsGetCharacterControllerPlatformVelocity,
    EcsGetCollisionEventCount,
    EcsGetCollisionEventOther,
    EcsGetCollisionEventNormal,
    EcsGetCollisionEventPoint,
    EcsGetDynamicComponent,
    EcsGetDynamicField,
    EcsGetEntityFollower,
    EcsGetEntityName,
    EcsGetLight,
    EcsGetLookAt,
    EcsGetMeshRenderer,
    EcsGetMeshRendererSourcePath,
    EcsGetMeshRendererMaterialPath,
    EcsGetSpriteRenderer,
    EcsGetSpriteRendererTexturePath,
    EcsGetText2d,
    EcsGetPhysics,
    EcsGetPhysicsGravity,
    EcsGetPhysicsPointProjectionHit,
    EcsGetPhysicsRayCastHit,
    EcsRayCast,
    EcsRayCastHasHit,
    EcsRayCastHitEntity,
    EcsRayCastPoint,
    EcsRayCastNormal,
    EcsRayCastToi,
    EcsSphereCast,
    EcsSphereCastHasHit,
    EcsSphereCastHitEntity,
    EcsSphereCastPoint,
    EcsSphereCastNormal,
    EcsSphereCastToi,
    EcsGetPhysicsRunning,
    EcsGetPhysicsShapeCastHit,
    EcsGetPhysicsVelocity,
    EcsGetPhysicsWorldDefaults,
    EcsGetSceneAsset,
    EcsGetScript,
    EcsGetScriptCount,
    EcsGetScriptField,
    EcsGetScriptPath,
    EcsGetScriptLanguage,
    EcsGetSelfScriptField,
    EcsGetSpline,
    EcsGetTransform,
    EcsGetTransformForward,
    EcsGetTransformRight,
    EcsGetTransformUp,
    EcsGetTriggerEventCount,
    EcsGetTriggerEventOther,
    EcsGetTriggerEventNormal,
    EcsGetTriggerEventPoint,
    EcsGetViewportMode,
    EcsGetViewportPreviewCamera,
    EcsHasComponent,
    EcsListAudioBuses,
    EcsListDynamicComponents,
    EcsListEntities,
    EcsListScriptFields,
    EcsListSelfScriptFields,
    EcsOpenScene,
    EcsPlayAnimClip,
    EcsRemoveAudioBus,
    EcsRemoveComponent,
    EcsRemoveDynamicComponent,
    EcsRemoveDynamicField,
    EcsRemoveSplinePoint,
    EcsSampleSpline,
    EcsSetActiveCamera,
    EcsSetAnimatorBlendChild,
    EcsSetAnimatorBlendNode,
    EcsSetAnimatorEnabled,
    EcsSetAnimatorLayerWeight,
    EcsSetAnimatorParamBool,
    EcsSetAnimatorParamFloat,
    EcsSetAnimatorTimeScale,
    EcsSetAnimatorTransition,
    EcsSetAudioBusName,
    EcsSetAudioBusVolume,
    EcsSetAudioEmitter,
    EcsSetAudioEmitterPath,
    EcsSetAudioEnabled,
    EcsSetAudioHeadWidth,
    EcsSetAudioListener,
    EcsSetAudioSceneVolume,
    EcsSetAudioSpeedOfSound,
    EcsSetAudioStreamingConfig,
    EcsSetCamera,
    EcsSetDynamicComponent,
    EcsSetDynamicField,
    EcsSetEntityFollower,
    EcsSetEntityName,
    EcsSetLight,
    EcsSetLookAt,
    EcsSetMeshRenderer,
    EcsSetMeshRendererSourcePath,
    EcsSetMeshRendererMaterialPath,
    EcsSetSpriteRenderer,
    EcsSetSpriteRendererSheetAnimation,
    EcsSetSpriteRendererSequence,
    EcsSetSpriteRendererTexturePath,
    EcsSetText2d,
    EcsSetPersistentForce,
    EcsSetPersistentTorque,
    EcsSetPhysics,
    EcsSetPhysicsGravity,
    EcsSetPhysicsRunning,
    EcsSetPhysicsVelocity,
    EcsSetPhysicsWorldDefaults,
    EcsSetSceneAsset,
    EcsSetScript,
    EcsSetScriptField,
    EcsSetSelfScriptField,
    EcsSetCharacterControllerDesiredTranslation,
    EcsSetSpline,
    EcsSetSplinePoint,
    EcsSetTransform,
    EcsSetViewportMode,
    EcsSetViewportPreviewCamera,
    EcsSpawnEntity,
    EcsSplineLength,
    EcsSwitchScene,
    EcsSelfScriptIndex,
    EcsTriggerAnimator,
    EcsListGraphTemplates,
    EcsGetGraphTemplate,
    EcsSetGraphTemplate,
    EcsGetRuntimeTuning,
    EcsSetRuntimeTuning,
    EcsGetRuntimeConfig,
    EcsSetRuntimeConfig,
    EcsGetRenderConfig,
    EcsSetRenderConfig,
    EcsGetShaderConstants,
    EcsSetShaderConstants,
    EcsGetStreamingTuning,
    EcsSetStreamingTuning,
    EcsGetRenderPasses,
    EcsSetRenderPasses,
    EcsGetGpuBudget,
    EcsSetGpuBudget,
    EcsGetAssetBudgets,
    EcsSetAssetBudgets,
    EcsGetWindowSettings,
    EcsSetWindowSettings,
    InputActionContext,
    InputActionDown,
    InputActionPressed,
    InputActionReleased,
    InputActionValue,
    InputBindAction,
    InputCursor,
    InputCursorDelta,
    InputCursorGrabMode,
    InputGamepadAxis,
    InputGamepadButton,
    InputGamepadButtonDown,
    InputGamepadButtonPressed,
    InputGamepadButtonReleased,
    InputGamepadCount,
    InputGamepadIds,
    InputGamepadTrigger,
    InputKey,
    InputKeyDown,
    InputKeyPressed,
    InputKeyReleased,
    InputModifiers,
    InputMouseButton,
    InputMouseDown,
    InputMousePressed,
    InputMouseReleased,
    InputScaleFactor,
    InputSetActionContext,
    InputSetCursorVisible,
    InputSetCursorGrab,
    InputUnbindAction,
    InputResetCursorControl,
    InputWantsKeyboard,
    InputWantsPointer,
    InputWheel,
    InputWindowSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualApiFlow {
    Exec,
    Pure,
}

#[derive(Debug, Clone, Copy)]
struct VisualApiInputSpec {
    label: &'static str,
    value_type: VisualValueType,
}

#[derive(Debug, Clone, Copy)]
struct VisualApiOperationSpec {
    operation: VisualApiOperation,
    table: VisualScriptApiTable,
    function: &'static str,
    title: &'static str,
    category: &'static str,
    flow: VisualApiFlow,
    inputs: &'static [VisualApiInputSpec],
    output_type: Option<VisualValueType>,
}

impl VisualApiOperation {
    fn spec(self) -> &'static VisualApiOperationSpec {
        for spec in &VISUAL_API_OPERATION_SPECS {
            if spec.operation == self {
                return spec;
            }
        }
        &VISUAL_API_OPERATION_SPECS[0]
    }

    fn from_table_function(table: VisualScriptApiTable, function: &str) -> Option<Self> {
        let function = function.trim();
        for spec in &VISUAL_API_OPERATION_SPECS {
            if spec.table == table && spec.function == function {
                return Some(spec.operation);
            }
        }
        None
    }

    #[allow(dead_code)]
    fn all() -> &'static [Self] {
        &VISUAL_API_OPERATION_ALL
    }
}

#[allow(dead_code)]
const VISUAL_API_OPERATION_ALL: [VisualApiOperation; 242] = [
    VisualApiOperation::EcsAddComponent,
    VisualApiOperation::EcsAddForce,
    VisualApiOperation::EcsAddForceAtPoint,
    VisualApiOperation::EcsAddPersistentForce,
    VisualApiOperation::EcsAddPersistentForceAtPoint,
    VisualApiOperation::EcsAddPersistentTorque,
    VisualApiOperation::EcsAddSplinePoint,
    VisualApiOperation::EcsAddTorque,
    VisualApiOperation::EcsApplyAngularImpulse,
    VisualApiOperation::EcsApplyImpulse,
    VisualApiOperation::EcsApplyImpulseAtPoint,
    VisualApiOperation::EcsApplyTorqueImpulse,
    VisualApiOperation::EcsClearAudioEmitters,
    VisualApiOperation::EcsClearPersistentForces,
    VisualApiOperation::EcsClearPhysics,
    VisualApiOperation::EcsCreateAudioBus,
    VisualApiOperation::EcsDeleteEntity,
    VisualApiOperation::EcsEntityExists,
    VisualApiOperation::EcsEmitEvent,
    VisualApiOperation::EcsFindEntityByName,
    VisualApiOperation::EcsFindScriptIndex,
    VisualApiOperation::EcsFollowSpline,
    VisualApiOperation::EcsGetAnimatorClips,
    VisualApiOperation::EcsGetAnimatorLayerWeights,
    VisualApiOperation::EcsGetAnimatorLayerWeight,
    VisualApiOperation::EcsGetAnimatorState,
    VisualApiOperation::EcsGetAnimatorStateTime,
    VisualApiOperation::EcsGetAnimatorCurrentState,
    VisualApiOperation::EcsGetAnimatorCurrentStateName,
    VisualApiOperation::EcsGetAnimatorTransitionActive,
    VisualApiOperation::EcsGetAnimatorTransitionFrom,
    VisualApiOperation::EcsGetAnimatorTransitionTo,
    VisualApiOperation::EcsGetAnimatorTransitionProgress,
    VisualApiOperation::EcsGetAnimatorTransitionElapsed,
    VisualApiOperation::EcsGetAnimatorTransitionDuration,
    VisualApiOperation::EcsGetAudioBusName,
    VisualApiOperation::EcsGetAudioBusVolume,
    VisualApiOperation::EcsGetAudioEmitter,
    VisualApiOperation::EcsGetAudioEmitterPath,
    VisualApiOperation::EcsGetAudioEnabled,
    VisualApiOperation::EcsGetAudioHeadWidth,
    VisualApiOperation::EcsGetAudioListener,
    VisualApiOperation::EcsGetAudioSceneVolume,
    VisualApiOperation::EcsGetAudioSpeedOfSound,
    VisualApiOperation::EcsGetAudioStreamingConfig,
    VisualApiOperation::EcsGetCamera,
    VisualApiOperation::EcsGetCharacterControllerOutput,
    VisualApiOperation::EcsGetCharacterControllerDesiredTranslation,
    VisualApiOperation::EcsGetCharacterControllerEffectiveTranslation,
    VisualApiOperation::EcsGetCharacterControllerRemainingTranslation,
    VisualApiOperation::EcsGetCharacterControllerGrounded,
    VisualApiOperation::EcsGetCharacterControllerSlidingDownSlope,
    VisualApiOperation::EcsGetCharacterControllerCollisionCount,
    VisualApiOperation::EcsGetCharacterControllerGroundNormal,
    VisualApiOperation::EcsGetCharacterControllerSlopeAngle,
    VisualApiOperation::EcsGetCharacterControllerHitNormal,
    VisualApiOperation::EcsGetCharacterControllerHitPoint,
    VisualApiOperation::EcsGetCharacterControllerHitEntity,
    VisualApiOperation::EcsGetCharacterControllerSteppedUp,
    VisualApiOperation::EcsGetCharacterControllerStepHeight,
    VisualApiOperation::EcsGetCharacterControllerPlatformVelocity,
    VisualApiOperation::EcsGetCollisionEventCount,
    VisualApiOperation::EcsGetCollisionEventOther,
    VisualApiOperation::EcsGetCollisionEventNormal,
    VisualApiOperation::EcsGetCollisionEventPoint,
    VisualApiOperation::EcsGetDynamicComponent,
    VisualApiOperation::EcsGetDynamicField,
    VisualApiOperation::EcsGetEntityFollower,
    VisualApiOperation::EcsGetEntityName,
    VisualApiOperation::EcsGetLight,
    VisualApiOperation::EcsGetLookAt,
    VisualApiOperation::EcsGetMeshRenderer,
    VisualApiOperation::EcsGetMeshRendererSourcePath,
    VisualApiOperation::EcsGetMeshRendererMaterialPath,
    VisualApiOperation::EcsGetSpriteRenderer,
    VisualApiOperation::EcsGetSpriteRendererTexturePath,
    VisualApiOperation::EcsGetText2d,
    VisualApiOperation::EcsGetPhysics,
    VisualApiOperation::EcsGetPhysicsGravity,
    VisualApiOperation::EcsGetPhysicsPointProjectionHit,
    VisualApiOperation::EcsGetPhysicsRayCastHit,
    VisualApiOperation::EcsRayCast,
    VisualApiOperation::EcsRayCastHasHit,
    VisualApiOperation::EcsRayCastHitEntity,
    VisualApiOperation::EcsRayCastPoint,
    VisualApiOperation::EcsRayCastNormal,
    VisualApiOperation::EcsRayCastToi,
    VisualApiOperation::EcsSphereCast,
    VisualApiOperation::EcsSphereCastHasHit,
    VisualApiOperation::EcsSphereCastHitEntity,
    VisualApiOperation::EcsSphereCastPoint,
    VisualApiOperation::EcsSphereCastNormal,
    VisualApiOperation::EcsSphereCastToi,
    VisualApiOperation::EcsGetPhysicsRunning,
    VisualApiOperation::EcsGetPhysicsShapeCastHit,
    VisualApiOperation::EcsGetPhysicsVelocity,
    VisualApiOperation::EcsGetPhysicsWorldDefaults,
    VisualApiOperation::EcsGetSceneAsset,
    VisualApiOperation::EcsGetScript,
    VisualApiOperation::EcsGetScriptCount,
    VisualApiOperation::EcsGetScriptField,
    VisualApiOperation::EcsGetScriptPath,
    VisualApiOperation::EcsGetScriptLanguage,
    VisualApiOperation::EcsGetSelfScriptField,
    VisualApiOperation::EcsGetSpline,
    VisualApiOperation::EcsGetTransform,
    VisualApiOperation::EcsGetTransformForward,
    VisualApiOperation::EcsGetTransformRight,
    VisualApiOperation::EcsGetTransformUp,
    VisualApiOperation::EcsGetTriggerEventCount,
    VisualApiOperation::EcsGetTriggerEventOther,
    VisualApiOperation::EcsGetTriggerEventNormal,
    VisualApiOperation::EcsGetTriggerEventPoint,
    VisualApiOperation::EcsGetViewportMode,
    VisualApiOperation::EcsGetViewportPreviewCamera,
    VisualApiOperation::EcsHasComponent,
    VisualApiOperation::EcsListAudioBuses,
    VisualApiOperation::EcsListDynamicComponents,
    VisualApiOperation::EcsListEntities,
    VisualApiOperation::EcsListScriptFields,
    VisualApiOperation::EcsListSelfScriptFields,
    VisualApiOperation::EcsOpenScene,
    VisualApiOperation::EcsPlayAnimClip,
    VisualApiOperation::EcsRemoveAudioBus,
    VisualApiOperation::EcsRemoveComponent,
    VisualApiOperation::EcsRemoveDynamicComponent,
    VisualApiOperation::EcsRemoveDynamicField,
    VisualApiOperation::EcsRemoveSplinePoint,
    VisualApiOperation::EcsSampleSpline,
    VisualApiOperation::EcsSetActiveCamera,
    VisualApiOperation::EcsSetAnimatorBlendChild,
    VisualApiOperation::EcsSetAnimatorBlendNode,
    VisualApiOperation::EcsSetAnimatorEnabled,
    VisualApiOperation::EcsSetAnimatorLayerWeight,
    VisualApiOperation::EcsSetAnimatorParamBool,
    VisualApiOperation::EcsSetAnimatorParamFloat,
    VisualApiOperation::EcsSetAnimatorTimeScale,
    VisualApiOperation::EcsSetAnimatorTransition,
    VisualApiOperation::EcsSetAudioBusName,
    VisualApiOperation::EcsSetAudioBusVolume,
    VisualApiOperation::EcsSetAudioEmitter,
    VisualApiOperation::EcsSetAudioEmitterPath,
    VisualApiOperation::EcsSetAudioEnabled,
    VisualApiOperation::EcsSetAudioHeadWidth,
    VisualApiOperation::EcsSetAudioListener,
    VisualApiOperation::EcsSetAudioSceneVolume,
    VisualApiOperation::EcsSetAudioSpeedOfSound,
    VisualApiOperation::EcsSetAudioStreamingConfig,
    VisualApiOperation::EcsSetCamera,
    VisualApiOperation::EcsSetDynamicComponent,
    VisualApiOperation::EcsSetDynamicField,
    VisualApiOperation::EcsSetEntityFollower,
    VisualApiOperation::EcsSetEntityName,
    VisualApiOperation::EcsSetLight,
    VisualApiOperation::EcsSetLookAt,
    VisualApiOperation::EcsSetMeshRenderer,
    VisualApiOperation::EcsSetMeshRendererSourcePath,
    VisualApiOperation::EcsSetMeshRendererMaterialPath,
    VisualApiOperation::EcsSetSpriteRenderer,
    VisualApiOperation::EcsSetSpriteRendererSheetAnimation,
    VisualApiOperation::EcsSetSpriteRendererSequence,
    VisualApiOperation::EcsSetSpriteRendererTexturePath,
    VisualApiOperation::EcsSetText2d,
    VisualApiOperation::EcsSetPersistentForce,
    VisualApiOperation::EcsSetPersistentTorque,
    VisualApiOperation::EcsSetPhysics,
    VisualApiOperation::EcsSetPhysicsGravity,
    VisualApiOperation::EcsSetPhysicsRunning,
    VisualApiOperation::EcsSetPhysicsVelocity,
    VisualApiOperation::EcsSetPhysicsWorldDefaults,
    VisualApiOperation::EcsSetSceneAsset,
    VisualApiOperation::EcsSetScript,
    VisualApiOperation::EcsSetScriptField,
    VisualApiOperation::EcsSetSelfScriptField,
    VisualApiOperation::EcsSetCharacterControllerDesiredTranslation,
    VisualApiOperation::EcsSetSpline,
    VisualApiOperation::EcsSetSplinePoint,
    VisualApiOperation::EcsSetTransform,
    VisualApiOperation::EcsSetViewportMode,
    VisualApiOperation::EcsSetViewportPreviewCamera,
    VisualApiOperation::EcsSpawnEntity,
    VisualApiOperation::EcsSplineLength,
    VisualApiOperation::EcsSwitchScene,
    VisualApiOperation::EcsSelfScriptIndex,
    VisualApiOperation::EcsTriggerAnimator,
    VisualApiOperation::EcsListGraphTemplates,
    VisualApiOperation::EcsGetGraphTemplate,
    VisualApiOperation::EcsSetGraphTemplate,
    VisualApiOperation::EcsGetRuntimeTuning,
    VisualApiOperation::EcsSetRuntimeTuning,
    VisualApiOperation::EcsGetRuntimeConfig,
    VisualApiOperation::EcsSetRuntimeConfig,
    VisualApiOperation::EcsGetRenderConfig,
    VisualApiOperation::EcsSetRenderConfig,
    VisualApiOperation::EcsGetShaderConstants,
    VisualApiOperation::EcsSetShaderConstants,
    VisualApiOperation::EcsGetStreamingTuning,
    VisualApiOperation::EcsSetStreamingTuning,
    VisualApiOperation::EcsGetRenderPasses,
    VisualApiOperation::EcsSetRenderPasses,
    VisualApiOperation::EcsGetGpuBudget,
    VisualApiOperation::EcsSetGpuBudget,
    VisualApiOperation::EcsGetAssetBudgets,
    VisualApiOperation::EcsSetAssetBudgets,
    VisualApiOperation::EcsGetWindowSettings,
    VisualApiOperation::EcsSetWindowSettings,
    VisualApiOperation::InputActionContext,
    VisualApiOperation::InputActionDown,
    VisualApiOperation::InputActionPressed,
    VisualApiOperation::InputActionReleased,
    VisualApiOperation::InputActionValue,
    VisualApiOperation::InputBindAction,
    VisualApiOperation::InputCursor,
    VisualApiOperation::InputCursorDelta,
    VisualApiOperation::InputCursorGrabMode,
    VisualApiOperation::InputGamepadAxis,
    VisualApiOperation::InputGamepadButton,
    VisualApiOperation::InputGamepadButtonDown,
    VisualApiOperation::InputGamepadButtonPressed,
    VisualApiOperation::InputGamepadButtonReleased,
    VisualApiOperation::InputGamepadCount,
    VisualApiOperation::InputGamepadIds,
    VisualApiOperation::InputGamepadTrigger,
    VisualApiOperation::InputKey,
    VisualApiOperation::InputKeyDown,
    VisualApiOperation::InputKeyPressed,
    VisualApiOperation::InputKeyReleased,
    VisualApiOperation::InputModifiers,
    VisualApiOperation::InputMouseButton,
    VisualApiOperation::InputMouseDown,
    VisualApiOperation::InputMousePressed,
    VisualApiOperation::InputMouseReleased,
    VisualApiOperation::InputScaleFactor,
    VisualApiOperation::InputSetActionContext,
    VisualApiOperation::InputSetCursorVisible,
    VisualApiOperation::InputSetCursorGrab,
    VisualApiOperation::InputUnbindAction,
    VisualApiOperation::InputResetCursorControl,
    VisualApiOperation::InputWantsKeyboard,
    VisualApiOperation::InputWantsPointer,
    VisualApiOperation::InputWheel,
    VisualApiOperation::InputWindowSize,
];

const API_INPUTS_0: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Component",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_1: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Force",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Wake Up",
        value_type: VisualValueType::Bool,
    },
];
const API_INPUTS_2: [VisualApiInputSpec; 4] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Force",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Point",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Wake Up",
        value_type: VisualValueType::Bool,
    },
];
const API_INPUTS_3: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Torque",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Wake Up",
        value_type: VisualValueType::Bool,
    },
];
const API_INPUTS_4: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Point",
        value_type: VisualValueType::Vec3,
    },
];
const API_INPUTS_5: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Impulse",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Wake Up",
        value_type: VisualValueType::Bool,
    },
];
const API_INPUTS_6: [VisualApiInputSpec; 4] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Impulse",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Point",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Wake Up",
        value_type: VisualValueType::Bool,
    },
];
const API_INPUTS_7: [VisualApiInputSpec; 0] = [];
const API_INPUTS_8: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Entity Id",
    value_type: VisualValueType::Entity,
}];
const API_INPUTS_9: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Name",
    value_type: VisualValueType::String,
}];
const API_INPUTS_10: [VisualApiInputSpec; 4] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Spline Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Speed",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Looped",
        value_type: VisualValueType::Bool,
    },
];
const API_INPUTS_11: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Layer Index",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_12: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Bus",
    value_type: VisualValueType::String,
}];
const API_INPUTS_13: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Scene Id",
    value_type: VisualValueType::Entity,
}];
const API_INPUTS_14: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Name",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_15: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Comp Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Field Name",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_16: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Path",
    value_type: VisualValueType::String,
}];
const API_INPUTS_17: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Layer Index",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_18: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Index (1-based, optional)",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_19: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "T",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_20: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Enabled",
        value_type: VisualValueType::Bool,
    },
];
const API_INPUTS_21: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Value",
        value_type: VisualValueType::Bool,
    },
];
const API_INPUTS_22: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Value",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_23: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Time Scale",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_24: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Bus",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Name",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_25: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Bus",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Volume",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_26: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::Spline,
    },
];
const API_INPUTS_27: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Enabled",
    value_type: VisualValueType::Bool,
}];
const API_INPUTS_28: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Width",
    value_type: VisualValueType::Number,
}];
const API_INPUTS_29: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Scene Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Volume",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_30: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Speed",
    value_type: VisualValueType::Number,
}];
const API_INPUTS_31: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Buffer Frames",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Chunk Frames",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_32: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Fields",
        value_type: VisualValueType::DynamicComponentFields,
    },
];
const API_INPUTS_33: [VisualApiInputSpec; 4] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Comp Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Field Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Value",
        value_type: VisualValueType::DynamicFieldValue,
    },
];
const API_INPUTS_34: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Gravity",
    value_type: VisualValueType::Vec3,
}];
const API_INPUTS_35: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Running",
    value_type: VisualValueType::Bool,
}];
const API_INPUTS_36: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Path",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_38: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Point",
        value_type: VisualValueType::Vec3,
    },
];
const API_INPUTS_39: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Mode",
    value_type: VisualValueType::String,
}];
const API_INPUTS_40: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Mode",
    value_type: VisualValueType::String,
}];
const API_INPUTS_41: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Samples",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_42: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Axis",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Id",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_43: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Button",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Id",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_44: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Side",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Id",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_45: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Key",
    value_type: VisualValueType::String,
}];
const API_INPUTS_46: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Button",
    value_type: VisualValueType::String,
}];
const API_INPUTS_47: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::Transform,
    },
];
const API_INPUTS_48: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::AudioEmitter,
    },
];
const API_INPUTS_49: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::AudioListener,
    },
];
const API_INPUTS_50: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::Camera,
    },
];
const API_INPUTS_51: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::Light,
    },
];
const API_INPUTS_52: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::MeshRenderer,
    },
];
const API_INPUTS_53: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::Physics,
    },
];
const API_INPUTS_54: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::PhysicsVelocity,
    },
];
const API_INPUTS_55: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::PhysicsWorldDefaults,
    },
];
const API_INPUTS_56: [VisualApiInputSpec; 6] = [
    VisualApiInputSpec {
        label: "Origin",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Direction",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Max Toi",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Solid",
        value_type: VisualValueType::Bool,
    },
    VisualApiInputSpec {
        label: "Filter",
        value_type: VisualValueType::PhysicsQueryFilter,
    },
    VisualApiInputSpec {
        label: "Exclude Entity",
        value_type: VisualValueType::Entity,
    },
];
const API_INPUTS_57: [VisualApiInputSpec; 4] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Path",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Language",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Index (1-based, optional)",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_58: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Target Entity",
        value_type: VisualValueType::Entity,
    },
];
const API_INPUTS_59: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Layer Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Weight",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_60: [VisualApiInputSpec; 9] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Layer Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Transition Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "From State",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "To State",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Duration",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Can Interrupt",
        value_type: VisualValueType::Bool,
    },
    VisualApiInputSpec {
        label: "Use Exit Time",
        value_type: VisualValueType::Bool,
    },
    VisualApiInputSpec {
        label: "Exit Time",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_61: [VisualApiInputSpec; 5] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Layer Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Node Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Normalize",
        value_type: VisualValueType::Bool,
    },
    VisualApiInputSpec {
        label: "Mode",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_62: [VisualApiInputSpec; 8] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Layer Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Node Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Child Index",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Weight",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Weight Param",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Weight Scale",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Weight Bias",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_63: [VisualApiInputSpec; 4] = [
    VisualApiInputSpec {
        label: "Action",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Binding (KeyW, MouseLeft, GamepadSouth...)",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Context",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Deadzone",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_64: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Action",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Binding (optional)",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Context",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_65: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Action",
    value_type: VisualValueType::String,
}];
const API_INPUTS_66: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Context",
    value_type: VisualValueType::String,
}];
const API_INPUTS_67: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Path",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Language",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_68: [VisualApiInputSpec; 6] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Target Entity",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Target Offset",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Offset In Target Space",
        value_type: VisualValueType::Bool,
    },
    VisualApiInputSpec {
        label: "Up",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Rotation Smooth Time",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_69: [VisualApiInputSpec; 7] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Target Entity",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Position Offset",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Offset In Target Space",
        value_type: VisualValueType::Bool,
    },
    VisualApiInputSpec {
        label: "Follow Rotation",
        value_type: VisualValueType::Bool,
    },
    VisualApiInputSpec {
        label: "Position Smooth Time",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Rotation Smooth Time",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_70: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Desired Translation",
        value_type: VisualValueType::Vec3,
    },
];
const API_INPUTS_71: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Phase (enter/stay/exit/all)",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_72: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Phase (enter/stay/exit/all)",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Event Index",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_73: [VisualApiInputSpec; 6] = [
    VisualApiInputSpec {
        label: "Origin",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Radius",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Direction",
        value_type: VisualValueType::Vec3,
    },
    VisualApiInputSpec {
        label: "Max TOI",
        value_type: VisualValueType::Number,
    },
    VisualApiInputSpec {
        label: "Filter",
        value_type: VisualValueType::PhysicsQueryFilter,
    },
    VisualApiInputSpec {
        label: "Exclude Entity",
        value_type: VisualValueType::Entity,
    },
];
const API_INPUTS_74: [VisualApiInputSpec; 3] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Field Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Index",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_75: [VisualApiInputSpec; 4] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Field Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Value",
        value_type: VisualValueType::Any,
    },
    VisualApiInputSpec {
        label: "Index",
        value_type: VisualValueType::Number,
    },
];
const API_INPUTS_76: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Field Name",
    value_type: VisualValueType::String,
}];
const API_INPUTS_77: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Field Name",
        value_type: VisualValueType::String,
    },
    VisualApiInputSpec {
        label: "Value",
        value_type: VisualValueType::Any,
    },
];
const API_INPUTS_91: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Tuning",
    value_type: VisualValueType::RuntimeTuning,
}];
const API_INPUTS_92: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Config",
    value_type: VisualValueType::RuntimeConfig,
}];
const API_INPUTS_93: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Patch",
    value_type: VisualValueType::RenderConfig,
}];
const API_INPUTS_94: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Patch",
    value_type: VisualValueType::ShaderConstants,
}];
const API_INPUTS_95: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Patch",
    value_type: VisualValueType::StreamingTuning,
}];
const API_INPUTS_96: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Passes",
    value_type: VisualValueType::RenderPasses,
}];
const API_INPUTS_97: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Budget",
    value_type: VisualValueType::GpuBudget,
}];
const API_INPUTS_98: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Budgets",
    value_type: VisualValueType::AssetBudgets,
}];
const API_INPUTS_99: [VisualApiInputSpec; 1] = [VisualApiInputSpec {
    label: "Settings",
    value_type: VisualValueType::WindowSettings,
}];
const API_INPUTS_100: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::SpriteRenderer,
    },
];
const API_INPUTS_101: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Data",
        value_type: VisualValueType::Text2d,
    },
];
const API_INPUTS_102: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Sheet",
        value_type: VisualValueType::SpriteRenderer,
    },
];
const API_INPUTS_103: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Entity Id",
        value_type: VisualValueType::Entity,
    },
    VisualApiInputSpec {
        label: "Sequence",
        value_type: VisualValueType::SpriteRenderer,
    },
];

const VISUAL_API_OPERATION_SPECS: [VisualApiOperationSpec; 242] = [
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsAddComponent,
        table: VisualScriptApiTable::Ecs,
        function: "add_component",
        title: "Add Component",
        category: "Add",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_0,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsAddForce,
        table: VisualScriptApiTable::Ecs,
        function: "add_force",
        title: "Add Force",
        category: "Add",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_1,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsAddForceAtPoint,
        table: VisualScriptApiTable::Ecs,
        function: "add_force_at_point",
        title: "Add Force At Point",
        category: "Add",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_2,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsAddPersistentForce,
        table: VisualScriptApiTable::Ecs,
        function: "add_persistent_force",
        title: "Add Persistent Force",
        category: "Add",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_1,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsAddPersistentForceAtPoint,
        table: VisualScriptApiTable::Ecs,
        function: "add_persistent_force_at_point",
        title: "Add Persistent Force At Point",
        category: "Add",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_2,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsAddPersistentTorque,
        table: VisualScriptApiTable::Ecs,
        function: "add_persistent_torque",
        title: "Add Persistent Torque",
        category: "Add",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_3,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsAddSplinePoint,
        table: VisualScriptApiTable::Ecs,
        function: "add_spline_point",
        title: "Add Spline Point",
        category: "Add",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_4,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsAddTorque,
        table: VisualScriptApiTable::Ecs,
        function: "add_torque",
        title: "Add Torque",
        category: "Add",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_3,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsApplyAngularImpulse,
        table: VisualScriptApiTable::Ecs,
        function: "apply_angular_impulse",
        title: "Apply Angular Impulse",
        category: "Physics",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_5,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsApplyImpulse,
        table: VisualScriptApiTable::Ecs,
        function: "apply_impulse",
        title: "Apply Impulse",
        category: "Physics",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_5,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsApplyImpulseAtPoint,
        table: VisualScriptApiTable::Ecs,
        function: "apply_impulse_at_point",
        title: "Apply Impulse At Point",
        category: "Physics",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_6,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsApplyTorqueImpulse,
        table: VisualScriptApiTable::Ecs,
        function: "apply_torque_impulse",
        title: "Apply Torque Impulse",
        category: "Physics",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_5,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsClearAudioEmitters,
        table: VisualScriptApiTable::Ecs,
        function: "clear_audio_emitters",
        title: "Clear Audio Emitters",
        category: "Clear",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsClearPersistentForces,
        table: VisualScriptApiTable::Ecs,
        function: "clear_persistent_forces",
        title: "Clear Persistent Forces",
        category: "Clear",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsClearPhysics,
        table: VisualScriptApiTable::Ecs,
        function: "clear_physics",
        title: "Clear Physics",
        category: "Clear",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsCreateAudioBus,
        table: VisualScriptApiTable::Ecs,
        function: "create_audio_bus",
        title: "Create Audio Bus",
        category: "Create",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_9,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsDeleteEntity,
        table: VisualScriptApiTable::Ecs,
        function: "delete_entity",
        title: "Delete Entity",
        category: "Gameplay",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsEntityExists,
        table: VisualScriptApiTable::Ecs,
        function: "entity_exists",
        title: "Entity Exists",
        category: "Gameplay",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsEmitEvent,
        table: VisualScriptApiTable::Ecs,
        function: "emit_event",
        title: "Emit Event",
        category: "Events",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_58,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsFindEntityByName,
        table: VisualScriptApiTable::Ecs,
        function: "find_entity_by_name",
        title: "Find Entity By Name",
        category: "Gameplay",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_9,
        output_type: Some(VisualValueType::Entity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsFindScriptIndex,
        table: VisualScriptApiTable::Ecs,
        function: "find_script_index",
        title: "Find Script Index",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_67,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsFollowSpline,
        table: VisualScriptApiTable::Ecs,
        function: "follow_spline",
        title: "Follow Spline",
        category: "Gameplay",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_10,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorClips,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_clips",
        title: "Get Animator Clips",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Array),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorLayerWeights,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_layer_weights",
        title: "Get Animator Layer Weights",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Array),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorLayerWeight,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_layer_weight",
        title: "Get Animator Layer Weight",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorState,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_state",
        title: "Get Animator State",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::AnimatorState),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorStateTime,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_state_time",
        title: "Get Animator State Time",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorCurrentState,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_current_state",
        title: "Get Animator Current State",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorCurrentStateName,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_current_state_name",
        title: "Get Animator Current State Name",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorTransitionActive,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_transition_active",
        title: "Get Animator Transition Active",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorTransitionFrom,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_transition_from",
        title: "Get Animator Transition From",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorTransitionTo,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_transition_to",
        title: "Get Animator Transition To",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorTransitionProgress,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_transition_progress",
        title: "Get Animator Transition Progress",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorTransitionElapsed,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_transition_elapsed",
        title: "Get Animator Transition Elapsed",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAnimatorTransitionDuration,
        table: VisualScriptApiTable::Ecs,
        function: "get_animator_transition_duration",
        title: "Get Animator Transition Duration",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_11,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioBusName,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_bus_name",
        title: "Get Audio Bus Name",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_12,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioBusVolume,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_bus_volume",
        title: "Get Audio Bus Volume",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_12,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioEmitter,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_emitter",
        title: "Get Audio Emitter",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::AudioEmitter),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioEmitterPath,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_emitter_path",
        title: "Get Audio Emitter Path",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioEnabled,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_enabled",
        title: "Get Audio Enabled",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioHeadWidth,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_head_width",
        title: "Get Audio Head Width",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioListener,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_listener",
        title: "Get Audio Listener",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::AudioListener),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioSceneVolume,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_scene_volume",
        title: "Get Audio Scene Volume",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_13,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioSpeedOfSound,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_speed_of_sound",
        title: "Get Audio Speed Of Sound",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAudioStreamingConfig,
        table: VisualScriptApiTable::Ecs,
        function: "get_audio_streaming_config",
        title: "Get Audio Streaming Config",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::AudioStreamingConfig),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCamera,
        table: VisualScriptApiTable::Ecs,
        function: "get_camera",
        title: "Get Camera",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Camera),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerOutput,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_output",
        title: "Get Character Controller Output",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::CharacterControllerOutput),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerDesiredTranslation,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_desired_translation",
        title: "Get Character Desired Translation",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerEffectiveTranslation,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_effective_translation",
        title: "Get Character Effective Translation",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerRemainingTranslation,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_remaining_translation",
        title: "Get Character Remaining Translation",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerGrounded,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_grounded",
        title: "Get Character Grounded",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerSlidingDownSlope,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_sliding_down_slope",
        title: "Get Character Sliding Down Slope",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerCollisionCount,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_collision_count",
        title: "Get Character Collision Count",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerGroundNormal,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_ground_normal",
        title: "Get Character Ground Normal",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerSlopeAngle,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_slope_angle",
        title: "Get Character Slope Angle",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerHitNormal,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_hit_normal",
        title: "Get Character Hit Normal",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerHitPoint,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_hit_point",
        title: "Get Character Hit Point",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerHitEntity,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_hit_entity",
        title: "Get Character Hit Entity",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Entity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerSteppedUp,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_stepped_up",
        title: "Get Character Stepped Up",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerStepHeight,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_step_height",
        title: "Get Character Step Height",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerPlatformVelocity,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_platform_velocity",
        title: "Get Character Platform Velocity",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCollisionEventCount,
        table: VisualScriptApiTable::Ecs,
        function: "get_collision_event_count",
        title: "Get Collision Event Count",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_71,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCollisionEventOther,
        table: VisualScriptApiTable::Ecs,
        function: "get_collision_event_other",
        title: "Get Collision Event Other",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_72,
        output_type: Some(VisualValueType::Entity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCollisionEventNormal,
        table: VisualScriptApiTable::Ecs,
        function: "get_collision_event_normal",
        title: "Get Collision Event Normal",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_72,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCollisionEventPoint,
        table: VisualScriptApiTable::Ecs,
        function: "get_collision_event_point",
        title: "Get Collision Event Point",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_72,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetDynamicComponent,
        table: VisualScriptApiTable::Ecs,
        function: "get_dynamic_component",
        title: "Get Dynamic Component",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_14,
        output_type: Some(VisualValueType::DynamicComponentFields),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetDynamicField,
        table: VisualScriptApiTable::Ecs,
        function: "get_dynamic_field",
        title: "Get Dynamic Field",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_15,
        output_type: Some(VisualValueType::DynamicFieldValue),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetEntityFollower,
        table: VisualScriptApiTable::Ecs,
        function: "get_entity_follower",
        title: "Get Entity Follower",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::EntityFollower),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetEntityName,
        table: VisualScriptApiTable::Ecs,
        function: "get_entity_name",
        title: "Get Entity Name",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetLight,
        table: VisualScriptApiTable::Ecs,
        function: "get_light",
        title: "Get Light",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Light),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetLookAt,
        table: VisualScriptApiTable::Ecs,
        function: "get_look_at",
        title: "Get Look At",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::LookAt),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetMeshRenderer,
        table: VisualScriptApiTable::Ecs,
        function: "get_mesh_renderer",
        title: "Get Mesh Renderer",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::MeshRenderer),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetMeshRendererSourcePath,
        table: VisualScriptApiTable::Ecs,
        function: "get_mesh_renderer_source_path",
        title: "Get Mesh Renderer Source Path",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetMeshRendererMaterialPath,
        table: VisualScriptApiTable::Ecs,
        function: "get_mesh_renderer_material_path",
        title: "Get Mesh Renderer Material Path",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysics,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics",
        title: "Get Physics",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Physics),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsGravity,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_gravity",
        title: "Get Physics Gravity",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsPointProjectionHit,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_point_projection_hit",
        title: "Get Physics Point Projection Hit",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::PhysicsPointProjectionHit),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsRayCastHit,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_ray_cast_hit",
        title: "Get Physics Ray Cast Hit",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::PhysicsRayCastHit),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRayCast,
        table: VisualScriptApiTable::Ecs,
        function: "ray_cast",
        title: "Ray Cast",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_56,
        output_type: Some(VisualValueType::PhysicsRayCastHit),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRayCastHasHit,
        table: VisualScriptApiTable::Ecs,
        function: "ray_cast_has_hit",
        title: "Ray Cast Has Hit",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_56,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRayCastHitEntity,
        table: VisualScriptApiTable::Ecs,
        function: "ray_cast_hit_entity",
        title: "Ray Cast Hit Entity",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_56,
        output_type: Some(VisualValueType::Entity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRayCastPoint,
        table: VisualScriptApiTable::Ecs,
        function: "ray_cast_point",
        title: "Ray Cast Point",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_56,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRayCastNormal,
        table: VisualScriptApiTable::Ecs,
        function: "ray_cast_normal",
        title: "Ray Cast Normal",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_56,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRayCastToi,
        table: VisualScriptApiTable::Ecs,
        function: "ray_cast_toi",
        title: "Ray Cast TOI",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_56,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSphereCast,
        table: VisualScriptApiTable::Ecs,
        function: "sphere_cast",
        title: "Sphere Cast",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_73,
        output_type: Some(VisualValueType::PhysicsShapeCastHit),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSphereCastHasHit,
        table: VisualScriptApiTable::Ecs,
        function: "sphere_cast_has_hit",
        title: "Sphere Cast Has Hit",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_73,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSphereCastHitEntity,
        table: VisualScriptApiTable::Ecs,
        function: "sphere_cast_hit_entity",
        title: "Sphere Cast Hit Entity",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_73,
        output_type: Some(VisualValueType::Entity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSphereCastPoint,
        table: VisualScriptApiTable::Ecs,
        function: "sphere_cast_point",
        title: "Sphere Cast Point",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_73,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSphereCastNormal,
        table: VisualScriptApiTable::Ecs,
        function: "sphere_cast_normal",
        title: "Sphere Cast Normal",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_73,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSphereCastToi,
        table: VisualScriptApiTable::Ecs,
        function: "sphere_cast_toi",
        title: "Sphere Cast TOI",
        category: "Query",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_73,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsRunning,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_running",
        title: "Get Physics Running",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsShapeCastHit,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_shape_cast_hit",
        title: "Get Physics Shape Cast Hit",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::PhysicsShapeCastHit),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsVelocity,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_velocity",
        title: "Get Physics Velocity",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::PhysicsVelocity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsWorldDefaults,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_world_defaults",
        title: "Get Physics World Defaults",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::PhysicsWorldDefaults),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetSceneAsset,
        table: VisualScriptApiTable::Ecs,
        function: "get_scene_asset",
        title: "Get Scene Asset",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetScript,
        table: VisualScriptApiTable::Ecs,
        function: "get_script",
        title: "Get Script",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_18,
        output_type: Some(VisualValueType::Script),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetScriptCount,
        table: VisualScriptApiTable::Ecs,
        function: "get_script_count",
        title: "Get Script Count",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsListScriptFields,
        table: VisualScriptApiTable::Ecs,
        function: "list_script_fields",
        title: "List Script Fields",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_18,
        output_type: Some(VisualValueType::Any),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetScriptField,
        table: VisualScriptApiTable::Ecs,
        function: "get_script_field",
        title: "Get Script Field",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_74,
        output_type: Some(VisualValueType::Any),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetScriptPath,
        table: VisualScriptApiTable::Ecs,
        function: "get_script_path",
        title: "Get Script Path",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_18,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetScriptLanguage,
        table: VisualScriptApiTable::Ecs,
        function: "get_script_language",
        title: "Get Script Language",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_18,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsListSelfScriptFields,
        table: VisualScriptApiTable::Ecs,
        function: "list_self_script_fields",
        title: "List Self Script Fields",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Any),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetSelfScriptField,
        table: VisualScriptApiTable::Ecs,
        function: "get_self_script_field",
        title: "Get Self Script Field",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_76,
        output_type: Some(VisualValueType::Any),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetSpline,
        table: VisualScriptApiTable::Ecs,
        function: "get_spline",
        title: "Get Spline",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Spline),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetTransform,
        table: VisualScriptApiTable::Ecs,
        function: "get_transform",
        title: "Get Transform",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Transform),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetTransformForward,
        table: VisualScriptApiTable::Ecs,
        function: "get_transform_forward",
        title: "Get Transform Forward",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetTransformRight,
        table: VisualScriptApiTable::Ecs,
        function: "get_transform_right",
        title: "Get Transform Right",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetTransformUp,
        table: VisualScriptApiTable::Ecs,
        function: "get_transform_up",
        title: "Get Transform Up",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetTriggerEventCount,
        table: VisualScriptApiTable::Ecs,
        function: "get_trigger_event_count",
        title: "Get Trigger Event Count",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_71,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetTriggerEventOther,
        table: VisualScriptApiTable::Ecs,
        function: "get_trigger_event_other",
        title: "Get Trigger Event Other",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_72,
        output_type: Some(VisualValueType::Entity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetTriggerEventNormal,
        table: VisualScriptApiTable::Ecs,
        function: "get_trigger_event_normal",
        title: "Get Trigger Event Normal",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_72,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetTriggerEventPoint,
        table: VisualScriptApiTable::Ecs,
        function: "get_trigger_event_point",
        title: "Get Trigger Event Point",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_72,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetViewportMode,
        table: VisualScriptApiTable::Ecs,
        function: "get_viewport_mode",
        title: "Get Viewport Mode",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetViewportPreviewCamera,
        table: VisualScriptApiTable::Ecs,
        function: "get_viewport_preview_camera",
        title: "Get Viewport Preview Camera",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Entity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsHasComponent,
        table: VisualScriptApiTable::Ecs,
        function: "has_component",
        title: "Has Component",
        category: "Gameplay",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_0,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsListAudioBuses,
        table: VisualScriptApiTable::Ecs,
        function: "list_audio_buses",
        title: "List Audio Buses",
        category: "List",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Array),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsListDynamicComponents,
        table: VisualScriptApiTable::Ecs,
        function: "list_dynamic_components",
        title: "List Dynamic Components",
        category: "List",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Array),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsListEntities,
        table: VisualScriptApiTable::Ecs,
        function: "list_entities",
        title: "List Entities",
        category: "List",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Array),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsOpenScene,
        table: VisualScriptApiTable::Ecs,
        function: "open_scene",
        title: "Open Scene",
        category: "Scene",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_16,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsPlayAnimClip,
        table: VisualScriptApiTable::Ecs,
        function: "play_anim_clip",
        title: "Play Anim Clip",
        category: "Animation",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_17,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRemoveAudioBus,
        table: VisualScriptApiTable::Ecs,
        function: "remove_audio_bus",
        title: "Remove Audio Bus",
        category: "Remove",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_12,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRemoveComponent,
        table: VisualScriptApiTable::Ecs,
        function: "remove_component",
        title: "Remove Component",
        category: "Remove",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_0,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRemoveDynamicComponent,
        table: VisualScriptApiTable::Ecs,
        function: "remove_dynamic_component",
        title: "Remove Dynamic Component",
        category: "Remove",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_14,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRemoveDynamicField,
        table: VisualScriptApiTable::Ecs,
        function: "remove_dynamic_field",
        title: "Remove Dynamic Field",
        category: "Remove",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_15,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsRemoveSplinePoint,
        table: VisualScriptApiTable::Ecs,
        function: "remove_spline_point",
        title: "Remove Spline Point",
        category: "Remove",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_18,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSampleSpline,
        table: VisualScriptApiTable::Ecs,
        function: "sample_spline",
        title: "Sample Spline",
        category: "Gameplay",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_19,
        output_type: Some(VisualValueType::Vec3),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetActiveCamera,
        table: VisualScriptApiTable::Ecs,
        function: "set_active_camera",
        title: "Set Active Camera",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAnimatorBlendChild,
        table: VisualScriptApiTable::Ecs,
        function: "set_animator_blend_child",
        title: "Set Animator Blend Child",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_62,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAnimatorBlendNode,
        table: VisualScriptApiTable::Ecs,
        function: "set_animator_blend_node",
        title: "Set Animator Blend Node",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_61,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAnimatorEnabled,
        table: VisualScriptApiTable::Ecs,
        function: "set_animator_enabled",
        title: "Set Animator Enabled",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_20,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAnimatorLayerWeight,
        table: VisualScriptApiTable::Ecs,
        function: "set_animator_layer_weight",
        title: "Set Animator Layer Weight",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_59,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAnimatorParamBool,
        table: VisualScriptApiTable::Ecs,
        function: "set_animator_param_bool",
        title: "Set Animator Param Bool",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_21,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAnimatorParamFloat,
        table: VisualScriptApiTable::Ecs,
        function: "set_animator_param_float",
        title: "Set Animator Param Float",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_22,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAnimatorTimeScale,
        table: VisualScriptApiTable::Ecs,
        function: "set_animator_time_scale",
        title: "Set Animator Time Scale",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_23,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAnimatorTransition,
        table: VisualScriptApiTable::Ecs,
        function: "set_animator_transition",
        title: "Set Animator Transition",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_60,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioBusName,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_bus_name",
        title: "Set Audio Bus Name",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_24,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioBusVolume,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_bus_volume",
        title: "Set Audio Bus Volume",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_25,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioEmitter,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_emitter",
        title: "Set Audio Emitter",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_48,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioEmitterPath,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_emitter_path",
        title: "Set Audio Emitter Path",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_36,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioEnabled,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_enabled",
        title: "Set Audio Enabled",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_27,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioHeadWidth,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_head_width",
        title: "Set Audio Head Width",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_28,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioListener,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_listener",
        title: "Set Audio Listener",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_49,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioSceneVolume,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_scene_volume",
        title: "Set Audio Scene Volume",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_29,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioSpeedOfSound,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_speed_of_sound",
        title: "Set Audio Speed Of Sound",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_30,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAudioStreamingConfig,
        table: VisualScriptApiTable::Ecs,
        function: "set_audio_streaming_config",
        title: "Set Audio Streaming Config",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_31,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetCamera,
        table: VisualScriptApiTable::Ecs,
        function: "set_camera",
        title: "Set Camera",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_50,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetDynamicComponent,
        table: VisualScriptApiTable::Ecs,
        function: "set_dynamic_component",
        title: "Set Dynamic Component",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_32,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetDynamicField,
        table: VisualScriptApiTable::Ecs,
        function: "set_dynamic_field",
        title: "Set Dynamic Field",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_33,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetEntityFollower,
        table: VisualScriptApiTable::Ecs,
        function: "set_entity_follower",
        title: "Set Entity Follower",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_69,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetEntityName,
        table: VisualScriptApiTable::Ecs,
        function: "set_entity_name",
        title: "Set Entity Name",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_14,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetLight,
        table: VisualScriptApiTable::Ecs,
        function: "set_light",
        title: "Set Light",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_51,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetLookAt,
        table: VisualScriptApiTable::Ecs,
        function: "set_look_at",
        title: "Set Look At",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_68,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetMeshRenderer,
        table: VisualScriptApiTable::Ecs,
        function: "set_mesh_renderer",
        title: "Set Mesh Renderer",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_52,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetMeshRendererSourcePath,
        table: VisualScriptApiTable::Ecs,
        function: "set_mesh_renderer_source_path",
        title: "Set Mesh Renderer Source Path",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_36,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetMeshRendererMaterialPath,
        table: VisualScriptApiTable::Ecs,
        function: "set_mesh_renderer_material_path",
        title: "Set Mesh Renderer Material Path",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_36,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetPersistentForce,
        table: VisualScriptApiTable::Ecs,
        function: "set_persistent_force",
        title: "Set Persistent Force",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_1,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetPersistentTorque,
        table: VisualScriptApiTable::Ecs,
        function: "set_persistent_torque",
        title: "Set Persistent Torque",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_3,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetPhysics,
        table: VisualScriptApiTable::Ecs,
        function: "set_physics",
        title: "Set Physics",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_53,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetPhysicsGravity,
        table: VisualScriptApiTable::Ecs,
        function: "set_physics_gravity",
        title: "Set Physics Gravity",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_34,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetPhysicsRunning,
        table: VisualScriptApiTable::Ecs,
        function: "set_physics_running",
        title: "Set Physics Running",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_35,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetPhysicsVelocity,
        table: VisualScriptApiTable::Ecs,
        function: "set_physics_velocity",
        title: "Set Physics Velocity",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_54,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetPhysicsWorldDefaults,
        table: VisualScriptApiTable::Ecs,
        function: "set_physics_world_defaults",
        title: "Set Physics World Defaults",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_55,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetSceneAsset,
        table: VisualScriptApiTable::Ecs,
        function: "set_scene_asset",
        title: "Set Scene Asset",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_36,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetScript,
        table: VisualScriptApiTable::Ecs,
        function: "set_script",
        title: "Set Script",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_57,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetScriptField,
        table: VisualScriptApiTable::Ecs,
        function: "set_script_field",
        title: "Set Script Field",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_75,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetSelfScriptField,
        table: VisualScriptApiTable::Ecs,
        function: "set_self_script_field",
        title: "Set Self Script Field",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_77,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetCharacterControllerDesiredTranslation,
        table: VisualScriptApiTable::Ecs,
        function: "set_character_controller_desired_translation",
        title: "Set Character Desired Translation",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_70,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetSpline,
        table: VisualScriptApiTable::Ecs,
        function: "set_spline",
        title: "Set Spline",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_26,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetSplinePoint,
        table: VisualScriptApiTable::Ecs,
        function: "set_spline_point",
        title: "Set Spline Point",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_38,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetTransform,
        table: VisualScriptApiTable::Ecs,
        function: "set_transform",
        title: "Set Transform",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_47,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetViewportMode,
        table: VisualScriptApiTable::Ecs,
        function: "set_viewport_mode",
        title: "Set Viewport Mode",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_39,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetViewportPreviewCamera,
        table: VisualScriptApiTable::Ecs,
        function: "set_viewport_preview_camera",
        title: "Set Viewport Preview Camera",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_40,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSpawnEntity,
        table: VisualScriptApiTable::Ecs,
        function: "spawn_entity",
        title: "Spawn Entity",
        category: "Gameplay",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_9,
        output_type: Some(VisualValueType::Entity),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSplineLength,
        table: VisualScriptApiTable::Ecs,
        function: "spline_length",
        title: "Spline Length",
        category: "Gameplay",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_41,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSwitchScene,
        table: VisualScriptApiTable::Ecs,
        function: "switch_scene",
        title: "Switch Scene",
        category: "Scene",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_16,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSelfScriptIndex,
        table: VisualScriptApiTable::Ecs,
        function: "self_script_index",
        title: "Self Script Index",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsTriggerAnimator,
        table: VisualScriptApiTable::Ecs,
        function: "trigger_animator",
        title: "Trigger Animator",
        category: "Animation",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_14,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputActionContext,
        table: VisualScriptApiTable::Input,
        function: "action_context",
        title: "Action Context",
        category: "Window",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputActionDown,
        table: VisualScriptApiTable::Input,
        function: "action_down",
        title: "Action Down",
        category: "Keyboard",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_65,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputActionPressed,
        table: VisualScriptApiTable::Input,
        function: "action_pressed",
        title: "Action Pressed",
        category: "Keyboard",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_65,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputActionReleased,
        table: VisualScriptApiTable::Input,
        function: "action_released",
        title: "Action Released",
        category: "Keyboard",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_65,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputActionValue,
        table: VisualScriptApiTable::Input,
        function: "action_value",
        title: "Action Value",
        category: "Keyboard",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_65,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputBindAction,
        table: VisualScriptApiTable::Input,
        function: "bind_action",
        title: "Register Action Binding",
        category: "Keyboard",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_63,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputCursor,
        table: VisualScriptApiTable::Input,
        function: "cursor",
        title: "Cursor",
        category: "Mouse",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Vec2),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputCursorDelta,
        table: VisualScriptApiTable::Input,
        function: "cursor_delta",
        title: "Cursor Delta",
        category: "Mouse",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Vec2),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputCursorGrabMode,
        table: VisualScriptApiTable::Input,
        function: "cursor_grab_mode",
        title: "Cursor Grab Mode",
        category: "Window",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputGamepadAxis,
        table: VisualScriptApiTable::Input,
        function: "gamepad_axis",
        title: "Gamepad Axis",
        category: "Gamepad",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_42,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputGamepadButton,
        table: VisualScriptApiTable::Input,
        function: "gamepad_button",
        title: "Gamepad Button",
        category: "Gamepad",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_46,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputGamepadButtonDown,
        table: VisualScriptApiTable::Input,
        function: "gamepad_button_down",
        title: "Gamepad Button Down",
        category: "Gamepad",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_43,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputGamepadButtonPressed,
        table: VisualScriptApiTable::Input,
        function: "gamepad_button_pressed",
        title: "Gamepad Button Pressed",
        category: "Gamepad",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_43,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputGamepadButtonReleased,
        table: VisualScriptApiTable::Input,
        function: "gamepad_button_released",
        title: "Gamepad Button Released",
        category: "Gamepad",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_43,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputGamepadCount,
        table: VisualScriptApiTable::Input,
        function: "gamepad_count",
        title: "Gamepad Count",
        category: "Gamepad",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputGamepadIds,
        table: VisualScriptApiTable::Input,
        function: "gamepad_ids",
        title: "Gamepad Ids",
        category: "Gamepad",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Array),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputGamepadTrigger,
        table: VisualScriptApiTable::Input,
        function: "gamepad_trigger",
        title: "Gamepad Trigger",
        category: "Gamepad",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_44,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputKey,
        table: VisualScriptApiTable::Input,
        function: "key",
        title: "Key",
        category: "Keyboard",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_45,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputKeyDown,
        table: VisualScriptApiTable::Input,
        function: "key_down",
        title: "Key Down",
        category: "Keyboard",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_45,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputKeyPressed,
        table: VisualScriptApiTable::Input,
        function: "key_pressed",
        title: "Key Pressed",
        category: "Keyboard",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_45,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputKeyReleased,
        table: VisualScriptApiTable::Input,
        function: "key_released",
        title: "Key Released",
        category: "Keyboard",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_45,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputModifiers,
        table: VisualScriptApiTable::Input,
        function: "modifiers",
        title: "Modifiers",
        category: "Window",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::InputModifiers),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputMouseButton,
        table: VisualScriptApiTable::Input,
        function: "mouse_button",
        title: "Mouse Button",
        category: "Mouse",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_46,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputMouseDown,
        table: VisualScriptApiTable::Input,
        function: "mouse_down",
        title: "Mouse Down",
        category: "Mouse",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_46,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputMousePressed,
        table: VisualScriptApiTable::Input,
        function: "mouse_pressed",
        title: "Mouse Pressed",
        category: "Mouse",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_46,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputMouseReleased,
        table: VisualScriptApiTable::Input,
        function: "mouse_released",
        title: "Mouse Released",
        category: "Mouse",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_46,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputScaleFactor,
        table: VisualScriptApiTable::Input,
        function: "scale_factor",
        title: "Scale Factor",
        category: "Window",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Number),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputSetActionContext,
        table: VisualScriptApiTable::Input,
        function: "set_action_context",
        title: "Set Action Context",
        category: "Window",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_66,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputSetCursorVisible,
        table: VisualScriptApiTable::Input,
        function: "set_cursor_visible",
        title: "Set Cursor Visible",
        category: "Window",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_27,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputSetCursorGrab,
        table: VisualScriptApiTable::Input,
        function: "set_cursor_grab",
        title: "Set Cursor Grab",
        category: "Window",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_39,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputUnbindAction,
        table: VisualScriptApiTable::Input,
        function: "unbind_action",
        title: "Unregister Action Binding",
        category: "Keyboard",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_64,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputResetCursorControl,
        table: VisualScriptApiTable::Input,
        function: "reset_cursor_control",
        title: "Reset Cursor Control",
        category: "Window",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputWantsKeyboard,
        table: VisualScriptApiTable::Input,
        function: "wants_keyboard",
        title: "Wants Keyboard",
        category: "Window",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputWantsPointer,
        table: VisualScriptApiTable::Input,
        function: "wants_pointer",
        title: "Wants Pointer",
        category: "Window",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputWheel,
        table: VisualScriptApiTable::Input,
        function: "wheel",
        title: "Wheel",
        category: "Mouse",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Vec2),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::InputWindowSize,
        table: VisualScriptApiTable::Input,
        function: "window_size",
        title: "Window Size",
        category: "Window",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Vec2),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsListGraphTemplates,
        table: VisualScriptApiTable::Ecs,
        function: "list_graph_templates",
        title: "List Graph Templates",
        category: "Render",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Array),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetGraphTemplate,
        table: VisualScriptApiTable::Ecs,
        function: "get_graph_template",
        title: "Get Graph Template",
        category: "Render",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetGraphTemplate,
        table: VisualScriptApiTable::Ecs,
        function: "set_graph_template",
        title: "Set Graph Template",
        category: "Render",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_9,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetRuntimeTuning,
        table: VisualScriptApiTable::Ecs,
        function: "get_runtime_tuning",
        title: "Get Runtime Tuning",
        category: "Runtime",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::RuntimeTuning),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetRuntimeTuning,
        table: VisualScriptApiTable::Ecs,
        function: "set_runtime_tuning",
        title: "Set Runtime Tuning",
        category: "Runtime",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_91,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetRuntimeConfig,
        table: VisualScriptApiTable::Ecs,
        function: "get_runtime_config",
        title: "Get Runtime Config",
        category: "Runtime",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::RuntimeConfig),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetRuntimeConfig,
        table: VisualScriptApiTable::Ecs,
        function: "set_runtime_config",
        title: "Set Runtime Config",
        category: "Runtime",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_92,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetRenderConfig,
        table: VisualScriptApiTable::Ecs,
        function: "get_render_config",
        title: "Get Render Config",
        category: "Rendering",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::RenderConfig),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetRenderConfig,
        table: VisualScriptApiTable::Ecs,
        function: "set_render_config",
        title: "Set Render Config",
        category: "Rendering",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_93,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetShaderConstants,
        table: VisualScriptApiTable::Ecs,
        function: "get_shader_constants",
        title: "Get Shader Constants",
        category: "Rendering",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::ShaderConstants),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetShaderConstants,
        table: VisualScriptApiTable::Ecs,
        function: "set_shader_constants",
        title: "Set Shader Constants",
        category: "Rendering",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_94,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetStreamingTuning,
        table: VisualScriptApiTable::Ecs,
        function: "get_streaming_tuning",
        title: "Get Streaming Tuning",
        category: "Streaming",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::StreamingTuning),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetStreamingTuning,
        table: VisualScriptApiTable::Ecs,
        function: "set_streaming_tuning",
        title: "Set Streaming Tuning",
        category: "Streaming",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_95,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetRenderPasses,
        table: VisualScriptApiTable::Ecs,
        function: "get_render_passes",
        title: "Get Render Passes",
        category: "Rendering",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::RenderPasses),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetRenderPasses,
        table: VisualScriptApiTable::Ecs,
        function: "set_render_passes",
        title: "Set Render Passes",
        category: "Rendering",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_96,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetGpuBudget,
        table: VisualScriptApiTable::Ecs,
        function: "get_gpu_budget",
        title: "Get GPU Budget",
        category: "Budgets",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::GpuBudget),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetGpuBudget,
        table: VisualScriptApiTable::Ecs,
        function: "set_gpu_budget",
        title: "Set GPU Budget",
        category: "Budgets",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_97,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetAssetBudgets,
        table: VisualScriptApiTable::Ecs,
        function: "get_asset_budgets",
        title: "Get Asset Budgets",
        category: "Budgets",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::AssetBudgets),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetAssetBudgets,
        table: VisualScriptApiTable::Ecs,
        function: "set_asset_budgets",
        title: "Set Asset Budgets",
        category: "Budgets",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_98,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetWindowSettings,
        table: VisualScriptApiTable::Ecs,
        function: "get_window_settings",
        title: "Get Window Settings",
        category: "Window",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::WindowSettings),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetWindowSettings,
        table: VisualScriptApiTable::Ecs,
        function: "set_window_settings",
        title: "Set Window Settings",
        category: "Window",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_99,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetSpriteRenderer,
        table: VisualScriptApiTable::Ecs,
        function: "get_sprite_renderer",
        title: "Get Sprite Renderer",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::SpriteRenderer),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetSpriteRendererTexturePath,
        table: VisualScriptApiTable::Ecs,
        function: "get_sprite_renderer_texture_path",
        title: "Get Sprite Texture Path",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::String),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetSpriteRenderer,
        table: VisualScriptApiTable::Ecs,
        function: "set_sprite_renderer",
        title: "Set Sprite Renderer",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_100,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetSpriteRendererSheetAnimation,
        table: VisualScriptApiTable::Ecs,
        function: "set_sprite_renderer_sheet_animation",
        title: "Set Sprite Sheet Animation",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_102,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetSpriteRendererSequence,
        table: VisualScriptApiTable::Ecs,
        function: "set_sprite_renderer_sequence",
        title: "Set Sprite Sequence",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_103,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetSpriteRendererTexturePath,
        table: VisualScriptApiTable::Ecs,
        function: "set_sprite_renderer_texture_path",
        title: "Set Sprite Texture Path",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_36,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetText2d,
        table: VisualScriptApiTable::Ecs,
        function: "get_text2d",
        title: "Get Text 2D",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Text2d),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetText2d,
        table: VisualScriptApiTable::Ecs,
        function: "set_text2d",
        title: "Set Text 2D",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_101,
        output_type: Some(VisualValueType::Bool),
    },
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualMathOp {
    #[default]
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Min,
    Max,
}

impl VisualMathOp {
    fn title(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::Multiply => "Multiply",
            Self::Divide => "Divide",
            Self::Modulo => "Modulo",
            Self::Min => "Min",
            Self::Max => "Max",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualTrigOp {
    #[default]
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
}

impl VisualTrigOp {
    fn title(self) -> &'static str {
        match self {
            Self::Sin => "Sin",
            Self::Cos => "Cos",
            Self::Tan => "Tan",
            Self::Asin => "Asin",
            Self::Acos => "Acos",
            Self::Atan => "Atan",
            Self::Atan2 => "Atan2",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualInterpolationOp {
    #[default]
    Lerp,
    SmoothStep,
    InverseLerp,
    Vec3Lerp,
    QuatSlerp,
}

impl VisualInterpolationOp {
    fn title(self) -> &'static str {
        match self {
            Self::Lerp => "Lerp (Number)",
            Self::SmoothStep => "SmoothStep",
            Self::InverseLerp => "Inverse Lerp",
            Self::Vec3Lerp => "Lerp (Vec3)",
            Self::QuatSlerp => "Slerp (Quat)",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualVectorMathOp {
    #[default]
    Dot,
    Cross,
    Length,
    Normalize,
    Distance,
}

impl VisualVectorMathOp {
    fn title(self) -> &'static str {
        match self {
            Self::Dot => "Dot",
            Self::Cross => "Cross",
            Self::Length => "Length",
            Self::Normalize => "Normalize",
            Self::Distance => "Distance",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualProceduralMathOp {
    #[default]
    Clamp,
    Remap,
    Saturate,
    Fract,
}

impl VisualProceduralMathOp {
    fn title(self) -> &'static str {
        match self {
            Self::Clamp => "Clamp",
            Self::Remap => "Remap",
            Self::Saturate => "Saturate",
            Self::Fract => "Fract",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualUtilityMathOp {
    #[default]
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,
    Sqrt,
    Pow,
    Exp,
    Log,
    Degrees,
    Radians,
}

impl VisualUtilityMathOp {
    fn title(self) -> &'static str {
        match self {
            Self::Abs => "Abs",
            Self::Sign => "Sign",
            Self::Floor => "Floor",
            Self::Ceil => "Ceil",
            Self::Round => "Round",
            Self::Sqrt => "Sqrt",
            Self::Pow => "Pow",
            Self::Exp => "Exp",
            Self::Log => "Log",
            Self::Degrees => "Radians -> Degrees",
            Self::Radians => "Degrees -> Radians",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualCompareOp {
    #[default]
    Equals,
    NotEquals,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

impl VisualCompareOp {
    fn title(self) -> &'static str {
        match self {
            Self::Equals => "Equals",
            Self::NotEquals => "Not Equals",
            Self::Less => "Less",
            Self::LessOrEqual => "Less Or Equal",
            Self::Greater => "Greater",
            Self::GreaterOrEqual => "Greater Or Equal",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualLogicalOp {
    #[default]
    And,
    Or,
}

impl VisualLogicalOp {
    fn title(self) -> &'static str {
        match self {
            Self::And => "And",
            Self::Or => "Or",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualVec2Component {
    #[default]
    X,
    Y,
}

impl VisualVec2Component {
    fn title(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualVec3Component {
    #[default]
    X,
    Y,
    Z,
}

impl VisualVec3Component {
    fn title(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualQuatComponent {
    #[default]
    X,
    Y,
    Z,
    W,
}

impl VisualQuatComponent {
    fn title(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
            Self::W => "W",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualTransformComponent {
    #[default]
    Position,
    Rotation,
    Scale,
}

impl VisualTransformComponent {
    fn title(self) -> &'static str {
        match self {
            Self::Position => "Position",
            Self::Rotation => "Rotation",
            Self::Scale => "Scale",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualPhysicsVelocityComponent {
    #[default]
    Linear,
    Angular,
    WakeUp,
}

impl VisualPhysicsVelocityComponent {
    fn title(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Angular => "Angular",
            Self::WakeUp => "Wake Up",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualInputActionPhase {
    #[default]
    Pressed,
    Released,
    Down,
}

impl VisualInputActionPhase {
    fn title(self) -> &'static str {
        match self {
            Self::Pressed => "Pressed",
            Self::Released => "Released",
            Self::Down => "Down",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VisualScriptNodeKind {
    OnStart,
    OnUpdate,
    OnStop,
    OnCollisionEnter,
    OnCollisionStay,
    OnCollisionExit,
    OnTriggerEnter,
    OnTriggerExit,
    OnInputAction {
        #[serde(default = "default_input_action_name")]
        action: String,
        #[serde(default)]
        phase: VisualInputActionPhase,
    },
    OnCustomEvent {
        #[serde(default = "default_custom_event_name")]
        name: String,
    },
    Sequence {
        #[serde(default = "default_sequence_outputs")]
        outputs: u8,
    },
    Branch {
        #[serde(default = "default_branch_condition")]
        condition: String,
    },
    LoopWhile {
        #[serde(default = "default_loop_condition")]
        condition: String,
        #[serde(default = "default_max_loop_iterations")]
        max_iterations: u32,
    },
    Log {
        #[serde(default = "default_log_message")]
        message: String,
    },
    SetVariable {
        #[serde(default)]
        variable_id: u64,
        #[serde(default = "default_var_name")]
        name: String,
        #[serde(default = "default_var_value")]
        value: String,
    },
    GetVariable {
        #[serde(default)]
        variable_id: u64,
        #[serde(default = "default_var_name")]
        name: String,
        #[serde(default)]
        default_value: String,
    },
    ClearVariable {
        #[serde(default)]
        variable_id: u64,
        #[serde(default = "default_var_name")]
        name: String,
    },
    CallApi {
        #[serde(default)]
        operation: VisualApiOperation,
        #[serde(default)]
        table: VisualScriptApiTable,
        #[serde(default)]
        function: String,
        #[serde(default)]
        args: Vec<String>,
    },
    QueryApi {
        #[serde(default)]
        operation: VisualApiOperation,
        #[serde(default)]
        table: VisualScriptApiTable,
        #[serde(default)]
        function: String,
        #[serde(default)]
        args: Vec<String>,
    },
    BoolLiteral {
        #[serde(default)]
        value: bool,
    },
    NumberLiteral {
        #[serde(default)]
        value: f64,
    },
    StringLiteral {
        #[serde(default)]
        value: String,
    },
    #[serde(alias = "json_literal")]
    AnyLiteral {
        #[serde(default)]
        value: String,
    },
    PhysicsQueryFilterLiteral {
        #[serde(default)]
        value: String,
    },
    SelfEntity,
    DeltaTime,
    TimeSinceStart,
    UnixTimeSeconds,
    TimeSince {
        #[serde(default = "default_time_since_origin")]
        origin_seconds: String,
    },
    WaitSeconds {
        #[serde(default = "default_wait_seconds")]
        seconds: String,
        #[serde(default = "default_wait_restart_on_retrigger")]
        restart_on_retrigger: bool,
    },
    FunctionStart {
        #[serde(default)]
        function_id: u64,
        #[serde(default)]
        inputs: Vec<VisualFunctionIoDefinition>,
    },
    FunctionReturn {
        #[serde(default)]
        function_id: u64,
        #[serde(default)]
        outputs: Vec<VisualFunctionIoDefinition>,
        #[serde(default)]
        values: Vec<String>,
    },
    CallFunction {
        #[serde(default)]
        function_id: u64,
        #[serde(default = "default_function_name")]
        name: String,
        #[serde(default)]
        inputs: Vec<VisualFunctionIoDefinition>,
        #[serde(default)]
        outputs: Vec<VisualFunctionIoDefinition>,
        #[serde(default)]
        args: Vec<String>,
    },
    MathBinary {
        #[serde(default)]
        op: VisualMathOp,
    },
    MathTrig {
        #[serde(default)]
        op: VisualTrigOp,
    },
    MathInterpolation {
        #[serde(default)]
        op: VisualInterpolationOp,
    },
    MathVector {
        #[serde(default)]
        op: VisualVectorMathOp,
    },
    MathProcedural {
        #[serde(default)]
        op: VisualProceduralMathOp,
    },
    MathUtility {
        #[serde(default)]
        op: VisualUtilityMathOp,
    },
    Compare {
        #[serde(default)]
        op: VisualCompareOp,
    },
    LogicalBinary {
        #[serde(default)]
        op: VisualLogicalOp,
    },
    Not,
    Select {
        #[serde(default = "default_select_value_type")]
        value_type: VisualValueType,
    },
    Vec2GetComponent {
        #[serde(default)]
        component: VisualVec2Component,
    },
    Vec2SetComponent {
        #[serde(default)]
        component: VisualVec2Component,
    },
    Vec3GetComponent {
        #[serde(default)]
        component: VisualVec3Component,
    },
    Vec3SetComponent {
        #[serde(default)]
        component: VisualVec3Component,
    },
    QuatGetComponent {
        #[serde(default)]
        component: VisualQuatComponent,
    },
    QuatSetComponent {
        #[serde(default)]
        component: VisualQuatComponent,
    },
    TransformGetComponent {
        #[serde(default)]
        component: VisualTransformComponent,
    },
    TransformSetComponent {
        #[serde(default)]
        component: VisualTransformComponent,
    },
    PhysicsVelocityGetComponent {
        #[serde(default)]
        component: VisualPhysicsVelocityComponent,
    },
    PhysicsVelocitySetComponent {
        #[serde(default)]
        component: VisualPhysicsVelocityComponent,
    },
    Vec2,
    Vec3,
    Quat,
    Transform,
    PhysicsVelocity,
    ArrayEmpty {
        #[serde(default = "default_array_item_type")]
        item_type: VisualValueType,
    },
    ArrayLength {
        #[serde(default = "default_array_item_type")]
        item_type: VisualValueType,
    },
    ArrayGet {
        #[serde(default = "default_array_item_type")]
        item_type: VisualValueType,
    },
    ArraySet {
        #[serde(default = "default_array_item_type")]
        item_type: VisualValueType,
    },
    ArrayPush {
        #[serde(default = "default_array_item_type")]
        item_type: VisualValueType,
    },
    ArrayRemoveAt {
        #[serde(default = "default_array_item_type")]
        item_type: VisualValueType,
    },
    ArrayClear {
        #[serde(default = "default_array_item_type")]
        item_type: VisualValueType,
    },
    RayCast,
    Comment {
        #[serde(default)]
        text: String,
    },
    Statement {
        #[serde(default)]
        code: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinKind {
    Exec,
    Data,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PinSlot {
    kind: PinKind,
    index: usize,
}

impl VisualScriptNodeKind {
    fn normalize(&mut self) {
        match self {
            Self::OnInputAction { action, .. } => {
                let normalized = normalize_event_name(action);
                if normalized.is_empty() {
                    *action = default_input_action_name();
                } else {
                    *action = normalized;
                }
            }
            Self::OnCustomEvent { name } => {
                let normalized = normalize_event_name(name);
                if normalized.is_empty() {
                    *name = default_custom_event_name();
                } else {
                    *name = normalized;
                }
            }
            Self::Sequence { outputs } => {
                *outputs = (*outputs).clamp(1, 8);
            }
            Self::LoopWhile { max_iterations, .. } => {
                *max_iterations = (*max_iterations).clamp(1, MAX_LOOP_ITERATIONS);
            }
            Self::CallApi {
                operation,
                table,
                function,
                args,
                ..
            }
            | Self::QueryApi {
                operation,
                table,
                function,
                args,
            } => {
                if !function.trim().is_empty() {
                    if let Some(mapped) = VisualApiOperation::from_table_function(*table, function)
                    {
                        *operation = mapped;
                    }
                }

                let spec = operation.spec();
                *table = spec.table;
                *function = spec.function.to_string();

                if args.len() > MAX_API_ARGS {
                    args.truncate(MAX_API_ARGS);
                }
                let expected = spec.inputs.len().min(MAX_API_ARGS);
                if args.len() > expected {
                    args.truncate(expected);
                }
                while args.len() < expected {
                    let input_index = args.len();
                    let default_value = default_literal_for_api_input(
                        *operation,
                        input_index,
                        spec.inputs[input_index].value_type,
                    );
                    args.push(default_value.to_string());
                }
            }
            Self::FunctionStart { inputs, .. } => {
                normalize_function_io_ports(inputs, "input");
            }
            Self::FunctionReturn {
                outputs, values, ..
            } => {
                normalize_function_io_ports(outputs, "output");
                values.truncate(outputs.len());
                while values.len() < outputs.len() {
                    let index = values.len();
                    let value_type = outputs
                        .get(index)
                        .map(|port| port.value_type)
                        .unwrap_or(VisualValueType::Any);
                    values.push(default_literal_for_type(value_type).to_string());
                }
            }
            Self::CallFunction {
                function_id,
                name,
                inputs,
                outputs,
                args,
            } => {
                if *function_id == 0 && !name.trim().is_empty() {
                    *function_id = hash_string_id(name);
                }
                normalize_function_io_ports(inputs, "input");
                normalize_function_io_ports(outputs, "output");
                args.truncate(inputs.len());
                while args.len() < inputs.len() {
                    let value_type = inputs
                        .get(args.len())
                        .map(|port| port.value_type)
                        .unwrap_or(VisualValueType::Any);
                    args.push(default_literal_for_type(value_type).to_string());
                }
            }
            Self::PhysicsQueryFilterLiteral { value } => {
                if value.trim().is_empty() {
                    *value =
                        default_literal_for_type(VisualValueType::PhysicsQueryFilter).to_string();
                } else {
                    *value = normalize_literal_for_data_type(
                        value,
                        VisualValueType::PhysicsQueryFilter,
                        None,
                    );
                }
            }
            Self::ArrayEmpty { item_type }
            | Self::ArrayLength { item_type }
            | Self::ArrayGet { item_type }
            | Self::ArraySet { item_type }
            | Self::ArrayPush { item_type }
            | Self::ArrayRemoveAt { item_type }
            | Self::ArrayClear { item_type } => {
                if matches!(*item_type, VisualValueType::Array | VisualValueType::Any) {
                    *item_type = default_array_item_type();
                }
            }
            _ => {}
        }
    }

    fn title(&self) -> String {
        match self {
            Self::OnStart => "On Start".to_string(),
            Self::OnUpdate => "On Update".to_string(),
            Self::OnStop => "On Stop".to_string(),
            Self::OnCollisionEnter => "On Collision Enter".to_string(),
            Self::OnCollisionStay => "On Collision Stay".to_string(),
            Self::OnCollisionExit => "On Collision Exit".to_string(),
            Self::OnTriggerEnter => "On Trigger Enter".to_string(),
            Self::OnTriggerExit => "On Trigger Exit".to_string(),
            Self::OnInputAction { action, phase } => {
                format!("On Input {} ({})", phase.title(), action)
            }
            Self::OnCustomEvent { name } => format!("On Event ({})", name),
            Self::Sequence { .. } => "Sequence".to_string(),
            Self::Branch { .. } => "Branch".to_string(),
            Self::LoopWhile { .. } => "Loop While".to_string(),
            Self::Log { .. } => "Log".to_string(),
            Self::SetVariable { .. } => "Set Variable".to_string(),
            Self::GetVariable { .. } => "Get Variable".to_string(),
            Self::ClearVariable { .. } => "Clear Variable".to_string(),
            Self::CallApi { operation, .. } => operation.spec().title.to_string(),
            Self::QueryApi { operation, .. } => operation.spec().title.to_string(),
            Self::BoolLiteral { .. } => "Bool".to_string(),
            Self::NumberLiteral { .. } => "Number".to_string(),
            Self::StringLiteral { .. } => "String".to_string(),
            Self::AnyLiteral { .. } => "Any Value".to_string(),
            Self::PhysicsQueryFilterLiteral { .. } => "Physics Query Filter".to_string(),
            Self::SelfEntity => "Self Entity".to_string(),
            Self::DeltaTime => "Delta Time".to_string(),
            Self::TimeSinceStart => "Time Since Start".to_string(),
            Self::UnixTimeSeconds => "Unix Time Seconds".to_string(),
            Self::TimeSince { .. } => "Time Since".to_string(),
            Self::WaitSeconds { .. } => "Wait Seconds".to_string(),
            Self::FunctionStart { .. } => "Function Start".to_string(),
            Self::FunctionReturn { .. } => "Function Return".to_string(),
            Self::CallFunction { name, .. } => {
                if name.trim().is_empty() {
                    "Call Function".to_string()
                } else {
                    format!("Call {}", name.trim())
                }
            }
            Self::MathBinary { op } => format!("Math: {}", op.title()),
            Self::MathTrig { op } => format!("Trig: {}", op.title()),
            Self::MathInterpolation { op } => format!("Interpolation: {}", op.title()),
            Self::MathVector { op } => format!("Vector: {}", op.title()),
            Self::MathProcedural { op } => format!("Procedural: {}", op.title()),
            Self::MathUtility { op } => format!("Utility: {}", op.title()),
            Self::Compare { op } => format!("Compare: {}", op.title()),
            Self::LogicalBinary { op } => format!("Logical: {}", op.title()),
            Self::Not => "Not".to_string(),
            Self::Select { .. } => "Select".to_string(),
            Self::Vec2GetComponent { component } => format!("Vec2 Get {}", component.title()),
            Self::Vec2SetComponent { component } => format!("Vec2 Set {}", component.title()),
            Self::Vec3GetComponent { component } => format!("Vec3 Get {}", component.title()),
            Self::Vec3SetComponent { component } => format!("Vec3 Set {}", component.title()),
            Self::QuatGetComponent { component } => format!("Quat Get {}", component.title()),
            Self::QuatSetComponent { component } => format!("Quat Set {}", component.title()),
            Self::TransformGetComponent { component } => {
                format!("Transform Get {}", component.title())
            }
            Self::TransformSetComponent { component } => {
                format!("Transform Set {}", component.title())
            }
            Self::PhysicsVelocityGetComponent { component } => {
                format!("Physics Velocity Get {}", component.title())
            }
            Self::PhysicsVelocitySetComponent { component } => {
                format!("Physics Velocity Set {}", component.title())
            }
            Self::Vec2 => "Vec2".to_string(),
            Self::Vec3 => "Vec3".to_string(),
            Self::Quat => "Quat".to_string(),
            Self::Transform => "Transform".to_string(),
            Self::PhysicsVelocity => "Physics Velocity".to_string(),
            Self::ArrayEmpty { .. } => "Array Empty".to_string(),
            Self::ArrayLength { .. } => "Array Length".to_string(),
            Self::ArrayGet { .. } => "Array Get".to_string(),
            Self::ArraySet { .. } => "Array Set".to_string(),
            Self::ArrayPush { .. } => "Array Push".to_string(),
            Self::ArrayRemoveAt { .. } => "Array Remove At".to_string(),
            Self::ArrayClear { .. } => "Array Clear".to_string(),
            Self::RayCast => "Ray Cast".to_string(),
            Self::Comment { .. } => "Comment".to_string(),
            Self::Statement { .. } => "Legacy Statement".to_string(),
        }
    }

    fn exec_input_count(&self) -> usize {
        match self {
            Self::OnStart
            | Self::OnUpdate
            | Self::OnStop
            | Self::OnCollisionEnter
            | Self::OnCollisionStay
            | Self::OnCollisionExit
            | Self::OnTriggerEnter
            | Self::OnTriggerExit
            | Self::OnInputAction { .. }
            | Self::OnCustomEvent { .. }
            | Self::GetVariable { .. }
            | Self::QueryApi { .. }
            | Self::BoolLiteral { .. }
            | Self::NumberLiteral { .. }
            | Self::StringLiteral { .. }
            | Self::AnyLiteral { .. }
            | Self::PhysicsQueryFilterLiteral { .. }
            | Self::SelfEntity
            | Self::DeltaTime
            | Self::TimeSinceStart
            | Self::UnixTimeSeconds
            | Self::TimeSince { .. }
            | Self::MathBinary { .. }
            | Self::MathTrig { .. }
            | Self::MathInterpolation { .. }
            | Self::MathVector { .. }
            | Self::MathProcedural { .. }
            | Self::MathUtility { .. }
            | Self::Compare { .. }
            | Self::LogicalBinary { .. }
            | Self::Not
            | Self::Select { .. }
            | Self::Vec2GetComponent { .. }
            | Self::Vec2SetComponent { .. }
            | Self::Vec3GetComponent { .. }
            | Self::Vec3SetComponent { .. }
            | Self::QuatGetComponent { .. }
            | Self::QuatSetComponent { .. }
            | Self::TransformGetComponent { .. }
            | Self::TransformSetComponent { .. }
            | Self::PhysicsVelocityGetComponent { .. }
            | Self::PhysicsVelocitySetComponent { .. }
            | Self::Vec2
            | Self::Vec3
            | Self::Quat
            | Self::Transform
            | Self::PhysicsVelocity
            | Self::ArrayEmpty { .. }
            | Self::ArrayLength { .. }
            | Self::ArrayGet { .. }
            | Self::ArraySet { .. }
            | Self::ArrayPush { .. }
            | Self::ArrayRemoveAt { .. }
            | Self::ArrayClear { .. }
            | Self::FunctionStart { .. }
            | Self::RayCast => 0,
            Self::Sequence { .. }
            | Self::Branch { .. }
            | Self::LoopWhile { .. }
            | Self::Log { .. }
            | Self::SetVariable { .. }
            | Self::ClearVariable { .. }
            | Self::WaitSeconds { .. }
            | Self::FunctionReturn { .. }
            | Self::CallFunction { .. }
            | Self::CallApi { .. }
            | Self::Comment { .. }
            | Self::Statement { .. } => 1,
        }
    }

    fn exec_output_count(&self) -> usize {
        match self {
            Self::OnStart
            | Self::OnUpdate
            | Self::OnStop
            | Self::OnCollisionEnter
            | Self::OnCollisionStay
            | Self::OnCollisionExit
            | Self::OnTriggerEnter
            | Self::OnTriggerExit
            | Self::OnInputAction { .. }
            | Self::OnCustomEvent { .. } => 1,
            Self::Sequence { outputs } => usize::from((*outputs).clamp(1, 8)),
            Self::Branch { .. } => 2,
            Self::LoopWhile { .. } => 2,
            Self::Log { .. }
            | Self::SetVariable { .. }
            | Self::ClearVariable { .. }
            | Self::WaitSeconds { .. }
            | Self::CallFunction { .. }
            | Self::CallApi { .. }
            | Self::FunctionStart { .. }
            | Self::Comment { .. }
            | Self::Statement { .. } => 1,
            Self::GetVariable { .. }
            | Self::QueryApi { .. }
            | Self::BoolLiteral { .. }
            | Self::NumberLiteral { .. }
            | Self::StringLiteral { .. }
            | Self::AnyLiteral { .. }
            | Self::PhysicsQueryFilterLiteral { .. }
            | Self::SelfEntity
            | Self::DeltaTime
            | Self::TimeSinceStart
            | Self::UnixTimeSeconds
            | Self::TimeSince { .. }
            | Self::MathBinary { .. }
            | Self::MathTrig { .. }
            | Self::MathInterpolation { .. }
            | Self::MathVector { .. }
            | Self::MathProcedural { .. }
            | Self::MathUtility { .. }
            | Self::Compare { .. }
            | Self::LogicalBinary { .. }
            | Self::Not
            | Self::Select { .. }
            | Self::Vec2GetComponent { .. }
            | Self::Vec2SetComponent { .. }
            | Self::Vec3GetComponent { .. }
            | Self::Vec3SetComponent { .. }
            | Self::QuatGetComponent { .. }
            | Self::QuatSetComponent { .. }
            | Self::TransformGetComponent { .. }
            | Self::TransformSetComponent { .. }
            | Self::PhysicsVelocityGetComponent { .. }
            | Self::PhysicsVelocitySetComponent { .. }
            | Self::Vec2
            | Self::Vec3
            | Self::Quat
            | Self::Transform
            | Self::PhysicsVelocity
            | Self::ArrayEmpty { .. }
            | Self::ArrayLength { .. }
            | Self::ArrayGet { .. }
            | Self::ArraySet { .. }
            | Self::ArrayPush { .. }
            | Self::ArrayRemoveAt { .. }
            | Self::ArrayClear { .. }
            | Self::RayCast
            | Self::FunctionReturn { .. } => 0,
        }
    }

    fn data_input_count(&self) -> usize {
        match self {
            Self::Branch { .. }
            | Self::LoopWhile { .. }
            | Self::Log { .. }
            | Self::SetVariable { .. }
            | Self::Not
            | Self::WaitSeconds { .. }
            | Self::TimeSince { .. }
            | Self::ArrayLength { .. }
            | Self::ArrayClear { .. } => 1,
            Self::CallApi { operation, .. } | Self::QueryApi { operation, .. } => {
                operation.spec().inputs.len().min(MAX_API_ARGS)
            }
            Self::CallFunction { inputs, .. } => inputs.len(),
            Self::FunctionReturn { outputs, .. } => outputs.len(),
            Self::MathBinary { .. } | Self::Compare { .. } | Self::LogicalBinary { .. } => 2,
            Self::MathTrig { op } => {
                if *op == VisualTrigOp::Atan2 {
                    2
                } else {
                    1
                }
            }
            Self::MathInterpolation { .. } => 3,
            Self::MathVector { op } => match op {
                VisualVectorMathOp::Dot
                | VisualVectorMathOp::Cross
                | VisualVectorMathOp::Distance => 2,
                VisualVectorMathOp::Length | VisualVectorMathOp::Normalize => 1,
            },
            Self::MathProcedural { op } => match op {
                VisualProceduralMathOp::Clamp => 3,
                VisualProceduralMathOp::Remap => 5,
                VisualProceduralMathOp::Saturate | VisualProceduralMathOp::Fract => 1,
            },
            Self::MathUtility { op } => match op {
                VisualUtilityMathOp::Pow | VisualUtilityMathOp::Log => 2,
                _ => 1,
            },
            Self::Vec2GetComponent { .. }
            | Self::Vec3GetComponent { .. }
            | Self::QuatGetComponent { .. }
            | Self::TransformGetComponent { .. }
            | Self::PhysicsVelocityGetComponent { .. } => 1,
            Self::Vec2SetComponent { .. }
            | Self::Vec3SetComponent { .. }
            | Self::QuatSetComponent { .. }
            | Self::TransformSetComponent { .. }
            | Self::PhysicsVelocitySetComponent { .. } => 2,
            Self::ArrayPush { .. } | Self::ArrayRemoveAt { .. } => 2,
            Self::ArrayGet { .. } | Self::ArraySet { .. } => 3,
            Self::RayCast => 6,
            Self::Vec2 => 2,
            Self::Vec3 | Self::Select { .. } | Self::Transform | Self::PhysicsVelocity => 3,
            Self::Quat => 4,
            Self::OnStart
            | Self::OnUpdate
            | Self::OnStop
            | Self::OnCollisionEnter
            | Self::OnCollisionStay
            | Self::OnCollisionExit
            | Self::OnTriggerEnter
            | Self::OnTriggerExit
            | Self::OnInputAction { .. }
            | Self::OnCustomEvent { .. }
            | Self::Sequence { .. }
            | Self::ClearVariable { .. }
            | Self::GetVariable { .. }
            | Self::BoolLiteral { .. }
            | Self::NumberLiteral { .. }
            | Self::StringLiteral { .. }
            | Self::AnyLiteral { .. }
            | Self::PhysicsQueryFilterLiteral { .. }
            | Self::SelfEntity
            | Self::DeltaTime
            | Self::TimeSinceStart
            | Self::UnixTimeSeconds
            | Self::FunctionStart { .. }
            | Self::ArrayEmpty { .. }
            | Self::Comment { .. }
            | Self::Statement { .. } => 0,
        }
    }

    fn data_output_count(&self) -> usize {
        match self {
            Self::GetVariable { .. }
            | Self::QueryApi { .. }
            | Self::BoolLiteral { .. }
            | Self::NumberLiteral { .. }
            | Self::StringLiteral { .. }
            | Self::AnyLiteral { .. }
            | Self::PhysicsQueryFilterLiteral { .. }
            | Self::SelfEntity
            | Self::DeltaTime
            | Self::TimeSinceStart
            | Self::UnixTimeSeconds
            | Self::TimeSince { .. }
            | Self::MathBinary { .. }
            | Self::MathTrig { .. }
            | Self::MathInterpolation { .. }
            | Self::MathVector { .. }
            | Self::MathProcedural { .. }
            | Self::MathUtility { .. }
            | Self::Compare { .. }
            | Self::LogicalBinary { .. }
            | Self::Not
            | Self::Select { .. }
            | Self::Vec2GetComponent { .. }
            | Self::Vec2SetComponent { .. }
            | Self::Vec3GetComponent { .. }
            | Self::Vec3SetComponent { .. }
            | Self::QuatGetComponent { .. }
            | Self::QuatSetComponent { .. }
            | Self::TransformGetComponent { .. }
            | Self::TransformSetComponent { .. }
            | Self::PhysicsVelocityGetComponent { .. }
            | Self::PhysicsVelocitySetComponent { .. }
            | Self::Vec2
            | Self::Vec3
            | Self::Quat
            | Self::Transform
            | Self::PhysicsVelocity
            | Self::ArrayEmpty { .. }
            | Self::ArrayLength { .. }
            | Self::ArrayGet { .. }
            | Self::ArraySet { .. }
            | Self::ArrayPush { .. }
            | Self::ArrayRemoveAt { .. }
            | Self::ArrayClear { .. }
            | Self::RayCast
            | Self::CallApi { .. } => 1,
            Self::FunctionStart { inputs, .. } => inputs.len(),
            Self::CallFunction { outputs, .. } => outputs.len(),
            Self::OnStart
            | Self::OnUpdate
            | Self::OnStop
            | Self::OnCollisionEnter
            | Self::OnCollisionStay
            | Self::OnCollisionExit
            | Self::OnTriggerEnter
            | Self::OnTriggerExit
            | Self::OnInputAction { .. }
            | Self::OnCustomEvent { .. }
            | Self::Sequence { .. }
            | Self::Branch { .. }
            | Self::LoopWhile { .. }
            | Self::Log { .. }
            | Self::SetVariable { .. }
            | Self::ClearVariable { .. }
            | Self::WaitSeconds { .. }
            | Self::FunctionReturn { .. }
            | Self::Comment { .. }
            | Self::Statement { .. } => 0,
        }
    }

    fn input_count(&self) -> usize {
        self.exec_input_count() + self.data_input_count()
    }

    fn output_count(&self) -> usize {
        self.exec_output_count() + self.data_output_count()
    }

    fn input_slot(&self, input: usize) -> Option<PinSlot> {
        let exec_inputs = self.exec_input_count();
        if input < exec_inputs {
            return Some(PinSlot {
                kind: PinKind::Exec,
                index: input,
            });
        }
        let data_index = input.saturating_sub(exec_inputs);
        if data_index < self.data_input_count() {
            return Some(PinSlot {
                kind: PinKind::Data,
                index: data_index,
            });
        }
        None
    }

    fn output_slot(&self, output: usize) -> Option<PinSlot> {
        let exec_outputs = self.exec_output_count();
        if output < exec_outputs {
            return Some(PinSlot {
                kind: PinKind::Exec,
                index: output,
            });
        }
        let data_index = output.saturating_sub(exec_outputs);
        if data_index < self.data_output_count() {
            return Some(PinSlot {
                kind: PinKind::Data,
                index: data_index,
            });
        }
        None
    }

    fn input_label(&self, slot: PinSlot) -> String {
        match slot.kind {
            PinKind::Exec => "In".to_string(),
            PinKind::Data => match self {
                Self::Branch { .. } | Self::LoopWhile { .. } | Self::Select { .. } => {
                    if slot.index == 0 {
                        "Condition".to_string()
                    } else if slot.index == 1 {
                        "True".to_string()
                    } else {
                        "False".to_string()
                    }
                }
                Self::Log { .. } => "Message".to_string(),
                Self::SetVariable { .. } => "Value".to_string(),
                Self::CallApi { operation, .. } | Self::QueryApi { operation, .. } => {
                    api_input_label(*operation, slot.index)
                }
                Self::MathBinary { .. } => {
                    if slot.index == 0 {
                        "A".to_string()
                    } else {
                        "B".to_string()
                    }
                }
                Self::MathTrig { op } => {
                    if *op == VisualTrigOp::Atan2 {
                        if slot.index == 0 {
                            "Y".to_string()
                        } else {
                            "X".to_string()
                        }
                    } else {
                        "Radians".to_string()
                    }
                }
                Self::MathInterpolation { op } => match op {
                    VisualInterpolationOp::InverseLerp => match slot.index {
                        0 => "A".to_string(),
                        1 => "B".to_string(),
                        _ => "Value".to_string(),
                    },
                    _ => match slot.index {
                        0 => "A".to_string(),
                        1 => "B".to_string(),
                        _ => "T".to_string(),
                    },
                },
                Self::MathVector { op } => match op {
                    VisualVectorMathOp::Length | VisualVectorMathOp::Normalize => {
                        "Vec3".to_string()
                    }
                    _ => {
                        if slot.index == 0 {
                            "A".to_string()
                        } else {
                            "B".to_string()
                        }
                    }
                },
                Self::MathProcedural { op } => match op {
                    VisualProceduralMathOp::Clamp => match slot.index {
                        0 => "Value".to_string(),
                        1 => "Min".to_string(),
                        _ => "Max".to_string(),
                    },
                    VisualProceduralMathOp::Remap => match slot.index {
                        0 => "Value".to_string(),
                        1 => "In Min".to_string(),
                        2 => "In Max".to_string(),
                        3 => "Out Min".to_string(),
                        _ => "Out Max".to_string(),
                    },
                    VisualProceduralMathOp::Saturate | VisualProceduralMathOp::Fract => {
                        "Value".to_string()
                    }
                },
                Self::MathUtility { op } => match op {
                    VisualUtilityMathOp::Pow => {
                        if slot.index == 0 {
                            "Value".to_string()
                        } else {
                            "Exponent".to_string()
                        }
                    }
                    VisualUtilityMathOp::Log => {
                        if slot.index == 0 {
                            "Value".to_string()
                        } else {
                            "Base".to_string()
                        }
                    }
                    _ => "Value".to_string(),
                },
                Self::Compare { .. } => {
                    if slot.index == 0 {
                        "Left".to_string()
                    } else {
                        "Right".to_string()
                    }
                }
                Self::LogicalBinary { .. } => {
                    if slot.index == 0 {
                        "Left".to_string()
                    } else {
                        "Right".to_string()
                    }
                }
                Self::Not => "Value".to_string(),
                Self::TimeSince { .. } => "From Seconds".to_string(),
                Self::WaitSeconds { .. } => "Seconds".to_string(),
                Self::FunctionReturn { outputs, .. } => outputs
                    .get(slot.index)
                    .map(|port| port.name.trim())
                    .filter(|name| !name.is_empty())
                    .map(|name| name.to_string())
                    .unwrap_or_else(|| format!("Return {}", slot.index + 1)),
                Self::CallFunction { inputs, .. } => inputs
                    .get(slot.index)
                    .map(|port| port.name.trim())
                    .filter(|name| !name.is_empty())
                    .map(|name| name.to_string())
                    .unwrap_or_else(|| format!("Arg {}", slot.index + 1)),
                Self::ArrayLength { .. } => "Array".to_string(),
                Self::ArrayGet { .. } => match slot.index {
                    0 => "Array".to_string(),
                    1 => "Index".to_string(),
                    _ => "Default".to_string(),
                },
                Self::ArraySet { .. } => match slot.index {
                    0 => "Array".to_string(),
                    1 => "Index".to_string(),
                    _ => "Value".to_string(),
                },
                Self::ArrayPush { .. } => {
                    if slot.index == 0 {
                        "Array".to_string()
                    } else {
                        "Value".to_string()
                    }
                }
                Self::ArrayRemoveAt { .. } => {
                    if slot.index == 0 {
                        "Array".to_string()
                    } else {
                        "Index".to_string()
                    }
                }
                Self::ArrayClear { .. } => "Array".to_string(),
                Self::RayCast => match slot.index {
                    0 => "Origin".to_string(),
                    1 => "Direction".to_string(),
                    2 => "Max TOI".to_string(),
                    3 => "Solid".to_string(),
                    4 => "Filter".to_string(),
                    _ => "Exclude Entity".to_string(),
                },
                Self::Vec2GetComponent { .. } => "Vec2".to_string(),
                Self::Vec2SetComponent { component } => {
                    if slot.index == 0 {
                        "Vec2".to_string()
                    } else {
                        component.title().to_string()
                    }
                }
                Self::Vec3GetComponent { .. } => "Vec3".to_string(),
                Self::Vec3SetComponent { component } => {
                    if slot.index == 0 {
                        "Vec3".to_string()
                    } else {
                        component.title().to_string()
                    }
                }
                Self::QuatGetComponent { .. } => "Quat".to_string(),
                Self::QuatSetComponent { component } => {
                    if slot.index == 0 {
                        "Quat".to_string()
                    } else {
                        component.title().to_string()
                    }
                }
                Self::TransformGetComponent { .. } => "Transform".to_string(),
                Self::TransformSetComponent { component } => {
                    if slot.index == 0 {
                        "Transform".to_string()
                    } else {
                        component.title().to_string()
                    }
                }
                Self::PhysicsVelocityGetComponent { .. } => "Physics Velocity".to_string(),
                Self::PhysicsVelocitySetComponent { component } => {
                    if slot.index == 0 {
                        "Physics Velocity".to_string()
                    } else {
                        component.title().to_string()
                    }
                }
                Self::Vec2 => {
                    if slot.index == 0 {
                        "X".to_string()
                    } else {
                        "Y".to_string()
                    }
                }
                Self::Vec3 => match slot.index {
                    0 => "X".to_string(),
                    1 => "Y".to_string(),
                    _ => "Z".to_string(),
                },
                Self::Quat => match slot.index {
                    0 => "X".to_string(),
                    1 => "Y".to_string(),
                    2 => "Z".to_string(),
                    _ => "W".to_string(),
                },
                Self::Transform => match slot.index {
                    0 => "Position".to_string(),
                    1 => "Rotation".to_string(),
                    _ => "Scale".to_string(),
                },
                Self::PhysicsVelocity => match slot.index {
                    0 => "Linear".to_string(),
                    1 => "Angular".to_string(),
                    _ => "Wake Up".to_string(),
                },
                _ => format!("Input {}", slot.index + 1),
            },
        }
    }

    fn output_label(&self, slot: PinSlot) -> String {
        match slot.kind {
            PinKind::Exec => match self {
                Self::OnStart => "Start".to_string(),
                Self::OnUpdate => "Tick".to_string(),
                Self::OnStop => "Stop".to_string(),
                Self::OnCollisionEnter => "Enter".to_string(),
                Self::OnCollisionStay => "Stay".to_string(),
                Self::OnCollisionExit => "Exit".to_string(),
                Self::OnTriggerEnter => "Enter".to_string(),
                Self::OnTriggerExit => "Exit".to_string(),
                Self::OnInputAction { phase, .. } => phase.title().to_string(),
                Self::OnCustomEvent { .. } => "Event".to_string(),
                Self::FunctionStart { .. } => "Start".to_string(),
                Self::Branch { .. } => {
                    if slot.index == 0 {
                        "True".to_string()
                    } else {
                        "False".to_string()
                    }
                }
                Self::LoopWhile { .. } => {
                    if slot.index == 0 {
                        "Loop".to_string()
                    } else {
                        "Done".to_string()
                    }
                }
                Self::Sequence { .. } => format!("Then {}", slot.index + 1),
                _ => "Next".to_string(),
            },
            PinKind::Data => match self {
                Self::Compare { .. } | Self::LogicalBinary { .. } | Self::Not => "Bool".to_string(),
                Self::CallApi { operation, .. } | Self::QueryApi { operation, .. } => operation
                    .spec()
                    .output_type
                    .map(|value| value.title().to_string())
                    .unwrap_or_else(|| "Result".to_string()),
                Self::Vec2GetComponent { component } => component.title().to_string(),
                Self::Vec2SetComponent { .. } => "Vec2".to_string(),
                Self::Vec3GetComponent { component } => component.title().to_string(),
                Self::Vec3SetComponent { .. } => "Vec3".to_string(),
                Self::QuatGetComponent { component } => component.title().to_string(),
                Self::QuatSetComponent { .. } => "Quat".to_string(),
                Self::TransformGetComponent { component } => component.title().to_string(),
                Self::TransformSetComponent { .. } => "Transform".to_string(),
                Self::PhysicsVelocityGetComponent { component } => component.title().to_string(),
                Self::PhysicsVelocitySetComponent { .. } => "Physics Velocity".to_string(),
                Self::Vec2 => "Vec2".to_string(),
                Self::Vec3 => "Vec3".to_string(),
                Self::Quat => "Quat".to_string(),
                Self::Transform => "Transform".to_string(),
                Self::PhysicsVelocity => "Physics Velocity".to_string(),
                Self::MathTrig { .. }
                | Self::MathInterpolation { .. }
                | Self::MathProcedural { .. }
                | Self::MathUtility { .. }
                | Self::TimeSince { .. }
                | Self::TimeSinceStart
                | Self::UnixTimeSeconds => "Number".to_string(),
                Self::MathVector { op } => match op {
                    VisualVectorMathOp::Cross | VisualVectorMathOp::Normalize => "Vec3".to_string(),
                    VisualVectorMathOp::Dot
                    | VisualVectorMathOp::Length
                    | VisualVectorMathOp::Distance => "Number".to_string(),
                },
                Self::FunctionStart { inputs, .. } => inputs
                    .get(slot.index)
                    .map(|port| {
                        let name = port.name.trim();
                        if name.is_empty() {
                            format!("Input {}", slot.index + 1)
                        } else {
                            name.to_string()
                        }
                    })
                    .unwrap_or_else(|| format!("Input {}", slot.index + 1)),
                Self::CallFunction { outputs, .. } => outputs
                    .get(slot.index)
                    .map(|port| {
                        let name = port.name.trim();
                        if name.is_empty() {
                            format!("Output {}", slot.index + 1)
                        } else {
                            name.to_string()
                        }
                    })
                    .unwrap_or_else(|| format!("Output {}", slot.index + 1)),
                Self::ArrayEmpty { .. }
                | Self::ArraySet { .. }
                | Self::ArrayPush { .. }
                | Self::ArrayRemoveAt { .. }
                | Self::ArrayClear { .. } => "Array".to_string(),
                Self::ArrayLength { .. } => "Length".to_string(),
                Self::ArrayGet { .. } => "Value".to_string(),
                Self::PhysicsQueryFilterLiteral { .. } => "Filter".to_string(),
                Self::RayCast => "Ray Cast Hit".to_string(),
                _ => "Value".to_string(),
            },
        }
    }

    fn pin_color(&self, slot: PinSlot, is_output: bool) -> Color32 {
        match slot.kind {
            PinKind::Data => PIN_COLOR_DATA,
            PinKind::Exec => match self {
                Self::OnStart
                | Self::OnUpdate
                | Self::OnStop
                | Self::OnCollisionEnter
                | Self::OnCollisionStay
                | Self::OnCollisionExit
                | Self::OnTriggerEnter
                | Self::OnTriggerExit
                | Self::OnInputAction { .. }
                | Self::OnCustomEvent { .. }
                    if is_output =>
                {
                    PIN_COLOR_EVENT
                }
                Self::Branch { .. } | Self::Sequence { .. } | Self::LoopWhile { .. } => {
                    PIN_COLOR_CONTROL
                }
                _ => PIN_COLOR_EXEC,
            },
        }
    }

    fn requires_body(&self) -> bool {
        !matches!(
            self,
            Self::OnStart
                | Self::OnUpdate
                | Self::OnStop
                | Self::OnCollisionEnter
                | Self::OnCollisionStay
                | Self::OnCollisionExit
                | Self::OnTriggerEnter
                | Self::OnTriggerExit
                | Self::SelfEntity
                | Self::DeltaTime
                | Self::TimeSinceStart
                | Self::UnixTimeSeconds
                | Self::FunctionStart { .. }
                | Self::RayCast
        )
    }
}

fn default_visual_script_version() -> u32 {
    VISUAL_SCRIPT_VERSION
}

fn default_true() -> bool {
    true
}

fn default_sequence_outputs() -> u8 {
    2
}

fn default_branch_condition() -> String {
    "true".to_string()
}

fn default_loop_condition() -> String {
    "true".to_string()
}

fn default_max_loop_iterations() -> u32 {
    256
}

fn default_log_message() -> String {
    "hello from visual script".to_string()
}

fn default_input_action_name() -> String {
    "jump".to_string()
}

fn default_custom_event_name() -> String {
    "event".to_string()
}

fn normalize_event_name(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        let normalized = byte.to_ascii_lowercase();
        if normalized.is_ascii_alphanumeric() {
            out.push(normalized as char);
        }
    }
    out
}

fn default_var_name() -> String {
    "value".to_string()
}

fn default_function_name() -> String {
    "function".to_string()
}

fn default_function_io_name() -> String {
    "value".to_string()
}

fn default_var_value() -> String {
    "0".to_string()
}

fn default_time_since_origin() -> String {
    "0".to_string()
}

fn default_wait_seconds() -> String {
    "1.0".to_string()
}

fn default_wait_restart_on_retrigger() -> bool {
    false
}

fn default_array_item_type() -> VisualValueType {
    VisualValueType::String
}

fn normalize_array_item_type(
    value_type: VisualValueType,
    array_item: &mut Option<VisualValueType>,
) {
    if value_type != VisualValueType::Array {
        *array_item = None;
        return;
    }
    let mut selected = array_item.unwrap_or(default_array_item_type());
    if matches!(selected, VisualValueType::Array | VisualValueType::Any) {
        selected = default_array_item_type();
    }
    *array_item = Some(selected);
}

fn default_select_value_type() -> VisualValueType {
    VisualValueType::Number
}

fn hash_string_id(value: &str) -> u64 {
    let mut hash = 14695981039346656037u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    hash.max(1)
}

fn normalize_function_io_ports(ports: &mut Vec<VisualFunctionIoDefinition>, fallback_prefix: &str) {
    if ports.len() > MAX_FUNCTION_IO_PORTS {
        ports.truncate(MAX_FUNCTION_IO_PORTS);
    }

    let mut seen_ids = HashSet::new();
    let mut seen_names = HashSet::new();
    let mut next_id = 1u64;

    for port in ports {
        port.name = port.name.trim().to_string();
        if port.id == 0 || !seen_ids.insert(port.id) {
            while seen_ids.contains(&next_id) {
                next_id = next_id.saturating_add(1);
            }
            port.id = next_id;
            seen_ids.insert(port.id);
            next_id = next_id.saturating_add(1);
        } else {
            next_id = next_id.max(port.id.saturating_add(1));
        }

        if port.name.is_empty() {
            port.name = format!("{}_{}", fallback_prefix, port.id);
        }

        if !seen_names.insert(port.name.clone()) {
            let base = port.name.clone();
            let mut suffix = 2u32;
            loop {
                let candidate = format!("{}_{}", base, suffix);
                suffix = suffix.saturating_add(1);
                if seen_names.insert(candidate.clone()) {
                    port.name = candidate;
                    break;
                }
            }
        }

        normalize_array_item_type(port.value_type, &mut port.array_item_type);
        if port.default_value.trim().is_empty() {
            port.default_value = default_literal_for_type(port.value_type).to_string();
        } else {
            port.default_value = normalize_literal_for_data_type(
                &port.default_value,
                port.value_type,
                port.array_item_type,
            );
        }
    }
}

fn next_function_id(functions: &[VisualScriptFunctionDefinition]) -> u64 {
    functions
        .iter()
        .map(|function| function.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn default_function_graph_data(
    function_id: u64,
    inputs: &[VisualFunctionIoDefinition],
    outputs: &[VisualFunctionIoDefinition],
) -> VisualScriptGraphData {
    let mut snarl = Snarl::new();
    let start = snarl.insert_node(
        egui::pos2(56.0, 110.0),
        VisualScriptNodeKind::FunctionStart {
            function_id,
            inputs: inputs.to_vec(),
        },
    );
    let return_values = outputs
        .iter()
        .map(|output| default_literal_for_type(output.value_type).to_string())
        .collect::<Vec<_>>();
    let ret = snarl.insert_node(
        egui::pos2(360.0, 110.0),
        VisualScriptNodeKind::FunctionReturn {
            function_id,
            outputs: outputs.to_vec(),
            values: return_values,
        },
    );
    snarl.connect(
        OutPinId {
            node: start,
            output: 0,
        },
        InPinId {
            node: ret,
            input: 0,
        },
    );
    graph_data_from_snarl(&snarl)
}

fn sync_function_signature_nodes(function: &mut VisualScriptFunctionDefinition) {
    if function.graph.nodes.is_empty() {
        function.graph =
            default_function_graph_data(function.id, &function.inputs, &function.outputs);
    }

    let mut start_count = 0usize;
    let mut return_count = 0usize;
    for node in &mut function.graph.nodes {
        match &mut node.kind {
            VisualScriptNodeKind::FunctionStart {
                function_id,
                inputs,
            } => {
                *function_id = function.id;
                *inputs = function.inputs.clone();
                start_count += 1;
            }
            VisualScriptNodeKind::FunctionReturn {
                function_id,
                outputs,
                values,
            } => {
                *function_id = function.id;
                *outputs = function.outputs.clone();
                values.truncate(outputs.len());
                while values.len() < outputs.len() {
                    let value_type = outputs
                        .get(values.len())
                        .map(|port| port.value_type)
                        .unwrap_or(VisualValueType::Any);
                    values.push(default_literal_for_type(value_type).to_string());
                }
                return_count += 1;
            }
            _ => {}
        }
    }

    let mut next_node_id = function
        .graph
        .nodes
        .iter()
        .map(|node| node.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    if start_count == 0 {
        function.graph.nodes.push(VisualScriptNodeRecord {
            id: next_node_id,
            kind: VisualScriptNodeKind::FunctionStart {
                function_id: function.id,
                inputs: function.inputs.clone(),
            },
            pos: [56, 110],
            open: true,
        });
        next_node_id = next_node_id.saturating_add(1);
    }

    if return_count == 0 {
        function.graph.nodes.push(VisualScriptNodeRecord {
            id: next_node_id,
            kind: VisualScriptNodeKind::FunctionReturn {
                function_id: function.id,
                outputs: function.outputs.clone(),
                values: function
                    .outputs
                    .iter()
                    .map(|port| default_literal_for_type(port.value_type).to_string())
                    .collect(),
            },
            pos: [360, 110],
            open: true,
        });
    }
}

fn default_literal_for_type(value_type: VisualValueType) -> &'static str {
    match value_type {
        VisualValueType::Bool => "false",
        VisualValueType::Number => "0",
        VisualValueType::String => "",
        VisualValueType::Entity => "0",
        VisualValueType::Array => "[]",
        VisualValueType::Vec2 => "{\"x\":0,\"y\":0}",
        VisualValueType::Vec3 => "{\"x\":0,\"y\":0,\"z\":0}",
        VisualValueType::Quat => "{\"x\":0,\"y\":0,\"z\":0,\"w\":1}",
        VisualValueType::Transform => {
            "{\"position\":{\"x\":0,\"y\":0,\"z\":0},\"rotation\":{\"x\":0,\"y\":0,\"z\":0,\"w\":1},\"scale\":{\"x\":1,\"y\":1,\"z\":1}}"
        }
        VisualValueType::Camera => {
            "{\"fov_y_rad\":1.0,\"aspect_ratio\":1.7777778,\"near_plane\":0.1,\"far_plane\":2000.0,\"active\":false}"
        }
        VisualValueType::Light => {
            "{\"type\":\"Point\",\"color\":{\"x\":1,\"y\":1,\"z\":1},\"intensity\":10.0}"
        }
        VisualValueType::MeshRenderer => {
            "{\"source\":\"Cube\",\"casts_shadow\":true,\"visible\":true}"
        }
        VisualValueType::SpriteRenderer => {
            "{\"color\":{\"x\":1,\"y\":1,\"z\":1,\"w\":1},\"texture\":null,\"uv_min\":{\"x\":0,\"y\":0},\"uv_max\":{\"x\":1,\"y\":1},\"sheet_animation\":{\"enabled\":false,\"columns\":1,\"rows\":1,\"start_frame\":0,\"frame_count\":0,\"fps\":12.0,\"playback\":\"loop\",\"phase\":0.0,\"paused\":false,\"paused_frame\":0,\"flip_x\":false,\"flip_y\":false,\"frame_uv_inset\":{\"x\":0,\"y\":0}},\"image_sequence\":{\"enabled\":false,\"textures\":[],\"start_frame\":0,\"frame_count\":0,\"fps\":12.0,\"playback\":\"loop\",\"phase\":0.0,\"paused\":false,\"paused_frame\":0,\"flip_x\":false,\"flip_y\":false},\"pivot\":{\"x\":0.5,\"y\":0.5},\"layer\":0.0,\"space\":\"world\",\"blend_mode\":\"alpha\",\"billboard\":false,\"visible\":true}"
        }
        VisualValueType::Text2d => {
            "{\"text\":\"\",\"color\":{\"x\":1,\"y\":1,\"z\":1,\"w\":1},\"font_size\":32.0,\"font_weight\":400.0,\"font_width\":1.0,\"font_style\":\"normal\",\"line_height_scale\":1.0,\"letter_spacing\":0.0,\"word_spacing\":0.0,\"underline\":false,\"strikethrough\":false,\"align_h\":\"left\",\"align_v\":\"baseline\",\"space\":\"world\",\"blend_mode\":\"alpha\",\"billboard\":false,\"visible\":true,\"layer\":0.0}"
        }
        VisualValueType::AudioEmitter => {
            "{\"path\":null,\"streaming\":false,\"bus\":\"Master\",\"volume\":1.0,\"pitch\":1.0,\"looping\":false,\"spatial\":true,\"min_distance\":1.0,\"max_distance\":30.0,\"rolloff\":1.0,\"spatial_blend\":1.0,\"play_on_spawn\":false,\"playback_state\":\"Stopped\"}"
        }
        VisualValueType::AudioListener => "{\"enabled\":true}",
        VisualValueType::Script => "{\"path\":\"\",\"language\":\"lua\"}",
        VisualValueType::LookAt => {
            "{\"target_entity\":null,\"target_offset\":{\"x\":0,\"y\":0,\"z\":0},\"offset_in_target_space\":false,\"up\":{\"x\":0,\"y\":1,\"z\":0},\"rotation_smooth_time\":0.0}"
        }
        VisualValueType::EntityFollower => {
            "{\"target_entity\":null,\"position_offset\":{\"x\":0,\"y\":0,\"z\":0},\"offset_in_target_space\":false,\"follow_rotation\":true,\"position_smooth_time\":0.0,\"rotation_smooth_time\":0.0}"
        }
        VisualValueType::AnimatorState => {
            "{\"layer_index\":0,\"layer_name\":\"\",\"layer_weight\":1.0,\"layer_additive\":false,\"state_time\":0.0,\"current_state\":0,\"current_state_name\":\"\",\"states\":[],\"transitions\":[],\"transition\":null}"
        }
        VisualValueType::InputModifiers => {
            "{\"shift\":false,\"ctrl\":false,\"alt\":false,\"super\":false}"
        }
        VisualValueType::AudioStreamingConfig => "{\"buffer_frames\":8192,\"chunk_frames\":2048}",
        VisualValueType::RuntimeTuning => {
            "{\"render_message_capacity\":96,\"asset_stream_queue_capacity\":96,\"asset_worker_queue_capacity\":96,\"max_pending_asset_uploads\":48,\"max_pending_asset_bytes\":536870912,\"asset_uploads_per_frame\":8,\"wgpu_poll_interval_frames\":1,\"wgpu_poll_mode\":1,\"pixels_per_line\":38,\"title_update_ms\":200,\"resize_debounce_ms\":500,\"max_logic_steps_per_frame\":4,\"target_tickrate\":120.0,\"target_fps\":0.0}"
        }
        VisualValueType::RuntimeConfig => {
            "{\"egui\":true,\"wgpu_experimental_features\":false,\"wgpu_backend\":\"auto\",\"binding_backend\":\"auto\",\"fixed_timestep\":false}"
        }
        VisualValueType::RenderConfig => include_str!("visual_defaults/render_config.json"),
        VisualValueType::ShaderConstants => include_str!("visual_defaults/shader_constants.json"),
        VisualValueType::StreamingTuning => include_str!("visual_defaults/streaming_tuning.json"),
        VisualValueType::RenderPasses => {
            "{\"gbuffer\":true,\"shadow\":true,\"direct_lighting\":true,\"sky\":true,\"ssgi\":true,\"ssgi_denoise\":true,\"ssr\":true,\"ddgi\":true,\"egui\":true,\"gizmo\":true,\"transparent\":true,\"occlusion\":true}"
        }
        VisualValueType::GpuBudget => {
            "{\"soft_mib\":0.0,\"hard_mib\":0.0,\"idle_frames\":0,\"mesh_soft_mib\":0.0,\"mesh_hard_mib\":0.0,\"material_soft_mib\":0.0,\"material_hard_mib\":0.0,\"texture_soft_mib\":0.0,\"texture_hard_mib\":0.0,\"texture_view_soft_mib\":0.0,\"texture_view_hard_mib\":0.0,\"sampler_soft_mib\":0.0,\"sampler_hard_mib\":0.0,\"buffer_soft_mib\":0.0,\"buffer_hard_mib\":0.0,\"external_soft_mib\":0.0,\"external_hard_mib\":0.0,\"transient_soft_mib\":0.0,\"transient_hard_mib\":0.0}"
        }
        VisualValueType::AssetBudgets => {
            "{\"mesh_mib\":0.0,\"texture_mib\":0.0,\"material_mib\":0.0,\"audio_mib\":0.0,\"scene_mib\":0.0}"
        }
        VisualValueType::WindowSettings => {
            "{\"title_mode\":\"stats\",\"custom_title\":\"helmer engine\",\"fullscreen\":false,\"resizable\":true,\"decorations\":true,\"maximized\":false,\"minimized\":false,\"visible\":true}"
        }
        VisualValueType::Spline => {
            "{\"points\":[],\"closed\":false,\"tension\":0.5,\"mode\":\"CatmullRom\"}"
        }
        VisualValueType::Physics => {
            "{\"body_kind\":{\"type\":\"Fixed\"},\"collider_shape\":{\"type\":\"Cuboid\"},\"rigid_body_properties\":{\"linear_damping\":0.0,\"angular_damping\":0.0,\"gravity_scale\":1.0,\"ccd_enabled\":false,\"can_sleep\":true,\"sleeping\":false,\"dominance_group\":0,\"lock_translation_x\":false,\"lock_translation_y\":false,\"lock_translation_z\":false,\"lock_rotation_x\":false,\"lock_rotation_y\":false,\"lock_rotation_z\":false,\"linear_velocity\":{\"x\":0,\"y\":0,\"z\":0},\"angular_velocity\":{\"x\":0,\"y\":0,\"z\":0}},\"collider_properties\":{\"friction\":0.5,\"restitution\":0.0,\"density\":1.0,\"enabled\":true,\"is_sensor\":false}}"
        }
        VisualValueType::PhysicsVelocity => {
            "{\"linear\":{\"x\":0,\"y\":0,\"z\":0},\"angular\":{\"x\":0,\"y\":0,\"z\":0},\"wake_up\":true}"
        }
        VisualValueType::PhysicsWorldDefaults => "{\"gravity\":{\"x\":0,\"y\":-9.81,\"z\":0}}",
        VisualValueType::CharacterControllerOutput => {
            "{\"desired_translation\":{\"x\":0,\"y\":0,\"z\":0},\"effective_translation\":{\"x\":0,\"y\":0,\"z\":0},\"remaining_translation\":{\"x\":0,\"y\":0,\"z\":0},\"grounded\":false,\"sliding_down_slope\":false,\"collision_count\":0,\"ground_normal\":{\"x\":0,\"y\":1,\"z\":0},\"slope_angle\":0.0,\"hit_normal\":{\"x\":0,\"y\":0,\"z\":0},\"hit_point\":{\"x\":0,\"y\":0,\"z\":0},\"hit_entity\":null,\"stepped_up\":false,\"step_height\":0.0,\"platform_velocity\":{\"x\":0,\"y\":0,\"z\":0}}"
        }
        VisualValueType::DynamicComponentFields => "{}",
        VisualValueType::DynamicFieldValue => "",
        VisualValueType::PhysicsQueryFilter => {
            "{\"flags\":0,\"groups_memberships\":4294967295,\"groups_filter\":4294967295,\"use_groups\":false}"
        }
        VisualValueType::PhysicsRayCastHit => {
            "{\"has_hit\":false,\"hit_entity\":null,\"point\":{\"x\":0,\"y\":0,\"z\":0},\"normal\":{\"x\":0,\"y\":1,\"z\":0},\"toi\":0.0}"
        }
        VisualValueType::PhysicsPointProjectionHit => {
            "{\"has_hit\":false,\"hit_entity\":null,\"projected_point\":{\"x\":0,\"y\":0,\"z\":0},\"is_inside\":false,\"distance\":0.0}"
        }
        VisualValueType::PhysicsShapeCastHit => {
            "{\"has_hit\":false,\"hit_entity\":null,\"toi\":0.0,\"witness1\":{\"x\":0,\"y\":0,\"z\":0},\"witness2\":{\"x\":0,\"y\":0,\"z\":0},\"normal1\":{\"x\":0,\"y\":1,\"z\":0},\"normal2\":{\"x\":0,\"y\":1,\"z\":0},\"status\":\"Unknown\"}"
        }
        VisualValueType::Any => "null",
    }
}

fn default_literal_for_api_input(
    operation: VisualApiOperation,
    input_index: usize,
    value_type: VisualValueType,
) -> &'static str {
    match (operation, input_index, value_type) {
        (VisualApiOperation::EcsSetRenderConfig, 0, VisualValueType::RenderConfig)
        | (VisualApiOperation::EcsSetShaderConstants, 0, VisualValueType::ShaderConstants)
        | (VisualApiOperation::EcsSetStreamingTuning, 0, VisualValueType::StreamingTuning)
        | (VisualApiOperation::EcsSetSpriteRenderer, 1, VisualValueType::SpriteRenderer)
        | (
            VisualApiOperation::EcsSetSpriteRendererSheetAnimation,
            1,
            VisualValueType::SpriteRenderer,
        )
        | (VisualApiOperation::EcsSetSpriteRendererSequence, 1, VisualValueType::SpriteRenderer)
        | (VisualApiOperation::EcsSetText2d, 1, VisualValueType::Text2d) => "{}",
        _ => default_literal_for_type(value_type),
    }
}

fn literal_string_for_value_type(value: &JsonValue, value_type: VisualValueType) -> String {
    match value_type {
        VisualValueType::Bool => {
            if is_truthy(value) {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        VisualValueType::Number | VisualValueType::Entity => json_to_log_string(value),
        VisualValueType::String => match value {
            JsonValue::String(text) => text.clone(),
            JsonValue::Null => String::new(),
            _ => json_to_log_string(value),
        },
        VisualValueType::Array => compact_json_string(value),
        VisualValueType::Vec2
        | VisualValueType::Vec3
        | VisualValueType::Quat
        | VisualValueType::Transform
        | VisualValueType::Camera
        | VisualValueType::Light
        | VisualValueType::MeshRenderer
        | VisualValueType::SpriteRenderer
        | VisualValueType::Text2d
        | VisualValueType::AudioEmitter
        | VisualValueType::AudioListener
        | VisualValueType::Script
        | VisualValueType::LookAt
        | VisualValueType::EntityFollower
        | VisualValueType::AnimatorState
        | VisualValueType::InputModifiers
        | VisualValueType::AudioStreamingConfig
        | VisualValueType::RuntimeTuning
        | VisualValueType::RuntimeConfig
        | VisualValueType::RenderConfig
        | VisualValueType::ShaderConstants
        | VisualValueType::StreamingTuning
        | VisualValueType::RenderPasses
        | VisualValueType::GpuBudget
        | VisualValueType::AssetBudgets
        | VisualValueType::WindowSettings
        | VisualValueType::Spline
        | VisualValueType::Physics
        | VisualValueType::PhysicsVelocity
        | VisualValueType::PhysicsWorldDefaults
        | VisualValueType::CharacterControllerOutput
        | VisualValueType::DynamicComponentFields
        | VisualValueType::DynamicFieldValue
        | VisualValueType::PhysicsQueryFilter
        | VisualValueType::PhysicsRayCastHit
        | VisualValueType::PhysicsPointProjectionHit
        | VisualValueType::PhysicsShapeCastHit
        | VisualValueType::Any => compact_json_string(value),
    }
}

fn normalize_literal_for_type(value: &str, value_type: VisualValueType) -> String {
    let parsed = parse_loose_literal(value);
    let coerced = coerce_json_to_visual_type(&parsed, value_type)
        .unwrap_or_else(|_| parse_loose_literal(default_literal_for_type(value_type)));
    literal_string_for_value_type(&coerced, value_type)
}

fn infer_visual_value_type_from_literal(value: &str) -> VisualValueType {
    infer_visual_value_type_from_json(&parse_loose_literal(value))
}

fn infer_visual_value_type_from_json(value: &JsonValue) -> VisualValueType {
    match value {
        JsonValue::Bool(_) => VisualValueType::Bool,
        JsonValue::Number(_) => VisualValueType::Number,
        JsonValue::String(_) => VisualValueType::String,
        JsonValue::Array(array) => {
            let numeric = |count: usize| {
                array.len() == count && array.iter().all(|entry| coerce_json_to_f64(entry).is_ok())
            };
            if numeric(2) {
                VisualValueType::Vec2
            } else if numeric(3) {
                VisualValueType::Vec3
            } else if numeric(4) {
                VisualValueType::Quat
            } else {
                VisualValueType::Array
            }
        }
        JsonValue::Object(object) => {
            let has_camera = (object.contains_key("fov_y_rad")
                || object.contains_key("aspect_ratio")
                || object.contains_key("near_plane")
                || object.contains_key("far_plane")
                || object.contains_key("active"))
                && object
                    .get("fov_y_rad")
                    .map(|entry| coerce_json_to_f64(entry).is_ok())
                    .unwrap_or(true)
                && object
                    .get("aspect_ratio")
                    .map(|entry| coerce_json_to_f64(entry).is_ok())
                    .unwrap_or(true)
                && object
                    .get("near_plane")
                    .map(|entry| coerce_json_to_f64(entry).is_ok())
                    .unwrap_or(true)
                && object
                    .get("far_plane")
                    .map(|entry| coerce_json_to_f64(entry).is_ok())
                    .unwrap_or(true);
            let has_light = (object.contains_key("type")
                || object.contains_key("color")
                || object.contains_key("intensity")
                || object.contains_key("angle"))
                && object
                    .get("type")
                    .map(|entry| matches!(entry, JsonValue::String(_) | JsonValue::Null))
                    .unwrap_or(true)
                && object
                    .get("color")
                    .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3).is_ok())
                    .unwrap_or(true)
                && object
                    .get("intensity")
                    .map(|entry| coerce_json_to_f64(entry).is_ok())
                    .unwrap_or(true)
                && object
                    .get("angle")
                    .map(|entry| coerce_json_to_f64(entry).is_ok())
                    .unwrap_or(true);
            let has_mesh_renderer = object.contains_key("source")
                || object.contains_key("material")
                || object.contains_key("casts_shadow")
                || object.contains_key("visible");
            let has_text2d = object.contains_key("text")
                || object.contains_key("font_path")
                || object.contains_key("font_family")
                || object.contains_key("font_size")
                || object.contains_key("font_weight")
                || object.contains_key("font_width")
                || object.contains_key("font_style")
                || object.contains_key("line_height_scale")
                || object.contains_key("letter_spacing")
                || object.contains_key("word_spacing")
                || object.contains_key("underline")
                || object.contains_key("strikethrough")
                || object.contains_key("max_width")
                || object.contains_key("align_h")
                || object.contains_key("align_v");
            let has_sprite_renderer = object.contains_key("texture_id")
                || object.contains_key("texture")
                || object.contains_key("uv_min")
                || object.contains_key("uv_max")
                || object.contains_key("sheet_animation")
                || object.contains_key("sheet")
                || object.contains_key("image_sequence")
                || object.contains_key("sequence")
                || object.contains_key("pivot")
                || ((object.contains_key("color")
                    || object.contains_key("clip_rect")
                    || object.contains_key("blend_mode")
                    || object.contains_key("billboard")
                    || object.contains_key("space")
                    || object.contains_key("pick_id"))
                    && !has_text2d
                    && !has_mesh_renderer);
            let has_audio_emitter = object.contains_key("path")
                || object.contains_key("streaming")
                || object.contains_key("bus")
                || object.contains_key("volume")
                || object.contains_key("pitch")
                || object.contains_key("looping")
                || object.contains_key("spatial")
                || object.contains_key("min_distance")
                || object.contains_key("max_distance")
                || object.contains_key("rolloff")
                || object.contains_key("spatial_blend")
                || object.contains_key("play_on_spawn")
                || object.contains_key("playback_state")
                || object.contains_key("clip_id");
            let has_audio_listener = object.contains_key("enabled") && object.len() == 1;
            let has_input_modifiers = object.contains_key("shift")
                || object.contains_key("ctrl")
                || object.contains_key("alt")
                || object.contains_key("super");
            let has_audio_streaming =
                object.contains_key("buffer_frames") || object.contains_key("chunk_frames");
            let has_spline = object.contains_key("points")
                || object.contains_key("closed")
                || object.contains_key("tension")
                || object.contains_key("mode");
            let has_script = object.contains_key("path") && object.contains_key("language");
            let has_physics_velocity = (object.contains_key("linear")
                || object.contains_key("angular")
                || object.contains_key("wake_up"))
                && object
                    .get("linear")
                    .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3).is_ok())
                    .unwrap_or(true)
                && object
                    .get("angular")
                    .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3).is_ok())
                    .unwrap_or(true);
            let has_physics_world_defaults = (object.contains_key("gravity")
                || object.contains_key("collider_properties")
                || object.contains_key("rigid_body_properties"))
                && !object.contains_key("body_kind")
                && !object.contains_key("collider_shape")
                && !object.contains_key("joint")
                && !object.contains_key("character_controller")
                && !object.contains_key("ray_cast")
                && !object.contains_key("point_projection")
                && !object.contains_key("shape_cast")
                && !object.contains_key("world_defaults");
            let has_character_output = object.contains_key("effective_translation")
                || object.contains_key("grounded")
                || object.contains_key("sliding_down_slope")
                || object.contains_key("collision_count");
            let has_ray_hit = object.contains_key("point")
                && object.contains_key("normal")
                && object.contains_key("toi");
            let has_point_hit =
                object.contains_key("projected_point") && object.contains_key("distance");
            let has_shape_hit = object.contains_key("witness1")
                && object.contains_key("witness2")
                && object.contains_key("normal1")
                && object.contains_key("normal2");
            let has_query_filter = object.contains_key("flags")
                && object.contains_key("groups_memberships")
                && object.contains_key("groups_filter");
            let has_dynamic_component_fields = object.is_empty()
                || object
                    .values()
                    .all(|entry| coerce_json_to_dynamic_field_value(entry).is_ok());
            let has_physics = object.contains_key("body_kind")
                || object.contains_key("collider_shape")
                || object.contains_key("collider_properties")
                || object.contains_key("collider_inheritance")
                || object.contains_key("rigid_body_properties")
                || object.contains_key("rigid_body_inheritance")
                || object.contains_key("joint")
                || object.contains_key("character_controller")
                || object.contains_key("character_input")
                || object.contains_key("ray_cast")
                || object.contains_key("point_projection")
                || object.contains_key("shape_cast")
                || object.contains_key("world_defaults")
                || object.contains_key("has_handle");
            let has_vec2 = object.contains_key("x")
                && object.contains_key("y")
                && !object.contains_key("z")
                && !object.contains_key("w")
                && coerce_json_to_f64(object.get("x").unwrap_or(&JsonValue::Null)).is_ok()
                && coerce_json_to_f64(object.get("y").unwrap_or(&JsonValue::Null)).is_ok();
            let has_vec3 = object.contains_key("x")
                && object.contains_key("y")
                && object.contains_key("z")
                && !object.contains_key("w")
                && coerce_json_to_f64(object.get("x").unwrap_or(&JsonValue::Null)).is_ok()
                && coerce_json_to_f64(object.get("y").unwrap_or(&JsonValue::Null)).is_ok()
                && coerce_json_to_f64(object.get("z").unwrap_or(&JsonValue::Null)).is_ok();
            let has_quat = object.contains_key("x")
                && object.contains_key("y")
                && object.contains_key("z")
                && object.contains_key("w")
                && coerce_json_to_f64(object.get("x").unwrap_or(&JsonValue::Null)).is_ok()
                && coerce_json_to_f64(object.get("y").unwrap_or(&JsonValue::Null)).is_ok()
                && coerce_json_to_f64(object.get("z").unwrap_or(&JsonValue::Null)).is_ok()
                && coerce_json_to_f64(object.get("w").unwrap_or(&JsonValue::Null)).is_ok();
            let has_transform = (object.contains_key("position")
                || object.contains_key("rotation")
                || object.contains_key("scale"))
                && object
                    .get("position")
                    .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3).is_ok())
                    .unwrap_or(true)
                && object
                    .get("rotation")
                    .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Quat).is_ok())
                    .unwrap_or(true)
                && object
                    .get("scale")
                    .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3).is_ok())
                    .unwrap_or(true);

            if has_shape_hit {
                VisualValueType::PhysicsShapeCastHit
            } else if has_point_hit {
                VisualValueType::PhysicsPointProjectionHit
            } else if has_ray_hit {
                VisualValueType::PhysicsRayCastHit
            } else if has_character_output {
                VisualValueType::CharacterControllerOutput
            } else if has_physics_velocity {
                VisualValueType::PhysicsVelocity
            } else if has_physics_world_defaults {
                VisualValueType::PhysicsWorldDefaults
            } else if has_query_filter {
                VisualValueType::PhysicsQueryFilter
            } else if has_physics {
                VisualValueType::Physics
            } else if has_spline {
                VisualValueType::Spline
            } else if has_script {
                VisualValueType::Script
            } else if has_audio_emitter {
                VisualValueType::AudioEmitter
            } else if has_audio_streaming {
                VisualValueType::AudioStreamingConfig
            } else if has_input_modifiers {
                VisualValueType::InputModifiers
            } else if has_audio_listener {
                VisualValueType::AudioListener
            } else if has_text2d {
                VisualValueType::Text2d
            } else if has_sprite_renderer {
                VisualValueType::SpriteRenderer
            } else if has_mesh_renderer {
                VisualValueType::MeshRenderer
            } else if has_camera {
                VisualValueType::Camera
            } else if has_light {
                VisualValueType::Light
            } else if has_transform {
                VisualValueType::Transform
            } else if has_quat {
                VisualValueType::Quat
            } else if has_vec3 {
                VisualValueType::Vec3
            } else if has_vec2 {
                VisualValueType::Vec2
            } else if has_dynamic_component_fields {
                VisualValueType::DynamicComponentFields
            } else {
                VisualValueType::Any
            }
        }
        JsonValue::Null => VisualValueType::Any,
    }
}

fn infer_array_item_type_from_literal(value: &str) -> Option<VisualValueType> {
    let parsed = parse_loose_literal(value);
    let JsonValue::Array(entries) = parsed else {
        return None;
    };
    let Some(first) = entries.first() else {
        return Some(default_array_item_type());
    };
    let mut inferred = infer_visual_value_type_from_json(first);
    if matches!(inferred, VisualValueType::Array | VisualValueType::Any) {
        inferred = default_array_item_type();
    }
    Some(inferred)
}

fn is_dynamic_field_value_type(value_type: VisualValueType) -> bool {
    matches!(
        value_type,
        VisualValueType::Bool
            | VisualValueType::Number
            | VisualValueType::String
            | VisualValueType::Vec3
            | VisualValueType::DynamicFieldValue
    )
}

fn are_data_types_compatible(from_type: VisualValueType, to_type: VisualValueType) -> bool {
    if from_type == to_type {
        return true;
    }

    if matches!(from_type, VisualValueType::Any) || matches!(to_type, VisualValueType::Any) {
        return true;
    }

    if is_dynamic_field_value_type(from_type) && is_dynamic_field_value_type(to_type) {
        return true;
    }

    matches!(
        (from_type, to_type),
        (VisualValueType::Number, VisualValueType::Entity)
            | (VisualValueType::Entity, VisualValueType::Number)
    )
}

fn are_data_types_compatible_strict_typed(
    from_type: VisualValueType,
    to_type: VisualValueType,
) -> bool {
    if from_type == to_type {
        return true;
    }

    if matches!(from_type, VisualValueType::Any) || matches!(to_type, VisualValueType::Any) {
        return true;
    }

    if is_dynamic_field_value_type(from_type) && is_dynamic_field_value_type(to_type) {
        return true;
    }

    matches!(
        (from_type, to_type),
        (VisualValueType::Number, VisualValueType::Entity)
            | (VisualValueType::Entity, VisualValueType::Number)
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualAssetPathKind {
    Scene,
    Script,
    Model,
    Texture,
    Material,
    Audio,
}

fn is_scene_asset_path(path: &Path) -> bool {
    let is_ron = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("ron"))
        .unwrap_or(false);
    if !is_ron {
        return false;
    }
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.ends_with(".hscene.ron"))
        .unwrap_or(false)
}

fn is_model_asset_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "glb" | "gltf"))
        .unwrap_or(false)
}

fn is_texture_asset_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png"
                    | "jpg"
                    | "jpeg"
                    | "tga"
                    | "bmp"
                    | "gif"
                    | "webp"
                    | "ktx"
                    | "ktx2"
                    | "dds"
                    | "hdr"
                    | "exr"
            )
        })
        .unwrap_or(false)
}

fn is_material_asset_path(path: &Path) -> bool {
    let is_ron = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("ron"))
        .unwrap_or(false);
    if !is_ron {
        return false;
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    !(name.eq_ignore_ascii_case("helmer_project.ron")
        || name.ends_with(".hscene.ron")
        || name.ends_with(".hanim.ron"))
}

fn is_audio_asset_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "wav" | "ogg" | "flac" | "mp3" | "aiff" | "aif" | "aifc"
            )
        })
        .unwrap_or(false)
}

fn asset_path_matches_kind(path: &Path, kind: VisualAssetPathKind) -> bool {
    match kind {
        VisualAssetPathKind::Scene => is_scene_asset_path(path) || is_model_asset_path(path),
        VisualAssetPathKind::Script => crate::editor::is_script_path(path),
        VisualAssetPathKind::Model => is_model_asset_path(path),
        VisualAssetPathKind::Texture => is_texture_asset_path(path),
        VisualAssetPathKind::Material => is_material_asset_path(path),
        VisualAssetPathKind::Audio => is_audio_asset_path(path),
    }
}

fn api_input_asset_path_kind(
    operation: VisualApiOperation,
    input_index: usize,
) -> Option<VisualAssetPathKind> {
    match operation {
        VisualApiOperation::EcsOpenScene | VisualApiOperation::EcsSwitchScene
            if input_index == 0 =>
        {
            Some(VisualAssetPathKind::Scene)
        }
        VisualApiOperation::EcsSetSceneAsset if input_index == 1 => {
            Some(VisualAssetPathKind::Scene)
        }
        VisualApiOperation::EcsFindScriptIndex if input_index == 1 => {
            Some(VisualAssetPathKind::Script)
        }
        VisualApiOperation::EcsSetScript if input_index == 1 => Some(VisualAssetPathKind::Script),
        VisualApiOperation::EcsSetMeshRendererSourcePath if input_index == 1 => {
            Some(VisualAssetPathKind::Model)
        }
        VisualApiOperation::EcsSetSpriteRendererTexturePath if input_index == 1 => {
            Some(VisualAssetPathKind::Texture)
        }
        VisualApiOperation::EcsSetMeshRendererMaterialPath if input_index == 1 => {
            Some(VisualAssetPathKind::Material)
        }
        VisualApiOperation::EcsSetAudioEmitterPath if input_index == 1 => {
            Some(VisualAssetPathKind::Audio)
        }
        _ => None,
    }
}

fn path_to_visual_literal(path: &Path, project_root: Option<&Path>) -> String {
    if let Some(root) = project_root {
        if let Ok(relative) = path.strip_prefix(root) {
            return relative.to_string_lossy().replace('\\', "/");
        }
        if let (Ok(path_canonical), Ok(root_canonical)) = (path.canonicalize(), root.canonicalize())
        {
            if let Ok(relative) = path_canonical.strip_prefix(&root_canonical) {
                return relative.to_string_lossy().replace('\\', "/");
            }
        }
    }
    path.to_string_lossy().replace('\\', "/")
}

fn typed_dnd_release_payload<Payload>(response: &egui::Response) -> Option<Arc<Payload>>
where
    Payload: Any + Send + Sync,
{
    if response.dnd_hover_payload::<Payload>().is_some() {
        response.dnd_release_payload::<Payload>()
    } else {
        None
    }
}

fn api_input_label(operation: VisualApiOperation, input_index: usize) -> String {
    let spec = operation.spec();
    let Some(input) = spec.inputs.get(input_index) else {
        return format!("Arg {}", input_index + 1);
    };

    match input.label {
        "Data" => spec
            .title
            .strip_prefix("Set ")
            .map(|label| label.to_string())
            .unwrap_or_else(|| "Data".to_string()),
        "Comp Name" => "Component Name".to_string(),
        "Fields" => "Component Fields".to_string(),
        "Id" => {
            if spec.title.starts_with("Input Gamepad") {
                "Gamepad Id".to_string()
            } else {
                "Id".to_string()
            }
        }
        "Value" if operation == VisualApiOperation::EcsSetViewportPreviewCamera => {
            "Camera Entity Id".to_string()
        }
        "Index"
            if matches!(
                operation,
                VisualApiOperation::EcsGetScript
                    | VisualApiOperation::EcsListScriptFields
                    | VisualApiOperation::EcsGetScriptPath
                    | VisualApiOperation::EcsGetScriptLanguage
                    | VisualApiOperation::EcsGetScriptField
                    | VisualApiOperation::EcsSetScript
                    | VisualApiOperation::EcsSetScriptField
            ) =>
        {
            "Script Index".to_string()
        }
        label => label.to_string(),
    }
}

fn with_data_type_suffix(label: String, value_type: Option<VisualValueType>) -> String {
    let Some(value_type) = value_type else {
        return label;
    };
    if value_type == VisualValueType::Any {
        return label;
    }
    let type_title = value_type.title();
    if label.eq_ignore_ascii_case(type_title) {
        return label;
    }
    let suffix = format!("({})", type_title);
    if label.contains(&suffix) {
        return label;
    }
    format!("{} ({})", label, type_title)
}

fn has_structured_api_default_editor(operation: VisualApiOperation, input_index: usize) -> bool {
    matches!(
        (operation, input_index),
        (VisualApiOperation::EcsSetCamera, 1)
            | (VisualApiOperation::EcsSetLight, 1)
            | (VisualApiOperation::EcsSetMeshRenderer, 1)
            | (VisualApiOperation::EcsSetSpriteRenderer, 1)
            | (VisualApiOperation::EcsSetSpriteRendererSheetAnimation, 1)
            | (VisualApiOperation::EcsSetSpriteRendererSequence, 1)
            | (VisualApiOperation::EcsSetText2d, 1)
            | (VisualApiOperation::EcsSetAudioEmitter, 1)
            | (VisualApiOperation::EcsSetAudioListener, 1)
            | (VisualApiOperation::EcsSetPhysics, 1)
            | (VisualApiOperation::EcsSetPhysicsVelocity, 1)
            | (VisualApiOperation::EcsSetPhysicsWorldDefaults, 1)
            | (VisualApiOperation::EcsSetSpline, 1)
            | (VisualApiOperation::EcsSetDynamicComponent, 2)
            | (VisualApiOperation::EcsSetDynamicField, 3)
            | (VisualApiOperation::EcsSetRuntimeTuning, 0)
            | (VisualApiOperation::EcsSetRuntimeConfig, 0)
            | (VisualApiOperation::EcsSetRenderConfig, 0)
            | (VisualApiOperation::EcsSetShaderConstants, 0)
            | (VisualApiOperation::EcsSetStreamingTuning, 0)
            | (VisualApiOperation::EcsSetRenderPasses, 0)
            | (VisualApiOperation::EcsSetGpuBudget, 0)
            | (VisualApiOperation::EcsSetAssetBudgets, 0)
            | (VisualApiOperation::EcsSetWindowSettings, 0)
    )
}

fn api_input_prefers_vertical_default_layout(
    operation: VisualApiOperation,
    input_index: usize,
) -> bool {
    matches!(
        (operation, input_index),
        (VisualApiOperation::EcsSetPhysics, 1)
            | (VisualApiOperation::EcsSetMeshRenderer, 1)
            | (VisualApiOperation::EcsSetSpriteRenderer, 1)
            | (VisualApiOperation::EcsSetSpriteRendererSheetAnimation, 1)
            | (VisualApiOperation::EcsSetSpriteRendererSequence, 1)
            | (VisualApiOperation::EcsSetText2d, 1)
            | (VisualApiOperation::EcsSetLight, 1)
            | (VisualApiOperation::EcsSetAudioEmitter, 1)
            | (VisualApiOperation::EcsSetSpline, 1)
            | (VisualApiOperation::EcsSetDynamicComponent, 2)
            | (VisualApiOperation::EcsSetRuntimeTuning, 0)
            | (VisualApiOperation::EcsSetRuntimeConfig, 0)
            | (VisualApiOperation::EcsSetRenderConfig, 0)
            | (VisualApiOperation::EcsSetShaderConstants, 0)
            | (VisualApiOperation::EcsSetStreamingTuning, 0)
            | (VisualApiOperation::EcsSetRenderPasses, 0)
            | (VisualApiOperation::EcsSetGpuBudget, 0)
            | (VisualApiOperation::EcsSetAssetBudgets, 0)
            | (VisualApiOperation::EcsSetWindowSettings, 0)
    )
}

fn value_type_prefers_vertical_default_layout(value_type: VisualValueType) -> bool {
    matches!(
        value_type,
        VisualValueType::PhysicsQueryFilter
            | VisualValueType::PhysicsRayCastHit
            | VisualValueType::PhysicsPointProjectionHit
            | VisualValueType::PhysicsShapeCastHit
            | VisualValueType::Transform
            | VisualValueType::Physics
            | VisualValueType::PhysicsVelocity
            | VisualValueType::PhysicsWorldDefaults
            | VisualValueType::MeshRenderer
            | VisualValueType::SpriteRenderer
            | VisualValueType::Text2d
            | VisualValueType::AudioEmitter
            | VisualValueType::AudioListener
            | VisualValueType::LookAt
            | VisualValueType::EntityFollower
            | VisualValueType::AnimatorState
            | VisualValueType::AudioStreamingConfig
            | VisualValueType::Spline
            | VisualValueType::Camera
            | VisualValueType::Light
            | VisualValueType::RuntimeTuning
            | VisualValueType::RuntimeConfig
            | VisualValueType::RenderConfig
            | VisualValueType::ShaderConstants
            | VisualValueType::StreamingTuning
            | VisualValueType::RenderPasses
            | VisualValueType::GpuBudget
            | VisualValueType::AssetBudgets
            | VisualValueType::WindowSettings
            | VisualValueType::CharacterControllerOutput
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum VisualApiMenuSection {
    Entity,
    Scene,
    Physics,
    Animation,
    Audio,
    Spline,
    Transform,
    Runtime,
    Render,
    Streaming,
    Budgets,
    WindowControl,
    Dynamic,
    InputKeyboard,
    InputMouse,
    InputGamepad,
    InputWindow,
    Utility,
}

impl VisualApiMenuSection {
    fn title(self) -> &'static str {
        match self {
            Self::Entity => "Entity",
            Self::Scene => "Scene",
            Self::Physics => "Physics",
            Self::Animation => "Animation",
            Self::Audio => "Audio",
            Self::Spline => "Spline",
            Self::Transform => "Transform",
            Self::Runtime => "Runtime",
            Self::Render => "Rendering",
            Self::Streaming => "Streaming",
            Self::Budgets => "Budgets",
            Self::WindowControl => "Window",
            Self::Dynamic => "Dynamic Components",
            Self::InputKeyboard => "Input / Keyboard",
            Self::InputMouse => "Input / Mouse",
            Self::InputGamepad => "Input / Gamepad",
            Self::InputWindow => "Input / Window",
            Self::Utility => "Utility",
        }
    }
}

const VISUAL_API_MENU_SECTION_ORDER: [VisualApiMenuSection; 18] = [
    VisualApiMenuSection::Entity,
    VisualApiMenuSection::Transform,
    VisualApiMenuSection::Runtime,
    VisualApiMenuSection::Render,
    VisualApiMenuSection::Streaming,
    VisualApiMenuSection::Budgets,
    VisualApiMenuSection::WindowControl,
    VisualApiMenuSection::Scene,
    VisualApiMenuSection::Physics,
    VisualApiMenuSection::Animation,
    VisualApiMenuSection::Audio,
    VisualApiMenuSection::Spline,
    VisualApiMenuSection::Dynamic,
    VisualApiMenuSection::InputKeyboard,
    VisualApiMenuSection::InputMouse,
    VisualApiMenuSection::InputGamepad,
    VisualApiMenuSection::InputWindow,
    VisualApiMenuSection::Utility,
];

fn api_menu_section(spec: &VisualApiOperationSpec) -> VisualApiMenuSection {
    if spec.table == VisualScriptApiTable::Input {
        return match spec.category {
            "Keyboard" => VisualApiMenuSection::InputKeyboard,
            "Mouse" => VisualApiMenuSection::InputMouse,
            "Gamepad" => VisualApiMenuSection::InputGamepad,
            "Window" => VisualApiMenuSection::InputWindow,
            _ => VisualApiMenuSection::Utility,
        };
    }

    let function = spec.function;
    if spec.category == "Runtime"
        || function.contains("runtime_tuning")
        || function.contains("runtime_config")
    {
        return VisualApiMenuSection::Runtime;
    }
    if spec.category == "Window" || function.contains("window_settings") {
        return VisualApiMenuSection::WindowControl;
    }
    if spec.category == "Budgets" {
        return VisualApiMenuSection::Budgets;
    }
    if spec.category == "Streaming" || function.contains("streaming_tuning") {
        return VisualApiMenuSection::Streaming;
    }
    if spec.category == "Rendering" || spec.category == "Graph" {
        return VisualApiMenuSection::Render;
    }
    if function.contains("spline") {
        return VisualApiMenuSection::Spline;
    }
    if function == "get_scene_asset" || function == "set_scene_asset" {
        return VisualApiMenuSection::Render;
    }
    if function.contains("physics")
        || function.contains("ray_cast")
        || function.contains("sphere_cast")
        || function.contains("shape_cast")
        || function.contains("point_projection")
        || function.contains("force")
        || function.contains("torque")
        || function.contains("impulse")
        || function.contains("character_controller")
        || function.contains("collision_event")
        || function.contains("trigger_event")
    {
        return VisualApiMenuSection::Physics;
    }
    if function.contains("anim") {
        return VisualApiMenuSection::Animation;
    }
    if function.contains("audio") {
        return VisualApiMenuSection::Audio;
    }
    if function.contains("transform") {
        return VisualApiMenuSection::Transform;
    }
    if function.contains("look_at") || function.contains("follower") {
        return VisualApiMenuSection::Transform;
    }
    if function.contains("camera")
        || function.contains("light")
        || function.contains("mesh")
        || function.contains("sprite_renderer")
        || function.contains("text2d")
        || function.contains("viewport")
        || function.contains("render")
        || function.contains("graph_template")
    {
        return VisualApiMenuSection::Render;
    }
    if function.contains("script") {
        return VisualApiMenuSection::Entity;
    }
    if function.contains("scene") {
        return VisualApiMenuSection::Scene;
    }
    if function.contains("dynamic") {
        return VisualApiMenuSection::Dynamic;
    }
    if spec.category == "Gameplay" || function.contains("entity") || function.contains("component")
    {
        return VisualApiMenuSection::Entity;
    }

    VisualApiMenuSection::Utility
}

fn add_node_search_matches(search: &str, candidate: &str) -> bool {
    search.is_empty() || candidate.to_ascii_lowercase().contains(search)
}

fn add_node_search_matches_any(search: &str, candidates: &[&str]) -> bool {
    candidates
        .iter()
        .any(|candidate| add_node_search_matches(search, candidate))
}

fn api_spec_matches_add_node_search(
    spec: &VisualApiOperationSpec,
    section: VisualApiMenuSection,
    search: &str,
) -> bool {
    add_node_search_matches_any(
        search,
        &[
            spec.title,
            spec.function,
            spec.category,
            spec.table.title(),
            section.title(),
        ],
    )
}

fn insert_api_node_from_spec(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    pos: egui::Pos2,
    spec: &VisualApiOperationSpec,
) -> NodeId {
    snarl.insert_node(pos, api_node_kind_from_spec(spec))
}

fn api_node_kind_from_spec(spec: &VisualApiOperationSpec) -> VisualScriptNodeKind {
    let args = spec
        .inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            default_literal_for_api_input(spec.operation, index, input.value_type).to_string()
        })
        .collect();
    match spec.flow {
        VisualApiFlow::Exec => VisualScriptNodeKind::CallApi {
            operation: spec.operation,
            table: spec.table,
            function: spec.function.to_string(),
            args,
        },
        VisualApiFlow::Pure => VisualScriptNodeKind::QueryApi {
            operation: spec.operation,
            table: spec.table,
            function: spec.function.to_string(),
            args,
        },
    }
}

fn node_slots_are_compatible(
    from_node: &VisualScriptNodeKind,
    from_slot: PinSlot,
    to_node: &VisualScriptNodeKind,
    to_slot: PinSlot,
    variables: &[VisualVariableDefinition],
) -> bool {
    if from_slot.kind != to_slot.kind {
        return false;
    }

    if matches!(from_slot.kind, PinKind::Data) {
        let Some(from_type) = node_data_output_type(from_node, from_slot.index, variables) else {
            return false;
        };
        let Some(to_type) = node_data_input_type(to_node, to_slot.index, variables) else {
            return false;
        };
        if !are_data_types_compatible_strict_typed(from_type, to_type) {
            return false;
        }
    }

    true
}

fn node_kind_has_compatible_pin_for_dropped_wire(
    candidate: &VisualScriptNodeKind,
    src_pins: &AnyPins<'_>,
    snarl: &Snarl<VisualScriptNodeKind>,
    variables: &[VisualVariableDefinition],
) -> bool {
    match src_pins {
        AnyPins::Out(outputs) => {
            if candidate.input_count() == 0 {
                return false;
            }
            for output in *outputs {
                let Some(from_node) = snarl.get_node(output.node) else {
                    continue;
                };
                let Some(from_slot) = from_node.output_slot(output.output) else {
                    continue;
                };
                for input_index in 0..candidate.input_count() {
                    let Some(to_slot) = candidate.input_slot(input_index) else {
                        continue;
                    };
                    if node_slots_are_compatible(
                        from_node, from_slot, candidate, to_slot, variables,
                    ) {
                        return true;
                    }
                }
            }
            false
        }
        AnyPins::In(inputs) => {
            if candidate.output_count() == 0 {
                return false;
            }
            for input in *inputs {
                let Some(to_node) = snarl.get_node(input.node) else {
                    continue;
                };
                let Some(to_slot) = to_node.input_slot(input.input) else {
                    continue;
                };
                for output_index in 0..candidate.output_count() {
                    let Some(from_slot) = candidate.output_slot(output_index) else {
                        continue;
                    };
                    if node_slots_are_compatible(candidate, from_slot, to_node, to_slot, variables)
                    {
                        return true;
                    }
                }
            }
            false
        }
    }
}

fn runtime_variable_key(variable_id: u64, name: &str) -> String {
    let trimmed = name.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    if variable_id == 0 {
        String::new()
    } else {
        format!("var_{}", variable_id)
    }
}

fn function_call_counter_key() -> String {
    "__visual_fn_call_counter".to_string()
}

fn function_active_call_key(function_id: u64) -> String {
    format!("__visual_fn_active_{}", function_id)
}

fn function_input_value_key(function_id: u64, token: u64, index: usize) -> String {
    format!("__visual_fn_input_{}_{}_{}", function_id, token, index)
}

fn function_output_value_key(function_id: u64, token: u64, index: usize) -> String {
    format!("__visual_fn_output_{}_{}_{}", function_id, token, index)
}

fn json_to_u64(value: &JsonValue) -> Option<u64> {
    match value {
        JsonValue::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|value| u64::try_from(value).ok())),
        JsonValue::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

fn sprite_section_patch_value(value: &JsonValue, primary_key: &str, alias_key: &str) -> JsonValue {
    value
        .as_object()
        .and_then(|object| {
            object
                .get(primary_key)
                .cloned()
                .or_else(|| object.get(alias_key).cloned())
        })
        .unwrap_or_else(|| value.clone())
}

fn next_visual_variable_id(variables: &[VisualVariableDefinition]) -> u64 {
    variables
        .iter()
        .map(|variable| variable.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn find_variable_definition<'a>(
    variables: &'a [VisualVariableDefinition],
    variable_id: u64,
    name: &str,
) -> Option<&'a VisualVariableDefinition> {
    if variable_id != 0 {
        if let Some(found) = variables.iter().find(|var| var.id == variable_id) {
            return Some(found);
        }
    }
    let trimmed = name.trim();
    if trimmed.is_empty() {
        None
    } else {
        variables.iter().find(|var| var.name == trimmed)
    }
}

fn find_function_definition<'a>(
    functions: &'a [VisualScriptFunctionDefinition],
    function_id: u64,
    name: &str,
) -> Option<&'a VisualScriptFunctionDefinition> {
    if function_id != 0 {
        if let Some(found) = functions.iter().find(|function| function.id == function_id) {
            return Some(found);
        }
    }
    let trimmed = name.trim();
    if trimmed.is_empty() {
        None
    } else {
        functions.iter().find(|function| function.name == trimmed)
    }
}

fn node_data_input_type(
    node: &VisualScriptNodeKind,
    input_index: usize,
    variables: &[VisualVariableDefinition],
) -> Option<VisualValueType> {
    match node {
        VisualScriptNodeKind::Branch { .. }
        | VisualScriptNodeKind::LoopWhile { .. }
        | VisualScriptNodeKind::Not => Some(VisualValueType::Bool),
        VisualScriptNodeKind::Log { .. } => Some(VisualValueType::Any),
        VisualScriptNodeKind::SetVariable {
            variable_id, name, ..
        } => find_variable_definition(variables, *variable_id, name)
            .map(|var| var.value_type)
            .or(Some(VisualValueType::Any)),
        VisualScriptNodeKind::CallApi { operation, .. }
        | VisualScriptNodeKind::QueryApi { operation, .. } => operation
            .spec()
            .inputs
            .get(input_index)
            .map(|pin| pin.value_type),
        VisualScriptNodeKind::CallFunction { inputs, .. } => {
            inputs.get(input_index).map(|port| port.value_type)
        }
        VisualScriptNodeKind::FunctionReturn { outputs, .. } => {
            outputs.get(input_index).map(|port| port.value_type)
        }
        VisualScriptNodeKind::WaitSeconds { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::MathBinary { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::MathTrig { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::MathInterpolation { op } => Some(match op {
            VisualInterpolationOp::Lerp
            | VisualInterpolationOp::SmoothStep
            | VisualInterpolationOp::InverseLerp => VisualValueType::Number,
            VisualInterpolationOp::Vec3Lerp => {
                if input_index < 2 {
                    VisualValueType::Vec3
                } else {
                    VisualValueType::Number
                }
            }
            VisualInterpolationOp::QuatSlerp => {
                if input_index < 2 {
                    VisualValueType::Quat
                } else {
                    VisualValueType::Number
                }
            }
        }),
        VisualScriptNodeKind::MathVector { .. } => Some(VisualValueType::Vec3),
        VisualScriptNodeKind::MathProcedural { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::MathUtility { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::Compare { .. } => Some(VisualValueType::Any),
        VisualScriptNodeKind::LogicalBinary { .. } => Some(VisualValueType::Bool),
        VisualScriptNodeKind::TimeSince { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::Select { value_type } => {
            if input_index == 0 {
                Some(VisualValueType::Bool)
            } else {
                Some(*value_type)
            }
        }
        VisualScriptNodeKind::Vec2GetComponent { .. } => Some(VisualValueType::Vec2),
        VisualScriptNodeKind::Vec2SetComponent { .. } => {
            if input_index == 0 {
                Some(VisualValueType::Vec2)
            } else {
                Some(VisualValueType::Number)
            }
        }
        VisualScriptNodeKind::Vec3GetComponent { .. } => Some(VisualValueType::Vec3),
        VisualScriptNodeKind::Vec3SetComponent { .. } => {
            if input_index == 0 {
                Some(VisualValueType::Vec3)
            } else {
                Some(VisualValueType::Number)
            }
        }
        VisualScriptNodeKind::QuatGetComponent { .. } => Some(VisualValueType::Quat),
        VisualScriptNodeKind::QuatSetComponent { .. } => {
            if input_index == 0 {
                Some(VisualValueType::Quat)
            } else {
                Some(VisualValueType::Number)
            }
        }
        VisualScriptNodeKind::TransformGetComponent { .. } => Some(VisualValueType::Transform),
        VisualScriptNodeKind::TransformSetComponent { component } => {
            if input_index == 0 {
                Some(VisualValueType::Transform)
            } else {
                Some(match component {
                    VisualTransformComponent::Position => VisualValueType::Vec3,
                    VisualTransformComponent::Rotation => VisualValueType::Quat,
                    VisualTransformComponent::Scale => VisualValueType::Vec3,
                })
            }
        }
        VisualScriptNodeKind::PhysicsVelocityGetComponent { .. } => {
            Some(VisualValueType::PhysicsVelocity)
        }
        VisualScriptNodeKind::PhysicsVelocitySetComponent { component } => {
            if input_index == 0 {
                Some(VisualValueType::PhysicsVelocity)
            } else {
                Some(match component {
                    VisualPhysicsVelocityComponent::Linear
                    | VisualPhysicsVelocityComponent::Angular => VisualValueType::Vec3,
                    VisualPhysicsVelocityComponent::WakeUp => VisualValueType::Bool,
                })
            }
        }
        VisualScriptNodeKind::Vec2 => Some(VisualValueType::Number),
        VisualScriptNodeKind::Vec3 => Some(VisualValueType::Number),
        VisualScriptNodeKind::Quat => Some(VisualValueType::Number),
        VisualScriptNodeKind::Transform => match input_index {
            0 => Some(VisualValueType::Vec3),
            1 => Some(VisualValueType::Quat),
            _ => Some(VisualValueType::Vec3),
        },
        VisualScriptNodeKind::PhysicsVelocity => match input_index {
            0 | 1 => Some(VisualValueType::Vec3),
            _ => Some(VisualValueType::Bool),
        },
        VisualScriptNodeKind::ArrayLength { .. }
        | VisualScriptNodeKind::ArrayGet { .. }
        | VisualScriptNodeKind::ArraySet { .. }
        | VisualScriptNodeKind::ArrayPush { .. }
        | VisualScriptNodeKind::ArrayRemoveAt { .. }
        | VisualScriptNodeKind::ArrayClear { .. } => Some(match node {
            VisualScriptNodeKind::ArrayLength { .. } | VisualScriptNodeKind::ArrayClear { .. } => {
                VisualValueType::Array
            }
            VisualScriptNodeKind::ArrayGet { item_type } => match input_index {
                0 => VisualValueType::Array,
                1 => VisualValueType::Number,
                _ => *item_type,
            },
            VisualScriptNodeKind::ArraySet { item_type } => match input_index {
                0 => VisualValueType::Array,
                1 => VisualValueType::Number,
                _ => *item_type,
            },
            VisualScriptNodeKind::ArrayPush { item_type } => {
                if input_index == 0 {
                    VisualValueType::Array
                } else {
                    *item_type
                }
            }
            VisualScriptNodeKind::ArrayRemoveAt { .. } => {
                if input_index == 0 {
                    VisualValueType::Array
                } else {
                    VisualValueType::Number
                }
            }
            _ => VisualValueType::Any,
        }),
        VisualScriptNodeKind::RayCast => Some(match input_index {
            0 | 1 => VisualValueType::Vec3,
            2 => VisualValueType::Number,
            3 => VisualValueType::Bool,
            4 => VisualValueType::PhysicsQueryFilter,
            _ => VisualValueType::Entity,
        }),
        _ => None,
    }
}

fn node_data_output_type(
    node: &VisualScriptNodeKind,
    output_index: usize,
    variables: &[VisualVariableDefinition],
) -> Option<VisualValueType> {
    if output_index != 0 {
        return match node {
            VisualScriptNodeKind::CallFunction { outputs, .. } => {
                outputs.get(output_index).map(|port| port.value_type)
            }
            VisualScriptNodeKind::FunctionStart { inputs, .. } => {
                inputs.get(output_index).map(|port| port.value_type)
            }
            _ => None,
        };
    }

    match node {
        VisualScriptNodeKind::GetVariable {
            variable_id, name, ..
        } => find_variable_definition(variables, *variable_id, name)
            .map(|var| var.value_type)
            .or(Some(VisualValueType::Any)),
        VisualScriptNodeKind::CallApi { operation, .. }
        | VisualScriptNodeKind::QueryApi { operation, .. } => {
            if output_index == 0 {
                operation.spec().output_type.or(Some(VisualValueType::Any))
            } else {
                None
            }
        }
        VisualScriptNodeKind::CallFunction { outputs, .. } => {
            outputs.first().map(|port| port.value_type)
        }
        VisualScriptNodeKind::FunctionStart { inputs, .. } => {
            inputs.first().map(|port| port.value_type)
        }
        VisualScriptNodeKind::BoolLiteral { .. }
        | VisualScriptNodeKind::Compare { .. }
        | VisualScriptNodeKind::LogicalBinary { .. }
        | VisualScriptNodeKind::Not => Some(VisualValueType::Bool),
        VisualScriptNodeKind::NumberLiteral { .. }
        | VisualScriptNodeKind::DeltaTime
        | VisualScriptNodeKind::TimeSinceStart
        | VisualScriptNodeKind::UnixTimeSeconds
        | VisualScriptNodeKind::TimeSince { .. }
        | VisualScriptNodeKind::MathBinary { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::MathTrig { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::MathInterpolation { op } => Some(match op {
            VisualInterpolationOp::Lerp
            | VisualInterpolationOp::SmoothStep
            | VisualInterpolationOp::InverseLerp => VisualValueType::Number,
            VisualInterpolationOp::Vec3Lerp => VisualValueType::Vec3,
            VisualInterpolationOp::QuatSlerp => VisualValueType::Quat,
        }),
        VisualScriptNodeKind::MathVector { op } => Some(match op {
            VisualVectorMathOp::Cross | VisualVectorMathOp::Normalize => VisualValueType::Vec3,
            VisualVectorMathOp::Dot | VisualVectorMathOp::Length | VisualVectorMathOp::Distance => {
                VisualValueType::Number
            }
        }),
        VisualScriptNodeKind::MathProcedural { .. } | VisualScriptNodeKind::MathUtility { .. } => {
            Some(VisualValueType::Number)
        }
        VisualScriptNodeKind::StringLiteral { .. } => Some(VisualValueType::String),
        VisualScriptNodeKind::AnyLiteral { .. } => Some(VisualValueType::Any),
        VisualScriptNodeKind::PhysicsQueryFilterLiteral { .. } => {
            Some(VisualValueType::PhysicsQueryFilter)
        }
        VisualScriptNodeKind::SelfEntity => Some(VisualValueType::Entity),
        VisualScriptNodeKind::Select { value_type } => Some(*value_type),
        VisualScriptNodeKind::Vec2GetComponent { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::Vec2SetComponent { .. } => Some(VisualValueType::Vec2),
        VisualScriptNodeKind::Vec3GetComponent { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::Vec3SetComponent { .. } => Some(VisualValueType::Vec3),
        VisualScriptNodeKind::QuatGetComponent { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::QuatSetComponent { .. } => Some(VisualValueType::Quat),
        VisualScriptNodeKind::TransformGetComponent { component } => Some(match component {
            VisualTransformComponent::Position => VisualValueType::Vec3,
            VisualTransformComponent::Rotation => VisualValueType::Quat,
            VisualTransformComponent::Scale => VisualValueType::Vec3,
        }),
        VisualScriptNodeKind::TransformSetComponent { .. } => Some(VisualValueType::Transform),
        VisualScriptNodeKind::PhysicsVelocityGetComponent { component } => Some(match component {
            VisualPhysicsVelocityComponent::Linear | VisualPhysicsVelocityComponent::Angular => {
                VisualValueType::Vec3
            }
            VisualPhysicsVelocityComponent::WakeUp => VisualValueType::Bool,
        }),
        VisualScriptNodeKind::PhysicsVelocitySetComponent { .. } => {
            Some(VisualValueType::PhysicsVelocity)
        }
        VisualScriptNodeKind::Vec2 => Some(VisualValueType::Vec2),
        VisualScriptNodeKind::Vec3 => Some(VisualValueType::Vec3),
        VisualScriptNodeKind::Quat => Some(VisualValueType::Quat),
        VisualScriptNodeKind::Transform => Some(VisualValueType::Transform),
        VisualScriptNodeKind::PhysicsVelocity => Some(VisualValueType::PhysicsVelocity),
        VisualScriptNodeKind::ArrayEmpty { .. }
        | VisualScriptNodeKind::ArraySet { .. }
        | VisualScriptNodeKind::ArrayPush { .. }
        | VisualScriptNodeKind::ArrayRemoveAt { .. }
        | VisualScriptNodeKind::ArrayClear { .. } => Some(VisualValueType::Array),
        VisualScriptNodeKind::ArrayLength { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::ArrayGet { item_type } => Some(*item_type),
        VisualScriptNodeKind::RayCast => Some(VisualValueType::PhysicsRayCastHit),
        VisualScriptNodeKind::FunctionReturn { .. } => None,
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualScriptEvent {
    Start,
    Update,
    Stop,
    CollisionEnter,
    CollisionStay,
    CollisionExit,
    TriggerEnter,
    TriggerExit,
    InputActionPressed(String),
    InputActionReleased(String),
    InputActionDown(String),
    CustomEvent(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VisualWaitTimerKey {
    pub scope: String,
    pub node_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VisualPendingCallKey {
    pub scope: String,
    pub node_id: u64,
}

#[derive(Debug, Clone)]
pub struct VisualPendingFunctionCall {
    pub function_id: u64,
    pub function_name: String,
    pub call_token: u64,
    pub output_defs: Vec<VisualFunctionIoDefinition>,
}

#[derive(Debug, Default, Clone)]
pub struct VisualScriptRuntimeState {
    pub variables: HashMap<String, JsonValue>,
    pub wait_timers: HashMap<VisualWaitTimerKey, f32>,
    pub pending_function_calls: HashMap<VisualPendingCallKey, VisualPendingFunctionCall>,
    pub elapsed_seconds: f64,
}

pub trait VisualScriptHost {
    fn invoke_api(
        &mut self,
        table: VisualScriptApiTable,
        function: &str,
        args: &[JsonValue],
    ) -> Result<JsonValue, String>;

    fn log(&mut self, message: &str);
}

#[derive(Debug, Clone)]
struct VisualCompiledFunctionProgram {
    id: u64,
    name: String,
    source_path: String,
    inputs: Vec<VisualFunctionIoDefinition>,
    outputs: Vec<VisualFunctionIoDefinition>,
    program: Box<VisualScriptProgram>,
}

#[derive(Debug, Clone)]
pub struct VisualScriptProgram {
    source_label: String,
    source_name: String,
    nodes: HashMap<u64, VisualScriptNodeKind>,
    exec_edges: HashMap<(u64, usize), Vec<u64>>,
    data_edges: HashMap<(u64, usize), (u64, usize)>,
    variable_defaults: HashMap<String, JsonValue>,
    variable_types: HashMap<String, VisualValueType>,
    variable_array_item_types: HashMap<String, Option<VisualValueType>>,
    on_start_nodes: Vec<u64>,
    on_update_nodes: Vec<u64>,
    on_stop_nodes: Vec<u64>,
    on_collision_enter_nodes: Vec<u64>,
    on_collision_stay_nodes: Vec<u64>,
    on_collision_exit_nodes: Vec<u64>,
    on_trigger_enter_nodes: Vec<u64>,
    on_trigger_exit_nodes: Vec<u64>,
    on_input_action_pressed_nodes: Vec<u64>,
    on_input_action_released_nodes: Vec<u64>,
    on_input_action_down_nodes: Vec<u64>,
    on_custom_event_nodes: Vec<u64>,
    functions: HashMap<u64, VisualCompiledFunctionProgram>,
    function_context: Option<(u64, String)>,
}

impl VisualScriptProgram {
    fn filter_input_action_roots(&self, candidates: &[u64], action: &str) -> Vec<u64> {
        let action = normalize_event_name(action);
        if action.is_empty() {
            return Vec::new();
        }

        candidates
            .iter()
            .copied()
            .filter(|node_id| {
                matches!(
                    self.nodes.get(node_id),
                    Some(VisualScriptNodeKind::OnInputAction {
                        action: node_action,
                        ..
                    }) if node_action == &action
                )
            })
            .collect()
    }

    fn filter_custom_event_roots(&self, candidates: &[u64], name: &str) -> Vec<u64> {
        let name = normalize_event_name(name);
        if name.is_empty() {
            return Vec::new();
        }

        candidates
            .iter()
            .copied()
            .filter(|node_id| {
                matches!(
                    self.nodes.get(node_id),
                    Some(VisualScriptNodeKind::OnCustomEvent { name: event_name }) if event_name == &name
                )
            })
            .collect()
    }

    pub fn describe(&self) -> String {
        let mut api_calls = 0usize;
        let mut query_calls = 0usize;
        let mut variables = 0usize;
        let mut call_functions = 0usize;

        for node in self.nodes.values() {
            match node {
                VisualScriptNodeKind::CallApi { .. } => api_calls += 1,
                VisualScriptNodeKind::QueryApi { .. } => query_calls += 1,
                VisualScriptNodeKind::CallFunction { .. } => call_functions += 1,
                VisualScriptNodeKind::SetVariable { .. }
                | VisualScriptNodeKind::GetVariable { .. }
                | VisualScriptNodeKind::ClearVariable { .. } => variables += 1,
                _ => {}
            }
        }

        let function_label = self
            .function_context
            .as_ref()
            .map(|(_, name)| format!(" (function: {})", name))
            .unwrap_or_default();

        format!(
            "Runtime Plan\nsource: {}{}\nnodes: {}\nexec edges: {}\ndata edges: {}\nvariable defs: {}\non_start: {}\non_update: {}\non_stop: {}\non_collision_enter: {}\non_collision_stay: {}\non_collision_exit: {}\non_trigger_enter: {}\non_trigger_exit: {}\non_input_action_pressed: {}\non_input_action_released: {}\non_input_action_down: {}\non_custom_event: {}\napi call nodes: {}\napi query nodes: {}\nfunction call nodes: {}\nvariable nodes: {}\nfunctions: {}\nstep budget/event: {}",
            self.source_label,
            function_label,
            self.nodes.len(),
            self.exec_edges
                .values()
                .map(|targets| targets.len())
                .sum::<usize>(),
            self.data_edges.len(),
            self.variable_defaults.len(),
            self.on_start_nodes.len(),
            self.on_update_nodes.len(),
            self.on_stop_nodes.len(),
            self.on_collision_enter_nodes.len(),
            self.on_collision_stay_nodes.len(),
            self.on_collision_exit_nodes.len(),
            self.on_trigger_enter_nodes.len(),
            self.on_trigger_exit_nodes.len(),
            self.on_input_action_pressed_nodes.len(),
            self.on_input_action_released_nodes.len(),
            self.on_input_action_down_nodes.len(),
            self.on_custom_event_nodes.len(),
            api_calls,
            query_calls,
            call_functions,
            variables,
            self.functions.len(),
            MAX_EXEC_STEPS_PER_EVENT,
        )
    }

    pub fn execute_event<H: VisualScriptHost>(
        &self,
        event: VisualScriptEvent,
        owner_entity_id: u64,
        dt: f32,
        state: &mut VisualScriptRuntimeState,
        host: &mut H,
    ) -> Result<(), String> {
        self.execute_event_internal(event, owner_entity_id, dt, state, host, &self.functions, 0)
    }

    fn execute_event_internal<H: VisualScriptHost>(
        &self,
        event: VisualScriptEvent,
        owner_entity_id: u64,
        dt: f32,
        state: &mut VisualScriptRuntimeState,
        host: &mut H,
        function_registry: &HashMap<u64, VisualCompiledFunctionProgram>,
        call_depth: usize,
    ) -> Result<(), String> {
        for (variable, default_value) in &self.variable_defaults {
            state
                .variables
                .entry(variable.clone())
                .or_insert_with(|| default_value.clone());
        }

        if matches!(&event, VisualScriptEvent::Update) {
            state.elapsed_seconds += f64::from(dt.max(0.0));
        }

        let roots: Cow<'_, [u64]> = match &event {
            VisualScriptEvent::Start => Cow::Borrowed(&self.on_start_nodes),
            VisualScriptEvent::Update => Cow::Borrowed(&self.on_update_nodes),
            VisualScriptEvent::Stop => Cow::Borrowed(&self.on_stop_nodes),
            VisualScriptEvent::CollisionEnter => Cow::Borrowed(&self.on_collision_enter_nodes),
            VisualScriptEvent::CollisionStay => Cow::Borrowed(&self.on_collision_stay_nodes),
            VisualScriptEvent::CollisionExit => Cow::Borrowed(&self.on_collision_exit_nodes),
            VisualScriptEvent::TriggerEnter => Cow::Borrowed(&self.on_trigger_enter_nodes),
            VisualScriptEvent::TriggerExit => Cow::Borrowed(&self.on_trigger_exit_nodes),
            VisualScriptEvent::InputActionPressed(action) => Cow::Owned(
                self.filter_input_action_roots(&self.on_input_action_pressed_nodes, action),
            ),
            VisualScriptEvent::InputActionReleased(action) => Cow::Owned(
                self.filter_input_action_roots(&self.on_input_action_released_nodes, action),
            ),
            VisualScriptEvent::InputActionDown(action) => {
                Cow::Owned(self.filter_input_action_roots(&self.on_input_action_down_nodes, action))
            }
            VisualScriptEvent::CustomEvent(name) => {
                Cow::Owned(self.filter_custom_event_roots(&self.on_custom_event_nodes, name))
            }
        };

        let mut context = VisualRuntimeContext {
            program: self,
            function_registry,
            state,
            host,
            owner_entity_id,
            dt,
            node_results: HashMap::new(),
            data_cache: HashMap::new(),
            steps_left: MAX_EXEC_STEPS_PER_EVENT,
            legacy_statement_warnings: HashSet::new(),
            call_depth,
        };

        if matches!(&event, VisualScriptEvent::Update) {
            context.resume_wait_timers()?;
            context.resume_pending_function_calls()?;
        }

        for root in roots.iter().copied() {
            context.execute_exec_targets(root, 0)?;
        }

        Ok(())
    }
}

struct VisualRuntimeContext<'a, H: VisualScriptHost> {
    program: &'a VisualScriptProgram,
    function_registry: &'a HashMap<u64, VisualCompiledFunctionProgram>,
    state: &'a mut VisualScriptRuntimeState,
    host: &'a mut H,
    owner_entity_id: u64,
    dt: f32,
    node_results: HashMap<u64, JsonValue>,
    data_cache: HashMap<(u64, usize), JsonValue>,
    steps_left: u32,
    legacy_statement_warnings: HashSet<u64>,
    call_depth: usize,
}

enum FunctionInvokeResult {
    Completed(Vec<JsonValue>),
    Pending(VisualPendingFunctionCall),
}

impl<'a, H: VisualScriptHost> VisualRuntimeContext<'a, H> {
    fn log_runtime(&mut self, message: &str) {
        let tagged = format!("[{}] {}", self.program.source_name, message);
        self.host.log(&tagged);
    }

    fn consume_step(&mut self, node_id: u64) -> Result<(), String> {
        if self.steps_left == 0 {
            return Err(format!(
                "Visual script '{}' exceeded execution step budget while executing node {}",
                self.program.source_label, node_id
            ));
        }
        self.steps_left -= 1;
        Ok(())
    }

    fn runtime_scope_key(&self) -> Result<String, String> {
        if let Some((function_id, _)) = self.program.function_context {
            let token = self
                .active_function_call_token(function_id)
                .ok_or_else(|| {
                    format!(
                        "Function runtime context for id {} has no active call token",
                        function_id
                    )
                })?;
            Ok(format!("function:{}:{}", function_id, token))
        } else {
            Ok(format!("root:{}", self.program.source_label))
        }
    }

    fn wait_timer_key(&self, node_id: u64) -> Result<VisualWaitTimerKey, String> {
        Ok(VisualWaitTimerKey {
            scope: self.runtime_scope_key()?,
            node_id,
        })
    }

    fn pending_call_key(&self, node_id: u64) -> Result<VisualPendingCallKey, String> {
        Ok(VisualPendingCallKey {
            scope: self.runtime_scope_key()?,
            node_id,
        })
    }

    fn resume_wait_timers(&mut self) -> Result<(), String> {
        if self.state.wait_timers.is_empty() {
            return Ok(());
        }

        let scope = self.runtime_scope_key()?;
        let dt = self.dt.max(0.0);
        let mut ready = Vec::<VisualWaitTimerKey>::new();
        for (key, remaining) in self.state.wait_timers.iter_mut() {
            if key.scope != scope {
                continue;
            }
            *remaining -= dt;
            if *remaining <= 0.0 {
                ready.push(key.clone());
            }
        }
        if ready.is_empty() {
            return Ok(());
        }

        ready.sort_by_key(|key| key.node_id);
        for key in &ready {
            self.state.wait_timers.remove(key);
        }

        for key in ready {
            self.execute_exec_targets(key.node_id, 0)?;
        }

        Ok(())
    }

    fn resume_pending_function_calls(&mut self) -> Result<(), String> {
        if self.state.pending_function_calls.is_empty() {
            return Ok(());
        }

        let scope = self.runtime_scope_key()?;
        let mut keys = self
            .state
            .pending_function_calls
            .keys()
            .filter(|key| key.scope == scope)
            .cloned()
            .collect::<Vec<_>>();
        if keys.is_empty() {
            return Ok(());
        }
        keys.sort_by_key(|key| key.node_id);

        for key in keys {
            let Some(pending) = self.state.pending_function_calls.get(&key).cloned() else {
                continue;
            };
            if let Some(values) = self.resume_function_call(&pending)? {
                self.state.pending_function_calls.remove(&key);
                for (index, value) in values.iter().enumerate() {
                    self.data_cache.insert((key.node_id, index), value.clone());
                }
                self.node_results
                    .insert(key.node_id, JsonValue::Array(values.clone()));
                self.execute_exec_targets(key.node_id, 0)?;
            }
        }

        Ok(())
    }

    fn execute_exec_targets(&mut self, node_id: u64, output_index: usize) -> Result<(), String> {
        let mut index = 0usize;
        while let Some(target) = self
            .program
            .exec_edges
            .get(&(node_id, output_index))
            .and_then(|targets| targets.get(index))
            .copied()
        {
            index += 1;
            self.execute_node(target)?;
        }

        Ok(())
    }

    fn execute_node(&mut self, node_id: u64) -> Result<(), String> {
        self.consume_step(node_id)?;

        let Some(node) = self.program.nodes.get(&node_id) else {
            return Err(format!("Visual script runtime missing node {}", node_id));
        };

        match node {
            VisualScriptNodeKind::OnStart
            | VisualScriptNodeKind::OnUpdate
            | VisualScriptNodeKind::OnStop
            | VisualScriptNodeKind::OnCollisionEnter
            | VisualScriptNodeKind::OnCollisionStay
            | VisualScriptNodeKind::OnCollisionExit
            | VisualScriptNodeKind::OnTriggerEnter
            | VisualScriptNodeKind::OnTriggerExit
            | VisualScriptNodeKind::OnInputAction { .. }
            | VisualScriptNodeKind::OnCustomEvent { .. } => {
                self.execute_exec_targets(node_id, 0)?;
            }
            VisualScriptNodeKind::Sequence { outputs } => {
                for output in 0..usize::from((*outputs).clamp(1, 8)) {
                    self.execute_exec_targets(node_id, output)?;
                }
            }
            VisualScriptNodeKind::Branch { condition } => {
                let result = self.resolve_bool_input(node_id, 0, Some(condition.as_str()))?;
                let output = if result { 0 } else { 1 };
                self.execute_exec_targets(node_id, output)?;
            }
            VisualScriptNodeKind::LoopWhile {
                condition,
                max_iterations,
            } => {
                let max_iterations = (*max_iterations).clamp(1, MAX_LOOP_ITERATIONS);
                let mut iterations = 0u32;

                while self.resolve_bool_input(node_id, 0, Some(condition.as_str()))? {
                    iterations = iterations.saturating_add(1);
                    if iterations > max_iterations {
                        return Err(format!(
                            "Loop While node {} exceeded max iterations ({})",
                            node_id, max_iterations
                        ));
                    }
                    self.execute_exec_targets(node_id, 0)?;
                }

                self.execute_exec_targets(node_id, 1)?;
            }
            VisualScriptNodeKind::Log { message } => {
                let mut stack = HashSet::new();
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some(message.as_str()),
                    &mut stack,
                )?;
                self.log_runtime(&json_to_log_string(&value));
                self.execute_exec_targets(node_id, 0)?;
            }
            VisualScriptNodeKind::SetVariable {
                variable_id,
                name,
                value,
            } => {
                let variable = runtime_variable_key(*variable_id, name);
                if variable.is_empty() {
                    return Err(format!(
                        "Set Variable node {} has empty variable name",
                        node_id
                    ));
                }

                let mut stack = HashSet::new();
                let resolved = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some(value.as_str()),
                    &mut stack,
                )?;
                let resolved = if let Some(value_type) = self.program.variable_types.get(&variable)
                {
                    let array_item_type = self
                        .program
                        .variable_array_item_types
                        .get(&variable)
                        .copied()
                        .flatten();
                    coerce_json_to_visual_type_with_array_item(
                        &resolved,
                        *value_type,
                        array_item_type,
                    )?
                } else {
                    resolved
                };

                self.state.variables.insert(variable, resolved.clone());
                self.node_results.insert(node_id, resolved);
                self.execute_exec_targets(node_id, 0)?;
            }
            VisualScriptNodeKind::ClearVariable { variable_id, name } => {
                let variable = runtime_variable_key(*variable_id, name);
                if !variable.is_empty() {
                    self.state.variables.remove(&variable);
                }
                self.execute_exec_targets(node_id, 0)?;
            }
            VisualScriptNodeKind::WaitSeconds {
                seconds,
                restart_on_retrigger,
            } => {
                let mut stack = HashSet::new();
                let wait_seconds = self
                    .resolve_number_input_with_stack(
                        node_id,
                        0,
                        Some(seconds.as_str()),
                        &mut stack,
                    )?
                    .max(0.0) as f32;
                let timer_key = self.wait_timer_key(node_id)?;
                if wait_seconds <= f32::EPSILON {
                    self.state.wait_timers.remove(&timer_key);
                    self.execute_exec_targets(node_id, 0)?;
                } else if let Some(existing) = self.state.wait_timers.get_mut(&timer_key) {
                    if *restart_on_retrigger {
                        *existing = wait_seconds;
                    }
                } else {
                    self.state.wait_timers.insert(timer_key, wait_seconds);
                }
            }
            VisualScriptNodeKind::CallApi {
                operation, args, ..
            } => {
                let spec = operation.spec();
                let call_args = self.collect_call_args(node_id, *operation, args, spec.inputs)?;
                let result = self
                    .host
                    .invoke_api(spec.table, spec.function, &call_args)
                    .map_err(|err| {
                        format!(
                            "{}.{} failed at node {}: {}",
                            spec.table.as_str(),
                            spec.function,
                            node_id,
                            err
                        )
                    })?;
                let result = if let Some(value_type) = spec.output_type {
                    coerce_json_to_visual_type(&result, value_type)?
                } else {
                    result
                };

                self.node_results.insert(node_id, result);
                self.execute_exec_targets(node_id, 0)?;
            }
            VisualScriptNodeKind::FunctionStart { .. } => {
                self.execute_exec_targets(node_id, 0)?;
            }
            VisualScriptNodeKind::FunctionReturn {
                function_id,
                outputs,
                values,
            } => {
                let Some(call_token) = self.active_function_call_token(*function_id) else {
                    return Err(format!(
                        "Function Return node {} executed without active call context",
                        node_id
                    ));
                };

                for (index, output) in outputs.iter().enumerate() {
                    let fallback = values
                        .get(index)
                        .map(|value| value.as_str())
                        .or(Some(output.default_value.as_str()));
                    let mut stack = HashSet::new();
                    let resolved =
                        self.resolve_data_input_with_stack(node_id, index, fallback, &mut stack)?;
                    let coerced = coerce_json_to_visual_type_with_array_item(
                        &resolved,
                        output.value_type,
                        output.array_item_type,
                    )?;
                    let key = function_output_value_key(*function_id, call_token, index);
                    self.state.variables.insert(key, coerced);
                }
            }
            VisualScriptNodeKind::CallFunction {
                function_id,
                name,
                inputs,
                outputs,
                args,
            } => {
                let pending_key = self.pending_call_key(node_id)?;
                if let Some(pending) = self.state.pending_function_calls.get(&pending_key).cloned()
                {
                    if let Some(values) = self.resume_function_call(&pending)? {
                        self.state.pending_function_calls.remove(&pending_key);
                        for (index, value) in values.iter().enumerate() {
                            self.data_cache.insert((node_id, index), value.clone());
                        }
                        self.node_results
                            .insert(node_id, JsonValue::Array(values.clone()));
                        self.execute_exec_targets(node_id, 0)?;
                    }
                } else {
                    let call_args = self.collect_function_call_args(node_id, inputs, args)?;
                    match self.invoke_function(
                        *function_id,
                        name.as_str(),
                        inputs,
                        outputs,
                        &call_args,
                    )? {
                        FunctionInvokeResult::Completed(values) => {
                            for (index, value) in values.iter().enumerate() {
                                self.data_cache.insert((node_id, index), value.clone());
                            }
                            self.node_results
                                .insert(node_id, JsonValue::Array(values.clone()));
                            self.execute_exec_targets(node_id, 0)?;
                        }
                        FunctionInvokeResult::Pending(pending) => {
                            self.state
                                .pending_function_calls
                                .insert(pending_key, pending);
                        }
                    }
                }
            }
            VisualScriptNodeKind::Comment { .. } => {
                self.execute_exec_targets(node_id, 0)?;
            }
            VisualScriptNodeKind::Statement { .. } => {
                if self.legacy_statement_warnings.insert(node_id) {
                    self.log_runtime(
                        "warning: Legacy Statement nodes no longer execute script text. Replace with structured visual nodes.",
                    );
                }
                self.execute_exec_targets(node_id, 0)?;
            }
            VisualScriptNodeKind::GetVariable { .. }
            | VisualScriptNodeKind::QueryApi { .. }
            | VisualScriptNodeKind::BoolLiteral { .. }
            | VisualScriptNodeKind::NumberLiteral { .. }
            | VisualScriptNodeKind::StringLiteral { .. }
            | VisualScriptNodeKind::AnyLiteral { .. }
            | VisualScriptNodeKind::PhysicsQueryFilterLiteral { .. }
            | VisualScriptNodeKind::SelfEntity
            | VisualScriptNodeKind::DeltaTime
            | VisualScriptNodeKind::TimeSinceStart
            | VisualScriptNodeKind::UnixTimeSeconds
            | VisualScriptNodeKind::TimeSince { .. }
            | VisualScriptNodeKind::MathBinary { .. }
            | VisualScriptNodeKind::MathTrig { .. }
            | VisualScriptNodeKind::MathInterpolation { .. }
            | VisualScriptNodeKind::MathVector { .. }
            | VisualScriptNodeKind::MathProcedural { .. }
            | VisualScriptNodeKind::MathUtility { .. }
            | VisualScriptNodeKind::Compare { .. }
            | VisualScriptNodeKind::LogicalBinary { .. }
            | VisualScriptNodeKind::Not
            | VisualScriptNodeKind::Select { .. }
            | VisualScriptNodeKind::Vec2GetComponent { .. }
            | VisualScriptNodeKind::Vec2SetComponent { .. }
            | VisualScriptNodeKind::Vec3GetComponent { .. }
            | VisualScriptNodeKind::Vec3SetComponent { .. }
            | VisualScriptNodeKind::QuatGetComponent { .. }
            | VisualScriptNodeKind::QuatSetComponent { .. }
            | VisualScriptNodeKind::TransformGetComponent { .. }
            | VisualScriptNodeKind::TransformSetComponent { .. }
            | VisualScriptNodeKind::PhysicsVelocityGetComponent { .. }
            | VisualScriptNodeKind::PhysicsVelocitySetComponent { .. }
            | VisualScriptNodeKind::Vec2
            | VisualScriptNodeKind::Vec3
            | VisualScriptNodeKind::Quat
            | VisualScriptNodeKind::Transform
            | VisualScriptNodeKind::PhysicsVelocity
            | VisualScriptNodeKind::ArrayEmpty { .. }
            | VisualScriptNodeKind::ArrayLength { .. }
            | VisualScriptNodeKind::ArrayGet { .. }
            | VisualScriptNodeKind::ArraySet { .. }
            | VisualScriptNodeKind::ArrayPush { .. }
            | VisualScriptNodeKind::ArrayRemoveAt { .. }
            | VisualScriptNodeKind::ArrayClear { .. }
            | VisualScriptNodeKind::RayCast => {}
        }

        Ok(())
    }

    fn collect_call_args(
        &mut self,
        node_id: u64,
        operation: VisualApiOperation,
        defaults: &[String],
        inputs: &[VisualApiInputSpec],
    ) -> Result<Vec<JsonValue>, String> {
        let mut args = Vec::with_capacity(defaults.len());
        for (index, default) in defaults.iter().enumerate() {
            let mut stack = HashSet::new();
            let value = self.resolve_data_input_with_stack(
                node_id,
                index,
                Some(default.as_str()),
                &mut stack,
            )?;
            let value_type = inputs
                .get(index)
                .map(|input| input.value_type)
                .unwrap_or(VisualValueType::Any);
            let coerced = coerce_json_to_visual_type(&value, value_type).map_err(|err| {
                let argument = api_input_label(operation, index);
                format!(
                    "Argument '{}' at node {} is invalid for {}: {}",
                    argument,
                    node_id,
                    value_type.title(),
                    err
                )
            })?;
            args.push(coerced);
        }
        self.normalize_api_call_args(operation, args)
    }

    fn normalize_api_call_args(
        &self,
        operation: VisualApiOperation,
        mut args: Vec<JsonValue>,
    ) -> Result<Vec<JsonValue>, String> {
        match operation {
            VisualApiOperation::EcsSetLookAt => {
                if args.len() < 6 {
                    return Ok(args);
                }
                let mut patch = JsonMap::new();
                patch.insert(
                    "target_entity".to_string(),
                    match json_to_u64(&args[1]) {
                        Some(id) if id > 0 => JsonValue::Number(JsonNumber::from(id)),
                        _ => JsonValue::Null,
                    },
                );
                patch.insert("target_offset".to_string(), args[2].clone());
                patch.insert("offset_in_target_space".to_string(), args[3].clone());
                patch.insert("up".to_string(), args[4].clone());
                patch.insert("rotation_smooth_time".to_string(), args[5].clone());
                Ok(vec![args.remove(0), JsonValue::Object(patch)])
            }
            VisualApiOperation::EcsSetEntityFollower => {
                if args.len() < 7 {
                    return Ok(args);
                }
                let mut patch = JsonMap::new();
                patch.insert(
                    "target_entity".to_string(),
                    match json_to_u64(&args[1]) {
                        Some(id) if id > 0 => JsonValue::Number(JsonNumber::from(id)),
                        _ => JsonValue::Null,
                    },
                );
                patch.insert("position_offset".to_string(), args[2].clone());
                patch.insert("offset_in_target_space".to_string(), args[3].clone());
                patch.insert("follow_rotation".to_string(), args[4].clone());
                patch.insert("position_smooth_time".to_string(), args[5].clone());
                patch.insert("rotation_smooth_time".to_string(), args[6].clone());
                Ok(vec![args.remove(0), JsonValue::Object(patch)])
            }
            VisualApiOperation::EcsSetAnimatorTransition => {
                if args.len() < 9 {
                    return Ok(args);
                }
                let mut patch = JsonMap::new();
                patch.insert("from".to_string(), args[3].clone());
                patch.insert("to".to_string(), args[4].clone());
                patch.insert("duration".to_string(), args[5].clone());
                patch.insert("can_interrupt".to_string(), args[6].clone());
                let use_exit_time = args[7].as_bool().unwrap_or(false);
                patch.insert(
                    "exit_time".to_string(),
                    if use_exit_time {
                        args[8].clone()
                    } else {
                        JsonValue::Null
                    },
                );
                Ok(vec![
                    args[0].clone(),
                    args[1].clone(),
                    args[2].clone(),
                    JsonValue::Object(patch),
                ])
            }
            VisualApiOperation::EcsSetSpriteRendererSheetAnimation => {
                if args.len() < 2 {
                    return Ok(args);
                }
                let section = sprite_section_patch_value(&args[1], "sheet_animation", "sheet");
                let normalized = coerce_json_to_sprite_sheet_animation_patch(&section)
                    .map_err(|err| format!("Sheet animation patch is invalid: {}", err))?;
                let mut patch = JsonMap::new();
                patch.insert("sheet_animation".to_string(), normalized.clone());
                patch.insert("sheet".to_string(), normalized);
                Ok(vec![args.remove(0), JsonValue::Object(patch)])
            }
            VisualApiOperation::EcsSetSpriteRendererSequence => {
                if args.len() < 2 {
                    return Ok(args);
                }
                let section = sprite_section_patch_value(&args[1], "image_sequence", "sequence");
                let normalized = coerce_json_to_sprite_image_sequence_patch(&section)
                    .map_err(|err| format!("Image sequence patch is invalid: {}", err))?;
                let mut patch = JsonMap::new();
                patch.insert("image_sequence".to_string(), normalized.clone());
                patch.insert("sequence".to_string(), normalized);
                Ok(vec![args.remove(0), JsonValue::Object(patch)])
            }
            VisualApiOperation::EcsSetAnimatorBlendNode => {
                if args.len() >= 5 {
                    let mode = args[4].as_str().unwrap_or("").trim();
                    if mode.is_empty() {
                        args[4] = JsonValue::Null;
                    }
                }
                Ok(args)
            }
            VisualApiOperation::EcsSetAnimatorBlendChild => {
                if args.len() >= 6 {
                    let param = args[5].as_str().unwrap_or("").trim();
                    if param.is_empty() {
                        args[5] = JsonValue::Null;
                    }
                }
                Ok(args)
            }
            VisualApiOperation::InputBindAction => {
                if args.len() >= 4 {
                    let context = args[2].as_str().unwrap_or("").trim();
                    if context.is_empty() {
                        args[2] = JsonValue::Null;
                    }
                }
                Ok(args)
            }
            VisualApiOperation::InputUnbindAction => {
                if args.len() >= 2 {
                    let binding = args[1].as_str().unwrap_or("").trim();
                    if binding.is_empty() {
                        args[1] = JsonValue::Null;
                    }
                }
                if args.len() >= 3 {
                    let context = args[2].as_str().unwrap_or("").trim();
                    if context.is_empty() {
                        args[2] = JsonValue::Null;
                    }
                }
                Ok(args)
            }
            VisualApiOperation::InputSetActionContext => {
                if let Some(context) = args.get(0) {
                    let context = context.as_str().unwrap_or("").trim();
                    if context.is_empty() {
                        args[0] = JsonValue::Null;
                    }
                }
                Ok(args)
            }
            VisualApiOperation::EcsFindScriptIndex => {
                if args.len() >= 3 {
                    let language = args[2].as_str().unwrap_or("").trim();
                    if language.is_empty() {
                        args[2] = JsonValue::Null;
                    }
                }
                Ok(args)
            }
            VisualApiOperation::EcsGetScript
            | VisualApiOperation::EcsListScriptFields
            | VisualApiOperation::EcsGetScriptPath
            | VisualApiOperation::EcsGetScriptLanguage => {
                if args.len() >= 2 {
                    let index = args[1].as_f64().unwrap_or(0.0).round();
                    if index <= 0.0 {
                        args[1] = JsonValue::Null;
                    } else {
                        args[1] = JsonValue::Number(JsonNumber::from(index as u64));
                    }
                }
                Ok(args)
            }
            VisualApiOperation::EcsGetScriptField => {
                if args.len() >= 3 {
                    let index = args[2].as_f64().unwrap_or(0.0).round();
                    if index <= 0.0 {
                        args[2] = JsonValue::Null;
                    } else {
                        args[2] = JsonValue::Number(JsonNumber::from(index as u64));
                    }
                }
                Ok(args)
            }
            VisualApiOperation::EcsSetScript => {
                if args.len() >= 3 {
                    let language = args[2].as_str().unwrap_or("").trim();
                    if language.is_empty() {
                        args[2] = JsonValue::Null;
                    }
                }
                if args.len() >= 4 {
                    let index = args[3].as_f64().unwrap_or(0.0).round();
                    if index <= 0.0 {
                        args[3] = JsonValue::Null;
                    } else {
                        args[3] = JsonValue::Number(JsonNumber::from(index as u64));
                    }
                }
                Ok(args)
            }
            VisualApiOperation::EcsSetScriptField => {
                if args.len() >= 4 {
                    let index = args[3].as_f64().unwrap_or(0.0).round();
                    if index <= 0.0 {
                        args[3] = JsonValue::Null;
                    } else {
                        args[3] = JsonValue::Number(JsonNumber::from(index as u64));
                    }
                }
                Ok(args)
            }
            VisualApiOperation::EcsGetCollisionEventCount
            | VisualApiOperation::EcsGetCollisionEventOther
            | VisualApiOperation::EcsGetCollisionEventNormal
            | VisualApiOperation::EcsGetCollisionEventPoint
            | VisualApiOperation::EcsGetTriggerEventCount
            | VisualApiOperation::EcsGetTriggerEventOther
            | VisualApiOperation::EcsGetTriggerEventNormal
            | VisualApiOperation::EcsGetTriggerEventPoint => {
                if args.len() >= 2 {
                    let phase = args[1].as_str().unwrap_or("").trim();
                    if phase.is_empty() {
                        args[1] = JsonValue::String("all".to_string());
                    }
                }
                if args.len() >= 3 {
                    let index = args[2].as_f64().unwrap_or(1.0).round().max(1.0);
                    args[2] = JsonValue::Number(JsonNumber::from(index as u64));
                }
                Ok(args)
            }
            VisualApiOperation::EcsSphereCast
            | VisualApiOperation::EcsSphereCastHasHit
            | VisualApiOperation::EcsSphereCastHitEntity
            | VisualApiOperation::EcsSphereCastPoint
            | VisualApiOperation::EcsSphereCastNormal
            | VisualApiOperation::EcsSphereCastToi => {
                if args.len() >= 2 {
                    let radius = args[1].as_f64().unwrap_or(0.1).max(0.0001);
                    args[1] = JsonValue::Number(
                        JsonNumber::from_f64(radius).unwrap_or(JsonNumber::from(0)),
                    );
                }
                if args.len() >= 4 {
                    let max_toi = args[3].as_f64().unwrap_or(10_000.0).max(0.0);
                    args[3] = JsonValue::Number(
                        JsonNumber::from_f64(max_toi).unwrap_or(JsonNumber::from(10_000)),
                    );
                }
                Ok(args)
            }
            VisualApiOperation::EcsGetAudioBusName
            | VisualApiOperation::EcsGetAudioBusVolume
            | VisualApiOperation::EcsSetAudioBusName
            | VisualApiOperation::EcsSetAudioBusVolume
            | VisualApiOperation::EcsRemoveAudioBus => {
                if let Some(bus) = args.get(0) {
                    let bus = bus.as_str().unwrap_or("").trim();
                    if bus.is_empty() {
                        args[0] = JsonValue::String("Master".to_string());
                    }
                }
                Ok(args)
            }
            VisualApiOperation::InputSetCursorGrab => {
                if let Some(mode) = args.get(0) {
                    let mode = mode.as_str().unwrap_or("").trim();
                    if mode.is_empty() {
                        args[0] = JsonValue::String("none".to_string());
                    }
                }
                Ok(args)
            }
            _ => Ok(args),
        }
    }

    fn collect_function_call_args(
        &mut self,
        node_id: u64,
        inputs: &[VisualFunctionIoDefinition],
        defaults: &[String],
    ) -> Result<Vec<JsonValue>, String> {
        let mut args = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            let fallback = defaults
                .get(index)
                .map(|value| value.as_str())
                .or(Some(input.default_value.as_str()));
            let mut stack = HashSet::new();
            let value = self.resolve_data_input_with_stack(node_id, index, fallback, &mut stack)?;
            let coerced = coerce_json_to_visual_type_with_array_item(
                &value,
                input.value_type,
                input.array_item_type,
            )
            .map_err(|err| {
                format!(
                    "Function argument '{}' at node {} is invalid for {}: {}",
                    input.name.trim(),
                    node_id,
                    input.value_type.title(),
                    err
                )
            })?;
            args.push(coerced);
        }
        Ok(args)
    }

    fn next_function_call_token(&mut self) -> u64 {
        let key = function_call_counter_key();
        let current = self
            .state
            .variables
            .get(&key)
            .and_then(json_to_u64)
            .unwrap_or(0);
        let next = current.saturating_add(1).max(1);
        self.state
            .variables
            .insert(key, JsonValue::Number(JsonNumber::from(next)));
        next
    }

    fn active_function_call_token(&self, function_id: u64) -> Option<u64> {
        let key = function_active_call_key(function_id);
        self.state.variables.get(&key).and_then(json_to_u64)
    }

    fn invoke_function(
        &mut self,
        function_id: u64,
        function_name: &str,
        _input_defs: &[VisualFunctionIoDefinition],
        output_defs: &[VisualFunctionIoDefinition],
        args: &[JsonValue],
    ) -> Result<FunctionInvokeResult, String> {
        let next_call_depth = self.call_depth.saturating_add(1);
        if next_call_depth > MAX_FUNCTION_CALL_DEPTH {
            return Err(format!(
                "Function call depth exceeded max of {} while invoking '{}'",
                MAX_FUNCTION_CALL_DEPTH, function_name
            ));
        }

        let function = if function_id != 0 {
            self.function_registry.get(&function_id).cloned()
        } else {
            self.function_registry
                .values()
                .find(|entry| entry.name == function_name.trim())
                .cloned()
        }
        .ok_or_else(|| {
            format!(
                "Function '{}' (id {}) is not defined in this visual script",
                function_name.trim(),
                function_id
            )
        })?;

        let token = self.next_function_call_token();
        let active_key = function_active_call_key(function.id);
        let previous_active = self.state.variables.get(&active_key).cloned();
        self.state.variables.insert(
            active_key.clone(),
            JsonValue::Number(JsonNumber::from(token)),
        );

        for (index, input) in function.inputs.iter().enumerate() {
            let value = args
                .get(index)
                .cloned()
                .unwrap_or_else(|| parse_loose_literal(&input.default_value));
            let coerced = coerce_json_to_visual_type_with_array_item(
                &value,
                input.value_type,
                input.array_item_type,
            )?;
            let key = function_input_value_key(function.id, token, index);
            self.state.variables.insert(key, coerced);
        }

        for index in 0..function.outputs.len() {
            self.state
                .variables
                .remove(&function_output_value_key(function.id, token, index));
        }

        let run_result = function.program.execute_event_internal(
            VisualScriptEvent::Start,
            self.owner_entity_id,
            self.dt,
            self.state,
            self.host,
            self.function_registry,
            next_call_depth,
        );

        match previous_active {
            Some(value) => {
                self.state.variables.insert(active_key, value);
            }
            None => {
                self.state.variables.remove(&active_key);
            }
        }

        run_result.map_err(|err| {
            if function.source_path.trim().is_empty() {
                format!("Function '{}' failed: {}", function.name, err)
            } else {
                format!(
                    "Function '{}' ({}) failed: {}",
                    function.name, function.source_path, err
                )
            }
        })?;

        if let Some(values) =
            self.collect_completed_function_outputs(&function, token, output_defs)?
        {
            return Ok(FunctionInvokeResult::Completed(values));
        }

        Ok(FunctionInvokeResult::Pending(VisualPendingFunctionCall {
            function_id: function.id,
            function_name: function.name.clone(),
            call_token: token,
            output_defs: output_defs.to_vec(),
        }))
    }

    fn collect_completed_function_outputs(
        &mut self,
        function: &VisualCompiledFunctionProgram,
        token: u64,
        output_defs: &[VisualFunctionIoDefinition],
    ) -> Result<Option<Vec<JsonValue>>, String> {
        for (index, _) in output_defs.iter().enumerate() {
            let key = function_output_value_key(function.id, token, index);
            if !self.state.variables.contains_key(&key) {
                return Ok(None);
            }
        }

        let mut out = Vec::with_capacity(output_defs.len());
        for (index, output) in output_defs.iter().enumerate() {
            let key = function_output_value_key(function.id, token, index);
            let value = self
                .state
                .variables
                .remove(&key)
                .unwrap_or_else(|| parse_loose_literal(&output.default_value));
            let coerced = coerce_json_to_visual_type_with_array_item(
                &value,
                output.value_type,
                output.array_item_type,
            )?;
            out.push(coerced);
        }

        for index in 0..function.inputs.len() {
            self.state
                .variables
                .remove(&function_input_value_key(function.id, token, index));
        }

        Ok(Some(out))
    }

    fn resume_function_call(
        &mut self,
        pending: &VisualPendingFunctionCall,
    ) -> Result<Option<Vec<JsonValue>>, String> {
        let next_call_depth = self.call_depth.saturating_add(1);
        if next_call_depth > MAX_FUNCTION_CALL_DEPTH {
            return Err(format!(
                "Function call depth exceeded max of {} while resuming '{}'",
                MAX_FUNCTION_CALL_DEPTH, pending.function_name
            ));
        }

        let function = if pending.function_id != 0 {
            self.function_registry.get(&pending.function_id).cloned()
        } else {
            self.function_registry
                .values()
                .find(|entry| entry.name == pending.function_name.trim())
                .cloned()
        }
        .ok_or_else(|| {
            format!(
                "Function '{}' (id {}) is not defined in this visual script",
                pending.function_name.trim(),
                pending.function_id
            )
        })?;

        let active_key = function_active_call_key(function.id);
        let previous_active = self.state.variables.get(&active_key).cloned();
        self.state.variables.insert(
            active_key.clone(),
            JsonValue::Number(JsonNumber::from(pending.call_token)),
        );

        let run_result = function.program.execute_event_internal(
            VisualScriptEvent::Update,
            self.owner_entity_id,
            self.dt,
            self.state,
            self.host,
            self.function_registry,
            next_call_depth,
        );

        match previous_active {
            Some(value) => {
                self.state.variables.insert(active_key, value);
            }
            None => {
                self.state.variables.remove(&active_key);
            }
        }

        run_result.map_err(|err| {
            if function.source_path.trim().is_empty() {
                format!("Function '{}' failed: {}", function.name, err)
            } else {
                format!(
                    "Function '{}' ({}) failed: {}",
                    function.name, function.source_path, err
                )
            }
        })?;

        self.collect_completed_function_outputs(&function, pending.call_token, &pending.output_defs)
    }

    fn resolve_bool_input(
        &mut self,
        node_id: u64,
        data_input: usize,
        fallback_literal: Option<&str>,
    ) -> Result<bool, String> {
        let mut stack = HashSet::new();
        let value =
            self.resolve_data_input_with_stack(node_id, data_input, fallback_literal, &mut stack)?;
        Ok(is_truthy(&value))
    }

    fn resolve_data_input_with_stack(
        &mut self,
        node_id: u64,
        data_input: usize,
        fallback_literal: Option<&str>,
        stack: &mut HashSet<(u64, usize)>,
    ) -> Result<JsonValue, String> {
        if let Some((from_node, from_output)) =
            self.program.data_edges.get(&(node_id, data_input)).copied()
        {
            return self.evaluate_data_output_with_stack(from_node, from_output, stack);
        }

        match fallback_literal {
            Some(text) => Ok(parse_loose_literal(text)),
            None => Ok(JsonValue::Null),
        }
    }

    fn evaluate_data_output_with_stack(
        &mut self,
        node_id: u64,
        data_output: usize,
        stack: &mut HashSet<(u64, usize)>,
    ) -> Result<JsonValue, String> {
        let key = (node_id, data_output);

        if !matches!(
            self.program.nodes.get(&node_id),
            Some(VisualScriptNodeKind::QueryApi { .. })
        ) {
            if let Some(value) = self.data_cache.get(&key) {
                return Ok(value.clone());
            }
        }

        if !stack.insert(key) {
            return Err(format!(
                "Data flow cycle detected while evaluating node {} output {}",
                node_id,
                data_output + 1
            ));
        }

        let Some(node) = self.program.nodes.get(&node_id) else {
            return Err(format!("Visual script runtime missing node {}", node_id));
        };

        let result = match node {
            VisualScriptNodeKind::GetVariable {
                variable_id,
                name,
                default_value,
            } => {
                let variable = runtime_variable_key(*variable_id, name);
                if variable.is_empty() {
                    JsonValue::Null
                } else {
                    let fallback = parse_loose_literal(default_value);
                    let fallback =
                        if let Some(value_type) = self.program.variable_types.get(&variable) {
                            let array_item_type = self
                                .program
                                .variable_array_item_types
                                .get(&variable)
                                .copied()
                                .flatten();
                            coerce_json_to_visual_type_with_array_item(
                                &fallback,
                                *value_type,
                                array_item_type,
                            )
                            .unwrap_or(fallback)
                        } else {
                            fallback
                        };
                    self.state
                        .variables
                        .get(&variable)
                        .cloned()
                        .or_else(|| self.program.variable_defaults.get(&variable).cloned())
                        .unwrap_or(fallback)
                }
            }
            VisualScriptNodeKind::CallApi { .. } => self
                .node_results
                .get(&node_id)
                .cloned()
                .unwrap_or(JsonValue::Null),
            VisualScriptNodeKind::QueryApi {
                operation, args, ..
            } => {
                let spec = operation.spec();
                let values = self.collect_call_args(node_id, *operation, args, spec.inputs)?;
                self.host
                    .invoke_api(spec.table, spec.function, &values)
                    .map_err(|err| {
                        format!(
                            "{}.{} failed at node {}: {}",
                            spec.table.as_str(),
                            spec.function,
                            node_id,
                            err
                        )
                    })
                    .and_then(|value| {
                        if let Some(value_type) = spec.output_type {
                            coerce_json_to_visual_type(&value, value_type)
                        } else {
                            Ok(value)
                        }
                    })?
            }
            VisualScriptNodeKind::BoolLiteral { value } => JsonValue::Bool(*value),
            VisualScriptNodeKind::NumberLiteral { value } => json_number(*value),
            VisualScriptNodeKind::StringLiteral { value } => JsonValue::String(value.clone()),
            VisualScriptNodeKind::AnyLiteral { value } => parse_loose_literal(value),
            VisualScriptNodeKind::PhysicsQueryFilterLiteral { value } => {
                let parsed = parse_loose_literal(value);
                coerce_json_to_visual_type(&parsed, VisualValueType::PhysicsQueryFilter)?
            }
            VisualScriptNodeKind::SelfEntity => {
                JsonValue::Number(JsonNumber::from(self.owner_entity_id))
            }
            VisualScriptNodeKind::DeltaTime => json_number(f64::from(self.dt)),
            VisualScriptNodeKind::TimeSinceStart => json_number(self.state.elapsed_seconds),
            VisualScriptNodeKind::UnixTimeSeconds => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|duration| duration.as_secs_f64())
                    .unwrap_or(0.0);
                json_number(now)
            }
            VisualScriptNodeKind::TimeSince { origin_seconds } => {
                let origin = self.resolve_number_input_with_stack(
                    node_id,
                    0,
                    Some(origin_seconds.as_str()),
                    stack,
                )?;
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|duration| duration.as_secs_f64())
                    .unwrap_or(0.0);
                json_number((now - origin).max(0.0))
            }
            VisualScriptNodeKind::FunctionStart {
                function_id,
                inputs,
            } => {
                let call_token = self.active_function_call_token(*function_id).ok_or_else(
                    || {
                        format!(
                            "Function Start node {} has no active call context for function id {}",
                            node_id, function_id
                        )
                    },
                )?;
                let key = function_input_value_key(*function_id, call_token, data_output);
                let value = self.state.variables.get(&key).cloned().or_else(|| {
                    inputs
                        .get(data_output)
                        .map(|input| parse_loose_literal(&input.default_value))
                });
                let Some(value) = value else {
                    return Err(format!(
                        "Function Start node {} requested unknown input {}",
                        node_id,
                        data_output + 1
                    ));
                };
                let (value_type, array_item_type) = inputs
                    .get(data_output)
                    .map(|input| (input.value_type, input.array_item_type))
                    .unwrap_or((VisualValueType::Any, None));
                coerce_json_to_visual_type_with_array_item(&value, value_type, array_item_type)?
            }
            VisualScriptNodeKind::CallFunction {
                function_id,
                name,
                inputs,
                outputs,
                args,
            } => {
                let pending_key = self.pending_call_key(node_id)?;
                let values = if let Some(JsonValue::Array(values)) = self.node_results.get(&node_id)
                {
                    values.clone()
                } else if let Some(pending) =
                    self.state.pending_function_calls.get(&pending_key).cloned()
                {
                    if let Some(values) = self.resume_function_call(&pending)? {
                        self.state.pending_function_calls.remove(&pending_key);
                        values
                    } else {
                        Vec::new()
                    }
                } else {
                    let call_args = self.collect_function_call_args(node_id, inputs, args)?;
                    match self.invoke_function(
                        *function_id,
                        name.as_str(),
                        inputs,
                        outputs,
                        &call_args,
                    )? {
                        FunctionInvokeResult::Completed(values) => values,
                        FunctionInvokeResult::Pending(pending) => {
                            self.state
                                .pending_function_calls
                                .insert(pending_key, pending);
                            Vec::new()
                        }
                    }
                };
                for (index, value) in values.iter().enumerate() {
                    self.data_cache.insert((node_id, index), value.clone());
                }
                self.node_results
                    .insert(node_id, JsonValue::Array(values.clone()));
                values.get(data_output).cloned().unwrap_or_else(|| {
                    outputs
                        .get(data_output)
                        .and_then(|output| {
                            coerce_json_to_visual_type_with_array_item(
                                &parse_loose_literal(&output.default_value),
                                output.value_type,
                                output.array_item_type,
                            )
                            .ok()
                        })
                        .unwrap_or(JsonValue::Null)
                })
            }
            VisualScriptNodeKind::MathBinary { op } => {
                let left = self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                let right = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                let output = match op {
                    VisualMathOp::Add => left + right,
                    VisualMathOp::Subtract => left - right,
                    VisualMathOp::Multiply => left * right,
                    VisualMathOp::Divide => {
                        if right == 0.0 {
                            return Err(format!("Division by zero at node {}", node_id));
                        }
                        left / right
                    }
                    VisualMathOp::Modulo => {
                        if right == 0.0 {
                            return Err(format!("Modulo by zero at node {}", node_id));
                        }
                        left % right
                    }
                    VisualMathOp::Min => left.min(right),
                    VisualMathOp::Max => left.max(right),
                };
                json_number(output)
            }
            VisualScriptNodeKind::MathTrig { op } => {
                let output = if *op == VisualTrigOp::Atan2 {
                    let y = self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    let x = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                    y.atan2(x)
                } else {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    match op {
                        VisualTrigOp::Sin => value.sin(),
                        VisualTrigOp::Cos => value.cos(),
                        VisualTrigOp::Tan => value.tan(),
                        VisualTrigOp::Asin => value.asin(),
                        VisualTrigOp::Acos => value.acos(),
                        VisualTrigOp::Atan => value.atan(),
                        VisualTrigOp::Atan2 => unreachable!(),
                    }
                };
                json_number(output)
            }
            VisualScriptNodeKind::MathInterpolation { op } => match op {
                VisualInterpolationOp::Lerp => {
                    let a = self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    let b = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                    let t = self.resolve_number_input_with_stack(node_id, 2, Some("0"), stack)?;
                    json_number(a + ((b - a) * t))
                }
                VisualInterpolationOp::SmoothStep => {
                    let edge0 =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    let edge1 =
                        self.resolve_number_input_with_stack(node_id, 1, Some("1"), stack)?;
                    let value =
                        self.resolve_number_input_with_stack(node_id, 2, Some("0"), stack)?;
                    if (edge1 - edge0).abs() <= f64::EPSILON {
                        json_number(0.0)
                    } else {
                        let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                        json_number(t * t * (3.0 - (2.0 * t)))
                    }
                }
                VisualInterpolationOp::InverseLerp => {
                    let a = self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    let b = self.resolve_number_input_with_stack(node_id, 1, Some("1"), stack)?;
                    let value =
                        self.resolve_number_input_with_stack(node_id, 2, Some("0"), stack)?;
                    if (b - a).abs() <= f64::EPSILON {
                        json_number(0.0)
                    } else {
                        json_number((value - a) / (b - a))
                    }
                }
                VisualInterpolationOp::Vec3Lerp => {
                    let a = self.resolve_data_input_with_stack(
                        node_id,
                        0,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let b = self.resolve_data_input_with_stack(
                        node_id,
                        1,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let t = self.resolve_number_input_with_stack(node_id, 2, Some("0"), stack)?;
                    let (ax, ay, az) = coerce_json_to_vec3_components(&a)?;
                    let (bx, by, bz) = coerce_json_to_vec3_components(&b)?;
                    vec3_json(
                        ax + ((bx - ax) * t),
                        ay + ((by - ay) * t),
                        az + ((bz - az) * t),
                    )
                }
                VisualInterpolationOp::QuatSlerp => {
                    let a = self.resolve_data_input_with_stack(
                        node_id,
                        0,
                        Some("{\"x\":0,\"y\":0,\"z\":0,\"w\":1}"),
                        stack,
                    )?;
                    let b = self.resolve_data_input_with_stack(
                        node_id,
                        1,
                        Some("{\"x\":0,\"y\":0,\"z\":0,\"w\":1}"),
                        stack,
                    )?;
                    let t = self
                        .resolve_number_input_with_stack(node_id, 2, Some("0"), stack)?
                        .clamp(0.0, 1.0);
                    let (ax, ay, az, aw) = coerce_json_to_quat_components(&a)?;
                    let (bx, by, bz, bw) = coerce_json_to_quat_components(&b)?;
                    let qa = DQuat::from_xyzw(ax, ay, az, aw);
                    let qb = DQuat::from_xyzw(bx, by, bz, bw);
                    let qa = if qa.length_squared() > 1.0e-12 {
                        qa.normalize()
                    } else {
                        DQuat::IDENTITY
                    };
                    let qb = if qb.length_squared() > 1.0e-12 {
                        qb.normalize()
                    } else {
                        DQuat::IDENTITY
                    };
                    let result = qa.slerp(qb, t).normalize();
                    quat_json(result.x, result.y, result.z, result.w)
                }
            },
            VisualScriptNodeKind::MathVector { op } => match op {
                VisualVectorMathOp::Dot => {
                    let a = self.resolve_data_input_with_stack(
                        node_id,
                        0,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let b = self.resolve_data_input_with_stack(
                        node_id,
                        1,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let (ax, ay, az) = coerce_json_to_vec3_components(&a)?;
                    let (bx, by, bz) = coerce_json_to_vec3_components(&b)?;
                    json_number((ax * bx) + (ay * by) + (az * bz))
                }
                VisualVectorMathOp::Cross => {
                    let a = self.resolve_data_input_with_stack(
                        node_id,
                        0,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let b = self.resolve_data_input_with_stack(
                        node_id,
                        1,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let (ax, ay, az) = coerce_json_to_vec3_components(&a)?;
                    let (bx, by, bz) = coerce_json_to_vec3_components(&b)?;
                    vec3_json(
                        (ay * bz) - (az * by),
                        (az * bx) - (ax * bz),
                        (ax * by) - (ay * bx),
                    )
                }
                VisualVectorMathOp::Length => {
                    let a = self.resolve_data_input_with_stack(
                        node_id,
                        0,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let (x, y, z) = coerce_json_to_vec3_components(&a)?;
                    json_number(((x * x) + (y * y) + (z * z)).sqrt())
                }
                VisualVectorMathOp::Normalize => {
                    let a = self.resolve_data_input_with_stack(
                        node_id,
                        0,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let (x, y, z) = coerce_json_to_vec3_components(&a)?;
                    let length = ((x * x) + (y * y) + (z * z)).sqrt();
                    if length <= 1.0e-8 {
                        vec3_json(0.0, 0.0, 0.0)
                    } else {
                        vec3_json(x / length, y / length, z / length)
                    }
                }
                VisualVectorMathOp::Distance => {
                    let a = self.resolve_data_input_with_stack(
                        node_id,
                        0,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let b = self.resolve_data_input_with_stack(
                        node_id,
                        1,
                        Some("{\"x\":0,\"y\":0,\"z\":0}"),
                        stack,
                    )?;
                    let (ax, ay, az) = coerce_json_to_vec3_components(&a)?;
                    let (bx, by, bz) = coerce_json_to_vec3_components(&b)?;
                    let dx = bx - ax;
                    let dy = by - ay;
                    let dz = bz - az;
                    json_number(((dx * dx) + (dy * dy) + (dz * dz)).sqrt())
                }
            },
            VisualScriptNodeKind::MathProcedural { op } => match op {
                VisualProceduralMathOp::Clamp => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    let min = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                    let max = self.resolve_number_input_with_stack(node_id, 2, Some("1"), stack)?;
                    let low = min.min(max);
                    let high = min.max(max);
                    json_number(value.clamp(low, high))
                }
                VisualProceduralMathOp::Remap => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    let in_min =
                        self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                    let in_max =
                        self.resolve_number_input_with_stack(node_id, 2, Some("1"), stack)?;
                    let out_min =
                        self.resolve_number_input_with_stack(node_id, 3, Some("0"), stack)?;
                    let out_max =
                        self.resolve_number_input_with_stack(node_id, 4, Some("1"), stack)?;
                    if (in_max - in_min).abs() <= f64::EPSILON {
                        return Err(format!("Remap input range is zero at node {}", node_id));
                    }
                    let t = (value - in_min) / (in_max - in_min);
                    json_number(out_min + ((out_max - out_min) * t))
                }
                VisualProceduralMathOp::Saturate => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.clamp(0.0, 1.0))
                }
                VisualProceduralMathOp::Fract => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.fract())
                }
            },
            VisualScriptNodeKind::MathUtility { op } => match op {
                VisualUtilityMathOp::Abs => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.abs())
                }
                VisualUtilityMathOp::Sign => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.signum())
                }
                VisualUtilityMathOp::Floor => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.floor())
                }
                VisualUtilityMathOp::Ceil => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.ceil())
                }
                VisualUtilityMathOp::Round => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.round())
                }
                VisualUtilityMathOp::Sqrt => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    if value < 0.0 {
                        return Err(format!(
                            "Sqrt input must be non-negative at node {}",
                            node_id
                        ));
                    }
                    json_number(value.sqrt())
                }
                VisualUtilityMathOp::Pow => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    let exponent =
                        self.resolve_number_input_with_stack(node_id, 1, Some("1"), stack)?;
                    json_number(value.powf(exponent))
                }
                VisualUtilityMathOp::Exp => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.exp())
                }
                VisualUtilityMathOp::Log => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("1"), stack)?;
                    let base =
                        self.resolve_number_input_with_stack(node_id, 1, Some("10"), stack)?;
                    if value <= 0.0 || base <= 0.0 || (base - 1.0).abs() <= f64::EPSILON {
                        return Err(format!("Invalid log inputs at node {}", node_id));
                    }
                    json_number(value.log(base))
                }
                VisualUtilityMathOp::Degrees => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.to_degrees())
                }
                VisualUtilityMathOp::Radians => {
                    let value =
                        self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                    json_number(value.to_radians())
                }
            },
            VisualScriptNodeKind::Compare { op } => {
                let left = self.resolve_data_input_with_stack(node_id, 0, Some("null"), stack)?;
                let right = self.resolve_data_input_with_stack(node_id, 1, Some("null"), stack)?;
                let result = compare_visual_values(&left, &right, *op);
                JsonValue::Bool(result)
            }
            VisualScriptNodeKind::LogicalBinary { op } => {
                let left = self.resolve_data_input_with_stack(node_id, 0, Some("false"), stack)?;
                let right = self.resolve_data_input_with_stack(node_id, 1, Some("false"), stack)?;
                let result = match op {
                    VisualLogicalOp::And => is_truthy(&left) && is_truthy(&right),
                    VisualLogicalOp::Or => is_truthy(&left) || is_truthy(&right),
                };
                JsonValue::Bool(result)
            }
            VisualScriptNodeKind::Not => {
                let value = self.resolve_data_input_with_stack(node_id, 0, Some("false"), stack)?;
                JsonValue::Bool(!is_truthy(&value))
            }
            VisualScriptNodeKind::Select { value_type } => {
                let condition =
                    self.resolve_data_input_with_stack(node_id, 0, Some("false"), stack)?;
                let selected = if is_truthy(&condition) {
                    self.resolve_data_input_with_stack(node_id, 1, Some("null"), stack)?
                } else {
                    self.resolve_data_input_with_stack(node_id, 2, Some("null"), stack)?
                };
                coerce_json_to_visual_type(&selected, *value_type)?
            }
            VisualScriptNodeKind::Vec2GetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0}"),
                    stack,
                )?;
                let (x, y) = coerce_json_to_vec2_components(&value)?;
                let number = match component {
                    VisualVec2Component::X => x,
                    VisualVec2Component::Y => y,
                };
                json_number(number)
            }
            VisualScriptNodeKind::Vec2SetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0}"),
                    stack,
                )?;
                let (mut x, mut y) = coerce_json_to_vec2_components(&value)?;
                let updated = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                match component {
                    VisualVec2Component::X => x = updated,
                    VisualVec2Component::Y => y = updated,
                }
                vec2_json(x, y)
            }
            VisualScriptNodeKind::Vec3GetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0,\"z\":0}"),
                    stack,
                )?;
                let (x, y, z) = coerce_json_to_vec3_components(&value)?;
                let number = match component {
                    VisualVec3Component::X => x,
                    VisualVec3Component::Y => y,
                    VisualVec3Component::Z => z,
                };
                json_number(number)
            }
            VisualScriptNodeKind::Vec3SetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0,\"z\":0}"),
                    stack,
                )?;
                let (mut x, mut y, mut z) = coerce_json_to_vec3_components(&value)?;
                let updated = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                match component {
                    VisualVec3Component::X => x = updated,
                    VisualVec3Component::Y => y = updated,
                    VisualVec3Component::Z => z = updated,
                }
                vec3_json(x, y, z)
            }
            VisualScriptNodeKind::QuatGetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0,\"z\":0,\"w\":1}"),
                    stack,
                )?;
                let (x, y, z, w) = coerce_json_to_quat_components(&value)?;
                let number = match component {
                    VisualQuatComponent::X => x,
                    VisualQuatComponent::Y => y,
                    VisualQuatComponent::Z => z,
                    VisualQuatComponent::W => w,
                };
                json_number(number)
            }
            VisualScriptNodeKind::QuatSetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0,\"z\":0,\"w\":1}"),
                    stack,
                )?;
                let (mut x, mut y, mut z, mut w) = coerce_json_to_quat_components(&value)?;
                let updated = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                match component {
                    VisualQuatComponent::X => x = updated,
                    VisualQuatComponent::Y => y = updated,
                    VisualQuatComponent::Z => z = updated,
                    VisualQuatComponent::W => w = updated,
                }
                quat_json(x, y, z, w)
            }
            VisualScriptNodeKind::TransformGetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some(default_literal_for_type(VisualValueType::Transform)),
                    stack,
                )?;
                let normalized = coerce_json_to_visual_type(&value, VisualValueType::Transform)?;
                let object = normalized
                    .as_object()
                    .ok_or_else(|| "Transform values must be structured objects".to_string())?;
                match component {
                    VisualTransformComponent::Position => object
                        .get("position")
                        .cloned()
                        .ok_or_else(|| "Transform value missing position".to_string())?,
                    VisualTransformComponent::Rotation => object
                        .get("rotation")
                        .cloned()
                        .ok_or_else(|| "Transform value missing rotation".to_string())?,
                    VisualTransformComponent::Scale => object
                        .get("scale")
                        .cloned()
                        .ok_or_else(|| "Transform value missing scale".to_string())?,
                }
            }
            VisualScriptNodeKind::TransformSetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some(default_literal_for_type(VisualValueType::Transform)),
                    stack,
                )?;
                let normalized = coerce_json_to_visual_type(&value, VisualValueType::Transform)?;
                let mut object = normalized
                    .as_object()
                    .cloned()
                    .ok_or_else(|| "Transform values must be structured objects".to_string())?;
                let replacement = match component {
                    VisualTransformComponent::Position => {
                        let value = self.resolve_data_input_with_stack(
                            node_id,
                            1,
                            Some("{\"x\":0,\"y\":0,\"z\":0}"),
                            stack,
                        )?;
                        coerce_json_to_visual_type(&value, VisualValueType::Vec3)?
                    }
                    VisualTransformComponent::Rotation => {
                        let value = self.resolve_data_input_with_stack(
                            node_id,
                            1,
                            Some("{\"x\":0,\"y\":0,\"z\":0,\"w\":1}"),
                            stack,
                        )?;
                        coerce_json_to_visual_type(&value, VisualValueType::Quat)?
                    }
                    VisualTransformComponent::Scale => {
                        let value = self.resolve_data_input_with_stack(
                            node_id,
                            1,
                            Some("{\"x\":1,\"y\":1,\"z\":1}"),
                            stack,
                        )?;
                        coerce_json_to_visual_type(&value, VisualValueType::Vec3)?
                    }
                };
                object.insert(
                    match component {
                        VisualTransformComponent::Position => "position",
                        VisualTransformComponent::Rotation => "rotation",
                        VisualTransformComponent::Scale => "scale",
                    }
                    .to_string(),
                    replacement,
                );
                JsonValue::Object(object)
            }
            VisualScriptNodeKind::PhysicsVelocityGetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some(default_literal_for_type(VisualValueType::PhysicsVelocity)),
                    stack,
                )?;
                let normalized =
                    coerce_json_to_visual_type(&value, VisualValueType::PhysicsVelocity)?;
                let object = normalized.as_object().ok_or_else(|| {
                    "Physics velocity values must be structured objects".to_string()
                })?;
                match component {
                    VisualPhysicsVelocityComponent::Linear => object
                        .get("linear")
                        .cloned()
                        .ok_or_else(|| "Physics velocity value missing linear".to_string())?,
                    VisualPhysicsVelocityComponent::Angular => object
                        .get("angular")
                        .cloned()
                        .ok_or_else(|| "Physics velocity value missing angular".to_string())?,
                    VisualPhysicsVelocityComponent::WakeUp => object
                        .get("wake_up")
                        .cloned()
                        .ok_or_else(|| "Physics velocity value missing wake_up".to_string())?,
                }
            }
            VisualScriptNodeKind::PhysicsVelocitySetComponent { component } => {
                let value = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some(default_literal_for_type(VisualValueType::PhysicsVelocity)),
                    stack,
                )?;
                let normalized =
                    coerce_json_to_visual_type(&value, VisualValueType::PhysicsVelocity)?;
                let mut object = normalized.as_object().cloned().ok_or_else(|| {
                    "Physics velocity values must be structured objects".to_string()
                })?;
                let replacement = match component {
                    VisualPhysicsVelocityComponent::Linear
                    | VisualPhysicsVelocityComponent::Angular => {
                        let value = self.resolve_data_input_with_stack(
                            node_id,
                            1,
                            Some("{\"x\":0,\"y\":0,\"z\":0}"),
                            stack,
                        )?;
                        coerce_json_to_visual_type(&value, VisualValueType::Vec3)?
                    }
                    VisualPhysicsVelocityComponent::WakeUp => {
                        let value =
                            self.resolve_data_input_with_stack(node_id, 1, Some("true"), stack)?;
                        coerce_json_to_visual_type(&value, VisualValueType::Bool)?
                    }
                };
                object.insert(
                    match component {
                        VisualPhysicsVelocityComponent::Linear => "linear",
                        VisualPhysicsVelocityComponent::Angular => "angular",
                        VisualPhysicsVelocityComponent::WakeUp => "wake_up",
                    }
                    .to_string(),
                    replacement,
                );
                JsonValue::Object(object)
            }
            VisualScriptNodeKind::Vec2 => {
                let x = self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                let y = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                vec2_json(x, y)
            }
            VisualScriptNodeKind::Vec3 => {
                let x = self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                let y = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                let z = self.resolve_number_input_with_stack(node_id, 2, Some("0"), stack)?;
                vec3_json(x, y, z)
            }
            VisualScriptNodeKind::Quat => {
                let x = self.resolve_number_input_with_stack(node_id, 0, Some("0"), stack)?;
                let y = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                let z = self.resolve_number_input_with_stack(node_id, 2, Some("0"), stack)?;
                let w = self.resolve_number_input_with_stack(node_id, 3, Some("1"), stack)?;
                quat_json(x, y, z, w)
            }
            VisualScriptNodeKind::Transform => {
                let position = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0,\"z\":0}"),
                    stack,
                )?;
                let rotation = self.resolve_data_input_with_stack(
                    node_id,
                    1,
                    Some("{\"x\":0,\"y\":0,\"z\":0,\"w\":1}"),
                    stack,
                )?;
                let scale = self.resolve_data_input_with_stack(
                    node_id,
                    2,
                    Some("{\"x\":1,\"y\":1,\"z\":1}"),
                    stack,
                )?;
                let position = coerce_json_to_visual_type(&position, VisualValueType::Vec3)?;
                let rotation = coerce_json_to_visual_type(&rotation, VisualValueType::Quat)?;
                let scale = coerce_json_to_visual_type(&scale, VisualValueType::Vec3)?;
                let mut object = JsonMap::new();
                object.insert("position".to_string(), position);
                object.insert("rotation".to_string(), rotation);
                object.insert("scale".to_string(), scale);
                JsonValue::Object(object)
            }
            VisualScriptNodeKind::PhysicsVelocity => {
                let linear = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0,\"z\":0}"),
                    stack,
                )?;
                let angular = self.resolve_data_input_with_stack(
                    node_id,
                    1,
                    Some("{\"x\":0,\"y\":0,\"z\":0}"),
                    stack,
                )?;
                let wake_up =
                    self.resolve_data_input_with_stack(node_id, 2, Some("true"), stack)?;
                let linear = coerce_json_to_visual_type(&linear, VisualValueType::Vec3)?;
                let angular = coerce_json_to_visual_type(&angular, VisualValueType::Vec3)?;
                let wake_up = coerce_json_to_visual_type(&wake_up, VisualValueType::Bool)?;
                let mut object = JsonMap::new();
                object.insert("linear".to_string(), linear);
                object.insert("angular".to_string(), angular);
                object.insert("wake_up".to_string(), wake_up);
                JsonValue::Object(object)
            }
            VisualScriptNodeKind::ArrayEmpty { .. } => JsonValue::Array(Vec::new()),
            VisualScriptNodeKind::ArrayLength { .. } => {
                let value = self.resolve_data_input_with_stack(node_id, 0, Some("[]"), stack)?;
                let normalized = coerce_json_to_visual_type(&value, VisualValueType::Array)?;
                let count = normalized
                    .as_array()
                    .map(|entries| entries.len())
                    .unwrap_or(0);
                json_number(count as f64)
            }
            VisualScriptNodeKind::ArrayGet { item_type } => {
                let value = self.resolve_data_input_with_stack(node_id, 0, Some("[]"), stack)?;
                let normalized = coerce_json_to_visual_type(&value, VisualValueType::Array)?;
                let index = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                let fallback = self.resolve_data_input_with_stack(
                    node_id,
                    2,
                    Some(default_literal_for_type(*item_type)),
                    stack,
                )?;
                let fallback = coerce_json_to_visual_type(&fallback, *item_type)?;
                if !index.is_finite() || index < 0.0 {
                    fallback
                } else {
                    let value = normalized
                        .as_array()
                        .and_then(|entries| entries.get(index.floor() as usize))
                        .cloned()
                        .unwrap_or(fallback.clone());
                    coerce_json_to_visual_type(&value, *item_type)?
                }
            }
            VisualScriptNodeKind::ArraySet { item_type } => {
                let value = self.resolve_data_input_with_stack(node_id, 0, Some("[]"), stack)?;
                let normalized = coerce_json_to_visual_type(&value, VisualValueType::Array)?;
                let index = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                let replacement = self.resolve_data_input_with_stack(
                    node_id,
                    2,
                    Some(default_literal_for_type(*item_type)),
                    stack,
                )?;
                let replacement = coerce_json_to_visual_type(&replacement, *item_type)?;
                let mut array = normalized.as_array().cloned().unwrap_or_default();
                if index.is_finite() && index >= 0.0 {
                    let index = index.floor() as usize;
                    if index >= array.len() {
                        array.resize(
                            index + 1,
                            parse_loose_literal(default_literal_for_type(*item_type)),
                        );
                    }
                    array[index] = replacement;
                }
                JsonValue::Array(array)
            }
            VisualScriptNodeKind::ArrayPush { item_type } => {
                let value = self.resolve_data_input_with_stack(node_id, 0, Some("[]"), stack)?;
                let normalized = coerce_json_to_visual_type(&value, VisualValueType::Array)?;
                let mut array = normalized.as_array().cloned().unwrap_or_default();
                let pushed = self.resolve_data_input_with_stack(
                    node_id,
                    1,
                    Some(default_literal_for_type(*item_type)),
                    stack,
                )?;
                let pushed = coerce_json_to_visual_type(&pushed, *item_type)?;
                array.push(pushed);
                JsonValue::Array(array)
            }
            VisualScriptNodeKind::ArrayRemoveAt { .. } => {
                let value = self.resolve_data_input_with_stack(node_id, 0, Some("[]"), stack)?;
                let normalized = coerce_json_to_visual_type(&value, VisualValueType::Array)?;
                let index = self.resolve_number_input_with_stack(node_id, 1, Some("0"), stack)?;
                let mut array = normalized.as_array().cloned().unwrap_or_default();
                if index.is_finite() && index >= 0.0 {
                    let index = index.floor() as usize;
                    if index < array.len() {
                        array.remove(index);
                    }
                }
                JsonValue::Array(array)
            }
            VisualScriptNodeKind::ArrayClear { .. } => JsonValue::Array(Vec::new()),
            VisualScriptNodeKind::RayCast => {
                let origin = self.resolve_data_input_with_stack(
                    node_id,
                    0,
                    Some("{\"x\":0,\"y\":0,\"z\":0}"),
                    stack,
                )?;
                let direction = self.resolve_data_input_with_stack(
                    node_id,
                    1,
                    Some("{\"x\":0,\"y\":-1,\"z\":0}"),
                    stack,
                )?;
                let max_toi =
                    self.resolve_number_input_with_stack(node_id, 2, Some("10000"), stack)?;
                let solid = self.resolve_data_input_with_stack(node_id, 3, Some("true"), stack)?;
                let filter = self.resolve_data_input_with_stack(
                    node_id,
                    4,
                    Some(default_literal_for_type(
                        VisualValueType::PhysicsQueryFilter,
                    )),
                    stack,
                )?;
                let exclude_entity =
                    self.resolve_data_input_with_stack(node_id, 5, Some("null"), stack)?;

                let origin = coerce_json_to_visual_type(&origin, VisualValueType::Vec3)?;
                let direction = coerce_json_to_visual_type(&direction, VisualValueType::Vec3)?;
                let solid = coerce_json_to_visual_type(&solid, VisualValueType::Bool)?;
                let filter =
                    coerce_json_to_visual_type(&filter, VisualValueType::PhysicsQueryFilter)?;
                let exclude_entity = if exclude_entity.is_null() {
                    JsonValue::Null
                } else {
                    coerce_json_to_visual_type(&exclude_entity, VisualValueType::Entity)?
                };

                let args = vec![
                    origin,
                    direction,
                    json_number(max_toi.max(0.0)),
                    solid,
                    filter,
                    exclude_entity,
                ];
                let result = self
                    .host
                    .invoke_api(VisualScriptApiTable::Ecs, "ray_cast", &args)
                    .map_err(|err| format!("ecs.ray_cast failed at node {}: {}", node_id, err))?;
                coerce_json_to_visual_type(&result, VisualValueType::PhysicsRayCastHit)?
            }
            VisualScriptNodeKind::OnStart
            | VisualScriptNodeKind::OnUpdate
            | VisualScriptNodeKind::OnStop
            | VisualScriptNodeKind::OnCollisionEnter
            | VisualScriptNodeKind::OnCollisionStay
            | VisualScriptNodeKind::OnCollisionExit
            | VisualScriptNodeKind::OnTriggerEnter
            | VisualScriptNodeKind::OnTriggerExit
            | VisualScriptNodeKind::OnInputAction { .. }
            | VisualScriptNodeKind::OnCustomEvent { .. }
            | VisualScriptNodeKind::Sequence { .. }
            | VisualScriptNodeKind::Branch { .. }
            | VisualScriptNodeKind::LoopWhile { .. }
            | VisualScriptNodeKind::Log { .. }
            | VisualScriptNodeKind::SetVariable { .. }
            | VisualScriptNodeKind::ClearVariable { .. }
            | VisualScriptNodeKind::WaitSeconds { .. }
            | VisualScriptNodeKind::FunctionReturn { .. }
            | VisualScriptNodeKind::Comment { .. }
            | VisualScriptNodeKind::Statement { .. } => JsonValue::Null,
        };

        stack.remove(&key);

        if !matches!(node, VisualScriptNodeKind::QueryApi { .. }) {
            self.data_cache.insert(key, result.clone());
        }

        Ok(result)
    }

    fn resolve_number_input_with_stack(
        &mut self,
        node_id: u64,
        data_input: usize,
        fallback_literal: Option<&str>,
        stack: &mut HashSet<(u64, usize)>,
    ) -> Result<f64, String> {
        let value =
            self.resolve_data_input_with_stack(node_id, data_input, fallback_literal, stack)?;
        coerce_json_to_f64(&value)
    }
}

fn compare_visual_values(left: &JsonValue, right: &JsonValue, op: VisualCompareOp) -> bool {
    let ordering = compare_visual_values_ordering(left, right);
    match op {
        VisualCompareOp::Equals => ordering == Ordering::Equal,
        VisualCompareOp::NotEquals => ordering != Ordering::Equal,
        VisualCompareOp::Less => ordering == Ordering::Less,
        VisualCompareOp::LessOrEqual => ordering != Ordering::Greater,
        VisualCompareOp::Greater => ordering == Ordering::Greater,
        VisualCompareOp::GreaterOrEqual => ordering != Ordering::Less,
    }
}

fn compare_visual_values_ordering(left: &JsonValue, right: &JsonValue) -> Ordering {
    let left_type = infer_visual_value_type_from_json(left);
    let right_type = infer_visual_value_type_from_json(right);

    if let Some(common_type) = shared_compare_type(left_type, right_type) {
        return compare_visual_values_as_type(left, right, common_type);
    }

    if let (Ok(left_number), Ok(right_number)) =
        (coerce_json_to_f64(left), coerce_json_to_f64(right))
    {
        return compare_f64(left_number, right_number);
    }

    let type_order =
        visual_value_type_compare_rank(left_type).cmp(&visual_value_type_compare_rank(right_type));
    if type_order != Ordering::Equal {
        return type_order;
    }

    canonical_json_string(left).cmp(&canonical_json_string(right))
}

fn shared_compare_type(
    left_type: VisualValueType,
    right_type: VisualValueType,
) -> Option<VisualValueType> {
    if left_type == right_type {
        return Some(left_type);
    }

    if matches!(
        (left_type, right_type),
        (VisualValueType::Number, VisualValueType::Entity)
            | (VisualValueType::Entity, VisualValueType::Number)
    ) {
        return Some(VisualValueType::Number);
    }

    if left_type == VisualValueType::Any {
        return Some(right_type);
    }

    if right_type == VisualValueType::Any {
        return Some(left_type);
    }

    None
}

fn compare_visual_values_as_type(
    left: &JsonValue,
    right: &JsonValue,
    value_type: VisualValueType,
) -> Ordering {
    match value_type {
        VisualValueType::Bool => is_truthy(left).cmp(&is_truthy(right)),
        VisualValueType::Number => {
            let left_number = coerce_json_to_f64(left).ok().unwrap_or(0.0);
            let right_number = coerce_json_to_f64(right).ok().unwrap_or(0.0);
            compare_f64(left_number, right_number)
        }
        VisualValueType::Entity => {
            let left_entity = coerce_json_to_visual_type(left, VisualValueType::Entity)
                .ok()
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            let right_entity = coerce_json_to_visual_type(right, VisualValueType::Entity)
                .ok()
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            left_entity.cmp(&right_entity)
        }
        VisualValueType::Array => {
            let left_array = coerce_json_to_visual_type(left, VisualValueType::Array)
                .unwrap_or_else(|_| JsonValue::Array(Vec::new()));
            let right_array = coerce_json_to_visual_type(right, VisualValueType::Array)
                .unwrap_or_else(|_| JsonValue::Array(Vec::new()));
            canonical_json_string(&left_array).cmp(&canonical_json_string(&right_array))
        }
        VisualValueType::String => {
            let left_text = coerce_json_to_visual_type(left, VisualValueType::String)
                .ok()
                .and_then(|value| value.as_str().map(|text| text.to_string()))
                .unwrap_or_else(|| json_to_log_string(left));
            let right_text = coerce_json_to_visual_type(right, VisualValueType::String)
                .ok()
                .and_then(|value| value.as_str().map(|text| text.to_string()))
                .unwrap_or_else(|| json_to_log_string(right));
            left_text.cmp(&right_text)
        }
        VisualValueType::Vec2 => {
            let left_vec = coerce_json_to_vec2_components(left).unwrap_or((0.0, 0.0));
            let right_vec = coerce_json_to_vec2_components(right).unwrap_or((0.0, 0.0));
            compare_vec2(left_vec, right_vec)
        }
        VisualValueType::Vec3 => {
            let left_vec = coerce_json_to_vec3_components(left).unwrap_or((0.0, 0.0, 0.0));
            let right_vec = coerce_json_to_vec3_components(right).unwrap_or((0.0, 0.0, 0.0));
            compare_vec3(left_vec, right_vec)
        }
        VisualValueType::Quat => {
            let left_quat = coerce_json_to_quat_components(left).unwrap_or((0.0, 0.0, 0.0, 1.0));
            let right_quat = coerce_json_to_quat_components(right).unwrap_or((0.0, 0.0, 0.0, 1.0));
            compare_quat(left_quat, right_quat)
        }
        VisualValueType::Transform => compare_transforms(left, right),
        VisualValueType::Any => canonical_json_string(left).cmp(&canonical_json_string(right)),
        _ => {
            let left_normalized =
                coerce_json_to_visual_type(left, value_type).unwrap_or_else(|_| left.clone());
            let right_normalized =
                coerce_json_to_visual_type(right, value_type).unwrap_or_else(|_| right.clone());
            canonical_json_string(&left_normalized).cmp(&canonical_json_string(&right_normalized))
        }
    }
}

fn compare_f64(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn compare_vec2(left: (f64, f64), right: (f64, f64)) -> Ordering {
    compare_f64(left.0, right.0).then_with(|| compare_f64(left.1, right.1))
}

fn compare_vec3(left: (f64, f64, f64), right: (f64, f64, f64)) -> Ordering {
    compare_f64(left.0, right.0)
        .then_with(|| compare_f64(left.1, right.1))
        .then_with(|| compare_f64(left.2, right.2))
}

fn compare_quat(left: (f64, f64, f64, f64), right: (f64, f64, f64, f64)) -> Ordering {
    compare_f64(left.0, right.0)
        .then_with(|| compare_f64(left.1, right.1))
        .then_with(|| compare_f64(left.2, right.2))
        .then_with(|| compare_f64(left.3, right.3))
}

fn compare_transforms(left: &JsonValue, right: &JsonValue) -> Ordering {
    let left_normalized = coerce_json_to_visual_type(left, VisualValueType::Transform)
        .unwrap_or_else(|_| {
            parse_loose_literal(default_literal_for_type(VisualValueType::Transform))
        });
    let right_normalized = coerce_json_to_visual_type(right, VisualValueType::Transform)
        .unwrap_or_else(|_| {
            parse_loose_literal(default_literal_for_type(VisualValueType::Transform))
        });

    let left_object = left_normalized.as_object();
    let right_object = right_normalized.as_object();

    let left_position = left_object
        .and_then(|object| object.get("position"))
        .and_then(|value| coerce_json_to_vec3_components(value).ok())
        .unwrap_or((0.0, 0.0, 0.0));
    let right_position = right_object
        .and_then(|object| object.get("position"))
        .and_then(|value| coerce_json_to_vec3_components(value).ok())
        .unwrap_or((0.0, 0.0, 0.0));
    let left_rotation = left_object
        .and_then(|object| object.get("rotation"))
        .and_then(|value| coerce_json_to_quat_components(value).ok())
        .unwrap_or((0.0, 0.0, 0.0, 1.0));
    let right_rotation = right_object
        .and_then(|object| object.get("rotation"))
        .and_then(|value| coerce_json_to_quat_components(value).ok())
        .unwrap_or((0.0, 0.0, 0.0, 1.0));
    let left_scale = left_object
        .and_then(|object| object.get("scale"))
        .and_then(|value| coerce_json_to_vec3_components(value).ok())
        .unwrap_or((1.0, 1.0, 1.0));
    let right_scale = right_object
        .and_then(|object| object.get("scale"))
        .and_then(|value| coerce_json_to_vec3_components(value).ok())
        .unwrap_or((1.0, 1.0, 1.0));

    compare_vec3(left_position, right_position)
        .then_with(|| compare_quat(left_rotation, right_rotation))
        .then_with(|| compare_vec3(left_scale, right_scale))
}

fn visual_value_type_compare_rank(value_type: VisualValueType) -> u8 {
    match value_type {
        VisualValueType::Bool => 0,
        VisualValueType::Number => 1,
        VisualValueType::String => 2,
        VisualValueType::Entity => 3,
        VisualValueType::Array => 4,
        VisualValueType::Vec2 => 5,
        VisualValueType::Vec3 => 6,
        VisualValueType::Quat => 7,
        VisualValueType::Transform => 8,
        VisualValueType::Camera => 9,
        VisualValueType::Light => 10,
        VisualValueType::MeshRenderer => 11,
        VisualValueType::SpriteRenderer => 12,
        VisualValueType::Text2d => 13,
        VisualValueType::AudioEmitter => 14,
        VisualValueType::AudioListener => 15,
        VisualValueType::Script => 16,
        VisualValueType::LookAt => 17,
        VisualValueType::EntityFollower => 18,
        VisualValueType::AnimatorState => 19,
        VisualValueType::InputModifiers => 20,
        VisualValueType::AudioStreamingConfig => 21,
        VisualValueType::RuntimeTuning => 22,
        VisualValueType::RuntimeConfig => 23,
        VisualValueType::RenderConfig => 24,
        VisualValueType::ShaderConstants => 25,
        VisualValueType::StreamingTuning => 26,
        VisualValueType::RenderPasses => 27,
        VisualValueType::GpuBudget => 28,
        VisualValueType::AssetBudgets => 29,
        VisualValueType::WindowSettings => 30,
        VisualValueType::Spline => 31,
        VisualValueType::Physics => 32,
        VisualValueType::PhysicsVelocity => 33,
        VisualValueType::PhysicsWorldDefaults => 34,
        VisualValueType::CharacterControllerOutput => 35,
        VisualValueType::DynamicComponentFields => 36,
        VisualValueType::DynamicFieldValue => 37,
        VisualValueType::PhysicsRayCastHit => 38,
        VisualValueType::PhysicsPointProjectionHit => 39,
        VisualValueType::PhysicsShapeCastHit => 40,
        VisualValueType::PhysicsQueryFilter => 41,
        VisualValueType::Any => 42,
    }
}

fn canonical_json_string(value: &JsonValue) -> String {
    serde_json::to_string(&canonicalize_json_value(value)).unwrap_or_else(|_| "null".to_string())
}

fn canonicalize_json_value(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(values) => {
            JsonValue::Array(values.iter().map(canonicalize_json_value).collect())
        }
        JsonValue::Object(values) => {
            let mut sorted_keys = values.keys().cloned().collect::<Vec<_>>();
            sorted_keys.sort();
            let mut object = JsonMap::new();
            for key in sorted_keys {
                if let Some(entry) = values.get(&key) {
                    object.insert(key, canonicalize_json_value(entry));
                }
            }
            JsonValue::Object(object)
        }
        _ => value.clone(),
    }
}

fn vec3_json(x: f64, y: f64, z: f64) -> JsonValue {
    let mut object = JsonMap::new();
    object.insert("x".to_string(), json_number(x));
    object.insert("y".to_string(), json_number(y));
    object.insert("z".to_string(), json_number(z));
    JsonValue::Object(object)
}

fn vec2_json(x: f64, y: f64) -> JsonValue {
    let mut object = JsonMap::new();
    object.insert("x".to_string(), json_number(x));
    object.insert("y".to_string(), json_number(y));
    JsonValue::Object(object)
}

fn vec4_json(x: f64, y: f64, z: f64, w: f64) -> JsonValue {
    let mut object = JsonMap::new();
    object.insert("x".to_string(), json_number(x));
    object.insert("y".to_string(), json_number(y));
    object.insert("z".to_string(), json_number(z));
    object.insert("w".to_string(), json_number(w));
    JsonValue::Object(object)
}

fn quat_json(x: f64, y: f64, z: f64, w: f64) -> JsonValue {
    let mut object = JsonMap::new();
    object.insert("x".to_string(), json_number(x));
    object.insert("y".to_string(), json_number(y));
    object.insert("z".to_string(), json_number(z));
    object.insert("w".to_string(), json_number(w));
    JsonValue::Object(object)
}

fn is_truthy(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null => false,
        JsonValue::Bool(value) => *value,
        JsonValue::Number(value) => value.as_f64().unwrap_or(0.0) != 0.0,
        JsonValue::String(value) => {
            let lower = value.trim().to_ascii_lowercase();
            !(lower.is_empty()
                || lower == "false"
                || lower == "0"
                || lower == "nil"
                || lower == "null")
        }
        JsonValue::Array(values) => !values.is_empty(),
        JsonValue::Object(values) => !values.is_empty(),
    }
}

fn coerce_json_to_f64(value: &JsonValue) -> Result<f64, String> {
    match value {
        JsonValue::Number(value) => value
            .as_f64()
            .ok_or_else(|| "Number is out of range".to_string()),
        JsonValue::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        JsonValue::String(value) => value
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("Cannot convert string '{}' to number", value)),
        JsonValue::Null => Ok(0.0),
        JsonValue::Array(_) | JsonValue::Object(_) => {
            Err("Cannot convert complex value to number".to_string())
        }
    }
}

fn coerce_json_to_visual_type(
    value: &JsonValue,
    value_type: VisualValueType,
) -> Result<JsonValue, String> {
    match value_type {
        VisualValueType::Any => Ok(value.clone()),
        VisualValueType::Bool => Ok(JsonValue::Bool(is_truthy(value))),
        VisualValueType::Number => Ok(json_number(coerce_json_to_f64(value)?)),
        VisualValueType::String => Ok(JsonValue::String(match value {
            JsonValue::String(text) => text.clone(),
            JsonValue::Null => String::new(),
            _ => json_to_log_string(value),
        })),
        VisualValueType::Entity => {
            let raw = coerce_json_to_f64(value)?;
            if !raw.is_finite() || raw < 0.0 {
                return Err("Entity ids must be finite non-negative numbers".to_string());
            }
            Ok(JsonValue::Number(JsonNumber::from(raw as u64)))
        }
        VisualValueType::Array => match value {
            JsonValue::Array(values) => Ok(JsonValue::Array(values.clone())),
            JsonValue::Null => Ok(JsonValue::Array(Vec::new())),
            JsonValue::String(text) => {
                coerce_json_to_visual_type(&parse_loose_literal(text), VisualValueType::Array)
            }
            _ => Err("Array values must be arrays or array literal strings".to_string()),
        },
        VisualValueType::Vec2 => {
            let (x, y) = coerce_json_to_vec2_components(value)?;
            let mut object = JsonMap::new();
            object.insert("x".to_string(), json_number(x));
            object.insert("y".to_string(), json_number(y));
            Ok(JsonValue::Object(object))
        }
        VisualValueType::Vec3 => {
            let (x, y, z) = coerce_json_to_vec3_components(value)?;
            Ok(vec3_json(x, y, z))
        }
        VisualValueType::Quat => {
            let (x, y, z, w) = coerce_json_to_quat_components(value)?;
            Ok(quat_json(x, y, z, w))
        }
        VisualValueType::Transform => {
            let object = coerce_json_to_loose_object(value, "Transform")?;
            let position = object
                .get("position")
                .cloned()
                .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
            let rotation = object
                .get("rotation")
                .cloned()
                .unwrap_or_else(|| quat_json(0.0, 0.0, 0.0, 1.0));
            let scale = object
                .get("scale")
                .cloned()
                .unwrap_or_else(|| vec3_json(1.0, 1.0, 1.0));
            let mut out = JsonMap::new();
            out.insert(
                "position".to_string(),
                coerce_json_to_visual_type(&position, VisualValueType::Vec3)?,
            );
            out.insert(
                "rotation".to_string(),
                coerce_json_to_visual_type(&rotation, VisualValueType::Quat)?,
            );
            out.insert(
                "scale".to_string(),
                coerce_json_to_visual_type(&scale, VisualValueType::Vec3)?,
            );
            Ok(JsonValue::Object(out))
        }
        VisualValueType::Camera => coerce_json_to_camera(value),
        VisualValueType::Light => coerce_json_to_light(value),
        VisualValueType::MeshRenderer => coerce_json_to_mesh_renderer(value),
        VisualValueType::SpriteRenderer => coerce_json_to_sprite_renderer(value),
        VisualValueType::Text2d => coerce_json_to_text2d(value),
        VisualValueType::AudioEmitter => coerce_json_to_audio_emitter(value),
        VisualValueType::AudioListener => coerce_json_to_audio_listener(value),
        VisualValueType::Script => coerce_json_to_script_ref(value),
        VisualValueType::LookAt => coerce_json_to_look_at(value),
        VisualValueType::EntityFollower => coerce_json_to_entity_follower(value),
        VisualValueType::AnimatorState => coerce_json_to_animator_state(value),
        VisualValueType::InputModifiers => coerce_json_to_input_modifiers(value),
        VisualValueType::AudioStreamingConfig => coerce_json_to_audio_streaming_config(value),
        VisualValueType::RuntimeTuning => coerce_json_to_default_object(
            value,
            "Runtime Tuning",
            default_literal_for_type(value_type),
        ),
        VisualValueType::RuntimeConfig => coerce_json_to_default_object(
            value,
            "Runtime Config",
            default_literal_for_type(value_type),
        ),
        VisualValueType::RenderConfig => coerce_json_to_schema_object(
            value,
            "Render Config",
            default_literal_for_type(value_type),
        ),
        VisualValueType::ShaderConstants => coerce_json_to_schema_object(
            value,
            "Shader Constants",
            default_literal_for_type(value_type),
        ),
        VisualValueType::StreamingTuning => coerce_json_to_schema_object(
            value,
            "Streaming Tuning",
            default_literal_for_type(value_type),
        ),
        VisualValueType::RenderPasses => coerce_json_to_default_object(
            value,
            "Render Passes",
            default_literal_for_type(value_type),
        ),
        VisualValueType::GpuBudget => {
            coerce_json_to_default_object(value, "GPU Budget", default_literal_for_type(value_type))
        }
        VisualValueType::AssetBudgets => coerce_json_to_default_object(
            value,
            "Asset Budgets",
            default_literal_for_type(value_type),
        ),
        VisualValueType::WindowSettings => coerce_json_to_default_object(
            value,
            "Window Settings",
            default_literal_for_type(value_type),
        ),
        VisualValueType::Spline => coerce_json_to_spline(value),
        VisualValueType::Physics => {
            coerce_json_to_default_object(value, "Physics", default_literal_for_type(value_type))
        }
        VisualValueType::PhysicsVelocity => coerce_json_to_physics_velocity(value),
        VisualValueType::PhysicsWorldDefaults => coerce_json_to_physics_world_defaults(value),
        VisualValueType::CharacterControllerOutput => coerce_json_to_character_output(value),
        VisualValueType::DynamicComponentFields => coerce_json_to_dynamic_component_fields(value),
        VisualValueType::DynamicFieldValue => coerce_json_to_dynamic_field_value(value),
        VisualValueType::PhysicsQueryFilter => coerce_json_to_physics_query_filter(value),
        VisualValueType::PhysicsRayCastHit => coerce_json_to_ray_cast_hit(value),
        VisualValueType::PhysicsPointProjectionHit => coerce_json_to_point_projection_hit(value),
        VisualValueType::PhysicsShapeCastHit => coerce_json_to_shape_cast_hit(value),
    }
}

fn coerce_json_to_visual_type_with_array_item(
    value: &JsonValue,
    value_type: VisualValueType,
    array_item_type: Option<VisualValueType>,
) -> Result<JsonValue, String> {
    if value_type != VisualValueType::Array {
        return coerce_json_to_visual_type(value, value_type);
    }

    let array = coerce_json_to_visual_type(value, VisualValueType::Array)?;
    let mut item_type = array_item_type;
    normalize_array_item_type(VisualValueType::Array, &mut item_type);
    let item_type = item_type.unwrap_or(default_array_item_type());

    let values = array.as_array().cloned().unwrap_or_default();
    let mut normalized = Vec::with_capacity(values.len());
    for entry in values {
        normalized.push(coerce_json_to_visual_type(&entry, item_type)?);
    }
    Ok(JsonValue::Array(normalized))
}

fn normalize_literal_for_data_type(
    value: &str,
    value_type: VisualValueType,
    array_item_type: Option<VisualValueType>,
) -> String {
    let parsed = parse_loose_literal(value);
    let coerced = coerce_json_to_visual_type_with_array_item(&parsed, value_type, array_item_type)
        .unwrap_or_else(|_| parse_loose_literal(default_literal_for_type(value_type)));
    literal_string_for_value_type(&coerced, value_type)
}

fn coerce_json_to_loose_object(
    value: &JsonValue,
    type_name: &str,
) -> Result<JsonMap<String, JsonValue>, String> {
    let parsed = match value {
        JsonValue::String(text) => parse_loose_literal(text),
        JsonValue::Null => JsonValue::Object(JsonMap::new()),
        other => other.clone(),
    };
    match parsed {
        JsonValue::Object(object) => Ok(object),
        _ => Err(format!("{} values must be structured objects", type_name)),
    }
}

fn coerce_json_to_default_object(
    value: &JsonValue,
    type_name: &str,
    default_literal: &str,
) -> Result<JsonValue, String> {
    let mut merged = parse_json_object_literal(default_literal);
    let input = coerce_json_to_loose_object(value, type_name)?;
    for (key, value) in input {
        merged.insert(key, value);
    }
    Ok(JsonValue::Object(merged))
}

fn strip_internal_schema_fields(value: &mut JsonValue) {
    match value {
        JsonValue::Object(object) => strip_internal_schema_fields_from_object(object),
        JsonValue::Array(values) => {
            for child in values {
                strip_internal_schema_fields(child);
            }
        }
        _ => {}
    }
}

fn strip_internal_schema_fields_from_object(object: &mut JsonMap<String, JsonValue>) {
    let keys = object.keys().cloned().collect::<Vec<_>>();
    for key in keys {
        if key.starts_with('_') {
            object.remove(&key);
            continue;
        }
        if let Some(child) = object.get_mut(&key) {
            strip_internal_schema_fields(child);
        }
    }
}

fn parse_schema_object_literal(default_literal: &str) -> JsonMap<String, JsonValue> {
    let mut schema = parse_json_object_literal(default_literal);
    strip_internal_schema_fields_from_object(&mut schema);
    schema
}

fn coerce_json_to_schema_object(
    value: &JsonValue,
    type_name: &str,
    schema_literal: &str,
) -> Result<JsonValue, String> {
    let schema = parse_schema_object_literal(schema_literal);
    let input = coerce_json_to_loose_object(value, type_name)?;
    Ok(JsonValue::Object(coerce_json_map_to_schema(
        input, &schema, type_name,
    )?))
}

fn coerce_json_map_to_schema(
    input: JsonMap<String, JsonValue>,
    schema: &JsonMap<String, JsonValue>,
    type_name: &str,
) -> Result<JsonMap<String, JsonValue>, String> {
    let mut out = JsonMap::new();
    for (key, value) in input {
        if key.starts_with('_') {
            continue;
        }
        let Some(schema_value) = schema.get(&key) else {
            return Err(format!("{} field '{}' is not supported", type_name, key));
        };
        let field_path = format!("{}.{}", type_name, key);
        out.insert(
            key,
            coerce_json_to_schema_value(&value, schema_value, &field_path)?,
        );
    }
    Ok(out)
}

fn coerce_json_to_schema_value(
    value: &JsonValue,
    schema: &JsonValue,
    field_path: &str,
) -> Result<JsonValue, String> {
    match schema {
        JsonValue::Bool(_) => Ok(JsonValue::Bool(is_truthy(value))),
        JsonValue::Number(_) => match value {
            JsonValue::Number(number) => Ok(JsonValue::Number(number.clone())),
            JsonValue::Bool(flag) => Ok(JsonValue::Number(JsonNumber::from(if *flag {
                1
            } else {
                0
            }))),
            JsonValue::String(text) => {
                let trimmed = text.trim();
                if let Ok(parsed) = trimmed.parse::<i64>() {
                    return Ok(JsonValue::Number(JsonNumber::from(parsed)));
                }
                if let Ok(parsed) = trimmed.parse::<u64>() {
                    return Ok(JsonValue::Number(JsonNumber::from(parsed)));
                }
                let parsed = trimmed
                    .parse::<f64>()
                    .map_err(|_| format!("{} must be a number", field_path))?;
                Ok(json_number(parsed))
            }
            JsonValue::Null => Ok(JsonValue::Number(JsonNumber::from(0))),
            JsonValue::Array(_) | JsonValue::Object(_) => {
                Err(format!("{} must be a number", field_path))
            }
        },
        JsonValue::String(_) => match value {
            JsonValue::String(text) => Ok(JsonValue::String(text.clone())),
            _ => Err(format!("{} must be a string", field_path)),
        },
        JsonValue::Array(schema_items) => {
            let source = match value {
                JsonValue::Array(values) => values.clone(),
                JsonValue::String(text) => match parse_loose_literal(text) {
                    JsonValue::Array(values) => values,
                    _ => return Err(format!("{} must be an array", field_path)),
                },
                _ => return Err(format!("{} must be an array", field_path)),
            };

            if schema_items.is_empty() {
                return Ok(JsonValue::Array(source));
            }

            let mut normalized = Vec::with_capacity(source.len());
            if schema_items.len() == 1 {
                for (index, entry) in source.iter().enumerate() {
                    let entry_path = format!("{}[{}]", field_path, index);
                    normalized.push(coerce_json_to_schema_value(
                        entry,
                        &schema_items[0],
                        &entry_path,
                    )?);
                }
                return Ok(JsonValue::Array(normalized));
            }

            if source.len() != schema_items.len() {
                return Err(format!(
                    "{} must contain exactly {} item(s)",
                    field_path,
                    schema_items.len()
                ));
            }

            for (index, entry) in source.iter().enumerate() {
                let entry_path = format!("{}[{}]", field_path, index);
                normalized.push(coerce_json_to_schema_value(
                    entry,
                    &schema_items[index],
                    &entry_path,
                )?);
            }
            Ok(JsonValue::Array(normalized))
        }
        JsonValue::Object(schema_object) => {
            let input = coerce_json_to_loose_object(value, field_path)?;
            Ok(JsonValue::Object(coerce_json_map_to_schema(
                input,
                schema_object,
                field_path,
            )?))
        }
        JsonValue::Null => Ok(JsonValue::Null),
    }
}

fn coerce_json_to_camera(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Camera")?;
    let mut out = JsonMap::new();
    let fov = object
        .get("fov_y_rad")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(1.0);
    let aspect = object
        .get("aspect_ratio")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(16.0 / 9.0);
    let near = object
        .get("near_plane")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.1)
        .max(0.0001);
    let far = object
        .get("far_plane")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(2000.0)
        .max(near);
    let active = object.get("active").map(is_truthy).unwrap_or(false);
    out.insert("fov_y_rad".to_string(), json_number(fov));
    out.insert("aspect_ratio".to_string(), json_number(aspect));
    out.insert("near_plane".to_string(), json_number(near));
    out.insert("far_plane".to_string(), json_number(far));
    out.insert("active".to_string(), JsonValue::Bool(active));
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_light(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Light")?;
    let mut out = JsonMap::new();
    let light_type = object
        .get("type")
        .and_then(JsonValue::as_str)
        .unwrap_or("Point")
        .to_string();
    let color = object
        .get("color")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(1.0, 1.0, 1.0));
    let intensity = object
        .get("intensity")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(10.0)
        .max(0.0);
    let angle = object
        .get("angle")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(45.0_f64.to_radians())
        .max(0.0);
    out.insert("type".to_string(), JsonValue::String(light_type.clone()));
    out.insert("color".to_string(), color);
    out.insert("intensity".to_string(), json_number(intensity));
    if light_type.eq_ignore_ascii_case("spot") {
        out.insert("angle".to_string(), json_number(angle));
    }
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_mesh_renderer(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Mesh Renderer")?;
    let mut out = JsonMap::new();
    let source = object
        .get("source")
        .and_then(JsonValue::as_str)
        .unwrap_or("Cube")
        .to_string();
    let material = object
        .get("material")
        .and_then(JsonValue::as_str)
        .map(|text| text.to_string())
        .unwrap_or_default();
    let casts_shadow = object.get("casts_shadow").map(is_truthy).unwrap_or(true);
    let visible = object.get("visible").map(is_truthy).unwrap_or(true);
    out.insert("source".to_string(), JsonValue::String(source));
    if !material.trim().is_empty() {
        out.insert("material".to_string(), JsonValue::String(material));
    }
    out.insert("casts_shadow".to_string(), JsonValue::Bool(casts_shadow));
    out.insert("visible".to_string(), JsonValue::Bool(visible));
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_finite_f64(value: &JsonValue, field_name: &str) -> Result<f64, String> {
    let number = coerce_json_to_f64(value)?;
    if !number.is_finite() {
        return Err(format!("{} must be a finite number", field_name));
    }
    Ok(number)
}

fn coerce_json_to_u32(value: &JsonValue, field_name: &str) -> Result<u32, String> {
    let number = coerce_json_to_finite_f64(value, field_name)?;
    if number < 0.0 {
        return Err(format!("{} must be non-negative", field_name));
    }
    Ok(number.round().clamp(0.0, u32::MAX as f64) as u32)
}

fn coerce_json_mode_index(value: &JsonValue) -> Option<i64> {
    match value {
        JsonValue::Number(number) => number
            .as_i64()
            .or_else(|| number.as_u64().and_then(|value| i64::try_from(value).ok()))
            .or_else(|| {
                number.as_f64().and_then(|value| {
                    if value.is_finite() {
                        Some(value as i64)
                    } else {
                        None
                    }
                })
            }),
        _ => None,
    }
}

fn coerce_sprite_space_value(value: &JsonValue) -> Option<&'static str> {
    match value {
        JsonValue::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "screen" => Some("screen"),
            "world" => Some("world"),
            _ => None,
        },
        _ => match coerce_json_mode_index(value) {
            Some(0) => Some("screen"),
            Some(1) => Some("world"),
            _ => None,
        },
    }
}

fn coerce_sprite_blend_mode_value(value: &JsonValue) -> Option<&'static str> {
    match value {
        JsonValue::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "alpha" => Some("alpha"),
            "premultiplied" | "premul" | "premult" => Some("premultiplied"),
            "additive" | "add" => Some("additive"),
            _ => None,
        },
        _ => match coerce_json_mode_index(value) {
            Some(0) => Some("alpha"),
            Some(1) => Some("premultiplied"),
            Some(2) => Some("additive"),
            _ => None,
        },
    }
}

fn coerce_sprite_playback_value(value: &JsonValue) -> Option<&'static str> {
    match value {
        JsonValue::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "loop" | "repeat" => Some("loop"),
            "once" | "one_shot" | "oneshot" => Some("once"),
            "pingpong" | "ping_pong" | "ping-pong" | "pong" => Some("pingpong"),
            _ => None,
        },
        _ => match coerce_json_mode_index(value) {
            Some(0) => Some("loop"),
            Some(1) => Some("once"),
            Some(2) => Some("pingpong"),
            _ => None,
        },
    }
}

fn coerce_text_font_style_value(value: &JsonValue) -> Option<&'static str> {
    match value {
        JsonValue::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "normal" => Some("normal"),
            "italic" => Some("italic"),
            "oblique" => Some("oblique"),
            _ => None,
        },
        _ => match coerce_json_mode_index(value) {
            Some(0) => Some("normal"),
            Some(1) => Some("italic"),
            Some(2) => Some("oblique"),
            _ => None,
        },
    }
}

fn coerce_text_align_h_value(value: &JsonValue) -> Option<&'static str> {
    match value {
        JsonValue::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "left" => Some("left"),
            "center" | "centre" => Some("center"),
            "right" => Some("right"),
            _ => None,
        },
        _ => match coerce_json_mode_index(value) {
            Some(0) => Some("left"),
            Some(1) => Some("center"),
            Some(2) => Some("right"),
            _ => None,
        },
    }
}

fn coerce_text_align_v_value(value: &JsonValue) -> Option<&'static str> {
    match value {
        JsonValue::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "top" => Some("top"),
            "center" | "centre" => Some("center"),
            "bottom" => Some("bottom"),
            "baseline" => Some("baseline"),
            _ => None,
        },
        _ => match coerce_json_mode_index(value) {
            Some(0) => Some("top"),
            Some(1) => Some("center"),
            Some(2) => Some("bottom"),
            Some(3) => Some("baseline"),
            _ => None,
        },
    }
}

fn coerce_json_to_sprite_texture_paths(
    value: &JsonValue,
    field_name: &str,
) -> Result<Vec<String>, String> {
    match value {
        JsonValue::Null => Ok(Vec::new()),
        JsonValue::String(path) => {
            let trimmed = path.trim();
            Ok(vec![trimmed.to_string()])
        }
        JsonValue::Array(paths) => {
            let mut normalized = Vec::with_capacity(paths.len());
            for (index, path) in paths.iter().enumerate() {
                match path {
                    JsonValue::Null => normalized.push(String::new()),
                    JsonValue::String(text) => {
                        let trimmed = text.trim();
                        normalized.push(trimmed.to_string());
                    }
                    _ => {
                        return Err(format!("{}[{}] must be a string", field_name, index));
                    }
                }
            }
            Ok(normalized)
        }
        _ => Err(format!(
            "{} must be a string, array of strings, or null",
            field_name
        )),
    }
}

fn coerce_json_to_sprite_sheet_animation_patch(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Sprite Sheet Animation")?;
    let mut out = JsonMap::new();

    if let Some(enabled) = object.get("enabled") {
        out.insert("enabled".to_string(), JsonValue::Bool(is_truthy(enabled)));
    }
    if let Some(columns) = object.get("columns") {
        out.insert(
            "columns".to_string(),
            JsonValue::Number(JsonNumber::from(
                coerce_json_to_u32(columns, "Sprite Sheet Animation.columns")?.max(1),
            )),
        );
    }
    if let Some(rows) = object.get("rows") {
        out.insert(
            "rows".to_string(),
            JsonValue::Number(JsonNumber::from(
                coerce_json_to_u32(rows, "Sprite Sheet Animation.rows")?.max(1),
            )),
        );
    }
    if let Some(start_frame) = object.get("start_frame") {
        out.insert(
            "start_frame".to_string(),
            JsonValue::Number(JsonNumber::from(coerce_json_to_u32(
                start_frame,
                "Sprite Sheet Animation.start_frame",
            )?)),
        );
    }
    if let Some(frame_count) = object.get("frame_count") {
        out.insert(
            "frame_count".to_string(),
            JsonValue::Number(JsonNumber::from(coerce_json_to_u32(
                frame_count,
                "Sprite Sheet Animation.frame_count",
            )?)),
        );
    }
    if let Some(fps) = object.get("fps") {
        out.insert(
            "fps".to_string(),
            json_number(coerce_json_to_finite_f64(
                fps,
                "Sprite Sheet Animation.fps",
            )?),
        );
    }
    if let Some(playback) = object.get("playback") {
        let Some(playback) = coerce_sprite_playback_value(playback) else {
            return Err("Sprite Sheet Animation.playback is invalid".to_string());
        };
        out.insert(
            "playback".to_string(),
            JsonValue::String(playback.to_string()),
        );
    }
    if let Some(phase) = object.get("phase") {
        out.insert(
            "phase".to_string(),
            json_number(coerce_json_to_finite_f64(
                phase,
                "Sprite Sheet Animation.phase",
            )?),
        );
    }
    if let Some(paused) = object.get("paused") {
        out.insert("paused".to_string(), JsonValue::Bool(is_truthy(paused)));
    }
    if let Some(paused_frame) = object.get("paused_frame") {
        out.insert(
            "paused_frame".to_string(),
            JsonValue::Number(JsonNumber::from(coerce_json_to_u32(
                paused_frame,
                "Sprite Sheet Animation.paused_frame",
            )?)),
        );
    }
    if let Some(flip_x) = object.get("flip_x") {
        out.insert("flip_x".to_string(), JsonValue::Bool(is_truthy(flip_x)));
    }
    if let Some(flip_y) = object.get("flip_y") {
        out.insert("flip_y".to_string(), JsonValue::Bool(is_truthy(flip_y)));
    }
    if let Some(frame_uv_inset) = object.get("frame_uv_inset") {
        let (x, y) = coerce_json_to_vec2_components(frame_uv_inset)?;
        out.insert(
            "frame_uv_inset".to_string(),
            vec2_json(x.clamp(0.0, 0.49), y.clamp(0.0, 0.49)),
        );
    }

    Ok(JsonValue::Object(out))
}

fn coerce_json_to_sprite_image_sequence_patch(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Sprite Image Sequence")?;
    let mut out = JsonMap::new();

    if let Some(enabled) = object.get("enabled") {
        out.insert("enabled".to_string(), JsonValue::Bool(is_truthy(enabled)));
    }
    if let Some(start_frame) = object.get("start_frame") {
        out.insert(
            "start_frame".to_string(),
            JsonValue::Number(JsonNumber::from(coerce_json_to_u32(
                start_frame,
                "Sprite Image Sequence.start_frame",
            )?)),
        );
    }
    if let Some(frame_count) = object.get("frame_count") {
        out.insert(
            "frame_count".to_string(),
            JsonValue::Number(JsonNumber::from(coerce_json_to_u32(
                frame_count,
                "Sprite Image Sequence.frame_count",
            )?)),
        );
    }
    if let Some(fps) = object.get("fps") {
        out.insert(
            "fps".to_string(),
            json_number(coerce_json_to_finite_f64(fps, "Sprite Image Sequence.fps")?),
        );
    }
    if let Some(playback) = object.get("playback") {
        let Some(playback) = coerce_sprite_playback_value(playback) else {
            return Err("Sprite Image Sequence.playback is invalid".to_string());
        };
        out.insert(
            "playback".to_string(),
            JsonValue::String(playback.to_string()),
        );
    }
    if let Some(phase) = object.get("phase") {
        out.insert(
            "phase".to_string(),
            json_number(coerce_json_to_finite_f64(
                phase,
                "Sprite Image Sequence.phase",
            )?),
        );
    }
    if let Some(paused) = object.get("paused") {
        out.insert("paused".to_string(), JsonValue::Bool(is_truthy(paused)));
    }
    if let Some(paused_frame) = object.get("paused_frame") {
        out.insert(
            "paused_frame".to_string(),
            JsonValue::Number(JsonNumber::from(coerce_json_to_u32(
                paused_frame,
                "Sprite Image Sequence.paused_frame",
            )?)),
        );
    }
    if let Some(flip_x) = object.get("flip_x") {
        out.insert("flip_x".to_string(), JsonValue::Bool(is_truthy(flip_x)));
    }
    if let Some(flip_y) = object.get("flip_y") {
        out.insert("flip_y".to_string(), JsonValue::Bool(is_truthy(flip_y)));
    }

    let mut textures: Option<Vec<String>> = None;
    for (key, field) in [
        ("textures", "Sprite Image Sequence.textures"),
        ("texture_paths", "Sprite Image Sequence.texture_paths"),
        ("frames", "Sprite Image Sequence.frames"),
    ] {
        if let Some(value) = object.get(key) {
            textures = Some(coerce_json_to_sprite_texture_paths(value, field)?);
        }
    }
    if let Some(textures) = textures {
        let values = JsonValue::Array(textures.into_iter().map(JsonValue::String).collect());
        out.insert("textures".to_string(), values.clone());
        out.insert("texture_paths".to_string(), values.clone());
        out.insert("frames".to_string(), values);
    }

    Ok(JsonValue::Object(out))
}

fn coerce_json_to_sprite_renderer(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Sprite Renderer")?;
    let mut out = JsonMap::new();

    if let Some(color) = object.get("color") {
        let (x, y, z, w) = coerce_json_to_vec4_components(color)?;
        out.insert("color".to_string(), vec4_json(x, y, z, w));
    }
    if let Some(texture_id) = object.get("texture_id") {
        if texture_id.is_null() {
            out.insert("texture_id".to_string(), JsonValue::Null);
        } else {
            let id = coerce_json_to_finite_f64(texture_id, "Sprite Renderer.texture_id")?;
            if id < 0.0 {
                return Err("Sprite Renderer.texture_id must be non-negative".to_string());
            }
            out.insert(
                "texture_id".to_string(),
                JsonValue::Number(JsonNumber::from(id.round() as u64)),
            );
        }
    }
    if let Some(texture) = object.get("texture") {
        match texture {
            JsonValue::Null => {
                out.insert("texture".to_string(), JsonValue::Null);
            }
            JsonValue::String(path) => {
                let trimmed = path.trim();
                out.insert(
                    "texture".to_string(),
                    if trimmed.is_empty() {
                        JsonValue::Null
                    } else {
                        JsonValue::String(trimmed.to_string())
                    },
                );
            }
            _ => {
                return Err("Sprite Renderer.texture must be a string or null".to_string());
            }
        }
    }
    if let Some(uv_min) = object.get("uv_min") {
        out.insert(
            "uv_min".to_string(),
            coerce_json_to_visual_type(uv_min, VisualValueType::Vec2)?,
        );
    }
    if let Some(uv_max) = object.get("uv_max") {
        out.insert(
            "uv_max".to_string(),
            coerce_json_to_visual_type(uv_max, VisualValueType::Vec2)?,
        );
    }
    if let Some(sheet_animation) = object.get("sheet_animation") {
        out.insert(
            "sheet_animation".to_string(),
            if sheet_animation.is_null() {
                JsonValue::Null
            } else {
                coerce_json_to_sprite_sheet_animation_patch(sheet_animation)?
            },
        );
    }
    if let Some(sheet_animation) = object.get("sheet") {
        out.insert(
            "sheet".to_string(),
            if sheet_animation.is_null() {
                JsonValue::Null
            } else {
                coerce_json_to_sprite_sheet_animation_patch(sheet_animation)?
            },
        );
    }
    if let Some(image_sequence) = object.get("image_sequence") {
        out.insert(
            "image_sequence".to_string(),
            if image_sequence.is_null() {
                JsonValue::Null
            } else {
                coerce_json_to_sprite_image_sequence_patch(image_sequence)?
            },
        );
    }
    if let Some(image_sequence) = object.get("sequence") {
        out.insert(
            "sequence".to_string(),
            if image_sequence.is_null() {
                JsonValue::Null
            } else {
                coerce_json_to_sprite_image_sequence_patch(image_sequence)?
            },
        );
    }
    if let Some(pivot) = object.get("pivot") {
        out.insert(
            "pivot".to_string(),
            coerce_json_to_visual_type(pivot, VisualValueType::Vec2)?,
        );
    }
    if let Some(clip_rect) = object.get("clip_rect") {
        out.insert(
            "clip_rect".to_string(),
            if clip_rect.is_null() {
                JsonValue::Null
            } else {
                let (x, y, z, w) = coerce_json_to_vec4_components(clip_rect)?;
                vec4_json(x, y, z, w)
            },
        );
    }
    if let Some(layer) = object.get("layer") {
        out.insert(
            "layer".to_string(),
            json_number(coerce_json_to_finite_f64(layer, "Sprite Renderer.layer")?),
        );
    }
    if let Some(space) = object.get("space") {
        let Some(space) = coerce_sprite_space_value(space) else {
            return Err("Sprite Renderer.space is invalid".to_string());
        };
        out.insert("space".to_string(), JsonValue::String(space.to_string()));
    }
    if let Some(blend_mode) = object.get("blend_mode") {
        let Some(blend_mode) = coerce_sprite_blend_mode_value(blend_mode) else {
            return Err("Sprite Renderer.blend_mode is invalid".to_string());
        };
        out.insert(
            "blend_mode".to_string(),
            JsonValue::String(blend_mode.to_string()),
        );
    }
    if let Some(billboard) = object.get("billboard") {
        out.insert(
            "billboard".to_string(),
            JsonValue::Bool(is_truthy(billboard)),
        );
    }
    if let Some(visible) = object.get("visible") {
        out.insert("visible".to_string(), JsonValue::Bool(is_truthy(visible)));
    }
    if let Some(pick_id) = object.get("pick_id") {
        if pick_id.is_null() {
            out.insert("pick_id".to_string(), JsonValue::Null);
        } else {
            let id = coerce_json_to_finite_f64(pick_id, "Sprite Renderer.pick_id")?;
            if id < 0.0 {
                return Err("Sprite Renderer.pick_id must be non-negative".to_string());
            }
            out.insert(
                "pick_id".to_string(),
                JsonValue::Number(JsonNumber::from(id.round() as u64)),
            );
        }
    }

    Ok(JsonValue::Object(out))
}

fn coerce_json_to_text2d(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Text 2D")?;
    let mut out = JsonMap::new();

    if let Some(text) = object.get("text") {
        match text {
            JsonValue::String(text) => {
                out.insert("text".to_string(), JsonValue::String(text.clone()));
            }
            _ => return Err("Text 2D.text must be a string".to_string()),
        }
    }
    if let Some(color) = object.get("color") {
        let (x, y, z, w) = coerce_json_to_vec4_components(color)?;
        out.insert("color".to_string(), vec4_json(x, y, z, w));
    }
    if let Some(font_path) = object.get("font_path") {
        match font_path {
            JsonValue::Null => {
                out.insert("font_path".to_string(), JsonValue::Null);
            }
            JsonValue::String(path) => {
                let trimmed = path.trim();
                out.insert(
                    "font_path".to_string(),
                    if trimmed.is_empty() {
                        JsonValue::Null
                    } else {
                        JsonValue::String(trimmed.to_string())
                    },
                );
            }
            _ => return Err("Text 2D.font_path must be a string or null".to_string()),
        }
    }
    if let Some(font_family) = object.get("font_family") {
        match font_family {
            JsonValue::Null => {
                out.insert("font_family".to_string(), JsonValue::Null);
            }
            JsonValue::String(family) => {
                let trimmed = family.trim();
                out.insert(
                    "font_family".to_string(),
                    if trimmed.is_empty() {
                        JsonValue::Null
                    } else {
                        JsonValue::String(trimmed.to_string())
                    },
                );
            }
            _ => return Err("Text 2D.font_family must be a string or null".to_string()),
        }
    }
    if let Some(font_size) = object.get("font_size") {
        out.insert(
            "font_size".to_string(),
            json_number(coerce_json_to_finite_f64(font_size, "Text 2D.font_size")?.max(0.01)),
        );
    }
    if let Some(font_weight) = object.get("font_weight") {
        out.insert(
            "font_weight".to_string(),
            json_number(
                coerce_json_to_finite_f64(font_weight, "Text 2D.font_weight")?.clamp(1.0, 1000.0),
            ),
        );
    }
    if let Some(font_width) = object.get("font_width") {
        out.insert(
            "font_width".to_string(),
            json_number(
                coerce_json_to_finite_f64(font_width, "Text 2D.font_width")?.clamp(0.25, 4.0),
            ),
        );
    }
    if let Some(font_style) = object.get("font_style") {
        let Some(font_style) = coerce_text_font_style_value(font_style) else {
            return Err("Text 2D.font_style is invalid".to_string());
        };
        out.insert(
            "font_style".to_string(),
            JsonValue::String(font_style.to_string()),
        );
    }
    if let Some(line_height_scale) = object.get("line_height_scale") {
        out.insert(
            "line_height_scale".to_string(),
            json_number(
                coerce_json_to_finite_f64(line_height_scale, "Text 2D.line_height_scale")?.max(0.1),
            ),
        );
    }
    if let Some(letter_spacing) = object.get("letter_spacing") {
        out.insert(
            "letter_spacing".to_string(),
            json_number(coerce_json_to_finite_f64(
                letter_spacing,
                "Text 2D.letter_spacing",
            )?),
        );
    }
    if let Some(word_spacing) = object.get("word_spacing") {
        out.insert(
            "word_spacing".to_string(),
            json_number(coerce_json_to_finite_f64(
                word_spacing,
                "Text 2D.word_spacing",
            )?),
        );
    }
    if let Some(underline) = object.get("underline") {
        out.insert(
            "underline".to_string(),
            JsonValue::Bool(is_truthy(underline)),
        );
    }
    if let Some(strikethrough) = object.get("strikethrough") {
        out.insert(
            "strikethrough".to_string(),
            JsonValue::Bool(is_truthy(strikethrough)),
        );
    }
    if let Some(max_width) = object.get("max_width") {
        out.insert(
            "max_width".to_string(),
            if max_width.is_null() {
                JsonValue::Null
            } else {
                let width = coerce_json_to_finite_f64(max_width, "Text 2D.max_width")?;
                if width <= 0.0 {
                    return Err("Text 2D.max_width must be positive or null".to_string());
                }
                json_number(width)
            },
        );
    }
    if let Some(align_h) = object.get("align_h") {
        let Some(align_h) = coerce_text_align_h_value(align_h) else {
            return Err("Text 2D.align_h is invalid".to_string());
        };
        out.insert(
            "align_h".to_string(),
            JsonValue::String(align_h.to_string()),
        );
    }
    if let Some(align_v) = object.get("align_v") {
        let Some(align_v) = coerce_text_align_v_value(align_v) else {
            return Err("Text 2D.align_v is invalid".to_string());
        };
        out.insert(
            "align_v".to_string(),
            JsonValue::String(align_v.to_string()),
        );
    }
    if let Some(space) = object.get("space") {
        let Some(space) = coerce_sprite_space_value(space) else {
            return Err("Text 2D.space is invalid".to_string());
        };
        out.insert("space".to_string(), JsonValue::String(space.to_string()));
    }
    if let Some(blend_mode) = object.get("blend_mode") {
        let Some(blend_mode) = coerce_sprite_blend_mode_value(blend_mode) else {
            return Err("Text 2D.blend_mode is invalid".to_string());
        };
        out.insert(
            "blend_mode".to_string(),
            JsonValue::String(blend_mode.to_string()),
        );
    }
    if let Some(billboard) = object.get("billboard") {
        out.insert(
            "billboard".to_string(),
            JsonValue::Bool(is_truthy(billboard)),
        );
    }
    if let Some(visible) = object.get("visible") {
        out.insert("visible".to_string(), JsonValue::Bool(is_truthy(visible)));
    }
    if let Some(layer) = object.get("layer") {
        out.insert(
            "layer".to_string(),
            json_number(coerce_json_to_finite_f64(layer, "Text 2D.layer")?),
        );
    }
    if let Some(clip_rect) = object.get("clip_rect") {
        out.insert(
            "clip_rect".to_string(),
            if clip_rect.is_null() {
                JsonValue::Null
            } else {
                let (x, y, z, w) = coerce_json_to_vec4_components(clip_rect)?;
                vec4_json(x, y, z, w)
            },
        );
    }
    if let Some(pick_id) = object.get("pick_id") {
        if pick_id.is_null() {
            out.insert("pick_id".to_string(), JsonValue::Null);
        } else {
            let id = coerce_json_to_finite_f64(pick_id, "Text 2D.pick_id")?;
            if id < 0.0 {
                return Err("Text 2D.pick_id must be non-negative".to_string());
            }
            out.insert(
                "pick_id".to_string(),
                JsonValue::Number(JsonNumber::from(id.round() as u64)),
            );
        }
    }

    Ok(JsonValue::Object(out))
}

fn coerce_json_to_audio_emitter(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Audio Emitter")?;
    let mut out = JsonMap::new();
    let path = object.get("path").cloned().unwrap_or(JsonValue::Null);
    let streaming = object.get("streaming").map(is_truthy).unwrap_or(false);
    let bus = object
        .get("bus")
        .and_then(JsonValue::as_str)
        .unwrap_or("Master")
        .to_string();
    let volume = object
        .get("volume")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(1.0)
        .max(0.0);
    let pitch = object
        .get("pitch")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(1.0)
        .max(0.001);
    let looping = object.get("looping").map(is_truthy).unwrap_or(false);
    let spatial = object.get("spatial").map(is_truthy).unwrap_or(true);
    let min_distance = object
        .get("min_distance")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(1.0)
        .max(0.0);
    let max_distance = object
        .get("max_distance")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(30.0)
        .max(min_distance);
    let rolloff = object
        .get("rolloff")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(1.0)
        .max(0.0);
    let spatial_blend = object
        .get("spatial_blend")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let playback_state = object
        .get("playback_state")
        .and_then(JsonValue::as_str)
        .unwrap_or("Stopped")
        .to_string();
    let play_on_spawn = object.get("play_on_spawn").map(is_truthy).unwrap_or(false);
    out.insert(
        "path".to_string(),
        match path {
            JsonValue::String(text) if !text.trim().is_empty() => JsonValue::String(text),
            JsonValue::String(_) | JsonValue::Null => JsonValue::Null,
            other => JsonValue::String(json_to_log_string(&other)),
        },
    );
    out.insert("streaming".to_string(), JsonValue::Bool(streaming));
    out.insert("bus".to_string(), JsonValue::String(bus));
    out.insert("volume".to_string(), json_number(volume));
    out.insert("pitch".to_string(), json_number(pitch));
    out.insert("looping".to_string(), JsonValue::Bool(looping));
    out.insert("spatial".to_string(), JsonValue::Bool(spatial));
    out.insert("min_distance".to_string(), json_number(min_distance));
    out.insert("max_distance".to_string(), json_number(max_distance));
    out.insert("rolloff".to_string(), json_number(rolloff));
    out.insert("spatial_blend".to_string(), json_number(spatial_blend));
    out.insert("play_on_spawn".to_string(), JsonValue::Bool(play_on_spawn));
    out.insert(
        "playback_state".to_string(),
        JsonValue::String(playback_state),
    );
    if let Some(clip_id) = object.get("clip_id") {
        if let Ok(raw) = coerce_json_to_f64(clip_id) {
            if raw.is_finite() && raw >= 0.0 {
                out.insert(
                    "clip_id".to_string(),
                    JsonValue::Number(JsonNumber::from(raw as u64)),
                );
            }
        }
    }
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_audio_listener(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Audio Listener")?;
    let mut out = JsonMap::new();
    out.insert(
        "enabled".to_string(),
        JsonValue::Bool(object.get("enabled").map(is_truthy).unwrap_or(true)),
    );
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_script_ref(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Script")?;
    let mut out = JsonMap::new();
    let path = object
        .get("path")
        .and_then(JsonValue::as_str)
        .unwrap_or("")
        .to_string();
    let language = object
        .get("language")
        .and_then(JsonValue::as_str)
        .unwrap_or("lua")
        .to_string();
    out.insert("path".to_string(), JsonValue::String(path));
    out.insert("language".to_string(), JsonValue::String(language));
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_look_at(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Look At")?;
    let mut out = JsonMap::new();

    let target_entity = object
        .get("target_entity")
        .cloned()
        .unwrap_or(JsonValue::Null);
    let target_entity = if target_entity.is_null() {
        JsonValue::Null
    } else {
        coerce_json_to_visual_type(&target_entity, VisualValueType::Entity)?
    };
    let target_offset = object
        .get("target_offset")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let up = object
        .get("up")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 1.0, 0.0));
    let offset_in_target_space = object
        .get("offset_in_target_space")
        .map(is_truthy)
        .unwrap_or(false);
    let rotation_smooth_time = object
        .get("rotation_smooth_time")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.0)
        .max(0.0);

    out.insert("target_entity".to_string(), target_entity);
    out.insert("target_offset".to_string(), target_offset);
    out.insert(
        "offset_in_target_space".to_string(),
        JsonValue::Bool(offset_in_target_space),
    );
    out.insert("up".to_string(), up);
    out.insert(
        "rotation_smooth_time".to_string(),
        json_number(rotation_smooth_time),
    );
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_entity_follower(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Entity Follower")?;
    let mut out = JsonMap::new();

    let target_entity = object
        .get("target_entity")
        .cloned()
        .unwrap_or(JsonValue::Null);
    let target_entity = if target_entity.is_null() {
        JsonValue::Null
    } else {
        coerce_json_to_visual_type(&target_entity, VisualValueType::Entity)?
    };
    let position_offset = object
        .get("position_offset")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let offset_in_target_space = object
        .get("offset_in_target_space")
        .map(is_truthy)
        .unwrap_or(false);
    let follow_rotation = object.get("follow_rotation").map(is_truthy).unwrap_or(true);
    let position_smooth_time = object
        .get("position_smooth_time")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.0)
        .max(0.0);
    let rotation_smooth_time = object
        .get("rotation_smooth_time")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.0)
        .max(0.0);

    out.insert("target_entity".to_string(), target_entity);
    out.insert("position_offset".to_string(), position_offset);
    out.insert(
        "offset_in_target_space".to_string(),
        JsonValue::Bool(offset_in_target_space),
    );
    out.insert(
        "follow_rotation".to_string(),
        JsonValue::Bool(follow_rotation),
    );
    out.insert(
        "position_smooth_time".to_string(),
        json_number(position_smooth_time),
    );
    out.insert(
        "rotation_smooth_time".to_string(),
        json_number(rotation_smooth_time),
    );
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_animator_state(value: &JsonValue) -> Result<JsonValue, String> {
    coerce_json_to_default_object(
        value,
        "Animator State",
        default_literal_for_type(VisualValueType::AnimatorState),
    )
}

fn coerce_json_to_input_modifiers(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Input Modifiers")?;
    let mut out = JsonMap::new();
    out.insert(
        "shift".to_string(),
        JsonValue::Bool(object.get("shift").map(is_truthy).unwrap_or(false)),
    );
    out.insert(
        "ctrl".to_string(),
        JsonValue::Bool(object.get("ctrl").map(is_truthy).unwrap_or(false)),
    );
    out.insert(
        "alt".to_string(),
        JsonValue::Bool(object.get("alt").map(is_truthy).unwrap_or(false)),
    );
    out.insert(
        "super".to_string(),
        JsonValue::Bool(object.get("super").map(is_truthy).unwrap_or(false)),
    );
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_audio_streaming_config(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Audio Streaming Config")?;
    let buffer_frames = object
        .get("buffer_frames")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(8192.0)
        .max(1.0)
        .round() as u64;
    let chunk_frames = object
        .get("chunk_frames")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(2048.0)
        .max(1.0)
        .round() as u64;
    let mut out = JsonMap::new();
    out.insert(
        "buffer_frames".to_string(),
        JsonValue::Number(JsonNumber::from(buffer_frames)),
    );
    out.insert(
        "chunk_frames".to_string(),
        JsonValue::Number(JsonNumber::from(chunk_frames)),
    );
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_spline(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Spline")?;
    let points = object
        .get("points")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Array))
        .transpose()?
        .unwrap_or_else(|| JsonValue::Array(Vec::new()));
    let points = points.as_array().cloned().unwrap_or_default();
    let mut normalized_points = Vec::with_capacity(points.len());
    for point in points {
        normalized_points.push(coerce_json_to_visual_type(&point, VisualValueType::Vec3)?);
    }
    let closed = object.get("closed").map(is_truthy).unwrap_or(false);
    let tension = object
        .get("tension")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let mut mode = object
        .get("mode")
        .and_then(JsonValue::as_str)
        .unwrap_or("CatmullRom")
        .trim()
        .to_string();
    if !matches!(mode.as_str(), "Linear" | "CatmullRom" | "Bezier") {
        mode = "CatmullRom".to_string();
    }

    let mut out = JsonMap::new();
    out.insert("points".to_string(), JsonValue::Array(normalized_points));
    out.insert("closed".to_string(), JsonValue::Bool(closed));
    out.insert("tension".to_string(), json_number(tension));
    out.insert("mode".to_string(), JsonValue::String(mode));
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_physics_velocity(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Physics Velocity")?;
    let mut out = JsonMap::new();
    let linear = object
        .get("linear")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let angular = object
        .get("angular")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let wake_up = object.get("wake_up").map(is_truthy).unwrap_or(true);
    out.insert("linear".to_string(), linear);
    out.insert("angular".to_string(), angular);
    out.insert("wake_up".to_string(), JsonValue::Bool(wake_up));
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_physics_world_defaults(value: &JsonValue) -> Result<JsonValue, String> {
    let mut merged = parse_json_object_literal(default_literal_for_type(
        VisualValueType::PhysicsWorldDefaults,
    ));
    let input = coerce_json_to_loose_object(value, "Physics World Defaults")?;
    if let Some(gravity) = input.get("gravity") {
        merged.insert(
            "gravity".to_string(),
            coerce_json_to_visual_type(gravity, VisualValueType::Vec3)?,
        );
    }
    if let Some(collider_properties) = input.get("collider_properties") {
        merged.insert(
            "collider_properties".to_string(),
            collider_properties.clone(),
        );
    }
    if let Some(rigid_body_properties) = input.get("rigid_body_properties") {
        merged.insert(
            "rigid_body_properties".to_string(),
            rigid_body_properties.clone(),
        );
    }
    Ok(JsonValue::Object(merged))
}

fn coerce_json_to_physics_query_filter(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Physics Query Filter")?;
    let mut out = JsonMap::new();

    let flags = object
        .get("flags")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.0)
        .max(0.0)
        .round() as u32;
    let groups_memberships = object
        .get("groups_memberships")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(u32::MAX as f64)
        .max(0.0)
        .round() as u32;
    let groups_filter = object
        .get("groups_filter")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(u32::MAX as f64)
        .max(0.0)
        .round() as u32;
    let use_groups = object.get("use_groups").map(is_truthy).unwrap_or(false);

    out.insert(
        "flags".to_string(),
        JsonValue::Number(JsonNumber::from(flags)),
    );
    out.insert(
        "groups_memberships".to_string(),
        JsonValue::Number(JsonNumber::from(groups_memberships)),
    );
    out.insert(
        "groups_filter".to_string(),
        JsonValue::Number(JsonNumber::from(groups_filter)),
    );
    out.insert("use_groups".to_string(), JsonValue::Bool(use_groups));
    Ok(JsonValue::Object(out))
}

fn coerce_entity_or_null(value: Option<&JsonValue>) -> JsonValue {
    let Some(value) = value else {
        return JsonValue::Null;
    };
    if matches!(value, JsonValue::Null) {
        return JsonValue::Null;
    }
    let Ok(raw) = coerce_json_to_f64(value) else {
        return JsonValue::Null;
    };
    if !raw.is_finite() || raw < 0.0 {
        return JsonValue::Null;
    }
    JsonValue::Number(JsonNumber::from(raw as u64))
}

fn coerce_json_to_character_output(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Character Output")?;
    let mut out = JsonMap::new();
    let desired_translation = object
        .get("desired_translation")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let effective_translation = object
        .get("effective_translation")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let remaining_translation = object
        .get("remaining_translation")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let grounded = object.get("grounded").map(is_truthy).unwrap_or(false);
    let sliding_down_slope = object
        .get("sliding_down_slope")
        .map(is_truthy)
        .unwrap_or(false);
    let collision_count = object
        .get("collision_count")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.0)
        .max(0.0);
    let ground_normal = object
        .get("ground_normal")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 1.0, 0.0));
    let slope_angle = object
        .get("slope_angle")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.0)
        .max(0.0);
    let hit_normal = object
        .get("hit_normal")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let hit_point = object
        .get("hit_point")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let stepped_up = object.get("stepped_up").map(is_truthy).unwrap_or(false);
    let step_height = object
        .get("step_height")
        .map(coerce_json_to_f64)
        .transpose()?
        .unwrap_or(0.0)
        .max(0.0);
    let platform_velocity = object
        .get("platform_velocity")
        .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
        .transpose()?
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    out.insert("desired_translation".to_string(), desired_translation);
    out.insert("effective_translation".to_string(), effective_translation);
    out.insert("remaining_translation".to_string(), remaining_translation);
    out.insert("grounded".to_string(), JsonValue::Bool(grounded));
    out.insert(
        "sliding_down_slope".to_string(),
        JsonValue::Bool(sliding_down_slope),
    );
    out.insert(
        "collision_count".to_string(),
        JsonValue::Number(JsonNumber::from(collision_count.round() as u64)),
    );
    out.insert("ground_normal".to_string(), ground_normal);
    out.insert("slope_angle".to_string(), json_number(slope_angle));
    out.insert("hit_normal".to_string(), hit_normal);
    out.insert("hit_point".to_string(), hit_point);
    out.insert(
        "hit_entity".to_string(),
        coerce_entity_or_null(object.get("hit_entity")),
    );
    out.insert("stepped_up".to_string(), JsonValue::Bool(stepped_up));
    out.insert("step_height".to_string(), json_number(step_height));
    out.insert("platform_velocity".to_string(), platform_velocity);
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_dynamic_field_value(value: &JsonValue) -> Result<JsonValue, String> {
    match value {
        JsonValue::Bool(value) => Ok(JsonValue::Bool(*value)),
        JsonValue::Number(value) => Ok(JsonValue::Number(value.clone())),
        JsonValue::String(value) => Ok(JsonValue::String(value.clone())),
        JsonValue::Null => Ok(JsonValue::String(String::new())),
        JsonValue::Array(_) | JsonValue::Object(_) => {
            let (x, y, z) = coerce_json_to_vec3_components(value)?;
            Ok(vec3_json(x, y, z))
        }
    }
}

fn coerce_json_to_dynamic_component_fields(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Dynamic Fields")?;
    let mut out = JsonMap::new();
    for (field, value) in object {
        if field.trim().is_empty() {
            continue;
        }
        out.insert(field, coerce_json_to_dynamic_field_value(&value)?);
    }
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_ray_cast_hit(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Ray Cast Hit")?;
    let mut out = JsonMap::new();
    out.insert(
        "has_hit".to_string(),
        JsonValue::Bool(object.get("has_hit").map(is_truthy).unwrap_or(false)),
    );
    out.insert(
        "hit_entity".to_string(),
        coerce_entity_or_null(object.get("hit_entity")),
    );
    out.insert(
        "point".to_string(),
        object
            .get("point")
            .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
            .transpose()?
            .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0)),
    );
    out.insert(
        "normal".to_string(),
        object
            .get("normal")
            .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
            .transpose()?
            .unwrap_or_else(|| vec3_json(0.0, 1.0, 0.0)),
    );
    out.insert(
        "toi".to_string(),
        json_number(
            object
                .get("toi")
                .map(coerce_json_to_f64)
                .transpose()?
                .unwrap_or(0.0)
                .max(0.0),
        ),
    );
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_point_projection_hit(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Point Projection Hit")?;
    let mut out = JsonMap::new();
    out.insert(
        "has_hit".to_string(),
        JsonValue::Bool(object.get("has_hit").map(is_truthy).unwrap_or(false)),
    );
    out.insert(
        "hit_entity".to_string(),
        coerce_entity_or_null(object.get("hit_entity")),
    );
    out.insert(
        "projected_point".to_string(),
        object
            .get("projected_point")
            .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
            .transpose()?
            .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0)),
    );
    out.insert(
        "is_inside".to_string(),
        JsonValue::Bool(object.get("is_inside").map(is_truthy).unwrap_or(false)),
    );
    out.insert(
        "distance".to_string(),
        json_number(
            object
                .get("distance")
                .map(coerce_json_to_f64)
                .transpose()?
                .unwrap_or(0.0)
                .max(0.0),
        ),
    );
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_shape_cast_hit(value: &JsonValue) -> Result<JsonValue, String> {
    let object = coerce_json_to_loose_object(value, "Shape Cast Hit")?;
    let mut out = JsonMap::new();
    out.insert(
        "has_hit".to_string(),
        JsonValue::Bool(object.get("has_hit").map(is_truthy).unwrap_or(false)),
    );
    out.insert(
        "hit_entity".to_string(),
        coerce_entity_or_null(object.get("hit_entity")),
    );
    out.insert(
        "toi".to_string(),
        json_number(
            object
                .get("toi")
                .map(coerce_json_to_f64)
                .transpose()?
                .unwrap_or(0.0)
                .max(0.0),
        ),
    );
    out.insert(
        "witness1".to_string(),
        object
            .get("witness1")
            .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
            .transpose()?
            .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0)),
    );
    out.insert(
        "witness2".to_string(),
        object
            .get("witness2")
            .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
            .transpose()?
            .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0)),
    );
    out.insert(
        "normal1".to_string(),
        object
            .get("normal1")
            .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
            .transpose()?
            .unwrap_or_else(|| vec3_json(0.0, 1.0, 0.0)),
    );
    out.insert(
        "normal2".to_string(),
        object
            .get("normal2")
            .map(|entry| coerce_json_to_visual_type(entry, VisualValueType::Vec3))
            .transpose()?
            .unwrap_or_else(|| vec3_json(0.0, 1.0, 0.0)),
    );
    out.insert(
        "status".to_string(),
        JsonValue::String(
            object
                .get("status")
                .and_then(JsonValue::as_str)
                .unwrap_or("Unknown")
                .to_string(),
        ),
    );
    Ok(JsonValue::Object(out))
}

fn coerce_json_to_vec2_components(value: &JsonValue) -> Result<(f64, f64), String> {
    match value {
        JsonValue::Null => Ok((0.0, 0.0)),
        JsonValue::Bool(value) => Ok((if *value { 1.0 } else { 0.0 }, 0.0)),
        JsonValue::Number(_) => Ok((coerce_json_to_f64(value)?, 0.0)),
        JsonValue::Object(object) => Ok((
            coerce_json_to_f64(
                object
                    .get("x")
                    .or_else(|| object.get("u"))
                    .unwrap_or(&JsonValue::Null),
            )?,
            coerce_json_to_f64(
                object
                    .get("y")
                    .or_else(|| object.get("v"))
                    .unwrap_or(&JsonValue::Null),
            )?,
        )),
        JsonValue::Array(array) if !array.is_empty() => {
            let x = coerce_json_to_f64(&array[0])?;
            let y = array
                .get(1)
                .map(coerce_json_to_f64)
                .transpose()?
                .unwrap_or(0.0);
            Ok((x, y))
        }
        JsonValue::String(text) => coerce_json_to_vec2_components(&parse_loose_literal(text)),
        _ => Err("Vec2 values must be objects, arrays, or literal strings".to_string()),
    }
}

fn coerce_json_to_vec3_components(value: &JsonValue) -> Result<(f64, f64, f64), String> {
    match value {
        JsonValue::Object(object) => Ok((
            coerce_json_to_f64(object.get("x").unwrap_or(&JsonValue::Null))?,
            coerce_json_to_f64(object.get("y").unwrap_or(&JsonValue::Null))?,
            coerce_json_to_f64(object.get("z").unwrap_or(&JsonValue::Null))?,
        )),
        JsonValue::Array(array) if array.len() >= 3 => Ok((
            coerce_json_to_f64(&array[0])?,
            coerce_json_to_f64(&array[1])?,
            coerce_json_to_f64(&array[2])?,
        )),
        JsonValue::String(text) => coerce_json_to_vec3_components(&parse_loose_literal(text)),
        _ => Err("Vec3 values must be objects, arrays, or literal strings".to_string()),
    }
}

fn coerce_json_to_quat_components(value: &JsonValue) -> Result<(f64, f64, f64, f64), String> {
    match value {
        JsonValue::Object(object) => Ok((
            coerce_json_to_f64(object.get("x").unwrap_or(&JsonValue::Null))?,
            coerce_json_to_f64(object.get("y").unwrap_or(&JsonValue::Null))?,
            coerce_json_to_f64(object.get("z").unwrap_or(&JsonValue::Null))?,
            coerce_json_to_f64(
                object
                    .get("w")
                    .unwrap_or(&JsonValue::Number(JsonNumber::from(1))),
            )?,
        )),
        JsonValue::Array(array) if array.len() >= 4 => Ok((
            coerce_json_to_f64(&array[0])?,
            coerce_json_to_f64(&array[1])?,
            coerce_json_to_f64(&array[2])?,
            coerce_json_to_f64(&array[3])?,
        )),
        JsonValue::String(text) => coerce_json_to_quat_components(&parse_loose_literal(text)),
        _ => Err("Quat values must be objects, arrays, or literal strings".to_string()),
    }
}

fn coerce_json_to_vec4_components(value: &JsonValue) -> Result<(f64, f64, f64, f64), String> {
    match value {
        JsonValue::Object(object) => Ok((
            coerce_json_to_f64(
                object
                    .get("x")
                    .or_else(|| object.get("r"))
                    .unwrap_or(&JsonValue::Null),
            )?,
            coerce_json_to_f64(
                object
                    .get("y")
                    .or_else(|| object.get("g"))
                    .unwrap_or(&JsonValue::Null),
            )?,
            coerce_json_to_f64(
                object
                    .get("z")
                    .or_else(|| object.get("b"))
                    .unwrap_or(&JsonValue::Null),
            )?,
            coerce_json_to_f64(
                object
                    .get("w")
                    .or_else(|| object.get("a"))
                    .unwrap_or(&JsonValue::Number(JsonNumber::from(1))),
            )?,
        )),
        JsonValue::Array(array) if array.len() >= 4 => Ok((
            coerce_json_to_f64(&array[0])?,
            coerce_json_to_f64(&array[1])?,
            coerce_json_to_f64(&array[2])?,
            coerce_json_to_f64(&array[3])?,
        )),
        JsonValue::String(text) => coerce_json_to_vec4_components(&parse_loose_literal(text)),
        _ => Err("Vec4 values must be objects, arrays, or literal strings".to_string()),
    }
}

fn json_number(value: f64) -> JsonValue {
    JsonNumber::from_f64(value)
        .map(JsonValue::Number)
        .unwrap_or(JsonValue::Null)
}

fn json_to_log_string(value: &JsonValue) -> String {
    match value {
        JsonValue::String(value) => value.clone(),
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(value) => value.to_string(),
        JsonValue::Number(value) => value.to_string(),
        JsonValue::Array(_) | JsonValue::Object(_) => {
            serde_json::to_string(value).unwrap_or_else(|_| "<json>".to_string())
        }
    }
}

fn parse_loose_literal(value: &str) -> JsonValue {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return JsonValue::Null;
    }

    if let Ok(parsed) = serde_json::from_str::<JsonValue>(trimmed) {
        return parsed;
    }

    let lower = trimmed.to_ascii_lowercase();
    if lower == "null" || lower == "nil" {
        return JsonValue::Null;
    }
    if lower == "true" {
        return JsonValue::Bool(true);
    }
    if lower == "false" {
        return JsonValue::Bool(false);
    }

    if let Ok(parsed_int) = trimmed.parse::<i64>() {
        return JsonValue::Number(JsonNumber::from(parsed_int));
    }

    if let Ok(parsed_float) = trimmed.parse::<f64>() {
        return json_number(parsed_float);
    }

    JsonValue::String(trimmed.to_string())
}

#[derive(Debug)]
pub struct VisualScriptOpenDocument {
    pub path: PathBuf,
    pub name: String,
    pub prelude: String,
    pub variables: Vec<VisualVariableDefinition>,
    pub functions: Vec<VisualScriptFunctionDefinition>,
    pub snarl: Snarl<VisualScriptNodeKind>,
    pub function_snarls: HashMap<u64, Snarl<VisualScriptNodeKind>>,
    pub active_graph_function: Option<u64>,
    pub dirty: bool,
    pub compile_preview: String,
    pub compile_error: Option<String>,
}

impl VisualScriptOpenDocument {
    fn from_document(path: &Path, mut document: VisualScriptDocument) -> Self {
        normalize_document(&mut document);
        let name = if document.name.trim().is_empty() {
            path.file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("visual_script")
                .to_string()
        } else {
            document.name
        };

        let mut out = Self {
            path: path.to_path_buf(),
            name,
            prelude: document.prelude,
            variables: document.variables,
            functions: document.functions.clone(),
            snarl: graph_data_to_snarl(&document.graph),
            function_snarls: document
                .functions
                .iter()
                .map(|function| (function.id, graph_data_to_snarl(&function.graph)))
                .collect(),
            active_graph_function: None,
            dirty: false,
            compile_preview: String::new(),
            compile_error: None,
        };
        out.recompile_preview();
        out
    }

    fn to_document(&self) -> VisualScriptDocument {
        let mut functions = self.functions.clone();
        for function in &mut functions {
            if let Some(snarl) = self.function_snarls.get(&function.id) {
                function.graph = graph_data_from_snarl(snarl);
            }
        }

        let mut document = VisualScriptDocument {
            version: VISUAL_SCRIPT_VERSION,
            name: self.name.trim().to_string(),
            prelude: self.prelude.clone(),
            variables: self.variables.clone(),
            functions,
            graph: graph_data_from_snarl(&self.snarl),
        };
        normalize_document(&mut document);
        document
    }

    fn recompile_preview(&mut self) {
        let source_label = self.path.to_string_lossy().to_string();
        match compile_visual_script_document(&self.to_document(), &source_label) {
            Ok(source) => {
                self.compile_preview = source;
                self.compile_error = None;
            }
            Err(err) => {
                self.compile_preview.clear();
                self.compile_error = Some(err);
            }
        }
    }

    fn save(&mut self) -> Result<(), String> {
        let document = self.to_document();
        let pretty = PrettyConfig::new().compact_arrays(false);
        let payload =
            ron::ser::to_string_pretty(&document, pretty).map_err(|err| err.to_string())?;
        fs::write(&self.path, payload).map_err(|err| err.to_string())?;
        self.dirty = false;
        Ok(())
    }
}

fn default_visual_graph_style() -> SnarlStyle {
    let mut style = SnarlStyle::new();
    style.pin_placement = Some(PinPlacement::Edge);
    style
}

fn node_layout_kind_label(kind: NodeLayoutKind) -> &'static str {
    match kind {
        NodeLayoutKind::Coil => "Coil",
        NodeLayoutKind::Sandwich => "Sandwich",
        NodeLayoutKind::FlippedSandwich => "Flipped Sandwich",
    }
}

fn pin_placement_label(placement: PinPlacement) -> &'static str {
    match placement {
        PinPlacement::Inside => "Inside",
        PinPlacement::Edge => "Edge",
        PinPlacement::Outside { .. } => "Outside",
    }
}

fn pin_shape_label(shape: PinShape) -> &'static str {
    match shape {
        PinShape::Circle => "Circle",
        PinShape::Triangle => "Triangle",
        PinShape::Square => "Square",
        PinShape::Star => "Star",
    }
}

fn wire_style_label(style: WireStyle) -> &'static str {
    match style {
        WireStyle::Line => "Line",
        WireStyle::AxisAligned { .. } => "Axis Aligned",
        WireStyle::Bezier3 => "Bezier 3",
        WireStyle::Bezier5 => "Bezier 5",
    }
}

fn wire_layer_label(layer: WireLayer) -> &'static str {
    match layer {
        WireLayer::BehindNodes => "Behind Nodes",
        WireLayer::AboveNodes => "Above Nodes",
    }
}

fn background_pattern_label(pattern: BackgroundPattern) -> &'static str {
    match pattern {
        BackgroundPattern::NoPattern => "None",
        BackgroundPattern::Grid(_) => "Grid",
    }
}

fn draw_stroke_editor(ui: &mut Ui, label: &str, stroke: &mut Stroke) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui
            .add(
                DragValue::new(&mut stroke.width)
                    .range(0.0..=16.0)
                    .speed(0.05),
            )
            .changed();
        changed |= ui.color_edit_button_srgba(&mut stroke.color).changed();
    });
    changed
}

fn draw_margin_editor(ui: &mut Ui, label: &str, margin: &mut egui::Margin) -> bool {
    let mut changed = false;
    let mut x = i32::from(margin.left.max(margin.right));
    let mut y = i32::from(margin.top.max(margin.bottom));
    ui.horizontal(|ui| {
        ui.label(label);
        ui.label("X");
        changed |= ui.add(DragValue::new(&mut x).range(-64..=64)).changed();
        ui.label("Y");
        changed |= ui.add(DragValue::new(&mut y).range(-64..=64)).changed();
    });
    if changed {
        *margin = egui::Margin::symmetric(x.clamp(-64, 64) as i8, y.clamp(-64, 64) as i8);
    }
    changed
}

fn draw_corner_radius_editor(
    ui: &mut Ui,
    label: &str,
    corner_radius: &mut egui::CornerRadius,
) -> bool {
    let mut changed = false;
    let mut uniform_radius = corner_radius.average().round() as i32;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui
            .add(DragValue::new(&mut uniform_radius).range(0..=255))
            .changed();
    });
    if changed {
        *corner_radius = egui::CornerRadius::same(uniform_radius.clamp(0, 255) as u8);
    }
    changed
}

fn draw_shadow_editor(ui: &mut Ui, label: &str, shadow: &mut egui::epaint::Shadow) -> bool {
    let mut changed = false;
    let mut offset_x = i32::from(shadow.offset[0]);
    let mut offset_y = i32::from(shadow.offset[1]);
    let mut blur = i32::from(shadow.blur);
    let mut spread = i32::from(shadow.spread);

    ui.horizontal(|ui| {
        ui.label(label);
        ui.label("X");
        changed |= ui
            .add(DragValue::new(&mut offset_x).range(-64..=64))
            .changed();
        ui.label("Y");
        changed |= ui
            .add(DragValue::new(&mut offset_y).range(-64..=64))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Blur");
        changed |= ui.add(DragValue::new(&mut blur).range(0..=255)).changed();
        ui.label("Spread");
        changed |= ui.add(DragValue::new(&mut spread).range(0..=255)).changed();
        changed |= ui.color_edit_button_srgba(&mut shadow.color).changed();
    });

    if changed {
        shadow.offset = [offset_x.clamp(-64, 64) as i8, offset_y.clamp(-64, 64) as i8];
        shadow.blur = blur.clamp(0, 255) as u8;
        shadow.spread = spread.clamp(0, 255) as u8;
    }
    changed
}

fn draw_frame_editor(ui: &mut Ui, label: &str, frame: &mut egui::Frame) -> bool {
    let mut changed = false;
    ui.collapsing(label, |ui| {
        ui.horizontal(|ui| {
            ui.label("Fill");
            changed |= ui.color_edit_button_srgba(&mut frame.fill).changed();
        });
        changed |= draw_stroke_editor(ui, "Stroke", &mut frame.stroke);
        changed |= draw_corner_radius_editor(ui, "Corner radius", &mut frame.corner_radius);
        changed |= draw_margin_editor(ui, "Inner margin", &mut frame.inner_margin);
        changed |= draw_margin_editor(ui, "Outer margin", &mut frame.outer_margin);
        changed |= draw_shadow_editor(ui, "Shadow", &mut frame.shadow);
    });
    changed
}

fn draw_selection_style_editor(ui: &mut Ui, style: &mut SelectionStyle) -> bool {
    let mut changed = false;
    changed |= draw_margin_editor(ui, "Margin", &mut style.margin);
    changed |= draw_corner_radius_editor(ui, "Corner radius", &mut style.rounding);
    ui.horizontal(|ui| {
        ui.label("Fill");
        changed |= ui.color_edit_button_srgba(&mut style.fill).changed();
    });
    changed |= draw_stroke_editor(ui, "Stroke", &mut style.stroke);
    changed
}

fn draw_visual_graph_style_menu(ui: &mut Ui, graph_style: &mut SnarlStyle) {
    ui.set_min_width(360.0);
    if ui.button("Reset Editor Default").clicked() {
        *graph_style = default_visual_graph_style();
        ui.ctx().request_repaint();
        return;
    }
    if ui.button("Reset Snarl Default").clicked() {
        *graph_style = SnarlStyle::new();
        ui.ctx().request_repaint();
        return;
    }
    ui.separator();

    ui.collapsing("Layout", |ui| {
        let mut node_layout = graph_style.node_layout.unwrap_or_else(NodeLayout::coil);
        ui.horizontal(|ui| {
            ui.label("Node layout");
            ComboBox::from_id_salt(ui.id().with("graph_style_node_layout"))
                .selected_text(node_layout_kind_label(node_layout.kind))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut node_layout.kind, NodeLayoutKind::Coil, "Coil");
                    ui.selectable_value(
                        &mut node_layout.kind,
                        NodeLayoutKind::Sandwich,
                        "Sandwich",
                    );
                    ui.selectable_value(
                        &mut node_layout.kind,
                        NodeLayoutKind::FlippedSandwich,
                        "Flipped Sandwich",
                    );
                });
        });
        ui.checkbox(
            &mut node_layout.equal_pin_row_heights,
            "Equal input/output row heights",
        );
        ui.horizontal(|ui| {
            ui.label("Min pin row height");
            ui.add(
                DragValue::new(&mut node_layout.min_pin_row_height)
                    .range(0.0..=128.0)
                    .speed(0.2),
            );
        });
        graph_style.node_layout = Some(node_layout);

        let mut header_drag_space = graph_style.header_drag_space.unwrap_or_else(|| {
            egui::vec2(ui.style().spacing.icon_width, ui.style().spacing.icon_width)
        });
        ui.horizontal(|ui| {
            ui.label("Header drag");
            ui.label("X");
            ui.add(
                DragValue::new(&mut header_drag_space.x)
                    .range(0.0..=128.0)
                    .speed(0.2),
            );
            ui.label("Y");
            ui.add(
                DragValue::new(&mut header_drag_space.y)
                    .range(0.0..=128.0)
                    .speed(0.2),
            );
        });
        graph_style.header_drag_space = Some(header_drag_space);

        let mut collapsible = graph_style.collapsible.unwrap_or(true);
        if ui.checkbox(&mut collapsible, "Collapsible nodes").changed() {
            graph_style.collapsible = Some(collapsible);
        }
    });

    ui.collapsing("Pins", |ui| {
        let mut pin_placement = graph_style.pin_placement.unwrap_or(PinPlacement::Edge);
        let mut pin_placement_kind = match pin_placement {
            PinPlacement::Inside => 0,
            PinPlacement::Edge => 1,
            PinPlacement::Outside { .. } => 2,
        };
        ui.horizontal(|ui| {
            ui.label("Placement");
            ComboBox::from_id_salt(ui.id().with("graph_style_pin_placement"))
                .selected_text(pin_placement_label(pin_placement))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut pin_placement_kind, 0, "Inside");
                    ui.selectable_value(&mut pin_placement_kind, 1, "Edge");
                    ui.selectable_value(&mut pin_placement_kind, 2, "Outside");
                });
        });
        pin_placement = match pin_placement_kind {
            0 => PinPlacement::Inside,
            1 => PinPlacement::Edge,
            _ => {
                let default_margin = 12.0;
                let mut margin = match pin_placement {
                    PinPlacement::Outside { margin } => margin,
                    _ => default_margin,
                };
                ui.horizontal(|ui| {
                    ui.label("Outside margin");
                    ui.add(DragValue::new(&mut margin).range(0.0..=128.0).speed(0.2));
                });
                PinPlacement::Outside {
                    margin: margin.max(0.0),
                }
            }
        };
        graph_style.pin_placement = Some(pin_placement);

        let mut pin_shape = graph_style.pin_shape.unwrap_or(PinShape::Circle);
        ui.horizontal(|ui| {
            ui.label("Shape");
            ComboBox::from_id_salt(ui.id().with("graph_style_pin_shape"))
                .selected_text(pin_shape_label(pin_shape))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut pin_shape, PinShape::Circle, "Circle");
                    ui.selectable_value(&mut pin_shape, PinShape::Triangle, "Triangle");
                    ui.selectable_value(&mut pin_shape, PinShape::Square, "Square");
                    ui.selectable_value(&mut pin_shape, PinShape::Star, "Star");
                });
        });
        graph_style.pin_shape = Some(pin_shape);

        let mut pin_size = graph_style
            .pin_size
            .unwrap_or(ui.style().spacing.interact_size.y * 0.6);
        ui.horizontal(|ui| {
            ui.label("Size");
            ui.add(DragValue::new(&mut pin_size).range(1.0..=64.0).speed(0.2));
        });
        graph_style.pin_size = Some(pin_size.max(1.0));

        let mut pin_fill = graph_style
            .pin_fill
            .unwrap_or(ui.style().visuals.widgets.active.bg_fill);
        ui.horizontal(|ui| {
            ui.label("Fill");
            if ui.color_edit_button_srgba(&mut pin_fill).changed() {
                graph_style.pin_fill = Some(pin_fill);
            }
            if ui.small_button("Theme").clicked() {
                graph_style.pin_fill = None;
            }
        });

        let mut pin_stroke = graph_style
            .pin_stroke
            .unwrap_or(ui.style().visuals.widgets.active.bg_stroke);
        if draw_stroke_editor(ui, "Stroke", &mut pin_stroke) {
            graph_style.pin_stroke = Some(pin_stroke);
        }
        if ui.small_button("Pin stroke from theme").clicked() {
            graph_style.pin_stroke = None;
        }
    });

    ui.collapsing("Wires", |ui| {
        let pin_size = graph_style
            .pin_size
            .unwrap_or(ui.style().spacing.interact_size.y * 0.6);
        let mut wire_width = graph_style.wire_width.unwrap_or(pin_size * 0.1);
        let mut wire_frame_size = graph_style.wire_frame_size.unwrap_or(pin_size * 3.0);
        let mut wire_smoothness = graph_style.wire_smoothness.unwrap_or(1.0);
        ui.horizontal(|ui| {
            ui.label("Width");
            ui.add(DragValue::new(&mut wire_width).range(0.1..=32.0).speed(0.1));
        });
        ui.horizontal(|ui| {
            ui.label("Frame size");
            ui.add(
                DragValue::new(&mut wire_frame_size)
                    .range(0.1..=512.0)
                    .speed(0.5),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Smoothness");
            ui.add(
                DragValue::new(&mut wire_smoothness)
                    .range(0.0..=10.0)
                    .speed(0.05),
            );
        });
        graph_style.wire_width = Some(wire_width.max(0.1));
        graph_style.wire_frame_size = Some(wire_frame_size.max(0.1));
        graph_style.wire_smoothness = Some(wire_smoothness.clamp(0.0, 10.0));

        let mut downscale_wire_frame = graph_style.downscale_wire_frame.unwrap_or(true);
        let mut upscale_wire_frame = graph_style.upscale_wire_frame.unwrap_or(false);
        if ui
            .checkbox(&mut downscale_wire_frame, "Downscale frame when close")
            .changed()
        {
            graph_style.downscale_wire_frame = Some(downscale_wire_frame);
        }
        if ui
            .checkbox(&mut upscale_wire_frame, "Upscale frame when far")
            .changed()
        {
            graph_style.upscale_wire_frame = Some(upscale_wire_frame);
        }

        let mut wire_style = graph_style.wire_style.unwrap_or(WireStyle::Bezier5);
        let mut wire_style_kind = match wire_style {
            WireStyle::Line => 0,
            WireStyle::AxisAligned { .. } => 1,
            WireStyle::Bezier3 => 2,
            WireStyle::Bezier5 => 3,
        };
        ui.horizontal(|ui| {
            ui.label("Style");
            ComboBox::from_id_salt(ui.id().with("graph_style_wire_style"))
                .selected_text(wire_style_label(wire_style))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut wire_style_kind, 0, "Line");
                    ui.selectable_value(&mut wire_style_kind, 1, "Axis Aligned");
                    ui.selectable_value(&mut wire_style_kind, 2, "Bezier 3");
                    ui.selectable_value(&mut wire_style_kind, 3, "Bezier 5");
                });
        });
        wire_style = match wire_style_kind {
            0 => WireStyle::Line,
            1 => {
                let mut corner_radius = match wire_style {
                    WireStyle::AxisAligned { corner_radius } => corner_radius,
                    _ => 10.0,
                };
                ui.horizontal(|ui| {
                    ui.label("Corner radius");
                    ui.add(
                        DragValue::new(&mut corner_radius)
                            .range(0.0..=128.0)
                            .speed(0.2),
                    );
                });
                WireStyle::AxisAligned {
                    corner_radius: corner_radius.max(0.0),
                }
            }
            2 => WireStyle::Bezier3,
            _ => WireStyle::Bezier5,
        };
        graph_style.wire_style = Some(wire_style);

        let mut wire_layer = graph_style.wire_layer.unwrap_or(WireLayer::BehindNodes);
        ui.horizontal(|ui| {
            ui.label("Layer");
            ComboBox::from_id_salt(ui.id().with("graph_style_wire_layer"))
                .selected_text(wire_layer_label(wire_layer))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut wire_layer, WireLayer::BehindNodes, "Behind Nodes");
                    ui.selectable_value(&mut wire_layer, WireLayer::AboveNodes, "Above Nodes");
                });
        });
        graph_style.wire_layer = Some(wire_layer);
    });

    ui.collapsing("Background", |ui| {
        let mut pattern_kind = match graph_style.bg_pattern {
            None => 0,
            Some(BackgroundPattern::NoPattern) => 1,
            Some(BackgroundPattern::Grid(_)) => 2,
        };
        let pattern_label = match graph_style.bg_pattern {
            None => "Default".to_string(),
            Some(pattern) => background_pattern_label(pattern).to_string(),
        };
        let mut pattern_kind_changed = false;
        ui.horizontal(|ui| {
            ui.label("Pattern");
            ComboBox::from_id_salt(ui.id().with("graph_style_bg_pattern"))
                .selected_text(pattern_label)
                .show_ui(ui, |ui| {
                    pattern_kind_changed |= ui
                        .selectable_value(&mut pattern_kind, 0, "Default")
                        .changed();
                    pattern_kind_changed |=
                        ui.selectable_value(&mut pattern_kind, 1, "None").changed();
                    pattern_kind_changed |=
                        ui.selectable_value(&mut pattern_kind, 2, "Grid").changed();
                });
        });
        if pattern_kind_changed {
            graph_style.bg_pattern = match pattern_kind {
                0 => None,
                1 => Some(BackgroundPattern::NoPattern),
                _ => Some(BackgroundPattern::Grid(Grid::default())),
            };
        }

        if pattern_kind == 2 {
            let mut grid = match graph_style.bg_pattern {
                Some(BackgroundPattern::Grid(grid)) => grid,
                _ => Grid::default(),
            };
            let mut grid_changed = false;
            ui.horizontal(|ui| {
                ui.label("Spacing X");
                grid_changed |= ui
                    .add(
                        DragValue::new(&mut grid.spacing.x)
                            .range(1.0..=1024.0)
                            .speed(1.0),
                    )
                    .changed();
                ui.label("Y");
                grid_changed |= ui
                    .add(
                        DragValue::new(&mut grid.spacing.y)
                            .range(1.0..=1024.0)
                            .speed(1.0),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("Angle (rad)");
                grid_changed |= ui
                    .add(
                        DragValue::new(&mut grid.angle)
                            .range(-std::f32::consts::TAU..=std::f32::consts::TAU)
                            .speed(0.01),
                    )
                    .changed();
            });
            if grid_changed {
                graph_style.bg_pattern = Some(BackgroundPattern::Grid(grid));
            }
        }

        let mut bg_pattern_stroke = graph_style
            .bg_pattern_stroke
            .unwrap_or(ui.style().visuals.widgets.noninteractive.bg_stroke);
        if draw_stroke_editor(ui, "Pattern stroke", &mut bg_pattern_stroke) {
            graph_style.bg_pattern_stroke = Some(bg_pattern_stroke);
        }
        if ui.small_button("Pattern stroke from theme").clicked() {
            graph_style.bg_pattern_stroke = None;
        }
    });

    ui.collapsing("Interaction", |ui| {
        let mut min_scale = graph_style.min_scale.unwrap_or(0.2);
        let mut max_scale = graph_style.max_scale.unwrap_or(2.0);
        ui.horizontal(|ui| {
            ui.label("Min scale");
            ui.add(DragValue::new(&mut min_scale).range(0.01..=8.0).speed(0.01));
            ui.label("Max");
            ui.add(DragValue::new(&mut max_scale).range(0.1..=16.0).speed(0.01));
        });
        min_scale = min_scale.clamp(0.01, 8.0);
        max_scale = max_scale.clamp(min_scale, 16.0);
        graph_style.min_scale = Some(min_scale);
        graph_style.max_scale = Some(max_scale);

        let mut centering = graph_style.centering.unwrap_or(true);
        if ui.checkbox(&mut centering, "Double-click center").changed() {
            graph_style.centering = Some(centering);
        }
        let mut crisp_text = graph_style.crisp_magnified_text.unwrap_or(false);
        if ui
            .checkbox(&mut crisp_text, "Crisp magnified text")
            .changed()
        {
            graph_style.crisp_magnified_text = Some(crisp_text);
        }
    });

    ui.collapsing("Selection", |ui| {
        let default_select_stroke = Stroke::new(
            ui.style().visuals.selection.stroke.width,
            ui.style()
                .visuals
                .selection
                .stroke
                .color
                .gamma_multiply(0.5),
        );
        let mut select_stroke = graph_style.select_stoke.unwrap_or(default_select_stroke);
        if draw_stroke_editor(ui, "Stroke", &mut select_stroke) {
            graph_style.select_stoke = Some(select_stroke);
        }
        if ui.small_button("Stroke from theme").clicked() {
            graph_style.select_stoke = None;
        }

        let mut select_fill = graph_style
            .select_fill
            .unwrap_or(ui.style().visuals.selection.bg_fill.gamma_multiply(0.3));
        ui.horizontal(|ui| {
            ui.label("Fill");
            if ui.color_edit_button_srgba(&mut select_fill).changed() {
                graph_style.select_fill = Some(select_fill);
            }
            if ui.small_button("Theme").clicked() {
                graph_style.select_fill = None;
            }
        });

        let mut contained = graph_style.select_rect_contained.unwrap_or(false);
        if ui
            .checkbox(&mut contained, "Selection requires full containment")
            .changed()
        {
            graph_style.select_rect_contained = Some(contained);
        }

        let mut use_custom_style = graph_style.select_style.is_some();
        if ui
            .checkbox(
                &mut use_custom_style,
                "Use custom node selection frame style",
            )
            .changed()
        {
            if use_custom_style {
                graph_style.select_style = Some(SelectionStyle {
                    margin: ui.style().spacing.window_margin,
                    rounding: ui.style().visuals.window_corner_radius,
                    fill: select_fill,
                    stroke: select_stroke,
                });
            } else {
                graph_style.select_style = None;
            }
        }
        if let Some(mut selection_style) = graph_style.select_style {
            if draw_selection_style_editor(ui, &mut selection_style) {
                graph_style.select_style = Some(selection_style);
            }
        }
    });

    ui.collapsing("Frames", |ui| {
        let default_node_frame = egui::Frame::window(ui.style());
        let default_header_frame = default_node_frame.shadow(egui::epaint::Shadow::NONE);
        let default_bg_frame = egui::Frame::canvas(ui.style());

        let mut node_frame = graph_style.node_frame.unwrap_or(default_node_frame);
        if draw_frame_editor(ui, "Node Frame", &mut node_frame) {
            graph_style.node_frame = Some(node_frame);
        }
        if ui.small_button("Node frame from theme").clicked() {
            graph_style.node_frame = None;
        }

        let mut header_frame = graph_style.header_frame.unwrap_or(default_header_frame);
        if draw_frame_editor(ui, "Header Frame", &mut header_frame) {
            graph_style.header_frame = Some(header_frame);
        }
        if ui.small_button("Header frame from theme").clicked() {
            graph_style.header_frame = None;
        }

        let mut bg_frame = graph_style.bg_frame.unwrap_or(default_bg_frame);
        if draw_frame_editor(ui, "Background Frame", &mut bg_frame) {
            graph_style.bg_frame = Some(bg_frame);
        }
        if ui.small_button("Background frame from theme").clicked() {
            graph_style.bg_frame = None;
        }
    });
}

#[derive(Resource)]
pub struct VisualScriptEditorState {
    pub open: bool,
    pub active_path: Option<PathBuf>,
    pub documents: HashMap<PathBuf, VisualScriptOpenDocument>,
    pub graph_style: SnarlStyle,
    pub graph_style_popup_open: bool,
}

impl Default for VisualScriptEditorState {
    fn default() -> Self {
        Self {
            open: false,
            active_path: None,
            documents: HashMap::new(),
            graph_style: default_visual_graph_style(),
            graph_style_popup_open: false,
        }
    }
}

pub fn is_visual_script_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(VISUAL_SCRIPT_EXTENSION))
        .unwrap_or(false)
}

pub fn default_visual_script_template_full() -> String {
    let document = default_visual_script_document();
    visual_script_template_to_string(&document)
}

pub fn default_visual_script_template_third_person() -> String {
    let document = default_visual_script_document_third_person();
    visual_script_template_to_string(&document)
}

fn visual_script_template_to_string(document: &VisualScriptDocument) -> String {
    let pretty = PrettyConfig::new().compact_arrays(false);
    ron::ser::to_string_pretty(document, pretty)
        .unwrap_or_else(|_| "(\n    version: 3,\n)\n".to_string())
}

pub fn open_visual_script_editor(world: &mut World, path: &Path) -> Result<(), String> {
    if !is_visual_script_path(path) {
        return Err(format!(
            "{} is not a supported visual script file",
            path.to_string_lossy()
        ));
    }

    let path = path.to_path_buf();
    let mut load_error = None;

    world.resource_scope::<VisualScriptEditorState, _>(|_world, mut state| {
        if !state.documents.contains_key(&path) {
            match load_visual_script_document(&path) {
                Ok(document) => {
                    state.documents.insert(path.clone(), document);
                }
                Err(err) => {
                    load_error = Some(err);
                }
            }
        }

        if load_error.is_none() {
            state.open = true;
            state.active_path = Some(path.clone());
        }
    });

    if let Some(err) = load_error {
        return Err(err);
    }

    Ok(())
}

pub fn close_visual_script_editor(world: &mut World) {
    if let Some(mut state) = world.get_resource_mut::<VisualScriptEditorState>() {
        state.open = false;
    }
}

pub fn draw_visual_script_editor_tab(ui: &mut Ui, world: &mut World, path: &Path) {
    if let Err(err) = open_visual_script_editor(world, path) {
        ui.colored_label(
            Color32::from_rgb(214, 89, 89),
            format!("Failed to open visual script: {}", err),
        );
        return;
    }
    if let Some(mut state) = world.get_resource_mut::<VisualScriptEditorState>() {
        state.open = true;
        state.active_path = Some(path.to_path_buf());
    }
    draw_visual_script_editor_window(ui, world);
}

pub fn draw_visual_script_editor_window(ui: &mut Ui, world: &mut World) {
    let mut status_message: Option<String> = None;
    let project_root = world
        .get_resource::<EditorProject>()
        .and_then(|project| project.root.clone());
    world.resource_scope::<VisualScriptEditorState, _>(|_world, mut state| {
        if state.active_path.is_none() {
            let mut paths = state.documents.keys().cloned().collect::<Vec<_>>();
            paths.sort();
            state.active_path = paths.into_iter().next();
        }

        egui::ScrollArea::both()
            .id_salt("visual_script_editor_root_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    if state.documents.is_empty() {
                        ui.label("No visual script is open");
                        return;
                    }

                    let mut document_paths = state.documents.keys().cloned().collect::<Vec<_>>();
                    document_paths.sort();

                    let mut active_path = state.active_path.clone().unwrap_or_else(|| {
                        document_paths
                            .first()
                            .cloned()
                            .unwrap_or_else(|| PathBuf::from("<unknown>"))
                    });
                    if !state.documents.contains_key(&active_path) {
                        if let Some(first) = document_paths.first() {
                            active_path = first.clone();
                        } else {
                            ui.label("No visual script is open");
                            return;
                        }
                    }
                    let (active_graph_path, active_graph_is_function) = state
                        .documents
                        .get(&active_path)
                        .map(active_graph_path_display)
                        .unwrap_or_else(|| ("Root Event Graph".to_string(), false));
                    let active_graph_full_path = format!(
                        "{} / {}",
                        path_display_name(&active_path),
                        active_graph_path
                    );

                    ui.horizontal_wrapped(|ui| {
                        ui.label("Document:");
                        ui.monospace(compact_display_text(&active_graph_full_path, 88));
                        if active_graph_is_function && ui.small_button("Open Root").clicked() {
                            if let Some(document) = state.documents.get_mut(&active_path) {
                                document.active_graph_function = None;
                            }
                        }

                        if ui.button("Save").clicked() {
                            if let Some(document) = state.documents.get_mut(&active_path) {
                                match document.save() {
                                    Ok(()) => {
                                        status_message = Some(format!(
                                            "Saved visual script {}",
                                            document.path.to_string_lossy()
                                        ));
                                    }
                                    Err(err) => {
                                        status_message = Some(format!(
                                            "Failed to save visual script {}: {}",
                                            document.path.to_string_lossy(),
                                            err
                                        ));
                                    }
                                }
                            }
                        }

                        if ui.button("Save All").clicked() {
                            let mut saved_count = 0usize;
                            let mut first_error: Option<String> = None;
                            for document in state.documents.values_mut() {
                                if !document.dirty {
                                    continue;
                                }
                                match document.save() {
                                    Ok(()) => {
                                        saved_count += 1;
                                    }
                                    Err(err) => {
                                        if first_error.is_none() {
                                            first_error = Some(format!(
                                                "{}: {}",
                                                document.path.to_string_lossy(),
                                                err
                                            ));
                                        }
                                    }
                                }
                            }
                            status_message = Some(match first_error {
                                Some(err) => format!("Failed to save visual scripts: {}", err),
                                None => format!("Saved {} visual script(s)", saved_count),
                            });
                        }

                        if ui.button("Validate").clicked() {
                            if let Some(document) = state.documents.get_mut(&active_path) {
                                document.recompile_preview();
                                status_message = Some(match document.compile_error.as_ref() {
                                    Some(err) => {
                                        format!("Visual script validation failed: {}", err)
                                    }
                                    None => "Visual script validation succeeded".to_string(),
                                });
                            }
                        }

                        ui.separator();
                        let graph_style_button = ui.button("Graph Style");
                        if graph_style_button.clicked() {
                            state.graph_style_popup_open = !state.graph_style_popup_open;
                        }
                        let mut graph_style_popup_open = state.graph_style_popup_open;
                        let _ = egui::Popup::from_response(&graph_style_button)
                            .id(ui.id().with("visual_graph_style_popup"))
                            .open_bool(&mut graph_style_popup_open)
                            .kind(egui::PopupKind::Popup)
                            .layout(egui::Layout::top_down(egui::Align::Min))
                            .info(egui::UiStackInfo::new(egui::UiKind::Popup))
                            .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
                            .show(|ui| {
                                let max_height =
                                    (ui.ctx().content_rect().height() * 0.85).clamp(560.0, 1400.0);
                                egui::ScrollArea::vertical()
                                    .id_salt("visual_graph_style_menu_scroll")
                                    .max_height(max_height)
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {
                                        draw_visual_graph_style_menu(ui, &mut state.graph_style);
                                    });
                            });
                        state.graph_style_popup_open = graph_style_popup_open;
                    });

                    state.active_path = Some(active_path.clone());
                    let graph_style = state.graph_style;

                    let Some(document) = state.documents.get_mut(&active_path) else {
                        return;
                    };

                    ui.horizontal_wrapped(|ui| {
                        if document.dirty {
                            ui.colored_label(Color32::from_rgb(214, 89, 89), "Unsaved changes");
                        } else {
                            ui.label("Saved");
                        }

                        if document.compile_error.is_none() {
                            ui.colored_label(Color32::from_rgb(90, 166, 86), "Runtime Plan: OK");
                        } else {
                            ui.colored_label(
                                Color32::from_rgb(214, 89, 89),
                                "Runtime Plan: Failed",
                            );
                        }
                    });

                    ui.add(
                        egui::Label::new(format!("Path: {}", document.path.to_string_lossy()))
                            .wrap_mode(egui::TextWrapMode::Wrap),
                    );
                    if let Some(err) = document.compile_error.as_ref() {
                        ui.add(
                            egui::Label::new(
                                RichText::new(format!("Error: {}", err))
                                    .color(Color32::from_rgb(214, 89, 89)),
                            )
                            .wrap_mode(egui::TextWrapMode::Wrap),
                        );
                    }

                    ui.separator();

                    let mut changed = false;
                    let split_id = ui.id().with(("visual_script_split", &document.path));
                    let mut split_ratio = ui
                        .data_mut(|data| data.get_persisted::<f32>(split_id))
                        .unwrap_or(0.62)
                        .clamp(0.05, 0.95);

                    let content_rect = ui.ctx().content_rect();
                    let mut available_width = ui.available_width();
                    if !available_width.is_finite() {
                        available_width = 960.0;
                    }
                    available_width = available_width.clamp(1.0, content_rect.width().max(1.0));

                    let mut pane_height = ui.available_height();
                    if !pane_height.is_finite() {
                        pane_height = 540.0;
                    }
                    pane_height = pane_height.clamp(1.0, content_rect.height().max(1.0));
                    let handle_width = 8.0_f32.min(available_width);
                    let content_width = (available_width - handle_width).max(0.0);
                    let min_pane_width = 120.0_f32.min(content_width);
                    let min_graph_width = min_pane_width;
                    let min_side_width = min_pane_width;
                    let max_graph_width = (content_width - min_side_width).max(min_graph_width);
                    let mut graph_width =
                        (content_width * split_ratio).clamp(min_graph_width, max_graph_width);
                    let side_width = (content_width - graph_width).max(0.0);

                    ui.horizontal(|ui| {
                        ui.allocate_ui_with_layout(
                            egui::vec2(graph_width, pane_height),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                if let Some(active_function) = document.active_graph_function {
                                    if !document
                                        .functions
                                        .iter()
                                        .any(|function| function.id == active_function)
                                    {
                                        document.active_graph_function = None;
                                    } else if !document
                                        .function_snarls
                                        .contains_key(&active_function)
                                    {
                                        let snarl = document
                                            .functions
                                            .iter()
                                            .find(|function| function.id == active_function)
                                            .map(|function| graph_data_to_snarl(&function.graph))
                                            .unwrap_or_else(Snarl::new);
                                        document.function_snarls.insert(active_function, snarl);
                                    }
                                }

                                let before =
                                    if let Some(function_id) = document.active_graph_function {
                                        document
                                            .function_snarls
                                            .get(&function_id)
                                            .map(graph_data_from_snarl)
                                            .unwrap_or_default()
                                    } else {
                                        graph_data_from_snarl(&document.snarl)
                                    };
                                let mut viewer = VisualScriptViewer::with_context(
                                    &document.variables,
                                    &document.functions,
                                    document.active_graph_function,
                                    project_root.as_deref(),
                                );
                                let min_size = egui::vec2(120.0, 120.0);
                                let snarl_response =
                                    if let Some(function_id) = document.active_graph_function {
                                        let snarl = document
                                            .function_snarls
                                            .get_mut(&function_id)
                                            .expect("function snarl should exist");
                                        SnarlWidget::new()
                                            .id_salt((
                                                "visual_script_graph",
                                                &document.path,
                                                function_id,
                                            ))
                                            .style(graph_style)
                                            .min_size(min_size)
                                            .show(snarl, &mut viewer, ui)
                                    } else {
                                        SnarlWidget::new()
                                            .id_salt(("visual_script_graph", &document.path))
                                            .style(graph_style)
                                            .min_size(min_size)
                                            .show(&mut document.snarl, &mut viewer, ui)
                                    };

                                if let Some(payload) =
                                    typed_dnd_release_payload::<AssetDragPayload>(&snarl_response)
                                {
                                    if !viewer.consumed_asset_drop {
                                        let drop_pos = snarl_response
                                            .interact_pointer_pos()
                                            .or_else(|| {
                                                ui.ctx().input(|i| i.pointer.interact_pos())
                                            })
                                            .unwrap_or_else(|| ui.ctx().content_rect().center());
                                        for (index, path) in payload.paths.iter().enumerate() {
                                            viewer.pending_asset_drop_nodes.push((
                                                path_to_visual_literal(
                                                    path,
                                                    viewer.project_root.as_deref(),
                                                ),
                                                egui::pos2(
                                                    drop_pos.x + index as f32 * 22.0,
                                                    drop_pos.y + index as f32 * 18.0,
                                                ),
                                            ));
                                        }
                                    }
                                }

                                if !viewer.pending_asset_drop_nodes.is_empty() {
                                    for (value, pos) in
                                        std::mem::take(&mut viewer.pending_asset_drop_nodes)
                                    {
                                        if let Some(function_id) = document.active_graph_function {
                                            if let Some(snarl) =
                                                document.function_snarls.get_mut(&function_id)
                                            {
                                                snarl.insert_node(
                                                    pos,
                                                    VisualScriptNodeKind::StringLiteral { value },
                                                );
                                            }
                                        } else {
                                            document.snarl.insert_node(
                                                pos,
                                                VisualScriptNodeKind::StringLiteral { value },
                                            );
                                        }
                                    }
                                    viewer.mark_changed();
                                }

                                let mut graph_switched = false;
                                if let Some(function_id) = viewer.open_function_graph_request {
                                    if document
                                        .functions
                                        .iter()
                                        .any(|function| function.id == function_id)
                                    {
                                        document.active_graph_function = Some(function_id);
                                        graph_switched = true;
                                    }
                                }

                                let after =
                                    if let Some(function_id) = document.active_graph_function {
                                        document
                                            .function_snarls
                                            .get(&function_id)
                                            .map(graph_data_from_snarl)
                                            .unwrap_or_default()
                                    } else {
                                        graph_data_from_snarl(&document.snarl)
                                    };
                                if ((!graph_switched) && before != after) || viewer.changed {
                                    changed = true;
                                }
                            },
                        );

                        let (splitter_rect, splitter_response) = ui.allocate_exact_size(
                            egui::vec2(handle_width, pane_height),
                            Sense::click_and_drag(),
                        );
                        let splitter_color =
                            if splitter_response.dragged() || splitter_response.hovered() {
                                Color32::from_rgb(120, 120, 120)
                            } else {
                                Color32::from_rgb(74, 74, 74)
                            };
                        ui.painter().line_segment(
                            [splitter_rect.center_top(), splitter_rect.center_bottom()],
                            egui::Stroke::new(2.0, splitter_color),
                        );
                        if splitter_response.dragged() {
                            graph_width = (graph_width + splitter_response.drag_delta().x)
                                .clamp(min_graph_width, max_graph_width);
                            split_ratio = if content_width > 0.0 {
                                (graph_width / content_width).clamp(0.05, 0.95)
                            } else {
                                0.5
                            };
                        }

                        ui.allocate_ui_with_layout(
                            egui::vec2(side_width, pane_height),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                egui::ScrollArea::vertical()
                            .id_salt(("visual_script_side_panel_scroll", &document.path))
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                draw_variable_definitions_panel(ui, document, &mut changed);
                                ui.separator();
                                draw_function_definitions_panel(ui, document, &mut changed);
                                ui.separator();

                                ui.label(RichText::new("Notes").strong());
                                let side_text_width = ui.available_width().max(120.0);
                                let prelude_response = ui.add(
                                    TextEdit::multiline(&mut document.prelude)
                                        .desired_rows(8)
                                        .desired_width(side_text_width),
                                );
                                if prelude_response.changed() {
                                    changed = true;
                                }

                                ui.separator();
                                ui.label(RichText::new("Runtime Plan").strong());
                                if let Some(err) = document.compile_error.as_ref() {
                                    ui.colored_label(Color32::from_rgb(214, 89, 89), err);
                                }
                                let mut preview = if document.compile_error.is_none() {
                                    document.compile_preview.clone()
                                } else {
                                    "Validation failed. Fix graph issues to view the runtime plan"
                                        .to_string()
                                };
                                ui.add(
                                    TextEdit::multiline(&mut preview)
                                        .desired_rows(28)
                                        .desired_width(side_text_width)
                                        .interactive(false)
                                        .code_editor(),
                                );
                            });
                            },
                        );
                    });
                    ui.data_mut(|data| data.insert_persisted(split_id, split_ratio));

                    if prune_invalid_wires(&mut document.snarl, &document.variables) > 0 {
                        changed = true;
                    }
                    for snarl in document.function_snarls.values_mut() {
                        if prune_invalid_wires(snarl, &document.variables) > 0 {
                            changed = true;
                        }
                    }

                    if changed {
                        document.dirty = true;
                        document.recompile_preview();
                    }
                });
            });
    });

    if let Some(message) = status_message {
        set_status_message(world, message);
    }
}

pub fn compile_visual_script_file(path: &Path) -> Result<String, String> {
    let source = fs::read_to_string(path).map_err(|err| err.to_string())?;
    compile_visual_script_source(&source, &path.to_string_lossy())
}

pub fn compile_visual_script_source(source: &str, source_label: &str) -> Result<String, String> {
    let document = parse_visual_script_document(source)?;
    compile_visual_script_document(&document, source_label)
}

pub fn compile_visual_script_runtime_source(
    source: &str,
    source_label: &str,
) -> Result<VisualScriptProgram, String> {
    let document = parse_visual_script_document(source)?;
    compile_visual_script_program(&document, source_label)
}

pub fn parse_visual_script_document(source: &str) -> Result<VisualScriptDocument, String> {
    let mut document: VisualScriptDocument =
        ron::de::from_str(source).map_err(|err| err.to_string())?;
    normalize_document(&mut document);
    Ok(document)
}

pub fn compile_visual_script_document(
    document: &VisualScriptDocument,
    source_label: &str,
) -> Result<String, String> {
    let program = compile_visual_script_program(document, source_label)?;
    Ok(program.describe())
}

pub fn compile_visual_script_program(
    document: &VisualScriptDocument,
    source_label: &str,
) -> Result<VisualScriptProgram, String> {
    compile_visual_script_program_internal(document, source_label, false)
}

fn compile_visual_script_program_internal(
    document: &VisualScriptDocument,
    source_label: &str,
    is_function_program: bool,
) -> Result<VisualScriptProgram, String> {
    let mut variable_defaults = HashMap::new();
    let mut variable_types = HashMap::new();
    let mut variable_array_item_types = HashMap::new();
    let mut variable_ids = HashSet::new();
    let mut variable_names = HashSet::new();
    for variable in &document.variables {
        if variable.id == 0 {
            return Err(format!(
                "Variable '{}' has invalid id 0",
                variable.name.trim()
            ));
        }
        if !variable_ids.insert(variable.id) {
            return Err(format!("Duplicate visual variable id {}", variable.id));
        }
        let variable_name = variable.name.trim();
        if variable_name.is_empty() {
            return Err(format!(
                "Variable with id {} has an empty name",
                variable.id
            ));
        }
        if !variable_names.insert(variable_name.to_string()) {
            return Err(format!(
                "Duplicate visual variable name '{}'",
                variable_name
            ));
        }
        let key = runtime_variable_key(variable.id, variable_name);
        let parsed_default = parse_loose_literal(&variable.default_value);
        let coerced_default = coerce_json_to_visual_type_with_array_item(
            &parsed_default,
            variable.value_type,
            variable.array_item_type,
        )
        .map_err(|err| format!("Invalid default for variable '{}': {}", variable_name, err))?;
        variable_types.insert(key.clone(), variable.value_type);
        variable_array_item_types.insert(key.clone(), variable.array_item_type);
        variable_defaults.insert(key, coerced_default);
    }

    let mut function_ids = HashSet::new();
    let mut function_names = HashSet::new();
    if !is_function_program {
        for function in &document.functions {
            if function.id == 0 {
                return Err(format!(
                    "Function '{}' has invalid id 0",
                    function.name.trim()
                ));
            }
            if !function_ids.insert(function.id) {
                return Err(format!("Duplicate function id {}", function.id));
            }
            let function_name = function.name.trim();
            if function_name.is_empty() {
                return Err(format!(
                    "Function with id {} has an empty name",
                    function.id
                ));
            }
            if !function_names.insert(function_name.to_string()) {
                return Err(format!("Duplicate function name '{}'", function_name));
            }
        }
    }

    let mut nodes = HashMap::new();
    for node in &document.graph.nodes {
        if nodes.contains_key(&node.id) {
            return Err(format!("Duplicate visual node id {}", node.id));
        }
        let mut kind = node.kind.clone();
        kind.normalize();
        nodes.insert(node.id, kind);
    }

    if nodes.is_empty() {
        return Err("Visual script graph has no nodes".to_string());
    }

    let mut exec_edges: HashMap<(u64, usize), Vec<u64>> = HashMap::new();
    let mut data_edges: HashMap<(u64, usize), (u64, usize)> = HashMap::new();

    for wire in &document.graph.wires {
        let from_kind = nodes
            .get(&wire.from_node)
            .ok_or_else(|| format!("Wire references missing node {}", wire.from_node))?;
        let to_kind = nodes
            .get(&wire.to_node)
            .ok_or_else(|| format!("Wire references missing node {}", wire.to_node))?;

        let from_slot = from_kind.output_slot(wire.from_pin).ok_or_else(|| {
            format!(
                "Wire references invalid output {} on node {} ({})",
                wire.from_pin,
                wire.from_node,
                from_kind.title()
            )
        })?;

        let to_slot = to_kind.input_slot(wire.to_pin).ok_or_else(|| {
            format!(
                "Wire references invalid input {} on node {} ({})",
                wire.to_pin,
                wire.to_node,
                to_kind.title()
            )
        })?;

        if from_slot.kind != to_slot.kind {
            return Err(format!(
                "Pin type mismatch between {}:{} and {}:{}",
                wire.from_node, wire.from_pin, wire.to_node, wire.to_pin
            ));
        }

        if matches!(from_slot.kind, PinKind::Data) {
            let from_value_type =
                node_data_output_type(from_kind, from_slot.index, &document.variables).ok_or_else(
                    || {
                        format!(
                            "Wire references invalid data output {} on node {} ({})",
                            wire.from_pin,
                            wire.from_node,
                            from_kind.title()
                        )
                    },
                )?;
            let to_value_type = node_data_input_type(to_kind, to_slot.index, &document.variables)
                .ok_or_else(|| {
                format!(
                    "Wire references invalid data input {} on node {} ({})",
                    wire.to_pin,
                    wire.to_node,
                    to_kind.title()
                )
            })?;
            if !are_data_types_compatible(from_value_type, to_value_type) {
                let from_label = from_kind.output_label(from_slot);
                let to_label = to_kind.input_label(to_slot);
                return Err(format!(
                    "Data type mismatch: node {} ({}) output '{}' is {} but node {} ({}) input '{}' expects {}",
                    wire.from_node,
                    from_kind.title(),
                    from_label,
                    from_value_type.title(),
                    wire.to_node,
                    to_kind.title(),
                    to_label,
                    to_value_type.title()
                ));
            }
        }

        match from_slot.kind {
            PinKind::Exec => {
                exec_edges
                    .entry((wire.from_node, from_slot.index))
                    .or_default()
                    .push(wire.to_node);
            }
            PinKind::Data => {
                let key = (wire.to_node, to_slot.index);
                if data_edges.contains_key(&key) {
                    return Err(format!(
                        "Data input {} on node {} has multiple drivers",
                        to_slot.index + 1,
                        wire.to_node
                    ));
                }
                data_edges.insert(key, (wire.from_node, from_slot.index));
            }
        }
    }

    for targets in exec_edges.values_mut() {
        targets.sort_unstable();
        targets.dedup();
    }

    for (node_id, node) in &nodes {
        match node {
            VisualScriptNodeKind::CallApi { operation, .. } => {
                if !matches!(operation.spec().flow, VisualApiFlow::Exec) {
                    return Err(format!(
                        "Node {} ({}) expects an executable API operation",
                        node_id,
                        node.title()
                    ));
                }
            }
            VisualScriptNodeKind::QueryApi { operation, .. } => {
                if !matches!(operation.spec().flow, VisualApiFlow::Pure) {
                    return Err(format!(
                        "Node {} ({}) expects a pure query API operation",
                        node_id,
                        node.title()
                    ));
                }
            }
            VisualScriptNodeKind::SetVariable {
                variable_id, name, ..
            }
            | VisualScriptNodeKind::GetVariable {
                variable_id, name, ..
            }
            | VisualScriptNodeKind::ClearVariable {
                variable_id, name, ..
            } => {
                if find_variable_definition(&document.variables, *variable_id, name).is_none() {
                    return Err(format!(
                        "Node {} ({}) references an undefined variable",
                        node_id,
                        node.title()
                    ));
                }
            }
            VisualScriptNodeKind::CallFunction {
                function_id,
                name,
                inputs,
                outputs,
                ..
            } => {
                let function = find_function_definition(&document.functions, *function_id, name)
                    .ok_or_else(|| {
                        format!(
                            "Node {} ({}) references an undefined function",
                            node_id,
                            node.title()
                        )
                    })?;
                if inputs.len() != function.inputs.len() || outputs.len() != function.outputs.len()
                {
                    return Err(format!(
                        "Node {} ({}) has stale function signature; resync the function call node",
                        node_id,
                        node.title()
                    ));
                }
            }
            VisualScriptNodeKind::FunctionStart { .. }
            | VisualScriptNodeKind::FunctionReturn { .. } => {
                if !is_function_program {
                    return Err(format!(
                        "Node {} ({}) is only valid inside function graphs",
                        node_id,
                        node.title()
                    ));
                }
            }
            _ => {}
        }
    }

    let mut on_start_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnStart) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_update_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnUpdate) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_stop_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnStop) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_collision_enter_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnCollisionEnter) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_collision_stay_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnCollisionStay) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_collision_exit_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnCollisionExit) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_trigger_enter_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnTriggerEnter) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_trigger_exit_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnTriggerExit) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_input_action_pressed_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(
                node,
                VisualScriptNodeKind::OnInputAction {
                    phase: VisualInputActionPhase::Pressed,
                    ..
                }
            ) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_input_action_released_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(
                node,
                VisualScriptNodeKind::OnInputAction {
                    phase: VisualInputActionPhase::Released,
                    ..
                }
            ) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_input_action_down_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(
                node,
                VisualScriptNodeKind::OnInputAction {
                    phase: VisualInputActionPhase::Down,
                    ..
                }
            ) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut on_custom_event_nodes = nodes
        .iter()
        .filter_map(|(id, node)| {
            if matches!(node, VisualScriptNodeKind::OnCustomEvent { .. }) {
                Some(*id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    on_start_nodes.sort_unstable();
    on_update_nodes.sort_unstable();
    on_stop_nodes.sort_unstable();
    on_collision_enter_nodes.sort_unstable();
    on_collision_stay_nodes.sort_unstable();
    on_collision_exit_nodes.sort_unstable();
    on_trigger_enter_nodes.sort_unstable();
    on_trigger_exit_nodes.sort_unstable();
    on_input_action_pressed_nodes.sort_unstable();
    on_input_action_released_nodes.sort_unstable();
    on_input_action_down_nodes.sort_unstable();
    on_custom_event_nodes.sort_unstable();

    let has_root_events = !on_start_nodes.is_empty()
        || !on_update_nodes.is_empty()
        || !on_stop_nodes.is_empty()
        || !on_collision_enter_nodes.is_empty()
        || !on_collision_stay_nodes.is_empty()
        || !on_collision_exit_nodes.is_empty()
        || !on_trigger_enter_nodes.is_empty()
        || !on_trigger_exit_nodes.is_empty()
        || !on_input_action_pressed_nodes.is_empty()
        || !on_input_action_released_nodes.is_empty()
        || !on_input_action_down_nodes.is_empty()
        || !on_custom_event_nodes.is_empty();

    if !has_root_events {
        if is_function_program {
            on_start_nodes = nodes
                .iter()
                .filter_map(|(id, node)| {
                    if matches!(node, VisualScriptNodeKind::FunctionStart { .. }) {
                        Some(*id)
                    } else {
                        None
                    }
                })
                .collect();
            on_start_nodes.sort_unstable();
            if on_start_nodes.is_empty() {
                return Err(
                    "Function graph must contain at least one Function Start node".to_string(),
                );
            }
        } else {
            return Err("Visual script must contain at least one event node".to_string());
        }
    }

    let mut functions = HashMap::new();
    if !is_function_program {
        for function in &document.functions {
            let graph = function.graph.clone();

            let mut start_nodes = Vec::new();
            let mut return_nodes = Vec::new();
            for node in &graph.nodes {
                match &node.kind {
                    VisualScriptNodeKind::FunctionStart { .. } => start_nodes.push(node.id),
                    VisualScriptNodeKind::FunctionReturn { .. } => return_nodes.push(node.id),
                    VisualScriptNodeKind::OnStart
                    | VisualScriptNodeKind::OnUpdate
                    | VisualScriptNodeKind::OnStop
                    | VisualScriptNodeKind::OnCollisionEnter
                    | VisualScriptNodeKind::OnCollisionStay
                    | VisualScriptNodeKind::OnCollisionExit
                    | VisualScriptNodeKind::OnTriggerEnter
                    | VisualScriptNodeKind::OnTriggerExit
                    | VisualScriptNodeKind::OnInputAction { .. }
                    | VisualScriptNodeKind::OnCustomEvent { .. } => {
                        return Err(format!(
                            "Function '{}' cannot contain event nodes",
                            function.name
                        ));
                    }
                    _ => {}
                }
            }

            if start_nodes.len() != 1 {
                return Err(format!(
                    "Function '{}' must contain exactly one Function Start node",
                    function.name
                ));
            }
            if return_nodes.is_empty() {
                return Err(format!(
                    "Function '{}' must contain at least one Function Return node",
                    function.name
                ));
            }

            let mut function_document = VisualScriptDocument {
                version: VISUAL_SCRIPT_VERSION,
                name: function.name.clone(),
                prelude: String::new(),
                variables: document.variables.clone(),
                functions: document.functions.clone(),
                graph,
            };
            normalize_document(&mut function_document);

            let function_label = format!("{}::{}", source_label, function.name);
            let mut compiled =
                compile_visual_script_program_internal(&function_document, &function_label, true)?;
            compiled.functions.clear();
            compiled.function_context = Some((function.id, function.name.clone()));
            functions.insert(
                function.id,
                VisualCompiledFunctionProgram {
                    id: function.id,
                    name: function.name.clone(),
                    source_path: function.source_path.clone(),
                    inputs: function.inputs.clone(),
                    outputs: function.outputs.clone(),
                    program: Box::new(compiled),
                },
            );
        }
    }

    Ok(VisualScriptProgram {
        source_label: source_label.to_string(),
        source_name: path_display_name(Path::new(source_label)),
        nodes,
        exec_edges,
        data_edges,
        variable_defaults,
        variable_types,
        variable_array_item_types,
        on_start_nodes,
        on_update_nodes,
        on_stop_nodes,
        on_collision_enter_nodes,
        on_collision_stay_nodes,
        on_collision_exit_nodes,
        on_trigger_enter_nodes,
        on_trigger_exit_nodes,
        on_input_action_pressed_nodes,
        on_input_action_released_nodes,
        on_input_action_down_nodes,
        on_custom_event_nodes,
        functions,
        function_context: None,
    })
}

fn normalize_document(document: &mut VisualScriptDocument) {
    document.version = VISUAL_SCRIPT_VERSION;
    if document.graph.nodes.is_empty() {
        document.graph = default_visual_script_document().graph;
    }

    let mut function_seen_ids = HashSet::new();
    let mut function_seen_names = HashSet::new();
    let mut next_fn_id = next_function_id(&document.functions);
    for function in &mut document.functions {
        function.name = function.name.trim().to_string();
        function.source_path = function.source_path.trim().to_string();

        if function.id == 0 || !function_seen_ids.insert(function.id) {
            while function_seen_ids.contains(&next_fn_id) {
                next_fn_id = next_fn_id.saturating_add(1);
            }
            function.id = next_fn_id;
            function_seen_ids.insert(function.id);
            next_fn_id = next_fn_id.saturating_add(1);
        } else {
            next_fn_id = next_fn_id.max(function.id.saturating_add(1));
        }

        if function.name.is_empty() {
            function.name = format!("function_{}", function.id);
        }
        if !function_seen_names.insert(function.name.clone()) {
            let base = function.name.clone();
            let mut suffix = 2u32;
            loop {
                let candidate = format!("{}_{}", base, suffix);
                suffix = suffix.saturating_add(1);
                if function_seen_names.insert(candidate.clone()) {
                    function.name = candidate;
                    break;
                }
            }
        }

        normalize_function_io_ports(&mut function.inputs, "input");
        normalize_function_io_ports(&mut function.outputs, "output");
        sync_function_signature_nodes(function);
    }
    document.functions.sort_by_key(|function| function.id);

    let mut seen_ids = HashSet::new();
    let mut seen_names = HashSet::new();
    let mut next_id = 1u64;

    for variable in &mut document.variables {
        variable.name = variable.name.trim().to_string();
        if variable.id == 0 || !seen_ids.insert(variable.id) {
            while seen_ids.contains(&next_id) {
                next_id = next_id.saturating_add(1);
            }
            variable.id = next_id;
            seen_ids.insert(variable.id);
            next_id = next_id.saturating_add(1);
        } else {
            next_id = next_id.max(variable.id.saturating_add(1));
        }

        if variable.name.is_empty() {
            variable.name = format!("var_{}", variable.id);
        }
        if !seen_names.insert(variable.name.clone()) {
            let base = variable.name.clone();
            let mut suffix = 2u32;
            loop {
                let candidate = format!("{}_{}", base, suffix);
                suffix = suffix.saturating_add(1);
                if seen_names.insert(candidate.clone()) {
                    variable.name = candidate;
                    break;
                }
            }
        }
    }

    if document.name.trim() == "third_person_controller" {
        for variable in &mut document.variables {
            let variable_name = variable.name.trim();
            if matches!(
                variable_name,
                "camera_orbit_yaw" | "camera_orbit_pitch" | "camera_entity"
            ) {
                variable.inspector_exposed = false;
                variable.inspector_asset_kind = None;
            }
            if variable_name == "camera_name" {
                let default_value = variable.default_value.trim();
                if default_value.is_empty() || default_value == "Main Camera" {
                    variable.default_value = "Scene Camera".to_string();
                }
            }
            if variable_name == "camera_pitch_min" {
                let parsed = variable.default_value.trim().parse::<f64>().ok();
                if parsed.is_none_or(|value| value < -20.0) {
                    variable.default_value = "-20".to_string();
                }
            }
        }
    }

    let mut name_to_id = HashMap::new();
    for variable in &document.variables {
        name_to_id.insert(variable.name.clone(), variable.id);
    }

    for node in &mut document.graph.nodes {
        match &mut node.kind {
            VisualScriptNodeKind::SetVariable {
                variable_id,
                name,
                value,
            } => {
                let trimmed = name.trim().to_string();
                if *variable_id == 0 && !trimmed.is_empty() {
                    if let Some(existing) = name_to_id.get(&trimmed).copied() {
                        *variable_id = existing;
                    } else {
                        while seen_ids.contains(&next_id) {
                            next_id = next_id.saturating_add(1);
                        }
                        let new_id = next_id;
                        next_id = next_id.saturating_add(1);
                        let value_type = infer_visual_value_type_from_literal(value);
                        let mut array_item_type = if value_type == VisualValueType::Array {
                            infer_array_item_type_from_literal(value)
                        } else {
                            None
                        };
                        normalize_array_item_type(value_type, &mut array_item_type);
                        document.variables.push(VisualVariableDefinition {
                            id: new_id,
                            name: trimmed.clone(),
                            value_type,
                            array_item_type,
                            default_value: normalize_literal_for_data_type(
                                value,
                                value_type,
                                array_item_type,
                            ),
                            inspector_exposed: false,
                            inspector_label: String::new(),
                            inspector_asset_kind: None,
                        });
                        name_to_id.insert(trimmed, new_id);
                        seen_ids.insert(new_id);
                        *variable_id = new_id;
                    }
                }
                if *variable_id != 0 {
                    if let Some(variable) =
                        document.variables.iter().find(|var| var.id == *variable_id)
                    {
                        *name = variable.name.clone();
                    }
                }
            }
            VisualScriptNodeKind::GetVariable {
                variable_id,
                name,
                default_value,
            } => {
                let trimmed = name.trim().to_string();
                if *variable_id == 0 && !trimmed.is_empty() {
                    if let Some(existing) = name_to_id.get(&trimmed).copied() {
                        *variable_id = existing;
                    } else {
                        while seen_ids.contains(&next_id) {
                            next_id = next_id.saturating_add(1);
                        }
                        let new_id = next_id;
                        next_id = next_id.saturating_add(1);
                        let value_type = infer_visual_value_type_from_literal(default_value);
                        let mut array_item_type = if value_type == VisualValueType::Array {
                            infer_array_item_type_from_literal(default_value)
                        } else {
                            None
                        };
                        normalize_array_item_type(value_type, &mut array_item_type);
                        document.variables.push(VisualVariableDefinition {
                            id: new_id,
                            name: trimmed.clone(),
                            value_type,
                            array_item_type,
                            default_value: normalize_literal_for_data_type(
                                default_value,
                                value_type,
                                array_item_type,
                            ),
                            inspector_exposed: false,
                            inspector_label: String::new(),
                            inspector_asset_kind: None,
                        });
                        name_to_id.insert(trimmed, new_id);
                        seen_ids.insert(new_id);
                        *variable_id = new_id;
                    }
                }
                if *variable_id != 0 {
                    if let Some(variable) =
                        document.variables.iter().find(|var| var.id == *variable_id)
                    {
                        *name = variable.name.clone();
                    }
                }
            }
            VisualScriptNodeKind::ClearVariable {
                variable_id, name, ..
            } => {
                let trimmed = name.trim().to_string();
                if *variable_id == 0 && !trimmed.is_empty() {
                    if let Some(existing) = name_to_id.get(&trimmed).copied() {
                        *variable_id = existing;
                    } else {
                        while seen_ids.contains(&next_id) {
                            next_id = next_id.saturating_add(1);
                        }
                        let new_id = next_id;
                        next_id = next_id.saturating_add(1);
                        document.variables.push(VisualVariableDefinition {
                            id: new_id,
                            name: trimmed.clone(),
                            value_type: VisualValueType::String,
                            array_item_type: None,
                            default_value: String::new(),
                            inspector_exposed: false,
                            inspector_label: String::new(),
                            inspector_asset_kind: None,
                        });
                        name_to_id.insert(trimmed, new_id);
                        seen_ids.insert(new_id);
                        *variable_id = new_id;
                    }
                }
                if *variable_id != 0 {
                    if let Some(variable) =
                        document.variables.iter().find(|var| var.id == *variable_id)
                    {
                        *name = variable.name.clone();
                    }
                }
            }
            VisualScriptNodeKind::CallFunction {
                function_id,
                name,
                inputs,
                outputs,
                args,
            } => {
                if let Some(function) =
                    find_function_definition(&document.functions, *function_id, name)
                {
                    *function_id = function.id;
                    *name = function.name.clone();
                    *inputs = function.inputs.clone();
                    *outputs = function.outputs.clone();
                    args.truncate(inputs.len());
                    while args.len() < inputs.len() {
                        let value_type = inputs
                            .get(args.len())
                            .map(|port| port.value_type)
                            .unwrap_or(VisualValueType::Any);
                        args.push(default_literal_for_type(value_type).to_string());
                    }
                }
            }
            _ => {}
        }

        node.kind.normalize();
    }

    document.variables.sort_by_key(|var| var.id);
    for variable in &mut document.variables {
        normalize_array_item_type(variable.value_type, &mut variable.array_item_type);
        if variable.default_value.trim().is_empty() {
            variable.default_value = default_literal_for_type(variable.value_type).to_string();
        } else {
            variable.default_value = normalize_literal_for_data_type(
                &variable.default_value,
                variable.value_type,
                variable.array_item_type,
            );
        }
        variable.inspector_label = variable.inspector_label.trim().to_string();
        if !variable.inspector_exposed
            || !visual_variable_supports_asset_kind(variable.value_type, variable.array_item_type)
        {
            variable.inspector_asset_kind = None;
        }
    }

    normalize_graph_wires(&mut document.graph, &document.variables);
    let function_defs = document.functions.clone();
    for function in &mut document.functions {
        for node in &mut function.graph.nodes {
            if let VisualScriptNodeKind::CallFunction {
                function_id,
                name,
                inputs,
                outputs,
                args,
            } = &mut node.kind
            {
                if let Some(target) = find_function_definition(&function_defs, *function_id, name) {
                    *function_id = target.id;
                    *name = target.name.clone();
                    *inputs = target.inputs.clone();
                    *outputs = target.outputs.clone();
                    args.truncate(inputs.len());
                    while args.len() < inputs.len() {
                        let value_type = inputs
                            .get(args.len())
                            .map(|port| port.value_type)
                            .unwrap_or(VisualValueType::Any);
                        args.push(default_literal_for_type(value_type).to_string());
                    }
                }
            }
            node.kind.normalize();
        }
        normalize_graph_wires(&mut function.graph, &document.variables);
    }
}

fn normalize_graph_wires(
    graph: &mut VisualScriptGraphData,
    variables: &[VisualVariableDefinition],
) {
    let mut node_map = HashMap::new();
    for node in &graph.nodes {
        node_map.insert(node.id, node.kind.clone());
    }

    let mut seen_wires = HashSet::new();
    let mut data_input_drivers = HashSet::new();
    graph.wires.retain(|wire| {
        let key = (wire.from_node, wire.from_pin, wire.to_node, wire.to_pin);
        if !seen_wires.insert(key) {
            return false;
        }

        let Some(from_node) = node_map.get(&wire.from_node) else {
            return false;
        };
        let Some(to_node) = node_map.get(&wire.to_node) else {
            return false;
        };

        let Some(from_slot) = from_node.output_slot(wire.from_pin) else {
            return false;
        };
        let Some(to_slot) = to_node.input_slot(wire.to_pin) else {
            return false;
        };

        if from_slot.kind != to_slot.kind {
            return false;
        }

        if matches!(from_slot.kind, PinKind::Data) {
            let Some(from_type) = node_data_output_type(from_node, from_slot.index, variables)
            else {
                return false;
            };
            let Some(to_type) = node_data_input_type(to_node, to_slot.index, variables) else {
                return false;
            };
            if !are_data_types_compatible(from_type, to_type) {
                return false;
            }
            if !data_input_drivers.insert((wire.to_node, to_slot.index)) {
                return false;
            }
        }

        true
    });
}

fn load_visual_script_document(path: &Path) -> Result<VisualScriptOpenDocument, String> {
    let source = fs::read_to_string(path).map_err(|err| err.to_string())?;
    let document = parse_visual_script_document(&source)?;
    Ok(VisualScriptOpenDocument::from_document(path, document))
}

fn graph_data_to_snarl(data: &VisualScriptGraphData) -> Snarl<VisualScriptNodeKind> {
    let mut snarl = Snarl::new();
    let mut node_map: HashMap<u64, NodeId> = HashMap::new();

    let mut nodes = data.nodes.clone();
    nodes.sort_by_key(|node| node.id);
    for mut node in nodes {
        node.kind.normalize();
        let pos = egui::pos2(node.pos[0] as f32, node.pos[1] as f32);
        let node_id = snarl.insert_node(pos, node.kind);
        snarl.open_node(node_id, node.open);
        node_map.insert(node.id, node_id);
    }

    for wire in &data.wires {
        let Some(&from_node) = node_map.get(&wire.from_node) else {
            continue;
        };
        let Some(&to_node) = node_map.get(&wire.to_node) else {
            continue;
        };
        let Some(from) = snarl.get_node(from_node) else {
            continue;
        };
        let Some(to) = snarl.get_node(to_node) else {
            continue;
        };
        if wire.from_pin >= from.output_count() || wire.to_pin >= to.input_count() {
            continue;
        }
        snarl.connect(
            OutPinId {
                node: from_node,
                output: wire.from_pin,
            },
            InPinId {
                node: to_node,
                input: wire.to_pin,
            },
        );
    }

    snarl
}

fn graph_data_from_snarl(snarl: &Snarl<VisualScriptNodeKind>) -> VisualScriptGraphData {
    let mut nodes = Vec::new();
    let mut node_ids = HashMap::new();

    for (node_id, node) in snarl.nodes_ids_data() {
        let stable_id = node_id.0 as u64;
        node_ids.insert(node_id, stable_id);
        nodes.push(VisualScriptNodeRecord {
            id: stable_id,
            kind: node.value.clone(),
            pos: [node.pos.x.round() as i32, node.pos.y.round() as i32],
            open: node.open,
        });
    }

    nodes.sort_by_key(|node| node.id);

    let mut wires = Vec::new();
    for (from, to) in snarl.wires() {
        let Some(&from_node) = node_ids.get(&from.node) else {
            continue;
        };
        let Some(&to_node) = node_ids.get(&to.node) else {
            continue;
        };
        wires.push(VisualScriptWireRecord {
            from_node,
            from_pin: from.output,
            to_node,
            to_pin: to.input,
        });
    }
    wires.sort_by_key(|wire| (wire.from_node, wire.from_pin, wire.to_node, wire.to_pin));
    wires.dedup();

    VisualScriptGraphData { nodes, wires }
}

fn template_insert_api_node(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    pos: egui::Pos2,
    operation: VisualApiOperation,
) -> NodeId {
    snarl.insert_node(pos, api_node_kind_from_spec(operation.spec()))
}

fn template_set_api_arg(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    node: NodeId,
    arg_index: usize,
    value: &str,
) {
    let Some(node) = snarl.get_node_mut(node) else {
        return;
    };
    match node {
        VisualScriptNodeKind::CallApi { args, .. }
        | VisualScriptNodeKind::QueryApi { args, .. } => {
            if let Some(arg) = args.get_mut(arg_index) {
                *arg = value.to_string();
            }
        }
        _ => {}
    }
}

fn template_connect_exec(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    from_node: NodeId,
    from_exec_pin: usize,
    to_node: NodeId,
    to_exec_pin: usize,
) {
    snarl.connect(
        OutPinId {
            node: from_node,
            output: from_exec_pin,
        },
        InPinId {
            node: to_node,
            input: to_exec_pin,
        },
    );
}

fn template_connect_data(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    from_node: NodeId,
    from_data_pin: usize,
    to_node: NodeId,
    to_data_pin: usize,
) {
    let from_pin = snarl
        .get_node(from_node)
        .map(|node| node.exec_output_count() + from_data_pin)
        .unwrap_or(from_data_pin);
    let to_pin = snarl
        .get_node(to_node)
        .map(|node| node.exec_input_count() + to_data_pin)
        .unwrap_or(to_data_pin);
    snarl.connect(
        OutPinId {
            node: from_node,
            output: from_pin,
        },
        InPinId {
            node: to_node,
            input: to_pin,
        },
    );
}

fn template_get_variable_node(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    pos: egui::Pos2,
    variable_id: u64,
    name: &str,
) -> NodeId {
    snarl.insert_node(
        pos,
        VisualScriptNodeKind::GetVariable {
            variable_id,
            name: name.to_string(),
            default_value: String::new(),
        },
    )
}

fn template_set_variable_node(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    pos: egui::Pos2,
    variable_id: u64,
    name: &str,
    value: &str,
) -> NodeId {
    snarl.insert_node(
        pos,
        VisualScriptNodeKind::SetVariable {
            variable_id,
            name: name.to_string(),
            value: value.to_string(),
        },
    )
}

fn template_number_node(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    pos: egui::Pos2,
    value: f64,
) -> NodeId {
    snarl.insert_node(pos, VisualScriptNodeKind::NumberLiteral { value })
}

fn template_bool_node(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    pos: egui::Pos2,
    value: bool,
) -> NodeId {
    snarl.insert_node(pos, VisualScriptNodeKind::BoolLiteral { value })
}

fn default_visual_script_document_third_person() -> VisualScriptDocument {
    const VAR_CAMERA_ENTITY: u64 = 1;
    const VAR_CAMERA_NAME: u64 = 2;
    const VAR_CAMERA_PIVOT_HEIGHT: u64 = 3;
    const VAR_CAMERA_MIN_DISTANCE: u64 = 4;
    const VAR_CAMERA_MAX_DISTANCE: u64 = 5;
    const VAR_CAMERA_ORBIT_YAW: u64 = 6;
    const VAR_CAMERA_ORBIT_PITCH: u64 = 7;
    const VAR_CAMERA_PITCH_MIN: u64 = 8;
    const VAR_CAMERA_PITCH_MAX: u64 = 9;
    const VAR_SPEED_WALK: u64 = 10;
    const VAR_SPEED_RUN: u64 = 11;
    const VAR_SPEED_SPRINT: u64 = 12;
    const VAR_SPEED_CROUCH: u64 = 13;
    const VAR_IS_SPRINTING: u64 = 14;
    const VAR_IS_CROUCHING: u64 = 15;
    const VAR_JUMP_QUEUED: u64 = 16;
    const VAR_LEFT_FOOT_GROUNDED: u64 = 17;
    const VAR_RIGHT_FOOT_GROUNDED: u64 = 18;
    const VAR_LEFT_FOOT_OFFSET: u64 = 19;
    const VAR_RIGHT_FOOT_OFFSET: u64 = 20;
    const VAR_PUSH_IMPULSE: u64 = 21;
    const VAR_MOVE_MAGNITUDE: u64 = 22;
    const VAR_MOVE_WORLD_DIRECTION: u64 = 23;
    const VAR_CAMERA_RUNTIME_ENTITY: u64 = 24;
    const VAR_CAMERA_STICK_SENSITIVITY: u64 = 25;
    const VAR_CAMERA_MOUSE_SENSITIVITY: u64 = 26;
    const VAR_JUMP_SPEED: u64 = 27;
    const VAR_GRAVITY_ACCELERATION: u64 = 28;
    const VAR_VERTICAL_VELOCITY: u64 = 29;
    const VAR_TURN_SMOOTH_SPEED: u64 = 30;
    const VAR_BASE_YAW_OFFSET: u64 = 31;
    const VAR_CLIP_IDLE: u64 = 32;
    const VAR_CLIP_WALK: u64 = 33;
    const VAR_CLIP_RUN: u64 = 34;
    const VAR_CLIP_SPRINT: u64 = 35;
    const VAR_CLIP_JUMP: u64 = 36;
    const VAR_CLIP_CROUCH: u64 = 37;
    const VAR_ACTIVE_ANIM_CLIP: u64 = 38;

    let make_var = |id: u64,
                    name: &str,
                    value_type: VisualValueType,
                    default_value: &str,
                    inspector_exposed: bool| {
        VisualVariableDefinition {
            id,
            name: name.to_string(),
            value_type,
            array_item_type: None,
            default_value: default_value.to_string(),
            inspector_exposed,
            inspector_label: String::new(),
            inspector_asset_kind: None,
        }
    };

    let variables = vec![
        make_var(
            VAR_CAMERA_ENTITY,
            "camera_entity",
            VisualValueType::Entity,
            "0",
            false,
        ),
        make_var(
            VAR_CAMERA_NAME,
            "camera_name",
            VisualValueType::String,
            "Scene Camera",
            true,
        ),
        make_var(
            VAR_CAMERA_PIVOT_HEIGHT,
            "camera_pivot_height",
            VisualValueType::Number,
            "1.65",
            true,
        ),
        make_var(
            VAR_CAMERA_MIN_DISTANCE,
            "camera_min_distance",
            VisualValueType::Number,
            "0.9",
            true,
        ),
        make_var(
            VAR_CAMERA_MAX_DISTANCE,
            "camera_max_distance",
            VisualValueType::Number,
            "4.6",
            true,
        ),
        make_var(
            VAR_CAMERA_ORBIT_YAW,
            "camera_orbit_yaw",
            VisualValueType::Number,
            "0",
            false,
        ),
        make_var(
            VAR_CAMERA_ORBIT_PITCH,
            "camera_orbit_pitch",
            VisualValueType::Number,
            "18",
            false,
        ),
        make_var(
            VAR_CAMERA_PITCH_MIN,
            "camera_pitch_min",
            VisualValueType::Number,
            "-20",
            true,
        ),
        make_var(
            VAR_CAMERA_PITCH_MAX,
            "camera_pitch_max",
            VisualValueType::Number,
            "70",
            true,
        ),
        make_var(
            VAR_CAMERA_STICK_SENSITIVITY,
            "camera_stick_sensitivity",
            VisualValueType::Number,
            "140",
            true,
        ),
        make_var(
            VAR_CAMERA_MOUSE_SENSITIVITY,
            "camera_mouse_sensitivity",
            VisualValueType::Number,
            "0.12",
            true,
        ),
        make_var(
            VAR_SPEED_WALK,
            "move_speed_walk",
            VisualValueType::Number,
            "2.4",
            true,
        ),
        make_var(
            VAR_SPEED_RUN,
            "move_speed_run",
            VisualValueType::Number,
            "4.2",
            true,
        ),
        make_var(
            VAR_SPEED_SPRINT,
            "move_speed_sprint",
            VisualValueType::Number,
            "6.8",
            true,
        ),
        make_var(
            VAR_SPEED_CROUCH,
            "move_speed_crouch",
            VisualValueType::Number,
            "1.55",
            true,
        ),
        make_var(
            VAR_IS_SPRINTING,
            "is_sprinting",
            VisualValueType::Bool,
            "false",
            false,
        ),
        make_var(
            VAR_IS_CROUCHING,
            "is_crouching",
            VisualValueType::Bool,
            "false",
            false,
        ),
        make_var(
            VAR_JUMP_QUEUED,
            "jump_queued",
            VisualValueType::Bool,
            "false",
            false,
        ),
        make_var(
            VAR_LEFT_FOOT_GROUNDED,
            "left_foot_grounded",
            VisualValueType::Bool,
            "false",
            false,
        ),
        make_var(
            VAR_RIGHT_FOOT_GROUNDED,
            "right_foot_grounded",
            VisualValueType::Bool,
            "false",
            false,
        ),
        make_var(
            VAR_LEFT_FOOT_OFFSET,
            "left_foot_offset",
            VisualValueType::Vec3,
            "{\"x\":-0.17,\"y\":0.18,\"z\":0.09}",
            true,
        ),
        make_var(
            VAR_RIGHT_FOOT_OFFSET,
            "right_foot_offset",
            VisualValueType::Vec3,
            "{\"x\":0.17,\"y\":0.18,\"z\":0.09}",
            true,
        ),
        make_var(
            VAR_PUSH_IMPULSE,
            "push_impulse",
            VisualValueType::Number,
            "3.5",
            true,
        ),
        make_var(
            VAR_JUMP_SPEED,
            "jump_speed",
            VisualValueType::Number,
            "5.4",
            true,
        ),
        make_var(
            VAR_GRAVITY_ACCELERATION,
            "gravity_acceleration",
            VisualValueType::Number,
            "-18.0",
            true,
        ),
        make_var(
            VAR_TURN_SMOOTH_SPEED,
            "turn_smooth_speed",
            VisualValueType::Number,
            "12.0",
            true,
        ),
        make_var(
            VAR_CLIP_IDLE,
            "clip_idle",
            VisualValueType::String,
            "idle",
            true,
        ),
        make_var(
            VAR_CLIP_WALK,
            "clip_walk",
            VisualValueType::String,
            "walk",
            true,
        ),
        make_var(
            VAR_CLIP_RUN,
            "clip_run",
            VisualValueType::String,
            "run",
            true,
        ),
        make_var(
            VAR_CLIP_SPRINT,
            "clip_sprint",
            VisualValueType::String,
            "sprint",
            true,
        ),
        make_var(
            VAR_CLIP_JUMP,
            "clip_jump",
            VisualValueType::String,
            "jump",
            true,
        ),
        make_var(
            VAR_CLIP_CROUCH,
            "clip_crouch",
            VisualValueType::String,
            "crouch",
            true,
        ),
        make_var(
            VAR_ACTIVE_ANIM_CLIP,
            "active_anim_clip",
            VisualValueType::String,
            "",
            false,
        ),
        make_var(
            VAR_MOVE_MAGNITUDE,
            "move_magnitude",
            VisualValueType::Number,
            "0",
            false,
        ),
        make_var(
            VAR_MOVE_WORLD_DIRECTION,
            "move_world_direction",
            VisualValueType::Vec3,
            "{\"x\":0,\"y\":0,\"z\":0}",
            false,
        ),
        make_var(
            VAR_CAMERA_RUNTIME_ENTITY,
            "camera_runtime_entity",
            VisualValueType::Entity,
            "0",
            false,
        ),
        make_var(
            VAR_VERTICAL_VELOCITY,
            "vertical_velocity",
            VisualValueType::Number,
            "0",
            false,
        ),
        make_var(
            VAR_BASE_YAW_OFFSET,
            "base_yaw_offset",
            VisualValueType::Number,
            "0",
            false,
        ),
    ];

    let mut snarl = Snarl::new();
    let self_entity = snarl.insert_node(egui::pos2(20.0, 20.0), VisualScriptNodeKind::SelfEntity);

    let on_start = snarl.insert_node(egui::pos2(40.0, 90.0), VisualScriptNodeKind::OnStart);
    let on_stop = snarl.insert_node(egui::pos2(40.0, 190.0), VisualScriptNodeKind::OnStop);
    let on_update = snarl.insert_node(egui::pos2(40.0, 820.0), VisualScriptNodeKind::OnUpdate);
    let on_sprint_down = snarl.insert_node(
        egui::pos2(40.0, 340.0),
        VisualScriptNodeKind::OnInputAction {
            action: "sprint".to_string(),
            phase: VisualInputActionPhase::Down,
        },
    );
    let on_sprint_released = snarl.insert_node(
        egui::pos2(40.0, 430.0),
        VisualScriptNodeKind::OnInputAction {
            action: "sprint".to_string(),
            phase: VisualInputActionPhase::Released,
        },
    );
    let on_crouch_pressed = snarl.insert_node(
        egui::pos2(40.0, 520.0),
        VisualScriptNodeKind::OnInputAction {
            action: "crouch".to_string(),
            phase: VisualInputActionPhase::Pressed,
        },
    );
    let on_jump_pressed = snarl.insert_node(
        egui::pos2(40.0, 610.0),
        VisualScriptNodeKind::OnInputAction {
            action: "jump".to_string(),
            phase: VisualInputActionPhase::Pressed,
        },
    );

    let on_update_seq = snarl.insert_node(
        egui::pos2(300.0, 820.0),
        VisualScriptNodeKind::Sequence { outputs: 4 },
    );
    template_connect_exec(&mut snarl, on_update, 0, on_update_seq, 0);

    let start_set_context = template_insert_api_node(
        &mut snarl,
        egui::pos2(320.0, 90.0),
        VisualApiOperation::InputSetActionContext,
    );
    template_set_api_arg(&mut snarl, start_set_context, 0, "gameplay");
    template_connect_exec(&mut snarl, on_start, 0, start_set_context, 0);

    let stop_reset_cursor = template_insert_api_node(
        &mut snarl,
        egui::pos2(320.0, 190.0),
        VisualApiOperation::InputResetCursorControl,
    );
    let stop_set_context = template_insert_api_node(
        &mut snarl,
        egui::pos2(560.0, 190.0),
        VisualApiOperation::InputSetActionContext,
    );
    template_set_api_arg(&mut snarl, stop_set_context, 0, "default");
    template_connect_exec(&mut snarl, on_stop, 0, stop_reset_cursor, 0);
    template_connect_exec(&mut snarl, stop_reset_cursor, 0, stop_set_context, 0);

    let mut start_cursor = start_set_context;
    let start_bindings = [
        ("move_forward", "w", "gameplay", "0.0"),
        ("move_backward", "s", "gameplay", "0.0"),
        ("move_left", "a", "gameplay", "0.0"),
        ("move_right", "d", "gameplay", "0.0"),
        ("jump", "space", "gameplay", "0.0"),
        ("jump", "south", "gameplay", "0.0"),
        ("sprint", "shiftleft", "gameplay", "0.0"),
        ("sprint", "lefttrigger", "gameplay", "0.25"),
        ("crouch", "ctrlleft", "gameplay", "0.0"),
        ("crouch", "righttrigger", "gameplay", "0.25"),
        ("look_x", "rightx", "gameplay", "0.1"),
        ("look_y", "righty", "gameplay", "0.1"),
    ];
    let mut bind_x = 620.0;
    for (action, binding, context, deadzone) in start_bindings {
        let bind_node = template_insert_api_node(
            &mut snarl,
            egui::pos2(bind_x, 90.0),
            VisualApiOperation::InputBindAction,
        );
        template_set_api_arg(&mut snarl, bind_node, 0, action);
        template_set_api_arg(&mut snarl, bind_node, 1, binding);
        template_set_api_arg(&mut snarl, bind_node, 2, context);
        template_set_api_arg(&mut snarl, bind_node, 3, deadzone);
        template_connect_exec(&mut snarl, start_cursor, 0, bind_node, 0);
        start_cursor = bind_node;
        bind_x += 300.0;
    }

    let start_cursor_visible = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::InputSetCursorVisible,
    );
    template_set_api_arg(&mut snarl, start_cursor_visible, 0, "false");
    template_connect_exec(&mut snarl, start_cursor, 0, start_cursor_visible, 0);
    start_cursor = start_cursor_visible;
    bind_x += 280.0;

    let start_cursor_grab = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::InputSetCursorGrab,
    );
    template_set_api_arg(&mut snarl, start_cursor_grab, 0, "locked");
    template_connect_exec(&mut snarl, start_cursor, 0, start_cursor_grab, 0);
    start_cursor = start_cursor_grab;
    bind_x += 300.0;

    let find_camera = template_insert_api_node(
        &mut snarl,
        egui::pos2(620.0, 260.0),
        VisualApiOperation::EcsFindEntityByName,
    );
    let camera_name_var = template_get_variable_node(
        &mut snarl,
        egui::pos2(620.0, 340.0),
        VAR_CAMERA_NAME,
        "camera_name",
    );
    let camera_entity_override = template_get_variable_node(
        &mut snarl,
        egui::pos2(860.0, 340.0),
        VAR_CAMERA_ENTITY,
        "camera_entity",
    );
    let has_camera_override = template_insert_api_node(
        &mut snarl,
        egui::pos2(1200.0, 340.0),
        VisualApiOperation::EcsEntityExists,
    );
    let resolved_camera_entity = snarl.insert_node(
        egui::pos2(1380.0, 340.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::Entity,
        },
    );
    template_connect_data(&mut snarl, camera_name_var, 0, find_camera, 0);
    template_connect_data(
        &mut snarl,
        camera_entity_override,
        0,
        has_camera_override,
        0,
    );
    template_connect_data(
        &mut snarl,
        has_camera_override,
        0,
        resolved_camera_entity,
        0,
    );
    template_connect_data(
        &mut snarl,
        camera_entity_override,
        0,
        resolved_camera_entity,
        1,
    );
    template_connect_data(&mut snarl, find_camera, 0, resolved_camera_entity, 2);
    let set_camera_runtime_entity = template_set_variable_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VAR_CAMERA_RUNTIME_ENTITY,
        "camera_runtime_entity",
        "0",
    );
    template_connect_exec(&mut snarl, start_cursor, 0, set_camera_runtime_entity, 0);
    template_connect_data(
        &mut snarl,
        resolved_camera_entity,
        0,
        set_camera_runtime_entity,
        0,
    );
    start_cursor = set_camera_runtime_entity;
    bind_x += 320.0;

    let start_reset_vertical_velocity = template_set_variable_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VAR_VERTICAL_VELOCITY,
        "vertical_velocity",
        "0",
    );
    template_connect_exec(
        &mut snarl,
        start_cursor,
        0,
        start_reset_vertical_velocity,
        0,
    );
    start_cursor = start_reset_vertical_velocity;
    bind_x += 320.0;

    let start_reset_active_anim_clip = template_set_variable_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VAR_ACTIVE_ANIM_CLIP,
        "active_anim_clip",
        "",
    );
    template_connect_exec(&mut snarl, start_cursor, 0, start_reset_active_anim_clip, 0);
    start_cursor = start_reset_active_anim_clip;
    bind_x += 320.0;

    let start_add_transform = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::EcsAddComponent,
    );
    template_set_api_arg(&mut snarl, start_add_transform, 1, "transform");
    template_connect_exec(&mut snarl, start_cursor, 0, start_add_transform, 0);
    template_connect_data(&mut snarl, self_entity, 0, start_add_transform, 0);
    start_cursor = start_add_transform;
    bind_x += 320.0;

    let start_add_collider_shape = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::EcsAddComponent,
    );
    template_set_api_arg(&mut snarl, start_add_collider_shape, 1, "collider_shape");
    template_connect_exec(&mut snarl, start_cursor, 0, start_add_collider_shape, 0);
    template_connect_data(&mut snarl, self_entity, 0, start_add_collider_shape, 0);
    start_cursor = start_add_collider_shape;
    bind_x += 320.0;

    let start_add_kinematic = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::EcsAddComponent,
    );
    template_set_api_arg(&mut snarl, start_add_kinematic, 1, "kinematic_rigid_body");
    template_connect_exec(&mut snarl, start_cursor, 0, start_add_kinematic, 0);
    template_connect_data(&mut snarl, self_entity, 0, start_add_kinematic, 0);
    start_cursor = start_add_kinematic;
    bind_x += 320.0;

    let start_add_character_controller = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::EcsAddComponent,
    );
    template_set_api_arg(
        &mut snarl,
        start_add_character_controller,
        1,
        "character_controller",
    );
    template_connect_exec(
        &mut snarl,
        start_cursor,
        0,
        start_add_character_controller,
        0,
    );
    template_connect_data(
        &mut snarl,
        self_entity,
        0,
        start_add_character_controller,
        0,
    );
    start_cursor = start_add_character_controller;
    bind_x += 320.0;

    let start_get_self_forward = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::EcsGetTransformForward,
    );
    template_connect_exec(&mut snarl, start_cursor, 0, start_get_self_forward, 0);
    template_connect_data(&mut snarl, self_entity, 0, start_get_self_forward, 0);
    start_cursor = start_get_self_forward;
    bind_x += 320.0;

    let start_forward_x = snarl.insert_node(
        egui::pos2(bind_x - 80.0, 220.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let start_forward_z = snarl.insert_node(
        egui::pos2(bind_x - 80.0, 300.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, start_get_self_forward, 0, start_forward_x, 0);
    template_connect_data(&mut snarl, start_get_self_forward, 0, start_forward_z, 0);
    let start_base_yaw = snarl.insert_node(
        egui::pos2(bind_x + 120.0, 260.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Atan2,
        },
    );
    template_connect_data(&mut snarl, start_forward_x, 0, start_base_yaw, 0);
    template_connect_data(&mut snarl, start_forward_z, 0, start_base_yaw, 1);

    let start_set_base_yaw_offset = template_set_variable_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VAR_BASE_YAW_OFFSET,
        "base_yaw_offset",
        "0",
    );
    template_connect_exec(&mut snarl, start_cursor, 0, start_set_base_yaw_offset, 0);
    template_connect_data(&mut snarl, start_base_yaw, 0, start_set_base_yaw_offset, 0);
    start_cursor = start_set_base_yaw_offset;
    bind_x += 320.0;

    let camera_start_offset =
        snarl.insert_node(egui::pos2(1120.0, 260.0), VisualScriptNodeKind::Vec3);
    let camera_start_y = template_get_variable_node(
        &mut snarl,
        egui::pos2(980.0, 350.0),
        VAR_CAMERA_PIVOT_HEIGHT,
        "camera_pivot_height",
    );
    let camera_start_z = template_get_variable_node(
        &mut snarl,
        egui::pos2(1260.0, 350.0),
        VAR_CAMERA_MAX_DISTANCE,
        "camera_max_distance",
    );
    let neg_one_start = template_number_node(&mut snarl, egui::pos2(1440.0, 350.0), -1.0);
    let neg_camera_z = snarl.insert_node(
        egui::pos2(1600.0, 350.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, camera_start_y, 0, camera_start_offset, 1);
    template_connect_data(&mut snarl, camera_start_z, 0, neg_camera_z, 0);
    template_connect_data(&mut snarl, neg_one_start, 0, neg_camera_z, 1);
    template_connect_data(&mut snarl, neg_camera_z, 0, camera_start_offset, 2);

    let start_set_follower = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::EcsSetEntityFollower,
    );
    template_set_api_arg(&mut snarl, start_set_follower, 3, "false");
    template_set_api_arg(&mut snarl, start_set_follower, 4, "false");
    template_set_api_arg(&mut snarl, start_set_follower, 5, "0.0");
    template_set_api_arg(&mut snarl, start_set_follower, 6, "0.0");
    template_connect_exec(&mut snarl, start_cursor, 0, start_set_follower, 0);
    template_connect_data(&mut snarl, resolved_camera_entity, 0, start_set_follower, 0);
    template_connect_data(&mut snarl, self_entity, 0, start_set_follower, 1);
    template_connect_data(&mut snarl, camera_start_offset, 0, start_set_follower, 2);
    start_cursor = start_set_follower;
    bind_x += 320.0;

    let camera_target_offset =
        snarl.insert_node(egui::pos2(1880.0, 260.0), VisualScriptNodeKind::Vec3);
    let camera_up_vec = snarl.insert_node(egui::pos2(2080.0, 260.0), VisualScriptNodeKind::Vec3);
    let one_lit = template_number_node(&mut snarl, egui::pos2(2120.0, 350.0), 1.0);
    template_connect_data(&mut snarl, camera_start_y, 0, camera_target_offset, 1);
    template_connect_data(&mut snarl, one_lit, 0, camera_up_vec, 1);

    let start_set_look_at = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::EcsSetLookAt,
    );
    template_set_api_arg(&mut snarl, start_set_look_at, 3, "true");
    template_set_api_arg(&mut snarl, start_set_look_at, 5, "0.0");
    template_connect_exec(&mut snarl, start_cursor, 0, start_set_look_at, 0);
    template_connect_data(&mut snarl, resolved_camera_entity, 0, start_set_look_at, 0);
    template_connect_data(&mut snarl, self_entity, 0, start_set_look_at, 1);
    template_connect_data(&mut snarl, camera_target_offset, 0, start_set_look_at, 2);
    template_connect_data(&mut snarl, camera_up_vec, 0, start_set_look_at, 4);
    start_cursor = start_set_look_at;
    bind_x += 320.0;

    let start_set_active_camera = template_insert_api_node(
        &mut snarl,
        egui::pos2(bind_x, 90.0),
        VisualApiOperation::EcsSetActiveCamera,
    );
    template_connect_exec(&mut snarl, start_cursor, 0, start_set_active_camera, 0);
    template_connect_data(
        &mut snarl,
        resolved_camera_entity,
        0,
        start_set_active_camera,
        0,
    );
    start_cursor = start_set_active_camera;
    bind_x += 320.0;

    let start_log = snarl.insert_node(
        egui::pos2(bind_x, 90.0),
        VisualScriptNodeKind::Log {
            message: "third-person controller graph initialized".to_string(),
        },
    );
    template_connect_exec(&mut snarl, start_cursor, 0, start_log, 0);

    let set_sprint_true = template_set_variable_node(
        &mut snarl,
        egui::pos2(320.0, 340.0),
        VAR_IS_SPRINTING,
        "is_sprinting",
        "true",
    );
    let set_sprint_false = template_set_variable_node(
        &mut snarl,
        egui::pos2(320.0, 430.0),
        VAR_IS_SPRINTING,
        "is_sprinting",
        "false",
    );
    template_connect_exec(&mut snarl, on_sprint_down, 0, set_sprint_true, 0);
    template_connect_exec(&mut snarl, on_sprint_released, 0, set_sprint_false, 0);
    let true_lit = template_bool_node(&mut snarl, egui::pos2(250.0, 300.0), true);
    let false_lit = template_bool_node(&mut snarl, egui::pos2(250.0, 470.0), false);
    template_connect_data(&mut snarl, true_lit, 0, set_sprint_true, 0);
    template_connect_data(&mut snarl, false_lit, 0, set_sprint_false, 0);

    let crouch_get = template_get_variable_node(
        &mut snarl,
        egui::pos2(300.0, 560.0),
        VAR_IS_CROUCHING,
        "is_crouching",
    );
    let crouch_not = snarl.insert_node(egui::pos2(500.0, 560.0), VisualScriptNodeKind::Not);
    let crouch_set = template_set_variable_node(
        &mut snarl,
        egui::pos2(700.0, 520.0),
        VAR_IS_CROUCHING,
        "is_crouching",
        "false",
    );
    template_connect_exec(&mut snarl, on_crouch_pressed, 0, crouch_set, 0);
    template_connect_data(&mut snarl, crouch_get, 0, crouch_not, 0);
    template_connect_data(&mut snarl, crouch_not, 0, crouch_set, 0);

    let jump_set = template_set_variable_node(
        &mut snarl,
        egui::pos2(320.0, 610.0),
        VAR_JUMP_QUEUED,
        "jump_queued",
        "true",
    );
    template_connect_exec(&mut snarl, on_jump_pressed, 0, jump_set, 0);
    template_connect_data(&mut snarl, true_lit, 0, jump_set, 0);

    let stage1_self_transform = template_insert_api_node(
        &mut snarl,
        egui::pos2(560.0, 910.0),
        VisualApiOperation::EcsGetTransform,
    );
    template_connect_data(&mut snarl, self_entity, 0, stage1_self_transform, 0);
    let stage1_position = snarl.insert_node(
        egui::pos2(760.0, 910.0),
        VisualScriptNodeKind::TransformGetComponent {
            component: VisualTransformComponent::Position,
        },
    );
    template_connect_data(&mut snarl, stage1_self_transform, 0, stage1_position, 0);
    let stage1_pos_x = snarl.insert_node(
        egui::pos2(960.0, 820.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let stage1_pos_y = snarl.insert_node(
        egui::pos2(960.0, 910.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let stage1_pos_z = snarl.insert_node(
        egui::pos2(960.0, 1000.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, stage1_position, 0, stage1_pos_x, 0);
    template_connect_data(&mut snarl, stage1_position, 0, stage1_pos_y, 0);
    template_connect_data(&mut snarl, stage1_position, 0, stage1_pos_z, 0);

    let left_offset_var = template_get_variable_node(
        &mut snarl,
        egui::pos2(1160.0, 820.0),
        VAR_LEFT_FOOT_OFFSET,
        "left_foot_offset",
    );
    let right_offset_var = template_get_variable_node(
        &mut snarl,
        egui::pos2(1160.0, 1080.0),
        VAR_RIGHT_FOOT_OFFSET,
        "right_foot_offset",
    );
    let left_offset_x = snarl.insert_node(
        egui::pos2(1360.0, 760.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let left_offset_y = snarl.insert_node(
        egui::pos2(1360.0, 850.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let left_offset_z = snarl.insert_node(
        egui::pos2(1360.0, 940.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, left_offset_var, 0, left_offset_x, 0);
    template_connect_data(&mut snarl, left_offset_var, 0, left_offset_y, 0);
    template_connect_data(&mut snarl, left_offset_var, 0, left_offset_z, 0);
    let right_offset_x = snarl.insert_node(
        egui::pos2(1360.0, 1040.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let right_offset_y = snarl.insert_node(
        egui::pos2(1360.0, 1130.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let right_offset_z = snarl.insert_node(
        egui::pos2(1360.0, 1220.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, right_offset_var, 0, right_offset_x, 0);
    template_connect_data(&mut snarl, right_offset_var, 0, right_offset_y, 0);
    template_connect_data(&mut snarl, right_offset_var, 0, right_offset_z, 0);

    let left_origin_x = snarl.insert_node(
        egui::pos2(1560.0, 760.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    let left_origin_y = snarl.insert_node(
        egui::pos2(1560.0, 850.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    let left_origin_z = snarl.insert_node(
        egui::pos2(1560.0, 940.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    template_connect_data(&mut snarl, stage1_pos_x, 0, left_origin_x, 0);
    template_connect_data(&mut snarl, left_offset_x, 0, left_origin_x, 1);
    template_connect_data(&mut snarl, stage1_pos_y, 0, left_origin_y, 0);
    template_connect_data(&mut snarl, left_offset_y, 0, left_origin_y, 1);
    template_connect_data(&mut snarl, stage1_pos_z, 0, left_origin_z, 0);
    template_connect_data(&mut snarl, left_offset_z, 0, left_origin_z, 1);
    let left_origin = snarl.insert_node(egui::pos2(1760.0, 850.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, left_origin_x, 0, left_origin, 0);
    template_connect_data(&mut snarl, left_origin_y, 0, left_origin, 1);
    template_connect_data(&mut snarl, left_origin_z, 0, left_origin, 2);

    let right_origin_x = snarl.insert_node(
        egui::pos2(1560.0, 1040.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    let right_origin_y = snarl.insert_node(
        egui::pos2(1560.0, 1130.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    let right_origin_z = snarl.insert_node(
        egui::pos2(1560.0, 1220.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    template_connect_data(&mut snarl, stage1_pos_x, 0, right_origin_x, 0);
    template_connect_data(&mut snarl, right_offset_x, 0, right_origin_x, 1);
    template_connect_data(&mut snarl, stage1_pos_y, 0, right_origin_y, 0);
    template_connect_data(&mut snarl, right_offset_y, 0, right_origin_y, 1);
    template_connect_data(&mut snarl, stage1_pos_z, 0, right_origin_z, 0);
    template_connect_data(&mut snarl, right_offset_z, 0, right_origin_z, 1);
    let right_origin = snarl.insert_node(egui::pos2(1760.0, 1130.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, right_origin_x, 0, right_origin, 0);
    template_connect_data(&mut snarl, right_origin_y, 0, right_origin, 1);
    template_connect_data(&mut snarl, right_origin_z, 0, right_origin, 2);

    let down_vec = snarl.insert_node(egui::pos2(1760.0, 1380.0), VisualScriptNodeKind::Vec3);
    let neg_one = template_number_node(&mut snarl, egui::pos2(1960.0, 1430.0), -1.0);
    template_connect_data(&mut snarl, neg_one, 0, down_vec, 1);

    let left_probe = template_insert_api_node(
        &mut snarl,
        egui::pos2(1960.0, 850.0),
        VisualApiOperation::EcsSphereCastHasHit,
    );
    template_set_api_arg(&mut snarl, left_probe, 1, "0.085");
    template_set_api_arg(&mut snarl, left_probe, 3, "0.95");
    template_connect_data(&mut snarl, left_origin, 0, left_probe, 0);
    template_connect_data(&mut snarl, down_vec, 0, left_probe, 2);
    template_connect_data(&mut snarl, self_entity, 0, left_probe, 5);

    let right_probe = template_insert_api_node(
        &mut snarl,
        egui::pos2(1960.0, 1130.0),
        VisualApiOperation::EcsSphereCastHasHit,
    );
    template_set_api_arg(&mut snarl, right_probe, 1, "0.085");
    template_set_api_arg(&mut snarl, right_probe, 3, "0.95");
    template_connect_data(&mut snarl, right_origin, 0, right_probe, 0);
    template_connect_data(&mut snarl, down_vec, 0, right_probe, 2);
    template_connect_data(&mut snarl, self_entity, 0, right_probe, 5);

    let set_left_grounded = template_set_variable_node(
        &mut snarl,
        egui::pos2(2200.0, 850.0),
        VAR_LEFT_FOOT_GROUNDED,
        "left_foot_grounded",
        "false",
    );
    let set_right_grounded = template_set_variable_node(
        &mut snarl,
        egui::pos2(2400.0, 850.0),
        VAR_RIGHT_FOOT_GROUNDED,
        "right_foot_grounded",
        "false",
    );
    template_connect_exec(&mut snarl, on_update_seq, 0, set_left_grounded, 0);
    template_connect_exec(&mut snarl, set_left_grounded, 0, set_right_grounded, 0);
    template_connect_data(&mut snarl, left_probe, 0, set_left_grounded, 0);
    template_connect_data(&mut snarl, right_probe, 0, set_right_grounded, 0);

    let move_right = template_insert_api_node(
        &mut snarl,
        egui::pos2(560.0, 1560.0),
        VisualApiOperation::InputActionValue,
    );
    template_set_api_arg(&mut snarl, move_right, 0, "move_right");
    let move_left = template_insert_api_node(
        &mut snarl,
        egui::pos2(560.0, 1650.0),
        VisualApiOperation::InputActionValue,
    );
    template_set_api_arg(&mut snarl, move_left, 0, "move_left");
    let move_forward = template_insert_api_node(
        &mut snarl,
        egui::pos2(560.0, 1740.0),
        VisualApiOperation::InputActionValue,
    );
    template_set_api_arg(&mut snarl, move_forward, 0, "move_forward");
    let move_backward = template_insert_api_node(
        &mut snarl,
        egui::pos2(560.0, 1830.0),
        VisualApiOperation::InputActionValue,
    );
    template_set_api_arg(&mut snarl, move_backward, 0, "move_backward");

    let move_axis_x = snarl.insert_node(
        egui::pos2(820.0, 1605.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Subtract,
        },
    );
    let move_axis_z = snarl.insert_node(
        egui::pos2(820.0, 1785.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Subtract,
        },
    );
    template_connect_data(&mut snarl, move_right, 0, move_axis_x, 0);
    template_connect_data(&mut snarl, move_left, 0, move_axis_x, 1);
    template_connect_data(&mut snarl, move_forward, 0, move_axis_z, 0);
    template_connect_data(&mut snarl, move_backward, 0, move_axis_z, 1);

    let move_local = snarl.insert_node(egui::pos2(1060.0, 1695.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, move_axis_x, 0, move_local, 0);
    template_connect_data(&mut snarl, move_axis_z, 0, move_local, 2);

    let move_magnitude_raw = snarl.insert_node(
        egui::pos2(1260.0, 1695.0),
        VisualScriptNodeKind::MathVector {
            op: VisualVectorMathOp::Length,
        },
    );
    template_connect_data(&mut snarl, move_local, 0, move_magnitude_raw, 0);
    let move_magnitude = snarl.insert_node(
        egui::pos2(1460.0, 1695.0),
        VisualScriptNodeKind::MathProcedural {
            op: VisualProceduralMathOp::Saturate,
        },
    );
    template_connect_data(&mut snarl, move_magnitude_raw, 0, move_magnitude, 0);

    let move_yaw_rad = snarl.insert_node(
        egui::pos2(1060.0, 1885.0),
        VisualScriptNodeKind::MathUtility {
            op: VisualUtilityMathOp::Radians,
        },
    );
    let move_yaw_sin = snarl.insert_node(
        egui::pos2(1260.0, 1840.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Sin,
        },
    );
    let move_yaw_cos = snarl.insert_node(
        egui::pos2(1260.0, 1930.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Cos,
        },
    );
    template_connect_data(&mut snarl, move_yaw_rad, 0, move_yaw_sin, 0);
    template_connect_data(&mut snarl, move_yaw_rad, 0, move_yaw_cos, 0);
    let move_right_x = snarl.insert_node(
        egui::pos2(1460.0, 1840.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let move_right_z = snarl.insert_node(
        egui::pos2(1460.0, 1930.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, move_yaw_cos, 0, move_right_x, 0);
    template_connect_data(&mut snarl, neg_one, 0, move_right_x, 1);
    template_connect_data(&mut snarl, move_yaw_sin, 0, move_right_z, 0);
    template_connect_data(&mut snarl, one_lit, 0, move_right_z, 1);
    let move_right_vec = snarl.insert_node(egui::pos2(1660.0, 1930.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, move_right_x, 0, move_right_vec, 0);
    template_connect_data(&mut snarl, move_right_z, 0, move_right_vec, 2);
    let move_forward_vec =
        snarl.insert_node(egui::pos2(1660.0, 2060.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, move_yaw_sin, 0, move_forward_vec, 0);
    template_connect_data(&mut snarl, move_yaw_cos, 0, move_forward_vec, 2);
    let right_x = snarl.insert_node(
        egui::pos2(1260.0, 1880.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let right_y = snarl.insert_node(
        egui::pos2(1260.0, 1970.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let right_z = snarl.insert_node(
        egui::pos2(1260.0, 2060.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    let forward_x = snarl.insert_node(
        egui::pos2(1260.0, 2150.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let forward_y = snarl.insert_node(
        egui::pos2(1260.0, 2240.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let forward_z = snarl.insert_node(
        egui::pos2(1260.0, 2330.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, move_right_vec, 0, right_x, 0);
    template_connect_data(&mut snarl, move_right_vec, 0, right_y, 0);
    template_connect_data(&mut snarl, move_right_vec, 0, right_z, 0);
    template_connect_data(&mut snarl, move_forward_vec, 0, forward_x, 0);
    template_connect_data(&mut snarl, move_forward_vec, 0, forward_y, 0);
    template_connect_data(&mut snarl, move_forward_vec, 0, forward_z, 0);

    let right_x_mul = snarl.insert_node(
        egui::pos2(1460.0, 1880.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let right_y_mul = snarl.insert_node(
        egui::pos2(1460.0, 1970.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let right_z_mul = snarl.insert_node(
        egui::pos2(1460.0, 2060.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let forward_x_mul = snarl.insert_node(
        egui::pos2(1460.0, 2150.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let forward_y_mul = snarl.insert_node(
        egui::pos2(1460.0, 2240.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let forward_z_mul = snarl.insert_node(
        egui::pos2(1460.0, 2330.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, right_x, 0, right_x_mul, 0);
    template_connect_data(&mut snarl, right_y, 0, right_y_mul, 0);
    template_connect_data(&mut snarl, right_z, 0, right_z_mul, 0);
    template_connect_data(&mut snarl, move_axis_x, 0, right_x_mul, 1);
    template_connect_data(&mut snarl, move_axis_x, 0, right_y_mul, 1);
    template_connect_data(&mut snarl, move_axis_x, 0, right_z_mul, 1);
    template_connect_data(&mut snarl, forward_x, 0, forward_x_mul, 0);
    template_connect_data(&mut snarl, forward_y, 0, forward_y_mul, 0);
    template_connect_data(&mut snarl, forward_z, 0, forward_z_mul, 0);
    template_connect_data(&mut snarl, move_axis_z, 0, forward_x_mul, 1);
    template_connect_data(&mut snarl, move_axis_z, 0, forward_y_mul, 1);
    template_connect_data(&mut snarl, move_axis_z, 0, forward_z_mul, 1);
    let zero_planar_y = template_number_node(&mut snarl, egui::pos2(1380.0, 2015.0), 0.0);
    template_connect_data(&mut snarl, zero_planar_y, 0, right_y_mul, 0);
    template_connect_data(&mut snarl, zero_planar_y, 0, forward_y_mul, 0);

    let world_x_add = snarl.insert_node(
        egui::pos2(1660.0, 2015.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    let world_y_add = snarl.insert_node(
        egui::pos2(1660.0, 2105.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    let world_z_add = snarl.insert_node(
        egui::pos2(1660.0, 2195.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    template_connect_data(&mut snarl, right_x_mul, 0, world_x_add, 0);
    template_connect_data(&mut snarl, forward_x_mul, 0, world_x_add, 1);
    template_connect_data(&mut snarl, right_y_mul, 0, world_y_add, 0);
    template_connect_data(&mut snarl, forward_y_mul, 0, world_y_add, 1);
    template_connect_data(&mut snarl, right_z_mul, 0, world_z_add, 0);
    template_connect_data(&mut snarl, forward_z_mul, 0, world_z_add, 1);
    let world_raw = snarl.insert_node(egui::pos2(1860.0, 2105.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, world_x_add, 0, world_raw, 0);
    template_connect_data(&mut snarl, world_y_add, 0, world_raw, 1);
    template_connect_data(&mut snarl, world_z_add, 0, world_raw, 2);
    let world_dir_normalized = snarl.insert_node(
        egui::pos2(2060.0, 2105.0),
        VisualScriptNodeKind::MathVector {
            op: VisualVectorMathOp::Normalize,
        },
    );
    template_connect_data(&mut snarl, world_raw, 0, world_dir_normalized, 0);
    let move_input_epsilon = template_number_node(&mut snarl, egui::pos2(2260.0, 2105.0), 0.0001);
    let has_move_input = snarl.insert_node(
        egui::pos2(2460.0, 2105.0),
        VisualScriptNodeKind::Compare {
            op: VisualCompareOp::Greater,
        },
    );
    template_connect_data(&mut snarl, move_magnitude_raw, 0, has_move_input, 0);
    template_connect_data(&mut snarl, move_input_epsilon, 0, has_move_input, 1);
    let zero_world_dir = snarl.insert_node(egui::pos2(2660.0, 2035.0), VisualScriptNodeKind::Vec3);
    let world_dir = snarl.insert_node(
        egui::pos2(2660.0, 2140.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::Vec3,
        },
    );
    template_connect_data(&mut snarl, has_move_input, 0, world_dir, 0);
    template_connect_data(&mut snarl, world_dir_normalized, 0, world_dir, 1);
    template_connect_data(&mut snarl, zero_world_dir, 0, world_dir, 2);

    let speed_walk = template_get_variable_node(
        &mut snarl,
        egui::pos2(2260.0, 1870.0),
        VAR_SPEED_WALK,
        "move_speed_walk",
    );
    let speed_run = template_get_variable_node(
        &mut snarl,
        egui::pos2(2260.0, 1960.0),
        VAR_SPEED_RUN,
        "move_speed_run",
    );
    let speed_sprint = template_get_variable_node(
        &mut snarl,
        egui::pos2(2260.0, 2050.0),
        VAR_SPEED_SPRINT,
        "move_speed_sprint",
    );
    let speed_crouch = template_get_variable_node(
        &mut snarl,
        egui::pos2(2260.0, 2140.0),
        VAR_SPEED_CROUCH,
        "move_speed_crouch",
    );
    let is_sprinting = template_get_variable_node(
        &mut snarl,
        egui::pos2(2260.0, 2230.0),
        VAR_IS_SPRINTING,
        "is_sprinting",
    );
    let is_crouching = template_get_variable_node(
        &mut snarl,
        egui::pos2(2260.0, 2320.0),
        VAR_IS_CROUCHING,
        "is_crouching",
    );
    let not_crouching = snarl.insert_node(egui::pos2(2460.0, 2320.0), VisualScriptNodeKind::Not);
    template_connect_data(&mut snarl, is_crouching, 0, not_crouching, 0);
    let can_sprint = snarl.insert_node(
        egui::pos2(2660.0, 2230.0),
        VisualScriptNodeKind::LogicalBinary {
            op: VisualLogicalOp::And,
        },
    );
    template_connect_data(&mut snarl, is_sprinting, 0, can_sprint, 0);
    template_connect_data(&mut snarl, not_crouching, 0, can_sprint, 1);
    let speed_base = snarl.insert_node(
        egui::pos2(2460.0, 1960.0),
        VisualScriptNodeKind::MathInterpolation {
            op: VisualInterpolationOp::Lerp,
        },
    );
    template_connect_data(&mut snarl, speed_walk, 0, speed_base, 0);
    template_connect_data(&mut snarl, speed_run, 0, speed_base, 1);
    template_connect_data(&mut snarl, move_magnitude, 0, speed_base, 2);
    let speed_sprint_or_base = snarl.insert_node(
        egui::pos2(2860.0, 2050.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::Number,
        },
    );
    template_connect_data(&mut snarl, can_sprint, 0, speed_sprint_or_base, 0);
    template_connect_data(&mut snarl, speed_sprint, 0, speed_sprint_or_base, 1);
    template_connect_data(&mut snarl, speed_base, 0, speed_sprint_or_base, 2);
    let speed_final = snarl.insert_node(
        egui::pos2(3060.0, 2140.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::Number,
        },
    );
    template_connect_data(&mut snarl, is_crouching, 0, speed_final, 0);
    template_connect_data(&mut snarl, speed_crouch, 0, speed_final, 1);
    template_connect_data(&mut snarl, speed_sprint_or_base, 0, speed_final, 2);

    let dt = snarl.insert_node(egui::pos2(3260.0, 2140.0), VisualScriptNodeKind::DeltaTime);
    let step_distance = snarl.insert_node(
        egui::pos2(3460.0, 2140.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, speed_final, 0, step_distance, 0);
    template_connect_data(&mut snarl, dt, 0, step_distance, 1);

    let world_dir_x = snarl.insert_node(
        egui::pos2(2260.0, 2440.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let world_dir_y = snarl.insert_node(
        egui::pos2(2260.0, 2530.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let world_dir_z = snarl.insert_node(
        egui::pos2(2260.0, 2620.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, world_dir, 0, world_dir_x, 0);
    template_connect_data(&mut snarl, world_dir, 0, world_dir_y, 0);
    template_connect_data(&mut snarl, world_dir, 0, world_dir_z, 0);

    let move_x_scaled = snarl.insert_node(
        egui::pos2(2460.0, 2440.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let move_y_scaled = snarl.insert_node(
        egui::pos2(2460.0, 2530.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let move_z_scaled = snarl.insert_node(
        egui::pos2(2460.0, 2620.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, world_dir_x, 0, move_x_scaled, 0);
    template_connect_data(&mut snarl, world_dir_y, 0, move_y_scaled, 0);
    template_connect_data(&mut snarl, world_dir_z, 0, move_z_scaled, 0);
    template_connect_data(&mut snarl, step_distance, 0, move_x_scaled, 1);
    template_connect_data(&mut snarl, step_distance, 0, move_y_scaled, 1);
    template_connect_data(&mut snarl, step_distance, 0, move_z_scaled, 1);

    let desired_move = snarl.insert_node(egui::pos2(2660.0, 2530.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, move_x_scaled, 0, desired_move, 0);
    template_connect_data(&mut snarl, move_y_scaled, 0, desired_move, 1);
    template_connect_data(&mut snarl, move_z_scaled, 0, desired_move, 2);

    let is_grounded = template_insert_api_node(
        &mut snarl,
        egui::pos2(2860.0, 2440.0),
        VisualApiOperation::EcsGetCharacterControllerGrounded,
    );
    template_connect_data(&mut snarl, self_entity, 0, is_grounded, 0);
    let jump_queued = template_get_variable_node(
        &mut snarl,
        egui::pos2(2860.0, 2530.0),
        VAR_JUMP_QUEUED,
        "jump_queued",
    );
    let do_jump = snarl.insert_node(
        egui::pos2(3060.0, 2485.0),
        VisualScriptNodeKind::LogicalBinary {
            op: VisualLogicalOp::And,
        },
    );
    template_connect_data(&mut snarl, is_grounded, 0, do_jump, 0);
    template_connect_data(&mut snarl, jump_queued, 0, do_jump, 1);
    let jump_speed = template_get_variable_node(
        &mut snarl,
        egui::pos2(3260.0, 2350.0),
        VAR_JUMP_SPEED,
        "jump_speed",
    );
    let gravity_acceleration = template_get_variable_node(
        &mut snarl,
        egui::pos2(3260.0, 2440.0),
        VAR_GRAVITY_ACCELERATION,
        "gravity_acceleration",
    );
    let vertical_velocity = template_get_variable_node(
        &mut snarl,
        egui::pos2(3260.0, 2530.0),
        VAR_VERTICAL_VELOCITY,
        "vertical_velocity",
    );
    let zero_number = template_number_node(&mut snarl, egui::pos2(3260.0, 2620.0), 0.0);
    let gravity_step = snarl.insert_node(
        egui::pos2(3460.0, 2440.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, gravity_acceleration, 0, gravity_step, 0);
    template_connect_data(&mut snarl, dt, 0, gravity_step, 1);
    let vertical_after_gravity = snarl.insert_node(
        egui::pos2(3660.0, 2485.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    template_connect_data(&mut snarl, vertical_velocity, 0, vertical_after_gravity, 0);
    template_connect_data(&mut snarl, gravity_step, 0, vertical_after_gravity, 1);
    let vertical_when_grounded = snarl.insert_node(
        egui::pos2(3860.0, 2485.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::Number,
        },
    );
    template_connect_data(&mut snarl, is_grounded, 0, vertical_when_grounded, 0);
    template_connect_data(&mut snarl, zero_number, 0, vertical_when_grounded, 1);
    template_connect_data(
        &mut snarl,
        vertical_after_gravity,
        0,
        vertical_when_grounded,
        2,
    );
    let vertical_target = snarl.insert_node(
        egui::pos2(4060.0, 2485.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::Number,
        },
    );
    template_connect_data(&mut snarl, do_jump, 0, vertical_target, 0);
    template_connect_data(&mut snarl, jump_speed, 0, vertical_target, 1);
    template_connect_data(&mut snarl, vertical_when_grounded, 0, vertical_target, 2);
    let vertical_step = snarl.insert_node(
        egui::pos2(4260.0, 2485.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, vertical_target, 0, vertical_step, 0);
    template_connect_data(&mut snarl, dt, 0, vertical_step, 1);
    let desired_with_vertical = snarl.insert_node(
        egui::pos2(4460.0, 2530.0),
        VisualScriptNodeKind::Vec3SetComponent {
            component: VisualVec3Component::Y,
        },
    );
    template_connect_data(&mut snarl, desired_move, 0, desired_with_vertical, 0);
    template_connect_data(&mut snarl, vertical_step, 0, desired_with_vertical, 1);

    let set_move_magnitude = template_set_variable_node(
        &mut snarl,
        egui::pos2(560.0, 1400.0),
        VAR_MOVE_MAGNITUDE,
        "move_magnitude",
        "0",
    );
    let set_move_dir = template_set_variable_node(
        &mut snarl,
        egui::pos2(780.0, 1400.0),
        VAR_MOVE_WORLD_DIRECTION,
        "move_world_direction",
        "{\"x\":0,\"y\":0,\"z\":0}",
    );
    let set_vertical_velocity = template_set_variable_node(
        &mut snarl,
        egui::pos2(1000.0, 1400.0),
        VAR_VERTICAL_VELOCITY,
        "vertical_velocity",
        "0",
    );
    let set_character_move = template_insert_api_node(
        &mut snarl,
        egui::pos2(1220.0, 1400.0),
        VisualApiOperation::EcsSetCharacterControllerDesiredTranslation,
    );
    let clear_jump_queued = template_set_variable_node(
        &mut snarl,
        egui::pos2(1440.0, 1400.0),
        VAR_JUMP_QUEUED,
        "jump_queued",
        "false",
    );
    let set_anim_move_input = template_insert_api_node(
        &mut snarl,
        egui::pos2(1660.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamFloat,
    );
    let set_anim_move_speed = template_insert_api_node(
        &mut snarl,
        egui::pos2(1880.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamFloat,
    );
    let set_anim_moving = template_insert_api_node(
        &mut snarl,
        egui::pos2(2100.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamBool,
    );
    let set_anim_grounded = template_insert_api_node(
        &mut snarl,
        egui::pos2(2320.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamBool,
    );
    let set_anim_jump = template_insert_api_node(
        &mut snarl,
        egui::pos2(2430.0, 1320.0),
        VisualApiOperation::EcsSetAnimatorParamBool,
    );
    let set_anim_sprint = template_insert_api_node(
        &mut snarl,
        egui::pos2(2540.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamBool,
    );
    let set_anim_crouch = template_insert_api_node(
        &mut snarl,
        egui::pos2(2760.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamBool,
    );
    let set_anim_left_foot = template_insert_api_node(
        &mut snarl,
        egui::pos2(2980.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamBool,
    );
    let set_anim_right_foot = template_insert_api_node(
        &mut snarl,
        egui::pos2(3200.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamBool,
    );
    let set_anim_slope = template_insert_api_node(
        &mut snarl,
        egui::pos2(3420.0, 1400.0),
        VisualApiOperation::EcsSetAnimatorParamFloat,
    );
    let turn_smooth_speed = template_get_variable_node(
        &mut snarl,
        egui::pos2(620.0, 1220.0),
        VAR_TURN_SMOOTH_SPEED,
        "turn_smooth_speed",
    );
    let facing_yaw = snarl.insert_node(
        egui::pos2(860.0, 1220.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Atan2,
        },
    );
    template_connect_data(&mut snarl, world_dir_x, 0, facing_yaw, 0);
    template_connect_data(&mut snarl, world_dir_z, 0, facing_yaw, 1);
    let base_yaw_offset = template_get_variable_node(
        &mut snarl,
        egui::pos2(1060.0, 1140.0),
        VAR_BASE_YAW_OFFSET,
        "base_yaw_offset",
    );
    let facing_yaw_with_offset = snarl.insert_node(
        egui::pos2(1060.0, 1220.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    template_connect_data(&mut snarl, facing_yaw, 0, facing_yaw_with_offset, 0);
    template_connect_data(&mut snarl, base_yaw_offset, 0, facing_yaw_with_offset, 1);
    let half_number = template_number_node(&mut snarl, egui::pos2(1260.0, 1220.0), 0.5);
    let facing_half_yaw = snarl.insert_node(
        egui::pos2(1460.0, 1220.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, facing_yaw_with_offset, 0, facing_half_yaw, 0);
    template_connect_data(&mut snarl, half_number, 0, facing_half_yaw, 1);
    let facing_sin = snarl.insert_node(
        egui::pos2(1660.0, 1180.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Sin,
        },
    );
    let facing_cos = snarl.insert_node(
        egui::pos2(1660.0, 1260.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Cos,
        },
    );
    template_connect_data(&mut snarl, facing_half_yaw, 0, facing_sin, 0);
    template_connect_data(&mut snarl, facing_half_yaw, 0, facing_cos, 0);
    let target_rotation = snarl.insert_node(egui::pos2(1860.0, 1220.0), VisualScriptNodeKind::Quat);
    template_connect_data(&mut snarl, facing_sin, 0, target_rotation, 1);
    template_connect_data(&mut snarl, facing_cos, 0, target_rotation, 3);
    let current_forward = template_insert_api_node(
        &mut snarl,
        egui::pos2(1660.0, 1310.0),
        VisualApiOperation::EcsGetTransformForward,
    );
    template_connect_data(&mut snarl, self_entity, 0, current_forward, 0);
    let current_forward_x = snarl.insert_node(
        egui::pos2(1860.0, 1220.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let current_forward_z = snarl.insert_node(
        egui::pos2(1860.0, 1300.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, current_forward, 0, current_forward_x, 0);
    template_connect_data(&mut snarl, current_forward, 0, current_forward_z, 0);
    let current_yaw = snarl.insert_node(
        egui::pos2(2060.0, 1260.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Atan2,
        },
    );
    template_connect_data(&mut snarl, current_forward_x, 0, current_yaw, 0);
    template_connect_data(&mut snarl, current_forward_z, 0, current_yaw, 1);
    let current_half_yaw = snarl.insert_node(
        egui::pos2(2260.0, 1260.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, current_yaw, 0, current_half_yaw, 0);
    template_connect_data(&mut snarl, half_number, 0, current_half_yaw, 1);
    let current_sin = snarl.insert_node(
        egui::pos2(2460.0, 1220.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Sin,
        },
    );
    let current_cos = snarl.insert_node(
        egui::pos2(2460.0, 1300.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Cos,
        },
    );
    template_connect_data(&mut snarl, current_half_yaw, 0, current_sin, 0);
    template_connect_data(&mut snarl, current_half_yaw, 0, current_cos, 0);
    let current_rotation =
        snarl.insert_node(egui::pos2(2660.0, 1260.0), VisualScriptNodeKind::Quat);
    template_connect_data(&mut snarl, current_sin, 0, current_rotation, 1);
    template_connect_data(&mut snarl, current_cos, 0, current_rotation, 3);
    let turn_lerp_raw = snarl.insert_node(
        egui::pos2(1860.0, 1310.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, dt, 0, turn_lerp_raw, 0);
    template_connect_data(&mut snarl, turn_smooth_speed, 0, turn_lerp_raw, 1);
    let turn_lerp = snarl.insert_node(
        egui::pos2(2060.0, 1310.0),
        VisualScriptNodeKind::MathProcedural {
            op: VisualProceduralMathOp::Saturate,
        },
    );
    template_connect_data(&mut snarl, turn_lerp_raw, 0, turn_lerp, 0);
    let smoothed_rotation = snarl.insert_node(
        egui::pos2(2260.0, 1310.0),
        VisualScriptNodeKind::MathInterpolation {
            op: VisualInterpolationOp::QuatSlerp,
        },
    );
    template_connect_data(&mut snarl, current_rotation, 0, smoothed_rotation, 0);
    template_connect_data(&mut snarl, target_rotation, 0, smoothed_rotation, 1);
    template_connect_data(&mut snarl, turn_lerp, 0, smoothed_rotation, 2);
    let rotated_transform = snarl.insert_node(
        egui::pos2(2460.0, 1310.0),
        VisualScriptNodeKind::TransformSetComponent {
            component: VisualTransformComponent::Rotation,
        },
    );
    template_connect_data(&mut snarl, stage1_self_transform, 0, rotated_transform, 0);
    template_connect_data(&mut snarl, smoothed_rotation, 0, rotated_transform, 1);
    let rotate_branch = snarl.insert_node(
        egui::pos2(1000.0, 1320.0),
        VisualScriptNodeKind::Branch {
            condition: "true".to_string(),
        },
    );
    let set_character_rotation = template_insert_api_node(
        &mut snarl,
        egui::pos2(1180.0, 1220.0),
        VisualApiOperation::EcsSetTransform,
    );
    template_connect_data(&mut snarl, self_entity, 0, set_character_rotation, 0);
    template_connect_data(&mut snarl, rotated_transform, 0, set_character_rotation, 1);

    template_set_api_arg(&mut snarl, set_anim_move_input, 1, "move_input");
    template_set_api_arg(&mut snarl, set_anim_move_speed, 1, "move_speed");
    template_set_api_arg(&mut snarl, set_anim_moving, 1, "moving");
    template_set_api_arg(&mut snarl, set_anim_grounded, 1, "grounded");
    template_set_api_arg(&mut snarl, set_anim_jump, 1, "jump");
    template_set_api_arg(&mut snarl, set_anim_sprint, 1, "sprint");
    template_set_api_arg(&mut snarl, set_anim_crouch, 1, "crouch");
    template_set_api_arg(&mut snarl, set_anim_left_foot, 1, "left_foot_grounded");
    template_set_api_arg(&mut snarl, set_anim_right_foot, 1, "right_foot_grounded");
    template_set_api_arg(&mut snarl, set_anim_slope, 1, "slope_angle");

    template_connect_exec(&mut snarl, on_update_seq, 2, set_move_magnitude, 0);
    template_connect_exec(&mut snarl, set_move_magnitude, 0, set_move_dir, 0);
    template_connect_exec(&mut snarl, set_move_dir, 0, rotate_branch, 0);
    template_connect_exec(&mut snarl, rotate_branch, 0, set_character_rotation, 0);
    template_connect_exec(
        &mut snarl,
        set_character_rotation,
        0,
        set_vertical_velocity,
        0,
    );
    template_connect_exec(&mut snarl, rotate_branch, 1, set_vertical_velocity, 0);
    template_connect_exec(&mut snarl, set_vertical_velocity, 0, set_character_move, 0);
    template_connect_exec(&mut snarl, set_character_move, 0, clear_jump_queued, 0);
    template_connect_exec(&mut snarl, clear_jump_queued, 0, set_anim_move_input, 0);
    template_connect_exec(&mut snarl, set_anim_move_input, 0, set_anim_move_speed, 0);
    template_connect_exec(&mut snarl, set_anim_move_speed, 0, set_anim_moving, 0);
    template_connect_exec(&mut snarl, set_anim_moving, 0, set_anim_grounded, 0);
    template_connect_exec(&mut snarl, set_anim_grounded, 0, set_anim_jump, 0);
    template_connect_exec(&mut snarl, set_anim_jump, 0, set_anim_sprint, 0);
    template_connect_exec(&mut snarl, set_anim_sprint, 0, set_anim_crouch, 0);
    template_connect_exec(&mut snarl, set_anim_crouch, 0, set_anim_left_foot, 0);
    template_connect_exec(&mut snarl, set_anim_left_foot, 0, set_anim_right_foot, 0);
    template_connect_exec(&mut snarl, set_anim_right_foot, 0, set_anim_slope, 0);

    template_connect_data(&mut snarl, move_magnitude, 0, set_move_magnitude, 0);
    template_connect_data(&mut snarl, world_dir, 0, set_move_dir, 0);
    template_connect_data(&mut snarl, vertical_target, 0, set_vertical_velocity, 0);
    template_connect_data(&mut snarl, self_entity, 0, set_character_move, 0);
    template_connect_data(&mut snarl, desired_with_vertical, 0, set_character_move, 1);
    template_connect_data(&mut snarl, false_lit, 0, clear_jump_queued, 0);

    let moving_threshold = template_number_node(&mut snarl, egui::pos2(1680.0, 1310.0), 0.1);
    let is_moving = snarl.insert_node(
        egui::pos2(1820.0, 1310.0),
        VisualScriptNodeKind::Compare {
            op: VisualCompareOp::Greater,
        },
    );
    template_connect_data(&mut snarl, move_magnitude, 0, is_moving, 0);
    template_connect_data(&mut snarl, moving_threshold, 0, is_moving, 1);
    template_connect_data(&mut snarl, is_moving, 0, rotate_branch, 0);

    let left_foot_grounded = template_get_variable_node(
        &mut snarl,
        egui::pos2(2620.0, 1310.0),
        VAR_LEFT_FOOT_GROUNDED,
        "left_foot_grounded",
    );
    let right_foot_grounded = template_get_variable_node(
        &mut snarl,
        egui::pos2(2840.0, 1310.0),
        VAR_RIGHT_FOOT_GROUNDED,
        "right_foot_grounded",
    );
    let slope_angle = template_insert_api_node(
        &mut snarl,
        egui::pos2(3060.0, 1310.0),
        VisualApiOperation::EcsGetCharacterControllerSlopeAngle,
    );
    template_connect_data(&mut snarl, self_entity, 0, slope_angle, 0);

    for node in [
        set_anim_move_input,
        set_anim_move_speed,
        set_anim_moving,
        set_anim_grounded,
        set_anim_jump,
        set_anim_sprint,
        set_anim_crouch,
        set_anim_left_foot,
        set_anim_right_foot,
        set_anim_slope,
    ] {
        template_connect_data(&mut snarl, self_entity, 0, node, 0);
    }
    template_connect_data(&mut snarl, move_magnitude, 0, set_anim_move_input, 2);
    template_connect_data(&mut snarl, speed_final, 0, set_anim_move_speed, 2);
    template_connect_data(&mut snarl, is_moving, 0, set_anim_moving, 2);
    template_connect_data(&mut snarl, is_grounded, 0, set_anim_grounded, 2);
    template_connect_data(&mut snarl, do_jump, 0, set_anim_jump, 2);
    template_connect_data(&mut snarl, is_sprinting, 0, set_anim_sprint, 2);
    template_connect_data(&mut snarl, is_crouching, 0, set_anim_crouch, 2);
    template_connect_data(&mut snarl, left_foot_grounded, 0, set_anim_left_foot, 2);
    template_connect_data(&mut snarl, right_foot_grounded, 0, set_anim_right_foot, 2);
    template_connect_data(&mut snarl, slope_angle, 0, set_anim_slope, 2);

    let clip_idle = template_get_variable_node(
        &mut snarl,
        egui::pos2(3420.0, 1220.0),
        VAR_CLIP_IDLE,
        "clip_idle",
    );
    let clip_walk = template_get_variable_node(
        &mut snarl,
        egui::pos2(3420.0, 1300.0),
        VAR_CLIP_WALK,
        "clip_walk",
    );
    let clip_run = template_get_variable_node(
        &mut snarl,
        egui::pos2(3420.0, 1380.0),
        VAR_CLIP_RUN,
        "clip_run",
    );
    let clip_sprint = template_get_variable_node(
        &mut snarl,
        egui::pos2(3420.0, 1460.0),
        VAR_CLIP_SPRINT,
        "clip_sprint",
    );
    let clip_jump = template_get_variable_node(
        &mut snarl,
        egui::pos2(3420.0, 1540.0),
        VAR_CLIP_JUMP,
        "clip_jump",
    );
    let clip_crouch = template_get_variable_node(
        &mut snarl,
        egui::pos2(3420.0, 1620.0),
        VAR_CLIP_CROUCH,
        "clip_crouch",
    );
    let run_threshold = template_number_node(&mut snarl, egui::pos2(3620.0, 1300.0), 0.55);
    let should_run = snarl.insert_node(
        egui::pos2(3800.0, 1300.0),
        VisualScriptNodeKind::Compare {
            op: VisualCompareOp::Greater,
        },
    );
    template_connect_data(&mut snarl, move_magnitude, 0, should_run, 0);
    template_connect_data(&mut snarl, run_threshold, 0, should_run, 1);
    let locomotion_clip = snarl.insert_node(
        egui::pos2(4000.0, 1300.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::String,
        },
    );
    template_connect_data(&mut snarl, should_run, 0, locomotion_clip, 0);
    template_connect_data(&mut snarl, clip_run, 0, locomotion_clip, 1);
    template_connect_data(&mut snarl, clip_walk, 0, locomotion_clip, 2);
    let sprint_or_locomotion = snarl.insert_node(
        egui::pos2(4200.0, 1380.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::String,
        },
    );
    template_connect_data(&mut snarl, can_sprint, 0, sprint_or_locomotion, 0);
    template_connect_data(&mut snarl, clip_sprint, 0, sprint_or_locomotion, 1);
    template_connect_data(&mut snarl, locomotion_clip, 0, sprint_or_locomotion, 2);
    let moving_or_idle = snarl.insert_node(
        egui::pos2(4400.0, 1380.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::String,
        },
    );
    template_connect_data(&mut snarl, is_moving, 0, moving_or_idle, 0);
    template_connect_data(&mut snarl, sprint_or_locomotion, 0, moving_or_idle, 1);
    template_connect_data(&mut snarl, clip_idle, 0, moving_or_idle, 2);
    let crouch_or_base = snarl.insert_node(
        egui::pos2(4600.0, 1460.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::String,
        },
    );
    template_connect_data(&mut snarl, is_crouching, 0, crouch_or_base, 0);
    template_connect_data(&mut snarl, clip_crouch, 0, crouch_or_base, 1);
    template_connect_data(&mut snarl, moving_or_idle, 0, crouch_or_base, 2);
    let desired_anim_clip = snarl.insert_node(
        egui::pos2(4800.0, 1540.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::String,
        },
    );
    template_connect_data(&mut snarl, do_jump, 0, desired_anim_clip, 0);
    template_connect_data(&mut snarl, clip_jump, 0, desired_anim_clip, 1);
    template_connect_data(&mut snarl, crouch_or_base, 0, desired_anim_clip, 2);
    let active_anim_clip = template_get_variable_node(
        &mut snarl,
        egui::pos2(5000.0, 1620.0),
        VAR_ACTIVE_ANIM_CLIP,
        "active_anim_clip",
    );
    let clip_changed = snarl.insert_node(
        egui::pos2(5200.0, 1620.0),
        VisualScriptNodeKind::Compare {
            op: VisualCompareOp::NotEquals,
        },
    );
    template_connect_data(&mut snarl, desired_anim_clip, 0, clip_changed, 0);
    template_connect_data(&mut snarl, active_anim_clip, 0, clip_changed, 1);
    let play_clip_branch = snarl.insert_node(
        egui::pos2(5000.0, 1380.0),
        VisualScriptNodeKind::Branch {
            condition: "true".to_string(),
        },
    );
    let play_anim_clip = template_insert_api_node(
        &mut snarl,
        egui::pos2(5000.0, 1460.0),
        VisualApiOperation::EcsPlayAnimClip,
    );
    let set_active_anim_clip = template_set_variable_node(
        &mut snarl,
        egui::pos2(5200.0, 1460.0),
        VAR_ACTIVE_ANIM_CLIP,
        "active_anim_clip",
        "",
    );
    template_set_api_arg(&mut snarl, play_anim_clip, 2, "0");
    template_connect_exec(&mut snarl, set_anim_slope, 0, play_clip_branch, 0);
    template_connect_data(&mut snarl, clip_changed, 0, play_clip_branch, 0);
    template_connect_exec(&mut snarl, play_clip_branch, 0, play_anim_clip, 0);
    template_connect_exec(&mut snarl, play_anim_clip, 0, set_active_anim_clip, 0);
    template_connect_data(&mut snarl, self_entity, 0, play_anim_clip, 0);
    template_connect_data(&mut snarl, desired_anim_clip, 0, play_anim_clip, 1);
    template_connect_data(&mut snarl, desired_anim_clip, 0, set_active_anim_clip, 0);

    let camera_entity_var = template_get_variable_node(
        &mut snarl,
        egui::pos2(520.0, 2840.0),
        VAR_CAMERA_RUNTIME_ENTITY,
        "camera_runtime_entity",
    );
    let look_x_action = template_insert_api_node(
        &mut snarl,
        egui::pos2(520.0, 2930.0),
        VisualApiOperation::InputActionValue,
    );
    let look_y_action = template_insert_api_node(
        &mut snarl,
        egui::pos2(520.0, 3020.0),
        VisualApiOperation::InputActionValue,
    );
    template_set_api_arg(&mut snarl, look_x_action, 0, "look_x");
    template_set_api_arg(&mut snarl, look_y_action, 0, "look_y");
    let cursor_delta = template_insert_api_node(
        &mut snarl,
        egui::pos2(520.0, 3200.0),
        VisualApiOperation::InputCursorDelta,
    );
    let cursor_x = snarl.insert_node(
        egui::pos2(720.0, 3160.0),
        VisualScriptNodeKind::Vec2GetComponent {
            component: VisualVec2Component::X,
        },
    );
    let cursor_y = snarl.insert_node(
        egui::pos2(720.0, 3250.0),
        VisualScriptNodeKind::Vec2GetComponent {
            component: VisualVec2Component::Y,
        },
    );
    template_connect_data(&mut snarl, cursor_delta, 0, cursor_x, 0);
    template_connect_data(&mut snarl, cursor_delta, 0, cursor_y, 0);
    let yaw_var = template_get_variable_node(
        &mut snarl,
        egui::pos2(520.0, 3330.0),
        VAR_CAMERA_ORBIT_YAW,
        "camera_orbit_yaw",
    );
    let pitch_var = template_get_variable_node(
        &mut snarl,
        egui::pos2(520.0, 3420.0),
        VAR_CAMERA_ORBIT_PITCH,
        "camera_orbit_pitch",
    );
    let camera_stick_sensitivity = template_get_variable_node(
        &mut snarl,
        egui::pos2(920.0, 3330.0),
        VAR_CAMERA_STICK_SENSITIVITY,
        "camera_stick_sensitivity",
    );
    let camera_mouse_sensitivity = template_get_variable_node(
        &mut snarl,
        egui::pos2(920.0, 3420.0),
        VAR_CAMERA_MOUSE_SENSITIVITY,
        "camera_mouse_sensitivity",
    );
    let look_x_dt = snarl.insert_node(
        egui::pos2(1120.0, 3330.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let look_y_dt = snarl.insert_node(
        egui::pos2(1120.0, 3420.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, look_x_action, 0, look_x_dt, 0);
    template_connect_data(&mut snarl, dt, 0, look_x_dt, 1);
    template_connect_data(&mut snarl, look_y_action, 0, look_y_dt, 0);
    template_connect_data(&mut snarl, dt, 0, look_y_dt, 1);
    let stick_yaw_delta = snarl.insert_node(
        egui::pos2(1320.0, 3330.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let stick_pitch_delta = snarl.insert_node(
        egui::pos2(1320.0, 3420.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, look_x_dt, 0, stick_yaw_delta, 0);
    template_connect_data(&mut snarl, camera_stick_sensitivity, 0, stick_yaw_delta, 1);
    template_connect_data(&mut snarl, look_y_dt, 0, stick_pitch_delta, 0);
    template_connect_data(
        &mut snarl,
        camera_stick_sensitivity,
        0,
        stick_pitch_delta,
        1,
    );
    let neg_one_mouse = template_number_node(&mut snarl, egui::pos2(1120.0, 3510.0), -1.0);
    let cursor_x_inverted = snarl.insert_node(
        egui::pos2(1220.0, 3510.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, cursor_x, 0, cursor_x_inverted, 0);
    template_connect_data(&mut snarl, neg_one_mouse, 0, cursor_x_inverted, 1);
    let mouse_yaw_delta = snarl.insert_node(
        egui::pos2(1420.0, 3510.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let mouse_pitch_delta = snarl.insert_node(
        egui::pos2(1320.0, 3600.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, cursor_x_inverted, 0, mouse_yaw_delta, 0);
    template_connect_data(&mut snarl, camera_mouse_sensitivity, 0, mouse_yaw_delta, 1);
    template_connect_data(&mut snarl, cursor_y, 0, mouse_pitch_delta, 0);
    template_connect_data(
        &mut snarl,
        camera_mouse_sensitivity,
        0,
        mouse_pitch_delta,
        1,
    );
    let yaw_delta = snarl.insert_node(
        egui::pos2(1520.0, 3330.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    let pitch_delta = snarl.insert_node(
        egui::pos2(1520.0, 3420.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    template_connect_data(&mut snarl, stick_yaw_delta, 0, yaw_delta, 0);
    template_connect_data(&mut snarl, mouse_yaw_delta, 0, yaw_delta, 1);
    template_connect_data(&mut snarl, stick_pitch_delta, 0, pitch_delta, 0);
    template_connect_data(&mut snarl, mouse_pitch_delta, 0, pitch_delta, 1);
    let yaw_updated = snarl.insert_node(
        egui::pos2(1720.0, 3330.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    let pitch_updated = snarl.insert_node(
        egui::pos2(1720.0, 3420.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Subtract,
        },
    );
    template_connect_data(&mut snarl, yaw_var, 0, yaw_updated, 0);
    template_connect_data(&mut snarl, yaw_delta, 0, yaw_updated, 1);
    template_connect_data(&mut snarl, yaw_updated, 0, move_yaw_rad, 0);
    template_connect_data(&mut snarl, pitch_var, 0, pitch_updated, 0);
    template_connect_data(&mut snarl, pitch_delta, 0, pitch_updated, 1);
    let pitch_min = template_get_variable_node(
        &mut snarl,
        egui::pos2(1720.0, 3500.0),
        VAR_CAMERA_PITCH_MIN,
        "camera_pitch_min",
    );
    let pitch_max = template_get_variable_node(
        &mut snarl,
        egui::pos2(1720.0, 3590.0),
        VAR_CAMERA_PITCH_MAX,
        "camera_pitch_max",
    );
    let pitch_clamped = snarl.insert_node(
        egui::pos2(1920.0, 3460.0),
        VisualScriptNodeKind::MathProcedural {
            op: VisualProceduralMathOp::Clamp,
        },
    );
    template_connect_data(&mut snarl, pitch_updated, 0, pitch_clamped, 0);
    template_connect_data(&mut snarl, pitch_min, 0, pitch_clamped, 1);
    template_connect_data(&mut snarl, pitch_max, 0, pitch_clamped, 2);

    let set_yaw = template_set_variable_node(
        &mut snarl,
        egui::pos2(2160.0, 3330.0),
        VAR_CAMERA_ORBIT_YAW,
        "camera_orbit_yaw",
        "0",
    );
    let set_pitch = template_set_variable_node(
        &mut snarl,
        egui::pos2(2380.0, 3330.0),
        VAR_CAMERA_ORBIT_PITCH,
        "camera_orbit_pitch",
        "18",
    );
    template_connect_exec(&mut snarl, on_update_seq, 1, set_yaw, 0);
    template_connect_exec(&mut snarl, set_yaw, 0, set_pitch, 0);
    template_connect_data(&mut snarl, yaw_updated, 0, set_yaw, 0);
    template_connect_data(&mut snarl, pitch_clamped, 0, set_pitch, 0);

    let yaw_rad = snarl.insert_node(
        egui::pos2(2160.0, 3520.0),
        VisualScriptNodeKind::MathUtility {
            op: VisualUtilityMathOp::Radians,
        },
    );
    let pitch_rad = snarl.insert_node(
        egui::pos2(2160.0, 3610.0),
        VisualScriptNodeKind::MathUtility {
            op: VisualUtilityMathOp::Radians,
        },
    );
    template_connect_data(&mut snarl, yaw_updated, 0, yaw_rad, 0);
    template_connect_data(&mut snarl, pitch_clamped, 0, pitch_rad, 0);
    let sin_yaw = snarl.insert_node(
        egui::pos2(2360.0, 3520.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Sin,
        },
    );
    let cos_yaw = snarl.insert_node(
        egui::pos2(2360.0, 3610.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Cos,
        },
    );
    let sin_pitch = snarl.insert_node(
        egui::pos2(2360.0, 3700.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Sin,
        },
    );
    let cos_pitch = snarl.insert_node(
        egui::pos2(2360.0, 3790.0),
        VisualScriptNodeKind::MathTrig {
            op: VisualTrigOp::Cos,
        },
    );
    template_connect_data(&mut snarl, yaw_rad, 0, sin_yaw, 0);
    template_connect_data(&mut snarl, yaw_rad, 0, cos_yaw, 0);
    template_connect_data(&mut snarl, pitch_rad, 0, sin_pitch, 0);
    template_connect_data(&mut snarl, pitch_rad, 0, cos_pitch, 0);
    let dir_x_mul = snarl.insert_node(
        egui::pos2(2560.0, 3520.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let dir_z_mul = snarl.insert_node(
        egui::pos2(2560.0, 3610.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, cos_pitch, 0, dir_x_mul, 0);
    template_connect_data(&mut snarl, sin_yaw, 0, dir_x_mul, 1);
    template_connect_data(&mut snarl, cos_pitch, 0, dir_z_mul, 0);
    template_connect_data(&mut snarl, cos_yaw, 0, dir_z_mul, 1);
    let dir_x = snarl.insert_node(
        egui::pos2(2760.0, 3520.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let dir_z = snarl.insert_node(
        egui::pos2(2760.0, 3610.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, dir_x_mul, 0, dir_x, 0);
    template_connect_data(&mut snarl, neg_one, 0, dir_x, 1);
    template_connect_data(&mut snarl, dir_z_mul, 0, dir_z, 0);
    template_connect_data(&mut snarl, neg_one, 0, dir_z, 1);
    let orbit_dir = snarl.insert_node(egui::pos2(2960.0, 3610.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, dir_x, 0, orbit_dir, 0);
    template_connect_data(&mut snarl, sin_pitch, 0, orbit_dir, 1);
    template_connect_data(&mut snarl, dir_z, 0, orbit_dir, 2);

    let cam_self_transform = template_insert_api_node(
        &mut snarl,
        egui::pos2(2160.0, 3880.0),
        VisualApiOperation::EcsGetTransform,
    );
    template_connect_data(&mut snarl, self_entity, 0, cam_self_transform, 0);
    let cam_position = snarl.insert_node(
        egui::pos2(2360.0, 3880.0),
        VisualScriptNodeKind::TransformGetComponent {
            component: VisualTransformComponent::Position,
        },
    );
    template_connect_data(&mut snarl, cam_self_transform, 0, cam_position, 0);
    let cam_pos_x = snarl.insert_node(
        egui::pos2(2560.0, 3830.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let cam_pos_y = snarl.insert_node(
        egui::pos2(2560.0, 3920.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let cam_pos_z = snarl.insert_node(
        egui::pos2(2560.0, 4010.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, cam_position, 0, cam_pos_x, 0);
    template_connect_data(&mut snarl, cam_position, 0, cam_pos_y, 0);
    template_connect_data(&mut snarl, cam_position, 0, cam_pos_z, 0);
    let cam_pivot_height = template_get_variable_node(
        &mut snarl,
        egui::pos2(2760.0, 3920.0),
        VAR_CAMERA_PIVOT_HEIGHT,
        "camera_pivot_height",
    );
    let cam_origin_y = snarl.insert_node(
        egui::pos2(2960.0, 3920.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    template_connect_data(&mut snarl, cam_pos_y, 0, cam_origin_y, 0);
    template_connect_data(&mut snarl, cam_pivot_height, 0, cam_origin_y, 1);
    let camera_origin = snarl.insert_node(egui::pos2(3160.0, 3920.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, cam_pos_x, 0, camera_origin, 0);
    template_connect_data(&mut snarl, cam_origin_y, 0, camera_origin, 1);
    template_connect_data(&mut snarl, cam_pos_z, 0, camera_origin, 2);

    let camera_max_distance = template_get_variable_node(
        &mut snarl,
        egui::pos2(3160.0, 4040.0),
        VAR_CAMERA_MAX_DISTANCE,
        "camera_max_distance",
    );
    let camera_min_distance = template_get_variable_node(
        &mut snarl,
        egui::pos2(3160.0, 4130.0),
        VAR_CAMERA_MIN_DISTANCE,
        "camera_min_distance",
    );
    let camera_cast_hit = template_insert_api_node(
        &mut snarl,
        egui::pos2(3360.0, 3920.0),
        VisualApiOperation::EcsSphereCastHasHit,
    );
    template_set_api_arg(&mut snarl, camera_cast_hit, 1, "0.2");
    template_connect_data(&mut snarl, camera_origin, 0, camera_cast_hit, 0);
    template_connect_data(&mut snarl, orbit_dir, 0, camera_cast_hit, 2);
    template_connect_data(&mut snarl, camera_max_distance, 0, camera_cast_hit, 3);
    template_connect_data(&mut snarl, self_entity, 0, camera_cast_hit, 5);
    let camera_cast_toi = template_insert_api_node(
        &mut snarl,
        egui::pos2(3360.0, 4050.0),
        VisualApiOperation::EcsSphereCastToi,
    );
    template_set_api_arg(&mut snarl, camera_cast_toi, 1, "0.2");
    template_connect_data(&mut snarl, camera_origin, 0, camera_cast_toi, 0);
    template_connect_data(&mut snarl, orbit_dir, 0, camera_cast_toi, 2);
    template_connect_data(&mut snarl, camera_max_distance, 0, camera_cast_toi, 3);
    template_connect_data(&mut snarl, self_entity, 0, camera_cast_toi, 5);
    let toi_padding = template_number_node(&mut snarl, egui::pos2(3560.0, 4130.0), 0.24);
    let toi_adjusted = snarl.insert_node(
        egui::pos2(3760.0, 4050.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Subtract,
        },
    );
    template_connect_data(&mut snarl, camera_cast_toi, 0, toi_adjusted, 0);
    template_connect_data(&mut snarl, toi_padding, 0, toi_adjusted, 1);
    let blocked_min_distance = template_number_node(&mut snarl, egui::pos2(3960.0, 3990.0), 0.05);
    let blocked_distance = snarl.insert_node(
        egui::pos2(3960.0, 4050.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Max,
        },
    );
    template_connect_data(&mut snarl, toi_adjusted, 0, blocked_distance, 0);
    template_connect_data(&mut snarl, blocked_min_distance, 0, blocked_distance, 1);
    let default_camera_distance = snarl.insert_node(
        egui::pos2(3960.0, 4140.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Max,
        },
    );
    template_connect_data(
        &mut snarl,
        camera_max_distance,
        0,
        default_camera_distance,
        0,
    );
    template_connect_data(
        &mut snarl,
        camera_min_distance,
        0,
        default_camera_distance,
        1,
    );
    let camera_distance = snarl.insert_node(
        egui::pos2(4160.0, 4095.0),
        VisualScriptNodeKind::Select {
            value_type: VisualValueType::Number,
        },
    );
    template_connect_data(&mut snarl, camera_cast_hit, 0, camera_distance, 0);
    template_connect_data(&mut snarl, blocked_distance, 0, camera_distance, 1);
    template_connect_data(&mut snarl, default_camera_distance, 0, camera_distance, 2);

    let orbit_dir_x = snarl.insert_node(
        egui::pos2(3360.0, 4270.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let orbit_dir_y = snarl.insert_node(
        egui::pos2(3360.0, 4360.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let orbit_dir_z = snarl.insert_node(
        egui::pos2(3360.0, 4450.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, orbit_dir, 0, orbit_dir_x, 0);
    template_connect_data(&mut snarl, orbit_dir, 0, orbit_dir_y, 0);
    template_connect_data(&mut snarl, orbit_dir, 0, orbit_dir_z, 0);
    let orbit_offset_x = snarl.insert_node(
        egui::pos2(3560.0, 4270.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let orbit_offset_y = snarl.insert_node(
        egui::pos2(3560.0, 4360.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let orbit_offset_z = snarl.insert_node(
        egui::pos2(3560.0, 4450.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, orbit_dir_x, 0, orbit_offset_x, 0);
    template_connect_data(&mut snarl, orbit_dir_y, 0, orbit_offset_y, 0);
    template_connect_data(&mut snarl, orbit_dir_z, 0, orbit_offset_z, 0);
    template_connect_data(&mut snarl, camera_distance, 0, orbit_offset_x, 1);
    template_connect_data(&mut snarl, camera_distance, 0, orbit_offset_y, 1);
    template_connect_data(&mut snarl, camera_distance, 0, orbit_offset_z, 1);
    let orbit_offset_y_with_pivot = snarl.insert_node(
        egui::pos2(3760.0, 4360.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Add,
        },
    );
    template_connect_data(&mut snarl, orbit_offset_y, 0, orbit_offset_y_with_pivot, 0);
    template_connect_data(
        &mut snarl,
        cam_pivot_height,
        0,
        orbit_offset_y_with_pivot,
        1,
    );
    let orbit_offset = snarl.insert_node(egui::pos2(3760.0, 4360.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, orbit_offset_x, 0, orbit_offset, 0);
    template_connect_data(&mut snarl, orbit_offset_y_with_pivot, 0, orbit_offset, 1);
    template_connect_data(&mut snarl, orbit_offset_z, 0, orbit_offset, 2);

    let camera_target_offset_runtime =
        snarl.insert_node(egui::pos2(3960.0, 4360.0), VisualScriptNodeKind::Vec3);
    template_connect_data(
        &mut snarl,
        cam_pivot_height,
        0,
        camera_target_offset_runtime,
        1,
    );

    let camera_set_follower = template_insert_api_node(
        &mut snarl,
        egui::pos2(2600.0, 3330.0),
        VisualApiOperation::EcsSetEntityFollower,
    );
    let camera_set_look_at = template_insert_api_node(
        &mut snarl,
        egui::pos2(2820.0, 3330.0),
        VisualApiOperation::EcsSetLookAt,
    );
    template_set_api_arg(&mut snarl, camera_set_follower, 3, "false");
    template_set_api_arg(&mut snarl, camera_set_follower, 4, "false");
    template_set_api_arg(&mut snarl, camera_set_follower, 5, "0.0");
    template_set_api_arg(&mut snarl, camera_set_follower, 6, "0.0");
    template_set_api_arg(&mut snarl, camera_set_look_at, 3, "true");
    template_set_api_arg(&mut snarl, camera_set_look_at, 5, "0.0");
    template_connect_exec(&mut snarl, play_clip_branch, 1, camera_set_follower, 0);
    template_connect_exec(&mut snarl, set_active_anim_clip, 0, camera_set_follower, 0);
    template_connect_exec(&mut snarl, camera_set_follower, 0, camera_set_look_at, 0);
    template_connect_data(&mut snarl, camera_entity_var, 0, camera_set_follower, 0);
    template_connect_data(&mut snarl, self_entity, 0, camera_set_follower, 1);
    template_connect_data(&mut snarl, orbit_offset, 0, camera_set_follower, 2);
    template_connect_data(&mut snarl, camera_entity_var, 0, camera_set_look_at, 0);
    template_connect_data(&mut snarl, self_entity, 0, camera_set_look_at, 1);
    template_connect_data(
        &mut snarl,
        camera_target_offset_runtime,
        0,
        camera_set_look_at,
        2,
    );
    template_connect_data(&mut snarl, camera_up_vec, 0, camera_set_look_at, 4);

    let hit_collision_count = template_insert_api_node(
        &mut snarl,
        egui::pos2(560.0, 4660.0),
        VisualApiOperation::EcsGetCharacterControllerCollisionCount,
    );
    template_connect_data(&mut snarl, self_entity, 0, hit_collision_count, 0);
    let has_collisions = snarl.insert_node(
        egui::pos2(760.0, 4660.0),
        VisualScriptNodeKind::Compare {
            op: VisualCompareOp::Greater,
        },
    );
    let zero_for_collision = template_number_node(&mut snarl, egui::pos2(760.0, 4750.0), 0.0);
    template_connect_data(&mut snarl, hit_collision_count, 0, has_collisions, 0);
    template_connect_data(&mut snarl, zero_for_collision, 0, has_collisions, 1);
    let hit_entity = template_insert_api_node(
        &mut snarl,
        egui::pos2(560.0, 4840.0),
        VisualApiOperation::EcsGetCharacterControllerHitEntity,
    );
    template_connect_data(&mut snarl, self_entity, 0, hit_entity, 0);
    let hit_entity_exists = template_insert_api_node(
        &mut snarl,
        egui::pos2(760.0, 4840.0),
        VisualApiOperation::EcsEntityExists,
    );
    template_connect_data(&mut snarl, hit_entity, 0, hit_entity_exists, 0);
    let hit_entity_dynamic = template_insert_api_node(
        &mut snarl,
        egui::pos2(760.0, 4930.0),
        VisualApiOperation::EcsHasComponent,
    );
    template_set_api_arg(&mut snarl, hit_entity_dynamic, 1, "dynamic_rigid_body");
    template_connect_data(&mut snarl, hit_entity, 0, hit_entity_dynamic, 0);
    let moving_mag_for_push = template_get_variable_node(
        &mut snarl,
        egui::pos2(560.0, 5020.0),
        VAR_MOVE_MAGNITUDE,
        "move_magnitude",
    );
    let moving_for_push = snarl.insert_node(
        egui::pos2(760.0, 5020.0),
        VisualScriptNodeKind::Compare {
            op: VisualCompareOp::Greater,
        },
    );
    let push_threshold = template_number_node(&mut snarl, egui::pos2(760.0, 5110.0), 0.1);
    template_connect_data(&mut snarl, moving_mag_for_push, 0, moving_for_push, 0);
    template_connect_data(&mut snarl, push_threshold, 0, moving_for_push, 1);

    let can_push_a = snarl.insert_node(
        egui::pos2(960.0, 4750.0),
        VisualScriptNodeKind::LogicalBinary {
            op: VisualLogicalOp::And,
        },
    );
    let can_push = snarl.insert_node(
        egui::pos2(1160.0, 4840.0),
        VisualScriptNodeKind::LogicalBinary {
            op: VisualLogicalOp::And,
        },
    );
    let can_push_b = snarl.insert_node(
        egui::pos2(1160.0, 4930.0),
        VisualScriptNodeKind::LogicalBinary {
            op: VisualLogicalOp::And,
        },
    );
    template_connect_data(&mut snarl, has_collisions, 0, can_push_a, 0);
    template_connect_data(&mut snarl, hit_entity_exists, 0, can_push_a, 1);
    template_connect_data(&mut snarl, can_push_a, 0, can_push_b, 0);
    template_connect_data(&mut snarl, hit_entity_dynamic, 0, can_push_b, 1);
    template_connect_data(&mut snarl, can_push_b, 0, can_push, 0);
    template_connect_data(&mut snarl, moving_for_push, 0, can_push, 1);

    let push_branch = snarl.insert_node(
        egui::pos2(560.0, 5200.0),
        VisualScriptNodeKind::Branch {
            condition: "true".to_string(),
        },
    );
    template_connect_exec(&mut snarl, on_update_seq, 3, push_branch, 0);
    template_connect_data(&mut snarl, can_push, 0, push_branch, 0);

    let push_dir = template_get_variable_node(
        &mut snarl,
        egui::pos2(1360.0, 4840.0),
        VAR_MOVE_WORLD_DIRECTION,
        "move_world_direction",
    );
    let push_strength = template_get_variable_node(
        &mut snarl,
        egui::pos2(1360.0, 4930.0),
        VAR_PUSH_IMPULSE,
        "push_impulse",
    );
    let push_min_strength = template_number_node(&mut snarl, egui::pos2(1360.0, 5020.0), 0.1);
    let push_strength_effective = snarl.insert_node(
        egui::pos2(1560.0, 5020.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Max,
        },
    );
    template_connect_data(&mut snarl, push_strength, 0, push_strength_effective, 0);
    template_connect_data(&mut snarl, push_min_strength, 0, push_strength_effective, 1);
    let push_dir_x = snarl.insert_node(
        egui::pos2(1560.0, 4750.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::X,
        },
    );
    let push_dir_y = snarl.insert_node(
        egui::pos2(1560.0, 4840.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Y,
        },
    );
    let push_dir_z = snarl.insert_node(
        egui::pos2(1560.0, 4930.0),
        VisualScriptNodeKind::Vec3GetComponent {
            component: VisualVec3Component::Z,
        },
    );
    template_connect_data(&mut snarl, push_dir, 0, push_dir_x, 0);
    template_connect_data(&mut snarl, push_dir, 0, push_dir_y, 0);
    template_connect_data(&mut snarl, push_dir, 0, push_dir_z, 0);

    let push_x = snarl.insert_node(
        egui::pos2(1760.0, 4750.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let push_y = snarl.insert_node(
        egui::pos2(1760.0, 4840.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    let push_z = snarl.insert_node(
        egui::pos2(1760.0, 4930.0),
        VisualScriptNodeKind::MathBinary {
            op: VisualMathOp::Multiply,
        },
    );
    template_connect_data(&mut snarl, push_dir_x, 0, push_x, 0);
    template_connect_data(&mut snarl, push_dir_y, 0, push_y, 0);
    template_connect_data(&mut snarl, push_dir_z, 0, push_z, 0);
    template_connect_data(&mut snarl, push_strength_effective, 0, push_x, 1);
    template_connect_data(&mut snarl, push_strength_effective, 0, push_y, 1);
    template_connect_data(&mut snarl, push_strength_effective, 0, push_z, 1);
    let push_vector = snarl.insert_node(egui::pos2(1960.0, 4840.0), VisualScriptNodeKind::Vec3);
    template_connect_data(&mut snarl, push_x, 0, push_vector, 0);
    template_connect_data(&mut snarl, push_y, 0, push_vector, 1);
    template_connect_data(&mut snarl, push_z, 0, push_vector, 2);

    let apply_push = template_insert_api_node(
        &mut snarl,
        egui::pos2(760.0, 5200.0),
        VisualApiOperation::EcsAddForce,
    );
    template_set_api_arg(&mut snarl, apply_push, 2, "true");
    template_connect_exec(&mut snarl, push_branch, 0, apply_push, 0);
    template_connect_data(&mut snarl, hit_entity, 0, apply_push, 0);
    template_connect_data(&mut snarl, push_vector, 0, apply_push, 1);

    VisualScriptDocument {
        version: VISUAL_SCRIPT_VERSION,
        name: "third_person_controller".to_string(),
        prelude: "".to_string(),
        variables,
        functions: Vec::new(),
        graph: graph_data_from_snarl(&snarl),
    }
}

fn default_visual_script_document() -> VisualScriptDocument {
    let mut snarl = Snarl::new();
    let on_start = snarl.insert_node(egui::pos2(48.0, 80.0), VisualScriptNodeKind::OnStart);
    let _on_update = snarl.insert_node(egui::pos2(48.0, 280.0), VisualScriptNodeKind::OnUpdate);
    let start_log = snarl.insert_node(
        egui::pos2(340.0, 80.0),
        VisualScriptNodeKind::Log {
            message: "visual script started".to_string(),
        },
    );

    snarl.connect(
        OutPinId {
            node: on_start,
            output: 0,
        },
        InPinId {
            node: start_log,
            input: 0,
        },
    );

    VisualScriptDocument {
        version: VISUAL_SCRIPT_VERSION,
        name: "visual_script".to_string(),
        prelude: "".to_string(),
        variables: Vec::new(),
        functions: Vec::new(),
        graph: graph_data_from_snarl(&snarl),
    }
}

fn path_display_name(path: &Path) -> String {
    if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
        name.to_string()
    } else {
        path.to_string_lossy().to_string()
    }
}

fn active_graph_path_display(document: &VisualScriptOpenDocument) -> (String, bool) {
    if let Some(function_id) = document.active_graph_function {
        if let Some(function) = document
            .functions
            .iter()
            .find(|function| function.id == function_id)
        {
            return (format!("functions/{}", function.name), true);
        }
    }
    ("Root Event Graph".to_string(), false)
}

fn compact_display_text(value: &str, max_chars: usize) -> String {
    let max_chars = max_chars.max(4);
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = String::with_capacity(max_chars);
    for ch in value.chars().take(max_chars - 3) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn set_status_message(world: &mut World, message: String) {
    if let Some(mut ui_state) = world.get_resource_mut::<crate::editor::EditorUiState>() {
        ui_state.status = Some(message.clone());
    }
    crate::editor::push_console_status(world, message);
}

fn visual_variable_supports_asset_kind(
    value_type: VisualValueType,
    array_item_type: Option<VisualValueType>,
) -> bool {
    if value_type == VisualValueType::String {
        return true;
    }
    if value_type == VisualValueType::Array {
        return array_item_type == Some(VisualValueType::String);
    }
    false
}

fn draw_variable_definitions_panel(
    ui: &mut Ui,
    document: &mut VisualScriptOpenDocument,
    changed: &mut bool,
) {
    ui.label(RichText::new("Variables").strong());

    let mut remove_index: Option<usize> = None;
    let mut spawn_get: Option<u64> = None;
    let mut spawn_set: Option<u64> = None;
    let mut spawn_clear: Option<u64> = None;
    let mut prune_wires = false;

    for (index, variable) in document.variables.iter_mut().enumerate() {
        ui.group(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("#{}", variable.id));
                let name_width = ui.available_width().clamp(80.0, 180.0);
                if ui
                    .add(TextEdit::singleline(&mut variable.name).desired_width(name_width))
                    .changed()
                {
                    *changed = true;
                }

                ComboBox::from_id_salt(("visual_variable_type", index))
                    .selected_text(variable.value_type.title())
                    .show_ui(ui, |ui| {
                        for value_type in VISUAL_VALUE_TYPE_CHOICES_NO_ANY {
                            if ui
                                .selectable_value(
                                    &mut variable.value_type,
                                    value_type,
                                    value_type.title(),
                                )
                                .changed()
                            {
                                normalize_array_item_type(
                                    variable.value_type,
                                    &mut variable.array_item_type,
                                );
                                if !visual_variable_supports_asset_kind(
                                    variable.value_type,
                                    variable.array_item_type,
                                ) {
                                    variable.inspector_asset_kind = None;
                                }
                                variable.default_value = normalize_literal_for_data_type(
                                    &variable.default_value,
                                    value_type,
                                    variable.array_item_type,
                                );
                                *changed = true;
                                prune_wires = true;
                            }
                        }
                    });

                if variable.value_type == VisualValueType::Array {
                    let selected = variable
                        .array_item_type
                        .unwrap_or(default_array_item_type());
                    ComboBox::from_id_salt(("visual_variable_array_item_type", index))
                        .selected_text(selected.title())
                        .show_ui(ui, |ui| {
                            for item_type in VISUAL_ARRAY_ITEM_TYPE_CHOICES {
                                if ui
                                    .selectable_value(
                                        variable.array_item_type.get_or_insert(selected),
                                        item_type,
                                        item_type.title(),
                                    )
                                    .changed()
                                {
                                    normalize_array_item_type(
                                        variable.value_type,
                                        &mut variable.array_item_type,
                                    );
                                    if !visual_variable_supports_asset_kind(
                                        variable.value_type,
                                        variable.array_item_type,
                                    ) {
                                        variable.inspector_asset_kind = None;
                                    }
                                    variable.default_value = normalize_literal_for_data_type(
                                        &variable.default_value,
                                        variable.value_type,
                                        variable.array_item_type,
                                    );
                                    *changed = true;
                                    prune_wires = true;
                                }
                            }
                        });
                }

                if ui.small_button("Delete").clicked() {
                    remove_index = Some(index);
                }
            });

            ui.horizontal(|ui| {
                ui.label("Default");
                if draw_typed_default_editor_with_array_item(
                    ui,
                    variable.value_type,
                    variable.array_item_type,
                    &mut variable.default_value,
                ) {
                    *changed = true;
                }
            });

            ui.horizontal_wrapped(|ui| {
                if ui
                    .checkbox(&mut variable.inspector_exposed, "Expose in Inspector")
                    .changed()
                {
                    *changed = true;
                }

                if variable.inspector_exposed {
                    ui.label("Label");
                    if ui
                        .text_edit_singleline(&mut variable.inspector_label)
                        .changed()
                    {
                        *changed = true;
                    }

                    if visual_variable_supports_asset_kind(
                        variable.value_type,
                        variable.array_item_type,
                    ) {
                        let mut asset_kind = variable
                            .inspector_asset_kind
                            .unwrap_or(VisualInspectorAssetKind::Any);
                        ComboBox::from_id_salt(("visual_variable_asset_kind", index))
                            .selected_text(asset_kind.title())
                            .show_ui(ui, |ui| {
                                for kind in [
                                    VisualInspectorAssetKind::Any,
                                    VisualInspectorAssetKind::Scene,
                                    VisualInspectorAssetKind::Model,
                                    VisualInspectorAssetKind::Material,
                                    VisualInspectorAssetKind::Audio,
                                    VisualInspectorAssetKind::Script,
                                    VisualInspectorAssetKind::Animation,
                                ] {
                                    if ui
                                        .selectable_value(&mut asset_kind, kind, kind.title())
                                        .changed()
                                    {
                                        *changed = true;
                                    }
                                }
                            });
                        variable.inspector_asset_kind = Some(asset_kind);
                    } else {
                        variable.inspector_asset_kind = None;
                    }
                } else {
                    variable.inspector_asset_kind = None;
                }
            });

            ui.horizontal(|ui| {
                if ui.small_button("Get Node").clicked() {
                    spawn_get = Some(variable.id);
                }
                if ui.small_button("Set Node").clicked() {
                    spawn_set = Some(variable.id);
                }
                if ui.small_button("Clear Node").clicked() {
                    spawn_clear = Some(variable.id);
                }
            });
        });
    }

    if let Some(index) = remove_index {
        if index < document.variables.len() {
            document.variables.remove(index);
            *changed = true;
            prune_wires = true;
        }
    }

    if ui.button("Add Variable").clicked() {
        let id = next_visual_variable_id(&document.variables);
        document.variables.push(VisualVariableDefinition {
            id,
            name: format!("var_{}", id),
            value_type: VisualValueType::String,
            array_item_type: None,
            default_value: String::new(),
            inspector_exposed: false,
            inspector_label: String::new(),
            inspector_asset_kind: None,
        });
        *changed = true;
    }

    if let Some(variable_id) = spawn_get {
        let name = document
            .variables
            .iter()
            .find(|var| var.id == variable_id)
            .map(|var| var.name.clone())
            .unwrap_or_else(default_var_name);
        document.snarl.insert_node(
            egui::pos2(180.0, 140.0),
            VisualScriptNodeKind::GetVariable {
                variable_id,
                name,
                default_value: String::new(),
            },
        );
        *changed = true;
    }

    if let Some(variable_id) = spawn_set {
        let (name, default_value) = document
            .variables
            .iter()
            .find(|var| var.id == variable_id)
            .map(|var| (var.name.clone(), var.default_value.clone()))
            .unwrap_or_else(|| (default_var_name(), default_var_value()));
        document.snarl.insert_node(
            egui::pos2(180.0, 220.0),
            VisualScriptNodeKind::SetVariable {
                variable_id,
                name,
                value: default_value,
            },
        );
        *changed = true;
    }

    if let Some(variable_id) = spawn_clear {
        let name = document
            .variables
            .iter()
            .find(|var| var.id == variable_id)
            .map(|var| var.name.clone())
            .unwrap_or_else(default_var_name);
        document.snarl.insert_node(
            egui::pos2(180.0, 300.0),
            VisualScriptNodeKind::ClearVariable { variable_id, name },
        );
        *changed = true;
    }

    if prune_wires {
        if prune_invalid_wires(&mut document.snarl, &document.variables) > 0 {
            *changed = true;
        }
        for snarl in document.function_snarls.values_mut() {
            if prune_invalid_wires(snarl, &document.variables) > 0 {
                *changed = true;
            }
        }
    }
}

fn sanitize_filename_segment(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    if out.is_empty() {
        "function".to_string()
    } else {
        out
    }
}

fn sync_call_function_nodes_in_snarl(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    functions: &[VisualScriptFunctionDefinition],
) -> bool {
    let mut changed = false;
    for (_node_id, node) in snarl.nodes_ids_data_mut() {
        if let VisualScriptNodeKind::CallFunction {
            function_id,
            name,
            inputs,
            outputs,
            args,
        } = &mut node.value
        {
            if let Some(function) = find_function_definition(functions, *function_id, name) {
                let previous_id = *function_id;
                let previous_name = name.clone();
                let previous_inputs = inputs.clone();
                let previous_outputs = outputs.clone();

                *function_id = function.id;
                *name = function.name.clone();
                *inputs = function.inputs.clone();
                *outputs = function.outputs.clone();
                args.truncate(inputs.len());
                while args.len() < inputs.len() {
                    let value_type = inputs
                        .get(args.len())
                        .map(|port| port.value_type)
                        .unwrap_or(VisualValueType::Any);
                    args.push(default_literal_for_type(value_type).to_string());
                }

                if previous_id != *function_id
                    || previous_name != *name
                    || previous_inputs != *inputs
                    || previous_outputs != *outputs
                {
                    changed = true;
                }
            }
        }
    }
    changed
}

fn draw_function_port_list(
    ui: &mut Ui,
    label: &str,
    ports: &mut Vec<VisualFunctionIoDefinition>,
    changed: &mut bool,
) -> bool {
    ui.label(RichText::new(label).strong());
    let mut signature_changed = false;
    let mut remove_index: Option<usize> = None;
    for (index, port) in ports.iter_mut().enumerate() {
        ui.group(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("#{}", port.id));
                let name_width = ui.available_width().clamp(80.0, 160.0);
                if ui
                    .add(TextEdit::singleline(&mut port.name).desired_width(name_width))
                    .changed()
                {
                    *changed = true;
                    signature_changed = true;
                }
                ComboBox::from_id_salt((label, "function_port_type", index))
                    .selected_text(port.value_type.title())
                    .show_ui(ui, |ui| {
                        for value_type in VISUAL_VALUE_TYPE_CHOICES_NO_ANY {
                            if ui
                                .selectable_value(
                                    &mut port.value_type,
                                    value_type,
                                    value_type.title(),
                                )
                                .changed()
                            {
                                normalize_array_item_type(
                                    port.value_type,
                                    &mut port.array_item_type,
                                );
                                port.default_value = normalize_literal_for_data_type(
                                    &port.default_value,
                                    value_type,
                                    port.array_item_type,
                                );
                                *changed = true;
                                signature_changed = true;
                            }
                        }
                    });
                if port.value_type == VisualValueType::Array {
                    let selected = port.array_item_type.unwrap_or(default_array_item_type());
                    ComboBox::from_id_salt((label, "function_port_array_item_type", index))
                        .selected_text(selected.title())
                        .show_ui(ui, |ui| {
                            for item_type in VISUAL_ARRAY_ITEM_TYPE_CHOICES {
                                if ui
                                    .selectable_value(
                                        port.array_item_type.get_or_insert(selected),
                                        item_type,
                                        item_type.title(),
                                    )
                                    .changed()
                                {
                                    normalize_array_item_type(
                                        port.value_type,
                                        &mut port.array_item_type,
                                    );
                                    port.default_value = normalize_literal_for_data_type(
                                        &port.default_value,
                                        port.value_type,
                                        port.array_item_type,
                                    );
                                    *changed = true;
                                    signature_changed = true;
                                }
                            }
                        });
                }
                if ui.small_button("Delete").clicked() {
                    remove_index = Some(index);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Default");
                if draw_typed_default_editor_with_array_item(
                    ui,
                    port.value_type,
                    port.array_item_type,
                    &mut port.default_value,
                ) {
                    *changed = true;
                }
            });
        });
    }
    if let Some(index) = remove_index {
        if index < ports.len() {
            ports.remove(index);
            *changed = true;
            signature_changed = true;
        }
    }
    if ui.small_button(format!("Add {}", label)).clicked() {
        ports.push(VisualFunctionIoDefinition {
            id: 0,
            name: default_function_io_name(),
            value_type: VisualValueType::String,
            array_item_type: None,
            default_value: String::new(),
        });
        *changed = true;
        signature_changed = true;
    }
    signature_changed
}

fn draw_function_definitions_panel(
    ui: &mut Ui,
    document: &mut VisualScriptOpenDocument,
    changed: &mut bool,
) {
    ui.label(RichText::new("Functions").strong());

    ui.horizontal_wrapped(|ui| {
        if ui.button("Root Graph").clicked() {
            document.active_graph_function = None;
        }
        if ui.button("Add Function").clicked() {
            let id = next_function_id(&document.functions);
            let mut function = VisualScriptFunctionDefinition {
                id,
                name: format!("function_{}", id),
                source_path: String::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                graph: VisualScriptGraphData::default(),
            };
            normalize_function_io_ports(&mut function.inputs, "input");
            normalize_function_io_ports(&mut function.outputs, "output");
            sync_function_signature_nodes(&mut function);
            document
                .function_snarls
                .insert(function.id, graph_data_to_snarl(&function.graph));
            document.active_graph_function = Some(function.id);
            document.functions.push(function);
            *changed = true;
        }
    });

    let asset_path_id = ui.id().with(("function_asset_path", &document.path));
    let mut asset_path = ui
        .data_mut(|data| data.get_temp::<String>(asset_path_id))
        .unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("Function Asset");
        ui.add(
            TextEdit::singleline(&mut asset_path).desired_width(ui.available_width().max(120.0)),
        );
    });
    ui.data_mut(|data| data.insert_temp(asset_path_id, asset_path.clone()));

    ui.horizontal_wrapped(|ui| {
        if ui.button("Import").clicked() {
            let import_path = if asset_path.trim().is_empty() {
                None
            } else {
                Some(PathBuf::from(asset_path.trim()))
            };
            if let Some(path) = import_path {
                if let Ok(source) = fs::read_to_string(&path) {
                    if let Ok(mut asset) = ron::de::from_str::<VisualScriptFunctionAsset>(&source) {
                        let id = next_function_id(&document.functions);
                        asset.function.id = id;
                        if asset.function.source_path.trim().is_empty() {
                            asset.function.source_path = path.to_string_lossy().to_string();
                        }
                        normalize_function_io_ports(&mut asset.function.inputs, "input");
                        normalize_function_io_ports(&mut asset.function.outputs, "output");
                        sync_function_signature_nodes(&mut asset.function);
                        document.function_snarls.insert(
                            asset.function.id,
                            graph_data_to_snarl(&asset.function.graph),
                        );
                        document.active_graph_function = Some(asset.function.id);
                        document.functions.push(asset.function);
                        *changed = true;
                    }
                }
            }
        }
        if ui.button("Export Active").clicked() {
            if let Some(function_id) = document.active_graph_function {
                if let Some(function) = document
                    .functions
                    .iter()
                    .find(|function| function.id == function_id)
                {
                    let output_path = if asset_path.trim().is_empty() {
                        let stem = document
                            .path
                            .file_stem()
                            .and_then(|value| value.to_str())
                            .unwrap_or("visual_script");
                        document.path.with_file_name(format!(
                            "{}__{}.{}",
                            stem,
                            sanitize_filename_segment(&function.name),
                            VISUAL_SCRIPT_FUNCTION_EXTENSION
                        ))
                    } else {
                        PathBuf::from(asset_path.trim())
                    };
                    let mut asset = VisualScriptFunctionAsset {
                        version: VISUAL_SCRIPT_VERSION,
                        function: function.clone(),
                    };
                    if let Some(snarl) = document.function_snarls.get(&function.id) {
                        asset.function.graph = graph_data_from_snarl(snarl);
                    }
                    let pretty = PrettyConfig::new().compact_arrays(false);
                    if let Ok(payload) = ron::ser::to_string_pretty(&asset, pretty) {
                        if fs::write(&output_path, payload).is_ok() {
                            ui.data_mut(|data| {
                                data.insert_temp(
                                    asset_path_id,
                                    output_path.to_string_lossy().to_string(),
                                )
                            });
                        }
                    }
                }
            }
        }
    });

    let mut remove_index: Option<usize> = None;
    let mut signatures_changed = false;
    let mut requested_active_graph = document.active_graph_function;
    let mut spawn_call_nodes: Vec<(
        u64,
        String,
        Vec<VisualFunctionIoDefinition>,
        Vec<VisualFunctionIoDefinition>,
        f32,
    )> = Vec::new();
    for (index, function) in document.functions.iter_mut().enumerate() {
        ui.group(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("#{}", function.id));
                let name_width = ui.available_width().clamp(80.0, 180.0);
                if ui
                    .add(TextEdit::singleline(&mut function.name).desired_width(name_width))
                    .changed()
                {
                    *changed = true;
                    signatures_changed = true;
                }
                if ui.small_button("Graph").clicked() {
                    requested_active_graph = Some(function.id);
                }
                if ui.small_button("Call Node").clicked() {
                    spawn_call_nodes.push((
                        function.id,
                        function.name.clone(),
                        function.inputs.clone(),
                        function.outputs.clone(),
                        160.0 + (index as f32 * 28.0),
                    ));
                }
                if ui.small_button("Delete").clicked() {
                    remove_index = Some(index);
                }
            });

            if requested_active_graph == Some(function.id) {
                ui.horizontal(|ui| {
                    ui.label("Source Path");
                    if ui
                        .add(
                            TextEdit::singleline(&mut function.source_path)
                                .desired_width(ui.available_width()),
                        )
                        .changed()
                    {
                        *changed = true;
                    }
                });
                signatures_changed |=
                    draw_function_port_list(ui, "Inputs", &mut function.inputs, changed);
                signatures_changed |=
                    draw_function_port_list(ui, "Outputs", &mut function.outputs, changed);
            }
        });
    }

    for (function_id, name, inputs, outputs, y) in spawn_call_nodes {
        document.snarl.insert_node(
            egui::pos2(220.0, y),
            VisualScriptNodeKind::CallFunction {
                function_id,
                name,
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                args: inputs
                    .iter()
                    .map(|port| default_literal_for_type(port.value_type).to_string())
                    .collect(),
            },
        );
        *changed = true;
    }

    if let Some(index) = remove_index {
        if index < document.functions.len() {
            let removed = document.functions.remove(index);
            document.function_snarls.remove(&removed.id);
            if requested_active_graph == Some(removed.id) {
                requested_active_graph = None;
            }
            *changed = true;
            signatures_changed = true;
        }
    }

    document.active_graph_function = requested_active_graph;

    if signatures_changed {
        for function in &mut document.functions {
            if let Some(snarl) = document.function_snarls.get(&function.id) {
                function.graph = graph_data_from_snarl(snarl);
            }
            normalize_function_io_ports(&mut function.inputs, "input");
            normalize_function_io_ports(&mut function.outputs, "output");
            sync_function_signature_nodes(function);
            document
                .function_snarls
                .insert(function.id, graph_data_to_snarl(&function.graph));
        }
        if sync_call_function_nodes_in_snarl(&mut document.snarl, &document.functions) {
            *changed = true;
        }
        if prune_invalid_wires(&mut document.snarl, &document.variables) > 0 {
            *changed = true;
        }
        for snarl in document.function_snarls.values_mut() {
            if sync_call_function_nodes_in_snarl(snarl, &document.functions) {
                *changed = true;
            }
            if prune_invalid_wires(snarl, &document.variables) > 0 {
                *changed = true;
            }
        }
    }
}

#[derive(Default)]
struct VisualScriptViewer {
    changed: bool,
    consumed_asset_drop: bool,
    pending_asset_drop_nodes: Vec<(String, egui::Pos2)>,
    project_root: Option<PathBuf>,
    variables: Vec<VisualVariableDefinition>,
    functions: Vec<VisualScriptFunctionDefinition>,
    active_function_graph: Option<u64>,
    open_function_graph_request: Option<u64>,
}

impl VisualScriptViewer {
    fn with_context(
        variables: &[VisualVariableDefinition],
        functions: &[VisualScriptFunctionDefinition],
        active_function_graph: Option<u64>,
        project_root: Option<&Path>,
    ) -> Self {
        Self {
            changed: false,
            consumed_asset_drop: false,
            pending_asset_drop_nodes: Vec::new(),
            project_root: project_root.map(Path::to_path_buf),
            variables: variables.to_vec(),
            functions: functions.to_vec(),
            active_function_graph,
            open_function_graph_request: None,
        }
    }

    fn mark_changed(&mut self) {
        self.changed = true;
    }

    fn is_function_graph(&self) -> bool {
        self.active_function_graph.is_some()
    }

    fn find_function(&self, function_id: u64) -> Option<&VisualScriptFunctionDefinition> {
        self.functions
            .iter()
            .find(|function| function.id == function_id)
    }

    fn add_node_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut Ui,
        snarl: &mut Snarl<VisualScriptNodeKind>,
        wire_context: Option<&AnyPins<'_>>,
    ) -> Option<NodeId> {
        enum AddNodeSearchAction {
            InsertNode(VisualScriptNodeKind),
            InsertApi(VisualApiOperation),
        }

        ui.label("Add node");
        let search_id = ui.id().with("visual_script_add_node_search");
        let mut search = ui
            .data_mut(|data| data.get_temp::<String>(search_id))
            .unwrap_or_default();
        ui.add(
            TextEdit::singleline(&mut search)
                .desired_width(280.0)
                .hint_text("Search nodes..."),
        );
        ui.data_mut(|data| data.insert_temp(search_id, search.clone()));
        let search = search.trim().to_ascii_lowercase();
        let has_search = !search.is_empty();

        let mut inserted = None;
        let mut has_visible = false;
        let force_compatible_only = wire_context.is_some();

        if force_compatible_only {
            ui.small("Showing only nodes compatible with the dropped wire");
        }

        if has_search || force_compatible_only {
            let mut results: Vec<(String, AddNodeSearchAction)> = Vec::new();

            let mut push_search_result =
                |label: String, terms: &[&str], action: AddNodeSearchAction| {
                    if add_node_search_matches_any(&search, terms) {
                        results.push((label, action));
                    }
                };

            if !self.is_function_graph() {
                push_search_result(
                    "On Start".to_string(),
                    &["on start", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnStart),
                );
                push_search_result(
                    "On Update".to_string(),
                    &["on update", "tick", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnUpdate),
                );
                push_search_result(
                    "On Stop".to_string(),
                    &["on stop", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnStop),
                );
                push_search_result(
                    "On Collision Enter".to_string(),
                    &["collision enter", "on collision", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnCollisionEnter),
                );
                push_search_result(
                    "On Collision Stay".to_string(),
                    &["collision stay", "on collision", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnCollisionStay),
                );
                push_search_result(
                    "On Collision Exit".to_string(),
                    &["collision exit", "on collision", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnCollisionExit),
                );
                push_search_result(
                    "On Trigger Enter".to_string(),
                    &["trigger enter", "on trigger", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnTriggerEnter),
                );
                push_search_result(
                    "On Trigger Exit".to_string(),
                    &["trigger exit", "on trigger", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnTriggerExit),
                );
                push_search_result(
                    "On Input Pressed".to_string(),
                    &["on input", "input action", "pressed", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnInputAction {
                        action: default_input_action_name(),
                        phase: VisualInputActionPhase::Pressed,
                    }),
                );
                push_search_result(
                    "On Input Released".to_string(),
                    &["on input", "input action", "released", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnInputAction {
                        action: default_input_action_name(),
                        phase: VisualInputActionPhase::Released,
                    }),
                );
                push_search_result(
                    "On Input Down".to_string(),
                    &["on input", "input action", "down", "held", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnInputAction {
                        action: default_input_action_name(),
                        phase: VisualInputActionPhase::Down,
                    }),
                );
                push_search_result(
                    "On Custom Event".to_string(),
                    &["custom event", "emit event", "on event", "event"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::OnCustomEvent {
                        name: default_custom_event_name(),
                    }),
                );
            } else if let Some(function_id) = self.active_function_graph {
                let inputs = self
                    .find_function(function_id)
                    .map(|function| function.inputs.clone())
                    .unwrap_or_default();
                let outputs = self
                    .find_function(function_id)
                    .map(|function| function.outputs.clone())
                    .unwrap_or_default();
                push_search_result(
                    "Function Start".to_string(),
                    &["function start", "entry", "start"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::FunctionStart {
                        function_id,
                        inputs,
                    }),
                );
                push_search_result(
                    "Function Return".to_string(),
                    &["function return", "return", "end"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::FunctionReturn {
                        function_id,
                        outputs: outputs.clone(),
                        values: outputs
                            .iter()
                            .map(|port| default_literal_for_type(port.value_type).to_string())
                            .collect(),
                    }),
                );
            }

            let mut api_specs: Vec<&VisualApiOperationSpec> = VISUAL_API_OPERATION_SPECS
                .iter()
                .filter(|spec| {
                    api_spec_matches_add_node_search(spec, api_menu_section(spec), &search)
                })
                .collect();
            api_specs.sort_by_key(|spec| spec.title);
            for spec in api_specs {
                let label = match spec.flow {
                    VisualApiFlow::Exec => format!("API / {} [Call]", spec.title),
                    VisualApiFlow::Pure => format!("API / {}", spec.title),
                };
                let section = api_menu_section(spec);
                push_search_result(
                    label,
                    &[
                        spec.title,
                        spec.function,
                        spec.category,
                        spec.table.title(),
                        section.title(),
                        "api",
                    ],
                    AddNodeSearchAction::InsertApi(spec.operation),
                );
            }

            for (label, node) in [
                (
                    "Sequence",
                    VisualScriptNodeKind::Sequence {
                        outputs: default_sequence_outputs(),
                    },
                ),
                (
                    "Branch",
                    VisualScriptNodeKind::Branch {
                        condition: default_branch_condition(),
                    },
                ),
                (
                    "Loop While",
                    VisualScriptNodeKind::LoopWhile {
                        condition: default_loop_condition(),
                        max_iterations: default_max_loop_iterations(),
                    },
                ),
                (
                    "Log",
                    VisualScriptNodeKind::Log {
                        message: default_log_message(),
                    },
                ),
            ] {
                push_search_result(
                    label.to_string(),
                    &[label, "flow"],
                    AddNodeSearchAction::InsertNode(node),
                );
            }

            for variable in &self.variables {
                let variable_label = if variable.value_type == VisualValueType::Array {
                    let item = variable
                        .array_item_type
                        .unwrap_or(default_array_item_type());
                    format!("{} (Array<{}>)", variable.name, item.title())
                } else {
                    format!("{} ({})", variable.name, variable.value_type.title())
                };
                let search_terms = [
                    variable_label.as_str(),
                    variable.name.as_str(),
                    "variable",
                    "set",
                    "get",
                    "clear",
                ];
                push_search_result(
                    format!("Variable / Set {}", variable.name),
                    &search_terms,
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::SetVariable {
                        variable_id: variable.id,
                        name: variable.name.clone(),
                        value: variable.default_value.clone(),
                    }),
                );
                push_search_result(
                    format!("Variable / Get {}", variable.name),
                    &search_terms,
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::GetVariable {
                        variable_id: variable.id,
                        name: variable.name.clone(),
                        default_value: variable.default_value.clone(),
                    }),
                );
                push_search_result(
                    format!("Variable / Clear {}", variable.name),
                    &search_terms,
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::ClearVariable {
                        variable_id: variable.id,
                        name: variable.name.clone(),
                    }),
                );
            }

            for function in &self.functions {
                let call_label = if self.active_function_graph == Some(function.id) {
                    format!("Call {} (recursive)", function.name)
                } else {
                    format!("Call {}", function.name)
                };
                push_search_result(
                    format!("Function / {}", call_label),
                    &[
                        call_label.as_str(),
                        function.name.as_str(),
                        "function",
                        "subgraph",
                        "call",
                    ],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::CallFunction {
                        function_id: function.id,
                        name: function.name.clone(),
                        inputs: function.inputs.clone(),
                        outputs: function.outputs.clone(),
                        args: function
                            .inputs
                            .iter()
                            .map(|port| default_literal_for_type(port.value_type).to_string())
                            .collect(),
                    }),
                );
            }

            for (label, node, terms) in [
                (
                    "Values / Bool",
                    VisualScriptNodeKind::BoolLiteral { value: false },
                    &["bool", "value", "values"][..],
                ),
                (
                    "Values / Number",
                    VisualScriptNodeKind::NumberLiteral { value: 0.0 },
                    &["number", "value", "values"][..],
                ),
                (
                    "Values / String",
                    VisualScriptNodeKind::StringLiteral {
                        value: "text".to_string(),
                    },
                    &["string", "value", "values", "text"][..],
                ),
                (
                    "Values / Self Entity",
                    VisualScriptNodeKind::SelfEntity,
                    &["self entity", "entity", "value", "values"][..],
                ),
            ] {
                push_search_result(
                    label.to_string(),
                    terms,
                    AddNodeSearchAction::InsertNode(node),
                );
            }

            for (label, node, terms) in [
                (
                    "Time / Delta Time",
                    VisualScriptNodeKind::DeltaTime,
                    &["delta time", "time"][..],
                ),
                (
                    "Time / Time Since Start",
                    VisualScriptNodeKind::TimeSinceStart,
                    &["time", "since start", "elapsed", "seconds"][..],
                ),
                (
                    "Time / Unix Time Seconds",
                    VisualScriptNodeKind::UnixTimeSeconds,
                    &["time", "unix", "date", "clock"][..],
                ),
                (
                    "Time / Time Since",
                    VisualScriptNodeKind::TimeSince {
                        origin_seconds: default_time_since_origin(),
                    },
                    &["time since", "duration", "time"][..],
                ),
                (
                    "Time / Wait Seconds",
                    VisualScriptNodeKind::WaitSeconds {
                        seconds: default_wait_seconds(),
                        restart_on_retrigger: default_wait_restart_on_retrigger(),
                    },
                    &["wait", "delay", "time"][..],
                ),
            ] {
                push_search_result(
                    label.to_string(),
                    terms,
                    AddNodeSearchAction::InsertNode(node),
                );
            }

            for op in [
                VisualMathOp::Add,
                VisualMathOp::Subtract,
                VisualMathOp::Multiply,
                VisualMathOp::Divide,
                VisualMathOp::Modulo,
                VisualMathOp::Min,
                VisualMathOp::Max,
            ] {
                let title = op.title();
                push_search_result(
                    format!("Math / Basic / {}", title),
                    &[title, "math", "basic"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::MathBinary { op }),
                );
            }
            for op in [
                VisualTrigOp::Sin,
                VisualTrigOp::Cos,
                VisualTrigOp::Tan,
                VisualTrigOp::Asin,
                VisualTrigOp::Acos,
                VisualTrigOp::Atan,
                VisualTrigOp::Atan2,
            ] {
                let title = op.title();
                push_search_result(
                    format!("Math / Trig / {}", title),
                    &[title, "math", "trig"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::MathTrig { op }),
                );
            }
            for op in [
                VisualInterpolationOp::Lerp,
                VisualInterpolationOp::SmoothStep,
                VisualInterpolationOp::InverseLerp,
                VisualInterpolationOp::Vec3Lerp,
                VisualInterpolationOp::QuatSlerp,
            ] {
                let title = op.title();
                push_search_result(
                    format!("Math / Interpolation / {}", title),
                    &[title, "math", "interpolation", "lerp", "slerp"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::MathInterpolation { op }),
                );
            }
            for op in [
                VisualVectorMathOp::Dot,
                VisualVectorMathOp::Cross,
                VisualVectorMathOp::Length,
                VisualVectorMathOp::Normalize,
                VisualVectorMathOp::Distance,
            ] {
                let title = op.title();
                push_search_result(
                    format!("Math / Vector / {}", title),
                    &[title, "math", "vector"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::MathVector { op }),
                );
            }
            for op in [
                VisualProceduralMathOp::Clamp,
                VisualProceduralMathOp::Remap,
                VisualProceduralMathOp::Saturate,
                VisualProceduralMathOp::Fract,
            ] {
                let title = op.title();
                push_search_result(
                    format!("Math / Procedural / {}", title),
                    &[title, "math", "procedural"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::MathProcedural { op }),
                );
            }
            for op in [
                VisualUtilityMathOp::Abs,
                VisualUtilityMathOp::Sign,
                VisualUtilityMathOp::Floor,
                VisualUtilityMathOp::Ceil,
                VisualUtilityMathOp::Round,
                VisualUtilityMathOp::Sqrt,
                VisualUtilityMathOp::Pow,
                VisualUtilityMathOp::Exp,
                VisualUtilityMathOp::Log,
                VisualUtilityMathOp::Degrees,
                VisualUtilityMathOp::Radians,
            ] {
                let title = op.title();
                push_search_result(
                    format!("Math / Utility / {}", title),
                    &[title, "math", "utility"],
                    AddNodeSearchAction::InsertNode(VisualScriptNodeKind::MathUtility { op }),
                );
            }

            push_search_result(
                "Logic / Compare".to_string(),
                &["compare", "bool"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Compare {
                    op: VisualCompareOp::Equals,
                }),
            );
            push_search_result(
                "Logic / Logical".to_string(),
                &["logical", "bool"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::LogicalBinary {
                    op: VisualLogicalOp::And,
                }),
            );
            push_search_result(
                "Logic / Not".to_string(),
                &["not", "bool"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Not),
            );
            push_search_result(
                "Logic / Select".to_string(),
                &["select", "bool"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Select {
                    value_type: default_select_value_type(),
                }),
            );

            for (label, node) in [
                (
                    "Array Empty",
                    VisualScriptNodeKind::ArrayEmpty {
                        item_type: default_array_item_type(),
                    },
                ),
                (
                    "Array Length",
                    VisualScriptNodeKind::ArrayLength {
                        item_type: default_array_item_type(),
                    },
                ),
                (
                    "Array Get",
                    VisualScriptNodeKind::ArrayGet {
                        item_type: default_array_item_type(),
                    },
                ),
                (
                    "Array Set",
                    VisualScriptNodeKind::ArraySet {
                        item_type: default_array_item_type(),
                    },
                ),
                (
                    "Array Push",
                    VisualScriptNodeKind::ArrayPush {
                        item_type: default_array_item_type(),
                    },
                ),
                (
                    "Array Remove At",
                    VisualScriptNodeKind::ArrayRemoveAt {
                        item_type: default_array_item_type(),
                    },
                ),
                (
                    "Array Clear",
                    VisualScriptNodeKind::ArrayClear {
                        item_type: default_array_item_type(),
                    },
                ),
            ] {
                push_search_result(
                    label.to_string(),
                    &[label, "array"],
                    AddNodeSearchAction::InsertNode(node),
                );
            }

            push_search_result(
                "Physics Queries / Ray Cast".to_string(),
                &["ray cast", "ray", "physics query"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::RayCast),
            );
            for (label, operation, terms) in [
                (
                    "Physics Queries / Ray Cast Hit",
                    VisualApiOperation::EcsRayCast,
                    &["ray cast", "ray", "hit", "physics query"][..],
                ),
                (
                    "Physics Queries / Ray Cast Has Hit",
                    VisualApiOperation::EcsRayCastHasHit,
                    &["ray cast", "ray", "has hit", "physics query"][..],
                ),
                (
                    "Physics Queries / Ray Cast Hit Entity",
                    VisualApiOperation::EcsRayCastHitEntity,
                    &["ray cast", "ray", "entity", "physics query"][..],
                ),
                (
                    "Physics Queries / Ray Cast Point",
                    VisualApiOperation::EcsRayCastPoint,
                    &["ray cast", "ray", "point", "physics query"][..],
                ),
                (
                    "Physics Queries / Ray Cast Normal",
                    VisualApiOperation::EcsRayCastNormal,
                    &["ray cast", "ray", "normal", "physics query"][..],
                ),
                (
                    "Physics Queries / Ray Cast TOI",
                    VisualApiOperation::EcsRayCastToi,
                    &["ray cast", "ray", "toi", "distance", "physics query"][..],
                ),
                (
                    "Physics Queries / Sphere Cast Hit",
                    VisualApiOperation::EcsSphereCast,
                    &["sphere cast", "shape cast", "hit", "physics query"][..],
                ),
                (
                    "Physics Queries / Sphere Cast Has Hit",
                    VisualApiOperation::EcsSphereCastHasHit,
                    &["sphere cast", "shape cast", "has hit", "physics query"][..],
                ),
                (
                    "Physics Queries / Sphere Cast Hit Entity",
                    VisualApiOperation::EcsSphereCastHitEntity,
                    &["sphere cast", "shape cast", "entity", "physics query"][..],
                ),
                (
                    "Physics Queries / Sphere Cast Point",
                    VisualApiOperation::EcsSphereCastPoint,
                    &["sphere cast", "shape cast", "point", "physics query"][..],
                ),
                (
                    "Physics Queries / Sphere Cast Normal",
                    VisualApiOperation::EcsSphereCastNormal,
                    &["sphere cast", "shape cast", "normal", "physics query"][..],
                ),
                (
                    "Physics Queries / Sphere Cast TOI",
                    VisualApiOperation::EcsSphereCastToi,
                    &["sphere cast", "shape cast", "toi", "physics query"][..],
                ),
                (
                    "Physics Queries / Get Physics Ray Cast Hit",
                    VisualApiOperation::EcsGetPhysicsRayCastHit,
                    &["get physics ray cast hit", "ray cast hit", "physics query"][..],
                ),
                (
                    "Physics Queries / Get Physics Shape Cast Hit",
                    VisualApiOperation::EcsGetPhysicsShapeCastHit,
                    &[
                        "get physics shape cast hit",
                        "sphere cast hit",
                        "physics query",
                    ][..],
                ),
                (
                    "Physics Queries / Get Physics Point Projection Hit",
                    VisualApiOperation::EcsGetPhysicsPointProjectionHit,
                    &[
                        "get physics point projection hit",
                        "point projection",
                        "physics query",
                    ][..],
                ),
            ] {
                push_search_result(
                    label.to_string(),
                    terms,
                    AddNodeSearchAction::InsertApi(operation),
                );
            }
            push_search_result(
                "Physics Queries / Physics Query Filter".to_string(),
                &[
                    "physics query filter",
                    "query filter",
                    "ray filter",
                    "physics query",
                ],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::PhysicsQueryFilterLiteral {
                    value: default_literal_for_type(VisualValueType::PhysicsQueryFilter)
                        .to_string(),
                }),
            );

            push_search_result(
                "Structured / Vec2".to_string(),
                &["vec2", "vector", "structured"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Vec2),
            );
            push_search_result(
                "Structured / Vec2 Get Component".to_string(),
                &["vec2", "component", "extract"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Vec2GetComponent {
                    component: VisualVec2Component::X,
                }),
            );
            push_search_result(
                "Structured / Vec2 Set Component".to_string(),
                &["vec2", "component", "set", "mutate"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Vec2SetComponent {
                    component: VisualVec2Component::X,
                }),
            );
            push_search_result(
                "Structured / Vec3".to_string(),
                &["vec3", "vector", "structured"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Vec3),
            );
            push_search_result(
                "Structured / Vec3 Get Component".to_string(),
                &["vec3", "component", "extract"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Vec3GetComponent {
                    component: VisualVec3Component::X,
                }),
            );
            push_search_result(
                "Structured / Vec3 Set Component".to_string(),
                &["vec3", "component", "set", "mutate"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Vec3SetComponent {
                    component: VisualVec3Component::X,
                }),
            );
            push_search_result(
                "Structured / Quat".to_string(),
                &["quat", "rotation", "structured"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Quat),
            );
            push_search_result(
                "Structured / Quat Get Component".to_string(),
                &["quat", "component", "extract"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::QuatGetComponent {
                    component: VisualQuatComponent::X,
                }),
            );
            push_search_result(
                "Structured / Quat Set Component".to_string(),
                &["quat", "component", "set", "mutate"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::QuatSetComponent {
                    component: VisualQuatComponent::X,
                }),
            );
            push_search_result(
                "Structured / Transform".to_string(),
                &["transform", "structured"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Transform),
            );
            push_search_result(
                "Structured / Transform Get Part".to_string(),
                &["transform", "extract", "part"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::TransformGetComponent {
                    component: VisualTransformComponent::Position,
                }),
            );
            push_search_result(
                "Structured / Transform Set Part".to_string(),
                &["transform", "set", "part", "mutate"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::TransformSetComponent {
                    component: VisualTransformComponent::Position,
                }),
            );
            push_search_result(
                "Structured / Physics Velocity".to_string(),
                &["physics velocity", "physics", "velocity", "structured"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::PhysicsVelocity),
            );
            push_search_result(
                "Structured / Physics Velocity Get Field".to_string(),
                &["physics velocity", "physics", "velocity", "extract", "get"],
                AddNodeSearchAction::InsertNode(
                    VisualScriptNodeKind::PhysicsVelocityGetComponent {
                        component: VisualPhysicsVelocityComponent::Linear,
                    },
                ),
            );
            push_search_result(
                "Structured / Physics Velocity Set Field".to_string(),
                &["physics velocity", "physics", "velocity", "set", "mutate"],
                AddNodeSearchAction::InsertNode(
                    VisualScriptNodeKind::PhysicsVelocitySetComponent {
                        component: VisualPhysicsVelocityComponent::Linear,
                    },
                ),
            );

            push_search_result(
                "Other / Comment".to_string(),
                &["comment", "notes"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Comment {
                    text: "notes".to_string(),
                }),
            );
            push_search_result(
                "Other / Legacy Statement".to_string(),
                &["legacy", "statement"],
                AddNodeSearchAction::InsertNode(VisualScriptNodeKind::Statement {
                    code: "-- legacy".to_string(),
                }),
            );

            if let Some(src_pins) = wire_context {
                results.retain(|(_, action)| {
                    let candidate = match action {
                        AddNodeSearchAction::InsertNode(node) => node.clone(),
                        AddNodeSearchAction::InsertApi(operation) => {
                            api_node_kind_from_spec(operation.spec())
                        }
                    };
                    node_kind_has_compatible_pin_for_dropped_wire(
                        &candidate,
                        src_pins,
                        snarl,
                        &self.variables,
                    )
                });
            }

            results.sort_by_key(|(label, _)| label.to_ascii_lowercase());
            if results.is_empty() {
                if force_compatible_only {
                    ui.small("No compatible nodes for this wire");
                } else {
                    ui.small("No nodes match the current search");
                }
            } else {
                egui::ScrollArea::vertical()
                    .max_height(340.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (label, action) in results {
                            if ui.button(label).clicked() {
                                inserted = Some(match action {
                                    AddNodeSearchAction::InsertNode(node) => {
                                        snarl.insert_node(pos, node)
                                    }
                                    AddNodeSearchAction::InsertApi(operation) => {
                                        insert_api_node_from_spec(snarl, pos, operation.spec())
                                    }
                                });
                                ui.close();
                            }
                        }
                    });
            }

            if inserted.is_some() {
                self.mark_changed();
            }

            return inserted;
        }

        macro_rules! add_node_button {
            ($ui:expr, $label:expr, $terms:expr, $node:expr) => {{
                if inserted.is_none() && add_node_search_matches_any(&search, $terms) {
                    has_visible = true;
                    if $ui.button($label).clicked() {
                        inserted = Some(snarl.insert_node(pos, $node));
                        $ui.close();
                    }
                }
            }};
        }

        ui.menu_button(
            if self.is_function_graph() {
                "Function Flow"
            } else {
                "Events"
            },
            |ui| {
                if !self.is_function_graph() {
                    add_node_button!(
                        ui,
                        "On Start",
                        &["on start", "event"],
                        VisualScriptNodeKind::OnStart
                    );
                    add_node_button!(
                        ui,
                        "On Update",
                        &["on update", "tick", "event"],
                        VisualScriptNodeKind::OnUpdate
                    );
                    add_node_button!(
                        ui,
                        "On Stop",
                        &["on stop", "event"],
                        VisualScriptNodeKind::OnStop
                    );
                    add_node_button!(
                        ui,
                        "On Collision Enter",
                        &["collision enter", "on collision", "event"],
                        VisualScriptNodeKind::OnCollisionEnter
                    );
                    add_node_button!(
                        ui,
                        "On Collision Stay",
                        &["collision stay", "on collision", "event"],
                        VisualScriptNodeKind::OnCollisionStay
                    );
                    add_node_button!(
                        ui,
                        "On Collision Exit",
                        &["collision exit", "on collision", "event"],
                        VisualScriptNodeKind::OnCollisionExit
                    );
                    add_node_button!(
                        ui,
                        "On Trigger Enter",
                        &["trigger enter", "on trigger", "event"],
                        VisualScriptNodeKind::OnTriggerEnter
                    );
                    add_node_button!(
                        ui,
                        "On Trigger Exit",
                        &["trigger exit", "on trigger", "event"],
                        VisualScriptNodeKind::OnTriggerExit
                    );
                    add_node_button!(
                        ui,
                        "On Input Pressed",
                        &["on input", "input action", "pressed", "event"],
                        VisualScriptNodeKind::OnInputAction {
                            action: default_input_action_name(),
                            phase: VisualInputActionPhase::Pressed,
                        }
                    );
                    add_node_button!(
                        ui,
                        "On Input Released",
                        &["on input", "input action", "released", "event"],
                        VisualScriptNodeKind::OnInputAction {
                            action: default_input_action_name(),
                            phase: VisualInputActionPhase::Released,
                        }
                    );
                    add_node_button!(
                        ui,
                        "On Input Down",
                        &["on input", "input action", "down", "held", "event"],
                        VisualScriptNodeKind::OnInputAction {
                            action: default_input_action_name(),
                            phase: VisualInputActionPhase::Down,
                        }
                    );
                    add_node_button!(
                        ui,
                        "On Custom Event",
                        &["custom event", "emit event", "on event", "event"],
                        VisualScriptNodeKind::OnCustomEvent {
                            name: default_custom_event_name(),
                        }
                    );
                } else if let Some(function_id) = self.active_function_graph {
                    let inputs = self
                        .find_function(function_id)
                        .map(|function| function.inputs.clone())
                        .unwrap_or_default();
                    let outputs = self
                        .find_function(function_id)
                        .map(|function| function.outputs.clone())
                        .unwrap_or_default();
                    add_node_button!(
                        ui,
                        "Function Start",
                        &["function start", "entry", "start"],
                        VisualScriptNodeKind::FunctionStart {
                            function_id,
                            inputs
                        }
                    );
                    add_node_button!(
                        ui,
                        "Function Return",
                        &["function return", "return", "end"],
                        VisualScriptNodeKind::FunctionReturn {
                            function_id,
                            outputs: outputs.clone(),
                            values: outputs
                                .iter()
                                .map(|port| default_literal_for_type(port.value_type).to_string())
                                .collect(),
                        }
                    );
                }
            },
        );

        ui.menu_button("API", |ui| {
            let mut sections_visible = false;
            for section in VISUAL_API_MENU_SECTION_ORDER {
                let mut specs: Vec<&VisualApiOperationSpec> = VISUAL_API_OPERATION_SPECS
                    .iter()
                    .filter(|spec| api_menu_section(spec) == section)
                    .filter(|spec| api_spec_matches_add_node_search(spec, section, &search))
                    .collect();
                if specs.is_empty() {
                    continue;
                }
                sections_visible = true;
                has_visible = true;
                specs.sort_by_key(|spec| spec.title);
                ui.menu_button(format!("{} ({})", section.title(), specs.len()), |ui| {
                    for spec in specs {
                        let label = match spec.flow {
                            VisualApiFlow::Exec => format!("{} [Call]", spec.title),
                            VisualApiFlow::Pure => spec.title.to_string(),
                        };
                        if ui.button(label).clicked() {
                            inserted = Some(insert_api_node_from_spec(snarl, pos, spec));
                            ui.close();
                        }
                    }
                });
            }
            if !sections_visible && add_node_search_matches_any(&search, &["api"]) {
                has_visible = true;
                ui.small("No API nodes match the current search");
            }
        });

        ui.menu_button("Flow", |ui| {
            add_node_button!(
                ui,
                "Sequence",
                &["sequence", "flow"],
                VisualScriptNodeKind::Sequence {
                    outputs: default_sequence_outputs(),
                }
            );
            add_node_button!(
                ui,
                "Branch",
                &["branch", "flow"],
                VisualScriptNodeKind::Branch {
                    condition: default_branch_condition(),
                }
            );
            add_node_button!(
                ui,
                "Loop While",
                &["loop while", "loop", "flow"],
                VisualScriptNodeKind::LoopWhile {
                    condition: default_loop_condition(),
                    max_iterations: default_max_loop_iterations(),
                }
            );
            add_node_button!(
                ui,
                "Log",
                &["log", "message"],
                VisualScriptNodeKind::Log {
                    message: default_log_message(),
                }
            );
        });

        ui.menu_button("Variables", |ui| {
            if self.variables.is_empty() {
                if add_node_search_matches_any(&search, &["variables", "variable"]) {
                    has_visible = true;
                    ui.small("Define variables in the side panel to add getter/setter nodes");
                }
                return;
            }

            for variable in &self.variables {
                let mut display = format!("{} ({})", variable.name, variable.value_type.title());
                if variable.value_type == VisualValueType::Array {
                    let item = variable
                        .array_item_type
                        .unwrap_or(default_array_item_type());
                    display = format!("{} (Array<{}>)", variable.name, item.title());
                }
                if !add_node_search_matches_any(
                    &search,
                    &[
                        display.as_str(),
                        variable.name.as_str(),
                        "variable",
                        "set",
                        "get",
                        "clear",
                    ],
                ) {
                    continue;
                }
                has_visible = true;
                ui.menu_button(display, |ui| {
                    if ui.button("Set").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::SetVariable {
                                variable_id: variable.id,
                                name: variable.name.clone(),
                                value: variable.default_value.clone(),
                            },
                        ));
                        ui.close();
                    }
                    if ui.button("Get").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::GetVariable {
                                variable_id: variable.id,
                                name: variable.name.clone(),
                                default_value: variable.default_value.clone(),
                            },
                        ));
                        ui.close();
                    }
                    if ui.button("Clear").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::ClearVariable {
                                variable_id: variable.id,
                                name: variable.name.clone(),
                            },
                        ));
                        ui.close();
                    }
                });
            }
        });

        ui.menu_button("Functions", |ui| {
            if self.functions.is_empty() {
                if add_node_search_matches_any(&search, &["function", "subgraph"]) {
                    has_visible = true;
                    ui.small("Create a function in the side panel to add call nodes");
                }
                return;
            }

            for function in &self.functions {
                let label = if self.active_function_graph == Some(function.id) {
                    format!("Call {} (recursive)", function.name)
                } else {
                    format!("Call {}", function.name)
                };
                if !add_node_search_matches_any(
                    &search,
                    &[
                        label.as_str(),
                        function.name.as_str(),
                        "function",
                        "subgraph",
                        "call",
                    ],
                ) {
                    continue;
                }
                has_visible = true;
                if ui.button(label).clicked() {
                    inserted = Some(
                        snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::CallFunction {
                                function_id: function.id,
                                name: function.name.clone(),
                                inputs: function.inputs.clone(),
                                outputs: function.outputs.clone(),
                                args: function
                                    .inputs
                                    .iter()
                                    .map(|port| {
                                        default_literal_for_type(port.value_type).to_string()
                                    })
                                    .collect(),
                            },
                        ),
                    );
                    ui.close();
                }
            }
        });

        ui.menu_button("Values", |ui| {
            add_node_button!(
                ui,
                "Bool",
                &["bool", "value"],
                VisualScriptNodeKind::BoolLiteral { value: false }
            );
            add_node_button!(
                ui,
                "Number",
                &["number", "value"],
                VisualScriptNodeKind::NumberLiteral { value: 0.0 }
            );
            add_node_button!(
                ui,
                "String",
                &["string", "value", "text"],
                VisualScriptNodeKind::StringLiteral {
                    value: "text".to_string(),
                }
            );
            add_node_button!(
                ui,
                "Self Entity",
                &["self entity", "entity", "value"],
                VisualScriptNodeKind::SelfEntity
            );
        });

        ui.menu_button("Time", |ui| {
            add_node_button!(
                ui,
                "Delta Time",
                &["delta time", "time"],
                VisualScriptNodeKind::DeltaTime
            );
            add_node_button!(
                ui,
                "Time Since Start",
                &["time", "since start", "elapsed", "seconds"],
                VisualScriptNodeKind::TimeSinceStart
            );
            add_node_button!(
                ui,
                "Unix Time Seconds",
                &["time", "unix", "date", "clock"],
                VisualScriptNodeKind::UnixTimeSeconds
            );
            add_node_button!(
                ui,
                "Time Since",
                &["time since", "duration", "time"],
                VisualScriptNodeKind::TimeSince {
                    origin_seconds: default_time_since_origin(),
                }
            );
            add_node_button!(
                ui,
                "Wait Seconds",
                &["wait", "delay", "time"],
                VisualScriptNodeKind::WaitSeconds {
                    seconds: default_wait_seconds(),
                    restart_on_retrigger: default_wait_restart_on_retrigger(),
                }
            );
        });

        ui.menu_button("Math", |ui| {
            ui.menu_button("Basic", |ui| {
                for op in [
                    VisualMathOp::Add,
                    VisualMathOp::Subtract,
                    VisualMathOp::Multiply,
                    VisualMathOp::Divide,
                    VisualMathOp::Modulo,
                    VisualMathOp::Min,
                    VisualMathOp::Max,
                ] {
                    let title = op.title();
                    add_node_button!(
                        ui,
                        title,
                        &[title, "math", "basic"],
                        VisualScriptNodeKind::MathBinary { op }
                    );
                }
            });
            ui.menu_button("Trig", |ui| {
                for op in [
                    VisualTrigOp::Sin,
                    VisualTrigOp::Cos,
                    VisualTrigOp::Tan,
                    VisualTrigOp::Asin,
                    VisualTrigOp::Acos,
                    VisualTrigOp::Atan,
                    VisualTrigOp::Atan2,
                ] {
                    let title = op.title();
                    add_node_button!(
                        ui,
                        title,
                        &[title, "math", "trig"],
                        VisualScriptNodeKind::MathTrig { op }
                    );
                }
            });
            ui.menu_button("Interpolation", |ui| {
                for op in [
                    VisualInterpolationOp::Lerp,
                    VisualInterpolationOp::SmoothStep,
                    VisualInterpolationOp::InverseLerp,
                    VisualInterpolationOp::Vec3Lerp,
                    VisualInterpolationOp::QuatSlerp,
                ] {
                    let title = op.title();
                    add_node_button!(
                        ui,
                        title,
                        &[title, "math", "interpolation", "lerp", "slerp"],
                        VisualScriptNodeKind::MathInterpolation { op }
                    );
                }
            });
            ui.menu_button("Vector", |ui| {
                for op in [
                    VisualVectorMathOp::Dot,
                    VisualVectorMathOp::Cross,
                    VisualVectorMathOp::Length,
                    VisualVectorMathOp::Normalize,
                    VisualVectorMathOp::Distance,
                ] {
                    let title = op.title();
                    add_node_button!(
                        ui,
                        title,
                        &[title, "math", "vector"],
                        VisualScriptNodeKind::MathVector { op }
                    );
                }
            });
            ui.menu_button("Procedural", |ui| {
                for op in [
                    VisualProceduralMathOp::Clamp,
                    VisualProceduralMathOp::Remap,
                    VisualProceduralMathOp::Saturate,
                    VisualProceduralMathOp::Fract,
                ] {
                    let title = op.title();
                    add_node_button!(
                        ui,
                        title,
                        &[title, "math", "procedural"],
                        VisualScriptNodeKind::MathProcedural { op }
                    );
                }
            });
            ui.menu_button("Utility", |ui| {
                for op in [
                    VisualUtilityMathOp::Abs,
                    VisualUtilityMathOp::Sign,
                    VisualUtilityMathOp::Floor,
                    VisualUtilityMathOp::Ceil,
                    VisualUtilityMathOp::Round,
                    VisualUtilityMathOp::Sqrt,
                    VisualUtilityMathOp::Pow,
                    VisualUtilityMathOp::Exp,
                    VisualUtilityMathOp::Log,
                    VisualUtilityMathOp::Degrees,
                    VisualUtilityMathOp::Radians,
                ] {
                    let title = op.title();
                    add_node_button!(
                        ui,
                        title,
                        &[title, "math", "utility"],
                        VisualScriptNodeKind::MathUtility { op }
                    );
                }
            });
        });

        ui.menu_button("Logic", |ui| {
            add_node_button!(
                ui,
                "Compare",
                &["compare", "bool"],
                VisualScriptNodeKind::Compare {
                    op: VisualCompareOp::Equals,
                }
            );
            add_node_button!(
                ui,
                "Logical",
                &["logical", "bool"],
                VisualScriptNodeKind::LogicalBinary {
                    op: VisualLogicalOp::And,
                }
            );
            add_node_button!(ui, "Not", &["not", "bool"], VisualScriptNodeKind::Not);
            add_node_button!(
                ui,
                "Select",
                &["select", "bool"],
                VisualScriptNodeKind::Select {
                    value_type: default_select_value_type(),
                }
            );
        });

        ui.menu_button("Arrays", |ui| {
            add_node_button!(
                ui,
                "Array Empty",
                &["array", "empty", "create"],
                VisualScriptNodeKind::ArrayEmpty {
                    item_type: default_array_item_type(),
                }
            );
            add_node_button!(
                ui,
                "Array Length",
                &["array", "length", "count"],
                VisualScriptNodeKind::ArrayLength {
                    item_type: default_array_item_type(),
                }
            );
            add_node_button!(
                ui,
                "Array Get",
                &["array", "get", "index"],
                VisualScriptNodeKind::ArrayGet {
                    item_type: default_array_item_type(),
                }
            );
            add_node_button!(
                ui,
                "Array Set",
                &["array", "set", "index"],
                VisualScriptNodeKind::ArraySet {
                    item_type: default_array_item_type(),
                }
            );
            add_node_button!(
                ui,
                "Array Push",
                &["array", "push", "append"],
                VisualScriptNodeKind::ArrayPush {
                    item_type: default_array_item_type(),
                }
            );
            add_node_button!(
                ui,
                "Array Remove At",
                &["array", "remove", "index"],
                VisualScriptNodeKind::ArrayRemoveAt {
                    item_type: default_array_item_type(),
                }
            );
            add_node_button!(
                ui,
                "Array Clear",
                &["array", "clear"],
                VisualScriptNodeKind::ArrayClear {
                    item_type: default_array_item_type(),
                }
            );
        });

        ui.menu_button("Physics Queries", |ui| {
            add_node_button!(
                ui,
                "Ray Cast",
                &["ray cast", "ray", "physics query"],
                VisualScriptNodeKind::RayCast
            );
            add_node_button!(
                ui,
                "Ray Cast Hit",
                &["ray cast", "ray", "hit", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsRayCast.spec())
            );
            add_node_button!(
                ui,
                "Ray Cast Has Hit",
                &["ray cast", "ray", "has hit", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsRayCastHasHit.spec())
            );
            add_node_button!(
                ui,
                "Ray Cast Hit Entity",
                &["ray cast", "ray", "entity", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsRayCastHitEntity.spec())
            );
            add_node_button!(
                ui,
                "Ray Cast Point",
                &["ray cast", "ray", "point", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsRayCastPoint.spec())
            );
            add_node_button!(
                ui,
                "Ray Cast Normal",
                &["ray cast", "ray", "normal", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsRayCastNormal.spec())
            );
            add_node_button!(
                ui,
                "Ray Cast TOI",
                &["ray cast", "ray", "toi", "distance", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsRayCastToi.spec())
            );
            add_node_button!(
                ui,
                "Sphere Cast Hit",
                &["sphere cast", "shape cast", "hit", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsSphereCast.spec())
            );
            add_node_button!(
                ui,
                "Sphere Cast Has Hit",
                &["sphere cast", "shape cast", "has hit", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsSphereCastHasHit.spec())
            );
            add_node_button!(
                ui,
                "Sphere Cast Hit Entity",
                &["sphere cast", "shape cast", "entity", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsSphereCastHitEntity.spec())
            );
            add_node_button!(
                ui,
                "Sphere Cast Point",
                &["sphere cast", "shape cast", "point", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsSphereCastPoint.spec())
            );
            add_node_button!(
                ui,
                "Sphere Cast Normal",
                &["sphere cast", "shape cast", "normal", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsSphereCastNormal.spec())
            );
            add_node_button!(
                ui,
                "Sphere Cast TOI",
                &["sphere cast", "shape cast", "toi", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsSphereCastToi.spec())
            );
            add_node_button!(
                ui,
                "Get Physics Ray Cast Hit",
                &["get physics ray cast hit", "ray cast hit", "physics query"],
                api_node_kind_from_spec(VisualApiOperation::EcsGetPhysicsRayCastHit.spec())
            );
            add_node_button!(
                ui,
                "Get Physics Shape Cast Hit",
                &[
                    "get physics shape cast hit",
                    "sphere cast hit",
                    "physics query"
                ],
                api_node_kind_from_spec(VisualApiOperation::EcsGetPhysicsShapeCastHit.spec())
            );
            add_node_button!(
                ui,
                "Get Physics Point Projection Hit",
                &[
                    "get physics point projection hit",
                    "point projection",
                    "physics query"
                ],
                api_node_kind_from_spec(VisualApiOperation::EcsGetPhysicsPointProjectionHit.spec())
            );
            add_node_button!(
                ui,
                "Physics Query Filter",
                &[
                    "physics query filter",
                    "query filter",
                    "ray filter",
                    "physics query"
                ],
                VisualScriptNodeKind::PhysicsQueryFilterLiteral {
                    value: default_literal_for_type(VisualValueType::PhysicsQueryFilter)
                        .to_string(),
                }
            );
        });

        ui.menu_button("Structured Values", |ui| {
            ui.menu_button("Vec2", |ui| {
                add_node_button!(
                    ui,
                    "Vec2",
                    &["vec2", "vector", "structured"],
                    VisualScriptNodeKind::Vec2
                );
                add_node_button!(
                    ui,
                    "Vec2 Get Component",
                    &["vec2", "component", "extract"],
                    VisualScriptNodeKind::Vec2GetComponent {
                        component: VisualVec2Component::X,
                    }
                );
                add_node_button!(
                    ui,
                    "Vec2 Set Component",
                    &["vec2", "component", "set", "mutate"],
                    VisualScriptNodeKind::Vec2SetComponent {
                        component: VisualVec2Component::X,
                    }
                );
            });
            ui.menu_button("Vec3", |ui| {
                add_node_button!(
                    ui,
                    "Vec3",
                    &["vec3", "vector", "structured"],
                    VisualScriptNodeKind::Vec3
                );
                add_node_button!(
                    ui,
                    "Vec3 Get Component",
                    &["vec3", "component", "extract"],
                    VisualScriptNodeKind::Vec3GetComponent {
                        component: VisualVec3Component::X,
                    }
                );
                add_node_button!(
                    ui,
                    "Vec3 Set Component",
                    &["vec3", "component", "set", "mutate"],
                    VisualScriptNodeKind::Vec3SetComponent {
                        component: VisualVec3Component::X,
                    }
                );
            });
            ui.menu_button("Quat", |ui| {
                add_node_button!(
                    ui,
                    "Quat",
                    &["quat", "rotation", "structured"],
                    VisualScriptNodeKind::Quat
                );
                add_node_button!(
                    ui,
                    "Quat Get Component",
                    &["quat", "component", "extract"],
                    VisualScriptNodeKind::QuatGetComponent {
                        component: VisualQuatComponent::X,
                    }
                );
                add_node_button!(
                    ui,
                    "Quat Set Component",
                    &["quat", "component", "set", "mutate"],
                    VisualScriptNodeKind::QuatSetComponent {
                        component: VisualQuatComponent::X,
                    }
                );
            });
            ui.menu_button("Transform", |ui| {
                add_node_button!(
                    ui,
                    "Transform",
                    &["transform", "structured"],
                    VisualScriptNodeKind::Transform
                );
                add_node_button!(
                    ui,
                    "Transform Get Part",
                    &["transform", "extract", "part"],
                    VisualScriptNodeKind::TransformGetComponent {
                        component: VisualTransformComponent::Position,
                    }
                );
                add_node_button!(
                    ui,
                    "Transform Set Part",
                    &["transform", "set", "part", "mutate"],
                    VisualScriptNodeKind::TransformSetComponent {
                        component: VisualTransformComponent::Position,
                    }
                );
            });
            ui.menu_button("Physics Velocity", |ui| {
                add_node_button!(
                    ui,
                    "Physics Velocity",
                    &["physics velocity", "physics", "velocity", "structured"],
                    VisualScriptNodeKind::PhysicsVelocity
                );
                add_node_button!(
                    ui,
                    "Physics Velocity Get Field",
                    &["physics velocity", "physics", "velocity", "extract", "get"],
                    VisualScriptNodeKind::PhysicsVelocityGetComponent {
                        component: VisualPhysicsVelocityComponent::Linear,
                    }
                );
                add_node_button!(
                    ui,
                    "Physics Velocity Set Field",
                    &["physics velocity", "physics", "velocity", "set", "mutate"],
                    VisualScriptNodeKind::PhysicsVelocitySetComponent {
                        component: VisualPhysicsVelocityComponent::Linear,
                    }
                );
            });
        });

        ui.menu_button("Other", |ui| {
            add_node_button!(
                ui,
                "Comment",
                &["comment", "notes"],
                VisualScriptNodeKind::Comment {
                    text: "notes".to_string(),
                }
            );
            add_node_button!(
                ui,
                "Legacy Statement",
                &["legacy", "statement"],
                VisualScriptNodeKind::Statement {
                    code: "-- legacy".to_string(),
                }
            );
        });

        if has_search && !has_visible {
            ui.small("No nodes match the current search");
        }

        if inserted.is_some() {
            self.mark_changed();
        }

        inserted
    }
}

impl SnarlViewer<VisualScriptNodeKind> for VisualScriptViewer {
    fn title(&mut self, node: &VisualScriptNodeKind) -> String {
        match node {
            VisualScriptNodeKind::SetVariable {
                variable_id, name, ..
            } => {
                let variable = find_variable_definition(&self.variables, *variable_id, name)
                    .map(|var| var.name.as_str())
                    .unwrap_or_else(|| name.as_str());
                format!("Set {}", variable)
            }
            VisualScriptNodeKind::GetVariable {
                variable_id, name, ..
            } => {
                let variable = find_variable_definition(&self.variables, *variable_id, name)
                    .map(|var| var.name.as_str())
                    .unwrap_or_else(|| name.as_str());
                format!("Get {}", variable)
            }
            VisualScriptNodeKind::ClearVariable {
                variable_id, name, ..
            } => {
                let variable = find_variable_definition(&self.variables, *variable_id, name)
                    .map(|var| var.name.as_str())
                    .unwrap_or_else(|| name.as_str());
                format!("Clear {}", variable)
            }
            _ => node.title(),
        }
    }

    fn show_header(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<VisualScriptNodeKind>,
    ) {
        let title = snarl
            .get_node(node)
            .map(|entry| self.title(entry))
            .unwrap_or_else(|| "Node".to_string());
        let response = ui.add(egui::Label::new(title).sense(Sense::click()));
        if response.double_clicked() {
            if let Some(VisualScriptNodeKind::CallFunction { function_id, .. }) =
                snarl.get_node(node)
            {
                if *function_id != 0 {
                    self.open_function_graph_request = Some(*function_id);
                }
            }
        }
    }

    fn inputs(&mut self, node: &VisualScriptNodeKind) -> usize {
        node.input_count()
    }

    #[allow(refining_impl_trait)]
    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut Ui,
        snarl: &mut Snarl<VisualScriptNodeKind>,
    ) -> PinInfo {
        let Some((
            slot,
            label,
            color,
            value_type,
            allow_inline_default,
            asset_path_kind,
            api_operation,
        )) = snarl.get_node(pin.id.node).and_then(|node| {
            let slot = node.input_slot(pin.id.input)?;
            let value_type = if matches!(slot.kind, PinKind::Data) {
                node_data_input_type(node, slot.index, &self.variables)
            } else {
                None
            };
            let label = if matches!(slot.kind, PinKind::Data) {
                with_data_type_suffix(node.input_label(slot), value_type)
            } else {
                node.input_label(slot)
            };
            let allow_inline_default = matches!(
                node,
                VisualScriptNodeKind::CallApi { .. } | VisualScriptNodeKind::QueryApi { .. }
            ) && matches!(slot.kind, PinKind::Data);
            let (asset_path_kind, api_operation) = match node {
                VisualScriptNodeKind::CallApi { operation, .. }
                | VisualScriptNodeKind::QueryApi { operation, .. } => (
                    api_input_asset_path_kind(*operation, slot.index),
                    Some(*operation),
                ),
                _ => (None, None),
            };
            Some((
                slot,
                label,
                node.pin_color(slot, false),
                value_type,
                allow_inline_default,
                asset_path_kind,
                api_operation,
            ))
        })
        else {
            return PinInfo::square().with_fill(PIN_COLOR_EXEC);
        };

        let has_structured_default = api_operation
            .map(|operation| has_structured_api_default_editor(operation, slot.index))
            .unwrap_or(false);
        let prefers_vertical_layout = api_operation
            .map(|operation| api_input_prefers_vertical_default_layout(operation, slot.index))
            .unwrap_or(false);
        let value_prefers_vertical_layout = value_type
            .map(value_type_prefers_vertical_default_layout)
            .unwrap_or(false);
        let render_default =
            |ui: &mut Ui,
             this: &mut VisualScriptViewer,
             snarl: &mut Snarl<VisualScriptNodeKind>| {
                if !allow_inline_default || !pin.remotes.is_empty() {
                    return;
                }
                let Some(node) = snarl.get_node_mut(pin.id.node) else {
                    return;
                };
                let Some(default_literal) = api_input_default_literal_mut(node, slot.index) else {
                    return;
                };

                ui.add_space(4.0);
                let input_type = value_type.unwrap_or(VisualValueType::Any);
                let mut changed = false;
                let mut handled_by_structured = false;

                if let Some(operation) = api_operation {
                    if has_structured_api_default_editor(operation, slot.index) {
                        handled_by_structured = true;
                        let project_root = this.project_root.clone();
                        changed |= draw_api_structured_default_editor(
                            ui,
                            operation,
                            slot.index,
                            default_literal,
                            &mut this.pending_asset_drop_nodes,
                            &mut this.consumed_asset_drop,
                            project_root.as_deref(),
                        );
                    }
                }

                if !handled_by_structured {
                    if input_type == VisualValueType::String && asset_path_kind.is_some() {
                        let response =
                            ui.add(TextEdit::singleline(default_literal).desired_width(112.0));
                        if response.changed() {
                            changed = true;
                        }
                        if let Some(path_kind) = asset_path_kind {
                            let project_root = this.project_root.clone();
                            changed |= handle_asset_path_drop(
                                ui,
                                &response,
                                path_kind,
                                default_literal,
                                &mut this.pending_asset_drop_nodes,
                                &mut this.consumed_asset_drop,
                                project_root.as_deref(),
                            );
                        }
                    } else if draw_typed_pin_input_editor(ui, input_type, default_literal) {
                        changed = true;
                    }
                }

                if changed {
                    this.mark_changed();
                }
            };

        if allow_inline_default
            && pin.remotes.is_empty()
            && ((has_structured_default && prefers_vertical_layout)
                || value_prefers_vertical_layout)
        {
            ui.vertical(|ui| {
                ui.small(label.as_str());
                render_default(ui, self, snarl);
            });
        } else {
            ui.horizontal(|ui| {
                ui.small(label.as_str());
                render_default(ui, self, snarl);
            });
        }

        PinInfo::square().with_fill(color)
    }

    fn outputs(&mut self, node: &VisualScriptNodeKind) -> usize {
        node.output_count()
    }

    #[allow(refining_impl_trait)]
    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut Ui,
        snarl: &mut Snarl<VisualScriptNodeKind>,
    ) -> PinInfo {
        if let Some(node) = snarl.get_node(pin.id.node) {
            if let Some(slot) = node.output_slot(pin.id.output) {
                let label = if matches!(slot.kind, PinKind::Data) {
                    with_data_type_suffix(
                        node.output_label(slot),
                        node_data_output_type(node, slot.index, &self.variables),
                    )
                } else {
                    node.output_label(slot)
                };
                ui.small(label);
                return PinInfo::square().with_fill(node.pin_color(slot, true));
            }
        }
        PinInfo::square().with_fill(PIN_COLOR_EXEC)
    }

    fn has_body(&mut self, node: &VisualScriptNodeKind) -> bool {
        node.requires_body()
    }

    fn show_body(
        &mut self,
        node_id: NodeId,
        inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<VisualScriptNodeKind>,
    ) {
        let mut prune_wires = false;
        {
            let Some(node) = snarl.get_node_mut(node_id) else {
                return;
            };

            match node {
                VisualScriptNodeKind::OnStart
                | VisualScriptNodeKind::OnUpdate
                | VisualScriptNodeKind::OnStop
                | VisualScriptNodeKind::OnCollisionEnter
                | VisualScriptNodeKind::OnCollisionStay
                | VisualScriptNodeKind::OnCollisionExit
                | VisualScriptNodeKind::OnTriggerEnter
                | VisualScriptNodeKind::OnTriggerExit
                | VisualScriptNodeKind::SelfEntity
                | VisualScriptNodeKind::DeltaTime
                | VisualScriptNodeKind::TimeSinceStart
                | VisualScriptNodeKind::UnixTimeSeconds
                | VisualScriptNodeKind::Not
                | VisualScriptNodeKind::Vec2
                | VisualScriptNodeKind::Vec3
                | VisualScriptNodeKind::Quat
                | VisualScriptNodeKind::Transform
                | VisualScriptNodeKind::PhysicsVelocity
                | VisualScriptNodeKind::RayCast => {}
                VisualScriptNodeKind::OnInputAction { action, phase } => {
                    ui.horizontal(|ui| {
                        ui.label("Action");
                        let mut value = action.clone();
                        if ui.text_edit_singleline(&mut value).changed() {
                            let normalized = normalize_event_name(&value);
                            *action = if normalized.is_empty() {
                                default_input_action_name()
                            } else {
                                normalized
                            };
                            self.mark_changed();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Phase");
                        ComboBox::from_id_salt(("visual_input_action_phase", node_id.0))
                            .selected_text(phase.title())
                            .show_ui(ui, |ui| {
                                for candidate in [
                                    VisualInputActionPhase::Pressed,
                                    VisualInputActionPhase::Released,
                                    VisualInputActionPhase::Down,
                                ] {
                                    if ui
                                        .selectable_value(phase, candidate, candidate.title())
                                        .changed()
                                    {
                                        self.mark_changed();
                                    }
                                }
                            });
                    });
                }
                VisualScriptNodeKind::OnCustomEvent { name } => {
                    ui.horizontal(|ui| {
                        ui.label("Event");
                        let mut value = name.clone();
                        if ui.text_edit_singleline(&mut value).changed() {
                            let normalized = normalize_event_name(&value);
                            *name = if normalized.is_empty() {
                                default_custom_event_name()
                            } else {
                                normalized
                            };
                            self.mark_changed();
                        }
                    });
                }
                VisualScriptNodeKind::Sequence { outputs } => {
                    let mut value = i32::from(*outputs);
                    ui.horizontal(|ui| {
                        ui.label("Outputs");
                        let response = ui.add(DragValue::new(&mut value).range(1..=8));
                        if response.changed() {
                            *outputs = value.clamp(1, 8) as u8;
                            self.mark_changed();
                        }
                    });
                }
                VisualScriptNodeKind::Branch { condition } => {
                    if node_input_is_disconnected(inputs, 1) {
                        ui.label("Default condition when unplugged");
                        if ui.text_edit_singleline(condition).changed() {
                            self.mark_changed();
                        }
                    }
                }
                VisualScriptNodeKind::LoopWhile {
                    condition,
                    max_iterations,
                } => {
                    if node_input_is_disconnected(inputs, 1) {
                        ui.label("Default condition when unplugged");
                        if ui.text_edit_singleline(condition).changed() {
                            self.mark_changed();
                        }
                    }
                    let mut value = *max_iterations;
                    ui.horizontal(|ui| {
                        ui.label("Max iterations");
                        let response =
                            ui.add(DragValue::new(&mut value).range(1..=MAX_LOOP_ITERATIONS));
                        if response.changed() {
                            *max_iterations = value.clamp(1, MAX_LOOP_ITERATIONS);
                            self.mark_changed();
                        }
                    });
                }
                VisualScriptNodeKind::Log { message } => {
                    if node_input_is_disconnected(inputs, 1) {
                        if ui
                            .add(TextEdit::singleline(message).hint_text("Default"))
                            .changed()
                        {
                            self.mark_changed();
                        }
                    }
                }
                VisualScriptNodeKind::SetVariable {
                    variable_id,
                    name,
                    value,
                } => {
                    if draw_variable_binding_selector(
                        ui,
                        ("set_variable_picker", node_id.0),
                        variable_id,
                        name,
                        &self.variables,
                    ) {
                        if let Some(variable) =
                            find_variable_definition(&self.variables, *variable_id, name)
                        {
                            *value = normalize_literal_for_data_type(
                                value,
                                variable.value_type,
                                variable.array_item_type,
                            );
                        }
                        self.mark_changed();
                        prune_wires = true;
                    }
                    if node_input_is_disconnected(inputs, 1) {
                        ui.horizontal(|ui| {
                            ui.label("Default when unplugged");
                            let (value_type, array_item_type) =
                                find_variable_definition(&self.variables, *variable_id, name)
                                    .map(|var| (var.value_type, var.array_item_type))
                                    .unwrap_or((VisualValueType::Any, None));
                            if draw_typed_default_editor_with_array_item(
                                ui,
                                value_type,
                                array_item_type,
                                value,
                            ) {
                                self.mark_changed();
                            }
                        });
                    }
                }
                VisualScriptNodeKind::GetVariable {
                    variable_id,
                    name,
                    default_value,
                } => {
                    if draw_variable_binding_selector(
                        ui,
                        ("get_variable_picker", node_id.0),
                        variable_id,
                        name,
                        &self.variables,
                    ) {
                        if let Some(variable) =
                            find_variable_definition(&self.variables, *variable_id, name)
                        {
                            *default_value = normalize_literal_for_data_type(
                                default_value,
                                variable.value_type,
                                variable.array_item_type,
                            );
                        }
                        self.mark_changed();
                        prune_wires = true;
                    }
                    ui.horizontal(|ui| {
                        ui.label("Default");
                        let (value_type, array_item_type) =
                            find_variable_definition(&self.variables, *variable_id, name)
                                .map(|var| (var.value_type, var.array_item_type))
                                .unwrap_or((VisualValueType::Any, None));
                        if draw_typed_default_editor_with_array_item(
                            ui,
                            value_type,
                            array_item_type,
                            default_value,
                        ) {
                            self.mark_changed();
                        }
                    });
                }
                VisualScriptNodeKind::ClearVariable {
                    variable_id, name, ..
                } => {
                    if draw_variable_binding_selector(
                        ui,
                        ("clear_variable_picker", node_id.0),
                        variable_id,
                        name,
                        &self.variables,
                    ) {
                        self.mark_changed();
                    }
                }
                VisualScriptNodeKind::CallApi {
                    operation, args, ..
                } => {
                    let change = draw_api_node_body(ui, node_id, operation, args, true);
                    if change.changed {
                        self.mark_changed();
                    }
                    prune_wires |= change.operation_changed;
                }
                VisualScriptNodeKind::QueryApi {
                    operation, args, ..
                } => {
                    let change = draw_api_node_body(ui, node_id, operation, args, false);
                    if change.changed {
                        self.mark_changed();
                    }
                    prune_wires |= change.operation_changed;
                }
                VisualScriptNodeKind::BoolLiteral { value } => {
                    if ui.checkbox(value, "Value").changed() {
                        self.mark_changed();
                    }
                }
                VisualScriptNodeKind::NumberLiteral { value } => {
                    if ui.add(DragValue::new(value).speed(0.1)).changed() {
                        self.mark_changed();
                    }
                }
                VisualScriptNodeKind::StringLiteral { value } => {
                    if ui.text_edit_singleline(value).changed() {
                        self.mark_changed();
                    }
                }
                VisualScriptNodeKind::AnyLiteral { value } => {
                    ui.label("Any value or loose literal");
                    if ui
                        .add(
                            TextEdit::multiline(value)
                                .desired_rows(3)
                                .desired_width(220.0),
                        )
                        .changed()
                    {
                        self.mark_changed();
                    }
                }
                VisualScriptNodeKind::PhysicsQueryFilterLiteral { value } => {
                    if draw_typed_default_editor(ui, VisualValueType::PhysicsQueryFilter, value) {
                        self.mark_changed();
                    }
                }
                VisualScriptNodeKind::TimeSince { origin_seconds } => {
                    if node_input_is_disconnected(inputs, 0) {
                        ui.horizontal(|ui| {
                            ui.label("Origin (Unix seconds)");
                            if draw_typed_default_editor(
                                ui,
                                VisualValueType::Number,
                                origin_seconds,
                            ) {
                                self.mark_changed();
                            }
                        });
                    }
                }
                VisualScriptNodeKind::WaitSeconds {
                    seconds,
                    restart_on_retrigger,
                } => {
                    if node_input_is_disconnected(inputs, 1) {
                        ui.horizontal(|ui| {
                            ui.label("Delay (seconds)");
                            if draw_typed_default_editor(ui, VisualValueType::Number, seconds) {
                                self.mark_changed();
                            }
                        });
                    }
                    if ui
                        .checkbox(restart_on_retrigger, "Restart timer when retriggered")
                        .changed()
                    {
                        self.mark_changed();
                    }
                }
                VisualScriptNodeKind::FunctionStart { .. } => {
                    ui.small("Function entry point");
                }
                VisualScriptNodeKind::FunctionReturn {
                    outputs, values, ..
                } => {
                    for (index, output) in outputs.iter().enumerate() {
                        let disconnected = node_input_is_disconnected(inputs, index + 1);
                        if !disconnected {
                            continue;
                        }
                        while values.len() <= index {
                            values.push(default_literal_for_type(output.value_type).to_string());
                        }
                        ui.horizontal(|ui| {
                            ui.label(format!("{} default", output.name));
                            if draw_typed_default_editor_with_array_item(
                                ui,
                                output.value_type,
                                output.array_item_type,
                                &mut values[index],
                            ) {
                                self.mark_changed();
                            }
                        });
                    }
                }
                VisualScriptNodeKind::CallFunction {
                    function_id,
                    name,
                    inputs: input_ports,
                    outputs,
                    args,
                } => {
                    ui.horizontal(|ui| {
                        ui.label("Function ID");
                        let mut value = i64::try_from(*function_id).unwrap_or(0);
                        if ui
                            .add(DragValue::new(&mut value).range(0..=i64::MAX))
                            .changed()
                        {
                            *function_id = u64::try_from(value).unwrap_or(0);
                            self.mark_changed();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Function Name");
                        if ui.text_edit_singleline(name).changed() {
                            self.mark_changed();
                        }
                    });
                    if input_ports.len() != args.len() {
                        args.truncate(input_ports.len());
                        while args.len() < input_ports.len() {
                            let value_type = input_ports
                                .get(args.len())
                                .map(|port| port.value_type)
                                .unwrap_or(VisualValueType::Any);
                            args.push(default_literal_for_type(value_type).to_string());
                        }
                        self.mark_changed();
                    }
                    for (index, input) in input_ports.iter().enumerate() {
                        if node_input_is_disconnected(inputs, index + 1) {
                            ui.horizontal(|ui| {
                                ui.label(format!("{} default", input.name));
                                if let Some(arg) = args.get_mut(index) {
                                    if draw_typed_default_editor_with_array_item(
                                        ui,
                                        input.value_type,
                                        input.array_item_type,
                                        arg,
                                    ) {
                                        self.mark_changed();
                                    }
                                }
                            });
                        }
                    }
                    ui.small(format!("Returns {} value(s)", outputs.len()));
                }
                VisualScriptNodeKind::ArrayEmpty { item_type }
                | VisualScriptNodeKind::ArrayLength { item_type }
                | VisualScriptNodeKind::ArrayGet { item_type }
                | VisualScriptNodeKind::ArraySet { item_type }
                | VisualScriptNodeKind::ArrayPush { item_type }
                | VisualScriptNodeKind::ArrayRemoveAt { item_type }
                | VisualScriptNodeKind::ArrayClear { item_type } => {
                    let selected = *item_type;
                    ui.horizontal(|ui| {
                        ui.label("Item Type");
                        ComboBox::from_id_salt(("visual_array_item_type", node_id.0))
                            .selected_text(selected.title())
                            .show_ui(ui, |ui| {
                                for candidate in VISUAL_ARRAY_ITEM_TYPE_CHOICES {
                                    if ui
                                        .selectable_value(item_type, candidate, candidate.title())
                                        .changed()
                                    {
                                        if matches!(
                                            *item_type,
                                            VisualValueType::Array | VisualValueType::Any
                                        ) {
                                            *item_type = default_array_item_type();
                                        }
                                        prune_wires = true;
                                        self.mark_changed();
                                    }
                                }
                            });
                    });
                }
                VisualScriptNodeKind::MathBinary { op } => {
                    ComboBox::from_id_salt(("visual_math_op", node_id.0))
                        .selected_text(op.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualMathOp::Add,
                                VisualMathOp::Subtract,
                                VisualMathOp::Multiply,
                                VisualMathOp::Divide,
                                VisualMathOp::Modulo,
                                VisualMathOp::Min,
                                VisualMathOp::Max,
                            ] {
                                if ui
                                    .selectable_value(op, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::MathTrig { op } => {
                    ComboBox::from_id_salt(("visual_trig_op", node_id.0))
                        .selected_text(op.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualTrigOp::Sin,
                                VisualTrigOp::Cos,
                                VisualTrigOp::Tan,
                                VisualTrigOp::Asin,
                                VisualTrigOp::Acos,
                                VisualTrigOp::Atan,
                                VisualTrigOp::Atan2,
                            ] {
                                if ui
                                    .selectable_value(op, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::MathInterpolation { op } => {
                    ComboBox::from_id_salt(("visual_interpolation_op", node_id.0))
                        .selected_text(op.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualInterpolationOp::Lerp,
                                VisualInterpolationOp::SmoothStep,
                                VisualInterpolationOp::InverseLerp,
                                VisualInterpolationOp::Vec3Lerp,
                                VisualInterpolationOp::QuatSlerp,
                            ] {
                                if ui
                                    .selectable_value(op, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::MathVector { op } => {
                    ComboBox::from_id_salt(("visual_vector_math_op", node_id.0))
                        .selected_text(op.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualVectorMathOp::Dot,
                                VisualVectorMathOp::Cross,
                                VisualVectorMathOp::Length,
                                VisualVectorMathOp::Normalize,
                                VisualVectorMathOp::Distance,
                            ] {
                                if ui
                                    .selectable_value(op, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::MathProcedural { op } => {
                    ComboBox::from_id_salt(("visual_procedural_math_op", node_id.0))
                        .selected_text(op.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualProceduralMathOp::Clamp,
                                VisualProceduralMathOp::Remap,
                                VisualProceduralMathOp::Saturate,
                                VisualProceduralMathOp::Fract,
                            ] {
                                if ui
                                    .selectable_value(op, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::MathUtility { op } => {
                    ComboBox::from_id_salt(("visual_utility_math_op", node_id.0))
                        .selected_text(op.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualUtilityMathOp::Abs,
                                VisualUtilityMathOp::Sign,
                                VisualUtilityMathOp::Floor,
                                VisualUtilityMathOp::Ceil,
                                VisualUtilityMathOp::Round,
                                VisualUtilityMathOp::Sqrt,
                                VisualUtilityMathOp::Pow,
                                VisualUtilityMathOp::Exp,
                                VisualUtilityMathOp::Log,
                                VisualUtilityMathOp::Degrees,
                                VisualUtilityMathOp::Radians,
                            ] {
                                if ui
                                    .selectable_value(op, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::Compare { op } => {
                    ComboBox::from_id_salt(("visual_compare_op", node_id.0))
                        .selected_text(op.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualCompareOp::Equals,
                                VisualCompareOp::NotEquals,
                                VisualCompareOp::Less,
                                VisualCompareOp::LessOrEqual,
                                VisualCompareOp::Greater,
                                VisualCompareOp::GreaterOrEqual,
                            ] {
                                if ui
                                    .selectable_value(op, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::LogicalBinary { op } => {
                    ComboBox::from_id_salt(("visual_logical_op", node_id.0))
                        .selected_text(op.title())
                        .show_ui(ui, |ui| {
                            for candidate in [VisualLogicalOp::And, VisualLogicalOp::Or] {
                                if ui
                                    .selectable_value(op, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::Select { value_type } => {
                    ComboBox::from_id_salt(("visual_select_type", node_id.0))
                        .selected_text(value_type.title())
                        .show_ui(ui, |ui| {
                            for candidate in VISUAL_VALUE_TYPE_CHOICES_NO_ANY {
                                if ui
                                    .selectable_value(value_type, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::Vec2GetComponent { component } => {
                    ComboBox::from_id_salt(("visual_vec2_get_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [VisualVec2Component::X, VisualVec2Component::Y] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::Vec2SetComponent { component } => {
                    ComboBox::from_id_salt(("visual_vec2_set_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [VisualVec2Component::X, VisualVec2Component::Y] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::Vec3GetComponent { component } => {
                    ComboBox::from_id_salt(("visual_vec3_get_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualVec3Component::X,
                                VisualVec3Component::Y,
                                VisualVec3Component::Z,
                            ] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::Vec3SetComponent { component } => {
                    ComboBox::from_id_salt(("visual_vec3_set_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualVec3Component::X,
                                VisualVec3Component::Y,
                                VisualVec3Component::Z,
                            ] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::QuatGetComponent { component } => {
                    ComboBox::from_id_salt(("visual_quat_get_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualQuatComponent::X,
                                VisualQuatComponent::Y,
                                VisualQuatComponent::Z,
                                VisualQuatComponent::W,
                            ] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::QuatSetComponent { component } => {
                    ComboBox::from_id_salt(("visual_quat_set_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualQuatComponent::X,
                                VisualQuatComponent::Y,
                                VisualQuatComponent::Z,
                                VisualQuatComponent::W,
                            ] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                }
                            }
                        });
                }
                VisualScriptNodeKind::TransformGetComponent { component } => {
                    ComboBox::from_id_salt(("visual_transform_get_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualTransformComponent::Position,
                                VisualTransformComponent::Rotation,
                                VisualTransformComponent::Scale,
                            ] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::TransformSetComponent { component } => {
                    ComboBox::from_id_salt(("visual_transform_set_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualTransformComponent::Position,
                                VisualTransformComponent::Rotation,
                                VisualTransformComponent::Scale,
                            ] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::PhysicsVelocityGetComponent { component } => {
                    ComboBox::from_id_salt(("visual_physics_velocity_get_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualPhysicsVelocityComponent::Linear,
                                VisualPhysicsVelocityComponent::Angular,
                                VisualPhysicsVelocityComponent::WakeUp,
                            ] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::PhysicsVelocitySetComponent { component } => {
                    ComboBox::from_id_salt(("visual_physics_velocity_set_component", node_id.0))
                        .selected_text(component.title())
                        .show_ui(ui, |ui| {
                            for candidate in [
                                VisualPhysicsVelocityComponent::Linear,
                                VisualPhysicsVelocityComponent::Angular,
                                VisualPhysicsVelocityComponent::WakeUp,
                            ] {
                                if ui
                                    .selectable_value(component, candidate, candidate.title())
                                    .changed()
                                {
                                    self.mark_changed();
                                    prune_wires = true;
                                }
                            }
                        });
                }
                VisualScriptNodeKind::Comment { text } => {
                    if ui
                        .add(
                            TextEdit::multiline(text)
                                .desired_rows(4)
                                .desired_width(220.0),
                        )
                        .changed()
                    {
                        self.mark_changed();
                    }
                }
                VisualScriptNodeKind::Statement { code } => {
                    ui.colored_label(
                        Color32::from_rgb(214, 89, 89),
                        "Legacy node: script text execution is disabled",
                    );
                    if ui
                        .add(
                            TextEdit::multiline(code)
                                .desired_rows(4)
                                .desired_width(220.0)
                                .code_editor(),
                        )
                        .changed()
                    {
                        self.mark_changed();
                    }
                }
            }
        }

        if prune_wires && prune_invalid_wires_for_node(snarl, node_id, &self.variables) > 0 {
            self.mark_changed();
        }
    }

    fn has_graph_menu(
        &mut self,
        _pos: egui::Pos2,
        _snarl: &mut Snarl<VisualScriptNodeKind>,
    ) -> bool {
        true
    }

    fn show_graph_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut Ui,
        snarl: &mut Snarl<VisualScriptNodeKind>,
    ) {
        let _ = self.add_node_menu(pos, ui, snarl, None);
    }

    fn has_dropped_wire_menu(
        &mut self,
        _src_pins: AnyPins,
        _snarl: &mut Snarl<VisualScriptNodeKind>,
    ) -> bool {
        true
    }

    fn show_dropped_wire_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut Ui,
        src_pins: AnyPins,
        snarl: &mut Snarl<VisualScriptNodeKind>,
    ) {
        if let Some(new_node) = self.add_node_menu(pos, ui, snarl, Some(&src_pins)) {
            match src_pins {
                AnyPins::Out(outputs) => {
                    let new_node_inputs = snarl
                        .get_node(new_node)
                        .map(|node| node.input_count())
                        .unwrap_or(0);
                    if new_node_inputs > 0 {
                        let mut connected = false;
                        for output in outputs {
                            for input_index in 0..new_node_inputs {
                                let target = InPinId {
                                    node: new_node,
                                    input: input_index,
                                };
                                if can_connect_pins(snarl, *output, target, &self.variables) {
                                    if snarl.connect(*output, target) {
                                        connected = true;
                                    }
                                    break;
                                }
                            }
                        }
                        if connected {
                            self.mark_changed();
                        }
                    }
                }
                AnyPins::In(inputs) => {
                    let new_node_outputs = snarl
                        .get_node(new_node)
                        .map(|node| node.output_count())
                        .unwrap_or(0);
                    if new_node_outputs > 0 {
                        let mut connected = false;
                        for input in inputs {
                            for output_index in 0..new_node_outputs {
                                let source = OutPinId {
                                    node: new_node,
                                    output: output_index,
                                };
                                if can_connect_pins(snarl, source, *input, &self.variables) {
                                    if snarl.connect(source, *input) {
                                        connected = true;
                                    }
                                    break;
                                }
                            }
                        }
                        if connected {
                            self.mark_changed();
                        }
                    }
                }
            }
        }
    }

    fn has_node_menu(&mut self, _node: &VisualScriptNodeKind) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<VisualScriptNodeKind>,
    ) {
        if ui.button("Duplicate").clicked() {
            if let Some(info) = snarl.get_node_info(node) {
                let mut pos = info.pos;
                pos.x += 36.0;
                pos.y += 26.0;
                let duplicate = info.value.clone();
                snarl.insert_node(pos, duplicate);
                self.mark_changed();
                ui.close_kind(egui::UiKind::Menu);
            }
        }

        if ui.button("Delete").clicked() {
            snarl.remove_node(node);
            self.mark_changed();
            ui.close_kind(egui::UiKind::Menu);
        }
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<VisualScriptNodeKind>) {
        if !can_connect_pins(snarl, from.id, to.id, &self.variables) {
            return;
        }

        for remote in &to.remotes {
            snarl.disconnect(*remote, to.id);
        }
        if snarl.connect(from.id, to.id) {
            self.mark_changed();
        }
    }

    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<VisualScriptNodeKind>) {
        if snarl.disconnect(from.id, to.id) {
            self.mark_changed();
        }
    }

    fn drop_outputs(&mut self, pin: &OutPin, snarl: &mut Snarl<VisualScriptNodeKind>) {
        if snarl.drop_outputs(pin.id) > 0 {
            self.mark_changed();
        }
    }

    fn drop_inputs(&mut self, pin: &InPin, snarl: &mut Snarl<VisualScriptNodeKind>) {
        if snarl.drop_inputs(pin.id) > 0 {
            self.mark_changed();
        }
    }
}

fn can_connect_pins(
    snarl: &Snarl<VisualScriptNodeKind>,
    from: OutPinId,
    to: InPinId,
    variables: &[VisualVariableDefinition],
) -> bool {
    let Some(from_node) = snarl.get_node(from.node) else {
        return false;
    };
    let Some(to_node) = snarl.get_node(to.node) else {
        return false;
    };
    let Some(from_slot) = from_node.output_slot(from.output) else {
        return false;
    };
    let Some(to_slot) = to_node.input_slot(to.input) else {
        return false;
    };

    if from_slot.kind != to_slot.kind {
        return false;
    }

    if matches!(from_slot.kind, PinKind::Data) {
        let Some(from_type) = node_data_output_type(from_node, from_slot.index, variables) else {
            return false;
        };
        let Some(to_type) = node_data_input_type(to_node, to_slot.index, variables) else {
            return false;
        };
        if !are_data_types_compatible(from_type, to_type) {
            return false;
        }
    }

    true
}

fn node_input_is_disconnected(inputs: &[InPin], input_index: usize) -> bool {
    inputs
        .iter()
        .find(|pin| pin.id.input == input_index)
        .map(|pin| pin.remotes.is_empty())
        .unwrap_or(true)
}

fn prune_invalid_wires_for_node(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    node_id: NodeId,
    variables: &[VisualVariableDefinition],
) -> usize {
    prune_invalid_wires_with_filter(snarl, variables, Some(node_id))
}

fn prune_invalid_wires(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    variables: &[VisualVariableDefinition],
) -> usize {
    prune_invalid_wires_with_filter(snarl, variables, None)
}

fn prune_invalid_wires_with_filter(
    snarl: &mut Snarl<VisualScriptNodeKind>,
    variables: &[VisualVariableDefinition],
    node_filter: Option<NodeId>,
) -> usize {
    let mut invalid_wires = Vec::new();
    for (from, to) in snarl.wires() {
        let relevant = node_filter
            .map(|node_id| from.node == node_id || to.node == node_id)
            .unwrap_or(true);
        if relevant && !can_connect_pins(snarl, from, to, variables) {
            invalid_wires.push((from, to));
        }
    }

    let mut removed = 0usize;
    for (from, to) in invalid_wires {
        if snarl.disconnect(from, to) {
            removed += 1;
        }
    }

    removed
}

fn api_input_default_literal_mut(
    node: &mut VisualScriptNodeKind,
    data_input_index: usize,
) -> Option<&mut String> {
    match node {
        VisualScriptNodeKind::CallApi { args, .. }
        | VisualScriptNodeKind::QueryApi { args, .. } => args.get_mut(data_input_index),
        _ => None,
    }
}

fn draw_variable_binding_selector<T: std::hash::Hash>(
    ui: &mut Ui,
    id_salt: T,
    variable_id: &mut u64,
    name: &mut String,
    variables: &[VisualVariableDefinition],
) -> bool {
    let mut changed = false;

    if *variable_id == 0 {
        if let Some(found) = variables
            .iter()
            .find(|variable| variable.name == name.trim())
        {
            *variable_id = found.id;
            changed = true;
        }
    }

    if variables.is_empty() {
        ui.small("No variables are defined");
        return changed;
    }

    let selected = find_variable_definition(variables, *variable_id, name)
        .map(|variable| {
            if variable.value_type == VisualValueType::Array {
                let item = variable
                    .array_item_type
                    .unwrap_or(default_array_item_type());
                format!("{} (Array<{}>)", variable.name, item.title())
            } else {
                format!("{} ({})", variable.name, variable.value_type.title())
            }
        })
        .unwrap_or_else(|| "Select variable".to_string());

    ComboBox::from_id_salt(id_salt)
        .selected_text(selected)
        .show_ui(ui, |ui| {
            for variable in variables {
                let label = format!("{} ({})", variable.name, variable.value_type.title());
                if ui
                    .selectable_label(*variable_id == variable.id, label)
                    .clicked()
                {
                    *variable_id = variable.id;
                    *name = variable.name.clone();
                    changed = true;
                }
            }
        });

    changed
}

#[derive(Debug, Default, Clone, Copy)]
struct ApiNodeBodyChange {
    changed: bool,
    operation_changed: bool,
}

fn draw_api_node_body(
    ui: &mut Ui,
    node_id: NodeId,
    operation: &mut VisualApiOperation,
    args: &mut Vec<String>,
    show_exec_ops: bool,
) -> ApiNodeBodyChange {
    let mut change = ApiNodeBodyChange::default();

    ComboBox::from_id_salt(("visual_api_operation", node_id.0))
        .selected_text(operation.spec().title)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show_ui(ui, |ui| {
            for section in VISUAL_API_MENU_SECTION_ORDER {
                let mut specs: Vec<&VisualApiOperationSpec> = VISUAL_API_OPERATION_SPECS
                    .iter()
                    .filter(|spec| match spec.flow {
                        VisualApiFlow::Exec => show_exec_ops,
                        VisualApiFlow::Pure => !show_exec_ops,
                    })
                    .filter(|spec| api_menu_section(spec) == section)
                    .collect();
                if specs.is_empty() {
                    continue;
                }

                specs.sort_by_key(|spec| spec.title);
                ui.collapsing(section.title(), |ui| {
                    for spec in specs {
                        if ui
                            .selectable_value(operation, spec.operation, spec.title)
                            .changed()
                        {
                            change.changed = true;
                            change.operation_changed = true;
                        }
                    }
                });
            }
        });

    let spec = operation.spec();
    let expected_len = spec.inputs.len().min(MAX_API_ARGS);
    if args.len() != expected_len {
        args.resize_with(expected_len, || "null".to_string());
        for (index, arg) in args.iter_mut().enumerate() {
            if arg.trim().is_empty() {
                *arg =
                    default_literal_for_api_input(*operation, index, spec.inputs[index].value_type)
                        .to_string();
            }
        }
        change.changed = true;
    }

    change
}

fn parse_json_object_literal(value: &str) -> JsonMap<String, JsonValue> {
    match parse_loose_literal(value) {
        JsonValue::Object(object) => object,
        _ => JsonMap::new(),
    }
}

fn json_object_field<'a>(
    object: &'a JsonMap<String, JsonValue>,
    key: &str,
) -> Option<&'a JsonValue> {
    object.get(key)
}

fn json_object_string(object: &JsonMap<String, JsonValue>, key: &str, default: &str) -> String {
    json_object_field(object, key)
        .and_then(|value| match value {
            JsonValue::String(text) => Some(text.clone()),
            JsonValue::Null => Some(String::new()),
            _ => None,
        })
        .unwrap_or_else(|| default.to_string())
}

fn json_object_bool(object: &JsonMap<String, JsonValue>, key: &str, default: bool) -> bool {
    json_object_field(object, key)
        .map(is_truthy)
        .unwrap_or(default)
}

fn json_object_f64(object: &JsonMap<String, JsonValue>, key: &str, default: f64) -> f64 {
    json_object_field(object, key)
        .and_then(|value| coerce_json_to_f64(value).ok())
        .unwrap_or(default)
}

fn json_object_i64(object: &JsonMap<String, JsonValue>, key: &str, default: i64) -> i64 {
    json_object_field(object, key)
        .and_then(|value| coerce_json_to_f64(value).ok())
        .map(|value| value.round() as i64)
        .unwrap_or(default)
}

fn json_object_u64(
    object: &JsonMap<String, JsonValue>,
    key: &str,
    default: Option<u64>,
) -> Option<u64> {
    json_object_field(object, key)
        .and_then(json_to_u64)
        .or(default)
}

fn json_object_vec2(
    object: &JsonMap<String, JsonValue>,
    key: &str,
    default: (f64, f64),
) -> (f64, f64) {
    json_object_field(object, key)
        .and_then(|value| coerce_json_to_vec2_components(value).ok())
        .unwrap_or(default)
}

fn json_object_vec3(
    object: &JsonMap<String, JsonValue>,
    key: &str,
    default: (f64, f64, f64),
) -> (f64, f64, f64) {
    json_object_field(object, key)
        .and_then(|value| coerce_json_to_vec3_components(value).ok())
        .unwrap_or(default)
}

fn json_object_vec4(
    object: &JsonMap<String, JsonValue>,
    key: &str,
    default: (f64, f64, f64, f64),
) -> (f64, f64, f64, f64) {
    json_object_field(object, key)
        .and_then(|value| coerce_json_to_vec4_components(value).ok())
        .unwrap_or(default)
}

fn json_object_nested(
    object: &JsonMap<String, JsonValue>,
    key: &str,
) -> JsonMap<String, JsonValue> {
    json_object_field(object, key)
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default()
}

fn draw_vec3_row(ui: &mut Ui, label: &str, value: &mut (f64, f64, f64), speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui.add(DragValue::new(&mut value.0).speed(speed)).changed();
        changed |= ui.add(DragValue::new(&mut value.1).speed(speed)).changed();
        changed |= ui.add(DragValue::new(&mut value.2).speed(speed)).changed();
    });
    changed
}

fn draw_vec2_row(ui: &mut Ui, label: &str, value: &mut (f64, f64), speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui.add(DragValue::new(&mut value.0).speed(speed)).changed();
        changed |= ui.add(DragValue::new(&mut value.1).speed(speed)).changed();
    });
    changed
}

fn draw_vec4_row(ui: &mut Ui, label: &str, value: &mut (f64, f64, f64, f64), speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui.add(DragValue::new(&mut value.0).speed(speed)).changed();
        changed |= ui.add(DragValue::new(&mut value.1).speed(speed)).changed();
        changed |= ui.add(DragValue::new(&mut value.2).speed(speed)).changed();
        changed |= ui.add(DragValue::new(&mut value.3).speed(speed)).changed();
    });
    changed
}

fn handle_asset_path_drop(
    ui: &Ui,
    response: &egui::Response,
    kind: VisualAssetPathKind,
    path_literal: &mut String,
    pending_asset_nodes: &mut Vec<(String, egui::Pos2)>,
    consumed_asset_drop: &mut bool,
    project_root: Option<&Path>,
) -> bool {
    let Some(payload) = typed_dnd_release_payload::<AssetDragPayload>(response) else {
        return false;
    };

    if let Some(path) = payload
        .paths
        .iter()
        .find(|path| asset_path_matches_kind(path, kind))
    {
        *path_literal = path_to_visual_literal(path, project_root);
        *consumed_asset_drop = true;
        return true;
    }

    let drop_pos = response
        .interact_pointer_pos()
        .or_else(|| ui.ctx().input(|i| i.pointer.interact_pos()))
        .unwrap_or_else(|| ui.ctx().content_rect().center());
    for (index, path) in payload.paths.iter().enumerate() {
        pending_asset_nodes.push((
            path_to_visual_literal(path, project_root),
            egui::pos2(
                drop_pos.x + index as f32 * 22.0,
                drop_pos.y + index as f32 * 18.0,
            ),
        ));
    }
    false
}

fn draw_camera_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;

    let mut fov = json_object_f64(&object, "fov_y_rad", 1.0);
    let mut aspect = json_object_f64(&object, "aspect_ratio", 16.0 / 9.0);
    let mut near = json_object_f64(&object, "near_plane", 0.1);
    let mut far = json_object_f64(&object, "far_plane", 2000.0);
    let mut active = json_object_bool(&object, "active", false);

    ui.horizontal(|ui| {
        ui.label("FOV Y");
        changed |= ui.add(DragValue::new(&mut fov).speed(0.01)).changed();
        ui.label("Aspect");
        changed |= ui.add(DragValue::new(&mut aspect).speed(0.01)).changed();
    });
    ui.horizontal(|ui| {
        ui.label("Near");
        changed |= ui.add(DragValue::new(&mut near).speed(0.01)).changed();
        ui.label("Far");
        changed |= ui.add(DragValue::new(&mut far).speed(1.0)).changed();
    });
    changed |= ui.checkbox(&mut active, "Active camera").changed();

    if changed {
        object.insert("fov_y_rad".to_string(), json_number(fov));
        object.insert("aspect_ratio".to_string(), json_number(aspect));
        object.insert("near_plane".to_string(), json_number(near.max(0.0001)));
        object.insert(
            "far_plane".to_string(),
            json_number(far.max(near.max(0.0001))),
        );
        object.insert("active".to_string(), JsonValue::Bool(active));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_light_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;

    let mut light_type = json_object_string(&object, "type", "Point");
    if light_type.is_empty() {
        light_type = "Point".to_string();
    }
    let mut color = json_object_vec3(&object, "color", (1.0, 1.0, 1.0));
    let mut intensity = json_object_f64(&object, "intensity", 10.0);
    let mut angle = json_object_f64(&object, "angle", 45.0_f64.to_radians());

    ui.horizontal(|ui| {
        ui.label("Type");
        ComboBox::from_id_salt(("visual_light_type_inline", ui.id()))
            .selected_text(light_type.clone())
            .show_ui(ui, |ui| {
                for kind in ["Directional", "Point", "Spot"] {
                    changed |= ui
                        .selectable_value(&mut light_type, kind.to_string(), kind)
                        .changed();
                }
            });
    });
    changed |= draw_vec3_row(ui, "Color", &mut color, 0.01);
    ui.horizontal(|ui| {
        ui.label("Intensity");
        changed |= ui.add(DragValue::new(&mut intensity).speed(0.1)).changed();
    });
    if light_type.eq_ignore_ascii_case("spot") {
        ui.horizontal(|ui| {
            ui.label("Angle (rad)");
            changed |= ui.add(DragValue::new(&mut angle).speed(0.01)).changed();
        });
    }

    if changed {
        object.insert("type".to_string(), JsonValue::String(light_type));
        object.insert("color".to_string(), vec3_json(color.0, color.1, color.2));
        object.insert("intensity".to_string(), json_number(intensity.max(0.0)));
        if object
            .get("type")
            .and_then(JsonValue::as_str)
            .map(|kind| kind.eq_ignore_ascii_case("spot"))
            .unwrap_or(false)
        {
            object.insert("angle".to_string(), json_number(angle.max(0.0)));
        } else {
            object.remove("angle");
        }
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_mesh_renderer_data_editor(
    ui: &mut Ui,
    value: &mut String,
    pending_asset_nodes: &mut Vec<(String, egui::Pos2)>,
    consumed_asset_drop: &mut bool,
    project_root: Option<&Path>,
) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;

    let mut source = json_object_string(&object, "source", "Cube");
    let source_was_primitive = PrimitiveKind::from_source_label(&source).is_some();
    let source_mode_asset_initial = !source_was_primitive;
    let mut source_mode_asset = source_mode_asset_initial;
    let mut material = json_object_string(&object, "material", "");
    let mut casts_shadow = json_object_bool(&object, "casts_shadow", true);
    let mut visible = json_object_bool(&object, "visible", true);

    ui.horizontal(|ui| {
        ui.label("Source");
        changed |= ui
            .selectable_value(&mut source_mode_asset, false, "Primitive")
            .changed();
        changed |= ui
            .selectable_value(&mut source_mode_asset, true, "Asset")
            .changed();
    });

    if source_mode_asset != source_mode_asset_initial {
        if source_mode_asset {
            if source_was_primitive {
                source.clear();
                changed = true;
            }
        } else if PrimitiveKind::from_source_label(&source).is_none() {
            source = "Cube".to_string();
            changed = true;
        }
    }

    if source_mode_asset {
        ui.horizontal(|ui| {
            ui.label("Mesh Path");
            let response = ui.add(TextEdit::singleline(&mut source).desired_width(160.0));
            changed |= response.changed();
            changed |= handle_asset_path_drop(
                ui,
                &response,
                VisualAssetPathKind::Model,
                &mut source,
                pending_asset_nodes,
                consumed_asset_drop,
                project_root,
            );
        });
    } else {
        if PrimitiveKind::from_source_label(&source).is_none() {
            source = "Cube".to_string();
            changed = true;
        }
        ui.horizontal(|ui| {
            ui.label("Primitive");
            ComboBox::from_id_salt(("visual_mesh_source_primitive_inline", ui.id()))
                .selected_text(source.clone())
                .show_ui(ui, |ui| {
                    for primitive in VISUAL_MESH_PRIMITIVE_CHOICES {
                        changed |= ui
                            .selectable_value(&mut source, primitive.to_string(), primitive)
                            .changed();
                    }
                });
        });
    }

    ui.horizontal(|ui| {
        ui.label("Material");
        let response = ui.add(TextEdit::singleline(&mut material).desired_width(160.0));
        changed |= response.changed();
        changed |= handle_asset_path_drop(
            ui,
            &response,
            VisualAssetPathKind::Material,
            &mut material,
            pending_asset_nodes,
            consumed_asset_drop,
            project_root,
        );
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut casts_shadow, "Casts Shadow").changed();
        changed |= ui.checkbox(&mut visible, "Visible").changed();
    });

    if changed {
        object.insert("source".to_string(), JsonValue::String(source));
        if material.trim().is_empty() {
            object.remove("material");
        } else {
            object.insert("material".to_string(), JsonValue::String(material));
        }
        object.insert("casts_shadow".to_string(), JsonValue::Bool(casts_shadow));
        object.insert("visible".to_string(), JsonValue::Bool(visible));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_sprite_renderer_data_editor(
    ui: &mut Ui,
    value: &mut String,
    pending_asset_nodes: &mut Vec<(String, egui::Pos2)>,
    consumed_asset_drop: &mut bool,
    project_root: Option<&Path>,
) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;

    let mut color = json_object_vec4(&object, "color", (1.0, 1.0, 1.0, 1.0));
    let mut texture = json_object_string(&object, "texture", "");
    let mut uv_min = json_object_vec2(&object, "uv_min", (0.0, 0.0));
    let mut uv_max = json_object_vec2(&object, "uv_max", (1.0, 1.0));
    let mut pivot = json_object_vec2(&object, "pivot", (0.5, 0.5));
    let mut layer = json_object_f64(&object, "layer", 0.0);
    let mut space = json_object_string(&object, "space", "world");
    if !matches!(space.as_str(), "world" | "screen") {
        space = "world".to_string();
    }
    let mut blend_mode = json_object_string(&object, "blend_mode", "alpha");
    if !matches!(blend_mode.as_str(), "alpha" | "premultiplied" | "additive") {
        blend_mode = "alpha".to_string();
    }
    let mut billboard = json_object_bool(&object, "billboard", false);
    let mut visible = json_object_bool(&object, "visible", true);
    let mut pick_id = json_object_u64(&object, "pick_id", None);
    let mut has_clip_rect = object
        .get("clip_rect")
        .is_some_and(|clip_rect| !clip_rect.is_null());
    let mut clip_rect = json_object_vec4(&object, "clip_rect", (0.0, 0.0, 1.0, 1.0));

    let mut sheet = json_object_nested(&object, "sheet_animation");
    if sheet.is_empty() {
        sheet = json_object_nested(&object, "sheet");
    }
    let mut sheet_enabled = json_object_bool(&sheet, "enabled", false);
    let mut sheet_columns = json_object_f64(&sheet, "columns", 1.0).max(1.0);
    let mut sheet_rows = json_object_f64(&sheet, "rows", 1.0).max(1.0);
    let mut sheet_start_frame = json_object_f64(&sheet, "start_frame", 0.0).max(0.0);
    let mut sheet_frame_count = json_object_f64(&sheet, "frame_count", 0.0).max(0.0);
    let mut sheet_fps = json_object_f64(&sheet, "fps", 12.0).max(0.0);
    let mut sheet_playback = json_object_string(&sheet, "playback", "loop");
    if !matches!(sheet_playback.as_str(), "loop" | "once" | "pingpong") {
        sheet_playback = "loop".to_string();
    }
    let mut sheet_phase = json_object_f64(&sheet, "phase", 0.0);
    let mut sheet_paused = json_object_bool(&sheet, "paused", false);
    let mut sheet_paused_frame = json_object_f64(&sheet, "paused_frame", 0.0).max(0.0);
    let mut sheet_flip_x = json_object_bool(&sheet, "flip_x", false);
    let mut sheet_flip_y = json_object_bool(&sheet, "flip_y", false);
    let mut sheet_frame_uv_inset = json_object_vec2(&sheet, "frame_uv_inset", (0.0, 0.0));

    let mut sequence = json_object_nested(&object, "image_sequence");
    if sequence.is_empty() {
        sequence = json_object_nested(&object, "sequence");
    }
    let mut sequence_enabled = json_object_bool(&sequence, "enabled", false);
    let mut sequence_start_frame = json_object_f64(&sequence, "start_frame", 0.0).max(0.0);
    let mut sequence_frame_count = json_object_f64(&sequence, "frame_count", 0.0).max(0.0);
    let mut sequence_fps = json_object_f64(&sequence, "fps", 12.0).max(0.0);
    let mut sequence_playback = json_object_string(&sequence, "playback", "loop");
    if !matches!(sequence_playback.as_str(), "loop" | "once" | "pingpong") {
        sequence_playback = "loop".to_string();
    }
    let mut sequence_phase = json_object_f64(&sequence, "phase", 0.0);
    let mut sequence_paused = json_object_bool(&sequence, "paused", false);
    let mut sequence_paused_frame = json_object_f64(&sequence, "paused_frame", 0.0).max(0.0);
    let mut sequence_flip_x = json_object_bool(&sequence, "flip_x", false);
    let mut sequence_flip_y = json_object_bool(&sequence, "flip_y", false);
    let mut sequence_textures = Vec::new();
    for key in ["textures", "texture_paths", "frames"] {
        if let Some(value) = sequence.get(key) {
            if let Ok(paths) = coerce_json_to_sprite_texture_paths(value, key) {
                sequence_textures = paths;
            }
        }
    }

    let initial_color = color;
    let initial_texture = texture.clone();
    let initial_uv_min = uv_min;
    let initial_uv_max = uv_max;
    let initial_pivot = pivot;
    let initial_layer = layer;
    let initial_space = space.clone();
    let initial_blend_mode = blend_mode.clone();
    let initial_billboard = billboard;
    let initial_visible = visible;
    let initial_pick_id = pick_id;
    let initial_has_clip_rect = has_clip_rect;
    let initial_clip_rect = clip_rect;
    let initial_sheet_enabled = sheet_enabled;
    let initial_sheet_columns = sheet_columns;
    let initial_sheet_rows = sheet_rows;
    let initial_sheet_start_frame = sheet_start_frame;
    let initial_sheet_frame_count = sheet_frame_count;
    let initial_sheet_fps = sheet_fps;
    let initial_sheet_playback = sheet_playback.clone();
    let initial_sheet_phase = sheet_phase;
    let initial_sheet_paused = sheet_paused;
    let initial_sheet_paused_frame = sheet_paused_frame;
    let initial_sheet_flip_x = sheet_flip_x;
    let initial_sheet_flip_y = sheet_flip_y;
    let initial_sheet_frame_uv_inset = sheet_frame_uv_inset;
    let initial_sequence_enabled = sequence_enabled;
    let initial_sequence_start_frame = sequence_start_frame;
    let initial_sequence_frame_count = sequence_frame_count;
    let initial_sequence_fps = sequence_fps;
    let initial_sequence_playback = sequence_playback.clone();
    let initial_sequence_phase = sequence_phase;
    let initial_sequence_paused = sequence_paused;
    let initial_sequence_paused_frame = sequence_paused_frame;
    let initial_sequence_flip_x = sequence_flip_x;
    let initial_sequence_flip_y = sequence_flip_y;
    let initial_sequence_textures = sequence_textures.clone();

    changed |= draw_vec4_row(ui, "Color", &mut color, 0.01);
    ui.horizontal(|ui| {
        ui.label("Texture");
        let response = ui.add(TextEdit::singleline(&mut texture).desired_width(180.0));
        changed |= response.changed();
        changed |= handle_asset_path_drop(
            ui,
            &response,
            VisualAssetPathKind::Texture,
            &mut texture,
            pending_asset_nodes,
            consumed_asset_drop,
            project_root,
        );
    });
    changed |= draw_vec2_row(ui, "UV Min", &mut uv_min, 0.01);
    changed |= draw_vec2_row(ui, "UV Max", &mut uv_max, 0.01);
    changed |= draw_vec2_row(ui, "Pivot", &mut pivot, 0.01);
    ui.horizontal(|ui| {
        ui.label("Layer");
        changed |= ui.add(DragValue::new(&mut layer).speed(0.1)).changed();
        changed |= ui.checkbox(&mut billboard, "Billboard").changed();
        changed |= ui.checkbox(&mut visible, "Visible").changed();
    });
    ui.horizontal(|ui| {
        ui.label("Space");
        ComboBox::from_id_salt(("visual_sprite_space_inline", ui.id()))
            .selected_text(space.clone())
            .show_ui(ui, |ui| {
                for candidate in ["world", "screen"] {
                    changed |= ui
                        .selectable_value(&mut space, candidate.to_string(), candidate)
                        .changed();
                }
            });
        ui.label("Blend");
        ComboBox::from_id_salt(("visual_sprite_blend_mode_inline", ui.id()))
            .selected_text(blend_mode.clone())
            .show_ui(ui, |ui| {
                for candidate in ["alpha", "premultiplied", "additive"] {
                    changed |= ui
                        .selectable_value(&mut blend_mode, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        let mut has_pick_id = pick_id.is_some();
        if ui.checkbox(&mut has_pick_id, "Pick Id").changed() {
            pick_id = if has_pick_id {
                Some(pick_id.unwrap_or(0))
            } else {
                None
            };
            changed = true;
        }
        if has_pick_id {
            let mut value = pick_id.unwrap_or(0);
            if ui
                .add(DragValue::new(&mut value).speed(1.0).range(0..=u64::MAX))
                .changed()
            {
                pick_id = Some(value);
                changed = true;
            }
            if ui.small_button("Clear").clicked() {
                pick_id = None;
                changed = true;
            }
        }
    });
    ui.horizontal(|ui| {
        if ui.checkbox(&mut has_clip_rect, "Clip Rect").changed() {
            changed = true;
        }
    });
    if has_clip_rect {
        changed |= draw_vec4_row(ui, "Rect", &mut clip_rect, 0.01);
    }

    ui.group(|ui| {
        ui.label("Sheet Animation");
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut sheet_enabled, "Enabled").changed();
            ui.label("Playback");
            ComboBox::from_id_salt(("visual_sprite_sheet_playback_inline", ui.id()))
                .selected_text(sheet_playback.clone())
                .show_ui(ui, |ui| {
                    for candidate in ["loop", "once", "pingpong"] {
                        changed |= ui
                            .selectable_value(&mut sheet_playback, candidate.to_string(), candidate)
                            .changed();
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Cols");
            changed |= ui
                .add(DragValue::new(&mut sheet_columns).speed(1.0))
                .changed();
            ui.label("Rows");
            changed |= ui.add(DragValue::new(&mut sheet_rows).speed(1.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Start");
            changed |= ui
                .add(DragValue::new(&mut sheet_start_frame).speed(1.0))
                .changed();
            ui.label("Count");
            changed |= ui
                .add(DragValue::new(&mut sheet_frame_count).speed(1.0))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("FPS");
            changed |= ui.add(DragValue::new(&mut sheet_fps).speed(0.1)).changed();
            ui.label("Phase");
            changed |= ui
                .add(DragValue::new(&mut sheet_phase).speed(0.01))
                .changed();
        });
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut sheet_paused, "Paused").changed();
            ui.label("Paused Frame");
            changed |= ui
                .add(DragValue::new(&mut sheet_paused_frame).speed(1.0))
                .changed();
        });
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut sheet_flip_x, "Flip X").changed();
            changed |= ui.checkbox(&mut sheet_flip_y, "Flip Y").changed();
        });
        changed |= draw_vec2_row(ui, "UV Inset", &mut sheet_frame_uv_inset, 0.005);
    });

    ui.group(|ui| {
        ui.label("Image Sequence");
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut sequence_enabled, "Enabled").changed();
            ui.label("Playback");
            ComboBox::from_id_salt(("visual_sprite_sequence_playback_inline", ui.id()))
                .selected_text(sequence_playback.clone())
                .show_ui(ui, |ui| {
                    for candidate in ["loop", "once", "pingpong"] {
                        changed |= ui
                            .selectable_value(
                                &mut sequence_playback,
                                candidate.to_string(),
                                candidate,
                            )
                            .changed();
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Start");
            changed |= ui
                .add(DragValue::new(&mut sequence_start_frame).speed(1.0))
                .changed();
            ui.label("Count");
            changed |= ui
                .add(DragValue::new(&mut sequence_frame_count).speed(1.0))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("FPS");
            changed |= ui
                .add(DragValue::new(&mut sequence_fps).speed(0.1))
                .changed();
            ui.label("Phase");
            changed |= ui
                .add(DragValue::new(&mut sequence_phase).speed(0.01))
                .changed();
        });
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut sequence_paused, "Paused").changed();
            ui.label("Paused Frame");
            changed |= ui
                .add(DragValue::new(&mut sequence_paused_frame).speed(1.0))
                .changed();
            changed |= ui.checkbox(&mut sequence_flip_x, "Flip X").changed();
            changed |= ui.checkbox(&mut sequence_flip_y, "Flip Y").changed();
        });
        let mut remove_index = None;
        for (index, path) in sequence_textures.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let response = ui.add(TextEdit::singleline(path).desired_width(180.0));
                changed |= response.changed();
                changed |= handle_asset_path_drop(
                    ui,
                    &response,
                    VisualAssetPathKind::Texture,
                    path,
                    pending_asset_nodes,
                    consumed_asset_drop,
                    project_root,
                );
                if ui.small_button("-").clicked() {
                    remove_index = Some(index);
                }
            });
        }
        if let Some(index) = remove_index {
            if index < sequence_textures.len() {
                sequence_textures.remove(index);
                changed = true;
            }
        }
        if ui.small_button("+ Frame").clicked() {
            sequence_textures.push(String::new());
            changed = true;
        }
    });

    if changed {
        let normalized_texture = texture.trim().to_string();
        let initial_normalized_texture = initial_texture.trim().to_string();
        let base_changed = color != initial_color
            || normalized_texture != initial_normalized_texture
            || uv_min != initial_uv_min
            || uv_max != initial_uv_max
            || pivot != initial_pivot
            || layer != initial_layer
            || space != initial_space
            || blend_mode != initial_blend_mode
            || billboard != initial_billboard
            || visible != initial_visible
            || pick_id != initial_pick_id
            || has_clip_rect != initial_has_clip_rect
            || (has_clip_rect && clip_rect != initial_clip_rect);
        let sheet_changed = sheet_enabled != initial_sheet_enabled
            || sheet_columns != initial_sheet_columns
            || sheet_rows != initial_sheet_rows
            || sheet_start_frame != initial_sheet_start_frame
            || sheet_frame_count != initial_sheet_frame_count
            || sheet_fps != initial_sheet_fps
            || sheet_playback != initial_sheet_playback
            || sheet_phase != initial_sheet_phase
            || sheet_paused != initial_sheet_paused
            || sheet_paused_frame != initial_sheet_paused_frame
            || sheet_flip_x != initial_sheet_flip_x
            || sheet_flip_y != initial_sheet_flip_y
            || sheet_frame_uv_inset != initial_sheet_frame_uv_inset;
        let normalized_sequence_paths = sequence_textures
            .iter()
            .map(|path| path.trim().to_string())
            .collect::<Vec<_>>();
        let initial_normalized_sequence_paths = initial_sequence_textures
            .iter()
            .map(|path| path.trim().to_string())
            .collect::<Vec<_>>();
        let sequence_changed = sequence_enabled != initial_sequence_enabled
            || sequence_start_frame != initial_sequence_start_frame
            || sequence_frame_count != initial_sequence_frame_count
            || sequence_fps != initial_sequence_fps
            || sequence_playback != initial_sequence_playback
            || sequence_phase != initial_sequence_phase
            || sequence_paused != initial_sequence_paused
            || sequence_paused_frame != initial_sequence_paused_frame
            || sequence_flip_x != initial_sequence_flip_x
            || sequence_flip_y != initial_sequence_flip_y
            || normalized_sequence_paths != initial_normalized_sequence_paths;

        if base_changed {
            object.insert(
                "color".to_string(),
                vec4_json(color.0, color.1, color.2, color.3),
            );
            object.insert(
                "texture".to_string(),
                if normalized_texture.is_empty() {
                    JsonValue::Null
                } else {
                    JsonValue::String(normalized_texture)
                },
            );
            object.insert("uv_min".to_string(), vec2_json(uv_min.0, uv_min.1));
            object.insert("uv_max".to_string(), vec2_json(uv_max.0, uv_max.1));
            object.insert("pivot".to_string(), vec2_json(pivot.0, pivot.1));
            object.insert("layer".to_string(), json_number(layer));
            object.insert("space".to_string(), JsonValue::String(space));
            object.insert("blend_mode".to_string(), JsonValue::String(blend_mode));
            object.insert("billboard".to_string(), JsonValue::Bool(billboard));
            object.insert("visible".to_string(), JsonValue::Bool(visible));
            object.insert(
                "pick_id".to_string(),
                pick_id
                    .map(|value| JsonValue::Number(JsonNumber::from(value)))
                    .unwrap_or(JsonValue::Null),
            );
            object.insert(
                "clip_rect".to_string(),
                if has_clip_rect {
                    vec4_json(clip_rect.0, clip_rect.1, clip_rect.2, clip_rect.3)
                } else {
                    JsonValue::Null
                },
            );
        }

        if sheet_changed {
            let mut sheet_object = JsonMap::new();
            sheet_object.insert("enabled".to_string(), JsonValue::Bool(sheet_enabled));
            sheet_object.insert(
                "columns".to_string(),
                JsonValue::Number(JsonNumber::from(sheet_columns.max(1.0).round() as u64)),
            );
            sheet_object.insert(
                "rows".to_string(),
                JsonValue::Number(JsonNumber::from(sheet_rows.max(1.0).round() as u64)),
            );
            sheet_object.insert(
                "start_frame".to_string(),
                JsonValue::Number(JsonNumber::from(sheet_start_frame.max(0.0).round() as u64)),
            );
            sheet_object.insert(
                "frame_count".to_string(),
                JsonValue::Number(JsonNumber::from(sheet_frame_count.max(0.0).round() as u64)),
            );
            sheet_object.insert("fps".to_string(), json_number(sheet_fps.max(0.0)));
            sheet_object.insert(
                "playback".to_string(),
                JsonValue::String(sheet_playback.clone()),
            );
            sheet_object.insert("phase".to_string(), json_number(sheet_phase));
            sheet_object.insert("paused".to_string(), JsonValue::Bool(sheet_paused));
            sheet_object.insert(
                "paused_frame".to_string(),
                JsonValue::Number(JsonNumber::from(sheet_paused_frame.max(0.0).round() as u64)),
            );
            sheet_object.insert("flip_x".to_string(), JsonValue::Bool(sheet_flip_x));
            sheet_object.insert("flip_y".to_string(), JsonValue::Bool(sheet_flip_y));
            sheet_object.insert(
                "frame_uv_inset".to_string(),
                vec2_json(
                    sheet_frame_uv_inset.0.clamp(0.0, 0.49),
                    sheet_frame_uv_inset.1.clamp(0.0, 0.49),
                ),
            );
            let sheet_value = JsonValue::Object(sheet_object);
            object.insert("sheet_animation".to_string(), sheet_value.clone());
            object.insert("sheet".to_string(), sheet_value);
        }

        if sequence_changed {
            let sequence_paths = normalized_sequence_paths
                .iter()
                .map(|path| JsonValue::String(path.to_string()))
                .collect::<Vec<_>>();
            let mut sequence_object = JsonMap::new();
            sequence_object.insert("enabled".to_string(), JsonValue::Bool(sequence_enabled));
            sequence_object.insert(
                "start_frame".to_string(),
                JsonValue::Number(JsonNumber::from(
                    sequence_start_frame.max(0.0).round() as u64
                )),
            );
            sequence_object.insert(
                "frame_count".to_string(),
                JsonValue::Number(JsonNumber::from(
                    sequence_frame_count.max(0.0).round() as u64
                )),
            );
            sequence_object.insert("fps".to_string(), json_number(sequence_fps.max(0.0)));
            sequence_object.insert(
                "playback".to_string(),
                JsonValue::String(sequence_playback.clone()),
            );
            sequence_object.insert("phase".to_string(), json_number(sequence_phase));
            sequence_object.insert("paused".to_string(), JsonValue::Bool(sequence_paused));
            sequence_object.insert(
                "paused_frame".to_string(),
                JsonValue::Number(JsonNumber::from(
                    sequence_paused_frame.max(0.0).round() as u64
                )),
            );
            sequence_object.insert("flip_x".to_string(), JsonValue::Bool(sequence_flip_x));
            sequence_object.insert("flip_y".to_string(), JsonValue::Bool(sequence_flip_y));
            sequence_object.insert(
                "textures".to_string(),
                JsonValue::Array(sequence_paths.clone()),
            );
            sequence_object.insert(
                "texture_paths".to_string(),
                JsonValue::Array(sequence_paths.clone()),
            );
            sequence_object.insert("frames".to_string(), JsonValue::Array(sequence_paths));
            let sequence_value = JsonValue::Object(sequence_object);
            object.insert("image_sequence".to_string(), sequence_value.clone());
            object.insert("sequence".to_string(), sequence_value);
        }

        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_sprite_sheet_animation_patch_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let object = parse_json_object_literal(value);
    let mut changed = false;
    let mut sheet = json_object_nested(&object, "sheet_animation");
    if sheet.is_empty() {
        sheet = json_object_nested(&object, "sheet");
    }
    if sheet.is_empty() {
        sheet = object;
    }

    let mut enabled = json_object_bool(&sheet, "enabled", false);
    let mut columns = json_object_f64(&sheet, "columns", 1.0).max(1.0);
    let mut rows = json_object_f64(&sheet, "rows", 1.0).max(1.0);
    let mut start_frame = json_object_f64(&sheet, "start_frame", 0.0).max(0.0);
    let mut frame_count = json_object_f64(&sheet, "frame_count", 0.0).max(0.0);
    let mut fps = json_object_f64(&sheet, "fps", 12.0).max(0.0);
    let mut playback = json_object_string(&sheet, "playback", "loop");
    if !matches!(playback.as_str(), "loop" | "once" | "pingpong") {
        playback = "loop".to_string();
    }
    let mut phase = json_object_f64(&sheet, "phase", 0.0);
    let mut paused = json_object_bool(&sheet, "paused", false);
    let mut paused_frame = json_object_f64(&sheet, "paused_frame", 0.0).max(0.0);
    let mut flip_x = json_object_bool(&sheet, "flip_x", false);
    let mut flip_y = json_object_bool(&sheet, "flip_y", false);
    let mut frame_uv_inset = json_object_vec2(&sheet, "frame_uv_inset", (0.0, 0.0));

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut enabled, "Enabled").changed();
        ui.label("Playback");
        ComboBox::from_id_salt(("visual_sprite_sheet_patch_playback_inline", ui.id()))
            .selected_text(playback.clone())
            .show_ui(ui, |ui| {
                for candidate in ["loop", "once", "pingpong"] {
                    changed |= ui
                        .selectable_value(&mut playback, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Cols");
        changed |= ui.add(DragValue::new(&mut columns).speed(1.0)).changed();
        ui.label("Rows");
        changed |= ui.add(DragValue::new(&mut rows).speed(1.0)).changed();
    });
    ui.horizontal(|ui| {
        ui.label("Start");
        changed |= ui
            .add(DragValue::new(&mut start_frame).speed(1.0))
            .changed();
        ui.label("Count");
        changed |= ui
            .add(DragValue::new(&mut frame_count).speed(1.0))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("FPS");
        changed |= ui.add(DragValue::new(&mut fps).speed(0.1)).changed();
        ui.label("Phase");
        changed |= ui.add(DragValue::new(&mut phase).speed(0.01)).changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut paused, "Paused").changed();
        ui.label("Paused Frame");
        changed |= ui
            .add(DragValue::new(&mut paused_frame).speed(1.0))
            .changed();
        changed |= ui.checkbox(&mut flip_x, "Flip X").changed();
        changed |= ui.checkbox(&mut flip_y, "Flip Y").changed();
    });
    changed |= draw_vec2_row(ui, "UV Inset", &mut frame_uv_inset, 0.005);

    if changed {
        let mut sheet_object = JsonMap::new();
        sheet_object.insert("enabled".to_string(), JsonValue::Bool(enabled));
        sheet_object.insert(
            "columns".to_string(),
            JsonValue::Number(JsonNumber::from(columns.max(1.0).round() as u64)),
        );
        sheet_object.insert(
            "rows".to_string(),
            JsonValue::Number(JsonNumber::from(rows.max(1.0).round() as u64)),
        );
        sheet_object.insert(
            "start_frame".to_string(),
            JsonValue::Number(JsonNumber::from(start_frame.max(0.0).round() as u64)),
        );
        sheet_object.insert(
            "frame_count".to_string(),
            JsonValue::Number(JsonNumber::from(frame_count.max(0.0).round() as u64)),
        );
        sheet_object.insert("fps".to_string(), json_number(fps.max(0.0)));
        sheet_object.insert("playback".to_string(), JsonValue::String(playback));
        sheet_object.insert("phase".to_string(), json_number(phase));
        sheet_object.insert("paused".to_string(), JsonValue::Bool(paused));
        sheet_object.insert(
            "paused_frame".to_string(),
            JsonValue::Number(JsonNumber::from(paused_frame.max(0.0).round() as u64)),
        );
        sheet_object.insert("flip_x".to_string(), JsonValue::Bool(flip_x));
        sheet_object.insert("flip_y".to_string(), JsonValue::Bool(flip_y));
        sheet_object.insert(
            "frame_uv_inset".to_string(),
            vec2_json(
                frame_uv_inset.0.clamp(0.0, 0.49),
                frame_uv_inset.1.clamp(0.0, 0.49),
            ),
        );
        let sheet_value = JsonValue::Object(sheet_object);
        let mut out = JsonMap::new();
        out.insert("sheet_animation".to_string(), sheet_value.clone());
        out.insert("sheet".to_string(), sheet_value);
        *value = compact_json_string(&JsonValue::Object(out));
    }

    changed
}

fn draw_sprite_image_sequence_patch_data_editor(
    ui: &mut Ui,
    value: &mut String,
    pending_asset_nodes: &mut Vec<(String, egui::Pos2)>,
    consumed_asset_drop: &mut bool,
    project_root: Option<&Path>,
) -> bool {
    let object = parse_json_object_literal(value);
    let mut changed = false;
    let mut sequence = json_object_nested(&object, "image_sequence");
    if sequence.is_empty() {
        sequence = json_object_nested(&object, "sequence");
    }
    if sequence.is_empty() {
        sequence = object;
    }

    let mut enabled = json_object_bool(&sequence, "enabled", false);
    let mut start_frame = json_object_f64(&sequence, "start_frame", 0.0).max(0.0);
    let mut frame_count = json_object_f64(&sequence, "frame_count", 0.0).max(0.0);
    let mut fps = json_object_f64(&sequence, "fps", 12.0).max(0.0);
    let mut playback = json_object_string(&sequence, "playback", "loop");
    if !matches!(playback.as_str(), "loop" | "once" | "pingpong") {
        playback = "loop".to_string();
    }
    let mut phase = json_object_f64(&sequence, "phase", 0.0);
    let mut paused = json_object_bool(&sequence, "paused", false);
    let mut paused_frame = json_object_f64(&sequence, "paused_frame", 0.0).max(0.0);
    let mut flip_x = json_object_bool(&sequence, "flip_x", false);
    let mut flip_y = json_object_bool(&sequence, "flip_y", false);
    let mut textures = Vec::new();
    for key in ["textures", "texture_paths", "frames"] {
        if let Some(value) = sequence.get(key) {
            if let Ok(paths) = coerce_json_to_sprite_texture_paths(value, key) {
                textures = paths;
            }
        }
    }

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut enabled, "Enabled").changed();
        ui.label("Playback");
        ComboBox::from_id_salt(("visual_sprite_sequence_patch_playback_inline", ui.id()))
            .selected_text(playback.clone())
            .show_ui(ui, |ui| {
                for candidate in ["loop", "once", "pingpong"] {
                    changed |= ui
                        .selectable_value(&mut playback, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Start");
        changed |= ui
            .add(DragValue::new(&mut start_frame).speed(1.0))
            .changed();
        ui.label("Count");
        changed |= ui
            .add(DragValue::new(&mut frame_count).speed(1.0))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("FPS");
        changed |= ui.add(DragValue::new(&mut fps).speed(0.1)).changed();
        ui.label("Phase");
        changed |= ui.add(DragValue::new(&mut phase).speed(0.01)).changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut paused, "Paused").changed();
        ui.label("Paused Frame");
        changed |= ui
            .add(DragValue::new(&mut paused_frame).speed(1.0))
            .changed();
        changed |= ui.checkbox(&mut flip_x, "Flip X").changed();
        changed |= ui.checkbox(&mut flip_y, "Flip Y").changed();
    });
    let mut remove_index = None;
    for (index, path) in textures.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            let response = ui.add(TextEdit::singleline(path).desired_width(180.0));
            changed |= response.changed();
            changed |= handle_asset_path_drop(
                ui,
                &response,
                VisualAssetPathKind::Texture,
                path,
                pending_asset_nodes,
                consumed_asset_drop,
                project_root,
            );
            if ui.small_button("-").clicked() {
                remove_index = Some(index);
            }
        });
    }
    if let Some(index) = remove_index {
        if index < textures.len() {
            textures.remove(index);
            changed = true;
        }
    }
    if ui.small_button("+ Frame").clicked() {
        textures.push(String::new());
        changed = true;
    }

    if changed {
        let texture_values = JsonValue::Array(
            textures
                .iter()
                .map(|path| JsonValue::String(path.trim().to_string()))
                .collect(),
        );
        let mut sequence_object = JsonMap::new();
        sequence_object.insert("enabled".to_string(), JsonValue::Bool(enabled));
        sequence_object.insert(
            "start_frame".to_string(),
            JsonValue::Number(JsonNumber::from(start_frame.max(0.0).round() as u64)),
        );
        sequence_object.insert(
            "frame_count".to_string(),
            JsonValue::Number(JsonNumber::from(frame_count.max(0.0).round() as u64)),
        );
        sequence_object.insert("fps".to_string(), json_number(fps.max(0.0)));
        sequence_object.insert("playback".to_string(), JsonValue::String(playback));
        sequence_object.insert("phase".to_string(), json_number(phase));
        sequence_object.insert("paused".to_string(), JsonValue::Bool(paused));
        sequence_object.insert(
            "paused_frame".to_string(),
            JsonValue::Number(JsonNumber::from(paused_frame.max(0.0).round() as u64)),
        );
        sequence_object.insert("flip_x".to_string(), JsonValue::Bool(flip_x));
        sequence_object.insert("flip_y".to_string(), JsonValue::Bool(flip_y));
        sequence_object.insert("textures".to_string(), texture_values.clone());
        sequence_object.insert("texture_paths".to_string(), texture_values.clone());
        sequence_object.insert("frames".to_string(), texture_values);
        let sequence_value = JsonValue::Object(sequence_object);
        let mut out = JsonMap::new();
        out.insert("image_sequence".to_string(), sequence_value.clone());
        out.insert("sequence".to_string(), sequence_value);
        *value = compact_json_string(&JsonValue::Object(out));
    }

    changed
}

fn draw_text2d_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;

    let mut text = json_object_string(&object, "text", "");
    let mut color = json_object_vec4(&object, "color", (1.0, 1.0, 1.0, 1.0));
    let mut font_path = json_object_string(&object, "font_path", "");
    let mut font_family = json_object_string(&object, "font_family", "");
    let mut font_size = json_object_f64(&object, "font_size", 32.0).max(0.01);
    let mut font_weight = json_object_f64(&object, "font_weight", 400.0).clamp(1.0, 1000.0);
    let mut font_width = json_object_f64(&object, "font_width", 1.0).clamp(0.25, 4.0);
    let mut font_style = json_object_string(&object, "font_style", "normal");
    if !matches!(font_style.as_str(), "normal" | "italic" | "oblique") {
        font_style = "normal".to_string();
    }
    let mut line_height_scale = json_object_f64(&object, "line_height_scale", 1.0).max(0.1);
    let mut letter_spacing = json_object_f64(&object, "letter_spacing", 0.0);
    let mut word_spacing = json_object_f64(&object, "word_spacing", 0.0);
    let mut underline = json_object_bool(&object, "underline", false);
    let mut strikethrough = json_object_bool(&object, "strikethrough", false);
    let mut max_width = json_object_field(&object, "max_width")
        .and_then(|value| {
            if value.is_null() {
                None
            } else {
                coerce_json_to_f64(value).ok().filter(|value| *value > 0.0)
            }
        })
        .or(None);
    let mut align_h = json_object_string(&object, "align_h", "left");
    if !matches!(align_h.as_str(), "left" | "center" | "right") {
        align_h = "left".to_string();
    }
    let mut align_v = json_object_string(&object, "align_v", "baseline");
    if !matches!(align_v.as_str(), "top" | "center" | "bottom" | "baseline") {
        align_v = "baseline".to_string();
    }
    let mut space = json_object_string(&object, "space", "world");
    if !matches!(space.as_str(), "world" | "screen") {
        space = "world".to_string();
    }
    let mut blend_mode = json_object_string(&object, "blend_mode", "alpha");
    if !matches!(blend_mode.as_str(), "alpha" | "premultiplied" | "additive") {
        blend_mode = "alpha".to_string();
    }
    let mut billboard = json_object_bool(&object, "billboard", false);
    let mut visible = json_object_bool(&object, "visible", true);
    let mut layer = json_object_f64(&object, "layer", 0.0);
    let mut has_clip_rect = object
        .get("clip_rect")
        .is_some_and(|clip_rect| !clip_rect.is_null());
    let mut clip_rect = json_object_vec4(&object, "clip_rect", (0.0, 0.0, 1.0, 1.0));
    let mut pick_id = json_object_u64(&object, "pick_id", None);

    ui.horizontal(|ui| {
        ui.label("Text");
        changed |= ui
            .add(TextEdit::singleline(&mut text).desired_width(220.0))
            .changed();
    });
    changed |= draw_vec4_row(ui, "Color", &mut color, 0.01);
    ui.horizontal(|ui| {
        ui.label("Font Path");
        changed |= ui
            .add(TextEdit::singleline(&mut font_path).desired_width(180.0))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Font Family");
        changed |= ui
            .add(TextEdit::singleline(&mut font_family).desired_width(140.0))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Size");
        changed |= ui.add(DragValue::new(&mut font_size).speed(0.1)).changed();
        ui.label("Weight");
        changed |= ui
            .add(DragValue::new(&mut font_weight).speed(1.0))
            .changed();
        ui.label("Width");
        changed |= ui
            .add(DragValue::new(&mut font_width).speed(0.01))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Style");
        ComboBox::from_id_salt(("visual_text2d_font_style_inline", ui.id()))
            .selected_text(font_style.clone())
            .show_ui(ui, |ui| {
                for candidate in ["normal", "italic", "oblique"] {
                    changed |= ui
                        .selectable_value(&mut font_style, candidate.to_string(), candidate)
                        .changed();
                }
            });
        ui.label("Line Height");
        changed |= ui
            .add(DragValue::new(&mut line_height_scale).speed(0.01))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Letter");
        changed |= ui
            .add(DragValue::new(&mut letter_spacing).speed(0.01))
            .changed();
        ui.label("Word");
        changed |= ui
            .add(DragValue::new(&mut word_spacing).speed(0.01))
            .changed();
        changed |= ui.checkbox(&mut underline, "Underline").changed();
        changed |= ui.checkbox(&mut strikethrough, "Strike").changed();
    });
    ui.horizontal(|ui| {
        let mut has_max_width = max_width.is_some();
        if ui.checkbox(&mut has_max_width, "Max Width").changed() {
            max_width = if has_max_width {
                Some(max_width.unwrap_or(512.0))
            } else {
                None
            };
            changed = true;
        }
        if has_max_width {
            let mut width = max_width.unwrap_or(512.0);
            if ui.add(DragValue::new(&mut width).speed(1.0)).changed() {
                max_width = Some(width.max(0.01));
                changed = true;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("Align H");
        ComboBox::from_id_salt(("visual_text2d_align_h_inline", ui.id()))
            .selected_text(align_h.clone())
            .show_ui(ui, |ui| {
                for candidate in ["left", "center", "right"] {
                    changed |= ui
                        .selectable_value(&mut align_h, candidate.to_string(), candidate)
                        .changed();
                }
            });
        ui.label("Align V");
        ComboBox::from_id_salt(("visual_text2d_align_v_inline", ui.id()))
            .selected_text(align_v.clone())
            .show_ui(ui, |ui| {
                for candidate in ["top", "center", "bottom", "baseline"] {
                    changed |= ui
                        .selectable_value(&mut align_v, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Space");
        ComboBox::from_id_salt(("visual_text2d_space_inline", ui.id()))
            .selected_text(space.clone())
            .show_ui(ui, |ui| {
                for candidate in ["world", "screen"] {
                    changed |= ui
                        .selectable_value(&mut space, candidate.to_string(), candidate)
                        .changed();
                }
            });
        ui.label("Blend");
        ComboBox::from_id_salt(("visual_text2d_blend_inline", ui.id()))
            .selected_text(blend_mode.clone())
            .show_ui(ui, |ui| {
                for candidate in ["alpha", "premultiplied", "additive"] {
                    changed |= ui
                        .selectable_value(&mut blend_mode, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut billboard, "Billboard").changed();
        changed |= ui.checkbox(&mut visible, "Visible").changed();
        ui.label("Layer");
        changed |= ui.add(DragValue::new(&mut layer).speed(0.1)).changed();
    });
    ui.horizontal(|ui| {
        let mut has_pick_id = pick_id.is_some();
        if ui.checkbox(&mut has_pick_id, "Pick Id").changed() {
            pick_id = if has_pick_id {
                Some(pick_id.unwrap_or(0))
            } else {
                None
            };
            changed = true;
        }
        if has_pick_id {
            let mut value = pick_id.unwrap_or(0);
            if ui
                .add(DragValue::new(&mut value).speed(1.0).range(0..=u64::MAX))
                .changed()
            {
                pick_id = Some(value);
                changed = true;
            }
            if ui.small_button("Clear").clicked() {
                pick_id = None;
                changed = true;
            }
        }
    });
    ui.horizontal(|ui| {
        if ui.checkbox(&mut has_clip_rect, "Clip Rect").changed() {
            changed = true;
        }
    });
    if has_clip_rect {
        changed |= draw_vec4_row(ui, "Rect", &mut clip_rect, 0.01);
    }

    if changed {
        object.insert("text".to_string(), JsonValue::String(text));
        object.insert(
            "color".to_string(),
            vec4_json(color.0, color.1, color.2, color.3),
        );
        object.insert(
            "font_path".to_string(),
            if font_path.trim().is_empty() {
                JsonValue::Null
            } else {
                JsonValue::String(font_path.trim().to_string())
            },
        );
        object.insert(
            "font_family".to_string(),
            if font_family.trim().is_empty() {
                JsonValue::Null
            } else {
                JsonValue::String(font_family.trim().to_string())
            },
        );
        object.insert("font_size".to_string(), json_number(font_size.max(0.01)));
        object.insert(
            "font_weight".to_string(),
            json_number(font_weight.clamp(1.0, 1000.0)),
        );
        object.insert(
            "font_width".to_string(),
            json_number(font_width.clamp(0.25, 4.0)),
        );
        object.insert("font_style".to_string(), JsonValue::String(font_style));
        object.insert(
            "line_height_scale".to_string(),
            json_number(line_height_scale.max(0.1)),
        );
        object.insert("letter_spacing".to_string(), json_number(letter_spacing));
        object.insert("word_spacing".to_string(), json_number(word_spacing));
        object.insert("underline".to_string(), JsonValue::Bool(underline));
        object.insert("strikethrough".to_string(), JsonValue::Bool(strikethrough));
        object.insert(
            "max_width".to_string(),
            max_width.map(json_number).unwrap_or(JsonValue::Null),
        );
        object.insert("align_h".to_string(), JsonValue::String(align_h));
        object.insert("align_v".to_string(), JsonValue::String(align_v));
        object.insert("space".to_string(), JsonValue::String(space));
        object.insert("blend_mode".to_string(), JsonValue::String(blend_mode));
        object.insert("billboard".to_string(), JsonValue::Bool(billboard));
        object.insert("visible".to_string(), JsonValue::Bool(visible));
        object.insert("layer".to_string(), json_number(layer));
        object.insert(
            "clip_rect".to_string(),
            if has_clip_rect {
                vec4_json(clip_rect.0, clip_rect.1, clip_rect.2, clip_rect.3)
            } else {
                JsonValue::Null
            },
        );
        object.insert(
            "pick_id".to_string(),
            pick_id
                .map(|value| JsonValue::Number(JsonNumber::from(value)))
                .unwrap_or(JsonValue::Null),
        );
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_audio_listener_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut enabled = json_object_bool(&object, "enabled", true);
    let changed = ui.checkbox(&mut enabled, "Enabled").changed();
    if changed {
        object.insert("enabled".to_string(), JsonValue::Bool(enabled));
        *value = compact_json_string(&JsonValue::Object(object));
    }
    changed
}

fn draw_audio_emitter_data_editor(
    ui: &mut Ui,
    value: &mut String,
    pending_asset_nodes: &mut Vec<(String, egui::Pos2)>,
    consumed_asset_drop: &mut bool,
    project_root: Option<&Path>,
) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;

    let mut path = json_object_string(&object, "path", "");
    let mut streaming = json_object_bool(&object, "streaming", false);
    let mut bus = json_object_string(&object, "bus", "Master");
    if bus.is_empty() {
        bus = "Master".to_string();
    }
    let mut volume = json_object_f64(&object, "volume", 1.0);
    let mut pitch = json_object_f64(&object, "pitch", 1.0);
    let mut looping = json_object_bool(&object, "looping", false);
    let mut spatial = json_object_bool(&object, "spatial", true);
    let mut min_distance = json_object_f64(&object, "min_distance", 1.0);
    let mut max_distance = json_object_f64(&object, "max_distance", 30.0);
    let mut rolloff = json_object_f64(&object, "rolloff", 1.0);
    let mut spatial_blend = json_object_f64(&object, "spatial_blend", 1.0);
    let mut play_on_spawn = json_object_bool(&object, "play_on_spawn", false);
    let mut playback_state = json_object_string(&object, "playback_state", "Stopped");

    ui.horizontal(|ui| {
        ui.label("Audio Path");
        let response = ui.add(TextEdit::singleline(&mut path).desired_width(180.0));
        changed |= response.changed();
        changed |= handle_asset_path_drop(
            ui,
            &response,
            VisualAssetPathKind::Audio,
            &mut path,
            pending_asset_nodes,
            consumed_asset_drop,
            project_root,
        );
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut streaming, "Streaming").changed();
        changed |= ui.checkbox(&mut looping, "Looping").changed();
        changed |= ui.checkbox(&mut spatial, "Spatial").changed();
    });
    ui.horizontal(|ui| {
        ui.label("Bus");
        changed |= ui
            .add(TextEdit::singleline(&mut bus).desired_width(110.0))
            .changed();
        ui.label("Playback");
        ComboBox::from_id_salt(("visual_audio_playback_inline", ui.id()))
            .selected_text(playback_state.clone())
            .show_ui(ui, |ui| {
                for state in ["Playing", "Paused", "Stopped"] {
                    changed |= ui
                        .selectable_value(&mut playback_state, state.to_string(), state)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Volume");
        changed |= ui.add(DragValue::new(&mut volume).speed(0.05)).changed();
        ui.label("Pitch");
        changed |= ui.add(DragValue::new(&mut pitch).speed(0.05)).changed();
        changed |= ui.checkbox(&mut play_on_spawn, "Play on spawn").changed();
    });
    ui.horizontal(|ui| {
        ui.label("Min/Max");
        changed |= ui
            .add(DragValue::new(&mut min_distance).speed(0.1))
            .changed();
        changed |= ui
            .add(DragValue::new(&mut max_distance).speed(0.1))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Rolloff");
        changed |= ui.add(DragValue::new(&mut rolloff).speed(0.05)).changed();
        ui.label("Spatial Blend");
        changed |= ui
            .add(DragValue::new(&mut spatial_blend).speed(0.05))
            .changed();
    });

    if changed {
        if path.trim().is_empty() {
            object.insert("path".to_string(), JsonValue::Null);
        } else {
            object.insert("path".to_string(), JsonValue::String(path));
        }
        object.insert("streaming".to_string(), JsonValue::Bool(streaming));
        object.insert("bus".to_string(), JsonValue::String(bus));
        object.insert("volume".to_string(), json_number(volume.max(0.0)));
        object.insert("pitch".to_string(), json_number(pitch.max(0.001)));
        object.insert("looping".to_string(), JsonValue::Bool(looping));
        object.insert("spatial".to_string(), JsonValue::Bool(spatial));
        object.insert(
            "min_distance".to_string(),
            json_number(min_distance.max(0.0)),
        );
        object.insert(
            "max_distance".to_string(),
            json_number(max_distance.max(min_distance.max(0.0))),
        );
        object.insert("rolloff".to_string(), json_number(rolloff.max(0.0)));
        object.insert(
            "spatial_blend".to_string(),
            json_number(spatial_blend.clamp(0.0, 1.0)),
        );
        object.insert("play_on_spawn".to_string(), JsonValue::Bool(play_on_spawn));
        object.insert(
            "playback_state".to_string(),
            JsonValue::String(playback_state),
        );
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_physics_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut root = parse_json_object_literal(value);
    let mut changed = false;

    let mut body_kind = json_object_nested(&root, "body_kind");
    let mut body_type = json_object_string(&body_kind, "type", "Dynamic");
    if body_type.is_empty() {
        body_type = "Dynamic".to_string();
    }
    let mut dynamic_mass = json_object_f64(&body_kind, "mass", 1.0);
    let mut kinematic_mode = json_object_string(&body_kind, "mode", "PositionBased");
    if kinematic_mode.is_empty() {
        kinematic_mode = "PositionBased".to_string();
    }

    ui.group(|ui| {
        ui.label("Body");
        ui.horizontal(|ui| {
            ui.label("Kind");
            ComboBox::from_id_salt(("visual_physics_body_type_inline", ui.id()))
                .selected_text(body_type.clone())
                .show_ui(ui, |ui| {
                    for kind in ["Dynamic", "Kinematic", "Fixed"] {
                        changed |= ui
                            .selectable_value(&mut body_type, kind.to_string(), kind)
                            .changed();
                    }
                });
        });
        if body_type.eq_ignore_ascii_case("dynamic") {
            ui.horizontal(|ui| {
                ui.label("Mass");
                changed |= ui
                    .add(DragValue::new(&mut dynamic_mass).speed(0.1))
                    .changed();
            });
        }
        if body_type.eq_ignore_ascii_case("kinematic") {
            ui.horizontal(|ui| {
                ui.label("Mode");
                ComboBox::from_id_salt(("visual_physics_kinematic_mode_inline", ui.id()))
                    .selected_text(kinematic_mode.clone())
                    .show_ui(ui, |ui| {
                        for mode in ["PositionBased", "VelocityBased"] {
                            changed |= ui
                                .selectable_value(&mut kinematic_mode, mode.to_string(), mode)
                                .changed();
                        }
                    });
            });
        }
    });

    let mut collider_shape = json_object_nested(&root, "collider_shape");
    let mut collider_type = json_object_string(&collider_shape, "type", "Cuboid");
    if collider_type.is_empty() {
        collider_type = "Cuboid".to_string();
    }
    let mut round_border = json_object_f64(&collider_shape, "border_radius", 0.05);
    let mut mesh_id = json_object_i64(&collider_shape, "mesh_id", -1);
    let mut mesh_lod = json_object_string(&collider_shape, "lod", "Lod0");
    if mesh_lod.is_empty() {
        mesh_lod = "Lod0".to_string();
    }
    let mut mesh_kind = json_object_string(&collider_shape, "kind", "TriMesh");
    if mesh_kind.is_empty() {
        mesh_kind = "TriMesh".to_string();
    }

    ui.group(|ui| {
        ui.label("Collider");
        ui.horizontal(|ui| {
            ui.label("Shape");
            ComboBox::from_id_salt(("visual_physics_shape_inline", ui.id()))
                .selected_text(collider_type.clone())
                .show_ui(ui, |ui| {
                    for shape in [
                        "Cuboid",
                        "Sphere",
                        "CapsuleY",
                        "CylinderY",
                        "ConeY",
                        "RoundCuboid",
                        "Mesh",
                    ] {
                        changed |= ui
                            .selectable_value(&mut collider_type, shape.to_string(), shape)
                            .changed();
                    }
                });
        });
        if collider_type.eq_ignore_ascii_case("roundcuboid") {
            ui.horizontal(|ui| {
                ui.label("Border Radius");
                changed |= ui
                    .add(DragValue::new(&mut round_border).speed(0.01))
                    .changed();
            });
        }
        if collider_type.eq_ignore_ascii_case("mesh") {
            ui.horizontal(|ui| {
                ui.label("Mesh Id");
                changed |= ui.add(DragValue::new(&mut mesh_id).speed(1.0)).changed();
            });
            ui.horizontal(|ui| {
                ui.label("LOD");
                ComboBox::from_id_salt(("visual_physics_mesh_lod_inline", ui.id()))
                    .selected_text(mesh_lod.clone())
                    .show_ui(ui, |ui| {
                        for lod in ["Lod0", "Lod1", "Lod2", "Lowest"] {
                            changed |= ui
                                .selectable_value(&mut mesh_lod, lod.to_string(), lod)
                                .changed();
                        }
                    });
                ui.label("Kind");
                ComboBox::from_id_salt(("visual_physics_mesh_kind_inline", ui.id()))
                    .selected_text(mesh_kind.clone())
                    .show_ui(ui, |ui| {
                        for kind in ["TriMesh", "ConvexHull"] {
                            changed |= ui
                                .selectable_value(&mut mesh_kind, kind.to_string(), kind)
                                .changed();
                        }
                    });
            });
        }
    });

    let mut rigid_body_props = json_object_nested(&root, "rigid_body_properties");
    let mut linear_damping = json_object_f64(&rigid_body_props, "linear_damping", 0.0);
    let mut angular_damping = json_object_f64(&rigid_body_props, "angular_damping", 0.0);
    let mut gravity_scale = json_object_f64(&rigid_body_props, "gravity_scale", 1.0);
    let mut ccd_enabled = json_object_bool(&rigid_body_props, "ccd_enabled", false);
    let mut can_sleep = json_object_bool(&rigid_body_props, "can_sleep", true);
    let mut sleeping = json_object_bool(&rigid_body_props, "sleeping", false);
    let mut dominance_group = json_object_f64(&rigid_body_props, "dominance_group", 0.0);
    let mut lock_tx = json_object_bool(&rigid_body_props, "lock_translation_x", false);
    let mut lock_ty = json_object_bool(&rigid_body_props, "lock_translation_y", false);
    let mut lock_tz = json_object_bool(&rigid_body_props, "lock_translation_z", false);
    let mut lock_rx = json_object_bool(&rigid_body_props, "lock_rotation_x", false);
    let mut lock_ry = json_object_bool(&rigid_body_props, "lock_rotation_y", false);
    let mut lock_rz = json_object_bool(&rigid_body_props, "lock_rotation_z", false);
    let mut linear_velocity =
        json_object_vec3(&rigid_body_props, "linear_velocity", (0.0, 0.0, 0.0));
    let mut angular_velocity =
        json_object_vec3(&rigid_body_props, "angular_velocity", (0.0, 0.0, 0.0));

    ui.group(|ui| {
        ui.label("Rigid Body");
        ui.horizontal(|ui| {
            ui.label("Linear Damp");
            changed |= ui
                .add(DragValue::new(&mut linear_damping).speed(0.01))
                .changed();
            ui.label("Angular Damp");
            changed |= ui
                .add(DragValue::new(&mut angular_damping).speed(0.01))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Gravity Scale");
            changed |= ui
                .add(DragValue::new(&mut gravity_scale).speed(0.05))
                .changed();
            ui.label("Dominance");
            changed |= ui
                .add(DragValue::new(&mut dominance_group).speed(1.0))
                .changed();
        });
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut ccd_enabled, "CCD").changed();
            changed |= ui.checkbox(&mut can_sleep, "Can Sleep").changed();
            changed |= ui.checkbox(&mut sleeping, "Sleeping").changed();
        });
        ui.horizontal(|ui| {
            ui.label("Lock T");
            changed |= ui.checkbox(&mut lock_tx, "X").changed();
            changed |= ui.checkbox(&mut lock_ty, "Y").changed();
            changed |= ui.checkbox(&mut lock_tz, "Z").changed();
        });
        ui.horizontal(|ui| {
            ui.label("Lock R");
            changed |= ui.checkbox(&mut lock_rx, "X").changed();
            changed |= ui.checkbox(&mut lock_ry, "Y").changed();
            changed |= ui.checkbox(&mut lock_rz, "Z").changed();
        });
        changed |= draw_vec3_row(ui, "Linear Vel", &mut linear_velocity, 0.05);
        changed |= draw_vec3_row(ui, "Angular Vel", &mut angular_velocity, 0.05);
    });

    let mut collider_props = json_object_nested(&root, "collider_properties");
    let mut friction = json_object_f64(&collider_props, "friction", 0.5);
    let mut restitution = json_object_f64(&collider_props, "restitution", 0.0);
    let mut density = json_object_f64(&collider_props, "density", 1.0);
    let mut is_sensor = json_object_bool(&collider_props, "is_sensor", false);
    let mut enabled = json_object_bool(&collider_props, "enabled", true);

    ui.group(|ui| {
        ui.label("Collider Properties");
        ui.horizontal(|ui| {
            ui.label("Friction");
            changed |= ui.add(DragValue::new(&mut friction).speed(0.01)).changed();
            ui.label("Restitution");
            changed |= ui
                .add(DragValue::new(&mut restitution).speed(0.01))
                .changed();
            ui.label("Density");
            changed |= ui.add(DragValue::new(&mut density).speed(0.05)).changed();
        });
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut enabled, "Enabled").changed();
            changed |= ui.checkbox(&mut is_sensor, "Sensor").changed();
        });
    });

    if changed {
        body_kind.insert("type".to_string(), JsonValue::String(body_type.clone()));
        if body_type.eq_ignore_ascii_case("dynamic") {
            body_kind.insert("mass".to_string(), json_number(dynamic_mass.max(0.0)));
            body_kind.remove("mode");
        } else if body_type.eq_ignore_ascii_case("kinematic") {
            body_kind.insert(
                "mode".to_string(),
                JsonValue::String(kinematic_mode.clone()),
            );
            body_kind.remove("mass");
        } else {
            body_kind.remove("mass");
            body_kind.remove("mode");
        }
        root.insert("body_kind".to_string(), JsonValue::Object(body_kind));

        collider_shape.insert("type".to_string(), JsonValue::String(collider_type.clone()));
        if collider_type.eq_ignore_ascii_case("roundcuboid") {
            collider_shape.insert(
                "border_radius".to_string(),
                json_number(round_border.max(0.0)),
            );
        } else {
            collider_shape.remove("border_radius");
        }
        if collider_type.eq_ignore_ascii_case("mesh") {
            if mesh_id >= 0 {
                collider_shape.insert(
                    "mesh_id".to_string(),
                    JsonValue::Number(JsonNumber::from(mesh_id)),
                );
            } else {
                collider_shape.insert("mesh_id".to_string(), JsonValue::Null);
            }
            collider_shape.insert("lod".to_string(), JsonValue::String(mesh_lod));
            collider_shape.insert("kind".to_string(), JsonValue::String(mesh_kind));
        } else {
            collider_shape.remove("mesh_id");
            collider_shape.remove("lod");
            collider_shape.remove("kind");
        }
        root.insert(
            "collider_shape".to_string(),
            JsonValue::Object(collider_shape),
        );

        rigid_body_props.insert(
            "linear_damping".to_string(),
            json_number(linear_damping.max(0.0)),
        );
        rigid_body_props.insert(
            "angular_damping".to_string(),
            json_number(angular_damping.max(0.0)),
        );
        rigid_body_props.insert("gravity_scale".to_string(), json_number(gravity_scale));
        rigid_body_props.insert("ccd_enabled".to_string(), JsonValue::Bool(ccd_enabled));
        rigid_body_props.insert("can_sleep".to_string(), JsonValue::Bool(can_sleep));
        rigid_body_props.insert("sleeping".to_string(), JsonValue::Bool(sleeping));
        rigid_body_props.insert(
            "dominance_group".to_string(),
            JsonValue::Number(JsonNumber::from(
                dominance_group
                    .round()
                    .clamp(i8::MIN as f64, i8::MAX as f64) as i64,
            )),
        );
        rigid_body_props.insert("lock_translation_x".to_string(), JsonValue::Bool(lock_tx));
        rigid_body_props.insert("lock_translation_y".to_string(), JsonValue::Bool(lock_ty));
        rigid_body_props.insert("lock_translation_z".to_string(), JsonValue::Bool(lock_tz));
        rigid_body_props.insert("lock_rotation_x".to_string(), JsonValue::Bool(lock_rx));
        rigid_body_props.insert("lock_rotation_y".to_string(), JsonValue::Bool(lock_ry));
        rigid_body_props.insert("lock_rotation_z".to_string(), JsonValue::Bool(lock_rz));
        rigid_body_props.insert(
            "linear_velocity".to_string(),
            vec3_json(linear_velocity.0, linear_velocity.1, linear_velocity.2),
        );
        rigid_body_props.insert(
            "angular_velocity".to_string(),
            vec3_json(angular_velocity.0, angular_velocity.1, angular_velocity.2),
        );
        root.insert(
            "rigid_body_properties".to_string(),
            JsonValue::Object(rigid_body_props),
        );

        collider_props.insert("friction".to_string(), json_number(friction.max(0.0)));
        collider_props.insert("restitution".to_string(), json_number(restitution.max(0.0)));
        collider_props.insert("density".to_string(), json_number(density.max(0.0)));
        collider_props.insert("enabled".to_string(), JsonValue::Bool(enabled));
        collider_props.insert("is_sensor".to_string(), JsonValue::Bool(is_sensor));
        root.insert(
            "collider_properties".to_string(),
            JsonValue::Object(collider_props),
        );

        *value = compact_json_string(&JsonValue::Object(root));
    }

    changed
}

fn draw_physics_world_defaults_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut gravity = json_object_vec3(&object, "gravity", (0.0, -9.81, 0.0));
    changed |= draw_vec3_row(ui, "Gravity", &mut gravity, 0.05);
    if changed {
        object.insert(
            "gravity".to_string(),
            vec3_json(gravity.0, gravity.1, gravity.2),
        );
        *value = compact_json_string(&JsonValue::Object(object));
    }
    changed
}

fn draw_physics_velocity_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut linear = json_object_vec3(&object, "linear", (0.0, 0.0, 0.0));
    let mut angular = json_object_vec3(&object, "angular", (0.0, 0.0, 0.0));
    let mut wake_up = json_object_bool(&object, "wake_up", true);

    changed |= draw_vec3_row(ui, "Linear", &mut linear, 0.05);
    changed |= draw_vec3_row(ui, "Angular", &mut angular, 0.05);
    changed |= ui.checkbox(&mut wake_up, "Wake Up").changed();

    if changed {
        object.insert(
            "linear".to_string(),
            vec3_json(linear.0, linear.1, linear.2),
        );
        object.insert(
            "angular".to_string(),
            vec3_json(angular.0, angular.1, angular.2),
        );
        object.insert("wake_up".to_string(), JsonValue::Bool(wake_up));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_dynamic_component_fields_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut remove_key: Option<String> = None;
    let mut rename_ops: Vec<(String, String)> = Vec::new();
    let mut insert_ops: Vec<(String, JsonValue)> = Vec::new();

    let mut keys: Vec<String> = object.keys().cloned().collect();
    keys.sort();

    for key in keys {
        let Some(current) = object.get(&key).cloned() else {
            continue;
        };
        let mut field_name = key.clone();
        let mut field_type = infer_visual_value_type_from_json(&current);
        if !matches!(
            field_type,
            VisualValueType::Bool
                | VisualValueType::Number
                | VisualValueType::String
                | VisualValueType::Vec3
        ) {
            field_type = VisualValueType::String;
        }
        let mut literal = literal_string_for_value_type(&current, field_type);

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("Field");
                if ui
                    .add(TextEdit::singleline(&mut field_name).desired_width(96.0))
                    .changed()
                {
                    changed = true;
                }
                ComboBox::from_id_salt(("visual_dynamic_component_field_type", ui.id(), &key))
                    .selected_text(field_type.title())
                    .show_ui(ui, |ui| {
                        for candidate in [
                            VisualValueType::Bool,
                            VisualValueType::Number,
                            VisualValueType::String,
                            VisualValueType::Vec3,
                        ] {
                            if ui
                                .selectable_value(&mut field_type, candidate, candidate.title())
                                .changed()
                            {
                                literal = normalize_literal_for_type(&literal, candidate);
                                changed = true;
                            }
                        }
                    });
                if ui.small_button("Remove").clicked() {
                    remove_key = Some(key.clone());
                    changed = true;
                }
            });
            changed |= draw_typed_default_editor(ui, field_type, &mut literal);
        });

        if remove_key.as_ref() == Some(&key) {
            continue;
        }

        if field_name.trim().is_empty() {
            remove_key = Some(key.clone());
            changed = true;
            continue;
        }

        let normalized = coerce_json_to_visual_type(&parse_loose_literal(&literal), field_type)
            .unwrap_or_else(|_| parse_loose_literal(default_literal_for_type(field_type)));
        if field_name != key {
            rename_ops.push((key.clone(), field_name.clone()));
        }
        insert_ops.push((field_name, normalized));
    }

    if let Some(key) = remove_key {
        object.remove(&key);
    }
    for (old_key, new_key) in rename_ops {
        if old_key != new_key {
            object.remove(&old_key);
        }
    }
    for (key, value) in insert_ops {
        object.insert(key, value);
    }

    ui.horizontal(|ui| {
        if ui.small_button("+ Bool Field").clicked() {
            let mut candidate = "field_bool".to_string();
            let mut suffix = 1usize;
            while object.contains_key(&candidate) {
                candidate = format!("field_bool_{}", suffix);
                suffix += 1;
            }
            object.insert(candidate, JsonValue::Bool(false));
            changed = true;
        }
        if ui.small_button("+ Number Field").clicked() {
            let mut candidate = "field_number".to_string();
            let mut suffix = 1usize;
            while object.contains_key(&candidate) {
                candidate = format!("field_number_{}", suffix);
                suffix += 1;
            }
            object.insert(candidate, json_number(0.0));
            changed = true;
        }
        if ui.small_button("+ String Field").clicked() {
            let mut candidate = "field_string".to_string();
            let mut suffix = 1usize;
            while object.contains_key(&candidate) {
                candidate = format!("field_string_{}", suffix);
                suffix += 1;
            }
            object.insert(candidate, JsonValue::String(String::new()));
            changed = true;
        }
        if ui.small_button("+ Vec3 Field").clicked() {
            let mut candidate = "field_vec3".to_string();
            let mut suffix = 1usize;
            while object.contains_key(&candidate) {
                candidate = format!("field_vec3_{}", suffix);
                suffix += 1;
            }
            object.insert(candidate, vec3_json(0.0, 0.0, 0.0));
            changed = true;
        }
    });

    if changed {
        *value = compact_json_string(&JsonValue::Object(object));
    }
    changed
}

fn draw_dynamic_field_value_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut changed = false;
    let mut value_type = infer_visual_value_type_from_literal(value);
    if value_type == VisualValueType::Entity {
        value_type = VisualValueType::Number;
    } else if value_type == VisualValueType::Any {
        value_type = VisualValueType::String;
    } else if !matches!(
        value_type,
        VisualValueType::Bool
            | VisualValueType::Number
            | VisualValueType::String
            | VisualValueType::Vec3
    ) {
        value_type = VisualValueType::String;
    }
    ui.horizontal(|ui| {
        ui.label("Type");
        ComboBox::from_id_salt(("visual_dynamic_field_type_inline", ui.id()))
            .selected_text(value_type.title())
            .show_ui(ui, |ui| {
                for candidate in [
                    VisualValueType::Bool,
                    VisualValueType::Number,
                    VisualValueType::String,
                    VisualValueType::Vec3,
                ] {
                    if ui
                        .selectable_value(&mut value_type, candidate, candidate.title())
                        .changed()
                    {
                        *value = normalize_literal_for_type(value, candidate);
                        changed = true;
                    }
                }
            });
    });
    changed |= draw_typed_pin_input_editor(ui, value_type, value);
    changed
}

fn draw_input_binding_editor(ui: &mut Ui, value: &mut String) -> bool {
    const PRESETS: [&str; 23] = [
        "KeyW",
        "KeyA",
        "KeyS",
        "KeyD",
        "KeySpace",
        "KeyLeftShift",
        "KeyLeftCtrl",
        "KeyQ",
        "KeyE",
        "KeyR",
        "MouseLeft",
        "MouseRight",
        "MouseMiddle",
        "GamepadSouth",
        "GamepadEast",
        "GamepadWest",
        "GamepadNorth",
        "GamepadLeftShoulder",
        "GamepadRightShoulder",
        "GamepadLeftStickX",
        "GamepadLeftStickY",
        "GamepadRightStickX",
        "GamepadRightStickY",
    ];

    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Preset");
        ComboBox::from_id_salt(("visual_input_binding_preset", ui.id()))
            .selected_text(if value.trim().is_empty() {
                "Custom".to_string()
            } else {
                value.clone()
            })
            .show_ui(ui, |ui| {
                for preset in PRESETS {
                    if ui.selectable_label(false, preset).clicked() {
                        *value = preset.to_string();
                        changed = true;
                        ui.close();
                    }
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Binding");
        changed |= ui
            .add(TextEdit::singleline(value).desired_width(140.0))
            .changed();
    });
    changed
}

fn draw_api_structured_default_editor(
    ui: &mut Ui,
    operation: VisualApiOperation,
    input_index: usize,
    default_literal: &mut String,
    pending_asset_nodes: &mut Vec<(String, egui::Pos2)>,
    consumed_asset_drop: &mut bool,
    project_root: Option<&Path>,
) -> bool {
    match operation {
        VisualApiOperation::EcsSetCamera if input_index == 1 => {
            draw_camera_data_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetLight if input_index == 1 => {
            draw_light_data_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetMeshRenderer if input_index == 1 => {
            draw_mesh_renderer_data_editor(
                ui,
                default_literal,
                pending_asset_nodes,
                consumed_asset_drop,
                project_root,
            )
        }
        VisualApiOperation::EcsSetSpriteRenderer if input_index == 1 => {
            draw_sprite_renderer_data_editor(
                ui,
                default_literal,
                pending_asset_nodes,
                consumed_asset_drop,
                project_root,
            )
        }
        VisualApiOperation::EcsSetSpriteRendererSheetAnimation if input_index == 1 => {
            draw_sprite_sheet_animation_patch_data_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetSpriteRendererSequence if input_index == 1 => {
            draw_sprite_image_sequence_patch_data_editor(
                ui,
                default_literal,
                pending_asset_nodes,
                consumed_asset_drop,
                project_root,
            )
        }
        VisualApiOperation::EcsSetText2d if input_index == 1 => {
            draw_text2d_data_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetAudioEmitter if input_index == 1 => {
            draw_audio_emitter_data_editor(
                ui,
                default_literal,
                pending_asset_nodes,
                consumed_asset_drop,
                project_root,
            )
        }
        VisualApiOperation::EcsSetAudioListener if input_index == 1 => {
            draw_audio_listener_data_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetPhysics if input_index == 1 => {
            draw_physics_data_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetPhysicsVelocity if input_index == 1 => {
            draw_physics_velocity_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetPhysicsWorldDefaults if input_index == 1 => {
            draw_physics_world_defaults_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetDynamicComponent if input_index == 2 => {
            draw_dynamic_component_fields_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetDynamicField if input_index == 3 => {
            draw_dynamic_field_value_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetRuntimeTuning if input_index == 0 => {
            draw_runtime_tuning_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetRuntimeConfig if input_index == 0 => {
            draw_runtime_config_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetRenderConfig if input_index == 0 => {
            draw_render_config_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetShaderConstants if input_index == 0 => {
            draw_shader_constants_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetStreamingTuning if input_index == 0 => {
            draw_streaming_tuning_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetRenderPasses if input_index == 0 => {
            draw_render_passes_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetGpuBudget if input_index == 0 => {
            draw_gpu_budget_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetAssetBudgets if input_index == 0 => {
            draw_asset_budgets_editor(ui, default_literal)
        }
        VisualApiOperation::EcsSetWindowSettings if input_index == 0 => {
            draw_window_settings_editor(ui, default_literal)
        }
        VisualApiOperation::InputBindAction if input_index == 1 => {
            draw_input_binding_editor(ui, default_literal)
        }
        VisualApiOperation::InputUnbindAction if input_index == 1 => {
            draw_input_binding_editor(ui, default_literal)
        }
        _ => false,
    }
}

fn draw_mesh_renderer_data_editor_plain(ui: &mut Ui, value: &mut String) -> bool {
    let mut pending = Vec::new();
    let mut consumed = false;
    draw_mesh_renderer_data_editor(ui, value, &mut pending, &mut consumed, None)
}

fn draw_sprite_renderer_data_editor_plain(ui: &mut Ui, value: &mut String) -> bool {
    let mut pending = Vec::new();
    let mut consumed = false;
    draw_sprite_renderer_data_editor(ui, value, &mut pending, &mut consumed, None)
}

fn draw_text2d_data_editor_plain(ui: &mut Ui, value: &mut String) -> bool {
    draw_text2d_data_editor(ui, value)
}

fn draw_audio_emitter_data_editor_plain(ui: &mut Ui, value: &mut String) -> bool {
    let mut pending = Vec::new();
    let mut consumed = false;
    draw_audio_emitter_data_editor(ui, value, &mut pending, &mut consumed, None)
}

fn draw_script_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut path = json_object_string(&object, "path", "");
    let mut language = json_object_string(&object, "language", "lua");

    ui.horizontal(|ui| {
        ui.label("Path");
        changed |= ui
            .add(TextEdit::singleline(&mut path).desired_width(180.0))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Language");
        changed |= ui
            .add(TextEdit::singleline(&mut language).desired_width(120.0))
            .changed();
    });

    if changed {
        object.insert("path".to_string(), JsonValue::String(path));
        object.insert("language".to_string(), JsonValue::String(language));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_look_at_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut target_entity = json_to_u64(object.get("target_entity").unwrap_or(&JsonValue::Null));
    let mut target_offset = json_object_vec3(&object, "target_offset", (0.0, 0.0, 0.0));
    let mut offset_in_target_space = json_object_bool(&object, "offset_in_target_space", false);
    let mut up = json_object_vec3(&object, "up", (0.0, 1.0, 0.0));
    let mut rotation_smooth_time = json_object_f64(&object, "rotation_smooth_time", 0.0).max(0.0);

    ui.horizontal(|ui| {
        ui.label("Target Entity");
        let mut raw = target_entity.unwrap_or(0);
        if ui
            .add(DragValue::new(&mut raw).speed(1.0).range(0..=u64::MAX))
            .changed()
        {
            target_entity = (raw > 0).then_some(raw);
            changed = true;
        }
        if ui.button("Clear").clicked() {
            target_entity = None;
            changed = true;
        }
    });
    changed |= draw_vec3_row(ui, "Target Offset", &mut target_offset, 0.05);
    changed |= draw_vec3_row(ui, "Up", &mut up, 0.05);
    ui.horizontal(|ui| {
        changed |= ui
            .checkbox(&mut offset_in_target_space, "Offset In Target Space")
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Rotation Smooth Time");
        changed |= ui
            .add(DragValue::new(&mut rotation_smooth_time).speed(0.01))
            .changed();
    });

    if changed {
        object.insert(
            "target_entity".to_string(),
            target_entity
                .map(|id| JsonValue::Number(JsonNumber::from(id)))
                .unwrap_or(JsonValue::Null),
        );
        object.insert(
            "target_offset".to_string(),
            vec3_json(target_offset.0, target_offset.1, target_offset.2),
        );
        object.insert(
            "offset_in_target_space".to_string(),
            JsonValue::Bool(offset_in_target_space),
        );
        object.insert("up".to_string(), vec3_json(up.0, up.1, up.2));
        object.insert(
            "rotation_smooth_time".to_string(),
            json_number(rotation_smooth_time.max(0.0)),
        );
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_entity_follower_data_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut target_entity = json_to_u64(object.get("target_entity").unwrap_or(&JsonValue::Null));
    let mut position_offset = json_object_vec3(&object, "position_offset", (0.0, 0.0, 0.0));
    let mut offset_in_target_space = json_object_bool(&object, "offset_in_target_space", false);
    let mut follow_rotation = json_object_bool(&object, "follow_rotation", true);
    let mut position_smooth_time = json_object_f64(&object, "position_smooth_time", 0.0).max(0.0);
    let mut rotation_smooth_time = json_object_f64(&object, "rotation_smooth_time", 0.0).max(0.0);

    ui.horizontal(|ui| {
        ui.label("Target Entity");
        let mut raw = target_entity.unwrap_or(0);
        if ui
            .add(DragValue::new(&mut raw).speed(1.0).range(0..=u64::MAX))
            .changed()
        {
            target_entity = (raw > 0).then_some(raw);
            changed = true;
        }
        if ui.button("Clear").clicked() {
            target_entity = None;
            changed = true;
        }
    });
    changed |= draw_vec3_row(ui, "Position Offset", &mut position_offset, 0.05);
    ui.horizontal(|ui| {
        changed |= ui
            .checkbox(&mut offset_in_target_space, "Offset In Target Space")
            .changed();
        changed |= ui
            .checkbox(&mut follow_rotation, "Follow Rotation")
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Position Smooth Time");
        changed |= ui
            .add(DragValue::new(&mut position_smooth_time).speed(0.01))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Rotation Smooth Time");
        changed |= ui
            .add(DragValue::new(&mut rotation_smooth_time).speed(0.01))
            .changed();
    });

    if changed {
        object.insert(
            "target_entity".to_string(),
            target_entity
                .map(|id| JsonValue::Number(JsonNumber::from(id)))
                .unwrap_or(JsonValue::Null),
        );
        object.insert(
            "position_offset".to_string(),
            vec3_json(position_offset.0, position_offset.1, position_offset.2),
        );
        object.insert(
            "offset_in_target_space".to_string(),
            JsonValue::Bool(offset_in_target_space),
        );
        object.insert(
            "follow_rotation".to_string(),
            JsonValue::Bool(follow_rotation),
        );
        object.insert(
            "position_smooth_time".to_string(),
            json_number(position_smooth_time.max(0.0)),
        );
        object.insert(
            "rotation_smooth_time".to_string(),
            json_number(rotation_smooth_time.max(0.0)),
        );
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_animator_state_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut layer_index = json_object_i64(&object, "layer_index", 0).max(0);
    let mut layer_name = json_object_string(&object, "layer_name", "");
    let mut layer_weight = json_object_f64(&object, "layer_weight", 1.0);
    let mut layer_additive = json_object_bool(&object, "layer_additive", false);
    let mut state_time = json_object_f64(&object, "state_time", 0.0).max(0.0);
    let mut current_state = json_object_i64(&object, "current_state", 0).max(0);
    let mut current_state_name = json_object_string(&object, "current_state_name", "");

    ui.horizontal(|ui| {
        ui.label("Layer Index");
        changed |= ui
            .add(DragValue::new(&mut layer_index).speed(1.0))
            .changed();
        ui.label("Layer Name");
        changed |= ui
            .add(TextEdit::singleline(&mut layer_name).desired_width(120.0))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Layer Weight");
        changed |= ui
            .add(DragValue::new(&mut layer_weight).speed(0.05))
            .changed();
        changed |= ui.checkbox(&mut layer_additive, "Additive").changed();
    });
    ui.horizontal(|ui| {
        ui.label("State Time");
        changed |= ui
            .add(DragValue::new(&mut state_time).speed(0.01))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Current State");
        changed |= ui
            .add(DragValue::new(&mut current_state).speed(1.0))
            .changed();
        ui.label("State Name");
        changed |= ui
            .add(TextEdit::singleline(&mut current_state_name).desired_width(120.0))
            .changed();
    });

    if changed {
        object.insert(
            "layer_index".to_string(),
            JsonValue::Number(JsonNumber::from(layer_index)),
        );
        object.insert("layer_name".to_string(), JsonValue::String(layer_name));
        object.insert("layer_weight".to_string(), json_number(layer_weight));
        object.insert(
            "layer_additive".to_string(),
            JsonValue::Bool(layer_additive),
        );
        object.insert("state_time".to_string(), json_number(state_time));
        object.insert(
            "current_state".to_string(),
            JsonValue::Number(JsonNumber::from(current_state)),
        );
        object.insert(
            "current_state_name".to_string(),
            JsonValue::String(current_state_name),
        );
        if !object.contains_key("states") {
            object.insert("states".to_string(), JsonValue::Array(Vec::new()));
        }
        if !object.contains_key("transitions") {
            object.insert("transitions".to_string(), JsonValue::Array(Vec::new()));
        }
        if !object.contains_key("transition") {
            object.insert("transition".to_string(), JsonValue::Null);
        }
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_input_modifiers_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut shift = json_object_bool(&object, "shift", false);
    let mut ctrl = json_object_bool(&object, "ctrl", false);
    let mut alt = json_object_bool(&object, "alt", false);
    let mut super_key = json_object_bool(&object, "super", false);
    let mut changed = false;

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut shift, "Shift").changed();
        changed |= ui.checkbox(&mut ctrl, "Ctrl").changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut alt, "Alt").changed();
        changed |= ui.checkbox(&mut super_key, "Super").changed();
    });

    if changed {
        object.insert("shift".to_string(), JsonValue::Bool(shift));
        object.insert("ctrl".to_string(), JsonValue::Bool(ctrl));
        object.insert("alt".to_string(), JsonValue::Bool(alt));
        object.insert("super".to_string(), JsonValue::Bool(super_key));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_audio_streaming_config_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut buffer_frames = json_object_i64(&object, "buffer_frames", 8192).max(1);
    let mut chunk_frames = json_object_i64(&object, "chunk_frames", 2048).max(1);

    ui.horizontal(|ui| {
        ui.label("Buffer Frames");
        changed |= ui
            .add(
                DragValue::new(&mut buffer_frames)
                    .speed(1.0)
                    .range(1..=i64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Chunk Frames");
        changed |= ui
            .add(
                DragValue::new(&mut chunk_frames)
                    .speed(1.0)
                    .range(1..=i64::MAX),
            )
            .changed();
    });

    if changed {
        object.insert(
            "buffer_frames".to_string(),
            JsonValue::Number(JsonNumber::from(buffer_frames)),
        );
        object.insert(
            "chunk_frames".to_string(),
            JsonValue::Number(JsonNumber::from(chunk_frames)),
        );
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_spline_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut closed = json_object_bool(&object, "closed", false);
    let mut tension = json_object_f64(&object, "tension", 0.5).clamp(0.0, 1.0);
    let mut mode = json_object_string(&object, "mode", "CatmullRom");
    if !matches!(mode.as_str(), "Linear" | "CatmullRom" | "Bezier") {
        mode = "CatmullRom".to_string();
    }

    let mut points = object
        .get("points")
        .and_then(JsonValue::as_array)
        .cloned()
        .unwrap_or_default();
    let mut remove_index: Option<usize> = None;

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut closed, "Closed").changed();
        ui.label("Mode");
        ComboBox::from_id_salt(("visual_spline_mode_inline", ui.id()))
            .selected_text(mode.clone())
            .show_ui(ui, |ui| {
                for candidate in ["Linear", "CatmullRom", "Bezier"] {
                    changed |= ui
                        .selectable_value(&mut mode, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Tension");
        changed |= ui.add(DragValue::new(&mut tension).speed(0.01)).changed();
    });
    ui.separator();
    ui.label("Points");
    for (index, point) in points.iter_mut().enumerate() {
        let mut components = coerce_json_to_vec3_components(point).unwrap_or((0.0, 0.0, 0.0));
        ui.horizontal(|ui| {
            ui.label(format!("#{}", index));
            let mut row_changed = false;
            row_changed |= ui
                .add(DragValue::new(&mut components.0).speed(0.05))
                .changed();
            row_changed |= ui
                .add(DragValue::new(&mut components.1).speed(0.05))
                .changed();
            row_changed |= ui
                .add(DragValue::new(&mut components.2).speed(0.05))
                .changed();
            if row_changed {
                *point = vec3_json(components.0, components.1, components.2);
                changed = true;
            }
            if ui.small_button("-").clicked() {
                remove_index = Some(index);
            }
        });
    }
    if let Some(index) = remove_index {
        if index < points.len() {
            points.remove(index);
            changed = true;
        }
    }
    if ui.small_button("+ Point").clicked() {
        points.push(vec3_json(0.0, 0.0, 0.0));
        changed = true;
    }

    if changed {
        object.insert("closed".to_string(), JsonValue::Bool(closed));
        object.insert("tension".to_string(), json_number(tension.clamp(0.0, 1.0)));
        object.insert("mode".to_string(), JsonValue::String(mode));
        object.insert("points".to_string(), JsonValue::Array(points));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_character_output_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut desired_translation = json_object_vec3(&object, "desired_translation", (0.0, 0.0, 0.0));
    let mut effective_translation =
        json_object_vec3(&object, "effective_translation", (0.0, 0.0, 0.0));
    let mut remaining_translation =
        json_object_vec3(&object, "remaining_translation", (0.0, 0.0, 0.0));
    let mut grounded = json_object_bool(&object, "grounded", false);
    let mut sliding_down_slope = json_object_bool(&object, "sliding_down_slope", false);
    let mut collision_count = json_object_i64(&object, "collision_count", 0).max(0);
    let mut ground_normal = json_object_vec3(&object, "ground_normal", (0.0, 1.0, 0.0));
    let mut slope_angle = json_object_f64(&object, "slope_angle", 0.0).max(0.0);
    let mut hit_normal = json_object_vec3(&object, "hit_normal", (0.0, 0.0, 0.0));
    let mut hit_point = json_object_vec3(&object, "hit_point", (0.0, 0.0, 0.0));
    let mut stepped_up = json_object_bool(&object, "stepped_up", false);
    let mut step_height = json_object_f64(&object, "step_height", 0.0).max(0.0);
    let mut platform_velocity = json_object_vec3(&object, "platform_velocity", (0.0, 0.0, 0.0));
    let mut hit_entity = json_to_u64(object.get("hit_entity").unwrap_or(&JsonValue::Null));

    changed |= draw_vec3_row(ui, "Desired Translation", &mut desired_translation, 0.05);
    changed |= draw_vec3_row(
        ui,
        "Effective Translation",
        &mut effective_translation,
        0.05,
    );
    changed |= draw_vec3_row(
        ui,
        "Remaining Translation",
        &mut remaining_translation,
        0.05,
    );
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut grounded, "Grounded").changed();
        changed |= ui
            .checkbox(&mut sliding_down_slope, "Sliding Down Slope")
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Collision Count");
        changed |= ui
            .add(DragValue::new(&mut collision_count).speed(1.0))
            .changed();
    });
    changed |= draw_vec3_row(ui, "Ground Normal", &mut ground_normal, 0.05);
    ui.horizontal(|ui| {
        ui.label("Slope Angle");
        changed |= ui
            .add(DragValue::new(&mut slope_angle).speed(0.01))
            .changed();
    });
    changed |= draw_vec3_row(ui, "Hit Normal", &mut hit_normal, 0.05);
    changed |= draw_vec3_row(ui, "Hit Point", &mut hit_point, 0.05);
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut stepped_up, "Stepped Up").changed();
        ui.label("Step Height");
        changed |= ui
            .add(DragValue::new(&mut step_height).speed(0.01))
            .changed();
    });
    changed |= draw_vec3_row(ui, "Platform Velocity", &mut platform_velocity, 0.05);
    ui.horizontal(|ui| {
        ui.label("Hit Entity");
        let mut raw = hit_entity.unwrap_or(0);
        if ui
            .add(DragValue::new(&mut raw).speed(1.0).range(0..=u64::MAX))
            .changed()
        {
            hit_entity = (raw > 0).then_some(raw);
            changed = true;
        }
        if ui.button("Clear").clicked() {
            hit_entity = None;
            changed = true;
        }
    });

    if changed {
        object.insert(
            "desired_translation".to_string(),
            vec3_json(
                desired_translation.0,
                desired_translation.1,
                desired_translation.2,
            ),
        );
        object.insert(
            "effective_translation".to_string(),
            vec3_json(
                effective_translation.0,
                effective_translation.1,
                effective_translation.2,
            ),
        );
        object.insert(
            "remaining_translation".to_string(),
            vec3_json(
                remaining_translation.0,
                remaining_translation.1,
                remaining_translation.2,
            ),
        );
        object.insert("grounded".to_string(), JsonValue::Bool(grounded));
        object.insert(
            "sliding_down_slope".to_string(),
            JsonValue::Bool(sliding_down_slope),
        );
        object.insert(
            "collision_count".to_string(),
            JsonValue::Number(JsonNumber::from(collision_count)),
        );
        object.insert(
            "ground_normal".to_string(),
            vec3_json(ground_normal.0, ground_normal.1, ground_normal.2),
        );
        object.insert("slope_angle".to_string(), json_number(slope_angle));
        object.insert(
            "hit_normal".to_string(),
            vec3_json(hit_normal.0, hit_normal.1, hit_normal.2),
        );
        object.insert(
            "hit_point".to_string(),
            vec3_json(hit_point.0, hit_point.1, hit_point.2),
        );
        object.insert("stepped_up".to_string(), JsonValue::Bool(stepped_up));
        object.insert("step_height".to_string(), json_number(step_height));
        object.insert(
            "platform_velocity".to_string(),
            vec3_json(
                platform_velocity.0,
                platform_velocity.1,
                platform_velocity.2,
            ),
        );
        object.insert(
            "hit_entity".to_string(),
            hit_entity
                .map(|value| JsonValue::Number(JsonNumber::from(value)))
                .unwrap_or(JsonValue::Null),
        );
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_physics_query_filter_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut flags = json_object_i64(&object, "flags", 0).max(0) as u32;
    let mut groups_memberships =
        json_object_i64(&object, "groups_memberships", i64::from(u32::MAX)).max(0) as u32;
    let mut groups_filter =
        json_object_i64(&object, "groups_filter", i64::from(u32::MAX)).max(0) as u32;
    let mut use_groups = json_object_bool(&object, "use_groups", false);

    ui.vertical(|ui| {
        ui.horizontal(|ui| {
            ui.label("Flags");
            let mut flags_i64 = i64::from(flags);
            if ui
                .add(
                    DragValue::new(&mut flags_i64)
                        .speed(1.0)
                        .range(0..=i64::from(u32::MAX)),
                )
                .changed()
            {
                flags = u32::try_from(flags_i64).unwrap_or(0);
                changed = true;
            }
        });
        for (label, bit) in [
            ("No Fixed", PHYSICS_QUERY_FLAG_EXCLUDE_FIXED),
            ("No Kinematic", PHYSICS_QUERY_FLAG_EXCLUDE_KINEMATIC),
            ("No Dynamic", PHYSICS_QUERY_FLAG_EXCLUDE_DYNAMIC),
            ("No Sensors", PHYSICS_QUERY_FLAG_EXCLUDE_SENSORS),
            ("No Solids", PHYSICS_QUERY_FLAG_EXCLUDE_SOLIDS),
        ] {
            let mut enabled = (flags & bit) != 0;
            if ui.checkbox(&mut enabled, label).changed() {
                if enabled {
                    flags |= bit;
                } else {
                    flags &= !bit;
                }
                changed = true;
            }
        }
        ui.horizontal(|ui| {
            ui.label("Groups Memberships");
            let mut value_i64 = i64::from(groups_memberships);
            if ui
                .add(
                    DragValue::new(&mut value_i64)
                        .speed(1.0)
                        .range(0..=i64::from(u32::MAX)),
                )
                .changed()
            {
                groups_memberships = u32::try_from(value_i64).unwrap_or(u32::MAX);
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Groups Filter");
            let mut value_i64 = i64::from(groups_filter);
            if ui
                .add(
                    DragValue::new(&mut value_i64)
                        .speed(1.0)
                        .range(0..=i64::from(u32::MAX)),
                )
                .changed()
            {
                groups_filter = u32::try_from(value_i64).unwrap_or(u32::MAX);
                changed = true;
            }
        });
        if ui.checkbox(&mut use_groups, "Use Groups").changed() {
            changed = true;
        }
    });

    if changed {
        object.insert(
            "flags".to_string(),
            JsonValue::Number(JsonNumber::from(flags)),
        );
        object.insert(
            "groups_memberships".to_string(),
            JsonValue::Number(JsonNumber::from(groups_memberships)),
        );
        object.insert(
            "groups_filter".to_string(),
            JsonValue::Number(JsonNumber::from(groups_filter)),
        );
        object.insert("use_groups".to_string(), JsonValue::Bool(use_groups));
        *value = compact_json_string(&JsonValue::Object(object));
    }
    changed
}

fn draw_ray_cast_hit_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut has_hit = json_object_bool(&object, "has_hit", false);
    let mut hit_entity = json_object_i64(&object, "hit_entity", -1);
    let mut point = json_object_vec3(&object, "point", (0.0, 0.0, 0.0));
    let mut normal = json_object_vec3(&object, "normal", (0.0, 1.0, 0.0));
    let mut toi = json_object_f64(&object, "toi", 0.0).max(0.0);

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut has_hit, "Has Hit").changed();
        ui.label("Hit Entity");
        changed |= ui.add(DragValue::new(&mut hit_entity).speed(1.0)).changed();
    });
    changed |= draw_vec3_row(ui, "Point", &mut point, 0.05);
    changed |= draw_vec3_row(ui, "Normal", &mut normal, 0.05);
    ui.horizontal(|ui| {
        ui.label("TOI");
        changed |= ui.add(DragValue::new(&mut toi).speed(0.05)).changed();
    });

    if changed {
        object.insert("has_hit".to_string(), JsonValue::Bool(has_hit));
        if hit_entity >= 0 {
            object.insert(
                "hit_entity".to_string(),
                JsonValue::Number(JsonNumber::from(hit_entity)),
            );
        } else {
            object.insert("hit_entity".to_string(), JsonValue::Null);
        }
        object.insert("point".to_string(), vec3_json(point.0, point.1, point.2));
        object.insert(
            "normal".to_string(),
            vec3_json(normal.0, normal.1, normal.2),
        );
        object.insert("toi".to_string(), json_number(toi.max(0.0)));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_point_projection_hit_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut has_hit = json_object_bool(&object, "has_hit", false);
    let mut hit_entity = json_object_i64(&object, "hit_entity", -1);
    let mut projected_point = json_object_vec3(&object, "projected_point", (0.0, 0.0, 0.0));
    let mut is_inside = json_object_bool(&object, "is_inside", false);
    let mut distance = json_object_f64(&object, "distance", 0.0).max(0.0);

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut has_hit, "Has Hit").changed();
        ui.label("Hit Entity");
        changed |= ui.add(DragValue::new(&mut hit_entity).speed(1.0)).changed();
    });
    changed |= draw_vec3_row(ui, "Projected Point", &mut projected_point, 0.05);
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut is_inside, "Inside").changed();
        ui.label("Distance");
        changed |= ui.add(DragValue::new(&mut distance).speed(0.05)).changed();
    });

    if changed {
        object.insert("has_hit".to_string(), JsonValue::Bool(has_hit));
        if hit_entity >= 0 {
            object.insert(
                "hit_entity".to_string(),
                JsonValue::Number(JsonNumber::from(hit_entity)),
            );
        } else {
            object.insert("hit_entity".to_string(), JsonValue::Null);
        }
        object.insert(
            "projected_point".to_string(),
            vec3_json(projected_point.0, projected_point.1, projected_point.2),
        );
        object.insert("is_inside".to_string(), JsonValue::Bool(is_inside));
        object.insert("distance".to_string(), json_number(distance.max(0.0)));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_shape_cast_hit_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut has_hit = json_object_bool(&object, "has_hit", false);
    let mut hit_entity = json_object_i64(&object, "hit_entity", -1);
    let mut toi = json_object_f64(&object, "toi", 0.0).max(0.0);
    let mut witness1 = json_object_vec3(&object, "witness1", (0.0, 0.0, 0.0));
    let mut witness2 = json_object_vec3(&object, "witness2", (0.0, 0.0, 0.0));
    let mut normal1 = json_object_vec3(&object, "normal1", (0.0, 1.0, 0.0));
    let mut normal2 = json_object_vec3(&object, "normal2", (0.0, 1.0, 0.0));
    let mut status = json_object_string(&object, "status", "Unknown");
    if status.is_empty() {
        status = "Unknown".to_string();
    }

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut has_hit, "Has Hit").changed();
        ui.label("Hit Entity");
        changed |= ui.add(DragValue::new(&mut hit_entity).speed(1.0)).changed();
        ui.label("TOI");
        changed |= ui.add(DragValue::new(&mut toi).speed(0.05)).changed();
    });
    changed |= draw_vec3_row(ui, "Witness 1", &mut witness1, 0.05);
    changed |= draw_vec3_row(ui, "Witness 2", &mut witness2, 0.05);
    changed |= draw_vec3_row(ui, "Normal 1", &mut normal1, 0.05);
    changed |= draw_vec3_row(ui, "Normal 2", &mut normal2, 0.05);
    ui.horizontal(|ui| {
        ui.label("Status");
        changed |= ui
            .add(TextEdit::singleline(&mut status).desired_width(120.0))
            .changed();
    });

    if changed {
        object.insert("has_hit".to_string(), JsonValue::Bool(has_hit));
        if hit_entity >= 0 {
            object.insert(
                "hit_entity".to_string(),
                JsonValue::Number(JsonNumber::from(hit_entity)),
            );
        } else {
            object.insert("hit_entity".to_string(), JsonValue::Null);
        }
        object.insert("toi".to_string(), json_number(toi.max(0.0)));
        object.insert(
            "witness1".to_string(),
            vec3_json(witness1.0, witness1.1, witness1.2),
        );
        object.insert(
            "witness2".to_string(),
            vec3_json(witness2.0, witness2.1, witness2.2),
        );
        object.insert(
            "normal1".to_string(),
            vec3_json(normal1.0, normal1.1, normal1.2),
        );
        object.insert(
            "normal2".to_string(),
            vec3_json(normal2.0, normal2.1, normal2.2),
        );
        object.insert("status".to_string(), JsonValue::String(status));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_json_array_editor(
    ui: &mut Ui,
    label: &str,
    values: &mut Vec<JsonValue>,
    id_prefix: &str,
) -> bool {
    let mut changed = false;
    if values.iter().all(JsonValue::is_number) {
        ui.horizontal(|ui| {
            ui.label(label);
            for entry in values.iter_mut() {
                let mut number = entry.as_f64().unwrap_or(0.0);
                if ui.add(DragValue::new(&mut number).speed(0.05)).changed() {
                    *entry = json_number(number);
                    changed = true;
                }
            }
        });
        return changed;
    }
    if values.iter().all(JsonValue::is_boolean) {
        ui.horizontal(|ui| {
            ui.label(label);
            for entry in values.iter_mut() {
                let mut flag = entry.as_bool().unwrap_or(false);
                if ui.checkbox(&mut flag, "").changed() {
                    *entry = JsonValue::Bool(flag);
                    changed = true;
                }
            }
        });
        return changed;
    }
    if values.iter().all(JsonValue::is_string) {
        ui.vertical(|ui| {
            ui.label(label);
            for (index, entry) in values.iter_mut().enumerate() {
                let mut text = entry.as_str().unwrap_or_default().to_string();
                ui.horizontal(|ui| {
                    ui.label(format!("#{index}"));
                    if ui
                        .add(TextEdit::singleline(&mut text).desired_width(120.0))
                        .changed()
                    {
                        *entry = JsonValue::String(text.clone());
                        changed = true;
                    }
                });
            }
        });
        return changed;
    }

    ui.collapsing(label, |ui| {
        for (index, entry) in values.iter_mut().enumerate() {
            let child_label = format!("[{index}]");
            changed |= draw_json_value_editor(ui, &child_label, entry, id_prefix);
        }
    });
    changed
}

fn draw_json_object_tree_editor(
    ui: &mut Ui,
    object: &mut JsonMap<String, JsonValue>,
    id_prefix: &str,
) -> bool {
    let mut changed = false;
    let mut keys = object.keys().cloned().collect::<Vec<_>>();
    keys.retain(|key| !key.starts_with('_'));
    keys.sort();
    for key in keys {
        if let Some(value) = object.get_mut(&key) {
            changed |= draw_json_value_editor(ui, &key, value, id_prefix);
        }
    }
    changed
}

fn draw_json_value_editor(
    ui: &mut Ui,
    label: &str,
    value: &mut JsonValue,
    id_prefix: &str,
) -> bool {
    match value {
        JsonValue::Bool(flag) => {
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label(label);
                changed |= ui.checkbox(flag, "").changed();
            });
            changed
        }
        JsonValue::Number(number) => {
            let mut changed = false;
            let mut current = number.as_f64().unwrap_or(0.0);
            ui.horizontal(|ui| {
                ui.label(label);
                changed |= ui.add(DragValue::new(&mut current).speed(0.05)).changed();
            });
            if changed {
                *value = json_number(current);
            }
            changed
        }
        JsonValue::String(text) => {
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label(label);
                changed |= ui
                    .add(TextEdit::singleline(text).desired_width(140.0))
                    .changed();
            });
            changed
        }
        JsonValue::Array(values) => draw_json_array_editor(ui, label, values, id_prefix),
        JsonValue::Object(object) => {
            let mut changed = false;
            let collapse_id = (id_prefix, label);
            ui.push_id(collapse_id, |ui| {
                ui.collapsing(label, |ui| {
                    changed |= draw_json_object_tree_editor(ui, object, id_prefix);
                });
            });
            changed
        }
        JsonValue::Null => {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.label("null");
            });
            false
        }
    }
}

fn default_patch_value_for_schema(schema_value: &JsonValue) -> JsonValue {
    match schema_value {
        JsonValue::Object(_) => JsonValue::Object(JsonMap::new()),
        _ => schema_value.clone(),
    }
}

fn prune_patch_unknown_fields(
    patch: &mut JsonMap<String, JsonValue>,
    schema: &JsonMap<String, JsonValue>,
) -> bool {
    let mut changed = false;
    let keys = patch.keys().cloned().collect::<Vec<_>>();
    for key in keys {
        if key.starts_with('_') {
            patch.remove(&key);
            changed = true;
            continue;
        }
        let Some(schema_value) = schema.get(&key) else {
            patch.remove(&key);
            changed = true;
            continue;
        };
        if let JsonValue::Object(schema_object) = schema_value {
            match patch.get_mut(&key) {
                Some(JsonValue::Object(patch_object)) => {
                    changed |= prune_patch_unknown_fields(patch_object, schema_object);
                }
                Some(_) | None => {}
            }
        }
    }
    changed
}

fn draw_schema_patch_object_editor(
    ui: &mut Ui,
    patch: &mut JsonMap<String, JsonValue>,
    schema: &JsonMap<String, JsonValue>,
    id_prefix: &str,
    path: &str,
) -> bool {
    let mut changed = false;
    let mut keys = schema
        .keys()
        .filter(|key| !key.starts_with('_'))
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();

    for key in keys {
        let Some(schema_value) = schema.get(&key) else {
            continue;
        };

        let mut include = patch.contains_key(&key);
        let had_include = include;
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut include, "").changed();
            ui.label(key.as_str());
        });

        if include {
            if !had_include {
                patch.insert(key.clone(), default_patch_value_for_schema(schema_value));
                changed = true;
            }

            let field_path = format!("{}.{}", path, key);
            if let Some(current) = patch.get(&key).cloned() {
                let normalized = coerce_json_to_schema_value(&current, schema_value, &field_path)
                    .unwrap_or_else(|_| default_patch_value_for_schema(schema_value));
                if normalized != current {
                    patch.insert(key.clone(), normalized);
                    changed = true;
                }
            }

            ui.indent((id_prefix, path, key.as_str()), |ui| {
                if let JsonValue::Object(schema_object) = schema_value {
                    if let Some(JsonValue::Object(child_patch)) = patch.get_mut(&key) {
                        changed |= draw_schema_patch_object_editor(
                            ui,
                            child_patch,
                            schema_object,
                            id_prefix,
                            &field_path,
                        );
                    }
                } else if let Some(entry) = patch.get_mut(&key) {
                    changed |= draw_json_value_editor(ui, "value", entry, id_prefix);
                }
            });
        } else if had_include {
            patch.remove(&key);
            changed = true;
        }
    }

    changed
}

fn draw_schema_patch_editor(
    ui: &mut Ui,
    value: &mut String,
    id_prefix: &str,
    type_name: &str,
    schema_literal: &str,
) -> bool {
    let schema = parse_schema_object_literal(schema_literal);
    let mut object = parse_json_object_literal(value);
    let mut changed = prune_patch_unknown_fields(&mut object, &schema);

    ui.horizontal(|ui| {
        if ui.small_button("Clear Patch").clicked() {
            object.clear();
            changed = true;
        }
        ui.small(format!("{} field(s) enabled", object.len()));
    });
    ui.small(format!(
        "{}: enable only the fields you want to update",
        type_name
    ));

    changed |= draw_schema_patch_object_editor(ui, &mut object, &schema, id_prefix, type_name);
    if changed {
        *value = compact_json_string(&JsonValue::Object(object));
    }
    changed
}

fn draw_runtime_tuning_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    object.remove("pending_asset_uploads");
    object.remove("pending_asset_bytes");
    let mut changed = false;

    let mut render_message_capacity =
        json_object_f64(&object, "render_message_capacity", 96.0).max(0.0) as u64;
    let mut asset_stream_queue_capacity =
        json_object_f64(&object, "asset_stream_queue_capacity", 96.0).max(0.0) as u64;
    let mut asset_worker_queue_capacity =
        json_object_f64(&object, "asset_worker_queue_capacity", 96.0).max(0.0) as u64;
    let mut max_pending_asset_uploads =
        json_object_f64(&object, "max_pending_asset_uploads", 48.0).max(0.0) as u64;
    let mut max_pending_asset_bytes =
        json_object_f64(&object, "max_pending_asset_bytes", 512.0 * 1024.0 * 1024.0).max(0.0)
            as u64;
    let mut asset_uploads_per_frame =
        json_object_f64(&object, "asset_uploads_per_frame", 8.0).max(0.0) as u64;

    let mut wgpu_poll_interval_frames =
        json_object_f64(&object, "wgpu_poll_interval_frames", 1.0).max(0.0) as u32;
    let mut wgpu_poll_mode = json_object_f64(&object, "wgpu_poll_mode", 1.0).max(0.0) as u32;
    let mut pixels_per_line = json_object_f64(&object, "pixels_per_line", 38.0).max(0.0) as u32;
    let mut title_update_ms = json_object_f64(&object, "title_update_ms", 200.0).max(0.0) as u32;
    let mut resize_debounce_ms =
        json_object_f64(&object, "resize_debounce_ms", 500.0).max(0.0) as u32;
    let mut max_logic_steps_per_frame =
        json_object_f64(&object, "max_logic_steps_per_frame", 4.0).max(0.0) as u32;

    let mut target_tickrate = json_object_f64(&object, "target_tickrate", 120.0).max(1.0);
    let mut target_fps = json_object_f64(&object, "target_fps", 0.0).max(0.0);

    let pending_asset_uploads =
        json_object_f64(&object, "pending_asset_uploads", 0.0).max(0.0) as u64;
    let pending_asset_bytes = json_object_f64(&object, "pending_asset_bytes", 0.0).max(0.0) as u64;

    if wgpu_poll_mode > 2 {
        wgpu_poll_mode = 1;
    }

    ui.horizontal(|ui| {
        ui.label("Render Message Capacity");
        changed |= ui
            .add(
                DragValue::new(&mut render_message_capacity)
                    .speed(1.0)
                    .range(0..=u64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Asset Stream Queue Capacity");
        changed |= ui
            .add(
                DragValue::new(&mut asset_stream_queue_capacity)
                    .speed(1.0)
                    .range(0..=u64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Asset Worker Queue Capacity");
        changed |= ui
            .add(
                DragValue::new(&mut asset_worker_queue_capacity)
                    .speed(1.0)
                    .range(0..=u64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Max Pending Uploads");
        changed |= ui
            .add(
                DragValue::new(&mut max_pending_asset_uploads)
                    .speed(1.0)
                    .range(0..=u64::MAX),
            )
            .changed();
        ui.label("Uploads Per Frame");
        changed |= ui
            .add(
                DragValue::new(&mut asset_uploads_per_frame)
                    .speed(1.0)
                    .range(0..=u64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Max Pending Bytes");
        changed |= ui
            .add(
                DragValue::new(&mut max_pending_asset_bytes)
                    .speed(1024.0)
                    .range(0..=u64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Poll Interval (frames)");
        changed |= ui
            .add(
                DragValue::new(&mut wgpu_poll_interval_frames)
                    .speed(1.0)
                    .range(0..=u32::MAX),
            )
            .changed();
        ui.label("Poll Mode");
        ComboBox::from_id_salt(("runtime_tuning_poll_mode", ui.id()))
            .selected_text(match wgpu_poll_mode {
                0 => "Off",
                2 => "Wait",
                _ => "Poll",
            })
            .show_ui(ui, |ui| {
                changed |= ui.selectable_value(&mut wgpu_poll_mode, 0, "Off").changed();
                changed |= ui
                    .selectable_value(&mut wgpu_poll_mode, 1, "Poll")
                    .changed();
                changed |= ui
                    .selectable_value(&mut wgpu_poll_mode, 2, "Wait")
                    .changed();
            });
    });
    ui.horizontal(|ui| {
        ui.label("Pixels Per Line");
        changed |= ui
            .add(
                DragValue::new(&mut pixels_per_line)
                    .speed(1.0)
                    .range(0..=u32::MAX),
            )
            .changed();
        ui.label("Title Update ms");
        changed |= ui
            .add(
                DragValue::new(&mut title_update_ms)
                    .speed(1.0)
                    .range(0..=u32::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Resize Debounce ms");
        changed |= ui
            .add(
                DragValue::new(&mut resize_debounce_ms)
                    .speed(1.0)
                    .range(0..=u32::MAX),
            )
            .changed();
        ui.label("Max Logic Steps/Frame");
        changed |= ui
            .add(
                DragValue::new(&mut max_logic_steps_per_frame)
                    .speed(1.0)
                    .range(0..=u32::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Target Tickrate");
        changed |= ui
            .add(
                DragValue::new(&mut target_tickrate)
                    .speed(1.0)
                    .range(1.0..=10_000.0),
            )
            .changed();
        ui.label("Target FPS (0 = uncapped)");
        changed |= ui
            .add(
                DragValue::new(&mut target_fps)
                    .speed(1.0)
                    .range(0.0..=10_000.0),
            )
            .changed();
    });

    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Pending Uploads");
        ui.monospace(pending_asset_uploads.to_string());
        ui.label("Pending Bytes");
        ui.monospace(pending_asset_bytes.to_string());
    });

    if changed {
        object.insert(
            "render_message_capacity".to_string(),
            JsonValue::Number(JsonNumber::from(render_message_capacity)),
        );
        object.insert(
            "asset_stream_queue_capacity".to_string(),
            JsonValue::Number(JsonNumber::from(asset_stream_queue_capacity)),
        );
        object.insert(
            "asset_worker_queue_capacity".to_string(),
            JsonValue::Number(JsonNumber::from(asset_worker_queue_capacity)),
        );
        object.insert(
            "max_pending_asset_uploads".to_string(),
            JsonValue::Number(JsonNumber::from(max_pending_asset_uploads)),
        );
        object.insert(
            "max_pending_asset_bytes".to_string(),
            JsonValue::Number(JsonNumber::from(max_pending_asset_bytes)),
        );
        object.insert(
            "asset_uploads_per_frame".to_string(),
            JsonValue::Number(JsonNumber::from(asset_uploads_per_frame)),
        );
        object.insert(
            "wgpu_poll_interval_frames".to_string(),
            JsonValue::Number(JsonNumber::from(wgpu_poll_interval_frames)),
        );
        object.insert(
            "wgpu_poll_mode".to_string(),
            JsonValue::Number(JsonNumber::from(wgpu_poll_mode)),
        );
        object.insert(
            "pixels_per_line".to_string(),
            JsonValue::Number(JsonNumber::from(pixels_per_line)),
        );
        object.insert(
            "title_update_ms".to_string(),
            JsonValue::Number(JsonNumber::from(title_update_ms)),
        );
        object.insert(
            "resize_debounce_ms".to_string(),
            JsonValue::Number(JsonNumber::from(resize_debounce_ms)),
        );
        object.insert(
            "max_logic_steps_per_frame".to_string(),
            JsonValue::Number(JsonNumber::from(max_logic_steps_per_frame)),
        );
        object.insert("target_tickrate".to_string(), json_number(target_tickrate));
        object.insert("target_fps".to_string(), json_number(target_fps));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_runtime_config_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut egui_enabled = json_object_bool(&object, "egui", true);
    let mut wgpu_experimental_features =
        json_object_bool(&object, "wgpu_experimental_features", false);
    let mut fixed_timestep = json_object_bool(&object, "fixed_timestep", false);
    let mut wgpu_backend = json_object_string(&object, "wgpu_backend", "auto");
    let mut binding_backend = json_object_string(&object, "binding_backend", "auto");

    if !matches!(
        wgpu_backend.as_str(),
        "auto" | "vulkan" | "dx12" | "metal" | "gl"
    ) {
        wgpu_backend = "auto".to_string();
    }
    if !matches!(
        binding_backend.as_str(),
        "auto" | "bindless_modern" | "bindless_fallback" | "bind_groups"
    ) {
        binding_backend = "auto".to_string();
    }

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut egui_enabled, "Egui").changed();
        changed |= ui
            .checkbox(&mut wgpu_experimental_features, "Experimental Features")
            .changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut fixed_timestep, "Fixed Timestep").changed();
    });
    ui.horizontal(|ui| {
        ui.label("WGPU Backend");
        ComboBox::from_id_salt(("runtime_config_backend", ui.id()))
            .selected_text(wgpu_backend.clone())
            .show_ui(ui, |ui| {
                for candidate in ["auto", "vulkan", "dx12", "metal", "gl"] {
                    changed |= ui
                        .selectable_value(&mut wgpu_backend, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Binding Backend");
        ComboBox::from_id_salt(("runtime_config_binding_backend", ui.id()))
            .selected_text(binding_backend.clone())
            .show_ui(ui, |ui| {
                for candidate in [
                    "auto",
                    "bindless_modern",
                    "bindless_fallback",
                    "bind_groups",
                ] {
                    changed |= ui
                        .selectable_value(&mut binding_backend, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });

    if changed {
        object.insert("egui".to_string(), JsonValue::Bool(egui_enabled));
        object.insert(
            "wgpu_experimental_features".to_string(),
            JsonValue::Bool(wgpu_experimental_features),
        );
        object.insert(
            "fixed_timestep".to_string(),
            JsonValue::Bool(fixed_timestep),
        );
        object.insert("wgpu_backend".to_string(), JsonValue::String(wgpu_backend));
        object.insert(
            "binding_backend".to_string(),
            JsonValue::String(binding_backend),
        );
        *value = compact_json_string(&JsonValue::Object(object));
    }
    changed
}

fn draw_render_config_editor(ui: &mut Ui, value: &mut String) -> bool {
    draw_schema_patch_editor(
        ui,
        value,
        "render_config_patch",
        "Render Config",
        default_literal_for_type(VisualValueType::RenderConfig),
    )
}

fn draw_shader_constants_editor(ui: &mut Ui, value: &mut String) -> bool {
    draw_schema_patch_editor(
        ui,
        value,
        "shader_constants_patch",
        "Shader Constants",
        default_literal_for_type(VisualValueType::ShaderConstants),
    )
}

fn draw_streaming_tuning_editor(ui: &mut Ui, value: &mut String) -> bool {
    draw_schema_patch_editor(
        ui,
        value,
        "streaming_tuning_patch",
        "Streaming Tuning",
        default_literal_for_type(VisualValueType::StreamingTuning),
    )
}

fn draw_render_passes_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut gbuffer = json_object_bool(&object, "gbuffer", true);
    let mut shadow = json_object_bool(&object, "shadow", true);
    let mut direct_lighting = json_object_bool(&object, "direct_lighting", true);
    let mut sky = json_object_bool(&object, "sky", true);
    let mut ssgi = json_object_bool(&object, "ssgi", true);
    let mut ssgi_denoise = json_object_bool(&object, "ssgi_denoise", true);
    let mut ssr = json_object_bool(&object, "ssr", true);
    let mut ddgi = json_object_bool(&object, "ddgi", true);
    let mut egui_enabled = json_object_bool(&object, "egui", true);
    let mut gizmo = json_object_bool(&object, "gizmo", true);
    let mut transparent = json_object_bool(&object, "transparent", true);
    let mut occlusion = json_object_bool(&object, "occlusion", true);

    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut gbuffer, "GBuffer").changed();
        changed |= ui.checkbox(&mut shadow, "Shadow").changed();
        changed |= ui.checkbox(&mut direct_lighting, "Direct").changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut sky, "Sky").changed();
        changed |= ui.checkbox(&mut ssgi, "SSGI").changed();
        changed |= ui.checkbox(&mut ssgi_denoise, "SSGI Denoise").changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut ssr, "SSR").changed();
        changed |= ui.checkbox(&mut ddgi, "DDGI").changed();
        changed |= ui.checkbox(&mut egui_enabled, "Egui").changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut gizmo, "Gizmo").changed();
        changed |= ui.checkbox(&mut transparent, "Transparent").changed();
        changed |= ui.checkbox(&mut occlusion, "Occlusion").changed();
    });

    if changed {
        object.insert("gbuffer".to_string(), JsonValue::Bool(gbuffer));
        object.insert("shadow".to_string(), JsonValue::Bool(shadow));
        object.insert(
            "direct_lighting".to_string(),
            JsonValue::Bool(direct_lighting),
        );
        object.insert("sky".to_string(), JsonValue::Bool(sky));
        object.insert("ssgi".to_string(), JsonValue::Bool(ssgi));
        object.insert("ssgi_denoise".to_string(), JsonValue::Bool(ssgi_denoise));
        object.insert("ssr".to_string(), JsonValue::Bool(ssr));
        object.insert("ddgi".to_string(), JsonValue::Bool(ddgi));
        object.insert("egui".to_string(), JsonValue::Bool(egui_enabled));
        object.insert("gizmo".to_string(), JsonValue::Bool(gizmo));
        object.insert("transparent".to_string(), JsonValue::Bool(transparent));
        object.insert("occlusion".to_string(), JsonValue::Bool(occlusion));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_gpu_budget_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    object.remove("used_mib");
    let mut changed = false;

    let mut soft_mib = json_object_f64(&object, "soft_mib", 0.0).max(0.0);
    let mut hard_mib = json_object_f64(&object, "hard_mib", 0.0).max(0.0);
    let used_mib = json_object_f64(&object, "used_mib", 0.0).max(0.0);
    let mut idle_frames = json_object_i64(&object, "idle_frames", 0).max(0) as u32;

    let mut kind_limits = [
        (
            "mesh_soft_mib",
            json_object_f64(&object, "mesh_soft_mib", 0.0).max(0.0),
        ),
        (
            "mesh_hard_mib",
            json_object_f64(&object, "mesh_hard_mib", 0.0).max(0.0),
        ),
        (
            "material_soft_mib",
            json_object_f64(&object, "material_soft_mib", 0.0).max(0.0),
        ),
        (
            "material_hard_mib",
            json_object_f64(&object, "material_hard_mib", 0.0).max(0.0),
        ),
        (
            "texture_soft_mib",
            json_object_f64(&object, "texture_soft_mib", 0.0).max(0.0),
        ),
        (
            "texture_hard_mib",
            json_object_f64(&object, "texture_hard_mib", 0.0).max(0.0),
        ),
        (
            "texture_view_soft_mib",
            json_object_f64(&object, "texture_view_soft_mib", 0.0).max(0.0),
        ),
        (
            "texture_view_hard_mib",
            json_object_f64(&object, "texture_view_hard_mib", 0.0).max(0.0),
        ),
        (
            "sampler_soft_mib",
            json_object_f64(&object, "sampler_soft_mib", 0.0).max(0.0),
        ),
        (
            "sampler_hard_mib",
            json_object_f64(&object, "sampler_hard_mib", 0.0).max(0.0),
        ),
        (
            "buffer_soft_mib",
            json_object_f64(&object, "buffer_soft_mib", 0.0).max(0.0),
        ),
        (
            "buffer_hard_mib",
            json_object_f64(&object, "buffer_hard_mib", 0.0).max(0.0),
        ),
        (
            "external_soft_mib",
            json_object_f64(&object, "external_soft_mib", 0.0).max(0.0),
        ),
        (
            "external_hard_mib",
            json_object_f64(&object, "external_hard_mib", 0.0).max(0.0),
        ),
        (
            "transient_soft_mib",
            json_object_f64(&object, "transient_soft_mib", 0.0).max(0.0),
        ),
        (
            "transient_hard_mib",
            json_object_f64(&object, "transient_hard_mib", 0.0).max(0.0),
        ),
    ];

    ui.horizontal(|ui| {
        ui.label("Used MiB");
        ui.monospace(format!("{used_mib:.2}"));
    });
    ui.horizontal(|ui| {
        ui.label("Soft MiB");
        changed |= ui
            .add(
                DragValue::new(&mut soft_mib)
                    .speed(1.0)
                    .range(0.0..=f64::MAX),
            )
            .changed();
        ui.label("Hard MiB");
        changed |= ui
            .add(
                DragValue::new(&mut hard_mib)
                    .speed(1.0)
                    .range(0.0..=f64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Idle Frames (0 = keep)");
        changed |= ui
            .add(
                DragValue::new(&mut idle_frames)
                    .speed(1.0)
                    .range(0..=u32::MAX),
            )
            .changed();
    });

    for chunk in kind_limits.chunks_mut(2) {
        ui.horizontal(|ui| {
            for (key, number) in chunk {
                let label = key
                    .replace("_soft_mib", " soft")
                    .replace("_hard_mib", " hard")
                    .replace('_', " ");
                ui.label(label);
                changed |= ui
                    .add(DragValue::new(number).speed(1.0).range(0.0..=f64::MAX))
                    .changed();
            }
        });
    }

    if hard_mib < soft_mib {
        hard_mib = soft_mib;
    }
    for chunk in kind_limits.chunks_exact_mut(2) {
        let soft = chunk[0].1;
        if chunk[1].1 < soft {
            chunk[1].1 = soft;
        }
    }

    if changed {
        object.insert("soft_mib".to_string(), json_number(soft_mib));
        object.insert("hard_mib".to_string(), json_number(hard_mib));
        object.insert(
            "idle_frames".to_string(),
            JsonValue::Number(JsonNumber::from(idle_frames)),
        );
        for (key, number) in kind_limits {
            object.insert(key.to_string(), json_number(number));
        }
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_asset_budgets_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;

    let mut mesh_mib = json_object_f64(&object, "mesh_mib", 0.0).max(0.0);
    let mut texture_mib = json_object_f64(&object, "texture_mib", 0.0).max(0.0);
    let mut material_mib = json_object_f64(&object, "material_mib", 0.0).max(0.0);
    let mut audio_mib = json_object_f64(&object, "audio_mib", 0.0).max(0.0);
    let mut scene_mib = json_object_f64(&object, "scene_mib", 0.0).max(0.0);

    ui.horizontal(|ui| {
        ui.label("Mesh MiB");
        changed |= ui
            .add(
                DragValue::new(&mut mesh_mib)
                    .speed(1.0)
                    .range(0.0..=f64::MAX),
            )
            .changed();
        ui.label("Texture MiB");
        changed |= ui
            .add(
                DragValue::new(&mut texture_mib)
                    .speed(1.0)
                    .range(0.0..=f64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Material MiB");
        changed |= ui
            .add(
                DragValue::new(&mut material_mib)
                    .speed(1.0)
                    .range(0.0..=f64::MAX),
            )
            .changed();
        ui.label("Audio MiB");
        changed |= ui
            .add(
                DragValue::new(&mut audio_mib)
                    .speed(1.0)
                    .range(0.0..=f64::MAX),
            )
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Scene MiB");
        changed |= ui
            .add(
                DragValue::new(&mut scene_mib)
                    .speed(1.0)
                    .range(0.0..=f64::MAX),
            )
            .changed();
    });

    if changed {
        object.insert("mesh_mib".to_string(), json_number(mesh_mib));
        object.insert("texture_mib".to_string(), json_number(texture_mib));
        object.insert("material_mib".to_string(), json_number(material_mib));
        object.insert("audio_mib".to_string(), json_number(audio_mib));
        object.insert("scene_mib".to_string(), json_number(scene_mib));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_window_settings_editor(ui: &mut Ui, value: &mut String) -> bool {
    let mut object = parse_json_object_literal(value);
    let mut changed = false;
    let mut title_mode = json_object_string(&object, "title_mode", "stats");
    let mut custom_title = json_object_string(&object, "custom_title", "helmer engine");
    let mut fullscreen = json_object_bool(&object, "fullscreen", false);
    let mut resizable = json_object_bool(&object, "resizable", true);
    let mut decorations = json_object_bool(&object, "decorations", true);
    let mut maximized = json_object_bool(&object, "maximized", false);
    let mut minimized = json_object_bool(&object, "minimized", false);
    let mut visible = json_object_bool(&object, "visible", true);

    if !matches!(
        title_mode.as_str(),
        "stats" | "custom" | "custom_with_stats"
    ) {
        title_mode = "stats".to_string();
    }

    ui.horizontal(|ui| {
        ui.label("Title Mode");
        ComboBox::from_id_salt(("window_title_mode", ui.id()))
            .selected_text(title_mode.clone())
            .show_ui(ui, |ui| {
                for candidate in ["stats", "custom", "custom_with_stats"] {
                    changed |= ui
                        .selectable_value(&mut title_mode, candidate.to_string(), candidate)
                        .changed();
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Custom Title");
        changed |= ui
            .add(TextEdit::singleline(&mut custom_title).desired_width(180.0))
            .changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut fullscreen, "Fullscreen").changed();
        changed |= ui.checkbox(&mut resizable, "Resizable").changed();
        changed |= ui.checkbox(&mut decorations, "Decorations").changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut maximized, "Maximized").changed();
        changed |= ui.checkbox(&mut minimized, "Minimized").changed();
        changed |= ui.checkbox(&mut visible, "Visible").changed();
    });

    if changed {
        object.insert("title_mode".to_string(), JsonValue::String(title_mode));
        object.insert("custom_title".to_string(), JsonValue::String(custom_title));
        object.insert("fullscreen".to_string(), JsonValue::Bool(fullscreen));
        object.insert("resizable".to_string(), JsonValue::Bool(resizable));
        object.insert("decorations".to_string(), JsonValue::Bool(decorations));
        object.insert("maximized".to_string(), JsonValue::Bool(maximized));
        object.insert("minimized".to_string(), JsonValue::Bool(minimized));
        object.insert("visible".to_string(), JsonValue::Bool(visible));
        *value = compact_json_string(&JsonValue::Object(object));
    }

    changed
}

fn draw_typed_default_editor(ui: &mut Ui, value_type: VisualValueType, value: &mut String) -> bool {
    draw_typed_default_editor_with_array_item(ui, value_type, None, value)
}

fn draw_typed_default_editor_with_array_item(
    ui: &mut Ui,
    value_type: VisualValueType,
    array_item_type: Option<VisualValueType>,
    value: &mut String,
) -> bool {
    draw_typed_editor_with_width(ui, value_type, array_item_type, value, 140.0)
}

fn draw_typed_pin_input_editor(
    ui: &mut Ui,
    value_type: VisualValueType,
    value: &mut String,
) -> bool {
    draw_typed_pin_input_editor_with_array_item(ui, value_type, None, value)
}

fn draw_typed_pin_input_editor_with_array_item(
    ui: &mut Ui,
    value_type: VisualValueType,
    array_item_type: Option<VisualValueType>,
    value: &mut String,
) -> bool {
    draw_typed_editor_with_width(ui, value_type, array_item_type, value, 112.0)
}

fn draw_typed_editor_with_width(
    ui: &mut Ui,
    value_type: VisualValueType,
    array_item_type: Option<VisualValueType>,
    value: &mut String,
    text_width: f32,
) -> bool {
    match value_type {
        VisualValueType::Bool => {
            let mut parsed = is_truthy(&parse_loose_literal(value));
            let response = ui.checkbox(&mut parsed, "");
            if response.changed() {
                *value = if parsed {
                    "true".to_string()
                } else {
                    "false".to_string()
                };
            }
            response.changed()
        }
        VisualValueType::Number => {
            let mut parsed = value.trim().parse::<f64>().unwrap_or(0.0);
            let response = ui.add(DragValue::new(&mut parsed).speed(0.1));
            if response.changed() {
                *value = parsed.to_string();
            }
            response.changed()
        }
        VisualValueType::String => ui
            .add(TextEdit::singleline(value).desired_width(text_width))
            .changed(),
        VisualValueType::Entity => {
            let mut parsed = value.trim().parse::<u64>().unwrap_or(0);
            let response = ui.add(DragValue::new(&mut parsed).speed(1));
            if response.changed() {
                *value = parsed.to_string();
            }
            response.changed()
        }
        VisualValueType::Array => {
            let mut item_type = array_item_type;
            normalize_array_item_type(value_type, &mut item_type);
            let item_type = item_type.unwrap_or(default_array_item_type());
            let parsed = parse_loose_literal(value);
            let mut items = match parsed {
                JsonValue::Array(values) => values,
                _ => Vec::new(),
            };
            let mut changed = false;
            let mut remove_index = None;
            ui.vertical(|ui| {
                for (index, item) in items.iter_mut().enumerate() {
                    let mut entry = literal_string_for_value_type(item, item_type);
                    ui.horizontal(|ui| {
                        let row_changed = draw_typed_editor_with_width(
                            ui, item_type, None, &mut entry, text_width,
                        );
                        if row_changed {
                            *item =
                                coerce_json_to_visual_type(&parse_loose_literal(&entry), item_type)
                                    .unwrap_or_else(|_| {
                                        parse_loose_literal(default_literal_for_type(item_type))
                                    });
                            changed = true;
                        }
                        if ui.small_button("-").clicked() {
                            remove_index = Some(index);
                        }
                    });
                }
                if ui.small_button("+ Item").clicked() {
                    items.push(parse_loose_literal(default_literal_for_type(item_type)));
                    changed = true;
                }
            });
            if let Some(index) = remove_index {
                if index < items.len() {
                    items.remove(index);
                    changed = true;
                }
            }
            if changed {
                *value = compact_json_string(&JsonValue::Array(items));
            }
            changed
        }
        VisualValueType::Vec2 => {
            let mut components = parse_vec2_literal(value);
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("X");
                changed |= ui
                    .add(DragValue::new(&mut components.0).speed(0.1))
                    .changed();
                ui.label("Y");
                changed |= ui
                    .add(DragValue::new(&mut components.1).speed(0.1))
                    .changed();
            });
            if changed {
                *value = compact_json_string(&vec2_json(components.0, components.1));
            }
            changed
        }
        VisualValueType::Vec3 => {
            let mut components = parse_vec3_literal(value);
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("X");
                changed |= ui
                    .add(DragValue::new(&mut components.0).speed(0.1))
                    .changed();
                ui.label("Y");
                changed |= ui
                    .add(DragValue::new(&mut components.1).speed(0.1))
                    .changed();
                ui.label("Z");
                changed |= ui
                    .add(DragValue::new(&mut components.2).speed(0.1))
                    .changed();
            });
            if changed {
                *value = compact_json_string(&vec3_json(components.0, components.1, components.2));
            }
            changed
        }
        VisualValueType::Quat => {
            let mut components = parse_quat_literal(value);
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("X");
                changed |= ui
                    .add(DragValue::new(&mut components.0).speed(0.05))
                    .changed();
                ui.label("Y");
                changed |= ui
                    .add(DragValue::new(&mut components.1).speed(0.05))
                    .changed();
                ui.label("Z");
                changed |= ui
                    .add(DragValue::new(&mut components.2).speed(0.05))
                    .changed();
                ui.label("W");
                changed |= ui
                    .add(DragValue::new(&mut components.3).speed(0.05))
                    .changed();
            });
            if changed {
                *value = compact_json_string(&quat_json(
                    components.0,
                    components.1,
                    components.2,
                    components.3,
                ));
            }
            changed
        }
        VisualValueType::Transform => {
            let mut components = parse_transform_literal(value);
            let mut changed = false;
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Pos");
                    changed |= ui
                        .add(DragValue::new(&mut components.position.0).speed(0.1))
                        .changed();
                    changed |= ui
                        .add(DragValue::new(&mut components.position.1).speed(0.1))
                        .changed();
                    changed |= ui
                        .add(DragValue::new(&mut components.position.2).speed(0.1))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Rot");
                    changed |= ui
                        .add(DragValue::new(&mut components.rotation.0).speed(0.05))
                        .changed();
                    changed |= ui
                        .add(DragValue::new(&mut components.rotation.1).speed(0.05))
                        .changed();
                    changed |= ui
                        .add(DragValue::new(&mut components.rotation.2).speed(0.05))
                        .changed();
                    changed |= ui
                        .add(DragValue::new(&mut components.rotation.3).speed(0.05))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Scl");
                    changed |= ui
                        .add(DragValue::new(&mut components.scale.0).speed(0.1))
                        .changed();
                    changed |= ui
                        .add(DragValue::new(&mut components.scale.1).speed(0.1))
                        .changed();
                    changed |= ui
                        .add(DragValue::new(&mut components.scale.2).speed(0.1))
                        .changed();
                });
            });
            if changed {
                let mut object = JsonMap::new();
                object.insert(
                    "position".to_string(),
                    vec3_json(
                        components.position.0,
                        components.position.1,
                        components.position.2,
                    ),
                );
                object.insert(
                    "rotation".to_string(),
                    quat_json(
                        components.rotation.0,
                        components.rotation.1,
                        components.rotation.2,
                        components.rotation.3,
                    ),
                );
                object.insert(
                    "scale".to_string(),
                    vec3_json(components.scale.0, components.scale.1, components.scale.2),
                );
                *value = compact_json_string(&JsonValue::Object(object));
            }
            changed
        }
        VisualValueType::Camera => draw_camera_data_editor(ui, value),
        VisualValueType::Light => draw_light_data_editor(ui, value),
        VisualValueType::MeshRenderer => draw_mesh_renderer_data_editor_plain(ui, value),
        VisualValueType::SpriteRenderer => draw_sprite_renderer_data_editor_plain(ui, value),
        VisualValueType::Text2d => draw_text2d_data_editor_plain(ui, value),
        VisualValueType::AudioEmitter => draw_audio_emitter_data_editor_plain(ui, value),
        VisualValueType::AudioListener => draw_audio_listener_data_editor(ui, value),
        VisualValueType::Script => draw_script_data_editor(ui, value),
        VisualValueType::LookAt => draw_look_at_data_editor(ui, value),
        VisualValueType::EntityFollower => draw_entity_follower_data_editor(ui, value),
        VisualValueType::AnimatorState => draw_animator_state_editor(ui, value),
        VisualValueType::InputModifiers => draw_input_modifiers_editor(ui, value),
        VisualValueType::AudioStreamingConfig => draw_audio_streaming_config_editor(ui, value),
        VisualValueType::RuntimeTuning => draw_runtime_tuning_editor(ui, value),
        VisualValueType::RuntimeConfig => draw_runtime_config_editor(ui, value),
        VisualValueType::RenderConfig => draw_render_config_editor(ui, value),
        VisualValueType::ShaderConstants => draw_shader_constants_editor(ui, value),
        VisualValueType::StreamingTuning => draw_streaming_tuning_editor(ui, value),
        VisualValueType::RenderPasses => draw_render_passes_editor(ui, value),
        VisualValueType::GpuBudget => draw_gpu_budget_editor(ui, value),
        VisualValueType::AssetBudgets => draw_asset_budgets_editor(ui, value),
        VisualValueType::WindowSettings => draw_window_settings_editor(ui, value),
        VisualValueType::Spline => draw_spline_editor(ui, value),
        VisualValueType::Physics => draw_physics_data_editor(ui, value),
        VisualValueType::PhysicsVelocity => draw_physics_velocity_editor(ui, value),
        VisualValueType::PhysicsWorldDefaults => draw_physics_world_defaults_editor(ui, value),
        VisualValueType::CharacterControllerOutput => draw_character_output_editor(ui, value),
        VisualValueType::DynamicComponentFields => draw_dynamic_component_fields_editor(ui, value),
        VisualValueType::DynamicFieldValue => draw_dynamic_field_value_editor(ui, value),
        VisualValueType::PhysicsQueryFilter => draw_physics_query_filter_editor(ui, value),
        VisualValueType::PhysicsRayCastHit => draw_ray_cast_hit_editor(ui, value),
        VisualValueType::PhysicsPointProjectionHit => draw_point_projection_hit_editor(ui, value),
        VisualValueType::PhysicsShapeCastHit => draw_shape_cast_hit_editor(ui, value),
        VisualValueType::Any => ui
            .add(TextEdit::singleline(value).desired_width(text_width))
            .changed(),
    }
}

#[derive(Debug, Clone, Copy)]
struct TransformLiteralComponents {
    position: (f64, f64, f64),
    rotation: (f64, f64, f64, f64),
    scale: (f64, f64, f64),
}

fn compact_json_string(value: &JsonValue) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn parse_vec2_literal(value: &str) -> (f64, f64) {
    let parsed = parse_loose_literal(value);
    coerce_json_to_vec2_components(&parsed).unwrap_or((0.0, 0.0))
}

fn parse_vec3_literal(value: &str) -> (f64, f64, f64) {
    let parsed = parse_loose_literal(value);
    coerce_json_to_vec3_components(&parsed).unwrap_or((0.0, 0.0, 0.0))
}

fn parse_quat_literal(value: &str) -> (f64, f64, f64, f64) {
    let parsed = parse_loose_literal(value);
    coerce_json_to_quat_components(&parsed).unwrap_or((0.0, 0.0, 0.0, 1.0))
}

fn parse_transform_literal(value: &str) -> TransformLiteralComponents {
    let parsed = parse_loose_literal(value);
    let normalized = coerce_json_to_visual_type(&parsed, VisualValueType::Transform)
        .unwrap_or_else(|_| {
            parse_loose_literal(default_literal_for_type(VisualValueType::Transform))
        });
    let object = normalized.as_object();
    let position_value = object
        .and_then(|obj| obj.get("position"))
        .cloned()
        .unwrap_or_else(|| vec3_json(0.0, 0.0, 0.0));
    let rotation_value = object
        .and_then(|obj| obj.get("rotation"))
        .cloned()
        .unwrap_or_else(|| quat_json(0.0, 0.0, 0.0, 1.0));
    let scale_value = object
        .and_then(|obj| obj.get("scale"))
        .cloned()
        .unwrap_or_else(|| vec3_json(1.0, 1.0, 1.0));

    TransformLiteralComponents {
        position: coerce_json_to_vec3_components(&position_value).unwrap_or((0.0, 0.0, 0.0)),
        rotation: coerce_json_to_quat_components(&rotation_value).unwrap_or((0.0, 0.0, 0.0, 1.0)),
        scale: coerce_json_to_vec3_components(&scale_value).unwrap_or((1.0, 1.0, 1.0)),
    }
}
