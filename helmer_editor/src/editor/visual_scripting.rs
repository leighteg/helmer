use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use bevy_ecs::prelude::{Resource, World};
use egui::{Color32, ComboBox, DragValue, Key, RichText, Sense, TextEdit, Ui};
use egui_snarl::ui::{AnyPins, PinInfo, SnarlViewer, SnarlWidget};
use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

pub const VISUAL_SCRIPT_EXTENSION: &str = "hvs";
const VISUAL_SCRIPT_VERSION: u32 = 1;
const MAX_API_ARGS: usize = 16;
const MAX_EXEC_STEPS_PER_EVENT: u32 = 10_000;
const MAX_LOOP_ITERATIONS: u32 = 4_096;

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
    Vec2,
    Vec3,
    Quat,
    Transform,
    #[default]
    Json,
}

impl VisualValueType {
    fn title(self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::Number => "Number",
            Self::String => "String",
            Self::Entity => "Entity",
            Self::Vec2 => "Vec2",
            Self::Vec3 => "Vec3",
            Self::Quat => "Quat",
            Self::Transform => "Transform",
            Self::Json => "Json",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisualVariableDefinition {
    #[serde(default)]
    pub id: u64,
    #[serde(default = "default_var_name")]
    pub name: String,
    #[serde(default)]
    pub value_type: VisualValueType,
    #[serde(default)]
    pub default_value: String,
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
    EcsFindEntityByName,
    EcsFollowSpline,
    EcsGetAnimatorClips,
    EcsGetAudioBusName,
    EcsGetAudioBusVolume,
    EcsGetAudioEmitter,
    EcsGetAudioEnabled,
    EcsGetAudioHeadWidth,
    EcsGetAudioListener,
    EcsGetAudioSceneVolume,
    EcsGetAudioSpeedOfSound,
    EcsGetAudioStreamingConfig,
    EcsGetCamera,
    EcsGetCharacterControllerOutput,
    EcsGetDynamicComponent,
    EcsGetDynamicField,
    EcsGetEntityName,
    EcsGetLight,
    EcsGetMeshRenderer,
    EcsGetPhysics,
    EcsGetPhysicsGravity,
    EcsGetPhysicsPointProjectionHit,
    EcsGetPhysicsRayCastHit,
    EcsGetPhysicsRunning,
    EcsGetPhysicsShapeCastHit,
    EcsGetPhysicsVelocity,
    EcsGetPhysicsWorldDefaults,
    EcsGetSceneAsset,
    EcsGetScript,
    EcsGetSpline,
    EcsGetTransform,
    EcsGetViewportMode,
    EcsGetViewportPreviewCamera,
    EcsHasComponent,
    EcsListAudioBuses,
    EcsListDynamicComponents,
    EcsListEntities,
    EcsOpenScene,
    EcsPlayAnimClip,
    EcsRemoveAudioBus,
    EcsRemoveComponent,
    EcsRemoveDynamicComponent,
    EcsRemoveDynamicField,
    EcsRemoveSplinePoint,
    EcsSampleSpline,
    EcsSetActiveCamera,
    EcsSetAnimatorEnabled,
    EcsSetAnimatorParamBool,
    EcsSetAnimatorParamFloat,
    EcsSetAnimatorTimeScale,
    EcsSetAudioBusName,
    EcsSetAudioBusVolume,
    EcsSetAudioEmitter,
    EcsSetAudioEnabled,
    EcsSetAudioHeadWidth,
    EcsSetAudioListener,
    EcsSetAudioSceneVolume,
    EcsSetAudioSpeedOfSound,
    EcsSetAudioStreamingConfig,
    EcsSetCamera,
    EcsSetDynamicComponent,
    EcsSetDynamicField,
    EcsSetEntityName,
    EcsSetLight,
    EcsSetMeshRenderer,
    EcsSetPersistentForce,
    EcsSetPersistentTorque,
    EcsSetPhysics,
    EcsSetPhysicsGravity,
    EcsSetPhysicsRunning,
    EcsSetPhysicsVelocity,
    EcsSetPhysicsWorldDefaults,
    EcsSetSceneAsset,
    EcsSetScript,
    EcsSetSpline,
    EcsSetSplinePoint,
    EcsSetTransform,
    EcsSetViewportMode,
    EcsSetViewportPreviewCamera,
    EcsSpawnEntity,
    EcsSplineLength,
    EcsSwitchScene,
    EcsTriggerAnimator,
    InputCursor,
    InputCursorDelta,
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
const VISUAL_API_OPERATION_ALL: [VisualApiOperation; 125] = [
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
    VisualApiOperation::EcsFindEntityByName,
    VisualApiOperation::EcsFollowSpline,
    VisualApiOperation::EcsGetAnimatorClips,
    VisualApiOperation::EcsGetAudioBusName,
    VisualApiOperation::EcsGetAudioBusVolume,
    VisualApiOperation::EcsGetAudioEmitter,
    VisualApiOperation::EcsGetAudioEnabled,
    VisualApiOperation::EcsGetAudioHeadWidth,
    VisualApiOperation::EcsGetAudioListener,
    VisualApiOperation::EcsGetAudioSceneVolume,
    VisualApiOperation::EcsGetAudioSpeedOfSound,
    VisualApiOperation::EcsGetAudioStreamingConfig,
    VisualApiOperation::EcsGetCamera,
    VisualApiOperation::EcsGetCharacterControllerOutput,
    VisualApiOperation::EcsGetDynamicComponent,
    VisualApiOperation::EcsGetDynamicField,
    VisualApiOperation::EcsGetEntityName,
    VisualApiOperation::EcsGetLight,
    VisualApiOperation::EcsGetMeshRenderer,
    VisualApiOperation::EcsGetPhysics,
    VisualApiOperation::EcsGetPhysicsGravity,
    VisualApiOperation::EcsGetPhysicsPointProjectionHit,
    VisualApiOperation::EcsGetPhysicsRayCastHit,
    VisualApiOperation::EcsGetPhysicsRunning,
    VisualApiOperation::EcsGetPhysicsShapeCastHit,
    VisualApiOperation::EcsGetPhysicsVelocity,
    VisualApiOperation::EcsGetPhysicsWorldDefaults,
    VisualApiOperation::EcsGetSceneAsset,
    VisualApiOperation::EcsGetScript,
    VisualApiOperation::EcsGetSpline,
    VisualApiOperation::EcsGetTransform,
    VisualApiOperation::EcsGetViewportMode,
    VisualApiOperation::EcsGetViewportPreviewCamera,
    VisualApiOperation::EcsHasComponent,
    VisualApiOperation::EcsListAudioBuses,
    VisualApiOperation::EcsListDynamicComponents,
    VisualApiOperation::EcsListEntities,
    VisualApiOperation::EcsOpenScene,
    VisualApiOperation::EcsPlayAnimClip,
    VisualApiOperation::EcsRemoveAudioBus,
    VisualApiOperation::EcsRemoveComponent,
    VisualApiOperation::EcsRemoveDynamicComponent,
    VisualApiOperation::EcsRemoveDynamicField,
    VisualApiOperation::EcsRemoveSplinePoint,
    VisualApiOperation::EcsSampleSpline,
    VisualApiOperation::EcsSetActiveCamera,
    VisualApiOperation::EcsSetAnimatorEnabled,
    VisualApiOperation::EcsSetAnimatorParamBool,
    VisualApiOperation::EcsSetAnimatorParamFloat,
    VisualApiOperation::EcsSetAnimatorTimeScale,
    VisualApiOperation::EcsSetAudioBusName,
    VisualApiOperation::EcsSetAudioBusVolume,
    VisualApiOperation::EcsSetAudioEmitter,
    VisualApiOperation::EcsSetAudioEnabled,
    VisualApiOperation::EcsSetAudioHeadWidth,
    VisualApiOperation::EcsSetAudioListener,
    VisualApiOperation::EcsSetAudioSceneVolume,
    VisualApiOperation::EcsSetAudioSpeedOfSound,
    VisualApiOperation::EcsSetAudioStreamingConfig,
    VisualApiOperation::EcsSetCamera,
    VisualApiOperation::EcsSetDynamicComponent,
    VisualApiOperation::EcsSetDynamicField,
    VisualApiOperation::EcsSetEntityName,
    VisualApiOperation::EcsSetLight,
    VisualApiOperation::EcsSetMeshRenderer,
    VisualApiOperation::EcsSetPersistentForce,
    VisualApiOperation::EcsSetPersistentTorque,
    VisualApiOperation::EcsSetPhysics,
    VisualApiOperation::EcsSetPhysicsGravity,
    VisualApiOperation::EcsSetPhysicsRunning,
    VisualApiOperation::EcsSetPhysicsVelocity,
    VisualApiOperation::EcsSetPhysicsWorldDefaults,
    VisualApiOperation::EcsSetSceneAsset,
    VisualApiOperation::EcsSetScript,
    VisualApiOperation::EcsSetSpline,
    VisualApiOperation::EcsSetSplinePoint,
    VisualApiOperation::EcsSetTransform,
    VisualApiOperation::EcsSetViewportMode,
    VisualApiOperation::EcsSetViewportPreviewCamera,
    VisualApiOperation::EcsSpawnEntity,
    VisualApiOperation::EcsSplineLength,
    VisualApiOperation::EcsSwitchScene,
    VisualApiOperation::EcsTriggerAnimator,
    VisualApiOperation::InputCursor,
    VisualApiOperation::InputCursorDelta,
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
    value_type: VisualValueType::Json,
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
        label: "Index",
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
        value_type: VisualValueType::Json,
    },
    VisualApiInputSpec {
        label: "Name",
        value_type: VisualValueType::String,
    },
];
const API_INPUTS_25: [VisualApiInputSpec; 2] = [
    VisualApiInputSpec {
        label: "Bus",
        value_type: VisualValueType::Json,
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
        value_type: VisualValueType::Json,
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
        value_type: VisualValueType::Json,
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
        value_type: VisualValueType::Json,
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
const API_INPUTS_37: [VisualApiInputSpec; 3] = [
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
    label: "Value",
    value_type: VisualValueType::Json,
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

const VISUAL_API_OPERATION_SPECS: [VisualApiOperationSpec; 125] = [
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
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCamera,
        table: VisualScriptApiTable::Ecs,
        function: "get_camera",
        title: "Get Camera",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetCharacterControllerOutput,
        table: VisualScriptApiTable::Ecs,
        function: "get_character_controller_output",
        title: "Get Character Controller Output",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetDynamicComponent,
        table: VisualScriptApiTable::Ecs,
        function: "get_dynamic_component",
        title: "Get Dynamic Component",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_14,
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetDynamicField,
        table: VisualScriptApiTable::Ecs,
        function: "get_dynamic_field",
        title: "Get Dynamic Field",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_15,
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetMeshRenderer,
        table: VisualScriptApiTable::Ecs,
        function: "get_mesh_renderer",
        title: "Get Mesh Renderer",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysics,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics",
        title: "Get Physics",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsRayCastHit,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_ray_cast_hit",
        title: "Get Physics Ray Cast Hit",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsVelocity,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_velocity",
        title: "Get Physics Velocity",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetPhysicsWorldDefaults,
        table: VisualScriptApiTable::Ecs,
        function: "get_physics_world_defaults",
        title: "Get Physics World Defaults",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
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
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsGetSpline,
        table: VisualScriptApiTable::Ecs,
        function: "get_spline",
        title: "Get Spline",
        category: "Get",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsListDynamicComponents,
        table: VisualScriptApiTable::Ecs,
        function: "list_dynamic_components",
        title: "List Dynamic Components",
        category: "List",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_8,
        output_type: Some(VisualValueType::Json),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsListEntities,
        table: VisualScriptApiTable::Ecs,
        function: "list_entities",
        title: "List Entities",
        category: "List",
        flow: VisualApiFlow::Pure,
        inputs: &API_INPUTS_7,
        output_type: Some(VisualValueType::Json),
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
        inputs: &API_INPUTS_26,
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
        inputs: &API_INPUTS_26,
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
        inputs: &API_INPUTS_26,
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
        inputs: &API_INPUTS_26,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetMeshRenderer,
        table: VisualScriptApiTable::Ecs,
        function: "set_mesh_renderer",
        title: "Set Mesh Renderer",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_26,
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
        inputs: &API_INPUTS_26,
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
        inputs: &API_INPUTS_26,
        output_type: Some(VisualValueType::Bool),
    },
    VisualApiOperationSpec {
        operation: VisualApiOperation::EcsSetPhysicsWorldDefaults,
        table: VisualScriptApiTable::Ecs,
        function: "set_physics_world_defaults",
        title: "Set Physics World Defaults",
        category: "Set",
        flow: VisualApiFlow::Exec,
        inputs: &API_INPUTS_26,
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
        inputs: &API_INPUTS_37,
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
        output_type: Some(VisualValueType::Json),
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
        output_type: Some(VisualValueType::Json),
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VisualScriptNodeKind {
    OnStart,
    OnUpdate,
    OnStop,
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
    JsonLiteral {
        #[serde(default)]
        value: String,
    },
    SelfEntity,
    DeltaTime,
    MathBinary {
        #[serde(default)]
        op: VisualMathOp,
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
    Vec3,
    Quat,
    Transform,
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
                    let default_value =
                        default_literal_for_type(spec.inputs[args.len()].value_type);
                    args.push(default_value.to_string());
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
            Self::JsonLiteral { .. } => "JSON".to_string(),
            Self::SelfEntity => "Self Entity".to_string(),
            Self::DeltaTime => "Delta Time".to_string(),
            Self::MathBinary { op } => format!("Math: {}", op.title()),
            Self::Compare { op } => format!("Compare: {}", op.title()),
            Self::LogicalBinary { op } => format!("Logical: {}", op.title()),
            Self::Not => "Not".to_string(),
            Self::Select { .. } => "Select".to_string(),
            Self::Vec3 => "Vec3".to_string(),
            Self::Quat => "Quat".to_string(),
            Self::Transform => "Transform".to_string(),
            Self::Comment { .. } => "Comment".to_string(),
            Self::Statement { .. } => "Legacy Statement".to_string(),
        }
    }

    fn exec_input_count(&self) -> usize {
        match self {
            Self::OnStart
            | Self::OnUpdate
            | Self::OnStop
            | Self::GetVariable { .. }
            | Self::QueryApi { .. }
            | Self::BoolLiteral { .. }
            | Self::NumberLiteral { .. }
            | Self::StringLiteral { .. }
            | Self::JsonLiteral { .. }
            | Self::SelfEntity
            | Self::DeltaTime
            | Self::MathBinary { .. }
            | Self::Compare { .. }
            | Self::LogicalBinary { .. }
            | Self::Not
            | Self::Select { .. }
            | Self::Vec3
            | Self::Quat
            | Self::Transform => 0,
            Self::Sequence { .. }
            | Self::Branch { .. }
            | Self::LoopWhile { .. }
            | Self::Log { .. }
            | Self::SetVariable { .. }
            | Self::ClearVariable { .. }
            | Self::CallApi { .. }
            | Self::Comment { .. }
            | Self::Statement { .. } => 1,
        }
    }

    fn exec_output_count(&self) -> usize {
        match self {
            Self::OnStart | Self::OnUpdate | Self::OnStop => 1,
            Self::Sequence { outputs } => usize::from((*outputs).clamp(1, 8)),
            Self::Branch { .. } => 2,
            Self::LoopWhile { .. } => 2,
            Self::Log { .. }
            | Self::SetVariable { .. }
            | Self::ClearVariable { .. }
            | Self::CallApi { .. }
            | Self::Comment { .. }
            | Self::Statement { .. } => 1,
            Self::GetVariable { .. }
            | Self::QueryApi { .. }
            | Self::BoolLiteral { .. }
            | Self::NumberLiteral { .. }
            | Self::StringLiteral { .. }
            | Self::JsonLiteral { .. }
            | Self::SelfEntity
            | Self::DeltaTime
            | Self::MathBinary { .. }
            | Self::Compare { .. }
            | Self::LogicalBinary { .. }
            | Self::Not
            | Self::Select { .. }
            | Self::Vec3
            | Self::Quat
            | Self::Transform => 0,
        }
    }

    fn data_input_count(&self) -> usize {
        match self {
            Self::Branch { .. }
            | Self::LoopWhile { .. }
            | Self::Log { .. }
            | Self::SetVariable { .. }
            | Self::Not => 1,
            Self::CallApi { operation, .. } | Self::QueryApi { operation, .. } => {
                operation.spec().inputs.len().min(MAX_API_ARGS)
            }
            Self::MathBinary { .. } | Self::Compare { .. } | Self::LogicalBinary { .. } => 2,
            Self::Vec3 | Self::Select { .. } | Self::Transform => 3,
            Self::Quat => 4,
            Self::OnStart
            | Self::OnUpdate
            | Self::OnStop
            | Self::Sequence { .. }
            | Self::ClearVariable { .. }
            | Self::GetVariable { .. }
            | Self::BoolLiteral { .. }
            | Self::NumberLiteral { .. }
            | Self::StringLiteral { .. }
            | Self::JsonLiteral { .. }
            | Self::SelfEntity
            | Self::DeltaTime
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
            | Self::JsonLiteral { .. }
            | Self::SelfEntity
            | Self::DeltaTime
            | Self::MathBinary { .. }
            | Self::Compare { .. }
            | Self::LogicalBinary { .. }
            | Self::Not
            | Self::Select { .. }
            | Self::Vec3
            | Self::Quat
            | Self::Transform
            | Self::CallApi { .. } => 1,
            Self::OnStart
            | Self::OnUpdate
            | Self::OnStop
            | Self::Sequence { .. }
            | Self::Branch { .. }
            | Self::LoopWhile { .. }
            | Self::Log { .. }
            | Self::SetVariable { .. }
            | Self::ClearVariable { .. }
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
                Self::Vec3 => "Vec3".to_string(),
                Self::Quat => "Quat".to_string(),
                Self::Transform => "Transform".to_string(),
                _ => "Value".to_string(),
            },
        }
    }

    fn pin_color(&self, slot: PinSlot, is_output: bool) -> Color32 {
        match slot.kind {
            PinKind::Data => PIN_COLOR_DATA,
            PinKind::Exec => match self {
                Self::OnStart | Self::OnUpdate | Self::OnStop if is_output => PIN_COLOR_EVENT,
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
            Self::OnStart | Self::OnUpdate | Self::OnStop | Self::SelfEntity | Self::DeltaTime
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

fn default_var_name() -> String {
    "value".to_string()
}

fn default_var_value() -> String {
    "0".to_string()
}

fn default_select_value_type() -> VisualValueType {
    VisualValueType::Json
}

fn default_literal_for_type(value_type: VisualValueType) -> &'static str {
    match value_type {
        VisualValueType::Bool => "false",
        VisualValueType::Number => "0",
        VisualValueType::String => "",
        VisualValueType::Entity => "0",
        VisualValueType::Vec2 => "{\"x\":0,\"y\":0}",
        VisualValueType::Vec3 => "{\"x\":0,\"y\":0,\"z\":0}",
        VisualValueType::Quat => "{\"x\":0,\"y\":0,\"z\":0,\"w\":1}",
        VisualValueType::Transform => {
            "{\"position\":{\"x\":0,\"y\":0,\"z\":0},\"rotation\":{\"x\":0,\"y\":0,\"z\":0,\"w\":1},\"scale\":{\"x\":1,\"y\":1,\"z\":1}}"
        }
        VisualValueType::Json => "null",
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
        label => label.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum VisualApiMenuSection {
    Gameplay,
    Scene,
    Physics,
    Animation,
    Audio,
    Spline,
    TransformRender,
    Dynamic,
    EntityComponent,
    InputKeyboard,
    InputMouse,
    InputGamepad,
    InputWindow,
    Utility,
}

impl VisualApiMenuSection {
    fn title(self) -> &'static str {
        match self {
            Self::Gameplay => "Gameplay",
            Self::Scene => "Scene",
            Self::Physics => "Physics",
            Self::Animation => "Animation",
            Self::Audio => "Audio",
            Self::Spline => "Spline",
            Self::TransformRender => "Transform & Render",
            Self::Dynamic => "Dynamic Components",
            Self::EntityComponent => "Entity & Components",
            Self::InputKeyboard => "Input / Keyboard",
            Self::InputMouse => "Input / Mouse",
            Self::InputGamepad => "Input / Gamepad",
            Self::InputWindow => "Input / Window",
            Self::Utility => "Utility",
        }
    }
}

const VISUAL_API_MENU_SECTION_ORDER: [VisualApiMenuSection; 14] = [
    VisualApiMenuSection::Gameplay,
    VisualApiMenuSection::Scene,
    VisualApiMenuSection::Physics,
    VisualApiMenuSection::Animation,
    VisualApiMenuSection::Audio,
    VisualApiMenuSection::Spline,
    VisualApiMenuSection::TransformRender,
    VisualApiMenuSection::Dynamic,
    VisualApiMenuSection::EntityComponent,
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
    if spec.category == "Gameplay" {
        return VisualApiMenuSection::Gameplay;
    }
    if function.contains("scene") {
        return VisualApiMenuSection::Scene;
    }
    if function.contains("physics")
        || function.contains("force")
        || function.contains("torque")
        || function.contains("impulse")
    {
        return VisualApiMenuSection::Physics;
    }
    if function.contains("anim") {
        return VisualApiMenuSection::Animation;
    }
    if function.contains("audio") {
        return VisualApiMenuSection::Audio;
    }
    if function.contains("spline") {
        return VisualApiMenuSection::Spline;
    }
    if function.contains("transform")
        || function.contains("camera")
        || function.contains("light")
        || function.contains("mesh")
        || function.contains("viewport")
    {
        return VisualApiMenuSection::TransformRender;
    }
    if function.contains("dynamic") {
        return VisualApiMenuSection::Dynamic;
    }
    if function.contains("entity") || function.contains("component") {
        return VisualApiMenuSection::EntityComponent;
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
    let args = spec
        .inputs
        .iter()
        .map(|input| default_literal_for_type(input.value_type).to_string())
        .collect();
    match spec.flow {
        VisualApiFlow::Exec => snarl.insert_node(
            pos,
            VisualScriptNodeKind::CallApi {
                operation: spec.operation,
                table: spec.table,
                function: spec.function.to_string(),
                args,
            },
        ),
        VisualApiFlow::Pure => snarl.insert_node(
            pos,
            VisualScriptNodeKind::QueryApi {
                operation: spec.operation,
                table: spec.table,
                function: spec.function.to_string(),
                args,
            },
        ),
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

fn node_data_input_type(
    node: &VisualScriptNodeKind,
    input_index: usize,
    variables: &[VisualVariableDefinition],
) -> Option<VisualValueType> {
    match node {
        VisualScriptNodeKind::Branch { .. }
        | VisualScriptNodeKind::LoopWhile { .. }
        | VisualScriptNodeKind::Not => Some(VisualValueType::Bool),
        VisualScriptNodeKind::Log { .. } => Some(VisualValueType::Json),
        VisualScriptNodeKind::SetVariable {
            variable_id, name, ..
        } => find_variable_definition(variables, *variable_id, name)
            .map(|var| var.value_type)
            .or(Some(VisualValueType::Json)),
        VisualScriptNodeKind::CallApi { operation, .. }
        | VisualScriptNodeKind::QueryApi { operation, .. } => operation
            .spec()
            .inputs
            .get(input_index)
            .map(|pin| pin.value_type),
        VisualScriptNodeKind::MathBinary { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::Compare { op } => match op {
            VisualCompareOp::Equals | VisualCompareOp::NotEquals => Some(VisualValueType::Json),
            VisualCompareOp::Less
            | VisualCompareOp::LessOrEqual
            | VisualCompareOp::Greater
            | VisualCompareOp::GreaterOrEqual => Some(VisualValueType::Number),
        },
        VisualScriptNodeKind::LogicalBinary { .. } => Some(VisualValueType::Bool),
        VisualScriptNodeKind::Select { value_type } => {
            if input_index == 0 {
                Some(VisualValueType::Bool)
            } else {
                Some(*value_type)
            }
        }
        VisualScriptNodeKind::Vec3 => Some(VisualValueType::Number),
        VisualScriptNodeKind::Quat => Some(VisualValueType::Number),
        VisualScriptNodeKind::Transform => match input_index {
            0 => Some(VisualValueType::Vec3),
            1 => Some(VisualValueType::Quat),
            _ => Some(VisualValueType::Vec3),
        },
        _ => None,
    }
}

fn node_data_output_type(
    node: &VisualScriptNodeKind,
    output_index: usize,
    variables: &[VisualVariableDefinition],
) -> Option<VisualValueType> {
    if output_index != 0 {
        return None;
    }

    match node {
        VisualScriptNodeKind::GetVariable {
            variable_id, name, ..
        } => find_variable_definition(variables, *variable_id, name)
            .map(|var| var.value_type)
            .or(Some(VisualValueType::Json)),
        VisualScriptNodeKind::CallApi { operation, .. }
        | VisualScriptNodeKind::QueryApi { operation, .. } => {
            operation.spec().output_type.or(Some(VisualValueType::Json))
        }
        VisualScriptNodeKind::BoolLiteral { .. }
        | VisualScriptNodeKind::Compare { .. }
        | VisualScriptNodeKind::LogicalBinary { .. }
        | VisualScriptNodeKind::Not => Some(VisualValueType::Bool),
        VisualScriptNodeKind::NumberLiteral { .. }
        | VisualScriptNodeKind::DeltaTime
        | VisualScriptNodeKind::MathBinary { .. } => Some(VisualValueType::Number),
        VisualScriptNodeKind::StringLiteral { .. } => Some(VisualValueType::String),
        VisualScriptNodeKind::JsonLiteral { .. } => Some(VisualValueType::Json),
        VisualScriptNodeKind::SelfEntity => Some(VisualValueType::Entity),
        VisualScriptNodeKind::Select { value_type } => Some(*value_type),
        VisualScriptNodeKind::Vec3 => Some(VisualValueType::Vec3),
        VisualScriptNodeKind::Quat => Some(VisualValueType::Quat),
        VisualScriptNodeKind::Transform => Some(VisualValueType::Transform),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualScriptEvent {
    Start,
    Update,
    Stop,
}

#[derive(Debug, Default, Clone)]
pub struct VisualScriptRuntimeState {
    pub variables: HashMap<String, JsonValue>,
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
pub struct VisualScriptProgram {
    source_label: String,
    source_name: String,
    nodes: HashMap<u64, VisualScriptNodeKind>,
    exec_edges: HashMap<(u64, usize), Vec<u64>>,
    data_edges: HashMap<(u64, usize), (u64, usize)>,
    variable_defaults: HashMap<String, JsonValue>,
    variable_types: HashMap<String, VisualValueType>,
    on_start_nodes: Vec<u64>,
    on_update_nodes: Vec<u64>,
    on_stop_nodes: Vec<u64>,
}

impl VisualScriptProgram {
    pub fn describe(&self) -> String {
        let mut api_calls = 0usize;
        let mut query_calls = 0usize;
        let mut variables = 0usize;

        for node in self.nodes.values() {
            match node {
                VisualScriptNodeKind::CallApi { .. } => api_calls += 1,
                VisualScriptNodeKind::QueryApi { .. } => query_calls += 1,
                VisualScriptNodeKind::SetVariable { .. }
                | VisualScriptNodeKind::GetVariable { .. }
                | VisualScriptNodeKind::ClearVariable { .. } => variables += 1,
                _ => {}
            }
        }

        format!(
            "Runtime Plan\nsource: {}\nnodes: {}\nexec edges: {}\ndata edges: {}\nvariable defs: {}\non_start: {}\non_update: {}\non_stop: {}\napi call nodes: {}\napi query nodes: {}\nvariable nodes: {}\nstep budget/event: {}",
            self.source_label,
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
            api_calls,
            query_calls,
            variables,
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
        for (variable, default_value) in &self.variable_defaults {
            state
                .variables
                .entry(variable.clone())
                .or_insert_with(|| default_value.clone());
        }

        let roots = match event {
            VisualScriptEvent::Start => &self.on_start_nodes,
            VisualScriptEvent::Update => &self.on_update_nodes,
            VisualScriptEvent::Stop => &self.on_stop_nodes,
        };

        if roots.is_empty() {
            return Ok(());
        }

        let mut context = VisualRuntimeContext {
            program: self,
            state,
            host,
            owner_entity_id,
            dt,
            node_results: HashMap::new(),
            data_cache: HashMap::new(),
            steps_left: MAX_EXEC_STEPS_PER_EVENT,
            legacy_statement_warnings: HashSet::new(),
        };

        for root in roots.iter().copied() {
            context.execute_exec_targets(root, 0)?;
        }

        Ok(())
    }
}

struct VisualRuntimeContext<'a, H: VisualScriptHost> {
    program: &'a VisualScriptProgram,
    state: &'a mut VisualScriptRuntimeState,
    host: &'a mut H,
    owner_entity_id: u64,
    dt: f32,
    node_results: HashMap<u64, JsonValue>,
    data_cache: HashMap<(u64, usize), JsonValue>,
    steps_left: u32,
    legacy_statement_warnings: HashSet<u64>,
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

    fn execute_exec_targets(&mut self, node_id: u64, output_index: usize) -> Result<(), String> {
        let targets = self
            .program
            .exec_edges
            .get(&(node_id, output_index))
            .cloned()
            .unwrap_or_default();

        for target in targets {
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
            | VisualScriptNodeKind::OnStop => {
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
                    coerce_json_to_visual_type(&resolved, *value_type)?
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
            | VisualScriptNodeKind::JsonLiteral { .. }
            | VisualScriptNodeKind::SelfEntity
            | VisualScriptNodeKind::DeltaTime
            | VisualScriptNodeKind::MathBinary { .. }
            | VisualScriptNodeKind::Compare { .. }
            | VisualScriptNodeKind::LogicalBinary { .. }
            | VisualScriptNodeKind::Not
            | VisualScriptNodeKind::Select { .. }
            | VisualScriptNodeKind::Vec3
            | VisualScriptNodeKind::Quat
            | VisualScriptNodeKind::Transform => {}
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
                .unwrap_or(VisualValueType::Json);
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
        Ok(args)
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
                            coerce_json_to_visual_type(&fallback, *value_type).unwrap_or(fallback)
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
            VisualScriptNodeKind::JsonLiteral { value } => parse_loose_literal(value),
            VisualScriptNodeKind::SelfEntity => {
                JsonValue::Number(JsonNumber::from(self.owner_entity_id))
            }
            VisualScriptNodeKind::DeltaTime => json_number(f64::from(self.dt)),
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
            VisualScriptNodeKind::Compare { op } => {
                let left = self.resolve_data_input_with_stack(node_id, 0, Some("null"), stack)?;
                let right = self.resolve_data_input_with_stack(node_id, 1, Some("null"), stack)?;
                let result = match op {
                    VisualCompareOp::Equals => left == right,
                    VisualCompareOp::NotEquals => left != right,
                    VisualCompareOp::Less => {
                        coerce_json_to_f64(&left)? < coerce_json_to_f64(&right)?
                    }
                    VisualCompareOp::LessOrEqual => {
                        coerce_json_to_f64(&left)? <= coerce_json_to_f64(&right)?
                    }
                    VisualCompareOp::Greater => {
                        coerce_json_to_f64(&left)? > coerce_json_to_f64(&right)?
                    }
                    VisualCompareOp::GreaterOrEqual => {
                        coerce_json_to_f64(&left)? >= coerce_json_to_f64(&right)?
                    }
                };
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
            VisualScriptNodeKind::Select { .. } => {
                let condition =
                    self.resolve_data_input_with_stack(node_id, 0, Some("false"), stack)?;
                if is_truthy(&condition) {
                    self.resolve_data_input_with_stack(node_id, 1, Some("null"), stack)?
                } else {
                    self.resolve_data_input_with_stack(node_id, 2, Some("null"), stack)?
                }
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
                let mut object = JsonMap::new();
                object.insert("position".to_string(), position);
                object.insert("rotation".to_string(), rotation);
                object.insert("scale".to_string(), scale);
                JsonValue::Object(object)
            }
            VisualScriptNodeKind::OnStart
            | VisualScriptNodeKind::OnUpdate
            | VisualScriptNodeKind::OnStop
            | VisualScriptNodeKind::Sequence { .. }
            | VisualScriptNodeKind::Branch { .. }
            | VisualScriptNodeKind::LoopWhile { .. }
            | VisualScriptNodeKind::Log { .. }
            | VisualScriptNodeKind::SetVariable { .. }
            | VisualScriptNodeKind::ClearVariable { .. }
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
            Err("Cannot convert complex JSON value to number".to_string())
        }
    }
}

fn coerce_json_to_visual_type(
    value: &JsonValue,
    value_type: VisualValueType,
) -> Result<JsonValue, String> {
    match value_type {
        VisualValueType::Json => Ok(value.clone()),
        VisualValueType::Bool => Ok(JsonValue::Bool(is_truthy(value))),
        VisualValueType::Number => Ok(json_number(coerce_json_to_f64(value)?)),
        VisualValueType::String => Ok(JsonValue::String(match value {
            JsonValue::String(text) => text.clone(),
            _ => json_to_log_string(value),
        })),
        VisualValueType::Entity => {
            let raw = coerce_json_to_f64(value)?;
            if !raw.is_finite() || raw < 0.0 {
                return Err("Entity ids must be finite non-negative numbers".to_string());
            }
            Ok(JsonValue::Number(JsonNumber::from(raw as u64)))
        }
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
            let object = value
                .as_object()
                .ok_or_else(|| "Transform values must be JSON objects".to_string())?;
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
    }
}

fn coerce_json_to_vec2_components(value: &JsonValue) -> Result<(f64, f64), String> {
    match value {
        JsonValue::Object(object) => Ok((
            coerce_json_to_f64(object.get("x").unwrap_or(&JsonValue::Null))?,
            coerce_json_to_f64(object.get("y").unwrap_or(&JsonValue::Null))?,
        )),
        JsonValue::Array(array) if array.len() >= 2 => Ok((
            coerce_json_to_f64(&array[0])?,
            coerce_json_to_f64(&array[1])?,
        )),
        JsonValue::String(text) => coerce_json_to_vec2_components(&parse_loose_literal(text)),
        _ => Err("Vec2 values must be objects, arrays, or JSON strings".to_string()),
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
        _ => Err("Vec3 values must be objects, arrays, or JSON strings".to_string()),
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
        _ => Err("Quat values must be objects, arrays, or JSON strings".to_string()),
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
    pub snarl: Snarl<VisualScriptNodeKind>,
    pub dirty: bool,
    pub compile_preview: String,
    pub compile_error: Option<String>,
    pub add_node_menu_open: bool,
    pub add_node_menu_pos: [f32; 2],
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
            snarl: graph_data_to_snarl(&document.graph),
            dirty: false,
            compile_preview: String::new(),
            compile_error: None,
            add_node_menu_open: false,
            add_node_menu_pos: [0.0, 0.0],
        };
        out.recompile_preview();
        out
    }

    fn to_document(&self) -> VisualScriptDocument {
        let mut document = VisualScriptDocument {
            version: VISUAL_SCRIPT_VERSION,
            name: self.name.trim().to_string(),
            prelude: self.prelude.clone(),
            variables: self.variables.clone(),
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

#[derive(Resource, Default)]
pub struct VisualScriptEditorState {
    pub open: bool,
    pub active_path: Option<PathBuf>,
    pub documents: HashMap<PathBuf, VisualScriptOpenDocument>,
}

pub fn is_visual_script_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(VISUAL_SCRIPT_EXTENSION))
        .unwrap_or(false)
}

pub fn default_visual_script_template_full() -> String {
    let document = default_visual_script_document();
    let pretty = PrettyConfig::new().compact_arrays(false);
    ron::ser::to_string_pretty(&document, pretty)
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
                        ui.label("No visual script is open.");
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
                            ui.label("No visual script is open.");
                            return;
                        }
                    }

                    ui.horizontal_wrapped(|ui| {
                        ui.label("Document:");
                        ui.monospace(compact_display_text(&path_display_name(&active_path), 64));

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
                    });

                    state.active_path = Some(active_path.clone());

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
                                let before = graph_data_from_snarl(&document.snarl);
                                let mut viewer =
                                    VisualScriptViewer::with_variables(&document.variables);
                                let min_size = egui::vec2(120.0, 120.0);
                                let snarl_response = SnarlWidget::new()
                                    .id_salt(("visual_script_graph", &document.path))
                                    .min_size(min_size)
                                    .show(&mut document.snarl, &mut viewer, ui);

                                if snarl_response.secondary_clicked() && !ui.ctx().is_popup_open() {
                                    let pointer_pos = ui
                                        .ctx()
                                        .input(|i| i.pointer.interact_pos())
                                        .or_else(|| snarl_response.interact_pointer_pos());
                                    if let Some(pos) = pointer_pos {
                                        document.add_node_menu_pos = [pos.x, pos.y];
                                        document.add_node_menu_open = true;
                                    }
                                }

                                if document.add_node_menu_open {
                                    let menu_pos = egui::pos2(
                                        document.add_node_menu_pos[0],
                                        document.add_node_menu_pos[1],
                                    );
                                    let popup_id = ui.make_persistent_id((
                                        "visual_script_add_node_popup",
                                        &document.path,
                                    ));
                                    let mut inserted = None;
                                    let menu_area = egui::Area::new(popup_id)
                                        .order(egui::Order::Tooltip)
                                        .movable(false)
                                        .interactable(true)
                                        .fixed_pos(menu_pos)
                                        .show(ui.ctx(), |ui| {
                                            egui::Frame::popup(ui.style()).show(ui, |ui| {
                                                ui.set_min_width(320.0);
                                                inserted = viewer.add_node_menu(
                                                    menu_pos,
                                                    ui,
                                                    &mut document.snarl,
                                                );
                                            });
                                        });

                                    if inserted.is_some() {
                                        document.add_node_menu_open = false;
                                    }

                                    let clicked_any = ui.input(|i| i.pointer.any_click());
                                    let clicked_secondary = snarl_response.secondary_clicked();
                                    let pointer_pos = ui.input(|i| i.pointer.interact_pos());
                                    let clicked_inside = pointer_pos
                                        .is_some_and(|pos| menu_area.response.rect.contains(pos));
                                    let close_by_escape = ui.input(|i| i.key_pressed(Key::Escape));
                                    if close_by_escape
                                        || (clicked_any && !clicked_inside && !clicked_secondary)
                                    {
                                        document.add_node_menu_open = false;
                                    }
                                }

                                let after = graph_data_from_snarl(&document.snarl);
                                if before != after || viewer.changed {
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
                                    "Validation failed. Fix graph issues to view the runtime plan."
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
    let mut variable_defaults = HashMap::new();
    let mut variable_types = HashMap::new();
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
        let coerced_default = coerce_json_to_visual_type(&parsed_default, variable.value_type)
            .map_err(|err| format!("Invalid default for variable '{}': {}", variable_name, err))?;
        variable_types.insert(key.clone(), variable.value_type);
        variable_defaults.insert(key, coerced_default);
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
            if from_value_type != to_value_type {
                let allow_log_any_input =
                    matches!(to_kind, VisualScriptNodeKind::Log { .. }) && to_slot.index == 0;
                if allow_log_any_input {
                    continue;
                }
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

    on_start_nodes.sort_unstable();
    on_update_nodes.sort_unstable();
    on_stop_nodes.sort_unstable();

    if on_start_nodes.is_empty() && on_update_nodes.is_empty() && on_stop_nodes.is_empty() {
        return Err("Visual script must contain at least one event node".to_string());
    }

    Ok(VisualScriptProgram {
        source_label: source_label.to_string(),
        source_name: path_display_name(Path::new(source_label)),
        nodes,
        exec_edges,
        data_edges,
        variable_defaults,
        variable_types,
        on_start_nodes,
        on_update_nodes,
        on_stop_nodes,
    })
}

fn normalize_document(document: &mut VisualScriptDocument) {
    document.version = VISUAL_SCRIPT_VERSION;
    if document.graph.nodes.is_empty() {
        document.graph = default_visual_script_document().graph;
    }

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
                        document.variables.push(VisualVariableDefinition {
                            id: new_id,
                            name: trimmed.clone(),
                            value_type: VisualValueType::Json,
                            default_value: value.clone(),
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
                        document.variables.push(VisualVariableDefinition {
                            id: new_id,
                            name: trimmed.clone(),
                            value_type: VisualValueType::Json,
                            default_value: default_value.clone(),
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
                            value_type: VisualValueType::Json,
                            default_value: "null".to_string(),
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
            _ => {}
        }

        node.kind.normalize();
    }

    document.variables.sort_by_key(|var| var.id);
    for variable in &mut document.variables {
        if variable.default_value.trim().is_empty() {
            variable.default_value = default_literal_for_type(variable.value_type).to_string();
        }
    }

    let mut node_map = HashMap::new();
    for node in &document.graph.nodes {
        node_map.insert(node.id, node.kind.clone());
    }

    let mut seen_wires = HashSet::new();
    let mut data_input_drivers = HashSet::new();
    document.graph.wires.retain(|wire| {
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
            let Some(from_type) =
                node_data_output_type(from_node, from_slot.index, &document.variables)
            else {
                return false;
            };
            let Some(to_type) = node_data_input_type(to_node, to_slot.index, &document.variables)
            else {
                return false;
            };
            if from_type != to_type {
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

fn default_visual_script_document() -> VisualScriptDocument {
    let mut snarl = Snarl::new();
    let on_start = snarl.insert_node(egui::pos2(48.0, 80.0), VisualScriptNodeKind::OnStart);
    let on_update = snarl.insert_node(egui::pos2(48.0, 280.0), VisualScriptNodeKind::OnUpdate);
    let start_log = snarl.insert_node(
        egui::pos2(340.0, 80.0),
        VisualScriptNodeKind::Log {
            message: "visual script started".to_string(),
        },
    );
    let update_note = snarl.insert_node(
        egui::pos2(340.0, 280.0),
        VisualScriptNodeKind::Comment {
            text: "Use typed API nodes to drive gameplay.\nExamples:\n- Set Transform\n- Spawn Entity\n- Input Key Down"
                .to_string(),
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

    snarl.connect(
        OutPinId {
            node: on_update,
            output: 0,
        },
        InPinId {
            node: update_note,
            input: 0,
        },
    );

    VisualScriptDocument {
        version: VISUAL_SCRIPT_VERSION,
        name: "visual_script".to_string(),
        prelude: "Graph runtime notes. This field is optional metadata.".to_string(),
        variables: vec![VisualVariableDefinition {
            id: 1,
            name: "speed".to_string(),
            value_type: VisualValueType::Number,
            default_value: "5".to_string(),
        }],
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
                        for value_type in [
                            VisualValueType::Bool,
                            VisualValueType::Number,
                            VisualValueType::String,
                            VisualValueType::Entity,
                            VisualValueType::Vec2,
                            VisualValueType::Vec3,
                            VisualValueType::Quat,
                            VisualValueType::Transform,
                            VisualValueType::Json,
                        ] {
                            if ui
                                .selectable_value(
                                    &mut variable.value_type,
                                    value_type,
                                    value_type.title(),
                                )
                                .changed()
                            {
                                if variable.default_value.trim().is_empty() {
                                    variable.default_value =
                                        default_literal_for_type(value_type).to_string();
                                }
                                *changed = true;
                                prune_wires = true;
                            }
                        }
                    });

                if ui.small_button("Delete").clicked() {
                    remove_index = Some(index);
                }
            });

            ui.horizontal(|ui| {
                ui.label("Default");
                if draw_typed_default_editor(ui, variable.value_type, &mut variable.default_value) {
                    *changed = true;
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
            value_type: VisualValueType::Json,
            default_value: "null".to_string(),
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

    if prune_wires && prune_invalid_wires(&mut document.snarl, &document.variables) > 0 {
        *changed = true;
    }
}

#[derive(Default)]
struct VisualScriptViewer {
    changed: bool,
    variables: Vec<VisualVariableDefinition>,
}

impl VisualScriptViewer {
    fn with_variables(variables: &[VisualVariableDefinition]) -> Self {
        Self {
            changed: false,
            variables: variables.to_vec(),
        }
    }

    fn mark_changed(&mut self) {
        self.changed = true;
    }

    fn add_node_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut Ui,
        snarl: &mut Snarl<VisualScriptNodeKind>,
    ) -> Option<NodeId> {
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

        let mut inserted = None;
        let mut has_visible = false;
        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(520.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.label(RichText::new("Events").strong());
                if add_node_search_matches_any(&search, &["on start", "event"]) {
                    has_visible = true;
                    if ui.button("On Start").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::OnStart));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["on update", "tick", "event"]) {
                    has_visible = true;
                    if ui.button("On Update").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::OnUpdate));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["on stop", "event"]) {
                    has_visible = true;
                    if ui.button("On Stop").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::OnStop));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }

                ui.separator();
                ui.label(RichText::new("API").strong());
                for section in VISUAL_API_MENU_SECTION_ORDER {
                    let mut specs: Vec<&VisualApiOperationSpec> = VISUAL_API_OPERATION_SPECS
                        .iter()
                        .filter(|spec| api_menu_section(spec) == section)
                        .filter(|spec| api_spec_matches_add_node_search(spec, section, &search))
                        .collect();
                    if specs.is_empty() {
                        continue;
                    }
                    has_visible = true;
                    specs.sort_by_key(|spec| spec.title);
                    ui.collapsing(format!("{} ({})", section.title(), specs.len()), |ui| {
                        for spec in specs {
                            let label = match spec.flow {
                                VisualApiFlow::Exec => format!("{} [Call]", spec.title),
                                VisualApiFlow::Pure => spec.title.to_string(),
                            };
                            if ui.button(label).clicked() {
                                inserted = Some(insert_api_node_from_spec(snarl, pos, spec));
                                ui.close_kind(egui::UiKind::Menu);
                                break;
                            }
                        }
                    });
                    if inserted.is_some() {
                        break;
                    }
                }

                ui.separator();
                ui.label(RichText::new("Flow").strong());
                if add_node_search_matches_any(&search, &["sequence", "flow"]) {
                    has_visible = true;
                    if ui.button("Sequence").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::Sequence {
                                outputs: default_sequence_outputs(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["branch", "flow"]) {
                    has_visible = true;
                    if ui.button("Branch").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::Branch {
                                condition: default_branch_condition(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["loop while", "loop", "flow"]) {
                    has_visible = true;
                    if ui.button("Loop While").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::LoopWhile {
                                condition: default_loop_condition(),
                                max_iterations: default_max_loop_iterations(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["log", "message"]) {
                    has_visible = true;
                    if ui.button("Log").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::Log {
                                message: default_log_message(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }

                ui.separator();
                ui.label(RichText::new("Variables").strong());
                if self.variables.is_empty() {
                    if add_node_search_matches_any(&search, &["variables", "variable"]) {
                        has_visible = true;
                        ui.small("Define variables in the side panel to add getter/setter nodes.");
                    }
                } else {
                    for variable in &self.variables {
                        let variable_label =
                            format!("{} {}", variable.name, variable.value_type.title());
                        let matches_variable = add_node_search_matches_any(
                            &search,
                            &[&variable_label, "variable", "set", "get", "clear"],
                        );
                        if !matches_variable {
                            continue;
                        }
                        has_visible = true;
                        ui.horizontal(|ui| {
                            ui.small(format!(
                                "{} ({})",
                                variable.name,
                                variable.value_type.title()
                            ));
                            if ui.small_button("Set").clicked() {
                                inserted = Some(snarl.insert_node(
                                    pos,
                                    VisualScriptNodeKind::SetVariable {
                                        variable_id: variable.id,
                                        name: variable.name.clone(),
                                        value: variable.default_value.clone(),
                                    },
                                ));
                                ui.close_kind(egui::UiKind::Menu);
                            }
                            if ui.small_button("Get").clicked() {
                                inserted = Some(snarl.insert_node(
                                    pos,
                                    VisualScriptNodeKind::GetVariable {
                                        variable_id: variable.id,
                                        name: variable.name.clone(),
                                        default_value: variable.default_value.clone(),
                                    },
                                ));
                                ui.close_kind(egui::UiKind::Menu);
                            }
                            if ui.small_button("Clear").clicked() {
                                inserted = Some(snarl.insert_node(
                                    pos,
                                    VisualScriptNodeKind::ClearVariable {
                                        variable_id: variable.id,
                                        name: variable.name.clone(),
                                    },
                                ));
                                ui.close_kind(egui::UiKind::Menu);
                            }
                        });
                        if inserted.is_some() {
                            break;
                        }
                    }
                }

                ui.separator();
                ui.label(RichText::new("Values").strong());
                if add_node_search_matches_any(&search, &["bool", "value"]) {
                    has_visible = true;
                    if ui.button("Bool").clicked() {
                        inserted =
                            Some(snarl.insert_node(
                                pos,
                                VisualScriptNodeKind::BoolLiteral { value: false },
                            ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["number", "value"]) {
                    has_visible = true;
                    if ui.button("Number").clicked() {
                        inserted =
                            Some(snarl.insert_node(
                                pos,
                                VisualScriptNodeKind::NumberLiteral { value: 0.0 },
                            ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["string", "value", "text"]) {
                    has_visible = true;
                    if ui.button("String").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::StringLiteral {
                                value: "text".to_string(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["json", "value"]) {
                    has_visible = true;
                    if ui.button("JSON").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::JsonLiteral {
                                value: "{\"x\":0}".to_string(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["self entity", "entity", "value"]) {
                    has_visible = true;
                    if ui.button("Self Entity").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::SelfEntity));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["delta time", "time", "value"]) {
                    has_visible = true;
                    if ui.button("Delta Time").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::DeltaTime));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }

                ui.separator();
                ui.label(RichText::new("Math/Logic").strong());
                if add_node_search_matches_any(&search, &["math", "number"]) {
                    has_visible = true;
                    if ui.button("Math").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::MathBinary {
                                op: VisualMathOp::Add,
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["compare", "bool"]) {
                    has_visible = true;
                    if ui.button("Compare").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::Compare {
                                op: VisualCompareOp::Equals,
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["logical", "bool"]) {
                    has_visible = true;
                    if ui.button("Logical").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::LogicalBinary {
                                op: VisualLogicalOp::And,
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["not", "bool"]) {
                    has_visible = true;
                    if ui.button("Not").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::Not));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["select", "bool"]) {
                    has_visible = true;
                    if ui.button("Select").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::Select {
                                value_type: default_select_value_type(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }

                ui.separator();
                ui.label(RichText::new("Structured Values").strong());
                if add_node_search_matches_any(&search, &["vec3", "vector", "structured"]) {
                    has_visible = true;
                    if ui.button("Vec3").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::Vec3));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["quat", "rotation", "structured"]) {
                    has_visible = true;
                    if ui.button("Quat").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::Quat));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["transform", "structured"]) {
                    has_visible = true;
                    if ui.button("Transform").clicked() {
                        inserted = Some(snarl.insert_node(pos, VisualScriptNodeKind::Transform));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }

                ui.separator();
                ui.label(RichText::new("Other").strong());
                if add_node_search_matches_any(&search, &["comment", "notes"]) {
                    has_visible = true;
                    if ui.button("Comment").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::Comment {
                                text: "notes".to_string(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
                if add_node_search_matches_any(&search, &["legacy", "statement"]) {
                    has_visible = true;
                    if ui.button("Legacy Statement").clicked() {
                        inserted = Some(snarl.insert_node(
                            pos,
                            VisualScriptNodeKind::Statement {
                                code: "-- legacy".to_string(),
                            },
                        ));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }

                if !has_visible {
                    ui.small("No nodes match the current search.");
                }
            });

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
        let Some((slot, label, color, value_type, allow_inline_default)) =
            snarl.get_node(pin.id.node).and_then(|node| {
                let slot = node.input_slot(pin.id.input)?;
                let value_type = if matches!(slot.kind, PinKind::Data) {
                    node_data_input_type(node, slot.index, &self.variables)
                } else {
                    None
                };
                let allow_inline_default = matches!(
                    node,
                    VisualScriptNodeKind::CallApi { .. } | VisualScriptNodeKind::QueryApi { .. }
                ) && matches!(slot.kind, PinKind::Data);
                Some((
                    slot,
                    node.input_label(slot),
                    node.pin_color(slot, false),
                    value_type,
                    allow_inline_default,
                ))
            })
        else {
            return PinInfo::square().with_fill(PIN_COLOR_EXEC);
        };

        ui.horizontal(|ui| {
            ui.small(label);
            if allow_inline_default && pin.remotes.is_empty() {
                if let Some(node) = snarl.get_node_mut(pin.id.node) {
                    if let Some(default_literal) = api_input_default_literal_mut(node, slot.index) {
                        ui.add_space(4.0);
                        let input_type = value_type.unwrap_or(VisualValueType::Json);
                        if draw_typed_pin_input_editor(ui, input_type, default_literal) {
                            self.mark_changed();
                        }
                    }
                }
            }
        });

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
                ui.small(node.output_label(slot));
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
                | VisualScriptNodeKind::SelfEntity
                | VisualScriptNodeKind::DeltaTime
                | VisualScriptNodeKind::Not
                | VisualScriptNodeKind::Vec3
                | VisualScriptNodeKind::Quat
                | VisualScriptNodeKind::Transform => {}
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
                        self.mark_changed();
                        prune_wires = true;
                    }
                    if node_input_is_disconnected(inputs, 1) {
                        ui.horizontal(|ui| {
                            ui.label("Default when unplugged");
                            let value_type =
                                find_variable_definition(&self.variables, *variable_id, name)
                                    .map(|var| var.value_type)
                                    .unwrap_or(VisualValueType::Json);
                            if draw_typed_default_editor(ui, value_type, value) {
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
                        self.mark_changed();
                        prune_wires = true;
                    }
                    ui.horizontal(|ui| {
                        ui.label("Default");
                        let value_type =
                            find_variable_definition(&self.variables, *variable_id, name)
                                .map(|var| var.value_type)
                                .unwrap_or(VisualValueType::Json);
                        if draw_typed_default_editor(ui, value_type, default_value) {
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
                VisualScriptNodeKind::JsonLiteral { value } => {
                    ui.label("JSON or loose literal");
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
                            for candidate in [
                                VisualValueType::Bool,
                                VisualValueType::Number,
                                VisualValueType::String,
                                VisualValueType::Entity,
                                VisualValueType::Vec2,
                                VisualValueType::Vec3,
                                VisualValueType::Quat,
                                VisualValueType::Transform,
                                VisualValueType::Json,
                            ] {
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
                        "Legacy node: script text execution is disabled.",
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
        false
    }

    fn show_graph_menu(
        &mut self,
        _pos: egui::Pos2,
        _ui: &mut Ui,
        _snarl: &mut Snarl<VisualScriptNodeKind>,
    ) {
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
        if let Some(new_node) = self.add_node_menu(pos, ui, snarl) {
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
        let from_type = node_data_output_type(from_node, from_slot.index, variables);
        let to_type = node_data_input_type(to_node, to_slot.index, variables);
        if from_type.is_none() || to_type.is_none() || from_type != to_type {
            let allow_log_any_input = matches!(to_node, VisualScriptNodeKind::Log { .. })
                && to_slot.index == 0
                && from_type.is_some();
            if allow_log_any_input {
                return true;
            }
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
        ui.small("No variables are defined.");
        return changed;
    }

    let selected = find_variable_definition(variables, *variable_id, name)
        .map(|variable| format!("{} ({})", variable.name, variable.value_type.title()))
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
                *arg = default_literal_for_type(spec.inputs[index].value_type).to_string();
            }
        }
        change.changed = true;
    }

    change
}

fn draw_typed_default_editor(ui: &mut Ui, value_type: VisualValueType, value: &mut String) -> bool {
    draw_typed_editor_with_width(ui, value_type, value, 140.0)
}

fn draw_typed_pin_input_editor(
    ui: &mut Ui,
    value_type: VisualValueType,
    value: &mut String,
) -> bool {
    draw_typed_editor_with_width(ui, value_type, value, 112.0)
}

fn draw_typed_editor_with_width(
    ui: &mut Ui,
    value_type: VisualValueType,
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
        VisualValueType::Json => ui
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
