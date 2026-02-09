use std::{
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
    "find_entity_by_name",
    "follow_spline",
    "get_animator_clips",
    "get_audio_bus_name",
    "get_audio_bus_volume",
    "get_audio_emitter",
    "get_audio_enabled",
    "get_audio_head_width",
    "get_audio_listener",
    "get_audio_scene_volume",
    "get_audio_speed_of_sound",
    "get_audio_streaming_config",
    "get_camera",
    "get_character_controller_output",
    "get_dynamic_component",
    "get_dynamic_field",
    "get_entity_name",
    "get_light",
    "get_mesh_renderer",
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
    "get_spline",
    "get_transform",
    "has_component",
    "list_audio_buses",
    "list_dynamic_components",
    "list_entities",
    "open_scene",
    "play_anim_clip",
    "remove_audio_bus",
    "remove_component",
    "remove_dynamic_component",
    "remove_dynamic_field",
    "remove_spline_point",
    "sample_spline",
    "set_active_camera",
    "set_animator_enabled",
    "set_animator_param_bool",
    "set_animator_param_float",
    "set_animator_time_scale",
    "set_audio_bus_name",
    "set_audio_bus_volume",
    "set_audio_emitter",
    "set_audio_enabled",
    "set_audio_head_width",
    "set_audio_listener",
    "set_audio_scene_volume",
    "set_audio_speed_of_sound",
    "set_audio_streaming_config",
    "set_camera",
    "set_dynamic_component",
    "set_dynamic_field",
    "set_entity_name",
    "set_light",
    "set_mesh_renderer",
    "set_persistent_force",
    "set_persistent_torque",
    "set_physics",
    "set_physics_gravity",
    "set_physics_running",
    "set_physics_velocity",
    "set_physics_world_defaults",
    "set_scene_asset",
    "set_script",
    "set_spline",
    "set_spline_point",
    "set_transform",
    "spawn_entity",
    "spline_length",
    "switch_scene",
    "trigger_animator",
];

pub const INPUT_FUNCTIONS: &[&str] = &[
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
    "wants_keyboard",
    "wants_pointer",
    "wheel",
    "window_size",
];

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
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

    pub fn get_light(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_light", (entity_id,))
    }

    pub fn set_light<S: Serialize>(&self, entity_id: EntityId, patch: S) -> Result<bool, String> {
        self.ecs_call("set_light", (entity_id, patch))
    }

    pub fn get_camera(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_camera", (entity_id,))
    }

    pub fn set_camera<S: Serialize>(&self, entity_id: EntityId, patch: S) -> Result<bool, String> {
        self.ecs_call("set_camera", (entity_id, patch))
    }

    pub fn set_active_camera(&self, entity_id: EntityId) -> Result<bool, String> {
        self.ecs_call("set_active_camera", (entity_id,))
    }

    pub fn get_mesh_renderer(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_mesh_renderer", (entity_id,))
    }

    pub fn set_mesh_renderer<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_mesh_renderer", (entity_id, patch))
    }

    pub fn list_dynamic_components(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("list_dynamic_components", (entity_id,))
    }

    pub fn get_dynamic_component(
        &self,
        entity_id: EntityId,
        component_name: &str,
    ) -> Result<Option<Value>, String> {
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

    pub fn get_dynamic_field(
        &self,
        entity_id: EntityId,
        component_name: &str,
        field_name: &str,
    ) -> Result<Option<Value>, String> {
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

    pub fn set_physics<S: Serialize>(&self, entity_id: EntityId, patch: S) -> Result<bool, String> {
        self.ecs_call("set_physics", (entity_id, patch))
    }

    pub fn clear_physics(&self, entity_id: EntityId) -> Result<bool, String> {
        self.ecs_call("clear_physics", (entity_id,))
    }

    pub fn get_physics_world_defaults(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_physics_world_defaults", (entity_id,))
    }

    pub fn set_physics_world_defaults<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_physics_world_defaults", (entity_id, patch))
    }

    pub fn get_physics_velocity(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_physics_velocity", (entity_id,))
    }

    pub fn set_physics_velocity<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_physics_velocity", (entity_id, patch))
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

    pub fn get_physics_gravity(&self) -> Result<Option<Value>, String> {
        self.ecs_call("get_physics_gravity", ())
    }

    pub fn get_audio_emitter(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_audio_emitter", (entity_id,))
    }

    pub fn set_audio_emitter<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
    ) -> Result<bool, String> {
        self.ecs_call("set_audio_emitter", (entity_id, patch))
    }

    pub fn get_audio_listener(&self, entity_id: EntityId) -> Result<Option<Value>, String> {
        self.ecs_call("get_audio_listener", (entity_id,))
    }

    pub fn set_audio_listener<S: Serialize>(
        &self,
        entity_id: EntityId,
        patch: S,
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

    pub fn create_audio_bus(&self, name: Option<&str>) -> Result<Option<i64>, String> {
        self.ecs_call("create_audio_bus", (name,))
    }

    pub fn remove_audio_bus(&self, bus: i64) -> Result<bool, String> {
        self.ecs_call("remove_audio_bus", (bus,))
    }

    pub fn set_audio_bus_volume(&self, bus: i64, volume: f32) -> Result<bool, String> {
        self.ecs_call("set_audio_bus_volume", (bus, volume))
    }

    pub fn get_audio_bus_volume(&self, bus: i64) -> Result<Option<f32>, String> {
        self.ecs_call("get_audio_bus_volume", (bus,))
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
