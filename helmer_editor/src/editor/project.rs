use std::{
    env, fs,
    io::ErrorKind,
    path::{Component, Path, PathBuf},
};

use bevy_ecs::prelude::Resource;
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

const DEFAULT_MATERIAL_FILE: &str = "default.ron";
const LUAURC_FILE_NAME: &str = ".luaurc";
const LUAURC_GENERATED_MARKER: &str = "\"__generated_by_helmer__\": true";
const HELMER_LUAU_API_FILE_NAME: &str = "helmer_api.d.luau";

pub use helmer_editor_runtime::project::ProjectConfig;

#[derive(Resource, Default, Clone)]
pub struct EditorProject {
    pub root: Option<PathBuf>,
    pub config: Option<ProjectConfig>,
}

pub fn create_project(root: &Path, name: &str) -> Result<ProjectConfig, String> {
    fs::create_dir_all(root).map_err(|err| err.to_string())?;

    let config = ProjectConfig::new(name.to_string());
    ensure_project_layout(root, &config)?;
    write_project_config(root, &config)?;
    write_default_material(root, &config)?;

    Ok(config)
}

pub fn load_project(root: &Path) -> Result<ProjectConfig, String> {
    let config_path = ProjectConfig::config_path(root);
    let config = if config_path.exists() {
        let data = fs::read_to_string(&config_path).map_err(|err| err.to_string())?;
        ron::de::from_str::<ProjectConfig>(&data).map_err(|err| err.to_string())?
    } else {
        let name = root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("helmer Project")
            .to_string();
        let config = ProjectConfig::new(name);
        write_project_config(root, &config)?;
        config
    };

    ensure_project_layout(root, &config)?;
    write_default_material(root, &config)?;

    Ok(config)
}

pub fn save_project_config(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    ensure_project_layout(root, config)?;
    write_project_config(root, config)?;
    write_default_material(root, config)?;
    Ok(())
}

fn write_project_config(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    let config_path = ProjectConfig::config_path(root);
    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(4)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(config, pretty).map_err(|err| err.to_string())?;
    fs::write(config_path, data).map_err(|err| err.to_string())
}

fn ensure_project_layout(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    fs::create_dir_all(config.assets_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.models_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.textures_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.materials_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.scenes_root(root)).map_err(|err| err.to_string())?;
    fs::create_dir_all(config.scripts_root(root)).map_err(|err| err.to_string())?;
    write_luau_support_files(root, config)?;
    Ok(())
}

fn write_default_material(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    let materials_root = config.materials_root(root);
    let default_path = materials_root.join(DEFAULT_MATERIAL_FILE);
    if default_path.exists() {
        return Ok(());
    }

    let template = default_material_template();
    fs::write(default_path, template).map_err(|err| err.to_string())
}

fn write_luau_support_files(root: &Path, config: &ProjectConfig) -> Result<(), String> {
    let scripts_root = config.scripts_root(root);
    fs::create_dir_all(&scripts_root).map_err(|err| err.to_string())?;

    let api_path = scripts_root.join(HELMER_LUAU_API_FILE_NAME);
    fs::write(&api_path, helmer_luau_api_types()).map_err(|err| err.to_string())?;

    let luaurc_path = root.join(LUAURC_FILE_NAME);
    let should_write_luaurc = match fs::read_to_string(&luaurc_path) {
        Ok(existing) => existing.contains(LUAURC_GENERATED_MARKER),
        Err(err) if err.kind() == ErrorKind::NotFound => true,
        Err(err) => return Err(err.to_string()),
    };

    if should_write_luaurc {
        fs::write(&luaurc_path, luau_rc_template(config)).map_err(|err| err.to_string())?;

        let vscode_settings_path = config.vscode_config_root(root);
        fs::create_dir_all(&vscode_settings_path).map_err(|err| err.to_string())?;
        fs::write(
            &vscode_settings_path.join("settings.json"),
            luau_vscode_settings_template(),
        )
        .map_err(|err| err.to_string())?;
    }

    Ok(())
}

fn luau_rc_template(config: &ProjectConfig) -> String {
    let scripts_glob = format!(
        "{}/**",
        config
            .scripts_dir
            .trim_start_matches("./")
            .replace('\\', "/")
    );
    format!(
        "{{\n  \"__generated_by_helmer__\": true,\n  \"languageMode\": \"strict\",\n  \"paths\": [\n    \"{}\"\n  ],\n  \"globals\": [\n    \"ecs\",\n    \"input\",\n    \"entity_id\"\n  ]\n}}\n",
        scripts_glob
    )
}

fn luau_vscode_settings_template() -> &'static str {
    r#"{
    "luau-lsp.plugin.enabled": true,
    "luau-lsp.platform.type": "standard",
    "luau-lsp.sourcemap.enabled": false,
    "luau-lsp.sourcemap.autogenerate": false,
    "luau-lsp.types.definitionFiles": {
        "@helmer": "assets/scripts/helmer_api.d.luau",
    }
}"#
}

fn helmer_luau_api_types() -> &'static str {
    r#"-- Generated by helmer_editor. Update by reopening or saving project preferences.

type EntityId = number

type Vec2 = { x: number, y: number, [number]: number }
type Vec3 = { x: number, y: number, z: number, [number]: number }
type Quat = { x: number, y: number, z: number, w: number, [number]: number }

type Transform = {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}

type TransformPatch = {
    position: Vec3?,
    rotation: Quat?,
    scale: Vec3?,
}

type SplineData = {
    points: { Vec3 },
    closed: boolean,
    tension: number,
    mode: string,
}

type SplinePatch = {
    points: { Vec3 }?,
    closed: boolean?,
    tension: number?,
    mode: string?,
}

type LightData = {
    type: string,
    color: Vec3,
    intensity: number,
    angle: number?,
}

type LightPatch = {
    type: string?,
    color: Vec3?,
    intensity: number?,
    angle: number?,
}

type CameraData = {
    fov_y_rad: number,
    aspect_ratio: number,
    near_plane: number,
    far_plane: number,
    active: boolean,
}

type CameraPatch = {
    fov_y_rad: number?,
    aspect_ratio: number?,
    near_plane: number?,
    far_plane: number?,
    active: boolean?,
}

type MeshRendererData = {
    source: string,
    material: string?,
    casts_shadow: boolean,
    visible: boolean,
}

type MeshRendererPatch = {
    source: string?,
    material: string?,
    casts_shadow: boolean?,
    visible: boolean?,
}

type ScriptData = {
    path: string,
    language: string,
}

type AudioBus = "Master" | "Music" | "Sfx" | "Ui" | "Ambience" | "World" | number
type AudioBusValue = AudioBus | string
type AudioPlaybackStateValue = "Playing" | "Paused" | "Stopped" | string | number

type AudioEmitterData = {
    path: string?,
    streaming: boolean,
    bus: AudioBus,
    volume: number,
    pitch: number,
    looping: boolean,
    spatial: boolean,
    min_distance: number,
    max_distance: number,
    rolloff: number,
    spatial_blend: number,
    playback_state: string,
    play_on_spawn: boolean,
    clip_id: number?,
}

type AudioEmitterPatch = {
    path: string?,
    streaming: boolean?,
    bus: AudioBusValue?,
    volume: number?,
    pitch: number?,
    looping: boolean?,
    spatial: boolean?,
    min_distance: number?,
    max_distance: number?,
    rolloff: number?,
    spatial_blend: number?,
    playback_state: AudioPlaybackStateValue?,
    play_on_spawn: boolean?,
}

type AudioListenerData = {
    enabled: boolean,
}

type AudioListenerPatch = {
    enabled: boolean?,
}

type AudioStreamingConfig = {
    buffer_frames: number,
    chunk_frames: number,
}

type PhysicsCombineRule = "Average" | "Min" | "Multiply" | "Max"
type KinematicMode = "PositionBased" | "VelocityBased"
type PhysicsJointKind = "Fixed" | "Spherical" | "Revolute" | "Prismatic"
type MeshColliderKind = "TriMesh" | "ConvexHull"
type MeshColliderLod = "Lod0" | "Lod1" | "Lod2" | "Lowest" | number
type PhysicsShapeCastStatus =
    "NoHit"
    | "Converged"
    | "OutOfIterations"
    | "Failed"
    | "PenetratingOrWithinTargetDist"

type PhysicsBodyKindData =
    { type: "Dynamic", mass: number }
    | { type: "Kinematic", mode: KinematicMode }
    | { type: "Fixed" }
type PhysicsBodyKindValue = PhysicsBodyKindData | "Dynamic" | "Kinematic" | "Fixed" | string

type ColliderShapeData =
    { type: "Cuboid" }
    | { type: "Sphere" }
    | { type: "CapsuleY" }
    | { type: "CylinderY" }
    | { type: "ConeY" }
    | { type: "RoundCuboid", border_radius: number }
    | { type: "Mesh", mesh_id: number?, lod: MeshColliderLod, kind: MeshColliderKind }
type ColliderShapePatch =
    { type: "Cuboid" }
    | { type: "Sphere" }
    | { type: "CapsuleY" }
    | { type: "CylinderY" }
    | { type: "ConeY" }
    | { type: "RoundCuboid", border_radius: number? }
    | { type: "Mesh", mesh_id: number?, lod: MeshColliderLod?, kind: MeshColliderKind? }
type ColliderShapeValue = ColliderShapePatch | string

type PhysicsQueryFilterData = {
    flags: number,
    groups_memberships: number,
    groups_filter: number,
    use_groups: boolean,
    exclude_fixed_flag: number?,
    exclude_kinematic_flag: number?,
    exclude_dynamic_flag: number?,
    exclude_sensors_flag: number?,
    exclude_solids_flag: number?,
}

type PhysicsQueryFilterPatch = {
    flags: number?,
    groups_memberships: number?,
    groups_filter: number?,
    use_groups: boolean?,
}

type ColliderPropertiesData = {
    friction: number,
    restitution: number,
    density: number,
    is_sensor: boolean,
    enabled: boolean,
    collision_memberships: number,
    collision_filter: number,
    solver_memberships: number,
    solver_filter: number,
    friction_combine_rule: PhysicsCombineRule,
    restitution_combine_rule: PhysicsCombineRule,
    translation_offset: Vec3,
    rotation_offset: Quat,
}

type ColliderPropertiesPatch = {
    friction: number?,
    restitution: number?,
    density: number?,
    is_sensor: boolean?,
    enabled: boolean?,
    collision_memberships: number?,
    collision_filter: number?,
    solver_memberships: number?,
    solver_filter: number?,
    friction_combine_rule: PhysicsCombineRule?,
    restitution_combine_rule: PhysicsCombineRule?,
    translation_offset: Vec3?,
    rotation_offset: Quat?,
}

type ColliderInheritanceData = {
    friction: boolean,
    restitution: boolean,
    density: boolean,
    is_sensor: boolean,
    enabled: boolean,
    collision_memberships: boolean,
    collision_filter: boolean,
    solver_memberships: boolean,
    solver_filter: boolean,
    friction_combine_rule: boolean,
    restitution_combine_rule: boolean,
    translation_offset: boolean,
    rotation_offset: boolean,
}

type ColliderInheritancePatch = {
    friction: boolean?,
    restitution: boolean?,
    density: boolean?,
    is_sensor: boolean?,
    enabled: boolean?,
    collision_memberships: boolean?,
    collision_filter: boolean?,
    solver_memberships: boolean?,
    solver_filter: boolean?,
    friction_combine_rule: boolean?,
    restitution_combine_rule: boolean?,
    translation_offset: boolean?,
    rotation_offset: boolean?,
}

type RigidBodyPropertiesData = {
    linear_damping: number,
    angular_damping: number,
    gravity_scale: number,
    ccd_enabled: boolean,
    can_sleep: boolean,
    sleeping: boolean,
    dominance_group: number,
    lock_translation_x: boolean,
    lock_translation_y: boolean,
    lock_translation_z: boolean,
    lock_rotation_x: boolean,
    lock_rotation_y: boolean,
    lock_rotation_z: boolean,
    linear_velocity: Vec3,
    angular_velocity: Vec3,
}

type RigidBodyPropertiesPatch = {
    linear_damping: number?,
    angular_damping: number?,
    gravity_scale: number?,
    ccd_enabled: boolean?,
    can_sleep: boolean?,
    sleeping: boolean?,
    dominance_group: number?,
    lock_translation_x: boolean?,
    lock_translation_y: boolean?,
    lock_translation_z: boolean?,
    lock_rotation_x: boolean?,
    lock_rotation_y: boolean?,
    lock_rotation_z: boolean?,
    linear_velocity: Vec3?,
    angular_velocity: Vec3?,
}

type RigidBodyInheritanceData = {
    linear_damping: boolean,
    angular_damping: boolean,
    gravity_scale: boolean,
    ccd_enabled: boolean,
    can_sleep: boolean,
    sleeping: boolean,
    dominance_group: boolean,
    lock_translation_x: boolean,
    lock_translation_y: boolean,
    lock_translation_z: boolean,
    lock_rotation_x: boolean,
    lock_rotation_y: boolean,
    lock_rotation_z: boolean,
    linear_velocity: boolean,
    angular_velocity: boolean,
}

type RigidBodyInheritancePatch = {
    linear_damping: boolean?,
    angular_damping: boolean?,
    gravity_scale: boolean?,
    ccd_enabled: boolean?,
    can_sleep: boolean?,
    sleeping: boolean?,
    dominance_group: boolean?,
    lock_translation_x: boolean?,
    lock_translation_y: boolean?,
    lock_translation_z: boolean?,
    lock_rotation_x: boolean?,
    lock_rotation_y: boolean?,
    lock_rotation_z: boolean?,
    linear_velocity: boolean?,
    angular_velocity: boolean?,
}

type PhysicsJointLimitsData = {
    min: number,
    max: number,
    [number]: number,
}

type PhysicsJointMotorData = {
    enabled: boolean,
    target_position: number,
    target_velocity: number,
    stiffness: number,
    damping: number,
    max_force: number,
}

type PhysicsJointMotorPatch = {
    enabled: boolean?,
    target_position: number?,
    target_velocity: number?,
    stiffness: number?,
    damping: number?,
    max_force: number?,
}

type PhysicsJointData = {
    target: EntityId?,
    kind: PhysicsJointKind,
    contacts_enabled: boolean,
    local_anchor1: Vec3,
    local_anchor2: Vec3,
    local_axis1: Vec3,
    local_axis2: Vec3,
    limit_enabled: boolean,
    limits: PhysicsJointLimitsData,
    motor: PhysicsJointMotorData,
}

type PhysicsJointPatch = {
    target: EntityId?,
    kind: PhysicsJointKind?,
    contacts_enabled: boolean?,
    local_anchor1: Vec3?,
    local_anchor2: Vec3?,
    local_axis1: Vec3?,
    local_axis2: Vec3?,
    limit_enabled: boolean?,
    limits: PhysicsJointLimitsData?,
    motor: PhysicsJointMotorPatch?,
}

type CharacterControllerData = {
    up: Vec3,
    offset: number,
    slide: boolean,
    autostep_max_height: number,
    autostep_min_width: number,
    autostep_include_dynamic_bodies: boolean,
    max_slope_climb_angle: number,
    min_slope_slide_angle: number,
    snap_to_ground: number,
    normal_nudge_factor: number,
    apply_impulses_to_dynamic_bodies: boolean,
    character_mass: number,
}

type CharacterControllerPatch = {
    up: Vec3?,
    offset: number?,
    slide: boolean?,
    autostep_max_height: number?,
    autostep_min_width: number?,
    autostep_include_dynamic_bodies: boolean?,
    max_slope_climb_angle: number?,
    min_slope_slide_angle: number?,
    snap_to_ground: number?,
    normal_nudge_factor: number?,
    apply_impulses_to_dynamic_bodies: boolean?,
    character_mass: number?,
}

type CharacterControllerInputData = {
    desired_translation: Vec3,
}

type CharacterControllerInputPatch = {
    desired_translation: Vec3?,
}

type CharacterControllerOutputData = {
    effective_translation: Vec3,
    grounded: boolean,
    sliding_down_slope: boolean,
    collision_count: number,
}

type PhysicsRayCastData = {
    origin: Vec3,
    direction: Vec3,
    max_toi: number,
    solid: boolean,
    filter: PhysicsQueryFilterData,
    exclude_self: boolean,
}

type PhysicsRayCastPatch = {
    origin: Vec3?,
    direction: Vec3?,
    max_toi: number?,
    solid: boolean?,
    filter: PhysicsQueryFilterPatch?,
    exclude_self: boolean?,
}

type PhysicsRayCastHitData = {
    has_hit: boolean,
    hit_entity: EntityId?,
    point: Vec3,
    normal: Vec3,
    toi: number,
}

type PhysicsPointProjectionData = {
    point: Vec3,
    solid: boolean,
    filter: PhysicsQueryFilterData,
    exclude_self: boolean,
}

type PhysicsPointProjectionPatch = {
    point: Vec3?,
    solid: boolean?,
    filter: PhysicsQueryFilterPatch?,
    exclude_self: boolean?,
}

type PhysicsPointProjectionHitData = {
    has_hit: boolean,
    hit_entity: EntityId?,
    projected_point: Vec3,
    is_inside: boolean,
    distance: number,
}

type PhysicsShapeCastData = {
    shape: ColliderShapeData,
    scale: Vec3,
    position: Vec3,
    rotation: Quat,
    velocity: Vec3,
    max_time_of_impact: number,
    target_distance: number,
    stop_at_penetration: boolean,
    compute_impact_geometry_on_penetration: boolean,
    filter: PhysicsQueryFilterData,
    exclude_self: boolean,
}

type PhysicsShapeCastPatch = {
    shape: ColliderShapeValue?,
    scale: Vec3?,
    position: Vec3?,
    rotation: Quat?,
    velocity: Vec3?,
    max_time_of_impact: number?,
    target_distance: number?,
    stop_at_penetration: boolean?,
    compute_impact_geometry_on_penetration: boolean?,
    filter: PhysicsQueryFilterPatch?,
    exclude_self: boolean?,
}

type PhysicsShapeCastHitData = {
    has_hit: boolean,
    hit_entity: EntityId?,
    toi: number,
    witness1: Vec3,
    witness2: Vec3,
    normal1: Vec3,
    normal2: Vec3,
    status: PhysicsShapeCastStatus,
}

type PhysicsWorldDefaultsData = {
    gravity: Vec3,
    collider_properties: ColliderPropertiesData,
    rigid_body_properties: RigidBodyPropertiesData,
}

type PhysicsWorldDefaultsPatch = {
    gravity: Vec3?,
    collider_properties: ColliderPropertiesPatch?,
    rigid_body_properties: RigidBodyPropertiesPatch?,
}

type PhysicsVelocityData = {
    linear: Vec3?,
    angular: Vec3?,
}

type PhysicsVelocityPatch = {
    linear: Vec3?,
    angular: Vec3?,
    wake_up: boolean?,
}

type PhysicsData = {
    collider_shape: ColliderShapeData?,
    body_kind: PhysicsBodyKindData?,
    collider_properties: ColliderPropertiesData?,
    collider_inheritance: ColliderInheritanceData?,
    rigid_body_properties: RigidBodyPropertiesData?,
    rigid_body_inheritance: RigidBodyInheritanceData?,
    joint: PhysicsJointData?,
    character_controller: CharacterControllerData?,
    character_input: CharacterControllerInputData?,
    character_output: CharacterControllerOutputData?,
    ray_cast: PhysicsRayCastData?,
    ray_cast_hit: PhysicsRayCastHitData?,
    point_projection: PhysicsPointProjectionData?,
    point_projection_hit: PhysicsPointProjectionHitData?,
    shape_cast: PhysicsShapeCastData?,
    shape_cast_hit: PhysicsShapeCastHitData?,
    world_defaults: PhysicsWorldDefaultsData?,
    has_handle: boolean,
}

type PhysicsPatch = {
    collider_shape: ColliderShapeValue?,
    body_kind: PhysicsBodyKindValue?,
    collider_properties: ColliderPropertiesPatch?,
    collider_inheritance: ColliderInheritancePatch?,
    rigid_body_properties: RigidBodyPropertiesPatch?,
    rigid_body_inheritance: RigidBodyInheritancePatch?,
    joint: PhysicsJointPatch?,
    character_controller: CharacterControllerPatch?,
    character_input: CharacterControllerInputPatch?,
    ray_cast: PhysicsRayCastPatch?,
    point_projection: PhysicsPointProjectionPatch?,
    shape_cast: PhysicsShapeCastPatch?,
    world_defaults: PhysicsWorldDefaultsPatch?,
}

type DynamicFieldValue = boolean | number | string | Vec3
type DynamicFields = { [string]: DynamicFieldValue }

type DynamicComponentData = {
    name: string,
    fields: DynamicFields,
}

type EcsComponentName =
    "name"
    | "transform"
    | "camera"
    | "light"
    | "mesh"
    | "mesh_renderer"
    | "spline"
    | "spline_follower"
    | "look_at"
    | "entity_follower"
    | "animator"
    | "scene"
    | "audio"
    | "audio_emitter"
    | "audio_listener"
    | "script"
    | "dynamic"
    | "physics"
    | "collider_shape"
    | "dynamic_rigid_body"
    | "kinematic_rigid_body"
    | "fixed_collider"
    | "collider_properties"
    | "collider_inheritance"
    | "rigid_body_properties"
    | "rigid_body_inheritance"
    | "physics_joint"
    | "character_controller"
    | "character_controller_input"
    | "character_controller_output"
    | "physics_ray_cast"
    | "physics_ray_cast_hit"
    | "physics_point_projection"
    | "physics_point_projection_hit"
    | "physics_shape_cast"
    | "physics_shape_cast_hit"
    | "physics_world_defaults"

type HelmerEcs = {
    list_entities: () -> { EntityId },
    spawn_entity: (name: string?) -> EntityId,
    entity_exists: (id: EntityId) -> boolean,
    find_entity_by_name: (name: string) -> EntityId?,
    get_entity_name: (id: EntityId) -> string?,
    set_entity_name: (id: EntityId, name: string) -> boolean,
    delete_entity: (id: EntityId) -> boolean,
    has_component: (id: EntityId, component: EcsComponentName) -> boolean,
    add_component: (id: EntityId, component: EcsComponentName) -> boolean,
    remove_component: (id: EntityId, component: EcsComponentName) -> boolean,

    get_transform: (id: EntityId) -> Transform?,
    set_transform: (id: EntityId, data: TransformPatch) -> boolean,

    get_spline: (id: EntityId) -> SplineData?,
    set_spline: (id: EntityId, data: SplinePatch) -> boolean,
    add_spline_point: (id: EntityId, point: Vec3) -> boolean,
    set_spline_point: (id: EntityId, index: number, point: Vec3) -> boolean,
    remove_spline_point: (id: EntityId, index: number) -> boolean,
    sample_spline: (id: EntityId, t: number) -> Vec3?,
    spline_length: (id: EntityId, samples: number?) -> number?,
    follow_spline: (id: EntityId, spline_id: EntityId?, speed: number?, looped: boolean?) -> boolean,

    set_animator_enabled: (id: EntityId, enabled: boolean) -> boolean,
    set_animator_time_scale: (id: EntityId, time_scale: number) -> boolean,
    set_animator_param_float: (id: EntityId, name: string, value: number) -> boolean,
    set_animator_param_bool: (id: EntityId, name: string, value: boolean) -> boolean,
    trigger_animator: (id: EntityId, name: string) -> boolean,
    get_animator_clips: (id: EntityId, layer_index: number?) -> { string }?,
    play_anim_clip: (id: EntityId, name: string, layer_index: number?) -> boolean,

    get_light: (id: EntityId) -> LightData?,
    set_light: (id: EntityId, data: LightPatch) -> boolean,

    get_camera: (id: EntityId) -> CameraData?,
    set_camera: (id: EntityId, data: CameraPatch) -> boolean,
    set_active_camera: (id: EntityId) -> boolean,
    get_viewport_mode: () -> string,
    set_viewport_mode: (mode: string) -> boolean,
    get_viewport_preview_camera: () -> EntityId?,
    set_viewport_preview_camera: (id: EntityId?) -> boolean,

    get_mesh_renderer: (id: EntityId) -> MeshRendererData?,
    set_mesh_renderer: (id: EntityId, data: MeshRendererPatch) -> boolean,

    get_scene_asset: (id: EntityId) -> string?,
    set_scene_asset: (id: EntityId, path: string) -> boolean,
    open_scene: (path: string) -> boolean,
    switch_scene: (path: string) -> boolean,

    get_script: (id: EntityId) -> ScriptData?,
    set_script: (id: EntityId, path: string, language: string?) -> boolean,

    get_audio_emitter: (id: EntityId) -> AudioEmitterData?,
    set_audio_emitter: (id: EntityId, data: AudioEmitterPatch) -> boolean,
    get_audio_listener: (id: EntityId) -> AudioListenerData?,
    set_audio_listener: (id: EntityId, data: AudioListenerPatch) -> boolean,

    set_audio_enabled: (enabled: boolean) -> boolean,
    get_audio_enabled: () -> boolean,
    list_audio_buses: () -> { AudioBus },
    create_audio_bus: (name: string?) -> AudioBus?,
    remove_audio_bus: (bus: AudioBusValue) -> boolean,
    get_audio_bus_name: (bus: AudioBusValue) -> string?,
    set_audio_bus_name: (bus: AudioBusValue, name: string) -> boolean,
    set_audio_bus_volume: (bus: AudioBusValue, volume: number) -> boolean,
    get_audio_bus_volume: (bus: AudioBusValue) -> number?,
    set_audio_scene_volume: (scene_id: number, volume: number) -> boolean,
    get_audio_scene_volume: (scene_id: number) -> number?,
    clear_audio_emitters: () -> boolean,
    set_audio_head_width: (width: number) -> boolean,
    get_audio_head_width: () -> number?,
    set_audio_speed_of_sound: (speed: number) -> boolean,
    get_audio_speed_of_sound: () -> number?,
    set_audio_streaming_config: (buffer_frames: number, chunk_frames: number) -> boolean,
    get_audio_streaming_config: () -> AudioStreamingConfig?,

    get_physics: (id: EntityId) -> PhysicsData?,
    set_physics: (id: EntityId, data: PhysicsPatch) -> boolean,
    clear_physics: (id: EntityId) -> boolean,
    get_physics_world_defaults: (id: EntityId) -> PhysicsWorldDefaultsData?,
    set_physics_world_defaults: (id: EntityId, data: PhysicsWorldDefaultsPatch) -> boolean,
    get_character_controller_output: (id: EntityId) -> CharacterControllerOutputData?,
    get_physics_ray_cast_hit: (id: EntityId) -> PhysicsRayCastHitData?,
    get_physics_point_projection_hit: (id: EntityId) -> PhysicsPointProjectionHitData?,
    get_physics_shape_cast_hit: (id: EntityId) -> PhysicsShapeCastHitData?,
    get_physics_velocity: (id: EntityId) -> PhysicsVelocityData?,
    set_physics_velocity: (id: EntityId, data: PhysicsVelocityPatch) -> boolean,
    add_force: (id: EntityId, force: Vec3, wake_up: boolean?) -> boolean,
    add_torque: (id: EntityId, torque: Vec3, wake_up: boolean?) -> boolean,
    add_force_at_point: (id: EntityId, force: Vec3, point: Vec3, wake_up: boolean?) -> boolean,
    add_persistent_force: (id: EntityId, force: Vec3, wake_up: boolean?) -> boolean,
    set_persistent_force: (id: EntityId, force: Vec3, wake_up: boolean?) -> boolean,
    add_persistent_torque: (id: EntityId, torque: Vec3, wake_up: boolean?) -> boolean,
    set_persistent_torque: (id: EntityId, torque: Vec3, wake_up: boolean?) -> boolean,
    add_persistent_force_at_point: (id: EntityId, force: Vec3, point: Vec3, wake_up: boolean?) -> boolean,
    clear_persistent_forces: (id: EntityId) -> boolean,
    apply_impulse: (id: EntityId, impulse: Vec3, wake_up: boolean?) -> boolean,
    apply_angular_impulse: (id: EntityId, impulse: Vec3, wake_up: boolean?) -> boolean,
    apply_torque_impulse: (id: EntityId, impulse: Vec3, wake_up: boolean?) -> boolean,
    apply_impulse_at_point: (id: EntityId, impulse: Vec3, point: Vec3, wake_up: boolean?) -> boolean,
    set_physics_running: (running: boolean) -> boolean,
    get_physics_running: () -> boolean,
    set_physics_gravity: (gravity: Vec3) -> boolean,
    get_physics_gravity: () -> Vec3?,

    list_dynamic_components: (id: EntityId) -> { DynamicComponentData }?,
    get_dynamic_component: (id: EntityId, name: string) -> DynamicFields?,
    set_dynamic_component: (id: EntityId, name: string, fields: DynamicFields) -> boolean,
    get_dynamic_field: (id: EntityId, component_name: string, field_name: string) -> DynamicFieldValue?,
    set_dynamic_field: (id: EntityId, component_name: string, field_name: string, value: DynamicFieldValue) -> boolean,
    remove_dynamic_component: (id: EntityId, name: string) -> boolean,
    remove_dynamic_field: (id: EntityId, component_name: string, field_name: string) -> boolean,
}

type InputHandle = any
type InputModifiers = { shift: boolean, ctrl: boolean, alt: boolean, super: boolean }

type HelmerInput = {
    keys: { [string]: InputHandle },
    mouse_buttons: { [string]: InputHandle },
    gamepad_buttons: { [string]: InputHandle },
    gamepad_axes: { [string]: InputHandle },

    key: (name: string) -> InputHandle?,
    mouse_button: (name: string) -> InputHandle?,
    gamepad_button: (name: string) -> InputHandle?,
    gamepad_axis_handle: (name: string) -> InputHandle?,
    gamepad_axis_ref: (name: string) -> InputHandle?,

    key_down: (key: InputHandle | string | number) -> boolean,
    key_pressed: (key: InputHandle | string | number) -> boolean,
    key_released: (key: InputHandle | string | number) -> boolean,

    mouse_down: (button: InputHandle | string | number) -> boolean,
    mouse_pressed: (button: InputHandle | string | number) -> boolean,
    mouse_released: (button: InputHandle | string | number) -> boolean,

    cursor: () -> Vec2?,
    cursor_delta: () -> Vec2?,
    wheel: () -> Vec2?,
    window_size: () -> Vec2?,
    scale_factor: () -> number?,
    modifiers: () -> InputModifiers?,
    wants_keyboard: () -> boolean,
    wants_pointer: () -> boolean,
    cursor_grab_mode: () -> string,
    set_cursor_visible: (visible: boolean) -> boolean,
    set_cursor_grab: (mode: "none" | "confined" | "locked" | boolean | nil) -> boolean,
    reset_cursor_control: () -> boolean,

    gamepad_ids: () -> { number },
    gamepad_count: () -> number,
    gamepad_axis: (axis: InputHandle | string, gamepad_id: number?) -> number,
    gamepad_button_down: (button: InputHandle | string, gamepad_id: number?) -> boolean,
    gamepad_button_pressed: (button: InputHandle | string, gamepad_id: number?) -> boolean,
    gamepad_button_released: (button: InputHandle | string, gamepad_id: number?) -> boolean,
    gamepad_trigger: (side: InputHandle | string, gamepad_id: number?) -> number,
}

declare entity_id: EntityId
declare ecs: HelmerEcs
declare input: HelmerInput
declare print: (...any) -> ()
"#
}

pub fn default_material_template() -> &'static str {
    "(\n    albedo: (1.0, 1.0, 1.0, 1.0),\n    metallic: 0.0,\n    roughness: 0.8,\n    ao: 1.0,\n    emission_strength: 0.0,\n    emission_color: (0.0, 0.0, 0.0),\n    albedo_texture: None,\n    normal_texture: None,\n    metallic_roughness_texture: None,\n    emission_texture: None,\n)\n"
}

pub fn default_scene_template() -> &'static str {
    "(\n    version: 1,\n    entities: [],\n)\n"
}

pub fn default_animation_template() -> &'static str {
    "(\n    version: 1,\n    name: \"New Animation\",\n    tracks: [],\n    clips: [],\n)\n"
}

pub fn default_script_template_simple() -> &'static str {
    r#"--!strict
local mover: number = -1
local t = 0.0
local speed = 10.0

function on_start()
    mover = ecs.spawn_entity("Orbiting Cube")
    ecs.set_mesh_renderer(mover, {
        source = "Cube",
        casts_shadow = true,
        visible = true,
    })
    ecs.set_transform(mover, { position = { x = 0.0, y = 0.5, z = 0.0 } })
end

function on_update(dt: number)
    if mover == nil then return end
    t = t + (dt * speed)
    local radius = 2.0
    local x = math.cos(t) * radius
    local y = math.cos(t / 2) * radius
    local z = math.sin(t) * radius
    ecs.set_transform(mover, { position = { x = x, y = y, z = z } })
end
"#
}

pub fn default_script_template_full() -> &'static str {
    r#"--!strict
-- helmer Luau API
-- Entry points: on_start(), on_update(dt), on_stop()
-- Globals:
--   entity_id : u64 id of the entity that owns this script
--   print(...) : log values to the editor console
-- input table:
--   input.key_down(key) -> bool
--   input.key_pressed(key) -> bool
--   input.key_released(key) -> bool
--   input.mouse_down(button) -> bool
--   input.mouse_pressed(button) -> bool
--   input.mouse_released(button) -> bool
--   input.cursor() -> {x,y}
--   input.cursor_delta() -> {x,y}
--   input.wheel() -> {x,y}
--   input.window_size() -> {x,y}
--   input.scale_factor() -> number
--   input.modifiers() -> {shift, ctrl, alt, super}
--   input.wants_keyboard() -> bool
--   input.wants_pointer() -> bool
--   input.cursor_grab_mode() -> "none"|"confined"|"locked"
--   input.set_cursor_visible(visible) -> bool
--   input.set_cursor_grab(mode) -> bool
--   input.reset_cursor_control() -> bool
--   input.gamepad_ids() -> {id, ...}
--   input.gamepad_count() -> number
--   input.gamepad_axis(axis, id?) -> number
--   input.gamepad_button_down(button, id?) -> bool
--   input.gamepad_button_pressed(button, id?) -> bool
--   input.gamepad_button_released(button, id?) -> bool
--   input.gamepad_trigger(side, id?) -> number
--   input.key(name) -> key|nil
--   input.mouse_button(name) -> button|nil
--   input.gamepad_button(name) -> button|nil
--   input.gamepad_axis_handle(name) -> axis|nil
--   input.gamepad_axis_ref(name) -> axis|nil
-- input constants:
--   input.keys.<Name>, input.mouse_buttons.<Name>, input.gamepad_buttons.<Name>, input.gamepad_axes.<Name>
-- ecs table:
--   ecs.list_entities() -> {id, ...}
--   ecs.spawn_entity(name?) -> id
--   ecs.entity_exists(id) -> bool
--   ecs.find_entity_by_name(name) -> id|nil
--   ecs.get_entity_name(id) -> string|nil
--   ecs.set_entity_name(id, name) -> bool (empty name removes Name)
--   ecs.delete_entity(id) -> bool
--   ecs.has_component(id, component) -> bool
--   ecs.add_component(id, component) -> bool
--   ecs.remove_component(id, component) -> bool
--     component:
--       "name"|"transform"|"camera"|"light"|"mesh"|"mesh_renderer"|"spline"|"spline_follower"|"look_at"|"entity_follower"|"animator"|"scene"|"audio"|"audio_emitter"|"audio_listener"|"script"|"dynamic"|"physics"|"collider_shape"|"dynamic_rigid_body"|"kinematic_rigid_body"|"fixed_collider"|"collider_properties"|"collider_inheritance"|"rigid_body_properties"|"rigid_body_inheritance"|"physics_joint"|"character_controller"|"character_controller_input"|"character_controller_output"|"physics_ray_cast"|"physics_ray_cast_hit"|"physics_point_projection"|"physics_point_projection_hit"|"physics_shape_cast"|"physics_shape_cast_hit"|"physics_world_defaults"
--   ecs.get_transform(id) -> {position={x,y,z}, rotation={x,y,z,w}, scale={x,y,z}}|nil
--   ecs.set_transform(id, {position?, rotation?, scale?}) -> bool
--   ecs.get_spline(id) -> {points={{x,y,z}}, closed, tension, mode}|nil
--   ecs.set_spline(id, {points?, closed?, tension?, mode?}) -> bool
--   ecs.add_spline_point(id, point) -> bool
--   ecs.set_spline_point(id, index, point) -> bool
--   ecs.remove_spline_point(id, index) -> bool
--   ecs.sample_spline(id, t) -> point|nil
--   ecs.spline_length(id, samples?) -> number|nil
--   ecs.follow_spline(id, spline_id?, speed?, looped?) -> bool
--   ecs.set_animator_enabled(id, enabled) -> bool
--   ecs.set_animator_time_scale(id, value) -> bool
--   ecs.set_animator_param_float(id, name, value) -> bool
--   ecs.set_animator_param_bool(id, name, value) -> bool
--   ecs.trigger_animator(id, name) -> bool
--   ecs.get_animator_clips(id, layer?) -> {name, ...}|nil
--   ecs.play_anim_clip(id, name, layer?) -> bool
--   ecs.get_light(id) -> {type, color={x,y,z}, intensity, angle?}|nil
--   ecs.set_light(id, {type?, color?, intensity?, angle?}) -> bool
--   ecs.get_camera(id) -> {fov_y_rad, aspect_ratio, near_plane, far_plane, active}|nil
--   ecs.set_camera(id, {fov_y_rad?, aspect_ratio?, near_plane?, far_plane?, active?}) -> bool
--   ecs.set_active_camera(id) -> bool
--   ecs.get_viewport_mode() -> "editor"|"gameplay"
--   ecs.set_viewport_mode(mode) -> bool
--   ecs.get_viewport_preview_camera() -> id|nil
--   ecs.set_viewport_preview_camera(id|nil) -> bool
--   ecs.get_mesh_renderer(id) -> {source, material?, casts_shadow, visible}|nil
--   ecs.set_mesh_renderer(id, {source?, material?, casts_shadow?, visible?}) -> bool
--   ecs.get_scene_asset(id) -> path|nil
--   ecs.set_scene_asset(id, path) -> bool
--   ecs.open_scene(path) -> bool
--   ecs.switch_scene(path) -> bool
--   ecs.get_script(id) -> {path, language}|nil
--   ecs.set_script(id, path, language?) -> bool
--   ecs.list_dynamic_components(id) -> [{name, fields={...}}, ...]|nil
--   ecs.get_dynamic_component(id, name) -> fields|nil
--   ecs.set_dynamic_component(id, name, fields_table) -> bool
--   ecs.get_dynamic_field(id, comp_name, field_name) -> value|nil
--   ecs.set_dynamic_field(id, comp_name, field_name, value) -> bool
--   ecs.remove_dynamic_component(id, name) -> bool
--   ecs.remove_dynamic_field(id, comp_name, field_name) -> bool
-- audio:
--   ecs.get_audio_emitter(id) -> {path?, streaming, bus, volume, pitch, looping, spatial, min_distance, max_distance, rolloff, spatial_blend, playback_state, play_on_spawn, clip_id?}|nil
--   ecs.set_audio_emitter(id, data) -> bool
--   ecs.get_audio_listener(id) -> {enabled}|nil
--   ecs.set_audio_listener(id, {enabled?}) -> bool
--   ecs.set_audio_enabled(enabled) -> bool
--   ecs.get_audio_enabled() -> bool
--   ecs.list_audio_buses() -> {bus, ...}
--   ecs.create_audio_bus(name?) -> bus|nil
--   ecs.remove_audio_bus(bus) -> bool
--   ecs.get_audio_bus_name(bus) -> string|nil
--   ecs.set_audio_bus_name(bus, name) -> bool
--   ecs.set_audio_bus_volume(bus, volume) -> bool
--   ecs.get_audio_bus_volume(bus) -> number|nil
--   ecs.set_audio_scene_volume(scene_id, volume) -> bool
--   ecs.get_audio_scene_volume(scene_id) -> number|nil
--   ecs.clear_audio_emitters() -> bool
--   ecs.set_audio_head_width(width) -> bool
--   ecs.get_audio_head_width() -> number|nil
--   ecs.set_audio_speed_of_sound(speed) -> bool
--   ecs.get_audio_speed_of_sound() -> number|nil
--   ecs.set_audio_streaming_config(buffer_frames, chunk_frames) -> bool
--   ecs.get_audio_streaming_config() -> {buffer_frames, chunk_frames}|nil
-- physics:
--   ecs.get_physics(id) -> table|nil
--   ecs.set_physics(id, patch_table) -> bool
--   ecs.clear_physics(id) -> bool
--   ecs.get_physics_world_defaults(id) -> {gravity, collider_properties, rigid_body_properties}|nil
--   ecs.set_physics_world_defaults(id, patch) -> bool
--   ecs.get_character_controller_output(id) -> {effective_translation, grounded, sliding_down_slope, collision_count}|nil
--   ecs.get_physics_ray_cast_hit(id) -> {has_hit, hit_entity?, point, normal, toi}|nil
--   ecs.get_physics_point_projection_hit(id) -> {has_hit, hit_entity?, projected_point, is_inside, distance}|nil
--   ecs.get_physics_shape_cast_hit(id) -> {has_hit, hit_entity?, toi, witness1, witness2, normal1, normal2, status}|nil
--   ecs.get_physics_velocity(id) -> {linear?, angular?}|nil
--   ecs.set_physics_velocity(id, {linear?, angular?, wake_up?}) -> bool
--   ecs.add_force(id, {x,y,z}, wake_up?) -> bool                 -- non-persistent (cleared each frame)
--   ecs.add_torque(id, {x,y,z}, wake_up?) -> bool                -- non-persistent (cleared each frame)
--   ecs.add_force_at_point(id, {x,y,z}, {x,y,z}, wake_up?) -> bool -- non-persistent (cleared each frame)
--   ecs.add_persistent_force(id, {x,y,z}, wake_up?) -> bool
--   ecs.set_persistent_force(id, {x,y,z}, wake_up?) -> bool
--   ecs.add_persistent_torque(id, {x,y,z}, wake_up?) -> bool
--   ecs.set_persistent_torque(id, {x,y,z}, wake_up?) -> bool
--   ecs.add_persistent_force_at_point(id, {x,y,z}, {x,y,z}, wake_up?) -> bool
--   ecs.clear_persistent_forces(id) -> bool
--   ecs.apply_impulse(id, {x,y,z}, wake_up?) -> bool
--   ecs.apply_angular_impulse(id, {x,y,z}, wake_up?) -> bool
--   ecs.apply_torque_impulse(id, {x,y,z}, wake_up?) -> bool
--   ecs.apply_impulse_at_point(id, {x,y,z}, {x,y,z}, wake_up?) -> bool
--   ecs.set_physics_running(running) -> bool
--   ecs.get_physics_running() -> bool
--   ecs.set_physics_gravity({x,y,z}) -> bool
--   ecs.get_physics_gravity() -> {x,y,z}|nil

local mover: number = -1
local t = 0.0
local origin = { x = 0.0, y = 0.0, z = 0.0 }

local speed = 5.0

function on_start()
    local owner_transform = ecs.get_transform(entity_id)
    if owner_transform ~= nil and owner_transform.position ~= nil then
        origin.x = owner_transform.position.x or 0.0
        origin.y = owner_transform.position.y or 0.0
        origin.z = owner_transform.position.z or 0.0
    end

    mover = ecs.spawn_entity("Orbiting Cube")
    ecs.set_mesh_renderer(mover, {
        source = "Cube",
        casts_shadow = true,
        visible = true,
    })
    ecs.set_transform(mover, { position = { x = origin.x, y = origin.y, z = origin.z } })
end

function on_update(dt: number)
    if mover == nil then return end

    local owner_transform = ecs.get_transform(entity_id)
    if owner_transform ~= nil and owner_transform.position ~= nil then
        origin.x = owner_transform.position.x or 0.0
        origin.y = owner_transform.position.y or 0.0
        origin.z = owner_transform.position.z or 0.0
    end

    t = t + (dt * speed)
    local radius = 2.0
    local x = origin.x + math.cos(t) * radius
    local y = origin.y + math.cos(t / 2) * radius
    local z = origin.z + math.sin(t) * radius
    ecs.set_transform(mover, { position = { x = x, y = y, z = z } })
end
"#
}

pub fn sanitize_rust_crate_name(name: &str) -> String {
    let mut value = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            value.push(ch.to_ascii_lowercase());
        } else {
            value.push('_');
        }
    }
    while value.starts_with('_') {
        value.remove(0);
    }
    while value.ends_with('_') {
        value.pop();
    }
    if value.is_empty() {
        "helmer_script".to_string()
    } else if value.chars().next().unwrap_or('a').is_ascii_digit() {
        format!("script_{}", value)
    } else {
        value
    }
}

pub fn rust_script_sdk_dependency_path(
    project_root: Option<&Path>,
    crate_dir: &Path,
) -> Result<PathBuf, String> {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../helmer_script_sdk");
    let source_root = source_root.canonicalize().unwrap_or(source_root);

    let sdk_root = match project_root {
        Some(root) => root.join(".helmer").join("helmer_script_sdk"),
        None => crate_dir.join(".helmer").join("helmer_script_sdk"),
    };

    sync_project_rust_sdk(&source_root, &sdk_root)?;

    let crate_dir = crate_dir
        .canonicalize()
        .unwrap_or_else(|_| crate_dir.to_path_buf());
    let sdk_root = sdk_root.canonicalize().unwrap_or(sdk_root);

    Ok(relative_path_from(&crate_dir, &sdk_root)
        .unwrap_or_else(|| PathBuf::from("./.helmer/helmer_script_sdk")))
}

fn sync_project_rust_sdk(source_root: &Path, target_root: &Path) -> Result<(), String> {
    let source_manifest = source_root.join("Cargo.toml");
    let source_lib = source_root.join("src").join("lib.rs");
    if !source_manifest.exists() || !source_lib.exists() {
        return Err(format!(
            "helmer_script_sdk source is missing at {}",
            source_root.to_string_lossy()
        ));
    }

    let target_src = target_root.join("src");
    fs::create_dir_all(&target_src).map_err(|err| err.to_string())?;
    copy_if_changed(&source_manifest, &target_root.join("Cargo.toml"))?;
    copy_if_changed(&source_lib, &target_src.join("lib.rs"))?;
    Ok(())
}

fn copy_if_changed(source: &Path, target: &Path) -> Result<(), String> {
    let source_bytes = fs::read(source).map_err(|err| err.to_string())?;
    let should_write = match fs::read(target) {
        Ok(existing) => existing != source_bytes,
        Err(_) => true,
    };
    if should_write {
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        fs::write(target, source_bytes).map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn relative_path_from(base: &Path, target: &Path) -> Option<PathBuf> {
    let base_components = base.components().collect::<Vec<_>>();
    let target_components = target.components().collect::<Vec<_>>();

    if let (Some(Component::Prefix(base_prefix)), Some(Component::Prefix(target_prefix))) =
        (base_components.first(), target_components.first())
    {
        if base_prefix != target_prefix {
            return None;
        }
    }

    let mut shared = 0usize;
    while shared < base_components.len()
        && shared < target_components.len()
        && base_components[shared] == target_components[shared]
    {
        shared += 1;
    }

    let mut relative = PathBuf::new();
    for component in &base_components[shared..] {
        if matches!(
            component,
            Component::Normal(_) | Component::CurDir | Component::ParentDir
        ) {
            relative.push("..");
        }
    }

    for component in &target_components[shared..] {
        match component {
            Component::CurDir => {}
            _ => relative.push(component.as_os_str()),
        }
    }

    if relative.as_os_str().is_empty() {
        relative.push(".");
    }

    Some(relative)
}

pub fn rust_script_manifest_template(crate_name: &str, sdk_path: &Path) -> String {
    let crate_name = sanitize_rust_crate_name(crate_name);
    let sdk_path = sdk_path.to_string_lossy().replace('\\', "/");
    format!(
        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lib]\ncrate-type = [\"cdylib\"]\n\n[dependencies]\nhelmer_script_sdk = {{ path = \"{}\" }}\n",
        crate_name, sdk_path
    )
}

pub fn default_rust_script_template_full() -> &'static str {
    r#"use helmer_script_sdk::{
    Host, MeshRendererPatch, Script, TransformPatch, Vec3, decode_state, encode_state,
    export_script,
};

#[derive(Default)]
struct OrbitScript {
    mover: Option<u64>,
    t: f32,
    speed: f32,
    origin: Vec3,
}

impl Script for OrbitScript {
    fn on_start(&mut self, host: &Host) {
        if self.speed == 0.0 {
            self.speed = 10.0;
        }

        if let Some(owner) = host.get_transform(host.entity_id()) {
            self.origin = owner.position;
        }

        let mover = host.spawn_entity(Some("Orbiting Cube"));
        self.mover = Some(mover);
        let _ = host.set_mesh_renderer_data(mover, &MeshRendererPatch::cube());
        let _ = host.set_transform(
            mover,
            &TransformPatch::with_position(Vec3 {
                x: self.origin.x,
                y: self.origin.y,
                z: self.origin.z,
            }),
        );
    }

    fn on_update(&mut self, host: &Host, dt: f32) {
        let Some(mover) = self.mover else {
            return;
        };

        if let Some(owner) = host.get_transform(host.entity_id()) {
            self.origin = owner.position;
        }

        self.t += dt * self.speed;
        let radius = 2.0_f32;
        let x = self.origin.x + self.t.cos() * radius;
        let y = self.origin.y + (self.t * 0.5).cos() * radius;
        let z = self.origin.z + self.t.sin() * radius;

        let _ = host.set_transform(
            mover,
            &TransformPatch::with_position(Vec3 { x, y, z }),
        );
    }

    fn save_state(&self) -> Option<Vec<u8>> {
        encode_state(&(self.mover, self.t, self.speed, self.origin))
    }

    fn load_state(&mut self, state: &[u8]) -> bool {
        let Some((mover, t, speed, origin)) =
            decode_state::<(Option<u64>, f32, f32, Vec3)>(state)
        else {
            return false;
        };

        self.mover = mover;
        self.t = t;
        self.speed = speed;
        self.origin = origin;
        true
    }
}

export_script!(OrbitScript);
"#
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RecentProjectsFile {
    paths: Vec<String>,
}

pub fn load_recent_projects() -> Vec<PathBuf> {
    let Some(path) = recent_projects_path() else {
        return Vec::new();
    };

    let data = match fs::read_to_string(&path) {
        Ok(data) => data,
        Err(_) => return Vec::new(),
    };

    let parsed = ron::de::from_str::<RecentProjectsFile>(&data).unwrap_or_default();
    parsed
        .paths
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .collect()
}

pub fn save_recent_projects(paths: &[PathBuf]) -> Result<(), String> {
    let Some(path) = recent_projects_path() else {
        return Ok(());
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }

    let payload = RecentProjectsFile {
        paths: paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
    };

    let pretty = PrettyConfig::new()
        .compact_arrays(false)
        .depth_limit(4)
        .enumerate_arrays(true);
    let data = ron::ser::to_string_pretty(&payload, pretty).map_err(|err| err.to_string())?;
    fs::write(path, data).map_err(|err| err.to_string())
}

fn recent_projects_path() -> Option<PathBuf> {
    let home = env::var("HOME").or_else(|_| env::var("USERPROFILE")).ok()?;
    Some(
        PathBuf::from(home)
            .join(".helmer_editor")
            .join("recent_projects.ron"),
    )
}
