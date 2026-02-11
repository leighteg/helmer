use std::collections::HashMap;

use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Component, Entity, Resource, World};
use egui::{TextureHandle, TextureId};
use glam::DVec2;

use helmer::provided::components::{ActiveCamera, AudioListener};
use helmer::runtime::runtime::{RuntimeCursorGrabMode, RuntimeCursorStateSnapshot};
use helmer_becs::{BevyActiveCamera, BevyAudioListener, BevyCamera, BevyTransform, BevyWrapper};

#[derive(Component, Debug, Clone, Copy)]
pub struct EditorViewportCamera {
    pub pane_id: u64,
}

impl Default for EditorViewportCamera {
    fn default() -> Self {
        Self { pane_id: 0 }
    }
}

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EditorPlayCamera;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayViewportKind {
    Editor,
    Gameplay,
}

impl PlayViewportKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Editor => "editor",
            Self::Gameplay => "gameplay",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "editor" => Some(Self::Editor),
            "gameplay" => Some(Self::Gameplay),
            _ => None,
        }
    }
}

impl Default for PlayViewportKind {
    fn default() -> Self {
        Self::Gameplay
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportResolutionPreset {
    Canvas,
    R640x360,
    R854x480,
    R1280x720,
    R1600x900,
    R1920x1080,
    R2560x1440,
    R3840x2160,
}

impl ViewportResolutionPreset {
    pub const ALL: [Self; 8] = [
        Self::Canvas,
        Self::R640x360,
        Self::R854x480,
        Self::R1280x720,
        Self::R1600x900,
        Self::R1920x1080,
        Self::R2560x1440,
        Self::R3840x2160,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Canvas => "Canvas (Auto)",
            Self::R640x360 => "640x360",
            Self::R854x480 => "854x480",
            Self::R1280x720 => "1280x720",
            Self::R1600x900 => "1600x900",
            Self::R1920x1080 => "1920x1080",
            Self::R2560x1440 => "2560x1440",
            Self::R3840x2160 => "3840x2160",
        }
    }

    pub fn target_size(self, canvas_size: [u32; 2]) -> [u32; 2] {
        match self {
            Self::Canvas => [canvas_size[0].max(1), canvas_size[1].max(1)],
            Self::R640x360 => [640, 360],
            Self::R854x480 => [854, 480],
            Self::R1280x720 => [1280, 720],
            Self::R1600x900 => [1600, 900],
            Self::R1920x1080 => [1920, 1080],
            Self::R2560x1440 => [2560, 1440],
            Self::R3840x2160 => [3840, 2160],
        }
    }
}

impl Default for ViewportResolutionPreset {
    fn default() -> Self {
        Self::Canvas
    }
}

pub const VIEWPORT_ID_EDITOR: u64 = 1;
pub const VIEWPORT_ID_GAMEPLAY: u64 = 2;
pub const VIEWPORT_ID_PREVIEW: u64 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ViewportRectPixels {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl ViewportRectPixels {
    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Option<Self> {
        if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
            return None;
        }
        if max_x <= min_x || max_y <= min_y {
            return None;
        }
        Some(Self {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    pub fn width(self) -> f32 {
        (self.max_x - self.min_x).max(1.0)
    }

    pub fn height(self) -> f32 {
        (self.max_y - self.min_y).max(1.0)
    }

    pub fn aspect_ratio(self) -> f32 {
        self.width() / self.height()
    }

    pub fn target_size(self) -> [u32; 2] {
        [
            self.width().round().max(1.0) as u32,
            self.height().round().max(1.0) as u32,
        ]
    }

    pub fn contains(self, cursor: DVec2) -> bool {
        let x = cursor.x as f32;
        let y = cursor.y as f32;
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct EditorCursorControlState {
    pub freecam_capture_active: bool,
    pub script_policy: Option<RuntimeCursorStateSnapshot>,
}

impl EditorCursorControlState {
    pub fn effective_policy(&self) -> RuntimeCursorStateSnapshot {
        if self.freecam_capture_active {
            RuntimeCursorStateSnapshot {
                visible: false,
                grab_mode: RuntimeCursorGrabMode::Locked,
            }
        } else {
            self.script_policy.unwrap_or_default()
        }
    }
}

impl Default for EditorCursorControlState {
    fn default() -> Self {
        Self {
            freecam_capture_active: false,
            script_policy: None,
        }
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct EditorViewportRuntime {
    pub editor_texture_id: Option<TextureId>,
    pub gameplay_texture_id: Option<TextureId>,
    pub preview_texture_id: Option<TextureId>,
    pub main_rect_pixels: Option<ViewportRectPixels>,
    pub main_target_size: Option<[u32; 2]>,
    pub main_resize_immediate: bool,
    pub preview_rect_pixels: Option<ViewportRectPixels>,
    pub preview_camera_entity: Option<Entity>,
    pub pointer_over_main: bool,
    pub keyboard_focus: bool,
    pub pane_requests: Vec<EditorViewportPaneRequest>,
    pub active_pane_id: Option<u64>,
    pub active_camera_entity: Option<Entity>,
}

impl EditorViewportRuntime {
    pub fn begin_frame(&mut self) {
        self.main_rect_pixels = None;
        self.main_target_size = None;
        self.main_resize_immediate = false;
        self.preview_rect_pixels = None;
        self.preview_camera_entity = None;
        self.pointer_over_main = false;
        self.pane_requests.clear();
        self.active_camera_entity = None;
    }
}

#[derive(Debug, Clone)]
pub struct EditorViewportPaneRequest {
    pub pane_id: u64,
    pub camera_entity: Entity,
    pub texture_id: TextureId,
    pub viewport_rect: ViewportRectPixels,
    pub pointer_over: bool,
    pub target_size: [u32; 2],
    pub temporal_history: bool,
    pub immediate_resize: bool,
    pub graph_template: String,
    pub gizmos_in_play: bool,
    pub show_camera_gizmos: bool,
    pub show_directional_light_gizmos: bool,
    pub show_point_light_gizmos: bool,
    pub show_spot_light_gizmos: bool,
    pub show_spline_paths: bool,
    pub show_spline_points: bool,
}

#[derive(Resource, Default)]
pub struct EditorViewportTextures {
    pub editor: Option<TextureHandle>,
    pub gameplay: Option<TextureHandle>,
    pub preview: Option<TextureHandle>,
    pub pane_textures: HashMap<u64, TextureHandle>,
}

#[derive(Resource, Debug, Clone)]
pub struct EditorViewportState {
    pub graph_template: String,
    pub play_mode_view: PlayViewportKind,
    pub render_resolution: ViewportResolutionPreset,
    pub pinned_camera: Option<Entity>,
    pub preview_position_norm: [f32; 2],
    pub preview_width_norm: f32,
    pub show_options_panel: bool,
    pub gizmos_in_play: bool,
    pub execute_scripts_in_edit_mode: bool,
    pub show_camera_gizmos: bool,
    pub show_directional_light_gizmos: bool,
    pub show_point_light_gizmos: bool,
    pub show_spot_light_gizmos: bool,
    pub show_spline_paths: bool,
    pub show_spline_points: bool,
}

impl Default for EditorViewportState {
    fn default() -> Self {
        Self {
            graph_template: "debug-graph".to_string(),
            play_mode_view: PlayViewportKind::Gameplay,
            render_resolution: ViewportResolutionPreset::Canvas,
            pinned_camera: None,
            preview_position_norm: [0.03, 0.74],
            preview_width_norm: 0.28,
            show_options_panel: false,
            gizmos_in_play: false,
            execute_scripts_in_edit_mode: false,
            show_camera_gizmos: true,
            show_directional_light_gizmos: true,
            show_point_light_gizmos: true,
            show_spot_light_gizmos: true,
            show_spline_paths: true,
            show_spline_points: true,
        }
    }
}

pub fn ensure_viewport_camera(world: &mut World) -> Entity {
    ensure_viewport_camera_for_pane(world, 0)
}

pub fn ensure_viewport_camera_for_pane(world: &mut World, pane_id: u64) -> Entity {
    if let Some((entity, _)) = world
        .query::<(Entity, &EditorViewportCamera)>()
        .iter(world)
        .find(|(_, camera)| camera.pane_id == pane_id)
    {
        return entity;
    }

    if let Some((entity, _)) = world
        .query::<(Entity, &EditorViewportCamera)>()
        .iter(world)
        .next()
    {
        if pane_id == 0 {
            return entity;
        }
    }

    let name = if pane_id == 0 {
        "Viewport Camera".to_string()
    } else {
        format!("Viewport Camera {}", pane_id)
    };

    world
        .spawn((
            EditorViewportCamera { pane_id },
            BevyTransform::default(),
            BevyCamera::default(),
            Name::new(name),
        ))
        .id()
}

pub fn activate_viewport_camera(world: &mut World) -> Entity {
    activate_viewport_camera_for_pane(world, 0)
}

pub fn activate_viewport_camera_for_pane(world: &mut World, pane_id: u64) -> Entity {
    let entity = ensure_viewport_camera_for_pane(world, pane_id);
    clear_active_camera(world);
    world
        .entity_mut(entity)
        .insert(BevyWrapper(ActiveCamera {}));
    entity
}

pub fn set_viewport_audio_listener_enabled(world: &mut World, enabled: bool) {
    let entities: Vec<Entity> = world
        .query::<(Entity, &EditorViewportCamera)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .collect();

    if entities.is_empty() && enabled {
        let entity = ensure_viewport_camera(world);
        world
            .entity_mut(entity)
            .insert(BevyWrapper(AudioListener { enabled: true }));
        return;
    }

    for entity in entities {
        if enabled {
            world
                .entity_mut(entity)
                .insert(BevyWrapper(AudioListener { enabled: true }));
        } else if let Some(mut listener) = world.get_mut::<BevyAudioListener>(entity) {
            listener.0.enabled = false;
        }
    }
}

pub fn set_play_camera(world: &mut World, entity: Entity) {
    if world.get_entity(entity).is_err() {
        return;
    }
    if world.get::<BevyCamera>(entity).is_none() || world.get::<BevyTransform>(entity).is_none() {
        return;
    }
    if world.get::<EditorViewportCamera>(entity).is_some() {
        return;
    }

    let existing: Vec<Entity> = world
        .query::<(Entity, &EditorPlayCamera)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .collect();
    for entity in existing {
        world.entity_mut(entity).remove::<EditorPlayCamera>();
    }

    world.entity_mut(entity).insert(EditorPlayCamera);
}

pub fn ensure_play_camera(world: &mut World) -> Option<Entity> {
    let mut fallback = None;
    let mut selected = None;

    let mut query = world.query::<(Entity, &BevyCamera, Option<&EditorPlayCamera>)>();
    for (entity, _, play_camera) in query.iter(world) {
        if world.get::<EditorViewportCamera>(entity).is_some() {
            continue;
        }

        if play_camera.is_some() {
            selected = Some(entity);
            break;
        }

        if fallback.is_none() {
            fallback = Some(entity);
        }
    }

    let target = selected.or(fallback);
    if let Some(entity) = target {
        set_play_camera(world, entity);
    }
    target
}

pub fn activate_play_camera(world: &mut World) -> Option<Entity> {
    let target = ensure_play_camera(world);
    let Some(entity) = target else {
        return Some(activate_viewport_camera(world));
    };

    clear_active_camera(world);
    world
        .entity_mut(entity)
        .insert(BevyWrapper(ActiveCamera {}));
    Some(entity)
}

fn clear_active_camera(world: &mut World) {
    let active_entities: Vec<Entity> = world
        .query::<(Entity, &BevyActiveCamera)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .collect();
    for entity in active_entities {
        world.entity_mut(entity).remove::<BevyActiveCamera>();
    }
}
