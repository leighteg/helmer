use std::collections::HashMap;

use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Component, Entity, Resource, World};
use egui::{TextureHandle, TextureId};
use glam::{DVec2, Vec3};

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

// reserve high IDs for built-in non-pane viewports to avoid collisions with pane IDs
pub const VIEWPORT_ID_EDITOR: u64 = u64::MAX - 2;
pub const VIEWPORT_ID_GAMEPLAY: u64 = u64::MAX - 1;
pub const VIEWPORT_ID_PREVIEW: u64 = u64::MAX;

pub const FREECAM_SENSITIVITY_DEFAULT: f32 = 0.12;
pub const FREECAM_SENSITIVITY_MIN: f32 = 0.01;
pub const FREECAM_SENSITIVITY_MAX: f32 = 2.0;
pub const FREECAM_SMOOTHING_DEFAULT: f32 = 0.04;
pub const FREECAM_SMOOTHING_MIN: f32 = 0.0;
pub const FREECAM_SMOOTHING_MAX: f32 = 0.25;
pub const FREECAM_MOVE_ACCEL_DEFAULT: f32 = 18.0;
pub const FREECAM_MOVE_ACCEL_MIN: f32 = 0.5;
pub const FREECAM_MOVE_ACCEL_MAX: f32 = 160.0;
pub const FREECAM_MOVE_DECEL_DEFAULT: f32 = 12.0;
pub const FREECAM_MOVE_DECEL_MIN: f32 = 0.5;
pub const FREECAM_MOVE_DECEL_MAX: f32 = 200.0;
pub const FREECAM_SPEED_STEP_DEFAULT: f32 = 2.0;
pub const FREECAM_SPEED_STEP_MIN: f32 = 0.05;
pub const FREECAM_SPEED_STEP_MAX: f32 = 40.0;
pub const FREECAM_SPEED_MIN_DEFAULT: f32 = 0.5;
pub const FREECAM_SPEED_MIN_MIN: f32 = 0.05;
pub const FREECAM_SPEED_MIN_MAX: f32 = 20.0;
pub const FREECAM_SPEED_MAX_DEFAULT: f32 = 80.0;
pub const FREECAM_SPEED_MAX_MIN: f32 = 1.0;
pub const FREECAM_SPEED_MAX_MAX: f32 = 500.0;
pub const FREECAM_BOOST_MULTIPLIER_DEFAULT: f32 = 2.5;
pub const FREECAM_BOOST_MULTIPLIER_MIN: f32 = 1.0;
pub const FREECAM_BOOST_MULTIPLIER_MAX: f32 = 10.0;
pub const FREECAM_ORBIT_DISTANCE_DEFAULT: f32 = 5.0;
pub const FREECAM_ORBIT_DISTANCE_MIN: f32 = 0.25;
pub const FREECAM_ORBIT_DISTANCE_MAX: f32 = 5000.0;
pub const FREECAM_ORBIT_PAN_SENSITIVITY_DEFAULT: f32 = 0.0020;
pub const FREECAM_ORBIT_PAN_SENSITIVITY_MIN: f32 = 0.0001;
pub const FREECAM_ORBIT_PAN_SENSITIVITY_MAX: f32 = 0.05;
pub const FREECAM_PAN_SENSITIVITY_DEFAULT: f32 = 0.0008;
pub const FREECAM_PAN_SENSITIVITY_MIN: f32 = 0.00005;
pub const FREECAM_PAN_SENSITIVITY_MAX: f32 = 0.05;

pub const NAV_GIZMO_RADIUS_SCALE_DEFAULT: f32 = 1.0;
pub const NAV_GIZMO_RADIUS_SCALE_MIN: f32 = 0.5;
pub const NAV_GIZMO_RADIUS_SCALE_MAX: f32 = 2.5;
pub const NAV_GIZMO_PADDING_DEFAULT: f32 = 12.0;
pub const NAV_GIZMO_PADDING_MIN: f32 = 4.0;
pub const NAV_GIZMO_PADDING_MAX: f32 = 48.0;
pub const NAV_GIZMO_ORBIT_RADIUS_SCALE_DEFAULT: f32 = 1.0;
pub const NAV_GIZMO_ORBIT_RADIUS_SCALE_MIN: f32 = 0.5;
pub const NAV_GIZMO_ORBIT_RADIUS_SCALE_MAX: f32 = 1.5;
pub const NAV_GIZMO_HOME_RADIUS_SCALE_DEFAULT: f32 = 1.0;
pub const NAV_GIZMO_HOME_RADIUS_SCALE_MIN: f32 = 0.4;
pub const NAV_GIZMO_HOME_RADIUS_SCALE_MAX: f32 = 2.5;
pub const NAV_GIZMO_MARKER_RADIUS_FRONT_DEFAULT: f32 = 8.0;
pub const NAV_GIZMO_MARKER_RADIUS_FRONT_MIN: f32 = 3.0;
pub const NAV_GIZMO_MARKER_RADIUS_FRONT_MAX: f32 = 20.0;
pub const NAV_GIZMO_MARKER_RADIUS_BACK_DEFAULT: f32 = 6.5;
pub const NAV_GIZMO_MARKER_RADIUS_BACK_MIN: f32 = 2.0;
pub const NAV_GIZMO_MARKER_RADIUS_BACK_MAX: f32 = 20.0;
pub const NAV_GIZMO_LINE_THICKNESS_FRONT_DEFAULT: f32 = 2.0;
pub const NAV_GIZMO_LINE_THICKNESS_FRONT_MIN: f32 = 0.5;
pub const NAV_GIZMO_LINE_THICKNESS_FRONT_MAX: f32 = 6.0;
pub const NAV_GIZMO_LINE_THICKNESS_BACK_DEFAULT: f32 = 1.3;
pub const NAV_GIZMO_LINE_THICKNESS_BACK_MIN: f32 = 0.5;
pub const NAV_GIZMO_LINE_THICKNESS_BACK_MAX: f32 = 6.0;
pub const NAV_GIZMO_CENTER_DOT_RADIUS_DEFAULT: f32 = 2.4;
pub const NAV_GIZMO_CENTER_DOT_RADIUS_MIN: f32 = 0.5;
pub const NAV_GIZMO_CENTER_DOT_RADIUS_MAX: f32 = 8.0;
pub const NAV_GIZMO_DRAG_SENSITIVITY_DEFAULT: f32 = 0.0085;
pub const NAV_GIZMO_DRAG_SENSITIVITY_MIN: f32 = 0.001;
pub const NAV_GIZMO_DRAG_SENSITIVITY_MAX: f32 = 0.05;
pub const NAV_GIZMO_BACKGROUND_ALPHA_DEFAULT: f32 = 0.71;
pub const NAV_GIZMO_BACKGROUND_ALPHA_MIN: f32 = 0.0;
pub const NAV_GIZMO_BACKGROUND_ALPHA_MAX: f32 = 1.0;
pub const NAV_GIZMO_OUTLINE_ALPHA_DEFAULT: f32 = 0.35;
pub const NAV_GIZMO_OUTLINE_ALPHA_MIN: f32 = 0.0;
pub const NAV_GIZMO_OUTLINE_ALPHA_MAX: f32 = 1.0;
pub const NAV_GIZMO_NEGATIVE_AXIS_BRIGHTNESS_DEFAULT: f32 = 0.58;
pub const NAV_GIZMO_NEGATIVE_AXIS_BRIGHTNESS_MIN: f32 = 0.2;
pub const NAV_GIZMO_NEGATIVE_AXIS_BRIGHTNESS_MAX: f32 = 1.0;
pub const NAV_GIZMO_HOVER_BRIGHTNESS_DEFAULT: f32 = 0.1;
pub const NAV_GIZMO_HOVER_BRIGHTNESS_MIN: f32 = 0.0;
pub const NAV_GIZMO_HOVER_BRIGHTNESS_MAX: f32 = 0.5;
pub const NAV_GIZMO_TEXT_SCALE_DEFAULT: f32 = 1.0;
pub const NAV_GIZMO_TEXT_SCALE_MIN: f32 = 0.5;
pub const NAV_GIZMO_TEXT_SCALE_MAX: f32 = 2.0;

#[derive(Debug, Clone, PartialEq)]
pub struct NavigationGizmoSettings {
    pub show_background: bool,
    pub show_labels: bool,
    pub orbit_selected_entity: bool,
    pub radius_scale: f32,
    pub padding: f32,
    pub orbit_radius_scale: f32,
    pub home_radius_scale: f32,
    pub marker_radius_front: f32,
    pub marker_radius_back: f32,
    pub line_thickness_front: f32,
    pub line_thickness_back: f32,
    pub center_dot_radius: f32,
    pub drag_sensitivity: f32,
    pub background_alpha: f32,
    pub outline_alpha: f32,
    pub axis_color_x: [f32; 3],
    pub axis_color_y: [f32; 3],
    pub axis_color_z: [f32; 3],
    pub negative_axis_brightness: f32,
    pub hover_brightness: f32,
    pub text_scale: f32,
}

impl Default for NavigationGizmoSettings {
    fn default() -> Self {
        Self {
            show_background: true,
            show_labels: true,
            orbit_selected_entity: true,
            radius_scale: NAV_GIZMO_RADIUS_SCALE_DEFAULT,
            padding: NAV_GIZMO_PADDING_DEFAULT,
            orbit_radius_scale: NAV_GIZMO_ORBIT_RADIUS_SCALE_DEFAULT,
            home_radius_scale: NAV_GIZMO_HOME_RADIUS_SCALE_DEFAULT,
            marker_radius_front: NAV_GIZMO_MARKER_RADIUS_FRONT_DEFAULT,
            marker_radius_back: NAV_GIZMO_MARKER_RADIUS_BACK_DEFAULT,
            line_thickness_front: NAV_GIZMO_LINE_THICKNESS_FRONT_DEFAULT,
            line_thickness_back: NAV_GIZMO_LINE_THICKNESS_BACK_DEFAULT,
            center_dot_radius: NAV_GIZMO_CENTER_DOT_RADIUS_DEFAULT,
            drag_sensitivity: NAV_GIZMO_DRAG_SENSITIVITY_DEFAULT,
            background_alpha: NAV_GIZMO_BACKGROUND_ALPHA_DEFAULT,
            outline_alpha: NAV_GIZMO_OUTLINE_ALPHA_DEFAULT,
            axis_color_x: [0.86, 0.27, 0.27],
            axis_color_y: [0.27, 0.8, 0.37],
            axis_color_z: [0.37, 0.57, 0.92],
            negative_axis_brightness: NAV_GIZMO_NEGATIVE_AXIS_BRIGHTNESS_DEFAULT,
            hover_brightness: NAV_GIZMO_HOVER_BRIGHTNESS_DEFAULT,
            text_scale: NAV_GIZMO_TEXT_SCALE_DEFAULT,
        }
    }
}

impl NavigationGizmoSettings {
    pub fn sanitize(&mut self) {
        fn sanitize_f32(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
            if value.is_finite() {
                value.clamp(min, max)
            } else {
                fallback
            }
        }
        fn sanitize_rgb(rgb: &mut [f32; 3], fallback: [f32; 3]) {
            for (channel, fallback_channel) in rgb.iter_mut().zip(fallback.iter()) {
                let value = if channel.is_finite() {
                    *channel
                } else {
                    *fallback_channel
                };
                *channel = value.clamp(0.0, 1.0);
            }
        }

        self.radius_scale = sanitize_f32(
            self.radius_scale,
            NAV_GIZMO_RADIUS_SCALE_DEFAULT,
            NAV_GIZMO_RADIUS_SCALE_MIN,
            NAV_GIZMO_RADIUS_SCALE_MAX,
        );
        self.padding = sanitize_f32(
            self.padding,
            NAV_GIZMO_PADDING_DEFAULT,
            NAV_GIZMO_PADDING_MIN,
            NAV_GIZMO_PADDING_MAX,
        );
        self.orbit_radius_scale = sanitize_f32(
            self.orbit_radius_scale,
            NAV_GIZMO_ORBIT_RADIUS_SCALE_DEFAULT,
            NAV_GIZMO_ORBIT_RADIUS_SCALE_MIN,
            NAV_GIZMO_ORBIT_RADIUS_SCALE_MAX,
        );
        self.home_radius_scale = sanitize_f32(
            self.home_radius_scale,
            NAV_GIZMO_HOME_RADIUS_SCALE_DEFAULT,
            NAV_GIZMO_HOME_RADIUS_SCALE_MIN,
            NAV_GIZMO_HOME_RADIUS_SCALE_MAX,
        );
        self.marker_radius_front = sanitize_f32(
            self.marker_radius_front,
            NAV_GIZMO_MARKER_RADIUS_FRONT_DEFAULT,
            NAV_GIZMO_MARKER_RADIUS_FRONT_MIN,
            NAV_GIZMO_MARKER_RADIUS_FRONT_MAX,
        );
        self.marker_radius_back = sanitize_f32(
            self.marker_radius_back,
            NAV_GIZMO_MARKER_RADIUS_BACK_DEFAULT,
            NAV_GIZMO_MARKER_RADIUS_BACK_MIN,
            NAV_GIZMO_MARKER_RADIUS_BACK_MAX,
        );
        self.line_thickness_front = sanitize_f32(
            self.line_thickness_front,
            NAV_GIZMO_LINE_THICKNESS_FRONT_DEFAULT,
            NAV_GIZMO_LINE_THICKNESS_FRONT_MIN,
            NAV_GIZMO_LINE_THICKNESS_FRONT_MAX,
        );
        self.line_thickness_back = sanitize_f32(
            self.line_thickness_back,
            NAV_GIZMO_LINE_THICKNESS_BACK_DEFAULT,
            NAV_GIZMO_LINE_THICKNESS_BACK_MIN,
            NAV_GIZMO_LINE_THICKNESS_BACK_MAX,
        );
        self.center_dot_radius = sanitize_f32(
            self.center_dot_radius,
            NAV_GIZMO_CENTER_DOT_RADIUS_DEFAULT,
            NAV_GIZMO_CENTER_DOT_RADIUS_MIN,
            NAV_GIZMO_CENTER_DOT_RADIUS_MAX,
        );
        self.drag_sensitivity = sanitize_f32(
            self.drag_sensitivity,
            NAV_GIZMO_DRAG_SENSITIVITY_DEFAULT,
            NAV_GIZMO_DRAG_SENSITIVITY_MIN,
            NAV_GIZMO_DRAG_SENSITIVITY_MAX,
        );
        self.background_alpha = sanitize_f32(
            self.background_alpha,
            NAV_GIZMO_BACKGROUND_ALPHA_DEFAULT,
            NAV_GIZMO_BACKGROUND_ALPHA_MIN,
            NAV_GIZMO_BACKGROUND_ALPHA_MAX,
        );
        self.outline_alpha = sanitize_f32(
            self.outline_alpha,
            NAV_GIZMO_OUTLINE_ALPHA_DEFAULT,
            NAV_GIZMO_OUTLINE_ALPHA_MIN,
            NAV_GIZMO_OUTLINE_ALPHA_MAX,
        );
        self.negative_axis_brightness = sanitize_f32(
            self.negative_axis_brightness,
            NAV_GIZMO_NEGATIVE_AXIS_BRIGHTNESS_DEFAULT,
            NAV_GIZMO_NEGATIVE_AXIS_BRIGHTNESS_MIN,
            NAV_GIZMO_NEGATIVE_AXIS_BRIGHTNESS_MAX,
        );
        self.hover_brightness = sanitize_f32(
            self.hover_brightness,
            NAV_GIZMO_HOVER_BRIGHTNESS_DEFAULT,
            NAV_GIZMO_HOVER_BRIGHTNESS_MIN,
            NAV_GIZMO_HOVER_BRIGHTNESS_MAX,
        );
        self.text_scale = sanitize_f32(
            self.text_scale,
            NAV_GIZMO_TEXT_SCALE_DEFAULT,
            NAV_GIZMO_TEXT_SCALE_MIN,
            NAV_GIZMO_TEXT_SCALE_MAX,
        );
        sanitize_rgb(&mut self.axis_color_x, [0.86, 0.27, 0.27]);
        sanitize_rgb(&mut self.axis_color_y, [0.27, 0.8, 0.37]);
        sanitize_rgb(&mut self.axis_color_z, [0.37, 0.57, 0.92]);
    }
}

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
    pub script_capture_allowed: bool,
    pub script_capture_suspended: bool,
    pub script_policy: Option<RuntimeCursorStateSnapshot>,
}

impl EditorCursorControlState {
    pub fn effective_policy(&self) -> RuntimeCursorStateSnapshot {
        if self.freecam_capture_active {
            RuntimeCursorStateSnapshot {
                visible: false,
                grab_mode: RuntimeCursorGrabMode::Locked,
            }
        } else if self.script_capture_allowed && !self.script_capture_suspended {
            self.script_policy.unwrap_or_default()
        } else {
            RuntimeCursorStateSnapshot::default()
        }
    }
}

impl Default for EditorCursorControlState {
    fn default() -> Self {
        Self {
            freecam_capture_active: false,
            script_capture_allowed: false,
            script_capture_suspended: false,
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
    pub orbit_selected_focus: Option<Vec3>,
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
        self.orbit_selected_focus = None;
    }
}

#[derive(Debug, Clone)]
pub struct EditorViewportPaneRequest {
    pub pane_id: u64,
    pub camera_entity: Entity,
    pub is_play_viewport: bool,
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
    pub show_navigation_gizmo: bool,
    pub navigation_gizmo: NavigationGizmoSettings,
    pub freecam_sensitivity: f32,
    pub freecam_smoothing: f32,
    pub freecam_move_accel: f32,
    pub freecam_move_decel: f32,
    pub freecam_speed_step: f32,
    pub freecam_speed_min: f32,
    pub freecam_speed_max: f32,
    pub freecam_boost_multiplier: f32,
    pub freecam_orbit_distance: f32,
    pub freecam_orbit_pan_sensitivity: f32,
    pub freecam_pan_sensitivity: f32,
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
            show_navigation_gizmo: true,
            navigation_gizmo: NavigationGizmoSettings::default(),
            freecam_sensitivity: FREECAM_SENSITIVITY_DEFAULT,
            freecam_smoothing: FREECAM_SMOOTHING_DEFAULT,
            freecam_move_accel: FREECAM_MOVE_ACCEL_DEFAULT,
            freecam_move_decel: FREECAM_MOVE_DECEL_DEFAULT,
            freecam_speed_step: FREECAM_SPEED_STEP_DEFAULT,
            freecam_speed_min: FREECAM_SPEED_MIN_DEFAULT,
            freecam_speed_max: FREECAM_SPEED_MAX_DEFAULT,
            freecam_boost_multiplier: FREECAM_BOOST_MULTIPLIER_DEFAULT,
            freecam_orbit_distance: FREECAM_ORBIT_DISTANCE_DEFAULT,
            freecam_orbit_pan_sensitivity: FREECAM_ORBIT_PAN_SENSITIVITY_DEFAULT,
            freecam_pan_sensitivity: FREECAM_PAN_SENSITIVITY_DEFAULT,
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

    let inherited_listener_enabled = world
        .query::<(&EditorViewportCamera, &BevyAudioListener)>()
        .iter(world)
        .next()
        .map(|(_, listener)| listener.0.enabled);

    let entity = world
        .spawn((
            EditorViewportCamera { pane_id },
            BevyTransform::default(),
            BevyCamera::default(),
            Name::new(name),
        ))
        .id();

    if let Some(enabled) = inherited_listener_enabled {
        world
            .entity_mut(entity)
            .insert(BevyWrapper(AudioListener { enabled }));
    }

    entity
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
    let active_viewport_entity = world
        .query::<(Entity, &EditorViewportCamera, Option<&BevyActiveCamera>)>()
        .iter(world)
        .find_map(|(entity, _, active)| active.map(|_| entity));
    let listener_entity = active_viewport_entity.or_else(|| entities.first().copied());

    if entities.is_empty() && enabled {
        let entity = ensure_viewport_camera(world);
        world
            .entity_mut(entity)
            .insert(BevyWrapper(AudioListener { enabled: true }));
        return;
    }

    for entity in entities {
        let should_enable = enabled && Some(entity) == listener_entity;
        if let Some(mut listener) = world.get_mut::<BevyAudioListener>(entity) {
            listener.0.enabled = should_enable;
        } else if should_enable {
            world
                .entity_mut(entity)
                .insert(BevyWrapper(AudioListener { enabled: true }));
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
