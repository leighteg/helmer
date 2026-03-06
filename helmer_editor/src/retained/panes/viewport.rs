use std::collections::HashMap;

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiImage, UiLabel, UiLayoutBuilder,
    UiRect, UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewportPaneMode {
    Edit,
    Play,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewportPaneSurfaceKey {
    pub tab_id: u64,
    pub mode: ViewportPaneMode,
}

const RETAINED_VIEWPORT_ID_EDIT_BASE: u64 = 0xED17_0000_0000_0000;
const RETAINED_VIEWPORT_ID_PLAY_BASE: u64 = 0xA11E_0000_0000_0000;
const RETAINED_VIEWPORT_ID_PREVIEW_BASE: u64 = 0x9D00_0000_0000_0000;
const RETAINED_VIEWPORT_ID_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

pub fn retained_viewport_id(mode: ViewportPaneMode, tab_id: u64) -> u64 {
    let base = match mode {
        ViewportPaneMode::Edit => RETAINED_VIEWPORT_ID_EDIT_BASE,
        ViewportPaneMode::Play => RETAINED_VIEWPORT_ID_PLAY_BASE,
    };
    base ^ tab_id.wrapping_mul(RETAINED_VIEWPORT_ID_MIX)
}

pub fn retained_viewport_texture_slot(mode: ViewportPaneMode, tab_id: u64) -> Option<usize> {
    usize::try_from(retained_viewport_id(mode, tab_id)).ok()
}

pub fn retained_viewport_preview_id(tab_id: u64) -> u64 {
    RETAINED_VIEWPORT_ID_PREVIEW_BASE ^ tab_id.wrapping_mul(RETAINED_VIEWPORT_ID_MIX)
}

pub fn retained_viewport_preview_texture_slot(tab_id: u64) -> Option<usize> {
    usize::try_from(retained_viewport_preview_id(tab_id)).ok()
}

#[derive(Clone, Debug, Default)]
pub struct ViewportPaneData {
    pub tab_id: u64,
    pub title: String,
    pub world_mode: String,
    pub play_mode_active: bool,
    pub scene_name: String,
    pub active_camera: String,
    pub entity_count: usize,
    pub resolution_label: String,
    pub texture_id: Option<usize>,
    pub preview_texture_id: Option<usize>,
    pub preview_camera_name: Option<String>,
    pub preview_position_norm: [f32; 2],
    pub preview_width_norm: f32,
    pub preview_aspect_ratio: f32,
}

#[derive(Clone, Debug)]
pub enum ViewportPaneAction {
    TogglePlayMode { tab_id: u64, mode: ViewportPaneMode },
    OpenCanvasMenu { tab_id: u64, mode: ViewportPaneMode },
    OpenRenderMenu { tab_id: u64, mode: ViewportPaneMode },
    OpenScriptingMenu { tab_id: u64, mode: ViewportPaneMode },
    OpenGizmosMenu { tab_id: u64, mode: ViewportPaneMode },
    OpenFreecamMenu { tab_id: u64, mode: ViewportPaneMode },
    OpenOrbitMenu { tab_id: u64, mode: ViewportPaneMode },
    OpenAdvancedMenu { tab_id: u64, mode: ViewportPaneMode },
    PreviewMove { tab_id: u64, mode: ViewportPaneMode },
    PreviewResize { tab_id: u64, mode: ViewportPaneMode },
}

#[derive(Clone, Debug, Default)]
pub struct ViewportPaneFrame {
    pub actions: HashMap<UiId, ViewportPaneAction>,
    pub scene_rect: UiRect,
    pub preview_rect: Option<UiRect>,
}

pub fn build_viewport_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    viewport: UiRect,
    mode: ViewportPaneMode,
    data: &ViewportPaneData,
) -> ViewportPaneFrame {
    let mut frame = ViewportPaneFrame::default();
    let pane_width = viewport.width.max(1.0);
    let pane_height = viewport.height.max(1.0);
    let padding = 8.0;
    let controls_height = 20.0;
    let controls_gap = 4.0;
    let controls_row_gap = 3.0;
    let controls_row_width = (pane_width - padding * 2.0 - 8.0).max(56.0);
    let toolbar_inner_padding = 4.0;

    let canvas_label = if data.resolution_label.trim().is_empty() {
        "Canvas (Auto)".to_string()
    } else {
        data.resolution_label.clone()
    };
    let controls: Vec<(String, ViewportPaneAction)> = vec![
        (
            canvas_label,
            ViewportPaneAction::OpenCanvasMenu {
                tab_id: data.tab_id,
                mode,
            },
        ),
        (
            "Render".to_string(),
            ViewportPaneAction::OpenRenderMenu {
                tab_id: data.tab_id,
                mode,
            },
        ),
        (
            "Scripting".to_string(),
            ViewportPaneAction::OpenScriptingMenu {
                tab_id: data.tab_id,
                mode,
            },
        ),
        (
            "Gizmos".to_string(),
            ViewportPaneAction::OpenGizmosMenu {
                tab_id: data.tab_id,
                mode,
            },
        ),
        (
            "Freecam".to_string(),
            ViewportPaneAction::OpenFreecamMenu {
                tab_id: data.tab_id,
                mode,
            },
        ),
        (
            "Orbit".to_string(),
            ViewportPaneAction::OpenOrbitMenu {
                tab_id: data.tab_id,
                mode,
            },
        ),
        (
            "Advanced".to_string(),
            ViewportPaneAction::OpenAdvancedMenu {
                tab_id: data.tab_id,
                mode,
            },
        ),
    ];
    let control_layout = controls
        .iter()
        .enumerate()
        .scan((0.0f32, 0.0f32), |(x, y), (index, (label, _))| {
            let width = (label.chars().count() as f32 * 7.0 + 22.0).clamp(52.0, 148.0);
            if *x > 0.0 && *x + width > controls_row_width {
                *x = 0.0;
                *y += controls_height + controls_row_gap;
            }
            let out = (index, *x, *y, width);
            *x += width + controls_gap;
            Some(out)
        })
        .collect::<Vec<_>>();
    let controls_rows = control_layout.last().map_or(1, |(_, _, y, _)| {
        ((*y / (controls_height + controls_row_gap)).floor() as usize) + 1
    });
    let controls_block_height = (controls_rows as f32 * controls_height)
        + ((controls_rows.saturating_sub(1)) as f32 * controls_row_gap);
    let toolbar_height = controls_block_height + toolbar_inner_padding * 2.0;
    let scene_y = padding + toolbar_height + 4.0;
    let scene_rect = UiRect {
        x: padding,
        y: scene_y,
        width: (pane_width - padding * 2.0).max(1.0),
        height: (pane_height - scene_y - padding).max(1.0),
    };
    frame.scene_rect = scene_rect;

    let background_id = root_id.child("background");
    let toolbar_id = root_id.child("toolbar");
    let controls_row_id = toolbar_id.child("controls-row");
    let scene_frame_id = root_id.child("scene-frame");
    let scene_image_id = scene_frame_id.child("image");
    let scene_placeholder_id = scene_frame_id.child("placeholder");
    let scene_play_inactive_id = scene_frame_id.child("play-inactive");
    let preview_frame_id = scene_frame_id.child("preview-frame");
    let preview_image_id = preview_frame_id.child("image");
    let preview_label_id = preview_frame_id.child("label");
    let preview_move_hit_id = preview_frame_id.child("move-hit");
    let preview_resize_hit_id = preview_frame_id.child("resize-hit");
    let preview_resize_handle_id = preview_frame_id.child("resize-handle");

    retained.upsert(RetainedUiNode::new(
        root_id,
        UiWidget::Container,
        fill_style(UiVisualStyle {
            clip: true,
            ..UiVisualStyle::default()
        }),
    ));
    retained.upsert(RetainedUiNode::new(
        background_id,
        UiWidget::Container,
        fill_style(UiVisualStyle {
            background: Some(UiColor::rgba(0.10, 0.12, 0.15, 0.90)),
            border_color: Some(UiColor::rgba(0.24, 0.30, 0.37, 0.94)),
            border_width: 1.0,
            corner_radius: 0.0,
            clip: true,
        }),
    ));

    retained.upsert(RetainedUiNode::new(
        toolbar_id,
        UiWidget::Container,
        absolute_style(
            padding,
            padding,
            (pane_width - padding * 2.0).max(1.0),
            toolbar_height,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.13, 0.17, 0.22, 0.92)),
                border_color: Some(UiColor::rgba(0.28, 0.35, 0.46, 0.92)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        controls_row_id,
        UiWidget::Container,
        absolute_style(
            padding + toolbar_inner_padding,
            padding + toolbar_inner_padding,
            controls_row_width.max(1.0),
            controls_block_height.max(1.0),
            UiVisualStyle::default(),
        ),
    ));

    let mut controls_children = Vec::new();
    for (index, control_x, control_y, width) in control_layout {
        let (label, action) = controls[index].clone();
        let row_id = controls_row_id.child("item").child(index as u64);
        let label_id = row_id.child("label");
        let hit_id = row_id.child("hit");
        retained.upsert(RetainedUiNode::new(
            row_id,
            UiWidget::Container,
            absolute_style(
                control_x,
                control_y,
                width,
                controls_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.19, 0.24, 0.32, 0.92)),
                    border_color: Some(UiColor::rgba(0.30, 0.38, 0.50, 0.92)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(label),
                style: UiTextStyle {
                    color: UiColor::rgba(0.92, 0.95, 1.0, 1.0),
                    font_size: 10.5,
                    align_h: UiTextAlign::Center,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                control_x,
                control_y,
                width,
                controls_height,
                UiVisualStyle::default(),
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            hit_id,
            UiWidget::HitBox,
            absolute_style(
                control_x,
                control_y,
                width,
                controls_height,
                UiVisualStyle::default(),
            ),
        ));
        frame.actions.insert(hit_id, action);
        controls_children.push(row_id);
        controls_children.push(label_id);
        controls_children.push(hit_id);
    }
    retained.set_children(controls_row_id, controls_children);
    retained.upsert(RetainedUiNode::new(
        scene_frame_id,
        UiWidget::Container,
        absolute_style(
            scene_rect.x,
            scene_rect.y,
            scene_rect.width,
            scene_rect.height,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.04, 0.05, 0.06, 0.98)),
                border_color: Some(UiColor::rgba(0.18, 0.24, 0.32, 0.96)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        scene_image_id,
        UiWidget::Image(UiImage {
            texture_id: data.texture_id,
            tint: UiColor::WHITE,
            ..UiImage::default()
        }),
        absolute_style(
            0.0,
            0.0,
            scene_rect.width,
            scene_rect.height,
            UiVisualStyle::default(),
        ),
    ));

    if data.texture_id.is_none() {
        retained.upsert(RetainedUiNode::new(
            scene_placeholder_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("Viewport render target unavailable"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.74, 0.79, 0.86, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Center,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                0.0,
                0.0,
                scene_rect.width,
                scene_rect.height,
                UiVisualStyle::default(),
            ),
        ));
    }
    if matches!(mode, ViewportPaneMode::Play) && !data.play_mode_active {
        retained.upsert(RetainedUiNode::new(
            scene_play_inactive_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("Play mode inactive"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.90, 0.92, 0.95, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Center,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                0.0,
                0.0,
                scene_rect.width,
                scene_rect.height,
                UiVisualStyle::default(),
            ),
        ));
    }

    if let (ViewportPaneMode::Edit, Some(preview_texture_id), Some(preview_camera_name)) = (
        mode,
        data.preview_texture_id,
        data.preview_camera_name.as_ref(),
    ) {
        let preview_aspect =
            if data.preview_aspect_ratio.is_finite() && data.preview_aspect_ratio > 0.01 {
                data.preview_aspect_ratio
            } else {
                16.0 / 9.0
            };
        let min_preview_w = scene_rect.width.min(120.0).max(72.0);
        let mut preview_width = (scene_rect.width * data.preview_width_norm).max(min_preview_w);
        let mut preview_height = (preview_width / preview_aspect).max(48.0);
        let max_preview_h = scene_rect.height.max(48.0);
        if preview_height > max_preview_h {
            preview_height = max_preview_h;
            preview_width = (preview_height * preview_aspect).max(min_preview_w);
        }
        preview_width = preview_width.min(scene_rect.width.max(1.0));
        preview_height = preview_height.min(scene_rect.height.max(1.0));

        let max_offset_x = (scene_rect.width - preview_width).max(0.0);
        let max_offset_y = (scene_rect.height - preview_height).max(0.0);
        let mut preview_x = data.preview_position_norm[0].clamp(0.0, 1.0) * max_offset_x;
        let mut preview_y = data.preview_position_norm[1].clamp(0.0, 1.0) * max_offset_y;
        if !preview_x.is_finite() || !preview_y.is_finite() {
            preview_x = 0.0;
            preview_y = 0.0;
        }
        preview_x = preview_x.clamp(0.0, max_offset_x);
        preview_y = preview_y.clamp(0.0, max_offset_y);
        frame.preview_rect = Some(UiRect {
            x: preview_x,
            y: preview_y,
            width: preview_width,
            height: preview_height,
        });

        retained.upsert(RetainedUiNode::new(
            preview_frame_id,
            UiWidget::Container,
            absolute_style(
                preview_x,
                preview_y,
                preview_width,
                preview_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.04, 0.06, 0.09, 0.96)),
                    border_color: Some(UiColor::rgba(0.95, 0.63, 0.18, 0.96)),
                    border_width: 2.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            preview_image_id,
            UiWidget::Image(UiImage {
                texture_id: Some(preview_texture_id),
                tint: UiColor::WHITE,
                ..UiImage::default()
            }),
            absolute_style(
                0.0,
                0.0,
                preview_width,
                preview_height,
                UiVisualStyle::default(),
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            preview_label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(preview_camera_name.clone()),
                style: UiTextStyle {
                    color: UiColor::rgba(0.94, 0.95, 0.98, 1.0),
                    font_size: 10.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                6.0,
                4.0,
                (preview_width - 12.0).max(1.0),
                14.0,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.05, 0.07, 0.10, 0.72)),
                    border_color: None,
                    border_width: 0.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));
        let resize_handle_size = 12.0;
        retained.upsert(RetainedUiNode::new(
            preview_resize_handle_id,
            UiWidget::Container,
            absolute_style(
                (preview_width - resize_handle_size).max(0.0),
                (preview_height - resize_handle_size).max(0.0),
                resize_handle_size,
                resize_handle_size,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.10, 0.12, 0.15, 0.75)),
                    border_color: Some(UiColor::rgba(0.95, 0.63, 0.18, 0.96)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            preview_move_hit_id,
            UiWidget::HitBox,
            absolute_style(
                0.0,
                0.0,
                preview_width,
                preview_height,
                UiVisualStyle::default(),
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            preview_resize_hit_id,
            UiWidget::HitBox,
            absolute_style(
                (preview_width - resize_handle_size).max(0.0),
                (preview_height - resize_handle_size).max(0.0),
                resize_handle_size,
                resize_handle_size,
                UiVisualStyle::default(),
            ),
        ));
        frame.actions.insert(
            preview_move_hit_id,
            ViewportPaneAction::PreviewMove {
                tab_id: data.tab_id,
                mode,
            },
        );
        frame.actions.insert(
            preview_resize_hit_id,
            ViewportPaneAction::PreviewResize {
                tab_id: data.tab_id,
                mode,
            },
        );
        retained.set_children(
            preview_frame_id,
            [
                preview_image_id,
                preview_label_id,
                preview_resize_handle_id,
                preview_move_hit_id,
                preview_resize_hit_id,
            ],
        );
    }

    let mut scene_children = vec![scene_image_id];
    if data.texture_id.is_none() {
        scene_children.push(scene_placeholder_id);
    }
    if matches!(mode, ViewportPaneMode::Play) && !data.play_mode_active {
        scene_children.push(scene_play_inactive_id);
    }
    if matches!(mode, ViewportPaneMode::Edit)
        && data.preview_texture_id.is_some()
        && data.preview_camera_name.is_some()
    {
        scene_children.push(preview_frame_id);
    }
    retained.set_children(scene_frame_id, scene_children);

    retained.set_children(
        root_id,
        vec![background_id, toolbar_id, controls_row_id, scene_frame_id],
    );

    frame
}

fn fill_style(visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .width(UiDimension::percent(1.0))
            .height(UiDimension::percent(1.0))
            .build(),
        visual,
    }
}

fn absolute_style(x: f32, y: f32, width: f32, height: f32, visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .position_type(helmer_ui::UiPositionType::Absolute)
            .left(UiDimension::points(x))
            .top(UiDimension::points(y))
            .width(UiDimension::points(width.max(0.0)))
            .height(UiDimension::points(height.max(0.0)))
            .build(),
        visual,
    }
}
