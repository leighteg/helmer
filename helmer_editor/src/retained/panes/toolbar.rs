use std::collections::HashMap;

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiImage, UiLabel, UiLayoutBuilder,
    UiRect, UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Debug, Default)]
pub struct ToolbarPaneData {
    pub project_label: String,
    pub scene_label: String,
    pub world_label: String,
    pub can_undo: bool,
    pub can_redo: bool,
}

#[derive(Clone, Debug)]
pub enum ToolbarPaneAction {
    NewScene,
    SaveScene,
    TogglePlayMode,
    Undo,
    Redo,
    OpenLayoutMenu,
}

#[derive(Clone, Debug, Default)]
pub struct ToolbarPaneFrame {
    pub actions: HashMap<UiId, ToolbarPaneAction>,
}

pub fn build_toolbar_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    viewport: UiRect,
    data: &ToolbarPaneData,
) -> ToolbarPaneFrame {
    let mut frame = ToolbarPaneFrame::default();
    let pane_width = viewport.width.max(1.0);
    let pane_height = viewport.height.max(1.0);
    let content_width = (pane_width - 12.0).max(1.0);
    let background_id = root_id.child("background");

    retained.upsert(RetainedUiNode::new(
        root_id,
        UiWidget::Container,
        fill_style(UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        background_id,
        UiWidget::Container,
        fill_style(UiVisualStyle {
            background: Some(UiColor::rgba(0.10, 0.12, 0.15, 0.86)),
            border_color: Some(UiColor::rgba(0.24, 0.30, 0.37, 0.9)),
            border_width: 1.0,
            corner_radius: 0.0,
            clip: true,
        }),
    ));

    let mut children = vec![background_id];

    let button_height = 20.0;
    let button_gap = 6.0;
    let mut button_x = 6.0;
    let mut button_y = 6.0;
    let actions = [
        ("Undo", ToolbarPaneAction::Undo, data.can_undo),
        ("Redo", ToolbarPaneAction::Redo, data.can_redo),
        ("New Scene", ToolbarPaneAction::NewScene, true),
        ("Save", ToolbarPaneAction::SaveScene, true),
        (
            if data.world_label == "Play" {
                "Stop"
            } else {
                "Play"
            },
            ToolbarPaneAction::TogglePlayMode,
            true,
        ),
        ("Layout", ToolbarPaneAction::OpenLayoutMenu, true),
    ];

    for (index, (label, action, enabled)) in actions.into_iter().enumerate() {
        let desired_width = (label.chars().count() as f32 * 7.2 + 26.0)
            .clamp(58.0, 146.0)
            .min(content_width.max(1.0));
        if button_x > 6.0 && button_x + desired_width > pane_width - 6.0 {
            button_x = 6.0;
            button_y += button_height + 4.0;
        }
        let row_id = root_id.child("action").child(index as u64);
        let label_id = row_id.child("label");
        let hit_id = row_id.child("hit");

        draw_toolbar_item(
            retained,
            row_id,
            label_id,
            hit_id,
            label,
            button_x,
            button_y,
            desired_width,
            enabled,
        );
        frame.actions.insert(hit_id, action);
        children.push(row_id);
        children.push(label_id);
        children.push(hit_id);
        button_x += desired_width + button_gap;
    }

    let info_start_y = button_y + button_height + 8.0;
    let info_rows = [
        format!("Project: {}", data.project_label),
        format!("Scene: {}", data.scene_label),
        format!("World: {}", data.world_label),
    ];
    for (index, row) in info_rows.iter().enumerate() {
        let y = info_start_y + index as f32 * 16.0;
        if y > pane_height - 14.0 {
            break;
        }
        let row_id = root_id.child("info").child(index as u64);
        retained.upsert(RetainedUiNode::new(
            row_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(row.clone()),
                style: UiTextStyle {
                    color: UiColor::rgba(0.78, 0.84, 0.92, 1.0),
                    font_size: 11.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                8.0,
                y,
                (pane_width - 16.0).max(1.0),
                16.0,
                UiVisualStyle::default(),
            ),
        ));
        children.push(row_id);
    }

    retained.set_children(root_id, children);
    frame
}

fn draw_toolbar_item(
    retained: &mut RetainedUi,
    row_id: UiId,
    label_id: UiId,
    hit_id: UiId,
    text: &str,
    x: f32,
    y: f32,
    width: f32,
    enabled: bool,
) {
    retained.upsert(RetainedUiNode::new(
        row_id,
        UiWidget::Container,
        absolute_style(
            x,
            y,
            width,
            20.0,
            UiVisualStyle {
                background: Some(if enabled {
                    UiColor::rgba(0.22, 0.28, 0.37, 0.9)
                } else {
                    UiColor::rgba(0.14, 0.18, 0.24, 0.65)
                }),
                border_color: Some(UiColor::rgba(0.30, 0.38, 0.50, 0.85)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(text.to_string()),
            style: UiTextStyle {
                color: if enabled {
                    UiColor::rgba(0.92, 0.95, 0.99, 1.0)
                } else {
                    UiColor::rgba(0.60, 0.66, 0.74, 1.0)
                },
                font_size: 11.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(x, y, width, 20.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        hit_id,
        UiWidget::Image(UiImage {
            tint: UiColor::TRANSPARENT,
            ..UiImage::default()
        }),
        absolute_style(x, y, width, 20.0, UiVisualStyle::default()),
    ));
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
