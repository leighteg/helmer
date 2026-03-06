use std::collections::HashMap;

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiImage, UiLabel, UiLayoutBuilder,
    UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Debug, Default)]
pub struct HistoryPaneEntry {
    pub label: String,
    pub active: bool,
}

#[derive(Clone, Debug, Default)]
pub struct HistoryPaneData {
    pub can_undo: bool,
    pub can_redo: bool,
    pub undo_label: Option<String>,
    pub redo_label: Option<String>,
    pub entries: Vec<HistoryPaneEntry>,
}

#[derive(Clone, Debug)]
pub enum HistoryPaneAction {
    Undo,
    Redo,
}

#[derive(Clone, Debug, Default)]
pub struct HistoryPaneFrame {
    pub actions: HashMap<UiId, HistoryPaneAction>,
}

pub fn build_history_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    data: &HistoryPaneData,
) -> HistoryPaneFrame {
    let mut frame = HistoryPaneFrame::default();
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

    let undo_row_id = root_id.child("toolbar").child("undo");
    let undo_label_id = undo_row_id.child("label");
    let undo_hit_id = undo_row_id.child("hit");
    draw_toolbar_item(
        retained,
        undo_row_id,
        undo_label_id,
        undo_hit_id,
        "Undo",
        10.0,
        8.0,
        66.0,
        data.can_undo,
    );
    frame.actions.insert(undo_hit_id, HistoryPaneAction::Undo);
    children.push(undo_row_id);
    children.push(undo_label_id);
    children.push(undo_hit_id);

    let redo_row_id = root_id.child("toolbar").child("redo");
    let redo_label_id = redo_row_id.child("label");
    let redo_hit_id = redo_row_id.child("hit");
    draw_toolbar_item(
        retained,
        redo_row_id,
        redo_label_id,
        redo_hit_id,
        "Redo",
        82.0,
        8.0,
        66.0,
        data.can_redo,
    );
    frame.actions.insert(redo_hit_id, HistoryPaneAction::Redo);
    children.push(redo_row_id);
    children.push(redo_label_id);
    children.push(redo_hit_id);

    let undo_desc_id = root_id.child("undo-desc");
    retained.upsert(RetainedUiNode::new(
        undo_desc_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(match data.undo_label.as_ref() {
                Some(label) => format!("Undo: {label}"),
                None => "Undo: <none>".to_string(),
            }),
            style: UiTextStyle {
                color: UiColor::rgba(0.78, 0.84, 0.92, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(154.0, 10.0, 620.0, 16.0, UiVisualStyle::default()),
    ));
    children.push(undo_desc_id);

    let redo_desc_id = root_id.child("redo-desc");
    retained.upsert(RetainedUiNode::new(
        redo_desc_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(match data.redo_label.as_ref() {
                Some(label) => format!("Redo: {label}"),
                None => "Redo: <none>".to_string(),
            }),
            style: UiTextStyle {
                color: UiColor::rgba(0.70, 0.77, 0.86, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(154.0, 26.0, 620.0, 16.0, UiVisualStyle::default()),
    ));
    children.push(redo_desc_id);

    let row_start = 48.0;
    let row_height = 18.0;
    if data.entries.is_empty() {
        let empty_id = root_id.child("empty");
        retained.upsert(RetainedUiNode::new(
            empty_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("No history entries"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.66, 0.72, 0.80, 1.0),
                    font_size: 11.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(10.0, row_start, 420.0, row_height, UiVisualStyle::default()),
        ));
        children.push(empty_id);
    } else {
        for (index, entry) in data.entries.iter().take(32).enumerate() {
            let row_id = root_id.child("entry").child(index as u64);
            let label_id = row_id.child("label");
            let y = row_start + index as f32 * row_height;
            retained.upsert(RetainedUiNode::new(
                row_id,
                UiWidget::Container,
                absolute_style(
                    6.0,
                    y,
                    740.0,
                    row_height - 1.0,
                    UiVisualStyle {
                        background: Some(if entry.active {
                            UiColor::rgba(0.22, 0.30, 0.45, 0.92)
                        } else {
                            UiColor::rgba(0.12, 0.16, 0.22, 0.72)
                        }),
                        border_color: Some(if entry.active {
                            UiColor::rgba(0.52, 0.68, 0.92, 0.95)
                        } else {
                            UiColor::rgba(0.24, 0.29, 0.36, 0.82)
                        }),
                        border_width: 1.0,
                        corner_radius: 0.0,
                        clip: false,
                    },
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                label_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from(entry.label.clone()),
                    style: UiTextStyle {
                        color: if entry.active {
                            UiColor::rgba(0.96, 0.98, 1.0, 1.0)
                        } else {
                            UiColor::rgba(0.82, 0.88, 0.95, 1.0)
                        },
                        font_size: 11.0,
                        align_h: UiTextAlign::Start,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                absolute_style(
                    12.0,
                    y + 1.0,
                    734.0,
                    row_height - 3.0,
                    UiVisualStyle::default(),
                ),
            ));
            children.push(row_id);
            children.push(label_id);
        }
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
