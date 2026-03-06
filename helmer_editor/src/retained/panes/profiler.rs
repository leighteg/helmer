use std::collections::HashMap;

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiImage, UiLabel, UiLayoutBuilder,
    UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Debug, Default)]
pub struct ProfilerPaneRow {
    pub label: String,
    pub value: String,
}

#[derive(Clone, Debug, Default)]
pub struct ProfilerPaneData {
    pub enabled: bool,
    pub rows: Vec<ProfilerPaneRow>,
}

#[derive(Clone, Debug)]
pub enum ProfilerPaneAction {
    ToggleEnabled,
    Reset,
}

#[derive(Clone, Debug, Default)]
pub struct ProfilerPaneFrame {
    pub actions: HashMap<UiId, ProfilerPaneAction>,
}

pub fn build_profiler_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    data: &ProfilerPaneData,
) -> ProfilerPaneFrame {
    let mut frame = ProfilerPaneFrame::default();
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
    let controls = [
        (
            if data.enabled {
                "Profiler: On"
            } else {
                "Profiler: Off"
            },
            ProfilerPaneAction::ToggleEnabled,
            10.0,
            8.0,
            120.0,
        ),
        ("Reset", ProfilerPaneAction::Reset, 136.0, 8.0, 66.0),
    ];

    for (index, (label, action, x, y, width)) in controls.into_iter().enumerate() {
        let row_id = root_id.child("control").child(index as u64);
        let label_id = row_id.child("label");
        let hit_id = row_id.child("hit");

        draw_toolbar_item(retained, row_id, label_id, hit_id, label, x, y, width);
        frame.actions.insert(hit_id, action);
        children.push(row_id);
        children.push(label_id);
        children.push(hit_id);
    }

    if data.rows.is_empty() {
        let empty_id = root_id.child("empty");
        retained.upsert(RetainedUiNode::new(
            empty_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("No profiler data"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.68, 0.73, 0.80, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(10.0, 36.0, 420.0, 18.0, UiVisualStyle::default()),
        ));
        children.push(empty_id);
    } else {
        for (index, row) in data.rows.iter().take(24).enumerate() {
            let row_id = root_id.child("row").child(index as u64);
            retained.upsert(RetainedUiNode::new(
                row_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from(format!("{}: {}", row.label, row.value)),
                    style: UiTextStyle {
                        color: UiColor::rgba(0.80, 0.86, 0.94, 1.0),
                        font_size: 11.0,
                        align_h: UiTextAlign::Start,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                absolute_style(
                    10.0,
                    36.0 + index as f32 * 18.0,
                    740.0,
                    16.0,
                    UiVisualStyle::default(),
                ),
            ));
            children.push(row_id);
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
                background: Some(UiColor::rgba(0.22, 0.28, 0.37, 0.9)),
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
                color: UiColor::rgba(0.93, 0.96, 1.0, 1.0),
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
