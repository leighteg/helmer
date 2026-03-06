use std::collections::HashMap;

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiImage, UiLabel, UiLayoutBuilder,
    UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Debug, Default)]
pub struct MaterialEditorPaneData {
    pub selected_asset: Option<String>,
    pub status: Option<String>,
}

#[derive(Clone, Debug)]
pub enum MaterialEditorPaneAction {
    RefreshSelection,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialEditorPaneFrame {
    pub actions: HashMap<UiId, MaterialEditorPaneAction>,
}

pub fn build_material_editor_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    data: &MaterialEditorPaneData,
) -> MaterialEditorPaneFrame {
    let mut frame = MaterialEditorPaneFrame::default();
    let background_id = root_id.child("background");
    let refresh_row_id = root_id.child("refresh");
    let refresh_label_id = refresh_row_id.child("label");
    let refresh_hit_id = refresh_row_id.child("hit");

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

    retained.upsert(RetainedUiNode::new(
        refresh_row_id,
        UiWidget::Container,
        absolute_style(
            10.0,
            8.0,
            144.0,
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
        refresh_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Refresh Selection"),
            style: UiTextStyle {
                color: UiColor::rgba(0.93, 0.96, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, 8.0, 144.0, 20.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        refresh_hit_id,
        UiWidget::Image(UiImage {
            tint: UiColor::TRANSPARENT,
            ..UiImage::default()
        }),
        absolute_style(10.0, 8.0, 144.0, 20.0, UiVisualStyle::default()),
    ));
    frame
        .actions
        .insert(refresh_hit_id, MaterialEditorPaneAction::RefreshSelection);

    let selected_id = root_id.child("selected");
    retained.upsert(RetainedUiNode::new(
        selected_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(match data.selected_asset.as_ref() {
                Some(path) => format!("Selected: {path}"),
                None => "Selected: <none>".to_string(),
            }),
            style: UiTextStyle {
                color: UiColor::rgba(0.80, 0.86, 0.94, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, 36.0, 740.0, 16.0, UiVisualStyle::default()),
    ));

    let status_id = root_id.child("status");
    retained.upsert(RetainedUiNode::new(
        status_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(
                data.status
                    .clone()
                    .unwrap_or_else(|| "No material selected".to_string()),
            ),
            style: UiTextStyle {
                color: UiColor::rgba(0.72, 0.78, 0.87, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, 54.0, 740.0, 16.0, UiVisualStyle::default()),
    ));

    retained.set_children(
        root_id,
        [
            background_id,
            refresh_row_id,
            refresh_label_id,
            refresh_hit_id,
            selected_id,
            status_id,
        ],
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
