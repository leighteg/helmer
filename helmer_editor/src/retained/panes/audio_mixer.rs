use std::collections::HashMap;

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiImage, UiLabel, UiLayoutBuilder,
    UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Debug, Default)]
pub struct AudioMixerPaneData {
    pub enabled: bool,
    pub emitter_count: usize,
    pub listener_count: usize,
}

#[derive(Clone, Debug)]
pub enum AudioMixerPaneAction {
    ToggleEnabled,
}

#[derive(Clone, Debug, Default)]
pub struct AudioMixerPaneFrame {
    pub actions: HashMap<UiId, AudioMixerPaneAction>,
}

pub fn build_audio_mixer_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    data: &AudioMixerPaneData,
) -> AudioMixerPaneFrame {
    let mut frame = AudioMixerPaneFrame::default();
    let background_id = root_id.child("background");
    let toggle_row_id = root_id.child("toggle");
    let toggle_label_id = toggle_row_id.child("label");
    let toggle_hit_id = toggle_row_id.child("hit");

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
        toggle_row_id,
        UiWidget::Container,
        absolute_style(
            10.0,
            8.0,
            120.0,
            20.0,
            UiVisualStyle {
                background: Some(if data.enabled {
                    UiColor::rgba(0.20, 0.32, 0.26, 0.92)
                } else {
                    UiColor::rgba(0.32, 0.20, 0.22, 0.92)
                }),
                border_color: Some(UiColor::rgba(0.30, 0.38, 0.50, 0.85)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        toggle_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(if data.enabled {
                "Audio: Enabled"
            } else {
                "Audio: Disabled"
            }),
            style: UiTextStyle {
                color: UiColor::rgba(0.93, 0.96, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, 8.0, 120.0, 20.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        toggle_hit_id,
        UiWidget::Image(UiImage {
            tint: UiColor::TRANSPARENT,
            ..UiImage::default()
        }),
        absolute_style(10.0, 8.0, 120.0, 20.0, UiVisualStyle::default()),
    ));
    frame
        .actions
        .insert(toggle_hit_id, AudioMixerPaneAction::ToggleEnabled);

    let rows = [
        format!("Emitters: {}", data.emitter_count),
        format!("Listeners: {}", data.listener_count),
    ];
    let mut children = vec![background_id, toggle_row_id, toggle_label_id, toggle_hit_id];
    for (index, row) in rows.iter().enumerate() {
        let row_id = root_id.child("row").child(index as u64);
        retained.upsert(RetainedUiNode::new(
            row_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(row.clone()),
                style: UiTextStyle {
                    color: UiColor::rgba(0.80, 0.86, 0.94, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                10.0,
                36.0 + index as f32 * 20.0,
                620.0,
                18.0,
                UiVisualStyle::default(),
            ),
        ));
        children.push(row_id);
    }

    retained.set_children(root_id, children);
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
