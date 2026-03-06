use std::collections::HashMap;

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiImage, UiLabel, UiLayoutBuilder,
    UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Debug, Default)]
pub struct TimelinePaneData {
    pub playing: bool,
    pub current_time: f32,
    pub duration: f32,
    pub playback_rate: f32,
    pub track_group_count: usize,
    pub selected_key_count: usize,
}

#[derive(Clone, Debug)]
pub enum TimelinePaneAction {
    TogglePlayPause,
    Stop,
    RewindStep,
    ForwardStep,
}

#[derive(Clone, Debug, Default)]
pub struct TimelinePaneFrame {
    pub actions: HashMap<UiId, TimelinePaneAction>,
}

pub fn build_timeline_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    data: &TimelinePaneData,
) -> TimelinePaneFrame {
    let mut frame = TimelinePaneFrame::default();
    let background_id = root_id.child("background");
    let play_row_id = root_id.child("action").child("play");
    let play_label_id = play_row_id.child("label");
    let play_hit_id = play_row_id.child("hit");
    let stop_row_id = root_id.child("action").child("stop");
    let stop_label_id = stop_row_id.child("label");
    let stop_hit_id = stop_row_id.child("hit");
    let rewind_row_id = root_id.child("action").child("rewind");
    let rewind_label_id = rewind_row_id.child("label");
    let rewind_hit_id = rewind_row_id.child("hit");
    let forward_row_id = root_id.child("action").child("forward");
    let forward_label_id = forward_row_id.child("label");
    let forward_hit_id = forward_row_id.child("hit");

    retained.upsert(RetainedUiNode::new(
        root_id,
        UiWidget::Container,
        fill_style(UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        background_id,
        UiWidget::Container,
        fill_style(UiVisualStyle {
            background: Some(UiColor::rgba(0.10, 0.12, 0.15, 0.82)),
            border_color: Some(UiColor::rgba(0.24, 0.30, 0.37, 0.9)),
            border_width: 1.0,
            corner_radius: 0.0,
            clip: true,
        }),
    ));

    draw_action_item(
        retained,
        play_row_id,
        play_label_id,
        play_hit_id,
        if data.playing { "Pause" } else { "Play" },
        10.0,
        8.0,
        66.0,
    );
    frame
        .actions
        .insert(play_hit_id, TimelinePaneAction::TogglePlayPause);

    draw_action_item(
        retained,
        stop_row_id,
        stop_label_id,
        stop_hit_id,
        "Stop",
        82.0,
        8.0,
        66.0,
    );
    frame.actions.insert(stop_hit_id, TimelinePaneAction::Stop);

    draw_action_item(
        retained,
        rewind_row_id,
        rewind_label_id,
        rewind_hit_id,
        "- Step",
        154.0,
        8.0,
        74.0,
    );
    frame
        .actions
        .insert(rewind_hit_id, TimelinePaneAction::RewindStep);

    draw_action_item(
        retained,
        forward_row_id,
        forward_label_id,
        forward_hit_id,
        "+ Step",
        234.0,
        8.0,
        74.0,
    );
    frame
        .actions
        .insert(forward_hit_id, TimelinePaneAction::ForwardStep);

    let rows = [
        format!(
            "State: {}",
            if data.playing { "Playing" } else { "Stopped" }
        ),
        format!(
            "Time: {:.2}s / {:.2}s",
            data.current_time,
            data.duration.max(0.0)
        ),
        format!("Playback Rate: {:.2}x", data.playback_rate),
        format!("Track Groups: {}", data.track_group_count),
        format!("Selected Keys: {}", data.selected_key_count),
    ];

    let mut children = vec![
        background_id,
        play_row_id,
        play_label_id,
        play_hit_id,
        stop_row_id,
        stop_label_id,
        stop_hit_id,
        rewind_row_id,
        rewind_label_id,
        rewind_hit_id,
        forward_row_id,
        forward_label_id,
        forward_hit_id,
    ];
    for (index, row) in rows.iter().enumerate() {
        let row_id = root_id.child("row").child(index as u64);
        retained.upsert(RetainedUiNode::new(
            row_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(row.clone()),
                style: UiTextStyle {
                    color: if index == 0 {
                        UiColor::rgba(0.95, 0.97, 1.0, 1.0)
                    } else {
                        UiColor::rgba(0.80, 0.86, 0.94, 1.0)
                    },
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                10.0,
                34.0 + index as f32 * 20.0,
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

fn draw_action_item(
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
                background: Some(UiColor::rgba(0.19, 0.26, 0.35, 0.9)),
                border_color: Some(UiColor::rgba(0.30, 0.39, 0.53, 0.9)),
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
