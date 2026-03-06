use std::collections::HashMap;

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiLabel, UiLayoutBuilder, UiRect,
    UiStyle, UiTextAlign, UiTextField, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConsolePaneLevel {
    Trace,
    Debug,
    Log,
    Info,
    Warn,
    Error,
}

impl ConsolePaneLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Trace => "Trace",
            Self::Debug => "Debug",
            Self::Log => "Log",
            Self::Info => "Info",
            Self::Warn => "Warn",
            Self::Error => "Error",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConsolePaneEntry {
    pub sequence: u64,
    pub level: ConsolePaneLevel,
    pub target: String,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct ConsolePaneData {
    pub auto_scroll: bool,
    pub show_trace: bool,
    pub show_debug: bool,
    pub show_log: bool,
    pub show_info: bool,
    pub show_warn: bool,
    pub show_error: bool,
    pub search: String,
    pub entries: Vec<ConsolePaneEntry>,
    pub scroll: f32,
    pub focused_field: Option<ConsolePaneTextField>,
    pub text_cursors: HashMap<ConsolePaneTextField, usize>,
}

impl ConsolePaneData {
    pub fn level_enabled(&self, level: ConsolePaneLevel) -> bool {
        match level {
            ConsolePaneLevel::Trace => self.show_trace,
            ConsolePaneLevel::Debug => self.show_debug,
            ConsolePaneLevel::Log => self.show_log,
            ConsolePaneLevel::Info => self.show_info,
            ConsolePaneLevel::Warn => self.show_warn,
            ConsolePaneLevel::Error => self.show_error,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ConsolePaneAction {
    Clear,
    ToggleAutoScroll,
    ToggleLevel(ConsolePaneLevel),
    LogSurface,
    LogScrollbar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConsolePaneTextField {
    Search,
}

#[derive(Clone, Debug, Default)]
pub struct ConsolePaneFrame {
    pub actions: HashMap<UiId, ConsolePaneAction>,
    pub text_fields: HashMap<UiId, ConsolePaneTextField>,
    pub log_scroll_max: f32,
    pub log_scroll_region: Option<UiRect>,
}

pub fn build_console_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    viewport: UiRect,
    data: &ConsolePaneData,
) -> ConsolePaneFrame {
    let mut frame = ConsolePaneFrame::default();
    let pane_width = viewport.width.max(1.0);
    let pane_height = viewport.height.max(1.0);
    let content_width = (pane_width - 12.0).max(1.0);
    let background_id = root_id.child("background");
    let surface_hit_id = root_id.child("surface-hit");

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
        surface_hit_id,
        hit_widget(),
        fill_style(UiVisualStyle::default()),
    ));
    frame
        .actions
        .insert(surface_hit_id, ConsolePaneAction::LogSurface);
    frame.actions.insert(root_id, ConsolePaneAction::LogSurface);
    frame
        .actions
        .insert(background_id, ConsolePaneAction::LogSurface);

    let toolbar_y = 8.0;
    let toolbar_gap = 6.0;
    let clear_width = ((content_width * 0.17).clamp(56.0, 104.0)).min(content_width.max(24.0));
    let auto_width_available = (content_width - clear_width - toolbar_gap).max(44.0);
    let auto_width = if auto_width_available >= 88.0 {
        auto_width_available.min(148.0)
    } else {
        auto_width_available
    };

    let clear_row_id = root_id.child("toolbar").child("clear");
    let clear_label_id = clear_row_id.child("label");
    let clear_hit_id = clear_row_id.child("hit");
    draw_toolbar_item(
        retained,
        clear_row_id,
        clear_label_id,
        clear_hit_id,
        "Clear",
        6.0,
        toolbar_y,
        clear_width,
        true,
    );
    frame.actions.insert(clear_hit_id, ConsolePaneAction::Clear);

    let auto_row_id = root_id.child("toolbar").child("auto");
    let auto_label_id = auto_row_id.child("label");
    let auto_hit_id = auto_row_id.child("hit");
    draw_toolbar_item(
        retained,
        auto_row_id,
        auto_label_id,
        auto_hit_id,
        if data.auto_scroll {
            "Auto-scroll: On"
        } else {
            "Auto-scroll: Off"
        },
        6.0 + clear_width + toolbar_gap,
        toolbar_y,
        auto_width,
        data.auto_scroll,
    );
    frame
        .actions
        .insert(auto_hit_id, ConsolePaneAction::ToggleAutoScroll);

    let mut children = vec![
        background_id,
        surface_hit_id,
        clear_row_id,
        clear_label_id,
        clear_hit_id,
        auto_row_id,
        auto_label_id,
        auto_hit_id,
    ];

    let levels = [
        ConsolePaneLevel::Trace,
        ConsolePaneLevel::Debug,
        ConsolePaneLevel::Log,
        ConsolePaneLevel::Info,
        ConsolePaneLevel::Warn,
        ConsolePaneLevel::Error,
    ];

    let level_gap = 4.0;
    let level_min_width = 54.0;
    let level_columns = ((content_width + level_gap) / (level_min_width + level_gap))
        .floor()
        .max(1.0) as usize;
    let level_width = ((content_width - level_gap * (level_columns.saturating_sub(1)) as f32)
        / level_columns as f32)
        .max(1.0);
    let level_row_height = 22.0;
    let level_origin_y = toolbar_y + 24.0;

    for (index, level) in levels.iter().copied().enumerate() {
        let row = index / level_columns;
        let col = index % level_columns;
        let x = 6.0 + col as f32 * (level_width + level_gap);
        let y = level_origin_y + row as f32 * level_row_height;
        let row_id = root_id.child("level").child(index as u64);
        let label_id = row_id.child("label");
        let hit_id = row_id.child("hit");
        draw_toolbar_item(
            retained,
            row_id,
            label_id,
            hit_id,
            level.label(),
            x,
            y,
            level_width,
            data.level_enabled(level),
        );
        frame
            .actions
            .insert(hit_id, ConsolePaneAction::ToggleLevel(level));
        children.push(row_id);
        children.push(label_id);
        children.push(hit_id);
    }

    let level_rows = levels.len().div_ceil(level_columns);
    let search_y = level_origin_y + level_rows as f32 * level_row_height + 4.0;

    let search_label_id = root_id.child("search").child("label");
    let search_field_id = root_id.child("search").child("field");
    retained.upsert(RetainedUiNode::new(
        search_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Search:"),
            style: UiTextStyle {
                color: UiColor::rgba(0.74, 0.80, 0.88, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(6.0, search_y, 56.0, 20.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        search_field_id,
        UiWidget::TextField(UiTextField {
            text: UiTextValue::from(data.search.clone()),
            suffix: None,
            scroll_x: 0.0,
            style: UiTextStyle {
                color: UiColor::rgba(0.94, 0.97, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
            enabled: true,
            focused: data.focused_field == Some(ConsolePaneTextField::Search),
            cursor: Some(
                data.text_cursors
                    .get(&ConsolePaneTextField::Search)
                    .copied()
                    .unwrap_or_else(|| data.search.chars().count())
                    .min(data.search.chars().count()),
            ),
            selection: None,
            show_caret: data.focused_field == Some(ConsolePaneTextField::Search),
            selection_color: UiColor::rgba(0.34, 0.52, 0.84, 0.46),
            caret_color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
        }),
        absolute_style(
            64.0,
            (search_y - 2.0).max(0.0),
            (content_width - 60.0).max(20.0),
            24.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.14, 0.17, 0.22, 0.94)),
                border_color: Some(UiColor::rgba(0.31, 0.38, 0.50, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    frame
        .text_fields
        .insert(search_field_id, ConsolePaneTextField::Search);
    children.push(search_label_id);
    children.push(search_field_id);

    let search_lower = data.search.trim().to_ascii_lowercase();
    let has_search = !search_lower.is_empty();

    let row_height = 19.0;
    let min_log_height = (pane_height - 10.0).clamp(40.0, 72.0);
    let max_header_height = (pane_height - min_log_height - 6.0).max(0.0);
    let start_y = (search_y + 28.0).min(max_header_height);
    let log_height = (pane_height - start_y - 6.0).max(1.0);
    frame.log_scroll_region = Some(UiRect {
        x: 6.0,
        y: start_y,
        width: content_width,
        height: log_height,
    });
    let log_clip_id = root_id.child("log-clip");
    retained.upsert(RetainedUiNode::new(
        log_clip_id,
        UiWidget::Container,
        absolute_style(
            6.0,
            start_y,
            content_width,
            log_height,
            UiVisualStyle {
                background: None,
                border_color: None,
                border_width: 0.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    frame
        .actions
        .insert(log_clip_id, ConsolePaneAction::LogSurface);
    children.push(log_clip_id);

    let mut log_children = Vec::new();
    let log_surface_id = log_clip_id.child("surface");
    retained.upsert(RetainedUiNode::new(
        log_surface_id,
        hit_widget(),
        absolute_style(
            0.0,
            0.0,
            content_width,
            log_height,
            UiVisualStyle::default(),
        ),
    ));
    frame
        .actions
        .insert(log_surface_id, ConsolePaneAction::LogSurface);
    log_children.push(log_surface_id);

    let mut visible_entries = Vec::new();
    for entry in &data.entries {
        if !data.level_enabled(entry.level) {
            continue;
        }
        if has_search {
            let target = entry.target.to_ascii_lowercase();
            let message = entry.message.to_ascii_lowercase();
            if !target.contains(&search_lower) && !message.contains(&search_lower) {
                continue;
            }
        }
        visible_entries.push(entry);
    }

    let total_height = visible_entries.len() as f32 * row_height;
    let max_scroll = (total_height - log_height).max(0.0);
    frame.log_scroll_max = max_scroll;
    let scroll = if data.auto_scroll {
        max_scroll
    } else {
        data.scroll.clamp(0.0, max_scroll)
    };

    let mut shown = 0usize;
    for (index, entry) in visible_entries.into_iter().enumerate() {
        let y = index as f32 * row_height - scroll;
        if y + row_height < -1.0 {
            continue;
        }
        if y > log_height + 1.0 {
            break;
        }
        let row_id = log_clip_id.child("entry").child(index as u64);
        let label_id = row_id.child("label");
        let hit_id = row_id.child("hit");

        retained.upsert(RetainedUiNode::new(
            row_id,
            UiWidget::Container,
            absolute_style(
                0.0,
                y,
                content_width,
                row_height - 1.0,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.12, 0.15, 0.20, 0.76)),
                    border_color: Some(UiColor::rgba(0.20, 0.25, 0.33, 0.80)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));

        let text = if entry.target.is_empty() {
            format!(
                "#{:06} [{}] {}",
                entry.sequence,
                entry.level.label(),
                entry.message
            )
        } else {
            format!(
                "#{:06} [{}] [{}] {}",
                entry.sequence,
                entry.level.label(),
                entry.target,
                entry.message
            )
        };

        retained.upsert(RetainedUiNode::new(
            label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(text),
                style: UiTextStyle {
                    color: level_color(entry.level),
                    font_size: 11.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: true,
                },
            }),
            absolute_style(
                6.0,
                y + 1.0,
                (content_width - 8.0).max(20.0),
                row_height - 3.0,
                UiVisualStyle::default(),
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            hit_id,
            hit_widget(),
            absolute_style(
                0.0,
                y,
                content_width,
                row_height - 1.0,
                UiVisualStyle::default(),
            ),
        ));
        frame.actions.insert(hit_id, ConsolePaneAction::LogSurface);
        frame.actions.insert(row_id, ConsolePaneAction::LogSurface);
        frame
            .actions
            .insert(label_id, ConsolePaneAction::LogSurface);

        log_children.push(row_id);
        log_children.push(label_id);
        log_children.push(hit_id);
        shown += 1;
    }

    if shown == 0 {
        let empty_id = log_clip_id.child("empty");
        retained.upsert(RetainedUiNode::new(
            empty_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("No console output"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.68, 0.73, 0.80, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                6.0,
                0.0,
                content_width,
                row_height,
                UiVisualStyle::default(),
            ),
        ));
        log_children.push(empty_id);
    }

    if max_scroll > f32::EPSILON {
        let track_id = log_clip_id.child("scrollbar-track");
        let thumb_id = log_clip_id.child("scrollbar-thumb");
        let hit_id = log_clip_id.child("scrollbar-hit");
        let track_w = 8.0;
        let track_x = (content_width - track_w - 2.0).max(0.0);
        let track_h = log_height.max(1.0);
        let thumb_h = ((log_height / total_height.max(1.0)) * track_h).clamp(18.0, track_h);
        let thumb_travel = (track_h - thumb_h).max(0.0);
        let thumb_y = if max_scroll <= f32::EPSILON {
            0.0
        } else {
            ((scroll / max_scroll) * thumb_travel).clamp(0.0, thumb_travel)
        };

        retained.upsert(RetainedUiNode::new(
            track_id,
            UiWidget::Container,
            absolute_style(
                track_x,
                0.0,
                track_w,
                track_h,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.10, 0.13, 0.17, 0.92)),
                    border_color: Some(UiColor::rgba(0.23, 0.29, 0.37, 0.92)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            thumb_id,
            UiWidget::Container,
            absolute_style(
                track_x + 1.0,
                thumb_y + 1.0,
                (track_w - 2.0).max(1.0),
                (thumb_h - 2.0).max(8.0),
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.47, 0.58, 0.73, 0.95)),
                    border_color: Some(UiColor::rgba(0.66, 0.77, 0.92, 0.95)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            hit_id,
            hit_widget(),
            absolute_style(track_x, 0.0, track_w, track_h, UiVisualStyle::default()),
        ));
        frame
            .actions
            .insert(hit_id, ConsolePaneAction::LogScrollbar);
        log_children.push(track_id);
        log_children.push(thumb_id);
        log_children.push(hit_id);
    }

    for id in log_children.iter().copied() {
        frame
            .actions
            .entry(id)
            .or_insert(ConsolePaneAction::LogSurface);
    }
    for id in children.iter().copied() {
        frame
            .actions
            .entry(id)
            .or_insert(ConsolePaneAction::LogSurface);
    }

    retained.set_children(log_clip_id, log_children);
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
        UiWidget::HitBox,
        absolute_style(x, y, width, 20.0, UiVisualStyle::default()),
    ));
}

fn hit_widget() -> UiWidget {
    UiWidget::HitBox
}

fn level_color(level: ConsolePaneLevel) -> UiColor {
    match level {
        ConsolePaneLevel::Trace => UiColor::rgba(0.62, 0.62, 0.65, 1.0),
        ConsolePaneLevel::Debug => UiColor::rgba(0.72, 0.76, 0.84, 1.0),
        ConsolePaneLevel::Log => UiColor::rgba(0.82, 0.84, 0.88, 1.0),
        ConsolePaneLevel::Info => UiColor::rgba(0.72, 0.82, 0.96, 1.0),
        ConsolePaneLevel::Warn => UiColor::rgba(0.92, 0.76, 0.44, 1.0),
        ConsolePaneLevel::Error => UiColor::rgba(0.92, 0.46, 0.46, 1.0),
    }
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
