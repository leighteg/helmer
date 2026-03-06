use std::{collections::HashMap, path::PathBuf};

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiLabel, UiLayoutBuilder, UiRect,
    UiStyle, UiTextAlign, UiTextField, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Debug)]
pub struct ContentBrowserPaneEntry {
    pub path: PathBuf,
    pub label: String,
    pub depth: usize,
    pub is_dir: bool,
    pub selected: bool,
    pub has_children: bool,
    pub expanded: bool,
}

#[derive(Clone, Debug)]
pub struct ContentBrowserPaneLocationEntry {
    pub path: PathBuf,
    pub label: String,
    pub selected: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ContentBrowserPaneData {
    pub root_path: Option<String>,
    pub current_dir: Option<String>,
    pub filter: String,
    pub status: Option<String>,
    pub sidebar_entries: Vec<ContentBrowserPaneEntry>,
    pub grid_entries: Vec<ContentBrowserPaneEntry>,
    pub location_entries: Vec<ContentBrowserPaneLocationEntry>,
    pub can_navigate_up: bool,
    pub location_dropdown_open: bool,
    pub grid_scroll: f32,
    pub tile_size: f32,
    pub focused_field: Option<ContentBrowserPaneTextField>,
    pub text_cursors: HashMap<ContentBrowserPaneTextField, usize>,
}

#[derive(Clone, Debug)]
pub enum ContentBrowserPaneAction {
    NavigateUp,
    Refresh,
    ToggleLocationDropdown,
    TileSizeSlider,
    SelectLocation(PathBuf),
    SelectFolder(PathBuf),
    ToggleFolderExpanded(PathBuf),
    SelectEntry {
        path: PathBuf,
        is_dir: bool,
        index: usize,
    },
    GridSurface,
    GridScrollbar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContentBrowserPaneTextField {
    Filter,
}

#[derive(Clone, Debug, Default)]
pub struct ContentBrowserPaneFrame {
    pub actions: HashMap<UiId, ContentBrowserPaneAction>,
    pub text_fields: HashMap<UiId, ContentBrowserPaneTextField>,
    pub grid_scroll_max: f32,
    pub grid_scroll_region: Option<UiRect>,
}

pub fn build_content_browser_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    viewport: UiRect,
    data: &ContentBrowserPaneData,
) -> ContentBrowserPaneFrame {
    let mut frame = ContentBrowserPaneFrame::default();

    let pane_width = viewport.width.max(1.0);
    let pane_height = viewport.height.max(1.0);

    let header_x = 8.0;
    let header_width = (pane_width - 16.0).max(48.0);
    let show_filter = false;
    let show_slider = false;
    let show_status = false;
    let mut header_cursor_y = 8.0;

    retained.upsert(RetainedUiNode::new(
        root_id,
        UiWidget::Container,
        fill_style(UiVisualStyle::default()),
    ));

    let background_id = root_id.child("background");
    let surface_hit_id = root_id.child("surface-hit");
    retained.upsert(RetainedUiNode::new(
        background_id,
        UiWidget::Container,
        fill_style(UiVisualStyle {
            background: Some(UiColor::rgba(0.10, 0.12, 0.15, 0.86)),
            border_color: Some(UiColor::rgba(0.24, 0.30, 0.37, 0.90)),
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
        .insert(surface_hit_id, ContentBrowserPaneAction::GridSurface);
    frame
        .actions
        .insert(root_id, ContentBrowserPaneAction::GridSurface);
    frame
        .actions
        .insert(background_id, ContentBrowserPaneAction::GridSurface);

    let mut children = vec![background_id, surface_hit_id];

    let controls_row_id = root_id.child("controls-row");
    let filter_label_id = controls_row_id.child("filter-label");
    let filter_field_id = controls_row_id.child("filter-field");
    let refresh_button_id = controls_row_id.child("refresh-button");
    let refresh_hit_id = refresh_button_id.child("hit");
    let slider_track_id = controls_row_id.child("tile-size-track");
    let slider_fill_id = slider_track_id.child("fill");
    let slider_knob_id = slider_track_id.child("knob");
    let slider_hit_id = slider_track_id.child("hit");
    let slider_value_id = controls_row_id.child("tile-size-value");
    let slider_text_id = controls_row_id.child("tile-size-text");

    let tile_size = data.tile_size.clamp(64.0, 220.0);
    let tile_min = 64.0;
    let tile_max = 220.0;
    let tile_norm = ((tile_size - tile_min) / (tile_max - tile_min)).clamp(0.0, 1.0);
    let row_h = 24.0;
    let filter_label_w = 38.0;
    let refresh_w = 58.0;
    let slider_w = (header_width * 0.26).clamp(84.0, 180.0);
    let value_w = 34.0;
    let slider_label_w = 50.0;
    let filter_field_w =
        (header_width - filter_label_w - refresh_w - slider_w - value_w - slider_label_w - 28.0)
            .max(52.0);
    let row_y = header_cursor_y;
    let filter_x = header_x + filter_label_w + 2.0;
    let refresh_x = filter_x + filter_field_w + 6.0;
    let slider_x = refresh_x + refresh_w + 8.0;
    let value_x = slider_x + slider_w + 6.0;
    let slider_text_x = value_x + value_w + 4.0;
    let knob_w = 10.0f32.min(slider_w.max(1.0));
    let knob_x = (slider_x + tile_norm * (slider_w - knob_w)).max(slider_x);

    retained.upsert(RetainedUiNode::new(
        controls_row_id,
        UiWidget::Container,
        absolute_style(
            header_x,
            row_y,
            header_width,
            row_h,
            UiVisualStyle::default(),
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        filter_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Filter:"),
            style: UiTextStyle {
                color: UiColor::rgba(0.72, 0.78, 0.87, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            header_x,
            row_y,
            filter_label_w,
            row_h,
            UiVisualStyle::default(),
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        filter_field_id,
        UiWidget::TextField(UiTextField {
            text: UiTextValue::from(data.filter.clone()),
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
            focused: data.focused_field == Some(ContentBrowserPaneTextField::Filter),
            cursor: Some(
                data.text_cursors
                    .get(&ContentBrowserPaneTextField::Filter)
                    .copied()
                    .unwrap_or_else(|| data.filter.chars().count())
                    .min(data.filter.chars().count()),
            ),
            selection: None,
            show_caret: data.focused_field == Some(ContentBrowserPaneTextField::Filter),
            selection_color: UiColor::rgba(0.34, 0.52, 0.84, 0.46),
            caret_color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
        }),
        absolute_style(
            filter_x,
            row_y,
            filter_field_w,
            row_h,
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
        .insert(filter_field_id, ContentBrowserPaneTextField::Filter);
    retained.upsert(RetainedUiNode::new(
        refresh_button_id,
        UiWidget::Container,
        absolute_style(
            refresh_x,
            row_y,
            refresh_w,
            row_h,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.18, 0.24, 0.33, 0.90)),
                border_color: Some(UiColor::rgba(0.30, 0.40, 0.53, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        refresh_button_id.child("label"),
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Refresh"),
            style: UiTextStyle {
                color: UiColor::rgba(0.90, 0.94, 1.0, 1.0),
                font_size: 10.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(refresh_x, row_y, refresh_w, row_h, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        refresh_hit_id,
        hit_widget(),
        absolute_style(refresh_x, row_y, refresh_w, row_h, UiVisualStyle::default()),
    ));
    frame
        .actions
        .insert(refresh_hit_id, ContentBrowserPaneAction::Refresh);

    retained.upsert(RetainedUiNode::new(
        slider_track_id,
        UiWidget::Container,
        absolute_style(
            slider_x,
            row_y + 7.0,
            slider_w,
            10.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.16, 0.20, 0.28, 0.92)),
                border_color: Some(UiColor::rgba(0.32, 0.39, 0.52, 0.92)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        slider_fill_id,
        UiWidget::Container,
        absolute_style(
            slider_x,
            row_y + 7.0,
            (slider_w * tile_norm).max(0.0),
            10.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.29, 0.41, 0.61, 0.96)),
                border_color: None,
                border_width: 0.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        slider_knob_id,
        UiWidget::Container,
        absolute_style(
            knob_x,
            row_y + 5.0,
            knob_w,
            14.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.84, 0.90, 1.0, 0.98)),
                border_color: Some(UiColor::rgba(0.38, 0.46, 0.60, 0.95)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        slider_hit_id,
        hit_widget(),
        absolute_style(
            slider_x,
            row_y + 2.0,
            slider_w,
            18.0,
            UiVisualStyle::default(),
        ),
    ));
    frame
        .actions
        .insert(slider_hit_id, ContentBrowserPaneAction::TileSizeSlider);
    retained.upsert(RetainedUiNode::new(
        slider_value_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(format!("{tile_size:.0}")),
            style: UiTextStyle {
                color: UiColor::rgba(0.88, 0.93, 0.99, 1.0),
                font_size: 10.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(value_x, row_y, value_w, row_h, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        slider_text_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Tile Size"),
            style: UiTextStyle {
                color: UiColor::rgba(0.74, 0.80, 0.88, 1.0),
                font_size: 10.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            slider_text_x,
            row_y,
            (header_x + header_width - slider_text_x).max(24.0),
            row_h,
            UiVisualStyle::default(),
        ),
    ));
    children.push(controls_row_id);
    children.push(filter_label_id);
    children.push(filter_field_id);
    children.push(refresh_button_id);
    children.push(refresh_button_id.child("label"));
    children.push(refresh_hit_id);
    children.push(slider_track_id);
    children.push(slider_fill_id);
    children.push(slider_knob_id);
    children.push(slider_hit_id);
    children.push(slider_value_id);
    children.push(slider_text_id);
    header_cursor_y += row_h + 6.0;

    let location_label_id = root_id.child("location-label");
    let location_button_id = root_id.child("location-button");
    let location_text_id = location_button_id.child("text");
    let location_hit_id = location_button_id.child("hit");
    let location_label_w = 56.0;
    let location_button_x = header_x + location_label_w;
    let location_button_w = (header_width - location_label_w).max(44.0);
    let location_label = data
        .current_dir
        .as_ref()
        .cloned()
        .unwrap_or_else(|| "<none>".to_string());
    let dropdown_glyph = if data.location_dropdown_open {
        "v"
    } else {
        ">"
    };

    retained.upsert(RetainedUiNode::new(
        location_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Location:"),
            style: UiTextStyle {
                color: UiColor::rgba(0.74, 0.80, 0.88, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            header_x,
            header_cursor_y,
            location_label_w,
            22.0,
            UiVisualStyle::default(),
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        location_button_id,
        UiWidget::Container,
        absolute_style(
            location_button_x,
            header_cursor_y,
            location_button_w,
            22.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.14, 0.17, 0.22, 0.94)),
                border_color: Some(UiColor::rgba(0.31, 0.38, 0.50, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        location_text_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(format!("{location_label} {dropdown_glyph}")),
            style: UiTextStyle {
                color: UiColor::rgba(0.90, 0.93, 0.98, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            location_button_x + 6.0,
            header_cursor_y,
            (location_button_w - 12.0).max(20.0),
            22.0,
            UiVisualStyle::default(),
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        location_hit_id,
        hit_widget(),
        absolute_style(
            location_button_x,
            header_cursor_y,
            location_button_w,
            22.0,
            UiVisualStyle::default(),
        ),
    ));
    frame.actions.insert(
        location_hit_id,
        ContentBrowserPaneAction::ToggleLocationDropdown,
    );
    children.push(location_label_id);
    children.push(location_button_id);
    children.push(location_text_id);
    children.push(location_hit_id);
    header_cursor_y += 24.0;

    if data.location_dropdown_open {
        let dropdown_space = (pane_height - header_cursor_y - 22.0).max(0.0);
        let max_dropdown_rows = (dropdown_space / 18.0).floor().max(0.0) as usize;
        let visible_locations = data
            .location_entries
            .iter()
            .take(max_dropdown_rows.min(10))
            .collect::<Vec<_>>();
        let dropdown_height = visible_locations.len() as f32 * 18.0;
        if dropdown_height > f32::EPSILON {
            let dropdown_bg_id = root_id.child("location-dropdown-bg");
            retained.upsert(RetainedUiNode::new(
                dropdown_bg_id,
                UiWidget::Container,
                absolute_style(
                    location_button_x,
                    header_cursor_y,
                    location_button_w,
                    dropdown_height,
                    UiVisualStyle {
                        background: Some(UiColor::rgba(0.12, 0.15, 0.20, 0.94)),
                        border_color: Some(UiColor::rgba(0.29, 0.36, 0.46, 0.92)),
                        border_width: 1.0,
                        corner_radius: 0.0,
                        clip: true,
                    },
                ),
            ));
            children.push(dropdown_bg_id);

            if visible_locations.is_empty() {
                let empty_id = root_id.child("location-empty");
                retained.upsert(RetainedUiNode::new(
                    empty_id,
                    UiWidget::Label(UiLabel {
                        text: UiTextValue::from("No directory entries"),
                        style: UiTextStyle {
                            color: UiColor::rgba(0.66, 0.72, 0.80, 1.0),
                            font_size: 10.0,
                            align_h: UiTextAlign::Start,
                            align_v: UiTextAlign::Center,
                            wrap: false,
                        },
                    }),
                    absolute_style(
                        location_button_x + 6.0,
                        header_cursor_y,
                        (location_button_w - 12.0).max(20.0),
                        18.0,
                        UiVisualStyle::default(),
                    ),
                ));
                children.push(empty_id);
            } else {
                for (index, entry) in visible_locations.into_iter().enumerate() {
                    let row_y = header_cursor_y + index as f32 * 18.0;
                    let row_id = root_id.child("location-row").child(index as u64);
                    let row_label_id = row_id.child("label");
                    let row_hit_id = row_id.child("hit");

                    retained.upsert(RetainedUiNode::new(
                        row_id,
                        UiWidget::Container,
                        absolute_style(
                            location_button_x + 1.0,
                            row_y,
                            (location_button_w - 2.0).max(1.0),
                            18.0,
                            UiVisualStyle {
                                background: Some(if entry.selected {
                                    UiColor::rgba(0.23, 0.32, 0.47, 0.90)
                                } else {
                                    UiColor::rgba(0.13, 0.17, 0.24, 0.78)
                                }),
                                border_color: None,
                                border_width: 0.0,
                                corner_radius: 0.0,
                                clip: false,
                            },
                        ),
                    ));
                    retained.upsert(RetainedUiNode::new(
                        row_label_id,
                        UiWidget::Label(UiLabel {
                            text: UiTextValue::from(entry.label.clone()),
                            style: UiTextStyle {
                                color: if entry.selected {
                                    UiColor::rgba(0.96, 0.98, 1.0, 1.0)
                                } else {
                                    UiColor::rgba(0.84, 0.89, 0.96, 1.0)
                                },
                                font_size: 10.0,
                                align_h: UiTextAlign::Start,
                                align_v: UiTextAlign::Center,
                                wrap: false,
                            },
                        }),
                        absolute_style(
                            location_button_x + 7.0,
                            row_y,
                            (location_button_w - 14.0).max(20.0),
                            18.0,
                            UiVisualStyle::default(),
                        ),
                    ));
                    retained.upsert(RetainedUiNode::new(
                        row_hit_id,
                        hit_widget(),
                        absolute_style(
                            location_button_x + 1.0,
                            row_y,
                            (location_button_w - 2.0).max(1.0),
                            18.0,
                            UiVisualStyle::default(),
                        ),
                    ));
                    frame.actions.insert(
                        row_hit_id,
                        ContentBrowserPaneAction::SelectLocation(entry.path.clone()),
                    );

                    children.push(row_id);
                    children.push(row_label_id);
                    children.push(row_hit_id);
                }
            }
            header_cursor_y += dropdown_height + 4.0;
        }
    }

    if show_filter {
        let filter_label_id = root_id.child("filter-label");
        retained.upsert(RetainedUiNode::new(
            filter_label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("Filter:"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.72, 0.78, 0.87, 1.0),
                    font_size: 11.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                header_x,
                header_cursor_y,
                44.0,
                22.0,
                UiVisualStyle::default(),
            ),
        ));
        children.push(filter_label_id);

        let filter_field_id = root_id.child("filter-field");
        retained.upsert(RetainedUiNode::new(
            filter_field_id,
            UiWidget::TextField(UiTextField {
                text: UiTextValue::from(data.filter.clone()),
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
                focused: data.focused_field == Some(ContentBrowserPaneTextField::Filter),
                cursor: Some(
                    data.text_cursors
                        .get(&ContentBrowserPaneTextField::Filter)
                        .copied()
                        .unwrap_or_else(|| data.filter.chars().count())
                        .min(data.filter.chars().count()),
                ),
                selection: None,
                show_caret: data.focused_field == Some(ContentBrowserPaneTextField::Filter),
                selection_color: UiColor::rgba(0.34, 0.52, 0.84, 0.46),
                caret_color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
            }),
            absolute_style(
                header_x + 52.0,
                header_cursor_y,
                (header_width - 52.0).max(20.0),
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
            .insert(filter_field_id, ContentBrowserPaneTextField::Filter);
        children.push(filter_field_id);
        header_cursor_y += 28.0;
    }

    if show_slider {
        let tile_size = data.tile_size.clamp(64.0, 220.0);
        let tile_min = 64.0;
        let tile_max = 220.0;
        let tile_norm = ((tile_size - tile_min) / (tile_max - tile_min)).clamp(0.0, 1.0);

        let tile_label_id = root_id.child("tile-size-label");
        retained.upsert(RetainedUiNode::new(
            tile_label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("Tile Size"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.74, 0.80, 0.88, 1.0),
                    font_size: 10.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                header_x,
                header_cursor_y,
                56.0,
                18.0,
                UiVisualStyle::default(),
            ),
        ));
        children.push(tile_label_id);

        let slider_track_x = header_x + 60.0;
        let slider_track_w = (header_width * 0.42)
            .clamp(96.0, 186.0)
            .min((header_width - 116.0).max(24.0));
        let slider_track_id = root_id.child("tile-size-track");
        let slider_fill_id = slider_track_id.child("fill");
        let slider_knob_id = slider_track_id.child("knob");
        let slider_hit_id = slider_track_id.child("hit");
        let knob_w = 10.0f32.min(slider_track_w.max(1.0));
        let knob_x = (slider_track_x + tile_norm * (slider_track_w - knob_w)).max(slider_track_x);
        retained.upsert(RetainedUiNode::new(
            slider_track_id,
            UiWidget::Container,
            absolute_style(
                slider_track_x,
                header_cursor_y + 4.0,
                slider_track_w,
                10.0,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.16, 0.20, 0.28, 0.92)),
                    border_color: Some(UiColor::rgba(0.32, 0.39, 0.52, 0.92)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            slider_fill_id,
            UiWidget::Container,
            absolute_style(
                slider_track_x,
                header_cursor_y + 4.0,
                (slider_track_w * tile_norm).max(0.0),
                10.0,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.29, 0.41, 0.61, 0.96)),
                    border_color: None,
                    border_width: 0.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            slider_knob_id,
            UiWidget::Container,
            absolute_style(
                knob_x,
                header_cursor_y + 2.0,
                knob_w,
                14.0,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.84, 0.90, 1.0, 0.98)),
                    border_color: Some(UiColor::rgba(0.38, 0.46, 0.60, 0.95)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        retained.upsert(RetainedUiNode::new(
            slider_hit_id,
            hit_widget(),
            absolute_style(
                slider_track_x,
                header_cursor_y,
                slider_track_w,
                18.0,
                UiVisualStyle::default(),
            ),
        ));
        frame
            .actions
            .insert(slider_hit_id, ContentBrowserPaneAction::TileSizeSlider);
        children.push(slider_track_id);
        children.push(slider_fill_id);
        children.push(slider_knob_id);
        children.push(slider_hit_id);
        let tile_value_id = root_id.child("tile-size-value");
        retained.upsert(RetainedUiNode::new(
            tile_value_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(format!("{tile_size:.0}")),
                style: UiTextStyle {
                    color: UiColor::rgba(0.88, 0.93, 0.99, 1.0),
                    font_size: 10.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                slider_track_x + slider_track_w + 8.0,
                header_cursor_y,
                40.0,
                18.0,
                UiVisualStyle::default(),
            ),
        ));
        children.push(tile_value_id);
        header_cursor_y += 22.0;
    }

    if show_status && let Some(status) = data.status.as_ref() {
        let status_id = root_id.child("status");
        retained.upsert(RetainedUiNode::new(
            status_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(status.clone()),
                style: UiTextStyle {
                    color: UiColor::rgba(0.88, 0.78, 0.58, 1.0),
                    font_size: 10.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                header_x,
                header_cursor_y,
                header_width,
                18.0,
                UiVisualStyle::default(),
            ),
        ));
        children.push(status_id);
        header_cursor_y += 20.0;
    }

    let header_height = header_cursor_y.max(0.0).min((pane_height - 8.0).max(0.0));
    let content_y = (header_height + 6.0).clamp(0.0, (pane_height - 1.0).max(0.0));
    let content_height = (pane_height - content_y - 8.0).max(1.0);

    let sidebar_x = 8.0;
    let sidebar_visible = pane_width >= 460.0;
    let sidebar_gap = if sidebar_visible { 8.0 } else { 0.0 };
    let sidebar_width = if sidebar_visible {
        (pane_width * 0.24)
            .clamp(140.0, 280.0)
            .min((pane_width - 24.0).max(80.0))
    } else {
        0.0
    };
    let grid_x = sidebar_x + sidebar_width + sidebar_gap;
    let grid_width = (pane_width - grid_x - 8.0).max(1.0);

    if sidebar_visible {
        let sidebar_bg_id = root_id.child("sidebar-bg");
        retained.upsert(RetainedUiNode::new(
            sidebar_bg_id,
            UiWidget::Container,
            absolute_style(
                sidebar_x,
                content_y,
                sidebar_width,
                content_height,
                UiVisualStyle {
                    background: Some(UiColor::rgba(0.12, 0.15, 0.20, 0.88)),
                    border_color: Some(UiColor::rgba(0.25, 0.32, 0.40, 0.90)),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            ),
        ));
        children.push(sidebar_bg_id);
        frame
            .actions
            .insert(sidebar_bg_id, ContentBrowserPaneAction::GridSurface);

        let sidebar_heading_id = root_id.child("sidebar-heading");
        retained.upsert(RetainedUiNode::new(
            sidebar_heading_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("Folders"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.90, 0.94, 1.0, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                sidebar_x + 8.0,
                content_y + 4.0,
                sidebar_width - 16.0,
                18.0,
                UiVisualStyle::default(),
            ),
        ));
        children.push(sidebar_heading_id);
        frame
            .actions
            .insert(sidebar_heading_id, ContentBrowserPaneAction::GridSurface);

        if data.sidebar_entries.is_empty() {
            let empty_id = root_id.child("sidebar-empty");
            retained.upsert(RetainedUiNode::new(
                empty_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from("No folders"),
                    style: UiTextStyle {
                        color: UiColor::rgba(0.70, 0.75, 0.82, 1.0),
                        font_size: 11.0,
                        align_h: UiTextAlign::Start,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                absolute_style(
                    sidebar_x + 8.0,
                    content_y + 28.0,
                    sidebar_width - 16.0,
                    18.0,
                    UiVisualStyle::default(),
                ),
            ));
            children.push(empty_id);
        } else {
            let row_height = 19.0;
            let row_start_y = content_y + 26.0;
            let max_rows = ((content_height - 28.0) / row_height).max(1.0) as usize;
            for (index, entry) in data.sidebar_entries.iter().take(max_rows).enumerate() {
                let row_y = row_start_y + index as f32 * row_height;
                let row_id = root_id.child("sidebar").child(index as u64);
                let label_id = row_id.child("label");
                let hit_id = row_id.child("hit");
                let indent = 10.0 + entry.depth.min(10) as f32 * 12.0;
                let toggle_id = row_id.child("toggle");
                let toggle_hit_id = row_id.child("toggle-hit");
                let toggle_width = if entry.has_children { 12.0 } else { 0.0 };

                retained.upsert(RetainedUiNode::new(
                    row_id,
                    UiWidget::Container,
                    absolute_style(
                        sidebar_x + 4.0,
                        row_y,
                        sidebar_width - 8.0,
                        row_height - 1.0,
                        UiVisualStyle {
                            background: Some(if entry.selected {
                                UiColor::rgba(0.23, 0.32, 0.47, 0.92)
                            } else {
                                UiColor::rgba(0.14, 0.18, 0.25, 0.72)
                            }),
                            border_color: Some(if entry.selected {
                                UiColor::rgba(0.56, 0.72, 0.95, 0.95)
                            } else {
                                UiColor::rgba(0.24, 0.30, 0.38, 0.84)
                            }),
                            border_width: 1.0,
                            corner_radius: 0.0,
                            clip: false,
                        },
                    ),
                ));
                children.push(row_id);

                for guide_depth in 0..entry.depth {
                    let guide_id = row_id.child("guide").child(guide_depth as u64);
                    let guide_x = sidebar_x + 12.0 + guide_depth as f32 * 12.0;
                    retained.upsert(RetainedUiNode::new(
                        guide_id,
                        UiWidget::Container,
                        absolute_style(
                            guide_x,
                            row_y + 1.0,
                            1.0,
                            row_height - 2.0,
                            UiVisualStyle {
                                background: Some(UiColor::rgba(0.31, 0.38, 0.47, 0.76)),
                                border_color: None,
                                border_width: 0.0,
                                corner_radius: 0.0,
                                clip: false,
                            },
                        ),
                    ));
                    children.push(guide_id);
                }
                if entry.depth > 0 {
                    let branch_id = row_id.child("branch");
                    let branch_x = sidebar_x + 12.0 + (entry.depth.saturating_sub(1)) as f32 * 12.0;
                    retained.upsert(RetainedUiNode::new(
                        branch_id,
                        UiWidget::Container,
                        absolute_style(
                            branch_x,
                            row_y + ((row_height - 1.0) * 0.5),
                            9.0,
                            1.0,
                            UiVisualStyle {
                                background: Some(UiColor::rgba(0.31, 0.38, 0.47, 0.76)),
                                border_color: None,
                                border_width: 0.0,
                                corner_radius: 0.0,
                                clip: false,
                            },
                        ),
                    ));
                    children.push(branch_id);
                }

                if entry.has_children {
                    retained.upsert(RetainedUiNode::new(
                        toggle_id,
                        UiWidget::Label(UiLabel {
                            text: UiTextValue::from(if entry.expanded { "v" } else { ">" }),
                            style: UiTextStyle {
                                color: if entry.selected {
                                    UiColor::rgba(0.96, 0.98, 1.0, 1.0)
                                } else {
                                    UiColor::rgba(0.84, 0.89, 0.95, 1.0)
                                },
                                font_size: 10.0,
                                align_h: UiTextAlign::Center,
                                align_v: UiTextAlign::Center,
                                wrap: false,
                            },
                        }),
                        absolute_style(
                            sidebar_x + indent,
                            row_y,
                            12.0,
                            row_height - 1.0,
                            UiVisualStyle::default(),
                        ),
                    ));
                    retained.upsert(RetainedUiNode::new(
                        toggle_hit_id,
                        hit_widget(),
                        absolute_style(
                            sidebar_x + indent,
                            row_y,
                            12.0,
                            row_height - 1.0,
                            UiVisualStyle::default(),
                        ),
                    ));
                    frame.actions.insert(
                        toggle_hit_id,
                        ContentBrowserPaneAction::ToggleFolderExpanded(entry.path.clone()),
                    );
                    children.push(toggle_id);
                    children.push(toggle_hit_id);
                }
                retained.upsert(RetainedUiNode::new(
                    label_id,
                    UiWidget::Label(UiLabel {
                        text: UiTextValue::from(entry.label.clone()),
                        style: UiTextStyle {
                            color: if entry.selected {
                                UiColor::rgba(0.96, 0.98, 1.0, 1.0)
                            } else {
                                UiColor::rgba(0.84, 0.89, 0.95, 1.0)
                            },
                            font_size: 11.0,
                            align_h: UiTextAlign::Start,
                            align_v: UiTextAlign::Center,
                            wrap: false,
                        },
                    }),
                    absolute_style(
                        sidebar_x + indent + toggle_width + 4.0,
                        row_y,
                        (sidebar_width - indent - toggle_width - 16.0).max(10.0),
                        row_height - 1.0,
                        UiVisualStyle::default(),
                    ),
                ));
                retained.upsert(RetainedUiNode::new(
                    hit_id,
                    hit_widget(),
                    absolute_style(
                        sidebar_x + 4.0,
                        row_y,
                        sidebar_width - 8.0,
                        row_height - 1.0,
                        UiVisualStyle::default(),
                    ),
                ));
                frame.actions.insert(
                    hit_id,
                    ContentBrowserPaneAction::SelectFolder(entry.path.clone()),
                );
                frame.actions.insert(
                    row_id,
                    ContentBrowserPaneAction::SelectFolder(entry.path.clone()),
                );
                frame.actions.insert(
                    label_id,
                    ContentBrowserPaneAction::SelectFolder(entry.path.clone()),
                );
                children.push(label_id);
                children.push(hit_id);
            }
        }
    }

    let grid_bg_id = root_id.child("grid-bg");
    retained.upsert(RetainedUiNode::new(
        grid_bg_id,
        UiWidget::Container,
        absolute_style(
            grid_x,
            content_y,
            grid_width,
            content_height,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.11, 0.14, 0.19, 0.88)),
                border_color: Some(UiColor::rgba(0.25, 0.31, 0.39, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    children.push(grid_bg_id);
    frame
        .actions
        .insert(grid_bg_id, ContentBrowserPaneAction::GridSurface);

    let grid_content_y = content_y + 4.0;
    let grid_content_height = (content_height - 4.0).max(1.0);
    frame.grid_scroll_region = Some(UiRect {
        x: grid_x,
        y: grid_content_y,
        width: grid_width,
        height: grid_content_height,
    });
    let grid_clip_id = root_id.child("grid-clip");
    retained.upsert(RetainedUiNode::new(
        grid_clip_id,
        UiWidget::Container,
        absolute_style(
            grid_x,
            grid_content_y,
            grid_width,
            grid_content_height,
            UiVisualStyle {
                background: None,
                border_color: None,
                border_width: 0.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    children.push(grid_clip_id);
    frame
        .actions
        .insert(grid_clip_id, ContentBrowserPaneAction::GridSurface);

    let mut grid_children = Vec::new();
    let grid_surface_id = grid_clip_id.child("surface");
    retained.upsert(RetainedUiNode::new(
        grid_surface_id,
        hit_widget(),
        absolute_style(
            0.0,
            0.0,
            grid_width,
            grid_content_height,
            UiVisualStyle::default(),
        ),
    ));
    frame
        .actions
        .insert(grid_surface_id, ContentBrowserPaneAction::GridSurface);
    grid_children.push(grid_surface_id);

    if data.grid_entries.is_empty() {
        let empty_id = grid_clip_id.child("empty");
        retained.upsert(RetainedUiNode::new(
            empty_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("No assets"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.70, 0.75, 0.82, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(8.0, 4.0, grid_width - 16.0, 18.0, UiVisualStyle::default()),
        ));
        grid_children.push(empty_id);
    } else {
        let tile_gap = 10.0;
        let label_height = 34.0;
        let visible_height = grid_content_height.max(1.0);
        let provisional_side = data
            .tile_size
            .clamp(64.0, 220.0)
            .min((grid_width - 16.0).max(48.0));
        let provisional_columns = ((grid_width - 16.0 + tile_gap) / (provisional_side + tile_gap))
            .floor()
            .max(1.0) as usize;
        let provisional_span = provisional_side + label_height + tile_gap;
        let provisional_rows = data.grid_entries.len().div_ceil(provisional_columns);
        let provisional_total_height = provisional_rows as f32 * provisional_span;
        let needs_scrollbar = provisional_total_height > visible_height + f32::EPSILON;

        let tile_region_width = if needs_scrollbar {
            (grid_width - 12.0).max(48.0)
        } else {
            grid_width
        };
        let tile_side = data
            .tile_size
            .clamp(64.0, 220.0)
            .min((tile_region_width - 16.0).max(48.0));
        let columns = ((tile_region_width - 16.0 + tile_gap) / (tile_side + tile_gap))
            .floor()
            .max(1.0) as usize;
        let tile_span = tile_side + label_height + tile_gap;

        let total_rows = data.grid_entries.len().div_ceil(columns);
        let total_height = total_rows as f32 * tile_span;
        let max_scroll = (total_height - visible_height).max(0.0);
        let scroll = data.grid_scroll.clamp(0.0, max_scroll);
        frame.grid_scroll_max = max_scroll;

        for (index, entry) in data.grid_entries.iter().enumerate() {
            let row = index / columns;
            let col = index % columns;
            let tile_x = 8.0 + col as f32 * (tile_side + tile_gap);
            let tile_y = 2.0 + row as f32 * tile_span - scroll;
            if tile_y + tile_side + label_height < -2.0 {
                continue;
            }
            if tile_y > grid_content_height + 2.0 {
                break;
            }

            let tile_id = grid_clip_id.child("tile").child(index as u64);
            let thumb_id = tile_id.child("thumb");
            let tag_id = tile_id.child("tag");
            let label_id = tile_id.child("label");
            let hit_id = tile_id.child("hit");

            retained.upsert(RetainedUiNode::new(
                tile_id,
                UiWidget::Container,
                absolute_style(
                    tile_x,
                    tile_y,
                    tile_side,
                    tile_side + label_height,
                    UiVisualStyle {
                        background: Some(if entry.selected {
                            UiColor::rgba(0.23, 0.32, 0.47, 0.92)
                        } else {
                            UiColor::rgba(0.14, 0.18, 0.24, 0.82)
                        }),
                        border_color: Some(if entry.selected {
                            UiColor::rgba(0.56, 0.72, 0.95, 0.95)
                        } else {
                            UiColor::rgba(0.24, 0.30, 0.38, 0.84)
                        }),
                        border_width: 1.0,
                        corner_radius: 0.0,
                        clip: false,
                    },
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                thumb_id,
                UiWidget::Container,
                absolute_style(
                    tile_x + 8.0,
                    tile_y + 8.0,
                    tile_side - 16.0,
                    tile_side - 24.0,
                    UiVisualStyle {
                        background: Some(asset_thumbnail_color(entry)),
                        border_color: Some(UiColor::rgba(0.08, 0.10, 0.14, 0.90)),
                        border_width: 1.0,
                        corner_radius: 0.0,
                        clip: false,
                    },
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                tag_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from(asset_thumbnail_tag(entry)),
                    style: UiTextStyle {
                        color: UiColor::rgba(0.97, 0.99, 1.0, 1.0),
                        font_size: 10.0,
                        align_h: UiTextAlign::Center,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                absolute_style(
                    tile_x + 8.0,
                    tile_y + 8.0,
                    tile_side - 16.0,
                    tile_side - 24.0,
                    UiVisualStyle::default(),
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                label_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from(entry.label.clone()),
                    style: UiTextStyle {
                        color: UiColor::rgba(0.88, 0.92, 0.98, 1.0),
                        font_size: 10.0,
                        align_h: UiTextAlign::Center,
                        align_v: UiTextAlign::Center,
                        wrap: true,
                    },
                }),
                absolute_style(
                    tile_x + 4.0,
                    tile_y + tile_side - 8.0,
                    tile_side - 8.0,
                    label_height + 10.0,
                    UiVisualStyle::default(),
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                hit_id,
                hit_widget(),
                absolute_style(
                    tile_x,
                    tile_y,
                    tile_side,
                    tile_side + label_height,
                    UiVisualStyle::default(),
                ),
            ));

            frame.actions.insert(
                hit_id,
                ContentBrowserPaneAction::SelectEntry {
                    path: entry.path.clone(),
                    is_dir: entry.is_dir,
                    index,
                },
            );
            frame.actions.insert(
                tile_id,
                ContentBrowserPaneAction::SelectEntry {
                    path: entry.path.clone(),
                    is_dir: entry.is_dir,
                    index,
                },
            );
            frame.actions.insert(
                thumb_id,
                ContentBrowserPaneAction::SelectEntry {
                    path: entry.path.clone(),
                    is_dir: entry.is_dir,
                    index,
                },
            );
            frame.actions.insert(
                tag_id,
                ContentBrowserPaneAction::SelectEntry {
                    path: entry.path.clone(),
                    is_dir: entry.is_dir,
                    index,
                },
            );
            frame.actions.insert(
                label_id,
                ContentBrowserPaneAction::SelectEntry {
                    path: entry.path.clone(),
                    is_dir: entry.is_dir,
                    index,
                },
            );
            grid_children.push(tile_id);
            grid_children.push(thumb_id);
            grid_children.push(tag_id);
            grid_children.push(label_id);
            grid_children.push(hit_id);
        }

        if max_scroll > f32::EPSILON {
            let track_id = grid_clip_id.child("scrollbar-track");
            let thumb_id = grid_clip_id.child("scrollbar-thumb");
            let hit_id = grid_clip_id.child("scrollbar-hit");
            let track_w = 8.0;
            let track_x = (grid_width - track_w - 2.0).max(0.0);
            let track_h = visible_height.max(1.0);
            let thumb_h = ((visible_height / total_height.max(1.0)) * track_h).clamp(18.0, track_h);
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
                .insert(hit_id, ContentBrowserPaneAction::GridScrollbar);
            grid_children.push(track_id);
            grid_children.push(thumb_id);
            grid_children.push(hit_id);
        }
    }

    for id in grid_children.iter().copied() {
        frame
            .actions
            .entry(id)
            .or_insert(ContentBrowserPaneAction::GridSurface);
    }
    for id in children.iter().copied() {
        frame
            .actions
            .entry(id)
            .or_insert(ContentBrowserPaneAction::GridSurface);
    }

    retained.set_children(grid_clip_id, grid_children);

    retained.set_children(root_id, children);
    frame
}

fn asset_thumbnail_tag(entry: &ContentBrowserPaneEntry) -> &'static str {
    if entry.is_dir {
        return "DIR";
    }

    let ext = entry
        .path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    match ext.as_deref() {
        Some("png" | "jpg" | "jpeg" | "tga" | "bmp" | "gif" | "webp") => "TEX",
        Some("fbx" | "obj" | "gltf" | "glb" | "blend") => "MESH",
        Some("scene" | "scn") => "SCN",
        Some("ron") => "RON",
        Some("lua" | "vs" | "visual" | "rs") => "SCR",
        Some("wav" | "ogg" | "mp3" | "flac") => "AUD",
        Some("mat" | "material") => "MAT",
        _ => "FILE",
    }
}

fn asset_thumbnail_color(entry: &ContentBrowserPaneEntry) -> UiColor {
    if entry.is_dir {
        return UiColor::rgba(0.43, 0.34, 0.14, 0.95);
    }

    let ext = entry
        .path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    match ext.as_deref() {
        Some("png" | "jpg" | "jpeg" | "tga" | "bmp" | "gif" | "webp") => {
            UiColor::rgba(0.24, 0.48, 0.72, 0.95)
        }
        Some("fbx" | "obj" | "gltf" | "glb" | "blend") => UiColor::rgba(0.48, 0.36, 0.22, 0.95),
        Some("scene" | "scn") => UiColor::rgba(0.29, 0.52, 0.44, 0.95),
        Some("ron") => UiColor::rgba(0.39, 0.34, 0.56, 0.95),
        Some("lua" | "vs" | "visual" | "rs") => UiColor::rgba(0.43, 0.34, 0.58, 0.95),
        Some("wav" | "ogg" | "mp3" | "flac") => UiColor::rgba(0.56, 0.30, 0.26, 0.95),
        Some("mat" | "material") => UiColor::rgba(0.20, 0.44, 0.56, 0.95),
        _ => UiColor::rgba(0.26, 0.31, 0.38, 0.95),
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

fn hit_widget() -> UiWidget {
    UiWidget::HitBox
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
