use std::{collections::HashMap, path::PathBuf};

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiLabel, UiLayoutBuilder,
    UiPositionType, UiRect, UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle,
    UiWidget,
};

#[derive(Clone, Debug, Default)]
pub struct ProjectPaneData {
    pub project_loaded: bool,
    pub project_name: String,
    pub root_path: Option<String>,
    pub assets_root: Option<String>,
    pub open_project_path: String,
    pub create_project_name: String,
    pub create_project_location: String,
    pub recent_projects: Vec<String>,
    pub status: Option<String>,
    pub focused_field: Option<ProjectPaneTextField>,
    pub text_cursors: HashMap<ProjectPaneTextField, usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProjectPaneTextField {
    OpenPath,
    CreateName,
    CreateLocation,
}

#[derive(Clone, Debug)]
pub enum ProjectPaneAction {
    BrowseOpenProject,
    OpenInputPath,
    OpenRecentProject(PathBuf),
    BrowseCreateLocation,
    CreateFromInputs,
    RefreshAssets,
    ReloadProjectConfig,
    CloseProject,
}

#[derive(Clone, Debug, Default)]
pub struct ProjectPaneFrame {
    pub actions: HashMap<UiId, ProjectPaneAction>,
    pub text_fields: HashMap<UiId, ProjectPaneTextField>,
}

pub fn build_project_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    viewport: UiRect,
    data: &ProjectPaneData,
) -> ProjectPaneFrame {
    let mut frame = ProjectPaneFrame::default();
    let mut panel_children = Vec::new();
    let pane_width = viewport.width.max(1.0);
    let pane_height = viewport.height.max(1.0);
    let panel_max_width = (pane_width - 16.0).max(140.0);
    let panel_max_height = (pane_height - 16.0).max(120.0);
    let panel_width = if panel_max_width <= 420.0 {
        panel_max_width
    } else {
        panel_max_width.clamp(420.0, 860.0)
    };
    let panel_height = if panel_max_height <= 420.0 {
        panel_max_height
    } else {
        panel_max_height.clamp(420.0, 560.0)
    };
    let content_width = (panel_width - 24.0)
        .max(120.0)
        .min((panel_width - 12.0).max(1.0));
    let content_x = ((panel_width - content_width) * 0.5).max(8.0);

    let background_id = root_id.child("background");
    let panel_id = root_id.child("panel");
    retained.upsert(RetainedUiNode::new(
        root_id,
        UiWidget::Container,
        centered_fill_style(UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        background_id,
        UiWidget::Container,
        absolute_fill_style(UiVisualStyle {
            background: Some(UiColor::rgba(0.10, 0.12, 0.16, 0.82)),
            border_color: Some(UiColor::rgba(0.28, 0.33, 0.42, 0.90)),
            border_width: 1.0,
            corner_radius: 0.0,
            clip: true,
        }),
    ));
    retained.upsert(RetainedUiNode::new(
        panel_id,
        UiWidget::Container,
        UiStyle {
            layout: UiLayoutBuilder::new()
                .width(UiDimension::points(panel_width))
                .height(UiDimension::points(panel_height))
                .build(),
            visual: UiVisualStyle {
                background: Some(UiColor::rgba(0.11, 0.14, 0.18, 0.80)),
                border_color: Some(UiColor::rgba(0.30, 0.36, 0.46, 0.88)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        },
    ));

    if !data.project_loaded {
        let create_section_top_max = (panel_height - 168.0).max(96.0);
        let create_section_top = if create_section_top_max <= 248.0 {
            create_section_top_max
        } else {
            (panel_height * 0.62).clamp(248.0, create_section_top_max)
        };
        let recent_start_y = 118.0;
        let recent_max_rows = ((create_section_top - recent_start_y - 8.0) / 24.0)
            .floor()
            .max(1.0) as usize;

        add_heading(
            retained,
            &mut panel_children,
            panel_id,
            "open-heading",
            "Open Project",
            content_x,
            8.0,
            content_width,
        );
        add_text_field(
            retained,
            &mut panel_children,
            &mut frame,
            panel_id,
            "open-path-input",
            "Path:",
            &data.open_project_path,
            ProjectPaneTextField::OpenPath,
            data.focused_field,
            data.text_cursors
                .get(&ProjectPaneTextField::OpenPath)
                .copied(),
            content_x,
            30.0,
            content_width,
        );

        let button_gap = 8.0;
        let browse_width = 128.0;
        let open_width = 96.0;
        let browse_hit = add_action_button(
            retained,
            &mut panel_children,
            panel_id,
            "browse-open",
            "Browse...",
            content_x,
            58.0,
            browse_width,
        );
        frame
            .actions
            .insert(browse_hit, ProjectPaneAction::BrowseOpenProject);

        let open_hit = add_action_button(
            retained,
            &mut panel_children,
            panel_id,
            "open-path",
            "Open",
            content_x + browse_width + button_gap,
            58.0,
            open_width,
        );
        frame
            .actions
            .insert(open_hit, ProjectPaneAction::OpenInputPath);

        add_heading(
            retained,
            &mut panel_children,
            panel_id,
            "recent-heading",
            "Recent Projects",
            content_x,
            94.0,
            content_width,
        );
        if data.recent_projects.is_empty() {
            add_line(
                retained,
                &mut panel_children,
                panel_id,
                "recent-empty",
                "No recent projects yet",
                content_x,
                118.0,
                content_width,
                UiColor::rgba(0.71, 0.76, 0.84, 1.0),
            );
        } else {
            for (index, path) in data
                .recent_projects
                .iter()
                .take(recent_max_rows.max(1))
                .enumerate()
            {
                let y = 118.0 + index as f32 * 24.0;
                let hit = add_action_button(
                    retained,
                    &mut panel_children,
                    panel_id,
                    &format!("recent-{index}"),
                    path,
                    content_x,
                    y,
                    content_width,
                );
                frame.actions.insert(
                    hit,
                    ProjectPaneAction::OpenRecentProject(PathBuf::from(path)),
                );
            }
        }

        add_heading(
            retained,
            &mut panel_children,
            panel_id,
            "create-heading",
            "Create Project",
            content_x,
            create_section_top,
            content_width,
        );
        add_text_field(
            retained,
            &mut panel_children,
            &mut frame,
            panel_id,
            "create-name-input",
            "Name:",
            &data.create_project_name,
            ProjectPaneTextField::CreateName,
            data.focused_field,
            data.text_cursors
                .get(&ProjectPaneTextField::CreateName)
                .copied(),
            content_x,
            create_section_top + 24.0,
            content_width,
        );
        add_text_field(
            retained,
            &mut panel_children,
            &mut frame,
            panel_id,
            "create-location-input",
            "Location:",
            &data.create_project_location,
            ProjectPaneTextField::CreateLocation,
            data.focused_field,
            data.text_cursors
                .get(&ProjectPaneTextField::CreateLocation)
                .copied(),
            content_x,
            create_section_top + 56.0,
            content_width,
        );

        let create_button_y = (create_section_top + 88.0).min(panel_height - 56.0);
        let browse_create_hit = add_action_button(
            retained,
            &mut panel_children,
            panel_id,
            "browse-create",
            "Browse Location...",
            content_x,
            create_button_y,
            160.0,
        );
        frame
            .actions
            .insert(browse_create_hit, ProjectPaneAction::BrowseCreateLocation);

        let create_hit = add_action_button(
            retained,
            &mut panel_children,
            panel_id,
            "create-project",
            "Create Project",
            content_x + 168.0,
            create_button_y,
            140.0,
        );
        frame
            .actions
            .insert(create_hit, ProjectPaneAction::CreateFromInputs);
    } else {
        let project_name = if data.project_name.trim().is_empty() {
            "<unknown>"
        } else {
            data.project_name.trim()
        };

        add_heading(
            retained,
            &mut panel_children,
            panel_id,
            "project-heading",
            &format!("Project: {project_name}"),
            content_x,
            8.0,
            content_width,
        );
        add_line(
            retained,
            &mut panel_children,
            panel_id,
            "project-root",
            &format!("Root: {}", data.root_path.as_deref().unwrap_or("<none>")),
            content_x,
            34.0,
            content_width,
            UiColor::rgba(0.84, 0.88, 0.95, 1.0),
        );
        add_line(
            retained,
            &mut panel_children,
            panel_id,
            "project-assets",
            &format!(
                "Assets: {}",
                data.assets_root.as_deref().unwrap_or("<none>")
            ),
            content_x,
            56.0,
            content_width,
            UiColor::rgba(0.78, 0.83, 0.91, 1.0),
        );

        let refresh_hit = add_action_button(
            retained,
            &mut panel_children,
            panel_id,
            "refresh-assets",
            "Refresh Assets",
            content_x,
            84.0,
            126.0,
        );
        frame
            .actions
            .insert(refresh_hit, ProjectPaneAction::RefreshAssets);

        let reload_hit = add_action_button(
            retained,
            &mut panel_children,
            panel_id,
            "reload-config",
            "Reload Config",
            content_x + 134.0,
            84.0,
            126.0,
        );
        frame
            .actions
            .insert(reload_hit, ProjectPaneAction::ReloadProjectConfig);

        let close_hit = add_action_button(
            retained,
            &mut panel_children,
            panel_id,
            "close-project",
            "Close Project",
            content_x + 268.0,
            84.0,
            126.0,
        );
        frame
            .actions
            .insert(close_hit, ProjectPaneAction::CloseProject);
    }

    if let Some(status) = data.status.as_ref().map(|value| value.trim())
        && !status.is_empty()
    {
        add_line(
            retained,
            &mut panel_children,
            panel_id,
            "status",
            status,
            content_x,
            panel_height - 30.0,
            content_width,
            UiColor::rgba(0.93, 0.79, 0.46, 1.0),
        );
    }

    retained.set_children(panel_id, panel_children);
    retained.set_children(root_id, [background_id, panel_id]);
    frame
}

fn add_heading(
    retained: &mut RetainedUi,
    children: &mut Vec<UiId>,
    root_id: UiId,
    key: &str,
    text: &str,
    x: f32,
    y: f32,
    width: f32,
) {
    let id = root_id.child(key);
    retained.upsert(RetainedUiNode::new(
        id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(text.to_string()),
            style: UiTextStyle {
                color: UiColor::rgba(0.96, 0.97, 1.0, 1.0),
                font_size: 14.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(x, y, width, 22.0, UiVisualStyle::default()),
    ));
    children.push(id);
}

fn add_line(
    retained: &mut RetainedUi,
    children: &mut Vec<UiId>,
    root_id: UiId,
    key: &str,
    text: &str,
    x: f32,
    y: f32,
    width: f32,
    color: UiColor,
) {
    let id = root_id.child(key);
    retained.upsert(RetainedUiNode::new(
        id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(text.to_string()),
            style: UiTextStyle {
                color,
                font_size: 12.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: true,
            },
        }),
        absolute_style(x, y, width, 20.0, UiVisualStyle::default()),
    ));
    children.push(id);
}

fn add_action_button(
    retained: &mut RetainedUi,
    children: &mut Vec<UiId>,
    root_id: UiId,
    key: &str,
    text: &str,
    x: f32,
    y: f32,
    width: f32,
) -> UiId {
    let button_id = root_id.child("action").child(key);
    retained.upsert(RetainedUiNode::new(
        button_id,
        UiWidget::Button(helmer_ui::UiButton {
            text: UiTextValue::from(text.to_string()),
            variant: helmer_ui::UiButtonVariant::Secondary,
            enabled: true,
            style: UiTextStyle {
                color: UiColor::rgba(0.93, 0.96, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            x,
            y,
            width,
            22.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.20, 0.27, 0.36, 0.90)),
                border_color: Some(UiColor::rgba(0.30, 0.40, 0.53, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    children.push(button_id);
    button_id
}

fn add_text_field(
    retained: &mut RetainedUi,
    children: &mut Vec<UiId>,
    frame: &mut ProjectPaneFrame,
    root_id: UiId,
    key: &str,
    label: &str,
    value: &str,
    field: ProjectPaneTextField,
    focused_field: Option<ProjectPaneTextField>,
    text_cursor: Option<usize>,
    x: f32,
    y: f32,
    width: f32,
) {
    let label_id = root_id.child(key).child("label");
    let field_id = root_id.child(key).child("field");
    retained.upsert(RetainedUiNode::new(
        label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(label.to_string()),
            style: UiTextStyle {
                color: UiColor::rgba(0.88, 0.91, 0.97, 1.0),
                font_size: 12.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(x, y, 86.0, 24.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        field_id,
        UiWidget::TextField(helmer_ui::UiTextField {
            text: UiTextValue::from(value.to_string()),
            suffix: None,
            scroll_x: 0.0,
            style: UiTextStyle {
                color: UiColor::rgba(0.95, 0.97, 1.0, 1.0),
                font_size: 12.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
            enabled: true,
            focused: focused_field == Some(field),
            cursor: Some(
                text_cursor
                    .unwrap_or_else(|| value.chars().count())
                    .min(value.chars().count()),
            ),
            selection: None,
            show_caret: focused_field == Some(field),
            selection_color: UiColor::rgba(0.34, 0.52, 0.84, 0.46),
            caret_color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
        }),
        absolute_style(
            x + 84.0,
            y,
            (width - 84.0).max(120.0),
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
    children.push(label_id);
    children.push(field_id);
    frame.text_fields.insert(field_id, field);
}

fn centered_fill_style(visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .width(UiDimension::percent(1.0))
            .height(UiDimension::percent(1.0))
            .align_items(helmer_ui::UiAlignItems::Center)
            .justify_content(helmer_ui::UiJustifyContent::Center)
            .build(),
        visual,
    }
}

fn absolute_fill_style(visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .position_type(UiPositionType::Absolute)
            .left(UiDimension::points(0.0))
            .top(UiDimension::points(0.0))
            .width(UiDimension::percent(1.0))
            .height(UiDimension::percent(1.0))
            .build(),
        visual,
    }
}

fn absolute_style(x: f32, y: f32, width: f32, height: f32, visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .position_type(UiPositionType::Absolute)
            .left(UiDimension::points(x))
            .top(UiDimension::points(y))
            .width(UiDimension::points(width.max(0.0)))
            .height(UiDimension::points(height.max(0.0)))
            .build(),
        visual,
    }
}
