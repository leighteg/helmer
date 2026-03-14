use helmer_becs::ecs::entity::Entity;
use std::collections::{HashMap, HashSet};

use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiLabel, UiLayoutBuilder, UiRect,
    UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle, UiWidget,
};

#[derive(Clone, Debug)]
pub struct HierarchyPaneEntry {
    pub entity: Entity,
    pub label: String,
    pub depth: usize,
    pub has_children: bool,
    pub expanded: bool,
}

#[derive(Clone, Debug, Default)]
pub struct HierarchyPaneData {
    pub selected: Option<Entity>,
    pub entries: Vec<HierarchyPaneEntry>,
    pub scroll: f32,
}

#[derive(Clone, Debug, Default)]
pub struct HierarchyPaneFrame {
    pub click_targets: HashMap<UiId, Entity>,
    pub actions: HashMap<UiId, HierarchyPaneAction>,
    pub drop_surface_hits: HashSet<UiId>,
    pub scroll_max: f32,
}

#[derive(Clone, Debug)]
pub enum HierarchyPaneAction {
    AddEntity,
    ToggleExpanded(Entity),
    ListSurface,
}

pub fn build_hierarchy_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    viewport: UiRect,
    data: &HierarchyPaneData,
) -> HierarchyPaneFrame {
    let mut frame = HierarchyPaneFrame::default();
    let pane_width = viewport.width.max(160.0);
    let pane_height = viewport.height.max(120.0);
    let content_width = (pane_width - 12.0).max(64.0);
    let background_id = root_id.child("background");
    let add_row_id = root_id.child("add-row");
    let add_label_id = add_row_id.child("label");
    let add_hit_id = add_row_id.child("hit");
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
            background: Some(UiColor::rgba(0.10, 0.12, 0.15, 0.82)),
            border_color: Some(UiColor::rgba(0.25, 0.30, 0.37, 0.9)),
            border_width: 1.0,
            corner_radius: 0.0,
            clip: true,
        }),
    ));
    retained.upsert(RetainedUiNode::new(
        add_row_id,
        UiWidget::Button(helmer_ui::UiButton {
            text: UiTextValue::from("Add"),
            variant: helmer_ui::UiButtonVariant::Secondary,
            enabled: true,
            style: UiTextStyle {
                color: UiColor::rgba(0.95, 0.98, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            8.0,
            8.0,
            72.0,
            22.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.20, 0.27, 0.36, 0.92)),
                border_color: Some(UiColor::rgba(0.36, 0.47, 0.62, 0.95)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        add_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Add"),
            style: UiTextStyle {
                color: UiColor::rgba(0.95, 0.98, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(8.0, 8.0, 72.0, 22.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        add_hit_id,
        UiWidget::HitBox,
        absolute_style(8.0, 8.0, 72.0, 22.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        surface_hit_id,
        UiWidget::HitBox,
        fill_style(UiVisualStyle::default()),
    ));
    frame
        .actions
        .insert(add_hit_id, HierarchyPaneAction::AddEntity);
    frame.drop_surface_hits.insert(surface_hit_id);

    let mut children = vec![
        background_id,
        surface_hit_id,
        add_row_id,
        add_label_id,
        add_hit_id,
    ];
    let row_height = 22.0;
    let row_start_y = 34.0;
    let list_height = (pane_height - row_start_y - 8.0).max(1.0);

    if data.entries.is_empty() {
        let empty_id = root_id.child("empty");
        retained.upsert(RetainedUiNode::new(
            empty_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("No entities"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.68, 0.73, 0.80, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                10.0,
                row_start_y,
                content_width,
                row_height,
                UiVisualStyle::default(),
            ),
        ));
        children.push(empty_id);
        retained.set_children(root_id, children);
        return frame;
    }

    let list_surface_id = root_id.child("list-surface");
    retained.upsert(RetainedUiNode::new(
        list_surface_id,
        UiWidget::HitBox,
        absolute_style(
            6.0,
            row_start_y,
            content_width,
            list_height,
            UiVisualStyle::default(),
        ),
    ));
    frame
        .actions
        .insert(list_surface_id, HierarchyPaneAction::ListSurface);
    children.push(list_surface_id);

    let total_height = data.entries.len() as f32 * row_height;
    let max_scroll = (total_height - list_height).max(0.0);
    let scroll = data.scroll.clamp(0.0, max_scroll);
    frame.scroll_max = max_scroll;

    for (index, entry) in data.entries.iter().enumerate() {
        let row_y = row_start_y + index as f32 * row_height - scroll;
        if row_y + row_height < row_start_y - 1.0 {
            continue;
        }
        if row_y > row_start_y + list_height + 1.0 {
            break;
        }
        let row_id = root_id.child("row").child(index as u64);
        let label_id = row_id.child("label");
        let hit_id = row_id.child("hit");
        let toggle_id = row_id.child("toggle");
        let toggle_hit_id = row_id.child("toggle-hit");

        let selected = data.selected == Some(entry.entity);
        let indent = 10.0 + entry.depth.min(12) as f32 * 16.0;
        retained.upsert(RetainedUiNode::new(
            row_id,
            UiWidget::Container,
            absolute_style(
                6.0,
                row_y,
                content_width,
                row_height - 2.0,
                UiVisualStyle {
                    background: Some(if selected {
                        UiColor::rgba(0.22, 0.30, 0.45, 0.92)
                    } else {
                        UiColor::rgba(0.13, 0.16, 0.21, 0.65)
                    }),
                    border_color: Some(if selected {
                        UiColor::rgba(0.52, 0.68, 0.92, 0.95)
                    } else {
                        UiColor::rgba(0.22, 0.26, 0.33, 0.70)
                    }),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));

        if entry.has_children {
            let glyph = if entry.expanded { "v" } else { ">" };
            retained.upsert(RetainedUiNode::new(
                toggle_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from(glyph),
                    style: UiTextStyle {
                        color: UiColor::rgba(0.80, 0.88, 0.99, 1.0),
                        font_size: 11.0,
                        align_h: UiTextAlign::Center,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                absolute_style(
                    indent,
                    row_y + 1.0,
                    12.0,
                    row_height - 4.0,
                    UiVisualStyle::default(),
                ),
            ));
            retained.upsert(RetainedUiNode::new(
                toggle_hit_id,
                UiWidget::HitBox,
                absolute_style(
                    indent,
                    row_y + 1.0,
                    12.0,
                    row_height - 4.0,
                    UiVisualStyle::default(),
                ),
            ));
            frame.actions.insert(
                toggle_hit_id,
                HierarchyPaneAction::ToggleExpanded(entry.entity),
            );
        }

        retained.upsert(RetainedUiNode::new(
            label_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(entry.label.clone()),
                style: UiTextStyle {
                    color: if selected {
                        UiColor::rgba(0.96, 0.98, 1.0, 1.0)
                    } else {
                        UiColor::rgba(0.86, 0.90, 0.95, 1.0)
                    },
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                indent + 16.0,
                row_y + 1.0,
                (content_width - indent - 24.0).max(40.0),
                row_height - 4.0,
                UiVisualStyle::default(),
            ),
        ));

        retained.upsert(RetainedUiNode::new(
            hit_id,
            UiWidget::HitBox,
            absolute_style(
                (indent + if entry.has_children { 14.0 } else { 4.0 }).max(6.0),
                row_y,
                (content_width - (indent + if entry.has_children { 14.0 } else { 4.0 })).max(12.0),
                row_height - 2.0,
                UiVisualStyle::default(),
            ),
        ));

        frame.click_targets.insert(hit_id, entry.entity);
        frame.click_targets.insert(label_id, entry.entity);
        frame.click_targets.insert(row_id, entry.entity);
        children.push(row_id);
        for guide_depth in 0..entry.depth {
            let guide_id = row_id.child("guide").child(guide_depth as u64);
            let guide_x = 14.0 + guide_depth as f32 * 16.0;
            retained.upsert(RetainedUiNode::new(
                guide_id,
                UiWidget::Container,
                absolute_style(
                    guide_x,
                    row_y + 1.0,
                    1.0,
                    row_height - 4.0,
                    UiVisualStyle {
                        background: Some(UiColor::rgba(0.32, 0.39, 0.48, 0.78)),
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
            let branch_start_x = 14.0 + (entry.depth.saturating_sub(1)) as f32 * 16.0;
            retained.upsert(RetainedUiNode::new(
                branch_id,
                UiWidget::Container,
                absolute_style(
                    branch_start_x,
                    row_y + (row_height * 0.5),
                    11.0,
                    1.0,
                    UiVisualStyle {
                        background: Some(UiColor::rgba(0.32, 0.39, 0.48, 0.78)),
                        border_color: None,
                        border_width: 0.0,
                        corner_radius: 0.0,
                        clip: false,
                    },
                ),
            ));
            children.push(branch_id);
        }
        children.push(label_id);
        children.push(hit_id);
        if entry.has_children {
            children.push(toggle_id);
            children.push(toggle_hit_id);
        }
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
