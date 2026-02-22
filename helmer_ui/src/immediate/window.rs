use glam::Vec2;

use crate::{
    IntoUiId, UiAlignItems, UiButtonVariant, UiDimension, UiInsets, UiLayoutStyle, UiNode,
    UiPositionType, UiStyle, UiVisualStyle, UiWidget,
};

use super::{UiContext, layout::base_column_layout};

#[derive(Clone, Debug)]
pub struct UiWindowOptions {
    pub title: Option<String>,
    pub position: Vec2,
    pub size: Vec2,
    pub movable: bool,
    pub resizable: bool,
    pub closable: bool,
    pub collapsible: bool,
    pub collapsed: bool,
    pub min_size: Vec2,
    pub title_bar_height: f32,
    pub resize_handle_size: f32,
    pub scroll_y: f32,
    pub max_scroll_y: f32,
}

impl Default for UiWindowOptions {
    fn default() -> Self {
        Self {
            title: None,
            position: Vec2::new(16.0, 16.0),
            size: Vec2::new(420.0, 520.0),
            movable: true,
            resizable: true,
            closable: true,
            collapsible: true,
            collapsed: false,
            min_size: Vec2::new(220.0, 140.0),
            title_bar_height: 30.0,
            resize_handle_size: 14.0,
            scroll_y: 0.0,
            max_scroll_y: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UiWindowResponse {
    pub collapsed: bool,
    pub close_requested: bool,
    pub content_viewport_id: Option<crate::UiId>,
    pub content_inner_id: Option<crate::UiId>,
    pub content_extent_id: Option<crate::UiId>,
}

impl UiContext {
    pub fn window<K, F>(&mut self, key: K, options: UiWindowOptions, build: F) -> UiWindowResponse
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        let id = self.derive_id(key);
        let min_size = Vec2::new(options.min_size.x.max(1.0), options.min_size.y.max(1.0));
        let expanded_size = Vec2::new(
            options.size.x.max(min_size.x),
            options.size.y.max(min_size.y),
        );
        let title = options.title.clone();
        let has_title_bar = title.is_some();
        let title_bar_height = options.title_bar_height.max(18.0);
        let mut collapsed = options.collapsed && options.collapsible;
        let mut close_requested = false;
        let mut content_viewport_id = None;
        let mut content_inner_id = None;
        let mut content_extent_id = None;
        let collapsed_height = (title_bar_height + self.theme.border_width * 2.0)
            .max(min_size.y.min(title_bar_height));
        let size = if collapsed {
            Vec2::new(expanded_size.x, collapsed_height)
        } else {
            expanded_size
        };
        let mut layout = base_column_layout(self.theme.spacing);
        layout.position_type = UiPositionType::Absolute;
        layout.left = UiDimension::points(options.position.x);
        layout.top = UiDimension::points(options.position.y);
        layout.width = UiDimension::points(size.x);
        layout.height = UiDimension::points(size.y);
        layout.gap = Vec2::ZERO;

        let style = UiStyle {
            layout,
            visual: UiVisualStyle {
                background: Some(self.theme.panel_background),
                border_color: Some(self.theme.panel_border),
                border_width: self.theme.border_width,
                corner_radius: self.theme.corner_radius,
                clip: true,
            },
        };

        self.with_node(UiNode::new(id, UiWidget::Container, style), |ui| {
            if let Some(title) = title {
                let mut title_layout = UiLayoutStyle::default();
                title_layout.flex_direction = crate::UiFlexDirection::Row;
                title_layout.align_items = UiAlignItems::Center;
                title_layout.height = UiDimension::points(title_bar_height);
                title_layout.gap = Vec2::new((ui.theme.item_padding * 0.4).max(3.0), 0.0);
                title_layout.padding = UiInsets {
                    left: ui.theme.window_padding,
                    right: ui.theme.window_padding,
                    top: 0.0,
                    bottom: 0.0,
                };
                ui.with_node(
                    UiNode::new(
                        id.child("window_title_bar"),
                        UiWidget::Container,
                        UiStyle {
                            layout: title_layout,
                            visual: UiVisualStyle {
                                background: Some(ui.theme.group_background.with_alpha(0.92)),
                                border_color: None,
                                border_width: 0.0,
                                corner_radius: ui.theme.corner_radius,
                                clip: false,
                            },
                        },
                    ),
                    |ui| {
                        ui.label_with_style_with_id(
                            id.child("window_title"),
                            title,
                            crate::UiTextStyle {
                                color: ui.theme.text,
                                font_size: ui.theme.heading_size,
                                align_h: crate::UiTextAlign::Start,
                                align_v: crate::UiTextAlign::Center,
                                wrap: false,
                            },
                        );

                        let has_window_controls = options.collapsible || options.closable;
                        if has_window_controls {
                            let mut spacer_layout = UiLayoutStyle::default();
                            spacer_layout.width = UiDimension::points(0.0);
                            spacer_layout.height = UiDimension::points(1.0);
                            spacer_layout.flex_grow = 1.0;
                            spacer_layout.flex_shrink = 1.0;
                            ui.push_leaf(UiNode::new(
                                id.child("window_title_controls_fill"),
                                UiWidget::Spacer,
                                UiStyle {
                                    layout: spacer_layout,
                                    visual: UiVisualStyle::default(),
                                },
                            ));
                            ui.spacer_with_id(
                                id.child("window_title_controls_gap"),
                                Vec2::new((ui.theme.item_padding * 0.85).max(8.0), 0.0),
                            );
                        }

                        if options.collapsible {
                            let fold_label = if collapsed { "+" } else { "-" };
                            if ui
                                .button_variant_with_id(
                                    id.child("window_fold"),
                                    fold_label,
                                    UiButtonVariant::Secondary,
                                )
                                .clicked()
                            {
                                collapsed = !collapsed;
                            }
                        }
                        if options.closable
                            && ui
                                .button_variant_with_id(
                                    id.child("window_close"),
                                    "x",
                                    UiButtonVariant::Danger,
                                )
                                .clicked()
                        {
                            close_requested = true;
                        }
                    },
                );
                if !collapsed {
                    ui.separator_with_id(id.child("window_sep"));
                }
            }
            if !collapsed {
                let content_viewport = id.child("window_content_viewport");
                let content_inner = id.child("window_content_inner");
                let content_extent = id.child("window_content_extent");
                content_viewport_id = Some(content_viewport);
                content_inner_id = Some(content_inner);
                content_extent_id = Some(content_extent);
                let max_scroll_y = options.max_scroll_y.max(0.0);
                let scroll_y = if max_scroll_y > 0.0 {
                    options.scroll_y.clamp(0.0, max_scroll_y)
                } else {
                    0.0
                };
                let viewport_height = (size.y
                    - if has_title_bar {
                        title_bar_height + 1.0
                    } else {
                        0.0
                    })
                .max(1.0);
                let viewport_padding = ui.theme.window_padding;

                let mut viewport_layout = base_column_layout(0.0);
                viewport_layout.flex_grow = 1.0;
                ui.with_node(
                    UiNode::new(
                        content_viewport,
                        UiWidget::Container,
                        UiStyle {
                            layout: viewport_layout,
                            visual: UiVisualStyle {
                                clip: true,
                                ..UiVisualStyle::default()
                            },
                        },
                    ),
                    |ui| {
                        let mut inner_layout = base_column_layout(ui.theme.spacing);
                        inner_layout.position_type = UiPositionType::Absolute;
                        inner_layout.left = UiDimension::points(0.0);
                        inner_layout.right = UiDimension::points(0.0);
                        inner_layout.top = UiDimension::points(-scroll_y);
                        inner_layout.width = UiDimension::percent(1.0);
                        inner_layout.flex_shrink = 0.0;
                        inner_layout.padding = UiInsets::all(viewport_padding);
                        if has_title_bar {
                            inner_layout.padding.top = ui.theme.window_padding * 0.75;
                        }
                        ui.with_node(
                            UiNode::new(
                                content_inner,
                                UiWidget::Container,
                                UiStyle {
                                    layout: inner_layout,
                                    visual: UiVisualStyle::default(),
                                },
                            ),
                            |ui| {
                                build(ui);
                                let mut marker_layout = UiLayoutStyle::default();
                                marker_layout.width = UiDimension::points(0.0);
                                marker_layout.height = UiDimension::points(0.0);
                                marker_layout.flex_shrink = 0.0;
                                ui.push_leaf(UiNode::new(
                                    content_extent,
                                    UiWidget::Spacer,
                                    UiStyle {
                                        layout: marker_layout,
                                        visual: UiVisualStyle::default(),
                                    },
                                ));
                            },
                        );

                        if options.max_scroll_y > 0.0 {
                            let track_inset = 3.0;
                            let track_width = 6.0;
                            let track_height = (viewport_height - track_inset * 2.0).max(8.0);
                            let visible_ratio = (viewport_height
                                / (viewport_height + options.max_scroll_y.max(0.0)))
                            .clamp(0.05, 1.0);
                            let thumb_height =
                                (track_height * visible_ratio).clamp(14.0, track_height);
                            let travel = (track_height - thumb_height).max(0.0);
                            let t = if options.max_scroll_y > 0.0 {
                                (scroll_y / options.max_scroll_y).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };

                            let mut track_layout = UiLayoutStyle::default();
                            track_layout.position_type = UiPositionType::Absolute;
                            track_layout.right = UiDimension::points(track_inset);
                            track_layout.top = UiDimension::points(track_inset);
                            track_layout.width = UiDimension::points(track_width);
                            track_layout.height = UiDimension::points(track_height);
                            ui.push_leaf(UiNode::new(
                                id.child("window_scroll_track"),
                                UiWidget::Container,
                                UiStyle {
                                    layout: track_layout,
                                    visual: UiVisualStyle {
                                        background: Some(ui.theme.panel_border.with_alpha(0.35)),
                                        corner_radius: 3.0,
                                        ..UiVisualStyle::default()
                                    },
                                },
                            ));

                            let mut thumb_layout = UiLayoutStyle::default();
                            thumb_layout.position_type = UiPositionType::Absolute;
                            thumb_layout.right = UiDimension::points(track_inset);
                            thumb_layout.top = UiDimension::points(track_inset + travel * t);
                            thumb_layout.width = UiDimension::points(track_width);
                            thumb_layout.height = UiDimension::points(thumb_height);
                            ui.push_leaf(UiNode::new(
                                id.child("window_scroll_thumb"),
                                UiWidget::Container,
                                UiStyle {
                                    layout: thumb_layout,
                                    visual: UiVisualStyle {
                                        background: Some(ui.theme.button_primary.with_alpha(0.82)),
                                        border_color: Some(ui.theme.button_text.with_alpha(0.48)),
                                        border_width: 1.0,
                                        corner_radius: 3.0,
                                        ..UiVisualStyle::default()
                                    },
                                },
                            ));
                        }
                    },
                );
            }
        });

        UiWindowResponse {
            collapsed,
            close_requested,
            content_viewport_id,
            content_inner_id,
            content_extent_id,
        }
    }
}
