use glam::Vec2;

use crate::{
    IntoUiId, UiAlignItems, UiDimension, UiInsets, UiLayoutStyle, UiNode, UiStyle, UiStyleBuilder,
    UiVisualStyle, UiWidget,
};

use super::UiContext;

impl UiContext {
    pub fn scope<F>(&mut self, build: F)
    where
        F: FnOnce(&mut Self),
    {
        let id = self.next_auto_id();
        self.with_id(id, build);
    }

    pub fn horizontal<F>(&mut self, build: F)
    where
        F: FnOnce(&mut Self),
    {
        self.row(build);
    }

    pub fn horizontal_with_id<K, F>(&mut self, key: K, build: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        self.row_with_id(key, build);
    }

    pub fn vertical<F>(&mut self, build: F)
    where
        F: FnOnce(&mut Self),
    {
        self.column(build);
    }

    pub fn vertical_with_id<K, F>(&mut self, key: K, build: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        self.column_with_id(key, build);
    }

    pub fn add_space(&mut self, amount: f32) {
        let amount = amount.max(0.0);
        self.spacer(Vec2::new(amount, amount));
    }

    pub fn add_space_x(&mut self, amount: f32) {
        self.spacer(Vec2::new(amount.max(0.0), 0.0));
    }

    pub fn add_space_y(&mut self, amount: f32) {
        self.spacer(Vec2::new(0.0, amount.max(0.0)));
    }

    pub fn column<F>(&mut self, build: F)
    where
        F: FnOnce(&mut Self),
    {
        let id = self.next_auto_id();
        self.column_with_id(id, build);
    }

    pub fn column_with_id<K, F>(&mut self, key: K, build: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        let id = self.derive_id(key);
        let style = UiStyle {
            layout: base_column_layout(self.theme.spacing),
            visual: UiVisualStyle::default(),
        };
        self.with_node(UiNode::new(id, UiWidget::Container, style), build);
    }

    pub fn row<F>(&mut self, build: F)
    where
        F: FnOnce(&mut Self),
    {
        let id = self.next_auto_id();
        self.row_with_id(id, build);
    }

    pub fn row_with_id<K, F>(&mut self, key: K, build: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        let id = self.derive_id(key);
        let style = UiStyle {
            layout: base_row_layout(self.theme.spacing),
            visual: UiVisualStyle::default(),
        };
        self.with_node(UiNode::new(id, UiWidget::Container, style), build);
    }

    pub fn group<F>(&mut self, build: F)
    where
        F: FnOnce(&mut Self),
    {
        let id = self.next_auto_id();
        self.group_with_id(id, build);
    }

    pub fn group_with_id<K, F>(&mut self, key: K, build: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        let id = self.derive_id(key);
        let mut layout = base_column_layout(self.theme.spacing * 0.75);
        layout.width = UiDimension::percent(1.0);
        layout.padding = UiInsets::all(self.theme.item_padding);

        self.with_node(
            UiNode::new(
                id,
                UiWidget::Container,
                UiStyle {
                    layout,
                    visual: UiVisualStyle {
                        background: Some(self.theme.group_background),
                        border_color: Some(self.theme.panel_border.with_alpha(0.8)),
                        border_width: self.theme.border_width,
                        corner_radius: self.theme.corner_radius * 0.5,
                        clip: false,
                    },
                },
            ),
            build,
        );
    }

    pub fn container<F>(&mut self, style: UiStyle, build: F)
    where
        F: FnOnce(&mut Self),
    {
        let id = self.next_auto_id();
        self.container_with_id(id, style, build);
    }

    pub fn container_with_id<K, F>(&mut self, key: K, style: UiStyle, build: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        let id = self.derive_id(key);
        self.with_node(UiNode::new(id, UiWidget::Container, style), build);
    }

    pub fn container_css<F, S>(&mut self, style: S, build: F)
    where
        F: FnOnce(&mut Self),
        S: FnOnce(UiStyleBuilder) -> UiStyleBuilder,
    {
        self.container(style(UiStyleBuilder::new()).build(), build);
    }

    pub fn container_css_with_id<K, F, S>(&mut self, key: K, style: S, build: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
        S: FnOnce(UiStyleBuilder) -> UiStyleBuilder,
    {
        self.container_with_id(key, style(UiStyleBuilder::new()).build(), build);
    }

    pub fn spacer(&mut self, size: Vec2) {
        let id = self.next_auto_id();
        self.spacer_with_id(id, size);
    }

    pub fn spacer_with_id<K: IntoUiId>(&mut self, key: K, size: Vec2) {
        let id = self.derive_id(key);
        let mut layout = UiLayoutStyle::default();
        layout.width = UiDimension::points(size.x.max(0.0));
        layout.height = UiDimension::points(size.y.max(0.0));

        self.push_leaf(UiNode::new(
            id,
            UiWidget::Spacer,
            UiStyle {
                layout,
                visual: UiVisualStyle::default(),
            },
        ));
    }

    pub fn separator(&mut self) {
        let id = self.next_auto_id();
        self.separator_with_id(id);
    }

    pub fn separator_with_id<K: IntoUiId>(&mut self, key: K) {
        let id = self.derive_id(key);
        let mut layout = UiLayoutStyle::default();
        layout.width = UiDimension::percent(1.0);
        layout.height = UiDimension::points(1.0);
        layout.flex_shrink = 0.0;

        self.push_leaf(UiNode::new(
            id,
            UiWidget::Container,
            UiStyle {
                layout,
                visual: UiVisualStyle {
                    background: Some(self.theme.panel_border.with_alpha(0.9)),
                    ..UiVisualStyle::default()
                },
            },
        ));
    }
}

pub(super) fn base_column_layout(spacing: f32) -> UiLayoutStyle {
    let mut layout = UiLayoutStyle::default();
    layout.flex_direction = crate::UiFlexDirection::Column;
    layout.gap = Vec2::splat(spacing.max(0.0));
    layout.flex_shrink = 0.0;
    layout
}

pub(super) fn base_row_layout(spacing: f32) -> UiLayoutStyle {
    let mut layout = UiLayoutStyle::default();
    layout.flex_direction = crate::UiFlexDirection::Row;
    layout.align_items = UiAlignItems::Center;
    layout.gap = Vec2::splat(spacing.max(0.0));
    layout.flex_shrink = 0.0;
    layout
}
