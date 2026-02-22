use glam::Vec2;

use crate::{
    IntoUiId, UiButton, UiButtonVariant, UiDimension, UiDisclosure, UiFlexWrap, UiId, UiImage,
    UiLabel, UiLayoutStyle, UiNode, UiStyle, UiTextAlign, UiTextField, UiTextStyle, UiTextValue,
    UiVisualStyle, UiWidget, estimate_text_size, estimate_text_width,
};

use super::{
    UiContext, UiResponse, Widget,
    layout::{base_column_layout, base_row_layout},
};

#[derive(Clone, Debug)]
pub struct LabelWidget {
    text: UiTextValue,
    style: UiTextStyle,
    id_source: Option<UiId>,
}

impl LabelWidget {
    pub fn new<S: Into<UiTextValue>>(text: S) -> Self {
        Self {
            text: text.into(),
            style: UiTextStyle::default(),
            id_source: None,
        }
    }

    pub fn style(mut self, style: UiTextStyle) -> Self {
        self.style = style;
        self
    }

    pub fn wrap(mut self, wrap: bool) -> Self {
        self.style.wrap = wrap;
        self
    }

    pub fn id<K: IntoUiId>(mut self, id_source: K) -> Self {
        self.id_source = Some(id_source.into_ui_id());
        self
    }
}

impl Widget for LabelWidget {
    fn ui(self, ui: &mut UiContext) -> UiResponse {
        ui.label_with_style_optional_id(self.id_source, self.text, self.style)
    }
}

#[derive(Clone, Debug)]
pub struct ButtonWidget {
    text: UiTextValue,
    variant: UiButtonVariant,
    fixed_width: Option<f32>,
    text_align_h: UiTextAlign,
    id_source: Option<UiId>,
}

impl ButtonWidget {
    pub fn new<S: Into<UiTextValue>>(text: S) -> Self {
        Self {
            text: text.into(),
            variant: UiButtonVariant::Primary,
            fixed_width: None,
            text_align_h: UiTextAlign::Center,
            id_source: None,
        }
    }

    pub fn variant(mut self, variant: UiButtonVariant) -> Self {
        self.variant = variant;
        self
    }

    pub fn fixed_width(mut self, width: f32) -> Self {
        self.fixed_width = Some(width.max(1.0));
        self
    }

    pub fn text_align(mut self, align: UiTextAlign) -> Self {
        self.text_align_h = align;
        self
    }

    pub fn id<K: IntoUiId>(mut self, id_source: K) -> Self {
        self.id_source = Some(id_source.into_ui_id());
        self
    }
}

impl Widget for ButtonWidget {
    fn ui(self, ui: &mut UiContext) -> UiResponse {
        ui.button_variant_with_options_optional_id(
            self.id_source,
            self.text,
            self.variant,
            self.fixed_width,
            self.text_align_h,
        )
    }
}

pub fn label<S: Into<UiTextValue>>(text: S) -> LabelWidget {
    LabelWidget::new(text)
}

pub fn button<S: Into<UiTextValue>>(text: S) -> ButtonWidget {
    ButtonWidget::new(text)
}

impl UiContext {
    fn resolve_widget_id(&mut self, id_source: Option<UiId>) -> UiId {
        match id_source {
            Some(id_source) => self.derive_id(id_source),
            None => self.next_auto_id(),
        }
    }

    fn label_with_style_optional_id<S: Into<UiTextValue>>(
        &mut self,
        id_source: Option<UiId>,
        text: S,
        style: UiTextStyle,
    ) -> UiResponse {
        let id = self.resolve_widget_id(id_source);
        let text: UiTextValue = text.into();
        let estimated = estimate_text_size(text.as_str(), style.font_size);
        let mut layout = UiLayoutStyle::default();
        if style.wrap {
            layout.width = UiDimension::percent(1.0);
            layout.height = UiDimension::Auto;
            layout.flex_shrink = 1.0;
        } else {
            layout.width = UiDimension::points(estimated.x);
            layout.height = UiDimension::points(estimated.y);
            layout.flex_shrink = 0.0;
        }

        self.push_leaf(UiNode::new(
            id,
            UiWidget::Label(UiLabel { text, style }),
            UiStyle {
                layout,
                visual: UiVisualStyle::default(),
            },
        ));

        self.response(id, true)
    }

    fn button_variant_with_options_optional_id<S: Into<UiTextValue>>(
        &mut self,
        id_source: Option<UiId>,
        text: S,
        variant: UiButtonVariant,
        fixed_width: Option<f32>,
        text_align_h: UiTextAlign,
    ) -> UiResponse {
        let id = self.resolve_widget_id(id_source);
        let text: UiTextValue = text.into();
        let text_style = UiTextStyle {
            color: self.theme.button_text,
            font_size: self.theme.font_size,
            align_h: text_align_h,
            align_v: UiTextAlign::Center,
            wrap: false,
        };
        let estimated = estimate_text_size(text.as_str(), text_style.font_size);

        let mut layout = UiLayoutStyle::default();
        layout.width = UiDimension::points(
            fixed_width
                .unwrap_or(estimated.x + self.theme.item_padding * 2.5)
                .max(1.0),
        );
        layout.height =
            UiDimension::points((estimated.y + self.theme.item_padding * 1.2).max(24.0));
        layout.flex_shrink = 0.0;

        self.push_leaf(UiNode::new(
            id,
            UiWidget::Button(UiButton {
                text,
                variant,
                enabled: true,
                style: text_style,
            }),
            UiStyle {
                layout,
                visual: UiVisualStyle {
                    border_color: Some(self.theme.panel_border.with_alpha(0.9)),
                    border_width: self.theme.border_width,
                    corner_radius: self.theme.corner_radius * 0.5,
                    ..UiVisualStyle::default()
                },
            },
        ));

        self.response(id, true)
    }

    pub fn label<S: Into<UiTextValue>>(&mut self, text: S) -> UiResponse {
        self.label_with_style_optional_id(None, text, UiTextStyle::default())
    }

    pub fn label_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
    ) -> UiResponse {
        self.label_with_style_optional_id(Some(key.into_ui_id()), text, UiTextStyle::default())
    }

    pub fn muted_label<S: Into<UiTextValue>>(&mut self, text: S) -> UiResponse {
        let style = UiTextStyle {
            color: self.theme.muted_text,
            font_size: self.theme.font_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: false,
        };
        self.label_with_style_optional_id(None, text, style)
    }

    pub fn muted_label_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
    ) -> UiResponse {
        let style = UiTextStyle {
            color: self.theme.muted_text,
            font_size: self.theme.font_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: false,
        };
        self.label_with_style_optional_id(Some(key.into_ui_id()), text, style)
    }

    pub fn wrapping_muted_label<S: Into<UiTextValue>>(&mut self, text: S) -> UiResponse {
        let style = UiTextStyle {
            color: self.theme.muted_text,
            font_size: self.theme.font_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: true,
        };
        self.label_with_style_optional_id(None, text, style)
    }

    pub fn wrapping_muted_label_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
    ) -> UiResponse {
        let style = UiTextStyle {
            color: self.theme.muted_text,
            font_size: self.theme.font_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: true,
        };
        self.label_with_style_optional_id(Some(key.into_ui_id()), text, style)
    }

    pub fn heading<S: Into<UiTextValue>>(&mut self, text: S) -> UiResponse {
        let style = UiTextStyle {
            color: self.theme.text,
            font_size: self.theme.heading_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: false,
        };
        self.label_with_style_optional_id(None, text, style)
    }

    pub fn heading_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
    ) -> UiResponse {
        let style = UiTextStyle {
            color: self.theme.text,
            font_size: self.theme.heading_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: false,
        };
        self.label_with_style_optional_id(Some(key.into_ui_id()), text, style)
    }

    pub fn wrapping_label<S: Into<UiTextValue>>(&mut self, text: S) -> UiResponse {
        let style = UiTextStyle {
            color: self.theme.text,
            font_size: self.theme.font_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: true,
        };
        self.label_with_style_optional_id(None, text, style)
    }

    pub fn wrapping_label_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
    ) -> UiResponse {
        let style = UiTextStyle {
            color: self.theme.text,
            font_size: self.theme.font_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: true,
        };
        self.label_with_style_optional_id(Some(key.into_ui_id()), text, style)
    }

    pub fn label_with_style<S: Into<UiTextValue>>(
        &mut self,
        text: S,
        style: UiTextStyle,
    ) -> UiResponse {
        self.label_with_style_optional_id(None, text, style)
    }

    pub fn label_with_style_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
        style: UiTextStyle,
    ) -> UiResponse {
        self.label_with_style_optional_id(Some(key.into_ui_id()), text, style)
    }

    pub fn button<S: Into<UiTextValue>>(&mut self, text: S) -> UiResponse {
        self.button_variant_with_options_optional_id(
            None,
            text,
            UiButtonVariant::Primary,
            None,
            UiTextAlign::Center,
        )
    }

    pub fn button_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
    ) -> UiResponse {
        self.button_variant_with_id(key, text, UiButtonVariant::Primary)
    }

    pub fn button_variant<S: Into<UiTextValue>>(
        &mut self,
        text: S,
        variant: UiButtonVariant,
    ) -> UiResponse {
        self.button_variant_with_options_optional_id(None, text, variant, None, UiTextAlign::Center)
    }

    pub fn button_variant_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
        variant: UiButtonVariant,
    ) -> UiResponse {
        self.button_variant_with_options_optional_id(
            Some(key.into_ui_id()),
            text,
            variant,
            None,
            UiTextAlign::Center,
        )
    }

    pub fn selectable_label<S: Into<UiTextValue>>(
        &mut self,
        selected: bool,
        text: S,
    ) -> UiResponse {
        let id = self.next_auto_id();
        self.selectable_label_with_id(id, selected, text)
    }

    pub fn selectable_label_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        selected: bool,
        text: S,
    ) -> UiResponse {
        let variant = if selected {
            UiButtonVariant::Primary
        } else {
            UiButtonVariant::Secondary
        };
        self.button_variant_with_options_with_id(key, text, variant, None, UiTextAlign::Start)
    }

    pub fn button_variant_with_options<S: Into<UiTextValue>>(
        &mut self,
        text: S,
        variant: UiButtonVariant,
        fixed_width: Option<f32>,
        text_align_h: UiTextAlign,
    ) -> UiResponse {
        self.button_variant_with_options_optional_id(None, text, variant, fixed_width, text_align_h)
    }

    pub fn button_variant_with_options_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
        variant: UiButtonVariant,
        fixed_width: Option<f32>,
        text_align_h: UiTextAlign,
    ) -> UiResponse {
        self.button_variant_with_options_optional_id(
            Some(key.into_ui_id()),
            text,
            variant,
            fixed_width,
            text_align_h,
        )
    }

    pub fn disclosure<S: Into<UiTextValue>>(&mut self, text: S, expanded: bool) -> UiResponse {
        let id = self.next_auto_id();
        self.disclosure_with_id(id, text, expanded)
    }

    pub fn disclosure_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
        expanded: bool,
    ) -> UiResponse {
        let id = self.derive_id(key);
        let text: UiTextValue = text.into();
        let text_style = UiTextStyle {
            color: self.theme.button_text,
            font_size: self.theme.font_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Center,
            wrap: false,
        };
        let estimated = estimate_text_size(text.as_str(), text_style.font_size);

        let mut layout = UiLayoutStyle::default();
        layout.width = UiDimension::percent(1.0);
        layout.height =
            UiDimension::points((estimated.y + self.theme.item_padding * 1.25).max(24.0));
        layout.flex_shrink = 0.0;

        self.push_leaf(UiNode::new(
            id,
            UiWidget::Disclosure(UiDisclosure {
                text,
                expanded,
                enabled: true,
                style: text_style,
            }),
            UiStyle {
                layout,
                visual: UiVisualStyle {
                    border_color: Some(self.theme.panel_border.with_alpha(0.9)),
                    border_width: self.theme.border_width,
                    corner_radius: self.theme.corner_radius * 0.5,
                    ..UiVisualStyle::default()
                },
            },
        ));

        self.response(id, true)
    }

    pub fn collapsing<S, F>(&mut self, text: S, build: F) -> UiResponse
    where
        S: Into<UiTextValue>,
        F: FnOnce(&mut Self),
    {
        let id = self.next_auto_id();
        self.collapsing_with_id(id, text, true, build)
    }

    pub fn collapsing_with_id<K, S, F>(
        &mut self,
        key: K,
        text: S,
        default_open: bool,
        build: F,
    ) -> UiResponse
    where
        K: IntoUiId,
        S: Into<UiTextValue>,
        F: FnOnce(&mut Self),
    {
        let id = self.derive_id(key);
        let text: UiTextValue = text.into();
        let mut expanded = self.bool_state_with_default(id, default_open);
        let response = self.disclosure_with_id(id.child("header"), text, expanded);
        if response.clicked() {
            expanded = !expanded;
            self.set_bool_state(id, expanded);
        }
        if expanded {
            let mut body_layout = base_column_layout(self.theme.spacing * 0.55);
            body_layout.width = UiDimension::percent(1.0);
            body_layout.padding.left = (self.theme.item_padding * 0.7).max(6.0);
            self.with_node(
                UiNode::new(
                    id.child("body"),
                    UiWidget::Container,
                    UiStyle {
                        layout: body_layout,
                        visual: UiVisualStyle::default(),
                    },
                ),
                build,
            );
        }
        response
    }

    pub fn collapsing_with_state<S, F>(
        &mut self,
        text: S,
        expanded: &mut bool,
        build: F,
    ) -> UiResponse
    where
        S: Into<UiTextValue>,
        F: FnOnce(&mut Self),
    {
        let id = self.next_auto_id();
        self.collapsing_with_state_with_id(id, text, expanded, build)
    }

    pub fn collapsing_with_state_with_id<K, S, F>(
        &mut self,
        key: K,
        text: S,
        expanded: &mut bool,
        build: F,
    ) -> UiResponse
    where
        K: IntoUiId,
        S: Into<UiTextValue>,
        F: FnOnce(&mut Self),
    {
        let id = self.derive_id(key);
        let response = self.disclosure_with_id(id.child("header"), text, *expanded);
        if response.clicked() {
            *expanded = !*expanded;
        }
        if *expanded {
            let mut body_layout = base_column_layout(self.theme.spacing * 0.55);
            body_layout.width = UiDimension::percent(1.0);
            body_layout.padding.left = (self.theme.item_padding * 0.7).max(6.0);
            self.with_node(
                UiNode::new(
                    id.child("body"),
                    UiWidget::Container,
                    UiStyle {
                        layout: body_layout,
                        visual: UiVisualStyle::default(),
                    },
                ),
                build,
            );
        }
        response
    }

    #[allow(clippy::too_many_arguments)]
    pub fn text_field_with_options<S: Into<UiTextValue>>(
        &mut self,
        text: S,
        fixed_width: Option<f32>,
        focused: bool,
        cursor: Option<usize>,
        selection: Option<(usize, usize)>,
        show_caret: bool,
        suffix: Option<UiTextValue>,
        scroll_x: f32,
    ) -> UiResponse {
        let id = self.next_auto_id();
        self.text_field_with_options_with_id(
            id,
            text,
            fixed_width,
            focused,
            cursor,
            selection,
            show_caret,
            suffix,
            scroll_x,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn text_field_with_options_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        text: S,
        fixed_width: Option<f32>,
        focused: bool,
        cursor: Option<usize>,
        selection: Option<(usize, usize)>,
        show_caret: bool,
        suffix: Option<UiTextValue>,
        scroll_x: f32,
    ) -> UiResponse {
        let id = self.derive_id(key);
        let text: UiTextValue = text.into();
        let text_style = UiTextStyle {
            color: self.theme.button_text,
            font_size: self.theme.font_size,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Center,
            wrap: false,
        };
        let estimated = estimate_text_size(text.as_str(), text_style.font_size);
        let base_text_width = estimate_text_width(text.as_str(), text_style.font_size)
            .max(conservative_text_width(text.as_str(), text_style.font_size))
            .max(text_style.font_size * 0.45);
        let text_guard = (text_style.font_size * 0.35).ceil().max(1.0);
        let measured_width = (base_text_width + text_guard).ceil();
        let text_padding_x = self.theme.item_padding.max(4.0);
        let suffix_width = suffix
            .as_ref()
            .map(|unit| {
                let base_suffix = estimate_text_width(unit.as_str(), text_style.font_size)
                    .max(conservative_text_width(unit.as_str(), text_style.font_size));
                let suffix_guard = (text_style.font_size * 0.20).ceil().max(1.0);
                (base_suffix + suffix_guard + text_padding_x * 0.35).ceil()
            })
            .unwrap_or(0.0);
        let auto_width = measured_width
            + suffix_width
            + text_padding_x * 2.0
            + self.theme.border_width.max(0.0) * 2.0;

        let mut layout = UiLayoutStyle::default();
        layout.width = UiDimension::points(fixed_width.unwrap_or(auto_width).max(1.0).ceil());
        layout.height =
            UiDimension::points((estimated.y + self.theme.item_padding * 1.2).max(24.0));
        layout.flex_shrink = 0.0;

        self.push_leaf(UiNode::new(
            id,
            UiWidget::TextField(UiTextField {
                text,
                suffix,
                scroll_x,
                style: text_style,
                enabled: true,
                focused,
                cursor,
                selection,
                show_caret,
                selection_color: self.theme.button_primary.with_alpha(0.30),
                caret_color: self.theme.button_text.with_alpha(0.95),
            }),
            UiStyle {
                layout,
                visual: UiVisualStyle {
                    border_color: Some(self.theme.panel_border.with_alpha(0.9)),
                    border_width: self.theme.border_width,
                    corner_radius: self.theme.corner_radius * 0.5,
                    ..UiVisualStyle::default()
                },
            },
        ));

        self.response(id, true)
    }

    pub fn text_edit_singleline<S: Into<UiTextValue>>(
        &mut self,
        text: S,
        focused: bool,
    ) -> UiResponse {
        self.text_field_with_options(text, None, focused, None, None, false, None, 0.0)
    }

    pub fn image(&mut self, texture_id: Option<usize>, size: Vec2) -> UiResponse {
        let id = self.next_auto_id();
        self.image_with_id(id, texture_id, size)
    }

    pub fn image_with_id<K: IntoUiId>(
        &mut self,
        key: K,
        texture_id: Option<usize>,
        size: Vec2,
    ) -> UiResponse {
        let id = self.derive_id(key);
        let mut layout = UiLayoutStyle::default();
        layout.width = UiDimension::points(size.x.max(1.0));
        layout.height = UiDimension::points(size.y.max(1.0));

        self.push_leaf(UiNode::new(
            id,
            UiWidget::Image(UiImage {
                texture_id,
                ..UiImage::default()
            }),
            UiStyle {
                layout,
                visual: UiVisualStyle {
                    border_color: Some(self.theme.panel_border.with_alpha(0.6)),
                    border_width: self.theme.border_width,
                    ..UiVisualStyle::default()
                },
            },
        ));

        self.response(id, true)
    }

    pub fn stepper_f32<S: Into<UiTextValue>>(
        &mut self,
        label: S,
        value: &mut f32,
        step: f32,
    ) -> bool {
        let id = self.next_auto_id();
        self.stepper_f32_with_id(id, label, value, step)
    }

    pub fn stepper_f32_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        label: S,
        value: &mut f32,
        step: f32,
    ) -> bool {
        let base = self.derive_id(key);
        let mut changed = false;
        let label_text: UiTextValue = label.into();

        self.row_with_id(base.child("stepper_row"), |ui| {
            ui.label_with_id(base.child("label"), label_text.clone());
            if ui
                .button_variant_with_id(base.child("minus"), "-", UiButtonVariant::Secondary)
                .clicked()
            {
                *value -= step;
                changed = true;
            }
            ui.label_with_id(base.child("value"), format!("{value:.3}"));
            if ui
                .button_variant_with_id(base.child("plus"), "+", UiButtonVariant::Secondary)
                .clicked()
            {
                *value += step;
                changed = true;
            }
        });

        changed
    }

    pub fn end_row(&mut self) {
        self.add_space_y(self.theme.spacing.max(2.0));
    }

    pub fn available_width(&self) -> f32 {
        self.path_stack
            .last()
            .and_then(|id| self.last_layout_rects.get(id).copied())
            .map(|rect| rect.width)
            .unwrap_or(0.0)
    }

    pub fn available_height(&self) -> f32 {
        self.path_stack
            .last()
            .and_then(|id| self.last_layout_rects.get(id).copied())
            .map(|rect| rect.height)
            .unwrap_or(0.0)
    }

    pub fn horizontal_wrapped<F>(&mut self, build: F)
    where
        F: FnOnce(&mut Self),
    {
        let id = self.next_auto_id();
        let mut style = base_row_layout(self.theme.spacing);
        style.width = UiDimension::percent(1.0);
        style.flex_wrap = UiFlexWrap::Wrap;
        self.with_node(
            UiNode::new(
                self.derive_id(id),
                UiWidget::Container,
                UiStyle {
                    layout: style,
                    visual: UiVisualStyle::default(),
                },
            ),
            build,
        );
    }
}

fn conservative_text_width(text: &str, font_size: f32) -> f32 {
    let measured = estimate_text_width(text, font_size).max(0.0);
    let conservative_floor = text.chars().count() as f32 * font_size.max(1.0) * 0.5;
    measured.max(conservative_floor)
}
