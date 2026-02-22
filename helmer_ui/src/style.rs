use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use glam::Vec2;

use crate::UiId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl UiColor {
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);
    pub const WHITE: Self = Self::rgba(1.0, 1.0, 1.0, 1.0);
    pub const BLACK: Self = Self::rgba(0.0, 0.0, 0.0, 1.0);

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    pub fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl Default for UiColor {
    fn default() -> Self {
        Self::WHITE
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl UiRect {
    pub fn right(self) -> f32 {
        self.x + self.width
    }

    pub fn bottom(self) -> f32 {
        self.y + self.height
    }

    pub fn contains(self, point: Vec2) -> bool {
        point.x >= self.x
            && point.y >= self.y
            && point.x <= self.right()
            && point.y <= self.bottom()
    }

    pub fn intersection(self, other: Self) -> Option<Self> {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some(Self {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiTextAlign {
    Start,
    Center,
    End,
}

impl Default for UiTextAlign {
    fn default() -> Self {
        Self::Start
    }
}

#[derive(Clone, Debug)]
pub enum UiTextValue {
    Static(&'static str),
    Shared(Arc<str>),
    Owned(String),
}

impl UiTextValue {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Static(text) => text,
            Self::Shared(text) => text.as_ref(),
            Self::Owned(text) => text.as_str(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.as_str().is_empty()
    }

    pub fn into_string(self) -> String {
        match self {
            Self::Static(text) => text.to_string(),
            Self::Shared(text) => text.as_ref().to_string(),
            Self::Owned(text) => text,
        }
    }
}

impl PartialEq for UiTextValue {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for UiTextValue {}

impl Hash for UiTextValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl Default for UiTextValue {
    fn default() -> Self {
        Self::Static("")
    }
}

impl AsRef<str> for UiTextValue {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<&'static str> for UiTextValue {
    fn from(value: &'static str) -> Self {
        Self::Static(value)
    }
}

impl From<String> for UiTextValue {
    fn from(value: String) -> Self {
        Self::Owned(value)
    }
}

impl From<Arc<str>> for UiTextValue {
    fn from(value: Arc<str>) -> Self {
        Self::Shared(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiTextStyle {
    pub color: UiColor,
    pub font_size: f32,
    pub align_h: UiTextAlign,
    pub align_v: UiTextAlign,
    pub wrap: bool,
}

impl Default for UiTextStyle {
    fn default() -> Self {
        Self {
            color: UiColor::WHITE,
            font_size: 16.0,
            align_h: UiTextAlign::Start,
            align_v: UiTextAlign::Start,
            wrap: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiVisualStyle {
    pub background: Option<UiColor>,
    pub border_color: Option<UiColor>,
    pub border_width: f32,
    pub corner_radius: f32,
    pub clip: bool,
}

impl Default for UiVisualStyle {
    fn default() -> Self {
        Self {
            background: None,
            border_color: None,
            border_width: 0.0,
            corner_radius: 0.0,
            clip: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiDimension {
    Auto,
    Points(f32),
    Percent(f32),
}

impl Default for UiDimension {
    fn default() -> Self {
        Self::Auto
    }
}

impl UiDimension {
    pub fn points(value: f32) -> Self {
        Self::Points(value)
    }

    pub fn percent(value: f32) -> Self {
        Self::Percent(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiInsets {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl UiInsets {
    pub const ZERO: Self = Self {
        left: 0.0,
        right: 0.0,
        top: 0.0,
        bottom: 0.0,
    };

    pub fn all(value: f32) -> Self {
        Self {
            left: value,
            right: value,
            top: value,
            bottom: value,
        }
    }
}

impl Default for UiInsets {
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiDisplay {
    Flex,
    None,
}

impl Default for UiDisplay {
    fn default() -> Self {
        Self::Flex
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiFlexDirection {
    Row,
    Column,
}

impl Default for UiFlexDirection {
    fn default() -> Self {
        Self::Column
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiFlexWrap {
    NoWrap,
    Wrap,
}

impl Default for UiFlexWrap {
    fn default() -> Self {
        Self::NoWrap
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiPositionType {
    Relative,
    Absolute,
}

impl Default for UiPositionType {
    fn default() -> Self {
        Self::Relative
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiAlignItems {
    Start,
    Center,
    End,
    Stretch,
}

impl Default for UiAlignItems {
    fn default() -> Self {
        Self::Stretch
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiLayoutStyle {
    pub display: UiDisplay,
    pub flex_direction: UiFlexDirection,
    pub flex_wrap: UiFlexWrap,
    pub position_type: UiPositionType,
    pub align_items: UiAlignItems,
    pub left: UiDimension,
    pub right: UiDimension,
    pub top: UiDimension,
    pub bottom: UiDimension,
    pub width: UiDimension,
    pub height: UiDimension,
    pub padding: UiInsets,
    pub margin: UiInsets,
    pub gap: Vec2,
    pub flex_grow: f32,
    pub flex_shrink: f32,
}

impl Default for UiLayoutStyle {
    fn default() -> Self {
        Self {
            display: UiDisplay::Flex,
            flex_direction: UiFlexDirection::Column,
            flex_wrap: UiFlexWrap::NoWrap,
            position_type: UiPositionType::Relative,
            align_items: UiAlignItems::Stretch,
            left: UiDimension::Auto,
            right: UiDimension::Auto,
            top: UiDimension::Auto,
            bottom: UiDimension::Auto,
            width: UiDimension::Auto,
            height: UiDimension::Auto,
            padding: UiInsets::ZERO,
            margin: UiInsets::ZERO,
            gap: Vec2::ZERO,
            flex_grow: 0.0,
            flex_shrink: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiStyle {
    pub layout: UiLayoutStyle,
    pub visual: UiVisualStyle,
}

impl Default for UiStyle {
    fn default() -> Self {
        Self {
            layout: UiLayoutStyle::default(),
            visual: UiVisualStyle::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiLabel {
    pub text: UiTextValue,
    pub style: UiTextStyle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiButtonVariant {
    Primary,
    Secondary,
    Danger,
    Ghost,
}

impl Default for UiButtonVariant {
    fn default() -> Self {
        Self::Primary
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiButton {
    pub text: UiTextValue,
    pub variant: UiButtonVariant,
    pub enabled: bool,
    pub style: UiTextStyle,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiDisclosure {
    pub text: UiTextValue,
    pub expanded: bool,
    pub enabled: bool,
    pub style: UiTextStyle,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiTextField {
    pub text: UiTextValue,
    pub suffix: Option<UiTextValue>,
    pub scroll_x: f32,
    pub style: UiTextStyle,
    pub enabled: bool,
    pub focused: bool,
    pub cursor: Option<usize>,
    pub selection: Option<(usize, usize)>,
    pub show_caret: bool,
    pub selection_color: UiColor,
    pub caret_color: UiColor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiImage {
    pub texture_id: Option<usize>,
    pub tint: UiColor,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
}

impl Default for UiImage {
    fn default() -> Self {
        Self {
            texture_id: None,
            tint: UiColor::WHITE,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiWidget {
    Container,
    Label(UiLabel),
    Button(UiButton),
    Disclosure(UiDisclosure),
    TextField(UiTextField),
    Image(UiImage),
    Spacer,
}

impl Default for UiWidget {
    fn default() -> Self {
        Self::Container
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiNode {
    pub id: UiId,
    pub widget: UiWidget,
    pub style: UiStyle,
    pub enabled: bool,
    pub children: Vec<UiNode>,
}

impl UiNode {
    pub fn new(id: UiId, widget: UiWidget, style: UiStyle) -> Self {
        Self {
            id,
            widget,
            style,
            enabled: true,
            children: Vec::new(),
        }
    }

    pub fn with_children(mut self, children: Vec<UiNode>) -> Self {
        self.children = children;
        self
    }
}

#[derive(Clone, Debug)]
pub struct UiTheme {
    pub panel_background: UiColor,
    pub panel_border: UiColor,
    pub group_background: UiColor,
    pub text: UiColor,
    pub muted_text: UiColor,
    pub button_primary: UiColor,
    pub button_secondary: UiColor,
    pub button_danger: UiColor,
    pub button_hover_overlay: UiColor,
    pub button_pressed_overlay: UiColor,
    pub button_text: UiColor,
    pub button_disabled: UiColor,
    pub spacing: f32,
    pub window_padding: f32,
    pub item_padding: f32,
    pub corner_radius: f32,
    pub border_width: f32,
    pub font_size: f32,
    pub heading_size: f32,
}

impl Default for UiTheme {
    fn default() -> Self {
        Self {
            panel_background: UiColor::rgba(0.10, 0.11, 0.13, 0.94),
            panel_border: UiColor::rgba(0.24, 0.27, 0.32, 0.95),
            group_background: UiColor::rgba(0.15, 0.17, 0.20, 0.85),
            text: UiColor::rgba(0.94, 0.95, 0.97, 1.0),
            muted_text: UiColor::rgba(0.72, 0.75, 0.80, 1.0),
            button_primary: UiColor::rgba(0.12, 0.42, 0.84, 1.0),
            button_secondary: UiColor::rgba(0.22, 0.25, 0.31, 1.0),
            button_danger: UiColor::rgba(0.73, 0.18, 0.20, 1.0),
            button_hover_overlay: UiColor::rgba(1.0, 1.0, 1.0, 0.10),
            button_pressed_overlay: UiColor::rgba(0.0, 0.0, 0.0, 0.18),
            button_text: UiColor::rgba(0.97, 0.98, 0.99, 1.0),
            button_disabled: UiColor::rgba(0.35, 0.36, 0.40, 1.0),
            spacing: 8.0,
            window_padding: 10.0,
            item_padding: 6.0,
            corner_radius: 8.0,
            border_width: 1.0,
            font_size: 15.0,
            heading_size: 17.0,
        }
    }
}

pub fn estimate_text_size(text: &str, font_size: f32) -> Vec2 {
    let line_count = text.lines().count().max(1) as f32;
    let longest_line = text
        .lines()
        .map(|line| estimate_text_width(line, font_size))
        .fold(0.0, f32::max);
    let width = longest_line.max(font_size * 0.5);
    let height = (line_count * font_size * 1.3).max(font_size);
    Vec2::new(width, height)
}

pub fn estimate_text_width(text: &str, font_size: f32) -> f32 {
    text.chars()
        .map(|ch| estimate_char_advance(ch, font_size))
        .sum::<f32>()
}

pub fn estimate_text_prefix_width(text: &str, end_chars: usize, font_size: f32) -> f32 {
    let mut width = 0.0f32;
    let mut count = 0usize;
    for ch in text.chars() {
        if count >= end_chars {
            break;
        }
        width += estimate_char_advance(ch, font_size);
        count += 1;
    }
    width
}

pub fn estimate_char_advance(ch: char, font_size: f32) -> f32 {
    let base = font_size.max(1.0);
    match ch {
        ' ' => base * 0.33,
        'i' | 'l' | '!' | '|' | ':' | ';' | '\'' | '"' => base * 0.30,
        't' | 'f' | 'r' | 'j' | '1' => base * 0.42,
        'm' | 'w' | 'M' | 'W' | '@' | '#' => base * 0.86,
        _ if ch.is_ascii_digit() => base * 0.58,
        _ if ch.is_ascii_uppercase() => base * 0.66,
        _ => base * 0.58,
    }
}

#[derive(Clone, Debug, Default)]
pub struct UiLayoutBuilder {
    layout: UiLayoutStyle,
}

impl UiLayoutBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn display(mut self, display: UiDisplay) -> Self {
        self.layout.display = display;
        self
    }

    pub fn flex_direction(mut self, direction: UiFlexDirection) -> Self {
        self.layout.flex_direction = direction;
        self
    }

    pub fn flex_wrap(mut self, wrap: UiFlexWrap) -> Self {
        self.layout.flex_wrap = wrap;
        self
    }

    pub fn position_type(mut self, position: UiPositionType) -> Self {
        self.layout.position_type = position;
        self
    }

    pub fn align_items(mut self, align: UiAlignItems) -> Self {
        self.layout.align_items = align;
        self
    }

    pub fn width(mut self, width: UiDimension) -> Self {
        self.layout.width = width;
        self
    }

    pub fn width_px(self, width: f32) -> Self {
        self.width(UiDimension::points(width))
    }

    pub fn width_percent(self, width: f32) -> Self {
        self.width(UiDimension::percent(width))
    }

    pub fn width_auto(self) -> Self {
        self.width(UiDimension::Auto)
    }

    pub fn height(mut self, height: UiDimension) -> Self {
        self.layout.height = height;
        self
    }

    pub fn height_px(self, height: f32) -> Self {
        self.height(UiDimension::points(height))
    }

    pub fn height_percent(self, height: f32) -> Self {
        self.height(UiDimension::percent(height))
    }

    pub fn height_auto(self) -> Self {
        self.height(UiDimension::Auto)
    }

    pub fn left(mut self, left: UiDimension) -> Self {
        self.layout.left = left;
        self
    }

    pub fn left_px(self, left: f32) -> Self {
        self.left(UiDimension::points(left))
    }

    pub fn right(mut self, right: UiDimension) -> Self {
        self.layout.right = right;
        self
    }

    pub fn right_px(self, right: f32) -> Self {
        self.right(UiDimension::points(right))
    }

    pub fn top(mut self, top: UiDimension) -> Self {
        self.layout.top = top;
        self
    }

    pub fn top_px(self, top: f32) -> Self {
        self.top(UiDimension::points(top))
    }

    pub fn bottom(mut self, bottom: UiDimension) -> Self {
        self.layout.bottom = bottom;
        self
    }

    pub fn bottom_px(self, bottom: f32) -> Self {
        self.bottom(UiDimension::points(bottom))
    }

    pub fn position_xy(self, x: f32, y: f32) -> Self {
        self.left_px(x).top_px(y)
    }

    pub fn padding(mut self, padding: UiInsets) -> Self {
        self.layout.padding = padding;
        self
    }

    pub fn padding_all(self, value: f32) -> Self {
        self.padding(UiInsets::all(value))
    }

    pub fn padding_x(mut self, value: f32) -> Self {
        self.layout.padding.left = value;
        self.layout.padding.right = value;
        self
    }

    pub fn padding_y(mut self, value: f32) -> Self {
        self.layout.padding.top = value;
        self.layout.padding.bottom = value;
        self
    }

    pub fn padding_left(mut self, value: f32) -> Self {
        self.layout.padding.left = value;
        self
    }

    pub fn padding_right(mut self, value: f32) -> Self {
        self.layout.padding.right = value;
        self
    }

    pub fn padding_top(mut self, value: f32) -> Self {
        self.layout.padding.top = value;
        self
    }

    pub fn padding_bottom(mut self, value: f32) -> Self {
        self.layout.padding.bottom = value;
        self
    }

    pub fn margin(mut self, margin: UiInsets) -> Self {
        self.layout.margin = margin;
        self
    }

    pub fn margin_all(self, value: f32) -> Self {
        self.margin(UiInsets::all(value))
    }

    pub fn margin_x(mut self, value: f32) -> Self {
        self.layout.margin.left = value;
        self.layout.margin.right = value;
        self
    }

    pub fn margin_y(mut self, value: f32) -> Self {
        self.layout.margin.top = value;
        self.layout.margin.bottom = value;
        self
    }

    pub fn margin_left(mut self, value: f32) -> Self {
        self.layout.margin.left = value;
        self
    }

    pub fn margin_right(mut self, value: f32) -> Self {
        self.layout.margin.right = value;
        self
    }

    pub fn margin_top(mut self, value: f32) -> Self {
        self.layout.margin.top = value;
        self
    }

    pub fn margin_bottom(mut self, value: f32) -> Self {
        self.layout.margin.bottom = value;
        self
    }

    pub fn gap(mut self, gap: Vec2) -> Self {
        self.layout.gap = gap;
        self
    }

    pub fn gap_xy(self, x: f32, y: f32) -> Self {
        self.gap(Vec2::new(x, y))
    }

    pub fn gap_x(mut self, x: f32) -> Self {
        self.layout.gap.x = x;
        self
    }

    pub fn gap_y(mut self, y: f32) -> Self {
        self.layout.gap.y = y;
        self
    }

    pub fn flex_grow(mut self, grow: f32) -> Self {
        self.layout.flex_grow = grow;
        self
    }

    pub fn flex_shrink(mut self, shrink: f32) -> Self {
        self.layout.flex_shrink = shrink;
        self
    }

    pub fn build(self) -> UiLayoutStyle {
        self.layout
    }
}

#[derive(Clone, Debug, Default)]
pub struct UiVisualStyleBuilder {
    visual: UiVisualStyle,
}

impl UiVisualStyleBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn background(mut self, color: UiColor) -> Self {
        self.visual.background = Some(color);
        self
    }

    pub fn border(mut self, color: UiColor, width: f32) -> Self {
        self.visual.border_color = Some(color);
        self.visual.border_width = width.max(0.0);
        self
    }

    pub fn border_color(mut self, color: UiColor) -> Self {
        self.visual.border_color = Some(color);
        self
    }

    pub fn border_width(mut self, width: f32) -> Self {
        self.visual.border_width = width.max(0.0);
        self
    }

    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.visual.corner_radius = radius.max(0.0);
        self
    }

    pub fn clip(mut self, clip: bool) -> Self {
        self.visual.clip = clip;
        self
    }

    pub fn build(self) -> UiVisualStyle {
        self.visual
    }
}

#[derive(Clone, Debug, Default)]
pub struct UiStyleBuilder {
    style: UiStyle,
}

impl UiStyleBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn layout(mut self, layout: UiLayoutStyle) -> Self {
        self.style.layout = layout;
        self
    }

    pub fn layout_builder(self, builder: UiLayoutBuilder) -> Self {
        self.layout(builder.build())
    }

    pub fn visual(mut self, visual: UiVisualStyle) -> Self {
        self.style.visual = visual;
        self
    }

    pub fn visual_builder(self, builder: UiVisualStyleBuilder) -> Self {
        self.visual(builder.build())
    }

    pub fn build(self) -> UiStyle {
        self.style
    }
}
