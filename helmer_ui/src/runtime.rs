use std::{cell::RefCell, sync::Arc};

use glam::Vec2;
use hashbrown::HashMap;
use taffy::{
    Overflow, TaffyTree,
    prelude::{
        AlignItems, AvailableSpace, Dimension, Display, FlexDirection, FlexWrap, LengthPercentage,
        LengthPercentageAuto, Position, Rect as TaffyRect, Size, Style, length,
    },
};

use crate::{
    RetainedUi, UiAlignItems, UiButtonVariant, UiColor, UiContext, UiDimension, UiDisplay,
    UiDragInputSnapshot, UiDragValueState, UiDrawCommand, UiFlexDirection, UiFlexWrap,
    UiFrameOutput, UiId, UiImageCommand, UiInteractionState, UiLayoutStyle, UiNode, UiPositionType,
    UiRect, UiRectCommand, UiTextAlign, UiTextCommand, UiTextValue, UiTheme, UiWidget,
    estimate_char_advance, estimate_text_width,
};

#[derive(Clone, Copy, Debug)]
pub struct UiInputSnapshot {
    pub viewport_size: Vec2,
    pub pointer_position: Option<Vec2>,
    pub pointer_down: bool,
    pub scroll_delta: Vec2,
}

impl Default for UiInputSnapshot {
    fn default() -> Self {
        Self {
            viewport_size: Vec2::new(1.0, 1.0),
            pointer_position: None,
            pointer_down: false,
            scroll_delta: Vec2::ZERO,
        }
    }
}

#[derive(Clone)]
struct LayoutTreeNode {
    taffy_node: taffy::NodeId,
    children: Vec<LayoutTreeNode>,
}

#[derive(Clone)]
enum LayoutMeasureContext {
    Text(TextMeasureContext),
}

#[derive(Clone)]
struct TextMeasureContext {
    text: Option<Arc<str>>,
    font_size: f32,
    wrap: bool,
    line_count: f32,
    unwrapped_width: f32,
    cached_wrap_width: f32,
    cached_wrapped_lines: f32,
    has_wrapped_cache: bool,
}

#[derive(Clone)]
struct CachedTextMetrics {
    text: Arc<str>,
    font_size: f32,
    wrap: bool,
    line_count: f32,
    unwrapped_width: f32,
    cached_wrap_width: f32,
    cached_wrapped_lines: f32,
    has_wrapped_cache: bool,
}

#[derive(Clone)]
struct CachedTextHash {
    text: UiTextValue,
    hash: u64,
}

pub struct UiRuntime {
    theme: UiTheme,
    retained: RetainedUi,
    last_layout_rects: HashMap<UiId, UiRect>,
    last_hit_rects: HashMap<UiId, UiRect>,
    last_paint_order: Vec<UiId>,
    active_id: Option<UiId>,
    pointer_down_previous: bool,
    interaction: UiInteractionState,
    draw_commands: Vec<UiDrawCommand>,
    frame_layout_rects: HashMap<UiId, UiRect>,
    frame_hit_rects: HashMap<UiId, UiRect>,
    frame_paint_order: Vec<UiId>,
    immediate_bool_state: HashMap<UiId, bool>,
    immediate_drag_value_state: UiDragValueState,
    layer_cursor: f32,
    frame_command_hash: u64,
    hash_draw_commands_enabled: bool,
    text_metrics_cache: HashMap<UiId, CachedTextMetrics>,
    text_hash_cache: HashMap<UiId, CachedTextHash>,
    cache_prune_counter: u8,
}

impl Default for UiRuntime {
    fn default() -> Self {
        Self {
            theme: UiTheme::default(),
            retained: RetainedUi::default(),
            last_layout_rects: HashMap::new(),
            last_hit_rects: HashMap::new(),
            last_paint_order: Vec::new(),
            active_id: None,
            pointer_down_previous: false,
            interaction: UiInteractionState::default(),
            draw_commands: Vec::new(),
            frame_layout_rects: HashMap::new(),
            frame_hit_rects: HashMap::new(),
            frame_paint_order: Vec::new(),
            immediate_bool_state: HashMap::new(),
            immediate_drag_value_state: UiDragValueState::default(),
            layer_cursor: 0.8,
            frame_command_hash: FNV_OFFSET_BASIS,
            hash_draw_commands_enabled: true,
            text_metrics_cache: HashMap::new(),
            text_hash_cache: HashMap::new(),
            cache_prune_counter: 0,
        }
    }
}

thread_local! {
    static TAFFY_SCRATCH: RefCell<TaffyTree<LayoutMeasureContext>> =
        RefCell::new(TaffyTree::new());
}

impl UiRuntime {
    pub fn theme(&self) -> &UiTheme {
        &self.theme
    }

    pub fn theme_mut(&mut self) -> &mut UiTheme {
        &mut self.theme
    }

    pub fn retained(&self) -> &RetainedUi {
        &self.retained
    }

    pub fn retained_mut(&mut self) -> &mut RetainedUi {
        &mut self.retained
    }

    pub fn interaction(&self) -> UiInteractionState {
        self.interaction
    }

    pub fn wants_pointer_input(&self) -> bool {
        self.interaction.pointer_captured
    }

    pub fn is_pointer_over_interactive(&self, pointer: Vec2) -> bool {
        self.hit_test_interactive(pointer).is_some()
    }

    pub fn take_drag_clipboard_write(&mut self) -> Option<String> {
        self.immediate_drag_value_state.take_clipboard_write()
    }

    pub fn layout_rect(&self, id: UiId) -> Option<UiRect> {
        self.last_layout_rects.get(&id).copied()
    }

    pub fn layout_rect_count(&self) -> usize {
        self.last_layout_rects.len()
    }

    pub fn recycle_draw_commands(&mut self, mut commands: Vec<UiDrawCommand>) {
        commands.clear();
        if commands.capacity() > self.draw_commands.capacity() {
            self.draw_commands = commands;
        }
    }

    fn hit_test_interactive(&self, pointer: Vec2) -> Option<UiId> {
        self.last_paint_order.iter().rev().find_map(|id| {
            self.last_hit_rects
                .get(id)
                .filter(|rect| rect.contains(pointer))
                .map(|_| *id)
        })
    }

    pub fn run_frame<F>(&mut self, input: UiInputSnapshot, build_immediate: F) -> UiFrameOutput
    where
        F: FnOnce(&mut UiContext),
    {
        self.run_frame_with_options(input, true, build_immediate)
    }

    pub fn run_frame_with_options<F>(
        &mut self,
        mut input: UiInputSnapshot,
        hash_draw_commands: bool,
        build_immediate: F,
    ) -> UiFrameOutput
    where
        F: FnOnce(&mut UiContext),
    {
        input.viewport_size.x = input.viewport_size.x.max(1.0);
        input.viewport_size.y = input.viewport_size.y.max(1.0);

        self.update_interaction(&input);

        let previous_layout_rects = std::mem::take(&mut self.last_layout_rects);
        let previous_hit_rects = std::mem::take(&mut self.last_hit_rects);
        let mut immediate = UiContext::new(
            self.theme.clone(),
            self.interaction,
            previous_layout_rects,
            previous_hit_rects,
            std::mem::take(&mut self.immediate_bool_state),
            std::mem::take(&mut self.immediate_drag_value_state),
            UiDragInputSnapshot::default(),
        );
        build_immediate(&mut immediate);
        let immediate = immediate.finish();
        self.immediate_bool_state = immediate.state_bools;
        self.immediate_drag_value_state = immediate.drag_value_state;

        let mut roots = if self.retained.roots().is_empty() {
            immediate.roots
        } else {
            let mut retained_roots = self.retained.build_roots();
            retained_roots.extend(immediate.roots);
            retained_roots
        };

        self.draw_commands.clear();
        self.frame_layout_rects.clear();
        self.frame_hit_rects.clear();
        self.frame_paint_order.clear();
        self.layer_cursor = 0.80;
        self.hash_draw_commands_enabled = hash_draw_commands;
        self.frame_command_hash = if hash_draw_commands {
            FNV_OFFSET_BASIS
        } else {
            0
        };

        self.layout_and_emit(input.viewport_size, roots);

        std::mem::swap(&mut self.last_layout_rects, &mut self.frame_layout_rects);
        self.frame_layout_rects.clear();
        std::mem::swap(&mut self.last_hit_rects, &mut self.frame_hit_rects);
        self.frame_hit_rects.clear();
        std::mem::swap(&mut self.last_paint_order, &mut self.frame_paint_order);
        self.frame_paint_order.clear();
        let stale_threshold = self
            .last_layout_rects
            .len()
            .saturating_mul(2)
            .saturating_add(256);
        let should_prune_text_cache = self.text_metrics_cache.len() > stale_threshold;
        let should_prune_hash_cache = self.text_hash_cache.len() > stale_threshold;
        let should_prune_periodic = self.cache_prune_counter >= 29;
        if should_prune_text_cache || should_prune_hash_cache || should_prune_periodic {
            self.text_metrics_cache
                .retain(|id, _| self.last_layout_rects.contains_key(id));
            self.text_hash_cache
                .retain(|id, _| self.last_layout_rects.contains_key(id));
            self.cache_prune_counter = 0;
        } else {
            self.cache_prune_counter = self.cache_prune_counter.saturating_add(1);
        }

        UiFrameOutput {
            draw_commands: std::mem::take(&mut self.draw_commands),
            layout_rects: HashMap::new(),
            subtree_rects: HashMap::new(),
            paint_order: Vec::new(),
            command_hash: self.frame_command_hash,
            command_hash_valid: hash_draw_commands,
            interaction: self.interaction,
        }
    }

    fn update_interaction(&mut self, input: &UiInputSnapshot) {
        let hovered = input
            .pointer_position
            .and_then(|pointer| self.hit_test_interactive(pointer));

        let pointer_pressed = input.pointer_down && !self.pointer_down_previous;
        let pointer_released = !input.pointer_down && self.pointer_down_previous;

        if pointer_pressed {
            self.active_id = hovered;
        }

        let mut clicked = None;
        if pointer_released {
            if self.active_id.is_some() && self.active_id == hovered {
                clicked = self.active_id;
            }
            self.active_id = None;
        }

        let active = if input.pointer_down {
            self.active_id
        } else {
            None
        };

        self.interaction = UiInteractionState {
            hovered,
            active,
            clicked,
            pointer_captured: hovered.is_some() || active.is_some(),
        };

        self.pointer_down_previous = input.pointer_down;
    }

    fn layout_and_emit(&mut self, viewport: Vec2, roots: Vec<UiNode>) {
        TAFFY_SCRATCH.with(|taffy_scratch| {
            let mut taffy = taffy_scratch.borrow_mut();
            taffy.clear();

            let mut built_roots = Vec::with_capacity(roots.len());
            let mut root_child_ids = Vec::with_capacity(roots.len());
            for node in &roots {
                let built_node = self.build_layout_tree(&mut taffy, node);
                root_child_ids.push(built_node.taffy_node);
                built_roots.push(built_node);
            }

            let mut root_style = Style::default();
            root_style.display = Display::Flex;
            root_style.flex_direction = FlexDirection::Column;
            root_style.size = Size {
                width: Dimension::length(viewport.x),
                height: Dimension::length(viewport.y),
            };
            root_style.position = Position::Relative;

            let Ok(root_node) = taffy.new_with_children(root_style, &root_child_ids) else {
                return;
            };

            if taffy
                .compute_layout_with_measure(
                    root_node,
                    Size {
                        width: AvailableSpace::Definite(viewport.x),
                        height: AvailableSpace::Definite(viewport.y),
                    },
                    |known_dimensions, available_space, _node_id, node_context, _style| {
                        measure_node(known_dimensions, available_space, node_context)
                    },
                )
                .is_err()
            {
                return;
            }

            let Ok(root_layout) = taffy.layout(root_node).copied() else {
                return;
            };

            let root_offset = Vec2::new(root_layout.location.x, root_layout.location.y);
            let root_clip = Some(UiRect {
                x: root_offset.x,
                y: root_offset.y,
                width: viewport.x,
                height: viewport.y,
            });

            for (node, built_node) in roots.into_iter().zip(built_roots.into_iter()) {
                self.emit_node(&taffy, node, built_node, root_offset, root_clip);
            }
        });
    }

    fn build_layout_tree(
        &mut self,
        taffy: &mut TaffyTree<LayoutMeasureContext>,
        node: &UiNode,
    ) -> LayoutTreeNode {
        let requires_measure = !is_definite_points_dimension(node.style.layout.width)
            || !is_definite_points_dimension(node.style.layout.height);
        let measure_context = match &node.widget {
            UiWidget::Label(label) if label.style.wrap || requires_measure => {
                Some(self.make_text_measure_context(
                    node.id,
                    label.text.as_str(),
                    label.style.font_size,
                    label.style.wrap,
                ))
            }
            UiWidget::TextField(field) if requires_measure => Some(self.make_text_measure_context(
                node.id,
                field.text.as_str(),
                field.style.font_size,
                false,
            )),
            _ => None,
        };
        let mut child_nodes = Vec::with_capacity(node.children.len());
        let mut child_ids = Vec::with_capacity(node.children.len());
        for child in &node.children {
            let child_node = self.build_layout_tree(taffy, child);
            child_ids.push(child_node.taffy_node);
            child_nodes.push(child_node);
        }

        let mut layout_style = to_taffy_style(&node.style.layout);
        if node.style.visual.clip {
            layout_style.overflow = taffy::geometry::Point {
                x: Overflow::Hidden,
                y: Overflow::Hidden,
            };
        }
        let taffy_node = if child_ids.is_empty() {
            taffy
                .new_leaf(layout_style)
                .unwrap_or_else(|_| taffy.new_leaf(Style::default()).expect("taffy leaf"))
        } else {
            taffy
                .new_with_children(layout_style, &child_ids)
                .unwrap_or_else(|_| {
                    taffy
                        .new_with_children(Style::default(), &child_ids)
                        .expect("taffy branch")
                })
        };

        if let Some(context) = measure_context {
            let _ = taffy.set_node_context(taffy_node, Some(context));
        }

        LayoutTreeNode {
            taffy_node,
            children: child_nodes,
        }
    }

    fn make_text_measure_context(
        &mut self,
        id: UiId,
        text: &str,
        font_size: f32,
        wrap: bool,
    ) -> LayoutMeasureContext {
        let font_size = font_size.max(1.0);
        let min_char_width = estimate_char_advance('0', font_size).max(font_size * 0.25);

        let text_changed = !self
            .text_metrics_cache
            .get(&id)
            .filter(|cached| {
                cached.wrap == wrap
                    && (cached.font_size - font_size).abs() <= f32::EPSILON
                    && cached.text.as_ref() == text
            })
            .is_some();

        if text_changed {
            let text_arc: Arc<str> = Arc::<str>::from(text);
            let line_count = text.lines().count().max(1) as f32;
            let unwrapped_width = text
                .lines()
                .map(|line| estimate_text_width(line, font_size))
                .fold(0.0f32, f32::max)
                .max(min_char_width);
            self.text_metrics_cache.insert(
                id,
                CachedTextMetrics {
                    text: text_arc,
                    font_size,
                    wrap,
                    line_count,
                    unwrapped_width,
                    cached_wrap_width: 0.0,
                    cached_wrapped_lines: line_count.max(1.0),
                    has_wrapped_cache: false,
                },
            );
        }

        let predicted_wrap_from_last_layout = if wrap {
            self.last_layout_rects
                .get(&id)
                .map(|rect| rect.width.max(min_char_width))
        } else {
            None
        };

        let cached = self
            .text_metrics_cache
            .get_mut(&id)
            .expect("text metrics cache entry must exist");

        if wrap {
            // use last frame's width as the predicted wrapping width so wrapped line counting is stable and amortized across frames instead of being recomputed every frame
            let predicted_wrap_width = predicted_wrap_from_last_layout
                .unwrap_or(cached.unwrapped_width)
                .max(min_char_width);
            if !cached.has_wrapped_cache
                || (cached.cached_wrap_width - predicted_wrap_width).abs() > 0.5
            {
                let wrapped = cached
                    .text
                    .lines()
                    .map(|line| estimate_wrapped_line_count(line, predicted_wrap_width, font_size))
                    .sum::<f32>()
                    .max(1.0);
                cached.cached_wrap_width = predicted_wrap_width;
                cached.cached_wrapped_lines = wrapped;
                cached.has_wrapped_cache = true;
            }
        }

        LayoutMeasureContext::Text(TextMeasureContext {
            text: if wrap {
                Some(cached.text.clone())
            } else {
                None
            },
            font_size,
            wrap,
            line_count: cached.line_count.max(1.0),
            unwrapped_width: cached.unwrapped_width.max(min_char_width),
            cached_wrap_width: cached.cached_wrap_width,
            cached_wrapped_lines: cached.cached_wrapped_lines.max(1.0),
            has_wrapped_cache: cached.has_wrapped_cache,
        })
    }

    fn emit_node(
        &mut self,
        taffy: &TaffyTree<LayoutMeasureContext>,
        ui_node: UiNode,
        node: LayoutTreeNode,
        parent_offset: Vec2,
        parent_clip: Option<UiRect>,
    ) {
        let UiNode {
            id,
            widget,
            style,
            enabled,
            children,
        } = ui_node;

        let layout = match taffy.layout(node.taffy_node) {
            Ok(layout) => *layout,
            Err(_) => return,
        };

        let rect = UiRect {
            x: parent_offset.x + layout.location.x,
            y: parent_offset.y + layout.location.y,
            width: layout.size.width.max(0.0),
            height: layout.size.height.max(0.0),
        };

        self.frame_layout_rects.insert(id, rect);

        let clip_rect = if style.visual.clip {
            match parent_clip {
                Some(parent) => parent.intersection(rect),
                None => Some(rect),
            }
        } else {
            parent_clip
        };

        let is_interactive = matches!(
            &widget,
            UiWidget::Button(_)
                | UiWidget::Disclosure(_)
                | UiWidget::TextField(_)
                | UiWidget::Image(_)
        );
        if is_interactive {
            let hit_rect = match clip_rect {
                Some(clip) => rect.intersection(clip),
                None => Some(rect),
            };
            if let Some(hit_rect) = hit_rect {
                self.frame_hit_rects.insert(id, hit_rect);
            }
        }

        // cull fully clipped subtrees before emission to avoid generating large offscreen command buffers (especially for scrollable windows with many rows)
        if let Some(clip) = clip_rect
            && rect.intersection(clip).is_none()
        {
            return;
        }

        self.emit_widget(id, enabled, widget, style, rect, clip_rect);

        let offset = Vec2::new(rect.x, rect.y);
        debug_assert_eq!(children.len(), node.children.len());
        for (child, child_layout) in children.into_iter().zip(node.children.into_iter()) {
            self.emit_node(taffy, child, child_layout, offset, clip_rect);
        }
    }

    fn emit_widget(
        &mut self,
        id: UiId,
        enabled: bool,
        widget: UiWidget,
        style: crate::UiStyle,
        rect: UiRect,
        clip_rect: Option<UiRect>,
    ) {
        match widget {
            UiWidget::Container => {
                if let Some(background) = style.visual.background {
                    self.push_rect(id, rect, background, clip_rect);
                }
                self.push_border(id, rect, style.visual, clip_rect);
            }
            UiWidget::Label(label) => {
                self.push_text(
                    id,
                    rect,
                    label.text,
                    label.style.color,
                    label.style.font_size,
                    label.style.align_h,
                    label.style.align_v,
                    label.style.wrap,
                    clip_rect,
                    None,
                    false,
                    None,
                    None,
                    None,
                );
            }
            UiWidget::Button(button) => {
                self.frame_paint_order.push(id);
                let mut fill = match button.variant {
                    UiButtonVariant::Primary => self.theme.button_primary,
                    UiButtonVariant::Secondary => self.theme.button_secondary,
                    UiButtonVariant::Danger => self.theme.button_danger,
                    UiButtonVariant::Ghost => self.theme.group_background,
                };
                let mut visual = style.visual;
                let mut border_color = visual
                    .border_color
                    .unwrap_or(self.theme.panel_border.with_alpha(0.85));
                let mut border_width = visual.border_width.max(self.theme.border_width);

                if !button.enabled || !enabled {
                    fill = self.theme.button_disabled;
                    border_color = self.theme.panel_border.with_alpha(0.7);
                } else if self.interaction.is_active(id) {
                    fill = Self::mix(fill, self.theme.button_pressed_overlay);
                    border_color = self.theme.button_text.with_alpha(0.92);
                    border_width = border_width.max(self.theme.border_width + 1.0);
                } else if self.interaction.is_hovered(id) {
                    fill = Self::mix(fill, self.theme.button_hover_overlay);
                    border_color = self.theme.button_text.with_alpha(0.68);
                    border_width = border_width.max(self.theme.border_width + 0.4);
                }

                visual.border_color = Some(border_color);
                visual.border_width = border_width;
                self.push_rect(id, rect, fill, clip_rect);
                self.push_border(id, rect, visual, clip_rect);
                self.push_text(
                    id,
                    rect,
                    button.text,
                    button.style.color,
                    button.style.font_size,
                    button.style.align_h,
                    button.style.align_v,
                    button.style.wrap,
                    clip_rect,
                    None,
                    false,
                    None,
                    None,
                    None,
                );
            }
            UiWidget::Disclosure(disclosure) => {
                self.frame_paint_order.push(id);
                let mut fill = if disclosure.expanded {
                    Self::mix(self.theme.group_background, self.theme.button_hover_overlay)
                } else {
                    self.theme.group_background
                };
                let mut visual = style.visual;
                let mut border_color = visual
                    .border_color
                    .unwrap_or(self.theme.panel_border.with_alpha(0.85));
                let mut border_width = visual.border_width.max(self.theme.border_width);

                if !disclosure.enabled || !enabled {
                    fill = self.theme.button_disabled;
                    border_color = self.theme.panel_border.with_alpha(0.7);
                } else if self.interaction.is_active(id) {
                    fill = Self::mix(fill, self.theme.button_pressed_overlay);
                    border_color = self.theme.button_text.with_alpha(0.90);
                    border_width = border_width.max(self.theme.border_width + 1.0);
                } else if self.interaction.is_hovered(id) {
                    fill = Self::mix(fill, self.theme.button_hover_overlay);
                    border_color = self.theme.button_text.with_alpha(0.65);
                    border_width = border_width.max(self.theme.border_width + 0.4);
                }

                visual.border_color = Some(border_color);
                visual.border_width = border_width;
                self.push_rect(id, rect, fill, clip_rect);
                self.push_border(id, rect, visual, clip_rect);

                let glyph_rect = UiRect {
                    x: rect.x + self.theme.item_padding.max(3.0),
                    y: rect.y,
                    width: (disclosure.style.font_size * 1.1).max(10.0),
                    height: rect.height,
                };
                let glyph_clip = match clip_rect {
                    Some(clip) => clip.intersection(glyph_rect),
                    None => Some(glyph_rect),
                };
                if glyph_clip.is_some() {
                    self.push_text(
                        id,
                        glyph_rect,
                        if disclosure.expanded {
                            UiTextValue::from("v")
                        } else {
                            UiTextValue::from(">")
                        },
                        disclosure.style.color,
                        disclosure.style.font_size,
                        UiTextAlign::Start,
                        UiTextAlign::Center,
                        false,
                        glyph_clip,
                        None,
                        false,
                        None,
                        None,
                        None,
                    );
                }

                let text_offset = (self.theme.item_padding.max(3.0) * 2.0)
                    + (disclosure.style.font_size * 1.1).max(10.0);
                let text_rect = UiRect {
                    x: (rect.x + text_offset).min(rect.right()),
                    y: rect.y,
                    width: (rect.width - text_offset).max(0.0),
                    height: rect.height,
                };
                let text_clip = match clip_rect {
                    Some(clip) => clip.intersection(text_rect),
                    None => Some(text_rect),
                };
                if text_clip.is_none() {
                    return;
                }
                self.push_text(
                    id,
                    text_rect,
                    disclosure.text,
                    disclosure.style.color,
                    disclosure.style.font_size,
                    UiTextAlign::Start,
                    disclosure.style.align_v,
                    disclosure.style.wrap,
                    text_clip,
                    None,
                    false,
                    None,
                    None,
                    None,
                );
            }
            UiWidget::TextField(field) => {
                self.frame_paint_order.push(id);
                let mut fill = self.theme.button_secondary;
                let mut visual = style.visual;
                let mut border_color = visual
                    .border_color
                    .unwrap_or(self.theme.panel_border.with_alpha(0.9));
                let mut border_width = visual.border_width.max(self.theme.border_width);

                if !field.enabled || !enabled {
                    fill = self.theme.button_disabled;
                    border_color = self.theme.panel_border.with_alpha(0.7);
                } else if self.interaction.is_active(id) {
                    fill = Self::mix(fill, self.theme.button_pressed_overlay);
                    border_color = self.theme.button_text.with_alpha(0.95);
                    border_width = border_width.max(self.theme.border_width + 0.9);
                } else if self.interaction.is_hovered(id) {
                    fill = Self::mix(fill, self.theme.button_hover_overlay);
                    border_color = self.theme.button_text.with_alpha(0.72);
                    border_width = border_width.max(self.theme.border_width + 0.35);
                }
                if field.focused {
                    fill = Self::mix(fill, self.theme.button_hover_overlay.with_alpha(0.14));
                    border_color = self.theme.button_text.with_alpha(0.96);
                    border_width = border_width.max(self.theme.border_width + 0.8);
                }

                visual.border_color = Some(border_color);
                visual.border_width = border_width;
                self.push_rect(id, rect, fill, clip_rect);
                self.push_border(id, rect, visual, clip_rect);

                let text_padding_x = self.theme.item_padding.max(4.0);
                let text_left = rect.x + text_padding_x;
                let text_right = rect.right() - text_padding_x;
                let mut text_rect = UiRect {
                    x: text_left,
                    y: rect.y,
                    width: (text_right - text_left).max(0.0),
                    height: rect.height,
                };

                let mut suffix_rect = None;
                if let Some(suffix) = field.suffix.as_ref() {
                    let suffix_width =
                        estimate_text_width(suffix.as_str(), field.style.font_size).max(0.0);
                    let suffix_x = (text_right - suffix_width).max(text_left);
                    let suffix_area_spacing = text_padding_x * 0.6;
                    text_rect.width = (suffix_x - text_left - suffix_area_spacing).max(0.0);
                    suffix_rect = Some(UiRect {
                        x: suffix_x,
                        y: rect.y,
                        width: (text_right - suffix_x).max(0.0),
                        height: rect.height,
                    });
                }
                let text_clip = match clip_rect {
                    Some(clip) => clip.intersection(text_rect),
                    None => Some(text_rect),
                };
                let scroll_x = field.scroll_x.max(0.0);
                let text_draw_rect = UiRect {
                    x: text_rect.x - scroll_x,
                    y: text_rect.y,
                    width: text_rect.width + scroll_x,
                    height: text_rect.height,
                };
                if text_clip.is_some() {
                    self.push_text(
                        id,
                        text_draw_rect,
                        field.text,
                        field.style.color,
                        field.style.font_size,
                        UiTextAlign::Start,
                        UiTextAlign::Center,
                        false,
                        text_clip,
                        field.cursor,
                        field.focused && field.show_caret,
                        Some(field.caret_color),
                        field.selection,
                        Some(field.selection_color),
                    );
                }

                if let (Some(suffix), Some(suffix_rect)) = (field.suffix, suffix_rect) {
                    let suffix_clip = match clip_rect {
                        Some(clip) => clip.intersection(suffix_rect),
                        None => Some(suffix_rect),
                    };
                    if suffix_clip.is_some() {
                        self.push_text(
                            id,
                            suffix_rect,
                            suffix,
                            self.theme.muted_text.with_alpha(0.95),
                            field.style.font_size,
                            UiTextAlign::End,
                            UiTextAlign::Center,
                            false,
                            suffix_clip,
                            None,
                            false,
                            None,
                            None,
                            None,
                        );
                    }
                }
            }
            UiWidget::Image(image) => {
                self.frame_paint_order.push(id);
                self.push_image(
                    id,
                    rect,
                    image.texture_id,
                    image.tint,
                    image.uv_min,
                    image.uv_max,
                    clip_rect,
                );
                self.push_border(id, rect, style.visual, clip_rect);
            }
            UiWidget::Spacer => {}
        }
    }

    fn push_rect(&mut self, id: UiId, rect: UiRect, color: UiColor, clip: Option<UiRect>) {
        if rect.width <= 0.0 || rect.height <= 0.0 || color.a <= 0.0 {
            return;
        }
        let layer = self.next_layer();
        let command = UiDrawCommand::Rect(UiRectCommand {
            id,
            rect,
            color,
            clip,
            layer,
        });
        self.hash_draw_command(&command);
        self.draw_commands.push(command);
    }

    fn push_image(
        &mut self,
        id: UiId,
        rect: UiRect,
        texture_id: Option<usize>,
        tint: UiColor,
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        clip: Option<UiRect>,
    ) {
        if rect.width <= 0.0 || rect.height <= 0.0 || tint.a <= 0.0 {
            return;
        }
        let layer = self.next_layer();
        let command = UiDrawCommand::Image(UiImageCommand {
            id,
            rect,
            texture_id,
            tint,
            uv_min,
            uv_max,
            clip,
            layer,
        });
        self.hash_draw_command(&command);
        self.draw_commands.push(command);
    }

    fn push_border(
        &mut self,
        id: UiId,
        rect: UiRect,
        style: crate::UiVisualStyle,
        clip: Option<UiRect>,
    ) {
        let color = match style.border_color {
            Some(color) => color,
            None => return,
        };
        let w = style.border_width.max(0.0);
        if w <= 0.0 || rect.width <= 0.0 || rect.height <= 0.0 {
            return;
        }

        let top = UiRect {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: w,
        };
        let bottom = UiRect {
            x: rect.x,
            y: (rect.bottom() - w).max(rect.y),
            width: rect.width,
            height: w.min(rect.height),
        };
        let left = UiRect {
            x: rect.x,
            y: rect.y,
            width: w,
            height: rect.height,
        };
        let right = UiRect {
            x: (rect.right() - w).max(rect.x),
            y: rect.y,
            width: w.min(rect.width),
            height: rect.height,
        };

        self.push_rect(id, top, color, clip);
        self.push_rect(id, bottom, color, clip);
        self.push_rect(id, left, color, clip);
        self.push_rect(id, right, color, clip);
    }

    fn push_text(
        &mut self,
        id: UiId,
        rect: UiRect,
        text: UiTextValue,
        color: UiColor,
        font_size: f32,
        align_h: UiTextAlign,
        align_v: UiTextAlign,
        wrap: bool,
        clip: Option<UiRect>,
        cursor: Option<usize>,
        show_caret: bool,
        caret_color: Option<UiColor>,
        selection: Option<(usize, usize)>,
        selection_color: Option<UiColor>,
    ) {
        if color.a <= 0.0 || (text.is_empty() && !show_caret && selection.is_none()) {
            return;
        }
        let layer = self.next_layer();
        let command = UiDrawCommand::Text(UiTextCommand {
            id,
            rect,
            text,
            color,
            font_size,
            align_h,
            align_v,
            wrap,
            cursor,
            show_caret,
            caret_color,
            selection,
            selection_color,
            clip,
            layer,
        });
        self.hash_draw_command(&command);
        self.draw_commands.push(command);
    }

    fn next_layer(&mut self) -> f32 {
        let layer = self.layer_cursor;
        self.layer_cursor += 0.000_05;
        layer
    }

    fn mix(base: UiColor, overlay: UiColor) -> UiColor {
        let alpha = overlay.a.clamp(0.0, 1.0);
        UiColor::rgba(
            base.r * (1.0 - alpha) + overlay.r * alpha,
            base.g * (1.0 - alpha) + overlay.g * alpha,
            base.b * (1.0 - alpha) + overlay.b * alpha,
            base.a,
        )
    }

    fn hash_draw_command(&mut self, command: &UiDrawCommand) {
        if !self.hash_draw_commands_enabled {
            return;
        }
        match command {
            UiDrawCommand::Rect(rect) => {
                hash_u64(&mut self.frame_command_hash, 1);
                hash_u64(&mut self.frame_command_hash, rect.id.0);
                hash_rect(&mut self.frame_command_hash, rect.rect);
                hash_color(&mut self.frame_command_hash, rect.color.to_array());
                hash_clip_rect(&mut self.frame_command_hash, rect.clip);
                hash_f32(&mut self.frame_command_hash, rect.layer);
            }
            UiDrawCommand::Image(image) => {
                hash_u64(&mut self.frame_command_hash, 2);
                hash_u64(&mut self.frame_command_hash, image.id.0);
                hash_rect(&mut self.frame_command_hash, image.rect);
                hash_u64(
                    &mut self.frame_command_hash,
                    image.texture_id.map(|texture| texture as u64).unwrap_or(0),
                );
                hash_color(&mut self.frame_command_hash, image.tint.to_array());
                hash_f32(&mut self.frame_command_hash, image.uv_min[0]);
                hash_f32(&mut self.frame_command_hash, image.uv_min[1]);
                hash_f32(&mut self.frame_command_hash, image.uv_max[0]);
                hash_f32(&mut self.frame_command_hash, image.uv_max[1]);
                hash_clip_rect(&mut self.frame_command_hash, image.clip);
                hash_f32(&mut self.frame_command_hash, image.layer);
            }
            UiDrawCommand::Text(text) => {
                hash_u64(&mut self.frame_command_hash, 3);
                hash_u64(&mut self.frame_command_hash, text.id.0);
                hash_rect(&mut self.frame_command_hash, text.rect);
                let cached_text_hash = self.cached_text_hash(text.id, &text.text);
                hash_u64(&mut self.frame_command_hash, cached_text_hash);
                hash_color(&mut self.frame_command_hash, text.color.to_array());
                hash_f32(&mut self.frame_command_hash, text.font_size);
                hash_u64(&mut self.frame_command_hash, text.align_h as u64);
                hash_u64(&mut self.frame_command_hash, text.align_v as u64);
                hash_u64(&mut self.frame_command_hash, text.wrap as u64);
                hash_u64(
                    &mut self.frame_command_hash,
                    text.cursor.unwrap_or(usize::MAX) as u64,
                );
                hash_u64(&mut self.frame_command_hash, text.show_caret as u64);
                hash_optional_color(&mut self.frame_command_hash, text.caret_color);
                match text.selection {
                    Some((start, end)) => {
                        hash_u64(&mut self.frame_command_hash, 1);
                        hash_u64(&mut self.frame_command_hash, start as u64);
                        hash_u64(&mut self.frame_command_hash, end as u64);
                    }
                    None => hash_u64(&mut self.frame_command_hash, 0),
                }
                hash_optional_color(&mut self.frame_command_hash, text.selection_color);
                hash_clip_rect(&mut self.frame_command_hash, text.clip);
                hash_f32(&mut self.frame_command_hash, text.layer);
            }
        }
    }

    fn cached_text_hash(&mut self, id: UiId, text: &UiTextValue) -> u64 {
        if let Some(cached) = self
            .text_hash_cache
            .get(&id)
            .filter(|cached| cached.text == *text)
        {
            return cached.hash;
        }

        let mut hash = FNV_OFFSET_BASIS;
        hash_bytes(&mut hash, text.as_str().as_bytes());
        self.text_hash_cache.insert(
            id,
            CachedTextHash {
                text: text.clone(),
                hash,
            },
        );
        hash
    }
}

fn to_taffy_style(layout: &UiLayoutStyle) -> Style {
    let mut style = Style::default();
    style.display = match layout.display {
        UiDisplay::Flex => Display::Flex,
        UiDisplay::None => Display::None,
    };
    style.flex_direction = match layout.flex_direction {
        UiFlexDirection::Row => FlexDirection::Row,
        UiFlexDirection::Column => FlexDirection::Column,
    };
    style.flex_wrap = match layout.flex_wrap {
        UiFlexWrap::NoWrap => FlexWrap::NoWrap,
        UiFlexWrap::Wrap => FlexWrap::Wrap,
    };
    style.position = match layout.position_type {
        UiPositionType::Relative => Position::Relative,
        UiPositionType::Absolute => Position::Absolute,
    };
    style.align_items = Some(match layout.align_items {
        UiAlignItems::Start => AlignItems::FlexStart,
        UiAlignItems::Center => AlignItems::Center,
        UiAlignItems::End => AlignItems::FlexEnd,
        UiAlignItems::Stretch => AlignItems::Stretch,
    });
    style.inset = TaffyRect {
        left: to_length_auto(layout.left),
        right: to_length_auto(layout.right),
        top: to_length_auto(layout.top),
        bottom: to_length_auto(layout.bottom),
    };
    style.size = Size {
        width: to_dimension(layout.width),
        height: to_dimension(layout.height),
    };
    style.padding = TaffyRect {
        left: LengthPercentage::length(layout.padding.left.max(0.0)),
        right: LengthPercentage::length(layout.padding.right.max(0.0)),
        top: LengthPercentage::length(layout.padding.top.max(0.0)),
        bottom: LengthPercentage::length(layout.padding.bottom.max(0.0)),
    };
    style.margin = TaffyRect {
        left: LengthPercentageAuto::length(layout.margin.left.max(0.0)),
        right: LengthPercentageAuto::length(layout.margin.right.max(0.0)),
        top: LengthPercentageAuto::length(layout.margin.top.max(0.0)),
        bottom: LengthPercentageAuto::length(layout.margin.bottom.max(0.0)),
    };
    style.gap = Size {
        width: length(layout.gap.x.max(0.0)),
        height: length(layout.gap.y.max(0.0)),
    };
    style.flex_grow = layout.flex_grow.max(0.0);
    style.flex_shrink = layout.flex_shrink.max(0.0);
    style.overflow = taffy::geometry::Point {
        x: Overflow::Visible,
        y: Overflow::Visible,
    };
    style
}

fn is_definite_points_dimension(dimension: UiDimension) -> bool {
    matches!(dimension, UiDimension::Points(_))
}

fn to_dimension(dimension: UiDimension) -> Dimension {
    match dimension {
        UiDimension::Auto => Dimension::auto(),
        UiDimension::Points(value) => Dimension::length(value.max(0.0)),
        UiDimension::Percent(value) => Dimension::percent(value),
    }
}

fn to_length_auto(dimension: UiDimension) -> LengthPercentageAuto {
    match dimension {
        UiDimension::Auto => LengthPercentageAuto::auto(),
        UiDimension::Points(value) => LengthPercentageAuto::length(value),
        UiDimension::Percent(value) => LengthPercentageAuto::percent(value),
    }
}

fn measure_node(
    known_dimensions: Size<Option<f32>>,
    available_space: Size<AvailableSpace>,
    node_context: Option<&mut LayoutMeasureContext>,
) -> Size<f32> {
    if let (Some(width), Some(height)) = (known_dimensions.width, known_dimensions.height) {
        return Size { width, height };
    }

    let Some(node_context) = node_context else {
        return known_dimensions.unwrap_or(Size::ZERO);
    };

    let measured = match node_context {
        LayoutMeasureContext::Text(text) => measure_text(text, known_dimensions, available_space),
    };
    known_dimensions.unwrap_or(measured)
}

fn measure_text(
    text: &mut TextMeasureContext,
    known_dimensions: Size<Option<f32>>,
    available_space: Size<AvailableSpace>,
) -> Size<f32> {
    let font_size = text.font_size.max(1.0);
    let min_char_width = estimate_char_advance('0', font_size).max(font_size * 0.25);
    let line_height = (font_size * 1.3).max(font_size);
    let line_count = text.line_count.max(1.0);
    let unwrapped_width = text.unwrapped_width.max(min_char_width);

    if !text.wrap {
        return Size {
            width: unwrapped_width,
            height: (line_count * line_height).max(line_height),
        };
    }

    let wrap_width = known_dimensions
        .width
        .or_else(|| definite_size(available_space.width))
        .unwrap_or(unwrapped_width)
        .max(min_char_width);

    let wrapped_lines =
        if text.has_wrapped_cache && (text.cached_wrap_width - wrap_width).abs() <= 0.5 {
            text.cached_wrapped_lines
        } else {
            let wrapped = text
                .text
                .as_deref()
                .unwrap_or("")
                .lines()
                .map(|line| estimate_wrapped_line_count(line, wrap_width, font_size))
                .sum::<f32>()
                .max(1.0);
            text.cached_wrap_width = wrap_width;
            text.cached_wrapped_lines = wrapped;
            text.has_wrapped_cache = true;
            wrapped
        };

    Size {
        width: wrap_width,
        height: (wrapped_lines * line_height).max(line_height),
    }
}

fn estimate_wrapped_line_count(line: &str, wrap_width: f32, font_size: f32) -> f32 {
    if line.trim().is_empty() {
        return 1.0;
    }

    let space_width = estimate_char_advance(' ', font_size);
    let mut lines = 1.0f32;
    let mut current_width = 0.0f32;
    for word in line.split_whitespace() {
        let word_width = estimate_text_width(word, font_size).max(0.0);
        if current_width <= f32::EPSILON {
            if word_width <= wrap_width {
                current_width = word_width;
            } else {
                let mut segment_width = 0.0f32;
                for ch in word.chars() {
                    let ch_width = estimate_char_advance(ch, font_size).max(0.0);
                    if segment_width > 0.0 && segment_width + ch_width > wrap_width {
                        lines += 1.0;
                        segment_width = ch_width;
                    } else {
                        segment_width += ch_width;
                    }
                }
                current_width = segment_width;
            }
            continue;
        }

        let candidate = current_width + space_width + word_width;
        if candidate <= wrap_width {
            current_width = candidate;
            continue;
        }

        lines += 1.0;
        if word_width <= wrap_width {
            current_width = word_width;
        } else {
            let mut segment_width = 0.0f32;
            for ch in word.chars() {
                let ch_width = estimate_char_advance(ch, font_size).max(0.0);
                if segment_width > 0.0 && segment_width + ch_width > wrap_width {
                    lines += 1.0;
                    segment_width = ch_width;
                } else {
                    segment_width += ch_width;
                }
            }
            current_width = segment_width;
        }
    }

    lines.max(1.0)
}

fn definite_size(space: AvailableSpace) -> Option<f32> {
    match space {
        AvailableSpace::Definite(value) => Some(value.max(0.0)),
        _ => None,
    }
}

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn hash_rect(hash: &mut u64, rect: UiRect) {
    hash_f32(hash, rect.x);
    hash_f32(hash, rect.y);
    hash_f32(hash, rect.width);
    hash_f32(hash, rect.height);
}

fn hash_color(hash: &mut u64, color: [f32; 4]) {
    for value in color {
        hash_f32(hash, value);
    }
}

fn hash_optional_color(hash: &mut u64, color: Option<UiColor>) {
    match color {
        Some(color) => {
            hash_u64(hash, 1);
            hash_color(hash, color.to_array());
        }
        None => hash_u64(hash, 0),
    }
}

fn hash_clip_rect(hash: &mut u64, clip: Option<UiRect>) {
    match clip {
        Some(clip) => {
            hash_u64(hash, 1);
            hash_rect(hash, clip);
        }
        None => hash_u64(hash, 0),
    }
}

fn hash_f32(hash: &mut u64, value: f32) {
    hash_u64(hash, value.to_bits() as u64);
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    hash_u64(hash, bytes.len() as u64);
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        hash_u64(hash, u64::from_le_bytes(buf));
    }
    for byte in chunks.remainder() {
        hash_u64(hash, *byte as u64);
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(FNV_PRIME);
}
