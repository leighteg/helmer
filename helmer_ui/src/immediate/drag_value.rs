use std::time::{Duration, Instant};

use glam::Vec2;
use hashbrown::HashMap;

use crate::{
    IntoUiId, UiId, UiRect, UiTextAlign, UiTextStyle, UiTextValue, estimate_text_prefix_width,
    estimate_text_width,
};

use super::{UiContext, UiResponse, Widget};

#[derive(Clone, Debug, Default)]
pub struct UiDragInputSnapshot {
    pub pointer_down: bool,
    pub pointer_pressed: bool,
    pub pointer_released: bool,
    pub cursor_pos: Vec2,
    pub just_pressed: Vec<UiKeyInput>,
    pub ctrl: bool,
    pub shift: bool,
    pub clipboard_text: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiKeyInput {
    Enter,
    Escape,
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    Home,
    End,
    Backspace,
    Delete,
    SelectAll,
    Copy,
    Cut,
    Paste,
    Char(char),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiDragDisplay {
    Scalar,
    Degrees,
}

#[derive(Clone, Debug)]
pub struct UiDragValueConfig {
    label: UiTextValue,
    step: f32,
    unit_suffix: Option<UiTextValue>,
    display: UiDragDisplay,
}

impl UiDragValueConfig {
    pub fn new<S: Into<UiTextValue>>(label: S) -> Self {
        Self {
            label: label.into(),
            step: 0.1,
            unit_suffix: None,
            display: UiDragDisplay::Scalar,
        }
    }

    pub fn step(mut self, step: f32) -> Self {
        self.step = step;
        self
    }

    pub fn suffix<S: Into<UiTextValue>>(mut self, suffix: S) -> Self {
        self.unit_suffix = Some(suffix.into());
        self
    }

    pub fn degrees(mut self) -> Self {
        self.display = UiDragDisplay::Degrees;
        self.unit_suffix = Some(UiTextValue::from("°"));
        self
    }

    pub fn display(mut self, display: UiDragDisplay) -> Self {
        self.display = display;
        self
    }
}

#[derive(Default)]
pub struct UiDragValueState {
    pending_drag: Option<PendingDragSession>,
    active_drag: Option<DragSession>,
    active_edit: Option<EditSession>,
    last_click: Option<(UiId, Instant)>,
    display_text_cache: HashMap<UiId, DragDisplayTextCache>,
    clipboard_text: String,
    clipboard_write: Option<String>,
}

struct DragDisplayTextCache {
    quantized_value: i64,
    text: std::sync::Arc<str>,
}

struct PendingDragSession {
    id: UiId,
    start_cursor_x: f32,
    start_value: f32,
}

struct DragSession {
    id: UiId,
    start_cursor_x: f32,
    start_value: f32,
}

struct EditSession {
    id: UiId,
    buffer: String,
    cursor_chars: usize,
    selection_anchor_chars: Option<usize>,
    pointer_select_anchor_chars: Option<usize>,
    horizontal_scroll_x: f32,
    started_at: Instant,
}

impl UiDragValueState {
    pub fn set_clipboard_text<S: Into<String>>(&mut self, text: S) {
        self.clipboard_text = text.into();
    }

    pub fn take_clipboard_write(&mut self) -> Option<String> {
        self.clipboard_write.take()
    }
}

impl UiContext {
    pub fn drag_angle(&mut self, radians: &mut f32) -> UiResponse {
        let id = self.next_auto_id();
        self.drag_angle_with_id(id, radians)
    }

    pub fn drag_angle_with_id<K: IntoUiId>(&mut self, key: K, radians: &mut f32) -> UiResponse {
        let id = self.derive_id(key);
        self.run_drag_value_internal(
            id,
            UiTextValue::default(),
            radians,
            1.0,
            Some(UiTextValue::from("°")),
            radians_to_degrees,
            degrees_to_radians,
        )
    }

    pub fn drag_value_f32<S: Into<UiTextValue>>(
        &mut self,
        label: S,
        value: &mut f32,
        step: f32,
    ) -> UiResponse {
        let id = self.next_auto_id();
        self.drag_value_f32_with_id(id, label, value, step)
    }

    pub fn drag_value_f32_with_id<K: IntoUiId, S: Into<UiTextValue>>(
        &mut self,
        key: K,
        label: S,
        value: &mut f32,
        step: f32,
    ) -> UiResponse {
        let id = self.derive_id(key);
        self.run_drag_value_internal(
            id,
            label.into(),
            value,
            step,
            None,
            scalar_identity,
            scalar_identity,
        )
    }

    pub fn drag_value(&mut self, value: &mut f32, config: UiDragValueConfig) -> UiResponse {
        let id = self.next_auto_id();
        self.drag_value_with_id(id, value, config)
    }

    pub fn drag_value_with_id<K: IntoUiId>(
        &mut self,
        key: K,
        value: &mut f32,
        config: UiDragValueConfig,
    ) -> UiResponse {
        let id = self.derive_id(key);
        let UiDragValueConfig {
            label,
            step,
            unit_suffix,
            display,
        } = config;
        let (to_display_value, from_display_value): (fn(f32) -> f32, fn(f32) -> f32) = match display
        {
            UiDragDisplay::Scalar => (scalar_identity, scalar_identity),
            UiDragDisplay::Degrees => (radians_to_degrees, degrees_to_radians),
        };
        self.run_drag_value_internal(
            id,
            label,
            value,
            step,
            unit_suffix,
            to_display_value,
            from_display_value,
        )
    }

    fn run_drag_value_internal(
        &mut self,
        id: UiId,
        label: UiTextValue,
        value: &mut f32,
        step: f32,
        unit_suffix: Option<UiTextValue>,
        to_display_value: fn(f32) -> f32,
        from_display_value: fn(f32) -> f32,
    ) -> UiResponse {
        let input = self.drag_input().clone();
        let mut drag_state = std::mem::take(self.drag_state_mut());
        let response = drag_value_f32_internal(
            self,
            id,
            &mut drag_state,
            &input,
            label,
            value,
            step,
            unit_suffix,
            to_display_value,
            from_display_value,
        );
        self.drag_value_state = drag_state;
        response
    }
}

pub struct DragValue<'a> {
    value: &'a mut f32,
    speed: f32,
    prefix: UiTextValue,
    suffix: UiTextValue,
    id_source: Option<UiId>,
}

impl<'a> DragValue<'a> {
    pub fn new(value: &'a mut f32) -> Self {
        Self {
            value,
            speed: 0.1,
            prefix: UiTextValue::default(),
            suffix: UiTextValue::default(),
            id_source: None,
        }
    }

    pub fn speed(mut self, speed: f64) -> Self {
        self.speed = speed as f32;
        self
    }

    pub fn prefix(mut self, prefix: impl Into<UiTextValue>) -> Self {
        self.prefix = prefix.into();
        self
    }

    pub fn suffix(mut self, suffix: impl Into<UiTextValue>) -> Self {
        self.suffix = suffix.into();
        self
    }

    pub fn id<K: IntoUiId>(mut self, id_source: K) -> Self {
        self.id_source = Some(id_source.into_ui_id());
        self
    }
}

impl Widget for DragValue<'_> {
    fn ui(self, ui: &mut UiContext) -> UiResponse {
        let DragValue {
            value,
            speed,
            prefix,
            suffix,
            id_source,
        } = self;
        let config = if suffix.is_empty() {
            UiDragValueConfig::new(prefix).step(speed)
        } else {
            UiDragValueConfig::new(prefix).step(speed).suffix(suffix)
        };
        match id_source {
            Some(id_source) => ui.drag_value_with_id(id_source, value, config),
            None => ui.drag_value(value, config),
        }
    }
}

fn scalar_identity(value: f32) -> f32 {
    value
}

fn radians_to_degrees(value: f32) -> f32 {
    value.to_degrees()
}

fn degrees_to_radians(value: f32) -> f32 {
    value.to_radians()
}

fn point_in_rect(point: Vec2, rect: UiRect) -> bool {
    point.x >= rect.x
        && point.y >= rect.y
        && point.x <= rect.x + rect.width
        && point.y <= rect.y + rect.height
}

fn char_count(text: &str) -> usize {
    text.chars().count()
}

fn drag_field_prefix_width(text: &str, end_chars: usize, font_size: f32) -> f32 {
    estimate_text_prefix_width(text, end_chars, font_size)
}

fn drag_field_text_width(text: &str, font_size: f32) -> f32 {
    let measured = estimate_text_width(text, font_size).max(0.0);
    let conservative_floor = char_count(text) as f32 * font_size.max(1.0) * 0.5;
    measured.max(conservative_floor)
}

fn byte_index_for_char_index(text: &str, char_index: usize) -> usize {
    if char_index == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_index)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn selection_range(edit: &EditSession) -> Option<(usize, usize)> {
    let anchor = edit.selection_anchor_chars?;
    if anchor == edit.cursor_chars {
        return None;
    }
    Some((anchor.min(edit.cursor_chars), anchor.max(edit.cursor_chars)))
}

fn clear_selection(edit: &mut EditSession) {
    edit.selection_anchor_chars = None;
}

fn set_cursor(edit: &mut EditSession, cursor: usize, extend_selection: bool) {
    let cursor = cursor.min(char_count(&edit.buffer));
    if extend_selection {
        if edit.selection_anchor_chars.is_none() {
            edit.selection_anchor_chars = Some(edit.cursor_chars);
        }
    } else {
        clear_selection(edit);
    }
    edit.cursor_chars = cursor;
}

fn delete_selection(edit: &mut EditSession) -> bool {
    let Some((start, end)) = selection_range(edit) else {
        return false;
    };
    let byte_start = byte_index_for_char_index(&edit.buffer, start);
    let byte_end = byte_index_for_char_index(&edit.buffer, end);
    edit.buffer.replace_range(byte_start..byte_end, "");
    edit.cursor_chars = start;
    clear_selection(edit);
    true
}

fn insert_text(edit: &mut EditSession, text: &str) {
    let _ = delete_selection(edit);
    let byte_cursor = byte_index_for_char_index(&edit.buffer, edit.cursor_chars);
    edit.buffer.insert_str(byte_cursor, text);
    edit.cursor_chars += text.chars().count();
    clear_selection(edit);
}

fn selected_text(edit: &EditSession) -> Option<String> {
    let (start, end) = selection_range(edit)?;
    let byte_start = byte_index_for_char_index(&edit.buffer, start);
    let byte_end = byte_index_for_char_index(&edit.buffer, end);
    Some(edit.buffer[byte_start..byte_end].to_string())
}

fn cursor_from_pointer(
    edit: &EditSession,
    rect: UiRect,
    pointer_x: f32,
    font_size: f32,
    text_padding_x: f32,
    text_visible_width: f32,
) -> usize {
    let text_left = rect.x + text_padding_x.max(4.0);
    let local_x = (pointer_x - text_left).clamp(0.0, text_visible_width.max(0.0));
    let target_x = (local_x + edit.horizontal_scroll_x).max(0.0);
    let total_chars = char_count(&edit.buffer);
    if total_chars == 0 {
        return 0;
    }

    let mut prev_edge = 0.0f32;
    for idx in 0..total_chars {
        let next_edge = drag_field_prefix_width(&edit.buffer, idx + 1, font_size).max(prev_edge);
        let midpoint = prev_edge + (next_edge - prev_edge) * 0.5;
        if target_x <= midpoint {
            return idx;
        }
        prev_edge = next_edge;
    }
    total_chars
}

fn text_field_visible_width(
    rect: UiRect,
    font_size: f32,
    text_padding_x: f32,
    suffix: Option<&UiTextValue>,
) -> f32 {
    let text_left = rect.x + text_padding_x.max(4.0);
    let text_right = rect.right() - text_padding_x.max(4.0);
    let mut width = (text_right - text_left).max(0.0);
    if let Some(unit) = suffix {
        let suffix_width = drag_field_text_width(unit.as_str(), font_size).max(0.0);
        let suffix_area_spacing = text_padding_x.max(4.0) * 0.6;
        width = (width - suffix_width - suffix_area_spacing).max(0.0);
    }
    width
}

fn clamp_edit_horizontal_scroll(edit: &mut EditSession, font_size: f32, visible_width: f32) {
    let full_width = drag_field_text_width(&edit.buffer, font_size).max(0.0);
    let end_padding = font_size.max(1.0) * 0.85;
    let max_scroll = (full_width - visible_width.max(0.0)).max(0.0);
    edit.horizontal_scroll_x = edit
        .horizontal_scroll_x
        .clamp(0.0, max_scroll + end_padding.max(0.0));
}

fn ensure_cursor_visible(edit: &mut EditSession, font_size: f32, visible_width: f32) {
    let visible_width = visible_width.max(0.0);
    let caret_x = drag_field_prefix_width(&edit.buffer, edit.cursor_chars, font_size);
    let caret_right_x = caret_x + font_size.max(1.0) * 0.65;
    let margin = font_size.max(1.0) * 0.12;
    let left = edit.horizontal_scroll_x + margin;
    let right = edit.horizontal_scroll_x + (visible_width - margin).max(0.0);

    if caret_x < left {
        edit.horizontal_scroll_x = (caret_x - margin).max(0.0);
    } else if caret_right_x > right {
        edit.horizontal_scroll_x = (caret_right_x - (visible_width - margin).max(0.0)).max(0.0);
    }
    clamp_edit_horizontal_scroll(edit, font_size, visible_width);
}

fn edit_display_text(edit: &EditSession) -> String {
    edit.buffer.clone()
}

fn quantize_drag_display_value(value: f32) -> i64 {
    if !value.is_finite() {
        return i64::MIN;
    }
    (value * 1_000.0).round() as i64
}

fn cached_drag_display_text(
    drag_state: &mut UiDragValueState,
    field_id: UiId,
    display_value: f32,
) -> UiTextValue {
    let quantized_value = quantize_drag_display_value(display_value);
    if let Some(entry) = drag_state.display_text_cache.get(&field_id)
        && entry.quantized_value == quantized_value
    {
        return UiTextValue::from(entry.text.clone());
    }

    let formatted: std::sync::Arc<str> = std::sync::Arc::<str>::from(format!("{display_value:.3}"));
    drag_state.display_text_cache.insert(
        field_id,
        DragDisplayTextCache {
            quantized_value,
            text: formatted.clone(),
        },
    );
    UiTextValue::from(formatted)
}

fn precision_step(base_step: f32, ctrl: bool, shift: bool) -> f32 {
    if ctrl && shift {
        base_step * 0.25
    } else if ctrl {
        base_step * 0.1
    } else if shift {
        base_step * 5.0
    } else {
        base_step
    }
}

#[allow(clippy::too_many_arguments)]
fn drag_value_f32_internal(
    ui: &mut UiContext,
    field_id: UiId,
    drag_state: &mut UiDragValueState,
    input: &UiDragInputSnapshot,
    label: UiTextValue,
    value: &mut f32,
    step: f32,
    unit_suffix: Option<UiTextValue>,
    to_display_value: fn(f32) -> f32,
    from_display_value: fn(f32) -> f32,
) -> UiResponse {
    const DOUBLE_CLICK_MS: u64 = 350;
    const DRAG_ACTIVATION_PX: f32 = 2.5;
    const DRAG_SENSITIVITY: f32 = 0.02;
    const FONT_SIZE: f32 = 10.5;

    if let Some(clipboard) = input.clipboard_text.as_ref() {
        drag_state.clipboard_text = clipboard.clone();
    }

    let editing_this = drag_state
        .active_edit
        .as_ref()
        .map(|edit| edit.id == field_id)
        .unwrap_or(false);
    let display_text = drag_state
        .active_edit
        .as_ref()
        .filter(|edit| edit.id == field_id)
        .map(|edit| UiTextValue::from(edit_display_text(edit)))
        .unwrap_or_else(|| {
            cached_drag_display_text(drag_state, field_id, to_display_value(*value))
        });
    let caret_visible = drag_state
        .active_edit
        .as_ref()
        .filter(|edit| edit.id == field_id)
        .map(|edit| ((Instant::now().duration_since(edit.started_at).as_millis() / 450) % 2) == 0)
        .unwrap_or(false);
    let selection_chars = drag_state
        .active_edit
        .as_ref()
        .filter(|edit| edit.id == field_id)
        .and_then(selection_range);
    let cursor_chars = drag_state
        .active_edit
        .as_ref()
        .filter(|edit| edit.id == field_id)
        .map(|edit| edit.cursor_chars);
    let text_scroll_x = drag_state
        .active_edit
        .as_ref()
        .filter(|edit| edit.id == field_id)
        .map(|edit| edit.horizontal_scroll_x)
        .unwrap_or(0.0);
    let field_font_size = ui.theme().font_size.max(1.0);
    let text_padding_x = ui.theme().item_padding.max(4.0);

    let mut value_response = UiResponse::default();
    let label_style = UiTextStyle {
        color: ui.theme().muted_text,
        font_size: FONT_SIZE,
        align_h: UiTextAlign::Start,
        align_v: UiTextAlign::Center,
        wrap: false,
    };
    ui.row_with_id(field_id.child("row"), |ui| {
        ui.label_with_style_with_id(field_id.child("label"), label.clone(), label_style);
        value_response = ui.text_field_with_options_with_id(
            field_id.child("value_field"),
            display_text,
            None,
            editing_this,
            cursor_chars,
            selection_chars,
            caret_visible,
            unit_suffix.clone(),
            text_scroll_x,
        );
    });

    let mut changed = false;
    let mut started_edit = false;

    if !editing_this && value_response.clicked() {
        let now = Instant::now();
        let double_clicked = drag_state
            .last_click
            .as_ref()
            .map(|(id, time)| {
                *id == field_id
                    && now.duration_since(*time) <= Duration::from_millis(DOUBLE_CLICK_MS)
            })
            .unwrap_or(false);
        drag_state.last_click = Some((field_id, now));

        if double_clicked {
            let initial_buffer = format!("{:.3}", to_display_value(*value));
            let initial_chars = char_count(&initial_buffer);
            drag_state.pending_drag = None;
            drag_state.active_drag = None;
            drag_state.active_edit = Some(EditSession {
                id: field_id,
                buffer: initial_buffer,
                cursor_chars: initial_chars,
                selection_anchor_chars: Some(0),
                pointer_select_anchor_chars: None,
                horizontal_scroll_x: 0.0,
                started_at: now,
            });
            started_edit = true;
        }
    }

    if !editing_this && !started_edit && input.pointer_down && value_response.active() {
        if drag_state
            .pending_drag
            .as_ref()
            .map(|pending| pending.id == field_id)
            .unwrap_or(false)
        {
            if let Some(pending) = drag_state.pending_drag.as_ref() {
                let drag_delta = (input.cursor_pos.x - pending.start_cursor_x).abs();
                if drag_delta >= DRAG_ACTIVATION_PX {
                    drag_state.active_drag = Some(DragSession {
                        id: field_id,
                        start_cursor_x: pending.start_cursor_x,
                        start_value: pending.start_value,
                    });
                    drag_state.pending_drag = None;
                }
            }
        } else if drag_state
            .active_drag
            .as_ref()
            .map(|session| session.id != field_id)
            .unwrap_or(true)
        {
            drag_state.pending_drag = Some(PendingDragSession {
                id: field_id,
                start_cursor_x: input.cursor_pos.x,
                start_value: to_display_value(*value),
            });
        }
    }

    if let Some(session) = drag_state.active_drag.as_mut() {
        if session.id == field_id && input.pointer_down {
            let delta_x = input.cursor_pos.x - session.start_cursor_x;
            let next_display = session.start_value
                + delta_x
                    * step.max(0.0001)
                    * DRAG_SENSITIVITY
                    * precision_step(1.0, input.ctrl, input.shift);
            let next = from_display_value(next_display);
            if (next - *value).abs() > f32::EPSILON {
                *value = next;
                changed = true;
            }
        }
    }

    if !input.pointer_down {
        if drag_state
            .pending_drag
            .as_ref()
            .map(|session| session.id == field_id)
            .unwrap_or(false)
        {
            drag_state.pending_drag = None;
        }
        if drag_state
            .active_drag
            .as_ref()
            .map(|session| session.id == field_id)
            .unwrap_or(false)
        {
            drag_state.active_drag = None;
        }
    }

    if let Some(edit) = drag_state.active_edit.as_mut() {
        if edit.id == field_id {
            let visible_width = value_response
                .rect()
                .map(|rect| {
                    text_field_visible_width(
                        rect,
                        field_font_size,
                        text_padding_x,
                        unit_suffix.as_ref(),
                    )
                })
                .unwrap_or(96.0);
            ensure_cursor_visible(edit, field_font_size, visible_width);

            if input.pointer_pressed
                && let Some(rect) = value_response.rect()
                && point_in_rect(input.cursor_pos, rect)
            {
                let cursor = cursor_from_pointer(
                    edit,
                    rect,
                    input.cursor_pos.x,
                    field_font_size,
                    text_padding_x,
                    visible_width,
                );
                let anchor = if input.shift {
                    edit.selection_anchor_chars.unwrap_or(edit.cursor_chars)
                } else {
                    cursor
                };
                set_cursor(edit, cursor, input.shift);
                ensure_cursor_visible(edit, field_font_size, visible_width);
                edit.pointer_select_anchor_chars = Some(anchor);
            }

            if input.pointer_down {
                if let Some(anchor) = edit.pointer_select_anchor_chars
                    && let Some(rect) = value_response.rect()
                {
                    let text_left = rect.x + text_padding_x.max(4.0);
                    let text_right = text_left + visible_width;
                    if input.cursor_pos.x < text_left {
                        let overshoot = text_left - input.cursor_pos.x;
                        edit.horizontal_scroll_x =
                            (edit.horizontal_scroll_x - overshoot * 0.8).max(0.0);
                    } else if input.cursor_pos.x > text_right {
                        let overshoot = input.cursor_pos.x - text_right;
                        edit.horizontal_scroll_x += overshoot * 0.8;
                    }
                    clamp_edit_horizontal_scroll(edit, field_font_size, visible_width);
                    let cursor = cursor_from_pointer(
                        edit,
                        rect,
                        input.cursor_pos.x,
                        field_font_size,
                        text_padding_x,
                        visible_width,
                    );
                    edit.cursor_chars = cursor;
                    edit.selection_anchor_chars = Some(anchor);
                    ensure_cursor_visible(edit, field_font_size, visible_width);
                }
            } else if input.pointer_released {
                edit.pointer_select_anchor_chars = None;
            }

            let mut commit = false;
            let mut cancel = false;
            for key in &input.just_pressed {
                match key {
                    UiKeyInput::Enter => commit = true,
                    UiKeyInput::Escape => cancel = true,
                    UiKeyInput::ArrowLeft => {
                        if edit.cursor_chars > 0 {
                            set_cursor(edit, edit.cursor_chars - 1, input.shift);
                            ensure_cursor_visible(edit, field_font_size, visible_width);
                        } else {
                            edit.horizontal_scroll_x = (edit.horizontal_scroll_x
                                - field_font_size.max(1.0) * 0.8)
                                .max(0.0);
                            clamp_edit_horizontal_scroll(edit, field_font_size, visible_width);
                        }
                    }
                    UiKeyInput::ArrowRight => {
                        let text_len = char_count(&edit.buffer);
                        if edit.cursor_chars < text_len {
                            set_cursor(edit, edit.cursor_chars + 1, input.shift);
                            ensure_cursor_visible(edit, field_font_size, visible_width);
                        } else {
                            edit.horizontal_scroll_x += field_font_size.max(1.0) * 0.8;
                            clamp_edit_horizontal_scroll(edit, field_font_size, visible_width);
                        }
                    }
                    UiKeyInput::Home => {
                        set_cursor(edit, 0, input.shift);
                        ensure_cursor_visible(edit, field_font_size, visible_width);
                    }
                    UiKeyInput::End => {
                        set_cursor(edit, char_count(&edit.buffer), input.shift);
                        ensure_cursor_visible(edit, field_font_size, visible_width);
                    }
                    UiKeyInput::Backspace => {
                        if !delete_selection(edit) && edit.cursor_chars > 0 {
                            let remove_start =
                                byte_index_for_char_index(&edit.buffer, edit.cursor_chars - 1);
                            let remove_end =
                                byte_index_for_char_index(&edit.buffer, edit.cursor_chars);
                            edit.buffer.replace_range(remove_start..remove_end, "");
                            edit.cursor_chars -= 1;
                        }
                        ensure_cursor_visible(edit, field_font_size, visible_width);
                    }
                    UiKeyInput::Delete => {
                        if !delete_selection(edit) {
                            let len = char_count(&edit.buffer);
                            if edit.cursor_chars < len {
                                let remove_start =
                                    byte_index_for_char_index(&edit.buffer, edit.cursor_chars);
                                let remove_end =
                                    byte_index_for_char_index(&edit.buffer, edit.cursor_chars + 1);
                                edit.buffer.replace_range(remove_start..remove_end, "");
                            }
                        }
                        ensure_cursor_visible(edit, field_font_size, visible_width);
                    }
                    UiKeyInput::ArrowUp => {
                        let next_display = to_display_value(*value)
                            + precision_step(step, input.ctrl, input.shift);
                        let next = from_display_value(next_display);
                        if (next - *value).abs() > f32::EPSILON {
                            *value = next;
                            changed = true;
                        }
                    }
                    UiKeyInput::ArrowDown => {
                        let next_display = to_display_value(*value)
                            - precision_step(step, input.ctrl, input.shift);
                        let next = from_display_value(next_display);
                        if (next - *value).abs() > f32::EPSILON {
                            *value = next;
                            changed = true;
                        }
                    }
                    UiKeyInput::SelectAll => {
                        edit.cursor_chars = char_count(&edit.buffer);
                        edit.selection_anchor_chars = Some(0);
                        ensure_cursor_visible(edit, field_font_size, visible_width);
                    }
                    UiKeyInput::Copy => {
                        if let Some(text) = selected_text(edit) {
                            drag_state.clipboard_write = Some(text);
                        }
                    }
                    UiKeyInput::Cut => {
                        if let Some(text) = selected_text(edit) {
                            drag_state.clipboard_write = Some(text);
                            let _ = delete_selection(edit);
                        }
                    }
                    UiKeyInput::Paste => {
                        if !drag_state.clipboard_text.is_empty() {
                            let text = drag_state.clipboard_text.clone();
                            insert_text(edit, &text);
                            ensure_cursor_visible(edit, field_font_size, visible_width);
                        }
                    }
                    UiKeyInput::Char(ch) => {
                        if ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+' | 'e' | 'E') {
                            insert_text(edit, &ch.to_string());
                            ensure_cursor_visible(edit, field_font_size, visible_width);
                        }
                    }
                }
            }

            if input.pointer_pressed
                && let Some(rect) = value_response.rect()
                && !point_in_rect(input.cursor_pos, rect)
            {
                commit = true;
            }

            if cancel {
                drag_state.active_edit = None;
            } else if commit {
                if let Ok(parsed_display) = edit.buffer.trim().parse::<f32>() {
                    let parsed = from_display_value(parsed_display);
                    if (parsed - *value).abs() > f32::EPSILON {
                        *value = parsed;
                        changed = true;
                    }
                }
                drag_state.active_edit = None;
            }
        }
    }

    value_response.changed = changed;
    value_response
}
