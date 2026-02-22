use hashbrown::HashMap;

use crate::{UiColor, UiId, UiRect, UiTextAlign, UiTextValue};

#[derive(Clone, Debug)]
pub struct UiRectCommand {
    pub id: UiId,
    pub rect: UiRect,
    pub color: UiColor,
    pub clip: Option<UiRect>,
    pub layer: f32,
}

#[derive(Clone, Debug)]
pub struct UiImageCommand {
    pub id: UiId,
    pub rect: UiRect,
    pub texture_id: Option<usize>,
    pub tint: UiColor,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub clip: Option<UiRect>,
    pub layer: f32,
}

#[derive(Clone, Debug)]
pub struct UiTextCommand {
    pub id: UiId,
    pub rect: UiRect,
    pub text: UiTextValue,
    pub color: UiColor,
    pub font_size: f32,
    pub align_h: UiTextAlign,
    pub align_v: UiTextAlign,
    pub wrap: bool,
    pub cursor: Option<usize>,
    pub show_caret: bool,
    pub caret_color: Option<UiColor>,
    pub selection: Option<(usize, usize)>,
    pub selection_color: Option<UiColor>,
    pub clip: Option<UiRect>,
    pub layer: f32,
}

#[derive(Clone, Debug)]
pub enum UiDrawCommand {
    Rect(UiRectCommand),
    Image(UiImageCommand),
    Text(UiTextCommand),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UiInteractionState {
    pub hovered: Option<UiId>,
    pub active: Option<UiId>,
    pub clicked: Option<UiId>,
    pub pointer_captured: bool,
}

impl UiInteractionState {
    pub fn is_hovered(self, id: UiId) -> bool {
        self.hovered == Some(id)
    }

    pub fn is_active(self, id: UiId) -> bool {
        self.active == Some(id)
    }

    pub fn is_clicked(self, id: UiId) -> bool {
        self.clicked == Some(id)
    }
}

#[derive(Clone, Debug, Default)]
pub struct UiFrameOutput {
    pub draw_commands: Vec<UiDrawCommand>,
    pub layout_rects: HashMap<UiId, UiRect>,
    pub subtree_rects: HashMap<UiId, UiRect>,
    pub paint_order: Vec<UiId>,
    pub command_hash: u64,
    pub command_hash_valid: bool,
    pub interaction: UiInteractionState,
}
