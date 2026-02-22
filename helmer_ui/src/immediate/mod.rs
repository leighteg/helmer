use hashbrown::HashMap;

use crate::{UiId, UiInteractionState, UiNode, UiRect, UiTheme};

mod core;
mod drag_value;
mod layout;
mod widgets;
mod window;

pub use drag_value::{
    DragValue, UiDragDisplay, UiDragInputSnapshot, UiDragValueConfig, UiDragValueState, UiKeyInput,
};
pub use widgets::{ButtonWidget, LabelWidget, button, label};
pub use window::{UiWindowOptions, UiWindowResponse};

pub trait Widget {
    fn ui(self, ui: &mut UiContext) -> UiResponse;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UiResponse {
    pub id: UiId,
    pub hovered: bool,
    pub active: bool,
    pub clicked: bool,
    pub changed: bool,
    pub enabled: bool,
    pub rect: Option<UiRect>,
}

impl UiResponse {
    pub fn hovered(self) -> bool {
        self.hovered
    }

    pub fn active(self) -> bool {
        self.active
    }

    pub fn clicked(self) -> bool {
        self.clicked
    }

    pub fn changed(self) -> bool {
        self.changed
    }

    pub fn enabled(self) -> bool {
        self.enabled
    }

    pub fn rect(self) -> Option<UiRect> {
        self.rect
    }
}

pub struct UiContext {
    theme: UiTheme,
    interaction: UiInteractionState,
    last_layout_rects: HashMap<UiId, UiRect>,
    last_hit_rects: HashMap<UiId, UiRect>,
    state_bools: HashMap<UiId, bool>,
    drag_value_state: UiDragValueState,
    drag_input: UiDragInputSnapshot,
    path_stack: Vec<UiId>,
    auto_index_stack: Vec<u64>,
    node_stack: Vec<UiNode>,
    roots: Vec<UiNode>,
}

pub struct UiBuildResult {
    pub roots: Vec<UiNode>,
    pub state_bools: HashMap<UiId, bool>,
    pub drag_value_state: UiDragValueState,
}
