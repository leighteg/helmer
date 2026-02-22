use crate::{IntoUiId, UiId, UiInteractionState, UiNode, UiRect, UiTheme};

use super::{UiBuildResult, UiContext, UiDragInputSnapshot, UiDragValueState, UiResponse, Widget};

impl UiContext {
    pub fn new(
        theme: UiTheme,
        interaction: UiInteractionState,
        last_layout_rects: hashbrown::HashMap<UiId, UiRect>,
        last_hit_rects: hashbrown::HashMap<UiId, UiRect>,
        state_bools: hashbrown::HashMap<UiId, bool>,
        drag_value_state: UiDragValueState,
        drag_input: UiDragInputSnapshot,
    ) -> Self {
        Self {
            theme,
            interaction,
            last_layout_rects,
            last_hit_rects,
            state_bools,
            drag_value_state,
            drag_input,
            path_stack: Vec::new(),
            auto_index_stack: vec![0],
            node_stack: Vec::new(),
            roots: Vec::new(),
        }
    }

    pub fn finish(self) -> UiBuildResult {
        UiBuildResult {
            roots: self.roots,
            state_bools: self.state_bools,
            drag_value_state: self.drag_value_state,
        }
    }

    pub fn theme(&self) -> &UiTheme {
        &self.theme
    }

    pub fn theme_mut(&mut self) -> &mut UiTheme {
        &mut self.theme
    }

    pub fn with_id<K, F>(&mut self, key: K, build: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        let id = self.derive_id(key);
        self.path_stack.push(id);
        self.auto_index_stack.push(0);
        build(self);
        self.auto_index_stack.pop();
        self.path_stack.pop();
    }

    pub fn push_id<K, F>(&mut self, id_source: K, add_contents: F)
    where
        K: IntoUiId,
        F: FnOnce(&mut Self),
    {
        self.with_id(id_source, add_contents);
    }

    pub fn add<W: Widget>(&mut self, widget: W) -> UiResponse {
        widget.ui(self)
    }

    pub fn set_drag_input_snapshot(&mut self, input: UiDragInputSnapshot) {
        self.drag_input = input;
    }

    pub(super) fn derive_id<K: IntoUiId>(&self, key: K) -> UiId {
        let key_id = key.into_ui_id();
        self.path_stack
            .last()
            .copied()
            .unwrap_or(UiId::ROOT)
            .child(key_id)
    }

    pub(super) fn next_auto_id(&mut self) -> UiId {
        let parent = self.path_stack.last().copied().unwrap_or(UiId::ROOT);
        let next_index = self
            .auto_index_stack
            .last_mut()
            .expect("auto id stack underflow");
        *next_index = next_index.saturating_add(1);
        parent.child(UiId::from_raw(*next_index))
    }

    pub(super) fn response(&self, id: UiId, enabled: bool) -> UiResponse {
        UiResponse {
            id,
            hovered: enabled && self.interaction.is_hovered(id),
            active: enabled && self.interaction.is_active(id),
            clicked: enabled && self.interaction.is_clicked(id),
            changed: false,
            enabled,
            rect: self
                .last_hit_rects
                .get(&id)
                .copied()
                .or_else(|| self.last_layout_rects.get(&id).copied()),
        }
    }

    pub(super) fn with_node<F>(&mut self, node: UiNode, build: F)
    where
        F: FnOnce(&mut Self),
    {
        let id = node.id;
        self.node_stack.push(node);
        self.path_stack.push(id);
        self.auto_index_stack.push(0);
        build(self);
        self.auto_index_stack.pop();
        self.path_stack.pop();
        let node = self.node_stack.pop().expect("node stack underflow");
        self.push_leaf(node);
    }

    pub(super) fn push_leaf(&mut self, node: UiNode) {
        if let Some(parent) = self.node_stack.last_mut() {
            parent.children.push(node);
        } else {
            self.roots.push(node);
        }
    }

    pub(super) fn set_bool_state(&mut self, id: UiId, value: bool) {
        self.state_bools.insert(id, value);
    }

    pub(super) fn bool_state_with_default(&mut self, id: UiId, default: bool) -> bool {
        *self.state_bools.entry(id).or_insert(default)
    }

    pub(super) fn drag_input(&self) -> &UiDragInputSnapshot {
        &self.drag_input
    }

    pub(super) fn drag_state_mut(&mut self) -> &mut UiDragValueState {
        &mut self.drag_value_state
    }
}
