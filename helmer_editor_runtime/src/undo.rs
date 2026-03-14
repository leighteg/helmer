use helmer_becs::ecs::prelude::Resource;

#[derive(Resource, Debug, Clone)]
pub struct EditorUndoState<E, G> {
    pub entries: Vec<E>,
    pub cursor: usize,
    pub max_entries: usize,
    pub pending_group: Option<G>,
    pub pending_commit: bool,
}

impl<E, G> Default for EditorUndoState<E, G> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            cursor: 0,
            max_entries: 128,
            pending_group: None,
            pending_commit: false,
        }
    }
}

impl<E, G> EditorUndoState<E, G> {
    pub fn can_undo(&self) -> bool {
        !self.entries.is_empty() && self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        !self.entries.is_empty() && self.cursor + 1 < self.entries.len()
    }

    pub fn undo_label<'a, F>(&'a self, label_of: F) -> Option<&'a str>
    where
        F: Fn(&'a E) -> Option<&'a str>,
    {
        if self.can_undo() {
            self.entries.get(self.cursor).and_then(label_of)
        } else {
            None
        }
    }

    pub fn redo_label<'a, F>(&'a self, label_of: F) -> Option<&'a str>
    where
        F: Fn(&'a E) -> Option<&'a str>,
    {
        if self.can_redo() {
            self.entries.get(self.cursor + 1).and_then(label_of)
        } else {
            None
        }
    }

    pub fn enforce_cap(&mut self) {
        let cap = self.max_entries.max(1);
        if self.entries.len() <= cap {
            return;
        }
        let remove_count = self.entries.len() - cap;
        self.entries.drain(0..remove_count);
        self.cursor = self.cursor.saturating_sub(remove_count);
    }
}
