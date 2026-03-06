use bevy_ecs::prelude::Resource;
use helmer_ui::UiId;
use helmer_ui_layout::{LayoutRect, WindowFrame, WindowLayoutSnapshot, WindowLayoutState};
use std::collections::HashMap;

#[derive(Resource, Clone, Debug)]
pub struct EditorRetainedLayoutState {
    pub workspace_bounds: LayoutRect,
    pub windows: WindowLayoutState,
}

impl Default for EditorRetainedLayoutState {
    fn default() -> Self {
        Self {
            workspace_bounds: LayoutRect::new(0.0, 0.0, 1.0, 1.0),
            windows: WindowLayoutState::default(),
        }
    }
}

impl EditorRetainedLayoutState {
    pub fn sync_workspace_bounds(&mut self, bounds: LayoutRect) {
        let previous_bounds = self.workspace_bounds;
        self.workspace_bounds = bounds;
        if previous_bounds.width > 1.0
            && previous_bounds.height > 1.0
            && ((previous_bounds.width - bounds.width).abs() > 1.0
                || (previous_bounds.height - bounds.height).abs() > 1.0)
        {
            self.windows.rescale_to_bounds(previous_bounds, bounds);
        } else {
            self.windows.normalize_to_bounds(bounds);
        }
    }

    pub fn ensure_window(&mut self, id: UiId, fallback: WindowFrame) -> &mut WindowFrame {
        self.windows.ensure_window(id, fallback)
    }

    pub fn remove_window(&mut self, id: UiId) {
        self.windows.remove_window(id);
    }
}

#[derive(Clone, Debug, Default)]
pub struct RetainedLayoutSnapshot {
    pub windows: WindowLayoutSnapshot,
}

#[derive(Resource, Clone, Debug)]
pub struct EditorRetainedLayoutCatalog {
    pub active: Option<String>,
    pub layouts: HashMap<String, RetainedLayoutSnapshot>,
    pub next_generated_name: u32,
    pub allow_layout_move: bool,
    pub allow_layout_resize: bool,
    pub live_reflow: bool,
}

impl Default for EditorRetainedLayoutCatalog {
    fn default() -> Self {
        Self {
            active: None,
            layouts: HashMap::new(),
            next_generated_name: 1,
            allow_layout_move: true,
            allow_layout_resize: true,
            live_reflow: true,
        }
    }
}

impl EditorRetainedLayoutCatalog {
    pub fn ensure_default(&mut self, windows: WindowLayoutSnapshot) {
        self.layouts
            .entry("Default".to_string())
            .or_insert(RetainedLayoutSnapshot { windows });
        if self.active.is_none() {
            self.active = Some("Default".to_string());
        }
    }

    pub fn save_active(&mut self, windows: WindowLayoutSnapshot) -> bool {
        let Some(active) = self.active.clone() else {
            return false;
        };
        self.layouts
            .insert(active, RetainedLayoutSnapshot { windows });
        true
    }

    pub fn save_as_new(&mut self, windows: WindowLayoutSnapshot) -> String {
        loop {
            let name = format!("Layout {}", self.next_generated_name.max(1));
            self.next_generated_name = self.next_generated_name.saturating_add(1);
            if self.layouts.contains_key(&name) {
                continue;
            }
            self.layouts
                .insert(name.clone(), RetainedLayoutSnapshot { windows });
            self.active = Some(name.clone());
            return name;
        }
    }
}
