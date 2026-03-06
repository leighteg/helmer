use bevy_ecs::prelude::Resource;
use helmer_ui::UiId;
use helmer_ui_docking::{DockAxis, DockState, DockTab};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct EditorDockWorkspace {
    pub id: UiId,
    pub docking: DockState,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct EditorRetainedDockingState {
    pub workspaces: HashMap<UiId, EditorDockWorkspace>,
}

impl EditorRetainedDockingState {
    pub fn ensure_workspace(
        &mut self,
        workspace_id: UiId,
        initial_tab: DockTab,
    ) -> &mut EditorDockWorkspace {
        self.workspaces
            .entry(workspace_id)
            .or_insert_with(|| EditorDockWorkspace {
                id: workspace_id,
                docking: DockState::new(initial_tab),
            })
    }

    pub fn workspace(&self, workspace_id: UiId) -> Option<&EditorDockWorkspace> {
        self.workspaces.get(&workspace_id)
    }

    pub fn workspace_mut(&mut self, workspace_id: UiId) -> Option<&mut EditorDockWorkspace> {
        self.workspaces.get_mut(&workspace_id)
    }

    pub fn add_tab(&mut self, workspace_id: UiId, tab: DockTab) {
        if let Some(workspace) = self.workspaces.get_mut(&workspace_id) {
            workspace.docking.add_tab_to_focused(tab);
        }
    }

    pub fn split_tab(
        &mut self,
        workspace_id: UiId,
        axis: DockAxis,
        ratio: f32,
        tab: DockTab,
        place_after: bool,
    ) {
        if let Some(workspace) = self.workspaces.get_mut(&workspace_id) {
            workspace
                .docking
                .split_focused(axis, ratio, tab, place_after);
        }
    }
}
