use bevy_ecs::prelude::Resource;
use helmer_ui::UiId;
use helmer_ui_graph::{
    GraphInteractionController, GraphState, RetainedGraphBuilder, RetainedGraphFrame,
};
use std::collections::HashMap;

#[derive(Resource, Clone, Debug, Default)]
pub struct EditorRetainedGraphState {
    pub graphs: HashMap<UiId, GraphState>,
    pub active_graph: Option<UiId>,
}

impl EditorRetainedGraphState {
    pub fn ensure_graph(&mut self, id: UiId) -> &mut GraphState {
        self.active_graph = Some(id);
        self.graphs.entry(id).or_default()
    }

    pub fn graph(&self, id: UiId) -> Option<&GraphState> {
        self.graphs.get(&id)
    }

    pub fn graph_mut(&mut self, id: UiId) -> Option<&mut GraphState> {
        self.graphs.get_mut(&id)
    }
}

#[derive(Resource, Default)]
pub struct EditorRetainedGraphRenderer(pub RetainedGraphBuilder);

#[derive(Resource, Clone, Debug, Default)]
pub struct EditorRetainedGraphInteractionState {
    pub controllers: HashMap<UiId, GraphInteractionController>,
    pub frames: HashMap<UiId, RetainedGraphFrame>,
    pub pointer_down_previous: bool,
}
