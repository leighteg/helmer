use std::path::Path;

use glam::Vec2;
use helmer_ui::{RetainedUi, UiId, UiRect};
use helmer_ui_graph::{
    GraphEdge, GraphNode, GraphPin, GraphPreviewEdge, GraphState, RetainedGraphBuilder,
    RetainedGraphFrame,
};

pub fn graph_id_for_tab(tab_id: u64, asset_path: Option<&Path>) -> UiId {
    if let Some(path) = asset_path {
        UiId::from_str(&format!("visual-script:{}", path.to_string_lossy()))
    } else {
        UiId::from_str(&format!("visual-script-tab:{}", tab_id))
    }
}

pub fn ensure_seed_graph(graph: &mut GraphState, graph_id: UiId) {
    if !graph.nodes.is_empty() {
        return;
    }

    let start_id = graph_id.child("node-start");
    let wait_id = graph_id.child("node-wait");
    let log_id = graph_id.child("node-log");

    let start_out = start_id.child("out-exec");
    let wait_in = wait_id.child("in-exec");
    let wait_out = wait_id.child("out-exec");
    let log_in = log_id.child("in-exec");

    let mut start = GraphNode::new(start_id, "On Start");
    start.position = Vec2::new(48.0, 70.0);
    start.size = Vec2::new(220.0, 92.0);
    start.outputs.push(GraphPin::new(start_out, "Exec"));

    let mut wait = GraphNode::new(wait_id, "Wait Seconds");
    wait.position = Vec2::new(340.0, 120.0);
    wait.size = Vec2::new(250.0, 126.0);
    wait.inputs.push(GraphPin::new(wait_in, "Exec In"));
    wait.outputs.push(GraphPin::new(wait_out, "Exec Out"));

    let mut log = GraphNode::new(log_id, "Log");
    log.position = Vec2::new(660.0, 140.0);
    log.size = Vec2::new(220.0, 94.0);
    log.inputs.push(GraphPin::new(log_in, "Exec In"));

    graph.add_node(start);
    graph.add_node(wait);
    graph.add_node(log);

    graph.connect(GraphEdge {
        from_node: start_id,
        from_pin: start_out,
        to_node: wait_id,
        to_pin: wait_in,
    });
    graph.connect(GraphEdge {
        from_node: wait_id,
        from_pin: wait_out,
        to_node: log_id,
        to_pin: log_in,
    });
}

pub fn build_visual_scripting_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    viewport: UiRect,
    renderer: &RetainedGraphBuilder,
    graph: &GraphState,
    preview: Option<GraphPreviewEdge>,
) -> RetainedGraphFrame {
    renderer.build_frame(retained, root_id, viewport, graph, preview)
}
