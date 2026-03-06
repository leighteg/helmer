use glam::Vec2;
use hashbrown::HashMap;
use helmer_ui::{
    RetainedUi, RetainedUiNode, UiColor, UiDimension, UiId, UiImage, UiLabel, UiLayoutStyle,
    UiPositionType, UiRect, UiStyle, UiTextAlign, UiTextStyle, UiTextValue, UiVisualStyle,
    UiWidget,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphPin {
    pub id: UiId,
    pub label: String,
}

impl GraphPin {
    pub fn new(id: UiId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphNode {
    pub id: UiId,
    pub title: String,
    pub position: Vec2,
    pub size: Vec2,
    pub inputs: Vec<GraphPin>,
    pub outputs: Vec<GraphPin>,
    pub selected: bool,
}

impl GraphNode {
    pub fn new(id: UiId, title: impl Into<String>) -> Self {
        Self {
            id,
            title: title.into(),
            position: Vec2::ZERO,
            size: Vec2::new(240.0, 160.0),
            inputs: Vec::new(),
            outputs: Vec::new(),
            selected: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphEdge {
    pub from_node: UiId,
    pub from_pin: UiId,
    pub to_node: UiId,
    pub to_pin: UiId,
}

#[derive(Clone, Debug)]
pub struct GraphState {
    pub nodes: HashMap<UiId, GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub pan: Vec2,
    pub zoom: f32,
}

impl Default for GraphState {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            pan: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

impl GraphState {
    pub fn add_node(&mut self, node: GraphNode) -> Option<GraphNode> {
        self.nodes.insert(node.id, node)
    }

    pub fn remove_node(&mut self, id: UiId) -> Option<GraphNode> {
        self.edges
            .retain(|edge| edge.from_node != id && edge.to_node != id);
        self.nodes.remove(&id)
    }

    pub fn connect(&mut self, edge: GraphEdge) {
        if !self.edges.iter().any(|candidate| *candidate == edge) {
            self.edges.push(edge);
        }
    }

    pub fn disconnect(&mut self, edge: GraphEdge) {
        self.edges.retain(|candidate| *candidate != edge);
    }

    pub fn node(&self, id: UiId) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    pub fn node_mut(&mut self, id: UiId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }

    pub fn clear_selection(&mut self) -> bool {
        let mut changed = false;
        for node in self.nodes.values_mut() {
            if node.selected {
                node.selected = false;
                changed = true;
            }
        }
        changed
    }

    pub fn select_only(&mut self, selected: UiId) -> bool {
        let mut changed = false;
        for node in self.nodes.values_mut() {
            let should_select = node.id == selected;
            if node.selected != should_select {
                node.selected = should_select;
                changed = true;
            }
        }
        changed
    }
}

#[derive(Clone, Debug)]
pub struct GraphStyle {
    pub canvas_background: UiColor,
    pub canvas_border: UiColor,
    pub node_background: UiColor,
    pub node_selected_background: UiColor,
    pub node_border: UiColor,
    pub title_background: UiColor,
    pub text_color: UiColor,
    pub edge_color: UiColor,
    pub pin_color: UiColor,
    pub title_height: f32,
    pub row_height: f32,
    pub pin_size: f32,
    pub edge_thickness: f32,
    pub node_padding: f32,
}

impl Default for GraphStyle {
    fn default() -> Self {
        Self {
            canvas_background: UiColor::rgba(0.06, 0.07, 0.09, 0.96),
            canvas_border: UiColor::rgba(0.20, 0.23, 0.28, 0.92),
            node_background: UiColor::rgba(0.16, 0.18, 0.22, 0.95),
            node_selected_background: UiColor::rgba(0.22, 0.26, 0.34, 0.98),
            node_border: UiColor::rgba(0.30, 0.34, 0.41, 0.95),
            title_background: UiColor::rgba(0.24, 0.28, 0.36, 0.96),
            text_color: UiColor::rgba(0.95, 0.96, 0.98, 1.0),
            edge_color: UiColor::rgba(0.66, 0.74, 0.88, 0.88),
            pin_color: UiColor::rgba(0.92, 0.95, 1.0, 0.95),
            title_height: 28.0,
            row_height: 22.0,
            pin_size: 7.0,
            edge_thickness: 2.0,
            node_padding: 10.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphPinKind {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphPinRef {
    pub node: UiId,
    pub pin: UiId,
    pub kind: GraphPinKind,
}

impl GraphPinRef {
    pub fn input(node: UiId, pin: UiId) -> Self {
        Self {
            node,
            pin,
            kind: GraphPinKind::Input,
        }
    }

    pub fn output(node: UiId, pin: UiId) -> Self {
        Self {
            node,
            pin,
            kind: GraphPinKind::Output,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphHitTarget {
    Canvas,
    NodeBody {
        node: UiId,
    },
    Pin {
        node: UiId,
        pin: UiId,
        kind: GraphPinKind,
    },
}

#[derive(Clone, Debug)]
pub struct RetainedGraphFrame {
    pub root_id: UiId,
    pub viewport: UiRect,
    pub canvas_hit_id: UiId,
    node_hits: HashMap<UiId, UiId>,
    input_pin_hits: HashMap<UiId, (UiId, UiId)>,
    output_pin_hits: HashMap<UiId, (UiId, UiId)>,
    pin_centers: HashMap<GraphPinRef, Vec2>,
}

impl Default for RetainedGraphFrame {
    fn default() -> Self {
        Self {
            root_id: UiId::from_raw(0),
            viewport: UiRect::default(),
            canvas_hit_id: UiId::from_raw(0),
            node_hits: HashMap::new(),
            input_pin_hits: HashMap::new(),
            output_pin_hits: HashMap::new(),
            pin_centers: HashMap::new(),
        }
    }
}

impl RetainedGraphFrame {
    pub fn new(root_id: UiId, viewport: UiRect) -> Self {
        Self {
            root_id,
            viewport,
            canvas_hit_id: root_id.child("canvas").child("hit"),
            node_hits: HashMap::new(),
            input_pin_hits: HashMap::new(),
            output_pin_hits: HashMap::new(),
            pin_centers: HashMap::new(),
        }
    }

    pub fn target_for_hit(&self, id: UiId) -> Option<GraphHitTarget> {
        if id == self.canvas_hit_id {
            return Some(GraphHitTarget::Canvas);
        }
        if let Some(node) = self.node_hits.get(&id).copied() {
            return Some(GraphHitTarget::NodeBody { node });
        }
        if let Some((node, pin)) = self.input_pin_hits.get(&id).copied() {
            return Some(GraphHitTarget::Pin {
                node,
                pin,
                kind: GraphPinKind::Input,
            });
        }
        if let Some((node, pin)) = self.output_pin_hits.get(&id).copied() {
            return Some(GraphHitTarget::Pin {
                node,
                pin,
                kind: GraphPinKind::Output,
            });
        }
        None
    }

    pub fn pin_center(&self, pin: GraphPinRef) -> Option<Vec2> {
        self.pin_centers.get(&pin).copied()
    }

    pub fn output_pin_center(&self, node: UiId, pin: UiId) -> Option<Vec2> {
        self.pin_center(GraphPinRef::output(node, pin))
    }

    pub fn input_pin_center(&self, node: UiId, pin: UiId) -> Option<Vec2> {
        self.pin_center(GraphPinRef::input(node, pin))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GraphInteractionInput {
    pub pointer_position: Option<Vec2>,
    pub pointer_down: bool,
    pub pointer_pressed: bool,
    pub pointer_released: bool,
    pub scroll_delta: Vec2,
    pub hovered: Option<UiId>,
    pub active: Option<UiId>,
    pub clicked: Option<UiId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphPendingConnection {
    pub from_node: UiId,
    pub from_pin: UiId,
}

#[derive(Clone, Copy, Debug)]
pub struct GraphPreviewEdge {
    pub from_node: UiId,
    pub from_pin: UiId,
    pub to_screen: Vec2,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GraphInteractionOutput {
    pub changed: bool,
    pub selection_changed: bool,
    pub moved_nodes: bool,
    pub panned: bool,
    pub zoomed: bool,
    pub connected: Option<GraphEdge>,
}

#[derive(Clone, Debug, Default)]
pub struct GraphInteractionController {
    dragging_node: Option<UiId>,
    panning_canvas: bool,
    pending_connection: Option<GraphPendingConnection>,
    pending_connection_pointer: Option<Vec2>,
    last_pointer: Option<Vec2>,
}

impl GraphInteractionController {
    pub fn pending_connection(&self) -> Option<GraphPendingConnection> {
        self.pending_connection
    }

    pub fn preview_edge(&self) -> Option<GraphPreviewEdge> {
        let pending = self.pending_connection?;
        let to_screen = self.pending_connection_pointer?;
        Some(GraphPreviewEdge {
            from_node: pending.from_node,
            from_pin: pending.from_pin,
            to_screen,
        })
    }

    pub fn clear_transient_state(&mut self) {
        self.dragging_node = None;
        self.panning_canvas = false;
        self.pending_connection = None;
        self.pending_connection_pointer = None;
        self.last_pointer = None;
    }

    pub fn update(
        &mut self,
        frame: &RetainedGraphFrame,
        graph: &mut GraphState,
        input: GraphInteractionInput,
    ) -> GraphInteractionOutput {
        let mut output = GraphInteractionOutput::default();

        let pointer = input.pointer_position;
        let pointer_inside = pointer
            .map(|pointer| frame.viewport.contains(pointer))
            .unwrap_or(false);
        let hovered_target = input.hovered.and_then(|id| frame.target_for_hit(id));
        let active_target = input.active.and_then(|id| frame.target_for_hit(id));
        let clicked_target = input.clicked.and_then(|id| frame.target_for_hit(id));

        if pointer_inside && input.scroll_delta.y.abs() > f32::EPSILON {
            let old_zoom = graph.zoom.max(0.2);
            let zoom_delta = 1.0 + input.scroll_delta.y * 0.08;
            let new_zoom = if zoom_delta > 0.0 {
                (old_zoom * zoom_delta).clamp(0.2, 3.5)
            } else {
                old_zoom
            };
            if (new_zoom - old_zoom).abs() > f32::EPSILON {
                if let Some(pointer) = pointer {
                    let viewport_origin = Vec2::new(frame.viewport.x, frame.viewport.y);
                    let world_under_cursor = (pointer - viewport_origin - graph.pan) / old_zoom;
                    graph.pan = pointer - viewport_origin - world_under_cursor * new_zoom;
                    output.panned = true;
                }
                graph.zoom = new_zoom;
                output.zoomed = true;
                output.changed = true;
            }
        }

        if input.pointer_pressed {
            self.last_pointer = pointer;
            self.dragging_node = None;
            self.panning_canvas = false;

            match active_target {
                Some(GraphHitTarget::NodeBody { node }) => {
                    if graph.select_only(node) {
                        output.selection_changed = true;
                        output.changed = true;
                    }
                    self.dragging_node = Some(node);
                }
                Some(GraphHitTarget::Pin {
                    node,
                    pin,
                    kind: GraphPinKind::Output,
                }) => {
                    if graph.select_only(node) {
                        output.selection_changed = true;
                        output.changed = true;
                    }
                    self.pending_connection = Some(GraphPendingConnection {
                        from_node: node,
                        from_pin: pin,
                    });
                    self.pending_connection_pointer = pointer;
                }
                Some(GraphHitTarget::Canvas) => {
                    if graph.clear_selection() {
                        output.selection_changed = true;
                        output.changed = true;
                    }
                    self.panning_canvas = true;
                }
                _ => {
                    if pointer_inside && graph.clear_selection() {
                        output.selection_changed = true;
                        output.changed = true;
                    }
                }
            }
        }

        if input.pointer_down {
            if let Some(pointer) = pointer {
                if let Some(previous) = self.last_pointer {
                    let delta_screen = pointer - previous;
                    if let Some(node_id) = self.dragging_node {
                        if delta_screen.length_squared() > f32::EPSILON {
                            let zoom = graph.zoom.max(0.2);
                            if let Some(node) = graph.node_mut(node_id) {
                                node.position += delta_screen / zoom;
                                output.moved_nodes = true;
                                output.changed = true;
                            }
                        }
                    } else if self.panning_canvas {
                        if delta_screen.length_squared() > f32::EPSILON {
                            graph.pan += delta_screen;
                            output.panned = true;
                            output.changed = true;
                        }
                    }
                }

                if self.pending_connection.is_some() {
                    self.pending_connection_pointer = Some(pointer);
                }
                self.last_pointer = Some(pointer);
            }
        }

        if input.pointer_released {
            if let Some(pending) = self.pending_connection.take() {
                if let Some(GraphHitTarget::Pin {
                    node,
                    pin,
                    kind: GraphPinKind::Input,
                }) = hovered_target
                {
                    let edge = GraphEdge {
                        from_node: pending.from_node,
                        from_pin: pending.from_pin,
                        to_node: node,
                        to_pin: pin,
                    };
                    let edge_count_before = graph.edges.len();
                    graph.connect(edge);
                    if graph.edges.len() != edge_count_before {
                        output.connected = Some(edge);
                        output.changed = true;
                    }
                }
            }

            if let Some(GraphHitTarget::NodeBody { node }) = clicked_target {
                if graph.select_only(node) {
                    output.selection_changed = true;
                    output.changed = true;
                }
            }

            self.pending_connection_pointer = None;
            self.dragging_node = None;
            self.panning_canvas = false;
            self.last_pointer = pointer;
        }

        if !input.pointer_down && !input.pointer_pressed && !input.pointer_released {
            self.dragging_node = None;
            self.panning_canvas = false;
            self.pending_connection_pointer = None;
            self.last_pointer = pointer;
        }

        output
    }
}

#[derive(Clone, Debug)]
pub struct RetainedGraphBuilder {
    pub style: GraphStyle,
}

impl Default for RetainedGraphBuilder {
    fn default() -> Self {
        Self {
            style: GraphStyle::default(),
        }
    }
}

impl RetainedGraphBuilder {
    pub fn build(
        &self,
        retained: &mut RetainedUi,
        root_id: UiId,
        viewport: UiRect,
        state: &GraphState,
    ) {
        let _ = self.build_frame(retained, root_id, viewport, state, None);
    }

    pub fn build_frame(
        &self,
        retained: &mut RetainedUi,
        root_id: UiId,
        viewport: UiRect,
        state: &GraphState,
        preview: Option<GraphPreviewEdge>,
    ) -> RetainedGraphFrame {
        retained.remove_subtree(root_id);

        let mut frame = RetainedGraphFrame::new(root_id, viewport);
        let canvas_id = root_id.child("canvas");
        frame.canvas_hit_id = canvas_id.child("hit");

        retained.upsert(RetainedUiNode::new(
            root_id,
            UiWidget::Container,
            UiStyle {
                layout: absolute_layout(viewport.x, viewport.y, viewport.width, viewport.height),
                visual: UiVisualStyle::default(),
            },
        ));
        retained.upsert(RetainedUiNode::new(
            canvas_id,
            UiWidget::Container,
            UiStyle {
                layout: fill_parent_layout(),
                visual: UiVisualStyle {
                    background: Some(self.style.canvas_background),
                    border_color: Some(self.style.canvas_border),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: true,
                },
            },
        ));
        retained.upsert(RetainedUiNode::new(
            frame.canvas_hit_id,
            UiWidget::Image(UiImage {
                tint: UiColor::TRANSPARENT,
                ..UiImage::default()
            }),
            UiStyle {
                layout: fill_parent_layout(),
                visual: UiVisualStyle::default(),
            },
        ));

        retained.set_children(root_id, [canvas_id]);

        let mut canvas_children = vec![frame.canvas_hit_id];
        let mut node_order: Vec<UiId> = state.nodes.keys().copied().collect();
        node_order.sort_by_key(|id| id.0);

        for (edge_index, edge) in state.edges.iter().enumerate() {
            let Some((from, to)) = self.edge_points(state, viewport, *edge) else {
                continue;
            };
            self.emit_edge(
                retained,
                root_id.child("edge").child(edge_index as u64),
                from,
                to,
                state,
                &mut canvas_children,
            );
        }

        if let Some(preview) = preview
            && let Some(from) =
                self.output_pin_point(state, viewport, preview.from_node, preview.from_pin)
        {
            self.emit_edge(
                retained,
                root_id.child("edge-preview"),
                from,
                preview.to_screen,
                state,
                &mut canvas_children,
            );
        }

        for node_id in &node_order {
            let Some(node) = state.nodes.get(node_id) else {
                continue;
            };

            let zoom_scale = state.zoom.max(0.75);
            let scaled_pos = viewport_offset(state, node.position);
            let scaled_size = (node.size * state.zoom.max(0.1)).max(Vec2::splat(40.0));
            let node_origin = Vec2::new(viewport.x + scaled_pos.x, viewport.y + scaled_pos.y);

            let body_id = node.id.child("body");
            let body_hit_id = node.id.child("body-hit");
            let title_id = node.id.child("title");

            frame.node_hits.insert(body_hit_id, node.id);

            retained.upsert(RetainedUiNode::new(
                body_id,
                UiWidget::Container,
                UiStyle {
                    layout: absolute_layout(
                        node_origin.x,
                        node_origin.y,
                        scaled_size.x,
                        scaled_size.y,
                    ),
                    visual: UiVisualStyle {
                        background: Some(if node.selected {
                            self.style.node_selected_background
                        } else {
                            self.style.node_background
                        }),
                        border_color: Some(self.style.node_border),
                        border_width: 1.0,
                        corner_radius: 5.0,
                        clip: true,
                    },
                },
            ));

            retained.upsert(RetainedUiNode::new(
                body_hit_id,
                UiWidget::Image(UiImage {
                    tint: UiColor::TRANSPARENT,
                    ..UiImage::default()
                }),
                UiStyle {
                    layout: absolute_layout(0.0, 0.0, scaled_size.x, scaled_size.y),
                    visual: UiVisualStyle::default(),
                },
            ));

            retained.upsert(RetainedUiNode::new(
                title_id,
                UiWidget::Label(UiLabel {
                    text: UiTextValue::from(node.title.clone()),
                    style: UiTextStyle {
                        color: self.style.text_color,
                        font_size: 14.0 * zoom_scale,
                        align_h: UiTextAlign::Start,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                UiStyle {
                    layout: absolute_layout(
                        self.style.node_padding,
                        2.0,
                        (scaled_size.x - self.style.node_padding * 2.0).max(12.0),
                        self.style.title_height * zoom_scale,
                    ),
                    visual: UiVisualStyle {
                        background: Some(self.style.title_background),
                        border_color: None,
                        border_width: 0.0,
                        corner_radius: 3.0,
                        clip: false,
                    },
                },
            ));

            let mut children = vec![body_hit_id, title_id];
            let row_height = self.style.row_height * zoom_scale;
            let pin_size = self.style.pin_size * zoom_scale;
            let header_height = self.style.title_height * zoom_scale;

            for (row, input) in node.inputs.iter().enumerate() {
                let pin_id = node.id.child("in").child(row as u64);
                let pin_hit_id = pin_id.child("hit");
                let label_id = pin_id.child("label");
                let pin_x = 6.0;
                let pin_y = header_height + row as f32 * row_height + 8.0;

                retained.upsert(RetainedUiNode::new(
                    pin_id,
                    UiWidget::Container,
                    UiStyle {
                        layout: absolute_layout(pin_x, pin_y, pin_size, pin_size),
                        visual: UiVisualStyle {
                            background: Some(self.style.pin_color),
                            border_color: Some(self.style.node_border),
                            border_width: 1.0,
                            corner_radius: 2.0,
                            clip: false,
                        },
                    },
                ));
                retained.upsert(RetainedUiNode::new(
                    pin_hit_id,
                    UiWidget::Image(UiImage {
                        tint: UiColor::TRANSPARENT,
                        ..UiImage::default()
                    }),
                    UiStyle {
                        layout: absolute_layout(pin_x, pin_y, pin_size, pin_size),
                        visual: UiVisualStyle::default(),
                    },
                ));
                retained.upsert(RetainedUiNode::new(
                    label_id,
                    UiWidget::Label(UiLabel {
                        text: UiTextValue::from(input.label.clone()),
                        style: UiTextStyle {
                            color: self.style.text_color,
                            font_size: 12.0 * zoom_scale,
                            align_h: UiTextAlign::Start,
                            align_v: UiTextAlign::Center,
                            wrap: false,
                        },
                    }),
                    UiStyle {
                        layout: absolute_layout(
                            pin_size + 12.0,
                            pin_y - 6.0,
                            (scaled_size.x * 0.48).max(20.0),
                            row_height.max(12.0),
                        ),
                        visual: UiVisualStyle::default(),
                    },
                ));

                frame.input_pin_hits.insert(pin_hit_id, (node.id, input.id));
                frame.pin_centers.insert(
                    GraphPinRef::input(node.id, input.id),
                    Vec2::new(
                        node_origin.x + pin_x + pin_size * 0.5,
                        node_origin.y + pin_y + pin_size * 0.5,
                    ),
                );

                children.push(pin_id);
                children.push(pin_hit_id);
                children.push(label_id);
            }

            for (row, output_pin) in node.outputs.iter().enumerate() {
                let pin_id = node.id.child("out").child(row as u64);
                let pin_hit_id = pin_id.child("hit");
                let label_id = pin_id.child("label");
                let pin_y = header_height + row as f32 * row_height + 8.0;
                let pin_x = scaled_size.x - pin_size - 6.0;

                retained.upsert(RetainedUiNode::new(
                    pin_id,
                    UiWidget::Container,
                    UiStyle {
                        layout: absolute_layout(pin_x, pin_y, pin_size, pin_size),
                        visual: UiVisualStyle {
                            background: Some(self.style.pin_color),
                            border_color: Some(self.style.node_border),
                            border_width: 1.0,
                            corner_radius: 2.0,
                            clip: false,
                        },
                    },
                ));
                retained.upsert(RetainedUiNode::new(
                    pin_hit_id,
                    UiWidget::Image(UiImage {
                        tint: UiColor::TRANSPARENT,
                        ..UiImage::default()
                    }),
                    UiStyle {
                        layout: absolute_layout(pin_x, pin_y, pin_size, pin_size),
                        visual: UiVisualStyle::default(),
                    },
                ));
                retained.upsert(RetainedUiNode::new(
                    label_id,
                    UiWidget::Label(UiLabel {
                        text: UiTextValue::from(output_pin.label.clone()),
                        style: UiTextStyle {
                            color: self.style.text_color,
                            font_size: 12.0 * zoom_scale,
                            align_h: UiTextAlign::End,
                            align_v: UiTextAlign::Center,
                            wrap: false,
                        },
                    }),
                    UiStyle {
                        layout: absolute_layout(
                            (scaled_size.x * 0.45).max(20.0),
                            pin_y - 6.0,
                            (scaled_size.x * 0.48).max(20.0),
                            row_height.max(12.0),
                        ),
                        visual: UiVisualStyle::default(),
                    },
                ));

                frame
                    .output_pin_hits
                    .insert(pin_hit_id, (node.id, output_pin.id));
                frame.pin_centers.insert(
                    GraphPinRef::output(node.id, output_pin.id),
                    Vec2::new(
                        node_origin.x + pin_x + pin_size * 0.5,
                        node_origin.y + pin_y + pin_size * 0.5,
                    ),
                );

                children.push(pin_id);
                children.push(pin_hit_id);
                children.push(label_id);
            }

            retained.set_children(body_id, children);
            canvas_children.push(body_id);
        }

        retained.set_children(canvas_id, canvas_children);
        frame
    }

    fn emit_edge(
        &self,
        retained: &mut RetainedUi,
        edge_base_id: UiId,
        from: Vec2,
        to: Vec2,
        state: &GraphState,
        canvas_children: &mut Vec<UiId>,
    ) {
        let thickness = self.style.edge_thickness.max(1.0) * state.zoom.max(0.75);
        let mid_x = (from.x + to.x) * 0.5;
        let segments = [
            UiRect {
                x: from.x.min(mid_x),
                y: from.y - thickness * 0.5,
                width: (mid_x - from.x).abs().max(thickness),
                height: thickness,
            },
            UiRect {
                x: mid_x - thickness * 0.5,
                y: from.y.min(to.y),
                width: thickness,
                height: (to.y - from.y).abs().max(thickness),
            },
            UiRect {
                x: mid_x.min(to.x),
                y: to.y - thickness * 0.5,
                width: (to.x - mid_x).abs().max(thickness),
                height: thickness,
            },
        ];

        for (segment_index, segment) in segments.into_iter().enumerate() {
            let edge_id = edge_base_id.child(segment_index as u64);
            retained.upsert(RetainedUiNode::new(
                edge_id,
                UiWidget::Container,
                UiStyle {
                    layout: absolute_layout(segment.x, segment.y, segment.width, segment.height),
                    visual: UiVisualStyle {
                        background: Some(self.style.edge_color),
                        border_color: None,
                        border_width: 0.0,
                        corner_radius: 1.0,
                        clip: false,
                    },
                },
            ));
            canvas_children.push(edge_id);
        }
    }

    fn output_pin_point(
        &self,
        state: &GraphState,
        viewport: UiRect,
        node_id: UiId,
        pin_id: UiId,
    ) -> Option<Vec2> {
        let node = state.node(node_id)?;
        let pin_index = node.outputs.iter().position(|pin| pin.id == pin_id)?;
        let zoom = state.zoom.max(0.1);
        let zoom_scale = state.zoom.max(0.75);
        let node_pos = viewport_offset(state, node.position);
        let node_size = (node.size * zoom).max(Vec2::splat(40.0));
        let pin_size = self.style.pin_size * zoom_scale;
        let row_height = self.style.row_height * zoom_scale;
        let header = self.style.title_height * zoom_scale;
        let pin_x = node_size.x - pin_size - 6.0;
        let pin_y = header + pin_index as f32 * row_height + 8.0;

        Some(Vec2::new(
            viewport.x + node_pos.x + pin_x + pin_size * 0.5,
            viewport.y + node_pos.y + pin_y + pin_size * 0.5,
        ))
    }

    fn edge_points(
        &self,
        state: &GraphState,
        viewport: UiRect,
        edge: GraphEdge,
    ) -> Option<(Vec2, Vec2)> {
        let from = self.output_pin_point(state, viewport, edge.from_node, edge.from_pin)?;

        let to_node = state.node(edge.to_node)?;
        let to_pin_index = to_node
            .inputs
            .iter()
            .position(|pin| pin.id == edge.to_pin)?;
        let zoom = state.zoom.max(0.1);
        let zoom_scale = state.zoom.max(0.75);
        let to_pos = viewport_offset(state, to_node.position);
        let pin_size = self.style.pin_size * zoom_scale;
        let row_height = self.style.row_height * zoom_scale;
        let header = self.style.title_height * zoom_scale;
        let to_y = header + to_pin_index as f32 * row_height + 8.0;

        let to = Vec2::new(
            viewport.x + to_pos.x + pin_size * 0.5,
            viewport.y + to_pos.y + to_y + pin_size * 0.5,
        );

        let _ = zoom;
        Some((from, to))
    }
}

fn viewport_offset(state: &GraphState, node_position: Vec2) -> Vec2 {
    node_position * state.zoom.max(0.1) + state.pan
}

fn absolute_layout(x: f32, y: f32, width: f32, height: f32) -> UiLayoutStyle {
    let mut layout = UiLayoutStyle::default();
    layout.position_type = UiPositionType::Absolute;
    layout.left = UiDimension::points(x);
    layout.top = UiDimension::points(y);
    layout.width = UiDimension::points(width.max(0.0));
    layout.height = UiDimension::points(height.max(0.0));
    layout
}

fn fill_parent_layout() -> UiLayoutStyle {
    let mut layout = UiLayoutStyle::default();
    layout.width = UiDimension::percent(1.0);
    layout.height = UiDimension::percent(1.0);
    layout
}
