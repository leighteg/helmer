use helmer_ui::{UiId, UiRect};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockTab {
    pub id: UiId,
    pub title: String,
}

impl DockTab {
    pub fn new(id: UiId, title: impl Into<String>) -> Self {
        Self {
            id,
            title: title.into(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DockLeaf {
    tabs: Vec<DockTab>,
    active: usize,
}

impl DockLeaf {
    pub fn with_tab(tab: DockTab) -> Self {
        Self {
            tabs: vec![tab],
            active: 0,
        }
    }

    pub fn tabs(&self) -> &[DockTab] {
        &self.tabs
    }

    pub fn tabs_mut(&mut self) -> &mut Vec<DockTab> {
        &mut self.tabs
    }

    pub fn active_index(&self) -> usize {
        self.active.min(self.tabs.len().saturating_sub(1))
    }

    pub fn active_tab(&self) -> Option<&DockTab> {
        self.tabs.get(self.active_index())
    }

    pub fn set_active_tab(&mut self, id: UiId) -> bool {
        if let Some(index) = self.tabs.iter().position(|tab| tab.id == id) {
            self.active = index;
            return true;
        }
        false
    }

    pub fn push_tab(&mut self, tab: DockTab) {
        self.tabs.push(tab);
        self.active = self.tabs.len().saturating_sub(1);
    }

    pub fn close_tab(&mut self, id: UiId) -> bool {
        if let Some(index) = self.tabs.iter().position(|tab| tab.id == id) {
            self.tabs.remove(index);
            if self.tabs.is_empty() {
                self.active = 0;
            } else if self.active >= self.tabs.len() {
                self.active = self.tabs.len() - 1;
            } else if index < self.active {
                self.active = self.active.saturating_sub(1);
            }
            return true;
        }
        false
    }

    fn contains_tab(&self, id: UiId) -> bool {
        self.tabs.iter().any(|tab| tab.id == id)
    }

    fn is_empty(&self) -> bool {
        self.tabs.is_empty()
    }
}

#[derive(Clone, Debug)]
pub enum DockNode {
    Split {
        axis: DockAxis,
        ratio: f32,
        first: Box<DockNode>,
        second: Box<DockNode>,
    },
    Leaf(DockLeaf),
}

impl DockNode {
    pub fn leaf(tab: DockTab) -> Self {
        Self::Leaf(DockLeaf::with_tab(tab))
    }

    fn first_leaf_mut(&mut self) -> Option<&mut DockLeaf> {
        match self {
            Self::Split { first, second, .. } => {
                first.first_leaf_mut().or_else(|| second.first_leaf_mut())
            }
            Self::Leaf(leaf) => Some(leaf),
        }
    }

    fn find_leaf_mut_for_tab(&mut self, id: UiId) -> Option<&mut DockLeaf> {
        match self {
            Self::Split { first, second, .. } => first
                .find_leaf_mut_for_tab(id)
                .or_else(|| second.find_leaf_mut_for_tab(id)),
            Self::Leaf(leaf) => leaf.contains_tab(id).then_some(leaf),
        }
    }

    fn split_leaf_for_tab(
        &mut self,
        focus: Option<UiId>,
        axis: DockAxis,
        ratio: f32,
        new_tab: DockTab,
        place_after: bool,
    ) -> bool {
        match self {
            Self::Split { first, second, .. } => {
                first.split_leaf_for_tab(focus, axis, ratio, new_tab.clone(), place_after)
                    || second.split_leaf_for_tab(focus, axis, ratio, new_tab, place_after)
            }
            Self::Leaf(leaf) => {
                let target = focus
                    .map(|id| leaf.contains_tab(id))
                    .unwrap_or_else(|| !leaf.tabs().is_empty());
                if !target {
                    return false;
                }
                let old_leaf = DockLeaf {
                    tabs: leaf.tabs().to_vec(),
                    active: leaf.active_index(),
                };
                let new_leaf = DockLeaf::with_tab(new_tab);
                let ratio = ratio.clamp(0.05, 0.95);
                let (first, second) = if place_after {
                    (
                        Box::new(DockNode::Leaf(old_leaf)),
                        Box::new(DockNode::Leaf(new_leaf)),
                    )
                } else {
                    (
                        Box::new(DockNode::Leaf(new_leaf)),
                        Box::new(DockNode::Leaf(old_leaf)),
                    )
                };
                *self = DockNode::Split {
                    axis,
                    ratio,
                    first,
                    second,
                };
                true
            }
        }
    }

    fn close_tab(&mut self, id: UiId) -> bool {
        match self {
            Self::Split { first, second, .. } => {
                let left_removed = first.close_tab(id);
                let right_removed = second.close_tab(id);
                if left_removed || right_removed {
                    collapse_if_empty(self);
                    return true;
                }
                false
            }
            Self::Leaf(leaf) => leaf.close_tab(id),
        }
    }
}

fn collapse_if_empty(node: &mut DockNode) {
    let should_collapse = matches!(
        node,
        DockNode::Split {
            first,
            second,
            ..
        } if matches!(first.as_ref(), DockNode::Leaf(leaf) if leaf.is_empty())
            || matches!(second.as_ref(), DockNode::Leaf(leaf) if leaf.is_empty())
    );
    if !should_collapse {
        return;
    }

    if let DockNode::Split { first, second, .. } = node {
        if matches!(first.as_ref(), DockNode::Leaf(leaf) if leaf.is_empty()) {
            *node = std::mem::replace(second, Box::new(DockNode::Leaf(DockLeaf::default())))
                .as_ref()
                .clone();
            return;
        }
        if matches!(second.as_ref(), DockNode::Leaf(leaf) if leaf.is_empty()) {
            *node = std::mem::replace(first, Box::new(DockNode::Leaf(DockLeaf::default())))
                .as_ref()
                .clone();
        }
    }
}

#[derive(Clone, Debug)]
pub struct DockState {
    root: DockNode,
    focused_tab: Option<UiId>,
}

impl DockState {
    pub fn new(initial_tab: DockTab) -> Self {
        let focused_tab = Some(initial_tab.id);
        Self {
            root: DockNode::leaf(initial_tab),
            focused_tab,
        }
    }

    pub fn root(&self) -> &DockNode {
        &self.root
    }

    pub fn root_mut(&mut self) -> &mut DockNode {
        &mut self.root
    }

    pub fn focused_tab(&self) -> Option<UiId> {
        self.focused_tab
    }

    pub fn set_focused_tab(&mut self, tab: UiId) {
        self.focused_tab = Some(tab);
    }

    pub fn add_tab_to_focused(&mut self, tab: DockTab) {
        if let Some(focused) = self.focused_tab
            && let Some(leaf) = self.root.find_leaf_mut_for_tab(focused)
        {
            leaf.push_tab(tab.clone());
            self.focused_tab = Some(tab.id);
            return;
        }
        if let Some(first) = self.root.first_leaf_mut() {
            first.push_tab(tab.clone());
            self.focused_tab = Some(tab.id);
            return;
        }
        self.root = DockNode::leaf(tab.clone());
        self.focused_tab = Some(tab.id);
    }

    pub fn split_focused(
        &mut self,
        axis: DockAxis,
        ratio: f32,
        new_tab: DockTab,
        place_after: bool,
    ) {
        if self
            .root
            .split_leaf_for_tab(self.focused_tab, axis, ratio, new_tab.clone(), place_after)
        {
            self.focused_tab = Some(new_tab.id);
            return;
        }
        self.root = DockNode::Split {
            axis,
            ratio: ratio.clamp(0.05, 0.95),
            first: Box::new(self.root.clone()),
            second: Box::new(DockNode::leaf(new_tab.clone())),
        };
        self.focused_tab = Some(new_tab.id);
    }

    pub fn close_tab(&mut self, id: UiId) {
        let removed = self.root.close_tab(id);
        if removed && self.focused_tab == Some(id) {
            self.focused_tab = self.find_first_tab_id();
        }
    }

    pub fn activate_tab(&mut self, id: UiId) -> bool {
        if let Some(leaf) = self.root.find_leaf_mut_for_tab(id) {
            let changed = leaf.set_active_tab(id);
            if changed {
                self.focused_tab = Some(id);
            }
            return changed;
        }
        false
    }

    pub fn layout(&self, bounds: UiRect) -> Vec<DockLeafLayout> {
        let mut out = Vec::new();
        layout_node(&self.root, bounds, &mut out);
        out
    }

    pub fn split_handles(&self, bounds: UiRect, thickness: f32) -> Vec<DockSplitHandleLayout> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        layout_split_handles(&self.root, bounds, thickness.max(1.0), &mut path, &mut out);
        out
    }

    pub fn set_split_ratio(&mut self, path: &[u8], ratio: f32) -> bool {
        set_split_ratio_at_path(&mut self.root, path, ratio.clamp(0.05, 0.95))
    }

    fn find_first_tab_id(&self) -> Option<UiId> {
        let leaves = self.layout(UiRect {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        });
        leaves
            .iter()
            .find_map(|leaf| leaf.active.or_else(|| leaf.tabs.first().copied()))
    }
}

#[derive(Clone, Debug)]
pub struct DockLeafLayout {
    pub rect: UiRect,
    pub tabs: Vec<UiId>,
    pub active: Option<UiId>,
}

#[derive(Clone, Debug)]
pub struct DockSplitHandleLayout {
    pub path: Vec<u8>,
    pub axis: DockAxis,
    pub handle_rect: UiRect,
    pub parent_rect: UiRect,
}

fn layout_node(node: &DockNode, rect: UiRect, out: &mut Vec<DockLeafLayout>) {
    match node {
        DockNode::Leaf(leaf) => {
            out.push(DockLeafLayout {
                rect,
                tabs: leaf.tabs().iter().map(|tab| tab.id).collect(),
                active: leaf.active_tab().map(|tab| tab.id),
            });
        }
        DockNode::Split {
            axis,
            ratio,
            first,
            second,
        } => {
            let ratio = ratio.clamp(0.05, 0.95);
            match axis {
                DockAxis::Horizontal => {
                    let split = rect.height * ratio;
                    let first_rect = UiRect {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: split.max(0.0),
                    };
                    let second_rect = UiRect {
                        x: rect.x,
                        y: rect.y + split,
                        width: rect.width,
                        height: (rect.height - split).max(0.0),
                    };
                    layout_node(first, first_rect, out);
                    layout_node(second, second_rect, out);
                }
                DockAxis::Vertical => {
                    let split = rect.width * ratio;
                    let first_rect = UiRect {
                        x: rect.x,
                        y: rect.y,
                        width: split.max(0.0),
                        height: rect.height,
                    };
                    let second_rect = UiRect {
                        x: rect.x + split,
                        y: rect.y,
                        width: (rect.width - split).max(0.0),
                        height: rect.height,
                    };
                    layout_node(first, first_rect, out);
                    layout_node(second, second_rect, out);
                }
            }
        }
    }
}

fn layout_split_handles(
    node: &DockNode,
    rect: UiRect,
    thickness: f32,
    path: &mut Vec<u8>,
    out: &mut Vec<DockSplitHandleLayout>,
) {
    match node {
        DockNode::Leaf(_) => {}
        DockNode::Split {
            axis,
            ratio,
            first,
            second,
        } => {
            let ratio = ratio.clamp(0.05, 0.95);
            let handle_rect = match axis {
                DockAxis::Horizontal => {
                    let split = rect.height * ratio;
                    UiRect {
                        x: rect.x,
                        y: rect.y + split - thickness * 0.5,
                        width: rect.width,
                        height: thickness,
                    }
                }
                DockAxis::Vertical => {
                    let split = rect.width * ratio;
                    UiRect {
                        x: rect.x + split - thickness * 0.5,
                        y: rect.y,
                        width: thickness,
                        height: rect.height,
                    }
                }
            };
            out.push(DockSplitHandleLayout {
                path: path.clone(),
                axis: *axis,
                handle_rect,
                parent_rect: rect,
            });

            match axis {
                DockAxis::Horizontal => {
                    let split = rect.height * ratio;
                    let first_rect = UiRect {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: split.max(0.0),
                    };
                    let second_rect = UiRect {
                        x: rect.x,
                        y: rect.y + split,
                        width: rect.width,
                        height: (rect.height - split).max(0.0),
                    };
                    path.push(0);
                    layout_split_handles(first, first_rect, thickness, path, out);
                    path.pop();
                    path.push(1);
                    layout_split_handles(second, second_rect, thickness, path, out);
                    path.pop();
                }
                DockAxis::Vertical => {
                    let split = rect.width * ratio;
                    let first_rect = UiRect {
                        x: rect.x,
                        y: rect.y,
                        width: split.max(0.0),
                        height: rect.height,
                    };
                    let second_rect = UiRect {
                        x: rect.x + split,
                        y: rect.y,
                        width: (rect.width - split).max(0.0),
                        height: rect.height,
                    };
                    path.push(0);
                    layout_split_handles(first, first_rect, thickness, path, out);
                    path.pop();
                    path.push(1);
                    layout_split_handles(second, second_rect, thickness, path, out);
                    path.pop();
                }
            }
        }
    }
}

fn set_split_ratio_at_path(node: &mut DockNode, path: &[u8], ratio: f32) -> bool {
    if path.is_empty() {
        if let DockNode::Split { ratio: current, .. } = node {
            *current = ratio.clamp(0.05, 0.95);
            return true;
        }
        return false;
    }

    match node {
        DockNode::Split { first, second, .. } => match path[0] {
            0 => set_split_ratio_at_path(first, &path[1..], ratio),
            1 => set_split_ratio_at_path(second, &path[1..], ratio),
            _ => false,
        },
        DockNode::Leaf(_) => false,
    }
}
