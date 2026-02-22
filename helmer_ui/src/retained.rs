use hashbrown::{HashMap, HashSet};

use crate::{IntoUiId, UiId, UiNode, UiStyle, UiWidget};

#[derive(Clone, Debug)]
pub struct RetainedUiNode {
    pub id: UiId,
    pub widget: UiWidget,
    pub style: UiStyle,
    pub enabled: bool,
    pub children: Vec<UiId>,
}

impl RetainedUiNode {
    pub fn new(id: UiId, widget: UiWidget, style: UiStyle) -> Self {
        Self {
            id,
            widget,
            style,
            enabled: true,
            children: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RetainedUi {
    roots: Vec<UiId>,
    nodes: HashMap<UiId, RetainedUiNode>,
}

impl RetainedUi {
    pub fn clear(&mut self) {
        self.roots.clear();
        self.nodes.clear();
    }

    pub fn roots(&self) -> &[UiId] {
        &self.roots
    }

    pub fn set_roots<I, K>(&mut self, roots: I)
    where
        I: IntoIterator<Item = K>,
        K: IntoUiId,
    {
        self.roots.clear();
        self.roots
            .extend(roots.into_iter().map(|id| id.into_ui_id()));
    }

    pub fn add_root<K: IntoUiId>(&mut self, root: K) {
        self.roots.push(root.into_ui_id());
    }

    pub fn upsert(&mut self, node: RetainedUiNode) {
        self.nodes.insert(node.id, node);
    }

    pub fn node(&self, id: UiId) -> Option<&RetainedUiNode> {
        self.nodes.get(&id)
    }

    pub fn node_mut(&mut self, id: UiId) -> Option<&mut RetainedUiNode> {
        self.nodes.get_mut(&id)
    }

    pub fn set_children<I, K>(&mut self, parent: UiId, children: I)
    where
        I: IntoIterator<Item = K>,
        K: IntoUiId,
    {
        if let Some(parent_node) = self.nodes.get_mut(&parent) {
            parent_node.children.clear();
            parent_node
                .children
                .extend(children.into_iter().map(|id| id.into_ui_id()));
        }
    }

    pub fn remove(&mut self, id: UiId) {
        self.nodes.remove(&id);
        self.roots.retain(|candidate| *candidate != id);
        for node in self.nodes.values_mut() {
            node.children.retain(|child| *child != id);
        }
    }

    pub fn remove_subtree(&mut self, root: UiId) {
        let mut pending = vec![root];
        let mut to_remove = HashSet::new();

        while let Some(id) = pending.pop() {
            if !to_remove.insert(id) {
                continue;
            }
            if let Some(node) = self.nodes.get(&id) {
                pending.extend(node.children.iter().copied());
            }
        }

        for id in &to_remove {
            self.nodes.remove(id);
        }
        self.roots.retain(|id| !to_remove.contains(id));
        for node in self.nodes.values_mut() {
            node.children.retain(|id| !to_remove.contains(id));
        }
    }

    pub fn build_roots(&self) -> Vec<UiNode> {
        let mut visiting = HashSet::new();
        self.roots
            .iter()
            .filter_map(|id| self.build_node_recursive(*id, &mut visiting))
            .collect()
    }

    fn build_node_recursive(&self, id: UiId, visiting: &mut HashSet<UiId>) -> Option<UiNode> {
        let source = self.nodes.get(&id)?;
        if !visiting.insert(id) {
            return None;
        }

        let mut children = Vec::with_capacity(source.children.len());
        for child in &source.children {
            if let Some(node) = self.build_node_recursive(*child, visiting) {
                children.push(node);
            }
        }

        visiting.remove(&id);
        Some(UiNode {
            id,
            widget: source.widget.clone(),
            style: source.style.clone(),
            enabled: source.enabled,
            children,
        })
    }
}
