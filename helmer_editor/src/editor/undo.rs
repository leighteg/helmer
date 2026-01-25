use std::{
    fs,
    path::{Path, PathBuf},
};

use bevy_ecs::prelude::{Resource, World};
use helmer_becs::provided::ui::inspector::InspectorSelectedEntityResource;

use crate::editor::{
    EditorProject,
    assets::EditorAssetCache,
    scene::{
        EditorSceneState, SceneDocument, WorldState, reset_editor_scene,
        restore_scene_transforms_from_document, serialize_scene, spawn_scene_from_document,
    },
    ui::{
        InspectorNameEditState, InspectorPinnedEntityResource, MaterialEditorCache,
        refresh_material_usage,
    },
};

#[derive(Debug, Clone)]
pub struct SceneSnapshot {
    pub document: SceneDocument,
    pub scene_path: Option<PathBuf>,
    pub scene_name: String,
    pub dirty: bool,
    pub selection_index: Option<usize>,
    pub pinned_index: Option<usize>,
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MaterialSnapshot {
    pub path: PathBuf,
    pub before: String,
    pub after: String,
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub enum UndoEntry {
    Scene(SceneSnapshot),
    Material(MaterialSnapshot),
}

impl UndoEntry {
    fn label(&self) -> Option<&str> {
        match self {
            UndoEntry::Scene(snapshot) => snapshot.label.as_deref(),
            UndoEntry::Material(snapshot) => snapshot.label.as_deref(),
        }
    }
}

#[derive(Debug, Clone)]
enum PendingUndoGroup {
    Scene {
        label: String,
    },
    Material {
        label: String,
        path: PathBuf,
        before: String,
    },
}

#[derive(Resource, Debug, Clone)]
pub struct EditorUndoState {
    pub entries: Vec<UndoEntry>,
    pub cursor: usize,
    pub max_entries: usize,
    pending_group: Option<PendingUndoGroup>,
    pub pending_commit: bool,
}

impl Default for EditorUndoState {
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

impl EditorUndoState {
    pub fn can_undo(&self) -> bool {
        !self.entries.is_empty() && self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        !self.entries.is_empty() && self.cursor + 1 < self.entries.len()
    }

    pub fn undo_label(&self) -> Option<&str> {
        if self.can_undo() {
            self.entries
                .get(self.cursor)
                .and_then(|entry| entry.label())
        } else {
            None
        }
    }

    pub fn redo_label(&self) -> Option<&str> {
        if self.can_redo() {
            self.entries
                .get(self.cursor + 1)
                .and_then(|entry| entry.label())
        } else {
            None
        }
    }
}

pub fn request_begin_undo_group(state: &mut EditorUndoState, label: &str) {
    if state.pending_group.is_none() {
        state.pending_group = Some(PendingUndoGroup::Scene {
            label: label.to_string(),
        });
    }
}

pub fn request_begin_material_undo_group(state: &mut EditorUndoState, path: &Path, label: &str) {
    if state.pending_group.is_none() {
        let before = fs::read_to_string(path).unwrap_or_default();
        state.pending_group = Some(PendingUndoGroup::Material {
            label: label.to_string(),
            path: path.to_path_buf(),
            before,
        });
    }
}

pub fn request_end_undo_group(state: &mut EditorUndoState) {
    state.pending_commit = true;
}

pub fn begin_undo_group(world: &mut World, label: &str) {
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|state| state.world_state);
    if world_state != Some(WorldState::Edit) {
        return;
    }
    let should_flush = world
        .get_resource::<EditorUndoState>()
        .map(|state| matches!(state.pending_group, Some(PendingUndoGroup::Material { .. })))
        .unwrap_or(false);
    if should_flush {
        flush_pending_group(world);
    }
    let Some(mut state) = world.get_resource_mut::<EditorUndoState>() else {
        return;
    };
    request_begin_undo_group(&mut state, label);
}

pub fn begin_material_undo_group(world: &mut World, path: &Path, label: &str) {
    let world_state = world
        .get_resource::<EditorSceneState>()
        .map(|state| state.world_state);
    if world_state != Some(WorldState::Edit) {
        return;
    }
    let should_flush = world
        .get_resource::<EditorUndoState>()
        .map(|state| match state.pending_group.as_ref() {
            Some(PendingUndoGroup::Scene { .. }) => true,
            Some(PendingUndoGroup::Material {
                path: pending_path, ..
            }) => pending_path != path,
            None => false,
        })
        .unwrap_or(false);
    if should_flush {
        flush_pending_group(world);
    }
    let Some(mut state) = world.get_resource_mut::<EditorUndoState>() else {
        return;
    };
    request_begin_material_undo_group(&mut state, path, label);
}

pub fn end_undo_group(world: &mut World) {
    {
        let Some(mut state) = world.get_resource_mut::<EditorUndoState>() else {
            return;
        };
        if state.pending_group.is_none() {
            return;
        }
        state.pending_commit = true;
    }
    commit_pending_group(world);
}

pub fn end_material_undo_group(world: &mut World) {
    let should_commit = {
        let Some(mut state) = world.get_resource_mut::<EditorUndoState>() else {
            return;
        };
        if !matches!(state.pending_group, Some(PendingUndoGroup::Material { .. })) {
            return;
        }
        state.pending_commit = true;
        true
    };
    if should_commit {
        commit_pending_group(world);
    }
}

pub fn push_undo_snapshot(world: &mut World, label: &str) {
    flush_pending_group(world);
    let snapshot = capture_snapshot(world, Some(label.to_string()));
    let Some(snapshot) = snapshot else {
        return;
    };
    push_entry(world, UndoEntry::Scene(snapshot));
}

pub fn reset_undo_history(world: &mut World) {
    let snapshot = capture_snapshot(world, None);
    let Some(snapshot) = snapshot else {
        if let Some(mut state) = world.get_resource_mut::<EditorUndoState>() {
            state.entries.clear();
            state.cursor = 0;
            state.pending_group = None;
            state.pending_commit = false;
        }
        return;
    };

    if let Some(mut state) = world.get_resource_mut::<EditorUndoState>() {
        state.entries.clear();
        state.entries.push(UndoEntry::Scene(snapshot));
        state.cursor = 0;
        state.pending_group = None;
        state.pending_commit = false;
    }
}

pub fn mark_undo_clean(world: &mut World) {
    if let Some(mut state) = world.get_resource_mut::<EditorUndoState>() {
        let mut index = state.cursor;
        loop {
            match state.entries.get_mut(index) {
                Some(UndoEntry::Scene(snapshot)) => {
                    snapshot.dirty = false;
                    break;
                }
                Some(_) => {
                    if index == 0 {
                        break;
                    }
                    index = index.saturating_sub(1);
                }
                None => break,
            }
        }
    }
}

pub fn undo_action(world: &mut World) -> Option<String> {
    {
        let scene_state = world.get_resource::<EditorSceneState>()?;
        if scene_state.world_state != WorldState::Edit {
            return None;
        }
    }

    flush_pending_group(world);

    let label = world
        .get_resource::<EditorUndoState>()
        .and_then(|state| state.undo_label().map(|label| label.to_string()))?;

    let entry = {
        let mut state = world.get_resource_mut::<EditorUndoState>()?;
        if !state.can_undo() {
            return None;
        }
        let entry = state.entries.get(state.cursor).cloned();
        state.cursor = state.cursor.saturating_sub(1);
        match entry {
            Some(UndoEntry::Scene(_)) => {
                find_previous_scene_snapshot(&state.entries, state.cursor).map(UndoEntry::Scene)
            }
            Some(UndoEntry::Material(_)) => entry,
            None => None,
        }
    };

    if let Some(entry) = entry {
        apply_entry(world, &entry, UndoDirection::Undo);
    }

    Some(label)
}

pub fn redo_action(world: &mut World) -> Option<String> {
    {
        let scene_state = world.get_resource::<EditorSceneState>()?;
        if scene_state.world_state != WorldState::Edit {
            return None;
        }
    }

    flush_pending_group(world);

    let label = world
        .get_resource::<EditorUndoState>()
        .and_then(|state| state.redo_label().map(|label| label.to_string()))?;

    let entry = {
        let mut state = world.get_resource_mut::<EditorUndoState>()?;
        if !state.can_redo() {
            return None;
        }
        state.cursor += 1;
        state.entries.get(state.cursor).cloned()
    };

    if let Some(entry) = entry {
        apply_entry(world, &entry, UndoDirection::Redo);
    }

    Some(label)
}

pub fn process_undo_requests(world: &mut World) {
    commit_pending_group(world);
}

pub fn editor_undo_request_system(world: &mut World) {
    process_undo_requests(world);
}

fn flush_pending_group(world: &mut World) {
    let should_commit = {
        let Some(mut state) = world.get_resource_mut::<EditorUndoState>() else {
            return;
        };
        if state.pending_group.is_none() {
            return;
        }
        state.pending_commit = true;
        true
    };

    if should_commit {
        commit_pending_group(world);
    }
}

fn commit_pending_group(world: &mut World) {
    let pending = {
        let Some(mut state) = world.get_resource_mut::<EditorUndoState>() else {
            return;
        };
        if !state.pending_commit {
            return;
        }
        state.pending_commit = false;
        state.pending_group.take()
    };

    let Some(pending) = pending else {
        return;
    };

    match pending {
        PendingUndoGroup::Scene { label } => {
            let snapshot = capture_snapshot(world, Some(label));
            let Some(snapshot) = snapshot else {
                return;
            };
            push_entry(world, UndoEntry::Scene(snapshot));
        }
        PendingUndoGroup::Material {
            label,
            path,
            before,
        } => {
            let after = fs::read_to_string(&path).unwrap_or_default();
            if before == after {
                return;
            }
            let snapshot = MaterialSnapshot {
                path,
                before,
                after,
                label: Some(label),
            };
            push_entry(world, UndoEntry::Material(snapshot));
        }
    }
}

fn capture_snapshot(world: &mut World, label: Option<String>) -> Option<SceneSnapshot> {
    let (scene_path, scene_name, dirty, world_state) = {
        let scene_state = world.get_resource::<EditorSceneState>()?;
        (
            scene_state.path.clone(),
            scene_state.name.clone(),
            scene_state.dirty,
            scene_state.world_state,
        )
    };
    if world_state != WorldState::Edit {
        return None;
    }

    let project = world.get_resource::<EditorProject>()?.clone();
    let (document, entity_order) = serialize_scene(world, &project);

    let selection = world
        .get_resource::<InspectorSelectedEntityResource>()
        .and_then(|selection| selection.0);
    let selection_index =
        selection.and_then(|entity| entity_order.iter().position(|ordered| *ordered == entity));

    let pinned = world
        .get_resource::<InspectorPinnedEntityResource>()
        .and_then(|pinned| pinned.0);
    let pinned_index =
        pinned.and_then(|entity| entity_order.iter().position(|ordered| *ordered == entity));

    Some(SceneSnapshot {
        document,
        scene_path,
        scene_name,
        dirty,
        selection_index,
        pinned_index,
        label,
    })
}

fn push_entry(world: &mut World, entry: UndoEntry) {
    let Some(mut state) = world.get_resource_mut::<EditorUndoState>() else {
        return;
    };

    if state.entries.is_empty() {
        state.entries.push(entry);
        state.cursor = 0;
        return;
    }

    if let Some(current) = state.entries.get(state.cursor) {
        if entries_equivalent(current, &entry) {
            return;
        }
    }

    let cursor = state.cursor;
    if cursor + 1 < state.entries.len() {
        state.entries.truncate(cursor + 1);
    }

    state.entries.push(entry);
    state.cursor = state.entries.len().saturating_sub(1);

    trim_entries(&mut state);
}

fn entries_equivalent(current: &UndoEntry, candidate: &UndoEntry) -> bool {
    match (current, candidate) {
        (UndoEntry::Scene(current), UndoEntry::Scene(candidate)) => {
            current.document == candidate.document
        }
        _ => false,
    }
}

fn find_previous_scene_snapshot(
    entries: &[UndoEntry],
    start_index: usize,
) -> Option<SceneSnapshot> {
    let mut index = start_index;
    loop {
        match entries.get(index) {
            Some(UndoEntry::Scene(snapshot)) => return Some(snapshot.clone()),
            Some(_) => {
                if index == 0 {
                    return None;
                }
                index = index.saturating_sub(1);
            }
            None => return None,
        }
    }
}

fn trim_entries(state: &mut EditorUndoState) {
    let cap = state.max_entries.max(1);
    if state.entries.len() > cap {
        let remove_count = state.entries.len() - cap;
        state.entries.drain(0..remove_count);
        state.cursor = state.cursor.saturating_sub(remove_count);
    }
}

pub fn enforce_undo_cap(state: &mut EditorUndoState) {
    trim_entries(state);
}

#[derive(Debug, Clone, Copy)]
enum UndoDirection {
    Undo,
    Redo,
}

fn apply_entry(world: &mut World, entry: &UndoEntry, direction: UndoDirection) {
    match entry {
        UndoEntry::Scene(snapshot) => apply_snapshot(world, snapshot),
        UndoEntry::Material(snapshot) => apply_material_snapshot(world, snapshot, direction),
    }
}

fn apply_material_snapshot(
    world: &mut World,
    snapshot: &MaterialSnapshot,
    direction: UndoDirection,
) {
    let payload = match direction {
        UndoDirection::Undo => &snapshot.before,
        UndoDirection::Redo => &snapshot.after,
    };
    if let Err(err) = fs::write(&snapshot.path, payload) {
        set_status_message(
            world,
            format!(
                "Failed to apply material history for {}: {}",
                snapshot.path.display(),
                err
            ),
        );
        return;
    }

    if let Some(mut cache) = world.get_resource_mut::<MaterialEditorCache>() {
        cache.entries.remove(&snapshot.path);
    }

    let project = world.get_resource::<EditorProject>().cloned();
    refresh_material_usage(world, &project, &snapshot.path);
}

fn apply_snapshot(world: &mut World, snapshot: &SceneSnapshot) {
    let project_snapshot = match world.get_resource::<EditorProject>() {
        Some(project) => project.clone(),
        None => return,
    };

    reset_editor_scene(world);
    if let Some(mut pinned) = world.get_resource_mut::<InspectorPinnedEntityResource>() {
        pinned.0 = None;
    }

    let created_entities = world.resource_scope::<EditorAssetCache, _>(|world, mut cache| {
        let asset_server = {
            let asset_server = world
                .get_resource::<helmer_becs::BevyAssetServer>()
                .expect("AssetServer missing");
            helmer_becs::BevyAssetServer(asset_server.0.clone())
        };
        spawn_scene_from_document(
            world,
            &snapshot.document,
            &project_snapshot,
            &mut cache,
            &asset_server,
        )
    });
    restore_scene_transforms_from_document(world, &snapshot.document, &created_entities);

    if let Some(mut scene_state) = world.get_resource_mut::<EditorSceneState>() {
        scene_state.path = snapshot.scene_path.clone();
        scene_state.name = snapshot.scene_name.clone();
        scene_state.dirty = snapshot.dirty;
        scene_state.world_state = WorldState::Edit;
        scene_state.play_backup = None;
        scene_state.play_selected_index = None;
    }

    if let Some(mut selection) = world.get_resource_mut::<InspectorSelectedEntityResource>() {
        selection.0 = snapshot
            .selection_index
            .and_then(|index| created_entities.get(index).copied());
    }

    if let Some(mut pinned) = world.get_resource_mut::<InspectorPinnedEntityResource>() {
        pinned.0 = snapshot
            .pinned_index
            .and_then(|index| created_entities.get(index).copied());
    }

    if let Some(mut name_state) = world.get_resource_mut::<InspectorNameEditState>() {
        name_state.entity = None;
        name_state.buffer.clear();
    }
}

fn set_status_message(world: &mut World, message: String) {
    if let Some(mut state) = world.get_resource_mut::<crate::editor::ui::EditorUiState>() {
        state.status = Some(message);
    }
}
