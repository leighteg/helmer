use bevy_ecs::name::Name;
use bevy_ecs::prelude::{Component, Entity, Resource, World};

use helmer::provided::components::{ActiveCamera, AudioListener};
use helmer_becs::{BevyActiveCamera, BevyAudioListener, BevyCamera, BevyTransform, BevyWrapper};

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EditorViewportCamera;

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EditorPlayCamera;

#[derive(Resource, Debug, Clone)]
pub struct EditorViewportState {
    pub graph_template: String,
    pub gizmos_in_play: bool,
    pub execute_scripts_in_edit_mode: bool,
    pub show_camera_gizmos: bool,
    pub show_directional_light_gizmos: bool,
    pub show_point_light_gizmos: bool,
    pub show_spot_light_gizmos: bool,
    pub show_spline_paths: bool,
    pub show_spline_points: bool,
}

impl Default for EditorViewportState {
    fn default() -> Self {
        Self {
            graph_template: "debug-graph".to_string(),
            gizmos_in_play: false,
            execute_scripts_in_edit_mode: false,
            show_camera_gizmos: true,
            show_directional_light_gizmos: true,
            show_point_light_gizmos: true,
            show_spot_light_gizmos: true,
            show_spline_paths: true,
            show_spline_points: true,
        }
    }
}

pub fn ensure_viewport_camera(world: &mut World) -> Entity {
    if let Some((entity, _)) = world
        .query::<(Entity, &EditorViewportCamera)>()
        .iter(world)
        .next()
    {
        return entity;
    }

    world
        .spawn((
            EditorViewportCamera,
            BevyTransform::default(),
            BevyCamera::default(),
            Name::new("Viewport Camera"),
        ))
        .id()
}

pub fn activate_viewport_camera(world: &mut World) -> Entity {
    let entity = ensure_viewport_camera(world);
    clear_active_camera(world);
    world
        .entity_mut(entity)
        .insert(BevyWrapper(ActiveCamera {}));
    entity
}

pub fn set_viewport_audio_listener_enabled(world: &mut World, enabled: bool) {
    let entity = ensure_viewport_camera(world);
    if enabled {
        world
            .entity_mut(entity)
            .insert(BevyWrapper(AudioListener { enabled: true }));
        return;
    }
    if let Some(mut listener) = world.get_mut::<BevyAudioListener>(entity) {
        listener.0.enabled = false;
    }
}

pub fn set_play_camera(world: &mut World, entity: Entity) {
    if world.get::<EditorViewportCamera>(entity).is_some() {
        return;
    }

    let existing: Vec<Entity> = world
        .query::<(Entity, &EditorPlayCamera)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .collect();
    for entity in existing {
        world.entity_mut(entity).remove::<EditorPlayCamera>();
    }

    world.entity_mut(entity).insert(EditorPlayCamera);
}

pub fn ensure_play_camera(world: &mut World) -> Option<Entity> {
    let mut fallback = None;
    let mut selected = None;

    let mut query = world.query::<(Entity, &BevyCamera, Option<&EditorPlayCamera>)>();
    for (entity, _, play_camera) in query.iter(world) {
        if world.get::<EditorViewportCamera>(entity).is_some() {
            continue;
        }

        if play_camera.is_some() {
            selected = Some(entity);
            break;
        }

        if fallback.is_none() {
            fallback = Some(entity);
        }
    }

    let target = selected.or(fallback);
    if let Some(entity) = target {
        set_play_camera(world, entity);
    }
    target
}

pub fn activate_play_camera(world: &mut World) -> Option<Entity> {
    let target = ensure_play_camera(world);
    let Some(entity) = target else {
        return Some(activate_viewport_camera(world));
    };

    clear_active_camera(world);
    world
        .entity_mut(entity)
        .insert(BevyWrapper(ActiveCamera {}));
    Some(entity)
}

fn clear_active_camera(world: &mut World) {
    let active_entities: Vec<Entity> = world
        .query::<(Entity, &BevyActiveCamera)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .collect();
    for entity in active_entities {
        world.entity_mut(entity).remove::<BevyActiveCamera>();
    }
}
