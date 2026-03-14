use bevy_ecs::prelude::{Entity, Query, Res, Without};
use std::collections::HashSet;

use crate::components::{ActiveCamera, AudioEmitter, AudioListener, Transform};
use helmer_audio::{
    AudioEmitterSettings, AudioEmitterSnapshot, AudioListenerSettings, AudioPlaybackState,
};

use crate::systems::scene_system::{SceneChild, SceneRoot};
use crate::{AudioBackendResource, BecsAssetServerParam, BecsSystemProfiler, DeltaTime};

pub fn audio_system(
    _dt: Res<DeltaTime>,
    asset_server: BecsAssetServerParam<'_>,
    audio_backend: Res<AudioBackendResource>,
    listeners: Query<(&Transform, &AudioListener)>,
    cameras: Query<(&Transform, &ActiveCamera), Without<AudioListener>>,
    mut emitters: Query<(
        Entity,
        &Transform,
        &mut AudioEmitter,
        Option<&SceneChild>,
        Option<&SceneRoot>,
    )>,
    system_profiler: Option<Res<BecsSystemProfiler>>,
) {
    let _system_scope = system_profiler
        .as_ref()
        .and_then(|profiler| profiler.0.begin_scope("helmer_becs::systems::audio_system"));

    if !audio_backend.0.enabled() {
        return;
    }

    let finished_emitters = audio_backend.0.drain_finished_emitters();
    let finished_set = if finished_emitters.is_empty() {
        None
    } else {
        Some(finished_emitters.into_iter().collect::<HashSet<u64>>())
    };

    let listener_transform = listeners
        .iter()
        .find(|(_, listener)| listener.enabled)
        .map(|(transform, _)| *transform)
        .or_else(|| cameras.iter().next().map(|(transform, _)| *transform));

    let listener_settings = listener_transform.map(|listener| AudioListenerSettings {
        position: listener.position,
        forward: listener.forward(),
        up: listener.up(),
    });

    let asset_server = asset_server.0.lock();
    let mut snapshots: Vec<AudioEmitterSnapshot> = Vec::new();

    for (entity, transform, mut emitter, scene_child, scene_root) in emitters.iter_mut() {
        if let Some(finished_set) = finished_set.as_ref() {
            if finished_set.contains(&entity.to_bits()) {
                emitter.playback_state = AudioPlaybackState::Stopped;
                emitter.play_on_spawn = false;
            }
        }
        if emitter.play_on_spawn {
            emitter.play_on_spawn = false;
            emitter.playback_state = AudioPlaybackState::Playing;
        }

        let Some(clip_id) = emitter.clip_id else {
            continue;
        };

        let Some(clip) = asset_server.get_audio(clip_id) else {
            continue;
        };

        let scene_id = scene_child
            .map(|child| child.scene_root.to_bits())
            .or_else(|| scene_root.map(|_| entity.to_bits()));
        let settings = AudioEmitterSettings {
            bus: emitter.bus,
            volume: emitter.volume,
            pitch: emitter.pitch,
            looping: emitter.looping,
            spatial: emitter.spatial,
            min_distance: emitter.min_distance,
            max_distance: emitter.max_distance,
            rolloff: emitter.rolloff,
            spatial_blend: emitter.spatial_blend,
            playback_state: emitter.playback_state,
            play_on_spawn: emitter.play_on_spawn,
            scene_id,
        };

        snapshots.push(AudioEmitterSnapshot {
            entity_id: entity.to_bits(),
            clip: Some(clip),
            settings,
            position: transform.position,
        });
    }

    audio_backend.0.send_frame(listener_settings, snapshots);
}
