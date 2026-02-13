use bevy_ecs::prelude::{Entity, Query, Res, Without};
use std::collections::HashSet;

use helmer::audio::{
    AudioEmitterSettings, AudioEmitterSnapshot, AudioListenerSettings, AudioPlaybackState,
};
use helmer::provided::components::{ActiveCamera, AudioEmitter, AudioListener, Transform};

use crate::systems::scene_system::{SceneChild, SceneRoot};
use crate::{
    AudioBackendResource, BevyAssetServerParam, BevySystemProfiler, BevyWrapper, DeltaTime,
};

pub fn audio_system(
    _dt: Res<DeltaTime>,
    asset_server: BevyAssetServerParam<'_>,
    audio_backend: Res<AudioBackendResource>,
    listeners: Query<(&BevyWrapper<Transform>, &BevyWrapper<AudioListener>)>,
    cameras: Query<
        (&BevyWrapper<Transform>, &BevyWrapper<ActiveCamera>),
        Without<BevyWrapper<AudioListener>>,
    >,
    mut emitters: Query<(
        Entity,
        &BevyWrapper<Transform>,
        &mut BevyWrapper<AudioEmitter>,
        Option<&SceneChild>,
        Option<&SceneRoot>,
    )>,
    system_profiler: Option<Res<BevySystemProfiler>>,
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
        .find(|(_, listener)| listener.0.enabled)
        .map(|(transform, _)| transform.0)
        .or_else(|| cameras.iter().next().map(|(transform, _)| transform.0));

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
                emitter.0.playback_state = AudioPlaybackState::Stopped;
                emitter.0.play_on_spawn = false;
            }
        }
        if emitter.0.play_on_spawn {
            emitter.0.play_on_spawn = false;
            emitter.0.playback_state = AudioPlaybackState::Playing;
        }

        let Some(clip_id) = emitter.0.clip_id else {
            continue;
        };

        let Some(clip) = asset_server.get_audio(clip_id) else {
            continue;
        };

        let scene_id = scene_child
            .map(|child| child.scene_root.to_bits())
            .or_else(|| scene_root.map(|_| entity.to_bits()));
        let settings = AudioEmitterSettings {
            bus: emitter.0.bus,
            volume: emitter.0.volume,
            pitch: emitter.0.pitch,
            looping: emitter.0.looping,
            spatial: emitter.0.spatial,
            min_distance: emitter.0.min_distance,
            max_distance: emitter.0.max_distance,
            rolloff: emitter.0.rolloff,
            spatial_blend: emitter.0.spatial_blend,
            playback_state: emitter.0.playback_state,
            play_on_spawn: emitter.0.play_on_spawn,
            scene_id,
        };

        snapshots.push(AudioEmitterSnapshot {
            entity_id: entity.to_bits(),
            clip: Some(clip),
            settings,
            position: transform.0.position,
        });
    }

    audio_backend.0.send_frame(listener_settings, snapshots);
}
