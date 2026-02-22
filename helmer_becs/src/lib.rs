use crossbeam_channel::Sender;
use hashbrown::HashMap;
use std::{collections::VecDeque, path::PathBuf, sync::Arc};
use web_time::Instant;

use bevy_ecs::prelude::{ReflectComponent, ReflectResource};
use bevy_ecs::{
    component::Component,
    reflect::AppTypeRegistry,
    resource::Resource,
    schedule::{IntoScheduleConfigs, Schedule},
    world::World,
};
use bevy_reflect::{Reflect, TypeRegistry, TypeRegistryArc};
use helmer::{
    animation::Animator,
    audio::AudioBackend,
    graphics::common::renderer::{RenderMessage, RendererStats, StreamingTuning},
    provided::components::{
        ActiveCamera, AudioEmitter, AudioListener, Camera, EntityFollower, Light, LookAt,
        MeshRenderer, PoseOverride, SkinnedMeshRenderer, Spline, SplineFollower,
        SpriteImageSequence, SpriteRenderer, Text2d, Transform,
    },
    runtime::{
        asset_server::AssetServer,
        config::RuntimeConfig,
        input_manager::InputManager,
        runtime::{
            PerformanceMetrics, Runtime, RuntimeCursorState, RuntimeProfiling, RuntimeTuning,
            RuntimeWindowControl,
        },
    },
};
use parking_lot::{Mutex, RwLock};

use crate::profiling::SystemProfiler;
use crate::provided::ui::inspector::InspectorSelectedEntityResource;
use crate::systems::animation_system::{SkinningResource, skinning_system};
use crate::systems::audio_system::audio_system;
use crate::systems::follow_system::{entity_follow_system, look_at_system};
use crate::systems::render_system::RenderGraphResource;
use crate::systems::spline_system::spline_follow_system;
use crate::{
    egui_integration::{EguiClipboard, EguiInputPassthrough, EguiResource, egui_system},
    physics::{
        physics_resource::PhysicsResource,
        systems::{
            apply_persistent_forces_system, apply_queued_impulses_system,
            apply_transient_forces_system, character_controller_system, cleanup_physics_system,
            physics_scene_query_system, physics_step_system, sync_entities_to_physics_system,
            sync_joints_to_physics_system, sync_physics_to_entities_system,
            sync_transforms_to_physics_system,
        },
    },
    systems::{
        render_system::RenderObjectCount,
        render_system::{
            RenderMainSceneToSwapchain, RenderPacket, RenderResetRequest, RenderSyncRequest,
            RenderViewportRequests, render_data_system,
        },
        scene_system::{
            SceneSpawnedChildren, apply_scene_commands_system, scene_child_skinning_system,
            scene_spawning_system, update_scene_child_transforms,
        },
    },
    ui_integration::{
        UiClipboard, UiPerfStats, UiRenderFrameOutput, UiRenderState, UiResource, ui_system,
    },
};

/// A generic wrapper to turn existing data structures into Bevy Components or Resources.
#[derive(Component, Resource, Clone, Copy, Debug, Default)]
pub struct BevyWrapper<T>(pub T);

// Component Wrappers
pub type BevyTransform = BevyWrapper<Transform>;
pub type BevyCamera = BevyWrapper<Camera>;
pub type BevyActiveCamera = BevyWrapper<ActiveCamera>;
pub type BevyMeshRenderer = BevyWrapper<MeshRenderer>;
pub type BevyLight = BevyWrapper<Light>;
pub type BevyAudioEmitter = BevyWrapper<AudioEmitter>;
pub type BevyAudioListener = BevyWrapper<AudioListener>;
pub type BevySpriteRenderer = BevyWrapper<SpriteRenderer>;

// Non-Copy components need dedicated wrappers.
#[derive(Component, Clone, Debug)]
pub struct BevySkinnedMeshRenderer(pub SkinnedMeshRenderer);

#[derive(Component, Clone, Debug)]
pub struct BevyAnimator(pub Animator);

#[derive(Component, Clone, Debug)]
pub struct BevyPoseOverride(pub PoseOverride);

#[derive(Component, Clone, Debug)]
pub struct BevySpline(pub Spline);

#[derive(Component, Clone, Debug)]
pub struct BevySplineFollower(pub SplineFollower);

#[derive(Component, Clone, Debug)]
pub struct BevyLookAt(pub LookAt);

#[derive(Component, Clone, Debug)]
pub struct BevyEntityFollower(pub EntityFollower);

#[derive(Component, Clone, Debug)]
pub struct BevyText2d(pub Text2d);

#[derive(Component, Clone, Debug)]
pub struct BevySpriteImageSequence(pub SpriteImageSequence);

// Resource Wrappers
#[cfg_attr(not(target_arch = "wasm32"), derive(Resource))]
pub struct BevyAssetServer(pub Arc<Mutex<AssetServer>>);
#[cfg(target_arch = "wasm32")]
pub type BevyAssetServerParam<'w> = bevy_ecs::system::NonSend<'w, BevyAssetServer>;
#[cfg(not(target_arch = "wasm32"))]
pub type BevyAssetServerParam<'w> = bevy_ecs::prelude::Res<'w, BevyAssetServer>;
#[derive(Resource)]
pub struct BevyInputManager(pub Arc<RwLock<InputManager>>);
#[derive(Resource)]
pub struct BevyPerformanceMetrics(pub Arc<PerformanceMetrics>);
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct BevyRuntimeConfig(pub RuntimeConfig);
#[derive(Resource)]
pub struct BevyRuntimeTuning(pub Arc<RuntimeTuning>);
#[derive(Resource)]
pub struct BevyRuntimeProfiling(pub Arc<RuntimeProfiling>);
#[derive(Resource)]
pub struct BevySystemProfiler(pub Arc<SystemProfiler>);
#[derive(Resource)]
pub struct BevyRuntimeCursorState(pub Arc<RuntimeCursorState>);
#[derive(Resource)]
pub struct BevyRuntimeWindowControl(pub Arc<RuntimeWindowControl>);
#[derive(Resource)]
pub struct BevyRendererStats(pub Arc<RendererStats>);
#[derive(Resource)]
pub struct BevyRenderSender(pub Sender<RenderMessage>);
#[derive(Resource, Clone, Copy, Debug)]
pub struct BevyStreamingTuning(pub StreamingTuning);
#[derive(Resource, Clone, Copy, Debug)]
pub struct BevyLodTuning(pub crate::systems::render_system::LodTuning);
#[derive(Resource, Clone, Copy, Debug)]
pub struct BevyRenderWorkerTuning(pub crate::systems::render_system::RenderWorkerTuning);
#[derive(Resource, Clone, Copy, Debug)]
pub struct BevySceneTuning(pub crate::systems::scene_system::SceneTuning);

impl Default for BevyStreamingTuning {
    fn default() -> Self {
        Self(StreamingTuning::default())
    }
}

impl Default for BevyLodTuning {
    fn default() -> Self {
        Self(crate::systems::render_system::LodTuning::default())
    }
}

impl Default for BevyRenderWorkerTuning {
    fn default() -> Self {
        Self(crate::systems::render_system::RenderWorkerTuning::default())
    }
}

impl Default for BevySceneTuning {
    fn default() -> Self {
        Self(crate::systems::scene_system::SceneTuning::default())
    }
}

#[derive(Resource, Default, Clone)]
pub struct DebugGraphHistory {
    pub vram_bytes: VecDeque<f64>,
    pub mesh_bytes: VecDeque<f64>,
    pub texture_bytes: VecDeque<f64>,
    pub material_bytes: VecDeque<f64>,
    pub audio_bytes: VecDeque<f64>,
    pub fps: VecDeque<f64>,
}

#[derive(Resource, Clone)]
pub struct AudioBackendResource(pub Arc<AudioBackend>);

#[derive(Resource, Default, Clone)]
pub struct ProfilingHistory {
    pub main_event_ms: VecDeque<f64>,
    pub main_update_ms: VecDeque<f64>,
    pub logic_frame_ms: VecDeque<f64>,
    pub logic_asset_ms: VecDeque<f64>,
    pub logic_input_ms: VecDeque<f64>,
    pub logic_tick_ms: VecDeque<f64>,
    pub logic_schedule_ms: VecDeque<f64>,
    pub logic_render_send_ms: VecDeque<f64>,
    pub ecs_render_data_ms: VecDeque<f64>,
    pub ecs_scene_spawn_ms: VecDeque<f64>,
    pub ecs_scene_update_ms: VecDeque<f64>,
    pub render_thread_frame_ms: VecDeque<f64>,
    pub render_thread_messages_ms: VecDeque<f64>,
    pub render_thread_upload_ms: VecDeque<f64>,
    pub render_thread_render_ms: VecDeque<f64>,
    pub render_prepare_globals_ms: VecDeque<f64>,
    pub render_streaming_plan_ms: VecDeque<f64>,
    pub render_occlusion_ms: VecDeque<f64>,
    pub render_graph_ms: VecDeque<f64>,
    pub render_graph_pass_ms: VecDeque<f64>,
    pub render_graph_encoder_create_ms: VecDeque<f64>,
    pub render_graph_encoder_finish_ms: VecDeque<f64>,
    pub render_graph_overhead_ms: VecDeque<f64>,
    pub render_resource_mgmt_ms: VecDeque<f64>,
    pub render_acquire_ms: VecDeque<f64>,
    pub render_submit_ms: VecDeque<f64>,
    pub render_present_ms: VecDeque<f64>,
    pub ui_system_ms: VecDeque<f64>,
    pub ui_run_frame_ms: VecDeque<f64>,
    pub ui_interaction_ms: VecDeque<f64>,
    pub ui_scroll_metrics_ms: VecDeque<f64>,
    pub ui_render_data_convert_ms: VecDeque<f64>,
    pub render_ui_build_ms: VecDeque<f64>,
    pub render_pass_ms: HashMap<String, VecDeque<f64>>,
    pub render_pass_last_ms: HashMap<String, f64>,
    pub render_pass_order: Vec<String>,
    pub audio_mix_ms: VecDeque<f64>,
    pub audio_callback_ms: VecDeque<f64>,
    pub audio_emitters: VecDeque<f64>,
    pub audio_streaming_emitters: VecDeque<f64>,
}

// resources
#[derive(Resource, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct DeltaTime(pub f32);

#[derive(Resource, Clone, Debug, Default)]
pub struct DraggedFile(pub Option<PathBuf>);

pub mod egui_integration;
pub mod physics;
pub mod profiling;
pub mod provided;
pub mod systems;
pub mod ui_integration;

fn helmer_becs_init_impl<F>(
    init_callback: fn(&mut World, &mut Schedule, &AssetServer),
    configure_runtime: F,
) where
    F: FnOnce(&mut Runtime<(World, Schedule)>),
{
    let world = World::new();
    let mut schedule = Schedule::default();
    #[cfg(target_arch = "wasm32")]
    schedule.set_executor_kind(bevy_ecs::schedule::ExecutorKind::SingleThreaded);
    #[cfg(not(target_arch = "wasm32"))]
    schedule.set_executor_kind(bevy_ecs::schedule::ExecutorKind::MultiThreaded);

    let mut type_registry = TypeRegistry::default();

    // Register primitive types for reflection
    type_registry.register::<f32>();
    type_registry.register::<f64>();
    type_registry.register::<i32>();
    type_registry.register::<u32>();
    type_registry.register::<i64>();
    type_registry.register::<u64>();
    type_registry.register::<bool>();
    type_registry.register::<String>();

    // Register Vec types
    type_registry.register::<Vec<f32>>();
    type_registry.register::<Vec<String>>();

    // Register reflectable resources
    type_registry.register::<DeltaTime>();

    let mut runtime: Runtime<(World, Schedule)> = Runtime::new(
        (world, schedule),
        move |runtime, (world, schedule)| {
            let registry_arc: TypeRegistryArc = TypeRegistryArc {
                internal: Arc::new(std::sync::RwLock::new(type_registry)),
            };
            world.insert_resource(AppTypeRegistry(registry_arc));

            world.insert_resource::<BevyRuntimeConfig>(BevyRuntimeConfig(
                runtime.config.as_ref().clone(),
            ));
            world.insert_resource::<BevyRuntimeTuning>(BevyRuntimeTuning(runtime.tuning.clone()));
            world.insert_resource::<BevyRuntimeProfiling>(BevyRuntimeProfiling(
                runtime.profiling.clone(),
            ));
            world.insert_resource::<BevySystemProfiler>(BevySystemProfiler(Arc::new(
                SystemProfiler::default(),
            )));
            world.insert_resource::<BevyRuntimeCursorState>(BevyRuntimeCursorState(
                runtime.cursor_state.clone(),
            ));
            world.insert_resource::<BevyRuntimeWindowControl>(BevyRuntimeWindowControl(
                runtime.window_control.clone(),
            ));
            #[cfg(target_arch = "wasm32")]
            world.insert_non_send_resource::<BevyAssetServer>(BevyAssetServer(
                runtime.asset_server.as_ref().unwrap().clone(),
            ));
            #[cfg(not(target_arch = "wasm32"))]
            world.insert_resource::<BevyAssetServer>(BevyAssetServer(
                runtime.asset_server.as_ref().unwrap().clone(),
            ));
            world.insert_resource::<BevyInputManager>(BevyInputManager(
                runtime.input_manager.clone(),
            ));
            world.insert_resource::<BevyPerformanceMetrics>(BevyPerformanceMetrics(
                runtime.metrics.clone(),
            ));
            world.insert_resource::<BevyRendererStats>(BevyRendererStats(
                runtime.renderer_stats.clone(),
            ));
            world.insert_resource::<BevyRenderSender>(BevyRenderSender(
                runtime.render_thread_sender.clone(),
            ));
            world.insert_resource::<BevyStreamingTuning>(BevyStreamingTuning::default());
            world.insert_resource::<BevyLodTuning>(BevyLodTuning::default());
            world.insert_resource::<BevyRenderWorkerTuning>(BevyRenderWorkerTuning::default());
            world.insert_resource::<BevySceneTuning>(BevySceneTuning::default());
            world.insert_resource::<DebugGraphHistory>(DebugGraphHistory::default());
            world.insert_resource::<ProfilingHistory>(ProfilingHistory::default());
            world.insert_resource::<UiPerfStats>(UiPerfStats::default());
            world.insert_resource::<DeltaTime>(DeltaTime(1.0));
            world.insert_resource::<SkinningResource>(SkinningResource::default());
            world.insert_resource::<RenderPacket>(RenderPacket::default());
            world.insert_resource::<RenderObjectCount>(RenderObjectCount::default());
            world.insert_resource::<RenderResetRequest>(RenderResetRequest::default());
            world.insert_resource::<RenderSyncRequest>(RenderSyncRequest::default());
            world.insert_resource::<RenderViewportRequests>(RenderViewportRequests::default());
            world.insert_resource::<RenderMainSceneToSwapchain>(
                RenderMainSceneToSwapchain::default(),
            );
            world.insert_resource(RenderGraphResource::default());
            world.insert_resource(SceneSpawnedChildren::default());
            world.insert_resource::<EguiResource>(EguiResource::default());
            world.insert_resource::<EguiClipboard>(EguiClipboard::default());
            world.insert_resource::<EguiInputPassthrough>(EguiInputPassthrough::default());
            world.insert_resource::<UiResource>(UiResource::default());
            world.insert_resource::<UiRenderState>(UiRenderState::default());
            world.insert_resource::<UiRenderFrameOutput>(UiRenderFrameOutput::default());
            world.insert_resource::<ui_integration::UiRuntimeState>(
                ui_integration::UiRuntimeState::default(),
            );
            world.insert_resource::<UiClipboard>(UiClipboard::default());
            world.insert_resource::<PhysicsResource>(PhysicsResource::default());
            world.insert_resource::<AudioBackendResource>(AudioBackendResource(Arc::new(
                AudioBackend::new(),
            )));
            world.insert_resource::<DraggedFile>(DraggedFile(None));
            world.insert_resource(InspectorSelectedEntityResource::default());

            if let Some(system_profiler) = world.get_resource::<BevySystemProfiler>() {
                system_profiler.0.register_systems(&[
                    "helmer_becs::systems::scene_spawning_system",
                    "helmer_becs::systems::scene_child_skinning_system",
                    "helmer_becs::systems::apply_scene_commands_system",
                    "helmer_becs::systems::update_scene_child_transforms",
                    "helmer_becs::systems::spline_follow_system",
                    "helmer_becs::systems::entity_follow_system",
                    "helmer_becs::systems::look_at_system",
                    "helmer_becs::systems::skinning_system",
                    "helmer_becs::systems::render_data_system",
                    "helmer_becs::systems::audio_system",
                    "helmer_becs::physics::cleanup_physics_system",
                    "helmer_becs::physics::sync_entities_to_physics_system",
                    "helmer_becs::physics::sync_joints_to_physics_system",
                    "helmer_becs::physics::sync_transforms_to_physics_system",
                    "helmer_becs::physics::character_controller_system",
                    "helmer_becs::physics::apply_transient_forces_system",
                    "helmer_becs::physics::apply_persistent_forces_system",
                    "helmer_becs::physics::apply_queued_impulses_system",
                    "helmer_becs::physics::physics_step_system",
                    "helmer_becs::physics::sync_physics_to_entities_system",
                    "helmer_becs::physics::physics_scene_query_system",
                    "helmer_becs::ui_integration::ui_system",
                    "helmer_becs::egui_integration::egui_system",
                ]);
            }

            // scene spawning/child setup should happen before skinning + render extraction
            schedule.add_systems(
                (
                    scene_spawning_system,
                    scene_child_skinning_system,
                    apply_scene_commands_system,
                    update_scene_child_transforms,
                )
                    .chain()
                    .before(skinning_system)
                    .before(render_data_system),
            );

            // core systems
            schedule.add_systems(
                (
                    spline_follow_system,
                    entity_follow_system,
                    look_at_system,
                    skinning_system,
                    render_data_system,
                )
                    .chain(),
            );
            schedule.add_systems((egui_system, ui_system));

            // physics systems
            schedule.add_systems(
                (
                    cleanup_physics_system,
                    sync_entities_to_physics_system,
                    sync_joints_to_physics_system,
                    sync_transforms_to_physics_system,
                    character_controller_system,
                    apply_transient_forces_system,
                    apply_persistent_forces_system,
                    apply_queued_impulses_system,
                    physics_step_system,
                    sync_physics_to_entities_system,
                    physics_scene_query_system,
                    audio_system,
                )
                    .chain(),
            );

            init_callback(
                world,
                schedule,
                &runtime.asset_server.as_ref().unwrap().lock(),
            );
        },
        |dt, (world, schedule)| {
            // egui input interception
            world.resource_scope::<BevyInputManager, _>(|world, input_manager| {
                let mut input_manager = input_manager.0.write();
                input_manager.ui_wants_pointer = world
                    .get_resource::<UiRenderState>()
                    .map(|state| state.wants_pointer_input)
                    .unwrap_or(false);
                world.resource_scope::<BevyRuntimeConfig, _>(|ecs_core, runtime_config| {
                    let runtime_config = runtime_config.0;
                    ecs_core.resource_scope::<EguiResource, _>(|ecs, egui_resource| {
                        let passthrough = ecs
                            .get_resource::<EguiInputPassthrough>()
                            .copied()
                            .unwrap_or_default();
                        if runtime_config.egui {
                            input_manager.egui_wants_pointer =
                                egui_resource.ctx.wants_pointer_input() && !passthrough.pointer;
                            input_manager.egui_wants_key =
                                egui_resource.ctx.wants_keyboard_input() && !passthrough.keyboard;
                        } else if egui_resource.accepting_input {
                            input_manager.clear_egui_state();
                        }
                    });
                });
            });

            world.resource_mut::<DeltaTime>().0 = dt;

            let profiling = world
                .get_resource::<BevyRuntimeProfiling>()
                .map(|p| p.0.clone());
            let profiling_enabled = profiling
                .as_ref()
                .map(|p| p.enabled.load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(false);
            let schedule_start = if profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            schedule.run(world);
            ui_integration::publish_ui_render_state(world);
            if let (Some(start), Some(profiling)) = (schedule_start, profiling.as_ref()) {
                profiling.logic_schedule_us.store(
                    start.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

            let render_delta = {
                if let Some(mut packet) = world.get_resource_mut::<RenderPacket>() {
                    packet.0.take()
                } else {
                    None
                }
            };

            let egui_data = {
                if let Some(mut egui_res) = world.get_resource_mut::<EguiResource>() {
                    egui_res.render_data.take()
                } else {
                    None
                }
            };

            (render_delta, egui_data)
        },
        |new_size, (world, _schedule)| {
            for (mut camera, _) in world
                .query::<(&mut BevyCamera, &BevyActiveCamera)>()
                .iter_mut(world)
            {
                camera.0.aspect_ratio = new_size.width as f32 / new_size.height as f32;
            }
        },
        |path, (world, _schedule)| {
            if let Some(mut dragged_file_res) = world.get_resource_mut::<DraggedFile>() {
                dragged_file_res.0 = Some(path)
            }
        },
    );
    configure_runtime(&mut runtime);
    runtime.init();
}

pub fn helmer_becs_init(init_callback: fn(&mut World, &mut Schedule, &AssetServer)) {
    helmer_becs_init_impl(init_callback, |_runtime| {});
}

pub fn helmer_becs_init_with_runtime<F>(
    init_callback: fn(&mut World, &mut Schedule, &AssetServer),
    configure_runtime: F,
) where
    F: FnOnce(&mut Runtime<(World, Schedule)>),
{
    helmer_becs_init_impl(init_callback, configure_runtime);
}
