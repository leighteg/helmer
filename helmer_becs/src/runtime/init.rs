use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use bevy_ecs::{
    reflect::AppTypeRegistry,
    schedule::{IntoScheduleConfigs, Schedule},
    world::World,
};
use bevy_reflect::{TypeRegistry, TypeRegistryArc};
use helmer_asset::runtime::asset_server::AssetServer;
use helmer_render::runtime::RuntimeConfig;
use helmer_window::runtime::{
    input_manager::InputManager,
    runtime::{PerformanceMetrics, Runtime},
};
use helmer_window::service::{WindowCallbacks, WindowService};
use parking_lot::RwLock;

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
    profiling::SystemProfiler,
    provided::ui::inspector::InspectorSelectedEntityResource,
    resources::{
        AudioBackendResource, BecsAssetServer, BecsInputManager, BecsLodTuning,
        BecsPerformanceMetrics, BecsRenderControlSender, BecsRenderSender, BecsRenderWorkerTuning,
        BecsRendererStats, BecsRuntimeConfig, BecsRuntimeCursorState, BecsRuntimeProfiling,
        BecsRuntimeTuning, BecsRuntimeWindowControl, BecsSceneTuning, BecsStreamingTuning,
        BecsSystemProfiler, DebugGraphHistory, DeltaTime, DraggedFile, ProfilingHistory,
    },
    systems::{
        animation_system::{SkinningResource, skinning_system},
        audio_system::audio_system,
        follow_system::{entity_follow_system, look_at_system},
        render_system::{
            RenderGraphResource, RenderMainSceneToSwapchain, RenderObjectCount, RenderPacket,
            RenderResetRequest, RenderSyncRequest, RenderViewportRequests, render_data_system,
        },
        scene_system::{
            SceneSpawnedChildren, apply_scene_commands_system, scene_child_skinning_system,
            scene_spawning_system, update_scene_child_transforms,
        },
        spline_system::spline_follow_system,
    },
    ui_integration::{
        UiClipboard, UiPerfStats, UiRenderFrameOutput, UiRenderState, UiResource, ui_system,
    },
};

#[cfg(not(target_arch = "wasm32"))]
use super::logic::run_becs_logic_thread;
use super::{
    config::RuntimeBootstrapConfig,
    logic::{BecsLogicEvent, BecsLogicState},
};

fn helmer_becs_init_impl<F>(
    init_callback: fn(&mut World, &mut Schedule, &AssetServer),
    configure_runtime: F,
) where
    F: FnOnce(&mut RuntimeBootstrapConfig),
{
    helmer::runtime::init_runtime_tracing();

    let mut world = World::new();
    let mut schedule = Schedule::default();
    #[cfg(target_arch = "wasm32")]
    schedule.set_executor_kind(bevy_ecs::schedule::ExecutorKind::SingleThreaded);
    #[cfg(not(target_arch = "wasm32"))]
    schedule.set_executor_kind(bevy_ecs::schedule::ExecutorKind::MultiThreaded);

    let mut type_registry = TypeRegistry::default();

    type_registry.register::<f32>();
    type_registry.register::<f64>();
    type_registry.register::<i32>();
    type_registry.register::<u32>();
    type_registry.register::<i64>();
    type_registry.register::<u64>();
    type_registry.register::<bool>();
    type_registry.register::<String>();

    type_registry.register::<Vec<f32>>();
    type_registry.register::<Vec<String>>();

    type_registry.register::<DeltaTime>();

    let mut runtime_bootstrap = RuntimeBootstrapConfig::default();
    configure_runtime(&mut runtime_bootstrap);

    #[cfg(target_arch = "wasm32")]
    let wasm_harness_config = runtime_bootstrap.wasm_harness.clone().unwrap_or_default();

    #[cfg(target_arch = "wasm32")]
    {
        let _ = wasm_harness_config.ensure_canvas();
    }

    let input_manager = Arc::new(RwLock::new(InputManager::new()));
    let metrics = Arc::new(PerformanceMetrics::default());

    let mut runtime = Runtime::builder()
        .build()
        .expect("failed to build helmer runtime");
    runtime
        .context()
        .resources()
        .insert(helmer::runtime::RuntimePerformanceMetricsResource(
            metrics.clone(),
        ));
    runtime
        .register_extension(helmer_input::extension::InputExtension::with_manager(
            input_manager.clone(),
        ))
        .expect("failed to register helmer_input extension");
    runtime
        .register_extension(helmer_render::extension::RenderExtension::new())
        .expect("failed to register helmer_render extension");
    let asset_extension = runtime_bootstrap
        .asset_base_path
        .as_ref()
        .map(|path| helmer_asset::extension::AssetExtension::with_asset_base_path(path.clone()))
        .unwrap_or_else(helmer_asset::extension::AssetExtension::new);
    runtime
        .register_extension(asset_extension)
        .expect("failed to register helmer_asset extension");
    runtime
        .register_extension(helmer_audio::extension::AudioExtension::new())
        .expect("failed to register helmer_audio extension");
    let window_service = WindowService::with_input_manager(
        helmer_window::config::RuntimeConfig::default(),
        input_manager.clone(),
    )
    .with_metrics(metrics.clone());
    #[cfg(target_arch = "wasm32")]
    let window_service = window_service.with_wasm_harness(wasm_harness_config.clone());

    runtime
        .register_extension(helmer_window::extension::WindowExtension::new(
            window_service,
        ))
        .expect("failed to register helmer_window extension");
    runtime.start().expect("failed to start helmer runtime");

    let resources = runtime.context().resources();
    let asset_server = resources
        .get::<helmer_asset::extension::AssetServerResource>()
        .map(|resource| resource.0.clone())
        .expect("AssetExtension did not register AssetServerResource");
    let render_sender = resources
        .get::<helmer_render::extension::RenderMessageSender>()
        .map(|resource| resource.0.clone())
        .expect("RenderExtension did not register RenderMessageSender");
    let render_control_sender = resources
        .get::<helmer_render::extension::RenderControlMessageSender>()
        .map(|resource| resource.0.clone())
        .expect("RenderExtension did not register RenderControlMessageSender");
    let runtime_tuning = resources
        .get::<helmer_render::extension::RenderRuntimeTuningResource>()
        .map(|resource| resource.0.clone())
        .expect("RenderExtension did not register RuntimeTuning");
    let runtime_profiling = resources
        .get::<helmer_render::extension::RenderRuntimeProfilingResource>()
        .map(|resource| resource.0.clone())
        .expect("RenderExtension did not register RuntimeProfiling");
    let renderer_stats = resources
        .get::<helmer_render::extension::RenderStatsResource>()
        .map(|resource| resource.0.clone())
        .expect("RenderExtension did not register RendererStats");
    let runtime_cursor_state = resources
        .get::<helmer_window::extension::WindowCursorStateResource>()
        .map(|resource| resource.0.clone())
        .expect("WindowExtension did not register RuntimeCursorState");
    let runtime_window_control = resources
        .get::<helmer_window::extension::WindowControlResource>()
        .map(|resource| resource.0.clone())
        .expect("WindowExtension did not register RuntimeWindowControl");
    let audio_backend = resources
        .get::<helmer_audio::extension::RuntimeAudioBackendResource>()
        .map(|resource| resource.0.clone())
        .expect("AudioExtension did not register RuntimeAudioBackendResource");
    let window_service = resources
        .get::<helmer_window::extension::WindowMainThreadServiceResource>()
        .and_then(|resource| resource.0.lock().take())
        .expect("WindowExtension did not register WindowService");

    let registry_arc: TypeRegistryArc = TypeRegistryArc {
        internal: Arc::new(std::sync::RwLock::new(type_registry)),
    };
    world.insert_resource(AppTypeRegistry(registry_arc));
    world.insert_resource::<BecsRuntimeConfig>(BecsRuntimeConfig(RuntimeConfig::default()));
    world.insert_resource::<BecsRuntimeTuning>(BecsRuntimeTuning(runtime_tuning.clone()));
    world.insert_resource::<BecsRuntimeProfiling>(BecsRuntimeProfiling(runtime_profiling.clone()));
    world.insert_resource::<BecsSystemProfiler>(BecsSystemProfiler(Arc::new(
        SystemProfiler::default(),
    )));
    world.insert_resource::<BecsRuntimeCursorState>(BecsRuntimeCursorState(runtime_cursor_state));
    let runtime_window_control_for_events = runtime_window_control.clone();
    world.insert_resource::<BecsRuntimeWindowControl>(BecsRuntimeWindowControl(
        runtime_window_control,
    ));
    #[cfg(target_arch = "wasm32")]
    world.insert_non_send_resource::<BecsAssetServer>(BecsAssetServer(asset_server.clone()));
    #[cfg(not(target_arch = "wasm32"))]
    world.insert_resource::<BecsAssetServer>(BecsAssetServer(asset_server.clone()));
    world.insert_resource::<BecsInputManager>(BecsInputManager(input_manager.clone()));
    world.insert_resource::<BecsPerformanceMetrics>(BecsPerformanceMetrics(metrics.clone()));
    world.insert_resource::<BecsRendererStats>(BecsRendererStats(renderer_stats));
    world.insert_resource::<BecsRenderSender>(BecsRenderSender(render_sender.clone()));
    world.insert_resource::<BecsRenderControlSender>(BecsRenderControlSender(
        render_control_sender.clone(),
    ));
    world.insert_resource::<BecsStreamingTuning>(BecsStreamingTuning::default());
    world.insert_resource::<BecsLodTuning>(BecsLodTuning::default());
    world.insert_resource::<BecsRenderWorkerTuning>(BecsRenderWorkerTuning::default());
    world.insert_resource::<BecsSceneTuning>(BecsSceneTuning::default());
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
    world.insert_resource::<RenderMainSceneToSwapchain>(RenderMainSceneToSwapchain::default());
    world.insert_resource(RenderGraphResource::default());
    world.insert_resource(SceneSpawnedChildren::default());
    world.insert_resource::<EguiResource>(EguiResource::default());
    world.insert_resource::<EguiClipboard>(EguiClipboard::default());
    world.insert_resource::<EguiInputPassthrough>(EguiInputPassthrough::default());
    world.insert_resource::<UiResource>(UiResource::default());
    world.insert_resource::<UiRenderState>(UiRenderState::default());
    world.insert_resource::<UiRenderFrameOutput>(UiRenderFrameOutput::default());
    world.insert_resource::<crate::ui_integration::UiRuntimeState>(
        crate::ui_integration::UiRuntimeState::default(),
    );
    world.insert_resource::<UiClipboard>(UiClipboard::default());
    world.insert_resource::<PhysicsResource>(PhysicsResource::default());
    world.insert_resource::<AudioBackendResource>(AudioBackendResource(audio_backend));
    world.insert_resource::<DraggedFile>(DraggedFile(None));
    world.insert_resource(InspectorSelectedEntityResource::default());

    if let Some(system_profiler) = world.get_resource::<BecsSystemProfiler>() {
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

    init_callback(&mut world, &mut schedule, &asset_server.lock());

    #[cfg(not(target_arch = "wasm32"))]
    let threaded_logic = !runtime.context().is_single_threaded();
    #[cfg(not(target_arch = "wasm32"))]
    let render_sender_for_window_events = render_control_sender.clone();

    let logic_state = BecsLogicState::new(
        world,
        schedule,
        input_manager,
        metrics,
        runtime,
        runtime_tuning,
        runtime_profiling,
        runtime_window_control_for_events,
        render_sender,
        render_control_sender,
    );

    #[cfg(not(target_arch = "wasm32"))]
    let (mut logic_sender, mut logic_thread, mut inline_logic_state) = if threaded_logic {
        let (sender, receiver) = mpsc::channel::<BecsLogicEvent>();
        let join = std::thread::Builder::new()
            .name("helmer-logic".to_string())
            .spawn(move || run_becs_logic_thread(logic_state, receiver))
            .expect("failed to spawn helmer logic thread");
        (Some(sender), Some(join), None)
    } else {
        (None, None, Some(logic_state))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let logic_sender_for_events = logic_sender.clone();
    #[cfg(not(target_arch = "wasm32"))]
    let mut pending_render_bootstrap: Option<(
        Arc<winit::window::Window>,
        helmer_window::event::WindowState,
    )> = None;
    #[cfg(not(target_arch = "wasm32"))]
    let mut next_render_bootstrap_retry_at: Option<web_time::Instant> = None;

    #[cfg(target_arch = "wasm32")]
    let mut inline_logic_state = Some(logic_state);

    let mut callbacks = WindowCallbacks {
        on_event: Some(Box::new(move |event| {
            let logic_event = match &event.kind {
                helmer_window::event::WindowRuntimeEventKind::CloseRequested => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let _ = render_sender_for_window_events.send(
                            helmer_render::graphics::common::renderer::RenderMessage::Shutdown,
                        );
                    }
                    Some(BecsLogicEvent::CloseRequested(None))
                }
                helmer_window::event::WindowRuntimeEventKind::Started { window, state } => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        pending_render_bootstrap = Some((window.clone(), *state));
                        next_render_bootstrap_retry_at = None;
                    }
                    Some(BecsLogicEvent::Started { state: *state })
                }
                helmer_window::event::WindowRuntimeEventKind::Resized(state) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if let Some((_, pending_state)) = pending_render_bootstrap.as_mut() {
                            *pending_state = *state;
                        }
                        let _ = render_sender_for_window_events.send(
                            helmer_render::graphics::common::renderer::RenderMessage::Resize(
                                winit::dpi::PhysicalSize::new(state.width, state.height),
                            ),
                        );
                    }
                    Some(BecsLogicEvent::Resized(*state))
                }
                helmer_window::event::WindowRuntimeEventKind::DroppedFile(path) => {
                    Some(BecsLogicEvent::DroppedFile(path.clone()))
                }
                helmer_window::event::WindowRuntimeEventKind::Tick { dt } => {
                    Some(BecsLogicEvent::Tick(*dt))
                }
                _ => None,
            };

            #[cfg(not(target_arch = "wasm32"))]
            {
                let now = web_time::Instant::now();
                let retry_allowed = match next_render_bootstrap_retry_at {
                    Some(deadline) => now >= deadline,
                    None => true,
                };
                if retry_allowed && let Some((window, state)) = pending_render_bootstrap.as_ref() {
                    match helmer_render::extension::create_native_render_bootstrap_message(
                        window.clone(),
                        winit::dpi::PhysicalSize::new(state.width, state.height),
                    ) {
                        Ok(message) => match render_sender_for_window_events.send(message) {
                            Ok(()) => {
                                tracing::debug!(
                                    width = state.width,
                                    height = state.height,
                                    scale = state.scale_factor,
                                    "sent main-thread render bootstrap message"
                                );
                                pending_render_bootstrap = None;
                                next_render_bootstrap_retry_at = None;
                            }
                            Err(err) => {
                                tracing::warn!(
                                    "failed to send main-thread render bootstrap message: {err}"
                                );
                                pending_render_bootstrap = None;
                                next_render_bootstrap_retry_at = None;
                            }
                        },
                        Err(err) => {
                            let backoff_ms = 50u64;
                            next_render_bootstrap_retry_at =
                                Some(now + Duration::from_millis(backoff_ms));
                            tracing::warn!("failed to create main-thread render bootstrap: {err}");
                        }
                    }
                }
            }

            if let Some(logic_event) = logic_event {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(sender) = logic_sender_for_events.as_ref() {
                        if matches!(logic_event, BecsLogicEvent::Tick(_)) {
                            return;
                        }
                        if sender.send(logic_event).is_err() {
                            tracing::warn!("logic event channel disconnected");
                        }
                    } else if let Some(state) = inline_logic_state.as_mut() {
                        state.handle_event(logic_event);
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(state) = inline_logic_state.as_mut() {
                        state.handle_event(logic_event);
                    }
                }
            }
        })),
    };

    window_service
        .run(std::mem::take(&mut callbacks))
        .expect("failed to run window service");

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(sender) = logic_sender.take() {
            let _ = sender.send(BecsLogicEvent::CloseRequested(None));
        }
        if let Some(join) = logic_thread.take() {
            let _ = join.join();
        }
    }
}

pub fn helmer_becs_init(init_callback: fn(&mut World, &mut Schedule, &AssetServer)) {
    helmer_becs_init_impl(init_callback, |_runtime| {});
}

pub fn helmer_becs_init_with_runtime<F>(
    init_callback: fn(&mut World, &mut Schedule, &AssetServer),
    configure_runtime: F,
) where
    F: FnOnce(&mut RuntimeBootstrapConfig),
{
    helmer_becs_init_impl(init_callback, configure_runtime);
}
