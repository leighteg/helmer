use crate::{
    ecs::component::Component,
    egui_integration::{EguiResource, EguiSystem},
    physics::{
        physics_resource::PhysicsResource,
        systems::{
            CleanupPhysicsSystem, PhysicsStepSystem, SyncEntitiesToPhysicsSystem,
            SyncPhysicsToEntitiesSystem,
        },
    },
    systems::{
        renderer_system::{RenderDataSystem, RenderPacket},
        scene_system::SceneSpawningSystem,
    },
};

use std::{any::TypeId, collections::HashSet, sync::Arc};

use helmer::{
    provided::components::{ActiveCamera, Camera, Light, MeshRenderer, Transform},
    runtime::{
        asset_server::AssetServer, config::RuntimeConfig, input_manager::InputManager,
        runtime::Runtime,
    },
};
use parking_lot::RwLock;

use crate::ecs::{ecs_core::ECSCore, system_scheduler::SystemScheduler};

impl Component for Transform {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for MeshRenderer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for Camera {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for ActiveCamera {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Component for Light {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub mod ecs;
pub mod egui_integration;
pub mod physics;
pub mod provided;
pub mod systems;

pub fn helmer_ecs_init(init_callback: fn(&mut ECSCore, &mut SystemScheduler, &AssetServer)) {
    let ecs_core = ECSCore::new();
    let scheduler = SystemScheduler::new();

    let mut runtime: Runtime<(ECSCore, SystemScheduler)> = Runtime::new(
        (ecs_core, scheduler),
        move |runtime, (ecs_core, scheduler)| {
            ecs_core.add_resource(runtime.config.as_ref().clone());
            ecs_core.add_resource(runtime.asset_server.as_ref().unwrap().clone());
            ecs_core.add_resource(runtime.input_manager.clone());
            ecs_core.add_resource(RenderPacket::default());
            ecs_core.add_resource(PhysicsResource::new());
            ecs_core.add_resource(EguiResource::default());
            ecs_core.add_resource(runtime.metrics.clone());

            scheduler.register_system(
                SceneSpawningSystem {},
                25,
                vec![],
                HashSet::from([TypeId::of::<Transform>()]),
                HashSet::from([TypeId::of::<Transform>()]),
            );

            // Priority 20: Pre-Physics Sync. Creates physics bodies from ECS components.
            // Must run *after* game logic and *before* the physics step.
            scheduler.register_system(
                SyncEntitiesToPhysicsSystem {},
                20,
                vec![],
                HashSet::from([TypeId::of::<Transform>()]),
                HashSet::from([TypeId::of::<Transform>()]),
            );

            // Priority 10: The Physics Step. The core simulation tick.
            // Must run *after* entities are synced to physics.
            scheduler.register_system(
                PhysicsStepSystem {},
                10,
                vec![],
                HashSet::from([TypeId::of::<Transform>()]),
                HashSet::from([TypeId::of::<Transform>()]),
            );

            // Priority 5: Post-Physics Sync. Applies simulation results back to ECS transforms.
            // Must run *after* the physics step and *before* rendering.
            scheduler.register_system(
                SyncPhysicsToEntitiesSystem {},
                5,
                vec![],
                HashSet::from([TypeId::of::<Transform>()]),
                HashSet::from([TypeId::of::<Transform>()]),
            );

            scheduler.register_system(
                CleanupPhysicsSystem::default(),
                4,
                vec![],
                HashSet::from([TypeId::of::<Transform>()]),
                HashSet::from([TypeId::of::<Transform>()]),
            );

            scheduler.register_system(
                EguiSystem {},
                1,
                vec![],
                HashSet::new(),
                HashSet::from([TypeId::of::<EguiResource>()]),
            );

            // Priority 0: Rendering. Runs last to ensure it uses the final state of all transforms.
            scheduler.register_system(
                RenderDataSystem::new(),
                0,
                vec![],
                HashSet::from([TypeId::of::<Transform>()]),
                HashSet::from([TypeId::of::<Transform>()]),
            );

            init_callback(
                ecs_core,
                scheduler,
                &runtime.asset_server.as_ref().unwrap().lock(),
            );
        },
        |dt, (ecs_core, scheduler)| {
            ecs_core.resource_scope::<Arc<RwLock<InputManager>>, _>(|ecs_core, input_manager| {
                let mut input_manager = input_manager.write();
                ecs_core.resource_scope::<RuntimeConfig, _>(|ecs_core, runtime_config| {
                    ecs_core.resource_scope::<EguiResource, _>(|ecs, egui_resource| {
                        if runtime_config.egui {
                            if input_manager.active_mouse_buttons.len() == 0 {
                                input_manager.egui_wants_pointer =
                                    egui_resource.ctx.wants_pointer_input();
                            }
                            input_manager.egui_wants_key = egui_resource.ctx.wants_keyboard_input();
                        } else if egui_resource.accepting_input {
                            input_manager.clear_egui_state();
                        }
                    });
                });

                scheduler.run_all(dt, ecs_core, &input_manager);
            });

            let render_delta = {
                if let Some(packet) = ecs_core.get_resource_mut::<RenderPacket>() {
                    packet.0.take()
                } else {
                    None
                }
            };

            let egui_data = {
                if let Some(egui_res) = ecs_core.get_resource_mut::<EguiResource>() {
                    egui_res.render_data.take()
                } else {
                    None
                }
            };

            (render_delta, egui_data)
        },
        |new_size, (ecs_core, _)| {
            ecs_core
                .component_pool
                .query_mut_for_each::<(Camera, ActiveCamera), _>(|_, (camera, _)| {
                    camera.aspect_ratio = new_size.width as f32 / new_size.height as f32;
                });
        },
        |_path, (_world, _schedule)| {},
    );

    runtime.init();
}
