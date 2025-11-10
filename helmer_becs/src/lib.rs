use std::{path::PathBuf, sync::Arc};

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
    provided::components::{ActiveCamera, Camera, Light, MeshRenderer, Transform},
    runtime::{
        asset_server::AssetServer,
        config::RuntimeConfig,
        input_manager::InputManager,
        runtime::{PerformanceMetrics, Runtime},
    },
};
use parking_lot::{Mutex, RwLock};

use crate::provided::ui::inspector::InspectorSelectedEntityResource;
use crate::{
    egui_integration::{EguiResource, egui_system},
    physics::{
        physics_resource::PhysicsResource,
        systems::{
            cleanup_physics_system, physics_step_system, sync_entities_to_physics_system,
            sync_physics_to_entities_system, sync_transforms_to_physics_system,
        },
    },
    systems::{
        render_system::{RenderPacket, render_data_system},
        scene_system::{scene_spawning_system, update_scene_child_transforms},
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

// Resource Wrappers
#[derive(Resource)]
pub struct BevyAssetServer(pub Arc<Mutex<AssetServer>>);
#[derive(Resource)]
pub struct BevyInputManager(pub Arc<RwLock<InputManager>>);
#[derive(Resource)]
pub struct BevyPerformanceMetrics(pub Arc<PerformanceMetrics>);
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct BevyRuntimeConfig(pub RuntimeConfig);

// resources
#[derive(Resource, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct DeltaTime(pub f32);

#[derive(Resource, Clone, Debug, Default)]
pub struct DraggedFile(pub Option<PathBuf>);

pub mod egui_integration;
pub mod physics;
pub mod provided;
pub mod systems;

pub fn helmer_becs_init(init_callback: fn(&mut World, &mut Schedule, &AssetServer)) {
    let world = World::new();
    let mut schedule = Schedule::default();
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

    // Note: BevyTransform, BevyCamera, etc. cannot be registered because
    // Transform, Camera, etc. from helmer don't implement Reflect.
    // You can only register types in the inspector that implement Reflect.
    // To make your own components inspectable, derive Reflect on them:
    //
    // #[derive(Component, Reflect)]
    // #[reflect(Component)]
    // pub struct MyComponent { ... }
    //
    // Then register with: type_registry.register::<MyComponent>();

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
            world.insert_resource::<BevyAssetServer>(BevyAssetServer(
                runtime.asset_server.as_ref().unwrap().clone(),
            ));
            world.insert_resource::<BevyInputManager>(BevyInputManager(
                runtime.input_manager.clone(),
            ));
            world.insert_resource::<BevyPerformanceMetrics>(BevyPerformanceMetrics(
                runtime.metrics.clone(),
            ));
            world.insert_resource::<DeltaTime>(DeltaTime(1.0));
            world.insert_resource::<RenderPacket>(RenderPacket::default());
            world.insert_resource::<EguiResource>(EguiResource::default());
            world.insert_resource::<PhysicsResource>(PhysicsResource::default());
            world.insert_resource::<DraggedFile>(DraggedFile(None));
            world.insert_resource(InspectorSelectedEntityResource::default());

            // core systems
            schedule.add_systems(render_data_system);
            schedule.add_systems(egui_system);
            schedule.add_systems((scene_spawning_system, update_scene_child_transforms).chain());

            // physics systems
            schedule.add_systems(
                (
                    cleanup_physics_system,
                    sync_entities_to_physics_system,
                    sync_transforms_to_physics_system,
                    physics_step_system,
                    sync_physics_to_entities_system,
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
                world.resource_scope::<BevyRuntimeConfig, _>(|ecs_core, runtime_config| {
                    let runtime_config = runtime_config.0;
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
            });

            world.resource_mut::<DeltaTime>().0 = dt;

            schedule.run(world);

            let render_data = {
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

            (render_data, egui_data)
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
    runtime.init();
}
