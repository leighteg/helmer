mod ecs;

use std::{any::TypeId, collections::HashSet};

use glam::Quat;
use helmer_rs::{
    ecs::system::System,
    provided::components::{Light, LightType, MeshAsset, MeshRenderer, Transform},
    runtime::Runtime,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .unwrap();

    tracing::info!("2025 leighton");

    let mut runtime = Runtime::new(|app| {
        let mut ecs_guard = app.ecs.write().unwrap();

        ecs_guard.system_scheduler.register_system(
            SpinnerSystem {},
            10,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );

        // Create some demo entities
        let cube_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            cube_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, 0.0),
                rotation: glam::Quat::from_array([30.0, 30.0, 0.0, 30.0]),
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(cube_entity, MeshRenderer::new(0, 0));

        let light_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity,
            Transform {
                position: glam::Vec3::new(0.0, 1.5, 0.0),
                rotation: glam::Quat::from_array([0.0, 0.0, 0.0, 1.0]),
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(
            light_entity,
            Light::point(glam::vec3(1.0, 1.0, 1.0), 100.0),
        );
        ecs_guard.add_component(light_entity, MeshRenderer::new(0, 0));
    });
    runtime.init();
}

struct SpinnerSystem {}
impl System for SpinnerSystem {
    fn name(&self) -> &str {
        "SpinnerSystem"
    }

    fn run(&self, ecs: &mut helmer_rs::ecs::ecs_core::ECSCore) {
        tracing::info!("SpinnerSystem running...");

        let rotation_speed = 0.01;
        let delta_x_rotation = Quat::from_axis_angle(glam::Vec3::X, rotation_speed); // rotation_speed * dt
        let delta_y_rotation = Quat::from_axis_angle(glam::Vec3::Y, rotation_speed);
        let delta_z_rotation = Quat::from_axis_angle(glam::Vec3::Z, rotation_speed);

        ecs.component_pool.query_exact_mut_for_each::<(Transform, MeshRenderer), _>(|(transform, mesh_renderer)| {
            transform.rotation *= delta_x_rotation * delta_y_rotation * delta_z_rotation;
        });
    }
}
