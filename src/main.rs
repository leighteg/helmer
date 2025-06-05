mod ecs;

use glam::Quat;
use helmer_rs::{
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

        let mut cube_transform = Transform {
            position: glam::Vec3::new(0.0, 0.0, 0.0),
            rotation: glam::Quat::from_array([30.0, 30.0, 0.0, 30.0]),
            scale: glam::Vec3::ONE,
        };

        // Create some demo entities
        let cube_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            cube_entity,
            cube_transform,
        );
        ecs_guard.add_component(
            cube_entity,
            MeshRenderer::new(0, 0),
        );

        let mut light_transform = Transform::default();
        light_transform.position = [2.0, 2.0, 2.0].into();

        let light_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity,
            light_transform,
        );
        ecs_guard.add_component(
            light_entity,
            Light::point(glam::vec3(1.0, 1.0, 1.0), 10.0),
        );
    });
    runtime.init();
}
