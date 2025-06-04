mod ecs;

use helmer_rs::{
    provided::components::{Light, LightType, MeshRenderer, Transform},
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

        // Create some demo entities
        let cube_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            cube_entity,
            Transform::default(),
        );
        ecs_guard.add_component(
            cube_entity,
            Transform::default(),
        );
        ecs_guard.add_component(
            cube_entity,
            MeshRenderer::new(1, 1),
        );

        let light_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity,
            Transform::default(),
        );
        ecs_guard.add_component(
            light_entity,
            Light::directional(Default::default(), Default::default()),
        );
    });
    runtime.init();
}
