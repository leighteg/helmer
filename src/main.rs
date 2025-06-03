mod ecs;

use helmer_rs::{
    provided::components::{LightComponent, LightType, MeshComponent, Transform},
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
            Transform {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
        );
        ecs_guard.add_component(
            cube_entity,
            MeshComponent {
                mesh_id: 1,
                material_id: 1,
            },
        );

        let light_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity,
            Transform {
                position: [5.0, 5.0, 5.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
        );
        ecs_guard.add_component(
            light_entity,
            LightComponent {
                color: [1.0, 1.0, 1.0],
                intensity: 1.0,
                light_type: LightType::Point,
            },
        );
    });
    runtime.init();
}
