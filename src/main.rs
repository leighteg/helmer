mod ecs;

use helmer_rs::{
    provided::components::{MeshComponent, Transform},
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
        let entity1 = ecs_guard.create_entity();
        ecs_guard.add_component(
            entity1,
            Transform {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
        );
        ecs_guard.add_component(
            entity1,
            MeshComponent {
                mesh_id: 0,
                material_id: 0,
            },
        );
    });
    runtime.init();
}
