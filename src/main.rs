mod ecs;

use std::{any::TypeId, collections::HashSet};

use glam::Quat;
use helmer_rs::{
    ecs::system::System, provided::components::{Light, LightType, MeshAsset, MeshRenderer, Transform}, runtime::Runtime
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

        ecs_guard.system_scheduler.register_system(SpinnerSystem {}, 10, vec![], HashSet::from([TypeId::of::<Transform>()]), HashSet::from([TypeId::of::<Transform>()]));

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

        let mut light_transform = Transform {
            position: glam::Vec3::new(0.0, 0.0, 0.0),
            rotation: glam::Quat::from_rotation_y(-90.0),
            scale: glam::Vec3::ONE,
        };

        let light_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            light_entity,
            light_transform,
        );
        ecs_guard.add_component(
            light_entity,
            Light::spot(glam::vec3(1.0, 1.0, 1.0), 100.0, 120.0),
        );
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
        for transform in ecs.get_all_components_of_type_mut::<Transform>() {
            transform.rotation.x += 0.1;
            transform.rotation.y += 0.1;
            transform.rotation.z += 0.1;
        }
    }
}