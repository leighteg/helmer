use std::{any::TypeId, collections::HashSet, env};

use helmer_editor::systems::{interaction::freecam::FreecamSystem, ui::inspector::InspectorSystem};
use helmer_engine::{provided::components::{ActiveCamera, Camera, Transform}, runtime::{egui_integration::EguiResource, runtime::Runtime}};

fn main() {
    let current_path = env::current_dir().expect("Failed to find executable path");
    if current_path.ends_with("helmer-rs") {
        env::set_current_dir(current_path.join("helmer-editor"))
            .expect("Failed to change working directory");
    }

    let mut runtime = Runtime::new(|app| {
        let mut ecs_guard = app.ecs.write();

        ecs_guard.system_scheduler.register_system(
            FreecamSystem::new(1.0, 0.5),
            30,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );

        ecs_guard.system_scheduler.register_system(
            InspectorSystem {},
            0,
            vec![],
            HashSet::from([]),
            HashSet::from([TypeId::of::<EguiResource>()]),
        );

        let camera_entity = ecs_guard.create_entity();
        ecs_guard.add_component(
            camera_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, -3.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
        );
        ecs_guard.add_component(
            camera_entity,
            Camera {
                far_plane: 300.0,
                ..Default::default()
            },
        );
        ecs_guard.add_component(camera_entity, ActiveCamera {});
    });
    runtime.init();
}