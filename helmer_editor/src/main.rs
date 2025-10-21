use std::{any::TypeId, collections::HashSet, env};

use glam::Quat;
use helmer_ecs::{egui_integration::EguiResource, helmer_ecs_init, physics::components::{ColliderShape, DynamicRigidBody, FixedCollider}};
use helmer_editor::systems::{
    core::state::{EditorStateResource, EditorStateSystem, WorldState},
    interaction::freecam::FreecamSystem,
    ui::inspector::InspectorSystem,
};
use helmer::provided::components::{ActiveCamera, Camera, Light, MeshAsset, MeshRenderer, Transform};

fn main() {
    let current_path = env::current_dir().expect("Failed to find executable path");
    if current_path.ends_with("helmer-rs") {
        env::set_current_dir(current_path.join("helmer_editor"))
            .expect("Failed to change working directory");
    }

    helmer_ecs_init(|ecs, scheduler, asset_server| {
        ecs.add_resource(EditorStateResource {
            world_state: WorldState::Edit,
        });

        scheduler.register_system(
            EditorStateSystem {
                last_world_state: WorldState::Edit,
            },
            0,
            vec![],
            HashSet::from([TypeId::of::<EditorStateResource>()]),
            HashSet::from([TypeId::of::<EditorStateResource>()]),
        );

        scheduler.register_system(
            FreecamSystem::new(1.0, 0.5),
            30,
            vec![],
            HashSet::from([TypeId::of::<Transform>()]),
            HashSet::from([TypeId::of::<Transform>()]),
        );

        scheduler.register_system(
            InspectorSystem {},
            0,
            vec![],
            HashSet::from([]),
            HashSet::from([TypeId::of::<EguiResource>()]),
        );

        let basic_material_handle =
            asset_server.load_material("../test_game/assets/materials/basic.ron");
        let blue_light_material_handle =
            asset_server.load_material("../test_game/assets/materials/blue_light.ron");
        let red_light_material_handle =
            asset_server.load_material("../test_game/assets/materials/red_light.ron");

        let cube_mesh = MeshAsset::cube("cube".to_owned());
        let cube_handle = asset_server.add_mesh(cube_mesh.vertices.unwrap(), cube_mesh.indices);

        let plane_mesh = MeshAsset::plane("plane".to_owned());
        let plane_handle = asset_server.add_mesh(plane_mesh.vertices.unwrap(), plane_mesh.indices);

        let camera_entity = ecs.create_entity();
        ecs.add_component(
            camera_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, -3.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
        );
        ecs.add_component(
            camera_entity,
            Camera {
                far_plane: 300.0,
                ..Default::default()
            },
        );
        ecs.add_component(camera_entity, ActiveCamera {});

        let sun_rotation = Quat::from_euler(
            glam::EulerRot::YXZ,
            20.0f32.to_radians(),
            -50.0f32.to_radians(),
            20.0f32.to_radians(),
        );

        let sun_entity: usize = ecs.create_entity();
        ecs.add_component(
            sun_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, 0.0),
                rotation: sun_rotation,
                scale: glam::Vec3::ONE,
            },
        );
        ecs.add_component(
            sun_entity,
            Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0),
        );

        let ground_entity = ecs.create_entity();
        ecs.add_component(
            ground_entity,
            Transform {
                position: glam::Vec3::new(0.0, -5.0, 0.0),
                rotation: glam::Quat::default(),
                scale: glam::Vec3::from([50.0, 0.001, 50.0]),
            },
        );
        ecs.add_component(
            ground_entity,
            MeshRenderer::new(plane_handle.id, basic_material_handle.id, false, true),
        );
        ecs.add_component(ground_entity, ColliderShape::Cuboid);
        ecs.add_component(ground_entity, FixedCollider {});

        let cube_entity = ecs.create_entity();
        ecs.add_component(
            cube_entity,
            Transform {
                position: glam::Vec3::new(0.0, 0.0, 0.0),
                rotation: glam::Quat::default(),
                scale: glam::Vec3::ONE,
            },
        );
        ecs.add_component(
            cube_entity,
            MeshRenderer::new(cube_handle.id, blue_light_material_handle.id, true, true),
        );
        ecs.add_component(cube_entity, ColliderShape::Cuboid);
        ecs.add_component(cube_entity, DynamicRigidBody { mass: 1.0 });

        let cube2_entity = ecs.create_entity();
        ecs.add_component(
            cube2_entity,
            Transform {
                position: glam::Vec3::new(3.0, 0.0, 0.0),
                rotation: glam::Quat::default(),
                scale: glam::Vec3::from_array([2.0; 3]),
            },
        );
        ecs.add_component(
            cube2_entity,
            MeshRenderer::new(cube_handle.id, red_light_material_handle.id, true, true),
        );
        ecs.add_component(cube2_entity, ColliderShape::Cuboid);
        ecs.add_component(cube2_entity, DynamicRigidBody { mass: 5.0 });
    });
}
