use std::env;

use bevy_ecs::{
    component::Component,
    query::With,
    system::{Query, Res},
};
use glam::Quat;
use helmer::provided::components::{ActiveCamera, Light, MeshAsset, MeshRenderer, Transform};
use helmer_becs::{
    BevyActiveCamera, BevyCamera, BevyLight, BevyMeshRenderer, BevyTransform, DeltaTime,
    helmer_becs_init, systems::scene_system::SceneRoot,
};

use crate::systems::freecam::{FreecamState, freecam_system};

pub mod systems;

fn main() {
    let current_path = env::current_dir().expect("Failed to find executable path");
    if current_path.ends_with("helmer-rs") {
        env::set_current_dir(current_path.join("becs_bench"))
            .expect("Failed to change working directory");
    }

    helmer_becs_init(|world, schedule, asset_server| {
        let basic_material_handle =
            asset_server.load_material("../test_game/assets/materials/basic.ron");
        let metal_material_handle =
            asset_server.load_material("../test_game/assets/materials/shiny_metal.ron");
        let blue_light_material_handle =
            asset_server.load_material("../test_game/assets/materials/blue_light.ron");
        let red_light_material_handle =
            asset_server.load_material("../test_game/assets/materials/red_light.ron");

        let city_scene_handle = asset_server.load_scene("../test_game/assets/models/city.glb");
        let sponza_scene_handle = asset_server.load_scene("../test_game/assets/models/sponza.glb");
        let ford_raptor_scene_handle =
            asset_server.load_scene("../test_game/assets/models/ford_raptor.glb");

        let cube_mesh = MeshAsset::cube("cube".to_owned());
        let cube_handle = asset_server.add_mesh(cube_mesh.vertices.unwrap(), cube_mesh.indices);

        let plane_mesh = MeshAsset::plane("plane".to_owned());
        let plane_mesh_handle =
            asset_server.add_mesh(plane_mesh.vertices.unwrap(), plane_mesh.indices);

        let camera_entity = world.spawn((
            BevyTransform::default(),
            BevyCamera::default(),
            BevyActiveCamera { 0: ActiveCamera {} },
        ));

        let ground_entity = world.spawn((
            BevyTransform {
                0: Transform {
                    position: glam::Vec3::new(0.0, -5.0, 0.0),
                    rotation: glam::Quat::default(),
                    scale: glam::Vec3::from([500.0, 0.001, 500.0]),
                },
            },
            BevyMeshRenderer {
                0: MeshRenderer::new(plane_mesh_handle.id, metal_material_handle.id, false, false),
            },
        ));

        let sun_rotation = Quat::from_euler(
            glam::EulerRot::YXZ,
            20.0f32.to_radians(),  // Y rotation - very slight side angle
            -50.0f32.to_radians(), // X rotation - steeper downward angle
            20.0f32.to_radians(),  // Z rotation - no roll
        );

        let sun_entity = world.spawn((
            BevyTransform {
                0: Transform {
                    position: glam::Vec3::new(0.0, 0.0, 0.0),
                    rotation: sun_rotation,
                    scale: glam::Vec3::ONE,
                },
            },
            BevyLight {
                0: Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0),
            },
        ));

        let cube_entity = world.spawn((
            BevyTransform {
                0: Transform::from_position([0.0, 0.0, 5.0]),
            },
            BevyMeshRenderer {
                0: MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
            },
            SpinnerObject {},
        ));

        let city_entity = world.spawn((
            BevyTransform {
                0: Transform {
                    position: glam::Vec3::new(0.0, -5.0, 0.0),
                    rotation: glam::Quat::default(),
                    scale: glam::Vec3::from_array([3.0; 3]),
                },
            },
            SceneRoot(city_scene_handle),
        ));

        let sponza_entity = world.spawn((
            BevyTransform {
                0: Transform {
                    position: glam::Vec3::new(25.0, -4.0, 0.0),
                    rotation: glam::Quat::default(),
                    scale: glam::Vec3::ONE,
                },
            },
            SceneRoot(sponza_scene_handle),
        ));

        let raptor_entity = world.spawn((
            BevyTransform {
                0: Transform {
                    position: glam::Vec3::new(0.0, -5.0, 0.0),
                    rotation: glam::Quat::from_rotation_y(90.0),
                    scale: glam::Vec3::ONE,
                },
            },
            SceneRoot(ford_raptor_scene_handle),
        ));

        schedule.add_systems(spinner_system);

        schedule.add_systems(freecam_system);
    });
}

#[derive(Component)]
struct SpinnerObject {}

fn spinner_system(
    dt: Res<DeltaTime>,
    objects_query: Query<(&mut BevyTransform, &BevyMeshRenderer), With<SpinnerObject>>,
) {
    let rotation_speed = 0.50 * dt.0;
    let delta_x_rotation = Quat::from_axis_angle(glam::Vec3::X, rotation_speed);
    let delta_y_rotation = Quat::from_axis_angle(glam::Vec3::Y, rotation_speed);
    let delta_z_rotation = Quat::from_axis_angle(glam::Vec3::Z, rotation_speed * 2.0);

    for (mut transform, mesh_renderer) in objects_query {
        transform.0.rotation *= delta_x_rotation * delta_y_rotation * delta_z_rotation;
        let _ = transform.0.rotation.normalize();
    }
}
