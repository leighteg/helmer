use std::env;

use bevy_ecs::system::{Query, Res};
use glam::Quat;
use helmer::provided::components::{ActiveCamera, MeshAsset, MeshRenderer, Transform};
use helmer_becs::{
    BevyActiveCamera, BevyCamera, BevyMeshRenderer, BevyTransform, DeltaTime, helmer_becs_init,
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
        let blue_light_material_handle =
            asset_server.load_material("../test_game/assets/materials/blue_light.ron");
        let red_light_material_handle =
            asset_server.load_material("../test_game/assets/materials/red_light.ron");

        let cube_mesh = MeshAsset::cube("cube".to_owned());
        let cube_handle = asset_server.add_mesh(cube_mesh.vertices.unwrap(), cube_mesh.indices);

        let plane_mesh = MeshAsset::plane("plane".to_owned());
        let plane_handle = asset_server.add_mesh(plane_mesh.vertices.unwrap(), plane_mesh.indices);

        let camera_entity = world.spawn((
            BevyTransform::default(),
            BevyCamera::default(),
            BevyActiveCamera { 0: ActiveCamera {} },
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
        ));

        schedule.add_systems(spinner_system);

        schedule.add_systems(freecam_system);
    });
}

fn spinner_system(
    dt: Res<DeltaTime>,
    objects_query: Query<(&mut BevyTransform, &BevyMeshRenderer)>,
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
