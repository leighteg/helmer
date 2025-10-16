use std::env;

use helmer::provided::components::{ActiveCamera, MeshAsset};
use helmer_becs::{helmer_becs_init, BevyActiveCamera, BevyCamera, BevyTransform};

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

        let camera_entity = world.spawn((BevyTransform::default(), BevyCamera::default(), BevyActiveCamera { 0: ActiveCamera {  } }));
    });
}