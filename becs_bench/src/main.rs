use glam::{Quat, Vec3};
use helmer_becs::ecs::{
    component::Component,
    name::Name,
    query::With,
    system::{Query, Res},
};
use helmer_becs::{
    components::{ActiveCamera, Light, MeshAsset, MeshRenderer, Transform},
    helmer_becs_init_with_runtime,
    physics::components::{ColliderShape, DynamicRigidBody, FixedCollider},
    systems::scene_system::SceneRoot,
};
#[cfg(target_arch = "wasm32")]
use helmer_window::runtime::wasm_harness::WasmHarnessConfig;

use crate::systems::{
    config_toggle::config_toggle_system,
    drag::drag_system,
    freecam::{FreecamState, freecam_system},
    spawner::{MeshRendererStore, SpawnerSystemResource, spawner_system},
    spinner::{SpinnerObject, spinner_system},
};

pub mod systems;

#[cfg(target_arch = "wasm32")]
fn resolve_asset_base_path() -> String {
    "assets".to_string()
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_asset_base_path() -> String {
    let current_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let candidates = [
        current_dir.join("test_game/assets"),
        current_dir.join("../test_game/assets"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return candidate.to_string_lossy().to_string();
        }
    }
    "test_game/assets".to_string()
}

fn main() {
    helmer_becs_init_with_runtime(
        |world, schedule, asset_server| {
            let basic_material_handle = asset_server.load_material("materials/basic.ron");
            let metal_material_handle = asset_server.load_material("materials/shiny_metal.ron");
            let blue_light_material_handle = asset_server.load_material("materials/blue_light.ron");
            let red_light_material_handle = asset_server.load_material("materials/red_light.ron");

            let city_scene_handle = asset_server.load_scene("models/city.glb");
            let sponza_scene_handle = asset_server.load_scene("models/sponza.glb");
            let ford_raptor_scene_handle = asset_server.load_scene("models/ford_raptor.glb");

            let cube_mesh = MeshAsset::cube("cube".to_owned());
            let cube_handle = asset_server.add_mesh(cube_mesh.vertices.unwrap(), cube_mesh.indices);

            let uv_sphere_mesh = MeshAsset::uv_sphere("uv sphere".to_owned(), 32, 32);
            let uv_sphere_mesh_handle =
                asset_server.add_mesh(uv_sphere_mesh.vertices.unwrap(), uv_sphere_mesh.indices);

            let plane_mesh = MeshAsset::plane("plane".to_owned());
            let plane_mesh_handle =
                asset_server.add_mesh(plane_mesh.vertices.unwrap(), plane_mesh.indices);

            let camera_entity = world.spawn((
                helmer_becs::Transform::default(),
                helmer_becs::Camera::default(),
                helmer_becs::ActiveCamera {},
            ));

            let ground_entity = world.spawn((
                helmer_becs::Transform {
                    position: glam::Vec3::new(0.0, -5.0, 0.0),
                    rotation: glam::Quat::default(),
                    scale: glam::Vec3::from([500.0, 0.001, 500.0]),
                },
                MeshRenderer::new(plane_mesh_handle.id, basic_material_handle.id, false, false),
                ColliderShape::Cuboid,
                FixedCollider {},
                Name::new("ground plane"),
            ));

            let sun_rotation = Quat::from_euler(
                glam::EulerRot::YXZ,
                20.0f32.to_radians(),
                -50.0f32.to_radians(),
                20.0f32.to_radians(),
            );

            let sun_entity = world.spawn((
                helmer_becs::Transform {
                    position: glam::Vec3::new(0.0, 0.0, 0.0),
                    rotation: sun_rotation,
                    scale: glam::Vec3::ONE,
                },
                Light::directional(glam::vec3(1.0, 1.0, 1.0), 50.0),
            ));

            let spin_cube_entity = world.spawn((
                helmer_becs::Transform {
                    position: Vec3::from_array([0.0, -4.5, 0.0]),
                    rotation: Quat::default(),
                    scale: Vec3::from_array([5.0, 1.0, 0.5]),
                },
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
                SpinnerObject {},
                ColliderShape::Cuboid,
                FixedCollider {},
                Name::new("spin cube"),
            ));

            let l_wall_entity = world.spawn((
                helmer_becs::Transform {
                    position: Vec3::from_array([3.0, -4.0, 0.0]),
                    rotation: Quat::default(),
                    scale: Vec3::from_array([0.5, 5.0, 6.0]),
                },
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
                ColliderShape::Cuboid,
                FixedCollider {},
                Name::new("l wall"),
            ));

            let r_wall_entity = world.spawn((
                helmer_becs::Transform {
                    position: Vec3::from_array([-3.0, -4.0, 0.0]),
                    rotation: Quat::default(),
                    scale: Vec3::from_array([0.5, 5.0, 6.0]),
                },
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
                ColliderShape::Cuboid,
                FixedCollider {},
                Name::new("r wall"),
            ));

            let t_wall_entity = world.spawn((
                helmer_becs::Transform {
                    position: Vec3::from_array([0.0, -4.0, 3.0]),
                    rotation: Quat::default(),
                    scale: Vec3::from_array([6.0, 5.0, 0.5]),
                },
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
                ColliderShape::Cuboid,
                FixedCollider {},
                Name::new("t wall"),
            ));

            let b_wall_entity = world.spawn((
                helmer_becs::Transform {
                    position: Vec3::from_array([0.0, -4.0, -3.0]),
                    rotation: Quat::default(),
                    scale: Vec3::from_array([6.0, 5.0, 0.5]),
                },
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
                ColliderShape::Cuboid,
                FixedCollider {},
                Name::new("b wall"),
            ));

            let cube_entity = world.spawn((
                Transform::from_position([0.0, 0.0, 5.0]),
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
                ColliderShape::Cuboid,
                DynamicRigidBody { mass: 1.0 },
                Name::new("cube"),
            ));

            let city_entity = world.spawn((
                helmer_becs::Transform {
                    position: glam::Vec3::new(0.0, -5.0, 0.0),
                    rotation: glam::Quat::default(),
                    scale: glam::Vec3::from_array([3.0; 3]),
                },
                SceneRoot(city_scene_handle),
                Name::new("city scene root"),
            ));

            let sponza_entity = world.spawn((
                helmer_becs::Transform {
                    position: glam::Vec3::new(25.0, -4.0, 0.0),
                    rotation: glam::Quat::default(),
                    scale: glam::Vec3::ONE,
                },
                SceneRoot(sponza_scene_handle),
                Name::new("sponza scene root"),
            ));

            let raptor_entity = world.spawn((
                helmer_becs::Transform {
                    position: glam::Vec3::new(0.0, -5.0, 5.0),
                    rotation: glam::Quat::from_rotation_y(90.0),
                    scale: glam::Vec3::ONE,
                },
                SceneRoot(ford_raptor_scene_handle),
                Name::new("raptor scene root"),
            ));

            schedule.add_systems(spinner_system);

            schedule.add_systems(freecam_system);

            let mut mesh_renderer_store = MeshRendererStore::default();
            mesh_renderer_store.mesh_renderers.insert(
                "default cube".to_string(),
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
            );
            mesh_renderer_store.mesh_renderers.insert(
                "blue cube".to_string(),
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: blue_light_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
            );
            mesh_renderer_store.mesh_renderers.insert(
                "red cube".to_string(),
                MeshRenderer {
                    mesh_id: cube_handle.id,
                    material_id: red_light_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
            );
            mesh_renderer_store.mesh_renderers.insert(
                "default sphere".to_string(),
                MeshRenderer {
                    mesh_id: uv_sphere_mesh_handle.id,
                    material_id: basic_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
            );
            mesh_renderer_store.mesh_renderers.insert(
                "blue sphere".to_string(),
                MeshRenderer {
                    mesh_id: uv_sphere_mesh_handle.id,
                    material_id: blue_light_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
            );
            mesh_renderer_store.mesh_renderers.insert(
                "red sphere".to_string(),
                MeshRenderer {
                    mesh_id: uv_sphere_mesh_handle.id,
                    material_id: red_light_material_handle.id,
                    casts_shadow: true,
                    visible: true,
                },
            );
            world.insert_resource(mesh_renderer_store);

            world.insert_resource(SpawnerSystemResource::default());
            schedule.add_systems(spawner_system);

            schedule.add_systems(config_toggle_system);

            schedule.add_systems(drag_system);
        },
        |runtime| {
            runtime.set_asset_base_path(resolve_asset_base_path());
            #[cfg(target_arch = "wasm32")]
            {
                let mut config = WasmHarnessConfig::default();
                config.mount_id = Some("helmer-root".to_string());
                config.canvas_id = Some("helmer-canvas".to_string());
                if let Err(err) = config.apply_to_runtime(runtime) {
                    eprintln!("wasm harness setup failed: {}", err);
                }
            }
        },
    );
}
