use bevy_ecs::{
    component::Component,
    query::With,
    system::{Query, Res},
};
use glam::Quat;
use helmer_becs::{BevyMeshRenderer, BevyTransform, DeltaTime};

#[derive(Component)]
pub struct SpinnerObject {}

pub fn spinner_system(
    dt: Res<DeltaTime>,
    objects_query: Query<(&mut BevyTransform, &BevyMeshRenderer), With<SpinnerObject>>,
) {
    let rotation_speed = 0.50 * dt.0;
    let delta_x_rotation = Quat::from_axis_angle(glam::Vec3::X, rotation_speed);
    let delta_y_rotation = Quat::from_axis_angle(glam::Vec3::Y, rotation_speed);
    let delta_z_rotation = Quat::from_axis_angle(glam::Vec3::Z, rotation_speed * 2.0);

    for (mut transform, _mesh_renderer) in objects_query {
        transform.0.rotation *= delta_x_rotation * delta_y_rotation * delta_z_rotation;
        let _ = transform.0.rotation.normalize();
    }
}
