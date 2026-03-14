use glam::Quat;
use helmer_becs::DeltaTime;
use helmer_becs::ecs::{
    component::Component,
    query::With,
    system::{Query, Res},
};

#[derive(Component)]
pub struct SpinnerObject {}

pub fn spinner_system(
    dt: Res<DeltaTime>,
    objects_query: Query<
        (&mut helmer_becs::Transform, &helmer_becs::MeshRenderer),
        With<SpinnerObject>,
    >,
) {
    let rotation_speed = 5.0 * dt.0;
    let delta_x_rotation = Quat::from_axis_angle(glam::Vec3::X, 0.0);
    let delta_y_rotation = Quat::from_axis_angle(glam::Vec3::Y, rotation_speed);
    let delta_z_rotation = Quat::from_axis_angle(glam::Vec3::Z, 0.0);

    for (mut transform, _mesh_renderer) in objects_query {
        transform.rotation *= delta_x_rotation * delta_y_rotation * delta_z_rotation;
        let _ = transform.rotation.normalize();
    }
}
