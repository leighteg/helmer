use bevy_ecs::prelude::{Entity, Query, Res, Without};
use glam::{Mat3, Vec3};

use crate::{BevySpline, BevySplineFollower, BevyTransform, DeltaTime};

pub fn spline_follow_system(
    time: Res<DeltaTime>,
    mut followers: Query<(Entity, &mut BevyTransform, &mut BevySplineFollower)>,
    splines: Query<(&BevySpline, Option<&BevyTransform>), Without<BevySplineFollower>>,
) {
    let dt = time.0;
    for (entity, mut transform, mut follower) in followers.iter_mut() {
        let target_bits = follower.0.spline_entity.unwrap_or(entity.to_bits());
        let Ok((spline, spline_transform)) = splines.get(Entity::from_bits(target_bits)) else {
            continue;
        };
        let spline = &spline.0;
        if !spline.is_valid() {
            continue;
        }

        let sample_count = follower.0.length_samples.max(2) as usize;
        let length = spline.approx_length(sample_count);
        if follower.0.speed != 0.0 && length > 0.0 {
            follower.0.t += follower.0.speed * dt / length;
        }

        let looped = follower.0.looped || spline.closed;
        if looped {
            follower.0.t = follower.0.t.rem_euclid(1.0);
        } else {
            follower.0.t = follower.0.t.clamp(0.0, 1.0);
        }

        let local_pos = spline.sample(follower.0.t) + follower.0.offset;
        let local_tangent = spline.sample_tangent(follower.0.t);
        let spline_transform = spline_transform.map(|t| t.0).unwrap_or_default();
        let spline_matrix = spline_transform.to_matrix();

        let world_pos = spline_matrix.transform_point3(local_pos);
        let mut world_tangent = spline_matrix.transform_vector3(local_tangent);
        if world_tangent.length_squared() > 1.0e-6 {
            world_tangent = world_tangent.normalize();
        }

        transform.0.position = world_pos;

        if follower.0.follow_rotation && world_tangent.length_squared() > 1.0e-6 {
            let mut up = follower.0.up;
            if up.length_squared() < 1.0e-6 {
                up = Vec3::Y;
            }
            let mut right = up.cross(world_tangent);
            if right.length_squared() < 1.0e-6 {
                right = Vec3::X;
            } else {
                right = right.normalize();
            }
            let up = world_tangent.cross(right).normalize_or_zero();
            let basis = Mat3::from_cols(right, up, world_tangent);
            transform.0.rotation = glam::Quat::from_mat3(&basis);
        }
    }
}
