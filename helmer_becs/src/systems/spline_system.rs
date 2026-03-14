use bevy_ecs::prelude::{Entity, Query, Res, Without};
use glam::{Mat3, Vec3};

use crate::{BecsSystemProfiler, DeltaTime, Spline, SplineFollower, Transform};

pub fn spline_follow_system(
    time: Res<DeltaTime>,
    mut followers: Query<(Entity, &mut Transform, &mut SplineFollower)>,
    splines: Query<(&Spline, Option<&Transform>), Without<SplineFollower>>,
    system_profiler: Option<Res<BecsSystemProfiler>>,
) {
    let _system_scope = system_profiler.as_ref().and_then(|profiler| {
        profiler
            .0
            .begin_scope("helmer_becs::systems::spline_follow_system")
    });

    let dt = time.0;
    for (entity, mut transform, mut follower) in followers.iter_mut() {
        let target_bits = follower.spline_entity.unwrap_or(entity.to_bits());
        let Some(target_entity) = Entity::try_from_bits(target_bits) else {
            continue;
        };
        let Ok((spline, spline_transform)) = splines.get(target_entity) else {
            continue;
        };
        let spline = spline;
        if !spline.is_valid() {
            continue;
        }

        let sample_count = follower.length_samples.max(2) as usize;
        let length = spline.approx_length(sample_count);
        if follower.speed != 0.0 && length > 0.0 {
            follower.t += follower.speed * dt / length;
        }

        let looped = follower.looped || spline.closed;
        if looped {
            follower.t = follower.t.rem_euclid(1.0);
        } else {
            follower.t = follower.t.clamp(0.0, 1.0);
        }

        let local_pos = spline.sample(follower.t) + follower.offset;
        let local_tangent = spline.sample_tangent(follower.t);
        let spline_transform = spline_transform.copied().unwrap_or_default();
        let spline_matrix = spline_transform.to_matrix();

        let world_pos = spline_matrix.transform_point3(local_pos);
        let mut world_tangent = spline_matrix.transform_vector3(local_tangent);
        if world_tangent.length_squared() > 1.0e-6 {
            world_tangent = world_tangent.normalize();
        }

        transform.position = world_pos;

        if follower.follow_rotation && world_tangent.length_squared() > 1.0e-6 {
            let mut up = follower.up;
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
            transform.rotation = glam::Quat::from_mat3(&basis);
        }
    }
}
