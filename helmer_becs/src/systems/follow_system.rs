use bevy_ecs::prelude::{Entity, Query, Res};
use bevy_ecs::system::ParamSet;
use glam::{Mat3, Quat, Vec3};

use helmer::provided::components::{EntityFollower, LookAt, Transform};

use crate::{BevyEntityFollower, BevyLookAt, BevyTransform, DeltaTime};

fn smooth_factor(smooth_time: f32, dt: f32) -> f32 {
    if smooth_time <= 1.0e-4 || dt <= 0.0 {
        1.0
    } else {
        1.0 - (-dt / smooth_time).exp()
    }
}

fn smooth_vec3(current: Vec3, target: Vec3, smooth_time: f32, dt: f32) -> Vec3 {
    let t = smooth_factor(smooth_time, dt);
    current.lerp(target, t)
}

fn smooth_quat(current: Quat, target: Quat, smooth_time: f32, dt: f32) -> Quat {
    let t = smooth_factor(smooth_time, dt);
    current.slerp(target, t)
}

pub fn entity_follow_system(
    time: Res<DeltaTime>,
    mut params: ParamSet<(
        Query<(Entity, &BevyTransform, &BevyEntityFollower)>,
        Query<&mut BevyTransform>,
        Query<&BevyTransform>,
    )>,
) {
    let dt = time.0;
    let mut updates: Vec<(Entity, Vec3, Option<Quat>)> = Vec::new();

    let follower_data: Vec<(Entity, Transform, EntityFollower)> = {
        let followers = params.p0();
        followers
            .iter()
            .map(|(entity, transform, follower)| (entity, transform.0, follower.0))
            .collect()
    };

    {
        let targets = params.p2();
        for (entity, transform, follower) in follower_data {
            let Some(target_bits) = follower.target_entity else {
                continue;
            };
            let Ok(target_transform) = targets.get(Entity::from_bits(target_bits)) else {
                continue;
            };

            let target_transform = target_transform.0;
            let offset = if follower.offset_in_target_space {
                target_transform.rotation * follower.position_offset
            } else {
                follower.position_offset
            };
            let target_pos = target_transform.position + offset;
            let desired_pos = smooth_vec3(
                transform.position,
                target_pos,
                follower.position_smooth_time,
                dt,
            );
            let desired_rot = if follower.follow_rotation {
                Some(smooth_quat(
                    transform.rotation,
                    target_transform.rotation,
                    follower.rotation_smooth_time,
                    dt,
                ))
            } else {
                None
            };
            updates.push((entity, desired_pos, desired_rot));
        }
    }

    let mut transforms = params.p1();
    for (entity, desired_pos, desired_rot) in updates {
        if let Ok(mut transform) = transforms.get_mut(entity) {
            transform.0.position = desired_pos;
            if let Some(rot) = desired_rot {
                transform.0.rotation = rot;
            }
        }
    }
}

pub fn look_at_system(
    time: Res<DeltaTime>,
    mut params: ParamSet<(
        Query<(Entity, &BevyTransform, &BevyLookAt)>,
        Query<&mut BevyTransform>,
        Query<&BevyTransform>,
    )>,
) {
    let dt = time.0;
    let mut updates: Vec<(Entity, Quat)> = Vec::new();

    let looker_data: Vec<(Entity, Transform, LookAt)> = {
        let lookers = params.p0();
        lookers
            .iter()
            .map(|(entity, transform, look_at)| (entity, transform.0, look_at.0))
            .collect()
    };

    {
        let targets = params.p2();
        for (entity, transform, look_at) in looker_data {
            let Some(target_bits) = look_at.target_entity else {
                continue;
            };
            if target_bits == entity.to_bits() {
                continue;
            }
            let Ok(target_transform) = targets.get(Entity::from_bits(target_bits)) else {
                continue;
            };

            let target_transform = target_transform.0;
            let offset = if look_at.offset_in_target_space {
                target_transform.rotation * look_at.target_offset
            } else {
                look_at.target_offset
            };
            let target_pos = target_transform.position + offset;
            let to_target = target_pos - transform.position;
            if to_target.length_squared() <= 1.0e-6 {
                continue;
            }

            let forward = to_target.normalize();
            let mut up = look_at.up;
            if up.length_squared() <= 1.0e-6 {
                up = Vec3::Y;
            }
            let mut right = up.cross(forward);
            if right.length_squared() <= 1.0e-6 {
                right = Vec3::X;
            } else {
                right = right.normalize();
            }
            let up = forward.cross(right).normalize_or_zero();
            let basis = Mat3::from_cols(right, up, forward);
            let target_rot = Quat::from_mat3(&basis);

            let desired_rot = smooth_quat(
                transform.rotation,
                target_rot,
                look_at.rotation_smooth_time,
                dt,
            );
            updates.push((entity, desired_rot));
        }
    }

    let mut transforms = params.p1();
    for (entity, desired_rot) in updates {
        if let Ok(mut transform) = transforms.get_mut(entity) {
            transform.0.rotation = desired_rot;
        }
    }
}
