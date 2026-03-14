use crate::components::{Spline, SplineFollower, Transform};
use crate::ecs::{ecs_core::ECSCore, system::System};
use glam::{Mat3, Vec3};
use hashbrown::HashMap;
use helmer_window::runtime::input_manager::InputManager;

pub struct SplineFollowSystem;

impl System for SplineFollowSystem {
    fn name(&self) -> &str {
        "SplineFollowSystem"
    }

    fn run(&mut self, dt: f32, ecs: &mut ECSCore, _input_manager: &InputManager) {
        let mut spline_map: HashMap<usize, (Spline, Transform)> = HashMap::new();
        ecs.component_pool.query_for_each::<(Spline, Transform), _>(
            |entity, (spline, transform)| {
                spline_map.insert(entity, (spline.clone(), *transform));
            },
        );
        ecs.component_pool
            .query_for_each::<(Spline, Transform), _>(|entity, (spline, _)| {
                spline_map
                    .entry(entity)
                    .or_insert_with(|| (spline.clone(), Transform::default()));
            });

        ecs.component_pool
            .query_mut_for_each::<(SplineFollower, Transform), _>(
                |entity, (follower, transform)| {
                    let target = follower
                        .spline_entity
                        .map(|id| id as usize)
                        .unwrap_or(entity);
                    let Some((spline, spline_transform)) = spline_map.get(&target) else {
                        return;
                    };
                    if !spline.is_valid() {
                        return;
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
                },
            );
    }
}
