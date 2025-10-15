use bevy_ecs::{resource::Resource, system::Query};
use helmer::{
    graphics::renderer::renderer::RenderData,
    provided::components::{MeshRenderer, Transform},
};

use crate::AnyComponent;

#[derive(Default, Resource)]
pub struct RenderPacket(pub Option<RenderData>);

pub fn render_collection_system(
    mut query: Query<(&AnyComponent<Transform>, &AnyComponent<MeshRenderer>)>,
) {
    for (transform, mesh_renderer) in &mut query {}
}
