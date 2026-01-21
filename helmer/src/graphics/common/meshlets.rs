use crate::graphics::common::{
    constants::{MESHLET_MAX_PRIMS, MESHLET_MAX_VERTS},
    renderer::{MeshletDesc, MeshletLodData, Vertex},
};
use meshopt::{VertexDataAdapter, build_meshlets, compute_meshlet_bounds};
use std::sync::Arc;

pub const DEFAULT_CONE_WEIGHT: f32 = 0.0;

pub fn build_meshlet_lod(vertices: &[Vertex], indices: &[u32]) -> MeshletLodData {
    if vertices.is_empty() || indices.len() < 3 {
        return MeshletLodData::default();
    }

    let tri_len = indices.len() - (indices.len() % 3);
    if tri_len == 0 {
        return MeshletLodData::default();
    }
    let indices = &indices[..tri_len];

    let adapter = match VertexDataAdapter::new(
        bytemuck::cast_slice(vertices),
        std::mem::size_of::<Vertex>(),
        0,
    ) {
        Ok(adapter) => adapter,
        Err(_) => return MeshletLodData::default(),
    };

    let meshlets = build_meshlets(
        indices,
        &adapter,
        MESHLET_MAX_VERTS,
        MESHLET_MAX_PRIMS,
        DEFAULT_CONE_WEIGHT,
    );

    if meshlets.meshlets.is_empty() {
        return MeshletLodData::default();
    }

    let mut descs = Vec::with_capacity(meshlets.meshlets.len());
    for (idx, meshlet) in meshlets.meshlets.iter().enumerate() {
        let bounds = compute_meshlet_bounds(meshlets.get(idx), &adapter);
        descs.push(MeshletDesc {
            vertex_offset: meshlet.vertex_offset,
            vertex_count: meshlet.vertex_count,
            index_offset: meshlet.triangle_offset,
            index_count: meshlet.triangle_count * 3,
            bounds_center: bounds.center,
            bounds_radius: bounds.radius,
        });
    }

    let indices = meshlets
        .triangles
        .iter()
        .map(|value| *value as u32)
        .collect::<Vec<u32>>();

    MeshletLodData {
        descs: Arc::from(descs),
        vertices: Arc::from(meshlets.vertices),
        indices: Arc::from(indices),
    }
}

pub fn meshlet_lod_size_bytes(meshlets: &MeshletLodData) -> usize {
    let descs = meshlets.descs.len() * std::mem::size_of::<MeshletDesc>();
    let vertices = meshlets.vertices.len() * std::mem::size_of::<u32>();
    let indices = meshlets.indices.len() * std::mem::size_of::<u32>();
    descs + vertices + indices
}
