use glam::{Mat4, Vec3};

use crate::graphics::renderer_common::common::{Aabb, Vertex};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtBvhNode {
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub left: u32,
    pub right: u32,
    pub first_index: u32,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtTriangle {
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
    pub n0: [f32; 4],
    pub n1: [f32; 4],
    pub n2: [f32; 4],
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub _pad0: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtBlasDesc {
    pub node_offset: u32,
    pub node_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub tri_offset: u32,
    pub tri_count: u32,
    pub _pad0: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtInstance {
    pub model: [[f32; 4]; 4],
    pub inv_model: [[f32; 4]; 4],
    pub blas_index: u32,
    pub material_id: u32,
    pub _pad0: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtConstants {
    pub rng_frame_index: u32,
    pub accumulation_frame: u32,
    pub max_bounces: u32,
    pub samples_per_frame: u32,
    pub light_count: u32,
    pub flags: u32,
    pub reset: u32,
    pub width: u32,
    pub height: u32,
    pub tlas_node_count: u32,
    pub tlas_index_count: u32,
    pub instance_count: u32,
    pub blas_desc_count: u32,
    pub direct_light_samples: u32,
    pub max_accumulation_frames: u32,
    pub sky_view_samples: u32,
    pub sky_sun_samples: u32,
    pub exposure: f32,
    pub env_intensity: f32,
    pub firefly_clamp: f32,
    pub shadow_bias: f32,
    pub ray_bias: f32,
    pub min_roughness: f32,
    pub normal_map_strength: f32,
    pub throughput_cutoff: f32,
    pub sky_multi_scatter_strength: f32,
    pub sky_multi_scatter_power: f32,
    pub texture_array_layers: u32,
}

pub const RT_FLAG_DIRECT_LIGHTING: u32 = 1 << 0;
pub const RT_FLAG_SHADOWS: u32 = 1 << 1;
pub const RT_FLAG_USE_TEXTURES: u32 = 1 << 2;
pub const RT_FLAG_SHADE_SMOOTH: u32 = 1 << 3;

#[derive(Clone, Debug)]
pub struct BlasBuild {
    pub nodes: Vec<RtBvhNode>,
    pub indices: Vec<u32>,
    pub triangles: Vec<RtTriangle>,
}

#[derive(Clone, Debug)]
pub struct TlasBuild {
    pub nodes: Vec<RtBvhNode>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct BoundsInfo {
    bounds: Aabb,
    centroid: Vec3,
}

pub fn build_blas(vertices: &[Vertex], indices: &[u32], leaf_size: usize) -> BlasBuild {
    let mut triangles = Vec::new();
    let mut infos = Vec::new();

    let tri_count = indices.len() / 3;
    triangles.reserve(tri_count);
    infos.reserve(tri_count);

    for tri in 0..tri_count {
        let i0 = indices[tri * 3] as usize;
        let i1 = indices[tri * 3 + 1] as usize;
        let i2 = indices[tri * 3 + 2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }

        let p0 = Vec3::from(vertices[i0].position);
        let p1 = Vec3::from(vertices[i1].position);
        let p2 = Vec3::from(vertices[i2].position);
        let n0 = Vec3::from(vertices[i0].normal);
        let n1 = Vec3::from(vertices[i1].normal);
        let n2 = Vec3::from(vertices[i2].normal);

        let min = p0.min(p1).min(p2);
        let max = p0.max(p1).max(p2);
        let centroid = (p0 + p1 + p2) / 3.0;

        let uv0 = vertices[i0].tex_coord;
        let uv1 = vertices[i1].tex_coord;
        let uv2 = vertices[i2].tex_coord;

        triangles.push(RtTriangle {
            v0: [p0.x, p0.y, p0.z, 0.0],
            v1: [p1.x, p1.y, p1.z, 0.0],
            v2: [p2.x, p2.y, p2.z, 0.0],
            n0: [n0.x, n0.y, n0.z, 0.0],
            n1: [n1.x, n1.y, n1.z, 0.0],
            n2: [n2.x, n2.y, n2.z, 0.0],
            uv0: [uv0[0], uv0[1]],
            uv1: [uv1[0], uv1[1]],
            uv2: [uv2[0], uv2[1]],
            _pad0: [0.0, 0.0],
        });

        infos.push(BoundsInfo {
            bounds: Aabb { min, max },
            centroid,
        });
    }

    let (nodes, leaf_indices) = build_bvh(&infos, leaf_size);

    BlasBuild {
        nodes,
        indices: leaf_indices,
        triangles,
    }
}

pub fn build_tlas(bounds: &[Aabb], leaf_size: usize) -> TlasBuild {
    let infos: Vec<BoundsInfo> = bounds
        .iter()
        .map(|b| BoundsInfo {
            bounds: *b,
            centroid: b.center(),
        })
        .collect();
    let (nodes, indices) = build_bvh(&infos, leaf_size);
    TlasBuild { nodes, indices }
}

pub fn transform_aabb(aabb: Aabb, transform: Mat4) -> Aabb {
    let corners = aabb.get_corners();
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for corner in corners.iter() {
        let p = transform.transform_point3(*corner);
        min = min.min(p);
        max = max.max(p);
    }
    Aabb { min, max }
}

fn build_bvh(items: &[BoundsInfo], leaf_size: usize) -> (Vec<RtBvhNode>, Vec<u32>) {
    let mut indices: Vec<u32> = (0..items.len() as u32).collect();
    let mut nodes = Vec::new();
    let mut leaf_indices = Vec::new();
    if !indices.is_empty() {
        let node_capacity = items.len().saturating_mul(2).saturating_sub(1);
        nodes.reserve(node_capacity);
        leaf_indices.reserve(items.len());
        build_node(
            &mut indices,
            items,
            leaf_size.max(1),
            &mut nodes,
            &mut leaf_indices,
        );
    }
    (nodes, leaf_indices)
}

fn build_node(
    indices: &mut [u32],
    items: &[BoundsInfo],
    leaf_size: usize,
    nodes: &mut Vec<RtBvhNode>,
    leaf_indices: &mut Vec<u32>,
) -> u32 {
    let bounds = compute_bounds(indices, items);
    let node_index = nodes.len() as u32;
    nodes.push(RtBvhNode {
        bounds_min: [bounds.min.x, bounds.min.y, bounds.min.z, 0.0],
        bounds_max: [bounds.max.x, bounds.max.y, bounds.max.z, 0.0],
        left: 0,
        right: 0,
        first_index: 0,
        index_count: 0,
    });

    if indices.len() <= leaf_size {
        let first = leaf_indices.len() as u32;
        leaf_indices.extend_from_slice(indices);
        let node = &mut nodes[node_index as usize];
        node.first_index = first;
        node.index_count = indices.len() as u32;
        return node_index;
    }

    let centroid_bounds = compute_centroid_bounds(indices, items);
    let extent = centroid_bounds.max - centroid_bounds.min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    let mid = indices.len() / 2;
    indices.select_nth_unstable_by(mid, |a, b| {
        let ca = items[*a as usize].centroid[axis];
        let cb = items[*b as usize].centroid[axis];
        ca.total_cmp(&cb)
    });
    let (left_indices, right_indices) = indices.split_at_mut(mid);
    if left_indices.is_empty() || right_indices.is_empty() {
        let first = leaf_indices.len() as u32;
        leaf_indices.extend_from_slice(indices);
        let node = &mut nodes[node_index as usize];
        node.first_index = first;
        node.index_count = indices.len() as u32;
        return node_index;
    }

    let left = build_node(left_indices, items, leaf_size, nodes, leaf_indices);
    let right = build_node(right_indices, items, leaf_size, nodes, leaf_indices);

    let node = &mut nodes[node_index as usize];
    node.left = left;
    node.right = right;
    node.index_count = 0;
    node.first_index = 0;
    node_index
}

fn compute_bounds(indices: &[u32], items: &[BoundsInfo]) -> Aabb {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &idx in indices {
        let b = items[idx as usize].bounds;
        min = min.min(b.min);
        max = max.max(b.max);
    }
    Aabb { min, max }
}

fn compute_centroid_bounds(indices: &[u32], items: &[BoundsInfo]) -> Aabb {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &idx in indices {
        let c = items[idx as usize].centroid;
        min = min.min(c);
        max = max.max(c);
    }
    Aabb { min, max }
}
