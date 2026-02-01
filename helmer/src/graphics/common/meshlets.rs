use crate::graphics::common::{
    constants::{MESHLET_MAX_PRIMS, MESHLET_MAX_VERTS},
    renderer::{MeshletDesc, MeshletLodData, Vertex},
};
#[cfg(target_arch = "wasm32")]
use glam::Vec3;
#[cfg(target_arch = "wasm32")]
use js_sys::Reflect;
#[cfg(not(target_arch = "wasm32"))]
use meshopt::{VertexDataAdapter, build_meshlets, compute_meshlet_bounds};
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;

pub const DEFAULT_CONE_WEIGHT: f32 = 0.0;

#[cfg(target_arch = "wasm32")]
fn web_global_bool(name: &str) -> Option<bool> {
    let global = js_sys::global();
    let value = Reflect::get(&global, &JsValue::from_str(name)).ok()?;
    if value.is_null() || value.is_undefined() {
        return None;
    }
    if let Some(value) = value.as_bool() {
        return Some(value);
    }
    if let Some(value) = value.as_f64() {
        return Some(value != 0.0);
    }
    if let Some(value) = value.as_string() {
        let value = value.trim().to_ascii_lowercase();
        if matches!(value.as_str(), "1" | "true" | "yes" | "on") {
            return Some(true);
        }
        if matches!(value.as_str(), "0" | "false" | "no" | "off") {
            return Some(false);
        }
    }
    None
}

fn meshlets_enabled() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        web_global_bool("HELMER_WASM_MESHLETS").unwrap_or(false)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}

pub fn build_meshlet_lod(vertices: &[Vertex], indices: &[u32]) -> MeshletLodData {
    if !meshlets_enabled() {
        return MeshletLodData::default();
    }

    if vertices.is_empty() || indices.len() < 3 {
        return MeshletLodData::default();
    }

    let tri_len = indices.len() - (indices.len() % 3);
    if tri_len == 0 {
        return MeshletLodData::default();
    }
    let indices = &indices[..tri_len];

    #[cfg(target_arch = "wasm32")]
    {
        return build_meshlets_simple(vertices, indices);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
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
}

#[cfg(target_arch = "wasm32")]
fn build_meshlets_simple(vertices: &[Vertex], indices: &[u32]) -> MeshletLodData {
    if vertices.is_empty() || indices.len() < 3 {
        return MeshletLodData::default();
    }

    let tri_len = indices.len() - (indices.len() % 3);
    if tri_len == 0 {
        return MeshletLodData::default();
    }
    let indices = &indices[..tri_len];

    let mut out_vertices: Vec<u32> = Vec::new();
    let mut out_indices: Vec<u32> = Vec::new();
    let mut descs: Vec<MeshletDesc> = Vec::new();

    let mut cursor = 0usize;
    while cursor + 2 < indices.len() {
        let mut local_vertices: Vec<u32> = Vec::new();
        let mut local_indices: Vec<u32> = Vec::new();

        while cursor + 2 < indices.len() {
            let tri = &indices[cursor..cursor + 3];
            let mut new_unique = 0usize;
            for &idx in tri {
                if !local_vertices.iter().any(|&v| v == idx) {
                    new_unique += 1;
                }
            }
            let prospective_vertices = local_vertices.len() + new_unique;
            let prospective_tris = (local_indices.len() / 3) + 1;
            if !local_indices.is_empty()
                && (prospective_vertices > MESHLET_MAX_VERTS
                    || prospective_tris > MESHLET_MAX_PRIMS)
            {
                break;
            }

            for &idx in tri {
                let local = if let Some(pos) = local_vertices.iter().position(|&v| v == idx) {
                    pos as u32
                } else {
                    local_vertices.push(idx);
                    (local_vertices.len() - 1) as u32
                };
                local_indices.push(local);
            }
            cursor += 3;
        }

        if local_vertices.is_empty() || local_indices.is_empty() {
            break;
        }

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for &idx in &local_vertices {
            if let Some(v) = vertices.get(idx as usize) {
                let pos = Vec3::from(v.position);
                min = min.min(pos);
                max = max.max(pos);
            }
        }
        let center = (min + max) * 0.5;
        let mut radius = 0.0f32;
        for &idx in &local_vertices {
            if let Some(v) = vertices.get(idx as usize) {
                let pos = Vec3::from(v.position);
                radius = radius.max((pos - center).length());
            }
        }

        let vertex_offset = out_vertices.len() as u32;
        let index_offset = out_indices.len() as u32;
        let vertex_count = local_vertices.len().min(u32::MAX as usize) as u32;
        let index_count = local_indices.len().min(u32::MAX as usize) as u32;

        out_vertices.extend_from_slice(&local_vertices);
        out_indices.extend_from_slice(&local_indices);

        descs.push(MeshletDesc {
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            bounds_center: center.to_array(),
            bounds_radius: radius,
        });
    }

    if descs.is_empty() {
        return MeshletLodData::default();
    }

    MeshletLodData {
        descs: Arc::from(descs),
        vertices: Arc::from(out_vertices),
        indices: Arc::from(out_indices),
    }
}

pub fn meshlet_lod_size_bytes(meshlets: &MeshletLodData) -> usize {
    let descs = meshlets.descs.len() * std::mem::size_of::<MeshletDesc>();
    let vertices = meshlets.vertices.len() * std::mem::size_of::<u32>();
    let indices = meshlets.indices.len() * std::mem::size_of::<u32>();
    descs + vertices + indices
}
