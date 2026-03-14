#![cfg(target_arch = "wasm32")]

use std::{cell::RefCell, collections::HashMap, path::PathBuf, sync::Arc};

use bytemuck::cast_slice;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::wasm_bindgen;

use crate::runtime::asset_server::{
    AssetKind, AssetStreamingTuning, IntermediateMaterial, MeshPayload, MeshPrimitiveDesc,
    StreamedBuffer, TextureRequest, WebAssetIo, build_mesh_payload, decode_texture_asset_web,
    decode_texture_file_bytes, estimate_primitive_bounds, generate_low_res_from_parts,
    is_gltf_path, load_gltf_streaming_web, load_scene_buffers_web, parse_glb, parse_mesh_document,
    parse_ron_material, parse_scene_document, process_primitive,
};
use helmer_render::graphics::common::renderer::{
    Aabb, AssetStreamKind, MeshletDesc, MeshletLodData, Vertex,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum WorkerTextureFormat {
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
}

impl WorkerTextureFormat {
    pub(crate) fn from_wgpu(format: wgpu::TextureFormat) -> Option<Self> {
        match format {
            wgpu::TextureFormat::Rgba8Unorm => Some(Self::Rgba8Unorm),
            wgpu::TextureFormat::Rgba8UnormSrgb => Some(Self::Rgba8UnormSrgb),
            wgpu::TextureFormat::Bc7RgbaUnorm => Some(Self::Bc7RgbaUnorm),
            wgpu::TextureFormat::Bc7RgbaUnormSrgb => Some(Self::Bc7RgbaUnormSrgb),
            _ => None,
        }
    }

    pub(crate) fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            Self::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Bc7RgbaUnorm => wgpu::TextureFormat::Bc7RgbaUnorm,
            Self::Bc7RgbaUnormSrgb => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerMeshletLodData {
    pub descs: Vec<MeshletDesc>,
    pub vertices: Vec<u32>,
    pub indices: Vec<u32>,
}

impl From<MeshletLodData> for WorkerMeshletLodData {
    fn from(data: MeshletLodData) -> Self {
        Self {
            descs: data.descs.as_ref().to_vec(),
            vertices: data.vertices.as_ref().to_vec(),
            indices: data.indices.as_ref().to_vec(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerMeshLodPayload {
    pub lod_index: usize,
    pub vertices: Vec<u8>,
    pub indices: Vec<u32>,
    pub meshlets: WorkerMeshletLodData,
}

impl From<helmer_render::graphics::common::renderer::MeshLodPayload> for WorkerMeshLodPayload {
    fn from(payload: helmer_render::graphics::common::renderer::MeshLodPayload) -> Self {
        let vertices = cast_slice::<Vertex, u8>(payload.vertices.as_ref()).to_vec();
        let indices = payload.indices.as_ref().to_vec();
        let meshlets = WorkerMeshletLodData::from(payload.meshlets);
        Self {
            lod_index: payload.lod_index,
            vertices,
            indices,
            meshlets,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerAabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl From<Aabb> for WorkerAabb {
    fn from(bounds: Aabb) -> Self {
        Self {
            min: bounds.min.to_array(),
            max: bounds.max.to_array(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerMeshPayload {
    pub total_lods: usize,
    pub lods: Vec<WorkerMeshLodPayload>,
    pub bounds: WorkerAabb,
}

impl From<MeshPayload> for WorkerMeshPayload {
    fn from(payload: MeshPayload) -> Self {
        let lods = payload
            .lods
            .into_iter()
            .map(WorkerMeshLodPayload::from)
            .collect();
        Self {
            total_lods: payload.total_lods,
            lods,
            bounds: WorkerAabb::from(payload.bounds),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerTexturePayload {
    pub data: Vec<u8>,
    pub format: WorkerTextureFormat,
    pub dimensions: (u32, u32),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerLowResTexture {
    pub data: Vec<u8>,
    pub format: WorkerTextureFormat,
    pub dimensions: (u32, u32),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerNodeDesc {
    pub node_index: usize,
    pub primitive_desc_index: usize,
    pub material_index: usize,
    pub transform: [f32; 16],
    pub skin_index: Option<usize>,
    pub parent_node_index: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerSceneSummary {
    pub buffers_bytes: usize,
    pub buffers_ready: bool,
    pub base_path: Option<String>,
    pub scene_path: Option<String>,
    pub textures: Vec<TextureRequest>,
    pub materials: Vec<IntermediateMaterial>,
    pub mesh_primitives: Vec<MeshPrimitiveDesc>,
    pub nodes: Vec<WorkerNodeDesc>,
    pub mesh_bounds: Vec<WorkerAabb>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum WorkerRequest {
    Mesh {
        id: usize,
        path: String,
        tuning: AssetStreamingTuning,
    },
    ProceduralMesh {
        id: usize,
        vertices: Vec<u8>,
        indices: Vec<u32>,
        tuning: AssetStreamingTuning,
    },
    Texture {
        id: usize,
        path: String,
        kind: AssetKind,
    },
    Material {
        id: usize,
        path: String,
    },
    Scene {
        id: usize,
        path: String,
    },
    SceneBuffers {
        scene_id: usize,
    },
    StreamMesh {
        id: usize,
        scene_id: usize,
        desc: MeshPrimitiveDesc,
        tuning: AssetStreamingTuning,
    },
    StreamTexture {
        id: usize,
        scene_id: usize,
        tex_index: usize,
        kind: AssetKind,
    },
    LowResTexture {
        id: usize,
        name: String,
        kind: AssetKind,
        data: Vec<u8>,
        format: WorkerTextureFormat,
        dimensions: (u32, u32),
        max_dim: u32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum WorkerResponse {
    Mesh {
        id: usize,
        scene_id: Option<usize>,
        payload: WorkerMeshPayload,
    },
    Texture {
        id: usize,
        scene_id: Option<usize>,
        name: String,
        kind: AssetKind,
        data: WorkerTexturePayload,
    },
    Material {
        id: usize,
        data: crate::runtime::asset_server::MaterialFile,
    },
    Scene {
        id: usize,
        summary: WorkerSceneSummary,
    },
    SceneBuffers {
        scene_id: usize,
        buffers_bytes: usize,
    },
    SceneBuffersFailed {
        scene_id: usize,
        error: String,
    },
    LowResTexture {
        id: usize,
        name: String,
        kind: AssetKind,
        data: WorkerLowResTexture,
    },
    StreamFailure {
        kind: AssetStreamKind,
        id: usize,
        scene_id: usize,
    },
    Error {
        message: String,
    },
}

#[derive(Clone)]
struct WorkerScene {
    doc: Arc<gltf::Document>,
    buffers: Option<Arc<Vec<StreamedBuffer>>>,
    buffers_bytes: usize,
    base_path: Option<PathBuf>,
    scene_path: Option<PathBuf>,
}

struct WorkerState {
    io: WebAssetIo,
    scenes: HashMap<usize, WorkerScene>,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            io: WebAssetIo::new(),
            scenes: HashMap::new(),
        }
    }
}

thread_local! {
    static WORKER_STATE: RefCell<WorkerState> = RefCell::new(WorkerState::new());
}

#[wasm_bindgen]
pub fn helmer_worker_set_opfs_enabled(enabled: bool) {
    WORKER_STATE.with(|state| {
        state.borrow().io.set_opfs_enabled(enabled);
    });
}

#[wasm_bindgen]
pub fn helmer_worker_store_virtual_asset(path: String, bytes: Vec<u8>) {
    let path_buf = PathBuf::from(path);
    WORKER_STATE.with(|state| {
        state
            .borrow()
            .io
            .store_virtual_asset_path(path_buf.as_path(), bytes);
    });
}

#[wasm_bindgen]
pub fn helmer_worker_release_scene_buffers(scene_id: usize) {
    WORKER_STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(scene) = state.scenes.get_mut(&scene_id) {
            scene.buffers = None;
            scene.buffers_bytes = 0;
        }
    });
}

#[wasm_bindgen]
pub async fn handle_worker_request(bytes: Vec<u8>) -> Vec<u8> {
    let request: WorkerRequest = match bincode::deserialize(&bytes) {
        Ok(request) => request,
        Err(err) => {
            let response = WorkerResponse::Error {
                message: format!("failed to decode worker request: {}", err),
            };
            return bincode::serialize(&response).unwrap_or_default();
        }
    };

    let response = process_request(request).await;
    bincode::serialize(&response).unwrap_or_default()
}

async fn process_request(request: WorkerRequest) -> WorkerResponse {
    match request {
        WorkerRequest::Mesh { id, path, tuning } => {
            let path_buf = PathBuf::from(&path);
            let io = WORKER_STATE.with(|state| state.borrow().io.clone());
            let parsed = if is_gltf_path(&path_buf) {
                match load_gltf_streaming_web(&path_buf, &io).await {
                    Ok((doc, buffers)) => match parse_mesh_document(&doc, &buffers) {
                        Ok(data) => Some(data),
                        Err(err) => {
                            return WorkerResponse::Error {
                                message: format!("failed to parse glTF mesh '{}': {}", path, err),
                            };
                        }
                    },
                    Err(err) => {
                        return WorkerResponse::Error {
                            message: format!("failed to load glTF mesh '{}': {}", path, err),
                        };
                    }
                }
            } else {
                match io.read_path(&path_buf).await {
                    Ok(bytes) => match parse_glb(&bytes) {
                        Ok(data) => Some(data),
                        Err(err) => {
                            return WorkerResponse::Error {
                                message: format!("failed to parse glb mesh '{}': {}", path, err),
                            };
                        }
                    },
                    Err(err) => {
                        return WorkerResponse::Error {
                            message: format!("failed to read mesh '{}': {}", path, err),
                        };
                    }
                }
            };

            let Some((vertices, indices, bounds)) = parsed else {
                return WorkerResponse::Error {
                    message: format!("mesh '{}' produced no geometry", path),
                };
            };

            match build_mesh_payload(vertices, indices, bounds, &tuning) {
                Some(payload) => WorkerResponse::Mesh {
                    id,
                    scene_id: None,
                    payload: WorkerMeshPayload::from(payload),
                },
                None => WorkerResponse::Error {
                    message: format!("mesh '{}' produced no payload", path),
                },
            }
        }
        WorkerRequest::ProceduralMesh {
            id,
            vertices,
            indices,
            tuning,
        } => {
            if vertices.len() % std::mem::size_of::<Vertex>() != 0 {
                return WorkerResponse::Error {
                    message: "procedural mesh vertex buffer size mismatch".to_string(),
                };
            }
            let vertex_data: Vec<Vertex> = cast_slice(&vertices).to_vec();
            let bounds = Aabb::calculate(&vertex_data);
            match build_mesh_payload(vertex_data, indices, bounds, &tuning) {
                Some(payload) => WorkerResponse::Mesh {
                    id,
                    scene_id: None,
                    payload: WorkerMeshPayload::from(payload),
                },
                None => WorkerResponse::Error {
                    message: "procedural mesh produced no payload".to_string(),
                },
            }
        }
        WorkerRequest::Texture { id, path, kind } => {
            let io = WORKER_STATE.with(|state| state.borrow().io.clone());
            let path_buf = PathBuf::from(&path);
            match io.read_path(&path_buf).await {
                Ok(bytes) => match decode_texture_file_bytes(kind, &bytes, &path_buf) {
                    Ok((data, format, dimensions)) => {
                        match WorkerTextureFormat::from_wgpu(format) {
                            Some(worker_format) => WorkerResponse::Texture {
                                id,
                                scene_id: None,
                                name: path.clone(),
                                kind,
                                data: WorkerTexturePayload {
                                    data,
                                    format: worker_format,
                                    dimensions,
                                },
                            },
                            None => WorkerResponse::Error {
                                message: format!("unsupported texture format for '{}'", path),
                            },
                        }
                    }
                    Err(err) => WorkerResponse::Error {
                        message: format!("failed to decode texture '{}': {}", path, err),
                    },
                },
                Err(err) => WorkerResponse::Error {
                    message: format!("failed to read texture '{}': {}", path, err),
                },
            }
        }
        WorkerRequest::Material { id, path } => {
            let io = WORKER_STATE.with(|state| state.borrow().io.clone());
            let path_buf = PathBuf::from(&path);
            match io.read_path(&path_buf).await {
                Ok(bytes) => match parse_ron_material(&bytes) {
                    Ok(data) => WorkerResponse::Material { id, data },
                    Err(err) => WorkerResponse::Error {
                        message: format!("failed to parse material '{}': {}", path, err),
                    },
                },
                Err(err) => WorkerResponse::Error {
                    message: format!("failed to read material '{}': {}", path, err),
                },
            }
        }
        WorkerRequest::Scene { id, path } => {
            let io = WORKER_STATE.with(|state| state.borrow().io.clone());
            let path_buf = PathBuf::from(&path);
            let parsed = if is_gltf_path(&path_buf) {
                match io.read_path(&path_buf).await {
                    Ok(bytes) => match gltf::Gltf::from_slice(&bytes) {
                        Ok(gltf) => parse_scene_document(
                            gltf.document,
                            None,
                            path_buf.parent(),
                            Some(path_buf.clone()),
                        ),
                        Err(err) => Err(err.to_string()),
                    },
                    Err(err) => Err(err),
                }
            } else {
                match io.read_path(&path_buf).await {
                    Ok(bytes) => match gltf::import_slice(&bytes) {
                        Ok((doc, buffers, _)) => {
                            let buffers = buffers
                                .into_iter()
                                .map(|data| StreamedBuffer::owned(data.to_vec()))
                                .collect();
                            parse_scene_document(
                                doc,
                                Some(buffers),
                                path_buf.parent(),
                                Some(path_buf.clone()),
                            )
                        }
                        Err(err) => Err(err.to_string()),
                    },
                    Err(err) => Err(err),
                }
            };

            match parsed {
                Ok(parsed) => {
                    let summary = build_scene_summary(&parsed);
                    let scene = WorkerScene {
                        doc: parsed.doc.clone(),
                        buffers: parsed.buffers.clone(),
                        buffers_bytes: parsed.buffers_bytes,
                        base_path: parsed.base_path.clone(),
                        scene_path: parsed.scene_path.clone(),
                    };
                    WORKER_STATE.with(|state| {
                        state.borrow_mut().scenes.insert(id, scene);
                    });
                    WorkerResponse::Scene { id, summary }
                }
                Err(err) => WorkerResponse::Error {
                    message: format!("failed to parse scene '{}': {}", path, err),
                },
            }
        }
        WorkerRequest::SceneBuffers { scene_id } => {
            let io = WORKER_STATE.with(|state| state.borrow().io.clone());
            let snapshot = WORKER_STATE.with(|state| {
                let state = state.borrow();
                state.scenes.get(&scene_id).cloned()
            });
            let Some(scene) = snapshot else {
                return WorkerResponse::SceneBuffersFailed {
                    scene_id,
                    error: "scene not found".to_string(),
                };
            };
            if scene.buffers.is_some() {
                return WorkerResponse::SceneBuffers {
                    scene_id,
                    buffers_bytes: scene.buffers_bytes,
                };
            }
            match load_scene_buffers_web(
                &scene.doc,
                scene.scene_path.as_deref(),
                scene.base_path.as_deref(),
                &io,
            )
            .await
            {
                Ok(buffers) => {
                    let buffers_bytes: usize = buffers.iter().map(|buffer| buffer.len()).sum();
                    let buffers = Arc::new(buffers);
                    WORKER_STATE.with(|state| {
                        if let Some(entry) = state.borrow_mut().scenes.get_mut(&scene_id) {
                            entry.buffers = Some(buffers);
                            entry.buffers_bytes = buffers_bytes;
                        }
                    });
                    WorkerResponse::SceneBuffers {
                        scene_id,
                        buffers_bytes,
                    }
                }
                Err(err) => WorkerResponse::SceneBuffersFailed {
                    scene_id,
                    error: err,
                },
            }
        }
        WorkerRequest::StreamMesh {
            id,
            scene_id,
            desc,
            tuning,
        } => {
            let io = WORKER_STATE.with(|state| state.borrow().io.clone());
            let mut snapshot = WORKER_STATE.with(|state| {
                let state = state.borrow();
                state.scenes.get(&scene_id).cloned()
            });
            if snapshot.is_none() {
                return WorkerResponse::StreamFailure {
                    kind: AssetStreamKind::Mesh,
                    id,
                    scene_id,
                };
            }
            let mut scene = snapshot.take().unwrap();
            if scene.buffers.is_none() {
                match load_scene_buffers_web(
                    &scene.doc,
                    scene.scene_path.as_deref(),
                    scene.base_path.as_deref(),
                    &io,
                )
                .await
                {
                    Ok(buffers) => {
                        let buffers_bytes: usize = buffers.iter().map(|buffer| buffer.len()).sum();
                        let buffers = Arc::new(buffers);
                        scene.buffers = Some(Arc::clone(&buffers));
                        scene.buffers_bytes = buffers_bytes;
                        WORKER_STATE.with(|state| {
                            if let Some(entry) = state.borrow_mut().scenes.get_mut(&scene_id) {
                                entry.buffers = Some(buffers);
                                entry.buffers_bytes = buffers_bytes;
                            }
                        });
                    }
                    Err(_) => {
                        return WorkerResponse::StreamFailure {
                            kind: AssetStreamKind::Mesh,
                            id,
                            scene_id,
                        };
                    }
                }
            }

            let Some(buffers) = scene.buffers.as_ref() else {
                return WorkerResponse::StreamFailure {
                    kind: AssetStreamKind::Mesh,
                    id,
                    scene_id,
                };
            };
            let primitive = scene
                .doc
                .meshes()
                .nth(desc.mesh_index)
                .and_then(|mesh| mesh.primitives().nth(desc.primitive_index));
            if let Some(primitive) = primitive {
                if let Some((vertices, indices, bounds)) =
                    process_primitive(&primitive, buffers.as_ref())
                {
                    match build_mesh_payload(vertices, indices, bounds, &tuning) {
                        Some(payload) => WorkerResponse::Mesh {
                            id,
                            scene_id: Some(scene_id),
                            payload: WorkerMeshPayload::from(payload),
                        },
                        None => WorkerResponse::StreamFailure {
                            kind: AssetStreamKind::Mesh,
                            id,
                            scene_id,
                        },
                    }
                } else {
                    WorkerResponse::StreamFailure {
                        kind: AssetStreamKind::Mesh,
                        id,
                        scene_id,
                    }
                }
            } else {
                WorkerResponse::StreamFailure {
                    kind: AssetStreamKind::Mesh,
                    id,
                    scene_id,
                }
            }
        }
        WorkerRequest::StreamTexture {
            id,
            scene_id,
            tex_index,
            kind,
        } => {
            let io = WORKER_STATE.with(|state| state.borrow().io.clone());
            let mut snapshot = WORKER_STATE.with(|state| {
                let state = state.borrow();
                state.scenes.get(&scene_id).cloned()
            });
            if snapshot.is_none() {
                return WorkerResponse::StreamFailure {
                    kind: AssetStreamKind::Texture,
                    id,
                    scene_id,
                };
            }
            let mut scene = snapshot.take().unwrap();
            if scene.buffers.is_none() {
                match load_scene_buffers_web(
                    &scene.doc,
                    scene.scene_path.as_deref(),
                    scene.base_path.as_deref(),
                    &io,
                )
                .await
                {
                    Ok(buffers) => {
                        let buffers_bytes: usize = buffers.iter().map(|buffer| buffer.len()).sum();
                        let buffers = Arc::new(buffers);
                        scene.buffers = Some(Arc::clone(&buffers));
                        scene.buffers_bytes = buffers_bytes;
                        WORKER_STATE.with(|state| {
                            if let Some(entry) = state.borrow_mut().scenes.get_mut(&scene_id) {
                                entry.buffers = Some(buffers);
                                entry.buffers_bytes = buffers_bytes;
                            }
                        });
                    }
                    Err(_) => {
                        return WorkerResponse::StreamFailure {
                            kind: AssetStreamKind::Texture,
                            id,
                            scene_id,
                        };
                    }
                }
            }

            let Some(buffers) = scene.buffers.as_ref() else {
                return WorkerResponse::StreamFailure {
                    kind: AssetStreamKind::Texture,
                    id,
                    scene_id,
                };
            };
            if let Some(gltf_tex) = scene.doc.textures().nth(tex_index) {
                let decoded = decode_texture_asset_web(
                    gltf_tex,
                    buffers.as_ref(),
                    scene.base_path.as_deref(),
                    &io,
                    kind,
                )
                .await;
                match decoded {
                    Some((name, kind, data, format, dimensions)) => {
                        match WorkerTextureFormat::from_wgpu(format) {
                            Some(worker_format) => WorkerResponse::Texture {
                                id,
                                scene_id: Some(scene_id),
                                name,
                                kind,
                                data: WorkerTexturePayload {
                                    data,
                                    format: worker_format,
                                    dimensions,
                                },
                            },
                            None => WorkerResponse::StreamFailure {
                                kind: AssetStreamKind::Texture,
                                id,
                                scene_id,
                            },
                        }
                    }
                    None => WorkerResponse::StreamFailure {
                        kind: AssetStreamKind::Texture,
                        id,
                        scene_id,
                    },
                }
            } else {
                WorkerResponse::StreamFailure {
                    kind: AssetStreamKind::Texture,
                    id,
                    scene_id,
                }
            }
        }
        WorkerRequest::LowResTexture {
            id,
            name,
            kind,
            data,
            format,
            dimensions,
            max_dim,
        } => {
            let low = generate_low_res_from_parts(&data, format.to_wgpu(), dimensions, max_dim);
            WorkerResponse::LowResTexture {
                id,
                name,
                kind,
                data: WorkerLowResTexture {
                    data: low.data.as_ref().to_vec(),
                    format: WorkerTextureFormat::from_wgpu(low.format)
                        .unwrap_or(WorkerTextureFormat::Rgba8Unorm),
                    dimensions: low.dimensions,
                },
            }
        }
    }
}

fn build_scene_summary(
    parsed: &crate::runtime::asset_server::ParsedGltfScene,
) -> WorkerSceneSummary {
    let mesh_bounds = parsed
        .mesh_primitives
        .iter()
        .map(|desc| {
            estimate_primitive_bounds(&parsed.doc, desc)
                .unwrap_or(Aabb {
                    min: glam::Vec3::ZERO,
                    max: glam::Vec3::ZERO,
                })
                .into()
        })
        .collect();

    let nodes = parsed
        .nodes
        .iter()
        .map(|node| WorkerNodeDesc {
            node_index: node.node_index,
            primitive_desc_index: node.primitive_desc_index,
            material_index: node.material_index,
            transform: node.transform.to_cols_array(),
            skin_index: node.skin_index,
            parent_node_index: node.parent_node_index,
        })
        .collect();

    WorkerSceneSummary {
        buffers_bytes: parsed.buffers_bytes,
        buffers_ready: parsed.buffers.is_some(),
        base_path: parsed
            .base_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        scene_path: parsed
            .scene_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        textures: parsed.textures.clone(),
        materials: parsed.materials.clone(),
        mesh_primitives: parsed.mesh_primitives.clone(),
        nodes,
        mesh_bounds,
    }
}

#[allow(dead_code)]
pub fn link_worker() {}
