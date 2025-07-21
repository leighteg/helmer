use basis_universal::transcoding::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
use glam::Vec3;
use gltf::import_slice;
use hashbrown::HashMap;
use ktx2::Reader;
use mikktspace::Geometry;
use parking_lot::RwLock;
use serde::Deserialize;
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};
use tracing::{info, warn};

use crate::{graphics::renderer::renderer::{Aabb, Vertex}, runtime::runtime::RenderMessage};

// FIX: Define AssetKind here so it's in scope.
#[derive(Debug, Clone, Copy)]
pub enum AssetKind {
    Albedo,
    Normal,
    MetallicRoughness,
    Emission,
}

// --- ASSET STRUCTS & HANDLES ---

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Handle<T> {
    pub id: usize,
    _phantom: PhantomData<T>,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[derive(Debug)]
pub struct Texture {
    pub dimensions: (u32, u32),
    pub data: Vec<u8>,
    pub format: wgpu::TextureFormat,
}

pub struct MaterialGpuData {
    pub id: usize, // The Handle ID for this material
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub emission_strength: f32,
    pub emission_color: [f32; 3],
    // We send the Handle IDs of the textures, not the final indices
    pub albedo_texture_id: Option<usize>,
    pub normal_texture_id: Option<usize>,
    pub metallic_roughness_texture_id: Option<usize>,
    pub emission_texture_id: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct MaterialFile {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub emission_strength: f32,
    pub emission_color: [f32; 3],
    pub albedo_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub emission_texture: Option<String>,
}

#[derive(Debug)]
pub struct Material {
    pub id: usize,
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub emission_strength: f32,
    pub emission_color: [f32; 3],
    pub albedo_texture: Option<Handle<Texture>>,
    pub normal_texture: Option<Handle<Texture>>,
    pub metallic_roughness_texture: Option<Handle<Texture>>,
    pub emission_texture: Option<Handle<Texture>>,
}

// --- WORKER THREAD COMMUNICATION ---

enum AssetLoadRequest {
    Mesh {
        id: usize,
        path: PathBuf,
    },
    Texture {
        id: usize,
        path: PathBuf,
        kind: AssetKind,
    },
    Material {
        id: usize,
        path: PathBuf,
    },
}

enum AssetLoadResult {
    Mesh {
        id: usize,
        data: (Vec<Vertex>, Vec<u32>, Aabb),
    },
    Texture {
        id: usize,
        name: String,
        kind: AssetKind,
        data: (Vec<u8>, wgpu::TextureFormat, (u32, u32)),
    },
    Material {
        id: usize,
        data: MaterialFile,
    },
}

// --- ASSET SERVER ---

pub struct AssetServer {
    // We only store materials loaded from .ron files here for now.
    // glTF materials are handled internally by the renderer.
    materials: Arc<RwLock<HashMap<usize, Arc<Material>>>>,
    next_id: AtomicUsize,
    request_sender: crossbeam_channel::Sender<AssetLoadRequest>,
    result_receiver: crossbeam_channel::Receiver<AssetLoadResult>,
    render_sender: mpsc::Sender<RenderMessage>,
    worker_handles: Vec<JoinHandle<()>>,
}

impl AssetServer {
    pub fn new(render_sender: mpsc::Sender<RenderMessage>) -> Self {
        let (request_sender, request_receiver) = crossbeam_channel::unbounded();
        let (result_sender, result_receiver) = crossbeam_channel::unbounded();
        let mut worker_handles = Vec::new();

        for _ in 0..4 {
            let req_receiver = request_receiver.clone();
            let res_sender = result_sender.clone();
            let handle = thread::spawn(move || {
                while let Ok(request) = req_receiver.recv() {
                    let path_str = request_path(&request).to_string_lossy().to_string();
                    let result = match request {
                        AssetLoadRequest::Mesh { id, path } => {
                            load_and_parse(id, &path, parse_glb, |id, data| AssetLoadResult::Mesh {
                                id,
                                data,
                            })
                        }
                        AssetLoadRequest::Texture { id, path, kind } => {
                            load_and_parse(id, &path, decode_ktx2, |id, data| {
                                AssetLoadResult::Texture {
                                    id,
                                    name: path_str.clone(),
                                    kind,
                                    data,
                                }
                            })
                        }
                        AssetLoadRequest::Material { id, path } => {
                            load_and_parse(id, &path, parse_ron_material, |id, data| {
                                AssetLoadResult::Material { id, data }
                            })
                        }
                    };
                    if let Some(result) = result {
                        if res_sender.send(result).is_err() {
                            break;
                        }
                    }
                }
            });
            worker_handles.push(handle);
        }

        info!("initialized AssetServer");

        Self {
            materials: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicUsize::new(0),
            request_sender,
            result_receiver,
            render_sender,
            worker_handles,
        }
    }

    pub fn load_mesh<P: AsRef<Path>>(&self, path: P) -> Handle<Mesh> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = AssetLoadRequest::Mesh {
            id,
            path: path.as_ref().to_path_buf(),
        };
        self.request_sender.send(request).unwrap();
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn load_texture<P: AsRef<Path>>(&self, path: P, kind: AssetKind) -> Handle<Texture> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = AssetLoadRequest::Texture {
            id,
            path: path.as_ref().to_path_buf(),
            kind,
        };
        self.request_sender.send(request).unwrap();
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn load_material<P: AsRef<Path>>(&self, path: P) -> Handle<Material> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = AssetLoadRequest::Material {
            id,
            path: path.as_ref().to_path_buf(),
        };
        self.request_sender.send(request).unwrap();
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn update(&self) {
        while let Ok(result) = self.result_receiver.try_recv() {
            match result {
                AssetLoadResult::Mesh { id, data } => {
                    let (vertices, indices, bounds) = data;
                    self.render_sender
                        .send(RenderMessage::CreateMesh {
                            id,
                            vertices,
                            indices,
                            bounds,
                        })
                        .unwrap();
                }
                AssetLoadResult::Texture {
                    id,
                    name,
                    kind,
                    data,
                } => {
                    let (texture_data, format, (width, height)) = data;
                    self.render_sender
                        .send(RenderMessage::CreateTexture {
                            id,
                            name,
                            kind,
                            data: texture_data,
                            format,
                            width,
                            height
                        })
                        .unwrap();
                }
                AssetLoadResult::Material { id, data } => {
                    let material_gpu_data = self.resolve_material_dependencies(id, data);
                    self.render_sender
                        .send(RenderMessage::CreateMaterial(material_gpu_data))
                        .unwrap();
                }
            }
        }
    }

    fn resolve_material_dependencies(&self, id: usize, data: MaterialFile) -> MaterialGpuData {
        MaterialGpuData {
            id,
            albedo: data.albedo,
            metallic: data.metallic,
            roughness: data.roughness,
            ao: data.ao,
            emission_strength: data.emission_strength,
            emission_color: data.emission_color,
            albedo_texture_id: data
                .albedo_texture
                .map(|p| self.load_texture(p, AssetKind::Albedo).id),
            normal_texture_id: data
                .normal_texture
                .map(|p| self.load_texture(p, AssetKind::Normal).id),
            metallic_roughness_texture_id: data
                .metallic_roughness_texture
                .map(|p| self.load_texture(p, AssetKind::MetallicRoughness).id),
            emission_texture_id: data
                .emission_texture
                .map(|p| self.load_texture(p, AssetKind::Emission).id),
        }
    }
}

// --- WORKER HELPER FUNCTIONS ---

fn request_path(req: &AssetLoadRequest) -> &Path {
    match req {
        AssetLoadRequest::Mesh { path, .. } => path,
        AssetLoadRequest::Texture { path, .. } => path,
        AssetLoadRequest::Material { path, .. } => path,
    }
}

fn load_and_parse<T, F, R>(
    id: usize,
    path: &Path,
    parse_fn: F,
    result_fn: R,
) -> Option<AssetLoadResult>
where
    F: Fn(&[u8]) -> Result<T, String>,
    R: Fn(usize, T) -> AssetLoadResult,
{
    match std::fs::read(path) {
        Ok(bytes) => match parse_fn(&bytes) {
            Ok(data) => Some(result_fn(id, data)),
            Err(e) => {
                warn!("Failed to parse file '{:?}' for handle {}: {}", path, id, e);
                None
            }
        },
        Err(e) => {
            warn!("Failed to read file '{:?}' for handle {}: {}", path, id, e);
            None
        }
    }
}

/// A robust KTX2 decoder that handles both uncompressed and supercompressed (Basis) files.
fn decode_ktx2(bytes: &[u8]) -> Result<(Vec<u8>, wgpu::TextureFormat, (u32, u32)), String> {
    let reader = Reader::new(bytes).map_err(|e| e.to_string())?;
    let header = reader.header();

    let dimensions = (header.pixel_width, header.pixel_height);

    // Case 1: The KTX2 file contains uncompressed data in a known format.
    if let Some(format) = header.format {
        let wgpu_format = match format {
            ktx2::Format::R8G8B8A8_UNORM => wgpu::TextureFormat::Rgba8Unorm,
            ktx2::Format::R8G8B8A8_SRGB => wgpu::TextureFormat::Rgba8UnormSrgb,
            _ => return Err(format!("Unsupported direct KTX2 format: {:?}", format)),
        };

        info!("KTX2 is uncompressed {:?}. Using raw data.", wgpu_format);
        let level_data = reader
            .levels()
            .next()
            .ok_or("No image levels found in uncompressed KTX2 file.")?;
        return Ok((level_data.data.to_vec(), wgpu_format, dimensions));
    }

    // Case 2: The KTX2 file is supercompressed with Basis Universal.
    info!("KTX2 is supercompressed. Attempting to transcode...");
    let mut transcoder = Transcoder::new();

    if transcoder.prepare_transcoding(bytes).is_err() {
        return Err("Failed to prepare Basis Universal transcoder. The KTX2 file might be invalid or not supercompressed.".to_string());
    }

    // We'll transcode to BC7, a common high-quality format for desktop GPUs.
    let target_basis_format = TranscoderTextureFormat::BC7_RGBA;
    let target_wgpu_format = wgpu::TextureFormat::Bc7RgbaUnormSrgb;

    let transcode_params = TranscodeParameters {
        level_index: 0,
        ..Default::default()
    };

    match transcoder.transcode_image_level(bytes, target_basis_format, transcode_params) {
        Ok(transcoded_data) => {
            info!("Successfully transcoded KTX2 to {:?}.", target_wgpu_format);
            Ok((transcoded_data, target_wgpu_format, dimensions))
        }
        Err(e) => Err(format!(
            "Failed to transcode KTX2 image level: {:?}. Is it a valid Basis Universal file?",
            e
        )),
    }
}

fn parse_ron_material(bytes: &[u8]) -> Result<MaterialFile, String> {
    ron::de::from_bytes(bytes).map_err(|e| e.to_string())
}

struct MikkTSpaceWrapper<'a> {
    vertices: &'a mut [Vertex],
    indices: &'a [u32],
}

impl<'a> Geometry for MikkTSpaceWrapper<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.vertices[self.indices[face * 3 + vert] as usize].position
    }
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.vertices[self.indices[face * 3 + vert] as usize].normal
    }
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.vertices[self.indices[face * 3 + vert] as usize].tex_coord
    }
    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.vertices[self.indices[face * 3 + vert] as usize].tangent = tangent;
    }
}

/// Parses a .glb file's binary content, now with tangent generation.
fn parse_glb(bytes: &[u8]) -> Result<(Vec<Vertex>, Vec<u32>, Aabb), String> {
    let (gltf, buffers, _) = gltf::import_slice(bytes).map_err(|e| e.to_string())?;

    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| {
                    format!(
                        "Primitive in mesh '{}' has no positions",
                        mesh.name().unwrap_or("unnamed")
                    )
                })?
                .collect();

            let vertex_count = positions.len();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count]);

            let tex_coords: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count]);

            let index_offset = all_vertices.len() as u32;
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|r| r.into_u32().map(|i| i + index_offset).collect())
                .ok_or_else(|| {
                    format!(
                        "Primitive in mesh '{}' has no indices",
                        mesh.name().unwrap_or("unnamed")
                    )
                })?;

            let mut vertices: Vec<Vertex> = positions
                .iter()
                .zip(normals.iter())
                .zip(tex_coords.iter())
                .map(|((&p, &n), &t)| Vertex::new(p, n, t, [0.0; 4]))
                .collect();

            if !indices.is_empty()
                && reader.read_normals().is_some()
                && reader.read_tex_coords(0).is_some()
            {
                let mut temp_indices: Vec<u32> =
                    reader.read_indices().unwrap().into_u32().collect();
                let mut wrapper = MikkTSpaceWrapper {
                    vertices: &mut vertices,
                    indices: &temp_indices,
                };
                if !mikktspace::generate_tangents(&mut wrapper) {
                    warn!(
                        "Failed to generate tangents for primitive in mesh '{}'",
                        mesh.name().unwrap_or("unnamed")
                    );
                }
            } else {
                warn!(
                    "Skipping tangent generation for primitive in mesh '{}' due to missing normals or texture coordinates.",
                    mesh.name().unwrap_or("unnamed")
                );
            }

            all_vertices.append(&mut vertices);
            all_indices.append(&mut indices.clone());
        }
    }

    if all_vertices.is_empty() {
        return Err("No valid primitives found in glTF file".to_string());
    }

    // --- Calculate bounds ---
    let mut min_bounds = Vec3::splat(f32::MAX);
    let mut max_bounds = Vec3::splat(f32::MIN);

    for vertex in &all_vertices {
        min_bounds = min_bounds.min(Vec3::from(vertex.position));
        max_bounds = max_bounds.max(Vec3::from(vertex.position));
    }

    let bounds = Aabb {
        min: min_bounds,
        max: max_bounds,
    };

    Ok((all_vertices, all_indices, bounds))
}
