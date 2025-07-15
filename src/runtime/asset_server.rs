use basis_universal::transcoding::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
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
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
};
use tracing::{info, warn};

use crate::{graphics::renderer::renderer::Vertex, runtime::runtime::RenderMessage};

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
    Mesh { id: usize, path: PathBuf },
    Texture { id: usize, path: PathBuf, kind: AssetKind },
    Material { id: usize, path: PathBuf },
}

enum AssetLoadResult {
    Mesh { id: usize, data: Mesh },
    Texture { id: usize, name: String, kind: AssetKind, data: Texture },
    Material { id: usize, data: MaterialFile },
}

// --- ASSET SERVER ---

pub struct AssetServer {
    materials: Arc<RwLock<HashMap<usize, Arc<Material>>>>,
    next_id: AtomicUsize,
    request_sender: mpsc::Sender<AssetLoadRequest>,
    result_receiver: mpsc::Receiver<AssetLoadResult>,
    render_sender: mpsc::Sender<RenderMessage>,
    worker_handles: Vec<JoinHandle<()>>,
}

impl AssetServer {
    pub fn new(render_sender: mpsc::Sender<RenderMessage>) -> Self {
        let (request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let request_receiver = Arc::new(Mutex::new(request_receiver));
        let mut worker_handles = Vec::new();

        for _ in 0..4 {
            let req_receiver = Arc::clone(&request_receiver);
            let res_sender = result_sender.clone();
            let handle = thread::spawn(move || {
                loop {
                    if let Ok(request) = req_receiver.lock().unwrap().recv() {
                        let path_for_name = request_path(&request).to_string_lossy().to_string();
                        let result = match request {
                            // FIX: Pass closures that construct the enum variants.
                            AssetLoadRequest::Mesh { id, path } => {
                                load_and_parse(id, &path, parse_glb, |id, data| AssetLoadResult::Mesh { id, data })
                            }
                            AssetLoadRequest::Texture { id, path, kind } => {
                                load_and_parse(id, &path, decode_ktx2, |id, data| AssetLoadResult::Texture {
                                    id, name: String::clone(&path_for_name), kind, data
                                })
                            }
                            AssetLoadRequest::Material { id, path } => {
                                load_and_parse(id, &path, parse_ron_material, |id, data| AssetLoadResult::Material { id, data })
                            }
                        };
                        if let Some(result) = result {
                            if res_sender.send(result).is_err() { break; }
                        }
                    } else {
                        break;
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
        let request = AssetLoadRequest::Mesh { id, path: path.as_ref().to_path_buf() };
        self.request_sender.send(request).unwrap();
        Handle { id, _phantom: PhantomData }
    }

    pub fn load_texture<P: AsRef<Path>>(&self, path: P, kind: AssetKind) -> Handle<Texture> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = AssetLoadRequest::Texture { id, path: path.as_ref().to_path_buf(), kind };
        self.request_sender.send(request).unwrap();
        Handle { id, _phantom: PhantomData }
    }

    pub fn load_material<P: AsRef<Path>>(&self, path: P) -> Handle<Material> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = AssetLoadRequest::Material { id, path: path.as_ref().to_path_buf() };
        self.request_sender.send(request).unwrap();
        Handle { id, _phantom: PhantomData }
    }

    pub fn update(&self) {
        while let Ok(result) = self.result_receiver.try_recv() {
            match result {
                AssetLoadResult::Mesh { id, data } => {
                    self.render_sender.send(RenderMessage::CreateMesh {
                        id,
                        vertices: data.vertices,
                        indices: data.indices,
                    }).unwrap();
                }
                AssetLoadResult::Texture { id, name, kind, data } => {
                    self.render_sender.send(RenderMessage::CreateTexture {
                        id,
                        name,
                        kind,
                        data: data.data,
                        format: data.format,
                    }).unwrap();
                }
                AssetLoadResult::Material { id, data } => {
                    let material = self.resolve_material_dependencies(id, data);
                    let material_gpu_data = MaterialGpuData {
                        id: material.id,
                        albedo: material.albedo,
                        metallic: material.metallic,
                        roughness: material.roughness,
                        ao: material.ao,
                        emission_strength: material.emission_strength,
                        emission_color: material.emission_color,
                        albedo_texture_id: material.albedo_texture.as_ref().map(|h| h.id),
                        normal_texture_id: material.normal_texture.as_ref().map(|h| h.id),
                        metallic_roughness_texture_id: material.metallic_roughness_texture.as_ref().map(|h| h.id),
                        emission_texture_id: material.emission_texture.as_ref().map(|h| h.id),
                    };
                    self.materials.write().insert(id, Arc::new(material));
                    self.render_sender.send(RenderMessage::CreateMaterial(material_gpu_data)).unwrap();
                }
            }
        }
    }
    
    fn resolve_material_dependencies(&self, id: usize, data: MaterialFile) -> Material {
        Material {
            id,
            albedo: data.albedo,
            metallic: data.metallic,
            roughness: data.roughness,
            ao: data.ao,
            emission_strength: data.emission_strength,
            emission_color: data.emission_color,
            albedo_texture: data.albedo_texture.map(|p| self.load_texture(p, AssetKind::Albedo)),
            normal_texture: data.normal_texture.map(|p| self.load_texture(p, AssetKind::Normal)),
            metallic_roughness_texture: data.metallic_roughness_texture.map(|p| self.load_texture(p, AssetKind::MetallicRoughness)),
            emission_texture: data.emission_texture.map(|p| self.load_texture(p, AssetKind::Emission)),
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

fn load_and_parse<T, F, R>(id: usize, path: &Path, parse_fn: F, result_fn: R) -> Option<AssetLoadResult>
where
    F: Fn(&[u8]) -> Result<T, String>,
    R: Fn(usize, T) -> AssetLoadResult,
{
    match std::fs::read(path) {
        Ok(bytes) => match parse_fn(&bytes) {
            Ok(data) => Some(result_fn(id, data)),
            Err(e) => { warn!("Failed to parse file '{:?}' for handle {}: {}", path, id, e); None }
        },
        Err(e) => { warn!("Failed to read file '{:?}' for handle {}: {}", path, id, e); None }
    }
}

fn decode_ktx2(bytes: &[u8]) -> Result<Texture, String> {
    let reader = Reader::new(bytes).map_err(|e| e.to_string())?;
    let header = reader.header();
    let level0_data = reader.levels().next().ok_or("No image levels in KTX2")?;
    
    let mut transcoder = Transcoder::new();
    let target_format = TranscoderTextureFormat::BC7_RGBA;

    let params = TranscodeParameters { level_index: 0, ..Default::default() };
    
    // FIX: Pass the raw byte slice of the level data, not the Level struct itself.
    let decoded_bytes = transcoder.transcode_image_level(&level0_data.data, target_format, params)
        .map_err(|e| format!("{:?}", e))?;

    Ok(Texture {
        dimensions: (header.pixel_width, header.pixel_height),
        data: decoded_bytes,
        format: wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    })
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
}

/// Parses a .glb file's binary content, now with tangent generation.
fn parse_glb(bytes: &[u8]) -> Result<Mesh, String> {
    let (gltf, buffers, _) = gltf::import_slice(bytes).map_err(|e| e.to_string())?;
    let mesh = gltf.meshes().next().ok_or("No meshes in glTF")?;
    let primitive = mesh.primitives().next().ok_or("No primitives in mesh")?;
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader.read_positions().ok_or("Primitive has no positions")?.collect();
    let vertex_count = positions.len();

    let normals: Vec<[f32; 3]> = reader.read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count]);
    
    let tex_coords: Vec<[f32; 2]> = reader.read_tex_coords(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count]);

    let indices: Vec<u32> = reader.read_indices()
        .map(|r| r.into_u32().collect())
        .unwrap_or_else(|| (0..vertex_count as u32).collect());

    // First, create vertices with a placeholder tangent.
    let mut vertices: Vec<Vertex> = positions.iter()
        .zip(normals.iter())
        .zip(tex_coords.iter())
        .map(|((&p, &n), &t)| Vertex::new(p, n, t, [0.0; 4]))
        .collect();

    // --- Generate tangents ---
    // The glTF spec says tangents should be generated if they don't exist and normal maps are used.
    // We will generate them here if the model has normals and texture coordinates.
    if primitive.material().normal_texture().is_some() && !indices.is_empty() {
        let mut wrapper = MikkTSpaceWrapper {
            vertices: &mut vertices,
            indices: &indices,
        };
        // This function calculates the tangents and writes them directly into our vertex data.
        if !mikktspace::generate_tangents(&mut wrapper) {
            warn!("failed to generate tangents for {}", mesh.name().unwrap());
        }
    }

    Ok(Mesh { vertices, indices })
}