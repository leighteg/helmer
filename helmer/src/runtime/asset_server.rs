use crate::graphics::renderer::renderer::{Aabb, RenderMessage, Vertex};
use basis_universal::transcoding::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
use glam::Mat4;
use gltf::Texture as GltfTexture;
use hashbrown::HashMap;
use ktx2::Reader;
use meshopt::{SimplifyOptions, VertexDataAdapter, optimize_vertex_cache, simplify};
use mikktspace::Geometry;
use parking_lot::RwLock;
use serde::Deserialize;
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};
use tracing::{info, warn};

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

impl<T> Handle<T> {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            _phantom: PhantomData::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub lod_indices: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct Texture {
    pub dimensions: (u32, u32),
    pub data: Vec<u8>,
    pub format: wgpu::TextureFormat,
}

#[derive(Debug, Clone)]
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

/// Represents a single drawable element in a scene graph, corresponding to a glTF primitive.
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub mesh: Handle<Mesh>,
    pub material: Handle<Material>,
    pub transform: glam::Mat4,
}

/// The final representation of a scene, stored in the AssetServer.
#[derive(Debug, Clone)]
pub struct Scene {
    pub nodes: Vec<SceneNode>,
}

/// Temporary data returned by a worker thread after parsing a glTF file.
/// All indices here are local to this structure's vectors.
#[derive(Debug)]
struct ParsedGltfScene {
    nodes: Vec<GltfNode>,
    meshes: Vec<(Vec<Vertex>, Vec<Vec<u32>>, Aabb)>,
    materials: Vec<IntermediateMaterial>,
    textures: Vec<(String, AssetKind, Vec<u8>, wgpu::TextureFormat, (u32, u32))>,
}

#[derive(Debug)]
struct GltfNode {
    mesh_index: usize,
    material_index: usize,
    transform: glam::Mat4,
}

/// An intermediate material representation that uses texture indices
/// instead of string paths, suitable for parsed scene data.
#[derive(Debug)]
struct IntermediateMaterial {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,
    emission_color: [f32; 3],
    albedo_texture_index: Option<usize>,
    normal_texture_index: Option<usize>,
    metallic_roughness_texture_index: Option<usize>,
    emission_texture_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct MaterialGpuData {
    pub id: usize,
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub emission_strength: f32,
    pub emission_color: [f32; 3],
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

// --- WORKER THREAD COMMUNICATION ---

enum AssetLoadRequest {
    Mesh {
        id: usize,
        path: PathBuf,
    },
    ProceduralMesh {
        id: usize,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
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
    Scene {
        id: usize,
        path: PathBuf,
    },
}

enum AssetLoadResult {
    Mesh {
        id: usize,
        vertices: Vec<Vertex>,
        lod_indices: Vec<Vec<u32>>,
        bounds: Aabb,
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
    Scene {
        id: usize,
        data: ParsedGltfScene,
    },
}

#[derive(Default)]
pub struct MeshAabbMap(pub HashMap<usize, Aabb>);

// --- ASSET SERVER ---

pub struct AssetServer {
    scenes: Arc<RwLock<HashMap<usize, Arc<Scene>>>>,
    pub mesh_aabb_map: Arc<RwLock<MeshAabbMap>>,
    next_id: AtomicUsize,
    request_sender: crossbeam_channel::Sender<AssetLoadRequest>,
    result_receiver: crossbeam_channel::Receiver<AssetLoadResult>,
    render_sender: mpsc::Sender<RenderMessage>,
    _worker_handles: Vec<JoinHandle<()>>,
}

impl AssetServer {
    pub fn new(render_sender: mpsc::Sender<RenderMessage>) -> Self {
        let (request_sender, request_receiver) = crossbeam_channel::unbounded();
        let (result_sender, result_receiver) = crossbeam_channel::unbounded();
        let mut worker_handles = Vec::new();

        let num_workers = num_cpus::get().min(4).max(1);
        for _ in 0..num_workers {
            let req_receiver = request_receiver.clone();
            let res_sender = result_sender.clone();
            let handle = thread::spawn(move || {
                while let Ok(request) = req_receiver.recv() {
                    let path_str = request_path(&request).to_string_lossy().to_string();
                    let result = match request {
                        AssetLoadRequest::Mesh { id, path } => load_and_parse(
                            id,
                            &path,
                            parse_glb,
                            |id, (vertices, lod_indices, bounds)| AssetLoadResult::Mesh {
                                id,
                                vertices,
                                lod_indices,
                                bounds,
                            },
                        ),
                        AssetLoadRequest::ProceduralMesh {
                            id,
                            vertices,
                            indices,
                        } => {
                            let lod_indices = generate_lods(&vertices, &indices);
                            let bounds = Aabb::calculate(&vertices);

                            Some(AssetLoadResult::Mesh {
                                id,
                                vertices,
                                lod_indices,
                                bounds,
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
                        AssetLoadRequest::Scene { id, path } => {
                            load_and_parse(id, &path, parse_scene_glb, |id, data| {
                                AssetLoadResult::Scene { id, data }
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

        info!(
            "AssetServer initialized with {} worker threads.",
            num_workers
        );

        Self {
            scenes: Arc::new(RwLock::new(HashMap::new())),
            mesh_aabb_map: Arc::new(RwLock::new(MeshAabbMap::default())),
            next_id: AtomicUsize::new(0),
            request_sender,
            result_receiver,
            render_sender,
            _worker_handles: worker_handles,
        }
    }

    pub fn add_mesh(&self, vertices: Vec<Vertex>, indices: Vec<u32>) -> Handle<Mesh> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let request = AssetLoadRequest::ProceduralMesh {
            id,
            vertices,
            indices,
        };

        self.request_sender.send(request).unwrap_or_else(|e| {
            warn!(
                "Failed to send procedural mesh request to worker thread: {}",
                e
            );
        });

        Handle::new(id)
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

    pub fn load_scene<P: AsRef<Path>>(&self, path: P) -> Handle<Scene> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = AssetLoadRequest::Scene {
            id,
            path: path.as_ref().to_path_buf(),
        };
        self.request_sender.send(request).unwrap();
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn get_scene(&self, handle: &Handle<Scene>) -> Option<Arc<Scene>> {
        self.scenes.read().get(&handle.id).cloned()
    }

    pub fn update(&self) {
        while let Ok(result) = self.result_receiver.try_recv() {
            match result {
                AssetLoadResult::Mesh {
                    id,
                    vertices,
                    lod_indices,
                    bounds,
                } => {
                    self.render_sender
                        .send(RenderMessage::CreateMesh {
                            id,
                            vertices,
                            lod_indices,
                            bounds,
                        })
                        .unwrap();
                    self.mesh_aabb_map.write().0.insert(id, bounds);
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
                            height,
                        })
                        .unwrap();
                }
                AssetLoadResult::Material { id, data } => {
                    let material_gpu_data = self.resolve_material_dependencies(id, data);
                    self.render_sender
                        .send(RenderMessage::CreateMaterial(material_gpu_data))
                        .unwrap();
                }
                AssetLoadResult::Scene { id: scene_id, data } => {
                    let mut texture_map = HashMap::new();
                    for (i, (name, kind, tex_data, format, (width, height))) in
                        data.textures.into_iter().enumerate()
                    {
                        let tex_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                        self.render_sender
                            .send(RenderMessage::CreateTexture {
                                id: tex_id,
                                name,
                                kind,
                                data: tex_data,
                                format,
                                width,
                                height,
                            })
                            .unwrap();
                        texture_map.insert(i, tex_id);
                    }

                    let mut material_map = HashMap::new();
                    for (i, mat) in data.materials.into_iter().enumerate() {
                        let mat_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                        let resolve_tex = |local_idx_opt: Option<usize>| {
                            local_idx_opt.and_then(|idx| texture_map.get(&idx).copied())
                        };
                        let gpu_data = MaterialGpuData {
                            id: mat_id,
                            albedo: mat.albedo,
                            metallic: mat.metallic,
                            roughness: mat.roughness,
                            ao: mat.ao,
                            emission_strength: mat.emission_strength,
                            emission_color: mat.emission_color,
                            albedo_texture_id: resolve_tex(mat.albedo_texture_index),
                            normal_texture_id: resolve_tex(mat.normal_texture_index),
                            metallic_roughness_texture_id: resolve_tex(
                                mat.metallic_roughness_texture_index,
                            ),
                            emission_texture_id: resolve_tex(mat.emission_texture_index),
                        };
                        self.render_sender
                            .send(RenderMessage::CreateMaterial(gpu_data))
                            .unwrap();
                        material_map.insert(i, mat_id);
                    }

                    let mut mesh_map = HashMap::new();
                    for (i, (vertices, lod_indices, bounds)) in data.meshes.into_iter().enumerate()
                    {
                        let mesh_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                        self.render_sender
                            .send(RenderMessage::CreateMesh {
                                id: mesh_id,
                                vertices,
                                lod_indices,
                                bounds,
                            })
                            .unwrap();
                        self.mesh_aabb_map.write().0.insert(mesh_id, bounds);
                        mesh_map.insert(i, mesh_id);
                    }

                    let scene_nodes = data
                        .nodes
                        .into_iter()
                        .filter_map(|node| {
                            let mesh_id = mesh_map.get(&node.mesh_index)?;
                            let material_id = material_map.get(&node.material_index)?;
                            Some(SceneNode {
                                mesh: Handle::new(*mesh_id),
                                material: Handle::new(*material_id),
                                transform: node.transform,
                            })
                        })
                        .collect();

                    let scene = Arc::new(Scene { nodes: scene_nodes });
                    self.scenes.write().insert(scene_id, scene);
                    info!("Successfully loaded scene with handle {}", scene_id);
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
        AssetLoadRequest::Scene { path, .. } => path,
        _ => Path::new(""),
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

fn decode_ktx2(bytes: &[u8]) -> Result<(Vec<u8>, wgpu::TextureFormat, (u32, u32)), String> {
    let reader = Reader::new(bytes).map_err(|e| e.to_string())?;
    let header = reader.header();
    let mut dimensions = (header.pixel_width, header.pixel_height);

    // Handle uncompressed formats first
    if let Some(format) = header.format {
        let wgpu_format = match format {
            ktx2::Format::R8G8B8A8_UNORM => wgpu::TextureFormat::Rgba8Unorm,
            ktx2::Format::R8G8B8A8_SRGB => wgpu::TextureFormat::Rgba8UnormSrgb,
            _ => return Err(format!("Unsupported direct KTX2 format: {:?}", format)),
        };
        let level_data = reader.levels().next().ok_or("No image levels found")?;
        let mut data = level_data.data.to_vec();

        if std::env::var("HELMER_PATH") == Ok("forwardTA".to_string()) {
            const TARGET_WIDTH: u32 = 512;
            const TARGET_HEIGHT: u32 = 512;
            if dimensions.0 != TARGET_WIDTH || dimensions.1 != TARGET_HEIGHT {
                // We assume R8G8B8A8 format here, which is 4 bytes per pixel.
                if let Some(image_buffer) = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                    dimensions.0,
                    dimensions.1,
                    data.clone(),
                ) {
                    let resized = image::imageops::resize(
                        &image_buffer,
                        TARGET_WIDTH,
                        TARGET_HEIGHT,
                        image::imageops::FilterType::Lanczos3,
                    );
                    data = resized.into_raw();
                    dimensions = (TARGET_WIDTH, TARGET_HEIGHT);
                } else {
                    warn!(
                        "Failed to create image buffer for resizing KTX2 texture. Data length mismatch."
                    );
                }
            }
        }
        return Ok((data, wgpu_format, dimensions));
    }

    // Handle compressed Basis Universal formats
    let mut transcoder = Transcoder::new();
    if transcoder.prepare_transcoding(bytes).is_err() {
        return Err("Failed to prepare Basis Universal transcoder.".to_string());
    }

    let transcode_params = TranscodeParameters {
        level_index: 0,
        ..Default::default()
    };

    if std::env::var("HELMER_PATH") == Ok("forwardTA".to_string()) {
        // Resize path: transcode to RGBA, resize, then use as uncompressed texture
        const TARGET_WIDTH: u32 = 512;
        const TARGET_HEIGHT: u32 = 512;

        match transcoder.transcode_image_level(
            bytes,
            TranscoderTextureFormat::RGBA32,
            transcode_params,
        ) {
            Ok(transcoded_data) => {
                let image_buffer = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                    dimensions.0,
                    dimensions.1,
                    transcoded_data,
                )
                .ok_or_else(|| {
                    "Failed to create image buffer from transcoded KTX2 data".to_string()
                })?;

                let resized = image::imageops::resize(
                    &image_buffer,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                    image::imageops::FilterType::Lanczos3,
                );
                let final_data = resized.into_raw();
                dimensions = (TARGET_WIDTH, TARGET_HEIGHT);

                // Assuming sRGB for color textures, which is a reasonable default.
                let final_format = wgpu::TextureFormat::Rgba8UnormSrgb;
                Ok((final_data, final_format, dimensions))
            }
            Err(e) => Err(format!(
                "Failed to transcode KTX2 to RGBA for resizing: {:?}",
                e
            )),
        }
    } else {
        // Original path: transcode to a compressed format
        let target_basis_format = TranscoderTextureFormat::BC7_RGBA;
        let target_wgpu_format = wgpu::TextureFormat::Bc7RgbaUnormSrgb;
        match transcoder.transcode_image_level(bytes, target_basis_format, transcode_params) {
            Ok(transcoded_data) => Ok((transcoded_data, target_wgpu_format, dimensions)),
            Err(e) => Err(format!("Failed to transcode KTX2 image level: {:?}", e)),
        }
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

fn generate_lods(vertices: &[Vertex], indices: &[u32]) -> Vec<Vec<u32>> {
    let mut lods = Vec::new();
    if indices.is_empty() || vertices.is_empty() {
        return lods;
    }

    // LOD 0: The original mesh, optimized for the GPU vertex cache.
    let lod0 = optimize_vertex_cache(indices, vertices.len());
    lods.push(lod0);

    // Prepare the vertex data adapter for the simplifier.
    // The adapter tells meshopt how to access the position data from your Vertex struct.
    let vertex_data_adapter = VertexDataAdapter::new(
        bytemuck::cast_slice(vertices), // The raw vertex buffer as a byte slice
        std::mem::size_of::<Vertex>(),  // The stride (size of one vertex)
        0, // The offset of the position attribute (it's the first field)
    )
    .unwrap();

    // Generate lower detail LODs from the original index buffer.
    const LOD_TARGETS: [f32; 3] = [0.7, 0.4, 0.15];
    const SIMPLIFICATION_ERROR_TOLERANCE: f32 = 0.02;

    for &ratio in &LOD_TARGETS {
        let target_index_count = (indices.len() as f32 * ratio) as usize;
        let target_index_count = target_index_count - (target_index_count % 3);

        if target_index_count < indices.len() {
            // Always simplify from the original, high-quality index buffer.
            let simplified_indices = simplify(
                indices,
                &vertex_data_adapter,
                target_index_count,
                SIMPLIFICATION_ERROR_TOLERANCE,
                SimplifyOptions::all(),
                None,
            );

            if !simplified_indices.is_empty() {
                // Optimize the new, smaller index buffer for the vertex cache.
                lods.push(optimize_vertex_cache(&simplified_indices, vertices.len()));
            } else if let Some(prev_lod) = lods.last() {
                // If simplification fails (e.g., target is too low), reuse the previous LOD.
                lods.push(prev_lod.clone());
            }
        }
    }
    lods
}

fn process_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Option<(Vec<Vertex>, Vec<Vec<u32>>, Aabb)> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
    if positions.is_empty() {
        return None;
    }
    let normals = reader
        .read_normals()
        .map(|n| n.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let tex_coords = reader
        .read_tex_coords(0)
        .map(|tc| tc.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let indices: Vec<u32> = reader.read_indices()?.into_u32().collect();
    if indices.is_empty() {
        return None;
    }

    let mut vertices: Vec<Vertex> = positions
        .iter()
        .zip(normals.iter())
        .zip(tex_coords.iter())
        .map(|((&p, &n), &t)| Vertex::new(p, n, t, [0.0; 4]))
        .collect();

    mikktspace::generate_tangents(&mut MikkTSpaceWrapper {
        vertices: &mut vertices,
        indices: &indices,
    });

    let bounds = Aabb::calculate(&vertices);
    let lods = generate_lods(&vertices, &indices);
    Some((vertices, lods, bounds))
}

fn parse_glb(bytes: &[u8]) -> Result<(Vec<Vertex>, Vec<Vec<u32>>, Aabb), String> {
    let (gltf, buffers, _) = gltf::import_slice(bytes).map_err(|e| e.to_string())?;
    let mut all_vertices: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            if let Some((mut vertices, lods, _)) = process_primitive(&primitive, &buffers) {
                let index_offset = all_vertices.len() as u32;
                all_vertices.append(&mut vertices);
                // We only care about the base LOD for combining meshes
                if let Some(base_lod_indices) = lods.get(0) {
                    all_indices.extend(base_lod_indices.iter().map(|i| i + index_offset));
                }
            }
        }
    }

    if all_vertices.is_empty() {
        return Err("GLTF contains no valid mesh primitives.".to_string());
    }
    let bounds = Aabb::calculate(&all_vertices);
    let final_lods = generate_lods(&all_vertices, &all_indices);
    Ok((all_vertices, final_lods, bounds))
}

fn parse_scene_glb(bytes: &[u8]) -> Result<ParsedGltfScene, String> {
    let (doc, buffers, images) = gltf::import_slice(bytes).map_err(|e| e.to_string())?;
    let mut out_meshes = Vec::new();
    let mut out_materials = Vec::new();
    let mut out_textures = Vec::new();
    let mut out_nodes = Vec::new();
    let mut material_map: HashMap<Option<usize>, usize> = HashMap::new();
    let mut texture_map: HashMap<usize, usize> = HashMap::new();

    out_textures.push((
        "DEFAULT_ERROR".to_string(),
        AssetKind::Albedo,
        vec![255, 0, 255, 255],
        wgpu::TextureFormat::Rgba8UnormSrgb,
        (1, 1),
    ));
    out_materials.push(IntermediateMaterial {
        albedo: [1.0; 4],
        metallic: 0.5,
        roughness: 0.5,
        ao: 1.0,
        emission_strength: 0.0,
        emission_color: [0.0; 3],
        albedo_texture_index: None,
        normal_texture_index: None,
        metallic_roughness_texture_index: None,
        emission_texture_index: None,
    });
    material_map.insert(None, 0);

    for scene in doc.scenes() {
        for node in scene.nodes() {
            process_node(
                &node,
                &Mat4::IDENTITY,
                &buffers,
                &images,
                &mut out_meshes,
                &mut out_materials,
                &mut out_textures,
                &mut out_nodes,
                &mut material_map,
                &mut texture_map,
            );
        }
    }
    Ok(ParsedGltfScene {
        nodes: out_nodes,
        meshes: out_meshes,
        materials: out_materials,
        textures: out_textures,
    })
}

fn process_node(
    node: &gltf::Node,
    parent_transform: &Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    out_meshes: &mut Vec<(Vec<Vertex>, Vec<Vec<u32>>, Aabb)>,
    out_materials: &mut Vec<IntermediateMaterial>,
    out_textures: &mut Vec<(String, AssetKind, Vec<u8>, wgpu::TextureFormat, (u32, u32))>,
    out_nodes: &mut Vec<GltfNode>,
    material_map: &mut HashMap<Option<usize>, usize>,
    texture_map: &mut HashMap<usize, usize>,
) {
    let transform = *parent_transform * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if let Some((vertices, lods, bounds)) = process_primitive(&primitive, buffers) {
                let mesh_index = out_meshes.len();
                out_meshes.push((vertices, lods, bounds));
                let material = primitive.material();
                let material_index = *material_map.entry(material.index()).or_insert_with(|| {
                    let my_mat_idx = out_materials.len();
                    let pbr = material.pbr_metallic_roughness();
                    let mut get_tex = |tex_opt: Option<GltfTexture>, kind| {
                        tex_opt.map(|t| {
                            *texture_map
                                .entry(t.index())
                                .or_insert_with(|| process_texture(t, buffers, out_textures, kind))
                        })
                    };
                    out_materials.push(IntermediateMaterial {
                        albedo: pbr.base_color_factor(),
                        metallic: pbr.metallic_factor(),
                        roughness: pbr.roughness_factor(),
                        ao: material.occlusion_texture().map_or(1.0, |t| t.strength()),
                        emission_strength: material.emissive_strength().unwrap_or(0.0),
                        emission_color: material.emissive_factor(),
                        albedo_texture_index: get_tex(
                            pbr.base_color_texture().map(|i| i.texture()),
                            AssetKind::Albedo,
                        ),
                        normal_texture_index: get_tex(
                            material.normal_texture().map(|i| i.texture()),
                            AssetKind::Normal,
                        ),
                        metallic_roughness_texture_index: get_tex(
                            pbr.metallic_roughness_texture().map(|i| i.texture()),
                            AssetKind::MetallicRoughness,
                        ),
                        emission_texture_index: get_tex(
                            material.emissive_texture().map(|i| i.texture()),
                            AssetKind::Emission,
                        ),
                    });
                    my_mat_idx
                });
                out_nodes.push(GltfNode {
                    mesh_index,
                    material_index,
                    transform,
                });
            }
        }
    }
    for child in node.children() {
        process_node(
            &child,
            &transform,
            buffers,
            images,
            out_meshes,
            out_materials,
            out_textures,
            out_nodes,
            material_map,
            texture_map,
        );
    }
}

fn process_texture(
    gltf_texture: gltf::Texture,
    buffers: &[gltf::buffer::Data],
    out_textures: &mut Vec<(String, AssetKind, Vec<u8>, wgpu::TextureFormat, (u32, u32))>,
    kind: AssetKind,
) -> usize {
    let image = gltf_texture.source();
    let name = gltf_texture
        .name()
        .unwrap_or("unnamed_gltf_texture")
        .to_string();

    let (pixels, mime_type) = match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            let parent_buffer_data = &buffers[view.buffer().index()];
            let pixel_data = &parent_buffer_data[view.offset()..view.offset() + view.length()];
            (pixel_data, mime_type)
        }
        gltf::image::Source::Uri { .. } => {
            warn!(
                "Skipping external image URI in glTF ('{}'), not yet supported.",
                name
            );
            return 0;
        }
    };

    let decoding_result: Result<(Vec<u8>, wgpu::TextureFormat, (u32, u32)), String> =
        match mime_type {
            "image/ktx2" => decode_ktx2(pixels),
            "image/png" | "image/jpeg" => match image::load_from_memory(pixels) {
                Ok(decoded) => {
                    let mut rgba = decoded.to_rgba8();
                    let mut dimensions = rgba.dimensions();

                    if std::env::var("HELMER_PATH") == Ok("forwardTA".to_string()) {
                        const TARGET_WIDTH: u32 = 512;
                        const TARGET_HEIGHT: u32 = 512;
                        if dimensions.0 != TARGET_WIDTH || dimensions.1 != TARGET_HEIGHT {
                            rgba = image::imageops::resize(
                                &rgba,
                                TARGET_WIDTH,
                                TARGET_HEIGHT,
                                image::imageops::FilterType::Lanczos3,
                            );
                            dimensions = (TARGET_WIDTH, TARGET_HEIGHT);
                        }
                    }

                    let data = rgba.into_raw();
                    let wgpu_format = match kind {
                        AssetKind::Albedo | AssetKind::Emission => {
                            wgpu::TextureFormat::Rgba8UnormSrgb
                        }
                        _ => wgpu::TextureFormat::Rgba8Unorm,
                    };
                    Ok((data, wgpu_format, dimensions))
                }
                Err(e) => Err(e.to_string()),
            },
            unsupported_mime => Err(format!(
                "Unsupported image MIME type in glTF: {}",
                unsupported_mime
            )),
        };

    match decoding_result {
        Ok((data, format, dimensions)) => {
            let my_idx = out_textures.len();
            out_textures.push((name, kind, data, format, dimensions));
            my_idx
        }
        Err(e) => {
            warn!(
                "Failed to process texture '{}': {}. Using fallback.",
                name, e
            );
            0
        }
    }
}
